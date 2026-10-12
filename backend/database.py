"""TALUS persistence layer.

The database is intentionally data-driven: mine sectors and historical observations
are derived from the processed telemetry dataset rather than being hard-coded.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.getenv("TALUS_DB_PATH", os.path.join(ROOT_DIR, "database", "talus.db"))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "cleaned_features.csv")
DB_SCHEMA_VERSION = 2


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_processed_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _schema_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM app_meta WHERE key='schema_version'").fetchone()
    return int(row[0]) if row else 0


def _set_schema_version(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT OR REPLACE INTO app_meta(key,value) VALUES('schema_version',?)", (str(DB_SCHEMA_VERSION),))


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mine_sectors (
            sector_id TEXT PRIMARY KEY,
            sector_name TEXT NOT NULL,
            pit_zone TEXT NOT NULL,
            slope_height_m REAL,
            slope_angle_deg REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sector_id TEXT NOT NULL,
            slope_height_m REAL,
            slope_angle_deg REAL,
            cohesion_kpa REAL,
            friction_angle_deg REAL,
            unit_weight_kn_m3 REAL,
            rainfall_mm REAL,
            groundwater_level REAL,
            pore_pressure_kpa REAL,
            deformation_rate_mm_day REAL,
            displacement_mm REAL,
            recent_blasting INTEGER,
            blast_intensity REAL,
            equipment_load REAL,
            factor_of_safety REAL,
            physics_category TEXT,
            ml_risk_label TEXT,
            ml_risk_probability REAL,
            top_contributors_json TEXT,
            scenario_name TEXT DEFAULT 'Historical Telemetry'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_sector_time ON prediction_history(sector_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_risk ON prediction_history(ml_risk_label)")
    conn.commit()


def _reset_data_tables(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM prediction_history")
    conn.execute("DELETE FROM mine_sectors")
    conn.commit()


def _seed_from_processed_data(conn: sqlite3.Connection) -> None:
    df = _load_processed_data()
    if df.empty or "sector_id" not in df.columns:
        return

    df = df.sort_values("timestamp")
    sectors = sorted(df["sector_id"].dropna().astype(str).unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

    sector_rows = []
    for sid in sectors:
        latest = df[df["sector_id"].astype(str) == sid].iloc[-1]
        sector_rows.append((
            sid,
            f"TALUS Monitoring Sector {sid}",
            "Telemetry Grid",
            float(latest.get("slope_height_m", 0)),
            float(latest.get("slope_angle_deg", 0)),
        ))
    conn.executemany(
        "INSERT INTO mine_sectors(sector_id,sector_name,pit_zone,slope_height_m,slope_angle_deg) VALUES(?,?,?,?,?)",
        sector_rows,
    )

    # Store a representative historical window per sector. This keeps the demo DB
    # useful while avoiding fabricated observations and hard-coded sector profiles.
    records: List[tuple] = []
    for sid in sectors:
        sdf = df[df["sector_id"].astype(str) == sid].tail(60)
        for _, r in sdf.iterrows():
            contributors = [
                {"feature": "factor_of_safety", "value": float(r.get("factor_of_safety", 0)), "impact_score": 0.0, "direction": "physics_indicator"},
                {"feature": "deformation_rate_mm_day", "value": float(r.get("deformation_rate_mm_day", 0)), "impact_score": 0.0, "direction": "telemetry"},
                {"feature": "pore_pressure_kpa", "value": float(r.get("pore_pressure_kpa", 0)), "impact_score": 0.0, "direction": "telemetry"},
            ]
            label = str(r.get("risk_label", "Low"))
            prob_map = {"Low": 0.15, "Moderate": 0.40, "High": 0.72, "Critical": 0.94}
            records.append((
                str(r.get("timestamp")), sid, float(r.get("slope_height_m", 0)), float(r.get("slope_angle_deg", 0)),
                float(r.get("cohesion_kpa", 0)), float(r.get("friction_angle_deg", 0)), float(r.get("unit_weight_kn_m3", 0)),
                float(r.get("rainfall_mm", 0)), float(r.get("groundwater_level", 0)), float(r.get("pore_pressure_kpa", 0)),
                float(r.get("deformation_rate_mm_day", 0)), float(r.get("displacement_mm", 0)), int(r.get("recent_blasting", 0)),
                float(r.get("blast_intensity", 0)), float(r.get("equipment_load", 0)), float(r.get("factor_of_safety", 0)),
                label if label in {"Low", "Moderate", "High", "Critical"} else "Low",
                label, prob_map.get(label, 0.15), json.dumps(contributors), "Historical Telemetry",
            ))

    conn.executemany("""
        INSERT INTO prediction_history(
            timestamp,sector_id,slope_height_m,slope_angle_deg,cohesion_kpa,friction_angle_deg,
            unit_weight_kn_m3,rainfall_mm,groundwater_level,pore_pressure_kpa,deformation_rate_mm_day,
            displacement_mm,recent_blasting,blast_intensity,equipment_load,factor_of_safety,
            physics_category,ml_risk_label,ml_risk_probability,top_contributors_json,scenario_name
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, records)
    conn.commit()


def init_db() -> None:
    conn = get_db_connection()
    _create_tables(conn)
    if _schema_version(conn) != DB_SCHEMA_VERSION or _table_count(conn, "mine_sectors") == 0 or _table_count(conn, "prediction_history") == 0:
        _reset_data_tables(conn)
        _seed_from_processed_data(conn)
        _set_schema_version(conn)
        conn.commit()
    conn.close()


def save_prediction(rec: Dict[str, Any]) -> int:
    init_db()
    conn = get_db_connection()
    contributors = json.dumps(rec.get("top_contributors", []))
    cur = conn.execute("""
        INSERT INTO prediction_history(
            timestamp,sector_id,slope_height_m,slope_angle_deg,cohesion_kpa,friction_angle_deg,
            unit_weight_kn_m3,rainfall_mm,groundwater_level,pore_pressure_kpa,deformation_rate_mm_day,
            displacement_mm,recent_blasting,blast_intensity,equipment_load,factor_of_safety,
            physics_category,ml_risk_label,ml_risk_probability,top_contributors_json,scenario_name
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        rec.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")), rec.get("sector_id"),
        rec.get("slope_height_m"), rec.get("slope_angle_deg"), rec.get("cohesion_kpa"), rec.get("friction_angle_deg"),
        rec.get("unit_weight_kn_m3"), rec.get("rainfall_mm"), rec.get("groundwater_level"), rec.get("pore_pressure_kpa"),
        rec.get("deformation_rate_mm_day"), rec.get("displacement_mm"), rec.get("recent_blasting", 0),
        rec.get("blast_intensity", 0), rec.get("equipment_load", 0), rec.get("factor_of_safety"),
        rec.get("physics_category"), rec.get("ml_risk_label"), rec.get("ml_risk_probability"), contributors,
        rec.get("scenario_name", "User Prediction"),
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return int(row_id)


def _decode(row: sqlite3.Row) -> Dict[str, Any]:
    item = dict(row)
    try:
        item["top_contributors"] = json.loads(item.pop("top_contributors_json") or "[]")
    except Exception:
        item["top_contributors"] = []
        item.pop("top_contributors_json", None)
    return item


def get_latest_mine_overview() -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT h.*, s.sector_name, s.pit_zone
        FROM prediction_history h
        JOIN (SELECT sector_id, MAX(id) max_id FROM prediction_history GROUP BY sector_id) latest
          ON h.id = latest.max_id
        JOIN mine_sectors s ON h.sector_id = s.sector_id
        ORDER BY CASE WHEN h.ml_risk_label='Critical' THEN 1 WHEN h.ml_risk_label='High' THEN 2 WHEN h.ml_risk_label='Moderate' THEN 3 ELSE 4 END,
                 h.sector_id
    """).fetchall()
    conn.close()
    return [_decode(r) for r in rows]


def get_sector_history(sector_id: str, limit: int = 60) -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM prediction_history WHERE sector_id=? ORDER BY timestamp ASC LIMIT ?", (sector_id.upper(), limit)).fetchall()
    conn.close()
    return [_decode(r) for r in rows]


def get_data_summary() -> Dict[str, Any]:
    df = _load_processed_data()
    if df.empty:
        return {"records": 0, "sectors": [], "date_min": None, "date_max": None}
    return {
        "records": int(len(df)),
        "sectors": sorted(df["sector_id"].dropna().astype(str).unique().tolist(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x),
        "date_min": str(df["timestamp"].min()),
        "date_max": str(df["timestamp"].max()),
        "features": [c for c in df.columns if c not in {"record_id", "sector_id", "timestamp", "risk_class", "risk_label", "dataset_type"}],
        "risk_distribution": df["risk_label"].value_counts().to_dict(),
    }
