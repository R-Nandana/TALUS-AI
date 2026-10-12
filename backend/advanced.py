"""Offline-first auxiliary services for TALUS demo features.

All outputs are prototype decision-support artifacts based on synthetic data.
"""
from __future__ import annotations
import io, json, math, os, sqlite3, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("TALUS_DB_PATH", ROOT / "database" / "talus.db"))
IMAGE_FEATURES = ROOT / "data" / "processed" / "image_features.csv"
IMAGE_ROOT = ROOT / "data" / "raw" / "images"
MATERIAL_MODEL = ROOT / "models" / "material_classifier.joblib"
MATERIAL_FEATURES = ["mean_r", "mean_g", "mean_b", "brightness", "contrast", "saturation", "hue", "entropy", "edge_density", "sharpness", "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]


def connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_alerts():
    with connection() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sector TEXT NOT NULL, level INTEGER NOT NULL,
            reason TEXT NOT NULL, created_at TEXT NOT NULL, acknowledged_by TEXT,
            acknowledged_at TEXT, resolved_at TEXT, status TEXT NOT NULL DEFAULT 'ACTIVE',
            demo INTEGER NOT NULL DEFAULT 0)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status, level)')


def list_alerts(active_only: bool = False):
    init_alerts()
    with connection() as c:
        sql = "SELECT * FROM alerts" + (" WHERE status='ACTIVE'" if active_only else "") + " ORDER BY id DESC LIMIT 300"
        return [dict(r) for r in c.execute(sql).fetchall()]


def create_alert(sector: str, level: int, reason: str, demo: bool = False):
    init_alerts()
    now = datetime.now().isoformat(timespec="seconds")
    # Hysteresis/debounce: do not create a duplicate active alert for same sector+level in 2 min.
    with connection() as c:
        existing = c.execute("SELECT * FROM alerts WHERE sector=? AND level=? AND status='ACTIVE' ORDER BY id DESC LIMIT 1", (sector, level)).fetchone()
        if existing:
            created = datetime.fromisoformat(existing["created_at"])
            if datetime.now() - created < timedelta(minutes=2):
                return dict(existing)
        cur = c.execute("INSERT INTO alerts(sector,level,reason,created_at,demo) VALUES(?,?,?,?,?)", (sector, level, reason, now, int(demo)))
        return dict(c.execute("SELECT * FROM alerts WHERE id=?", (cur.lastrowid,)).fetchone())


def acknowledge_alert(alert_id: int, who: str = "Engineer"):
    init_alerts()
    now = datetime.now().isoformat(timespec="seconds")
    with connection() as c:
        cur = c.execute("UPDATE alerts SET acknowledged_by=?, acknowledged_at=?, status='ACKNOWLEDGED' WHERE id=? AND status='ACTIVE'", (who, now, alert_id))
        row = c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(row) if row else None


def resolve_alert(alert_id: int):
    init_alerts()
    with connection() as c:
        c.execute("UPDATE alerts SET resolved_at=?, status='RESOLVED' WHERE id=?", (datetime.now().isoformat(timespec="seconds"), alert_id))
        row = c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(row) if row else None


def parse_assistant(query: str, sectors: list[dict[str, Any]], alerts: list[dict[str, Any]]):
    q = query.lower().strip()
    if any(w in q for w in ["help", "what can", "commands"]):
        text = "Ask me for overall status, the most critical sector, a sector status, why a sector is risky, active alerts, worker precautions, or a what-if scenario."
        intent = "help"
    elif "alert" in q:
        active = [a for a in alerts if a.get("status") == "ACTIVE"]
        text = f"There are {len(active)} active alerts." + (" Highest level is Level " + str(max(a['level'] for a in active)) + " for " + ", ".join(sorted(set(a['sector'] for a in active))) + "." if active else " No active alerts are recorded.")
        intent = "active_alerts"
    elif any(w in q for w in ["worker", "precaution", "what should", "safety action"]):
        text = "Follow the site-approved response plan. Restrict access to affected areas when instructed, stop nearby work if directed, notify the control room, verify drainage and deformation readings, use required PPE, and defer all operational decisions to the responsible geotechnical engineer."
        intent = "worker_precautions"
    elif any(w in q for w in ["most critical", "highest risk", "worst sector", "critical sector"]):
        ranked = sorted(sectors, key=lambda s: (s.get("ml_risk_label") != "Critical", -float(s.get("ml_risk_probability", 0))))
        s = ranked[0] if ranked else {}
        text = f"The highest-ranked sector in the current prototype snapshot is {s.get('sector_id','unknown')}, labelled {s.get('ml_risk_label','unknown')}, with FoS {float(s.get('factor_of_safety',0)):.2f}. This is synthetic-data decision support, not a field warning."
        intent = "most_critical"
    elif any(w in q for w in ["overall", "mine status", "status of mine", "summary"]):
        high = sum(s.get("ml_risk_label") in ("High", "Critical") for s in sectors)
        critical = sum(s.get("ml_risk_label") == "Critical" for s in sectors)
        text = f"The prototype currently lists {len(sectors)} sectors. {high} are labelled High or Critical, including {critical} Critical. Confirm conditions against approved site instrumentation and engineering procedures."
        intent = "overall_status"
    else:
        import re
        m = re.search(r"\b(s\s*\d+)\b", q, re.I)
        sid = m.group(1).replace(" ", "").upper() if m else None
        if sid and any(s.get("sector_id", "").upper() == sid for s in sectors):
            s = next(s for s in sectors if s.get("sector_id", "").upper() == sid)
            if any(w in q for w in ["why", "cause", "risky", "risk"]):
                contributors = s.get("top_contributors", []) or []
                top = ", ".join(str(c.get("feature")) for c in contributors[:3]) or "available telemetry and physics indicators"
                text = f"{sid} is labelled {s.get('ml_risk_label','unknown')} in the prototype. The stored leading indicators are {top}. Review the original measurements and the model's limitations before acting."
                intent = "sector_explanation"
            else:
                text = f"{sid} is labelled {s.get('ml_risk_label','unknown')}, with factor of safety {float(s.get('factor_of_safety',0)):.2f} and model probability {float(s.get('ml_risk_probability',0))*100:.1f} percent. These are synthetic prototype results."
                intent = "sector_status"
        else:
            text = "I could not map that question to a supported intent. Try: 'Which sector is most critical?', 'What is the status of S4?', 'Read active alerts', or 'What should workers do?'"
            intent = "fallback"
    return {"intent": intent, "text": text, "speak": text, "action": {"type": intent}}


def monte_carlo_pof(payload: dict[str, Any], n: int = 5000):
    from backend.physics import calculate_factor_of_safety
    rng = np.random.default_rng(42)
    samples = []
    # Assumed variation is explicitly illustrative, not estimated field uncertainty.
    for _ in range(n):
        c = max(0, rng.normal(payload["cohesion_kpa"], max(1.0, payload["cohesion_kpa"] * .10)))
        phi = float(np.clip(rng.normal(payload["friction_angle_deg"], 2.0), 1, 75))
        u = max(0, rng.normal(payload["pore_pressure_kpa"], max(1.0, payload["pore_pressure_kpa"] * .15)))
        samples.append(calculate_factor_of_safety(payload["slope_height_m"], payload["slope_angle_deg"], c, phi, payload["unit_weight_kn_m3"], u, payload.get("recent_blasting", 0), payload.get("blast_intensity", 0), payload.get("equipment_load", 0)))
    a = np.asarray(samples)
    pof = float(np.mean(a < 1.0))
    from scipy.stats import norm
    beta = float(-norm.ppf(np.clip(pof, 1e-6, 1 - 1e-6)))
    return {"simulations": n, "probability_of_failure": pof, "reliability_index_beta": beta, "fos_mean": float(a.mean()), "fos_std": float(a.std(ddof=1)), "histogram": {"counts": np.histogram(a, bins=30)[0].tolist(), "edges": np.histogram(a, bins=30)[1].tolist()}, "assumption": "Illustrative Monte Carlo: cohesion ±10%, friction angle SD 2 degrees, pore pressure ±15%; distributions are assumptions, not field-calibrated uncertainty."}


def time_to_failure(sector_id: str):
    from backend.database import get_sector_history
    history = get_sector_history(sector_id, 500)
    if len(history) < 5:
        return {"sector_id": sector_id, "available": False, "message": "At least five historical observations are required."}
    df = pd.DataFrame(history)
    y = pd.to_numeric(df["deformation_rate_mm_day"], errors="coerce").dropna().to_numpy()
    x = np.arange(len(y), dtype=float)
    if len(y) < 5 or np.any(y <= 0):
        return {"sector_id": sector_id, "available": False, "message": "Inverse-velocity estimate requires at least five positive deformation-rate values."}
    inv = 1.0 / y
    slope, intercept = np.polyfit(x, inv, 1)
    if slope >= 0:
        return {"sector_id": sector_id, "available": False, "message": "Inverse velocity is not decreasing in this window; no finite extrapolated window is reported."}
    crossing = -intercept / slope
    steps = crossing - (len(y) - 1)
    return {"sector_id": sector_id, "available": bool(steps > 0), "estimated_observation_steps": float(steps) if steps > 0 else None, "slope": float(slope), "intercept": float(intercept), "method": "Illustrative inverse-velocity linear extrapolation", "message": "This is not a validated time-to-failure forecast. Synthetic history and sparse observations make the estimate indicative only."}


def train_material_classifier():
    if MATERIAL_MODEL.exists():
        try:
            return joblib.load(MATERIAL_MODEL)
        except Exception:
            pass
    df = pd.read_csv(IMAGE_FEATURES)
    features = [c for c in MATERIAL_FEATURES if c in df.columns]
    data = df.dropna(subset=features + ["material"])
    model = RandomForestClassifier(n_estimators=120, class_weight="balanced", random_state=42, n_jobs=1)
    model.fit(data[features], data["material"].astype(str))
    joblib.dump({"model": model, "features": features, "classes": sorted(data["material"].astype(str).unique())}, MATERIAL_MODEL)
    return joblib.load(MATERIAL_MODEL)


def image_features_from_bytes(raw: bytes):
    im = Image.open(io.BytesIO(raw)).convert("RGB").resize((128, 128))
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    from skimage.feature import graycomatrix, graycoprops
    gray8 = np.array(im.convert("L"), dtype=np.uint8, copy=True)
    gray_levels = (gray8 // 32).astype(np.uint8)
    glcm = graycomatrix(gray_levels, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
    gx, gy = np.gradient(gray)
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    hist, _ = np.histogram(gray, bins=256, range=(0, 1), density=False)
    hist = hist.astype(float) / max(1, hist.sum())
    hist = hist[hist > 0]
    return {"mean_r": float(rgb[:,:,0].mean()), "mean_g": float(rgb[:,:,1].mean()), "mean_b": float(rgb[:,:,2].mean()), "brightness": float(gray.mean()), "contrast": float(gray.std()), "saturation": float(hsv[:,:,1].mean()), "hue": float(hsv[:,:,0].mean()), "entropy": float(-(hist * np.log2(hist + 1e-12)).sum()), "edge_density": float(np.mean(np.sqrt(gx*gx+gy*gy) > .08)), "sharpness": float(np.var(np.gradient(gray)[0]) + np.var(np.gradient(gray)[1])), "glcm_contrast": float(graycoprops(glcm, "contrast")[0,0]), "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0,0]), "glcm_energy": float(graycoprops(glcm, "energy")[0,0]), "glcm_correlation": float(graycoprops(glcm, "correlation")[0,0])}


def classify_material(raw: bytes):
    pack = train_material_classifier()
    feat = image_features_from_bytes(raw)
    x = pd.DataFrame([[feat[c] for c in pack["features"]]], columns=pack["features"])
    model = pack["model"]
    probs = model.predict_proba(x)[0]
    label = str(model.classes_[int(np.argmax(probs))])
    material_norm = label.lower().replace(" ", "_")
    # These are illustrative material ranges for UI initialization only; confirm with lab tests.
    ranges = {
        "basalt": {"cohesion_kpa": [100, 400], "friction_angle_deg": [35, 50]},
        "granite": {"cohesion_kpa": [150, 500], "friction_angle_deg": [35, 55]},
        "limestone": {"cohesion_kpa": [80, 350], "friction_angle_deg": [30, 45]},
        "sandstone": {"cohesion_kpa": [30, 200], "friction_angle_deg": [28, 42]},
        "shale": {"cohesion_kpa": [15, 150], "friction_angle_deg": [20, 35]},
        "clay": {"cohesion_kpa": [10, 100], "friction_angle_deg": [10, 28]},
        "sandy_soil": {"cohesion_kpa": [0, 30], "friction_angle_deg": [28, 38]},
        "gravel_colluvium": {"cohesion_kpa": [0, 40], "friction_angle_deg": [30, 42]},
    }
    return {"predicted_material": label, "confidence": float(np.max(probs)), "class_probabilities": {str(k): float(v) for k, v in zip(model.classes_, probs)}, "texture_features": feat, "illustrative_strength_ranges": ranges.get(material_norm, {"cohesion_kpa": [10, 100], "friction_angle_deg": [20, 40]}), "disclaimer": "Image classification is a prototype aid. Image-derived material labels cannot replace geological logging, sampling, or laboratory testing."}
