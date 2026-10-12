"""
TALUS - Raw Data Generator (data-quality scenarios)
===================================================
Produces two RAW datasets that behave like real field exports (i.e. they are messy):

 1. data/raw/slope_monitoring_raw_field.csv
    The clean synthetic slope-monitoring master (slope_monitoring_synthetic.csv) with
    realistic data-quality problems injected: missing readings, duplicated rows,
    sensor glitches, sentinel codes (-999), unit typos, inconsistent text/date formats.
    Because the clean master is known, the preprocessing pipeline can be VALIDATED against it.

 2. data/raw/rock_soil_properties.csv
    Laboratory-style geotechnical properties for 5 rock types and 3 soil types
    (typical/illustrative ranges - synthetic samples, not measurements from a real mine),
    also with injected problems (missing values, g/cm3-vs-kN/m3 unit error, name typos, duplicates).

Usage: python scripts/generate_raw_data.py
"""
import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

# ------------------------------------------------------------------ slope data
def make_messy_slope(clean: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = clean.copy()
    n = len(df)
    df.insert(0, "record_id", [f"R{i + 1:04d}" for i in range(n)])
    pick = lambda frac: rng.choice(n, size=int(round(frac * n)), replace=False)

    # object-typed columns so mixed formats can be stored
    for c in ["sector_id", "timestamp", "recent_blasting", "risk_label"]:
        df[c] = df[c].astype(object)

    # inconsistent text formats
    for i in pick(0.06):
        df.at[i, "sector_id"] = rng.choice([df.at[i, "sector_id"].lower(), f" {df.at[i, 'sector_id']} "])
    for i in pick(0.05):
        df.at[i, "risk_label"] = rng.choice([df.at[i, "risk_label"].upper(), df.at[i, "risk_label"].lower()])
    for i in pick(0.06):
        df.at[i, "recent_blasting"] = rng.choice(["Y", "yes"]) if df.at[i, "recent_blasting"] == 1 else rng.choice(["N", "no"])
    for i in pick(0.08):                                       # DD/MM/YYYY HH:MM
        df.at[i, "timestamp"] = pd.Timestamp(df.at[i, "timestamp"]).strftime("%d/%m/%Y %H:%M")

    # missing sensor readings (random, ~3% per column)
    for col in ["slope_height_m", "slope_angle_deg", "cohesion_kpa", "friction_angle_deg", "unit_weight_kn_m3",
                "rainfall_mm", "groundwater_level", "pore_pressure_kpa", "deformation_rate_mm_day",
                "displacement_mm", "equipment_load"]:
        df.loc[pick(0.03), col] = np.nan
    df.loc[pick(0.015), "factor_of_safety"] = np.nan

    # sentinel code and physically impossible values
    df.loc[pick(0.01), "groundwater_level"] = -999
    idx = pick(0.008)
    df.loc[idx, "rainfall_mm"] = -(df.loc[idx, "rainfall_mm"].fillna(10) + 5)   # negative rain = sensor fault
    idx = pick(0.005)
    df.loc[idx, "slope_angle_deg"] = df.loc[idx, "slope_angle_deg"].fillna(40) * 10  # decimal-point typo
    idx = pick(0.010)
    df.loc[idx, "deformation_rate_mm_day"] = df.loc[idx, "deformation_rate_mm_day"] * 12  # sensor spike

    # duplicated transmissions
    dup = df.sample(45, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

# ------------------------------------------------------------ rock & soil data
# name: (type, unit_wt, strength_MPa, cohesion_kPa, friction_deg, porosity_%, moisture_%, log10_k, RQD, GSI)
# each numeric entry is (mean, sd); values are typical/illustrative ranges for demonstration.
MATERIALS = {
    "Granite":          ("Rock", (26.5, .6), (170, 35), (1000, 250), (48, 4), (1, .5),  (0.4, .2), (-10, .7), (78, 12), (70, 8)),
    "Basalt":           ("Rock", (28.0, .7), (200, 45), (1100, 300), (46, 4), (3, 1.5), (0.8, .4), (-9.5, .8), (72, 14), (65, 9)),
    "Sandstone":        ("Rock", (23.5, 1.2), (70, 25), (600, 200), (38, 4), (14, 5),  (3.5, 1.5), (-8, .8), (62, 15), (55, 10)),
    "Limestone":        ("Rock", (25.0, 1.0), (90, 30), (700, 250), (40, 4), (6, 3),   (1.5, .8), (-8.5, 1), (66, 15), (60, 10)),
    "Shale":            ("Rock", (22.5, 1.3), (30, 12), (300, 120), (28, 4), (12, 5),  (5.0, 2), (-10.5, .8), (42, 16), (40, 10)),
    "Clay":             ("Soil", (18.5, 1.2), (0.15, .06), (25, 10), (18, 4), (45, 6), (28, 7), (-9.5, .8), None, None),
    "Sandy Soil":       ("Soil", (17.5, 1.0), (0.02, .01), (3, 3), (33, 3), (38, 5),   (10, 4), (-4.5, .7), None, None),
    "Gravel Colluvium": ("Soil", (19.0, 1.0), (0.03, .015), (8, 6), (38, 4), (30, 5),  (8, 3), (-3.0, .7), None, None),
}
COLS = ["unit_weight_kn_m3", "compressive_strength_mpa", "cohesion_kpa", "friction_angle_deg",
        "porosity_pct", "moisture_content_pct", "log10_permeability_m_s", "rqd_pct", "gsi"]

def make_rock_soil(seed: int = 11, n_per_material: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, (mtype, *params) in MATERIALS.items():
        for _ in range(n_per_material):
            depth = float(rng.uniform(3, 120))
            if mtype == "Rock":  # deeper rock is less weathered -> stronger
                wg = int(np.clip(round(3.6 - depth / 45 + rng.normal(0, .6)), 1, 5))
            else:
                wg = 5
            factor = 1 - 0.10 * (wg - 2) if mtype == "Rock" else 1.0
            row = {"material": name, "material_type": mtype, "depth_m": round(depth, 1)}
            for col, p in zip(COLS, params):
                if p is None:
                    row[col] = np.nan
                    continue
                v = rng.normal(*p)
                if col in ("compressive_strength_mpa", "cohesion_kpa", "gsi") and mtype == "Rock":
                    v *= factor
                row[col] = v
            row["weathering_grade"] = wg
            rows.append(row)
    df = pd.DataFrame(rows)
    df["unit_weight_kn_m3"] = df.unit_weight_kn_m3.clip(14, 30)
    for c in ["compressive_strength_mpa", "cohesion_kpa", "porosity_pct", "moisture_content_pct"]:
        df[c] = df[c].clip(lower=0.005)
    df["rqd_pct"] = df.rqd_pct.clip(0, 100)
    df["gsi"] = df.gsi.clip(5, 100)
    df["friction_angle_deg"] = df.friction_angle_deg.clip(10, 60)
    df = df.round({"unit_weight_kn_m3": 2, "compressive_strength_mpa": 3, "cohesion_kpa": 1, "friction_angle_deg": 1,
                   "porosity_pct": 1, "moisture_content_pct": 1, "log10_permeability_m_s": 2, "rqd_pct": 0, "gsi": 0})
    df.insert(0, "sample_id", [f"LAB{i + 1:04d}" for i in range(len(df))])

    # ---- inject data-quality issues
    n = len(df)
    pick = lambda frac: rng.choice(n, size=int(round(frac * n)), replace=False)
    for col in ["unit_weight_kn_m3", "compressive_strength_mpa", "cohesion_kpa", "friction_angle_deg",
                "porosity_pct", "moisture_content_pct", "log10_permeability_m_s"]:
        df.loc[pick(0.025), col] = np.nan
    rock = df.index[df.material_type == "Rock"]
    df.loc[rng.choice(rock, 12, replace=False), "rqd_pct"] = np.nan
    df.loc[rng.choice(rock, 10, replace=False), "gsi"] = np.nan
    df.loc[pick(0.02), "unit_weight_kn_m3"] = df.unit_weight_kn_m3 / 9.81      # recorded in g/cm3 by mistake
    df.loc[pick(0.01), "friction_angle_deg"] = 480                             # typo
    df["material"] = df["material"].astype(object)
    for i in pick(0.05):
        m = df.at[i, "material"]
        df.at[i, "material"] = rng.choice([m.lower(), m.upper(), f"{m} "])
    dup = df.sample(8, random_state=seed)
    return pd.concat([df, dup], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    clean = pd.read_csv(os.path.join(RAW_DIR, "slope_monitoring_synthetic.csv"))
    slope = make_messy_slope(clean)
    slope.to_csv(os.path.join(RAW_DIR, "slope_monitoring_raw_field.csv"), index=False)
    rs = make_rock_soil()
    rs.to_csv(os.path.join(RAW_DIR, "rock_soil_properties.csv"), index=False)
    print(f"slope_monitoring_raw_field.csv : {len(slope)} rows ({len(clean)} true + {len(slope) - len(clean)} duplicates)")
    print(f"rock_soil_properties.csv       : {len(rs)} rows")

if __name__ == "__main__":
    main()
