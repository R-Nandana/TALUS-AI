"""
TALUS - Data Preprocessing Pipeline
===================================
Cleans the three raw data sources and writes results to data/processed/.

  1. Slope monitoring (tabular)   raw_field.csv     -> cleaned_features.csv
  2. Rock & soil properties       rock_soil_properties.csv -> rock_soil_clean.csv
  3. Rock & soil images           data/raw/images/*  -> image_features.csv, image_channel_stats.json,
                                                        image_split.csv

Every step counts what it changed and the totals go to data/processed/preprocessing_report.json
(shown on the Streamlit "Data Intelligence" page).

Usage: python scripts/preprocessing.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from backend.physics import calculate_factor_of_safety   # noqa: E402

RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")

# ============================================================ 1. SLOPE MONITORING
# Physical plausibility limits (domain knowledge) - values outside are sensor/typing errors
SLOPE_LIMITS = {
    "slope_height_m": (5, 300), "slope_angle_deg": (5, 85), "cohesion_kpa": (0, 500),
    "friction_angle_deg": (5, 60), "unit_weight_kn_m3": (15, 32), "rainfall_mm": (0, 400),
    "groundwater_level": (0, 100), "pore_pressure_kpa": (0, 400), "deformation_rate_mm_day": (0, 50),
    "displacement_mm": (0, 1000), "blast_intensity": (0, 20), "equipment_load": (0, 200),
    "factor_of_safety": (0, 5),
}
RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}
BOOL_MAP = {"1": 1, "0": 0, "1.0": 1, "0.0": 0, "y": 1, "n": 0, "yes": 1, "no": 0, "true": 1, "false": 0}


def preprocess_slope(raw: pd.DataFrame):
    """Returns (cleaned_df, report_dict, audit_df). audit_df logs every cell that was changed."""
    df = raw.copy()
    rep = {"rows_raw": int(len(df))}
    audit = []   # (record_id, column, issue, old, new)

    def log(mask, col, issue, new=None):
        for rid, old in zip(df.loc[mask, "record_id"], df.loc[mask, col]):
            audit.append((rid, col, issue, old, new))

    # -- 1. text standardisation
    before = df["sector_id"].astype(str)
    df["sector_id"] = before.str.strip().str.upper()
    rep["sector_id_fixed"] = int((before != df["sector_id"]).sum())

    # -- 2. timestamps: two known formats -> one ISO format
    s = df["timestamp"].astype(str).str.strip()
    ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    alt = pd.to_datetime(s[ts.isna()], format="%d/%m/%Y %H:%M", errors="coerce")
    ts = ts.fillna(alt)
    rep["timestamps_reformatted"] = int(alt.notna().sum())
    rep["timestamps_unparseable"] = int(ts.isna().sum())
    df["timestamp"] = ts

    # -- 3. boolean flag -> 0/1
    b = df["recent_blasting"].astype(str).str.strip().str.lower()
    rep["blasting_flag_fixed"] = int((~b.isin(["0", "1"])).sum())
    df["recent_blasting"] = b.map(BOOL_MAP)

    # -- 4. numeric coercion (any text like 'N/A' becomes NaN)
    num_cols = list(SLOPE_LIMITS.keys())
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -- 5. duplicates (same record transmitted twice)
    n0 = len(df)
    df = df.drop_duplicates(subset="record_id", keep="first").reset_index(drop=True)
    rep["duplicates_removed"] = int(n0 - len(df))

    # -- 6. domain-rule validation: impossible values & sentinel codes -> NaN
    rep["missing_before_cleaning"] = int(df[num_cols].isna().sum().sum())
    rep["invalid_values_by_column"] = {}
    for c, (lo, hi) in SLOPE_LIMITS.items():
        bad = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
        if bad.any():
            rep["invalid_values_by_column"][c] = int(bad.sum())
            log(bad, c, f"outside physical range [{lo}, {hi}]")
            df.loc[bad, c] = np.nan
    rep["sentinel_999_found"] = int((pd.to_numeric(raw["groundwater_level"], errors="coerce") == -999).sum())

    # -- 7. kinematic ratio (displacement / rate, in days) - used ONLY for imputation, not for deleting data.
    #       (A ratio-based spike detector was tested and rejected: it flagged legitimate rows, so it is not used.)
    ratio = df["displacement_mm"] / df["deformation_rate_mm_day"]
    med_ratio = float(ratio[ratio > 0].median())

    # -- 8. label consistency: risk_label re-derived from risk_class (the numeric target is authoritative)
    fixed_labels = df["risk_label"].astype(str).str.strip().str.capitalize()
    rep["risk_label_case_fixed"] = int((df["risk_label"].astype(str) != df["risk_label"].astype(str).str.capitalize()).sum())
    df["risk_label"] = df["risk_class"].map(RISK_LABELS)
    rep["risk_label_mismatch_after_fix"] = int((fixed_labels != df["risk_label"]).sum())

    # -- 9. imputation (order matters: independent variables first)
    rep["missing_to_impute"] = int(df[num_cols].isna().sum().sum())
    rep["imputation"] = {}
    # 9a. pore pressure ~ groundwater head + rainfall infiltration (fit on complete rows)
    comp = df[["pore_pressure_kpa", "groundwater_level", "rainfall_mm"]].dropna()
    X = np.c_[np.ones(len(comp)), comp.groundwater_level, comp.rainfall_mm]
    coef, *_ = np.linalg.lstsq(X, comp.pore_pressure_kpa, rcond=None)
    rep["pore_pressure_model"] = {"intercept": round(float(coef[0]), 3), "per_m_groundwater": round(float(coef[1]), 3),
                                  "per_mm_rain": round(float(coef[2]), 3)}
    for c in ["groundwater_level", "rainfall_mm"]:                 # medians first (needed by 9a)
        n_miss = int(df[c].isna().sum()); df[c] = df[c].fillna(df[c].median())
        rep["imputation"][c] = f"median ({n_miss})"
    m = df["pore_pressure_kpa"].isna()
    est = coef[0] + coef[1] * df.groundwater_level + coef[2] * df.rainfall_mm
    df.loc[m, "pore_pressure_kpa"] = est[m].clip(0, 300)
    rep["imputation"]["pore_pressure_kpa"] = f"regression on groundwater+rainfall ({int(m.sum())})"
    # 9b. deformation from displacement (kinematic ratio), displacement from deformation
    m = df["deformation_rate_mm_day"].isna() & df["displacement_mm"].notna()
    df.loc[m, "deformation_rate_mm_day"] = (df.loc[m, "displacement_mm"] / med_ratio).clip(0.1, 50)
    rep["imputation"]["deformation_rate_mm_day"] = f"displacement / median ratio {med_ratio:.1f} d ({int(m.sum())})"
    m = df["displacement_mm"].isna() & df["deformation_rate_mm_day"].notna()
    df.loc[m, "displacement_mm"] = df.loc[m, "deformation_rate_mm_day"] * med_ratio
    # 9c. everything else -> median (EDA showed no sector-level structure in these variables)
    for c in ["slope_height_m", "slope_angle_deg", "cohesion_kpa", "friction_angle_deg", "unit_weight_kn_m3",
              "equipment_load", "deformation_rate_mm_day", "displacement_mm"]:
        n_miss = int(df[c].isna().sum())
        if n_miss:
            df[c] = df[c].fillna(df[c].median())
            rep["imputation"].setdefault(c, f"median ({n_miss})")
    # 9d. blast intensity must be 0 when no blasting happened
    df.loc[df.recent_blasting == 0, "blast_intensity"] = 0.0
    df["blast_intensity"] = df["blast_intensity"].fillna(0.0)
    # 9e. missing Factor of Safety -> recompute with the project's physics engine
    m = df["factor_of_safety"].isna()
    df.loc[m, "factor_of_safety"] = df.loc[m].apply(lambda r: calculate_factor_of_safety(
        r.slope_height_m, r.slope_angle_deg, r.cohesion_kpa, r.friction_angle_deg, r.unit_weight_kn_m3,
        r.pore_pressure_kpa, int(r.recent_blasting), r.blast_intensity, r.equipment_load), axis=1)
    rep["imputation"]["factor_of_safety"] = f"recomputed by Limit-Equilibrium physics ({int(m.sum())})"
    rep["missing_after_cleaning"] = int(df[num_cols].isna().sum().sum())

    # -- 10. statistical outliers: FLAGGED (3xIQR fences) but RETAINED - extreme readings are exactly what
    #        a slope-risk system must learn from; deleting them would bias the model towards "safe".
    rep["iqr_outliers_flagged_retained"] = {}
    for c in ["rainfall_mm", "deformation_rate_mm_day", "displacement_mm", "pore_pressure_kpa", "factor_of_safety"]:
        q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
        rep["iqr_outliers_flagged_retained"][c] = int(((df[c] < q1 - 3 * iqr) | (df[c] > q3 + 3 * iqr)).sum())

    # -- 11. final types / order
    df["recent_blasting"] = df["recent_blasting"].astype(int)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in num_cols:
        df[c] = df[c].round(3)
    df = df.sort_values(["timestamp", "sector_id"]).reset_index(drop=True)
    rep["rows_clean"] = int(len(df))
    audit_df = pd.DataFrame(audit, columns=["record_id", "column", "issue", "old_value", "new_value"])
    return df, rep, audit_df


# ============================================================ 2. ROCK & SOIL PROPERTIES
RS_LIMITS = {"unit_weight_kn_m3": (14, 32), "compressive_strength_mpa": (0.001, 400), "cohesion_kpa": (0, 5000),
             "friction_angle_deg": (5, 65), "porosity_pct": (0, 70), "moisture_content_pct": (0, 60),
             "log10_permeability_m_s": (-14, 0), "rqd_pct": (0, 100), "gsi": (0, 100)}
CANON = {"granite": "Granite", "basalt": "Basalt", "sandstone": "Sandstone", "limestone": "Limestone",
         "shale": "Shale", "clay": "Clay", "sandysoil": "Sandy Soil", "gravelcolluvium": "Gravel Colluvium"}


def preprocess_rock_soil(raw: pd.DataFrame):
    df = raw.copy()
    rep = {"rows_raw": int(len(df))}
    key = df["material"].astype(str).str.lower().str.replace(r"\s+", "", regex=True)
    rep["material_names_fixed"] = int((df["material"].astype(str) != key.map(CANON)).sum())
    df["material"] = key.map(CANON)
    n0 = len(df)
    df = df.drop_duplicates(subset="sample_id").reset_index(drop=True)
    rep["duplicates_removed"] = int(n0 - len(df))

    # unit weight recorded in g/cm3 (values ~2-3) -> kN/m3
    gcc = df["unit_weight_kn_m3"] < 5
    rep["unit_weight_gcm3_converted"] = int(gcc.sum())
    df.loc[gcc, "unit_weight_kn_m3"] *= 9.81

    rep["invalid_values_by_column"] = {}
    for c, (lo, hi) in RS_LIMITS.items():
        bad = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
        if bad.any():
            rep["invalid_values_by_column"][c] = int(bad.sum())
            df.loc[bad, c] = np.nan

    # legit missing: RQD & GSI do not apply to soils (they are NOT imputed)
    soil = df["material_type"] == "Soil"
    applicable = [c for c in RS_LIMITS if c not in ("rqd_pct", "gsi")]
    rep["missing_to_impute"] = int(df[applicable].isna().sum().sum() + df.loc[~soil, ["rqd_pct", "gsi"]].isna().sum().sum())
    for c in applicable + ["rqd_pct", "gsi"]:
        grp_med = df.groupby("material")[c].transform("median")
        fill_mask = df[c].isna() & (~soil if c in ("rqd_pct", "gsi") else True)
        df.loc[fill_mask, c] = grp_med[fill_mask]          # median of the SAME material
    rep["missing_after_cleaning"] = int(df[applicable].isna().sum().sum() + df.loc[~soil, ["rqd_pct", "gsi"]].isna().sum().sum())
    rep["legit_missing_soil_rqd_gsi"] = int(df.loc[soil, ["rqd_pct", "gsi"]].isna().sum().sum())
    df = df.round(3).sort_values(["material_type", "material", "depth_m"]).reset_index(drop=True)
    rep["rows_clean"] = int(len(df))
    return df, rep


# ============================================================ 3. IMAGES
def _ahash(gray, size=8):
    from PIL import Image
    small = np.asarray(Image.fromarray(gray).resize((size, size)), float)
    return "".join("1" if v else "0" for v in (small > small.mean()).ravel())


def image_features(path):
    """Loads one image and returns a dict of colour + texture features (or None if unreadable)."""
    from PIL import Image
    from scipy import ndimage as ndi
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import shannon_entropy
    try:
        with Image.open(path) as im:
            im.load()
            orig_size = im.size
            im = im.convert("RGB").resize((128, 128))
    except Exception:
        return None
    rgb = np.asarray(im, float) / 255.0
    gray_u8 = np.asarray(im.convert("L"))
    hsv = np.asarray(im.convert("HSV"), float) / 255.0
    q = (gray_u8 // 8).astype(np.uint8)   # 32 grey levels for GLCM
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    gx, gy = ndi.sobel(gray_u8 / 255.0, 0), ndi.sobel(gray_u8 / 255.0, 1)
    edge = np.hypot(gx, gy)
    return {
        "orig_width": orig_size[0], "orig_height": orig_size[1],
        "mean_r": rgb[..., 0].mean(), "mean_g": rgb[..., 1].mean(), "mean_b": rgb[..., 2].mean(),
        "brightness": gray_u8.mean() / 255.0, "contrast": gray_u8.std() / 255.0,
        "saturation": hsv[..., 1].mean(), "hue": hsv[..., 0].mean(),
        "entropy": shannon_entropy(gray_u8),
        "edge_density": float((edge > 0.35).mean()),
        "sharpness": float(ndi.laplace(gray_u8 / 255.0).var()),
        "glcm_contrast": graycoprops(glcm, "contrast").mean(),
        "glcm_homogeneity": graycoprops(glcm, "homogeneity").mean(),
        "glcm_energy": graycoprops(glcm, "energy").mean(),
        "glcm_correlation": graycoprops(glcm, "correlation").mean(),
        "_hash": _ahash(gray_u8),
        "_rgb_sum": rgb.sum(axis=(0, 1)), "_rgb_sqsum": (rgb ** 2).sum(axis=(0, 1)),
    }


def preprocess_images():
    """Scans data/raw/images/<class>/* (folder = class label). Returns (features_df, report, channel_stats)."""
    root = os.path.join(RAW_DIR, "images")
    meta_path = os.path.join(RAW_DIR, "image_metadata.csv")
    meta = pd.read_csv(meta_path).set_index("filepath") if os.path.exists(meta_path) else None
    rows, rep = [], {"files_found": 0, "unreadable_removed": 0, "duplicates_removed": 0, "quality_rejected": 0}
    seen = {}
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            rep["files_found"] += 1
            rel = f"images/{cls}/{fn}"
            f = image_features(os.path.join(cdir, fn))
            if f is None:
                rep["unreadable_removed"] += 1
                continue
            if f["_hash"] in seen:                               # near-identical image
                rep["duplicates_removed"] += 1
                continue
            seen[f["_hash"]] = rel
            row = {"image_id": os.path.splitext(fn)[0], "filepath": rel,
                   "material": meta.at[rel, "material"] if meta is not None and rel in meta.index else cls.replace("_", " ").title(),
                   "material_type": meta.at[rel, "material_type"] if meta is not None and rel in meta.index else "Unknown",
                   "moisture_state": meta.at[rel, "moisture_state"] if meta is not None and rel in meta.index else "unknown"}
            row.update(f)
            rows.append(row)
    df = pd.DataFrame(rows)
    # quality gate: reject nearly black / blown-out frames
    bad = (df.brightness < 0.04) | (df.brightness > 0.97) | (df.contrast < 0.005)
    rep["quality_rejected"] = int(bad.sum())
    df = df[~bad].reset_index(drop=True)
    # per-channel mean/std over the dataset -> used to standardise pixels for a CNN
    n_pix = len(df) * 128 * 128
    rgb_sum = np.sum(np.stack(df["_rgb_sum"]), 0); rgb_sq = np.sum(np.stack(df["_rgb_sqsum"]), 0)
    mean = rgb_sum / n_pix; std = np.sqrt(rgb_sq / n_pix - mean ** 2)
    stats = {"channel_mean_rgb": mean.round(4).tolist(), "channel_std_rgb": std.round(4).tolist(), "image_size": [128, 128]}
    df = df.drop(columns=["_hash", "_rgb_sum", "_rgb_sqsum"])
    # stratified 80/20 split (fixed seed)
    rng = np.random.default_rng(42)
    df["split"] = "train"
    for _, idx in df.groupby("material").groups.items():
        idx = rng.permutation(list(idx))
        df.loc[idx[: max(1, int(round(0.2 * len(idx))))], "split"] = "test"
    rep.update({"images_kept": int(len(df)), "classes": int(df.material.nunique()),
                "train": int((df.split == "train").sum()), "test": int((df.split == "test").sum())})
    return df.round(5), rep, stats


# ============================================================ main
def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    report = {}
    print("[1/3] Slope monitoring data ...")
    raw = pd.read_csv(os.path.join(RAW_DIR, "slope_monitoring_raw_field.csv"), dtype={"recent_blasting": str})
    slope, rep, audit = preprocess_slope(raw)
    slope.to_csv(os.path.join(PROC_DIR, "cleaned_features.csv"), index=False)
    audit.to_csv(os.path.join(PROC_DIR, "slope_cleaning_audit_log.csv"), index=False)
    report["slope"] = rep
    print(f"      {rep['rows_raw']} -> {rep['rows_clean']} rows | duplicates {rep['duplicates_removed']} | "
          f"missing {rep['missing_before_cleaning']}+{rep['missing_to_impute'] - rep['missing_before_cleaning']} -> {rep['missing_after_cleaning']}")

    print("[2/3] Rock & soil properties ...")
    rs, rep = preprocess_rock_soil(pd.read_csv(os.path.join(RAW_DIR, "rock_soil_properties.csv")))
    rs.to_csv(os.path.join(PROC_DIR, "rock_soil_clean.csv"), index=False)
    report["rock_soil"] = rep
    print(f"      {rep['rows_raw']} -> {rep['rows_clean']} rows | names fixed {rep['material_names_fixed']} | missing filled {rep['missing_to_impute']}")

    print("[3/3] Images ...")
    imgs, rep, stats = preprocess_images()
    imgs.to_csv(os.path.join(PROC_DIR, "image_features.csv"), index=False)
    imgs[["image_id", "filepath", "material", "split"]].to_csv(os.path.join(PROC_DIR, "image_split.csv"), index=False)
    json.dump(stats, open(os.path.join(PROC_DIR, "image_channel_stats.json"), "w"), indent=2)
    report["images"] = rep
    print(f"      kept {rep['images_kept']}/{rep['files_found']} images | classes {rep['classes']} | train/test {rep['train']}/{rep['test']}")

    json.dump(report, open(os.path.join(PROC_DIR, "preprocessing_report.json"), "w"), indent=2)
    print("Report saved -> data/processed/preprocessing_report.json")


if __name__ == "__main__":
    main()
