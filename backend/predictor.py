"""
TALUS Predictor Module
======================
Loads persisted model & preprocessor, performs physics + ML inference,
computes explainability feature contributions, and generates prototype safety recommendations.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from backend.physics import run_physics_assessment
from backend.ml_engine import ModelExplainer, PreprocessingPipeline
from backend.database import save_prediction

MODEL_PATH = os.path.join(ROOT_DIR, "models", "model.joblib")
PREP_PATH = os.path.join(ROOT_DIR, "models", "preprocessor.joblib")
META_PATH = os.path.join(ROOT_DIR, "models", "model_metadata.json")
CLEANED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "cleaned_features.csv")

# Global cached predictor state
_MODEL = None
_PREPROCESSOR = None
_METADATA = None
_EXPLAINER = None
_BACKGROUND_X = None


def load_predictor():
    global _MODEL, _PREPROCESSOR, _METADATA, _EXPLAINER, _BACKGROUND_X
    if _MODEL is not None and _PREPROCESSOR is not None:
        return

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREP_PATH):
        raise FileNotFoundError("Model or preprocessor artifact not found! Run generate_data_and_train.py first.")

    _MODEL = joblib.load(MODEL_PATH)
    _PREPROCESSOR = joblib.load(PREP_PATH)

    with open(META_PATH, "r") as f:
        _METADATA = json.load(f)

    if os.path.exists(CLEANED_DATA_PATH):
        df_bg = pd.read_csv(CLEANED_DATA_PATH)
        _BACKGROUND_X = _PREPROCESSOR.transform(df_bg)
    else:
        _BACKGROUND_X = np.zeros((10, len(_PREPROCESSOR.feature_names)))

    _EXPLAINER = ModelExplainer(
        model=_MODEL,
        preprocessor=_PREPROCESSOR,
        feature_names=_PREPROCESSOR.feature_names,
        background_X=_BACKGROUND_X
    )


def get_decision_recommendation(ml_risk_label: str, fos: float) -> str:
    if ml_risk_label == "Critical" or fos < 1.0:
        return "CRITICAL: Restrict unnecessary access to sector immediately. Activate site-specific geotechnical emergency protocol and initiate slope displacement radar review."
    elif ml_risk_label == "High" or fos < 1.3:
        return "HIGH RISK: Engineering inspection and access review recommended. Increase automated piezometer & Prism monitoring frequency to 15-minute intervals."
    elif ml_risk_label == "Moderate" or fos < 1.5:
        return "MODERATE RISK: Increase observation frequency. Verify pit-wall drainage channels and record post-blast displacement trends."
    else:
        return "LOW RISK: Standard slope monitoring regimen active. Routine visual inspection & daily sensor telemetry."


def predict_sector_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    load_predictor()

    # 1. Physics Engine Assessment
    physics_res = run_physics_assessment(input_data)
    fos = physics_res["factor_of_safety"]

    # 2. Combine inputs with calculated FoS
    full_input = input_data.copy()
    full_input["factor_of_safety"] = fos

    # 3. Create DataFrame and apply preprocessing
    df_single = pd.DataFrame([full_input])
    X_scaled = _PREPROCESSOR.transform(df_single)

    # 4. Run ML Inference
    probs = _MODEL.predict_proba(X_scaled)[0]
    pred_class = int(_MODEL.predict(X_scaled)[0])
    risk_labels = _METADATA.get("class_labels", ["Low", "Moderate", "High", "Critical"])
    ml_risk_label = risk_labels[pred_class]
    ml_probability = float(probs[pred_class])

    # 5. Explainable AI Feature Contribution
    contributors = _EXPLAINER.explain_instance(X_scaled, pred_class)

    # 6. Safety Recommendation
    recommendation = get_decision_recommendation(ml_risk_label, fos)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_record = {
        "sector_id": str(input_data.get("sector_id", "S1")),
        "timestamp": timestamp_str,
        "slope_height_m": float(input_data.get("slope_height_m", 50.0)),
        "slope_angle_deg": float(input_data.get("slope_angle_deg", 45.0)),
        "cohesion_kpa": float(input_data.get("cohesion_kpa", 35.0)),
        "friction_angle_deg": float(input_data.get("friction_angle_deg", 30.0)),
        "unit_weight_kn_m3": float(input_data.get("unit_weight_kn_m3", 24.0)),
        "rainfall_mm": float(input_data.get("rainfall_mm", 25.0)),
        "groundwater_level": float(input_data.get("groundwater_level", 10.0)),
        "pore_pressure_kpa": float(input_data.get("pore_pressure_kpa", 50.0)),
        "deformation_rate_mm_day": float(input_data.get("deformation_rate_mm_day", 3.5)),
        "displacement_mm": float(input_data.get("displacement_mm", 25.0)),
        "recent_blasting": int(input_data.get("recent_blasting", 0)),
        "blast_intensity": float(input_data.get("blast_intensity", 0.0)),
        "equipment_load": float(input_data.get("equipment_load", 0.0)),
        "factor_of_safety": fos,
        "physics_category": physics_res["physics_category"],
        "physics_description": physics_res["physics_description"],
        "ml_risk_label": ml_risk_label,
        "ml_risk_probability": round(ml_probability, 4),
        "class_probabilities": {str(risk_labels[i]): round(float(probs[i]), 6) for i in range(min(len(risk_labels), len(probs)))},
        "decision_recommendation": recommendation,
        "top_contributors": contributors[:5],
        "scenario_name": str(input_data.get("scenario_name", "User Prediction")),
        "formulation": physics_res["formulation"],
        "disclaimer": "PROTOTYPE DECISION-SUPPORT SYSTEM: TALUS predictions are prototype analytical aids. Final slope safety actions require certified geotechnical engineering review."
    }

    # 7. Save to SQLite Database
    db_id = save_prediction(output_record)
    output_record["saved_db_id"] = db_id

    return output_record


def get_model_info() -> Dict[str, Any]:
    load_predictor()
    return _METADATA
