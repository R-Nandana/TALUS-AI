"""
TALUS Data Generation & ML Model Training Script
=================================================
1. Generates realistic synthetic geotechnical slope monitoring dataset (labeled DEMO/SYNTHETIC).
2. Performs scientific preprocessing, schema validation, outlier handling, and train/test split.
3. Computes physics-derived Factor of Safety (FoS).
4. Trains & evaluates 4 models: Logistic Regression, Support Vector Machine, Random Forest, Gradient Boosting.
5. Generates performance plots (Confusion Matrix, ROC Curve, Feature Importance) and saves to results/figures/.
6. Persists best model, preprocessor, and metadata via Joblib & JSON.
7. Initializes SQLite database.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive background plot generation
import matplotlib.pyplot as plt

# Ensure TALUS root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from backend.physics import calculate_factor_of_safety
from backend.ml_engine import (
    PreprocessingPipeline,
    MultiClassLogisticRegression,
    SupportVectorMachineOvR,
    RandomForestClassifierModel,
    GradientBoostingClassifierModel,
    compute_metrics,
    compute_confusion_matrix,
)
from backend.database import init_db


def generate_synthetic_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    sector_ids = [f"S{i}" for i in range(1, 11)]
    start_date = pd.Timestamp("2026-05-01")
    
    rows = []
    for i in range(n_samples):
        sector = np.random.choice(sector_ids)
        day_offset = np.random.randint(0, 120)
        timestamp = (start_date + pd.Timedelta(days=day_offset, hours=np.random.randint(0, 24))).strftime("%Y-%m-%d %H:%M:%S")
        
        # Sector baseline geometry
        sec_num = int(sector[1:])
        slope_height_m = round(float(np.random.uniform(25.0, 140.0) + sec_num * 3.0), 1)
        slope_angle_deg = round(float(np.random.uniform(28.0, 64.0)), 1)
        
        # Geotechnical properties
        cohesion_kpa = round(float(np.random.uniform(12.0, 95.0)), 1)
        friction_angle_deg = round(float(np.random.uniform(18.0, 42.0)), 1)
        unit_weight_kn_m3 = round(float(np.random.uniform(19.0, 28.0)), 1)
        
        # Environmental & pore pressure
        rainfall_mm = round(float(np.random.exponential(scale=20.0)), 1)
        rainfall_mm = min(150.0, rainfall_mm)
        groundwater_level = round(float(np.random.uniform(1.0, 30.0)), 1)
        pore_pressure_kpa = round(float((groundwater_level * 9.81) + rainfall_mm * 0.85 + np.random.normal(0, 5)), 1)
        pore_pressure_kpa = max(0.0, min(300.0, pore_pressure_kpa))
        
        # Operational parameters
        recent_blasting = int(np.random.binomial(1, 0.35))
        blast_intensity = round(float(np.random.uniform(1.5, 9.5) if recent_blasting else 0.0), 1)
        equipment_load = round(float(np.random.uniform(5.0, 90.0)), 1)
        
        # Physics Factor of Safety
        fos = calculate_factor_of_safety(
            slope_height_m=slope_height_m,
            slope_angle_deg=slope_angle_deg,
            cohesion_kpa=cohesion_kpa,
            friction_angle_deg=friction_angle_deg,
            unit_weight_kn_m3=unit_weight_kn_m3,
            pore_pressure_kpa=pore_pressure_kpa,
            recent_blasting=recent_blasting,
            blast_intensity=blast_intensity,
            equipment_load=equipment_load
        )
        
        # Deformation and displacement (simulated monitoring with correlation to physical stress)
        stress_indicator = (slope_angle_deg / 60.0) * (pore_pressure_kpa / 150.0) / max(0.5, fos)
        deformation_rate_mm_day = round(float(np.random.exponential(scale=2.5 * stress_indicator) + (1.0 / max(0.4, fos))), 2)
        deformation_rate_mm_day = max(0.1, min(50.0, deformation_rate_mm_day))
        
        displacement_mm = round(float(deformation_rate_mm_day * np.random.uniform(3.0, 15.0) + np.random.normal(0, 10)), 1)
        displacement_mm = max(0.5, min(500.0, displacement_mm))
        
        # Multidimensional target risk mapping with stochastic noise (avoids exact formula leakage)
        # Combine FoS, deformation rate, pore pressure, and blast impact
        risk_score = (
            (1.5 / max(0.4, fos)) * 0.35 +
            (deformation_rate_mm_day / 15.0) * 0.30 +
            (pore_pressure_kpa / 120.0) * 0.20 +
            (blast_intensity / 10.0 if recent_blasting else 0.0) * 0.15 +
            np.random.normal(0, 0.08)
        )
        
        if risk_score < 0.65:
            risk_label = "Low"
            risk_class = 0
        elif risk_score < 1.05:
            risk_label = "Moderate"
            risk_class = 1
        elif risk_score < 1.55:
            risk_label = "High"
            risk_class = 2
        else:
            risk_label = "Critical"
            risk_class = 3

        rows.append({
            "sector_id": sector,
            "timestamp": timestamp,
            "slope_height_m": slope_height_m,
            "slope_angle_deg": slope_angle_deg,
            "cohesion_kpa": cohesion_kpa,
            "friction_angle_deg": friction_angle_deg,
            "unit_weight_kn_m3": unit_weight_kn_m3,
            "rainfall_mm": rainfall_mm,
            "groundwater_level": groundwater_level,
            "pore_pressure_kpa": pore_pressure_kpa,
            "deformation_rate_mm_day": deformation_rate_mm_day,
            "displacement_mm": displacement_mm,
            "recent_blasting": recent_blasting,
            "blast_intensity": blast_intensity,
            "equipment_load": equipment_load,
            "factor_of_safety": fos,
            "risk_class": risk_class,
            "risk_label": risk_label,
            "dataset_type": "DEMO/SYNTHETIC"
        })
        
    df = pd.DataFrame(rows)
    return df


def main():
    print("==================================================")
    print("   TALUS — Data Generation & Model Training Pipeline")
    print("==================================================")
    
    # 1. Directories setup
    raw_dir = os.path.join(ROOT_DIR, "data", "raw")
    proc_dir = os.path.join(ROOT_DIR, "data", "processed")
    models_dir = os.path.join(ROOT_DIR, "models")
    results_dir = os.path.join(ROOT_DIR, "results")
    fig_dir = os.path.join(results_dir, "figures")
    
    for d in [raw_dir, proc_dir, models_dir, results_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. Generate Synthetic Dataset
    print("\n[Step 1/6] Generating synthetic mine monitoring dataset...")
    df = generate_synthetic_dataset(n_samples=1200)
    raw_path = os.path.join(raw_dir, "slope_monitoring_synthetic.csv")
    df.to_csv(raw_path, index=False)
    print(f"-> Saved raw dataset: {raw_path} ({len(df)} records)")

    # 3. Preprocessing & Data Cleaning
    print("\n[Step 2/6] Performing preprocessing & schema validation...")
    
    # Feature columns list
    feature_cols = [
        "slope_height_m", "slope_angle_deg", "cohesion_kpa", "friction_angle_deg",
        "unit_weight_kn_m3", "rainfall_mm", "groundwater_level", "pore_pressure_kpa",
        "deformation_rate_mm_day", "displacement_mm", "recent_blasting",
        "blast_intensity", "equipment_load", "factor_of_safety"
    ]
    
    target_col = "risk_class"

    # Preprocessing transformation (Fit ONLY on train set to prevent leakage)
    train_size = int(len(df) * 0.8)
    # Shuffle dataset reproducibly
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    train_df = df_shuffled.iloc[:train_size].copy()
    test_df = df_shuffled.iloc[train_size:].copy()
    
    preprocessor = PreprocessingPipeline(feature_cols)
    X_train_scaled = preprocessor.fit_transform(train_df)
    y_train = train_df[target_col].values
    
    X_test_scaled = preprocessor.transform(test_df)
    y_test = test_df[target_col].values

    # Save cleaned processed dataset
    cleaned_path = os.path.join(proc_dir, "cleaned_features.csv")
    df_shuffled.to_csv(cleaned_path, index=False)
    print(f"-> Saved processed dataset: {cleaned_path}")
    print(f"-> Train samples: {len(X_train_scaled)}, Test samples: {len(X_test_scaled)}")

    # 4. Model Training & Comparison
    print("\n[Step 3/6] Training & comparing machine learning models...")
    
    models = {
        "Logistic Regression": MultiClassLogisticRegression(lr=0.08, n_epochs=350, reg=0.01),
        "Support Vector Machine": SupportVectorMachineOvR(C=1.5, n_epochs=300, lr=0.02),
        "Random Forest": RandomForestClassifierModel(n_estimators=20, max_depth=7),
        "Gradient Boosting": GradientBoostingClassifierModel(n_estimators=20, lr=0.1, max_depth=4)
    }

    comparison_results = []
    fitted_models = {}
    test_predictions = {}
    test_probabilities = {}

    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train_scaled, y_train, num_classes=4)
        
        y_probs = model.predict_proba(X_test_scaled)
        y_pred = model.predict(X_test_scaled)
        
        metrics = compute_metrics(y_test, y_pred, y_probs, num_classes=4)
        print(f"      Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
        metrics["model_name"] = name
        comparison_results.append(metrics)
        fitted_models[name] = model
        test_predictions[name] = y_pred
        test_probabilities[name] = y_probs

    # Save comparison CSV
    comp_df = pd.DataFrame(comparison_results)[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    comp_path = os.path.join(results_dir, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"-> Saved model comparison matrix: {comp_path}")

    # Select Best Model dynamically based on F1-Score
    best_res = max(comparison_results, key=lambda x: x["f1_score"])
    best_model_name = best_res["model_name"]
    best_model = fitted_models[best_model_name]
    best_y_pred = test_predictions[best_model_name]
    best_y_probs = test_probabilities[best_model_name]
    
    print(f"\n[Step 4/6] Dynamically selected best model: **{best_model_name}** (F1-Score: {best_res['f1_score']:.4f})")

    # 5. Generate Evaluation Plots
    print("\n[Step 5/6] Generating evaluation figures (Confusion Matrix, ROC Curve, Feature Importance)...")
    
    # Plot 1: Confusion Matrix
    cm = compute_confusion_matrix(y_test, best_y_pred, num_classes=4)
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.colorbar()
    tick_marks = np.arange(4)
    labels = ["Low", "Moderate", "High", "Critical"]
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    thresh = cm.max() / 2.0
    for i in range(4):
        for j in range(4):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Risk Label")
    plt.xlabel("Predicted Risk Label")
    plt.tight_layout()
    cm_fig_path = os.path.join(fig_dir, "confusion_matrix.png")
    plt.savefig(cm_fig_path)
    plt.close()

    # Plot 2: ROC Curve (One-vs-Rest for Best Model)
    plt.figure(figsize=(7, 5), dpi=150)
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    for c in range(4):
        y_binary = (y_test == c).astype(int)
        p_scores = best_y_probs[:, c]
        sort_idx = np.argsort(-p_scores)
        y_sorted = y_binary[sort_idx]
        n_pos = np.sum(y_binary)
        n_neg = len(y_binary) - n_pos
        if n_pos > 0 and n_neg > 0:
            tpr = np.cumsum(y_sorted) / n_pos
            fpr = np.cumsum(1 - y_sorted) / n_neg
            plt.plot(fpr, tpr, color=colors[c], label=f"Class {labels[c]} (AUC)")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) — {best_model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_fig_path = os.path.join(fig_dir, "roc_curve.png")
    plt.savefig(roc_fig_path)
    plt.close()

    # Plot 3: Feature Importance
    if hasattr(best_model, "feature_importances"):
        importances = best_model.feature_importances()
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(8, 6), dpi=150)
    plt.barh(range(len(feature_cols)), importances[sorted_idx], align="center", color="#3498db")
    plt.yticks(range(len(feature_cols)), [feature_cols[i] for i in sorted_idx])
    plt.xlabel("Relative Feature Importance Weight")
    plt.title(f"Feature Importances — {best_model_name}")
    plt.tight_layout()
    feat_fig_path = os.path.join(fig_dir, "feature_importance.png")
    plt.savefig(feat_fig_path)
    plt.close()

    # 6. Save Model Artifacts & Metadata
    model_path = os.path.join(models_dir, "model.joblib")
    prep_path = os.path.join(models_dir, "preprocessor.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, prep_path)

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": "f1_score",
        "performance": best_res,
        "feature_names": feature_cols,
        "class_labels": ["Low", "Moderate", "High", "Critical"],
        "all_models_evaluated": [r["model_name"] for r in comparison_results],
        "dataset_size": len(df),
        "train_size": len(X_train_scaled),
        "test_size": len(X_test_scaled),
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"-> Saved model artifact: {model_path}")
    print(f"-> Saved preprocessor: {prep_path}")
    print(f"-> Saved model metadata: {meta_path}")

    # 7. Initialize SQLite Database
    print("\n[Step 6/6] Initializing SQLite database & seeding initial sector observations...")
    init_db()
    
    print("\n==================================================")
    print("   SUCCESS! Training pipeline completed.")
    print("==================================================")


if __name__ == "__main__":
    main()
