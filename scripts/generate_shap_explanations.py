"""Generate SHAP global bar/beeswarm and one local waterfall from synthetic data.

This is an optional, potentially slower offline explanation job. It fits a
balanced Random Forest explainer model for analysis; it does not change the
serving model. If SHAP's installed version cannot render a plot, the script
exits with an explanatory message and permutation importance remains available.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import backend.ml_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
FEATURES=["slope_height_m","slope_angle_deg","cohesion_kpa","friction_angle_deg","unit_weight_kn_m3","rainfall_mm","groundwater_level","pore_pressure_kpa","deformation_rate_mm_day","displacement_mm","recent_blasting","blast_intensity","equipment_load","factor_of_safety"]
LABELS=["Low","Moderate","High","Critical"]

def main():
    try:
        import shap
        import matplotlib.pyplot as plt
        df=pd.read_csv(ROOT/"data/processed/cleaned_features.csv").dropna(subset=FEATURES+["risk_class"])
        prep=joblib.load(ROOT/"models/preprocessor.joblib")
        X=prep.transform(df); y=df["risk_class"].astype(int).to_numpy()
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        model=RandomForestClassifier(n_estimators=120,class_weight="balanced",min_samples_leaf=2,random_state=42,n_jobs=1).fit(X_train,y_train)
        sample=X_test[:min(100,len(X_test))]
        explainer=shap.TreeExplainer(model)
        values=explainer.shap_values(sample)
        # SHAP APIs differ: old releases return list[class](rows,features); newer may return rows,features,classes.
        if isinstance(values,list): class_values=np.asarray(values[-1]); base=np.asarray(explainer.expected_value)[-1]
        else:
            arr=np.asarray(values)
            if arr.ndim==3: class_values=arr[:,:,min(3,arr.shape[2]-1)]; base=np.asarray(explainer.expected_value)[min(3,len(np.asarray(explainer.expected_value))-1)]
            else: class_values=arr; base=float(np.asarray(explainer.expected_value).reshape(-1)[0])
        out=ROOT/"results"; out.mkdir(exist_ok=True)
        plt.figure(); shap.summary_plot(class_values,sample,feature_names=FEATURES,show=False); plt.tight_layout(); plt.savefig(out/"ml_rigor_shap_beeswarm.png",dpi=160,bbox_inches="tight"); plt.close()
        plt.figure(); shap.summary_plot(class_values,sample,feature_names=FEATURES,plot_type="bar",show=False); plt.tight_layout(); plt.savefig(out/"ml_rigor_shap_bar.png",dpi=160,bbox_inches="tight"); plt.close()
        exp=shap.Explanation(values=class_values[0],base_values=float(base),data=sample[0],feature_names=FEATURES)
        plt.figure(); shap.plots.waterfall(exp,max_display=12,show=False); plt.tight_layout(); plt.savefig(out/"ml_rigor_shap_waterfall.png",dpi=160,bbox_inches="tight"); plt.close()
        print("Saved SHAP global bar, beeswarm and local waterfall. This explainer is a separate research Random Forest, not the serving model.")
    except Exception as exc:
        print(f"SHAP artifact generation unavailable: {type(exc).__name__}: {exc}")
        print("Use results/ml_rigor_importance.png (permutation importance) as the supported global fallback.")

if __name__=="__main__": main()
