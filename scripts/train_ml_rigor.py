"""Reproducible ML-rigor audit for TALUS synthetic telemetry.

Benchmarks four classifiers with stratified 5-fold macro-F1, evaluates an
8-sector/2-sector hold-out, measures FoS ablation and calibration, and writes
artifacts. Replaces the serving classifier only after a full CV benchmark.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, brier_score_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import backend.ml_engine  # register custom preprocessing class for the persisted joblib artifact
DATA=ROOT/"data/processed/cleaned_features.csv"
MODEL_PATH=ROOT/"models/model.joblib"
PREP_PATH=ROOT/"models/preprocessor.joblib"
META_PATH=ROOT/"models/model_metadata.json"
RESULTS=ROOT/"results"
RESULTS.mkdir(exist_ok=True)
FEATURES=["slope_height_m","slope_angle_deg","cohesion_kpa","friction_angle_deg","unit_weight_kn_m3","rainfall_mm","groundwater_level","pore_pressure_kpa","deformation_rate_mm_day","displacement_mm","recent_blasting","blast_intensity","equipment_load","factor_of_safety"]
LABELS=["Low","Moderate","High","Critical"]

def main():
    df=pd.read_csv(DATA)
    df=df.dropna(subset=FEATURES+["risk_class","risk_label"])
    X=df[FEATURES].astype(float); y=df["risk_class"].astype(int)
    oldprep=joblib.load(PREP_PATH)
    # Use the project's persisted scaler/imputer so the selected model remains compatible with API inference.
    Xs=oldprep.transform(df)
    models={
      "Logistic Regression": LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42),
      "Support Vector Machine": SVC(C=2.0,kernel="rbf",probability=True,class_weight="balanced",random_state=42),
      "Random Forest": RandomForestClassifier(n_estimators=100,class_weight="balanced",min_samples_leaf=2,random_state=42,n_jobs=1),
      "Gradient Boosting": GradientBoostingClassifier(n_estimators=70,learning_rate=.06,max_depth=2,random_state=42)
    }
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    rows=[]; cv_by_name={}
    for name,model in models.items():
        scores=cross_validate(model,Xs,y,cv=cv,scoring={"macro_f1":"f1_macro","weighted_f1":"f1_weighted","accuracy":"accuracy","precision_macro":"precision_macro","recall_macro":"recall_macro","roc_auc_ovr":"roc_auc_ovr"},n_jobs=1,return_train_score=False)
        cv_by_name[name]=scores
        rows.append({"model_name":name,"accuracy":float(np.mean(scores["test_accuracy"])),"precision":float(np.mean(scores["test_precision_macro"])),"recall":float(np.mean(scores["test_recall_macro"])),"f1_score":float(np.mean(scores["test_macro_f1"])),"macro_f1_mean":float(np.mean(scores["test_macro_f1"])),"macro_f1_std":float(np.std(scores["test_macro_f1"])),"weighted_f1":float(np.mean(scores["test_weighted_f1"])),"roc_auc":float(np.mean(scores["test_roc_auc_ovr"]))})
        print(name, "macro-F1", round(rows[-1]["macro_f1_mean"],4), "+/-", round(rows[-1]["macro_f1_std"],4))
    comparison=pd.DataFrame(rows).sort_values("macro_f1_mean",ascending=False)
    comparison.to_csv(RESULTS/"ml_rigor_model_comparison.csv",index=False)
    comparison[["model_name","accuracy","precision","recall","f1_score","roc_auc"]].to_csv(RESULTS/"model_comparison.csv",index=False)
    selected_name=str(comparison.iloc[0]["model_name"])
    selected=clone(models[selected_name])
    X_train,X_test,y_train,y_test=train_test_split(Xs,y,test_size=.2,random_state=42,stratify=y)
    selected.fit(X_train,y_train)
    pred=selected.predict(X_test); prob=selected.predict_proba(X_test)
    report=classification_report(y_test,pred,labels=[0,1,2,3],target_names=LABELS,output_dict=True,zero_division=0)
    # Calibration: multiclass Brier score and classwise reliability curves.
    y_bin=label_binarize(y_test,classes=[0,1,2,3])
    brier=float(np.mean(np.sum((prob-y_bin)**2,axis=1)))
    plt.figure(figsize=(7,5))
    for idx,label in enumerate(LABELS):
        frac,mean=calibration_curve(y_bin[:,idx],prob[:,idx],n_bins=8,strategy="quantile")
        plt.plot(mean,frac,marker="o",label=label)
    plt.plot([0,1],[0,1],"k--",label="Ideal calibration"); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction"); plt.title("Multiclass reliability curves"); plt.legend(); plt.tight_layout(); plt.savefig(RESULTS/"ml_rigor_calibration.png",dpi=160); plt.close()
    # CV chart
    plt.figure(figsize=(9,5)); plt.bar(comparison.model_name,comparison.macro_f1_mean,yerr=comparison.macro_f1_std,capsize=4); plt.ylabel("Macro-F1 (5-fold mean ± SD)"); plt.title("Stratified 5-fold model comparison"); plt.xticks(rotation=18,ha="right"); plt.tight_layout(); plt.savefig(RESULTS/"ml_rigor_cv.png",dpi=160); plt.close()
    # FoS ablation on the same stratified hold-out split using Random Forest.
    nofos=[f for f in FEATURES if f!="factor_of_safety"]
    idx=np.array([FEATURES.index(f) for f in nofos])
    ablation=[]
    for label,xx in [("With Factor of Safety",Xs),("Without Factor of Safety",Xs[:,idx])]:
        xx_train,xx_test,yy_train,yy_test=train_test_split(xx,y,test_size=.2,random_state=42,stratify=y)
        m=RandomForestClassifier(n_estimators=100,class_weight="balanced",min_samples_leaf=2,random_state=42,n_jobs=1).fit(xx_train,yy_train)
        pp=m.predict(xx_test)
        ablation.append({"setting":label,"macro_f1":float(f1_score(yy_test,pp,average="macro")),"weighted_f1":float(f1_score(yy_test,pp,average="weighted")),"accuracy":float(accuracy_score(yy_test,pp))})
    pd.DataFrame(ablation).to_csv(RESULTS/"ml_rigor_fos_ablation.csv",index=False)
    plt.figure(figsize=(7,5)); plt.bar([a["setting"] for a in ablation],[a["macro_f1"] for a in ablation],color=["#4f9d8f","#d99c45"]); plt.ylabel("Macro-F1 on stratified hold-out"); plt.title("Factor of Safety ablation"); plt.ylim(0,1); plt.xticks(rotation=10); plt.tight_layout(); plt.savefig(RESULTS/"ml_rigor_ablation.png",dpi=160); plt.close()
    # Sector hold-out: sorted first 8 sectors train, final 2 sectors test. Report missing-class limitation.
    sector_order=sorted(df.sector_id.astype(str).unique(),key=lambda z:int(z[1:]) if z[1:].isdigit() else z)
    train_sectors=set(sector_order[:-2]); test_sectors=set(sector_order[-2:])
    train_mask=df.sector_id.astype(str).isin(train_sectors).to_numpy(); test_mask=df.sector_id.astype(str).isin(test_sectors).to_numpy()
    holdout_model=clone(models[selected_name]); holdout_model.fit(Xs[train_mask],y.to_numpy()[train_mask]); hold_pred=holdout_model.predict(Xs[test_mask])
    sector_holdout={"train_sectors":sorted(train_sectors),"test_sectors":sorted(test_sectors),"records_train":int(train_mask.sum()),"records_test":int(test_mask.sum()),"macro_f1":float(f1_score(y.to_numpy()[test_mask],hold_pred,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y.to_numpy()[test_mask],hold_pred,average="weighted",zero_division=0)),"accuracy":float(accuracy_score(y.to_numpy()[test_mask],hold_pred)),"class_report":classification_report(y.to_numpy()[test_mask],hold_pred,labels=[0,1,2,3],target_names=LABELS,output_dict=True,zero_division=0)}
    # Explainability plot: permutation importance is always available. Optional SHAP artifacts are attempted separately.
    from sklearn.inspection import permutation_importance
    perm=permutation_importance(selected,X_test,y_test,n_repeats=5,random_state=42,scoring="f1_macro",n_jobs=1)
    importance=pd.DataFrame({"feature":FEATURES,"importance_mean":perm.importances_mean,"importance_std":perm.importances_std}).sort_values("importance_mean",ascending=False)
    importance.to_csv(RESULTS/"ml_rigor_feature_importance.csv",index=False)
    top=importance.head(12).sort_values("importance_mean")
    plt.figure(figsize=(8,6)); plt.barh(top.feature,top.importance_mean,xerr=top.importance_std); plt.xlabel("Permutation importance (macro-F1 decrease)"); plt.title("Global feature importance"); plt.tight_layout(); plt.savefig(RESULTS/"ml_rigor_importance.png",dpi=160); plt.close()
    shap_status="not generated; permutation importance used"
    try:
        import shap
        sample=X_test[:min(150,len(X_test))]
        explainer=shap.TreeExplainer(selected) if hasattr(selected,"estimators_") or hasattr(selected,"tree_") else None
        if explainer is None: raise RuntimeError("Selected estimator is not tree-based; permutation importance is used instead of expensive Kernel SHAP")
        sv=explainer.shap_values(sample)
        plt.figure(); shap.summary_plot(sv, sample, feature_names=FEATURES, show=False); plt.tight_layout(); plt.savefig(RESULTS/"ml_rigor_shap_beeswarm.png",dpi=160,bbox_inches="tight"); plt.close()
        plt.figure(); shap.summary_plot(sv, sample, feature_names=FEATURES, plot_type="bar", show=False); plt.tight_layout(); plt.savefig(RESULTS/"ml_rigor_shap_bar.png",dpi=160,bbox_inches="tight"); plt.close()
        shap_status="SHAP summary artifacts generated; inspect plots for multi-class shape interpretation"
    except Exception as exc:
        shap_status=f"SHAP unavailable in this environment ({type(exc).__name__}: {exc}); permutation importance artifacts are provided instead"
    # Save selected model for serving; API preprocessor already emits same 14 scaled features.
    joblib.dump(selected,MODEL_PATH)
    metadata=json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    old_perf=metadata.get("performance",{})
    metadata.update({"selected_model":selected_name,"selection_metric":"5-fold stratified CV macro_f1 (mean)","all_models_evaluated":list(models.keys()),"dataset_size":int(len(df)),"train_size":int(len(X_train)),"test_size":int(len(X_test)),"saved_at":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")})
    metadata["performance"]={**old_perf,"model_name":selected_name,"accuracy":float(accuracy_score(y_test,pred)),"precision":float(precision_score(y_test,pred,average="macro",zero_division=0)),"recall":float(recall_score(y_test,pred,average="macro",zero_division=0)),"f1_score":float(f1_score(y_test,pred,average="macro",zero_division=0)),"macro_f1":float(f1_score(y_test,pred,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y_test,pred,average="weighted",zero_division=0)),"roc_auc":float(roc_auc_score(y_test,prob,multi_class="ovr",average="macro"))}
    summary={"selected_model_by_5fold_macro_f1":selected_name,"selection_metric":"Stratified 5-fold CV macro-F1 mean; class_weight='balanced' where supported","cross_validation":{r["model_name"]:{"macro_f1_mean":r["macro_f1_mean"],"macro_f1_std":r["macro_f1_std"],"weighted_f1_mean":r["weighted_f1"]} for r in rows},"test_metrics":metadata["performance"],"per_class_report":{k:{metric:float(v) for metric,v in vals.items() if isinstance(v,(int,float))} for k,vals in report.items() if isinstance(vals,dict)},"factor_of_safety_ablation":ablation,"sector_holdout":sector_holdout,"calibration":{"multiclass_brier_score":brier,"definition":"mean per-row sum of squared probability error across classes; lower is better"},"explainability":shap_status,"dataset_class_counts":df.risk_label.value_counts().to_dict(),"important_limitations":["Data is synthetic and generated by the project pipeline.","Factor of Safety is related to the synthetic label construction; ablation measures dependence but cannot prove real-world leakage.","Sector hold-out uses the last two sector IDs and may omit classes; interpret macro-F1 alongside per-class support.","Metrics are not operational mine-site performance."]}
    if (RESULTS/"ml_rigor_shap_beeswarm.png").exists() and (RESULTS/"ml_rigor_shap_waterfall.png").exists():
        summary["explainability"] = "SHAP global bar, beeswarm and local waterfall artifacts exist from scripts/generate_shap_explanations.py using a separate research Random Forest; serving-model permutation importance is also available."
    (RESULTS/"ml_rigor_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    META_PATH.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    # Maintain frontend's existing expected CSV fields while updating model selection evidence.
    comparison.rename(columns={"macro_f1_mean":"cv_macro_f1_mean"}).to_csv(RESULTS/"ml_rigor_model_comparison.csv",index=False)
    print("Selected serving model:",selected_name)
    print("Summary:",RESULTS/"ml_rigor_summary.json")
    print("SHAP status:",shap_status)

if __name__=="__main__": main()
