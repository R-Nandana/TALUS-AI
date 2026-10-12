# TALUS Viva Questions & Answers

> TALUS is an academic prototype. The bundled monitoring data is synthetic and must not be presented as real mine-site evidence.

1. **What problem does TALUS address?** It combines simplified slope-stability physics, ML risk classes, telemetry history and decision-support explanations in one local control-room interface.
2. **What does Factor of Safety (FoS) mean?** It is a ratio-style stability indicator comparing available resistance with driving demand; a lower value indicates less modeled margin.
3. **Which physics model is used?** A simplified infinite-slope / limit-equilibrium formulation with pore-pressure and pseudo-static blasting terms.
4. **What is a key physics limitation?** It assumes simplified geometry and material behaviour and does not replace 2D/3D numerical modelling or engineering review.
5. **Why combine physics and ML?** Physics contributes interpretable geotechnical structure; ML models patterns in multivariate synthetic telemetry.
6. **What is the target?** The four risk classes: Low, Moderate, High and Critical, encoded as 0–3 for model training.
7. **Why is the dataset synthetic?** The included pipeline generates data to support repeatable academic experimentation without claiming access to an operating mine's instrumentation.
8. **What is class imbalance?** The classes have unequal support; the provided data contains 30 Low, 211 Moderate, 405 High and 554 Critical records.
9. **Why can accuracy mislead?** A model can predict majority classes frequently and still miss rare classes. Macro-F1 gives equal weight to each class.
10. **What is macro-F1?** The arithmetic mean of per-class F1 scores.
11. **What is weighted-F1?** Per-class F1 averaged using each class's support as its weight.
12. **What does class_weight='balanced' do?** It increases the loss influence of underrepresented classes, approximately inversely to their frequencies.
13. **What is stratified cross-validation?** Each fold aims to preserve class proportions, making the evaluation less dependent on one split.
14. **Why five folds?** It provides five train/validation estimates and a mean and standard deviation while keeping the dataset usable for training.
15. **What is sector hold-out?** Training on eight sector IDs and testing on two unseen sector IDs to probe spatial/group generalization.
16. **What is data leakage?** Information unavailable at real prediction time, or target-derived information, unintentionally enters training features or preprocessing.
17. **Why investigate FoS leakage?** FoS is an input and is closely tied to the synthetic label construction. An ablation measures how performance changes when it is removed.
18. **Does ablation prove leakage?** No. It measures dependence under this experiment; causal interpretation and operational validity need independent data and a label-generation audit.
19. **What is calibration?** Agreement between predicted probabilities and observed outcome frequencies.
20. **What is the Brier score?** A probability error score based on squared differences between predicted probabilities and one-hot outcomes; lower is better.
21. **What is ROC-AUC?** A ranking metric measuring separation of positives from negatives over thresholds; multiclass results are commonly aggregated one-vs-rest.
22. **Why also use a precision–recall curve?** It is informative when the positive class is rare or classes are imbalanced.
23. **What is explainability in TALUS?** The app reports local feature perturbation contributors; the ML-rigor script also saves permutation importance and attempts SHAP artifacts when supported.
24. **Are local contributors causal?** No. They describe model sensitivity, not causal proof.
25. **What does the material classifier do?** It trains a lightweight classifier on the supplied image-feature table and extracts similar texture/color features from an uploaded image.
26. **Can image classification determine rock strength?** No. Any cohesion/friction ranges are illustrative UI hints and require geological logging and lab testing.
27. **Why FastAPI?** It provides a validated API boundary between the model/physics services and the Streamlit interface.
28. **Why SQLite?** It offers a lightweight local database for prediction history and demo alerts without a separate server.
29. **How does TALUS Voice work?** A deterministic intent parser maps supported phrases to current sector/alert data; browser speech APIs provide optional recognition and speech synthesis.
30. **Can TALUS issue real evacuation orders?** No. It is not certified safety software, has no external alert integration enabled, and all real decisions must follow approved site procedures and qualified geotechnical authority.
