# TALUS Extended Features Addendum

## Added to the supplied Phase II/III project

This addendum records the extra control-room work completed on top of `TALUS_PHASE2_PHASE3_COMPLETE_v2.zip`.

- Added new Streamlit destinations for Executive Summary, 3D Mine Model, Materials Gallery, ML Rigor, Virtual Scenarios, TALUS Voice, Alert Console, Advanced Analytics and System Health.
- Added a procedural Plotly 3D terraced visualization colored by the current risk snapshot; schematic cross-section, water-table reference, blast/surcharge markers, PCA feature space, response surfaces and displacement playback.
- Added deterministic `/assistant/query`, SQLite alert history/acknowledgement, an explicit synthetic S4 demo, browser tone test, `/simulate/pof`, `/physics/response-surfaces`, inverse-velocity `/sector/{id}/ttf`, image upload `/image/classify`, scenario guidance and PDF report endpoints.
- Added an offline texture/color Random Forest classifier trained from the existing `image_features.csv`; upload features are calculated from the image bytes with Pillow/scikit-image.
- Added a reproducible model-rigor benchmark and changed the serving classifier selection to Support Vector Machine because it had the highest mean stratified 5-fold macro-F1 in this run. The persisted preprocessor and existing metadata keys are retained.
- Added FoS ablation, sector hold-out, classwise report, multiclass Brier score, reliability curves, permutation importance and separate research-Random-Forest SHAP bar/beeswarm/waterfall artifacts.
- Added 30 viva questions, a five-minute runbook, API smoke tests, expanded endpoint tests and local SVG assets.

## Verification performed

- Python syntax compilation succeeded for backend, frontend and scripts.
- `pytest -q`: 9 passed.
- `python scripts/smoke_test.py`: 16/16 endpoint checks passed.
- SHAP artifact script generated global bar, beeswarm and one local waterfall example.

## Boundaries that remain intentionally honest

The geometry and scenario animations are conceptual rather than photorealistic calibrated simulations. The supplied telemetry remains synthetic. Browser STT support varies. Alerts and sirens are demo interactions, not an always-on emergency service; there is no real worker geofence or 60-second background escalation process. Monte Carlo parameter distributions are assumptions, and time-to-failure is indicative. User roles/authentication, production drift gates, validated mitigation gain and external mine-site verification remain future work. Never use this prototype as the sole basis for mine safety decisions.
