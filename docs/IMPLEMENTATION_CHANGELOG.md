# TALUS Phase II + III Implementation Changelog

## Correctness / architecture
- Replaced the Streamlit local-inference fallback with an explicit FastAPI-only inference path.
- Added `TALUS_API_URL` configuration for local, Docker and cloud environments.
- Added API health, model metadata and processed-data summary endpoints.
- Changed CORS to environment-configurable origins.
- Added Pydantic request contract with schema examples.

## Data integrity
- Removed the fixed S1–S6 database seed.
- Database sectors are derived from `data/processed/cleaned_features.csv`.
- Historical observations are sourced from the processed telemetry rather than fabricated profiles.
- Added database schema versioning and indexes.

## UI
- Rebuilt the dashboard around live API data.
- Sector watchboard now renders all available sectors.
- Added EDA views for all three processed data branches.
- Added model-performance evidence and persisted model-selection explanation.
- Added what-if scenario simulation based on the selected sector's live baseline.
- Added deployment and viva guidance inside the application.

## Deployment
- Added backend/frontend Dockerfiles.
- Added Docker Compose.
- Added Render blueprint.
- Added GitHub Actions CI.

## Documentation
- Added phase rubric evidence.
- Added 7-minute demo runbook.
- Added GitHub checklist.
- Added formal project report and presentation.

## Validation
- Automated suite: **6/6 tests passing**.
- Python syntax compilation: passing.
- Dataset-driven database check: 10 sectors, 1,200 processed telemetry records.

## Extended control-room update — 2026-10-12

- Added Executive Summary, procedural 3D terrain/cross-section/PCA view, Materials Gallery and photo classification, ML Rigor, conceptual scenario playback, TALUS Voice, Alert Console, Advanced Analytics and System Health pages.
- Added `/assistant/query`, alert CRUD-lite endpoints, `/demo/critical`, `/simulate/pof`, `/sector/{id}/ttf`, `/image/classify`, `/scenario/{type}`, `/report/{sector_id}.pdf` and `/system/health-detail`.
- Added a 5-fold stratified macro-F1 benchmark and selected SVM for serving based on highest mean CV macro-F1 in the bundled synthetic dataset; updated `model.joblib` and metadata while retaining the existing preprocessor and metadata keys.
- Added FoS ablation, sector hold-out, reliability curves/Brier score, classwise metrics, permutation importance and SHAP global/local example artifacts.
- Added 30-question viva guide, five-minute runbook, endpoint smoke test, additional API tests and local SVG assets.
- Verification: `pytest -q` → 9 passed; `python scripts/smoke_test.py` → 17/17 passed; Python syntax compilation passed.
- Remaining boundaries are listed in `docs/EXTENDED_FEATURES_ADDENDUM.md` and `docs/DEMO_RUNBOOK.md`.
