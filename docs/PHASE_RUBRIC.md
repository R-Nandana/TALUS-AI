# Phase II / III Rubric Evidence

| Rubric | Evidence in TALUS |
|---|---|
| UI + backend connectivity | `frontend/app.py` uses REST calls to FastAPI; no silent local inference fallback |
| Preprocessing + EDA | `scripts/preprocessing.py`, processed CSVs, executed EDA notebook, Data Intelligence page |
| Model building | `notebooks/02_model_training.ipynb`, persisted joblib model, `results/model_comparison.csv` |
| Result analysis + business insights | risk watchboard, XAI contributors, recommendations, timeline, model comparison |
| Deployment + demo | `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`, `render.yaml`, demo runbook |
| Presentation + viva | `docs/DEMO_RUNBOOK.md`, generated presentation, System & Viva page |
| Report + GitHub | generated report, `docs/GITHUB_CHECKLIST.md`, README deployment instructions |
