# TALUS 5-Minute Exam Demo Runbook

## Before the examiner arrives
1. Create a virtual environment and install `requirements.txt`.
2. Run `pytest -q` and `python scripts/smoke_test.py`.
3. Start `python run_backend.py`, then in a second terminal start `python run_frontend.py`.
4. Keep the disclaimer visible: the supplied telemetry is synthetic and outputs are prototype decision support.

## 0:00–0:35 — Problem & honesty
Open **Executive Summary**. Explain that TALUS brings simplified slope physics, ML, telemetry and explainability together. Explicitly state that the supplied data is synthetic and this is not a real operational warning system.

## 0:35–1:10 — 3D control-room visual
Open **3D Mine Model**. Rotate the procedural terraced terrain and point out sector markers colored by the latest prototype risk snapshot. Explain that geometry is illustrative, not surveyed terrain; inspect the slope cross-section and water-table reference.

## 1:10–1:55 — Prediction + explanation
Open **Analyze Sector**. Run an assessment for a sector and explain FoS, risk class, recommendation, and local feature contributors. Download the sector PDF report if time allows.

## 1:55–2:35 — ML rigor
Open **ML Rigor**. Show class counts, macro-F1/weighted-F1, stratified CV, FoS ablation, sector hold-out and calibration. Explain why accuracy alone can mislead. Mention that FoS is related to the synthetic label construction and requires an ablation/leakage discussion.

## 2:35–3:10 — Materials
Open **Materials Gallery**. Browse local images and, if time allows, upload a sample rock/soil photo. Explain that the lightweight model classifies texture features; it cannot replace geological sampling or lab testing.

## 3:10–3:45 — Voice + alerts
Open **TALUS Voice**, click Enable voice and ask “Which sector is most critical?” If recognition is unavailable, type the same question. Then open **Alert Console**, trigger the S4 Critical demo alert, and acknowledge it. The sound button is a local browser tone demo, not an external alert.

## 3:45–4:25 — Scenario and advanced analysis
Open **Virtual Scenarios** and play a conceptual hazard animation, then read the precaution checklist. Open **Advanced Analytics** and run the 5,000-sample Monte Carlo simulation. State that parameter variation is assumed, not calibrated from field uncertainty.

## 4:25–5:00 — Architecture & limits
Open **System & Viva** and **System Health**. Explain Streamlit → FastAPI → physics/model → SQLite. Finish by stating that external site validation, reliable sensor data, calibrated uncertainty, security and certified engineering approval are mandatory before real use.

## Run commands (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_backend.py
# second terminal
.venv\Scripts\activate
python run_frontend.py
```

## Known demo limits
- Terrain and scenario animation are conceptual visualizations, not photogrammetry, finite-element simulation or calibrated 3D mine geometry.
- Browser speech recognition may rely on browser-installed/online services; the text assistant fallback is available.
- Alert tone is user-triggered in the browser. There is no always-on 60-second escalation daemon, worker GPS/geofence integration, or external SMS/email provider.
- Probability-of-failure inputs use transparent illustrative distributions, not site-calibrated parameter uncertainty.
