# TALUS — Comprehensive 30-Question Viva Voce Q&A Master Guide

> **EXAMINER DEFENSE NOTICE**: TALUS is a physics-informed decision-support research prototype utilizing synthetic open-pit telemetry. The system demonstrates end-to-end industrial software engineering, geotechnical limit equilibrium principles, supervised machine learning rigor, explainable AI, and real-time control room UX.

---

### Category 1: Geotechnical Engineering & Physics Formulations

#### Q1: What core engineering problem does TALUS address?
**Answer:** TALUS monitors slope stability in open-pit mines. Traditional geotechnical safety relies either on manual limit-equilibrium calculations (which are static and do not adapt to continuous streaming telemetry) or black-box machine learning (which lacks physical constraints and can predict physically impossible failure states). TALUS bridges this divide using a **Physics-Informed Machine Learning (PIML)** framework, combining an analytical infinite-slope stability formulation with multiclass predictive ML, explainability, and real-time Trigger-Action-Response Plan (TARP) alerting.

#### Q2: How is the Factor of Safety (FoS) defined and mathematically formulated in TALUS?
**Answer:** Factor of Safety ($\text{FoS}$) is the ratio of available resisting shear strength $\tau_f$ along a potential slip plane to the mobilized shear driving stress $\tau_m$ induced by gravity and dynamic loads:
$$\text{FoS} = \frac{\tau_f}{\tau_m}$$
In our extended infinite-slope limit equilibrium model:
$$\text{FoS} = \frac{c' + \left[(\gamma H + q)\cos^2\beta - u\right]\tan\phi'}{(\gamma H + q)\left(\sin\beta\cos\beta + k_c\right)}$$
Where:
- $H$: Slope bench height (m)
- $\beta$: Slope angle (radians)
- $c'$: Effective rock mass cohesion (kPa)
- $\phi'$: Effective internal friction angle (radians)
- $\gamma$: Unit weight of rock/soil mass ($\text{kN/m}^3$)
- $u$: Joint pore water pressure (kPa)
- $q$: Crest equipment surcharge load (kPa)
- $k_c$: Pseudostatic blasting seismic coefficient ($k_c = 0.015 \times \text{blast\_intensity}$)

#### Q3: What do the numerical values of FoS signify operationally?
**Answer:**
- **$\text{FoS} < 1.0$ (Critical)**: Driving stress exceeds resisting shear strength. The slope is in active limit equilibrium failure or accelerating tertiary creep.
- **$1.0 \le \text{FoS} < 1.3$ (High Risk / Marginally Stable)**: Inadequate safety margin for heavy mining machinery. Sensitive to transient rainfall and blast vibrations.
- **$1.3 \le \text{FoS} < 1.5$ (Moderate Risk / Conditionally Acceptable)**: Stable under normal operating conditions; requires piezometric monitoring during monsoon events.
- **$\text{FoS} \ge 1.5$ (Low Risk / Stable)**: Standard regulatory design threshold for permanent open-pit mine slopes.

#### Q4: How does pore water pressure ($u$) physically destabilize a slope bench?
**Answer:** By Terzaghi's Principle of Effective Stress:
$$\sigma_n' = \sigma_n - u$$
Water pressure inside joints and discontinuities reduces the effective normal clamping stress $\sigma_n'$ pushing the rock blocks together. Because shear strength is governed by friction ($\tau_f = c' + \sigma_n' \tan\phi'$), elevated pore pressure directly reduces available shear resistance without reducing the gravitational driving load, causing sudden slope slumps.

#### Q5: How does pseudostatic blasting acceleration ($k_c$) affect slope equilibrium?
**Answer:** Blasting creates transient seismic shockwaves. In limit equilibrium analysis, this dynamic force is approximated pseudostatically by applying a horizontal inertial force coefficient $k_c$:
$$F_{\text{seismic}} = k_c \times W$$
This adds directly to the driving shear stress denominator $(\sin\beta\cos\beta + k_c)$, lowering FoS momentarily during blasting operations and potentially initiating dynamic joint shearing.

#### Q6: What are the primary geotechnical limitations of the infinite slope model?
**Answer:**
1. It assumes a planar translational slip surface parallel to the slope face at depth $H$.
2. It assumes homogeneous isotropic rock mass properties.
3. It does not account for complex 3D geological structural kinematics (e.g. wedge intersections, toppling along joint sets, or deep-seated circular rotational failure in soil). In industrial practice, 2D/3D numerical modeling (FLAC, Slide3, RS3) and physical borehole core logging are mandatory.

---

### Category 2: Machine Learning Rigor, Validation & Explainability

#### Q7: Why is raw classification accuracy a misleading metric for the TALUS dataset?
**Answer:** The dataset exhibits severe class imbalance:
- **Critical**: 554 records (46.2%)
- **High**: 405 records (33.8%)
- **Moderate**: 211 records (17.6%)
- **Low**: 30 records (2.5%)
A trivial, naive model predicting only the majority classes (Critical and High) would achieve **79.9% accuracy** while having **0% recall on the rare Low class**. To ensure safety-critical integrity across all hazard levels, we evaluate using **Stratified 5-Fold Macro-F1**, which weights each class equally regardless of sample frequency.

#### Q8: Is Factor of Safety "target leakage" in the machine learning model?
**Answer:** **No, it is a domain physics prior.** In Physics-Informed Machine Learning (PIML), providing analytical physical constraints prevents the ML model from violating physical laws (e.g. predicting a slope is safe when FoS is 0.5).
**The Ablation Defense:** To prove the model does not merely memorize the FoS formula, we performed an ablation study:
- **Model With FoS**: Macro-F1 = 0.794, Weighted-F1 = 0.845
- **Model Without FoS**: Macro-F1 = 0.802, Weighted-F1 = 0.788
Even when FoS is completely withheld, the model maintains strong Macro-F1 using only raw sensor features (pore pressure, deformation velocity, rainfall, blast history). This proves raw telemetry contains genuine predictive signal independently.

#### Q9: What is the significance of the Spatial Sector Hold-Out test?
**Answer:** If we only performed random train-test splits, records from the same sector could appear in both training and test sets, causing spatial data leakage. In our Sector Hold-Out test:
- **Train on Sectors S1–S8** (947 records)
- **Test on unseen Sectors S9–S10** (253 records)
The model achieved **0.8257 Macro-F1** and **86.6% accuracy** on unseen sectors, proving spatial generalization across distinct pit zones.

#### Q10: What candidate ML models were benchmarked, and why was the final model selected?
**Answer:** Four models were evaluated with Stratified 5-Fold Cross-Validation:
1. **Support Vector Machine (RBF kernel, balanced)**: Macro-F1 = **0.7998 ± 0.0366**, Weighted-F1 = 0.8571, ROC-AUC = 0.9668
2. **Random Forest (100 trees, balanced)**: Macro-F1 = 0.7943 ± 0.0109, Weighted-F1 = 0.8674, ROC-AUC = **0.9690**
3. **Logistic Regression (Multinomial, balanced)**: Macro-F1 = 0.7807 ± 0.0473, Weighted-F1 = 0.8572, ROC-AUC = 0.9520
4. **Gradient Boosting (70 estimators)**: Macro-F1 = 0.7512 ± 0.0403, Weighted-F1 = 0.8579, ROC-AUC = 0.9410
**Selection Rationale:** Support Vector Machine was selected as the primary serving model for achieving the highest mean Stratified 5-Fold Macro-F1 with balanced margin optimization. Random Forest is maintained alongside for Tree-based SHAP explainability.

#### Q11: What is probability calibration and what does the Brier Score measure?
**Answer:** Many classifiers output uncalibrated scores that do not reflect true posterior probabilities. Calibration measures whether a predicted probability of 80% corresponds to true positive occurrences 80% of the time.
The **Multiclass Brier Score** measures mean squared probability error:
$$\text{BS} = \frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$
Where $p_{ik}$ is predicted probability for class $k$ and $y_{ik} \in \{0, 1\}$ is the one-hot ground truth. TALUS achieves a multiclass Brier score of **0.2102**, with reliability curves demonstrating tight adherence to ideal calibration.

#### Q12: How does SHAP (SHapley Additive exPlanations) explain predictions?
**Answer:** Based on cooperative game theory, SHAP computes Shapley values representing each feature's marginal contribution to pushing the model output away from the base expected value.
- **SHAP Summary Beeswarm Plot**: Shows global feature importance and directional impact (e.g. high pore pressure consistently shifts predictions toward Critical risk).
- **SHAP Waterfall Plot**: Decomposes an individual sector assessment, showing step-by-step how each local sensor measurement pushed that sector into its specific risk classification.

---

### Category 3: Trigger-Action-Response Plan (TARP) & Control Room Operations

#### Q13: What is a TARP and how is it implemented in TALUS?
**Answer:** A Trigger-Action-Response Plan is a regulatory mining risk management protocol defining tiered operational responses based on sensor threshold triggers:
- **Level 0 (Normal / Green)**: FoS $\ge 1.5$, radar deformation rate $< 8$ mm/day. Standard operational monitoring.
- **Level 1 (Advisory / Yellow)**: FoS $1.3 - 1.5$, deformation rate $8 - 15$ mm/day. Conduct daily geotechnical visual inspections; review piezometer trends.
- **Level 2 (Warning / Orange)**: FoS $1.0 - 1.3$, deformation rate $15 - 30$ mm/day. Restrict heavy haul truck access; suspend blasting within 200m; relocate equipment from crest.
- **Level 3 (Evacuate / Red)**: FoS $< 1.0$, deformation rate $> 30$ mm/day, or ML Critical probability $\ge 75\%$. Sound emergency siren; immediately evacuate all personnel outside the 50m exclusion perimeter.

#### Q14: How does the worker geofence tracking feature protect personnel?
**Answer:** TALUS simulates real-time GPS coordinates of active personnel (excavator operators, haul truck drivers, survey technicians) and vehicles across pit sectors. If any worker is stationed in a sector that escalates to Level 2 or Level 3 TARP status, the system triggers a targeted **"WORKER IN DANGER ZONE"** alarm identifying the specific worker by name, vehicle, and sector, enabling immediate radio dispatch.

#### Q15: How does the Web Audio API oscillator siren operate without external audio files?
**Answer:** To guarantee 100% offline functionality without external media files or network streaming, TALUS synthesizes acoustic alerts using the browser's native HTML5 Web Audio API:
- **Level 1**: Dual-frequency harmonic chime (520 Hz $\to$ 660 Hz).
- **Level 2**: Alternating caution pulse (660 Hz $\leftrightarrow$ 880 Hz).
- **Level 3**: Continuous rising sawtooth siren ramping exponentially from 450 Hz to 950 Hz.

#### Q16: What is the 60-second unacknowledged escalation mechanism?
**Answer:** In mining control rooms, unattended critical alarms present catastrophic risk. When an alert reaches Level 3, if a geotechnical engineer does not acknowledge the alarm in the database within 60 seconds, the system logs an `UNACKNOWLEDGED ESCALATION` flag and re-triggers continuous emergency audio sirens until signed off.

---

### Category 4: Advanced Geotechnical Analytics & Modeling

#### Q17: How does the Fukuzono Inverse-Velocity Method forecast Time-to-Failure (TTF)?
**Answer:** In the tertiary creep stage prior to slope collapse, displacement velocity accelerates asymptotically. Fukuzono discovered that the inverse of deformation velocity ($1/v = 1/(dx/dt)$) decreases linearly with time:
$$\frac{1}{v} = A(t_f - t)$$
By fitting a linear regression line to historical $1/v$ sensor observations and extrapolating to the horizontal axis intercept where $1/v \to 0$ (velocity approaches infinity), TALUS estimates the projected failure window $t_f$.

#### Q18: What is the Monte Carlo Probability of Failure ($P_f$) simulation?
**Answer:** Deterministic FoS calculations use fixed mean parameter values, ignoring geotechnical variability. TALUS runs 5,000 Monte Carlo iterations, sampling cohesion, friction angle, and pore pressure from stochastic Gaussian distributions ($\mu \pm \sigma$). It computes:
- **Probability of Failure**: $P_f = P(\text{FoS} < 1.0) = \frac{N_{\text{FoS}<1}}{N_{\text{total}}}$
- **Cornell Reliability Index $\beta$**: $\beta = \frac{\mu_{\text{FoS}} - 1.0}{\sigma_{\text{FoS}}}, \quad P_f = \Phi(-\beta)$
This provides a rigorous probabilistic risk metric for risk-informed decision making.

#### Q19: What is the Tornado Sensitivity Chart and how does it prioritize remediation?
**Answer:** The Tornado chart perturbs each input parameter by $\pm 20\%$ while holding others constant, measuring the resulting delta swing in Factor of Safety $\Delta\text{FoS}$. Parameters are ranked from largest to smallest swing:
1. Slope Angle ($\beta$)
2. Pore Water Pressure ($u$)
3. Cohesion ($c'$)
4. Friction Angle ($\phi'$)
This reveals that horizontal sub-bench dewatering boreholes (reducing $u$) and bench angle re-profiling (reducing $\beta$) provide the highest return on engineering remediation effort.

#### Q20: How does TALUS evaluate data drift using the Population Stability Index (PSI)?
**Answer:** Data drift occurs when operational sensor distributions deviate from the baseline training distribution (e.g. extreme monsoon rainfall exceeding historical records). TALUS calculates PSI across feature bins:
$$\text{PSI} = \sum_{b=1}^B (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right)$$
Where $P_b$ is the recent telemetry fraction and $Q_b$ is the training baseline fraction. A $\text{PSI} < 0.10$ indicates stability; $\text{PSI} > 0.25$ indicates significant drift requiring model retraining.

---

### Category 5: Computer Vision, Voice & System Architecture

#### Q21: How does the offline rock and soil photo classifier work?
**Answer:** The user uploads a macro photo of the rock bench. Using PIL and scikit-image, TALUS extracts 14 texture and color features:
- GLCM texture contrast, homogeneity, energy, and correlation
- Shannon entropy of the luminance histogram
- Sobel edge density and gradient variance (sharpness)
- RGB and HSV color statistics
These features are evaluated by an offline Random Forest model trained on 320 geological images, predicting the material class (e.g. Basalt, Granite, Shale, Clay) and automatically pre-populating cohesion and friction angle ranges in the Analyze Sector form.

#### Q22: Why is Streamlit decoupled from the model via a FastAPI REST service?
**Answer:**
1. **Microservice Architecture**: Decoupling separates UI presentation from computationally intensive inference and database persistence.
2. **Deployability & Scalability**: The FastAPI backend can be deployed independently in Docker containers on cloud nodes or local edge servers.
3. **Integration**: External sensors, IoT data loggers, or drone photogrammetry pipelines can post telemetry directly to `/predict` without needing the Streamlit UI.
4. **Testability**: Endpoints can be unit-tested automatically using `fastapi.testclient.TestClient` and `pytest`.

#### Q23: How does the TALUS Voice Assistant operate completely offline?
**Answer:** It uses the browser's native **Web Speech API** (`SpeechRecognition` for STT and `SpeechSynthesis` for TTS). The transcript is sent via `fetch` to FastAPI's `/assistant/query` endpoint, where a deterministic regex intent parser matches queries (e.g. status of S4, why is S4 risky, what should workers do) against live SQLite database records, returning text and spoken responses with zero external LLM dependencies or paid API keys.

#### Q24: Why does TALUS support English, Tamil, and Hindi in the control room?
**Answer:** In mining regions across India (e.g. Neyveli lignite mines in Tamil Nadu, Singrauli coalfields in Madhya Pradesh), control room engineers and field workers communicate across diverse regional languages. Providing multilingual intent parsing and speech synthesis ensures critical evacuation warnings are understood immediately across language barriers.

#### Q25: How is the 3D terraced open-pit model rendered offline in Three.js?
**Answer:** The JavaScript library `three.min.js` (603 KB) is vendored locally inside `frontend/assets/`. When the user navigates to Virtual Scenarios, the Python server injects the local JavaScript code directly into an iframe HTML component, enabling GPU-accelerated 3D rendering, particle systems (rain, blast shockwaves), and camera orbit controls without requiring an active internet connection.

#### Q26: What role does SQLite play in the TALUS architecture?
**Answer:** SQLite provides ACID-compliant, single-file relational persistence (`database/talus.db`) without requiring an external database server like PostgreSQL or MySQL. It logs:
- Complete prediction audit trails with timestamps, parameters, FoS, ML labels, and XAI vectors
- 10-sector telemetry history
- Trigger-Action-Response Plan (TARP) alert logs, acknowledgment timestamps, and personnel IDs

#### Q27: How does Role-Based Access Control (RBAC) function in TALUS?
**Answer:** The control room provides three operational tiers:
- **Viewer**: Read-only visualization of sector status, 3D pit models, and historical charts.
- **Geotechnical Engineer**: Full authorization to submit sensor assessments, run what-if simulations, and generate signed PDF reports.
- **Mine Safety Administrator**: Exclusive authorization to acknowledge Level 3 TARP alarms, trigger system-wide sirens, and export audit trails.

#### Q28: How does TALUS generate PDF reports offline?
**Answer:** Using Python's `reportlab` library in `backend/main.py` (`GET /report/{sector_id}.pdf`). When requested, it programmatically formats sector metadata, physics FoS calculations, ML risk distributions, geotechnical mitigation recommendations, and synthetic prototype disclaimers into a downloadable letter-sized PDF document.

#### Q29: Can TALUS be used directly to command an active mine site today?
**Answer:** **No. TALUS is explicitly a research decision-support prototype.** Real-world mine deployment requires site-specific geotechnical borehole core calibration, certified radar QA/QC, structural discontinuity mapping, 3D finite-element verification (FLAC3D/Slide3), and regulatory sign-off by a licensed Professional Engineer (PE / R.P.Geo).

#### Q30: What is the single most critical message to convey to the examiner?
**Answer:** TALUS demonstrates that modern artificial intelligence should not operate as an unconstrained black box in safety-critical engineering domains. By grounding supervised machine learning in fundamental limit-equilibrium physics, auditing class imbalance with Stratified 5-Fold Macro-F1, explaining local failure drivers with SHAP, and delivering actionable TARP alerts in a mission-control UX, TALUS represents a complete, defensible, and rigorous engineering prototype.
