# SafeSphere AI — Streamlit MVP

**"From Early Warning to Safe Action"** — SIH26178 prototype.

This is a single-app Streamlit version of SafeSphere AI: Environmental Data → AI Risk
Prediction (RandomForest) → Dynamic Risk Map (Folium) → Safe-Zone Recommendation →
Risk-Aware Route, all running in one process for fast, dependency-light demoing.

## 1. Setup

```bash
# (recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 2. Generate the dataset

```bash
python generate_data.py
```
Creates `flood_dataset.csv` — 1200 synthetic rows with realistic correlations between
rainfall, water level, flow rate, humidity, elevation, historical-flood flag, and risk.

## 3. Train the model

```bash
python train_model.py
```
This runs the full pipeline — load → preprocess → train/test split → train
RandomForestClassifier → evaluate → save — and prints accuracy, precision, recall,
F1, confusion matrix, and feature importances to the terminal. Saves `flood_model.pkl`.

## 4. Run the app

```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Demo Flow (2–3 min)

1. Dashboard loads with current simulated environmental data.
2. Hazard = Flood is selected by default.
3. Adjust the Rainfall / Water Level sliders in the sidebar.
4. Click **Predict Risk**.
5. The app calls the trained RandomForest model directly (in-process — no separate
   backend needed for this version) and returns risk score + level.
6. Dashboard, hazard cards, and risk map update.
7. Map shows the risk zone, ranked safe-zone candidates, and the recommended zone.
8. A risk-aware route (prototype straight-line paths, distance + exposure scored) is drawn.
9. Toggle **Emergency Mode** to see the compact emergency view.
10. Re-adjust sliders and re-predict to show the live update loop.

## Architecture note (vs. the full React/Node/Flask design)

This Streamlit build folds the frontend, backend, and ML service into one Python
process for a fast, single-command demo. The underlying logic is identical to the
full architecture:

- `generate_data.py` — synthetic dataset generation
- `train_model.py` — the real ML pipeline (equivalent to `ml-service/train.py`)
- `logic.py` — safe-zone ranking + risk-aware routing (equivalent to backend services)
- `app.py` — dashboard + map + simulation UI (equivalent to frontend + API glue)

If judges ask "why not React/Node/Flask?" — the honest answer: same architecture and
same ML pipeline, just consolidated into one app for demo speed. The model training
code, feature engineering, and evaluation metrics are unchanged and portable to the
Flask `/predict` endpoint described in the original architecture doc.

## Technical Questions Judges May Ask

**Why RandomForest?**
Handles a small/synthetic tabular dataset well without heavy tuning, is robust to
noise, and gives free feature-importance scores — used directly in the Explainable
AI panel. It's also easy to explain to non-ML judges.

**Classification or regression?**
Classification — predicting one of LOW / MODERATE / HIGH. A separate regression
model could predict the raw risk_score (0–100) directly; the app currently derives
a continuous score from class probabilities as a practical middle ground.

**Train/test split?**
80/20, stratified by class so LOW/MODERATE/HIGH are proportionally represented in
both sets.

**Overfitting risk?**
`max_depth=8` and evaluating on a held-out 20% test set guards against this. With
only ~1200 synthetic rows, the model is explicitly labeled a prototype, not
production-grade.

**False positives/negatives?**
A false LOW (missed HIGH risk) is more dangerous than a false HIGH (extra caution).
`class_weight="balanced"` partially compensates for this; a production system would
weight the loss function explicitly toward recall on the HIGH class.

**Why is HIGH-risk precision better than recall in the report?**
HIGH is the minority class (~57 of 1200 rows) — model correctly flags most true
HIGH cases but with real-world data collection, that recall would need improving
before deployment.

**How is safe-zone ranking calculated?**
Composite score = 0.6 × adjusted hazard risk + 0.35 × distance penalty − 0.05 ×
capacity bonus. Adjusted risk scales up with the current event's severity. This is
why the nearest zone isn't always recommended — see `logic.py: rank_safe_zones()`.

**How is route risk calculated?**
Two candidate paths (direct vs. detour) are scored on 0.3 × distance + 0.7 ×
simulated hazard exposure; the lower-scoring path is selected. This is explicitly a
prototype — not a real road-network routing engine (no OSRM/GraphHopper integration
yet). See `logic.py: build_route()`.

**How would sensors be integrated later?**
`generate_data.py`'s synthetic feature set (rainfall, water_level, flow_rate, etc.)
maps directly onto real IoT sensor channels (water-level sensors, flow sensors,
LoRa/GSM telemetry). Swapping the CSV/slider inputs for a live sensor-data ingestion
endpoint requires no change to the trained model's feature schema.

**Do you claim high real-world accuracy?**
No. All UI copy uses "predicted," "estimated," "relatively safer," and "prototype"
language deliberately, per the honesty requirement for this MVP.

## Known limitations (by design, for MVP scope)

- Forest Fire and Pollution hazard cards use simulated demo data (UI/architecture
  ready, no trained model yet).
- Route prototype uses straight-line/offset paths, not real road/terrain routing.
- No live sensor integration — dataset and simulation sliders are the data source.
- Single-process app; the full production architecture (React/Node/MongoDB/Flask)
  is documented separately for scaling beyond MVP.
