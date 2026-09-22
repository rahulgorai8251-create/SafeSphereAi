"""
SafeSphere AI - Streamlit MVP
"From Early Warning to Safe Action"

Run with: streamlit run app.py
"""
import time
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from logic import rank_safe_zones, build_route, CANDIDATE_ZONES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SafeSphere AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_LAT, BASE_LNG = 22.5726, 88.3639  # demo base location (Kolkata area)

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load("flood_model.pkl")
    return bundle

try:
    bundle = load_model()
    MODEL = bundle["model"]
    LE = bundle["label_encoder"]
    FEATURES = bundle["features"]
    METRICS = bundle["metrics"]
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "emergency_mode" not in st.session_state:
    st.session_state.emergency_mode = False

# ---------------------------------------------------------------------------
# Sidebar - controls / simulation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌊 SafeSphere AI")
    st.caption("Environmental Intelligence + Disaster Safety — Prototype MVP")

    st.markdown("---")
    hazard = st.selectbox("Select Hazard", ["Flood", "Forest Fire", "Pollution"])

    if hazard != "Flood":
        st.info(f"{hazard} module: architecture in place, using simulated demo data. "
                f"Full ML pipeline currently implemented for Flood only.")

    st.markdown("### 🎛️ Disaster Simulation")
    rainfall = st.slider("Rainfall (mm)", 0, 300, 120)
    water_level = st.slider("Water Level (m)", 0.2, 8.0, 4.2)
    flow_rate = st.slider("Flow Rate (m³/s)", 5, 200, 85)
    temperature = st.slider("Temperature (°C)", 15, 42, 29)
    humidity = st.slider("Humidity (%)", 30, 100, 88)
    elevation = st.slider("Elevation (m)", 0, 200, 20)
    historical_flag = st.checkbox("Historical flood indicator", value=True)

    predict_clicked = st.button("🔮 Predict Risk", type="primary", use_container_width=True)

    st.markdown("---")
    st.session_state.emergency_mode = st.toggle("🚨 Emergency Mode", value=st.session_state.emergency_mode)

    st.markdown("---")
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
    if MODEL_LOADED:
        st.caption(f"Model: RandomForest (prototype) · Acc {METRICS['accuracy']:.2f}")
    else:
        st.error("Model not found. Run generate_data.py + train_model.py first.")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_flood_risk(rainfall, water_level, flow_rate, temperature, humidity, elevation, hist_flag):
    X = pd.DataFrame([{
        "rainfall": rainfall, "water_level": water_level, "flow_rate": flow_rate,
        "temperature": temperature, "humidity": humidity, "elevation": elevation,
        "historical_flood_flag": int(hist_flag),
    }])[FEATURES]
    pred_encoded = MODEL.predict(X)[0]
    pred_level = LE.inverse_transform([pred_encoded])[0]
    proba = MODEL.predict_proba(X)[0]
    # risk score = weighted probability toward HIGH end, scaled 0-100
    classes = list(LE.classes_)
    score_map = {"LOW": 20, "MODERATE": 55, "HIGH": 90}
    risk_score = sum(p * score_map.get(c, 50) for p, c in zip(proba, classes))
    return pred_level, round(risk_score, 1), dict(zip(FEATURES, MODEL.feature_importances_))


if MODEL_LOADED and (predict_clicked or not st.session_state.history):
    risk_level, risk_score, importances = predict_flood_risk(
        rainfall, water_level, flow_rate, temperature, humidity, elevation, historical_flag
    )
    st.session_state.history.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "risk_score": risk_score, "risk_level": risk_level,
    })
    st.session_state.last = {
        "risk_level": risk_level, "risk_score": risk_score, "importances": importances,
        "inputs": {"rainfall": rainfall, "water_level": water_level, "flow_rate": flow_rate,
                   "humidity": humidity},
    }
elif MODEL_LOADED and "last" not in st.session_state:
    risk_level, risk_score, importances = predict_flood_risk(
        rainfall, water_level, flow_rate, temperature, humidity, elevation, historical_flag
    )
    st.session_state.last = {"risk_level": risk_level, "risk_score": risk_score,
                              "importances": importances,
                              "inputs": {"rainfall": rainfall, "water_level": water_level,
                                         "flow_rate": flow_rate, "humidity": humidity}}

if not MODEL_LOADED:
    st.error("⚠️ Model file `flood_model.pkl` not found. Run `python generate_data.py` "
             "then `python train_model.py` in this folder before launching the app.")
    st.stop()

last = st.session_state.last
risk_level, risk_score = last["risk_level"], last["risk_score"]

RISK_COLORS = {"LOW": "#2ecc71", "MODERATE": "#f39c12", "HIGH": "#e74c3c"}
risk_color = RISK_COLORS[risk_level]

# ---------------------------------------------------------------------------
# EMERGENCY MODE VIEW
# ---------------------------------------------------------------------------
if st.session_state.emergency_mode:
    st.markdown(
        f"<div style='background:{risk_color};padding:18px;border-radius:10px;color:white;'>"
        f"<h2 style='margin:0;'>🚨 EMERGENCY MODE — {hazard.upper()}</h2>"
        f"<p style='margin:4px 0 0 0;font-size:18px;'>Risk Level: <b>{risk_level}</b> "
        f"&nbsp;|&nbsp; Risk Score: <b>{risk_score}</b>/100</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("### 📋 Emergency Instructions")
    st.markdown("""
    - Move to higher ground immediately if in a flood-prone area.
    - Avoid crossing flowing water on foot or by vehicle.
    - Keep emergency kit (water, torch, documents, medicines) ready.
    - Follow the recommended safe route below — do not take shortcuts through low-lying roads.
    """)

zones, best_zone, reason = rank_safe_zones(BASE_LAT, BASE_LNG, risk_score)
routes, chosen_route_key = build_route(BASE_LAT, BASE_LNG, best_zone["lat"], best_zone["lng"], risk_score)
chosen_route = routes[chosen_route_key]

if st.session_state.emergency_mode:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🏠 Nearest Recommended Safe Zone**")
        st.markdown(f"### {best_zone['name']}")
        st.caption(f"{best_zone['distance_km']} km away · Risk: {best_zone['risk_level']}")
    with c2:
        st.markdown("**🛣️ Route**")
        st.markdown(f"### {chosen_route_key.title()} path selected")
        st.caption(f"{chosen_route['distance_km']} km · exposure score {chosen_route['exposure_score']}")
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")

# ---------------------------------------------------------------------------
# MAIN DASHBOARD HEADER
# ---------------------------------------------------------------------------
st.markdown("# 🌊 SafeSphere AI")
st.caption("**From Early Warning to Safe Action** · System Status: 🟢 Operational (prototype data)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Hazard", hazard)
col2.metric("Risk Level", risk_level)
col3.metric("Risk Score", f"{risk_score}/100")
col4.metric("Last Updated", datetime.datetime.now().strftime("%H:%M:%S"))

# ---------------------------------------------------------------------------
# HAZARD CARDS
# ---------------------------------------------------------------------------
st.markdown("### Hazard Overview")
hc1, hc2, hc3 = st.columns(3)
hazard_cards = {
    "Flood": (risk_level, risk_score, True),
    "Forest Fire": ("LOW", 18.0, False),
    "Pollution": ("MODERATE", 48.0, False),
}
for col, (name, (lvl, score, real)) in zip([hc1, hc2, hc3], hazard_cards.items()):
    with col:
        badge = "🔴" if lvl == "HIGH" else "🟠" if lvl == "MODERATE" else "🟢"
        tag = "(live ML model)" if real else "(simulated demo data)"
        st.markdown(f"**{badge} {name}**")
        st.markdown(f"Risk: **{lvl}** · Score: **{score}**")
        st.caption(tag)

st.markdown("---")

# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------
st.markdown("### 🗺️ Dynamic Risk Map")
m = folium.Map(location=[BASE_LAT, BASE_LNG], zoom_start=12, tiles="CartoDB positron")

# Current location / hazard center
folium.Marker(
    [BASE_LAT, BASE_LNG],
    popup="Current Location (Hazard Center)",
    icon=folium.Icon(color="red" if risk_level == "HIGH" else "orange" if risk_level == "MODERATE" else "green",
                      icon="home"),
).add_to(m)

# Risk zone circle around hazard center, radius scales with score
folium.Circle(
    [BASE_LAT, BASE_LNG],
    radius=300 + risk_score * 25,
    color=risk_color, fill=True, fill_opacity=0.25, weight=2,
    popup=f"{hazard} risk zone — {risk_level}",
).add_to(m)

# Safe zone markers
for z in zones:
    color = "green" if z["risk_level"] == "LOW" else "orange" if z["risk_level"] == "MODERATE" else "red"
    is_best = z["name"] == best_zone["name"]
    folium.Marker(
        [z["lat"], z["lng"]],
        popup=f"{z['name']} — {z['risk_level']} risk, {z['distance_km']} km"
              + (" ⭐ RECOMMENDED" if is_best else ""),
        icon=folium.Icon(color=color, icon="star" if is_best else "flag"),
    ).add_to(m)

# Route
route_color = "#3498db" if chosen_route_key == "detour" else "#9b59b6"
folium.PolyLine(chosen_route["path"], color=route_color, weight=4, opacity=0.8,
                 tooltip=f"Risk-aware route ({chosen_route_key})").add_to(m)

st_folium(m, width=None, height=480, returned_objects=[])

st.caption("🔴 High risk zone · ⭐ Recommended safe zone · 🔵 Risk-aware route (prototype — "
           "straight-line demo paths, not a real road-network routing engine)")

st.markdown("---")

# ---------------------------------------------------------------------------
# SAFE ZONE + ROUTE SECTIONS
# ---------------------------------------------------------------------------
sc1, sc2 = st.columns(2)

with sc1:
    st.markdown("### 🏠 Safe-Zone Recommendation")
    st.success(f"**Recommended: {best_zone['name']}**")
    st.write(f"Risk: **{best_zone['risk_level']}** ({best_zone['risk_score']}) · "
             f"Distance: **{best_zone['distance_km']} km** · Capacity: {best_zone['capacity']}")
    st.caption(f"Reason: {reason}")
    with st.expander("See all candidate zones (ranked)"):
        df_zones = pd.DataFrame(zones)[["name", "risk_level", "risk_score", "distance_km", "composite_score"]]
        st.dataframe(df_zones, hide_index=True, use_container_width=True)

with sc2:
    st.markdown("### 🛣️ Risk-Aware Route (prototype)")
    st.success(f"**Selected: {chosen_route_key.title()} path** to {best_zone['name']}")
    st.write(f"Distance: **{chosen_route['distance_km']} km** · "
             f"Estimated exposure: **{chosen_route['exposure_score']}**")
    other_key = "direct" if chosen_route_key == "detour" else "detour"
    other = routes[other_key]
    st.caption(f"Compared against {other_key} path ({other['distance_km']} km, "
               f"exposure {other['exposure_score']}) — chosen route has lower overall risk score.")
    st.caption("⚠️ Prototype only: does not use a real road-network routing engine yet.")

st.markdown("---")

# ---------------------------------------------------------------------------
# EXPLAINABLE AI
# ---------------------------------------------------------------------------
st.markdown("### 🧠 Why is the risk " + risk_level + "?")
imp_df = pd.DataFrame(
    sorted(last["importances"].items(), key=lambda x: -x[1]),
    columns=["Feature", "Model Importance"]
)
ec1, ec2 = st.columns([1, 1])
with ec1:
    st.bar_chart(imp_df.set_index("Feature"))
with ec2:
    st.markdown("**Top contributing factors (RandomForest feature importance):**")
    for _, row in imp_df.head(4).iterrows():
        st.markdown(f"- {row['Feature'].replace('_', ' ').title()} — importance {row['Model Importance']:.2f}")
    st.caption("These are model feature-importance values, not claims of exact real-world causality.")

st.markdown("---")

# ---------------------------------------------------------------------------
# MODEL / TECHNICAL PANEL
# ---------------------------------------------------------------------------
with st.expander("📊 Model Details & Evaluation Metrics (for technical judges)"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{METRICS['accuracy']:.2f}")
    m2.metric("Precision", f"{METRICS['precision']:.2f}")
    m3.metric("Recall", f"{METRICS['recall']:.2f}")
    m4.metric("F1-score", f"{METRICS['f1']:.2f}")
    st.caption("Algorithm: RandomForestClassifier (150 trees, max_depth=8), trained on a synthetic "
               "1200-row prototype dataset with an 80/20 train/test split. This is a demo-scale "
               "model, not a production-accuracy system.")

with st.expander("📈 Prediction History (this session)"):
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.line_chart(hist_df.set_index("time")["risk_score"])
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No predictions yet.")

st.markdown("---")
st.caption("SafeSphere AI — Prototype MVP for SIH26178. Predictions are estimated risk from a "
           "prototype model on simulated data. This system does not guarantee safety and routes "
           "are not guaranteed to be the safest possible path.")
