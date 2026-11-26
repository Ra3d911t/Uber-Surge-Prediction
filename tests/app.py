
"""Streamlit GUI (Uber-style): pick a pickup and drop-off area, see the predicted Lyft surge for every ride type,
a live map with moving cars, and an estimated fare.   Run from the project root:  streamlit run app.py
"""
import datetime as dt
import json
from pathlib import Path

import joblib
import streamlit as st

from src.features import LOCATIONS, PRODUCTS, WEATHER_SUMMARIES
from src.map_data import AREA_COORDS, PRODUCT_COLORS, PRODUCT_ICONS, estimate_fare, load_tables, trip_distance
from src.map_view import build_map_html
from src.predict import predict_surges

MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("reports/final_metrics.json")

st.set_page_config(page_title="Surge Ride", page_icon="🚗", layout="wide")
st.markdown("""
<style>
  
.block-container {
    padding-top: 1.4rem;
    max-width: 1250px;
}

.stApp {
    background: #ffffff;
    color: #000000;
}

[data-testid="stSidebar"] {
    background: #f5f5f5;
    border-right: 1px solid #dddddd;
}

[data-testid="stHeader"] {
    background: #ffffff;
}

h1, h2, h3, p, label, span {
    color: #000000;
}

.price-card {
    background: #f7f7f7;
    border: 1px solid #dddddd;
    border-radius: 16px;
    padding: 16px 20px;
    margin: 6px 0 4px 0;
}

.price-card .big {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.1;
    color: #000000;
}

.price-card .small {
    color: #666666;
    font-size: .85rem;
}

.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: .85rem;
    color: #ffffff;
    background: #000000 !important;
    border: 1px solid #000000;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stDateInput"] input,
div[data-testid="stTimeInput"] input {
    background: #ffffff;
    border-color: #cccccc;
    color: #000000;
}

div[data-testid="stExpander"] {
    border-color: #dddddd;
}

div[role="radiogroup"] label {
    color: #000000;
}

</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🚗 Surge Ride")
st.caption("Predicted surge multiplier for a ride in Boston. The model was trained on real Lyft quotes (Nov–Dec 2018).")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH}`. Run `notebooks/05_model_tuning.ipynb` first.")
    st.stop()
try:
    model = load_model()
    tables = load_tables()
except Exception as e:  # corrupted file / version mismatch
    st.error(f"Could not load the model: {e}")
    st.stop()

left, right = st.columns([1, 1.55], gap="large")

with left:
    st.subheader("Where to?")
    pickup = st.selectbox("Pickup", LOCATIONS, index=LOCATIONS.index("Back Bay"))
    dropoff = st.selectbox("Drop-off", LOCATIONS, index=LOCATIONS.index("South Station"))
    c1, c2 = st.columns(2)
    date = c1.date_input("Date", value=dt.date(2018, 12, 17))
    time = c2.time_input("Time", value=dt.time(18, 0), step=dt.timedelta(minutes=15))

    with st.expander("Weather"):
        w1, w2 = st.columns(2)
        summary = w1.selectbox("Conditions", WEATHER_SUMMARIES, index=WEATHER_SUMMARIES.index("Overcast"))
        temperature = w2.number_input("Temperature (°F)", min_value=-20.0, max_value=110.0, value=40.0)
        precip = w1.number_input("Rain (in/hr)", min_value=0.0, max_value=2.0, value=0.0, step=0.01, format="%.3f")
        humidity = w2.slider("Humidity", 0.0, 1.0, 0.72, 0.01)
        wind = w1.number_input("Wind (mph)", min_value=0.0, max_value=60.0, value=5.0)
        visibility = w2.number_input("Visibility (mi)", min_value=0.0, max_value=10.0, value=10.0)
        cloud = st.slider("Cloud cover", 0.0, 1.0, 0.8, 0.05)

    same_place = pickup == dropoff
    if same_place:
        st.error("Pickup and drop-off areas must be different.")
        miles, surges, product = 0.0, {}, "Lyft"
    else:
        miles = trip_distance(pickup, dropoff)
        weather = {"short_summary": summary, "temperature": temperature, "precipIntensity": precip,
                   "humidity": humidity, "windSpeed": wind, "visibility": visibility, "cloudCover": cloud}
        surges = predict_surges(model, pickup, dropoff, miles, dt.datetime.combine(date, time), weather)

        st.subheader("Choose a ride")
        product = st.radio(
            "Choose a ride", PRODUCTS, index=PRODUCTS.index("Lyft"), label_visibility="collapsed",
            format_func=lambda p: f"{PRODUCT_ICONS[p]}  {p}   ·   {surges[p]:.2f}x   ·   ${estimate_fare(p, miles, surges[p]):.2f}")

        s = surges[product]
        normal = estimate_fare(product, miles)
        est = estimate_fare(product, miles, s)
        pill = "#292929"
        st.markdown(
            f"""<div class="price-card"><div class="small">{PRODUCT_ICONS[product]} {product} · {miles:.1f} mi</div>
            <div class="big">${est:.2f}</div>
            <div class="small">normal fare ${normal:.2f} &nbsp;×&nbsp; <span class="pill" style="background:{pill}">surge {s:.2f}x</span></div></div>""",
            unsafe_allow_html=True)
        st.metric("Predicted surge", f"{s:.2f}x")
        if miles > tables["max_distance"]:
            st.warning(f"This trip is longer than any in the training data (max {tables['max_distance']:.2f} mi).")
        st.caption("Estimated fare = typical fare for this distance × predicted surge. It is an estimate, "
                   "not an official Lyft/Uber price. Distance is the typical real distance between the two areas.")

with right:
    html = build_map_html(AREA_COORDS, pickup, None if same_place else dropoff, miles,
                          surges.get(product, 1.0), product, "#000000")
    st.iframe(html, height=660)   # HTML is built only from our own constants, never from user text
    st.caption("The selected trip route follows streets when OSRM routing is available. "
               "Animated vehicles are illustrative, not live GPS. Map tiles and routing require internet access.")

if METRICS_PATH.exists():
    with st.expander("About the model"):
        m = json.loads(METRICS_PATH.read_text())
        st.write(f"**Model:** {m['model']}")
        st.write(f"**Test RMSE:** {m['test']['RMSE']:.4f} (baseline {m['baseline_test']['RMSE']:.4f}) · "
                 f"**Test R²:** {m['test']['R2']:.4f}")
        st.write("Surge in this dataset depends mostly on driver supply and rider demand, which are not "
                 "available as inputs, so predictions stay close to 1.0. See `reports/final_evaluation.md`.")
