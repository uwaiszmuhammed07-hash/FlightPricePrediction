import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, time

st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b3e 50%, #0a0f1e 100%); }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.hero { text-align: center; padding: 2.5rem 0 1.5rem 0; }
.hero-tag { display: inline-block; background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3); color: #38bdf8; font-size: 11px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; padding: 5px 16px; border-radius: 20px; margin-bottom: 16px; }
.hero-title { font-size: 2.8rem; font-weight: 700; color: #f0f6ff; line-height: 1.15; margin-bottom: 10px; letter-spacing: -0.02em; }
.hero-title span { background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { font-size: 1rem; color: #94a3b8; font-weight: 300; }
.pred-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 2rem; margin: 1.5rem 0; }
.section-label { font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; margin-bottom: 12px; margin-top: 8px; }
.result-box { background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(129,140,248,0.12)); border: 1px solid rgba(56,189,248,0.35); border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1.5rem; }
.result-label { font-size: 12px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #38bdf8; margin-bottom: 8px; }
.result-price { font-size: 3.2rem; font-weight: 700; color: #f0f6ff; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em; }
.result-range { font-size: 13px; color: #94a3b8; margin-top: 8px; }
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.5rem 0; }
.pill-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; justify-content: center; }
.pill { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 5px 14px; font-size: 12px; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
.pill span { color: #e2e8f0; font-weight: 500; }
.stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 10px !important; color: #e2e8f0 !important; }
label { color: #94a3b8 !important; font-size: 13px !important; font-weight: 400 !important; }
.stButton > button { background: linear-gradient(90deg, #38bdf8, #818cf8) !important; color: #0a0f1e !important; font-weight: 700 !important; font-size: 15px !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important; width: 100% !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-tag">✈ PRCP-1025 · ML Project</div>
    <div class="hero-title">Flight Price<br><span>Predictor</span></div>
    <div class="hero-sub">Enter your flight details and get an instant AI-powered price estimate</div>
</div>
""", unsafe_allow_html=True)

AIRLINES     = ['Air Asia','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy']
SOURCES      = ['Banglore','Chennai','Delhi','Kolkata','Mumbai']
DESTINATIONS = ['Banglore','Cochin','Delhi','Hyderabad','Kolkata','New Delhi']
STOPS        = ['non-stop','1 stop','2 stops','3 stops','4 stops']

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

model = load_model()

def rule_based_estimate(airline, stops_str, dep_hour, duration_minutes, journey_month):
    airline_premium = {
        'Jet Airways Business': 25000, 'Multiple carriers Premium economy': 20000,
        'Vistara Premium economy': 18000, 'Vistara': 8000, 'Air India': 6500,
        'Jet Airways': 6000, 'Multiple carriers': 5500, 'Air Asia': 4200,
        'GoAir': 3800, 'IndiGo': 3500, 'SpiceJet': 3200, 'Trujet': 3000,
    }
    stops_add  = {'non-stop': 0, '1 stop': 800, '2 stops': 1800, '3 stops': 2800, '4 stops': 4000}
    duration_add  = max(0, (duration_minutes - 120) * 8)
    month_factor  = 1.2 if journey_month in [3,4,5,10,11,12] else 1.0
    peak_add      = 600 if (6 <= dep_hour <= 9 or 17 <= dep_hour <= 20) else 0
    price = (airline_premium.get(airline, 4500) + stops_add.get(stops_str, 0) + duration_add + peak_add)
    price *= month_factor
    return max(1500, int(price + np.random.uniform(-300, 300)))

def build_features(airline, source, destination, stops_str, journey_day, journey_month,
                   dep_hour, dep_min, arrival_hour, arrival_min, duration_minutes):
    stops_map = {'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}
    base = {
        'Total_Stops': stops_map[stops_str], 'Journey_Day': journey_day,
        'Journey_Month': journey_month, 'Dep_Hour': dep_hour, 'Dep_Min': dep_min,
        'Arrival_Hour': arrival_hour, 'Arrival_Min': arrival_min,
        'Duration_minutes': duration_minutes,
        'is_short_flight': int(duration_minutes < 120),
        'is_long_flight': int(duration_minutes > 360),
        'is_peak_departure': int((6<=dep_hour<=9) or (17<=dep_hour<=20)),
        'is_late_night': int(dep_hour >= 23 or dep_hour <= 5),
        'is_weekend': int(journey_day % 7 in [4,5,6]),
    }
    for a in AIRLINES[1:]:
        base[f'Airline_{a}'] = int(airline == a)
    for s in SOURCES[1:]:
        base[f'Source_{s}'] = int(source == s)
    for d in DESTINATIONS[1:]:
        base[f'Destination_{d}'] = int(destination == d)
    return pd.DataFrame([base])

st.markdown('<div class="pred-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Flight Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox("Airline", AIRLINES)
    source  = st.selectbox("Source City", SOURCES)
with col2:
    destination = st.selectbox("Destination City", DESTINATIONS)
    stops       = st.selectbox("Total Stops", STOPS)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Date & Time</div>', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    journey_date = st.date_input("Date of Journey", value=date(2025, 5, 15))
with col4:
    dep_time = st.time_input("Departure Time", value=time(8, 0))
with col5:
    arr_time = st.time_input("Arrival Time",   value=time(10, 30))

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Flight Duration</div>', unsafe_allow_html=True)
col6, col7 = st.columns(2)
with col6:
    dur_hours = st.number_input("Duration Hours",   min_value=0, max_value=24, value=2)
with col7:
    dur_mins  = st.number_input("Duration Minutes", min_value=0, max_value=59, value=30)
st.markdown("</div>", unsafe_allow_html=True)

if st.button("✈  Predict Flight Price"):
    duration_minutes = dur_hours * 60 + dur_mins
    if model is not None:
        try:
            features  = build_features(airline, source, destination, stops,
                                       journey_date.day, journey_date.month,
                                       dep_time.hour, dep_time.minute,
                                       arr_time.hour, arr_time.minute, duration_minutes)
            predicted = int(model.predict(features)[0])
        except Exception:
            predicted = rule_based_estimate(airline, stops, dep_time.hour, duration_minutes, journey_date.month)
    else:
        predicted = rule_based_estimate(airline, stops, dep_time.hour, duration_minutes, journey_date.month)

    low  = max(1000, int(predicted * 0.88))
    high = int(predicted * 1.12)
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Ticket Price</div>
        <div class="result-price">\u20b9{predicted:,}</div>
        <div class="result-range">Likely range: \u20b9{low:,} — \u20b9{high:,}</div>
        <div class="pill-row">
            <div class="pill">✈ <span>{airline}</span></div>
            <div class="pill">📍 <span>{source} → {destination}</span></div>
            <div class="pill">🛑 <span>{stops}</span></div>
            <div class="pill">⏱ <span>{dur_hours}h {dur_mins}m</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#334155;font-size:12px;">
    PRCP-1025 · Flight Price Prediction · Built with Streamlit
</div>
""", unsafe_allow_html=True)
