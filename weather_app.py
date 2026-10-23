import streamlit as st
import numpy as np
import pickle
import pandas as pd
import calendar
from datetime import datetime

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="WeatherMind AI",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_bundle():
    with open("weather_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["features"]   # the 9 features, in the order the model expects
cap_bounds = bundle["cap_bounds"]      # outlier bounds learned on the train set
classes = bundle["classes"]            # weather category names

# ======================================================
# GLOBAL CSS — Dark glassmorphism + cinematic design
# ======================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">

<style>
/* ── ROOT VARIABLES ── */
:root {
    --glass: rgba(255,255,255,0.06);
    --glass-border: rgba(255,255,255,0.12);
    --accent: #38bdf8;
    --accent2: #818cf8;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --card-radius: 20px;
    --font: 'Outfit', sans-serif;
    --mono: 'JetBrains Mono', monospace;
}

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: var(--font) !important;
    background: #020c1b !important;
    color: var(--text) !important;
    min-height: 100vh;
}

/* Animated star-field background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(56,189,248,0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(129,140,248,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 60% 80%, rgba(34,211,238,0.06) 0%, transparent 40%),
        #020c1b;
    pointer-events: none;
    z-index: 0;
}

/* ── HIDE STREAMLIT JUNK ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
    position: relative;
    z-index: 1;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: rgba(2,12,27,0.95) !important;
    border-right: 1px solid var(--glass-border) !important;
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] > div {
    padding: 2rem 1.5rem;
}

/* Sidebar label */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label {
    font-family: var(--font) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* Sidebar inputs — handled by nuclear fix below */
section[data-testid="stSidebar"] input {
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    width: 100% !important;
    padding: 1rem 2rem !important;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: var(--font) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 14px !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 0 30px rgba(56,189,248,0.3) !important;
    margin-top: 1rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 50px rgba(56,189,248,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── GLASS CARD ── */
.glass-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: var(--card-radius);
    padding: 1.8rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: transform 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}

/* ── METRIC CARD ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: all 0.3s;
}
.metric-card:hover {
    background: rgba(56,189,248,0.08);
    border-color: rgba(56,189,248,0.3);
    transform: translateY(-3px);
}
.metric-value {
    font-family: var(--mono) !important;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}
.metric-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }

/* ── RESULT BANNER ── */
.result-banner {
    background: linear-gradient(135deg,
        rgba(56,189,248,0.15),
        rgba(129,140,248,0.15));
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.6s ease forwards;
}
.result-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%,
        rgba(56,189,248,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.result-emoji { font-size: 5rem; display: block; margin-bottom: 0.5rem; }
.result-weather {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #fff, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.result-confidence {
    font-family: var(--mono);
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* ── CONFIDENCE BAR ── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 10px;
    margin: 1rem 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 1s ease;
    box-shadow: 0 0 12px rgba(56,189,248,0.6);
}

/* ── BADGE ── */
.badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-sky { background: rgba(56,189,248,0.15); color: var(--accent); border: 1px solid rgba(56,189,248,0.3); }
.badge-purple { background: rgba(129,140,248,0.15); color: var(--accent2); border: 1px solid rgba(129,140,248,0.3); }
.badge-green { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-amber { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

/* ── PROBA TABLE ── */
.proba-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.7rem;
}
.proba-name {
    width: 120px;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
    flex-shrink: 0;
}
.proba-bar-wrap {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.proba-bar { height: 100%; border-radius: 999px; }
.proba-pct {
    width: 45px;
    text-align: right;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--muted);
    flex-shrink: 0;
}

/* ── DIVIDER ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--glass-border), transparent);
    margin: 1.5rem 0;
}

/* ── SECTION TITLE ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── ANIMATION ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 20px rgba(56,189,248,0.3); }
    50%      { box-shadow: 0 0 40px rgba(56,189,248,0.6); }
}

/* ── STREAMLIT OVERRIDES ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
div[data-testid="stInfo"] {
    background: rgba(56,189,248,0.08) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

/* ══════════════════════════════════════════
   NUCLEAR INPUT FIX — forces ALL inputs dark
   ══════════════════════════════════════════ */

/* Every BaseWeb input container */
div[data-baseweb="input"],
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="base-input"] > div {
    background: #0d1f35 !important;
    background-color: #0d1f35 !important;
    border-color: rgba(56,189,248,0.25) !important;
    border-radius: 12px !important;
}

/* The actual <input> element */
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="date"],
section[data-testid="stSidebar"] input[type="time"],
input[class*="st-"] {
    background: #0d1f35 !important;
    background-color: #0d1f35 !important;
    color: #e2e8f0 !important;
    caret-color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}

/* Autofill override (Chrome paints inputs yellow) */
div[data-baseweb="input"] input:-webkit-autofill,
div[data-baseweb="base-input"] input:-webkit-autofill {
    -webkit-box-shadow: 0 0 0 1000px #0d1f35 inset !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}

/* Focus glow */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="base-input"]:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.18), 0 0 20px rgba(56,189,248,0.12) !important;
}

/* Step buttons (+ / −) */
div[data-baseweb="input"] button,
div[data-baseweb="base-input"] button {
    background: transparent !important;
    color: #38bdf8 !important;
    border: none !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    transition: color 0.2s, transform 0.15s;
}
div[data-baseweb="input"] button:hover,
div[data-baseweb="base-input"] button:hover {
    color: #fff !important;
    transform: scale(1.2);
}

/* Selectbox dark fix */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] div[class*="ValueContainer"],
div[data-baseweb="select"] div[class*="control"] {
    background: #0d1f35 !important;
    background-color: #0d1f35 !important;
    border-color: rgba(56,189,248,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #e2e8f0 !important;
}

/* Dropdown menu */
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] {
    background: #0d1f35 !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
}
div[data-baseweb="menu"] li:hover {
    background: rgba(56,189,248,0.12) !important;
}
div[data-baseweb="menu"] li {
    color: #e2e8f0 !important;
}

/* Dropdown options (Precipitation Type, Month): explicit light text on the dark menu */
div[data-baseweb="popover"] ul[role="listbox"],
div[data-baseweb="popover"] div[data-baseweb="menu"] {
    background: #0d1f35 !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [role="option"] {
    background: transparent !important;
    color: #e2e8f0 !important;
}
div[data-baseweb="popover"] li *,
div[data-baseweb="popover"] [role="option"] * {
    color: #e2e8f0 !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background: rgba(56,189,248,0.22) !important;
}
div[data-baseweb="popover"] li:hover *,
div[data-baseweb="popover"] [role="option"]:hover *,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] * {
    color: #ffffff !important;
}

/* Date / time inputs */
input[type="date"],
input[type="time"] {
    background: #0d1f35 !important;
    color: #e2e8f0 !important;
    color-scheme: dark;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 12px !important;
    padding: 0.5rem 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Streamlit's own number_input wrapper */
.stNumberInput > div > div,
.stTextInput > div > div,
.stDateInput > div,
.stTimeInput > div {
    background: #0d1f35 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
}
.stNumberInput input,
.stTextInput input {
    background: #0d1f35 !important;
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# ANIMATED HEADER
# ======================================================
st.markdown("""
<div style="text-align:center; padding: 2.5rem 0 1.5rem;">
    <div style="
        display:inline-block;
        background: linear-gradient(135deg,rgba(56,189,248,0.1),rgba(129,140,248,0.1));
        border:1px solid rgba(56,189,248,0.2);
        border-radius:999px;
        padding:0.4rem 1.2rem;
        font-size:0.72rem;
        font-weight:700;
        letter-spacing:0.14em;
        text-transform:uppercase;
        color:#38bdf8;
        margin-bottom:1.2rem;
    ">⚡ Powered by Machine Learning</div>
    <h1 style="
        font-family:'Outfit',sans-serif;
        font-size:3.8rem;
        font-weight:900;
        letter-spacing:-0.03em;
        line-height:1.1;
        background: linear-gradient(135deg,#fff 30%,#38bdf8 70%,#818cf8);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        margin-bottom:0.6rem;
    ">WeatherMind AI</h1>
    <p style="
        font-size:1.1rem;
        color:#94a3b8;
        font-weight:300;
        letter-spacing:0.02em;
        max-width:500px;
        margin:0 auto;
    ">Intelligent atmospheric prediction using end-to-end ML pipeline</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR INPUTS
# ======================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:2rem;">
        <div style="font-size:2.5rem;">🌡️</div>
        <div style="font-family:'Outfit',sans-serif; font-size:1.1rem; font-weight:700; color:white; margin-top:0.4rem;">Input Parameters</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:0.2rem;">Configure your atmospheric data</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">🌍 Atmospheric Data</p>', unsafe_allow_html=True)
    temperature = st.number_input("🌡 Temperature (°C)", value=15.0, step=0.5)
    apparent_temp = st.number_input("🥵 Apparent Temperature (°C)", value=15.0, step=0.5)
    humidity = st.number_input("💧 Humidity (%)", value=70.0, min_value=0.0, max_value=100.0, step=1.0)
    wind_speed = st.number_input("🌬 Wind Speed (km/h)", value=10.0, min_value=0.0, step=0.5)
    wind_bearing = st.number_input("🧭 Wind Direction (degrees)", value=180, min_value=0, max_value=360, step=10)
    visibility = st.number_input("👁 Visibility (km)", value=10.0, min_value=0.0, step=0.1)
    pressure = st.number_input("🔵 Pressure (mb)", value=1015.0, step=0.5)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🌧 Precipitation & Month</p>', unsafe_allow_html=True)

    precip_type = st.selectbox(
        "Precipitation Type", ["None", "Rain", "Snow"],
        help="The training data has only rain and snow (missing values were filled with rain), so None and Rain are treated the same.")
    month = st.selectbox("📅 Month", list(range(1, 13)), index=datetime.now().month - 1,
                         format_func=lambda m: calendar.month_name[m])

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    predict_btn = st.button("🚀 Run Prediction")

# ======================================================
# FEATURE ENGINEERING
# ======================================================
# Season is only shown on screen, the model uses Month
if month in [12, 1, 2]:
    season_name = "Winter"; season_emoji = "❄️"; season_badge = "badge-sky"
elif month in [3, 4, 5]:
    season_name = "Spring"; season_emoji = "🌸"; season_badge = "badge-green"
elif month in [6, 7, 8]:
    season_name = "Summer"; season_emoji = "☀️"; season_badge = "badge-amber"
else:
    season_name = "Autumn"; season_emoji = "🍂"; season_badge = "badge-purple"

# ======================================================
# BUILD INPUT DATAFRAME
# ======================================================
raw_input = {
    "Temperature (C)": temperature,
    "Apparent Temperature (C)": apparent_temp,
    "Humidity": humidity / 100,          # the model was trained on 0-1, the app asks for percent
    "Wind Speed (km/h)": wind_speed,
    "Wind Bearing (degrees)": wind_bearing,
    "Visibility (km)": visibility,
    "Pressure (millibars)": pressure,
    "Month": month,
    "Precip Type_snow": 1 if precip_type == "Snow" else 0,
}
# Same columns in the same order as in training (raises an error if a name does not match)
input_df = pd.DataFrame([raw_input])[feature_columns]

# Same outlier capping as in training, with the bounds learned on the train set
for col, (low, high) in cap_bounds.items():
    input_df[col] = input_df[col].clip(low, high)

# ======================================================
# LIVE METRICS (always visible)
# ======================================================
st.markdown('<p class="section-title" style="margin-top:0.5rem">📡 Live Sensor Readings</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "🌡", f"{temperature:.1f}°C", "Temperature"),
    (c2, "💧", f"{humidity:.0f}%",      "Humidity"),
    (c3, "🌬", f"{wind_speed:.1f} km/h", "Wind Speed"),
    (c4, "👁", f"{visibility:.1f} km",   "Visibility"),
    (c5, "🔵", f"{pressure:.0f} mb",     "Pressure"),
]
for col, icon, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="fancy-divider" style="margin-top:1.5rem"></div>', unsafe_allow_html=True)

# ======================================================
# VFX
# ======================================================
def apply_vfx(choice):
    rain_css = """
    <style>
    @keyframes rainFall { from {background-position:0 0;} to {background-position:-80px 250px;} }
    .rain-overlay {
        position:fixed; inset:0; pointer-events:none; z-index:9999;
        background-image: repeating-linear-gradient(
            180deg,
            transparent 0px, transparent 18px,
            rgba(147,197,253,0.07) 18px, rgba(147,197,253,0.07) 20px
        ), repeating-linear-gradient(
            90deg,
            transparent 0px, transparent 28px,
            rgba(147,197,253,0.04) 28px, rgba(147,197,253,0.04) 30px
        );
        animation: rainFall 0.25s linear infinite;
    }
    </style><div class="rain-overlay"></div>
    """
    snow_css = """
    <style>
    @keyframes snowFall { from {background-position:0 0;} to {background-position:60px 400px;} }
    .snow-overlay {
        position:fixed; inset:0; pointer-events:none; z-index:9999;
        background-image:
            radial-gradient(circle, rgba(255,255,255,0.35) 1px, transparent 1px),
            radial-gradient(circle, rgba(255,255,255,0.2) 1.5px, transparent 1.5px);
        background-size: 60px 60px, 90px 90px;
        background-position: 0 0, 30px 30px;
        animation: snowFall 4s linear infinite;
    }
    </style><div class="snow-overlay"></div>
    """
    if choice.lower() == "rain":
        st.markdown(rain_css, unsafe_allow_html=True)
    elif choice.lower() == "snow":
        st.markdown(snow_css, unsafe_allow_html=True)


# ======================================================
# PREDICTION
# ======================================================
WEATHER_META = {
    "rain":    {"emoji":"🌧️", "color":"#38bdf8", "gradient":"135deg,#1e3c72,#2a5298", "advice":"Carry an umbrella and stay dry! ☔", "badge":"badge-sky"},
    "clear":   {"emoji":"☀️", "color":"#fbbf24", "gradient":"135deg,#f59e0b,#d97706", "advice":"Perfect weather for outdoor activities! 🏃", "badge":"badge-amber"},
    "sun":     {"emoji":"☀️", "color":"#fbbf24", "gradient":"135deg,#f59e0b,#d97706", "advice":"Enjoy the sunshine! 🌈", "badge":"badge-amber"},
    "cloud":   {"emoji":"⛅",  "color":"#94a3b8", "gradient":"135deg,#475569,#64748b", "advice":"Mild day ahead — no extremes. 🌤️", "badge":"badge-purple"},
    "fog":     {"emoji":"🌫️", "color":"#94a3b8", "gradient":"135deg,#475569,#64748b", "advice":"Drive carefully — low visibility. 🚗", "badge":"badge-purple"},
    "snow":    {"emoji":"❄️", "color":"#bfdbfe", "gradient":"135deg,#60a5fa,#93c5fd", "advice":"Bundle up, it's freezing outside! 🧥", "badge":"badge-sky"},
    "wind":    {"emoji":"💨", "color":"#a5f3fc", "gradient":"135deg,#06b6d4,#0891b2", "advice":"Hold onto your hat! 🌬️", "badge":"badge-sky"},
    "breezy":  {"emoji":"💨", "color":"#a5f3fc", "gradient":"135deg,#06b6d4,#0891b2", "advice":"Hold onto your hat! 🌬️", "badge":"badge-sky"},
    "overcast":{"emoji":"☁️", "color":"#94a3b8", "gradient":"135deg,#475569,#64748b", "advice":"Grey skies, but no extremes. 🌥️", "badge":"badge-purple"},
    "default": {"emoji":"🌤️", "color":"#38bdf8", "gradient":"135deg,#0f172a,#1e3a5f", "advice":"Have a wonderful day! ✨", "badge":"badge-sky"},
}

def get_meta(weather):
    w = weather.lower()
    for key, val in WEATHER_META.items():
        if key in w:
            return val
    return WEATHER_META["default"]


if predict_btn:
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    weather = classes[prediction]
    confidence = np.max(probabilities) * 100

    meta = get_meta(weather)
    apply_vfx(precip_type)

    # ── RESULT BANNER ──
    st.markdown(f"""
    <div class="result-banner" style="margin-bottom:2rem;">
        <span class="result-emoji">{meta['emoji']}</span>
        <div class="result-weather">{weather}</div>
        <div class="result-confidence">Model probability</div>
        <div class="conf-bar-wrap" style="max-width:400px;margin:0.8rem auto;">
            <div class="conf-bar-fill" style="width:{confidence:.1f}%"></div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:700; color:{meta['color']}; margin-top:0.3rem;">{confidence:.1f}%</div>
        <div style="margin-top:1rem;">
            <span class="badge {meta['badge']}">{season_name} {season_emoji}</span>
            &nbsp;
            <span class="badge badge-purple">📅 {calendar.month_name[month]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("The model sees one hour of readings, but the category describes the whole day's sky. "
               "Probabilities are class-weighted, so Rainy, Breezy and Clear are inflated and unreliable (very few days in the training data).")

    # ── DETAIL COLUMNS ──
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 Class Probability Distribution</p>', unsafe_allow_html=True)

        sorted_idx = np.argsort(probabilities)[::-1]
        bar_colors = ["#38bdf8","#818cf8","#34d399","#fbbf24","#f87171","#a78bfa","#fb923c"]

        for i, idx in enumerate(sorted_idx):
            pct = probabilities[idx] * 100
            color = bar_colors[i % len(bar_colors)]
            st.markdown(f"""
            <div class="proba-row">
                <div class="proba-name">{classes[idx]}</div>
                <div class="proba-bar-wrap">
                    <div class="proba-bar" style="width:{pct:.1f}%;background:{color};box-shadow:0 0 8px {color}66;"></div>
                </div>
                <div class="proba-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🎯 AI Recommendation</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:2.5rem; text-align:center; margin-bottom:0.8rem;">{meta['emoji']}</div>
        <div style="font-size:1rem; line-height:1.6; color:#e2e8f0; text-align:center;">{meta['advice']}</div>
        <div class="fancy-divider"></div>
        <p class="section-title">🗓️ Contextual Info</p>
        <div style="display:flex; flex-direction:column; gap:0.6rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#64748b; font-size:0.82rem;">Season</span>
                <span class="badge {season_badge}">{season_emoji} {season_name}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#64748b; font-size:0.82rem;">Month</span>
                <span style="font-family:'JetBrains Mono',monospace; color:#38bdf8; font-size:0.85rem;">{calendar.month_name[month]}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#64748b; font-size:0.82rem;">Precipitation</span>
                <span class="badge badge-sky">{precip_type}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── FEATURE SUMMARY ROW ──
    st.markdown('<div class="fancy-divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🧾 Input Feature Summary</p>', unsafe_allow_html=True)

    feat_cols = st.columns(4)
    features_display = [
        ("Apparent Temp", f"{apparent_temp:.1f}°C"),
        ("Wind Direction", f"{wind_bearing}°"),
        ("Month", calendar.month_name[month]),
        ("Precipitation", precip_type),
    ]
    for i, (label, val) in enumerate(features_display):
        with feat_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="padding:1rem;">
                <div class="metric-value" style="font-size:1.3rem;">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── IDLE STATE ──
    st.markdown("""
    <div style="
        text-align:center;
        padding:5rem 2rem;
        border:1px dashed rgba(255,255,255,0.08);
        border-radius:24px;
        margin-top:1rem;
    ">
        <div style="font-size:4rem; margin-bottom:1rem;">🌦️</div>
        <div style="font-size:1.2rem; font-weight:600; color:#e2e8f0; margin-bottom:0.5rem;">
            Ready to Predict
        </div>
        <div style="font-size:0.9rem; color:#475569; max-width:400px; margin:0 auto;">
            Adjust the atmospheric parameters in the sidebar and click <strong style="color:#38bdf8">Run Prediction</strong> to get your AI weather forecast.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div style="
    margin-top:4rem;
    padding:2rem;
    border-top:1px solid rgba(255,255,255,0.06);
    display:flex;
    justify-content:space-between;
    align-items:center;
    flex-wrap:wrap;
    gap:1rem;
">
    <div>
        <span style="font-weight:700; color:white; font-size:0.9rem;">WeatherMind AI</span>
        <span style="color:#334155; margin:0 0.5rem;">•</span>
        <span style="color:#475569; font-size:0.82rem;">End-to-End ML Deployment</span>
    </div>
    <div style="color:#334155; font-size:0.8rem; font-family:'JetBrains Mono',monospace;">
        Made by Kalyan
    </div>
    <div style="display:flex; gap:0.5rem; align-items:center;">
        <span class="badge badge-sky">scikit-learn</span>
        <span class="badge badge-purple">streamlit</span>
        <span class="badge badge-green">python</span>
    </div>
</div>
""", unsafe_allow_html=True)