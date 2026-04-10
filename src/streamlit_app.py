import streamlit as st
import pickle
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

@st.cache_resource
def load_artifacts():
    model  = pickle.load(open(MODELS_DIR / "smartphone_addiction_model.sav", "rb"))
    scaler = pickle.load(open(MODELS_DIR / "scaler.sav", "rb"))
    return model, scaler

model, scaler = load_artifacts()

st.title("📱 Smartphone Addiction Predictor")
st.markdown("Enter your daily smartphone usage habits below to see whether the model predicts addiction risk.")

st.divider()

daily_screen_time  = st.number_input("Daily Screen Time (hours)",    min_value=0.0, max_value=24.0, value=4.0, step=0.5)
weekend_screen_time = st.number_input("Weekend Screen Time (hours)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
social_media_hours  = st.number_input("Social Media Usage (hours)",  min_value=0.0, max_value=24.0, value=2.0, step=0.5)
app_opens           = st.number_input("App Opens Per Day",           min_value=0,   max_value=500,  value=50,  step=5)

st.divider()

if st.button("Predict", use_container_width=True):
    raw    = [[daily_screen_time, weekend_screen_time, social_media_hours, app_opens]]
    scaled = scaler.transform(raw)
    result = model.predict(scaled)[0]

    if result == 1:
        st.error(" Prediction: **Addicted**")
    else:
        st.success(" Prediction: **Not Addicted**")

    proba = model.predict_proba(scaled)[0]
    st.markdown(f"**Confidence:** Not Addicted `{proba[0]:.1%}` | Addicted `{proba[1]:.1%}`")
