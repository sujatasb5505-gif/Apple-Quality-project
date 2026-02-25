import streamlit as st
import pandas as pd
import os
from joblib import load

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler
model = load(os.path.join(BASE_DIR, "apple_quality_model.joblib"))
scaler = load(os.path.join(BASE_DIR, "scaler.joblib"))

# Streamlit UI
st.title("🍎 Apple Quality Prediction")
st.header("Enter Apple Features")

size = st.number_input("Size (cm)", min_value=0.0, step=0.1)
weight = st.number_input("Weight (g)", min_value=0.0, step=0.1)
sweetness = st.number_input("Sweetness (1–10)", 1, 10)
crunchiness = st.number_input("Crunchiness (1–10)", 1, 10)
juiciness = st.number_input("Juiciness (1–10)", 1, 10)
ripeness = st.number_input("Ripeness (1–10)", 1, 10)
acidity = st.number_input("Acidity (pH)", min_value=0.0, step=0.1)

if st.button("Predict Quality"):
    input_data = pd.DataFrame({
        "Size": [size],
        "Weight": [weight],
        "Sweetness": [sweetness],
        "Crunchiness": [crunchiness],
        "Juiciness": [juiciness],
        "Ripeness": [ripeness],
        "Acidity": [acidity]
    })

    # ONLY transform (do NOT fit)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ The apple is predicted to be of GOOD quality")
    else:
        st.error("❌ The apple is predicted to be of POOR quality")
