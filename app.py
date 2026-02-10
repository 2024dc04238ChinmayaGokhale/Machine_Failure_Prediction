import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Machine Failure Prediction App")
st.write("Predict machine failure type using trained ML model")

# --------------------------------------------------
# Load model & scaler (cached)
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------
# Feature columns (MUST match training exactly)
# --------------------------------------------------
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("🔧 Input Parameters")

type_map = {"Low (L)": 1, "Medium (M)": 2, "High (H)": 3}
type_label = st.sidebar.selectbox("Machine Type", list(type_map.keys()))
type_value = type_map[type_label]

air_temp = st.sidebar.number_input(
    "Air temperature [K]", min_value=250.0, max_value=350.0, value=298.0
)

process_temp = st.sidebar.number_input(
    "Process temperature [K]", min_value=250.0, max_value=350.0, value=308.0
)

rpm = st.sidebar.number_input(
    "Rotational speed [rpm]", min_value=1000, max_value=3000, value=1500
)

torque = st.sidebar.number_input(
    "Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0
)

tool_wear = st.sidebar.number_input(
    "Tool wear [min]", min_value=0, max_value=300, value=100
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Failure Type"):
    # Build input DataFrame (CRITICAL)
    input_dict = {
        "Type": type_value,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    input_data = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Optional probability (if supported)
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(input_scaled)) * 100
    else:
        confidence = None

    # --------------------------------------------------
    # Decode prediction (adjust labels if needed)
    # --------------------------------------------------
    failure_map = {
        0: "No Failure",
        1: "Tool Wear Failure",
        2: "Heat Dissipation Failure",
        3: "Power Failure",
        4: "Overstrain Failure",
        5: "Random Failure"
    }

    result = failure_map.get(prediction, "Unknown")

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------
    st.subheader("📊 Prediction Result")
    st.success(f"**Predicted Failure Type:** {result}")

    if confidence is not None:
        st.info(f"**Model Confidence:** {confidence:.2f}%")

    st.write("### 🔎 Input Summary")
    st.dataframe(input_data)
