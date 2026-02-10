import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Machine Predictive Maintenance",
    layout="centered"
)

st.title("Machine Predictive Maintenance App")
st.write("Predict machine failure type using a trained ML model.")

# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Machine Parameters")

air_temp = st.number_input("Air Temperature [K]", min_value=250.0, max_value=350.0)
process_temp = st.number_input("Process Temperature [K]", min_value=250.0, max_value=350.0)
rot_speed = st.number_input("Rotational Speed [rpm]", min_value=0.0)
torque = st.number_input("Torque [Nm]", min_value=0.0)
tool_wear = st.number_input("Tool Wear [min]", min_value=0.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Failure Type"):
    input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Failure Type: {prediction[0]}")
