# Machine Predictive Maintenance – ML Deployment

This project demonstrates deployment of a Machine Learning model
for predictive maintenance using Streamlit Community Cloud.

## 🚀 Overview
The application predicts machine failure type based on operational
parameters such as temperature, speed, torque, and tool wear.

## 🧠 Model Artifacts
- model.pkl – trained ML model
- scaler.pkl – feature scaler
- DataForML.pkl – dataset (used during training)

## 🛠️ Tech Stack
- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- XGBoost

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
