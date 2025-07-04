import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Load Models ---
model = load_model("water_quality_ann.h5")
iso_model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- WHO Limits ---
who_limits = {
    'pH': (6.5, 8.5),
    'TDS': (None, 1000),
    'Hardness': (None, 500),
    'EC_val': (None, 1500),
    'DO': (5, None),
    'Ecoli_Present': (0, 0),
    'Salmonella_Present': (0, 0)
}

# --- UI ---
st.title("💧 Smart Water Quality Analyzer")
st.write("This app uses AI to assess physicochemical and microbial water safety based on WHO guidelines.")

# Input Fields
inputs = {}
inputs['pH'] = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
inputs['DO'] = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, value=6.0)
inputs['TDS'] = st.number_input("TDS (mg/L)", min_value=0.0, value=400.0)
inputs['EC_val'] = st.number_input("Electrical Conductivity (μS/cm)", min_value=0.0, value=800.0)
inputs['Temp'] = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)
inputs['Hardness'] = st.number_input("Hardness (mg/L as CaCO₃)", min_value=0.0, value=150.0)
inputs['Ecoli_Present'] = 1 if st.selectbox("E. coli Detected?", ["No", "Yes"]) == "Yes" else 0
inputs['Salmonella_Present'] = 1 if st.selectbox("Salmonella Detected?", ["No", "Yes"]) == "Yes" else 0

# Feature Engineering
hardness_safe = inputs['Hardness'] if inputs['Hardness'] > 0 else 0.1
inputs['TDS_Hardness'] = inputs['TDS'] / hardness_safe
inputs['DO_EC'] = inputs['DO'] * inputs['EC_val']
inputs['pH_Deviation'] = abs(inputs['pH'] - 7)
inputs['Bacteria_Load'] = inputs['Ecoli_Present'] + inputs['Salmonella_Present']

# Order of Inputs
features_order = ['pH', 'DO', 'TDS', 'EC_val', 'Temp', 'Hardness',
                  'Ecoli_Present', 'Salmonella_Present',
                  'TDS_Hardness', 'DO_EC', 'pH_Deviation', 'Bacteria_Load']

X = np.array([[inputs[f] for f in features_order]])
X_scaled = scaler.transform(X)

# --- Prediction ---
pred = model.predict(X_scaled)
pred_class = np.argmax(pred)
label_map = {0: "Poor", 1: "Moderate", 2: "Good"}
st.subheader(f"💡 Predicted Water Quality: **{label_map[pred_class]}**")

# --- WHO Compliance ---
def check_who(values):
    results = []
    for param, (min_val, max_val) in who_limits.items():
        val = values.get(param)
        if min_val is not None and val < min_val:
            results.append(f"🔻 {param} below WHO minimum ({min_val})")
        if max_val is not None and val > max_val:
            results.append(f"🔺 {param} exceeds WHO maximum ({max_val})")
    return results or ["✅ Compliant with all WHO parameters"]

st.subheader("🧪 WHO Compliance Check")
for msg in check_who(inputs):
    st.write(msg)

# --- Anomaly Detection ---
anomaly = iso_model.predict(X_scaled)[0]
if anomaly == -1:
    st.error("⚠️ Anomaly Detected: This sample has unusual parameter patterns.")
else:
    st.success("✅ No anomaly detected.")

