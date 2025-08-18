import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Water Quality Analyzer", page_icon="💧")
st.title("💧 Water Safety Dashboard")
st.write("Evaluate water safety using WHO guidelines and AI predictions.")

# 🧠 Educational Section
with st.expander("ℹ️ WHO Guidelines & Key Indicators"):
    st.markdown("""
    - **Coliforms**: 0 CFU/100 mL → 🚨 Unsafe if > 0  
    - **pH**: 6.5–8.5  
    - **TDS**: ≤ 1000 mg/L  
    - **EC**: ≤ 1400 µS/cm  
    - **Hardness**: >500 mg/L may cause scaling  
    - **DO (Dissolved Oxygen)**: >6 mg/L preferred for freshness  
    """)

# ---------------- Caching Functions ----------------
@st.cache_resource
def load_model_cached():
    return load_model("water_quality_ann.h5")

@st.cache_resource
def load_scaler_cached():
    return joblib.load("scaler.pkl")

@st.cache_data
def read_csv_cached(file):
    return pd.read_csv(file)

# ---------------- Mode Selection ----------------
mode = st.radio("📌 Select Data Entry Mode", ["Upload CSV Files", "Manual Input"])

if mode == "Upload CSV Files":
    phys_file = st.file_uploader("Upload Physical Parameter CSV", type=["csv"])
    bact_file = st.file_uploader("Upload Bacterial Test CSV", type=["csv"])

    if phys_file and bact_file:
        start = time.time()
        phys_df = read_csv_cached(phys_file)
        bact_df = read_csv_cached(bact_file)
        st.success("✅ Files uploaded!")

        try:
            phys_df[['ec_val', 'temp']] = phys_df['EC'].str.split('/', expand=True).astype(float)
        except Exception:
            st.error("⚠️ Unable to split 'EC'. Please ensure it's in 'value/temp' format (e.g., '1400/25').")

        df = pd.merge(phys_df, bact_df, on="Sample", how="inner")
        st.write(f"⏱️ Data processing time: {time.time() - start:.2f} seconds")
    else:
        df = None

elif mode == "Manual Input":
    st.info("Enter water quality parameters for a single sample:")
    pH = st.number_input("pH", 0.0, 14.0, step=0.01)
    ec_val = st.number_input("Electrical Conductivity (µS/cm)", 0.0, step=1.0)
    temp = st.number_input("Temperature (°C)", 0.0, 100.0, step=0.1)
    tds = st.number_input("TDS (mg/L)", 0.0, step=1.0)
    hardness = st.number_input("Hardness (mg/L as CaCO3)", 0.0, step=1.0)
    do = st.number_input("Dissolved Oxygen (mg/L)", 0.0, step=0.1)
    coliform = st.selectbox("Coliform presence", ["No", "Yes"])

    df = pd.DataFrame({
        "sample": ["ManualEntry1"],
        "ph": [pH],
        "ec_val": [ec_val],
        "temp": [temp],
        "tds": [tds],
        "hardness": [hardness],
        "do": [do],
        "coliform": [0 if coliform == "No" else 1]
    })

# ---------------- Process if Data Available ----------------
if df is not None and not df.empty:
    df.columns = df.columns.str.strip().str.lower()

    # WHO Checks
    if 'ph' in df.columns:
        df["ph_status"] = df["ph"].apply(lambda x: "✅ OK" if 6.5 <= x <= 7.5 else "⚠️ Out of Range")
    if 'tds' in df.columns:
        df["tds_status"] = df["tds"].apply(lambda x: "✅ OK" if x <= 300 else "⚠️ High")
    if 'ec_val' in df.columns:
        df["ec_status"] = df["ec_val"].apply(lambda x: "✅ OK" if x <= 400 else "⚠️ High")
    if 'coliform' in df.columns:
        df["coliform_status"] = df["coliform"].apply(lambda x: "✅ Safe" if x == 0 else "🚨 Unsafe")
    if "hardness" in df.columns:
        df["hardness_status"] = df["hardness"].apply(
            lambda x: "💧 Soft" if x <= 60 else
                      "🧂 Moderate" if x <= 120 else
                      "🪨 Hard" if x <= 180 else
                      "⚠️ Very Hard"
        )
    if "do" in df.columns:
        df["do_status"] = df["do"].apply(lambda x: "✅ Good" if x >= 6 else "⚠️ Low")

    # AI prediction
    try:
        start = time.time()
        model = load_model_cached()
        scaler = load_scaler_cached()
        features = df[["ec_val", "temp", "ph", "tds"]]
        X_scaled = scaler.transform(features)
        prediction = model.predict(X_scaled)
        df["prediction"] = np.argmax(prediction, axis=1)
        df["interpretation"] = df["prediction"].map({0: "Good", 1: "Moderate", 2: "Poor"})
        st.success("🧠 AI predictions generated!")
        st.write(f"⏱️ Prediction time: {time.time() - start:.2f} seconds")
    except Exception as e:
        st.warning(f"⚠️ Model/scaler issue: {e}")
        df["interpretation"] = "Unavailable"

    # Pie chart function
    def pie_chart(col, title):
        if col in df.columns:
            counts = df[col].value_counts()
            labels = counts.index.tolist()
            sizes = counts.values.tolist()
            colors = ["#4CAF50" if "✅" in l or "💧" in l else
                      "#FFC107" if "⚠️" in l or "🧂" in l or "🪨" in l else
                      "#F44336" for l in labels]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.subheader(f"📊 {title}")
            st.pyplot(fig)

    pie_chart("ph_status", "pH Compliance")
    pie_chart("tds_status", "TDS Compliance")
    pie_chart("ec_status", "Electrical Conductivity")
    pie_chart("coliform_status", "Coliform Presence")
    pie_chart("hardness_status", "Water Hardness")
    pie_chart("do_status", "Dissolved Oxygen")

    # Advisory
    if "🚨 Unsafe" in df.get("coliform_status", []):
        st.error("🚨 Coliform bacteria detected!")
        with st.warning("💡 Boiling Water Advisory"):
            st.markdown("""
            One or more samples show microbial contamination.  
            **Please boil water for at least 1 minute at a rolling boil** before drinking or cooking.  
            Vulnerable groups (infants, elderly, immunocompromised) are especially at risk.
            """)

    # Text summary
    st.subheader("📋 Overall Safety Summary")
    param_columns = [
        "ph_status", "tds_status", "ec_status",
        "coliform_status", "hardness_status", "do_status"
    ]
    for col in param_columns:
        if col in df.columns:
            total = len(df)
            safe = df[col].str.contains("✅|💧|🧂|🪨").sum()
            st.markdown(f"**{col.replace('_status','').upper()}**: {safe/total*100:.1f}% samples within acceptable range.")

else:
    if mode == "Upload CSV Files":
        st.info("📂 Please upload both Physical and Bacterial CSV files to continue.")








