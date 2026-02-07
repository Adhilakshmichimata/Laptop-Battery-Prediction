import streamlit as st
import pandas as pd
import joblib
import base64

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Laptop Battery Health Prediction",
    page_icon="🔋",
    layout="centered"
)

# ===============================
# Background Image Function
# ===============================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 👉 SET YOUR IMAGE HERE
set_bg("background.jpg")

# ===============================
# Load Model Files
# ===============================
model = joblib.load("best_battery_health_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ===============================
# App Title
# ===============================
st.markdown(
    "<h1 style='text-align:center;color:white;'>🔋 Laptop Battery Health Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;color:white;'>Predict battery health using usage & charging behavior</h4>",
    unsafe_allow_html=True
)

st.write("")

# ===============================
# User Inputs
# ===============================
brand = st.selectbox("Brand", ["Dell", "HP", "Lenovo", "Asus", "Apple"])
os = st.selectbox("Operating System", ["Windows", "Linux", "macOS"])
usage_type = st.selectbox("Usage Type", ["Office", "Gaming", "Student", "Business"])

model_year = st.number_input("Model Year", 2015, 2025, 2022)
daily_usage_hours = st.slider("Daily Usage Hours", 1, 15, 5)
charging_cycles = st.number_input("Charging Cycles", 50, 2000, 300)
avg_charge_limit_percent = st.slider("Average Charge Limit (%)", 50, 100, 80)
battery_age_months = st.number_input("Battery Age (Months)", 1, 60, 24)
overheating_issues = st.selectbox("Overheating Issues", ["No", "Yes"])
performance_rating = st.slider("Performance Rating", 1, 5, 4)

# Convert Yes/No
overheating_issues = 1 if overheating_issues == "Yes" else 0

# ===============================
# Predict Button
# ===============================
if st.button("🔍 Predict Battery Health"):

    input_df = pd.DataFrame([{
        "brand": brand,
        "model_year": model_year,
        "os": os,
        "usage_type": usage_type,
        "daily_usage_hours": daily_usage_hours,
        "charging_cycles": charging_cycles,
        "avg_charge_limit_percent": avg_charge_limit_percent,
        "battery_age_months": battery_age_months,
        "overheating_issues": overheating_issues,
        "performance_rating": performance_rating
    }])

    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            value = input_df[col].iloc[0]
            if value in encoder.classes_:
                input_df[col] = encoder.transform([value])
            else:
                input_df[col] = encoder.transform([encoder.classes_[0]])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Display Result
    st.markdown(
        f"""
        <div style='background:rgba(0,0,0,0.6);padding:20px;border-radius:15px;text-align:center;'>
            <h2 style='color:#00ffcc;'>🔋 Predicted Battery Health</h2>
            <h1 style='color:white;'>{round(prediction,2)} %</h1>
        </div>
        """,
        unsafe_allow_html=True
    )