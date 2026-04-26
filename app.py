import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved files
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="HR Attrition Predictor")
st.title("📊 HR Attrition Prediction App")

st.write("Enter employee details to predict attrition:")

# ---------------- INPUT FIELDS ---------------- #

age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 200000, 50000)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
years_at_company = st.number_input("Years at Company", 0, 40, 5)
overtime = st.selectbox("Overtime", ["Yes", "No"])

# ---------------- PREPROCESS INPUT ---------------- #

# Convert categorical
overtime = 1 if overtime == "Yes" else 0

# Create input dataframe (IMPORTANT: match training columns)
input_dict = {
    "Age": age,
    "MonthlyIncome": monthly_income,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "YearsAtCompany": years_at_company,
    "OverTime_Yes": overtime
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Align with training features
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[features]

# Scale
input_scaled = scaler.transform(input_df)

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Employee is likely to leave (Risk: {probability:.2f})")
    else:
        st.success(f"✅ Employee is likely to stay (Confidence: {1 - probability:.2f})")
