import streamlit as st
import pandas as pd
import joblib

import joblib

data = joblib.load("model/lr_model.pkl")
model = data["model"]
features = data["columns"]


st.title("Heart Disease Prediction App")

age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=1.0)

weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=1.0)

daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=5000, step=500)

calories = st.number_input("Calories Intake per day", min_value=500, max_value=6000, value=2000, step=100)

sleep = st.number_input("Hours of Sleep", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

heart_rate = st.number_input("Heart Rate", min_value=30, max_value=220, value=72, step=1)

exercise = st.number_input("Exercise Hours per Week", min_value=0.0, max_value=40.0, value=3.0, step=0.5)

smoker = st.selectbox("Smoker", ["No", "Yes"])
smoker = 1 if smoker == "Yes" else 0

alcohol = st.number_input("Alcohol Consumption per Week", min_value=0, max_value=50, value=2, step=1)

diabetic = st.selectbox("Diabetic", ["No", "Yes"])
diabetic = 1 if diabetic == "Yes" else 0

bp_sys = st.number_input("BP Systolic", min_value=70, max_value=250, value=120, step=1)

bp_dia = st.number_input("BP Diastolic", min_value=40, max_value=150, value=80, step=1)


input_data = [age, gender, height, weight, daily_steps, calories, sleep, heart_rate, exercise, smoker, alcohol, diabetic, bp_sys, bp_dia]

input_df = pd.DataFrame([input_data], columns=features)

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    st.success(f" Heart Disease Risk: {prob * 100:.2f}%")




