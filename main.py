import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load("heart_model_v2_xgb.pkl")

st.set_page_config(page_title="Heart Attack Prediction", page_icon="❤️")
st.title("Heart Attack Prediction App")
st.subheader("🚑 Made by Abhishek Choudhary")

with st.expander("ℹ️ About the Dataset"):
    st.markdown("""
    - Dataset from Zheen hospital, Erbil (2019)
    - Includes demographic, clinical, lifestyle features
    - Heart Attack Risk label: 1=Positive, 0=Negative
    """)

with st.form("input_form"):
    # Optional field
    patient_id = st.text_input("Patient ID (Optional)")

    # Important fields
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
    blood_pressure = st.text_input("Blood Pressure (Systolic/Diastolic)", value="120/80")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=80)
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    obesity = st.selectbox("Obesity", ["No", "Yes"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    exercise_hours_per_week = st.number_input("Exercise Hours Per Week", min_value=0, max_value=168, value=3)
    diet = st.selectbox("Diet Type", ["Balanced", "Unhealthy", "Healthy", "Other"])
    previous_heart_problems = st.selectbox("Previous Heart Problems", ["No", "Yes"])
    medication_use = st.selectbox("Medication Use", ["No", "Yes"])
    stress_level = st.slider("Stress Level (1=Low to 10=High)", min_value=1, max_value=10, value=5)
    sedentary_hours_per_day = st.number_input("Sedentary Hours Per Day", min_value=0, max_value=24, value=6)
    income = st.text_input("Income (Optional)")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=0, max_value=1000, value=150)
    physical_activity_days_per_week = st.number_input("Physical Activity Days Per Week", min_value=0, max_value=7, value=3)
    sleep_hours_per_day = st.number_input("Sleep Hours Per Day", min_value=0, max_value=24, value=7)
    country = st.text_input("Country (Optional)")
    continent = st.text_input("Continent (Optional)")
    hemisphere = st.text_input("Hemisphere (Optional)")

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Validate Blood Pressure format
        try:
            systolic_str, diastolic_str = blood_pressure.split('/')
            systolic_bp = int(systolic_str.strip())
            diastolic_bp = int(diastolic_str.strip())
            if systolic_bp <= diastolic_bp:
                st.error("Systolic BP must be higher than Diastolic BP!")
                st.stop()
        except Exception:
            st.error("Invalid Blood Pressure format! Please use Systolic/Diastolic format like '120/80'.")
            st.stop()

        # Encode categorical binary fields
        def yes_no_to_bin(x):
            return 1 if x.lower() == 'yes' else 0

        gender_bin = 1 if sex == "Male" else 0

        # Prepare input dictionary with all required fields
        input_dict = {
            'age': age,
            'gender': gender_bin,
            'cholesterol': cholesterol,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'diabetes': yes_no_to_bin(diabetes),
            'family_history': yes_no_to_bin(family_history),
            'smoking': yes_no_to_bin(smoking),
            'obesity': yes_no_to_bin(obesity),
            'alcohol_consumption': yes_no_to_bin(alcohol_consumption),
            'exercise_hours_per_week': exercise_hours_per_week,
            'previous_heart_problems': yes_no_to_bin(previous_heart_problems),
            'medication_use': yes_no_to_bin(medication_use),
            'stress_level': stress_level,
            'sedentary_hours_per_day': sedentary_hours_per_day,
            'bmi': bmi,
            'triglycerides': triglycerides,
            'physical_activity_days_per_week': physical_activity_days_per_week,
            'sleep_hours_per_day': sleep_hours_per_day,
        }

        # Handle diet: map to dummy variables - the model expects one-hot encoding with drop_first=True in training
        diet_options = ['Balanced', 'Healthy', 'Other', 'Unhealthy']
        for d in diet_options[1:]:  # drop_first means first category omitted
            input_dict[f'diet_{d}'] = 1 if diet == d else 0

        # Optional fields - fill missing with default or zero
        # If your model does not use Patient ID, Country, Continent, Hemisphere, Income, just ignore or set 0
        # Here we include them as columns with default 0 or empty string to keep shape consistent if needed

        # For demonstration, we skip them since model does not use them

        # Convert input_dict to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Predict
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ The patient is at **high risk** of a heart attack.")
        else:
            st.success("✅ The patient is **not at risk** of a heart attack.")

st.markdown("---")
st.markdown("""
Made with ❤️ by **Abhishek Choudhary**  
🔗 [Hashnode Blog](https://hashnode.com/@opensere)  
🔗 [GitHub](https://github.com/OneSere)

© 2025 – All rights reserved.

_Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical diagnosis._
""")
