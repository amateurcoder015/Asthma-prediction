import streamlit as st
import joblib 
import numpy as np 
import pandas as pd 

model = joblib.load('asthma_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')
df = pd.read_csv('synthetic_asthma_dataset.csv')
st.title("Asthma Prediction")
st.markdown("Provide the following details to check your asthma risk:")

age = st.slider("Age", 0, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
smoking_status = st.selectbox("Smoking Status", ["Non-Smoker", "Former Smoker", "Current Smoker"])
family_history = st.selectbox("Family History of Asthma", df['Family_History'].unique())
allergies = st.selectbox("Allergies", df['Allergies'].unique())
air_pollution_level = st.selectbox("Air Pollution Level", df['Air_Pollution_Level'].unique())
physical_activity = st.selectbox("Physical Activity Level", df['Physical_Activity_Level'].unique())
occupation_type = st.selectbox("Occupation Type", df['Occupation_Type'].unique())
comorbidities = st.selectbox("Comorbidities", df['Comorbidities'].unique())
medication_adherence = st.slider("Medication Adherence", 0, 100, 50)
er_visits = st.slider("ER Visits", 0, 10, 5)
peak_flow = st.number_input("Peak Flow", 0, 1000, 300)
FeNO_level = st.number_input("FeNO Level", 0, 100, 50)

if st.button("Predict"):

    raw_input = {
        'Age': age,
        'BMI': bmi,
        'Family_History': family_history,
        'Gender_'+gender: 1,
        'Medication_Adherence': medication_adherence,
        'ER_Visits': er_visits,
        'Peak_Flow': peak_flow,
        'FeNO_Level': FeNO_level,
        'Smoking_Status_'+smoking_status: 1,
        'Allergies_'+allergies: 1,
        'Air_Pollution_Level_'+air_pollution_level: 1,
        'Physical_Activity_Level_'+physical_activity: 1,
        'Occupation_Type_'+occupation_type: 1,
        'Comorbidities_'+comorbidities: 1,

    }

    # Preprocess the input
    input_df = pd.DataFrame([raw_input])
    input_df = input_df[features]
    input_df = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df)

    # Display result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("High risk of asthma exacerbation.")
    else:
        st.success("Low risk of asthma exacerbation.")
