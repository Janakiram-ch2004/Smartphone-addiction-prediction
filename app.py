import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(page_title="Smartphone Addiction Predictor", layout="wide")
st.title("📱 Smartphone Addiction Predictor")
st.write("Enter your daily screen time habits below, and our Machine Learning model will predict your addiction risk.")


try:
    model = joblib.load('random_forest_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    selector = joblib.load('feature_selector.pkl')
except FileNotFoundError:
    st.error("Error: Required model files (.pkl) not found. Please run the training script first.")
    st.stop()


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=22)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    daily_screen_time = st.number_input("Daily Screen Time (Hours)", 0.0, 24.0, 5.0)
    social_media = st.number_input("Social Media (Hours)", 0.0, 24.0, 2.0)
    gaming = st.number_input("Gaming (Hours)", 0.0, 24.0, 1.0)
    work_study = st.number_input("Work/Study (Hours)", 0.0, 24.0, 6.0)

with col2:
    sleep = st.number_input("Sleep (Hours)", 0.0, 24.0, 7.0)
    notifications = st.number_input("Notifications per Day", 0, 1000, 50)
    app_opens = st.number_input("App Opens per Day", 0, 500, 30)
    weekend_screen_time = st.number_input("Weekend Screen Time (Hours)", 0.0, 24.0, 8.0)
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    academic_impact = st.selectbox("Academic Work Impact", ["Yes", "No"])


if st.button("Predict Addiction", type="primary"):
    
    input_data = pd.DataFrame({
        'age': [age], 'gender': [gender], 'daily_screen_time_hours': [daily_screen_time],
        'social_media_hours': [social_media], 'gaming_hours': [gaming], 
        'work_study_hours': [work_study], 'sleep_hours': [sleep],
        'notifications_per_day': [notifications], 'app_opens_per_day': [app_opens],
        'weekend_screen_time': [weekend_screen_time], 'stress_level': [stress_level],
        'academic_work_impact': [academic_impact]
    })

    for col in input_data.select_dtypes(include=['object']).columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])
            
    
    scaled_data = scaler.transform(input_data)
    
    final_features = selector.transform(scaled_data)
    
   
    prediction = model.predict(final_features)
    
  
    st.markdown("---")
    if prediction[0] == 1:
        st.error(" Warning: High signs of smartphone addiction detected! Consider reducing your screen time.")
    else:
        st.success(" Great! You are in the safe zone. No severe addiction detected.")
