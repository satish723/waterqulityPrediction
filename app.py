# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st

# Load the model and structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Streamlit page configuration
st.set_page_config(page_title="Water Pollutants Predictor", page_icon="💧", layout="centered")

# App title
st.title("💧 Water Pollutants Predictor")
st.write("🌊 **Predict water pollutant levels based on Year and Station ID**")

st.markdown("---")

# User inputs
year_input = st.number_input("📅 Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("🏷️ Enter Station ID", value='1')

st.markdown("---")

# To encode and then predict
if st.button('🔮 Predict'):
    if not station_id:
        st.warning('⚠️ Please enter the station ID.')
    else:
        # Prepare the input
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model cols
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.subheader(f"🔬 Predicted Pollutant Levels for Station **{station_id}** in **{year_input}**:")

        # Display results in a cleaner format
        result_df = pd.DataFrame({
            'Pollutant': pollutants,
            'Predicted Level': [f"{val:.2f}" for val in predicted_pollutants]
        })

        st.table(result_df)

        # Optional: highlight pollutants exceeding certain thresholds (example)
        st.markdown("✅ **Prediction Complete. Ensure safe limits are maintained.**")

st.markdown("---")
st.caption("Developed by KANDREGULA SATISH 🚀")
