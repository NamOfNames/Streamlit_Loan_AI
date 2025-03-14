import os
import warnings
import tensorflow
import joblib
import streamlit as st
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


warnings.filterwarnings("ignore")

# Load trained model
model = tensorflow.keras.models.load_model("loan_model.h5")
scaler = joblib.load("scaler.pkl")  # Load the saved scaler

# Streamlit UI
st.title("Loan Approval Prediction App ($)")
st.write("Enter your details to check loan approval status")

# User Inputs
dependents = st.number_input("Number of Dependents", min_value=0)
graduate = st.selectbox("Are you a Graduate?", ["Yes", "No"])
self_employed = st.selectbox("Are you Self-Employed?", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income (Per Year)", min_value=0)
# coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in USD)", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=1)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850)
residential_assets_value = st.number_input("Residential assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury assets Value", min_value=0)
bank_asset_value = st.number_input("Bank assets Value", min_value=0)

# Convert categorical inputs
graduate = 1 if graduate == "Yes" else 0
self_employed = 0 if self_employed == "Yes" else 1

# Prepare input for model
input_data = np.array(
    [[dependents, graduate, self_employed, applicant_income, loan_amount, loan_term, credit_score,
      residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]])
input_data_scaled = scaler.transform(input_data)  # Scale input

# Prediction
if st.button("Check Loan Status"):
    prediction = model.predict(input_data_scaled)
    result = "Approved" if prediction[0][0] > 0.5 else "Rejected"
    st.subheader(f"Loan Status: {result}")
