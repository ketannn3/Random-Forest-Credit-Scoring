import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and training columns
model = joblib.load("credit_random_forest.pkl")

# You must paste your full feature column list here from training (after pd.get_dummies)
# This MUST match the training set column names exactly:
feature_columns = [
    'duration', 'amount', 'installment_rate', 'age',
    'checking_<0', 'checking_0<=X<200', 'checking_>=200', 'checking_no checking',
    'credit_history_all paid', 'credit_history_critical/other existing credit',
    'credit_history_delayed', 'credit_history_existing paid', 'credit_history_no credits/all paid',
    'purpose_business', 'purpose_car', 'purpose_education', 'purpose_furniture/equipment',
    'purpose_others', 'purpose_radio/TV', 'purpose_repair', 'purpose_vacation',
    'savings_<100', 'savings_100<=X<500', 'savings_500<=X<1000',
    'savings_>=1000', 'savings_unknown',
    'employment_unemployed', 'employment_<1', 'employment_1<=X<4',
    'employment_4<=X<7', 'employment_>=7',
    'personal_male single', 'personal_female',
    'personal_male married/widowed', 'personal_male divorced/separated'
    # add all remaining dummies used during training
]

# Input form
st.title("🏦 Credit Risk Prediction (One-Hot Encoded)")
checking = st.selectbox("Checking Account Status", ["<0", "0<=X<200", ">=200", "no checking"])
duration = st.slider("Duration (months)", 4, 72, 12)
credit_history = st.selectbox("Credit History", [
    "no credits/all paid", "all paid", "existing paid", "delayed", "critical/other existing credit"
])
purpose = st.selectbox("Purpose", [
    "radio/TV", "education", "furniture/equipment", "car", "business", "repair", "vacation", "others"
])
amount = st.number_input("Credit Amount", 100, 20000, 1000)
savings = st.selectbox("Savings Account", ["<100", "100<=X<500", "500<=X<1000", ">=1000", "unknown"])
employment = st.selectbox("Employment Duration", ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"])
installment_rate = st.slider("Installment Rate", 1, 4, 2)
personal_status = st.selectbox("Personal Status", [
    "male single", "female", "male married/widowed", "male divorced/separated"
])
age = st.slider("Age", 18, 75, 30)

if st.button("🔮 Predict Credit Risk"):
    try:
        # Raw input dictionary
        raw_data = {
            "duration": duration,
            "amount": amount,
            "installment_rate": installment_rate,
            "age": age,
            "checking": checking,
            "credit_history": credit_history,
            "purpose": purpose,
            "savings": savings,
            "employment": employment,
            "personal": personal_status
        }

        # Create DataFrame
        df = pd.DataFrame([raw_data])

        # One-hot encode with get_dummies
        df_encoded = pd.get_dummies(df)

        # Add any missing columns (from training)
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Reorder columns
        df_encoded = df_encoded[feature_columns]

        # Predict
        pred = model.predict(df_encoded)[0]
        if pred == 0:
            st.success("✅ Prediction: Good Credit Risk (0)")
        else:
            st.error("❌ Prediction: Bad Credit Risk (1)")

    except Exception as e:
        st.error(f"Error: {e}")
