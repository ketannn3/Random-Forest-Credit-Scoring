import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("🏦 German Credit Risk Prediction App")
st.write("Predict whether a person is a good or bad credit risk using a trained Random Forest model.")

# Load the trained model
model = joblib.load("credit_random_forest.pkl")

# --- Encoding maps ---
checking_status_map = {"<0": 0, "0<=X<200": 1, ">=200": 2, "no checking": 3}
credit_history_map = {
    "no credits/all paid": 0,
    "all paid": 1,
    "existing paid": 2,
    "delayed": 3,
    "critical/other existing credit": 4
}
purpose_map = {
    "car": 0, "furniture/equipment": 1, "radio/TV": 2,
    "education": 3, "business": 4, "repair": 5, "vacation": 6, "others": 7
}
savings_account_map = {"<100": 0, "100<=X<500": 1, "500<=X<1000": 2, ">=1000": 3, "unknown": 4}
employment_map = {"unemployed": 0, "<1": 1, "1<=X<4": 2, "4<=X<7": 3, ">=7": 4}
personal_status_map = {
    "male single": 0, "female": 1,
    "male married/widowed": 2, "male divorced/separated": 3
}

# --- Input fields ---
checking = st.selectbox("Status of checking account", list(checking_status_map.keys()))
duration = st.slider("Duration in months", 4, 72, 12)
credit_history = st.selectbox("Credit history", list(credit_history_map.keys()))
purpose = st.selectbox("Purpose", list(purpose_map.keys()))
amount = st.number_input("Credit amount", 100, 20000, 1000)
savings = st.selectbox("Savings account/bonds", list(savings_account_map.keys()))
employment = st.selectbox("Present employment since", list(employment_map.keys()))
installment_rate = st.slider("Installment rate (% of income)", 1, 4, 2)
personal_status = st.selectbox("Personal status and sex", list(personal_status_map.keys()))
age = st.slider("Age in years", 18, 75, 30)

# --- Predict Button ---
if st.button("🔮 Predict Credit Risk"):
    try:
        # Convert to numerical inputs
        features = [
            checking_status_map[checking],
            duration,
            credit_history_map[credit_history],
            purpose_map[purpose],
            amount,
            savings_account_map[savings],
            employment_map[employment],
            installment_rate,
            personal_status_map[personal_status],
            age
        ]

        input_data = np.array([features])  # Shape (1, 10)
        prediction = model.predict(input_data)[0]

        # Show result
        if prediction == 0:
            st.success("✅ Prediction: Good Credit Risk (0)")
        else:
            st.error("❌ Prediction: Bad Credit Risk (1)")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
