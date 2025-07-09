import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("credit_random_forest.pkl")

# Column list used during training (one-hot encoded)
columns = joblib.load("model_columns.pkl")  # We'll save these from training

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("🏦 German Credit Risk Prediction App")

# --- User Inputs ---
status = st.selectbox("Status of Checking Account", ["A11", "A12", "A13", "A14"])
duration = st.slider("Duration in Months", 4, 72, 12)
credit_history = st.selectbox("Credit History", ["A30", "A31", "A32", "A33", "A34"])
purpose = st.selectbox("Purpose", ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49"])
credit_amount = st.number_input("Credit Amount", 250, 20000, 1000)
savings = st.selectbox("Savings Account", ["A61", "A62", "A63", "A64", "A65"])
employment = st.selectbox("Employment Since", ["A71", "A72", "A73", "A74", "A75"])
installment_rate = st.slider("Installment Rate %", 1, 4, 2)
personal_status = st.selectbox("Personal Status and Sex", ["A91", "A92", "A93", "A94"])
debtors = st.selectbox("Other Debtors/Guarantors", ["A101", "A102", "A103"])
residence = st.slider("Residence Duration (Years)", 1, 4, 2)
property = st.selectbox("Property", ["A121", "A122", "A123", "A124"])
age = st.slider("Age", 18, 75, 30)
installment_plan = st.selectbox("Other Installment Plans", ["A141", "A142", "A143"])
housing = st.selectbox("Housing", ["A151", "A152", "A153"])
existing_credits = st.slider("Number of Existing Credits", 1, 4, 1)
job = st.selectbox("Job", ["A171", "A172", "A173", "A174"])
liable_people = st.slider("Number of Liable People", 1, 2, 1)
telephone = st.selectbox("Telephone", ["A191", "A192"])
foreign_worker = st.selectbox("Foreign Worker", ["A201", "A202"])

# --- Prediction Button ---
if st.button("🔮 Predict Credit Risk"):
    input_dict = {
        "Status": status,
        "Duration": duration,
        "CreditHistory": credit_history,
        "Purpose": purpose,
        "CreditAmount": credit_amount,
        "Savings": savings,
        "Employment": employment,
        "InstallmentRate": installment_rate,
        "PersonalStatusSex": personal_status,
        "Debtors": debtors,
        "ResidenceDuration": residence,
        "Property": property,
        "Age": age,
        "OtherInstallmentPlans": installment_plan,
        "Housing": housing,
        "ExistingCredits": existing_credits,
        "Job": job,
        "LiablePeople": liable_people,
        "Telephone": telephone,
        "ForeignWorker": foreign_worker
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)

    # Ensure all training columns exist
    for col in columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Match column order
    input_encoded = input_encoded[columns]

    # Predict
    prediction = model.predict(input_encoded)[0]

    if prediction == 0:
        st.success("✅ Prediction: Good Credit Risk (0)")
    else:
        st.error("❌ Prediction: Bad Credit Risk (1)")
