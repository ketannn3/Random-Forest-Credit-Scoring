import streamlit as st
import pandas as pd
import joblib

# Load model and training column list
model = joblib.load("credit_random_forest.pkl")
columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Credit Risk Predictor")
st.title("🏦 German Credit Scoring Prediction")

# --- Input form ---
status = st.selectbox("Status", ["A11", "A12", "A13", "A14"])
duration = st.slider("Duration (months)", 4, 72, 12)
credit_history = st.selectbox("Credit History", ["A30", "A31", "A32", "A33", "A34"])
purpose = st.selectbox("Purpose", ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49"])
credit_amount = st.number_input("Credit Amount", 250, 20000, 1000)
savings = st.selectbox("Savings", ["A61", "A62", "A63", "A64", "A65"])
employment = st.selectbox("Employment", ["A71", "A72", "A73", "A74", "A75"])
installment_rate = st.slider("Installment Rate", 1, 4, 2)
personal_status = st.selectbox("Personal Status & Sex", ["A91", "A92", "A93", "A94"])
debtors = st.selectbox("Other Debtors", ["A101", "A102", "A103"])
residence = st.slider("Residence Duration", 1, 4, 2)
property = st.selectbox("Property", ["A121", "A122", "A123", "A124"])
age = st.slider("Age", 18, 75, 30)
installment_plan = st.selectbox("Installment Plans", ["A141", "A142", "A143"])
housing = st.selectbox("Housing", ["A151", "A152", "A153"])
existing_credits = st.slider("Existing Credits", 1, 4, 1)
job = st.selectbox("Job", ["A171", "A172", "A173", "A174"])
liable_people = st.slider("Liable People", 1, 2, 1)
telephone = st.selectbox("Telephone", ["A191", "A192"])
foreign_worker = st.selectbox("Foreign Worker", ["A201", "A202"])

# Predict button
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

    # 🔧 FIX: Align columns
    for col in columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[columns]

    prediction = model.predict(input_encoded)[0]

    if prediction == 0:
        st.success("✅ Prediction: Good Credit Risk (0)")
    else:
        st.error("❌ Prediction: Bad Credit Risk (1)")
