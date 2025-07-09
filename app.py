import streamlit as st
import pandas as pd
import joblib

model = joblib.load("credit_random_forest.pkl")

st.set_page_config(page_title="Credit Risk Predictor")
st.title("🏦 Simplified Credit Risk Prediction (Safe Fields)")

# Safe numeric and manually encodable inputs
duration = st.number_input("Duration (months)", 1, 72, 12)
credit_amount = st.number_input("Credit Amount", 250, 20000, 1000)
age = st.slider("Age", 18, 75, 30)
existing_credits = st.slider("Existing Credits", 1, 4, 1)
installment_rate = st.slider("Installment Rate", 1, 4, 2)
residence_duration = st.slider("Residence Duration (Years)", 1, 4, 2)
liable_people = st.slider("Number of Liable People", 1, 2, 1)

# Categorical inputs (encoded manually)
foreign_worker = st.selectbox("Foreign Worker", ["Yes (A201)", "No (A202)"])
telephone = st.selectbox("Has Telephone?", ["No (A191)", "Yes (A192)"])

# Encoding manually
foreign_worker_val = 1 if foreign_worker == "Yes (A201)" else 0
telephone_val = 1 if telephone == "Yes (A192)" else 0

if st.button("🔮 Predict"):
    input_list = [
        duration,
        credit_amount,
        age,
        existing_credits,
        installment_rate,
        residence_duration,
        liable_people,
        telephone_val,
        foreign_worker_val
    ]

    # pad to 48 features
    input_list += [0] * (48 - len(input_list))
    input_df = pd.DataFrame([input_list])

    try:
        prediction = model.predict(input_df)[0]
        if prediction == 0:
            st.success("✅ Prediction: Good Credit Risk (0)")
        else:
            st.error("❌ Prediction: Bad Credit Risk (1)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
