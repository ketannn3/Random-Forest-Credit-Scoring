import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("credit_random_forest.pkl")

st.title("🏦 Credit Scoring with Random Forest")
st.write("Enter customer details to predict credit risk (0 = Good, 1 = Bad)")

# Input form
status = st.selectbox("Status of checking account", ["<0", "0<=X<200", ">=200", "no checking"])
duration = st.slider("Duration (in months)", 4, 72, 24)
credit_history = st.selectbox("Credit history", ["no credits", "all paid", "existing paid", "delayed", "critical"])
purpose = st.selectbox("Purpose", ["car", "furniture", "radio/tv", "education", "business", "repair", "vacation"])
amount = st.number_input("Credit Amount", 250, 20000, step=100)

# Dummy simple input (we won't encode 20 full features here)
if st.button("Predict"):
    st.warning("This is a demo — model expects encoded values! Full form needs 20 preprocessed features.")
    input_data = np.zeros((1, model.n_features_in_))  # Dummy input
    prediction = model.predict(input_data)
    result = "Good Credit (0)" if prediction[0] == 0 else "Bad Credit (1)"
    st.success(f"Prediction: {result}")
