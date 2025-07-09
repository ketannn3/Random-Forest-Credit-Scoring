import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("credit_random_forest.pkl")

st.set_page_config(page_title="Credit Risk Predictor")
st.title("🏦 German Credit Risk Prediction (Simplified Demo)")

# --- Only asking numeric inputs that match the model
st.write("⚠️ This demo works only with preprocessed numeric inputs the model understands.")

duration = st.number_input("Duration (months)", 1, 72, 12)
credit_amount = st.number_input("Credit Amount", 250, 20000, 1000)
age = st.slider("Age", 18, 75, 30)
existing_credits = st.slider("Existing Credits", 1, 4, 1)

# You can add more numeric features here if you like

if st.button("🔮 Predict"):
    # Basic demo inputs — model expects 48 features, so we'll pad rest with 0
    input_list = [duration, credit_amount, age, existing_credits]
    input_list += [0] * (48 - len(input_list))  # padding to match model input size

    input_df = pd.DataFrame([input_list])

    try:
        prediction = model.predict(input_df)[0]
        if prediction == 0:
            st.success("✅ Prediction: Good Credit Risk (0)")
        else:
            st.error("❌ Prediction: Bad Credit Risk (1)")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
