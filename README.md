# 🎯 Random Forest Credit Scoring

This project predicts whether a person is a good or bad credit risk using a Random Forest model trained on the German Credit dataset from UCI.

---

## 📊 Live Demo

👉 [Click here to try the Streamlit App](https://random-forest-credit-scoring-ky2jjykzybbvknwmoxtpal.streamlit.app/)

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app for user input and prediction |
| `credit_random_forest.pkl` | Trained Random Forest model |
| `Random_Forest_Credit_Scoring.ipynb` | Colab notebook with full training pipeline |
| `requirements.txt` | Libraries used for deployment |
| `README.md` | Project summary and instructions |

---

## ✅ Features

- Built with **Random Forest Classifier**
- Predicts whether an applicant is a **Good (0)** or **Bad (1)** credit risk
- Trained on [UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- Interactive user interface using **Streamlit**

---

## 📊 Model Accuracy

| Metric | Value |
|--------|-------|
| Accuracy | 74.5% |
| Recall (Good Credit) | ~91% |
| Recall (Bad Credit) | ~34% |

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
