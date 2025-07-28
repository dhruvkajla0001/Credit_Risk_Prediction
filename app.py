import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# Load Model, Scaler, and Features
# ================================
model = joblib.load("xgboost_credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
train_df = pd.read_csv("credit_risk_train_resampled.csv")
feature_columns = train_df.drop('loan_status', axis=1).columns  # Model input features

# ================================
# Prediction Function
# ================================
def predict_borrower_risk(applicant_data):
    applicant_df = pd.DataFrame([applicant_data])
    for col in feature_columns:
        if col not in applicant_df.columns:
            applicant_df[col] = 0
    applicant_df = applicant_df[feature_columns]
    applicant_scaled = scaler.transform(applicant_df)
    prob_default = model.predict_proba(applicant_scaled)[0][1]
    prediction = model.predict(applicant_scaled)[0]
    return prob_default, prediction

# ================================
# Page Config & Header
# ================================
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="centered")
st.markdown(
    """
    <style>
    .main { background-color: #f7f8fa; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("💳 Credit Risk Prediction Dashboard")
st.markdown("Predict whether a loan applicant is **likely to default or repay** using a trained AI model.")

st.markdown("---")
st.subheader("Applicant Information")

# ================================
# User Inputs
# ================================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=1000, max_value=500000, value=60000, step=1000)
    emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    credit_hist = st.number_input("Credit History (years)", min_value=0, max_value=50, value=6)

with col2:
    loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=10000, step=500)
    interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=10.5, step=0.1)
    loan_percent_income = st.number_input("Loan % of Income", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    default_flag = st.radio("Has Previous Default?", ["No", "Yes"])
    default_flag = 1 if default_flag == "Yes" else 0

# Dropdowns
home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "OTHER"])
loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
loan_grade = st.selectbox("Loan Grade", ["B", "C", "D", "E", "F", "G"])

# Prepare applicant dict
applicant_data = {
    'person_age': age,
    'person_income': income,
    'person_emp_length': emp_length,
    'loan_amnt': loan_amount,
    'loan_int_rate': interest_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_default_on_file': default_flag,
    'cb_person_cred_hist_length': credit_hist,
    # One-hot encoding
    'person_home_ownership_other': 1 if home_ownership == "OTHER" else 0,
    'person_home_ownership_own': 1 if home_ownership == "OWN" else 0,
    'person_home_ownership_rent': 1 if home_ownership == "RENT" else 0,
    'loan_intent_education': 1 if loan_intent == "EDUCATION" else 0,
    'loan_intent_homeimprovement': 1 if loan_intent == "HOMEIMPROVEMENT" else 0,
    'loan_intent_medical': 1 if loan_intent == "MEDICAL" else 0,
    'loan_intent_personal': 1 if loan_intent == "PERSONAL" else 0,
    'loan_intent_venture': 1 if loan_intent == "VENTURE" else 0,
    'loan_grade_b': 1 if loan_grade == "B" else 0,
    'loan_grade_c': 1 if loan_grade == "C" else 0,
    'loan_grade_d': 1 if loan_grade == "D" else 0,
    'loan_grade_e': 1 if loan_grade == "E" else 0,
    'loan_grade_f': 1 if loan_grade == "F" else 0,
    'loan_grade_g': 1 if loan_grade == "G" else 0,
}

# ================================
# Prediction Button & Result
# ================================
if st.button("🔍 Predict Credit Risk"):
    prob, pred = predict_borrower_risk(applicant_data)

    st.markdown("---")
    st.subheader("Prediction Result")

    # Risk Meter (progress bar)
    st.write("### Default Risk Probability")
    st.progress(float(prob))

    if pred == 1:
        st.error(f"⚠️ **High Risk of Default!**\nPredicted Probability: {prob:.2%}")
    else:
        st.success(f"✅ **Likely to Repay!**\nPredicted Probability of Default: {prob:.2%}")

    st.markdown("---")
    st.caption("This prediction is based on historical data using a trained XGBoost model.")
