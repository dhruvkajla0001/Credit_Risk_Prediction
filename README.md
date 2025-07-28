<!DOCTYPE html>
<html>
<head>
  <title>Credit Risk Prediction - README</title>
</head>
<body>

  <h1>Credit Risk Prediction Project</h1>

  <h2>Overview</h2>
  <p>
    This project is a machine learning application for predicting whether a loan applicant is likely to default or repay their loan.
    The model is trained on historical credit data and uses features such as income, employment history, loan amount, interest rate, and credit history.
    The final product includes a user-friendly <strong>Streamlit web app</strong> where users can input applicant details and get an instant prediction.
  </p>

  <h2>Key Features</h2>
  <ul>
    <li>Data preprocessing with handling missing values, encoding, scaling, and balancing (SMOTE).</li>
    <li>Model training with <strong>XGBoost</strong> and hyperparameter tuning.</li>
    <li>Streamlit dashboard for interactive predictions.</li>
    <li>Progress bar showing default risk probability.</li>
    <li>Easy deployment-ready structure.</li>
  </ul>

  <h2>Dataset</h2>
  <p>
    The dataset includes features such as:
    <ul>
      <li>Applicant age, income, employment length</li>
      <li>Loan amount, interest rate, loan-to-income ratio</li>
      <li>Credit history length and previous default history</li>
      <li>Home ownership, loan purpose, and loan grade (B–G)</li>
    </ul>
  </p>

  <h2>How It Works</h2>
  <ol>
    <li>Data is cleaned and missing values are imputed (median for numerical columns).</li>
    <li>Categorical variables are encoded using one-hot encoding.</li>
    <li>Numerical variables are scaled using <strong>StandardScaler</strong>.</li>
    <li>Class imbalance is handled with <strong>SMOTE</strong>.</li>
    <li>The XGBoost model is trained on the processed data and saved using <code>joblib</code>.</li>
    <li>The Streamlit app loads the model, accepts user inputs, scales them, and predicts default probability and class.</li>
  </ol>

  <h2>Running the Project</h2>
  <p>
    Steps to run the project:
  </p>
  <ol>
    <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
    <li>Ensure you have the files:
      <ul>
        <li><code>xgboost_credit_risk_model.pkl</code> (trained model)</li>
        <li><code>scaler.pkl</code> (trained scaler)</li>
        <li><code>credit_risk_train_resampled.csv</code> (feature structure)</li>
      </ul>
    </li>
    <li>Run the app: <code>streamlit run app.py</code></li>
    <li>Open the app in your browser and input applicant details to get predictions.</li>
  </ol>

  <h2>Sample High-Risk Input</h2>
  <p>Use this test applicant to trigger a high-risk prediction:</p>
  <pre>
{
  "person_age": 21,
  "person_income": 15000,
  "person_emp_length": 0.5,
  "loan_amnt": 30000,
  "loan_int_rate": 24.5,
  "loan_percent_income": 0.9,
  "cb_person_default_on_file": 1,
  "cb_person_cred_hist_length": 1,
  "person_home_ownership_other": 0,
  "person_home_ownership_own": 0,
  "person_home_ownership_rent": 1,
  "loan_intent_education": 0,
  "loan_intent_homeimprovement": 0,
  "loan_intent_medical": 0,
  "loan_intent_personal": 1,
  "loan_intent_venture": 0,
  "loan_grade_b": 0,
  "loan_grade_c": 0,
  "loan_grade_d": 0,
  "loan_grade_e": 0,
  "loan_grade_f": 0,
  "loan_grade_g": 1
}
  </pre>

  <h2>Dhruv Kajla</h2>
  <p>Developed as a practice project for building FinTech-focused machine learning skills.</p>

</body>
</html>
