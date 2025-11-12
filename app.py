# app.py
import streamlit as st
import joblib
import pandas as pd

# --- 1. LOAD THE SAVED PIPELINE ---
# Make sure 'churn_model_pipeline.joblib' is in the same folder or provide the correct path
try:
    model = joblib.load("model/churn_model_pipeline.joblib")
except FileNotFoundError:
    st.error("Model file (churn_model_pipeline.joblib) not found. Please place it in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model: {e}")
    st.stop()


# --- 2. DEFINE ALL MODEL COLUMNS ---
# This is the exact list you provided
all_model_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_Yes', 'InternetService_Fiber optic',
    'InternetService_No', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
    'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
    'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]


# --- 3. CREATE THE APP INTERFACE ---
st.title("Telco Customer Churn Predictor")
st.write("Enter customer details to predict the likelihood of churn. This interface is built to match the model's expected inputs.")

# Create columns for layout
col1, col2 = st.columns(2)

# --- 4. CREATE INPUT WIDGETS ---

with col1:
    st.header("Demographics & Billing")
    
    # Simple Binary & Numerical Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    tenure = st.slider("Tenure (Months)", 0, 72, 24)
    MonthlyCharges = st.slider("Monthly Charges ($)", 0.0, 120.0, 50.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)
    
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])

with col2:
    st.header("Contract & Services")
    
    # Contract (3 options -> 2 columns)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    
    # PaymentMethod (4 options -> 3 columns)
    PaymentMethod = st.selectbox("Payment Method", 
                                 ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
    
    # Phone & Internet Services
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"]) # Assumes "No phone service" was mapped to "No"
    
    # InternetService (3 options -> 2 columns)
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    # Add-on Services (Yes/No)
    # Assumes "No internet service" was mapped to "No"
    st.subheader("Add-on Services")
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])


# --- 5. THE PREDICTION LOGIC ---

# Button to make prediction
if st.button("Predict Churn", type="primary"):
    
    # 1. Create the input dictionary with default 0s
    input_data = {col: 0 for col in all_model_columns}

    # 2. Update dictionary with user's numerical inputs
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = MonthlyCharges
    input_data['TotalCharges'] = TotalCharges
    
    # 3. Update dictionary with user's binary (Yes/No or 1/0) inputs
    input_data['gender'] = 1 if gender == "Male" else 0
    input_data['SeniorCitizen'] = 1 if SeniorCitizen == "Yes" else 0
    input_data['Partner'] = 1 if Partner == "Yes" else 0
    input_data['Dependents'] = 1 if Dependents == "Yes" else 0
    input_data['PhoneService'] = 1 if PhoneService == "Yes" else 0
    input_data['PaperlessBilling'] = 1 if PaperlessBilling == "Yes" else 0

    # 4. Update dictionary with one-hot encoded inputs
    
    # Services (simple Yes/No)
    if MultipleLines == "Yes": input_data['MultipleLines_Yes'] = 1
    if OnlineSecurity == "Yes": input_data['OnlineSecurity_Yes'] = 1
    if OnlineBackup == "Yes": input_data['OnlineBackup_Yes'] = 1
    if DeviceProtection == "Yes": input_data['DeviceProtection_Yes'] = 1
    if TechSupport == "Yes": input_data['TechSupport_Yes'] = 1
    if StreamingTV == "Yes": input_data['StreamingTV_Yes'] = 1
    if StreamingMovies == "Yes": input_data['StreamingMovies_Yes'] = 1
        
    # InternetService (Base case is "DSL")
    if InternetService == "Fiber optic":
        input_data['InternetService_Fiber optic'] = 1
    elif InternetService == "No":
        input_data['InternetService_No'] = 1
        
    # Contract (Base case is "Month-to-month")
    if Contract == "One year":
        input_data['Contract_One year'] = 1
    elif Contract == "Two year":
        input_data['Contract_Two year'] = 1
        
    # PaymentMethod (Base case is "Bank transfer (automatic)")
    if PaymentMethod == "Credit card (automatic)":
        input_data['PaymentMethod_Credit card (automatic)'] = 1
    elif PaymentMethod == "Electronic check":
        input_data['PaymentMethod_Electronic check'] = 1
    elif PaymentMethod == "Mailed check":
        input_data['PaymentMethod_Mailed check'] = 1
    
    # 5. Convert dictionary to DataFrame
    # Using 'all_model_columns' ensures the column order is correct
    try:
        input_df = pd.DataFrame([input_data], columns=all_model_columns)
    except Exception as e:
        st.error(f"Error creating DataFrame: {e}. Check column list.")
        st.stop()

    # 6. Make prediction
    try:
        # Use predict_proba to get the probability
        churn_probability = model.predict_proba(input_df)[0][1]
        
        # Format the output
        st.subheader("Prediction Result")
        
        churn_prob_percent = churn_probability * 100
        
        if churn_probability > 0.5:
            # Display in a red "error" box
            st.error(f"Prediction: LIKELY TO CHURN (Probability: {churn_prob_percent:.2f}%)")
            st.progress(churn_prob_percent / 100)
        else:
            # Display in a green "success" box
            st.success(f"Prediction: Likely to Stay (Probability: {churn_prob_percent:.2f}%)")
            st.progress(churn_prob_percent / 100)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Input data sent to model:", input_df) # Debugging line