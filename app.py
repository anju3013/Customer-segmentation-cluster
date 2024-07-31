import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Function to load the models and encoders
def load_pickle(file_name):
    with open(file_name, "rb") as file:
        return pickle.load(file)

# Load the models and encoders
db = load_pickle("db_model.pkl")
pca = load_pickle("pca_model.pkl")
scaler = load_pickle("scaler.pkl")

default_le = load_pickle("default_le.pkl")
personal_loan_le = load_pickle("personal_loan_le.pkl")
housing_loan_le = load_pickle("housing_loan_le.pkl")
communication_type_oe = load_pickle("communication_type_oe.pkl")
education_oe = load_pickle("education_oe.pkl")
job_type_oe = load_pickle("job_type_oe.pkl")
marital_oe = load_pickle("marital_oe.pkl")

balance_skew = load_pickle("balance_skew.pkl")
last_contact_duration_skew = load_pickle("last_contact_duration_skew.pkl")

# Helper function to predict cluster
def predict(model, features):
    return model.fit_predict(features)[0]

st.title("Customer Segmentation Cluster Prediction")

st.header("Customer Input")
customer_age = st.number_input("Customer Age", min_value=15, max_value=90, step=1)

job_type_options = ["blue-collar", "management", "technician", "admin.", "services", "retired",
                    "self-employed", "entrepreneur", "unemployed", "housemaid", "student", "unknown"]
job_type = st.selectbox("Job Type", job_type_options)

marital_options = ["married", "single", "divorced", "unknown"]
marital = st.selectbox("Marital Status", marital_options)

education_options = ["secondary", "tertiary", "primary", "unknown"]
education = st.selectbox("Education", education_options)

options = ['yes', 'no']
default = st.selectbox("Default", options)

balance = st.text_input("Balance")
try:
    balance = float(balance)
except ValueError:
    balance = 0.0
    st.write("Please enter a valid number for Balance.")

housing_loan = st.selectbox("Housing Loan", options)
personal_loan = st.selectbox("Personal Loan", options)

communication_type_options = ["cellular", "unknown", "telephone"]
communication_type = st.selectbox("Communication Type", communication_type_options)

last_contact_duration = st.text_input("Last Contact Duration")
try:
    last_contact_duration = float(last_contact_duration)
except ValueError:
    last_contact_duration = 0.0
    st.write("Please enter a valid number for Last Contact Duration.")

num_contacts_prev_campaign = st.number_input("Number of Contacts in Previous Campaign", value=0, step=1, format="%d")

if st.button("Predict Cluster"):
    # Transform inputs
    job_type_transformed = job_type_oe.transform([[job_type]])[0]
    marital_transformed = marital_oe.transform([[marital]])[0]
    education_transformed = education_oe.transform([[education]])[0]
    default_transformed = default_le.transform([default])
    balance_transformed = balance_skew.transform([[balance]])[0]
    housing_loan_transformed = housing_loan_le.transform([housing_loan])
    personal_loan_transformed = personal_loan_le.transform([personal_loan])
    communication_type_transformed = communication_type_oe.transform([[communication_type]])[0]
    last_contact_duration_transformed = last_contact_duration_skew.transform([[last_contact_duration]])[0]

    features = np.hstack([customer_age,
                          job_type_transformed,
                          marital_transformed,
                          education_transformed,
                          default_transformed,
                          balance_transformed,
                          housing_loan_transformed,
                          personal_loan_transformed,
                          communication_type_transformed,
                          last_contact_duration_transformed]).reshape(1, -1)

    # Create a DataFrame to pass to the scaler with correct feature names
    feature_names = [
        "customer_age",
        "job_type",
        "marital",
        "education",
        "default",
        "balance",
        "housing_loan",
        "personal_loan",
        "communication_type",
        "last_contact_duration"
    ]

    features_df = pd.DataFrame(features, columns=feature_names)

    features_scaled = scaler.transform(features_df)
    features_pca = pca.transform(features_scaled)
    

    st.write(f"Transformed Features: {features_pca}")

    cluster = predict(db, features_pca)

    st.write(f"Predicted Cluster: {cluster}")

    if cluster == -1:
        st.write("The input data point is classified as noise by DBSCAN. Try adjusting the input values or check the DBSCAN parameters.")
