import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
from sklearn.preprocessing import LabelEncoder

# Load pre-trained Random Forest model from a compressed file
def load_model():
    with gzip.open(r'app/best_random_forest_model_compressed.pkl.gz', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Encoders for categorical features
def get_encoders():
    encoders = {
        'job': LabelEncoder(),
        'marital': LabelEncoder(),
        'education': LabelEncoder(),
        'default': LabelEncoder(),
        'housing': LabelEncoder(),
        'loan': LabelEncoder(),
        'contact': LabelEncoder(),
        'month': LabelEncoder(),
        'poutcome': LabelEncoder()
    }

    # Fit encoders with expected categories
    encoders['job'].fit(["unknown", "admin.", "unemployed", "management", "housemaid", "entrepreneur", "student",
                         "blue-collar", "self-employed", "retired", "technician", "services"])
    encoders['marital'].fit(["married", "divorced", "single"])
    encoders['education'].fit(["unknown", "secondary", "primary", "tertiary"])
    encoders['default'].fit(["yes", "no"])
    encoders['housing'].fit(["yes", "no"])
    encoders['loan'].fit(["yes", "no"])
    encoders['contact'].fit(["unknown", "telephone", "cellular"])
    encoders['month'].fit(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    encoders['poutcome'].fit(["unknown", "other", "failure", "success"])

    return encoders

encoders = get_encoders()

# Preprocess input data
def preprocess_input_data(data):
    for col, encoder in encoders.items():
        if col in data:
            data[col] = encoder.transform(data[col])
    return data

# Page 1: Introduction and background
def page1():
    st.title("Bank Term Deposit Prediction App")
    st.image(r"app\Marketing-campaign-image-for-article-4939049309430393093.jpg", use_column_width=True)

    st.header("Introduction")
    st.write("""
    This application predicts whether a client will subscribe to a term deposit based on key demographic, financial, and campaign-related features. 
    By leveraging data from past campaigns, financial institutions can identify potential customers and optimize their marketing strategies.
    """)

    st.header("How it Works")
    st.write("""
    The app uses a Random Forest classification model trained on banking campaign data to make predictions. 
    Users can input client details or upload datasets for batch predictions.
    """)

# Page 2: Single Prediction
def page2():
    st.title("Single Client Prediction")
    st.image(r"app\Finance-and-Retail-Banking-Blog-Post.jpg", use_column_width=True)

    st.header("Enter Client Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        job = st.selectbox("Job", encoders['job'].classes_)
        marital = st.selectbox("Marital Status", encoders['marital'].classes_)
        education = st.selectbox("Education", encoders['education'].classes_)

    with col2:
        default = st.selectbox("Has Credit in Default?", encoders['default'].classes_)
        balance = st.number_input("Yearly Average Balance (Euros)", min_value=0, max_value=100000, step=100)
        housing = st.selectbox("Housing Loan", encoders['housing'].classes_)
        loan = st.selectbox("Personal Loan", encoders['loan'].classes_)

    contact = st.selectbox("Contact Type", encoders['contact'].classes_)
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, step=1)
    month = st.selectbox("Last Contact Month", encoders['month'].classes_)
    duration = st.number_input("Duration of Last Call (seconds)", min_value=0, step=1)
    campaign = st.number_input("Number of Contacts During Campaign", min_value=1, step=1)
    pdays = st.number_input("Days Since Last Contact (-1 for no contact)", min_value=-1, max_value=500, step=1)
    previous = st.number_input("Number of Contacts Before Campaign", min_value=0, step=1)
    poutcome = st.selectbox("Outcome of Previous Campaign", encoders['poutcome'].classes_)

    # Predict
    if st.button("Predict Subscription"):
        input_data = pd.DataFrame({
            "age": [age], "job": [job], "marital": [marital], "education": [education], 
            "default": [default], "balance": [balance], "housing": [housing], "loan": [loan], 
            "contact": [contact], "day": [day], "month": [month], "duration": [duration], 
            "campaign": [campaign], "pdays": [pdays], "previous": [previous], "poutcome": [poutcome]
        })

        # Preprocess and predict
        input_data = preprocess_input_data(input_data)
        prediction = model.predict(input_data)
        result = "Subscribed" if prediction[0] == 1 else "Not Subscribed"
        st.write(f"Prediction: {result}")

# Page 3: Batch Prediction
def page3():
    st.title("Batch Prediction")
    st.write("Upload a dataset to predict term deposit subscriptions for multiple clients.")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", data.head())

        # Preprocess and predict
        data = preprocess_input_data(data)
        predictions = model.predict(data)
        data['Subscription Prediction'] = ["Subscribed" if pred == 1 else "Not Subscribed" for pred in predictions]
        st.write("Prediction Results:")
        st.write(data)

        # Download button
        st.download_button(
            label="Download Predictions",
            data=data.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

# Sidebar navigation
pages = {
    "Home": page1,
    "Single Prediction": page2,
    "Batch Prediction": page3
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Display selected page
pages[selected_page]()
