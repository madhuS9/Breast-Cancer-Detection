import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to make predictions
def predict(input_data):
    # Transform the input data using the loaded scaler
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    # Get prediction from the model
    prediction = model.predict(input_data_scaled)
    return prediction

# Streamlit interface
st.title("Breast Cancer Diagnosis Prediction")
st.write("This app predicts whether the diagnosis is malignant (M) or benign (B).")

# User inputs
features = []
st.write("Please enter the values for the following features:")
for i in range(1, 8):
    feature_value = st.number_input(f"Feature_{i}", min_value=0.0, step=0.01)
    features.append(feature_value)

if st.button("Predict"):
    # Get prediction when button is pressed
    prediction = predict(features)
    st.write(f"The predicted diagnosis is: {'Malignant' if prediction == 'M' else 'Benign'}")
