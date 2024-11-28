import streamlit as st
import numpy as np
import pickle

with open('ada_boost.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


def predict(input_data):
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = model.predict(input_data_scaled)
    return prediction[0]


st.title("Breast Cancer Diagnosis Prediction")
st.write("This app predicts whether the diagnosis is malignant (M) or benign (B).")


st.write("Please enter the values for the following features:")
features = [st.number_input(f"Feature_{i}", min_value=0.0, step=0.01) for i in range(1, 31)]

if st.button("Predict"):
    prediction = predict(features)
    result = "Malignant (M)" if prediction == 1 else "Benign (B)"
    st.write(f"The predicted diagnosis is: {result}")
