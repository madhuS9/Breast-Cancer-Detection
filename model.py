import streamlit as st
import pickle
import numpy as np

with open('ada_boost.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Breast Cancer Diagnosis Prediction")
st.sidebar.header('Input Features')

def user_input_features():
    features = []
    for i in range(1, 18):
        feature_value = st.sidebar.slider(f'Feature_{i}', min_value=0.0, max_value=10.0, step=0.1)
        features.append(feature_value)
    return np.array(features).reshape(1, -1)


input_features = user_input_features()
scaled_input = scaler.transform(input_features)
prediction = model.predict(scaled_input)[0]  # Get the single prediction

if prediction == 'M':
    st.write("🔴 The diagnosis is **Malignant**. Please consult a healthcare professional.")
else:
    st.write("🟢 The diagnosis is **Benign**. The tumor is not cancerous.")
