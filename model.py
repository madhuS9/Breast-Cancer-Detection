import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import streamlit as st

file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]
data = pd.read_csv(file_path, header=None, names=column_names)
data.drop("ID", axis=1, inplace=True)
data["Diagnosis"] = data["Diagnosis"].map({"M": 1, "B": 0})


X = data.drop("Diagnosis", axis=1)
y = data["Diagnosis"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

ada_classifier = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.1,
    algorithm="SAMME",
    random_state=42
)
ada_classifier.fit(X_train_smote, y_train_smote)


st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🔍 Breast Cancer Tumor Classification")


st.sidebar.header("📝 Enter Tumor Features")
user_inputs = {}
for feature in X.columns:
    user_inputs[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ')}", min_value=0.0, value=1.0, format="%.2f")



st.header("🔮 Prediction")
if st.button("Predict Tumor Diagnosis"):
    user_input_array = np.array(list(user_inputs.values())).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input_array)

    
    prediction = ada_classifier.predict(user_input_scaled)
    result = "Malignant" if prediction[0] == 1 else "Benign"

    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 20px;">
        <h2 style="color: {'#FF6347' if prediction[0] == 1 else '#4682B4'};">
            The tumor is predicted to be: <strong>{result}</strong>
        </h2>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <p style="text-align: center; color: #6c757d;">
        Built with Streamlit | © 2024
    </p>
    """, unsafe_allow_html=True)
