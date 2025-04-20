import streamlit as st
import pandas as pd
import joblib

# Load model artifacts
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

def user_input_features():
    Age = st.number_input('Age', 18, 100)
    Sex = st.selectbox('Sex', ['Male', 'Female'])
    ChestPainType = st.selectbox('Chest Pain Type ', ['Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic]', 'Typical Angina'])
    RestingBP = st.number_input('Resting BP [mm Hg]', 50, 200)
    Cholesterol = st.number_input('Cholesterol: serum cholesterol [mm/dl]', 0, 600)
    FastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dl [1: if FastingBS > 120 mg/dl, 0: otherwise]', [0, 1])
    RestingECG = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
    MaxHR = st.number_input('Max Heart Rate [Numeric value between 60 and 202]', 60, 220)
    ExerciseAngina = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    Oldpeak = st.number_input('Oldpeak = ST [Numeric value measured in depression]', 0.0, 10.0, step=0.1)
    ST_Slope = st.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])

    input_dict = {
        'Age': Age,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }

    return pd.DataFrame([input_dict])

def preprocess_input(input_df):
    input_encoded = pd.get_dummies(input_df, drop_first=False)

    # Add any missing columns with 0
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Ensure correct order
    input_encoded = input_encoded[model_columns]

    return input_encoded

def main():
    st.title("Heart Disease Prediction App 🫀")
    input_df = user_input_features()

    if st.button("Predict"):
        processed_input = preprocess_input(input_df)
        scaled_input = scaler.transform(processed_input)
        prediction = model.predict(scaled_input)

        if prediction[0] == 1:
            st.error("⚠️ High risk of Heart Disease")
        else:
            st.success("✅ Low risk of Heart Disease")

if __name__ == '__main__':
    main()
