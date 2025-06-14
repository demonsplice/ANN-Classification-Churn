import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model

# Load the model and encoders
model = load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('One_hot_encoder_geo_.pkl','rb') as file:
    One_hot_encoder_geo_ = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title('Customer Churn Prediction')

# Input fields
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=40)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance', min_value=0.0, value=60000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

if st.button('Predict'):
    # Prepare input
    sample = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
    sample_df = pd.DataFrame([sample])

    # Encode 'Gender'
    sample_df['Gender'] = label_encoder_gender.transform(sample_df['Gender'])

    # One-hot encode 'Geography'
    geo_encoded = One_hot_encoder_geo_.transform(sample_df[['Geography']]).toarray()
    geo_feature_names = One_hot_encoder_geo_.get_feature_names_out(['Geography'])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_feature_names)

    # Drop original 'Geography' and concatenate encoded columns
    sample_df = sample_df.drop('Geography', axis=1)
    sample_df = pd.concat([sample_df, geo_df], axis=1)

    # Scale features
    sample_scaled = scaler.transform(sample_df)

    # Predict
    prediction = model.predict(sample_scaled)
    pred_class = (prediction > 0.5).astype(int)[0][0]

    st.write(f"Prediction probability: {prediction[0][0]:.2f}")
    if pred_class == 1:
        st.success("The customer is likely to churn.")
    else:
        st.info("The customer is not likely to churn.")



