import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('ohe_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="Churn Predictor", page_icon="🔍", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Input
st.sidebar.header("Enter Customer Details")

geography = st.sidebar.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 52, 30)
balance = st.sidebar.number_input('Balance')
credit_score = st.sidebar.number_input('Credit Score')
estimated_salary = st.sidebar.number_input('Estimated Salary')
tenure = st.sidebar.slider('Tenure', 0, 10, 3)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])

# Prepare input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],    
})

geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_enc_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_enc_df], axis=1)

input_data_scaled = scalar.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display Result
st.markdown("## 🔍 Prediction Result")
st.metric(label="Churn Probability", value=f"{prediction_prob:.2%}")

if prediction_prob > 0.5:
    st.error("⚠️ The customer is **likely to churn**.")
else:
    st.success("✅ The customer is **not likely to churn**.")

# Optional: show input data
with st.expander("🔎 Show processed input data"):
    st.dataframe(input_data)
