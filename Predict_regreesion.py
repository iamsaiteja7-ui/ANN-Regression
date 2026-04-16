import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

# Base directory — same folder as this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scalers
model        = load_model(os.path.join(BASE_DIR, 'ann_regression_model.keras'))
with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, 'target_scaler.pkl'), 'rb') as f:
    target_scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, 'one_hot_encoder.pkl'), 'rb') as f:
    onehot_encoder = pickle.load(f)

# Streamlit app
st.title("House Price Prediction (ANN Regression)")

area             = float(st.text_input("Area (sqft)", value="1500"))
bedrooms         = int(st.text_input("Bedrooms", value="3"))
bathrooms        = int(st.text_input("Bathrooms", value="2"))
age_years        = int(st.text_input("Age of House (years)", value="5"))
distance         = int(st.text_input("Distance from City (km)", value="10"))
parking          = int(st.text_input("Parking Spots", value="1"))
furnishing       = st.selectbox("Furnishing", options=['Furnished', 'Semi-Furnished', 'Unfurnished'])
location         = st.selectbox("Location", options=['Urban', 'Suburban', 'Rural'])

# Build input DataFrame
input_data = {
    'area_sqft':             [area],
    'bedrooms':              [bedrooms],
    'bathrooms':             [bathrooms],
    'age_years':             [age_years],
    'distance_from_city_km': [distance],
    'parking':               [parking],
    'furnishing':            [furnishing],
    'location':              [location]
}
input_df = pd.DataFrame(input_data)

# Encode categorical columns
enc    = onehot_encoder.transform(input_df[['furnishing', 'location']])
enc_df = pd.DataFrame(enc.toarray(), columns=onehot_encoder.get_feature_names_out(['furnishing', 'location']))
input_final  = pd.concat([input_df.drop(['furnishing', 'location'], axis=1), enc_df], axis=1)

# Scale and predict
input_scaled = scaler.transform(input_final)
pred_scaled  = model.predict(input_scaled)
pred_actual  = target_scaler.inverse_transform(pred_scaled)

st.write("### Predicted House Price")
st.success(f"₹ {pred_actual.flatten()[0]:,.0f}")




