import streamlit as st
import pickle

# Load trained model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌱 Crop Recommender System")
st.write("Enter soil and climate values to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=100.0)

# Predict button
if st.button("🌾 Recommend Crop"):
    sample = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(sample)
    st.success(f"✅ Recommended Crop: {prediction[0]}")

