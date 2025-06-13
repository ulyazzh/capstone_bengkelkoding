import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Judul Aplikasi
st.set_page_config(page_title="Prediksi Tingkat Obesitas", layout="centered")
st.title("📊 Prediksi Tingkat Obesitas")
st.markdown("Masukkan detail Anda untuk memprediksi tingkat obesitas.")

# Load model dan encoder
model = joblib.load("obesity_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Input Form
gender = st.selectbox("Jenis Kelamin", options=["Female", "Male"])
age = st.slider("Usia", 10, 100, 30)
height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0)
family_history = st.selectbox("Riwayat Keluarga Obesitas?", ["no", "yes"])
favc = st.selectbox("Sering makan makanan kalori tinggi?", ["no", "yes"])
fcvc = st.slider("Frekuensi makan sayuran per hari", 0, 5, 2)
ncp = st.slider("Jumlah makan besar per hari", 1, 5, 3)
caec = st.selectbox("Makan camilan di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Apakah merokok?", ["no", "yes"])
ch2o = st.slider("Minum air per hari (liter)", 0.0, 5.0, 2.0)
scc = st.selectbox("Memantau asupan kalori harian?", ["no", "yes"])
faf = st.slider("Aktivitas fisik per minggu", 0.0, 5.0, 1.0)
tue = st.slider("Waktu layar/hari (jam)", 0, 10, 3)
calc = st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

if st.button("🔍 Prediksi"):
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })

    # Encode data input
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    prediction_label = label_encoders['NObeyesdad'].inverse_transform([prediction])[0]

    st.success(f"Prediksi Tingkat Obesitas: **{prediction_label}**")