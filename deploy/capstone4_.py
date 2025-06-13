import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Obesitas", layout="wide")
st.title("🧠 Prediksi Kategori Obesitas Berdasarkan Data Gaya Hidup")

# === 1. Upload Dataset ===
uploaded_file = st.file_uploader("Upload file CSV", type="csv")
if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai.")
    st.stop()

# Baca dataset
df = pd.read_csv(uploaded_file)

# Daftar kolom
kontinu_cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF']
integer_cols = ['FCVC', 'TUE']
biner_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
kategorikal_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']
target_col = 'NObeyesdad'

# === 2. Pembersihan Awal Dataset ===
# Ganti karakter aneh menjadi NaN
df.replace(['?', '', ' ', 'nan'], np.nan, inplace=True)

# Pastikan semua kolom numerik benar-benar numerik
for col in kontinu_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for col in integer_cols:
    df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')

# Konversi tipe kategorikal
for col in biner_cols + kategorikal_cols:
    df[col] = df[col].astype('category')

# Pisahkan fitur dan target
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline preprocessing
num_imputer = SimpleImputer(strategy='median')
int_imputer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_imputer, kontinu_cols),
        ('int', int_imputer, integer_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), kategorikal_cols),
        ('bin', OrdinalEncoder(), biner_cols),
    ])

# Jalankan preprocessing
X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

# Validasi hasil preprocessing
if np.isnan(X_train_processed).any():
    st.error("Masih ada nilai NaN di X_train_processed setelah preprocessing!")
    st.write("Contoh isi X_train_processed:")
    st.write(X_train_processed[:5])
    st.stop()

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

# Tombol pelatihan model
if st.button("🚀 Mulai Pelatihan Model"):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Akurasi: {acc:.2f}")
    st.text(classification_report(y_test, y_pred, target_names=le_y.classes_, zero_division=0))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le_y.classes_, yticklabels=le_y.classes_)
    st.pyplot(fig)
