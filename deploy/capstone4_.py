import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Library untuk ML
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Untuk SMOTE
from imblearn.over_sampling import SMOTE

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Obesitas", layout="wide")
st.title("🧠 Prediksi Kategori Obesitas Berdasarkan Data Gaya Hidup")

# === 1. Upload Dataset ===
uploaded_file = st.file_uploader("Upload file CSV", type="csv")
if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📂 Dataset Awal")
st.dataframe(df.head())

# Tentukan jenis kolom
kontinu_cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF']
integer_cols = ['FCVC', 'TUE']
biner_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
kategorikal_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']

target_col = 'NObeyesdad'

# Konversi tipe data
for col in kontinu_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for col in integer_cols:
    df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
for col in biner_cols + kategorikal_cols:
    df[col] = df[col].astype('category')

# Visualisasi distribusi target
st.subheader("📊 Distribusi Kelas Obesitas")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x=target_col, order=df[target_col].value_counts().index, palette="viridis", ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), kontinu_cols),
        ('int', SimpleImputer(strategy='most_frequent'), integer_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), kategorikal_cols),
        ('bin', OrdinalEncoder(), biner_cols),
    ])

# Pisahkan fitur dan target
X = df.drop(columns=[target_col])
y = df[target_col]

# Label encode target
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline preprocessing
scaler = StandardScaler()
X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# Validasi input sebelum SMOTE
if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
    st.error("❌ Data X_train_scaled masih mengandung NaN atau Inf!")
    st.write("Jumlah NaN:", np.isnan(X_train_scaled).sum())
    st.write("Jumlah Inf:", np.isinf(X_train_scaled).sum())
    st.stop()

# Tambahkan info debugging
st.write("✅ X_train_scaled siap digunakan untuk SMOTE")

# SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

# Tombol pelatihan model
if st.button("🚀 Mulai Pelatihan Model"):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=10)
    }

    for name, model in models.items():
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        st.markdown(f"### 🧪 Model: **{name}**")
        st.success(f"**Akurasi:** {acc:.2f}")
        st.text(classification_report(y_test, y_pred, target_names=le_y.classes_, zero_division=0))

        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=le_y.classes_, yticklabels=le_y.classes_)
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        st.pyplot(fig_cm)
