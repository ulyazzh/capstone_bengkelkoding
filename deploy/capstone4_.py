
# capstone4_streamlit.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Prediksi Obesitas", layout="wide")
st.title("📊 Prediksi Kategori Obesitas Berdasarkan Data Gaya Hidup")

# === 1. Load Dataset ===
uploaded_file = st.file_uploader("Upload file CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Awal")
    st.dataframe(df.head())

    # Konversi tipe data
    kontinu_cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF']
    integer_cols = ['FCVC', 'TUE']
    biner_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    kategorikal_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS', 'NObeyesdad']

    for col in kontinu_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in integer_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
    for col in kategorikal_cols + biner_cols:
        df[col] = df[col].astype('category')

    # Visualisasi kelas
    st.subheader("🔍 Distribusi Kelas Obesitas")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='NObeyesdad', order=df['NObeyesdad'].value_counts().index, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Imputasi dan Preprocessing
    imputer_num = SimpleImputer(strategy='median')
    df[kontinu_cols] = imputer_num.fit_transform(df[kontinu_cols])
    imputer_int = SimpleImputer(strategy='most_frequent')
    df[integer_cols] = imputer_int.fit_transform(df[integer_cols])
    for col in kategorikal_cols + biner_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[[col]] = imputer_cat.fit_transform(df[[col]])

    # Label Encoding
    le = LabelEncoder()
    for col in kategorikal_cols + biner_cols:
        df[col] = le.fit_transform(df[col])

    # SMOTE
    from sklearn.preprocessing import LabelEncoder
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    # Pastikan y numerik
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    # SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    # Skala data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)

    # Model
    st.subheader("🧠 Training dan Evaluasi Model")
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=10)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.markdown(f"#### {name}")
        st.write(f"**Akurasi:** {acc:.2f}")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)
