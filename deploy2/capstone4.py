import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

# STREAMLIT START
st.title("Prediksi Kategori Obesitas 🚀")
uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Data Awal")
    st.write(df.head())

    # === Preprocessing ===
    kontinu_cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF']
    integer_cols = ['FCVC', 'TUE']
    biner_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    cat_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS', 'NObeyesdad']

    for col in kontinu_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in integer_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')

    imputer_num = SimpleImputer(strategy='median')
    df[kontinu_cols] = imputer_num.fit_transform(df[kontinu_cols])

    imputer_int = SimpleImputer(strategy='most_frequent')
    df[integer_cols] = imputer_int.fit_transform(df[integer_cols])

    for col in cat_cols + biner_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[[col]] = imputer_cat.fit_transform(df[[col]])

    df.drop_duplicates(inplace=True)

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    y = df['NObeyesdad']
    X = df.drop('NObeyesdad', axis=1)

    numeric_cols = kontinu_cols + integer_cols
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=10)
    }

    st.subheader("📊 Hasil Evaluasi Model")

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results.append({
            "Model": name,
            "Akurasi": acc,
            "Presisi": prec,
            "Recall": rec,
            "F1-Score": f1
        })

        with st.expander(f"🔍 Detail: {name}"):
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred, zero_division=0))
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f'Confusion Matrix - {name}')
            st.pyplot(fig)

    results_df = pd.DataFrame(results)
    st.dataframe(results_df.set_index("Model"))

    st.subheader("📈 Grafik Perbandingan")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Model", y="value", hue="variable",
                data=pd.melt(results_df, id_vars=["Model"]),
                ax=ax)
    ax.set_title("Perbandingan Performa Model")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
