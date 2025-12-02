import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧠 Feature Selection & Feature Importance")

# ======================================================
# Cek apakah hasil preprocessing sudah ada
# ======================================================
required_keys = ["X_train", "y_train", "X_test", "y_test"]
if not all(k in st.session_state for k in required_keys):
    st.error(
        "Data belum dipreprocessing. Silakan buka halaman **Preprocessing** terlebih dahulu "
        "agar X_train, y_train, X_test, dan y_test tersimpan di session_state."
    )
    st.stop()

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

st.write("Shape X_train:", X_train.shape)
st.write("Jumlah fitur:", X_train.shape[1])

# ======================================================
# 1. FEATURE IMPORTANCE – RANDOM FOREST
# ======================================================
st.markdown("---")
st.subheader("1. Feature Importance – Random Forest")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

top_n = 15
st.write(f"Top {top_n} fitur berdasarkan Random Forest:")
st.dataframe(rf_importance.head(top_n))

fig_rf = px.bar(
    rf_importance.head(top_n).sort_values("importance"),
    x="importance",
    y="feature",
    orientation="h",
    labels={"importance": "Importance", "feature": "Fitur"}
)
st.plotly_chart(fig_rf, use_container_width=True)

# Highlight fitur penting (level variabel asli)
important_vars = ["age", "avg_glucose_level", "hypertension",
                  "heart_disease", "smoking_status"]
st.info(
    "Dari Random Forest, terlihat bahwa fitur yang sering muncul dengan importance tinggi "
    f"antara lain: **{', '.join(important_vars)}** (baik dalam bentuk asli maupun dummy hasil encoding)."
)

# ======================================================
# 2. FEATURE IMPORTANCE – LOGISTIC REGRESSION COEFFICIENTS
# ======================================================
st.markdown("---")
st.subheader("2. Feature Importance – Logistic Regression Coefficients")

logreg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)
logreg.fit(X_train, y_train)

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coef": logreg.coef_[0]
})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)

st.write(f"Top {top_n} fitur berdasarkan nilai absolut koefisien Logistic Regression:")
st.dataframe(coef_df.head(top_n))

fig_lr = px.bar(
    coef_df.head(top_n).sort_values("abs_coef"),
    x="abs_coef",
    y="feature",
    orientation="h",
    labels={"abs_coef": "|Koefisien|", "feature": "Fitur"}
)
st.plotly_chart(fig_lr, use_container_width=True)

st.caption(
    "Semakin besar nilai absolut koefisien, semakin kuat pengaruh fitur tersebut "
    "terhadap probabilitas stroke (dengan asumsi fitur lain konstan)."
)

# ======================================================
# 3. CORRELATION ANALYSIS (NUMERIC + TARGET)
# ======================================================
st.markdown("---")
st.subheader("3. Analisis Korelasi (Pearson) untuk Fitur Numerik")

# Load ulang data mentah hanya untuk melihat korelasi numerik + target
df_corr = pd.read_csv("healthcare-dataset-stroke-data.csv")
df_corr = df_corr.drop(columns=["id"])

num_cols = ["age", "avg_glucose_level", "bmi"]
corr_cols = num_cols + ["stroke"]

corr = df_corr[corr_cols].corr(method="pearson")

fig_corr, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
plt.tight_layout()
st.pyplot(fig_corr)

st.caption(
    "Korelasi Pearson antara fitur numerik dan target `stroke`. "
    "Nilai korelasi yang lebih tinggi (positif maupun negatif) "
    "mengindikasikan hubungan linear yang lebih kuat."
)

# ======================================================
# 4. (OPSIONAL) PCA UNTUK FITUR NUMERIK
# ======================================================
st.markdown("---")
st.subheader("4. (Opsional) PCA pada Fitur Numerik")

use_pca = st.checkbox("Aktifkan PCA untuk fitur numerik", value=False)

if use_pca:
    # Ambil hanya kolom numerik dari X_train (setelah scaling)
    numeric_in_X = [c for c in X_train.columns if any(c.startswith(n) for n in num_cols)]
    X_num = X_train[numeric_in_X]

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_num)

    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "stroke": y_train.values
    })

    st.write("Proporsi varian yang dijelaskan oleh masing-masing komponen:")
    expl = pca.explained_variance_ratio_
    st.write(f"PC1: {expl[0]:.3f}, PC2: {expl[1]:.3f}, Total: {(expl[0]+expl[1]):.3f}")

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=pca_df["stroke"].astype(str),
        opacity=0.7,
        labels={"color": "Stroke"},
        title="PCA (2 Komponen Utama) – Fitur Numerik"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# ======================================================
# 5. RANGKUMAN TEKS
# ======================================================
st.markdown("---")
st.subheader("5. Ringkasan Feature Selection")

st.markdown(
    """
Berdasarkan hasil **Random Forest feature importance**, **koefisien Logistic Regression**, 
dan korelasi numerik, terlihat bahwa beberapa variabel yang **paling berpengaruh** 
terhadap kejadian stroke adalah:

- `age` → risiko stroke meningkat pada usia lanjut  
- `avg_glucose_level` → kadar glukosa darah yang tinggi terkait dengan peningkatan risiko  
- `hypertension` → adanya hipertensi memberi kontribusi besar terhadap kemungkinan stroke  
- `heart_disease` → riwayat penyakit jantung menjadi salah satu faktor penting  
- `smoking_status` → kebiasaan merokok (current / former) cenderung meningkatkan risiko stroke  

Variabel-variabel ini akan menjadi fokus utama dalam interpretasi model 
dan dapat disorot secara khusus pada bagian pembahasan di laporan.
"""
)
