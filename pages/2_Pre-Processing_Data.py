import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("🧼 Preprocessing Data Stroke Prediction")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    return df

df_raw = load_data()

st.markdown("### 0. Data Mentah")
st.dataframe(df_raw.head())
st.write(f"Shape awal: **{df_raw.shape[0]} baris × {df_raw.shape[1]} kolom**")

# ======================
# 1. DROP KOLOM ID
# ======================
st.markdown("---")
st.subheader("1. Drop Kolom Tidak Relevan (`id`)")

df = df_raw.drop(columns=["id"])
st.write(f"Shape setelah drop `id`: **{df.shape[0]} baris × {df.shape[1]} kolom**")

# ======================
# 2. HANDLING MISSING VALUES
# ======================
st.markdown("---")
st.subheader("2. Handling Missing Values")

num_cols = ["age", "avg_glucose_level", "bmi"]
cat_cols = ["gender", "ever_married", "work_type",
            "Residence_type", "smoking_status"]
target_col = "stroke"

st.write("Missing value **sebelum** imputasi:")
st.write(df.isnull().sum())

# numerik -> median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# kategorik -> modus
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

st.write("Missing value **setelah** imputasi:")
st.write(df.isnull().sum())

# ======================
# 3. WINSORIZING OUTLIER (IQR)
# ======================
st.markdown("---")
st.subheader("3. Penanganan Outlier (Winsorizing, 1.5 × IQR)")

def winsorize_series(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    series_w = series.clip(lower, upper)
    return series_w, q1, q3, iqr, lower, upper

outlier_info = []

for col in num_cols:
    before = df[col].copy()
    after, q1, q3, iqr, lower, upper = winsorize_series(before)
    df[col] = after

    n_out = ((before < lower) | (before > upper)).sum()
    perc_out = n_out / len(before) * 100

    outlier_info.append({
        "Variabel": col,
        "Q1": round(q1, 2),
        "Q3": round(q3, 2),
        "IQR": round(iqr, 2),
        "Lower bound": round(lower, 2),
        "Upper bound": round(upper, 2),
        "Jumlah Outlier (sebelum winsorize)": int(n_out),
        "Persentase Outlier (%)": round(perc_out, 2)
    })

st.write("Ringkasan outlier sebelum winsorizing:")
st.dataframe(pd.DataFrame(outlier_info))

st.write("Boxplot variabel numerik **setelah** winsorizing:")
fig_box = px.box(df, y=num_cols, points="outliers")
st.plotly_chart(fig_box, use_container_width=True)

# ======================
# 4. ONE HOT ENCODING
# ======================
st.markdown("---")
st.subheader("4. Encoding Variabel Kategorik (One Hot Encoding)")

X = df.drop(columns=[target_col])
y = df[target_col]

X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

st.write(f"Shape setelah One Hot Encoding: **{X_enc.shape[0]} baris × {X_enc.shape[1]} kolom**")
st.write("Contoh 5 baris fitur setelah encoding:")
st.dataframe(X_enc.head())

# ======================
# 5. TRAIN–TEST SPLIT (STRATIFIED)
# ======================
st.markdown("---")
st.subheader("5. Train–Test Split (Stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, stratify=y, random_state=42
)

st.write(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")
st.write("Distribusi kelas **train** sebelum SMOTE:")
st.write(y_train.value_counts())

# ======================
# 6. SMOTE (IMBALANCED HANDLING)
# ======================
st.markdown("---")
st.subheader("6. Handling Imbalanced Data dengan SMOTE (hanya pada train set)")

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

st.write("Distribusi kelas **train** setelah SMOTE:")
counts_smote = y_train_res.value_counts().sort_index()
st.write(counts_smote)

plot_smote = pd.DataFrame({
    "stroke": counts_smote.index.astype(str),
    "jumlah": counts_smote.values
})

fig_smote = px.bar(
    plot_smote,
    x="stroke",
    y="jumlah",
    text="jumlah",
    labels={"stroke": "Stroke (0 = tidak, 1 = ya)", "jumlah": "Jumlah Observasi"}
)
fig_smote.update_traces(textposition="outside")
fig_smote.update_layout(yaxis_title="Jumlah Observasi")
st.plotly_chart(fig_smote, use_container_width=True)

# ======================
# 7. SCALING NUMERIK
# ======================
st.markdown("---")
st.subheader("7. Standarisasi Fitur Numerik")

scaler = StandardScaler()
num_cols_in_enc = [c for c in X_train_res.columns if c in num_cols]

X_train_res[num_cols_in_enc] = scaler.fit_transform(X_train_res[num_cols_in_enc])
X_test[num_cols_in_enc] = scaler.transform(X_test[num_cols_in_enc])

st.write("Contoh 5 baris fitur numerik setelah scaling (train):")
st.dataframe(X_train_res[num_cols_in_enc].head())

# ======================
# 8. CEK KORELASI NUMERIK (MULTIKOLINEARITAS SEDERHANA)
# ======================
st.markdown("---")
st.subheader("8. Korelasi Antar Variabel Numerik")

corr = df[num_cols].corr()

fig_corr, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
plt.tight_layout()
st.pyplot(fig_corr)

st.caption(
    "Jika ada korelasi yang sangat tinggi (misalnya > 0.8), "
    "perlu dipertimbangkan pengurangan atau penggabungan variabel."
)

# ======================
# 9. SIMPAN KE SESSION_STATE UNTUK MODELING
# ======================
st.markdown("---")
st.success(
    "Preprocessing selesai. Objek berikut disimpan di `st.session_state` "
    "dan bisa dipakai di halaman Modeling."
)

st.session_state["X_train"] = X_train_res
st.session_state["y_train"] = y_train_res
st.session_state["X_test"] = X_test
st.session_state["y_test"] = y_test
