import plotly.express as px
import pandas as pd
import streamlit as st

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

st.subheader("Distribusi Target (Stroke) – Imbalanced Data")
counts = df["stroke"].value_counts().sort_index()
plot_df = pd.DataFrame({
    "stroke": counts.index,
    "jumlah": counts.values
})
fig = px.bar(
    plot_df,
    x="stroke",
    y="jumlah",
    text="jumlah",
    labels={"stroke": "Stroke (0 = tidak, 1 = ya)", "jumlah": "Jumlah Pasien"},
    color=plot_df["stroke"].astype(str)  # supaya warna beda per kelas
)

fig.update_traces(textposition="outside")
fig.update_layout(yaxis_title="Jumlah Pasien")

st.plotly_chart(fig, use_container_width=True)

st.write("Jumlah masing-masing kelas:")
st.write(df["stroke"].value_counts())


import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

st.subheader("Missing Values per Variabel")

# Hitung missing
missing = df.isnull().sum()

# Ambil hanya variabel yang benar-benar punya missing
missing = missing[missing > 0].sort_values(ascending=True)

if missing.empty:
    st.success("Tidak ada missing value pada dataset.")
else:
    fig, ax = plt.subplots(figsize=(10, 3))

    # Horizontal bar chart
    ax.barh(missing.index, missing.values)

    ax.set_xlabel("Jumlah Missing")
    ax.set_ylabel("Variabel")
    ax.set_title("Missing Values per Variabel")

    # Tambah label angka di ujung bar
    for i, v in enumerate(missing.values):
        ax.text(v + 2, i, str(v), va="center")

    plt.tight_layout()
    st.pyplot(fig)

    st.write("Ringkasan missing value:")
    st.write(missing)


st.subheader("Tipe Variabel")

data_types = pd.DataFrame({
    "variable": df.columns,
    "dtype": df.dtypes.astype(str)
})

st.dataframe(data_types)

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# =======================
# MISSING VALUES
# =======================
st.subheader("Missing Values per Variabel")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=True)

if missing.empty:
    st.success("Tidak ada missing value pada dataset.")
else:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(missing.index, missing.values)
    ax.set_xlabel("Jumlah Missing")
    ax.set_ylabel("Variabel")
    ax.set_title("Missing Values per Variabel")

    for i, v in enumerate(missing.values):
        ax.text(v + 2, i, str(v), va="center")

    plt.tight_layout()
    st.pyplot(fig)

    st.write("Ringkasan missing value:")
    st.write(missing)

# =======================
# OUTLIER CHECK (IQR)
# =======================
st.subheader("Deteksi Outlier (Metode IQR)")

# Pilih variabel numerik kontinu aja
num_cols = ["age", "avg_glucose_level", "bmi"]

outlier_info = []

for col in num_cols:
    series = df[col].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask_out = (series < lower) | (series > upper)
    n_out = mask_out.sum()
    perc_out = n_out / series.shape[0] * 100

    outlier_info.append({
        "Variabel": col,
        "Q1": round(q1, 2),
        "Q3": round(q3, 2),
        "IQR": round(iqr, 2),
        "Lower bound": round(lower, 2),
        "Upper bound": round(upper, 2),
        "Jumlah Outlier": int(n_out),
        "Persentase Outlier (%)": round(perc_out, 2)
    })

outlier_df = pd.DataFrame(outlier_info)
st.write("Ringkasan outlier (berdasarkan batas 1.5 × IQR):")
st.dataframe(outlier_df, use_container_width=True)

# =======================
# BOX PLOT VISUAL
# =======================
st.subheader("Boxplot Variabel Numerik")

fig_box = px.box(
    df,
    y=num_cols,
    points="outliers",
    labels={"value": "Nilai", "variable": "Variabel"}
)
st.plotly_chart(fig_box, use_container_width=True)

