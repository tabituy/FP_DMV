import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import plotly.graph_objects as go

st.title("🤖 Model Klasifikasi Stroke")

# ======================================================
# Ambil data dari session_state (hasil preprocessing)
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

st.write(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")
st.write("Distribusi kelas pada **test set**:")
st.write(y_test.value_counts())

# ======================================================
# Definisikan model-model
# ======================================================
st.markdown("---")
st.subheader("1. Model yang Digunakan")

st.markdown(
    """
- **Model 1 – Logistic Regression**  
  Sederhana, interpretable, dan baik sebagai baseline.

- **Model 2 – Random Forest**  
  Dapat menangkap hubungan non-linear dan cukup robust terhadap berbagai jenis fitur.

- **Model 3 – SVM (RBF Kernel)**  
  Model margin maksimal, sering bekerja baik pada data dengan batas kelas kompleks.
"""
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}

# ======================================================
# Fungsi evaluasi
# ======================================================
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    # Prediksi kelas & probabilitas
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback kalau tidak ada predict_proba
        y_scores = model.decision_function(X_test)
        # normalisasi ke [0,1] biar bisa dipakai AUC
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    acc = accuracy_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred, pos_label=1)  # recall kelas stroke

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "auc": auc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr
    }

# ======================================================
# Train & evaluate: semua model
# ======================================================
st.markdown("---")
st.subheader("2. Hasil Evaluasi per Model (Test Set)")

results = []
for name, mdl in models.items():
    with st.spinner(f"Melatih {name}..."):
        res = evaluate_model(name, mdl, X_train, y_train, X_test, y_test)
        results.append(res)

# Tabel ringkasan metric
summary = pd.DataFrame(
    [
        {
            "Model": r["name"],
            "Akurasi": round(r["accuracy"], 3),
            "Sensitivitas (Recall=1)": round(r["sensitivity"], 3),
            "Spesifisitas": round(r["specificity"], 3),
            "AUC": round(r["auc"], 3)
        }
        for r in results
    ]
).set_index("Model")

st.write("**Ringkasan performa pada test set:**")
st.dataframe(summary)

# ======================================================
# Tampilkan confusion matrix tiap model (opsional)
# ======================================================
st.markdown("---")
st.subheader("3. Confusion Matrix per Model")

for r in results:
    cm = r["cm"]
    tn, fp, fn, tp = cm.ravel()

    with st.expander(f"Confusion Matrix – {r['name']}"):
        st.write(
            pd.DataFrame(
                cm,
                index=["Actual 0 (Tidak Stroke)", "Actual 1 (Stroke)"],
                columns=["Pred 0", "Pred 1"]
            )
        )
        st.markdown(
            f"""
            - **True Negative (TN)** = {tn}  
            - **False Positive (FP)** = {fp}  
            - **False Negative (FN)** = {fn}  
            - **True Positive (TP)** = {tp}  
            """
        )

# ======================================================
# ROC curve gabungan
# ======================================================
st.markdown("---")
st.subheader("4. Kurva ROC dan Nilai AUC")

fig_roc = go.Figure()

for r in results:
    fig_roc.add_trace(
        go.Scatter(
            x=r["fpr"],
            y=r["tpr"],
            mode="lines",
            name=f"{r['name']} (AUC = {r['auc']:.3f})"
        )
    )

# Garis diagonal (model random)
fig_roc.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash", color="gray")
    )
)

fig_roc.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate (Sensitivity)",
    legend_title="Model",
    width=800,
    height=500
)

st.plotly_chart(fig_roc, use_container_width=True)

st.caption(
    "Kurva ROC menggambarkan trade-off antara True Positive Rate dan False Positive Rate. "
    "Model dengan AUC yang lebih tinggi umumnya memiliki kemampuan diskriminasi yang lebih baik."
)

# ======================================================
# Ringkasan tertulis
# ======================================================
st.markdown("---")
st.subheader("5. Ringkasan Analisis Klasifikasi")

best_model = summary["AUC"].idxmax()
st.markdown(
    f"""
Berdasarkan hasil evaluasi di atas, model dengan **AUC tertinggi** pada test set adalah  
**{best_model}**.

Dalam konteks deteksi risiko stroke:

- **Sensitivitas** penting karena *False Negative* (pasien stroke yang tidak terdeteksi) sangat berbahaya.  
- **Spesifisitas** tetap perlu dijaga agar *False Positive* tidak terlalu banyak, meskipun secara klinis masih lebih dapat diterima dibanding *False Negative*.  
- **AUC** memberikan gambaran menyeluruh mengenai kemampuan model membedakan kelas stroke dan bukan stroke pada berbagai threshold.

Model–model ini nantinya bisa dibandingkan lagi dengan skema **repeated holdout** dan **k-fold cross validation** pada bagian berikutnya untuk memastikan performa yang stabil dan tidak overfitting.
"""
)
