import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import plotly.graph_objects as go

st.title("📊 G. Training & Testing – Repeated Holdout & K-Fold CV")

# ----------------------------------------------------
# Cek data dari preprocessing
# ----------------------------------------------------
required_keys = ["X_train", "y_train"]
if not all(k in st.session_state for k in required_keys):
    st.error(
        "Data training belum tersedia. Buka dulu halaman **Pre-Processing Data** "
        "supaya X_train dan y_train tersimpan di `st.session_state`."
    )
    st.stop()

X = st.session_state["X_train"]  # ini sudah SMOTE + scaling
y = st.session_state["y_train"]

st.write(f"Dataset untuk validasi (hasil SMOTE): **{X.shape[0]} baris × {X.shape[1]} fitur**")
st.write("Distribusi kelas (balanced):")
st.write(y.value_counts())

# ----------------------------------------------------
# Definisi model
# ----------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
}


def eval_once(model, X_tr, y_tr, X_te, y_te):
    """Latih + evaluasi 1x, kembalikan metrik dan ROC info."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # probabilitas / skor untuk AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_te)[:, 1]
    else:
        scores = model.decision_function(X_te)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min())

    acc = accuracy_score(y_te, y_pred)
    sens = recall_score(y_te, y_pred, pos_label=1)
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    auc = roc_auc_score(y_te, y_proba)
    fpr, tpr, _ = roc_curve(y_te, y_proba)

    return acc, sens, spec, auc, fpr, tpr


# ====================================================
# 1. REPEATED HOLDOUT
# ====================================================
st.markdown("---")
st.subheader("1. Repeated Holdout")

col1, col2 = st.columns(2)
with col1:
    test_size = st.slider("Proporsi test", 0.2, 0.4, 0.3, 0.05)
with col2:
    n_repeats = st.slider("Jumlah ulangan", 3, 10, 5, 1)

st.caption(
    "Repeated holdout dilakukan pada **data training yang sudah di-SMOTE**. "
    "Setiap ulangan: split ulang train/test, latih model di train, evaluasi di test."
)

rh_results = {}  # dict: model -> list metrik

for name in models.keys():
    rh_results[name] = {
        "acc": [],
        "sens": [],
        "spec": [],
        "auc": [],
        "fpr_last": None,
        "tpr_last": None,
    }

for rep in range(n_repeats):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42 + rep
    )

    for name, mdl in models.items():
        acc, sens, spec, auc, fpr, tpr = eval_once(mdl, X_tr, y_tr, X_te, y_te)
        rh_results[name]["acc"].append(acc)
        rh_results[name]["sens"].append(sens)
        rh_results[name]["spec"].append(spec)
        rh_results[name]["auc"].append(auc)
        # simpan ROC dari run terakhir (buat plot)
        rh_results[name]["fpr_last"] = fpr
        rh_results[name]["tpr_last"] = tpr

# summary table
rh_summary = []
for name, res in rh_results.items():
    rh_summary.append(
        {
            "Model": name,
            "Mean Accuracy": np.mean(res["acc"]),
            "Std Accuracy": np.std(res["acc"]),
            "Mean Sensitivity": np.mean(res["sens"]),
            "Std Sensitivity": np.std(res["sens"]),
            "Mean Specificity": np.mean(res["spec"]),
            "Std Specificity": np.std(res["spec"]),
            "Mean AUC": np.mean(res["auc"]),
            "Std AUC": np.std(res["auc"]),
        }
    )

rh_df = pd.DataFrame(rh_summary).set_index("Model")
st.write("**Ringkasan Repeated Holdout (berdasarkan data training):**")
st.dataframe(rh_df.style.format("{:.3f}"))

# ROC plot (pakai run terakhir sebagai representatif)
fig_rh = go.Figure()
for name, res in rh_results.items():
    fig_rh.add_trace(
        go.Scatter(
            x=res["fpr_last"],
            y=res["tpr_last"],
            mode="lines",
            name=f"{name} (last run, AUC≈{np.mean(res['auc']):.3f})",
        )
    )

fig_rh.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash", color="gray"),
    )
)

fig_rh.update_layout(
    title="ROC Curve – Repeated Holdout (run terakhir)",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate (Sensitivity)",
)
st.plotly_chart(fig_rh, use_container_width=True)

# ====================================================
# 2. STRATIFIED K-FOLD CROSS VALIDATION
# ====================================================
st.markdown("---")
st.subheader("2. Stratified K-Fold Cross Validation")

k = st.radio("Pilih jumlah fold (k):", [5, 10], horizontal=True)

st.caption(
    "K-Fold CV juga dilakukan pada **data training yang sudah di-SMOTE**. "
    "StratifiedKFold menjaga proporsi kelas di setiap fold."
)

cv_results = {name: {"acc": [], "sens": [], "spec": [], "auc": []} for name in models}

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    for name, mdl in models.items():
        acc, sens, spec, auc, _, _ = eval_once(mdl, X_tr, y_tr, X_val, y_val)
        cv_results[name]["acc"].append(acc)
        cv_results[name]["sens"].append(sens)
        cv_results[name]["spec"].append(spec)
        cv_results[name]["auc"].append(auc)

cv_summary = []
for name, res in cv_results.items():
    cv_summary.append(
        {
            "Model": name,
            "Mean Accuracy": np.mean(res["acc"]),
            "Std Accuracy": np.std(res["acc"]),
            "Mean Sensitivity": np.mean(res["sens"]),
            "Std Sensitivity": np.std(res["sens"]),
            "Mean Specificity": np.mean(res["spec"]),
            "Std Specificity": np.std(res["spec"]),
            "Mean AUC": np.mean(res["auc"]),
            "Std AUC": np.std(res["auc"]),
        }
    )

cv_df = pd.DataFrame(cv_summary).set_index("Model")
st.write(f"**Ringkasan Stratified {k}-Fold CV:**")
st.dataframe(cv_df.style.format("{:.3f}"))

# ====================================================
# 3. Ringkasan naratif
# ====================================================
st.markdown("---")
st.subheader("3. Interpretasi Singkat")

best_rh = rh_df["Mean AUC"].idxmax()
best_cv = cv_df["Mean AUC"].idxmax()

st.markdown(
    f"""
- Pada skema **Repeated Holdout**, model dengan rata-rata AUC tertinggi adalah **{best_rh}**.  
- Pada skema **Stratified {k}-Fold CV**, model dengan rata-rata AUC tertinggi adalah **{best_cv}**.  

Jika konsisten dengan hasil di halaman **Modelling** (holdout test set), maka model tersebut dapat
dipilih sebagai kandidat utama untuk deployment / rekomendasi akhir.

Kamu bisa membandingkan:

- AUC & sensitivitas di **test set** (page Modelling)  
- dengan **Mean AUC & Mean Sensitivity** di halaman ini  

untuk cek apakah model cenderung stabil atau ada indikasi overfitting/underfitting.
"""
)
