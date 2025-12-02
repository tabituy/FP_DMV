import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Stroke Prediction Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Stroke Prediction Dashboard")
st.markdown("### Analisis Faktor Risiko dan Prediksi Kejadian Stroke")

# --- 1. Deskripsi Dataset ---
st.subheader("1. Deskripsi Dataset")
st.write(
    """
    Dataset **Stroke Prediction** berasal dari platform Kaggle dan berisi data pasien 
    dengan berbagai karakteristik demografis dan klinis. 
    Setiap baris merepresentasikan satu pasien, dan variabel target (**`stroke`**) 
    menunjukkan apakah pasien pernah mengalami stroke (1) atau tidak (0).
    Tujuan analisis adalah **mengidentifikasi faktor-faktor risiko** yang berhubungan 
    dengan kejadian stroke serta membangun model klasifikasi untuk memprediksi 
    kemungkinan stroke pada pasien baru.
    """
)

# --- 2. Latar Belakang & Masalah ---
st.subheader("2. Latar Belakang dan Permasalahan")

st.markdown(
    """
    Stroke merupakan salah satu **penyebab utama kematian dan kecacatan** di dunia.  
    Deteksi dini individu berisiko tinggi sangat penting agar intervensi pencegahan dapat dilakukan
    (perubahan gaya hidup, pengobatan hipertensi, kontrol gula darah, dsb).

    Pada praktiknya, tenaga medis sering dihadapkan pada **banyak faktor risiko** yang saling 
    berinteraksi, seperti usia, riwayat hipertensi, penyakit jantung, kadar glukosa darah, 
    status merokok, dan lain-lain. 
    Analisis statistik dan *machine learning* dapat membantu:
    
    - Menggali **pola** antara faktor risiko dan kejadian stroke  
    - Menyusun **model prediksi** untuk mengklasifikasikan pasien berisiko tinggi  
    - Memberikan **informasi kuantitatif** untuk mendukung pengambilan keputusan klinis
    
    Namun, dataset ini memiliki beberapa tantangan:
    - **Kelas tidak seimbang (imbalanced)**: proporsi pasien stroke jauh lebih sedikit dibandingkan non-stroke  
    - Terdapat **missing value** (terutama pada variabel `bmi`)  
    - Kombinasi variabel numerik dan kategorik sehingga perlu preprocessing yang tepat  
    """
)

# --- 3. Perumusan Masalah ---
st.subheader("3. Rumusan Masalah")

st.markdown(
    """
    Berdasarkan latar belakang tersebut, analisis ini difokuskan pada pertanyaan:
    
    1. Faktor-faktor apa saja yang paling berhubungan dengan kejadian stroke?  
    2. Bagaimana performa beberapa metode klasifikasi (misalnya **Logistic Regression** dan 
       **Random Forest**) dalam memprediksi stroke?  
    3. Bagaimana hasil **training–testing** dengan skema *repeated holdout* dan 
       **K-fold cross validation (k = 5/10)** jika dinilai menggunakan:
       - Akurasi  
       - Sensitivitas (Recall untuk kelas stroke)  
       - Spesifisitas  
       - Kurva ROC dan nilai AUC  
    """
)

# --- 4. Deskripsi Variabel ---
st.subheader("4. Deskripsi Variabel dalam Dataset")

var_desc = pd.DataFrame(
    {
        "Nama Variabel": [
            "id",
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
            "stroke",
        ],
        "Tipe": [
            "Integer",
            "Kategori",
            "Numerik (kontinu)",
            "Biner (0/1)",
            "Biner (0/1)",
            "Kategori (Yes/No)",
            "Kategori",
            "Kategori (Urban/Rural)",
            "Numerik (kontinu)",
            "Numerik (kontinu)",
            "Kategori",
            "Biner (0/1)",
        ],
        "Keterangan": [
            "ID unik untuk setiap pasien",
            "Jenis kelamin pasien",
            "Usia pasien (dalam tahun)",
            "1 = pasien memiliki hipertensi, 0 = tidak",
            "1 = pasien memiliki penyakit jantung, 0 = tidak",
            "Status pernah menikah atau belum",
            "Jenis pekerjaan (Private, Self-employed, Govt_job, dll.)",
            "Tipe tempat tinggal (Urban atau Rural)",
            "Rata-rata kadar glukosa darah pasien",
            "Body Mass Index (BMI) pasien",
            "Status merokok (formerly smoked, never smoked, smokes, unknown)",
            "Variabel target: 1 = pernah stroke, 0 = tidak",
        ],
    }
)

st.table(var_desc)

st.info(
    """
    **Catatan penting:**  
    - Variabel **`stroke`** akan digunakan sebagai *target* pada pemodelan klasifikasi.  
    - Variabel numerik (`age`, `avg_glucose_level`, `bmi`) akan distandarisasi.  
    - Variabel kategorik akan di-*encoding* (misalnya dengan One-Hot Encoding).  
    - Karena kelas stroke jarang, metrik **sensitivitas** dan **AUC** menjadi sangat penting,
      bukan hanya akurasi.
    """
)
