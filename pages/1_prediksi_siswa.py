import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- FUNGSI BARU UNTUK MEMBACA CSV DENGAN SEMICOLON ---
def read_csv_semicolon(filepath, **kwargs):
    return pd.read_csv(filepath, sep=";", **kwargs)
# --------------------------------------------------------
# Muat Model
try:
    with open('model/student_dropout_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
except FileNotFoundError:
    st.error("Model file tidak ditemukan. Pastikan 'student_dropout_pipeline.pkl' ada di folder 'model/'.")
    st.stop()

# Muat Struktur Data Default
try:
    df_default = pd.read_csv('model/default_input_structure.csv')
except FileNotFoundError:
    st.error("File struktur default tidak ditemukan. Jalankan Notebook dahulu.")
    st.stop()

# Definisikan Kelas Target
TARGET_CLASSES = ['Dropout', 'Enrolled', 'Graduate'] 

# --- Fitur Kritis (Top 10) ---
# Berdasarkan Feature Importance
TOP_FEATURES_INPUT = [
    'Curricular_units_2nd_sem_approved', # 1. Paling Penting
    'Curricular_units_1st_sem_grade',    # 2. Nilai Sem 1
    'Curricular_units_1st_sem_approved', # 3. Unit Lulus Sem 1
    'Curricular_units_2nd_sem_grade',    # 4. Nilai Sem 2
    'Curricular_units_1st_sem_enrolled', # 5. Unit Diambil Sem 1
    'Age_at_enrollment',                 # 6. Usia
    'Admission_grade',                   # 7. Nilai Masuk
    'Tuition_fees_up_to_date',           # 8. Status Keuangan
    'Debtor',                            # 9. Status Berhutang
    'Scholarship_holder'                 # 10. Penerima Beasiswa
]

# --- UI Prediksi (Di Tengah Halaman) ---

st.title("🔍 Prediksi Status Siswa")
st.subheader("Masukkan Data Kritis Siswa untuk Mendapatkan Prediksi Risiko Dropout")

with st.form("prediction_form"):
    
    st.header("1. Data Akademik Awal (Faktor Kritis)")
    col1, col2, col3 = st.columns(3)
    
    # INPUT FAKTOR AKADEMIK
    approved_2nd = col1.number_input(
        "Unit Disetujui Sem. 2", 
        min_value=0, max_value=25, value=int(df_default['Curricular_units_2nd_sem_approved'].iloc[0]), 
        help="Jumlah unit mata kuliah yang berhasil disetujui di semester 2."
    )
    grade_1st = col2.slider(
        "Nilai Rata-rata Sem. 1 (0-20)",
        min_value=0.0, max_value=20.0, 
        value=df_default['Curricular_units_1st_sem_grade'].iloc[0], 
        step=0.1
    )
    approved_1st = col3.number_input(
        "Unit Disetujui Sem. 1", 
        min_value=0, max_value=20, value=int(df_default['Curricular_units_1st_sem_approved'].iloc[0])
    )
    
    st.header("2. Data Finansial dan Demografi")
    col4, col5, col6 = st.columns(3)

    # INPUT FAKTOR SOSIAL/FINANSIAL
    fees_paid = col4.selectbox(
        "Biaya Kuliah Lunas (Up to Date)?",
        options=[(1, "Ya"), (0, "Tidak")],
        format_func=lambda x: x[1], index=int(df_default['Tuition_fees_up_to_date'].iloc[0])
    )[0]

    is_debtor = col5.selectbox(
        "Status Debtor (Berhutang)?",
        options=[(1, "Ya"), (0, "Tidak")],
        format_func=lambda x: x[1], index=int(df_default['Debtor'].iloc[0])
    )[0]
    
    is_scholar = col6.selectbox(
        "Penerima Beasiswa?",
        options=[(1, "Ya"), (0, "Tidak")],
        format_func=lambda x: x[1], index=int(df_default['Scholarship_holder'].iloc[0])
    )[0]
    
    st.subheader("3. Input Lanjutan (Diisi Otomatis)")
    
    # Input yang kurang penting diisi otomatis, tapi bisa diubah jika perlu
    with st.expander("Ubah Input Lanjutan (Opsional)"):
        # Hanya tampilkan 4 fitur penting sisanya di sini
        col_exp_1, col_exp_2, col_exp_3 = st.columns(3)
        
        age = col_exp_1.number_input(
            "Usia saat Pendaftaran", 
            min_value=17, max_value=60, 
            value=int(df_default['Age_at_enrollment'].iloc[0])
        )
        admission = col_exp_2.number_input(
            "Nilai Masuk (Admission Grade)", 
            min_value=0.0, max_value=200.0, 
            value=df_default['Admission_grade'].iloc[0]
        )
        grade_2nd = col_exp_3.slider(
            "Nilai Rata-rata Sem. 2 (0-20)",
            min_value=0.0, max_value=20.0, 
            value=df_default['Curricular_units_2nd_sem_grade'].iloc[0], 
            step=0.1
        )
        
    submitted = st.form_submit_button("Prediksi Status Siswa")


if submitted:
    # --- Peta Input ke 36 Fitur ---
    
    # 1. Mulai dengan struktur default (yang berisi 36 nilai median/modus)
    X_new = df_default.copy()
    
    # 2. Timpa nilai-nilai default dengan input dari user
    X_new['Curricular_units_2nd_sem_approved'] = approved_2nd
    X_new['Curricular_units_1st_sem_grade'] = grade_1st
    X_new['Curricular_units_1st_sem_approved'] = approved_1st
    X_new['Tuition_fees_up_to_date'] = fees_paid
    X_new['Debtor'] = is_debtor
    X_new['Scholarship_holder'] = is_scholar
    X_new['Age_at_enrollment'] = age
    X_new['Admission_grade'] = admission
    X_new['Curricular_units_2nd_sem_grade'] = grade_2nd
    
    # 3. Lakukan Prediksi
    try:
        prediction_encoded = pipeline.predict(X_new)[0]
        prediction_proba = pipeline.predict_proba(X_new)[0]
        
        status_prediksi = TARGET_CLASSES[prediction_encoded]
        
        # --- Tampilkan Hasil ---
        st.subheader("✅ Hasil Prediksi Status:")
        
        if status_prediksi == 'Dropout':
            st.error(f"## ❌ RISIKO TINGGI: {status_prediksi}")
            st.markdown(f"Probabilitas Dropout: **{prediction_proba[0]:.2f}**")
            st.warning("Rekomendasi: Siswa ini memerlukan intervensi konseling akademik dan finansial MENDESAK.")
        elif status_prediksi == 'Enrolled':
            st.warning(f"## 🔔 RISIKO SEDANG: {status_prediksi}")
            st.markdown(f"Probabilitas Enrolled: **{prediction_proba[1]:.2f}**")
            st.info("Rekomendasi: Lakukan monitoring rutin dan tawarkan bimbingan belajar.")
        else:
            st.success(f"## ⭐ STATUS BAIK: {status_prediksi}")
            st.markdown(f"Probabilitas Graduate: **{prediction_proba[2]:.2f}**")
            st.info("Rekomendasi: Pertahankan performa siswa ini.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan prediksi model: {e}")

st.markdown("---")
st.caption("Solusi Machine Learning oleh Andreas Wirawan Dananjaya untuk Jaya Jaya Institut.")