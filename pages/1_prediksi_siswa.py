# pages/1_Prediksi_Siswa.py (KOREKSI FINAL UNTUK MENGATASI NAMERROR)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- FUNGSI BARU: MUAT & TRAIN MODEL DENGAN CACHE ---
@st.cache_resource
def load_and_train_model():
    st.info("Memuat dan Melatih Model ML... Ini hanya terjadi satu kali per deployment.")

    # 1. LOAD DATA
    def read_csv_semicolon(filepath):
        return pd.read_csv(filepath, sep=";")
    
    try:
        df = read_csv_semicolon('data.csv') 
    except Exception as e:
        st.error(f"Gagal memuat data untuk pelatihan model: {e}")
        return None, None, None
        
    # --- 2. PREPROCESSING LENGKAP (SESUAI NOTEBOOK) ---
    
    # Koreksi dan Konversi Target
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True) 
    df.rename(columns={'Curricularunits1stsemgrade': 'Curricular_units_1st_sem_grade', 
                       'Tuitionfeesuptodate': 'Tuition_fees_up_to_date'}, inplace=True)
    
    le = LabelEncoder()
    df['Target_Encoded'] = le.fit_transform(df['Status'])
    
    X = df.drop(['Status', 'Target_Encoded'], axis=1)
    y = df['Target_Encoded']

    # Definisikan Fitur
    kolom_integer_kategorikal = [
        'Marital_status', 'Application_mode', 'Application_order', 'Course', 
        'Daytime_evening_attendance', 'Previous_qualification', 'Nacionality', 
        # ... (Semua kolom kategorikal kode integer) ...
        'Gender', 'Scholarship_holder', 'International'
    ]
    numerical_features = X.select_dtypes(include=['float64']).columns.tolist()
    numerical_features.extend([col for col in X.select_dtypes(include=['int64']).columns.tolist() if col not in kolom_integer_kategorikal])
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    categorical_features.extend(kolom_integer_kategorikal)
    
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 3. MODELING (TRAINING) ---
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_model)])
    pipeline.fit(X_train, y_train)
    
    # --- 4. GENERATE DEFAULT INPUT STRUCTURE (UNTUK UI) ---
    default_data = {}
    for col in X.columns:
        if col in categorical_features or X[col].dtype == 'object':
            default_data[col] = X[col].mode()[0]
        else:
            default_data[col] = X[col].median()
            
    df_default = pd.DataFrame([default_data])

    st.success("Model berhasil dilatih dan di-cache!")
    return pipeline, le, df_default # Mengembalikan model, encoder, dan data default

# --- Panggil Fungsi Caching Utama ---
pipeline, le, df_default = load_and_train_model()

if pipeline is None:
    st.stop()

TARGET_CLASSES = le.classes_ 

# --- UI Prediksi (Di Tengah Halaman) ---
# ... (Sisanya kode UI prediksi tetap sama, menggunakan df_default yang sudah di-cache)
st.title("🔍 Prediksi Status Siswa")
st.subheader("Masukkan Data Kritis Siswa untuk Mendapatkan Prediksi Risiko Dropout")

# --- Sisanya kode UI (Form, Input, Submitted) ---
# Gunakan df_default yang sudah di-cache.

# [Lanjutkan dengan kode UI yang ada, karena df_default sudah tersedia di sini]

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
        format_func=lambda x: x[1], index=0 if df_default['Tuition_fees_up_to_date'].iloc[0] == 0 else 1
    )[0]

    is_debtor = col5.selectbox(
        "Status Debtor (Berhutang)?",
        options=[(1, "Ya"), (0, "Tidak")],
        format_func=lambda x: x[1], index=0 if df_default['Debtor'].iloc[0] == 0 else 1
    )[0]
    
    is_scholar = col6.selectbox(
        "Penerima Beasiswa?",
        options=[(1, "Ya"), (0, "Tidak")],
        format_func=lambda x: x[1], index=0 if df_default['Scholarship_holder'].iloc[0] == 0 else 1
    )[0]
    
    st.subheader("3. Input Lanjutan (Diisi Otomatis)")
    
    with st.expander("Ubah Input Lanjutan (Opsional)"):
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
    X_new = df_default.copy()
    
    # 2. Timpa nilai-nilai default dengan input dari user (Top 9 fitur)
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
st.caption("Solusi Machine Learning oleh **Andreas Wirawan Dananjaya** (ID Dicoding: **andreaswd31**) untuk Jaya Jaya Institut.")
