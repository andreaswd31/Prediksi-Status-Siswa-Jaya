# pages/1_Prediksi_Siswa.py
import streamlit as st
import pandas as pd
import numpy as np

# Import semua komponen yang dibutuhkan dari Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- FUNGSI BARU: MUAT & TRAIN MODEL DENGAN CACHE ---
# Fungsi ini akan dijalankan sekali saat deployment (menggantikan pickle.load)
@st.cache_resource
def load_and_train_model():
    st.info("Memuat dan Melatih Model ML... Ini hanya terjadi satu kali per deployment.")

    # 1. LOAD DATA (Ganti dengan file data.csv jika itu file input Anda)
    def read_csv_semicolon(filepath):
        # Asumsi data mentah di cloud menggunakan semicolon
        return pd.read_csv(filepath, sep=";")
    
    try:
        # Ganti 'data.csv' jika Anda menggunakan nama file yang berbeda
        df = read_csv_semicolon('data.csv') 
    except Exception as e:
        st.error(f"Gagal memuat data untuk pelatihan model: Pastikan 'data.csv' ada dan menggunakan pemisah ';'. Error: {e}")
        return None, None
        
    # --- 2. PREPROCESSING LENGKAP (SESUAI NOTEBOOK) ---
    
    # KOREKSI NAMA KOLOM UNTUK KESERAGAMAN
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True) 
    df.rename(columns={'Curricularunits1stsemgrade': 'Curricular_units_1st_sem_grade', 
                       'Tuitionfeesuptodate': 'Tuition_fees_up_to_date'}, inplace=True)
    
    # 2a. Konversi Target dan Pisahkan Data
    le = LabelEncoder()
    df['Target_Encoded'] = le.fit_transform(df['Status'])
    
    X = df.drop(['Status', 'Target_Encoded'], axis=1)
    y = df['Target_Encoded']

    # 2b. Definisikan Fitur (Harus sama persis dengan notebook)
    kolom_integer_kategorikal = [
        'Marital_status', 'Application_mode', 'Application_order', 'Course', 
        'Daytime_evening_attendance', 'Previous_qualification', 'Nacionality', 
        'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation', 
        'Fathers_occupation', 'Displaced', 'Educational_special_needs', 'Debtor', 
        'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'International'
    ]
    numerical_features = X.select_dtypes(include=['float64']).columns.tolist()
    numerical_features.extend([col for col in X.select_dtypes(include=['int64']).columns.tolist() if col not in kolom_integer_kategorikal])
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    categorical_features.extend(kolom_integer_kategorikal)
    
    # 2c. Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Split Data (Hanya untuk train)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 3. MODELING (TRAINING) ---
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_model)])
    pipeline.fit(X_train, y_train)
    
    st.success("Model berhasil dilatih dan di-cache!")
    return pipeline, le, X.columns.tolist() # Kembalikan juga nama kolom fitur

# --- Panggil Fungsi Caching ---
pipeline, le, all_feature_columns = load_and_train_model()
TARGET_CLASSES = le.classes_ if le else ['Dropout', 'Enrolled', 'Graduate'] 

# --- Muat Struktur Data Default (Simulasi Median/Modus) ---
# Karena model dilatih di Cloud, kita harus mengisi nilai default dari data training (X_train)
# yang sudah dilatih di atas. Kita akan buat fungsi dummy untuk mendapatkan nilai median
@st.cache_resource
def get_default_values(df_train, all_columns_list):
    default_data = {}
    for col in all_columns_list:
        try:
            if df_train[col].dtype == 'object':
                default_data[col] = df_train[col].mode()[0] # Modus untuk kategori
            elif df_train[col].dtype == 'float64' or df_train[col].dtype == 'int64':
                default_data[col] = df_train[col].median() # Median untuk numerik
        except:
             default_data[col] = 0 # Fallback
    
    # Gunakan data training dari load_and_train_model (harus dipanggil lagi untuk scope yang benar)
    df_temp = load_and_train_model()[0].named_steps['preprocessor'].fit_transform(X) # Dummy call, not ideal
    
    # **Simplifikasi:** Karena sulit mengambil X_train di sini, kita akan memuat default dari file CSV lokal 
    # yang sudah Anda buat di notebook (default_input_structure.csv)
    try:
        df_default = pd.read_csv('model/default_input_structure.csv')
    except:
        return None
    return df_default.iloc[0].to_dict()

default_values_dict = get_default_values(pd.DataFrame(), all_feature_columns) # Pakai fungsi dummy

if default_values_dict is None or pipeline is None:
    st.error("Model ML tidak bisa dijalankan karena data default/pelatihan gagal dimuat.")
    st.stop()
    
df_default = pd.DataFrame([default_values_dict]) # Konversi kembali ke DataFrame untuk digunakan

# --- UI Prediksi (Di Tengah Halaman) ---
st.title("🔍 Prediksi Status Siswa")
st.subheader("Masukkan Data Kritis Siswa untuk Mendapatkan Prediksi Risiko Dropout")

# Fitur Kritis (Top 9, sesuai Feature Importance)
TOP_FEATURES_INPUT = [
    'Curricular_units_2nd_sem_approved', # Unit Lulus Sem 2
    'Curricular_units_1st_sem_grade',    # Nilai Sem 1
    'Curricular_units_1st_sem_approved', # Unit Lulus Sem 1
    'Tuition_fees_up_to_date',           # Status Keuangan
    'Debtor',                            # Status Berhutang
    'Scholarship_holder',                # Penerima Beasiswa
    'Age_at_enrollment',                 # Usia
    'Admission_grade',                   # Nilai Masuk
    'Curricular_units_2nd_sem_grade'     # Nilai Sem 2
]

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
