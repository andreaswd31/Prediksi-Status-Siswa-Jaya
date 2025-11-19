# Home.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- FUNGSI BARU UNTUK MEMBACA CSV DENGAN SEMICOLON ---
def read_csv_semicolon(filepath, **kwargs):
    return pd.read_csv(filepath, sep=";", **kwargs)
# --------------------------------------------------------

# --- Setup Global ---
st.set_page_config(
    page_title="Jaya Jaya Institut - Dashboard",
    layout="wide"
)


# Coba muat data bersih untuk visualisasi
try:
    # Ganti 'data.csv' atau nama file data bersih siswa Anda
    df_clean = read_csv_semicolon('data.csv') 
    # Setelah dimuat, ganti nama kolom untuk menyesuaikan dengan nama kolom yang digunakan di notebook
    # Contoh: 'Curricular units 1st sem (grade)' diganti dengan 'Curricular_units_1st_sem_grade' jika itu yang digunakan di notebook Anda.
    
    # KOREKSI NAMA KOLOM UNTUK PLOTTING (Harus diselaraskan dengan hasil preprocessing di notebook)
    # Ini sering terjadi saat kolom memiliki spasi atau tanda kurung
    df_clean.columns = df_clean.columns.str.replace('[^A-Za-z0-9_]', '', regex=True) # Hapus karakter non-alfanumerik
    df_clean.rename(columns={'Curricularunits1stsemgrade': 'Curricular_units_1st_sem_grade', 
                              'Tuitionfeesuptodate': 'Tuition_fees_up_to_date'}, inplace=True)

except FileNotFoundError:
    st.error("File data bersih tidak ditemukan. Visualisasi tidak dapat dimuat.")
    df_clean = None

# Home.py (Bagian yang menampilkan visualisasi)

# ... (Kode Setup dan Muat Data di awal tetap sama) ...

st.title("🎓 Sistem Monitoring & Prediksi Dropout Jaya Jaya Institut")
st.markdown("Oleh: **Andreas Wirawan Dananjaya**")

st.markdown("""
---
## Analisis Kritis: Faktor Risiko Dropout
Dashboard ini menyajikan temuan utama dari *Exploratory Data Analysis* (EDA) yang menunjukkan pemicu utama risiko putus sekolah.
""")

if df_clean is not None:
    # --- VISUALISASI 1: Nilai Semester 1 vs. Status (Faktor Akademik Kritis) ---
    st.subheader("1. Perbedaan Nilai Rata-rata Semester 1")
    
    # Gunakan layout kolom untuk menempatkan info dan grafik bersebelahan
    col_info_1, col_chart_1 = st.columns([1, 2])
    
    with col_info_1:
        st.markdown("**Nilai Semester Awal Adalah Prediktor Terkuat.**")
        st.info("Nilai rata-rata Semester 1 adalah prediktor tunggal terkuat yang membedakan siswa **Dropout** dari **Graduate**.")
    
    with col_chart_1:
        # Visualisasi Density Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(data=df_clean, x='Curricular_units_1st_sem_grade', hue='Status', fill=True, palette=['#4169E1', '#8A2BE2', '#228B22'], ax=ax)
        ax.set_title('Distribusi Nilai Rata-rata Semester 1 (0-20)')
        ax.set_xlabel('Nilai Rata-rata')
        st.pyplot(fig)
    
    st.markdown("---")

    # --- VISUALISASI 2: Status Pembayaran vs. Dropout (Faktor Finansial Kritis) ---
    st.subheader("2. Dampak Status Biaya Kuliah (Tuition Fees)")
    
    # Gunakan layout kolom untuk menempatkan info dan grafik bersebelahan
    col_chart_2, col_info_2 = st.columns([2, 1])

    with col_chart_2:
        # Hitung persentase untuk visualisasi yang mudah
        status_by_fees = pd.crosstab(df_clean['Tuition_fees_up_to_date'], df_clean['Status'], normalize='index') * 100
        status_by_fees.rename(index={0: 'Tidak Lunas', 1: 'Lunas'}, inplace=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        # Visualisasi Stacked Bar
        status_by_fees[['Dropout', 'Graduate']].plot(kind='bar', stacked=True, ax=ax, color=['#DC143C', '#228B22'])
        ax.set_title('Status Siswa Berdasarkan Pembayaran Biaya Kuliah')
        ax.set_ylabel('Persentase (%)')
        ax.set_xlabel('Biaya Kuliah Up-to-Date')
        ax.legend(title='Status')
        plt.xticks(rotation=0) # Pastikan label X tidak miring
        st.pyplot(fig)

    with col_info_2:
        st.markdown("**Status Keuangan Berhubungan Langsung dengan Dropout.**")
        st.warning("Rasio **Dropout** secara signifikan lebih tinggi pada kelompok siswa yang biaya kuliahnya **Tidak Lunas**.")

st.markdown("""
---
## Tujuan Aplikasi
Gunakan halaman **prediksi siswa** untuk menguji data siswa baru dan mendapatkan rekomendasi intervensi.
""")
st.info("Silakan beralih ke halaman **prediksi siswa** di sidebar untuk menjalankan model ML.")
