# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah beroperasi sejak tahun 2000 dengan reputasi baik dalam mencetak lulusan. Institusi pendidikan tinggi mana pun beroperasi di bawah mandat ganda: menjaga standar akademik dan memastikan keberhasilan siswa. Namun, Jaya Jaya Institut menghadapi ancaman serius terhadap kedua mandat tersebut: tingginya tingkat dropout siswa.

Tingginya dropout rate ini memiliki dampak yang merugikan baik secara finansial maupun reputasi. Secara finansial, setiap siswa yang dropout berarti kehilangan pendapatan dari biaya kuliah (tuition fees). Secara reputasi, tingkat kelulusan yang rendah mengurangi daya tarik institusi dan merusak citra kualitas pendidikan. Proyek ini diinisiasi untuk memberikan solusi preskriptif dan prediktif kepada manajemen institusi dan tim konseling, memungkinkan mereka untuk melakukan intervensi yang tepat waktu, alih-alih merespon setelah kerugian terjadi.

### Permasalahan Bisnis
**1. Dampak Finansial Eksponensial**

Setiap siswa yang dropout berarti institusi kehilangan potensi pendapatan biaya kuliah (tuition fees) untuk sisa durasi studinya. Lebih jauh, jika dropout rate tinggi, dana operasional (seperti yang dialokasikan untuk fasilitas dan tenaga pengajar) menjadi tidak efisien. Biaya untuk rekrutmen dan onboarding siswa baru (pemasaran, administrasi) untuk menggantikan siswa yang keluar menciptakan kerugian berulang (recurring losses) yang merusak stabilitas keuangan institusi.

**2. Erosi Nilai Pendidikan dan Reputasi**

Tingkat dropout yang tinggi merusak metrik kinerja kunci institusi, seperti tingkat kelulusan dan tingkat retensi. Di mata publik dan lembaga akreditasi, dropout tinggi diterjemahkan menjadi kegagalan sistem pendidikan, yang secara langsung mengurangi daya saing institusi dan mempersulit perekrutan talenta akademik serta siswa baru di masa depan.

**3. Urgensi Intervensi Dini (Waktu adalah Kritis)**

Data akademik dan sosial-finansial siswa (seperti Nilai Semester 1 dan status Pembayaran Biaya Kuliah) adalah prediktor yang paling kuat. Hal ini menciptakan window of opportunity yang sangat sempit—jika siswa yang berisiko tidak diidentifikasi sebelum akhir Semester 1 atau awal Semester 2, upaya intervensi (konseling akademik, bantuan finansial) seringkali sudah terlambat atau kurang efektif. Oleh karena itu, urgency proyek ini terletak pada kemampuan sistem untuk menghasilkan prediksi yang akurat dan actionable sejak siswa baru mulai menunjukkan tanda-tanda kesulitan.

### Cakupan Proyek
Cakupan proyek ini meliputi seluruh proses Data Science end-to-end, dari preprocessing hingga deployment, dengan fokus utama pada pemecahan masalah klasifikasi multi-kelas (Dropout, Enrolled, Graduate):uliskan cakupan proyek yang akan dikerjakan.

**1. Analisis Eksploratif Data (EDA)**

Mengidentifikasi fitur-fitur yang paling membedakan siswa Dropout (misalnya, Nilai Semester 1, Status Pembayaran Biaya Kuliah) untuk validasi model dan dashboard.

**2. Pembangunan Model Machine Learning**

Mengembangkan, melatih, dan mengevaluasi model Random Forest Classifier yang mampu memprediksi status akhir siswa dengan akurasi yang memadai.

**3. Deployment Business Dashboard (Metabase)**

Membuat dashboard visual yang informatif, berfungsi sebagai alat monitor risiko dropout bagi tim konseling.

**4. Deployment Prototype Sistem Prediksi (Streamlit Cloud)**

Mengembangkan web application prototype menggunakan Streamlit yang memuat model ML, memungkinkan user memasukkan data siswa baru, dan mendapatkan prediksi status secara real-time dan dapat diakses dari jarak jauh (remote access).

### Persiapan

Sumber data: Students' Performance — Dataset yang dikumpulkan dari berbagai database institusi pendidikan tinggi (termasuk faktor akademik, demografi, dan sosial-ekonomi) yang digunakan untuk memprediksi dropout dan kesuksesan akademik. berikut link sumber dataset
https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv 

Setup environment:

1. Membuat Virtual Environment
```
conda create --name ds-jaya-institute python=3.11
```

2. Aktifkan Environment
```
conda activate ds-jaya-institute
```

3. Instalasi Dependensi requirements.txt
```
pip install -r requirements.txt
```

**Petunjuk Menjalankan Dashboard Metabase (Port 3001)**
1. Pastikan Docker Desktop berjalan dan file metabase.db.mv.db ada di root proyek.
2. Jalankan Container Metabase dan Muat Database (Volume Mount).
```
docker run -d -p 3001:3000 --name institus-app -v "${PWD}\metabase.db.mv.db:/metabase.db/metabase.db.mv.db" metabase/metabase
```

3. Akses Dashboard: http://localhost:3001
4. Login: email: root@mail.com, password: root123


## Business Dashboard
Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.

## Menjalankan Sistem Machine Learning
**Petunjuk Menjalankan Prototype Streamlit**

klik link berikut : 

## Conclusion
Proyek ini berhasil mengatasi problem statement (risiko dropout tinggi) dengan mengidentifikasi faktor-faktor pendorong utama dan menyediakan alat prediksi yang siap digunakan

1. **Faktor Pendorong Utama (Pola Kritis)**
    - **Akademik Semester 1/2**
    
    Analisis Feature Importance menegaskan bahwa kinerja akademik awal (Curricular units approved dan Grade) adalah prediktor tunggal terkuat. Siswa yang memiliki Nilai Rata-rata di bawah 10 dan Jumlah Unit Disetujui yang rendah (hampir nol) di Semester 1 berada pada risiko Dropout tertinggi.
        
    - **Finansial**
    
    Status pembayaran biaya kuliah (Tuition fees up to date) dan status Debtor merupakan faktor finansial yang signifikan. Siswa yang Belum Melunasi biaya kuliah memiliki risiko Dropout yang secara signifikan lebih tinggi.

2. **Kinerja Model dan Solusi**
    - Model Random Forest Classifier memberikan akurasi yang memadai (76%) dalam klasifikasi multi-kelas, dengan Recall yang baik untuk kelas Graduate (0.88) dan Recall yang cukup baik untuk kelas Dropout (0.71).
    
    - Sistem ini menyediakan solusi prediktif, memungkinkan tim konseling untuk mengalihkan fokus dari reaksi pasca-kejadian menjadi intervensi proaktif berdasarkan data risiko.

3. Risiko dropout di Jaya Jaya Institut utamanya merupakan hasil dari kombinasi kesulitan akademik yang muncul di awal semester dan tekanan finansial yang tidak terselesaikan. Intervensi harus bersifat holistik (menggabungkan dukungan akademik dan finansial) dan dilakukan secepatnya setelah data akademik Semester 1 tersedia.

### Rekomendasi Action Items
Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.
1. **Intervensi Akademik Dini Berbasis Nilai (Traffic Light System)**

    - Tindakan: Tim konseling harus segera meluncurkan program mentoring atau bimbingan belajar wajib (mandatory tutoring) bagi semua siswa yang memiliki Nilai Rata-rata Semester 1 di bawah 10 dan Unit Disetujui kurang dari 3.

    - Target: Meningkatkan approved units pada kelompok berisiko ini di semester berikutnya.

2. **Program Bantuan Finansial Proaktif dan Terstruktur**

    - Tindakan: Identifikasi siswa yang berstatus Debtor dan memiliki biaya kuliah Tuition fees up to date = 0 (Tidak Lunas). Tawarkan skema cicilan yang lebih fleksibel atau bantuan dana darurat/beasiswa mikro yang ditargetkan sebelum awal Semester 2.

    - Justifikasi: Mengatasi masalah finansial adalah cara tercepat untuk memotong risiko dropout yang bukan disebabkan oleh ketidakmampuan akademik.

3. **Pemanfaatan Machine Learning Prototype untuk Penentuan Prioritas**


    - Tindakan: Tim HR/Konseling wajib menggunakan aplikasi Streamlit yang telah di-deploy untuk menguji semua siswa baru di akhir Semester 1.

    - Proses: Siswa yang diprediksi memiliki status Dropout harus diberi konseling pribadi dalam kurun waktu 14 hari sejak hasil prediksi tersedia.

4. **Audit Kurikulum pada Program Berisiko**

    - Tindakan: Lakukan analisis dropout rate per Course (Meskipun tidak di Top 5 Feature Importance, ini penting secara bisnis). Jika dropout rate di suatu jurusan jauh lebih tinggi dari rata-rata, lakukan audit mendalam terhadap kurikulum atau ekspektasi yang diberikan kepada mahasiswa baru pada jurusan tersebut.
