# 🍷 WineQuality AI - Aplikasi Prediksi & Diagnosis Kualitas Red Wine

Proyek ini adalah **Aplikasi Web Interaktif berbasis Machine Learning** untuk melakukan pengujian dan klasifikasi kualitas minuman anggur merah (*Red Wine Quality*) berdasarkan 11 parameter fisikokimia laboratorium. Proyek ini dikembangkan sebagai **Laporan Final Project Ujian Akhir Semester (UAS) Machine Learning (TK075)** di **Universitas Amikom Yogyakarta**.

---

## 🚀 Fitur Utama

1. **Interactive Assessment**: Form input dinamis untuk memasukkan 11 atribut fisikokimia produk minuman.
2. **AI Risk Assessment**: Prediksi tingkat kualitas (*High Quality* vs *Standard Quality*) secara *real-time* berbasis probabilitas pemodelan.
3. **Model Performance Insights**: Tab khusus untuk memvisualisasikan grafik performa model seperti Confusion Matrix, Kurva ROC-AUC, Feature Importance, dan Perkembangan 5x Eksperimen Tuning Parameter.
4. **Laboratory Recommendation**: Rekomendasi teknis otomatis berbasis hasil uji fisikokimia (misal: jika kadar keasaman volatil terlalu tinggi atau kadar alkohol rendah).
5. **Modern Premium Design**: Antarmuka pengguna (UI) bertema gelap (*Dark Mode*) dengan efek glassmorphism, warna harmonis, serta animasi responsif.

---

## 📊 Dataset & Model Machine Learning

* **Dataset**: UCI Red Wine Quality Dataset (1.599 sampel data laboratorium).
* **Fitur Prediksi (11 Atribut Fisikokimia)**: *Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates,* dan *Alcohol*.
* **Model yang Digunakan**: **Random Forest Classifier** dan **Support Vector Machine (SVM)** dengan 5x variasi eksperimen tuning hyperparameter.
* **Performa Model Terbaik**: Random Forest mencapai **Akurasi 80.31%**, **F1-Score 81.31%**, dan **ROC-AUC 0.9038**.

---

## 🛠️ Panduan Penggunaan & Instalasi

### 1. Instalasi Dependensi
Buka terminal pada direktori proyek ini dan jalankan perintah berikut:
```bash
pip install -r requirements.txt
```

### 2. Pelatihan & Evaluasi Model ML
Jalankan file `train_model.py` untuk mengunduh dataset, memproses data, melatih model, dan meng-ekspor model terbaik (`wine_model.joblib`), `scaler.joblib`, serta grafik visualisasi ke folder `static/images/`:
```bash
python train_model.py
```

### 3. Menjalankan Aplikasi Web
Jalankan server lokal Flask dengan perintah berikut:
```bash
python app.py
```
Setelah server aktif, buka browser dan akses alamat: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📁 Struktur Direktori Proyek

```text
UAS_MACHINE_LEARNING/
│
├── static/
│   ├── css/
│   │   └── style.css                   # Stylesheet Glassmorphism & Dark Mode UI
│   └── images/
│       ├── confusion_matrix.png         # Grafik Confusion Matrix RF vs SVM
│       ├── roc_curve.png                # Grafik Kurva ROC-AUC
│       ├── feature_importance.png       # Grafik Atribut Paling Berpengaruh
│       └── experiments_comparison.png   # Grafik Perkembangan 5x Tuning Parameter
│
├── templates/
│   └── index.html                       # Dashboard Antarmuka Web Utama
│
├── app.py                               # Backend Flask Server & REST API
├── train_model.py                       # Script Training & Export Model ML
├── winequality-red.csv                  # Dataset UCI Red Wine Quality
├── wine_model.joblib                    # Model Terpilih (Random Forest)
├── scaler.joblib                        # Fitted StandardScaler
├── model_info.txt                       # Catatan Performa Model Terpilih
├── requirements.txt                     # Daftar Library Python yang Dipakai
│
├── Laporan_UAS_Machine_Learning.pdf     # Laporan UAS Resmi Siap Cetak (PDF)
├── Laporan_UAS_Machine_Learning.docx    # Laporan UAS Format Word (.docx)
├── Laporan_UAS_Machine_Learning.md      # Laporan UAS Format Markdown
└── Tugas_UAS_Machine_Learning.ipynb     # Jupyter Notebook Google Colab
```

---

## 📑 Berkas Laporan Resmi UAS

* 📄 **[Laporan PDF Resmi](Laporan_UAS_Machine_Learning.pdf)**: Dokumen laporan 7 halaman lengkap dengan Cover, Latar Belakang, Cara Kerja Algoritma, Tabel 5x Eksperimen, Diagram Visualisasi, dan Tabel Kontribusi Tim.
* 📝 **[Laporan Word (.docx)](Laporan_UAS_Machine_Learning.docx)**: File Word untuk mengedit nama/NIM secara langsung.
* 📓 **[Jupyter Notebook Colab (.ipynb)](Tugas_UAS_Machine_Learning.ipynb)**: Notebook Python siap dijalankan di Google Colab.

---
© 2026 Universitas Amikom Yogyakarta | Program Studi Teknik Komputer
