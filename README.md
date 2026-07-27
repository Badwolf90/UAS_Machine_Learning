# UAS Machine Learning (TK075) - Deteksi Dini Penyakit Jantung (SVM vs XGBoost)

Proyek Ujian Akhir Semester Genap TA 2025/2026 Universitas Amikom Yogyakarta.

---

## 📁 Struktur Folder Project

```text
TUGAS UAS/
├── dataset/
│   └── heart.csv                          # Dataset Medis Penyakit Jantung (UCI Cleveland / Kaggle)
├── images/
│   ├── confusion_matrices.png              # Confusion Matrix SVM vs XGBoost
│   ├── feature_importance.png              # Tingkat Kepentingan Fitur Medis (XGBoost)
│   ├── hyperparameter_tuning_experiments.png# Plot Progres 5x Tuning Hyperparameter
│   └── roc_auc_curves.png                  # Kurva ROC-AUC Model Terbaik
├── notebooks/
│   └── Heart_Disease_Prediction_UAS.ipynb # Jupyter Notebook (Siap Diunggah ke Google Colab)
├── results/
│   └── experiment_results.csv             # Tabel Hasil Metrik Evaluasi 10 Eksperimen
├── src/
│   ├── train_and_evaluate.py               # Script Utama Pelatihan & Evaluasi 5x Tuning
│   ├── predict_test.py                     # Script Testing Prediksi Interaktif / Demo
│   └── generate_notebook.py                # Generator Notebook Colab
├── Laporan_UAS_Machine_Learning_TK075.md  # Dokumen Laporan Final Project UAS
└── f72581d1-d075-f111-8389-d0338df1818f_TK075_20260702114516.pdf # Soal & Ketentuan UAS
```

---

## 🚀 Cara Menjalankan Pengujian (Testing)

### 1. Uji Prediksi Pasien (Demo Testing Script)
Jalankan perintah berikut di terminal untuk menguji prediksi model terbaik pada sampel rekam medis pasien:
```bash
python src/predict_test.py
```

### 2. Jalankan Ulang Seluruh Eksperimen (5x Tuning SVM & XGBoost)
Untuk melatih ulang model dan memperbarui seluruh grafik visualisasi & tabel hasil:
```bash
python src/train_and_evaluate.py
```

### 3. Menggunakan Google Colab
Unggah file `notebooks/Heart_Disease_Prediction_UAS.ipynb` ke [Google Colab](https://colab.research.google.com/) dan jalankan sel kode dari atas ke bawah.
