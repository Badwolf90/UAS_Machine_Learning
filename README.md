# UAS Machine Learning (TK075) - Deteksi Dini Penyakit Jantung (SVM vs XGBoost)

Proyek Ujian Akhir Semester Genap TA 2025/2026 Universitas Amikom Yogyakarta.

---

## 📁 Struktur Folder Project

```text
TUGAS UAS/
├── dataset/
│   └── heart.csv                          # Dataset Medis Penyakit Jantung (UCI Cleveland / Kaggle)
├── docs/
│   └── Soal_UAS_TK075.pdf                 # Soal & Petunjuk Resmi UAS
├── models/
│   ├── scaler.joblib                      # Fitted StandardScaler Object
│   ├── svm_model.joblib                   # Model Terbaik SVM (Linear C=0.01)
│   └── xgb_model.joblib                   # Model Terbaik XGBoost Classifier
├── notebooks/
│   └── Heart_Disease_Prediction_UAS.ipynb # Jupyter Notebook (Siap Diunggah ke Google Colab)
├── results/
│   └── experiment_results.csv             # Tabel Hasil Metrik Evaluasi 10 Eksperimen
├── src/
│   ├── train_and_evaluate.py               # Script Utama Pelatihan & Evaluasi 5x Tuning
│   └── predict_test.py                     # Script Testing Prediksi Interaktif / Demo
├── static/
│   ├── css/
│   │   └── style.css                      # Modern Glassmorphic Dark UI Styling
│   └── images/
│       ├── confusion_matrices.png          # Confusion Matrix SVM vs XGBoost
│       ├── feature_importance.png          # Feature Importance Chart (XGBoost)
│       ├── hyperparameter_tuning_experiments.png # Plot Progres 5x Tuning
│       └── roc_auc_curves.png              # Kurva ROC-AUC Model Terbaik
├── templates/
│   └── index.html                         # Template Flask Web Application UI
├── .gitignore                             # Git Ignore Configuration
├── Laporan_UAS_Machine_Learning_TK075.md  # Dokumen Laporan Final Project UAS
├── README.md                              # Dokumentasi Repository Project
├── app.py                                 # Server Flask Web Application
└── requirements.txt                       # Python Package Dependencies
```

---

## 🚀 Cara Menjalankan Project

### 1. Menjalankan Aplikasi Web (Flask Web App)
```bash
python app.py
```
Akses di browser pada URL `http://127.0.0.1:5000/`.

### 2. Uji Prediksi Pasien (Demo Terminal Testing Script)
```bash
python src/predict_test.py
```

### 3. Melatih Ulang & Evaluasi Model (5x Tuning SVM vs XGBoost)
```bash
python src/train_and_evaluate.py
```
