import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# LAPORAN FINAL PROJECT UJIAN AKHIR SEMESTER (UAS)
## MATA KULIAH: MACHINE LEARNING (TK075) - 2 SKS
**Program Studi Teknik Komputer | Fakultas Ilmu Komputer**
**Universitas Amikom Yogyakarta**

---

### 📌 COVER LAPORAN
* **Judul Project**: Klasifikasi Kualitas Produk Minuman (Red Wine Quality Classification) Menggunakan Algoritma Random Forest dan Support Vector Machine (SVM)
* **Nama Anggota Tim**: `[NAMA MAHASISWA]`
* **NIM**: `[NIM MAHASISWA]`
* **Dosen Pengampu**:
  1. Afrig Aminuddin, S.Kom., M.Eng., Ph.D
  2. Dr. Hartatik, S.T., M.Cs.
  3. I Made Artha Agastya, Ph.D
  4. Norhikmah, M.Kom
  5. Robert Marco, S.T., M.T., Ph.D.

---
"""))

# Cell 1: Install Dependencies
cells.append(nbf.v4.new_code_cell("""# Automatic installation for missing packages in notebook environment
%pip install -q seaborn pandas scikit-learn matplotlib openpyxl
"""))

# Cell 2: Import Libraries
cells.append(nbf.v4.new_markdown_cell("""## 1. Import Libraries"""))
cells.append(nbf.v4.new_code_cell("""# Import pustaka utama yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Configuration style visualisasi
sns.set_theme(style="whitegrid")
print("[SUCCESS] Library successfully loaded!")
"""))

# Cell 2: Load Dataset
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Dataset dan Pra-pemrosesan Data
Dataset yang digunakan adalah **UCI Red Wine Quality Dataset** yang memuat 1.599 sampel dengan 11 atribut fisikokimia. Target dipeta-kan menjadi 2 kelas:
- **0 (Kualitas Standar / Low Quality)**: Nilai kualitas < 6
- **1 (Kualitas Tinggi / High Quality)**: Nilai kualitas >= 6
"""))

cells.append(nbf.v4.new_code_cell("""url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')

# Transformasi target ke Biner (0: Standar, 1: Tinggi)
df['target'] = (df['quality'] >= 6).astype(int)

print(f"Bentuk Dataset: {df.shape}")
print("\\n5 Baris Pertama Dataset:")
display(df.head())

print("\\nDistribusi Kelas Target:")
print(df['target'].value_counts())
"""))

# Cell 3: Data Splitting & Scaling
cells.append(nbf.v4.new_markdown_cell("""## 2. Pembagian Data (Train-Test Split) dan Normalisasi Fitur"""))
cells.append(nbf.v4.new_code_cell("""X = df.drop(columns=['quality', 'target'])
y = df['target']

# Split data 80% Training dan 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standarisasi fitur (Sangat krusial untuk SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Jumlah Data Latih (Train): {X_train.shape[0]}")
print(f"Jumlah Data Uji (Test): {X_test.shape[0]}")
"""))

# Cell 4: 5 Experiments Random Forest
cells.append(nbf.v4.new_markdown_cell("""## 3. Eksperimen Model 1: Random Forest Classifier (Minimal 5x Variasi Parameter)"""))
cells.append(nbf.v4.new_code_cell("""rf_configs = [
    {"name": "Exp 1 (RF Base)", "params": {"n_estimators": 10, "max_depth": 3, "random_state": 42}},
    {"name": "Exp 2 (RF Tuning 1)", "params": {"n_estimators": 50, "max_depth": 5, "random_state": 42}},
    {"name": "Exp 3 (RF Tuning 2)", "params": {"n_estimators": 100, "max_depth": 10, "criterion": "entropy", "random_state": 42}},
    {"name": "Exp 4 (RF Tuning 3)", "params": {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42}},
    {"name": "Exp 5 (RF Optimal)", "params": {"n_estimators": 300, "max_depth": 20, "min_samples_split": 2, "criterion": "entropy", "random_state": 42}},
]

rf_results = []
best_rf_model = None
best_rf_f1 = 0

for config in rf_configs:
    model = RandomForestClassifier(**config["params"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    rf_results.append({
        "Eksperimen": config["name"],
        "Akurasi": acc,
        "Presisi": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })
    if f1 > best_rf_f1:
        best_rf_f1 = f1
        best_rf_model = model

df_rf_res = pd.DataFrame(rf_results)
display(df_rf_res)
"""))

# Cell 5: 5 Experiments SVM
cells.append(nbf.v4.new_markdown_cell("""## 4. Eksperimen Model 2: Support Vector Machine / SVM (Minimal 5x Variasi Parameter)"""))
cells.append(nbf.v4.new_code_cell("""svm_configs = [
    {"name": "Exp 1 (SVM Base)", "params": {"kernel": "linear", "C": 0.1, "probability": True, "random_state": 42}},
    {"name": "Exp 2 (SVM Tuning 1)", "params": {"kernel": "linear", "C": 1.0, "probability": True, "random_state": 42}},
    {"name": "Exp 3 (SVM Tuning 2)", "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True, "random_state": 42}},
    {"name": "Exp 4 (SVM Tuning 3)", "params": {"kernel": "rbf", "C": 10.0, "gamma": "auto", "probability": True, "random_state": 42}},
    {"name": "Exp 5 (SVM Optimal)", "params": {"kernel": "rbf", "C": 50.0, "gamma": "scale", "probability": True, "random_state": 42}},
]

svm_results = []
best_svm_model = None
best_svm_f1 = 0

for config in svm_configs:
    model = SVC(**config["params"])
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    svm_results.append({
        "Eksperimen": config["name"],
        "Akurasi": acc,
        "Presisi": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })
    if f1 > best_svm_f1:
        best_svm_f1 = f1
        best_svm_model = model

df_svm_res = pd.DataFrame(svm_results)
display(df_svm_res)
"""))

# Cell 6: Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 5. Visualisasi Hasil Evaluasi dan Grafik Perbandingan"""))
cells.append(nbf.v4.new_code_cell("""# Visualisasi Confusion Matrix
y_pred_rf = best_rf_model.predict(X_test)
y_pred_svm = best_svm_model.predict(X_test_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Confusion Matrix - Random Forest Optimal")
axes[0].set_xlabel("Prediksi")
axes[0].set_ylabel("Aktual")

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Confusion Matrix - SVM Optimal")
axes[1].set_xlabel("Prediksi")
axes[1].set_ylabel("Aktual")

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Feature Importance Random Forest
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Atribut Paling Berpengaruh terhadap Kualitas Wine (Random Forest)")
plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Relative Importance Score")
plt.show()
"""))

nb.cells = cells

with open('Tugas_UAS_Machine_Learning.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("[SUCCESS] Tugas_UAS_Machine_Learning.ipynb successfully created!")
