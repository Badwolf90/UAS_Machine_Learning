import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# LAPORAN FINAL PROJECT MACHINE LEARNING (TK075)
## Deteksi Dini Penyakit Jantung Menggunakan SVM dan XGBoost dengan 5x Hyperparameter Tuning

- **Mata Kuliah:** Machine Learning (TK075)
- **Topik:** Classification - Heart Disease Prediction
- **Dataset:** Kaggle / UCI Cleveland Heart Disease Dataset
- **Model:** Support Vector Machine (SVM) vs XGBoost Classifier
"""),
    nbf.v4.new_markdown_cell("""### 1. Import Library & Load Dataset"""),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=cols, na_values='?')
df = df.dropna()
df['target'] = (df['target'] > 0).astype(int)

print(f"Shape Dataset: {df.shape}")
df.head()
"""),
    nbf.v4.new_markdown_cell("""### 2. Preprocessing & Data Splitting"""),
    nbf.v4.new_code_cell("""# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Train Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data Preprocessing & Scaling Selesai.")
"""),
    nbf.v4.new_markdown_cell("""### 3. Eksperimen Hyperparameter Tuning (5x Run SVM)"""),
    nbf.v4.new_code_cell("""svm_experiments = [
    {"exp": 1, "name": "SVM Exp 1 (Linear, C=0.01)", "params": {"kernel": "linear", "C": 0.01, "probability": True}},
    {"exp": 2, "name": "SVM Exp 2 (Linear, C=1.0)", "params": {"kernel": "linear", "C": 1.0, "probability": True}},
    {"exp": 3, "name": "SVM Exp 3 (RBF, C=1.0, gamma='scale')", "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True}},
    {"exp": 4, "name": "SVM Exp 4 (RBF, C=10.0, gamma=0.01)", "params": {"kernel": "rbf", "C": 10.0, "gamma": 0.01, "probability": True}},
    {"exp": 5, "name": "SVM Exp 5 (Poly deg=3, C=1.0)", "params": {"kernel": "poly", "degree": 3, "C": 1.0, "probability": True}},
]

svm_results = []
for item in svm_experiments:
    model = SVC(**item["params"], random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    svm_results.append({
        "Exp": item["exp"],
        "Model": "SVM",
        "Config": item["name"],
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

df_svm = pd.DataFrame(svm_results)
df_svm
"""),
    nbf.v4.new_markdown_cell("""### 4. Eksperimen Hyperparameter Tuning (5x Run XGBoost)"""),
    nbf.v4.new_code_cell("""xgb_experiments = [
    {"exp": 1, "name": "XGB Exp 1 (n_est=50, depth=3, lr=0.01)", "params": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.01}},
    {"exp": 2, "name": "XGB Exp 2 (n_est=100, depth=3, lr=0.1)", "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}},
    {"exp": 3, "name": "XGB Exp 3 (n_est=100, depth=5, lr=0.1)", "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}},
    {"exp": 4, "name": "XGB Exp 4 (n_est=200, depth=4, lr=0.05, sub=0.8)", "params": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8}},
    {"exp": 5, "name": "XGB Exp 5 (n_est=300, depth=6, lr=0.01, sub=0.9, col=0.8)", "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.01, "subsample": 0.9, "colsample_bytree": 0.8}},
]

xgb_results = []
for item in xgb_experiments:
    model = XGBClassifier(**item["params"], random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    xgb_results.append({
        "Exp": item["exp"],
        "Model": "XGBoost",
        "Config": item["name"],
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

df_xgb = pd.DataFrame(xgb_results)
df_xgb
"""),
    nbf.v4.new_markdown_cell("""### 5. Visualisasi Evaluasi Performa Model"""),
    nbf.v4.new_code_cell("""# Visualisasi Perbandingan Eksperimen Tuning
exps = [1, 2, 3, 4, 5]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(exps, df_svm['Accuracy'], marker='o', linewidth=2.5, label='SVM Accuracy', color='#3498db')
ax[0].plot(exps, df_xgb['Accuracy'], marker='s', linewidth=2.5, label='XGBoost Accuracy', color='#e74c3c')
ax[0].set_title('Perbandingan Accuracy 5x Eksperimen Tuning', fontweight='bold')
ax[0].set_xlabel('Nomor Eksperimen')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[0].grid(True, linestyle='--')

ax[1].plot(exps, df_svm['F1-Score'], marker='o', linewidth=2.5, label='SVM F1-Score', color='#2ecc71')
ax[1].plot(exps, df_xgb['F1-Score'], marker='s', linewidth=2.5, label='XGBoost F1-Score', color='#9b59b6')
ax[1].set_title('Perbandingan F1-Score 5x Eksperimen Tuning', fontweight='bold')
ax[1].set_xlabel('Nomor Eksperimen')
ax[1].set_ylabel('F1-Score')
ax[1].legend()
ax[1].grid(True, linestyle='--')

plt.tight_layout()
plt.show()
""")
]

nb['cells'] = cells

with open('Heart_Disease_Prediction_UAS.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Jupyter Notebook generated successfully!")
