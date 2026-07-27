import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Configure Vercel & Notion Style Dark Theme for Matplotlib
plt.style.use('dark_background')
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 11,
    'figure.facecolor': '#0a0a0c',
    'axes.facecolor': '#121216',
    'axes.edgecolor': '#27272a',
    'axes.labelcolor': '#ededed',
    'xtick.color': '#a1a1aa',
    'ytick.color': '#a1a1aa',
    'grid.color': '#27272a',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'text.color': '#ededed',
    'savefig.facecolor': '#0a0a0c',
    'savefig.edgecolor': '#0a0a0c'
})

# Directory setup
os.makedirs('dataset', exist_ok=True)
os.makedirs(os.path.join('static', 'images'), exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load Dataset
data_path = os.path.join('dataset', 'heart.csv')
df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape}")

# Features & Target
X = df.drop(columns=['target'])
y = df['target']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# -------------------------------------------------------------
# 2. EXPERIMENTS: SVM (5 Tuning Runs)
# -------------------------------------------------------------
svm_experiments = [
    {"exp": 1, "name": "SVM Exp 1 (Linear, C=0.01)", "params": {"kernel": "linear", "C": 0.01, "probability": True}},
    {"exp": 2, "name": "SVM Exp 2 (Linear, C=1.0)", "params": {"kernel": "linear", "C": 1.0, "probability": True}},
    {"exp": 3, "name": "SVM Exp 3 (RBF, C=1.0, gamma='scale')", "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True}},
    {"exp": 4, "name": "SVM Exp 4 (RBF, C=10.0, gamma=0.01)", "params": {"kernel": "rbf", "C": 10.0, "gamma": 0.01, "probability": True}},
    {"exp": 5, "name": "SVM Exp 5 (Poly deg=3, C=1.0)", "params": {"kernel": "poly", "degree": 3, "C": 1.0, "probability": True}},
]

svm_results = []
svm_best_model = None
svm_best_acc = 0.0

for item in svm_experiments:
    model = SVC(**item["params"], random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    res = {
        "Exp": item["exp"],
        "Model": "SVM",
        "Config": item["name"],
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "model_obj": model,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    svm_results.append(res)
    if acc > svm_best_acc:
        svm_best_acc = acc
        svm_best_model = res

# -------------------------------------------------------------
# 3. EXPERIMENTS: XGBoost (5 Tuning Runs)
# -------------------------------------------------------------
xgb_experiments = [
    {"exp": 1, "name": "XGB Exp 1 (n_est=50, depth=3, lr=0.01)", "params": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.01}},
    {"exp": 2, "name": "XGB Exp 2 (n_est=100, depth=3, lr=0.1)", "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}},
    {"exp": 3, "name": "XGB Exp 3 (n_est=100, depth=5, lr=0.1)", "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}},
    {"exp": 4, "name": "XGB Exp 4 (n_est=200, depth=4, lr=0.05, sub=0.8)", "params": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8}},
    {"exp": 5, "name": "XGB Exp 5 (n_est=300, depth=6, lr=0.01, sub=0.9, col=0.8)", "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.01, "subsample": 0.9, "colsample_bytree": 0.8}},
]

xgb_results = []
xgb_best_model = None
xgb_best_acc = 0.0

for item in xgb_experiments:
    model = XGBClassifier(**item["params"], random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    res = {
        "Exp": item["exp"],
        "Model": "XGBoost",
        "Config": item["name"],
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "model_obj": model,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    xgb_results.append(res)
    if acc > xgb_best_acc:
        xgb_best_acc = acc
        xgb_best_model = res

df_svm_res = pd.DataFrame(svm_results).drop(columns=['model_obj', 'y_pred', 'y_prob'])
df_xgb_res = pd.DataFrame(xgb_results).drop(columns=['model_obj', 'y_pred', 'y_prob'])

# Save results & models
all_res = pd.concat([df_svm_res, df_xgb_res], ignore_index=True)
all_res.to_csv(os.path.join('results', 'experiment_results.csv'), index=False)

joblib.dump(svm_best_model['model_obj'], os.path.join('models', 'svm_model.joblib'))
joblib.dump(xgb_best_model['model_obj'], os.path.join('models', 'xgb_model.joblib'))
joblib.dump(scaler, os.path.join('models', 'scaler.joblib'))

# -------------------------------------------------------------
# 4. GENERATE VERCEL / NOTION STYLE CHARTS
# -------------------------------------------------------------
img_dir = os.path.join('static', 'images')

# Chart 1: Experiment Progression
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
exps = [1, 2, 3, 4, 5]

ax[0].plot(exps, df_svm_res['Accuracy'], marker='o', markersize=8, linewidth=2.5, label='SVM Accuracy', color='#0070f3')
ax[0].plot(exps, df_xgb_res['Accuracy'], marker='s', markersize=8, linewidth=2.5, label='XGBoost Accuracy', color='#ff0080')
ax[0].set_title('Tuning Progress vs Accuracy', fontsize=13, fontweight='bold', pad=12)
ax[0].set_xlabel('Experiment Run (1 - 5)')
ax[0].set_ylabel('Accuracy Score')
ax[0].set_xticks(exps)
ax[0].set_ylim(0.70, 0.95)
ax[0].legend(frameon=True, facecolor='#18181b', edgecolor='#27272a')
ax[0].grid(True)

ax[1].plot(exps, df_svm_res['F1-Score'], marker='o', markersize=8, linewidth=2.5, label='SVM F1-Score', color='#50e3c2')
ax[1].plot(exps, df_xgb_res['F1-Score'], marker='s', markersize=8, linewidth=2.5, label='XGBoost F1-Score', color='#7928ca')
ax[1].set_title('Tuning Progress vs F1-Score', fontsize=13, fontweight='bold', pad=12)
ax[1].set_xlabel('Experiment Run (1 - 5)')
ax[1].set_ylabel('F1-Score')
ax[1].set_xticks(exps)
ax[1].set_ylim(0.70, 0.95)
ax[1].legend(frameon=True, facecolor='#18181b', edgecolor='#27272a')
ax[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'hyperparameter_tuning_experiments.png'), dpi=300, bbox_inches='tight')
plt.close()

# Chart 2: Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
cm_svm = confusion_matrix(y_test, svm_best_model['y_pred'])
cm_xgb = confusion_matrix(y_test, xgb_best_model['y_pred'])

sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False, annot_kws={"size": 18, "weight": "bold", "color": "#ffffff"})
ax[0].set_title(f"Confusion Matrix: Best SVM\n({svm_best_model['Config']})", fontsize=12, fontweight='bold', pad=12)
ax[0].set_xlabel('Predicted Label')
ax[0].set_ylabel('True Label')
ax[0].set_xticklabels(['Sehat (0)', 'Sakit (1)'])
ax[0].set_yticklabels(['Sehat (0)', 'Sakit (1)'])

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Purples', ax=ax[1], cbar=False, annot_kws={"size": 18, "weight": "bold", "color": "#ffffff"})
ax[1].set_title(f"Confusion Matrix: Best XGBoost\n({xgb_best_model['Config']})", fontsize=12, fontweight='bold', pad=12)
ax[1].set_xlabel('Predicted Label')
ax[1].set_ylabel('True Label')
ax[1].set_xticklabels(['Sehat (0)', 'Sakit (1)'])
ax[1].set_yticklabels(['Sehat (0)', 'Sakit (1)'])

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# Chart 3: ROC-AUC Curves
fig, ax = plt.subplots(figsize=(8.5, 6))
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_best_model['y_prob'])
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_best_model['y_prob'])

ax.plot(fpr_svm, tpr_svm, label=f"Best SVM (AUC = {svm_best_model['ROC-AUC']:.4f})", color='#0070f3', linewidth=3)
ax.plot(fpr_xgb, tpr_xgb, label=f"Best XGBoost (AUC = {xgb_best_model['ROC-AUC']:.4f})", color='#ff0080', linewidth=3)
ax.plot([0, 1], [0, 1], 'k--', label='Baseline / Random Classifier', color='#71717a', linewidth=1.5)

ax.set_title('Perbandingan Kurva ROC-AUC (Vercel Dark Theme)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
ax.legend(loc='lower right', frameon=True, facecolor='#18181b', edgecolor='#27272a')
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'roc_auc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# Chart 4: Feature Importance
importances = xgb_best_model['model_obj'].feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
bars = feat_imp.plot(kind='barh', color='#50e3c2', ax=ax, width=0.7)
ax.set_title('Feature Importance Analysis (XGBoost)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Importance Score')
ax.set_ylabel('Atribut Medis')
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nSUCCESS: All models trained and Vercel & Notion styled charts saved to static/images/!")
