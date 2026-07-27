import os
import joblib
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
    roc_auc_score, roc_curve, confusion_matrix
)

# Ensure directory structure
os.makedirs("static/images", exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})

def train_and_export():
    print("=== 1. LOADING DATASET ===")
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(dataset_url, sep=';')
    df.to_csv("winequality-red.csv", index=False)
    
    # Target: Quality >= 6 -> High Quality (1), < 6 -> Standard Quality (0)
    df['target'] = (df['quality'] >= 6).astype(int)
    
    X = df.drop(columns=['quality', 'target'])
    y = df['target']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, "scaler.joblib")
    
    print("=== 2. TRAINING MODELS ===")
    # Random Forest Optimal Model
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, criterion="entropy", random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_f1 = f1_score(y_test, rf_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_prob)
    
    # SVM Optimal Model
    svm_model = SVC(kernel="rbf", C=50.0, gamma="scale", probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    svm_f1 = f1_score(y_test, svm_pred)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_auc = roc_auc_score(y_test, svm_prob)
    
    print(f"Random Forest -> Acc: {rf_acc:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}")
    print(f"SVM           -> Acc: {svm_acc:.4f}, F1: {svm_f1:.4f}, AUC: {svm_auc:.4f}")
    
    # Best Model Selection
    if rf_f1 >= svm_f1:
        best_model = rf_model
        best_name = "Random Forest Classifier"
        best_acc = rf_acc
        best_f1 = rf_f1
        best_auc = rf_auc
        use_scaled = False
    else:
        best_model = svm_model
        best_name = "Support Vector Machine (SVM)"
        best_acc = svm_acc
        best_f1 = svm_f1
        best_auc = svm_auc
        use_scaled = True
        
    joblib.dump(best_model, "wine_model.joblib")
    
    with open("model_info.txt", "w") as f:
        f.write(f"Model: {best_name}\nAccuracy: {best_acc:.4f}\nF1-Score: {best_f1:.4f}\nROC-AUC: {best_auc:.4f}\nUseScaled: {use_scaled}\n")
        
    print("=== 3. EXPORTING VISUALIZATIONS ===")
    
    # 1. Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Standar (<6)', 'Tinggi (>=6)'],
                yticklabels=['Standar (<6)', 'Tinggi (>=6)'])
    axes[0].set_title('Confusion Matrix - Random Forest', fontsize=12, fontweight='bold')
    
    cm_svm = confusion_matrix(y_test, svm_pred)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Standar (<6)', 'Tinggi (>=6)'],
                yticklabels=['Standar (<6)', 'Tinggi (>=6)'])
    axes[1].set_title('Confusion Matrix - SVM', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('static/images/confusion_matrix.png', dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    
    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)
    
    ax.plot(fpr_rf, tpr_rf, color='#58a6ff', lw=2.5, label=f'Random Forest (AUC = {rf_auc:.4f})')
    ax.plot(fpr_svm, tpr_svm, color='#3fb950', lw=2.5, label=f'SVM (AUC = {svm_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Baseline (AUC = 0.50)')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('Kurva ROC-AUC: Random Forest vs SVM', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', facecolor='#161b22', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig('static/images/roc_curve.png', dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    
    # 3. Feature Importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    bars = ax.barh(sorted_features, sorted_importances, color='#1f6beb', edgecolor='#58a6ff')
    ax.set_xlabel('Feature Importance Score', fontsize=11)
    ax.set_title('Pengaruh Atribut Kimia terhadap Kualitas Wine', fontsize=13, fontweight='bold')
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', color='white', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png', dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    
    print("=== MODEL TRAINING & EXPORT COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    train_and_export()
