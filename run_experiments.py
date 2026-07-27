import os
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

# Set style for publication quality plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})

def main():
    print("=== STEP 1: LOADING AND PREPROCESSING DATASET ===")
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    # Download and save locally
    df = pd.read_csv(dataset_url, sep=';')
    csv_path = "winequality-red.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path}. Shape: {df.shape}")
    
    # Target transformation: Binary Classification
    # Quality >= 6 -> Good Quality (1), Quality < 6 -> Standard/Poor Quality (0)
    df['target'] = (df['quality'] >= 6).astype(int)
    print(f"Target distribution (0: Standard, 1: High Quality):\n{df['target'].value_counts()}")
    
    X = df.drop(columns=['quality', 'target'])
    y = df['target']
    feature_names = X.columns.tolist()
    
    # Train test split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling (Crucial for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n=== STEP 2: RUNNING 5 EXPERIMENTS FOR RANDOM FOREST ===")
    rf_configs = [
        {"name": "Exp 1 (RF Base)", "params": {"n_estimators": 10, "max_depth": 3, "random_state": 42}},
        {"name": "Exp 2 (RF Tuning 1)", "params": {"n_estimators": 50, "max_depth": 5, "random_state": 42}},
        {"name": "Exp 3 (RF Tuning 2)", "params": {"n_estimators": 100, "max_depth": 10, "criterion": "entropy", "random_state": 42}},
        {"name": "Exp 4 (RF Tuning 3)", "params": {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42}},
        {"name": "Exp 5 (RF Optimal)", "params": {"n_estimators": 300, "max_depth": 20, "min_samples_split": 2, "criterion": "entropy", "random_state": 42}},
    ]
    
    rf_results = []
    best_rf_model = None
    best_rf_f1 = 0
    best_rf_y_prob = None
    best_rf_y_pred = None

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
        
        res = {
            "Experiment": config["name"],
            "Parameters": str(config["params"]),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }
        rf_results.append(res)
        print(f"[{config['name']}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        if f1 > best_rf_f1:
            best_rf_f1 = f1
            best_rf_model = model
            best_rf_y_prob = y_prob
            best_rf_y_pred = y_pred

    print("\n=== STEP 3: RUNNING 5 EXPERIMENTS FOR SUPPORT VECTOR MACHINE (SVM) ===")
    svm_configs = [
        {"name": "Exp 1 (SVM Base)", "params": {"kernel": "linear", "C": 0.1, "probability": True, "random_state": 42}},
        {"name": "Exp 2 (SVM Tuning 1)", "params": {"kernel": "linear", "C": 1.0, "probability": True, "random_state": 42}},
        {"name": "Exp 3 (SVM Tuning 2)", "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True, "random_state": 42}},
        {"name": "Exp 4 (SVM Tuning 3)", "params": {"kernel": "rbf", "C": 10.0, "gamma": "auto", "probability": True, "random_state": 42}},
        {"name": "Exp 5 (SVM Optimal)", "params": {"kernel": "rbf", "C": 50.0, "gamma": "scale", "probability": True, "random_state": 42}},
    ]
    
    svm_results = []
    best_svm_model = None
    best_svm_f1 = 0
    best_svm_y_prob = None
    best_svm_y_pred = None

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
        
        res = {
            "Experiment": config["name"],
            "Parameters": str(config["params"]),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }
        svm_results.append(res)
        print(f"[{config['name']}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        if f1 > best_svm_f1:
            best_svm_f1 = f1
            best_svm_model = model
            best_svm_y_prob = y_prob
            best_svm_y_pred = y_pred

    # Save summary tables to CSV
    df_rf_res = pd.DataFrame(rf_results)
    df_svm_res = pd.DataFrame(svm_results)
    df_rf_res.to_csv("rf_experiments_results.csv", index=False)
    df_svm_res.to_csv("svm_experiments_results.csv", index=False)
    
    print("\n=== STEP 4: GENERATING VISUALIZATIONS AND DIAGRAMS ===")
    
    # 1. Confusion Matrix Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm_rf = confusion_matrix(y_test, best_rf_y_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Kualitas Standar (<6)', 'Kualitas Tinggi (>=6)'],
                yticklabels=['Kualitas Standar (<6)', 'Kualitas Tinggi (>=6)'])
    axes[0].set_title('Confusion Matrix - Random Forest (Optimal)', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Prediksi Model', fontsize=12)
    axes[0].set_ylabel('Aktual Data', fontsize=12)
    
    cm_svm = confusion_matrix(y_test, best_svm_y_pred)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Kualitas Standar (<6)', 'Kualitas Tinggi (>=6)'],
                yticklabels=['Kualitas Standar (<6)', 'Kualitas Tinggi (>=6)'])
    axes[1].set_title('Confusion Matrix - Support Vector Machine (Optimal)', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Prediksi Model', fontsize=12)
    axes[1].set_ylabel('Aktual Data', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png', dpi=300)
    plt.close()
    print("Saved confusion_matrix_comparison.png")
    
    # 2. ROC-AUC Comparison Curve
    plt.figure(figsize=(8, 6))
    fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf_y_prob)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, best_svm_y_prob)
    
    plt.plot(fpr_rf, tpr_rf, color='#1f77b4', lw=2.5, label=f'Random Forest (AUC = {roc_auc_score(y_test, best_rf_y_prob):.4f})')
    plt.plot(fpr_svm, tpr_svm, color='#2ca02c', lw=2.5, label=f'SVM (AUC = {roc_auc_score(y_test, best_svm_y_prob):.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Baseline (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Perbandingan Kurva ROC-AUC: Random Forest vs SVM', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('roc_curve_comparison.png', dpi=300)
    plt.close()
    print("Saved roc_curve_comparison.png")
    
    # 3. Feature Importance Plot (Random Forest)
    importances = best_rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    palette = sns.color_palette("viridis", len(sorted_features))
    bars = plt.barh(sorted_features[::-1], sorted_importances[::-1], color=palette[::-1], edgecolor='black', alpha=0.85)
    
    plt.xlabel('Tingkat Kepentingan Fitur (Feature Importance Score)', fontsize=12)
    plt.ylabel('Atribut Kimia Kualitas Wine', fontsize=12)
    plt.title('Tingkat Pengaruh Atribut terhadap Kualitas Wine (Random Forest)', fontsize=14, fontweight='bold', pad=12)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.003, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                 va='center', ha='left', fontsize=10, fontweight='bold')
                 
    plt.xlim(0, max(sorted_importances) * 1.15)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    print("Saved feature_importance.png")
    
    # 4. 5-Experiment Comparison Graph
    plt.figure(figsize=(12, 6))
    exp_labels = [f"Exp {i+1}" for i in range(5)]
    
    rf_accs = [r["Accuracy"] for r in rf_results]
    svm_accs = [r["Accuracy"] for r in svm_results]
    rf_f1s = [r["F1-Score"] for r in rf_results]
    svm_f1s = [r["F1-Score"] for r in svm_results]
    
    plt.plot(exp_labels, rf_accs, marker='o', linewidth=2.5, markersize=8, label='Random Forest (Accuracy)', color='#1f77b4')
    plt.plot(exp_labels, svm_accs, marker='s', linewidth=2.5, markersize=8, label='SVM (Accuracy)', color='#2ca02c')
    plt.plot(exp_labels, rf_f1s, marker='^', linewidth=2, linestyle='--', markersize=7, label='Random Forest (F1-Score)', color='#aec7e8')
    plt.plot(exp_labels, svm_f1s, marker='D', linewidth=2, linestyle='--', markersize=7, label='SVM (F1-Score)', color='#98df8a')
    
    plt.title('Perkembangan Performa Model dalam 5 Eksperimen Tuning Hyperparameter', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Variasi Eksperimen Hyperparameter', fontsize=12)
    plt.ylabel('Skor Evaluasi (0.0 - 1.0)', fontsize=12)
    plt.ylim([0.60, 0.90])
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('experiments_comparison.png', dpi=300)
    plt.close()
    print("Saved experiments_comparison.png")
    
    print("\n=== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
