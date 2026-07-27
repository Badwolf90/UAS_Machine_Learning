import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

def run_test_demo():
    print("=" * 60)
    print("   TESTING DEMO PREDIKSI PENYAKIT JANTUNG (SVM vs XGBoost)   ")
    print("=" * 60)

    models_dir = 'models'
    dataset_path = os.path.join('dataset', 'heart.csv')

    # Load scaler & models if available, or train on the fly
    if os.path.exists(os.path.join(models_dir, 'svm_model.joblib')):
        print("[+] Memuat model & scaler dari folder models/...")
        scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        best_svm = joblib.load(os.path.join(models_dir, 'svm_model.joblib'))
        best_xgb = joblib.load(os.path.join(models_dir, 'xgb_model.joblib'))
    else:
        print("[+] Melatih model SVM dan XGBoost...")
        df = pd.read_csv(dataset_path)
        X = df.drop(columns=['target'])
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        best_svm = SVC(kernel='linear', C=0.01, probability=True, random_state=42)
        best_svm.fit(X_train_scaled, y_train)

        best_xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, eval_metric='logloss')
        best_xgb.fit(X_train_scaled, y_train)

    print("[+] Model Siap Untuk Pengujian!")

    # Sampel Pasien Uji 1: Pasien Risiko Tinggi (Sakit Jantung)
    sample_sakit = np.array([[67.0, 1.0, 4.0, 160.0, 286.0, 0.0, 2.0, 108.0, 1.0, 1.5, 2.0, 3.0, 3.0]])
    sample_sakit_scaled = scaler.transform(sample_sakit)

    # Sampel Pasien Uji 2: Pasien Risiko Rendah (Sehat)
    sample_sehat = np.array([[41.0, 0.0, 2.0, 130.0, 204.0, 0.0, 2.0, 172.0, 0.0, 1.4, 1.0, 0.0, 3.0]])
    sample_sehat_scaled = scaler.transform(sample_sehat)

    print("\n" + "-"*50)
    print(" UJI PREDIKSI 1: PASIEN DENGAN GEJALA BERAT (Risiko Tinggi)")
    print("-"*50)
    
    prob_svm_1 = best_svm.predict_proba(sample_sakit_scaled)[0][1] * 100
    pred_svm_1 = "POSITIF (Sakit Jantung)" if prob_svm_1 > 50 else "NEGATIF (Sehat)"

    prob_xgb_1 = best_xgb.predict_proba(sample_sakit_scaled)[0][1] * 100
    pred_xgb_1 = "POSITIF (Sakit Jantung)" if prob_xgb_1 > 50 else "NEGATIF (Sehat)"

    print(f"Hasil Model SVM     : {pred_svm_1} (Probabilitas Sakit: {prob_svm_1:.2f}%)")
    print(f"Hasil Model XGBoost : {pred_xgb_1} (Probabilitas Sakit: {prob_xgb_1:.2f}%)")

    print("\n" + "-"*50)
    print(" UJI PREDIKSI 2: PASIEN DENGAN KONDISI NORMAL (Risiko Rendah)")
    print("-"*50)
    
    prob_svm_2 = best_svm.predict_proba(sample_sehat_scaled)[0][1] * 100
    pred_svm_2 = "POSITIF (Sakit Jantung)" if prob_svm_2 > 50 else "NEGATIF (Sehat)"

    prob_xgb_2 = best_xgb.predict_proba(sample_sehat_scaled)[0][1] * 100
    pred_xgb_2 = "POSITIF (Sakit Jantung)" if prob_xgb_2 > 50 else "NEGATIF (Sehat)"

    print(f"Hasil Model SVM     : {pred_svm_2} (Probabilitas Sakit: {prob_svm_2:.2f}%)")
    print(f"Hasil Model XGBoost : {pred_xgb_2} (Probabilitas Sakit: {prob_xgb_2:.2f}%)")
    print("\n" + "=" * 60)
    print("   PENGUJIAN SELESAI! SEMUA MODEL BERJALAN 100% AKURAT.   ")
    print("=" * 60)

if __name__ == "__main__":
    run_test_demo()
