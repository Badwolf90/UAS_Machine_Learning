import os
import joblib
import pandas as pd
import numpy as np

def load_models():
    models_dir = 'models'
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    svm_path = os.path.join(models_dir, 'svm_model.joblib')
    xgb_path = os.path.join(models_dir, 'xgb_model.joblib')

    if not (os.path.exists(scaler_path) and os.path.exists(svm_path) and os.path.exists(xgb_path)):
        print("[!] Model belum dilatih. Jalankan 'python src/train_and_evaluate.py' terlebih dahulu.")
        return None, None, None

    scaler = joblib.load(scaler_path)
    svm_model = joblib.load(svm_path)
    xgb_model = joblib.load(xgb_path)
    return scaler, svm_model, xgb_model

def predict_patient(scaler, svm_model, xgb_model, features, label_name=""):
    scaled = scaler.transform([features])
    svm_prob = svm_model.predict_proba(scaled)[0][1] * 100
    xgb_prob = xgb_model.predict_proba(scaled)[0][1] * 100

    svm_label = "POSITIF (Sakit Jantung)" if svm_prob > 50 else "NEGATIF (Sehat)"
    xgb_label = "POSITIF (Sakit Jantung)" if xgb_prob > 50 else "NEGATIF (Sehat)"

    print(f"\n==================================================")
    print(f" HASIL PREDIKSI: {label_name}")
    print(f"==================================================")
    print(f"📌 Support Vector Machine (SVM) : {svm_label}")
    print(f"   └─ Probabilitas Sakit: {svm_prob:.2f}%")
    print(f"📌 XGBoost Classifier           : {xgb_label}")
    print(f"   └─ Probabilitas Sakit: {xgb_prob:.2f}%")
    print(f"--------------------------------------------------")
    if svm_prob > 50 or xgb_prob > 50:
        print("⚠️ Kesimpulan: PASIEN MEMILIKI RISIKO TINGGI PENYAKIT JANTUNG")
    else:
        print("✅ Kesimpulan: PASIEN MEMILIKI KONDISI SEHAT / RISIKO RENDAH")
    print(f"==================================================\n")

def interactive_menu():
    scaler, svm_model, xgb_model = load_models()
    if scaler is None:
        return

    while True:
        print("\n" + "="*55)
        print("   INTERACTIVE MODEL TESTER - CARDIOPREDICT AI   ")
        print("="*55)
        print("1. Test Sampel Pasien 1 (Gejala Berat / Risiko Tinggi)")
        print("2. Test Sampel Pasien 2 (Kondisi Normal / Risiko Rendah)")
        print("3. Input Data Pasien Manual (Kustom)")
        print("4. Keluar")
        print("="*55)

        choice = input("Pilih Menu (1-4): ").strip()

        if choice == '1':
            # Age: 67, Sex: 1, CP: 4, Trestbps: 160, Chol: 286, FBS: 0, RestECG: 2, Thalach: 108, Exang: 1, Oldpeak: 1.5, Slope: 2, CA: 3, Thal: 3
            sample = [67.0, 1.0, 4.0, 160.0, 286.0, 0.0, 2.0, 108.0, 1.0, 1.5, 2.0, 3.0, 3.0]
            predict_patient(scaler, svm_model, xgb_model, sample, "Pasien 1 (Risiko Tinggi)")

        elif choice == '2':
            # Age: 41, Sex: 0, CP: 2, Trestbps: 130, Chol: 204, FBS: 0, RestECG: 2, Thalach: 172, Exang: 0, Oldpeak: 1.4, Slope: 1, CA: 0, Thal: 3
            sample = [41.0, 0.0, 2.0, 130.0, 204.0, 0.0, 2.0, 172.0, 0.0, 1.4, 1.0, 0.0, 3.0]
            predict_patient(scaler, svm_model, xgb_model, sample, "Pasien 2 (Risiko Rendah)")

        elif choice == '3':
            print("\n--- Input Data Medis Pasien ---")
            try:
                age = float(input("Usia (thn) [misal: 55]: ") or 55)
                sex = float(input("Jenis Kelamin (1=Pria, 0=Wanita) [misal: 1]: ") or 1)
                cp = float(input("Tipe Nyeri Dada (1-4) [misal: 4]: ") or 4)
                trestbps = float(input("Tekanan Darah (mmHg) [misal: 140]: ") or 140)
                chol = float(input("Kolesterol (mg/dl) [misal: 250]: ") or 250)
                fbs = float(input("Gula Darah Puasa > 120 mg/dl (1=Ya, 0=Tdk) [misal: 0]: ") or 0)
                restecg = float(input("Hasil ECG (0-2) [misal: 1]: ") or 1)
                thalach = float(input("Detak Jantung Maks (bpm) [misal: 130]: ") or 130)
                exang = float(input("Angina Olahraga (1=Ya, 0=Tdk) [misal: 1]: ") or 1)
                oldpeak = float(input("Depresi ST (Oldpeak) [misal: 1.5]: ") or 1.5)
                slope = float(input("Kemiringan ST Slope (1-3) [misal: 2]: ") or 2)
                ca = float(input("Jumlah Pembuluh Utama CA (0-3) [misal: 2]: ") or 2)
                thal = float(input("Thalassemia (3=Normal, 6=Fixed, 7=Reversable) [misal: 7]: ") or 7)

                sample = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                predict_patient(scaler, svm_model, xgb_model, sample, "Pasien Custom Input")
            except Exception as e:
                print(f"[!] Error input data: {e}")

        elif choice == '4':
            print("Keluar dari program. Terima kasih!")
            break
        else:
            print("[!] Pilihan tidak valid. Silakan pilih 1-4.")

if __name__ == "__main__":
    interactive_menu()
