import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load trained models & scaler from models/ directory
models_dir = 'models'
scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
svm_model = joblib.load(os.path.join(models_dir, 'svm_model.joblib'))
xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.joblib'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # Extract features
        age = float(data.get('age', 55))
        sex = float(data.get('sex', 1))
        cp = float(data.get('cp', 1))
        trestbps = float(data.get('trestbps', 130))
        chol = float(data.get('chol', 240))
        fbs = float(data.get('fbs', 0))
        restecg = float(data.get('restecg', 0))
        thalach = float(data.get('thalach', 150))
        exang = float(data.get('exang', 0))
        oldpeak = float(data.get('oldpeak', 1.0))
        slope = float(data.get('slope', 1))
        ca = float(data.get('ca', 0))
        thal = float(data.get('thal', 3))

        input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        scaled_features = scaler.transform(input_features)

        # SVM Prediction
        svm_prob = float(svm_model.predict_proba(scaled_features)[0][1])
        svm_pred = int(svm_prob > 0.5)

        # XGBoost Prediction
        xgb_prob = float(xgb_model.predict_proba(scaled_features)[0][1])
        xgb_pred = int(xgb_prob > 0.5)

        return jsonify({
            'status': 'success',
            'svm': {
                'prediction': svm_pred,
                'label': 'POSITIF (Sakit Jantung)' if svm_pred == 1 else 'NEGATIF (Sehat)',
                'probability': round(svm_prob * 100, 2)
            },
            'xgb': {
                'prediction': xgb_pred,
                'label': 'POSITIF (Sakit Jantung)' if xgb_pred == 1 else 'NEGATIF (Sehat)',
                'probability': round(xgb_prob * 100, 2)
            },
            'summary': {
                'final_verdict': 'POSITIF (Risiko Tinggi Penyakit Jantung)' if (svm_pred == 1 or xgb_pred == 1) else 'NEGATIF (Risiko Rendah / Sehat)',
                'risk_level': 'Tinggi' if (svm_prob > 0.5 or xgb_prob > 0.5) else 'Rendah'
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
