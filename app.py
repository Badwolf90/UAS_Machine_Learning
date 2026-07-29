import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, 'models')

# Load parameters from JSON for ultra-fast serverless execution
params_path = os.path.join(models_dir, 'model_params.json')
if os.path.exists(params_path):
    with open(params_path, 'r') as f:
        params = json.load(f)
    scaler_mean = np.array(params['scaler_mean'])
    scaler_scale = np.array(params['scaler_scale'])
    svm_coef = np.array(params['svm_coef'])
    svm_intercept = float(params['svm_intercept'])
else:
    scaler_mean = np.zeros(13)
    scaler_scale = np.ones(13)
    svm_coef = np.zeros(13)
    svm_intercept = 0.0

# Load XGBoost tree parameters from JSON
xgb_json_path = os.path.join(models_dir, 'xgb_model.json')
xgb_trees = []
xgb_base_margin = 0.0
if os.path.exists(xgb_json_path):
    with open(xgb_json_path, 'r') as f:
        xgb_data = json.load(f)
    xgb_trees = xgb_data['learner']['gradient_booster']['model']['trees']
    base_s_str = xgb_data['learner']['learner_model_param']['base_score'].strip('[]')
    base_s = float(base_s_str)
    xgb_base_margin = float(np.log(base_s / (1.0 - base_s)))

def predict_xgb_trees(scaled_row):
    score = xgb_base_margin
    for tree in xgb_trees:
        lefts = tree['left_children']
        rights = tree['right_children']
        indices = tree['split_indices']
        conds = tree['split_conditions']
        weights = tree['base_weights']
        defaults = tree['default_left']
        node = 0
        while lefts[node] != -1:
            feat = indices[node]
            val = scaled_row[feat]
            if np.isnan(val):
                node = lefts[node] if defaults[node] == 1 else rights[node]
            elif val < conds[node]:
                node = lefts[node]
            else:
                node = rights[node]
        score += weights[node]
    return float(1.0 / (1.0 + np.exp(-score)))

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

        input_features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        scaled_features = (input_features - scaler_mean) / scaler_scale

        # SVM Prediction via Decision Function
        svm_df = np.dot(scaled_features, svm_coef) + svm_intercept
        svm_prob = float(1.0 / (1.0 + np.exp(-svm_df)))
        svm_pred = int(svm_prob > 0.5)

        # XGBoost Prediction via JSON Tree Evaluator
        xgb_prob = predict_xgb_trees(scaled_features)
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
