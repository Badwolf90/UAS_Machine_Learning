import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load Model, Scaler, and Info
MODEL_PATH = "wine_model.joblib"
SCALER_PATH = "scaler.joblib"
INFO_PATH = "model_info.txt"

model = None
scaler = None
model_info = {}

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

if os.path.exists(INFO_PATH):
    with open(INFO_PATH, "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                model_info[k.strip()] = v.strip()

@app.route("/")
def index():
    return render_template("index.html", model_info=model_info)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # 11 Chemical Properties
        features = [
            float(data.get("fixed_acidity", 7.4)),
            float(data.get("volatile_acidity", 0.70)),
            float(data.get("citric_acid", 0.00)),
            float(data.get("residual_sugar", 1.9)),
            float(data.get("chlorides", 0.076)),
            float(data.get("free_sulfur_dioxide", 11.0)),
            float(data.get("total_sulfur_dioxide", 34.0)),
            float(data.get("density", 0.9978)),
            float(data.get("ph", 3.51)),
            float(data.get("sulphates", 0.56)),
            float(data.get("alcohol", 9.4))
        ]
        
        X_input = np.array([features])
        
        # Use scaled data if SVM or required
        if model_info.get("UseScaled", "False") == "True" and scaler is not None:
            X_input_final = scaler.transform(X_input)
        else:
            X_input_final = X_input
            
        prediction = int(model.predict(X_input_final)[0])
        probabilities = model.predict_proba(X_input_final)[0]
        prob_high = round(float(probabilities[1]) * 100, 2)
        prob_standard = round(float(probabilities[0]) * 100, 2)
        
        # Recommendations generator
        recommendations = []
        alcohol = features[10]
        volatile_acidity = features[1]
        sulphates = features[9]
        
        if alcohol < 10.0:
            recommendations.append("Kadar alkohol relatif rendah (< 10%). Meningkatkan masirasi fermentasi dapat meningkatkan cita rasa khas wine.")
        elif alcohol > 12.0:
            recommendations.append("Kadar alkohol tinggi (> 12%), memberikan bodi wine yang kuat dan memperpanjang masa simpan.")
            
        if volatile_acidity > 0.6:
            recommendations.append("Peringatan Keasaman Menguap Tinggi (> 0.6 g/L). Berisiko memberikan aroma asam cuka yang menurunkan skor kualitas.")
        else:
            recommendations.append("Keasaman volatil stabil (≤ 0.6 g/L), menjaga aroma segar dan bersih.")
            
        if sulphates < 0.5:
            recommendations.append("Kadar senyawa sulfat agak rendah. Pertimbangkan menambahkan antioksidan alami untuk stabilitas kemasan.")
        else:
            recommendations.append("Kadar sulfat optimal untuk perlindungan antioksidan dan pengawetan rasa alami.")

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "quality_label": "Kualitas Tinggi (High Quality)" if prediction == 1 else "Kualitas Standar (Standard Quality)",
            "prob_high": prob_high,
            "prob_standard": prob_standard,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
