import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

def train_and_save():
    print("[+] Loading Heart Disease Dataset...")
    data_path = os.path.join('dataset', 'heart.csv')
    if not os.path.exists(data_path):
        data_path = 'heart.csv'
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[+] Training Best SVM Model...")
    svm_model = SVC(kernel='linear', C=0.01, probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    print("[+] Training Best XGBoost Model...")
    xgb_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)

    print("[+] Saving Trained Models and Scaler...")
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(xgb_model, 'xgb_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("SUCCESS: Models saved as svm_model.joblib, xgb_model.joblib, and scaler.joblib!")

if __name__ == '__main__':
    train_and_save()
