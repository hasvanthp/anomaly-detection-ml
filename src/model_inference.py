import joblib
import numpy as np

MODEL_PATH = "../models/xgboost_fraud_model.pkl"
SCALER_PATH = "../models/scaler.pkl"

def predict_transaction(transaction_features):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(np.array(transaction_features).reshape(1, -1))
    pred = model.predict(X_scaled)[0]
    risk = model.predict_proba(X_scaled)[0][1]
    return pred, risk