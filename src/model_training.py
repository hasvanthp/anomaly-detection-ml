import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from fraud_detection.src.data_preprocessing import load_and_preprocess

print("🔹 Loading dataset...")
X_train, X_test, y_train, y_test, scaler = load_and_preprocess("fraud_detection\data\creditcard.csv")

print("🚀 Training XGBoost Fraud Detection Model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"✅ AUC Score: {auc:.4f}\n")
print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model + scaler
joblib.dump(model, "fraud_detection/models/xgboost_fraud_model.pkl")
joblib.dump(scaler, "fraud_detection/models/scaler.pkl")
print("\n💾 Model saved successfully at: fraud_detection/models/xgboost_fraud_model.pkl")
