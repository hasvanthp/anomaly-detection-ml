import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# ✅ Load your new dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Fraud.csv")
df = pd.read_csv(DATA_PATH)

# ✅ Standardize column names (important!)
df.columns = [c.strip().title() for c in df.columns]

# ✅ Ensure required columns exist
required = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
missing = [c for c in required if c not in df.columns]
for m in missing:
    df[m] = 0

# ✅ Separate features and labels
X = df[['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']]
y = df['Class']

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ Save model + scaler
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_fraud_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("✅ Model and scaler saved successfully!")