# import os
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # ==============================
# # 1️⃣ Basic Setup
# # ==============================
# st.set_page_config(page_title="💳 Real-Time Fraud Detection", layout="wide")
#
# st.title("💳 Real-Time Fraud Detection System")
# st.write("Detect fraudulent transactions using an XGBoost model trained on financial data.")
#
# # ==============================
# # 2️⃣ Model + Scaler Loading
# # ==============================
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgboost_fraud_model.pkl")
# SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")
#
# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)
#
# # ==============================
# # 3️⃣ Manual Input Section
# # ==============================
# st.sidebar.header("🔧 Input Transaction Details")
#
# amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, max_value=20000.0, value=100.0)
# time = st.sidebar.number_input("Transaction Time (sec)", min_value=0, max_value=172800, value=10000)
# v_features = [st.sidebar.number_input(f"V{i}", value=0.0) for i in range(1, 29)]
#
# if st.sidebar.button("🚀 Predict Fraud"):
#     features = np.array([[time, *v_features, amount]])
#     scaled = scaler.transform(features)
#     prediction = model.predict(scaled)[0]
#     risk = model.predict_proba(scaled)[0][1]
#
#     if prediction == 1:
#         st.error(f"⚠️ Fraudulent Transaction Detected! (Risk Score: {risk:.4f})")
#     else:
#         st.success(f"✅ Legitimate Transaction (Risk Score: {risk:.4f})")
#
# # ==============================
# # 4️⃣ Batch Upload Section
# # ==============================
# st.divider()
# st.subheader("📊 Batch Transaction Analysis")
#
# uploaded = st.file_uploader("Upload CSV File", type=["csv"])
# if uploaded:
#     data = pd.read_csv(uploaded)
#     st.write("### Uploaded Data", data.head())
#
#     # Expected feature structure
#     EXPECTED_FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
#
#     # Add missing features as 0
#     for col in EXPECTED_FEATURES:
#         if col not in data.columns:
#             data[col] = 0
#
#     # Keep only relevant columns
#     data = data[EXPECTED_FEATURES]
#
#     # Scale and predict
#     scaled_data = scaler.transform(data)
#     preds = model.predict(scaled_data)
#     data["Prediction"] = preds
#
#     st.write("### Prediction Results", data.head())
#
#     # Visualization
#     fig, ax = plt.subplots()
#     sns.countplot(x="Prediction", data=data, palette="coolwarm", ax=ax)
#     ax.set_title("Fraud vs Legit Predictions")
#     st.pyplot(fig)

# fraud_detection/app.py
import os
import re
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Config & paths
# -------------------------
st.set_page_config(page_title="💳 Fraud Detection System", layout="wide")
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# -------------------------
# Load model + scaler
# -------------------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("❌ Model or scaler file not found in `models/`. Train the model first (train_model.py).")
    st.stop()

# -------------------------
# Expected features used by model (exact names & order)
# -------------------------
EXPECTED_FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# -------------------------
# Small helpers for column normalization
# -------------------------
def normalize_column_name(col: str) -> str:
    """Normalize a single column string for matching."""
    c = col.strip()
    c_lower = c.lower()
    # common mappings
    if c_lower in ("amount", "amt", "transactionamount", "value"):
        return "Amount"
    if c_lower in ("time", "timestamp", "sec", "seconds"):
        return "Time"
    # match v1..v28 flexible patterns: v01, v1, v_1, v-1, pca1 etc.
    m = re.match(r'^(?:v|pca|component|pc|feat|feature)[\s_\-]*(\d{1,2})$', c_lower)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= 28:
            return f"V{idx}"
    # direct V# match like 'V1' or 'v1' or 'V01'
    m2 = re.match(r'^[^\d]*?(\d{1,2})$', c_lower)
    if m2:
        # cautious: only return numeric if the rest of name isn't something else ambiguous
        num = int(m2.group(1))
        if 1 <= num <= 28 and c_lower.startswith(('v', 'v0', 'v')):
            return f"V{num}"
    # fallback: preserve capitalization
    return c

def map_columns_to_expected(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map columns in df to EXPECTED_FEATURES where possible (case-insensitive & flexible).
    Any missing expected columns are created with zeros.
    Extra columns are ignored.
    """
    orig_cols = list(df.columns)
    mapping = {}
    # Build normalized -> original list (handle collisions by first occurrence)
    norm_to_orig = {}
    for col in orig_cols:
        norm = normalize_column_name(col)
        # keep first mapping for a normalized name
        if norm not in norm_to_orig:
            norm_to_orig[norm] = col

    # Prepare a result DataFrame with expected features in order
    result = pd.DataFrame(index=df.index)
    for feat in EXPECTED_FEATURES:
        if feat in norm_to_orig:                       # exact normalized match found
            result[feat] = df[norm_to_orig[feat]]
        else:
            # try case-insensitive exact match
            found = None
            for col in orig_cols:
                if col.strip().lower() == feat.lower():
                    found = col
                    break
            if found:
                result[feat] = df[found]
            else:
                # try to find columns like 'v01' for V1, etc.
                if feat.startswith("V"):
                    num = feat[1:]
                    pattern = re.compile(rf'\D*{num}\D*$', re.IGNORECASE)
                    candidate = None
                    for col in orig_cols:
                        if pattern.search(col):
                            candidate = col
                            break
                    if candidate:
                        result[feat] = df[candidate]
                        continue
                # if not found at all, create zeros
                result[feat] = 0.0
    return result

# -------------------------
# Title & description
# -------------------------
st.title("💳 Real-Time Fraud Detection System")
st.write("Detect fraudulent transactions using an **XGBoost** model trained on financial data.")

# -------------------------
# Sidebar: Autofill + Inputs
# -------------------------
st.sidebar.header("🔧 Input Transaction Details")

# session state keys
keys = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = 0.0

# predefined examples
fraud_values = {"Amount": 15000.0, "Time": 85000}
fraud_list = [-4.2, 3.9, -5.1, 4.6, -3.7, 3.2, -4.9, 5.4, -3.8, 4.3,
              -4.5, 3.7, -3.9, 4.8, -5.2, 4.5, -3.6, 3.8, -4.1, 4.9,
              -3.4, 4.7, -4.0, 4.3, -3.5, 3.9, -4.6, 4.2]
for i, v in enumerate(fraud_list, 1):
    fraud_values[f"V{i}"] = v

borderline_values = {"Amount": 1200.0, "Time": 45000}
border_list = [-1.2, 0.9, -1.5, 1.1, -0.8, 0.7, -1.0, 1.3, -0.9, 1.0,
               -1.1, 0.8, -0.7, 1.2, -1.3, 0.9, -0.6, 0.7, -1.0, 1.1,
               -0.5, 1.0, -0.8, 0.9, -0.6, 0.7, -1.0, 1.0]
for i, v in enumerate(border_list, 1):
    borderline_values[f"V{i}"] = v

col1, col2 = st.sidebar.columns(2)
if col1.button("⚠️ Fill Fraud Example"):
    for k, val in fraud_values.items():
        st.session_state[k] = val
if col2.button("🟡 Fill Borderline"):
    for k, val in borderline_values.items():
        st.session_state[k] = val

# render inputs reading/writing session_state
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, max_value=200000.0,
                                 value=float(st.session_state["Amount"]), key="Amount")
time = st.sidebar.number_input("Transaction Time (sec)", min_value=0, max_value=10**7,
                               value=int(st.session_state["Time"]), key="Time")
v_features = [st.sidebar.number_input(f"V{i}", value=float(st.session_state[f"V{i}"]), key=f"V{i}") for i in range(1, 29)]

# -------------------------
# Single prediction
# -------------------------
if st.sidebar.button("🚀 Predict Fraud"):
    X_single = pd.DataFrame([[time] + v_features + [amount]], columns=EXPECTED_FEATURES)
    try:
        scaled_single = scaler.transform(X_single)
        pred = model.predict(scaled_single)[0]
        prob = model.predict_proba(scaled_single)[0][1]
        if pred == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! (Risk Score: {prob:.4f})")
        else:
            st.success(f"✅ Legitimate Transaction (Risk Score: {prob:.4f})")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------
# Batch upload and prediction
# -------------------------
st.divider()
st.subheader("📊 Batch Transaction Analysis")

uploaded = st.file_uploader("Upload CSV file (any column names) for batch prediction", type=["csv"])
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()

    st.write("### Uploaded data (first rows)", raw_df.head())

    # Map / normalize / fill to EXPECTED_FEATURES
    mapped = map_columns_to_expected(raw_df)
    st.write("### Auto-normalized features (first rows)", mapped.head())

    # Scaling + predict
    try:
        scaled = scaler.transform(mapped)
        preds = model.predict(scaled)
        mapped["Prediction"] = preds
        st.write("### Prediction Results (first rows)", mapped.head())

        fig, ax = plt.subplots()
        sns.countplot(x="Prediction", data=mapped, palette="coolwarm", ax=ax)
        ax.set_title("Fraud vs Legit Predictions")
        st.pyplot(fig)

        fraud_count = int((mapped["Prediction"] == 1).sum())
        total = len(mapped)
        st.info(f"🔍 Detected {fraud_count} fraudulent transactions out of {total} records.")
    except ValueError as ve:
        st.error("Feature mismatch when scaling/predicting. Details: " + str(ve))
    except Exception as e:
        st.error("Error during prediction: " + str(e))