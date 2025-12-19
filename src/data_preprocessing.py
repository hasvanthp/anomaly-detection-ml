import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess(data_path="../data/creditcard.csv"):
    print("🔹 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop missing values
    df = df.dropna()
    print(f"✅ After dropping nulls: {df.shape[0]} rows remain\n")

    # Class distribution
    print("🔍 Class distribution:")
    print(df["Class"].value_counts())
    sns.countplot(x="Class", data=df, palette="coolwarm")
    plt.title("Fraud vs Legit Transactions")
    plt.show()

    # Split data
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print(f"✅ Data preprocessing complete!\nTraining set: {len(X_train)} samples\nTesting set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test, scaler