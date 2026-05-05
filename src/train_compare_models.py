import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

os.makedirs("models", exist_ok=True)

# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv("data/processed_dataset.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date'])

    # Remove low volatility rows (optional)
    df = df[df['volatility'] > 0.005]

    features = [
        'return', 'return_5', 'return_10', 'return_20', 'return_60',
        'ma10', 'ma20',
        'volatility', 'rsi',
        'price_change', 'high_low_range', 'volume_change', 'ma_ratio'
    ]

    X = df[features]
    y = df['target']

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test, features


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.6).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    return f1


# =========================
# TRAIN + COMPARE MODELS
# =========================
def train_and_compare():
    X_train, X_test, y_train, y_test, features = load_data()

    results = {}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    results["Random Forest"] = (rf, evaluate("Random Forest", rf, X_test, y_test))

    # Logistic Regression
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    results["Logistic Regression"] = (
        lr_pipeline,
        evaluate("Logistic Regression", lr_pipeline, X_test, y_test)
    )

    # XGBoost
    xgb = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric='logloss'
)
    xgb.fit(X_train, y_train)
    results["XGBoost"] = (xgb, evaluate("XGBoost", xgb, X_test, y_test))

    # Select best model (based on F1)
    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]

    print(f"\nBest Model: {best_name}")

    # Save model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Save features
    with open("models/features.pkl", "wb") as f:
        pickle.dump(features, f)

    print("Model and features saved successfully!")


if __name__ == "__main__":
    train_and_compare()