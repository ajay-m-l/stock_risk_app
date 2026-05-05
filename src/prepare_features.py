import pandas as pd
import numpy as np
import os


# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['stock', 'Date'])

    # 🔹 Log return (1-day)
    df['return'] = df.groupby('stock')['Close'].transform(
    lambda x: np.log(x / x.shift(1))
)

    # 🔹 Multi-timeframe returns
    df['return_5'] = df.groupby('stock')['Close'].pct_change(5)
    df['return_10'] = df.groupby('stock')['Close'].pct_change(10)
    df['return_20'] = df.groupby('stock')['Close'].pct_change(20)
    df['return_60'] = df.groupby('stock')['Close'].pct_change(60)

    # 🔹 Moving averages
    df['ma10'] = df.groupby('stock')['Close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('stock')['Close'].transform(lambda x: x.rolling(20).mean())

    # 🔹 Volatility (20-day)
    df['volatility'] = df.groupby('stock')['return'].transform(lambda x: x.rolling(20).std())

    # 🔹 RSI (14-day)
    delta = df.groupby('stock')['Close'].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.groupby(df['stock']).transform(lambda x: x.rolling(14).mean())
    avg_loss = loss.groupby(df['stock']).transform(lambda x: x.rolling(14).mean())

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 🔥 NEW FEATURES

    # Price movement
    df['price_change'] = df['Close'] - df['Open']

    # Daily range
    df['high_low_range'] = df['High'] - df['Low']

    # Volume change
    df['volume_change'] = df.groupby('stock')['Volume'].pct_change()

    # Trend strength
    df['ma_ratio'] = df['Close'] / df['ma20']

    return df


# =========================
# TARGET CREATION
# =========================
def add_target(df):
    future_return = df.groupby('stock')['Close'].shift(-10) / df['Close'] - 1

    threshold = 0.015
    df['target'] = (future_return > threshold).astype(int)

    return df


# =========================
# RISK CLASSIFICATION
# =========================
def add_risk(df):
    def classify(vol):
        if vol < 0.01:
            return "LOW"
        elif vol < 0.02:
            return "MEDIUM"
        else:
            return "HIGH"

    df['risk'] = df['volatility'].apply(classify)
    return df


# =========================
# MAIN PIPELINE
# =========================
def prepare_dataset():
    print("📥 Loading dataset...")
    df = pd.read_csv("data/final_dataset.csv")

    print("⚙️ Adding features...")
    df = add_features(df)

    print("🎯 Creating target...")
    df = add_target(df)

    print("⚠️ Calculating risk...")
    df = add_risk(df)

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 🔥 Drop NaN values
    df = df.dropna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(-10, 10)

    # 🔹 Final dataset columns
    df = df[
        [
            'Date', 'stock',
            'return', 'return_5', 'return_10', 'return_20', 'return_60',
            'ma10', 'ma20',
            'volatility', 'rsi',
            'price_change', 'high_low_range', 'volume_change', 'ma_ratio',
            'target', 'risk'
        ]
    ]

    df = df.sort_values(by=['stock', 'Date'])

    # 🔹 Save dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_dataset.csv", index=False)

    print("\n✅ Processed dataset saved at data/processed_dataset.csv")
    print("Shape:", df.shape)
    print("\nSample Data:")
    print(df.head())


# =========================
# RUN SCRIPT
# =========================
if __name__ == "__main__":
    prepare_dataset()