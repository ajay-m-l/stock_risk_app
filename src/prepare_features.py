import pandas as pd
import numpy as np
import os


# 🔹 Add financial features
def add_features(df):
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort properly (VERY IMPORTANT)
    df = df.sort_values(by=['stock', 'Date'])

    # 1. Returns (log return for better stability)
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. Moving Average (MA20)
    df['ma20'] = (
        df.groupby('stock')['Close']
        .rolling(window=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 3. Volatility (std of returns)
    df['volatility'] = (
        df.groupby('stock')['return']
        .rolling(window=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df


# 🔹 Add target (UP / DOWN)
def add_target(df):
    df['target'] = (
        df.groupby('stock')['Close'].shift(-1) > df['Close']
    ).astype(int)

    return df


# 🔹 Add risk based on volatility
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


# 🔹 Main pipeline
def prepare_dataset():
    # Load dataset
    df = pd.read_csv("data/final_dataset.csv")

    # Apply transformations
    df = add_features(df)
    df = add_target(df)
    df = add_risk(df)

    # Drop NaN values (from rolling + shift)
    df = df.dropna()

    # Keep only important columns
    df = df[
        ['Date', 'stock', 'return', 'ma20', 'volatility', 'target', 'risk']
    ]

    # Sort final dataset
    df = df.sort_values(by=['stock', 'Date'])

    # Create folder if not exists
    os.makedirs("data", exist_ok=True)

    # Save dataset
    df.to_csv("data/processed_dataset.csv", index=False)

    print("✅ Processed dataset saved at: data/processed_dataset.csv")
    print("Shape:", df.shape)
    print("\nSample Data:")
    print(df.head())


# 🔹 Run script
if __name__ == "__main__":
    prepare_dataset()