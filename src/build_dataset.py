import pandas as pd
import os
from config import STOCKS
from data_loader import fetch_stock_data


def build_dataset():
    all_data = []

    for stock in STOCKS:
        print(f"Fetching data for {stock}...")

        try:
            df = fetch_stock_data(stock)

            if df.empty:
                print(f"Skipping {stock}")
                continue

            # 🔥 Fix: flatten columns if multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Reset index → Date column
            df = df.reset_index()

            # Add stock name
            df['stock'] = stock

            # Keep only required columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'stock']]

            # Drop any remaining NaN rows
            df = df.dropna()

            all_data.append(df)

        except Exception as e:
            print(f"Error fetching {stock}: {e}")
            continue

    if not all_data:
        print("No data collected!")
        return pd.DataFrame()

    # Combine all stocks
    final_df = pd.concat(all_data, ignore_index=True)

    # Sort properly
    final_df = final_df.sort_values(by=["Date", "stock"])

    return final_df


if __name__ == "__main__":
    data = build_dataset()

    if not data.empty:
        print("\nSample Data:")
        print(data.head())

        print("\nShape:", data.shape)

        # Create data folder if not exists
        os.makedirs("data", exist_ok=True)

        # Save dataset
        data.to_csv("data/final_dataset.csv", index=False)

        print("\n✅ Dataset saved at: data/final_dataset.csv")

    else:
        print("Dataset is empty.")