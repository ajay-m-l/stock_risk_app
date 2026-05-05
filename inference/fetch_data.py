import yfinance as yf
import pandas as pd

def fetch_data(stock):
    df = yf.download(stock, period="120d")

    df.reset_index(inplace=True)

    # 🔥 FIX: Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    return df