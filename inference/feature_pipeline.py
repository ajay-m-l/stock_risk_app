import pandas as pd
import numpy as np


def create_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    if len(df) < 60:
        raise ValueError("Not enough data to generate features")

    df['return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['return_5'] = df['Close'].pct_change(5)
    df['return_10'] = df['Close'].pct_change(10)
    df['return_20'] = df['Close'].pct_change(20)
    df['return_60'] = df['Close'].pct_change(60)

    df['ma10'] = df['Close'].rolling(10).mean()
    df['ma20'] = df['Close'].rolling(20).mean()

    df['volatility'] = df['return'].rolling(20).std()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['price_change'] = df['Close'] - df['Open']
    df['high_low_range'] = df['High'] - df['Low']
    df['volume_change'] = df['Volume'].pct_change()

    # safer division
    df['ma_ratio'] = df['Close'] / (df['ma20'] + 1e-9)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def get_latest_features(df):
    latest = df.iloc[-1]

    return {
        'return': latest['return'],
        'return_5': latest['return_5'],
        'return_10': latest['return_10'],
        'return_20': latest['return_20'],
        'return_60': latest['return_60'],
        'ma10': latest['ma10'],
        'ma20': latest['ma20'],
        'volatility': latest['volatility'],
        'rsi': latest['rsi'],
        'price_change': latest['price_change'],
        'high_low_range': latest['high_low_range'],
        'volume_change': latest['volume_change'],
        'ma_ratio': latest['ma_ratio']
    }