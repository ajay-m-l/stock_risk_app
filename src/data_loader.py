import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(ticker):
    end_date=datetime.today()
    start_date = end_date - timedelta(days=5*365)

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d"
    )

    data = data.dropna()
    return data