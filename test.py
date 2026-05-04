from src.data_loader import fetch_stock_data

data = fetch_stock_data("RELIANCE.NS")

print(data.head())
print(data.tail())