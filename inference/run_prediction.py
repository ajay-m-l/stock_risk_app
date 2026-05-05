from inference.fetch_data import fetch_data
from inference.feature_pipeline import create_features, get_latest_features
from inference.predictor import predict_stock
from inference.risk import get_risk


def run(stock):
    df = fetch_data(stock)
    df = create_features(df)

    input_data = get_latest_features(df)

    prediction, prob = predict_stock(input_data)
    risk = get_risk(input_data['volatility'])

    print(f"\nStock: {stock}")
    print("Prediction:", prediction)
    print("Confidence:", round(prob, 4))
    print("Risk Level:", risk)


if __name__ == "__main__":
    run("HDFCBANK.NS")