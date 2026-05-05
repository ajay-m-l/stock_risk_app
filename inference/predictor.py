import pickle
import pandas as pd

# Load model + features
model = pickle.load(open("models/model.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))


def predict_stock(input_data):
    df = pd.DataFrame([input_data])

    # Ensure correct feature order
    df = df[features]

    prob = model.predict_proba(df)[0][1]

    # Decision logic
    if prob > 0.55:
        prediction = "BUY 📈"
    elif prob < 0.4:
        prediction = "SELL 📉"
    else:
        prediction = "HOLD ⏳"

    return prediction, prob