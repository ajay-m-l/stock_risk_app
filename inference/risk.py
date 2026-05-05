def get_risk(volatility):
    if volatility < 0.01:
        return "LOW"
    elif volatility < 0.02:
        return "MEDIUM"
    else:
        return "HIGH"