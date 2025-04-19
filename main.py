import requests
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from json.decoder import JSONDecodeError
from datetime import datetime

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIG ---
API_URL = "https://your-stock-api.com/data?ticker={ticker}"  # Example, replace with real one
TICKER = "TCS.NS"
INVESTMENT_AMOUNT = 10000000.0
INVESTMENT_DAYS = 6


# --- SYNTHETIC FALLBACK DATA FUNCTION ---
def get_synthetic_data():
    logging.warning("Using synthetic fallback data.")
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=236)
    prices = np.cumsum(np.random.normal(loc=0.1, scale=1.0, size=len(dates))) + 3500
    return pd.DataFrame({'date': dates, 'close': prices})


# --- DATA FETCH FUNCTION ---
def fetch_stock_data(ticker):
    try:
        url = API_URL.format(ticker=ticker)
        response = requests.get(url, timeout=5)

        if response.status_code == 200 and response.content.strip():
            try:
                json_data = response.json()
                df = pd.DataFrame(json_data['historical'])
                df = df[['date', 'close']]
                df['date'] = pd.to_datetime(df['date'])
                return df
            except (KeyError, JSONDecodeError) as e:
                logging.error(f"[{ticker}]: JSON decode failed or key error: {e}")
                return get_synthetic_data()
        else:
            logging.error(f"[{ticker}]: API returned status {response.status_code} or empty content.")
            return get_synthetic_data()

    except requests.RequestException as e:
        logging.error(f"[{ticker}]: Request failed: {e}")
        return get_synthetic_data()


# --- MODEL TRAINING & PREDICTION ---
def train_and_predict(df, investment_days, investment_amount):
    df = df.sort_values("date")
    df['return'] = df['close'].pct_change().fillna(0)

    df['future_price'] = df['close'].shift(-investment_days)
    df = df.dropna()

    df['target_return'] = (df['future_price'] - df['close']) / df['close']
    X = df[['return']]
    y = df['target_return']

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
    logging.info(f"Model trained with RMSE: {rmse:.4f}")

    # Make a prediction based on latest data
    latest_return = df.iloc[-1]['return']
    pred = model.predict([[latest_return]])[0]

    predicted_value = investment_amount * (1 + pred)
    result = {
        "ticker": TICKER,
        "investment_amount": investment_amount,
        "investment_days": investment_days,
        "predicted_return_percent": round(pred * 100, 2),
        "predicted_value": round(predicted_value, 2),
        "risk_level": assess_risk(pred),
        "data_points_used": len(df),
        "data_source": "synthetic fallback data" if "synthetic" in df.columns else "real API"
    }

    logging.info(f"Response: {result}")
    return result


# --- RISK ASSESSMENT LOGIC ---
def assess_risk(pred_return):
    if pred_return > 0.10:
        return "High Gain (Moderate Risk)"
    elif pred_return > 0.03:
        return "Low Gain or Neutral (Low Risk)"
    elif pred_return < -0.05:
        return "High Risk of Loss"
    else:
        return "Slight Risk (Stable)"


# --- MAIN FLOW ---
if __name__ == "__main__":
    df = fetch_stock_data(TICKER)
    result = train_and_predict(df, INVESTMENT_DAYS, INVESTMENT_AMOUNT)
