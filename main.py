from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# ------------- Request & Response Models -------------
class PredictionRequest(BaseModel):
    ticker: str
    investment_amount: float
    investment_days: int

class PredictionResponse(BaseModel):
    ticker: str
    investment_amount: float
    investment_days: int
    predicted_return_percent: float
    predicted_value: float
    risk_level: str
    data_points_used: int
    data_source: str

# ------------- Fallback Data -------------
def get_synthetic_data():
    logging.warning("Using synthetic fallback data.")
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=236)
    prices = np.cumsum(np.random.normal(loc=0.1, scale=1.0, size=len(dates))) + 3500
    return pd.DataFrame({'date': dates, 'close': prices})

# ------------- Risk Analysis -------------
def assess_risk(pred_return):
    if pred_return > 0.10:
        return "High Gain (Moderate Risk)"
    elif pred_return > 0.03:
        return "Low Gain or Neutral (Low Risk)"
    elif pred_return < -0.05:
        return "High Risk of Loss"
    else:
        return "Slight Risk (Stable)"

# ------------- Training and Prediction Logic -------------
def train_and_predict(df, investment_days, investment_amount, ticker):
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

    latest_return = df.iloc[-1]['return']
    pred = model.predict([[latest_return]])[0]
    predicted_value = investment_amount * (1 + pred)

    return {
        "ticker": ticker,
        "investment_amount": investment_amount,
        "investment_days": investment_days,
        "predicted_return_percent": round(pred * 100, 2),
        "predicted_value": round(predicted_value, 2),
        "risk_level": assess_risk(pred),
        "data_points_used": len(df),
        "data_source": "synthetic fallback data"
    }

# ------------- API Endpoint -------------
@app.post("/predict", response_model=PredictionResponse)
def predict_stock_return(request: PredictionRequest):
    df = get_synthetic_data()  # replace this with actual data fetching if available
    result = train_and_predict(df, request.investment_days, request.investment_amount, request.ticker)
    return result
