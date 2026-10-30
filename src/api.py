"""FastAPI service exposing the winning forecasting model."""

import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import N_LAGS, feature_row
from .models import ArimaModel, NaiveSeasonal, XGBModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")

app = FastAPI(title="Time-Series Forecasting API")
ARTIFACT = joblib.load(ARTIFACT_PATH)
LAST_DATE = pd.Timestamp(ARTIFACT["last_date"])


class ForecastRequest(BaseModel):
    periods: int = Field(12, ge=1, le=36, description="months to forecast (1-36)")


def _forecast(periods):
    name = ARTIFACT["model_name"]
    if name == "arima":
        model = ArimaModel.__new__(ArimaModel)
        model.result_ = ARTIFACT["fitted"]
        preds = model.forecast(periods)
    elif name == "xgboost":
        model = XGBModel.__new__(XGBModel)
        model.model_ = ARTIFACT["fitted"]
        values = list(ARTIFACT["last_values"])
        dates = [LAST_DATE - pd.DateOffset(months=i)
                 for i in reversed(range(len(values)))]
        model.history_ = pd.Series(values, index=pd.DatetimeIndex(dates))
        preds = model.forecast(periods)
    elif name == "naive_seasonal":
        model = NaiveSeasonal().fit(pd.Series(ARTIFACT["last_values"]))
        preds = model.forecast(periods)
    else:
        raise RuntimeError(f"unknown model in artifact: {name}")
    return preds


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model": ARTIFACT["model_name"],
        "metrics": ARTIFACT["metrics"],
        "history_end": ARTIFACT["last_date"],
        "max_periods": 36,
    }


@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        preds = _forecast(req.periods)
    except Exception as exc:  # noqa: BLE001 - surface as 500 with message
        raise HTTPException(status_code=500, detail=str(exc))
    dates = pd.date_range(LAST_DATE + pd.DateOffset(months=1),
                          periods=req.periods, freq="MS")
    return {
        "model": ARTIFACT["model_name"],
        "forecasts": [{"date": d.strftime("%Y-%m"), "value": round(float(v), 1)}
                      for d, v in zip(dates, preds)],
    }
