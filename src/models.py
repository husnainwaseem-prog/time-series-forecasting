"""The three candidate models, all sharing a fit(y) / forecast(n) interface.

- NaiveSeasonal: repeats the last 12 months (seasonal naive baseline).
- ArimaModel: SARIMAX with a small AIC grid search for (p,d,q)x(P,D,Q,12).
- XGBModel: XGBoost on lag + rolling-mean + calendar features, recursive
  multi-step forecasting.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from .features import N_LAGS, feature_row, make_frame

ARIMA_GRID = [((1, 1, 1), (1, 1, 1, 12)),
              ((2, 1, 1), (1, 1, 1, 12)),
              ((1, 1, 1), (0, 1, 1, 12)),
              ((0, 1, 2), (0, 1, 1, 12))]


class NaiveSeasonal:
    name = "naive_seasonal"
    label = "Naive seasonal baseline"

    def __init__(self, period=12):
        self.period = period

    def fit(self, y):
        self.last_ = list(np.asarray(y, dtype=float)[-self.period:])
        return self

    def forecast(self, n):
        reps = (n // self.period) + 1
        return np.array((self.last_ * reps)[:n], dtype=float)


class ArimaModel:
    name = "arima"
    label = "ARIMA (SARIMAX, AIC-selected)"

    def __init__(self, order=None, seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order

    @staticmethod
    def select_order(y):
        best, best_aic = None, np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order, seasonal in ARIMA_GRID:
                try:
                    res = SARIMAX(y, order=order, seasonal_order=seasonal,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False).fit(disp=False)
                except Exception:
                    continue
                if res.aic < best_aic:
                    best, best_aic = (order, seasonal), res.aic
        if best is None:
            raise RuntimeError("no ARIMA order converged")
        return best

    def fit(self, y):
        if self.order is None:
            self.order, self.seasonal_order = self.select_order(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = SARIMAX(
                y, order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
        return self

    def forecast(self, n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.result_.get_forecast(steps=n).predicted_mean
        return np.asarray(fc, dtype=float)


class XGBModel:
    name = "xgboost"
    label = "XGBoost (lag + calendar features)"

    def __init__(self):
        self.model_ = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=2,
        )

    def fit(self, y):
        X, target = make_frame(y)
        self.model_.fit(X, target)
        self.history_ = y.copy()
        self.t_offset_ = 0  # history_ starts at t=0 when fitted on the full series
        return self

    def forecast(self, n):
        values = list(np.asarray(self.history_, dtype=float))
        dates = list(self.history_.index)
        preds = []
        for _ in range(n):
            next_date = dates[-1] + pd.DateOffset(months=1)
            # t must continue the training time index even when only a short
            # tail of history is kept (API artifact) -> t_offset_ corrects it.
            t = getattr(self, "t_offset_", 0) + len(values)
            row = feature_row(values[-N_LAGS:], next_date, t)
            pred = float(self.model_.predict(pd.DataFrame([row]))[0])
            preds.append(pred)
            values.append(pred)
            dates.append(next_date)
        return np.array(preds, dtype=float)


MODELS = {"naive_seasonal": NaiveSeasonal,
          "arima": ArimaModel,
          "xgboost": XGBModel}
