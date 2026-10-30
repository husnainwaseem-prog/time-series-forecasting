"""Tests: data, features, models, time-based CV, metrics, and the API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data_loader import load_series
from src.features import N_LAGS, feature_row, make_frame, FEATURE_COLS
from src.evaluate import mae, rmse, time_splits, evaluate
from src.models import NaiveSeasonal, ArimaModel, XGBModel, MODELS, ARIMA_GRID


@pytest.fixture(scope="module")
def series():
    return load_series()


def test_load_series_shape_and_freq(series):
    assert len(series) == 144
    assert series.index.freqstr == "MS"
    assert not series.isna().any()
    assert (series.index == series.index.sort_values()).all()
    assert series.iloc[0] == 112.0  # Jan 1949


def test_feature_row_lags():
    past = list(range(100, 112))  # 12 values, most recent = 111
    row = feature_row(past, pd.Timestamp("1961-01-01"), 144)
    assert row["lag_1"] == 111
    assert row["lag_12"] == 100
    assert row["roll_mean_3"] == np.mean([109, 110, 111])
    assert row["roll_mean_12"] == np.mean(past)
    assert row["month_sin"] == pytest.approx(np.sin(2 * np.pi * 1 / 12))
    assert row["t"] == 144
    with pytest.raises(ValueError):
        feature_row([1, 2, 3], pd.Timestamp("1961-01-01"), 0)


def test_make_frame_shapes(series):
    X, y = make_frame(series)
    assert X.shape == (132, len(FEATURE_COLS))
    assert len(y) == 132
    assert not X.isna().any().any()
    assert X.iloc[0]["lag_1"] == series.iloc[11]  # first row predicts month 12
    assert y.iloc[0] == series.iloc[12]


def test_naive_seasonal_tiles(series):
    train = series.iloc[:120]
    preds = NaiveSeasonal().fit(train).forecast(14)
    assert len(preds) == 14
    np.testing.assert_allclose(preds[:12], train.iloc[-12:].to_numpy())
    np.testing.assert_allclose(preds[12:], train.iloc[-12:-10].to_numpy())


def test_arima_fit_and_forecast(series):
    model = ArimaModel().fit(series.iloc[:60])
    preds = model.forecast(6)
    assert len(preds) == 6
    assert np.all(np.isfinite(preds))
    assert model.order in [o for o, _ in ARIMA_GRID]


def test_xgb_fit_and_recursive_forecast(series):
    model = XGBModel().fit(series.iloc[:60])
    preds = model.forecast(6)
    assert len(preds) == 6
    assert np.all(np.isfinite(preds))


def test_xgb_time_index_continues_from_training_length():
    """With a truncated history (as in the API artifact) t must still be ~n_obs."""
    class Recorder:
        def __init__(self):
            self.ts = []

        def predict(self, X):
            self.ts.append(float(X["t"].iloc[0]))
            return [100.0]

    idx = pd.date_range("1959-01-01", periods=24, freq="MS")
    model = XGBModel.__new__(XGBModel)
    model.model_ = Recorder()
    model.history_ = pd.Series(np.arange(24, dtype=float), index=idx)
    model.t_offset_ = 144 - 24
    model.forecast(3)
    assert model.model_.ts == [144.0, 145.0, 146.0]


def test_mae_rmse_known_values():
    assert mae([1, 2, 3], [1, 2, 4]) == pytest.approx(1 / 3)
    assert rmse([1, 2, 3], [1, 2, 4]) == pytest.approx(np.sqrt(1 / 3))
    assert mae([5, 5], [5, 5]) == 0.0


def test_time_splits_are_forward_only():
    splits = list(time_splits(144, n_splits=3, horizon=12))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(test_idx) == 12
        assert max(train_idx) < min(test_idx)  # no leakage: test strictly after train
    # expanding window: later folds train on more data
    assert len(splits[0][0]) < len(splits[1][0]) < len(splits[2][0])
    # last test block is the final 12 months
    assert splits[2][1] == list(range(132, 144))


def test_evaluate_returns_fold_means(series):
    out = evaluate(NaiveSeasonal, series, n_splits=2, horizon=12)
    assert out["mae"] > 0 and out["rmse"] > 0
    assert len(out["mae_folds"]) == 2 and len(out["rmse_folds"]) == 2
    assert out["rmse"] >= out["mae"]


def _client():
    from src import api
    return TestClient(api.app)


def test_api_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_api_info():
    info = _client().get("/info").json()
    assert info["model"] in MODELS
    assert info["history_end"] == "1960-12-01T00:00:00"
    assert info["metrics"]["rmse"] > 0


def test_api_forecast_shape_and_dates():
    r = _client().post("/forecast", json={"periods": 6})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecasts"]) == 6
    assert body["forecasts"][0]["date"] == "1961-01"
    assert body["forecasts"][-1]["date"] == "1961-06"
    assert all(np.isfinite(f["value"]) for f in body["forecasts"])


def test_api_forecast_rejects_bad_periods():
    assert _client().post("/forecast", json={"periods": 0}).status_code == 422
    assert _client().post("/forecast", json={"periods": 37}).status_code == 422
