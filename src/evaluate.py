"""Metrics and time-based cross-validation (expanding window, never shuffled)."""

import numpy as np


def mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def time_splits(n, n_splits=3, horizon=12):
    """Yield (train_index, test_index); test blocks march forward in time."""
    if n_splits * horizon >= n:
        raise ValueError("not enough data for these splits")
    first_test_start = n - n_splits * horizon
    for k in range(n_splits):
        test_start = first_test_start + k * horizon
        yield list(range(test_start)), list(range(test_start, test_start + horizon))


def evaluate(model_factory, series, n_splits=3, horizon=12):
    """Mean MAE/RMSE over the expanding-window folds."""
    maes, rmses = [], []
    for train_idx, test_idx in time_splits(len(series), n_splits, horizon):
        train = series.iloc[train_idx]
        test = series.iloc[test_idx]
        model = model_factory().fit(train)
        preds = model.forecast(len(test_idx))
        maes.append(mae(test, preds))
        rmses.append(rmse(test, preds))
    return {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
            "mae_folds": maes, "rmse_folds": rmses}
