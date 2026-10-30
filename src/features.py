"""Lag + calendar features for the XGBoost model.

One row per month: 12 lags, 3 rolling means, month as sin/cos, and a linear
time index (trend). feature_row() is the single source of truth — make_frame()
just calls it in a loop, and recursive forecasting reuses it too.
"""

import numpy as np
import pandas as pd

N_LAGS = 12
FEATURE_COLS = ([f"lag_{i}" for i in range(1, N_LAGS + 1)]
                + ["roll_mean_3", "roll_mean_6", "roll_mean_12",
                   "month_sin", "month_cos", "t"])


def feature_row(past_values, date, t):
    """past_values: >=12 values, most recent last. date: target month. t: time index."""
    v = np.asarray(past_values, dtype=float)
    if len(v) < N_LAGS:
        raise ValueError(f"need at least {N_LAGS} past values")
    month = pd.Timestamp(date).month
    row = {f"lag_{i}": v[-i] for i in range(1, N_LAGS + 1)}
    row.update({
        "roll_mean_3": v[-3:].mean(),
        "roll_mean_6": v[-6:].mean(),
        "roll_mean_12": v[-12:].mean(),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "t": t,
    })
    return row


def make_frame(series):
    """Build the (X, y) training frame; first 12 months have no full lags."""
    rows, targets = [], []
    values = series.to_numpy()
    for i in range(N_LAGS, len(series)):
        rows.append(feature_row(values[i - N_LAGS:i], series.index[i], i))
        targets.append(values[i])
    X = pd.DataFrame(rows, index=series.index[N_LAGS:], columns=FEATURE_COLS)
    y = pd.Series(targets, index=series.index[N_LAGS:], name=series.name)
    return X, y
