# Time-Series Forecasting System — Report

## Dataset
Monthly international airline passengers, Jan 1949 – Dec 1960 (144 months),
a public benchmark demand series (bundled at `data/airline-passengers.csv`,
source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv).
The series has a strong upward trend plus a yearly seasonal cycle (summer peaks).

## Models
1. **Naive seasonal baseline** — forecasts each future month as the same month
   one year earlier (the honest "do nothing smart" reference point).
2. **ARIMA (statistical)** — SARIMAX; the (p,d,q)×(P,D,Q,12) order is chosen per
   training fold by AIC from a small grid, so trend + yearly seasonality are
   modelled explicitly.
3. **XGBoost (machine learning)** — 400 trees on engineered features: lags 1–12,
   rolling means (3/6/12 months), month as sin/cos, and a linear time index.
   Multi-step forecasts are recursive (each predicted month feeds the next).

## Evaluation — time-based cross-validation, not random splits
Random splits would let the model train on the future and test on the past.
Instead: 3 expanding-window folds, each forecasting the next 12 months
(train →108 / test 108–120, train →120 / test 120–132, train →132 / test 132–144).
Metric reported is the mean over folds.

| Model | MAE | RMSE |
|---|---|---|
| Naive seasonal baseline | 35.92 | 38.99 |
| ARIMA (SARIMAX, AIC-selected) **(winner)** | 26.58 | 29.61 |
| XGBoost (lag + calendar features) | 32.93 | 38.54 |

![forecasts vs actual](results/forecast_plot.png)

## Per-fold results (RMSE)

| Model | Fold 1 (1958) | Fold 2 (1959) | Fold 3 (1960) | Mean |
|---|---|---|---|---|
| Naive seasonal baseline | 17.01 | 49.25 | 50.71 | 38.99 |
| ARIMA | 20.46 | 46.60 | 21.78 | **29.61** |
| XGBoost | 31.05 | 43.78 | 40.78 | 38.54 |

## Why ARIMA won
ARIMA has the lowest mean MAE (26.58) and RMSE (29.61), about 26% (MAE) and 24% (RMSE)
better than the naive baseline. Where the gap comes from:

- **It models trend and season explicitly.** The series is a steady upward trend
  with a stable yearly cycle. Differencing plus seasonal terms capture that with
  a handful of parameters. In folds 2 and 3 the naive baseline is off by ~47
  passengers every month (mean error +47, i.e. it under-predicts) because it
  repeats last year's level and ignores growth. ARIMA's fold 3 RMSE is 21.8
  against 50.7 for naive.
- **XGBoost is held back by the trend.** Tree models predict within the range of
  values seen in training, so on a rising series they tend to lag behind new
  highs, and recursive multi-step forecasts compound error. Its fold 3 RMSE
  (40.8) is nearly double ARIMA's. This is the likely cause, not something
  isolated with a separate experiment.

**Caveats (read before trusting the ranking):**
- The win is not uniform. In fold 2 (1959) XGBoost (RMSE 43.8) beat ARIMA (46.6),
  and in fold 1 (1958) the naive baseline beat both models (17.0 vs 20.5 and 31.1).
  ARIMA's advantage comes mostly from fold 3, so with only 3 folds of 12 months
  the ranking between ARIMA and XGBoost carries real uncertainty.
- The ARIMA order is re-selected by AIC inside every fold (no look-ahead), which
  is the fair way to do it but means the order can differ between folds.
- With 144 points and one series, differences of a few passengers are not
  meaningful; the clear conclusions are that both ARIMA and XGBoost are
  competitive with the baseline, and that ARIMA is the best of the three here.

## Serving
The winning ARIMA model, refit on all 144 months, is served by the FastAPI app
(`src/api.py`): `GET /health`, `GET /info`, `POST /forecast {"periods": 12}`.
Example 12-month forecast (1961): Jan 447 … Jul 649 (peak) … Dec 465 —
continuing the trend and seasonal shape of the 1950s.

## Reproduce
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.cli train     # CV, winner, artifact, results/
.venv/bin/python -m pytest tests/ -q  # 14 tests
.venv/bin/python -m src.cli serve     # API on http://127.0.0.1:8000
```
