# Time-Series Forecasting System

Forecast monthly demand (public airline-passengers dataset, 1949–1960) with three
models, compare them honestly, and serve the winner through an API.

## Portal requirements checklist

| Requirement | Where |
|---|---|
| Public time-series dataset (demand/sales) | `data/airline-passengers.csv` (monthly passengers, 144 months) |
| Naive baseline | `src/models.py` → `NaiveSeasonal` (seasonal naive, 12-month) |
| Statistical model (ARIMA/Prophet) | `src/models.py` → `ArimaModel` (SARIMAX, AIC order selection) |
| ML model (XGBoost/LightGBM or similar) with engineered lag + calendar features | `src/models.py` → `XGBModel`; features in `src/features.py` (lags 1–12, rolling means, month sin/cos, trend) |
| Time-based cross-validation, not random splits | `src/evaluate.py` → `time_splits` (3 expanding-window folds, 12-month horizon) |
| MAE and RMSE per model + winner explained | `results/results_table.md`, `results/metrics.json`, `REPORT.md` |
| Best model served through an API | `src/api.py` (FastAPI: `/health`, `/info`, `POST /forecast`) |
| Code, report, README | this repo, `REPORT.md`, this file |

## Results (mean over 3 time-based folds)

| Model | MAE | RMSE |
|---|---|---|
| Naive seasonal baseline | 35.92 | 38.99 |
| **ARIMA (winner)** | **26.58** | **29.61** |
| XGBoost | 32.93 | 38.54 |

ARIMA wins on both metrics: the series is trend + stable yearly seasonality,
which SARIMAX captures with a few parameters, while XGBoost struggles to follow
the rising trend and its recursive forecasts accumulate error. The margin is not
uniform across folds (XGBoost beats ARIMA in fold 2), so treat the ranking
between those two with some caution. Per-fold numbers and caveats are in `REPORT.md`.

## Quickstart

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Windows cmd:        .venv\Scripts\activate
.venv/bin/pip install -r requirements.txt   # Linux/macOS
.venv/bin/python -m src.cli train           # compare models, save winner + results/
.venv/bin/python -m pytest tests/ -q        # 14 tests
.venv/bin/python -m src.cli forecast --periods 12
.venv/bin/python -m src.cli serve           # API → http://127.0.0.1:8000
```

API example:

```bash
curl -X POST http://127.0.0.1:8000/forecast -H "Content-Type: application/json" -d '{"periods": 6}'
```

## Layout

```
src/            data_loader, features, models, evaluate, train, api, cli
data/           airline-passengers.csv (bundled public dataset)
models/         best_model.joblib (winning ARIMA, refit on all data)
results/        metrics.json, results_table.md, forecast_plot.png
tests/          14 pytest tests
```
