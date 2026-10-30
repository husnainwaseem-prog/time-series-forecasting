"""Train + compare the three models, refit the winner, save the API artifact."""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data_loader import load_series
from .evaluate import evaluate
from .models import MODELS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def run_comparison(series, n_splits=3, horizon=12):
    results = {}
    for name, factory in MODELS.items():
        print(f"evaluating {name} ...", flush=True)
        results[name] = evaluate(factory, series, n_splits, horizon)
        print(f"  MAE={results[name]['mae']:.2f}  RMSE={results[name]['rmse']:.2f}",
              flush=True)
    return results


def pick_winner(results):
    return min(results, key=lambda k: results[k]["rmse"])


def save_artifact(model, model_name, series, metrics):
    os.makedirs(MODELS_DIR, exist_ok=True)
    fitted = getattr(model, "result_", None) or getattr(model, "model_", None)
    artifact = {
        "model_name": model_name,
        "fitted": fitted,                       # ARIMA results / XGB regressor
        "last_values": list(series.iloc[-24:]),  # history for naive / recursive
        "last_date": series.index[-1].isoformat(),
        "metrics": metrics,
    }
    path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(artifact, path)
    return path


def save_results_table(results, winner, path):
    lines = ["| Model | MAE | RMSE |",
             "|---|---|---|"]
    for name, m in results.items():
        mark = " **(winner)**" if name == winner else ""
        lines.append(f"| {MODELS[name].label}{mark} | {m['mae']:.2f} | {m['rmse']:.2f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_plot(series, path, train_end=132, horizon=12):
    """Actual vs each model's 12-month forecast (models fit on first 132 months)."""
    train, actual = series.iloc[:train_end], series.iloc[train_end:train_end + horizon]
    plt.figure(figsize=(10, 5))
    plt.plot(series.index[train_end - 36:train_end + horizon], series.iloc[train_end - 36:train_end + horizon],
             label="actual", color="black", linewidth=2)
    for name, factory in MODELS.items():
        preds = factory().fit(train).forecast(horizon)
        idx = pd.date_range(actual.index[0], periods=horizon, freq="MS")
        plt.plot(idx, preds, label=factory.label, linestyle="--")
    plt.axvline(train.index[-1], color="grey", linestyle=":")
    plt.title("12-month forecasts vs actual (last CV fold)")
    plt.ylabel("passengers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    series = load_series()
    print(f"loaded {len(series)} months: {series.index[0].date()} -> {series.index[-1].date()}")

    results = run_comparison(series)
    winner = pick_winner(results)
    print(f"\nwinner (lowest mean RMSE): {winner}")

    model = MODELS[winner]().fit(series)
    artifact_path = save_artifact(model, winner, series, results[winner])
    print(f"saved API artifact: {artifact_path}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({"results": results, "winner": winner}, f, indent=2)
    save_results_table(results, winner, os.path.join(RESULTS_DIR, "results_table.md"))
    save_plot(series, os.path.join(RESULTS_DIR, "forecast_plot.png"))
    print(f"saved results table + plot in {RESULTS_DIR}/")

    print("\n12-month forecast from the winning model:")
    future_idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
    for d, v in zip(future_idx, model.forecast(12)):
        print(f"  {d.strftime('%Y-%m')}: {v:.0f}")

    return winner, results


if __name__ == "__main__":
    main()
