"""CLI: python -m src.cli train | forecast [--periods N] | serve"""

import argparse
import sys

from .data_loader import load_series
from .models import MODELS
from .train import main as train_main


def cmd_forecast(periods):
    import joblib
    import pandas as pd
    from .api import _forecast, ARTIFACT, LAST_DATE
    preds = _forecast(periods)
    dates = pd.date_range(LAST_DATE + pd.DateOffset(months=1), periods=periods, freq="MS")
    print(f"model: {ARTIFACT['model_name']}")
    for d, v in zip(dates, preds):
        print(f"{d.strftime('%Y-%m')}: {v:.1f}")


def cmd_serve():
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Time-series forecasting")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="run CV, pick winner, save artifact + results")
    p = sub.add_parser("forecast", help="print a forecast from the saved model")
    p.add_argument("--periods", type=int, default=12)
    sub.add_parser("serve", help="start the FastAPI server")
    args = parser.parse_args(argv)

    if args.cmd == "train":
        train_main()
    elif args.cmd == "forecast":
        cmd_forecast(args.periods)
    elif args.cmd == "serve":
        cmd_serve()


if __name__ == "__main__":
    main()
