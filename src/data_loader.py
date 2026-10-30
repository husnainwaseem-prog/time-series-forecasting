"""Load the public monthly airline-passengers dataset (1949-1960).

Source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
A local copy is bundled at data/airline-passengers.csv so runs are reproducible.
"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "airline-passengers.csv")
DATASET_URL = ("https://raw.githubusercontent.com/jbrownlee/"
               "Datasets/master/airline-passengers.csv")


def load_series(path=DATA_PATH):
    """Return the monthly passenger counts as a sorted, gap-free Series."""
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
    series = (df.set_index("Month")["Passengers"].astype(float)
                .sort_index().asfreq("MS"))
    if series.isna().any():
        raise ValueError("series has missing months")
    series.name = "passengers"
    return series
