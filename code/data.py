from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

HOUR_DATA_PATH = DATA_DIR / "hour.csv"
DAY_DATA_PATH = DATA_DIR / "day.csv"
DATA_PATH = HOUR_DATA_PATH


def load_raw_hourly(path=HOUR_DATA_PATH):
    raw_df = pd.read_csv(path, parse_dates=["dteday"])
    raw_df["timestamp"] = raw_df["dteday"] + pd.to_timedelta(raw_df["hr"], unit="h")
    return raw_df.sort_values("timestamp").reset_index(drop=True)
