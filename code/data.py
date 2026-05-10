from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

HOUR_DATA_PATH = DATA_DIR / "hour.csv"
DAY_DATA_PATH = DATA_DIR / "day.csv"
DATA_PATH = HOUR_DATA_PATH


# Load the raw hourly data
def load_raw_hourly(path=HOUR_DATA_PATH):
    """
    Load the raw hourly data and create a timestamp column.
    """
    raw_df = pd.read_csv(path, parse_dates=["dteday"])
    raw_df["timestamp"] = raw_df["dteday"] + pd.to_timedelta(raw_df["hr"], unit="h")
    return raw_df.sort_values("timestamp").reset_index(drop=True)

# Basic data overview
def data_overview(raw_df):
    """
    Provide a basic overview of the raw data, including missing timestamps.
    """
    full_index = pd.date_range(
        raw_df["timestamp"].min(), 
        raw_df["timestamp"].max(), 
        freq="h"
    )
    return pd.Series(
        {
            "rows": len(raw_df),
            "start_timestamp": raw_df["timestamp"].min(),
            "end_timestamp": raw_df["timestamp"].max(),
            "duplicate_timestamps": int(raw_df["timestamp"].duplicated().sum()),
            "expected_hours": len(full_index),
            "observed_hours": len(raw_df),
            "missing_hours": len(full_index) - len(raw_df),
        }
    )
