import numpy as np
import pandas as pd

from paths import HOUR_DATA_PATH


def load_raw_hourly(path=HOUR_DATA_PATH):
    raw_df = pd.read_csv(path, parse_dates=["dteday"])
    raw_df["timestamp"] = raw_df["dteday"] + pd.to_timedelta(raw_df["hr"], unit="h")
    return raw_df.sort_values("timestamp").reset_index(drop=True)


def data_overview(raw_df):
    full_index = pd.date_range(raw_df["timestamp"].min(), raw_df["timestamp"].max(), freq="h")
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


def prepare_hourly_data(raw_df):
    full_index = pd.date_range(raw_df["timestamp"].min(), raw_df["timestamp"].max(), freq="h")
    hourly_df = raw_df.set_index("timestamp").reindex(full_index)
    hourly_df.index.name = "timestamp"

    hourly_df["is_imputed"] = hourly_df["cnt"].isna().astype(int)
    hourly_df["date"] = hourly_df.index.normalize()

    holiday_map = raw_df.groupby(raw_df["timestamp"].dt.normalize())["holiday"].first()
    hourly_df["holiday"] = hourly_df["date"].map(holiday_map).fillna(0).astype(int)

    hourly_df["yr"] = hourly_df.index.year - 2011
    hourly_df["mnth"] = hourly_df.index.month
    hourly_df["hr"] = hourly_df.index.hour
    hourly_df["weekday"] = (hourly_df.index.dayofweek + 1) % 7
    hourly_df["workingday"] = (
        hourly_df["weekday"].isin([1, 2, 3, 4, 5]) & (hourly_df["holiday"] == 0)
    ).astype(int)

    month_to_season = {
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 3,
        10: 4,
        11: 4,
        12: 4,
    }
    hourly_df["season"] = hourly_df["mnth"].map(month_to_season)

    for col in ["cnt", "temp", "atemp", "hum", "windspeed"]:
        hourly_df[col] = hourly_df[col].interpolate(method="time")

    hourly_df["cnt"] = hourly_df["cnt"].clip(lower=0)
    hourly_df["weathersit"] = hourly_df["weathersit"].ffill().bfill()

    keep_cols = [
        "cnt",
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "hum",
        "windspeed",
        "is_imputed",
    ]
    return hourly_df[keep_cols].copy()


def basic_summary(model_df):
    workingday_summary = pd.DataFrame(
        {
            "mean_cnt": model_df.groupby("workingday")["cnt"].mean(),
            "median_cnt": model_df.groupby("workingday")["cnt"].median(),
        }
    )
    weather_corr = model_df[["cnt", "temp", "hum", "windspeed"]].corr()["cnt"]
    return workingday_summary, weather_corr.sort_values(ascending=False)
