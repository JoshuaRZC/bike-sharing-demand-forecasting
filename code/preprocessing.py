import numpy as np
import pandas as pd


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


def chronological_split(df, train_frac=0.6, valid_frac=0.2):
    n = len(df)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:valid_end].copy(),
        df.iloc[valid_end:].copy(),
    )


def split_summary(train_df, valid_df, test_df):
    return pd.DataFrame(
        {
            "rows": [len(train_df), len(valid_df), len(test_df)],
            "start": [train_df.index.min(), valid_df.index.min(), test_df.index.min()],
            "end": [train_df.index.max(), valid_df.index.max(), test_df.index.max()],
        },
        index=["train", "validation", "test"],
    )


def add_lag_features(df, target="cnt", lags=(1, 2, 3, 24, 168)):
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)
    return out.dropna().copy()


def make_regression_frame(df):
    return pd.get_dummies(
        df,
        columns=["season", "mnth", "hr", "weekday", "weathersit"],
        drop_first=True,
    )


def make_sequence_arrays(df, feature_cols, target_col="cnt", window=24):
    X_list, y_list, index_list = [], [], []
    values = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[target_col].to_numpy(dtype=np.float32)

    for i in range(window, len(df)):
        X_list.append(values[i - window : i])
        y_list.append(target[i])
        index_list.append(df.index[i])

    return np.stack(X_list), np.array(y_list, dtype=np.float32), np.array(index_list)
