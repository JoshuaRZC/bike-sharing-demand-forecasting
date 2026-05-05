import numpy as np
import pandas as pd


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
