import pandas as pd


def prepare_hourly_data(raw_df):
    """
    Create a complete hourly modeling frame from the raw hourly data.
    """
    # Reindex to the full hourly range so missing timestamps become rows.
    full_index = pd.date_range(
        raw_df["timestamp"].min(),
        raw_df["timestamp"].max(),
        freq="h"
    )
    hourly_df = raw_df.set_index("timestamp").reindex(full_index)
    hourly_df.index.name = "timestamp"

    hourly_df["date"] = hourly_df.index.normalize()

    # Calendar fields are rebuilt from the timestamp.
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
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2,
        7: 3, 8: 3, 9: 3,
        10: 4, 11: 4, 12: 4,
    }
    hourly_df["season"] = hourly_df["mnth"].map(month_to_season)

    # Interpolate continuous variables; keep weather situation as a category.
    for col in ["cnt", "temp", "atemp", "hum", "windspeed"]:
        hourly_df[col] = hourly_df[col].interpolate(method="time")

    hourly_df["cnt"] = hourly_df["cnt"].clip(lower=0)
    hourly_df["weathersit"] = hourly_df["weathersit"].ffill().bfill()

    # Keep only variables used later in the analysis.
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
    ]
    return hourly_df[keep_cols].copy()


def time_based_split(df, train_frac=0.6, valid_frac=0.2, gap_hours=168):
    """
    Split the hourly frame with a gap between periods.
    """
    n = len(df)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end + gap_hours:valid_end].copy(),
        df.iloc[valid_end + gap_hours:].copy(),
    )


def split_summary(train_df, valid_df, test_df):
    """
    Summarize the train, validation, and test splits.
    """
    starts = [train_df.index.min(), valid_df.index.min(), test_df.index.min()]
    ends = [train_df.index.max(), valid_df.index.max(), test_df.index.max()]
    gaps = [0]
    for start, previous_end in zip(starts[1:], ends[:-1]):
        gaps.append(int((start - previous_end) / pd.Timedelta(hours=1)) - 1)

    return pd.DataFrame(
        {
            "rows": [len(train_df), len(valid_df), len(test_df)],
            "start": starts,
            "end": ends,
            "gap_from_previous_hours": gaps,
        },
        index=["train", "validation", "test"],
    )
