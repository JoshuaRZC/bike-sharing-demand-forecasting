import pandas as pd


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


def basic_summary(model_df):
    workingday_summary = pd.DataFrame(
        {
            "mean_cnt": model_df.groupby("workingday")["cnt"].mean(),
            "median_cnt": model_df.groupby("workingday")["cnt"].median(),
        }
    )
    weather_corr = model_df[["cnt", "temp", "hum", "windspeed"]].corr()["cnt"]
    return workingday_summary, weather_corr.sort_values(ascending=False)


def lag_autocorr(series, lags):
    return pd.Series({f"lag_{lag}": series.autocorr(lag=lag) for lag in lags})
