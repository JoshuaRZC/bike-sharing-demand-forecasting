import pandas as pd


# Basic summary statistics and correlations
def basic_summary(model_df):
    workingday_summary = pd.DataFrame(
        {
            "mean_cnt": model_df.groupby("workingday")["cnt"].mean(),
            "median_cnt": model_df.groupby("workingday")["cnt"].median(),
        }
    )
    weather_corr = model_df[["cnt", "temp", "hum", "windspeed"]].corr()["cnt"]
    return workingday_summary, weather_corr.sort_values(ascending=False)


# Lag autocorrelations for the target variable
def lag_autocorr(series, lags):
    return pd.Series({f"lag_{lag}": series.autocorr(lag=lag) for lag in lags})
