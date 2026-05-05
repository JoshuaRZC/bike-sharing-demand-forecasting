import pandas as pd
from statsmodels.tsa.stattools import adfuller


def lag_autocorr(series, lags):
    return pd.Series({f"lag_{lag}": series.autocorr(lag=lag) for lag in lags})


def adf_summary(series):
    stat, pvalue, used_lag, nobs, critical_values, _ = adfuller(series.dropna())
    return pd.Series(
        {
            "adf_statistic": stat,
            "p_value": pvalue,
            "used_lag": used_lag,
            "n_obs": nobs,
            **{f"critical_{k}": v for k, v in critical_values.items()},
        }
    )


def residual_summary(residuals):
    residuals = pd.Series(residuals).dropna()
    return pd.Series(
        {
            "mean": residuals.mean(),
            "std": residuals.std(),
            "min": residuals.min(),
            "max": residuals.max(),
        }
    )
