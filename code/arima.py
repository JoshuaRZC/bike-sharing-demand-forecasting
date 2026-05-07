import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from analysis import metric_dict


def fit_arima_candidates(train_df, valid_df, seasonal_lag=24, orders=None):
    if orders is None:
        orders = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1)]

    z_train = train_df["cnt"].diff(seasonal_lag).dropna()
    combined = pd.concat([train_df["cnt"], valid_df["cnt"]])
    z_valid = combined.diff(seasonal_lag).loc[valid_df.index]

    rows = []
    for order in orders:
        fitted = ARIMA(z_train, order=order).fit()
        state = fitted.extend(z_valid)
        pred = state.fittedvalues + combined.shift(seasonal_lag).loc[valid_df.index]
        rows.append(
            {
                "order": order,
                "AIC": fitted.aic,
                "BIC": fitted.bic,
                **metric_dict(valid_df["cnt"], pred),
            }
        )

    return pd.DataFrame(rows).sort_values(["RMSE", "AIC"]).reset_index(drop=True)
