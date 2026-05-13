import json
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from analysis import count_predictions, model_metrics
from data import RESULTS_DIR


def build_lagged_regression_frame(model_df):
    """
    Create the log-target regression frame with lags and dummies.
    """
    reg_df = model_df.copy()
    reg_df["log_cnt"] = np.log1p(reg_df["cnt"])
    # These lags mirror the short-term, daily, and weekly structure from EDA.
    for lag in (1, 2, 3, 24, 168):
        reg_df[f"log_lag_{lag}"] = reg_df["log_cnt"].shift(lag)

    for col in ["season", "mnth", "hr", "weekday", "weathersit"]:
        reg_df[col] = reg_df[col].astype(int)

    reg_df = pd.get_dummies(
        reg_df,
        columns=["season", "mnth", "hr", "weekday", "weathersit"],
        drop_first=True,
    ).dropna()
    return reg_df.astype(float)


def run_lagged_regression(model_df, train_df, valid_df, test_df, use_cache=True):
    """
    Fit or load the log-target lagged regression model.
    """
    paths = {
        "model": RESULTS_DIR / "lagged_regression_model.pkl",
        "valid_pred": RESULTS_DIR / "lagged_regression_valid_pred.csv",
        "test_pred": RESULTS_DIR / "lagged_regression_test_pred.csv",
        "metrics": RESULTS_DIR / "lagged_regression_metrics.csv",
        "details": RESULTS_DIR / "lagged_regression_details.json",
    }

    if paths["model"].exists() and use_cache:
        with open(paths["model"], "rb") as f:
            model = pickle.load(f)
        details = json.loads(paths["details"].read_text())

        valid_cache = pd.read_csv(paths["valid_pred"], parse_dates=["timestamp"])
        test_cache = pd.read_csv(paths["test_pred"], parse_dates=["timestamp"])
        valid_pred = pd.Series(
            valid_cache["prediction"].to_numpy(),
            index=pd.DatetimeIndex(valid_cache["timestamp"]),
            name="prediction",
        )
        test_pred = pd.Series(
            test_cache["prediction"].to_numpy(),
            index=pd.DatetimeIndex(test_cache["timestamp"]),
            name="prediction",
        )
        return {
            "name": "Lagged Regression",
            "valid_pred": valid_pred,
            "test_pred": test_pred,
            "metrics": pd.read_csv(paths["metrics"]),
            "model": model,
            "details": details,
        }

    reg_df = build_lagged_regression_frame(model_df)
    reg_train = reg_df.loc[reg_df.index.intersection(train_df.index)].copy()
    reg_valid = reg_df.loc[reg_df.index.intersection(valid_df.index)].copy()
    reg_test = reg_df.loc[reg_df.index.intersection(test_df.index)].copy()
    feature_cols = [col for col in reg_df.columns if col not in ["cnt", "log_cnt"]]

    # Fit on log demand, then score after converting back to counts.
    model = LinearRegression()
    model.fit(reg_train[feature_cols], reg_train["log_cnt"])
    valid_log_pred = pd.Series(model.predict(reg_valid[feature_cols]), index=reg_valid.index)
    test_log_pred = pd.Series(model.predict(reg_test[feature_cols]), index=reg_test.index)
    valid_pred = count_predictions(np.expm1(valid_log_pred)).set_axis(reg_valid.index)
    test_pred = count_predictions(np.expm1(test_log_pred)).set_axis(reg_test.index)

    metrics = model_metrics("Lagged Regression", reg_valid["cnt"], valid_pred, reg_test["cnt"], test_pred)
    details = {"target": "log1p(cnt)", "feature_cols": feature_cols}

    with open(paths["model"], "wb") as f:
        pickle.dump(model, f)
    valid_pred.rename("prediction").to_csv(paths["valid_pred"], index_label="timestamp")
    test_pred.rename("prediction").to_csv(paths["test_pred"], index_label="timestamp")
    metrics.to_csv(paths["metrics"], index=False)
    paths["details"].write_text(json.dumps(details, indent=2))

    return {
        "name": "Lagged Regression",
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "metrics": metrics,
        "model": model,
        "details": details,
    }


def lagged_regression_coefficients(model_df, train_df):
    """
    Estimate coefficients and p-values for the lagged regression.
    """
    reg_df = build_lagged_regression_frame(model_df)
    reg_train = reg_df.loc[reg_df.index.intersection(train_df.index)].copy()
    feature_cols = [col for col in reg_df.columns if col not in ["cnt", "log_cnt"]]

    X = sm.add_constant(reg_train[feature_cols], has_constant="add")
    ols = sm.OLS(reg_train["log_cnt"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    coef_df = pd.DataFrame({
        "term": ols.params.index,
        "coef": ols.params.to_numpy(),
        "std_err": ols.bse.to_numpy(),
        "p_value": ols.pvalues.to_numpy(),
    })
    coef_df["sig"] = pd.cut(
        coef_df["p_value"],
        bins=[-np.inf, 0.001, 0.01, 0.05, 0.1, np.inf],
        labels=["***", "**", "*", ".", ""],
    ).astype(str)
    coef_df.to_csv(RESULTS_DIR / "lagged_regression_coefficients.csv", index=False)
    return coef_df


def run_poisson_lagged_regression(model_df, train_df, valid_df, test_df, use_cache=True):
    """
    Fit or load the Poisson lagged regression model.
    """
    paths = {
        "model": RESULTS_DIR / "poisson_lagged_regression_model.pkl",
        "valid_pred": RESULTS_DIR / "poisson_lagged_regression_valid_pred.csv",
        "test_pred": RESULTS_DIR / "poisson_lagged_regression_test_pred.csv",
        "metrics": RESULTS_DIR / "poisson_lagged_regression_metrics.csv",
        "details": RESULTS_DIR / "poisson_lagged_regression_details.json",
        "dispersion": RESULTS_DIR / "poisson_lagged_regression_dispersion.csv",
    }

    if paths["model"].exists() and use_cache:
        with open(paths["model"], "rb") as f:
            model = pickle.load(f)
        details = json.loads(paths["details"].read_text())

        valid_cache = pd.read_csv(paths["valid_pred"], parse_dates=["timestamp"])
        test_cache = pd.read_csv(paths["test_pred"], parse_dates=["timestamp"])
        valid_pred = pd.Series(
            valid_cache["prediction"].to_numpy(),
            index=pd.DatetimeIndex(valid_cache["timestamp"]),
            name="prediction",
        )
        test_pred = pd.Series(
            test_cache["prediction"].to_numpy(),
            index=pd.DatetimeIndex(test_cache["timestamp"]),
            name="prediction",
        )
        result = {
            "name": "Poisson Lagged Regression",
            "valid_pred": valid_pred,
            "test_pred": test_pred,
            "metrics": pd.read_csv(paths["metrics"]),
            "model": model,
            "details": details,
        }
        return result, pd.read_csv(paths["dispersion"])

    reg_df = build_lagged_regression_frame(model_df)
    reg_train = reg_df.loc[reg_df.index.intersection(train_df.index)].copy()
    reg_valid = reg_df.loc[reg_df.index.intersection(valid_df.index)].copy()
    reg_test = reg_df.loc[reg_df.index.intersection(test_df.index)].copy()
    feature_cols = [col for col in reg_df.columns if col not in ["cnt", "log_cnt"]]

    # Poisson GLM uses a log link and predicts the count mean directly.
    X_train = sm.add_constant(reg_train[feature_cols], has_constant="add")
    X_valid = sm.add_constant(reg_valid[feature_cols], has_constant="add")
    X_test = sm.add_constant(reg_test[feature_cols], has_constant="add")
    poisson = sm.GLM(reg_train["cnt"], X_train, family=sm.families.Poisson()).fit(maxiter=100)
    valid_pred = count_predictions(poisson.predict(X_valid)).set_axis(reg_valid.index)
    test_pred = count_predictions(poisson.predict(X_test)).set_axis(reg_test.index)

    metrics = model_metrics(
        "Poisson Lagged Regression",
        reg_valid["cnt"],
        valid_pred,
        reg_test["cnt"],
        test_pred,
    )
    dispersion = pd.DataFrame([{
        "pearson_chi2": poisson.pearson_chi2,
        "df_resid": poisson.df_resid,
        "dispersion": poisson.pearson_chi2 / poisson.df_resid,
    }])
    details = {"target": "cnt", "link": "log", "feature_cols": feature_cols}
    model_artifact = {
        "param_names": poisson.params.index.tolist(),
        "params": [float(param) for param in poisson.params],
        "family": "Poisson",
        "link": "log",
    }

    with open(paths["model"], "wb") as f:
        pickle.dump(model_artifact, f)
    valid_pred.rename("prediction").to_csv(paths["valid_pred"], index_label="timestamp")
    test_pred.rename("prediction").to_csv(paths["test_pred"], index_label="timestamp")
    metrics.to_csv(paths["metrics"], index=False)
    dispersion.to_csv(paths["dispersion"], index=False)
    paths["details"].write_text(json.dumps(details, indent=2))

    result = {
        "name": "Poisson Lagged Regression",
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "metrics": metrics,
        "model": model_artifact,
        "details": details,
    }
    return result, dispersion
