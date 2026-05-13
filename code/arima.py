import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analysis import count_predictions, metric_dict, model_metrics
from data import FIGURES_DIR, RESULTS_DIR


PRIMARY = "#234E70"
ACCENT = "#D97B66"


def plot_arima_order_search(order_search):
    """
    Plot validation RMSE against ARIMA order complexity.
    """
    plot_df = order_search.copy()
    parts = plot_df["order"].str.strip("()").str.split(",", expand=True).astype(int)
    plot_df[["p", "d", "q"]] = parts
    plot_df["complexity"] = plot_df["p"] + plot_df["q"]
    plot_df["label"] = plot_df.apply(lambda row: f"({int(row['p'])},{int(row['d'])},{int(row['q'])})", axis=1)
    plot_df["x_plot"] = plot_df["complexity"].astype(float)
    for complexity, group in plot_df.groupby("complexity"):
        offsets = np.linspace(-0.07, 0.07, len(group)) if len(group) > 1 else np.array([0.0])
        ordered_index = group.sort_values(["RMSE", "p", "q"]).index
        plot_df.loc[ordered_index, "x_plot"] = complexity + offsets

    best_idx = plot_df["RMSE"].idxmin()
    best = plot_df.loc[best_idx]
    q1, q3 = plot_df["RMSE"].quantile([0.25, 0.75])
    threshold = q3 + 1.5 * (q3 - q1)
    outliers = plot_df[plot_df["RMSE"] > threshold].copy()
    main_df = plot_df[plot_df["RMSE"] <= threshold].copy()
    if outliers.empty:
        outliers = plot_df.loc[[plot_df["RMSE"].idxmax()]].copy()
        main_df = plot_df.drop(outliers.index).copy()

    fig, (top_ax, ax) = plt.subplots(
        2,
        1,
        figsize=(9.4, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.07},
    )

    for current_ax, current_df in [(top_ax, outliers), (ax, main_df)]:
        current_ax.scatter(
            current_df["x_plot"],
            current_df["RMSE"],
            s=82,
            color=PRIMARY,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.9,
            zorder=3,
        )

    ax.scatter(
        best["x_plot"],
        best["RMSE"],
        s=82,
        color=ACCENT,
        edgecolor="white",
        linewidth=1.4,
        zorder=4,
    )

    # Extreme fits are shown separately so the main comparison stays readable.
    for _, row in outliers.iterrows():
        top_ax.annotate(
            row["label"],
            (row["x_plot"], row["RMSE"]),
            xytext=(14, -2),
            textcoords="offset points",
            fontsize=8.5,
            color="#263238",
            va="center",
        )
    label_df = main_df.sort_values(["complexity", "RMSE"]).copy()
    label_df["label_rank"] = label_df.groupby("complexity").cumcount()
    offsets = [8, 20, 32, 44]
    label_offsets = {
        "(2,0,1)": (42, -24),
        "(3,0,0)": (-38, 28),
    }
    leader_labels = {"(2,0,1)", "(3,0,0)"}
    for _, row in label_df.iterrows():
        xytext = label_offsets.get(row["label"], (7, offsets[int(row["label_rank"]) % len(offsets)]))
        arrowprops = None
        if row["label"] in leader_labels:
            arrowprops = {"arrowstyle": "-", "color": "#5F6B73", "lw": 0.8, "shrinkA": 2, "shrinkB": 4}
        ax.annotate(
            row["label"],
            (row["x_plot"], row["RMSE"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8.5,
            color="#263238",
            arrowprops=arrowprops,
        )

    top_ax.set_title("ARIMA Order Search", loc="left")
    ax.set_xlabel("# ARMA parameters (p + q)")
    ax.set_xticks(sorted(plot_df["complexity"].unique()))
    ax.set_xlim(plot_df["complexity"].min() - 0.35, plot_df["complexity"].max() + 0.35)
    ax.set_ylim(main_df["RMSE"].min() - 0.08, main_df["RMSE"].max() + 0.25)
    top_pad = max((outliers["RMSE"].max() - outliers["RMSE"].min()) * 0.12, outliers["RMSE"].max() * 0.02)
    top_ax.set_ylim(outliers["RMSE"].min() - top_pad, outliers["RMSE"].max() + top_pad)

    top_ax.spines.bottom.set_visible(False)
    ax.spines.top.set_visible(False)
    top_ax.tick_params(labelbottom=False, bottom=False)
    top_ax.grid(True, axis="y", alpha=0.25)
    top_ax.grid(True, axis="x", alpha=0.12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", alpha=0.12)

    break_marks = {"marker": [(-1, -0.5), (1, 0.5)], "markersize": 9, "linestyle": "none", "color": "#666666", "mec": "#666666", "mew": 1}
    top_ax.plot([0, 1], [0, 0], transform=top_ax.transAxes, clip_on=False, **break_marks)
    ax.plot([0, 1], [1, 1], transform=ax.transAxes, clip_on=False, **break_marks)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PRIMARY,
               markeredgecolor="white", markersize=8, label="Candidate order"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=ACCENT,
               markeredgecolor="white", markersize=8, label="Selected order"),
    ]
    fig.text(0.04, 0.5, "Validation RMSE", va="center", rotation="vertical")
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.88))

    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.13)
    fig.savefig(FIGURES_DIR / "model_arima_order_search.png", bbox_inches="tight")
    plt.show()


def run_arima(
    model_df,
    train_df,
    valid_df,
    test_df,
    seasonal_lag=24,
    orders=(
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (1, 0, 1),
        (2, 0, 1),
        (3, 0, 1),
        (1, 0, 2),
        (2, 0, 2),
        (3, 0, 2),
    ),
    use_cache=True,
    search_maxiter=500,
    final_maxiter=800,
):
    """
    Fit or load the 24-hour differenced ARIMA benchmark.
    """
    paths = {
        "model": RESULTS_DIR / "arima_model.pkl",
        "valid_pred": RESULTS_DIR / "arima_valid_pred.csv",
        "test_pred": RESULTS_DIR / "arima_test_pred.csv",
        "metrics": RESULTS_DIR / "arima_metrics.csv",
        "details": RESULTS_DIR / "arima_details.json",
        "order_search": RESULTS_DIR / "arima_order_search.csv",
    }

    if paths["model"].exists() and use_cache:
        with open(paths["model"], "rb") as f:
            model = pickle.load(f)
        details = json.loads(paths["details"].read_text())
        details["best_order"] = tuple(details["best_order"])

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
            "name": "ARIMA",
            "valid_pred": valid_pred,
            "test_pred": test_pred,
            "metrics": pd.read_csv(paths["metrics"]),
            "model": model,
            "details": details,
        }
        return result, pd.read_csv(paths["order_search"])

    # Work with the 24-hour difference, then add the lagged level back.
    full_cnt = model_df["cnt"]
    z_full = full_cnt.diff(seasonal_lag)
    lagged_level = full_cnt.shift(seasonal_lag)
    z_train = z_full.loc[train_df.index].dropna()
    valid_context = model_df.loc[(model_df.index > train_df.index.max()) & (model_df.index <= valid_df.index.max())]
    z_valid_context = z_full.loc[valid_context.index].dropna()

    rows = []
    for order in orders:
        fitted = ARIMA(
            z_train.to_numpy(),
            order=order,
        ).fit(method_kwargs={"maxiter": search_maxiter})
        valid_state = fitted.extend(z_valid_context.to_numpy())
        valid_fitted = pd.Series(valid_state.fittedvalues, index=z_valid_context.index)
        valid_pred = valid_fitted.loc[valid_df.index] + lagged_level.loc[valid_df.index]
        rows.append({
            "order": str(order),
            "AIC": fitted.aic,
            "BIC": fitted.bic,
            **metric_dict(valid_df["cnt"], count_predictions(valid_pred)),
        })

    order_search = pd.DataFrame(rows).sort_values(["RMSE", "AIC"]).reset_index(drop=True)
    best_order = tuple(int(x.strip()) for x in order_search.loc[0, "order"].strip("()").split(","))

    valid_fit = ARIMA(
        z_train.to_numpy(),
        order=best_order,
    ).fit(method_kwargs={"maxiter": final_maxiter})
    valid_state = valid_fit.extend(z_valid_context.to_numpy())
    valid_fitted = pd.Series(valid_state.fittedvalues, index=z_valid_context.index)
    valid_pred = count_predictions(
        valid_fitted.loc[valid_df.index] + lagged_level.loc[valid_df.index]
    ).set_axis(valid_df.index)

    train_valid_index = train_df.index.append(valid_df.index)
    z_train_valid = z_full.loc[train_valid_index].dropna()
    test_context = model_df.loc[(model_df.index > valid_df.index.max()) & (model_df.index <= test_df.index.max())]
    z_test_context = z_full.loc[test_context.index].dropna()
    # Refit on train plus validation before the final test forecast.
    final_fit = ARIMA(
        z_train_valid.to_numpy(),
        order=best_order,
    ).fit(method_kwargs={"maxiter": final_maxiter})
    test_state = final_fit.extend(z_test_context.to_numpy())
    test_fitted = pd.Series(test_state.fittedvalues, index=z_test_context.index)
    test_pred = count_predictions(
        test_fitted.loc[test_df.index] + lagged_level.loc[test_df.index]
    ).set_axis(test_df.index)

    metrics = model_metrics("ARIMA", valid_df["cnt"], valid_pred, test_df["cnt"], test_pred)
    details = {
        "best_order": best_order,
        "seasonal_lag": seasonal_lag,
        "search_maxiter": search_maxiter,
        "final_maxiter": final_maxiter,
    }

    with open(paths["model"], "wb") as f:
        pickle.dump(final_fit, f)
    valid_pred.rename("prediction").to_csv(paths["valid_pred"], index_label="timestamp")
    test_pred.rename("prediction").to_csv(paths["test_pred"], index_label="timestamp")
    metrics.to_csv(paths["metrics"], index=False)
    order_search.to_csv(paths["order_search"], index=False)
    paths["details"].write_text(
        json.dumps({
            "best_order": list(best_order),
            "seasonal_lag": seasonal_lag,
            "search_maxiter": search_maxiter,
            "final_maxiter": final_maxiter,
        }, indent=2)
    )

    result = {
        "name": "ARIMA",
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "metrics": metrics,
        "model": final_fit,
        "details": details,
    }
    return result, order_search


def run_sarimax(
    model_df,
    train_df,
    valid_df,
    test_df,
    orders=((1, 0, 1), (2, 0, 1), (2, 0, 2)),
    seasonal_orders=((1, 1, 0, 24),),
    use_cache=True,
    search_maxiter=100,
    final_maxiter=150,
):
    """
    Fit or load the SARIMAX benchmark with calendar and weather covariates.
    """
    paths = {
        "model": RESULTS_DIR / "sarimax_model.pkl",
        "valid_pred": RESULTS_DIR / "sarimax_valid_pred.csv",
        "test_pred": RESULTS_DIR / "sarimax_test_pred.csv",
        "metrics": RESULTS_DIR / "sarimax_metrics.csv",
        "details": RESULTS_DIR / "sarimax_details.json",
        "order_search": RESULTS_DIR / "sarimax_order_search.csv",
    }

    if paths["model"].exists() and use_cache:
        with open(paths["model"], "rb") as f:
            model = pickle.load(f)
        details = json.loads(paths["details"].read_text())
        details["order"] = tuple(details["order"])
        details["seasonal_order"] = tuple(details["seasonal_order"])

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
            "name": "SARIMAX",
            "valid_pred": valid_pred,
            "test_pred": test_pred,
            "metrics": pd.read_csv(paths["metrics"]),
            "model": model,
            "details": details,
        }, pd.read_csv(paths["order_search"])

    sarimax_df = model_df[["cnt", "yr", "workingday", "holiday", "temp", "hum", "windspeed", "weathersit", "hr", "weekday", "mnth"]].copy()
    # Cyclic encodings keep calendar variables close at their wraparound points.
    sarimax_df["hr_sin"] = np.sin(2 * np.pi * sarimax_df["hr"] / 24)
    sarimax_df["hr_cos"] = np.cos(2 * np.pi * sarimax_df["hr"] / 24)
    sarimax_df["week_sin"] = np.sin(2 * np.pi * sarimax_df["weekday"] / 7)
    sarimax_df["week_cos"] = np.cos(2 * np.pi * sarimax_df["weekday"] / 7)
    sarimax_df["month_sin"] = np.sin(2 * np.pi * sarimax_df["mnth"] / 12)
    sarimax_df["month_cos"] = np.cos(2 * np.pi * sarimax_df["mnth"] / 12)
    sarimax_df = sarimax_df.drop(columns=["hr", "weekday", "mnth"]).astype(float)

    sarimax_train = sarimax_df.loc[sarimax_df.index.intersection(train_df.index)].copy()
    sarimax_valid = sarimax_df.loc[sarimax_df.index.intersection(valid_df.index)].copy()
    sarimax_test = sarimax_df.loc[sarimax_df.index.intersection(test_df.index)].copy()
    exog_cols = [col for col in sarimax_df.columns if col != "cnt"]

    valid_context = sarimax_df.loc[
        (sarimax_df.index > train_df.index.max()) & (sarimax_df.index <= valid_df.index.max())
    ]
    y_mean = sarimax_train["cnt"].mean()
    y_std = sarimax_train["cnt"].std()
    exog_means = sarimax_train[exog_cols].mean()
    exog_stds = sarimax_train[exog_cols].std().replace(0, 1)
    train_y = ((sarimax_train["cnt"] - y_mean) / y_std).to_numpy()
    train_exog = ((sarimax_train[exog_cols] - exog_means) / exog_stds).to_numpy()
    valid_y = ((valid_context["cnt"] - y_mean) / y_std).to_numpy()
    valid_exog = ((valid_context[exog_cols] - exog_means) / exog_stds).to_numpy()

    rows = []
    for order in orders:
        for seasonal_order in seasonal_orders:
            fitted = SARIMAX(
                train_y,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=search_maxiter)
            valid_state = fitted.extend(valid_y, exog=valid_exog)
            valid_fitted = pd.Series(valid_state.fittedvalues * y_std + y_mean, index=valid_context.index)
            valid_pred = count_predictions(valid_fitted.loc[sarimax_valid.index]).set_axis(sarimax_valid.index)
            rows.append({
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "AIC": fitted.aic,
                "BIC": fitted.bic,
                **metric_dict(sarimax_valid["cnt"], valid_pred),
            })

    order_search = pd.DataFrame(rows).sort_values(["RMSE", "AIC"]).reset_index(drop=True)
    best_order = tuple(int(x.strip()) for x in order_search.loc[0, "order"].strip("()").split(","))
    best_seasonal_order = tuple(int(x.strip()) for x in order_search.loc[0, "seasonal_order"].strip("()").split(","))

    valid_fit = SARIMAX(
        train_y,
        exog=train_exog,
        order=best_order,
        seasonal_order=best_seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=final_maxiter)
    valid_state = valid_fit.extend(valid_y, exog=valid_exog)
    valid_fitted = pd.Series(valid_state.fittedvalues * y_std + y_mean, index=valid_context.index)
    valid_pred = count_predictions(valid_fitted.loc[sarimax_valid.index]).set_axis(sarimax_valid.index)

    train_valid_index = train_df.index.append(valid_df.index)
    sarimax_train_valid = sarimax_df.loc[train_valid_index].copy()
    test_context = sarimax_df.loc[
        (sarimax_df.index > valid_df.index.max()) & (sarimax_df.index <= test_df.index.max())
    ]
    final_y_mean = sarimax_train_valid["cnt"].mean()
    final_y_std = sarimax_train_valid["cnt"].std()
    final_exog_means = sarimax_train_valid[exog_cols].mean()
    final_exog_stds = sarimax_train_valid[exog_cols].std().replace(0, 1)
    final_y = ((sarimax_train_valid["cnt"] - final_y_mean) / final_y_std).to_numpy()
    final_exog = ((sarimax_train_valid[exog_cols] - final_exog_means) / final_exog_stds).to_numpy()
    test_y = ((test_context["cnt"] - final_y_mean) / final_y_std).to_numpy()
    test_exog = ((test_context[exog_cols] - final_exog_means) / final_exog_stds).to_numpy()
    # Refit on train plus validation before the final test forecast.
    final_fit = SARIMAX(
        final_y,
        exog=final_exog,
        order=best_order,
        seasonal_order=best_seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=final_maxiter)
    test_state = final_fit.extend(test_y, exog=test_exog)
    test_fitted = pd.Series(test_state.fittedvalues * final_y_std + final_y_mean, index=test_context.index)
    test_pred = count_predictions(
        test_fitted.loc[sarimax_test.index]
    ).set_axis(sarimax_test.index)

    metrics = model_metrics("SARIMAX", sarimax_valid["cnt"], valid_pred, sarimax_test["cnt"], test_pred)
    details = {
        "order": best_order,
        "seasonal_order": best_seasonal_order,
        "exog_cols": exog_cols,
        "scaling": "standardized y and exog",
        "search_maxiter": search_maxiter,
        "final_maxiter": final_maxiter,
    }

    model_artifact = {
        "param_names": final_fit.param_names,
        "params": [float(param) for param in final_fit.params],
        "order": best_order,
        "seasonal_order": best_seasonal_order,
        "exog_cols": exog_cols,
    }
    with open(paths["model"], "wb") as f:
        pickle.dump(model_artifact, f)
    valid_pred.rename("prediction").to_csv(paths["valid_pred"], index_label="timestamp")
    test_pred.rename("prediction").to_csv(paths["test_pred"], index_label="timestamp")
    metrics.to_csv(paths["metrics"], index=False)
    order_search.to_csv(paths["order_search"], index=False)
    paths["details"].write_text(
        json.dumps({
            "order": list(best_order),
            "seasonal_order": list(best_seasonal_order),
            "exog_cols": exog_cols,
            "scaling": "standardized y and exog",
            "search_maxiter": search_maxiter,
            "final_maxiter": final_maxiter,
        }, indent=2)
    )

    return {
        "name": "SARIMAX",
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "metrics": metrics,
        "model": final_fit,
        "details": details,
    }, order_search
