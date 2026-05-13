import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf

from analysis import metric_dict
from data import FIGURES_DIR, RESULTS_DIR


PRIMARY = "#234E70"
SECONDARY = "#4F6D7A"
ACCENT = "#D97B66"
NEUTRAL = "#9AA5B1"


def model_slug(model_name):
    """
    Convert a model name into a file-friendly label.
    """
    if model_name.startswith("RNN"):
        return "rnn"
    if model_name.startswith("LSTM"):
        return "lstm"
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def residual_frame(actual, predicted):
    """
    Combine actual, fitted, and residual values.
    """
    frame = pd.concat({"actual": actual, "fitted": predicted}, axis=1).dropna()
    frame["residual"] = frame["actual"] - frame["fitted"]
    return frame


def residual_diagnostics(actual, predicted, model_name, split="validation", lags=(1, 24, 168)):
    """
    Summarize forecast residuals without formal tests.
    """
    frame = residual_frame(actual, predicted)
    resid = frame["residual"]
    metrics = metric_dict(frame["actual"], frame["fitted"])
    pacf_values = pacf(resid, nlags=max(lags), method="ywm")

    summary = {
        "model": model_name,
        "split": split,
        "n": len(frame),
        "bias": resid.mean(),
        "residual_std": resid.std(),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
    }
    for lag in lags:
        summary[f"acf_lag_{lag}"] = resid.autocorr(lag=lag)
        summary[f"pacf_lag_{lag}"] = pacf_values[lag]
    summary = pd.DataFrame([summary])

    path_prefix = RESULTS_DIR / f"diagnostic_{model_slug(model_name)}_{split}"
    frame.to_csv(f"{path_prefix}_residuals.csv", index_label="timestamp")
    summary.to_csv(f"{path_prefix}_summary.csv", index=False)
    return summary


def plot_residual_diagnostics(actual, predicted, model_name, split="validation", acf_lags=72):
    """
    Plot residual time series, residual spread, and residual ACF.
    """
    frame = residual_frame(actual, predicted)
    resid = frame["residual"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.4))
    axes = axes.ravel()

    axes[0].plot(frame.index, resid, color=PRIMARY, lw=0.8)
    axes[0].axhline(0, color=ACCENT, lw=1.2)
    axes[0].set_title("A. Residuals Over Time", loc="left")
    axes[0].set_ylabel("Actual - fitted")

    axes[1].scatter(frame["fitted"], resid, s=10, color=PRIMARY, alpha=0.28, edgecolor="none")
    axes[1].axhline(0, color=ACCENT, lw=1.2)
    axes[1].set_title("B. Residuals vs Fitted", loc="left")
    axes[1].set_xlabel("Fitted demand")
    axes[1].set_ylabel("Residual")

    standardized = (resid - resid.mean()) / resid.std()
    theoretical, ordered = stats.probplot(standardized, dist="norm", fit=False)
    axes[2].scatter(theoretical, ordered, s=12, color=PRIMARY, alpha=0.45, edgecolor="none")
    axes[2].plot(theoretical, theoretical, color=ACCENT, lw=1.2)
    axes[2].set_title("C. Residual QQ Plot", loc="left")
    axes[2].set_xlabel("Theoretical quantiles")
    axes[2].set_ylabel("Standardized residual quantiles")

    plot_acf(
        resid,
        lags=acf_lags,
        ax=axes[3],
        color=PRIMARY,
        vlines_kwargs={"colors": PRIMARY, "linewidth": 1.0},
        marker="o",
        markersize=3,
    )
    axes[3].set_title("D. Residual ACF", loc="left")
    axes[3].set_xlabel("Lag")
    axes[3].set_ylabel("Autocorrelation")

    fig.suptitle(f"{model_name}: {split.title()} Residual Diagnostics", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / f"diagnostic_{model_slug(model_name)}_{split}.png", bbox_inches="tight")
    plt.show()


def poisson_dispersion_diagnostic(dispersion_df):
    """
    Report the Pearson dispersion check for the Poisson model.
    """
    out = dispersion_df.copy()
    out["assessment"] = np.where(out["dispersion"] > 1.5, "overdispersion", "close to Poisson")
    out.to_csv(RESULTS_DIR / "diagnostic_poisson_dispersion.csv", index=False)
    return out


def plot_training_history(result):
    """
    Plot train and validation loss by epoch.
    """
    history = result["history"]
    best = history.loc[history["val_loss"].idxmin()]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(history["epoch"], history["train_loss"], color=NEUTRAL, lw=1.8, label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], color=PRIMARY, lw=2.0, label="Validation loss")
    ax.scatter(best["epoch"], best["val_loss"], color=ACCENT, s=55, zorder=3, label="Best validation epoch")
    ax.set_title(f"{result['name']}: Training History", loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on scaled log demand")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"diagnostic_{model_slug(result['name'])}_training.png", bbox_inches="tight")
    plt.show()
