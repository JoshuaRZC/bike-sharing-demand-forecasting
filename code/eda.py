import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from scipy.signal import welch
from statsmodels.tsa.stattools import acf, pacf

from data import FIGURES_DIR


PRIMARY = "#234E70"
SECONDARY = "#4F6D7A"
ACCENT = "#D97B66"
WEEKEND = "#F4A261"
NEUTRAL = "#9AA5B1"


def set_analysis_style():
    """
    Apply the common display and plot style for the notebook.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
    })
    pd.set_option("display.max_columns", 100)


def basic_summary(model_df):
    """
    Summarize missing values and basic ranges in the modeling data.
    """
    summary = model_df.describe().T
    summary.insert(1, "missing", model_df.isna().sum())
    summary["count"] = summary["count"].astype(int)
    summary["missing"] = summary["missing"].astype(int)
    return summary


def lag_autocorr(series, lags):
    """
    Compute autocorrelations at selected lags.
    """
    return pd.Series({f"lag_{lag}": series.autocorr(lag=lag) for lag in lags})


def plot_demand_scale(model_df):
    """
    Plot the target distribution and daily demand trend.
    """
    daily_cnt = model_df["cnt"].resample("D").mean()
    rolling_7 = daily_cnt.rolling(7, min_periods=1).mean()
    median_cnt = model_df["cnt"].median()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)

    axes[0].hist(
        model_df["cnt"],
        bins=42,
        color=PRIMARY,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.92,
    )
    axes[0].axvline(median_cnt, color=ACCENT, lw=2, label=f"Median = {median_cnt:.0f}")
    axes[0].set_title("A. Demand Distribution", loc="left")
    axes[0].set_xlabel("Hourly rentals")
    axes[0].set_ylabel("Number of hours")
    axes[0].legend(loc="upper right")

    axes[1].plot(daily_cnt.index, daily_cnt, color=NEUTRAL, lw=0.8, alpha=0.75, label="Daily mean")
    axes[1].plot(rolling_7.index, rolling_7, color=PRIMARY, lw=2.0, label="7-day moving average")
    axes[1].set_title("B. Daily Average Demand", loc="left")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Average hourly rentals")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].legend(loc="upper left")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(FIGURES_DIR / "eda_demand_scale.png", bbox_inches="tight")
    plt.show()


def plot_calendar_patterns(model_df):
    """
    Plot monthly, weekly, and hourly demand patterns.
    """
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    calendar_df = model_df.copy()
    calendar_df["day_name"] = pd.Categorical(
        calendar_df.index.day_name().str[:3],
        categories=day_order,
        ordered=True,
    )

    hour_week = (
        calendar_df.groupby(["day_name", "hr"], observed=False)["cnt"]
        .median()
        .unstack()
        .loc[day_order]
    )

    profile = (
        model_df.assign(day_type=model_df["workingday"].map({1: "Working day", 0: "Non-working day"}))
        .groupby(["day_type", "hr"])["cnt"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    monthly_groups = [model_df.loc[model_df["mnth"] == month, "cnt"] for month in range(1, 13)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)

    box = axes[0].boxplot(
        monthly_groups,
        positions=np.arange(1, 13),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": ACCENT, "linewidth": 1.7},
        boxprops={"edgecolor": PRIMARY, "linewidth": 1.1},
        whiskerprops={"color": PRIMARY, "linewidth": 1.0},
        capprops={"color": PRIMARY, "linewidth": 1.0},
    )
    fill_boxes(box)

    axes[0].set_title("A. Monthly Demand Pattern", loc="left")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Hourly rentals")
    axes[0].set_xticks(np.arange(1, 13))

    im = axes[1].imshow(hour_week.values, aspect="auto", cmap="Blues", origin="upper")
    axes[1].set_title("B. Weekday-Hour Demand Pattern", loc="left")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Day of week")
    axes[1].set_xticks(np.arange(0, 24, 3))
    axes[1].set_yticks(np.arange(len(day_order)))
    axes[1].set_yticklabels(day_order)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Median rentals")

    for label, color in [("Working day", PRIMARY), ("Non-working day", ACCENT)]:
        sub = profile[profile["day_type"] == label].sort_values("hr")
        axes[2].plot(sub["hr"], sub["median"], color=color, lw=2.0, label=label)
        axes[2].fill_between(
            sub["hr"].to_numpy(),
            sub["q25"].to_numpy(),
            sub["q75"].to_numpy(),
            color=color,
            alpha=0.16,
        )

    axes[2].set_title("C. Hourly Pattern by Working Status", loc="left")
    axes[2].set_xlabel("Hour of day")
    axes[2].set_ylabel("Hourly rentals")
    axes[2].set_xticks(np.arange(0, 24, 3))
    axes[2].legend(loc="upper left")

    for ax in [axes[0], axes[2]]:
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    fig.savefig(FIGURES_DIR / "eda_calendar_patterns.png", bbox_inches="tight")
    plt.show()


def plot_weather_relationships(model_df, bins=10):
    """
    Plot demand against weather variables and weather condition.
    """
    weather_specs = [
        ("temp", "A. Temperature", "Temperature"),
        ("hum", "B. Humidity", "Humidity"),
        ("windspeed", "C. Wind Speed", "Wind speed"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6), constrained_layout=True)
    axes = axes.flatten()

    for ax, (col, title, xlabel) in zip(axes[:3], weather_specs):
        weather_bins = pd.cut(model_df[col], bins=bins)
        summary = (
            model_df.assign(weather_bin=weather_bins)
            .groupby("weather_bin", observed=False)
            .agg(
                x_mid=(col, "mean"),
                median_cnt=("cnt", "median"),
                q25=("cnt", lambda x: x.quantile(0.25)),
                q75=("cnt", lambda x: x.quantile(0.75)),
            )
            .reset_index(drop=True)
            .dropna(subset=["x_mid", "median_cnt"])
        )

        ax.plot(summary["x_mid"], summary["median_cnt"], marker="o", color=PRIMARY, lw=2, ms=4)
        ax.fill_between(
            summary["x_mid"].to_numpy(),
            summary["q25"].to_numpy(),
            summary["q75"].to_numpy(),
            color=PRIMARY,
            alpha=0.16,
        )

        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Hourly rentals")
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    weather_labels = {
        1: "Clear",
        2: "Mist",
        3: "Light rain/snow",
        4: "Heavy rain/snow",
    }
    weather_groups = [model_df.loc[model_df["weathersit"] == key, "cnt"] for key in weather_labels]

    box = axes[3].boxplot(weather_groups, positions=np.arange(1, 5), **box_style())
    fill_boxes(box)

    axes[3].set_title("D. Weather Situation", loc="left")
    axes[3].set_xlabel("Weather situation")
    axes[3].set_ylabel("Hourly rentals")
    axes[3].set_xticks(np.arange(1, 5))
    axes[3].set_xticklabels([f"{key}\n{label}" for key, label in weather_labels.items()])
    axes[3].grid(True, axis="y", alpha=0.25)
    axes[3].grid(False, axis="x")

    fig.savefig(FIGURES_DIR / "eda_weather_pattern.png", bbox_inches="tight")
    plt.show()


def plot_mean_variance_check(model_df):
    """
    Compare rolling mean-variance patterns on raw and log scales.
    """
    rolling_window = 24 * 14
    min_periods = 24 * 7
    log_cnt = np.log1p(model_df["cnt"])

    scale_df = pd.DataFrame({
        "raw_mean": model_df["cnt"].rolling(rolling_window, min_periods=min_periods).mean(),
        "raw_var": model_df["cnt"].rolling(rolling_window, min_periods=min_periods).var(),
        "log_mean": log_cnt.rolling(rolling_window, min_periods=min_periods).mean(),
        "log_var": log_cnt.rolling(rolling_window, min_periods=min_periods).var(),
    }).dropna()

    raw_corr = scale_df["raw_mean"].corr(scale_df["raw_var"])
    log_corr = scale_df["log_mean"].corr(scale_df["log_var"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), constrained_layout=True)

    axes[0].scatter(
        scale_df["raw_mean"],
        scale_df["raw_var"],
        s=10,
        color=PRIMARY,
        alpha=0.5,
        edgecolors="none",
    )
    axes[0].set_title(f"A. Raw Count Scale (r = {raw_corr:.2f})", loc="left")
    axes[0].set_xlabel("14-day rolling mean")
    axes[0].set_ylabel("14-day rolling variance")

    axes[1].scatter(
        scale_df["log_mean"],
        scale_df["log_var"],
        s=10,
        color=ACCENT,
        alpha=0.5,
        edgecolors="none",
    )
    axes[1].set_title(f"B. Log1p Count Scale (r = {log_corr:.2f})", loc="left")
    axes[1].set_xlabel("14-day rolling mean of log1p(cnt)")
    axes[1].set_ylabel("14-day rolling variance of log1p(cnt)")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.savefig(FIGURES_DIR / "eda_mean_variance_check.png", bbox_inches="tight")
    plt.show()


def log_dependence_data(model_df, lags=(1, 2, 3, 24, 48, 168)):
    """
    Build log-transformed series and selected autocorrelation summaries.
    """
    log_cnt = np.log1p(model_df["cnt"])
    log_diff_1 = log_cnt.diff().dropna()
    log_diff_24 = log_cnt.diff(24).dropna()

    acf_summary = pd.DataFrame({
        "log_cnt": lag_autocorr(log_cnt, lags),
        "log_diff_1": lag_autocorr(log_diff_1, lags),
        "log_diff_24": lag_autocorr(log_diff_24, lags),
    })

    return acf_summary, log_cnt, log_diff_1, log_diff_24


def plot_acf_pacf_diagnostics(log_cnt, log_diff_1, log_diff_24):
    """
    Plot ACF and PACF diagnostics for the log target.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), constrained_layout=True)
    title_prefix = "Log1p Hourly Demand"

    plot_acf_stem(
        axes[0, 0],
        log_cnt.to_numpy(),
        f"A. ACF ({title_prefix})",
        max_lag=168,
        highlight_lags=[1, 2, 3, 24, 48, 168],
    )
    plot_acf_stem(
        axes[0, 1],
        log_diff_1.to_numpy(),
        f"B. First-Differenced ACF ({title_prefix})",
        max_lag=168,
        highlight_lags=[1, 2, 3, 24, 48, 168],
    )
    plot_acf_stem(
        axes[1, 0],
        log_diff_24.to_numpy(),
        f"C. 24-Hour Differenced ACF ({title_prefix})",
        max_lag=168,
        highlight_lags=[1, 2, 3, 24, 48, 168],
    )
    plot_pacf_stem(
        axes[1, 1],
        log_diff_24.to_numpy(),
        f"D. 24-Hour Differenced PACF ({title_prefix})",
        max_lag=72,
    )

    fig.savefig(FIGURES_DIR / "eda_acf_pacf_diagnostics.png", bbox_inches="tight")
    plt.show()


def plot_welch_periodograms(log_cnt, log_diff_1, log_diff_24):
    """
    Plot Welch periodograms for log demand and its differences.
    """
    periodogram_series = [
        (log_cnt.dropna(), "A. Log1p Demand"),
        (log_diff_1, "B. First Difference"),
        (log_diff_24, "C. 24-Hour Difference"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.4), constrained_layout=True)

    for ax, (series, title) in zip(axes, periodogram_series):
        freq, power = welch(
            series - series.mean(),
            fs=1,
            window="hann",
            nperseg=24 * 28,
            noverlap=24 * 14,
            detrend="constant",
            scaling="density",
        )
        period_hours = 1 / freq[1:]
        power = power[1:]

        ax.plot(period_hours, power, color=PRIMARY, lw=1.5)
        for period, label, color in [(24, "24-hour cycle", ACCENT), (168, "168-hour cycle", WEEKEND)]:
            idx = np.argmin(np.abs(period_hours - period))
            ax.axvline(period, color=color, ls="--", lw=1.4, label=label)
            ax.scatter(period_hours[idx], power[idx], color=color, s=26, zorder=3)

        ax.set_title(title, loc="left")
        ax.set_xlabel("Period (hours)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(2, 336)
        ax.set_xticks([2, 4, 8, 24, 48, 168, 336])
        ax.set_xticklabels(["2", "4", "8", "24", "48", "168", "336"])
        ax.grid(True, which="major", alpha=0.25)

    for ax in axes:
        ax.set_ylabel("Power spectral density")

    axes[0].legend(loc="upper right")

    fig.savefig(FIGURES_DIR / "eda_welch_periodograms.png", bbox_inches="tight")
    plt.show()


def box_style():
    return {
        "widths": 0.56,
        "patch_artist": True,
        "showfliers": False,
        "medianprops": {"color": ACCENT, "linewidth": 1.7},
        "boxprops": {"edgecolor": PRIMARY, "linewidth": 1.1},
        "whiskerprops": {"color": PRIMARY, "linewidth": 1.0},
        "capprops": {"color": PRIMARY, "linewidth": 1.0},
    }


def fill_boxes(box):
    for patch in box["boxes"]:
        patch.set_facecolor(PRIMARY)
        patch.set_alpha(0.18)


def add_confidence_band(ax, lags, values, confint):
    band = confint[1:] - values[1:, None]
    ax.fill_between(lags, band[:, 0], band[:, 1], color=NEUTRAL, alpha=0.26, zorder=0)
    ax.plot(lags, band[:, 0], color=NEUTRAL, lw=0.7, alpha=0.65, zorder=1)
    ax.plot(lags, band[:, 1], color=NEUTRAL, lw=0.7, alpha=0.65, zorder=1)


def plot_acf_stem(ax, values, title, max_lag, highlight_lags=None):
    acf_vals, confint = acf(values, nlags=max_lag, fft=True, alpha=0.05)
    lags = np.arange(1, max_lag + 1)

    add_confidence_band(ax, lags, acf_vals, confint)
    ax.vlines(lags, 0, acf_vals[1:], color=SECONDARY, lw=1.0, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)

    if highlight_lags is not None:
        highlight_lags = np.array([lag for lag in highlight_lags if lag <= max_lag])
        ax.vlines(highlight_lags, 0, acf_vals[highlight_lags], color=ACCENT, lw=2.1)
        ax.scatter(highlight_lags, acf_vals[highlight_lags], color=ACCENT, s=22, zorder=3)

    ax.set_title(title, loc="left")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    ax.set_xlim(0, max_lag + 2)
    ax.set_ylim(-0.4, 1.0)
    ax.grid(True, axis="y", alpha=0.25)


def plot_pacf_stem(ax, values, title, max_lag):
    pacf_vals, confint = pacf(values, nlags=max_lag, method="ywm", alpha=0.05)
    lags = np.arange(1, max_lag + 1)

    add_confidence_band(ax, lags, pacf_vals, confint)
    ax.vlines(lags, 0, pacf_vals[1:], color=SECONDARY, lw=1.0, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("PACF")
    ax.set_xlim(0, max_lag + 2)
    ax.set_ylim(-0.4, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
