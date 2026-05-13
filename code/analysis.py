import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import FIGURES_DIR, RESULTS_DIR


def metric_dict(y_true, y_pred):
    """
    Return MAE and RMSE for one prediction vector.
    """
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def count_predictions(pred):
    """
    Keep count-scale predictions in the feasible range.
    """
    return pd.Series(pred).clip(lower=0)


def model_metrics(model_name, valid_true, valid_pred, test_true, test_pred):
    """
    Compute validation and test metrics on the count scale.
    """
    # Demand forecasts are clipped before scoring because counts cannot be negative.
    valid_pred = count_predictions(valid_pred).set_axis(valid_true.index)
    test_pred = count_predictions(test_pred).set_axis(test_true.index)
    return pd.DataFrame(
        [{"model": model_name, "split": "validation", **metric_dict(valid_true, valid_pred)},
         {"model": model_name, "split": "test", **metric_dict(test_true, test_pred)}]
    )


def make_comparison_table(model_results):
    """
    Combine model metrics and save the final comparison table.
    """
    comparison = pd.concat([result["metrics"] for result in model_results], ignore_index=True)
    # Keep validation first because it is used for model selection.
    comparison["_split_order"] = pd.Categorical(
        comparison["split"],
        categories=["validation", "test"],
        ordered=True,
    )
    comparison = (
        comparison.sort_values(["_split_order", "RMSE"])
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    return comparison


def plot_forecast_comparison(actual, predictions, title, start):
    """
    Plot actual demand against one or more forecast series.
    """
    actual = actual.loc[start:]
    predictions = {name: pred.loc[start:] for name, pred in predictions.items()}

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual, label="Actual", color="black", lw=2)
    for name, pred in predictions.items():
        ax.plot(pred, label=name, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("cnt")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "analysis_forecast_comparison.png", bbox_inches="tight")
    return ax


def plot_model_comparison(actual, model_results, split="test", days=7):
    """
    Plot recent actual demand against model forecasts.
    """
    pred_key = f"{split}_pred"
    start = actual.index[-24 * days]
    predictions = {result["name"]: result[pred_key] for result in model_results}
    title = f"Last {days} Days of {split.title()} Period: Actual vs Predicted"
    return plot_forecast_comparison(actual, predictions, title, start=start)


def plot_interactive_model_comparison(actual, model_results, split="test", days=7, window_options=(7, 14, 30, 60)):
    """
    Plot forecasts with model highlighting and time-window controls.
    """
    pred_key = f"{split}_pred"
    predictions = {result["name"]: result[pred_key] for result in model_results}
    colors = ["#D97B66", "#234E70", "#4F6D7A", "#7A9E7E", "#B08968", "#6C5B7B"]

    def window_range(option):
        start_pos = max(0, len(actual) - 24 * option)
        start_time = actual.index[start_pos]
        end_time = actual.index[-1]
        window_values = [actual.loc[start_time:end_time]]
        window_values += [pred.loc[start_time:end_time] for pred in predictions.values()]
        y_values = pd.concat(window_values).dropna()
        y_min, y_max = y_values.min(), y_values.max()
        pad = max((y_max - y_min) * 0.08, 10)
        return [start_time, end_time], [max(0, y_min - pad), y_max + pad]

    window_ranges = {option: window_range(option) for option in window_options}
    active_window = list(window_options).index(days)
    window_buttons = [
        {
            "label": f"{option}d",
            "method": "relayout",
            "args": [{"xaxis.range": x_range, "yaxis.range": y_range}],
        }
        for option, (x_range, y_range) in window_ranges.items()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual,
        mode="lines",
        name="Actual",
        line={"color": "#111111", "width": 2.6},
    ))
    for i, (name, pred) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(
            x=pred.index,
            y=pred,
            mode="lines",
            name=name,
            line={"color": colors[i % len(colors)], "width": 1.35},
            opacity=0.85,
        ))

    trace_ids = list(range(1 + len(predictions)))
    buttons = [{
        "label": "All models",
        "method": "restyle",
        "args": [
            {"opacity": [1] + [0.85] * len(predictions),
             "line.width": [2.6] + [1.35] * len(predictions)},
            trace_ids,
        ],
    }]
    for i, name in enumerate(predictions):
        opacities = [1] + [0.14] * len(predictions)
        widths = [2.7] + [0.75] * len(predictions)
        opacities[i + 1] = 1
        widths[i + 1] = 1.6
        buttons.append({
            "label": name,
            "method": "restyle",
            "args": [
                {"opacity": opacities, "line.width": widths},
                trace_ids,
            ],
        })

    fig.update_layout(
        title={"text": "Actual vs Predicted", "x": 0.07, "xanchor": "left", "y": 0.87, "yanchor": "top", "font": {"size": 25}},
        xaxis_title="Time",
        yaxis_title="cnt",
        template="plotly_white",
        width=980,
        height=450,
        hovermode="x unified",
        updatemenus=[
            {
                "buttons": window_buttons,
                "type": "dropdown",
                "direction": "down",
                "active": active_window,
                "x": 0.67,
                "xanchor": "right",
                "y": 1.18,
                "yanchor": "top",
            },
            {
                "buttons": buttons,
                "type": "dropdown",
                "direction": "down",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.18,
                "yanchor": "top",
            },
        ],
        legend={"yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
        margin={"l": 60, "r": 180, "t": 100, "b": 70},
    )
    x_range, y_range = window_ranges[days]
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)
    fig.write_html(FIGURES_DIR / "analysis_interactive_forecast_comparison.html")
    return fig
