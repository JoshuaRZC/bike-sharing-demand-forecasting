from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def metric_dict(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def plot_forecast_comparison(actual, predictions, title, start=None):
    if start is not None:
        actual = actual.loc[start:]
        predictions = {name: pred.loc[start:] for name, pred in predictions.items()}

    plt.figure(figsize=(14, 5))
    plt.plot(actual, label="Actual", color="black", lw=2)
    for name, pred in predictions.items():
        plt.plot(pred, label=name, alpha=0.9)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("cnt")
    plt.legend()
    plt.tight_layout()
    return plt.gca()
