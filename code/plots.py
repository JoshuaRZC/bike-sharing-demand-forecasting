import matplotlib.pyplot as plt

from paths import FIGURES_DIR


def save_current_figure(filename):
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    return path


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
