import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_dict(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
    }


def comparison_table(rows):
    return pd.DataFrame(rows)
