import pandas as pd
from sklearn.linear_model import LinearRegression

from metrics import metric_dict


def fit_lagged_regression(reg_train, reg_valid, reg_test, target_col="cnt"):
    feature_cols = [col for col in reg_train.columns if col != target_col]

    model = LinearRegression()
    model.fit(reg_train[feature_cols], reg_train[target_col])

    valid_pred = pd.Series(model.predict(reg_valid[feature_cols]), index=reg_valid.index)
    test_pred = pd.Series(model.predict(reg_test[feature_cols]), index=reg_test.index)

    results = pd.DataFrame(
        [
            {"split": "validation", **metric_dict(reg_valid[target_col], valid_pred)},
            {"split": "test", **metric_dict(reg_test[target_col], test_pred)},
        ]
    )
    return model, valid_pred, test_pred, results
