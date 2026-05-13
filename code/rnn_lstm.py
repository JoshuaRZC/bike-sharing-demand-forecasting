import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analysis import count_predictions, model_metrics
from data import RESULTS_DIR


HISTORY_FEATURES = [
    "log_cnt",
    "temp",
    "hum",
    "windspeed",
    "yr",
    "workingday",
    "holiday",
    "weathersit",
    "hr_sin",
    "hr_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]

TARGET_FEATURES = [
    "temp",
    "hum",
    "windspeed",
    "yr",
    "workingday",
    "holiday",
    "weathersit",
    "hr_sin",
    "hr_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "log_lag_1",
    "log_lag_2",
    "log_lag_3",
    "log_lag_24",
    "log_lag_168",
]
SEQ_LAGS = (1, 2, 3, 24, 168)


class SeqNet(nn.Module):
    """
    RNN/LSTM forecaster for one-step-ahead log demand.
    """

    def __init__(
        self,
        n_history_features,
        n_target_features,
        model_type="lstm",
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if model_type == "lstm":
            self.rnn = nn.LSTM(
                n_history_features,
                hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        else:
            self.rnn = nn.RNN(
                n_history_features,
                hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                nonlinearity="tanh",
                batch_first=True,
            )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_target_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, history_x, target_x):
        """
        Combine the history state with target-time covariates.
        """
        out, _ = self.rnn(history_x)
        combined = torch.cat([out[:, -1, :], target_x], dim=1)
        return self.head(combined)


def run_lstm_window_search(
    model_df,
    train_df,
    valid_df,
    test_df,
    windows=(24, 72, 168),
    epochs=80,
    hidden_size=96,
    batch_size=128,
    learning_rate=0.0008,
    num_layers=2,
    dropout=0.15,
    use_cache=True,
    seed=248,
):
    """
    Fit or load LSTM runs and choose the best validation window.
    """
    return run_sequence_window_search(
        "lstm",
        model_df,
        train_df,
        valid_df,
        test_df,
        windows=windows,
        epochs=epochs,
        hidden_size=hidden_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_layers=num_layers,
        dropout=dropout,
        use_cache=use_cache,
        seed=seed,
    )


def run_rnn_window_search(
    model_df,
    train_df,
    valid_df,
    test_df,
    windows=(24, 72, 168),
    epochs=40,
    hidden_size=32,
    batch_size=256,
    learning_rate=0.001,
    num_layers=1,
    dropout=0.0,
    use_cache=True,
    seed=248,
):
    """
    Fit or load RNN runs and choose the best validation window.
    """
    return run_sequence_window_search(
        "rnn",
        model_df,
        train_df,
        valid_df,
        test_df,
        windows=windows,
        epochs=epochs,
        hidden_size=hidden_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_layers=num_layers,
        dropout=dropout,
        use_cache=use_cache,
        seed=seed,
    )


def run_sequence_window_search(
    model_type,
    model_df,
    train_df,
    valid_df,
    test_df,
    windows=(24, 72, 168),
    epochs=20,
    hidden_size=32,
    batch_size=256,
    learning_rate=0.001,
    num_layers=1,
    dropout=0.0,
    use_cache=True,
    seed=248,
):
    """
    Run one sequence model over several window lengths.
    """
    final_paths = {
        "model": RESULTS_DIR / f"{model_type}_model.pt",
        "valid_pred": RESULTS_DIR / f"{model_type}_valid_pred.csv",
        "test_pred": RESULTS_DIR / f"{model_type}_test_pred.csv",
        "metrics": RESULTS_DIR / f"{model_type}_metrics.csv",
        "history": RESULTS_DIR / f"{model_type}_history.csv",
        "details": RESULTS_DIR / f"{model_type}_details.csv",
        "window_search": RESULTS_DIR / f"{model_type}_window_search.csv",
    }

    if final_paths["model"].exists() and final_paths["window_search"].exists() and use_cache:
        window_df = pd.read_csv(final_paths["window_search"])
        best_window = int(window_df.loc[0, "window"])
        best_run = run_sequence_model(
            model_type,
            best_window,
            model_df,
            train_df,
            valid_df,
            test_df,
            use_cache=True,
        )
        return best_run, window_df

    runs = [
        run_sequence_model(
            model_type,
            window,
            model_df,
            train_df,
            valid_df,
            test_df,
            epochs=epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_layers=num_layers,
            dropout=dropout,
            use_cache=False,
            seed=seed,
            save_artifacts=False,
        )
        for window in windows
    ]

    rows = []
    for run in runs:
        metrics = run["metrics"]
        valid_metrics = metrics.loc[metrics["split"] == "validation"].iloc[0]
        test_metrics = metrics.loc[metrics["split"] == "test"].iloc[0]
        rows.append({
            "window": run["details"]["window"],
            "val_MAE": valid_metrics["MAE"],
            "val_RMSE": valid_metrics["RMSE"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
        })

    window_df = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    window_df.to_csv(RESULTS_DIR / f"{model_type}_window_search.csv", index=False)

    best_window = int(window_df.loc[0, "window"])
    best_run = next(run for run in runs if run["details"]["window"] == best_window)
    torch.save({"state_dict": best_run["model"].state_dict(), "details": best_run["details"]}, final_paths["model"])
    best_run["valid_pred"].rename("prediction").to_csv(final_paths["valid_pred"], index_label="timestamp")
    best_run["test_pred"].rename("prediction").to_csv(final_paths["test_pred"], index_label="timestamp")
    best_run["metrics"].to_csv(final_paths["metrics"], index=False)
    best_run["history"].to_csv(final_paths["history"], index=False)
    pd.Series(best_run["details"]).to_csv(final_paths["details"], header=["value"])
    return best_run, window_df


def run_sequence_model(
    model_type,
    window,
    model_df,
    train_df,
    valid_df,
    test_df,
    epochs=20,
    hidden_size=32,
    batch_size=256,
    learning_rate=0.001,
    num_layers=1,
    dropout=0.0,
    use_cache=True,
    seed=248,
    save_artifacts=True,
):
    """
    Train or load one RNN/LSTM model for a fixed window.
    """
    paths = {
        "model": RESULTS_DIR / f"{model_type}_model.pt",
        "valid_pred": RESULTS_DIR / f"{model_type}_valid_pred.csv",
        "test_pred": RESULTS_DIR / f"{model_type}_test_pred.csv",
        "metrics": RESULTS_DIR / f"{model_type}_metrics.csv",
        "history": RESULTS_DIR / f"{model_type}_history.csv",
        "details": RESULTS_DIR / f"{model_type}_details.csv",
    }

    name = f"{model_type.upper()} (window={window})"
    if paths["model"].exists() and use_cache:
        checkpoint = torch.load(paths["model"], map_location="cpu", weights_only=False)
        details = checkpoint["details"]
        name = f"{model_type.upper()} (window={details['window']})"
        model = SeqNet(
            len(details["history_feature_cols"]),
            len(details["target_feature_cols"]),
            model_type=model_type,
            hidden_size=details["hidden_size"],
            num_layers=details["num_layers"],
            dropout=details["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

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
            "name": name,
            "valid_pred": valid_pred,
            "test_pred": test_pred,
            "metrics": pd.read_csv(paths["metrics"]),
            "model": model,
            "details": details,
            "history": pd.read_csv(paths["history"]),
        }

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = make_sequence_data(model_df, train_df, valid_df, test_df, window)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(data["X_train"], data["target_train"], data["y_train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model = SeqNet(
        data["X_train"].shape[-1],
        data["target_train"].shape[-1],
        model_type=model_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    history_rows = []

    # Keep the epoch with the best validation loss.
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, tb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb, tb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(data["X_valid"], data["target_valid"]), data["y_valid"]).item()

        history_rows.append({
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "val_loss": float(val_loss),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        valid_scaled = model(data["X_valid"], data["target_valid"]).squeeze(-1).detach().cpu().numpy()
        test_scaled = model(data["X_test"], data["target_test"]).squeeze(-1).detach().cpu().numpy()

    # Undo target scaling and the log transform before computing metrics.
    valid_log_pred = valid_scaled * data["target_std"] + data["target_mean"]
    test_log_pred = test_scaled * data["target_std"] + data["target_mean"]
    valid_pred = count_predictions(np.expm1(valid_log_pred)).set_axis(data["valid_index"])
    test_pred = count_predictions(np.expm1(test_log_pred)).set_axis(data["test_index"])

    metrics = model_metrics(
        name,
        model_df.loc[data["valid_index"], "cnt"],
        valid_pred,
        model_df.loc[data["test_index"], "cnt"],
        test_pred,
    )
    history = pd.DataFrame(history_rows)
    details = {
        "model_type": model_type,
        "window": int(window),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "history_feature_cols": HISTORY_FEATURES,
        "target_feature_cols": TARGET_FEATURES,
        "lags": list(SEQ_LAGS),
        "target": "log1p(cnt)",
        "seed": int(seed),
    }

    if save_artifacts:
        torch.save({"state_dict": model.state_dict(), "details": details}, paths["model"])
        valid_pred.rename("prediction").to_csv(paths["valid_pred"], index_label="timestamp")
        test_pred.rename("prediction").to_csv(paths["test_pred"], index_label="timestamp")
        metrics.to_csv(paths["metrics"], index=False)
        history.to_csv(paths["history"], index=False)
        pd.Series(details).to_csv(paths["details"], header=["value"])

    return {
        "name": name,
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "metrics": metrics,
        "model": model,
        "details": details,
        "history": history,
    }


def make_sequence_data(model_df, train_df, valid_df, test_df, window):
    """
    Build scaled history windows and target-time features.
    """
    seq_df = add_sequence_features(model_df)

    # Scaling is learned from training data only.
    history_train = seq_df.loc[train_df.index, HISTORY_FEATURES]
    target_train = seq_df.loc[train_df.index, TARGET_FEATURES]
    history_means = history_train.mean()
    history_stds = history_train.std().replace(0, 1)
    target_means = target_train.mean()
    target_stds = target_train.std().replace(0, 1)
    target_mean = float(seq_df.loc[train_df.index, "log_cnt"].mean())
    target_std = float(seq_df.loc[train_df.index, "log_cnt"].std())

    scaled_history = (seq_df[HISTORY_FEATURES].astype(float) - history_means) / history_stds
    scaled_target_features = (seq_df[TARGET_FEATURES].astype(float) - target_means) / target_stds
    scaled_y = (seq_df["log_cnt"].astype(float) - target_mean) / target_std

    X_list, target_list, y_list, index_list = [], [], [], []
    history_values = scaled_history.to_numpy(dtype=np.float32)
    target_feature_values = scaled_target_features.to_numpy(dtype=np.float32)
    target_values = scaled_y.to_numpy(dtype=np.float32)

    for i in range(max(window, max(SEQ_LAGS)), len(seq_df)):
        X_list.append(history_values[i - window : i])
        target_list.append(target_feature_values[i])
        y_list.append(target_values[i])
        index_list.append(seq_df.index[i])

    X_all = np.stack(X_list).astype(np.float32, copy=False)
    target_all = np.asarray(target_list, dtype=np.float32)
    y_all = np.asarray(y_list, dtype=np.float32)
    index_array = pd.DatetimeIndex(index_list)

    train_mask = index_array.isin(train_df.index)
    valid_mask = index_array.isin(valid_df.index)
    test_mask = index_array.isin(test_df.index)

    return {
        "X_train": torch.from_numpy(X_all[train_mask]),
        "target_train": torch.from_numpy(target_all[train_mask]),
        "y_train": torch.from_numpy(y_all[train_mask]).unsqueeze(-1),
        "X_valid": torch.from_numpy(X_all[valid_mask]),
        "target_valid": torch.from_numpy(target_all[valid_mask]),
        "y_valid": torch.from_numpy(y_all[valid_mask]).unsqueeze(-1),
        "X_test": torch.from_numpy(X_all[test_mask]),
        "target_test": torch.from_numpy(target_all[test_mask]),
        "valid_index": index_array[valid_mask],
        "test_index": index_array[test_mask],
        "target_mean": target_mean,
        "target_std": target_std,
    }


def add_sequence_features(model_df):
    """
    Add log target, lags, and cyclical calendar features.
    """
    seq_df = model_df.copy()
    seq_df["log_cnt"] = np.log1p(seq_df["cnt"])
    seq_df["hr_sin"] = np.sin(2 * np.pi * seq_df["hr"] / 24)
    seq_df["hr_cos"] = np.cos(2 * np.pi * seq_df["hr"] / 24)
    seq_df["weekday_sin"] = np.sin(2 * np.pi * seq_df["weekday"] / 7)
    seq_df["weekday_cos"] = np.cos(2 * np.pi * seq_df["weekday"] / 7)
    seq_df["month_sin"] = np.sin(2 * np.pi * (seq_df["mnth"] - 1) / 12)
    seq_df["month_cos"] = np.cos(2 * np.pi * (seq_df["mnth"] - 1) / 12)
    for lag in SEQ_LAGS:
        seq_df[f"log_lag_{lag}"] = seq_df["log_cnt"].shift(lag)
    return seq_df
