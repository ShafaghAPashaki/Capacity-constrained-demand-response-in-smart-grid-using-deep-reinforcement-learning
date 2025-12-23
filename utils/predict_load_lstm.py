import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utils.config_loader import load_config
from load_demand import load_demand


# reproducibility
def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# sequence dataset for LSTM
class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence-to-one forecasting."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# LSTM forecaster
class LSTMForecaster(nn.Module):
    """A LSTM-based model for one-step-ahead load forecasting. It takes a sequence of past loads and outputs the next-hour load."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """x: (batch, seq_len, input_size)"""
        out, (h_n, c_n) = self.lstm(x) # out: (batch, seq_len, hidden)
        last_hidden = out[:, -1, :] # (batch, hidden)
        y = self.fc(last_hidden) # (batch, 1)
        return y


# build LSTM sequences from raw demand data
def build_lstm_sequences(df: pd.DataFrame, cfg):
    house_ids = cfg["environment"]["house_ids"]
    horizon = cfg["forecast"]["horizon"]

    tr_cfg = cfg["training"]
    # sequence length: use training.seq_len if present, otherwise time_steps_train, otherwise 24
    seq_len = tr_cfg.get("seq_len", tr_cfg.get("time_steps_train", 24))

    df = df.sort_values(["dataid", "dt"]).reset_index(drop=True)

    X_list = []
    y_list = []
    doy_list = []
    ts_list = []
    house_list = []

    for hid in house_ids:
        sub = df[df["dataid"] == hid].sort_values("dt").reset_index(drop=True)
        if sub.empty:
            continue

        values = sub["total"].values.astype(float)
        times = sub["dt"].values  

        T = len(sub)
        min_index = seq_len - 1
        max_target = T - horizon

        if max_target <= min_index:
            # not enough points for this house
            continue

        for idx in range(min_index, max_target):
            target_idx = idx + horizon
            if target_idx >= T:
                break

            start_idx = idx - seq_len + 1
            end_idx = idx + 1  # exclusive

            seq_feats = []
            for t_idx in range(start_idx, end_idx):
                t_time = pd.Timestamp(times[t_idx])
                load_val = values[t_idx]

                # calendar “cover” features
                month = t_time.month                           
                iso_week = t_time.isocalendar().week          
                day_of_month = t_time.day                     
                hour_of_day = t_time.hour                     
                is_holiday = 0                                
                is_weekend = 1 if t_time.weekday() >= 5 else 0

                dummy = [0.0] * 7

                feat_vec = [load_val, month, iso_week, day_of_month, hour_of_day, is_holiday, is_weekend] + dummy   # length 14

                seq_feats.append(feat_vec)

            seq_feats = np.array(seq_feats, dtype=float) # (seq_len, 14)

            t_target = pd.Timestamp(times[target_idx])
            doy = t_target.timetuple().tm_yday

            X_list.append(seq_feats)
            y_list.append(values[target_idx])
            doy_list.append(doy)
            ts_list.append(t_target)
            house_list.append(hid)

    X = np.array(X_list) # (N, seq_len, 14)
    y = np.array(y_list)[:, None] # (N, 1)

    meta = {"doy": np.array(doy_list), "timestamp": np.array(ts_list), "house_id": np.array(house_list)}

    print("LSTM load sequences (14-D cover features) built.")
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Sequence length:", X.shape[1], "Input size:", X.shape[2])
    return X, y, meta


# train/val/test split and normalization for sequences
def split_and_scale_sequences(X, y, meta, cfg):
    tr_cfg = cfg["training"]
    train_ranges = tr_cfg["train_ranges"]
    val_range = tr_cfg["val_range"]
    test_range = tr_cfg["test_range"]

    doy = meta["doy"]

    def in_ranges(d, ranges):
        return any(start <= d <= end for start, end in ranges)

    train_idx = [i for i, d in enumerate(doy) if in_ranges(d, train_ranges)]
    val_idx = [i for i, d in enumerate(doy) if val_range[0] <= d <= val_range[1]]
    test_idx = [i for i, d in enumerate(doy) if test_range[0] <= d <= test_range[1]]

    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # feature-wise normalization for X
    # we flatten over samples and time steps, but keep the last dimension
    # as "channels" (here we only have 1 channel: total load).
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)  # shape (1,1,input_size)
    X_std = X_train.std(axis=(0, 1), keepdims=True)
    X_std[X_std == 0.0] = 1.0

    X_train_n = (X_train - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std

    if X_train_n.shape[2] > 1:
        X_train_n[..., 1:] = 0.0
        X_val_n[..., 1:] = 0.0
        X_test_n[..., 1:] = 0.0

    # scalar normalization for y
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std == 0.0:
        y_std = 1.0

    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    scaler = {"X_mean": X_mean, "X_std": X_std, "y_mean": float(y_mean), "y_std": float(y_std)}

    def slice_meta(idx):
        return {"doy": meta["doy"][idx], "timestamp": meta["timestamp"][idx], "house_id": meta["house_id"][idx]}

    meta_train = slice_meta(train_idx)
    meta_val = slice_meta(val_idx)
    meta_test = slice_meta(test_idx)

    print("Train size:", X_train.shape[0])
    print("Val size  :", X_val.shape[0])
    print("Test size :", X_test.shape[0])

    return ((X_train_n, y_train_n, meta_train), (X_val_n, y_val_n, meta_val), (X_test_n, y_test_n, meta_test), scaler)


# Training loop with history and early stopping
def train_model(model, train_loader, val_loader, cfg, device):
    tr_cfg = cfg["training"]
    lr = tr_cfg["lr"]
    weight_decay = float(tr_cfg["weight_decay"])
    n_epochs = tr_cfg["n_epochs"]
    patience = tr_cfg["patience"]
    log_every = tr_cfg["log_every"]
    verbose = tr_cfg["verbose"]

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    epoch_hist = []
    train_hist = []
    val_hist = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            train_loss_sum += loss.item() * batch_size
            n_train += batch_size

        train_loss = train_loss_sum / max(n_train, 1)

        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                batch_size = xb.size(0)
                val_loss_sum += loss.item() * batch_size
                n_val += batch_size

        val_loss = val_loss_sum / max(n_val, 1)

        epoch_hist.append(epoch)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(f"Epoch {epoch:4d} | "f"train_loss = {train_loss:.5f} | "f"val_loss = {val_loss:.5f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}, "f"best_val_loss = {best_val_loss:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {"epoch": epoch_hist, "train_loss": train_hist, "val_loss": val_hist}
    return model, history


def predict(model, data_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())
    return np.vstack(preds)  # (N, 1)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main():
    set_seed(42)
    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load data
    data_path = cfg["general"]["load_file"]
    house_ids = cfg["environment"]["house_ids"]
    print("Houses:", house_ids)

    df = load_demand(data_path, house_ids=house_ids)
    print("Raw data shape:", df.shape)
    print("Date range:", df["dt"].min(), "→", df["dt"].max())

    # build sequences
    X, y, meta = build_lstm_sequences(df, cfg)

    # train/val/test split + normalization
    (X_train, y_train, meta_train), \
    (X_val, y_val, meta_val), \
    (X_test, y_test, meta_test), \
    scaler = split_and_scale_sequences(X, y, meta, cfg)

    batch_size = cfg["training"]["batch_size"]

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # build LSTM model
    f_cfg = cfg["forecast"]
    input_size = X_train.shape[2]    
    hidden_size = f_cfg["hidden_size"]
    num_layers = f_cfg["num_layers"]
    dropout = f_cfg["dropout"]

    model = LSTMForecaster(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)

    print(model)

    # train model
    model, history = train_model(model, train_loader, val_loader, cfg, device)

    # save loss curve
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training & validation loss (LSTM load model)")
    plt.legend()
    plt.tight_layout()
    loss_png = os.path.join("results", "lstm_load_loss_curve.png")
    plt.savefig(loss_png, dpi=150)
    plt.close()
    print("Saved loss curve to:", loss_png)

    # predict on validation and test sets
    y_val_pred_n = predict(model, val_loader, device)
    y_test_pred_n = predict(model, test_loader, device)

    y_mean = scaler["y_mean"]
    y_std = scaler["y_std"]

    # denormalize
    y_val_true = y_val * y_std + y_mean
    y_val_pred = y_val_pred_n * y_std + y_mean

    y_test_true = y_test * y_std + y_mean
    y_test_pred = y_test_pred_n * y_std + y_mean


    df_val_results = pd.DataFrame({"timestamp": meta_val["timestamp"], "doy": meta_val["doy"], "house_id": meta_val["house_id"], "y_true": y_val_true.flatten(), "y_pred": y_val_pred.flatten()})
    val_results_csv = os.path.join("results", "lstm_val_results.csv")
    df_val_results.to_csv(val_results_csv, index=False)
    print("Saved validation predictions to:", val_results_csv)

    df_test_results = pd.DataFrame({"timestamp": meta_test["timestamp"], "doy": meta_test["doy"], "house_id": meta_test["house_id"], "y_true": y_test_true.flatten(), "y_pred": y_test_pred.flatten()})
    test_results_csv = os.path.join("results", "lstm_test_results.csv")
    df_test_results.to_csv(test_results_csv, index=False)
    print("Saved test predictions to:", test_results_csv)

    val_metrics_rows = []

    print("\nValidation metrics per house (LSTM, full val_range):")
    for hid in sorted(df_val_results["house_id"].unique()):
        sub = df_val_results[df_val_results["house_id"] == hid]
        y_t = sub["y_true"].values
        y_p = sub["y_pred"].values

        m_mae = mae(y_t, y_p)
        m_mape = mape(y_t, y_p)
        m_r2 = r2_score_np(y_t, y_p)

        val_metrics_rows.append({"house_id": hid, "MAE_kW": m_mae,"MAPE_percent": m_mape, "R2": m_r2})

        print(f"House {hid}: MAE = {m_mae:.4f}, MAPE = {m_mape:.2f}%, R^2 = {m_r2:.4f}")

    val_metrics_df = pd.DataFrame(val_metrics_rows)
    val_metrics_csv = os.path.join("results", "lstm_val_metrics_per_house.csv")
    val_metrics_df.to_csv(val_metrics_csv, index=False)
    print("Saved validation metrics per house to:", val_metrics_csv)

    test_metrics_rows = []

    print("\nTest metrics per house (LSTM, full test_range):")
    for hid in sorted(df_test_results["house_id"].unique()):
        sub = df_test_results[df_test_results["house_id"] == hid]
        y_t = sub["y_true"].values
        y_p = sub["y_pred"].values

        m_mae = mae(y_t, y_p)
        m_mape = mape(y_t, y_p)
        m_r2 = r2_score_np(y_t, y_p)

        test_metrics_rows.append({"house_id": hid, "MAE_kW": m_mae, "MAPE_percent": m_mape, "R2": m_r2})

        print(f"House {hid}: MAE = {m_mae:.4f}, MAPE = {m_mape:.2f}%, R^2 = {m_r2:.4f}")

    test_metrics_df = pd.DataFrame(test_metrics_rows)
    test_metrics_csv = os.path.join("results", "lstm_test_metrics_per_house.csv")
    test_metrics_df.to_csv(test_metrics_csv, index=False)
    print("Saved test metrics per house to:", test_metrics_csv)

    start_last_week = pd.Timestamp("2018-07-25 00:00:00")
    end_last_week = pd.Timestamp("2018-08-01 00:00:00")

    mask_last_week = ((df_test_results["timestamp"] >= start_last_week) & (df_test_results["timestamp"] < end_last_week))
    df_last_week = df_test_results[mask_last_week].copy()

    out_csv_last = os.path.join("results", "lstm_last_week_july.csv")
    df_last_week.to_csv(out_csv_last, index=False)
    print("Saved last-week-of-July predictions to:", out_csv_last)

    # per-house metrics and plots for last week of July
    house_ids_unique = sorted(df_last_week["house_id"].unique())
    metrics_rows = []

    for hid in house_ids_unique:
        sub = df_last_week[df_last_week["house_id"] == hid].sort_values("timestamp")

        y_true = sub["y_true"].values
        y_pred = sub["y_pred"].values

        # metrics for this house
        m_mae = mae(y_true, y_pred)
        m_mape = mape(y_true, y_pred)

        metrics_rows.append({"house_id": hid, "MAE_kW": m_mae, "MAPE_percent": m_mape})

        # build hourly index: 1..N
        hours = np.arange(1, len(sub) + 1)

        # plotting (bigger figure and fonts, no title)
        plt.figure(figsize=(14, 4.5))
        plt.plot(hours, y_true, color="red", label="Real")
        plt.plot(hours, y_pred, linestyle="--", color="blue", label="Forecast")

        plt.xlabel("Time (h)", fontsize=22)
        plt.ylabel("Load (kW)", fontsize=22)

        ax = plt.gca()
        ax.tick_params(axis="both", labelsize=20)
        plt.legend(fontsize=20, loc="upper right")

        plt.tight_layout()

        out_png = os.path.join("results", f"lstm_last_week_july_house_{hid}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved plot for house {hid} to:", out_png)

    # save metrics for all houses
    metrics_df = pd.DataFrame(metrics_rows)
    print("\nLast week of July metrics (LSTM model):")
    print(metrics_df)

    metrics_csv = os.path.join("results", "lstm_last_week_july_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print("Saved metrics to:", metrics_csv)

    print("Done.")


if __name__ == "__main__":
    main()
