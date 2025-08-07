import os
import joblib
import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from utils.config_loader import load_config
from utils.load_price import load_price

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16, 'legend.title_fontsize': 18})

class PriceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: list, window: int):
        arr = df[features].values
        prices_log = df['price_log'].values
        X, y = [], []
        for i in range(window, len(df)):
            X.append(arr[i-window:i])
            y.append(prices_log[i])
        self.X = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GRUForecaster(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(in_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)

def main():
    cfg = load_config()
    df = load_price(cfg['general']['price_file'])
    df['dt'] = pd.to_datetime(df['dt'])

    # Log-transform target
    df['price_log'] = np.log1p(df['price'])

    # Feature engineering
    df['month'] = df['dt'].dt.month
    df['dayofweek'] = df['dt'].dt.dayofweek + 1
    df['hour'] = df['dt'].dt.hour + 1
    df['is_weekend'] = (df['dt'].dt.weekday >= 5).astype(int)
    us_hols = holidays.US(years=[df['dt'].dt.year.min()])
    df['is_holiday'] = df['dt'].dt.date.isin(us_hols).astype(int)
    for lag in [1, 2, 3, 24, 25, 26, 48, 49, 50]:
        df[f'lag{lag}'] = df['price_log'].shift(lag)
    df.dropna(inplace=True)

    features = ['month', 'dayofweek', 'hour','is_weekend', 'is_holiday'] + [f'lag{lag}' for lag in [1, 2, 3, 24, 25, 26, 48, 49, 50]]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    df['doy'] = df['dt'].dt.dayofyear

    # Split train/val
    t0, t1 = cfg['training']['training_start_day'], cfg['training']['training_end_day']
    v0, v1 = cfg['training']['testing_start_day'], cfg['training']['testing_end_day']
    train_df = df[df['doy'].between(t0, t1)].reset_index(drop=True)
    val_df = df[df['doy'].between(v0, v1)].reset_index(drop=True)

    window = cfg['forecast']['window_size']
    bs = cfg['training']['batch_size']
    max_epochs = min(cfg['training']['n_epochs'], 1000)
    patience = cfg['training'].get('patience', 50)

    train_ds = PriceDataset(train_df, features, window)
    val_ds = PriceDataset(val_df, features, window)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUForecaster(in_size=len(features), hidden_size=cfg['forecast']['hidden_size'], num_layers=cfg['forecast']['num_layers'], dropout=cfg['forecast']['dropout']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    criterion = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    wait = 0
    best_state = None

    for ep in range(1, max_epochs + 1):
        model.train()
        total_train = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * Xb.size(0)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                loss = criterion(out, yb)
                total_val += loss.item() * Xb.size(0)

        avg_train = total_train / len(train_ds)
        avg_val = total_val / len(val_ds)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1

        if ep % 50 == 0 or wait == 0:
            print(f"Epoch {ep}/{max_epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch={ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Prediction
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            out = model(Xb).cpu().numpy()
            all_preds.append(np.expm1(out))
            all_actuals.append(np.expm1(yb.numpy()))
    preds_array = np.vstack(all_preds)
    actuals_array = np.vstack(all_actuals)

    pred_times = val_df['dt'].iloc[window:].reset_index(drop=True)
    df_pred = pd.DataFrame({'dt': pred_times, 'real': actuals_array[:, 0], 'forecast': preds_array[:, 0]})

    df_pred[['real', 'forecast']] = df_pred[['real', 'forecast']].clip(upper=80)

    mae = mean_absolute_error(df_pred['real'], df_pred['forecast'])
    mape = mean_absolute_percentage_error(df_pred['real'], df_pred['forecast']) * 100
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    weeks = [
        ("2018-07-01", "2018-07-08"),
        ("2018-07-08", "2018-07-15"),
        ("2018-07-15", "2018-07-22"),
        ("2018-07-22", "2018-07-29"),
    ]
    for start_str, end_str in weeks:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        mask = (df_pred['dt'] >= start) & (df_pred['dt'] < end)
        week_df = df_pred.loc[mask].copy()

        plt.figure(figsize=(10, 5))
        plt.plot(week_df['dt'], week_df['real'], label='Real', color='blue', linewidth=2.5)
        plt.plot(week_df['dt'], week_df['forecast'], label='Forecast', color='red', linestyle='--', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Price ($/MWh)')
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        plt.xticks(rotation=0)
        plt.legend()
        plt.tight_layout()
        fn = f"fig_price_{start_str.replace('-', '')}_{end_str.replace('-', '')}.png"
        plt.savefig(fn, dpi=300)
        plt.close()
        print(f"Saved plot → {fn}")

    joblib.dump(scaler, 'scaler_price.bin')
    torch.save(model.state_dict(), 'price_onestep.pth')
    print("Saved inference artifacts → scaler_price.bin, price_onestep.pth")

if __name__ == "__main__":
    main()

