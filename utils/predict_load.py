import os
import joblib
import holidays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from utils.config_loader import load_config      
from utils.load_demand import load_demand  

# Plot style parameters
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18
})

class LoadDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: list, window: int):
        arr = df[features].values
        loads_log = df['total_log'].values
        X, y = [], []
        for i in range(window, len(df)):
            X.append(arr[i-window:i])
            y.append(loads_log[i])
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
    # Ensure config forecast.horizon = 1 for single-step
    df_all = load_demand(cfg['general']['load_file'], cfg['environment']['house_ids'])

    window = cfg['forecast']['window_size']
    bs = cfg['training']['batch_size']
    max_epochs = min(cfg['training']['n_epochs'], 1000)
    patience = cfg['training'].get('patience', 50)

    for house in cfg['environment']['house_ids']:
        df = df_all[df_all['dataid'] == house].copy()
        df.sort_values('dt', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Log-transform target
        df['total_log'] = np.log1p(df['total'])

        # Feature engineering
        df['dt'] = pd.to_datetime(df['dt'])
        df['month'] = df['dt'].dt.month
        df['dayofweek'] = df['dt'].dt.dayofweek + 1
        df['hour'] = df['dt'].dt.hour + 1
        df['is_weekend'] = (df['dt'].dt.weekday >= 5).astype(int)
        us_hols = holidays.US(years=[df['dt'].dt.year.min()])
        df['is_holiday'] = df['dt'].dt.date.isin(us_hols).astype(int)
        for lag in [1, 2, 3, 24, 25, 26, 48, 49, 50]:
            df[f'lag{lag}'] = df['total_log'].shift(lag)
        df.dropna(inplace=True)

        features = ['month', 'dayofweek', 'hour', 'is_weekend', 'is_holiday'] + \
                   [f'lag{lag}' for lag in [1, 2, 3, 24, 25, 26, 48, 49, 50]]
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        df['doy'] = df['dt'].dt.dayofyear

        t0, t1 = cfg['training']['training_start_day'], cfg['training']['training_end_day']
        v0, v1 = cfg['training']['testing_start_day'], cfg['training']['testing_end_day']
        train_df = df[df['doy'].between(t0, t1)].reset_index(drop=True)
        val_df = df[df['doy'].between(v0, v1)].reset_index(drop=True)

        # Datasets and loaders
        train_ds = LoadDataset(train_df, features, window)
        val_ds = LoadDataset(val_df, features, window)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs)

        # Model setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GRUForecaster(len(features), cfg['forecast']['hidden_size'],
                              cfg['forecast']['num_layers'], cfg['forecast']['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        criterion = nn.SmoothL1Loss()

        # Training loop with early stopping
        best_val_loss, wait, best_state = float('inf'), 0, None
        for ep in range(1, max_epochs + 1):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            total_val = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    total_val += criterion(model(Xb), yb).item() * Xb.size(0)
            avg_val = total_val / len(val_ds)
            scheduler.step(avg_val)

            if avg_val <= best_val_loss:
                best_val_loss, best_state, wait = avg_val, model.state_dict(), 0
            else:
                wait += 1
            if wait >= patience:
                print(f"House {house} early stopping at {ep}")
                break
            if ep % 50 == 0 or wait == 0:
                print(f"House {house} Ep {ep}/{max_epochs} val_loss={avg_val:.4f}")

        if best_state:
            model.load_state_dict(best_state)

        # Single-step prediction
        X_all = torch.tensor(
            np.stack([val_df[features].values[i-window:i] for i in range(window, len(val_df))]),
            dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            preds_log = model(X_all).detach().cpu().numpy().squeeze(1)
        preds = np.expm1(preds_log)
        times = val_df['dt'].iloc[window:].reset_index(drop=True)
        acts = val_df['total'].iloc[window:].values

        mae = mean_absolute_error(acts, preds)
        mask = acts != 0
        mape = np.nan
        if mask.sum() > 0:
            mape = np.mean(np.abs((acts[mask] - preds[mask]) / acts[mask])) * 100
        print(f"House {house} → MAE: {mae:.4f}, MAPE: {mape:.2f}%")

        # Weekly plots for July
        weeks = [
            ("2018-07-01", "2018-07-08"),
            ("2018-07-08", "2018-07-15"),
            ("2018-07-15", "2018-07-22"),
            ("2018-07-22", "2018-07-29"),
        ]
        for start_str, end_str in weeks:
            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
            mask_w = (times >= start) & (times < end)
            week_times = times[mask_w]
            week_acts = acts[mask_w]
            week_preds = preds[mask_w]

            plt.figure(figsize=(10,5))
            plt.plot(week_times, week_acts, label='Real', color='blue', linewidth=2.5)
            plt.plot(week_times, week_preds, label='Forecast', color='red', linestyle='--', linewidth=2)
            plt.xlabel('Date')
            plt.ylabel('Load')
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            plt.xticks(rotation=0)
            plt.legend()
            plt.tight_layout()
            fn = f"fig_load_house{house}_{start_str.replace('-','')}_{end_str.replace('-','')}.png"
            plt.savefig(fn, dpi=300)
            plt.close()
            print(f"Saved plot → {fn}")

        joblib.dump(scaler, f'scaler_house{house}.bin')
        torch.save(model.state_dict(), f'model_house{house}.pt')

if __name__ == "__main__":
    main()
