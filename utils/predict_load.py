import joblib
import os
import holidays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from utils.config_loader import load_config      
from utils.load_demand import load_demand  
    
class LoadDataset(Dataset):
    """Dataset for multi-step load forecasting"""
    def __init__(self, df: pd.DataFrame, features: list, window: int, horizon: int):
        arr   = df[features].values
        loads = df['total'].values
        X, y  = [], []
        for i in range(window, len(df) - horizon):
            X.append(arr[i-window:i])
            y.append(loads[i:i+horizon])
        self.X = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y = torch.tensor(np.stack(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GRUForecaster(nn.Module):
    def __init__(self, in_size:int, hidden_size:int, num_layers:int, horizon:int, dropout:float=0.2):
        super().__init__()
        self.gru = nn.GRU(in_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        last   = out[:, -1, :]
        return self.fc(last)

def main():
    # load config
    cfg = load_config()

    # load all three houses' data
    df_all = load_demand(cfg['general']['load_file'],
                         cfg['environment']['house_ids'])  # contentReference[oaicite:3]{index=3}

    # common pipeline parameters
    window  = cfg['forecast']['window_size']
    horizon = cfg['forecast']['horizon']
    bs      = cfg['training']['batch_size']
    epochs  = cfg['training']['n_epochs']
    lr      = cfg['training']['lr']

    # we'll collect per-house metrics here
    results = {}

    for house in cfg['environment']['house_ids']:
        # subset
        df = df_all[df_all['dataid']==house].copy()
        df = df.sort_values('dt').reset_index(drop=True)

        # feature engineering
        df['dt']         = pd.to_datetime(df['dt'])
        df['month']      = df['dt'].dt.month
        df['dayofweek']  = df['dt'].dt.dayofweek + 1
        df['hour']       = df['dt'].dt.hour + 1

        # weekend & holiday
        df['is_weekend'] = (df['dt'].dt.weekday >= 5).astype(int)
        us_hols = holidays.US(years=[df['dt'].dt.year.min()])
        df['is_holiday'] = df['dt'].dt.date.isin(us_hols).astype(int)

        # lags short & long
        for lag in [1,2,3,24,25,26,48,49,50]:
            df[f'lag{lag}'] = df['total'].shift(lag)
        df.dropna(inplace=True)

        features = [
            'month','dayofweek','hour',
            'is_weekend','is_holiday'
        ] + [f'lag{lag}' for lag in [1,2,3,24,25,26,48,49,50]]
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        # split by day‐of‐year
        df['doy'] = df['dt'].dt.dayofyear
        t0, t1 = cfg['training']['training_start_day'], cfg['training']['training_end_day']
        v0, v1 = cfg['training']['testing_start_day'],  cfg['training']['testing_end_day']
        train_df = df[df['doy'].between(t0, t1)].reset_index(drop=True)
        val_df   = df[df['doy'].between(v0, v1)].reset_index(drop=True)

        # DataLoaders
        train_ds = LoadDataset(train_df, features, window, horizon)
        val_ds   = LoadDataset(val_df,   features, window, horizon)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=bs)

        # model / optimizer / loss
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GRUForecaster(
            in_size    = len(features),
            hidden_size= cfg['forecast']['hidden_size'],
            num_layers = cfg['forecast']['num_layers'],
            horizon    = horizon,
            dropout    = cfg['forecast']['dropout']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # training loop
        for ep in range(1, epochs+1):
            model.train()
            total_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(Xb)
                loss  = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * Xb.size(0)

        # one‐step forecasts on validation
        all_preds, all_actuals = [], []
        model.eval()
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                out = model(Xb).cpu().numpy()
                all_preds.append(out)
                all_actuals.append(yb.numpy())
        preds_array   = np.vstack(all_preds)  
        actuals_array = np.vstack(all_actuals)

        # align times to first horizon step
        pred_times = val_df['dt'].iloc[window:len(val_df)-horizon].reset_index(drop=True)
        pred_1     = preds_array[:, 0]
        act_1      = actuals_array[:, 0]

        df_pred = pd.DataFrame({
            'dt':       pred_times,
            'real':     act_1,
            'forecast': pred_1
        })

        joblib.dump(scaler,           f'scaler_house{house}.bin')
        torch.save(model.state_dict(), f'model_house{house}.pt')

        mae = mean_absolute_error(df_pred['real'], df_pred['forecast'])

        mask = df_pred['real'] != 0
        mape = mean_absolute_percentage_error(
            df_pred['real'][mask],
            df_pred['forecast'][mask]
        )

        print(f"House {house} → MAE: {mae:.4f}, MAPE: {mape:.4f}")

        week_starts = pd.to_datetime([
            "2018-07-01","2018-07-08","2018-07-15","2018-07-22"
        ])
        for i, ws in enumerate(week_starts, start=1):
            we = ws + pd.Timedelta(days=7)
            wk = df_pred[(df_pred['dt']>=ws)&(df_pred['dt']<we)].copy()
            wk['hours'] = (wk['dt'] - ws).dt.total_seconds()/3600

            plt.figure(figsize=(10,5))
            plt.plot(wk['hours'], wk['real'],    label='Real')
            plt.plot(wk['hours'], wk['forecast'],label='Forecast')
            plt.xlabel('Time (h)')
            plt.ylabel('Load (kW)')
            plt.title(f'House {house}: Week {i} ({ws.date()}–{(we- pd.Timedelta(days=1)).date()})')
            plt.legend()
            plt.tight_layout()

            fn = f'fig_load_house{house}_week{i}.png'
            plt.savefig(fn, dpi=300)
            plt.close()
            print(f"Saved plot → {fn}")

    print("\nAll houses forecast performance:")
    for h, m in results.items():
        print(f" • House {h}: MAE={m['MAE']:.4f}, MAPE={m['MAPE']:.4f}")

_scalers    = {}    # house_id -> MinMaxScaler
_models     = {}    # house_id -> GRUForecaster
_demand_df  = None  # full demand DataFrame

def forecast_load(day: int, hour: int, house_id: int) -> np.ndarray:
    global _scalers, _models, _demand_df

    cfg     = load_config()
    window  = cfg['forecast']['window_size']
    horizon = cfg['forecast']['horizon']

    if _demand_df is None:
        df = load_demand(
            data_path = cfg['general']['load_file'],
            house_ids = cfg['environment']['house_ids']
        )
        # feature engineering exactly as in training —
        df['dt']         = pd.to_datetime(df['dt'])
        df['month']      = df['dt'].dt.month
        df['dayofweek']  = df['dt'].dt.dayofweek + 1
        df['hour']       = df['dt'].dt.hour + 1

        # weekend & holiday
        df['is_weekend'] = (df['dt'].dt.weekday >= 5).astype(int)
        us_hols = holidays.US(years=[df['dt'].dt.year.min()])
        df['is_holiday'] = df['dt'].dt.date.isin(us_hols).astype(int)

        # lags short & long
        for lag in [1,2,3,24,25,26,48,49,50]:
            df[f'lag{lag}'] = df['total'].shift(lag)
        df.dropna(inplace=True)  # remove rows with NaN from shifts
        _demand_df = df.reset_index(drop=True)

    if house_id not in _scalers:
        scaler = joblib.load(f"scaler_house{house_id}.bin")
        _scalers[house_id] = scaler

        m = GRUForecaster(
            in_size    = 14,  # ['month','dayofweek','hour','lag1','lag2','lag3']
            hidden_size= cfg['forecast']['hidden_size'],
            num_layers = cfg['forecast']['num_layers'],
            horizon    = cfg['forecast']['horizon'],
            dropout    = cfg['forecast']['dropout']
        )
        m.load_state_dict(torch.load(f"model_house{house_id}.pt", map_location='cpu'))
        m.eval()
        _models[house_id] = m

    scaler = _scalers[house_id]
    model  = _models[house_id]

    df_h = (_demand_df[_demand_df['dataid']==house_id]
            .sort_values('dt')
            .reset_index(drop=True))
    mask = (df_h['dt'].dt.dayofyear == day) & (df_h['dt'].dt.hour == hour)
    idx  = np.where(mask)[0][0]
    block = df_h.iloc[idx-window:idx]

    features = ['month','dayofweek','hour',
    'is_weekend','is_holiday',
    'lag1','lag2','lag3',
    'lag24','lag25','lag26',
    'lag48','lag49','lag50']
    X  = scaler.transform(block[features])
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(Xt).cpu().numpy().flatten()

    return out[:horizon]

if __name__ == "__main__":
    main()

