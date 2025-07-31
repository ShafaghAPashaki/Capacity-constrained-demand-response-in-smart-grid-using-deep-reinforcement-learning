import os
import yaml
import joblib
import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from utils.config_loader import load_config      
from utils.load_price import load_price      

plt.rcParams.update({'font.size': 16,'axes.labelsize': 18,'xtick.labelsize': 14,
                     'ytick.labelsize': 14,'legend.fontsize': 14,'legend.title_fontsize': 16})

class PriceDataset(Dataset):
    """Dataset for multi-step price forecasting"""
    def __init__(self, df: pd.DataFrame, features: list, window: int, horizon: int):
        arr    = df[features].values
        prices = df['price'].values
        X, y = [], []
        for i in range(window, len(df) - horizon):
            X.append(arr[i-window:i])
            y.append(prices[i:i+horizon])
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

    cfg = load_config()


    df = load_price(cfg['general']['price_file'])
    df['month']     = df['dt'].dt.month
    df['dayofweek'] = df['dt'].dt.dayofweek + 1
    df['hour']      = df['dt'].dt.hour + 1
    # weekend & holiday
    df['is_weekend'] = (df['dt'].dt.weekday >= 5).astype(int)
    us_hols = holidays.US(years=[df['dt'].dt.year.min()])
    df['is_holiday'] = df['dt'].dt.date.isin(us_hols).astype(int)
    # lags short & long
    for lag in [1,2,3,24,25,26,48,49,50]:
        df[f'lag{lag}'] = df['price'].shift(lag)
    df.dropna(inplace=True)

    features = [
        'month','dayofweek','hour',
        'is_weekend','is_holiday'
    ] + [f'lag{lag}' for lag in [1,2,3,24,25,26,48,49,50]]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    df['doy'] = df['dt'].dt.dayofyear
    t0, t1 = cfg['training']['training_start_day'], cfg['training']['training_end_day']
    v0, v1 = cfg['training']['testing_start_day'],  cfg['training']['testing_end_day']
    train_df = df[df['doy'].between(t0, t1)].reset_index(drop=True)
    val_df   = df[df['doy'].between(v0, v1)].reset_index(drop=True)

    window  = cfg['forecast']['window_size']
    horizon = cfg['forecast']['horizon']
    bs      = cfg['training']['batch_size']

    train_ds = PriceDataset(train_df, features, window, horizon)
    val_ds   = PriceDataset(val_df,   features, window, horizon)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUForecaster(
        in_size    = len(features),
        hidden_size= cfg['forecast']['hidden_size'],   
        num_layers = cfg['forecast']['num_layers'],    
        horizon    = horizon,
        dropout    = cfg['forecast']['dropout']       
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    criterion = nn.MSELoss()

    epochs = cfg['training']['n_epochs']  
    train_losses, val_losses = [], []

    for ep in range(1, epochs+1):
        model.train()
        total_train = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss= criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * Xb.size(0)
        train_losses.append(total_train / len(train_ds))

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                total_val += criterion(model(Xb), yb).item() * Xb.size(0)
        val_losses.append(total_val / len(val_ds))

        print(f"Epoch {ep}/{epochs}  train_loss={train_losses[-1]:.4f}"
              f"  val_loss={val_losses[-1]:.4f}")
        
    all_preds, all_actuals = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            out = model(Xb).cpu().numpy()
            all_preds.append(out)
            all_actuals.append(yb.numpy())
    preds_array   = np.vstack(all_preds)    # shape: (num_samples, horizon)
    actuals_array = np.vstack(all_actuals)

    pred_times = val_df['dt'].iloc[window:len(val_df)-horizon].reset_index(drop=True)
    pred_1     = preds_array[:, 0]
    act_1      = actuals_array[:, 0]

    df_pred = pd.DataFrame({
        'dt':       pred_times,
        'real':     act_1,
        'forecast': pred_1
    })

    start = pd.to_datetime("2018-07-20")
    end   = pd.to_datetime("2018-07-27")  

    mask = (df_pred['dt'] >= start) & (df_pred['dt'] < end)
    week_df = df_pred.loc[mask].copy()

    week_df['hours'] = (week_df['dt'] - start).dt.total_seconds() / 3600

    plt.figure(figsize=(10,5))
    plt.plot(week_df['hours'], week_df['real'],    label='Real')
    plt.plot(week_df['hours'], week_df['forecast'],label='Forecast')
    plt.xlabel('Hour')
    plt.ylabel('Price ($/MWh)')
    plt.legend()
    plt.tight_layout()

    fn = 'fig_price_Jul20-27.png'
    plt.savefig(fn, dpi=300)
    plt.close()
    print(f"Saved plot → {fn}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            preds = model(Xb).cpu().numpy()
            y_pred.extend(preds.flatten())
            y_true.extend(yb.numpy().flatten())

    mae_val  = mean_absolute_error(y_true, y_pred)
    mape_val = mean_absolute_percentage_error(y_true, y_pred)
    print("\nTable 3 – Forecast Performance")
    print(f"MAE: {mae_val:.4f}")
    print(f"MAPE: {mape_val:.4f}")

    joblib.dump(scaler, 'scaler_price.bin')
    torch.save(model.state_dict(), 'price2step.pth')
    print("Saved inference artifacts → scaler_price.bin, price2step.pth")

# --- module‐level cache ---
_price_scaler = None
_price_model  = None
_price_df     = None

def forecast_price(day: int, hour: int) -> np.ndarray:
    global _price_scaler, _price_model, _price_df

    cfg     = load_config()
    window  = cfg['forecast']['window_size']
    horizon = cfg['forecast']['horizon']

    if _price_df is None:
        df = load_price(cfg['general']['price_file'])
        # feature engineering exactly as in training 
        df['dt'] = pd.to_datetime(df['dt'])
        df['month']     = df['dt'].dt.month
        df['dayofweek'] = df['dt'].dt.dayofweek + 1
        df['hour']      = df['dt'].dt.hour + 1
        # weekend & holiday
        df['is_weekend'] = (df['dt'].dt.weekday >= 5).astype(int)
        us_hols = holidays.US(years=[df['dt'].dt.year.min()])
        df['is_holiday'] = df['dt'].dt.date.isin(us_hols).astype(int)
        # lags short & long
        for lag in [1,2,3,24,25,26,48,49,50]:
            df[f'lag{lag}'] = df['price'].shift(lag)
        df.dropna(inplace=True)
        _price_df = df.sort_values('dt').reset_index(drop=True)

    if _price_scaler is None:
        _price_scaler = joblib.load("scaler_price.bin")
        m = GRUForecaster(
            in_size    = 14,  # ['month','dayofweek','hour','is_holiday','lag1','lag2','lag3']
            hidden_size= cfg['forecast']['hidden_size'],
            num_layers = cfg['forecast']['num_layers'],
            horizon    = horizon,
            dropout    = cfg['forecast']['dropout']
        )
        m.load_state_dict(torch.load("price2step.pth", map_location='cpu'))
        m.eval()
        _price_model = m

    df     = _price_df
    scaler = _price_scaler
    model  = _price_model

    # inference
    mask  = (df['dt'].dt.dayofyear == day) & (df['dt'].dt.hour == hour)
    idx   = np.where(mask)[0][0]
    block = df.iloc[idx-window:idx]

    features = ['month','dayofweek','hour',
    'is_weekend','is_holiday',
    'lag1','lag2','lag3',
    'lag24','lag25','lag26',
    'lag48','lag49','lag50']
    X   = scaler.transform(block[features])
    Xt  = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(Xt).cpu().numpy().flatten()

    return out[:horizon]

if __name__ == "__main__":
    main()
