import pandas as pd
import datetime, os
from utils.config_loader import load_config

data_dir = 'data/'
data_path = os.path.join(data_dir, 'load_hourly_2018.csv')
max_steps = 24

def load_demand(data_path, house_ids=None):
    df = pd.read_csv(data_path, parse_dates=['time'])
    df = df.rename(columns={'time': 'dt'})
    df = df.set_index('dt').reset_index()
    df = df[df['dt'] >= '2018-01-02']
    if house_ids is not None:
        df = df[df['dataid'].isin(house_ids)]
    df = df[['dt', 'dataid', 'total']]
    return df

def load_day(df, day, max_steps):
    time_delta = pd.to_timedelta(max_steps, unit='h')
    start_date = datetime.datetime.strptime('{} {}'.format(day, 2018), '%j %Y')
    end_date = start_date + time_delta
    mask = (df['dt'] >= start_date) & (df['dt'] < end_date)
    df = df.loc[mask]
    return df

def get_peak_demand(df):
    df = df.groupby(pd.Grouper(key='dt', freq='1h')).sum()
    return df['total'].max()

def load_baselines(df):
    baselines = df[['dataid', 'dt', 'total']].copy()
    baselines.columns = ['house_id', 'timestamp', 'baseline_demand']
    return baselines

def load_device_demands(data_path, house_ids=None):
    cfg = load_config()
    DEVICES = cfg['environment']['devices']
    df = pd.read_csv(data_path, parse_dates=['time'])
    df = df.rename(columns={'time': 'dt'})
    if house_ids is not None:
        df = df[df['dataid'].isin(house_ids)]
    return df[['dt', 'dataid'] + DEVICES]

def get_device_demands(df_devices, data_ids, day, h):
    cfg = load_config()
    DEVICES = cfg['environment']['devices']
    start = datetime.datetime.strptime(f"{day} 2018", "%j %Y")
    end   = start + datetime.timedelta(hours=len(df_devices['dt'].unique())//len(data_ids))
    df_day = df_devices[(df_devices['dt'] >= start) & (df_devices['dt'] < end)].copy()
    df_day['hour_idx'] = ((df_day['dt'] - start) / pd.Timedelta(hours=1)).astype(int)
    df_h = df_day[df_day['hour_idx'] == h]
    usage = df_h.set_index('dataid')[DEVICES]
    usage = usage.reindex(data_ids).fillna(0.0)
    return usage.to_numpy()







