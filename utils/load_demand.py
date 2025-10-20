import pandas as pd
import numpy as np
import os
from utils.config_loader import load_config

data_dir = 'data/'
data_path = os.path.join(data_dir, 'load_hourly_2018.csv')
max_steps = 24

def load_demand(data_path, house_ids=None):
    df = pd.read_csv(data_path, parse_dates=['time'])
    df = df.rename(columns={'time': 'dt'})
    df['dt'] = pd.to_datetime(df['dt'], format='%d/%m/%Y %H:%M')
    df = df[df['dt'] >= '2018-01-02']
    if house_ids is not None:
        df = df[df['dataid'].isin(house_ids)]
    return df

def load_day(df, day_of_year, max_hours, year=2018):
    start = pd.to_datetime(f'{year}-{day_of_year}', format='%Y-%j')
    end = start + pd.to_timedelta(max_hours, unit='h')
    return df[(df['dt'] >= start) & (df['dt'] < end)]

def load_baselines(df):
    baselines = df[['dataid', 'dt', 'total']].copy()
    baselines.columns = ['house_id', 'timestamp', 'baseline_demand']
    return baselines

def get_peak_demand(df):
    peak = df.resample('1h', on='dt')['total'].sum().max()
    return peak

def load_device_demands(data_path, house_ids=None):
    cfg = load_config()
    DEVICES = cfg['environment']['DEVICES']
    df = pd.read_csv(data_path, parse_dates=['time'])
    df = df.rename(columns={'time': 'dt'})
    df['dt'] = pd.to_datetime(df['dt'], format='%d/%m/%Y %H:%M')
    if house_ids is not None:
        df = df[df['dataid'].isin(house_ids)]
    return df[['dt', 'dataid'] + DEVICES]

def get_device_demands(df_devices, data_ids, day, h):
    cfg = load_config()
    DEVICES = cfg['environment']['DEVICES']
    start = pd.to_datetime(f'{2018}-{day}', format='%Y-%j')
    end = start + pd.Timedelta(hours=24)
    df_day = df_devices[(df_devices['dt'] >= start) & (df_devices['dt'] < end)].copy()
    df_day['hour_idx'] = ((df_day['dt'] - start) / pd.Timedelta(hours=1)).astype(int)
    df_h = df_day[df_day['hour_idx'] == h]
    usage = df_h.set_index('dataid')[DEVICES]
    usage = usage.reindex(data_ids).fillna(0.0)
    return usage.to_numpy()


# Test code
if __name__ == "__main__":
    cfg = load_config()
    house_ids = cfg['environment']['house_ids']
    
    print("=== Testing load_demand function ===")
    df = load_demand(data_path, house_ids)
    print(f"Data shape: {df.shape}")
    print(f"House IDs in data: {df['dataid'].unique()}")
    print(f"Date range: {df['dt'].min()} to {df['dt'].max()}")
    print()
    
    print("=== Testing load_day function ===")
    day_data = load_day(df, day_of_year=200, max_hours=24)
    print(f"Day 200 data shape: {day_data.shape}")
    print(f"Hours in day 200: {day_data['dt'].nunique()}")
    print()
    
    print("=== Testing load_baselines function ===")
    baselines = load_baselines(df)
    print(f"Baselines shape: {baselines.shape}")
    print("Baselines head:")
    print(baselines.head())
    print()
    
    print("=== Testing get_peak_demand function ===")
    peak = get_peak_demand(df)
    print(f"Peak demand: {peak:.2f} kW")
    print()
    
    print("=== Testing load_device_demands function ===")
    device_demands = load_device_demands(data_path, house_ids)
    print(f"Device demands shape: {device_demands.shape}")
    print(f"Device demands date range: {device_demands['dt'].min()} to {device_demands['dt'].max()}")
    print(f"Device demands dtypes:")
    print(device_demands.dtypes)
    print("Device demands head:")
    print(device_demands.head())
    print()
    
    print("=== Testing get_device_demands function ===")
    try:
        device_matrix = get_device_demands(device_demands, house_ids, day=200, h=12)
        print(f"Device demands matrix shape: {device_matrix.shape}")
        print(f"Device demands for hour 12, day 200:")
        print(device_matrix)
        print(f"Sum of device demands: {np.sum(device_matrix, axis=1)}")
    except Exception as e:
        print(f"Error in get_device_demands: {e}")
        print(f"Debug info - day 200 start: {pd.to_datetime(f'{2018}-{200}', format='%Y-%j')}")
        print(f"Debug info - device_demands date range: {device_demands['dt'].min()} to {device_demands['dt'].max()}")
    print()
    
    print("=== All tests completed! ===")








