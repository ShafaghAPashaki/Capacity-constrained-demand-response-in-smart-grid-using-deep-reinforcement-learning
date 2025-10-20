import pandas as pd
import numpy as np
import os

data_dir = 'data/'
data_path = os.path.join(data_dir, 'ercot_hourly_price.csv')

def load_price(data_path):
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.rename(columns={'timestamp': 'dt', 'Price': 'price'})
    df['price'] = df['price'] / 10.0
    df = df[df['dt'] >= '2018-01-02']
    df.set_index('dt', inplace=True)
    return df



