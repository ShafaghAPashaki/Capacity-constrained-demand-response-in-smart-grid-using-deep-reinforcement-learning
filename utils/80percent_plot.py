import pandas as pd
import matplotlib.pyplot as plt
from load_demand import load_demand

def main():

    house_ids = [4031, 7800, 8156]
    start_date = '2018-04-01'
    end_date   = '2018-10-31'


    df = load_demand(data_path='data/load_hourly_2018.csv', house_ids=house_ids)
    df = df[(df['dt'] >= start_date) & (df['dt'] <= end_date)].copy()


    summed = (
        df
        .groupby('dt')['total']
        .sum()
        .reset_index(name='sum_total')
    )


    summed['date'] = summed['dt'].dt.floor('d')
    daily_peaks = summed.groupby('date')['sum_total'].max()


    avg_peak = daily_peaks.mean()
    threshold_80 = avg_peak * 0.8


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_peaks.index, daily_peaks.values, label='Daily peak')
    ax.axhline(threshold_80,
               color='green',
               linestyle='--',
               label=f'80% of avg ({threshold_80:.2f} kW)')
    ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    ax.set_xlabel('Date')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Daily Peak Demand (Combined Houses) 2018-04-01 to 2018-10-31')
    ax.legend()
    plt.tight_layout()
    plt.show()


    print(f"Average daily peak (combined): {avg_peak:.2f} kW")
    print(f"80% of that average:          {threshold_80:.2f} kW")

if __name__ == '__main__':
    main()
