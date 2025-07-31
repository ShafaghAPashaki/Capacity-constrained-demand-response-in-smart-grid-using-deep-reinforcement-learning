import numpy as np
import torch
from agent_dqn import DQNAgent, device
from env import Environment
from utils.config_loader import load_config
import matplotlib.pyplot as plt
import matplotlib as mpl
BIN_SIZE = 2 

plt.rcParams.update({'font.size': 18,'axes.labelsize': 18,'xtick.labelsize': 16,
                     'ytick.labelsize': 16,'legend.fontsize': 16,'legend.title_fontsize': 18})

mpl.rcParams['hatch.linewidth'] = 3  # Make hatch lines thicker


def plot_day_profiles(baselines_per_house, reductions_per_house, incentives_per_house, prices, days, house_ids, rho):
    d_color     = "#FFDD00"  # yellow for Demand
    c_color     = "#919291"  # green for Consumption
    price_color = '#D62728'  # red for Wholesale Price
    inc_color   = '#1F77B4'  # blue for Incentive Rate

    for day in days:
        B = baselines_per_house[day]
        R = reductions_per_house[day]
        I = incentives_per_house[day]
        P = prices[day]

        masked_I = I.copy()
        for j in range(masked_I.shape[1]):
            masked_I[R[:, j] == 0, j] = 0.0

        hours = np.arange(1, len(P) + 1)
        xticks = np.arange(1, len(P) + 1, BIN_SIZE)

        for idx, hid in enumerate(house_ids):
            original_load = B[:, idx]
            consumption   = original_load - R[:, idx]

            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.fill_between(hours, original_load, 0, color=d_color, alpha=0.7, label='Demand', zorder=1)
            ax1.fill_between(hours, consumption, 0, color=c_color, alpha=0.2, label='_nolegend_', zorder=2)
            ax1.fill_between(hours, consumption, 0, facecolor='none', edgecolor=c_color, hatch='/', linewidth=2, label='Consumption', zorder=3)

            ax1.plot(hours, original_load, color=d_color, linewidth=2, zorder=4)
            ax1.plot(hours, consumption, color=c_color, linewidth=1.5, zorder=4)

            ax1.set_xlabel('Hour')
            ax1.set_ylabel('Energy (kWh)')
            ax1.set_xticks(xticks)
            ax1.set_xlim(1, len(P))
            ax1.set_ylim(0, original_load.max() * 1.1)
            ax1.grid(False)

            ax2 = ax1.twinx()
            ax2.plot(hours, P, marker='o', color=price_color, linewidth=3, markersize=6, label='Wholesale Price', zorder=5)
            ax2.plot(hours, masked_I[:, idx], marker='s', color=inc_color, linewidth=3, markersize=6, label='Incentive Rate', zorder=5)  
            ax2.set_ylabel('Rate ($/MWh)')
            ax2.set_ylim(0, max(P.max(), masked_I[:, idx].max()) * 1.2)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=0.9)

            #plt.title(f'House {hid} - Day {day} (ρ={rho})')
            plt.tight_layout()

            filename = f'energy_profile_house_{hid}_day_{day}_rho{rho}.png'
            plt.savefig(filename, dpi=300)
            plt.close(fig)
            print(f"Saved → {filename}")

def plot_aggregated_load(baselines_per_house, reductions_per_house, day, rho, output_path=None):
    B = baselines_per_house[day]
    R = reductions_per_house[day]
    agg_baseline = B.sum(axis=1)
    agg_consumption = agg_baseline - R.sum(axis=1)
    hours  = np.arange(1, len(agg_baseline) + 1)
    xticks = np.arange(1, len(agg_baseline) + 1, BIN_SIZE)

    target_level = 8.3

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hours, agg_baseline, marker='o', color='red', linewidth=2, markersize=6,
            label='Aggregated Original Load')
    ax.plot(hours, agg_consumption, marker='o', color='blue', linestyle='--', linewidth=2, markersize=6,
            label='Aggregated Load After Reduction')
    ax.hlines(target_level, xmin=hours.min(), xmax=hours.max(),
               colors='green', linestyles='--', linewidth=2,
               label=f'Capacity Threshold ({target_level} kW)')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Power (kW)')
    ax.set_xticks(xticks)
    ax.set_xlim(1, len(agg_baseline))
    ax.set_ylim(0, max(agg_baseline) * 1.1)
    ax.grid(False)
    ax.legend(loc='upper right')
    #plt.title(f'Aggregated Load Curve - Day {day} (ρ={rho})')
    plt.tight_layout()

    if output_path:
        filename = output_path
    else:
        filename = f'aggregated_load_day_{day}_rho{rho}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved aggregated load plot for Day {day}, ρ={rho} → {filename}")


def main(rho):
    cfg = load_config()
    house_ids = cfg['environment']['house_ids']
    cfg['environment']['rho'] = rho
    env = Environment(house_ids, cfg_override=cfg)

    model_path = f"dqn_model_rho{rho}.pth"
    sample_state = env.reset(197)
    agent = DQNAgent(sample_state.shape[0], len(env.all_actions), cfg_override=cfg)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()
    
    baselines_dict, reductions_dict, incentives_dict, prices_dict = {}, {}, {}, {}

    # Simulate only day 208
    day = 208
    state = env.reset(day)
    done = False
    hourly_baselines, hourly_reductions = [], []
    hourly_incentives, hourly_prices = [], []

    while not done:
        st = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.policy_net(st).argmax().item()
        next_state, _, done, _ = env.step(action)
        h = env.curr_step - 1
        if h >= 0:
            hourly_baselines.append(env.baseline_per_house[h].copy())
            hourly_reductions.append(env.reductions[h].copy())
            hourly_incentives.append(env.incentives[h].copy())
            hourly_prices.append(env.prices[h])
        state = next_state

    baselines_dict[day]   = np.array(hourly_baselines)
    reductions_dict[day]  = np.array(hourly_reductions)
    incentives_dict[day]  = np.array(hourly_incentives)
    prices_dict[day]      = np.array(hourly_prices)

    plot_aggregated_load(baselines_dict, reductions_dict, day, rho,
                         output_path=f"aggregated_load_day_{day}_rho{rho}.png")
    plot_day_profiles(baselines_dict, reductions_dict, incentives_dict, prices_dict,
                      days=[day], house_ids=house_ids, rho=rho)


if __name__ == "__main__":
    for rho in [0.7]:
        print(f"\n=== Plotting day profiles for ρ = {rho} ===")
        main(rho)
