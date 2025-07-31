import numpy as np
import torch
from agent_dqn import DQNAgent, device
from env import Environment
from utils.config_loader import load_config
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16,'axes.labelsize': 18,'xtick.labelsize': 14,
                     'ytick.labelsize': 14,'legend.fontsize': 14,'legend.title_fontsize': 16})

def compute_monthly_profits(rho):
    # Load config & override rho
    cfg = load_config()
    house_ids = cfg['environment']['house_ids']
    cfg['environment']['rho'] = rho
    env = Environment(house_ids, cfg_override=cfg)

    start_day = cfg['training']['testing_start_day']
    end_day   = cfg['training']['testing_end_day']
    july_days = [d for d in range(start_day, end_day+1) if 182 <= d <= 212]

    sample = env.reset(july_days[0])
    agent = DQNAgent(sample.shape[0], len(env.all_actions), cfg_override=cfg)
    agent.policy_net.load_state_dict(torch.load(f"dqn_model_rho{rho}.pth", map_location=device))
    agent.policy_net.eval()

    days = len(july_days)
    H    = env.max_steps
    N    = len(house_ids)

    B = np.zeros((days*H, N))
    R = np.zeros((days*H, N))
    L = np.zeros((days*H, N))
    P = np.zeros(days*H)

    idx = 0
    for day in july_days:
        state = env.reset(day)
        base  = env.baseline_per_house
        for h in range(H):
            st     = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.policy_net(st).argmax().item()
            nxt, _, done, _ = env.step(action)

            B[idx, :] = base[h]
            R[idx, :] = env.reductions[h]
            L[idx, :] = env.incentives[h]
            P[idx]    = env.prices[h]

            state = nxt
            idx += 1

    sp_revenue       = np.sum((P/1000) * R.sum(axis=1)) * 100
    income_cu        = np.sum((L/1000) * R, axis=0) * 100
    sp_payment_to_cu = income_cu.sum()
    sp_profit        = sp_revenue - sp_payment_to_cu

    mu_array = np.array(env.mu)
    omega    = env.omega
    R_mwh    = R / 1000.0
    dis_cost = np.sum((mu_array * (R_mwh**2) / 2 + omega * R_mwh), axis=0) * 100
    cu_profit = income_cu - dis_cost

    return sp_profit, cu_profit

if __name__ == "__main__":
    rhos = [0.1, 0.3, 0.5, 0.7, 0.9]

    cfg = load_config()
    start_day = cfg['training']['testing_start_day']
    end_day   = cfg['training']['testing_end_day']
    july_days = [d for d in range(start_day, end_day+1) if 182 <= d <= 212]
    days = len(july_days)

    sp_avg, cu_avg = [], []
    for rho in rhos:
        sp_profit_c, cu_profit_arr = compute_monthly_profits(rho)
        sp_avg.append(sp_profit_c / days)
        cu_avg.append(np.mean(cu_profit_arr) / days)
        print(f"ρ={rho} | SP avg profit: {sp_avg[-1]:.2f}¢/day | CU avg profit: {cu_avg[-1]:.2f}¢/day")

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    ax1.plot(rhos, sp_avg, marker='o', linestyle='-', color='tab:blue', label='SP avg profit')
    ax2.plot(rhos, cu_avg, marker='s', linestyle='--', color='tab:red', label='Customers avg profit')

    ax1.set_xlabel('Weighting factor ρ')
    ax1.set_ylabel('Average profit of service provider (¢)')
    ax2.set_ylabel('Average profit of customers (¢)')

    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    fig.patch.set_facecolor('white')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('avg_profit_vs_rho_july_cents.png', dpi=300)
    print("Saved plot → avg_profit_vs_rho_july_cents.png")
