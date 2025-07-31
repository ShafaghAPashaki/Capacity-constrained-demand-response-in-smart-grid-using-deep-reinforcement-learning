import numpy as np
import torch
from agent_dqn import DQNAgent, device
from env import Environment
from utils.config_loader import load_config

def main(rho):

    cfg = load_config()
    house_ids = cfg['environment']['house_ids']
    cfg['environment']['rho'] = rho
    env = Environment(house_ids, cfg_override=cfg)


    start_day = cfg['training']['testing_start_day']
    end_day   = cfg['training']['testing_end_day']
    test_days = list(range(start_day, end_day + 1))
    july_days = [d for d in test_days if 182 <= d <= 212]

  
    sample_state = env.reset(july_days[0])
    state_dim    = sample_state.shape[0]
    action_dim   = len(env.all_actions)
    agent = DQNAgent(state_dim, action_dim, cfg_override=cfg)
    state_dict = torch.load(f"dqn_model_rho{rho}.pth", map_location=device)
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()
    if hasattr(agent, "target_net"):
        agent.target_net.to(device)

  
    days  = len(july_days)
    hours = env.max_steps
    N     = len(house_ids)
    R_j   = np.empty((days*hours, N), dtype=float)
    L_j   = np.empty((days*hours, N), dtype=float)
    B_j   = np.empty((days*hours, N), dtype=float)
    P_h   = np.empty(days*hours,       dtype=float)

    for di, day in enumerate(july_days):
        state = env.reset(day)
        base  = env.baseline_per_house
        for h in range(hours):
            st     = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.policy_net(st).argmax().item()
            next_state, _, done, _ = env.step(action)

            idx = di*hours + h
            R_j[idx, :] = env.reductions[h]
            L_j[idx, :] = env.incentives[h]
            B_j[idx, :] = base[h]
            P_h[idx]    = env.prices[h]

            state = next_state

 
    mu_array = np.array(env.mu)
    omega    = env.omega

    avg_inc            = L_j.mean(axis=0)
    total_red_kwh      = R_j.sum(axis=0)
    avg_daily_red_kwh  = total_red_kwh / days
    total_income_cents = np.sum((L_j/1000)*R_j, axis=0)*100

    R_j_mwh           = R_j / 1000.0
    total_dis_cents   = np.sum((mu_array*(R_j_mwh**2)/2 + omega*R_j_mwh), axis=0)*100
    total_cu_profit   = total_income_cents - total_dis_cents

    sp_revenue_c = np.sum((P_h[:,None]/1000)*R_j)*100
    sp_payment_c = total_income_cents.sum()
    sp_profit_c  = sp_revenue_c - sp_payment_c

    load_no_dr = B_j.sum(axis=1)
    load_dr    = (B_j - R_j).sum(axis=1)
    peak_no, peak_dr = load_no_dr.max(), load_dr.max()
    mean_no, mean_dr = load_no_dr.mean(), load_dr.mean()
    par_no = peak_no / mean_no
    par_dr = peak_dr / mean_dr

   
    header = (
        "CUID | μ (kWh) | Avg inc ($/MWh) | Avg daily red (kWh) | Total red (kWh) | "
        "Inc income (¢) | Dis cost (¢) | Profit (¢)"
    )
    sep = "-" * len(header)

    print("\n" + "="*len(header))
    print(f"Customer‐Level Metrics (Total July), ρ = {rho}")
    print("="*len(header))
    print(header)
    print(sep)
    for i, hid in enumerate(house_ids):
        print(
            f"CU{hid:<4}| "
            f"{mu_array[i]:>6.1f} | "
            f"{avg_inc[i]:>16.2f} | "
            f"{avg_daily_red_kwh[i]:>20.2f} | "
            f"{total_red_kwh[i]:>15.2f} | "
            f"{total_income_cents[i]:>14.2f} | "
            f"{total_dis_cents[i]:>11.2f} | "
            f"{total_cu_profit[i]:>10.2f}"
        )

    print("\nService Provider Metrics (Total July), ρ = {}".format(rho))
    print(f"SP revenue from GO (¢)        | {sp_revenue_c:>9.2f}")
    print(f"SP payment to CU (¢)          | {sp_payment_c:>9.2f}")
    print(f"SP profit (¢)                 | {sp_profit_c:>9.2f}")

    print("\nPAR Metrics (July), ρ = {}".format(rho))
    print("Metric                   No DR     DR")
    print("-----------------------------------")
    print(f"Peak load (kW)           {peak_no:>6.2f}    {peak_dr:>6.2f}")
    print(f"Mean load (kW)           {mean_no:>6.2f}    {mean_dr:>6.2f}")
    print(f"PAR                      {par_no:>6.2f}    {par_dr:>6.2f}")

if __name__ == "__main__":
    for rho in [0.7]:
        print(f"\n=== July Financials for ρ = {rho} ===")
        main(rho)
