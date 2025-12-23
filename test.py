import os
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from env import Environment
from agent.agent_dqn import DQNAgent
from utils.config_loader import load_config


plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18,
})
mpl.rcParams['hatch.linewidth'] = 3  # hatch line width
BIN_SIZE = 2


class ModelTester:
    def __init__(self, model_path, cfg_override=None):
        self.cfg = cfg_override or load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # Setup environment
        house_ids = self.cfg['environment']['house_ids']
        self.env = Environment(data_ids=house_ids, cfg_override=self.cfg)
        
        # Get state and action dimensions
        state = self.env.reset(mode='test')
        state_dim = len(state)
        action_dim = self.env.num_actions
        
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Device: {self.device}")
        
        # Setup agent and load trained model
        self.agent = DQNAgent(state_dim, action_dim, cfg_override=self.cfg)
        self.load_model(model_path)
        
        # Set results directory to model directory by default
        self.results_dir = os.path.dirname(model_path)
        print(f"Results will be saved to: {self.results_dir}")


    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.agent.policy_net.eval()
        print(f"Model successfully loaded from: {model_path}")


    def plot_day_profiles(self, baselines_per_house, reductions_per_house, incentives_per_house,
                        prices, device_after_per_house_dict, nonshift_dict, days, house_ids, rho):
        # Colors (match reference)
        d_color = "#FFDD00" # Demand (yellow)
        c_color = "#919291" # Consumption (grey)
        price_color = "#D62728"
        inc_color = "#1F77B4"

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
                original_load = B[:, idx]  # Demand (baseline)
                # after = (sum devices after) + non-shiftable
                A_ctrl = device_after_per_house_dict[day].sum(axis=2)  # (H, N)
                NS = nonshift_dict[day] # (H, N)
                consumption = A_ctrl[:, idx] + NS[:, idx]

                fig, ax1 = plt.subplots(figsize=(10, 6))

                # Area: Demand + Consumption (hatch)
                ax1.fill_between(hours, original_load, 0, color=d_color, alpha=0.7, label='Demand', zorder=1)
                ax1.fill_between(hours, consumption, 0, color=c_color, alpha=0.2, label='_nolegend_', zorder=2)
                ax1.fill_between(hours, consumption, 0, facecolor='none', edgecolor=c_color,
                                hatch='/', linewidth=2, label='Consumption', zorder=3)

                # Demand/Consumption
                ax1.plot(hours, original_load, color=d_color, linewidth=2, zorder=4)
                ax1.plot(hours, consumption,   color=c_color, linewidth=1.5, zorder=4)

                ax1.set_xlabel('Hour')
                ax1.set_ylabel('Energy (kWh)')
                ax1.set_xticks(xticks)
                ax1.set_xlim(1, len(P))
                ax1.set_ylim(0, max(1e-9, original_load.max()) * 1.1)
                ax1.grid(False)

                # Price & Incentive
                ax2 = ax1.twinx()
                ax2.plot(hours, P, marker='o', color=price_color, linewidth=3, markersize=6,
                        label='Wholesale Price', zorder=5)
                ax2.plot(hours, masked_I[:, idx], marker='s', color=inc_color,   linewidth=3, markersize=6,
                        label='Incentive Rate',  zorder=5)
                ax2.set_ylabel('Rate (¢/kWh)')
                ax2.set_ylim(0, max(P.max(), masked_I[:, idx].max(), 1e-9) * 1.2)

                # Combined legend   
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=0.9)

                plt.tight_layout()
                out = os.path.join(self.results_dir, f'energy_profile_house_{hid}_day_{day}_rho{rho}.png')
                plt.savefig(out, dpi=300)
                plt.close(fig)


    def plot_aggregated_load(self, baselines_per_house, after_total_dict, day, rho, output_path=None):
        agg_baseline = baselines_per_house[day].sum(axis=1) # (H,)
        agg_after = after_total_dict[day].sum(axis=1) # (H,)

        hours = np.arange(1, len(agg_baseline) + 1)
        xticks = np.arange(1, len(agg_baseline) + 1, BIN_SIZE)
        target_level = 7 # kWh

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hours, agg_baseline, marker='o', color='red', linewidth=2, markersize=6, label='Aggregated Original Load')
        ax.plot(hours, agg_after, marker='o', color='blue', linestyle='--', linewidth=2, markersize=6,
                label='Aggregated Load After Reduction')
        ax.hlines(target_level, xmin=hours.min(), xmax=hours.max(), colors='green', linestyles='--', linewidth=2,
                label=f'Capacity Threshold ({target_level} kW)')

        ax.set_xlabel('Hour')
        ax.set_ylabel('Aggregated load (kW)')
        ax.set_xticks(xticks)
        ax.set_xlim(1, len(agg_baseline))
        ax.set_ylim(0, max(agg_baseline.max(), 1e-9) * 1.1)
        ax.grid(False)
        ax.legend(loc='upper right')
        plt.tight_layout()

        filename = output_path or os.path.join(self.results_dir, f'aggregated_load_day_{day}_rho{rho}.png')
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Aggregated load plot saved: {filename}")


    def plot_devices_house_split(
        self,
        device_baselines_per_house_dict, # (H, N, D) = self.env.device_before
        device_reductions_per_house_dict, # (H, N, D) = max(before - after, 0)
        device_after_per_house_dict, # (H, N, D) = self.env.device_after
        day,
        house_ids,
        device_names=None
    ):
        
        if day not in device_baselines_per_house_dict:
            print(f"[WARN] day {day} not found in device_baselines_per_house_dict")
            return

        B = device_baselines_per_house_dict[day] # (H, N, D)
        A = device_after_per_house_dict[day]
        R = device_reductions_per_house_dict.get(day, None)
        if R is None:
            R = np.zeros_like(B)

        H, N, D = B.shape
        hours = np.arange(1, H + 1)
        xticks = np.arange(1, H + 1, BIN_SIZE)

        if device_names is None and hasattr(self.env, "DEVICES"):
            device_names = [str(n).lower() for n in self.env.DEVICES]
        elif device_names is None:
            device_names = [f"dev{d}" for d in range(D)]
        else:
            device_names = [str(n).lower() for n in device_names]

        for hid in house_ids:
            try:
                idx = list(self.env.data_ids).index(hid)
            except ValueError:
                print(f"[WARN] house_id {hid} not found in env.data_ids; skip.")
                continue

            base_hd = B[:, idx, :] # (H, D)
            after_hd = A[:, idx, :]
            red_hd  = R[:, idx, :] # (H, D)

            for d in range(D):
                name = device_names[d] if d < len(device_names) else f"dev{d}"
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(hours, base_hd[:, d], color='#D62728', linestyle='-', linewidth=2.4, label=f'{name} (baseline)', zorder=2)
                ax.plot(hours, after_hd[:, d], color='#1F77B4', linestyle='--', linewidth=3.0, label=f'{name} (after)', zorder=3)

                ax.set_xlabel('Hour')
                ax.set_ylabel('Energy (kWh)')
                ax.set_xticks(xticks)
                ax.set_xlim(1, H)

                ymax = max(base_hd[:, d].max(), after_hd[:, d].max(), 1e-9)
                ax.set_ylim(0, ymax * 1.15)
                ax.grid(False)
                ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
                ax.set_title(f'{name} – House {hid} – Day {day}')

                plt.tight_layout()
                outpath = os.path.join(self.results_dir, f'device_{name}_house_{hid}_day_{day}.png')
                plt.savefig(outpath, dpi=300)
                plt.close()
                print(f"Saved: {outpath}")


    def plot_total_incentive_rate(self, incentives_dict, day):
        I = incentives_dict[day] # shape: (H, N), ¢/kWh
        H, _ = I.shape
        hours = np.arange(1, H + 1)
        xticks = np.arange(1, H + 1, BIN_SIZE)

        sum_rate = I.sum(axis=1) #(¢/kWh)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hours, sum_rate, linewidth=3)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Incentives (¢/kWh)")
        ax.set_xticks(xticks)
        ax.set_xlim(1, H)
        ax.grid(False)
        plt.tight_layout()

        out = os.path.join(self.results_dir, f"total_incentive_rate_day_{day}.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved: {out}")


    def plot_total_incentive_per_house(self, incentives_dict, reductions_dict, day, house_ids):
        if day not in incentives_dict:
            print(f"[WARN] day {day} not found in incentives_dict.")
            return
        if day not in reductions_dict:
            print(f"[WARN] day {day} not found in reductions_dict.")
            return

        # incentives: (H, N) in ¢/kWh
        I = incentives_dict[day]
        # reductions: (H, N) in kWh
        R = reductions_dict[day]

        # Total payout per EU in cent: sum_h lambda * ΔE
        total_payout = (I * R).sum(axis=0) # (N,), ¢
        # Total reduction per EU in kWh: sum_h ΔE
        total_reduction = R.sum(axis=0) # (N,), kWh

        x = np.arange(len(house_ids))
        width = 0.35

        fig, ax_left = plt.subplots(figsize=(10, 5))
        ax_right = ax_left.twinx()

        # Blue bars: total payout (left axis)
        bars_payout = ax_left.bar(
            x - width/2,
            total_payout,
            width=width,
            label='Total payout (¢)'
        )

        # Orange bars: total reduction (right axis)
        bars_reduction = ax_right.bar(
            x + width/2,
            total_reduction,
            width=width,
            label='Total reduction (kWh)',
            color='darkorange'
        )

        # X-ticks as End User 1, 2, 3
        ax_left.set_xticks(x)
        ax_left.set_xticklabels([f'End User {i+1}' for i in range(len(house_ids))])
        ax_left.set_xlabel('End users')

        # Y labels
        ax_left.set_ylabel('Total payout (¢)')
        ax_right.set_ylabel('Total reduction (kWh)')

        # No title
        # ax_left.set_title(...)

        # Combined legend
        handles = [bars_payout, bars_reduction]
        labels = [h.get_label() for h in handles]
        ax_left.legend(handles, labels, loc='upper left')

        ax_left.grid(False)
        fig.tight_layout()

        out = os.path.join(self.results_dir, f'total_payout_and_reduction_per_house_day_{day}.png')
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved: {out}")


    def test_plots(self):
        rho = self.cfg['environment']['rho']
        house_ids = self.cfg['environment']['house_ids']
        
        # Get test days from config
        target_day = 208
        july_days = [target_day]
        print(f"Generating aggregated-load plot only for day {target_day}")

        print(f"Generating test plots for {len(july_days)} days: {july_days}")

        baselines_dict, reductions_dict, incentives_dict, prices_dict = {}, {}, {}, {}
        nonshift_dict, after_total_dict = {}, {}
        device_baselines_per_house_dict, device_reductions_per_house_dict = {}, {}
        device_after_per_house_dict = {} # (H, N, D_ctrl)

        for day in july_days:
            hourly_baselines, hourly_reductions, hourly_incentives, hourly_prices = [], [], [], []
            hourly_device_baselines, hourly_device_reductions = [], []
            hourly_device_baselines_per_house, hourly_device_reductions_per_house = [], []
            hourly_device_after_per_house = []

            state = self.env.reset(day=day, mode='test')
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.agent.policy_net(state_tensor).argmax().item()
                next_state, _, done, _ = self.env.step(action)

                h = self.env.curr_step - 1
                if h >= 0:
                    true_baseline = self.env.baseline_per_house[h].copy() #(N,)
                    after_house = self.env.device_after[h].sum(axis=1) if hasattr(self.env, "device_after") else (true_baseline - self.env.reductions[h])
                    red_house = np.maximum(true_baseline - after_house, 0.0)
                    hourly_baselines.append(true_baseline)
                    hourly_reductions.append(red_house)
                    hourly_incentives.append(self.env.incentives[h].copy())
                    hourly_prices.append(self.env.prices[h])
                    hourly_device_baselines.append(self.env.device_after_agg[h].copy())
                    hourly_device_reductions.append(self.env.reductions_by_device[h].copy())
                    hourly_device_baselines_per_house.append(self.env.device_before[h].copy())
                    hourly_device_reductions_per_house.append(np.maximum(self.env.device_before[h] - self.env.device_after[h], 0.0))
                    hourly_device_after_per_house.append(self.env.device_after[h].copy())

                state = next_state

            baselines_dict[day] = self.env.baseline_per_house.copy() # (H, N)
            # Build reductions per house (H, N) robustly:
            if hasattr(self.env, "delta_E"):
                 reductions_dict[day] = self.env.delta_E.copy()
            elif hasattr(self.env, "before_total_per_house") and hasattr(self.env, "after_total_per_house"):
                 reductions_dict[day] = np.maximum(
                     self.env.before_total_per_house - self.env.after_total_per_house, 0.0)
            else:
                 # Fallback from device-level and non-shiftable (shouldn’t be needed if env is patched)
                 B_ctrl = self.env.device_before.sum(axis=2) # (H, N)
                 A_ctrl = self.env.device_after.sum(axis=2) # (H, N)
                 if hasattr(self.env, "nonshift_baseline_per_house"):
                     NS = self.env.nonshift_baseline_per_house
                 else:
                     NS = np.zeros_like(B_ctrl)
                 before_total = B_ctrl + NS
                 after_total = A_ctrl + NS
                 reductions_dict[day] = np.maximum(before_total - after_total, 0.0)
            prices_dict[day] = self.env.prices.copy() # (H,)
            incentives_dict[day] = self.env.incentives.copy() # (H, N)
            device_after_per_house_dict[day] = self.env.device_after.copy() # (H, N, D_ctrl)
            nonshift_dict[day] = self.env.nonshift_baseline_per_house.copy()  # (H, N)
            after_total_dict[day] = self.env.after_total_per_house.copy() # (H, N)
            device_baselines_per_house_dict[day] = self.env.device_before.copy() # (H, N, D)
            device_reductions_per_house_dict[day] = np.maximum(self.env.device_before - self.env.device_after, 0.0)   

            self.plot_aggregated_load(
             baselines_per_house=baselines_dict,
             after_total_dict=after_total_dict,
             day=day, rho=rho
         )
        # Plot daily profiles for a specific target day
        target_day = 208
        if target_day in july_days:
            self.plot_day_profiles(
                baselines_dict,
                reductions_dict,
                incentives_dict,
                prices_dict,
                device_after_per_house_dict,  
                nonshift_dict,                 
                days=[target_day],
                house_ids=house_ids,
                rho=rho
            )
            self.plot_aggregated_load(
                baselines_per_house=baselines_dict,
                after_total_dict=after_total_dict,
                day=day, rho=rho
            )
            self.plot_devices_house_split(
                device_baselines_per_house_dict,
                device_reductions_per_house_dict,
                device_after_per_house_dict,   
                day=target_day,
                house_ids=house_ids,
                device_names=getattr(self.env, "DEVICES", None)
            )
            self.plot_total_incentive_rate(incentives_dict, day=target_day)
            self.plot_total_incentive_per_house(
                incentives_dict,
                reductions_dict,
                day=target_day,
                house_ids=house_ids
            )
            

        print("All test plots generated successfully")


    def financial_report(self):
        rho = self.cfg['environment']['rho']
        house_ids = self.cfg['environment']['house_ids']

        target_day = 208
        july_days = [target_day]
        print(f"Generating financial report for day {target_day}")

        # Collect simulation data
        days, hours, N = len(july_days), self.env.max_steps, len(house_ids)

        # Buffers per hour
        R_j = np.zeros((days * hours, N), dtype=float)  # delta_E (kWh)
        L_j = np.zeros((days * hours, N), dtype=float)  # incentives (¢/kWh)
        B_j = np.zeros((days * hours, N), dtype=float)  # baseline per house (kWh)
        P_h = np.zeros((days * hours, 1), dtype=float)  # price (¢/kWh)
        D_j = np.zeros((days * hours, N), dtype=float)  # discomfort per house (cent-equivalent)
        A_j = np.zeros((days * hours, N), dtype=float)  # AFTER consumption per house (kWh) 

        valid_rows = 0

        for di, day in enumerate(july_days):  
            state = self.env.reset(day=day, mode='test')
            for h in range(hours):
                idx = di * hours + h

                # Greedy eval action (no exploration)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.agent.policy_net(state_tensor).argmax().item()

                next_state, _, done, _ = self.env.step(action)

                # Pull hour-level arrays from env buffers (NO recomputation)
                R_j[idx] = self.env.delta_E[h] # kWh
                L_j[idx] = self.env.incentives[h] # ¢/kWh
                B_j[idx] = self.env.baseline_per_house[h] # kWh
                P_h[idx] = self.env.prices[h] # ¢/kWh
                D_j[idx] = self.env.discomforts[h] # cent-equivalent (PC + TS)
                A_j[idx] = self.env.after_total_per_house[h].copy()  # (controllable + nonshift)

                valid_rows += 1
                state = next_state
                if done:
                    break

        # Trim to valid rows
        R_valid = R_j[:valid_rows]  # (H, N)
        L_valid = L_j[:valid_rows]  # (H, N)
        B_valid = B_j[:valid_rows]  # (H, N)
        P_valid = P_h[:valid_rows]  # (H, 1)
        D_valid = D_j[:valid_rows]  # (H, N)
        A_valid = A_j[:valid_rows]  # (H, N) 

        # SP revenue from GO (¢): price * total reduction
        sp_revenue_direct = float((P_valid * R_valid).sum())  # ¢

        # Per-house payout: sum_h λ_{h,i} * ΔE_{h,i}
        income_raw = (L_valid * R_valid).sum(axis=0) # ¢ per house 

        # Raw discomfort: sum over hours of recorded discomfort 
        dis_raw = D_valid.sum(axis=0)

        # Effective values 
        income_vec = rho * income_raw            
        dis_cost_vec = (1.0 - rho) * dis_raw       

        sp_revenue = float(sp_revenue_direct) # ¢
        sp_payment = float(income_vec.sum()) # ¢
        sp_profit = float(sp_revenue - sp_payment) # ¢
        load_no_dr = B_valid.sum(axis=1) # kWh per hour (aggregated baseline)
        load_dr = A_valid.sum(axis=1) # kWh per hour (aggregated after DR)

        par_data = {
            'day': target_day,
            'Peak_load_no_DR_kWh': float(load_no_dr.max()),
            'Peak_load_DR_kWh': float(load_dr.max()),
            'Mean_load_no_DR_kWh': float(load_no_dr.mean()),
            'Mean_load_DR_kWh': float(load_dr.mean()),
            'PAR_no_DR': float(load_no_dr.max() / max(load_no_dr.mean(), 1e-9)),
            'PAR_DR': float(load_dr.max() / max(load_dr.mean(), 1e-9)),
        }

        customer_metrics = []
        for i, hid in enumerate(house_ids):
            total_red_i = float(R_valid[:, i].sum())  # kWh
            avg_inc_i = float(L_valid[:, i].mean()) # ¢/kWh
            income_i = float(income_vec[i]) # ¢
            dis_i = float(dis_cost_vec[i]) # ¢
            customer_metrics.append({
                'day': target_day,
                'CUID': f'CU{hid}',
                'Avg_inc_¢_per_kWh': avg_inc_i,
                'Total_red_kWh': total_red_i,
                'Inc_income_cents': income_i,
                'Discomfort_cents': dis_i,
                'Profit_cents': income_i - dis_i
            })

        sp_data = {
            'day': target_day,
            'SP_revenue_from_GO_cents': sp_revenue,
            'SP_payment_to_CU_cents': sp_payment,
            'SP_profit_cents': sp_profit
        }

        total_B = B_valid.sum()
        total_A = A_valid.sum()
        total_R = R_valid.sum()

        #print(f"[INFO] total baseline kWh : {total_B:.3f}")
        #print(f"[INFO] total after kWh : {total_A:.3f}")
        #print(f"[INFO] real reduction kWh : {total_B - total_A:.3f}")
        #print(f"[INFO] sum(delta_E) kWh : {total_R:.3f}")

        if (R_valid < -1e-9).any():
            print("[WARN] Negative delta_E found.")
        if (A_valid < -1e-9).any():
            print("[WARN] Negative AFTER load found.")

        cost_hourly = (L_valid * R_valid).sum(axis=1)

        cap = float(self.env.capacity_threshold)
        over = np.where(A_valid.sum(axis=1) - cap > 1e-6)[0]
        if len(over) > 0:
            print(f"[NOTE] {len(over)} hour(s) exceed capacity after DR (soft check).")

        os.makedirs(self.results_dir, exist_ok=True)
        excel_path = os.path.join(self.results_dir, f'financial_report_day{target_day}_rho{rho}.xlsx')
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                pd.DataFrame(customer_metrics).to_excel(writer, sheet_name='Customer_Metrics', index=False)
                pd.DataFrame([sp_data]).to_excel(writer, sheet_name='Service_Provider', index=False)
                pd.DataFrame([par_data]).to_excel(writer, sheet_name='PAR_Metrics', index=False)

                df_hourly = pd.DataFrame({
                    'hour': np.arange(valid_rows),
                    'price_c_per_kWh': P_valid.squeeze(),
                    'baseline_total_kWh': load_no_dr,
                    'after_total_kWh': load_dr,
                    'reduction_total_kWh': R_valid.sum(axis=1),
                    'cost_cents': cost_hourly,
                    'revenue_cents': (P_valid.squeeze() * R_valid.sum(axis=1)),
                    'profit_cents': (P_valid.squeeze() * R_valid.sum(axis=1)) - cost_hourly
                })
                df_hourly.to_excel(writer, sheet_name='Hourly_Audit', index=False)

            print(f"Financial report saved to: {excel_path}")
        except Exception as e:
            print(f"[warn] {e}. Saving CSVs instead.")
            pd.DataFrame(customer_metrics).to_csv(os.path.join(self.results_dir, f'customer_metrics_day{target_day}_rho{rho}.csv'), index=False)
            pd.DataFrame([sp_data]).to_csv(os.path.join(self.results_dir, f'service_provider_day{target_day}_rho{rho}.csv'), index=False)
            pd.DataFrame([par_data]).to_csv(os.path.join(self.results_dir, f'par_metrics_day{target_day}_rho{rho}.csv'), index=False)
            print("Financial reports saved as CSV files.")


    def run_complete_analysis(self):
        print("=" * 60)
        print("Starting Complete Model Analysis")
        print("=" * 60)
        
        self.test_plots()
        
        self.financial_report()
        
        print("=" * 60)
        print("Analysis Complete!")
        print(f"All results saved to: {self.results_dir}")
        print("=" * 60)


def main():
    """Main function to run model testing. Usage: python test.py --model_path /path/to/model.pth"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained DQN model and generate reports')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file (.pth)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Initialize tester and run complete analysis
    tester = ModelTester(args.model_path)
    tester.run_complete_analysis()


if __name__ == "__main__":

    main()
