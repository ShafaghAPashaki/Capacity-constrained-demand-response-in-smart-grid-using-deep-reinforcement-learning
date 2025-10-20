import os
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from env import Environment
from agent.agent_dqn import DQNAgent
from utils.config_loader import load_config

# Font settings for plots (same as main.py)
plt.rcParams.update({'font.size': 24,'axes.labelsize': 24,'xtick.labelsize': 22, 
                     'ytick.labelsize': 22,'legend.fontsize': 20,'legend.title_fontsize': 22})
mpl.rcParams['hatch.linewidth'] = 3  # Make hatch lines thicker
BIN_SIZE = 2


class ModelTester:
    """
    Test and analyze trained DQN model
    This class handles visualization and financial reporting for trained models
    """
    def __init__(self, model_path, cfg_override=None):
        """Initialize ModelTester with trained model"""
        self.cfg = cfg_override or load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # Setup environment
        house_ids = self.cfg['environment']['house_ids']
        self.env = Environment(data_ids=house_ids, cfg_override=self.cfg)
        
        # Get state and action dimensions
        state = self.env.reset(mode='train')
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
        """Load trained model weights"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.agent.policy_net.eval()
        print(f"Model successfully loaded from: {model_path}")


    def plot_day_profiles(self, baselines_per_house, reductions_per_house, incentives_per_house, prices, days, house_ids, rho):
        """Plot daily energy profiles for individual houses"""
        # Color definitions 
        d_color = "#FFDD00"  # yellow for Demand
        c_color = "#919291"  # green for Consumption
        price_color = '#D62728'  # red for Wholesale Price
        inc_color = '#1F77B4'  # blue for Incentive Rate

        for day in days:
            B = baselines_per_house[day]
            R = reductions_per_house[day]
            I = incentives_per_house[day]
            P = prices[day]

            # Mask incentives where no reduction occurred
            masked_I = I.copy()
            for j in range(masked_I.shape[1]):
                masked_I[R[:, j] == 0, j] = 0.0

            hours = np.arange(1, len(P) + 1)
            xticks = np.arange(1, len(P) + 1, BIN_SIZE)

            for idx, hid in enumerate(house_ids):
                original_load = B[:, idx]
                consumption = original_load - R[:, idx]

                fig, ax1 = plt.subplots(figsize=(10, 6))

                # Plot demand and consumption areas
                ax1.fill_between(hours, original_load, 0, color=d_color, alpha=0.7, label='Demand', zorder=1)
                ax1.fill_between(hours, consumption, 0, color=c_color, alpha=0.2, label='_nolegend_', zorder=2)
                ax1.fill_between(hours, consumption, 0, facecolor='none', edgecolor=c_color, hatch='/', linewidth=2, label='Consumption', zorder=3)

                # Plot demand and consumption lines
                ax1.plot(hours, original_load, color=d_color, linewidth=2, zorder=4)
                ax1.plot(hours, consumption, color=c_color, linewidth=1.5, zorder=4)

                ax1.set_xlabel('Hour')
                ax1.set_ylabel('Energy (kWh)')
                ax1.set_xticks(xticks)
                ax1.set_xlim(1, len(P))
                ax1.set_ylim(0, original_load.max() * 1.1)
                ax1.grid(False)

                # Create secondary axis for prices and incentives
                ax2 = ax1.twinx()
                ax2.plot(hours, P, marker='o', color=price_color, linewidth=3, markersize=6, label='Wholesale Price', zorder=5)
                ax2.plot(hours, masked_I[:, idx], marker='s', color=inc_color, linewidth=3, markersize=6, label='Incentive Rate', zorder=5)  
                ax2.set_ylabel('Rate (¢/kWh)')
                ax2.set_ylim(0, max(P.max(), masked_I[:, idx].max()) * 1.2)

                # Combine legends from both axes
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=0.9)

                plt.tight_layout()

                # Save individual house profile
                filename = os.path.join(self.results_dir, f'energy_profile_house_{hid}_day_{day}_rho{rho}.png')
                plt.savefig(filename, dpi=300)
                plt.close(fig)
                
        print(f"Daily profiles plotted for {len(days)} days")


    def plot_aggregated_load(self, baselines_per_house, reductions_per_house, day, rho, output_path=None):
        """Plot aggregated load curve for a specific day"""
        B = baselines_per_house[day]
        R = reductions_per_house[day]
        agg_baseline = B.sum(axis=1)
        agg_consumption = agg_baseline - R.sum(axis=1)
        hours = np.arange(1, len(agg_baseline) + 1)
        xticks = np.arange(1, len(agg_baseline) + 1, BIN_SIZE)

        target_level = 8.5  # Capacity threshold

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hours, agg_baseline, marker='o', color='red', linewidth=2, markersize=6, label='Aggregated Original Load')
        ax.plot(hours, agg_consumption, marker='o', color='blue', linestyle='--', linewidth=2, markersize=6, label='Aggregated Load After Reduction')
        ax.hlines(target_level, xmin=hours.min(), xmax=hours.max(), colors='green', linestyles='--', linewidth=2, label=f'Capacity Threshold ({target_level} kW)')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Power (kW)')
        ax.set_xticks(xticks)
        ax.set_xlim(1, len(agg_baseline))
        ax.set_ylim(0, max(agg_baseline) * 1.1)
        ax.grid(False)
        ax.legend(loc='upper right')
        plt.tight_layout()

        if output_path:
            filename = output_path
        else:
            filename = os.path.join(self.results_dir, f'aggregated_load_day_{day}_rho{rho}.png')
        plt.savefig(filename, dpi=300)
        plt.close()
        
        print(f"Aggregated load plot saved: {filename}")


    def test_plots(self):
        """Generate test plots for specified test days. This runs the trained model on test data and creates visualizations"""
        rho = self.cfg['environment']['rho']
        house_ids = self.cfg['environment']['house_ids']
        
        # Get test days from config
        tr = self.cfg['training']
        start_day, end_day = map(int, tr['test_range'])   
        test_days = list(range(start_day, end_day + 1))
        july_days = [d for d in test_days if 204 <= d <= 210]  

        print(f"Generating test plots for {len(july_days)} days: {july_days}")

        baselines_dict, reductions_dict, incentives_dict, prices_dict = {}, {}, {}, {}

        # Generate data for test days using the trained model
        for day in july_days:
            state = self.env.reset(day)
            done = False
            hourly_baselines, hourly_reductions, hourly_incentives, hourly_prices = [], [], [], []

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.agent.policy_net(state_tensor).argmax().item()
                next_state, _, done, _ = self.env.step(action)

                h = self.env.curr_step - 1
                if h >= 0:
                    hourly_baselines.append(self.env.baseline_per_house[h].copy())
                    hourly_reductions.append(self.env.reductions[h].copy())
                    hourly_incentives.append(self.env.incentives[h].copy())
                    hourly_prices.append(self.env.prices[h])

                state = next_state

            # Store collected data
            baselines_dict[day] = np.array(hourly_baselines)
            reductions_dict[day] = np.array(hourly_reductions)
            incentives_dict[day] = np.array(hourly_incentives)
            prices_dict[day] = np.array(hourly_prices)

            # Plot aggregated load for each day
            self.plot_aggregated_load(baselines_dict, reductions_dict, day, rho)

        # Plot daily profiles for a specific target day
        target_day = 208
        if target_day in july_days:
            self.plot_day_profiles(baselines_dict, reductions_dict, incentives_dict, prices_dict, 
                                 days=[target_day], house_ids=house_ids, rho=rho)
            print(f"Detailed daily profiles created for day {target_day}")
        
        print("All test plots generated successfully")

    def financial_report(self):
        """Generate and save financial report ONLY for day 208."""

        rho = self.cfg['environment']['rho']
        house_ids = self.cfg['environment']['house_ids']

        target_day = 208
        july_days = [target_day]
        print(f"Generating financial report for day {target_day}")

        # Collect simulation data
        days, hours, N = len(july_days), self.env.max_steps, len(house_ids)

        R_j = np.zeros((days * hours, N), dtype=float)   # reductions kWh
        L_j = np.zeros((days * hours, N), dtype=float)   # incentives ¢/kWh
        B_j = np.zeros((days * hours, N), dtype=float)   # baselines kWh
        P_h = np.zeros((days * hours, 1), dtype=float)   # prices ¢/kWh

        valid_rows = 0

        for di, day in enumerate(july_days):  # only 208
            state = self.env.reset(day)
            for h in range(hours):
                idx = di * hours + h

                # Greedy eval action (no exploration)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.agent.policy_net(state_tensor).argmax().item()

                next_state, _, done, _ = self.env.step(action)

                # Collect hour-level arrays
                R_j[idx] = self.env.reductions[h] # kWh
                L_j[idx] = self.env.incentives[h] # ¢/kWh
                B_j[idx] = self.env.baseline_per_house[h] # kWh
                P_h[idx] = self.env.prices[h] # ¢/kWh

                valid_rows += 1
                state = next_state
                if done:
                    break

        # Use only valid rows
        R_valid = R_j[:valid_rows]
        L_valid = L_j[:valid_rows]
        B_valid = B_j[:valid_rows]
        P_valid = P_h[:valid_rows]

        sp_revenue_direct = float((P_valid * R_valid).sum()) # cent

        # Financial metrics (all in cent)
        mu_array = np.array(self.env.mu) # ¢/kWh^2 (per-customer)
        omega = self.env.omega # ¢/kWh (scalar or per-customer; here assumed scalar)

        income_vec = (L_valid * R_valid).sum(axis=0) # cent per CU
        dis_cost_vec = (0.5 * mu_array * (R_valid ** 2) + omega * R_valid).sum(axis=0) # cent per CU

        sp_revenue = float(sp_revenue_direct) # cent
        sp_payment = float(income_vec.sum()) # cent
        sp_profit  = float(sp_revenue - sp_payment) # cent

        # Customer metrics table
        customer_metrics = []
        for i, hid in enumerate(house_ids):
            total_red_i = float(R_valid[:, i].sum()) # kWh
            avg_inc_i = float(L_valid[:, i].mean()) # ¢/kWh
            income_i = float(income_vec[i]) # cent
            dis_i = float(dis_cost_vec[i]) # cent
            customer_metrics.append({'day': target_day, 'CUID': f'CU{hid}',
                'mu_cents_per_kWh2': float(mu_array[i]), 'omega_cents_per_kWh': float(omega),
                'Avg_inc_¢_per_kWh': avg_inc_i, 'Total_red_kWh': total_red_i,
                'Inc_income_cents': income_i, 'Dis_cost_cents': dis_i,
                'Profit_cents': income_i - dis_i})

        # Service provider metrics
        sp_data = {'day': target_day, 'SP_revenue_from_GO_cents': sp_revenue,
                    'SP_payment_to_CU_cents': sp_payment, 'SP_profit_cents': sp_profit}

        # PAR-like metrics on hourly energy series 
        load_no_dr = B_valid.sum(axis=1) # kWh per hour (aggregated)
        load_dr = (B_valid - R_valid).sum(axis=1)
        par_data = {'day': target_day, 'Peak_load_no_DR_kWh': float(load_no_dr.max()),
            'Peak_load_DR_kWh': float(load_dr.max()), 'Mean_load_no_DR_kWh': float(load_no_dr.mean()),
            'Mean_load_DR_kWh': float(load_dr.mean()), 'PAR_no_DR': float(load_no_dr.max() / max(load_no_dr.mean(), 1e-9)),
            'PAR_DR': float(load_dr.max() / max(load_dr.mean(), 1e-9)),}

        # Ensure output dir exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Save to Excel (fallback to CSV if openpyxl missing)
        excel_path = os.path.join(self.results_dir, f'financial_report_day{target_day}_rho{rho}.xlsx')
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                pd.DataFrame(customer_metrics).to_excel(writer, sheet_name='Customer_Metrics', index=False)
                pd.DataFrame([sp_data]).to_excel(writer, sheet_name='Service_Provider', index=False)
                pd.DataFrame([par_data]).to_excel(writer, sheet_name='PAR_Metrics', index=False)
            print(f"Financial report saved to: {excel_path}")
        except Exception as e:
            print(f"[warn] {e}. Saving CSVs instead.")
            pd.DataFrame(customer_metrics).to_csv(os.path.join(self.results_dir, f'customer_metrics_day{target_day}_rho{rho}.csv'), index=False)
            pd.DataFrame([sp_data]).to_csv(os.path.join(self.results_dir, f'service_provider_day{target_day}_rho{rho}.csv'), index=False)
            pd.DataFrame([par_data]).to_csv(os.path.join(self.results_dir, f'par_metrics_day{target_day}_rho{rho}.csv'), index=False)
            print("Financial reports saved as CSV files.")


    def run_complete_analysis(self):
        """Run complete analysis pipeline. Generates all test plots and financial reports"""
        print("=" * 60)
        print("Starting Complete Model Analysis")
        print("=" * 60)
        
        # Generate test plots
        self.test_plots()
        
        # Generate financial report
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