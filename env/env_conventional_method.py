import numpy as np
import pandas as pd
import itertools
import holidays
import joblib
import torch
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from utils.load_demand import load_demand, load_day, load_baselines
from utils.load_price  import load_price
from utils.config_loader import load_config
from utils.predict_load import GRUForecaster as LoadForecaster
from utils.predict_price import GRUForecaster as PriceForecaster
 
class Environment:
    # initialization
    def __init__(self, data_ids, cfg_override=None): 
        # load config and IDs
        self.cfg = load_config()
        if cfg_override is not None:
           cfg = cfg_override
        self.cfg = cfg
        self.data_ids = data_ids
        self.N = len(self.data_ids)
        # initialize episode counter
        self.episode = 0
        # Initialize reductions
        self.reductions = None
        # load data
        self.df_demand = load_demand(data_path=self.cfg['general']['load_file'], house_ids=self.data_ids)
        self.df_price = load_price(data_path=self.cfg['general']['price_file']) \
                            .set_index('dt')['price']
        self.df_price.index = pd.to_datetime(self.df_price.index)
        self.df_demand['dt'] = pd.to_datetime(self.df_demand['dt'])
        self.features_load = (['month', 'dayofweek', 'hour', 'is_weekend', 'is_holiday'] + [f'lag{lag}' for lag in [1,2,3,24,25,26,48,49,50]])
        self.features_price = ['month','dayofweek','hour','is_holiday','lag1','lag2','lag3']
        self.window = self.cfg['forecast']['window_size']
        self.max_steps = self.cfg['training']['time_steps_train'] 
        # set up training/testing date ranges
        self.max_steps = self.cfg['training']['time_steps_train']
        self.train_start = self.cfg['training']['training_start_day']
        self.test_start = self.cfg['training']['testing_start_day']
        self.test_end = self.cfg['training']['testing_end_day']
        self.train_end = self.cfg['training']['training_end_day']
        # environment parameters
        self.weight = self.cfg['environment']['customer_reward_weight']
        self.rho = self.cfg['environment']['rho']
        self.mu = self.cfg['environment']['mu']
        self.omega = self.cfg['environment']['omega']
        self.lam_frac_min = self.cfg['environment']['lam_frac_min']
        self.lam_frac_max = self.cfg['environment']['lam_frac_max']
        # discrete action space
        self.discrete_actions = np.linspace(0, 1, 5)
        self.all_actions = np.array(list(itertools.product(self.discrete_actions, repeat=self.N)))
        self.num_actions = len(self.all_actions)
        # US holidays (2018)
        self.us_holidays = holidays.US(years=[2018])
        self.load_scalers = {}
        self.load_models  = {}
        for hid in self.data_ids:
            scaler_path = f"scaler_house{hid}.bin"
            model_path  = f"model_house{hid}.pt"
            self.load_scalers[hid] = joblib.load(scaler_path)
            m = LoadForecaster(in_size = 14, hidden_size= self.cfg['forecast']['hidden_size'], num_layers = self.cfg['forecast']['num_layers'], dropout = self.cfg['forecast']['dropout'])
            m.load_state_dict(torch.load(model_path, map_location='cpu'))
            m.eval()
            self.load_models[hid] = m
        self.price_scaler = joblib.load("scaler_price.bin")
        self.price_model  = PriceForecaster(in_size = 14, hidden_size= self.cfg['forecast']['hidden_size'], num_layers = self.cfg['forecast']['num_layers'], dropout = self.cfg['forecast']['dropout'])
        self.price_model.load_state_dict(torch.load("price_onestep.pth", map_location='cpu'))
        self.price_model.eval()

    def reset(self, day=None):
        # day Selection
        if day is None:
            day_range = [(self.train_start, self.test_start), (self.test_end, self.train_end)][np.random.randint(0, 2)]
            self.day = np.random.randint(*day_range)
        else:
            self.day = day 
        # state initialization
        self.curr_step = 0 # resets the timestep
        self.episode += 1  # flags the episode as ongoing
        self.done = False  # tracks how many episodes have occurred

        day_price_df = load_price(data_path=self.cfg['general']['price_file'])
        day_price_df.loc[:, 'dt'] = pd.to_datetime(day_price_df['dt'])
        day_price_df.set_index('dt', inplace=True)

        day_price_df = day_price_df.dropna(subset=['price'])
        day_price_df.loc[:, 'price'] = day_price_df['price'].clip(lower=0)
        day_price_df.loc[:, 'price_log'] = np.log1p(day_price_df['price'])

        day_price_df.loc[:, 'month']      = day_price_df.index.month
        day_price_df.loc[:, 'dayofweek']  = day_price_df.index.dayofweek + 1
        day_price_df.loc[:, 'hour']       = day_price_df.index.hour
        day_price_df.loc[:, 'is_weekend'] = (day_price_df.index.weekday >= 5).astype(int)
        day_price_df.loc[:, 'is_holiday'] = (day_price_df.index.to_series().dt.date.isin(self.us_holidays).astype(int))

        price_lags = [1,2,3,24,25,26,48,49,50]
        for lag in price_lags:
            day_price_df.loc[:, f'lag{lag}'] = day_price_df['price_log'].shift(lag)
        day_price_df.loc[:, [f'lag{lag}' for lag in price_lags]] = \
            day_price_df[[f'lag{lag}' for lag in price_lags]].fillna(0)

        self.price_df = day_price_df.loc[day_price_df.index.dayofyear == self.day].copy()
        self.price_df = self.price_df.iloc[:self.max_steps]

        self.prices  = self.price_df['price_log'].values
        self.hours  = self.price_df.index.hour.values
        self.features_price = (['month','dayofweek','hour','is_weekend','is_holiday']+ [f'lag{lag}' for lag in price_lags])

        self.day_df = load_day(self.df_demand, self.day, self.max_steps).copy()

        self.day_df.loc[:, 'month']      = self.day_df['dt'].dt.month
        self.day_df.loc[:, 'dayofweek']  = self.day_df['dt'].dt.dayofweek + 1
        self.day_df.loc[:, 'hour']       = self.day_df['dt'].dt.hour
        self.day_df.loc[:, 'is_weekend'] = (self.day_df['dt'].dt.weekday >= 5).astype(int)
        self.day_df.loc[:, 'is_holiday'] = (self.day_df.loc[:, 'dt'].dt.date.isin(self.us_holidays).astype(int))

        id_col = 'house_id' if 'house_id' in self.day_df.columns else 'dataid'
        lags = [1,2,3,24,25,26,48,49,50]
        for lag in lags:self.day_df.loc[:, f'lag{lag}'] = (self.day_df.sort_values([id_col, 'dt']).groupby(id_col)['total'].shift(lag))
        self.day_df.loc[:, [f'lag{lag}' for lag in lags]] = \
            self.day_df[[f'lag{lag}' for lag in lags]].fillna(0)

        # compute weekend/holiday flags
        date = self.price_df.index[0].date()
        self.is_weekend = 1.0 if date.weekday() >= 5 else 0.0
        self.is_holiday = 1.0 if date in self.us_holidays else 0.0
        # load baseline per house, (max_steps, num_houses)
        all_baselines = load_baselines(self.df_demand)
        baselines_day = all_baselines[all_baselines['timestamp'].dt.dayofyear==self.day]
        self.baseline_per_house = np.zeros((self.max_steps, len(self.data_ids))) 
        for i,hid in enumerate(self.data_ids):
            bdf = baselines_day[baselines_day['house_id'] == hid].set_index('timestamp')
            self.baseline_per_house[:,i] = bdf.reindex(self.price_df.index)['baseline_demand'].values[:self.max_steps]
        self.total_baseline = self.baseline_per_house.sum(axis=1)  
        # Compute min/max incentive bounds
        self.lam_min, self.lam_max = self.get_lam_bounds()
        # initialize outputs
        self.incentives = np.zeros((self.max_steps,len(self.data_ids)))
        self.rewards_customers = np.zeros_like(self.incentives)
        self.rewards_service_provider = np.zeros(self.max_steps)
        self.rewards_total = np.zeros(self.max_steps)
        # Initialize reductions array
        self.reductions = np.zeros((self.max_steps, len(self.data_ids)))
        # initialize state for episode
        state = self.get_state()
        return state

    def step(self, action):
        # Convert discrete action index to continuous action vector
        raw_action = self.all_actions[action]
        # apply agent action
        self.apply_incentives(raw_action)
        # compute rewards
        self.compute_service_provider_reward()
        self.compute_customers_reward()
        reward = self.compute_total_reward()
        # get next state
        observation = self.get_state()
        # Store reductions after computing rewards
        _, _, _, _, _, delta_E = self.get_step_data()
        self.reductions[self.curr_step] = delta_E
        # moves forward by one step and check if done
        self.curr_step += 1
        done = (self.curr_step >= self.max_steps)
        # return results
        return observation, reward, done, {}

    def get_state(self):
        h, baselines, _, elasticity, price, delta_E = self.get_step_data()
        # assemble and return the full state vector
        core = np.empty(2*self.N + 5, dtype=float)
        core[:self.N] = delta_E          
        core[self.N:2*self.N] = baselines             
        core[2*self.N] = price                  
        core[2*self.N+1] = elasticity                      
        core[2*self.N+2] = h / (self.max_steps - 1) 
        core[2*self.N+3] = self.is_weekend
        core[2*self.N+4] = self.is_holiday 
        pred_loads = []
        for hid in self.data_ids:
            # prepare sequence with padding
            seq_df = self.day_df[self.day_df['dataid']==hid]
            raw_seq = seq_df[self.features_load]
            start = h - self.window
            if start < 0:
                pad = np.zeros((-start, len(self.features_load)))
                avail = raw_seq.iloc[0:h].values
                seq = np.vstack([pad, avail])
            else:
                seq = raw_seq.iloc[start:h].values

            # convert to DataFrame for scaler
            df_seq = pd.DataFrame(seq, columns=self.features_load)
            scaled = self.load_scalers[hid].transform(df_seq)
            Xt = torch.tensor(scaled[np.newaxis, ...], dtype=torch.float32)
            with torch.no_grad():
                pred_log = self.load_models[hid](Xt).cpu().numpy().squeeze()
            p = np.expm1(pred_log)
            pred_loads.append(p)

        raw_price = self.price_df[self.features_price]
        start = h - self.window
        if start < 0:
            pad = np.zeros((-start, len(self.features_price)))
            avail = raw_price.iloc[0:h].values
            price_seq = np.vstack([pad, avail])
        else:
            price_seq = raw_price.iloc[start:h].values

        # DataFrame conversion for price scaler
        df_price_seq = pd.DataFrame(price_seq, columns=self.features_price)
        scaled_price = self.price_scaler.transform(df_price_seq)
        Xt_price = torch.tensor(scaled_price[np.newaxis, ...], dtype=torch.float32)
        with torch.no_grad():
            pred_log_p = self.price_model(Xt_price).cpu().numpy().squeeze()
        pp = np.expm1(pred_log_p)
        pred_prices = [pp]

        return np.concatenate([core, np.array(pred_loads), np.array(pred_prices)])
    
    def apply_incentives(self, raw_action):
        h = self.curr_step

        lam_min_h = self.lam_min[h]
        lam_max_h = self.lam_max[h]

        inc = lam_min_h + raw_action * (lam_max_h - lam_min_h)

        self.incentives[h] = inc

    def compute_service_provider_reward(self):
        h, _, lambdas, _, price, delta_E = self.get_step_data()
        # market revenue and cost
        revenue = price * delta_E.sum()
        cost = np.dot(lambdas, delta_E)
        self.rewards_service_provider[h] = revenue - cost
        return self.rewards_service_provider[h]

    def compute_customers_reward(self):
        h, _, lambdas, _, _, delta_E = self.get_step_data()
        rewards = np.zeros(len(self.data_ids))
        for i in range(len(self.data_ids)):
            delta_i = delta_E[i]  
            benefit = self.rho * lambdas[i] * delta_i
            #discomfort = self.mu[i] * delta_i**2 + self.omega * delta_i
            discomfort = self.mu[i] * delta_i**2 / 2 + self.omega * delta_i
            rewards[i] = benefit - (1 - self.rho) * discomfort
        self.rewards_customers[h] = rewards
        return rewards.sum()

    def compute_total_reward(self):
        h = self.curr_step
        sp_reward = self.rewards_service_provider[h]
        cu_reward = self.rewards_customers[h].sum()
        total_reward = self.weight * sp_reward + (1 - self.weight) * cu_reward
        self.rewards_total[h] = total_reward
        return total_reward

    def compute_total_demand(self, step=None):
        if step is None:
            return self.baseline_per_house.sum(axis=1)
        return float(self.baseline_per_house[step].sum())

    def compute_total_reduction(self, step=None):
        if step is None:
            return np.array([self.compute_total_reduction(h) for h in range(self.max_steps)])
        _, _, _, _, _, delta_E = self.get_step_data(step)  
        return float(delta_E.sum())
    
    def compute_total_consumption(self, step=None):
        if step is None:
            return self.baseline_per_house.sum(axis=1) - self.compute_total_reduction()
        baseline_total = float(self.baseline_per_house[step].sum())
        reduction = self.compute_total_reduction(step) 
        return baseline_total - reduction
    
    def get_elasticity(self, hour):
        env = self.cfg['environment']
        if hour in env['off_peak_hours']:
            return env['elasticity']['off_peak']
        if hour in env['mid_peak_hours']:
            return env['elasticity']['mid_peak']
        if hour in env['on_peak_hours']:
            return env['elasticity']['on_peak']
        raise ValueError(f"Hour {hour} not defined in any peak period")

    def get_lam_bounds(self):
        lam_min = 0.3 * self.prices    
        lam_max = 0.8 * self.prices    
        return lam_min, lam_max

    def compute_delta_E(self, baselines, incentives, elasticity):
        h = self.curr_step

        lam_min_h = self.lam_min[h]
        lam_max_h = self.lam_max[h]

        denom = max(lam_max_h - lam_min_h, 1e-6)

        raw_delta = baselines * elasticity * (incentives - lam_min_h) / denom
        delta_E = np.clip(raw_delta, 0, 0.3 * baselines)

        return delta_E
    
    def get_step_data(self, step=None):
        if step is None:
            h = self.curr_step
        else:
            h = step
        baselines = self.baseline_per_house[h]
        lambdas = self.incentives[h]
        elasticity = self.get_elasticity(self.hours[h])
        price = self.prices[h]
        delta_E = self.compute_delta_E(baselines, lambdas, elasticity)
        return h, baselines, lambdas, elasticity, price, delta_E
    
    