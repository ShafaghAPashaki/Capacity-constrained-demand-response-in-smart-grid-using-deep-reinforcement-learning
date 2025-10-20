import numpy as np
import itertools
import holidays
import torch
import random
from utils.load_demand import load_demand, load_baselines
from utils.load_price import load_price
from utils.config_loader import load_config

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Environment:
    # initialization
    def __init__(self, data_ids, cfg_override=None): 

        # load config and IDs
        self.cfg = cfg_override or load_config()
        self.data_ids = data_ids
        self.N = len(self.data_ids)

        # initialize episode counter
        self.episode = 0

        # initialize reductions
        self.reductions = None

        # load data
        self.df_demand = load_demand(data_path=self.cfg['general']['load_file'], house_ids=self.data_ids)
        self.df_price = load_price(data_path=self.cfg['general']['price_file']) 
        self.all_baselines = load_baselines(self.df_demand)

        # training parameters
        self.train_ranges = self.cfg['training']['train_ranges']  
        self.val_range = self.cfg['training']['val_range']        
        self.test_range = self.cfg['training']['test_range']      
        self.max_steps = self.cfg['training']['time_steps_train']

        # environment parameters
        self.weight = self.cfg['environment']['customer_reward_weight']
        self.rho = self.cfg['environment']['rho']
        self.mu = self.cfg['environment']['mu']
        self.omega = self.cfg['environment']['omega']
        self.capacity_threshold = self.cfg['environment']['capacity_threshold']
        self.max_reduction_fraction = self.cfg['environment']['max_reduction_fraction']
        self.lam_frac_min = self.cfg['environment']['lam_frac_min']
        self.lam_frac_max = self.cfg['environment']['lam_frac_max']

        # discrete action space
        self.discrete_actions = np.linspace(0.0, 1.0, 4)
        self.all_actions = np.array(list(itertools.product(self.discrete_actions, repeat=self.N))) #4^3 = 64 actions for N=3
        self.num_actions = len(self.all_actions)

        # US holidays (2018)
        self.us_holidays = holidays.US(years=[2018])

    def reset(self, day=None, mode = 'train'):
        # day selection based on mode
        if day is None:
            if mode == 'train':
                range_idx = np.random.randint(0, len(self.train_ranges))
                start, end = self.train_ranges[range_idx]
                self.day = np.random.randint(start, end + 1)
            elif mode == 'val':
                self.day = np.random.randint(self.val_range[0], self.val_range[1] + 1)
            elif mode == 'test':
                self.day = np.random.randint(self.test_range[0], self.test_range[1] + 1)
        else:
            self.day = day 

        # state initialization
        self.curr_step = 0 # resets the timestep
        self.episode += 1  # flags the episode as ongoing
        self.done = False  # tracks how many episodes have occurred

        # load price data for the selected day
        self.price_df = self.df_price[self.df_price.index.dayofyear == self.day].copy()
        self.prices = self.price_df['price'].values
        self.hours = self.price_df.index.hour.values

        # compute weekend/holiday flags
        timestamp = self.price_df.index[0]
        self.is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
        self.is_holiday = 1.0 if timestamp.date() in self.us_holidays else 0.0

        # load baseline (original demand) per house, (max_steps, num_houses)
        baselines_day = self.all_baselines[self.all_baselines['timestamp'].dt.dayofyear == self.day]
        baseline_wide = baselines_day.pivot_table(index='timestamp', columns='house_id',
                                                   values='baseline_demand', fill_value=0.0)
        baseline_wide = baseline_wide.reindex(self.price_df.index).ffill().fillna(0.0)
        self.baseline_per_house = baseline_wide[self.data_ids].values
        self.total_baseline = self.baseline_per_house.sum(axis=1)

        # compute total baseline demand per hour
        self.total_baseline = self.baseline_per_house.sum(axis=1)

        # compute min/max incentive bounds
        self.lam_min, self.lam_max = self.get_lam_bounds()

        # initialize incentives and rewards arrays
        self.incentives = np.zeros((self.max_steps,len(self.data_ids)))
        self.rewards_customers = np.zeros_like(self.incentives)
        self.rewards_service_provider = np.zeros(self.max_steps)
        self.rewards_total = np.zeros(self.max_steps)

        # initialize reductions array
        self.reductions = np.zeros((self.max_steps, len(self.data_ids)))

        # initialize state for episode
        state = self.get_state()
        self.state_dim = state.shape[0]
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

        # Store reductions after computing rewards
        _, _, _, _, delta_E = self.get_step_data()
        self.reductions[self.curr_step] = delta_E

        # moves forward by one step and check if done
        self.curr_step += 1
        done = (self.curr_step >= self.max_steps)

        # return results
        if done:
            observation = np.zeros(self.state_dim, dtype=np.float32) 
        else:
            observation = self.get_state()  

        return observation, reward, done, {}

    def get_state(self):
        # get current step data
        h, baselines, _, price, delta_E = self.get_step_data()
        upper_clip = 80.0
        price_for_state = np.log1p(np.clip(price, 0.0, upper_clip))

        # compute critical flag and reduction requirements
        total_baseline = baselines.sum()

        # compute holiday and weekend flags
        Holiday_flag = self.is_holiday
        Weekend_flag = self.is_weekend

        # create state vector
        inputs = [delta_E, baselines, [price_for_state], [h / (self.max_steps - 1)], [total_baseline], [Holiday_flag], [Weekend_flag]]
        return np.concatenate([np.atleast_1d(p) for p in inputs]).astype(float)

    def apply_incentives(self, raw_action):
        h = self.curr_step
        if self.total_baseline[h] <= self.capacity_threshold:
            incentives = np.zeros(self.N)  # zero incentives below threshold
        else:
            # only apply incentives when above threshold
            incentives = self.lam_min[h] + raw_action * (self.lam_max[h] - self.lam_min[h])
        self.incentives[self.curr_step] = incentives
        return incentives

    def compute_service_provider_reward(self):
        h, _, lambdas, price, delta_E = self.get_step_data()

        # market revenue and cost
        revenue = price * delta_E.sum()
        cost = np.dot(lambdas, delta_E)

        #Sepehr's suggestion.
        #discomfort = np.zeros(len(self.data_ids))
        #for i in range(len(self.data_ids)):
        #    delta_i = delta_E[i]
        #    phi = self.mu[i] * delta_i**2 / 2 + self.omega * delta_i
        #    discomfort[i] = lambdas[i] * phi  
        #cost = discomfort.sum()

        self.rewards_service_provider[h] = revenue - cost
        return self.rewards_service_provider[h]

    def compute_customers_reward(self):
        h, _, lambdas, _, delta_E = self.get_step_data()
        rewards = np.zeros(len(self.data_ids))
        for i in range(len(self.data_ids)):
            delta_i = delta_E[i]  
            discomfort = self.mu[i] * delta_i**2 / 2 + self.omega * delta_i
            benefit = self.rho * lambdas[i] * delta_i
            #benefit = self.rho * lambdas[i] * discomfort
            rewards[i] = benefit - (1 - self.rho) * discomfort
        self.rewards_customers[h] = rewards
        return rewards.sum()

    def compute_total_reward(self):
        h = self.curr_step
        sp_reward = self.rewards_service_provider[h]
        cu_reward = self.rewards_customers[h].sum()
        total_reward = self.weight * sp_reward + (1 - self.weight) * cu_reward 
       
        # Grid Stability baseline
        stable = 1.0 if self.total_baseline[h] <= self.capacity_threshold else 0.0
        b = 0.005  # bonus for stability
        total_reward += b * stable
        self.rewards_total[h] = total_reward
        return total_reward
    
    def compute_total_demand(self, step=None):
        if step is None:
            return self.baseline_per_house.sum(axis=1)
        return float(self.baseline_per_house[step].sum())

    def compute_total_reduction(self, step=None):
        if step is None:
            return np.array([self.compute_total_reduction(h) for h in range(self.max_steps)])
        _, _, _, _, delta_E = self.get_step_data(step)  
        return float(delta_E.sum())
    
    def compute_total_consumption(self, step=None):
        if step is None:
            return self.baseline_per_house.sum(axis=1) - self.compute_total_reduction()
        baseline_total = float(self.baseline_per_house[step].sum())
        reduction = self.compute_total_reduction(step) 
        return baseline_total - reduction

    def get_lam_bounds(self):
        # per-hour dynamic bounds
        lam_min = self.lam_frac_min * self.prices
        lam_max = self.lam_frac_max * self.prices
        
        # ensure min <= max
        np.putmask(lam_min, lam_min > lam_max, 0.95 * lam_max)
        return lam_min, lam_max

    def compute_delta_E(self, baselines, incentives, h):
        # amount we need to cut to hit the static threshold
        total_baseline = baselines.sum()
        needed = max(0.0, total_baseline - self.capacity_threshold)
        
        if needed <= 0:
            return np.zeros(self.N)  # no reduction needed
        
        # compute each house's max reduction cap 
        max_red = self.max_reduction_fraction * baselines
        
        # build net cost = - incentive signal
        net_cost = -incentives
        
        # greedy continuous knapsack: allocate reductions where net_cost is lowest
        delta = np.zeros(self.N)
        rem = needed
        
        for i in np.argsort(net_cost):
            if rem <= 0:
                break
            avail = max_red[i]
            take = min(avail, rem)
            if take > 0:
                delta[i] = take
                rem -= take
        
        # clip to each house's cap
        return np.minimum(delta, max_red)

    def get_step_data(self, step=None):
        h = step if step is not None else self.curr_step
        baselines = self.baseline_per_house[h]
        lambdas = self.incentives[h]
        price = self.prices[h]
        delta_E = self.compute_delta_E(baselines, lambdas, h)
        return h, baselines, lambdas, price, delta_E