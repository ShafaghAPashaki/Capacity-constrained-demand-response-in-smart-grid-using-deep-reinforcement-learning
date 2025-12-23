import numpy as np
import itertools
import holidays
import torch
import random

from utils.load_demand import load_demand, load_baselines, get_device_demands, load_device_demands
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

        # disable power curtailment flag
        self.disable_pc = bool(self.cfg['environment'].get('DISABLE_PC', False))

        # initialize episode counter
        self.episode = 0

        # load data
        self.df_demand = load_demand(data_path=self.cfg['general']['load_file'], house_ids=self.data_ids)
        self.df_device_demands = load_device_demands(data_path=self.cfg['general']['load_file'], house_ids=self.data_ids)
        self.df_price = load_price(data_path=self.cfg['general']['price_file']) 
        self.all_baselines = load_baselines(self.df_demand)

        # training parameters
        self.train_ranges = self.cfg['training']['train_ranges']  
        self.val_range = self.cfg['training']['val_range']        
        self.test_range = self.cfg['training']['test_range']      
        self.max_steps = self.cfg['training']['time_steps_train']

        # environment parameters
        self.rho = self.cfg['environment']['rho']
        self.capacity_threshold = self.cfg['environment']['capacity_threshold']

        # discrete action space
        self.discrete_actions = np.linspace(0.0, 1.0, 5)
        self.all_actions = np.array(list(itertools.product(self.discrete_actions, repeat=self.N))) #5^3 = 125 actions for N=3
        self.num_actions = len(self.all_actions)

        # US holidays (2018)
        self.us_holidays = holidays.US(years=[2018])

        # device parameters 
        self.DEVICES = self.cfg['environment']['DEVICES']
        self.num_devices = len(self.DEVICES)
        self.idx_car = self.DEVICES.index("car") if "car" in self.DEVICES else None

        # device type masks: PC, TS-I, TS-NI
        dni = np.array(self.cfg['environment']['DEVICE_NON_INTERRUPTIBLE'])

        # base masks from configuration
        self.PC_MASK = (dni == 0)
        self.TS_NI_MASK = (dni == 1)

        # car
        self.TS_I_MASK = np.zeros(self.num_devices, dtype=bool)
        if self.idx_car is not None:
            self.TS_I_MASK[self.idx_car] = True # Mark car as TSI
            self.PC_MASK[self.idx_car] = False # Remove car from PC
            self.TS_NI_MASK[self.idx_car] = False # Remove car from TSNI

        # combined mask for all TS devices
        self.TS_MASK = self.TS_I_MASK | self.TS_NI_MASK

        if self.disable_pc:
            self.PC_MASK[:] = False

        # deadlines
        self.ts_deadline_hour = self.cfg['environment']['ts_deadline_hour']

        # power rate (PC)
        self.POWER_RATE = np.array(self.cfg['environment']['POWER_RATE'])
        if 0.0 not in self.POWER_RATE:
            self.POWER_RATE = [0.0] + list(self.POWER_RATE)
        self.POWER_RATE = np.array(sorted(set([r for r in self.POWER_RATE if 0.0 <= r <= 1.0])), dtype=float)

        # dissatisfaction coefficients
        self.heterogeneous = True
        self.coeff_base = np.array(self.cfg['environment']['DISSATISFACTION_COEFFICIENTS'])
        self.coeff_std = np.array(self.cfg['environment']['DISSATISFACTION_COEFFICIENTS_STD'])
        self.coeff_min = np.array(self.cfg['environment']['DISSATISFACTION_COEFFICIENTS_MIN'])
        self.dissatisfaction_coefficients = np.tile(self.coeff_base, (self.N, 1)).astype(float)


    def reset(self, day=None, mode='train'):
        # day selection
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
                self.day = self.train_ranges[0][0]
        else:
            self.day = day

        # episode counters
        self.curr_step = 0
        self.episode += 1
        self.done = False

        # price & time
        self.price_df = self.df_price[self.df_price.index.dayofyear == self.day].copy()
        self.prices = self.price_df['price'].values
        self.hours = self.price_df.index.hour.values
        self.T = len(self.prices)

        # weekend / holiday flags (per-episode)
        ts0 = self.price_df.index[0]
        self.is_weekend = 1.0 if ts0.weekday() >= 5 else 0.0
        self.is_holiday = 1.0 if ts0.date() in self.us_holidays else 0.0

        # baselines (per-house)
        day_mask = (self.all_baselines['timestamp'].dt.dayofyear == self.day)
        baselines_day = self.all_baselines[day_mask]
        baseline_wide = baselines_day.pivot_table(index='timestamp', columns='house_id', values='baseline_demand', fill_value=0.0)
        baseline_wide = baseline_wide.reindex(self.price_df.index).ffill().fillna(0.0)
        self.baseline_per_house = baseline_wide[self.data_ids].values
        self.total_baseline = self.baseline_per_house.sum(axis=1)

        # device demands
        self.device_demands = np.zeros((self.T, self.N, self.num_devices), dtype=float)
        for h in range(self.T):
            self.device_demands[h] = get_device_demands(self.df_device_demands, self.data_ids, self.day, h)

        # mark TS-NI devices that are already running from previous day
        self.ts_carry_over = np.zeros((self.N, self.num_devices), dtype=bool)
        if self.day > 1:
            # assume same number of hours per day
            last_h_prev = self.T - 1
            prev_last = get_device_demands(self.df_device_demands, self.data_ids, self.day - 1, last_h_prev)

            # carry-over job if device was ON at last hour of previous day
            self.ts_carry_over = (prev_last > 1e-9) & self.TS_NI_MASK[None, :]

        # before device logs
        self.device_before = self.device_demands.copy()

        raw_ctrl_sum = self.device_demands.sum(axis=2)           

        # non-shiftable baseline per house per hour
        self.nonshift_baseline_per_house = np.clip(self.baseline_per_house - raw_ctrl_sum, 0.0, None)  

        # incentives / rewards / buffers
        self.lam_min, self.lam_max = self.get_lam_bounds()
        self.incentives = np.zeros((self.T, self.N), dtype=float)
        self.rewards_customers = np.zeros((self.T, self.N), dtype=float)
        self.rewards_service_provider = np.zeros(self.T, dtype=float)
        self.rewards_total = np.zeros(self.T, dtype=float)
        self.delta_E = np.zeros((self.T, self.N), dtype=float)
        self.discomforts = np.zeros((self.T, self.N), dtype=float)

        # aggregated device logs
        self.device_after_agg = np.zeros((self.T, self.num_devices), dtype=float)
        self.device_baseline_raw = np.zeros((self.T, self.num_devices), dtype=float)
        self.ts_exec_agg = np.zeros((self.T, self.num_devices), dtype=float)

        # per-house per-device logs (for plots)
        self.device_after = np.zeros((self.T, self.N, self.num_devices), dtype=float)
        self.device_future_inj = np.zeros((self.T, self.N, self.num_devices), dtype=float)
        self._removed_mask = np.zeros((self.T, self.N, self.num_devices), dtype=bool)
        self.after_total_per_house  = np.zeros((self.T, self.N), dtype=float)   
        self.before_total_per_house = np.zeros((self.T, self.N), dtype=float)   
        self.reductions_by_device = np.zeros((self.T, self.num_devices), dtype=float)     

        # capacity planning buffers
        self.planned_load = np.zeros(self.T, dtype=float)

        # TS scheduling buffers
        self.ts_backlog = np.zeros((self.N, self.num_devices), dtype=float) # energy waiting
        self.ts_due_time = np.full((self.N, self.num_devices), -1, dtype=int) # due time

        # per-episode sampling of dissatisfaction coefficients
        if getattr(self, "heterogeneous", True) and mode == 'train':
            rng = np.random.default_rng(getattr(self, "seed", 0))  
            sampled = rng.normal(loc=self.coeff_base, scale=self.coeff_std, size=(self.N, self.num_devices))
            sampled = np.maximum(self.coeff_min, sampled)
            self.dissatisfaction_coefficients = sampled.astype(float)
        else:
            self.dissatisfaction_coefficients = np.tile(self.coeff_base, (self.N, 1)).astype(float)

        # init state
        state = self.get_state()
        self.state_dim = state.shape[0]
        return state


    def step(self, action):
        h = self.curr_step

        # incentives
        raw_action = self.all_actions[action]
        incentives = self.apply_incentives(raw_action) # stores self.incentives[h]

        # compute and store delta_E exactly once
        delta_E = self.compute_delta_E(incentives, h)
        self.delta_E[h] = delta_E

        # rewards 
        sp_r = self.compute_service_provider_reward()
        cu_r = self.compute_customers_reward()
        reward = self.compute_total_reward()

        # advance
        self.curr_step += 1
        done = (self.curr_step >= self.max_steps)

        # next obs
        observation = np.zeros(self.state_dim, dtype=np.float32) if done else self.get_state()
        return observation, reward, done, {}

    
    def get_state(self):
        # get current step data
        h, baselines, _, price, _= self.get_step_data()

        # compute log-scaled price for state representation
        upper_clip = 80.0
        price_for_state = np.log1p(np.clip(price, 0.0, upper_clip))

        # compute holiday and weekend flags
        Holiday_flag = self.is_holiday
        Weekend_flag = self.is_weekend

        # compute needed reduction
        total_baseline = baselines.sum()
        needed = max(0.0, total_baseline - self.capacity_threshold)
        need_flag = 1.0 if needed > 0 else 0.0
        needed_normalized = needed / self.capacity_threshold if self.capacity_threshold > 0 else 0.0

        # create state vector
        inputs = [baselines, [price_for_state], [h / (self.max_steps - 1)], [total_baseline], [Holiday_flag], [Weekend_flag], [need_flag], [needed_normalized]]
        return np.concatenate([np.atleast_1d(p) for p in inputs]).astype(float)


    def apply_incentives(self, raw_action):
        h = self.curr_step
        incentives = self.lam_min[h] + raw_action * (self.lam_max[h] - self.lam_min[h])
        self.incentives[self.curr_step] = incentives
        return incentives


    def compute_service_provider_reward(self):
        h, _, lambdas, price, delta_E = self.get_step_data()

        achieved = float(delta_E.sum())
        revenue = float(price) * achieved
        cost = float(np.dot(lambdas, delta_E))

        sp_reward = revenue - cost
        self.rewards_service_provider[h] = sp_reward
        return sp_reward

    def compute_customers_reward(self):
        h, _, lambdas, _, delta_E = self.get_step_data()

        rewards = np.zeros(len(self.data_ids))
        for i in range(len(self.data_ids)):
            delta_i = delta_E[i]
            discomfort = float(self.discomforts[h, i])
            benefit = self.rho * lambdas[i] * delta_i
            rewards[i] = benefit - (1 - self.rho) * discomfort

        self.rewards_customers[h] = rewards
        return rewards.sum()


    def compute_total_reward(self):
        h = self.curr_step

        sp_reward = self.rewards_service_provider[h]
        cu_reward = self.rewards_customers[h].sum()
        total_reward = sp_reward + cu_reward

        # additional penalties / bonuses
        lambdas = self.incentives[h]
        total_baseline = self.total_baseline[h]
        needed_reduction = max(0.0, total_baseline - self.capacity_threshold)
        actual_reduction = self.compute_total_reduction(h)
        sent_any_incentive = np.any(lambdas > 0.0)
        no_incentive = np.all(lambdas == 0.0)

        # penalties / bonuses
        # no reduction needed
        if needed_reduction <= 0.0:
            # no incentive was sent = correct behavior
            if no_incentive:
                total_reward += 5.0
            # you sent an incentive even though it wasn’t needed
            else:
                unnecessary_penalty = 5.0 * float(np.sum(lambdas))
                total_reward -= unnecessary_penalty

        # reduction needed
        else:
            # penalty for not reaching the target
            missing_reduction = max(0.0, needed_reduction - actual_reduction)
            reduction_penalty = 15.0 * missing_reduction

            # if no incentive was sent and reduction was needed → higher penalty
            if no_incentive:
                reduction_penalty *= 2 

            total_reward -= reduction_penalty

            # penalty for over-reduction
            over_reduction = max(0.0, actual_reduction - needed_reduction)
            total_reward -= 0.5 * over_reduction

        self.rewards_total[h] = total_reward
        return total_reward
    

    def compute_total_demand(self, step=None):
        if step is None:
            return self.baseline_per_house.sum(axis=1)
        return float(self.baseline_per_house[step].sum())


    def compute_total_reduction(self, step=None):
        if step is None:
            return self.delta_E.sum(axis=1)
        return float(self.delta_E[step].sum())


    def compute_total_consumption(self, step=None):
        if step is None:
            return self.after_total_per_house.sum(axis=1)
        return float(self.after_total_per_house[step].sum())


    def get_lam_bounds(self):
        lam_min = np.zeros_like(self.prices, dtype=float)  
        lam_max = 0.95 * self.prices
        return lam_min, lam_max
    

    def compute_delta_E(self, incentives, h):
        N, D = self.N, self.num_devices
        cap_h = float(self.capacity_threshold)
        capacity_arr = np.full(self.T, cap_h, dtype=float)

        # before (raw): baseline device load at this hour for controllable devices
        raw_ctrl = self.device_demands[h].copy()
        self.device_before[h] = raw_ctrl

        # effective consumption: apply removed-mask zeroing first, then apply scheduled future injections
        cons_eff = raw_ctrl.copy()
        mask_h = self._removed_mask[h]                
        cons_eff[mask_h] = 0.0 # zero the removed origin hours in the baseline profile only

        cons_eff += self.device_future_inj[h] # now add the injections
        self.device_future_inj[h, :, :] = 0.0 # clear the queue

       # used for capacity scheduling / capacity planning
        self.planned_load[h] = float(cons_eff.sum())

        # PC curtailment (only if not disabled)
        if not self.disable_pc:
            for i in range(N):
                lam_i = float(incentives[i])
                for d in range(D):
                    if not self.PC_MASK[d]:
                        continue
                    cons_d = cons_eff[i, d]
                    if cons_d <= 1e-9:
                        continue

                   # choose the optimal reduction rate
                    m = max(cons_d, 1e-6)  # for stability in division
                    kappa = float(self.dissatisfaction_coefficients[i, d])

                    best_val, best_de = 0.0, 0.0
                    for r in self.POWER_RATE: # e.g., [0.0, 0.1, ..., 0.6]
                        de = float(r) * cons_d
                        cost = kappa * ((de / m) ** 2)  # PC discomfort cost for choice
                        val = lam_i * de - cost
                        if val > best_val:
                            best_val, best_de = val, de

                    if best_de > 1e-9:
                        # apply reduction at the current hour
                        cons_eff[i, d] -= best_de
                        self.planned_load[h] -= best_de

                        # log customer discomfort for reward calculation
                        pc_cost = kappa * ((best_de / m) ** 2)
                        self.discomforts[h, i] += pc_cost

        # TS_NI: if job at h doesn't fit, defer one-shot
        # planned_load[h] currently includes E_job; pushing to backlog should subtract it
        for i in range(N):
            lam_i = float(incentives[i])
            may_move = (lam_i > 0.0)
            if not may_move:
                continue  #TS_NI devices are not modified unless the incentive is strictly positive
            for d in range(D):
                if not self.TS_NI_MASK[d]:
                    continue

                if self._removed_mask[h, i, d]:
                    # already zeroed in cons_eff by mask; just skip
                    continue

                E_job = cons_eff[i, d]
                if E_job <= 1e-9:
                    continue

                # TS_NI: move whole block, preserve shape
                # find the complete block around h (backward and forward)
                t_scan = h
                while t_scan - 1 >= 0 and self.device_before[t_scan - 1, i, d] > 1e-9:
                    t_scan -= 1
                t0 = t_scan

                t_scan = h
                while t_scan + 1 < self.T and self.device_before[t_scan + 1, i, d] > 1e-9:
                    t_scan += 1
                t1 = t_scan

                # profile and block length
                prof = self.device_before[t0:t1 + 1, i, d].copy()
                L = int(prof.shape[0])
                if L == 0:
                    continue

                # if this block started on previous day, do not shift it
                if t0 == 0 and self.ts_carry_over[i, d]:
                    # this is a carry-over job from yesterday → treat as fixed
                    continue

                # if h is in the middle of the block, do not shift at all (to avoid fragmenting the job)
                if h != t0:
                    continue

                # remove the effect of hour h from planned and zero out consumption at this hour
                for k in range(L):
                    t_orig = t0 + k
                    self._removed_mask[t_orig, i, d] = True
                    self.planned_load[t_orig] -= float(prof[k])

                # future hours will be zeroed when we arrive there thanks to the mask
                cons_eff[i, d] = 0.0

                # search for a window where the entire block finishes before the deadline
                dev_name = str(self.DEVICES[d]).lower()
                ddl = int(self.ts_deadline_hour.get(dev_name, 23))  # if you have human-readable hours 1..24 in config, do ddl -= 1 once

                def fits_at(start):
                    if start + L - 1 >= self.T:
                        return False
                    for k in range(L):
                        t_check = start + k
                        # destination hour load = planned_load[t] plus its pending injection queue
                        base_t = self.planned_load[t_check] + self.device_future_inj[t_check].sum()
                        if base_t + float(prof[k]) > capacity_arr[t_check] + 1e-12:
                            return False
                    return True

                start_from = min(h + 1, self.T - L) # ensure the start time occurs after h
                latest_start = min(self.T - L, ddl - L + 1) # ensure the entire block finishes by ddl

                # select the best window based on the lowest average load
                t_fit = None
                best_score = float('inf')

                if latest_start >= start_from:
                    for t_start in range(start_from, latest_start + 1):
                        # if this start is not acceptable in terms of capacity, skip it
                        if not fits_at(t_start):
                            continue

                        # average load in the proposed L-hour window
                        window_load = 0.0
                        for k in range(L):
                            t_check = t_start + k
                            base_total = float(self.total_baseline[t_check])
                            window_load += base_total

                        avg_load = window_load / L

                        if avg_load < best_score:
                            best_score = avg_load
                            t_fit = t_start

                if t_fit is None:
                    # stop injecting into start_from or h+1
                    # push the entire block into the backlog for handling in step 4
                    amount = float(prof.sum())
                    self.ts_backlog[i, d] += amount
                    # due-time: the device-specific deadline for completion
                    self.ts_due_time[i, d] = ddl
                else:
                    # injection at the found destination
                    for k in range(L):
                        tt = t_fit + k
                        add = float(prof[k])
                        self.device_future_inj[tt, i, d] += add
                        self.planned_load[tt] += add
                        # temporal discomfort
                        phi = float(self.dissatisfaction_coefficients[i, d])
                        delay = (tt + 1) - (h + 1)
                        self.discomforts[tt, i] += phi * (delay ** 2)

        # TS scheduling under capacity:
        # CAR (TS-I): interruptible, shift if capacity overflow and λ>0
        # TS-NI: one-shot scheduling with λ>0
        for d in range(D):
            if not self.TS_MASK[d]:
                continue

            dev_name = str(self.DEVICES[d]).lower()
            ddl = int(self.ts_deadline_hour.get(dev_name, 23))
            is_car = bool(self.TS_I_MASK[d])

            for i in range(N):
                lam_i = float(incentives[i])
                if lam_i <= 0.0:
                    continue

                # CAR (interruptible TS)
                if is_car:

                    e_h = cons_eff[i, d] # car consumption at this hour
                    total_h = self.planned_load[h] # total consumption at this hour

                    # if car is on and capacity is exceeded → immediate cut
                    if e_h > 1e-12 and total_h > cap_h + 1e-12:

                        backlog = e_h

                        # zero out consumption at this hour
                        cons_eff[i, d] = 0.0
                        self.device_after[h, i, d] = 0.0
                        self.planned_load[h] -= e_h

                        # fully transfer to the earliest acceptable hours
                        t_push = h + 1
                        while backlog > 1e-12 and t_push < self.T:

                            future_load = (self.planned_load[t_push] + self.device_future_inj[t_push].sum())

                            # if the entire backlog fits
                            if future_load + backlog <= capacity_arr[t_push] + 1e-12:
                                self.device_future_inj[t_push][i, d] += backlog
                                backlog = 0.0
                                break

                            # if no space → fill up to capacity
                            cap_left = capacity_arr[t_push] - future_load
                            if cap_left > 1e-12:
                                move = min(cap_left, backlog)
                                self.device_future_inj[t_push][i, d] += move
                                backlog -= move

                            t_push += 1

                    # if capacity was not exceeded, or car was off → do nothing
                    continue

                # TS-NI (dishwasher, dry, clotheswasher)
                # one-shot scheduling with λ>0
                amount = float(self.ts_backlog[i, d])
                if amount <= 1e-12:
                    continue

                # try to execute/run the device in this hour
                if self.planned_load[h] + amount <= cap_h:
                    cons_eff[i, d] += amount
                    self.ts_exec_agg[h, d] += amount
                    self.planned_load[h] += amount
                    self.device_future_inj[h, i, d] += 0.0
                    self.ts_backlog[i, d] = 0.0
                    self.ts_due_time[i, d] = -1

                else:
                    # finding the first acceptable hour (h+1..ddl)
                    t_fit = None
                    for t in range(h + 1, min(self.T - 1, ddl) + 1):
                        if self.planned_load[t] + amount <= capacity_arr[t]:
                            t_fit = t
                            break

                    if t_fit is not None:
                        self.ts_exec_agg[t_fit, d] += amount
                        self.planned_load[t_fit] += amount
                        self.device_future_inj[t_fit][i, d] += amount
                        self.ts_backlog[i, d] = 0.0
                        self.ts_due_time[i, d] = -1

                    else:
                        # fallback: enforce at ddl
                        t_fit = min(ddl, self.T - 1)
                        self.ts_exec_agg[t_fit, d] += amount
                        self.planned_load[t_fit] += amount
                        self.device_future_inj[t_fit][i, d] += amount
                        self.ts_backlog[i, d] = 0.0
                        self.ts_due_time[i, d] = -1

        # finalize hour h
        self.device_after[h] = cons_eff

        # (this variable is effectively 'after'; for clarity, consider renaming it to device_after_agg)
        self.device_after_agg[h] = cons_eff.sum(axis=0)

        # non-shiftable component equals the baseline value at hour h
        nonshift = self.nonshift_baseline_per_house[h].copy()

        # house-level total (before/after) = sum of controllable loads plus the non-shiftable component
        before_total_house = raw_ctrl.sum(axis=1) + nonshift
        after_total_house = cons_eff.sum(axis=1) + nonshift

        self.before_total_per_house[h] = before_total_house
        self.after_total_per_house[h] = after_total_house

        # house-level reduction (only positive)
        delta_E = np.maximum(0.0, before_total_house - after_total_house)

        # device-level reduction for this hour (only controllable; no double counting)
        before_dev = raw_ctrl.sum(axis=0)
        after_dev = cons_eff.sum(axis=0)
        self.reductions_by_device[h] = np.maximum(before_dev - after_dev, 0.0)

        return delta_E


    def get_step_data(self, step=None):
        h = self.curr_step if step is None else step
        baselines = self.baseline_per_house[h]
        lambdas = self.incentives[h]
        price = self.prices[h]
        delta_E = self.delta_E[h]
        return h, baselines, lambdas, price, delta_E

    
    def get_device_data(self, step=None):
        if step is None:
            step = self.curr_step
        return self.device_demands[step]


    def get_total_device_consumption(self, step=None):
        if step is None:
            step = self.curr_step
        return self.device_demands[step].sum(axis=1) 
    
