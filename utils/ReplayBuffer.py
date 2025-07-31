import random
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, n_step=2, gamma=0.99):
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            s0, a0, r0, s1, d0 = self.n_step_buffer[0]
            cum_reward = sum((self.gamma**i) * exp[2] for i, exp in enumerate(self.n_step_buffer))
            _, _, _, sn, dn = self.n_step_buffer[-1]
            e = self.experience(s0, a0, cum_reward, sn, d0 or dn)
            self.memory.append(e)

    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.memory)
