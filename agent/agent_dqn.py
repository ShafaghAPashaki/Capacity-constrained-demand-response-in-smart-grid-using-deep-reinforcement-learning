import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from env import Environment
from utils.ReplayBuffer import ReplayBuffer
from utils.config_loader import load_config

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, cfg_override=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Load config if not provided
        # self.cfg = cfg if cfg else load_config()
        # DQN Hyperparameters from config
        self.cfg = cfg_override if cfg_override else load_config() 
        self.gamma = self.cfg['DQN']['gamma']
        self.epsilon = self.cfg['DQN']['epsilon_start']
        self.epsilon_min = self.cfg['DQN']['epsilon_min']
        self.epsilon_decay = self.cfg['DQN']['epsilon_decay']
        self.tau = self.cfg['DQN']['TAU']
        self.batch_size = self.cfg['DQN']['BATCH_SIZE']
        self.buffer_size = self.cfg['DQN']['BUFFER_SIZE']
        self.lr = self.cfg['DQN']['LEARNING_RATE_DQN']
        self.hidden_layer_size = self.cfg['DQN']['HIDDEN_LAYER_SIZE']
        self.training_interval = self.cfg['DQN']['TRAINING_INTERVAL']
        self.replace_target_interval = self.cfg['DQN']['REPLACE_TARGET_INTERVAL']
        # Q-Networks
        self.policy_net = DQN(state_dim, action_dim, self.hidden_layer_size).to(device)
        self.target_net = DQN(state_dim, action_dim, self.hidden_layer_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), self.lr, weight_decay=1e-5)
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, n_step=2, gamma=self.gamma)
        # Initialize time step 
        self.steps_done = 0
        # Track losses
        self.losses = [] 

    def step(self, state, action, reward, next_state, done):
        # Save experience to replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every TRAINING_INTERVAL steps
        self.steps_done += 1
        if len(self.memory) > self.batch_size and self.steps_done % self.training_interval == 0:
            # Only store loss when training occurs
            loss_value = self.learn() 
            self.losses.append(loss_value)
            # Soft update
            if self.steps_done % self.replace_target_interval == 0:
              self.soft_update_target_network()
            return
        elif self.steps_done % self.replace_target_interval == 0:
            self.soft_update_target_network()

    def soft_update_target_network(self): 
        # Soft update target network parameters
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def act(self, state):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()
        else:
            return random.randrange(self.action_dim)
        
    def learn(self):
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample()
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        # Get current Q values for chosen actions
        q_values = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze()
        # Compute next Q values using target network
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        # Compute target Q values
        target_q_values = rewards + (self.gamma) * (1-dones) * next_q_values
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        loss_value = loss.item()
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss_value 

    def save(self, path):
        # Save policy network weights
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        # Load policy network weights
        self.policy_net.load_state_dict(torch.load(path))


def train_dqn(rho=None, eval_interval=100, max_episodes=None, house_ids=None):
    """
    Train a DQN agent, optionally overriding the environment’s rho parameter.
    Returns:
        reward_history: list of total reward per episode
        q_trace: list of average max‐Q over fixed reference states every eval_interval
    """
    # 1. Load config and apply rho/house_ids overrides
    cfg = load_config()
    if rho is not None:
        cfg['environment']['rho'] = rho
    if house_ids is not None:
        cfg['environment']['house_ids'] = house_ids

    # 2. Initialize environment and agent
    house_ids = cfg['environment']['house_ids']
    env = Environment(house_ids, cfg_override=cfg)
    first_state = env.reset()
    state_dim = first_state.shape[0]
    action_dim = len(env.all_actions)

    agent = DQNAgent(state_dim, action_dim, cfg_override=cfg)

    # 3. Prepare fixed states for Q‐trace evaluation
    fixed_states = [env.reset() for _ in range(10)]
    q_trace = []
    reward_history = []

    # 4. Determine number of episodes
    if max_episodes is None:
        max_episodes = cfg['DQN']['episodes']
    print_interval = max(1, max_episodes // 10)

    # 5. Training loop
    for ep in range(1, max_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward or 0.0, next_state, done)
            state = next_state
            total_reward += reward or 0.0

        reward_history.append(total_reward)

        # 6. Periodic Q‐trace logging
        if ep % eval_interval == 0:
            with torch.no_grad():
                qs = []
                for s in fixed_states:
                    s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                    q_val = agent.policy_net(s_tensor).max().item()
                    qs.append(q_val)
            q_trace.append(sum(qs) / len(qs))

        # 7. Progress printout
        if ep % print_interval == 0:
            avg_loss = np.mean(agent.losses) if agent.losses else 0.0
            agent.losses = []
            print(f"[ρ={rho}] Ep {ep}/{max_episodes} | R:{total_reward:.2f} | "
                  f"ε:{agent.epsilon:.3f} | Loss:{avg_loss:.4f}")

    # 8. Save model with rho in filename
    model_name = f"dqn_model_rho{rho}.pth" if rho is not None else "dqn_model.pth"
    agent.save(model_name)
    print(f"Model saved → {model_name}")

    return reward_history, q_trace

    
if __name__ == "__main__":
    rho_vals = [0.7]   
    for rho in rho_vals:
        print(f"\n=== TRAINING ρ={rho} ===")
        train_dqn(rho=rho, eval_interval=100)


        
    




           





        

        






      
      
