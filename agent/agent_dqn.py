import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.ReplayBuffer import ReplayBuffer
from utils.config_loader import load_config

# Define Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# Training the DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, cfg_override=None):
        self.cfg = cfg_override if cfg_override else load_config()

        # DQN Hyperparameters from config
        self.lr = self.cfg['DQN']['LEARNING_RATE_DQN'] 
        self.gamma = self.cfg['DQN']['gamma']
        self.epsilon = self.cfg['DQN']['epsilon_start']
        self.epsilon_min = self.cfg['DQN']['epsilon_min']
        self.epsilon_decay = self.cfg['DQN']['epsilon_decay']
        self.batch_size = self.cfg['DQN']['BATCH_SIZE']
        self.buffer_size = self.cfg['DQN']['BUFFER_SIZE']
        self.training_interval = self.cfg['DQN']['TRAINING_INTERVAL']
        self.tau = self.cfg['DQN']['TAU']

        # Dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Training counters 
        self.steps_done = 0

        # For logging
        self.training_history = {'loss': [], 'episode_rewards': [], 'epsilon': [], 'q_values': []}

    def select_action(self, state, eval_mode=False, eval_epsilon=0.01):
        if eval_mode:
            # Evaluation mode: mostly greedy with very small exploration
            if random.random() < eval_epsilon:
                return random.randint(0, self.action_dim - 1)
            # Otherwise, select best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()  

        # Training mode: epsilon-greedy
        self.steps_done += 1
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()  # Exploit
                
    def update_epsilon(self):
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_history['epsilon'].append(self.epsilon)
            
    def train(self):
        # Skip training if replay buffer not ready or not at training interval
        if len(self.memory) < self.batch_size * 5 or self.steps_done % self.training_interval != 0:
            return None
        
        # Sample a batch of experiences from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            # DOUBLE DQN
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            
            # DQN
            # next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values.squeeze())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        # Soft update of target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save({'policy_net_state_dict': self.policy_net.state_dict(), 'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 'training_history': self.training_history, 'epsilon': self.epsilon,
            'steps_done': self.steps_done}, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"Model loaded from {filepath}")


    



