import os
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from env import Environment
from agent.agent_dqn import DQNAgent
from utils.config_loader import load_config
from datetime import datetime

# Font settings for plots 
plt.rcParams.update({'font.size': 24,'axes.labelsize': 24,'xtick.labelsize': 22, 
                     'ytick.labelsize': 22,'legend.fontsize': 20,'legend.title_fontsize': 22})
mpl.rcParams['hatch.linewidth'] = 3  # Make hatch lines thicker


class DQNTrainer:
    def __init__(self, cfg_override=None):
        self.cfg = cfg_override or load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup environment
        house_ids = self.cfg['environment']['house_ids']
        self.env = Environment(data_ids=house_ids, cfg_override=self.cfg)
        
        # Get state and action dimensions
        state = self.env.reset(mode='train')
        state_dim = len(state)
        action_dim = self.env.num_actions
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        
        # Setup agent
        self.agent = DQNAgent(state_dim, action_dim, cfg_override=self.cfg)

        # Training parameters
        self.episodes = self.cfg['DQN']['episodes']
        self.max_steps = self.cfg['DQN']['max_steps']

        # Evaluation parameters
        self.eval_interval = self.cfg['DQN']['eval_interval']
        self.eval_episodes = self.cfg['DQN']['eval_episodes']    
        self.eval_epsilon = self.cfg['DQN']['eval_epsilon']    

        # Tracking metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.episode_q_values = []  
        self.current_episode_q_values = []

        # Tracking evaluation metrics
        self.eval_rewards = []      
        self.eval_timesteps = []   

        # Create results directory
        self.results_dir = f"results/ddqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results directory: {self.results_dir}")


    def evaluate_policy(self, episode):
        """Evaluate the current policy on validation episodes"""
        eval_rewards = []
        
        for eval_ep in range(self.eval_episodes):
            state = self.env.reset(mode='val')
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < self.max_steps:
                action = self.agent.select_action(state, eval_mode=True, eval_epsilon=self.eval_epsilon)
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
            eval_rewards.append(episode_reward)
        
        avg_eval_reward = np.mean(eval_rewards)
        self.eval_rewards.append(avg_eval_reward)
        self.eval_timesteps.append(episode)
        print(f"Evaluation at episode {episode}: Average Reward = {avg_eval_reward:.2f}")


    def train(self):
        """Main training loop"""
        print("Starting DQN Training...")
        print(f"Device: {self.device}")
        print(f"Total episodes: {self.episodes}")

        for episode in range(self.episodes):
            # Reset environment
            state = self.env.reset(mode='train')
            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            self.current_episode_q_values = [] 

            for step in range(self.max_steps):
                # Select action
                action = self.agent.select_action(state, eval_mode=False)

                # Sample Q-values periodically for monitoring
                if step % 1 == 0:  
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.agent.policy_net(state_tensor)
                        max_q = q_values.max().item()
                        self.current_episode_q_values.append(max_q)

                # Take action in environment
                next_state, reward, done, _ = self.env.step(action)

                # Store experience in replay buffer
                self.agent.memory.add(state, action, reward, next_state, done)

                # Train the agent
                loss = self.agent.train()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1

                # Update target network (soft update happens in agent)
                self.agent.update_target_network()

                # Update state and reward
                state = next_state
                episode_reward += reward
                
                if done:
                    break

            # Evaluate policy periodically
            if episode % self.eval_interval == 0:
                self.evaluate_policy(episode)

            # Track Q-values
            if self.current_episode_q_values:
                avg_q = np.mean(self.current_episode_q_values)
                self.episode_q_values.append(avg_q)
            else:
                self.episode_q_values.append(0)  

            # Update epsilon after each episode
            self.agent.update_epsilon()

            # Calculate average loss for the episode
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0

            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(avg_loss)
            self.episode_lengths.append(step + 1)
            self.epsilon_history.append(self.agent.epsilon)

            # Logging
            if episode % 50 == 0:
                self.log_progress(episode)

            # Save model periodically
            if episode % 500 == 0 and episode > 0:
                self.save_checkpoint(episode)

        self.finalize_training()


    def log_progress(self, episode):
        """Log training progress"""
        window_size = min(50, len(self.episode_rewards))
        if window_size > 0:
            avg_reward = np.mean(self.episode_rewards[-window_size:])
            avg_loss = np.mean(self.episode_losses[-window_size:])
        else:
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0
             
        print(f"Episode {episode+1}/{self.episodes} | "
            f"Reward: {self.episode_rewards[-1]:.2f} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {self.agent.epsilon:.4f}")


    def save_checkpoint(self, episode):
        """Save model checkpoint and training metrics"""
        checkpoint_path = os.path.join(self.results_dir, f"dqn_checkpoint_episode_{episode}.pth")
        self.agent.save(checkpoint_path)
        
        # Save training metrics
        metrics = {
            'episode_rewards': self.episode_rewards, 
            'episode_losses': self.episode_losses, 
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'episode_q_values': self.episode_q_values,
            'eval_rewards': self.eval_rewards,          
            'eval_timesteps': self.eval_timesteps 
        }
        np.save(os.path.join(self.results_dir, 'training_metrics.npy'), metrics)
        print(f"Checkpoint saved: {checkpoint_path}")


    def finalize_training(self):
        """Finalize training and save final model"""
        # Save final model
        final_model_path = os.path.join(self.results_dir, "dqn_final_model.pth")
        self.agent.save(final_model_path)
        
        # Save training curves
        self.plot_training_curves()
        
        print(f"\nTraining completed!")
        print(f"Final model saved: {final_model_path}")


    def plot_training_curves(self):
        """Plot and save training curves"""
        try:
            window_size = 50  
            
            # Reward curve
            if self.episode_rewards and len(self.episode_rewards) >= window_size:
                rewards = np.array(self.episode_rewards)
                reward_moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                x_axis = np.arange(window_size, len(reward_moving_avg) + window_size)
                
                plt.figure(figsize=(12, 8))
                plt.plot(x_axis, reward_moving_avg, color='blue', linewidth=2.0)
                plt.title('Training Reward', fontsize=18)
                plt.xlabel('Training Episode', fontsize=18)
                plt.ylabel('Average Reward', fontsize=18)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'episode_reward.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Figure 'Training Reward' plotted and saved successfully")

            # Loss curve
            if self.episode_losses and len(self.episode_losses) >= window_size:
                losses = np.array(self.episode_losses)
                loss_moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                x_axis = np.arange(window_size, len(loss_moving_avg) + window_size)
                
                plt.figure(figsize=(12, 8))
                plt.plot(x_axis, loss_moving_avg, color='red', linewidth=2.0)
                plt.title('Training Loss', fontsize=18)
                plt.xlabel('Training Episode', fontsize=18)
                plt.ylabel('Average Loss', fontsize=18)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Figure 'Training Loss' plotted and saved successfully")

            # Q-values curve
            if self.episode_q_values and len(self.episode_q_values) >= window_size:
                q_vals = np.array(self.episode_q_values)
                q_moving_avg = np.convolve(q_vals, np.ones(window_size)/window_size, mode='valid')
                x_axis = np.arange(window_size, len(q_moving_avg) + window_size)
                
                plt.figure(figsize=(12, 8))
                plt.plot(x_axis, q_moving_avg, color='green', linewidth=2.0)
                plt.title('Q-values', fontsize=18)
                plt.xlabel('Training Episode', fontsize=18)
                plt.ylabel('Average Q-value', fontsize=18)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'q_values.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Figure 'Q-values' plotted and saved successfully")

            # Epsilon decay curve
            if self.epsilon_history:
                epsilon_hist = np.array(self.epsilon_history)
                plt.figure(figsize=(12, 8))
                plt.plot(epsilon_hist, color='purple', linewidth=2.0)
                plt.title('Epsilon Decay', fontsize=18)
                plt.xlabel('Training Episode', fontsize=18)
                plt.ylabel('Epsilon', fontsize=18)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'epsilon_decay.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Figure 'Epsilon Decay' plotted and saved successfully")

            # Evaluation curve
            if self.eval_rewards and self.eval_timesteps:
                plt.figure(figsize=(12, 8))
                plt.plot(self.eval_timesteps, self.eval_rewards, 'o-', color='darkorange', 
                        linewidth=3.0, markersize=8, label='Evaluation Reward')
                plt.title('Policy Evaluation', fontsize=18)
                plt.xlabel('Training Episode', fontsize=18)
                plt.ylabel('Average Evaluation Reward', fontsize=18)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'evaluation_curve.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Figure 'Policy Evaluation' plotted and saved successfully")

            # Training vs Validation comparison
            if self.eval_rewards and self.eval_timesteps and len(self.episode_rewards) >= window_size:
                plt.figure(figsize=(12, 8))
                
                train_rewards = np.array(self.episode_rewards)
                train_moving_avg = np.convolve(train_rewards, np.ones(window_size)/window_size, mode='valid')
                train_episodes = np.arange(window_size, len(train_moving_avg) + window_size)
                
                plt.plot(train_episodes, train_moving_avg, 'b-', linewidth=2.5, 
                        label='Training Reward')
                
                # Validation
                plt.plot(self.eval_timesteps, self.eval_rewards, 'ro-', linewidth=3.0, markersize=8, 
                        markerfacecolor='red', markeredgecolor='darkred', label='Validation Reward')
                
                plt.title('Training vs Validation Performance', fontsize=18)
                plt.xlabel('Training Episode', fontsize=18)
                plt.ylabel('Average Reward', fontsize=18)
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'train_vs_validation.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Figure 'Training vs Validation Performance' plotted and saved successfully")
                
            
        except Exception as e:
            print(f"Error plotting training curves: {e}")


def main():
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Initialize and run trainer
    trainer = DQNTrainer()
    trainer.train()


if __name__ == "__main__":
    main()