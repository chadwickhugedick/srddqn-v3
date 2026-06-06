import gymnasium as gym
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.timesnet import TimesNet

class SRDRLWrapper(gym.Wrapper):
    """
    Self-Rewarding Deep Reinforcement Learning Wrapper.
    
    Intercepts the environment's step function and augments the reward
    using the pre-trained TimesNet's prediction of the Min-Max macro reward.
    
    r_t = max(r_env, r_timesnet[a_t])
    """
    def __init__(self, env, model_path, seq_len=10, num_features=27):
        super().__init__(env)
        self.seq_len = seq_len
        self.num_features = num_features
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained TimesNet
        self.timesnet = TimesNet(seq_len=seq_len, num_features=num_features, d_model=32, d_ff=64, e_layers=2, top_k=2).to(self.device)
        
        if os.path.exists(model_path):
            self.timesnet.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"Loaded pre-trained TimesNet from {model_path}")
        else:
            raise FileNotFoundError(f"TimesNet model not found at {model_path}")
            
        self.timesnet.eval()

    def step(self, action):
        # Take step in the underlying environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get the underlying sequence of features for TimesNet
        # The base CryptoEnv stores historical features
        unwrapped_env = self.env.unwrapped
        
        start_idx = unwrapped_env.current_step - self.seq_len
        end_idx = unwrapped_env.current_step
        
        if start_idx >= 0:
            hist_features = unwrapped_env.features[start_idx:end_idx]
            
            # Predict reward via TimesNet
            with torch.no_grad():
                x = torch.tensor(hist_features, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, Seq, Feat]
                pred_rewards = self.timesnet(x).squeeze(0).cpu().numpy() # [3]
                
            # The action passed in is 0 (Flat), 1 (Long), or 2 (Short)
            r_timesnet = pred_rewards[action]
            
            # The SRDRL Mechanism: max(r_env, r_timesnet)
            # This ensures the agent gets a positive reward if the macro trend is favorable,
            # even if the immediate 1H step was noisy or hit a small fee.
            final_reward = max(float(reward), float(r_timesnet))
        else:
            final_reward = reward
            
        info['original_reward'] = reward
        info['augmented_reward'] = final_reward
            
        return obs, final_reward, terminated, truncated, info
