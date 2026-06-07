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
        
        # PRECOMPUTE REWARDS FOR ENTIRE DATASET
        unwrapped_env = self.env.unwrapped
        total_steps = len(unwrapped_env.features)
        self.precomputed_rewards = np.zeros((total_steps, 3), dtype=np.float32)
        
        print(f"Precomputing TimesNet rewards for {total_steps} timesteps on {self.device}...")
        
        batch_size = 1024
        valid_indices = []
        valid_seqs = []
        
        for i in range(self.seq_len, total_steps):
            valid_indices.append(i)
            valid_seqs.append(unwrapped_env.features[i - self.seq_len : i])
            
        if len(valid_seqs) > 0:
            valid_seqs = np.array(valid_seqs, dtype=np.float32)
            
            for i in range(0, len(valid_seqs), batch_size):
                batch = torch.tensor(valid_seqs[i:i+batch_size]).to(self.device)
                with torch.no_grad():
                    preds = self.timesnet(batch).cpu().numpy()
                self.precomputed_rewards[valid_indices[i:i+batch_size]] = preds
                
        print("TimesNet precomputation complete! RL training will now run at full speed.")
        
        # Free memory since we won't need the model for stepping anymore
        del self.timesnet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def step(self, action):
        # Take step in the underlying environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        unwrapped_env = self.env.unwrapped
        current_step = unwrapped_env.current_step
        
        if current_step >= self.seq_len and current_step < len(self.precomputed_rewards):
            # Fetch the precomputed TimesNet prediction for this step
            r_timesnet = self.precomputed_rewards[current_step][action]
            
            # The SRDRL Mechanism: max(r_env, r_timesnet)
            final_reward = max(float(reward), float(r_timesnet))
        else:
            final_reward = reward
            
        info['original_reward'] = reward
        info['augmented_reward'] = final_reward
            
        return obs, final_reward, terminated, truncated, info
