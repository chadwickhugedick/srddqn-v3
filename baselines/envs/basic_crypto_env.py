import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BasicCryptoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, df: pd.DataFrame, feature_cols: list, lookback_window: int = 10):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.lookback_window = lookback_window
        
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.close_prices = self.df['close'].values.astype(np.float32)
        
        self.n_steps = len(self.df)
        
        # Action space: 0: Flat, 1: Long, 2: Short
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Flat array of (lookback_window * num_features) + 1 (current position)
        obs_dim = (self.lookback_window * len(self.feature_cols)) + 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        
        self.current_step = self.lookback_window
        self.current_position = 0 # -1, 0, 1
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start at a random index, leaving enough room for an episode
        # Default episode length will be handled by a TimeLimit wrapper or externally
        if options is not None and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            self.current_step = self.np_random.integers(self.lookback_window, max(self.lookback_window + 1, self.n_steps - 1000))
        
        self.current_position = 0
        
        return self._get_obs(), {}
        
    def step(self, action: int):
        # Map action 0,1,2 to position 0, 1, -1
        if action == 0:
            target_position = 0
        elif action == 1:
            target_position = 1
        elif action == 2:
            target_position = -1
            
        # Calculate log return for the step
        current_price = self.close_prices[self.current_step]
        next_price = self.close_prices[self.current_step + 1]
        
        step_log_return = np.log(next_price / current_price)
        
        # Reward is proportional to our current position.
        fee_penalty = 0.0
        if target_position != self.current_position:
            fee_penalty = 0.0 # 0.1% fee simulation
            
        reward = (target_position * step_log_return) - fee_penalty
        
        self.current_position = target_position
        self.current_step += 1
        
        terminated = False
        truncated = False
        
        if self.current_step >= self.n_steps - 1:
            truncated = True
            
        info = {
            "price": next_price,
            "position": self.current_position,
            "reward": reward
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info
        
    def _get_obs(self):
        # Get historical features
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        hist_features = self.features[start_idx:end_idx].flatten()
        
        # Append current position state
        obs = np.append(hist_features, float(self.current_position))
        
        # Clip to bounds just in case of extreme outliers
        return np.clip(obs.astype(np.float32), -10.0, 10.0)
