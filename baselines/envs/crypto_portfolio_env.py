import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from baselines.envs.basic_crypto_env import BasicCryptoEnv

class CryptoPortfolioEnv(BasicCryptoEnv):
    """
    A more realistic Crypto Environment that inherits from BasicCryptoEnv.
    Adds:
    - Equity tracking
    - Real trading fees (percentage based)
    - Bankruptcy termination
    """
    
    def __init__(self, df: pd.DataFrame, feature_cols: list, lookback_window: int = 10,
                 initial_capital: float = 10000.0, fee_pct: float = 0.001):
        super().__init__(df, feature_cols, lookback_window)
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        
        # Need to append equity to observation space
        # Original obs_dim: (lookback_window * len(feature_cols)) + 1
        # New obs_dim: Original + 1 (for equity)
        orig_obs_dim = (self.lookback_window * len(self.feature_cols)) + 1
        new_obs_dim = orig_obs_dim + 1
        
        # Expanding the observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32)
        
        self.equity = self.initial_capital
        
    def reset(self, seed=None, options=None):
        # Call parent reset to get features and step reset
        orig_obs, info = super().reset(seed=seed, options=options)
        
        # Reset portfolio
        self.equity = self.initial_capital
        
        return self._get_portfolio_obs(orig_obs), info
        
    def step(self, action: int):
        # Target positions mapping: 0=Flat, 1=Long, 2=Short
        target_position = 0
        if action == 1:
            target_position = 1
        elif action == 2:
            target_position = -1
            
        current_price = self.close_prices[self.current_step]
        next_price = self.close_prices[self.current_step + 1]
        
        # Calculate actual dollar PnL based on position and price change
        price_change_pct = (next_price - current_price) / current_price
        
        # If we changed position, we pay a fee on the equity
        fee_cost = 0.0
        if target_position != self.current_position:
            # We charge a percentage fee on the entire equity when entering/exiting a position
            # If flipping from Long to Short, technically it's 2 trades, but for simplicity we charge once per position change
            # To be more realistic, a flip from 1 to -1 is double the fee of 1 to 0.
            trade_size = abs(target_position - self.current_position)
            fee_cost = self.equity * self.fee_pct * trade_size
            
        # PnL on the position
        pnl = self.equity * self.current_position * price_change_pct
        
        # Update equity
        self.equity = self.equity + pnl - fee_cost
        
        # Reward is the percentage change in equity
        # Using percentage change instead of raw PnL makes the reward scale-invariant
        if self.equity > 0:
            reward = (pnl - fee_cost) / self.initial_capital  # Normalized by initial capital to avoid extreme values
        else:
            reward = -1.0 # Bankruptcy penalty
            
        self.current_position = target_position
        self.current_step += 1
        
        terminated = False
        truncated = False
        
        if self.current_step >= self.n_steps - 1:
            truncated = True
            
        if self.equity <= 0:
            terminated = True
            self.equity = 0
            
        info = {
            "price": next_price,
            "position": self.current_position,
            "equity": self.equity,
            "reward": reward
        }
        
        # Get parent obs to maintain historical features, then append equity
        # We temporarily decrement current_step because _get_obs uses current_step (which we just incremented)
        self.current_step -= 1
        orig_obs = self._get_obs()
        self.current_step += 1
        
        return self._get_portfolio_obs(orig_obs), float(reward), terminated, truncated, info

    def _get_portfolio_obs(self, orig_obs):
        # Normalize equity relative to initial capital to keep network inputs reasonably scaled
        normalized_equity = self.equity / self.initial_capital
        obs = np.append(orig_obs, normalized_equity)
        return obs.astype(np.float32)
