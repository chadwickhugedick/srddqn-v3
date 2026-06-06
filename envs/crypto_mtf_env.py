import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from envs.crypto_portfolio_env import CryptoPortfolioEnv

class CryptoMtfEnv(CryptoPortfolioEnv):
    """
    Multi-Timeframe Crypto Environment.
    Inherits equity tracking and fees from CryptoPortfolioEnv.
    Outputs a 2D-flattened sequence of features across all timeframes.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list, lookback_window: int = 10,
                 initial_capital: float = 10000.0, fee_pct: float = 0.001):
        
        # Initialize CryptoPortfolioEnv
        super().__init__(df, feature_cols, lookback_window, initial_capital, fee_pct)
        
        # Original obs_dim: (lookback_window * len(feature_cols))
        # Plus 1 for position, plus 1 for equity
        flat_dim = (self.lookback_window * len(self.feature_cols)) + 2
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

    def _get_obs(self):
        # We override _get_obs just to ensure it properly flattens the 2D MTF matrix
        # base _get_obs already does this: `self.features[start:end].flatten()`
        # which is exactly what we need for an MLP baseline.
        # The parent classes will automatically append position and equity.
        
        # Get historical features (seq_len, num_features)
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        hist_features = self.features[start_idx:end_idx]
        
        # Flatten to 1D
        flat_features = hist_features.flatten()
        
        # Append position
        obs = np.append(flat_features, float(self.current_position))
        
        # Clip to arbitrary large bounds
        obs = np.clip(obs.astype(np.float32), -100.0, 100.0)
        
        return obs
