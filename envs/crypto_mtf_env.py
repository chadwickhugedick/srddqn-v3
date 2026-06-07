import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from baselines.envs.crypto_portfolio_env import CryptoPortfolioEnv


class CryptoMtfEnv(CryptoPortfolioEnv):
    """
    Multi-Timeframe Crypto Environment.

    Inherits equity tracking, fees, and holding cost from CryptoPortfolioEnv.
    Outputs a 2D-flattened sequence of features across all timeframes.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 lookback_window: int = 10,
                 initial_capital: float = 10000.0,
                 fee_pct: float = 0.0002,
                 holding_cost_pct: float = 0.00001,
                 bankruptcy_floor: float = 0.1):
        # Pass all parameters through to CryptoPortfolioEnv
        super().__init__(
            df, feature_cols,
            lookback_window=lookback_window,
            initial_capital=initial_capital,
            fee_pct=fee_pct,
            holding_cost_pct=holding_cost_pct,
            bankruptcy_floor=bankruptcy_floor,
        )

        # Override observation space:
        # (lookback_window × num_features) + position + equity
        flat_dim = (self.lookback_window * len(self.feature_cols)) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def _get_obs(self):
        """Flatten the (lookback_window, num_features) matrix and append position."""
        start_idx    = self.current_step - self.lookback_window
        end_idx      = self.current_step
        hist_features = self.features[start_idx:end_idx]

        flat_features = hist_features.flatten()
        obs = np.append(flat_features, float(self.current_position))
        obs = np.clip(obs.astype(np.float32), -100.0, 100.0)
        return obs
