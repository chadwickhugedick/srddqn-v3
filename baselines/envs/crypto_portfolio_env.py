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
    - Real trading fees (percentage based on equity, charged on position changes)
    - Holding cost (small per-step cost for being in an active position,
      discouraging pointless constant trading on hourly data)
    - Bankruptcy termination
    - Early termination if equity falls below a configurable floor

    Fee note
    --------
    The original fee_pct=0.001 (0.1%) was calibrated for daily data.  On hourly
    data the agent can flip positions hundreds of times per episode, which
    compounds the fee drag catastrophically.  The new default (0.0002, 0.02%) is
    closer to realistic crypto perpetual futures taker fees.

    Holding cost
    ------------
    holding_cost_pct (default 0.00001 = 0.001% per step) is subtracted from
    equity every step the agent holds an active (non-flat) position.  This is
    tiny individually but gives the agent a non-zero incentive to be flat when
    there is no clear signal, reducing overtrading.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 lookback_window: int = 10,
                 initial_capital: float = 10000.0,
                 fee_pct: float = 0.0002,
                 holding_cost_pct: float = 0.00001,
                 bankruptcy_floor: float = 0.1):
        """
        Args:
            df:                 DataFrame with market data.
            feature_cols:       List of feature column names.
            lookback_window:    Number of historical steps in the observation.
            initial_capital:    Starting equity ($).
            fee_pct:            Fee as a fraction of equity charged on each
                                position change (default 0.02%).
            holding_cost_pct:   Per-step cost as a fraction of equity while
                                holding a non-flat position (default 0.001%).
            bankruptcy_floor:   Episode terminates if equity drops below
                                this fraction of initial_capital (default 10%).
        """
        super().__init__(df, feature_cols, lookback_window)
        self.initial_capital  = initial_capital
        self.fee_pct          = fee_pct
        self.holding_cost_pct = holding_cost_pct
        self.bankruptcy_floor = bankruptcy_floor

        # Observation space: flattened features + position + equity
        orig_obs_dim = (self.lookback_window * len(self.feature_cols)) + 1
        new_obs_dim  = orig_obs_dim + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32
        )

        self.equity = self.initial_capital

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        orig_obs, info = super().reset(seed=seed, options=options)
        self.equity = self.initial_capital
        return self._get_portfolio_obs(orig_obs), info

    # ------------------------------------------------------------------
    def step(self, action: int):
        # Ensure action is a standard Python int (SB3 predict can return numpy array/scalar)
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        # Map discrete actions to position targets
        target_position = {0: 0, 1: 1, 2: -1}.get(action, 0)

        current_price = self.close_prices[self.current_step]
        next_price    = self.close_prices[self.current_step + 1]
        price_change_pct = (next_price - current_price) / current_price

        # ── Transaction fee (only on position change) ──────────────────
        fee_cost = 0.0
        if target_position != self.current_position:
            # A flip from Long (+1) → Short (-1) traverses 2 units of position;
            # charge proportionally.
            trade_size = abs(target_position - self.current_position)
            fee_cost   = self.equity * self.fee_pct * trade_size

        # ── Holding cost (per-step cost for being in a position) ────────
        holding_cost = 0.0
        if self.current_position != 0:
            holding_cost = self.equity * self.holding_cost_pct

        # ── PnL on existing position ────────────────────────────────────
        pnl = self.equity * target_position * price_change_pct

        # ── Update equity ───────────────────────────────────────────────
        self.equity = self.equity + pnl - fee_cost - holding_cost

        # ── Reward: normalised by initial capital ───────────────────────
        # Using initial_capital normalisation makes the reward scale-invariant
        # regardless of how much equity has drifted.
        if self.equity > 0:
            reward = (pnl - fee_cost - holding_cost) / self.initial_capital
        else:
            reward = -1.0  # Bankruptcy penalty

        self.current_position = target_position
        self.current_step    += 1

        # ── Termination checks ──────────────────────────────────────────
        truncated  = self.current_step >= self.n_steps - 1
        terminated = self.equity <= self.initial_capital * self.bankruptcy_floor

        if terminated:
            self.equity = max(self.equity, 0.0)

        # ── Build obs ───────────────────────────────────────────────────
        # Temporarily step back to call _get_obs (which uses current_step)
        self.current_step -= 1
        orig_obs = self._get_obs()
        self.current_step += 1

        info = {
            "price":      next_price,
            "position":   self.current_position,
            "equity":     self.equity,
            "reward":     reward,
            "pnl":        pnl,
            "fee_cost":   fee_cost,
            "holding_cost": holding_cost,
        }

        return self._get_portfolio_obs(orig_obs), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_portfolio_obs(self, orig_obs):
        normalized_equity = self.equity / self.initial_capital
        obs = np.append(orig_obs, normalized_equity)
        return obs.astype(np.float32)
