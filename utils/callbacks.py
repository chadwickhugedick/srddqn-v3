import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class FinancialMetricsCallback(BaseCallback):
    """
    Custom callback for plotting financial metrics during training.
    Logs final equity, max/min equity, and position distribution per episode.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.equities = []
        self.positions = []
        self.original_rewards = []
        self.augmented_rewards = []

    def _on_step(self) -> bool:
        # We assume a single environment (not vectorized, or just taking the first one)
        info = self.locals["infos"][0]
        
        if "equity" in info:
            self.equities.append(info["equity"])
        if "position" in info:
            self.positions.append(info["position"])
        if "original_reward" in info:
            self.original_rewards.append(info["original_reward"])
        if "augmented_reward" in info:
            self.augmented_rewards.append(info["augmented_reward"])
            
        # Check if episode is done
        if self.locals["dones"][0]:
            if len(self.equities) > 0:
                final_equity = self.equities[-1]
                min_equity = min(self.equities)
                max_equity = max(self.equities)
                
                self.logger.record("financial/final_equity", final_equity)
                self.logger.record("financial/min_equity", min_equity)
                self.logger.record("financial/max_equity", max_equity)
                
                pos = np.array(self.positions)
                self.logger.record("financial/pct_long", float(np.mean(pos == 1)))
                self.logger.record("financial/pct_short", float(np.mean(pos == -1)))
                self.logger.record("financial/pct_flat", float(np.mean(pos == 0)))
                
                if len(self.original_rewards) > 0:
                    self.logger.record("financial/mean_original_reward", float(np.mean(self.original_rewards)))
                    self.logger.record("financial/mean_augmented_reward", float(np.mean(self.augmented_rewards)))
                
            # Reset for next episode
            self.equities = []
            self.positions = []
            self.original_rewards = []
            self.augmented_rewards = []
            
        return True
