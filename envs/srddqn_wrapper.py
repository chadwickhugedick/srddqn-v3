import gymnasium as gym
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.timesnet import TimesNet


class SRDRLWrapper(gym.Wrapper):
    """Self-Rewarding Deep Reinforcement Learning Wrapper.

    Intercepts the environment's step function and augments the reward
    using the pre-trained TimesNet's prediction of the Min-Max macro reward.

        r_t = max(r_env, r_timesnet_scaled[a_t])

    The TimesNet reward is **re-scaled** to match the environment reward
    magnitude before taking the max, preventing the augmented signal from
    completely overriding real financial feedback.

    Args:
        env:              Underlying Gymnasium environment.
        model_path:       Path to the pre-trained TimesNet .pth file.
        seq_len:          Lookback sequence length expected by TimesNet.
        num_features:     Number of input features per timestep.
                          Must match the saved model's embedding dimension.
        reward_scale:     Manual scale factor applied to r_timesnet before
                          max(). If None (default), it is estimated
                          automatically from the expert reward distribution.
    """

    def __init__(self, env, model_path: str, seq_len: int = 10,
                 num_features: int = None, reward_scale: float = None):
        super().__init__(env)

        if num_features is None:
            raise ValueError(
                "num_features must be specified explicitly. "
                "Pass len(feature_cols) from the training script."
            )

        self.seq_len      = seq_len
        self.num_features = num_features
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------
        # Load pre-trained TimesNet
        # ------------------------------------------------------------------
        timesnet = TimesNet(
            seq_len=seq_len,
            num_features=num_features,
            d_model=32,
            d_ff=64,
            e_layers=2,
            top_k=2,
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"TimesNet model not found at {model_path}\n"
                f"Run training/train_timesnet_reward.py first."
            )

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        timesnet.load_state_dict(state_dict)
        print(f"Loaded pre-trained TimesNet from {model_path}")

        # ------------------------------------------------------------------
        # Dimension safety check
        # ------------------------------------------------------------------
        saved_in_features = timesnet.embedding.weight.shape[1]
        if saved_in_features != num_features:
            raise RuntimeError(
                f"Dimension mismatch: TimesNet at '{model_path}' was trained with "
                f"{saved_in_features} input features, but the current environment "
                f"has {num_features} features.\n"
                f"Ensure the feature_mode used for training the TimesNet matches "
                f"the feature_mode used here."
            )

        timesnet.eval()

        # ------------------------------------------------------------------
        # Pre-compute rewards for the entire dataset (batched, fast)
        # ------------------------------------------------------------------
        unwrapped_env    = self.env.unwrapped
        total_steps      = len(unwrapped_env.features)
        raw_timesnet_rewards = np.zeros((total_steps, 3), dtype=np.float32)

        print(f"Precomputing TimesNet rewards for {total_steps} timesteps on {self.device}...")

        batch_size    = 1024
        valid_indices = []
        valid_seqs    = []

        for i in range(self.seq_len, total_steps):
            valid_indices.append(i)
            valid_seqs.append(unwrapped_env.features[i - self.seq_len : i])

        if valid_seqs:
            valid_seqs_arr = np.array(valid_seqs, dtype=np.float32)

            for i in range(0, len(valid_seqs_arr), batch_size):
                batch = torch.tensor(valid_seqs_arr[i : i + batch_size]).to(self.device)
                with torch.no_grad():
                    preds = timesnet(batch).cpu().numpy()
                # Assign predictions to the correct rows
                for j, idx in enumerate(valid_indices[i : i + len(preds)]):
                    raw_timesnet_rewards[idx] = preds[j]

        # ------------------------------------------------------------------
        # Compute reward_scale to align r_timesnet → r_env magnitude
        #
        # The environment reward is:
        #   r_env ≈ (equity × position × price_change%) / initial_capital
        #
        # On BTC/USDT hourly with ~0.5% daily vol → ~0.07% per hour:
        #   r_env ≈ 10_000 × 1 × 0.0007 / 10_000 = 0.0007
        #
        # The expert reward magnitude is ~0.1–2.0 (volatility-normalised).
        # We compute the actual ratio from the precomputed predictions.
        # ------------------------------------------------------------------
        if reward_scale is not None:
            self.reward_scale = reward_scale
            print(f"Using manual reward_scale = {self.reward_scale:.6f}")
        else:
            # Estimate env reward scale from environment characteristics
            initial_capital = unwrapped_env.initial_capital
            # Approximate expected |r_env| per step:
            # Use median absolute log-return of close prices as proxy for
            # per-step price move, times 1 (full position), normalised by
            # initial_capital.
            close_prices = unwrapped_env.close_prices
            log_rets = np.abs(np.diff(np.log(close_prices + 1e-10)))
            median_move = float(np.nanmedian(log_rets))
            env_reward_scale = median_move  # ≈ equity × 1 × move / initial_capital (normalised)

            # Expert reward median absolute value
            nonzero_mask = raw_timesnet_rewards != 0
            if nonzero_mask.any():
                expert_reward_scale = float(np.nanmedian(np.abs(raw_timesnet_rewards[nonzero_mask])))
            else:
                expert_reward_scale = 1.0

            if expert_reward_scale > 0:
                self.reward_scale = env_reward_scale / expert_reward_scale
            else:
                self.reward_scale = 1.0

            print(f"Auto reward_scale: env_scale={env_reward_scale:.6f}, "
                  f"expert_scale={expert_reward_scale:.6f}, "
                  f"ratio={self.reward_scale:.6f}")

        # Scale and store final precomputed rewards
        self.precomputed_rewards = raw_timesnet_rewards * self.reward_scale
        print("TimesNet precomputation complete. RL training will run at full speed.")

        # Free GPU memory
        del timesnet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        unwrapped_env = self.env.unwrapped
        current_step  = unwrapped_env.current_step

        if self.seq_len <= current_step < len(self.precomputed_rewards):
            r_timesnet   = self.precomputed_rewards[current_step][action]
            final_reward = max(float(reward), float(r_timesnet))
        else:
            final_reward = reward

        info['original_reward']  = reward
        info['augmented_reward'] = final_reward
        info['timesnet_reward']  = (
            float(self.precomputed_rewards[current_step][action])
            if self.seq_len <= current_step < len(self.precomputed_rewards)
            else 0.0
        )

        return obs, final_reward, terminated, truncated, info
