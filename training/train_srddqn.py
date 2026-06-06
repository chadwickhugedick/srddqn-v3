import os
import sys
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.crypto_mtf_env import CryptoMtfEnv
from envs.srddqn_wrapper import SRDRLWrapper

def main():
    data_path = "data/processed/BTCUSDT_1h_MTF_features.parquet"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    
    # Train/Val split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'expert_trend', 'reward_0', 'reward_1', 'reward_2']
    exclude_cols += [f'4h_{c}' for c in exclude_cols] + [f'1d_{c}' for c in exclude_cols]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Training SRDDQN with {len(feature_cols)} MTF features.")
    
    # Path to pre-trained TimesNet
    model_path = "logs/models/timesnet_reward_model.pth"
    
    # Create realistic MTF environments with fees
    train_env = CryptoMtfEnv(train_df, feature_cols, lookback_window=10, fee_pct=0.001)
    val_env = CryptoMtfEnv(val_df, feature_cols, lookback_window=10, fee_pct=0.001)
    
    # Wrap with SRDRL Self-Rewarding Mechanism
    train_env = SRDRLWrapper(train_env, model_path=model_path, seq_len=10, num_features=len(feature_cols))
    val_env = SRDRLWrapper(val_env, model_path=model_path, seq_len=10, num_features=len(feature_cols))
    
    # Wrap in Monitor for SB3 logging
    os.makedirs("logs", exist_ok=True)
    train_env = Monitor(train_env)
    val_env = Monitor(val_env)
    
    # Callback to save best model
    os.makedirs("logs/best_model_srddqn", exist_ok=True)
    eval_callback = EvalCallback(
        val_env, 
        best_model_save_path='./logs/best_model_srddqn/',
        log_path='./logs/results_srddqn/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    # Initialize DQN model (which is Double DQN under the hood in SB3)
    model = DQN(
        "MlpPolicy", 
        train_env, 
        learning_rate=3e-4, 
        buffer_size=100000,
        learning_starts=10000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/tensorboard_srddqn/"
    )
    
    print("Starting SRDDQN Training (500k Timesteps)...")
    print("Tip: Run `tensorboard --logdir logs/tensorboard_srddqn` in another terminal.")
    model.learn(total_timesteps=500000, callback=eval_callback)
    
    model.save("logs/srddqn_final")
    print("Training complete.")

if __name__ == "__main__":
    main()
