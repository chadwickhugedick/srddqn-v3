import os
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.crypto_mtf_env import CryptoMtfEnv

def main():
    data_path = "data/processed/BTCUSDT_1h_MTF_features.parquet"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}. Please run mtf_aligner.py and features_mtf.py first.")
        return
        
    df = pd.read_parquet(data_path)
    
    # Train/Val split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    # We load ALL columns except for open/high/low/close/volume
    # So we want all the indicator columns
    exclude_cols = ['open', 'high', 'low', 'close', 'volume']
    exclude_cols += [f'4h_{c}' for c in exclude_cols] + [f'1d_{c}' for c in exclude_cols]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"Training with {len(feature_cols)} MTF features.")
    
    # Create realistic MTF environments with fees
    train_env = CryptoMtfEnv(train_df, feature_cols, lookback_window=10, fee_pct=0.001)
    val_env = CryptoMtfEnv(val_df, feature_cols, lookback_window=10, fee_pct=0.001)
    
    # Wrap in Monitor for SB3 logging
    os.makedirs("logs", exist_ok=True)
    train_env = Monitor(train_env)
    val_env = Monitor(val_env)
    
    # Callback to save best model
    os.makedirs("logs/best_model_mtf", exist_ok=True)
    eval_callback = EvalCallback(
        val_env, 
        best_model_save_path='./logs/best_model_mtf/',
        log_path='./logs/results_mtf/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    # Initialize PPO model
    # We use MLP policy just for benchmarking the MTF features natively
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log="./logs/tensorboard_mtf/"
    )
    
    print("Starting PPO Training on MTF Environment (500k Timesteps)...")
    print("Tip: Run `tensorboard --logdir logs/tensorboard_mtf` in another terminal.")
    model.learn(total_timesteps=500000, callback=eval_callback)
    
    model.save("logs/ppo_mtf_final")
    print("Training complete.")

if __name__ == "__main__":
    main()
