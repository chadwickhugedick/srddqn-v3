import os
import pandas as pd
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add root directory to python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.basic_crypto_env import BasicCryptoEnv

def make_env(df, feature_cols):
    def _init():
        env = BasicCryptoEnv(df, feature_cols, lookback_window=10)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        return env
    return _init

def main():
    data_path = "data/processed/BTCUSDT_1h_features.parquet"
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Please run data_pipeline/features.py first.")
        return
        
    df = pd.read_parquet(data_path)
    
    # Simple split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    feature_cols = [
        'returns', 'log_returns', 'volatility_20', 'rsi_14',
        'macd', 'macd_signal', 'macd_hist', 'dist_sma_20', 'dist_sma_50'
    ]
    
    env = DummyVecEnv([make_env(train_df, feature_cols) for _ in range(4)])
    eval_env = DummyVecEnv([make_env(val_df, feature_cols)])
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/best_model',
        log_path='./logs/results', 
        eval_freq=5000,
        deterministic=True, 
        render=False
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log="./logs/tensorboard/"
    )
    
    print("Starting Full PPO Training Baseline (2 Million Timesteps)...")
    print("Tip: Run `tensorboard --logdir logs/tensorboard` in another terminal to monitor progress.")
    # Train for enough timesteps to actually learn a policy
    model.learn(total_timesteps=500000, callback=eval_callback)
    
    model.save("logs/ppo_baseline_final")
    print("Training complete.")

if __name__ == "__main__":
    main()
