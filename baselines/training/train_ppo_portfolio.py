import os
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.crypto_portfolio_env import CryptoPortfolioEnv

def main():
    data_path = "data/processed/BTCUSDT_1h_features.parquet"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}. Please run data_pipeline first.")
        return
        
    df = pd.read_parquet(data_path)
    
    # Train/Val split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    feature_cols = [
        'returns', 'log_returns', 'volatility_20', 'rsi_14',
        'macd', 'macd_signal', 'macd_hist', 'dist_sma_20', 'dist_sma_50'
    ]
    
    # Create realistic environments with fees
    train_env = CryptoPortfolioEnv(train_df, feature_cols, lookback_window=10, fee_pct=0.001)
    val_env = CryptoPortfolioEnv(val_df, feature_cols, lookback_window=10, fee_pct=0.001)
    
    # Wrap in Monitor for SB3 logging
    os.makedirs("logs", exist_ok=True)
    train_env = Monitor(train_env)
    val_env = Monitor(val_env)
    
    # Callback to save best model
    os.makedirs("logs/best_model_portfolio", exist_ok=True)
    eval_callback = EvalCallback(
        val_env, 
        best_model_save_path='./logs/best_model_portfolio/',
        log_path='./logs/results_portfolio/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    # Initialize PPO model
    # We use MLP policy just for benchmarking
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log="./logs/tensorboard_portfolio/"
    )
    
    print("Starting PPO Training on Realistic Portfolio Environment (500k Timesteps)...")
    print("Tip: Run `tensorboard --logdir logs/tensorboard_portfolio` in another terminal.")
    model.learn(total_timesteps=500000, callback=eval_callback)
    
    model.save("logs/ppo_portfolio_final")
    print("Training complete.")

if __name__ == "__main__":
    main()
