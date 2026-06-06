import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.basic_crypto_env import BasicCryptoEnv

def main():
    data_path = "data/processed/BTCUSDT_1h_features.parquet"
    model_path = "logs/best_model/best_model.zip"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run training first.")
        return
        
    df = pd.read_parquet(data_path)
    
    # Use the validation split (last 20%)
    train_size = int(len(df) * 0.8)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    feature_cols = [
        'returns', 'log_returns', 'volatility_20', 'rsi_14',
        'macd', 'macd_signal', 'macd_hist', 'dist_sma_20', 'dist_sma_50'
    ]
    
    env = BasicCryptoEnv(val_df, feature_cols, lookback_window=10)
    model = PPO.load(model_path)
    
    obs, info = env.reset()
    
    rewards = []
    positions = []
    prices = []
    
    print("Evaluating model on validation data...")
    done = False
    
    # Run a full episode
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        positions.append(info['position'])
        prices.append(info['price'])
        
        done = terminated or truncated

    # Calculate cumulative returns
    cumulative_returns = np.cumsum(rewards)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(prices, label="Price", color='black')
    ax1.set_title("BTC/USDT Price (Validation)")
    ax1.legend()
    
    ax2.plot(positions, label="Position (-1=Short, 0=Flat, 1=Long)", color='blue')
    ax2.set_title("Agent Positions")
    ax2.set_yticks([-1, 0, 1])
    ax2.legend()
    
    ax3.plot(cumulative_returns, label="Cumulative Log Return", color='green')
    ax3.set_title("Agent Cumulative Log Return")
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs("logs/plots", exist_ok=True)
    plt.savefig("logs/plots/evaluation_results.png")
    print("Evaluation complete. Results saved to logs/plots/evaluation_results.png")
    
    total_return_pct = (np.exp(cumulative_returns[-1]) - 1) * 100
    print(f"Total Cumulative Return on Validation Set: {total_return_pct:.2f}%")

if __name__ == "__main__":
    main()
