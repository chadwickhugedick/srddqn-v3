import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.crypto_mtf_env import CryptoMtfEnv

def main():
    data_path = "data/processed/BTCUSDT_1h_MTF_features.parquet"
    model_path = "logs/best_model_srddqn/best_model.zip"
    timesnet_path = "logs/models/timesnet_reward_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        return
        
    df = pd.read_parquet(data_path)
    
    # Use the validation set
    train_size = int(len(df) * 0.8)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    # We will plot a subset of the validation set to make the markers visible
    # Let's plot the last 1000 hours (~41 days)
    plot_df = val_df.iloc[-1000:].reset_index(drop=True)
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'expert_trend', 'reward_0', 'reward_1', 'reward_2']
    exclude_cols += [f'4h_{c}' for c in exclude_cols] + [f'1d_{c}' for c in exclude_cols]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    base_env = CryptoMtfEnv(plot_df, feature_cols, lookback_window=10, fee_pct=0.001)
    env = SRDRLWrapper(base_env, model_path=timesnet_path, seq_len=10, num_features=len(feature_cols))
    
    model = DQN.load(model_path)
    
    obs, info = env.reset(options={'start_step': 10})
    
    prices = []
    positions = []
    
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        prices.append(info['price'])
        positions.append(info['position'])
        
        done = terminated or truncated

    # Detect position changes
    long_entries_x = []
    long_entries_y = []
    short_entries_x = []
    short_entries_y = []
    exits_x = []
    exits_y = []
    
    prev_pos = 0
    for i, pos in enumerate(positions):
        if pos != prev_pos:
            if pos == 1:
                long_entries_x.append(i)
                long_entries_y.append(prices[i])
            elif pos == -1:
                short_entries_x.append(i)
                short_entries_y.append(prices[i])
            elif pos == 0:
                exits_x.append(i)
                exits_y.append(prices[i])
        prev_pos = pos

    plt.figure(figsize=(16, 8))
    
    # Plot price
    plt.plot(prices, label='BTC/USDT Price', color='black', alpha=0.6, linewidth=1.5)
    
    # Plot Long entries
    plt.scatter(long_entries_x, long_entries_y, marker='^', color='green', s=150, zorder=5, label='Long Entry')
    
    # Plot Short entries
    plt.scatter(short_entries_x, short_entries_y, marker='v', color='red', s=150, zorder=5, label='Short Entry')
    
    # Plot Exits to Flat
    plt.scatter(exits_x, exits_y, marker='x', color='blue', s=100, zorder=5, label='Exit to Flat')
    
    plt.title('SRDDQN Prototype: Trading Entries & Exits (Last 1000 Hours)', fontsize=16)
    plt.xlabel('Timestep (Hours)', fontsize=12)
    plt.ylabel('Price (USDT)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    os.makedirs("logs/plots", exist_ok=True)
    save_path = "logs/plots/srddqn_trading_visualization.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()
