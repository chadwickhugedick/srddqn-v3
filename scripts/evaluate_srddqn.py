import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.crypto_mtf_env import CryptoMtfEnv
from envs.srddqn_wrapper import SRDRLWrapper
from utils.config import load_config
from data_pipeline.fetch_data import get_or_fetch_data

def calculate_mdd(cumulative_returns):
    if len(cumulative_returns) == 0:
        return 0.0
    peak = cumulative_returns[0]
    mdd = 0.0
    for value in cumulative_returns:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > mdd:
            mdd = dd
    return mdd

def main():
    config = load_config()
    
    # 1. Fetch and process data automatically based on config
    features_df, _ = get_or_fetch_data(config)
    
    if features_df.empty:
        print("Error: No data fetched or generated.")
        return
        
    df = features_df.copy()
    
    model_path = config['paths']['best_srddqn_model'] if 'paths' in config and 'best_srddqn_model' in config['paths'] else "logs/best_model_srddqn/best_model.zip"
    timesnet_path = config['paths']['timesnet_model'] if 'paths' in config and 'timesnet_model' in config['paths'] else "logs/models/timesnet_reward_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        return
    
    train_size = int(len(df) * 0.8)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'expert_trend', 'reward_0', 'reward_1', 'reward_2', 'symbol']
    exclude_cols += [f'4h_{c}' for c in exclude_cols] + [f'1d_{c}' for c in exclude_cols]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    lookback = config['environment']['lookback_window']
    fee = config['environment']['fee_pct']
    seq = config['timesnet']['seq_len']
    
    base_env = CryptoMtfEnv(val_df, feature_cols, lookback_window=lookback, fee_pct=fee)
    env = SRDRLWrapper(base_env, model_path=timesnet_path, seq_len=seq, num_features=len(feature_cols))
    
    model = DQN.load(model_path)
    
    obs, info = env.reset()
    
    rewards = []
    positions = []
    prices = []
    equities = []
    
    print("Evaluating SRDDQN on validation environment...")
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        positions.append(info['position'])
        prices.append(info['price'])
        equities.append(info['equity'])
        
        done = terminated or truncated

    initial_equity = base_env.initial_capital
    final_equity = equities[-1]
    
    cumulative_return = (final_equity - initial_equity) / initial_equity
    
    hours_per_year = 24 * 365
    total_hours = len(equities)
    years = total_hours / hours_per_year
    annualized_return = ((final_equity / initial_equity) ** (1 / years)) - 1 if years > 0 else 0
    
    hourly_returns = pd.Series(equities).pct_change().dropna()
    mean_return = hourly_returns.mean()
    std_return = hourly_returns.std()
    
    if std_return > 0:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(hours_per_year)
    else:
        sharpe_ratio = 0.0
        
    mdd = calculate_mdd(equities)
    
    print("\n" + "="*40)
    print("FINAL BENCHMARK METRICS (Phase 5 - SRDDQN)")
    print("="*40)
    print(f"Cumulative Return (CR):  {cumulative_return * 100:.2f}%")
    print(f"Annualized Return (AR):  {annualized_return * 100:.2f}%")
    print(f"Sharpe Ratio (SR):       {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown (MDD):  {mdd * 100:.2f}%")
    print(f"Final Equity:            ${final_equity:.2f} (from ${initial_equity:.2f})")
    print("="*40)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(prices, label="Price", color='black')
    ax1.set_title("BTC/USDT Price (Validation)")
    ax1.legend()
    
    ax2.plot(positions, label="Position (-1=Short, 0=Flat, 1=Long)", color='blue')
    ax2.set_title("SRDDQN Positions")
    ax2.set_yticks([-1, 0, 1])
    ax2.legend()
    
    ax3.plot(equities, label="Portfolio Equity ($)", color='green')
    ax3.set_title("SRDDQN Portfolio Equity")
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs("logs/plots", exist_ok=True)
    plt.savefig("logs/plots/evaluation_srddqn_results.png")
    print("\nEvaluation plot saved to logs/plots/evaluation_srddqn_results.png")

if __name__ == "__main__":
    main()
