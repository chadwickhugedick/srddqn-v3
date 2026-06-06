import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.crypto_portfolio_env import CryptoPortfolioEnv

def calculate_mdd(cumulative_returns):
    """Calculate Maximum Drawdown"""
    if len(cumulative_returns) == 0:
        return 0.0
    
    # cumulative_returns is a series of portfolio equity curves (or returns)
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
    data_path = "data/processed/BTCUSDT_1h_features.parquet"
    model_path = "logs/best_model_portfolio/best_model.zip"
    
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
    
    env = CryptoPortfolioEnv(val_df, feature_cols, lookback_window=10, fee_pct=0.001)
    model = PPO.load(model_path)
    
    obs, info = env.reset()
    
    rewards = []
    positions = []
    prices = []
    equities = []
    
    print("Evaluating model on realistic validation environment...")
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        positions.append(info['position'])
        prices.append(info['price'])
        equities.append(info['equity'])
        
        done = terminated or truncated

    # Calculate metrics
    initial_equity = env.initial_capital
    final_equity = equities[-1]
    
    cumulative_return = (final_equity - initial_equity) / initial_equity
    
    # Annualized Return (assuming 1h data)
    hours_per_year = 24 * 365
    total_hours = len(equities)
    years = total_hours / hours_per_year
    annualized_return = ((final_equity / initial_equity) ** (1 / years)) - 1 if years > 0 else 0
    
    # Sharpe Ratio (using simple hourly returns, assuming risk-free rate is 0)
    hourly_returns = pd.Series(equities).pct_change().dropna()
    mean_return = hourly_returns.mean()
    std_return = hourly_returns.std()
    
    if std_return > 0:
        # Annualize sharpe ratio
        sharpe_ratio = (mean_return / std_return) * np.sqrt(hours_per_year)
    else:
        sharpe_ratio = 0.0
        
    mdd = calculate_mdd(equities)
    
    # Print Metrics
    print("\n" + "="*40)
    print("BENCHMARK METRICS (Phase 2)")
    print("="*40)
    print(f"Cumulative Return (CR):  {cumulative_return * 100:.2f}%")
    print(f"Annualized Return (AR):  {annualized_return * 100:.2f}%")
    print(f"Sharpe Ratio (SR):       {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown (MDD):  {mdd * 100:.2f}%")
    print(f"Final Equity:            ${final_equity:.2f} (from ${initial_equity:.2f})")
    print("="*40)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(prices, label="Price", color='black')
    ax1.set_title("BTC/USDT Price (Validation)")
    ax1.legend()
    
    ax2.plot(positions, label="Position (-1=Short, 0=Flat, 1=Long)", color='blue')
    ax2.set_title("Agent Positions")
    ax2.set_yticks([-1, 0, 1])
    ax2.legend()
    
    ax3.plot(equities, label="Portfolio Equity ($)", color='green')
    ax3.set_title("Agent Portfolio Equity")
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs("logs/plots", exist_ok=True)
    plt.savefig("logs/plots/evaluation_portfolio_results.png")
    print("\nEvaluation plot saved to logs/plots/evaluation_portfolio_results.png")

if __name__ == "__main__":
    main()
