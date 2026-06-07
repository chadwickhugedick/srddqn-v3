import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

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

def evaluate_model(model, env):
    obs, info = env.reset()
    done = False
    equities = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        equities.append(info['equity'])
        done = terminated or truncated

    if not equities:
        return 0, 0, 0, 0

    initial_equity = env.unwrapped.initial_capital if hasattr(env, 'unwrapped') else env.initial_capital
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
    
    return cumulative_return, annualized_return, sharpe_ratio, mdd

def main():
    config = load_config()
    
    # 1. Fetch and process data automatically based on config
    features_df, _ = get_or_fetch_data(config)
    
    if features_df.empty:
        print("Error: No data fetched or generated.")
        return
        
    df = features_df.copy()
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'expert_trend', 'reward_0', 'reward_1', 'reward_2', 'symbol']
    exclude_cols += [f'4h_{c}' for c in exclude_cols] + [f'1d_{c}' for c in exclude_cols]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    model_path = config['paths']['timesnet_model'] if 'paths' in config and 'timesnet_model' in config['paths'] else "logs/models/timesnet_reward_model.pth"

    
    start_date = df.index.min()
    end_date = df.index.max()
    
    current_train_start = start_date
    fold = 1
    
    results = []
    
    print(f"Starting Walk-Forward Validation (12m train, 3m val)")
    print(f"Dataset bounds: {start_date} to {end_date}")
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/wfv_models", exist_ok=True)

    while True:
        train_end = current_train_start + pd.DateOffset(months=12)
        val_end = train_end + pd.DateOffset(months=3)
        
        if val_end > end_date:
            break
            
        print(f"\n{'='*50}")
        print(f"FOLD {fold}")
        print(f"Train: {current_train_start.date()} to {train_end.date()}")
        print(f"Val:   {train_end.date()} to {val_end.date()}")
        print(f"{'='*50}")
        
        train_df = df[(df.index >= current_train_start) & (df.index < train_end)].reset_index(drop=True)
        val_df = df[(df.index >= train_end) & (df.index < val_end)].reset_index(drop=True)
        
        if len(train_df) < 100 or len(val_df) < 100:
            print("Not enough data in fold, skipping...")
            current_train_start += pd.DateOffset(months=3)
            continue
            
        lookback = config['environment']['lookback_window']
        fee = config['environment']['fee_pct']
        seq = config['timesnet']['seq_len']
        
        train_env = CryptoMtfEnv(train_df, feature_cols, lookback_window=lookback, fee_pct=fee)
        val_env = CryptoMtfEnv(val_df, feature_cols, lookback_window=lookback, fee_pct=fee)
        
        train_env = SRDRLWrapper(train_env, model_path=model_path, seq_len=seq, num_features=len(feature_cols))
        val_env = SRDRLWrapper(val_env, model_path=model_path, seq_len=seq, num_features=len(feature_cols))
        
        train_env = Monitor(train_env)
        val_env = Monitor(val_env)
        
        rl = config['rl_agent']
        model = DQN(
            "MlpPolicy", 
            train_env, 
            learning_rate=rl['learning_rate'], 
            buffer_size=rl['buffer_size'],
            learning_starts=1000,
            batch_size=rl['batch_size'],
            tau=rl['tau'],
            gamma=rl['gamma'],
            train_freq=4,
            gradient_steps=1,
            target_update_interval=rl['target_update_interval'],
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            verbose=0
        )
        
        # Train for a shorter period per fold for the sake of WFV
        model.learn(total_timesteps=50000)
        
        model.save(f"logs/wfv_models/srddqn_fold_{fold}")
        
        cr, ar, sr, mdd = evaluate_model(model, val_env)
        print(f"Fold {fold} Metrics: CR={cr*100:.2f}%, AR={ar*100:.2f}%, SR={sr:.4f}, MDD={mdd*100:.2f}%")
        
        results.append({
            'fold': fold,
            'train_start': current_train_start,
            'train_end': train_end,
            'val_end': val_end,
            'cr': cr,
            'ar': ar,
            'sr': sr,
            'mdd': mdd
        })
        
        current_train_start += pd.DateOffset(months=3)
        fold += 1

    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("WALK-FORWARD VALIDATION AVERAGE METRICS")
        print("="*50)
        print(f"Average CR:  {results_df['cr'].mean() * 100:.2f}%")
        print(f"Average AR:  {results_df['ar'].mean() * 100:.2f}%")
        print(f"Average SR:  {results_df['sr'].mean():.4f}")
        print(f"Average MDD: {results_df['mdd'].mean() * 100:.2f}%")
        print("="*50)
        
        results_df.to_csv("logs/wfv_results.csv", index=False)
        print("Detailed results saved to logs/wfv_results.csv")

if __name__ == "__main__":
    main()
