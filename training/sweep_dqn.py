import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.crypto_mtf_env import CryptoMtfEnv
from envs.srddqn_wrapper import SRDRLWrapper

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
        
    return cumulative_return, annualized_return, sharpe_ratio

def main():
    wandb.init(sync_tensorboard=True)
    config = wandb.config
    
    data_path = "data/processed/BTCUSDT_1h_MTF_features.parquet"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'expert_trend', 'reward_0', 'reward_1', 'reward_2']
    exclude_cols += [f'4h_{c}' for c in exclude_cols] + [f'1d_{c}' for c in exclude_cols]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    model_path = "logs/models/timesnet_reward_model.pth"
    
    train_env = CryptoMtfEnv(train_df, feature_cols, lookback_window=10, fee_pct=0.001)
    val_env = CryptoMtfEnv(val_df, feature_cols, lookback_window=10, fee_pct=0.001)
    
    train_env = SRDRLWrapper(train_env, model_path=model_path, seq_len=10, num_features=len(feature_cols))
    val_env = SRDRLWrapper(val_env, model_path=model_path, seq_len=10, num_features=len(feature_cols))
    
    train_env = Monitor(train_env)
    
    model = DQN(
        "MlpPolicy", 
        train_env, 
        learning_rate=config.learning_rate, 
        buffer_size=config.buffer_size,
        learning_starts=10000,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=config.target_update_interval,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=0,
        tensorboard_log=f"logs/wandb_tensorboard/"
    )
    
    model.learn(total_timesteps=150000)
    
    cr, ar, sr = evaluate_model(model, val_env)
    
    wandb.log({
        "cumulative_return": cr,
        "annualized_return": ar,
        "sharpe_ratio": sr
    })

if __name__ == "__main__":
    main()
