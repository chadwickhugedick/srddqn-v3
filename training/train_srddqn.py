import os
import sys
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.crypto_mtf_env import CryptoMtfEnv
from envs.srddqn_wrapper import SRDRLWrapper
from utils.config import load_config
from data_pipeline.fetch_data import get_or_fetch_data
from utils.callbacks import FinancialMetricsCallback

def main():
    config = load_config()
    
    # 1. Fetch and process data automatically based on config
    features_df, _ = get_or_fetch_data(config)
    
    if features_df.empty:
        print("Error: No data fetched or generated.")
        return
        
    df = features_df.copy()
    
    # Train/Val split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'expert_trend', 'reward_0', 'reward_1', 'reward_2', 'symbol']
    exclude_cols += [f'4h_{c}' for c in exclude_cols] + [f'1d_{c}' for c in exclude_cols]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Training SRDDQN with {len(feature_cols)} MTF features.")
    
    # Path to pre-trained TimesNet
    model_path = config['paths']['timesnet_model'] if 'paths' in config and 'timesnet_model' in config['paths'] else "logs/models/timesnet_reward_model.pth"
    
    lookback = config['environment']['lookback_window']
    fee = config['environment']['fee_pct']
    seq = config['timesnet']['seq_len']
    
    # Create realistic MTF environments with fees
    train_env = CryptoMtfEnv(train_df, feature_cols, lookback_window=lookback, fee_pct=fee)
    val_env = CryptoMtfEnv(val_df, feature_cols, lookback_window=lookback, fee_pct=fee)
    
    # Wrap with SRDRL Self-Rewarding Mechanism
    train_env = SRDRLWrapper(train_env, model_path=model_path, seq_len=seq, num_features=len(feature_cols))
    val_env = SRDRLWrapper(val_env, model_path=model_path, seq_len=seq, num_features=len(feature_cols))
    
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
    
    financial_callback = FinancialMetricsCallback()
    callbacks = CallbackList([eval_callback, financial_callback])
    
    rl = config['rl_agent']
    model = DQN(
        "MlpPolicy", 
        train_env, 
        learning_rate=rl['learning_rate'], 
        buffer_size=rl['buffer_size'],
        learning_starts=10000,
        batch_size=rl['batch_size'],
        tau=rl['tau'],
        gamma=rl['gamma'],
        train_freq=4,
        gradient_steps=1,
        target_update_interval=rl['target_update_interval'],
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/tensorboard_srddqn/"
    )
    
    print("Starting SRDDQN Training (500k Timesteps)...")
    print("Tip: Run `tensorboard --logdir logs/tensorboard_srddqn` in another terminal.")
    model.learn(total_timesteps=500000, callback=callbacks)
    
    model.save("logs/srddqn_final")
    print("Training complete.")

if __name__ == "__main__":
    main()
