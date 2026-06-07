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


def get_model_path(config, mode):
    template = config.get('paths', {}).get('timesnet_model',
                                           'logs/models/timesnet_reward_{mode}.pth')
    return template.replace('{mode}', mode)


def get_save_path(config, mode, run_dir=None):
    if run_dir:
        return f"models/runs/{run_dir}/srddqn_final_{mode}"
    template = config.get('paths', {}).get('srddqn_model',
                                           'logs/models/srddqn_final_{mode}')
    return template.replace('{mode}', mode)


def main(feature_mode: str = None, timestamp: str = None):
    config = load_config()

    mode = feature_mode or config.get('features', {}).get('mode', 'indicators')
    print(f"\n{'='*60}")
    print(f" Training SRDDQN Agent  |  mode = {mode}")
    print(f"{'='*60}\n")

    # 1. Fetch / load data
    features_df, _ = get_or_fetch_data(config, feature_mode=mode)

    if features_df.empty:
        print("Error: No data fetched or generated.")
        return

    df = features_df.copy()

    # 2. Train/Val split (80/20 chronological)
    train_size = int(len(df) * 0.8)
    train_df   = df.iloc[:train_size].reset_index(drop=True)
    val_df     = df.iloc[train_size:].reset_index(drop=True)

    # 3. Determine feature columns (exclude raw OHLCV, labels, metadata)
    _exclude_base = ['open', 'high', 'low', 'close', 'volume',
                     'expert_trend', 'reward_0', 'reward_1', 'reward_2', 'symbol']
    exclude_cols = _exclude_base.copy()
    exclude_cols += [f'4h_{c}' for c in _exclude_base]
    exclude_cols += [f'1d_{c}' for c in _exclude_base]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Training SRDDQN with {len(feature_cols)} features [{mode}].")

    # 4. Resolve model paths
    timesnet_path = get_model_path(config, mode)

    from datetime import datetime
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{timestamp}_{mode}"
    save_path = get_save_path(config, mode, run_dir=run_dir)

    lookback = config['environment']['lookback_window']
    fee      = config['environment']['fee_pct']
    hold_cost = config['environment'].get('holding_cost_pct', 0.00001)
    bankrupt  = config['environment'].get('bankruptcy_floor', 0.1)
    seq      = config['timesnet']['seq_len']

    # 5. Build environments
    train_env = CryptoMtfEnv(train_df, feature_cols, lookback_window=lookback,
                             fee_pct=fee, holding_cost_pct=hold_cost,
                             bankruptcy_floor=bankrupt)
    val_env   = CryptoMtfEnv(val_df,   feature_cols, lookback_window=lookback,
                             fee_pct=fee, holding_cost_pct=hold_cost,
                             bankruptcy_floor=bankrupt)

    # Wrap with SRDRL self-rewarding mechanism
    train_env = SRDRLWrapper(train_env, model_path=timesnet_path,
                             seq_len=seq, num_features=len(feature_cols))
    val_env   = SRDRLWrapper(val_env,   model_path=timesnet_path,
                             seq_len=seq, num_features=len(feature_cols))

    train_env = Monitor(train_env)
    val_env   = Monitor(val_env)

    # 6. Callbacks
    best_model_dir = f"models/runs/{run_dir}/best_model"
    results_dir    = f"logs/runs/{run_dir}/eval"
    tb_log_dir     = f"logs/runs/{run_dir}/tb"

    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=best_model_dir,
        log_path=results_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    financial_callback = FinancialMetricsCallback()
    callbacks = CallbackList([eval_callback, financial_callback])

    # 7. DQN agent
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
        tensorboard_log=tb_log_dir,
    )

    print(f"Starting SRDDQN Training (500k timesteps) — mode: {mode}")
    print(f"Tip: Run `tensorboard --logdir {tb_log_dir}` in another terminal.")
    model.learn(total_timesteps=500000, callback=callbacks)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None,
                        help="Feature mode (overrides config.yaml)")
    parser.add_argument("--timestamp", default=None,
                        help="Dynamic timestamp/run ID override")
    args = parser.parse_args()
    main(feature_mode=args.mode, timestamp=args.timestamp)
