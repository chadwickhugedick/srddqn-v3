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
    mdd  = 0.0
    for value in cumulative_returns:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > mdd:
            mdd = dd
    return mdd


def evaluate_model(model, env):
    obs, info = env.reset()
    done      = False
    equities  = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        equities.append(info['equity'])
        done = terminated or truncated

    if not equities:
        return 0, 0, 0, 0

    initial_equity = (env.unwrapped.initial_capital
                      if hasattr(env, 'unwrapped') else env.initial_capital)
    final_equity   = equities[-1]

    cumulative_return = (final_equity - initial_equity) / initial_equity

    hours_per_year   = 24 * 365
    total_hours      = len(equities)
    years            = total_hours / hours_per_year
    annualized_return = (
        ((final_equity / initial_equity) ** (1 / years)) - 1 if years > 0 else 0
    )

    hourly_returns = pd.Series(equities).pct_change().dropna()
    mean_return    = hourly_returns.mean()
    std_return     = hourly_returns.std()
    sharpe_ratio   = (mean_return / std_return) * np.sqrt(hours_per_year) if std_return > 0 else 0.0

    mdd = calculate_mdd(equities)
    return cumulative_return, annualized_return, sharpe_ratio, mdd


def get_model_path(config, mode):
    template = config.get('paths', {}).get('timesnet_model',
                                           'logs/models/timesnet_reward_{mode}.pth')
    return template.replace('{mode}', mode)


def main(feature_mode: str = None, timestamp: str = None):
    config = load_config()

    mode = feature_mode or config.get('features', {}).get('mode', 'indicators')
    print(f"\n{'='*60}")
    print(f" Walk-Forward Validation  |  mode = {mode}")
    print(f"{'='*60}\n")

    features_df, _ = get_or_fetch_data(config, feature_mode=mode)

    if features_df.empty:
        print("Error: No data fetched or generated.")
        return

    df = features_df.copy()

    _exclude_base = ['open', 'high', 'low', 'close', 'volume',
                     'expert_trend', 'reward_0', 'reward_1', 'reward_2', 'symbol']
    exclude_cols = _exclude_base.copy()
    exclude_cols += [f'4h_{c}' for c in _exclude_base]
    exclude_cols += [f'1d_{c}' for c in _exclude_base]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    timesnet_path = get_model_path(config, mode)

    start_date = df.index.min()
    end_date   = df.index.max()

    current_train_start = start_date
    fold    = 1
    results = []

    print(f"Starting Walk-Forward Validation (12m train, 3m val) — mode: {mode}")
    print(f"Dataset bounds: {start_date} to {end_date}")

    from datetime import datetime
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wfv_run_dir = f"models/wfv_runs/{timestamp}_{mode}"
    os.makedirs(wfv_run_dir, exist_ok=True)

    while True:
        train_end = current_train_start + pd.DateOffset(months=12)
        val_end   = train_end           + pd.DateOffset(months=3)

        if val_end > end_date:
            break

        print(f"\n{'='*50}")
        print(f"FOLD {fold}  [{mode}]")
        print(f"Train: {current_train_start.date()} to {train_end.date()}")
        print(f"Val:   {train_end.date()} to {val_end.date()}")
        print(f"{'='*50}")

        train_df = df[(df.index >= current_train_start) & (df.index < train_end)].reset_index(drop=True)
        val_df   = df[(df.index >= train_end) & (df.index < val_end)].reset_index(drop=True)

        if len(train_df) < 100 or len(val_df) < 100:
            print("Not enough data in fold, skipping...")
            current_train_start += pd.DateOffset(months=3)
            continue

        lookback = config['environment']['lookback_window']
        fee      = config['environment']['fee_pct']
        hold_cost = config['environment'].get('holding_cost_pct', 0.00001)
        bankrupt  = config['environment'].get('bankruptcy_floor', 0.1)
        seq      = config['timesnet']['seq_len']

        train_env = CryptoMtfEnv(train_df, feature_cols, lookback_window=lookback,
                                 fee_pct=fee, holding_cost_pct=hold_cost,
                                 bankruptcy_floor=bankrupt)
        val_env   = CryptoMtfEnv(val_df,   feature_cols, lookback_window=lookback,
                                 fee_pct=fee, holding_cost_pct=hold_cost,
                                 bankruptcy_floor=bankrupt)

        train_env = SRDRLWrapper(train_env, model_path=timesnet_path,
                                 seq_len=seq, num_features=len(feature_cols))
        val_env   = SRDRLWrapper(val_env,   model_path=timesnet_path,
                                 seq_len=seq, num_features=len(feature_cols))

        train_env = Monitor(train_env)
        val_env   = Monitor(val_env)

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
            verbose=0,
        )

        model.learn(total_timesteps=50000)
        model.save(f"{wfv_run_dir}/srddqn_{mode}_fold_{fold}")

        cr, ar, sr, mdd = evaluate_model(model, val_env)
        print(f"Fold {fold} Metrics: CR={cr*100:.2f}%  AR={ar*100:.2f}%  SR={sr:.4f}  MDD={mdd*100:.2f}%")

        results.append({
            'mode':        mode,
            'fold':        fold,
            'train_start': current_train_start,
            'train_end':   train_end,
            'val_end':     val_end,
            'cr':          cr,
            'ar':          ar,
            'sr':          sr,
            'mdd':         mdd,
        })

        current_train_start += pd.DateOffset(months=3)
        fold += 1

    if results:
        results_df = pd.DataFrame(results)
        print(f"\n{'='*50}")
        print(f"WFV AVERAGE METRICS — mode: {mode}")
        print(f"{'='*50}")
        print(f"Average CR:  {results_df['cr'].mean() * 100:.2f}%")
        print(f"Average AR:  {results_df['ar'].mean() * 100:.2f}%")
        print(f"Average SR:  {results_df['sr'].mean():.4f}")
        print(f"Average MDD: {results_df['mdd'].mean() * 100:.2f}%")
        print(f"{'='*50}")

        wfv_path = f"{wfv_run_dir}/wfv_results_{mode}.csv"
        results_df.to_csv(wfv_path, index=False)
        print(f"Detailed results saved to {wfv_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None,
                        help="Feature mode (overrides config.yaml)")
    parser.add_argument("--timestamp", default=None,
                        help="Dynamic timestamp/run ID override")
    args = parser.parse_args()
    main(feature_mode=args.mode, timestamp=args.timestamp)
