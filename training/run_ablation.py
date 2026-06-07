"""
run_ablation.py
---------------
Automated 5-mode ablation runner for SRDDQN.

Sequentially trains TimesNet + SRDDQN for all five feature modes and
writes a consolidated results table to logs/ablation_results.csv.

Usage:
    python training/run_ablation.py                     # all 5 modes
    python training/run_ablation.py --modes indicators raw_ratios
    python training/run_ablation.py --skip-completed    # skip modes already in results CSV
"""

import os
import sys
import argparse
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from data_pipeline.fetch_data import get_or_fetch_data

ALL_MODES = [
    "indicators",
    "raw_ratios",
    "raw_prices",
    "combined_ratios",
    "combined_prices",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate(model, env):
    """Run deterministic rollout and return (cr, ar, sharpe, mdd)."""
    import numpy as np

    obs, _ = env.reset()
    done   = False
    equities = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        equities.append(info.get('equity', 0.0))
        done = terminated or truncated

    if not equities:
        return 0.0, 0.0, 0.0, 0.0

    initial = env.unwrapped.initial_capital
    final   = equities[-1]
    cr      = (final - initial) / initial

    hours_yr = 24 * 365
    years    = len(equities) / hours_yr
    ar       = ((final / initial) ** (1 / years) - 1) if years > 0 else 0.0

    rets  = pd.Series(equities).pct_change().dropna()
    sr    = (rets.mean() / rets.std() * (hours_yr ** 0.5)) if rets.std() > 0 else 0.0

    peak  = equities[0]
    mdd   = 0.0
    for v in equities:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > mdd:
            mdd = dd

    return cr, ar, sr, mdd


def _run_mode(config, mode: str, timestamp: str):
    """Train TimesNet + SRDDQN for one mode and return metrics dict."""
    from training.train_timesnet_reward import main as train_timesnet
    from training.train_srddqn import main as train_srddqn

    print(f"\n{'#'*60}")
    print(f"# MODE: {mode}")
    print(f"{'#'*60}")

    t0 = time.time()

    # ── Phase 1: TimesNet supervised pre-training ───────────────────────────
    print(f"\n[1/2] Training TimesNet reward network ({mode})...")
    timesnet_metrics = train_timesnet(feature_mode=mode)
    best_val_mse     = timesnet_metrics['best_val_mse'] if timesnet_metrics else None

    # ── Phase 2: SRDDQN reinforcement learning ──────────────────────────────
    print(f"\n[2/2] Training SRDDQN agent ({mode})...")
    train_srddqn(feature_mode=mode, timestamp=timestamp)

    elapsed = (time.time() - t0) / 60

    # ── Quick evaluation on val set ─────────────────────────────────────────
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from envs.crypto_mtf_env import CryptoMtfEnv
    from envs.srddqn_wrapper import SRDRLWrapper

    features_df, _ = get_or_fetch_data(config, feature_mode=mode)
    df = features_df.copy()

    _exclude_base = ['open', 'high', 'low', 'close', 'volume',
                     'expert_trend', 'reward_0', 'reward_1', 'reward_2', 'symbol']
    exclude_cols  = _exclude_base + [f'4h_{c}' for c in _exclude_base] + [f'1d_{c}' for c in _exclude_base]
    feature_cols  = [c for c in df.columns if c not in exclude_cols]

    val_df = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)

    timesnet_path_tmpl = config.get('paths', {}).get('timesnet_model',
                                                      'logs/models/timesnet_reward_{mode}.pth')
    timesnet_path = timesnet_path_tmpl.replace('{mode}', mode)

    run_dir = f"{timestamp}_{mode}"
    srddqn_path = f"models/runs/{run_dir}/srddqn_final_{mode}"

    lookback = config['environment']['lookback_window']
    fee      = config['environment']['fee_pct']
    seq      = config['timesnet']['seq_len']

    val_env = CryptoMtfEnv(val_df, feature_cols, lookback_window=lookback, fee_pct=fee)
    val_env = SRDRLWrapper(val_env, model_path=timesnet_path,
                           seq_len=seq, num_features=len(feature_cols))
    val_env = Monitor(val_env)

    model = DQN.load(srddqn_path, env=val_env)
    cr, ar, sr, mdd = _evaluate(model, val_env)

    return {
        'mode':         mode,
        'best_val_mse': best_val_mse,
        'cr':           cr,
        'ar':           ar,
        'sr':           sr,
        'mdd':          mdd,
        'elapsed_min':  elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run the full 5-mode SRDDQN ablation study.")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES,
                        choices=ALL_MODES,
                        help="Subset of modes to run (default: all 5)")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip modes already present in the results CSV")
    args = parser.parse_args()

    config = load_config()
    os.makedirs("logs/models", exist_ok=True)
    os.makedirs("logs/plots",  exist_ok=True)

    results_path = config.get('paths', {}).get('ablation_results', 'logs/ablation_results.csv')
    completed_modes = set()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.skip_completed and os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        completed_modes = set(existing['mode'].tolist())
        print(f"Skipping already-completed modes: {sorted(completed_modes)}")

    all_results = []
    if os.path.exists(results_path):
        all_results = pd.read_csv(results_path).to_dict('records')

    modes_to_run = [m for m in args.modes if m not in completed_modes]

    if not modes_to_run:
        print("Nothing to run — all specified modes are already completed.")
        _print_table(pd.DataFrame(all_results))
        return

    print(f"\nRunning modes: {modes_to_run} with run timestamp: {timestamp}")
    total_start = time.time()

    for mode in modes_to_run:
        try:
            result = _run_mode(config, mode, timestamp)
            all_results.append(result)

            # Save incrementally after each mode in case of crash
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(results_path, index=False)
            
            # Save a copy to the run-specific log directory
            run_log_dir = f"logs/runs/{timestamp}_ablation"
            os.makedirs(run_log_dir, exist_ok=True)
            results_df.to_csv(os.path.join(run_log_dir, "ablation_results.csv"), index=False)
            
            print(f"\n✓ Mode '{mode}' complete. Results saved to {results_path}")

        except Exception as e:
            print(f"\n✗ Mode '{mode}' FAILED: {e}")
            all_results.append({'mode': mode, 'error': str(e)})
            pd.DataFrame(all_results).to_csv(results_path, index=False)

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"Ablation complete in {total_elapsed:.1f} minutes.")
    print(f"{'='*60}")

    results_df = pd.DataFrame(all_results)
    _print_table(results_df)


def _print_table(df):
    """Pretty-print the results table."""
    cols = ['mode', 'best_val_mse', 'cr', 'ar', 'sr', 'mdd', 'elapsed_min']
    cols = [c for c in cols if c in df.columns]
    print("\n" + df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
