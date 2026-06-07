import os
import json
import sys
import pandas as pd
import ccxt
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from data_pipeline.mtf_aligner import create_mtf_dataset
from data_pipeline.features_mtf import process_mtf_data_for_mode
from data_pipeline.expert_labels import create_expert_rewards

VALID_MODES = ["indicators", "raw_ratios", "raw_prices", "combined_ratios", "combined_prices"]


def fetch_ohlcv(exchange_id, symbol, timeframe, since, limit=1000):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})

    all_ohlcv = []
    current_since = since

    print(f"Fetching {symbol} from {exchange_id} starting at {datetime.utcfromtimestamp(since/1000)}...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1
            print(f"Fetched {len(ohlcv)} candles. Last date: {datetime.utcfromtimestamp(ohlcv[-1][0]/1000)}")

            if len(ohlcv) < limit:
                break

            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def get_or_fetch_data(config=None, feature_mode: str = None, force_refresh: bool = False):
    """Fetch raw data, run MTF alignment, compute features, and generate expert labels.

    Args:
        config:       Loaded config dict. If None, loaded from disk.
        feature_mode: One of VALID_MODES. Falls back to config['features']['mode']
                      then 'indicators' if unset.
        force_refresh: Re-fetch raw data even if cached parquets exist.

    Returns:
        (features_df, expert_df) — concatenated across all configured symbols.
    """
    if config is None:
        config = load_config()

    # Resolve feature mode
    if feature_mode is None:
        feature_mode = config.get('features', {}).get('mode', 'indicators')

    if feature_mode not in VALID_MODES:
        raise ValueError(f"Unknown feature_mode '{feature_mode}'. Choose from {VALID_MODES}")

    exchange_id = config['data']['exchange']
    symbols     = config['data']['symbols']
    timeframe   = config['data']['base_timeframe']

    start_date_str = config['data']['start_date']
    since = int(pd.Timestamp(start_date_str).timestamp() * 1000)

    os.makedirs("data/raw",       exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    all_features_dfs = []
    all_expert_dfs   = []

    for symbol in symbols:
        safe_symbol   = symbol.replace("/", "")
        raw_path      = f"data/raw/{safe_symbol}_{timeframe}_raw.parquet"
        mtf_raw_path  = f"data/processed/{safe_symbol}_{timeframe}_MTF_raw.parquet"
        features_path = f"data/processed/{safe_symbol}_{timeframe}_MTF_features_{feature_mode}.parquet"
        expert_path   = f"data/processed/{safe_symbol}_{timeframe}_MTF_expert_{feature_mode}.parquet"

        # Norm stats paths (only used for modes with raw prices)
        stats_path = None
        if feature_mode in ("raw_prices", "combined_prices"):
            stats_path = features_path.replace(".parquet", "_norm_stats.json")

        # ----------------------------------------------------------------
        # Fast path: all derived files already exist
        # ----------------------------------------------------------------
        if (not force_refresh
                and os.path.exists(features_path)
                and os.path.exists(expert_path)):
            print(f"[{feature_mode}] Found processed data for {symbol}.")
            df_feat = pd.read_parquet(features_path)
            df_exp  = pd.read_parquet(expert_path)
            df_feat['symbol'] = symbol
            df_exp['symbol']  = symbol
            all_features_dfs.append(df_feat)
            all_expert_dfs.append(df_exp)
            continue

        print(f"[{feature_mode}] Processed data not found for {symbol}. Generating...")

        # ----------------------------------------------------------------
        # Step 1: Fetch raw OHLCV (shared across all modes — cached once)
        # ----------------------------------------------------------------
        if os.path.exists(raw_path) and not force_refresh:
            print(f"Loading cached raw data for {symbol}...")
            raw_df = pd.read_parquet(raw_path)
        else:
            raw_df = fetch_ohlcv(exchange_id, symbol, timeframe, since)
            raw_df.to_parquet(raw_path)

        # ----------------------------------------------------------------
        # Step 2: MTF alignment (shared across all modes — cached once)
        # ----------------------------------------------------------------
        if os.path.exists(mtf_raw_path) and not force_refresh:
            print(f"Loading cached MTF raw data for {symbol}...")
        else:
            print(f"Aligning MTF data for {symbol}...")
            mtf_df = create_mtf_dataset(raw_path, target_freqs=['4h', '1d'])
            mtf_df.to_parquet(mtf_raw_path)

        # ----------------------------------------------------------------
        # Step 3: Feature generation (per mode)
        # ----------------------------------------------------------------
        print(f"Generating features [{feature_mode}] for {symbol}...")
        _, norm_stats = process_mtf_data_for_mode(
            feature_mode,
            mtf_raw_path,
            features_path,
            stats_path=stats_path,
        )

        # ----------------------------------------------------------------
        # Step 4: Expert labels (computed on the feature file so the
        #         label rows align with whichever mode's dropna() removed)
        # ----------------------------------------------------------------
        print(f"Generating expert labels for {symbol}...")
        create_expert_rewards(features_path, expert_path, threshold_pct=0.02)

        df_feat = pd.read_parquet(features_path)
        df_exp  = pd.read_parquet(expert_path)
        df_feat['symbol'] = symbol
        df_exp['symbol']  = symbol
        all_features_dfs.append(df_feat)
        all_expert_dfs.append(df_exp)

    return pd.concat(all_features_dfs), pd.concat(all_expert_dfs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None,
                        help="Feature mode override (default: read from config.yaml)")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    feat, exp = get_or_fetch_data(feature_mode=args.mode,
                                  force_refresh=args.force_refresh)
    print(f"Final Features Shape: {feat.shape}")
    print(f"Final Expert Shape:   {exp.shape}")
