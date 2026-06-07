import os
import json
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helper: technical indicator computation
# ---------------------------------------------------------------------------

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, short_period=12, long_period=26, signal_period=9):
    ema_short = series.ewm(span=short_period, adjust=False).mean()
    ema_long = series.ewm(span=long_period, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal


def _indicator_features_for_prefix(df, close_col, prefix=""):
    """Compute the 9 technical indicator features for one timeframe.

    Returns a dict of {col_name: Series}.
    """
    ret_col      = f"{prefix}returns"
    logret_col   = f"{prefix}log_returns"
    vol_col      = f"{prefix}volatility_20"
    rsi_col      = f"{prefix}rsi_14"
    macd_col     = f"{prefix}macd"
    sig_col      = f"{prefix}macd_signal"
    hist_col     = f"{prefix}macd_hist"
    d20_col      = f"{prefix}dist_sma_20"
    d50_col      = f"{prefix}dist_sma_50"

    returns    = df[close_col].pct_change()
    log_ret    = np.log(df[close_col] / df[close_col].shift(1))
    vol        = returns.rolling(window=20).std()
    rsi        = calculate_rsi(df[close_col], period=14)
    macd, sig  = calculate_macd(df[close_col])
    hist       = macd - sig
    sma20      = df[close_col].rolling(window=20).mean()
    sma50      = df[close_col].rolling(window=50).mean()
    d_sma20    = (df[close_col] - sma20) / sma20
    d_sma50    = (df[close_col] - sma50) / sma50

    return {
        ret_col:   returns,
        logret_col: log_ret,
        vol_col:   vol,
        rsi_col:   rsi,
        macd_col:  macd,
        sig_col:   sig,
        hist_col:  hist,
        d20_col:   d_sma20,
        d50_col:   d_sma50,
    }, [ret_col, logret_col, vol_col, rsi_col, macd_col, sig_col, hist_col, d20_col, d50_col]


def _raw_ratio_features_for_prefix(df, open_col, high_col, low_col, close_col, vol_col_raw, prefix=""):
    """Compute 4 scale-invariant OHLCV ratio features for one timeframe.

    All outputs are dimensionless, making them robust to BTC's multi-year
    price range ($3 k → $70 k+).
    """
    lr_col  = f"{prefix}log_return"     # log(close / open)
    hl_col  = f"{prefix}hl_ratio"       # log(high / low)  — candle range
    co_col  = f"{prefix}co_ratio"       # (close - open) / open — body fraction
    vr_col  = f"{prefix}volume_rel"     # volume / rolling_20_mean(volume)

    log_return  = np.log(df[close_col] / df[open_col].replace(0, np.nan))
    hl_ratio    = np.log((df[high_col] / df[low_col].replace(0, np.nan)).clip(lower=1e-8))
    co_ratio    = (df[close_col] - df[open_col]) / df[open_col].replace(0, np.nan)
    vol_mean    = df[vol_col_raw].rolling(window=20).mean().replace(0, np.nan)
    volume_rel  = df[vol_col_raw] / vol_mean

    return {
        lr_col: log_return,
        hl_col: hl_ratio,
        co_col: co_ratio,
        vr_col: volume_rel,
    }, [lr_col, hl_col, co_col, vr_col]


def _raw_price_features_for_prefix(df, open_col, high_col, low_col, close_col, vol_col_raw, prefix=""):
    """Return the 5 raw OHLCV columns renamed with the given prefix.

    Normalization (z-score) is applied *outside* this function so that
    training-set statistics can be reused for val/test splits.
    """
    o_col = f"{prefix}open"
    h_col = f"{prefix}high"
    l_col = f"{prefix}low"
    c_col = f"{prefix}close"
    v_col = f"{prefix}volume"

    return {
        o_col: df[open_col],
        h_col: df[high_col],
        l_col: df[low_col],
        c_col: df[close_col],
        v_col: df[vol_col_raw],
    }, [o_col, h_col, l_col, c_col, v_col]


def _get_timeframe_prefixes(df):
    """Return list of (prefix, open_col, high_col, low_col, close_col, vol_col) tuples.

    Base timeframe (1h) has no prefix; higher TFs have e.g. '4h_', '1d_'.
    """
    timeframes = []

    base_cols = ['open', 'high', 'low', 'close', 'volume']
    if all(c in df.columns for c in base_cols):
        timeframes.append(("", "open", "high", "low", "close", "volume"))

    prefixes = set()
    for col in df.columns:
        if '_' in col and col.endswith('_close'):
            parts = col.split('_', 1)
            pfx = parts[0] + "_"
            prefixes.add(pfx)

    for pfx in sorted(prefixes):
        o = f"{pfx}open"
        h = f"{pfx}high"
        l = f"{pfx}low"
        c = f"{pfx}close"
        v = f"{pfx}volume"
        if all(col in df.columns for col in [o, h, l, c, v]):
            timeframes.append((pfx, o, h, l, c, v))

    return timeframes


def _zscore_normalize(df, feature_cols, stats=None):
    """Z-score normalize feature_cols in df.

    If stats is None, compute from df (training mode).
    Returns (df_normalized, stats_dict).
    """
    if stats is None:
        mean = df[feature_cols].mean()
        std  = df[feature_cols].std().replace(0, 1)
        stats = {"mean": mean.to_dict(), "std": std.to_dict()}
    else:
        mean = pd.Series(stats["mean"])
        std  = pd.Series(stats["std"])

    df_out = df.copy()
    df_out[feature_cols] = (df[feature_cols] - mean) / std
    return df_out, stats


# ---------------------------------------------------------------------------
# MODE 1 (original): indicators only
# ---------------------------------------------------------------------------

def process_mtf_data(input_path: str, output_path: str):
    """Original function — kept for backwards compatibility."""
    print(f"[indicators] Processing MTF data from {input_path}")
    df = pd.read_parquet(input_path)

    all_feature_cols = []

    base_cols = ['open', 'high', 'low', 'close', 'volume']
    if all(c in df.columns for c in base_cols):
        feats, cols = _indicator_features_for_prefix(df, 'close', prefix="")
        for col, series in feats.items():
            df[col] = series
        all_feature_cols.extend(cols)

    prefixes = set()
    for col in df.columns:
        if '_' in col and (col.endswith('_close') or col.endswith('_open')):
            parts = col.split('_')
            prefixes.add(parts[0] + "_")

    for pfx in sorted(prefixes):
        close_col = f"{pfx}close"
        if close_col not in df.columns:
            continue
        feats, cols = _indicator_features_for_prefix(df, close_col, prefix=pfx)
        for col, series in feats.items():
            df[col] = series
        all_feature_cols.extend(cols)

    df = df.dropna()

    features_mean = df[all_feature_cols].mean()
    features_std  = df[all_feature_cols].std().replace(0, 1)
    df_normalized = df.copy()
    df_normalized[all_feature_cols] = (df[all_feature_cols] - features_mean) / features_std

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_normalized.to_parquet(output_path)
    print(f"[indicators] Saved to {output_path} — {len(all_feature_cols)} features.")
    return all_feature_cols


# ---------------------------------------------------------------------------
# MODE 2: raw_ratios
# ---------------------------------------------------------------------------

def process_mtf_data_raw_ratios(input_path: str, output_path: str):
    """Scale-invariant OHLCV ratio features only (4 per timeframe)."""
    print(f"[raw_ratios] Processing MTF data from {input_path}")
    df = pd.read_parquet(input_path)

    all_feature_cols = []
    timeframes = _get_timeframe_prefixes(df)

    for pfx, o, h, l, c, v in timeframes:
        feats, cols = _raw_ratio_features_for_prefix(df, o, h, l, c, v, prefix=pfx)
        for col, series in feats.items():
            df[col] = series
        all_feature_cols.extend(cols)

    df = df.dropna()
    df, stats = _zscore_normalize(df, all_feature_cols)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"[raw_ratios] Saved to {output_path} — {len(all_feature_cols)} features.")
    return all_feature_cols


# ---------------------------------------------------------------------------
# MODE 3: raw_prices
# ---------------------------------------------------------------------------

def process_mtf_data_raw_prices(input_path: str, output_path: str,
                                  norm_stats=None, stats_path=None):
    """Raw OHLCV values, z-score normalized.

    norm_stats: if provided, use these training statistics (for val/test splits).
    stats_path: if provided, save computed norm_stats as JSON here.
    """
    print(f"[raw_prices] Processing MTF data from {input_path}")
    df = pd.read_parquet(input_path)

    all_feature_cols = []
    timeframes = _get_timeframe_prefixes(df)

    for pfx, o, h, l, c, v in timeframes:
        feats, cols = _raw_price_features_for_prefix(df, o, h, l, c, v, prefix=pfx)
        for col, series in feats.items():
            df[col] = series
        all_feature_cols.extend(cols)

    df = df.dropna()
    df, stats = _zscore_normalize(df, all_feature_cols, stats=norm_stats)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)

    if stats_path is not None:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[raw_prices] Saved norm stats to {stats_path}")

    print(f"[raw_prices] Saved to {output_path} — {len(all_feature_cols)} features.")
    return all_feature_cols, stats


# ---------------------------------------------------------------------------
# MODE 4: combined_ratios  (indicators + raw_ratios)
# ---------------------------------------------------------------------------

def process_mtf_data_combined_ratios(input_path: str, output_path: str):
    """Indicators (9/TF) + scale-invariant raw ratios (4/TF) = 13/TF."""
    print(f"[combined_ratios] Processing MTF data from {input_path}")
    df = pd.read_parquet(input_path)

    all_feature_cols = []
    timeframes = _get_timeframe_prefixes(df)

    for pfx, o, h, l, c, v in timeframes:
        # Indicators
        ind_feats, ind_cols = _indicator_features_for_prefix(df, c, prefix=pfx)
        for col, series in ind_feats.items():
            df[col] = series
        all_feature_cols.extend(ind_cols)

        # Raw ratios
        ratio_feats, ratio_cols = _raw_ratio_features_for_prefix(df, o, h, l, c, v, prefix=pfx)
        for col, series in ratio_feats.items():
            df[col] = series
        all_feature_cols.extend(ratio_cols)

    df = df.dropna()
    df, stats = _zscore_normalize(df, all_feature_cols)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"[combined_ratios] Saved to {output_path} — {len(all_feature_cols)} features.")
    return all_feature_cols


# ---------------------------------------------------------------------------
# MODE 5: combined_prices  (indicators + raw_prices)
# ---------------------------------------------------------------------------

def process_mtf_data_combined_prices(input_path: str, output_path: str,
                                       norm_stats=None, stats_path=None):
    """Indicators (9/TF) + raw OHLCV prices (5/TF) = 14/TF."""
    print(f"[combined_prices] Processing MTF data from {input_path}")
    df = pd.read_parquet(input_path)

    all_feature_cols = []
    timeframes = _get_timeframe_prefixes(df)

    for pfx, o, h, l, c, v in timeframes:
        # Indicators
        ind_feats, ind_cols = _indicator_features_for_prefix(df, c, prefix=pfx)
        for col, series in ind_feats.items():
            df[col] = series
        all_feature_cols.extend(ind_cols)

        # Raw prices
        price_feats, price_cols = _raw_price_features_for_prefix(df, o, h, l, c, v, prefix=pfx)
        for col, series in price_feats.items():
            df[col] = series
        all_feature_cols.extend(price_cols)

    df = df.dropna()
    df, stats = _zscore_normalize(df, all_feature_cols, stats=norm_stats)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)

    if stats_path is not None:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[combined_prices] Saved norm stats to {stats_path}")

    print(f"[combined_prices] Saved to {output_path} — {len(all_feature_cols)} features.")
    return all_feature_cols, stats


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def process_mtf_data_for_mode(mode: str, mtf_raw_path: str,
                               output_path: str,
                               norm_stats=None, stats_path=None):
    """Dispatch to the correct processing function based on mode string.

    Returns (feature_cols, norm_stats_or_None).
    """
    if mode == "indicators":
        cols = process_mtf_data(mtf_raw_path, output_path)
        return cols, None

    elif mode == "raw_ratios":
        cols = process_mtf_data_raw_ratios(mtf_raw_path, output_path)
        return cols, None

    elif mode == "raw_prices":
        cols, stats = process_mtf_data_raw_prices(
            mtf_raw_path, output_path,
            norm_stats=norm_stats, stats_path=stats_path)
        return cols, stats

    elif mode == "combined_ratios":
        cols = process_mtf_data_combined_ratios(mtf_raw_path, output_path)
        return cols, None

    elif mode == "combined_prices":
        cols, stats = process_mtf_data_combined_prices(
            mtf_raw_path, output_path,
            norm_stats=norm_stats, stats_path=stats_path)
        return cols, stats

    else:
        raise ValueError(f"Unknown feature mode: '{mode}'. "
                         f"Choose from: indicators, raw_ratios, raw_prices, "
                         f"combined_ratios, combined_prices")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MTF features for a given mode.")
    parser.add_argument("--mode", default="indicators",
                        choices=["indicators", "raw_ratios", "raw_prices",
                                 "combined_ratios", "combined_prices"])
    parser.add_argument("--input",  default="data/processed/BTCUSDT_1h_MTF_raw.parquet")
    parser.add_argument("--output", default=None,
                        help="Output path (auto-derived from mode if not set)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/processed/BTCUSDT_1h_MTF_features_{args.mode}.parquet"

    stats_path = None
    if args.mode in ("raw_prices", "combined_prices"):
        stats_path = args.output.replace(".parquet", "_norm_stats.json")

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run mtf_aligner.py first.")
    else:
        cols, _ = process_mtf_data_for_mode(
            args.mode, args.input, args.output, stats_path=stats_path)
        print(f"Done. {len(cols)} feature columns.")
