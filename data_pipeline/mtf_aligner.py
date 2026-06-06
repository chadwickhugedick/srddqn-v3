import pandas as pd
import numpy as np

def resample_and_align(base_df: pd.DataFrame, datetime_col: str, target_freq: str, 
                       agg_dict: dict = None) -> pd.DataFrame:
    """
    Resamples a base dataframe to a target frequency and aligns it back to the base dataframe's index
    WITHOUT introducing look-ahead bias.
    
    Args:
        base_df: The base dataframe (e.g. 1h data)
        datetime_col: Name of the datetime column
        target_freq: The pandas frequency string to resample to (e.g. '4h', '1d')
        agg_dict: Dictionary specifying how to aggregate columns (e.g. {'open': 'first', 'high': 'max', ...})
                  If None, a default OHLCV aggregation is used.
    
    Returns:
        Aligned dataframe with same index as base_df, containing the most recently closed target_freq features.
    """
    df = base_df.copy()
    
    if datetime_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
    else:
        # Assume index is already the datetime
        pass
    
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        # Keep only columns that exist in the dataframe
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    # 1. Resample to the higher frequency
    # closed='right' and label='right' ensures that a '4h' candle spanning 00:00 to 04:00 is labeled 04:00.
    # We use label='right' so the timestamp represents the exact moment the candle is fully formed and closed.
    resampled_df = df.resample(target_freq, closed='right', label='right').agg(agg_dict).dropna()
    
    # Prefix columns to avoid collision
    resampled_df.columns = [f"{target_freq}_{col}" for col in resampled_df.columns]
    
    # 2. Reindex back to the base timeframe. 
    # We use ffill() (forward fill). Since the resampled label is exactly when the candle closes,
    # forward filling will propagate that closed candle to all subsequent base candles until a new higher-TF candle closes.
    # This completely eliminates look-ahead bias.
    aligned_df = resampled_df.reindex(df.index, method='ffill')
    
    # If base_df didn't have datetime as index, reset it back
    if datetime_col in base_df.columns and base_df.index.name != datetime_col:
        aligned_df = aligned_df.reset_index()
        
    return aligned_df

def create_mtf_dataset(base_df_path: str, datetime_col: str = 'timestamp', 
                       target_freqs: list = ['4h', '1d']) -> pd.DataFrame:
    """
    Helper to load a base dataset and attach multiple higher timeframes.
    """
    base_df = pd.read_parquet(base_df_path)
    
    result_df = base_df.copy()
    
    for freq in target_freqs:
        print(f"Aligning {freq} timeframe...")
        aligned = resample_and_align(base_df, datetime_col, freq)
        # Drop the datetime col from aligned since it's already in result_df index
        if datetime_col in aligned.columns:
            aligned = aligned.drop(columns=[datetime_col])
        elif aligned.index.name == datetime_col:
            pass # Keep index
        result_df = pd.concat([result_df, aligned], axis=1)
        
    return result_df

if __name__ == "__main__":
    # Quick test
    print("Testing MTF Aligner on raw data...")
    try:
        raw_path = "data/raw/BTCUSDT_1h_2000-01-01_to_present.parquet"
        mtf_df = create_mtf_dataset(raw_path, target_freqs=['4h', '1d'])
        print(f"Original shape: {pd.read_parquet(raw_path).shape}")
        print(f"MTF shape: {mtf_df.shape}")
        print("Columns added:", [c for c in mtf_df.columns if '4h' in c or '1d' in c])
        
        # Save a sample to verify
        mtf_df.to_parquet("data/processed/BTCUSDT_1h_MTF_raw.parquet")
        print("Saved MTF raw dataset to data/processed/BTCUSDT_1h_MTF_raw.parquet")
    except Exception as e:
        print(f"Test failed: {e}")
