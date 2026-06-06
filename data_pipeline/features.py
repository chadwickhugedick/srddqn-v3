import os
import pandas as pd
import numpy as np

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

def process_data(input_path: str, output_path: str):
    print(f"Processing data from {input_path}")
    df = pd.read_parquet(input_path)
    
    # Calculate simple features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], period=14)
    
    # MACD
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['dist_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # Drop rows with NaNs
    df = df.dropna()
    
    # Select features for observation space
    feature_cols = [
        'returns', 'log_returns', 'volatility_20', 'rsi_14',
        'macd', 'macd_signal', 'macd_hist', 'dist_sma_20', 'dist_sma_50'
    ]
    
    # Normalize features (Z-score normalization)
    features_mean = df[feature_cols].mean()
    features_std = df[feature_cols].std()
    
    df_normalized = df.copy()
    df_normalized[feature_cols] = (df[feature_cols] - features_mean) / features_std
    
    # Save processed dataframe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_normalized.to_parquet(output_path)
    print(f"Saved processed data to {output_path}")
    
if __name__ == "__main__":
    raw_path = "data/raw/BTCUSDT_1h_2000-01-01_to_present.parquet"
    if not os.path.exists(raw_path):
        raw_path = "data/raw/BTC_USDT_USDT_1h_2000-01-01_to_present.parquet"
    output_path = "data/processed/BTCUSDT_1h_features.parquet"
    process_data(raw_path, output_path)
