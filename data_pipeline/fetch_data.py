import os
import sys
import pandas as pd
import ccxt
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from data_pipeline.mtf_aligner import create_mtf_dataset
from data_pipeline.features_mtf import process_mtf_data
from data_pipeline.expert_labels import create_expert_rewards

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

def get_or_fetch_data(config=None, force_refresh=False):
    if config is None:
        config = load_config()
        
    exchange_id = config['data']['exchange']
    symbols = config['data']['symbols']
    timeframe = config['data']['base_timeframe']
    
    start_date_str = config['data']['start_date']
    since = int(pd.Timestamp(start_date_str).timestamp() * 1000)
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    all_features_dfs = []
    all_expert_dfs = []
    
    for symbol in symbols:
        safe_symbol = symbol.replace("/", "")
        raw_path = f"data/raw/{safe_symbol}_{timeframe}_raw.parquet"
        mtf_raw_path = f"data/processed/{safe_symbol}_{timeframe}_MTF_raw.parquet"
        features_path = f"data/processed/{safe_symbol}_{timeframe}_MTF_features.parquet"
        expert_path = f"data/processed/{safe_symbol}_{timeframe}_MTF_expert.parquet"
        
        if not force_refresh and os.path.exists(expert_path) and os.path.exists(features_path):
            print(f"Found processed data for {symbol}.")
            df_feat = pd.read_parquet(features_path)
            df_exp = pd.read_parquet(expert_path)
            df_feat['symbol'] = symbol
            df_exp['symbol'] = symbol
            all_features_dfs.append(df_feat)
            all_expert_dfs.append(df_exp)
            continue
            
        print(f"Processed data not found for {symbol}. Fetching and generating...")
        
        # 1. Fetch Raw Data
        if os.path.exists(raw_path) and not force_refresh:
            print(f"Loading cached raw data for {symbol}...")
            raw_df = pd.read_parquet(raw_path)
        else:
            raw_df = fetch_ohlcv(exchange_id, symbol, timeframe, since)
            raw_df.to_parquet(raw_path)
            
        # 2. Align MTF Data
        print(f"Aligning MTF data for {symbol}...")
        mtf_df = create_mtf_dataset(raw_path, target_freqs=['4h', '1d'])
        mtf_df.to_parquet(mtf_raw_path)
        
        # 3. Generate Features
        print(f"Generating features for {symbol}...")
        process_mtf_data(mtf_raw_path, features_path)
        
        # 4. Generate Expert Labels
        print(f"Generating expert labels for {symbol}...")
        create_expert_rewards(features_path, expert_path, threshold_pct=0.02)
        
        df_feat = pd.read_parquet(features_path)
        df_exp = pd.read_parquet(expert_path)
        df_feat['symbol'] = symbol
        df_exp['symbol'] = symbol
        all_features_dfs.append(df_feat)
        all_expert_dfs.append(df_exp)
        
    return pd.concat(all_features_dfs), pd.concat(all_expert_dfs)

if __name__ == "__main__":
    feat, exp = get_or_fetch_data()
    print(f"Final Features Shape: {feat.shape}")
    print(f"Final Expert Shape: {exp.shape}")
