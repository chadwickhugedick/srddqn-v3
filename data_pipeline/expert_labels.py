import os
import pandas as pd
import numpy as np

def generate_zigzag_labels(prices, threshold_pct=0.02):
    """
    Generates idealized Min-Max (ZigZag) trend labels for a price series.
    
    1: Uptrend (between local min and local max)
    -1: Downtrend (between local max and local min)
    
    This provides a smoothed "macro" direction that ignores small fluctuations,
    acting as the perfect hindsight expert label.
    """
    trends = np.zeros(len(prices))
    
    if len(prices) == 0:
        return trends
        
    # Find peaks and troughs
    last_peak_idx = 0
    last_trough_idx = 0
    last_peak_val = prices.iloc[0]
    last_trough_val = prices.iloc[0]
    
    # 1 for looking for peak, -1 for looking for trough
    direction = 0 
    
    for i in range(1, len(prices)):
        current_price = prices.iloc[i]
        
        if direction == 0:
            if current_price > last_peak_val * (1 + threshold_pct):
                direction = 1
                last_peak_val = current_price
                last_peak_idx = i
            elif current_price < last_trough_val * (1 - threshold_pct):
                direction = -1
                last_trough_val = current_price
                last_trough_idx = i
        elif direction == 1:
            if current_price > last_peak_val:
                last_peak_val = current_price
                last_peak_idx = i
            elif current_price < last_peak_val * (1 - threshold_pct):
                # Peak confirmed, we are now in a downtrend
                trends[last_trough_idx:last_peak_idx] = 1
                direction = -1
                last_trough_val = current_price
                last_trough_idx = i
        elif direction == -1:
            if current_price < last_trough_val:
                last_trough_val = current_price
                last_trough_idx = i
            elif current_price > last_trough_val * (1 + threshold_pct):
                # Trough confirmed, we are now in an uptrend
                trends[last_peak_idx:last_trough_idx] = -1
                direction = 1
                last_peak_val = current_price
                last_peak_idx = i
                
    # Fill the last remaining segment
    if direction == 1:
        trends[last_trough_idx:] = 1
    elif direction == -1:
        trends[last_peak_idx:] = -1
        
    return trends

def create_expert_rewards(input_path: str, output_path: str, threshold_pct=0.03):
    print(f"Generating Min-Max Expert Labels for {input_path}")
    df = pd.read_parquet(input_path)
    
    # Generate the idealized trend
    df['expert_trend'] = generate_zigzag_labels(df['close'], threshold_pct=threshold_pct)
    
    # The expert reward for each action (0=Flat, 1=Long, 2=Short)
    # We map the trend (-1, 1) to rewards for each possible action.
    # If trend is 1 (UP), Long gets positive reward, Short gets negative.
    # If trend is -1 (DOWN), Short gets positive reward, Long gets negative.
    
    # We use log returns of the step to scale the reward realistically, 
    # but the DIRECTION is dictated by the macro trend.
    step_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # Smooth the reward magnitude based on volatility so it's scale-invariant
    vol = step_returns.rolling(100).std().fillna(step_returns.std())
    vol = vol.replace(0, 1e-8)
    
    # Normalize the reward magnitude
    magnitude = (step_returns.abs() / vol).clip(0, 5) # Cap extreme outliers
    
    # Reward matrix: [Flat, Long, Short]
    # For Flat: 0 reward (minus opportunity cost?) let's just make it 0.
    reward_flat = np.zeros(len(df))
    
    # For Long: if expert_trend is 1, positive magnitude. If -1, negative magnitude.
    reward_long = df['expert_trend'] * magnitude
    
    # For Short: inverse of Long
    reward_short = -df['expert_trend'] * magnitude
    
    # We will save these as arrays in the dataframe for the supervised learning target
    df['reward_0'] = reward_flat
    df['reward_1'] = reward_long
    df['reward_2'] = reward_short
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved dataset with expert labels to {output_path}")
    
    # Print some stats
    print("Trend distribution:")
    print(df['expert_trend'].value_counts(normalize=True))

if __name__ == "__main__":
    input_path = "data/processed/BTCUSDT_1h_MTF_features.parquet"
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
    else:
        output_path = "data/processed/BTCUSDT_1h_MTF_expert.parquet"
        create_expert_rewards(input_path, output_path, threshold_pct=0.02)
