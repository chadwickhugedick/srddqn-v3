import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.timesnet import TimesNet

class MTFRewardDataset(Dataset):
    def __init__(self, df, feature_cols, reward_cols, seq_len=10):
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[reward_cols].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y)

def main():
    wandb.init()
    config = wandb.config
    
    data_path = "data/processed/BTCUSDT_1h_MTF_expert.parquet"
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return

    df = pd.read_parquet(data_path)
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'expert_trend', 'reward_0', 'reward_1', 'reward_2']
    exclude_cols += [f'4h_{c}' for c in exclude_cols] + [f'1d_{c}' for c in exclude_cols]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    reward_cols = ['reward_0', 'reward_1', 'reward_2']
    
    train_dataset = MTFRewardDataset(train_df, feature_cols, reward_cols, config.seq_len)
    val_dataset = MTFRewardDataset(val_df, feature_cols, reward_cols, config.seq_len)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TimesNet(
        seq_len=config.seq_len, 
        num_features=len(feature_cols), 
        d_model=config.d_model, 
        d_ff=config.d_ff, 
        e_layers=config.e_layers, 
        top_k=config.top_k
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        wandb.log({
            "epoch": epoch,
            "train_mse": avg_train_loss,
            "val_mse": avg_val_loss
        })

if __name__ == "__main__":
    main()
