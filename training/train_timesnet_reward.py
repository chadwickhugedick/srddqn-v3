import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
        # Features: [seq_len, num_features]
        x = self.features[idx : idx + self.seq_len]
        # Target: The reward at the END of the sequence
        y = self.targets[idx + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y)

def main():
    data_path = "data/processed/BTCUSDT_1h_MTF_expert.parquet"
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return

    df = pd.read_parquet(data_path)
    
    # Train/Val split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'expert_trend', 'reward_0', 'reward_1', 'reward_2']
    exclude_cols += [f'4h_{c}' for c in exclude_cols] + [f'1d_{c}' for c in exclude_cols]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    reward_cols = ['reward_0', 'reward_1', 'reward_2']
    
    print(f"Training TimesNet on {len(feature_cols)} features to predict {len(reward_cols)} expert rewards.")
    
    seq_len = 10
    batch_size = 128
    
    train_dataset = MTFRewardDataset(train_df, feature_cols, reward_cols, seq_len)
    val_dataset = MTFRewardDataset(val_df, feature_cols, reward_cols, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TimesNet(seq_len=seq_len, num_features=len(feature_cols), d_model=32, d_ff=64, e_layers=2, top_k=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 10
    train_losses = []
    val_losses = []
    
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
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
        
    os.makedirs("logs/models", exist_ok=True)
    torch.save(model.state_dict(), "logs/models/timesnet_reward_model.pth")
    print("Saved TimesNet to logs/models/timesnet_reward_model.pth")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Val MSE')
    plt.title('TimesNet Supervised Pre-training Loss (Min-Max Reward)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    os.makedirs("logs/plots", exist_ok=True)
    plt.savefig("logs/plots/timesnet_training_loss.png")
    print("Saved loss plot to logs/plots/timesnet_training_loss.png")

if __name__ == "__main__":
    main()
