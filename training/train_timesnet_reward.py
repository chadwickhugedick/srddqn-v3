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
from utils.config import load_config
from data_pipeline.fetch_data import get_or_fetch_data


class MTFRewardDataset(Dataset):
    def __init__(self, df, feature_cols, reward_cols, seq_len=10):
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets  = df[reward_cols].values.astype(np.float32)
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y)


def get_model_path(config, mode):
    template = config.get('paths', {}).get('timesnet_model',
                                           'logs/models/timesnet_reward_{mode}.pth')
    return template.replace('{mode}', mode)


def main(feature_mode: str = None):
    config = load_config()

    # Resolve mode: CLI arg > config file > default
    mode = feature_mode or config.get('features', {}).get('mode', 'indicators')
    print(f"\n{'='*60}")
    print(f" Training TimesNet Reward Network  |  mode = {mode}")
    print(f"{'='*60}\n")

    # Fetch / load data
    _, df_expert = get_or_fetch_data(config, feature_mode=mode)

    if df_expert.empty:
        print("Error: No data fetched or generated.")
        return

    df = df_expert.copy()

    # Train/Val split (80/20 chronological)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df   = df.iloc[train_size:].reset_index(drop=True)

    # Exclude non-feature columns
    _exclude_base = ['open', 'high', 'low', 'close', 'volume',
                     'expert_trend', 'reward_0', 'reward_1', 'reward_2', 'symbol']
    exclude_cols = _exclude_base.copy()
    exclude_cols += [f'4h_{c}' for c in _exclude_base]
    exclude_cols += [f'1d_{c}' for c in _exclude_base]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    reward_cols  = ['reward_0', 'reward_1', 'reward_2']

    print(f"Features: {len(feature_cols)}  |  Rewards: {len(reward_cols)}")
    print(f"Train rows: {len(train_df)}  |  Val rows: {len(val_df)}")

    t_conf     = config['timesnet']
    seq_len    = t_conf['seq_len']
    batch_size = 128

    train_dataset = MTFRewardDataset(train_df, feature_cols, reward_cols, seq_len)
    val_dataset   = MTFRewardDataset(val_df,   feature_cols, reward_cols, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TimesNet(
        seq_len=seq_len,
        num_features=len(feature_cols),
        d_model=t_conf['d_model'],
        d_ff=t_conf['d_ff'],
        e_layers=t_conf['e_layers'],
        top_k=t_conf['top_k'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs      = 10
    train_losses = []
    val_losses   = []
    best_val     = float('inf')

    model_save_path = get_model_path(config, mode)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

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

        # Save best checkpoint
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  → Saved best model (val MSE {best_val:.6f}) to {model_save_path}")

    # Also save final epoch regardless
    torch.save(model.state_dict(), model_save_path)
    print(f"\nFinal model saved to {model_save_path}")

    # Plot
    os.makedirs("logs/plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses,   label='Val MSE')
    plt.title(f'TimesNet Pre-training Loss — mode: {mode}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plot_path = f"logs/plots/timesnet_training_loss_{mode}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")

    return {"mode": mode, "best_val_mse": best_val,
            "train_losses": train_losses, "val_losses": val_losses}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None,
                        help="Feature mode (overrides config.yaml)")
    args = parser.parse_args()
    main(feature_mode=args.mode)
