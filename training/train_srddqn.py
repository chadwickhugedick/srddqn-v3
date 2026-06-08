import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from gymnasium.wrappers import TimeLimit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.crypto_mtf_env import CryptoMtfEnv
from utils.config import load_config
from data_pipeline.fetch_data import get_or_fetch_data
from models.timesnet import TimesNet
from training.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

class QNetwork(nn.Module):
    """
    Q-Network wrapper around TimesNet.
    Takes [batch, seq_len, num_features] and concatenates with [pos, equity]
    """
    def __init__(self, seq_len, num_features, d_model=32):
        super().__init__()
        self.timesnet = TimesNet(seq_len=seq_len, num_features=num_features, 
                                 d_model=d_model, d_ff=d_model*2, e_layers=2, top_k=2)
        # Override the projection to just return the features
        self.timesnet.projection = nn.Identity()
        
        # Policy head (d_model from TimesNet + 2 from pos/equity -> 3 actions)
        self.fc = nn.Sequential(
            nn.Linear(d_model + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, seq_obs, pos_equity):
        features = self.timesnet(seq_obs)
        x = torch.cat([features, pos_equity], dim=1)
        return self.fc(x)

def get_save_path(config, mode, run_dir=None):
    if run_dir:
        return f"models/runs/{run_dir}/srddqn_native_{mode}.pth"
    template = config.get('paths', {}).get('srddqn_model', 'logs/models/srddqn_native_{mode}.pth')
    return template.replace('{mode}', mode)

def unpack_obs(obs_tensor, seq_len, num_features):
    """ Unpacks flat obs [B, (seq_len * num_features) + 2] into sequence and scalar states """
    seq_flat = obs_tensor[:, :-2]
    pos_equity = obs_tensor[:, -2:]
    seq_obs = seq_flat.view(-1, seq_len, num_features)
    return seq_obs, pos_equity

def main(feature_mode: str = None, timestamp: str = None):
    config = load_config()
    mode = feature_mode or config.get('features', {}).get('mode', 'indicators')
    
    print(f"\n{'='*60}")
    print(f" Training NATIVE SRDDQN Agent  |  mode = {mode}")
    print(f"{'='*60}\n")

    # 1. Fetch data
    features_df, _ = get_or_fetch_data(config, feature_mode=mode)
    if features_df.empty:
        print("Error: No data fetched or generated.")
        return

    df = features_df.copy()
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)

    _exclude_base = ['open', 'high', 'low', 'close', 'volume', 'expert_trend', 'reward_0', 'reward_1', 'reward_2', 'symbol']
    exclude_cols = _exclude_base.copy()
    exclude_cols += [f'4h_{c}' for c in _exclude_base]
    exclude_cols += [f'1d_{c}' for c in _exclude_base]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    num_features = len(feature_cols)

    # 2. Config parameters
    lookback = config['environment']['lookback_window']
    fee = config['environment']['fee_pct']
    hold_cost = config['environment'].get('holding_cost_pct', 0.00001)
    bankrupt = config['environment'].get('bankruptcy_floor', 0.1)
    seq = config['timesnet']['seq_len']
    
    rl = config['rl_agent']
    gamma = rl['gamma']
    batch_size = rl['batch_size']
    learning_rate = rl['learning_rate']
    buffer_size = rl['buffer_size']
    tau = rl.get('tau', 1.0)
    learning_starts = 10000
    train_freq = 4
    target_update_interval = rl['target_update_interval']
    max_steps = 500000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Build environment
    base_env = CryptoMtfEnv(train_df, feature_cols, lookback_window=lookback,
                             fee_pct=fee, holding_cost_pct=hold_cost, bankruptcy_floor=bankrupt)
    train_env = TimeLimit(base_env, max_episode_steps=1000)

    obs_shape = train_env.observation_space.shape
    buffer = ReplayBuffer(buffer_size, obs_shape, device)

    # 4. Initialize Networks
    q_net = QNetwork(seq, num_features, d_model=32).to(device)
    target_net = QNetwork(seq, num_features, d_model=32).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    # Reward Network (pure TimesNet, outputs 3 rewards)
    reward_net = TimesNet(seq_len=seq, num_features=num_features, d_model=32, d_ff=64, e_layers=2, top_k=2).to(device)

    reward_model_path = config.get('paths', {}).get('timesnet_model', '').replace('{mode}', mode)
    if os.path.exists(reward_model_path):
        reward_net.load_state_dict(torch.load(reward_model_path, map_location=device))
        print(f"Loaded pretrained Reward Network from {reward_model_path}")
    else:
        print(f"WARNING: Pretrained Reward Network not found! Initializing randomly.")

    optimizer_q = optim.Adam(q_net.parameters(), lr=learning_rate)
    optimizer_r = optim.Adam(reward_net.parameters(), lr=learning_rate)

    # 5. Logging
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{timestamp}_{mode}"
    tb_log_dir = f"logs/runs/{run_dir}/tb"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)

    print(f"Starting Native Training. Using {device}")
    print(f"TensorBoard log dir: {tb_log_dir}")

    # 6. Training Loop
    obs, _ = train_env.reset()
    episode_reward = 0
    episodes = 0
    ep_equities = []
    ep_positions = []

    epsilon_start = 0.9
    epsilon_final = 0.05
    epsilon_decay = 100000

    for step in range(1, max_steps + 1):
        # Epsilon greedy
        epsilon = max(epsilon_final, epsilon_start - step * (epsilon_start - epsilon_final) / epsilon_decay)
        
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            seq_obs, pos_equity = unpack_obs(obs_t, seq, num_features)
            
            # Step 4: Predict r_s for all 3 actions
            r_s = reward_net(seq_obs).squeeze(0).cpu().numpy()
            
            # Q-network action selection
            if np.random.rand() < epsilon:
                action = train_env.action_space.sample()
            else:
                q_values = q_net(seq_obs, pos_equity)
                action = q_values.argmax(dim=1).item()

        next_obs, env_reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        
        # Step 5: Expert rewards from dataframe (or 0 if missing)
        r_e = info.get("expert_rewards", np.zeros(3, dtype=np.float32))
        
        # Step 6: Hybrid Reward Vector r_l = max(r_s, r_e)
        r_l = np.maximum(r_s, r_e)
        
        # Step 7: Agent receives the scalar reward for the taken action
        hybrid_step_reward = r_l[action]
        episode_reward += hybrid_step_reward
        
        if "equity" in info: ep_equities.append(info["equity"])
        if "position" in info: ep_positions.append(info["position"])

        # Step 8: Store the VECTOR r_l in the buffer!
        buffer.add(obs, action, r_l, next_obs, done)
        obs = next_obs

        if done:
            final_equity = ep_equities[-1] if ep_equities else 0
            writer.add_scalar("Rollout/Episode_Reward", episode_reward, step)
            writer.add_scalar("Financial/Final_Equity", final_equity, step)
            
            if ep_equities:
                writer.add_scalar("Financial/Min_Equity", min(ep_equities), step)
                writer.add_scalar("Financial/Max_Equity", max(ep_equities), step)
            if ep_positions:
                pos = np.array(ep_positions)
                writer.add_scalar("Financial/Pct_Long", float(np.mean(pos == 1)), step)
                writer.add_scalar("Financial/Pct_Short", float(np.mean(pos == -1)), step)
                writer.add_scalar("Financial/Pct_Flat", float(np.mean(pos == 0)), step)
                
            print(f"✅ Episode {episodes + 1} completed at step {step} | Total Reward: {episode_reward:.4f} | Final Equity: ${final_equity:.2f}")
            
            obs, _ = train_env.reset()
            episode_reward = 0
            ep_equities = []
            ep_positions = []
            episodes += 1

        # Synchronous Training Step
        if step > learning_starts and step % train_freq == 0:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)
            
            b_seq_obs, b_pos_eq = unpack_obs(b_obs, seq, num_features)
            b_next_seq_obs, b_next_pos_eq = unpack_obs(b_next_obs, seq, num_features)

            # --- A. Reward Network Update ---
            # Step 13: b_rewards is now the full hybrid reward vector r_l [B, 3]
            pred_rewards_all = reward_net(b_seq_obs)
            loss_r = nn.MSELoss()(pred_rewards_all, b_rewards)
            
            optimizer_r.zero_grad()
            loss_r.backward()
            optimizer_r.step()
            loss_r_val = loss_r.item()

            # --- B. Q-Network Update ---
            # Step 10/11: Extract the scalar reward for the specific action taken
            b_scalar_rewards = b_rewards.gather(1, b_actions)
            
            with torch.no_grad():
                # Double DQN logic
                next_q_main = q_net(b_next_seq_obs, b_next_pos_eq)
                next_actions = next_q_main.argmax(dim=1, keepdim=True)
                
                next_q_target = target_net(b_next_seq_obs, b_next_pos_eq)
                next_q_vals = next_q_target.gather(1, next_actions)
                
                y = b_scalar_rewards + gamma * (1 - b_dones) * next_q_vals

            current_q = q_net(b_seq_obs, b_pos_eq).gather(1, b_actions)
            loss_q = nn.MSELoss()(current_q, y)

            optimizer_q.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer_q.step()

            writer.add_scalar("Train/Loss_Q", loss_q.item(), step)
            writer.add_scalar("Train/Loss_RewardNet", loss_r_val, step)
            writer.add_scalar("Train/Average_Q", current_q.mean().item(), step)
            writer.add_scalar("Train/Average_Hybrid_R", b_scalar_rewards.mean().item(), step)

            if step % 1000 == 0:
                print(f"🔄 Step {step}/{max_steps} | Loss Q: {loss_q.item():.6f} | Loss R: {loss_r_val:.6f} | Avg Q: {current_q.mean().item():.4f} | Avg R: {b_scalar_rewards.mean().item():.4f}")

        if step > learning_starts and step % target_update_interval == 0:
            if tau == 1.0:
                target_net.load_state_dict(q_net.state_dict())
            else:
                for target_param, q_param in zip(target_net.parameters(), q_net.parameters()):
                    target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

    save_path = get_save_path(config, mode, run_dir=run_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(q_net.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, help="Feature mode (overrides config.yaml)")
    parser.add_argument("--timestamp", default=None, help="Dynamic timestamp/run ID override")
    args = parser.parse_args()
    main(feature_mode=args.mode, timestamp=args.timestamp)
