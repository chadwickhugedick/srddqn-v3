import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros((capacity, 3), dtype=np.float32) # Store vector!
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device).unsqueeze(1),
            torch.as_tensor(self.rewards[idxs], device=self.device), # No unsqueeze since it is [B, 3]
            torch.as_tensor(self.next_obs[idxs], device=self.device),
            torch.as_tensor(self.dones[idxs], device=self.device).unsqueeze(1)
        )
