import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        # Ensure out_channels is divisible by 4 for the 4 branches
        assert out_channels % 4 == 0
        c = out_channels // 4
        
        self.branch1 = nn.Conv2d(in_channels, c, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=1),
            nn.Conv2d(c, c, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=1),
            nn.Conv2d(c, c, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c, kernel_size=1)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

def FFT_for_Period(x, k=3):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    
    # Calculate amplitude
    frequency_list = abs(xf).mean(0).mean(-1)
    # Ignore 0 frequency
    frequency_list[0] = 0
    
    # Find top k frequencies
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    
    # Convert frequencies to periods
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, top_k, d_ff):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff),
            nn.GELU(),
            InceptionBlock(d_ff, d_model)
        )
        
    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.size()
        period_list, weight = FFT_for_Period(x, self.k)
        
        res = []
        for i in range(self.k):
            period = period_list[i]
            # Padding if seq_len is not perfectly divisible
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([B, length - T, C], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
                
            # Reshape 1D to 2D
            out = out.reshape(B, length // period, period, C).permute(0, 3, 1, 2).contiguous()
            
            # 2D Conv
            out = self.conv(out)
            
            # Reshape 2D back to 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, C)
            res.append(out[:, :T, :])
            
        res = torch.stack(res, dim=-1)
        # Weight aggregation
        weight = F.softmax(weight, dim=-1).unsqueeze(1).unsqueeze(1)
        res = (res * weight).sum(-1)
        
        return res + x # Residual connection

class TimesNet(nn.Module):
    def __init__(self, seq_len=10, num_features=27, d_model=64, d_ff=64, e_layers=2, top_k=3, num_classes=3):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        
        # Feature embedding
        self.embedding = nn.Linear(num_features, d_model)
        
        # TimesBlocks
        self.model = nn.ModuleList([
            TimesBlock(seq_len, d_model, top_k, d_ff)
            for _ in range(e_layers)
        ])
        
        # Output mapping (Predicting Min-Max rewards for 3 actions)
        self.projection = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: [B, T, C]
        
        # Project inputs to d_model space
        x = self.embedding(x)
        
        # Pass through TimesBlocks
        for layer in self.model:
            x = layer(x)
            
        # We only care about the last timestep's prediction for the current state's action
        # Actually, in RL, the reward is often associated with the final state of the sequence
        out = x[:, -1, :] 
        
        # Map to 3 classes (rewards)
        out = self.projection(out)
        
        return out
