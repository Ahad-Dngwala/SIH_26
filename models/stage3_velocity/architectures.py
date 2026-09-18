import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnFeatureExtractor(nn.Module):
    def __init__(self, in_channels=7, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        # x: (B, seq_len, in_channels)
        x = x.transpose(1, 2) # (B, in_channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.transpose(1, 2) # (B, seq_len_downsampled, hidden_dim)
        return x

class ModelB(nn.Module):
    """Direct Speed Model"""
    def __init__(self, in_channels=7, hidden_dim=64):
        super().__init__()
        self.extractor = CnnFeatureExtractor(in_channels, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (B, seq_len, 7)
        features = self.extractor(x)
        out, _ = self.gru(features)
        final_state = out[:, -1, :] # last hidden state
        speed = self.fc(final_state)
        return speed.squeeze(-1) # (B,)

class ModelC(nn.Module):
    """Residual Speed Model"""
    def __init__(self, in_channels=7, hidden_dim=64):
        super().__init__()
        # Similar architecture, but we'll also input UKF speed in the training loop
        self.extractor = CnnFeatureExtractor(in_channels, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32), # +1 for UKF speed scalar
            nn.ReLU(),
            nn.Linear(32, 1) # outputs delta_v
        )
        
    def forward(self, x, ukf_speed):
        features = self.extractor(x)
        out, _ = self.gru(features)
        final_state = out[:, -1, :]
        
        # Concat ukf_speed (B, 1)
        fused = torch.cat([final_state, ukf_speed.unsqueeze(-1)], dim=-1)
        delta_v = self.fc(fused)
        return delta_v.squeeze(-1) # (B,)

class ModelD(nn.Module):
    """Residual + Uncertainty Model"""
    def __init__(self, in_channels=7, hidden_dim=64):
        super().__init__()
        self.extractor = CnnFeatureExtractor(in_channels, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # outputs [mu_delta, log_var_delta]
        )
        
    def forward(self, x, ukf_speed):
        features = self.extractor(x)
        out, _ = self.gru(features)
        final_state = out[:, -1, :]
        
        fused = torch.cat([final_state, ukf_speed.unsqueeze(-1)], dim=-1)
        out = self.fc(fused)
        mu_delta = out[:, 0]
        log_var_delta = out[:, 1]
        
        # Constrain variance to be reasonable (exp to ensure positive)
        var_delta = torch.exp(log_var_delta) + 1e-4
        return mu_delta, var_delta

