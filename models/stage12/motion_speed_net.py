import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.chomp = Chomp1d(padding)

    def forward(self, x):
        return self.chomp(self.conv(x))

class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super(TemporalResidualBlock, self).__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(4, channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm2 = nn.GroupNorm(4, channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.dropout1(self.relu1(self.norm1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.norm2(self.conv2(out))))
        return out + residual

class MotionSpeedNet(nn.Module):
    """
    Lightweight Causal MotionSpeedNet (Stage 12)
    Target parameter count: ~40k - 100k parameters.
    Default input sequence length: 50 samples (5.0 seconds at 10 Hz).
    """
    def __init__(self, in_acc_dim=4, in_gy_dim=4, hidden_dim=32, gru_dim=64):
        super(MotionSpeedNet, self).__init__()
        
        # Accelerometer branch
        self.acc_stem = nn.Sequential(
            CausalConv1d(in_acc_dim, hidden_dim, kernel_size=3, dilation=1),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU()
        )
        self.acc_tcn1 = TemporalResidualBlock(hidden_dim, kernel_size=3, dilation=1)
        self.acc_tcn2 = TemporalResidualBlock(hidden_dim, kernel_size=3, dilation=2)
        
        # Gyroscope branch
        self.gy_stem = nn.Sequential(
            CausalConv1d(in_gy_dim, hidden_dim, kernel_size=3, dilation=1),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU()
        )
        self.gy_tcn1 = TemporalResidualBlock(hidden_dim, kernel_size=3, dilation=1)
        self.gy_tcn2 = TemporalResidualBlock(hidden_dim, kernel_size=3, dilation=2)
        
        # Fusion and Temporal Representation
        fusion_dim = hidden_dim * 2  # 64
        self.gru = nn.GRU(input_size=fusion_dim, hidden_size=gru_dim, 
                          batch_first=True, num_layers=1)
        
        # Speed Heads (mean and log_var)
        self.speed_head = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.speed_log_var_head = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Yaw Rate Correction Head (predicts yaw_correction_target = vehicle_yaw_rate - phone_gyro_yaw_rate)
        self.yaw_head = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_acc, x_gy):
        """
        x_acc: (B, seq_len, in_acc_dim)
        x_gy:  (B, seq_len, in_gy_dim)
        """
        # Transpose to (B, C, L) for Conv1D
        acc = x_acc.transpose(1, 2)
        gy = x_gy.transpose(1, 2)
        
        feat_acc = self.acc_stem(acc)
        feat_acc = self.acc_tcn1(feat_acc)
        feat_acc = self.acc_tcn2(feat_acc)
        
        feat_gy = self.gy_stem(gy)
        feat_gy = self.gy_tcn1(feat_gy)
        feat_gy = self.gy_tcn2(feat_gy)
        
        # Fusion along channel dimension: (B, 64, L) -> transpose to (B, L, 64)
        fused = torch.cat([feat_acc, feat_gy], dim=1).transpose(1, 2)
        
        gru_out, _ = self.gru(fused)
        # Final step representation
        last_rep = gru_out[:, -1, :]  # (B, gru_dim)
        
        # Speed prediction: softplus ensures strictly non-negative physical speed
        raw_speed = self.speed_head(last_rep).squeeze(-1)
        speed_mean = F.softplus(raw_speed)
        
        # Log variance clamped to prevent numerical explosion / collapse
        speed_log_var = torch.clamp(self.speed_log_var_head(last_rep).squeeze(-1), min=-6.0, max=6.0)
        
        # Yaw rate correction in rad/s
        yaw_rate_correction = self.yaw_head(last_rep).squeeze(-1)
        
        return {
            'speed_mean': speed_mean,
            'speed_log_var': speed_log_var,
            'yaw_rate_correction': yaw_rate_correction
        }

def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = MotionSpeedNet()
    total_params = get_parameter_count(model)
    print(f"MotionSpeedNet instantiated successfully.")
    print(f"Total trainable parameters: {total_params:,}")
    
    # Test forward pass with 50 samples (5s)
    x_a = torch.randn(8, 50, 4)
    x_g = torch.randn(8, 50, 4)
    out = model(x_a, x_g)
    print(f"Speed mean shape: {out['speed_mean'].shape}")
    print(f"Speed log-var shape: {out['speed_log_var'].shape}")
    print(f"Yaw correction shape: {out['yaw_rate_correction'].shape}")
