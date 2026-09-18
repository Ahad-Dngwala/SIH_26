import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from models.stage12.motion_speed_net import MotionSpeedNet
from tools.baseline.data_loader import load_session

# Fixed seed for reproducible scientific benchmarks
torch.manual_seed(42)
np.random.seed(42)

WINDOW_LEN = 50  # 5.0 seconds at 10 Hz

class WindowDataset(Dataset):
    def __init__(self, x_acc_list, x_gy_list, speed_targets, yaw_targets):
        self.x_acc = torch.FloatTensor(np.array(x_acc_list))
        self.x_gy = torch.FloatTensor(np.array(x_gy_list))
        self.speed = torch.FloatTensor(np.array(speed_targets))
        self.yaw = torch.FloatTensor(np.array(yaw_targets))
        
    def __len__(self):
        return len(self.speed)
        
    def __getitem__(self, idx):
        return self.x_acc[idx], self.x_gy[idx], self.speed[idx], self.yaw[idx]

def extract_windows_from_session(df, norm_stats, stride=5):
    """
    Extracts sliding windows of length WINDOW_LEN with specified stride.
    No future leakage.
    """
    ax = (df['ax_body'].values - norm_stats['mean']['ax_body']) / norm_stats['std']['ax_body']
    ay = (df['ay_body'].values - norm_stats['mean']['ay_body']) / norm_stats['std']['ay_body']
    az = (df['az_body'].values - norm_stats['mean']['az_body']) / norm_stats['std']['az_body']
    acc_mag = (df['accel_mag'].values - norm_stats['mean']['accel_mag']) / norm_stats['std']['accel_mag']
    
    gx = (df['gx'].values - norm_stats['mean']['gx']) / norm_stats['std']['gx']
    gy = (df['gy'].values - norm_stats['mean']['gy']) / norm_stats['std']['gy']
    gz = (df['gz'].values - norm_stats['mean']['gz']) / norm_stats['std']['gz']
    gyro_mag = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    gyro_mag = (gyro_mag - norm_stats['mean']['gyro_mag']) / norm_stats['std']['gyro_mag']
    
    acc_feat = np.stack([ax, ay, az, acc_mag], axis=-1)
    gy_feat = np.stack([gx, gy, gz, gyro_mag], axis=-1)
    
    speeds = df['true_speed'].values
    # Yaw correction target: vehicle_yaw_rate (rad/s) - phone gy (rad/s)
    v_yaw_rate_rad = np.radians(df['v_yaw_rate'].values)
    yaw_targets = v_yaw_rate_rad - df['gy'].values
    
    x_acc_list = []
    x_gy_list = []
    y_speed_list = []
    y_yaw_list = []
    
    n = len(df)
    for i in range(WINDOW_LEN, n, stride):
        s_i = i - WINDOW_LEN
        # Ensure valid target values
        s_val = speeds[i]
        y_val = yaw_targets[i]
        if np.isnan(s_val) or np.isnan(y_val) or s_val < 0:
            continue
        x_acc_list.append(acc_feat[s_i:i])
        x_gy_list.append(gy_feat[s_i:i])
        y_speed_list.append(s_val)
        y_yaw_list.append(y_val)
        
    return x_acc_list, x_gy_list, y_speed_list, y_yaw_list

def compute_train_norm_stats(train_sessions, max_sessions=15):
    """
    Computes mean and std strictly on the training set.
    """
    print(f"Computing normalization statistics over training sessions...")
    acc_records = []
    gy_records = []
    
    used_sessions = 0
    for s_name in train_sessions:
        try:
            df = load_session(s_name, verbose_ts=False)
            if df is None or len(df) < 500:
                continue
            acc_records.append(df[['ax_body', 'ay_body', 'az_body', 'accel_mag']].values)
            gyro_mag = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2).values[:, None]
            gy_records.append(np.hstack([df[['gx', 'gy', 'gz']].values, gyro_mag]))
            used_sessions += 1
            if used_sessions >= max_sessions:
                break
        except Exception as e:
            continue
            
    all_acc = np.vstack(acc_records)
    all_gy = np.vstack(gy_records)
    
    norm_stats = {
        'mean': {
            'ax_body': float(np.mean(all_acc[:, 0])),
            'ay_body': float(np.mean(all_acc[:, 1])),
            'az_body': float(np.mean(all_acc[:, 2])),
            'accel_mag': float(np.mean(all_acc[:, 3])),
            'gx': float(np.mean(all_gy[:, 0])),
            'gy': float(np.mean(all_gy[:, 1])),
            'gz': float(np.mean(all_gy[:, 2])),
            'gyro_mag': float(np.mean(all_gy[:, 3])),
        },
        'std': {
            'ax_body': float(np.std(all_acc[:, 0]) + 1e-6),
            'ay_body': float(np.std(all_acc[:, 1]) + 1e-6),
            'az_body': float(np.std(all_acc[:, 2]) + 1e-6),
            'accel_mag': float(np.std(all_acc[:, 3]) + 1e-6),
            'gx': float(np.std(all_gy[:, 0]) + 1e-6),
            'gy': float(np.std(all_gy[:, 1]) + 1e-6),
            'gz': float(np.std(all_gy[:, 2]) + 1e-6),
            'gyro_mag': float(np.std(all_gy[:, 3]) + 1e-6),
        }
    }
    
    with open('models/stage12/config/norm_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print("Normalization stats saved to models/stage12/config/norm_stats.json")
    return norm_stats

def robust_heteroscedastic_loss(pred_mean, pred_log_var, target):
    """
    Heteroscedastic regression with Huber loss core:
    residual = |target - pred_mean|
    huber = 0.5 * r^2 if r <= 1.0 else (r - 0.5)
    loss = exp(-log_var) * huber + 0.5 * log_var
    """
    res = torch.abs(pred_mean - target)
    huber = torch.where(res <= 1.0, 0.5 * (res ** 2), res - 0.5)
    inv_var = torch.exp(-pred_log_var)
    loss = inv_var * huber + 0.5 * pred_log_var
    return torch.mean(loss)

def train_model():
    with open('data/processed/session_split_manifest.json') as f:
        manifest = json.load(f)
        
    train_sessions = manifest['train']
    val_sessions = manifest['validation']
    
    norm_stats = compute_train_norm_stats(train_sessions, max_sessions=15)
    
    print("\nExtracting training windows...")
    tr_acc, tr_gy, tr_spd, tr_yaw = [], [], [], []
    for s_name in train_sessions[:12]: # Representative sample of diverse sessions
        try:
            df = load_session(s_name, verbose_ts=False)
            if df is None or len(df) < 500:
                continue
            a, g, s, y = extract_windows_from_session(df, norm_stats, stride=5)
            tr_acc.extend(a)
            tr_gy.extend(g)
            tr_spd.extend(s)
            tr_yaw.extend(y)
            print(f"Loaded {s_name}: {len(s)} windows")
        except Exception as e:
            continue
            
    print(f"Total training windows: {len(tr_spd)}")
    
    print("\nExtracting validation windows...")
    va_acc, va_gy, va_spd, va_yaw = [], [], [], []
    for s_name in val_sessions:
        try:
            df = load_session(s_name, verbose_ts=False)
            if df is None:
                continue
            a, g, s, y = extract_windows_from_session(df, norm_stats, stride=5)
            va_acc.extend(a)
            va_gy.extend(g)
            va_spd.extend(s)
            va_yaw.extend(y)
            print(f"Loaded Val {s_name}: {len(s)} windows")
        except Exception as e:
            continue
            
    print(f"Total validation windows: {len(va_spd)}")
    
    train_ds = WindowDataset(tr_acc, tr_gy, tr_spd, tr_yaw)
    val_ds = WindowDataset(va_acc, va_gy, va_spd, va_yaw)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining MotionSpeedNet on {device}...")
    
    model = MotionSpeedNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    huber_yaw = nn.HuberLoss(delta=0.1)
    
    best_val_loss = float('inf')
    best_ckpt_path = 'models/stage12/checkpoints/best_motionspeednet.pth'
    
    for epoch in range(1, 16):
        model.train()
        train_loss = 0.0
        for b_acc, b_gy, b_spd, b_yaw in train_loader:
            b_acc = b_acc.to(device)
            b_gy = b_gy.to(device)
            b_spd = b_spd.to(device)
            b_yaw = b_yaw.to(device)
            
            optimizer.zero_grad()
            preds = model(b_acc, b_gy)
            
            loss_spd = robust_heteroscedastic_loss(preds['speed_mean'], preds['speed_log_var'], b_spd)
            loss_yaw = huber_yaw(preds['yaw_rate_correction'], b_yaw)
            
            loss = loss_spd + loss_yaw
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(b_spd)
            
        train_loss /= len(train_ds)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0
        val_bias = 0.0
        
        with torch.no_grad():
            for b_acc, b_gy, b_spd, b_yaw in val_loader:
                b_acc = b_acc.to(device)
                b_gy = b_gy.to(device)
                b_spd = b_spd.to(device)
                b_yaw = b_yaw.to(device)
                
                preds = model(b_acc, b_gy)
                loss_spd = robust_heteroscedastic_loss(preds['speed_mean'], preds['speed_log_var'], b_spd)
                loss_yaw = huber_yaw(preds['yaw_rate_correction'], b_yaw)
                v_loss = loss_spd + loss_yaw
                
                val_loss += v_loss.item() * len(b_spd)
                err = preds['speed_mean'] - b_spd
                val_mae += torch.sum(torch.abs(err)).item()
                val_rmse += torch.sum(err ** 2).item()
                val_bias += torch.sum(err).item()
                
        val_loss /= len(val_ds)
        val_mae /= len(val_ds)
        val_rmse = np.sqrt(val_rmse / len(val_ds))
        val_bias /= len(val_ds)
        
        print(f"Epoch {epoch:02d}/15 | Tr Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.3f} m/s | Val Bias: {val_bias:+.3f} m/s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  --> Checkpoint saved (Val Loss: {val_loss:.4f})")
            
    print(f"\nTraining Complete! Best model saved to {best_ckpt_path}")

if __name__ == '__main__':
    train_model()
