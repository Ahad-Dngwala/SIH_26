import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tools.baseline.data_loader import load_session

class VelocityDataset(Dataset):
    def __init__(self, sessions, window_size=50, step=10):
        self.X = []
        self.y = []
        
        for session in sessions:
            df = load_session(session, verbose_ts=False)
            if df is None: continue
            
            # Features: ax, ay, az, gx, gy, gz, accel_mag
            features = df[['ax_body', 'ay_body', 'az_body', 'gx', 'gy', 'gz', 'accel_mag']].values
            target = df['true_speed'].values
            
            for i in range(0, len(features) - window_size, step):
                window_f = features[i:i+window_size]
                # Target is the speed at the END of the window
                speed = target[i+window_size-1]
                
                # Exclude if it has NaNs
                if np.isnan(window_f).any() or np.isnan(speed):
                    continue
                
                self.X.append(window_f)
                self.y.append(speed)
                
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
        # Standardize features
        self.mean = np.mean(self.X, axis=(0, 1), keepdims=True)
        self.std = np.std(self.X, axis=(0, 1), keepdims=True) + 1e-6
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 1D CNN expects shape (channels, length)
        return torch.tensor(self.X[idx].T), torch.tensor(self.y[idx])

class SpeedCNN(nn.Module):
    def __init__(self, in_channels=7, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(hidden * 12, 32), # 50 -> 25 -> 12
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_model():
    print("Loading data...")
    # S-S1 is train, S-S2 is test
    train_dataset = VelocityDataset(["S-S1"])
    test_dataset = VelocityDataset(["S-S2"])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeedCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    best_loss = float('inf')
    for epoch in range(15):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += criterion(pred, y).item() * len(X)
        test_loss /= len(test_dataset)
        
        print(f"Epoch {epoch+1:2d} | Train MAE: {train_loss:.3f} m/s | Test MAE: {test_loss:.3f} m/s")
        
        if test_loss < best_loss:
            best_loss = test_loss
            os.makedirs("models/velocity/weights", exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'mean': train_dataset.mean,
                'std': train_dataset.std
            }, "models/velocity/weights/best_speed_cnn.pth")
            
    print(f"Best Test MAE: {best_loss:.3f} m/s")

if __name__ == "__main__":
    train_model()
