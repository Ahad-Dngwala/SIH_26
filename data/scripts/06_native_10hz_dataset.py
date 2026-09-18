import os
import json
import torch
import numpy as np
from tqdm import tqdm
import sys

# Ensure fusion_core can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tools.baseline.data_loader import load_session

def create_native_dataset(window_size=50):
    """
    Reads native 10 Hz S-dataset using data_loader.py, extracts fixed-length
    windows, and saves them as PyTorch tensors for rapid training.
    """
    manifest_path = "data/processed/split_manifest.json"
    if not os.path.exists(manifest_path):
        print("Manifest not found.")
        return
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    out_dir = f"data/processed/stage3_10hz_{window_size}"
    os.makedirs(out_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")
        X_all = []
        Y_speed = []
        
        sessions = manifest.get('sessions', {})
        split_sessions = [name for name, info in sessions.items() if info.get('split') == split]
        
        for session in tqdm(split_sessions):
            # The data loader expects 'S-S1' style names instead of 'S1'
            session_id = f"S-{session}"
                
            df = load_session(session_id)
            if df is None:
                continue
                
            # Features: ax_body, ay_body, az_body, gx, gy, gz, dt
            features = df[['ax_body', 'ay_body', 'az_body', 'gx', 'gy', 'gz', 'dt']].values
            speeds = df['true_speed'].values
            
            # Extract sliding windows (stride = window_size // 2 for train, window_size for val/test)
            stride = window_size // 2 if split == 'train' else window_size
            
            for i in range(0, len(features) - window_size, stride):
                window = features[i:i+window_size]
                # Target is the speed at the END of the window, or average?
                # The UKF predicts the state at the end of the window. Let's predict speed at the end of the window.
                target_speed = speeds[i+window_size-1]
                
                X_all.append(window)
                Y_speed.append(target_speed)
                
        if len(X_all) > 0:
            X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32)
            Y_tensor = torch.tensor(np.array(Y_speed), dtype=torch.float32)
            
            torch.save(X_tensor, os.path.join(out_dir, f"X_{split}.pt"))
            torch.save(Y_tensor, os.path.join(out_dir, f"Y_{split}.pt"))
            
            print(f"Saved {split}: X={X_tensor.shape}, Y={Y_tensor.shape}")

if __name__ == "__main__":
    create_native_dataset(window_size=50) # 5 seconds at 10Hz
