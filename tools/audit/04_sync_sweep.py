import pandas as pd
import numpy as np
import glob
from pathlib import Path

def get_lag_correlation(v_speed, s_accel, max_lag=50):
    """Simple cross correlation wrapper."""
    if len(v_speed) < max_lag * 2 or len(s_accel) < max_lag * 2:
        return 0, 0
    # Normalize
    v_norm = (v_speed - np.mean(v_speed)) / (np.std(v_speed) + 1e-6)
    s_norm = (s_accel - np.mean(s_accel)) / (np.std(s_accel) + 1e-6)
    
    corrs = np.correlate(v_norm, s_norm, mode='full') / len(v_norm)
    lags = np.arange(-len(v_norm) + 1, len(v_norm))
    
    valid_mask = (lags >= -max_lag) & (lags <= max_lag)
    valid_corrs = corrs[valid_mask]
    valid_lags = lags[valid_mask]
    
    best_idx = np.argmax(valid_corrs)
    return valid_lags[best_idx], valid_corrs[best_idx]

def sync_sweep():
    # Use the resampled aligned files which are on a uniform 100Hz grid
    aligned_files = glob.glob("data/processed/aligned/*.parquet")
    if not aligned_files:
        print("No aligned parquet files found.")
        return
        
    print("--- Synchronization Sweep (Low Speed vs High Speed) ---")
    for f in aligned_files[:3]:
        df = pd.read_parquet(f)
        
        # We need forward accel and velocity
        if 'v_velocity_kmh' not in df.columns or 's_accel_y' not in df.columns:
            continue
            
        # Differentiate velocity to get acceleration (proxy for IMU accel)
        df['v_accel'] = df['v_velocity_kmh'].diff()
        df = df.dropna()
        
        # Low speed mask
        low_speed = df[df['v_velocity_kmh'] < 15] # < 15 km/h
        high_speed = df[df['v_velocity_kmh'] >= 15]
        
        lag_all, corr_all = get_lag_correlation(df['v_accel'].values, df['s_accel_y'].values, max_lag=200)
        lag_low, corr_low = get_lag_correlation(low_speed['v_accel'].values, low_speed['s_accel_y'].values, max_lag=200)
        lag_high, corr_high = get_lag_correlation(high_speed['v_accel'].values, high_speed['s_accel_y'].values, max_lag=200)
        
        print(f"File: {Path(f).name}")
        print(f"  All data:   Lag {lag_all*10}ms, Corr {corr_all:.3f}")
        print(f"  High speed: Lag {lag_high*10}ms, Corr {corr_high:.3f}")
        print(f"  Low speed:  Lag {lag_low*10}ms, Corr {corr_low:.3f}")
        print()

if __name__ == "__main__":
    sync_sweep()
