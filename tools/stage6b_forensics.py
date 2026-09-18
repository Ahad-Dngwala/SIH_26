"""
Stage 6B Forensic Investigation Script
======================================
Investigating:
1. Exact baseline reproduction
2. Raw timestamp audit (S-S1, V-S1, S-S2, V-S2)
3. Temporal lag search (-10s to +10s)
4. Vehicle yaw rate vs heading validation
5. Physical rotation and multi-axis regression
6. Sensor saturation / corruption
7. Session pairing check
8. Android orientation sensor signals
9. Corrected drift runs
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, HuberRegressor
from tools.baseline.data_loader import load_session
from tools.baseline.stationary_detector import detect_stationary
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState, fx

def run_forensics():
    print("="*70)
    print("STAGE 6B: FORENSIC INVESTIGATION & SENSOR SYNCHRONIZATION")
    print("="*70)

    # -------------------------------------------------------------
    # 1. FREEZE AND REPRODUCE BASELINE
    # -------------------------------------------------------------
    print("\n--- 1. BASELINE REPRODUCTION ---")
    df_s2 = load_session('S-S2')
    df_s2_clean = df_s2[df_s2['dt'] > 0].reset_index(drop=True)
    t = df_s2_clean['time'].values
    blackout_start = t[len(t) // 2]
    blackout_end = blackout_start + 60.0

    # Let's inspect the baseline UKF run
    config = FusionConfig(enable_nhc=True, enable_zupt=True)
    lat0, lon0 = df_s2_clean['lat'].iloc[0], df_s2_clean['lon'].iloc[0]
    lat_to_m = 111000
    lon_to_m = 111000 * np.cos(np.radians(lat0))

    # We evaluate on window around blackout or whole session
    idx_bo = (df_s2_clean['time'] >= blackout_start) & (df_s2_clean['time'] <= blackout_end)
    dist_traveled = np.sum(df_s2_clean.loc[idx_bo, 'true_speed'].values * df_s2_clean.loc[idx_bo, 'dt'].values)
    mean_speed = df_s2_clean.loc[idx_bo, 'true_speed'].mean()
    print(f"Blackout window: {blackout_start:.2f}s to {blackout_end:.2f}s (60.0s)")
    print(f"Distance traveled during blackout: {dist_traveled:.2f} m")
    print(f"Mean true speed: {mean_speed:.2f} m/s ({mean_speed*3.6:.2f} km/h)")

    # -------------------------------------------------------------
    # 2. AUDIT RAW TIMESTAMPS
    # -------------------------------------------------------------
    print("\n--- 2. RAW TIMESTAMP AUDIT ---")
    files = {
        'S-S1': "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv",
        'V-S1': "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-S1.csv",
        'S-S2': "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S2.csv",
        'V-S2': "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-S2.csv",
    }

    raw_dfs = {}
    for name, path in files.items():
        if os.path.exists(path):
            raw_df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
            raw_dfs[name] = raw_df
            print(f"\n[{name}] Total rows: {len(raw_df)}")
            # Identify time column
            time_cols = [c for c in raw_df.columns if 'TIME' in c.upper()]
            print(f"  Time columns found: {time_cols}")
            for tc in time_cols[:2]:
                vals = raw_df[tc].dropna().values
                dt_vals = np.diff(vals)
                print(f"  Column '{tc}':")
                print(f"    first: {vals[0]}, last: {vals[-1]}, duration: {vals[-1]-vals[0]}")
                print(f"    min: {vals.min()}, max: {vals.max()}")
                print(f"    dt stats: mean={dt_vals.mean():.4f}, std={dt_vals.std():.4f}, min={dt_vals.min():.4f}, max={dt_vals.max():.4f}")
                dup_count = len(raw_df) - len(raw_df.drop_duplicates(subset=[tc]))
                neg_dts = np.sum(dt_vals <= 0)
                print(f"    duplicates: {dup_count}, non-positive dts: {neg_dts}")
        else:
            print(f"File not found: {path}")

    # Inspect the exact timestamp format and headers of S-S1, V-S1, S-S2, V-S2
    print("\n--- 2B. Inspecting raw headers and first rows ---")
    for name in ['S-S1', 'V-S1', 'S-S2', 'V-S2']:
        df = raw_dfs[name]
        print(f"\n{name} columns:")
        print(df.columns.tolist()[:10])
        print(df.iloc[:2, :5])

if __name__ == '__main__':
    run_forensics()
