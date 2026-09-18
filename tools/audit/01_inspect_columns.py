import pandas as pd
import glob
from pathlib import Path

v_dir = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset"
s_dir = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset"

v_files = glob.glob(f"{v_dir}/*.csv")
s_files = glob.glob(f"{s_dir}/*.csv")

print(f"Found {len(v_files)} V-Dataset files and {len(s_files)} S-Dataset files.")

if v_files:
    v_df = pd.read_csv(v_files[0], nrows=100, encoding='latin1')
    print("\n--- V-Dataset Columns ---")
    for col in v_df.columns:
        print(f"- {col}")

if s_files:
    s_df = pd.read_csv(s_files[0], nrows=100, encoding='latin1')
    print("\n--- S-Dataset Columns ---")
    for col in s_df.columns:
        print(f"- {col}")
        
    print("\n--- S-Dataset Sampling Rate Check ---")
    s_full = pd.read_csv(s_files[0], usecols=[7], encoding='latin1') 
    time_col = s_full.columns[0]
    diffs = s_full[time_col].diff().dropna() / 1000.0 # Convert ms to seconds
    mean_diff = diffs.mean()
    print(f"Mean time difference: {mean_diff:.5f} s")
    print(f"Implied sample rate: {1/mean_diff:.2f} Hz")
    print(f"Samples in 2s window: {2 / mean_diff:.1f}")
    print(f"Samples in 5s window: {5 / mean_diff:.1f}")
    print(f"Samples in 10s window: {10 / mean_diff:.1f}")
