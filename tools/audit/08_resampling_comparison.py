import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def compare_resampling():
    s_raw_file = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv"
    
    # Check if resampled file exists
    s_resampled_file = "data/raw/_resampled/S1_S.parquet"
    if not os.path.exists(s_resampled_file):
        print(f"File {s_resampled_file} not found. Resampling pipeline hasn't been fully run for this session.")
        return
        
    raw_df = pd.read_csv(s_raw_file, encoding='latin1', on_bad_lines='skip', usecols=[7, 9, 10, 11])
    raw_df = raw_df.drop_duplicates(subset=[raw_df.columns[0]])
    raw_time = raw_df.iloc[:, 0].values / 1000.0
    raw_accel_x = raw_df.iloc[:, 1].values
    
    resampled_df = pd.read_parquet(s_resampled_file)
    res_time = resampled_df['time_s'].values
    res_accel_x = resampled_df['accel_x'].values
    
    # Find a low speed window to plot (e.g., around 100-110s where it might be stationary/low speed)
    # Just plot the first 10 seconds of data for comparison
    start_t = max(raw_time[0], res_time[0]) + 100
    end_t = start_t + 5
    
    mask_raw = (raw_time >= start_t) & (raw_time <= end_t)
    mask_res = (res_time >= start_t) & (res_time <= end_t)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(res_time[mask_res], res_accel_x[mask_res], 'b-', alpha=0.7, label='Interpolated (100 Hz)')
    plt.plot(raw_time[mask_raw], raw_accel_x[mask_raw], 'ro', markersize=4, label='Native (10 Hz)')
    
    plt.title('Time Domain Comparison: Native vs Interpolated')
    plt.xlabel('Time (s)')
    plt.ylabel('Accel X')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/native_vs_interpolated.png')
    print("Saved reports/native_vs_interpolated.png")
    
    # Calculate variance before and after
    print(f"Variance (Raw): {np.var(raw_accel_x):.4f}")
    print(f"Variance (Interpolated): {np.var(res_accel_x):.4f}")
    
    # Calculate derivative statistics (jerk)
    raw_jerk = np.diff(raw_accel_x) / np.diff(raw_time)
    res_jerk = np.diff(res_accel_x) / np.diff(res_time)
    
    print(f"Mean Absolute Jerk (Raw): {np.mean(np.abs(raw_jerk)):.4f}")
    print(f"Mean Absolute Jerk (Interpolated): {np.mean(np.abs(res_jerk)):.4f}")

if __name__ == "__main__":
    compare_resampling()
