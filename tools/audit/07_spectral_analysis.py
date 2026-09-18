import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path

def analyze_spectrum():
    v_file = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-M.csv"
    s_file = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-M.csv"
    
    # We load S and V. To do this perfectly we should use the aligned parquets, 
    # but the aligned parquets are ALREADY INTERPOLATED to 100Hz!
    # To get RAW 10Hz/17Hz PSD grouped by speed, we have to align them ourselves roughly, 
    # or just use the V-Dataset timestamps to align with S-Dataset timestamps (since they are synced at start).
    # Since V-M and S-M are 'Synchronised V and S datasets', they have matched start times.
    
    v_df = pd.read_csv(v_file, encoding='latin1', on_bad_lines='skip', usecols=[1, 4]) # time, velocity
    s_df = pd.read_csv(s_file, encoding='latin1', on_bad_lines='skip', usecols=[7, 9, 10, 11]) # time, accel x/y/z
    
    # Drop duplicates to get true sampling rate
    s_df = s_df.drop_duplicates(subset=[s_df.columns[0]])
    
    # Convert times to seconds
    v_time = v_df.iloc[:, 0].values
    v_speed = v_df.iloc[:, 1].values / 3.6 # m/s
    
    s_time = s_df.iloc[:, 0].values / 1000.0
    s_accel_x = s_df.iloc[:, 1].values
    s_accel_y = s_df.iloc[:, 2].values
    s_accel_z = s_df.iloc[:, 3].values
    s_accel_mag = np.sqrt(s_accel_x**2 + s_accel_y**2 + s_accel_z**2)
    
    # Interpolate speed to S timestamps to categorize
    s_speed = np.interp(s_time, v_time, v_speed)
    
    regimes = {
        'Stationary': (s_speed < 0.1),
        '0-5 m/s': (s_speed >= 0.1) & (s_speed < 5),
        '5-10 m/s': (s_speed >= 5) & (s_speed < 10),
        '10-15 m/s': (s_speed >= 10) & (s_speed < 15),
        '15-20 m/s': (s_speed >= 15) & (s_speed < 20),
        '> 20 m/s': (s_speed >= 20)
    }
    
    plt.figure(figsize=(12, 8))
    
    fs = 10.0 # native sampling rate
    
    for name, mask in regimes.items():
        if np.sum(mask) < 100:
            continue
            
        data = s_accel_mag[mask]
        data = data - np.mean(data) # zero mean
        
        # Calculate PSD
        f, Pxx = welch(data, fs=fs, nperseg=min(256, len(data)))
        
        plt.semilogy(f, Pxx, label=name)
        
    plt.axvline(x=fs/2, color='r', linestyle='--', label='Nyquist (5 Hz)')
    plt.title('Power Spectral Density of Native IMU by Speed Regime')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (g^2/Hz)')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/psd_by_speed_regime.png')
    print("Saved reports/psd_by_speed_regime.png")

if __name__ == "__main__":
    analyze_spectrum()
