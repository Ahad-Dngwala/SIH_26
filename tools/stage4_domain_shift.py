import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.baseline.data_loader import load_session

def analyze_domain_shift():
    df1 = load_session('S-S1')
    df2 = load_session('S-S2')
    
    print("--- Domain Shift Analysis ---")
    print(f"{'Metric':<25} | {'S-S1 (Train)':<15} | {'S-S2 (Test)':<15}")
    print("-" * 60)
    
    metrics = [
        ('Mean Speed (m/s)', 'true_speed', np.mean),
        ('Max Speed (m/s)', 'true_speed', np.max),
        ('Mean Accel Mag', 'accel_mag', np.mean),
        ('Std Accel Mag', 'accel_mag', np.std),
        ('Mean Gyro Z (rad/s)', 'gz', np.mean),
        ('Std Gyro Z (rad/s)', 'gz', np.std),
    ]
    
    for name, col, func in metrics:
        val1 = func(df1[col].dropna())
        val2 = func(df2[col].dropna())
        print(f"{name:<25} | {val1:<15.4f} | {val2:<15.4f}")
        
if __name__ == "__main__":
    analyze_domain_shift()
