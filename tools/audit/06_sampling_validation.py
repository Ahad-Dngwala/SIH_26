import pandas as pd
import numpy as np
import glob
from pathlib import Path

def validate_sampling():
    s_dir = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset"
    s_files = sorted(glob.glob(f"{s_dir}/*.csv"))
    
    if not s_files:
        print("No S-Dataset files found.")
        return
        
    print("| Session | Median Hz | Mean Hz | Samples/5s | dt std (ms) | Large gaps (>200ms) |")
    print("| ------- | --------: | ------: | ---------: | ----------: | ------------------: |")
    
    # Process a representative subset of files to save time, or all if quick enough
    # Let's do 10 representative files
    for f in s_files[:10]:
        try:
            df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip', usecols=[' TIME SINCE START (ms)'])
            time_col = ' TIME SINCE START (ms)'
            if time_col not in df.columns:
                continue
                
            times = df[time_col].dropna().values
            dt_ms = np.diff(times)
            
            # Remove negative or zero dts which indicate messy data
            dt_ms_valid = dt_ms[dt_ms > 0]
            
            if len(dt_ms_valid) == 0:
                continue
                
            median_dt = np.median(dt_ms_valid)
            mean_dt = np.mean(dt_ms_valid)
            std_dt = np.std(dt_ms_valid)
            large_gaps = np.sum(dt_ms_valid > 200)
            
            median_hz = 1000.0 / median_dt
            mean_hz = 1000.0 / mean_dt
            samples_5s = int(5.0 * mean_hz)
            
            session_name = Path(f).stem
            print(f"| {session_name} | {median_hz:.2f} | {mean_hz:.2f} | {samples_5s} | {std_dt:.2f} | {large_gaps} |")
            
        except Exception as e:
            print(f"| {Path(f).stem} | ERROR | ERROR | ERROR | ERROR | ERROR |")

if __name__ == "__main__":
    validate_sampling()
