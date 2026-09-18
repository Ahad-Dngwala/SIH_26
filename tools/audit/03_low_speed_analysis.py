import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

def analyze_low_speed():
    v_dir = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset"
    v_files = glob.glob(f"{v_dir}/*.csv")
    
    if not v_files:
        print("No V-Dataset files found.")
        return
        
    # Analyze first 5 files to save time
    results = []
    
    for f in v_files[:5]:
        try:
            df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip')
        except Exception as e:
            continue
            
        # Standardize column names (basic matching for the audit)
        cols = {
            'gps_speed': [c for c in df.columns if 'Velocity (km' in c or 'velocity' in c.lower()],
            'obd_speed': [c for c in df.columns if 'Indicated Vehicle Speed' in c],
            'wheel_fl': [c for c in df.columns if 'Wheel Speed Front Left' in c],
            'wheel_fr': [c for c in df.columns if 'Wheel Speed Front Right' in c],
            'wheel_rl': [c for c in df.columns if 'Wheel Speed Rear Left' in c],
            'wheel_rr': [c for c in df.columns if 'Wheel Speed Rear Right' in c],
            'brake': [c for c in df.columns if 'Brake Position' in c or 'Brake Pressure' in c],
            'handbrake': [c for c in df.columns if 'Handbrake' in c]
        }
        
        if not cols['gps_speed']:
            continue
            
        gps_speed_col = cols['gps_speed'][0]
        # Convert km/h to m/s
        df['gps_speed_mps'] = df[gps_speed_col] / 3.6
        
        # Mean wheel speed if available
        if cols['wheel_fl'] and cols['wheel_fr']:
            # Wheel speed is rad/s. Multiply by roughly 0.3 (tire radius) to get m/s as a proxy
            df['wheel_speed_mps'] = (df[cols['wheel_fl'][0]] + df[cols['wheel_fr'][0]]) / 2 * 0.3
        else:
            df['wheel_speed_mps'] = np.nan
            
        if cols['obd_speed']:
            df['obd_speed_mps'] = df[cols['obd_speed'][0]] / 3.6
        else:
            df['obd_speed_mps'] = np.nan
            
        # Bins
        bins = [0, 1, 2, 3, 4, 5, 10, 20, 100]
        labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-20', '20+']
        df['speed_bin'] = pd.cut(df['gps_speed_mps'], bins=bins, labels=labels, right=False)
        
        results.append(df)
        
    if not results:
        return
        
    full_df = pd.concat(results, ignore_index=True)
    
    print("--- Speed Bin Analysis ---")
    bin_stats = full_df.groupby('speed_bin').agg({
        'gps_speed_mps': ['count', 'mean', 'std'],
        'obd_speed_mps': ['mean', 'std'],
        'wheel_speed_mps': ['mean', 'std']
    }).round(3)
    print(bin_stats)
    
    print("\n--- Stationary Analysis ---")
    # Identify stationary periods using OBD or Wheel Speed
    if 'wheel_speed_mps' in full_df.columns and not full_df['wheel_speed_mps'].isna().all():
        stationary = full_df[full_df['wheel_speed_mps'] < 0.1]
    elif 'obd_speed_mps' in full_df.columns and not full_df['obd_speed_mps'].isna().all():
        stationary = full_df[full_df['obd_speed_mps'] < 0.1]
    else:
        stationary = full_df[full_df['gps_speed_mps'] < 0.5]
        
    print(f"Found {len(stationary)} stationary samples.")
    if len(stationary) > 0:
        print(f"Mean GPS Speed when stationary: {stationary['gps_speed_mps'].mean():.4f} m/s")
        print(f"Std GPS Speed when stationary: {stationary['gps_speed_mps'].std():.4f} m/s")
        print(f"% of stationary samples where GPS reports > 0.5 m/s: {(stationary['gps_speed_mps'] > 0.5).mean() * 100:.2f}%")
        print(f"% of stationary samples where GPS reports > 1.0 m/s: {(stationary['gps_speed_mps'] > 1.0).mean() * 100:.2f}%")

    print("\n--- Data Anomalies ---")
    print(f"NaNs in GPS speed: {full_df['gps_speed_mps'].isna().sum()}")
    print(f"Impossible velocities (>100 m/s): {(full_df['gps_speed_mps'] > 100).sum()}")

if __name__ == "__main__":
    analyze_low_speed()
