import pandas as pd
import numpy as np
import glob

def compare_ground_truth():
    v_files = glob.glob("data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/*.csv")
    
    print("--- Ground Truth Sensor Agreement ---")
    print("| Speed Regime | GPS vs OBD (MAE) | GPS vs Wheel (MAE) | Samples |")
    print("| ------------ | ----------------: | -----------------: | -------: |")
    
    results = []
    
    for f in v_files[:10]:
        try:
            df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip', 
                             usecols=[' Velocity (km/hr)', ' Indicated Vehicle Speed (km/hr)', 
                                      ' Wheel Speed Front Left (rad/sec)', ' Wheel Speed Front Right (rad/sec)'])
            
            df = df.dropna()
            
            gps = df[' Velocity (km/hr)'].values / 3.6
            obd = df[' Indicated Vehicle Speed (km/hr)'].values / 3.6
            wheel = (df[' Wheel Speed Front Left (rad/sec)'].values + df[' Wheel Speed Front Right (rad/sec)'].values) / 2 * 0.3 # approx rad to m/s
            
            df_calc = pd.DataFrame({
                'gps': gps,
                'obd': obd,
                'wheel': wheel
            })
            results.append(df_calc)
        except Exception as e:
            continue
            
    if not results:
        return
        
    full_df = pd.concat(results, ignore_index=True)
    
    regimes = {
        'Stationary': (full_df['gps'] < 0.1),
        '0-5 m/s': (full_df['gps'] >= 0.1) & (full_df['gps'] < 5),
        '5-10 m/s': (full_df['gps'] >= 5) & (full_df['gps'] < 10),
        '10-15 m/s': (full_df['gps'] >= 10) & (full_df['gps'] < 15),
        '15-20 m/s': (full_df['gps'] >= 15) & (full_df['gps'] < 20),
        '> 20 m/s': (full_df['gps'] >= 20)
    }
    
    for name, mask in regimes.items():
        sub_df = full_df[mask]
        if len(sub_df) == 0:
            continue
            
        mae_obd = np.mean(np.abs(sub_df['gps'] - sub_df['obd']))
        mae_wheel = np.mean(np.abs(sub_df['gps'] - sub_df['wheel']))
        
        print(f"| {name} | {mae_obd:.3f} m/s | {mae_wheel:.3f} m/s | {len(sub_df)} |")

if __name__ == "__main__":
    compare_ground_truth()
