import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

def get_lag_correlation(v_speed, s_accel, max_lag=50):
    if len(v_speed) < max_lag * 2 or len(s_accel) < max_lag * 2:
        return 0, 0, np.zeros(0), np.zeros(0)
        
    v_norm = (v_speed - np.mean(v_speed)) / (np.std(v_speed) + 1e-6)
    s_norm = (s_accel - np.mean(s_accel)) / (np.std(s_accel) + 1e-6)
    
    corrs = np.correlate(v_norm, s_norm, mode='full') / len(v_norm)
    lags = np.arange(-len(v_norm) + 1, len(v_norm))
    
    valid_mask = (lags >= -max_lag) & (lags <= max_lag)
    valid_corrs = corrs[valid_mask]
    valid_lags = lags[valid_mask]
    
    if len(valid_corrs) == 0:
        return 0, 0, valid_lags, valid_corrs
        
    best_idx = np.argmax(valid_corrs)
    return valid_lags[best_idx], valid_corrs[best_idx], valid_lags, valid_corrs

def run_sweep():
    # Use aligned resampled files which are on 100Hz grid (dt = 10ms)
    aligned_files = glob.glob("data/raw/_resampled/*_V.parquet")
    
    print("| Session | Assumed Lag (0ms) Corr | Best Lag (ms) | Best Corr |")
    print("| ------- | ---------------------: | ------------: | --------: |")
    
    for v_f in aligned_files[:5]:
        s_f = v_f.replace("_V.parquet", "_S.parquet")
        
        try:
            v_df = pd.read_parquet(v_f)
            s_df = pd.read_parquet(s_f)
            
            # Compute acceleration from GPS speed
            v_df['v_accel'] = v_df['velocity_kmh'].diff() / 3.6
            v_df = v_df.dropna(subset=['v_accel'])
            
            # Align lengths
            min_len = min(len(v_df), len(s_df))
            v_accel = v_df['v_accel'].values[:min_len]
            s_accel = s_df['accel_y'].values[:min_len] # Assuming Y is longitudinal
            
            # Sweep +/- 100 lags (1.0s at 100Hz)
            best_lag, best_corr, lags, corrs = get_lag_correlation(v_accel, s_accel, max_lag=100)
            
            if len(corrs) > 0:
                zero_lag_idx = np.where(lags == 0)[0][0]
                zero_corr = corrs[zero_lag_idx]
                
                session_name = v_f.split('\\')[-1].replace('_V.parquet', '')
                print(f"| {session_name} | {zero_corr:.3f} | {best_lag * 10} | {best_corr:.3f} |")
                
                # Plot the first one
                if session_name == "S1":
                    plt.figure(figsize=(10, 5))
                    plt.plot(lags * 10, corrs)
                    plt.axvline(x=0, color='r', linestyle='--', label='Current Alignment (0 ms)')
                    plt.axvline(x=best_lag * 10, color='g', linestyle='--', label=f'Best Alignment ({best_lag*10} ms)')
                    plt.title('Synchronization Lag Sweep')
                    plt.xlabel('Lag (ms)')
                    plt.ylabel('Cross Correlation')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig('reports/sync_lag_sweep.png')
        except Exception as e:
            continue

if __name__ == "__main__":
    run_sweep()
