"""
Stage 6 Alignment Analysis: Find the best phone-to-vehicle alignment
for all sessions, then estimate scale/bias calibration.
"""
import pandas as pd, numpy as np, sys, glob, os
sys.path.append('.')
from scipy.stats import pearsonr

BASE_S = 'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset'
BASE_V = 'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset'

SESSIONS = ['S-S1', 'S-S2']  # sessions with both S and V files

def load_raw_aligned(session):
    s_file = f'{BASE_S}/{session}.csv'
    v_name = session.replace('S-', 'V-')
    v_file = f'{BASE_V}/{v_name}.csv'
    if not os.path.exists(s_file) or not os.path.exists(v_file):
        return None
    s_df = pd.read_csv(s_file, encoding='latin1', on_bad_lines='skip')
    v_df = pd.read_csv(v_file, encoding='latin1', on_bad_lines='skip')
    
    col_gx = [c for c in s_df.columns if 'GYROSCOPE X' in c.upper()][0]
    col_gy = [c for c in s_df.columns if 'GYROSCOPE Y' in c.upper()][0]
    col_gz = [c for c in s_df.columns if 'GYROSCOPE Z' in c.upper()][0]
    col_ax = [c for c in s_df.columns if 'ACCELEROMETER X' in c.upper()][0]
    col_ay = [c for c in s_df.columns if 'ACCELEROMETER Y' in c.upper()][0]
    col_yaw = [c for c in v_df.columns if 'YAW RATE' in c.upper()][0]
    col_long_acc = [c for c in v_df.columns if 'LONGITUDINAL' in c.upper()][0]
    col_lat_acc = [c for c in v_df.columns if 'LATERAL' in c.upper()][0]
    col_vhead = [c for c in v_df.columns if 'HEADING' in c.upper()][0]
    col_vt = [c for c in v_df.columns if 'TIME SINCE START' in c.upper()][0]
    col_st = [c for c in s_df.columns if 'TIME' in c.upper()][0]
    col_vel = [c for c in v_df.columns if 'VELOCITY (KM/HR)' in c.upper()][0]
    
    s_t = s_df[col_st].values / 1000.0; s_t -= s_t[0]
    v_t = v_df[col_vt].values; v_t -= v_t[0]
    
    v_yaw = np.interp(s_t, v_t, v_df[col_yaw].values)  # deg/s
    v_long = np.interp(s_t, v_t, v_df[col_long_acc].values)  # g
    v_lat = np.interp(s_t, v_t, v_df[col_lat_acc].values)    # g
    v_head = np.interp(s_t, v_t, v_df[col_vhead].values)  # degrees
    v_speed = np.interp(s_t, v_t, v_df[col_vel].values) / 3.6
    
    gx = s_df[col_gx].values * 180/np.pi  # to deg/s
    gy = s_df[col_gy].values * 180/np.pi
    gz = s_df[col_gz].values * 180/np.pi
    ax = s_df[col_ax].values
    ay = s_df[col_ay].values
    
    return {
        'time': s_t, 'gx': gx, 'gy': gy, 'gz': gz,
        'ax': ax, 'ay': ay,
        'v_yaw': v_yaw, 'v_long': v_long, 'v_lat': v_lat,
        'v_head': v_head, 'v_speed': v_speed
    }

print("=" * 70)
print("STAGE 6: Phone-to-Vehicle Alignment Audit")
print("=" * 70)

all_gy = []; all_gz = []; all_gx = []
all_vyaw = []

for sess in SESSIONS:
    data = load_raw_aligned(sess)
    if data is None:
        print(f"  {sess}: MISSING")
        continue
    
    # Moving only
    moving = data['v_speed'] > 1.0
    
    gy_m = data['gy'][moving]
    gz_m = data['gz'][moving]
    gx_m = data['gx'][moving]
    vyaw_m = data['v_yaw'][moving]
    
    print(f"\n--- {sess} (moving samples: {moving.sum()}) ---")
    for name, sig in [('gx', gx_m), ('gy', gy_m), ('gz', gz_m)]:
        if len(sig) < 10: continue
        c, _ = pearsonr(sig, vyaw_m)
        scale = np.cov(sig, vyaw_m)[0,1] / max(np.var(sig), 1e-9)
        bias = np.mean(vyaw_m) - scale*np.mean(sig)
        print(f"  {name} vs v_yaw: r={c:.4f}  scale={scale:.4f}  bias(deg/s)={bias:.4f}")
    
    all_gy.extend(gy_m.tolist())
    all_gz.extend(gz_m.tolist())
    all_gx.extend(gx_m.tolist())
    all_vyaw.extend(vyaw_m.tolist())

print("\n--- POOLED (S-S1 + S-S2 moving) ---")
all_gy = np.array(all_gy)
all_gz = np.array(all_gz)
all_gx = np.array(all_gx)
all_vyaw = np.array(all_vyaw)

for name, sig in [('gx', all_gx), ('gy', all_gy), ('gz', all_gz)]:
    c, _ = pearsonr(sig, all_vyaw)
    scale = np.cov(sig, all_vyaw)[0,1] / max(np.var(sig), 1e-9)
    bias = np.mean(all_vyaw) - scale*np.mean(sig)
    print(f"  {name} vs v_yaw: r={c:.4f}  scale={scale:.4f}  bias(deg/s)={bias:.4f}")
