import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState, fx
from tools.baseline.data_loader import load_session
from tools.baseline.stationary_detector import detect_stationary

def run_oracle_experiment(df, blackout_start, blackout_end, oracle_mode):
    # Disable NHC for GT Vector because NHC forces velocity to align with drifting heading!
    nhc = False if oracle_mode == "GT_VECTOR" else True
    config = FusionConfig(enable_nhc=nhc, enable_zupt=True)
    
    lat0, lon0 = df['lat'].iloc[0], df['lon'].iloc[0]
    lat_to_m = 111000
    lon_to_m = 111000 * np.cos(np.radians(lat0))
    
    gnss_pos = np.zeros((len(df), 2))
    gnss_pos[:, 0] = (df['lat'].values - lat0) * lat_to_m
    gnss_pos[:, 1] = (df['lon'].values - lon0) * lon_to_m
    
    gnss_vel = np.zeros((len(df), 2))
    gnss_vel[:, 0] = df['gps_speed'].values 
    
    # Calculate GT Velocity Vector (v_N, v_E) properly
    # gnss_pos freezes frequently, so we calculate heading only on changes and forward-fill
    gnss_heading = np.zeros(len(df))
    last_heading = 0.0
    for i in range(1, len(df)):
        dy = gnss_pos[i, 0] - gnss_pos[i-1, 0] # North
        dx = gnss_pos[i, 1] - gnss_pos[i-1, 1] # East
        if abs(dx) > 1e-3 or abs(dy) > 1e-3:
            last_heading = np.arctan2(dx, dy)
        gnss_heading[i] = last_heading
        
    # Smooth the heading slightly to remove GPS zig-zag
    gnss_heading = pd.Series(gnss_heading).rolling(window=50, min_periods=1, center=True).mean().values
    
    # GT Vector = True Speed * True Heading
    gt_v_N = df['true_speed'].values * np.cos(gnss_heading)
    gt_v_E = df['true_speed'].values * np.sin(gnss_heading)
    
    is_stationary = detect_stationary(df)
    
    initial_state = UkfState(pos=gnss_pos[0], vel=gnss_vel[0], heading=0.0)
    ukf = DualChannelUkf(initial_state=initial_state, config=config)
    
    pos = np.zeros((len(df), 2))
    
    for i in range(1, len(df)):
        dt = df['dt'].iloc[i]
        t = df['time'].iloc[i]
        in_blackout = blackout_start <= t <= blackout_end
        
        g_pos = None if in_blackout else gnss_pos[i]
        g_vel = None if in_blackout else gnss_vel[i]
        
        speed_input = 0.0
        ukf.config.r_channel_a = 1e6
        
        if oracle_mode == "GT_SCALAR":
            speed_input = df['true_speed'].iloc[i]
            ukf.config.r_channel_a = 0.01
            state = ukf.step(
                dt=dt, gyro_yaw=df['gz'].iloc[i-1],
                channel_a_speed=speed_input, channel_b_speed=0.0,
                gnss_pos=g_pos, gnss_vel=g_vel,
                road_signature_pos=None, road_signature_confidence=0.0,
                is_stationary=is_stationary[i]
            )
            
        elif oracle_mode == "GT_VECTOR":
            # Pass the true velocity vector as a measurement with very high confidence
            r_mult = ukf._gnss_r_multiplier(dt, gnss_available=False)
            
            ukf.ukf.Q = ukf._process_noise(dt, True)
            ukf.ukf.predict(dt=dt, fx=fx, gyro_yaw=df['gz'].iloc[i-1])
            
            # The UKF's hx_velocity returns [v_n, v_e]. 
            # We provide the ground-truth vector as the measurement z.
            z_a = np.array([gt_v_N[i], gt_v_E[i]])
            
            # We trust this ground-truth vector completely
            def hx_vel(x):
                return np.array([x[2], x[3]])
                
            ukf.ukf.update(z_a, R=np.eye(2) * (1e-4)**2, hx=hx_vel)
            ukf._symmetrize_p()
            
            state = UkfState(
                pos=ukf.ukf.x[0:2].copy(),
                vel=ukf.ukf.x[2:4].copy(),
                heading=ukf.ukf.x[4]
            )
            
        else:
            state = ukf.step(
                dt=dt, gyro_yaw=df['gz'].iloc[i-1],
                channel_a_speed=0.0, channel_b_speed=0.0,
                gnss_pos=g_pos, gnss_vel=g_vel,
                road_signature_pos=None, road_signature_confidence=0.0,
                is_stationary=is_stationary[i]
            )
            
        pos[i] = state.pos
        
    return pos, gnss_pos

def evaluate_stage5a():
    session = 'S-S2' 
    df = load_session(session)
    
    t = df['time'].values
    blackout_start = t[len(t)//2]
    blackout_end = blackout_start + 60.0 
    
    mask_eval = (df['time'] >= blackout_start - 30) & (df['time'] <= blackout_end + 30)
    df = df[mask_eval].copy().reset_index(drop=True)
    t = df['time'].values
    
    mask = (t >= blackout_start) & (t <= blackout_end)
    true_speed = df['true_speed'].values[mask]
    dt_blackout = df['dt'].values[mask]
    distance_traveled = np.sum(true_speed * dt_blackout)
    
    print(f"--- Stage 5A UKF Forensics on {session} (60s Blackout) ---")
    print(f"Distance Traveled: {distance_traveled:.2f} m\n")
    
    idx_end = np.searchsorted(t, blackout_end)
    
    configs = [
        ("Current Baseline", "BASELINE"),
        ("GT Scalar Speed", "GT_SCALAR"),
        ("GT Velocity Vector", "GT_VECTOR")
    ]
    
    print(f"{'Configuration':<25} | {'Drift %':>10} | {'Endpoint Error':>15}")
    print("-" * 60)
    for name, mode in configs:
        pos, gt_pos = run_oracle_experiment(df, blackout_start, blackout_end, mode)
        
        idx_start = np.searchsorted(t, blackout_start)
        idx_end = np.searchsorted(t, blackout_end)
        
        err = np.linalg.norm(pos[idx_end] - gt_pos[idx_end])
        drift = (err / max(distance_traveled, 1.0)) * 100
        print(f"{name:<25} | {drift:>9.2f}% | {err:>13.2f} m")
        if mode == "GT_VECTOR":
            print(f"    UKF Start Pos: {pos[idx_start]}")
            print(f"    GT Start Pos:  {gt_pos[idx_start]}")
            print(f"    UKF End Pos: {pos[idx_end]}")
            print(f"    GT End Pos:  {gt_pos[idx_end]}")

if __name__ == "__main__":
    evaluate_stage5a()
