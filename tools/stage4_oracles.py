import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState
from tools.baseline.data_loader import load_session
from tools.baseline.stationary_detector import detect_stationary

def run_oracle_benchmark(df, blackout_start, blackout_end, oracle_speed=False, oracle_heading=False):
    config = FusionConfig(enable_nhc=True, enable_zupt=True)
    
    lat0, lon0 = df['lat'].iloc[0], df['lon'].iloc[0]
    lat_to_m = 111000
    lon_to_m = 111000 * np.cos(np.radians(lat0))
    
    gnss_pos = np.zeros((len(df), 2))
    gnss_pos[:, 0] = (df['lat'].values - lat0) * lat_to_m
    gnss_pos[:, 1] = (df['lon'].values - lon0) * lon_to_m
    
    gnss_vel = np.zeros((len(df), 2))
    gnss_vel[:, 0] = df['gps_speed'].values 
    
    # Calculate GNSS heading (pseudo true heading)
    gnss_heading = np.zeros(len(df))
    for i in range(1, len(df)):
        dy = gnss_pos[i, 0] - gnss_pos[i-1, 0] # North
        dx = gnss_pos[i, 1] - gnss_pos[i-1, 1] # East
        if df['gps_speed'].iloc[i] > 1.0:
            gnss_heading[i] = np.arctan2(dx, dy) # arctan2(East, North)
        else:
            gnss_heading[i] = gnss_heading[i-1]
    
    is_stationary = detect_stationary(df)
    
    # We will initialize with the ground truth heading
    initial_state = UkfState(pos=gnss_pos[0], vel=gnss_vel[0], heading=gnss_heading[0])
    ukf = DualChannelUkf(initial_state=initial_state, config=config)
    
    pos = np.zeros((len(df), 2))
    
    for i in range(1, len(df)):
        dt = df['dt'].iloc[i]
        t = df['time'].iloc[i]
        in_blackout = blackout_start <= t <= blackout_end
        
        g_pos = None if in_blackout else gnss_pos[i]
        g_vel = None if in_blackout else gnss_vel[i]
        
        # Determine speed input
        if oracle_speed:
            speed_input = df['true_speed'].iloc[i]
            ukf.config.r_channel_a = 0.01 # High trust
        else:
            speed_input = 0.0 # Standard classical assumes 0 or purely inertial without external input
            ukf.config.r_channel_a = 1e6 # Low trust
            
        # Optional Oracle Heading injection
        if in_blackout and oracle_heading:
            ukf.ukf.x[4] = gnss_heading[i]
            
        state = ukf.step(
            dt=dt,
            gyro_yaw=df['gz'].iloc[i-1],
            channel_a_speed=speed_input,
            channel_b_speed=0.0,
            gnss_pos=g_pos,
            gnss_vel=g_vel,
            road_signature_pos=None,
            road_signature_confidence=0.0,
            is_stationary=is_stationary[i]
        )
        
        # Enforce Oracle Heading again post-update
        if in_blackout and oracle_heading:
            ukf.ukf.x[4] = gnss_heading[i]
            
        pos[i] = state.pos
        
    return pos, gnss_pos

def evaluate_oracles():
    session = 'S-S2' 
    df = load_session(session)
    if df is None: return
        
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
    
    print(f"--- Stage 4 Oracle Isolation on {session} (60s Blackout) ---")
    print(f"Distance Traveled: {distance_traveled:.2f} m\n")
    
    idx_end = np.searchsorted(t, blackout_end)
    
    # Oracles to test
    configs = [
        ("Current System (Estimated Speed & Heading)", False, False),
        ("Oracle A (Perfect Speed, Est Heading)", True, False),
        ("Oracle B (Est Speed, Perfect Heading)", False, True),
        ("Oracle C (Perfect Speed & Heading)", True, True)
    ]
    
    print(f"{'Configuration':<45} | {'Drift %':>10}")
    print("-" * 60)
    for name, s_orc, h_orc in configs:
        pos, gt_pos = run_oracle_benchmark(df, blackout_start, blackout_end, s_orc, h_orc)
        err = np.linalg.norm(pos[idx_end] - gt_pos[idx_end])
        drift = (err / max(distance_traveled, 1.0)) * 100
        print(f"{name:<45} | {drift:>9.2f}%")
        
if __name__ == "__main__":
    evaluate_oracles()
