import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools.baseline.data_loader import load_session
from tools.baseline.stationary_detector import detect_stationary
import sys
import os

# Ensure fusion_core can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState

def run_baseline_A(df):
    """Raw double integration (no UKF)"""
    pos = np.zeros((len(df), 2))
    vel = np.zeros((len(df), 2))
    heading = np.zeros(len(df))
    
    # Initialize perfectly
    pos[0] = [df['lat'].iloc[0], df['lon'].iloc[0]] # Just dummy integration in local coords
    vel[0] = [0, 0] # Assume start stationary for simplicity, or we can use first true_speed
    
    for i in range(1, len(df)):
        dt = df['dt'].iloc[i]
        
        # Heading integration
        heading[i] = heading[i-1] + df['gz'].iloc[i-1] * dt
        
        # Accelerations in body frame
        ax, ay = df['ax_body'].iloc[i-1], df['ay_body'].iloc[i-1]
        
        # Rotate to world
        ax_world = ax * np.cos(heading[i-1]) - ay * np.sin(heading[i-1])
        ay_world = ax * np.sin(heading[i-1]) + ay * np.cos(heading[i-1])
        
        vel[i] = vel[i-1] + np.array([ax_world, ay_world]) * dt
        pos[i] = pos[i-1] + (vel[i-1] + vel[i]) / 2.0 * dt
        
    return pos, vel, heading

def run_ukf_baseline(df, blackout_start, blackout_end, enable_nhc=False, enable_zupt=False):
    config = FusionConfig(enable_nhc=enable_nhc, enable_zupt=enable_zupt)
    
    # We fake lat/lon to meters by subtracting initial and scaling roughly
    lat0, lon0 = df['lat'].iloc[0], df['lon'].iloc[0]
    lat_to_m = 111000
    lon_to_m = 111000 * np.cos(np.radians(lat0))
    
    gnss_pos = np.zeros((len(df), 2))
    gnss_pos[:, 0] = (df['lat'].values - lat0) * lat_to_m
    gnss_pos[:, 1] = (df['lon'].values - lon0) * lon_to_m
    
    # Assume GNSS speed is perfectly in the direction of heading for dummy gnss_vel
    gnss_vel = np.zeros((len(df), 2))
    gnss_vel[:, 0] = df['gps_speed'].values # simplistic
    
    is_stationary = detect_stationary(df)
    
    initial_state = UkfState(
        pos=gnss_pos[0],
        vel=gnss_vel[0],
        heading=0.0 # start heading 0
    )
    ukf = DualChannelUkf(initial_state=initial_state, config=config)
    
    pos = np.zeros((len(df), 2))
    vel = np.zeros((len(df), 2))
    
    for i in range(1, len(df)):
        dt = df['dt'].iloc[i]
        
        # Simulated blackout
        t = df['time'].iloc[i]
        in_blackout = blackout_start <= t <= blackout_end
        
        g_pos = None if in_blackout else gnss_pos[i]
        g_vel = None if in_blackout else gnss_vel[i]
        
        # In classical baseline, we don't have Channel A/B (ML models)
        # So we pass 0, 0 but we must inflate their R massively so they don't corrupt the filter
        # Alternatively, we could modify UKF to skip them, but passing 0 with huge R works
        ukf.config.r_channel_a = 1e6
        ukf.config.r_channel_b = 1e6
        
        state = ukf.step(
            dt=dt,
            gyro_yaw=df['gz'].iloc[i-1],
            channel_a_speed=0.0,
            channel_b_speed=0.0,
            gnss_pos=g_pos,
            gnss_vel=g_vel,
            road_signature_pos=None,
            road_signature_confidence=0.0,
            is_stationary=is_stationary[i]
        )
        
        pos[i] = state.pos
        vel[i] = state.vel
        
    return pos, gnss_pos

def evaluate_baselines():
    session = 'S-S1'
    df = load_session(session)
    if df is None:
        print("Failed to load session")
        return
        
    # Find a 30s period for blackout
    t = df['time'].values
    blackout_start = t[len(t)//2]
    blackout_end = blackout_start + 30.0
    
    # Trim dataframe to [blackout_start - 30, blackout_end + 30] to speed up evaluation
    mask_eval = (df['time'] >= blackout_start - 30) & (df['time'] <= blackout_end + 30)
    df = df[mask_eval].copy().reset_index(drop=True)
    t = df['time'].values
    
    # Calculate ground truth distance traveled in blackout
    mask = (t >= blackout_start) & (t <= blackout_end)
    true_speed = df['true_speed'].values[mask]
    dt_blackout = df['dt'].values[mask]
    distance_traveled = np.sum(true_speed * dt_blackout)
    
    print(f"--- Blackout Evaluation on {session} ({blackout_start:.1f}s to {blackout_end:.1f}s) ---")
    print(f"Distance Traveled: {distance_traveled:.2f} m\n")
    
    pos_A, vel_A, heading_A = run_baseline_A(df)
    
    idx_start = np.searchsorted(t, blackout_start)
    idx_end = np.searchsorted(t, blackout_end)
    
    # We must align Baseline A to the UKF GNSS start position at the blackout to make a fair endpoint error comparison
    # But since Baseline A just starts at 0, 0 local, we calculate the drift during blackout
    
    distance_A = np.linalg.norm(pos_A[idx_end] - pos_A[idx_start])
    
    err_A = np.linalg.norm(pos_A[idx_end] - pos_A[idx_start] - (gt_pos[idx_end] - gt_pos[idx_start])) if 'gt_pos' in locals() else 0 # this is tricky since Baseline A has its own heading, we will just use UKF for Baseline A comparison loosely.
    
    # Actually, Baseline A drift is massive. We'll just print its distance traveled vs true distance.
    print(f"Baseline A (Raw Integration):")
    print(f"  Distance Traveled: {distance_A:.2f} m (True: {distance_traveled:.2f} m)")
    print()
    
    configs = {
        'Baseline B (UKF only)': {'enable_nhc': False, 'enable_zupt': False},
        'Baseline C (UKF + NHC)': {'enable_nhc': True, 'enable_zupt': False},
        'Baseline D (UKF + ZUPT)': {'enable_nhc': False, 'enable_zupt': True},
        'Baseline E (UKF + NHC + ZUPT)': {'enable_nhc': True, 'enable_zupt': True},
    }
    
    for name, cfg in configs.items():
        pos, gt_pos = run_ukf_baseline(df, blackout_start, blackout_end, **cfg)
        
        # Get position at end of blackout
        idx_end = np.searchsorted(t, blackout_end)
        
        err = np.linalg.norm(pos[idx_end] - gt_pos[idx_end])
        drift_pct = (err / max(distance_traveled, 1.0)) * 100
        
        print(f"{name}:")
        print(f"  Endpoint Error: {err:.2f} m")
        print(f"  Drift %:        {drift_pct:.2f} %")
        print()
        
if __name__ == "__main__":
    evaluate_baselines()
