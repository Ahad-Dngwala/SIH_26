import pandas as pd
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState
from tools.baseline.data_loader import load_session
from tools.baseline.stationary_detector import detect_stationary

# Load models
from models.stage3_velocity.architectures import ModelB, ModelC, ModelD
model_b = ModelB()
model_b.load_state_dict(torch.load("models/stage3_velocity/weights/model_b.pt"))
model_b.eval()

model_c = ModelC()
model_c.load_state_dict(torch.load("models/stage3_velocity/weights/model_c.pt"))
model_c.eval()

model_d = ModelD()
model_d.load_state_dict(torch.load("models/stage3_velocity/weights/model_d.pt"))
model_d.eval()

def run_ml_ukf_baseline(df, blackout_start, blackout_end, model_type='B'):
    config = FusionConfig(enable_nhc=True, enable_zupt=True)
    
    lat0, lon0 = df['lat'].iloc[0], df['lon'].iloc[0]
    lat_to_m = 111000
    lon_to_m = 111000 * np.cos(np.radians(lat0))
    
    gnss_pos = np.zeros((len(df), 2))
    gnss_pos[:, 0] = (df['lat'].values - lat0) * lat_to_m
    gnss_pos[:, 1] = (df['lon'].values - lon0) * lon_to_m
    
    gnss_vel = np.zeros((len(df), 2))
    gnss_vel[:, 0] = df['gps_speed'].values 
    
    is_stationary = detect_stationary(df)
    
    initial_state = UkfState(pos=gnss_pos[0], vel=gnss_vel[0], heading=0.0)
    ukf = DualChannelUkf(initial_state=initial_state, config=config)
    
    pos = np.zeros((len(df), 2))
    vel = np.zeros((len(df), 2))
    
    # Pre-extract features for fast NN inference
    features = df[['ax_body', 'ay_body', 'az_body', 'gx', 'gy', 'gz', 'dt']].values
    window_size = 50
    
    for i in range(1, len(df)):
        dt = df['dt'].iloc[i]
        t = df['time'].iloc[i]
        in_blackout = blackout_start <= t <= blackout_end
        
        g_pos = None if in_blackout else gnss_pos[i]
        g_vel = None if in_blackout else gnss_vel[i]
        
        ukf.config.r_channel_a = 1e6
        
        # Get ML prediction if we have enough history
        predicted_speed = 0.0
        if i >= window_size:
            window = features[i - window_size : i]
            X = torch.tensor(window, dtype=torch.float32).unsqueeze(0) # (1, 50, 7)
            
            with torch.no_grad():
                if model_type == 'B':
                    predicted_speed = model_b(X).item()
                    ukf.config.r_channel_a = 2.0 # Assume fixed R
                elif model_type == 'C':
                    current_ukf_speed = np.linalg.norm(ukf.ukf.x[2:4])
                    ukf_v = torch.tensor([current_ukf_speed], dtype=torch.float32)
                    delta_v = model_c(X, ukf_v).item()
                    predicted_speed = current_ukf_speed + delta_v
                    ukf.config.r_channel_a = 2.0
                elif model_type == 'D':
                    current_ukf_speed = np.linalg.norm(ukf.ukf.x[2:4])
                    ukf_v = torch.tensor([current_ukf_speed], dtype=torch.float32)
                    mu, var = model_d(X, ukf_v)
                    predicted_speed = current_ukf_speed + mu.item()
                    ukf.config.r_channel_a = np.sqrt(var.item())
        
        # CHEAT: Inject perfect heading to isolate speed model performance
        # Without this, gyro drift destroys the heading, making even perfect speed predictions drift.
        # But wait, df doesn't have true_heading. df['gz'] integrates to heading. Let's assume gnss_vel gives true heading.
        
        state = ukf.step(
            dt=dt,
            gyro_yaw=df['gz'].iloc[i-1],
            channel_a_speed=predicted_speed,
            channel_b_speed=0.0,
            gnss_pos=g_pos,
            gnss_vel=g_vel,
            road_signature_pos=None,
            road_signature_confidence=0.0,
            is_stationary=is_stationary[i]
        )
        
        # CHEAT CONTINUED: Overwrite heading in UKF state if in blackout
        # True velocity from V-Dataset could give true heading, but we don't have it easily.
        # Instead, we just measure the models on their own merit (Velocity RMSE) rather than position drift.
        
        pos[i] = state.pos
        vel[i] = state.vel
        
    return pos, vel

def evaluate_models():
    session = 'S-S2' # Using held-out session S2
    df = load_session(session)
    if df is None:
        print("Failed to load session")
        return
        
    t = df['time'].values
    blackout_start = t[len(t)//2]
    blackout_end = blackout_start + 60.0 # 60 second blackout
    
    # Trim dataframe to speed up evaluation
    mask_eval = (df['time'] >= blackout_start - 30) & (df['time'] <= blackout_end + 30)
    df = df[mask_eval].copy().reset_index(drop=True)
    t = df['time'].values
    
    # Recalculate distance
    mask = (t >= blackout_start) & (t <= blackout_end)
    true_speed = df['true_speed'].values[mask]
    dt_blackout = df['dt'].values[mask]
    distance_traveled = np.sum(true_speed * dt_blackout)
    
    print(f"--- Stage 3 ML Blackout Evaluation on {session} (60s Blackout) ---")
    print(f"Distance Traveled: {distance_traveled:.2f} m\n")
    
    # Need to compare with Baseline E from Stage 2
    from tools.baseline.evaluate import run_ukf_baseline
    pos_E, gt_pos = run_ukf_baseline(df, blackout_start, blackout_end, enable_nhc=True, enable_zupt=True)
    idx_end = np.searchsorted(t, blackout_end)
    
    err_E = np.linalg.norm(pos_E[idx_end] - gt_pos[idx_end])
    drift_E = (err_E / max(distance_traveled, 1.0)) * 100
    print(f"Baseline E (Classical): {drift_E:.2f}% drift ({err_E:.2f}m)")
    
    for m in ['B', 'C', 'D']:
        pos, vel = run_ml_ukf_baseline(df, blackout_start, blackout_end, model_type=m)
        err = np.linalg.norm(pos[idx_end] - gt_pos[idx_end])
        drift = (err / max(distance_traveled, 1.0)) * 100
        
        ukf_speed = np.linalg.norm(vel[mask], axis=1)
        rmse = np.sqrt(np.mean((ukf_speed - true_speed)**2))
        
        print(f"Model {m}: {drift:.2f}% drift ({err:.2f}m) | Velocity RMSE: {rmse:.2f} m/s")
        print(f"  -> Mean Pred Speed: {np.mean(ukf_speed):.2f} m/s (True: {np.mean(true_speed):.2f} m/s)")
        
if __name__ == "__main__":
    evaluate_models()
