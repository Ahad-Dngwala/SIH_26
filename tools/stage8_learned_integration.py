import numpy as np
import torch
from tools.baseline.data_loader import load_session
from tools.baseline.stationary_detector import detect_stationary
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState
from tools.map_matching.map_provider import RoadNetwork
from tools.map_matching.matcher import RealisticMatcher
from models.velocity.train import SpeedCNN
import os

def extract_features(df, i, window_size=50):
    if i < window_size:
        f = df[['ax_body', 'ay_body', 'az_body', 'gx', 'gy', 'gz', 'accel_mag']].values[:i]
        pad = np.zeros((window_size - i, 7))
        f = np.vstack([pad, f])
    else:
        f = df[['ax_body', 'ay_body', 'az_body', 'gx', 'gy', 'gz', 'accel_mag']].values[i-window_size:i]
    return f

def main():
    print("Loading data...")
    df = load_session('S-S2', verbose_ts=False)
    df = df[df['dt'] > 0].reset_index(drop=True)
    t = df['time'].values
    BLACKOUT_START_ROW = 46938
    blackout_start = t[BLACKOUT_START_ROW]
    blackout_end = blackout_start + 60.0

    mask_bo = (t >= blackout_start) & (t <= blackout_end)
    true_speed = df['true_speed'].values[mask_bo]
    dt_bo = df['dt'].values[mask_bo]
    dist = np.sum(true_speed * dt_bo)
    idx_end = np.searchsorted(t, blackout_end)

    lat0, lon0 = df['lat'].iloc[0], df['lon'].iloc[0]
    lat_to_m = 111000
    lon_to_m = 111000 * np.cos(np.radians(lat0))
    gnss_pos = np.zeros((len(df), 2))
    gnss_pos[:, 0] = (df['lat'].values - lat0) * lat_to_m
    gnss_pos[:, 1] = (df['lon'].values - lon0) * lon_to_m

    # Fixed GNSS Vel
    gnss_vel = np.zeros((len(df), 2))
    for i in range(1, len(df)):
        dp = gnss_pos[i] - gnss_pos[i-1]
        norm = np.linalg.norm(dp)
        if norm > 1e-3:
            d = dp / norm
        else:
            d = np.array([1.0, 0.0]) if i == 1 else gnss_vel[i-1] / (np.linalg.norm(gnss_vel[i-1]) + 1e-9)
        gnss_vel[i] = d * df['gps_speed'].iloc[i]
    gnss_vel[0] = gnss_vel[1]

    is_stationary = detect_stationary(df)
    
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    print("Loading Map...")
    net = RoadNetwork(lat_min, lat_max, lon_min, lon_max, lat0, lon0)
    
    print("Loading Speed Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeedCNN().to(device)
    checkpoint = torch.load("models/velocity/weights/best_speed_cnn.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    f_mean = checkpoint['mean']
    f_std = checkpoint['std']
    
    def run_sim(use_map=False):
        matcher = RealisticMatcher(net)
        config = FusionConfig(enable_nhc=True, enable_zupt=True)
        # Use ML Model R (tuned up slightly due to ~2.6m/s MAE)
        config.r_channel_a = 3.0
        config.r_channel_b = 3.0

        initial_heading = np.arctan2(gnss_vel[0,1], gnss_vel[0,0])
        initial_state = UkfState(pos=gnss_pos[0], vel=gnss_vel[0], heading=initial_heading)
        ukf = DualChannelUkf(initial_state=initial_state, config=config)
        
        pos_traj = np.zeros((len(df), 2))
        
        for i in range(1, len(df)):
            dt = df['dt'].iloc[i]
            t_i = df['time'].iloc[i]
            in_blackout = blackout_start <= t_i <= blackout_end
            
            # Predict speed using ML model
            f = extract_features(df, i)
            f_norm = (f - f_mean.squeeze()) / f_std.squeeze()
            X = torch.tensor(f_norm.T, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                spd = float(model(X).item())
                
            # ML can't predict negative speed, clip to 0
            spd = max(0.0, spd)
            
            if not in_blackout:
                spd = 0.0 # Ignore ML outside blackout
                
            g_pos = None if in_blackout else gnss_pos[i]
            g_vel = None if in_blackout else gnss_vel[i]
            yaw_rate_signal = df['gy'].iloc[i-1]
            
            map_pos = None
            map_head = None
            r_map_pos = 5.0
            r_map_head = 0.5
            
            if use_map and in_blackout:
                m_pos, m_head, score = matcher.step(ukf.ukf.x[:2], ukf.ukf.x[4])
                if m_pos is not None:
                    map_pos = m_pos
                    map_head = m_head
                    if score > 20:
                        r_map_pos = 20.0
                        r_map_head = 1.0

            state = ukf.step(
                dt=dt, gyro_yaw=yaw_rate_signal,
                channel_a_speed=spd, channel_b_speed=spd,
                gnss_pos=g_pos, gnss_vel=g_vel,
                road_signature_pos=None, road_signature_confidence=0.0,
                is_stationary=is_stationary[i],
                map_matched_pos=map_pos,
                r_map_pos=r_map_pos,
                map_matched_heading=map_head,
                r_map_heading=r_map_head
            )
            pos_traj[i] = state.pos

        err = np.linalg.norm(pos_traj[idx_end] - gnss_pos[idx_end])
        drift = (err / max(dist, 1.0)) * 100
        print(f'Learned Speed (Map={use_map}) | Err: {err:6.2f}m | Drift: {drift:6.2f}%')
        
    print("--- 1D CNN Speed Tests ---")
    run_sim(use_map=False)
    run_sim(use_map=True)

if __name__ == '__main__':
    main()
