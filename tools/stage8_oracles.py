import numpy as np
from tools.baseline.data_loader import load_session
from tools.baseline.stationary_detector import detect_stationary
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState
from tools.map_matching.map_provider import RoadNetwork

def get_ground_truth_road(net, true_pos, true_heading=None):
    candidates = net.get_nearby_segments(true_pos, radius=30.0)
    if not candidates:
        return None
    # Just return the absolute closest road to the TRUE GNSS position
    # This acts as our "Perfect Road ID"
    return candidates[0]

def run_oracle(net, df_eval, mask_bo, dist, blackout_start, blackout_end, idx_end, gnss_pos, gnss_vel, is_stationary, oracle_type='id'):
    config = FusionConfig(enable_nhc=True, enable_zupt=True)
    # Using True Speed (Speed Oracle) for all map tests
    config.r_channel_a = 0.5
    config.r_channel_b = 0.5

    # Correct heading init
    initial_heading = np.arctan2(gnss_vel[0,1], gnss_vel[0,0])
    initial_state = UkfState(pos=gnss_pos[0], vel=gnss_vel[0], heading=initial_heading)
    ukf = DualChannelUkf(initial_state=initial_state, config=config)
    
    pos_traj = np.zeros((len(df_eval), 2))
    
    for i in range(1, len(df_eval)):
        dt = df_eval['dt'].iloc[i]
        t_i = df_eval['time'].iloc[i]
        in_blackout = blackout_start <= t_i <= blackout_end
        
        spd = df_eval['true_speed'].iloc[i] if in_blackout else 0.0
        g_pos = None if in_blackout else gnss_pos[i]
        g_vel = None if in_blackout else gnss_vel[i]
        yaw_rate_signal = df_eval['gy'].iloc[i-1] # Phone GY, NOT perfect yaw (to test map correction)
        
        map_pos = None
        map_head = None
        
        # Oracles activate during blackout
        if in_blackout:
            true_p = gnss_pos[i] # the GT position we're trying to match
            gt_road = get_ground_truth_road(net, true_p)
            
            if gt_road:
                # Oracle 1: Perfect Road ID (We know the exact segment, UKF position is projected onto it)
                if oracle_type == 'id':
                    # Project current UKF pos onto the GT road segment
                    p1 = gt_road['segment']['p1']
                    p2 = gt_road['segment']['p2']
                    l2 = gt_road['segment']['length']**2
                    t_proj = max(0, min(1, np.dot(ukf.ukf.x[:2] - p1, p2 - p1) / l2))
                    ukf_proj = p1 + t_proj * (p2 - p1)
                    
                    map_pos = ukf_proj
                    map_head = gt_road['segment']['heading']
                    
                # Oracle 2: Perfect Bearing (Use GT road bearing, but project UKF onto closest road among candidates)
                elif oracle_type == 'bearing':
                    map_head = gt_road['segment']['heading']
                    
                    # realistic position constraint (project UKF to NEAREST road, not necessarily GT)
                    cands = net.get_nearby_segments(ukf.ukf.x[:2], radius=30.0)
                    if cands:
                        map_pos = cands[0]['proj']
                        
                # Oracle 3: Perfect Position (Use GT road lateral position, but NO heading constraint)
                elif oracle_type == 'position':
                    p1 = gt_road['segment']['p1']
                    p2 = gt_road['segment']['p2']
                    l2 = gt_road['segment']['length']**2
                    t_proj = max(0, min(1, np.dot(ukf.ukf.x[:2] - p1, p2 - p1) / l2))
                    map_pos = p1 + t_proj * (p2 - p1)
                    map_head = None
                    
                # Oracle 4: Perfect Position + Bearing (Direct constraints from the GT road)
                elif oracle_type == 'both':
                    p1 = gt_road['segment']['p1']
                    p2 = gt_road['segment']['p2']
                    l2 = gt_road['segment']['length']**2
                    t_proj = max(0, min(1, np.dot(ukf.ukf.x[:2] - p1, p2 - p1) / l2))
                    map_pos = p1 + t_proj * (p2 - p1)
                    map_head = gt_road['segment']['heading']

        state = ukf.step(
            dt=dt, gyro_yaw=yaw_rate_signal,
            channel_a_speed=spd, channel_b_speed=spd,
            gnss_pos=g_pos, gnss_vel=g_vel,
            road_signature_pos=None, road_signature_confidence=0.0,
            is_stationary=is_stationary[i],
            map_matched_pos=map_pos,
            r_map_pos=3.0,
            map_matched_heading=map_head,
            r_map_heading=0.1
        )
        pos_traj[i] = state.pos

    err = np.linalg.norm(pos_traj[idx_end] - gnss_pos[idx_end])
    drift = (err / max(dist, 1.0)) * 100
    
    print(f'Oracle={oracle_type:10s} | Err: {err:6.2f}m | Drift: {drift:6.2f}%')

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
    
    print("--- Map Oracles ---")
    run_oracle(net, df, mask_bo, dist, blackout_start, blackout_end, idx_end, gnss_pos, gnss_vel, is_stationary, 'id')
    run_oracle(net, df, mask_bo, dist, blackout_start, blackout_end, idx_end, gnss_pos, gnss_vel, is_stationary, 'bearing')
    run_oracle(net, df, mask_bo, dist, blackout_start, blackout_end, idx_end, gnss_pos, gnss_vel, is_stationary, 'position')
    run_oracle(net, df, mask_bo, dist, blackout_start, blackout_end, idx_end, gnss_pos, gnss_vel, is_stationary, 'both')

if __name__ == '__main__':
    main()
