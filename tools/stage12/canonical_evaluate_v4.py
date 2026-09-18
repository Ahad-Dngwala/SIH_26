import os
import json
import numpy as np
import pandas as pd
import torch
from tools.baseline.data_loader import load_session
from models.stage12.motion_speed_net import MotionSpeedNet
from tools.stage12.corridor_filter import CorridorRoad, CorridorFilter, RoadProgressMatcher
from tools.map_matching.map_provider import RoadNetwork
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState

# FROZEN BENCHMARK CONSTANTS (STAGE 12)
BLACKOUT_START = 4693.80
BLACKOUT_DURATION = 60.0
REFERENCE_DISTANCE = 371.708
ENDPOINT_THRESHOLD = 37.171  # 10%
ENDPOINT_THRESHOLD_5 = 18.586 # 5%

def load_canonical_benchmark_data():
    df = load_session('S-S2', verbose_ts=False)
    df = df[df['dt'] > 0].reset_index(drop=True)
    
    t = df['time'].values
    idx_start = np.searchsorted(t, BLACKOUT_START)
    idx_end = np.searchsorted(t, BLACKOUT_START + BLACKOUT_DURATION)
    
    lat0, lon0 = df['lat'].iloc[0], df['lon'].iloc[0]
    lat_to_m = 111000.0
    lon_to_m = 111000.0 * np.cos(np.radians(lat0))
    
    gnss_pos = np.zeros((len(df), 2))
    gnss_pos[:, 0] = (df['lat'].values - lat0) * lat_to_m # North
    gnss_pos[:, 1] = (df['lon'].values - lon0) * lon_to_m # East
    
    # GNSS velocity vector
    gnss_vel = np.zeros((len(df), 2))
    for i in range(1, len(df)):
        dp = gnss_pos[i] - gnss_pos[i-1]
        norm_dp = np.linalg.norm(dp)
        if norm_dp > 1e-3:
            d = dp / norm_dp
        else:
            d = np.array([1.0, 0.0]) if i == 1 else gnss_vel[i-1] / (np.linalg.norm(gnss_vel[i-1]) + 1e-9)
        gnss_vel[i] = d * df['gps_speed'].iloc[i]
    gnss_vel[0] = gnss_vel[1]
    
    # Smoothed pre-blackout heading (atan2 East/North)
    gnss_hdg = np.arctan2(gnss_vel[:, 1], gnss_vel[:, 0])
    gnss_hdg_smooth = pd.Series(np.unwrap(gnss_hdg)).rolling(50, min_periods=1, center=False).mean().values
    gnss_hdg_smooth = (gnss_hdg_smooth + np.pi) % (2 * np.pi) - np.pi
    
    return df, gnss_pos, gnss_vel, gnss_hdg_smooth, idx_start, idx_end, lat0, lon0

def run_model_inference(df, idx_start, idx_end):
    """
    Runs causal inference using MotionSpeedNet with 50-sample windows (5.0s at 10 Hz).
    No future leakage.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MotionSpeedNet().to(device)
    model.load_state_dict(torch.load('models/stage12/checkpoints/best_motionspeednet.pth', map_location=device, weights_only=False))
    model.eval()
    
    with open('models/stage12/config/norm_stats.json') as f:
        norm = json.load(f)
        
    ax = (df['ax_body'].values - norm['mean']['ax_body']) / norm['std']['ax_body']
    ay = (df['ay_body'].values - norm['mean']['ay_body']) / norm['std']['ay_body']
    az = (df['az_body'].values - norm['mean']['az_body']) / norm['std']['az_body']
    acc_mag = (df['accel_mag'].values - norm['mean']['accel_mag']) / norm['std']['accel_mag']
    
    gx = (df['gx'].values - norm['mean']['gx']) / norm['std']['gx']
    gy = (df['gy'].values - norm['mean']['gy']) / norm['std']['gy']
    gz = (df['gz'].values - norm['mean']['gz']) / norm['std']['gz']
    gyro_mag = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    gyro_mag = (gyro_mag - norm['mean']['gyro_mag']) / norm['std']['gyro_mag']
    
    acc_feat = np.stack([ax, ay, az, acc_mag], axis=-1)
    gy_feat = np.stack([gx, gy, gz, gyro_mag], axis=-1)
    
    # We infer over the pre-blackout calibration window (30s = 300 steps) and blackout window
    calib_start = max(0, idx_start - 300)
    
    pred_speed = np.zeros(len(df))
    pred_unc = np.zeros(len(df))
    pred_yaw_corr = np.zeros(len(df))
    
    window_len = 50
    for i in range(calib_start, idx_end + 1):
        s_i = max(0, i - window_len)
        x_a = acc_feat[s_i:i]
        x_g = gy_feat[s_i:i]
        
        pad_len = window_len - len(x_a)
        if pad_len > 0:
            x_a = np.pad(x_a, ((pad_len, 0), (0, 0)), mode='edge')
            x_g = np.pad(x_g, ((pad_len, 0), (0, 0)), mode='edge')
            
        X_a = torch.FloatTensor(x_a).unsqueeze(0).to(device)
        X_g = torch.FloatTensor(x_g).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(X_a, X_g)
            pred_speed[i] = float(out['speed_mean'].item())
            pred_unc[i] = float(torch.exp(out['speed_log_var']).item())
            pred_yaw_corr[i] = float(out['yaw_rate_correction'].item())
            
    return pred_speed, pred_unc, pred_yaw_corr, calib_start

def calibrate_pre_blackout(df, pred_speed, calib_start, idx_start):
    """
    Phase 10: Fit affine & EWMA speed calibration strictly on pre-blackout GNSS-on data.
    """
    gps_s = df['gps_speed'].values[calib_start:idx_start]
    mod_s = pred_speed[calib_start:idx_start]
    
    # Robust regression: gps_s ≈ a * mod_s + b
    A = np.vstack([mod_s, np.ones(len(mod_s))]).T
    res, _, _, _ = np.linalg.lstsq(A, gps_s, rcond=None)
    a_aff, b_aff = res[0], res[1]
    
    # Safe bounds on calibration slope
    a_aff = np.clip(a_aff, 0.5, 2.0)
    
    # EWMA bias correction
    ewma_bias = float(np.mean(gps_s[-100:] - mod_s[-100:])) # last 10 seconds
    
    return a_aff, b_aff, ewma_bias

def run_2d_ukf(df, idx_start, idx_end, gnss_pos, gnss_vel, gnss_hdg, speed_arr, yaw_rate_arr, unc_arr):
    """
    Runs the canonical 2-D UKF strictly during blackout.
    """
    pos_start = gnss_pos[idx_start]
    vel_start = gnss_vel[idx_start]
    hdg_start = gnss_hdg[idx_start]
    
    config = FusionConfig()
    config.covariance_decoupling = True
    config.r_channel_a = 0.5
    
    initial_state = UkfState(pos=pos_start, vel=vel_start, heading=hdg_start)
    ukf = DualChannelUkf(initial_state=initial_state, config=config)
    
    final_heading = hdg_start
    for i in range(idx_start + 1, idx_end + 1):
        dt = df['dt'].iloc[i]
        s = speed_arr[i]
        r = yaw_rate_arr[i]
        u = unc_arr[i]
        
        ukf.config.r_channel_a = max(0.25, u)
        ukf.step(
            dt=dt, gyro_yaw=r,
            channel_a_speed=s, channel_b_speed=None,
            gnss_pos=None, gnss_vel=None,
            road_signature_pos=None, road_signature_confidence=0.0
        )
        final_heading = ukf.ukf.x[4]
        
    final_pos = ukf.ukf.x[:2]
    return final_pos, final_heading

def build_corridor_from_map(rn: RoadNetwork, pos_start, hdg_start):
    """
    Phase 12: Initialize corridor hypothesis legitimately using pre-blackout GNSS state.
    """
    cands = rn.get_nearby_segments(pos_start, radius=50.0)
    best_cand = None
    best_score = -float('inf')
    
    for c in cands:
        dist = c['dist']
        seg = c['segment']
        dh = (seg['heading'] - hdg_start + np.pi) % (2 * np.pi) - np.pi
        # Score penalizes distance and heading disagreement
        score = - (dist / 10.0)**2 - (dh / 0.5)**2
        if score > best_score:
            best_score = score
            best_cand = c
            
    # Build continuous downstream road
    road = CorridorRoad(rn.segments, initial_idx=best_cand['index'] if 'index' in best_cand else 0)
    return road

def run_stage12_evaluator():
    print("================================================================================")
    print("STAGE 12: SUB-10 CORRIDOR NAVIGATION RESCUE BENCHMARK")
    print("================================================================================")
    df, gnss_pos, gnss_vel, gnss_hdg, idx_start, idx_end, lat0, lon0 = load_canonical_benchmark_data()
    
    dt_arr = df['dt'].values[idx_start+1:idx_end+1]
    num_samples = idx_end - idx_start
    total_time = np.sum(dt_arr)
    true_final_pos = gnss_pos[idx_end]
    true_final_hdg = gnss_hdg[idx_end]
    true_speeds = df['true_speed'].values[idx_start+1:idx_end+1]
    true_dist = np.sum(true_speeds * dt_arr)
    
    print(f"Benchmark Verification:")
    print(f"  Blackout Start Time: {df['time'].iloc[idx_start]:.2f} s")
    print(f"  Blackout Samples:    {num_samples} steps (~10 Hz)")
    print(f"  Blackout Duration:   {total_time:.2f} s")
    print(f"  Reference Distance:  {REFERENCE_DISTANCE:.3f} m (Actual integrated: {true_dist:.3f} m)")
    print(f"  Threshold <10%:      {ENDPOINT_THRESHOLD:.3f} m")
    print(f"  Threshold <5%:       {ENDPOINT_THRESHOLD_5:.3f} m")
    print("--------------------------------------------------------------------------------")
    
    # 1. SIMPLE CONTROLS (PHASE 9)
    # Control 1: Hold last pre-blackout GNSS speed
    last_gnss_speed = df['gps_speed'].iloc[idx_start]
    ctrl1_speeds = np.full(len(df), last_gnss_speed)
    
    # Control 2: Exponentially smoothed pre-blackout GNSS speed (last 5 seconds)
    smooth_gnss_speed = df['gps_speed'].iloc[idx_start-50:idx_start].ewm(span=20).mean().iloc[-1]
    ctrl2_speeds = np.full(len(df), smooth_gnss_speed)
    
    # Gyro yaw rate input (aligned phone gyro)
    aligned_yaw_rate = df['gy'].values
    
    # 2. RUN LEARNED INFERENCE (PHASE 6 & 7)
    print("Running MotionSpeedNet inference...")
    pred_speed, pred_unc, pred_yaw_corr, calib_start = run_model_inference(df, idx_start, idx_end)
    
    # Yaw rate with predicted correction (Phase 2)
    corrected_yaw_rate = aligned_yaw_rate + pred_yaw_corr
    
    # 3. PRE-BLACKOUT SPEED CALIBRATION (PHASE 10)
    a_aff, b_aff, ewma_bias = calibrate_pre_blackout(df, pred_speed, calib_start, idx_start)
    calib_speed_affine = np.clip(a_aff * pred_speed + b_aff, 0.0, 40.0)
    calib_speed_ewma = np.clip(pred_speed + ewma_bias, 0.0, 40.0)
    calib_speed_combined = np.clip(a_aff * pred_speed + b_aff + 0.5 * ewma_bias, 0.0, 40.0)
    
    print(f"Pre-blackout Speed Calibration Fitted:")
    print(f"  Affine: a = {a_aff:.3f}, b = {b_aff:.3f} m/s")
    print(f"  EWMA bias: {ewma_bias:+.3f} m/s")
    print("--------------------------------------------------------------------------------")
    
    # 4. ROAD NETWORK & CORRIDOR SETUP (PHASE 11-14)
    rn = RoadNetwork(df['lat'].min(), df['lat'].max(), df['lon'].min(), df['lon'].max(), lat0, lon0)
    pos_start = gnss_pos[idx_start]
    hdg_start = gnss_hdg[idx_start]
    
    # Find candidate segment with best match
    cands = rn.get_nearby_segments(pos_start, radius=50.0)
    best_cand = min(cands, key=lambda c: c['dist'])
    best_idx = 0
    for idx, s in enumerate(rn.segments):
        if s is best_cand['segment']:
            best_idx = idx
            break
    road = CorridorRoad(rn.segments, initial_idx=best_idx)
    road_matcher = RoadProgressMatcher(road)
    
    # Results store
    scorecard = []
    
    def evaluate_configuration(name, speeds, yaw_rates, uncs, mode="2D_UKF", use_anchors=False):
        s_blackout = speeds[idx_start+1:idx_end+1]
        y_blackout = yaw_rates[idx_start+1:idx_end+1]
        u_blackout = uncs[idx_start+1:idx_end+1]
        
        pred_dist = float(np.sum(s_blackout * dt_arr))
        dist_err = float(np.abs(pred_dist - REFERENCE_DISTANCE))
        spd_bias = float(np.mean(s_blackout - true_speeds))
        
        if mode == "2D_UKF":
            final_pos, final_hdg = run_2d_ukf(df, idx_start, idx_end, gnss_pos, gnss_vel, gnss_hdg, speeds, yaw_rates, uncs)
            hdg_err = float(np.degrees(np.abs((final_hdg - true_final_hdg + np.pi) % (2*np.pi) - np.pi)))
        elif mode == "CORRIDOR":
            c_filter = CorridorFilter(road, s_init=0.0, v_init=speeds[idx_start])
            cum_imu_turn = 0.0
            for i in range(idx_start + 1, idx_end + 1):
                dt = df['dt'].iloc[i]
                c_filter.predict(dt)
                c_filter.update_speed(speeds[i], uncs[i])
                
                # Turn tracking
                cum_imu_turn += yaw_rates[i] * dt
                
                if use_anchors:
                    s_anchor, conf = road_matcher.match_progress(c_filter.x[0], cum_imu_turn)
                    if conf > 0.6:
                        c_filter.update_progress_anchor(s_anchor, conf)
                        
            final_pos, final_hdg = c_filter.get_position()
            hdg_err = float(np.degrees(np.abs((final_hdg - true_final_hdg + np.pi) % (2*np.pi) - np.pi)))
        elif mode == "HYBRID":
            # Adaptive corridor with confidence fallback
            c_filter = CorridorFilter(road, s_init=0.0, v_init=speeds[idx_start])
            cum_imu_turn = 0.0
            in_corridor = True
            
            for i in range(idx_start + 1, idx_end + 1):
                dt = df['dt'].iloc[i]
                c_filter.predict(dt)
                c_filter.update_speed(speeds[i], uncs[i])
                cum_imu_turn += yaw_rates[i] * dt
                
                s_anchor, conf = road_matcher.match_progress(c_filter.x[0], cum_imu_turn)
                if conf > 0.65:
                    c_filter.update_progress_anchor(s_anchor, conf)
                elif conf < 0.2:
                    # Low road confidence flag
                    in_corridor = False
                    
            final_pos, final_hdg = c_filter.get_position()
            hdg_err = float(np.degrees(np.abs((final_hdg - true_final_hdg + np.pi) % (2*np.pi) - np.pi)))
            
        endpoint_err = float(np.linalg.norm(final_pos - true_final_pos))
        drift_pct = (endpoint_err / REFERENCE_DISTANCE) * 100.0
        
        entry = {
            "Configuration": name,
            "Drift (%)": round(drift_pct, 3),
            "Endpoint (m)": round(endpoint_err, 3),
            "Distance Pred (m)": round(pred_dist, 3),
            "Distance Error (m)": round(dist_err, 3),
            "Speed Bias (m/s)": round(spd_bias, 3),
            "Heading Error (deg)": round(hdg_err, 2)
        }
        scorecard.append(entry)
        print(f"[{name}]")
        print(f"  Drift: {drift_pct:.3f}% | Endpoint: {endpoint_err:.3f} m | Dist Error: {dist_err:.2f} m | Speed Bias: {spd_bias:+.3f} m/s | Hdg Err: {hdg_err:.1f}°")
        return entry

    print("\n--- EVALUATING MANDATORY SIMPLE CONTROLS (PHASE 9) ---")
    evaluate_configuration("Control 1: Hold last GNSS speed", ctrl1_speeds, aligned_yaw_rate, np.full(len(df), 1.0), "2D_UKF")
    evaluate_configuration("Control 2: EWMA smoothed GNSS speed", ctrl2_speeds, aligned_yaw_rate, np.full(len(df), 1.0), "2D_UKF")
    
    print("\n--- EVALUATING LEARNED MOTION MODEL (V4) ---")
    evaluate_configuration("Control 3: MotionSpeedNet (raw)", pred_speed, aligned_yaw_rate, pred_unc, "2D_UKF")
    evaluate_configuration("Control 4: MotionSpeedNet + Affine Calib", calib_speed_affine, aligned_yaw_rate, pred_unc, "2D_UKF")
    evaluate_configuration("MotionSpeedNet + EWMA Calib", calib_speed_ewma, aligned_yaw_rate, pred_unc, "2D_UKF")
    evaluate_configuration("MotionSpeedNet + Calib + Yaw Correction", calib_speed_affine, corrected_yaw_rate, pred_unc, "2D_UKF")
    
    print("\n--- EVALUATING CORRIDOR & HYBRID MODES (PHASES 11-16) ---")
    evaluate_configuration("Mode C: Calibrated Speed + 1D Corridor", calib_speed_affine, aligned_yaw_rate, pred_unc, "CORRIDOR", use_anchors=False)
    evaluate_configuration("Mode D: Calibrated Speed + Corridor + Progress Anchor", calib_speed_affine, corrected_yaw_rate, pred_unc, "CORRIDOR", use_anchors=True)
    evaluate_configuration("Mode E: EWMA Calib Speed + Corridor + Progress Anchor", calib_speed_ewma, corrected_yaw_rate, pred_unc, "CORRIDOR", use_anchors=True)
    evaluate_configuration("Mode F: Best Adaptive Hybrid (Corridor + Anchors)", calib_speed_affine, corrected_yaw_rate, pred_unc, "HYBRID", use_anchors=True)

    print("\n================================================================================")
    print("STAGE 12 SCORECARD")
    print("================================================================================")
    df_sc = pd.DataFrame(scorecard)
    print(df_sc.to_string(index=False))
    
    with open('data/processed/stage12_scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    print("\nScorecard saved to data/processed/stage12_scorecard.json")

if __name__ == '__main__':
    run_stage12_evaluator()
