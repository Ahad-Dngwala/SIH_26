"""
Stage 6: Learned Orientation + Phone Alignment + Gyro-Bias Correction
=======================================================================
Approach:
1. Estimate phone-to-vehicle yaw-rate alignment from S-S1 (scale + bias)
2. Train a lightweight 1D-CNN to predict vehicle yaw rate from phone IMU
3. At runtime: use corrected yaw rate to drive the UKF heading propagation
4. Benchmark the same S-S2 60-second blackout at each step

Target: reduce from 46.88% baseline drift.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.baseline.data_loader import load_session
from tools.baseline.stationary_detector import detect_stationary
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState, fx
from scipy.stats import pearsonr

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# STEP 0: Freeze Baseline
# ============================================================
def compute_drift(df, blackout_start, blackout_end, gyro_override_fn=None):
    """
    Run the UKF blackout benchmark.
    gyro_override_fn(i, df) -> yaw_rate_rad_s (override if not None)
    Returns (drift_pct, endpoint_error_m, distance_m)
    """
    config = FusionConfig(enable_nhc=True, enable_zupt=True)
    lat0, lon0 = df['lat'].iloc[0], df['lon'].iloc[0]
    lat_to_m = 111000
    lon_to_m = 111000 * np.cos(np.radians(lat0))

    # Filter out bad dt rows (timestamp glitches cause covariance singularity)
    df = df[df['dt'] > 0].reset_index(drop=True)


    gnss_pos = np.zeros((len(df), 2))
    gnss_pos[:, 0] = (df['lat'].values - lat0) * lat_to_m
    gnss_pos[:, 1] = (df['lon'].values - lon0) * lon_to_m

    gnss_vel = np.zeros((len(df), 2))
    gnss_vel[:, 0] = df['gps_speed'].values

    is_stationary = detect_stationary(df)

    t = df['time'].values
    mask = (t >= blackout_start) & (t <= blackout_end)
    true_speed = df['true_speed'].values[mask]
    dt_bo = df['dt'].values[mask]
    distance_traveled = np.sum(true_speed * dt_bo)

    initial_state = UkfState(pos=gnss_pos[0], vel=gnss_vel[0], heading=0.0)
    ukf = DualChannelUkf(initial_state=initial_state, config=config)

    idx_end = np.searchsorted(t, blackout_end)
    pos = np.zeros((len(df), 2))

    for i in range(1, len(df)):
        dt_i = df['dt'].iloc[i]
        t_i = df['time'].iloc[i]
        in_blackout = blackout_start <= t_i <= blackout_end

        g_pos = None if in_blackout else gnss_pos[i]
        g_vel = None if in_blackout else gnss_vel[i]

        if gyro_override_fn is not None and in_blackout:
            gyro_yaw = gyro_override_fn(i, df)
        else:
            gyro_yaw = df['gz'].iloc[i - 1]

        state = ukf.step(
            dt=dt_i, gyro_yaw=gyro_yaw,
            channel_a_speed=0.0, channel_b_speed=0.0,
            gnss_pos=g_pos, gnss_vel=g_vel,
            road_signature_pos=None, road_signature_confidence=0.0,
            is_stationary=is_stationary[i]
        )
        pos[i] = state.pos

    err = np.linalg.norm(pos[idx_end] - gnss_pos[idx_end])
    drift = (err / max(distance_traveled, 1.0)) * 100
    return drift, err, distance_traveled


# ============================================================
# STEP 1: Static Alignment (scale + bias from S-S1 training)
# ============================================================
def estimate_alignment(df_train, min_speed=1.0):
    """
    Fit a linear model: v_yaw â‰ˆ scale * gy + bias
    Using S-S1 moving periods where gy has r=0.93 correlation.
    Returns (scale, bias) in (deg/s per rad/s, deg/s)
    """
    moving = df_train['true_speed'] > min_speed
    gy_deg = df_train['gy'].values[moving] * 180 / np.pi  # rad/s -> deg/s
    v_yaw = df_train['v_yaw_rate'].values[moving]         # deg/s

    scale = np.cov(gy_deg, v_yaw)[0, 1] / max(np.var(gy_deg), 1e-9)
    bias = np.mean(v_yaw) - scale * np.mean(gy_deg)
    r, _ = pearsonr(gy_deg, v_yaw)
    print(f"  [Alignment] gy: r={r:.4f}  scale={scale:.4f}  bias={bias:.4f} deg/s")
    return scale, bias


def make_aligned_gyro_fn(scale, bias):
    """Returns a gyro override function using gy axis with calibration."""
    def fn(i, df):
        gy_rad = df['gy'].iloc[i - 1]          # rad/s
        corrected_deg = scale * (gy_rad * 180 / np.pi) + bias
        return corrected_deg * np.pi / 180      # back to rad/s
    return fn


# ============================================================
# STEP 2: Static Bias Correction (pre-blackout stationary)
# ============================================================
def estimate_stationary_bias(df, blackout_start, window_s=5.0):
    """
    Estimate gyro bias from stationary period immediately before blackout.
    """
    t = df['time'].values
    mask = (t >= blackout_start - window_s) & (t < blackout_start)
    if mask.sum() < 5:
        return 0.0
    gz_window = df['gz'].values[mask]
    bias = gz_window.mean()
    return bias  # rad/s


# ============================================================
# STEP 3: Lightweight CNN Gyro Correction
# ============================================================
WINDOW = 20  # 2 seconds at 10 Hz
FEATURES = ['gx', 'gy', 'gz', 'ax_body', 'ay_body', 'az_body']

class GyroCorrectionNet(nn.Module):
    """
    Lightweight 1D-CNN: predicts vehicle yaw rate from phone IMU window.
    Input:  (batch, 6, 20) - 6 IMU features, 20 timesteps
    Output: (batch, 1)     - corrected yaw rate in deg/s
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.head(self.conv(x))


def build_windows(df, window=WINDOW):
    """Build sliding windows and targets (v_yaw_rate in deg/s)."""
    feats = df[FEATURES].values.astype(np.float32)
    targets = df['v_yaw_rate'].values.astype(np.float32)
    X, Y = [], []
    for i in range(window, len(df)):
        X.append(feats[i - window:i].T)  # (6, W)
        Y.append(targets[i])
    return np.array(X), np.array(Y)


def normalize_windows(X_train, X_val=None):
    """Standardize each feature channel using training set stats."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, 6, 1)
    std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train_n = (X_train - mean) / std
    if X_val is not None:
        X_val_n = (X_val - mean) / std
        return X_train_n, X_val_n, mean, std
    return X_train_n, mean, std


def train_gyro_net(df_train, epochs=30, lr=1e-3, batch_size=256):
    X, Y = build_windows(df_train)
    X_n, mean, std = normalize_windows(X)

    X_t = torch.tensor(X_n)
    Y_t = torch.tensor(Y).unsqueeze(1)

    ds = TensorDataset(X_t, Y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = GyroCorrectionNet()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss()

    model.train()
    for ep in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss={total_loss/len(ds):.4f}")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    model.eval()
    return model, mean, std


def make_nn_gyro_fn(model, mean, std, df_full, window=WINDOW):
    """
    Returns a gyro override function that uses the NN at inference time.
    For indices where we don't have a full window, fall back to gz.
    """
    feats = df_full[FEATURES].values.astype(np.float32)
    mean_t = torch.tensor(mean)
    std_t  = torch.tensor(std)

    def fn(i, df):
        if i < window:
            return df['gz'].iloc[i - 1]
        win = feats[i - window:i].T  # (6, W)
        x = torch.tensor(win).unsqueeze(0)  # (1, 6, W)
        x_n = (x - mean_t) / std_t
        with torch.no_grad():
            pred_deg = model(x_n).item()  # deg/s
        return pred_deg * np.pi / 180       # rad/s
    return fn


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("STAGE 6: Orientation + Alignment + Learned Gyro Correction")
    print("=" * 70)

    # Load sessions
    df_train = load_session('S-S1')
    df_test  = load_session('S-S2')

    t = df_test['time'].values
    blackout_start = t[len(t) // 2]
    blackout_end   = blackout_start + 60.0

    # Trim to window around blackout to speed up UKF loop (Â±300s)
    margin = 300.0
    df_test = df_test[(df_test['time'] >= blackout_start - margin) &
                      (df_test['time'] <= blackout_end + margin)].copy().reset_index(drop=True)

    print(f"\nBlackout window: {blackout_start:.1f}s â€“ {blackout_end:.1f}s")

    results = []

    # ---- Experiment A: Baseline ----
    drift_a, err_a, dist = compute_drift(df_test, blackout_start, blackout_end)
    results.append(('Stage 5A Baseline (gz)', drift_a, err_a, 'â€”'))
    print(f"\nExperiment A â€“ Baseline: {drift_a:.2f}%  err={err_a:.1f}m  dist={dist:.1f}m")

    # ---- Experiment B: Static Alignment (gy axis + scale/bias from S-S1) ----
    print("\n--- Experiment B: Static Alignment (gy axis) ---")
    scale, bias = estimate_alignment(df_train)
    fn_b = make_aligned_gyro_fn(scale, bias)
    drift_b, err_b, _ = compute_drift(df_test, blackout_start, blackout_end, fn_b)
    delta_b = drift_b - drift_a
    results.append(('Static Alignment (gy)', drift_b, err_b, f"{delta_b:+.2f}pp"))
    print(f"Experiment B â€“ Alignment only: {drift_b:.2f}%  err={err_b:.1f}m  Î”={delta_b:+.2f}pp")

    # ---- Experiment C: Static Bias Correction (pre-blackout gz bias) ----
    print("\n--- Experiment C: Pre-Blackout gz Bias Correction ---")
    gz_bias = estimate_stationary_bias(df_test, blackout_start)
    print(f"  Estimated pre-blackout gz bias: {gz_bias*180/np.pi:.4f} deg/s")
    def fn_c(i, df):
        return df['gz'].iloc[i - 1] - gz_bias
    drift_c, err_c, _ = compute_drift(df_test, blackout_start, blackout_end, fn_c)
    delta_c = drift_c - drift_a
    results.append(('gz bias correction', drift_c, err_c, f"{delta_c:+.2f}pp"))
    print(f"Experiment C â€“ Bias correction: {drift_c:.2f}%  err={err_c:.1f}m  Î”={delta_c:+.2f}pp")

    # ---- Experiment D: Alignment (gy) + bias correction ----
    print("\n--- Experiment D: Alignment (gy) + bias correction ---")
    gy_bias = estimate_stationary_bias(df_test, blackout_start)
    def fn_d(i, df):
        gy_rad = df['gy'].iloc[i - 1]
        corrected_deg = scale * (gy_rad * 180 / np.pi) + bias
        return (corrected_deg * np.pi / 180) - gy_bias
    drift_d, err_d, _ = compute_drift(df_test, blackout_start, blackout_end, fn_d)
    delta_d = drift_d - drift_a
    results.append(('Alignment(gy) + bias', drift_d, err_d, f"{delta_d:+.2f}pp"))
    print(f"Experiment D â€“ Alignment+Bias: {drift_d:.2f}%  err={err_d:.1f}m  Î”={delta_d:+.2f}pp")

    # ---- Experiment E: Learned Gyro Correction (CNN) ----
    print("\n--- Experiment E: Training Learned Gyro Correction CNN ---")
    print("  Training on S-S1...")
    model, mean_t, std_t = train_gyro_net(df_train, epochs=40)

    # Validate on S-S1 itself (yaw rate RMSE)
    X_tr, Y_tr = build_windows(df_train)
    X_tr_n, _, _ = normalize_windows(X_tr)
    with torch.no_grad():
        Y_pred = model(torch.tensor(X_tr_n)).squeeze().numpy()
    rmse_tr = np.sqrt(np.mean((Y_pred - Y_tr)**2))
    r_tr, _ = pearsonr(Y_pred, Y_tr)
    print(f"  Train yaw-rate RMSE: {rmse_tr:.3f} deg/s  r={r_tr:.4f}")

    # Validate on S-S2 v_yaw_rate (moving only)
    X_te, Y_te = build_windows(df_test)
    X_te_n = (X_te - mean_t) / std_t
    with torch.no_grad():
        Y_te_pred = model(torch.tensor(X_te_n)).squeeze().numpy()
    rmse_te = np.sqrt(np.mean((Y_te_pred - Y_te)**2))
    r_te, _ = pearsonr(Y_te_pred, Y_te)
    print(f"  S-S2 yaw-rate RMSE: {rmse_te:.3f} deg/s  r={r_te:.4f}")

    fn_e = make_nn_gyro_fn(model, mean_t, std_t, df_test)
    drift_e, err_e, _ = compute_drift(df_test, blackout_start, blackout_end, fn_e)
    delta_e = drift_e - drift_a
    results.append(('Learned CNN (S-S1â†’S-S2)', drift_e, err_e, f"{delta_e:+.2f}pp"))
    print(f"Experiment E â€“ Learned CNN: {drift_e:.2f}%  err={err_e:.1f}m  Î”={delta_e:+.2f}pp")

    # ---- Summary Table ----
    print("\n" + "=" * 70)
    print("STAGE 6 DRIFT TABLE")
    print("=" * 70)
    print(f"{'Configuration':<35} | {'Drift %':>8} | {'Err (m)':>9} | {'vs Baseline':>12}")
    print("-" * 70)
    for name, d, e, chg in results:
        print(f"{name:<35} | {d:>7.2f}% | {e:>9.1f} | {chg:>12}")

    best = min(results, key=lambda x: x[1])
    print(f"\nBest configuration: {best[0]}  â†’  {best[1]:.2f}% drift")

    return results

if __name__ == '__main__':
    main()
