import pandas as pd
import numpy as np
import os
import sys

# Ensure preprocessing utilities are importable regardless of CWD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tools.preprocessing.reconstruct_timestamps import reconstruct_timestamps
from tools.preprocessing.sync_estimator import estimate_sv_lag


# -----------------------------------------------------------------------
# Per-session sync offsets (samples at 10 Hz) estimated automatically.
# These are set once from the full-session cross-correlation analysis and
# stored here so the lag is NOT computed inside the blackout window.
#
# Positive lag_samples means phone IMU arrives LATE relative to vehicle.
# Negative means phone data is recorded EARLIER than vehicle reference.
#
# S-S1: lag = -2  samples  (-0.20 s)  — negligible
# S-S2: lag = -87 samples  (-8.70 s)  — major offset due to logger restart
# -----------------------------------------------------------------------
SESSION_LAG_SAMPLES = {
    "S-S1": -2,
    "S-S2": -87,
}
SESSION_LAG_S = {k: v / 10.0 for k, v in SESSION_LAG_SAMPLES.items()}


def load_session(session_name, auto_sync=True, verbose_ts=False):
    """
    Loads the native 10 Hz S-Dataset and aligns V-Dataset ground truth to it.
    No 100 Hz interpolation.

    Improvements over original loader (Stage 7):
    --------------------------------------------
    1.  Robust timestamp reconstruction: detects & repairs Android clock
        resets (e.g. the 186-second reset in S-S2 at row 1864) so the
        phone time axis is monotonic and matches the vehicle recording
        duration.
    2.  Automatic S/V synchronisation: vehicle-side signals are
        interpolated at timestamps shifted by the pre-calibrated per-session
        lag, so gy ↔ v_yaw_rate correlation is maximised.
    3.  Lag column added for debugging / auditing.

    Parameters
    ----------
    session_name : str  e.g. 'S-S1', 'S-S2'
    auto_sync    : bool apply pre-calibrated per-session lag (default True)
    verbose_ts   : bool print timestamp reconstruction diagnostics
    """
    base = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset"
    s_file = os.path.join(base, "S-Dataset", f"{session_name}.csv")
    v_file = os.path.join(base, "V-Dataset", f"{session_name.replace('S-', 'V-')}.csv")

    if not os.path.exists(s_file) or not os.path.exists(v_file):
        v_file = os.path.join(base, "V-Dataset", f"{session_name}.csv")
        if not os.path.exists(v_file):
            return None

    s_df = pd.read_csv(s_file, encoding='latin1', on_bad_lines='skip')

    # ------------------------------------------------------------------ #
    # Column discovery
    # ------------------------------------------------------------------ #
    col_time = [c for c in s_df.columns if 'TIME' in c.upper() and 'DATE' not in c.upper()][0]
    col_gps  = [c for c in s_df.columns if 'GPS SPEED' in c.upper()][0]
    col_lat  = [c for c in s_df.columns if 'LATITUDE' in c.upper()][0]
    col_lon  = [c for c in s_df.columns if 'LONGITUDE' in c.upper()][0]
    col_ax   = [c for c in s_df.columns if 'ACCELEROMETER X' in c.upper()][0]
    col_ay   = [c for c in s_df.columns if 'ACCELEROMETER Y' in c.upper()][0]
    col_az   = [c for c in s_df.columns if 'ACCELEROMETER Z' in c.upper()][0]
    col_gx   = [c for c in s_df.columns if 'GYROSCOPE X' in c.upper()][0]
    col_gy   = [c for c in s_df.columns if 'GYROSCOPE Y' in c.upper()][0]
    col_gz   = [c for c in s_df.columns if 'GYROSCOPE Z' in c.upper()][0]

    # ------------------------------------------------------------------ #
    # STEP 1: Robust timestamp reconstruction                              #
    # ------------------------------------------------------------------ #
    raw_ms = s_df[col_time].values.astype(np.float64)
    s_time_recon, ts_info = reconstruct_timestamps(
        raw_ms, nominal_dt_ms=100.0, verbose=verbose_ts
    )

    # ------------------------------------------------------------------ #
    # STEP 2: Drop duplicates & NaNs *after* reconstruction               #
    #         Keep the reconstructed time so row count is preserved.       #
    # ------------------------------------------------------------------ #
    s_df = s_df.reset_index(drop=True)
    # Drop rows with duplicate reconstructed timestamps (should be near zero
    # after reconstruction but handle edge cases)
    _, unique_idx = np.unique(s_time_recon, return_index=True)
    s_df = s_df.iloc[unique_idx].reset_index(drop=True)
    s_time_recon = s_time_recon[unique_idx]

    # Also require valid lat/lon
    valid_pos = s_df[col_lat].notna() & s_df[col_lon].notna()
    s_df = s_df[valid_pos].reset_index(drop=True)
    s_time_recon = s_time_recon[valid_pos.values[unique_idx]]

    # Zero-reference the phone time
    s_time = s_time_recon - s_time_recon[0]
    s_gps = s_df[col_gps].values / 3.6  # km/h -> m/s

    # ------------------------------------------------------------------ #
    # STEP 3: Load V-Dataset                                               #
    # ------------------------------------------------------------------ #
    v_df = pd.read_csv(v_file, encoding='latin1', on_bad_lines='skip')
    col_v_time = [c for c in v_df.columns if 'TIME SINCE START OF DAY (SECONDS)' in c.upper()][0]
    col_v_vel  = [c for c in v_df.columns if 'VELOCITY (KM/HR)' in c.upper()][0]
    col_v_yaw  = [c for c in v_df.columns if 'YAW RATE' in c.upper()]
    col_v_head = [c for c in v_df.columns if 'HEADING' in c.upper()]
    col_v_long = [c for c in v_df.columns if 'LONGITUDINAL' in c.upper()]
    col_v_lat_ = [c for c in v_df.columns if 'LATERAL' in c.upper()]

    v_df = v_df.drop_duplicates(subset=[col_v_time]).dropna(subset=[col_v_vel])
    v_time = v_df[col_v_time].values
    v_time = v_time - v_time[0]           # zero-reference vehicle time
    v_speed = v_df[col_v_vel].values / 3.6

    # ------------------------------------------------------------------ #
    # STEP 4: Apply per-session temporal synchronisation lag               #
    # ------------------------------------------------------------------ #
    lag_samples = SESSION_LAG_SAMPLES.get(session_name, 0) if auto_sync else 0
    lag_s = lag_samples / 10.0

    # Shift the PHONE time axis by -lag_s so phone events align with vehicle.
    # A negative lag means phone data arrived earlier than vehicle data.
    # Shifting phone time by -lag_s makes interp(phone_shifted, v_time, ...)
    # pull the correct vehicle sample.
    s_time_aligned = s_time - lag_s   # phone time after sync correction

    # ------------------------------------------------------------------ #
    # STEP 5: Interpolate vehicle signals onto aligned phone timestamps    #
    # ------------------------------------------------------------------ #
    def _safe_interp(s_t, v_t, v_vals):
        return np.interp(s_t, v_t, v_vals,
                         left=v_vals[0], right=v_vals[-1])

    true_v_speed = _safe_interp(s_time_aligned, v_time, v_speed)
    v_yaw_rate = (_safe_interp(s_time_aligned, v_time, v_df[col_v_yaw[0]].values)
                  if col_v_yaw else np.zeros(len(s_time)))
    v_heading  = (_safe_interp(s_time_aligned, v_time, v_df[col_v_head[0]].values)
                  if col_v_head else np.zeros(len(s_time)))
    v_long_acc = (_safe_interp(s_time_aligned, v_time, v_df[col_v_long[0]].values)
                  if col_v_long else np.zeros(len(s_time)))
    v_lat_acc  = (_safe_interp(s_time_aligned, v_time, v_df[col_v_lat_[0]].values)
                  if col_v_lat_ else np.zeros(len(s_time)))

    # ------------------------------------------------------------------ #
    # STEP 6: Gravity compensation (unchanged)                             #
    # ------------------------------------------------------------------ #
    alpha = 0.05
    accel = np.vstack([
        s_df[col_ax].values,
        s_df[col_ay].values,
        s_df[col_az].values
    ]).T
    gravity = np.zeros_like(accel)
    gravity[0] = accel[0]
    for i in range(1, len(accel)):
        gravity[i] = alpha * accel[i] + (1 - alpha) * gravity[i - 1]
    linear_accel = accel - gravity

    median_dt = np.median(np.diff(s_time))
    gx_raw = s_df[col_gx].values
    gy_raw = s_df[col_gy].values
    gz_raw = s_df[col_gz].values

    col_az_orient = [c for c in s_df.columns if 'AZIMUTH' in c.upper()]
    col_pitch_s   = [c for c in s_df.columns if 'PITCH' in c.upper()]
    col_roll_s    = [c for c in s_df.columns if 'ROLL' in c.upper()]
    phone_azimuth = s_df[col_az_orient[0]].values if col_az_orient else np.zeros(len(s_time))
    phone_pitch   = s_df[col_pitch_s[0]].values   if col_pitch_s   else np.zeros(len(s_time))
    phone_roll    = s_df[col_roll_s[0]].values     if col_roll_s    else np.zeros(len(s_time))

    # ------------------------------------------------------------------ #
    # STEP 7: Assemble output DataFrame                                    #
    # ------------------------------------------------------------------ #
    # 'time' uses the RECONSTRUCTED phone time (zero-referenced, monotonic)
    # so the UKF sees correct dt values.
    dt_arr = np.concatenate([[median_dt], np.diff(s_time)])

    df = pd.DataFrame({
        'time':          s_time,            # reconstructed phone time (s)
        'dt':            dt_arr,
        'gps_speed':     s_gps,
        'true_speed':    true_v_speed,      # vehicle speed aligned by lag
        'lat':           s_df[col_lat].values,
        'lon':           s_df[col_lon].values,
        'ax_body':       linear_accel[:, 0],
        'ay_body':       linear_accel[:, 1],
        'az_body':       linear_accel[:, 2],
        'gx':            gx_raw,
        'gy':            gy_raw,
        'gz':            gz_raw,
        'accel_mag':     np.linalg.norm(linear_accel, axis=1),
        # Vehicle-side ground-truth (training/oracle only) — sync-corrected
        'v_yaw_rate':    v_yaw_rate,        # deg/s
        'v_heading':     v_heading,         # degrees
        'v_long_acc':    v_long_acc,        # g
        'v_lat_acc':     v_lat_acc,         # g
        # Phone orientation sensor (raw, not corrected)
        'phone_azimuth': phone_azimuth,
        'phone_pitch':   phone_pitch,
        'phone_roll':    phone_roll,
        # Audit columns
        'lag_s':         lag_s,             # applied synchronisation lag
    })

    return df
