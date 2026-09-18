"""
tools/preprocessing/sync_estimator.py
=======================================
Automatic temporal synchronisation estimation between smartphone IMU and
vehicle reference data for IO-VNBD sessions.

Background
----------
After timestamp reconstruction the smartphone stream may still have a
session-specific clock offset relative to the vehicle CAN / RT3000 reference.

Stage 6B established an empirical offset of ~-8.7 s for S-S2 using manual
turn-peak inspection.  This module automates that process so the offset can
be recalculated reliably for any session.

Algorithm
---------
1.  Build a scalar 'motion magnitude' signal from the phone IMU
    (primarily the gy axis which correlates best with yaw rate).
2.  Build a matching scalar signal from the vehicle reference
    (yaw rate from the V-Dataset).
3.  Compute normalised cross-correlation over a search window
    (-MAX_LAG_S .. +MAX_LAG_S).
4.  Find the peak correlation and return the corresponding lag.

The function also computes:
    - correlation at lag 0     (before alignment)
    - correlation at best lag  (after alignment)
    - a simple confidence score (peak_corr / mean_corr over search range)

Notes
-----
* Only moving periods (vehicle speed > MIN_SPEED_KMH) are used to avoid
  fitting noise from stationary periods.
* The search is performed on the full session (not just the blackout).
* DO NOT use blackout ground truth to calibrate the lag.
"""

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
from typing import Tuple, Dict


MAX_LAG_S = 15.0          # search range ± seconds
MIN_SPEED_KMH = 5.0       # only use moving periods


def estimate_sv_lag(
    gy_phone: np.ndarray,
    v_yaw_rate: np.ndarray,
    vehicle_speed_kmh: np.ndarray,
    sample_rate_hz: float = 10.0,
    max_lag_s: float = MAX_LAG_S,
    min_speed_kmh: float = MIN_SPEED_KMH,
    verbose: bool = True,
) -> Dict:
    """
    Estimate the temporal offset between the phone gyroscope (gy) and the
    vehicle yaw rate.

    Returns a dict with:
        lag_samples   – integer offset (phone shifted by this many samples)
        lag_s         – lag in seconds
        corr_at_zero  – Pearson r at zero lag (moving periods)
        corr_at_best  – Pearson r at best lag (moving periods)
        confidence    – peak / mean of |corr| over search range
        search_range_s– (min_lag, max_lag) searched
    """
    max_lag_samples = int(max_lag_s * sample_rate_hz)
    N = min(len(gy_phone), len(v_yaw_rate), len(vehicle_speed_kmh))
    gy = gy_phone[:N]
    yaw = v_yaw_rate[:N]
    spd = vehicle_speed_kmh[:N]

    # Standardise both signals
    def _zscore(x):
        s = x.std()
        return (x - x.mean()) / (s if s > 1e-9 else 1.0)

    # Moving mask
    moving = spd > min_speed_kmh

    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    corrs = np.zeros(len(lags))

    for k, lag in enumerate(lags):
        if lag < 0:
            g_sub = gy[:lag]
            y_sub = yaw[-lag:]
            m_sub = moving[-lag:]
        elif lag > 0:
            g_sub = gy[lag:]
            y_sub = yaw[:-lag]
            m_sub = moving[:-lag]
        else:
            g_sub = gy
            y_sub = yaw
            m_sub = moving

        valid = m_sub & ~np.isnan(g_sub) & ~np.isnan(y_sub)
        if valid.sum() > 100:
            corrs[k] = np.corrcoef(g_sub[valid], y_sub[valid])[0, 1]
        else:
            corrs[k] = 0.0

    # Find zero-lag index
    zero_idx = np.searchsorted(lags, 0)
    corr_at_zero = corrs[zero_idx]

    # Best lag
    best_idx = np.argmax(np.abs(corrs))
    best_lag = lags[best_idx]
    corr_at_best = corrs[best_idx]

    # Confidence
    mean_abs_corr = np.abs(corrs).mean()
    confidence = abs(corr_at_best) / (mean_abs_corr + 1e-9)

    result = {
        "lag_samples": int(best_lag),
        "lag_s": best_lag / sample_rate_hz,
        "corr_at_zero": corr_at_zero,
        "corr_at_best": corr_at_best,
        "confidence": confidence,
        "search_range_s": (-max_lag_s, max_lag_s),
        "n_moving_samples": int(moving.sum()),
    }

    if verbose:
        print(f"[SyncEstimator]")
        print(f"  Best lag:         {best_lag:+d} samples  ({best_lag/sample_rate_hz:+.2f} s)")
        print(f"  Corr at lag=0:    {corr_at_zero:+.4f}")
        print(f"  Corr at best lag: {corr_at_best:+.4f}")
        print(f"  Confidence:       {confidence:.2f}")

    return result


def estimate_session_lag(
    s_df: pd.DataFrame,
    v_df: pd.DataFrame,
    sample_rate_hz: float = 10.0,
    max_lag_s: float = MAX_LAG_S,
    verbose: bool = True,
) -> Dict:
    """
    High-level wrapper: estimate lag given loaded S and V DataFrames
    (as produced by the raw CSV reader or load_session).

    The DataFrames must have been row-aligned and have the same length.
    Uses column names matching IO-VNBD conventions.
    """
    col_gy = [c for c in s_df.columns if 'GYROSCOPE Y' in c.upper() or c == 'gy']
    col_yaw = [c for c in v_df.columns if 'YAW RATE' in c.upper() or c == 'v_yaw_rate']
    col_spd = [c for c in v_df.columns if 'VELOCITY' in c.upper() or c == 'true_speed']

    if not col_gy or not col_yaw or not col_spd:
        raise ValueError(f"Cannot find required columns. gy cols: {col_gy}, yaw: {col_yaw}, spd: {col_spd}")

    gy = s_df[col_gy[0]].values.astype(float)
    yaw = v_df[col_yaw[0]].values.astype(float)
    spd = v_df[col_spd[0]].values.astype(float)

    return estimate_sv_lag(
        gy_phone=np.degrees(gy) if gy.max() < 50 else gy,  # convert rad/s if needed
        v_yaw_rate=yaw,
        vehicle_speed_kmh=spd,
        sample_rate_hz=sample_rate_hz,
        max_lag_s=max_lag_s,
        verbose=verbose,
    )


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    BASE_S = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset"
    BASE_V = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset"

    for session in ["S-S1", "S-S2"]:
        v_session = session.replace("S-", "V-")
        s_df = pd.read_csv(os.path.join(BASE_S, f"{session}.csv"), encoding="latin1", on_bad_lines="skip")
        v_df = pd.read_csv(os.path.join(BASE_V, f"{v_session}.csv"), encoding="latin1", on_bad_lines="skip")
        print(f"\n{'='*50}\nSession: {session}")
        result = estimate_session_lag(s_df, v_df, verbose=True)
        print(f"  => Phone leads vehicle by {result['lag_s']:+.2f} s")
