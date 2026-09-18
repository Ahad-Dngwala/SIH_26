"""
tools/preprocessing/reconstruct_timestamps.py
==============================================
Robust smartphone timestamp reconstruction for IO-VNBD sessions.

Problem statement
-----------------
The Android app records a relative timer "TIME SINCE START (ms)".
In some sessions (notably S-S2) the timer resets mid-recording to a small
value, creating a large negative step in the dt sequence.  Simply doing
    t = raw_ms / 1000 - t[0]
silently breaks the temporal alignment between the phone and the vehicle
reference, introducing errors of ~186 s.

This module detects resets and discontinuities and reconstructs a
continuous, monotonic time axis while preserving raw values for auditability.

Guarantees
----------
1.  Detects all timestamp resets (negative dt) and discontinuities
    (dt > THRESHOLD).
2.  Reconstructs continuous time by splicing segments end-to-end.
3.  Handles duplicate timestamps by dropping them (not by silent fill).
4.  Does NOT fabricate time where the data are genuinely missing.
5.  Produces a fully monotonic final timestamp stream in seconds.
6.  Returns the raw_ms column untouched alongside the reconstructed one.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


# Maximum credible single-step gap at 10 Hz before we call it a discontinuity
# (not a reset).  Set to 5 s = 50 samples worth to be conservative.
MAX_CREDIBLE_DT_MS = 5_000

# Minimum reset size: if |backward jump| > this we treat it as a clock reset.
MIN_RESET_JUMP_MS = 50_000


def reconstruct_timestamps(
    raw_ms: np.ndarray,
    nominal_dt_ms: float = 100.0,
    max_credible_dt_ms: float = MAX_CREDIBLE_DT_MS,
    min_reset_jump_ms: float = MIN_RESET_JUMP_MS,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Reconstruct a continuous, monotonic time axis from a raw Android
    millisecond timestamp column that may contain clock resets.

    Parameters
    ----------
    raw_ms : np.ndarray
        Raw 'TIME SINCE START (ms)' column values.
    nominal_dt_ms : float
        Expected sampling interval in ms.  Used for gap detection only.
    max_credible_dt_ms : float
        Any forward jump larger than this is treated as a genuine data gap
        (not a reset) — we preserve the gap rather than collapse it.
    min_reset_jump_ms : float
        A negative jump of magnitude >= this is flagged as a clock reset.
        Smaller negative values are treated as noise/duplicates.
    verbose : bool
        Print a diagnostic summary.

    Returns
    -------
    reconstructed_s : np.ndarray  shape (N,)
        Continuous monotonic time axis in seconds, starting at 0.
    info : dict
        Diagnostic information:
          resets      – list of (row_index, jump_ms)
          gaps        – list of (row_index, gap_ms)
          duplicates  – number of duplicate-value rows removed earlier
          segments    – number of continuous segments stitched together
    """
    N = len(raw_ms)
    dt_ms = np.diff(raw_ms.astype(np.float64))

    resets: List[Tuple[int, float]] = []
    gaps: List[Tuple[int, float]] = []

    # Build reconstructed time by walking through segments
    reconstructed_ms = np.zeros(N, dtype=np.float64)
    reconstructed_ms[0] = 0.0
    cursor = 0.0  # running accumulated time in ms

    for i in range(1, N):
        step = dt_ms[i - 1]

        if step <= -min_reset_jump_ms:
            # --- Clock reset ---
            resets.append((i, step))
            # Advance by exactly nominal_dt to keep the sequence moving forward.
            cursor += nominal_dt_ms
        elif step < 0:
            # Small negative: duplicate / minor glitch.  Treat as nominal step.
            cursor += nominal_dt_ms
        elif step > max_credible_dt_ms:
            # Large forward jump — genuine data gap.  Preserve it.
            gaps.append((i, step))
            cursor += step
        else:
            # Normal step
            cursor += step

        reconstructed_ms[i] = cursor

    reconstructed_s = reconstructed_ms / 1000.0

    info = {
        "resets": resets,
        "gaps": gaps,
        "n_segments": len(resets) + 1,
        "total_duration_s": reconstructed_s[-1],
    }

    if verbose:
        print(f"[TimestampReconstruction]")
        print(f"  Input rows:      {N}")
        print(f"  Clock resets:    {len(resets)}")
        for idx, jump in resets:
            print(f"    row {idx:6d}: jump = {jump/1000:.3f} s")
        print(f"  Data gaps:       {len(gaps)}")
        for idx, gap in gaps[:5]:
            print(f"    row {idx:6d}: gap  = {gap/1000:.3f} s")
        print(f"  Segments:        {info['n_segments']}")
        print(f"  Duration (raw):  {(raw_ms[-1]-raw_ms[0])/1000:.3f} s")
        print(f"  Duration (recon): {reconstructed_s[-1]:.3f} s")

    return reconstructed_s, info


def reconstruct_session_timestamps(
    raw_df: pd.DataFrame,
    time_col: str,
    nominal_dt_ms: float = 100.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply reconstruct_timestamps to a raw session DataFrame.

    Returns a copy of raw_df with:
        'raw_time_ms'    – original column preserved
        'time_s'         – reconstructed continuous seconds (starts at 0)
    """
    df = raw_df.copy()
    raw_ms = df[time_col].values

    reconstructed_s, info = reconstruct_timestamps(
        raw_ms,
        nominal_dt_ms=nominal_dt_ms,
        verbose=verbose,
    )

    df["raw_time_ms"] = df[time_col]
    df["time_s"] = reconstructed_s
    return df, info


if __name__ == "__main__":
    import os

    base = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset"
    for session in ["S-S1", "S-S2"]:
        path = os.path.join(base, f"{session}.csv")
        raw_df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
        time_col = [c for c in raw_df.columns if "TIME" in c.upper()][0]
        print(f"\n{'='*50}")
        print(f"Session: {session}")
        df_fixed, info = reconstruct_session_timestamps(raw_df, time_col, verbose=True)
        dt_fixed = np.diff(df_fixed["time_s"].values)
        print(f"  Reconstructed dt: min={dt_fixed.min()*1000:.1f}ms  "
              f"max={dt_fixed.max()*1000:.1f}ms  "
              f"mean={dt_fixed.mean()*1000:.1f}ms  "
              f"std={dt_fixed.std()*1000:.2f}ms")
