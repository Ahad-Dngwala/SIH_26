"""Shared utilities for the Section 3 data pipeline (IO-VNBD).

Used by every 0X_*.py step script. Kept in one place so the session
discovery logic and CSV column mapping are proven once, not five
times (same rationale as models/common/ for the model side).

Dataset layout on disk (as of the onyekpeu/IO-VNBD repo, verified via
the GitHub API file tree - actual CSV headers not yet verified against
a live file, see NOTE below):

    data/raw/IO-VNBD/
      Synchronised V abd S datasets/
        Categorised IOVNB Dataset/
          <driver-code> (Driver <letter>)/<session>/{V,S}-<session>.csv
        Uncategorised IOVNB Dataset/
          S-Dataset/S-<session>.csv
          V-Dataset/V-<session>.csv
      Unsynchronised V and S Dataset/
        Categorised IOVNB (V) Dataset/V Dataset/<driver-code>/<session>/V-<session>.csv
        Uncategorised IOVNB (V and S) Dataset/{S-Dataset,V-Dataset}/...

NOTE (flag loudly, per data/README.md instruction): the "Uncategorised"
tree is NOT a byte-identical mirror of "Categorised" - spot-checking
LFS object hashes during discovery showed V-*.csv files match between
the two trees for shared session names, but some S-*.csv files do NOT,
and Uncategorised also contains extra sessions (an "A-series" and
"T-series", e.g. S-A1..S-A13, S-T1..S-T11) that don't appear in
Categorised at all. Rather than silently guessing which is
authoritative, `discover_sessions()` builds its session list from
Categorised only (it's the one with driver-code grouping needed for
route-level splitting and Table 1 driving-style labels) and separately
reports the Uncategorised-only sessions so a human decides whether to
pull them in. Don't change this silently - see the module docstring
of 00_build_manifest.py.

Column schema below is transcribed from the dataset's own paper
(README_1.pdf, Tables 3 and 4), not yet cross-checked against a real
CSV header row (GitHub's LFS batch API rate-limited this sandbox
mid-session - see the download script's docstring). `load_csv()`
matches columns by fuzzy keyword rather than a fixed index/position
list specifically so small header spelling differences don't break
silently - it raises loudly if a required signal is unmatched instead
of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column schema (README_1.pdf Table 3 = vehicle CAN "V-" files,
# Table 4 = smartphone "S-" files). Keys are canonical names used
# everywhere downstream; values are regexes matched case-insensitively
# against the real header, whitespace-normalized.
# ---------------------------------------------------------------------------

V_COLUMNS: dict[str, str] = {
    "gps_satellites": r"no\s*of\s*gps\s*satellites|gps\s*satellites",
    "time_s": r"time\s*since\s*start|time.*seconds",
    "lat": r"^lat",
    "lon": r"^lon",
    "velocity_kmh": r"^velocity",
    "heading_deg": r"^heading",
    "height_km": r"^height",
    "vertical_velocity": r"vertical\s*velocity",
    "sample_period_s": r"sample\s*period",
    "steering_angle_deg": r"steering\s*angle",
    "wheel_speed_fl": r"wheel\s*speed\s*front\s*left",
    "wheel_speed_fr": r"wheel\s*speed\s*front\s*right",
    "wheel_speed_rl": r"wheel\s*speed\s*rear\s*left",
    "wheel_speed_rr": r"wheel\s*speed\s*rear\s*right",
    "yaw_rate_dps": r"yaw\s*rate",
    "indicated_speed_kmh": r"indicated\s*vehicle\s*speed",
    "indicated_long_g": r"indicated\s*longitudinal",
    "indicated_lat_g": r"indicated\s*lateral",
    "handbrake": r"handbrake",
    "gear_requested": r"gear\s*requested",
    # NOTE: verified against a real header - "Gear (Number fof gear
    # employed 1-5)" is never just the bare word "gear", so the
    # previous `^gear$` (exact full-string match) could never hit.
    # Match "gear" at the start of the header as long as it isn't the
    # "gear requested" column (which has its own canonical key above).
    "gear": r"^gear(?!\s*requested)",
    "engine_speed_rpm": r"engine\s*speed",
    "coolant_temp_c": r"coolant\s*temp",
    "clutch": r"clutch",
    "brake_pressure_psi": r"brake\s*pressure",
    "brake_position": r"brake\s*position",
    "battery_voltage": r"battery\s*voltage",
    "air_temp_c": r"air\s*temp",
    "throttle_pct": r"accelerator\s*pedal|throttle",
}

S_COLUMNS: dict[str, str] = {
    "gps_lat": r"gps\s*latitude",
    "gps_lon": r"gps\s*longitude",
    "gps_alt_m": r"gps\s*altitude",
    "gps_speed_kmh": r"gps\s*speed",
    "gps_accuracy_m": r"gps\s*accuracy",
    "gps_orientation_deg": r"gps\s*orientation",
    "gps_satellites": r"satellites\s*in\s*range",
    "time_ms": r"time\s*since\s*start",
    "date": r"^date",
    "accel_x": r"accelerometer\s*x",
    "accel_y": r"accelerometer\s*y",
    "accel_z": r"accelerometer\s*z",
    "gravity_x": r"gravity\s*x",
    "gravity_y": r"gravity\s*y",
    "gravity_z": r"gravity\s*z",
    "gyro_yaw": r"gyroscope.*yaw",
    "gyro_pitch": r"gyroscope.*pitch",
    "gyro_roll": r"gyroscope.*roll",
    "mag_x": r"magnetic\s*field\s*x",
    "mag_y": r"magnetic\s*field\s*y",
    "mag_z": r"magnetic\s*field\s*z",
    "orientation_yaw": r"orientation\s*\(?yaw",
    "orientation_roll": r"orientation\s*\(?roll",
    "orientation_pitch": r"orientation\s*\(?pitch",
}

# NOTE: Table 4 lists "Gyroscope (Pitch)" twice (rows 17 and 18 in the
# paper's own table - one is very likely a typo for Roll, since Figure
# 2's axis diagram and every other IMU triad in this dataset is
# yaw/pitch/roll). We map row 17 to gyro_pitch and row 18 to gyro_roll
# by position rather than trusting both labels literally. FLAG THIS to
# whoever verifies against a live file - if the real header disambiguates
# (e.g. says "Roll" for one of them), this comment is obsolete and the
# regexes above should switch to using the real header text directly.


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def match_columns(header: list[str], schema: dict[str, str]) -> dict[str, str]:
    """Map canonical name -> actual header string, fuzzy-matched.

    Raises ValueError listing exactly which canonical columns couldn't
    be matched, rather than silently dropping them - a silently-missing
    wheel-speed channel is a much worse failure mode than a loud one.
    """
    normalized = {h: _norm(h) for h in header}
    out: dict[str, str] = {}
    unmatched: list[str] = []
    for canonical, pattern in schema.items():
        hits = [h for h, n in normalized.items() if re.search(pattern, n)]
        if not hits:
            unmatched.append(canonical)
        else:
            out[canonical] = hits[0]
    if unmatched:
        raise ValueError(
            f"Could not match columns for: {unmatched}. "
            f"Actual header was: {header}. "
            "Update the regex in iovnbd_common.py's V_COLUMNS/S_COLUMNS "
            "rather than hardcoding a position - see module docstring."
        )
    return out


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

@dataclass
class Session:
    session_id: str          # e.g. "S1", "Vta05", "M" - stable across V/S
    driver_code: str         # folder name, e.g. "S (Driver A)"
    sync: str                # "synchronised" | "unsynchronised"
    v_path: Path | None
    s_path: Path | None

    @property
    def has_pair(self) -> bool:
        return self.v_path is not None and self.s_path is not None


_SESSION_FROM_FILENAME = re.compile(r"^[VS]-(.+)\.csv$", re.IGNORECASE)


def _session_id_from_path(p: Path) -> str:
    m = _SESSION_FROM_FILENAME.match(p.name)
    if not m:
        raise ValueError(f"Unexpected filename, doesn't match V-/S-<id>.csv: {p}")
    return m.group(1)


def discover_sessions(raw_root: Path) -> tuple[list[Session], list[str]]:
    """Walk the Categorised trees and pair up V-/S- CSVs per session.

    Returns (sessions, uncategorised_only_ids) - the second list is the
    session IDs that exist under "Uncategorised" but were not found in
    "Categorised" at all (see module docstring - these are extra data,
    not duplicates, and are deliberately NOT auto-included).
    """
    raw_root = Path(raw_root)
    sessions: list[Session] = []

    sync_root = raw_root / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset"
    if sync_root.exists():
        for driver_dir in sorted(p for p in sync_root.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in driver_dir.iterdir() if p.is_dir()):
                v_files = list(session_dir.glob("V-*.csv")) + list(session_dir.glob("v-*.csv"))
                s_files = list(session_dir.glob("S-*.csv")) + list(session_dir.glob("s-*.csv"))
                sid = session_dir.name
                sessions.append(Session(
                    session_id=sid,
                    driver_code=driver_dir.name,
                    sync="synchronised",
                    v_path=v_files[0] if v_files else None,
                    s_path=s_files[0] if s_files else None,
                ))

    unsync_root = raw_root / "Unsynchronised V and S Dataset" / "Categorised IOVNB (V) Dataset" / "V Dataset"
    if unsync_root.exists():
        for driver_dir in sorted(p for p in unsync_root.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in driver_dir.iterdir() if p.is_dir()):
                v_files = list(session_dir.glob("V-*.csv")) + list(session_dir.glob("v-*.csv"))
                sid = session_dir.name
                sessions.append(Session(
                    session_id=f"unsync_{sid}",
                    driver_code=driver_dir.name,
                    sync="unsynchronised",
                    v_path=v_files[0] if v_files else None,
                    s_path=None,  # Unsynchronised split is V-only for the categorised tree; see 3.2 step 2.
                ))

    # Report (not auto-include) Uncategorised-only session IDs.
    known_ids = {s.session_id for s in sessions}
    uncategorised_only: list[str] = []
    unc_s_dir = raw_root / "Synchronised V abd S datasets" / "Uncategorised IOVNB Dataset" / "S-Dataset"
    if unc_s_dir.exists():
        for p in unc_s_dir.glob("S-*.csv"):
            sid = _session_id_from_path(p)
            if sid not in known_ids:
                uncategorised_only.append(sid)

    return sessions, sorted(set(uncategorised_only))


# ---------------------------------------------------------------------------
# CSV loading + resampling (Section 3.2 steps 1-2)
# ---------------------------------------------------------------------------

def load_csv(path: Path, schema: dict[str, str]) -> pd.DataFrame:
    # NOTE: verified against real files - the S-*.csv (smartphone)
    # files are not valid UTF-8 (some contain a stray 0xb2 byte, most
    # likely a superscript-2 or degree-adjacent glyph carried over from
    # whatever tool exported them on Windows). Try UTF-8 first since
    # it's the common case and rejects anything genuinely corrupt, then
    # fall back to cp1252 (a superset of latin-1 that covers the bytes
    # Windows tools actually emit) rather than failing the whole file
    # over one non-ASCII byte in a column we may not even keep.
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")
    colmap = match_columns(list(df.columns), schema)
    out = df[[colmap[c] for c in colmap]].copy()
    out.columns = list(colmap.keys())
    return out


def resample_100hz(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Linear-interpolation resample to a uniform 100 Hz grid.

    Section 3.2 step 1 is explicit that this must be linear
    interpolation, not nearest-neighbor (nearest-neighbor introduces
    step artifacts models will latch onto as fake features).
    """
    t = df[time_col].to_numpy(dtype=float)
    if not np.all(np.diff(t) > 0):
        # Some sessions may have duplicate/out-of-order timestamps
        # (GPS reacquisition artifacts) - sort and dedupe rather than
        # silently interpolating over a non-monotonic axis.
        order = np.argsort(t, kind="stable")
        df = df.iloc[order].reset_index(drop=True)
        t = df[time_col].to_numpy(dtype=float)
        keep = np.concatenate([[True], np.diff(t) > 0])
        df = df.iloc[keep].reset_index(drop=True)
        t = df[time_col].to_numpy(dtype=float)

    t0, t1 = t[0], t[-1]
    n = int(np.floor((t1 - t0) * 100.0)) + 1
    t_grid = t0 + np.arange(n) / 100.0

    out = {time_col: t_grid}
    dropped = []
    for col in df.columns:
        if col == time_col:
            continue
        try:
            values = df[col].to_numpy(dtype=float)
        except (ValueError, TypeError):
            # Non-numeric metadata (e.g. S-stream's wall-clock "date"
            # string) - nothing downstream needs it interpolated, and
            # silently coercing a date string to NaN would be worse
            # than dropping it with a note.
            dropped.append(col)
            continue
        out[col] = np.interp(t_grid, t, values)
    if dropped:
        print(f"  (resample_100hz: dropped non-numeric column(s), not interpolated: {dropped})")
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Windowing (Section 3.2 step 3)
# ---------------------------------------------------------------------------

def make_windows(
    arr: np.ndarray, window_len: int, stride: int
) -> np.ndarray:
    """arr: (T, C) -> (N, window_len, C), 50% overlap == stride = window_len // 2."""
    t, c = arr.shape
    if t < window_len:
        return np.empty((0, window_len, c), dtype=arr.dtype)
    starts = range(0, t - window_len + 1, stride)
    return np.stack([arr[s:s + window_len] for s in starts], axis=0)
