"""Window per-model and derive labels (Section 3.2 step 3 + Section 3.3).

IMPORTANT correction vs. a literal reading of Section 4's "input: accel
xyz, gyro xyz [, mag xyz]" for alignment_net/channel_a/channel_b: the
IO-VNBD *vehicle* (V-) stream does NOT contain raw accelerometer/
gyroscope/magnetometer data - it's CAN-bus data (Table 3), whose only
motion-sensor-like fields are 2-axis "Indicated Longitudinal/Lateral g"
and no gyro at all. Only the *smartphone* (S-) stream has a real 9-axis
IMU (Table 4). Since the deployed system only ever has the phone's IMU
at inference time (Section 0: "phone-based dead reckoning" / Section
7.1: raw SensorManager), the S- stream is unambiguously the correct
input source for every model in Section 4, and V- is ground truth
(labels) only. Get this backwards and Channel A trains on 6 channels
that don't exist on-device. Flagging this loudly since Section 4 itself
doesn't name the source stream.

Outputs, per model, per split, into data/processed/<model>/<split>/:
  windows.npy  (N, window_len, n_channels) float32, RAW (not normalized -
               normalization happens at Dataset.__getitem__ time per
               models/*/dataset.py's own TODO comments, using
               norm_stats.json from 04_normalize.py).
  labels.npy   (N, label_dim) float32
  meta.json    session_id per window (index-aligned with the two arrays
               above) - for traceability back to a source route, e.g.
               when the benchmark replay tool needs to know which
               session a bad prediction came from.

Only alignment_net and channel_a_velocity are implemented here, per
Section 12's own priority order for Layer 1 Person A ("these are the
two models everything else is easiest to sanity-check against").
channel_b_velocity / road_signature / calibration_adapter are stubbed
with their window configs in WINDOW_CONFIGS but no label-derivation
function yet - see the TODOs below and Section 12's stated priority
order (Channel B second, road-signature third, calibration adapter
last) before implementing them.

QUALITY GATE (added after 02_align.py's xcorr-fallback run on real
IO-VNBD data showed 8/71 synchronised sessions are not reliably
aligned - 6 below --min-corr even after a +/-10s lag search, 2 with an
undefined/NaN correlation, see that script's docstring): this script
used to window every `_aligned.parquet` file it found with no regard
for whether the alignment was actually any good, which meant a session
like Vta03 (corr=-0.02, almost certainly a mis-paired V/S file) would
have gone straight into training data with zero indication anything
was wrong. It now reads `alignment_report.json` from --aligned-dir (if
present) and skips any session whose recorded corr is null, below
--min-corr, or landed on the lag-search boundary (per 02_align.py's own
docstring, a boundary lag means the search never converged to an
interior optimum, so the recorded offset - and the label timing that
depends on it - can't be trusted even though corr cleared --min-corr;
e.g. Vta20 at corr=0.58, lag=+10.00s, exactly on the +/-10.0s edge),
printing exactly which sessions were skipped and why. Pass
--include-flagged to window everything anyway (e.g. to inspect what a
bad session's windows actually look like) - it prints a loud warning
banner when used so it's never silently on. If alignment_report.json
is missing, this script warns loudly and falls back to windowing every
aligned file with no gate, so an out-of-date data/raw/_aligned/ from
before this quality gate existed doesn't fail quietly either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iovnbd_common import make_windows

WINDOW_CONFIGS = {
    "alignment_net": {"window_len": 200, "stride": 100},       # 2s / 50% overlap, Section 4.1
    "channel_a_velocity": {"window_len": 200, "stride": 100},   # 2s / 50% overlap, Section 4.2
    "channel_b_velocity": {"window_len": 400, "stride": 200},   # 4s / 50% overlap, Section 4.3 - labeling TODO
    "road_signature": {"window_len": 200, "stride": 100},        # Section 4.4 - labeling TODO (needs a per-corridor OSM segment map, Section 6, not just this dataset)
    "calibration_adapter": {"window_len": 200, "stride": 100},   # Section 4.5 - TODO, needs a real vehicle calibration-drive session, not IO-VNBD
}


def _wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def build_alignment_net(df: pd.DataFrame, cfg: dict, min_speed_kmh: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Section 4.1 / 3.3: pitch/roll from the gravity vector, yaw offset
    from device heading vs. GPS course-over-ground - only where the GNSS
    course is well-defined, i.e. the vehicle is actually moving. Windows
    below `min_speed_kmh` mean speed produce no yaw label and are
    dropped (see module docstring for why we can't just leave yaw at 0).
    """
    g_cols = ["s_gravity_x", "s_gravity_y", "s_gravity_z"]
    imu_cols = ["s_accel_x", "s_accel_y", "s_accel_z",
                "s_gyro_yaw", "s_gyro_pitch", "s_gyro_roll",
                "s_mag_x", "s_mag_y", "s_mag_z"]
    missing = [c for c in imu_cols + g_cols + ["v_heading_deg", "v_velocity_kmh"] if c not in df.columns]
    if missing:
        raise ValueError(f"alignment_net needs columns not present in aligned frame: {missing}")

    x_windows = make_windows(df[imu_cols].to_numpy(dtype=np.float32), **cfg)
    g_windows = make_windows(df[g_cols].to_numpy(dtype=np.float32), **cfg)
    heading_windows = make_windows(df[["v_heading_deg"]].to_numpy(dtype=np.float32), **cfg)
    speed_windows = make_windows(df[["v_velocity_kmh"]].to_numpy(dtype=np.float32), **cfg)

    g_mean = g_windows.mean(axis=1)  # (N, 3)
    gx, gy, gz = g_mean[:, 0], g_mean[:, 1], g_mean[:, 2]
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-8
    pitch = np.arcsin(np.clip(-gx / g_norm, -1, 1))
    roll = np.arcsin(np.clip(gy / g_norm, -1, 1))

    mean_speed = speed_windows.mean(axis=(1, 2))
    valid = mean_speed >= min_speed_kmh

    orientation_yaw = None  # placeholder if S orientation_yaw exists; else fall back to a zero prior
    if "s_orientation_yaw" in df.columns:
        yaw_windows = make_windows(df[["s_orientation_yaw"]].to_numpy(dtype=np.float32), **cfg)
        orientation_yaw = yaw_windows[:, -1, 0]  # value at window's end, matches Section 4.1's "yaw offset" being a single number per window
    course_over_ground = heading_windows[:, -1, 0]

    if orientation_yaw is not None:
        yaw_offset_deg = _wrap_deg(orientation_yaw - course_over_ground)
    else:
        yaw_offset_deg = np.zeros_like(course_over_ground)

    yaw_rad = np.deg2rad(yaw_offset_deg)
    labels = np.stack([pitch, roll, np.sin(yaw_rad), np.cos(yaw_rad)], axis=1).astype(np.float32)

    return x_windows[valid], labels[valid]


def build_channel_a(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Section 4.2 / 3.3: 6ch phone IMU (accel+gyro) in, forward velocity
    (m/s, from the V/CAN ground truth) at the window's end timestamp out.
    """
    imu_cols = ["s_accel_x", "s_accel_y", "s_accel_z",
                "s_gyro_yaw", "s_gyro_pitch", "s_gyro_roll"]
    missing = [c for c in imu_cols + ["v_velocity_kmh"] if c not in df.columns]
    if missing:
        raise ValueError(f"channel_a_velocity needs columns not present in aligned frame: {missing}")

    x_windows = make_windows(df[imu_cols].to_numpy(dtype=np.float32), **cfg)
    v_windows = make_windows(df[["v_velocity_kmh"]].to_numpy(dtype=np.float32), **cfg)
    velocity_mps = v_windows[:, -1, 0] / 3.6
    return x_windows, velocity_mps.reshape(-1, 1).astype(np.float32)


BUILDERS = {
    "alignment_net": build_alignment_net,
    "channel_a_velocity": build_channel_a,
    # channel_b_velocity / road_signature / calibration_adapter: TODO, see module docstring.
}


def _load_alignment_quality(aligned_dir: Path) -> dict[str, dict] | None:
    """Read alignment_report.json written by 02_align.py, if present.
    Returns None (not {}) when the file is missing, so the caller can
    tell "no report, gate disabled" apart from "report exists, empty".
    """
    report_path = aligned_dir / "alignment_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(WINDOW_CONFIGS.keys()))
    ap.add_argument("--manifest", default="data/processed/split_manifest.json")
    ap.add_argument("--aligned-dir", default="data/raw/_aligned")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--min-corr", type=float, default=0.5,
                     help="Sessions with a recorded GPS-speed/velocity_kmh correlation below this "
                          "(per alignment_report.json) are skipped rather than windowed. Should "
                          "normally match whatever --min-corr 02_align.py was run with.")
    ap.add_argument("--include-flagged", action="store_true",
                     help="Window every aligned session regardless of recorded correlation, "
                          "including ones 02_align.py flagged as low-confidence or undefined. "
                          "Off by default - only use this deliberately (e.g. to inspect what a "
                          "bad session's windows look like), never to silently widen the dataset.")
    args = ap.parse_args()

    if args.model not in BUILDERS:
        raise SystemExit(
            f"{args.model} isn't implemented yet - see this script's module "
            "docstring for the priority order (Channel B, then road-signature, "
            "then calibration adapter)."
        )

    manifest = json.loads(Path(args.manifest).read_text())
    cfg = WINDOW_CONFIGS[args.model]
    builder = BUILDERS[args.model]
    aligned_dir = Path(args.aligned_dir)
    out_root = Path(args.out_dir) / args.model

    quality = _load_alignment_quality(aligned_dir)
    if quality is None:
        print(
            f"WARNING: no alignment_report.json found in {aligned_dir} - the alignment quality "
            "gate is DISABLED and every *_aligned.parquet file will be windowed with no check on "
            "how good that alignment actually was. Re-run 02_align.py (it always writes this "
            "report) unless you have a specific reason to skip the gate."
        )
    if args.include_flagged:
        print(
            "WARNING: --include-flagged is set - low-confidence and undefined-correlation "
            "sessions will be windowed anyway. Do not use this for a dataset that trains a model "
            "you intend to keep."
        )

    per_split: dict[str, list] = {"train": [], "val": [], "test": []}
    per_split_meta: dict[str, list] = {"train": [], "val": [], "test": []}
    skipped: list[tuple[str, str]] = []

    for sid, info in manifest["sessions"].items():
        split = info.get("split")
        if split not in per_split:
            continue
        aligned_path = aligned_dir / f"{sid}_aligned.parquet"
        if not aligned_path.exists():
            continue

        if quality is not None and not args.include_flagged:
            entry = quality.get(sid)
            if entry is None:
                skipped.append((sid, "no entry in alignment_report.json"))
                continue
            corr = entry.get("corr")
            if corr is None:
                skipped.append((sid, "undefined (NaN) alignment correlation"))
                continue
            if corr < args.min_corr:
                skipped.append((sid, f"alignment corr={corr:.2f} below --min-corr={args.min_corr}"))
                continue
            if entry.get("note") == "boundary":
                skipped.append((
                    sid,
                    f"alignment lag landed on the search boundary (corr={corr:.2f}) - per "
                    "02_align.py's own docstring this means the search didn't converge to an "
                    "interior optimum, so the recorded lag is probably not the true offset. "
                    "Re-run 02_align.py with a larger --max-lag for this session rather than "
                    "training on it as-is.",
                ))
                continue

        df = pd.read_parquet(aligned_path)
        try:
            windows, labels = builder(df, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED windowing {sid} for {args.model}: {e}")
            continue
        if len(windows) == 0:
            continue
        per_split[split].append((windows, labels))
        per_split_meta[split].extend([sid] * len(windows))

    if skipped:
        print(f"\n{len(skipped)} session(s) skipped due to low-confidence alignment (see --include-flagged to override):")
        for sid, reason in skipped:
            print(f"  {sid}: {reason}")
        print()

    for split, chunks in per_split.items():
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        if not chunks:
            print(f"[{args.model}/{split}] no windows produced.")
            continue
        windows = np.concatenate([c[0] for c in chunks], axis=0)
        labels = np.concatenate([c[1] for c in chunks], axis=0)
        np.save(split_dir / "windows.npy", windows)
        np.save(split_dir / "labels.npy", labels)
        (split_dir / "meta.json").write_text(json.dumps({"session_ids": per_split_meta[split]}))
        print(f"[{args.model}/{split}] {windows.shape[0]} windows, shape {windows.shape[1:]}, labels {labels.shape[1:]}")


if __name__ == "__main__":
    main()
