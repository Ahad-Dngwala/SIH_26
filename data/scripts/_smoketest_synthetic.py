"""Not part of the pipeline - a one-off smoke test using synthetic CSVs
shaped like the documented schema, to validate the mechanics
(column-matching, resample, align, window, normalize, split-leakage
check) without real IO-VNBD data. Delete once real data is verified,
or keep as a regression test - your call.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/tmp/iovnbd_smoketest")
if ROOT.exists():
    shutil.rmtree(ROOT)
RAW = ROOT / "raw" / "IO-VNBD"
PROCESSED = ROOT / "processed"


def make_session(session_dir: Path, sid: str, duration_s: float = 30.0, driving: bool = True):
    session_dir.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * 10)  # 10 Hz native rate
    t = np.linspace(0, duration_s, n)

    speed_kmh = (30 + 10 * np.sin(t / 5)) if driving else np.zeros(n)
    heading = (t * 3) % 360

    v_df = pd.DataFrame({
        "No of GPS satellites available": np.full(n, 8),
        "Time since start of day": t,
        "Latitude": 52.4 + t * 1e-5,
        "Longitude": -1.5 + t * 1e-5,
        "velocity": speed_kmh,
        "Heading": heading,
        "Height": np.full(n, 0.1),
        "Vertical velocity": np.zeros(n),
        "Sampleperiod": np.full(n, 0.1),
        "Steering Angle": np.sin(t) * 5,
        "Wheel Speed Front Left": speed_kmh / 3.6,
        "Wheel Speed Front Right": speed_kmh / 3.6,
        "Wheel Speed Rear Left": speed_kmh / 3.6,
        "Wheel Speed Rear Right": speed_kmh / 3.6,
        "Yaw Rate": np.gradient(heading, t),
        "Indicated Vehicle Speed": speed_kmh,
        "Indicated Longitudinal Acceleration": np.gradient(speed_kmh / 3.6, t),
        "Indicated Lateral Acceleration": np.zeros(n),
        "Handbrake Activated or not": np.zeros(n),
        "Gear Requested": np.full(n, 3),
        "Gear": np.full(n, 3),
        "Engine Speed": np.full(n, 2000),
        "Coolant Temperature": np.full(n, 90),
        "Clutch Position": np.zeros(n),
        "Brake Pressure": np.zeros(n),
        "Brake Position": np.zeros(n),
        "Battery Voltage": np.full(n, 12.5),
        "Air Temperature": np.full(n, 20),
        "Accelerator Pedal Position": np.full(n, 30),
    })
    v_df.to_csv(session_dir / f"V-{sid}.csv", index=False)

    pitch_true, roll_true = np.deg2rad(2), np.deg2rad(-1)
    g = 9.81
    gravity_x = -g * np.sin(pitch_true) * np.ones(n)
    gravity_y = g * np.sin(roll_true) * np.ones(n)
    gravity_z = g * np.cos(pitch_true) * np.cos(roll_true) * np.ones(n)

    s_df = pd.DataFrame({
        "GPS Latitude": 52.4 + t * 1e-5,
        "GPS Longitude": -1.5 + t * 1e-5,
        "GPS Altitude": np.full(n, 100.0),
        "GPS Speed": speed_kmh,
        "GPS Accuracy": np.full(n, 3.0),
        "GPS Orientation": heading,
        "GPS Satellites In Range": np.full(n, 9),
        "Time Since Start": t * 1000,
        "Date": ["2024-01-01 00:00:00_000"] * n,
        "Accelerometer X": np.random.normal(0, 0.3, n) + gravity_x,
        "Accelerometer Y": np.random.normal(0, 0.3, n) + gravity_y,
        "Accelerometer Z": np.random.normal(0, 0.3, n) + gravity_z,
        "Gravity X": gravity_x,
        "Gravity Y": gravity_y,
        "Gravity Z": gravity_z,
        "Gyroscope (Yaw)": np.deg2rad(np.gradient(heading, t)),
        "Gyroscope (Pitch)": np.random.normal(0, 0.05, n),
        "Gyroscope (Roll)": np.random.normal(0, 0.05, n),
        "Magnetic Field X": np.random.normal(20, 2, n),
        "Magnetic Field Y": np.random.normal(5, 2, n),
        "Magnetic Field Z": np.random.normal(-40, 2, n),
        "Orientation (Yaw)": (heading + 5) % 360,  # +5 deg simulated mount offset
        "Orientation (Roll)": np.rad2deg(roll_true) * np.ones(n),
        "Orientation (Pitch)": np.rad2deg(pitch_true) * np.ones(n),
    })
    s_df.to_csv(session_dir / f"S-{sid}.csv", index=False)


def build_fixture():
    sync_root = RAW / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset"
    for i in range(1, 9):
        make_session(sync_root / "S (Driver A)" / f"S{i}", f"S{i}", duration_s=40.0)
    for i in range(1, 4):
        make_session(sync_root / "M (Driver B)" / f"M{i}", f"M{i}", duration_s=25.0)


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def main():
    build_fixture()
    py = sys.executable
    manifest = PROCESSED / "split_manifest.json"

    run([py, "00_build_manifest.py", "--raw-dir", str(RAW), "--out", str(manifest)])
    run([py, "01_resample.py", "--manifest", str(manifest), "--out-dir", str(ROOT / "raw" / "_resampled")])
    run([py, "02_align.py", "--manifest", str(manifest),
         "--resampled-dir", str(ROOT / "raw" / "_resampled"), "--out-dir", str(ROOT / "raw" / "_aligned")])
    for model in ("alignment_net", "channel_a_velocity"):
        run([py, "03_window.py", "--model", model, "--manifest", str(manifest),
             "--aligned-dir", str(ROOT / "raw" / "_aligned"), "--out-dir", str(PROCESSED)])
        run([py, "04_normalize.py", "--model", model, "--processed-dir", str(PROCESSED)])
    run([py, "05_split.py", "--manifest", str(manifest), "--processed-dir", str(PROCESSED)])

    # A couple of hard assertions beyond "did it crash":
    align_labels = np.load(PROCESSED / "alignment_net" / "train" / "labels.npy")
    assert align_labels.shape[1] == 4, align_labels.shape
    pitch_deg = np.rad2deg(align_labels[:, 0])
    assert np.all(np.abs(pitch_deg - 2) < 1.0), f"pitch label off: {pitch_deg[:5]}"
    print(f"\nalignment_net pitch label sanity check OK (recovered ~2 deg, got {pitch_deg[:3]})")

    ca_labels = np.load(PROCESSED / "channel_a_velocity" / "train" / "labels.npy")
    assert ca_labels.shape[1] == 1
    assert np.all(ca_labels > 0) and np.all(ca_labels < 20), f"velocity label out of range: {ca_labels[:5]}"
    print("channel_a_velocity velocity label sanity check OK")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
