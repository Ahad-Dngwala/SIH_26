"""Generate a synthetic session in the exact format the Kotlin app will
write.

Why bother, when the point of `tools/phone_replay/` is real data: the
app does not exist yet, and the loader, the mount-leveling front-end
and the drift accounting all have to be right *before* someone spends
an afternoon driving. A synthetic session in the real format lets the
whole chain be tested today, and gives the Android session a known-good
file to compare its own output against.

What this does NOT give you: any claim about real-world accuracy. The
IMU here is clean physics plus Gaussian noise, with no engine
harmonics, no pothole shocks, no mount slop and no temperature drift -
precisely the effects PS 26168 calls out as the hard part. Numbers from
a synthetic session measure the plumbing, not the system.

Deliberately included, because they break naive code:
  * an arbitrary phone mount orientation (the whole reason
    MountLeveling exists)
  * gravity present in the accelerometer
  * a full stop mid-route, so ZUPT has something to detect
  * a stationary window at the start, so leveling has something to use
  * ragged IMU timestamps, since Android never delivers a clean grid
  * a magnetometer stream, which nothing consumes, so that the logger
    and the schema are exercised on the optional fields too
"""

from __future__ import annotations

import argparse

import numpy as np

from tools.phone_replay.session import PhoneSession, write_session
from tools.phone_replay.to_route import EARTH_RADIUS_M

GRAVITY = 9.80665


def _mount_rotation(rng: np.random.Generator) -> np.ndarray:
    """A random but fixed vehicle->phone rotation, as if the phone were
    dropped into a holder at an angle nobody measured."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.2, 1.2)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def make_session(
    duration_s: float = 180.0,
    imu_hz: float = 100.0,
    seed: int = 0,
    lat0: float = 23.0225,
    lon0: float = 72.5714,
) -> PhoneSession:
    rng = np.random.default_rng(seed)
    dt = 1.0 / imu_hz
    n = int(duration_s * imu_hz)
    t = np.arange(n) * dt

    # --- speed profile: park, pull away, cruise with variation, stop, go -----
    speed = np.zeros(n)
    park_end, stop_start, stop_end = 8.0, 95.0, 115.0
    for i, ti in enumerate(t):
        if ti < park_end:
            speed[i] = 0.0
        elif ti < stop_start:
            ramp = min(1.0, (ti - park_end) / 8.0)
            speed[i] = ramp * (13.0 + 3.0 * np.sin(2 * np.pi * (ti - park_end) / 40.0))
        elif ti < stop_end:
            decel = max(0.0, 1.0 - (ti - stop_start) / 6.0)
            speed[i] = decel * 13.0
        else:
            ramp = min(1.0, (ti - stop_end) / 8.0)
            speed[i] = ramp * (12.0 + 2.5 * np.sin(2 * np.pi * (ti - stop_end) / 30.0))

    # --- heading: a couple of turns, none while stationary -------------------
    yaw_rate = np.zeros(n)
    for i, ti in enumerate(t):
        if 30.0 < ti < 40.0:
            yaw_rate[i] = np.radians(6.0)
        elif 60.0 < ti < 68.0:
            yaw_rate[i] = np.radians(-7.5)
        elif 130.0 < ti < 142.0:
            yaw_rate[i] = np.radians(5.0)
    yaw_rate[speed < 0.5] = 0.0
    heading = np.cumsum(yaw_rate) * dt

    # --- true trajectory -----------------------------------------------------
    vn = speed * np.cos(heading)
    ve = speed * np.sin(heading)
    pos = np.cumsum(np.stack([vn, ve], axis=1), axis=0) * dt

    # --- body-frame specific force -------------------------------------------
    longitudinal = np.gradient(speed, dt)
    lateral = speed * yaw_rate
    accel_vehicle = np.stack([longitudinal, lateral, np.full(n, GRAVITY)], axis=1)
    gyro_vehicle = np.stack([np.zeros(n), np.zeros(n), yaw_rate], axis=1)

    # Magnetometer, microtesla, phone frame. A nominal mid-latitude
    # field of 25 uT horizontal and 40 uT downward, rotated by vehicle
    # heading and then by the mount. No hard-iron offset, no charger
    # interference, no engine transients, so this stream is good for
    # exercising the logger and the schema and for nothing else. Nothing
    # in the fusion path reads it, by design.
    field_world_h, field_world_v = 25.0, 40.0
    mag_vehicle = np.stack(
        [
            field_world_h * np.cos(heading),
            -field_world_h * np.sin(heading),
            np.full(n, -field_world_v),
        ],
        axis=1,
    )

    rotation = _mount_rotation(rng)
    accel_phone = accel_vehicle @ rotation.T
    gyro_phone = gyro_vehicle @ rotation.T
    mag_phone = mag_vehicle @ rotation.T
    mag_phone += rng.normal(0.0, 0.4, size=mag_phone.shape)

    accel_phone += rng.normal(0.0, 0.08, size=accel_phone.shape)
    accel_phone += rng.normal(0.0, 0.03, size=3)  # fixed bias per session
    gyro_phone += rng.normal(0.0, 0.004, size=gyro_phone.shape)
    gyro_phone += rng.normal(0.0, 0.001, size=3)

    # Ragged delivery: Android jitters sample timestamps by a few ms.
    imu_t = t + rng.normal(0.0, 0.0015, size=n)
    imu_t = np.maximum.accumulate(imu_t)

    # --- GNSS at 1 Hz --------------------------------------------------------
    gnss_t = np.arange(0.0, duration_s, 1.0)
    idx = np.clip((gnss_t / dt).astype(int), 0, n - 1)
    true_ne = pos[idx]
    noisy_ne = true_ne + rng.normal(0.0, 3.0, size=true_ne.shape)

    lat = lat0 + np.degrees(noisy_ne[:, 0] / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(
        noisy_ne[:, 1] / (EARTH_RADIUS_M * np.cos(np.radians(lat0)))
    )
    gnss_speed = np.maximum(0.0, speed[idx] + rng.normal(0.0, 0.25, size=len(idx)))
    gnss_bearing = np.degrees(heading[idx]) % 360.0

    return PhoneSession(
        header={
            "device": "synthetic",
            "notes": (
                "synthetic session from tools/phone_replay/synth_session.py - "
                "plumbing test only, NOT a real-world result"
            ),
            "synthetic": True,
            "seed": seed,
        },
        imu_t=imu_t,
        accel_xyz=accel_phone,
        gyro_xyz=gyro_phone,
        gnss_t=gnss_t,
        gnss_lat=lat,
        gnss_lon=lon,
        gnss_speed=gnss_speed,
        gnss_bearing=gnss_bearing,
        gnss_accuracy=np.full(len(gnss_t), 3.0),
        gnss_withheld=np.zeros(len(gnss_t), dtype=bool),
        mag_xyz=mag_phone,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tools/phone_replay/sessions/synthetic.jsonl")
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    session = make_session(duration_s=args.duration, seed=args.seed)
    write_session(args.output, session)
    print(
        f"Wrote {args.output}: {len(session.imu_t)} IMU samples "
        f"({session.imu_rate_hz:.1f} Hz), {len(session.gnss_t)} GNSS fixes, "
        f"{session.duration_s:.0f} s"
    )


if __name__ == "__main__":
    main()
