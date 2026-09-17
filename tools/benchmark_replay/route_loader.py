"""Load a route into the common in-memory format the rest of the
benchmark tool consumes.

MIP Section 9 step 1 / handoff doc section 7: `data/processed/` is
empty as of this writing (Layer 1 hasn't run Section 3 yet), so the
only usable source right now is a synthetic route - a straight-line or
constant-turn trajectory with synthetic ground truth and synthetic
IMU-like noise, in the same spirit as the fusion-core unit tests in
Section 11.2. Swap to `load_io_vnbd_route` once `data/processed/` has
something real; coordinate with Layer 1 on the exact file format they
land on before wiring it up for real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Route:
    """Common in-memory route format. Body-frame IMU + world-frame
    ground truth, one row per timestep at a fixed dt.

    Units: seconds, meters, meters/second, radians, radians/second,
    meters/second^2. Position is in a local flat-earth (north, east)
    frame in meters, not lat/lon - fine for synthetic routes and for
    IO-VNBD segments short enough that curvature doesn't matter; note
    this assumption if a real route ever needs otherwise.
    """

    name: str
    dt_s: float
    t: np.ndarray  # (N,) timestamps, seconds from route start
    pos: np.ndarray  # (N, 2) ground-truth [north, east] position, meters
    vel: np.ndarray  # (N, 2) ground-truth [v_north, v_east], m/s
    heading: np.ndarray  # (N,) ground-truth heading, radians
    accel_body: np.ndarray  # (N, 2) synthetic/measured body-frame [ax, ay], m/s^2, WITH noise+bias
    gyro_yaw: np.ndarray  # (N,) synthetic/measured yaw rate, rad/s, WITH noise
    gnss_available: np.ndarray  # (N,) bool, True unless a blackout.py mask has been applied

    # Raw 6-channel IMU (Phase 4): (N, 6) float32 — [accel_x, accel_y, accel_z,
    # gyro_yaw, gyro_pitch, gyro_roll] in body frame at 100Hz.
    # Used by RealVelocityEstimator.estimate() to build input windows on the fly.
    # For synthetic routes, synthesized from accel_body + gyro_yaw; for real IO-VNBD
    # routes, loaded directly from the S-stream _aligned.parquet.
    imu_raw: np.ndarray = field(default_factory=lambda: np.empty((0, 6), dtype=np.float32))

    @property
    def n_steps(self) -> int:
        return len(self.t)

    def get_imu_window(self, i: int, window_len: int = 200,
                       n_imu_channels: int = 6) -> np.ndarray:
        """Build a raw (not normalized) IMU window of shape (window_len, n_imu_channels)
        ending at step i (inclusive).

        Zero-pads at the start of the route where insufficient history exists.
        This matches the 03_window.py make_windows() convention.

        Args:
            i:               current route step index (0-indexed).
            window_len:      timesteps in window (200 for Channel A, 400 for Channel B).
            n_imu_channels:  IMU channels to return (6 for Channel A, 3 for Channel B).

        Returns:
            (window_len, n_imu_channels) float32, raw (not normalized).
        """
        if len(self.imu_raw) == 0:
            # Synthetic routes don't populate imu_raw; synthesize from existing fields.
            synthetic_imu = np.zeros((len(self.t), 6), dtype=np.float32)
            synthetic_imu[:, :2] = self.accel_body.astype(np.float32)  # ax, ay
            synthetic_imu[:, 3]  = self.gyro_yaw.astype(np.float32)    # gyro_yaw
            raw = synthetic_imu
        else:
            raw = self.imu_raw

        start = max(0, i - window_len + 1)
        chunk = raw[start : i + 1, :n_imu_channels]  # (≤window_len, n_imu_channels)
        if len(chunk) < window_len:
            pad = np.zeros((window_len - len(chunk), n_imu_channels), dtype=np.float32)
            chunk = np.concatenate([pad, chunk], axis=0)
        return chunk.astype(np.float32)


def _speed_profile(
    kind: str,
    t: np.ndarray,
    speed_mps: float,
    speed_variation_mps: float,
    speed_period_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (speed, longitudinal_accel) over `t` for a route kind.

    `straight` and `constant_turn` are flat: speed is constant and
    longitudinal acceleration is exactly zero.

    `varying_speed` adds a smooth sinusoidal speed profile. This kind
    exists because the two constant-speed kinds cannot discriminate a
    velocity channel *at all*: `ukf.fx` is a constant-speed coast
    model, so on a constant-speed route the coast is already exactly
    correct and any velocity channel - learned or physical - can only
    add noise to a perfect answer. Benchmarking a velocity channel on
    a constant-speed route measures nothing useful and will make a
    working channel look harmful. Use this kind whenever the thing
    under test is a velocity source.

    A sinusoid rather than a step or a random walk because it is
    smooth (no unphysical jerk for the IMU synthesis to represent),
    has an analytic derivative, and is zero-mean in acceleration -
    which means it does not secretly help or hurt an integrator the
    way a monotone ramp would.
    """
    if kind in ("straight", "constant_turn"):
        return np.full(len(t), speed_mps), np.zeros(len(t))

    omega = 2.0 * np.pi / speed_period_s
    speed = speed_mps + speed_variation_mps * np.sin(omega * t)
    longitudinal_accel = speed_variation_mps * omega * np.cos(omega * t)
    if np.any(speed <= 0.0):
        raise ValueError(
            f"speed_variation_mps ({speed_variation_mps}) is too large for "
            f"speed_mps ({speed_mps}) - the profile would reverse direction, "
            "which the CTCV model and the drift metric both assume never "
            "happens."
        )
    return speed, longitudinal_accel


def generate_synthetic_route(
    kind: str = "constant_turn",
    duration_s: float = 120.0,
    dt_s: float = 0.1,
    speed_mps: float = 16.7,
    turn_rate_dps: float = 3.0,
    accel_noise_std: float = 0.05,
    gyro_noise_std: float = 0.01,
    accel_bias: float = 0.03,
    speed_variation_mps: float = 0.0,
    speed_period_s: float = 40.0,
    seed: int = 0,
) -> Route:
    """Build a synthetic route, either straight, constant-turn-rate, or
    constant-turn-rate with a varying speed profile, with synthetic
    body-frame IMU derived from the ground-truth motion plus noise and
    a constant accelerometer bias.

    The accelerometer bias is deliberate, not an oversight: it's what
    makes the raw double-integration baseline in pipeline.py actually
    drift (see HLD/main.tex Section 1 - "quadratic-in-time error
    term"). Zero bias would make the naive baseline look artificially
    good and defeat the point of the comparison.

    `kind="varying_speed"` uses `speed_variation_mps` / `speed_period_s`
    to modulate speed around `speed_mps` - see `_speed_profile` for why
    that kind exists and when it must be used.
    """
    if kind not in ("straight", "constant_turn", "varying_speed"):
        raise ValueError(f"Unknown synthetic route kind: {kind!r}")
    if kind == "varying_speed" and speed_variation_mps <= 0.0:
        raise ValueError(
            "kind='varying_speed' with speed_variation_mps <= 0 is just a "
            "constant-speed route under a misleading name - set a positive "
            "speed_variation_mps or use kind='constant_turn'."
        )

    rng = np.random.default_rng(seed)
    n = int(round(duration_s / dt_s)) + 1
    t = np.arange(n) * dt_s

    turn_rate_rps = (
        np.deg2rad(turn_rate_dps) if kind in ("constant_turn", "varying_speed") else 0.0
    )
    heading = turn_rate_rps * t  # rad, starts at 0

    speed, longitudinal_accel = _speed_profile(
        kind, t, speed_mps, speed_variation_mps, speed_period_s
    )

    # Ground-truth world-frame velocity and position.
    vel = np.stack([speed * np.cos(heading), speed * np.sin(heading)], axis=1)
    pos = np.zeros((n, 2))
    pos[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt_s, axis=0)

    # Body-frame ground truth: forward accel is the speed profile's
    # derivative (zero for the constant-speed kinds), lateral accel
    # comes from the turn (v * yaw_rate).
    accel_body_true = np.zeros((n, 2))
    accel_body_true[:, 0] = longitudinal_accel  # forward accel, m/s^2
    accel_body_true[:, 1] = speed * turn_rate_rps  # lateral accel, m/s^2
    gyro_yaw_true = np.full(n, turn_rate_rps)

    accel_body = (
        accel_body_true
        + accel_bias
        + rng.normal(0.0, accel_noise_std, size=(n, 2))
    )
    gyro_yaw = gyro_yaw_true + rng.normal(0.0, gyro_noise_std, size=n)

    return Route(
        name=f"synthetic_{kind}",
        dt_s=dt_s,
        t=t,
        pos=pos,
        vel=vel,
        heading=heading,
        accel_body=accel_body,
        gyro_yaw=gyro_yaw,
        gnss_available=np.ones(n, dtype=bool),
    )


def load_io_vnbd_route(processed_path: str | Path) -> Route:
    """Load a held-out IO-VNBD (or collected) route from
    `data/processed/`.

    Not implemented yet: `data/processed/` is empty as of this
    writing (see data/README.md - Layer 1 hasn't run Section 3). Once
    it exists, implement this against whatever file layout Layer 1
    lands on (coordinate with them per data/README.md's note that
    changing the interface after Layer 2 depends on it is expensive),
    and update config.yaml's `route.source` default once it works.
    """
    raise NotImplementedError(
        "IO-VNBD loading is not implemented yet - data/processed/ is empty "
        "(Layer 1 hasn't run Section 3). Use route.source: synthetic in "
        "config.yaml until a processed route exists. See "
        "docs/HANDOFF_benchmark_replay_tool.md section 7."
    )


def load_route(config: dict) -> Route:
    """Dispatch on config.yaml's `route.source`."""
    route_cfg = config["route"]
    source = route_cfg["source"]
    if source == "synthetic":
        return generate_synthetic_route(**route_cfg["synthetic"])
    if source == "io_vnbd":
        processed_path = route_cfg["io_vnbd"]["processed_path"]
        if not processed_path:
            raise ValueError(
                "route.source is 'io_vnbd' but route.io_vnbd.processed_path "
                "is not set in config.yaml."
            )
        return load_io_vnbd_route(processed_path)
    raise ValueError(f"Unknown route.source: {source!r}")
