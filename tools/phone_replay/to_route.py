"""Turn a `PhoneSession` into the `Route` the existing benchmark
pipeline already knows how to run.

The point of doing it this way, rather than writing a second pipeline
for phone data: `tools/benchmark_replay/` already has the UKF wiring,
Channel P, ZUPT, NHC, the config switches and the drift metric. If a
phone log can be presented as a `Route`, all of that runs on real data
unchanged, and the synthetic and real results are directly comparable
because they went through identical code.

What this module has to solve that synthetic routes never did
-------------------------------------------------------------
1. **Unknown mount orientation.** Synthetic routes hand the pipeline a
   clean body-frame `[ax, ay]`. A phone hands you three axes pointing
   wherever the holder happens to point, with gravity in them. That is
   what `MountLeveling` and `estimate_forward_axis` in
   `fusion_core/python_prototype/physics_speed.py` are for; this module
   is their first real caller, and it is deliberately the same code the
   Kotlin app will port, so a bug found here is a bug fixed in both.

2. **Ragged timestamps.** Android does not deliver 100 Hz on a clean
   grid. Everything is interpolated onto a uniform grid before the
   filter sees it.

3. **Ground truth that is itself noisy.** The withheld GNSS track is
   the only truth available, and it carries metres of per-fix error.
   That is a real limitation of this experiment and it is stated in
   the README rather than hidden: it puts a floor of roughly the GNSS
   accuracy on any error we can claim to measure. It does not
   invalidate the result, because the drift being measured over a
   minute of dead reckoning is expected to be much larger than the
   floor, but it does mean a sub-5-metre claim from this rig would be
   meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fusion_core.python_prototype.physics_speed import (
    MountLeveling,
    estimate_forward_axis,
)
from tools.benchmark_replay.route_loader import Route
from tools.phone_replay.session import PhoneSession

EARTH_RADIUS_M = 6378137.0


@dataclass
class ConversionReport:
    """What the conversion had to assume. Print it, don't bury it."""

    gravity_source: str
    gravity_magnitude: float
    forward_axis: np.ndarray
    forward_axis_sign_resolved: bool
    n_grid_steps: int
    dt_s: float
    warnings: list[str]

    def describe(self) -> str:
        lines = [
            f"  grid:          {self.n_grid_steps} steps @ {self.dt_s * 1000:.1f} ms",
            f"  gravity:       {self.gravity_magnitude:.3f} m/s^2 ({self.gravity_source})",
            f"  forward axis:  [{self.forward_axis[0]:+.3f}, {self.forward_axis[1]:+.3f}]"
            + ("" if self.forward_axis_sign_resolved else "  (SIGN NOT RESOLVED)"),
        ]
        for warning in self.warnings:
            lines.append(f"  warning:       {warning}")
        return "\n".join(lines)


def latlon_to_ne(
    lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
) -> np.ndarray:
    """Equirectangular projection to a local north/east frame in metres.

    Valid to well under a metre over the few kilometres a demo drive
    covers, and it keeps the state vector's `[pn, pe]` convention
    identical to the synthetic routes.
    """
    lat_rad = np.radians(lat)
    north = np.radians(lat - lat0) * EARTH_RADIUS_M
    east = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    del lat_rad
    return np.stack([north, east], axis=1)


def _estimate_gravity(
    session: PhoneSession, warnings: list[str]
) -> tuple[MountLeveling, str]:
    """Prefer an explicitly stationary window at session start; fall
    back to the whole-session mean.

    The fallback is sound for a drive: vehicle acceleration is
    zero-mean over a few minutes of normal driving, so the mean
    specific force is dominated by gravity. It is less accurate than a
    genuine stationary window, which is why the app should always begin
    a session parked for a few seconds - and why the report says which
    one was used.
    """
    accel = session.accel_xyz
    dt = 1.0 / max(session.imu_rate_hz, 1.0)
    window_n = max(int(2.0 / dt), 10)

    if len(accel) > window_n:
        head = accel[:window_n]
        # Variance of the magnitude, not of the axes: an unmoving phone
        # in any orientation has near-constant |accel|.
        if float(np.var(np.linalg.norm(head, axis=1))) < 0.05:
            return MountLeveling.from_stationary_window(head), "stationary window at start"

    warnings.append(
        "no stationary window found at session start - gravity estimated from the "
        "whole-session mean, which is less accurate. Start future sessions parked."
    )
    return (
        MountLeveling.from_stationary_window(accel),
        "whole-session mean (fallback)",
    )


def session_to_route(
    session: PhoneSession,
    dt_s: float = 0.01,
    name: str | None = None,
) -> tuple[Route, ConversionReport]:
    """Resample, level, and project a phone session onto a `Route`."""
    warnings: list[str] = []
    warnings.extend(session.sanity_report())

    t_start = max(float(session.imu_t[0]), float(session.gnss_t[0]))
    t_end = min(float(session.imu_t[-1]), float(session.gnss_t[-1]))
    if t_end - t_start < 5.0:
        raise ValueError(
            f"IMU and GNSS streams only overlap for {t_end - t_start:.1f} s - "
            "check that both used the same elapsedRealtime clock"
        )

    grid = np.arange(t_start, t_end, dt_s)
    n = len(grid)

    # --- IMU onto the grid ---------------------------------------------------
    accel_grid = np.stack(
        [np.interp(grid, session.imu_t, session.accel_xyz[:, k]) for k in range(3)],
        axis=1,
    )
    gyro_grid = np.stack(
        [np.interp(grid, session.imu_t, session.gyro_xyz[:, k]) for k in range(3)],
        axis=1,
    )

    # --- mount leveling ------------------------------------------------------
    leveling, gravity_source = _estimate_gravity(session, warnings)
    horizontal = leveling.horizontal(accel_grid)  # (N, 2), level frame

    # Yaw rate is rotation about the gravity-aligned vertical axis, not
    # about the phone's own z. Getting this wrong is the single easiest
    # way to make a phone-mounted filter turn the wrong way.
    up = leveling.rotation[2]
    gyro_yaw = gyro_grid @ up

    # --- GNSS onto the grid --------------------------------------------------
    lat0 = float(session.gnss_lat[0])
    lon0 = float(session.gnss_lon[0])
    gnss_ne = latlon_to_ne(session.gnss_lat, session.gnss_lon, lat0, lon0)

    pos = np.stack(
        [np.interp(grid, session.gnss_t, gnss_ne[:, k]) for k in range(2)], axis=1
    )

    speed = np.interp(grid, session.gnss_t, session.gnss_speed)
    bearing_rad = np.radians(session.gnss_bearing)
    # Interpolate bearing through its unwrapped form so a north crossing
    # does not produce a spurious 360-degree sweep.
    bearing_grid = np.interp(grid, session.gnss_t, np.unwrap(bearing_rad))
    vel = np.stack([speed * np.cos(bearing_grid), speed * np.sin(bearing_grid)], axis=1)

    # --- forward axis --------------------------------------------------------
    # Resolved over a driving stretch, using GNSS speed change to fix the
    # sign. Stationary stretches carry no longitudinal signal, so they are
    # excluded rather than allowed to dilute the PCA.
    moving = speed > 2.0
    sign_resolved = False
    if moving.sum() > 300:
        speed_delta = np.gradient(speed[moving])
        forward_axis = estimate_forward_axis(horizontal[moving], speed_delta)
        sign_resolved = True
    else:
        warnings.append(
            "less than 3 s of driving above 2 m/s - forward axis estimated without "
            "a speed reference, so its sign is arbitrary and Channel P may integrate "
            "backwards"
        )
        forward_axis = estimate_forward_axis(horizontal)

    longitudinal = horizontal @ forward_axis
    lateral = horizontal @ np.array([-forward_axis[1], forward_axis[0]])
    accel_body = np.stack([longitudinal, lateral], axis=1)

    heading = np.unwrap(bearing_grid)

    imu_raw = np.zeros((n, 6), dtype=np.float32)
    imu_raw[:, :3] = accel_grid.astype(np.float32)
    imu_raw[:, 3] = gyro_yaw.astype(np.float32)

    route = Route(
        name=name or f"phone_{session.header.get('device', 'session')}",
        dt_s=dt_s,
        t=grid - grid[0],
        pos=pos,
        vel=vel,
        heading=heading,
        accel_body=accel_body,
        gyro_yaw=gyro_yaw,
        gnss_available=np.ones(n, dtype=bool),
        imu_raw=imu_raw,
    )

    report = ConversionReport(
        gravity_source=gravity_source,
        gravity_magnitude=leveling.gravity_magnitude,
        forward_axis=forward_axis,
        forward_axis_sign_resolved=sign_resolved,
        n_grid_steps=n,
        dt_s=dt_s,
        warnings=warnings,
    )
    return route, report


def distance_from_gnss_speed(
    session: PhoneSession, start_s: float, end_s: float
) -> float:
    """Distance travelled over a window, integrated from GNSS
    speed-over-ground.

    This is the preferred denominator for the PS 26168 drift
    percentage on real logs. Summing the distances between consecutive
    GNSS *positions* also works, but each fix carries metres of
    independent noise, and that noise inflates a summed path length
    upward - which would deflate our own reported drift percentage.
    Speed over ground on Android is Doppler-derived rather than
    differenced from positions, so it does not share that bias.

    `tools/phone_replay/run.py` reports both, and they should agree to
    within a few percent. If they don't, distrust the log before
    distrusting the filter.
    """
    mask = (session.gnss_t >= start_s) & (session.gnss_t <= end_s)
    if mask.sum() < 2:
        raise ValueError("fewer than 2 GNSS fixes inside the window")
    t = session.gnss_t[mask]
    speed = session.gnss_speed[mask]
    return float(np.trapezoid(speed, t))
