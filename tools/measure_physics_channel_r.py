"""Measure Channel P's speed-estimate error, so its UKF `r` is a number
somebody measured rather than a number somebody hoped for.

Background: `FusionConfig.r_channel_a = 0.5` is MIP Section 4.2's
*target* RMSE for a trained Channel A. No trained channel has ever met
it (the one that was trained came in at 4.89 m/s and was rejected for
bias - see `final_report.md`). Channel P is a different thing entirely
and must not inherit that number. This script produces its replacement.

What it measures
----------------
Channel P is reseeded from GNSS whenever GNSS is available, so its
error is essentially zero outside a blackout and grows inside one.
Quoting a whole-route RMSE would therefore be dominated by the easy
part and would understate the error in the only regime that matters.
This script reports the RMSE of Channel P's forward-speed estimate
against ground-truth speed **during the blackout window only**.

Caveat that belongs on any slide quoting this number: a single
integration of a biased accelerometer produces a *drift*, not white
noise, so consecutive errors are strongly correlated. The Kalman
independence assumption behind R is violated regardless of what value
is chosen. A measured blackout-window RMSE is a defensible pragmatic
proxy; it is not a statistically clean sigma, and the honest version of
this claim says so.

Run:
    python -m tools.measure_physics_channel_r
    python -m tools.measure_physics_channel_r --kind varying_speed
"""

from __future__ import annotations

import argparse

import numpy as np

from fusion_core.python_prototype.physics_speed import (
    PhysicsSpeedChannel,
    PhysicsSpeedConfig,
)
from tools.benchmark_replay.blackout import BlackoutWindow, apply_blackout, window_indices
from tools.benchmark_replay.route_loader import generate_synthetic_route


def measure(
    kind: str = "constant_turn",
    duration_s: float = 120.0,
    blackout_start_s: float = 40.0,
    blackout_end_s: float = 100.0,
    speed_variation_mps: float = 0.0,
    gnss_speed_noise_std: float = 0.3,
    seed: int = 3,
) -> dict:
    route = generate_synthetic_route(
        kind=kind,
        duration_s=duration_s,
        speed_variation_mps=speed_variation_mps,
    )
    window = BlackoutWindow(start_s=blackout_start_s, end_s=blackout_end_s)
    route = apply_blackout(route, window)

    channel = PhysicsSpeedChannel(PhysicsSpeedConfig())
    rng = np.random.default_rng(seed)

    estimated = np.full(route.n_steps, np.nan)
    for i in range(1, route.n_steps):
        dt = float(route.t[i] - route.t[i - 1])
        gnss_speed = None
        if route.gnss_available[i]:
            true_speed = float(np.linalg.norm(route.vel[i]))
            gnss_speed = true_speed + rng.normal(0.0, gnss_speed_noise_std)
        out = channel.update(
            dt=dt,
            longitudinal_accel=float(route.accel_body[i - 1][0]),
            gnss_speed=gnss_speed,
        )
        if out is not None:
            estimated[i] = out

    true_speed = np.linalg.norm(route.vel, axis=1)
    idx = window_indices(route, window)
    idx = idx[~np.isnan(estimated[idx])]

    error = estimated[idx] - true_speed[idx]
    return {
        "kind": kind,
        "n_blackout_samples": int(len(idx)),
        "rmse_mps": float(np.sqrt(np.mean(error**2))),
        "bias_mps": float(np.mean(error)),
        "max_abs_error_mps": float(np.max(np.abs(error))),
        "error_at_end_of_blackout_mps": float(error[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        default="constant_turn",
        choices=["straight", "constant_turn", "varying_speed"],
    )
    parser.add_argument("--speed-variation-mps", type=float, default=0.0)
    args = parser.parse_args()

    variation = args.speed_variation_mps
    if args.kind == "varying_speed" and variation <= 0.0:
        variation = 4.0

    result = measure(kind=args.kind, speed_variation_mps=variation)

    print(f"Channel P speed error, blackout window only, route={result['kind']}")
    print(f"  samples:                  {result['n_blackout_samples']}")
    print(f"  RMSE:                     {result['rmse_mps']:.3f} m/s   <- use this as r_mps")
    print(f"  bias:                     {result['bias_mps']:+.3f} m/s")
    print(f"  max |error|:              {result['max_abs_error_mps']:.3f} m/s")
    print(f"  error at blackout end:    {result['error_at_end_of_blackout_mps']:+.3f} m/s")
    print()
    print("Reminder: this error is a drift, not white noise. Consecutive")
    print("samples are strongly correlated, so this RMSE is a pragmatic")
    print("proxy for R, not a statistically clean sigma. Say so out loud.")


if __name__ == "__main__":
    main()
