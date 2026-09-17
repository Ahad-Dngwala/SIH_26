"""Generate the cross-implementation parity fixture.

After stream 2 lands, the same filter math exists in three places:
`fusion_core/python_prototype/ukf.py`, `fusion_core/cpp/`, and the
Kotlin app. Three implementations of a numerically delicate filter
diverge silently unless something checks them - and "silently" is the
operative word, because a filter that is subtly wrong still produces
plausible-looking trajectories.

This script dumps a committed fixture: the exact per-cycle inputs of a
benchmark run, plus the Python prototype's per-cycle output state. Any
other implementation loads the same inputs, replays them, and compares
against the same expected outputs within a stated tolerance. That is
the single guardrail that makes "reimplement it in Kotlin" a reasonable
decision rather than a reckless one.

The Python prototype is the reference by definition here - not because
it is known correct, but because it is the one with unit tests and
measured benchmark numbers behind it. If C++ or Kotlin disagrees, that
is a signal to go and find out which one is wrong, not an instruction
to change the fixture until it passes.

Run:
    python -m tools.parity.generate_fixture

Output:
    tools/parity/fixture_ukf_reference.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState
from tools.benchmark_replay.blackout import BlackoutWindow, apply_blackout
from tools.benchmark_replay.route_loader import generate_synthetic_route

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "fixture_ukf_reference.json"

# Held deliberately small and deterministic. A fixture that takes a
# minute to replay is a fixture nobody runs.
FIXTURE_SCENARIO = {
    "kind": "varying_speed",
    "duration_s": 60.0,
    "dt_s": 0.1,
    "speed_mps": 16.7,
    "turn_rate_dps": 3.0,
    "speed_variation_mps": 4.0,
    "speed_period_s": 20.0,
    "accel_noise_std": 0.05,
    "gyro_noise_std": 0.01,
    "accel_bias": 0.03,
    "seed": 0,
}
FIXTURE_BLACKOUT = {"start_s": 20.0, "end_s": 45.0}

# Tolerances a conforming implementation must meet. Generous enough to
# absorb float32-vs-float64 and platform math-library differences,
# tight enough that a genuine logic divergence cannot hide underneath.
TOLERANCES = {
    "position_m": 0.5,
    "velocity_mps": 0.1,
    "heading_rad": 0.01,
}


def build_fixture() -> dict:
    route = generate_synthetic_route(**FIXTURE_SCENARIO)
    window = BlackoutWindow(**FIXTURE_BLACKOUT)
    route = apply_blackout(route, window)

    ukf = DualChannelUkf(
        UkfState(
            pos=route.pos[0].copy(),
            vel=route.vel[0].copy(),
            heading=float(route.heading[0]),
        ),
        FusionConfig(),
    )

    cycles = []
    for i in range(1, route.n_steps):
        dt = float(route.t[i] - route.t[i - 1])
        gnss_available = bool(route.gnss_available[i])
        gnss_pos = route.pos[i].copy() if gnss_available else None
        gyro_yaw = float(route.gyro_yaw[i - 1])

        state = ukf.step(
            dt=dt,
            gyro_yaw=gyro_yaw,
            # Both channels absent: this fixture pins the filter's own
            # math, not a channel's. A channel is a separate
            # implementation with its own parity question.
            channel_a_speed=None,
            channel_b_speed=None,
            gnss_pos=gnss_pos,
            gnss_vel=None,
            road_signature_pos=None,
            road_signature_confidence=0.0,
        )

        cycles.append(
            {
                "i": i,
                "t_s": float(route.t[i]),
                "input": {
                    "dt_s": dt,
                    "gyro_yaw_rps": gyro_yaw,
                    "accel_body_mps2": [float(v) for v in route.accel_body[i - 1]],
                    "gnss_available": gnss_available,
                    "gnss_pos_m": (
                        [float(v) for v in gnss_pos] if gnss_pos is not None else None
                    ),
                },
                "expected": {
                    "pos_m": [float(v) for v in state.pos],
                    "vel_mps": [float(v) for v in state.vel],
                    "heading_rad": float(state.heading),
                },
            }
        )

    config = FusionConfig()
    return {
        "schema_version": 1,
        "generated_by": "tools/parity/generate_fixture.py",
        "reference_implementation": "fusion_core/python_prototype/ukf.py",
        "note": (
            "Replay `cycles` in order, feeding each entry's `input` to your "
            "implementation's step function, and compare its output against "
            "`expected` using `tolerances`. Both velocity channels are absent "
            "throughout - this pins the filter's own math only. If your "
            "implementation disagrees, find out which one is wrong; do not "
            "regenerate this file to make the failure go away."
        ),
        "scenario": FIXTURE_SCENARIO,
        "blackout": FIXTURE_BLACKOUT,
        "tolerances": TOLERANCES,
        "initial_state": {
            "pos_m": [0.0, 0.0],
            "vel_mps": [float(FIXTURE_SCENARIO["speed_mps"]), 0.0],
            "heading_rad": 0.0,
        },
        "fusion_config": {
            "alpha": config.alpha,
            "beta": config.beta,
            "kappa": config.kappa,
            "q_pos_gnss": config.q_pos_gnss,
            "q_vel_gnss": config.q_vel_gnss,
            "q_pos_blackout": config.q_pos_blackout,
            "q_vel_blackout": config.q_vel_blackout,
            "q_psi": config.q_psi,
            "q_ba": config.q_ba,
            "q_bg": config.q_bg,
            "r_gnss_pos": config.r_gnss_pos,
            "r_gnss_vel": config.r_gnss_vel,
            "r_channel_a": config.r_channel_a,
            "r_channel_b": config.r_channel_b,
            "enable_nhc": config.enable_nhc,
            "r_nhc": config.r_nhc,
            "gnss_reacquire_ramp_s": config.gnss_reacquire_ramp_s,
            "gnss_reacquire_r_multiplier_start": config.gnss_reacquire_r_multiplier_start,
            "initial_p_diag": [1.0, 1.0, 0.5, 0.5, 0.05, 0.01, 1e-4],
        },
        "n_cycles": len(cycles),
        "cycles": cycles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    fixture = build_fixture()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=1))

    final = fixture["cycles"][-1]["expected"]
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    print(f"  cycles:      {fixture['n_cycles']}")
    print(f"  final pos:   {np.round(final['pos_m'], 3).tolist()}")
    print(f"  final vel:   {np.round(final['vel_mps'], 3).tolist()}")
    print(f"  tolerances:  {fixture['tolerances']}")


if __name__ == "__main__":
    main()
