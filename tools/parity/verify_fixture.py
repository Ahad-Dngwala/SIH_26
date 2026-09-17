"""Replay the parity fixture through the Python prototype and report
any divergence.

This is both a self-check - if the reference implementation stops
reproducing its own fixture, the filter's behaviour changed and someone
needs to know deliberately rather than by surprise - and the executable
specification that the C++ and Kotlin ports are written against.

Run:
    python -m tools.parity.verify_fixture
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState

_DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixture_ukf_reference.json"


def replay(fixture: dict) -> list[dict]:
    """Replay the fixture's inputs and return per-cycle outputs."""
    initial = fixture["initial_state"]
    ukf = DualChannelUkf(
        UkfState(
            pos=np.array(initial["pos_m"], dtype=float),
            vel=np.array(initial["vel_mps"], dtype=float),
            heading=float(initial["heading_rad"]),
        ),
        FusionConfig(),
    )

    outputs = []
    for cycle in fixture["cycles"]:
        step_input = cycle["input"]
        gnss_pos = (
            np.array(step_input["gnss_pos_m"], dtype=float)
            if step_input["gnss_available"]
            else None
        )
        state = ukf.step(
            dt=step_input["dt_s"],
            gyro_yaw=step_input["gyro_yaw_rps"],
            channel_a_speed=None,
            channel_b_speed=None,
            gnss_pos=gnss_pos,
            gnss_vel=None,
            road_signature_pos=None,
            road_signature_confidence=0.0,
        )
        outputs.append(
            {
                "pos_m": state.pos,
                "vel_mps": state.vel,
                "heading_rad": state.heading,
            }
        )
    return outputs


def compare(fixture: dict, outputs: list[dict]) -> dict:
    tolerances = fixture["tolerances"]
    worst = {"position_m": 0.0, "velocity_mps": 0.0, "heading_rad": 0.0}
    failures = []

    for cycle, actual in zip(fixture["cycles"], outputs):
        expected = cycle["expected"]
        deltas = {
            "position_m": float(
                np.linalg.norm(actual["pos_m"] - np.array(expected["pos_m"]))
            ),
            "velocity_mps": float(
                np.linalg.norm(actual["vel_mps"] - np.array(expected["vel_mps"]))
            ),
            "heading_rad": abs(actual["heading_rad"] - expected["heading_rad"]),
        }
        for key, value in deltas.items():
            worst[key] = max(worst[key], value)
            if value > tolerances[key]:
                failures.append((cycle["i"], cycle["t_s"], key, value))

    return {"worst": worst, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=_DEFAULT_FIXTURE)
    args = parser.parse_args()

    fixture = json.loads(Path(args.fixture).read_text())
    result = compare(fixture, replay(fixture))

    print(f"Replayed {fixture['n_cycles']} cycles against {Path(args.fixture).name}")
    for key, value in result["worst"].items():
        print(f"  worst {key:14s} {value:.3e}   (tolerance {fixture['tolerances'][key]})")

    if result["failures"]:
        print(f"\n{len(result['failures'])} cycle(s) out of tolerance:")
        for i, t_s, key, value in result["failures"][:10]:
            print(f"  cycle {i} (t={t_s:.1f}s): {key} off by {value:.4f}")
        raise SystemExit(1)

    print("\nPASS - reference implementation reproduces its own fixture.")


if __name__ == "__main__":
    main()
