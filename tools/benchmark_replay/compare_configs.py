"""Run the benchmark across several channel configurations and print
the comparison table.

Exists because a single drift number is close to meaningless on its
own. "1.49%" only means something next to the number the same filter
produces with no velocity channel at all, and next to the route it was
measured on. This script produces all of them together so nobody has to
reconstruct the comparison from memory two days before a presentation.

Run:
    python -m tools.benchmark_replay.compare_configs
    python -m tools.benchmark_replay.compare_configs --markdown
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from tools.benchmark_replay.blackout import BlackoutWindow
from tools.benchmark_replay.components import build_components
from tools.benchmark_replay.drift import compute_drift
from tools.benchmark_replay.pipeline import run_pipeline
from tools.benchmark_replay.route_loader import load_route

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# (channel_a, channel_b, label, what the number actually means)
CONFIGURATIONS = [
    (
        "none",
        "none",
        "no velocity channel",
        "the honest 'what the phone does today' baseline - the UKF coasts "
        "on its CTCV process model through the blackout",
    ),
    (
        "physics",
        "none",
        "Channel P only",
        "classical integrate-and-reseed forward speed, no ML, no dataset",
    ),
    (
        "dummy",
        "dummy",
        "dummy A+B",
        "noised GROUND TRUTH velocity - not achievable, measures the UKF's "
        "own math quality only, never a system accuracy figure",
    ),
]

ROUTES = [
    (
        "constant_turn",
        0.0,
        "constant speed - a velocity channel has almost nothing to add here, "
        "since the CTCV coast is already nearly right",
    ),
    (
        "varying_speed",
        4.0,
        "speed varies +/-4 m/s - this is the route that actually discriminates "
        "a velocity channel",
    ),
]


def run_one(base: dict, channel_a: str, channel_b: str, kind: str, variation: float) -> float:
    config = copy.deepcopy(base)
    config["components"]["channel_a"] = channel_a
    config["components"]["channel_b"] = channel_b
    config["route"]["synthetic"]["kind"] = kind
    config["route"]["synthetic"]["speed_variation_mps"] = variation

    route = load_route(config)
    window = BlackoutWindow(
        start_s=config["blackout"]["start_s"], end_s=config["blackout"]["end_s"]
    )
    result = run_pipeline(route, window, build_components(config))
    return compute_drift(route, window, result.fused_pos).drift_pct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--markdown", action="store_true", help="emit a markdown table for pasting"
    )
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.config).read_text())
    results: dict[tuple[str, str], float] = {}

    for kind, variation, _ in ROUTES:
        for channel_a, channel_b, label, _meaning in CONFIGURATIONS:
            results[(kind, label)] = run_one(base, channel_a, channel_b, kind, variation)

    if args.markdown:
        header = " | ".join(["Configuration"] + [k for k, _, _ in ROUTES])
        print(f"| {header} |")
        print("|" + "---|" * (len(ROUTES) + 1))
        for _, _, label, _meaning in CONFIGURATIONS:
            cells = " | ".join(f"{results[(k, label)]:.2f}%" for k, _, _ in ROUTES)
            print(f"| {label} | {cells} |")
    else:
        for kind, _variation, note in ROUTES:
            print(f"route: {kind}")
            print(f"  ({note})")
            for _, _, label, meaning in CONFIGURATIONS:
                print(f"    {label:22s} {results[(kind, label)]:6.2f}%   - {meaning}")
            print()

    print(
        "All numbers are synthetic-route only. No real-data validation exists.\n"
        "Quoting any of these without that sentence attached is how a wrong\n"
        "figure ends up on a slide."
    )


if __name__ == "__main__":
    main()
