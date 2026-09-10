"""CLI entry point for the benchmark replay tool.

MIP Section 9 / Section 11.3. Run from the repo root inside the `ml`
Docker service so PYTHONPATH is already set:

    docker compose run --rm ml python tools/benchmark_replay/run.py

Or locally, with the repo root on PYTHONPATH:

    python -m tools.benchmark_replay.run

Checkpoint 0 bar (current state of this repo): all five components
dummy, route is synthetic. See config.yaml to change any of that, and
docs/HANDOFF_benchmark_replay_tool.md section 9 for what "done" looks
like at each later checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tools.benchmark_replay.blackout import BlackoutWindow
from tools.benchmark_replay.components import build_components
from tools.benchmark_replay.drift import compute_drift
from tools.benchmark_replay.pipeline import run_pipeline
from tools.benchmark_replay.plot import plot_trajectories
from tools.benchmark_replay.regression_log import append_log, build_entry
from tools.benchmark_replay.route_loader import load_route

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main(config_path: str | Path = _DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)

    route = load_route(config)
    window = BlackoutWindow(
        start_s=config["blackout"]["start_s"],
        end_s=config["blackout"]["end_s"],
    )
    components = build_components(config)

    result = run_pipeline(route, window, components)
    drift = compute_drift(route, window, result.fused_pos)

    print(f"Route: {route.name} ({route.n_steps} steps @ {route.dt_s}s)")
    print(f"Blackout window: {window.start_s}s - {window.end_s}s")
    print(
        f"Drift: {drift.drift_pct:.2f}% "
        f"({drift.position_error_at_end_m:.2f} m over "
        f"{drift.distance_traveled_m:.2f} m traveled)"
    )

    repo_root = Path(__file__).resolve().parents[2]
    plot_path = repo_root / config["output"]["plot_path"]
    plot_trajectories(result, drift, plot_path)
    print(f"Plot saved to {plot_path}")

    log_path = repo_root / config["output"]["regression_log_path"]
    entry = build_entry(route.name, drift, config["components"])
    append_log(log_path, entry)
    print(f"Regression log updated at {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the benchmark replay tool.")
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: tools/benchmark_replay/config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
