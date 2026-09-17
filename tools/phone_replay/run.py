"""Measure drift on a real phone session.

    python -m tools.phone_replay.run --session <log.jsonl> \
        --blackout-start 60 --blackout-end 120 --channel-a physics

This is the tool that produces the project's first real-world number.
Everything it prints is computed offline from the raw log, not from
whatever the app displayed at the time, so a bug in the app's UI cannot
flatter the result.

The measurement it implements is PS 26168's Performance Benchmark:
positional drift as a percentage of distance travelled during the GNSS
blackout, with the bar at 10%. See `tools/benchmark_replay/drift.py` for
the formula and for what the problem statement does and does not pin
down.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from tools.benchmark_replay.blackout import BlackoutWindow, window_indices
from tools.benchmark_replay.components import build_components
from tools.benchmark_replay.drift import compute_drift
from tools.benchmark_replay.pipeline import run_pipeline
from tools.phone_replay.session import read_session
from tools.phone_replay.to_route import distance_from_gnss_speed, session_to_route

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "benchmark_replay" / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="path to a .jsonl session log")
    parser.add_argument("--blackout-start", type=float, default=None)
    parser.add_argument("--blackout-end", type=float, default=None)
    parser.add_argument(
        "--channel-a",
        default=None,
        choices=["none", "dummy", "physics", "real"],
        help="overrides components.channel_a from config.yaml",
    )
    parser.add_argument("--nhc", action="store_true", help="enable the NHC preset")
    parser.add_argument("--zupt", action="store_true", help="enable ZUPT")
    parser.add_argument("--dt", type=float, default=0.01, help="resample grid, seconds")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.channel_a:
        config["components"]["channel_a"] = args.channel_a
    config["components"]["channel_b"] = "none"
    config.setdefault("nhc", {})["enabled"] = bool(args.nhc)
    config.setdefault("zupt", {})["enabled"] = bool(args.zupt)

    session = read_session(args.session)
    if session.header.get("synthetic"):
        print(
            "NOTE: this is a synthetic session. The number below tests the "
            "measurement chain, not real-world accuracy.\n"
        )

    route, report = session_to_route(session, dt_s=args.dt)
    print(f"Session: {args.session}")
    print(report.describe())

    # --- pick the blackout window -------------------------------------------
    # Default to whatever the app actually withheld, which is the honest
    # thing to measure: it is the window the live filter really ran blind
    # through. An explicit override exists so one recording can be
    # re-cut into several windows offline.
    if args.blackout_start is None or args.blackout_end is None:
        withheld = session.gnss_t[session.gnss_withheld]
        if len(withheld) < 2:
            raise SystemExit(
                "no withheld GNSS window in this log and no --blackout-start/"
                "--blackout-end given, so there is nothing to measure"
            )
        start_s = float(withheld[0])
        end_s = float(withheld[-1])
        print(f"  blackout:      {start_s:.1f}s - {end_s:.1f}s (from the log's own flags)")
    else:
        start_s, end_s = args.blackout_start, args.blackout_end
        print(f"  blackout:      {start_s:.1f}s - {end_s:.1f}s (from the command line)")

    window = BlackoutWindow(start_s=start_s, end_s=end_s)
    components = build_components(config)
    result = run_pipeline(route, window, components)
    drift = compute_drift(route, window, result.fused_pos)

    # --- two denominators, reported side by side ----------------------------
    distance_by_speed = distance_from_gnss_speed(session, start_s, end_s)
    distance_by_path = drift.distance_traveled_m
    disagreement = abs(distance_by_speed - distance_by_path) / max(distance_by_path, 1e-6)
    drift_pct_by_speed = drift.position_error_at_end_m / distance_by_speed * 100.0

    idx = window_indices(route, window)
    truth_accuracy = float(np.mean(session.gnss_accuracy))

    print("\n--- result ---")
    print(f"  blackout duration:      {end_s - start_s:.1f} s")
    print(f"  distance (GNSS speed):  {distance_by_speed:.1f} m   <- preferred denominator")
    print(f"  distance (path length): {distance_by_path:.1f} m")
    if disagreement > 0.05:
        print(
            f"  WARNING: the two distance estimates disagree by {disagreement * 100:.1f}%. "
            "Distrust the log before the filter."
        )
    print(f"  error at reacquisition: {drift.position_error_at_end_m:.1f} m")
    print(f"  worst error in window:  {drift.max_error_m:.1f} m")
    print(f"  RMS error in window:    {drift.rms_error_m:.1f} m")
    print(f"\n  DRIFT: {drift_pct_by_speed:.2f}%  (PS 26168 bar is 10%)")
    print(f"  verdict: {'PASS' if drift_pct_by_speed < 10.0 else 'FAIL'}")
    print(
        f"\n  Ground truth is the withheld GNSS track, mean reported accuracy "
        f"{truth_accuracy:.1f} m. Errors near or below that are not measurable "
        f"with this rig."
    )
    print(f"  Samples in window: {len(idx)}")


if __name__ == "__main__":
    main()
