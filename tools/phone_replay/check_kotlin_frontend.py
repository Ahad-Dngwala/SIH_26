"""Second correctness gate: the Kotlin front end against the Python reference.

`tools/parity/` pins the filter math. It cannot pin anything phone-specific, because
its fixture is a clean body-frame route with no mount, no gravity and no GNSS speed.
Everything that is only a problem on a phone lives upstream of it:

  * mount leveling, which has to recover gravity from an arbitrary orientation
  * forward-axis resolution, whose sign decides whether Channel P integrates forwards
  * yaw about the gravity axis rather than about the phone's own z
  * Channel P reseeding and the blackout gate

None of those fail loudly. A flipped forward axis produces a filter that runs fine
and accelerates backwards. That is why this exists, and why it runs against a session
with a deliberately arbitrary mount rotation.

What is compared, and what is deliberately not
-----------------------------------------------
The Python front end is offline and batch: it resamples the whole session onto a
uniform grid and estimates the axis over every moving sample at once. The Kotlin one
is online and causal, because a phone does not get to see the future. They are the
same estimators fed differently, so the honest comparison is of the quantities that
should be identical regardless of ordering (gravity magnitude, forward axis) plus a
sanity band on the resulting drift, not bit equality of a trajectory.

Tolerances below are stated as constants with a reason each. Widening one to make a
failure go away is the same mistake as regenerating the parity fixture.

Run:
    python -m tools.phone_replay.check_kotlin_frontend
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from tools.benchmark_replay.blackout import BlackoutWindow
from tools.benchmark_replay.components import build_components
from tools.benchmark_replay.drift import compute_drift
from tools.benchmark_replay.pipeline import run_pipeline
from tools.phone_replay.export_session_csv import export
from tools.phone_replay.session import read_session
from tools.phone_replay.synth_session import make_session
from tools.phone_replay.to_route import distance_from_gnss_speed, session_to_route
from tools.phone_replay.session import write_session

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FUSION_SRC = (
    _REPO_ROOT
    / "android_app/app/src/main/java/org/sih26/deadreckoning/fusion"
)
_HARNESS_SRC = _REPO_ROOT / "tools/phone_replay/kotlin/ReplaySession.kt"

# Gravity is estimated from the same stationary window by the same arithmetic in both
# implementations, differing only in that Kotlin sees raw samples and Python sees a
# resampled grid. Interpolation cannot move a mean by more than millimetres per
# second squared, so anything above this is a real divergence.
GRAVITY_TOLERANCE_MPS2 = 0.02

# The forward axis is a fixed property of the mount. Both should land on the same
# unit vector with the same sign. 0.999 leaves room for the different sample sets
# feeding the PCA while still failing a genuinely different axis, and any negative
# dot product is a sign flip, which is the failure this check exists for.
AXIS_DOT_MINIMUM = 0.999

# Drift is where the two legitimately differ: the online pipeline only has Channel P
# once the axis is resolved, and it steps on ragged timestamps rather than a uniform
# grid. This is a sanity band, not a parity claim. Both must clear PS 26168's bar.
PS_26168_BAR_PERCENT = 10.0


def _python_reference(session, blackout_start: float, blackout_end: float) -> dict:
    """Run the existing offline chain, which is the reference by convention."""
    config_path = _REPO_ROOT / "tools/benchmark_replay/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["components"]["channel_a"] = "physics"
    config["components"]["channel_b"] = "none"
    config.setdefault("nhc", {})["enabled"] = False
    config.setdefault("zupt", {})["enabled"] = False

    route, report = session_to_route(session, dt_s=0.01)
    window = BlackoutWindow(start_s=blackout_start, end_s=blackout_end)
    result = run_pipeline(route, window, build_components(config))
    drift = compute_drift(route, window, result.fused_pos)

    distance = distance_from_gnss_speed(session, blackout_start, blackout_end)

    return {
        "gravity_magnitude": float(report.gravity_magnitude),
        "forward_axis": np.asarray(report.forward_axis, dtype=float),
        "blackout_distance_m": float(distance),
        "error_at_end_m": float(drift.position_error_at_end_m),
        "worst_error_m": float(drift.max_error_m),
        "drift_percent": float(drift.position_error_at_end_m / distance * 100.0),
    }


def _run_kotlin(session_csv: Path, blackout_start: float, blackout_end: float) -> dict:
    if shutil.which("kotlinc") is None:
        raise SystemExit(
            "kotlinc not found on PATH. Install the Kotlin command line compiler, or "
            "run this same check as a Gradle unit test once the Android project is "
            "set up. Skipping it is not an option: this is the only thing standing "
            "between a flipped forward axis and a wasted afternoon of driving."
        )

    with tempfile.TemporaryDirectory() as tmp:
        jar = Path(tmp) / "harness.jar"
        sources = sorted(str(p) for p in _FUSION_SRC.glob("*.kt"))
        subprocess.run(
            ["kotlinc", *sources, str(_HARNESS_SRC), "-include-runtime", "-d", str(jar)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        completed = subprocess.run(
            [
                "java",
                "-cp",
                str(jar),
                "org.sih26.deadreckoning.harness.ReplaySessionKt",
                str(session_csv),
                str(blackout_start),
                str(blackout_end),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    values: dict = {}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, raw = line.partition("=")
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        default=None,
        help="a recorded .jsonl session; a synthetic one is generated when omitted",
    )
    parser.add_argument("--blackout-start", type=float, default=60.0)
    parser.add_argument("--blackout-end", type=float, default=90.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if args.session:
            session_path = Path(args.session)
        else:
            session_path = tmp_path / "synthetic.jsonl"
            write_session(session_path, make_session(duration_s=180.0, seed=0))
            print("No session given, generated a synthetic one. This checks the")
            print("plumbing and the front end, not real-world accuracy.\n")

        session = read_session(session_path)
        csv_path = tmp_path / "session.csv"
        csv_path.write_text(export(session), encoding="utf-8")

        reference = _python_reference(session, args.blackout_start, args.blackout_end)
        kotlin = _run_kotlin(csv_path, args.blackout_start, args.blackout_end)

    axis_python = reference["forward_axis"]
    axis_kotlin = np.array([kotlin["forward_axis_x"], kotlin["forward_axis_y"]])
    axis_dot = float(np.dot(axis_python, axis_kotlin))
    gravity_delta = abs(reference["gravity_magnitude"] - kotlin["gravity_magnitude"])

    print(f"Blackout {args.blackout_start:.0f}s to {args.blackout_end:.0f}s\n")
    print(f"{'quantity':26s} {'python':>12s} {'kotlin':>12s}")
    print(f"{'gravity (m/s^2)':26s} {reference['gravity_magnitude']:12.4f} {kotlin['gravity_magnitude']:12.4f}")
    print(f"{'forward axis x':26s} {axis_python[0]:12.4f} {axis_kotlin[0]:12.4f}")
    print(f"{'forward axis y':26s} {axis_python[1]:12.4f} {axis_kotlin[1]:12.4f}")
    print(f"{'blackout distance (m)':26s} {reference['blackout_distance_m']:12.1f} {kotlin['blackout_distance_m']:12.1f}")
    print(f"{'error at reacquire (m)':26s} {reference['error_at_end_m']:12.1f} {kotlin['error_at_end_m']:12.1f}")
    print(f"{'worst error (m)':26s} {reference['worst_error_m']:12.1f} {kotlin['worst_error_m']:12.1f}")
    print(f"{'drift (%)':26s} {reference['drift_percent']:12.2f} {kotlin['drift_percent']:12.2f}")
    print(f"\n  coast-only error, no GNSS and no velocity channel: {kotlin['coast_only_error_m']:.1f} m")
    print(f"  axis agreement (dot product): {axis_dot:.6f}")
    print(f"  gravity agreement: {gravity_delta:.4f} m/s^2")

    failures = []
    if gravity_delta > GRAVITY_TOLERANCE_MPS2:
        failures.append(
            f"gravity magnitudes differ by {gravity_delta:.4f} m/s^2 "
            f"(limit {GRAVITY_TOLERANCE_MPS2}). Mount leveling does not agree, so "
            "everything downstream of it is suspect."
        )
    if axis_dot < AXIS_DOT_MINIMUM:
        detail = (
            "the sign is flipped, which makes Channel P integrate backwards"
            if axis_dot < 0
            else "the axes point in different directions"
        )
        failures.append(
            f"forward axes disagree, dot product {axis_dot:.4f} "
            f"(limit {AXIS_DOT_MINIMUM}): {detail}"
        )
    if kotlin["phase"] != "RUNNING":
        failures.append(
            f"the Kotlin pipeline ended in phase {kotlin['phase']} rather than "
            "RUNNING, so it never initialised"
        )
    for name, value in (("python", reference["drift_percent"]), ("kotlin", kotlin["drift_percent"])):
        if not value < PS_26168_BAR_PERCENT:
            failures.append(
                f"{name} drift is {value:.2f}%, over PS 26168's {PS_26168_BAR_PERCENT}% bar"
            )

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  {failure}")
        sys.exit(1)

    print("\nPASS - the Kotlin front end agrees with the Python reference.")


if __name__ == "__main__":
    main()
