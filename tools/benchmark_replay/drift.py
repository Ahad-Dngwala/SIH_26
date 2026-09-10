"""Drift metric: percentage of distance traveled during the blackout
window, matching ISRO's own benchmark definition "exactly" per MIP
Section 9 step 5.

Per docs/HANDOFF_benchmark_replay_tool.md section 4, this repo does
not contain ISRO's actual problem-statement document (PS 26168) with a
precise formal definition - the only concrete numbers available are
HLD/main.tex Section 1's "<5m over 50m/1min, or <100m over 1km at
60km/h" examples, which are consistent with more than one formula.
CONFIRM THE EXACT FORMULA against PS 26168 if/when someone on the team
has it, before trusting this for the real screening submission. Kept
in this one isolated function specifically so it's a one-place change
if the definition needs correcting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.benchmark_replay.blackout import BlackoutWindow, window_indices
from tools.benchmark_replay.route_loader import Route


@dataclass
class DriftResult:
    drift_pct: float
    position_error_at_end_m: float
    distance_traveled_m: float


def path_length(positions: np.ndarray) -> float:
    """Actual path length (not straight-line start-to-end distance) -
    a curved route needs the real path length, per the handoff doc."""
    if len(positions) < 2:
        return 0.0
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return float(np.sum(segment_lengths))


def compute_drift(
    route: Route,
    window: BlackoutWindow,
    fused_pos: np.ndarray,
) -> DriftResult:
    """Most literal reading per the handoff doc, section 4:

        drift_pct = (position_error_at_end_of_blackout_window
                     / distance_traveled_during_blackout_window) * 100

    where the numerator is the straight-line (Euclidean) distance
    between the fused and ground-truth position at the moment GNSS
    reacquires, and the denominator is the ground-truth path length
    covered during the blackout window.
    """
    idx = window_indices(route, window)
    if len(idx) == 0:
        raise ValueError(
            f"blackout window [{window.start_s}, {window.end_s}] contains "
            "no route samples - check dt_s against the window bounds."
        )

    end_idx = idx[-1]
    ground_truth_end = route.pos[end_idx]
    fused_end = fused_pos[end_idx]
    position_error_at_end_m = float(np.linalg.norm(fused_end - ground_truth_end))

    distance_traveled_m = path_length(route.pos[idx])
    if distance_traveled_m <= 0:
        raise ValueError(
            "distance traveled during blackout window is zero - route is "
            "stationary over this window, drift percentage is undefined."
        )

    drift_pct = position_error_at_end_m / distance_traveled_m * 100.0

    return DriftResult(
        drift_pct=drift_pct,
        position_error_at_end_m=position_error_at_end_m,
        distance_traveled_m=distance_traveled_m,
    )
