"""Drift metric: positional drift as a percentage of distance traveled
during the GNSS blackout. MIP Section 9 step 5.

CONFIRMED against PS 26168 (checked 2026-09-18, previously flagged
unconfirmed in docs/HANDOFF_benchmark_replay_tool.md section 4).

The problem statement's Performance Benchmark reads: the solution must
restrict positional drift to less than 10% of the total distance
travelled using smartphone IMU sensors during GNSS blackout, with the
worked examples being under 5 m of drift over a 50 m GNSS-denied
stretch in under a minute, or under 100 m over a 1 km GNSS-denied
stretch at 60 km/h.

Both examples are exactly 10%, which confirms the shape of the
formula:

    drift_pct = positional_drift / distance_travelled_during_blackout

`compute_drift` below implements that. Two things the PS does NOT
pin down, and how this module handles each:

1. *Which* positional drift - the error at the moment GNSS reacquires,
   or the worst error at any point inside the blackout? The PS says
   "positional drift" without qualifying it, but it also requires
   "maintaining lane-level accuracy" throughout an outage, which is a
   statement about every instant, not just the last one.

   `drift_pct` stays the end-of-window reading, because that is the
   most literal reading of the benchmark sentence and it is the number
   to report against the 10% bar. But `max_error_m` and `rms_error_m`
   are now computed alongside it, because the end-of-window reading on
   its own is genuinely misleading: it samples error at a single
   instant and therefore rewards trajectories whose errors happen to
   cancel by the end. This is not hypothetical - stream 1's ZUPT work
   hit exactly that case, improving position error at every point
   through a stop while still scoring worse on end-of-window drift.
   Report `drift_pct` against the bar; look at `max_error_m` before
   believing any change actually helped.

2. Distance travelled measured how - ground-truth path length, or
   straight-line displacement? Path length is used here: it is what
   "distance travelled" means in plain English, and the 1 km-at-60 km/h
   example is plainly an odometer distance rather than a chord.
   Note this is the *conservative* choice: on a curved route path
   length exceeds displacement, making the denominator larger and the
   reported percentage smaller, so anyone checking our numbers should
   know we picked the reading that is harder to accidentally inflate
   in our favour... which is to say, check it.

On real phone logs the ground truth is the withheld GNSS track, whose
per-fix noise inflates a naively summed path length. See
`tools/phone_replay/` for how the denominator is computed there.
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
    # Added after the PS 26168 confirmation pass - see module docstring
    # point 1. These do not change `drift_pct`; they exist so a change
    # that helps everywhere except at the sampling instant cannot be
    # mistaken for a regression.
    max_error_m: float = 0.0
    max_error_pct: float = 0.0
    rms_error_m: float = 0.0

    @property
    def meets_ps_benchmark(self) -> bool:
        """PS 26168 Performance Benchmark: drift under 10% of distance
        travelled during the blackout."""
        return self.drift_pct < 10.0


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

    # Error at every instant inside the window, not just the last one.
    errors = np.linalg.norm(fused_pos[idx] - route.pos[idx], axis=1)
    max_error_m = float(np.max(errors))
    rms_error_m = float(np.sqrt(np.mean(errors**2)))

    return DriftResult(
        drift_pct=drift_pct,
        position_error_at_end_m=position_error_at_end_m,
        distance_traveled_m=distance_traveled_m,
        max_error_m=max_error_m,
        max_error_pct=max_error_m / distance_traveled_m * 100.0,
        rms_error_m=rms_error_m,
    )
