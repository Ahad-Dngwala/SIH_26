"""Plot ground truth, raw dead reckoning, and fused trajectories on
the same basemap, per MIP Section 9 step 4.

No real basemap yet: `pipeline.py` currently only runs against
synthetic routes in a local flat-earth (north, east) meter frame (see
route_loader.py), so this plots on plain axes. Once real IO-VNBD/OSM
routes are wired up (handoff doc section 7), swap the axes for an OSM
basemap (e.g. contextily or osmnx's own plotting) - keep the three-line
plotting logic itself unchanged, only the background/projection
should need to change.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed to save a PNG
import matplotlib.pyplot as plt
import numpy as np

from tools.benchmark_replay.blackout import BlackoutWindow
from tools.benchmark_replay.drift import DriftResult
from tools.benchmark_replay.pipeline import PipelineResult


def plot_trajectories(
    result: PipelineResult,
    drift: DriftResult,
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(
        result.ground_truth_pos[:, 1],
        result.ground_truth_pos[:, 0],
        color="black",
        linewidth=2,
        label="Ground truth",
        zorder=3,
    )
    ax.plot(
        result.raw_dead_reckoning_pos[:, 1],
        result.raw_dead_reckoning_pos[:, 0],
        color="tab:red",
        linewidth=1.5,
        linestyle="--",
        label="Raw double-integration DR",
        zorder=2,
    )
    ax.plot(
        result.fused_pos[:, 1],
        result.fused_pos[:, 0],
        color="tab:blue",
        linewidth=1.5,
        label="Fused output",
        zorder=2,
    )

    _shade_blackout_window(ax, result, result.blackout_window)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(
        f"{result.route.name} - drift {drift.drift_pct:.1f}% "
        f"({drift.position_error_at_end_m:.1f} m over "
        f"{drift.distance_traveled_m:.1f} m blackout path)"
    )
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _shade_blackout_window(ax, result: PipelineResult, window: BlackoutWindow) -> None:
    """Mark which stretch of ground truth falls inside the blackout
    window, so the plot makes the comparison window visually obvious
    rather than just implied by the drift number in the title."""
    route = result.route
    in_window = (route.t >= window.start_s) & (route.t < window.end_s)
    if not np.any(in_window):
        return
    blackout_pos = route.pos[in_window]
    ax.plot(
        blackout_pos[:, 1],
        blackout_pos[:, 0],
        color="black",
        linewidth=5,
        alpha=0.15,
        solid_capstyle="round",
        label="Blackout window",
        zorder=1,
    )
