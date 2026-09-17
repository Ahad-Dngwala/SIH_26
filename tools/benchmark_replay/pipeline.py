"""Wire route + blackout + components together and produce the three
trajectories MIP Section 9 step 4 plots: raw double-integration dead
reckoning, the fused output, and ground truth.

Per handoff doc section 3, the three have very different dependencies
- ground truth has none, raw dead reckoning is pure signal processing
with no model/UKF dependency, and the fused trajectory is the only one
that depends on components.py's current dummy/real mix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.benchmark_replay.blackout import BlackoutWindow
from tools.benchmark_replay.components import ComponentSet, FusionState
from tools.benchmark_replay.route_loader import Route


@dataclass
class PipelineResult:
    route: Route
    ground_truth_pos: np.ndarray  # (N, 2)
    raw_dead_reckoning_pos: np.ndarray  # (N, 2)
    fused_pos: np.ndarray  # (N, 2), map-matched (snapped) position
    fused_pos_unsnapped: np.ndarray  # (N, 2), pre-map-matching, for debugging
    blackout_window: BlackoutWindow


def run_raw_dead_reckoning(route: Route) -> np.ndarray:
    """The naive baseline the whole system exists to beat (HLD Section
    1: "blows up... within seconds of GNSS loss"). Double-integrates
    the raw, noisy, biased body-frame accelerometer once for velocity,
    once more for position - no models, no UKF, no GNSS correction at
    all, ever (not just during the blackout window). This is
    intentional: it is the pure open-loop integration baseline, and
    correcting it with GNSS would defeat the point of showing how
    badly it drifts.

    Heading is integrated from the (also noisy) gyro, since raw
    dead-reckoning has no other source of orientation - this is where
    the "quadratic-in-time" error term from HLD Section 1 actually
    comes from: heading error compounds into velocity direction error,
    which compounds into position error.
    """
    n = route.n_steps
    heading = np.zeros(n)
    vel = np.zeros((n, 2))
    pos = np.zeros((n, 2))

    for i in range(1, n):
        dt = route.t[i] - route.t[i - 1]
        heading[i] = heading[i - 1] + route.gyro_yaw[i - 1] * dt
        # Rotate body-frame accel into the world frame using the
        # (drifting) integrated heading, then integrate once for
        # velocity, once more for position.
        ax, ay = route.accel_body[i - 1]
        accel_world = np.array(
            [
                ax * np.cos(heading[i - 1]) - ay * np.sin(heading[i - 1]),
                ax * np.sin(heading[i - 1]) + ay * np.cos(heading[i - 1]),
            ]
        )
        vel[i] = vel[i - 1] + accel_world * dt
        pos[i] = pos[i - 1] + (vel[i - 1] + vel[i]) / 2.0 * dt

    return pos


def run_fused_pipeline(
    route: Route,
    blackout: BlackoutWindow,
    components: ComponentSet,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full per-cycle loop (Channel A, Channel B, road-
    signature, UKF, map-matching) over the route, honoring the
    blackout window's GNSS mask. Returns (snapped_pos, unsnapped_pos).

    Mirrors the runtime loop shape in MIP Section 7.3, minus the
    on-device-specific steps (ring buffer, periodic-not-every-cycle
    alignment net) that don't apply to an offline replay.
    """
    n = route.n_steps
    unsnapped = np.zeros((n, 2))
    snapped = np.zeros((n, 2))

    state = FusionState(pos=route.pos[0].copy(), vel=route.vel[0].copy(), heading=route.heading[0])
    unsnapped[0] = state.pos
    snapped[0] = components.map_matching.snap(state.pos)

    for i in range(1, n):
        dt = route.t[i] - route.t[i - 1]

        channel_a_speed = components.channel_a.estimate(route, i)
        channel_b_speed = components.channel_b.estimate(route, i)
        road_signature = components.road_signature.classify(route, i)

        gnss_pos = route.pos[i].copy() if route.gnss_available[i] else None

        state = components.fusion.step(
            state=state,
            dt=dt,
            accel_body=route.accel_body[i - 1],
            gyro_yaw=route.gyro_yaw[i - 1],
            channel_a_speed=channel_a_speed,
            channel_b_speed=channel_b_speed,
            gnss_pos=gnss_pos,
            road_signature=road_signature,
            road_signature_threshold=components.road_signature_threshold,
            r_channel_a_override=components.channel_a_r_override,
        )
        unsnapped[i] = state.pos
        snapped[i] = components.map_matching.snap(state.pos)

    return snapped, unsnapped


def run_pipeline(
    route: Route,
    blackout: BlackoutWindow,
    components: ComponentSet,
) -> PipelineResult:
    from tools.benchmark_replay.blackout import apply_blackout

    blacked_out_route = apply_blackout(route, blackout)

    raw_dr_pos = run_raw_dead_reckoning(route)
    fused_pos, fused_pos_unsnapped = run_fused_pipeline(
        blacked_out_route, blackout, components
    )

    return PipelineResult(
        route=route,
        ground_truth_pos=route.pos,
        raw_dead_reckoning_pos=raw_dr_pos,
        fused_pos=fused_pos,
        fused_pos_unsnapped=fused_pos_unsnapped,
        blackout_window=blackout,
    )
