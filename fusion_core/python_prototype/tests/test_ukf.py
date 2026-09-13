"""MIP Section 11.2: feed synthetic sensor data with known ground
truth (straight line, constant turn) and check the UKF converges to
the expected state.

Run with: python -m pytest fusion_core/python_prototype/tests/ -v
(from repo root, or inside `docker compose run --rm ml bash`).
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState


def _synthetic_truth(kind: str, duration_s: float, dt_s: float, speed_mps: float, turn_rate_dps: float = 0.0):
    n = int(round(duration_s / dt_s)) + 1
    t = np.arange(n) * dt_s
    turn_rate_rps = np.deg2rad(turn_rate_dps) if kind == "constant_turn" else 0.0
    heading = turn_rate_rps * t
    vel = np.stack([speed_mps * np.cos(heading), speed_mps * np.sin(heading)], axis=1)
    pos = np.zeros((n, 2))
    pos[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt_s, axis=0)
    return t, pos, vel, heading, np.full(n, turn_rate_rps)


def _run_filter(t, pos, vel, heading, gyro_yaw_true, rng, gnss_every_n=1, noise_std=0.05):
    dt = float(t[1] - t[0])
    initial = UkfState(pos=pos[0].copy(), vel=vel[0].copy(), heading=float(heading[0]))
    ukf = DualChannelUkf(initial, FusionConfig())

    est_pos = np.zeros_like(pos)
    est_pos[0] = pos[0]

    for i in range(1, len(t)):
        speed_true = float(np.linalg.norm(vel[i]))
        channel_a = speed_true + rng.normal(0.0, noise_std)
        channel_b = speed_true + rng.normal(0.0, noise_std * 3)
        gyro_meas = gyro_yaw_true[i - 1] + rng.normal(0.0, 0.005)

        gnss_available = (i % gnss_every_n) == 0
        gnss_pos = pos[i].copy() + rng.normal(0.0, 0.5, size=2) if gnss_available else None
        gnss_vel = vel[i].copy() + rng.normal(0.0, 0.05, size=2) if gnss_available else None

        state = ukf.step(
            dt=dt,
            gyro_yaw=gyro_meas,
            channel_a_speed=channel_a,
            channel_b_speed=channel_b,
            gnss_pos=gnss_pos,
            gnss_vel=gnss_vel,
            road_signature_pos=None,
            road_signature_confidence=0.0,
        )
        est_pos[i] = state.pos

    return est_pos


@pytest.mark.parametrize("kind,turn_rate_dps", [("straight", 0.0), ("constant_turn", 3.0)])
def test_ukf_converges_with_gnss(kind, turn_rate_dps):
    """With GNSS continuously available, the fused position should
    stay tightly bound to ground truth throughout - this is the
    baseline sanity check before testing blackout behavior."""
    t, pos, vel, heading, gyro_yaw_true = _synthetic_truth(
        kind, duration_s=60.0, dt_s=0.1, speed_mps=16.7, turn_rate_dps=turn_rate_dps
    )
    rng = np.random.default_rng(0)
    est_pos = _run_filter(t, pos, vel, heading, gyro_yaw_true, rng, gnss_every_n=1)

    # Skip the first couple of seconds to let the filter converge from
    # its initial covariance before checking steady-state error.
    settle_idx = int(2.0 / (t[1] - t[0]))
    err = np.linalg.norm(est_pos[settle_idx:] - pos[settle_idx:], axis=1)
    assert np.mean(err) < 3.0, f"mean position error too high with GNSS available: {np.mean(err):.2f} m"


@pytest.mark.parametrize("kind,turn_rate_dps", [("straight", 0.0), ("constant_turn", 3.0)])
def test_ukf_bounds_drift_during_blackout(kind, turn_rate_dps):
    """During a GNSS blackout, the filter (Channel A/B only) should
    stay far closer to ground truth than naive open-loop double
    integration would (HLD Section 1) - it won't be perfect, but it
    must not blow up."""
    duration_s = 120.0
    dt_s = 0.1
    t, pos, vel, heading, gyro_yaw_true = _synthetic_truth(
        kind, duration_s=duration_s, dt_s=dt_s, speed_mps=16.7, turn_rate_dps=turn_rate_dps
    )
    rng = np.random.default_rng(1)

    initial = UkfState(pos=pos[0].copy(), vel=vel[0].copy(), heading=float(heading[0]))
    ukf = DualChannelUkf(initial, FusionConfig())

    blackout_start_s, blackout_end_s = 40.0, 100.0
    est_pos = np.zeros_like(pos)
    est_pos[0] = pos[0]

    for i in range(1, len(t)):
        speed_true = float(np.linalg.norm(vel[i]))
        channel_a = speed_true + rng.normal(0.0, 0.3)
        channel_b = speed_true + rng.normal(0.0, 1.0)
        gyro_meas = gyro_yaw_true[i - 1] + rng.normal(0.0, 0.01)

        in_blackout = blackout_start_s <= t[i] < blackout_end_s
        gnss_pos = None if in_blackout else pos[i].copy() + rng.normal(0.0, 0.5, size=2)
        gnss_vel = None if in_blackout else vel[i].copy() + rng.normal(0.0, 0.05, size=2)

        state = ukf.step(
            dt=dt_s,
            gyro_yaw=gyro_meas,
            channel_a_speed=channel_a,
            channel_b_speed=channel_b,
            gnss_pos=gnss_pos,
            gnss_vel=gnss_vel,
            road_signature_pos=None,
            road_signature_confidence=0.0,
        )
        est_pos[i] = state.pos

    end_idx = np.searchsorted(t, blackout_end_s) - 1
    error_at_end_m = float(np.linalg.norm(est_pos[end_idx] - pos[end_idx]))
    distance_traveled_m = float(
        np.sum(np.linalg.norm(np.diff(pos[np.searchsorted(t, blackout_start_s):end_idx + 1], axis=0), axis=1))
    )
    drift_pct = error_at_end_m / distance_traveled_m * 100.0

    # Loose bound for a synthetic-noise, no-road-signature-anchor
    # scenario - not asserting the <10% product benchmark itself here
    # (that needs the real Channel A/B and the anchor), just that a
    # correctly-wired UKF meaningfully beats raw open-loop drift.
    assert drift_pct < 15.0, f"drift too high during blackout: {drift_pct:.2f}%"


def test_road_signature_anchor_pulls_position_toward_segment():
    """A high-confidence road-signature match should measurably pull
    the fused position toward the anchor, per Section 5.3 point 4."""
    initial = UkfState(pos=np.array([0.0, 0.0]), vel=np.array([16.7, 0.0]), heading=0.0)
    ukf = DualChannelUkf(initial, FusionConfig())

    # Run a few blackout steps with no anchor, drifting slightly.
    for _ in range(20):
        ukf.step(
            dt=0.1,
            gyro_yaw=0.02,  # small unmodeled turn the filter can't fully explain from Ch A/B alone
            channel_a_speed=16.7,
            channel_b_speed=16.7,
            gnss_pos=None,
            gnss_vel=None,
            road_signature_pos=None,
            road_signature_confidence=0.0,
        )

    pos_before = ukf.ukf.x[:2].copy()
    anchor = pos_before + np.array([5.0, -5.0])  # a deliberately offset "true" segment midpoint

    state = ukf.step(
        dt=0.1,
        gyro_yaw=0.02,
        channel_a_speed=16.7,
        channel_b_speed=16.7,
        gnss_pos=None,
        gnss_vel=None,
        road_signature_pos=anchor,
        road_signature_confidence=0.9,
    )

    dist_before = float(np.linalg.norm(pos_before - anchor))
    dist_after = float(np.linalg.norm(state.pos - anchor))
    assert dist_after < dist_before, "road-signature anchor did not pull position toward the segment"


def test_channel_a_downweighted_on_disagreement():
    """Section 5.4: when Channel A and Channel B disagree sharply, the
    filter should lean toward Channel B rather than trusting Channel A
    outright - verified indirectly by checking the filter does not
    fully adopt Channel A's (wrong) speed in one step."""
    initial = UkfState(pos=np.array([0.0, 0.0]), vel=np.array([16.7, 0.0]), heading=0.0)
    ukf = DualChannelUkf(initial, FusionConfig())

    # Channel A reports a wild, wrong speed (simulating a pothole
    # gyro-glitch/mount-slip event); Channel B stays near truth.
    state = ukf.step(
        dt=0.1,
        gyro_yaw=0.0,
        channel_a_speed=40.0,
        channel_b_speed=16.7,
        gnss_pos=None,
        gnss_vel=None,
        road_signature_pos=None,
        road_signature_confidence=0.0,
    )
    fused_speed = float(np.linalg.norm(state.vel))
    # Should sit far closer to Channel B's 16.7 than to Channel A's 40.
    assert abs(fused_speed - 16.7) < abs(fused_speed - 40.0)


def test_nhc_disabled_by_default():
    """Regression guard for a real finding (see FusionConfig.enable_nhc's
    docstring): enabling the NHC pseudo-measurement measurably hurts
    drift on this system's synthetic constant-turn benchmark, because
    Channel A/B's own MIP-specified design (rotating scalar speed by
    the current heading estimate) already bakes in the same
    zero-lateral-velocity assumption NHC asserts a second time. If
    this test ever fails because someone flipped the default to
    True, that's a deliberate design change, not a typo - it should
    come with a fresh ablation re-run (see this file's git history for
    the numbers) and a docstring update, not a silent flip."""
    assert FusionConfig().enable_nhc is False
