"""Tests for Channel P (`physics_speed.py`).

The leveling front-end is tested here rather than through the benchmark
because the benchmark's synthetic routes carry a 2-axis, already
gravity-free `accel_body` and therefore cannot exercise it at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion_core.python_prototype.physics_speed import (
    GRAVITY_MPS2,
    MountLeveling,
    PhysicsSpeedChannel,
    PhysicsSpeedConfig,
    estimate_forward_axis,
)


# --- leveling front-end -------------------------------------------------------


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def test_leveling_recovers_gravity_from_a_tilted_mount():
    """A phone sitting at an arbitrary tilt should still resolve which
    way is up to within a degree."""
    rng = np.random.default_rng(0)
    tilt = _rotation_from_axis_angle(np.array([0.3, 0.8, 0.1]), np.deg2rad(35.0))
    up_world = np.array([0.0, 0.0, 1.0])
    stationary = (tilt @ (up_world * GRAVITY_MPS2))[None, :] + rng.normal(
        0.0, 0.02, size=(200, 3)
    )

    leveling = MountLeveling.from_stationary_window(stationary)

    assert leveling.gravity_magnitude == pytest.approx(GRAVITY_MPS2, abs=0.05)
    recovered_up = leveling.rotation @ (tilt @ up_world)
    angle_error = np.degrees(np.arccos(np.clip(recovered_up[2], -1.0, 1.0)))
    assert angle_error < 1.0, f"leveling off by {angle_error:.2f} degrees"


def test_leveling_rejects_a_non_stationary_window():
    """Handing it a gravity-compensated or moving window should fail
    loudly rather than silently produce a garbage rotation."""
    with pytest.raises(ValueError, match="plausible gravity magnitude"):
        MountLeveling.from_stationary_window(np.zeros((50, 3)))


def test_forward_axis_finds_the_longitudinal_direction():
    """Accelerate/brake events dominate horizontal acceleration
    variance and lie along the vehicle's forward axis."""
    rng = np.random.default_rng(1)
    true_axis = np.array([np.cos(0.7), np.sin(0.7)])
    magnitudes = rng.normal(0.0, 1.5, size=400)
    horizontal = magnitudes[:, None] * true_axis + rng.normal(0.0, 0.1, size=(400, 2))

    axis = estimate_forward_axis(horizontal, reference_speed_delta=magnitudes)

    assert abs(float(np.dot(axis, true_axis)) - 1.0) < 0.02


def test_forward_axis_sign_follows_the_speed_reference():
    """PCA gives an axis, not a direction. With a speed reference the
    sign must point the way the vehicle actually accelerates."""
    rng = np.random.default_rng(2)
    true_axis = np.array([1.0, 0.0])
    magnitudes = rng.normal(0.0, 1.5, size=400)
    horizontal = magnitudes[:, None] * true_axis + rng.normal(0.0, 0.05, size=(400, 2))

    forward = estimate_forward_axis(horizontal, reference_speed_delta=magnitudes)
    backward = estimate_forward_axis(horizontal, reference_speed_delta=-magnitudes)

    assert float(np.dot(forward, backward)) < -0.9


def test_forward_axis_joint_estimator_survives_correlated_turn_and_accel():
    """The failure the single-regressor estimator still has, and the joint one
    fixes.

    Speed change and cornering are correlated with each other whenever a
    route's early turns happen to coincide with its early speed changes, which
    is common in practice (both cluster around junctions). When they are
    correlated, correlating horizontal accel against speed delta alone partly
    picks up the lateral component too, because the lateral component is
    itself correlated with what is being regressed against. A joint
    least-squares fit against both speed delta and the gyro-derived lateral
    regressor apportions each its own share of the variance and is not fooled
    by the correlation between them.

    Constructed so the turn's cornering magnitude and the speed delta are
    deliberately correlated (both driven by the same underlying `phase`
    signal), which is exactly the regime the single-regressor estimator
    degrades in.
    """
    rng = np.random.default_rng(11)
    n = 800
    forward_axis = np.array([np.cos(0.3), np.sin(0.3)])
    lateral_axis = np.array([-forward_axis[1], forward_axis[0]])

    phase = np.linspace(0.0, 1.0, n)
    speed_delta = 0.6 * np.sin(2 * np.pi * phase * 2) + rng.normal(0.0, 0.05, n)
    # Cornering correlated with the same phase signal, so a naive correlation
    # against speed_delta alone cannot separate the two contributions.
    lateral_regressor = 1.3 * np.sin(2 * np.pi * phase * 2 + 0.3) + rng.normal(0.0, 0.05, n)

    horizontal = (
        speed_delta[:, None] * forward_axis
        + lateral_regressor[:, None] * lateral_axis
        + rng.normal(0.0, 0.02, size=(n, 2))
    )

    joint = estimate_forward_axis(horizontal, speed_delta, lateral_regressor)
    single = estimate_forward_axis(horizontal, speed_delta)

    assert float(np.dot(joint, forward_axis)) > 0.98
    # The single-regressor estimator is not asserted to fail here - only that
    # the joint one is at least as good, so a future change to the fallback
    # cannot silently regress this test's protection.
    assert float(np.dot(joint, forward_axis)) >= float(np.dot(single, forward_axis)) - 1e-6


def test_forward_axis_joint_estimator_falls_back_when_regressors_collinear():
    """If speed delta and the lateral regressor are (numerically) collinear,
    the 2x2 design matrix is rank-deficient and the joint fit cannot separate
    the two directions. The estimator must fall back to the single-regressor
    path rather than return whatever lstsq's minimum-norm solution happens to
    produce."""
    rng = np.random.default_rng(13)
    n = 500
    true_axis = np.array([1.0, 0.0])
    speed_delta = rng.normal(0.0, 1.0, n)
    lateral_regressor = 2.0 * speed_delta  # exactly collinear

    horizontal = speed_delta[:, None] * true_axis + rng.normal(0.0, 0.05, size=(n, 2))

    axis = estimate_forward_axis(horizontal, speed_delta, lateral_regressor)
    assert abs(float(np.dot(axis, true_axis))) > 0.95


def test_forward_axis_survives_a_turn_dominated_window():
    """The failure this estimator was changed to fix.

    In a turn, lateral specific force is speed times yaw rate, which at road
    speeds is several times larger than the longitudinal content of ordinary
    speed variation. A window with enough turning in it therefore has its
    variance dominated by the lateral axis, and a PCA-first estimator
    confidently returns the axis at ninety degrees to the right answer. Channel
    P then integrates cornering force as though it were acceleration.

    The offline whole-session caller never saw this, because a full session has
    enough straight driving to dilute the turns. An online caller on a phone
    only has the samples up to now, and passes through exactly this regime
    during the first minute of every drive.

    Constructed so the lateral signal is genuinely larger and genuinely
    uncorrelated with speed change, which is what makes the correlation
    estimator immune to it.
    """
    rng = np.random.default_rng(7)
    n = 800
    forward_axis = np.array([np.cos(0.4), np.sin(0.4)])
    lateral_axis = np.array([-forward_axis[1], forward_axis[0]])

    speed_delta = rng.normal(0.0, 0.5, size=n)
    cornering = np.zeros(n)
    cornering[200:600] = 1.4  # a sustained turn, three times the longitudinal scale

    horizontal = (
        speed_delta[:, None] * forward_axis
        + cornering[:, None] * lateral_axis
        + rng.normal(0.0, 0.05, size=(n, 2))
    )

    axis = estimate_forward_axis(horizontal, reference_speed_delta=speed_delta)

    assert float(np.dot(axis, forward_axis)) > 0.95


def test_forward_axis_falls_back_to_pca_without_a_speed_reference():
    """No speed reference means nothing to correlate against, so the estimator
    has to fall back to variance and the caller owns the sign ambiguity. This
    path still has to work: it is what runs before the first GNSS fix."""
    rng = np.random.default_rng(3)
    true_axis = np.array([np.cos(1.1), np.sin(1.1)])
    magnitudes = rng.normal(0.0, 1.5, size=400)
    horizontal = magnitudes[:, None] * true_axis + rng.normal(0.0, 0.05, size=(400, 2))

    axis = estimate_forward_axis(horizontal)

    assert abs(float(np.dot(axis, true_axis))) > 0.98


# --- the channel itself -------------------------------------------------------


def test_channel_returns_none_before_the_first_gnss_reseed():
    """Before any GNSS fix the channel has no absolute reference. It
    must say so rather than fabricate a starting speed - a fabricated
    zero would brake the filter on every cycle."""
    channel = PhysicsSpeedChannel()
    for _ in range(50):
        assert channel.update(dt=0.1, longitudinal_accel=0.4) is None


def test_channel_tracks_a_known_acceleration_after_reseed():
    channel = PhysicsSpeedChannel()
    channel.reseed(10.0)
    for _ in range(100):  # 10 s at 1.0 m/s^2
        channel.update(dt=0.1, longitudinal_accel=1.0)
    assert channel.speed == pytest.approx(20.0, abs=0.2)


def test_gnss_reseed_clears_accumulated_integration_error():
    """The whole reason this channel is usable is that its open-loop
    integration only ever runs for the length of one blackout."""
    channel = PhysicsSpeedChannel()
    channel.reseed(16.7)
    for _ in range(600):  # 60 s of pure bias, no GNSS
        channel.update(dt=0.1, longitudinal_accel=0.03)
    drifted = channel.speed
    assert drifted > 18.0, "a 0.03 m/s^2 bias over 60 s should visibly drift"

    channel.update(dt=0.1, longitudinal_accel=0.03, gnss_speed=16.7)
    assert channel.speed == pytest.approx(16.7, abs=1e-6)


def test_speed_is_clamped_to_physical_bounds():
    channel = PhysicsSpeedChannel(PhysicsSpeedConfig(max_speed_mps=30.0))
    channel.reseed(20.0)
    for _ in range(200):
        channel.update(dt=0.1, longitudinal_accel=5.0)
    assert channel.speed == pytest.approx(30.0)

    for _ in range(200):
        channel.update(dt=0.1, longitudinal_accel=-5.0)
    assert channel.speed == pytest.approx(0.0)


def test_zupt_zeroes_the_integrated_speed():
    channel = PhysicsSpeedChannel()
    channel.reseed(12.0)
    assert channel.apply_zupt() == 0.0
    assert channel.speed == 0.0


def test_leak_pulls_toward_the_last_gnss_speed():
    """The leak is off by default; when on, it bounds how far an
    integrated bias can run away, at the cost of being wrong during
    genuine sustained acceleration. Both halves are the point."""
    config = PhysicsSpeedConfig(leak_tau_s=5.0)
    leaky = PhysicsSpeedChannel(config)
    plain = PhysicsSpeedChannel()
    leaky.reseed(16.7)
    plain.reseed(16.7)

    for _ in range(600):
        leaky.update(dt=0.1, longitudinal_accel=0.03)
        plain.update(dt=0.1, longitudinal_accel=0.03)

    assert leaky.speed < plain.speed
    assert abs(leaky.speed - 16.7) < abs(plain.speed - 16.7)
