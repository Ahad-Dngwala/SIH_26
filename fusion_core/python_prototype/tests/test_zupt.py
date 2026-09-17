"""Tests for the ZUPT detector.

Note what these do and do not claim. They verify the detector's logic -
that it fires when the IMU says stopped, stays quiet when it says
moving, and never consults the filter's own speed estimate. They do
*not* claim ZUPT improves benchmark drift; it currently does not, for
reasons recorded in `zupt.py` and the fusion-core README.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion_core.python_prototype.zupt import ZuptConfig, ZuptDetector


def _feed(detector, accel_rows, gyro_values):
    fired = []
    for accel, gyro in zip(accel_rows, gyro_values):
        fired.append(detector.update(accel_body=accel, gyro_yaw=gyro))
    return fired


def test_fires_when_stationary():
    rng = np.random.default_rng(0)
    detector = ZuptDetector()
    accel = rng.normal(0.0, 0.02, size=(60, 2))
    gyro = rng.normal(0.0, 0.005, size=60)
    fired = _feed(detector, accel, gyro)
    assert all(fired[ZuptConfig().window_n :]), "should hold once the window fills"


def test_silent_during_a_turn():
    """Centripetal acceleration means a cornering vehicle is not
    stationary, however steady its IMU variance looks."""
    rng = np.random.default_rng(1)
    detector = ZuptDetector()
    accel = np.zeros((60, 2))
    accel[:, 1] = 16.7 * np.deg2rad(3.0)  # v * yaw_rate
    accel += rng.normal(0.0, 0.05, size=(60, 2))
    gyro = np.full(60, np.deg2rad(3.0)) + rng.normal(0.0, 0.01, size=60)
    assert not any(_feed(detector, accel, gyro))


def test_silent_while_accelerating():
    rng = np.random.default_rng(2)
    detector = ZuptDetector()
    accel = np.zeros((60, 2))
    accel[:, 0] = 1.2
    accel += rng.normal(0.0, 0.05, size=(60, 2))
    gyro = rng.normal(0.0, 0.005, size=60)
    assert not any(_feed(detector, accel, gyro))


def test_stays_quiet_until_the_window_fills():
    """A spurious early fire would brake a moving filter; a missed one
    only forgoes a correction. Bias toward the latter."""
    detector = ZuptDetector(ZuptConfig(window_n=10))
    accel = np.zeros((9, 2))
    gyro = np.zeros(9)
    assert not any(_feed(detector, accel, gyro))


def test_detector_never_sees_the_filter_state():
    """Guard against the feedback loop: the detector's only inputs are
    raw IMU. If someone adds a speed-estimate argument, this fails."""
    import inspect

    parameters = set(inspect.signature(ZuptDetector.update).parameters)
    assert parameters == {"self", "accel_body", "gyro_yaw"}, (
        "ZuptDetector.update must depend only on raw IMU - gating ZUPT on the "
        "filter's own speed estimate creates a self-confirming feedback loop"
    )


def test_straight_line_cruise_is_a_known_false_positive():
    """Documents the Galilean limit rather than pretending it away: an
    IMU cannot distinguish rest from constant velocity, and on a
    vibration-free synthetic signal this detector will wrongly fire
    during straight-line cruise. Real hardware is saved by its
    vibration signature; synthetic data has none."""
    rng = np.random.default_rng(3)
    detector = ZuptDetector()
    accel = rng.normal(0.0, 0.02, size=(60, 2))  # straight, constant speed
    gyro = rng.normal(0.0, 0.005, size=60)
    assert any(_feed(detector, accel, gyro)), (
        "if this ever stops firing, the detector gained a real motion "
        "discriminator and zupt.py's Galilean caveat needs revisiting"
    )
