"""Tests for the phone session harness.

Run: python -m pytest tools/phone_replay/test_phone_replay.py -v

These exist so the loader and the mount-leveling front-end are known
good *before* anyone drives anywhere. A bad afternoon of recording is
expensive; a failing test is free.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.phone_replay.session import PhoneSession, read_session, write_session
from tools.phone_replay.synth_session import make_session
from tools.phone_replay.to_route import (
    distance_from_gnss_speed,
    latlon_to_ne,
    session_to_route,
)


@pytest.fixture(scope="module")
def session() -> PhoneSession:
    return make_session(duration_s=180.0, seed=0)


def test_round_trip_preserves_samples(tmp_path, session):
    path = tmp_path / "s.jsonl"
    write_session(path, session)
    reloaded = read_session(path)

    assert len(reloaded.imu_t) == len(session.imu_t)
    assert len(reloaded.gnss_t) == len(session.gnss_t)
    np.testing.assert_allclose(reloaded.accel_xyz, session.accel_xyz, atol=1e-4)
    np.testing.assert_allclose(reloaded.gyro_xyz, session.gyro_xyz, atol=1e-5)


def test_magnetometer_round_trips(tmp_path, session):
    """PS 26168 lists the magnetometer as an expected input. We do not
    fuse it, but the log has to carry it or we cannot say that as a
    choice."""
    path = tmp_path / "s.jsonl"
    write_session(path, session)
    reloaded = read_session(path)

    assert reloaded.has_magnetometer
    assert reloaded.mag_xyz.shape == session.mag_xyz.shape
    np.testing.assert_allclose(reloaded.mag_xyz, session.mag_xyz, atol=1e-2)


def test_log_without_magnetometer_still_loads(tmp_path, session):
    """Older logs, and any phone whose magnetometer is missing or
    disabled, must not become unreadable. The stream is optional."""
    path = tmp_path / "s.jsonl"
    stripped = tmp_path / "no_mag.jsonl"
    write_session(path, session)

    with open(path, encoding="utf-8") as src, open(stripped, "w", encoding="utf-8") as dst:
        for line in src:
            record = json.loads(line)
            for key in ("mx", "my", "mz"):
                record.pop(key, None)
            dst.write(json.dumps(record) + "\n")

    reloaded = read_session(stripped)
    assert not reloaded.has_magnetometer
    assert len(reloaded.imu_t) == len(session.imu_t)
    assert "magnetometer" in " ".join(reloaded.sanity_report())


def test_truncated_final_line_is_tolerated(tmp_path, session):
    """The app can be killed mid-write. That must not cost the whole
    drive."""
    path = tmp_path / "s.jsonl"
    write_session(path, session)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"type":"imu","t":999.0,"ax":0.1')

    reloaded = read_session(path)
    assert len(reloaded.imu_t) == len(session.imu_t)


def test_gravity_compensated_log_is_flagged(session):
    """Using TYPE_LINEAR_ACCELERATION instead of TYPE_ACCELEROMETER is
    the single most likely app-side mistake, and it silently destroys
    mount leveling. The loader has to notice."""
    broken = PhoneSession(
        header=session.header,
        imu_t=session.imu_t,
        accel_xyz=session.accel_xyz - session.accel_xyz.mean(axis=0),
        gyro_xyz=session.gyro_xyz,
        gnss_t=session.gnss_t,
        gnss_lat=session.gnss_lat,
        gnss_lon=session.gnss_lon,
        gnss_speed=session.gnss_speed,
        gnss_bearing=session.gnss_bearing,
        gnss_accuracy=session.gnss_accuracy,
        gnss_withheld=session.gnss_withheld,
    )
    problems = " ".join(broken.sanity_report())
    assert "gravity-compensated" in problems


def test_mount_leveling_recovers_gravity(session):
    """The phone is at an arbitrary angle. Leveling must still land on
    a plausible gravity magnitude."""
    _, report = session_to_route(session)
    assert 9.5 < report.gravity_magnitude < 10.1
    assert report.gravity_source.startswith("stationary")


def test_forward_axis_sign_is_resolved(session):
    """An unresolved sign makes Channel P integrate backwards, which is
    worse than not having it."""
    _, report = session_to_route(session)
    assert report.forward_axis_sign_resolved
    assert np.isclose(np.linalg.norm(report.forward_axis), 1.0)


def test_route_conversion_shapes_and_frames(session):
    route, _ = session_to_route(session, dt_s=0.01)
    assert route.pos.shape == (route.n_steps, 2)
    assert route.accel_body.shape == (route.n_steps, 2)
    assert route.gyro_yaw.shape == (route.n_steps,)
    assert np.isclose(route.dt_s, 0.01)
    # Grid must be uniform even though the input timestamps were ragged.
    assert np.allclose(np.diff(route.t), 0.01, atol=1e-9)


def test_recovered_yaw_rate_tracks_truth(session):
    """Yaw about the gravity axis, not about the phone's own z. If this
    regresses, a phone-mounted filter turns the wrong way."""
    route, _ = session_to_route(session)
    turning = (route.t > 31.0) & (route.t < 39.0)
    assert np.degrees(np.mean(route.gyro_yaw[turning])) == pytest.approx(6.0, abs=0.5)


def test_speed_denominator_beats_path_length_on_noisy_truth(session):
    """The reason `run.py` prefers the speed integral: independent
    per-fix position noise inflates a summed path length, which would
    deflate our own reported drift percentage. This test pins the
    direction of that bias so nobody 'simplifies' the denominator
    later."""
    route, _ = session_to_route(session)
    start_s, end_s = 60.0, 90.0

    by_speed = distance_from_gnss_speed(session, start_s, end_s)
    in_window = (route.t >= start_s) & (route.t <= end_s)
    by_path = float(np.sum(np.linalg.norm(np.diff(route.pos[in_window], axis=0), axis=1)))

    assert by_path > by_speed
    assert by_speed > 0.9 * by_path * 0.8  # sanity: same order of magnitude


def test_latlon_projection_round_trips():
    lat0, lon0 = 23.0225, 72.5714
    lat = np.array([lat0, lat0 + 0.001])
    lon = np.array([lon0, lon0 + 0.001])
    ne = latlon_to_ne(lat, lon, lat0, lon0)

    assert ne[0, 0] == pytest.approx(0.0, abs=1e-9)
    # 0.001 degrees of latitude is about 111 m.
    assert ne[1, 0] == pytest.approx(111.0, abs=1.0)
    # Longitude is foreshortened by cos(lat).
    assert ne[1, 1] == pytest.approx(111.0 * np.cos(np.radians(lat0)), abs=1.0)


def test_fused_and_coast_streams_round_trip(tmp_path, session):
    """Fused and coast trajectory streams are optional in the log format,
    but when present they must load correctly into PhoneSession."""
    path = tmp_path / "with_fused_and_coast.jsonl"
    write_session(path, session)

    # Append fused and coast sample records in the format SessionLogger writes
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"type":"fused","t":10.000000,"pn":12.345678,"pe":98.765432}\n')
        handle.write('{"type":"coast","t":10.000000,"pn":12.345678,"pe":98.765432}\n')
        handle.write('{"type":"fused","t":10.010000,"pn":12.500000,"pe":98.900000}\n')
        handle.write('{"type":"coast","t":10.010000,"pn":12.480000,"pe":98.880000}\n')

    reloaded = read_session(path)
    assert len(reloaded.fused_t) == 2
    assert len(reloaded.coast_t) == 2
    assert reloaded.fused_pos.shape == (2, 2)
    assert reloaded.coast_pos.shape == (2, 2)
    np.testing.assert_allclose(reloaded.fused_pos[0], [12.345678, 98.765432])
    np.testing.assert_allclose(reloaded.coast_pos[1], [12.480000, 98.880000])
