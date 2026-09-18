"""Phone session log: schema, reader, writer.

This is the contract between the Kotlin app (`android_app/`) and every
offline tool in this repo. The app writes one of these per drive; the
rest of `tools/phone_replay/` turns it into a `Route` and runs the
existing fusion pipeline over it.

Why this file exists at all
---------------------------
Every drift number in this repo before this module was synthetic. PS
26168's benchmark is stated over real GNSS-denied driving, and
`data/processed/` is empty. A phone logging raw IMU and raw GNSS on a
real road, with the blackout applied in *software* so the withheld
fixes survive as ground truth, is the cheapest real benchmark available
to this team. That makes the log format a deliverable, not debug
plumbing - if the app writes something this loader can't read, there is
no real number.

Format
------
JSON Lines, one record per line, UTF-8, append-only. A line is either a
header, an IMU sample, or a GNSS fix, discriminated by `"type"`.

    {"type":"header","schema":1,"device":"...","started_utc":"...",
     "notes":"free text, e.g. phone flat on dash, Honda City"}
    {"type":"imu","t":<float seconds>,
     "ax":..,"ay":..,"az":..,"gx":..,"gy":..,"gz":..,
     "mx":..,"my":..,"mz":..}
    {"type":"gnss","t":<float seconds>,"lat":..,"lon":..,
     "speed":<m/s>,"bearing":<deg from north>,"accuracy":<m>,
     "withheld":<bool>}

Field notes, each of which is a thing that will otherwise go wrong:

* `t` is seconds since session start, derived from
  `SensorEvent.timestamp` / `Location.getElapsedRealtimeNanos()` - the
  monotonic elapsedRealtime clock, *not* wall time. Both streams must
  share that one clock or they cannot be aligned afterwards.
* `ax..az` are RAW `TYPE_ACCELEROMETER`, m/s^2, gravity INCLUDED, in
  the phone's own frame. Do not pre-level, do not use
  `TYPE_LINEAR_ACCELERATION`. `MountLeveling` needs the gravity vector
  and cannot recover it once a vendor filter has removed it.
* `gx..gz` are RAW `TYPE_GYROSCOPE`, rad/s, phone frame. Not
  `TYPE_ROTATION_VECTOR`.
* `mx..mz` are RAW `TYPE_MAGNETIC_FIELD`, microtesla, phone frame, and
  are OPTIONAL: a log without them loads fine and every number this
  repo computes is unchanged. They are in the schema because PS 26168
  names "accelerometer, gyroscope, and magnetometer/compass" as the
  expected on-device inputs. Nothing in this repo consumes them today.
  Registering one more listener is the difference between "we do not
  use the magnetometer" and "we cannot", and only the first of those
  is a design decision. Do not let them creep into the fusion path
  without a measured reason: a magnetometer inside a steel car body,
  next to a phone charger, is a heading source that is confidently
  wrong rather than noisily right.
* GNSS comes from `LocationManager.GPS_PROVIDER`, never the fused
  provider - see `tools/phone_replay/README.md` for why that
  distinction decides whether the experiment means anything.
* `withheld` records whether this fix was hidden from the live
  on-device filter. It is always logged either way. That flag is the
  entire experiment: withheld fixes are the ground truth the drift is
  measured against.

The app should also record its own live fused output, but that is a
separate optional stream (`{"type":"fused",...}`) used only to check
that the Kotlin filter and the Python one agree on the same input. The
drift measurement itself is computed offline from this file, so a bug
in the app's display cannot flatter the result.
Coast samples (`{"type":"coast",...}`) record the live uncorrected
inertial baseline in the same local tangent frame for direct comparison
with fused and GNSS ground truth before, during, and after blackout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class PhoneSession:
    """Raw contents of one session log, unresampled and unprocessed."""

    header: dict

    imu_t: np.ndarray  # (N,) seconds since session start
    accel_xyz: np.ndarray  # (N, 3) m/s^2, phone frame, gravity included
    gyro_xyz: np.ndarray  # (N, 3) rad/s, phone frame

    gnss_t: np.ndarray  # (M,) seconds
    gnss_lat: np.ndarray  # (M,) degrees
    gnss_lon: np.ndarray  # (M,) degrees
    gnss_speed: np.ndarray  # (M,) m/s, speed over ground
    gnss_bearing: np.ndarray  # (M,) degrees from north
    gnss_accuracy: np.ndarray  # (M,) meters
    gnss_withheld: np.ndarray  # (M,) bool, True if hidden from the live filter

    # Optional streams. Both are empty on a log that does not carry
    # them, so `len(...) == 0` is the presence test. mag_xyz sits here
    # rather than next to gyro_xyz only because a defaulted dataclass
    # field cannot precede the undefaulted gnss_* ones; it is an IMU
    # stream and is indexed in lockstep with imu_t when present.
    mag_xyz: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    fused_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    fused_pos: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    coast_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    coast_pos: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    @property
    def has_magnetometer(self) -> bool:
        return len(self.mag_xyz) > 0

    @property
    def duration_s(self) -> float:
        return float(self.imu_t[-1] - self.imu_t[0]) if len(self.imu_t) else 0.0

    @property
    def imu_rate_hz(self) -> float:
        if len(self.imu_t) < 2:
            return 0.0
        return float((len(self.imu_t) - 1) / (self.imu_t[-1] - self.imu_t[0]))

    def sanity_report(self) -> list[str]:
        """Problems worth knowing about before trusting any number
        computed from this session. Returns human-readable strings;
        empty means nothing obviously wrong.

        Deliberately returns warnings rather than raising: a session
        recorded once on a real road is expensive and should still be
        usable with its flaws stated, not thrown away.
        """
        problems: list[str] = []

        if len(self.imu_t) < 100:
            problems.append(f"only {len(self.imu_t)} IMU samples")
        if len(self.gnss_t) < 10:
            problems.append(f"only {len(self.gnss_t)} GNSS fixes")

        rate = self.imu_rate_hz
        if rate and not (60.0 <= rate <= 250.0):
            problems.append(
                f"mean IMU rate {rate:.1f} Hz is outside the plausible 60-250 Hz band"
            )

        if len(self.imu_t) > 1:
            gaps = np.diff(self.imu_t)
            worst = float(np.max(gaps))
            if worst > 0.2:
                problems.append(
                    f"largest IMU gap is {worst * 1000:.0f} ms - the app was throttled "
                    "or the foreground service was killed"
                )

        if len(self.accel_xyz):
            magnitudes = np.linalg.norm(self.accel_xyz, axis=1)
            median = float(np.median(magnitudes))
            if median < 5.0:
                problems.append(
                    f"median |accel| is {median:.2f} m/s^2, far below gravity - this "
                    "log looks gravity-compensated, so TYPE_LINEAR_ACCELERATION was "
                    "probably used by mistake and MountLeveling cannot work"
                )

        if not self.has_magnetometer:
            problems.append(
                "no magnetometer samples in this log - nothing here needs them, but "
                "PS 26168 lists magnetometer among the expected inputs, so a "
                "recording without them is weaker as evidence than it needs to be"
            )

        if len(self.gnss_t) and not self.gnss_withheld.any():
            problems.append(
                "no fix is marked withheld - this session contains no blackout, so "
                "there is no dead-reckoning result to measure"
            )

        return problems


def read_session(path: str | Path) -> PhoneSession:
    """Read a JSONL session log. Unknown record types are ignored, so
    the app can add streams without breaking this reader."""
    header: dict = {}
    imu: list[tuple] = []
    gnss: list[tuple] = []
    fused: list[tuple] = []
    coast: list[tuple] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A truncated final line is normal if the app was killed
                # mid-write; anything else is worth knowing about.
                continue

            kind = record.get("type")
            if kind == "header":
                header = record
            elif kind == "imu":
                imu.append(
                    (
                        record["t"],
                        record["ax"],
                        record["ay"],
                        record["az"],
                        record["gx"],
                        record["gy"],
                        record["gz"],
                        # Magnetometer is optional and, on a real phone,
                        # arrives on its own slower cadence, so a given
                        # IMU record may carry no sample. NaN rather than
                        # zero: zero is a plausible field reading and
                        # would be silently averaged in later.
                        record.get("mx", float("nan")),
                        record.get("my", float("nan")),
                        record.get("mz", float("nan")),
                    )
                )
            elif kind == "gnss":
                gnss.append(
                    (
                        record["t"],
                        record["lat"],
                        record["lon"],
                        record.get("speed", float("nan")),
                        record.get("bearing", float("nan")),
                        record.get("accuracy", float("nan")),
                        bool(record.get("withheld", False)),
                    )
                )
            elif kind == "fused":
                fused.append((record["t"], record["pn"], record["pe"]))
            elif kind == "coast":
                coast.append((record["t"], record["pn"], record["pe"]))

    if not imu:
        raise ValueError(f"{path} contains no IMU samples")
    if not gnss:
        raise ValueError(f"{path} contains no GNSS fixes")

    imu_arr = np.array(imu, dtype=float)
    mag_arr = imu_arr[:, 7:10]
    # A log where every magnetometer cell is NaN carried no magnetometer
    # at all. Collapse that to an empty array so `has_magnetometer` is a
    # straight answer rather than "yes, all NaN".
    if not np.isfinite(mag_arr).any():
        mag_arr = np.empty((0, 3))
    gnss_arr = np.array([g[:6] for g in gnss], dtype=float)
    withheld = np.array([g[6] for g in gnss], dtype=bool)
    fused_arr = np.array(fused, dtype=float) if fused else np.empty((0, 3))
    coast_arr = np.array(coast, dtype=float) if coast else np.empty((0, 3))

    t0 = float(imu_arr[0, 0])

    return PhoneSession(
        header=header,
        imu_t=imu_arr[:, 0] - t0,
        accel_xyz=imu_arr[:, 1:4],
        gyro_xyz=imu_arr[:, 4:7],
        gnss_t=gnss_arr[:, 0] - t0,
        gnss_lat=gnss_arr[:, 1],
        gnss_lon=gnss_arr[:, 2],
        gnss_speed=gnss_arr[:, 3],
        gnss_bearing=gnss_arr[:, 4],
        gnss_accuracy=gnss_arr[:, 5],
        gnss_withheld=withheld,
        mag_xyz=mag_arr,
        fused_t=fused_arr[:, 0] - t0 if len(fused_arr) else np.empty(0),
        fused_pos=fused_arr[:, 1:3] if len(fused_arr) else np.empty((0, 2)),
        coast_t=coast_arr[:, 0] - t0 if len(coast_arr) else np.empty(0),
        coast_pos=coast_arr[:, 1:3] if len(coast_arr) else np.empty((0, 2)),
    )


def write_session(path: str | Path, session: PhoneSession) -> None:
    """Write a session back out in the same format. Used by
    `synth_session.py` to produce test fixtures in exactly the shape
    the app will emit, so the loader is exercised before any phone
    exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        header = dict(session.header)
        header.update({"type": "header", "schema": SCHEMA_VERSION})
        handle.write(json.dumps(header) + "\n")

        has_mag = session.has_magnetometer
        for i in range(len(session.imu_t)):
            record = {
                "type": "imu",
                "t": round(float(session.imu_t[i]), 6),
                "ax": round(float(session.accel_xyz[i, 0]), 5),
                "ay": round(float(session.accel_xyz[i, 1]), 5),
                "az": round(float(session.accel_xyz[i, 2]), 5),
                "gx": round(float(session.gyro_xyz[i, 0]), 6),
                "gy": round(float(session.gyro_xyz[i, 1]), 6),
                "gz": round(float(session.gyro_xyz[i, 2]), 6),
            }
            if has_mag and np.isfinite(session.mag_xyz[i]).all():
                record["mx"] = round(float(session.mag_xyz[i, 0]), 3)
                record["my"] = round(float(session.mag_xyz[i, 1]), 3)
                record["mz"] = round(float(session.mag_xyz[i, 2]), 3)
            handle.write(json.dumps(record) + "\n")

        for i in range(len(session.gnss_t)):
            handle.write(
                json.dumps(
                    {
                        "type": "gnss",
                        "t": round(float(session.gnss_t[i]), 6),
                        "lat": float(session.gnss_lat[i]),
                        "lon": float(session.gnss_lon[i]),
                        "speed": round(float(session.gnss_speed[i]), 4),
                        "bearing": round(float(session.gnss_bearing[i]), 3),
                        "accuracy": round(float(session.gnss_accuracy[i]), 2),
                        "withheld": bool(session.gnss_withheld[i]),
                    }
                )
                + "\n"
            )

        for i in range(len(session.fused_t)):
            handle.write(
                json.dumps(
                    {
                        "type": "fused",
                        "t": round(float(session.fused_t[i]), 6),
                        "pn": float(session.fused_pos[i, 0]),
                        "pe": float(session.fused_pos[i, 1]),
                    }
                )
                + "\n"
            )

        for i in range(len(session.coast_t)):
            handle.write(
                json.dumps(
                    {
                        "type": "coast",
                        "t": round(float(session.coast_t[i]), 6),
                        "pn": float(session.coast_pos[i, 0]),
                        "pe": float(session.coast_pos[i, 1]),
                    }
                )
                + "\n"
            )
