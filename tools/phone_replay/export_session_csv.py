"""Flatten a JSONL session log into a CSV the Kotlin harness can replay.

Same reasoning as `tools/parity/export_fixture_csv.py`: the Kotlin side of this
project has to be checkable with `kotlinc`, a JVM and Python, and nothing else. A
hand-rolled JSON parser in Kotlin would be a second thing that can be subtly wrong
inside a tool whose whole job is catching things that are subtly wrong.

The CSV is derived and disposable. The JSONL log is the record.

Format, one record per line, discriminated by the first column:

    imu,<t>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>
    gnss,<t>,<lat>,<lon>,<speed>,<bearing>,<accuracy>,<withheld 0|1>

Records are emitted in timestamp order across both streams, because that is the order
the phone saw them and the order the online pipeline has to cope with.

Run:
    python -m tools.phone_replay.export_session_csv --session log.jsonl --output /tmp/s.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.phone_replay.session import read_session


def export(session) -> str:
    rows: list[tuple[float, str]] = []

    for i in range(len(session.imu_t)):
        rows.append(
            (
                float(session.imu_t[i]),
                "imu,{:.6f},{:.6f},{:.6f},{:.6f},{:.8f},{:.8f},{:.8f}".format(
                    session.imu_t[i],
                    session.accel_xyz[i, 0],
                    session.accel_xyz[i, 1],
                    session.accel_xyz[i, 2],
                    session.gyro_xyz[i, 0],
                    session.gyro_xyz[i, 1],
                    session.gyro_xyz[i, 2],
                ),
            )
        )

    for i in range(len(session.gnss_t)):
        rows.append(
            (
                float(session.gnss_t[i]),
                "gnss,{:.6f},{:.9f},{:.9f},{:.6f},{:.6f},{:.3f},{:d}".format(
                    session.gnss_t[i],
                    session.gnss_lat[i],
                    session.gnss_lon[i],
                    session.gnss_speed[i],
                    session.gnss_bearing[i],
                    session.gnss_accuracy[i],
                    int(session.gnss_withheld[i]),
                ),
            )
        )

    # Stable sort on time, with GNSS after IMU at the same instant so the pipeline
    # sees a fix staged for the next cycle rather than applied to a cycle that has
    # already run. This mirrors the app, where the location callback and the sensor
    # callback are on different threads and the fix is always consumed by a later
    # IMU sample.
    rows.sort(key=lambda row: row[0])
    return "\n".join(row[1] for row in rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    session = read_session(args.session)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(export(session), encoding="utf-8")
    print(
        f"Wrote {output}: {len(session.imu_t)} IMU samples, {len(session.gnss_t)} GNSS fixes"
    )


if __name__ == "__main__":
    main()
