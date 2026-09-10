"""Append drift % + timestamp + commit hash to a simple log, per MIP
Section 11.3: run the benchmark replay tool after every significant
change to any model or to the fusion core, track drift percentage over
time so regressions are caught immediately.

Per handoff doc section 8, this is worth building from day one even
while everything is dummy - a dummy-pipeline drift number in the log
proves the logging mechanism works before it matters for real.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from tools.benchmark_replay.drift import DriftResult

_FIELDNAMES = [
    "timestamp_utc",
    "commit_hash",
    "route_name",
    "drift_pct",
    "position_error_at_end_m",
    "distance_traveled_m",
    "components_summary",
]


@dataclass
class RegressionLogEntry:
    timestamp_utc: str
    commit_hash: str
    route_name: str
    drift_pct: float
    position_error_at_end_m: float
    distance_traveled_m: float
    components_summary: str  # e.g. "channel_a=dummy,channel_b=dummy,fusion=dummy,..."


def _current_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except Exception:
        # Not fatal - a dirty/detached checkout or missing git binary
        # shouldn't block logging a drift number.
        return "unknown"


def build_entry(
    route_name: str,
    drift: DriftResult,
    components_config: dict,
) -> RegressionLogEntry:
    components_summary = ",".join(f"{k}={v}" for k, v in sorted(components_config.items()))
    return RegressionLogEntry(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        commit_hash=_current_commit_hash(),
        route_name=route_name,
        drift_pct=drift.drift_pct,
        position_error_at_end_m=drift.position_error_at_end_m,
        distance_traveled_m=drift.distance_traveled_m,
        components_summary=components_summary,
    )


def append_log(log_path: str | Path, entry: RegressionLogEntry) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(entry))
