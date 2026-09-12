"""Resample every session's V and S streams to 100 Hz (Section 3.2 step 1).

Linear interpolation, not nearest-neighbor - see iovnbd_common.resample_100hz
docstring for why. Writes one parquet per (session, stream) into
data/raw/_resampled/, keyed off the manifest from 00_build_manifest.py so
this can be re-run standalone once the manifest exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iovnbd_common import S_COLUMNS, V_COLUMNS, load_csv, resample_100hz


def resample_one(csv_path: Path, schema: dict[str, str], time_col: str, out_path: Path) -> None:
    df = load_csv(csv_path, schema)
    if time_col == "time_ms":
        # S-stream time is milliseconds-since-start-of-day (see
        # 02_align.py, which already knows this and divides by 1000
        # there). resample_100hz builds its grid as t*100 assuming t is
        # in seconds - fed raw milliseconds, a ~10 minute drive's
        # (t1 - t0) is ~600,000 instead of ~600, so the grid comes out
        # ~1000x too long (hundreds of millions of points, multi-GiB
        # arrays, the allocation failures/hang seen in testing).
        # Convert to seconds here, once, before resampling, and rename
        # so every stream downstream sees a consistent "time_s" column
        # (02_align.py already falls back to "time_s" when "time_ms"
        # isn't present, so this is backward compatible).
        df = df.copy()
        df["time_ms"] = df["time_ms"] / 1000.0
        df = df.rename(columns={"time_ms": "time_s"})
        time_col = "time_s"
    df_rs = resample_100hz(df, time_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_rs.to_parquet(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/split_manifest.json")
    ap.add_argument("--out-dir", default="data/raw/_resampled")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir)

    n_ok, n_fail = 0, 0
    for sid, info in manifest["sessions"].items():
        for stream, path_key, schema, time_col in [
            ("V", "v_path", V_COLUMNS, "time_s"),
            ("S", "s_path", S_COLUMNS, "time_ms"),
        ]:
            path = info.get(path_key)
            if not path:
                continue
            out_path = out_dir / f"{sid}_{stream}.parquet"
            try:
                resample_one(Path(path), schema, time_col, out_path)
                n_ok += 1
            except Exception as e:  # noqa: BLE001 - report and keep going, don't abort the whole batch on one bad session
                print(f"FAILED {sid} [{stream}]: {e}")
                n_fail += 1

    # Also resample the unsynchronised (V-only) noise-augmentation pool.
    for entry in manifest.get("unsynchronised_sessions", []):
        sid, path = entry["session_id"], entry["v_path"]
        out_path = out_dir / f"{sid}_V.parquet"
        try:
            resample_one(Path(path), V_COLUMNS, "time_s", out_path)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {sid} [V, unsynchronised]: {e}")
            n_fail += 1

    print(f"\nResampled {n_ok} stream(s) to {out_dir}, {n_fail} failure(s).")
    if n_fail:
        print(
            "Failures are almost certainly the CSV column-matching regexes in "
            "iovnbd_common.py not yet being verified against a real header row - "
            "see that file's module docstring."
        )


if __name__ == "__main__":
    main()
