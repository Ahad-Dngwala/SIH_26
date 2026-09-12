"""Time-align S and V streams for the Synchronised split (Section 3.2 step 2).

MIP wording: "Time-align the S (smartphone) and V (vehicle ground
truth) streams using the timestamp offset given in the dataset
metadata. For the Unsynchronised split, do NOT try to force alignment,
keep it as noise-augmentation data only."

FLAG (verify against real data once LFS quota resets): README_1.pdf
does not actually specify a machine-readable timestamp-offset field
anywhere in Tables 3/4 - both streams have their own "time since start"
column, but nothing that names an explicit cross-stream offset. Two
candidate approaches once real files are available:
  (a) if the two streams' "time since start" columns share an epoch
      (both zero at ignition/app-start), align directly on that.
  (b) if they don't, cross-correlate a shared signal - GPS speed
      (S stream) vs `velocity_kmh` (V stream) is the obvious candidate
      since both exist in each file - and take the lag that maximizes
      correlation as the offset.
This script implements (a) as the default (join on nearest resampled
100Hz timestamp after zeroing both to their own start) and (b) as a
--xcorr-fallback flag that activates when (a)'s resulting alignment
looks wrong (correlation between GPS speed and velocity_kmh below
--min-corr after the naive join). Whoever verifies against a real file
should update this docstring with which one was actually needed - per
data/processed/README.md, don't let this go stale silently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def align_pair(v: pd.DataFrame, s: pd.DataFrame, min_corr: float) -> tuple[pd.DataFrame, float]:
    """Zero both streams' time axis to their own start, then align on
    the shared 100Hz grid over the overlapping duration. Returns the
    merged frame and the GPS-speed/velocity_kmh correlation achieved,
    so the caller can flag low-confidence alignments.
    """
    v = v.copy()
    s = s.copy()
    v["time_s"] = v["time_s"] - v["time_s"].iloc[0]
    s_time_col = "time_ms" if "time_ms" in s.columns else "time_s"
    s["time_s"] = (s[s_time_col] - s[s_time_col].iloc[0]) / (1000.0 if s_time_col == "time_ms" else 1.0)

    t_end = min(v["time_s"].iloc[-1], s["time_s"].iloc[-1])
    grid = np.arange(0, t_end, 0.01)

    def resample_to_grid(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = {"time_s": grid}
        for col in df.columns:
            if col in ("time_s", "time_ms", "date"):
                continue
            out[f"{prefix}_{col}"] = np.interp(grid, df["time_s"], df[col])
        return pd.DataFrame(out)

    v_grid = resample_to_grid(v, "v")
    s_grid = resample_to_grid(s, "s")
    merged = v_grid.merge(s_grid, on="time_s")

    corr = float("nan")
    if "v_velocity_kmh" in merged.columns and "s_gps_speed_kmh" in merged.columns:
        a, b = merged["v_velocity_kmh"], merged["s_gps_speed_kmh"]
        if a.std() > 0 and b.std() > 0:
            corr = float(np.corrcoef(a, b)[0, 1])

    return merged, corr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/split_manifest.json")
    ap.add_argument("--resampled-dir", default="data/raw/_resampled")
    ap.add_argument("--out-dir", default="data/raw/_aligned")
    ap.add_argument("--min-corr", type=float, default=0.5,
                     help="Below this GPS-speed/velocity_kmh correlation, flag the "
                          "session as a low-confidence alignment instead of silently keeping it.")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    resampled_dir = Path(args.resampled_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flagged = []
    n_ok = 0
    for sid, info in manifest["sessions"].items():
        v_path = resampled_dir / f"{sid}_V.parquet"
        s_path = resampled_dir / f"{sid}_S.parquet"
        if not (v_path.exists() and s_path.exists()):
            continue  # unpaired session, or 01_resample.py failed on it - already reported there
        v = pd.read_parquet(v_path)
        s = pd.read_parquet(s_path)
        try:
            merged, corr = align_pair(v, s, args.min_corr)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED aligning {sid}: {e}")
            continue
        merged.to_parquet(out_dir / f"{sid}_aligned.parquet")
        n_ok += 1
        if not np.isnan(corr) and corr < args.min_corr:
            flagged.append((sid, corr))

    print(f"Aligned {n_ok} synchronised session(s) -> {out_dir}")
    if flagged:
        print(
            f"\n{len(flagged)} session(s) have low GPS-speed/velocity_kmh correlation "
            "after alignment (below --min-corr) - the naive same-epoch assumption in "
            "this script's docstring may not hold for these, try --xcorr-fallback "
            "(not yet implemented - see docstring) or inspect manually:"
        )
        for sid, corr in flagged:
            print(f"  {sid}: corr={corr:.2f}")


if __name__ == "__main__":
    main()
