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
100Hz timestamp after zeroing both to their own start) and (b) as an
xcorr fallback that activates when (a)'s resulting alignment looks
wrong (correlation between GPS speed and velocity_kmh below
--min-corr after the naive join).

VERIFIED against real IO-VNBD sessions (see 02_align.py run output,
71 synchronised sessions): (a) alone leaves 15/71 sessions below
--min-corr=0.5, some strongly negative (e.g. Vta03: -0.71). The xcorr
fallback below searches a bounded lag window and re-aligns on
whichever shift maximizes correlation, which is enabled by default
(--xcorr-fallback / --no-xcorr-fallback to toggle) since it only does
extra work for sessions that already failed the naive join - it can't
make an already-good alignment worse. A session that STILL comes back
below --min-corr after the lag search is not a timing-offset problem
- most likely a V/S file mis-pairing, a unit/sign issue in one of the
streams, or a session where the vehicle simply wasn't moving for most
of the overlap (std ~ 0, correlation undefined/noisy either way) - and
should be excluded or hand-inspected rather than trusted for training,
per data/processed/README.md's "don't let this go stale silently"
instruction. See alignment_report.json written alongside the output
for the per-session corr/lag actually used, so that decision doesn't
have to be re-derived from stdout scrollback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _build_merge(v: pd.DataFrame, s: pd.DataFrame, lag_s: float) -> tuple[pd.DataFrame, float]:
    """Merge v and s (already zeroed to their own start in time_s) onto
    a shared 100Hz grid, shifting s's clock forward by lag_s seconds
    relative to v before resampling. lag_s > 0 means "s's timestamp t
    corresponds to v's timestamp t - lag_s" (s lags v).
    """
    s_shifted = s.copy()
    s_shifted["time_s"] = s_shifted["time_s"] + lag_s

    t_start = max(v["time_s"].iloc[0], s_shifted["time_s"].iloc[0])
    t_end = min(v["time_s"].iloc[-1], s_shifted["time_s"].iloc[-1])
    if t_end <= t_start:
        return pd.DataFrame(), float("nan")
    grid = np.arange(t_start, t_end, 0.01)

    def resample_to_grid(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = {"time_s": grid}
        for col in df.columns:
            if col in ("time_s", "time_ms", "date"):
                continue
            out[f"{prefix}_{col}"] = np.interp(grid, df["time_s"], df[col])
        return pd.DataFrame(out)

    v_grid = resample_to_grid(v, "v")
    s_grid = resample_to_grid(s_shifted, "s")
    merged = v_grid.merge(s_grid, on="time_s")

    corr = float("nan")
    if "v_velocity_kmh" in merged.columns and "s_gps_speed_kmh" in merged.columns:
        a, b = merged["v_velocity_kmh"], merged["s_gps_speed_kmh"]
        if a.std() > 0 and b.std() > 0:
            corr = float(np.corrcoef(a, b)[0, 1])
    return merged, corr


def align_pair(
    v: pd.DataFrame,
    s: pd.DataFrame,
    min_corr: float,
    xcorr_fallback: bool = True,
    max_lag_s: float = 5.0,
    lag_step_s: float = 0.05,
) -> tuple[pd.DataFrame, float, float]:
    """Zero both streams' time axis to their own start, align on the
    shared 100Hz grid, and - if the naive (zero-lag) alignment scores
    below min_corr and xcorr_fallback is enabled - search a bounded
    lag window for the shift that maximizes GPS-speed/velocity_kmh
    correlation. Returns (merged, corr_achieved, lag_s_used).
    """
    v = v.copy()
    s = s.copy()
    v["time_s"] = v["time_s"] - v["time_s"].iloc[0]
    s_time_col = "time_ms" if "time_ms" in s.columns else "time_s"
    s["time_s"] = (s[s_time_col] - s[s_time_col].iloc[0]) / (1000.0 if s_time_col == "time_ms" else 1.0)

    merged, corr = _build_merge(v, s, 0.0)
    best_lag = 0.0

    if xcorr_fallback and (np.isnan(corr) or corr < min_corr):
        best_corr = corr if not np.isnan(corr) else -np.inf
        best_merged = merged
        for lag in np.arange(-max_lag_s, max_lag_s + 1e-9, lag_step_s):
            if lag == 0.0:
                continue
            m2, c2 = _build_merge(v, s, lag)
            if not np.isnan(c2) and c2 > best_corr:
                best_corr, best_lag, best_merged = c2, float(lag), m2
        if best_corr > (corr if not np.isnan(corr) else -np.inf):
            merged, corr = best_merged, best_corr

    return merged, corr, best_lag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/split_manifest.json")
    ap.add_argument("--resampled-dir", default="data/raw/_resampled")
    ap.add_argument("--out-dir", default="data/raw/_aligned")
    ap.add_argument("--min-corr", type=float, default=0.5,
                     help="Below this GPS-speed/velocity_kmh correlation, flag the "
                          "session as a low-confidence alignment instead of silently keeping it.")
    ap.add_argument("--xcorr-fallback", action=argparse.BooleanOptionalAction, default=True,
                     help="When the naive zero-lag join scores below --min-corr, search a bounded "
                          "lag window for the shift that maximizes correlation instead of accepting "
                          "the naive join as-is. On by default - use --no-xcorr-fallback to reproduce "
                          "the old naive-only behaviour.")
    ap.add_argument("--max-lag", type=float, default=5.0,
                     help="Seconds of lag to search in either direction when --xcorr-fallback triggers.")
    ap.add_argument("--lag-step", type=float, default=0.05,
                     help="Grid step (seconds) for the lag search.")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    resampled_dir = Path(args.resampled_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    still_flagged = []
    recovered = []
    report = {}
    n_ok = 0
    for sid, info in manifest["sessions"].items():
        v_path = resampled_dir / f"{sid}_V.parquet"
        s_path = resampled_dir / f"{sid}_S.parquet"
        if not (v_path.exists() and s_path.exists()):
            continue  # unpaired session, or 01_resample.py failed on it - already reported there
        v = pd.read_parquet(v_path)
        s = pd.read_parquet(s_path)
        try:
            merged, corr, lag_used = align_pair(
                v, s, args.min_corr, args.xcorr_fallback, args.max_lag, args.lag_step
            )
        except Exception as e:  # noqa: BLE001
            print(f"FAILED aligning {sid}: {e}")
            continue
        merged.to_parquet(out_dir / f"{sid}_aligned.parquet")
        n_ok += 1
        report[sid] = {"corr": corr, "lag_s": lag_used}
        if not np.isnan(corr) and corr < args.min_corr:
            still_flagged.append((sid, corr, lag_used))
        elif lag_used != 0.0:
            recovered.append((sid, corr, lag_used))

    (out_dir / "alignment_report.json").write_text(json.dumps(report, indent=2))
    print(f"Aligned {n_ok} synchronised session(s) -> {out_dir}")

    if recovered:
        print(f"\n{len(recovered)} session(s) recovered via xcorr-fallback lag search:")
        for sid, corr, lag in recovered:
            print(f"  {sid}: corr={corr:.2f} at lag={lag:+.2f}s")

    if still_flagged:
        print(
            f"\n{len(still_flagged)} session(s) STILL have low GPS-speed/velocity_kmh "
            f"correlation after the lag search (below --min-corr={args.min_corr}) - this is "
            "no longer a timing-offset problem (the search already tried +/-"
            f"{args.max_lag}s), so it's most likely a mis-paired V/S file, a sign/unit issue "
            "in one of the streams, or a session where the vehicle wasn't moving for most of "
            "the overlap. Exclude these from training or inspect by hand before windowing:"
        )
        for sid, corr, lag in still_flagged:
            print(f"  {sid}: best corr={corr:.2f} (best lag tried: {lag:+.2f}s)")
        print(f"\nFull per-session corr/lag written to {out_dir / 'alignment_report.json'}")


if __name__ == "__main__":
    main()
