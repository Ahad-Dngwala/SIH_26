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

VERIFIED against real IO-VNBD sessions (71 synchronised, --max-lag
widened to its current default of 10.0s after the first pass at 5.0s
showed several sessions pinned at that edge): 53/71 pass the naive
zero-lag join outright, the lag search recovers 10 more (corr 0.58 to
0.91), and 6 are still below --min-corr=0.5 after the full +/-10s
search (S2, S3b, S4, Vta03, Vw07, Y1) - since the search window is now
wide and several of these still sit on its edge (see BOUNDARY NOTE),
this reads as a mis-paired V/S file or a sign/unit issue rather than a
timing problem. 2 more (Vw01, Vw15) come back with an undefined (NaN)
correlation - inspect these by hand, don't assume either bucket.

One recovered session, Vta20 (corr=0.58 at lag=+10.00s), is worth a
second look even though it clears --min-corr: its winning lag also
sits exactly on the search boundary, so - per the BOUNDARY NOTE below -
0.58 is likely a floor, not the true achievable correlation. It's not
in the "still flagged" bucket because it does pass the threshold, but
`note` is still "boundary" for it in alignment_report.json for exactly
this reason - don't read a "boundary" note as synonymous with "bad
enough to exclude", the two are independent signals.

PERFORMANCE NOTE (verified - the naive brute-force version of this
search took ~7 min of 100% CPU on a real machine for the ~16 flagged
sessions and was silently doing it with no progress output, which
looked like a hang): the lag search below does NOT rebuild the full
multi-column merged DataFrame for every candidate lag. It resamples
only the two speed channels once (velocity_kmh, gps_speed_kmh) onto a
coarse uniform grid at `--lag-step` spacing, then scores every
candidate lag as a cheap integer-index slice + np.corrcoef on those
two 1-D arrays - no DataFrame construction, no per-column
interpolation, no merge, per candidate. The expensive full-column
_build_merge() now runs exactly once per session, at the winning lag.

BOUNDARY NOTE (also verified - 7/16 fallback sessions landed exactly
on the +/-5.0s edge of the old default --max-lag): landing on the
boundary means the search didn't converge to an interior optimum, so
the true best offset is probably further out. main() now detects this
(abs(lag) >= max_lag - one lag_step) and calls it out by name instead
of quietly reporting a number that likely isn't the true optimum.

NAN NOTE (also verified - real output had 2 sessions with corr=NaN
after the fallback ran, e.g. a constant-value speed channel or too
little overlap at every candidate lag, which the *previous* version
of this script silently wrote to disk without appearing in EITHER the
"recovered" or "still flagged" summary - worse than reporting a low
number, since it looked like nothing was wrong). NaN-correlation
sessions are now their own explicit bucket.
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

    This does the FULL multi-column resample + merge - expensive, and
    should only be called once per session (at lag=0.0 for the naive
    attempt, and again at the winning lag if the fallback search finds
    a better one). The search itself uses `_lag_search` below instead.
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


def _resample_1d(time_s: pd.Series, value: pd.Series, dt: float) -> np.ndarray:
    """Interpolate one column onto a uniform grid from t=0 (both
    streams are already zeroed to their own start before this is
    called) at spacing dt. Coarser than the 100Hz output grid on
    purpose - plenty of resolution to score a lag, far cheaper to
    build over and over during the search.
    """
    t = time_s.to_numpy(dtype=float)
    v = value.to_numpy(dtype=float)
    grid = np.arange(0.0, t[-1], dt)
    return np.interp(grid, t, v)


def _lag_search(
    va: np.ndarray, sa: np.ndarray, dt: float, max_lag_s: float, min_overlap_s: float
) -> tuple[float, float]:
    """Score every integer-sample shift of `sa` against `va` and return
    (best_corr, best_lag_s). No DataFrame, no interpolation inside the
    loop - just index slicing and np.corrcoef on the two arrays already
    built once by the caller. best_corr is -inf if nothing scoreable
    was found (every candidate too short an overlap, or zero variance).
    """
    max_k = int(round(max_lag_s / dt))
    min_overlap_n = max(2, int(round(min_overlap_s / dt)))
    best_corr, best_k = -np.inf, 0
    for k in range(-max_k, max_k + 1):
        if k == 0:
            continue
        if k >= 0:
            i0, i1 = k, min(len(va), len(sa) + k)
        else:
            i0, i1 = 0, min(len(va), len(sa) + k)
        if i1 - i0 < min_overlap_n:
            continue
        v_sub = va[i0:i1]
        s_sub = sa[i0 - k:i1 - k]
        if v_sub.std() == 0 or s_sub.std() == 0:
            continue
        c = float(np.corrcoef(v_sub, s_sub)[0, 1])
        if c > best_corr:
            best_corr, best_k = c, k
    return best_corr, best_k * dt


def align_pair(
    v: pd.DataFrame,
    s: pd.DataFrame,
    min_corr: float,
    xcorr_fallback: bool = True,
    max_lag_s: float = 10.0,
    lag_step_s: float = 0.05,
    min_overlap_s: float = 10.0,
) -> tuple[pd.DataFrame, float, float, str | None]:
    """Zero both streams' time axis to their own start, align on the
    shared 100Hz grid, and - if the naive (zero-lag) alignment scores
    below min_corr and xcorr_fallback is enabled - search a bounded
    lag window (cheaply, see _lag_search) for the shift that maximizes
    GPS-speed/velocity_kmh correlation, then do the one expensive full
    merge at that winning lag.

    Returns (merged, corr_achieved, lag_s_used, note), where note is
    None, "boundary" (best lag found sits on the edge of the searched
    window - the true optimum is probably further out, widen
    --max-lag), or "no-speed-columns" (search couldn't run at all).
    """
    v = v.copy()
    s = s.copy()
    v["time_s"] = v["time_s"] - v["time_s"].iloc[0]
    s_time_col = "time_ms" if "time_ms" in s.columns else "time_s"
    s["time_s"] = (s[s_time_col] - s[s_time_col].iloc[0]) / (1000.0 if s_time_col == "time_ms" else 1.0)

    merged, corr = _build_merge(v, s, 0.0)
    best_lag = 0.0
    note = None

    if xcorr_fallback and (np.isnan(corr) or corr < min_corr):
        if "velocity_kmh" not in v.columns or "gps_speed_kmh" not in s.columns:
            note = "no-speed-columns"
        else:
            va = _resample_1d(v["time_s"], v["velocity_kmh"], lag_step_s)
            sa = _resample_1d(s["time_s"], s["gps_speed_kmh"], lag_step_s)
            search_corr, search_lag = _lag_search(va, sa, lag_step_s, max_lag_s, min_overlap_s)
            if search_corr > (corr if not np.isnan(corr) else -np.inf):
                merged, corr = _build_merge(v, s, search_lag)
                best_lag = search_lag
                if abs(best_lag) >= max_lag_s - lag_step_s / 2:
                    note = "boundary"

    return merged, corr, best_lag, note


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
    ap.add_argument("--max-lag", type=float, default=10.0,
                     help="Seconds of lag to search in either direction when --xcorr-fallback triggers.")
    ap.add_argument("--lag-step", type=float, default=0.05,
                     help="Grid step (seconds) for the lag search.")
    ap.add_argument("--min-overlap", type=float, default=10.0,
                     help="Minimum seconds of overlap required for a candidate lag to be scoreable - "
                          "guards against a tiny, coincidentally-correlated sliver at an extreme lag.")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    resampled_dir = Path(args.resampled_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = [(sid, info) for sid, info in manifest["sessions"].items()]
    still_flagged = []
    recovered = []
    boundary_hit = []
    undefined = []
    report = {}
    n_ok = 0
    n_paired = sum(
        1 for sid, _ in sessions
        if (resampled_dir / f"{sid}_V.parquet").exists() and (resampled_dir / f"{sid}_S.parquet").exists()
    )
    done = 0
    for sid, info in sessions:
        v_path = resampled_dir / f"{sid}_V.parquet"
        s_path = resampled_dir / f"{sid}_S.parquet"
        if not (v_path.exists() and s_path.exists()):
            continue  # unpaired session, or 01_resample.py failed on it - already reported there
        v = pd.read_parquet(v_path)
        s = pd.read_parquet(s_path)
        try:
            merged, corr, lag_used, note = align_pair(
                v, s, args.min_corr, args.xcorr_fallback, args.max_lag, args.lag_step, args.min_overlap
            )
        except Exception as e:  # noqa: BLE001
            print(f"FAILED aligning {sid}: {e}")
            continue
        done += 1
        print(f"[{done}/{n_paired}] {sid}: corr={corr:.2f}" if not np.isnan(corr) else f"[{done}/{n_paired}] {sid}: corr=NaN",
              f"(lag={lag_used:+.2f}s)" if lag_used else "")
        merged.to_parquet(out_dir / f"{sid}_aligned.parquet")
        n_ok += 1
        report[sid] = {"corr": None if np.isnan(corr) else corr, "lag_s": lag_used, "note": note}
        if np.isnan(corr):
            undefined.append(sid)
        elif corr < args.min_corr:
            still_flagged.append((sid, corr, lag_used))
        elif lag_used != 0.0:
            recovered.append((sid, corr, lag_used))
        if note == "boundary":
            boundary_hit.append((sid, corr, lag_used))

    (out_dir / "alignment_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nAligned {n_ok} synchronised session(s) -> {out_dir}")

    if recovered:
        print(f"\n{len(recovered)} session(s) recovered via xcorr-fallback lag search:")
        for sid, corr, lag in recovered:
            print(f"  {sid}: corr={corr:.2f} at lag={lag:+.2f}s")

    if still_flagged:
        print(
            f"\n{len(still_flagged)} session(s) STILL have low GPS-speed/velocity_kmh "
            f"correlation after the lag search (below --min-corr={args.min_corr}) - if not also "
            "listed under the boundary warning below, this is no longer a timing-offset problem, "
            "so it's most likely a mis-paired V/S file, a sign/unit issue in one of the streams, "
            "or a session where the vehicle wasn't moving for most of the overlap. Exclude these "
            "from training or inspect by hand before windowing:"
        )
        for sid, corr, lag in still_flagged:
            print(f"  {sid}: best corr={corr:.2f} (best lag tried: {lag:+.2f}s)")

    if undefined:
        print(
            f"\n{len(undefined)} session(s) have UNDEFINED correlation (NaN) even after the lag "
            "search - every candidate lag either had too little overlap (< --min-overlap) or a "
            "zero-variance speed channel (e.g. GPS stuck / vehicle stationary the whole session). "
            "These were still written to disk but are NOT verified aligned - do not window them "
            "without inspecting by hand:"
        )
        for sid in undefined:
            print(f"  {sid}")

    if boundary_hit:
        print(
            f"\n{len(boundary_hit)} session(s) landed exactly on the +/-{args.max_lag:.1f}s edge of "
            "the search window - the search did not converge to an interior optimum, so this lag "
            "is probably not the true best offset. Re-run with a larger --max-lag for these:"
        )
        for sid, corr, lag in boundary_hit:
            print(f"  {sid}: corr={corr:.2f} at lag={lag:+.2f}s (boundary)")

    if still_flagged or undefined or boundary_hit:
        print(f"\nFull per-session corr/lag/note written to {out_dir / 'alignment_report.json'}")


if __name__ == "__main__":
    main()