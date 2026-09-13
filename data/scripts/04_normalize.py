"""Compute norm_stats.json from the TRAIN split only (Section 3.2 step 4).

Does NOT rewrite windows.npy in normalized form - every model's
dataset.py applies models.common.normalization.apply() at
__getitem__ time (see their TODO comments), so data/processed/ stays
raw and norm_stats.json is the only normalization artifact. This
keeps a single source of truth for the stats that also gets baked into
on-device preprocessing later (Section 7.1), rather than having two
copies (raw + pre-normalized) drift apart.

Which split is TRAIN comes from 00_build_manifest.py's route-level
assignment, not decided here - see that script's docstring for why
step 4 (this one) can't come before the split exists, despite Section
3.2's numbering implying otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `models.common`
from models.common.normalization import compute_stats, save_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()

    train_windows_path = Path(args.processed_dir) / args.model / "train" / "windows.npy"
    if not train_windows_path.exists():
        raise SystemExit(
            f"{train_windows_path} doesn't exist - run 03_window.py --model {args.model} first."
        )

    train_windows = np.load(train_windows_path)
    stats = compute_stats(train_windows)
    out_path = Path(args.processed_dir) / args.model / "norm_stats.json"
    save_stats(stats, out_path)

    means = np.array(stats["mean"])
    stds = np.array(stats["std"])
    print(f"Wrote {out_path}")
    print(f"  mean: {np.round(means, 4).tolist()}")
    print(f"  std:  {np.round(stds, 4).tolist()}")
    n_floored = int(np.sum(stds <= 1e-6 + 1e-12))
    if n_floored:
        print(
            f"  WARNING: {n_floored} channel(s) hit the 1e-6 std floor in "
            "models/common/normalization.py - that's a near-constant channel "
            "in this training data (e.g. a stationary axis), worth checking "
            "it's expected before training, per that file's own TODO about "
            "the floor value not being a tuned choice."
        )


if __name__ == "__main__":
    main()
