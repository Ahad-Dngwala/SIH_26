"""Discover sessions and assign each to train/val/test BY ROUTE.

Why this script exists and isn't just "step 5" as MIP Section 3.2
numbers it: step 4 (normalize) needs to know which windows are TRAIN
before it can compute train-only stats, but step 5 (split) is where
the plan says the split happens. That's a real ordering bug in the
plan, not a style choice - normalization literally cannot be computed
correctly in the order the steps are numbered. Fix: assign the
route-level split here, first, and have every later step (03_window.py,
04_normalize.py) read the assignment from here rather than deciding it
themselves. 05_split.py then becomes a leakage check + report against
this manifest, not the place the split is decided - still a real,
separate deliverable, just not where the split originates. Flagged in
data/processed/README.md too.

Split unit is the SESSION (== route), per 3.2 step 5's explicit
warning: splitting by random window leaks information across
train/val/test because adjacent windows overlap 50%. 70/15/15 by
session count, seeded for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from iovnbd_common import discover_sessions


def assign_splits(session_ids: list[str], seed: int = 0) -> dict[str, str]:
    ids = sorted(session_ids)  # sort first so the shuffle is deterministic regardless of filesystem order
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * 0.70)
    n_val = round(n * 0.15)
    out = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            out[sid] = "train"
        elif i < n_train + n_val:
            out[sid] = "val"
        else:
            out[sid] = "test"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/IO-VNBD")
    ap.add_argument("--out", default="data/processed/split_manifest.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sessions, uncategorised_only = discover_sessions(Path(args.raw_dir))
    if not sessions:
        raise SystemExit(
            f"No sessions found under {args.raw_dir} - has download_iovnbd.py "
            "been run, and did the LFS pull actually complete (see its "
            "docstring re: rate limiting)?"
        )

    sync_sessions = [s for s in sessions if s.sync == "synchronised"]
    unsync_sessions = [s for s in sessions if s.sync == "unsynchronised"]

    # Only synchronised, paired (V+S) sessions get a train/val/test
    # label - they're what Channel A/B/alignment/road-signature train
    # on directly. Unsynchronised is Section 3.4 noise-augmentation
    # material, always attached to whatever split its window ends up
    # feeding (it never defines held-out routes itself).
    paired = [s for s in sync_sessions if s.has_pair]
    unpaired = [s for s in sync_sessions if not s.has_pair]

    split_of = assign_splits([s.session_id for s in paired], seed=args.seed)

    manifest = {
        "seed": args.seed,
        "sessions": {
            s.session_id: {
                "driver_code": s.driver_code,
                "sync": s.sync,
                "v_path": str(s.v_path) if s.v_path else None,
                "s_path": str(s.s_path) if s.s_path else None,
                "split": split_of.get(s.session_id),
            }
            for s in paired
        },
        "unpaired_synchronised_sessions": [s.session_id for s in unpaired],
        "unsynchronised_sessions": [
            {"session_id": s.session_id, "v_path": str(s.v_path)} for s in unsync_sessions
        ],
        "uncategorised_only_sessions_NOT_included": uncategorised_only,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    n_train = sum(1 for v in split_of.values() if v == "train")
    n_val = sum(1 for v in split_of.values() if v == "val")
    n_test = sum(1 for v in split_of.values() if v == "test")
    print(f"Paired synchronised sessions: {len(paired)} (train={n_train}, val={n_val}, test={n_test})")
    print(f"Unpaired synchronised sessions (V or S only, excluded from training): {len(unpaired)}")
    print(f"Unsynchronised (V-only) sessions available as noise-augmentation: {len(unsync_sessions)}")
    if uncategorised_only:
        print(
            f"\n{len(uncategorised_only)} session(s) exist ONLY under 'Uncategorised' "
            f"(e.g. {uncategorised_only[:5]}...) and were NOT included - "
            "see iovnbd_common.py module docstring before deciding to add them."
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
