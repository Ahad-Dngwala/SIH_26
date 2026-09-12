"""Verify + report the route-level split (Section 3.2 step 5).

The split itself is decided in 00_build_manifest.py, before
normalization can happen (see that script's docstring for why). This
script is the actual step-5 deliverable that remains once that's true:
a leakage check against exactly the failure mode Section 3.2 step 5
warns about ("splitting by random window will leak information and
give fake high accuracy that collapses in the field") - here re-framed
as "no session_id appears in more than one split's meta.json for any
model" - plus a human-readable report to paste into
data/processed/README.md per that file's own instructions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/split_manifest.json")
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    processed_dir = Path(args.processed_dir)

    print("=== Route-level split (from 00_build_manifest.py, seed=%s) ===" % manifest["seed"])
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sid, info in manifest["sessions"].items():
        if info.get("split"):
            by_split[info["split"]].append(sid)
    for split, ids in by_split.items():
        print(f"  {split}: {len(ids)} session(s) - {ids}")

    print("\n=== Leakage check across each implemented model's windowed output ===")
    any_leak = False
    for model_dir in sorted(processed_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        session_sets: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            meta_path = model_dir / split / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            session_sets[split] = set(meta["session_ids"])
        if not session_sets:
            continue
        splits = list(session_sets.keys())
        leak_found = False
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap = session_sets[splits[i]] & session_sets[splits[j]]
                if overlap:
                    leak_found = True
                    any_leak = True
                    print(f"  [{model_dir.name}] LEAK: sessions {overlap} appear in both "
                          f"'{splits[i]}' and '{splits[j]}'")
        if not leak_found:
            print(f"  [{model_dir.name}] OK - no session appears in more than one split.")

    if any_leak:
        raise SystemExit(
            "\nLeakage detected - do not train on this data until fixed. This "
            "would most likely mean 00_build_manifest.py's manifest was edited "
            "by hand after 03_window.py already ran against an older version."
        )

    print(
        "\nCopy the counts above into data/processed/README.md per that file's "
        "own instruction to record what was actually run, with which "
        "norm_stats.json files exist for which models."
    )


if __name__ == "__main__":
    main()
