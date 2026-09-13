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

QUALITY GATE cross-check (added once 03_window.py started skipping
low-confidence-alignment sessions per 02_align.py's real run, 8/71
sessions on the real dataset): the manifest's route-level split counts
now don't match what's actually in each model's meta.json, since a
session the manifest assigned to (say) train can still be entirely
absent from train/windows.npy if 02_align.py flagged it. That's
correct behaviour, not leakage, but reporting only the manifest counts
without saying so would look like a mismatch nobody explained. This
script now cross-references data/raw/_aligned/alignment_report.json
(if present) against the manifest split and reports, per split, how
many assigned sessions were actually excluded by the quality gate and
which ones - so "why does train only have 44 sessions' worth of
windows when the manifest assigned 50" has an answer right here
instead of requiring a re-read of 03_window.py's stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_alignment_quality(aligned_dir: Path) -> dict[str, dict] | None:
    """Same report 02_align.py writes and 03_window.py reads - kept as
    a local copy rather than an import from 03_window.py, since this
    script intentionally has no other dependency on it (see module
    docstring: 05_split.py checks the manifest and meta.json, not the
    windowing script itself).
    """
    report_path = aligned_dir / "alignment_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text())


def _is_excluded(entry: dict | None, min_corr: float) -> bool:
    if entry is None:
        return False  # no report entry at all is a different case, reported separately below
    corr = entry.get("corr")
    return corr is None or corr < min_corr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/split_manifest.json")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--aligned-dir", default="data/raw/_aligned")
    ap.add_argument("--min-corr", type=float, default=0.5,
                     help="Must match whatever --min-corr 03_window.py was run with, so the "
                          "excluded-session cross-check below reflects what was actually skipped.")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    processed_dir = Path(args.processed_dir)
    aligned_dir = Path(args.aligned_dir)

    quality = _load_alignment_quality(aligned_dir)

    print("=== Route-level split (from 00_build_manifest.py, seed=%s) ===" % manifest["seed"])
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sid, info in manifest["sessions"].items():
        if info.get("split"):
            by_split[info["split"]].append(sid)
    for split, ids in by_split.items():
        print(f"  {split}: {len(ids)} session(s) - {ids}")

    if quality is None:
        print(
            "\n(no alignment_report.json found under "
            f"{aligned_dir} - skipping the quality-gate cross-check below; "
            "run 02_align.py first if you expect one to exist)"
        )
    else:
        print(
            "\n=== Sessions the quality gate excluded from windowing, per split "
            f"(--min-corr={args.min_corr}) ==="
        )
        any_excluded = False
        for split, ids in by_split.items():
            excluded = [sid for sid in ids if _is_excluded(quality.get(sid), args.min_corr)]
            no_entry = [sid for sid in ids if sid not in quality]
            if excluded or no_entry:
                any_excluded = True
            if excluded:
                print(f"  {split}: {len(excluded)}/{len(ids)} session(s) excluded - {excluded}")
            if no_entry:
                print(
                    f"  {split}: {len(no_entry)}/{len(ids)} session(s) have no "
                    f"alignment_report.json entry at all (never aligned, or aligned "
                    f"before this report existed) - {no_entry}"
                )
        if not any_excluded:
            print("  none - every manifest-assigned session passed the quality gate.")

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
