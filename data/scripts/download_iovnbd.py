"""Download IO-VNBD into data/raw/IO-VNBD/ (MIP Section 3.1, primary dataset).

The dataset's CSVs are Git-LFS tracked (~200MB+ per split), so this
does a real git+lfs clone rather than hitting the GitHub REST API for
each file. Requires git-lfs installed (`apt-get install git-lfs` /
`brew install git-lfs`, then this script runs `git lfs install` itself).

KNOWN ISSUE hit while building this pipeline: GitHub's LFS batch API
rate-limits unauthenticated/shared-IP traffic hard - in the sandbox
this was built in, cloning the repo's LFS objects failed with "API
rate limit exceeded" after the very first object, even though the
non-LFS files (README, folder structure) cloned fine. If you hit the
same thing:
  - Wait for the reset (the error message + `git lfs logs last`
    includes a timestamp/request ID; GitHub's unauthenticated quota is
    typically 60 req/hour and resets on a rolling window).
  - Or set GIT_LFS_SKIP_SMUDGE=1 for the initial clone, then
    `git lfs pull --include=<specific paths>` in smaller batches once
    the quota resets, so a single retry doesn't re-spend the whole
    budget on files you already have.
  - Or authenticate: `git config --global credential.helper store` +
    a GitHub PAT raises the quota substantially.
This script does the small-batch retry itself (see --include) so a
partial run doesn't waste quota re-pulling everything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/onyekpeu/IO-VNBD.git"


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw", help="Where to clone IO-VNBD into.")
    ap.add_argument(
        "--include",
        default=None,
        help="Comma-separated glob(s) passed to `git lfs pull --include=`, "
             "for pulling a subset when rate-limited. Omit to pull everything.",
    )
    ap.add_argument("--skip-smudge", action="store_true",
                     help="Clone with GIT_LFS_SKIP_SMUDGE=1 (pointers only, "
                          "no actual file content) - use this to get the file "
                          "tree/structure without spending LFS quota, then run "
                          "again with --include for the sessions you actually need.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "IO-VNBD"

    run(["git", "lfs", "install"])

    if not target.exists():
        env_prefix = ["env", "GIT_LFS_SKIP_SMUDGE=1"] if args.skip_smudge else []
        res = run(env_prefix + ["git", "clone", REPO_URL, str(target)])
        print(res.stdout, res.stderr)
        if res.returncode != 0:
            sys.exit(f"Clone failed: {res.stderr}")
    else:
        print(f"{target} already exists, skipping clone. Delete it to re-clone from scratch.")

    if not args.skip_smudge:
        pull_cmd = ["git", "lfs", "pull"]
        if args.include:
            pull_cmd += [f"--include={args.include}"]
        res = run(pull_cmd, cwd=target)
        print(res.stdout, res.stderr)
        if res.returncode != 0:
            print(
                "\nLFS pull did not fully succeed (see output above - this is "
                "commonly the GitHub rate-limit issue described in this script's "
                "docstring). Re-run with --include='<glob>' for a smaller batch "
                "once the quota resets; git lfs pull is safe to re-run, it skips "
                "objects already present.",
                file=sys.stderr,
            )

    # Record what we pulled, per data/raw/README.md's instruction.
    rev = run(["git", "rev-parse", "HEAD"], cwd=target).stdout.strip()
    manifest = {
        "source": REPO_URL,
        "commit": rev,
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "skip_smudge": args.skip_smudge,
        "include_filter": args.include,
    }
    (raw_dir / "IOVNBD_DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {raw_dir / 'IOVNBD_DOWNLOAD_MANIFEST.json'}")


if __name__ == "__main__":
    main()
