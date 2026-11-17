r"""
build_round1_comparisons.py -- the round-1 paired comparisons, as JSON artifacts.
=================================================================================

    python paper/scripts/build_round1_comparisons.py          # ~5-8 min per comparison
    python paper/scripts/build_round1_comparisons.py --force   # recompute existing

WHY THIS IS A SEPARATE STEP. Round 1's paired comparisons against Baseline A exist in
the repository only as **console logs** (``runs/bakeoff_*_compare.log``). A number in a
table should be read from a machine-readable artifact, not scraped out of prose, so this
script re-runs the project's own ``scripts/compare_methods.py`` over the saved score
dumps and files each result as JSON under ``paper/data/``.

**It re-implements nothing and it re-estimates nothing.** It shells out to
``compare_methods.py`` at its own defaults -- class-conditional paired bootstrap,
$B = 1000$, seed 0, the shared negative pool resampled once per replicate and reused
across folds -- which is the estimator round 1 published under. The regenerated CORAL
comparison reproduces the logged $+0.016$ $[-0.006, +0.043]$ exactly.

It reads saved score vectors. It trains nothing and touches no GPU. It is slow (a
1000-replicate bootstrap over 489 positives and 26,399 shared negatives, five folds,
six metrics) which is the only reason it is not folded into ``make_t3_round1.py``.

Writes only into ``paper/data/``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import _common as C

DATA = C.DATA

# (tag, method run dir, display name) -- all against runs/bakeoff_baseline_A
METHODS = [
    ("coral", "bakeoff_coral", "CORAL"),
    ("adamatch", "bakeoff_adamatch_5fold", "AdaMatch"),
    ("freematch", "bakeoff_freematch_5fold", "FreeMatch"),
]


def out_path(tag: str) -> str:
    return os.path.join(DATA, f"round1_paired_{tag}.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the JSON already exists")
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    base = C.R("runs", "bakeoff_baseline_A")

    for tag, run, name in METHODS:
        dest = out_path(tag)
        if os.path.exists(dest) and not a.force:
            print(f"  have {os.path.relpath(dest, C.REPO)} (use --force to recompute)")
            continue
        tmp = os.path.join(DATA, f"_cmp_{tag}")
        cmd = [sys.executable, "-u", os.path.join(C.REPO_SCRIPTS, "compare_methods.py"),
               base, C.R("runs", run),
               "--name-a", "Baseline A", "--name-b", name, "--out-dir", tmp]
        print("  running:", " ".join(cmd[2:]))
        subprocess.run(cmd, check=True, cwd=C.REPO,
                       stdout=subprocess.DEVNULL, stderr=None)
        shutil.move(os.path.join(tmp, "comparison.json"), dest)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  wrote {os.path.relpath(dest, C.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
