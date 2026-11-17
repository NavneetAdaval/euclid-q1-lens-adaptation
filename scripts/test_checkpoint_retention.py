"""
test_checkpoint_retention.py
============================
Task 5, deliverable 1: prove that a run RETAINS a checkpoint at every curve point.

WHAT ACTUALLY WENT WRONG (worth stating, because it changes what needs testing)
-------------------------------------------------------------------------------
``nb1``, ``nb3`` and ``nb0_image_kaggle`` came home with only ``step_8000.pt``, which made
IM/SND/ATC uncomputable on the two runs the A1 verdict rested on. The TRAINER WAS NEVER AT
FAULT: ``run_screen_unfreeze.py`` has always written a checkpoint at every curve point, and
the three LOCAL runs of the same code have all 32 on disk. The loss happened in the Kaggle
COLLECTION cell, which copied only the R1 checkpoint into the downloadable directory.

So retention needs TWO independent checks, and this file covers the first:
  1. HERE / end of training  -- ``assert_checkpoint_retention``: the files exist on the
     training box.
  2. ``package_screen_results.py`` -- the files survived into the downloadable archives.
Only the second would have caught the failure that actually happened; only the first
catches a genuine trainer regression. Both are needed.

    python scripts/test_checkpoint_retention.py
    python scripts/test_checkpoint_retention.py --no-integration   # unit tests only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_screen_unfreeze as rs                               # noqa: E402

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def _mk(tmp, steps, curve_steps=None):
    """A fake run dir with checkpoints at ``steps`` and a curve.csv at ``curve_steps``."""
    ck = os.path.join(tmp, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    for s in steps:
        open(os.path.join(ck, f"step_{s}.pt"), "wb").close()
    curve = os.path.join(tmp, "curve.csv")
    with open(curve, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "fold0_tpr_1e-3"])
        for s in (curve_steps if curve_steps is not None else steps):
            w.writerow([s, 0.1])
    return ck, curve


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------

def test_expected_steps():
    print("\n[1] expected_checkpoint_steps")
    got = rs.expected_checkpoint_steps(8000, 250)
    check(len(got) == 32, f"8000/250 -> 32 steps, got {len(got)}")
    check(got[0] == 250 and got[-1] == 8000,
          f"first 250 / last 8000, got {got[0]} / {got[-1]}")
    check(rs.expected_checkpoint_steps(4, 2) == [2, 4], "small case 4/2 -> [2, 4]")
    check(len(rs.expected_checkpoint_steps(32, 1)) == 32, "32/1 -> 32 steps")


def test_guard_passes_when_complete():
    print("\n[2] guard PASSES on a complete run")
    tmp = tempfile.mkdtemp()
    try:
        ck, curve = _mk(tmp, rs.expected_checkpoint_steps(8000, 250))
        have = rs.assert_checkpoint_retention(ck, curve, 8000, 250, verbose=False)
        check(len(have) == 32, f"returns all 32 steps, got {len(have)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guard_catches_the_real_failure():
    print("\n[3] guard CATCHES the exact nb1/nb3 failure (only step_8000 survives)")
    tmp = tempfile.mkdtemp()
    try:
        # this is literally what came home from Kaggle: one checkpoint, 32 curve rows
        ck, curve = _mk(tmp, [8000], curve_steps=rs.expected_checkpoint_steps(8000, 250))
        try:
            rs.assert_checkpoint_retention(ck, curve, 8000, 250, verbose=False)
        except SystemExit as e:
            msg = str(e)
            check("31 of 32" in msg, f"names the count of missing checkpoints: {msg[:90]}")
            check("nb1/nb3" in msg, "points at the Task-4 precedent so the cause is obvious")
        else:
            check(False, "guard accepted a run with only step_8000 -- the Task-4 failure")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guard_catches_a_single_gap():
    print("\n[4] guard catches ONE missing mid-run checkpoint")
    tmp = tempfile.mkdtemp()
    try:
        want = rs.expected_checkpoint_steps(8000, 250)
        ck, curve = _mk(tmp, [s for s in want if s != 3750], curve_steps=want)
        try:
            rs.assert_checkpoint_retention(ck, curve, 8000, 250, verbose=False)
        except SystemExit as e:
            check("3750" in str(e), "names the missing step 3750")
        else:
            check(False, "guard missed a single-checkpoint gap")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guard_catches_curve_mismatch():
    print("\n[5] guard catches curve.csv / checkpoint-set disagreement")
    tmp = tempfile.mkdtemp()
    try:
        # all checkpoints present, but the curve has FEWER rows -> cannot align a validator
        # series to the TPR series, which is the whole point of retaining them
        want = rs.expected_checkpoint_steps(8000, 250)
        ck, curve = _mk(tmp, want, curve_steps=want[:20])
        try:
            rs.assert_checkpoint_retention(ck, curve, 8000, 250, verbose=False)
        except SystemExit as e:
            check("20 curve rows" in str(e), f"names the row count: {str(e)[:90]}")
        else:
            check(False, "guard missed a curve/checkpoint count disagreement")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# integration -- a REAL short run must emit one checkpoint per curve point
# ---------------------------------------------------------------------------

def test_short_run_emits_32():
    """A genuine 32-step run at --curve-every 1 must leave 32 checkpoints on disk.

    32 is the number that matters (8000/250 in the real screen), so the test asserts the
    real target rather than a token 2 or 3. It costs ~1-2 min: the frozen n_blocks=0 path
    scores its curve from the feature cache, so only the 32 training steps are real work.
    """
    print("\n[6] INTEGRATION: a real 32-step run emits 32 checkpoints")
    suffix = "_rettest32"
    out_dir = os.path.join(_ROOT, "runs", f"baselineA_fold0_nb0_image{suffix}")
    data = os.path.join(_ROOT, "data")
    if not os.path.isdir(os.path.join(data, "feature_cache")):
        print("  [SKIP] no feature cache")
        return
    shutil.rmtree(out_dir, ignore_errors=True)
    argv = ["--run", "nb0_image", "--fp16", "--diagnostic", "--dir-suffix", suffix,
            "--budget-steps", "32", "--curve-every", "1", "--num-workers", "0",
            "--control-scores", os.path.join(_ROOT, "kaggle_screen_bundle",
                                             "control_scores")]
    rc = rs.main(argv)
    check(rc == 0, f"run exited 0, got {rc}")

    ck = os.path.join(out_dir, "checkpoints")
    files = sorted(f for f in os.listdir(ck) if f.startswith("step_"))
    check(len(files) == 32, f"32 checkpoints on disk, got {len(files)}")
    steps = sorted(int(f[5:-3]) for f in files)
    check(steps == list(range(1, 33)), f"steps are 1..32, got {steps[:5]}...{steps[-3:]}")

    with open(os.path.join(out_dir, "curve.csv"), newline="") as fh:
        n_rows = sum(1 for _ in csv.reader(fh)) - 1
    check(n_rows == 32, f"curve.csv has 32 rows, got {n_rows}")

    cfg = json.load(open(os.path.join(out_dir, "run_config.json")))
    check(cfg.get("n_checkpoints_saved") == 32,
          f"run_config records n_checkpoints_saved=32, got {cfg.get('n_checkpoints_saved')}")
    check(cfg.get("n_checkpoints_expected") == 32,
          f"run_config records n_checkpoints_expected=32, "
          f"got {cfg.get('n_checkpoints_expected')}")

    # every checkpoint must be a real, loadable checkpoint -- not a truncated file
    import evaluate as ev
    for f in (files[0], files[len(files) // 2], files[-1]):
        head, std, _ = ev.load_checkpoint(os.path.join(ck, f), "cpu")
        assert head is not None and std.is_fitted
    check(True, "first / middle / last checkpoints load and carry fitted standardizer stats")

    shutil.rmtree(out_dir, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-integration", action="store_true")
    a = ap.parse_args(argv)

    print("=" * 74)
    print("test_checkpoint_retention.py")
    print("=" * 74)
    test_expected_steps()
    test_guard_passes_when_complete()
    test_guard_catches_the_real_failure()
    test_guard_catches_a_single_gap()
    test_guard_catches_curve_mismatch()
    if not a.no_integration:
        test_short_run_emits_32()
    else:
        print("\n[6] INTEGRATION: skipped (--no-integration)")

    print("\n" + "=" * 74)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED")
        for m in FAILED:
            print("  -", m)
        return 1
    print("ALL RETENTION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
