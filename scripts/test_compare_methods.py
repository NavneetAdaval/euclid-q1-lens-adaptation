"""
test_compare_methods.py
=======================
Tests for compare_methods.py (the paired method-vs-method significance test). All
synthetic -- it fabricates two tiny "run" directories of dumped scores and checks that
the paired test reaches the RIGHT verdict on cases with a known ground truth, across the
FULL metric suite (TPR@1e-3, AUPRC, purity@50%-comp raw + corrected-to-1:2200,
completeness@purity, AUROC):

  1. B == A exactly            -> every metric's mean difference CI must BRACKET 0,
                                  verdict "not distinguishable".
  2. B strictly better / fold  -> mean TPR / AUPRC / purity differences POSITIVE and
                                  their CIs EXCLUDE 0 (distinguishable improvement).
  3. neg_ids misaligned        -> the alignment guard FIRES (comparison raises), rather
                                  than silently comparing mismatched rows.

Run:
    python scripts\\test_compare_methods.py
Exit 0 on success, 1 on a failed check.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

import compare_methods as cm

_fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


# ---------------------------------------------------------------------------
# synthetic dumped-score fixtures
# ---------------------------------------------------------------------------

def _dump_fold(run_dir, tag, k, pos_scores, neg_scores, neg_ids, target_fpr=1e-3):
    sdir = os.path.join(run_dir, "scores")
    os.makedirs(sdir, exist_ok=True)
    np.savez(os.path.join(sdir, f"{tag}_fold{k}.npz"),
             pos_ids=np.array([f"pos{k}_{i}" for i in range(len(pos_scores))], dtype="<U20"),
             pos_scores=np.asarray(pos_scores, np.float32),
             neg_ids=np.asarray(neg_ids, dtype="<U40"),
             neg_scores=np.asarray(neg_scores, np.float32),
             target_fpr=np.asarray(target_fpr))


def _make_runs(root, folds, n_pos, n_neg, seed, b_shift):
    """Two run dirs A and B over shared negatives (identical neg_ids per fold).

    A: positives ~ N(2,1); B: positives = A + b_shift (row-aligned paired shift).
    Negatives are shared and IDENTICAL rows between A and B (each method would score
    them differently in reality, but for a controlled test we give B the same negative
    scores so only the positive shift drives the measured difference)."""
    rng = np.random.default_rng(seed)
    a_dir = os.path.join(root, "A"); b_dir = os.path.join(root, "B")
    for k in folds:
        neg_ids = np.array([f"neg_{k}_{i}" for i in range(n_neg)], dtype="<U40")
        neg = rng.normal(0.0, 1.0, n_neg)
        pos_a = rng.normal(2.0, 1.0, n_pos)
        pos_b = pos_a + b_shift                        # strictly higher when b_shift>0
        _dump_fold(a_dir, "baseline_linear_ep100", k, pos_a, neg, neg_ids)
        _dump_fold(b_dir, "coral", k, pos_b, neg, neg_ids)
    return a_dir, b_dir


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_equal():
    print("\n[1] B == A exactly  -> difference CI brackets 0, 'not distinguishable'")
    root = tempfile.mkdtemp(prefix="cmp_equal_")
    try:
        a_dir, b_dir = _make_runs(root, folds=range(5), n_pos=100, n_neg=5000,
                                  seed=1, b_shift=0.0)
        r = cm.compare(a_dir, b_dir, B=500, seed=0, out_dir=None, verbose=False)
        ag = r["aggregate"]
        # B==A on every row -> every metric's difference is exactly 0 and NOT distinguishable
        for mk in cm.SUITE_KEYS:
            m = ag[mk]
            check(m["diff"] is not None and abs(m["diff"]) < 1e-9,
                  f"{mk}: mean diff is exactly 0 ({m['diff']})")
            lo, hi = m["ci"]
            check((lo is None or lo <= 0.0) and (hi is None or hi >= 0.0),
                  f"{mk}: diff CI brackets 0 ([{lo}, {hi}])")
            check(not m["distinguishable"], f"{mk}: verdict NOT distinguishable")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_better():
    print("\n[2] B strictly better every fold  -> positive diff, CI excludes 0")
    root = tempfile.mkdtemp(prefix="cmp_better_")
    try:
        a_dir, b_dir = _make_runs(root, folds=range(5), n_pos=100, n_neg=5000,
                                  seed=2, b_shift=1.5)
        r = cm.compare(a_dir, b_dir, B=1000, seed=0, out_dir=None, verbose=False)
        ag = r["aggregate"]
        # every fold's positive shift is +1.5 -> B recovers strictly more at fixed FPR
        per_fold_pos = all(r["per_fold"][k]["metrics"]["tpr"]["diff"] > 0 for k in r["folds"])
        check(per_fold_pos, "every fold has a positive TPR difference")
        # HEADLINE metrics: TPR, corrected purity, AUPRC all improve and are distinguishable
        for mk in ("tpr", "purity_corr", "auprc"):
            m = ag[mk]
            check(m["diff"] > 0, f"aggregate {mk} diff positive ({m['diff']:.4f})")
            lo, hi = m["ci"]
            check(lo is not None and lo > 0.0,
                  f"{mk} diff CI excludes 0 from below ([{lo:.4f}, {hi:.4f}])")
            check(m["distinguishable"], f"{mk} verdict: DISTINGUISHABLE improvement")
        # raw-pool purity also improves; AUROC (secondary) improves too
        check(ag["purity_raw"]["diff"] > 0, f"raw purity diff positive ({ag['purity_raw']['diff']:.4f})")
        check(ag["auroc"]["diff"] > 0 and ag["auroc"]["distinguishable"],
              f"AUROC (secondary) diff positive and distinguishable ({ag['auroc']['diff']:.4f})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_alignment_guard():
    print("\n[3] misaligned neg_ids  -> alignment guard fires (raises)")
    root = tempfile.mkdtemp(prefix="cmp_misalign_")
    try:
        a_dir = os.path.join(root, "A"); b_dir = os.path.join(root, "B")
        rng = np.random.default_rng(3)
        neg = rng.normal(0, 1, 500)
        pos = rng.normal(2, 1, 50)
        _dump_fold(a_dir, "baseline_linear_ep100", 0, pos, neg,
                   [f"neg_{i}" for i in range(500)])
        # SAME shape, DIFFERENT ids -> must be caught (not silently compared)
        _dump_fold(b_dir, "coral", 0, pos, neg,
                   [f"OTHER_{i}" for i in range(500)])
        raised = False
        try:
            cm.compare(a_dir, b_dir, B=100, seed=0, verbose=False)
        except AssertionError as e:
            raised = True
            print(f"    guard message: {str(e).splitlines()[0][:80]} ...")
        check(raised, "misaligned neg_ids raise AssertionError (fail loud, not silent)")

        # sanity: also fires on a length mismatch
        _dump_fold(b_dir, "coral", 0, pos, rng.normal(0, 1, 400),
                   [f"neg_{i}" for i in range(400)])
        raised_len = False
        try:
            cm.compare(a_dir, b_dir, B=100, seed=0, verbose=False)
        except AssertionError:
            raised_len = True
        check(raised_len, "mismatched negative COUNT also raises")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    print("=" * 72)
    print("compare_methods paired-test unit tests (synthetic)")
    print("=" * 72)
    test_equal()
    test_better()
    test_alignment_guard()

    print("\n" + "=" * 72)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL COMPARE-METHODS CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
