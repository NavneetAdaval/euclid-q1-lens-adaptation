"""
report_bakeoff.py
=================
POST-HOC FULL-METRIC-SUITE REPORT for a bake-off run (SPEC bakeoff_1 4/5).

evaluate.py reports TPR@1e-3 + AUROC only. PROJECT.md 2 defines a wider reportable
family (TPR@1e-3 HEADLINE, AUPRC, purity@50%-completeness raw AND corrected to
N_N/N_P=1:2200, completeness@24%-purity, AUROC as a SECONDARY sanity metric). Rather
than modify the evaluation path, this script recomputes the whole family POST-HOC from
the score vectors evaluate.py already dumped to ``<run>/scores/*.npz``.

NO NEW MATH. Every metric comes from ``metrics.metric_suite``; every CI comes from
``metrics.paired_bootstrap`` (class-conditional, positives and negatives resampled
independently with replacement). This module owns only plumbing: which vectors to read,
which rows to pool, how to print.

WHY POOLING ACROSS FOLDS IS LEGITIMATE HERE (PROJECT.md 4.4)
------------------------------------------------------------
Baseline A is SOURCE-ONLY, so a single trained model scores every fold, and the ~26,399
test negatives are SHARED across folds by construction. The FPR=1e-3 threshold therefore
comes from one identical negative-score vector for all five folds, which is exactly what
makes the pooled 489-positive aggregate a valid ROC rather than a mixture of five
differently-thresholded ones. This script ASSERTS that identity (``--check-shared-negs``,
on by default) and refuses to pool if the negative vectors ever differ -- if a future
per-fold-model method (CORAL onward) breaks the assumption, the assert fires instead of
silently reporting a meaningless pooled number.

USAGE
-----
    python scripts/report_bakeoff.py runs/bakeoff_baseline_A
    python scripts/report_bakeoff.py runs/bakeoff_baseline_A --bootstrap 2000
    python scripts/report_bakeoff.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import (metric_suite, paired_bootstrap,           # noqa: E402
                     DEFAULT_TARGET_FPR, DEFAULT_TARGET_COMPLETENESS,
                     DEFAULT_TARGET_PURITY, NN_NP_Q1)

# The reportable family, in report order. (key, label, note)
# 'tpr' is the HEADLINE; 'auroc' is SECONDARY-only and never a selector (PROJECT.md 2).
METRICS = [
    ("tpr",            f"TPR@FPR={DEFAULT_TARGET_FPR:g}",  "HEADLINE"),
    ("auprc",          "AUPRC",                            ""),
    ("purity_raw",     "purity@50%-comp (raw pool)",       ""),
    ("purity_corr",    f"purity@50%-comp (1:{int(NN_NP_Q1)})", "Lines-comparable"),
    ("comp_at_purity", f"completeness@{DEFAULT_TARGET_PURITY:g}-purity", ""),
    ("auroc",          "AUROC",                            "SECONDARY / sanity"),
]


def _metric_fn(key, target_fpr):
    """A ``metric_fn(y, s) -> float`` for paired_bootstrap that pulls one field out of
    metric_suite. One suite call per replicate per metric -- slightly redundant, but it
    keeps metrics.paired_bootstrap as the ONLY resampler (no bootstrap reimplemented)."""
    def _fn(y, s):
        return metric_suite(y, s, target_fpr=target_fpr)[key]
    return _fn


# ---------------------------------------------------------------------------
# reading the dumped score vectors
# ---------------------------------------------------------------------------

def load_fold_scores(run_dir: str) -> dict:
    """{fold: {pos_scores, neg_scores, pos_ids, neg_ids}} from <run_dir>/scores/*.npz."""
    sdir = os.path.join(run_dir, "scores")
    if not os.path.isdir(sdir):
        raise SystemExit(f"[report] ERROR: no scores/ dir in {run_dir} "
                         f"(evaluate.py writes it; run the evaluation first).")
    out = {}
    for p in sorted(glob.glob(os.path.join(sdir, "*.npz"))):
        m = re.search(r"fold(\d+)\.npz$", os.path.basename(p))
        if not m:
            continue
        with np.load(p, allow_pickle=False) as d:
            if "pos_scores" not in d.files or "neg_scores" not in d.files:
                continue
            out[int(m.group(1))] = {
                "pos_scores": np.asarray(d["pos_scores"], float),
                "neg_scores": np.asarray(d["neg_scores"], float),
                "pos_ids": d["pos_ids"] if "pos_ids" in d.files else None,
                "neg_ids": d["neg_ids"] if "neg_ids" in d.files else None,
                "path": p,
            }
    if not out:
        raise SystemExit(f"[report] ERROR: no *_fold{{k}}.npz score vectors under {sdir}")
    return dict(sorted(out.items()))


def _ys(pos_scores, neg_scores):
    y = np.concatenate([np.ones(pos_scores.size, np.int64),
                        np.zeros(neg_scores.size, np.int64)])
    s = np.concatenate([pos_scores, neg_scores])
    return y, s


def shared_negatives_identical(folds: dict) -> bool:
    """True iff every fold was scored against the byte-identical shared negative vector.

    This is the precondition for single-ROC pooling: it holds for a SINGLE-model method
    (Baseline A -- one model scores the shared pool once, so one FPR=1e-3 threshold), and
    it FAILS for a PER-FOLD-MODEL method (CORAL/AdaMatch/FreeMatch -- five models produce
    five different negative-score vectors, hence five thresholds), where pooling is invalid
    (PROJECT.md 4.4). Non-raising -- callers decide what to do."""
    ks = list(folds)
    ref = folds[ks[0]]["neg_scores"]
    for k in ks[1:]:
        cur = folds[k]["neg_scores"]
        if cur.shape != ref.shape or not np.array_equal(cur, ref):
            return False
    return True


def check_shared_negatives(folds: dict) -> bool:
    """Raising wrapper around :func:`shared_negatives_identical` (kept for the self-test /
    any caller that wants pooling to be a hard precondition)."""
    if not shared_negatives_identical(folds):
        raise SystemExit(
            "[report] ERROR: fold negative scores differ across folds. The pooled aggregate "
            "assumes ONE model scoring the SHARED negative pool (PROJECT.md 4.4); with "
            "per-fold models pooling is invalid.")
    return True


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def compute(folds: dict, *, target_fpr=DEFAULT_TARGET_FPR, B=1000, seed=0,
            pool=True, verbose=True) -> dict:
    """Per-fold + pooled metric suite with paired-bootstrap CIs on every metric."""
    res = {"target_fpr": target_fpr, "bootstrap_B": B, "bootstrap_seed": seed,
           "nn_np_ratio": float(NN_NP_Q1),
           "target_completeness": float(DEFAULT_TARGET_COMPLETENESS),
           "target_purity": float(DEFAULT_TARGET_PURITY),
           "per_fold": {}, "across_fold": {}, "pooled": None}

    # ---- per fold ----
    for k, d in folds.items():
        y, s = _ys(d["pos_scores"], d["neg_scores"])
        suite = metric_suite(y, s, target_fpr=target_fpr)
        row = {"n_pos": int(d["pos_scores"].size), "n_neg": int(d["neg_scores"].size)}
        for key, _lbl, _n in METRICS:
            row[key] = _metric_cell(y, s, key, suite, target_fpr, B, seed)
        res["per_fold"][int(k)] = row
        if verbose:
            print(f"  [fold {k}] done ({row['n_pos']} pos vs {row['n_neg']} neg)", flush=True)

    # ---- across-fold mean +- std (the 5 folds as 5 observations) ----
    for key, _lbl, _n in METRICS:
        vals = np.array([res["per_fold"][k][key]["value"] for k in res["per_fold"]],
                        dtype=float)
        n_def = int(np.isfinite(vals).sum())
        with np.errstate(invalid="ignore"):          # nanmean of all-NaN -> NaN (n_def==0)
            mean = _num(np.nanmean(vals)) if n_def else None
            std = _num(np.nanstd(vals)) if n_def else None
        res["across_fold"][key] = {"mean": mean, "std": std, "n": n_def}

    # ---- pooled aggregate: all 489 positives vs the shared negatives ----
    if pool:
        ks = list(folds)
        ppos = np.concatenate([folds[k]["pos_scores"] for k in ks])
        neg = folds[ks[0]]["neg_scores"]          # identical across folds (asserted)
        y, s = _ys(ppos, neg)
        suite = metric_suite(y, s, target_fpr=target_fpr)
        pooled = {"n_pos": int(ppos.size), "n_neg": int(neg.size)}
        for key, _lbl, _n in METRICS:
            pooled[key] = _metric_cell(y, s, key, suite, target_fpr, B, seed)
        res["pooled"] = pooled
        if verbose:
            print(f"  [pooled] done ({pooled['n_pos']} pos vs {pooled['n_neg']} neg)",
                  flush=True)
    return res


def _num(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _metric_cell(y, s, key, suite, target_fpr, B, seed) -> dict:
    """One metric's {value, ci, median} cell, with the undefined-metric rule enforced.

    If the point value is undefined on this pool (e.g. completeness@target-purity when the
    model never reaches that purity), report value=None and NO CI. A paired_bootstrap CI in
    that case is conditional on the replicates where the metric happened to be reachable --
    a reachability-conditional interval that over-states a metric that is genuinely n/a
    (SPEC spec_bakeoff_2_coral.md §6: "n/a and NO CI ... never a bootstrap CI conditional on
    reachability"). So the CI is computed ONLY when the point value is defined."""
    val = _num(suite[key])
    if val is None:
        return {"value": None, "ci": [None, None], "median": None}
    ci = paired_bootstrap(y, s, metric_fn=_metric_fn(key, target_fpr), B=B, seed=seed)
    return {"value": val, "ci": [_num(ci["lo"]), _num(ci["hi"])],
            "median": _num(ci.get("median"))}


def _f(x, nd=4):
    return "  n/a " if x is None else f"{x:.{nd}f}"


def _val_ci(c) -> str:
    """Format 'value   [lo, hi]', or 'n/a   (undefined -- no CI)' when the metric is n/a."""
    if c["value"] is None:
        return f"{'n/a':>9}   {'(undefined -- no CI)':>20}"
    lo, hi = c["ci"]
    return f"{_f(c['value']):>9}   [{_f(lo)}, {_f(hi)}]"


def print_report(res: dict, title="BAKE-OFF FULL METRIC SUITE"):
    folds = sorted(res["per_fold"])
    print("\n" + "=" * 100)
    print(title)
    print(f"(paired class-conditional bootstrap, B={res['bootstrap_B']}, "
          f"seed={res['bootstrap_seed']}; purity corrected to 1:{int(res['nn_np_ratio'])})")
    print("=" * 100)

    for key, label, note in METRICS:
        suffix = f"   [{note}]" if note else ""
        print(f"\n{label}{suffix}")
        print(f"  {'fold':<8}{'value':>9}   {'95% CI':>20}")
        print("  " + "-" * 40)
        for k in folds:
            print(f"  {k:<8}{_val_ci(res['per_fold'][k][key])}")
        af = res["across_fold"][key]
        n_def, n_tot = af["n"], len(folds)
        if n_def == 0:
            print(f"  {'mean+-sd':<8}{'n/a':>9}   (undefined on all {n_tot} folds)")
        elif n_def < n_tot:
            # partial-definition metric (e.g. completeness): show the mean over ONLY the
            # defined folds, explicitly flagged -- never presented as a full-fold aggregate.
            print(f"  {'mean+-sd':<8}{_f(af['mean']):>9}   +- {_f(af['std'])}"
                  f"   (over {n_def}/{n_tot} DEFINED folds only -- not a {n_tot}-fold aggregate)")
        else:
            print(f"  {'mean+-sd':<8}{_f(af['mean']):>9}   +- {_f(af['std'])}"
                  f"   (across {n_def} folds)")
        if res["pooled"]:
            print(f"  {'POOLED':<8}{_val_ci(res['pooled'][key])}"
                  f"   <- {res['pooled']['n_pos']} pos vs shared negs")


# ---------------------------------------------------------------------------
# self-test (mock + adversarial)
# ---------------------------------------------------------------------------

def self_test():
    print("SELF-TEST (mock + adversarial) ...")
    rng = np.random.default_rng(7)
    neg = rng.normal(0.0, 1.0, 3000)
    folds = {k: {"pos_scores": rng.normal(2.0, 1.0, 40), "neg_scores": neg.copy(),
                 "pos_ids": None, "neg_ids": None, "path": f"<mock{k}>"} for k in range(5)}

    # 1. shared-negative check passes on identical vectors ...
    assert check_shared_negatives(folds)
    print("  [PASS] shared-negative identity check accepts identical negative vectors.")

    # ... and FIRES when one fold's negatives are perturbed (the adversarial case).
    bad = {k: dict(v) for k, v in folds.items()}
    bad[3]["neg_scores"] = neg + 1e-9
    try:
        check_shared_negatives(bad)
        raise AssertionError("shared-negative check did NOT fire on differing negatives")
    except SystemExit:
        print("  [PASS] shared-negative check FIRES on a perturbed fold (pooling blocked).")

    # 2. metric values match a direct metrics.py call (no drift in the plumbing).
    res = compute(folds, B=50, seed=0, verbose=False)
    y, s = _ys(folds[0]["pos_scores"], folds[0]["neg_scores"])
    ref = metric_suite(y, s)
    for key, _l, _n in METRICS:
        got = res["per_fold"][0][key]["value"]
        exp = _num(ref[key])
        assert (got is None and exp is None) or abs(got - exp) < 1e-12, f"{key}: {got} vs {exp}"
    print("  [PASS] every reported per-fold metric equals a direct metric_suite() call.")

    # 2b. undefined metric => value None AND ci [None, None] (no reachability-conditional
    # CI). Build a fold whose positives all rank BELOW the negatives: purity never reaches
    # the target, so completeness@target-purity is undefined.
    und = dict(folds[0])
    und["pos_scores"] = rng.normal(-5.0, 0.5, 40)   # positives buried under the negatives
    ru = compute({0: und}, B=50, seed=0, pool=False, verbose=False)
    cc = ru["per_fold"][0]["comp_at_purity"]
    assert cc["value"] is None and cc["ci"] == [None, None], \
        f"undefined completeness must carry NO CI, got {cc}"
    print("  [PASS] an undefined metric reports value=None with ci=[None,None] (no CI).")

    # 3. pooled positives = sum of fold positives; negatives NOT duplicated.
    assert res["pooled"]["n_pos"] == 200, res["pooled"]["n_pos"]
    assert res["pooled"]["n_neg"] == 3000, res["pooled"]["n_neg"]
    print("  [PASS] pooling concatenates positives (200) and uses shared negs ONCE (3000).")

    # 4. CIs bracket the point estimate (sanity on the bootstrap wiring).
    for key, _l, _n in METRICS:
        c = res["pooled"][key]
        if c["value"] is None:
            continue
        lo, hi = c["ci"]
        assert lo is not None and lo <= c["value"] + 1e-9 and hi + 1e-9 >= c["value"], \
            f"{key}: {lo} !<= {c['value']} !<= {hi}"
    print("  [PASS] every pooled CI brackets its point estimate.")

    # 5. corrected purity <= raw purity (1:2200 is rarer than this mock pool).
    pr = res["pooled"]["purity_raw"]["value"]
    pc = res["pooled"]["purity_corr"]["value"]
    assert pc <= pr + 1e-12, f"corrected {pc} > raw {pr}"
    print(f"  [PASS] corrected purity ({pc:.4f}) <= raw purity ({pr:.4f}).")

    print("SELF-TEST: ALL PASSED")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="run dir containing scores/*.npz")
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="write JSON here (default: <run_dir>/metric_suite.json)")
    ap.add_argument("--force-pool", action="store_true",
                    help="force the pooled row even when negatives differ across folds "
                         "(single-ROC pooling is invalid for per-fold models -- debug only)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not args.run_dir:
        ap.error("provide a run dir, or --self-test")

    run_dir = os.path.abspath(args.run_dir)
    folds = load_fold_scores(run_dir)
    print(f"[report] {run_dir}: folds {list(folds)} "
          f"({', '.join(str(folds[k]['pos_scores'].size) for k in folds)} positives)")

    # POOLING RULE (auto-detected). Single-ROC pooling is valid ONLY when every fold shares
    # the identical negative-score vector, i.e. a SINGLE-model method (Baseline A). For a
    # PER-FOLD-MODEL method (CORAL/AdaMatch/FreeMatch) the negatives are scored by different
    # models, so there is no common FPR=1e-3 threshold and pooled = n/a -- report per-fold +
    # fold-mean only. This is the confirmed bake-off standard; no flag needed.
    identical = shared_negatives_identical(folds)
    pool = identical or args.force_pool
    if identical:
        print("[report] shared-negative identity: OK (single-model regime; pooling valid).")
    elif args.force_pool:
        print("[report] WARNING: --force-pool set but negatives DIFFER across folds; the "
              "pooled row is single-ROC-invalid for per-fold models. Debug only.")
    else:
        print("[report] negatives DIFFER across folds => PER-FOLD-MODEL method: single-ROC "
              "pooling is invalid, so POOLED = n/a. Reporting per-fold + fold-mean only "
              "(confirmed pooling standard; PROJECT.md 4.4).")

    res = compute(folds, target_fpr=args.target_fpr, B=args.bootstrap, seed=args.seed,
                  pool=pool)
    res["run_dir"] = run_dir
    print_report(res)

    out = args.out or os.path.join(run_dir, "metric_suite.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n[out] full metric suite -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
