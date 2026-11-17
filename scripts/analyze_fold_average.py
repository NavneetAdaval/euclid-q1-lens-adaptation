"""
analyze_fold_average.py
=======================
ANALYSIS ONLY on existing artifacts. Does FOLD-AVERAGING resolve a peak that the individual
curves could not locate?

Averaging is the operation that suppresses per-checkpoint noise, so this is the right last
check before committing quota. It is run on folds 0 and 1 -- the only two UNFROZEN runs --
over their common range 0..3375 at 125-step spacing (28 points). Fold 3 is the frozen spot
check and is excluded: it has no unfreezing trajectory to average.

THE DECISIVE STATISTIC IS THE NOISE REDUCTION, NOT THE ARGMAX.
Averaging two independent noisy curves reduces sd by exactly sqrt(2). If the folds share a
real shape, their per-step deviations are POSITIVELY CORRELATED and the average retains
more than that -- sd falls by LESS than sqrt(2). If they are independent noise around
their own means, the reduction is exactly sqrt(2) and there is no common shape to find.
So:

    sd reduction ~= sqrt(2)  ->  the folds share NO step-locked structure; the averaged
                                 argmax is the max of 28 draws of noise and locates nothing.
    sd reduction <  sqrt(2)  ->  a shared shape exists and the averaged peak means something.

A WITHIN-FOLD PERMUTATION TEST makes the same point directly: shuffling each fold's values
across steps independently destroys any cross-fold alignment while preserving each fold's
own distribution exactly. If the real averaged peak is not extreme against that null, the
folds do not peak together.

Rates only, per PROJECT.md §9. Fold 1's step-2000 value is never compared against round 1
here -- every comparison in this file is within the averaged curve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import (metric_suite, paired_bootstrap, default_tpr_metric,
                     DEFAULT_TARGET_FPR, DEFAULT_TARGET_PURITY,
                     DEFAULT_TARGET_COMPLETENESS, NN_NP_Q1)      # noqa: E402

ONE_POSITIVE = 1.0 / 98.0
UNFROZEN = {0: "runs/freematch_fold0_nb3_unfrozen",
            1: "runs/freematch_nb3_unfrozen_fold1"}
METRICS = (("tpr", "TPR@1e-3"), ("comp_at_purity", "comp@24%-purity"),
           ("purity_corr", "purity@50%-comp"))


def scores(run_dir, fold, step):
    p = os.path.join(run_dir, "logits", f"step_{int(step)}.npz")
    if not os.path.isfile(p):
        return None, None
    with np.load(p, allow_pickle=False) as z:
        return z[f"test_pos_fold{fold}"], z["test_negatives_shared"]


def suite(pos, neg):
    y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
    s = np.concatenate([pos, neg]).astype(float)
    return metric_suite(y, s, target_fpr=DEFAULT_TARGET_FPR,
                        target_completeness=DEFAULT_TARGET_COMPLETENESS,
                        nn_np_ratio=NN_NP_Q1, target_purity=DEFAULT_TARGET_PURITY)


def common_steps(root):
    sets = []
    for k, rel in UNFROZEN.items():
        d = os.path.join(root, rel)
        ss = set()
        for f in os.listdir(os.path.join(d, "logits")):
            if f.startswith("step_") and f.endswith(".npz"):
                ss.add(int(f[5:-4]))
        sets.append(ss)
    return sorted(set.intersection(*sets))


def sd_reduction_report(per_fold, avg, label):
    """The decisive statistic -- see the module docstring."""
    sds = {k: float(np.nanstd(v)) for k, v in per_fold.items()}
    mean_ind = float(np.mean(list(sds.values())))
    sd_avg = float(np.nanstd(avg))
    obs = mean_ind / sd_avg if sd_avg > 0 else float("inf")
    a, b = (np.asarray(per_fold[k], float) for k in sorted(per_fold))
    ok = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    print(f"\n  {label} -- NOISE REDUCTION")
    for k in sorted(sds):
        print(f"    fold {k} sd                   {sds[k]:.4f}")
    print(f"    mean individual sd           {mean_ind:.4f}")
    print(f"    averaged-curve sd            {sd_avg:.4f}")
    print(f"    observed reduction factor    {obs:.3f}   (sqrt(2) = 1.414 if independent)")
    print(f"    per-step correlation r       {r:+.3f}")
    verdict = ("NO shared step-locked structure -- averaging behaves as if the folds were "
               "independent noise" if obs >= 1.30 else
               ("shared shape: the folds move together" if obs < 1.15 else
                "between the two -- weak or no shared structure"))
    print(f"    -> {verdict}")
    return {"sd_per_fold": sds, "mean_individual_sd": mean_ind, "sd_average": sd_avg,
            "reduction_factor": obs, "sqrt2": float(np.sqrt(2)), "corr": r,
            "verdict": verdict}


def permutation_peak_test(per_fold, steps, n=20000, seed=0):
    """Is the averaged peak extreme against a null with NO cross-fold alignment?

    Each fold's values are shuffled across steps INDEPENDENTLY, preserving that fold's own
    marginal distribution exactly and destroying only the step-locked alignment between
    folds. The statistic is (max - mean) of the averaged curve.
    """
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(per_fold[k], float) for k in sorted(per_fold)]
    ok = np.all([np.isfinite(a) for a in arrs], axis=0)
    arrs = [a[ok] for a in arrs]
    real = np.mean(arrs, axis=0)
    stat = float(real.max() - real.mean())
    null = np.empty(n)
    for i in range(n):
        perm = np.mean([rng.permutation(a) for a in arrs], axis=0)
        null[i] = perm.max() - perm.mean()
    p = float((null >= stat).mean())
    return {"statistic": stat, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p_value": p,
            "n_perm": n,
            "verdict": ("the averaged peak is NOT distinguishable from a curve with no "
                        "cross-fold alignment" if p > 0.05 else
                        "the averaged peak IS extreme against the no-alignment null")}


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(root, "runs", "fold_average.json"))
    args = ap.parse_args(argv)

    steps = common_steps(root)
    print("#" * 100)
    print("# FOLD-AVERAGED CURVE -- folds 0 and 1 (the two UNFROZEN runs)")
    print("#" * 100)
    print(f"  common range {steps[0]}..{steps[-1]} at {steps[1]-steps[0]}-step spacing, "
          f"{len(steps)} points")
    print("  fold 3 EXCLUDED: it is the frozen spot check, with no unfreezing trajectory")

    # ---- gather every metric at every common step, per fold ----
    vals = {m: {k: [] for k in UNFROZEN} for m, _ in METRICS}
    keep = {k: {} for k in UNFROZEN}
    for k, rel in UNFROZEN.items():
        d = os.path.join(root, rel)
        for s in steps:
            pos, neg = scores(d, k, s)
            su = suite(pos, neg)
            keep[k][s] = (pos, neg)
            for m, _ in METRICS:
                v = su[m]
                vals[m][k].append(float(v) if (v is not None and np.isfinite(v)) else np.nan)

    result = {"steps": steps, "metrics": {}}
    for m, label in METRICS:
        per = {k: np.asarray(vals[m][k], float) for k in UNFROZEN}
        avg = np.nanmean(np.vstack([per[k] for k in sorted(per)]), axis=0)
        print("\n" + "=" * 100)
        print(f"{label}")
        print("=" * 100)
        print(f"  {'step':>6}{'fold 0':>10}{'fold 1':>10}{'AVERAGE':>10}")
        for i, s in enumerate(steps):
            mk = ""
            if np.isfinite(avg[i]) and avg[i] == np.nanmax(avg):
                mk = "  <-- argmax"
            print(f"  {s:>6}{per[0][i]:>10.4f}{per[1][i]:>10.4f}{avg[i]:>10.4f}{mk}")
        i = int(np.nanargmax(avg))
        near = [s for s, v in zip(steps, avg)
                if np.isfinite(v) and v >= np.nanmax(avg) - 2 * ONE_POSITIVE]
        print(f"\n  averaged argmax {np.nanmax(avg):.4f} @ step {steps[i]}")
        print(f"  steps within 2 positives of it ({len(near)} of {len(steps)}): {near}")
        sdr = sd_reduction_report(per, avg, label)
        perm = permutation_peak_test(per, steps, n=args.perm)
        print(f"\n  PERMUTATION TEST (no cross-fold alignment null, {perm['n_perm']} draws)")
        print(f"    observed (max - mean)   {perm['statistic']:.4f}")
        print(f"    null mean               {perm['null_mean']:.4f}   null p95 "
              f"{perm['null_p95']:.4f}")
        print(f"    p = {perm['p_value']:.3f}  ->  {perm['verdict']}")
        result["metrics"][m] = {"label": label, "per_fold": {k: per[k].tolist() for k in per},
                                "average": avg.tolist(), "argmax_step": steps[i],
                                "argmax": float(np.nanmax(avg)),
                                "within_2_positives": near,
                                "noise": sdr, "permutation": perm}

    # ---- is ANY budget in 500..3375 distinguishable from the best? ----
    print("\n" + "=" * 100)
    print("IS ANY BUDGET DISTINGUISHABLE?  paired bootstrap of (best step - candidate),")
    print("per fold on that fold's own positives + the shared negatives, then averaged.")
    print("=" * 100)
    tpr_avg = np.asarray(result["metrics"]["tpr"]["average"], float)
    best = result["metrics"]["tpr"]["argmax_step"]
    cands = [s for s in steps if 500 <= s <= 3375]
    print(f"  best averaged step = {best}\n")
    print(f"  {'step':>6}{'avg TPR':>10}{'diff vs best':>14}{'CI (folds avg)':>26}"
          f"{'distinguishable':>17}")
    rows = {}
    for s in cands:
        diffs, los, his = [], [], []
        for k in sorted(UNFROZEN):
            pb, nb = keep[k][best]
            ps, ns = keep[k][s]
            y = np.concatenate([np.ones(pb.size, np.int64), np.zeros(nb.size, np.int64)])
            bs = paired_bootstrap(y, np.concatenate([pb, nb]).astype(float),
                                  np.concatenate([ps, ns]).astype(float),
                                  metric_fn=default_tpr_metric(DEFAULT_TARGET_FPR),
                                  B=args.bootstrap, seed=0)
            diffs.append(float(bs.get("point", np.nan)))
            los.append(float(bs.get("lo", np.nan))); his.append(float(bs.get("hi", np.nan)))
        d, lo, hi = np.mean(diffs), np.mean(los), np.mean(his)
        dist = bool(lo > 0)
        i = steps.index(s)
        print(f"  {s:>6}{tpr_avg[i]:>10.4f}{d:>+14.4f}   [{lo:>+7.4f}, {hi:>+7.4f}]"
              f"{str(dist):>17}")
        rows[s] = {"avg_tpr": float(tpr_avg[i]), "diff_vs_best": float(d),
                   "ci": [float(lo), float(hi)], "distinguishable": dist}
    n_dist = sum(1 for v in rows.values() if v["distinguishable"])
    print(f"\n  {n_dist} of {len(rows)} candidate budgets in 500..3375 are distinguishable "
          f"from the best.")
    result["budget_comparison"] = {"best_step": best, "rows": rows,
                                   "n_distinguishable": n_dist, "n_candidates": len(rows)}

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
