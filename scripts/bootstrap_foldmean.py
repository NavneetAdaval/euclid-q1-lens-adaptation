"""
bootstrap_foldmean.py
=====================
The PRE-REGISTERED confirmatory test: folds-1-4 fold-mean TPR@FPR=1e-3, unfrozen minus
round-1 frozen, with a bootstrap that respects the SHARED NEGATIVE POOL.

WHY A NAIVE BOOTSTRAP IS WRONG HERE
-----------------------------------
The four confirmatory folds share ONE negative pool (PROJECT.md 4.4: the full ~26k P4 set,
identical every fold) and differ only in their ~98 positives. At FPR=1e-3 the operating
threshold is set BY THAT SHARED POOL, so a negative-tail fluctuation moves every fold's
threshold in the same direction at once. The folds are therefore POSITIVELY CORRELATED, and
treating them as four independent samples understates the variance of their mean.

THE STRUCTURE IMPLEMENTED (one bootstrap draw b)
------------------------------------------------
  1. Resample the shared negative pool ONCE: draw n_neg row indices with replacement. The
     SAME index vector is applied to every fold and to both arms. This is what carries the
     across-fold correlation -- one negative-tail realisation for the whole draw.
  2. For each fold k, that fold's model has its own scores on those same resampled
     negatives, so each fold derives its OWN threshold from them (correct: each fold is a
     different model scoring a common pool).
  3. Resample fold k's positives INDEPENDENTLY of the other folds (they are disjoint sets
     of real lenses), with replacement, n_pos_k indices.
  4. The SAME positive indices are applied to the unfrozen and the round-1 arm, so the
     within-fold comparison stays PAIRED.
  5. TPR@1e-3 per fold per arm on that draw -> average over the four folds -> one draw of
     the fold-mean for each arm, and one draw of their difference.

The reported CI is the percentile interval of the difference over B draws. The point
estimate is computed on the real data, not the bootstrap median.

A naive across-fold paired t-test is computed too, as a SANITY ANCHOR only. It assumes the
four folds are independent, which the shared pool makes false, so it is not the result.

Rates only (PROJECT.md 9). Fold 0 is DEVELOPMENT and is excluded from every aggregate here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import tpr_at_fpr, DEFAULT_TARGET_FPR      # noqa: E402

CONFIRMATORY = (1, 2, 3, 4)
DEV_FOLD = 0


def load_arm_scores(root, fold, r1_step=None):
    """(unfrozen pos, unfrozen neg, round1 pos, round1 neg) for one fold, id-verified."""
    d = os.path.join(root, "runs", f"freematch_nb3_unfrozen_fold{fold}")
    cfg = json.load(open(os.path.join(d, "run_config.json")))
    step = int(r1_step or cfg["max_steps"])
    with np.load(os.path.join(d, "logits", f"step_{step}.npz"), allow_pickle=False) as z:
        up, un = z[f"test_pos_fold{fold}"].astype(float), z["test_negatives_shared"].astype(float)
    r1 = os.path.join(root, "runs", "bakeoff_freematch_5fold", "scores",
                      f"freematch_fold{fold}.npz")
    with np.load(r1, allow_pickle=False) as z:
        rp, rn = z["pos_scores"].astype(float), z["neg_scores"].astype(float)
    if up.size != rp.size or un.size != rn.size:
        raise SystemExit(f"fold {fold}: score lengths differ "
                         f"({up.size},{un.size}) vs ({rp.size},{rn.size})")
    return up, un, rp, rn, step


def tpr_of(pos, neg, target_fpr):
    y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
    return tpr_at_fpr(y, np.concatenate([pos, neg]), target_fpr)


def fold_mean_bootstrap(arms, B=5000, seed=0, target_fpr=DEFAULT_TARGET_FPR, verbose=True):
    """See the module docstring for the sampling structure."""
    folds = sorted(arms)
    n_neg = arms[folds[0]][1].size
    for k in folds:
        if arms[k][1].size != n_neg:
            raise SystemExit(f"fold {k} has {arms[k][1].size} negatives, expected {n_neg} "
                             f"-- the pool is supposed to be SHARED and identical")
    rng = np.random.default_rng(seed)
    dif = np.empty(B)
    unf = np.empty(B)
    r1m = np.empty(B)
    for b in range(B):
        # (1) ONE negative resample for the whole draw -- the shared-pool correlation
        nidx = rng.integers(0, n_neg, size=n_neg)
        u_vals, r_vals = [], []
        for k in folds:
            up, un, rp, rn, _ = arms[k]
            # (3)(4) positives resampled per fold, SAME indices for both arms (paired)
            pidx = rng.integers(0, up.size, size=up.size)
            # (2) each fold's own model -> its own threshold from the shared draw
            u_vals.append(tpr_of(up[pidx], un[nidx], target_fpr))
            r_vals.append(tpr_of(rp[pidx], rn[nidx], target_fpr))
        u, r = float(np.mean(u_vals)), float(np.mean(r_vals))
        unf[b], r1m[b], dif[b] = u, r, u - r
        if verbose and (b + 1) % max(B // 10, 1) == 0:
            print(f"    {b+1}/{B} draws", flush=True)
    return unf, r1m, dif


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--out", default=os.path.join(root, "runs", "foldmean_bootstrap.json"))
    args = ap.parse_args(argv)

    arms = {k: load_arm_scores(root, k) for k in CONFIRMATORY}
    print("=" * 92)
    print("FOLD-AWARE BOOTSTRAP -- folds 1-4 fold-mean, unfrozen minus round-1 frozen")
    print("=" * 92)
    print("  structure: ONE shared-negative resample per draw (applied to all folds AND")
    print("  both arms) -> each fold derives its own threshold from it -> positives")
    print("  resampled per fold, SAME indices for both arms (paired within fold).")
    n_neg = arms[CONFIRMATORY[0]][1].size
    print(f"  shared negatives {n_neg}   positives per fold "
          f"{[arms[k][0].size for k in CONFIRMATORY]}   R1 step "
          f"{[arms[k][4] for k in CONFIRMATORY]}")

    # point estimates on the REAL data
    u_pt = {k: tpr_of(arms[k][0], arms[k][1], args.target_fpr) for k in CONFIRMATORY}
    r_pt = {k: tpr_of(arms[k][2], arms[k][3], args.target_fpr) for k in CONFIRMATORY}
    U, R = float(np.mean(list(u_pt.values()))), float(np.mean(list(r_pt.values())))
    print(f"\n  {'fold':>5}{'unfrozen':>11}{'round 1':>11}{'diff':>10}")
    for k in CONFIRMATORY:
        print(f"  {k:>5}{u_pt[k]:>11.4f}{r_pt[k]:>11.4f}{u_pt[k]-r_pt[k]:>+10.4f}")
    print(f"  {'mean':>5}{U:>11.4f}{R:>11.4f}{U-R:>+10.4f}")

    print(f"\n  running {args.bootstrap} draws ...", flush=True)
    unf, r1m, dif = fold_mean_bootstrap(arms, B=args.bootstrap, seed=args.seed,
                                        target_fpr=args.target_fpr)
    lo, hi = np.percentile(dif, [100 * args.alpha / 2, 100 * (1 - args.alpha / 2)])
    p_le0 = float((dif <= 0).mean())
    half = (hi - lo) / 2

    print("\n" + "=" * 92)
    print("RESULT")
    print("=" * 92)
    print(f"  point estimate (fold-mean diff)  {U-R:+.4f}")
    print(f"  {100*(1-args.alpha):.0f}% CI                          "
          f"[{lo:+.4f}, {hi:+.4f}]   half-width {half:.4f}")
    print(f"  bootstrap median                 {np.median(dif):+.4f}")
    print(f"  fraction of draws <= 0           {p_le0:.4f}")
    print(f"  CI excludes 0                    {bool(lo > 0)}")
    print(f"\n  unfrozen fold-mean  {U:.4f}  CI [{np.percentile(unf,2.5):.4f}, "
          f"{np.percentile(unf,97.5):.4f}]")
    print(f"  round-1  fold-mean  {R:.4f}  CI [{np.percentile(r1m,2.5):.4f}, "
          f"{np.percentile(r1m,97.5):.4f}]")

    # SANITY ANCHOR ONLY -- assumes fold independence, which the shared pool violates
    d = np.array([u_pt[k] - r_pt[k] for k in CONFIRMATORY])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
    try:
        from scipy import stats
        p_t = float(2 * (1 - stats.t.cdf(abs(t), d.size - 1)))
    except Exception:
        p_t = None
    print(f"\n  [sanity anchor, NOT the result] across-fold paired t = {t:.2f}, "
          f"{d.size-1} df, p = {p_t if p_t is None else f'{p_t:.3f}'}")
    print("    It treats the four folds as independent; they share one negative pool and")
    print("    one FPR=1e-3 threshold, so it understates the variance of their mean.")

    out = {"structure": "shared-negative resample once per draw; per-fold thresholds; "
                        "paired positive indices across arms",
           "n_shared_negatives": int(n_neg),
           "per_fold": {k: {"unfrozen": u_pt[k], "round1": r_pt[k],
                            "diff": u_pt[k] - r_pt[k]} for k in CONFIRMATORY},
           "unfrozen_fold_mean": U, "round1_fold_mean": R, "diff": U - R,
           "ci": [float(lo), float(hi)], "ci_half_width": float(half),
           "bootstrap_median": float(np.median(dif)),
           "frac_draws_le_0": p_le0, "ci_excludes_zero": bool(lo > 0),
           "B": args.bootstrap, "seed": args.seed,
           "sanity_anchor_paired_t": {"t": float(t), "df": int(d.size - 1), "p": p_t,
                                      "note": "assumes fold independence -- NOT the result"}}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
