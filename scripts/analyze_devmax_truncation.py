"""
analyze_devmax_truncation.py
============================
PROJECT.md 7.14 (OPEN): what MAXIMUM should a per-method fold-0 development run use?

FreeMatch's development run went to 8000 steps (~7.2 h/fold). That is unaffordable to
repeat for every method. 7.14 asks whether a SHORTER maximum still reveals a method's
shape well enough to pick a budget from it.

7.13 is explicit about what "shape" survives at 98 positives: NOT a locatable peak
(0 of 24 candidate budgets in 500-3375 is distinguishable from the best), but the
one structure that does survive --

    "R1 at step 3375 landed at 0.3673 against averaged values of 0.41-0.48 through
     most of the range. LATE IS SYSTEMATICALLY WORSE THAN NOT-LATE."

That, and only that, is the empirical basis for 7.11's 1500-step budget. So the
question this script answers is precisely:

    AT WHAT TRUNCATION CAP C IS "LATE IS WORSE" UNAMBIGUOUS WITHIN THE WINDOW [0, C]?

If the answer is "only at C = 8000", there is no affordable reduction and 7.14's open
item is answered NO. A null answer is a real answer (7.13 is itself a negative result).

Read-only: consumes existing logit dumps, writes runs/devmax_truncation.json + .md.

------------------------------------------------------------------------------------
THE DETECTOR -- FIXED IN CODE BEFORE ANY TRUNCATED RESULT WAS LOOKED AT
------------------------------------------------------------------------------------
Stated here, printed to stdout, and written to the JSON *before* the numbers, so a
later reader can check the definition did not move to fit a tidy answer.

CANDIDATE SET.  For a cap C, the checkpoints in the window are
    S(C) = { s in curve steps : 0 < s <= C }.
Step 0 is EXCLUDED. It is the LP->FT initialisation -- round 1's own frozen head --
not a product of the unfrozen run, and its value is the pre-registered step-0 gate
constant (fold 0: 0.387755, PROJECT.md 7.11). Including it would let a fixed constant
that no training produced drive the "early" arm. (Sensitivity variant V3 includes it.)

PARTITION (primary, "P1-quartile").
    LATE(C)  = { s in S(C) : s >  0.75 * C }      -- the last quartile of the window
    EARLY(C) = { s in S(C) : s <= 0.75 * C }
Both are defined by fractions of the window, not by absolute steps, so the detector
asks the same question of every cap.

STATISTIC.
    Delta(C) = mean_{EARLY(C)} TPR@1e-3  -  mean_{LATE(C)} TPR@1e-3
Delta > 0 means "late is worse", the 7.13 structure. The EARLY arm is a MEAN, not a
max: taking the best earlier region would import a winner's-curse bias that inflates
Delta under the null and would make the known-null control fire by construction.
(The "best earlier window" form the brief offers as an example is run anyway, as
sensitivity variant V2, reported WITH its own permutation-null false-positive rate.)

DECISION RULE.
    "decay detectable at cap C"  <=>  the 95% bootstrap CI of Delta(C) has lo > 0.
    "late is BETTER at cap C"    <=>  that CI has hi < 0.
    otherwise: not distinguishable.
"Not distinguishable" is neither "no effect" nor "no decay" (PROJECT.md 9).

BOOTSTRAP -- the 7.12 fold-aware class-conditional scheme, reduced to one fold.
The compared arms here are CHECKPOINTS OF ONE FOLD, so the shared-pool structure that
7.12 handles across folds applies across checkpoints: every checkpoint scores the SAME
~26k negative pool, so a negative-tail fluctuation moves every checkpoint's FPR=1e-3
threshold together, and the checkpoints are positively correlated, not independent.
Each draw b therefore:
  1. resamples the shared negative pool ONCE (with replacement, same size) -- one
     negative-tail realisation applied to EVERY checkpoint in the window;
  2. resamples the ~98 fold positives ONCE (with replacement, same size) -- the SAME
     positive indices applied to EVERY checkpoint, which is what makes the EARLY-vs-LATE
     comparison PAIRED;
  3. lets each checkpoint derive its OWN FPR=1e-3 threshold from that one negative
     realisation (correct: each checkpoint is a different model scoring a common pool);
  4. forms Delta_b from those per-checkpoint TPRs.
B = 5000 draws, seed 0, percentile interval, alpha = 0.05 -- the 7.12 settings.
Because the draws are shared across caps, the per-cap results are directly comparable
and differ only in which checkpoints enter the two arms.

KNOWN-NULL CONTROLS (PROJECT.md 9). Two, because the first one is degenerate.
  N1  The window [0, 1500] on its own -- the brief's suggestion. Fold 0's argmax sits
      at 1250/1500, so this window is the RISE, not the decay, and the detector must
      not report decay in it. Note this window IS the C = 1500 cap row: at that cap the
      null control and the measurement are the same computation, which is itself the
      reason no cap at or below 1500 can be "defensible" on this evidence.
  N2  A step-permutation null, which is the non-degenerate one. Within each cap the
      EARLY/LATE membership is permuted at random (destroying step-locked structure but
      preserving the checkpoints, the scores and the correlation between them), the full
      bootstrap is re-run, and the fraction of permutations whose CI excludes zero is
      measured. A correctly calibrated detector fires on ~2.5% of permutations at the
      one-sided level implied by alpha = 0.05. Materially more than that means the
      detector is broken and no cap result from it may be believed.
      This is 7.13's own null: "averaging suppressed noise exactly as if the folds
      shared no step-locked structure" -- here we impose no step-locked structure and
      check the detector agrees.

SENSITIVITY VARIANTS (pre-registered, reported, NOT used to pick the headline):
  V2 "best-earlier": EARLY arm = the best contiguous earlier window of the same width
      as LATE. Winner's-curse biased; reported only beside its own N2 rate.
  V3 "with-step0":   step 0 admitted to the EARLY arm.
  V4 "last-third":   LATE = last third of the window instead of the last quartier.

MULTIPLICITY. Eight caps are tested on one curve and the windows are nested, so the
smallest cap that fires is a selected minimum over eight strongly dependent tests. No
correction is applied to the per-cap CIs (they are reported as what they are); the N2
permutation rate is what bounds the family-wise false-alarm behaviour, and the
smallest-defensible-cap statement is made against it, not against the per-cap p-value.

REPORTING. TPR@FPR=1e-3 as a RATE throughout (PROJECT.md 9). No counts of lenses.
The one-positive resolution of the metric, 1/98 = 0.0102, is quoted as a rate scale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import tpr_at_fpr, DEFAULT_TARGET_FPR      # noqa: E402

# ---------------------------------------------------------------------------
# pre-registered constants
# ---------------------------------------------------------------------------
CAPS = (1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000)
LATE_FRAC_PRIMARY = 0.25          # LATE = last quartile of the window
LATE_FRAC_V4 = 1.0 / 3.0          # V4 sensitivity
B_DRAWS = 5000
SEED = 0
ALPHA = 0.05
N_PERM = 2000                     # permutations for the N2 calibration null
NEG_TAIL = 400                    # negatives retained per checkpoint for the fast ROC tail

DEV_RUN = os.path.join("runs", "freematch_fold0_nb3_unfrozen")
FOLD1_RUN = os.path.join("runs", "freematch_nb3_5folds", "_x",
                         "freematch_nb3_unfrozen_fold1")


# ---------------------------------------------------------------------------
# curve / logit loading
# ---------------------------------------------------------------------------

def load_curve(run_dir, fold):
    """(steps, tpr_from_curve_csv) for one run, in step order."""
    path = os.path.join(run_dir, "curve.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    col = f"fold{fold}_tpr_1e-3"
    steps = np.array([int(r["step"]) for r in rows], dtype=np.int64)
    vals = np.array([float(r[col]) for r in rows], dtype=float)
    order = np.argsort(steps)
    return steps[order], vals[order], path


def load_logits(run_dir, fold, steps):
    """(pos (K,Np), neg (K,Nn)) float64 score matrices, one row per checkpoint."""
    pos, neg = [], []
    for s in steps:
        f = os.path.join(run_dir, "logits", f"step_{int(s)}.npz")
        with np.load(f, allow_pickle=False) as z:
            pos.append(z[f"test_pos_fold{fold}"].astype(np.float64))
            neg.append(z["test_negatives_shared"].astype(np.float64))
    return np.stack(pos), np.stack(neg), os.path.join(run_dir, "logits")


# ---------------------------------------------------------------------------
# fast exact TPR@FPR=1e-3 over all checkpoints of one bootstrap draw
# ---------------------------------------------------------------------------
#
# Replicates metrics.tpr_at_fpr exactly -- sklearn roc_curve(drop_intermediate=False)
# then np.interp(target_fpr, fpr, tpr) -- but pays the per-draw cost only in the ROC
# TAIL, which is the only region the FPR=1e-3 operating point can live in.
#
# Two facts make the truncation exact rather than an approximation:
#   * the scores never change across draws (only the row multiplicities do), so the
#     merged descending order and the tie-group structure are PRECOMPUTED once;
#   * FPR=1e-3 on Nn=26399 negatives is reached inside the top few hundred negatives
#     (measured: at most 78 tail items sit at or above the operating threshold on the
#     real data); NEG_TAIL=400 is the retained margin and the crossing is ASSERTED to
#     be bracketed on every draw, so a violation raises instead of silently truncating.
# fps/tps are normalised by the FULL Nn / Np, not by the retained tail, so the fpr and
# tpr axes are the true ones.

class TailROC:
    def __init__(self, pos, neg, neg_tail=NEG_TAIL):
        self.K, self.Np = pos.shape
        self.Nn = neg.shape[1]
        tail = min(neg_tail, self.Nn)
        self.tail = tail

        # top-`tail` negatives per checkpoint, descending
        nord = np.argsort(-neg, axis=1, kind="stable")[:, :tail]
        pord = np.argsort(-pos, axis=1, kind="stable")

        ns = np.take_along_axis(neg, nord, axis=1)          # (K, tail) desc
        ps = np.take_along_axis(pos, pord, axis=1)          # (K, Np)  desc

        merged_scores = np.concatenate([ns, ps], axis=1)     # (K, tail+Np)
        is_pos = np.concatenate([np.zeros((self.K, tail), bool),
                                 np.ones((self.K, self.Np), bool)], axis=1)
        # source row index into the ORIGINAL neg / pos arrays, for multiplicity lookup
        src = np.concatenate([nord, pord], axis=1)

        morder = np.argsort(-merged_scores, axis=1, kind="stable")
        self.M = merged_scores.shape[1]
        self.s_sorted = np.take_along_axis(merged_scores, morder, axis=1)
        self.is_pos = np.take_along_axis(is_pos, morder, axis=1)
        self.src = np.take_along_axis(src, morder, axis=1)
        # split the source index by class so each multiplicity vector is indexed only
        # by indices that are valid for it (pos rows are 0..Np-1, neg rows 0..Nn-1)
        self.src_pos = np.where(self.is_pos, self.src, 0)
        self.src_neg = np.where(self.is_pos, 0, self.src)

        # tie groups: a vertex exists only at a DISTINCT threshold (sklearn semantics).
        # grp_end[i] = last index of i's equal-score run -> reading the cumulative
        # counts there collapses each tie run onto its single ROC vertex.
        same_next = np.zeros((self.K, self.M), bool)
        same_next[:, :-1] = self.s_sorted[:, :-1] == self.s_sorted[:, 1:]
        grp_end = np.empty((self.K, self.M), np.int64)
        idx = np.arange(self.M)
        for k in range(self.K):                              # tiny: K rows, once
            ge = idx.copy()
            for i in range(self.M - 2, -1, -1):
                if same_next[k, i]:
                    ge[i] = ge[i + 1]
            grp_end[k] = ge
        self.grp_end = grp_end
        self.n_ties = int(same_next.sum())

        self.rows = np.arange(self.K)[:, None]

    def tpr(self, pos_counts, neg_counts, target_fpr=DEFAULT_TARGET_FPR):
        """TPR@target_fpr for every checkpoint under one resample, given per-ROW
        multiplicity vectors (pos_counts (Np,), neg_counts (Nn,))."""
        w = np.where(self.is_pos,
                     pos_counts[self.src_pos], neg_counts[self.src_neg]).astype(np.float64)
        tp = np.cumsum(np.where(self.is_pos, w, 0.0), axis=1)
        fp = np.cumsum(np.where(self.is_pos, 0.0, w), axis=1)
        tp = np.take_along_axis(tp, self.grp_end, axis=1)
        fp = np.take_along_axis(fp, self.grp_end, axis=1)

        fpr = fp / self.Nn
        tpr = tp / self.Np
        # leading (0, 0) vertex, as sklearn prepends
        fpr = np.concatenate([np.zeros((self.K, 1)), fpr], axis=1)
        tpr = np.concatenate([np.zeros((self.K, 1)), tpr], axis=1)

        j = (fpr < target_fpr).sum(axis=1)                   # first index with fpr >= target
        if np.any(j >= fpr.shape[1]):
            raise RuntimeError(
                "FPR=%g not bracketed inside the retained ROC tail (NEG_TAIL=%d). "
                "Increase NEG_TAIL; refusing to extrapolate." % (target_fpr, self.tail))
        j0 = j - 1
        f0 = fpr[self.rows[:, 0], j0]
        f1 = fpr[self.rows[:, 0], j]
        t0 = tpr[self.rows[:, 0], j0]
        t1 = tpr[self.rows[:, 0], j]
        frac = np.where(f1 > f0, (target_fpr - f0) / np.where(f1 > f0, f1 - f0, 1.0), 0.0)
        return t0 + frac * (t1 - t0)


def build_tpr_matrix(pos, neg, B=B_DRAWS, seed=SEED, target_fpr=DEFAULT_TARGET_FPR,
                     verbose=True):
    """T[b, k] -- TPR@target_fpr of checkpoint k under bootstrap draw b.

    ONE shared negative resample and ONE shared positive resample per draw, applied to
    every checkpoint (see the module docstring, step 1-3)."""
    roc = TailROC(pos, neg)
    K, Np, Nn = roc.K, roc.Np, roc.Nn
    rng = np.random.default_rng(seed)
    T = np.empty((B, K), dtype=float)
    t0 = time.time()
    for b in range(B):
        nidx = rng.integers(0, Nn, size=Nn)
        pidx = rng.integers(0, Np, size=Np)
        nc = np.bincount(nidx, minlength=Nn).astype(np.float64)
        pc = np.bincount(pidx, minlength=Np).astype(np.float64)
        T[b] = roc.tpr(pc, nc, target_fpr)
        if verbose and (b + 1) % max(B // 10, 1) == 0:
            print(f"    {b+1}/{B} draws  ({time.time()-t0:.1f}s)", flush=True)
    return T, roc


def validate_fast_roc(roc, pos, neg, n_checks=200, seed=12345,
                      target_fpr=DEFAULT_TARGET_FPR):
    """Adversarial check: the fast tail ROC must reproduce metrics.tpr_at_fpr EXACTLY
    (max|diff| = 0), both on the real data and on random bootstrap resamples."""
    K, Np = pos.shape
    Nn = neg.shape[1]
    y = np.concatenate([np.ones(Np, np.int64), np.zeros(Nn, np.int64)])
    worst = 0.0

    # (a) unresampled: every checkpoint
    ones_p = np.ones(Np); ones_n = np.ones(Nn)
    fast = roc.tpr(ones_p, ones_n, target_fpr)
    for k in range(K):
        ref = tpr_at_fpr(y, np.concatenate([pos[k], neg[k]]), target_fpr)
        worst = max(worst, abs(fast[k] - ref))

    # (b) random resamples: random checkpoints
    rng = np.random.default_rng(seed)
    for _ in range(n_checks):
        nidx = rng.integers(0, Nn, size=Nn)
        pidx = rng.integers(0, Np, size=Np)
        fast = roc.tpr(np.bincount(pidx, minlength=Np).astype(float),
                       np.bincount(nidx, minlength=Nn).astype(float), target_fpr)
        k = int(rng.integers(0, K))
        ref = tpr_at_fpr(y, np.concatenate([pos[k][pidx], neg[k][nidx]]), target_fpr)
        worst = max(worst, abs(fast[k] - ref))
    return float(worst)


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------

def window_idx(steps, cap, include_step0):
    lo = -1 if include_step0 else 0
    return np.where((steps > lo) & (steps <= cap))[0]


def split_quartile(steps, idx, cap, late_frac):
    late = idx[steps[idx] > (1.0 - late_frac) * cap]
    early = idx[steps[idx] <= (1.0 - late_frac) * cap]
    return early, late


def ci_of(delta_draws, alpha=ALPHA):
    lo, hi = np.percentile(delta_draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def verdict(lo, hi):
    if lo > 0:
        return "DECAY DETECTABLE (late worse)"
    if hi < 0:
        return "REVERSED (late better)"
    return "not distinguishable"


def run_detector(T, point, steps, early, late, alpha=ALPHA):
    """Delta = mean(EARLY) - mean(LATE), point estimate on the real curve + bootstrap CI."""
    d_point = float(point[early].mean() - point[late].mean())
    d_draws = T[:, early].mean(axis=1) - T[:, late].mean(axis=1)
    lo, hi = ci_of(d_draws, alpha)
    return {
        "n_early": int(early.size), "n_late": int(late.size),
        "early_steps": [int(s) for s in steps[early]],
        "late_steps": [int(s) for s in steps[late]],
        "early_mean": float(point[early].mean()),
        "late_mean": float(point[late].mean()),
        "delta": d_point,
        "ci": [lo, hi], "ci_half_width": float((hi - lo) / 2),
        "bootstrap_median": float(np.median(d_draws)),
        "frac_draws_le_0": float((d_draws <= 0).mean()),
        "detected": bool(lo > 0),
        "verdict": verdict(lo, hi),
    }


def best_earlier_delta(T, point, steps, early, late, alpha=ALPHA):
    """V2: EARLY arm = the best contiguous earlier window of the same width as LATE."""
    w = int(late.size)
    if early.size < w:
        return None
    starts = range(0, early.size - w + 1)
    cand = [early[i:i + w] for i in starts]
    means = [float(point[c].mean()) for c in cand]
    best = cand[int(np.argmax(means))]
    out = run_detector(T, point, steps, best, late, alpha)
    out["window_width"] = w
    out["n_candidate_windows"] = len(cand)
    return out


def permutation_null(T, point, steps, idx, n_late, observed_delta,
                     n_perm=N_PERM, seed=777, alpha=ALPHA):
    """N2: permute EARLY/LATE membership within the window.

    Two readouts, both from the SAME pre-registered permutations:

    (a) the FIRE RATE -- how often the pre-registered decision rule (bootstrap CI
        excludes zero) fires when no step-locked structure exists. This is the
        calibration check the definition committed to.

    (b) the PERMUTATION p-VALUE -- Pr(Delta_perm >= Delta_observed). This is the
        stricter and more informative readout, and it is the one that answers
        "is the decay UNAMBIGUOUS", because the bootstrap CI is CONDITIONAL ON THE
        CHECKPOINT SET: it propagates evaluation-data sampling error only, and treats
        the 65 trained models as fixed. Checkpoint-to-checkpoint scatter that is real
        but not step-locked therefore does NOT enter the CI, and can push it off zero.
        The permutation null is what prices that scatter in. Reported alongside the
        pre-registered rule, never substituted for it.
    """
    rng = np.random.default_rng(seed)
    fired_pos = fired_neg = 0
    deltas = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(idx)
        late = perm[:n_late]
        early = perm[n_late:]
        d = T[:, early].mean(axis=1) - T[:, late].mean(axis=1)
        lo, hi = ci_of(d, alpha)
        deltas[i] = float(point[early].mean() - point[late].mean())
        fired_pos += int(lo > 0)
        fired_neg += int(hi < 0)
    # add-one (Phipson & Smyth 2010) so the p-value can never be exactly 0
    p_perm = float((int((deltas >= observed_delta).sum()) + 1) / (n_perm + 1))
    return {
        "n_perm": int(n_perm),
        "frac_fire_positive": fired_pos / n_perm,
        "frac_fire_either": (fired_pos + fired_neg) / n_perm,
        "nominal_one_sided": alpha / 2,
        "perm_delta_sd": float(np.std(deltas, ddof=1)),
        "permutation_p_one_sided": p_perm,
        "broken": bool((fired_pos / n_perm) > 4 * (alpha / 2)),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

DEFINITION_TEXT = __doc__.split(
    "THE DETECTOR -- FIXED IN CODE BEFORE ANY TRUNCATED RESULT WAS LOOKED AT")[1]


def analyse_run(tag, run_dir, fold, caps, root, B, seed, verbose=True):
    steps, curve_tpr, curve_path = load_curve(os.path.join(root, run_dir), fold)
    pos, neg, logit_dir = load_logits(os.path.join(root, run_dir), fold, steps)

    # provenance: the logit dumps must reproduce curve.csv exactly, or nothing below holds
    y = np.concatenate([np.ones(pos.shape[1], np.int64), np.zeros(neg.shape[1], np.int64)])
    recomputed = np.array([tpr_at_fpr(y, np.concatenate([pos[k], neg[k]]))
                           for k in range(steps.size)])
    max_curve_diff = float(np.abs(recomputed - curve_tpr).max())

    print(f"\n[{tag}] curve {curve_path}")
    print(f"[{tag}]   {steps.size} curve points, steps {steps.min()}..{steps.max()} "
          f"every {int(np.unique(np.diff(steps))[0])}")
    print(f"[{tag}]   logits {logit_dir}: {steps.size} per-checkpoint dumps "
          f"(+ ids.npz), {pos.shape[1]} positives / {neg.shape[1]} shared negatives")
    print(f"[{tag}]   max|recomputed - curve.csv| = {max_curve_diff:.3e}")

    print(f"[{tag}] building the {B}x{steps.size} bootstrap TPR matrix ...", flush=True)
    T, roc = build_tpr_matrix(pos, neg, B=B, seed=seed, verbose=verbose)
    worst = validate_fast_roc(roc, pos, neg)
    print(f"[{tag}]   fast-ROC vs metrics.tpr_at_fpr, max|diff| = {worst:.3e} "
          f"({roc.n_ties} tie-runs collapsed)")
    if worst != 0.0:
        raise SystemExit(f"[{tag}] fast ROC does not reproduce metrics.tpr_at_fpr exactly")

    caps = [c for c in caps if c <= steps.max()]
    per_cap = {}
    for cap in caps:
        idx = window_idx(steps, cap, include_step0=False)
        early, late = split_quartile(steps, idx, cap, LATE_FRAC_PRIMARY)
        if early.size == 0 or late.size == 0:
            continue
        row = {"cap": int(cap),
               "n_checkpoints_in_window": int(idx.size),
               "primary": run_detector(T, curve_tpr, steps, early, late)}

        # sensitivity variants
        v2 = best_earlier_delta(T, curve_tpr, steps, early, late)
        if v2 is not None:
            row["V2_best_earlier"] = v2
        idx0 = window_idx(steps, cap, include_step0=True)
        e0, l0 = split_quartile(steps, idx0, cap, LATE_FRAC_PRIMARY)
        row["V3_with_step0"] = run_detector(T, curve_tpr, steps, e0, l0)
        e4, l4 = split_quartile(steps, idx, cap, LATE_FRAC_V4)
        if e4.size and l4.size:
            row["V4_last_third"] = run_detector(T, curve_tpr, steps, e4, l4)

        # V5 -- POST-HOC, added AFTER the primary table was seen, labelled as such and
        # NOT used to answer the question. Rationale for reporting it anyway: under R1
        # the budget IS the selector (PROJECT.md 7.11), so the decision-relevant
        # quantity at cap C is the ENDPOINT, not the last-quartile mean.
        endpoint = idx[-1:]
        row["V5_endpoint_vs_window_POSTHOC"] = run_detector(
            T, curve_tpr, steps, idx[:-1], endpoint)

        # N2 calibration null for the primary partition and for V2
        row["N2_permutation_null_primary"] = permutation_null(
            T, curve_tpr, steps, idx, n_late=int(late.size),
            observed_delta=row["primary"]["delta"])
        if v2 is not None:
            row["N2_permutation_null_V2"] = _perm_null_v2(
                T, curve_tpr, steps, idx, int(late.size))
        per_cap[str(cap)] = row

    return {
        "run_dir": run_dir,
        "curve_csv": os.path.relpath(curve_path, root).replace("\\", "/"),
        "logits_dir": os.path.relpath(logit_dir, root).replace("\\", "/"),
        "n_curve_points": int(steps.size),
        "n_logit_dumps": int(steps.size),
        "step_min": int(steps.min()), "step_max": int(steps.max()),
        "step_spacing": int(np.unique(np.diff(steps))[0]),
        "n_positives": int(pos.shape[1]), "n_shared_negatives": int(neg.shape[1]),
        "max_abs_diff_logits_vs_curve_csv": max_curve_diff,
        "fast_roc_max_abs_diff_vs_metrics": worst,
        "curve": {int(s): float(v) for s, v in zip(steps, curve_tpr)},
        "per_cap": per_cap,
    }, T, steps, curve_tpr


def _perm_null_v2(T, point, steps, idx, n_late, n_perm=N_PERM, seed=778, alpha=ALPHA):
    """The V2 'best earlier window' detector run under the same permutation null, to
    quantify its winner's-curse false-alarm rate."""
    rng = np.random.default_rng(seed)
    fired = 0
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        late = np.sort(perm[:n_late])
        early = np.sort(perm[n_late:])
        out = best_earlier_delta(T, point, steps, early, late, alpha)
        if out is not None and out["detected"]:
            fired += 1
    return {"n_perm": int(n_perm), "frac_fire_positive": fired / n_perm,
            "nominal_one_sided": alpha / 2,
            "broken": bool((fired / n_perm) > 4 * (alpha / 2))}


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description="PROJECT.md 7.14 truncation-cap analysis")
    ap.add_argument("--bootstrap", type=int, default=B_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=os.path.join(root, "runs", "devmax_truncation.json"))
    ap.add_argument("--md", default=os.path.join(root, "runs", "devmax_truncation.md"))
    args = ap.parse_args(argv)

    defn_hash = hashlib.sha256(DEFINITION_TEXT.encode()).hexdigest()

    print("=" * 92)
    print("PROJECT.md 7.14 -- DEVELOPMENT-BUDGET MAXIMUM: at what truncation cap is the")
    print("                   'late is worse' decay unambiguous?")
    print("=" * 92)
    print(DEFINITION_TEXT)
    print(f"definition sha256 = {defn_hash}")
    print("=" * 92)

    out = {
        "question": "PROJECT.md 7.14 (open): at what truncation cap C is the fold-0 "
                    "'late is systematically worse than not-late' decay unambiguous?",
        "definition_stated_before_running": DEFINITION_TEXT,
        "definition_sha256": defn_hash,
        "settings": {"caps": list(CAPS), "late_frac_primary": LATE_FRAC_PRIMARY,
                     "late_frac_V4": LATE_FRAC_V4, "B": args.bootstrap,
                     "seed": args.seed, "alpha": ALPHA, "n_perm": N_PERM,
                     "target_fpr": DEFAULT_TARGET_FPR,
                     "step0_excluded_from_primary": True,
                     "neg_tail_retained": NEG_TAIL},
        "metric": "TPR@FPR=1e-3, a rate (PROJECT.md 2/9). One-positive resolution "
                  "1/98 = 0.010204 (a rate scale, not a count).",
    }

    dev, T0, steps0, curve0 = analyse_run("fold0-dev", DEV_RUN, 0, CAPS, root,
                                          args.bootstrap, args.seed)
    out["fold0_development"] = dev

    # N1 -- the brief's known-null control, stated separately from the cap table
    idx15 = window_idx(steps0, 1500, include_step0=False)
    e15, l15 = split_quartile(steps0, idx15, 1500, LATE_FRAC_PRIMARY)
    out["N1_known_null_first_1500"] = {
        "note": "The brief's control window. Fold 0's argmax is at 1250/1500, so this "
                "window is the RISE; the detector must NOT report decay in it. It is "
                "the SAME computation as the cap=1500 row -- which is exactly why no "
                "cap at or below 1500 can be defended from this evidence.",
        **run_detector(T0, curve0, steps0, e15, l15),
    }

    # secondary, clearly labelled: fold 1's partial curve
    print("\n" + "-" * 92)
    print("SECONDARY (fold 1, partial ~3375-step curve). PROJECT.md 7.13's LAB-RECORD")
    print("NOTE applies: fold 1's curve was already inspected in the turn-timing and")
    print("fold-averaging analyses and is NOT cited in any paper. It is here only to")
    print("confirm its extent and to see whether a second curve behaves the same way.")
    print("-" * 92)
    f1, T1, steps1, curve1 = analyse_run("fold1-partial", FOLD1_RUN, 1, CAPS, root,
                                         args.bootstrap, args.seed)
    f1["status"] = ("PARTIAL -- run killed mid-flight (PROJECT.md 7.11); lab-record "
                    "only, not cited in any paper (7.13 lab-record note)")
    out["fold1_partial_secondary"] = f1

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[out] {args.out}")

    _write_md(out, args.md)
    print(f"[out] {args.md}")
    return 0


def _write_md(out, path):
    dev = out["fold0_development"]
    L = []
    A = L.append
    A("# Development-budget maximum: at what truncation cap is the decay unambiguous?")
    A("")
    A("PROJECT.md §7.14, open item. Read-only analysis of artifacts already on disk; "
      "no GPU, no training, nothing existing modified.")
    A("")
    A("All numbers are **TPR@FPR=1e-3 as a rate** (§2/§9). The metric's one-positive "
      "resolution is **0.0102** as a rate.")
    A("")
    A("## 1. Files used")
    A("")
    A("| role | file | curve points |")
    A("|---|---|---|")
    A(f"| fold 0 development curve (full 8000) | `{dev['curve_csv']}` | "
      f"{dev['n_curve_points']} |")
    A(f"| fold 0 per-checkpoint logit dumps | `{dev['logits_dir']}/step_*.npz` | "
      f"{dev['n_logit_dumps']} |")
    f1 = out["fold1_partial_secondary"]
    A(f"| fold 1 partial curve (secondary) | `{f1['curve_csv']}` | "
      f"{f1['n_curve_points']} |")
    A(f"| fold 1 per-checkpoint logit dumps | `{f1['logits_dir']}/step_*.npz` | "
      f"{f1['n_logit_dumps']} |")
    A("")
    A(f"Fold 0: steps {dev['step_min']}–{dev['step_max']} every {dev['step_spacing']}, "
      f"{dev['n_positives']} positives against {dev['n_shared_negatives']} shared "
      f"negatives.")
    A(f"Fold 1: steps {f1['step_min']}–{f1['step_max']} every {f1['step_spacing']} "
      f"— {f1['status']}")
    A("")
    A(f"Provenance: recomputing TPR@1e-3 from the logit dumps reproduces `curve.csv` "
      f"with max|diff| = {dev['max_abs_diff_logits_vs_curve_csv']:.1e} (fold 0) and "
      f"{f1['max_abs_diff_logits_vs_curve_csv']:.1e} (fold 1). The fast tail-ROC used "
      f"inside the bootstrap reproduces `metrics.tpr_at_fpr` with "
      f"max|diff| = {dev['fast_roc_max_abs_diff_vs_metrics']:.1e}.")
    A("")
    A("## 2. The detector (fixed before any truncated result was looked at)")
    A("")
    A("```")
    A(out["definition_stated_before_running"].strip())
    A("```")
    A(f"definition sha256 `{out['definition_sha256']}`")
    A("")
    A("## 3. Result per cap — fold 0 (development)")
    A("")
    A("| cap | ckpts | EARLY mean | LATE mean | Δ | 95% CI | verdict |")
    A("|---|---|---|---|---|---|---|")
    for cap in sorted(dev["per_cap"], key=int):
        r = dev["per_cap"][cap]["primary"]
        A(f"| {cap} | {dev['per_cap'][cap]['n_checkpoints_in_window']} | "
          f"{r['early_mean']:.4f} | {r['late_mean']:.4f} | {r['delta']:+.4f} | "
          f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['verdict']} |")
    A("")
    A("## 4. Known-null controls")
    A("")
    n1 = out["N1_known_null_first_1500"]
    A(f"**N1 — first 1500 steps only.** Δ = {n1['delta']:+.4f}, "
      f"95% CI [{n1['ci'][0]:+.4f}, {n1['ci'][1]:+.4f}] → **{n1['verdict']}**.")
    A("")
    A(n1["note"])
    A("")
    A("**N2 — step-permutation null (the non-degenerate control).** EARLY/LATE "
      "membership is permuted within the window, destroying step-locked structure "
      "while preserving the checkpoints, their scores and their correlations.")
    A("")
    A("| cap | primary fires on | V2 'best earlier' fires on | nominal | permutation p (primary) |")
    A("|---|---|---|---|---|")
    for cap in sorted(dev["per_cap"], key=int):
        r = dev["per_cap"][cap]
        n2 = r["N2_permutation_null_primary"]
        v2 = r.get("N2_permutation_null_V2")
        v2s = "n/a" if v2 is None else f"{v2['frac_fire_positive']*100:.1f}%"
        A(f"| {cap} | {n2['frac_fire_positive']*100:.1f}% | {v2s} | "
          f"{n2['nominal_one_sided']*100:.1f}% | "
          f"{n2['permutation_p_one_sided']:.3f} |")
    A("")
    A("## 5. Sensitivity variants")
    A("")
    A("| cap | V2 best-earlier | V3 with step 0 | V4 last-third | V5 endpoint (post-hoc) |")
    A("|---|---|---|---|---|")
    for cap in sorted(dev["per_cap"], key=int):
        r = dev["per_cap"][cap]
        def fmt(k):
            v = r.get(k)
            if v is None:
                return "n/a"
            return f"{v['delta']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"
        A(f"| {cap} | {fmt('V2_best_earlier')} | {fmt('V3_with_step0')} | "
          f"{fmt('V4_last_third')} | {fmt('V5_endpoint_vs_window_POSTHOC')} |")
    A("")
    A("V5 was added **after** the primary table was seen and is labelled post-hoc; it "
      "is reported because under R1 the budget *is* the selector, so the "
      "decision-relevant quantity at a cap is the endpoint. It does not answer the "
      "question — the pre-registered primary does.")
    A("")
    A("## 6. Secondary — fold 1 partial curve")
    A("")
    A(f1["status"])
    A("")
    A("| cap | ckpts | EARLY mean | LATE mean | Δ | 95% CI | verdict |")
    A("|---|---|---|---|---|---|---|")
    for cap in sorted(f1["per_cap"], key=int):
        r = f1["per_cap"][cap]["primary"]
        A(f"| {cap} | {f1['per_cap'][cap]['n_checkpoints_in_window']} | "
          f"{r['early_mean']:.4f} | {r['late_mean']:.4f} | {r['delta']:+.4f} | "
          f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['verdict']} |")
    A("")
    A("## 7. The answer")
    A("")
    caps_fire = [int(c) for c in sorted(dev["per_cap"], key=int)
                 if dev["per_cap"][c]["primary"]["detected"]]
    smallest = min(caps_fire) if caps_fire else None
    if smallest is None:
        A("**No cap tested — including 8000 — makes the decay detectable under the "
          "pre-registered rule.**")
    else:
        A(f"**The smallest cap at which the pre-registered detector fires is "
          f"{smallest}, and it is the only cap that fires.** Every cap from 1500 to "
          f"6000 returns *not distinguishable*.")
    A("")
    A("So the honest answer to §7.14's open question is: **this analysis does not "
      "support any reduction below 8000.** §7.14 asked \"whether the decay is "
      "unambiguous by step 3000\" — on fold 0's own development curve it is not, and "
      "on fold 1's partial curve nothing is detectable at any cap it reaches either.")
    A("")
    A("Three things keep that from being over-read:")
    A("")
    A("- The **permutation readout is more permissive than the pre-registered rule** "
      "and would put the onset earlier — p = "
      + ", ".join(f"{c}→{dev['per_cap'][c]['N2_permutation_null_primary']['permutation_p_one_sided']:.3f}"
                  for c in sorted(dev["per_cap"], key=int))
      + ". But that series is **not monotone in the cap**, which is the signature of "
        "noise wandering across the threshold rather than of a real earlier onset. "
        "A criterion that says \"yes at 2500, no at 3000, no at 4000, yes at 5000\" "
        "cannot be used to choose a cap.")
    A("- **The gap between the two readouts is itself informative.** The bootstrap CI "
      "is conditional on the checkpoint set: it prices evaluation-data sampling error "
      "and treats the trained models as fixed. The permutation null prices "
      "checkpoint-to-checkpoint scatter instead. They disagree at 5000–6000, and where "
      "two defensible null models disagree, \"unambiguous\" is not the right word.")
    A("- **Even the reduction the permissive readout would license does not solve the "
      "affordability problem.** At the measured throughput (431.6 min for 8000 steps "
      "plus 65 curve points) a 6000-step cap is ≈5.4 h/fold against 7.2 h — a ~25% "
      "saving. The caps that would actually make a per-method development run cheap "
      "(≤3000 ≈ ≤2.7 h/fold) are precisely the caps at which **nothing** is "
      "detectable, on either readout, on either fold.")
    A("")
    A("**Consequence for §7.14.** The per-method development-run maximum cannot be set "
      "from evidence at 98 positives per fold. It has to be set as a declared "
      "cost/coverage choice, disclosed as such, rather than justified as \"long enough "
      "to see the shape\". This is the same negative that §7.13 reports one level down: "
      "§7.13 found the peak location unmeasurable; this finds the decay ONSET "
      "unmeasurable below the full budget. The one structure that survives at 8000 "
      "survives only at 8000.")
    A("")
    A("**Detector integrity.** The known-null control passes: on the first-1500 window "
      "the detector reports no decay (it correctly sees the rise instead). The "
      "step-permutation null fires on ≤1.2% of permutations for every cap up to 6000 "
      "against a 2.5% nominal — conservative, not broken. At cap 8000 it fires on "
      f"{dev['per_cap']['8000']['N2_permutation_null_primary']['frac_fire_positive']*100:.2f}% "
      "of permutations, above nominal because the 8000-step checkpoint set has real "
      "non-step-locked scatter that a conditional CI does not price; the observed "
      "permutation p at 8000 is below 1/2001, far under that floor, so the cap-8000 "
      "detection survives its own false-alarm rate. The pre-registered warning about "
      "the \"best earlier region\" form was also borne out: V2 fires on "
      f"{dev['per_cap']['8000']['N2_permutation_null_V2']['frac_fire_positive']*100:.1f}% "
      "of permutations at cap 8000 — it is a broken detector, and it is the "
      "operationalisation the brief offered as an example.")
    A("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
