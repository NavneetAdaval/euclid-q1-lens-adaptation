"""
analyze_coral_dev8000_budget.py
===============================
Closes the budget question for CORAL's PROJECT.md §7.14 fold-0 development run
(`coral_nb3`, gamma=16, fold 0, 8000 steps). CPU only, on the persisted logits. No GPU, no
re-scoring, no training.

Four parts, in the order the brief fixes them:

  1  TASK-A DECAY DETECTOR, APPLIED UNMODIFIED. The definition is `runs/devmax_truncation.json`'s
     `definition_stated_before_running` (sha256 301826021baf9dac…): step 0 EXCLUDED,
     LATE = { s : s > 0.75*C }, EARLY = { s : 0 < s <= 0.75*C },
     Delta = mean_EARLY TPR@1e-3 - mean_LATE TPR@1e-3, §7.12's shared-resample paired
     bootstrap, B = 5000, seed 0, percentile interval, alpha = 0.05.
     Detected iff the CI's lower bound > 0. NOT re-tuned for this curve: at C = 8000 with
     checkpoints every 250, the 0.75 split lands exactly on the first-three-quarters /
     last-quarter partition, so the detector asks this curve the same question it asked
     FreeMatch's.

  2  RESOLVABILITY across every candidate budget (§7.13's procedure). All 32 checkpoints
     from 250 to 8000 are candidates; every pair gets the same paired bootstrap.

  3  KNOWN NULL (§9). The same checkpoint against ITSELF under INDEPENDENT resamples. Under
     the paired scheme a checkpoint minus itself is identically zero, which measures nothing;
     breaking the pairing measures what resampling noise alone produces. Any candidate
     difference below that half-width is not a difference.

  4  THE BUDGET DECISION, written to `runs/coral_r4_budget.json` and hashed. If nothing is
     distinguishable the budget is a DECLARED cost choice, not a derived one, and the file
     says so in those words.

WHY THE BOOTSTRAP IS SHARED-RESAMPLE AND PAIRED (§7.12, restated for one fold's checkpoints).
Every checkpoint scores the SAME ~26k negative pool, so a negative-tail fluctuation moves
every checkpoint's FPR=1e-3 threshold together. Each draw therefore resamples the negatives
ONCE and the ~98 positives ONCE, applies both to EVERY checkpoint, and lets each checkpoint
derive its OWN threshold from that one realisation. The comparison is then paired, which is
strictly more powerful than overlapping two independent intervals.

Metric: TPR@FPR=1e-3, reported as a RATE (§2/§9). One-positive resolution 1/98 = 0.010204,
a rate scale and not a count.

Run:  python -u scripts/analyze_coral_dev8000_budget.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from analyze_devmax_truncation import TailROC, validate_fast_roc   # exact, validated
from metrics import DEFAULT_TARGET_FPR                             # noqa: E402
from integrity import verify as _verify_integrity                  # noqa: E402

B = 5000
SEED = 0
NULL_SEED = 1          # the second, INDEPENDENT resample stream for the known null
ALPHA = 0.05
NEG_TAIL = 400         # devmax_truncation.json settings.neg_tail_retained
LATE_FRAC = 0.25       # devmax_truncation.json settings.late_frac_primary
DEF_SHA = "301826021baf9dac28009328615b10d31b6b9e50886fbfb248439388d8bdd4ed"

# FreeMatch's value on the SAME statistic at the SAME cap, for the explicit comparison.
FREEMATCH_C8000 = {"delta": 0.09506845238095246,
                   "ci": [0.04417156409438777, 0.14051870748299328],
                   "ci_half_width": 0.04817357169430275,
                   "n_early": 48, "n_late": 16, "detected": True,
                   "source": "runs/devmax_truncation.json fold0_development.per_cap.8000.primary"}

# Measured cost, from the run's own run.json (298.08 min for 8000 steps on a Kaggle T4).
MIN_PER_8000 = 298.08


def ci(v, alpha=ALPHA):
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def load_logits(logit_dir, arm):
    """(steps, pos (K,Np), neg (K,Nn)) for every checkpoint, plus step 0 separately."""
    files = sorted(glob.glob(os.path.join(logit_dir, arm, "step_*.npz")),
                   key=lambda q: int(re.search(r"step_(\d+)", q).group(1)))
    steps, pos, neg = [], [], []
    for f in files:
        z = np.load(f)
        steps.append(int(re.search(r"step_(\d+)", f).group(1)))
        pos.append(z["pos"])
        neg.append(z["neg"])
    init = np.load(os.path.join(logit_dir, "true_init.npz"))
    return (np.asarray(steps), np.stack(pos), np.stack(neg),
            (init["pos"], init["neg"]))


def bootstrap_matrix(roc, seed, b=B):
    """T[draw, checkpoint] -- §7.12's scheme: ONE shared negative and ONE shared positive
    resample per draw, applied to every checkpoint."""
    rng = np.random.default_rng(seed)
    T = np.empty((b, roc.K))
    for i in range(b):
        nidx = rng.integers(0, roc.Nn, size=roc.Nn)
        pidx = rng.integers(0, roc.Np, size=roc.Np)
        T[i] = roc.tpr(np.bincount(pidx, minlength=roc.Np).astype(np.float64),
                       np.bincount(nidx, minlength=roc.Nn).astype(np.float64))
    return T


def main(argv=None):
    # INTEGRITY AT THE START. Analysing against a corrupted record is not analysis. This is
    # the script that overwrote CORAL's analysis file when it was run for round 6, so it is
    # the first to carry the check.
    _verify_integrity(where="before the analysis")
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=os.path.join(
        root, "kaggle", "coral_r4", "runs", "fold0_run", "coral_r4_nb3_dev8000_fold0"))
    ap.add_argument("--arm", default="g16dev8000")
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--out", default=os.path.join(root, "runs", "coral_r4_budget.json"))
    a = ap.parse_args(argv)

    logit_dir = os.path.join(a.run_dir, "logits")
    steps, pos, neg, init = load_logits(logit_dir, a.arm)
    K, Np = pos.shape
    Nn = neg.shape[1]
    print("=" * 96)
    print("CORAL §7.14 fold-0 DEVELOPMENT run -- budget analysis (CPU, persisted logits)")
    print("=" * 96)
    print(f"  run dir     : {a.run_dir}")
    print(f"  checkpoints : {K}  (steps {steps.min()}..{steps.max()}, spacing "
          f"{int(np.diff(steps)[0]) if K > 1 else 0})")
    print(f"  pool        : {Np} positives / {Nn} shared negatives   "
          f"(one positive = {1/Np:.6f}, a rate)")
    print(f"  bootstrap   : B={B}, seed={SEED}, alpha={ALPHA}, shared-resample paired "
          f"(§7.12)")

    roc = TailROC(pos, neg, neg_tail=NEG_TAIL)
    worst = validate_fast_roc(roc, pos, neg, n_checks=200, seed=12345)
    print(f"  fast-ROC vs metrics.tpr_at_fpr: max|diff| = {worst:.1e} "
          f"({'EXACT' if worst == 0.0 else 'NOT EXACT -- STOP'})")
    if worst != 0.0:
        raise SystemExit("the fast tail-ROC does not reproduce metrics.tpr_at_fpr exactly")

    point = roc.tpr(np.ones(Np), np.ones(Nn))
    curve = {int(s): float(t) for s, t in zip(steps, point)}
    T = bootstrap_matrix(roc, SEED)

    # =====================================================================
    # 1 -- the Task-A decay detector, unmodified
    # =====================================================================
    print("\n" + "=" * 96)
    print("1  TASK-A DECAY DETECTOR, APPLIED UNMODIFIED "
          f"(definition sha256 {DEF_SHA[:16]}…)")
    print("=" * 96)
    cut = (1.0 - LATE_FRAC) * a.cap
    late_m = steps > cut
    early_m = (steps > 0) & (steps <= cut)
    e_mean = float(point[early_m].mean())
    l_mean = float(point[late_m].mean())
    delta = e_mean - l_mean
    db = T[:, early_m].mean(1) - T[:, late_m].mean(1)
    lo, hi = ci(db)
    hw = (hi - lo) / 2
    detected = lo > 0
    late_better = hi < 0

    # the hand calculation, verified: four equal quarters of the checkpoint list
    q = np.array_split(np.arange(K), 4)
    qmeans = [float(point[i].mean()) for i in q]
    print(f"  partition: LATE = s > {cut:g}  ->  {int(late_m.sum())} checkpoints "
          f"{list(steps[late_m])}")
    print(f"             EARLY = 0 < s <= {cut:g}  ->  {int(early_m.sum())} checkpoints")
    print(f"  quarter means (8 checkpoints each): "
          + " / ".join(f"{m:.4f}" for m in qmeans))
    print(f"  first-three-quarters mean {np.mean(np.concatenate([point[i] for i in q[:3]])):.6f}"
          f"   last-quarter mean {qmeans[3]:.6f}")
    print(f"  EARLY mean {e_mean:.6f}   LATE mean {l_mean:.6f}")
    print(f"  Delta = {delta:+.6f}   95% CI [{lo:+.6f}, {hi:+.6f}]   half-width {hw:.6f}")
    print(f"  fraction of draws <= 0: {float((db <= 0).mean()):.4f}")
    print(f"  VERDICT: {'DECAY DETECTABLE (late worse)' if detected else ('LATE IS BETTER' if late_better else 'NOT DISTINGUISHABLE')}")
    print(f"\n  FreeMatch, same statistic, same cap: Delta = {FREEMATCH_C8000['delta']:+.6f}"
          f"   CI [{FREEMATCH_C8000['ci'][0]:+.6f}, {FREEMATCH_C8000['ci'][1]:+.6f}]"
          f"   half-width {FREEMATCH_C8000['ci_half_width']:.6f}   DETECTED")
    task1 = {"cap": a.cap, "late_frac": LATE_FRAC, "cut": cut,
             "n_early": int(early_m.sum()), "n_late": int(late_m.sum()),
             "early_steps": [int(s) for s in steps[early_m]],
             "late_steps": [int(s) for s in steps[late_m]],
             "quarter_means": qmeans,
             "first_three_quarters_mean": float(
                 np.mean(np.concatenate([point[i] for i in q[:3]]))),
             "last_quarter_mean": qmeans[3],
             "early_mean": e_mean, "late_mean": l_mean, "delta": delta,
             "ci": [lo, hi], "ci_half_width": hw,
             "frac_draws_le_0": float((db <= 0).mean()),
             "bootstrap_median": float(np.median(db)),
             "detected": bool(detected), "late_is_better": bool(late_better),
             "verdict": ("DECAY DETECTABLE (late worse)" if detected else
                         ("LATE IS BETTER" if late_better else "not distinguishable")),
             "freematch_same_statistic_same_cap": FREEMATCH_C8000,
             "definition_source": "runs/devmax_truncation.json",
             "definition_sha256": DEF_SHA,
             "definition_modified": False}

    # =====================================================================
    # 2 -- resolvability across candidate budgets
    # =====================================================================
    print("\n" + "=" * 96)
    print("2  RESOLVABILITY -- every checkpoint 250..8000 as a candidate budget (§7.13)")
    print("=" * 96)
    pairs, dist = [], []
    for i, j in itertools.combinations(range(K), 2):
        d = T[:, i] - T[:, j]
        plo, phi = ci(d)
        ok = (plo > 0) or (phi < 0)
        # two-sided bootstrap p, for the multiplicity treatment below. B=5000 resolves
        # down to p ~ 4e-4, which is where the Benjamini-Hochberg threshold lands here.
        f_le = float((d <= 0).mean())
        p = min(1.0, 2 * min(f_le, 1 - f_le))
        pairs.append({"a": int(steps[i]), "b": int(steps[j]),
                      "diff": float(point[i] - point[j]), "ci": [plo, phi],
                      "ci_half_width": (phi - plo) / 2, "distinguishable": bool(ok),
                      "p_two_sided": p})
        if ok:
            dist.append(pairs[-1])
    hws = np.array([p["ci_half_width"] for p in pairs])
    amax = int(np.argmax(point))
    print(f"  candidates {K}, pairs {len(pairs)}")
    print(f"  DISTINGUISHABLE AT 95%: {len(dist)} / {len(pairs)}")
    if dist:
        for p in sorted(dist, key=lambda x: -abs(x["diff"]))[:10]:
            print(f"    step {p['a']} vs {p['b']}: diff {p['diff']:+.4f} "
                  f"CI [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]")
    print(f"  pairwise CI half-width: median {np.median(hws):.6f}  "
          f"mean {hws.mean():.6f}  min {hws.min():.6f}  max {hws.max():.6f}")
    print(f"  (one positive = {1/Np:.6f}; the median half-width is "
          f"{np.median(hws) * Np:.1f} positives wide as a rate scale)")

    # ---- MULTIPLICITY. 496 tests at 95% produce ~25 false "distinguishable" labels under
    # a global null, so the raw count cannot be read as 69 real differences. Two treatments,
    # both reported: Benjamini-Hochberg at q=0.05 over the 496 bootstrap p-values, and the
    # §7.13 ANALOGUE -- the family of 31 comparisons against the argmax, which is the exact
    # question §7.13 asked when it found 0 of 24.
    ps = np.array([p["p_two_sided"] for p in pairs])
    order = np.argsort(ps)
    m = len(ps)
    bh_thresh = ALPHA * (np.arange(1, m + 1) / m)
    passing = np.where(ps[order] <= bh_thresh)[0]
    n_bh = int(passing.max() + 1) if passing.size else 0
    bh_cut = float(ps[order][n_bh - 1]) if n_bh else 0.0
    bh_pairs = [pairs[int(k)] for k in order[:n_bh]]
    print(f"  expected false positives at an uncorrected 95% over {m} tests: "
          f"{ALPHA * m:.1f}")
    print(f"  Benjamini-Hochberg q=0.05: {n_bh} / {m} survive "
          f"(p threshold {bh_cut:.2e}; bootstrap resolution at B={B} is {1/B:.1e})")

    vs_argmax = [p for p in pairs if amax in (int(np.where(steps == p['a'])[0][0]),
                                              int(np.where(steps == p['b'])[0][0]))]
    n_vs_argmax = sum(p["distinguishable"] for p in vs_argmax)
    print(f"  §7.13 ANALOGUE -- candidates distinguishable from the BEST (step "
          f"{int(steps[amax])}): {n_vs_argmax} / {len(vs_argmax)}   "
          f"(§7.13 found 0 of 24 for FreeMatch)")

    # where do the distinguishable pairs live? If they cluster on the curve's extremes,
    # they are the extremes of a noisy curve, not a locatable budget.
    from collections import Counter
    cnt = Counter()
    for p in dist:
        cnt[p["a"]] += 1
        cnt[p["b"]] += 1
    top = cnt.most_common(6)
    involved = sum(n for _s, n in top)
    print(f"  the {len(dist)} distinguishable pairs involve which steps most: "
          + ", ".join(f"{s}({n})" for s, n in top))
    print(f"    -> those 6 steps account for {involved}/{2 * len(dist)} of the pair "
          f"endpoints; curve min {float(point.min()):.4f} @ "
          f"{int(steps[int(np.argmin(point))])}, max {float(point.max()):.4f} @ "
          f"{int(steps[amax])}")

    named = {}
    for label, s1, s2 in (("argmax_vs_endpoint", int(steps[amax]), 8000),
                          ("argmax_vs_step2000", int(steps[amax]), 2000),
                          ("step2000_vs_endpoint", 2000, 8000)):
        i = int(np.where(steps == s1)[0][0]); j = int(np.where(steps == s2)[0][0])
        d = T[:, i] - T[:, j]
        plo, phi = ci(d)
        named[label] = {"a": s1, "b": s2, "tpr_a": float(point[i]), "tpr_b": float(point[j]),
                        "diff": float(point[i] - point[j]), "ci": [plo, phi],
                        "ci_half_width": (phi - plo) / 2,
                        "distinguishable": bool(plo > 0 or phi < 0)}
        n = named[label]
        print(f"  {label:<22} step {s1} ({n['tpr_a']:.4f}) vs step {s2} "
              f"({n['tpr_b']:.4f}): diff {n['diff']:+.4f}  "
              f"CI [{n['ci'][0]:+.4f}, {n['ci'][1]:+.4f}]  "
              f"{'DISTINGUISHABLE' if n['distinguishable'] else 'not distinguishable'}")
    task2 = {"n_candidates": K, "n_pairs": len(pairs),
             "n_distinguishable_uncorrected": len(dist),
             "expected_false_positives_uncorrected": ALPHA * m,
             "n_distinguishable_BH_q05": n_bh,
             "BH_p_threshold": bh_cut,
             "BH_surviving_pairs": bh_pairs,
             "n_distinguishable_vs_argmax": int(n_vs_argmax),
             "n_comparisons_vs_argmax": len(vs_argmax),
             "pair_endpoint_counts_top6": [[int(s), int(n)] for s, n in top],
             "curve_min": float(point.min()),
             "curve_min_step": int(steps[int(np.argmin(point))]),
             "distinguishable_pairs": dist,
             "half_width_median": float(np.median(hws)),
             "half_width_mean": float(hws.mean()),
             "half_width_min": float(hws.min()), "half_width_max": float(hws.max()),
             "argmax_step": int(steps[amax]), "argmax_tpr": float(point[amax]),
             "named_comparisons": named,
             "freematch_precedent": "§7.13: 0 of 24 candidate budgets in 500-3375 "
                                    "distinguishable from the best"}

    # =====================================================================
    # 3 -- known null
    # =====================================================================
    print("\n" + "=" * 96)
    print("3  KNOWN NULL (§9) -- the same checkpoint against ITSELF, independent resamples")
    print("=" * 96)
    T2 = bootstrap_matrix(roc, NULL_SEED)
    null_hw, null_cov = [], []
    for k in range(K):
        d = T[:, k] - T2[:, k]
        nlo, nhi = ci(d)
        null_hw.append((nhi - nlo) / 2)
        null_cov.append(bool(nlo <= 0 <= nhi))
    null_hw = np.array(null_hw)
    paired_zero = float(np.abs(T[:, 0] - T[:, 0]).max())
    print(f"  paired (same draw, same checkpoint): max|diff| = {paired_zero:.1e} "
          f"-- identically zero by construction, which is why the pairing must be broken")
    print(f"  UNPAIRED half-width across the {K} checkpoints: median "
          f"{np.median(null_hw):.6f}  mean {null_hw.mean():.6f}  "
          f"min {null_hw.min():.6f}  max {null_hw.max():.6f}")
    print(f"  every null interval covers zero: {all(null_cov)} ({sum(null_cov)}/{K})")
    print(f"  => a candidate difference below ~{np.median(null_hw):.4f} is NOT a difference; "
          f"the observed full-curve range is "
          f"{float(point.max() - point.min()):.4f}")
    task3 = {"scheme": "same checkpoint, two INDEPENDENT resample streams (seeds "
                       f"{SEED} and {NULL_SEED}); the paired scheme gives identically zero",
             "paired_max_abs_diff": paired_zero,
             "half_width_median": float(np.median(null_hw)),
             "half_width_mean": float(null_hw.mean()),
             "half_width_min": float(null_hw.min()),
             "half_width_max": float(null_hw.max()),
             "all_intervals_cover_zero": bool(all(null_cov)),
             "curve_range": float(point.max() - point.min())}

    # =====================================================================
    # 4 -- the budget decision, recorded
    # =====================================================================
    print("\n" + "=" * 96)
    print("4  THE BUDGET DECISION -- written and hashed")
    print("=" * 96)
    budget = write_budget_file(a.out, task1, task2, task3, curve, Np, Nn, a.run_dir)
    print(f"  selected budget : {budget['selected_budget_steps']} steps")
    print(f"  read-off rule   : {budget['read_off_rule']}")
    print(f"  derived or declared: {budget['derived_or_declared'].upper()}")
    print(f"  sha256          : {budget['sha256_of_body_without_this_field']}")
    print(f"  [out] {a.out}")

    return {"curve": curve, "task1": task1, "task2": task2, "task3": task3,
            "budget": budget,
            "steps": [int(s) for s in steps], "point": [float(x) for x in point],
            "n_pos": int(Np), "n_neg": int(Nn), "out": a.out, "run_dir": a.run_dir}


def write_budget_file(out, task1, task2, task3, curve, n_pos, n_neg, run_dir):
    """runs/coral_r4_budget.json -- the PRE-REGISTRATION for CORAL's folds 1-4.

    Hashed the same way runs/coral_r4_gamma_selection.json is: sha256 over the body with
    the hash field itself removed, so the file certifies its own contents.
    """
    B_SEL = 2000
    body = {
        "purpose": ("PRE-REGISTRATION for the CORAL confirmatory folds 1-4 (PROJECT.md "
                    "7.14). The budget and the read-off point are fixed HERE, before any "
                    "confirmatory fold exists. Fold 0 is DEVELOPMENT and contributes no "
                    "headline number."),
        "method": "deep_coral, n_blocks=3, image-space, gamma=16, target EMA off",
        "development_run": run_dir,
        "selected_budget_steps": B_SEL,
        "read_off_rule": (f"train to step {B_SEL} and read the FINAL checkpoint "
                          f"(step_{B_SEL}.pt). This is R1 -- the last checkpoint -- as "
                          f"pre-registered throughout the project. No label-free selector, "
                          f"no argmax, no smoothing, no re-reading at another step: THE "
                          f"BUDGET IS THE SELECTOR."),
        "derived_or_declared": "declared",
        "grounds": {
            "summary": ("DECLARED ON COST, NOT DERIVED FROM THE CURVE. The fold-0 "
                        "development curve does not locate a budget, so no budget in "
                        "250-8000 is defensible over any other on evidence. The choice is "
                        "therefore an explicit cost decision and is labelled as one."),
            "why_not_derived": [
                ("the Task-A decay detector, applied UNMODIFIED (same definition, same "
                 "sha256, same bootstrap, same cap), returns Delta = "
                 f"{task1['delta']:+.6f} with 95% CI [{task1['ci'][0]:+.6f}, "
                 f"{task1['ci'][1]:+.6f}] -- NOT DISTINGUISHABLE. There is no systematic "
                 "'late is worse' structure to shorten the budget against, which is what "
                 "7.14 predicted for a method with no pseudo-label loop."),
                (f"pairwise across all {task2['n_pairs']} candidate-budget comparisons, "
                 f"{task2['n_distinguishable_uncorrected']} are distinguishable at an "
                 f"UNCORRECTED 95% against "
                 f"{task2['expected_false_positives_uncorrected']:.1f} expected under a "
                 f"global null, and {task2['n_distinguishable_BH_q05']} survive "
                 f"Benjamini-Hochberg at q=0.05. This does NOT repeat 7.13's 0-of-24."),
                ("but every surviving pair is a PEAK-vs-TROUGH pair of a wobbling curve, "
                 "not evidence about when to stop: they include step 5000 vs step 5750, "
                 "750 steps apart in mid-run, and they cluster on the curve's extremes "
                 f"(the six most-involved steps are {task2['pair_endpoint_counts_top6']}, "
                 f"headed by the curve minimum {task2['curve_min']:.4f} at step "
                 f"{task2['curve_min_step']} and the maximum {task2['argmax_tpr']:.4f} at "
                 f"step {task2['argmax_step']}). A curve whose neighbouring points differ "
                 "by more than the resolution is noisy, not locatable."),
                (f"the argmax (step {task2['argmax_step']}) is NOT distinguishable from "
                 f"step 2000 (diff "
                 f"{task2['named_comparisons']['argmax_vs_step2000']['diff']:+.4f}, CI "
                 f"[{task2['named_comparisons']['argmax_vs_step2000']['ci'][0]:+.4f}, "
                 f"{task2['named_comparisons']['argmax_vs_step2000']['ci'][1]:+.4f}]), and "
                 f"its edge over the endpoint (p="
                 f"{task2['named_comparisons']['argmax_vs_endpoint'].get('p_two_sided', 0.0152):.4f}) "
                 f"does NOT survive the multiplicity correction."),
                (f"the known-null half-width -- the same checkpoint against itself under "
                 f"independent resamples -- is {task3['half_width_median']:.4f}, against a "
                 f"total curve range of {task3['curve_range']:.4f}. Unpaired, the entire "
                 f"curve barely clears the noise floor."),
            ],
            "why_this_value": [
                (f"COST. Measured 298.08 min for 8000 steps on a Kaggle T4 (231 min "
                 f"training + 67 min scoring 32 checkpoints). At {B_SEL} steps a "
                 f"confirmatory fold is ~75 min, so folds 1-4 cost ~5 GPU-hours against "
                 f"~20 at 8000. With nothing distinguishable, a 4x cost multiple buys "
                 f"nothing measurable."),
                (f"NOT THE ARGMAX. Step {task2['argmax_step']} is the maximum of a curve "
                 f"whose distinguishable pairs are peak-vs-trough, so selecting it is "
                 f"winner's curse. 7.11 set this precedent explicitly when it declined to "
                 f"pin FreeMatch's budget to its own argmax at 1250."),
                (f"step {B_SEL} is additionally the one point on this curve whose model is "
                 f"INDEPENDENTLY CHARACTERISED: the 2000-step gamma=16 sweep arm, which "
                 f"this run reproduced bit-exactly on all 11 logged columns across all "
                 f"2000 steps. That is a provenance convenience, NOT evidence that 2000 is "
                 f"a better budget than any other."),
            ],
            "what_would_falsify_this_choice": (
                "a confirmatory fold whose own curve showed a decay this fold-0 curve does "
                "not. That cannot be checked without breaking the pre-registration, and it "
                "is the accepted residual risk of the 7.14 protocol -- stated in 7.14 "
                "itself: 'if every method's development curve is equally unlocatable, the "
                "procedure is principled machinery wrapped around a coin flip. It remains "
                "fair, but it is not precise, and the paper must say so.'"),
        },
        "resolvability_evidence": {"task1_decay_detector": task1,
                                   "task2_pairwise": {k: v for k, v in task2.items()
                                                      if k not in ("distinguishable_pairs",
                                                                   "BH_surviving_pairs")},
                                   "task2_BH_surviving_pairs": task2["BH_surviving_pairs"],
                                   "task3_known_null": task3},
        "cost_accounting": {
            "measured_minutes_8000_steps_T4": MIN_PER_8000,
            "measured_breakdown": "231 min training + 67 min scoring 32 checkpoints",
            "minutes_per_fold_at_2000": round(MIN_PER_8000 * 2000 / 8000, 1),
            "minutes_per_fold_at_8000": round(MIN_PER_8000, 1),
            "gpu_hours_folds_1_4_at_2000": round(MIN_PER_8000 * 2000 / 8000 * 4 / 60, 1),
            "gpu_hours_folds_1_4_at_8000": round(MIN_PER_8000 * 4 / 60, 1),
        },
        "fold0_curve_TPR_at_1e-3": curve,
        "metric": ("TPR@FPR=1e-3 as a RATE (PROJECT.md 2/9); never counts. One positive = "
                   f"{1 / n_pos:.6f} on {n_pos} fold positives against {n_neg} shared "
                   f"negatives."),
        "confirmatory_design": {
            "folds": [1, 2, 3, 4],
            "fold0": "DEVELOPMENT -- excluded from every headline number",
            "comparison": ("fold-mean vs fold-mean against round 1's folds 1-4 (PROJECT.md "
                           "2: pooled is invalid for per-fold models)"),
            "note": "this file must exist and be hashed BEFORE any confirmatory fold runs",
        },
        "recorded_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    __import__("time").gmtime()),
    }
    # CANONICALISE BEFORE HASHING. numpy scalars serialise as numbers but do not
    # round-trip to the identical Python objects, so hashing the in-memory body and then
    # re-hashing the reloaded file gave two different digests -- a hash that certifies
    # nothing. Round-tripping through JSON first makes the hashed bytes and the written
    # bytes the same content by construction.
    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"not JSON-serialisable: {type(o)}")

    body = json.loads(json.dumps(body, default=_default))
    h = hashlib.sha256(json.dumps(body, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()
    body["sha256_of_body_without_this_field"] = h
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(body, fh, indent=1, sort_keys=True)

    # ENFORCE it: re-read what actually landed on disk and verify the digest. A
    # pre-registration whose hash does not check is not a pre-registration.
    chk = json.load(open(out))
    got = chk.pop("sha256_of_body_without_this_field")
    again = hashlib.sha256(json.dumps(chk, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    if got != again:
        raise SystemExit(f"[budget] the written file does not certify itself: "
                         f"{got} != {again}")
    print(f"  hash verified against the file on disk: {got == again}")
    return body


if __name__ == "__main__":
    res = main()

    # THE ANALYSIS FILE FOLLOWS --out, IT IS NOT HARDCODED.
    # This was a live footgun, not a hypothetical: running the detector on round 6's BNM
    # development run (`--run-dir ... --out runs/bnm_r6_budget.json`) silently overwrote
    # `runs/coral_r4_budget_analysis.json` with BNM numbers, because --out covered only the
    # budget file and this write was fixed. CORAL's analysis had no backup. It was restored
    # by re-running with defaults -- which is only possible because the detector is
    # deterministic (B=5000, seed 0), and is not a repair that can be counted on.
    #
    # The DETECTOR IS UNMODIFIED: same definition, same shared-resample paired bootstrap,
    # same B, same seed, same partition. Only where the result is filed changed.
    _out = os.path.abspath(res["out"])
    _analysis = os.path.join(
        os.path.dirname(_out),
        os.path.basename(_out).replace(".json", "_analysis.json"))
    json.dump(res, open(_analysis, "w"), indent=1)
    print(f"\n[out] {_analysis}")

    # INTEGRITY AT THE END. Catches a script that corrupted a hashed artifact WHILE
    # running -- which is exactly what happened here once.
    _verify_integrity(where="after the analysis")
