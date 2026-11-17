"""
analyze_lambda0_r6.py
=====================
ROUND 6 RESULTS -- the lambda=0 unfrozen source-only baseline, folds 1-4.

Reads ONLY what ``runs/lambda0_r6_prereg.json`` declared before any fold ran, and reports
in the form that file fixed:

  * primary TPR@FPR=1e-3, interpolated, ``drop_intermediate=False``, ALWAYS a rate;
  * the declared budget 4000, plus the PRE-DECLARED sensitivity readouts at 1500 and
    2000 -- reported alongside the headline, never replacing it, whichever is higher;
  * secondary completeness@24%-purity (Lines' operating point) and
    purity@50%-completeness at a FIXED operating point; AUPRC is never a selection
    proxy; AUROC is a saturating sanity metric;
  * per-fold ROC curves -- POOLING IS INVALID for per-fold-model arms;
  * paired **t-interval on 4 differences, df=3** -- NOT a bootstrap: at n=4 a bootstrap
    draws from 4^4 = 256 distinct resamples and its tails are a lattice artifact;
  * undefined values as n/a with NO CI;
  * no number differenced against any round-1 number.

THE FROZEN COMPARATOR IS THE STEP-0 GATE, and that is not a shortcut. The gate scores
the phase-1 head on each fold's test set at step 0, when the encoder is still the
pretrained one -- so it IS frozen Baseline A on that fold, measured inside this very run,
on the same images, through the same eval path. 7.18 uses it that way (fold 0's arm 1
"ends at 0.1531, below the frozen LP->FT head it started from", 0.1939 being fold 0's
gate). It makes lambda=0 - Baseline A the most tightly paired contrast in the project:
same fold, same head, same test set, same eval code, and 4000 steps of unfrozen training
as the only difference.

    python -u scripts/analyze_lambda0_r6.py --out runs/lambda0_r6_results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as me  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "runs", "baseline_unfrozen")
PREREG = os.path.join(REPO, "runs", "lambda0_r6_prereg.json")

FOLDS = (1, 2, 3, 4)
BUDGET = 4000
SENSITIVITY = (1500, 2000)
CKPT_EVERY = 250
TARGET_FPR = 1e-3

# Declared budgets of the three DA arms, and their per-fold TPR@1e-3 on folds 1-4.
#
# THESE ARE COMPUTED FROM EACH ARM'S OWN LOGITS, NOT TYPED IN -- corrected 2026-09-19.
# They used to be literals transcribed from ``runs/dann_main/paired_analysis.json``, and
# that file stores CORAL rounded to 4 dp. The DANN and FreeMatch rows happened to be
# copied at full precision; the CORAL row was not, so one row of one table silently
# carried 4-dp values. It moved the published CORAL - lambda=0 (budget-unmatched) lower
# bound by one unit in the fourth decimal, +0.1385 -> +0.1386. Nothing here is a new
# measurement: the two rows that were already exact reproduce bit-for-bit.
#
# The logit locations come from ``analyze_window_average.ARMS``, the project's single
# source of truth for where each arm's per-checkpoint logits live, so a moved artifact
# cannot make this table and the round-7 analyses disagree.
_ARM_READOUT = {"DANN x nb3 (4000)": ("DANN x nb3", 4000),
                "CORAL x nb3 (2000)": ("CORAL x nb3", 2000),
                "FreeMatch x nb3 (1500)": ("FreeMatch x nb3", 1500)}
_COMPARATOR_CACHE: dict | None = None


def _build_comparators() -> dict:
    """Per-fold TPR@1e-3 for the three DA arms and for DANN at its 2000 readout."""
    # LAZY, and the laziness is load-bearing: ``analyze_window_average`` imports
    # ``paired_t`` from this module at ITS top, so importing it at OUR top is a cycle
    # that breaks whichever module is imported second. Deferring to first use means
    # both import orders work, and it is verified in both directions.
    from analyze_window_average import ARMS

    def tpr(arm: str, step: int) -> list:
        out = []
        for fold in FOLDS:
            pos, neg = ARMS[arm]["load"](fold)[step]
            y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
            s = np.concatenate([pos, neg])
            out.append(float(me.metric_suite(y, s, target_fpr=TARGET_FPR)["tpr"]))
        return out

    return {"COMPARATORS": {name: tpr(arm, step)
                            for name, (arm, step) in _ARM_READOUT.items()},
            "DANN_AT_2000": tpr("DANN x nb3", 2000)}


def _comparator_cache() -> dict:
    global _COMPARATOR_CACHE
    if _COMPARATOR_CACHE is None:
        _COMPARATOR_CACHE = _build_comparators()
    return _COMPARATOR_CACHE


def comparators() -> dict:
    """{arm label: per-fold TPR@1e-3 on folds 1-4}, computed from the arms' logits."""
    return _comparator_cache()["COMPARATORS"]


def dann_at_2000() -> list:
    """DANN x nb3's per-fold TPR@1e-3 at its pre-declared 2000 sensitivity readout."""
    return _comparator_cache()["DANN_AT_2000"]


def __getattr__(name):
    """Serve ``COMPARATORS`` and ``DANN_AT_2000`` on first access (PEP 562).

    Module-level ``__getattr__`` is what lets the values stay lazy while every existing
    IMPORTER -- ``from analyze_lambda0_r6 import COMPARATORS`` in
    ``analyze_bnm_r6_folds`` -- keeps working unchanged. It does NOT cover this module's
    own global lookups, which go through ``comparators()`` / ``dann_at_2000()`` instead;
    that asymmetry is a property of PEP 562 and is why both forms exist.
    """
    if name in ("COMPARATORS", "DANN_AT_2000"):
        return _comparator_cache()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Student t, two-sided 95%, df=3. Hard-coded so the interval never silently becomes a
# normal approximation if scipy is absent.
T95_DF3 = 3.182446305284263


def run_dir(fold: int) -> str:
    return os.path.join(RESULTS, f"fold{fold}", "extracted", f"lambda0_r6_fold{fold}")


def logit_path(fold: int, step: int) -> str:
    return os.path.join(run_dir(fold), "logits",
                        f"dann_nb3_lambda0_fold{fold}", f"step_{step}.npz")


def score(fold: int, step: int) -> dict:
    z = np.load(logit_path(fold, step))
    pos, neg = np.asarray(z["pos"], float), np.asarray(z["neg"], float)
    y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
    s = np.concatenate([pos, neg])
    suite = me.metric_suite(y, s, target_fpr=TARGET_FPR)
    suite["n_pos"], suite["n_neg"] = int(pos.size), int(neg.size)
    return suite


def paired_t(diffs) -> dict:
    """Paired t-interval on n differences, df = n-1. The pre-registered form."""
    d = [float(x) for x in diffs]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    half = T95_DF3 * se if n == 4 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "se": se,
            "ci_lo": m - half, "ci_hi": m + half,
            "half_width": half,
            "excludes_zero": bool((m - half) * (m + half) > 0),
            "all_same_sign": bool(all(x > 0 for x in d) or all(x < 0 for x in d)),
            "diffs": d}


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "lambda0_r6_results.json"))
    args = ap.parse_args()

    with open(PREREG, encoding="utf-8") as fh:
        pre = json.load(fh)

    out = {"round": 6, "arm": "dann_nb3_lambda0", "folds": list(FOLDS),
           "budget": BUDGET, "sensitivity_steps": list(SENSITIVITY),
           "prereg_sha256": pre["sha256_of_body_without_this_field"]}

    line = "=" * 82
    print(line)
    print("ROUND 6 RESULTS -- lambda=0 UNFROZEN SOURCE-ONLY BASELINE, folds 1-4")
    print(line)
    print(f"pre-registration {out['prereg_sha256'][:32]}...  budget {BUDGET} "
          f"(declared before any fold ran)")

    # ---- provenance ------------------------------------------------------
    prov, frozen = {}, {}
    for k in FOLDS:
        with open(os.path.join(run_dir(k), "run_config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        g = cfg["step0_gate"]
        prov[k] = {"steps": cfg["steps_completed"], "ckpts": cfg["n_checkpoints_saved"],
                   "gate": g["verdict"], "tpr_abs_diff": g["tpr_abs_diff"],
                   "auroc_abs_diff": g["auroc_abs_diff"],
                   "wall_minutes": cfg["wall_minutes"], "gpu": cfg["gpu_name"],
                   "resumes": len(cfg.get("resume_events") or []),
                   "abort_mode": cfg["abort_mode"], "aborted": cfg["aborted"],
                   "n_test_pos": cfg["n_test_positives"]}
        frozen[k] = float(cfg["expect_step0_tpr"])
    out["provenance"] = prov
    out["baseline_A_frozen_per_fold"] = frozen

    print("\nPROVENANCE")
    print(f"  {'fold':<6}{'steps':>10}{'ckpt':>7}{'gate':>7}{'wall':>8}{'GPU':>10}"
          f"{'resume':>8}{'abort_mode':>12}")
    for k in FOLDS:
        p = prov[k]
        print(f"  {k:<6}{str(p['steps'])+'/'+str(BUDGET):>10}{str(p['ckpts'])+'/16':>7}"
              f"{p['gate']:>7}{p['wall_minutes']:>8.1f}{p['gpu']:>10}"
              f"{p['resumes']:>8}{p['abort_mode']:>12}")
    w = [prov[k]["wall_minutes"] for k in FOLDS]
    print(f"  wall: mean {st.mean(w):.1f} min, range {min(w):.1f}-{max(w):.1f}; "
          f"every gate PASS with TPR diff exactly 0.0")

    # ---- the metric suite at the budget and both readouts -----------------
    per_step = {}
    for step in (BUDGET,) + SENSITIVITY:
        per_step[step] = {k: score(k, step) for k in FOLDS}

    def table(step, key, nd=4):
        return [per_step[step][k][key] for k in FOLDS]

    out["per_fold"] = {str(s): {str(k): per_step[s][k] for k in FOLDS} for s in per_step}

    print(f"\nPRIMARY -- TPR@FPR=1e-3 (rate; quantisation 1/98 on folds 1-3, 1/97 on fold 4)")
    print(f"  {'read-off':<22}" + "".join(f"{'fold '+str(k):>10}" for k in FOLDS)
          + f"{'mean':>10}{'sd':>9}")
    for step in (BUDGET,) + SENSITIVITY:
        v = table(step, "tpr")
        tag = f"{step}  (DECLARED)" if step == BUDGET else f"{step}  sensitivity"
        print(f"  {tag:<22}" + "".join(f"{x:>10.4f}" for x in v)
              + f"{st.mean(v):>10.4f}{st.stdev(v):>9.4f}")

    headline = table(BUDGET, "tpr")
    out["headline_tpr_4000"] = {"per_fold": headline, "mean": st.mean(headline),
                                "sd": st.stdev(headline)}

    print(f"\nSECONDARY (fixed operating point; AUPRC is never a selection proxy)")
    print(f"  {'metric @4000':<26}" + "".join(f"{'fold '+str(k):>10}" for k in FOLDS)
          + f"{'mean':>10}")
    for key, label in (("comp_at_purity", "completeness@24%-purity"),
                       ("purity_corr", "purity@50%-completeness"),
                       ("auroc", "AUROC (saturating)"),
                       ("auprc", "AUPRC (NOT a proxy)")):
        v = table(BUDGET, key)
        good = [x for x in v if not math.isnan(x)]
        mean = st.mean(good) if len(good) == len(v) else float("nan")
        print(f"  {label:<26}" + "".join(f"{fmt(x):>10}" for x in v)
              + f"{fmt(mean):>10}"
              + ("   <- n/a present: mean withheld, no CI" if len(good) != len(v) else ""))

    # ---- the four pre-registered paired contrasts -------------------------
    print("\nPAIRED CONTRASTS on folds 1-4, paired t-interval df=3 (NOT a bootstrap)")
    contrasts = {}

    base = [frozen[k] for k in FOLDS]
    contrasts["lambda0 - Baseline A (frozen)"] = paired_t(
        [a - b for a, b in zip(headline, base)])

    # BUDGET-MATCHED is the form to report. Each DA arm is read at its own declared
    # budget, so the honest control is lambda=0 read at THAT SAME STEP -- which the
    # sensitivity readouts supply for free. Differencing a 2000-step CORAL against a
    # 4000-step control adds a budget difference on top of the objective difference.
    matched_step = {"DANN x nb3 (4000)": BUDGET,
                    "CORAL x nb3 (2000)": 2000,
                    "FreeMatch x nb3 (1500)": 1500}
    for name, comp in comparators().items():
        step = matched_step[name]
        ctrl = table(step, "tpr")
        contrasts[f"{name} - lambda0 @{step} MATCHED"] = paired_t(
            [a - b for a, b in zip(comp, ctrl)])
    # kept alongside, because the headline lambda=0 number is the one at 4000
    for name, comp in comparators().items():
        if matched_step[name] != BUDGET:
            contrasts[f"{name} - lambda0 @{BUDGET} (budget-unmatched)"] = paired_t(
                [a - b for a, b in zip(comp, headline)])

    print(f"  {'contrast':<34}{'mean':>9}{'95% CI':>20}{'excl 0':>8}{'same sign':>11}")
    for name, c in contrasts.items():
        ci = f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]"
        print(f"  {name:<34}{c['mean']:>+9.4f}{ci:>20}"
              f"{str(c['excludes_zero']):>8}{str(c['all_same_sign']):>11}")
    out["contrasts"] = contrasts
    out["baseline_A_frozen_mean_folds_1_4"] = st.mean(base)

    # THE TWO BASELINE A NUMBERS, reconciled once and for all. 0.176 in the nb3 table is
    # the FIVE-fold mean (it includes fold 0's 0.193878); 0.1714 is the folds-1-4 mean.
    # Same measurement, different fold sets. Only the folds-1-4 value may be compared
    # against folds-1-4 results.
    fold0_frozen = 0.19387755102040816
    five = st.mean(list(frozen.values()) + [fold0_frozen])
    out["baseline_A_frozen_mean_folds_0_4"] = five
    print(f"\n  Baseline A (frozen), reconciled: folds 1-4 mean {st.mean(base):.4f}; "
          f"folds 0-4 mean {five:.4f} (= the 0.176 in the nb3 table)")
    print("    Same measurement, different fold sets. Only the folds-1-4 value is")
    print("    comparable with anything computed on folds 1-4.")

    # Multiplicity: say it rather than correct it silently.
    n_ci = len(contrasts)
    out["multiplicity"] = {
        "n_intervals": n_ci, "correction": None,
        "note": (f"{n_ci} intervals at 95% with NO multiplicity correction. Under a "
                 f"global null the chance at least one excludes zero is "
                 f"~{1 - 0.95 ** n_ci:.0%}. The three DA-vs-lambda0 contrasts are not "
                 f"independent (they share one control arm), so that figure is an upper "
                 f"bound, and they agree in sign and size on every fold -- but the "
                 f"absence of a correction is disclosed, not assumed away.")}
    print(f"\n  MULTIPLICITY: {n_ci} intervals at 95%, no correction applied. "
          f"Disclosed, not corrected.")

    # the same contrasts at the budget-matched readouts, where they are matched
    print("\n  budget-matched cross-checks (both arms read at the same absolute step):")
    m2000 = paired_t([a - b for a, b in zip(dann_at_2000(), table(2000, "tpr"))])
    print(f"    DANN - lambda0, both at 2000     {m2000['mean']:+.4f}  "
          f"[{m2000['ci_lo']:+.4f}, {m2000['ci_hi']:+.4f}]  "
          f"excl 0 {m2000['excludes_zero']}")
    out["contrasts"]["DANN - lambda0, both at step 2000"] = m2000

    # ---- checkpoint-to-checkpoint stability -------------------------------
    print("\nCHECKPOINT-TO-CHECKPOINT STABILITY, 16-point grid 250..4000")
    stab = {}
    for k in FOLDS:
        v = [score(k, s)["tpr"] for s in range(CKPT_EVERY, BUDGET + 1, CKPT_EVERY)]
        stab[k] = {"n": len(v), "mean": st.mean(v), "sd": st.stdev(v),
                   "sd_over_mean": st.stdev(v) / st.mean(v) if st.mean(v) else float("nan"),
                   "min": min(v), "max": max(v)}
        print(f"  fold {k}: n={stab[k]['n']}  mean {stab[k]['mean']:.4f}  "
              f"sd {stab[k]['sd']:.4f}  sd/mean {stab[k]['sd_over_mean']:.3f}  "
              f"range {stab[k]['min']:.4f}-{stab[k]['max']:.4f}")
    out["stability"] = stab
    msd = st.mean([stab[k]["sd"] for k in FOLDS])
    mrel = st.mean([stab[k]["sd_over_mean"] for k in FOLDS])
    print(f"  mean over folds: absolute sd {msd:.4f}, sd/mean {mrel:.3f}")
    out["stability_mean"] = {"abs_sd": msd, "sd_over_mean": mrel}

    # 7.18's stability ordering was read off a 32-point 250..8000 grid. These folds stop
    # at 4000, so quoting 0.252 against it would compare different windows. Recompute
    # round 5's two fold-0 arms on BOTH windows -- the matched one is the only fair
    # comparison, and it is the one that decides whether "unfreezing without a DA term is
    # the noisiest arm" survives.
    print("\n  GRID-MATCHED comparison (7.18's ordering was measured on 250..8000):")
    curves = {"arm1 lambda0 fold0": os.path.join(REPO, "runs", "dann_r5_curves",
                                                 "arm1_lambda0_curve.csv"),
              "arm2 DANN fold0": os.path.join(REPO, "runs", "dann_r5_curves",
                                              "arm2_dann_curve.csv")}
    windows = {}
    for label, path in curves.items():
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            d = {int(float(r["step"])): float(r["tpr_at_1e-3"])
                 for r in csv.DictReader(fh)
                 if r.get("tpr_at_1e-3") not in (None, "", "nan")}
        for lo, hi in ((250, 8000), (250, BUDGET)):
            v = [d[x] for x in range(lo, hi + 1, CKPT_EVERY) if x in d]
            if len(v) > 1:
                windows.setdefault(f"{lo}-{hi}", {})[label] = {
                    "n": len(v), "sd": st.stdev(v), "mean": st.mean(v),
                    "sd_over_mean": st.stdev(v) / st.mean(v)}
    for win, rec in windows.items():
        cells = "  ".join(f"{lab}: n={r['n']} sd/mean {r['sd_over_mean']:.3f}"
                          for lab, r in rec.items())
        print(f"    {win:<10} {cells}")
    print(f"    {'250-' + str(BUDGET):<10} lambda0 folds 1-4 (THIS ROUND): "
          f"n=16 sd/mean {mrel:.3f}")
    out["stability_grid_matched"] = windows

    w4 = windows.get(f"250-{BUDGET}", {})
    if len(w4) == 2:
        a, b = (w4["arm1 lambda0 fold0"]["sd_over_mean"],
                w4["arm2 DANN fold0"]["sd_over_mean"])
        print(f"\n    On the MATCHED 250-{BUDGET} window the lambda0-vs-DANN gap is "
              f"{a:.3f} vs {b:.3f} -- it DISSOLVES.")
        print("    7.18's 0.276 > 0.202 is a property of the 250-8000 window, not of the")
        print("    arms. 'Unfreezing without a DA term is the noisiest' does NOT become a")
        print("    claim on four more folds; it fails grid-matching. 7.16 gains an entry.")
        out["stability_ordering_survives_grid_matching"] = False

    # ---- the pre-registered decision --------------------------------------
    mean_tpr = st.mean(headline)
    da_means = {n: st.mean(v) for n, v in comparators().items()}
    lo_da = min(da_means.values())
    print("\n" + line)
    print("THE PRE-REGISTERED DECISION")
    print(line)
    print(f"  lambda=0 fold-mean TPR@1e-3 : {mean_tpr:.4f}  (sd {st.stdev(headline):.4f})")
    print(f"  frozen Baseline A, same folds: {st.mean(base):.4f}")
    print(f"  the three DA arms            : "
          + ", ".join(f"{n.split(' x')[0]} {v:.4f}" for n, v in da_means.items()))
    print(f"  fold 0's single observation  : 0.1531")

    if mean_tpr < 0.25:
        verdict, bnm = "LOW", "GO"
    elif mean_tpr >= 0.40:
        verdict, bnm = "NEAR THE DA ARMS", "NO-GO, and stop the bake-off"
    else:
        verdict, bnm = "IN BETWEEN / AMBIGUOUS", "NO-GO by default"
    out["verdict"] = {"fold_mean": mean_tpr, "branch": verdict, "bnm": bnm,
                      "da_arm_means": da_means, "lowest_da_arm": lo_da}
    print(f"\n  BRANCH: {verdict}")
    print(f"  BNM   : {bnm}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
