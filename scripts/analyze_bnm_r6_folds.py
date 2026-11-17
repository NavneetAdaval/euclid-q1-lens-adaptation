"""
analyze_bnm_r6_folds.py
=======================
ROUND 6 Stage 4 -- BNM x nb3 confirmatory folds 1-4.

Everything here follows the hashed pre-registration (``runs/bnm_r6_prereg.json``,
sha256 d205f678...) and its two annotations, PROJECT.md 7.26 and 7.27.

    python -u scripts/analyze_bnm_r6_folds.py [--out PATH] [--force]

The result file (default ``runs/bnm_r6_folds_results.json``) is never overwritten silently:
if it exists the script refuses before computing anything, unless ``--force`` is given.

DELIBERATELY REUSED, NOT REIMPLEMENTED:
  * the metric path -- ``metrics.metric_suite``, exactly as ``analyze_lambda0_r6.score``
    calls it, so BNM and its control are measured by the same code;
  * the comparator per-fold values -- imported from ``analyze_lambda0_r6.COMPARATORS``
    rather than retyped, so there is one source of truth for the arms BNM is read against;
  * the paired t-interval -- ``analyze_lambda0_r6.paired_t``, df = 3. Never a bootstrap:
    at n = 4 a bootstrap draws from a 4^4 = 256 lattice.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics as st
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

import metrics as me                                            # noqa: E402
from analyze_lambda0_r6 import COMPARATORS, paired_t            # noqa: E402
from analyze_lambda0_r6 import score as lambda0_score           # noqa: E402
from integrity import verify as _verify_integrity               # noqa: E402
from run_dann_r5 import FOLD_GATE                               # noqa: E402

ROOT = os.path.join(REPO, "runs", "bnm_r6_folds")
FOLDS = (1, 2, 3, 4)
READOUTS = (1000, 1500, 2000, 4000)
TARGET_FPR = 1e-3
BUDGET = 1000

# lambda=0, folds 1-4 -- the known-null for this arm.
#
# COMPUTED FROM THAT ARM'S OWN LOGITS, NOT TRANSCRIBED -- corrected 2026-09-19. This was
# a dict of 4-dp literals read off PROJECT.md 7.22's printed table, which is a rounded
# PRESENTATION of the measurement rather than the measurement. The exact fold-1 value at
# step 4000 is 0.18367347, not 0.1837, and the difference moved the published headline
# BNM(4000) - lambda=0(4000) upper bound by one unit in the fourth decimal, +0.2560 ->
# +0.2561. The module docstring above already claimed one source of truth for comparator
# values; this makes that true of the control as well.
#
# ``lambda0_score`` is ``analyze_lambda0_r6.score`` -- the same metric path, on the same
# logits, that produced 7.22's own numbers.
_LAMBDA0_CACHE: dict = {}


def lambda0_per_fold(step: int) -> list:
    """lambda=0's per-fold TPR@1e-3 at ``step``, straight from its stored logits."""
    if step not in _LAMBDA0_CACHE:
        _LAMBDA0_CACHE[step] = [float(lambda0_score(f, step)["tpr"]) for f in FOLDS]
    return _LAMBDA0_CACHE[step]


def run_dir(fold):
    return os.path.join(ROOT, f"bnm_r6_fold{fold}")


def score(fold, step):
    p = os.path.join(run_dir(fold), "logits", f"bnm_nb3_fold{fold}",
                     f"step_{step}.npz")
    z = np.load(p)
    pos, neg = np.asarray(z["pos"], float), np.asarray(z["neg"], float)
    y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
    s = np.concatenate([pos, neg])
    suite = me.metric_suite(y, s, target_fpr=TARGET_FPR)
    suite["n_pos"], suite["n_neg"] = int(pos.size), int(neg.size)
    return suite


def lattice_check(fold, tpr):
    """FPR=1e-3 falls between 26 and 27 false positives out of 26,399.

    An off-lattice TPR is legitimate and must sit at exactly the interpolation weight
    0.39898 between adjacent lattice points. A cheap integrity check on the metric path.
    """
    n_pos = 98 if fold in (1, 2, 3) else 97
    q = 1.0 / n_pos
    k = tpr / q
    nearest = round(k)
    frac = k - math.floor(k)
    on_lattice = abs(k - nearest) < 1e-9
    return {"n_pos": n_pos, "quantum": q, "k": k, "on_lattice": on_lattice,
            "frac": frac}


OUT_DEFAULT = os.path.join(REPO, "runs", "bnm_r6_folds_results.json")


def main(argv=None):
    # OVERWRITE GUARD (added 2026-09-26, CHECKLIST_FIXES_REPORT.md): the result file used to
    # be rewritten unconditionally at a repo-anchored path. Refuse to replace an existing
    # output unless --force is given; --out redirects the write (default unchanged).
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help="result JSON path (default: runs/bnm_r6_folds_results.json)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists")
    args = ap.parse_args(argv)
    out_path = os.path.abspath(args.out)
    if os.path.exists(out_path) and not args.force:
        print(f"[refused] {out_path} exists; pass --force to overwrite it, or --out "
              f"to write elsewhere. Nothing was computed or written.", file=sys.stderr)
        return 2

    _verify_integrity(where="before the folds-1-4 analysis")

    pre = json.load(open(os.path.join(REPO, "runs", "bnm_r6_prereg.json"),
                         encoding="utf-8"))
    out = {"round": 6, "arm": "bnm_nb3", "folds": list(FOLDS),
           "gamma": pre["gamma"]["value"],
           "budget": BUDGET,
           "budget_outcome": "no_matched_arm_budget",
           "prereg_sha256": pre.get("status") and None}

    print("=" * 96)
    print("ROUND 6 -- BNM x nb3, CONFIRMATORY FOLDS 1-4")
    print("=" * 96)
    print(f"  gamma {pre['gamma']['value']}   budget {BUDGET} "
          f"(no_matched_arm_budget -- NOT cross-arm comparable)")
    print("  fold 0 is development and enters no mean. Nothing about ranking from it.")

    # ---------------- per-fold, per-readout ----------------
    per = {s: {} for s in READOUTS}
    aur = {s: {} for s in READOUTS}
    for f in FOLDS:
        for s in READOUTS:
            suite = score(f, s)
            per[s][f] = float(suite["tpr"])
            aur[s][f] = float(suite["auroc"])

    print("\n" + "=" * 96)
    print("1. TPR@FPR=1e-3 -- interpolated, drop_intermediate=False, ALWAYS a rate")
    print("=" * 96)
    print(f"  {'read-off':<28}{'fold 1':>9}{'fold 2':>9}{'fold 3':>9}{'fold 4':>9}"
          f"{'mean':>10}{'sd':>9}")
    for s in READOUTS:
        v = [per[s][f] for f in FOLDS]
        role = {1000: "BUDGET (own-objective)", 1500: "matched FreeMatch",
                2000: "matched CORAL", 4000: "MANDATORY (lambda=0)"}[s]
        print(f"  {str(s) + ' -- ' + role:<28}" +
              "".join(f"{x:>9.4f}" for x in v) +
              f"{st.mean(v):>10.4f}{st.stdev(v):>9.4f}")
        out.setdefault("tpr_per_fold", {})[str(s)] = {str(f): per[s][f] for f in FOLDS}
        out.setdefault("tpr_mean", {})[str(s)] = st.mean(v)
        out.setdefault("tpr_sd", {})[str(s)] = st.stdev(v)

    print(f"\n  AUROC (saturating secondary sanity metric only, never a selector)")
    for s in READOUTS:
        v = [aur[s][f] for f in FOLDS]
        print(f"    {s:>5}: " + " ".join(f"{x:.4f}" for x in v) +
              f"   mean {st.mean(v):.4f}")
        out.setdefault("auroc_mean", {})[str(s)] = st.mean(v)

    # ---------------- lattice check ----------------
    print("\n" + "=" * 96)
    print("2. QUANTISATION CHECK -- 1/98 folds 1-3, 1/97 fold 4")
    print("=" * 96)
    offs = 0
    for s in READOUTS:
        for f in FOLDS:
            lc = lattice_check(f, per[s][f])
            if not lc["on_lattice"]:
                offs += 1
    print(f"  off-lattice values: {offs} of {len(READOUTS) * len(FOLDS)}")
    print("  (off-lattice is LEGITIMATE: FPR=1e-3 falls between 26 and 27 false positives")
    print("   out of 26,399, so interpolation is expected. Values are counts/n_pos where")
    print("   the interpolation lands on a lattice point, and between them otherwise.)")

    # ---------------- contrasts ----------------
    print("\n" + "=" * 96)
    print("3. THE FOUR PAIRED CONTRASTS -- paired t on 4 differences, df = 3")
    print("=" * 96)
    print("  EVERY ONE carries the 7.18 disclosure: a budget match removes the BUDGET")
    print("  confound ONLY. Head initialisation and the gamma-tuning budget still travel.")
    print()

    contrasts = [
        ("BNM(4000) - lambda=0(4000)  [HEADLINE]", 4000, lambda0_per_fold(4000)),
        ("BNM(1500) - FreeMatch(1500)", 1500, COMPARATORS["FreeMatch x nb3 (1500)"]),
        ("BNM(2000) - CORAL(2000)", 2000, COMPARATORS["CORAL x nb3 (2000)"]),
        ("BNM(4000) - DANN(4000)", 4000, COMPARATORS["DANN x nb3 (4000)"]),
    ]
    print(f"  {'contrast':<40}{'mean':>9}{'95% CI':>22}{'half':>8}  excl0  same-sign")
    rows = []
    for label, step, comp in contrasts:
        diffs = [per[step][f] - comp[i] for i, f in enumerate(FOLDS)]
        t = paired_t(diffs)
        rows.append((label, t))
        print(f"  {label:<40}{t['mean']:>+9.4f}"
              f"{'[' + f'{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}' + ']':>22}"
              f"{t['half_width']:>8.4f}"
              f"{'  yes ' if t['excludes_zero'] else '  no  '}"
              f"{'   yes' if t['all_same_sign'] else '   no'}")
        out.setdefault("contrasts", {})[label] = t

    print(f"\n  MULTIPLICITY: {len(contrasts)} intervals computed. "
          f"NO correction applied, disclosed as required.")

    print("\n  per-fold differences:")
    for label, t in rows:
        print(f"    {label:<40}" + " ".join(f"{d:+.4f}" for d in t["diffs"]))

    # ---------------- 704/78, all three readings ----------------
    print("\n" + "=" * 96)
    print("4. THE 704/78 DIAGNOSTIC -- all THREE readings declared in 7.27")
    print("=" * 96)
    readoff, anymax, withinmean = [], [], []
    for f in FOLDS:
        rows_f = list(csv.DictReader(open(os.path.join(run_dir(f), "checkpoints.csv"))))
        gaps = {int(r["step"]): float(r["nuclear_gap"]) for r in rows_f}
        allg = [float(r["nuclear_gap"]) for r in rows_f]
        readoff.append(gaps[BUDGET])
        anymax.append(max(allg))
        withinmean.append(st.mean(allg))

    def summarise(name, vals, primary=False):
        m, sd = st.mean(vals), st.stdev(vals)
        print(f"\n  {name}{'  [PRIMARY]' if primary else ''}")
        print(f"    per fold: " + " ".join(f"{v:+.4f}" for v in vals))
        print(f"    fold-mean {m:+.4f}   sd {sd:.4f}")
        return {"per_fold": vals, "mean": m, "sd": sd}

    r1 = summarise(f"(1) gap at the read-off checkpoint (step {BUDGET})", readoff, True)
    r2 = summarise("(2) any-checkpoint maximum", anymax)
    r3 = summarise("(3) within-fold mean over checkpoints", withinmean)
    out["diagnostic_704_78"] = {"read_off": r1, "any_checkpoint_max": r2,
                                "within_fold_mean": r3}

    print("\n  SCORED against the pre-registered prediction [+0.02, +0.08],")
    print("  falsified by consistently negative gaps OR magnitudes above ~0.15:")
    for nm, r in (("(1) read-off  [PRIMARY]", r1), ("(2) any-ckpt max", r2),
                  ("(3) within-fold mean", r3)):
        m = r["mean"]
        inband = 0.02 <= m <= 0.08
        indep = 0.04 <= m <= 0.08
        verdict = ("INSIDE the registered band" if inband else
                   "ABOVE 0.15 -- FALSIFIED by magnitude" if m > 0.15 else
                   "BELOW the band" if m < 0.02 else "ABOVE the band")
        print(f"    {nm:<26} mean {m:+.4f}  -> {verdict}"
              f"{'  (also inside the independent [0.04,0.08])' if indep else ''}")

    n_neg = sum(1 for v in readoff if v < 0)
    print(f"\n    folds with a NEGATIVE read-off gap: {n_neg}/4 "
          f"(one is TOLERATED as noise; consistently negative falsifies)")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n[out] {out_path}")

    _verify_integrity(where="after the folds-1-4 analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
