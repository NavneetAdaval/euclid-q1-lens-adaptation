"""
decompose_aug_precision.py
==========================
Split the MEASURED -0.0714 TPR gap between the two frozen controls into an AUGMENTATION
component and a PRECISION component.

WHY THIS EXISTS. Task 2 measured, on the frozen encoder (n_blocks=0, fold 0):

    nb0_image (aug=weak, fp16)      TPR@1e-3 = 0.122449
    nb0_image (aug=none, fp32)      TPR@1e-3 = 0.193878
    paired diff                     -0.0714, CI [-0.1327, -0.0102]  (excludes 0)

Those two runs differ in TWO ways at once, so the attribution was tagged [UNVERIFIED]. It
matters for three separate reasons:
  * the unfrozen rungs (nb1/nb3) run at fp16 -- if PRECISION costs ~0.07 TPR, their numbers
    carry a precision handicap that has nothing to do with unfreezing;
  * round 1 compared feature-space methods (Baseline A, CORAL -- cached aug="none"
    features) against image-space methods (AdaMatch, FreeMatch -- aug="weak"), so an
    augmentation effect of this size is an uncontrolled difference in the published table;
  * PROJECT.md 7.3 mandates +-3px augmentation at TRAINING time, and if that HURTS a frozen
    encoder it is a finding about the protocol, not a nuisance.

THE DESIGN. A third run at (aug=weak, fp32) sits diagonally between them, giving three of
the four cells of a 2x2:

                     fp32                 fp16
    aug=none    A: 0.193878          D: (not run)
    aug=weak    B: this run          C: 0.122449

    augmentation effect, held at fp32 : B - A
    precision effect, held at aug=weak: C - B
    and (B - A) + (C - B) == C - A     == the total -0.0714, exactly.

So the total decomposes along the path A -> B -> C with no residual. The INTERACTION term
is NOT identified without cell D, and this script says so rather than implying additivity.

Run (after the weak/fp32 run completes):
    python scripts/decompose_aug_precision.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metrics import (tpr_at_fpr, auroc, paired_bootstrap, default_tpr_metric,
                     DEFAULT_TARGET_FPR)                       # noqa: E402

_ROOT = _HERE.parent
SCORE = "scores/baselineA_fold0_nb0_image_fold0.npz"

CELLS = {
    "A_none_fp32": ("baselineA_fold0_nb0_image_noaug_fp32", "aug=none, fp32"),
    "B_weak_fp32": ("baselineA_fold0_nb0_image_weak_fp32", "aug=weak, fp32"),
    "C_weak_fp16": ("baselineA_fold0_nb0_image", "aug=weak, fp16"),
}


def load(run_dir):
    p = _ROOT / "runs" / run_dir / SCORE
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {"pos": z["pos_scores"], "neg": z["neg_scores"],
                "pos_ids": z["pos_ids"] if "pos_ids" in z.files else None,
                "neg_ids": z["neg_ids"] if "neg_ids" in z.files else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runs" / "lever1_aug_precision.json"))
    args = ap.parse_args(argv)

    print("=" * 82)
    print("AUG vs PRECISION DECOMPOSITION -- frozen encoder (n_blocks=0), fold 0")
    print("=" * 82)

    have = {}
    for k, (d, desc) in CELLS.items():
        v = load(d)
        if v is None:
            print(f"  [missing] {k}  ({desc})  <- runs/{d}/{SCORE}")
        else:
            have[k] = v
    if len(have) < 3:
        print("\nSTOP: need all three cells (A, B, C). Run the weak/fp32 decomposition "
              "run:\n  python scripts/run_screen_unfreeze.py --run nb0_image --aug weak "
              "--dir-suffix _weak_fp32 --diagnostic --num-workers 2")
        return 3

    # row alignment is a precondition for the paired bootstrap, not an assumption
    ref = have["A_none_fp32"]
    for k, v in have.items():
        if v["pos_ids"] is None or not (
                np.array_equal(v["pos_ids"], ref["pos_ids"])
                and np.array_equal(v["neg_ids"], ref["neg_ids"])):
            print(f"\nSTOP: {k} is not row-aligned with A (or lacks ids); a paired "
                  f"bootstrap would be invalid.")
            return 1

    y = np.concatenate([np.ones(ref["pos"].size, np.int64),
                        np.zeros(ref["neg"].size, np.int64)])
    s = {k: np.concatenate([v["pos"], v["neg"]]) for k, v in have.items()}
    t = {k: tpr_at_fpr(y, v, args.target_fpr) for k, v in s.items()}
    au = {k: auroc(y, v) for k, v in s.items()}

    print(f"\n  {'cell':<14} {'config':<18} {'TPR@1e-3':>10} {'AUROC':>9}")
    print("  " + "-" * 56)
    for k in ("A_none_fp32", "B_weak_fp32", "C_weak_fp16"):
        print(f"  {k:<14} {CELLS[k][1]:<18} {t[k]:10.6f} {au[k]:9.6f}")

    def diff(k1, k0, label):
        bs = paired_bootstrap(y, s[k1], s[k0],
                              metric_fn=default_tpr_metric(args.target_fpr),
                              B=args.bootstrap, seed=args.seed)
        sig = bool(bs.get("significant"))
        print(f"  {label:<44} {bs['point']:+.4f}  CI [{bs['lo']:+.4f}, {bs['hi']:+.4f}]  "
              f"{'SIGNIFICANT' if sig else 'not distinguishable'}")
        return {"diff": float(bs["point"]), "ci": [bs["lo"], bs["hi"]], "significant": sig}

    print("\n  DECOMPOSITION (path A -> B -> C; the two legs sum to the total exactly)")
    print("  " + "-" * 78)
    aug = diff("B_weak_fp32", "A_none_fp32", "AUGMENTATION effect (held at fp32)   B - A")
    prec = diff("C_weak_fp16", "B_weak_fp32", "PRECISION effect (held at aug=weak)  C - B")
    tot = diff("C_weak_fp16", "A_none_fp32", "TOTAL (what Task 2 measured)         C - A")

    print("\n  " + "=" * 78)
    resid = tot["diff"] - (aug["diff"] + prec["diff"])
    print(f"  arithmetic check: (B-A) + (C-B) = {aug['diff'] + prec['diff']:+.4f}  vs  "
          f"(C-A) = {tot['diff']:+.4f}   residual {resid:+.1e}")

    # attribute, in the metric's own units (1 lens = 1/98)
    one = 1.0 / ref["pos"].size
    print(f"\n  In lenses (1 lens = {one:.4f} TPR on {ref['pos'].size} fold-0 positives):")
    print(f"    augmentation : {aug['diff']/one:+.1f} lenses")
    print(f"    precision    : {prec['diff']/one:+.1f} lenses")
    print(f"    total        : {tot['diff']/one:+.1f} lenses")

    dominant = "augmentation" if abs(aug["diff"]) > abs(prec["diff"]) else "precision"
    ratio = (max(abs(aug["diff"]), abs(prec["diff"]))
             / max(min(abs(aug["diff"]), abs(prec["diff"])), 1e-12))
    if aug["significant"] or prec["significant"]:
        print(f"\n  VERDICT: the gap is driven mainly by {dominant.upper()}.")
    else:
        # Do NOT name a driver from point estimates alone when neither leg resolves --
        # that is exactly the "backed by a measurement" line this project draws.
        print(f"\n  VERDICT: NO leg is individually resolvable. The point estimate favours "
              f"{dominant.upper()}")
        print(f"           ({abs(aug['diff'])/one:.0f} vs {abs(prec['diff'])/one:.0f} "
              f"lenses, {ratio:.1f}x), but both CIs contain 0, so this is a RANKING OF "
              f"POINT")
        print(f"           ESTIMATES, not an established attribution. The TOTAL remains "
              f"significant.")
    if aug["significant"] and not prec["significant"]:
        print("    Augmentation is the significant leg; precision is not distinguishable.")
        print("    => fp16 is SAFE for nb1/nb3, and the +-3px augmentation (PROJECT.md 7.3)")
        print("       HURTS a frozen encoder -- which also means round 1 compared")
        print("       cached-feature methods (no aug) against image-space methods (aug)")
        print("       across an uncontrolled difference of this size.")
    elif prec["significant"] and not aug["significant"]:
        print("    Precision is the significant leg; augmentation is not distinguishable.")
        print("    => fp16 costs real TPR here. nb1/nb3 (fp16) carry that handicap, and it")
        print("       is NOT an unfreezing effect. Consider fp32 for the 5-fold commit and")
        print("       disclose the precision asymmetry (PROJECT.md 7.7 already flags one).")
    elif aug["significant"] and prec["significant"]:
        print("    BOTH legs are individually significant.")
    else:
        print("    NEITHER leg is individually significant even though the total is --")
        print("    expected when one real effect is split across two legs on ~98 positives")
        print("    (each leg gets ~half the signal but a similarly wide CI).")
        print("    ACTIONABLE READING, stated at the strength the data supports:")
        print(f"      * precision costs {abs(prec['diff'])/one:.0f} lenses by point "
              f"estimate, CI spans 0 -> fp16 is NOT shown")
        print("        to be harmful, so keeping fp16 for nb1/nb3 is defensible; it is not")
        print("        shown to be harmless either, so DISCLOSE it rather than assert it.")
        print(f"      * augmentation costs {abs(aug['diff'])/one:.0f} lenses by point "
              f"estimate, CI upper bound at 0.0")
        print("        -> suggestive that +-3px HURTS a frozen encoder (PROJECT.md 7.3),")
        print("        and that round 1's cached-feature vs image-space comparison carried")
        print("        an uncontrolled difference of about this size. Suggestive, not shown.")
    print("\n  [NOT IDENTIFIED] the interaction term needs the 4th cell (aug=none, fp16).")
    print("    The path A->B->C is exact, but 'augmentation effect' is measured AT fp32 and")
    print("    'precision effect' AT weak aug; they need not hold at the other level.")

    out = {"cells": {k: {"config": CELLS[k][1], "tpr": float(t[k]), "auroc": float(au[k])}
                     for k in t},
           "augmentation_effect_at_fp32": aug,
           "precision_effect_at_weak_aug": prec,
           "total": tot,
           "interaction": "NOT IDENTIFIED (requires cell D: aug=none, fp16)",
           "n_pos": int(ref["pos"].size), "one_lens": one}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
