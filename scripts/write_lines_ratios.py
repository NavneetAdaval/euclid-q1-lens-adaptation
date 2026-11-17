"""
write_lines_ratios.py
=====================
Persist the Lines-et-al. comparison ratio for every nb3-era arm to disk.

    python -u scripts/write_lines_ratios.py --out runs/round7/lines_ratios.json

WHY THIS EXISTS. `completeness@24%-purity` is the quantity the project's "recovers X% of
Lines' sim-only benchmark" sentence is built on. Only the round-1 FROZEN
`metric_suite.json` artifacts ever stored it; **every nb3-era ratio lived as prose in
PROJECT.md** until this file. A number in an abstract should be re-derivable from an
artifact, and this one was not.

The metric is `metrics.metric_suite`'s `comp_at_purity` -- target_purity 0.24,
nn_np_ratio 2200 (Lines' Q1 rarity, so purity is prevalence-corrected and comparable),
computed on the same ROC pass as the headline TPR. Nothing is redefined here; the arm
score loaders are imported from `analyze_window_average` so there is one source of truth
for where each arm's logits live.

THE n/a CASES ARE WRITTEN OUT EXPLICITLY, not silently dropped. 24% purity is unreachable
on DANN x nb3 fold 1 and on lambda=0 folds 2-3; the guard in `completeness_at_purity`
refuses to extrapolate and returns nan. Under this project's rule an arm with any
undefined fold gets **n/a with no CI and no fold-mean**, and the artifact says which folds
and why rather than leaving a reader to infer it from a missing key.

Writes NOTHING outside --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

import metrics as me                                   # noqa: E402
from analyze_lambda0_r6 import paired_t                # noqa: E402
from analyze_window_average import ARMS                # noqa: E402
from integrity import verify as _verify_integrity      # noqa: E402

LINES_COMPLETENESS = me.LINES_SOURCE_ONLY_COMPLETENESS   # 0.50
LINES_PURITY = me.LINES_SOURCE_ONLY_PURITY               # 0.24

# Each arm's mandatory readouts, and which one the paper would quote. Copied from each
# arm's own budget/pre-registration record -- none is chosen here.
QUOTED = {"CORAL x nb3": 2000, "FreeMatch x nb3": 1500, "DANN x nb3": 4000,
          "lambda=0": 4000, "BNM x nb3": 4000, "Baseline A (frozen)": 8000}
NOTES = {
    "BNM x nb3": ("budget 1000 is `no_matched_arm_budget` (7.28) and is BNM's own-objective "
                  "reading only -- never cross-arm comparable. 4000 is the mandatory readout "
                  "and is the one quoted."),
    "DANN x nb3": "2000 is 7.18's pre-declared sensitivity readout.",
    "Baseline A (frozen)": ("single source-only model, so folds 0-4 and pooling are both "
                            "legitimate for it (2); the folds-1-4 mean is given alongside "
                            "for like-for-like comparison with the per-fold-model arms."),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    _verify_integrity(where="before writing the Lines ratios")

    out = {
        "what": "completeness at 24% purity, and it as a fraction of Lines et al.'s 0.50",
        "metric": "metrics.metric_suite -> comp_at_purity",
        "target_purity": LINES_PURITY,
        "nn_np_ratio": 2200,
        "nn_np_ratio_why": ("Lines' Q1 rarity, 497 lenses in 1,086,554 images; purity is "
                            "prevalence-corrected to it so the two are comparable"),
        "lines_reference_completeness": LINES_COMPLETENESS,
        "basis": ("folds 1-4 fold-mean for every per-fold-model arm; Baseline A also "
                  "reports folds 0-4 because it is a single model"),
        "rule_for_undefined": ("any fold where 24% purity is unreachable makes the ARM n/a "
                               "-- no fold-mean, no CI, never an average over the reachable "
                               "subset"),
        "interval": "paired t on the per-fold values, df = n-1. Never a bootstrap at n = 4.",
        "arms": {},
    }

    print("=" * 104)
    print("completeness@24%-purity, and X = it / 0.50 -- EVERY nb3-era ARM, PERSISTED")
    print("=" * 104)
    print(f"  {'arm':<22}{'step':>6}{'f0':>9}{'f1':>9}{'f2':>9}{'f3':>9}{'f4':>9}"
          f"{'mean':>9}{'X':>8}   95% CI on X")

    for name, spec in ARMS.items():
        folds = list(spec["folds"])
        loaded = {f: spec["load"](f) for f in folds}
        rec = {"folds": folds, "quoted_readout": QUOTED[name], "readouts": {}}
        if name in NOTES:
            rec["note"] = NOTES[name]

        for ro in spec["readouts"]:
            comps, undef = {}, []
            for f in folds:
                if ro not in loaded[f]:
                    comps[f] = None
                    undef.append(f)
                    continue
                pos, neg = loaded[f][ro]
                y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
                su = me.metric_suite(y, np.concatenate([pos, neg]), target_fpr=1e-3)
                c = float(su["comp_at_purity"])
                if math.isfinite(c):
                    comps[f] = c
                else:
                    comps[f] = None
                    undef.append(f)

            defined = [comps[f] for f in folds if comps[f] is not None]
            all_ok = len(defined) == len(folds)
            entry = {
                "comp_at_purity_per_fold": {str(f): comps[f] for f in folds},
                "undefined_folds": undef,
                "n_defined": len(defined), "n_folds": len(folds),
                "defined_on_all_folds": all_ok,
            }
            if all_ok:
                m = st.mean(defined)
                t = paired_t(defined)
                entry.update({
                    "fold_mean": m, "sd": t["sd"],
                    "pct_of_lines": 100.0 * m / LINES_COMPLETENESS,
                    "ci95": [t["ci_lo"], t["ci_hi"]],
                    "pct_of_lines_ci95": [100.0 * t["ci_lo"] / LINES_COMPLETENESS,
                                          100.0 * t["ci_hi"] / LINES_COMPLETENESS]})
                cells = "".join(f"{comps[f]:>9.4f}" for f in folds)
                pad = "" if 0 in folds else f"{'--':>9}"
                print(f"  {name:<22}{ro:>6}{pad}{cells}{m:>9.4f}"
                      f"{100.0 * m / LINES_COMPLETENESS:>7.1f}%"
                      f"   [{100.0 * t['ci_lo'] / LINES_COMPLETENESS:>5.1f}%,"
                      f"{100.0 * t['ci_hi'] / LINES_COMPLETENESS:>6.1f}%]")
            else:
                entry.update({"fold_mean": None, "pct_of_lines": None, "ci95": None,
                              "why_na": (f"24% purity is unreachable on fold(s) {undef}; "
                                         f"the arm is n/a with no fold-mean and no CI")})
                cells = "".join((f"{comps[f]:>9.4f}" if comps[f] is not None else f"{'n/a':>9}")
                                for f in folds)
                pad = "" if 0 in folds else f"{'--':>9}"
                print(f"  {name:<22}{ro:>6}{pad}{cells}{'n/a':>9}{'n/a':>8}"
                      f"   (undefined on folds {undef})")

            # Baseline A: the folds-1-4 basis alongside, for like-for-like
            if 0 in folds:
                sub = [comps[f] for f in folds if f != 0]
                if all(x is not None for x in sub):
                    t2 = paired_t(sub)
                    entry["folds_1_4"] = {
                        "fold_mean": st.mean(sub), "sd": t2["sd"],
                        "pct_of_lines": 100.0 * st.mean(sub) / LINES_COMPLETENESS,
                        "ci95": [t2["ci_lo"], t2["ci_hi"]]}
            rec["readouts"][str(ro)] = entry
        out["arms"][name] = rec

    print("\n  BASIS WARNING carried from 2: a folds-1-4 number is never compared against a")
    print("  5-fold one. Round 1's frozen 5-fold 0.1412 (28%) and its folds-1-4 0.1101")
    print("  (22.0%) already have artifacts (runs/bakeoff_*/metric_suite.json) and are not")
    print("  recomputed here.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n[out] {os.path.relpath(os.path.abspath(a.out), REPO)}")

    _verify_integrity(where="after writing the Lines ratios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
