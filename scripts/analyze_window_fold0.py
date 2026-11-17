"""
analyze_window_fold0.py
=======================
ROUND 7, TASK 3, PART B -- 7.16's EIGHTH ENTRY and ROW 9, tested on the data they were
measured on. **DECLARED SECONDARY ANALYSIS.** Replaces no hashed headline.

    python -u scripts/analyze_window_fold0.py --out runs/round7/window_fold0.json

WHY THIS IS A SEPARATE SCRIPT, AND THE DISCLOSURE IT CARRIES. The declared scope in
``runs/round7/window_prereg.json`` is the confirmatory folds at each arm's mandatory
readouts, and ``analyze_window_average.py`` executed exactly that. But 7.16's eighth
entry is a statement about **fold 0 on an identical 32-checkpoint 250->8000 grid**, and
7.22's stability row is a statement about **sd/mean on fold 0's two round-5 arms**.
Testing either on folds 1-4 tests a different quantity. This part was therefore added
AFTER the folds-1-4 run, and it uses:

  * the GRID 7.16 names (250..8000 step 250, 32 points; FreeMatch's 125-cadence run is
    subsampled to that grid so every arm is on the identical one, as 7.16 requires);
  * the STATISTIC each entry names -- absolute checkpoint sd for the eighth entry,
    sd/mean for 7.22's stability row;
  * the SAME k, window rule and score combinations as part A, read from the same hashed
    declaration. Nothing tunable was re-chosen.

Fold 0 is development. Nothing here is a ranking claim, enters a mean, or is differenced
against a confirmatory number. It answers one question: whether the scatter 7.16 measures
is a property of the data or of the single-point read-off.

Writes NOTHING outside --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics as st
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

from analyze_window_average import (PREREG, evaluate, _scan_dir, L)  # noqa: E402
from integrity import verify as _verify_integrity                    # noqa: E402

GRID = list(range(250, 8001, 250))          # the identical 32-point grid 7.16 names

# 7.16's eighth-entry table, verbatim, so the reproduction can be checked against it.
SEVEN_SIXTEEN = {"CORAL x nb3": 0.0513, "FreeMatch x nb3": 0.0606,
                 "DANN arm 2 (lambda live)": 0.0914, "DANN arm 1 (lambda=0)": 0.0402}
DANN_MINUS_CORAL = 0.018                    # the difference the table is asked to resolve

ARMS_F0 = {
    "CORAL x nb3": L("kaggle", "coral_r4", "runs", "fold0_run",
                     "coral_r4_nb3_dev8000_fold0", "logits", "g16dev8000"),
    # the 8000-step fold-0 DIAGNOSTIC run (7.10), not the 1500-step confirmatory fold 0:
    # 7.16's FreeMatch entry is on the 32-point 250..8000 grid, which only this run has.
    "FreeMatch x nb3": L("runs", "freematch_nb3_fold0", "_x",
                         "freematch_fold0_nb3_unfrozen", "logits"),
    "DANN arm 2 (lambda live)": L("runs", "dann_arm2_fold0", "rescore",
                                  "dann_r5_nb3_fold0", "logits", "dann_nb3_fold0"),
    "DANN arm 1 (lambda=0)": L("runs", "dann_arm2_fold0", "rescore",
                               "dann_r5_nb3_lambda0_fold0", "logits",
                               "dann_nb3_lambda0_fold0"),
    "BNM x nb3": L("runs", "bnm_r6_dev8000", "bnm_r6_dev8000_fold0", "logits",
                   "bnm_nb3_fold0"),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    _verify_integrity(where="before the round-7 fold-0 window analysis")

    pre = json.load(open(PREREG, encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k != "sha256_of_body_without_this_field"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != pre["sha256_of_body_without_this_field"]:
        raise SystemExit("the window declaration does not hash to itself -- STOP")
    K_GRID = list(pre["k"]["grid"])
    K_PRIMARY = int(pre["k"]["primary"])

    print("=" * 100)
    print("ROUND 7, TASK 3 PART B -- 7.16's EIGHTH ENTRY AND ROW 9, ON FOLD 0's OWN GRID")
    print("*** DECLARED SECONDARY ANALYSIS -- replaces no hashed headline ***")
    print("*** ADDED AFTER PART A RAN; grid and statistic are 7.16's own, k is unchanged ***")
    print("=" * 100)
    print(f"  declaration {os.path.relpath(PREREG, REPO)}  sha256 {digest[:16]}...")
    print(f"  grid: 250..8000 step 250, {len(GRID)} points, identical for every arm")
    print("  fold 0 is development: no ranking claim, no mean, no cross-round difference")

    out = {"round": 7, "task": "window_averaging_fold0_7_16_entries",
           "status": "DECLARED SECONDARY ANALYSIS -- replaces no hashed headline",
           "scope_disclosure": ("added after part A; grid and statistic taken from 7.16 "
                                "and 7.22 themselves, k and window rule unchanged"),
           "prereg_sha256": digest, "grid": GRID, "arms": {}}

    loaded = {}
    for name, d in ARMS_F0.items():
        all_steps = _scan_dir(d, 0)
        got = {s: all_steps[s] for s in GRID if s in all_steps}
        loaded[name] = got
        print(f"  [{name}] {len(got)}/{len(GRID)} of the grid present"
              f"{'' if len(got) == len(GRID) else '  <-- INCOMPLETE'}")

    # ---------------- the eighth entry ----------------
    print("\n" + "=" * 100)
    print("1. THE EIGHTH ENTRY -- within-run checkpoint sd, one run scored against itself")
    print("=" * 100)
    print("  7.16: 'the scatter a single run shows against itself, across checkpoints 250")
    print("  steps apart, is 2.8-5.1x larger than the differences the nb3 ranking table is")
    print(f"  trying to resolve.' The difference to beat is DANN - CORAL = {DANN_MINUS_CORAL}.")
    print(f"\n  {'arm':<28}{'7.16 sd':>9}" +
          "".join(f"{'k=' + str(k):>10}" for k in K_GRID) +
          f"{'  k=' + str(K_PRIMARY) + ' vs 0.018':>16}")

    res = {}
    for name in ARMS_F0:
        steps = sorted(loaded[name])
        if not steps:
            continue
        row, cells = {}, []
        for k in K_GRID:
            if len(steps) < k:
                row[str(k)] = None
                cells.append(float("nan"))
                continue
            pos_list = [steps[i - k + 1:i + 1] for i in range(k - 1, len(steps))]
            v = [evaluate(loaded[name], w, "mean_logit")["tpr"] for w in pos_list]
            v = [x for x in v if np.isfinite(x)]
            sd = st.stdev(v)
            row[str(k)] = {"n_positions": len(v), "sd": sd, "mean": st.mean(v),
                           "sd_over_mean": sd / st.mean(v) if st.mean(v) else None,
                           "min": min(v), "max": max(v),
                           "multiple_of_dann_minus_coral": sd / DANN_MINUS_CORAL,
                           "ratio_to_k1": None}
            cells.append(sd)
        for k in K_GRID:
            if row.get(str(k)) and row.get("1"):
                row[str(k)]["ratio_to_k1"] = row["1"]["sd"] / row[str(k)]["sd"]
        mult = (row[str(K_PRIMARY)]["multiple_of_dann_minus_coral"]
                if row.get(str(K_PRIMARY)) else float("nan"))
        print(f"  {name:<28}{SEVEN_SIXTEEN.get(name, float('nan')):>9.4f}" +
              "".join(f"{c:>10.4f}" for c in cells) + f"{mult:>15.1f}x")
        res[name] = row
    out["arms"] = res

    print("\n  the k=1 column IS 7.16's table, recomputed -- it must reproduce the")
    print("  ' 7.16 sd' column to within the rounding those figures carry.")

    print("\n  SCATTER REDUCTION vs sqrt(k):")
    print(f"    {'arm':<28}" + "".join(f"{'k=' + str(k):>10}" for k in K_GRID if k > 1) +
          "      (achieved sd_1/sd_k over sqrt(k))")
    for name, row in res.items():
        cells = []
        for k in K_GRID:
            if k == 1:
                continue
            r = row.get(str(k))
            cells.append(r["ratio_to_k1"] / math.sqrt(k) if r else float("nan"))
        print(f"    {name:<28}" + "".join(f"{c:>10.2f}" for c in cells))

    # ---------------- 7.22's stability row ----------------
    print("\n" + "=" * 100)
    print("2. ROW 9 -- 'unfreezing without a DA term is the least stable arm'")
    print("=" * 100)
    print("  7.22's statistic is sd/mean, and its verdict was: holds on 250-8000")
    print("  (lambda=0 0.276 vs DANN 0.202), DISSOLVES grid-matched on 250-4000")
    print("  (0.258 vs 0.256). Both windows, both k, same statistic.")
    row9 = {}
    for win_name, hi in (("250-8000", 8000), ("250-4000", 4000)):
        row9[win_name] = {}
        print(f"\n    window {win_name}")
        print(f"      {'k':>3}  {'lambda=0 sd/mean':>18}{'DANN sd/mean':>15}"
              f"{'gap':>9}   verdict")
        for k in (1, K_PRIMARY):
            vals = {}
            for label, arm in (("lambda=0", "DANN arm 1 (lambda=0)"),
                               ("DANN", "DANN arm 2 (lambda live)")):
                steps = [s for s in sorted(loaded[arm]) if s <= hi]
                pos_list = [steps[i - k + 1:i + 1] for i in range(k - 1, len(steps))]
                v = [evaluate(loaded[arm], w, "mean_logit")["tpr"] for w in pos_list]
                v = [x for x in v if np.isfinite(x)]
                vals[label] = st.stdev(v) / st.mean(v)
            gap = vals["lambda=0"] - vals["DANN"]
            verdict = ("lambda=0 IS the least stable" if gap > 0.02 else
                       "DANN is" if gap < -0.02 else "DISSOLVES (gap < 0.02)")
            print(f"      {k:>3}  {vals['lambda=0']:>18.3f}{vals['DANN']:>15.3f}"
                  f"{gap:>+9.3f}   {verdict}")
            row9[win_name][str(k)] = {"lambda0_sd_over_mean": vals["lambda=0"],
                                      "dann_sd_over_mean": vals["DANN"], "gap": gap,
                                      "verdict": verdict}
    out["row9_stability"] = row9

    # ---------------- the adjacent-checkpoint swing ----------------
    print("\n" + "=" * 100)
    print("3. THE ADJACENT-CHECKPOINT SWING 7.16 QUOTES")
    print("=" * 100)
    print("  7.16: arm 2's curve contains 0.5000 -> 0.1735 -> 0.5816 inside 250 steps --")
    print("  'a collapse and a full recovery between two adjacent checkpoints'. If window")
    print("  averaging is doing what it claims, that swing must shrink.")
    sw = {}
    for name in ARMS_F0:
        steps = sorted(loaded[name])
        if not steps:
            continue
        rec = {}
        for k in (1, K_PRIMARY):
            if len(steps) < k:
                continue
            pos_list = [steps[i - k + 1:i + 1] for i in range(k - 1, len(steps))]
            v = [evaluate(loaded[name], w, "mean_logit")["tpr"] for w in pos_list]
            v = [x for x in v if np.isfinite(x)]
            jumps = [abs(v[i + 1] - v[i]) for i in range(len(v) - 1)]
            rec[str(k)] = {"max_adjacent_jump": max(jumps),
                           "mean_adjacent_jump": st.mean(jumps)}
        if "1" in rec and str(K_PRIMARY) in rec:
            print(f"  {name:<28} max adjacent jump  k=1 {rec['1']['max_adjacent_jump']:.4f}"
                  f"   k={K_PRIMARY} {rec[str(K_PRIMARY)]['max_adjacent_jump']:.4f}"
                  f"   ({rec['1']['max_adjacent_jump'] / rec[str(K_PRIMARY)]['max_adjacent_jump']:.1f}x smaller)")
        sw[name] = rec
    out["adjacent_swing"] = sw

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n[out] {os.path.relpath(os.path.abspath(a.out), REPO)}")

    _verify_integrity(where="after the round-7 fold-0 window analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
