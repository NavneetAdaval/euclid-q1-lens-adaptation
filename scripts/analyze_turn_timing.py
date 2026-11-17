"""
analyze_turn_timing.py
======================
ANALYSIS ONLY -- no GPU, no new runs, no change to the §7.11 pre-registration.

Asks three things of the artifacts already on disk:

  Q1  On fold 0, does any LABEL-FREE diagnostic turn where TPR@1e-3 turns?
  Q2  Do the other folds turn at the SAME step, on whatever range each covers?
  Q3  Would a label-free stopping rule have selected a good checkpoint?

WHAT EACH FOLD CAN AND CANNOT ANSWER -- read this before the numbers
-------------------------------------------------------------------
  fold 0  UNFROZEN, 8000 steps, 65 curve points. DEVELOPMENT fold. The only run that
          spans the full rise-and-decay, so it is the only one where "where does TPR turn"
          is directly observable.
  fold 1  UNFROZEN, ~3375 steps, 28 curve points (killed mid-run). Spans the transition, so
          it is the ONE independent test of whether the turn timing generalises. Its
          step-2000 TPR is QUARANTINED by the pre-registration and is not carried into any
          comparison here.
  fold 3  **FROZEN (n_blocks=0)** -- the continuation spot check, not an unfrozen run.
          It has NO unfreezing decay to turn, so it cannot corroborate turn timing.
          It is instead the STEP CONTROL this analysis needs: any diagnostic that moves on
          fold 3 as well is tracking TRAINING TIME, not the unfreezing-specific decay
          (§7.6 -- the null for a trajectory signal is the step, never zero).

So Q2 rests on fold 1 alone. That is one independent fold, and the report says so rather
than presenting fold 3 as a second confirmation.

Run:
    python scripts/analyze_turn_timing.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import metric_suite, DEFAULT_TARGET_FPR, DEFAULT_TARGET_PURITY, \
    DEFAULT_TARGET_COMPLETENESS, NN_NP_Q1                      # noqa: E402

ONE_POSITIVE = 1.0 / 98.0

# label-free series: every diagnostic computable with NO test labels
LABEL_FREE = [
    ("memorization_gap", "memorization.csv", "target_minus_test_matched"),
    ("tau_global",       "diagnostics.csv",  "tau_global_last"),
    ("local_pos",        "diagnostics.csv",  "local_pos_last"),
    ("acceptance",       "diagnostics.csv",  "acceptance_mean"),
    ("mask_rate_pos",    "diagnostics.csv",  "mask_rate_pos_mean"),
    ("mask_rate_neg",    "diagnostics.csv",  "mask_rate_neg_mean"),
    ("pseudo_pos_frac",  "diagnostics.csv",  "pseudo_pos_fraction_all_mean"),
    ("accepted_pos_frac", "diagnostics.csv", "accepted_pos_fraction_mean"),
    ("pbar_pos",         "diagnostics.csv",  "pbar_pos_last"),
    ("sup_loss",         "diagnostics.csv",  "sup_loss_mean"),
    ("consistency",      "diagnostics.csv",  "consistency_mean"),
    ("saf",              "diagnostics.csv",  "saf_mean"),
    ("conf_target",      "memorization.csv", "conf_target_mean"),
    ("conf_test_matched", "memorization.csv", "conf_test_matched"),
]


def read_rows(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v) if v not in (None, "") else None
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
    return out


def load_fold(run_dir, fold):
    """TPR curve + every label-free series, aligned on step."""
    curve = read_rows(os.path.join(run_dir, "curve.csv"))
    tpr = {int(r["step"]): r.get(f"fold{fold}_tpr_1e-3") for r in curve
           if r.get("step") is not None}
    agr = {int(r["step"]): r.get(f"agreement_fold{fold}") for r in curve
           if r.get("step") is not None}
    files = {}
    series = {}
    for name, fn, col in LABEL_FREE:
        if fn not in files:
            files[fn] = {int(r["step"]): r for r in read_rows(os.path.join(run_dir, fn))
                         if r.get("step") is not None}
        s = {k: v.get(col) for k, v in files[fn].items() if v.get(col) is not None}
        if len(s) >= 5:
            series[name] = s
    series["agreement"] = {k: v for k, v in agr.items() if v is not None}
    steps = sorted(k for k, v in tpr.items() if v is not None)
    return steps, tpr, series


def secondary_metric(run_dir, fold, step, key="comp_at_purity"):
    p = os.path.join(run_dir, "logits", f"step_{int(step)}.npz")
    if not os.path.isfile(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        pos, neg = z[f"test_pos_fold{fold}"], z["test_negatives_shared"]
    y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
    s = np.concatenate([pos, neg]).astype(float)
    v = metric_suite(y, s, target_fpr=DEFAULT_TARGET_FPR,
                     target_completeness=DEFAULT_TARGET_COMPLETENESS,
                     nn_np_ratio=NN_NP_Q1, target_purity=DEFAULT_TARGET_PURITY)[key]
    return None if (v is None or not np.isfinite(v)) else float(v)


# ---------------------------------------------------------------------------
# where does a series turn?
# ---------------------------------------------------------------------------

def describe(steps, vals):
    """Monotonicity, saturation point, and where the first derivative dies."""
    s = np.asarray(steps, float)
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    s, v = s[ok], v[ok]
    if s.size < 5:
        return None
    rng = float(v.max() - v.min())
    rising = v[-1] >= v[0]
    # fraction of consecutive steps moving the dominant direction = monotonicity
    d = np.diff(v)
    mono = float(np.mean(d >= 0) if rising else np.mean(d <= 0))
    # SATURATION: first step at which the series has covered 90% of its total travel.
    # For a monotone-saturating quantity (tau, acceptance) this is the "it has stopped
    # moving" point, and it is what a threshold rule would fire on.
    target = v[0] + 0.90 * (v[-1] - v[0])
    idx = np.where(v >= target)[0] if rising else np.where(v <= target)[0]
    sat = int(s[idx[0]]) if idx.size else None
    # half-travel, the analogue of a midpoint crossing
    half = v[0] + 0.50 * (v[-1] - v[0])
    idx2 = np.where(v >= half)[0] if rising else np.where(v <= half)[0]
    mid = int(s[idx2[0]]) if idx2.size else None
    return {"first": float(v[0]), "last": float(v[-1]), "range": rng,
            "rising": bool(rising), "monotonicity": mono,
            "half_travel_step": mid, "saturation_step_90pct": sat,
            "spearman_vs_step": _spear(s, v)}


def _spear(x, y):
    if len(x) < 3:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(x)),
                             np.argsort(np.argsort(y)))[0, 1])


def smooth_argmax(steps, vals, w=3):
    """Argmax of a w-point moving average -- fold-0's per-checkpoint sd is 0.0641, so the
    raw argmax is the top of many noisy draws (§7.8). The smoothed peak is the location
    estimate a budget could actually rest on."""
    s = np.asarray(steps, float)
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    s, v = s[ok], v[ok]
    if s.size < w:
        return None, None
    k = np.ones(w) / w
    sm = np.convolve(v, k, mode="valid")
    off = (w - 1) // 2
    i = int(np.argmax(sm))
    return int(s[off + i]), float(sm[i])


# ---------------------------------------------------------------------------
# Q3 -- candidate stopping rules
# ---------------------------------------------------------------------------

def rule_first_crossing(steps, series, thr, rising=True):
    s = np.asarray(steps, float)
    v = np.asarray([series.get(int(x)) for x in s], dtype=float)
    ok = np.isfinite(v)
    s, v = s[ok], v[ok]
    idx = np.where(v >= thr)[0] if rising else np.where(v <= thr)[0]
    return int(s[idx[0]]) if idx.size else None


def evaluate_pick(steps, tpr, picked):
    if picked is None:
        return None
    s = np.asarray(steps, float)
    v = np.asarray([tpr[int(x)] for x in s], dtype=float)
    j = int(np.argmin(np.abs(s - picked)))
    return {"step": int(s[j]), "tpr": float(v[j]), "oracle": float(np.nanmax(v)),
            "regret": float(np.nanmax(v) - v[j]),
            "regret_positives": float((np.nanmax(v) - v[j]) / ONE_POSITIVE)}


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(root, "runs", "turn_timing.json"))
    args = ap.parse_args(argv)

    RUNS = {
        0: (os.path.join(root, "runs", "freematch_fold0_nb3_unfrozen"), 0,
            "UNFROZEN 8000 (development)"),
        1: (os.path.join(root, "runs", "freematch_nb3_unfrozen_fold1"), 1,
            "UNFROZEN ~3375 (killed mid-run; step-2000 value QUARANTINED)"),
        3: (os.path.join(root, "runs", "freematch_nb0_control_fold3"), 3,
            "**FROZEN** spot check -- the STEP CONTROL, not an unfrozen run"),
    }

    print("#" * 104)
    print("# TURN-TIMING ANALYSIS -- existing artifacts only. §7.11 unchanged.")
    print("#" * 104)
    for k, (_d, _f, what) in RUNS.items():
        print(f"  fold {k}: {what}")
    print("\n  Q2 therefore rests on FOLD 1 ALONE. Fold 3 is frozen: it has no unfreezing")
    print("  decay to turn, and serves as the step control instead.")

    result = {}
    for k, (d, fold, what) in RUNS.items():
        if not os.path.isdir(d):
            print(f"\n[skip] fold {k}: {d} not found")
            continue
        steps, tpr, series = load_fold(d, fold)
        tv = [tpr[s] for s in steps]
        raw_i = int(np.nanargmax(tv))
        sm_step, sm_val = smooth_argmax(steps, tv, w=3)

        print("\n" + "=" * 104)
        print(f"FOLD {k} -- {what}")
        print("=" * 104)
        print(f"  curve points {len(steps)}  range {steps[0]}..{steps[-1]}")
        print(f"  TPR raw argmax  {tv[raw_i]:.4f} @ step {steps[raw_i]}")
        print(f"  TPR 3-pt smoothed argmax {sm_val:.4f} @ step {sm_step}"
              if sm_step else "  (too few points to smooth)")
        print(f"  step0 {tv[0]:.4f}  final {tv[-1]:.4f}  "
              f"sd {np.nanstd(tv):.4f}  range {np.nanmax(tv)-np.nanmin(tv):.4f} "
              f"({(np.nanmax(tv)-np.nanmin(tv))/ONE_POSITIVE:.0f} positives)")

        print(f"\n  FULL TPR@1e-3 CURVE:")
        for i, s in enumerate(steps):
            mark = "  <-- raw argmax" if i == raw_i else ("  <-- smoothed peak" if s == sm_step else "")
            print(f"    {s:>5}  {tpr[s]:.4f}{mark}")

        print(f"\n  LABEL-FREE SERIES -- where each turns:")
        print(f"    {'series':<20}{'first':>9}{'last':>9}{'range':>9}{'mono':>7}"
              f"{'half@':>7}{'sat90@':>8}{'rho(step)':>10}")
        desc = {}
        for name in series:
            dd = describe(steps, [series[name].get(s) for s in steps])
            if dd is None:
                continue
            desc[name] = dd
            print(f"    {name:<20}{dd['first']:>9.4f}{dd['last']:>9.4f}{dd['range']:>9.4f}"
                  f"{dd['monotonicity']:>7.2f}{str(dd['half_travel_step']):>7}"
                  f"{str(dd['saturation_step_90pct']):>8}"
                  f"{(f'{dd['spearman_vs_step']:+.3f}' if dd['spearman_vs_step'] is not None else 'n/a'):>10}")

        result[k] = {"what": what, "n_points": len(steps),
                     "steps": steps, "tpr": [tpr[s] for s in steps],
                     "raw_argmax_step": steps[raw_i], "raw_argmax": tv[raw_i],
                     "smoothed_argmax_step": sm_step, "smoothed_argmax": sm_val,
                     "label_free": desc,
                     "frozen": (k == 3)}

    # ---- the transition window on fold 0, and what the secondary metric does there ----
    if 0 in result:
        d0 = RUNS[0][0]
        print("\n" + "=" * 104)
        print("FOLD 0 TRANSITION WINDOW -- two metrics at different operating points")
        print("=" * 104)
        print(f"  {'step':>6}{'TPR@1e-3':>10}{'comp@24%':>10}   label-free at that step")
        for s in (1250, 1375, 1500, 1625, 1750, 1875, 2000):
            c = secondary_metric(d0, 0, s)
            lf = result[0]["label_free"]
            _steps, _tpr, ser = load_fold(d0, 0)
            bits = []
            for nm in ("memorization_gap", "tau_global", "acceptance"):
                v = ser.get(nm, {}).get(s)
                if v is not None:
                    bits.append(f"{nm}={v:.4f}")
            print(f"  {s:>6}{_tpr.get(s, float('nan')):>10.4f}"
                  f"{(c if c is not None else float('nan')):>10.4f}   {'  '.join(bits)}")

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\n[out] {args.out}")
    return result


if __name__ == "__main__":
    main()
