"""
analyze_coral_gamma_sweep.py
============================
LOCAL evaluation of the round-4 CORAL gamma sweep's PRE-REGISTERED selection criterion.

This is CPU work on small CSVs (2000 rows per arm) and deliberately does NOT run on the
Kaggle box: GPU session time is the scarce resource and none of this needs a GPU. Download
the three sweep directories, point ``--sweep-dir`` at their parent, and run it here.

The criterion is ``coral_round4.SWEEP_CRITERION``. It was fixed in code and printed by
``run_coral_gamma_sweep.py`` before the first optimizer step of the first arm. This script
prints it again before reporting anything, and implements it exactly:

  (i)  raw CORAL is NOT monotonically rising over the SECOND HALF of the window
       -> OLS slope vs step, moving-block bootstrap (block 50, B=5000)
       -> PASS iff the 95% CI's LOWER bound <= 0 (fail only if demonstrably rising)
  (ii) feature-norm growth over the FULL window is measurably damped vs the gamma=1.0 arm
       -> growth = mean(last 10%) - mean(first 10%) on the SOURCE domain
       -> PASS iff the 95% CI of [growth(gamma) - growth(1.0)] lies entirely BELOW zero
       -> INAPPLICABLE (not passed) if growth(1.0) <= 0

Choose the SMALLEST gamma passing both. If none does, report that and pick nothing --
the pre-registered instruction is to propose extending the sweep upward, never to relax
either clause.

Reads ``steps.csv`` (the per-step series) because the slope fit and the block bootstrap want
every step, not the ~16-point ``trace.csv`` summary. Falls back to ``trace.csv`` only if
``steps.csv`` is absent, and says so.

Run:
    python scripts/analyze_coral_gamma_sweep.py --sweep-dir runs/coral_r4_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from coral_round4 import (SWEEP_CRITERION, SWEEP_GAMMAS, SWEEP_BLOCK, SWEEP_B,
                          SWEEP_ALPHA, slope_with_block_ci, norm_growth_with_ci,
                          damping_verdict)

REF_GAMMA = 1.0


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_arm(d):
    """(steps, per-step dict of arrays, meta) for one sweep directory."""
    steps_csv = os.path.join(d, "steps.csv")
    trace_csv = os.path.join(d, "trace.csv")
    meta = {}
    sj = os.path.join(d, "sweep.json")
    if os.path.isfile(sj):
        meta = json.load(open(sj))
    if os.path.isfile(steps_csv):
        rows, src, suffix = _read_csv(steps_csv), "steps.csv", ""
        cols = {"step": "step", "coral": "coral_raw", "sup": "sup_loss",
                "src_l2": "src_feat_l2", "tgt_l2": "tgt_feat_l2",
                "src_sd": "src_perdim_std", "tgt_sd": "tgt_perdim_std"}
    elif os.path.isfile(trace_csv):
        rows, src, suffix = _read_csv(trace_csv), "trace.csv", "_mean"
        cols = {"step": "step", "coral": "coral_raw_mean", "sup": "sup_loss_mean",
                "src_l2": "src_feat_l2_mean", "tgt_l2": "tgt_feat_l2_mean",
                "src_sd": "src_perdim_std_mean", "tgt_sd": "tgt_perdim_std_mean"}
        print(f"  [warn] {d}: steps.csv absent -- falling back to the {len(rows)}-point "
              f"trace.csv. The slope fit and block bootstrap are coarser on it.")
    else:
        raise SystemExit(f"[analyze] no steps.csv or trace.csv in {d}")

    def col(name):
        out = []
        for r in rows:
            v = r.get(cols[name], "")
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.asarray(out, float)

    data = {k: col(k) for k in cols}
    return data, {"dir": d, "source": src, "n_rows": len(rows), **meta}


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", default=os.path.join(root, "runs", "coral_r4_sweep"),
                   help="parent holding g1/, g4/, g16/ (or --arm-dir overrides)")
    p.add_argument("--arm-dir", action="append", default=None,
                   metavar="GAMMA=PATH", help="explicit gamma=path, repeatable")
    p.add_argument("--block", type=int, default=SWEEP_BLOCK)
    p.add_argument("--bootstrap", type=int, default=SWEEP_B)
    p.add_argument("--alpha", type=float, default=SWEEP_ALPHA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--md", default=None)
    args = p.parse_args(argv)

    print("=" * 92)
    print("ROUND-4 CORAL GAMMA SWEEP -- criterion evaluation (LOCAL, CPU)")
    print("=" * 92)
    print(SWEEP_CRITERION)
    print("=" * 92, flush=True)

    # ---- locate the arms ----
    arms = {}
    if args.arm_dir:
        for spec in args.arm_dir:
            g, _, path = spec.partition("=")
            arms[float(g)] = os.path.abspath(path)
    else:
        for g in SWEEP_GAMMAS:
            d = os.path.join(args.sweep_dir, f"g{int(g)}")
            if os.path.isdir(d):
                arms[float(g)] = d
    if REF_GAMMA not in arms:
        raise SystemExit(
            f"[analyze] the gamma={REF_GAMMA:g} reference arm is required for criterion "
            f"(ii) and was not found under {args.sweep_dir}. Found: {sorted(arms)}")

    out = {"criterion": SWEEP_CRITERION, "reference_gamma": REF_GAMMA,
           "block": int(args.block), "bootstrap": int(args.bootstrap),
           "alpha": float(args.alpha), "arms": {}}

    # ---- load + per-arm descriptive trace ----
    loaded = {}
    print("\nTRACES")
    print(f"  {'gamma':>6} {'rows':>6} {'steps':>6} {'abort':>6} "
          f"{'sup first->last':>22} {'coral first->last':>24} {'src||f|| first->last':>24}")
    for g in sorted(arms):
        data, meta = load_arm(arms[g])
        loaded[g] = (data, meta)
        n = data["step"].size
        h = max(2, int(0.10 * n))
        def fl(k):
            return f"{np.nanmean(data[k][:h]):.4f} -> {np.nanmean(data[k][-h:]):.4f}"
        print(f"  {g:>6g} {meta['n_rows']:>6} {int(np.nanmax(data['step'])):>6} "
              f"{str(bool(meta.get('aborted'))):>6} {fl('sup'):>22} "
              f"{fl('coral'):>24} {fl('src_l2'):>24}")
        out["arms"][str(g)] = {
            "dir": meta["dir"], "source": meta["source"], "n_rows": meta["n_rows"],
            "steps_completed": int(meta.get("steps_completed", np.nanmax(data["step"]))),
            "aborted": bool(meta.get("aborted", False)),
            "abort_reason": meta.get("abort_reason"),
            "sup_first10pct": float(np.nanmean(data["sup"][:h])),
            "sup_last10pct": float(np.nanmean(data["sup"][-h:])),
            "coral_first10pct": float(np.nanmean(data["coral"][:h])),
            "coral_last10pct": float(np.nanmean(data["coral"][-h:])),
            "src_l2_first10pct": float(np.nanmean(data["src_l2"][:h])),
            "src_l2_last10pct": float(np.nanmean(data["src_l2"][-h:])),
            "tgt_l2_first10pct": float(np.nanmean(data["tgt_l2"][:h])),
            "tgt_l2_last10pct": float(np.nanmean(data["tgt_l2"][-h:])),
            "src_sd_last10pct": float(np.nanmean(data["src_sd"][-h:])),
            "tgt_sd_last10pct": float(np.nanmean(data["tgt_sd"][-h:])),
        }

    # ---- WINDOWING: longest window common to all arms (an abort truncates) ----
    common = int(min(np.nanmax(d["step"]) for d, _ in loaded.values()))
    out["common_window_steps"] = common
    truncated = [g for g in sorted(arms)
                 if int(np.nanmax(loaded[g][0]["step"])) > common]
    if truncated:
        print(f"\n  [window] an arm stopped early -- all comparisons use the longest "
              f"COMMON window, steps 1..{common}. Truncated for comparison: {truncated}")
    out["truncated_for_comparison"] = [float(g) for g in truncated]

    def clip(d):
        m = d["step"] <= common
        return {k: v[m] for k, v in d.items()}

    # ---- criterion (i): raw CORAL slope over the SECOND HALF ----
    half = common / 2.0
    print(f"\nCRITERION (i) -- raw CORAL slope over steps ({half:.0f}, {common}]  "
          f"(block {args.block}, B={args.bootstrap})")
    print(f"  {'gamma':>6} {'slope/step':>14} {'95% CI':>30} {'not rising':>11} "
          f"{'strictly falling':>17}")
    for g in sorted(arms):
        d = clip(loaded[g][0])
        m = d["step"] > half
        s = slope_with_block_ci(d["step"][m], d["coral"][m], block=args.block,
                                B=args.bootstrap, alpha=args.alpha, seed=args.seed)
        out["arms"][str(g)]["criterion_i"] = s
        print(f"  {g:>6g} {s['slope']:>+14.3e} "
              f"[{s['ci'][0]:>+.3e}, {s['ci'][1]:>+.3e}] {str(s['not_rising']):>11} "
              f"{str(s['strictly_falling']):>17}")

    # ---- criterion (ii): feature-norm growth vs the gamma=1.0 arm ----
    print(f"\nCRITERION (ii) -- source feature-norm growth vs the gamma={REF_GAMMA:g} arm")
    growths = {}
    for g in sorted(arms):
        d = clip(loaded[g][0])
        growths[g] = norm_growth_with_ci(d["step"], d["src_l2"], block=args.block,
                                         B=args.bootstrap, alpha=args.alpha,
                                         seed=args.seed + int(g))
        tg = norm_growth_with_ci(d["step"], d["tgt_l2"], block=args.block,
                                 B=args.bootstrap, alpha=args.alpha, seed=args.seed + int(g))
        out["arms"][str(g)]["src_norm_growth"] = {
            k: v for k, v in growths[g].items() if k != "draws"}
        out["arms"][str(g)]["tgt_norm_growth"] = {k: v for k, v in tg.items()
                                                  if k != "draws"}
        out["arms"][str(g)]["domains_agree_in_sign"] = bool(
            np.sign(growths[g]["growth"]) == np.sign(tg["growth"]))
    ref = growths[REF_GAMMA]
    print(f"  reference growth (gamma={REF_GAMMA:g}) = {ref['growth']:+.4f}")
    print(f"  {'gamma':>6} {'growth':>10} {'diff vs ref':>13} {'95% CI of diff':>28} "
          f"{'damped':>8}")
    for g in sorted(arms):
        if g == REF_GAMMA:
            print(f"  {g:>6g} {growths[g]['growth']:>+10.4f} {'(reference)':>13} "
                  f"{'':>28} {'n/a':>8}")
            out["arms"][str(g)]["criterion_ii"] = {"reference": True, "passed": None}
            continue
        v = damping_verdict(growths[g], ref, alpha=args.alpha)
        out["arms"][str(g)]["criterion_ii"] = {k: val for k, val in v.items()}
        ci = (f"[{v['ci'][0]:>+.4f}, {v['ci'][1]:>+.4f}]" if v.get("ci") else "n/a")
        print(f"  {g:>6g} {growths[g]['growth']:>+10.4f} {v['diff']:>+13.4f} {ci:>28} "
              f"{str(v['passed']):>8}")

    # ---- the verdict ----
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    passing = []
    for g in sorted(arms):
        a = out["arms"][str(g)]
        if a["aborted"]:
            print(f"  gamma={g:<5g} ABORTED at step {a['steps_completed']} "
                  f"(feature-norm ceiling) -- an arm that aborts has not passed anything")
            a["passes_both"] = False
            continue
        i_ok = bool(a["criterion_i"]["not_rising"])
        ii = a["criterion_ii"]
        ii_ok = bool(ii.get("passed")) if not ii.get("reference") else None
        a["passes_both"] = bool(i_ok and ii_ok) if ii_ok is not None else False
        if g == REF_GAMMA:
            print(f"  gamma={g:<5g} (i) {'PASS' if i_ok else 'FAIL'}   "
                  f"(ii) n/a -- this arm IS the reference, never a candidate")
        else:
            note = "" if ii.get("applicable", True) else f"  [{ii.get('reason')}]"
            print(f"  gamma={g:<5g} (i) {'PASS' if i_ok else 'FAIL'}   "
                  f"(ii) {'PASS' if ii_ok else 'FAIL'}   "
                  f"-> {'PASSES BOTH' if a['passes_both'] else 'does not pass'}{note}")
        if a["passes_both"]:
            passing.append(g)

    if passing:
        sel = min(passing)
        out["selected_gamma"] = float(sel)
        out["passing_gammas"] = [float(x) for x in passing]
        print(f"\n  SELECTED gamma = {sel:g}  (the SMALLEST passing both clauses; "
              f"passing set {passing})")
    else:
        out["selected_gamma"] = None
        out["passing_gammas"] = []
        out["recommendation"] = ("NO gamma in the swept set satisfies both clauses. The "
                                 "pre-registered instruction is to EXTEND THE SWEEP UPWARD "
                                 "(gamma 64, then 256) and NOT to relax either clause.")
        print("\n  NO GAMMA SATISFIES BOTH CLAUSES. Nothing is selected.")
        print("  Per the pre-registered criterion: extend the sweep upward (gamma 64, then")
        print("  256). Do NOT relax either clause to make one of {1, 4, 16} qualify.")
    print("=" * 92)

    outp = args.out or os.path.join(root, "runs", "coral_r4_sweep_verdict.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[out] {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
