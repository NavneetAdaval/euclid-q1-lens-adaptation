"""
sweep_metrics_from_logits.py
============================
Recompute the FULL metric suite at EVERY curve point, from the dumped logits. No training,
no re-encoding, no new runs -- this is exactly what the derived-logit dump (BRIEF 5.2) was
built to make possible.

For each curve point of each run: TPR@FPR=1e-3, completeness@24%-purity, purity@50%-
completeness (both corrected to the Lines Q1 rarity N_N/N_P = 2200), AUPRC and AUROC. Then
the argmax checkpoint per metric, and whether the metrics agree on which checkpoint is best.

A run with only weights and no logit dumps (every round-1 run) is handled by scoring its
checkpoints against the frozen feature cache -- valid there precisely because those runs
were frozen.

Run:
    python scripts/sweep_metrics_from_logits.py
    python scripts/sweep_metrics_from_logits.py --runs runs/freematch_fold0_nb3_unfrozen
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import (metric_suite, DEFAULT_TARGET_FPR, DEFAULT_TARGET_COMPLETENESS,
                     DEFAULT_TARGET_PURITY, NN_NP_Q1)     # noqa: E402

KEYS = ("tpr", "comp_at_purity", "purity_corr", "auprc", "auroc")
LABEL = {"tpr": "TPR@1e-3", "comp_at_purity": f"comp@{DEFAULT_TARGET_PURITY:.0%}-purity",
         "purity_corr": f"purity@{DEFAULT_TARGET_COMPLETENESS:.0%}-comp",
         "auprc": "AUPRC", "auroc": "AUROC"}
# Lines et al. arXiv:2512.05899, simulation-only model on real Euclid Q1
LINES_SIMONLY_COMPLETENESS = 0.50


def suite_from(pos, neg, fold=0):
    y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
    s = np.concatenate([pos, neg]).astype(float)
    return metric_suite(y, s, target_fpr=DEFAULT_TARGET_FPR,
                        target_completeness=DEFAULT_TARGET_COMPLETENESS,
                        nn_np_ratio=NN_NP_Q1, target_purity=DEFAULT_TARGET_PURITY)


def sweep_logits(run_dir, fold=0):
    rows = []
    files = sorted(glob.glob(os.path.join(run_dir, "logits", "step_*.npz")),
                   key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    for p in files:
        step = int(re.search(r"step_(\d+)", p).group(1))
        with np.load(p, allow_pickle=False) as z:
            pos, neg = z[f"test_pos_fold{fold}"], z["test_negatives_shared"]
        rows.append({"step": step, **{k: _num(suite_from(pos, neg)[k]) for k in KEYS}})
    return rows


def sweep_checkpoints(run_dir, cache_dir, fold=0, device="cpu"):
    """For runs with no logit dumps: score each checkpoint off the FROZEN feature cache.

    Only valid for frozen runs -- which is exactly what the round-1 reference is.
    """
    import evaluate as ev
    cache = ev.FeatureCache(cache_dir)
    posf = cache.load(f"test_pos_fold{fold}")["features"]
    negf = cache.load("test_negatives_shared")["features"]
    rows = []
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pt")),
                 key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    for p in cks:
        step = int(re.search(r"step_(\d+)", p).group(1))
        head, std, _ = ev.load_checkpoint(p, device)
        pos = ev.score_features(head, std, posf, device)
        neg = ev.score_features(head, std, negf, device)
        rows.append({"step": step, **{k: _num(suite_from(pos, neg)[k]) for k in KEYS}})
    return rows


def _num(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def argmaxes(rows):
    out = {}
    for k in KEYS:
        cand = [(r[k], r["step"]) for r in rows if r.get(k) is not None]
        out[k] = {"value": max(cand)[0], "step": max(cand)[1]} if cand else None
    return out


def report(name, rows, every=1):
    print("\n" + "=" * 100)
    print(f"{name}   ({len(rows)} curve points)")
    print("=" * 100)
    hdr = f"  {'step':>6} " + " ".join(f"{LABEL[k]:>18}" for k in KEYS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, r in enumerate(rows):
        if i % every and i != len(rows) - 1:
            continue
        print(f"  {r['step']:>6} " + " ".join(
            (f"{r[k]:>18.4f}" if r.get(k) is not None else f"{'n/a':>18}") for k in KEYS))
    am = argmaxes(rows)
    print("\n  ARGMAX per metric:")
    for k in KEYS:
        a = am[k]
        print(f"    {LABEL[k]:<22} {a['value']:.4f}  @ step {a['step']}"
              if a else f"    {LABEL[k]:<22} n/a")
    steps = {k: am[k]["step"] for k in KEYS if am[k]}
    uniq = sorted(set(steps.values()))
    print(f"\n  metrics agree on the best checkpoint: {len(uniq) == 1}"
          f"   (distinct argmax steps: {uniq})")
    return {"rows": rows, "argmax": am, "argmax_steps": steps,
            "metrics_agree": len(uniq) == 1}


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[
        os.path.join(root, "runs", "freematch_fold0_nb3_unfrozen"),
        os.path.join(root, "runs", "freematch_fold0_nb0_control_FIXED"),
    ])
    ap.add_argument("--round1", default=os.path.join(root, "runs",
                                                     "bakeoff_freematch_5fold", "fold0"))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="print every Nth row")
    ap.add_argument("--out", default=os.path.join(root, "runs", "metric_sweep.json"))
    args = ap.parse_args(argv)

    result = {}
    for rd in args.runs:
        rd = os.path.abspath(rd)
        if not os.path.isdir(rd):
            print(f"[skip] {rd} not found")
            continue
        rows = sweep_logits(rd, args.fold)
        if not rows:
            print(f"[skip] {rd} has no logit dumps")
            continue
        result[os.path.basename(rd)] = report(os.path.basename(rd), rows, args.every)

    if args.round1 and os.path.isdir(args.round1):
        rows = sweep_checkpoints(os.path.abspath(args.round1),
                                 os.path.abspath(args.cache_dir), args.fold)
        if rows:
            result["round1_fold0"] = report(
                "round-1 FreeMatch fold 0 (frozen; scored from the feature cache)",
                rows, args.every)

    print("\n" + "=" * 100)
    print(f"LINES et al. sim-only reference: completeness@24%-purity = "
          f"{LINES_SIMONLY_COMPLETENESS:.2f}")
    print("=" * 100)
    for name, r in result.items():
        a = r["argmax"].get("comp_at_purity")
        if a:
            print(f"  {name:<42} peak {a['value']:.4f} @ step {a['step']}   "
                  f"= {a['value']/LINES_SIMONLY_COMPLETENESS:.1%} of Lines")

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
