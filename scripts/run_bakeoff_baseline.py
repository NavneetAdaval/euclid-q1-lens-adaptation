"""
run_bakeoff_baseline.py
=======================
MATCHED 5-FOLD BAKE-OFF, run 1 of 4: Baseline A (source-only) at the LOCKED 8000-step
budget (SPEC spec_bakeoff_1_baseline.md; PROJECT.md 7.4).

This is a thin DRIVER, not a new training loop. It calls
``run_baseline.train_source_only`` (the existing runner) and ``evaluate.evaluate`` (the
existing harness) with the pre-registered settings wired in so they cannot drift.

WHY A DRIVER RATHER THAN run_baseline.run_baseline()
----------------------------------------------------
``run_baseline.run_baseline()`` hands ALL its epoch-checkpoints to evaluate.py, which then
picks one by best SOURCE-VAL metric. The bake-off pre-registers a different, stricter rule:

    R1 -- the LAST checkpoint (step 8000), chosen before any result was seen.

So this driver passes evaluate.py exactly ONE candidate: ``checkpoints/step_8000.pt``.
With a single candidate, source-val selection is a no-op and SOURCE-RISK == ORACLE == R1
by construction -- R1 is enforced structurally, not by discipline. (PROJECT.md 7.6:
source-val AUROC saturates ~0.99 and is a DEAD selector anyway, so letting it choose would
be selection-by-noise.)

ONE TRAINING RUN, FIVE-FOLD EVALUATION (the spec's §3 corrected)
----------------------------------------------------------------
Baseline A is SOURCE-ONLY: training never touches fold-specific data, so the fold enters
ONLY at evaluation. Training five times at one seed would produce five identical models;
varying the seed would measure SEED variance, not fold variance, and would break
PROJECT.md 4.4's requirement that the FPR=1e-3 threshold be identical across folds (one
model scoring the shared negative pool once). Hence: train once, evaluate across all five
folds. The per-fold TEST score vectors -- the paired-bootstrap comparator every DA method
is measured against -- are produced exactly as specified.

BUDGET ARITHMETIC (the 187-vs-188 correction)
----------------------------------------------
The class-balanced sampler yields ceil(48000/256) = 188 batches/epoch, not the 187 the
spec states (48000/256 = 187.5). So 8000 steps = 42.55 epochs: we train 43 epochs (8084
steps) and take the STEP-checkpoint at exactly 8000. The ~84 steps past the budget are
never evaluated and never selected from.

Run:
    python scripts/run_bakeoff_baseline.py
    python scripts/run_bakeoff_baseline.py --out-dir runs/bakeoff_baseline_A --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import evaluate as ev                                    # noqa: E402
from run_baseline import train_source_only               # noqa: E402  (existing runner)
from metrics import DEFAULT_TARGET_FPR                   # noqa: E402

# ---- PRE-REGISTERED SETTINGS (SPEC §1) -- frozen before the run, not tuned on results ----
BUDGET_STEPS = 8000        # [RB] locked step budget (PROJECT.md 7.4)
CURVE_EVERY = 250          # [EC] ~32 curve points, so R1/R2/R3 stay re-checkable at 5-fold res
EPOCHS = 43                # ceil(8000/188) -- trains just past the budget; R1 = step 8000
BATCH_SIZE = 256           # [RB] matched across all four methods (RLSbench)
LR = 1e-3                  # [RB] tuned ONCE on source, shared by every method
WEIGHT_DECAY = 1e-4
POS_FRACTION = 0.5         # [RB] balanced 1:1 source (PROJECT.md 4.1)
SEED = 42
HEAD_TYPE = "linear"
ALL_FOLDS = [0, 1, 2, 3, 4]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--out-dir", default=os.path.join(root, "runs", "bakeoff_baseline_A"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the frozen config and exit without training")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    cache_dir = os.path.abspath(args.cache_dir)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    curve_csv = os.path.join(out_dir, "curve.csv")
    r1_ckpt = os.path.join(ckpt_dir, f"step_{BUDGET_STEPS}.pt")

    cfg = {"budget_steps": BUDGET_STEPS, "epochs_trained": EPOCHS,
           "steps_per_epoch": 188, "curve_every": CURVE_EVERY,
           "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
           "pos_fraction": POS_FRACTION, "seed": SEED, "head_type": HEAD_TYPE,
           "selection_rule": "R1_last_checkpoint_step8000",
           "curve_folds": ALL_FOLDS, "target_fpr": args.target_fpr,
           "method": "baseline_A_source_only", "domain_adaptation": None}

    print("=" * 78)
    print("MATCHED 5-FOLD BAKE-OFF -- run 1/4: BASELINE A (source-only) @ 8000 steps")
    print("=" * 78)
    print(f"  cache   : {cache_dir}")
    print(f"  out     : {out_dir}")
    for k, v in cfg.items():
        print(f"  {k:<18}: {v}")
    print(f"  R1 ckpt : {r1_ckpt}")
    print("=" * 78, flush=True)
    if args.dry_run:
        return 0

    cache = ev.FeatureCache(cache_dir)
    for need in ("source_train", "val", "test_negatives_shared"):
        if not cache.has(need):
            raise SystemExit(f"[bakeoff] ERROR: missing cache split {need}.npz in {cache_dir}")
    if cache.folds() != ALL_FOLDS:
        raise SystemExit(f"[bakeoff] ERROR: expected folds {ALL_FOLDS}, cache has {cache.folds()}")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(curve_csv):
        raise SystemExit(f"[bakeoff] ERROR: {curve_csv} already exists -- appending would "
                         f"corrupt the curve. Move/delete the old run dir first.")

    # ---- TRAIN (existing runner; reads ONLY source_train + val, guarded) ----
    train_source_only(
        cache_dir, ckpt_dir, head_type=HEAD_TYPE, epochs=(EPOCHS,),
        batch_size=BATCH_SIZE, pos_fraction=POS_FRACTION, lr=LR,
        weight_decay=WEIGHT_DECAY, seed=SEED, device="cpu",
        eval_every_steps=CURVE_EVERY, curve_csv=curve_csv, curve_folds=ALL_FOLDS,
        eval_target_fpr=args.target_fpr, eval_device="cpu", verbose=True)

    if not os.path.isfile(r1_ckpt):
        raise SystemExit(f"[bakeoff] ERROR: R1 checkpoint {r1_ckpt} was not written "
                         f"(is {BUDGET_STEPS} a multiple of --curve-every {CURVE_EVERY}?)")

    # ---- EVALUATE: exactly ONE candidate => R1 enforced structurally ----
    print(f"\n[eval] R1 = single candidate {os.path.basename(r1_ckpt)} "
          f"(source-val selection is a no-op with one candidate)", flush=True)
    results = ev.evaluate([r1_ckpt], cache_dir, device="cpu", target_fpr=args.target_fpr,
                          B=args.bootstrap, seed=args.eval_seed, out_dir=out_dir)

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"\n[out] run_config.json + results.json + curve.csv + scores/ -> {out_dir}")
    print(f"[next] python scripts/report_bakeoff.py {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
