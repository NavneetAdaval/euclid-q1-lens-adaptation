"""
run_bakeoff_coral.py
====================
MATCHED 5-FOLD BAKE-OFF, run 2 of 4: Deep CORAL at the LOCKED 8000-step budget
(SPEC spec_bakeoff_2_coral.md; PROJECT.md 7.4).

This is a thin DRIVER, exactly like run_bakeoff_baseline.py wrapped train_source_only.
It reuses the existing CORAL math verbatim -- ``run_coral.train_coral_fold`` (loss loop,
``da_losses.CoralLoss``, the warmup-cosine gamma schedule) and ``run_coral.evaluate_per_fold``
(per-fold-model scoring). Nothing about the CORAL training is re-implemented here; the
driver only (a) wires the pre-registered matched-protocol settings so they cannot drift,
(b) turns on the per-step curve+agreement+step-checkpoint hook, and (c) enforces R1.

WHY A PER-FOLD DRIVER (CORAL differs from Baseline A)
----------------------------------------------------
Baseline A is source-only -> ONE model, five evaluations. CORAL USES the target, and the
target rotates per fold (PROJECT.md 4.3), so there are FIVE distinct models, one per fold,
each trained against its own ``target_fold{k}`` pool and scored on its own held-out test.
Each fold trains into ``<out>/fold{k}/`` (own checkpoints/ + curve.csv); the per-fold R1
test score vectors all land in the shared ``<out>/scores/`` so report_bakeoff.py reads them.

R1 ENFORCED STRUCTURALLY (same trick as run 1)
----------------------------------------------
The pre-registered rule is R1 = the LAST checkpoint (step 8000). We hand
``evaluate_per_fold`` exactly ONE candidate per fold -- ``fold{k}/checkpoints/step_8000.pt``
-- so source-val selection is a no-op and SOURCE-RISK == ORACLE == R1 by construction.

MATCHED PROTOCOL (frozen; SPEC §1) -- identical to Baseline A where applicable
-----------------------------------------------------------------------------
  * 8000-step budget, 43 epochs (ceil(8000/188)); step_8000.pt is R1.
  * batch 256, lr 1e-3, weight-decay 1e-4, pos_fraction 0.5, seed 42, linear head.
  * CORAL-specific: warmup_cosine gamma schedule ON (the blow-up fix), PEAK gamma 1.0
    (fixed, not tuned), shrinkage 0.1, include-mean + target-EMA (published defaults).
  * NO early stopping: stop_on_blowup=False so the full 8000-step curve exists for the
    POST-HOC collapse gate (SPEC §5 -- the gate never halts training).
  * Agreement column logged every 250 steps: Spearman(this ckpt on target_fold{k},
    Baseline A step_8000 on target_fold{k}); read-only, never in the loss (SPEC §4).

Run:
    python scripts/run_bakeoff_coral.py --folds 0                 # fold-0 smoke (STOP + review)
    python scripts/run_bakeoff_coral.py --folds 1,2,3,4           # the rest, after review
    python scripts/run_bakeoff_coral.py --dry-run                 # print frozen config, exit
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
from metrics import DEFAULT_TARGET_FPR                   # noqa: E402
from run_coral import train_coral_fold, evaluate_per_fold  # noqa: E402  (existing CORAL math)

# ---- PRE-REGISTERED SETTINGS (SPEC §1) -- frozen before the run, not tuned on results ----
BUDGET_STEPS = 8000        # [RB] locked step budget (PROJECT.md 7.4)
CURVE_EVERY = 250          # [EC] ~32 curve points/fold
EPOCHS = 43                # ceil(8000/188) -- trains just past the budget; R1 = step 8000
BATCH_SIZE = 256           # [RB] matched across all four methods (RLSbench)
LR = 1e-3                  # [RB] tuned ONCE on source, shared by every method
WEIGHT_DECAY = 1e-4
POS_FRACTION = 0.5         # [RB] balanced 1:1 source (PROJECT.md 4.1)
SEED = 42
HEAD_TYPE = "linear"
ALL_FOLDS = [0, 1, 2, 3, 4]

# ---- CORAL-specific pre-registered settings (SPEC §1) ----
CORAL_PEAK_GAMMA = 1.0            # FIXED peak, not tuned on target (RLSbench)
CORAL_SCHEDULE = "warmup_cosine"  # the blow-up fix -- ON (SPEC §1 / constraint)
CORAL_WARMUP_FRAC = 0.1
CORAL_MIN_FRAC = 0.1
CORAL_SHRINKAGE = 0.1
CORAL_INCLUDE_MEAN = True
CORAL_TARGET_EMA = True
CORAL_EMA_MOMENTUM = 0.9
STOP_ON_BLOWUP = False           # SPEC §5: gate is post-hoc; training is never halted

# The matched-budget Baseline A reference for the agreement column (SPEC §0 / Flag B).
# NEVER the old runs/baseline_A/...ep100.pt default that lives in collapse_gate.py.
DEFAULT_REF = os.path.join(_HERE, "..", "runs", "bakeoff_baseline_A",
                           "checkpoints", "step_8000.pt")


def compute_agreement_refs(cache_dir, ref_ckpt, folds, device="cpu"):
    """Score Baseline A's step_8000 ONCE on each fold's UNLABELED target pool.

    Returns {fold: ref_scores}. These fixed reference vectors are the agreement column's
    anchor (SPEC §4); computing them once here keeps the per-step hook cheap and reuses
    evaluate.score_features (no new scoring path)."""
    head, std, _ = ev.load_checkpoint(ref_ckpt, device)
    cache = ev.FeatureCache(cache_dir)
    refs = {}
    for k in folds:
        feats = cache.load(f"target_fold{k}")["features"]
        refs[k] = ev.score_features(head, std, feats, device)
    return refs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--out-dir", default=os.path.join(root, "runs", "bakeoff_coral"))
    ap.add_argument("--ref", default=os.path.normpath(DEFAULT_REF),
                    help="Baseline A matched-budget reference for the agreement column")
    ap.add_argument("--folds", default="0,1,2,3,4", help="comma-separated fold ids")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the frozen config and exit without training")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    cache_dir = os.path.abspath(args.cache_dir)
    ref_ckpt = os.path.abspath(args.ref)
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]

    cfg = {"budget_steps": BUDGET_STEPS, "epochs_trained": EPOCHS,
           "steps_per_epoch": 188, "curve_every": CURVE_EVERY,
           "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
           "pos_fraction": POS_FRACTION, "seed": SEED, "head_type": HEAD_TYPE,
           "selection_rule": "R1_last_checkpoint_step8000",
           "curve_folds": ALL_FOLDS, "target_fpr": args.target_fpr,
           "method": "deep_coral", "domain_adaptation": "coral",
           "coral_peak_gamma": CORAL_PEAK_GAMMA, "coral_weight_schedule": CORAL_SCHEDULE,
           "coral_warmup_frac": CORAL_WARMUP_FRAC, "coral_min_frac": CORAL_MIN_FRAC,
           "coral_shrinkage": CORAL_SHRINKAGE, "coral_include_mean": CORAL_INCLUDE_MEAN,
           "coral_target_ema": CORAL_TARGET_EMA, "coral_ema_momentum": CORAL_EMA_MOMENTUM,
           "stop_on_blowup": STOP_ON_BLOWUP, "agreement_ref": ref_ckpt,
           "folds_this_invocation": folds}

    print("=" * 78)
    print("MATCHED 5-FOLD BAKE-OFF -- run 2/4: DEEP CORAL @ 8000 steps (per-fold models)")
    print("=" * 78)
    print(f"  cache   : {cache_dir}")
    print(f"  out     : {out_dir}")
    print(f"  ref     : {ref_ckpt}")
    for k, v in cfg.items():
        print(f"  {k:<20}: {v}")
    print("=" * 78, flush=True)
    if args.dry_run:
        return 0

    # ---- preconditions ----
    cache = ev.FeatureCache(cache_dir)
    need = (["source_train", "val", "test_negatives_shared"]
            + [f"target_fold{k}" for k in folds] + [f"test_pos_fold{k}" for k in folds])
    for n in need:
        if not cache.has(n):
            raise SystemExit(f"[bakeoff-coral] ERROR: missing cache split {n}.npz in {cache_dir}")
    if not os.path.isfile(ref_ckpt):
        raise SystemExit(f"[bakeoff-coral] ERROR: agreement reference not found: {ref_ckpt} "
                         f"(run 1 / Baseline A must be trained first)")
    os.makedirs(out_dir, exist_ok=True)

    # ---- agreement reference (Baseline A step_8000 on each fold's target pool) ----
    print(f"[ref] scoring Baseline A {os.path.basename(ref_ckpt)} on target pools "
          f"{folds} for the agreement column ...", flush=True)
    agreement_refs = compute_agreement_refs(cache_dir, ref_ckpt, folds)

    # ---- train each fold (own dir), R1 = step_8000 ----
    fold_ckpts = {}
    for k in folds:
        fold_dir = os.path.join(out_dir, f"fold{k}")
        ckpt_dir = os.path.join(fold_dir, "checkpoints")
        curve_csv = os.path.join(fold_dir, "curve.csv")
        r1 = os.path.join(ckpt_dir, f"step_{BUDGET_STEPS}.pt")
        if os.path.exists(curve_csv):
            raise SystemExit(f"[bakeoff-coral] ERROR: {curve_csv} already exists -- appending "
                             f"would corrupt the curve. Move/delete {fold_dir} first.")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\n[train fold {k}] -> {fold_dir}   (warmup_cosine, peak gamma={CORAL_PEAK_GAMMA}, "
              f"stop_on_blowup={STOP_ON_BLOWUP})", flush=True)

        # existing CORAL math; only the matched-protocol knobs + per-step hook are wired.
        train_coral_fold(
            cache, k, ckpt_dir, head_type=HEAD_TYPE, epochs=(EPOCHS,),
            batch_size=BATCH_SIZE, pos_fraction=POS_FRACTION, lr=LR,
            weight_decay=WEIGHT_DECAY, coral_weight=CORAL_PEAK_GAMMA,
            coral_include_mean=CORAL_INCLUDE_MEAN, coral_shrinkage=CORAL_SHRINKAGE,
            coral_target_ema=CORAL_TARGET_EMA, coral_ema_momentum=CORAL_EMA_MOMENTUM,
            coral_weight_schedule=CORAL_SCHEDULE, coral_warmup_frac=CORAL_WARMUP_FRAC,
            coral_min_frac=CORAL_MIN_FRAC, stop_on_blowup=STOP_ON_BLOWUP,
            eval_every_steps=CURVE_EVERY, curve_csv=curve_csv, curve_folds=ALL_FOLDS,
            agreement_refs={k: agreement_refs[k]}, eval_target_fpr=args.target_fpr,
            eval_device="cpu", seed=SEED, device="cpu", verbose=True)

        if not os.path.isfile(r1):
            raise SystemExit(f"[bakeoff-coral] ERROR: R1 checkpoint {r1} was not written "
                             f"(is {BUDGET_STEPS} a multiple of --curve-every {CURVE_EVERY} "
                             f"and <= total steps?)")
        fold_ckpts[k] = [r1]

    # ---- evaluate: exactly ONE candidate per fold => R1 enforced structurally ----
    print(f"\n[eval] R1 = single candidate step_{BUDGET_STEPS}.pt per fold "
          f"(source-val selection is a no-op with one candidate)", flush=True)
    results = evaluate_per_fold(
        fold_ckpts, cache_dir, device="cpu", target_fpr=args.target_fpr,
        B=args.bootstrap, seed=args.eval_seed, out_dir=out_dir, verbose=True,
        method="deep_coral", score_prefix="coral")

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"\n[out] run_config.json + results.json + fold*/curve.csv + scores/ -> {out_dir}")
    print(f"[next] python scripts/report_bakeoff.py {out_dir}")
    for k in folds:
        print(f"[next] python scripts/collapse_gate.py --run {os.path.join(out_dir, f'fold{k}')} "
              f"--ref {ref_ckpt} --split target_fold{k} --tpr-col fold{k}_tpr_1e-3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
