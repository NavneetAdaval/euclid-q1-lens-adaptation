"""
run_bakeoff_freematch.py
========================
MATCHED 5-FOLD BAKE-OFF, run 4 of 4: FreeMatch at the LOCKED 8000-step budget
(PROJECT.md 7.4; matched protocol established in spec_bakeoff_2_coral.md §1).

Thin DRIVER over the existing FreeMatch math -- ``run_freematch.train_freematch_fold``
(FreeMatch SAT + SAF loss, mu cosine warmup, confirmation-bias detector, frozen encoder in
the loop) and ``run_coral.evaluate_per_fold`` (per-fold-model scoring). Nothing about the
FreeMatch training is re-implemented; the driver wires the pre-registered matched-protocol
settings, turns on the per-step curve+agreement+step-checkpoint hook, and enforces R1.
Same image-space structure as run_bakeoff_adamatch.py (encoder built once, per fold).

  ****  THE FROZEN VARIANT: local_threshold_mode = "standard"  (NOT "floor")  ****
  FreeMatch has one make-or-break knob (PROJECT.md 7.4): the local-threshold mode.
    * "standard" -- the paper's tau_t(lens) = (pbar(lens)/max_c pbar(c)) * tau_t. Its lower
                    lens-class bar (with SAF holding fairness) is what drives adaptation.
                    Fold-0 SCREEN: TPR@1e-3 ~= 0.357  -- ~2x baseline (the winning variant).
    * "floor"    -- clamps tau_t(lens) >= tau_t, suppressing weak lens pseudo-labels.
                    Fold-0 SCREEN: TPR@1e-3 ~= 0.174  -- UNDERPERFORMS baseline.
  This matched 5-fold run is FROZEN to "standard" -- the variant that cleared baseline in
  the screen. The value is a hard constant here (FREEMATCH_LOCAL_MODE) and is recorded in
  run_config.json ("local_threshold_mode": "standard") so the reported result states exactly
  which variant produced it. Do NOT run "floor" through this driver; it is a documented
  negative, not the matched contender.

MATCHED PROTOCOL (frozen; identical to CORAL/AdaMatch/Baseline A where applicable)
---------------------------------------------------------------------------------
  * 8000-step budget, batch 256 (matched -- RLSbench), lr 1e-3, weight-decay 1e-4,
    pos_fraction 0.5, seed 42, linear head. 48000 source / 256 = 187 steps/epoch, so 43
    epochs (8041 steps); R1 = step_8000.
  * FreeMatch published-default knobs (NOT tuned on target): mu_max 1.0, saf_weight 0.01
    (USB default), sat_ema_momentum 0.999, consistency_warmup_frac 0.1, balanced fairness
    target [0.5,0.5], pos_class 1, AND local_threshold_mode "standard" (frozen; above).
  * NO early stopping (confirmation-bias detector is a LOGGED diagnostic, never a stopper);
    the collapse gate is post-hoc.
  * Agreement column every 250 steps: Spearman(this ckpt on target_fold{k}, Baseline A
    step_8000 on target_fold{k}); read-only, never in the loss.

COMPUTE NOTE: image-space (~3 frozen-encoder forwards/step), so run per fold on a cloud GPU
(the Kaggle recipe in RUN3_ADAMATCH_COLAB.md-style flow -- swap only the driver name +
out-dir). Fold-0 first, then stop for review.

Run:
    python scripts/run_bakeoff_freematch.py --folds 0 --device cuda --fp16 --num-workers 4
    python scripts/run_bakeoff_freematch.py --dry-run
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

import evaluate as ev                                     # noqa: E402
import run_coral as rc                                    # noqa: E402  (per-fold evaluator)
from metrics import DEFAULT_TARGET_FPR                    # noqa: E402
from run_freematch import train_freematch_fold, _suite_from_scores  # noqa: E402  (existing math)
from run_bakeoff_coral import compute_agreement_refs      # noqa: E402  (reuse, don't duplicate)

# ---- PRE-REGISTERED matched settings (frozen; identical to CORAL/AdaMatch where applicable) ----
BUDGET_STEPS = 8000
CURVE_EVERY = 250
EPOCHS = 43                # ceil(8000/187); 48000 source / batch 256 = 187 steps/epoch
BATCH_SIZE = 256           # matched across all four methods (RLSbench)
LR = 1e-3
WEIGHT_DECAY = 1e-4
POS_FRACTION = 0.5
SEED = 42
HEAD_TYPE = "linear"
ALL_FOLDS = [0, 1, 2, 3, 4]

# ---- FreeMatch published-default knobs (NOT tuned on target) ----
MU_MAX = 1.0
SAF_WEIGHT = 0.01          # USB default
SAT_EMA_MOMENTUM = 0.999
CONSISTENCY_WARMUP_FRAC = 0.1
FAIRNESS_TARGET = None     # None -> balanced [0.5, 0.5]
POS_CLASS = 1
# THE FROZEN VARIANT (see module docstring): the screen-winning "standard" mode, NOT "floor".
FREEMATCH_LOCAL_MODE = "standard"

# Matched-budget Baseline A reference for the agreement column (never the ep100 default).
DEFAULT_REF = os.path.join(_HERE, "..", "runs", "bakeoff_baseline_A",
                           "checkpoints", "step_8000.pt")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap.add_argument("--data-root", default=os.path.join(root, "data"))
    ap.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--out-dir", default=os.path.join(root, "runs", "bakeoff_freematch"))
    ap.add_argument("--ref", default=os.path.normpath(DEFAULT_REF),
                    help="Baseline A matched-budget reference for the agreement column")
    ap.add_argument("--folds", default="0,1,2,3,4", help="comma-separated fold ids")
    ap.add_argument("--device", default=None, help="cuda if available else cpu")
    ap.add_argument("--eval-device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--fp16", action="store_true", help="autocast the encoder forwards")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    cache_dir = os.path.abspath(args.cache_dir)
    ref_ckpt = os.path.abspath(args.ref)
    splits_path = os.path.abspath(args.splits)
    data_root = os.path.abspath(args.data_root)
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {"budget_steps": BUDGET_STEPS, "epochs_trained": EPOCHS, "steps_per_epoch": 187,
           "curve_every": CURVE_EVERY, "batch_size": BATCH_SIZE, "lr": LR,
           "weight_decay": WEIGHT_DECAY, "pos_fraction": POS_FRACTION, "seed": SEED,
           "head_type": HEAD_TYPE, "selection_rule": "R1_last_checkpoint_step8000",
           "curve_folds": ALL_FOLDS, "target_fpr": args.target_fpr, "method": "freematch",
           "domain_adaptation": "freematch",
           # THE FROZEN VARIANT -- stated explicitly so the result names its own mode.
           "local_threshold_mode": FREEMATCH_LOCAL_MODE,
           "freematch_variant": f"FreeMatch-{FREEMATCH_LOCAL_MODE} (screen-winning; floor is a documented negative)",
           "mu_max": MU_MAX, "saf_weight": SAF_WEIGHT, "sat_ema_momentum": SAT_EMA_MOMENTUM,
           "consistency_warmup_frac": CONSISTENCY_WARMUP_FRAC, "fairness_target": FAIRNESS_TARGET,
           "pos_class": POS_CLASS, "device": device, "fp16": bool(args.fp16),
           "num_workers": args.num_workers, "agreement_ref": ref_ckpt,
           "folds_this_invocation": folds}

    print("=" * 78)
    print("MATCHED 5-FOLD BAKE-OFF -- run 4/4: FREEMATCH @ 8000 steps (image-space, per-fold)")
    print(f"           VARIANT = FreeMatch-{FREEMATCH_LOCAL_MODE.upper()}  (frozen; the screen-winning mode)")
    print("=" * 78)
    print(f"  data    : {data_root}\n  splits  : {splits_path}\n  cache   : {cache_dir}")
    print(f"  out     : {out_dir}\n  ref     : {ref_ckpt}")
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
            raise SystemExit(f"[bakeoff-freematch] ERROR: missing cache split {n}.npz in {cache_dir}")
    if not os.path.isfile(splits_path):
        raise SystemExit(f"[bakeoff-freematch] ERROR: splits not found: {splits_path}")
    if not os.path.isfile(ref_ckpt):
        raise SystemExit(f"[bakeoff-freematch] ERROR: agreement reference not found: {ref_ckpt}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- agreement reference (Baseline A step_8000 on each fold's target pool) ----
    print(f"[ref] scoring Baseline A {os.path.basename(ref_ckpt)} on target pools {folds} ...",
          flush=True)
    agreement_refs = compute_agreement_refs(cache_dir, ref_ckpt, folds, device=args.eval_device)

    # ---- frozen encoder built ONCE, shared across folds (image-space training) ----
    from zoobot_encoder import ZoobotEncoder
    print("[load] building the frozen Zoobot encoder (once) ...", flush=True)
    encoder = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=False)

    # ---- train each fold (own dir), R1 = step_8000 ----
    fold_ckpts = {}
    for k in folds:
        fold_dir = os.path.join(out_dir, f"fold{k}")
        ckpt_dir = os.path.join(fold_dir, "checkpoints")
        curve_csv = os.path.join(fold_dir, "curve.csv")
        r1 = os.path.join(ckpt_dir, f"step_{BUDGET_STEPS}.pt")
        if os.path.exists(curve_csv):
            raise SystemExit(f"[bakeoff-freematch] ERROR: {curve_csv} already exists -- "
                             f"appending would corrupt the curve. Move/delete {fold_dir} first.")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\n[train fold {k}] -> {fold_dir}  (image-space; encoder in loop; "
              f"mode={FREEMATCH_LOCAL_MODE}, mu_max={MU_MAX}, fp16={args.fp16})", flush=True)

        # existing FreeMatch math; only matched knobs + per-step hook are wired.
        train_freematch_fold(
            splits_path, data_root, cache_dir, k, ckpt_dir, encoder=encoder,
            head_type=HEAD_TYPE, epochs=(EPOCHS,), batch_size=BATCH_SIZE,
            pos_fraction=POS_FRACTION, lr=LR, weight_decay=WEIGHT_DECAY,
            mu_max=MU_MAX, saf_weight=SAF_WEIGHT, sat_ema_momentum=SAT_EMA_MOMENTUM,
            consistency_warmup_frac=CONSISTENCY_WARMUP_FRAC,
            local_threshold_mode=FREEMATCH_LOCAL_MODE, pos_class=POS_CLASS,
            fairness_target=FAIRNESS_TARGET, seed=SEED, device=device,
            num_workers=args.num_workers, fp16=args.fp16,
            eval_every_steps=CURVE_EVERY, curve_csv=curve_csv, curve_folds=ALL_FOLDS,
            agreement_refs={k: agreement_refs[k]}, eval_target_fpr=args.target_fpr,
            eval_device=args.eval_device, verbose=True)

        if not os.path.isfile(r1):
            raise SystemExit(f"[bakeoff-freematch] ERROR: R1 checkpoint {r1} was not written.")
        fold_ckpts[k] = [r1]

    # ---- evaluate: exactly ONE candidate per fold => R1 enforced structurally ----
    print(f"\n[eval] R1 = single candidate step_{BUDGET_STEPS}.pt per fold", flush=True)
    results = rc.evaluate_per_fold(
        fold_ckpts, cache_dir, device=args.eval_device, target_fpr=args.target_fpr,
        B=args.bootstrap, seed=args.eval_seed, out_dir=out_dir, verbose=True,
        method="freematch", score_prefix="freematch")
    results["metric_suite"] = _suite_from_scores(out_dir, folds, target_fpr=args.target_fpr)
    results["local_threshold_mode"] = FREEMATCH_LOCAL_MODE      # travels with the numbers

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[out] run_config.json + results.json + fold*/curve.csv + scores/ -> {out_dir}")
    print(f"[note] variant = FreeMatch-{FREEMATCH_LOCAL_MODE} (frozen)")
    print(f"[next] python scripts/report_bakeoff.py {out_dir}")
    for k in folds:
        print(f"[next] python scripts/collapse_gate.py --run {os.path.join(out_dir, f'fold{k}')} "
              f"--ref {ref_ckpt} --split target_fold{k} --tpr-col fold{k}_tpr_1e-3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
