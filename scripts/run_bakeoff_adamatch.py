"""
run_bakeoff_adamatch.py
=======================
MATCHED 5-FOLD BAKE-OFF, run 3 of 4: AdaMatch at the LOCKED 8000-step budget
(PROJECT.md 7.4; matched protocol established in spec_bakeoff_2_coral.md §1).

Thin DRIVER over the existing AdaMatch math -- ``run_adamatch.train_adamatch_fold``
(AdaMatch consistency loss, mu cosine warmup, confirmation-bias detector, encoder in the
loop) and ``run_coral.evaluate_per_fold`` (per-fold-model scoring). Nothing about the
AdaMatch training is re-implemented; the driver wires the pre-registered matched-protocol
settings, turns on the per-step curve+agreement+step-checkpoint hook, and enforces R1.

WHY AdaMatch differs from CORAL (image-space vs feature-space)
-------------------------------------------------------------
AdaMatch is a consistency method: it needs weak+strong augmented views PRE-encoder, so it
trains through the frozen Zoobot encoder IN THE LOOP (encoder under no_grad; only the head
trains) reading raw images -- it cannot use the feature cache for TRAINING. But EVALUATION
is the deterministic aug="none" path = exactly the cached 640-d features, and the saved
head/standardizer is the SAME format as CORAL. So the curve hook, the agreement column,
collapse_gate and report_bakeoff are all identical to CORAL (feature-space); only training
is image-space. It is a PER-FOLD-MODEL method (target rotates per fold), so pooled = n/a
(report_bakeoff auto-detects this from the differing per-fold negatives).

MATCHED PROTOCOL (frozen; identical to CORAL/Baseline A where applicable)
------------------------------------------------------------------------
  * 8000-step budget, batch 256 (matched -- RLSbench batch-size confound), lr 1e-3,
    weight-decay 1e-4, pos_fraction 0.5, seed 42, linear head. 48000 source / 256 = 187
    steps/epoch, so 43 epochs (8041 steps); R1 = step_8000.
  * AdaMatch-specific published-default knobs (NOT tuned on target): mu_max 1.0, tau 0.9,
    ema_momentum 0.9, consistency_warmup_frac 0.1, align_reference = source-EMA (None).
  * NO early stopping (the confirmation-bias detector is a LOGGED diagnostic, never a
    stopper); the collapse gate is post-hoc.
  * Agreement column every 250 steps: Spearman(this ckpt on target_fold{k}, Baseline A
    step_8000 on target_fold{k}); read-only, never in the loss.

COMPUTE NOTE: image-space, so ~4-7 s/step on a laptop GPU => ~10-15 h/fold. Use --folds to
run one fold at a time (fold-0 smoke first). GPU + --fp16 assumed.

Run:
    python scripts/run_bakeoff_adamatch.py --folds 0 --device cuda --fp16   # fold-0 smoke
    python scripts/run_bakeoff_adamatch.py --folds 1,2,3,4 --device cuda --fp16
    python scripts/run_bakeoff_adamatch.py --dry-run
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
from run_adamatch import train_adamatch_fold, _suite_from_scores  # noqa: E402  (existing math)
from run_bakeoff_coral import compute_agreement_refs      # noqa: E402  (reuse, don't duplicate)

# ---- PRE-REGISTERED matched settings (frozen; identical to CORAL where applicable) ----
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

# ---- AdaMatch published-default knobs (NOT tuned on target) ----
MU_MAX = 1.0
TAU = 0.9
EMA_MOMENTUM = 0.9
CONSISTENCY_WARMUP_FRAC = 0.1
ALIGN_REFERENCE = None      # None -> source-EMA distribution-alignment reference

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
    ap.add_argument("--out-dir", default=os.path.join(root, "runs", "bakeoff_adamatch"))
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
           "curve_folds": ALL_FOLDS, "target_fpr": args.target_fpr, "method": "adamatch",
           "domain_adaptation": "adamatch", "mu_max": MU_MAX, "tau": TAU,
           "ema_momentum": EMA_MOMENTUM, "consistency_warmup_frac": CONSISTENCY_WARMUP_FRAC,
           "align_reference": ALIGN_REFERENCE, "device": device, "fp16": bool(args.fp16),
           "num_workers": args.num_workers, "agreement_ref": ref_ckpt,
           "folds_this_invocation": folds}

    print("=" * 78)
    print("MATCHED 5-FOLD BAKE-OFF -- run 3/4: ADAMATCH @ 8000 steps (image-space, per-fold)")
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
            raise SystemExit(f"[bakeoff-adamatch] ERROR: missing cache split {n}.npz in {cache_dir}")
    for p in (splits_path,):
        if not os.path.isfile(p):
            raise SystemExit(f"[bakeoff-adamatch] ERROR: splits not found: {p}")
    if not os.path.isfile(ref_ckpt):
        raise SystemExit(f"[bakeoff-adamatch] ERROR: agreement reference not found: {ref_ckpt}")
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
            raise SystemExit(f"[bakeoff-adamatch] ERROR: {curve_csv} already exists -- "
                             f"appending would corrupt the curve. Move/delete {fold_dir} first.")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\n[train fold {k}] -> {fold_dir}  (image-space; encoder in loop; "
              f"mu_max={MU_MAX}, tau={TAU}, fp16={args.fp16})", flush=True)

        # existing AdaMatch math; only matched knobs + per-step hook are wired.
        train_adamatch_fold(
            splits_path, data_root, cache_dir, k, ckpt_dir, encoder=encoder,
            head_type=HEAD_TYPE, epochs=(EPOCHS,), batch_size=BATCH_SIZE,
            pos_fraction=POS_FRACTION, lr=LR, weight_decay=WEIGHT_DECAY,
            mu_max=MU_MAX, tau=TAU, ema_momentum=EMA_MOMENTUM,
            consistency_warmup_frac=CONSISTENCY_WARMUP_FRAC, align_reference=ALIGN_REFERENCE,
            seed=SEED, device=device, num_workers=args.num_workers, fp16=args.fp16,
            eval_every_steps=CURVE_EVERY, curve_csv=curve_csv, curve_folds=ALL_FOLDS,
            agreement_refs={k: agreement_refs[k]}, eval_target_fpr=args.target_fpr,
            eval_device=args.eval_device, verbose=True)

        if not os.path.isfile(r1):
            raise SystemExit(f"[bakeoff-adamatch] ERROR: R1 checkpoint {r1} was not written.")
        fold_ckpts[k] = [r1]

    # ---- evaluate: exactly ONE candidate per fold => R1 enforced structurally ----
    print(f"\n[eval] R1 = single candidate step_{BUDGET_STEPS}.pt per fold", flush=True)
    results = rc.evaluate_per_fold(
        fold_ckpts, cache_dir, device=args.eval_device, target_fpr=args.target_fpr,
        B=args.bootstrap, seed=args.eval_seed, out_dir=out_dir, verbose=True,
        method="adamatch", score_prefix="adamatch")
    results["metric_suite"] = _suite_from_scores(out_dir, folds, target_fpr=args.target_fpr)

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[out] run_config.json + results.json + fold*/curve.csv + scores/ -> {out_dir}")
    print(f"[next] python scripts/report_bakeoff.py {out_dir}")
    for k in folds:
        print(f"[next] python scripts/collapse_gate.py --run {os.path.join(out_dir, f'fold{k}')} "
              f"--ref {ref_ckpt} --split target_fold{k} --tpr-col fold{k}_tpr_1e-3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
