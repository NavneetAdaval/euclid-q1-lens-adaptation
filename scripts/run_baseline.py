"""
run_baseline.py
===============
Baseline A -- the SOURCE-ONLY runner for the label-free sim-to-real strong-lens
pipeline (Euclid Q1). This is the "no domain adaptation" reference point every UDA
method in the bake-off is measured against: train the head on the fixed labelled
SOURCE (sim lenses + negatives) with NO target information whatsoever, then report
it against the real test folds via evaluate.py.

WHAT "BASELINE A / source-only" MEANS (PROJECT.md 1, 7.4)
--------------------------------------------------------
  * NO domain adaptation. The per-fold TARGET pools are never opened here; no target
    batch is ever formed, so the DA-loss hook of train_skeleton is identically the
    inert zero (there is nothing for it to consume). We deliberately do NOT add any
    real DA term -- that is exactly what makes this the source-only control.
  * ONE training run, not five. We "train once on the fixed source"; the 5 folds
    differ ONLY at evaluation (each fold's held-out real-lens positives scored against
    the shared negative pool), which evaluate.py already handles. So this script trains
    a single model and hands it to evaluate.py to be scored across every fold.

READS CACHED FEATURES ONLY (PROJECT.md 7.1 / cache_features.py)
---------------------------------------------------------------
The Zoobot encoder is FROZEN, so cache_features.py already encoded every split to
disk once (the single GPU step). This runner therefore NEVER loads the encoder or a
raw image: it reads the pre-computed 640-d feature rows from ``<cache>/source_train.npz``
and ``<cache>/val.npz`` and trains a pure feature -> head classifier. Fast on CPU.

MODEL SELECTION IS SOURCE-VAL ONLY (label-free-legal; PROJECT.md 4.2)
--------------------------------------------------------------------
We save a checkpoint at a schedule of epochs. Those epoch-checkpoints are the
SOURCE-RISK *selection candidates*: evaluate.py picks the reported one by the best
SOURCE-val metric (val is labelled held-out SIM -- the only label-free-legal selector)
and reports THAT checkpoint against the real test folds (+ an ORACLE upper bound and
the alert-rate guardrail). The per-epoch ``val_auroc`` printed here is only a
diagnostic; the authoritative selection is done inside evaluate.py.

  ****  LEAKAGE CONTRACT (asserted + commented below)  ****
  The head is NEVER trained against, nor selected on, the real TEST lenses. Training
  reads ONLY source_train (+ val, which is source-domain SIM). The 5 test folds and
  the target pools are touched EXCLUSIVELY inside evaluate.evaluate(), for scoring.

Run (after cache_features.py has populated the cache):
    python scripts\\run_baseline.py --cache-dir data\\feature_cache --out-dir runs\\baseline_A
    python scripts\\run_baseline.py --head-type mlp --epochs 1,2,5,10,20,50   # linear/mlp diagnostic
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

# heads + the balanced sampler are REUSED verbatim (same head, same checkpoint
# format, same class-balanced batching as train_skeleton) -- no re-implementation.
from heads import build_head, FeatureStandardizer, EMBED_DIM, LINEAR, HEAD_TYPES
from samplers import ClassBalancedBatchSampler, POS_LABEL, NEG_LABEL
from metrics import auroc, DEFAULT_TARGET_FPR
import evaluate as ev
import curve_hook

# The cache files this runner is ALLOWED to read during training. Anything else --
# test_pos_fold*, test_negatives_shared, target_fold* -- is off-limits here and is
# opened only by evaluate.py. Enforced by _guarded_load() below.
TRAIN_ALLOWED = ("source_train", "val")


# ---------------------------------------------------------------------------
# reproducibility (mirrors train_skeleton.set_seed -- ONE fixed-seed run)
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# cache access (leakage-guarded)
# ---------------------------------------------------------------------------

def _guarded_load(cache: ev.FeatureCache, name: str) -> dict:
    """Load a cached split, but REFUSE to read anything but source_train / val.

    This is the runtime half of the leakage contract: a bug that tried to peek at a
    test/target split from inside training would raise here instead of silently
    leaking. (evaluate.py loads those splits itself, later, for scoring.)
    """
    assert name in TRAIN_ALLOWED, (
        f"run_baseline training may only read {TRAIN_ALLOWED}; refusing to load "
        f"'{name}' (the test folds / target pool are evaluate.py's job, not training's).")
    if not cache.has(name):
        raise FileNotFoundError(
            f"required cache split '{name}.npz' not found under {cache.dir}; "
            f"run cache_features.py first.")
    return cache.load(name)


# ---------------------------------------------------------------------------
# epoch schedule + checkpoint naming
# ---------------------------------------------------------------------------

def _cf(x, nd=4):
    """Format a curve cell (None-safe) for the per-step log line."""
    return "nan" if x is None else f"{x:.{nd}f}"


def parse_epochs(spec: str) -> list[int]:
    """Parse a comma-separated epoch schedule (e.g. '1,2,5,10,20,50,100')."""
    eps = sorted({int(x) for x in str(spec).split(",") if x.strip() != ""})
    if not eps or eps[0] < 1:
        raise ValueError(f"epoch schedule must be positive ints, got {spec!r}")
    return eps


def ckpt_name(head_type: str, epoch: int, width: int) -> str:
    """Name a checkpoint so evaluate._epoch_from_name() ('(?:ep|epoch)[_-]?(\\d+)')
    recovers the epoch -> e.g. baseline_linear_ep005.pt / baseline_mlp_ep050.pt."""
    return f"baseline_{head_type}_ep{epoch:0{width}d}.pt"


def save_checkpoint(path, head, standardizer, head_type, epoch, cfg_extra: dict):
    """Write a checkpoint in the EXACT train_skeleton.Trainer.save() format
    (head_type / head state_dict / standardizer state_dict / cfg with fold+epoch)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cfg = {"head_type": head_type, "fold": None, "epoch": int(epoch), **cfg_extra}
    # fold=None is deliberate: Baseline A is a SINGLE fold-agnostic model evaluated
    # across all folds (PROJECT.md 4.4) -- there is no per-fold training here.
    torch.save({
        "head_type": head_type,
        "head": head.state_dict(),
        "standardizer": standardizer.state_dict(),
        "cfg": cfg,
    }, path)


# ---------------------------------------------------------------------------
# the source-only training run (feature-space; ONE run, fixed seed)
# ---------------------------------------------------------------------------

def train_source_only(cache_dir, ckpt_dir, *, head_type=LINEAR,
                      epochs=(1, 2, 5, 10, 20, 50, 100), batch_size=256,
                      pos_fraction=0.5, lr=1e-3, weight_decay=1e-4,
                      steps_per_epoch=None, seed=42, device="cpu",
                      eval_every_steps=None, curve_csv=None, curve_folds=None,
                      eval_target_fpr=DEFAULT_TARGET_FPR, val_select_metric="auroc",
                      eval_device="cpu", verbose=True) -> list[str]:
    """Train the head on the cached SOURCE features and dump epoch-checkpoints.

    Returns the list of checkpoint paths (the SOURCE-RISK selection candidates).
    Reads ONLY source_train (+ val, a source-domain diagnostic). Never touches the
    test folds or the target pool -- those live exclusively in evaluate.py.
    """
    if head_type not in HEAD_TYPES:
        raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type!r}")
    set_seed(seed)
    dev = torch.device(device)
    cache = ev.FeatureCache(cache_dir)

    # ---- source_train: the ONLY thing the classifier learns from ----
    src = _guarded_load(cache, "source_train")
    feats = torch.from_numpy(np.asarray(src["features"], dtype=np.float32)).to(dev)
    labels = np.asarray(src["labels"], dtype=np.int64)
    if feats.shape[1] != EMBED_DIM:
        raise ValueError(f"cached source features are {feats.shape[1]}-d, expected {EMBED_DIM}")
    pos_idx = np.where(labels == POS_LABEL)[0]
    neg_idx = np.where(labels == NEG_LABEL)[0]
    y = torch.from_numpy((labels == POS_LABEL).astype(np.float32)).view(-1, 1).to(dev)
    if verbose:
        print(f"[source] {feats.shape[0]} rows  ({pos_idx.size} pos / {neg_idx.size} neg), "
              f"{feats.shape[1]}-d frozen features")

    # ---- FeatureStandardizer: fit ONCE on the source features, then freeze ----
    # These are already the frozen 640-d Zoobot features, so this is pure per-feature
    # standardization -- no encoder forward. Saved with each checkpoint so evaluate.py
    # applies the identical (x-mean)/std to the test features.
    standardizer = FeatureStandardizer(EMBED_DIM).to(dev)
    standardizer.fit(feats)
    x = standardizer(feats).detach()          # standardized source features (frozen std)

    # ---- head + optimizer (head params ONLY; there is no encoder here) ----
    head = build_head(head_type).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    # ---- class-balanced batches (REUSED sampler; balances on the real source labels) ----
    sampler = ClassBalancedBatchSampler(
        pos_idx, neg_idx, batch_size=batch_size, num_batches=steps_per_epoch,
        pos_fraction=pos_fraction, seed=seed)
    steps = len(sampler)

    # ---- optional val diagnostic (source-domain SIM; label-free-legal) ----
    val = None
    if cache.has("val"):
        v = _guarded_load(cache, "val")
        val = (torch.from_numpy(np.asarray(v["features"], dtype=np.float32)).to(dev),
               np.asarray(v["labels"], dtype=np.int64))

    # curve fold set: default (0,) = the legacy fold-0-only curve. The 5-fold bake-off
    # passes [0..4] so the per-fold flatness re-check (SPEC 4) has a column per fold.
    cfolds = tuple(curve_hook.DEFAULT_FOLDS if curve_folds is None
                   else (int(k) for k in curve_folds))

    schedule = sorted(set(int(e) for e in epochs))
    max_epoch = schedule[-1]
    width = max(2, len(str(max_epoch)))
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg_extra = {"encoder_mode": "frozen", "lr": lr, "weight_decay": weight_decay,
                 "batch_size": batch_size, "pos_fraction": pos_fraction, "seed": seed,
                 "steps_per_epoch": steps, "baseline": "A_source_only"}

    if verbose:
        print(f"[train] head={head_type}  epochs->{schedule}  {steps} steps/epoch  "
              f"batch={batch_size} (pos_frac~{sampler.actual_pos_fraction:.2f})  seed={seed}")

    checkpoints: list[str] = []
    gstep = 0                                  # global optimizer-step counter (curve hook)
    for epoch in range(1, max_epoch + 1):
        sampler.set_epoch(epoch)
        head.train()
        last = 0.0
        for batch in sampler:                 # each `batch` is a list of source row indices
            idx = torch.as_tensor(batch, dtype=torch.long, device=dev)
            opt.zero_grad(set_to_none=True)
            logits = head(x[idx])             # supervised BCE on the SOURCE only
            loss = bce(logits, y[idx])
            # DA-loss hook is identically 0 here: source-only means NO target batch
            # exists to align against (see module docstring). We add nothing.
            loss.backward()
            opt.step()
            last = float(loss.detach())
            gstep += 1
            # curve hook (behind --eval-every-steps; default runs never enter here):
            # save the trainable weights at this step + append one fold-0 curve.csv row.
            if eval_every_steps and curve_csv and (gstep % int(eval_every_steps) == 0):
                spath = os.path.join(ckpt_dir, f"step_{gstep}.pt")
                save_checkpoint(spath, head, standardizer, head_type, epoch, cfg_extra)
                row = curve_hook.append_curve_row(
                    curve_csv, gstep, spath, cache_dir, folds=cfolds,
                    target_fpr=eval_target_fpr,
                    val_select_metric=val_select_metric, device=eval_device)
                if verbose:
                    per_fold = "  ".join(
                        f"f{k}={_cf(row[f'fold{k}_tpr_1e-3'])}" for k in cfolds)
                    print(f"    [step {gstep:>6}] curve  "
                          f"val_auroc={_cf(row['source_risk_auroc'])}"
                          f" val_tpr={_cf(row['source_risk_tpr'])}"
                          f" | tpr@1e-3  {per_fold}", flush=True)

        if epoch in schedule:
            path = os.path.join(ckpt_dir, ckpt_name(head_type, epoch, width))
            save_checkpoint(path, head, standardizer, head_type, epoch, cfg_extra)
            checkpoints.append(path)
            if verbose:
                msg = f"  [ep {epoch:>{width}}] loss={last:.4f}  saved {os.path.basename(path)}"
                if val is not None:
                    head.eval()
                    with torch.no_grad():
                        vlog = head(standardizer(val[0])).view(-1).cpu().numpy()
                    msg += (f"  val_auroc={auroc(val[1], vlog):.4f}"
                            f" (diagnostic; evaluate.py does the real selection)")
                print(msg, flush=True)

    return checkpoints


# ---------------------------------------------------------------------------
# the full Baseline A: train-once  ->  evaluate across all folds
# ---------------------------------------------------------------------------

def run_baseline(cache_dir, out_dir, *, head_type=LINEAR,
                 epochs=(1, 2, 5, 10, 20, 50, 100), batch_size=256, pos_fraction=0.5,
                 lr=1e-3, weight_decay=1e-4, steps_per_epoch=None, seed=42,
                 train_device="cpu", eval_device="cpu", target_fpr=DEFAULT_TARGET_FPR,
                 bootstrap=1000, eval_seed=0, val_select_metric="auroc",
                 eval_every_steps=None, curve_folds=None, verbose=True) -> dict:
    """Baseline A end-to-end: train the source-only head once, then hand ALL epoch
    checkpoints to evaluate.evaluate() (SOURCE-RISK selection + ORACLE + per-fold /
    pooled TPR@FPR & AUROC with paired-bootstrap CIs + guardrail + raw score dump).

    ``eval_every_steps`` (default None) turns on the fold-0 learning-curve hook: every N
    optimizer steps a step-checkpoint is saved and one row is appended to out_dir/curve.csv.
    Leaves the end-of-run evaluate.py behavior below completely unchanged."""
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    curve_csv = os.path.join(out_dir, "curve.csv") if eval_every_steps else None
    checkpoints = train_source_only(
        cache_dir, ckpt_dir, head_type=head_type, epochs=epochs, batch_size=batch_size,
        pos_fraction=pos_fraction, lr=lr, weight_decay=weight_decay,
        steps_per_epoch=steps_per_epoch, seed=seed, device=train_device,
        eval_every_steps=eval_every_steps, curve_csv=curve_csv, curve_folds=curve_folds,
        eval_target_fpr=target_fpr,
        val_select_metric=val_select_metric, eval_device=eval_device, verbose=verbose)

    if verbose:
        print(f"\n[eval] handing {len(checkpoints)} epoch-checkpoint(s) to evaluate.py "
              f"(SOURCE-RISK selects among them on source-val; test folds touched only here)")
    # evaluate.py owns ALL contact with the test folds + target pools. It selects the
    # reported checkpoint by source-val (label-free), reports it vs the ORACLE upper
    # bound, and writes results.json + raw per-fold scores. THIS is Baseline A.
    results = ev.evaluate(
        checkpoints, cache_dir, device=eval_device, target_fpr=target_fpr,
        B=bootstrap, seed=eval_seed, out_dir=out_dir, val_select_metric=val_select_metric)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.normpath(os.path.join(here, "..", "data"))
    p = argparse.ArgumentParser(description="Baseline A -- source-only runner (feature-space).")
    p.add_argument("--cache-dir", default=os.path.join(default_root, "feature_cache"))
    p.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "runs", "baseline_A")))
    p.add_argument("--head-type", default=LINEAR, choices=HEAD_TYPES,
                   help="linear (headline) or mlp (linear/mlp diagnostic)")
    p.add_argument("--epochs", default="1,2,5,10,20,50,100",
                   help="comma-separated epoch schedule = SOURCE-RISK selection candidates")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pos-fraction", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="default: one nominal pass over the source pool")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-device", default=None, help="default: cuda if available else cpu")
    p.add_argument("--eval-device", default="cpu", help="evaluate.py scores cached features on CPU")
    p.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--val-select-metric", default="auroc", choices=("auroc", "tpr"))
    p.add_argument("--eval-every-steps", type=int, default=None,
                   help="if set, every N optimizer steps save checkpoints/step_N.pt and "
                        "append a fold-0 row to <out-dir>/curve.csv (learning-curve "
                        "diagnostic; default off -> normal run unchanged)")
    p.add_argument("--curve-folds", default=None,
                   help="comma-separated folds to log in curve.csv (default: 0 = the "
                        "legacy fold-0-only curve). Use '0,1,2,3,4' for the matched "
                        "5-fold bake-off's per-fold flatness check.")
    args = p.parse_args()

    if not os.path.isdir(args.cache_dir):
        print(f"SKIP: feature cache dir not found: {args.cache_dir} (run cache_features.py first)")
        return 3
    cache = ev.FeatureCache(args.cache_dir)
    if not cache.has("source_train"):
        print(f"SKIP: {os.path.join(args.cache_dir, 'source_train.npz')} missing "
              f"(run cache_features.py first)")
        return 3
    if not cache.has("test_negatives_shared") or not cache.folds():
        print(f"SKIP: test cache incomplete under {args.cache_dir} "
              f"(need test_negatives_shared + test_pos_fold*.npz)")
        return 3

    train_device = args.train_device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = parse_epochs(args.epochs)
    curve_folds = ([int(k) for k in args.curve_folds.split(",") if k.strip() != ""]
                   if args.curve_folds else None)
    print(f"Baseline A (source-only)  cache={args.cache_dir}  out={args.out_dir}")
    run_baseline(
        args.cache_dir, args.out_dir, head_type=args.head_type, epochs=epochs,
        batch_size=args.batch_size, pos_fraction=args.pos_fraction, lr=args.lr,
        weight_decay=args.weight_decay, steps_per_epoch=args.steps_per_epoch,
        seed=args.seed, train_device=train_device, eval_device=args.eval_device,
        target_fpr=args.target_fpr, bootstrap=args.bootstrap, eval_seed=args.eval_seed,
        val_select_metric=args.val_select_metric, eval_every_steps=args.eval_every_steps,
        curve_folds=curve_folds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
