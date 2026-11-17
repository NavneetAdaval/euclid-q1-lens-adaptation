"""
run_adamatch.py
===============
AdaMatch -- the SECOND real domain-adaptation method in the bake-off (after Deep CORAL).
Like run_coral it trains ONE MODEL PER FOLD (the target rotates per fold; PROJECT.md 4.3)
and evaluates each fold's model on that fold's held-out real-lens positives + the shared
negatives, aggregating across folds. It shares run_coral's head, standardizer, checkpoint
format, per-fold evaluator and metric suite, so the two DA methods are directly comparable.

WHAT IS DIFFERENT FROM CORAL (and WHY it can't use the feature cache for TRAINING)
----------------------------------------------------------------------------------
AdaMatch is a consistency method: it needs WEAK and STRONG augmented views of every image,
and augmentation happens PRE-encoder. The frozen features in the cache were computed with
a single deterministic (aug="none") view, so there is no weak/strong pair to build from
them. AdaMatch therefore trains through the IMAGE-SPACE ``train_skeleton.Trainer`` with the
frozen Zoobot encoder IN THE LOOP (encoder under no_grad, only the head trains), reading
raw images from ``LensDataset(n_views=2)`` (weak="weak", strong="strong") for BOTH domains.

EVALUATION still uses the cache. The trained head+standardizer operate on the frozen 640-d
features either way, and evaluation is the deterministic aug="none" path -- exactly the
cached features. So we hand the per-fold checkpoints to run_coral's per-fold evaluator
(reused verbatim, just re-labelled "adamatch"), which reads the cache and does the same
label-free SOURCE-RISK selection + per-fold/across-fold TPR@1e-3 & AUROC as CORAL. On top
of that we compute the compare_methods metric suite (TPR@1e-3, AUPRC, purity@50%-comp raw
+ 1:2200-corrected) on the dumped SOURCE-RISK scores, so AdaMatch reports the same family
of numbers CORAL/compare_methods do.

FIXED-BUDGET PROTOCOL (PROJECT.md 7.4)
--------------------------------------
mu_max (peak consistency weight, 1.0), tau (relative-confidence factor, 0.9) and the
distribution-alignment reference (the balanced SOURCE distribution) are all FIXED config
values, NOT tuned on the target. The mu cosine warmup + the minimum consistency-warmup
gate are fixed stability priors, not oracle knobs.

CONFIRMATION-BIAS DIAGNOSTIC (logged, never a selector)
-------------------------------------------------------
Rare positives make pseudo-labeling prone to collapse. Every step we log the pseudo-label
ACCEPTANCE RATE and the CLASS BALANCE of accepted pseudo-labels, and run a
``ConfirmationBiasDetector`` (the BlowupDetector pattern extended to AdaMatch): it flags
either acceptance ~ 0 (nothing confident) or accepted pseudo-labels nearly all one class
(everything called lens / non-lens). It is LOGGED per fold, never used to select a model.

  ****  LEAKAGE CONTRACT  ****
  Fold k trains on source_train + val (source-domain SIM) + target_fold{k} ONLY. Fold k's
  target pool by construction EXCLUDES fold k's own lenses (PROJECT.md 4.3), and the test
  folds are touched exclusively by the evaluator. No model is trained on a lens it is tested
  on. (The evaluator reuses the SAME leakage-guarded per-fold path as CORAL.)

Run (needs the real splits + the frozen encoder weights; eval needs the feature cache):
    python scripts\\run_adamatch.py --data-root data --cache-dir data\\feature_cache \\
        --out-dir runs\\adamatch
    python scripts\\run_adamatch.py --folds 0 --epochs 1,3,8   # quick single fold
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from heads import LINEAR, HEAD_TYPES
from lens_dataset import LensDataset, UNLABELED
from samplers import make_balanced_loader, PairedDomainLoader
from da_losses import ConfirmationBiasDetector, adamatch_mu_at_step
from train_skeleton import Trainer, TrainConfig
from metrics import metric_suite, DEFAULT_TARGET_FPR, DEFAULT_TARGET_COMPLETENESS, NN_NP_Q1
from run_baseline import parse_epochs, set_seed
import run_coral as rc
import evaluate as ev
import curve_hook


# ---------------------------------------------------------------------------
# checkpoint naming (train_skeleton format, fold+epoch recoverable)
# ---------------------------------------------------------------------------

def ckpt_name(head_type: str, fold: int, epoch: int, width: int) -> str:
    """e.g. adamatch_linear_fold0_ep008.pt -- evaluate._epoch_from_name() recovers the
    epoch via '(?:ep|epoch)[_-]?(\\d+)'; the fold is also stored inside cfg."""
    return f"adamatch_{head_type}_fold{fold}_ep{epoch:0{width}d}.pt"


# ---------------------------------------------------------------------------
# per-fold AdaMatch training (IMAGE-SPACE; frozen encoder in the loop)
# ---------------------------------------------------------------------------

def train_adamatch_fold(splits_path, data_root, cache_dir, fold, ckpt_dir, *,
                        encoder=None, head_type=LINEAR, epochs=(1, 2, 5, 10, 20),
                        batch_size=64, pos_fraction=0.5, lr=1e-3, weight_decay=1e-4,
                        steps_per_epoch=None, mu_max=1.0, tau=0.9, ema_momentum=0.9,
                        consistency_warmup_frac=0.1, align_reference=None,
                        cb_patience=None, seed=42, device="cpu", num_workers=0,
                        fp16=False, eval_every_steps=None, curve_csv=None,
                        curve_folds=None, agreement_refs=None,
                        eval_target_fpr=DEFAULT_TARGET_FPR, eval_device="cpu",
                        verbose=True):
    """Train the head on source (BCE on the STRONG view) + AdaMatch consistency for one
    fold, dumping epoch checkpoints. Reads images from LensDataset(n_views=2) for
    source_train + target_fold{fold}; fits the standardizer from the CACHED source features
    (so train/eval standardization is identical). Returns (checkpoints, stability_log)."""
    if head_type not in HEAD_TYPES:
        raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type!r}")
    set_seed(seed + fold)                            # decorrelate folds' inits/shuffles
    dev = torch.device(device)

    # ---- datasets: weak+strong views for BOTH domains ----
    src_ds = LensDataset(splits_path, data_root, role="source_train",
                         aug="weak", n_views=2)
    tgt_ds = LensDataset(splits_path, data_root, role="target", fold=fold,
                         aug="weak", n_views=2)
    assert np.all(tgt_ds.labels == UNLABELED), "target labels must all be UNLABELED(-1)"

    schedule = sorted(set(int(e) for e in epochs))
    max_epoch = schedule[-1]
    width = max(2, len(str(max_epoch)))
    steps = steps_per_epoch or (len(src_ds) // batch_size)
    total_steps = steps * max_epoch                  # planned run -> mu schedule span

    # ---- balanced loaders: SOURCE on real labels, TARGET on latent identity ----
    src_loader = make_balanced_loader(src_ds, batch_size, use_latent=False,
                                      num_batches=steps, pos_fraction=pos_fraction,
                                      seed=seed + fold, num_workers=num_workers)
    tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True,
                                      num_batches=steps, pos_fraction=pos_fraction,
                                      seed=seed + fold + 1000, num_workers=num_workers)
    paired = PairedDomainLoader(src_loader, tgt_loader)

    # ---- Trainer with the frozen encoder + AdaMatch loss ----
    cfg = TrainConfig(
        splits_path=splits_path, data_root=data_root, fold=fold, head_type=head_type,
        encoder_mode="frozen", lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        pos_fraction=pos_fraction, use_adamatch=True, adamatch_mu_max=mu_max,
        adamatch_tau=tau, adamatch_ema_momentum=ema_momentum,
        adamatch_consistency_warmup_frac=consistency_warmup_frac,
        adamatch_align_reference=align_reference, total_steps=total_steps,
        epochs=max_epoch, seed=seed + fold, device=device, num_workers=num_workers,
        amp=fp16)                                    # fp16 -> autocast the forwards only
    trainer = Trainer(cfg, encoder=encoder)
    use_fp16 = bool(fp16) and trainer.device.type == "cuda"

    # ---- standardizer: fit from the CACHED source features (train/eval consistency) ----
    cache = ev.FeatureCache(cache_dir)
    if not cache.has("source_train"):
        raise FileNotFoundError(
            f"source_train.npz not found under {cache_dir}; run cache_features.py first "
            f"(AdaMatch fits the standardizer from the cached source features).")
    src_feats = torch.from_numpy(
        np.asarray(cache.load("source_train")["features"], dtype=np.float32)).to(dev)
    trainer.standardizer.fit(src_feats)
    del src_feats

    warmup_steps = int(math.ceil(consistency_warmup_frac * total_steps))
    patience = int(cb_patience) if cb_patience else steps
    detector = ConfirmationBiasDetector(patience=patience, warmup_steps=warmup_steps)

    if verbose:
        pk = adamatch_mu_at_step(total_steps // 2, total_steps, mu_max)
        print(f"  [fold {fold}] head={head_type}  {steps} steps/epoch  batch={batch_size}"
              f"  src={len(src_ds)}  tgt={len(tgt_ds)}  mu_max={mu_max}(mid={pk:.2f})"
              f"  tau={tau}  warmup={warmup_steps} steps  cb_patience={patience}  "
              f"fp16={'on' if use_fp16 else 'off'}", flush=True)

    cfg_extra = {"encoder_mode": "frozen", "lr": lr, "weight_decay": weight_decay,
                 "batch_size": batch_size, "pos_fraction": pos_fraction, "seed": seed,
                 "steps_per_epoch": steps, "method": "adamatch", "mu_max": mu_max,
                 "tau": tau, "ema_momentum": ema_momentum,
                 "consistency_warmup_frac": consistency_warmup_frac,
                 "align_reference": (None if align_reference is None
                                     else list(align_reference))}

    # curve fold set for the optional per-step hook (matched bake-off passes [0..4]);
    # None -> the legacy fold-0-only default (existing run_adamatch callers unchanged).
    cfolds = tuple(curve_hook.DEFAULT_FOLDS if curve_folds is None
                   else (int(k) for k in curve_folds))

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints: list[str] = []
    epoch_logs: list[dict] = []
    global_step = 0
    for epoch in range(1, max_epoch + 1):
        src_loader.batch_sampler.set_epoch(epoch)
        tgt_loader.batch_sampler.set_epoch(epoch)
        trainer.head.train()
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()               # fair wall-clock (async kernels)
        t_epoch = time.time()
        acc_list, posfrac_list = [], []
        last = {"sup_loss": float("nan"), "consistency": float("nan"),
                "mu_eff": 0.0, "acceptance": float("nan"),
                "accepted_pos_fraction": float("nan"), "threshold": float("nan")}
        epoch_fired = False
        for sbatch, tbatch in paired:
            stats = trainer.train_step(sbatch, tbatch, step=global_step,
                                       total_steps=total_steps)
            active = global_step >= warmup_steps
            if detector.update(stats["acceptance"], stats["accepted_pos_fraction"],
                               active=active):
                epoch_fired = True
            acc_list.append(stats["acceptance"])
            if math.isfinite(stats["accepted_pos_fraction"]):
                posfrac_list.append(stats["accepted_pos_fraction"])
            last = stats
            global_step += 1

            # matched-bake-off curve hook (behind eval_every_steps; default off so the
            # existing epoch-schedule run_adamatch is unchanged). Saves checkpoints/step_{N}.pt
            # at this exact head state and appends one FEATURE-SPACE curve row (all fold
            # columns + the read-only agreement column). Eval is deterministic aug="none" =
            # the cached features, so this reuses curve_hook exactly like CORAL -- the only
            # difference from CORAL is that TRAINING is image-space.
            if eval_every_steps and curve_csv and (global_step % int(eval_every_steps) == 0):
                spath = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                rc.save_checkpoint(spath, trainer.head, trainer.standardizer, head_type,
                                   fold, epoch, cfg_extra)
                curve_hook.append_curve_row(
                    curve_csv, global_step, spath, cache_dir, folds=cfolds,
                    target_fpr=eval_target_fpr, device=eval_device,
                    agreement_refs=agreement_refs)
                trainer.head.train()          # curve_hook loaded a separate head; be safe

        if trainer.device.type == "cuda":
            torch.cuda.synchronize()               # flush the epoch's kernels before timing
        epoch_secs = time.time() - t_epoch

        saved = epoch in schedule
        if saved:
            path = os.path.join(ckpt_dir, ckpt_name(head_type, fold, epoch, width))
            rc.save_checkpoint(path, trainer.head, trainer.standardizer, head_type,
                               fold, epoch, cfg_extra)
            checkpoints.append(path)

        mean_acc = float(np.mean(acc_list)) if acc_list else float("nan")
        mean_pf = float(np.mean(posfrac_list)) if posfrac_list else float("nan")
        epoch_logs.append({
            "epoch": epoch, "sup_loss": last["sup_loss"],
            "consistency": last["consistency"], "mu_eff": last["mu_eff"],
            "mean_acceptance": mean_acc, "mean_accepted_pos_fraction": mean_pf,
            "threshold": last["threshold"], "cb_fired": bool(epoch_fired),
            "secs": float(epoch_secs), "saved": bool(saved)})

        if verbose:
            print(f"    [ep {epoch:>{width}}] {epoch_secs:6.1f}s "
                  f"bce={last['sup_loss']:.4f} "
                  f"cons={last['consistency']:.4f} mu={last['mu_eff']:.3f} "
                  f"accept={mean_acc:.3f} acc_pos_frac={mean_pf:.3f} "
                  f"thr={last['threshold']:.3f} "
                  f"cb={'FIRED' if epoch_fired else 'ok'}"
                  + (f"  saved {os.path.basename(path)}" if saved else ""), flush=True)

    stability = {
        "fold": int(fold), "mu_max": float(mu_max), "tau": float(tau),
        "consistency_warmup_frac": float(consistency_warmup_frac),
        "warmup_steps": int(warmup_steps), "cb_patience": int(patience),
        "cb_detected": bool(detector.detected), "cb_reason": detector.reason,
        "align_reference": (None if align_reference is None else list(align_reference)),
        "n_saved_checkpoints": len(checkpoints), "epochs": epoch_logs}
    return checkpoints, stability


# ---------------------------------------------------------------------------
# metric suite on the dumped SOURCE-RISK per-fold scores (compare_methods family)
# ---------------------------------------------------------------------------

_SUITE_KEYS = ("tpr", "purity_corr", "purity_raw", "auprc", "comp_at_purity", "auroc")


def _suite_from_scores(out_dir, folds, *, target_fpr=DEFAULT_TARGET_FPR,
                       target_completeness=DEFAULT_TARGET_COMPLETENESS,
                       nn_np_ratio=NN_NP_Q1):
    """Compute the compare_methods metric suite on the SOURCE-RISK scores the evaluator
    dumped (scores/adamatch_fold{k}.npz), per fold + across-fold mean."""
    sdir = os.path.join(out_dir, "scores")
    per_fold, agg = {}, {k: [] for k in _SUITE_KEYS}
    for k in folds:
        p = os.path.join(sdir, f"adamatch_fold{k}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=False) as z:
            pos, neg = z["pos_scores"], z["neg_scores"]
        y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
        s = np.concatenate([pos, neg]).astype(float)
        su = metric_suite(y, s, target_fpr=target_fpr,
                          target_completeness=target_completeness, nn_np_ratio=nn_np_ratio)
        per_fold[int(k)] = {kk: _num(su[kk]) for kk in _SUITE_KEYS}
        for kk in _SUITE_KEYS:
            agg[kk].append(su[kk])
    mean = {kk: _num(float(np.nanmean(agg[kk]))) if agg[kk] else None for kk in _SUITE_KEYS}
    # std only meaningful with >=2 folds (avoids a nanstd DoF<=0 warning on a single fold)
    std = {kk: (_num(float(np.nanstd(agg[kk]))) if len(agg[kk]) > 1 else 0.0) if agg[kk]
           else None for kk in _SUITE_KEYS}
    return {"per_fold": per_fold, "mean": mean, "std": std,
            "target_completeness": target_completeness, "nn_np_ratio": float(nn_np_ratio)}


def _num(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _fmt(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


def _report_suite(suite, folds):
    print("\n" + "=" * 80)
    print("ADAMATCH METRIC SUITE  (SOURCE-RISK per-fold scores; compare_methods family)")
    print(f"  purity corrected to N_N/N_P = 1:{int(round(suite['nn_np_ratio']))} "
          f"(Lines Q1 rarity);  purity/comp @ {suite['target_completeness']*100:g}% completeness")
    print("=" * 80)
    m, s = suite["mean"], suite["std"]
    labels = [("tpr", "TPR@1e-3           (HEADLINE)"),
              ("purity_corr", "purity@50% (corr 1:2200) HEAD"),
              ("auprc", "AUPRC              (HEADLINE)"),
              ("purity_raw", "purity@50% (raw pool)"),
              ("comp_at_purity", "completeness@24% purity"),
              ("auroc", "AUROC              (secondary)")]
    for key, lab in labels:
        print(f"    {lab:<30} across-fold mean = {_fmt(m[key])} +- {_fmt(s[key])}")


def _report_stability(stability, folds):
    print("\n" + "=" * 80)
    print("ADAMATCH CONFIRMATION-BIAS LOG  (acceptance + accepted class balance; diagnostic)")
    print("=" * 80)
    for k in sorted(folds):
        st = stability.get(int(k)) or stability.get(str(k))
        if st is None:
            continue
        print(f"\n  [fold {k}]  mu_max={st['mu_max']:g}  tau={st['tau']:g}  "
              f"warmup={st['warmup_steps']} steps  "
              f"collapse={'DETECTED (' + str(st['cb_reason']) + ')' if st['cb_detected'] else 'none'}")
        print("    epoch |   bce   |  cons  |  mu   | accept | acc_pos_frac | cb    | saved")
        print("    " + "-" * 72)
        for e in st["epochs"]:
            print(f"    {e['epoch']:>5} | {_fmt(e['sup_loss'],4):>7} | "
                  f"{_fmt(e['consistency'],4):>6} | {_fmt(e['mu_eff'],3):>5} | "
                  f"{_fmt(e['mean_acceptance'],3):>6} | "
                  f"{_fmt(e['mean_accepted_pos_fraction'],3):>12} | "
                  f"{'FIRED' if e['cb_fired'] else ' ok':>5} | "
                  f"{'yes' if e['saved'] else ' no'}")
    n_bad = sum(1 for k in folds
                if (stability.get(int(k)) or stability.get(str(k), {})).get("cb_detected"))
    print("\n" + "=" * 80)
    print(f"CONFIRMATION-BIAS SUMMARY: {n_bad}/{len(list(folds))} folds tripped the collapse "
          f"diagnostic (logged, NOT a selector).")
    print("=" * 80)


# ---------------------------------------------------------------------------
# full AdaMatch run: train per fold (image-space) -> per-fold eval -> aggregate
# ---------------------------------------------------------------------------

def run_adamatch(splits_path, data_root, cache_dir, out_dir, *, folds=(0, 1, 2, 3, 4),
                 head_type=LINEAR, epochs=(1, 2, 5, 10, 20), batch_size=64,
                 pos_fraction=0.5, lr=1e-3, weight_decay=1e-4, steps_per_epoch=None,
                 mu_max=1.0, tau=0.9, ema_momentum=0.9, consistency_warmup_frac=0.1,
                 align_reference=None, cb_patience=None, seed=42, train_device="cpu",
                 eval_device="cpu", num_workers=0, fp16=False, target_fpr=DEFAULT_TARGET_FPR,
                 bootstrap=1000, eval_seed=0, val_select_metric="auroc",
                 encoder=None, verbose=True) -> dict:
    from zoobot_encoder import ZoobotEncoder
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if verbose:
        print(f"AdaMatch  splits={splits_path}  cache={cache_dir}  out={out_dir}  "
              f"folds={list(folds)}  mu_max={mu_max} tau={tau} "
              f"warmup_frac={consistency_warmup_frac} "
              f"align={'source-EMA' if align_reference is None else list(align_reference)}")

    # frozen encoder built ONCE (image-space training runs it in the loop)
    if encoder is None:
        if verbose:
            print("[load] building the frozen Zoobot encoder (once) ...", flush=True)
        encoder = ZoobotEncoder(encoder_mode="frozen", device=train_device, verbose=False)

    fold_ckpts, fold_stability = {}, {}
    for k in folds:
        if verbose:
            print(f"\n[train fold {k}] (image-space AdaMatch; encoder in the loop) ...")
        fold_ckpts[k], fold_stability[k] = train_adamatch_fold(
            splits_path, data_root, cache_dir, k, ckpt_dir, encoder=encoder,
            head_type=head_type, epochs=epochs, batch_size=batch_size,
            pos_fraction=pos_fraction, lr=lr, weight_decay=weight_decay,
            steps_per_epoch=steps_per_epoch, mu_max=mu_max, tau=tau,
            ema_momentum=ema_momentum, consistency_warmup_frac=consistency_warmup_frac,
            align_reference=align_reference, cb_patience=cb_patience, seed=seed,
            device=train_device, num_workers=num_workers, fp16=fp16, verbose=verbose)

    empty = [k for k in folds if not fold_ckpts[k]]
    if empty:
        raise RuntimeError(f"folds {empty} produced NO checkpoints (schedule empty?).")

    if verbose:
        print(f"\n[eval] per-fold models -> per-fold TPR@{target_fpr:g} + across-fold aggregate")
    results = rc.evaluate_per_fold(
        fold_ckpts, cache_dir, device=eval_device, target_fpr=target_fpr, B=bootstrap,
        seed=eval_seed, val_select_metric=val_select_metric, out_dir=out_dir,
        verbose=verbose, method="adamatch", score_prefix="adamatch")

    # the compare_methods metric suite (AUPRC + purity) on the SOURCE-RISK scores
    suite = _suite_from_scores(out_dir, folds, target_fpr=target_fpr)
    results["metric_suite"] = suite
    results["adamatch_stability"] = {int(k): fold_stability[k] for k in fold_stability}
    if verbose:
        _report_suite(suite, folds)
        _report_stability(results["adamatch_stability"], folds)

    if out_dir:
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.normpath(os.path.join(here, "..", "data"))
    p = argparse.ArgumentParser(description="AdaMatch -- per-fold DA runner (image-space).")
    p.add_argument("--data-root", default=os.environ.get("LENS_DATA_ROOT", default_root))
    p.add_argument("--splits", default=None,
                   help="splits.json path (default: <data-root>/splits/splits.json)")
    p.add_argument("--cache-dir", default=os.path.join(default_root, "feature_cache"),
                   help="feature cache (used for the standardizer fit + evaluation)")
    p.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "runs", "adamatch")))
    p.add_argument("--folds", default="0,1,2,3,4", help="comma-separated fold ids")
    p.add_argument("--head-type", default=LINEAR, choices=HEAD_TYPES)
    p.add_argument("--epochs", default="1,2,5,10,20",
                   help="comma-separated epoch schedule = SOURCE-RISK selection candidates")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pos-fraction", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--mu-max", type=float, default=1.0,
                   help="peak consistency weight -- FIXED, not tuned on target")
    p.add_argument("--tau", type=float, default=0.9, help="relative confidence factor")
    p.add_argument("--ema-momentum", type=float, default=0.9)
    p.add_argument("--consistency-warmup-frac", type=float, default=0.1,
                   help="fraction of steps to withhold the consistency loss entirely")
    p.add_argument("--align-reference", default=None,
                   help="'p_neg,p_pos' to override the alignment reference (default: source "
                        "EMA ~[0.5,0.5]); e.g. '2000,1' for a 1:2000 ablation")
    p.add_argument("--cb-patience", type=int, default=None,
                   help="confirmation-bias detector patience in steps (default: one epoch)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-device", default=None)
    p.add_argument("--eval-device", default="cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--fp16", action="store_true",
                   help="autocast the encoder+head FORWARDS to fp16 on CUDA (loss/EMA/"
                        "alignment stay fp32); default off")
    p.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--val-select-metric", default="auroc", choices=("auroc", "tpr"))
    args = p.parse_args()

    splits_path = args.splits or os.path.join(args.data_root, "splits", "splits.json")
    if not os.path.exists(splits_path):
        print(f"SKIP: real splits not found at {splits_path}")
        return 3
    if not os.path.isdir(args.cache_dir):
        print(f"SKIP: feature cache dir not found: {args.cache_dir} (run cache_features.py first)")
        return 3
    cache = ev.FeatureCache(args.cache_dir)
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    for need in ["source_train", "test_negatives_shared"] + [f"target_fold{k}" for k in folds] \
            + [f"test_pos_fold{k}" for k in folds]:
        if not cache.has(need):
            print(f"SKIP: cache missing {need}.npz (run cache_features.py first)")
            return 3

    align_reference = None
    if args.align_reference:
        align_reference = [float(x) for x in args.align_reference.split(",")]
        if len(align_reference) != 2:
            print("--align-reference must be 'p_neg,p_pos'")
            return 2

    train_device = args.train_device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = parse_epochs(args.epochs)
    run_adamatch(
        splits_path, args.data_root, args.cache_dir, args.out_dir, folds=folds,
        head_type=args.head_type, epochs=epochs, batch_size=args.batch_size,
        pos_fraction=args.pos_fraction, lr=args.lr, weight_decay=args.weight_decay,
        steps_per_epoch=args.steps_per_epoch, mu_max=args.mu_max, tau=args.tau,
        ema_momentum=args.ema_momentum,
        consistency_warmup_frac=args.consistency_warmup_frac,
        align_reference=align_reference, cb_patience=args.cb_patience, seed=args.seed,
        train_device=train_device, eval_device=args.eval_device,
        num_workers=args.num_workers, fp16=args.fp16, target_fpr=args.target_fpr,
        bootstrap=args.bootstrap, eval_seed=args.eval_seed,
        val_select_metric=args.val_select_metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
