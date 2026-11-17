"""
run_coral.py
============
Deep CORAL -- the FIRST real domain-adaptation method in the bake-off, and the first
runner that actually uses the target domain. It is the CORAL counterpart of
run_baseline.py (source-only Baseline A) and shares its feature-cache, head,
standardizer, sampler, checkpoint format, and evaluation metrics -- so the two are
directly comparable.

WHAT IS DIFFERENT FROM BASELINE A (and WHY)
-------------------------------------------
  * CORAL USES THE TARGET, and the target ROTATES PER FOLD (PROJECT.md 4.3): fold k's
    target pool = the ~391 real lenses of the OTHER four folds (UNLABELED) + ~391 fresh
    random negatives. So unlike Baseline A (one fold-agnostic model), CORAL trains ONE
    MODEL PER FOLD, each on that fold's own target pool, and each evaluated on that
    fold's held-out test positives + the shared negatives. This is the per-fold-model
    path evaluate.py explicitly left "for the bake-off" -- implemented here.

  * TOTAL LOSS = BCE(source only) + coral_weight * CORAL(src_adapter, tgt_adapter).
    coral_weight (gamma) is a FIXED config value (default 1.0), NOT tuned on the target
    -- RLSbench-style fixed-budget protocol (PROJECT.md 7.4), so it can never become an
    accidental oracle knob. The CORAL loss lives in da_losses.CoralLoss and consumes
    the HEAD/adapter features only (never the frozen encoder), so it cannot backprop
    into the encoder. The target batch stays UNLABELED (assert_target_unlabeled runs
    every step); CORAL uses only its features.

  * LABEL-FREE MODEL SELECTION is unchanged: we save a schedule of epoch checkpoints
    per fold and pick the reported epoch by the best SOURCE-VAL metric (val is labeled
    held-out SIM -- the only label-free-legal selector; PROJECT.md 4.2). An ORACLE
    column (best real-TEST epoch) is printed only to expose the selection gap.

READS CACHED FEATURES ONLY (PROJECT.md 7.1 / cache_features.py)
--------------------------------------------------------------
The encoder is frozen, so cache_features.py already encoded every split to disk. This
runner reads source_train.npz, val.npz, and target_fold{k}.npz for training, and hands
the checkpoints to a per-fold evaluator that reads test_pos_fold{k}.npz +
test_negatives_shared.npz. It never loads the encoder. Fast on CPU.

  ****  LEAKAGE CONTRACT (asserted below)  ****
  Training fold k may read ONLY source_train, val, and target_fold{k}. It NEVER reads
  the test folds (that is the evaluator's job), and fold k's target pool by construction
  EXCLUDES fold k's own lenses (PROJECT.md 4.3), so no model is ever trained on a lens
  it is later tested on.

AGGREGATION (per-fold models -> one reported number)
----------------------------------------------------
Because each fold has its OWN model, the shared-negative single-ROC pooling that
Baseline A uses is NOT valid here (there is no single threshold across models). Instead
we report per-fold TPR@1e-3 / AUROC and the ACROSS-FOLD mean +- std, with a fold-aware
class-conditional bootstrap CI (resample each fold's positives and the shared negatives
with replacement, average the metric across folds per replicate). This is the honest
aggregation for a per-fold-model method and is the extension evaluate.py flagged.

Run (after cache_features.py has populated the cache):
    python scripts\\run_coral.py --cache-dir data\\feature_cache --out-dir runs\\coral
    python scripts\\run_coral.py --folds 0 --epochs 1,3,8            # quick single fold
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Reused verbatim from the baseline / skeleton -- same head, same checkpoint format,
# same class-balanced batching, same metrics -- so CORAL is comparable to Baseline A.
from heads import build_head, FeatureStandardizer, EMBED_DIM, LINEAR, HEAD_TYPES
from samplers import ClassBalancedBatchSampler, assert_target_unlabeled, POS_LABEL, NEG_LABEL
from da_losses import CoralLoss, BlowupDetector
from lens_dataset import UNLABELED


# ---------------------------------------------------------------------------
# CORAL WEIGHT (gamma) SCHEDULE  -- the "cheap" stability fix, part 1
# ---------------------------------------------------------------------------
#
# The late-training Frobenius covariance-term blow-up (arXiv:1705.08180) is a GEOMETRY
# pathology, not exploding gradients -- so instead of clipping or changing the cov math
# we SHAPE HOW the FIXED peak gamma is applied over training steps, so the blow-up can
# never dominate the loss when it kicks in late.
#
#   * "constant"      -- gamma == peak every step (the original behaviour; default for
#                        backward-compat).
#   * "warmup_cosine" -- linearly ramp gamma 0 -> peak over ``warmup_frac`` of the total
#                        steps (warmup keeps CORAL from destabilising the head before it
#                        can classify), then COSINE-decay peak -> ``min_frac``*peak over
#                        the rest (the decay starves the late-training cov-term of weight
#                        exactly when the Frobenius pathology appears).
#   * "warmup_hold"   -- ramp 0 -> peak over warmup, then HOLD peak (simpler comparison).
#
# IMPORTANT (honesty): the PEAK gamma is the un-tuned config value (default 1.0); the
# schedule only shapes HOW it is applied across steps. The schedule SHAPE (warmup_frac,
# min_frac, cosine) is FIXED, NOT tuned on the target -- it is a stability prior, not an
# oracle knob, exactly like the fixed gamma (RLSbench fixed-budget protocol).

CORAL_SCHEDULES = ("constant", "warmup_cosine", "warmup_hold")


def coral_gamma_at_step(step: int, total_steps: int, peak: float, *,
                        schedule: str = "constant", warmup_frac: float = 0.1,
                        min_frac: float = 0.1) -> float:
    """Effective CORAL weight (gamma) at 0-based global ``step`` of ``total_steps``.

    ``constant`` -> ``peak``; ``warmup_hold`` ramps 0->peak then holds; ``warmup_cosine``
    ramps 0->peak over ``warmup_frac`` then cosine-decays to ``min_frac*peak``. The peak
    is reached exactly at the end of warmup for both scheduled variants.
    """
    if schedule not in CORAL_SCHEDULES:
        raise ValueError(f"coral schedule must be one of {CORAL_SCHEDULES}, got {schedule!r}")
    peak = float(peak)
    if schedule == "constant":
        return peak
    total = max(int(total_steps), 1)
    frac = min(max(step / total, 0.0), 1.0)          # progress through training in [0,1]
    wf = min(max(float(warmup_frac), 0.0), 1.0)
    if wf > 0.0 and frac < wf:                        # linear warmup 0 -> peak
        return peak * (frac / wf)
    if schedule == "warmup_hold":
        return peak
    # warmup_cosine: cosine decay peak -> min_frac*peak over the post-warmup remainder
    min_val = float(min_frac) * peak
    denom = max(1.0 - wf, 1e-12)
    decay_progress = min(max((frac - wf) / denom, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))   # 1 at start -> 0 at end
    return float(min_val + (peak - min_val) * cosine)
from metrics import auroc, tpr_at_fpr, DEFAULT_TARGET_FPR
from run_baseline import parse_epochs, set_seed, _cf
import evaluate as ev
import curve_hook

# Training for fold k may read ONLY these cache splits (the runtime half of the
# leakage contract). test_pos_fold* / test_negatives_shared are the evaluator's.
def _train_allowed(fold: int) -> tuple[str, ...]:
    return ("source_train", "val", f"target_fold{fold}")


def _guarded_load(cache: ev.FeatureCache, name: str, fold: int) -> dict:
    allowed = _train_allowed(fold)
    assert name in allowed, (
        f"run_coral fold {fold} training may only read {allowed}; refusing to load "
        f"'{name}' (test folds are the evaluator's job, not training's).")
    if not cache.has(name):
        raise FileNotFoundError(
            f"required cache split '{name}.npz' not found under {cache.dir}; "
            f"run cache_features.py first.")
    return cache.load(name)


# ---------------------------------------------------------------------------
# checkpoint naming / saving (train_skeleton format, with the fold recorded)
# ---------------------------------------------------------------------------

def ckpt_name(head_type: str, fold: int, epoch: int, width: int) -> str:
    """e.g. coral_linear_fold0_ep008.pt -- evaluate._epoch_from_name() recovers the
    epoch via '(?:ep|epoch)[_-]?(\\d+)'; the fold is also stored inside cfg."""
    return f"coral_{head_type}_fold{fold}_ep{epoch:0{width}d}.pt"


def save_checkpoint(path, head, standardizer, head_type, fold, epoch, cfg_extra: dict,
                    da_buffers: dict | None = None, optimizer_state: dict | None = None,
                    schedule: dict | None = None):
    """Write a checkpoint.

    ``da_buffers`` / ``optimizer_state`` / ``schedule`` are the TRAINING STATE around the
    model. Round-1 checkpoints have exactly four keys and none of it, which is why
    continuing from one restarted FreeMatch's self-adaptive threshold from 1/C and Adam's
    moments from zero -- measured as a 0.153 TPR excursion on the fold-0 control. They are
    optional so every existing caller keeps producing the identical four-key file, and the
    format is additive so `evaluate.load_checkpoint` reads both unchanged.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cfg = {"head_type": head_type, "fold": int(fold), "epoch": int(epoch), **cfg_extra}
    payload = {
        "head_type": head_type,
        "head": head.state_dict(),
        "standardizer": standardizer.state_dict(),
        "cfg": cfg,
    }
    if da_buffers is not None or optimizer_state is not None or schedule is not None:
        payload["format_version"] = 2
        payload["da_buffers"] = da_buffers or {}
        payload["schedule"] = dict(schedule or {})
        if optimizer_state is not None:
            payload["optimizer"] = optimizer_state
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# per-fold CORAL training (feature-space; ONE model for this fold)
# ---------------------------------------------------------------------------

def train_coral_fold(cache, fold, ckpt_dir, *, head_type=LINEAR,
                     epochs=(1, 2, 5, 10, 20, 50), batch_size=256, pos_fraction=0.5,
                     lr=1e-3, weight_decay=1e-4, steps_per_epoch=None,
                     coral_weight=1.0, coral_include_mean=True, coral_shrinkage=0.1,
                     coral_target_ema=True, coral_ema_momentum=0.9,
                     coral_weight_schedule="constant", coral_warmup_frac=0.1,
                     coral_min_frac=0.1, blowup_factor=5.0, blowup_baseline_steps=None,
                     stop_on_blowup=True,
                     eval_every_steps=None, curve_csv=None, curve_folds=None,
                     agreement_refs=None, eval_target_fpr=DEFAULT_TARGET_FPR,
                     eval_device="cpu",
                     seed=42, device="cpu", verbose=True):
    """Train the head on source (BCE) + gamma(step)*CORAL(src_adapter, tgt_adapter) for
    one fold, dumping epoch checkpoints. Reads ONLY source_train / val / target_fold{fold}.

    gamma(step) follows ``coral_weight_schedule`` (constant / warmup_cosine / warmup_hold)
    with FIXED peak ``coral_weight``; a per-fold :class:`BlowupDetector` watches the CORAL
    cov-term and, if ``stop_on_blowup``, stops the fold at the first blow-up epoch WITHOUT
    saving that epoch's (or any later) checkpoint.

    Returns ``(checkpoints, stability)`` -- the SOURCE-RISK selection candidates (stable
    checkpoints only) and the per-fold stability diagnostic log.
    """
    if head_type not in HEAD_TYPES:
        raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type!r}")
    set_seed(seed + fold)                       # decorrelate folds' inits/shuffles
    dev = torch.device(device)

    # ---- source (labeled): the supervised signal ----
    src = _guarded_load(cache, "source_train", fold)
    src_feats = torch.from_numpy(np.asarray(src["features"], dtype=np.float32)).to(dev)
    src_labels = np.asarray(src["labels"], dtype=np.int64)
    if src_feats.shape[1] != EMBED_DIM:
        raise ValueError(f"source features are {src_feats.shape[1]}-d, expected {EMBED_DIM}")
    pos_idx = np.where(src_labels == POS_LABEL)[0]
    neg_idx = np.where(src_labels == NEG_LABEL)[0]
    y = torch.from_numpy((src_labels == POS_LABEL).astype(np.float32)).view(-1, 1).to(dev)

    # ---- standardizer: fit ONCE on source, freeze, apply to BOTH domains ----
    standardizer = FeatureStandardizer(EMBED_DIM).to(dev)
    standardizer.fit(src_feats)
    x_src = standardizer(src_feats).detach()    # standardized source features (constant)

    # ---- target (UNLABELED, this fold's pool): the alignment domain ----
    tgt = _guarded_load(cache, f"target_fold{fold}", fold)
    tgt_feats = torch.from_numpy(np.asarray(tgt["features"], dtype=np.float32)).to(dev)
    tgt_served = np.asarray(tgt["labels"], dtype=np.int64)          # must be all UNLABELED
    tgt_latent = np.asarray(tgt["latent_labels"], dtype=np.int64)   # 0/1 identity (balancing only)
    assert np.all(tgt_served == UNLABELED), (
        f"fold {fold} target labels must all be UNLABELED({UNLABELED}); "
        f"got {np.unique(tgt_served).tolist()}")
    x_tgt = standardizer(tgt_feats).detach()    # SAME source-fit standardizer
    tpos_idx = np.where(tgt_latent == POS_LABEL)[0]
    tneg_idx = np.where(tgt_latent == NEG_LABEL)[0]

    # ---- head + optimizer (head params ONLY; there is no encoder here) ----
    head = build_head(head_type).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    coral = CoralLoss(head.feat_dim, include_mean=coral_include_mean,
                      shrinkage=coral_shrinkage, target_ema=coral_target_ema,
                      ema_momentum=coral_ema_momentum).to(dev)

    # ---- class-balanced batches: SOURCE on real labels, TARGET on latent identity ----
    src_sampler = ClassBalancedBatchSampler(
        pos_idx, neg_idx, batch_size=batch_size, num_batches=steps_per_epoch,
        pos_fraction=pos_fraction, seed=seed + fold)
    steps = len(src_sampler)
    # target sampler yields the SAME number of batches (cycled with replacement over the
    # small ~391/391 pool) so each step has a paired (source, target) batch -- the
    # PairedDomainLoader pattern realised with two equal-length samplers.
    tgt_sampler = ClassBalancedBatchSampler(
        tpos_idx, tneg_idx, batch_size=batch_size, num_batches=steps,
        pos_fraction=pos_fraction, seed=seed + fold + 1000)

    if coral_weight_schedule not in CORAL_SCHEDULES:
        raise ValueError(f"coral_weight_schedule must be one of {CORAL_SCHEDULES}, "
                         f"got {coral_weight_schedule!r}")
    schedule = sorted(set(int(e) for e in epochs))
    max_epoch = schedule[-1]
    width = max(2, len(str(max_epoch)))
    total_steps = steps * max_epoch                    # schedule spans the PLANNED run
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg_extra = {"encoder_mode": "frozen", "lr": lr, "weight_decay": weight_decay,
                 "batch_size": batch_size, "pos_fraction": pos_fraction, "seed": seed,
                 "steps_per_epoch": steps, "method": "deep_coral",
                 "coral_weight": coral_weight, "coral_include_mean": coral_include_mean,
                 "coral_shrinkage": coral_shrinkage, "coral_target_ema": coral_target_ema,
                 "coral_ema_momentum": coral_ema_momentum,
                 "coral_weight_schedule": coral_weight_schedule,
                 "coral_warmup_frac": coral_warmup_frac, "coral_min_frac": coral_min_frac}

    # blow-up detector: healthy window = one epoch of steps unless overridden
    baseline_window = int(blowup_baseline_steps) if blowup_baseline_steps else steps
    detector = BlowupDetector(baseline_window=baseline_window, blowup_factor=blowup_factor)

    # curve fold set for the optional per-step hook (matched bake-off passes [0..4]);
    # None -> the legacy fold-0-only default (existing run_coral callers unchanged).
    cfolds = tuple(curve_hook.DEFAULT_FOLDS if curve_folds is None
                   else (int(k) for k in curve_folds))

    if verbose:
        print(f"  [fold {fold}] head={head_type}  {steps} steps/epoch  batch={batch_size}"
              f" (pos_frac~{src_sampler.actual_pos_fraction:.2f})  "
              f"src={pos_idx.size}+{neg_idx.size}  tgt={tpos_idx.size}+{tneg_idx.size}  "
              f"peak_gamma={coral_weight}  schedule={coral_weight_schedule}  "
              f"blowup(x{blowup_factor:g},win={baseline_window},stop={stop_on_blowup})")

    def _fmt_base(x):
        return "n/a" if x is None else f"{x:.4g}"

    checkpoints: list[str] = []
    epoch_logs: list[dict] = []
    blowup_epoch = None
    global_step = 0
    last_gamma = coral_gamma_at_step(0, total_steps, coral_weight,
                                     schedule=coral_weight_schedule,
                                     warmup_frac=coral_warmup_frac, min_frac=coral_min_frac)
    for epoch in range(1, max_epoch + 1):
        src_sampler.set_epoch(epoch)
        tgt_sampler.set_epoch(epoch)
        head.train(); coral.train()
        last_sup = last_da = 0.0
        epoch_blewup = False
        for sbatch, tbatch in zip(src_sampler, tgt_sampler):
            sidx = torch.as_tensor(sbatch, dtype=torch.long, device=dev)
            tidx = torch.as_tensor(tbatch, dtype=torch.long, device=dev)
            # label-free guard, EVERY step: the target's served labels stay UNLABELED
            assert_target_unlabeled({"label": torch.as_tensor(tgt_served[tbatch])})

            gamma_eff = coral_gamma_at_step(global_step, total_steps, coral_weight,
                                            schedule=coral_weight_schedule,
                                            warmup_frac=coral_warmup_frac,
                                            min_frac=coral_min_frac)
            opt.zero_grad(set_to_none=True)
            src_logits, src_adapter = head.forward_with_features(x_src[sidx])
            _, tgt_adapter = head.forward_with_features(x_tgt[tidx])
            sup_loss = bce(src_logits, y[sidx])
            da_loss = gamma_eff * coral(src_adapter, tgt_adapter)
            loss = sup_loss + da_loss
            loss.backward()
            opt.step()
            last_sup = float(sup_loss.detach()); last_da = float(da_loss.detach())
            last_gamma = gamma_eff
            global_step += 1

            # matched-bake-off curve hook (behind eval_every_steps; default off so the
            # existing epoch-schedule run_coral is byte-for-byte unchanged). Saves
            # checkpoints/step_{N}.pt at this exact model state and appends one curve row
            # (all fold TPR/AUPRC/AUROC columns PLUS the read-only agreement column). All
            # scoring/metrics/Spearman are reused from curve_hook -- nothing re-implemented,
            # and agreement never touches the loss.
            if eval_every_steps and curve_csv and (global_step % int(eval_every_steps) == 0):
                spath = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                save_checkpoint(spath, head, standardizer, head_type, fold, epoch, cfg_extra)
                row = curve_hook.append_curve_row(
                    curve_csv, global_step, spath, cache.dir, folds=cfolds,
                    target_fpr=eval_target_fpr, device=eval_device,
                    agreement_refs=agreement_refs)
                head.train(); coral.train()          # curve_hook loaded a separate head; be safe
                if verbose:
                    per_fold = "  ".join(
                        f"f{k}={_cf(row[f'fold{k}_tpr_1e-3'])}" for k in cfolds)
                    agr = ("" if not agreement_refs else "  agr " + ",".join(
                        f"f{k}={_cf(row.get(f'agreement_fold{k}'))}"
                        for k in sorted(agreement_refs)))
                    print(f"    [step {global_step:>6}] curve  bce={last_sup:.4f} "
                          f"coral={last_da:.4f} | tpr@1e-3 {per_fold}{agr}", flush=True)

            # stability diagnostic: feed the RAW (unweighted) cov-term every step
            if detector.update(coral.last_cov_term):
                epoch_blewup = True
                if stop_on_blowup:
                    break                              # stop the fold at the blow-up

        blewup_and_stopping = epoch_blewup and stop_on_blowup
        saved = (epoch in schedule) and not blewup_and_stopping
        if saved:
            path = os.path.join(ckpt_dir, ckpt_name(head_type, fold, epoch, width))
            save_checkpoint(path, head, standardizer, head_type, fold, epoch, cfg_extra)
            checkpoints.append(path)

        epoch_logs.append({
            "epoch": epoch, "cov_term": float(coral.last_cov_term),
            "cov_ema": (None if detector.ema is None else float(detector.ema)),
            "baseline_cov": detector.baseline, "gamma_eff": float(last_gamma),
            "blowup": bool(epoch_blewup), "saved": bool(saved)})

        if verbose:
            print(f"    [ep {epoch:>{width}}] gamma={last_gamma:.4f} bce={last_sup:.4f} "
                  f"coral={last_da:.4f} cov={coral.last_cov_term:.4g} "
                  f"baseline={_fmt_base(detector.baseline)} "
                  f"blowup={'YES' if epoch_blewup else 'no'}"
                  + (f"  saved {os.path.basename(path)}" if saved else ""), flush=True)

        if blewup_and_stopping:
            blowup_epoch = epoch
            if verbose:
                print(f"    [fold {fold}] BLOW-UP at epoch {epoch} "
                      f"(cov={coral.last_cov_term:.4g} > {blowup_factor:g}x baseline "
                      f"{_fmt_base(detector.baseline)}) -> stopping; no checkpoints from "
                      f"epoch {epoch} onward.", flush=True)
            break

    stability = {
        "fold": int(fold), "schedule": coral_weight_schedule,
        "peak_gamma": float(coral_weight), "warmup_frac": float(coral_warmup_frac),
        "min_frac": float(coral_min_frac), "blowup_factor": float(blowup_factor),
        "baseline_window_steps": int(baseline_window),
        "baseline_cov": detector.baseline, "stop_on_blowup": bool(stop_on_blowup),
        "blowup_epoch": blowup_epoch, "n_saved_checkpoints": len(checkpoints),
        "epochs": epoch_logs}
    return checkpoints, stability


# ---------------------------------------------------------------------------
# per-fold-model evaluation (each fold's model on its own held-out fold)
# ---------------------------------------------------------------------------

def _val_metric(head, std, cache, device, metric="auroc",
                target_fpr=DEFAULT_TARGET_FPR):
    if not cache.has("val"):
        return np.nan
    v = cache.load("val")
    s = ev.score_features(head, std, v["features"], device)
    return (auroc(v["labels"], s) if metric == "auroc"
            else tpr_at_fpr(v["labels"], s, target_fpr))


def _score_ckpt_on_fold(ckpt, cache, fold, neg_feats, device):
    """Score fold `fold`'s test positives + the shared negatives with `ckpt`'s model."""
    head, std, _ = ev.load_checkpoint(ckpt, device)
    pos = ev.score_features(head, std, cache.load(f"test_pos_fold{fold}")["features"], device)
    neg = ev.score_features(head, std, neg_feats, device)
    return pos, neg


def _aggregate(selected_scores, target_fpr, B, seed):
    """selected_scores: {fold: (pos_scores, neg_scores)}.
    Returns per-fold metrics, across-fold mean+-std, and a fold-aware class-conditional
    bootstrap CI on the across-fold MEAN of TPR and AUROC."""
    folds = sorted(selected_scores)
    per_fold = {}
    for k in folds:
        pos, neg = selected_scores[k]
        y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
        s = np.concatenate([pos, neg])
        per_fold[k] = {"n_pos": int(pos.size),
                       "tpr": tpr_at_fpr(y, s, target_fpr), "auroc": auroc(y, s)}
    tprs = [per_fold[k]["tpr"] for k in folds]
    aus = [per_fold[k]["auroc"] for k in folds]
    mean_tpr, std_tpr = float(np.nanmean(tprs)), float(np.nanstd(tprs))
    mean_auroc, std_auroc = float(np.nanmean(aus)), float(np.nanstd(aus))

    # bootstrap: resample the shared negatives ONCE per replicate (paired across the
    # per-fold models, which all score the same negative rows), resample each fold's
    # positives independently, average the metric across folds.
    rng = np.random.default_rng(seed)
    n_neg = next(iter(selected_scores.values()))[1].size
    reps_tpr = np.empty(B, float); reps_au = np.empty(B, float)
    for b in range(B):
        nidx = rng.integers(0, n_neg, n_neg)
        t_b, a_b = [], []
        for k in folds:
            pos, neg = selected_scores[k]
            pidx = rng.integers(0, pos.size, pos.size)
            y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(nidx.size, np.int64)])
            s = np.concatenate([pos[pidx], neg[nidx]])
            t_b.append(tpr_at_fpr(y, s, target_fpr)); a_b.append(auroc(y, s))
        reps_tpr[b] = np.nanmean(t_b); reps_au[b] = np.nanmean(a_b)

    def _ci(reps, point):
        return {"point": point, "median": float(np.nanmedian(reps)),
                "lo": float(np.nanpercentile(reps, 2.5)),
                "hi": float(np.nanpercentile(reps, 97.5))}
    return {"per_fold": per_fold, "mean_tpr": mean_tpr, "std_tpr": std_tpr,
            "mean_auroc": mean_auroc, "std_auroc": std_auroc,
            "tpr_ci": _ci(reps_tpr, mean_tpr), "auroc_ci": _ci(reps_au, mean_auroc)}


def evaluate_per_fold(fold_ckpts, cache_dir, *, device="cpu",
                      target_fpr=DEFAULT_TARGET_FPR, B=1000, seed=0,
                      val_select_metric="auroc", out_dir=None, verbose=True,
                      method="deep_coral", score_prefix="coral") -> dict:
    """fold_ckpts: {fold: [epoch_ckpt_paths]}. For each fold pick the SOURCE-RISK epoch
    (best source-val) and the ORACLE epoch (best test), evaluate the selected model on
    that fold's held-out positives + shared negatives, and aggregate across folds.

    ``method`` / ``score_prefix`` label the run so this per-fold evaluator is reusable by
    other per-fold-model DA methods (e.g. AdaMatch) without duplicating the machinery;
    the defaults preserve CORAL's exact output (method='deep_coral', coral_fold{k}.npz)."""
    cache = ev.FeatureCache(cache_dir)
    if not cache.has("test_negatives_shared"):
        raise FileNotFoundError("test_negatives_shared.npz missing; run cache_features.py")
    neg = cache.load("test_negatives_shared")
    neg_feats, neg_ids = neg["features"], neg["ids"]
    folds = sorted(fold_ckpts)

    # per-(fold, ckpt): val metric (label-free selector) + test metrics (for ORACLE)
    sr_scores, or_scores = {}, {}          # fold -> (pos, neg) for the picked model
    sr_pick, or_pick = {}, {}              # fold -> {ckpt, epoch, val_metric}
    for k in folds:
        cands = fold_ckpts[k]
        rows = []
        for ck in cands:
            head, std, meta = ev.load_checkpoint(ck, device)
            vmet = _val_metric(head, std, cache, device, val_select_metric, target_fpr)
            pos = ev.score_features(head, std, cache.load(f"test_pos_fold{k}")["features"], device)
            negk = ev.score_features(head, std, neg_feats, device)
            y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(negk.size, np.int64)])
            s = np.concatenate([pos, negk])
            rows.append({"ckpt": ck, "epoch": meta["epoch"], "val": vmet, "pos": pos,
                         "neg": negk, "tpr": tpr_at_fpr(y, s, target_fpr),
                         "auroc": auroc(y, s)})
        # SOURCE-RISK: best source-val (fall back to last epoch if no val cached)
        if any(np.isfinite(r["val"]) for r in rows):
            sr = max(rows, key=lambda r: np.nan_to_num(r["val"], nan=-np.inf))
        else:
            sr = rows[-1]
        # ORACLE: best real-test (peeks at labels -> NOT a valid UDA result)
        orc = max(rows, key=lambda r: (np.nan_to_num(r["tpr"], nan=-np.inf),
                                       np.nan_to_num(r["auroc"], nan=-np.inf)))
        sr_scores[k] = (sr["pos"], sr["neg"]); or_scores[k] = (orc["pos"], orc["neg"])
        sr_pick[k] = {"ckpt": sr["ckpt"], "epoch": sr["epoch"], "val_metric": _num(sr["val"])}
        or_pick[k] = {"ckpt": orc["ckpt"], "epoch": orc["epoch"], "val_metric": _num(orc["val"])}

    sr_agg = _aggregate(sr_scores, target_fpr, B, seed)
    or_agg = _aggregate(or_scores, target_fpr, B, seed)

    results = {
        "method": method, "target_fpr": target_fpr, "folds": folds,
        "val_select_metric": val_select_metric,
        "source_risk": _pack(sr_pick, sr_agg),
        "oracle": _pack(or_pick, or_agg),
        "selection_gap": {
            "tpr": _num(or_agg["mean_tpr"] - sr_agg["mean_tpr"]),
            "auroc": _num(or_agg["mean_auroc"] - sr_agg["mean_auroc"]),
            "same_pick": all(sr_pick[k]["ckpt"] == or_pick[k]["ckpt"] for k in folds),
        },
    }

    if verbose:
        _report(results, sr_pick)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        # dump raw per-fold scores for the SOURCE-RISK models (paired bake-off compares)
        sdir = os.path.join(out_dir, "scores"); os.makedirs(sdir, exist_ok=True)
        for k in folds:
            pos, negk = sr_scores[k]
            np.savez(os.path.join(sdir, f"{score_prefix}_fold{k}.npz"),
                     pos_scores=pos, neg_ids=neg_ids, neg_scores=negk,
                     target_fpr=np.asarray(target_fpr))
        if verbose:
            print(f"\n[out] results.json + raw per-fold scores -> {out_dir}")
    return results


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------

def _num(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _pack(pick, agg):
    return {"per_fold": {int(k): {**pick[k], "n_pos": agg["per_fold"][k]["n_pos"],
                                  "tpr": _num(agg["per_fold"][k]["tpr"]),
                                  "auroc": _num(agg["per_fold"][k]["auroc"])}
                         for k in agg["per_fold"]},
            "mean_tpr": _num(agg["mean_tpr"]), "std_tpr": _num(agg["std_tpr"]),
            "mean_auroc": _num(agg["mean_auroc"]), "std_auroc": _num(agg["std_auroc"]),
            "mean_tpr_ci": [_num(agg["tpr_ci"]["lo"]), _num(agg["tpr_ci"]["hi"])],
            "mean_auroc_ci": [_num(agg["auroc_ci"]["lo"]), _num(agg["auroc_ci"]["hi"])]}


def _fmt(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


def _report_stability(stability: dict, folds):
    """Print the per-fold CORAL stability log (schedule + blow-up diagnostic)."""
    print("\n" + "=" * 80)
    print("CORAL STABILITY LOG  (gamma schedule + cov-term blow-up diagnostic)")
    print("=" * 80)
    for k in sorted(folds):
        st = stability.get(int(k)) or stability.get(str(k))
        if st is None:
            continue
        be = st["blowup_epoch"]
        base = "n/a" if st["baseline_cov"] is None else f"{st['baseline_cov']:.4g}"
        print(f"\n  [fold {k}]  schedule={st['schedule']}  peak_gamma={st['peak_gamma']:g}"
              f"  baseline_cov={base}  (x{st['blowup_factor']:g} threshold, "
              f"win={st['baseline_window_steps']} steps)")
        print(f"    blow-up epoch : {be if be is not None else 'none'}"
              f"    stable checkpoints saved: {st['n_saved_checkpoints']}")
        print("    epoch |  gamma_eff |    cov_term |  cov_ema  | blowup | saved")
        print("    " + "-" * 62)
        for e in st["epochs"]:
            ema = "   n/a  " if e["cov_ema"] is None else f"{e['cov_ema']:.3g}"
            print(f"    {e['epoch']:>5} | {e['gamma_eff']:>10.4f} | {e['cov_term']:>11.4g} | "
                  f"{ema:>8} | {'YES' if e['blowup'] else ' no':>6} | "
                  f"{'yes' if e['saved'] else ' no'}")
    n_blew = sum(1 for k in folds
                 if (stability.get(int(k)) or stability.get(str(k), {})).get("blowup_epoch")
                 is not None)
    print("\n" + "=" * 80)
    print(f"STABILITY SUMMARY: {n_blew}/{len(list(folds))} folds hit a blow-up stop.")
    print("=" * 80)


def _report(results, sr_pick):
    tf = results["target_fpr"]; folds = results["folds"]
    title = results.get("method", "deep_coral").replace("_", " ").upper()
    print("\n" + "=" * 80)
    print(f"{title}  (per-fold models; metric TPR@FPR={tf:g} + AUROC)")
    print("=" * 80)
    for label, block in (("SOURCE-RISK  (headline, label-free selection)", results["source_risk"]),
                         ("ORACLE  (best-test epoch per fold -- NOT a valid UDA result)",
                          results["oracle"])):
        print(f"\n[{label}]")
        print("    fold |  epoch |  n_pos |  val_sel |  TPR@fpr |   AUROC")
        print("    " + "-" * 56)
        for k in folds:
            pf = block["per_fold"][k] if k in block["per_fold"] else block["per_fold"][str(k)]
            print(f"    {k:>4} | {str(pf['epoch']):>6} | {pf['n_pos']:>6} | "
                  f"{_fmt(pf['val_metric']):>8} | {_fmt(pf['tpr']):>8} | {_fmt(pf['auroc']):>7}")
        print(f"    across-fold mean  TPR = {_fmt(block['mean_tpr'])} +- {_fmt(block['std_tpr'])}"
              f"   [95% CI {_fmt(block['mean_tpr_ci'][0])}, {_fmt(block['mean_tpr_ci'][1])}]")
        print(f"    across-fold mean  AUROC = {_fmt(block['mean_auroc'])} +- {_fmt(block['std_auroc'])}"
              f"   [95% CI {_fmt(block['mean_auroc_ci'][0])}, {_fmt(block['mean_auroc_ci'][1])}]")
    g = results["selection_gap"]
    print("\n" + "=" * 80)
    print(f"SELECTION GAP (oracle - source-risk):  TPR {_fmt(g['tpr'])}   AUROC {_fmt(g['auroc'])}"
          + ("   [same epoch picked every fold]" if g["same_pick"] else ""))
    print("=" * 80)


# ---------------------------------------------------------------------------
# the full CORAL run: train per fold -> evaluate per fold -> aggregate
# ---------------------------------------------------------------------------

def run_coral(cache_dir, out_dir, *, folds=(0, 1, 2, 3, 4), head_type=LINEAR,
              epochs=(1, 2, 5, 10, 20, 50), batch_size=256, pos_fraction=0.5,
              lr=1e-3, weight_decay=1e-4, steps_per_epoch=None, coral_weight=1.0,
              coral_include_mean=True, coral_shrinkage=0.1, coral_target_ema=True,
              coral_ema_momentum=0.9, coral_weight_schedule="constant",
              coral_warmup_frac=0.1, coral_min_frac=0.1, blowup_factor=5.0,
              blowup_baseline_steps=None, stop_on_blowup=True,
              seed=42, train_device="cpu", eval_device="cpu",
              target_fpr=DEFAULT_TARGET_FPR, bootstrap=1000, eval_seed=0,
              val_select_metric="auroc", verbose=True) -> dict:
    cache = ev.FeatureCache(cache_dir)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    fold_ckpts = {}
    fold_stability = {}
    if verbose:
        print(f"Deep CORAL  cache={cache_dir}  out={out_dir}  folds={list(folds)}  "
              f"gamma={coral_weight} schedule={coral_weight_schedule} "
              f"shrink={coral_shrinkage} ema={coral_target_ema} "
              f"stop_on_blowup={stop_on_blowup}")
    for k in folds:
        if verbose:
            print(f"\n[train fold {k}] ...")
        fold_ckpts[k], fold_stability[k] = train_coral_fold(
            cache, k, ckpt_dir, head_type=head_type, epochs=epochs, batch_size=batch_size,
            pos_fraction=pos_fraction, lr=lr, weight_decay=weight_decay,
            steps_per_epoch=steps_per_epoch, coral_weight=coral_weight,
            coral_include_mean=coral_include_mean, coral_shrinkage=coral_shrinkage,
            coral_target_ema=coral_target_ema, coral_ema_momentum=coral_ema_momentum,
            coral_weight_schedule=coral_weight_schedule, coral_warmup_frac=coral_warmup_frac,
            coral_min_frac=coral_min_frac, blowup_factor=blowup_factor,
            blowup_baseline_steps=blowup_baseline_steps, stop_on_blowup=stop_on_blowup,
            seed=seed, device=train_device, verbose=verbose)

    # a fold whose whole schedule blew up before any stable checkpoint would leave the
    # selector with nothing -- surface it loudly rather than silently dropping the fold.
    empty = [k for k in folds if not fold_ckpts[k]]
    if empty:
        raise RuntimeError(
            f"folds {empty} produced NO stable checkpoints (blew up before the first "
            f"scheduled epoch). Lower the peak gamma, widen the schedule's early epochs, "
            f"or relax blowup_factor -- refusing to report a fold with no stable model.")

    if verbose:
        print(f"\n[eval] per-fold models -> per-fold TPR@{target_fpr:g} + across-fold aggregate")
    results = evaluate_per_fold(
        fold_ckpts, cache_dir, device=eval_device, target_fpr=target_fpr,
        B=bootstrap, seed=eval_seed, val_select_metric=val_select_metric,
        out_dir=out_dir, verbose=verbose)

    # attach the per-fold stability diagnostic and re-write results.json so the blow-up
    # story travels with the numbers.
    results["coral_stability"] = {int(k): fold_stability[k] for k in fold_stability}
    if verbose:
        _report_stability(results["coral_stability"], folds)
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
    p = argparse.ArgumentParser(description="Deep CORAL -- per-fold DA runner (feature-space).")
    p.add_argument("--cache-dir", default=os.path.join(default_root, "feature_cache"))
    p.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "runs", "coral")))
    p.add_argument("--folds", default="0,1,2,3,4", help="comma-separated fold ids")
    p.add_argument("--head-type", default=LINEAR, choices=HEAD_TYPES)
    p.add_argument("--epochs", default="1,2,5,10,20,50",
                   help="comma-separated epoch schedule = SOURCE-RISK selection candidates")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pos-fraction", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--coral-weight", type=float, default=1.0,
                   help="gamma -- FIXED, not tuned on target (RLSbench fixed-budget)")
    p.add_argument("--coral-no-mean", action="store_true",
                   help="ablate the first-moment term (second-moment-only CORAL)")
    p.add_argument("--coral-shrinkage", type=float, default=0.1)
    p.add_argument("--coral-no-target-ema", action="store_true")
    p.add_argument("--coral-ema-momentum", type=float, default=0.9)
    # --- stability fix: gamma schedule + blow-up-aware early stop ---
    p.add_argument("--coral-weight-schedule", default="constant", choices=CORAL_SCHEDULES,
                   help="shape the FIXED peak gamma over steps (warmup_cosine recommended)")
    p.add_argument("--coral-warmup-frac", type=float, default=0.1,
                   help="fraction of total steps to ramp gamma 0->peak")
    p.add_argument("--coral-min-frac", type=float, default=0.1,
                   help="warmup_cosine floor: gamma decays toward min_frac*peak")
    p.add_argument("--blowup-factor", type=float, default=5.0,
                   help="cov-term > factor x early-window baseline => blow-up")
    p.add_argument("--blowup-baseline-steps", type=int, default=None,
                   help="healthy-window length in steps (default: one epoch)")
    p.add_argument("--no-stop-on-blowup", action="store_true",
                   help="log the blow-up diagnostic but do NOT stop / drop checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-device", default=None)
    p.add_argument("--eval-device", default="cpu")
    p.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--val-select-metric", default="auroc", choices=("auroc", "tpr"))
    args = p.parse_args()

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

    train_device = args.train_device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = parse_epochs(args.epochs)
    run_coral(
        args.cache_dir, args.out_dir, folds=folds, head_type=args.head_type, epochs=epochs,
        batch_size=args.batch_size, pos_fraction=args.pos_fraction, lr=args.lr,
        weight_decay=args.weight_decay, steps_per_epoch=args.steps_per_epoch,
        coral_weight=args.coral_weight, coral_include_mean=not args.coral_no_mean,
        coral_shrinkage=args.coral_shrinkage, coral_target_ema=not args.coral_no_target_ema,
        coral_ema_momentum=args.coral_ema_momentum,
        coral_weight_schedule=args.coral_weight_schedule,
        coral_warmup_frac=args.coral_warmup_frac, coral_min_frac=args.coral_min_frac,
        blowup_factor=args.blowup_factor, blowup_baseline_steps=args.blowup_baseline_steps,
        stop_on_blowup=not args.no_stop_on_blowup, seed=args.seed,
        train_device=train_device, eval_device=args.eval_device,
        target_fpr=args.target_fpr, bootstrap=args.bootstrap, eval_seed=args.eval_seed,
        val_select_metric=args.val_select_metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
