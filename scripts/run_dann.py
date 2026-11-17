"""
run_dann.py
===========
DANN (Domain-Adversarial Neural Network, Ganin & Lempitsky arXiv:1409.7495 /
arXiv:1505.07818) -- a FOLD-0-ONLY viability / CEILING probe for the label-free
sim-to-real strong-lens bake-off (Euclid Q1). It is the gradient-reversal counterpart
of run_coral.py / run_freematch.py and, like Baseline A and CORAL, reads CACHED frozen
Zoobot features -- the encoder is NEVER instantiated in the training loop (see the
"COMPUTE REGIME" note below). So this runs LOCAL and CHEAP (feature-space CPU/GPU), in
the same seconds-to-low-minutes-per-epoch regime as baseline/CORAL, NOT the image-space
~6 min/epoch FreeMatch regime.

WHY FOLD-0 ONLY (a probe, not a reportable 5-fold result)
---------------------------------------------------------
DANN / WDGRL were listed as candidates in PROJECT.md 7.4 but had NOT been run. This
runner answers the narrow question "is adversarial marginal alignment even VIABLE here,
and where does its fold-0 real-lens curve flatten?" via the SAME fold-0 learning-curve
diagnostic (curve.csv) that baseline/FreeMatch use. It is a screen that justifies (or
kills) a full 5-fold DANN run; it is not itself a reportable number.

  ****  THE ONE STRUCTURAL ASYMMETRY (method-intrinsic, disclosed)  ****
  DANN structurally REQUIRES a trainable feature extractor G_f for the gradient-reversal
  to push on. The linear-head methods (CORAL / AdaMatch / FreeMatch) do NOT have one --
  they align on their LINEAR head's post-LayerNorm features. So DANN, and ONLY DANN, is
  given a small BOTTLENECK ADAPTER as G_f:

        G_f:  Linear(640 -> d) -> ReLU -> Linear(d -> 640)   with an internal SKIP
              (residual) connection,   d = 64  FIXED.

  d=64 is FIXED and deliberately NOT exposed as a tunable flag: it must not be tuned to
  favour DANN. The adapter exists because DANN cannot work without a trainable G_f; the
  linear-head methods are deliberately NOT given this adapter, because adding a trainable
  adapter to a linear probe on good frozen features risks overfitting / degrading them
  (adapter-tuning literature, e.g. arXiv:1902.00751 and the project's own SSL
  rank-collapse finding, PROJECT.md 1). This asymmetry is method-intrinsic and is
  disclosed here rather than hidden -- DANN is NOT given a free capacity advantage as a
  "trick"; the adapter is the minimum structure the method's own math demands.

VANILLA DANN (no label-shift correction) -- justified
-----------------------------------------------------
The fold-0 target is BALANCED (391 lens : 391 neg, splits_v6; confirmed from the cache).
Because there is NO label shift in this setup by construction (PROJECT.md 4.3), we use
the ORIGINAL vanilla DANN objective, NOT the prior-reweighted / label-shift-corrected
variant (which would only matter under source/target class-ratio mismatch, GLS
arXiv:2003.04475). Using vanilla DANN here is the correct, un-embellished baseline of the
adversarial family.

THE MODEL (all on the standardized 640-d frozen features)
---------------------------------------------------------
  * G_f  bottleneck adapter (above)          -- the trainable feature extractor.
  * G_y  label head:  Linear(640 -> 1)       -- BCE on SOURCE labels only.
  * G_d  domain discriminator: 3 FC layers (1024, 1024, 1), BatchNorm+ReLU BETWEEN the
         FC layers, fed the adapter output THROUGH a Gradient Reversal Layer (GRL).
  * GRL  multiplies the gradient by  -lambda  in the backward pass, identity forward.

LAMBDA SCHEDULE (fixed by formula, NOT tuned; PROJECT.md 7.4 fixed-budget protocol)
-----------------------------------------------------------------------------------
    lambda(p) = 2 / (1 + exp(-10 * p)) - 1 ,   p = global_step / total_steps
The canonical DANN anneal 0 -> 1 (arXiv:1505.07818 eq. for the GRL weight). It is a FIXED
formula, never tuned on the target; the effective lambda per step is logged so the anneal
is auditable.

OPTIMIZER
---------
SGD, momentum 0.9 (canonical DANN), at the shared source-tuned LR (default 1e-3 -- the
same LR VALUE baseline/CORAL use, kept identical for cross-method comparability). NOTE /
FLAG: baseline/CORAL's optimizer is AdamW at that LR, not SGD -- the request pins SGD+
momentum for DANN (the method's canonical optimizer), so the "same optimizer family"
holds only at the LR value, not the update rule. This is stated, not silently chosen.

CURVE-HOOK DIAGNOSTIC (the deliverable) -- schema reused, scoring is DANN-aware
-------------------------------------------------------------------------------
``--eval-every-steps N`` writes runs/<out>/curve.csv with EXACTLY curve_hook's columns
    step, source_risk_auroc, source_risk_tpr, fold0_tpr_1e-3, fold0_auprc, fold0_auroc
and saves checkpoints/step_*.pt -- byte-identical schema to the baseline/FreeMatch curve.
We REUSE curve_hook.CURVE_COLUMNS + curve_hook._cached_split + curve_hook._num + the
metrics.py functions verbatim (no metric re-implemented, same fold-0-only splits: val +
test_pos_fold0 + test_negatives_shared). The ONLY thing that cannot be reused verbatim is
evaluate.load_checkpoint / score_features: those rebuild a heads.ClassifierHead and score
head(std(x)), but DANN's inference network is std -> G_f adapter -> G_y label head, which
is NOT a ClassifierHead. So scoring is DANN-aware here (it must be), while everything else
-- the schema, the splits, the math -- is the shared machinery. evaluate.py / curve_hook.py
/ metrics.py are untouched (read-only elsewhere).

  ****  LEAKAGE CONTRACT (asserted below)  ****
  Training reads ONLY source_train, val (source-domain SIM), and target_fold0. The target's
  served labels are UNLABELED(-1) and are asserted so EVERY step (the domain discriminator
  uses only the DOMAIN identity source-vs-target, never the target's class labels). The
  fold-0 test positives + shared negatives are touched ONLY by the fold-0 curve/eval scorer,
  never for training or selection.

Run (after cache_features.py has populated the cache):
    # smoke + timing probe (STOP here for confirmation before the full run):
    python scripts\\run_dann.py --epochs 2 --eval-every-steps 200
    # full fold-0 curve:
    python scripts\\run_dann.py --epochs 50 --eval-every-steps 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Reused verbatim -- same standardizer + checkpoint-adjacent conventions, same balanced
# sampler + label-free guard, same metrics, same curve.csv schema -- so the DANN fold-0
# probe is comparable to Baseline A / CORAL / FreeMatch.
from heads import FeatureStandardizer, EMBED_DIM
from samplers import ClassBalancedBatchSampler, assert_target_unlabeled, POS_LABEL, NEG_LABEL
from metrics import tpr_at_fpr, auroc, metric_suite, DEFAULT_TARGET_FPR
from run_baseline import parse_epochs, set_seed, _cf
import evaluate as ev
import curve_hook

# fold-0 probe: this runner only ever reads / trains on fold 0.
FOLD = 0
BOTTLENECK_DIM = 64            # d for G_f -- FIXED (must not be tuned to favour DANN)
HEAD_TAG = "dann"             # checkpoint tag; evaluate._epoch_from_name still finds epNNN

# Training may read ONLY these cache splits (runtime half of the leakage contract).
TRAIN_ALLOWED = ("source_train", "val", f"target_fold{FOLD}")


def _guarded_load(cache: ev.FeatureCache, name: str) -> dict:
    assert name in TRAIN_ALLOWED, (
        f"run_dann training may only read {TRAIN_ALLOWED}; refusing to load '{name}' "
        f"(the fold-0 test splits are the eval scorer's job, not training's).")
    if not cache.has(name):
        raise FileNotFoundError(
            f"required cache split '{name}.npz' not found under {cache.dir}; "
            f"run cache_features.py first.")
    return cache.load(name)


# ===========================================================================
# GRADIENT REVERSAL LAYER  (identity forward; gradient * -lambda backward)
# ===========================================================================

class GradientReversalFn(torch.autograd.Function):
    """Ganin's GRL: forward is the identity, backward multiplies the incoming gradient by
    ``-lambda``. Placed between G_f (adapter) and G_d (discriminator): the discriminator
    minimises the domain BCE normally, but the adapter receives the SIGN-FLIPPED gradient,
    so it learns DOMAIN-INVARIANT features (it is trained to CONFUSE the discriminator)."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)                      # identity (a fresh view, so grad flows)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None    # -lambda * grad ; None for lambda_


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambda_)


def dann_lambda_at_step(step: int, total_steps: int) -> float:
    """The canonical DANN GRL weight anneal (arXiv:1505.07818):
        lambda(p) = 2/(1+exp(-10 p)) - 1 ,  p = step/total_steps in [0,1].
    lambda(0)=0 (no adversary while the label head is still random) rising smoothly to ~1.
    FIXED formula -- never tuned on the target."""
    p = min(max(step / max(int(total_steps), 1), 0.0), 1.0)
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ===========================================================================
# THE DANN MODULES (all operate on the standardized 640-d frozen features)
# ===========================================================================

class BottleneckAdapter(nn.Module):
    """G_f -- the trainable feature extractor DANN structurally requires.

        Linear(dim -> bottleneck) -> ReLU -> Linear(bottleneck -> dim),  + internal skip.

    The residual skip (``x + up(relu(down(x)))``) keeps the map near identity at init so
    the frozen features are not destroyed before the label head can use them. ``bottleneck``
    is FIXED at 64 (module-level constant) -- deliberately not a tunable knob."""

    def __init__(self, dim: int = EMBED_DIM, bottleneck: int = BOTTLENECK_DIM):
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))       # internal skip connection


class DomainDiscriminator(nn.Module):
    """G_d -- 3 FC layers (1024, 1024, 1) with BatchNorm+ReLU BETWEEN the FC layers, fed
    the adapter output through the GRL. One logit: P(sample is TARGET). BatchNorm is on the
    concatenated source+target batch (standard DANN); it lives ONLY in the training graph
    (never used at inference / scoring)."""

    def __init__(self, dim: int = EMBED_DIM, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                               # (B, 1) domain logit


class DANNModel(nn.Module):
    """The INFERENCE network: adapter G_f -> label head G_y. (G_d is separate -- it exists
    only for the adversarial training term, never for scoring.) ``forward`` expects features
    that are ALREADY standardized (the FeatureStandardizer is applied outside, exactly as in
    baseline/CORAL, and is saved alongside so eval standardizes identically)."""

    def __init__(self, dim: int = EMBED_DIM, bottleneck: int = BOTTLENECK_DIM):
        super().__init__()
        self.adapter = BottleneckAdapter(dim, bottleneck)
        self.label_head = nn.Linear(dim, 1)              # G_y

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)                           # adapter output (G_d / G_y input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.label_head(self.adapter(x))          # (B, 1) label logit


# ===========================================================================
# checkpoint save / load / score  (DANN-aware; NOT a heads.ClassifierHead)
# ===========================================================================

def ckpt_name(epoch: int, width: int) -> str:
    """e.g. dann_fold0_ep050.pt -- evaluate._epoch_from_name() still recovers the epoch via
    '(?:ep|epoch)[_-]?(\\d+)' if anything downstream wants it."""
    return f"{HEAD_TAG}_fold{FOLD}_ep{epoch:0{width}d}.pt"


def save_dann_checkpoint(path, model: DANNModel, standardizer: FeatureStandardizer,
                         epoch: int, cfg_extra: dict):
    """Serialize the DANN INFERENCE model (adapter + label head) + the frozen standardizer.
    The discriminator is intentionally NOT saved -- it plays no role in scoring. A distinct
    ``method='dann'`` / ``head_type='dann_adapter'`` tag makes these checkpoints clearly not
    heads.ClassifierHead checkpoints (so nothing accidentally loads them with build_head)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cfg = {"head_type": "dann_adapter", "method": "dann", "fold": int(FOLD),
           "epoch": int(epoch), "bottleneck": int(BOTTLENECK_DIM), **cfg_extra}
    torch.save({
        "method": "dann",
        "head_type": "dann_adapter",
        "adapter": model.adapter.state_dict(),
        "label_head": model.label_head.state_dict(),
        "standardizer": standardizer.state_dict(),
        "cfg": cfg,
    }, path)


def load_dann_checkpoint(path: str, device: str = "cpu"):
    """Reconstruct (DANNModel, standardizer, meta) from a run_dann checkpoint. The DANN
    analogue of evaluate.load_checkpoint (which only knows heads.ClassifierHead)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("method") != "dann":
        raise ValueError(f"{path} is not a DANN checkpoint (method={ckpt.get('method')!r})")
    model = DANNModel(EMBED_DIM, int(ckpt.get("cfg", {}).get("bottleneck", BOTTLENECK_DIM)))
    model.adapter.load_state_dict(ckpt["adapter"])
    model.label_head.load_state_dict(ckpt["label_head"])
    model.eval().to(device)
    std = FeatureStandardizer(EMBED_DIM)
    std.load_state_dict(ckpt["standardizer"])
    std.eval().to(device)
    cfg = ckpt.get("cfg", {}) or {}
    meta = {"path": path, "method": "dann", "fold": cfg.get("fold"),
            "epoch": ev._epoch_from_name(path)}
    return model, std, meta


@torch.no_grad()
def score_dann(model: DANNModel, std: FeatureStandardizer, feats: np.ndarray,
               device: str = "cpu") -> np.ndarray:
    """Label-head LOGIT per row (monotonic in probability -> a valid ranking score for
    TPR@FPR / AUROC). The DANN analogue of evaluate.score_features."""
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(device)
    logits = model(std(x)).view(-1)
    return logits.detach().cpu().numpy()


# ===========================================================================
# fold-0 curve row  (SAME schema + splits + math as curve_hook; DANN scoring)
# ===========================================================================

def eval_dann_fold0(model, std, cache: ev.FeatureCache, *,
                    target_fpr: float = DEFAULT_TARGET_FPR, device: str = "cpu") -> dict:
    """The DANN counterpart of curve_hook.eval_checkpoint_fold0: score source-val (both
    label-free selectors) + fold-0 test, returning curve_hook's five metric columns. Reads
    ONLY val + test_pos_fold0 + test_negatives_shared (memoized via curve_hook._cached_split
    so a long curve does not re-read the 26k negatives every point). Every metric is a
    metrics.py function -- nothing is re-implemented here."""
    model.eval()

    # source_risk: label-free selector on source-domain SIM val ONLY (log BOTH ways)
    v = curve_hook._cached_split(cache, "val")
    vs = score_dann(model, std, v["features"], device)
    source_risk_auroc = auroc(v["labels"], vs)
    source_risk_tpr = tpr_at_fpr(v["labels"], vs, target_fpr)

    # fold-0 diagnostic: held-out real positives vs the SHARED negatives
    pos = curve_hook._cached_split(cache, f"test_pos_fold{FOLD}")
    neg = curve_hook._cached_split(cache, "test_negatives_shared")
    ps = score_dann(model, std, pos["features"], device)
    ns = score_dann(model, std, neg["features"], device)
    y = np.concatenate([np.ones(ps.size, dtype=np.int64), np.zeros(ns.size, dtype=np.int64)])
    s = np.concatenate([ps, ns])

    return {"source_risk_auroc": curve_hook._num(source_risk_auroc),
            "source_risk_tpr": curve_hook._num(source_risk_tpr),
            "fold0_tpr_1e-3": curve_hook._num(tpr_at_fpr(y, s, target_fpr)),
            "fold0_auprc": curve_hook._num(metric_suite(y, s, target_fpr=target_fpr)["auprc"]),
            "fold0_auroc": curve_hook._num(auroc(y, s))}


def append_dann_curve_row(curve_csv, step, ckpt_path, cache, *,
                          target_fpr=DEFAULT_TARGET_FPR, device="cpu") -> dict:
    """Compute the fold-0 curve row for the checkpoint at ``ckpt_path`` and append it to
    ``curve_csv`` in curve_hook's EXACT column order (header first if the file is new). Like
    the baseline/FreeMatch hook, it scores the JUST-SAVED checkpoint loaded on ``device`` --
    so the curve eval device is independent of the (possibly CUDA) training device."""
    model, std, _ = load_dann_checkpoint(ckpt_path, device)
    m = eval_dann_fold0(model, std, cache, target_fpr=target_fpr, device=device)
    row = {"step": int(step), **m}
    os.makedirs(os.path.dirname(os.path.abspath(curve_csv)), exist_ok=True)
    write_header = not os.path.exists(curve_csv)
    with open(curve_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=curve_hook.CURVE_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row[k] for k in curve_hook.CURVE_COLUMNS})
    return row


# ===========================================================================
# the fold-0 DANN training run  (feature-space; encoder NEVER in the loop)
# ===========================================================================

def train_dann_fold0(cache_dir, ckpt_dir, *, epochs=(1, 2, 5, 10, 20, 50),
                     batch_size=256, pos_fraction=0.5, lr=1e-3, weight_decay=1e-4,
                     momentum=0.9, steps_per_epoch=None, disc_hidden=1024, seed=42,
                     device="cpu", eval_every_steps=None, curve_csv=None,
                     eval_target_fpr=DEFAULT_TARGET_FPR, eval_device="cpu",
                     verbose=True):
    """Train G_f + G_y (label BCE on source) + G_d (domain BCE via GRL) on the CACHED
    fold-0 features, dumping epoch checkpoints (+ optional per-step curve checkpoints).
    Reads ONLY source_train / val / target_fold0. Returns (checkpoints, log)."""
    set_seed(seed)
    dev = torch.device(device)
    cache = ev.FeatureCache(cache_dir)

    # ---- source (labeled): the supervised signal for G_y ----
    src = _guarded_load(cache, "source_train")
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
    x_src = standardizer(src_feats).detach()             # standardized source (constant)

    # ---- target (UNLABELED, fold-0 pool): the domain-alignment domain ----
    tgt = _guarded_load(cache, f"target_fold{FOLD}")
    tgt_feats = torch.from_numpy(np.asarray(tgt["features"], dtype=np.float32)).to(dev)
    tgt_served = np.asarray(tgt["labels"], dtype=np.int64)          # must be all UNLABELED
    tgt_latent = np.asarray(tgt["latent_labels"], dtype=np.int64)   # 0/1 identity (balancing)
    assert np.all(tgt_served == -1), (
        f"fold-{FOLD} target labels must all be UNLABELED(-1); got "
        f"{np.unique(tgt_served).tolist()} -- the discriminator uses DOMAIN identity only.")
    x_tgt = standardizer(tgt_feats).detach()             # SAME source-fit standardizer
    tpos_idx = np.where(tgt_latent == POS_LABEL)[0]
    tneg_idx = np.where(tgt_latent == NEG_LABEL)[0]

    # ---- model + discriminator + ONE optimizer over ALL trainable params ----
    model = DANNModel(EMBED_DIM, BOTTLENECK_DIM).to(dev)
    disc = DomainDiscriminator(EMBED_DIM, disc_hidden).to(dev)
    params = list(model.parameters()) + list(disc.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    # ---- class-balanced batches: SOURCE on real labels, TARGET on latent identity ----
    src_sampler = ClassBalancedBatchSampler(
        pos_idx, neg_idx, batch_size=batch_size, num_batches=steps_per_epoch,
        pos_fraction=pos_fraction, seed=seed)
    steps = len(src_sampler)
    tgt_sampler = ClassBalancedBatchSampler(
        tpos_idx, tneg_idx, batch_size=batch_size, num_batches=steps,
        pos_fraction=pos_fraction, seed=seed + 1000)

    schedule = sorted(set(int(e) for e in epochs))
    max_epoch = schedule[-1]
    width = max(2, len(str(max_epoch)))
    total_steps = steps * max_epoch                      # lambda anneal spans the PLANNED run
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg_extra = {"encoder_mode": "frozen", "optimizer": "sgd", "lr": lr, "momentum": momentum,
                 "weight_decay": weight_decay, "batch_size": batch_size,
                 "pos_fraction": pos_fraction, "seed": seed, "steps_per_epoch": steps,
                 "disc_hidden": disc_hidden, "lambda_schedule": "dann_2sigmoid_10p",
                 "total_steps": total_steps}

    if verbose:
        print(f"  [fold {FOLD}] DANN  G_f=640->{BOTTLENECK_DIM}->640(+skip)  "
              f"G_d=[{disc_hidden},{disc_hidden},1]  {steps} steps/epoch  batch={batch_size}"
              f" (pos_frac~{src_sampler.actual_pos_fraction:.2f})  "
              f"src={pos_idx.size}+{neg_idx.size}  tgt={tpos_idx.size}+{tneg_idx.size}  "
              f"SGD(lr={lr},mom={momentum},wd={weight_decay})  total_steps={total_steps}",
              flush=True)

    checkpoints: list[str] = []
    epoch_logs: list[dict] = []
    global_step = 0
    last_lambda = 0.0
    for epoch in range(1, max_epoch + 1):
        src_sampler.set_epoch(epoch)
        tgt_sampler.set_epoch(epoch)
        model.train(); disc.train()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_epoch = time.time()
        last_lab = last_dom = 0.0
        for sbatch, tbatch in zip(src_sampler, tgt_sampler):
            sidx = torch.as_tensor(sbatch, dtype=torch.long, device=dev)
            tidx = torch.as_tensor(tbatch, dtype=torch.long, device=dev)
            # label-free guard, EVERY step: the target's served labels stay UNLABELED.
            assert_target_unlabeled({"label": torch.as_tensor(tgt_served[tbatch])})

            lam = dann_lambda_at_step(global_step, total_steps)   # FIXED anneal
            opt.zero_grad(set_to_none=True)

            f_src = model.features(x_src[sidx])          # G_f(source)
            f_tgt = model.features(x_tgt[tidx])          # G_f(target)

            # --- label loss: G_y on SOURCE only (normal gradient) ---
            label_loss = bce(model.label_head(f_src), y[sidx])

            # --- domain loss: G_d on GRL(G_f) of BOTH domains; domain id source=0,target=1 ---
            feat_all = torch.cat([f_src, f_tgt], dim=0)
            dom_logits = disc(grad_reverse(feat_all, lam))
            dom_labels = torch.cat([torch.zeros(f_src.shape[0], 1, device=dev),
                                    torch.ones(f_tgt.shape[0], 1, device=dev)], dim=0)
            domain_loss = bce(dom_logits, dom_labels)

            loss = label_loss + domain_loss
            loss.backward()
            opt.step()
            last_lab = float(label_loss.detach()); last_dom = float(domain_loss.detach())
            last_lambda = lam
            global_step += 1

            # curve hook (behind --eval-every-steps; default runs never enter here):
            # save the DANN inference model at this step + append one fold-0 curve row.
            if eval_every_steps and curve_csv and (global_step % int(eval_every_steps) == 0):
                spath = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                save_dann_checkpoint(spath, model, standardizer, epoch, cfg_extra)
                row = append_dann_curve_row(curve_csv, global_step, spath, cache,
                                            target_fpr=eval_target_fpr, device=eval_device)
                if verbose:
                    print(f"      [step {global_step:>6}] lam={lam:.3f} curve  "
                          f"val_auroc={_cf(row['source_risk_auroc'])}"
                          f" val_tpr={_cf(row['source_risk_tpr'])}"
                          f" | fold0 tpr@1e-3={_cf(row['fold0_tpr_1e-3'])}"
                          f" auprc={_cf(row['fold0_auprc'])}"
                          f" auroc={_cf(row['fold0_auroc'])}", flush=True)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        epoch_secs = time.time() - t_epoch

        saved = epoch in schedule
        if saved:
            path = os.path.join(ckpt_dir, ckpt_name(epoch, width))
            save_dann_checkpoint(path, model, standardizer, epoch, cfg_extra)
            checkpoints.append(path)
        epoch_logs.append({"epoch": epoch, "label_loss": last_lab, "domain_loss": last_dom,
                           "lambda_eff": float(last_lambda), "secs": float(epoch_secs),
                           "saved": bool(saved)})
        if verbose:
            print(f"    [ep {epoch:>{width}}] {epoch_secs:6.1f}s lam={last_lambda:.3f} "
                  f"label_bce={last_lab:.4f} domain_bce={last_dom:.4f}"
                  + (f"  saved {os.path.basename(path)}" if saved else ""), flush=True)

    log = {"fold": int(FOLD), "optimizer": "sgd", "lr": float(lr), "momentum": float(momentum),
           "weight_decay": float(weight_decay), "bottleneck": int(BOTTLENECK_DIM),
           "disc_hidden": int(disc_hidden), "steps_per_epoch": int(steps),
           "total_steps": int(total_steps), "lambda_schedule": "dann_2sigmoid_10p",
           "n_saved_checkpoints": len(checkpoints), "epochs": epoch_logs}
    return checkpoints, log


# ===========================================================================
# full fold-0 DANN probe: train -> pick source-risk ckpt -> fold-0 report + scores dump
# ===========================================================================

def _num(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def run_dann(cache_dir, out_dir, *, epochs=(1, 2, 5, 10, 20, 50), batch_size=256,
             pos_fraction=0.5, lr=1e-3, weight_decay=1e-4, momentum=0.9,
             steps_per_epoch=None, disc_hidden=1024, seed=42, train_device="cpu",
             eval_device="cpu", target_fpr=DEFAULT_TARGET_FPR, eval_every_steps=None,
             verbose=True) -> dict:
    """Fold-0 DANN probe end-to-end: train (with the optional per-step curve), then pick the
    SOURCE-RISK checkpoint (best source-val AUROC, label-free) among the saved epoch
    checkpoints, report its fold-0 metrics + an ORACLE upper bound (best fold-0 epoch), and
    dump the selected checkpoint's fold-0 scores for later paired comparison."""
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    curve_csv = os.path.join(out_dir, "curve.csv") if eval_every_steps else None
    # FRESH curve per run: the curve hook APPENDS, so a stale curve.csv / stale step_*.pt
    # from a previous run into the same dir would accumulate duplicate rows and pollute the
    # selection pool. Clear them so each run's curve.csv is exactly this run's points.
    if curve_csv and os.path.exists(curve_csv):
        os.remove(curve_csv)
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.startswith("step_") and f.endswith(".pt"):
                os.remove(os.path.join(ckpt_dir, f))
    if verbose:
        print(f"DANN fold-{FOLD} probe  cache={cache_dir}  out={out_dir}  "
              f"vanilla (balanced target -> no label-shift correction)")
    checkpoints, log = train_dann_fold0(
        cache_dir, ckpt_dir, epochs=epochs, batch_size=batch_size, pos_fraction=pos_fraction,
        lr=lr, weight_decay=weight_decay, momentum=momentum, steps_per_epoch=steps_per_epoch,
        disc_hidden=disc_hidden, seed=seed, device=train_device, eval_every_steps=eval_every_steps,
        curve_csv=curve_csv, eval_target_fpr=target_fpr, eval_device=eval_device, verbose=verbose)

    # ---- build the label-free SELECTION POOL (fold-0 only) ----------------
    # Prefer the per-STEP curve checkpoints: they are the whole learning curve, so the
    # SOURCE-RISK selector actually gets a choice (selecting only among the sparse epoch
    # schedule -- e.g. a single ep80 for `--epochs 80` -- would make source-risk==oracle
    # trivially and hide the real selection gap). Metrics come straight from curve.csv
    # (already computed by the hook -- no recompute); we reload ONLY the selected
    # checkpoint to dump its scores. Fall back to epoch checkpoints if the curve is off.
    cache = ev.FeatureCache(cache_dir)
    neg = cache.load("test_negatives_shared")
    pos0 = cache.load(f"test_pos_fold{FOLD}")
    cand = []                                        # {ckpt, tag, <curve metric cols>}
    if curve_csv and os.path.exists(curve_csv):
        with open(curve_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                step = int(r["step"])
                sp = os.path.join(ckpt_dir, f"step_{step}.pt")
                if not os.path.exists(sp):
                    continue
                m = {k: (float(r[k]) if r[k] not in ("", None) else None)
                     for k in curve_hook.CURVE_COLUMNS if k != "step"}
                cand.append({"ckpt": sp, "tag": f"step{step}", "step": step, **m})
        pool = "step"
    else:                                            # no curve -> epoch checkpoints
        for ck in checkpoints:
            model, std, meta = load_dann_checkpoint(ck, eval_device)
            m = eval_dann_fold0(model, std, cache, target_fpr=target_fpr, device=eval_device)
            cand.append({"ckpt": ck, "tag": f"ep{meta['epoch']}", "epoch": meta["epoch"], **m})
        pool = "epoch"

    results = {"method": "dann", "fold": FOLD, "target_fpr": target_fpr,
               "vanilla_no_label_shift_correction": True, "selection_pool": pool,
               "n_candidates": len(cand), "train_log": log,
               "candidates": [{k: c.get(k) for k in
                   ("tag", "source_risk_auroc", "source_risk_tpr",
                    "fold0_tpr_1e-3", "fold0_auprc", "fold0_auroc")} for c in cand]}
    if cand:
        # SOURCE-RISK: best label-free source-val AUROC (the reportable, label-free pick).
        sr = max(cand, key=lambda c: (c["source_risk_auroc"]
                                      if c["source_risk_auroc"] is not None else -np.inf))
        # ORACLE: best fold-0 TPR (peeks at real labels -> upper bound, NOT a valid result).
        orc = max(cand, key=lambda c: (c["fold0_tpr_1e-3"]
                                       if c["fold0_tpr_1e-3"] is not None else -np.inf))
        keys = ("tag", "source_risk_auroc", "source_risk_tpr",
                "fold0_tpr_1e-3", "fold0_auprc", "fold0_auroc")
        results["source_risk"] = {k: sr.get(k) for k in keys}
        results["oracle"] = {k: orc.get(k) for k in keys}
        results["selection_gap"] = {
            "tpr": _num((orc["fold0_tpr_1e-3"] or 0.0) - (sr["fold0_tpr_1e-3"] or 0.0)),
            "auroc": _num((orc["fold0_auroc"] or 0.0) - (sr["fold0_auroc"] or 0.0)),
            "same_pick": sr["ckpt"] == orc["ckpt"]}
        # dump the SOURCE-RISK model's fold-0 scores (paired bake-off comparison later)
        if out_dir:
            model, std, _ = load_dann_checkpoint(sr["ckpt"], eval_device)
            ps = score_dann(model, std, pos0["features"], eval_device)
            ns = score_dann(model, std, neg["features"], eval_device)
            sdir = os.path.join(out_dir, "scores"); os.makedirs(sdir, exist_ok=True)
            np.savez(os.path.join(sdir, f"dann_fold{FOLD}.npz"),
                     pos_scores=ps, neg_scores=ns, neg_ids=neg["ids"],
                     target_fpr=np.asarray(target_fpr))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)

    if verbose and cand:
        sr, orc, g = results["source_risk"], results["oracle"], results["selection_gap"]
        print("\n" + "=" * 80)
        print(f"DANN fold-{FOLD} PROBE  (metric TPR@FPR={target_fpr:g}; fold-0 only; "
              f"{len(cand)} {pool} candidates)")
        print("=" * 80)
        print(f"  SOURCE-RISK (label-free, best source-val AUROC): {sr['tag']}  "
              f"TPR@1e-3={_cf(sr['fold0_tpr_1e-3'])}  AUPRC={_cf(sr['fold0_auprc'])}  "
              f"AUROC={_cf(sr['fold0_auroc'])}")
        print(f"  ORACLE (best fold-0 TPR -- NOT a valid UDA result): {orc['tag']}  "
              f"TPR@1e-3={_cf(orc['fold0_tpr_1e-3'])}  AUROC={_cf(orc['fold0_auroc'])}")
        print(f"  SELECTION GAP (oracle - source-risk):  TPR {_cf(g['tpr'])}  "
              f"AUROC {_cf(g['auroc'])}"
              + ("   [same checkpoint]" if g["same_pick"] else "   [DIFFERENT checkpoints]"))
        if curve_csv:
            print(f"  curve -> {curve_csv}")
        print("=" * 80)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.normpath(os.path.join(here, "..", "data"))
    p = argparse.ArgumentParser(
        description="DANN (gradient-reversal) fold-0 viability probe (feature-space).")
    p.add_argument("--cache-dir", default=os.path.join(default_root, "feature_cache"))
    p.add_argument("--out-dir",
                   default=os.path.normpath(os.path.join(here, "..", "runs", "curve_dann_fold0")))
    p.add_argument("--epochs", default="1,2,5,10,20,50",
                   help="epoch budget/schedule (comma-separated); 187 steps/epoch convention")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pos-fraction", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="shared source-tuned LR (baseline/CORAL value; SGD here per DANN)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (canonical DANN)")
    p.add_argument("--disc-hidden", type=int, default=1024,
                   help="G_d FC width (3 layers: hidden,hidden,1) -- fixed structural choice")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="default: one nominal pass over the source pool (~187 at batch 256)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-device", default=None, help="default: cuda if available else cpu")
    p.add_argument("--eval-device", default="cpu", help="fold-0 scoring on cached features (CPU)")
    p.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    p.add_argument("--eval-every-steps", type=int, default=None,
                   help="if set, every N optimizer steps save checkpoints/step_N.pt and append "
                        "a fold-0 row to <out-dir>/curve.csv (the learning-curve diagnostic)")
    args = p.parse_args()

    if not os.path.isdir(args.cache_dir):
        print(f"SKIP: feature cache dir not found: {args.cache_dir} (run cache_features.py first)")
        return 3
    cache = ev.FeatureCache(args.cache_dir)
    for need in ("source_train", "val", f"target_fold{FOLD}",
                 f"test_pos_fold{FOLD}", "test_negatives_shared"):
        if not cache.has(need):
            print(f"SKIP: cache missing {need}.npz (run cache_features.py first)")
            return 3

    train_device = args.train_device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = parse_epochs(args.epochs)
    run_dann(
        args.cache_dir, args.out_dir, epochs=epochs, batch_size=args.batch_size,
        pos_fraction=args.pos_fraction, lr=args.lr, weight_decay=args.weight_decay,
        momentum=args.momentum, steps_per_epoch=args.steps_per_epoch,
        disc_hidden=args.disc_hidden, seed=args.seed, train_device=train_device,
        eval_device=args.eval_device, target_fpr=args.target_fpr,
        eval_every_steps=args.eval_every_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
