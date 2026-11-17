"""
train_skeleton.py
=================
Training-loop SKELETON for the label-free sim-to-real strong-lens pipeline
(Euclid Q1). Its ONLY job is to prove the plumbing runs correctly end-to-end:
frozen encoder -> standardizer -> head -> supervised loss + an inert DA hook,
with the optimizer touching the head/standardizer and NEVER the encoder.

Scope discipline (read this before adding anything):
  * NO real DA loss here. The DA term is a clearly-marked inert STUB (exactly 0)
    that still exercises the head-feature graph, so shapes/devices/grad-routing are
    proven. DANN / AdaMatch / CORAL plug into the marked hook later, unchanged.
  * NO accuracy tuning. Hyperparameters are placeholders.

Grounded in PROJECT.md (tags: RB=research-backed, EC=engineering-choice):

  * FROZEN ENCODER (RB) -- ``ZoobotEncoder(encoder_mode="frozen")`` runs under no_grad
    and returns DETACHED (B,640) features. Because the head sees detached features,
    the DA loss on head/adapter features provably cannot backprop into the encoder --
    that back-prop is the representation collapse we already measured. The
    ``encoder_mode`` switch is respected: "frozen" is implemented; the partial-unfreeze
    modes ("lp_then_lastblock" / "lp_then_lora") raise NotImplementedError at a single
    clean hook so they drop in later without a rewrite.

  * STANDARDIZER (RB) -- fit on SOURCE_TRAIN once (streaming no_grad pass), frozen,
    applied to every role before the head. Saved/loaded WITH the head checkpoint.

  * SOURCE-DOMAIN VALIDATION (RB) -- role="val", aug="none": labelled held-out SOURCE,
    for model selection / early-stopping. It never sees the target domain or real
    lenses (the only label-free-legal selector; PROJECT.md 4.2).

  * OPTIMIZER SCOPE (EC) -- AdamW over head params + any future standardizer params,
    explicitly NOT the encoder. Fixed seed. AMP optional (GPU only).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from lens_dataset import LensDataset
from zoobot_encoder import ZoobotEncoder, EMBED_DIM
from heads import build_head, FeatureStandardizer, LINEAR
from samplers import (PairedDomainLoader, make_balanced_loader,
                      assert_target_unlabeled)
from da_losses import CoralLoss, AdaMatchLoss, FreeMatchLoss

FROZEN = "frozen"
FUTURE_ENCODER_MODES = ("lp_then_lastblock", "lp_then_lora")


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# encoder-mode-aware feature extraction (the partial-unfreeze hook)
# ---------------------------------------------------------------------------

def encode_features(encoder: ZoobotEncoder, images: torch.Tensor,
                    encoder_mode: str = FROZEN) -> torch.Tensor:
    """Return (B,640) encoder features under the given ``encoder_mode``.

    The ENCODER owns the frozen/trainable decision (via its ``n_blocks``), not this
    function -- so this stays a pure dispatch and there is exactly one place that decides
    whether gradients flow:

      * ``encoder.n_blocks == 0`` -- the encoder runs under no_grad and returns DETACHED
        features, so the head/DA graph starts clean above the encoder (unchanged behaviour).
      * ``encoder.n_blocks > 0``  -- the top n_blocks stages are trainable, so the returned
        features CARRY GRAD and backprop reaches those stages.

    ****  OPEN ITEM, flagged not silently decided (PROJECT.md 7.4 / Lever-1 scope)  ****
    At n_blocks>0 the features carry grad, so a DA loss placed on head/adapter features
    would ALSO backprop into the unfrozen encoder stages -- which is a genuine change in
    what the DA methods optimise, not just a speed knob. Lever 1 is deliberately screened
    with Baseline A (source-only, NO DA loss), so nothing here changes any DA method's
    behaviour today. Combining n_blocks>0 with a DA loss is a SEPARATE decision that must
    be made (and measured) explicitly.
    """
    if encoder_mode == FROZEN:
        return encoder(images)
    if encoder_mode in FUTURE_ENCODER_MODES:
        raise NotImplementedError(
            f"encoder_mode={encoder_mode!r} (partial unfreeze) is a reserved hook; "
            f"only {FROZEN!r} is implemented in the skeleton.")
    raise ValueError(f"unknown encoder_mode {encoder_mode!r}")


# ---------------------------------------------------------------------------
# LP->FT: load a converged linear-probe head before phase 2 (Kumar arXiv:2202.10054)
# ---------------------------------------------------------------------------

def load_head_weights(head: nn.Module, path: str, map_location="cpu") -> dict:
    """Initialise ``head`` from a converged probe checkpoint. FAILS LOUDLY on mismatch.

    Research basis (RB): Kumar et al. arXiv:2202.10054 -- fine-tuning from a RANDOM (or
    otherwise fixed) head distorts pretrained features and underperforms out-of-distribution
    (avg ~2% better ID but ~7% WORSE OOD than a linear probe across 10 shift datasets);
    linear-probe-then-finetune (LP-FT) recovers ~10% OOD over full FT. Our shift is large
    and the pretrained features are good, so this ordering is load-bearing, not a nicety.

    Phase 1 is ALREADY DONE: the existing frozen runs in ``runs/`` ARE the linear probe.
    This loads those head weights; it never retrains phase 1.

    Asserted: exact key match, exact shape match, and that the loaded head is not an
    all-zeros / uninitialised tensor set. Silent partial loads are THE failure mode here,
    so ``strict=True`` plus these checks, never a best-effort merge.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "head" not in ckpt:
        raise KeyError(
            f"--init-head-from checkpoint {path} has no 'head' entry (keys: "
            f"{sorted(ckpt.keys())}); it is not a train_skeleton/run_baseline checkpoint.")
    sd = ckpt["head"]

    want, got = head.state_dict(), sd
    if set(want) != set(got):
        raise RuntimeError(
            f"--init-head-from key mismatch for {path}:\n"
            f"  missing from checkpoint : {sorted(set(want) - set(got))}\n"
            f"  unexpected in checkpoint: {sorted(set(got) - set(want))}\n"
            f"(head_type mismatch? checkpoint says {ckpt.get('head_type')!r})")
    for k in want:
        if tuple(want[k].shape) != tuple(got[k].shape):
            raise RuntimeError(
                f"--init-head-from shape mismatch for {path} at '{k}': "
                f"model expects {tuple(want[k].shape)}, checkpoint has {tuple(got[k].shape)}")

    # an all-zero / all-identical head is an UNTRAINED head -- loading it would silently
    # give us exactly the random-head initialisation LP-FT exists to avoid.
    classifier_w = [v for k, v in got.items() if k.endswith("weight") and v.dim() == 2]
    if not classifier_w:
        raise RuntimeError(f"--init-head-from {path}: no 2-D weight tensor found in the head "
                           f"state_dict; cannot verify it is a converged probe.")
    if all(float(w.abs().max()) == 0.0 for w in classifier_w):
        raise RuntimeError(
            f"--init-head-from {path}: every classifier weight matrix is all-zeros -- this "
            f"is an uninitialised/never-trained head. Refusing to start LP->FT from it.")

    head.load_state_dict(sd, strict=True)
    return ckpt


# ---------------------------------------------------------------------------
# Zoobot v2.0.1 PUBLISHED fine-tuning defaults -- ADOPTED WITHOUT TUNING (C1/C2)
# ---------------------------------------------------------------------------
# RATIONALE (recorded verbatim, and written into every run_config.json):
ZOOBOT_PROTOCOL_MATCH = "zoobot_v2.0.1_defaults"
ZOOBOT_PROTOCOL_RATIONALE = (
    "Kumar et al.'s LP-FT protocol is an LR sweep plus ID-validation early stopping. We "
    "cannot do the early stopping -- source-val AUROC is saturated. Reconstructing half the "
    "protocol is worse than not running it, and any sweep would be tuned on fold-0 test "
    "data, contaminating the very number we report. We therefore adopt the benchmark's "
    "published hyperparameters without tuning."
)
# Verified from source: FinetuneableZoobotAbstract.configure_optimizers.
#   learning_rate = 1e-4   -- applied to the head AND, via lr_decay**0, to the TOP block.
#                             The head/top-block equality is not an inference: the head
#                             group is created with `lr`, and encoder block i=0 receives
#                             `lr * (lr_decay ** 0)` -- the same value. The source comment
#                             on the default reads "# 10x lower than typical, you may like
#                             to experiment", i.e. a deliberate choice.
#   lr_decay      = 0.75
#   weight_decay  = 0.05
#   cosine_schedule = False -- kept off.
ZOOBOT_LR = 1e-4
ZOOBOT_LR_DECAY = 0.75
ZOOBOT_WEIGHT_DECAY = 0.05
ZOOBOT_COSINE_SCHEDULE = False


def zoobot_protocol_provenance() -> dict:
    """The provenance block every runner writes into ``run_config.json`` (C2 deliverable 2),
    so the write-up can state these values were ADOPTED, not tuned."""
    return {
        "protocol_match": ZOOBOT_PROTOCOL_MATCH,
        "protocol_match_rationale": ZOOBOT_PROTOCOL_RATIONALE,
        "protocol_match_source": ("Zoobot v2.0.1 "
                                  "FinetuneableZoobotAbstract.configure_optimizers"),
        "protocol_match_values": {
            "learning_rate_head": ZOOBOT_LR,
            "learning_rate_top_block": ZOOBOT_LR,
            "lr_decay": ZOOBOT_LR_DECAY,
            "weight_decay": ZOOBOT_WEIGHT_DECAY,
            "cosine_schedule": ZOOBOT_COSINE_SCHEDULE,
        },
        "protocol_match_tuned": False,
    }


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # data
    splits_path: str = ""
    data_root: str = ""
    fold: int = 0
    # model
    head_type: str = LINEAR
    encoder_mode: str = FROZEN
    # ---- LEVER 1: partial encoder unfreezing (LP->FT) ----
    # n_blocks=0 (DEFAULT) == the frozen encoder, bit-for-bit: a SINGLE-group optimizer
    # over head(+standardizer) params, exactly as before. n_blocks>0 unfreezes the top
    # n_blocks of Zoobot's own tuneable_blocks (stages[3], stages[2], ...) and adds one
    # layer-decayed param group per block, replicating Zoobot's configure_optimizers:
    #   head at head_lr (NO decay factor), encoder block i (0-based from the top) at
    #   backbone_lr * (lr_decay ** i).
    # Zoobot defaults: lr_decay=0.75, weight_decay=0.05, learning_rate=1e-4, AdamW.
    n_blocks: int = 0
    # None -> EQUALS the head lr, which is what Zoobot does: block i=0 gets
    # lr * (lr_decay ** 0) == lr. (Before the C1/C2 protocol match this sentinel resolved
    # to lr/10, an untuned engineering guess; see _build_optimizer.) Still overridable.
    backbone_lr: float | None = None
    lr_decay: float = ZOOBOT_LR_DECAY    # Zoobot default (unchanged by the protocol match)
    init_head_from: str | None = None    # LP->FT: converged probe head (Kumar 2202.10054)
    # Trade activation memory for recompute so batch 256 fits at n_blocks=3. NUMERICALLY
    # EXACT and batch-composition-preserving -- unlike gradient accumulation, which is not
    # available to us because CORAL/AdaMatch/FreeMatch are batch-coupled (PROJECT.md 7.4
    # pins batch size). No-op at n_blocks=0. See ZoobotEncoder.set_grad_checkpointing.
    grad_checkpointing: bool = False
    # optim -- ADOPTED FROM ZOOBOT v2.0.1, NOT TUNED (see ZOOBOT_PROTOCOL_RATIONALE).
    # Every bake-off runner (run_baseline / run_coral / run_adamatch / run_freematch /
    # run_dann) pins lr=1e-3, weight_decay=1e-4 in its OWN argparse, so round-1 results
    # remain bit-for-bit reproducible and are NOT retroactively changed by this default.
    lr: float = ZOOBOT_LR
    weight_decay: float = ZOOBOT_WEIGHT_DECAY
    batch_size: int = 64
    pos_fraction: float = 0.5
    # ---- DA / CORAL (the da_loss hook) ----
    # use_coral=False keeps the da_loss hook the INERT zero, so this stays a pure
    # plumbing skeleton and Baseline A's source-only semantics are unchanged. Set
    # True (run_coral / the CORAL tests do) to make the hook the real Deep CORAL loss.
    use_coral: bool = False
    # coral_weight (gamma) is a FIXED config value, NOT tuned on the target -- per the
    # RLSbench-style fixed-budget protocol (PROJECT.md 7.4), so it can never become an
    # accidental oracle knob. Total loss = BCE(source) + coral_weight * CORAL.
    coral_weight: float = 1.0
    coral_include_mean: bool = True      # align first moment too (DomainBed form); ablatable
    coral_shrinkage: float = 0.1         # covariance shrinkage lam (both domains)
    coral_target_ema: bool = True        # EMA the small target's statistics
    coral_ema_momentum: float = 0.9
    # ---- DA / AdaMatch (the da_loss hook, IMAGE-SPACE path) ----
    # AdaMatch needs weak+strong views, produced PRE-encoder, so it runs through THIS
    # image-space Trainer with the frozen encoder in the loop (source_batch/target_batch
    # must each carry "image"=weak + "image_strong"=strong, i.e. LensDataset n_views=2).
    # use_adamatch=False keeps the hook inert; True makes train_step the AdaMatch path.
    # These peaks/factors are FIXED, not tuned on target (RLSbench fixed-budget).
    use_adamatch: bool = False
    adamatch_mu_max: float = 1.0             # peak consistency weight
    adamatch_tau: float = 0.9               # relative confidence factor
    adamatch_ema_momentum: float = 0.9
    adamatch_consistency_warmup_frac: float = 0.1   # hard-withhold consistency this long
    adamatch_align_reference: tuple | None = None   # None -> source-dist EMA (~[0.5,0.5])
    # ---- DA / FreeMatch (the da_loss hook, IMAGE-SPACE path) ----
    # FreeMatch also needs weak+strong views (n_views=2) and runs through THIS image-space
    # Trainer with the frozen encoder in the loop. use_freematch=False keeps the hook inert;
    # True makes train_step the FreeMatch path. These are FIXED, not tuned on target.
    use_freematch: bool = False
    freematch_mu_max: float = 1.0                   # peak consistency weight
    freematch_saf_weight: float = 0.01              # fairness penalty weight (USB default)
    freematch_sat_ema_momentum: float = 0.999       # SAT EMA momentum (FreeMatch default)
    freematch_consistency_warmup_frac: float = 0.1  # hard-withhold consistency this long
    freematch_local_threshold_mode: str = "standard"   # "standard" | "floor" (make-or-break)
    freematch_pos_class: int = 1                    # which class is "lens" (floor protects it)
    freematch_fairness_target: tuple | None = None  # None -> uniform [0.5,0.5] (balanced pool)
    total_steps: int | None = None          # planned total steps (for the mu schedule)
    # schedule
    epochs: int = 1
    steps_per_epoch: int | None = None      # None -> len(source loader)
    max_steps: int | None = None            # hard cap across all epochs (smoke runs)
    standardizer_max_batches: int = 50      # streaming stats over this many src batches
    # misc
    seed: int = 42
    amp: bool = False
    device: str | None = None
    num_workers: int = 0
    ckpt_path: str | None = None
    verbose_encoder: bool = False


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Holds the frozen encoder + head + standardizer and runs the skeleton loop."""

    def __init__(self, cfg: TrainConfig, encoder: ZoobotEncoder | None = None):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(
            cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # encoder (shared instance allowed, e.g. across head types in a bake-off).
        # cfg.n_blocks selects the unfreeze scope; 0 == the original frozen encoder.
        self.encoder = encoder or ZoobotEncoder(
            encoder_mode=cfg.encoder_mode, n_blocks=cfg.n_blocks, device=self.device,
            verbose=cfg.verbose_encoder, grad_checkpointing=cfg.grad_checkpointing)
        # a SHARED encoder (bake-off reuse) may have been built with a different flag;
        # the Trainer's cfg is authoritative for the run it is about to drive.
        if encoder is not None and hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(cfg.grad_checkpointing)
        if getattr(self.encoder, "n_blocks", 0) != cfg.n_blocks:
            raise ValueError(
                f"a pre-built encoder with n_blocks={self.encoder.n_blocks} was passed to a "
                f"Trainer configured for n_blocks={cfg.n_blocks}. The optimizer would build "
                f"LR groups for blocks that are not actually trainable -- refusing.")

        self.head = build_head(cfg.head_type).to(self.device)
        # LP->FT: start phase 2 from the CONVERGED probe head, not a random one.
        if cfg.init_head_from:
            src = load_head_weights(self.head, cfg.init_head_from, map_location=self.device)
            # PROVENANCE LOG: the source path + the loaded weights' L2 norm, so every run's
            # log proves WHICH head it started from and that it was not a fresh init. A
            # silent no-op here would turn LP->FT into fine-tuning from a RANDOM head --
            # exactly the configuration Kumar et al. show underperforms a linear probe OOD --
            # and the result would read as "unfreezing didn't help".
            l2 = float(torch.sqrt(sum((p.detach().float() ** 2).sum()
                                      for p in self.head.parameters())))
            print(f"[LP->FT] head initialised from {cfg.init_head_from}")
            print(f"[LP->FT]   head_type={src.get('head_type')}  "
                  f"epoch={(src.get('cfg') or {}).get('epoch')}  "
                  f"loaded||head||_2 = {l2:.6f}")
        self.standardizer = FeatureStandardizer(EMBED_DIM).to(self.device)

        # The da_loss hook. CoralLoss operates on the head's ADAPTER features
        # (head.feat_dim wide) and owns NO trainable params (only EMA buffers), so it
        # is deliberately NOT added to the optimizer's parameter set.
        self.coral = CoralLoss(
            self.head.feat_dim, include_mean=cfg.coral_include_mean,
            shrinkage=cfg.coral_shrinkage, target_ema=cfg.coral_target_ema,
            ema_momentum=cfg.coral_ema_momentum).to(self.device)

        if sum([bool(cfg.use_coral), bool(cfg.use_adamatch),
                bool(cfg.use_freematch)]) > 1:
            raise ValueError("use_coral / use_adamatch / use_freematch are mutually "
                             "exclusive (one da_loss hook at a time).")

        # AdaMatch operates on head LOGITS of weak/strong views (not adapter features)
        # and owns only EMA buffers (no trainable params) -- like CORAL it is NOT added to
        # the optimizer's parameter set.
        self.adamatch = AdaMatchLoss(
            mu_max=cfg.adamatch_mu_max, tau=cfg.adamatch_tau,
            ema_momentum=cfg.adamatch_ema_momentum,
            consistency_warmup_frac=cfg.adamatch_consistency_warmup_frac,
            align_reference=cfg.adamatch_align_reference).to(self.device)

        # FreeMatch, like AdaMatch, operates on head LOGITS of weak/strong views and owns
        # only SAT buffers (no trainable params) -- NOT added to the optimizer's params.
        self.freematch = FreeMatchLoss(
            mu_max=cfg.freematch_mu_max, saf_weight=cfg.freematch_saf_weight,
            sat_ema_momentum=cfg.freematch_sat_ema_momentum,
            consistency_warmup_frac=cfg.freematch_consistency_warmup_frac,
            local_threshold_mode=cfg.freematch_local_threshold_mode,
            pos_class=cfg.freematch_pos_class,
            fairness_target=cfg.freematch_fairness_target).to(self.device)

        self.bce = nn.BCEWithLogitsLoss()
        self.opt = self._build_optimizer()

        self._amp_enabled = bool(cfg.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

    # -- optimizer scope: head + (future) standardizer params, NEVER the encoder --
    def _trainable_params(self):
        params = list(self.head.parameters())
        params += [p for p in self.standardizer.parameters() if p.requires_grad]
        return params

    # -- optimizer construction (single-group frozen path / layer-decayed unfrozen path) --

    def _build_optimizer(self):
        """AdamW over the head (+ any standardizer params) and, at n_blocks>0, one
        LAYER-DECAYED group per unfrozen encoder block.

        n_blocks == 0 takes the ORIGINAL code path verbatim -- one group, lr=cfg.lr -- so
        the frozen runs in ``runs/`` stay reproducible.

        n_blocks > 0 replicates Zoobot's ``configure_optimizers`` LR schedule:
            head            -> head_lr                          (no decay factor)
            encoder block i -> backbone_lr * (lr_decay ** i)     (i 0-based FROM THE TOP)
        consuming ``encoder.tuneable_blocks_in_lr_order`` so the LR index and the unfreeze
        index come from the SAME list and cannot drift apart.
        """
        cfg = self.cfg
        if cfg.n_blocks == 0:
            return torch.optim.AdamW(
                self._trainable_params(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        head_lr = cfg.lr
        # DEFAULT backbone_lr == head_lr, matching Zoobot: its head group is created with
        # `lr` and encoder block i=0 receives `lr * (lr_decay ** 0)`, i.e. THE SAME VALUE.
        # This replaced an earlier head_lr/10 engineering guess as part of the C1/C2
        # protocol match -- adopting the benchmark's published configuration rather than
        # tuning our own (ZOOBOT_PROTOCOL_RATIONALE). Overridable with --backbone-lr.
        backbone_lr = cfg.backbone_lr if cfg.backbone_lr is not None else head_lr

        groups = [{"params": list(self.head.parameters())
                   + [p for p in self.standardizer.parameters() if p.requires_grad],
                   "lr": head_lr, "name": "head"}]
        for i, (name, block) in enumerate(
                self.encoder.tuneable_blocks_in_lr_order[:cfg.n_blocks]):
            bl = backbone_lr * (cfg.lr_decay ** i)
            groups.append({"params": [p for p in block.parameters() if p.requires_grad],
                           "lr": bl, "name": name})

        opt = torch.optim.AdamW(groups, lr=head_lr, weight_decay=cfg.weight_decay)
        self._log_param_groups(opt)
        return opt

    def _log_param_groups(self, opt):
        """Print group index / block name / LR / param count at startup (§1b)."""
        print(f"[optim] n_blocks={self.cfg.n_blocks}  head_lr={self.cfg.lr:g}  "
              f"backbone_lr={(self.cfg.backbone_lr if self.cfg.backbone_lr is not None else self.cfg.lr):g}"
              f"  lr_decay={self.cfg.lr_decay:g}  weight_decay={self.cfg.weight_decay:g}")
        total = 0
        for gi, g in enumerate(opt.param_groups):
            n = sum(p.numel() for p in g["params"])
            total += n
            print(f"[optim]   group {gi}: {g.get('name', '?'):<10} lr={g['lr']:.3e}  "
                  f"params={n:,}")
        print(f"[optim]   TOTAL trainable = {total:,} params")

    def _autocast(self):
        if self._amp_enabled:
            # dtype PINNED to float16, never left to the autocast default. T4 (sm_75) has
            # NO bfloat16 support, and PROJECT.md 7.7 already records fp16 as the
            # image-space precision. Task-2 measurement showed augmentation+precision is
            # worth -0.0714 TPR on this metric -- comparable to the effect being measured --
            # so a silent dtype change between venues would be indistinguishable from a
            # real result. Explicit float16 makes that impossible.
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    @property
    def precision(self) -> str:
        """'fp16' or 'fp32' -- the resolved precision, for the run config + gate check."""
        return "fp16" if self._amp_enabled else "fp32"

    # -- standardizer fit (streaming no_grad pass over source_train) --------

    @torch.no_grad()
    def fit_standardizer(self, source_loader, max_batches: int | None = None):
        max_batches = max_batches or self.cfg.standardizer_max_batches
        n = 0
        s = torch.zeros(EMBED_DIM, dtype=torch.float64, device=self.device)
        ss = torch.zeros(EMBED_DIM, dtype=torch.float64, device=self.device)
        for bi, batch in enumerate(source_loader):
            if max_batches and bi >= max_batches:
                break
            imgs = batch["image"].to(self.device, non_blocking=True)
            f = encode_features(self.encoder, imgs, self.cfg.encoder_mode).double()
            n += f.shape[0]
            s += f.sum(0)
            ss += (f * f).sum(0)
        if n == 0:
            raise RuntimeError("no source batches seen while fitting the standardizer")
        mean = s / n
        var = (ss / n - mean * mean).clamp_min(1e-12)
        self.standardizer.set_stats(mean.float(), var.sqrt().float())
        return self.standardizer

    # -- one training step --------------------------------------------------

    def train_step(self, source_batch: dict, target_batch: dict | None = None,
                   step: int = 0, total_steps: int | None = None) -> dict:
        if target_batch is None:
            # SOURCE-ONLY (Baseline A, image-space). There is no target batch, so there is
            # nothing for the DA hook to consume and NO DA term is added -- which is exactly
            # what makes this the source-only control (mirrors run_baseline.py's
            # feature-space loop, PROJECT.md 7.4). Any use_* DA flag here would be a
            # contradiction, so it is refused rather than silently ignored.
            if self.cfg.use_coral or self.cfg.use_adamatch or self.cfg.use_freematch:
                raise ValueError(
                    "train_step() got target_batch=None (source-only) but a DA loss is "
                    "enabled (coral/adamatch/freematch). A DA method needs a target batch.")
            return self._source_only_train_step(source_batch)

        assert_target_unlabeled(target_batch)          # label-free guard, every step
        self.head.train()
        self.opt.zero_grad(set_to_none=True)

        if self.cfg.use_adamatch:
            return self._adamatch_train_step(source_batch, target_batch, step, total_steps)
        if self.cfg.use_freematch:
            return self._freematch_train_step(source_batch, target_batch, step, total_steps)

        with self._autocast():
            # encode BOTH domains under the frozen encoder (detached features)
            src_feat = self.standardizer(
                encode_features(self.encoder, source_batch["image"].to(self.device),
                                self.cfg.encoder_mode))
            tgt_feat = self.standardizer(
                encode_features(self.encoder, target_batch["image"].to(self.device),
                                self.cfg.encoder_mode))

            src_logits, src_adapter = self.head.forward_with_features(src_feat)
            _, tgt_adapter = self.head.forward_with_features(tgt_feat)

            # supervised BCE on the SOURCE (labeled) only
            y = source_batch["label"].float().to(self.device).view(-1, 1)
            sup_loss = self.bce(src_logits, y)

            # ===================== DA LOSS HOOK (CORAL) ======================
            # Consumes the HEAD/adapter features (src_adapter / tgt_adapter) -- never
            # the raw encoder features -- so it CANNOT backprop into the frozen
            # encoder (the collapse we measured); the encoder features arrived
            # detached and CoralLoss asserts the adapter features carry grad.
            # The target's adapter features are used WITHOUT its labels
            # (assert_target_unlabeled already ran above, every step).
            if self.cfg.use_coral:
                da_loss = self.cfg.coral_weight * self.coral(src_adapter, tgt_adapter)
            else:
                # inert path: exactly 0 but still touches the head-feature graph, so
                # the plumbing (shapes/devices/head-only grad routing) stays proven.
                da_loss = 0.0 * (src_adapter.pow(2).mean() + tgt_adapter.pow(2).mean())
            # =================================================================

            loss = sup_loss + da_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        return {"loss": float(loss.detach()),
                "sup_loss": float(sup_loss.detach()),
                "da_loss": float(da_loss.detach())}

    # -- one SOURCE-ONLY training step (image-space Baseline A) --------------

    def _source_only_train_step(self, source_batch: dict) -> dict:
        """Supervised BCE on the SOURCE weak view only -- the image-space Baseline A step.

        This is the image-space twin of run_baseline.py's cached-feature loop. It exists so
        Lever 1 can be screened on Baseline A: at n_blocks=0 it must reproduce the cached
        number (the A0-image control), and at n_blocks>0 the encoder gradient path is the
        ONLY thing that changed.
        """
        self.head.train()
        self.opt.zero_grad(set_to_none=True)

        with self._autocast():
            feat = self.standardizer(
                encode_features(self.encoder, source_batch["image"].to(self.device),
                                self.cfg.encoder_mode))
            logits = self.head(feat)
            y = source_batch["label"].float().to(self.device).view(-1, 1)
            loss = self.bce(logits, y)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return {"loss": float(loss.detach()), "sup_loss": float(loss.detach()),
                "da_loss": 0.0}

    # -- one AdaMatch training step (image-space; weak+strong views) ---------

    def _adamatch_train_step(self, source_batch: dict, target_batch: dict,
                             step: int, total_steps: int | None) -> dict:
        """AdaMatch step: encode weak+strong views of BOTH domains through the frozen
        encoder -> standardizer -> head logits, then AdaMatchLoss. Requires both batches
        to carry "image" (weak) and "image_strong" (strong) (LensDataset n_views=2)."""
        if "image_strong" not in source_batch or "image_strong" not in target_batch:
            raise KeyError(
                "AdaMatch needs weak+strong views: source_batch/target_batch must each "
                "have 'image' (weak) and 'image_strong' (strong) -- build the datasets "
                "with n_views=2.")
        total_steps = total_steps or self.cfg.total_steps or 1
        self.adamatch.train()

        def _logits(imgs):
            feat = self.standardizer(
                encode_features(self.encoder, imgs.to(self.device), self.cfg.encoder_mode))
            return self.head(feat)                       # (B,1) logits

        # AUTOCAST COVERS ONLY THE FORWARDS (frozen encoder + trainable head). Under
        # --fp16/cfg.amp these run in fp16 on CUDA for speed; the frozen encoder is pure
        # inference (its no_grad forward needs no GradScaler), and the head's fp16 backward
        # is handled by self.scaler. Outside cuda-AMP, self._autocast() is a nullcontext,
        # so this is a no-op and the run stays fp32.
        with self._autocast():
            src_weak = _logits(source_batch["image"])
            src_strong = _logits(source_batch["image_strong"])
            tgt_weak = _logits(target_batch["image"])
            tgt_strong = _logits(target_batch["image_strong"])

        # Exit autocast and cast the logits back to fp32 so the AdaMatch loss / EMA /
        # distribution-alignment / thresholding math is EXACTLY the fp32 path -- fp16
        # never touches the loss computation, only the forwards that produced the logits.
        src_weak, src_strong = src_weak.float(), src_strong.float()
        tgt_weak, tgt_strong = tgt_weak.float(), tgt_strong.float()
        y = source_batch["label"].float().to(self.device)

        # ===================== DA LOSS HOOK (AdaMatch) =======================
        # STRONG-view logits (source strong -> supervised BCE; target strong -> masked
        # consistency) carry the gradient and flow from the trainable head; the frozen
        # encoder ran under no_grad so nothing backprops into it. The target labels are
        # never used (assert_target_unlabeled ran in train_step; pseudo-labels come only
        # from the model's own aligned target-weak preds).
        loss, info = self.adamatch(src_weak, src_strong, tgt_weak, tgt_strong,
                                   y, step, total_steps)
        # =====================================================================

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        return {"loss": float(loss.detach()),
                "sup_loss": info["sup_loss"],
                "da_loss": info["mu_eff"] * info["consistency"],
                "consistency": info["consistency"], "mu": info["mu"],
                "mu_eff": info["mu_eff"], "acceptance": info["acceptance"],
                "accepted_pos_fraction": info["accepted_pos_fraction"],
                "threshold": info["threshold"], "n_accepted": info["n_accepted"]}

    # -- one FreeMatch training step (image-space; weak+strong views) ---------

    def _freematch_train_step(self, source_batch: dict, target_batch: dict,
                              step: int, total_steps: int | None) -> dict:
        """FreeMatch step: encode the SOURCE weak view (supervised) + the TARGET weak+strong
        views through the frozen encoder -> standardizer -> head logits, then FreeMatchLoss
        (SAT masking + FixMatch consistency + self-adaptive fairness). Requires both batches
        to carry "image" (weak); the TARGET must also carry "image_strong" (LensDataset
        n_views=2). Unlike AdaMatch, FreeMatch's supervised term uses the SOURCE WEAK view
        (FixMatch convention) and needs NO source strong view, so we do not compute one."""
        if "image_strong" not in target_batch:
            raise KeyError(
                "FreeMatch needs weak+strong TARGET views: target_batch must have 'image' "
                "(weak) and 'image_strong' (strong) -- build the target dataset with "
                "n_views=2.")
        total_steps = total_steps or self.cfg.total_steps or 1
        self.freematch.train()

        def _logits(imgs):
            feat = self.standardizer(
                encode_features(self.encoder, imgs.to(self.device), self.cfg.encoder_mode))
            return self.head(feat)                       # (B,1) logits

        # AUTOCAST COVERS ONLY THE FORWARDS (frozen encoder + trainable head) -- identical
        # to the AdaMatch path: under --fp16/cfg.amp these run in fp16 on CUDA for speed;
        # the loss / SAT-EMA / SAF math is then done in fp32 (logits cast back below).
        with self._autocast():
            src_weak = _logits(source_batch["image"])
            tgt_weak = _logits(target_batch["image"])
            tgt_strong = _logits(target_batch["image_strong"])

        src_weak = src_weak.float()
        tgt_weak, tgt_strong = tgt_weak.float(), tgt_strong.float()
        y = source_batch["label"].float().to(self.device)

        # ===================== DA LOSS HOOK (FreeMatch) ======================
        # SOURCE weak logits -> supervised BCE; TARGET strong logits -> masked consistency
        # + fairness. Both carry gradient and flow from the trainable head; the frozen
        # encoder ran under no_grad so nothing backprops into it. Target labels never used
        # (assert_target_unlabeled ran in train_step; pseudo-labels come only from the
        # model's own SAT-masked target-weak preds).
        loss, info = self.freematch(src_weak, tgt_weak, tgt_strong, y, step, total_steps)
        # =====================================================================

        # ---- fp16 SKIPPED-STEP DETECTION (see the note below) ----
        scale_before = self.scaler.get_scale() if self._amp_enabled else None
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        # WHY THIS IS INSTRUMENTED AND NOT ASSUMED AWAY. torch.amp.GradScaler SILENTLY
        # discards an optimizer step whose gradients contain inf/nan and then halves the
        # scale. With a frozen encoder (1,921 trainable params) that essentially never
        # happened, so round 1 never needed to know. At n_blocks=3 (14.8M trainable, deep
        # backward through the encoder in fp16) it does: a short local smoke produced two
        # BIT-IDENTICAL consecutive checkpoints, i.e. real steps that changed nothing. A run
        # losing a large fraction of its updates this way would look exactly like
        # "unfreezing does not help", which is precisely the conclusion this run exists to
        # draw -- so the skip RATE is logged as a first-class diagnostic. A decrease in the
        # scale across update() is the standard detection idiom (update() may also GROW the
        # scale, which is why this tests for a decrease rather than any change).
        skipped = bool(self._amp_enabled and self.scaler.get_scale() < scale_before)

        out = {"loss": float(loss.detach()),
               "sup_loss": info["sup_loss"],
               "da_loss": info["mu_eff"] * info["consistency"] + self.freematch.saf_weight * info["saf"],
               "consistency": info["consistency"], "saf": info["saf"],
               "mu": info["mu"], "mu_eff": info["mu_eff"],
               "acceptance": info["acceptance"],
               "accepted_pos_fraction": info["accepted_pos_fraction"],
               "tau_global": info["tau_global"], "local_neg": info["local_neg"],
               "local_pos": info["local_pos"], "n_accepted": info["n_accepted"],
               "step_skipped": float(skipped),
               "amp_scale": (float(self.scaler.get_scale()) if self._amp_enabled
                             else float("nan"))}
        # BRIEF 5.3 per-class mask-rate / pseudo-label-distribution diagnostics. Forwarded
        # verbatim when FreeMatchLoss provides them, so an older loss module (or a mock in
        # a unit test) simply yields a stats dict without them rather than a KeyError.
        for k in ("mask_rate_pos", "mask_rate_neg", "n_pseudo_pos", "n_pseudo_neg",
                  "n_accepted_pos", "n_accepted_neg", "pseudo_pos_fraction_all",
                  "mean_confidence_target", "pbar_neg", "pbar_pos"):
            if k in info:
                out[k] = info[k]
        return out

    # -- source-domain validation (model selection / early stopping) --------

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        self.head.eval()
        tot_loss, n, correct = 0.0, 0, 0
        for batch in val_loader:
            feat = self.standardizer(
                encode_features(self.encoder, batch["image"].to(self.device),
                                self.cfg.encoder_mode))
            logits = self.head(feat)
            y = batch["label"].float().to(self.device).view(-1, 1)
            tot_loss += float(self.bce(logits, y)) * y.shape[0]
            n += y.shape[0]
            pred = (logits.view(-1) > 0).long().cpu()
            correct += int((pred == batch["label"].view(-1)).sum())
        return {"val_loss": tot_loss / max(n, 1), "val_acc": correct / max(n, 1)}

    # -- fit loop -----------------------------------------------------------

    def fit(self, paired_loader: PairedDomainLoader, val_loader=None) -> list[float]:
        history: list[float] = []
        step = 0
        # total planned steps for the AdaMatch mu schedule (cfg override wins)
        total_steps = self.cfg.total_steps or (len(paired_loader) * self.cfg.epochs)
        for epoch in range(self.cfg.epochs):
            for source_batch, target_batch in paired_loader:
                stats = self.train_step(source_batch, target_batch,
                                        step=step, total_steps=total_steps)
                history.append(stats["loss"])
                step += 1
                if self.cfg.max_steps and step >= self.cfg.max_steps:
                    return history
            if val_loader is not None:
                vs = self.validate(val_loader)
                print(f"[epoch {epoch}] val_loss={vs['val_loss']:.4f} "
                      f"val_acc={vs['val_acc']:.4f}")
        return history

    # -- checkpoint (head + standardizer ONLY; encoder is reproducible) -----

    def encoder_trainable_state(self) -> dict:
        """The unfrozen encoder params ONLY (empty at n_blocks=0).

        At n_blocks=0 the encoder is bit-for-bit the published checkpoint, so storing it
        would be pure duplication -- the frozen checkpoint format stays unchanged. At
        n_blocks>0 the encoder weights ARE part of the model and MUST travel with the
        checkpoint or the run is unreproducible and unscorable.
        """
        if self.cfg.n_blocks == 0:
            return {}
        return {k: v.detach().cpu().clone()
                for k, v in self.encoder.model.state_dict().items()
                if k in self._trainable_encoder_keys()}

    def _trainable_encoder_keys(self) -> set:
        keys = set()
        for name, block in self.encoder.tuneable_blocks_in_lr_order[:self.cfg.n_blocks]:
            for pname, p in block.named_parameters():
                if p.requires_grad:
                    keys.add(f"{name}.{pname}")
        return keys

    def da_buffer_state(self) -> dict:
        """Every DA module's EMA/running buffers, by module name.

        These are TRAINING STATE, not model weights, and their absence is what made a
        continuation run restart FreeMatch's self-adaptive threshold from 1/C -- measured
        as part of a 0.153 TPR excursion on the fold-0 control. They are a handful of
        floats, so they ride in EVERY checkpoint unconditionally.
        """
        return {"freematch": {k: v.detach().cpu().clone()
                              for k, v in self.freematch.named_buffers()},
                "adamatch": {k: v.detach().cpu().clone()
                             for k, v in self.adamatch.named_buffers()},
                "coral": {k: v.detach().cpu().clone()
                          for k, v in self.coral.named_buffers()}}

    def save(self, path: str, full_state: bool = False, schedule: dict | None = None):
        """Write a checkpoint.

        ``full_state=True`` additionally stores the OPTIMIZER state (AdamW's first and
        second moments plus the per-parameter step counter). That is what a continuation
        run needs and what round 1 did not save -- but at n_blocks=3 it is two extra
        float32 tensors per parameter, i.e. ~118 MB on top of a 59.4 MB checkpoint. Since
        only the LAST checkpoint is ever continued from, the runners pass full_state=True
        for that one and leave the intermediate curve-point checkpoints lean.

        The DA buffers and the schedule position are tiny and are always written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "head_type": self.cfg.head_type,
            "head": self.head.state_dict(),
            "standardizer": self.standardizer.state_dict(),
            "cfg": asdict(self.cfg),
            # ---- training state around the model (the round-1 omission) ----
            "da_buffers": self.da_buffer_state(),
            "schedule": dict(schedule or {}),
            "format_version": 2,
        }
        if full_state:
            payload["optimizer"] = self.opt.state_dict()
        # n_blocks=0 -> these keys are simply absent, so the checkpoint is byte-compatible
        # with every existing reader (evaluate.load_checkpoint, curve_hook, collapse_gate).
        if self.cfg.n_blocks > 0:
            payload["n_blocks"] = self.cfg.n_blocks
            payload["encoder_trainable"] = self.encoder_trainable_state()
            payload["encoder_mode_effective"] = self.encoder.effective_mode
        torch.save(payload, path)

    def load(self, path: str, map_location=None, load_training_state: bool = True):
        ckpt = torch.load(path, map_location=map_location or self.device,
                          weights_only=False)
        self.head.load_state_dict(ckpt["head"])
        self.standardizer.load_state_dict(ckpt["standardizer"])
        # ---- training state, when the checkpoint has it (format_version >= 2) ----
        # A format-1 checkpoint silently has none, which is exactly the failure this
        # format exists to end -- so the caller is TOLD rather than left to assume.
        if load_training_state:
            bufs = ckpt.get("da_buffers") or {}
            for name, mod in (("freematch", self.freematch), ("adamatch", self.adamatch),
                              ("coral", self.coral)):
                if bufs.get(name):
                    missing, unexpected = mod.load_state_dict(bufs[name], strict=False)
                    if unexpected:
                        raise RuntimeError(
                            f"unexpected {name} buffer keys in {path}: {sorted(unexpected)}")
            if ckpt.get("optimizer"):
                self.opt.load_state_dict(ckpt["optimizer"])
            if not bufs:
                print(f"[ckpt] NOTE: {os.path.basename(path)} is format "
                      f"{ckpt.get('format_version', 1)} (weights only) -- it carries no DA "
                      f"buffers and no optimizer state. Continuing from it RESTARTS the "
                      f"self-adaptive threshold and Adam's moments; use "
                      f"continuation_state.py to reconstruct them.")
        if ckpt.get("encoder_trainable"):
            if int(ckpt.get("n_blocks", 0)) != self.cfg.n_blocks:
                raise ValueError(
                    f"checkpoint was trained with n_blocks={ckpt.get('n_blocks')} but this "
                    f"Trainer is configured for n_blocks={self.cfg.n_blocks}; the encoder "
                    f"weights would land on the wrong blocks.")
            missing, unexpected = self.encoder.model.load_state_dict(
                ckpt["encoder_trainable"], strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected encoder keys in {path}: {sorted(unexpected)}")
        return ckpt


# ---------------------------------------------------------------------------
# dataset / loader construction for a real run
# ---------------------------------------------------------------------------

def build_loaders(cfg: TrainConfig):
    """Build the balanced source loader, the target loader, the paired iterator,
    and the source-domain val loader for a real training run."""
    src_ds = LensDataset(cfg.splits_path, cfg.data_root, role="source_train", aug="weak")
    tgt_ds = LensDataset(cfg.splits_path, cfg.data_root, role="target", fold=cfg.fold,
                         aug="weak")
    val_ds = LensDataset(cfg.splits_path, cfg.data_root, role="val", aug="none")

    steps = cfg.steps_per_epoch or (len(src_ds) // cfg.batch_size)
    src_loader = make_balanced_loader(
        src_ds, cfg.batch_size, use_latent=False, num_batches=steps,
        pos_fraction=cfg.pos_fraction, seed=cfg.seed, num_workers=cfg.num_workers)
    tgt_loader = make_balanced_loader(
        tgt_ds, cfg.batch_size, use_latent=True, num_batches=steps,
        pos_fraction=cfg.pos_fraction, seed=cfg.seed + 1, num_workers=cfg.num_workers)
    paired = PairedDomainLoader(src_loader, tgt_loader)

    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)
    return src_loader, paired, val_loader


# ---------------------------------------------------------------------------
# CLI (a short real smoke; not a full training run)
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Label-free DA training skeleton (plumbing only).")
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.normpath(os.path.join(here, "..", "data"))
    p.add_argument("--data-root", default=os.environ.get("LENS_DATA_ROOT", default_root))
    p.add_argument("--head-type", default=LINEAR, choices=("linear", "mlp"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=20, help="short smoke by default")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n-blocks", type=int, default=0,
                   help="how many of Zoobot's tuneable_blocks (top-first: stages[3], "
                        "stages[2], ...) to unfreeze. 0 (default) = frozen encoder, "
                        "unchanged. NOTE n_blocks=3 is effectively FULL fine-tuning "
                        "(99.2%% of encoder params).")
    p.add_argument("--backbone-lr", type=float, default=None,
                   help="LR of the TOP unfrozen block (default: head lr / 10); lower "
                        "blocks get backbone_lr * lr_decay**i")
    p.add_argument("--lr-decay", type=float, default=0.75,
                   help="per-block LR decay, Zoobot default 0.75")
    p.add_argument("--init-head-from", default=None,
                   help="LP->FT: path to a CONVERGED probe checkpoint whose head weights "
                        "initialise phase 2 (Kumar arXiv:2202.10054). Fails loudly on any "
                        "key/shape mismatch or an all-zeros head.")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="recompute encoder activations in backward instead of storing them "
                        "(numerically exact; keeps TRUE batch 256, unlike accumulation "
                        "which is invalid for the batch-coupled DA losses). No-op at "
                        "n_blocks=0. Default OFF.")
    args = p.parse_args()

    cfg = TrainConfig(
        splits_path=os.path.join(args.data_root, "splits", "splits.json"),
        data_root=args.data_root, fold=args.fold, head_type=args.head_type,
        batch_size=args.batch_size, max_steps=args.max_steps, amp=args.amp,
        ckpt_path=args.ckpt, verbose_encoder=True, n_blocks=args.n_blocks,
        backbone_lr=args.backbone_lr, lr_decay=args.lr_decay,
        init_head_from=args.init_head_from,
        grad_checkpointing=args.grad_checkpointing)

    if not os.path.exists(cfg.splits_path):
        print(f"SKIP: real splits not found at {cfg.splits_path}")
        return

    trainer = Trainer(cfg)
    src_loader, paired, val_loader = build_loaders(cfg)
    print("[standardizer] fitting on source_train ...")
    trainer.fit_standardizer(src_loader)
    print(f"[train] running {cfg.max_steps} skeleton steps (head={cfg.head_type}) ...")
    hist = trainer.fit(paired, val_loader=val_loader)
    print(f"[train] first loss={hist[0]:.4f} last loss={hist[-1]:.4f}")
    if cfg.ckpt_path:
        trainer.save(cfg.ckpt_path)
        print(f"[ckpt] saved head+standardizer to {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
