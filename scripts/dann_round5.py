"""
dann_round5.py
==============
ROUND-5 support machinery for **DANN x n_blocks=3**, mirroring ``coral_round4.py``
one-for-one: the installer that swaps the ``da_loss`` hook, the per-step monitor
window, the gradient-reachability probe, the held-out target slice, and the
no-BatchNorm assertion.

The loss math itself lives in ``da_losses.py`` (``GradientReversal`` /
``DomainDiscriminator`` / ``DannLoss``). This module is the plumbing around it.

WHY DANN GOES THROUGH THE ``coral`` HOOK SLOT
---------------------------------------------
``train_skeleton.Trainer.train_step`` reaches its DA term through exactly one
attribute::

    da_loss = self.cfg.coral_weight * self.coral(src_adapter, tgt_adapter)

Round 4 exploited that to install ``Fp32CoralLoss`` "so nothing in train_skeleton
needs to change or even to know". Round 5 does the same thing with ``DannLoss``,
at ``coral_weight = 1.0``:

  * ``train_skeleton.py`` stays **byte-identical**. That matters because its
    sha256 is recorded in every round-4 ``run.json`` and because the CORAL /
    FreeMatch / AdaMatch paths run through the same function -- an additive
    ``use_dann`` branch would be a change to a module three completed rounds
    depend on, for no behavioural gain.
  * ``coral_weight = 1.0`` is not a fudge: canonical DANN puts the domain term
    into the total loss with coefficient **1**. lambda lives in the GRL, where it
    scales the gradient flowing back into the features, NOT in front of the loss.
    Giving DANN a gamma-like weight as well would be a second, undeclared knob.

The cost of the trick is that ``run_config.json`` would otherwise read
``use_coral: true``, so the runner writes ``method: "dann"`` and
``da_hook_slot: "Trainer.coral (see dann_round5.install_dann)"`` next to it, and
``install_dann`` refuses to run on a Trainer that already has a real CORAL module
installed.

WHAT ``Trainer.save`` DOES NOT KNOW ABOUT
------------------------------------------
D owns ~1.7M trainable parameters and ``Trainer.save`` stores only the head, the
standardizer and the unfrozen encoder blocks. Every checkpoint written by the
round-5 runner therefore carries D's ``state_dict`` inside the ``schedule``
payload (``discriminator_state``), and the resume path restores it. A checkpoint
without it is not continuable -- the same class of silent restart PROJECT.md 7.10
paid a round to find.
"""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from da_losses import (DANN_DROPOUT, DANN_HIDDEN,                 # noqa: E402
                       DANN_LAMBDA_HORIZON, DannLoss)
from samplers import NEG_LABEL, POS_LABEL                       # noqa: E402


# ---------------------------------------------------------------------------
# 1. the monitored loss
# ---------------------------------------------------------------------------

DEFAULT_HOLDOUT_FRAC = 0.10


class MonitoredDannLoss(DannLoss):
    """:class:`da_losses.DannLoss` plus the per-step representation monitors.

    NO fp32 PINNING, and that is a considered difference from round 4.
    ``Fp32CoralLoss`` exists because a 640x640 second moment and its 409,600-term
    Frobenius reduction are a half-precision hazard. DANN's domain term has no such
    reduction: it is three Linear layers and a BCE, and ``torch.amp`` already
    promotes ``binary_cross_entropy_with_logits`` to fp32 on its own (which is
    also why the fused form is mandatory -- a hand-rolled ``sigmoid`` + ``BCELoss``
    does NOT get that promotion). The MONITORS below are still computed in fp32,
    because an eigen-decomposition of a half-precision covariance is meaningless.

    The monitors are a read-only by-product of a forward pass that already
    happened: the adapter tensors are in hand, so this costs one norm, one std and
    (every ``spectrum_every`` steps) one 640x640 ``eigvalsh`` per domain. Measured
    locally at ~10 ms per ``eigvalsh``, i.e. ~2.6 min over an 8000-step run.

    ``feat_std_per_dim`` -- mean PER-DIMENSION std -- is the headline scale
    indicator, NOT mean ||f||. PROJECT.md 7.15 records why: writing z = mu + delta,
    ||z|| is dominated by mu while the covariance depends only on delta, so round
    4's total feature variance fell 56x while mean ||f|| moved 2.5% and the
    pre-registered monitor saw a rounding error. Mean ||f|| is still logged, as a
    scale sanity check that is known to be blind.
    """

    def __init__(self, feat_dim: int = 640, *, hidden: int = DANN_HIDDEN,
                 lambda_horizon: int = DANN_LAMBDA_HORIZON, lambda_zero: bool = False,
                 holdout_frac: float = DEFAULT_HOLDOUT_FRAC, spectrum_every: int = 1,
                 dropout: float = DANN_DROPOUT, n_dropout_layers: int = 2):
        super().__init__(feat_dim, hidden=hidden, lambda_horizon=lambda_horizon,
                         lambda_zero=lambda_zero, holdout_frac=holdout_frac,
                         dropout=dropout, n_dropout_layers=n_dropout_layers)
        self.spectrum_every = max(1, int(spectrum_every))
        self.n_forward = 0
        self.last_scale = {}
        self.last_spectrum = {}
        self.last_holdout = {}
        self.last_morph = {}

    # -- monitors ----------------------------------------------------------

    @staticmethod
    def scale_stats(z: torch.Tensor) -> tuple[float, float]:
        """(mean L2 norm of a feature VECTOR, mean per-dimension std across the batch).

        The same pair ``coral_round4.Fp32CoralLoss.scale_stats`` reports, so the two
        rounds' diagnostics are directly comparable. With a LayerNorm at unit gain
        the first is sqrt(D) = 25.298 for D = 640.
        """
        z = z.detach().float()
        l2 = float(z.norm(dim=1).mean())
        sd = float(z.std(dim=0, unbiased=True).mean()) if z.shape[0] > 1 else float("nan")
        return l2, sd

    @staticmethod
    def spectrum_stats(z: torch.Tensor) -> dict:
        """Participation ratio + 90%/99% variance dimensionality of a batch covariance.

        PR is computed from the trace/Frobenius identity

            PR = (sum lambda_i)^2 / sum lambda_i^2 = tr(C)^2 / ||C||_F^2

        which is EXACTLY the eigenvalue definition, not an approximation of it, and
        needs no decomposition. ``dims_at_90`` / ``dims_at_99`` do need the sorted
        spectrum, so they carry the eigvalsh cost.

        This is the RANK-collapse detector. Round 4 measured a 56x SCALE collapse
        with no rank loss (dims@90 29 -> 26, dims@99 164 -> 168), so the two have to
        be logged separately or the distinction PROJECT.md 7.15 spent a section
        establishing is not recoverable from the logs.
        """
        z = z.detach().float()
        if z.shape[0] < 3:
            return {"participation_ratio": float("nan"), "dims_at_90": float("nan"),
                    "dims_at_99": float("nan"), "total_var": float("nan")}
        C = torch.cov(z.T.double())
        tr = float(torch.diagonal(C).sum())
        fro2 = float((C ** 2).sum())
        pr = (tr * tr / fro2) if fro2 > 0 else float("nan")
        ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)
        tot = float(ev.sum())
        if not (tot > 0) or not math.isfinite(tot):
            # A degenerate batch (every row identical) has no spectrum. Report NaN
            # rather than the d+1 a zero-division would otherwise produce -- a
            # dimensionality of 641 out of 640 is a number that would be plotted.
            return {"participation_ratio": float("nan"), "dims_at_90": float("nan"),
                    "dims_at_99": float("nan"), "total_var": tot}
        cum = torch.cumsum(ev, 0) / tot
        return {"participation_ratio": pr,
                "dims_at_90": float(int((cum < 0.90).sum()) + 1),
                "dims_at_99": float(int((cum < 0.99).sum()) + 1),
                "total_var": tot}

    def set_morph_stats(self, stats: dict | None) -> None:
        """Record the morphology-proxy reading the training loop measured.

        Kept OUT of :meth:`forward` for the same reason the holdout is: the morphology
        pool must never touch the loss, and a monitor that lives inside the loss
        function is one refactor away from being wired into it. See
        ``morph_proxy.MorphProxy``.
        """
        self.last_morph = dict(stats or {})

    def set_holdout_stats(self, stats: dict | None) -> None:
        """Record the held-out-target-slice numbers the training loop measured.

        Kept OUT of :meth:`forward` deliberately: the holdout must never touch the
        loss, and a monitor that lives inside the loss function is one refactor away
        from being wired into it.
        """
        self.last_holdout = dict(stats or {})

    # -- forward -----------------------------------------------------------

    def forward(self, src_adapter, tgt_adapter, lambda_=None):
        loss = super().forward(src_adapter, tgt_adapter, lambda_=lambda_)

        s32 = src_adapter.detach().float()
        t32 = tgt_adapter.detach().float()
        s_l2, s_sd = self.scale_stats(s32)
        t_l2, t_sd = self.scale_stats(t32)
        self.last_scale = {"src_feat_l2": s_l2, "src_perdim_std": s_sd,
                           "tgt_feat_l2": t_l2, "tgt_perdim_std": t_sd,
                           "feat_l2_ref_unit_gain": math.sqrt(int(self.feat_dim))}
        if self.n_forward % self.spectrum_every == 0:
            ss = self.spectrum_stats(s32)
            ts = self.spectrum_stats(t32)
            self.last_spectrum = {
                "src_participation_ratio": ss["participation_ratio"],
                "src_dims_at_90": ss["dims_at_90"], "src_dims_at_99": ss["dims_at_99"],
                "src_total_var": ss["total_var"],
                "tgt_participation_ratio": ts["participation_ratio"],
                "tgt_dims_at_90": ts["dims_at_90"], "tgt_dims_at_99": ts["dims_at_99"],
                "tgt_total_var": ts["total_var"],
            }
        self.n_forward += 1
        return loss

    # -- the per-step record ----------------------------------------------

    def step_stats(self, sup_loss: float | None = None) -> dict:
        """The row the training loop pushes into the diagnostics window.

        ``ratio_lambda_domain_over_sup`` needs the supervised loss, which the module
        never sees, so the caller passes it. It is ``lambda * L_domain / L_sup``
        rather than ``L_domain / L_sup``: the domain term enters the total loss at
        coefficient 1, but the PRESSURE it puts on the representation is scaled by
        lambda in the GRL, and it is that pressure the balance question is about.
        """
        out = {**self.last_stats, **self.last_scale, **self.last_spectrum,
               **(self.last_holdout or {}), **(self.last_morph or {})}
        out.setdefault("d_acc_holdout", float("nan"))
        out.setdefault("d_holdout_logit_mean", float("nan"))
        raw = out.get("loss_domain_raw")
        lam = out.get("lambda_current")
        if sup_loss is not None and raw is not None and lam is not None:
            s = float(sup_loss)
            out["ratio_lambda_domain_over_sup"] = (
                float(lam) * float(raw) / s if (math.isfinite(s) and s > 0) else float("nan"))
        # RECOVERED in brief v2 -- ``morph_proxy.py``, ML-FP vs P1-reserve. The
        # setdefault survives only for the case where the runner is configured with
        # --morph-every > 1 and no reading has landed yet: NaN, never a fabricated
        # number and never a stale one carried across a reset.
        out.setdefault("morph_head_auroc", float("nan"))
        out.setdefault("morph_head_auroc_pool", float("nan"))
        out.setdefault("morph_proxy_auroc", float("nan"))
        out.setdefault("morph_proxy_auroc_encoder", float("nan"))
        return out


def install_dann(trainer, *, lambda_horizon: int = DANN_LAMBDA_HORIZON,
                 lambda_zero: bool = False, holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
                 spectrum_every: int = 1, d_lr: float | None = None,
                 weight_decay: float | None = None, dropout: float = DANN_DROPOUT,
                 n_dropout_layers: int = 2) -> MonitoredDannLoss:
    """Install DANN as the Trainer's ``da_loss`` hook and add D to the EXISTING AdamW.

    **ONE OPTIMISER. THIS IS NOT A STYLE CHOICE.** D's parameters become one extra
    param group on the optimizer the Trainer already built, at the HEAD learning
    rate (1e-4 by PROJECT.md 7.1a), with **no ``lr_decay`` factor** (D is not an
    encoder block; layer-wise decay indexes depth in the backbone and D has no
    depth in that sense) and the same weight decay (0.05).

    A second optimiser, or a separate learning rate for D, would reintroduce the
    discriminator/encoder BALANCE problem -- the thing gradient reversal exists to
    remove, and the entire reason DANN was chosen over ADDA for this round. Tuning
    a D learning rate would also be a method-specific hyperparameter search that
    PROJECT.md 7.14a's zero-tuning-budget position does not permit.

    Refuses to overwrite a real CORAL module, because the ``coral`` attribute is
    being used as the hook slot and silently discarding an installed
    ``Fp32CoralLoss`` would produce a run that logs CORAL settings and trains DANN.
    """
    from coral_round4 import Fp32CoralLoss
    if isinstance(trainer.coral, Fp32CoralLoss):
        raise RuntimeError(
            "install_dann() was handed a Trainer with Fp32CoralLoss already installed. "
            "The two methods share the da_loss hook slot; installing DANN over CORAL "
            "would train DANN while every CORAL setting still reads as configured.")
    if getattr(trainer, "dann", None) is not None:
        raise RuntimeError("install_dann() called twice on the same Trainer -- D would be "
                           "added to the optimizer twice.")

    dann = MonitoredDannLoss(trainer.head.feat_dim, lambda_horizon=lambda_horizon,
                             lambda_zero=lambda_zero, holdout_frac=holdout_frac,
                             spectrum_every=spectrum_every, dropout=dropout,
                             n_dropout_layers=n_dropout_layers).to(trainer.device)
    # D is the FIRST module in this pipeline with real train/eval semantics (7.1a: the
    # encoder has 0 BatchNorm and 0 active dropout). Set the mode explicitly at install
    # rather than relying on nn.Module's default, so it is visible in the code that
    # something now depends on it.
    dann.train()
    trainer.dann = dann
    trainer.coral = dann                      # THE da_loss hook slot -- see module docstring
    trainer.cfg.use_coral = True
    trainer.cfg.coral_weight = 1.0            # canonical DANN: the domain term is unweighted

    lr = float(trainer.cfg.lr if d_lr is None else d_lr)
    wd = float(trainer.cfg.weight_decay if weight_decay is None else weight_decay)
    trainer.opt.add_param_group({"params": list(dann.discriminator.parameters()),
                                 "lr": lr, "weight_decay": wd,
                                 "name": "domain_discriminator"})
    n_d = sum(p.numel() for p in dann.discriminator.parameters())
    n_drop = sum(1 for m in dann.discriminator.modules() if isinstance(m, nn.Dropout))
    print(f"[dann] D installed: {n_d:,} params in ONE extra AdamW group "
          f"(lr={lr:g}, weight_decay={wd:g}, no lr_decay); "
          f"lambda_horizon={lambda_horizon} lambda_zero={lambda_zero}; "
          f"dropout={dropout:g} x{n_drop}, D.training={dann.discriminator.training}")
    return dann


# ---------------------------------------------------------------------------
# 2. the no-BatchNorm assertion
# ---------------------------------------------------------------------------

def assert_no_batchnorm(*modules, context: str = "the assembled model") -> dict:
    """Raise if ANY ``nn.BatchNorm*`` module exists anywhere in ``modules``.

    PROJECT.md 7.1a measured the ConvNeXt encoder at **0 BatchNorm**, and this
    round's D deliberately substitutes LayerNorm for tllib's BatchNorm1d. The
    assertion turns both of those from statements into checks: a BN anywhere in
    this path would give train/eval mode real semantics, would couple every
    sample's activation to its batch's domain composition, and would be exactly the
    domain-adaptation footgun 7.1a says does not apply here.
    """
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    found, n_modules = [], 0
    for m in modules:
        if m is None:
            continue
        for name, sub in m.named_modules():
            n_modules += 1
            if isinstance(sub, bn_types):
                found.append(f"{type(m).__name__}.{name}:{type(sub).__name__}")
    if found:
        raise AssertionError(
            f"BatchNorm found in {context}: {found}. This path must be BatchNorm-free "
            f"(PROJECT.md 7.1a; and BN inside D is a domain-leak path).")
    return {"n_modules_scanned": n_modules, "n_batchnorm": 0, "context": context}


# ---------------------------------------------------------------------------
# 3. the held-out target slice
# ---------------------------------------------------------------------------

def build_target_holdout(target_ds, *, frac: float = DEFAULT_HOLDOUT_FRAC,
                         seed: int) -> dict:
    """Split the fold's target pool into a TRAIN pool and a held-out monitor slice.

    Returns ``{"train_pos", "train_neg", "holdout", "holdout_by_latent", ...}`` --
    dataset POSITIONS, not ids.

    THREE PROPERTIES THAT MATTER, each of which the runner records:

    1. **Deterministic given ``seed``.** ``numpy.random.default_rng(seed)`` and a
       sorted input order, so the same fold seed always yields the same slice. The
       indices go into ``run_config.json``; a slice that cannot be reconstructed
       makes ``d_acc_holdout`` uninterpretable across runs.

    2. **Excluded from D's loss by removal from the SAMPLER'S POOL**, not by a
       per-sample loss mask. A mask would keep the held-out images in the batch,
       burning encoder compute on samples that contribute nothing and shrinking D's
       effective target batch below the 256 PROJECT.md 7.4 pins. Removing them from
       the pool keeps the batch composition exactly as declared.

    3. **Stratified by LATENT class**, so the monitor slice has the same ~1:1
       lens/non-lens mix as the pool it is drawn from. The latent labels are
       diagnostics-only metadata (``LensDataset`` serves every target item as
       UNLABELED) and are used here for STRATIFICATION ONLY -- never as a
       supervision signal, never read by D, never in a loss. That is the same use
       ``samplers.ClassBalancedBatchSampler(use_latent=True)`` already makes of
       them for the balanced target loader.
    """
    if not (0.0 <= frac < 0.5):
        raise ValueError(f"holdout frac must be in [0, 0.5), got {frac}")
    by_latent = target_ds.indices_by_latent
    if POS_LABEL not in by_latent or NEG_LABEL not in by_latent:
        raise ValueError(f"target dataset latent map missing a class: {sorted(by_latent)}")

    rng = np.random.default_rng(int(seed))
    holdout, train = {}, {}
    for cls in (POS_LABEL, NEG_LABEL):
        pool = np.sort(np.asarray(by_latent[cls], dtype=np.int64))
        k = int(round(len(pool) * float(frac)))
        # never strip a class out of D's training pool entirely -- the balanced
        # sampler requires >= 1 of each class. The clamp never binds on the real
        # ~782-image target pool; it exists for the mock.
        k = min(max(k, 0), max(len(pool) - 1, 0))
        pick = np.sort(rng.choice(pool, size=k, replace=False)) if k else pool[:0]
        holdout[cls] = pick
        train[cls] = np.sort(np.setdiff1d(pool, pick, assume_unique=True))

    all_holdout = np.sort(np.concatenate([holdout[POS_LABEL], holdout[NEG_LABEL]]))
    return {
        "frac_requested": float(frac),
        "frac_actual": float(all_holdout.size / max(len(target_ds), 1)),
        "seed": int(seed),
        "n_pool": int(len(target_ds)),
        "train_pos": train[POS_LABEL], "train_neg": train[NEG_LABEL],
        "holdout": all_holdout,
        "holdout_pos": holdout[POS_LABEL], "holdout_neg": holdout[NEG_LABEL],
        "n_train": int(train[POS_LABEL].size + train[NEG_LABEL].size),
        "n_holdout": int(all_holdout.size),
        "note": ("dataset POSITIONS in the fold's target LensDataset. Held out of the "
                 "target SAMPLER's pool, so D never sees them in its loss; used only for "
                 "the d_acc_holdout memorisation monitor."),
    }


@torch.no_grad()
def materialise_holdout_images(target_ds, indices) -> torch.Tensor:
    """One FIXED augmented view of each held-out image, drawn once and reused.

    A fixed probe on purpose. If the slice were re-augmented every step, a change in
    ``d_acc_holdout`` could be D moving or the augmentation drawing differently, and
    the monitor exists to say which. Cost is ~78 x 1 x 224 x 224 float32 ~= 15 MB.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return torch.zeros(0, 1, 224, 224)
    return torch.stack([target_ds[int(i)]["image"] for i in idx], dim=0)


@torch.no_grad()
def holdout_domain_stats(trainer, dann, images: torch.Tensor, *, amp: bool = False) -> dict:
    """D's accuracy on the held-out TARGET slice. NO grad, NO loss, NO training.

    **D IS PUT IN EVAL MODE AND RESTORED.** Since brief v2 added Dropout(0.2) to D,
    this is no longer a formality: a memorisation monitor measured with dropout ON is
    not comparable to one measured with it off, the reading would carry a per-step
    random mask, and this column arms abort condition 1. ``DannLoss.eval_mode`` is a
    context manager that restores the PREVIOUS mode rather than forcing ``.train()``,
    and ``test_da_dann.py`` asserts ``D.training is True`` afterwards.

    ``d_acc_train``, by contrast, stays as measured inside the training forward -- in
    train mode, under dropout. That is deliberate: it reports D as it was actually
    trained. The pair is therefore NOT a like-for-like difference, and
    ``d_train_minus_holdout`` carries that caveat; the trend in each column is the
    signal, not the absolute gap.

    ONE-SIDED BY CONSTRUCTION, and stated rather than discovered: the slice is all
    target, so "accuracy" here is the fraction D correctly calls target. Read it
    ALONGSIDE ``d_acc_train`` and ``d_pred_source_frac`` -- a D that has collapsed to
    predicting "target" for everything scores 1.0 here and is caught by
    ``d_pred_source_frac`` going to 0, not by this column.
    """
    from train_skeleton import encode_features
    if images is None or images.numel() == 0:
        return {"d_acc_holdout": float("nan"), "d_holdout_logit_mean": float("nan"),
                "n_holdout_scored": 0.0, "d_holdout_measured_in_eval": 1.0}
    was_head_train = trainer.head.training
    trainer.head.eval()
    trainer.encoder.eval()
    ctx = trainer._autocast() if amp else _nullctx()
    with dann.eval_mode():                      # dropout OFF for the measurement
        assert not dann.discriminator.training, "D must be in eval for the holdout monitor"
        with ctx:
            feat = trainer.standardizer(
                encode_features(trainer.encoder, images.to(trainer.device),
                                trainer.cfg.encoder_mode))
            _, adapter = trainer.head.forward_with_features(feat)
            logits = dann.score_domain(adapter.float())
    if was_head_train:
        trainer.head.train()
    pred_target = (logits <= 0)                    # source == 1 == positive logit
    return {"d_acc_holdout": float(pred_target.float().mean()),
            "d_holdout_logit_mean": float(logits.float().mean()),
            "n_holdout_scored": float(logits.numel()),
            "d_holdout_measured_in_eval": 1.0}


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# 4. gradient reachability -- which parameters the DOMAIN term can actually move
# ---------------------------------------------------------------------------

def dann_reachable_params(trainer, src_images: torch.Tensor, tgt_images: torch.Tensor,
                          *, lambda_: float = 1.0, zero_tol: float = 0.0) -> dict:
    """Backprop the DOMAIN term ALONE and report which parameters it reaches.

    Reproduces round 4's method (``coral_round4.coral_reachable_params``) verbatim,
    on the DANN objective, so the two rounds' reachability claims are the same
    measurement rather than two similar-sounding ones.

    ANALYTICAL EXPECTATION, stated before the measurement:
      * ``n_blocks=3`` -> the head's LayerNorm gains PLUS ``encoder.stages.1/2/3``
        (**14,835,520** elements, PROJECT.md 7.1a), and NOTHING in ``stages.0`` or
        ``stem``.
      * ``head.classifier`` is NOT reachable: D consumes the adapter tensor, which
        is upstream of the classifier. (Same as CORAL.)
      * D's own parameters are reachable at FULL strength whatever lambda is -- the
        GRL sits between the features and D, so it scales the gradient going
        DOWNWARD only.

    **D IS PROBED IN EVAL MODE (dropout off).** Brief v2 added Dropout(0.2) to D, and
    with it on, the count of D parameters receiving a non-zero gradient depends on
    which units a random mask happened to keep -- so the probe would return a
    different number every call and the reachable set would stop being a structural
    claim. Dropout also cannot create or remove a PATH, only sample one, so switching
    it off is the measurement of the graph rather than of one draw from it. The
    encoder and head side is unaffected either way.

    **ONE PLACE THIS DIFFERS FROM ROUND 4, AND IT IS A REAL DIFFERENCE, NOT NOISE.**
    CORAL's LayerNorm **bias** gradient is analytically zero (covariance is
    shift-invariant and the first-moment term is a difference of means), and round 4
    had to attribute a 1e-11 residual to float cancellation. DANN has no such
    invariance: D is an arbitrary function of the features, so a constant shift
    genuinely changes its output and the bias is **genuinely reachable**. Expect
    ~1,280 head elements here (640 gains + 640 biases) where round 4 expected 640.
    A test that copies round 4's bias assertion across would fail for the right
    reason; do not copy it.

    Expect a small shortfall from the exactly-14,835,520 encoder count: dead GELU
    units in ``stages.3 mlp.fc2`` that no image in a small probe batch activates.
    Round 4 measured 27,524 (0.19%). Report the count; do not assert equality.
    """
    from train_skeleton import encode_features

    dann = getattr(trainer, "dann", None) or trainer.coral
    trainer.opt.zero_grad(set_to_none=True)
    trainer.head.train()

    # fp32, no autocast: this is an identity check on the GRAPH, and a half-precision
    # underflow could zero a gradient that is genuinely non-zero, turning a plumbing
    # question into a numerics question.
    src_feat = trainer.standardizer(
        encode_features(trainer.encoder, src_images.to(trainer.device),
                        trainer.cfg.encoder_mode))
    tgt_feat = trainer.standardizer(
        encode_features(trainer.encoder, tgt_images.to(trainer.device),
                        trainer.cfg.encoder_mode))
    _, src_adapter = trainer.head.forward_with_features(src_feat)
    _, tgt_adapter = trainer.head.forward_with_features(tgt_feat)

    with dann.eval_mode():                     # dropout OFF -- see the docstring
        da = dann(src_adapter, tgt_adapter, lambda_=lambda_)  # NO supervised term
        da.backward()

    named = list(trainer.head.named_parameters(prefix="head"))
    named += [(f"encoder.{n}", p) for n, p in trainer.encoder.model.named_parameters()]
    named += [(f"discriminator.{n}", p)
              for n, p in dann.discriminator.named_parameters()]

    reachable, zero_grad, none_grad, scale = {}, {}, [], {}
    for name, p in named:
        if p.grad is None:
            none_grad.append(name)
            continue
        g = p.grad.detach()
        scale[name] = {"rms": float(g.pow(2).mean().sqrt()),
                       "max": float(g.abs().max()),
                       "numel": int(g.numel()),
                       "n_exact_zero": int((g == 0).sum())}
        nz = int((g.abs() > zero_tol).sum())
        if nz:
            reachable[name] = nz
        else:
            zero_grad[name] = int(p.numel())

    trainer.opt.zero_grad(set_to_none=True)
    prefixes = sorted({n.split(".")[0] + "." + n.split(".")[1] if "." in n else n
                       for n in reachable})
    enc = {k: v for k, v in scale.items() if k.startswith("encoder.")}
    head_elems = sum(v for k, v in reachable.items() if k.startswith("head."))
    d_elems = sum(v for k, v in reachable.items() if k.startswith("discriminator."))
    return {
        "n_blocks": int(trainer.cfg.n_blocks),
        "lambda_used": float(lambda_),
        "reachable_param_elements": int(sum(reachable.values())),
        "reachable_tensors": len(reachable),
        "reachable_by_tensor": reachable,
        "zero_grad_tensors": zero_grad,
        "no_grad_tensors": len(none_grad),
        "reachable_prefixes": prefixes,
        "grad_scale_by_tensor": scale,
        "head_elements_reachable": int(head_elems),
        "discriminator_elements_reachable": int(d_elems),
        "encoder_elements_with_grad": int(sum(v["numel"] for v in enc.values())),
        "encoder_elements_exactly_zero": int(sum(v["n_exact_zero"] for v in enc.values())),
        "encoder_elements_reachable": int(sum(v for k, v in reachable.items()
                                              if k.startswith("encoder."))),
        "layernorm_gain_rms": scale.get("head.trunk.0.weight", {}).get("rms"),
        "layernorm_bias_rms": scale.get("head.trunk.0.bias", {}).get("rms"),
        "domain_loss_value": float(da.detach()),
        "d_probed_in_eval_mode": True,
        "d_probed_in_eval_note": ("dropout OFF for the probe, so the reachable-D count "
                                  "is a property of the graph rather than of one random "
                                  "mask. Encoder and head reachability is unaffected."),
    }


# ---------------------------------------------------------------------------
# 5. the per-step diagnostics window  (mirrors coral_round4.CoralStepWindow)
# ---------------------------------------------------------------------------

DANN_STEP_KEYS = (
    "sup_loss",                     # supervised BCE on the SOURCE weak view
    "da_loss",                      # what actually entered the total loss (== raw, weight 1)
    "loss_domain_raw",              # the unweighted domain BCE
    "lambda_current",               # the GRL coefficient THIS step
    "ratio_lambda_domain_over_sup",  # objective balance: lambda * L_domain / L_sup
    # ---- the discriminator, three columns that must be read together ----
    "d_acc_train",                  # D accuracy on the training batch
    "d_acc_holdout",                # D accuracy on the held-out target slice
    "d_pred_source_frac",           # NOT derivable from accuracy -- see DannLoss
    "d_acc_source", "d_acc_target", "d_logit_mean", "d_holdout_logit_mean",
    # ---- representation: SCALE (7.15's corrected indicator) ----
    "src_feat_l2", "src_perdim_std", "tgt_feat_l2", "tgt_perdim_std",
    # ---- representation: RANK (distinct from scale -- 7.15) ----
    "src_participation_ratio", "src_dims_at_90", "src_dims_at_99", "src_total_var",
    "tgt_participation_ratio", "tgt_dims_at_90", "tgt_dims_at_99", "tgt_total_var",
    # ---- morphology, TWO INSTRUMENTS on one ML-FP vs P1-reserve pool ----
    # morph_head_auroc  : the TRAINED HEAD's own lens logit. The ~0.61-scale
    #                     instrument, the one that MOVED in the v5 failure, and the
    #                     one abort condition 3 reads (PROJECT.md 7.17).
    # morph_proxy_*     : the ridge probe. NO detector attached. It is what makes a
    #                     head drop interpretable -- head down + probe held = ACCESS
    #                     destroyed; both down = the REPRESENTATION went.
    "morph_head_auroc",               # eval half, paired with the probe column
    "morph_head_auroc_pool",          # whole pool: lower variance, unpaired
    "morph_proxy_auroc",              # on the adapter: what the classifier reads
    "morph_proxy_auroc_encoder",      # on raw encoder features: the historical scale
    "d_holdout_measured_in_eval",
    # ---- abort condition 3's raw difference, logged EVERY monitor step whether or
    # not anything fires (brief v3 3). NaN when there is no control curve, or when
    # the control has no row at this exact step -- never interpolated. ----
    "morph_minus_control", "control_morph_proxy_auroc",
    # ---- fp16 health (7.10: lost updates and "it did not help" look identical) ----
    "step_skipped", "amp_scale",
)

DANN_LAST_KEYS = ("loss_domain_raw", "lambda_current", "d_acc_train", "d_acc_holdout",
                  "src_perdim_std", "tgt_perdim_std", "morph_head_auroc",
                  "morph_proxy_auroc", "morph_minus_control", "amp_scale")

# The abort streak/fired columns ride in the PER-STEP csv only. They are counters,
# not measurements, so a window MEAN of them would be meaningless -- the trace carries
# the streaks' `_last` value through DANN_LAST_KEYS instead.
# CONDITIONS 3 AND 4 ARE ADVISORY ON EVERY ARM, INCLUDING CONFIRMATORY.
#
# They were built as SAFETY aborts -- stop burning GPU on a run producing garbage. That
# job is done: DANN's nb3 behaviour is characterised on fold 0 by arms 1 and 2, and the
# probe drop is real, early, sustained and now EXPECTED. There is no uncertainty left for
# an abort to protect against, and 3b at sustain 0 would abort a confirmatory fold at
# step 451 -- destroying the TPR@1e-3 that fold exists to produce, to re-establish a fact
# already in hand.
#
# This is STRICTLY LESS INTERVENTION than the pre-registered arrangement. The detectors
# still compute, still log every column, and still record when they would have fired; all
# that changes is that they no longer stop a run. The reported metric is unchanged, and a
# detector that cannot halt anything cannot bias a result toward any value -- the failure
# mode a post-hoc threshold change usually carries (choosing which runs survive) is not
# available to it.
#
# Conditions 1 and 2 STAY ARMED. They catch the adversarial game genuinely breaking, they
# are cheap, and on arm 2 they were nowhere near firing (longest streaks 5 and 9 steps
# against a 500-step sustain), so arming them costs nothing and covers the one failure
# these runs have not yet demonstrated they are immune to.
ADVISORY_ONLY_DETECTORS = ("abort3_morph_vs_control", "abort3b_morph_sign_flip",
                           "abort4src_rank_collapse", "abort4tgt_rank_collapse")

ABORT_DETECTOR_NAMES = ("abort1_d_overpowering", "abort2_degenerate_collapse",
                        "abort3_morph_vs_control", "abort3b_morph_sign_flip",
                        "abort4src_rank_collapse", "abort4tgt_rank_collapse")

STEP_COLUMNS = (["step"] + list(DANN_STEP_KEYS)
                + ["src_pr_minus_control", "tgt_pr_minus_control"]
                + [c for d in ABORT_DETECTOR_NAMES
                   for c in (f"{d}_streak", f"{d}_fired")])

TRACE_COLUMNS = (
    ["step", "n_steps_in_window", "wall_secs"]
    + [f"{k}_mean" for k in DANN_STEP_KEYS]
    + [f"{k}_last" for k in DANN_LAST_KEYS]
    + ["domain_over_sup", "src_l2_over_unit_gain", "d_train_minus_holdout"]
)

CHECKPOINT_COLUMNS = [
    "step", "wall_secs", "score_secs", "n_steps_in_window",
    "sup_loss_mean", "loss_domain_raw_mean", "lambda_current_mean",
    "d_acc_train_mean", "d_acc_holdout_mean", "d_pred_source_frac_mean",
    "src_feat_l2_mean", "src_perdim_std_mean", "tgt_feat_l2_mean", "tgt_perdim_std_mean",
    "src_participation_ratio_mean", "src_dims_at_90_mean", "src_dims_at_99_mean",
    "tgt_participation_ratio_mean", "tgt_dims_at_90_mean", "tgt_dims_at_99_mean",
    "ratio_lambda_domain_over_sup_mean",
    "morph_head_auroc_mean", "morph_head_auroc_pool_mean",
    "morph_proxy_auroc_mean", "morph_proxy_auroc_encoder_mean",
    "morph_minus_control_mean",
    "loss_domain_raw_last", "lambda_current_last", "morph_proxy_auroc_last",
    "amp_scale_last",
    "skipped_cumulative", "logit_file",
]


def _num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class DannStepWindow:
    """Accumulates per-step DANN statistics between two trace points.

    Window MEANS for the same reason ``coral_round4.CoralStepWindow`` uses them: the
    per-batch quantities are noisy on 256 draws from a ~700-image target pool, and
    every question these monitors answer ("is D memorising?", "is the representation
    shrinking?") is about a trend.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._acc = {k: [] for k in DANN_STEP_KEYS}
        self._last = {k: float("nan") for k in DANN_STEP_KEYS}
        self.n = 0

    def update(self, stats: dict):
        for k in DANN_STEP_KEYS:
            v = stats.get(k)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            self._last[k] = v
            if math.isfinite(v):
                self._acc[k].append(v)
        self.n += 1

    def mean(self, key):
        a = self._acc.get(key) or []
        return float(np.mean(a)) if a else float("nan")

    def last(self, key):
        return self._last.get(key, float("nan"))

    def row(self, step: int) -> dict:
        row = {"step": int(step), "n_steps_in_window": int(self.n)}
        for k in DANN_STEP_KEYS:
            row[f"{k}_mean"] = _num(self.mean(k))
        for k in DANN_LAST_KEYS:
            row[f"{k}_last"] = _num(self.last(k))
        sup, dom = self.mean("sup_loss"), self.mean("loss_domain_raw")
        row["domain_over_sup"] = (_num(dom / sup)
                                  if (math.isfinite(sup) and sup > 0) else None)
        l2 = self.mean("src_feat_l2")
        row["src_l2_over_unit_gain"] = (_num(l2 / math.sqrt(640.0))
                                        if math.isfinite(l2) else None)
        tr, ho = self.mean("d_acc_target"), self.mean("d_acc_holdout")
        row["d_train_minus_holdout"] = (_num(tr - ho)
                                        if (math.isfinite(tr) and math.isfinite(ho))
                                        else None)
        return row


# ---------------------------------------------------------------------------
# 6. an adversarial-health verdict, reported as a TRACE (never as a gate)
# ---------------------------------------------------------------------------

# PROJECT.md 7.12's rule for the collapse gate applies unchanged here: log the
# trace, never the verdict. These anchors say what the numbers MEAN; they do not
# stop a run.
DANN_HEALTH_ANCHORS = {
    "d_acc_train_at_chance": 0.5,
    "d_acc_train_note": ("0.5 is the adversarial equilibrium AND what a collapsed D "
                         "that predicts one domain scores when the batch is balanced. "
                         "d_pred_source_frac separates them: ~0.5 at equilibrium, ~0 or "
                         "~1 on collapse. Never read d_acc_train alone."),
    "memorisation_note": ("d_acc_target (train batch) minus d_acc_holdout. A persistent "
                          "positive gap is D separating images it trains on from "
                          "statistically identical images it does not -- i.e. the domain "
                          "'signal' it is teaching the encoder to remove is partly "
                          "memorised identity, not domain shift."),
    "round1_dann_reference": ("runs/curve_dann_fold0 -- CACHED-FEATURE, bottleneck-adapter, "
                              "BatchNorm D, lambda from run length. NOT comparable "
                              "step-for-step; kept as a sanity range only."),
    "feat_std_reference_round4_coral": ("nb3 CORAL gamma=16: per-dimension std fell ~7.7x "
                                        "while mean ||f|| moved 2.5%. If DANN shrinks the "
                                        "representation the same way, per-dim std is where "
                                        "it shows and ||f|| is where it does not."),
    "morph_proxy_note": ("ML-FP vs P1-reserve linear-probe AUROC (morph_proxy.py), the "
                         "RECOVERED instrument that caught the original DANN failure. "
                         "Step-0 reading and a PROPOSED abort level are in "
                         "runs/dann_r5_morph_step0.json -- do NOT inherit the design "
                         "doc's absolute 0.55, which belongs to a different instrument "
                         "with a 0.61 baseline."),
    "morph_proxy_train_eval_note": ("d_acc_train is measured in TRAIN mode (dropout on, "
                                    "as trained) and d_acc_holdout in EVAL mode (dropout "
                                    "off, so the memorisation monitor is not carrying a "
                                    "random mask). The pair is therefore not a "
                                    "like-for-like difference; read each column's TREND."),
}


# ---------------------------------------------------------------------------
# 7. the pre-registered abort conditions  (brief v3)
# ---------------------------------------------------------------------------
#
# Four conditions, from ROUND5_DANN_DESIGN.md 6, with condition 3 REPLACED by brief
# v3's control-referenced form. Each is a STREAK detector measured in ABSOLUTE STEPS,
# not in monitor samples -- ``--morph-every 25`` means condition 3 sees ~20 readings
# across its 500-step window, and a counter of readings would silently mean something
# different at a different cadence.
#
# ACTION IS GATED ON THE ARM, and the gate is a RECORDED SETTING, never inferred:
#
#   development (fold 0, 8000 steps)  LOG AND CONTINUE.  If DANN destroys classifier
#                                     access, that IS the paper result and the full
#                                     curve is the evidence. Aborting at step 1200
#                                     throws away exactly what you would be aborting
#                                     because you found.
#   lambda=0 control                  LOG AND CONTINUE.  It is the reference trajectory
#                                     for every monitor (brief v3 5) and must always
#                                     complete.
#   confirmatory (folds 1-4)          ABORT ON FIRE.  The trace has no scientific value
#                                     there and a dead run is wasted T4 time.
#
# WHAT CHANGED IN CONDITION 3, AND WHY IT MATTERS. Earlier briefs stated that round-1
# DANN drove a morphology proxy from 0.61 to 0.33-0.46, and the design doc's "AUROC
# below 0.55" was calibrated on that. **PROJECT.md attributes that failure to
# IMPORTANCE WEIGHTING, not to DANN, and records no numbers for it**; the only
# morphology figure in that part of the document is the frozen-probe baseline at
# ~0.90-0.91 -- the scale morph_proxy.py reproduces at 0.9214. So 0.55 was invented,
# not measured, and any port of it inherits nothing. It is withdrawn. The band is now
# measured against the lambda=0 CONTROL at the same absolute step, which subtracts
# unfreezing-only drift instead of mistaking it for adversarial damage.

ABORT_MODE_LOG_ONLY = "log_only"
ABORT_MODE_ARMED = "armed"

# Declared constants. Each is a judgement call and is labelled as one.
COND1_HOLDOUT_ACC = 0.95            # design doc 6.1
COND1_SUSTAIN = 1000                # design doc 6.1
COND2_MARGINAL_BAND = (0.2, 0.8)    # design doc 6.2
COND2_SUSTAIN = 500                 # design doc 6.2
# =====================================================================
# CONDITIONS 3 AND 4 -- RE-SPECIFIED FROM THE ARM-1 CONTROL (2026-08-28)
# =====================================================================
# NOT a pre-registration. The pre-registered decision tree anticipated 3a being
# uncalibrable with 3b carrying the condition; it did NOT anticipate 3b firing on the
# control. The tree ran out, and what follows is a CONTROL-DERIVED RE-SPECIFICATION,
# recorded as such. It is clean because arm 2 is `log_only`: no detector acts on the
# reported development run, so the thresholds were changed BEFORE any run they could
# influence. They bind on folds 1-4 only.
#
# WHAT ARM 1 SHOWED (runs/dann_arm1, lambda = 0, no adversary at all):
#   morph_head_auroc_pool   0.5657 (step 0) -> median 0.4948, 54% of readings < 0.5
#   morph_proxy_auroc       0.9172 -> 0.9009, median 0.9076
#   src participation ratio 9.07 -> 2.18, run max 9.69   (never above 20)
#   tgt participation ratio 10.40 -> 6.85, run max 13.40 (never above 20)
#
# WHY THE HEAD COLUMN CANNOT BE SALVAGED. ML-FP and P1-reserve are BOTH class 0 in the
# lens task, so the head is fine-tuned to give both low lens logits -- trained to make
# them indistinguishable. At step 0 it is Baseline A's frozen linear probe (1,921
# params, barely able to act) so residual morphology structure leaks through at 0.5657;
# unfreeze 14.8M parameters and the model learns what it is asked to learn and the two
# classes converge. AUROC -> 0.5. That is THE CLASSIFIER SUCCEEDING, not access being
# destroyed, and it was going to happen at nb3 whatever lambda did. Measured J = 0.033
# is about the standard error of an at-chance AUROC at n=256 (Hanley-McNeil ~0.036):
# the column is noise around 0.5 with the signal gone. A degradation cannot be detected
# in a quantity the control has already degraded to chance.
#
# WHY THE FROZEN PROBE IS NOW THE RIGHT INSTRUMENT. It was moved OFF the probe because
# the record says the probe held ~0.90 through the historical failures -- but THOSE
# FAILURES WERE AT nb0. With the encoder frozen the probe measures fixed features and
# was STRUCTURALLY INCAPABLE of moving; "it did not move" was guaranteed, not evidence.
# At nb3 the probe is live, and arm 1 gives it what the head column lacks: a control
# level of 0.9076 with 0.4076 of headroom over chance, and J = 0.0078.
COND3_COLUMN = "morph_proxy_auroc"          # the DETECTOR column (frozen probe, adapter)
COND3_INTERPRETATION_COLUMNS = ("morph_head_auroc_pool", "morph_head_auroc")
# ^ kept logged with NO DETECTOR ATTACHED. Arm 1 showed what they actually measure:
#   whether the model learned the LENS TASK (both probe classes are class 0, so success
#   drives them to chance), NOT whether morphology access survived.

# ---- ceilings are FRACTIONS OF MEASURED HEADROOM, not absolutes ----
# The original 0.04 was 0.6088 of the head column's 0.0657 headroom. Stating the
# fraction rather than the absolute is what makes it travel correctly when the column
# changes -- which it now has, twice.
COND_CEILING_FRACTION = 0.6088
COND3_HEADROOM_FLOOR = 0.5                  # chance, for an AUROC column
COND3_CONTROL_LEVEL = 0.9076                # arm 1 median, morph_proxy_auroc
COND3_HEADROOM = COND3_CONTROL_LEVEL - COND3_HEADROOM_FLOOR          # 0.4076
COND3_M_CEILING = COND_CEILING_FRACTION * COND3_HEADROOM             # 0.2481
COND3_MARGIN = None                 # UNSET. Declared by the run owner, then hashed.
COND3_SUSTAIN = 500
# 3b's sustain. 0 = fire on a single reading below the floor -- correct when 3b was a
# SIGN FLIP at 0.5 (one below-chance reading is decisive), and a DECLARED CHOICE now that
# it is a floor at COND3_SIGN_FLIP. Arm 2 fired it at step 451 and held it for 7,550 of
# 8,000 steps, so an armed fold aborts at 451 unless this is raised deliberately.
COND3B_SUSTAIN = 0

# ---- 3b's absolute floor, RE-DERIVED. 0.5 was the HEAD column's inversion point ----
# On a column whose control sits at 0.9076, a fall to 0.5 is 40 J below the control's
# own minimum -- far past catastrophic, and useless as an early signal. The floor is
# instead HALF THE MEASURED HEADROOM: 0.5 + 0.5 * 0.4076 = 0.7038, rounded down to
# 0.70. Chosen because (a) it means something -- half the linear morphology signal
# destroyed; (b) it is stated as a fraction of measured headroom, so it travels; and
# (c) arm 1's minimum reading was 0.8108, which sits ~13 J above it, so the control
# never came close. A DECLARED judgement call, derived from the control, not inherited.
COND3_FLOOR_FRACTION = 0.5
COND3_SIGN_FLIP = 0.70              # name kept for schema stability; it is a FLOOR now

# ---- CONDITION 4, RE-SPECIFIED: DIFFERENTIAL, and src/tgt SEPARATELY ----
# The absolute "PR < 20 of 640" was ported from the SimSiam finding in a DIFFERENT
# feature space. The LP->FT init sits at PR ~9-10 in this adapter space before a single
# step, and the condition fired 7,999 of 8,000 times on the control. That is 7.15's
# monitor-design error repeated: an indicator's threshold set without checking what the
# indicator reads on the model it monitors.
#
# Replaced by the same differential form as 3a -- PR below the lambda=0 arm's value AT
# THE SAME ABSOLUTE STEP by more than a declared margin, sustained. src and tgt are
# separate detectors because arm 1 showed them behaving differently (src 9.07 -> 2.18,
# tgt 10.40 -> 6.85); one combined threshold would hide that.
#
# Headroom for a PR column is measured to 1.0, not to chance: a fully collapsed
# representation has all its variance in one direction, PR = 1.
COND4_ABSOLUTE_PR_FLOOR_WITHDRAWN = 20.0    # kept ONLY so the record says what it was
COND4_PR_FLOOR = 1.0                        # the collapse limit, for headroom maths
COND4_SRC_CONTROL_LEVEL = 5.2582            # arm 1 median, src_participation_ratio
COND4_TGT_CONTROL_LEVEL = 10.2114           # arm 1 median, tgt_participation_ratio
COND4_SRC_CEILING = COND_CEILING_FRACTION * (COND4_SRC_CONTROL_LEVEL - COND4_PR_FLOOR)
COND4_TGT_CEILING = COND_CEILING_FRACTION * (COND4_TGT_CONTROL_LEVEL - COND4_PR_FLOOR)
COND4_SRC_MARGIN = None             # UNSET. Declared by the run owner, then hashed.
COND4_TGT_MARGIN = None
COND4_SUSTAIN = 500                 # DECLARED ADDITION -- the design doc gives none


class StreakDetector:
    """One abort condition, as a streak measured in ABSOLUTE STEPS.

    ``update(step, stats)`` is called once per training step. A detector whose input
    column is not fresh this step (``morph_proxy_auroc`` at ``--morph-every 25``)
    returns ``None`` and neither extends nor resets the streak -- so a sparse monitor
    measures the same 500-step window a dense one does.

    THE STREAK IS A SPAN, NOT A COUNT. ``streak_steps = step - first_violating_step``.
    A counter of violating SAMPLES would mean "500 steps" at cadence 1 and "12,500
    steps" at cadence 25, which is the class of silent unit error PROJECT.md 7.10
    records for FreeMatch's warmup.
    """

    def __init__(self, name, sustain, description, *, action_note=""):
        self.name = str(name)
        self.sustain = int(sustain)
        self.description = str(description)
        self.action_note = str(action_note)
        self.reset()

    def reset(self):
        self.start_step = None
        self.last_step = None
        self.n_samples = 0
        self.fired = False
        self.fired_at = None
        self.detail = {}
        self.n_evaluated = 0
        self.max_streak_steps = 0

    @property
    def streak_steps(self) -> int:
        if self.start_step is None or self.last_step is None:
            return 0
        return int(self.last_step - self.start_step)

    def observe(self, step, violating: bool | None, detail=None):
        """Feed one evaluation. ``None`` = no fresh reading; leaves the streak alone."""
        if violating is None:
            return None
        self.n_evaluated += 1
        if violating:
            if self.start_step is None:
                self.start_step = int(step)
                self.n_samples = 0
            self.last_step = int(step)
            self.n_samples += 1
            self.detail = dict(detail or {})
            self.max_streak_steps = max(self.max_streak_steps, self.streak_steps)
            if not self.fired and self.streak_steps >= self.sustain:
                self.fired = True
                self.fired_at = int(step)
        else:
            self.start_step = None
            self.last_step = None
            self.n_samples = 0
        return self.fired

    def row(self) -> dict:
        return {f"{self.name}_streak": self.streak_steps,
                f"{self.name}_fired": 1.0 if self.fired else 0.0}

    def summary(self) -> dict:
        return {"name": self.name, "sustain_steps": self.sustain,
                "description": self.description, "action_note": self.action_note,
                "fired": bool(self.fired), "fired_at_step": self.fired_at,
                "max_streak_steps": int(self.max_streak_steps),
                "n_evaluations": int(self.n_evaluated),
                "streak_open_at_end": self.streak_steps,
                "detail_at_fire_or_last": self.detail}


class ControlCurve:
    """The lambda=0 arm's per-step series, read by ABSOLUTE STEP.

    Brief v3 3 calls the file ``curve.csv``. **This path does not write one** -- the
    round-5 runner writes ``steps.csv`` (per step), ``trace.csv`` (per ``--log-every``)
    and ``checkpoints.csv``, mirroring ``run_coral_r4_dev8000.py``. Any of the three is
    accepted here and the column is looked up by name, so the disagreement is a file
    name and nothing else; ``steps.csv`` is the one to pass, because it is the only one
    with a row at every absolute step.

    THE COLUMN IS ``morph_head_auroc``, not ``morph_proxy_auroc``. Condition 3 compares
    the two arms' HEAD instrument; the frozen probe is logged on both arms with no
    detector attached.

    NEVER INTERPOLATES. If the development run measures morphology at a step the
    control has no row for, :meth:`at` returns ``None`` and the comparison is refused
    for that step, loudly. Interpolating would invent a reference value and then
    compare a real number against it.
    """

    EXTRA_COLUMNS = ("src_participation_ratio", "tgt_participation_ratio")

    def __init__(self, path, column=COND3_COLUMN, extra_columns=EXTRA_COLUMNS):
        self.path = str(path)
        self.column = str(column)
        wanted = [self.column] + [c for c in (extra_columns or ()) if c != self.column]
        self.columns = {}
        with open(self.path, newline="") as fh:
            rdr = csv.DictReader(fh)
            fields = rdr.fieldnames or []
            if self.column not in fields:
                raise SystemExit(
                    f"[dann-r5] ABORT: the control curve {self.path} has no column "
                    f"{self.column!r} (has {fields}).\n"
                    f"  Condition 3 compares against the lambda=0 arm step for step; a "
                    f"control without the column cannot be the reference.")
            have = [c for c in wanted if c in fields]
            self.missing_columns = [c for c in wanted if c not in fields]
            for c in have:
                self.columns[c] = {}
            for r in rdr:
                try:
                    step = int(float(r["step"]))
                except (TypeError, ValueError, KeyError):
                    continue
                for c in have:
                    try:
                        v = float(r[c])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if math.isfinite(v):
                        self.columns[c][step] = v
        self.values = self.columns.get(self.column, {})
        if not self.values:
            raise SystemExit(
                f"[dann-r5] ABORT: the control curve {self.path} has no finite "
                f"{self.column!r} values. Was the control run with --morph-every set so "
                f"high that nothing landed, or with the monitor off?")
        self.steps = sorted(self.values)

    def at(self, step, column=None):
        """The control's value at EXACTLY ``step``, or None. Never interpolates.

        ``column`` selects one of the other series the control carries -- condition 4
        reads the two participation-ratio columns through the same object, with the
        same absolute-step lookup and the same refusal to interpolate.
        """
        src = self.values if column is None else self.columns.get(column, {})
        return src.get(int(step))

    def covers(self, steps) -> list:
        """Which of ``steps`` the control has no row for."""
        return [int(s) for s in steps if int(s) not in self.values]

    def summary(self) -> dict:
        return {"path": self.path, "column": self.column,
                "columns_loaded": sorted(self.columns),
                "columns_missing": list(getattr(self, "missing_columns", [])),
                "n_steps": len(self.values),
                "step_min": self.steps[0], "step_max": self.steps[-1],
                "stride_mode": _mode_stride(self.steps),
                "no_interpolation": ("values are read at EXACTLY the requested step; a "
                                     "missing step refuses the comparison rather than "
                                     "inventing a reference")}


def banner(title, lines, char="!"):
    """Same shape as ``run_coral_r4_dev8000.banner``, defined locally.

    Deliberately NOT imported from there: that module pulls in the whole round-4 CORAL
    stack, and this one is imported by the reachability probe and by the tests, which
    should not depend on another round's runner to print a box.
    """
    w = 92
    print("\n" + char * w, flush=True)
    print(f"{char * 3} {title}")
    for ln in lines:
        print(f"{char * 3} {ln}")
    print(char * w + "\n", flush=True)


def _mode_stride(steps):
    if len(steps) < 2:
        return None
    d = {}
    for a, b in zip(steps, steps[1:]):
        d[b - a] = d.get(b - a, 0) + 1
    return max(d, key=d.get)


class DannAbortWatch:
    """The four pre-registered conditions, evaluated every step.

    ``mode`` is ``log_only`` or ``armed`` and is a RECORDED SETTING, not something
    derived at read time (brief v3 4.2). In ``log_only`` a fire is banner-printed and
    recorded and the run continues; in ``armed`` :meth:`should_abort` goes True and the
    caller stops after writing a final scorable checkpoint.
    """

    def __init__(self, mode=ABORT_MODE_LOG_ONLY, *, control: "ControlCurve | None" = None,
                 cond3_margin: float | None = COND3_MARGIN,
                 cond4_sustain: int = COND4_SUSTAIN, cond3_median_window: int = 1,
                 cond4_src_margin: float | None = COND4_SRC_MARGIN,
                 cond4_tgt_margin: float | None = COND4_TGT_MARGIN,
                 cond3b_sustain: int = COND3B_SUSTAIN,
                 verbose: bool = True):
        if mode not in (ABORT_MODE_LOG_ONLY, ABORT_MODE_ARMED):
            raise ValueError(f"abort mode must be {ABORT_MODE_LOG_ONLY!r} or "
                             f"{ABORT_MODE_ARMED!r}, got {mode!r}")
        # AN ARMED RUN WITH AN UNSET MARGIN IS REFUSED AT CONSTRUCTION, which is before
        # the first optimizer step. Arms 1 and 2 are log_only and run BEFORE the margin
        # exists -- that is the point of them -- so the refusal is scoped to `armed`.
        # A confirmatory fold that silently ran with condition 3a disabled would be a
        # run whose pre-registration says it was monitored and which was not.
        # The refusal covers margins belonging to detectors that CAN abort. Conditions 3
        # and 4 are advisory on every arm now, so an unset margin there disables a
        # LOGGING line, not a safety net -- refusing to start over it would be theatre.
        # It is still WARNED about, because an advisory detector silently not evaluating
        # is how a monitor comes to be trusted for something it never did.
        _arming_margins = ()          # conditions 1 and 2 take no margin
        unset = [n for n, v in _arming_margins if v is None]
        _unset_advisory = [n for n, v in (("condition 3's margin M", cond3_margin),
                                          ("condition 4's src margin", cond4_src_margin),
                                          ("condition 4's tgt margin", cond4_tgt_margin))
                           if v is None]
        if verbose and _unset_advisory:
            print(f"[abort] ADVISORY detectors with an UNSET margin (they will not "
                  f"evaluate, and they could not abort anything even if they did): "
                  f"{_unset_advisory}", flush=True)
        if mode == ABORT_MODE_ARMED and unset:
            raise SystemExit(
                f"[dann-r5] ABORT: --abort-mode armed with {len(unset)} margin(s) UNSET: "
                f"{unset}.\n"
                f"  Conditions 3a and 4 are BOTH differential against the lambda=0 arm, "
                f"and both margins are declared off arm 1's own trajectories and hashed "
                f"before the confirmatory folds (PROJECT.md 7.17). They are deliberately "
                f"not defaulted.\n"
                f"  Admissible bands measured from arm 1: cond3 M in [0.0234, "
                f"{COND3_M_CEILING:.4f}]; cond4 src in [0.5175, {COND4_SRC_CEILING:.4f}]; "
                f"cond4 tgt in [1.3263, {COND4_TGT_CEILING:.4f}].\n"
                f"  Pass --abort-cond3-margin / --abort-cond4-src-margin / "
                f"--abort-cond4-tgt-margin with the hashed values, or run --abort-mode "
                f"log_only. Condition 3b (floor {COND3_SIGN_FLIP}) needs no margin and is "
                f"active either way.")
        if False and mode == ABORT_MODE_ARMED and cond3_margin is None:
            raise SystemExit(
                "[dann-r5] ABORT: --abort-mode armed with abort condition 3's margin "
                "UNSET.\n"
                "  The margin is declared off arm 1's lambda=0 control trajectory on "
                "the morph_head_auroc column and hashed BEFORE arm 2 starts (PROJECT.md "
                "7.17). It is deliberately not defaulted: brief v3's 0.10 was declared "
                "against the FROZEN PROBE (step 0 = 0.9202, headroom 0.4202) and is "
                "scale-wrong for the HEAD column (step 0 ~= 0.5781, headroom ~0.078), "
                "where a 0.10 drop lands below chance.\n"
                "  Pass --abort-cond3-margin <M> with the hashed value, or run "
                "--abort-mode log_only. Condition 3b (sign flip, AUROC < 0.5) needs no "
                "margin and is active either way.")
        self.mode = mode
        self.control = control
        self.cond3_margin = (None if cond3_margin is None else float(cond3_margin))
        # THE SMOOTHING WINDOW, and the ORDER IN WHICH IT IS FIXED.
        #   k = 1 is no smoothing and is the default. k > 1 makes 3a threshold the
        #   TRAILING MEDIAN OF THE LAST k PAIRED DIFFERENCES rather than the
        #   instantaneous one.
        #   Smoothing the DIFFERENCE, not each side separately: it is the difference 3a
        #   thresholds, and smoothing both sides independently would need an assumption
        #   about the control's own smoothness that nothing supports.
        #   **k MUST BE DECLARED BEFORE M** (PROJECT.md 7.17). A window chosen after a
        #   candidate M is on the table has been selected to make that M work.
        #   With k > 1 the window must be FULL before 3a evaluates at all, so the
        #   earliest possible fire moves later by (k-1) * morph_every steps -- 175 steps
        #   at k=8, morph_every=25, against a 500-step sustain.
        self.cond3_median_window = max(1, int(cond3_median_window))
        self.cond3b_sustain = max(0, int(cond3b_sustain))
        self._delta_window = []
        self.cond4_src_margin = (None if cond4_src_margin is None
                                 else float(cond4_src_margin))
        self.cond4_tgt_margin = (None if cond4_tgt_margin is None
                                 else float(cond4_tgt_margin))
        self.last_pr_delta = {"src": float("nan"), "tgt": float("nan")}
        self.verbose = bool(verbose)
        self.missing_control_steps = []
        self.last_delta = float("nan")
        self.domain_loss_window = []          # (step, loss) for condition 1's second half

        self.c1 = StreakDetector(
            "abort1_d_overpowering", COND1_SUSTAIN,
            f"held-out-target domain accuracy > {COND1_HOLDOUT_ACC} for "
            f"{COND1_SUSTAIN} consecutive steps WITH the domain loss not falling",
            action_note=("design doc 6.1: this is the SDAT AdamW failure, and it is the "
                         "one condition that reverses the DANN-over-ADDA decision. "
                         "Action if it fires: switch to ADDA, whose separate "
                         "discriminator optimiser sidesteps it."))
        self.c2 = StreakDetector(
            "abort2_degenerate_collapse", COND2_SUSTAIN,
            f"d_pred_source_frac outside {COND2_MARGINAL_BAND} for {COND2_SUSTAIN} "
            f"consecutive steps",
            action_note=("design doc 6.2: D has collapsed to predicting one domain. "
                         "d_acc_train near 0.5 is then NOT evidence of alignment, which "
                         "is why the marginal is logged separately from accuracy."))
        self.c3 = StreakDetector(
            "abort3_morph_vs_control", COND3_SUSTAIN,
            f"{COND3_COLUMN} below the lambda=0 control's value AT THE SAME ABSOLUTE "
            f"STEP by more than "
            f"{'UNSET -- 3a INACTIVE' if cond3_margin is None else cond3_margin}, "
            f"for {COND3_SUSTAIN} consecutive steps",
            action_note=("SECONDARY / graded early warning. Adversarial pressure has "
                         "degraded linear access to morphology RELATIVE TO UNFREEZING "
                         "ALONE. Step-matched against the control so unfreezing-only "
                         "drift is subtracted rather than mistaken for adversarial "
                         "damage. This is a REPORTABLE RESULT, not a bug."))
        self.c3b = StreakDetector(
            "abort3b_morph_sign_flip", int(cond3b_sustain),
            f"{COND3_COLUMN} < {COND3_SIGN_FLIP} -- HALF the linear morphology signal "
            f"the lambda=0 control retains has been destroyed"
            + (f", for {int(cond3b_sustain)} consecutive steps"
               if int(cond3b_sustain) else " (fires on a SINGLE reading)"),
            action_note=(
                "**PRIMARY DETECTOR.** Name kept for schema stability; it is a FLOOR "
                "now, not a sign flip. The 0.5 inversion point belonged to the "
                "WITHDRAWN head column and sits ~40 J below this column's control "
                "minimum -- far past catastrophic. The floor is half the measured "
                "headroom (0.5 + 0.5*0.4076 = 0.7038, rounded down); arm 1's minimum "
                "was 0.8108, ~13 J above it.\n"
                "  SUSTAIN: this detector was written with sustain 0 -- correct for a "
                "sign flip, where one reading below chance is decisive. It is NOT "
                "obviously correct for a floor. On arm 2 it first fired at step 451 and "
                "held for 7,550 of 8,000 steps, so an ARMED fold would abort at 451. "
                "Whether that is the right behaviour is a DECLARED CHOICE, not a "
                "default: if the destruction is real and sustained, aborting IS the "
                "finding. Set --abort3b-sustain deliberately before arming."))
        def _c4(tag, ceiling, margin):
            return StreakDetector(
                f"abort4{tag}_rank_collapse", int(cond4_sustain),
                f"{tag}_participation_ratio below the lambda=0 control's value AT THE "
                f"SAME ABSOLUTE STEP by more than "
                f"{'UNSET -- INACTIVE' if margin is None else margin}, for "
                f"{int(cond4_sustain)} consecutive steps",
                action_note=(
                    "RE-SPECIFIED FROM THE CONTROL (2026-08-28). The absolute 'PR < 20 "
                    "of 640' was ported from the SimSiam finding in a different feature "
                    "space and fired 7,999 of 8,000 times on arm 1 -- the LP->FT init "
                    "sits at PR ~9-10 before a single step and the run never exceeded "
                    "13.40. Now differential against the control, src and tgt SEPARATELY "
                    "because arm 1 showed them behaving differently (src 9.07 -> 2.18, "
                    f"tgt 10.40 -> 6.85). Admissible band from arm 1: up to {ceiling:.4f}. "
                    "DANN's characteristic failure is collapse to a low-dimensional "
                    "domain-invariant subspace; this now measures that RELATIVE TO "
                    "UNFREEZING ALONE."))

        self.c4src = _c4("src", COND4_SRC_CEILING, self.cond4_src_margin)
        self.c4tgt = _c4("tgt", COND4_TGT_CEILING, self.cond4_tgt_margin)
        self.all = (self.c1, self.c2, self.c3, self.c3b, self.c4src, self.c4tgt)
        self._announced = set()

    # -- the per-step evaluation ------------------------------------------

    def update(self, step: int, stats: dict) -> dict:
        """Evaluate all four conditions. Returns the columns to log this step."""
        # ---- condition 1: D overpowering ----
        raw = _num(stats.get("loss_domain_raw"))
        if raw is not None:
            self.domain_loss_window.append((int(step), raw))
            if len(self.domain_loss_window) > 4 * COND1_SUSTAIN:
                del self.domain_loss_window[:COND1_SUSTAIN]
        acc_h = _num(stats.get("d_acc_holdout"))
        if acc_h is None:
            self.c1.observe(step, None)
        else:
            over = acc_h > COND1_HOLDOUT_ACC
            # THE SECOND HALF OF THE CONDITION, and it is not decoration. A D that is
            # at 0.99 while its loss is still falling is a D that is WINNING, which is
            # ordinary early-training behaviour. The failure is a D that is pinned high
            # while the objective has stopped moving.
            falling = self._domain_loss_falling(step)
            self.c1.observe(step, bool(over and not falling),
                            {"d_acc_holdout": acc_h, "domain_loss_falling": falling})

        # ---- condition 2: degenerate collapse ----
        frac = _num(stats.get("d_pred_source_frac"))
        lo, hi = COND2_MARGINAL_BAND
        self.c2.observe(step, None if frac is None else not (lo <= frac <= hi),
                        {"d_pred_source_frac": frac})

        # ---- condition 3: the HEAD column vs the lambda=0 control, step-matched ----
        morph = _num(stats.get(COND3_COLUMN))
        delta, ctrl = None, None
        if morph is None:
            self.c3.observe(step, None)
            self.c3b.observe(step, None)
        else:
            # 3b FIRST and unconditionally: it is the primary detector and it needs
            # neither a control nor a margin.
            self.c3b.observe(step, morph < COND3_SIGN_FLIP, {COND3_COLUMN: morph})
            if self.cond3_margin is None:
                # 3a REFUSES TO EVALUATE while the margin is unset. Not "passes" --
                # refuses. A detector that silently scores every step against a missing
                # threshold is a detector that reports healthy forever.
                self.c3.observe(step, None)
            elif self.control is None:
                self.c3.observe(step, None)          # no reference -> no comparison
            else:
                ctrl = self.control.at(step)
                if ctrl is None:
                    # NEVER INTERPOLATE. Refuse the comparison for this step and say so.
                    self.missing_control_steps.append(int(step))
                    if len(self.missing_control_steps) <= 3:
                        print(f"[abort3] NO CONTROL ROW AT STEP {step}: the lambda=0 arm "
                              f"has no morph reading there, so the comparison is REFUSED "
                              f"for this step. Not interpolated -- a reference value that "
                              f"was invented is not a reference. Check that both arms ran "
                              f"the same --morph-every.", flush=True)
                    self.c3.observe(step, None)
                else:
                    delta = morph - ctrl
                    self._delta_window.append(delta)
                    if len(self._delta_window) > self.cond3_median_window:
                        del self._delta_window[0]
                    if len(self._delta_window) < self.cond3_median_window:
                        # partial window: 3a does not evaluate yet. Not "passes".
                        self.c3.observe(step, None)
                    else:
                        stat = float(np.median(self._delta_window))
                        self.c3.observe(step, stat < -self.cond3_margin,
                                        {COND3_COLUMN: morph, "control": ctrl,
                                         "delta": delta, "delta_median_k": stat,
                                         "k": self.cond3_median_window})
        # The RAW difference is logged every monitor step whether or not 3a is
        # evaluating -- including while the margin is unset, which is exactly the
        # trajectory Navneet declares the margin from.
        if delta is None and morph is not None and self.control is not None:
            c = self.control.at(step)
            if c is not None:
                delta, ctrl = morph - c, c
        self.last_delta = delta if delta is not None else float("nan")

        # ---- condition 4: rank collapse, DIFFERENTIAL, src and tgt separately ----
        for tag, det, margin in (("src", self.c4src, self.cond4_src_margin),
                                 ("tgt", self.c4tgt, self.cond4_tgt_margin)):
            colname = f"{tag}_participation_ratio"
            pr = _num(stats.get(colname))
            d4 = None
            if pr is None or margin is None or self.control is None:
                # margin UNSET or no reference -> REFUSE to evaluate, never "pass"
                det.observe(step, None)
            else:
                cref = self.control.at(step, column=colname)
                if cref is None:
                    self.missing_control_steps.append(int(step))
                    det.observe(step, None)
                else:
                    d4 = pr - cref
                    det.observe(step, d4 < -margin,
                                {colname: pr, "control": cref, "delta": d4})
            if d4 is None and pr is not None and self.control is not None:
                c_ = self.control.at(step, column=colname)
                if c_ is not None:
                    d4 = pr - c_
            self.last_pr_delta[tag] = d4 if d4 is not None else float("nan")

        for d in self.all:
            if d.fired and d.name not in self._announced:
                self._announced.add(d.name)
                self._announce(d, step)

        row = {"morph_minus_control": (None if delta is None else delta),
               "control_morph_proxy_auroc": (None if ctrl is None else ctrl),
               "src_pr_minus_control": (None if math.isnan(self.last_pr_delta["src"])
                                        else self.last_pr_delta["src"]),
               "tgt_pr_minus_control": (None if math.isnan(self.last_pr_delta["tgt"])
                                        else self.last_pr_delta["tgt"])}
        for d in self.all:
            row.update(d.row())
        return row

    def _domain_loss_falling(self, step, window=COND1_SUSTAIN) -> bool:
        """Is the raw domain loss still decreasing over the trailing window?

        Compared as two half-window MEANS rather than a slope: the series is noisy per
        batch, and a mean-vs-mean comparison needs no fit and no tolerance argument.
        With fewer than a full window of history the answer is True (= "still falling"),
        which makes condition 1 UNABLE to fire early in a run. That is deliberate: a D
        at 0.99 in the first few hundred steps is a D that is winning, not one that has
        overpowered the optimiser.
        """
        w = [v for s, v in self.domain_loss_window if s > step - 2 * window]
        if len(w) < 2 * window:
            return True
        h = len(w) // 2
        return float(np.mean(w[h:])) < float(np.mean(w[:h]))

    def _announce(self, d, step):
        armed = self.mode == ABORT_MODE_ARMED
        banner(f"ABORT CONDITION FIRED: {d.name}  (step {step})",
               [d.description,
                f"streak {d.streak_steps} steps >= sustain {d.sustain}; detail "
                f"{d.detail}",
                d.action_note,
                "",
                ("MODE = armed: this run STOPS after writing a final scorable "
                 "checkpoint." if armed else
                 "MODE = log_only: the run CONTINUES. On a development or control arm "
                 "the full curve IS the evidence -- aborting here would throw away "
                 "exactly what you would be aborting because you found.")])

    # -- verdicts ----------------------------------------------------------

    def fired(self) -> list:
        """Every detector that has fired, advisory ones included. This is the REPORT."""
        return [d.name for d in self.all if d.fired]

    def fired_arming(self) -> list:
        """Only the detectors that can actually stop a run: conditions 1 and 2."""
        return [d.name for d in self.all
                if d.fired and d.name not in ADVISORY_ONLY_DETECTORS]

    def fired_advisory(self) -> list:
        return [d.name for d in self.all
                if d.fired and d.name in ADVISORY_ONLY_DETECTORS]

    def should_abort(self) -> bool:
        """Conditions 3 and 4 are ADVISORY on every arm and never abort.

        `fired()` still reports them, so a fold whose probe collapsed is not a fold that
        looked healthy -- it is a fold that recorded the collapse and kept running to
        produce the TPR the collapse was going to be interpreted against.
        """
        return self.mode == ABORT_MODE_ARMED and bool(self.fired_arming())

    def summary(self) -> dict:
        return {
            "abort_mode": self.mode,
            "advisory_only_detectors": list(ADVISORY_ONLY_DETECTORS),
            "arming_detectors": [d.name for d in self.all
                                 if d.name not in ADVISORY_ONLY_DETECTORS],
            "advisory_note": (
                "conditions 3 and 4 are ADVISORY ON EVERY ARM, including confirmatory: "
                "they compute, they log, they record when they WOULD have fired, and "
                "they never stop a run. Derived from arms 1 and 2 on 2026-08-30, NOT "
                "pre-registered. They were safety aborts against an uncharacterised "
                "method; the method is now characterised on fold 0, and 3b at sustain 0 "
                "would abort a confirmatory fold at step 451 -- trading the TPR that "
                "fold exists to produce for a fact already in hand. STRICTLY LESS "
                "INTERVENTION: the reported metric is unchanged, and a detector that "
                "cannot halt a run cannot bias a result toward any value. Conditions 1 "
                "and 2 stay ARMED."),
            "fired_arming": self.fired_arming(),
            "fired_advisory": self.fired_advisory(),
            "cond3_column": COND3_COLUMN,
            "cond3_interpretation_columns": list(COND3_INTERPRETATION_COLUMNS),
            "cond3_column_note": (
                f"BOTH DETECTORS read {COND3_COLUMN} -- the FROZEN-FEATURE ridge probe, "
                f"re-specified from arm 1 (PROJECT.md 7.17). 3a is probe-to-control "
                f"across arms at matched absolute steps; 3b is probe-to-a-floor. "
                f"The two HEAD columns stay logged with NO DETECTOR: arm 1 showed they "
                f"measure whether the model learned the LENS TASK (both probe classes "
                f"are class 0, so success drives them to chance), not whether morphology "
                f"access survived."),
            "cond3b_sustain": int(self.cond3b_sustain),
            "cond3b_sustain_note": (
                "0 = fire on a single reading below the floor. That was correct when 3b "
                "was a SIGN FLIP at 0.5 (one below-chance reading is decisive) and is a "
                "DECLARED CHOICE now that 3b is a floor at "
                f"{COND3_SIGN_FLIP}: on arm 2 it fired at step 451 and held 7,550 of "
                "8,000 steps, so an armed fold aborts at 451."),
            "cond3_median_window_k": self.cond3_median_window,
            "cond3_median_window_note": (
                "k=1 is no smoothing. k>1 thresholds the trailing median of the last k "
                "PAIRED DIFFERENCES (not each side separately). k MUST BE DECLARED "
                "BEFORE M: a window chosen after a candidate M is on the table has been "
                "selected to make that M work."),
            "cond3_margin": self.cond3_margin,
            "cond3_margin_status": (
                "UNSET -- condition 3a is INACTIVE and refuses to evaluate"
                if self.cond3_margin is None else
                f"SET to {self.cond3_margin}; declared off arm 1's control trajectory "
                f"on the head column and hashed before arm 2"),
            "cond3_primary_detector": (f"abort3b_morph_sign_flip -- now a FLOOR at "
                                       f"{COND3_SIGN_FLIP} on {COND3_COLUMN}, not a "
                                       f"sign flip at 0.5"),
            "cond4_src_margin": self.cond4_src_margin,
            "cond4_tgt_margin": self.cond4_tgt_margin,
            "cond4_form": ("DIFFERENTIAL against the lambda=0 control at matched "
                           "absolute steps, src and tgt separately. The absolute "
                           "PR < 20 of 640 is WITHDRAWN -- it fired 7,999 of 8,000 "
                           "times on arm 1."),
            "respecified_from_control": (
                "conditions 3 and 4 were RE-SPECIFIED from arm 1 (runs/dann_arm1) on "
                "2026-08-28. NOT a pre-registration -- the pre-registered tree ran out "
                "when 3b fired on the control. Clean because arm 2 is log_only, so no "
                "detector acts on the reported development run; these bind on folds 1-4."),
            "control": (None if self.control is None else self.control.summary()),
            "missing_control_steps": self.missing_control_steps[:64],
            "n_missing_control_steps": len(self.missing_control_steps),
            "conditions": [d.summary() for d in self.all],
            "fired": self.fired(),
            "withdrawn": (
                "0.55 (the design doc) is WITHDRAWN -- but NOT because it was invented. "
                "It sat just under the HEAD instrument's real ~0.61 baseline "
                "(PROJECT_STATE v2.md 6/8-F4). It is withdrawn because it is meaningless "
                "on the frozen-probe column, which reads ~0.92. Brief v2's 0.6910 "
                "rescaling of it is withdrawn too, because rescaling a threshold across "
                "a change of instrument assumes the two measure the same thing. Brief "
                "v3's 0.10 is withdrawn as well: it was declared against the frozen "
                "probe's 0.4202 headroom and is scale-wrong for the head column's "
                "~0.078."),
        }



# ---------------------------------------------------------------------------
# 8. the arm-completion summary -- the numbers M gets declared against
# ---------------------------------------------------------------------------

# The declaration criterion, PRE-REGISTERED before arm 1 runs (PROJECT.md 7.17).
# It is written here as code so it cannot drift from the prose.
# COND3_M_CEILING is defined ONCE, near the condition-3 constants above, as a FRACTION
# OF MEASURED HEADROOM. It is deliberately NOT redefined here: an earlier revision had
# the absolute 0.04 in both places, and the later definition silently won -- which is
# how a re-specified ceiling can look applied while the old number is what runs.
COND3_M_J_FACTOR = 3.0      # a DECLARED judgement call
COND3_J_WINDOW = (2000, 8000)
COND3_IQR_LAST_STEPS = 4000


COND3_J_EARLY_WINDOW = (0, 2000)
COND3_J_EARLY_RATIO_WARN = 2.0


def _trailing_median(steps, vals, k):
    """The series 3a actually thresholds at window ``k``, as (steps, values).

    Trailing median over the last ``k`` readings, with the first ``k-1`` DROPPED --
    the detector refuses to evaluate until its window is full, so a J measured over
    partial windows would be measuring something 3a never sees. ``k=1`` returns the
    input unchanged, so every number downstream is bit-identical.
    """
    k = max(1, int(k))
    if k == 1:
        return list(steps), [vals[t] for t in steps]
    out_s, out_v, buf = [], [], []
    for t in steps:
        buf.append(vals[t])
        if len(buf) > k:
            del buf[0]
        if len(buf) == k:
            out_s.append(t)
            out_v.append(float(np.median(buf)))
    return out_s, out_v


def morph_trajectory_stats(steps_csv, *, column=COND3_COLUMN, k=1,
                           j_window=COND3_J_WINDOW,
                           j_early_window=COND3_J_EARLY_WINDOW,
                           iqr_last_steps=COND3_IQR_LAST_STEPS,
                           m_ceiling=COND3_M_CEILING,
                           j_factor=COND3_M_J_FACTOR,
                           early_ratio_warn=COND3_J_EARLY_RATIO_WARN,
                           morph_every=None) -> dict:
    """J, the trajectory IQR, and the pre-registered verdict on whether M is declarable.

    THE CRITERION, FIXED BEFORE ARM 1 RAN (PROJECT.md 7.17)::

        Let J = median absolute step-to-step change in morph_head_auroc_pool
                over arm 1, steps 2000-8000.

        If a candidate M >= 3*J can be declared while M <= 0.04:
            declare M, hash it, proceed with 3a active.
        Else:
            declare 3a UNCALIBRABLE at this pool size, record J and the reason,
            and let 3b carry condition 3 alone.

    So the criterion is feasible exactly when ``3*J <= 0.04``, and the smallest
    admissible margin is ``3*J``. Both are computed here rather than eyeballed, because
    the whole point of fixing the rule in advance is that the arithmetic is not done
    while looking at the curve.

    **"Step-to-step" means BETWEEN CONSECUTIVE READINGS**, which at ``--morph-every 25``
    are 25 steps apart. That is the jitter 3a actually sees; a per-step interpolation
    would be measuring something the detector never looks at.

    **J IS COMPUTED ON THE SERIES 3a WILL ACTUALLY THRESHOLD.** If ``k > 1`` is declared,
    3a thresholds a median-smoothed difference whose step-to-step noise is roughly
    ``J_raw / sqrt(k)``. Comparing M against the RAW J would then over-constrain by ~2.8x
    at k=8 and could send a viable 3a to UNCALIBRABLE for no reason. So the declared k is
    applied to the trajectory FIRST and J is computed on the smoothed series. This is a
    CLARIFICATION of the pre-registered criterion, not a change to it: the criterion has
    always been about the noise in the quantity 3a thresholds, and the ordering constraint
    (k before M) is what makes applying k first legitimate rather than circular.
    ``J_raw`` and ``J_at_k`` are both emitted, and ``j_used_for_criterion`` names which
    one ``m_min_admissible`` came from. At ``k=1`` they are the same number.

    ONE HONEST GAP: on arm 1 alone there is no control, so the smoothing is applied to
    the arm's own TRAJECTORY as a proxy for the difference series. That is the same proxy
    J itself is (see above), one level further in.

    **``J_early`` (steps 0-2000) IS EMITTED BUT IS NOT THE CRITERION.** The criterion
    window is 2000-8000, which measures SETTLED jitter -- but if a collapse happens it
    happens early, and that is exactly where 3a operates. An M sized on late-window noise
    will throw false streaks in the first thousand steps if early jitter is larger. So
    ``J_early`` and the ratio ``J_early / J`` are reported as a SANITY CHECK to look at
    before writing M down, and the verdict says so explicitly when the ratio exceeds
    ``early_ratio_warn`` (default 2). ``M_DECLARABLE`` is unchanged and is still computed
    from the 2000-8000 window.

    WHY J IS THE RIGHT PROXY AND WHAT IT DOES NOT CAPTURE. 3a reads a PAIRED
    DIFFERENCE: both arms score the identical fixed pool at the identical step, so
    pool-selection variance is largely COMMON and substantially cancels. It will not
    cancel exactly -- different models rank the same images differently -- so arm 1's
    own step-to-step jitter LOWER-BOUNDS the noise in the difference. It is the only
    measurable proxy available from arm 1 alone, and it is labelled as a bound rather
    than as the quantity itself.

    **UNCALIBRABLE IS A LEGITIMATE OUTCOME, NOT A FAILURE.** It is reported as a
    property of the historical instrument, which was never designed as a graded monitor.
    """
    raw_rows = {}
    with open(steps_csv, newline="") as fh:
        rdr = csv.DictReader(fh)
        if column not in (rdr.fieldnames or []):
            raise SystemExit(f"[morph-stats] {steps_csv} has no column {column!r} "
                             f"(has {rdr.fieldnames})")
        order = []
        for r in rdr:
            v = _num(r.get(column))
            if v is not None:
                st = int(float(r["step"]))
                raw_rows[st] = v
                order.append(st)
    if not raw_rows:
        raise SystemExit(f"[morph-stats] no finite {column!r} readings in {steps_csv}")

    # ---- SELECT THE ACTUAL MEASUREMENT GRID, not every row ------------------
    # steps.csv carries ONE ROW PER STEP and the morph column is FORWARD-FILLED
    # between measurements (the monitor runs every --morph-every steps). Treating
    # every row as a reading makes the median step-to-step change come out EXACTLY
    # ZERO, because consecutive rows are literally the same number -- which is what
    # this function measured on arm 1 before this was fixed, reporting J = 0.000000
    # and a meaningless M_DECLARABLE = True.
    #
    # The grid is recovered from where the value CHANGES: the modal gap is the
    # cadence and the first change is the offset. Rows are then selected ON that
    # grid rather than by de-duplication, so two consecutive readings that happen to
    # be numerically equal (AUROC on a fixed pool takes discrete values, so ties are
    # possible) are kept rather than silently merged.
    all_steps = sorted(raw_rows)
    changes = [all_steps[0]] + [b for a, b in zip(all_steps, all_steps[1:])
                                if raw_rows[b] != raw_rows[a]]
    stride = _mode_stride(changes) or 1
    offset = changes[0]
    grid = [t for t in all_steps if (t - offset) % stride == 0]
    vals = {t: raw_rows[t] for t in grid}
    forward_filled = len(all_steps) > len(grid)
    cadence_note = (
        f"steps.csv holds {len(all_steps)} rows; the morph column is forward-filled "
        f"between measurements, so the {len(grid)} rows on the measurement grid "
        f"(offset {offset}, stride {stride}) are the actual readings. Every statistic "
        f"below is computed on those, never on the filled rows."
        if forward_filled else
        f"{len(grid)} rows, one per reading (stride {stride}); no forward-fill detected.")
    if morph_every is not None and int(morph_every) != stride:
        cadence_note += (f" *** the observed stride ({stride}) does NOT match the "
                         f"declared --morph-every ({morph_every}); the OBSERVED grid is "
                         f"used, because it is what is in the file. ***")

    steps = sorted(vals)
    last_step = steps[-1]
    k = max(1, int(k))

    def _J(series_steps, series_vals, window):
        lo_, hi_ = int(window[0]), int(window[1])
        idx = [i for i, t in enumerate(series_steps) if lo_ <= t <= hi_]
        v = [series_vals[i] for i in idx]
        d = [abs(b - a) for a, b in zip(v, v[1:])]
        return (float(np.median(d)) if d else float("nan")), len(d)

    # the RAW series (k=1) and the series 3a will actually threshold (at k)
    raw_s, raw_v = _trailing_median(steps, vals, 1)
    sm_s, sm_v = _trailing_median(steps, vals, k)

    J_raw, n_raw = _J(raw_s, raw_v, j_window)
    J_at_k, n_at_k = _J(sm_s, sm_v, j_window)
    J = J_at_k                                    # what the criterion is evaluated on
    J_early_raw, n_early_raw = _J(raw_s, raw_v, j_early_window)
    J_early, n_early = _J(sm_s, sm_v, j_early_window)
    early_ratio = (J_early / J) if (math.isfinite(J) and J > 0
                                    and math.isfinite(J_early)) else float("nan")
    early_warn = bool(math.isfinite(early_ratio) and early_ratio > early_ratio_warn)

    def _iqr(series_steps, series_vals):
        idx = [i for i, t in enumerate(series_steps)
               if t > last_step - int(iqr_last_steps)]
        v = [series_vals[i] for i in idx]
        if len(v) < 4:
            return float("nan"), float("nan"), len(v)
        return float(np.percentile(v, 25)), float(np.percentile(v, 75)), len(v)

    q25, q75, n_tail = _iqr(raw_s, raw_v)
    q25_k, q75_k, _ = _iqr(sm_s, sm_v)
    iqr_raw = (q75 - q25) if math.isfinite(q75) and math.isfinite(q25) else float("nan")
    iqr_at_k = ((q75_k - q25_k) if math.isfinite(q75_k) and math.isfinite(q25_k)
                else float("nan"))
    tail = [vals[t] for t in steps if t > last_step - int(iqr_last_steps)]

    m_min = j_factor * J if math.isfinite(J) else float("nan")
    declarable = bool(math.isfinite(m_min) and m_min <= m_ceiling)
    _early_clause = (
        f" EARLY-WINDOW WARNING: J_early ({J_early:.6f}) is {early_ratio:.1f}x the "
        f"criterion's J ({J:.6f}); an M admissible on the LATE window may not be safe in "
        f"the first {j_early_window[1]} steps, which is exactly where a collapse would "
        f"happen and where 3a operates. Look at this before writing M down."
        if early_warn else "")
    return {
        "column": column,
        "n_rows_in_file": len(all_steps),
        "n_readings_total": len(steps),
        "reading_stride": stride,
        "reading_offset": offset,
        "forward_filled": forward_filled,
        "cadence_note": cadence_note,
        "k": k,
        "J_window_steps": [int(j_window[0]), int(j_window[1])],
        "J_n_consecutive_pairs": n_at_k,
        "J_median_abs_step_to_step": J,
        "J_raw": J_raw,
        "J_raw_n_pairs": n_raw,
        "J_at_k": J_at_k,
        "j_used_for_criterion": ("J_at_k" if k > 1 else "J_raw (k=1, so J_at_k == J_raw)"),
        "J_k_note": (
            "J is computed on the series 3a WILL ACTUALLY THRESHOLD. At k>1 that is the "
            "trailing median of the last k readings, whose step-to-step noise is roughly "
            "J_raw/sqrt(k); comparing M against the RAW J would over-constrain by ~2.8x "
            "at k=8 and could send a viable 3a to UNCALIBRABLE for no reason. The first "
            "k-1 readings are DROPPED, matching the detector's refusal to evaluate on a "
            "partial window. A CLARIFICATION of the criterion, not a change to it -- and "
            "legitimate only because k is declared BEFORE M."),
        "J_note": ("median |change| between CONSECUTIVE READINGS (25 steps apart at "
                   "--morph-every 25). A LOWER BOUND on the noise in the paired "
                   "difference 3a thresholds, not that noise itself."),
        # ---- the EARLY window: reported, NOT part of the criterion ----
        "J_early_window_steps": [int(j_early_window[0]), int(j_early_window[1])],
        "J_early": J_early,
        "J_early_raw": J_early_raw,
        "J_early_n_pairs": n_early,
        "J_early_over_J": early_ratio,
        "J_early_ratio_warn_at": early_ratio_warn,
        "J_early_WARNING": early_warn,
        "J_early_note": (
            "the criterion window (2000-8000) measures SETTLED jitter. A collapse, if it "
            "happens, happens EARLY -- which is exactly where 3a operates. J_early is a "
            "SANITY CHECK to look at before writing M down; M_DECLARABLE is still "
            f"computed from the {int(j_window[0])}-{int(j_window[1])} window and is NOT "
            f"affected by it."),
        "iqr_window_last_steps": int(iqr_last_steps),
        "iqr_n_readings": n_tail,
        "iqr_q25": q25, "iqr_q75": q75,
        "iqr": iqr_raw,
        "iqr_raw": iqr_raw,
        "iqr_at_k": iqr_at_k,
        "J_reduction_factor_at_k": ((J_raw / J_at_k) if (math.isfinite(J_raw)
                                                         and math.isfinite(J_at_k)
                                                         and J_at_k > 0)
                                    else float("nan")),
        "iqr_reduction_factor_at_k": ((iqr_raw / iqr_at_k) if (math.isfinite(iqr_raw)
                                                               and math.isfinite(iqr_at_k)
                                                               and iqr_at_k > 0)
                                      else float("nan")),
        "k_caution": (
            "MEASURED, and it matters if k>1 is declared. Smoothing reduces J MUCH more "
            "than the ~sqrt(k) the criterion's rationale assumed: consecutive trailing "
            "medians share k-1 of their k inputs, so the SMOOTHED series' step-to-step "
            "change collapses (measured ~11x at k=8, ~5x at k=4, against sqrt(k) = 2.8 "
            "and 2.0) while its LEVEL noise -- the thing that could actually hold a "
            "500-step false streak -- falls only about sqrt(k). Compare "
            "J_reduction_factor_at_k against iqr_reduction_factor_at_k: if J has fallen "
            "far more than the IQR, the criterion has been satisfied by autocorrelation "
            "rather than by real noise reduction. NOT ACTED ON HERE -- the criterion is "
            "pre-registered and this file does not change it. Flagged so the k decision "
            "is made with the number visible."),
        "trajectory_first": vals[steps[0]], "trajectory_last": vals[last_step],
        "trajectory_min": float(min(vals.values())),
        "trajectory_max": float(max(vals.values())),
        # ---- the pre-registered verdict ----
        "criterion": (f"declare M iff a candidate with M >= {j_factor}*J can be chosen "
                      f"while M <= {m_ceiling}; i.e. iff {j_factor}*J <= {m_ceiling}, "
                      f"with J measured on the k={k} series 3a will threshold"),
        "m_factor": j_factor, "m_ceiling": m_ceiling,
        "m_ceiling_note": (f"{m_ceiling} is roughly 60% of the whole-pool headroom over "
                           f"chance (0.0657). An M above it would only fire in territory "
                           f"3b already covers."),
        "m_min_admissible": m_min,
        "M_DECLARABLE": declarable,
        "verdict": ((f"AT k={k}: DECLARE M in [{m_min:.4f}, {m_ceiling}] and hash it "
                     f"before arm 2." if declarable else
                     f"AT k={k}: 3a is UNCALIBRABLE at this pool size -- "
                     f"{j_factor:g}*J = {m_min:.4f} exceeds the {m_ceiling} ceiling. "
                     f"Record J and the reason, and let 3b carry condition 3 alone. THIS "
                     f"IS A LEGITIMATE OUTCOME -- a property of the historical "
                     f"instrument, which was never designed as a graded monitor.")
                    + _early_clause),
        "k_ordering_constraint": (
            "if J is borderline, --abort3-median-window k > 1 may be used, and k MUST BE "
            "DECLARED BEFORE M. A window chosen after a candidate M is on the table has "
            "been selected to make that M work."),
    }


def print_morph_trajectory_stats(st: dict) -> None:
    print("\n" + "=" * 92)
    print(f"ARM-COMPLETION SUMMARY -- the numbers M is declared against ({st['column']})")
    print("=" * 92)
    print(f"  readings                 : {st['n_readings_total']} at stride "
          f"{st['reading_stride']} (offset {st['reading_offset']})"
          + (f"   [from {st['n_rows_in_file']} forward-filled rows]"
             if st.get("forward_filled") else ""))
    print(f"  trajectory               : first {st['trajectory_first']:.4f}  last "
          f"{st['trajectory_last']:.4f}  min {st['trajectory_min']:.4f}  max "
          f"{st['trajectory_max']:.4f}")
    print(f"  k (median window)        : {st['k']}"
          + ("   (no smoothing)" if st['k'] == 1 else
             f"   -> J computed on the SMOOTHED series 3a will threshold"))
    print(f"  J_raw  (k=1, {st['J_raw_n_pairs']:>3} pairs)   : {st['J_raw']:.6f}")
    print(f"  J_at_k (k={st['k']}, {st['J_n_consecutive_pairs']:>3} pairs)   : "
          f"{st['J_at_k']:.6f}   <- {st['j_used_for_criterion']}")
    print(f"  J_early (steps {st['J_early_window_steps'][0]}-"
          f"{st['J_early_window_steps'][1]}, {st['J_early_n_pairs']} pairs): "
          f"{st['J_early']:.6f}   ratio J_early/J = {st['J_early_over_J']:.2f}"
          + ("   *** WARNING ***" if st["J_early_WARNING"] else ""))
    print(f"  IQR over the last {st['iqr_window_last_steps']} steps "
          f"({st['iqr_n_readings']} readings): {st['iqr']:.6f}   "
          f"[{st['iqr_q25']:.4f}, {st['iqr_q75']:.4f}]")
    if st["k"] > 1:
        print(f"  k>1 SANITY: J fell {st['J_reduction_factor_at_k']:.1f}x while the IQR "
              f"fell {st['iqr_reduction_factor_at_k']:.1f}x (sqrt(k) = "
              f"{math.sqrt(st['k']):.1f}).")
        print(f"             If J has fallen far more than the IQR, the criterion was "
              f"satisfied by AUTOCORRELATION,")
        print(f"             not by real noise reduction -- consecutive trailing medians "
              f"share k-1 of their k inputs.")
    print("  " + "-" * 88)
    print(f"  CRITERION (pre-registered): {st['criterion']}")
    print(f"  {st['m_factor']:g}*J = {st['m_min_admissible']:.6f}   ceiling "
          f"{st['m_ceiling']}   ->   M_DECLARABLE = {st['M_DECLARABLE']}")
    print(f"  {st['verdict']}")
    print(f"  {st['k_ordering_constraint']}")
    print("=" * 92, flush=True)


ABORT_COLUMNS = (["morph_minus_control", "control_morph_proxy_auroc",
                  "src_pr_minus_control", "tgt_pr_minus_control"]
                 + [c for d in ABORT_DETECTOR_NAMES
                    for c in (f"{d}_streak", f"{d}_fired")])
