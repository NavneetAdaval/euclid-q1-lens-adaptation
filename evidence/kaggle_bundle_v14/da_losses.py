"""
da_losses.py
============
Home for the DOMAIN-ADAPTATION loss modules of the label-free sim-to-real
strong-lens pipeline (Euclid Q1). Deep CORAL is the first real DA method; DANN /
AdaMatch / WDGRL will land here as sibling modules with the same call contract.

Every loss here obeys the project's non-negotiable interface:

  * It consumes the HEAD/ADAPTER features (``head.features(x)`` -- i.e. the
    ``src_adapter`` / ``tgt_adapter`` that ``forward_with_features`` returns), NEVER
    the raw frozen encoder features. The encoder is frozen and its features arrive
    DETACHED, so a DA loss on the adapter provably cannot backprop into the encoder
    -- that back-prop is the representation collapse PROJECT.md 1 says we already
    measured. We ASSERT the graph is head-only (see ``training`` guard below).
  * The TARGET batch is UNLABELED. CORAL uses only its FEATURES, never its labels.

------------------------------------------------------------------------------
Deep CORAL  (``CoralLoss``)
------------------------------------------------------------------------------
Canonical references followed verbatim (no improvised covariance math):
  * Sun & Saenko, "Deep CORAL", ECCV 2016 workshops (arXiv:1607.01719).
  * DenisDsh/PyTorch-Deep-CORAL  (the widely-cited reference implementation).
  * Musgrave, pytorch-adapt ``CORALLoss`` (its ``correlation`` helper is the exact
    covariance formula below).
  * DomainBed ``CORAL`` -- which ADDS the first-moment (mean) term to the original
    second-moment-only CORAL; we adopt that "align first AND second moments" form.

The loss (on the standardized adapter features of the two domains):

    L = ||mean_S - mean_T||^2  +  ||Cov_S - Cov_T||_F^2                 (both terms)

Covariance per domain (the request's / Musgrave's exact form):

    C = (1/(n-1)) ( z^T z  -  (1/n) (1^T z)^T (1^T z) )

i.e. the ordinary unbiased sample covariance (identical to the centered
``cent^T cent / (n-1)`` DomainBed/DenisDsh write).

NORMALIZATION (FLAG -- a scale convention, stated explicitly):
  We divide each squared distance by the number of matrix/vector entries -- i.e.
  ``(Cov_S-Cov_T).pow(2).mean()`` and ``(mean_S-mean_T).pow(2).mean()``. This is
  EXACTLY DomainBed's ``coral()`` (it uses ``.pow(2).mean()`` on both terms), and it
  is what keeps the CORAL term on a scale comparable to the supervised BCE so the
  FIXED ``coral_weight`` (gamma) is meaningful rather than swamping BCE by a factor
  of d^2 (d=640 for the linear adapter). The literal Frobenius-SUM form differs only
  by this fixed d / d^2 factor, which would otherwise be absorbed into gamma. We
  follow DomainBed's mean-normalized convention and flag it here.

STABILITY ADD-ONS the request requires (sourced -- small ~391/fold target batches
make a raw batch covariance noisy/unstable):

  * COVARIANCE SHRINKAGE (required, both domains):
        Sigma_shr = (1 - lam) * Sigma_hat  +  lam * (tr(Sigma_hat)/d) * I
    with a fixed ``lam`` (default 0.1). Ledoit-Wolf-style shrinkage toward a scaled
    identity; it conditions the estimate before the Frobenius distance. Applied to
    BOTH domains so the alignment target is unbiased.

  * EMA TARGET STATISTICS (default on for the target only):
        maintain a running (momentum, default 0.9) estimate of the target mean and
    covariance across batches, so the alignment reference is not a single noisy
    ~batch-sized covariance. The SOURCE uses its per-batch statistics (the 48k source
    pool is plentiful). Toggleable via ``target_ema``. The current batch still
    contributes gradient (through its ``(1-momentum)`` share) so CORAL keeps training
    the head symmetrically; only the historical part is detached.

  * NUMERICAL SAFETY: a small diagonal jitter is added after shrinkage, and the final
    loss is asserted finite. (CORAL's Frobenius distance does not itself require
    positive-definiteness -- there is no Cholesky/inverse/logdet here -- so the jitter
    is precautionary conditioning, not a correctness requirement.)
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# stability diagnostic: late-training CORAL covariance-term BLOW-UP detector
# ---------------------------------------------------------------------------
#
# Root cause (sourced: Chen et al., "The Pros and Cons of Feature-Space Alignment",
# arXiv:1705.08180): the RAW Frobenius covariance distance a Deep-CORAL loss minimises
# can grow pathologically LATE in training even when the network weights barely move --
# it is a GEOMETRY problem of the covariance term, not exploding gradients. We do NOT
# clip gradients and do NOT touch the covariance math (that would mask the geometry
# signal); we DETECT the blow-up and let the runner stop before/at it.
#
# The detector is a pure running diagnostic on ``CoralLoss.last_cov_term`` (already
# exposed): it learns a HEALTHY baseline = EMA of the cov-term over an early window of
# steps, then flags any later step whose cov-term exceeds ``blowup_factor`` x that
# baseline. It owns no tensors/graph -- it consumes plain floats -- so it is trivially
# testable on a synthetic cov-term series.

class BlowupDetector:
    """Flag late-training CORAL covariance-term blow-up against an early baseline.

    Parameters
    ----------
    baseline_window : int
        Number of initial steps that define the HEALTHY window. The baseline is the
        EMA of the cov-term over exactly these steps; while the window is still filling
        the detector NEVER flags (it is calibrating). Typically one epoch of steps.
    blowup_factor : float, default 5.0
        A step is a blow-up when ``cov_term > blowup_factor * baseline``.
    ema_momentum : float, default 0.9
        Momentum for the running cov-term EMA (both the early-window baseline EMA and
        the all-steps EMA reported for logging).

    Use
    ---
    Call :meth:`update` once per step with ``coral.last_cov_term``; it returns whether
    THIS step is a blow-up. ``detected`` is sticky (True once any step blew up),
    ``baseline`` is the frozen healthy reference (None until the window fills), and
    ``ema`` is the running cov-term EMA for logging.
    """

    def __init__(self, baseline_window: int = 50, blowup_factor: float = 5.0,
                 ema_momentum: float = 0.9):
        if baseline_window < 1:
            raise ValueError(f"baseline_window must be >=1, got {baseline_window}")
        if blowup_factor <= 1.0:
            raise ValueError(f"blowup_factor must be >1, got {blowup_factor}")
        if not (0.0 <= ema_momentum < 1.0):
            raise ValueError(f"ema_momentum must be in [0,1), got {ema_momentum}")
        self.baseline_window = int(baseline_window)
        self.blowup_factor = float(blowup_factor)
        self.ema_momentum = float(ema_momentum)
        self.n = 0
        self.ema: float | None = None            # all-steps EMA (logging)
        self._early_ema: float | None = None     # EMA over the early window
        self.baseline: float | None = None       # frozen healthy baseline
        self.detected = False                    # sticky: has blow-up ever fired

    def update(self, cov_term: float) -> bool:
        """Feed one step's cov-term; return True iff THIS step is a blow-up."""
        c = float(cov_term)
        m = self.ema_momentum
        self.ema = c if self.ema is None else m * self.ema + (1.0 - m) * c
        self.n += 1
        if self.n <= self.baseline_window:
            # still inside the healthy window: grow its EMA, never flag
            self._early_ema = (c if self._early_ema is None
                               else m * self._early_ema + (1.0 - m) * c)
            if self.n == self.baseline_window:
                self.baseline = self._early_ema
            return False
        step_blowup = (self.baseline is not None
                       and c > self.blowup_factor * self.baseline)
        if step_blowup:
            self.detected = True
        return bool(step_blowup)


# ---------------------------------------------------------------------------
# covariance (the canonical unbiased sample covariance -- NOT improvised)
# ---------------------------------------------------------------------------

def _mean_cov(z: torch.Tensor):
    """Return ``(mean, cov)`` for rows-are-samples ``z`` of shape ``(n, d)``.

    cov = (1/(n-1)) ( z^T z - (1/n) (1^T z)^T (1^T z) )    (Musgrave/DenisDsh/DomainBed)
    mean = (1/n) (1^T z)

    Both are permutation-invariant over the ``n`` rows (a fact the row-order test
    exercises). Requires ``n >= 2`` for the unbiased (n-1) denominator.
    """
    if z.dim() != 2:
        raise ValueError(f"expected 2-D (n,d) features, got shape {tuple(z.shape)}")
    n, d = z.shape
    if n < 2:
        raise ValueError(f"covariance needs n>=2 rows, got n={n}")
    ones_z = z.sum(dim=0, keepdim=True)                 # (1,d) = 1^T z
    ztz = z.t() @ z                                     # (d,d)
    outer = (ones_z.t() @ ones_z) / n                   # (1/n) (1^T z)^T (1^T z)
    cov = (ztz - outer) / (n - 1)
    mean = ones_z.squeeze(0) / n                        # (d,)
    return mean, cov


def _shrink(cov: torch.Tensor, lam: float, jitter: float) -> torch.Tensor:
    """Ledoit-Wolf-style shrinkage toward a scaled identity, plus a jitter floor.

        Sigma_shr = (1-lam) Sigma_hat + lam (tr(Sigma_hat)/d) I  +  jitter * I
    """
    d = cov.shape[0]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    if lam > 0.0:
        mu = torch.diagonal(cov).sum() / d              # tr(Sigma)/d
        cov = (1.0 - lam) * cov + lam * mu * eye
    if jitter > 0.0:
        cov = cov + jitter * eye
    return cov


# ---------------------------------------------------------------------------
# Deep CORAL loss module
# ---------------------------------------------------------------------------

class CoralLoss(nn.Module):
    """DomainBed-style Deep CORAL (first + second moment) on adapter features.

    Parameters
    ----------
    feat_dim : int
        Adapter feature dimension (640 for the linear head, 128 for the mlp head).
        Sizes the EMA target buffers.
    include_mean : bool, default True
        Include the first-moment term ``||mean_S - mean_T||^2``. Toggle it off to
        ablate down to the original second-moment-only CORAL.
    shrinkage : float, default 0.1
        The fixed shrinkage coefficient ``lam`` applied to BOTH domains' covariances.
    target_ema : bool, default True
        Maintain an EMA of the TARGET mean/covariance across batches (the alignment
        reference); the source always uses its plentiful per-batch statistics.
    ema_momentum : float, default 0.9
        EMA momentum ``m``: ``ema <- m*ema + (1-m)*batch``.
    jitter : float, default 1e-6
        Diagonal jitter added after shrinkage (numerical safety).

    forward(src_adapter, tgt_adapter) -> scalar tensor
        The RAW (unweighted) CORAL loss. The caller multiplies by ``coral_weight``.
        ``src_adapter``/``tgt_adapter`` are the head/adapter features of the two
        domains; the target's adapter features carry NO label information.
    """

    def __init__(self, feat_dim: int, *, include_mean: bool = True,
                 shrinkage: float = 0.1, target_ema: bool = True,
                 ema_momentum: float = 0.9, jitter: float = 1e-6):
        super().__init__()
        if feat_dim < 1:
            raise ValueError(f"feat_dim must be >=1, got {feat_dim}")
        if not (0.0 <= shrinkage <= 1.0):
            raise ValueError(f"shrinkage lam must be in [0,1], got {shrinkage}")
        if not (0.0 <= ema_momentum < 1.0):
            raise ValueError(f"ema_momentum must be in [0,1), got {ema_momentum}")
        self.feat_dim = int(feat_dim)
        self.include_mean = bool(include_mean)
        self.shrinkage = float(shrinkage)
        self.target_ema = bool(target_ema)
        self.ema_momentum = float(ema_momentum)
        self.jitter = float(jitter)

        # EMA target statistics live in buffers (detached history, serialized, moved
        # by .to()). ``ema_ready`` flags whether they have been initialized yet.
        self.register_buffer("ema_mean", torch.zeros(self.feat_dim))
        self.register_buffer("ema_cov", torch.zeros(self.feat_dim, self.feat_dim))
        self.register_buffer("ema_ready", torch.zeros(1))

        # last-computed component terms, for logging (floats; not part of the graph)
        self.last_mean_term = 0.0
        self.last_cov_term = 0.0

    # -- EMA blending for the target reference ------------------------------

    def _target_stats(self, mean_cur: torch.Tensor, cov_cur: torch.Tensor):
        """Return the (mean, cov) used as the TARGET alignment reference.

        With ``target_ema`` off (or in eval / no-grad), this is just the per-batch
        statistic. With EMA on during training, it is
        ``m*history + (1-m)*current`` where ``history`` is the detached EMA buffer and
        ``current`` still carries gradient (so CORAL keeps training the head through
        the target side). The buffers are then updated in-place with the fully
        detached current statistic.
        """
        updating = self.target_ema and self.training and torch.is_grad_enabled()
        if not self.target_ema:
            return mean_cur, cov_cur

        if float(self.ema_ready) < 1.0:
            # first sight of the target: seed the EMA and use the current batch as-is
            # (so identical source/target features -> exactly zero loss on step 1).
            if updating:
                with torch.no_grad():
                    self.ema_mean.copy_(mean_cur.detach())
                    self.ema_cov.copy_(cov_cur.detach())
                    self.ema_ready.fill_(1.0)
            return mean_cur, cov_cur

        m = self.ema_momentum
        prev_mean = self.ema_mean.clone()               # detached buffer snapshot
        prev_cov = self.ema_cov.clone()
        # differentiable reference: history (detached) + current (grad-carrying) share
        tgt_mean = m * prev_mean + (1.0 - m) * mean_cur
        tgt_cov = m * prev_cov + (1.0 - m) * cov_cur
        if updating:
            with torch.no_grad():
                self.ema_mean.copy_(m * prev_mean + (1.0 - m) * mean_cur.detach())
                self.ema_cov.copy_(m * prev_cov + (1.0 - m) * cov_cur.detach())
        return tgt_mean, tgt_cov

    def reset_ema(self):
        """Clear the EMA target statistics (call between folds / fresh runs)."""
        with torch.no_grad():
            self.ema_mean.zero_()
            self.ema_cov.zero_()
            self.ema_ready.zero_()

    # -- forward ------------------------------------------------------------

    def forward(self, src_adapter: torch.Tensor, tgt_adapter: torch.Tensor) -> torch.Tensor:
        if src_adapter.dim() != 2 or tgt_adapter.dim() != 2:
            raise ValueError(
                f"CORAL expects 2-D adapter features; got src={tuple(src_adapter.shape)} "
                f"tgt={tuple(tgt_adapter.shape)}")
        if src_adapter.shape[1] != tgt_adapter.shape[1]:
            raise ValueError(
                f"source/target feature dims differ: {src_adapter.shape[1]} vs "
                f"{tgt_adapter.shape[1]}")
        if src_adapter.shape[1] != self.feat_dim:
            raise ValueError(
                f"adapter dim {src_adapter.shape[1]} != configured feat_dim {self.feat_dim}")

        # Head-only-graph guard: in a real training step the adapter features MUST
        # carry grad (they flow from the trainable head). Detached tensors here would
        # mean the caller wired the raw *encoder* features by mistake -- exactly the
        # backprop-into-encoder we forbid. (Skipped under no_grad / eval numeric tests.)
        if self.training and torch.is_grad_enabled():
            assert src_adapter.requires_grad and tgt_adapter.requires_grad, (
                "CORAL expected grad-carrying ADAPTER features (trainable head output); "
                "got detached tensors -- is it wired to the frozen encoder features?")

        # first + second moments
        src_mean, src_cov = _mean_cov(src_adapter)                  # per-batch (grad)
        tgt_mean_cur, tgt_cov_cur = _mean_cov(tgt_adapter)          # per-batch (grad)
        tgt_mean, tgt_cov = self._target_stats(tgt_mean_cur, tgt_cov_cur)

        # shrinkage (+ jitter) on BOTH domains before the Frobenius distance
        src_cov = _shrink(src_cov, self.shrinkage, self.jitter)
        tgt_cov = _shrink(tgt_cov, self.shrinkage, self.jitter)

        cov_term = (src_cov - tgt_cov).pow(2).mean()               # ||.||_F^2 / d^2
        loss = cov_term
        mean_term = src_cov.new_zeros(())
        if self.include_mean:
            mean_term = (src_mean - tgt_mean).pow(2).mean()        # ||.||^2 / d
            loss = loss + mean_term

        assert torch.isfinite(loss), f"CORAL loss is non-finite: {float(loss)!r}"
        self.last_cov_term = float(cov_term.detach())
        self.last_mean_term = float(mean_term.detach())
        return loss

    def extra_repr(self) -> str:
        return (f"feat_dim={self.feat_dim}, include_mean={self.include_mean}, "
                f"shrinkage={self.shrinkage}, target_ema={self.target_ema}, "
                f"ema_momentum={self.ema_momentum}, jitter={self.jitter}")


# ===========================================================================
# AdaMatch  (``AdaMatchLoss``)  -- the SECOND real DA method
# ===========================================================================
#
# Canonical references followed:
#   * Berthelot, Roelofs, Sohn, Carlini, Kurakin, "AdaMatch: A Unified Approach to
#     Semi-Supervised Learning and Domain Adaptation" (arXiv:2106.04732).
#   * The Keras reference implementation (keras.io/examples/vision/adamatch) --
#     cross-checked for the distribution-alignment + relative-threshold arithmetic.
#
# AdaMatch needs WEAK + STRONG augmented views, which are produced PRE-encoder. So --
# unlike CORAL -- it CANNOT run on the cached features: it runs through the image-space
# Trainer with the FROZEN encoder in the loop (encoder under no_grad, only the head
# trains). LensDataset serves n_views=2 (weak="weak", strong="strong") for BOTH source
# and target.
#
# WHAT WE IMPLEMENT (three "working legs" of AdaMatch on a frozen LayerNorm head):
#   1. relative confidence threshold, 2. distribution alignment, 3. the FixMatch
#   consistency backbone. Total loss = BCE(source strong) + mu(step) * consistency.
#
# DELIBERATE, SOURCED OMISSION -- RANDOM LOGIT INTERPOLATION (RLI) IS DROPPED.
#   Canonical AdaMatch interpolates source logits computed with vs without the target's
#   BatchNorm statistics, to neutralise the BN-statistic shift between a source-only and
#   a source+target forward pass. Our encoder is FROZEN and the head uses LayerNorm (no
#   batch-coupled running stats anywhere in the trainable path), so there are no BN
#   statistics to interpolate -- RLI is INERT here. The semiconductor-DA study
#   ("AdaMatch ... for semiconductor defect classification", which ablates AdaMatch on
#   non-BN backbones) confirms RLI specifically targets the BN-statistic shift. We drop
#   it as a no-op, and document it rather than shipping dead code.
#
# ANOTHER FLAG (build-request deviation from the canonical source loss):
#   Canonical AdaMatch computes the supervised CE on BOTH source views (weak & strong,
#   after RLI). Per the build request we use the SOURCE STRONG view for the supervised
#   BCE and the SOURCE WEAK view only as the reference for (a) the confidence threshold
#   and (b) the distribution-alignment source distribution. This is stated, not hidden.


def adamatch_mu_at_step(step: int, total_steps: int, mu_max: float = 1.0) -> float:
    """Target (consistency) weight mu at 0-based global ``step`` of ``total_steps``.

    Canonical AdaMatch cosine warmup over the FIRST HALF of training then hold:

        mu = mu_max * ( 0.5 - cos( min(pi, 2*pi*step/total) ) / 2 )

    mu(0)=0, mu(total/2)=mu_max, and mu stays at mu_max for the second half (the
    ``min(pi, .)`` clamp). The peak ``mu_max`` is the un-tuned config value (default 1.0);
    the schedule SHAPE is fixed, not tuned on the target (RLSbench fixed-budget protocol).
    """
    total = max(int(total_steps), 1)
    x = min(math.pi, 2.0 * math.pi * float(step) / total)
    return float(mu_max * (0.5 - math.cos(x) / 2.0))


def align_target_distribution(tgt_prob2: torch.Tensor, reference_dist: torch.Tensor,
                              target_dist: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """AdaMatch distribution alignment (binary, 2-class): rectify per-sample target
    WEAK predictions by the ratio ``reference_dist / target_dist`` and renormalise.

        aligned_i = normalize( p_i * (reference_dist / target_dist) )

    ``tgt_prob2`` is ``(B,2)`` ([p_neg, p_pos] per row); ``reference_dist`` and
    ``target_dist`` are ``(2,)`` average class distributions (the model's mean predicted
    distribution on weakly-augmented SOURCE and TARGET respectively). Because our source
    is balanced 1:1 the default reference is ~[0.5, 0.5]; the ratio pushes each target
    prediction toward that reference. Pure/stateless so it is unit-testable on a hand
    example.
    """
    if tgt_prob2.dim() != 2 or tgt_prob2.shape[1] != 2:
        raise ValueError(f"expected (B,2) target probs, got {tuple(tgt_prob2.shape)}")
    ratio = reference_dist / (target_dist + eps)          # (2,)
    aligned = tgt_prob2 * ratio                            # (B,2) broadcast
    aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(eps)
    return aligned


# ---------------------------------------------------------------------------
# confirmation-bias collapse detector (the BlowupDetector pattern, for AdaMatch)
# ---------------------------------------------------------------------------
#
# Rare-positive pseudo-labeling can collapse two ways (sourced as the essential guards
# for scarce positives -- FixMatch/AdaMatch confirmation bias, and the class-imbalanced
# SSL literature, e.g. DARP arXiv:2007.08844): either
#   (a) NOTHING is confident  -> acceptance rate ~ 0 (the model learns nothing new), or
#   (b) accepted pseudo-labels are nearly ALL ONE CLASS -> the model is calling
#       everything a lens (or everything a non-lens) and reinforcing that.
# Like the CORAL BlowupDetector this is a pure running DIAGNOSTIC on plain floats
# (acceptance rate + accepted-positive fraction). It is LOGGED, never a model selector.

class ConfirmationBiasDetector:
    """Flag pseudo-label collapse from the per-step acceptance rate + accepted class mix.

    Parameters
    ----------
    min_acceptance : float, default 0.01
        Acceptance rate below this (while pseudo-labeling is active) is an
        "acceptance collapse" (nothing confident).
    class_eps : float, default 0.05
        Accepted-positive fraction <= class_eps or >= 1-class_eps (with a non-trivial
        acceptance rate) is a "class collapse" (all one class).
    patience : int, default 20
        Number of CONSECUTIVE collapse-signature steps required before firing (so a
        transient blip does not trip it). ``detected`` is sticky once it fires.
    warmup_steps : int, default 0
        Steps at the start during which the detector never fires (pseudo-labeling not
        yet active / calibrating). Steps fed with ``active=False`` are likewise ignored.

    Use
    ---
    Call :meth:`update(acceptance_rate, accepted_pos_fraction, active=...)` once per step;
    it returns whether THIS step tripped the (patience-gated) collapse flag. ``reason`` is
    the first collapse kind seen ("acceptance_collapse" / "class_collapse").
    """

    def __init__(self, min_acceptance: float = 0.01, class_eps: float = 0.05,
                 patience: int = 20, warmup_steps: int = 0):
        if patience < 1:
            raise ValueError(f"patience must be >=1, got {patience}")
        if not (0.0 <= class_eps < 0.5):
            raise ValueError(f"class_eps must be in [0,0.5), got {class_eps}")
        self.min_acceptance = float(min_acceptance)
        self.class_eps = float(class_eps)
        self.patience = int(patience)
        self.warmup_steps = int(warmup_steps)
        self.n = 0
        self.detected = False
        self.reason: str | None = None
        self._streak = 0
        self.last_acceptance: float | None = None
        self.last_pos_fraction: float | None = None

    def update(self, acceptance_rate: float, accepted_pos_fraction, active: bool = True) -> bool:
        """Feed one step; return True iff this step tripped the collapse flag."""
        self.n += 1
        ar = float(acceptance_rate)
        pf = (None if accepted_pos_fraction is None or not math.isfinite(float(accepted_pos_fraction))
              else float(accepted_pos_fraction))
        self.last_acceptance = ar
        self.last_pos_fraction = pf

        if not active or self.n <= self.warmup_steps:
            self._streak = 0
            return False

        reason = None
        if ar < self.min_acceptance:
            reason = "acceptance_collapse"
        elif pf is not None and (pf <= self.class_eps or pf >= 1.0 - self.class_eps):
            reason = "class_collapse"

        if reason is not None:
            self._streak += 1
            if self._streak >= self.patience:
                self.detected = True
                if self.reason is None:
                    self.reason = reason
                return True
        else:
            self._streak = 0
        return False


# ---------------------------------------------------------------------------
# AdaMatch loss module
# ---------------------------------------------------------------------------

class AdaMatchLoss(nn.Module):
    """AdaMatch objective (binary) on head LOGITS of weak/strong views of both domains.

    Total loss = BCE(source STRONG logits, source labels) + mu(step) * consistency,
    where consistency is the FixMatch masked BCE between the target STRONG logits and the
    pseudo-labels formed from the DISTRIBUTION-ALIGNED, RELATIVE-CONFIDENCE-THRESHOLDED
    target WEAK predictions. Random logit interpolation is intentionally omitted (frozen
    encoder + LayerNorm head -> no BN statistics to interpolate; see module docstring).

    Parameters
    ----------
    mu_max : float, default 1.0
        Peak consistency weight (FIXED, not tuned on target). See ``adamatch_mu_at_step``.
    tau : float, default 0.9
        Relative confidence factor: threshold = tau * (EMA mean max-confidence on weakly-
        augmented SOURCE). Only target-weak predictions whose aligned max-prob clears this
        relative threshold are pseudo-labeled.
    ema_momentum : float, default 0.9
        Momentum for the running source-dist / target-dist / source-confidence EMAs.
    consistency_warmup_frac : float, default 0.1
        Fraction of total steps to WITHHOLD the consistency loss entirely (mu_eff=0), a
        hard floor ON TOP of the mu warmup so early garbage pseudo-labels can't seed the
        collapse.
    align_reference : sequence[float] | None, default None
        The distribution-alignment reference. None -> use the running SOURCE-dist EMA
        (default; ~[0.5,0.5] for our balanced source). An explicit [p_neg,p_pos] (it is
        renormalised) overrides it -- e.g. to align to deployment 1:2000 in an ablation.
    eps : float, default 1e-6
        Numerical floor for the alignment renormalisation.

    forward(src_weak_logits, src_strong_logits, tgt_weak_logits, tgt_strong_logits,
            src_labels, step, total_steps) -> (loss, info)
        All ``*_logits`` are ``(B,1)`` (or ``(B,)``) head logits. ``src_labels`` are the
        SOURCE labels (0/1). ``info`` is a dict of plain floats for logging (sup_loss,
        consistency, mu, mu_eff, acceptance, accepted_pos_fraction, threshold, n_accepted,
        src_confidence). The target's labels are NEVER used (pseudo-labels come only from
        the model's own aligned target-weak predictions).
    """

    def __init__(self, *, mu_max: float = 1.0, tau: float = 0.9, ema_momentum: float = 0.9,
                 consistency_warmup_frac: float = 0.1, align_reference=None,
                 eps: float = 1e-6):
        super().__init__()
        if not (0.0 <= ema_momentum < 1.0):
            raise ValueError(f"ema_momentum must be in [0,1), got {ema_momentum}")
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0,1], got {tau}")
        if not (0.0 <= consistency_warmup_frac < 1.0):
            raise ValueError(f"consistency_warmup_frac must be in [0,1), got {consistency_warmup_frac}")
        self.mu_max = float(mu_max)
        self.tau = float(tau)
        self.ema_momentum = float(ema_momentum)
        self.consistency_warmup_frac = float(consistency_warmup_frac)
        self.eps = float(eps)

        # running (detached) EMAs used as the alignment / threshold references
        self.register_buffer("ema_src_dist", torch.zeros(2))
        self.register_buffer("ema_tgt_dist", torch.zeros(2))
        self.register_buffer("ema_src_conf", torch.zeros(1))
        self.register_buffer("ema_ready", torch.zeros(1))

        if align_reference is not None:
            ar = torch.as_tensor(align_reference, dtype=torch.float32).view(-1)
            if ar.numel() != 2:
                raise ValueError(f"align_reference must have 2 entries, got {tuple(ar.shape)}")
            ar = (ar / ar.sum().clamp_min(self.eps)).contiguous()
            self.register_buffer("align_reference", ar)
            self.has_fixed_reference = True
        else:
            self.register_buffer("align_reference", torch.zeros(2))
            self.has_fixed_reference = False

        # last-computed diagnostics (floats; not part of the graph)
        self.last_sup = 0.0
        self.last_consistency = 0.0
        self.last_mu = 0.0
        self.last_mu_eff = 0.0
        self.last_acceptance = 0.0
        self.last_accepted_pos_fraction = float("nan")
        self.last_threshold = 0.0
        self.last_n_accepted = 0.0
        self.last_src_confidence = 0.0

    def reset_ema(self):
        """Clear the running EMAs (call between folds / fresh runs)."""
        with torch.no_grad():
            self.ema_src_dist.zero_()
            self.ema_tgt_dist.zero_()
            self.ema_src_conf.zero_()
            self.ema_ready.zero_()

    @staticmethod
    def _prob2(logits: torch.Tensor) -> torch.Tensor:
        """(B,1)|(B,) binary logits -> (B,2) [p_neg, p_pos] via sigmoid."""
        p = torch.sigmoid(logits.reshape(-1))
        return torch.stack([1.0 - p, p], dim=1)

    def forward(self, src_weak_logits, src_strong_logits, tgt_weak_logits,
                tgt_strong_logits, src_labels, step: int, total_steps: int):
        # head-only-graph guard: the STRONG-view logits (which carry the loss gradient)
        # MUST come from the trainable head, never the detached frozen-encoder features.
        if self.training and torch.is_grad_enabled():
            assert src_strong_logits.requires_grad and tgt_strong_logits.requires_grad, (
                "AdaMatch expected grad-carrying STRONG-view logits (trainable head "
                "output); got detached tensors -- is the head wired to the frozen encoder "
                "features by mistake?")

        total_steps = max(int(total_steps), 1)

        # ---- supervised: BCE on the SOURCE STRONG view (labels used here only) ----
        y = src_labels.detach().float().reshape(-1)
        sup_loss = F.binary_cross_entropy_with_logits(src_strong_logits.reshape(-1), y)

        # ---- pseudo-label pipeline on the WEAK views (all detached: these are TARGETS) ----
        with torch.no_grad():
            sw = self._prob2(src_weak_logits)                 # (Bs,2)
            tw = self._prob2(tgt_weak_logits)                 # (Bt,2)
            src_dist_b = sw.mean(0)                            # (2,)
            tgt_dist_b = tw.mean(0)                            # (2,)
            src_conf_b = sw.max(dim=1).values.mean().reshape(1)   # (1,)

            m = self.ema_momentum
            if float(self.ema_ready) < 1.0:                   # seed EMAs on first sight
                self.ema_src_dist.copy_(src_dist_b)
                self.ema_tgt_dist.copy_(tgt_dist_b)
                self.ema_src_conf.copy_(src_conf_b)
                self.ema_ready.fill_(1.0)
            else:
                self.ema_src_dist.mul_(m).add_(src_dist_b * (1.0 - m))
                self.ema_tgt_dist.mul_(m).add_(tgt_dist_b * (1.0 - m))
                self.ema_src_conf.mul_(m).add_(src_conf_b * (1.0 - m))

            ref = self.align_reference if self.has_fixed_reference else self.ema_src_dist
            aligned = align_target_distribution(tw, ref, self.ema_tgt_dist, self.eps)
            aligned_max, pseudo = aligned.max(dim=1)          # (Bt,), (Bt,) argmax class
            threshold = float(self.tau) * float(self.ema_src_conf)
            mask = (aligned_max >= threshold).to(aligned.dtype)   # (Bt,)
            n_acc = float(mask.sum())
            acceptance = float(mask.mean())
            accepted_pos_fraction = (float((pseudo.to(aligned.dtype) * mask).sum() / n_acc)
                                     if n_acc > 0 else float("nan"))
            pseudo_f = pseudo.to(aligned.dtype)

        # ---- consistency: masked BCE(target STRONG logits, pseudo-label) ----
        ce = F.binary_cross_entropy_with_logits(
            tgt_strong_logits.reshape(-1), pseudo_f, reduction="none")   # (Bt,)
        consistency = (ce * mask).mean()                      # mean over the full target batch

        # ---- mu schedule + hard consistency-warmup gate ----
        mu = adamatch_mu_at_step(step, total_steps, self.mu_max)
        warmup_steps = int(math.ceil(self.consistency_warmup_frac * total_steps))
        mu_eff = 0.0 if step < warmup_steps else mu
        loss = sup_loss + mu_eff * consistency

        assert torch.isfinite(loss), f"AdaMatch loss is non-finite: {float(loss)!r}"

        self.last_sup = float(sup_loss.detach())
        self.last_consistency = float(consistency.detach())
        self.last_mu = float(mu)
        self.last_mu_eff = float(mu_eff)
        self.last_acceptance = acceptance
        self.last_accepted_pos_fraction = accepted_pos_fraction
        self.last_threshold = threshold
        self.last_n_accepted = n_acc
        self.last_src_confidence = float(self.ema_src_conf)

        info = {"sup_loss": self.last_sup, "consistency": self.last_consistency,
                "mu": self.last_mu, "mu_eff": self.last_mu_eff,
                "acceptance": acceptance, "accepted_pos_fraction": accepted_pos_fraction,
                "threshold": threshold, "n_accepted": n_acc,
                "src_confidence": self.last_src_confidence}
        return loss, info

    def extra_repr(self) -> str:
        return (f"mu_max={self.mu_max}, tau={self.tau}, ema_momentum={self.ema_momentum}, "
                f"consistency_warmup_frac={self.consistency_warmup_frac}, "
                f"fixed_reference={self.has_fixed_reference}")


# ===========================================================================
# FreeMatch  (``FreeMatchLoss``)  -- the THIRD real DA method
# ===========================================================================
#
# Canonical references followed:
#   * Wang, Chen, Fan, Sun, Tao, Hou, Wang, Yang, Schiele, Xie, Raj, Savvides,
#     "FreeMatch: Self-Adaptive Thresholding for Semi-Supervised Learning"
#     (ICLR 2023, arXiv:2205.07246).
#   * microsoft/Semi-supervised-learning (USB) reference implementation -- the
#     ``FreeMatchThresholingHook`` (self-adaptive threshold) + the self-adaptive
#     fairness (SAF) entropy term.
#
# Like AdaMatch, FreeMatch is a consistency method needing WEAK + STRONG augmented
# views produced PRE-encoder, so it CANNOT run on the cached features: it runs through
# the image-space Trainer with the FROZEN encoder in the loop (encoder under no_grad,
# only the head trains). It reuses AdaMatch's scaffolding verbatim -- LensDataset
# n_views=2, the paired balanced loaders, the per-fold runner, the metric suite, and the
# ConfirmationBiasDetector.
#
# WHAT WE IMPLEMENT (the three FreeMatch legs on a frozen LayerNorm head):
#   1. SELF-ADAPTIVE THRESHOLD (SAT): a global confidence EMA tau_t plus a per-class
#      EMA pbar_t(c) that defines a LOCAL, class-wise threshold tau_t(c). No fixed 0.95
#      like FixMatch, no source-referenced relative threshold like AdaMatch -- the
#      threshold is estimated ONLY from the model's own confidence on the unlabeled
#      target, and it starts low (1/C) and self-raises as the model gets confident.
#   2. SELF-ADAPTIVE FAIRNESS (SAF): a regulariser pushing the batch-mean predicted
#      distribution over CONFIDENT unlabeled samples toward a target distribution. Our
#      adaptation pool is ~1:1 balanced (PROJECT.md 4.3), so the fairness target is
#      uniform [0.5, 0.5] -- this is the anti-collapse mechanism AdaMatch lacked.
#   3. the FixMatch consistency backbone (pseudo-label from weak, masked CE on strong).
#
#   Total loss = BCE(source) + mu(step) * consistency + saf_weight * SAF.
#   The mu cosine warmup + hard consistency-warmup gate are reused from AdaMatch (defer
#   pseudo-labeling early). Random logit interpolation is intentionally omitted (frozen
#   encoder + LayerNorm head -> no BN statistics to interpolate; see AdaMatch docstring).
#
# ****  THE MAKE-OR-BREAK KNOB: local_threshold_mode  ****
#   FreeMatch's local threshold LOWERS the bar for the UNDER-LEARNED class (the one with
#   the smaller pbar). Here the under-learned class is "lens" (rare positives), and
#   lowering its bar is exactly the confirmation-bias trap AdaMatch collapsed into --
#   admitting false lens pseudo-labels that destroy purity. So local_threshold_mode is a
#   FIRST-CLASS config, NOT a silent port of the paper:
#     * "standard": the paper's tau_t(c) = (pbar_t(c)/max_c pbar_t(c)) * tau_t.
#     * "floor":    never let the lens-class local threshold drop below the global tau_t,
#                   i.e. tau_t(lens) = max(tau_t(lens), tau_t) -- prevents the minority-
#                   lowering from admitting weak lens pseudo-labels.
#   Both are logged so the fold-0 screen can compare them.
#
# ONE STATED DEVIATION (build request wording): masking accepts a sample for class c*
#   when its confidence is STRICTLY GREATER than tau_t(c*) (the request says "> tau_t(c*)").
#   The USB reference uses ">=" (torch.ge); the difference is a measure-zero boundary and
#   we follow the request's ">".


def binary_prob2(logits: torch.Tensor) -> torch.Tensor:
    """(B,1)|(B,) binary logits -> (B,2) [p_neg, p_pos] via sigmoid.

    The 2-class probability view our binary sigmoid head implies, so the FreeMatch
    C-class SAT/SAF formulas apply unchanged with C=2 ([neg, pos] = [non-lens, lens])."""
    p = torch.sigmoid(logits.reshape(-1))
    return torch.stack([1.0 - p, p], dim=1)


def freematch_sat_update(prob_w: torch.Tensor, tau_prev, pbar_prev, momentum: float):
    """One Self-Adaptive Threshold EMA update from a weak-aug UNLABELED probability batch.

    Given ``prob_w`` (B,C) [the model's softmax/2-class probabilities on the weakly-
    augmented unlabeled batch] and the previous EMAs, returns the updated
    ``(tau_new, pbar_new)`` following FreeMatch Eq. (7)-(8):

        tau_t   = m*tau_{t-1}  + (1-m)*mean_b( max_c prob_w[b,c] )      (global)
        pbar_t  = m*pbar_{t-1} + (1-m)*mean_b( prob_w[b, :] )          (per-class)

    Pure/stateless (operates on the passed-in tensors), so the hand-example unit test can
    pin the exact arithmetic. ``m`` is the EMA momentum (FreeMatch default 0.999)."""
    if prob_w.dim() != 2:
        raise ValueError(f"expected (B,C) weak probs, got {tuple(prob_w.shape)}")
    max_prob = prob_w.max(dim=1).values
    tau_new = momentum * tau_prev + (1.0 - momentum) * max_prob.mean()
    pbar_new = momentum * pbar_prev + (1.0 - momentum) * prob_w.mean(0)
    return tau_new, pbar_new


def freematch_local_thresholds(tau_global, pbar, *, pos_class: int = 1,
                               mode: str = "standard") -> torch.Tensor:
    """The per-class LOCAL thresholds tau_t(c) from the global tau_t and per-class pbar_t.

    FreeMatch Eq. (9) local threshold:

        tau_t(c) = ( pbar_t(c) / max_c' pbar_t(c') ) * tau_t

    ``mode="standard"`` returns exactly that. ``mode="floor"`` additionally clamps the
    POSITIVE / lens class so its local threshold never drops below the global tau_t:

        tau_t(lens) = max( tau_t(lens), tau_t )

    which blocks the minority-lowering that would otherwise admit weak lens pseudo-labels
    (our rare-positive confirmation-bias guard). Returns a (C,) tensor."""
    if mode not in FREEMATCH_LOCAL_MODES:
        raise ValueError(f"local_threshold_mode must be one of {FREEMATCH_LOCAL_MODES}, "
                         f"got {mode!r}")
    local = (pbar / pbar.max()) * tau_global
    if mode == "floor":
        local = local.clone()
        local[pos_class] = torch.maximum(local[pos_class],
                                         torch.as_tensor(tau_global, dtype=local.dtype,
                                                         device=local.device))
    return local


def self_adaptive_fairness_penalty(mean_pred: torch.Tensor, fairness_target: torch.Tensor,
                                   eps: float = 1e-6) -> torch.Tensor:
    """SAF penalty = KL( fairness_target || mean_pred ), a scalar >= 0.

    ``mean_pred`` is the batch-mean predicted class distribution over the CONFIDENT
    (masked) unlabeled samples; ``fairness_target`` is the desired marginal (uniform
    [0.5,0.5] for our balanced adaptation pool). This is FreeMatch's self-adaptive
    fairness cross-entropy, written as the KL divergence

        L_SAF = sum_c ft_c * ( log ft_c - log mean_pred_c )

    which equals the cross-entropy H(ft, mean_pred) offset by the (constant) target
    entropy H(ft), so the penalty is EXACTLY 0 when mean_pred == fairness_target and
    grows without bound as the confident predictions collapse to a single class
    (mean_pred_c -> 0). Minimising it pushes the confident-sample marginal back toward
    balance -- the collapse the AdaMatch run had no term to resist. Gradient flows through
    ``mean_pred`` (the strong-aug predictions), never the frozen encoder."""
    if mean_pred.dim() != 1 or fairness_target.dim() != 1:
        raise ValueError("SAF expects 1-D (C,) distributions")
    mp = mean_pred.clamp_min(eps)
    ft = fairness_target.clamp_min(eps)
    return (ft * (torch.log(ft) - torch.log(mp))).sum()


FREEMATCH_LOCAL_MODES = ("standard", "floor")


# ---------------------------------------------------------------------------
# FreeMatch loss module
# ---------------------------------------------------------------------------

class FreeMatchLoss(nn.Module):
    """FreeMatch objective (binary) on head LOGITS of weak/strong views.

    Total loss = BCE(source logits, source labels)
                 + mu(step) * consistency
                 + saf_weight * SAF,
    where the pseudo-labels come from the SELF-ADAPTIVE-THRESHOLD-masked weak-aug TARGET
    predictions and the consistency is the FixMatch masked BCE of the strong-aug target
    logits against them. SAT self-estimates a global tau_t and per-class local tau_t(c)
    purely from the model's confidence on the unlabeled target (no fixed threshold, no
    source reference). SAF regularises the confident-sample marginal toward a balanced
    target. RLI is omitted (frozen encoder + LayerNorm head; see module docstring).

    Parameters
    ----------
    num_classes : int, default 2
        C (our binary lens/non-lens problem is C=2). Sizes the SAT buffers; tau init
        1/C = 0.5, pbar init 1/C each -- the FreeMatch initialisation.
    mu_max : float, default 1.0
        Peak consistency weight (FIXED, not tuned on target). See ``adamatch_mu_at_step``.
    saf_weight : float, default 0.01
        Fixed weight on the self-adaptive-fairness penalty (the FreeMatch/USB fairness
        ``ent_loss_ratio`` default). NOT tuned on the target.
    sat_ema_momentum : float, default 0.999
        EMA momentum m for the SAT global/per-class statistics (FreeMatch default 0.999).
    consistency_warmup_frac : float, default 0.1
        Fraction of total steps to WITHHOLD the consistency loss entirely (mu_eff=0), a
        hard floor on top of the mu warmup so early garbage pseudo-labels can't seed the
        collapse (reused from AdaMatch).
    local_threshold_mode : {"standard","floor"}, default "standard"
        The make-or-break knob (see module docstring). "floor" clamps the lens-class
        local threshold to be >= the global tau_t.
    pos_class : int, default 1
        Which class index is "lens" (the rare positive) -- the class ``floor`` protects.
    fairness_target : sequence[float] | None, default None
        SAF target marginal. None -> uniform [1/C]*C (balanced, matches our 1:1 pool). An
        explicit distribution (renormalised) overrides it (e.g. an imbalanced ablation).
    eps : float, default 1e-6
        Numerical floor for the SAF logs.

    forward(src_logits, tgt_weak_logits, tgt_strong_logits, src_labels, step, total_steps)
        -> (loss, info). ``src_logits`` are the SOURCE (labeled) logits used for the
        supervised BCE; ``tgt_weak_logits`` drive SAT + pseudo-labeling (detached);
        ``tgt_strong_logits`` carry the consistency + SAF gradient. ``info`` is a dict of
        plain floats for logging (sup_loss, consistency, saf, mu, mu_eff, acceptance,
        accepted_pos_fraction, tau_global, local_neg, local_pos, n_accepted). The target's
        labels are NEVER used (pseudo-labels come only from the model's own SAT-masked
        target-weak predictions)."""

    def __init__(self, *, num_classes: int = 2, mu_max: float = 1.0,
                 saf_weight: float = 0.01, sat_ema_momentum: float = 0.999,
                 consistency_warmup_frac: float = 0.1,
                 local_threshold_mode: str = "standard", pos_class: int = 1,
                 fairness_target=None, eps: float = 1e-6):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >=2, got {num_classes}")
        if not (0.0 <= sat_ema_momentum < 1.0):
            raise ValueError(f"sat_ema_momentum must be in [0,1), got {sat_ema_momentum}")
        if not (0.0 <= consistency_warmup_frac < 1.0):
            raise ValueError(f"consistency_warmup_frac must be in [0,1), got "
                             f"{consistency_warmup_frac}")
        if local_threshold_mode not in FREEMATCH_LOCAL_MODES:
            raise ValueError(f"local_threshold_mode must be one of {FREEMATCH_LOCAL_MODES}, "
                             f"got {local_threshold_mode!r}")
        if not (0 <= pos_class < num_classes):
            raise ValueError(f"pos_class must be in [0,{num_classes}), got {pos_class}")
        self.num_classes = int(num_classes)
        self.mu_max = float(mu_max)
        self.saf_weight = float(saf_weight)
        self.sat_ema_momentum = float(sat_ema_momentum)
        self.consistency_warmup_frac = float(consistency_warmup_frac)
        self.local_threshold_mode = str(local_threshold_mode)
        self.pos_class = int(pos_class)
        self.eps = float(eps)

        # SAT statistics live in buffers (detached history, serialized, moved by .to()).
        # FreeMatch initialisation: tau=1/C, pbar=1/C each (a uniform, low starting bar).
        init = 1.0 / self.num_classes
        self.register_buffer("tau_global", torch.tensor(init))
        self.register_buffer("pbar", torch.full((self.num_classes,), init))

        if fairness_target is None:
            ft = torch.full((self.num_classes,), init)
        else:
            ft = torch.as_tensor(fairness_target, dtype=torch.float32).view(-1)
            if ft.numel() != self.num_classes:
                raise ValueError(f"fairness_target must have {self.num_classes} entries, "
                                 f"got {tuple(ft.shape)}")
            ft = (ft / ft.sum().clamp_min(self.eps)).contiguous()
        self.register_buffer("fairness_target", ft)

        # last-computed diagnostics (floats; not part of the graph)
        self.last_sup = 0.0
        self.last_consistency = 0.0
        self.last_saf = 0.0
        self.last_mu = 0.0
        self.last_mu_eff = 0.0
        self.last_acceptance = 0.0
        self.last_accepted_pos_fraction = float("nan")
        self.last_tau_global = float(init)
        self.last_local = [float(init)] * self.num_classes
        self.last_n_accepted = 0.0

    def reset_ema(self):
        """Reset the SAT statistics to the FreeMatch init (call between folds / runs)."""
        with torch.no_grad():
            init = 1.0 / self.num_classes
            self.tau_global.fill_(init)
            self.pbar.fill_(init)

    def forward(self, src_logits, tgt_weak_logits, tgt_strong_logits, src_labels,
                step: int, total_steps: int):
        # head-only-graph guard: the SOURCE logits (supervised gradient) and the TARGET
        # STRONG-view logits (consistency + SAF gradient) MUST come from the trainable
        # head, never the detached frozen-encoder features.
        if self.training and torch.is_grad_enabled():
            assert src_logits.requires_grad and tgt_strong_logits.requires_grad, (
                "FreeMatch expected grad-carrying SOURCE + target STRONG logits (trainable "
                "head output); got detached tensors -- is the head wired to the frozen "
                "encoder features by mistake?")

        total_steps = max(int(total_steps), 1)

        # ---- supervised: BCE on the SOURCE (labels used here only) ----
        y = src_labels.detach().float().reshape(-1)
        sup_loss = F.binary_cross_entropy_with_logits(src_logits.reshape(-1), y)

        # ---- SAT update + masking on the WEAK target view (all detached: a TARGET) ----
        with torch.no_grad():
            prob_w = binary_prob2(tgt_weak_logits)                 # (Bt,2)
            tau_new, pbar_new = freematch_sat_update(
                prob_w, self.tau_global, self.pbar, self.sat_ema_momentum)
            self.tau_global.copy_(tau_new)
            self.pbar.copy_(pbar_new)
            local = freematch_local_thresholds(
                self.tau_global, self.pbar, pos_class=self.pos_class,
                mode=self.local_threshold_mode)                    # (2,)
            max_prob, pseudo = prob_w.max(dim=1)                   # (Bt,), argmax class
            thr_per = local[pseudo]                                # per-sample local thr
            mask = (max_prob > thr_per).to(prob_w.dtype)           # strict '>' (request)
            n_acc = float(mask.sum())
            acceptance = float(mask.mean())
            accepted_pos_fraction = (
                float((pseudo.to(prob_w.dtype) * mask).sum() / n_acc)
                if n_acc > 0 else float("nan"))
            pseudo_f = pseudo.to(prob_w.dtype)

            # ---- PER-CLASS mask-rate diagnostics (BRIEF 5.3) -- purely additive ----
            # Nothing here enters the loss, the SAT/SAF math or the graph: these are counts
            # taken from tensors already computed above, inside the same no_grad block. The
            # brief asks for "mask rate ... per class" and "pseudo-label class distribution",
            # and the existing `acceptance` (whole-batch) plus `accepted_pos_fraction`
            # (composition of the ACCEPTED set) cannot be decomposed into them after the
            # fact -- the per-class DENOMINATORS are needed and are only available here.
            is_pos = (pseudo == self.pos_class)
            n_pseudo_pos = float(is_pos.sum())
            n_pseudo_neg = float(pseudo.numel() - n_pseudo_pos)
            n_acc_pos = float((mask * is_pos.to(mask.dtype)).sum())
            n_acc_neg = n_acc - n_acc_pos
            # mask rate WITHIN each argmax class (nan when the class is absent this batch --
            # a rate over an empty denominator is undefined, never 0; PROJECT.md 7.4 reports
            # undefined metrics as undefined rather than fabricating a value).
            mask_rate_pos = (n_acc_pos / n_pseudo_pos) if n_pseudo_pos > 0 else float("nan")
            mask_rate_neg = (n_acc_neg / n_pseudo_neg) if n_pseudo_neg > 0 else float("nan")
            # the pre-mask pseudo-label class distribution: is SAF keeping the model's raw
            # predictions diverse, independent of what the threshold then admits?
            pseudo_pos_fraction_all = float(is_pos.to(prob_w.dtype).mean())
            mean_conf_target = float(max_prob.mean())

        # ---- consistency: masked BCE(target STRONG logits, pseudo-label) ----
        ce = F.binary_cross_entropy_with_logits(
            tgt_strong_logits.reshape(-1), pseudo_f, reduction="none")   # (Bt,)
        consistency = (ce * mask).mean()                          # mean over full batch

        # ---- SAF: push the CONFIDENT-sample strong-aug marginal toward the target ----
        prob_s = binary_prob2(tgt_strong_logits)                  # (Bt,2), grad-carrying
        if n_acc > 0:
            mean_pred = (prob_s * mask.unsqueeze(1)).sum(0) / n_acc     # (2,)
            saf = self_adaptive_fairness_penalty(mean_pred, self.fairness_target, self.eps)
        else:
            saf = prob_s.sum() * 0.0                               # 0, keeps the graph

        # ---- mu schedule + hard consistency-warmup gate (reused from AdaMatch) ----
        mu = adamatch_mu_at_step(step, total_steps, self.mu_max)
        warmup_steps = int(math.ceil(self.consistency_warmup_frac * total_steps))
        mu_eff = 0.0 if step < warmup_steps else mu
        loss = sup_loss + mu_eff * consistency + self.saf_weight * saf

        assert torch.isfinite(loss), f"FreeMatch loss is non-finite: {float(loss)!r}"

        self.last_sup = float(sup_loss.detach())
        self.last_consistency = float(consistency.detach())
        self.last_saf = float(saf.detach())
        self.last_mu = float(mu)
        self.last_mu_eff = float(mu_eff)
        self.last_acceptance = acceptance
        self.last_accepted_pos_fraction = accepted_pos_fraction
        self.last_tau_global = float(self.tau_global)
        self.last_local = [float(x) for x in local]
        self.last_n_accepted = n_acc

        info = {"sup_loss": self.last_sup, "consistency": self.last_consistency,
                "saf": self.last_saf, "mu": self.last_mu, "mu_eff": self.last_mu_eff,
                "acceptance": acceptance, "accepted_pos_fraction": accepted_pos_fraction,
                "tau_global": self.last_tau_global,
                "local_neg": self.last_local[0], "local_pos": self.last_local[self.pos_class],
                "n_accepted": n_acc,
                # BRIEF 5.3 additions (diagnostics only -- see the no_grad block above)
                "mask_rate_pos": mask_rate_pos, "mask_rate_neg": mask_rate_neg,
                "n_pseudo_pos": n_pseudo_pos, "n_pseudo_neg": n_pseudo_neg,
                "n_accepted_pos": n_acc_pos, "n_accepted_neg": n_acc_neg,
                "pseudo_pos_fraction_all": pseudo_pos_fraction_all,
                "mean_confidence_target": mean_conf_target,
                "pbar_neg": float(self.pbar[0]),
                "pbar_pos": float(self.pbar[self.pos_class])}
        return loss, info

    def extra_repr(self) -> str:
        return (f"num_classes={self.num_classes}, mu_max={self.mu_max}, "
                f"saf_weight={self.saf_weight}, sat_ema_momentum={self.sat_ema_momentum}, "
                f"consistency_warmup_frac={self.consistency_warmup_frac}, "
                f"local_threshold_mode={self.local_threshold_mode!r}, pos_class={self.pos_class}")


# ===========================================================================
# DANN  (``GradientReversal`` / ``DomainDiscriminator`` / ``DannLoss``)
# ===========================================================================
#
# ROUND 5. Domain-Adversarial Neural Networks -- Ganin & Lempitsky
# (arXiv:1409.7495), Ganin et al. JMLR 2016 (arXiv:1505.07818).
#
# WHAT THIS IS, AND WHAT IT IS NOT. ``run_dann.py`` (round 1) carries its OWN
# GRL + discriminator, wired to a bottleneck adapter over the CACHED frozen
# features. That code is untouched and its fold-0 probe stays reproducible. The
# three classes below are the IMAGE-SPACE round-5 pieces: they attach to the same
# 640-d post-LayerNorm adapter tensor CORAL uses (``head.forward_with_features``'s
# second return), so at ``n_blocks=3`` the adversarial gradient flows through the
# adapter and into the encoder. That is the experiment; do not move the attachment.
#
# FOUR DELIBERATE CHOICES, each of which a reader will want to challenge:
#
#   1. **LayerNorm inside D, not BatchNorm.** The reference ``tllib``
#      ``DomainDiscriminator`` uses BatchNorm1d. We substitute LayerNorm. Reason:
#      D is fed a SINGLE tensor holding both domains, so a batch-coupled
#      normalisation makes every sample's activation depend on the domain
#      composition of the batch -- a leak path that lets D read the domain label
#      out of the normalisation statistics rather than out of the features. The
#      encoder is LayerNorm-throughout already (PROJECT.md 7.1a: 0 BatchNorm
#      modules), so this also keeps the assembled model BN-free end to end and
#      keeps train/eval mode a no-op. Do not "fix" it back.
#
#   2. **ONE forward pass through D, on the concatenation.** Two separate calls
#      D(src) and D(tgt) would, under ANY batch-dependent normalisation, hand D
#      the domain label for free. The single-pass form removes the question
#      permanently rather than relying on choice 1 to hold forever.
#      ``DannLoss`` asserts the concatenation; ``test_da_dann.py`` counts the call.
#
#   3. **Domain label convention: SOURCE = 1, TARGET = 0.** Fixed here, documented
#      here, never flipped. Every downstream monitor (``d_pred_source_frac``) reads
#      this convention.
#
#   4. **``BCEWithLogitsLoss``, never ``sigmoid`` + ``BCELoss``.** The runs are fp16
#      under autocast; the fused form is the numerically stable one (and autocast
#      promotes it to fp32 automatically, which the split form does not get).
#      ``DomainDiscriminator`` therefore returns LOGITS and applies no sigmoid.
#
# THE OBJECTIVE. Total loss = BCE(source labels) + L_domain, with the GRL between
# the adapter features and D. D minimises L_domain at full strength; the gradient
# reaching the features (and, at n_blocks>0, the encoder) is multiplied by
# ``-lambda``. That is why DANN needs no D-vs-encoder learning-rate balance and why
# it was chosen over ADDA -- see the runner's ONE-OPTIMISER note.

DANN_LAMBDA_HORIZON = 2000      # DECLARED ABSOLUTE CONSTANT -- see dann_lambda_at_step
DANN_SOURCE_LABEL = 1.0         # the convention, pinned
DANN_TARGET_LABEL = 0.0
DANN_HIDDEN = 1024              # tllib's DomainDiscriminator width (Ganin's Office D)
DANN_DROPOUT = 0.2              # Belfiore et al. 2024 (A&A), astronomy sim->real DANN


def dann_lambda_at_step(step: int, lambda_horizon: int = DANN_LAMBDA_HORIZON) -> float:
    """The canonical DANN GRL coefficient at 0-based global ``step``.

        lambda(step) = 2 / (1 + exp(-10 * p)) - 1 ,   p = min(step / horizon, 1)

    ``lambda(0) = 0`` (no adversary while the label head is still moving fast),
    rising smoothly to ~1.

    ------------------------------------------------------------------------
    ``lambda_horizon`` IS NOT ``total_steps``. READ THIS BEFORE CHANGING IT.
    ------------------------------------------------------------------------
    Round 1's ``run_dann.dann_lambda_at_step`` takes ``total_steps`` and computes
    ``p = step / total_steps``. That is the textbook formula and it is WRONG for
    PROJECT.md 7.14's protocol, in a way that produces no error and no warning:

        development run  8000 steps -> at step 2000, p = 0.25, lambda = 0.848
        confirmatory run 2000 steps -> at step 2000, p = 1.00, lambda = 0.999

    Two runs that are supposed to be the same trajectory see different objectives
    at the same step, so the development curve stops predicting the confirmatory
    result -- and BOTH runs still train, still converge and still produce plausible
    numbers. Same failure class as FreeMatch's warmup restarting on resume
    (PROJECT.md 7.10), which cost a round to find.

    So the horizon is a DECLARED ABSOLUTE CONSTANT: passed by flag, recorded in
    ``run_config.json``, and IDENTICAL in the development run and every
    confirmatory run. ``test_da_dann.py``'s horizon-invariance test is the load-
    bearing check and exists for exactly this.

    NO HIDDEN STATE. lambda is a pure function of ``(step, lambda_horizon)``: no
    buffer, no EMA, no counter, nothing that a resume could restart. A fresh object
    at step k and a resumed one at step k agree bit-for-bit, which is also tested.
    """
    horizon = int(lambda_horizon)
    if horizon < 1:
        raise ValueError(f"lambda_horizon must be >= 1, got {lambda_horizon}")
    p = min(max(float(step), 0.0) / float(horizon), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


class GradientReversal(torch.autograd.Function):
    """Identity forward; ``grad -> -lambda_ * grad`` backward.

    Sits between the 640-d adapter feature and the discriminator, and OUTSIDE the
    gradient-checkpointed encoder region (the checkpointed region ends at the
    encoder output; the adapter and everything above it are ordinary autograd).
    ``test_da_dann.py`` verifies that placement rather than arguing it.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)                 # identity, but a distinct node in the graph

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None      # None: lambda_ is not a tensor


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Functional form of :class:`GradientReversal`."""
    return GradientReversal.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """D: ``in_dim -> hidden -> hidden -> 1``, LayerNorm + ReLU + Dropout, LOGIT out.

    Topology follows ``tllib.modules.domain_discriminator.DomainDiscriminator``
    (640 -> 1024 -> 1024 -> 1, Ganin's Office discriminator) with TWO declared
    changes, both inherited rather than tuned:

      * **LayerNorm replaces BatchNorm1d**, for the domain-leak reason in this
        section's header.
      * **Dropout(0.2) after each hidden ReLU** -- Belfiore et al. (2024, A&A), who
        found it reduced overfitting in an astronomy sim->real DANN whose target pool
        was ~20x larger than ours (so if anything we need it more).

    > **AMBIGUITY, FLAGGED NOT SILENTLY RESOLVED.** The build brief says "insert
    > ``nn.Dropout(0.2)`` between each pair of hidden layers". Read literally, two
    > hidden layers form ONE pair and that is ONE dropout. Implemented here as
    > dropout after EACH hidden ReLU (TWO layers), which is the standard placement in
    > Ganin's own discriminator and in the astronomy replications, and the reading
    > consistent with "reduced overfitting". ``--dann-d-dropout-layers 1`` selects the
    > literal single-dropout reading; the count is recorded in ``run_config.json``.

    No sigmoid is applied here -- the caller uses ``BCEWithLogitsLoss``.

    **Dropout gives D train/eval semantics, which NOTHING else in this pipeline has**
    (PROJECT.md 7.1a: the encoder has 0 BatchNorm and 0 active dropout, so train/eval
    was a no-op everywhere until now). Two consequences the callers must honour:
    D trains in ``.train()``, and the ``d_acc_holdout`` memorisation monitor MUST be
    measured in ``.eval()`` and the mode restored afterwards -- a monitor measured
    under dropout is not comparable to one measured without it, and that monitor arms
    abort condition 1. See ``dann_round5.holdout_domain_stats``.
    """

    def __init__(self, in_dim: int = 640, hidden: int = DANN_HIDDEN,
                 dropout: float = DANN_DROPOUT, n_dropout_layers: int = 2):
        super().__init__()
        if in_dim < 1 or hidden < 1:
            raise ValueError(f"in_dim/hidden must be >=1, got {in_dim}/{hidden}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")
        if n_dropout_layers not in (0, 1, 2):
            raise ValueError(f"n_dropout_layers must be 0, 1 or 2, got {n_dropout_layers}")
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.n_dropout_layers = int(n_dropout_layers)

        def drop(after_block: int):
            # after_block 1 == after the FIRST hidden ReLU, 2 == after the second.
            # n_dropout_layers=1 keeps ONLY the first, i.e. the one that literally sits
            # BETWEEN the two hidden layers -- the brief's literal reading.
            if self.n_dropout_layers >= after_block:
                return [nn.Dropout(self.dropout)]
            return []

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.ReLU(inplace=True),
            *drop(1),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.ReLU(inplace=True),
            *drop(2),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"D expects a 2-D (B, {self.in_dim}) tensor, got "
                             f"{tuple(z.shape)}")
        if z.shape[1] != self.in_dim:
            raise ValueError(f"D expects width {self.in_dim}, got {z.shape[1]}")
        return self.net(z)                                    # (B, 1) LOGITS

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, hidden={self.hidden}, "
                f"norm=LayerNorm (NOT BatchNorm), dropout={self.dropout} "
                f"x{self.n_dropout_layers}")


class DannLoss(nn.Module):
    """The DANN domain term on adapter features: GRL -> D -> BCEWithLogits.

    Unlike :class:`CoralLoss` / :class:`AdaMatchLoss` / :class:`FreeMatchLoss`, this
    module OWNS TRAINABLE PARAMETERS (D). The runner adds them to the existing AdamW
    as one extra param group -- see ``run_dann_r5.py``'s ONE-OPTIMISER note -- and
    saves ``discriminator_state`` alongside every checkpoint, because
    ``Trainer.save`` only knows about the head, the standardizer and the unfrozen
    encoder blocks.

    Parameters
    ----------
    feat_dim : int
        Adapter width (640 for the linear head). Must match ``head.feat_dim``.
    hidden : int, default 1024
        D's hidden width (tllib's value).
    lambda_horizon : int, default 8000
        The DECLARED ABSOLUTE horizon of the lambda schedule. See
        :func:`dann_lambda_at_step` -- this is NOT the run length.
    lambda_zero : bool, default False
        Pin lambda to 0.0 at every step. **The GRL and D still run**; only the
        coefficient is zero, so the control arm exercises the identical code path
        (identical RNG consumption, identical parameter count, identical optimizer
        groups). This is what makes it a control rather than a different program.
    holdout_frac : float, default 0.10
        Recorded here for provenance only -- the held-out target slice is
        constructed by the runner (it is a property of the DATA, not of the loss)
        and is EXCLUDED FROM D'S LOSS by being removed from the target sampler's
        pool. See ``run_dann_r5.build_target_holdout``.

    forward(src_adapter, tgt_adapter, lambda_=None) -> scalar tensor
        The RAW domain BCE. Canonical DANN puts it into the total loss with
        coefficient 1 -- lambda lives in the GRL, not in front of the loss -- so the
        caller adds it unweighted.
    """

    def __init__(self, feat_dim: int = 640, *, hidden: int = DANN_HIDDEN,
                 lambda_horizon: int = DANN_LAMBDA_HORIZON, lambda_zero: bool = False,
                 holdout_frac: float = 0.10, dropout: float = DANN_DROPOUT,
                 n_dropout_layers: int = 2):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.lambda_horizon = int(lambda_horizon)
        self.lambda_zero = bool(lambda_zero)
        self.holdout_frac = float(holdout_frac)
        self.discriminator = DomainDiscriminator(self.feat_dim, hidden, dropout=dropout,
                                                 n_dropout_layers=n_dropout_layers)
        self.bce = nn.BCEWithLogitsLoss()

        # lambda for the CURRENT step. NOT hidden state: set_step() RECOMPUTES it
        # from `step` alone and stores nothing else, so it is reconstructible on
        # resume from the step counter with no buffer to restore. It is a plain
        # float (not a buffer) precisely so it can never be serialized and revived
        # stale.
        self.lambda_ = 0.0
        self.last_step = 0
        self.last_stats = {}

    # -- the schedule ------------------------------------------------------

    def lambda_at(self, step: int) -> float:
        """lambda for ``step`` under THIS module's declared horizon (0.0 if pinned)."""
        if self.lambda_zero:
            return 0.0
        return dann_lambda_at_step(step, self.lambda_horizon)

    def set_step(self, step: int) -> float:
        """Set the current lambda from ``step``. Pure recomputation, no counter."""
        self.last_step = int(step)
        self.lambda_ = self.lambda_at(step)
        return self.lambda_

    # -- forward -----------------------------------------------------------

    def forward(self, src_adapter: torch.Tensor, tgt_adapter: torch.Tensor,
                lambda_: float | None = None) -> torch.Tensor:
        if src_adapter.dim() != 2 or tgt_adapter.dim() != 2:
            raise ValueError(
                f"DANN expects 2-D adapter features; got src={tuple(src_adapter.shape)} "
                f"tgt={tuple(tgt_adapter.shape)}")
        if src_adapter.shape[1] != tgt_adapter.shape[1]:
            raise ValueError(f"source/target feature dims differ: "
                             f"{src_adapter.shape[1]} vs {tgt_adapter.shape[1]}")
        if src_adapter.shape[1] != self.feat_dim:
            raise ValueError(f"adapter dim {src_adapter.shape[1]} != configured feat_dim "
                             f"{self.feat_dim}")

        # Same head-only-graph guard CoralLoss carries: detached tensors here would
        # mean the caller wired the raw (detached) ENCODER features by mistake.
        if self.training and torch.is_grad_enabled():
            assert src_adapter.requires_grad and tgt_adapter.requires_grad, (
                "DANN expected grad-carrying ADAPTER features (trainable head output); "
                "got detached tensors -- is it wired to the frozen encoder features?")

        lam = self.lambda_ if lambda_ is None else float(lambda_)
        if self.lambda_zero:
            lam = 0.0

        n_s, n_t = int(src_adapter.shape[0]), int(tgt_adapter.shape[0])

        # ---- ONE tensor, ONE forward pass through D (design choice 2) ----
        z = torch.cat([src_adapter, tgt_adapter], dim=0)
        assert z.shape[0] == n_s + n_t, (
            f"the concatenation lost rows: {z.shape[0]} != {n_s} + {n_t}")
        logits = self.discriminator(grad_reverse(z, lam)).view(-1)

        d_labels = torch.cat([
            z.new_full((n_s,), DANN_SOURCE_LABEL),
            z.new_full((n_t,), DANN_TARGET_LABEL)]).float()
        loss = self.bce(logits.float(), d_labels)

        with torch.no_grad():
            pred_src = (logits > 0)                      # source == 1 == positive logit
            truth_src = torch.cat([torch.ones(n_s, dtype=torch.bool, device=z.device),
                                   torch.zeros(n_t, dtype=torch.bool, device=z.device)])
            acc = float((pred_src == truth_src).float().mean())
            self.last_stats = {
                "loss_domain_raw": float(loss.detach()),
                "lambda_current": float(lam),
                "d_acc_train": acc,
                # Logged SEPARATELY from accuracy on purpose: D can sit at 50%
                # accuracy by predicting one domain and oscillating, and accuracy
                # alone cannot tell that apart from the adversarial equilibrium.
                "d_pred_source_frac": float(pred_src.float().mean()),
                "d_acc_source": float(pred_src[:n_s].float().mean()),
                "d_acc_target": float((~pred_src[n_s:]).float().mean()),
                "d_logit_mean": float(logits.detach().float().mean()),
                "d_n_source": float(n_s), "d_n_target": float(n_t),
            }
        return loss

    def state_dict_d(self) -> dict:
        """D's weights, for the runner's checkpoint sidecar."""
        return {k: v.detach().cpu().clone()
                for k, v in self.discriminator.state_dict().items()}

    @torch.no_grad()
    def score_domain(self, feats: torch.Tensor) -> torch.Tensor:
        """D's logits on ``feats`` with NO GRL and NO grad -- monitor use only.

        Used for ``d_acc_holdout``. It never touches the loss, so it cannot train D
        on the held-out slice; ``test_da_dann.py`` counts D's calls inside
        :meth:`forward` rather than globally for exactly this reason.

        **Does NOT change D's mode.** Since D now carries dropout, the CALLER is
        responsible for putting D in ``.eval()`` and restoring it -- see
        :meth:`eval_mode` and ``dann_round5.holdout_domain_stats``. Deliberately not
        done here: a function that silently flips a module mode is how a training
        step ends up running under eval semantics.
        """
        return self.discriminator(feats).view(-1)

    @contextlib.contextmanager
    def eval_mode(self):
        """Put D in eval (dropout OFF) for a monitor, then RESTORE the previous mode.

        Restores what the mode WAS rather than forcing ``.train()``, so nesting this
        inside an eval block cannot silently switch training back on.

        This exists because D is the FIRST module in this pipeline for which
        ``.train()`` / ``.eval()`` means anything at all (PROJECT.md 7.1a measured 0
        BatchNorm and 0 active dropout in the encoder). ``d_acc_holdout`` is compared
        against ``d_acc_target`` on the training batch to detect memorisation, and a
        comparison between a dropout-on and a dropout-off measurement is not a
        comparison.
        """
        was = self.discriminator.training
        self.discriminator.eval()
        try:
            yield
        finally:
            self.discriminator.train(was)

    def extra_repr(self) -> str:
        return (f"feat_dim={self.feat_dim}, lambda_horizon={self.lambda_horizon}, "
                f"lambda_zero={self.lambda_zero}, holdout_frac={self.holdout_frac}, "
                f"labels: source={DANN_SOURCE_LABEL} target={DANN_TARGET_LABEL}")


# ---------------------------------------------------------------------------
# ROUND 6 -- BNM (Batch Nuclear-norm Maximization)
# ---------------------------------------------------------------------------
#
# Cui, Wang, Zhuo, Li, Huang & Tian, "Towards Discriminability and Diversity:
# Batch Nuclear-norm Maximization under Label Insufficient Situations", CVPR 2020
# (oral), arXiv:2003.12237; journal extension arXiv:2107.06154.
#
# The objective, on the target batch's prediction matrix A (B x C, row-stochastic):
#
#       L_bnm = - nuclear_norm(A) / min(B, C)
#
# Maximising the nuclear norm simultaneously raises PREDICTION CONFIDENCE (the nuclear
# norm is bounded by, and rises with, the Frobenius norm, which is maximal for one-hot
# rows) and PREDICTION DIVERSITY (for a given Frobenius norm the nuclear norm is maximal
# when A's mass spreads across independent directions, i.e. across classes). The
# min(B,C) normalisation is the paper's; with C=2 it is simply 2 for any batch of 2+.
#
# WHY THIS ARM EXISTS -- read PROJECT.md 7.20 before touching this. BNM is not a
# fourth entry in the nb3 ranking table (7.16 bounds any true difference there at
# +/-0.17 and the within-run checkpoint sd is 3-5x the differences a fourth row would
# speak to). It is the MINIMAL-TARGET-SIGNAL CONTROL: no domain comparison in the loss
# at all (CoralLoss and DannLoss both require source and target together; this touches
# target predictions only), no pseudo-labels, no thresholds, no EMA, no discriminator,
# no gradient reversal, no ramp. One term, one weight gamma.
#
# THE BINARY CAVEAT, which is the whole of Stage 0a. Our head emits ONE logit, so A is
# built through `binary_prob2` and C = 2 always. rank(A) <= 2, so the rank-1-vs-rank-C
# "diversity" pressure BNM exercises over Office-Home's 31 classes reduces here to
# rank-1-vs-rank-2. The objective still separates the states it needs to separate --
# see `bnm_analytic_reference` and test_da_bnm.py -- but the margin it gives gamma to
# work with is 41%, not the order of magnitude the multi-class setting offers. That
# number is measured, not assumed.
#
# NO RAMP ON GAMMA. Deliberate and load-bearing: BNM's appeal in this project is that
# it has no horizon, so unlike DANN (7.18) its budget faces no schedule constraint. If a
# ramp is ever added here, the arm stops being the minimal control.

BNM_EPS = 1e-12          # guards the entropy monitor's log only


def bnm_nuclear_norm(prob2: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Nuclear norm of a (B,C) row-stochastic prediction matrix, divided by min(B,C).

    Differentiable in ``prob2``. ``torch.linalg.svdvals`` carries a gradient wherever
    the singular values are distinct; the balanced-confident and collapsed-confident
    states we care about are both well away from the degenerate tie.
    """
    if prob2.dim() != 2:
        raise ValueError(f"BNM expects a (B,C) probability matrix, got {tuple(prob2.shape)}")
    sv = torch.linalg.svdvals(prob2.float())
    nuc = sv.sum()
    if not normalize:
        return nuc
    denom = float(min(prob2.shape[0], prob2.shape[1]))
    return nuc / denom


def bnm_loss(logits: torch.Tensor) -> torch.Tensor:
    """The BNM term for our one-logit binary head, negated so that it MINIMISES.

    TARGET-ONLY by construction -- it takes target logits and nothing else.
    """
    return -bnm_nuclear_norm(binary_prob2(logits))


def bnm_analytic_reference(batch: int = 256, n_classes: int = 2) -> dict:
    """The three analytic batch states Stage 0a pins, with BOTH norms.

    Returned so the unit test and the methods-section note read the SAME numbers
    rather than two hand-typed copies of them. For B=256, C=2, normalised by
    min(B,C)=2:

        collapsed-confident   nuc/2 =  8.000   frobenius = 16.000
        balanced-confident    nuc/2 = 11.314   frobenius = 16.000
        maximally-uncertain   nuc/2 =  5.657   frobenius = 11.314

    The FROBENIUS norm CANNOT distinguish collapsed-confident from balanced-confident
    (16.00 both); the NUCLEAR norm can (8.00 vs 11.31). That is the structural reason
    BNM is chosen over plain confidence maximisation, and it is why this function
    reports both.

    The honest limitation, recorded WITH the numbers rather than after them:
    collapsed-confident (8.00) still scores ABOVE maximally-uncertain (5.66), so BNM
    prefers collapse to uncertainty. The 41% collapsed-to-balanced margin is all that
    gamma has to work with.
    """
    if n_classes != 2:
        raise NotImplementedError("Stage-0a reference is stated for the binary head (C=2)")
    b = int(batch)
    states = {}

    collapsed = torch.zeros(b, 2)
    collapsed[:, 1] = 1.0
    balanced = torch.zeros(b, 2)
    balanced[: b // 2, 1] = 1.0
    balanced[b // 2:, 0] = 1.0
    uncertain = torch.full((b, 2), 0.5)

    for name, a in (("collapsed_confident", collapsed),
                    ("balanced_confident", balanced),
                    ("maximally_uncertain", uncertain)):
        states[name] = {
            "nuclear_over_min_bc": float(bnm_nuclear_norm(a)),
            "frobenius": float(torch.linalg.matrix_norm(a, ord="fro")),
        }
    lo = states["collapsed_confident"]["nuclear_over_min_bc"]
    hi = states["balanced_confident"]["nuclear_over_min_bc"]
    states["_margin_collapsed_to_balanced"] = (hi - lo) / lo
    return states


class BNMLoss(nn.Module):
    """Batch Nuclear-norm Maximization on the TARGET batch's predictions.

    Structurally the simplest module in this file, and that is the point. Unlike
    :class:`CoralLoss` / :class:`DannLoss` it never sees the source batch; unlike
    :class:`AdaMatchLoss` / :class:`FreeMatchLoss` it holds no EMA, no threshold and
    no pseudo-label state. It owns NO trainable parameters, so -- unlike
    :class:`DannLoss` -- the runner adds no optimizer param group and saves no extra
    state alongside a checkpoint.

    It is also STATELESS across steps, which is why it is exempt from 7.10's
    weights-only-continuation problem: there is nothing here to restore.

    Parameters
    ----------
    gamma : float
        The single weight. FIXED for the whole run -- see the no-ramp note above.

    forward(tgt_logits) -> scalar tensor
        The RAW (unweighted, negated) nuclear-norm term. The caller multiplies by
        ``gamma``, matching CoralLoss's contract.
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.last_stats = {}

    def forward(self, tgt_logits: torch.Tensor) -> torch.Tensor:
        if tgt_logits.dim() not in (1, 2):
            raise ValueError(f"BNM expects (B,) or (B,1) target logits, got "
                             f"{tuple(tgt_logits.shape)}")
        if tgt_logits.dim() == 2 and tgt_logits.shape[1] != 1:
            raise ValueError(f"BNM is wired to the ONE-LOGIT binary head; got "
                             f"{tgt_logits.shape[1]} logit columns")

        # Same head-only-graph guard CoralLoss and DannLoss carry, one level later:
        # detached logits here mean the caller wired a no_grad monitor forward.
        if self.training and torch.is_grad_enabled():
            assert tgt_logits.requires_grad, (
                "BNM expected grad-carrying TARGET logits; got a detached tensor -- "
                "is it wired to a no_grad monitor forward?")

        prob2 = binary_prob2(tgt_logits)
        nuc = bnm_nuclear_norm(prob2)

        with torch.no_grad():
            p_pos = prob2[:, 1]
            # The three BNM-specific instruments 7.20 3.5 requires at every checkpoint.
            # `bnm_pred_pos_frac` is the COLLAPSE MONITOR: it is the quantity BNM has
            # no explicit term protecting.
            ent = -(prob2.clamp_min(BNM_EPS).log() * prob2).sum(1).mean()
            self.last_stats = {
                "bnm_nuclear_norm": float(nuc),
                "bnm_target_entropy": float(ent),
                "bnm_pred_pos_frac": float((p_pos > 0.5).float().mean()),
                "bnm_mean_p_pos": float(p_pos.mean()),
            }
        return -nuc

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, target_only=True, stateful=False, ramp=None"
