"""
coral_round4.py
===============
The round-4 CORAL library: everything the two round-4 arms need that does NOT already
exist method-agnostically in ``train_skeleton`` / ``zoobot_encoder`` / ``unfreeze_instrument``.

Round 4 runs Deep CORAL IMAGE-SPACE (weak aug, fp16) in two arms that are identical in
every respect except ``n_blocks``:

    coral_nb0_control   n_blocks=0   -- the CORAL gradient reaches only the head adapter
    coral_nb3           n_blocks=3   -- the CORAL gradient reaches the encoder as well

**Nothing here touches ``run_coral.py`` or any round-1 artifact.** Round-1 CORAL
(``runs/bakeoff_coral/``) stays the round-1 result and is never diffed against these runs;
it is used for exactly one thing, the LP->FT phase-1 head and the step-0 known-null that
proves the head loaded correctly.

WHY THIS FILE EXISTS RATHER THAN AN EDIT TO ``da_losses.py``
-----------------------------------------------------------
Round 4 needs two behaviours the shipped ``CoralLoss`` does not have, and both are wanted
ONLY here:

  1. **fp32 covariance under fp16 autocast.** ``Trainer.train_step`` computes the CORAL term
     inside ``with self._autocast()``. Under autocast a 640x640 ``z.t() @ z`` and the
     ``.pow(2).mean()`` reduction over 409,600 entries would run in half precision. fp16
     carries ~3 decimal digits and saturates at 65504; the Frobenius reduction is exactly
     the shape of computation that loses a small aligned difference between two large
     matrices. :class:`Fp32CoralLoss` forces that whole computation to fp32 inside an
     autocast-DISABLED block.
  2. **The feature-scale monitor** (below) needs the two adapter tensors, labelled, once per
     step. The loss module is the only place that receives both with their domains known.

Editing ``da_losses.CoralLoss`` in place would change what a re-run of ``run_coral.py``
does. Subclassing changes nothing outside round 4 and is swapped in with a single
assignment (``trainer.coral = Fp32CoralLoss(...)``), because ``train_step`` calls
``self.coral(...)`` by attribute.

CONSTANT GAMMA -- THE SCHEDULE IS GONE FROM THIS PATH
-----------------------------------------------------
Round 1 applied ``warmup_cosine``: gamma ramped 0 -> peak over the first 10% of the planned
steps, then cosine-decayed to 0.1 * peak. With the R1 read-off at the last step, the model
was measured while gamma sat at ~0.100 -- the CORAL term was suppressed exactly where the
number was taken. Sun & Saenko (arXiv:1607.01719 §4) set lambda so the classification and
CORAL losses are of roughly the same magnitude at the END of training; a schedule that
decays the DA term into the read-off point inverts that. Round 4 therefore uses a CONSTANT
gamma, and ``--calibrate`` measures the value that satisfies Sun & Saenko's rule from loss
magnitudes alone. ``coral_gamma_at_step`` is deliberately NOT imported here.

THE FEATURE-SCALE MONITOR -- A PRE-REGISTERED PREDICTION, WRITTEN BEFORE THE RUNS
---------------------------------------------------------------------------------
Recorded here, in code, before either arm has been run, and NOT to be adjusted afterwards.

CORAL's objective is ``||Sigma_s - Sigma_t||_F^2`` (plus a first-moment term). Scaling the
representation by a factor ``eps`` scales every feature by ``eps``, hence both covariance
matrices by ``eps^2``, hence their difference by ``eps^2``, hence the SQUARED Frobenius norm
by **eps^4**. The mean term falls as eps^2. So the CORAL loss can be driven toward zero by
SHRINKING THE REPRESENTATION rather than by aligning anything: a degenerate solution that
the objective cannot distinguish from success.

What differs between the arms is how much capacity is available to do that:

  * ``nb0`` -- only the 640 LayerNorm GAINS can rescale the representation (the bias cannot:
    see :func:`coral_reachable_params`). Shrinking them also shrinks the logit, which the
    supervised BCE resists. This arm is the NEAR-NULL: a small, bounded drift at most.
  * ``nb3`` -- 14.8M encoder parameters can reshape the representation, including shrinking
    it, with the classifier free to re-scale afterwards. The BCE resistance is much weaker.

**The prediction:** a FALLING post-LayerNorm feature norm alongside a FALLING raw CORAL loss
is the collapse signature. Falling CORAL loss with a FLAT feature norm is genuine alignment.
The two are distinguishable only if the norm is logged, which is why it is logged from step
0 on both arms, per domain, at every curve point.

The monitor is READ-ONLY. It never enters a loss, never selects anything, and never stops a
run (PROJECT.md §7.6's Job-1 discipline for the collapse gate applies to it verbatim).
"""

from __future__ import annotations

import contextlib
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from da_losses import CoralLoss                                   # noqa: E402
from unfreeze_instrument import CurveInstrument, append_row       # noqa: E402


# ===========================================================================
# 1. the loss: fp32 covariance + the per-step feature-scale record
# ===========================================================================

class Fp32CoralLoss(CoralLoss):
    """:class:`da_losses.CoralLoss` with the whole moment computation pinned to fp32.

    Behaviourally identical to the parent at fp32 (verified bit-exact by
    ``test_coral_round4.py``); the only difference appears under ``torch.amp.autocast``,
    where the parent would compute a 640x640 covariance and a 409,600-term Frobenius
    reduction in half precision.

    The cast is a differentiable op, so the autograd graph is unbroken: gradients flow back
    through ``.float()`` into the fp16 activations and, at ``n_blocks>0``, into the encoder.

    It also records, per step and per DOMAIN, the feature-scale statistics of the two
    adapter tensors it is handed. This is the pre-registered collapse monitor described in
    the module docstring; it is a read-only by-product of a forward pass that already
    happened, so it costs one norm and one std over a (256, 640) tensor per step.
    """

    def __init__(self, feat_dim: int, **kw):
        super().__init__(feat_dim, **kw)
        self.last_raw_loss = float("nan")
        self.last_scale = {}
        self.n_forward = 0

    @staticmethod
    def scale_stats(z: torch.Tensor) -> tuple[float, float]:
        """(mean L2 norm of a feature VECTOR, mean per-dimension std ACROSS the batch).

        The first is a per-sample quantity averaged over the batch: with a LayerNorm at unit
        gain it is sqrt(D) = 25.298 for D=640, which is the reference value the monitor is
        read against. The second is a per-dimension quantity averaged over dimensions -- it
        is the diagonal of the very covariance CORAL is minimising, so it is the direct
        scale of the objective rather than a proxy for it.
        """
        z = z.detach().float()
        l2 = float(z.norm(dim=1).mean())
        sd = float(z.std(dim=0, unbiased=True).mean()) if z.shape[0] > 1 else float("nan")
        return l2, sd

    def forward(self, src_adapter: torch.Tensor, tgt_adapter: torch.Tensor) -> torch.Tensor:
        # Autocast is disabled for the DEVICE THE TENSORS ARE ON, not unconditionally for
        # "cuda": the same code has to run on a CPU-only test machine, where
        # autocast("cuda", ...) is meaningless.
        dev_type = src_adapter.device.type
        with torch.amp.autocast(dev_type, enabled=False):
            s32 = src_adapter.float()
            t32 = tgt_adapter.float()
            loss = super().forward(s32, t32)

        # -- pre-registered feature-scale monitor (read-only) --
        s_l2, s_sd = self.scale_stats(s32)
        t_l2, t_sd = self.scale_stats(t32)
        self.last_scale = {
            "src_feat_l2": s_l2, "src_perdim_std": s_sd,
            "tgt_feat_l2": t_l2, "tgt_perdim_std": t_sd,
            "feat_l2_ref_unit_gain": math.sqrt(int(self.feat_dim)),
        }
        self.last_raw_loss = float(loss.detach())
        self.n_forward += 1
        return loss

    def step_stats(self) -> dict:
        """The per-step record the training loop pushes into the diagnostics window.

        ``coral_raw`` is the UNWEIGHTED loss -- gamma is applied by the caller
        (``Trainer.train_step`` multiplies by ``cfg.coral_weight``), so this column stays
        comparable across gamma settings and is what ``--calibrate`` reads.
        """
        return {"coral_raw": self.last_raw_loss,
                "cov_term": float(self.last_cov_term),
                "mean_term": float(self.last_mean_term),
                **self.last_scale}


# ---------------------------------------------------------------------------
# the feature-norm ceiling -- a HARD abort, not a warning
# ---------------------------------------------------------------------------

DEFAULT_FEAT_NORM_CEILING = 110.0
DEFAULT_FEAT_NORM_FLOOR = 40.0


class FeatureNormCeiling(RuntimeError):
    """Raised when the mean post-LayerNorm feature norm leaves the configured band.

    ONE exception type for both directions, because the two failures are the same event --
    the representation has left the scale the run was configured at -- and a caller that
    catches one must catch the other."""


def check_feature_norm_ceiling(step: int, scale: dict, ceiling: float,
                               raw_coral: float,
                               floor: float = DEFAULT_FEAT_NORM_FLOOR) -> None:
    """Abort the run if either domain's mean feature norm leaves ``[floor, ceiling]``.

    THE BAND IS TWO-SIDED, AND THE FLOOR IS THE SIDE THAT MATTERS FOR THIS OBJECTIVE.
    The ceiling catches the fp16 overflow direction. Nothing caught the other one -- and
    the eps^4 degenerate solution drives the norm DOWN, not up: scaling features by eps
    scales the squared-Frobenius CORAL objective by eps^4, so shrinking the representation
    minimises the loss without aligning anything.

    That pressure is measured, monotone in gamma, and still accelerating at the selected
    value: source-norm change over 2000 steps was -1.39% / -1.69% / -2.50% at gamma
    1 / 4 / 16. gamma=16 is the selection (runs/coral_r4_gamma_selection.json, branch C),
    and the confirmatory runs are 8000 steps -- FOUR TIMES the window those numbers were
    measured over. A one-sided guard would have watched the direction the run is not going.

    The floor of 40 is an ENGINEERING CHOICE, not a measurement: LP->FT init sits at ~66,
    so 40 is a ~40% collapse, far outside anything observed (worst measured: -2.5%). It is
    placed to catch a runaway, not to police ordinary drift.

    WHY THIS IS A HARD ABORT AND NOT A LOGGED WARNING. The fp16 sweep in
    ``test_coral_round4.t3`` measured the unguarded half-precision covariance going
    NON-FINITE at ||z|| ~= 126, because z^T z grows as ||z||^2 and saturates fp16's 65504.
    Round 4's own loss is fp32-guarded and does not overflow there, but the norm reaching
    that scale at all means the representation has grown ~2x from its LP->FT start and the
    run is no longer doing what it was configured to do. LP->FT init sits at ~66 and the
    500-step calibration already reached 82 by step 400, so this is a live trajectory, not
    a hypothetical.

    The default ceiling of 110 sits between the measured start (~66), the measured
    400-step value (~82) and the measured fp16 failure point (~126). It is an
    ENGINEERING CHOICE, not a measurement: it is placed below the point where the
    numerics are known to break, with room above the trajectory observed so far.

    An 8000-step run must not silently produce NaNs hours in -- so the step number, BOTH
    domains' norms and the raw CORAL value travel in the exception message, which is
    everything needed to read the failure without re-running anything.
    """
    s = float(scale.get("src_feat_l2", float("nan")))
    t = float(scale.get("tgt_feat_l2", float("nan")))
    finite = [v for v in (s, t) if math.isfinite(v)]

    # ---- FLOOR: the eps^4 collapse direction ----
    if floor and math.isfinite(floor) and finite and min(finite) < float(floor):
        raise FeatureNormCeiling(
            f"FEATURE-NORM FLOOR BREACHED at step {int(step)}: "
            f"src ||f|| = {s:.3f}, tgt ||f|| = {t:.3f}, floor = {float(floor):.1f} "
            f"(raw CORAL = {raw_coral:.6g}; LP->FT init ~66). The representation is "
            f"COLLAPSING, which minimises the squared-Frobenius CORAL objective as eps^4 "
            f"WITHOUT aligning the distributions -- the degenerate solution the "
            f"feature-scale monitor was pre-registered to catch. A falling norm alongside "
            f"a falling raw CORAL is the collapse signature. Aborting. Read the "
            f"feature-scale columns in diagnostics.csv; do not simply lower the floor.")

    if not (ceiling and math.isfinite(ceiling)):
        return
    worst = max(finite) if finite else -1.0
    if worst <= float(ceiling):
        return
    raise FeatureNormCeiling(
        f"FEATURE-NORM CEILING EXCEEDED at step {int(step)}: "
        f"src ||f|| = {s:.3f}, tgt ||f|| = {t:.3f}, ceiling = {float(ceiling):.1f} "
        f"(raw CORAL = {raw_coral:.6g}; unit-gain reference sqrt(640) = 25.298; "
        f"LP->FT init ~66). The representation has grown far enough that the fp16 "
        f"covariance path is measured to go non-finite near ||z|| ~= 126. Aborting rather "
        f"than continuing toward NaN. This is the pre-registered collapse/blow-up guard, "
        f"not a numerical hiccup -- read the feature-scale columns in diagnostics.csv "
        f"before rerunning, and do not simply raise the ceiling.")


def install_fp32_coral(trainer, *, include_mean=True, shrinkage=0.1,
                       target_ema=True, ema_momentum=0.9) -> Fp32CoralLoss:
    """Swap the Trainer's CORAL module for the fp32 round-4 one, in place.

    ``Trainer.train_step`` reaches the loss through ``self.coral``, and
    ``Trainer.da_buffer_state`` / ``Trainer.load`` reach its buffers through
    ``self.coral.named_buffers()`` -- a subclass satisfies both, so nothing in
    ``train_skeleton`` needs to change or even to know.
    """
    coral = Fp32CoralLoss(trainer.head.feat_dim, include_mean=include_mean,
                          shrinkage=shrinkage, target_ema=target_ema,
                          ema_momentum=ema_momentum).to(trainer.device)
    trainer.coral = coral
    return coral


# ===========================================================================
# 2. gradient reachability -- which parameters the CORAL term can actually move
# ===========================================================================

def coral_reachable_params(trainer, src_images: torch.Tensor, tgt_images: torch.Tensor,
                           *, zero_tol: float = 0.0) -> dict:
    """Backprop the CORAL term ALONE and report which parameters it reaches.

    This is the check ``train_skeleton.encode_features``'s open-item comment asks for: it
    was reasoned through for FreeMatch and never for CORAL. Rather than reason about it,
    measure it -- run the real forward the real trainer runs, backward the CORAL term with
    NO supervised term at all, and read the gradients.

    Returns a dict with the reachable parameter COUNT (elements with a non-zero gradient),
    the tensors that received a gradient tensor but an ALL-ZERO one, and the module
    prefixes touched.

    ANALYTICAL EXPECTATION, stated before the measurement so the test can assert it:
      * the LayerNorm **bias** carries NO CORAL signal, at any ``n_blocks``. A bias shifts
        every sample of BOTH domains by the same constant. Covariance is shift-invariant, so
        the covariance term is exactly unchanged; and the first-moment term is a DIFFERENCE
        of means, so the shift cancels there too. The gradient is analytically zero.
      * the **classifier** (``classifier.weight`` / ``classifier.bias``) is not reachable:
        CORAL consumes the adapter tensor, which is upstream of the classifier.
      * ``n_blocks=0`` -> the 640 LayerNorm GAINS.
      * ``n_blocks=3`` -> the 640 gains PLUS ``stages.1``, ``stages.2``, ``stages.3``
        (14,835,520 encoder params, PROJECT.md §7.1a), and NOTHING in ``stages.0`` / ``stem``.

    TWO MEASURED CAVEATS, recorded because the raw element count does NOT land on the
    analytical number and pretending otherwise would be the error:
      1. The bias gradient comes back at ~1e-11 against a gain gradient of ~1e-6 -- a ratio
         of ~3e-6, which is float cancellation around an analytically-zero value, not a
         path. Counting "elements with a bitwise non-zero gradient" therefore counts noise.
         The threshold-free way to settle it is :func:`shift_invariance_check`, which
         perturbs the bias and shows the loss does not move; that is what the test asserts,
         rather than a tuned epsilon.
      2. At ``n_blocks=3`` a few 1e4 encoder elements come back EXACTLY zero on a small
         probe batch -- concentrated in ``stages.3.blocks.*.mlp.fc2.weight``, i.e. GELU
         units that no sample in the probe activated. That is a property of the probe batch,
         not of the graph, so the reachable TENSOR set is the structural claim and the
         element count is reported with its shortfall attributed.
    """
    from train_skeleton import encode_features

    trainer.opt.zero_grad(set_to_none=True)
    trainer.head.train()

    # fp32 and no autocast: this is an identity check on the GRAPH, and a half-precision
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

    da = trainer.coral(src_adapter, tgt_adapter)      # NO supervised term
    da.backward()

    named = list(trainer.head.named_parameters(prefix="head"))
    named += [(f"encoder.{n}", p) for n, p in trainer.encoder.model.named_parameters()]

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
    gain = scale.get("head.trunk.0.weight", {}).get("rms")
    bias = scale.get("head.trunk.0.bias", {}).get("rms")
    enc = {k: v for k, v in scale.items() if k.startswith("encoder.")}
    return {
        "n_blocks": int(trainer.cfg.n_blocks),
        "reachable_param_elements": int(sum(reachable.values())),
        "reachable_tensors": len(reachable),
        "reachable_by_tensor": reachable,
        "zero_grad_tensors": zero_grad,
        "no_grad_tensors": len(none_grad),
        "reachable_prefixes": prefixes,
        "grad_scale_by_tensor": scale,
        "layernorm_bias_over_gain_rms": (None if not (gain and bias) else bias / gain),
        "encoder_elements_with_grad": int(sum(v["numel"] for v in enc.values())),
        "encoder_elements_exactly_zero": int(sum(v["n_exact_zero"] for v in enc.values())),
        "coral_value": float(da.detach()),
    }


def shift_invariance_check(trainer, src_images: torch.Tensor, tgt_images: torch.Tensor,
                           *, delta: float = 1.0, gain_delta: float = 0.10,
                           coral=None) -> dict:
    """Threshold-free proof that the adapter BIAS carries no CORAL signal and the GAIN does.

    Perturbing the LayerNorm bias by a constant shifts every sample of both domains
    identically. Covariance is shift-invariant and the first-moment term is a difference of
    means, so the CORAL loss must not move AT ALL. Perturbing the gain rescales the
    representation, which the eps^4 law says must move it a lot.

    This settles the "is the bias reachable?" question by measuring the FUNCTION rather than
    by choosing an epsilon on a gradient that is analytically zero and numerically ~1e-11.

    ``coral`` defaults to a FRESH ``target_ema=False`` module, which is what makes this a
    test of the CORAL objective rather than of whatever state the trainer's own loss module
    happens to be carrying. Pass a primed module to measure the stale-EMA case instead --
    see :func:`stale_ema_shift_sensitivity`, which is a separate and, it turns out, quite
    different question.
    """
    from train_skeleton import encode_features

    if coral is None:
        coral = Fp32CoralLoss(trainer.head.feat_dim,
                              include_mean=trainer.coral.include_mean,
                              shrinkage=trainer.coral.shrinkage,
                              target_ema=False).to(trainer.device)

    @torch.no_grad()
    def coral_now():
        sf = trainer.standardizer(
            encode_features(trainer.encoder, src_images.to(trainer.device),
                            trainer.cfg.encoder_mode))
        tf = trainer.standardizer(
            encode_features(trainer.encoder, tgt_images.to(trainer.device),
                            trainer.cfg.encoder_mode))
        _, sa = trainer.head.forward_with_features(sf)
        _, ta = trainer.head.forward_with_features(tf)
        was = coral.training
        coral.eval()                            # do not disturb any EMA buffers
        v = float(coral(sa, ta))
        coral.train(was)
        return v

    ln = trainer.head.trunk[0]
    base = coral_now()
    with torch.no_grad():
        ln.bias += delta
    shifted = coral_now()
    with torch.no_grad():
        ln.bias -= delta
        ln.weight *= (1.0 + gain_delta)
    scaled = coral_now()
    with torch.no_grad():
        ln.weight /= (1.0 + gain_delta)
    restored = coral_now()

    return {
        "coral_base": base,
        "coral_after_bias_shift": shifted,
        "bias_shift": float(delta),
        "bias_rel_change": abs(shifted - base) / base if base else float("nan"),
        "coral_after_gain_scale": scaled,
        "gain_scale": 1.0 + float(gain_delta),
        "gain_rel_change": abs(scaled - base) / base if base else float("nan"),
        "coral_restored": restored,
        "restore_rel_change": abs(restored - base) / base if base else float("nan"),
    }


def stale_ema_shift_sensitivity(trainer, src_images, tgt_images, *, delta=1.0) -> dict:
    """MEASURE, do not assume: how much shift-invariance does ``target_ema=True`` cost?

    THE FINDING THIS EXISTS TO RECORD. Deep CORAL's objective is shift-invariant -- that is
    the property that makes the LayerNorm bias unreachable. But ``CoralLoss`` with
    ``target_ema=True`` does not evaluate the objective; it evaluates

        ||Sigma_s(now) - [m*Sigma_t(BUFFERED) + (1-m)*Sigma_t(now)]||_F^2

    and the buffered half was computed with an EARLIER model. Shift the representation and
    the source mean moves by the full delta while the target reference moves by only
    ``(1-m)*delta``, so the first-moment term acquires a ``(m*delta)^2`` penalty that pure
    CORAL does not have. The loss stops being shift-invariant, and the adapter bias -- which
    is analytically unreachable for CORAL -- becomes a live knob whose gradient points at
    "undo the offset relative to a stale snapshot".

    At ``n_blocks=0`` this barely matters: the frozen encoder means the buffer goes stale
    slowly and only 640 gains + 640 biases can respond. At ``n_blocks=3`` the encoder moves
    every step, so the buffer is stale by construction, and 14.8M parameters can chase it.
    That is a property of the IMPLEMENTATION, not of Deep CORAL, and it is the reason
    ``--coral-no-target-ema`` exists on the round-4 runner.

    Returns the measured relative change so the effect is a number in the record rather
    than an argument in a comment. UNVERIFIED whether it helps or hurts the metric.
    """
    from train_skeleton import encode_features

    coral = Fp32CoralLoss(trainer.head.feat_dim,
                          include_mean=trainer.coral.include_mean,
                          shrinkage=trainer.coral.shrinkage,
                          target_ema=True,
                          ema_momentum=trainer.coral.ema_momentum).to(trainer.device)

    def adapters():
        sf = trainer.standardizer(
            encode_features(trainer.encoder, src_images.to(trainer.device),
                            trainer.cfg.encoder_mode))
        tf = trainer.standardizer(
            encode_features(trainer.encoder, tgt_images.to(trainer.device),
                            trainer.cfg.encoder_mode))
        _, sa = trainer.head.forward_with_features(sf)
        _, ta = trainer.head.forward_with_features(tf)
        return sa, ta

    # PRIMING MUST RUN WITH GRAD ENABLED. CoralLoss only writes its EMA buffers when
    # ``target_ema and self.training and torch.is_grad_enabled()`` -- i.e. during real
    # training steps, never during eval scoring. That is correct behaviour, and it is why
    # this probe primes outside no_grad; priming under no_grad leaves ema_ready at 0 and
    # silently measures the NON-EMA path instead.
    coral.train()
    sa, ta = adapters()
    _ = coral(sa, ta)                                   # writes ema_mean / ema_cov
    assert float(coral.ema_ready) >= 1.0, "EMA priming failed -- buffers were not written"

    coral.eval()                                        # freeze the buffers; measure only
    with torch.no_grad():
        sa, ta = adapters()
        base = float(coral(sa, ta))
        ln = trainer.head.trunk[0]
        ln.bias += delta
        sa2, ta2 = adapters()                           # buffer is now STALE by `delta`
        shifted = float(coral(sa2, ta2))
        ln.bias -= delta
    return {"ema_momentum": float(coral.ema_momentum),
            "ema_primed": bool(float(coral.ema_ready) >= 1.0),
            "coral_base": base, "coral_after_bias_shift": shifted,
            "bias_shift": float(delta),
            "rel_change": abs(shifted - base) / base if base else float("nan"),
            "note": ("target_ema makes the loss NON-shift-invariant against a stale "
                     "reference; pure Deep CORAL is shift-invariant. See the docstring.")}


# ===========================================================================
# 3. the test-label lockout for --calibrate
# ===========================================================================

class TestLabelAccess(RuntimeError):
    """Raised when the calibration path tries to touch a metric or a test split."""


_FORBIDDEN_SPLIT_PREFIXES = ("test_",)


@contextlib.contextmanager
def test_label_lockout(verbose: bool = True):
    """Make TPR / AUROC / AUPRC / the metric suite / the curve hook / any ``test_*`` cache
    split UNREACHABLE for the duration of the block.

    THE RULE THIS ENFORCES. ``--calibrate`` chooses gamma from LOSS MAGNITUDES ONLY.
    gamma must never be adjusted after seeing a metric -- that would make it an oracle knob
    tuned on fold-0 test labels, exactly the contamination PROJECT.md §7.8/§7.14 exist to
    prevent, and it would silently invalidate every confirmatory fold downstream.

    A comment saying "calibration does not read metrics" is not enforcement; this is. The
    patch is applied to the FUNCTION OBJECTS wherever they are bound inside this project's
    own modules, not just on ``metrics``, because ``from metrics import tpr_at_fpr`` creates
    an independent binding that patching the module attribute would miss.

    ``source_train`` and ``val`` remain loadable: the standardizer is fitted from the cached
    SOURCE features, which carry no target-domain and no test information.
    """
    import evaluate as ev
    import metrics as me

    targets = {id(me.tpr_at_fpr), id(me.auroc), id(me.auprc), id(me.metric_suite),
               id(me.purity_at_completeness), id(me.completeness_at_purity)}

    def _boom(*a, **k):
        raise TestLabelAccess(
            "the calibration path attempted to compute a TEST-SET METRIC. gamma is "
            "calibrated from loss magnitudes only and may never be adjusted after seeing "
            "a metric. This is a hard rule, not a guideline -- see test_label_lockout().")

    patched = []
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f or os.path.dirname(os.path.abspath(f)) != _HERE:
            continue                                  # this project's modules only
        for attr, val in list(vars(mod).items()):
            if callable(val) and id(val) in targets:
                patched.append((mod, attr, val))
                setattr(mod, attr, _boom)

    # the curve hook is the only other way a metric could be reached
    import curve_hook as ch
    for attr in ("append_curve_row", "eval_checkpoint_folds"):
        if hasattr(ch, attr):
            patched.append((ch, attr, getattr(ch, attr)))
            setattr(ch, attr, _boom)

    orig_load = ev.FeatureCache.load

    def guarded_load(self, name, *a, **k):
        if str(name).startswith(_FORBIDDEN_SPLIT_PREFIXES):
            raise TestLabelAccess(
                f"the calibration path attempted to load the cache split {name!r}. "
                f"Calibration may read source_train (for the standardizer) and nothing "
                f"that carries a test label.")
        return orig_load(self, name, *a, **k)

    ev.FeatureCache.load = guarded_load
    patched.append((ev.FeatureCache, "load", orig_load))

    if verbose:
        print(f"[lockout] test-label lockout ACTIVE: {len(patched)} binding(s) sealed "
              f"(TPR/AUROC/AUPRC/metric_suite/purity/curve_hook + test_* cache splits).")
    try:
        yield
    finally:
        for holder, attr, val in patched:
            setattr(holder, attr, val)
        if verbose:
            print("[lockout] released.")


# ===========================================================================
# 4. per-step diagnostics window + the round-4 curve instrument
# ===========================================================================

# Every one of these is produced by the round-4 training loop from quantities that already
# exist: Trainer.train_step's return dict plus Fp32CoralLoss.step_stats(). Nothing is
# recomputed and nothing extra is forwarded through the encoder.
CORAL_STEP_KEYS = (
    "sup_loss",             # supervised BCE on the SOURCE weak view
    "da_loss",              # gamma * CORAL (what actually entered the total loss)
    "coral_raw",            # the UNWEIGHTED CORAL loss -- gamma-independent, calibratable
    "cov_term",             # ||Sigma_s - Sigma_t||_F^2 / d^2
    "mean_term",            # ||mu_s - mu_t||^2 / d
    # ---- the pre-registered feature-scale monitor, per domain ----
    "src_feat_l2", "src_perdim_std",
    "tgt_feat_l2", "tgt_perdim_std",
    # ---- fp16 health (§7.10: lost updates and "unfreezing does not help" look identical) --
    "step_skipped",
    "amp_scale",
)

CORAL_DIAG_COLUMNS = (
    ["step", "n_steps_in_window"]
    + [f"{k}_mean" for k in CORAL_STEP_KEYS]
    # amp_scale is a stepwise power-of-two ladder; a window MEAN of it is a value the ladder
    # never takes. The _last column is the instantaneous one -- read that (§7.10's note).
    + [f"{k}_last" for k in ("coral_raw", "src_feat_l2", "tgt_feat_l2", "amp_scale")]
    + ["coral_over_sup",            # the Sun & Saenko ratio, live
       "src_l2_over_unit_gain"]     # feature norm / sqrt(640) -- 1.0 == no rescaling
)


class CoralStepWindow:
    """Accumulates per-step CORAL statistics between two curve points.

    Window MEANS rather than a single sampled step, for the same reason §5.3 gives for
    FreeMatch: per-batch covariance on 256 draws from a 782-image target pool is noisy, and
    the question the monitor asks ("is the representation shrinking?") is about the trend.
    ``_last`` columns are kept for the quantities whose instantaneous value is the
    meaningful one.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._acc = {k: [] for k in CORAL_STEP_KEYS}
        self._last = {k: float("nan") for k in CORAL_STEP_KEYS}
        self.n = 0

    def update(self, stats: dict):
        for k in CORAL_STEP_KEYS:
            v = stats.get(k)
            if v is None:
                continue
            v = float(v)
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
        for k in CORAL_STEP_KEYS:
            row[f"{k}_mean"] = _num(self.mean(k))
        for k in ("coral_raw", "src_feat_l2", "tgt_feat_l2", "amp_scale"):
            row[f"{k}_last"] = _num(self.last(k))
        sup, cor = self.mean("sup_loss"), self.mean("coral_raw")
        row["coral_over_sup"] = _num(cor / sup) if (math.isfinite(sup) and sup > 0) else None
        l2 = self.mean("src_feat_l2")
        row["src_l2_over_unit_gain"] = _num(l2 / math.sqrt(640.0)) if math.isfinite(l2) else None
        return row


class CoralCurveInstrument(CurveInstrument):
    """:class:`unfreeze_instrument.CurveInstrument` with CORAL's diagnostics.

    Everything method-AGNOSTIC is inherited untouched and unmodified: the lazy live-encoder
    factory and ``split_loader`` (§7.7's Windows-spawn ordering fix), the derived-logit dump
    (§5.2), the memorization probe (§5.4), the fp16 skip accounting, and ``summary()``.
    Only the per-step diagnostics differ, because FreeMatch's ``tau_global`` / ``pbar`` /
    ``acceptance`` columns do not exist for a method with no pseudo-label loop -- and
    CORAL's covariance and feature-scale columns do not exist for FreeMatch.
    """

    def __init__(self, out_dir, fold, **kw):
        super().__init__(out_dir, fold, **kw)
        self.window = CoralStepWindow()

    def on_curve_point(self, step: int, scores: dict) -> dict:
        from unfreeze_instrument import memorization_row, MEMO_COLUMNS, dump_logits

        diag = self.window.row(step)
        append_row(self.diag_csv, CORAL_DIAG_COLUMNS, diag)
        skip = diag.get("step_skipped_mean")
        if skip is not None and skip > 0:
            self.max_window_skip_rate = max(self.max_window_skip_rate, float(skip))
            if skip >= self.SKIP_WARN_RATE and self.verbose:
                print(f"      [WARN step {step}] fp16 GradScaler DISCARDED "
                      f"{skip*100:.1f}% of this window's optimizer steps. Those steps "
                      f"changed nothing -- read the final rate with that in mind.",
                      flush=True)
        self.window.reset()

        memo = memorization_row(step, scores, self.fold)
        append_row(self.memo_csv, MEMO_COLUMNS, memo)

        if self.dump_logits:
            dump_logits(self.logit_dir, step, scores, self.fold,
                        write_ids=not self._wrote_ids)
            self._wrote_ids = True
            self.n_logit_dumps += 1
        self.n_curve_points += 1
        self._last_diag = diag
        return memo

    def last_diag(self) -> dict:
        return getattr(self, "_last_diag", {}) or {}


def _num(x):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else x


# ===========================================================================
# 5. gamma calibration -- Sun & Saenko's rule, from loss magnitudes ONLY
# ===========================================================================

# ===========================================================================
# 6. the gamma sweep -- PRE-REGISTERED selection criterion + its evaluation
# ===========================================================================

SWEEP_GAMMAS = (1.0, 4.0, 16.0)
SWEEP_STEPS = 2000
SWEEP_BLOCK = 50          # moving-block bootstrap block length, in steps
SWEEP_B = 5000            # bootstrap draws (matches §7.12)
SWEEP_ALPHA = 0.05

SWEEP_CRITERION = """
PRE-REGISTERED SELECTION CRITERION -- gamma sweep, nb3, fold 0, 2000 steps per arm
==================================================================================
Fixed in code and printed before any run. NOT to be amended afterwards.

WHY A SWEEP AT ALL. The 500-step calibration showed magnitude-matching is ILL-POSED on the
unfrozen arm. Raw CORAL depends on the feature scale as scale^4, the unfrozen encoder moves
that scale, so "equalise the two losses at the end of training" has no fixed point: nb3's
equalising gamma halved across its own tail window (1.138 -> 0.629, CV 0.868). And at
gamma=1.0 the CORAL term is LOSING -- sup_loss falls, raw CORAL rises, the feature norm
rises 25%. Running 8000 steps in that state would make a null result uninterpretable,
because "CORAL did not help" and "CORAL never got any traction" look identical in the
metric.

CHOOSE THE SMALLEST gamma satisfying BOTH:

  (i)  RAW CORAL IS NOT MONOTONICALLY RISING over the SECOND HALF of the window
       (steps in (1000, 2000]).
       Fit an ordinary-least-squares slope of raw CORAL against step over that half.
       Resample with a MOVING-BLOCK BOOTSTRAP (block length 50 steps, B=5000) -- blocks,
       not independent steps, because a training trace is autocorrelated and an iid
       bootstrap would understate the slope's uncertainty by a large factor.
       PASS iff the 95% percentile CI of the slope has LOWER BOUND <= 0.
       i.e. FAIL only when the CI lies ENTIRELY ABOVE zero -- demonstrably rising.
       This is the direct operationalisation of "not monotonically rising": the test is
       against the claim that it IS rising, and a slope indistinguishable from zero passes.
       The stricter reading (CI entirely BELOW zero = demonstrably falling) is computed and
       reported alongside as `strictly_falling`, but it is NOT the criterion.

  (ii) FEATURE-NORM GROWTH over the FULL window is MEASURABLY DAMPED relative to the
       gamma=1.0 arm.
       growth = mean ||f|| over the LAST 10% of steps  minus  mean ||f|| over the FIRST 10%,
       computed on the SOURCE domain (the target domain is reported and must agree in sign).
       Compare growth(gamma) against growth(1.0) with the same moving-block bootstrap,
       resampling each arm independently (they are independent runs).
       PASS iff the 95% CI of [growth(gamma) - growth(1.0)] lies ENTIRELY BELOW zero.
       "Measurably" is the CI excluding zero; "damped" is the sign being negative.
       VACUOUS-REFERENCE GUARD: if growth(1.0) <= 0 the reference arm did not grow, there
       is nothing to damp, and criterion (ii) is reported INAPPLICABLE rather than passed.
       gamma=1.0 is the reference and is never a candidate under (ii).

IF NO GAMMA SATISFIES BOTH: report that, pick nothing, and propose
EXTENDING THE SWEEP UPWARD (gamma 64, then 256). Do not relax either clause.

WINDOWING UNDER AN ABORT. If an arm trips the feature-norm ceiling it stops early. All
comparisons are then made on the LONGEST WINDOW COMMON TO BOTH ARMS, and the truncation is
reported next to the verdict. An arm that aborts has not passed anything.

METRIC HYGIENE. This sweep reads LOSS DYNAMICS and FEATURE STATISTICS ONLY. No TPR, no
AUROC, no test-set quantity is computed, loaded, or reachable -- enforced by
coral_round4.test_label_lockout(), not by convention.
"""


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    den = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - ym)).sum() / den) if den > 0 else float("nan")


def _block_indices(n: int, block: int, rng) -> np.ndarray:
    """One moving-block bootstrap resample of positions 0..n-1."""
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


def slope_with_block_ci(steps, values, *, block=SWEEP_BLOCK, B=SWEEP_B,
                        alpha=SWEEP_ALPHA, seed=0) -> dict:
    """OLS slope of ``values`` against ``steps`` with a moving-block bootstrap CI."""
    x = np.asarray(steps, float); y = np.asarray(values, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        raise ValueError(f"need >=4 finite points to fit a slope, got {x.size}")
    point = _ols_slope(x, y)
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        i = _block_indices(x.size, block, rng)
        reps[b] = _ols_slope(x[i], y[i])
    lo, hi = np.percentile(reps[np.isfinite(reps)],
                           [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"slope": point, "ci": [float(lo), float(hi)],
            "n_points": int(x.size), "block": int(block), "B": int(B),
            "not_rising": bool(lo <= 0.0),          # THE criterion (i) test
            "strictly_falling": bool(hi < 0.0),     # reported, not the criterion
            "strictly_rising": bool(lo > 0.0)}


def norm_growth_with_ci(steps, values, *, frac=0.10, block=SWEEP_BLOCK, B=SWEEP_B,
                        alpha=SWEEP_ALPHA, seed=0) -> dict:
    """growth = mean(last ``frac`` of the window) - mean(first ``frac``), with a
    block-bootstrap distribution of the same statistic (returned so two arms can be
    differenced).

    THE HEAD AND TAIL WINDOWS ARE RESAMPLED SEPARATELY, and this is not a detail.
    ``growth`` is a statement about the ORDER of the series -- where the trace started
    against where it ended. Resampling blocks across the whole window (the obvious first
    implementation, and the one this replaced) shuffles late blocks into the head and early
    blocks into the tail, destroying exactly the structure the statistic measures. The
    resulting bootstrap distribution is enormous and the criterion can never fire: measured
    on a clean linear ramp from 66 to 70 against a reference ramp 66 to 90, it returned a
    difference of -18.02 with a CI of [-12, +13] -- an interval that does not even contain
    its own point estimate, which is the signature of a statistic and a resampling scheme
    that disagree about what is being estimated.

    The uncertainty that actually matters here is in ESTIMATING EACH WINDOW'S MEAN, so each
    window is block-resampled within itself (blocks, still, because a training trace is
    autocorrelated inside a window too) and the difference is taken per draw.
    """
    y = np.asarray(values, float)
    y = y[np.isfinite(y)]
    n = y.size
    k = max(2, int(round(frac * n)))
    if n < 2 * k:
        raise ValueError(f"window too short for a {frac:.0%} head/tail split: n={n}")
    head, tail = y[:k], y[-k:]
    point = float(tail.mean() - head.mean())
    blk = max(1, min(int(block), k))
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        h = head[_block_indices(k, blk, rng)]
        t = tail[_block_indices(k, blk, rng)]
        reps[b] = t.mean() - h.mean()
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"growth": point, "ci": [float(lo), float(hi)], "n_points": int(n),
            "head_tail_n": int(k), "block": int(blk), "draws": reps}


def damping_verdict(arm, ref, *, alpha=SWEEP_ALPHA) -> dict:
    """Criterion (ii): is arm's feature-norm growth measurably BELOW the gamma=1.0 arm's?"""
    if ref["growth"] <= 0.0:
        return {"applicable": False, "passed": False,
                "reason": ("the gamma=1.0 reference arm did not grow "
                           f"(growth {ref['growth']:+.4f}); there is nothing to damp, so "
                           "criterion (ii) is INAPPLICABLE, not passed"),
                "diff": float(arm["growth"] - ref["growth"])}
    d = arm["draws"] - ref["draws"]
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"applicable": True,
            "diff": float(arm["growth"] - ref["growth"]),
            "ci": [float(lo), float(hi)],
            "passed": bool(hi < 0.0),
            "arm_growth": arm["growth"], "ref_growth": ref["growth"]}


def equalising_gamma(sup: np.ndarray, coral_raw: np.ndarray, *, tail: int = 100) -> dict:
    """gamma such that ``gamma * CORAL`` matches the classification loss in magnitude.

    Sun & Saenko (arXiv:1607.01719 §4) set lambda so the two losses are of roughly the same
    magnitude at the END of training, so the point estimate is taken over the TRAILING
    ``tail`` steps, not the whole window.

    Stability is reported rather than assumed. A single ratio-of-means is a number whatever
    the data does; what says whether it can be trusted is (a) the spread of the per-step
    ratio and (b) whether the first and second halves of the tail agree. Both are returned,
    and neither is a metric -- they are functions of two loss series and nothing else.
    """
    sup = np.asarray(sup, dtype=float)
    cor = np.asarray(coral_raw, dtype=float)
    if sup.size != cor.size:
        raise ValueError(f"series lengths differ: {sup.size} vs {cor.size}")
    n = min(int(tail), sup.size)
    if n < 2:
        raise ValueError(f"need >=2 steps in the tail window, got {n}")
    s, c = sup[-n:], cor[-n:]
    ok = np.isfinite(s) & np.isfinite(c) & (c > 0)
    if not ok.any():
        raise ValueError("no finite, positive CORAL values in the tail window")
    s, c = s[ok], c[ok]

    mean_s, mean_c = float(s.mean()), float(c.mean())
    gamma = mean_s / mean_c
    per_step = s / c                                   # per-step equalising gamma
    h = len(s) // 2
    g1 = float(s[:h].mean() / c[:h].mean()) if h >= 1 else float("nan")
    g2 = float(s[h:].mean() / c[h:].mean()) if len(s) - h >= 1 else float("nan")
    return {
        "gamma_equalising": gamma,
        "tail_steps": int(len(s)),
        "sup_loss_tail_mean": mean_s,
        "coral_raw_tail_mean": mean_c,
        "per_step_gamma_median": float(np.median(per_step)),
        "per_step_gamma_sd": float(np.std(per_step, ddof=1)),
        "per_step_gamma_cv": float(np.std(per_step, ddof=1) / np.mean(per_step)),
        "per_step_gamma_p10": float(np.percentile(per_step, 10)),
        "per_step_gamma_p90": float(np.percentile(per_step, 90)),
        "gamma_first_half": g1,
        "gamma_second_half": g2,
        "half_ratio": float(g2 / g1) if g1 else float("nan"),
        "basis": ("loss magnitudes only -- no TPR, no AUROC, no test split was reachable "
                  "while these series were produced (see test_label_lockout)"),
    }
