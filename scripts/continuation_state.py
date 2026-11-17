"""
continuation_state.py
=====================
RECONSTRUCT the training state that a weights-only checkpoint did not save, so a
continuation run resumes the objective it left off under instead of restarting three
separate transients.

THE PROBLEM THIS SOLVES, AND THE EVIDENCE THAT FORCED IT
--------------------------------------------------------
Round 1's FreeMatch checkpoints store exactly four keys -- ``head_type``, ``head``,
``standardizer``, ``cfg``. The MODEL survives; the training state AROUND the model does
not. Continuing from such a checkpoint therefore restarts three things at once:

  1. the FreeMatch self-adaptive threshold (``tau_global``, ``pbar``), re-initialised to
     1/C and re-converging from scratch;
  2. the consistency weight ``mu_eff``, which the cosine warmup holds at EXACTLY 0 for the
     first 10% of the schedule -- so the run trains on pure source BCE, with no target term
     in the loss at all, for 800 steps;
  3. AdamW's first and second moments, which start at zero and take thousands of steps to
     become meaningful at beta2=0.999.

MEASURED consequence on the fold-0 frozen control (2000 steps): TPR@1e-3 went
0.3878 -> 0.2347 (step 500) -> 0.3367 (step 2000). Spread 0.102, max drift 0.1531 from
step 0, against a one-positive resolution of 1/98 = 0.0102 -- ten to fifteen times the
resolvable difference, and V-shaped, bottoming inside the mu=0 window.

That excursion is NOT the model continuing to train. Measured over an equal number of
steps, round 1's own last 2000 steps moved the head 0.0416 of its norm for a TPR range of
<= 3 positives; the control's first 2000 moved it 0.0469 -- 1.13x as far -- for 15
positives. Near-identical displacement, five times the metric excursion: round 1 was
diffusing along a flat manifold, the continuation stepped off it.

RECONSTRUCTION, NOT RESTORATION
-------------------------------
Round 1's values were never written to disk, so nothing here restores them. Each quantity
is REBUILT to the fixed point it would reach for THIS model on THIS data:

  * ``tau_global`` / ``pbar``: the SAT update is an EMA fed by forward-pass outputs, so with
    the weights held fixed its fixed point is exactly the pool mean. Averaging once lands on
    the value the EMA would drift to over thousands of steps.
  * AdamW's second moment: an EMA of squared GRADIENTS, so it cannot be computed from
    forward passes. It is warmed by running real steps and updating only the moments.

[UNVERIFIED] whether either matches round 1's exact step-8000 state -- round 1's threshold
was itself mid-trajectory, chasing a model that was still moving. The claim made here is
only that each reaches the fixed point for the current model on the current data. Written
as ``reconstructed`` everywhere, never ``restored``.

WHAT IS NOT REBUILT. ``fairness_target`` is a fixed configuration value (the balanced
[0.5, 0.5] marginal SAF pushes toward, PROJECT.md 4.3), not an EMA, and is left alone.
Enumerated from the module rather than assumed: ``FreeMatchLoss`` registers exactly three
buffers -- ``tau_global``, ``pbar``, ``fairness_target`` -- in both ``local_threshold_mode``
variants, and the SAF term keeps no running histogram (it computes its marginal from the
current batch each step). The ``last_*`` attributes are logging floats, not state.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from da_losses import binary_prob2, freematch_sat_update
from train_skeleton import encode_features

# rebuilt buffers, and the one deliberately left alone
SAT_BUFFERS = ("tau_global", "pbar")
NOT_REBUILT = ("fairness_target",)

DEFAULT_PASSES = 40          # passes over the target pool for the direct average
DEFAULT_EMA_ITERS = 5000     # cross-check length; 0.999**5000 = 0.0067 init weight left
DEFAULT_WARM_STEPS = 200     # Adam moment warm-up
DEFAULT_PROBE_STEPS = (50, 200, 500)


# ---------------------------------------------------------------------------
# 1. SAT buffer reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_target_batch_stats(trainer, target_loader, n_batches, verbose=True):
    """Per-batch (mean max-prob, mean per-class prob) on the WEAK target view.

    Batches come from the SAME loader the training loop uses -- class-balanced on latent
    identity, weak augmentation -- because the EMA's fixed point is an expectation over the
    batch distribution the training loop actually feeds it, not over a plain pass through
    the pool. With a ~1:1 pool the two nearly coincide, but "faithful to what the EMA sees"
    is the property being reconstructed.

    Runs under the run's own autocast, so an fp16 run rebuilds from fp16 forwards.
    """
    trainer.head.eval()
    out_max, out_prob = [], []
    it = iter(target_loader)
    for i in range(int(n_batches)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(target_loader)
            batch = next(it)
        imgs = batch["image"].to(trainer.device, non_blocking=True)   # weak view
        with trainer._autocast():
            feat = trainer.standardizer(
                encode_features(trainer.encoder, imgs, trainer.cfg.encoder_mode))
            logits = trainer.head(feat)
        prob = binary_prob2(logits.float())                # (B,2)
        out_max.append(float(prob.max(dim=1).values.mean()))
        out_prob.append(prob.mean(0).detach().cpu().numpy())
        if verbose and (i + 1) % 25 == 0:
            print(f"    [sat] {i+1}/{n_batches} batches", flush=True)
    trainer.head.train()
    return np.asarray(out_max, dtype=np.float64), np.asarray(out_prob, dtype=np.float64)


def ema_crosscheck(batch_max, batch_prob, momentum, n_iter=DEFAULT_EMA_ITERS, seed=0):
    """Drive the REAL SAT update rule for ``n_iter`` iterations and see where it lands.

    The point is to confirm the implemented update matches arXiv:2205.07246's rule, i.e.
    that its fixed point really is the pool mean. It is driven by RESAMPLING the collected
    per-batch statistics rather than by 5000 fresh forward passes (which would cost 1.28M
    encoder forwards): the EMA's fixed point depends only on the distribution of per-batch
    statistics, so resampling from the empirical distribution preserves it exactly.

    A KNOWN residual remains by construction: after n_iter updates the 1/C initialisation
    still carries weight ``momentum**n_iter`` (0.0067 at 0.999^5000), so the EMA lands
    slightly short of the direct mean. That offset is predicted and reported, and the
    comparison that matters is the one CORRECTED for it -- otherwise a real mismatch could
    hide behind a known bias.
    """
    rng = np.random.default_rng(seed)
    tau = torch.tensor(0.5, dtype=torch.float64)
    pbar = torch.full((batch_prob.shape[1],), 1.0 / batch_prob.shape[1],
                      dtype=torch.float64)
    idx = rng.integers(0, len(batch_max), size=int(n_iter))
    for j in idx:
        # the real rule, one batch at a time: tau <- m*tau + (1-m)*mean(max_prob)
        tau = momentum * tau + (1.0 - momentum) * torch.tensor(batch_max[j],
                                                               dtype=torch.float64)
        pbar = momentum * pbar + (1.0 - momentum) * torch.tensor(batch_prob[j],
                                                                 dtype=torch.float64)
    return float(tau), pbar.numpy()


def rebuild_sat_buffers(trainer, target_loader, *, n_passes=DEFAULT_PASSES,
                        pool_size=None, batch_size=None, ema_iters=DEFAULT_EMA_ITERS,
                        seed=0, verbose=True):
    """Reconstruct ``tau_global`` and ``pbar`` in place; return a report dict.

    Returns both the direct-average estimate (which is assigned) and the EMA cross-check,
    plus their agreement. The caller is expected to ABORT on a disagreement larger than a
    few 1e-3 once the known init-bleed offset is removed: that would mean the buffer update
    path does not implement the paper's rule and the whole reconstruction is unsound.
    """
    fm = trainer.freematch
    m = float(fm.sat_ema_momentum)
    pool = int(pool_size or 0)
    bs = int(batch_size or trainer.cfg.batch_size)
    n_batches = max(int(n_passes) * max(int(math.ceil(pool / bs)), 1), 1) if pool \
        else int(n_passes)

    if verbose:
        print(f"  [sat] reconstructing tau_global/pbar: {n_passes} passes over a "
              f"{pool}-image pool at batch {bs} -> {n_batches} batches "
              f"(weak aug, {'fp16' if trainer._amp_enabled else 'fp32'})", flush=True)

    bmax, bprob = collect_target_batch_stats(trainer, target_loader, n_batches,
                                             verbose=verbose)
    tau_direct = float(bmax.mean())
    pbar_direct = bprob.mean(0)

    tau_ema, pbar_ema = ema_crosscheck(bmax, bprob, m, n_iter=ema_iters, seed=seed)

    # the EMA still carries this much of the 1/C init after ema_iters updates
    bleed = m ** int(ema_iters)
    tau_ema_pred = tau_direct + bleed * (0.5 - tau_direct)
    pbar_ema_pred = pbar_direct + bleed * (1.0 / len(pbar_direct) - pbar_direct)

    report = {
        "method": "reconstruction (NOT restoration -- round 1's values were never saved)",
        "n_passes": int(n_passes), "n_batches": int(n_batches), "batch_size": bs,
        "pool_size": pool, "sat_ema_momentum": m, "ema_iters": int(ema_iters),
        "tau_direct": tau_direct, "pbar_direct": [float(x) for x in pbar_direct],
        "tau_ema": tau_ema, "pbar_ema": [float(x) for x in pbar_ema],
        "init_bleed_weight": bleed,
        "tau_ema_predicted": tau_ema_pred,
        "tau_abs_diff_raw": abs(tau_ema - tau_direct),
        "tau_abs_diff_bleed_corrected": abs(tau_ema - tau_ema_pred),
        "pbar_abs_diff_bleed_corrected": float(np.max(np.abs(
            np.asarray(pbar_ema) - pbar_ema_pred))),
        "batch_max_sd": float(bmax.std()),
        "assigned": "direct average (the EMA fixed point with weights held fixed)",
    }

    with torch.no_grad():
        fm.tau_global.fill_(tau_direct)
        fm.pbar.copy_(torch.as_tensor(pbar_direct, dtype=fm.pbar.dtype,
                                      device=fm.pbar.device))
        fm.last_tau_global = float(fm.tau_global)
    report["buffers_rebuilt"] = list(SAT_BUFFERS)
    report["buffers_left_alone"] = list(NOT_REBUILT)
    report["fairness_target_after"] = [float(x) for x in fm.fairness_target]

    if verbose:
        print(f"  [sat] tau_global : direct {tau_direct:.6f} | EMA({ema_iters}) "
              f"{tau_ema:.6f} | predicted-with-bleed {tau_ema_pred:.6f}")
        print(f"  [sat]   agreement: raw |diff| {report['tau_abs_diff_raw']:.6f}  "
              f"bleed-corrected |diff| {report['tau_abs_diff_bleed_corrected']:.2e}")
        print(f"  [sat] pbar       : direct {np.round(pbar_direct, 6).tolist()} | "
              f"EMA {np.round(pbar_ema, 6).tolist()}  "
              f"(bleed-corrected max|diff| {report['pbar_abs_diff_bleed_corrected']:.2e})")
        print(f"  [sat] fairness_target LEFT ALONE at {report['fairness_target_after']} "
              f"(a config value, not an EMA)")
    return report


def sat_check_or_raise(report, tol=5e-3):
    """ABORT unless the EMA cross-check agrees with the direct average.

    Disagreement beyond a few 1e-3, after removing the predicted init bleed, means the
    implemented buffer update is not the paper's EMA -- in which case the direct average is
    not its fixed point and the whole reconstruction is unsound. Better to stop than to
    train 2000 steps from a fabricated threshold.
    """
    bad = []
    if report["tau_abs_diff_bleed_corrected"] > tol:
        bad.append(f"tau {report['tau_abs_diff_bleed_corrected']:.4e}")
    if report["pbar_abs_diff_bleed_corrected"] > tol:
        bad.append(f"pbar {report['pbar_abs_diff_bleed_corrected']:.4e}")
    if bad:
        raise SystemExit(
            f"[continuation] ABORT: the SAT EMA cross-check disagrees with the direct "
            f"average beyond {tol:g} after correcting for the known init bleed: {bad}.\n"
            f"  The buffer update path does not match arXiv:2205.07246's rule, so the pool "
            f"mean is not its fixed point and the reconstruction would be fabricated.")
    return True


# ---------------------------------------------------------------------------
# 2. AdamW moment warm-up
# ---------------------------------------------------------------------------

def second_moment_norm(opt):
    """L2 norm of the concatenated exp_avg_sq across every param with state."""
    tot, n = 0.0, 0
    for g in opt.param_groups:
        for p in g["params"]:
            st = opt.state.get(p)
            if st and "exp_avg_sq" in st:
                tot += float((st["exp_avg_sq"].double() ** 2).sum())
                n += st["exp_avg_sq"].numel()
    return math.sqrt(tot), n


def optimizer_step_counter(opt):
    """Max Adam ``step`` across params -- the counter bias correction reads."""
    steps = []
    for g in opt.param_groups:
        for p in g["params"]:
            st = opt.state.get(p)
            if st and "step" in st:
                s = st["step"]
                steps.append(float(s) if not torch.is_tensor(s) else float(s.item()))
    return max(steps) if steps else 0.0


def warm_adam_moments(trainer, paired_loader, *, n_steps=DEFAULT_WARM_STEPS,
                      total_steps, mu_step_offset, probe_steps=DEFAULT_PROBE_STEPS,
                      verbose=True):
    """Accumulate AdamW's moments WITHOUT moving the weights; return a report dict.

    HOW THE WEIGHTS ARE HELD FIXED. Each iteration snapshots every optimizer-owned
    parameter, runs a real ``train_step`` (forward, backward, ``scaler.step``,
    ``scaler.update``), then copies the snapshot back. The optimizer's state -- ``exp_avg``,
    ``exp_avg_sq`` and the per-param ``step`` counter -- is updated exactly as Adam would
    have updated it, including AdamW's decoupled weight decay (which is applied to the
    parameters and therefore also undone by the restore).

    THE ENCODER IS DELIBERATELY LEFT COLD. Its ``requires_grad`` is switched off for the
    duration, so at ``n_blocks>0`` no encoder gradients are produced and AdamW never creates
    state for those parameters. The asymmetry is correct and intended: the HEAD's optimizer
    state existed in round 1 and was lost, so it is reconstructed; the ENCODER's never
    existed, because round 1 never trained the encoder. Warming it would be inventing state
    for a phase of training that did not happen. It also makes this phase bit-identical
    between the frozen and unfrozen arms.

    THE OBJECTIVE IS THE CORRECTED ONE. ``mu_step_offset`` is passed straight through as
    the step, so ``mu_eff`` is already at full value -- the moments are accumulated under
    the loss the run will actually train on, not under the warmup's pure-BCE surrogate.

    THE STEP COUNTER MUST CARRY THROUGH into training, or bias correction re-applies and
    reintroduces the transient this exists to remove. It does, because the same optimizer
    object continues into the training loop; the report records the counter before and
    after so the property is verified rather than assumed.

    ``probe_steps`` beyond ``n_steps`` are measured and then ROLLED BACK: the optimizer
    state is deep-copied at ``n_steps``, the probe continues to the largest probe step to
    record how the second moment is still developing, and the ``n_steps`` state is then
    reloaded. So the run uses exactly ``n_steps`` of warm-up while still reporting whether
    that was enough.
    """
    opt = trainer.opt
    probe = sorted(set(int(s) for s in probe_steps))
    n_run = max(int(n_steps), max(probe) if probe else 0)

    enc_params = [p for p in trainer.encoder.parameters() if p.requires_grad]
    for p in enc_params:
        p.requires_grad_(False)

    params = [p for g in opt.param_groups for p in g["params"]]
    step0 = optimizer_step_counter(opt)
    norms, skipped, state_at_n = {}, 0, None

    if verbose:
        print(f"  [adam] warming moments for {n_steps} steps (probing to {n_run}); "
              f"weights restored every step; encoder left COLD "
              f"({len(enc_params)} params frozen for the duration)", flush=True)

    it = iter(paired_loader)
    for k in range(1, n_run + 1):
        try:
            sb, tb = next(it)
        except StopIteration:
            it = iter(paired_loader)
            sb, tb = next(it)
        backup = [p.detach().clone() for p in params]
        stats = trainer.train_step(sb, tb, step=int(mu_step_offset),
                                   total_steps=int(total_steps))
        skipped += int(bool(stats.get("step_skipped", 0.0)))
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.copy_(b)
        del backup
        if k in probe:
            nrm, cnt = second_moment_norm(opt)
            # THE RAW NORM IS THE WRONG THING TO JUDGE CONVERGENCE ON. exp_avg_sq is an EMA
            # started at zero, so after k steps it is ~(1 - beta2**k) * E[g^2] and MUST keep
            # climbing regardless of how well E[g^2] is estimated. Adam never uses the raw
            # value -- it divides by exactly that factor (bias correction). The debiased
            # norm below is the quantity that actually enters the update, and it is what
            # "has the warm-up converged?" has to be asked of.
            beta2 = float(opt.param_groups[0].get("betas", (0.9, 0.999))[1])
            debias = 1.0 - beta2 ** k
            norms[k] = {"second_moment_norm": nrm,
                        "bias_correction_factor": debias,
                        "second_moment_norm_debiased": nrm / debias if debias else None,
                        "n_params_with_state": cnt,
                        "step_counter": optimizer_step_counter(opt)}
            if verbose:
                print(f"  [adam]   step {k:>4}: ||exp_avg_sq|| = {nrm:.6e}   "
                      f"debiased = {nrm/debias:.6e}   ({cnt} params with state)",
                      flush=True)
        if k == int(n_steps):
            state_at_n = copy.deepcopy(opt.state_dict())

    if state_at_n is not None and n_run > int(n_steps):
        opt.load_state_dict(state_at_n)
        if verbose:
            print(f"  [adam] rolled optimizer state back to step {n_steps} "
                  f"(the probe beyond it is diagnostic only)", flush=True)

    for p in enc_params:
        p.requires_grad_(True)

    step1 = optimizer_step_counter(opt)
    final_norm, final_cnt = second_moment_norm(opt)

    # "still climbing at n_steps?" -- asked of BOTH the raw and the debiased norm, because
    # they answer different questions. The raw one always climbs (EMA bias); only the
    # debiased one can tell you whether E[g^2] is still being learned.
    climbing = climbing_debiased = None
    if len(probe) >= 2:
        a = norms.get(int(n_steps)) or norms.get(probe[-2])
        b = norms.get(probe[-1])
        if a and b and a["second_moment_norm"] > 0:
            climbing = (b["second_moment_norm"] - a["second_moment_norm"]) \
                / a["second_moment_norm"]
        if a and b and (a.get("second_moment_norm_debiased") or 0) > 0:
            climbing_debiased = (b["second_moment_norm_debiased"]
                                 - a["second_moment_norm_debiased"]) \
                / a["second_moment_norm_debiased"]

    report = {
        "method": ("reconstruction: moments accumulated from real gradients under the "
                   "corrected objective; weights restored every step"),
        "n_warm_steps": int(n_steps), "n_probe_steps": int(n_run),
        "mu_step_offset": int(mu_step_offset),
        "probes": norms,
        "second_moment_norm_after": final_norm,
        "n_params_with_state_after": final_cnt,
        "step_counter_before": step0, "step_counter_after": step1,
        "step_counter_carried": bool(step1 >= int(n_steps)),
        "fp16_steps_skipped_during_warmup": skipped,
        "encoder_params_left_cold": len(enc_params),
        "encoder_asymmetry_note": (
            "the head's optimizer state existed in round 1 and was lost, so it is "
            "reconstructed; the encoder's never existed because round 1 never trained the "
            "encoder, so it is deliberately left at zero -- warming it would invent state "
            "for a phase of training that did not happen"),
        "relative_growth_over_probe": climbing,
        "relative_growth_over_probe_debiased": climbing_debiased,
        "converged": (None if climbing_debiased is None
                      else bool(abs(climbing_debiased) <= 0.05)),
        "convergence_note": (
            "judged on the DEBIASED second moment (v / (1 - beta2**k)), which is what Adam "
            "actually divides by. The raw norm necessarily climbs as (1 - beta2**k) no "
            "matter how well E[g^2] is estimated, so raw growth is not evidence that the "
            "warm-up was too short."),
    }
    if verbose:
        print(f"  [adam] step counter {step0:g} -> {step1:g} "
              f"(carried into training: {report['step_counter_carried']})")
        if climbing is not None:
            print(f"  [adam] raw ||exp_avg_sq|| grew {climbing*100:+.1f}% over the probe "
                  f"-- expected: it is an EMA from zero, not a convergence signal")
        if climbing_debiased is not None:
            print(f"  [adam] DEBIASED second moment moved {climbing_debiased*100:+.2f}% "
                  f"-- {'CONVERGED' if abs(climbing_debiased) <= 0.05 else 'STILL MOVING'}"
                  f" (this is the quantity Adam uses)")
    return report


# ---------------------------------------------------------------------------
# 3. checkpoint-format audit
# ---------------------------------------------------------------------------

LEGACY_KEYS = {"cfg", "head", "head_type", "standardizer"}


def audit_checkpoint(path):
    """Classify one checkpoint: does it carry the training state around the model?"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    keys = set(ck.keys())
    return {
        "path": str(path), "keys": sorted(keys),
        "has_optimizer": "optimizer" in keys,
        "has_da_buffers": "da_buffers" in keys,
        "has_schedule": "schedule" in keys,
        "has_encoder_delta": "encoder_trainable" in keys,
        "legacy_weights_only": keys == LEGACY_KEYS,
        "continuable": {"optimizer", "da_buffers", "schedule"} <= keys,
    }
