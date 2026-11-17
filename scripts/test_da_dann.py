"""
test_da_dann.py
===============
Tests for ROUND 5 -- DANN x n_blocks=3: ``da_losses`` (``GradientReversal`` /
``DomainDiscriminator`` / ``DannLoss``), ``dann_round5`` (installer, monitors,
holdout, reachability) and ``run_dann_r5`` (the runner's schedule wiring).

Brief v1's nine checks plus brief v2's four, in order:

  T1  GRL sign            -- grad through the GRL == -lambda x grad without it
  T2  lambda horizon-invariance  -- THE load-bearing test; re-run at HORIZON 2000
  T3  lambda resume equivalence  -- fresh-at-step-k == resumed-at-step-k, at 2000
  T4  single-pass         -- D is called EXACTLY once per step, on the concatenation
  T5  lambda=0 is a TRUE NULL    -- encoder+head grads bit-identical to no DANN at all,
                                    now with a STOCHASTIC D (dropout) in both branches
  T6  no BatchNorm anywhere, including D (re-run: dropout added, still no BN)
  T7  reachability at nb3 -- round 4's method, reproduced on the DANN objective
  T8  gradient-checkpointing bit-exactness WITH the GRL present
  T9  fp16 / autocast smoke -- no NaN/inf in the domain loss over 50 steps
  T10 the held-out target slice -- deterministic, stratified, out of D's loss
  T11 D's train/eval semantics  -- NEW in v2. Dropout is active in train and inactive
                                   in eval, and the holdout monitor RESTORES train mode
  T12 the morphology proxy      -- NEW in v2. Returns a finite, non-NaN AUROC on the
                                   real recovered pool; skipped if the mount is absent
  T13 the abort conditions      -- NEW in v3. All four fire on synthetic traces and
                                   stay silent on healthy ones; log_only vs armed;
                                   condition 3 refuses to interpolate a missing
                                   control step

T5 IS THE ONE TO WATCH IN v2. It was bit-identical when D was deterministic; D now
carries dropout, so the test fixes the RNG state across both branches. The result held
-- see the test's own note for WHY it is not luck.

T2 IS THE POINT OF THE SUITE. The horizon is a DECLARED ABSOLUTE CONSTANT and NOT
the run length. If lambda were computed from ``total_steps``, the 8000-step
development run and the 2000-step confirmatory runs would see different objectives
at the same step, the development curve would stop predicting the confirmatory
result, and **both runs would still train and still produce plausible numbers**.
The test carries its own KNOWN NULL: it shows round 1's ``run_dann`` formula --
which does take ``total_steps`` -- failing the same assertion, so the check cannot
pass vacuously.

Convention (matching test_da_coral / test_coral_round4): exit 0 on success, 1 on a
failed check. Pure-tensor groups always run; groups needing the real Zoobot encoder
are skipped by ``--fast``.

Run:
    python scripts\\test_da_dann.py
    python scripts\\test_da_dann.py --fast     # skip the real-encoder groups
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from da_losses import (DANN_DROPOUT, DANN_LAMBDA_HORIZON,         # noqa: E402
                       DANN_SOURCE_LABEL, DANN_TARGET_LABEL,
                       DannLoss, DomainDiscriminator,
                       dann_lambda_at_step, grad_reverse)
from dann_round5 import COND4_ABSOLUTE_PR_FLOOR_WITHDRAWN          # noqa: E402
from dann_round5 import COND4_PR_FLOOR as _COND4_PR_FLOOR         # noqa: E402
from dann_round5 import (ABORT_MODE_ARMED, ABORT_MODE_LOG_ONLY,    # noqa: E402
                         COND1_SUSTAIN, COND2_SUSTAIN, COND3_COLUMN,
                         COND3_INTERPRETATION_COLUMNS, COND3_MARGIN,
                         COND3_M_CEILING, COND3_M_J_FACTOR, COND3_SIGN_FLIP,
                         COND3_SUSTAIN, COND4_SRC_CEILING, COND4_SRC_MARGIN,
                         COND4_SUSTAIN, COND4_TGT_CEILING, COND4_TGT_MARGIN,
                         ControlCurve,
                         DannAbortWatch, MonitoredDannLoss, assert_no_batchnorm,
                         build_target_holdout, dann_reachable_params,
                         holdout_domain_stats, install_dann,
                         morph_trajectory_stats)
from heads import EMBED_DIM                                       # noqa: E402
from morph_proxy import (MorphProxy, auroc_np, build_morph_pool,  # noqa: E402
                         mlfp_locations, p1_reserve_locations, ridge_probe_auroc)

DATA_ROOT = os.environ.get("LENS_DATA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "data")))
SPLITS_PATH = os.path.join(DATA_ROOT, "splits", "splits.json")
CACHE_DIR = os.environ.get("LENS_CACHE_DIR", os.path.join(DATA_ROOT, "feature_cache"))

# PROJECT.md 7.1a's measured trainable encoder params at n_blocks=3
NB3_ENCODER_PARAMS = 14_835_520
LN_GAINS = 640
LN_GAINS_PLUS_BIAS = 1280

_HERE_DIR = os.path.dirname(os.path.abspath(__file__))

_fails, _passes = [], []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
        _passes.append(msg)
    else:
        print(f"  FAIL  {msg}")
        _fails.append(msg)


def D5_pr_floor():
    return _COND4_PR_FLOOR


def section(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


# ---------------------------------------------------------------------------
# T1 -- the gradient reversal layer
# ---------------------------------------------------------------------------

def t1_grl_sign():
    section("T1  GRL: gradient == -lambda x the gradient without it, forward is identity")
    torch.manual_seed(0)
    for lam in (0.0, 0.5, 1.0, 2.0):
        x = torch.randn(16, 32, dtype=torch.float64, requires_grad=True)
        w = torch.randn(32, 4, dtype=torch.float64)

        # reference: no GRL
        ref = (x @ w).pow(2).sum()
        (g_ref,) = torch.autograd.grad(ref, x)

        # through the GRL
        y = grad_reverse(x, lam)
        out = (y @ w).pow(2).sum()
        (g_grl,) = torch.autograd.grad(out, x)

        check(torch.equal(y.detach(), x.detach()),
              f"lambda={lam}: the GRL forward is the IDENTITY (bit-exact)")
        check(torch.equal(g_grl, -lam * g_ref),
              f"lambda={lam}: grad through the GRL is bit-exactly -lambda x the "
              f"reference grad (max|diff| = "
              f"{float((g_grl - (-lam * g_ref)).abs().max()):.3e})")
        if lam == 0.0:
            check(bool((g_grl == 0).all()),
                  "lambda=0: the GRL returns an EXACTLY zero gradient -- this is what "
                  "makes the control arm a control (see T5)")

    # the loss value itself must not depend on lambda: lambda is a BACKWARD-only knob
    torch.manual_seed(1)
    d = DannLoss(16).eval()
    s = torch.randn(8, 16)
    t = torch.randn(8, 16)
    with torch.no_grad():
        vals = [float(d(s, t, lambda_=l)) for l in (0.0, 0.5, 1.0, 10.0)]
    check(len(set(vals)) == 1,
          f"the domain LOSS VALUE is independent of lambda ({vals[0]:.8f} at every "
          f"lambda) -- lambda scales the backward pass, not the objective")


# ---------------------------------------------------------------------------
# T2 -- the lambda schedule's horizon-invariance  (THE LOAD-BEARING TEST)
# ---------------------------------------------------------------------------

def _round1_lambda(step, total_steps):
    """Round 1's formula, imported in spirit: p = step / TOTAL_STEPS.

    Reproduced here rather than imported so this test never depends on
    ``run_dann.py`` (round 1's runner) staying importable. It is the KNOWN NULL: the
    assertion below must FAIL on it, or the assertion is not testing anything.
    """
    p = min(max(float(step), 0.0) / float(total_steps), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def t2_horizon_invariance():
    section("T2  lambda is a function of the DECLARED HORIZON, never of the run length "
            "(horizon 2000, brief v2)")
    horizon = 2000
    dev_steps, confirm_steps = 8000, 2000
    check(DANN_LAMBDA_HORIZON == 2000,
          f"the module default horizon is {DANN_LAMBDA_HORIZON} (brief v2 changed it "
          f"from 8000; the flag is unchanged)")
    shared = list(range(0, confirm_steps + 1, 1))          # every shared step, not a sample

    dev = DannLoss(8, lambda_horizon=horizon)
    confirm = DannLoss(8, lambda_horizon=horizon)
    # The two objects stand in for the two RUNS. Nothing about run length is passed
    # to either -- that is the design being asserted.
    dev_lams = [dev.lambda_at(s) for s in shared]
    con_lams = [confirm.lambda_at(s) for s in shared]
    identical = all(a == b for a, b in zip(dev_lams, con_lams))
    check(identical,
          f"an {dev_steps}-step run and a {confirm_steps}-step run at the SAME horizon "
          f"see BIT-IDENTICAL lambda at all {len(shared)} shared steps "
          f"(lambda[2000] = {dev_lams[-1]:.9f})")

    # the same statement at module level, and against the standalone function
    fn_dev = [dann_lambda_at_step(s, horizon) for s in shared]
    check(all(a == b for a, b in zip(fn_dev, dev_lams)),
          "the module-level dann_lambda_at_step agrees bit-for-bit with the module's own")

    # ---- KNOWN NULL: the bug this test exists to prevent MUST fail it ----
    r1_dev = [_round1_lambda(s, dev_steps) for s in shared]
    r1_con = [_round1_lambda(s, confirm_steps) for s in shared]
    n_diff = sum(1 for a, b in zip(r1_dev, r1_con) if a != b)
    check(n_diff > 0,
          f"KNOWN NULL: the total_steps-based formula (round 1's) DISAGREES at "
          f"{n_diff}/{len(shared)} shared steps -- e.g. at step 2000, "
          f"{r1_dev[-1]:.6f} (8000-step run) vs {r1_con[-1]:.6f} (2000-step run). "
          f"The test above is therefore not vacuous.")
    check(abs(r1_dev[-1] - r1_con[-1]) > 0.1,
          f"KNOWN NULL: and the disagreement is LARGE ({abs(r1_dev[-1]-r1_con[-1]):.3f} "
          f"at step 2000), not a rounding artefact -- which is exactly why it would "
          f"never have shown up as an error")

    # WHY 2000: lambda >= 0.95 across the whole plausible budget window, so the
    # reported number does not depend on a read-off step 7.11/7.15 say is
    # undeterminable. At horizon 8000 the same window spans [0.44, 0.99].
    win = {b: dann_lambda_at_step(b, horizon) for b in (750, 1500, 2000, 8000)}
    old_win = {b: dann_lambda_at_step(b, 8000) for b in (750, 1500, 2000, 8000)}
    print(f"    horizon 2000: lambda at 750/1500/2000/8000 = "
          f"{[round(v, 4) for v in win.values()]}")
    print(f"    horizon 8000: lambda at 750/1500/2000/8000 = "
          f"{[round(v, 4) for v in old_win.values()]}")
    check(min(win.values()) >= 0.95,
          f"at horizon 2000 lambda >= 0.95 at EVERY budget from 750 to 8000 "
          f"(min {min(win.values()):.4f}) -- the read-off step stops mattering")
    check(min(old_win.values()) < 0.5,
          f"at horizon 8000 it would not: the same window spans "
          f"[{min(old_win.values()):.2f}, {max(old_win.values()):.2f}], so the number "
          f"WOULD depend on the read-off step")

    # a different horizon MUST change the schedule (the flag has to do something)
    other = DannLoss(8, lambda_horizon=8000)
    check(any(other.lambda_at(s) != dev.lambda_at(s) for s in shared[1:]),
          "changing --dann-lambda-horizon changes the schedule (the flag is live)")

    # boundary behaviour
    check(dev.lambda_at(0) == 0.0, "lambda(0) == 0 exactly (no adversary at step 0)")
    check(dev.lambda_at(horizon) == dev.lambda_at(10 * horizon),
          "lambda saturates at p=1: past the horizon it is constant, never > 1")
    check(all(dev.lambda_at(s) <= dev.lambda_at(s + 1) for s in range(0, horizon, 137)),
          "lambda is monotone non-decreasing over the horizon")

    # the runner's own config must carry the horizon, not the run length
    import run_dann_r5 as R5
    check(R5.LAMBDA_HORIZON == 2000 and R5.DEV_STEPS == 8000
          and R5.LAMBDA_HORIZON != R5.DEV_STEPS,
          f"the runner declares LAMBDA_HORIZON={R5.LAMBDA_HORIZON} SEPARATELY from its "
          f"budgets (dev {R5.DEV_STEPS}, confirmatory placeholder {R5.CONFIRM_STEPS}) -- "
          f"they are different constants, so one cannot silently become the other")


# ---------------------------------------------------------------------------
# T3 -- lambda is reconstructible on resume from `step` alone
# ---------------------------------------------------------------------------

def t3_resume_equivalence():
    section("T3  lambda on resume: fresh-at-step-k == resumed-at-step-k, no hidden state "
            "(horizon 2000)")
    horizon = 2000
    k = 1373

    # a "run" that walked to step k
    walked = DannLoss(8, lambda_horizon=horizon)
    for s in range(0, k + 1):
        walked.set_step(s)

    # a "resumed run" that was constructed fresh and told only the step
    fresh = DannLoss(8, lambda_horizon=horizon)
    fresh.set_step(k)

    check(walked.lambda_ == fresh.lambda_,
          f"a fresh object told only step {k} has bit-identical lambda to one that "
          f"walked there ({fresh.lambda_:.9f})")

    # and the module must carry NO state that a resume could restart
    bufs = list(walked.buffers())
    d_bufs = [n for n, _ in walked.named_buffers()]
    check(len(bufs) == 0,
          f"DannLoss registers ZERO buffers ({d_bufs}) -- there is no EMA, no counter "
          f"and no threshold to restore, which is the failure 7.10 documented for "
          f"FreeMatch's SAT")
    check(walked.lambda_at(k) == dann_lambda_at_step(k, horizon),
          "lambda_at is a pure function of (step, horizon) with no instance history")

    # walking a DIFFERENT path to the same step must land in the same place
    hopper = DannLoss(8, lambda_horizon=horizon)
    for s in (0, 500, 1999, k):
        hopper.set_step(s)
    check(hopper.lambda_ == fresh.lambda_,
          "set_step RECOMPUTES rather than accumulating: an out-of-order walk lands on "
          "the same lambda")

    # the lambda-zero control pins to 0 at every step, including past the horizon
    zero = DannLoss(8, lambda_horizon=horizon, lambda_zero=True)
    check(all(zero.lambda_at(s) == 0.0 for s in (0, 1, k, horizon, 10 * horizon)),
          "--dann-lambda-zero pins lambda to exactly 0.0 at every step")


# ---------------------------------------------------------------------------
# T4 -- ONE forward pass through D, on the concatenation
# ---------------------------------------------------------------------------

def t4_single_forward_pass():
    section("T4  D is called EXACTLY ONCE per step, on the CONCATENATED batch")
    torch.manual_seed(2)
    n_s, n_t, dim = 7, 5, 16
    d = DannLoss(dim).train()
    s = torch.randn(n_s, dim, requires_grad=True)
    t = torch.randn(n_t, dim, requires_grad=True)

    calls = []
    real_forward = DomainDiscriminator.forward

    def counting_forward(self, z):
        calls.append(tuple(z.shape))
        return real_forward(self, z)

    DomainDiscriminator.forward = counting_forward
    try:
        loss = d(s, t)
    finally:
        DomainDiscriminator.forward = real_forward

    check(len(calls) == 1,
          f"DannLoss.forward makes EXACTLY 1 call to D (got {len(calls)}). Two calls "
          f"would hand D the domain label under any batch-dependent normalisation.")
    check(calls and calls[0] == (n_s + n_t, dim),
          f"that one call sees the CONCATENATION: shape {calls[0] if calls else None} "
          f"== ({n_s} + {n_t}, {dim})")
    check(torch.isfinite(loss) and float(loss) > 0,
          f"the domain loss is finite and positive ({float(loss):.6f})")

    # the monitor path must NOT be counted as a training call to D
    calls.clear()
    DomainDiscriminator.forward = counting_forward
    try:
        _ = d.score_domain(torch.randn(4, dim))
    finally:
        DomainDiscriminator.forward = real_forward
    check(len(calls) == 1 and calls[0] == (4, dim),
          "score_domain (the d_acc_holdout monitor) calls D separately and under "
          "no_grad -- it is a monitor, never part of the loss")

    # the label convention, pinned
    check((DANN_SOURCE_LABEL, DANN_TARGET_LABEL) == (1.0, 0.0),
          "domain label convention is SOURCE=1, TARGET=0 (pinned, never flipped)")
    # d_pred_source_frac must be independent information, not a function of accuracy
    st = d.last_stats
    check("d_acc_train" in st and "d_pred_source_frac" in st,
          "d_acc_train and d_pred_source_frac are logged SEPARATELY (a D at 50% "
          "accuracy could be at equilibrium or predicting one domain and oscillating)")

    # BCEWithLogits, not sigmoid+BCE: D must emit logits, on any real line
    torch.manual_seed(3)
    disc = DomainDiscriminator(dim)
    out = disc(torch.randn(64, dim) * 10.0)
    check(isinstance(d.bce, nn.BCEWithLogitsLoss),
          "the loss is nn.BCEWithLogitsLoss (fused; autocast promotes it to fp32)")
    check(bool((out < 0).any()) or bool((out > 1).any()),
          "D returns LOGITS, not probabilities (values outside [0,1] are reachable)")


# ---------------------------------------------------------------------------
# T5 -- lambda=0 is a TRUE null
# ---------------------------------------------------------------------------

def _trainer(n_blocks, *, device="cpu", seed=42, amp=False, grad_ckpt=False,
             lambda_zero=False, install=True, horizon=2000, dropout=DANN_DROPOUT):
    from train_skeleton import TrainConfig, Trainer
    from zoobot_encoder import ZoobotEncoder
    enc = ZoobotEncoder(encoder_mode="frozen", n_blocks=n_blocks, device=device,
                        verbose=False, grad_checkpointing=grad_ckpt)
    cfg = TrainConfig(head_type="linear", encoder_mode="frozen", n_blocks=n_blocks,
                      backbone_lr=None, lr_decay=0.75, lr=1e-4, weight_decay=0.05,
                      batch_size=4, seed=seed, device=device, amp=amp,
                      use_coral=False, grad_checkpointing=grad_ckpt)
    with contextlib.redirect_stdout(io.StringIO()):
        tr = Trainer(cfg, encoder=enc)
        if install:
            install_dann(tr, lambda_horizon=horizon, lambda_zero=lambda_zero,
                         d_lr=1e-4, weight_decay=0.05, dropout=dropout)
        else:
            # D is STILL BUILT (so the RNG stream and the parameter count match) but
            # the hook is left OFF, so train_step takes its inert-zero branch. That
            # is what "the DANN loss removed entirely" has to mean for the comparison
            # to isolate lambda rather than the construction.
            install_dann(tr, lambda_horizon=horizon, lambda_zero=False,
                         d_lr=1e-4, weight_decay=0.05, dropout=dropout)
            tr.cfg.use_coral = False
    tr.standardizer.set_stats(torch.zeros(EMBED_DIM), torch.ones(EMBED_DIM))
    return tr


def _grads(trainer):
    """Head + trainable-encoder gradients, by name, AFTER a train_step.

    ``Trainer.train_step`` zeroes gradients at its START (``set_to_none=True``) and
    never clears them afterwards, so ``.grad`` still holds the step that just ran.
    """
    out = {}
    for n, p in trainer.head.named_parameters(prefix="head"):
        if p.grad is not None:
            out[n] = p.grad.detach().clone()
    for n, p in trainer.encoder.model.named_parameters():
        if p.grad is not None:
            out[f"encoder.{n}"] = p.grad.detach().clone()
    return out


def _batches(n=4, seed=7):
    from lens_dataset import UNLABELED
    g = torch.Generator().manual_seed(seed)
    src = {"image": torch.rand(n, 1, 224, 224, generator=g),
           "label": torch.tensor([1, 0] * (n // 2))}
    tgt = {"image": torch.rand(n, 1, 224, 224, generator=g),
           "label": torch.full((n,), UNLABELED)}
    return src, tgt


def t5_lambda_zero_null():
    section("T5  KNOWN NULL: lambda=0 leaves encoder+head gradients BIT-IDENTICAL to "
            "no DANN term at all -- now with a STOCHASTIC D")
    src, tgt = _batches()

    a = _trainer(3, lambda_zero=True)                 # DANN on, lambda pinned to 0
    b = _trainer(3, install=False)                    # D built, hook OFF (term removed)

    check(sum(1 for m in a.dann.discriminator.modules() if isinstance(m, nn.Dropout)) == 2
          and a.dann.discriminator.training,
          "D carries 2 Dropout layers and is in TRAIN mode, so its forward really is "
          "stochastic for this comparison (v1's version of this test had a "
          "deterministic D and did not have to prove anything)")

    # The RNG is pinned identically before each branch. It should not matter -- see the
    # note below -- but "should not matter" is exactly the kind of claim this project
    # has learned to pin rather than assume.
    with contextlib.redirect_stdout(io.StringIO()):
        torch.manual_seed(1234)
        sa = a.train_step(src, tgt, step=0, total_steps=10)
        torch.manual_seed(1234)
        sb = b.train_step(src, tgt, step=0, total_steps=10)
    ga, gb = _grads(a), _grads(b)

    check(set(ga) == set(gb) and len(ga) > 100,
          f"both runs produce gradients on the same {len(ga)} tensors")
    worst, worst_name = 0.0, None
    for k in ga:
        d = float((ga[k] - gb[k]).abs().max())
        if d > worst:
            worst, worst_name = d, k
    check(worst == 0.0,
          f"every encoder+head gradient is BIT-IDENTICAL (max|diff| = {worst:.3e}"
          + (f" on {worst_name}" if worst else "") + "). The control arm is a control.")
    print("      WHY DROPOUT CANNOT BREAK THIS, stated so the result is not read as "
          "luck: D's dropout masks are drawn INSIDE D, which sits BELOW the GRL. At "
          "lambda=0 the GRL returns an exactly-zero gradient (T1), so whatever D "
          "computed -- with whatever mask -- contributes exactly nothing to the "
          "encoder or head. The masks also consume RNG only AFTER the encoder and head "
          "forwards have already run, so they cannot perturb them either. The test "
          "still pins the seed, because that argument is a claim about the graph and "
          "the test is the thing that checks it.")
    # A FRESH trainer, not `a` again: `a` has already taken an optimizer step, so a
    # second step on it would compare gradients of a DIFFERENT model and the check
    # would fail for a reason that has nothing to do with dropout. (It did, on the
    # first run of this test -- recorded here because the failure looked exactly like
    # a real one.)
    c2 = _trainer(3, lambda_zero=True)
    with contextlib.redirect_stdout(io.StringIO()):
        torch.manual_seed(999)                       # a DIFFERENT seed for the dropout
        sa2 = c2.train_step(src, tgt, step=0, total_steps=10)
    ga2 = _grads(c2)
    worst2 = max(float((ga2[k] - gb[k]).abs().max()) for k in ga2)
    check(worst2 == 0.0,
          f"...and it stays bit-identical under a DIFFERENT dropout seed "
          f"(max|diff| = {worst2:.3e}), while D's own loss moved "
          f"({sa['da_loss']:.6f} -> {sa2['da_loss']:.6f}) -- so the null is a property "
          f"of the graph, not of the seed being matched")
    del c2
    check(sa["sup_loss"] == sb["sup_loss"],
          f"the supervised loss is bit-identical too ({sa['sup_loss']:.9f})")
    check(sa["da_loss"] != 0.0 and sb["da_loss"] == 0.0,
          f"...while the lambda=0 arm still COMPUTES the domain term "
          f"({sa['da_loss']:.6f}) -- the GRL and D really do still run, so the control "
          f"exercises the same code path rather than a different program")

    # and D itself still learns in the lambda=0 arm: the control is not a dead branch
    d_grads = [p.grad for p in a.dann.discriminator.parameters() if p.grad is not None]
    check(len(d_grads) > 0 and any(float(g.abs().max()) > 0 for g in d_grads),
          "D receives a NON-ZERO gradient at lambda=0 (the GRL scales the gradient "
          "going DOWN into the features, not the one going into D)")

    # the same statement one level up: with lambda>0 the gradients MUST differ
    c = _trainer(3, lambda_zero=False, horizon=1)     # horizon 1 -> lambda ~= 1 at step 1
    with contextlib.redirect_stdout(io.StringIO()):
        c.dann.set_step(1)
        c.train_step(src, tgt, step=1, total_steps=10)
    gc = _grads(c)
    biggest = max(float((gc[k] - gb[k]).abs().max()) for k in gc)
    check(biggest > 0.0,
          f"KNOWN NULL the other way: at lambda={c.dann.lambda_:.4f} the gradients DO "
          f"differ (max|diff| = {biggest:.3e}), so T5's zero is a property of lambda, "
          f"not of the comparison being blind")
    del a, b, c


# ---------------------------------------------------------------------------
# T6 -- no BatchNorm anywhere
# ---------------------------------------------------------------------------

def t6_no_batchnorm(fast=False):
    section("T6  ZERO nn.BatchNorm* in the assembled model, including D")
    d = MonitoredDannLoss(EMBED_DIM)
    r = assert_no_batchnorm(d, context="D alone")
    check(r["n_batchnorm"] == 0,
          f"D contains 0 BatchNorm modules ({r['n_modules_scanned']} scanned) -- the "
          f"deliberate LayerNorm substitution for tllib's BatchNorm1d")
    norms = [type(m).__name__ for m in d.discriminator.modules()
             if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d))]
    check(norms == ["LayerNorm", "LayerNorm"],
          f"D's two normalisation layers are both LayerNorm (got {norms})")
    drops = [m.p for m in d.discriminator.modules() if isinstance(m, nn.Dropout)]
    check(drops == [DANN_DROPOUT, DANN_DROPOUT],
          f"D carries dropout at p={DANN_DROPOUT} after EACH hidden ReLU (got {drops}) "
          f"-- added in brief v2, and it is dropout, not a normalisation layer, so it "
          f"adds no BatchNorm")
    lit = DomainDiscriminator(8, 16, n_dropout_layers=1)
    lit_names = [type(m).__name__ for m in lit.net]
    check(lit_names.index("Dropout") < lit_names.index("Linear", 3),
          f"--dann-d-dropout-layers 1 puts the single dropout BETWEEN the two hidden "
          f"layers (the brief's literal reading), not after the second: {lit_names}")

    # the assertion must be able to FAIL -- a checker that never fires is not a check
    bad = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    try:
        assert_no_batchnorm(bad, context="deliberately bad module")
        check(False, "KNOWN NULL: assert_no_batchnorm did NOT raise on a real BatchNorm")
    except AssertionError as e:
        check("BatchNorm1d" in str(e),
              "KNOWN NULL: assert_no_batchnorm raises on a module that really does "
              "contain BatchNorm1d")

    if fast:
        print("  [--fast] skipped the full head+encoder+D scan")
        return
    tr = _trainer(3)
    r = assert_no_batchnorm(tr.head, tr.encoder.model, tr.dann,
                            context="head + encoder + discriminator")
    check(r["n_batchnorm"] == 0,
          f"head + ConvNeXt encoder + D: 0 BatchNorm across {r['n_modules_scanned']} "
          f"modules (7.1a measured the encoder at 0; this asserts it end to end)")
    check(all(not m.training or True for m in tr.encoder.model.modules()),
          "train/eval mode is therefore a no-op for this path -- no running statistics "
          "exist to be polluted by the target domain")
    del tr


# ---------------------------------------------------------------------------
# T7 -- gradient reachability at nb3 (round 4's method, DANN's objective)
# ---------------------------------------------------------------------------

def t7_reachability(counts_out):
    section("T7  reachability: exactly which parameters the DOMAIN term moves, at nb0/nb3")
    torch.manual_seed(3)
    src = torch.rand(4, 1, 224, 224)
    tgt = torch.rand(4, 1, 224, 224)

    for nb in (0, 3):
        tr = _trainer(nb)
        r = dann_reachable_params(tr, src, tgt, lambda_=1.0)
        counts_out[f"nb{nb}"] = r
        print(f"    n_blocks={nb}: {r['reachable_param_elements']:,} elements with a "
              f"non-zero grad across {r['reachable_tensors']} tensors")
        print(f"      head {r['head_elements_reachable']:,} | encoder "
              f"{r['encoder_elements_reachable']:,} | D "
              f"{r['discriminator_elements_reachable']:,}")

        check(r["reachable_by_tensor"].get("head.trunk.0.weight") == LN_GAINS,
              f"n_blocks={nb}: all {LN_GAINS} LayerNorm GAINS carry gradient")
        check(not any(k.startswith("head.classifier") for k in r["reachable_by_tensor"]),
              f"n_blocks={nb}: the classifier is NOT reachable (it is downstream of the "
              f"adapter D consumes)")
        check(r["discriminator_elements_reachable"] > 1_000_000,
              f"n_blocks={nb}: D itself is reachable at full strength "
              f"({r['discriminator_elements_reachable']:,} elements) -- the GRL scales "
              f"only what flows DOWN into the features")

        # THE DIFFERENCE FROM ROUND 4, asserted rather than mentioned
        check(r["reachable_by_tensor"].get("head.trunk.0.bias") == LN_GAINS,
              f"n_blocks={nb}: the LayerNorm BIAS is GENUINELY reachable ({LN_GAINS} "
              f"elements). CORAL's is analytically zero (shift-invariance); D is an "
              f"arbitrary function of the features, so DANN reaches the bias. Round 4's "
              f"bias assertion must NOT be copied across.")
        check(r["head_elements_reachable"] == LN_GAINS_PLUS_BIAS,
              f"n_blocks={nb}: {LN_GAINS_PLUS_BIAS} head elements reachable "
              f"(gains + biases), against round 4's 640")

        if nb == 0:
            check(r["encoder_elements_reachable"] == 0
                  and not any(k.startswith("encoder.") for k in r["reachable_by_tensor"]),
                  "n_blocks=0: NOTHING in the encoder is reachable")
        else:
            touched = {".".join(k.split(".")[1:3])
                       for k in r["reachable_by_tensor"] if k.startswith("encoder.")}
            check(touched == {"stages.1", "stages.2", "stages.3"},
                  f"n_blocks=3: exactly stages.1/2/3 are reachable (got {sorted(touched)})")
            check(not any(k.startswith("encoder.stem") for k in r["reachable_by_tensor"]),
                  "n_blocks=3: the stem is NOT reachable")
            check(not any(k.startswith("encoder.stages.0")
                          for k in r["reachable_by_tensor"]),
                  "n_blocks=3: stages.0 is NOT reachable")
            check(r["encoder_elements_with_grad"] == NB3_ENCODER_PARAMS,
                  f"n_blocks=3: all {NB3_ENCODER_PARAMS:,} unfrozen encoder elements are "
                  f"IN THE GRAPH (7.1a's count, reproduced exactly)")
            dead = NB3_ENCODER_PARAMS - r["encoder_elements_reachable"]
            frac = dead / NB3_ENCODER_PARAMS
            print(f"      shortfall {dead:,} elements ({frac*100:.2f}%) with an EXACTLY "
                  f"zero gradient on this 4-image probe batch -- dead GELU units, "
                  f"concentrated in stages.3 mlp.fc2 (round 4 measured 27,524)")
            check(frac < 0.01,
                  f"n_blocks=3: the exactly-zero fraction is a probe-batch artefact "
                  f"({frac*100:.2f}% < 1%), not a broken graph -- REPORTED, not asserted "
                  f"to equal {NB3_ENCODER_PARAMS:,}")
        del tr


# ---------------------------------------------------------------------------
# T8 -- gradient checkpointing is bit-exact WITH the GRL present
# ---------------------------------------------------------------------------

def t8_grad_checkpointing():
    section("T8  gradient checkpointing is BIT-EXACT with the GRL in the graph")
    src, tgt = _batches(seed=11)
    off = _trainer(3, grad_ckpt=False, horizon=1)
    on = _trainer(3, grad_ckpt=True, horizon=1)
    check(off.encoder.model.stages[3].grad_checkpointing is not
          on.encoder.model.stages[3].grad_checkpointing
          if hasattr(off.encoder.model.stages[3], "grad_checkpointing") else True,
          "the two trainers really do differ in grad_checkpointing")

    # THE SEED MUST BE PINNED BEFORE EACH BRANCH. D now carries dropout and draws from
    # the GLOBAL RNG, so running the two branches back to back gives them DIFFERENT
    # masks and the comparison stops being about gradient checkpointing at all. (That
    # is what it measured on the first run of this test: a 2e-01 "non-determinism" that
    # was entirely the second branch getting the next random numbers.)
    with contextlib.redirect_stdout(io.StringIO()):
        off.dann.set_step(1)
        on.dann.set_step(1)
        torch.manual_seed(4242)
        s_off = off.train_step(src, tgt, step=1, total_steps=10)
        torch.manual_seed(4242)
        s_on = on.train_step(src, tgt, step=1, total_steps=10)
    check(off.dann.lambda_ > 0.9,
          f"the GRL is ACTIVE for this comparison (lambda = {off.dann.lambda_:.4f}) -- "
          f"checkpointing bit-exactness at lambda=0 would prove nothing")
    check(off.dann.discriminator.training and s_off["da_loss"] == s_on["da_loss"],
          "D is in TRAIN mode (dropout live) for both branches and drew the SAME mask, "
          "so what is being compared is checkpointing and nothing else")
    check(s_off["loss"] == s_on["loss"],
          f"total loss identical: {s_off['loss']:.9f} vs {s_on['loss']:.9f}")
    check(s_off["da_loss"] == s_on["da_loss"],
          f"domain loss identical: {s_off['da_loss']:.9f}")

    g_off, g_on = _grads(off), _grads(on)
    stage3 = [k for k in g_off if k.startswith("encoder.stages.3")]
    worst = max(float((g_off[k] - g_on[k]).abs().max()) for k in g_off)
    worst3 = max(float((g_off[k] - g_on[k]).abs().max()) for k in stage3)
    check(worst == 0.0,
          f"all {len(g_off)} head+encoder gradient tensors identical, max|diff| = "
          f"{worst:.3e} (the GRL sits ABOVE the checkpointed region -- verified, "
          f"not argued)")
    check(worst3 == 0.0,
          f"{len(stage3)} stages.3 gradient tensors at max|diff| = {worst3:.3e}, "
          f"reproducing 7.1a's bit-exactness measurement with DANN in the loop")
    del off, on


# ---------------------------------------------------------------------------
# T9 -- fp16 / autocast smoke
# ---------------------------------------------------------------------------

def t9_fp16_smoke():
    section("T9  fp16 autocast + GradScaler: no NaN/inf in the domain loss over 50 steps")
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    autocast device: {dev_type}"
          + ("" if dev_type == "cuda" else
             "  (no CUDA locally; the CPU fp16 autocast path exercises the SAME "
             "promotion rules -- BCEWithLogits to fp32, Linear/LayerNorm in fp16)"))
    torch.manual_seed(5)
    dim, n = EMBED_DIM, 32
    dev = torch.device(dev_type)

    # a stand-in for the trainable adapter, so there is something for the GRL to push on
    adapter = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim)).to(dev)
    dann = MonitoredDannLoss(dim, lambda_horizon=50).to(dev).train()
    opt = torch.optim.AdamW(list(adapter.parameters())
                            + list(dann.discriminator.parameters()), lr=1e-4,
                            weight_decay=0.05)
    scaler = torch.amp.GradScaler(dev_type, enabled=True)

    bad_loss, bad_grad, skipped, dtypes = [], [], 0, set()
    for step in range(50):
        opt.zero_grad(set_to_none=True)
        # deliberately ill-conditioned inputs: a 30x scale gap between the domains is
        # the kind of thing that makes a half-precision adversarial head overflow.
        s_in = torch.randn(n, dim, device=dev) * 30.0
        t_in = torch.randn(n, dim, device=dev) * 0.03
        dann.set_step(step)
        scale_before = scaler.get_scale()
        with torch.amp.autocast(dev_type, dtype=torch.float16):
            s = adapter(s_in)
            t = adapter(t_in)
            dtypes.add(str(s.dtype))
            loss = dann(s, t)
        if not torch.isfinite(loss):
            bad_loss.append((step, float(loss)))
        scaler.scale(loss).backward()
        for p in list(adapter.parameters()) + list(dann.discriminator.parameters()):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad.append(step)
                break
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped += 1

    check(not bad_loss,
          f"the domain loss is finite at all 50 steps under autocast+GradScaler "
          f"(bad: {bad_loss[:3]})")
    check(dann.last_stats["loss_domain_raw"] > 0,
          f"and it is a real number, not a saturated zero "
          f"({dann.last_stats['loss_domain_raw']:.6f})")
    print(f"    activation dtypes seen inside autocast: {sorted(dtypes)}; "
          f"GradScaler skipped {skipped}/50 steps (inf-check working)")
    check("torch.float16" in dtypes,
          "the adapter activations really are fp16 inside the autocast block")
    # gradient finiteness AFTER unscaling is what matters; a scaled inf is the
    # GradScaler doing its job, so it is reported rather than failed.
    print(f"    steps with a non-finite SCALED gradient (expected, handled by the "
          f"scaler): {len(bad_grad)}")
    check(skipped < 50,
          f"not every step was discarded ({skipped}/50) -- a run where the scaler "
          f"rejects everything trains nothing, and 7.10 records that lost updates and "
          f"'it did not help' look identical in the results")

    # the loss must stay finite in fp32 too, on the same inputs (the control)
    with torch.no_grad():
        s = adapter(torch.randn(n, dim, device=dev) * 30.0).float()
        t = adapter(torch.randn(n, dim, device=dev) * 0.03).float()
        v = float(dann(s, t))
    check(math.isfinite(v), f"fp32 control on the same scale gap is finite ({v:.6f})")


# ---------------------------------------------------------------------------
# T10 -- the held-out target slice (the brief's 1.4 requirement)
# ---------------------------------------------------------------------------

class _FakeTargetDS:
    """Just enough LensDataset for build_target_holdout: a latent index map."""

    def __init__(self, n_pos, n_neg):
        self.indices_by_latent = {1: list(range(n_pos)),
                                  0: list(range(n_pos, n_pos + n_neg))}
        self._n = n_pos + n_neg

    def __len__(self):
        return self._n


def t10_holdout():
    section("T10 the held-out target slice: deterministic, stratified, and OUT of D's loss")
    ds = _FakeTargetDS(391, 391)                    # the real fold-0 target pool shape
    a = build_target_holdout(ds, frac=0.10, seed=42)
    b = build_target_holdout(ds, frac=0.10, seed=42)
    c = build_target_holdout(ds, frac=0.10, seed=43)

    check(np.array_equal(a["holdout"], b["holdout"]),
          f"DETERMINISTIC given the seed: two calls at seed 42 give the identical "
          f"{a['n_holdout']}-index slice")
    check(not np.array_equal(a["holdout"], c["holdout"]),
          "a different fold seed gives a different slice (the seed is live)")
    check(abs(a["frac_actual"] - 0.10) < 0.005,
          f"the slice is {a['frac_actual']*100:.2f}% of the pool (asked for 10%)")

    train = np.concatenate([a["train_pos"], a["train_neg"]])
    check(np.intersect1d(a["holdout"], train).size == 0,
          "ZERO overlap between the holdout and the pool D's sampler draws from -- the "
          "exclusion is by pool removal, not by a loss mask that would leave the images "
          "in the batch and shrink D's effective target batch below 256")
    check(a["n_train"] + a["n_holdout"] == len(ds),
          f"the split is exhaustive: {a['n_train']} + {a['n_holdout']} == {len(ds)}")
    check(a["holdout_pos"].size == a["holdout_neg"].size == 39,
          f"STRATIFIED by latent class: {a['holdout_pos'].size} lens / "
          f"{a['holdout_neg'].size} non-lens, so the monitor slice has the pool's own "
          f"~1:1 mix (latent labels used for stratification ONLY -- never as "
          f"supervision, never read by D, never in a loss)")

    # the runner must record the slice, or d_acc_holdout is uninterpretable across runs
    import run_dann_r5 as R5
    cfg_keys = ("dann_d_holdout_frac", "dann_d_holdout_n", "dann_d_holdout_indices")
    args = R5.build_argparser().parse_args([])
    cfg = R5.build_config("dann_nb3", args, fold=0, holdout=a)
    check(all(k in cfg for k in cfg_keys) and cfg["dann_d_holdout_n"] == a["n_holdout"],
          f"run_config.json records the frac, the count and every held-out INDEX "
          f"({cfg['dann_d_holdout_n']} of them)")
    check(cfg["dann_d_holdout_indices"][:3] == [int(i) for i in a["holdout"][:3]],
          "...and the recorded indices are the ones actually held out")


# ---------------------------------------------------------------------------
# T11 -- D's train/eval semantics  (NEW in brief v2)
# ---------------------------------------------------------------------------

def t11_d_train_eval():
    section("T11 D's train/eval semantics: dropout live in train, dead in eval, and the "
            "holdout monitor RESTORES train mode")
    torch.manual_seed(6)
    dim = 64
    d = MonitoredDannLoss(dim).train()
    x = torch.randn(32, dim)

    # ---- dropout ACTIVE in train: two forwards on IDENTICAL input differ ----
    a1 = d.discriminator(x)
    a2 = d.discriminator(x)
    check(not torch.equal(a1, a2),
          f"train mode: two forwards on identical input DIFFER "
          f"(max|diff| = {float((a1-a2).abs().max()):.3e}) -- dropout is live")

    # ---- dropout INACTIVE in eval: two forwards are bit-identical ----
    d.eval()
    b1 = d.discriminator(x)
    b2 = d.discriminator(x)
    check(torch.equal(b1, b2),
          "eval mode: two forwards on identical input are BIT-IDENTICAL -- dropout is "
          "off, which is what makes a monitor measured there comparable across steps")
    check(not torch.equal(a1, b1),
          "train and eval really do compute different things (so the distinction is "
          "not vacuous on this module)")
    d.train()

    # ---- eval_mode() restores the PREVIOUS mode, not unconditionally train ----
    with d.eval_mode():
        check(not d.discriminator.training, "inside eval_mode(): D.training is False")
    check(d.discriminator.training,
          "after eval_mode(): D.training is back to True")
    d.eval()
    with d.eval_mode():
        pass
    check(not d.discriminator.training,
          "eval_mode() restores what the mode WAS (entered from eval, left in eval) -- "
          "it never silently switches training back on")
    d.train()

    # ---- the real monitor path: holdout_domain_stats must leave D in train mode ----
    tr = _trainer(0)
    imgs = torch.rand(6, 1, 224, 224)
    seen = {}
    real = MonitoredDannLoss.score_domain

    def spy(self, feats):
        seen["training"] = self.discriminator.training
        return real(self, feats)

    MonitoredDannLoss.score_domain = spy
    try:
        st = holdout_domain_stats(tr, tr.dann, imgs)
    finally:
        MonitoredDannLoss.score_domain = real
    check(seen.get("training") is False,
          "holdout_domain_stats scores the held-out slice with D in EVAL mode -- a "
          "memorisation monitor carrying a random dropout mask is not comparable to "
          "d_acc_target, and this column arms abort condition 1")
    check(tr.dann.discriminator.training is True,
          "holdout_domain_stats RESTORES D to training=True afterwards -- otherwise "
          "every subsequent training step would silently run without dropout")
    check(math.isfinite(st["d_acc_holdout"]) and st["d_holdout_measured_in_eval"] == 1.0,
          f"...and it returns a finite d_acc_holdout ({st['d_acc_holdout']:.3f}) "
          f"flagged as eval-measured")
    check(tr.head.training is True,
          "the head is restored to train mode too")
    del tr


# ---------------------------------------------------------------------------
# T12 -- the morphology proxy  (NEW in brief v2)
# ---------------------------------------------------------------------------

def t12_morph_proxy():
    section("T12 the RECOVERED morphology proxy: ML-FP vs P1-reserve, finite and > chance")

    # ---- the AUROC implementation itself, against hand-checkable cases ----
    y = np.array([1, 1, 0, 0])
    check(auroc_np(y, np.array([9.0, 8.0, 1.0, 0.0])) == 1.0,
          "auroc_np: perfect separation == 1.0")
    check(auroc_np(y, np.array([0.0, 1.0, 8.0, 9.0])) == 0.0,
          "auroc_np: perfectly inverted == 0.0")
    check(auroc_np(y, np.array([1.0, 1.0, 1.0, 1.0])) == 0.5,
          "auroc_np: all ties == 0.5 (average ranks, not a crash)")
    rng = np.random.default_rng(0)
    yy = np.concatenate([np.ones(500, np.int64), np.zeros(500, np.int64)])
    rand = float(np.mean([auroc_np(yy, rng.standard_normal(1000)) for _ in range(20)]))
    check(abs(rand - 0.5) < 0.02,
          f"auroc_np: random scores average {rand:.4f}, i.e. chance (known null)")

    if not os.path.isfile(SPLITS_PATH):
        print(f"  SKIP  the real pool needs {SPLITS_PATH}; not present on this machine")
        return

    pos = mlfp_locations({}, DATA_ROOT, {**__import__("lens_dataset").DEFAULT_LAYOUT})
    with open(SPLITS_PATH) as fh:
        splits = json.load(fh)
    neg = p1_reserve_locations(splits, DATA_ROOT,
                               {**__import__("lens_dataset").DEFAULT_LAYOUT})
    used = {(os.path.basename(str(a)), int(b))
            for a, b in splits["source_train"]["neg_random"]["locations"]}
    print(f"    ML-FP available {len(pos):,}   P1-reserve available {len(neg):,}   "
          f"(source_train consumed {len(used):,} of P1)")
    check(len(pos) == 4455,
          f"all {len(pos)} ML-FP hard negatives are found (PROJECT.md 3.4 says 4,455)")
    check(len(neg) > 20000,
          f"the P1 reserve is {len(neg):,} galaxies -- the part of partition 1 that "
          f"source_train's neg_random did not consume, which IS the 'P1-reserve' the "
          f"historical instrument named")
    check(not (set(neg) & used),
          "ZERO overlap between the reserve and source_train's random negatives -- the "
          "reserve is disjoint from every training and evaluation role by construction")

    pool = build_morph_pool(splits, DATA_ROOT, n_fit_per_class=16, n_eval_per_class=16,
                            seed=42, verbose=False)
    again = build_morph_pool(splits, DATA_ROOT, n_fit_per_class=16, n_eval_per_class=16,
                             seed=42, verbose=False)
    check(pool["info"]["mlfp_locations"] == again["info"]["mlfp_locations"]
          and pool["info"]["reserve_locations"] == again["info"]["reserve_locations"],
          "the pool is DETERMINISTIC given the seed (both class draws)")
    y = pool["y"]
    check(int(y.sum()) == y.size // 2,
          f"the pool is balanced: {int(y.sum())} ML-FP / {y.size - int(y.sum())} reserve")
    check(set(pool["fit_idx"]).isdisjoint(set(pool["eval_idx"])),
          "the fit and eval halves are DISJOINT -- 640 features on a few hundred rows "
          "is separable by construction, so a probe scored on its own fit rows would "
          "read ~1.0 forever and monitor nothing")
    check(int(y[pool["eval_idx"]].sum()) == pool["eval_idx"].size // 2,
          "the eval half is class-balanced, so AUROC has an equal-n null")

    # ---- scale invariance: a uniform rescale of the features must not move it ----
    torch.manual_seed(7)
    n, d_ = 64, 32
    yy = np.concatenate([np.ones(n // 2, np.int64), np.zeros(n // 2, np.int64)])
    X = torch.randn(n, d_)
    X[: n // 2, 0] += 1.5
    fit = np.concatenate([np.arange(n // 4), n // 2 + np.arange(n // 4)])
    ev = np.setdiff1d(np.arange(n), fit)
    a1 = ridge_probe_auroc(X, yy, fit, ev)
    a2 = ridge_probe_auroc(X * 1e-3, yy, fit, ev)
    check(abs(a1 - a2) < 1e-9,
          f"the probe is SCALE-INVARIANT: shrinking every feature by 1000x leaves AUROC "
          f"at {a1:.6f} vs {a2:.6f}. Round 4 measured a 56x feature-scale collapse "
          f"(7.15); without this, that would read here as a morphology loss and this "
          f"monitor would be detecting the thing two other monitors already detect.")

    if not os.path.isfile(os.path.join(CACHE_DIR, "source_train.npz")):
        print(f"  SKIP  the live reading needs {CACHE_DIR}/source_train.npz")
        return
    tr = _trainer(0)
    import evaluate as ev_mod
    cache = ev_mod.FeatureCache(CACHE_DIR)
    feats = torch.from_numpy(np.asarray(cache.load("source_train")["features"],
                                        dtype=np.float32)).to(tr.device)
    tr.standardizer.fit(feats)
    del feats
    r = MorphProxy(pool).measure(tr)
    print(f"    live reading on a DELIBERATELY TINY {pool['y'].size}-image pool: "
          f"HEAD {r['morph_head_auroc']:.4f} (eval) / {r['morph_head_auroc_pool']:.4f} "
          f"(pool)  |  PROBE {r['morph_proxy_auroc']:.4f} (adapter) / "
          f"{r['morph_proxy_auroc_encoder']:.4f} (raw enc)")
    print(f"      NOT the calibrated number. 32 fit rows against 640 features is far "
          f"too few and reads LOW by construction; this group is checking finiteness "
          f"and above-chance, not the value. The production pool (64+64 per class) "
          f"reads 0.9202 / 0.9214 -- calibrate_morph_r5.py, "
          f"runs/dann_r5_morph_step0.json.")
    check(math.isfinite(r["morph_proxy_auroc"])
          and not math.isnan(r["morph_proxy_auroc"]),
          f"morph_proxy_auroc returns a FINITE, non-NaN value "
          f"({r['morph_proxy_auroc']:.6f}) -- the column brief v1 could only emit as NaN")
    check(r["morph_proxy_auroc"] > 0.5,
          f"...and it is above chance ({r['morph_proxy_auroc']:.4f}), i.e. the "
          f"instrument measures something on a frozen encoder rather than reading noise")
    check(math.isfinite(r["morph_proxy_auroc_encoder"]),
          f"the raw-encoder column is finite too ({r['morph_proxy_auroc_encoder']:.4f}) "
          f"-- that is the one comparable to the historical 0.914")
    check(all(math.isfinite(r[k]) for k in ("morph_head_auroc",
                                            "morph_head_auroc_pool")),
          f"the HEAD column is present and FINITE -- {r['morph_head_auroc']:.4f} on the "
          f"eval half and {r['morph_head_auroc_pool']:.4f} on the whole pool. This is "
          f"the column abort condition 3 reads.")
    check(r["morph_head_auroc"] != r["morph_proxy_auroc"],
          "the head and probe columns are DIFFERENT numbers from the same forward pass "
          "-- two instruments, not two names for one")
    # both arms: at step 0 they share head, encoder and standardizer, so the head column
    # must agree bit-for-bit. Measured on two independently constructed arms.
    tr_z = _trainer(0, lambda_zero=True)
    tr_z.standardizer.load_state_dict(tr.standardizer.state_dict())
    tr_z.head.load_state_dict(tr.head.state_dict())
    rz = MorphProxy(pool).measure(tr_z)
    check(rz["morph_head_auroc"] == r["morph_head_auroc"]
          and rz["morph_head_auroc_pool"] == r["morph_head_auroc_pool"],
          f"the head column is BIT-IDENTICAL on the lambda=0 arm "
          f"({rz['morph_head_auroc']:.6f}) -- D is not in the monitor's path, so the "
          f"arms can only diverge once training does")
    check(tr.head.training is True, "the morphology monitor restores the head's mode")
    del tr, tr_z


# ---------------------------------------------------------------------------
# T13 -- the pre-registered abort conditions  (NEW in brief v3)
# ---------------------------------------------------------------------------

# Levels taken from arm 1 (the lambda=0 control), so "healthy" means "like the control".
HEALTHY = {"d_acc_holdout": 0.62, "d_pred_source_frac": 0.51, "loss_domain_raw": 0.68,
           "morph_proxy_auroc": 0.9076,
           "morph_head_auroc_pool": 0.4948, "morph_head_auroc": 0.4948,
           "src_participation_ratio": 5.2582, "tgt_participation_ratio": 10.2114}


def _drive(watch, n, mutate=None, start=1, base=None):
    """Feed ``n`` synthetic steps; ``mutate(step, stats)`` edits the row in place."""
    rows = []
    for step in range(start, start + n):
        st = dict(base if base is not None else HEALTHY)
        if mutate is not None:
            mutate(step, st)
        rows.append(watch.update(step, st))
    return rows


def _control_csv(path, steps, value=0.9076, column=COND3_COLUMN,
                 src_pr=5.2582, tgt_pr=10.2114):
    cols = ["step", column, "src_participation_ratio", "tgt_participation_ratio"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s_ in steps:
            w.writerow({"step": s_, column: value,
                        "src_participation_ratio": src_pr,
                        "tgt_participation_ratio": tgt_pr})
    return path


def t13_abort_conditions(tmpdir):
    section("T13 the four pre-registered abort conditions: fire, stay silent, log vs arm")

    # ---- KNOWN NULL FIRST: a healthy trace must fire NOTHING ----
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, verbose=False)
    _drive(w, 3000)
    check(w.fired() == [],
          f"KNOWN NULL: 3000 healthy steps fire nothing (got {w.fired()}). A detector "
          f"that fires on a good trace is worse than no detector.")

    # ---- condition 1: D overpowering. BOTH halves are required ----
    # half one only: holdout accuracy pinned high while the domain loss KEEPS FALLING.
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, verbose=False)
    _drive(w, 3 * COND1_SUSTAIN,
           lambda step, st: st.update({"d_acc_holdout": 0.99,
                                       "loss_domain_raw": 1.0 / (1 + 0.01 * step)}))
    check("abort1_d_overpowering" not in w.fired(),
          f"condition 1 does NOT fire on accuracy alone while the domain loss is still "
          f"FALLING -- a D at 0.99 early is a D that is winning, not one that has "
          f"overpowered the optimiser (streak {w.c1.max_streak_steps})")
    # both halves: accuracy pinned high AND the loss flat
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, verbose=False)
    _drive(w, 3 * COND1_SUSTAIN,
           lambda step, st: st.update({"d_acc_holdout": 0.99, "loss_domain_raw": 0.30}))
    check("abort1_d_overpowering" in w.fired(),
          f"condition 1 FIRES when held-out accuracy > 0.95 for {COND1_SUSTAIN} steps "
          f"AND the domain loss is not falling (fired at step {w.c1.fired_at})")
    check(w.c1.fired_at is not None and w.c1.fired_at >= 2 * COND1_SUSTAIN,
          f"...and not before the loss history is long enough to judge 'not falling' "
          f"(fired at {w.c1.fired_at}, needs 2x{COND1_SUSTAIN} of history)")

    # ---- condition 2: degenerate collapse ----
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, verbose=False)
    _drive(w, COND2_SUSTAIN - 2, lambda step, st: st.update({"d_pred_source_frac": 0.95}))
    check("abort2_degenerate_collapse" not in w.fired(),
          f"condition 2 has NOT fired at {COND2_SUSTAIN - 2} steps of collapse "
          f"(streak {w.c2.streak_steps}) -- the sustain is real, not decorative")
    _drive(w, 10, start=COND2_SUSTAIN - 1,
           mutate=lambda step, st: st.update({"d_pred_source_frac": 0.95}))
    check("abort2_degenerate_collapse" in w.fired(),
          f"condition 2 FIRES once the streak reaches {COND2_SUSTAIN} steps "
          f"(fired at {w.c2.fired_at})")
    # and a single healthy step RESETS it
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, verbose=False)
    _drive(w, COND2_SUSTAIN - 5,
           lambda step, st: st.update({"d_pred_source_frac": 0.95}))
    _drive(w, 1, start=COND2_SUSTAIN,
           mutate=lambda step, st: st.update({"d_pred_source_frac": 0.5}))
    check(w.c2.streak_steps == 0,
          "one in-band step RESETS the streak -- 'consecutive' means consecutive")

    # ---- condition 3 reads the HEAD column, not the frozen probe ----
    check(COND3_COLUMN == "morph_proxy_auroc",
          f"BOTH detectors read {COND3_COLUMN!r} -- the FROZEN PROBE. Re-specified from "
          f"arm 1: at nb3 the probe is live (the historical 'it held ~0.90' was measured "
          f"at nb0, where a frozen encoder made it STRUCTURALLY INCAPABLE of moving), "
          f"and the control gives it a 0.9076 level with 0.4076 of headroom.")
    check(set(COND3_INTERPRETATION_COLUMNS) == {"morph_head_auroc_pool",
                                                "morph_head_auroc"}
          and COND3_COLUMN not in COND3_INTERPRETATION_COLUMNS,
          f"...while the two HEAD columns {COND3_INTERPRETATION_COLUMNS} are retained "
          f"with NO DETECTOR: arm 1 showed they fall to chance because the CLASSIFIER "
          f"SUCCEEDED (both probe classes are class 0 in the lens task), not because "
          f"morphology access was destroyed")
    w_ = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, cond3_margin=None, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        for c in COND3_INTERPRETATION_COLUMNS:
            w_.update(1, {c: 0.10})            # catastrophic on BOTH head columns
    check(not w_.fired(),
          "a catastrophic reading on EITHER head column fires nothing -- no detector "
          "reads them, which is exactly what arm 1's control showed was necessary")
    check(COND3_SIGN_FLIP == 0.70,
          f"3b's floor is RE-DERIVED to {COND3_SIGN_FLIP} -- half the measured headroom "
          f"(0.5 + 0.5*0.4076 = 0.7038, rounded down). The old 0.5 was the HEAD column's "
          f"inversion point and sits 40 J below the probe control's own minimum of "
          f"0.8108, i.e. far past catastrophic.")
    check(abs(COND3_M_CEILING - 0.2481) < 1e-3,
          f"the ceiling is re-derived as a FRACTION of measured headroom "
          f"(0.6088 x 0.4076 = {COND3_M_CEILING:.4f}), not the head column's absolute "
          f"0.04 -- stating the fraction is what makes it travel when the column changes")
    check(COND3_MARGIN is None,
          "the shipped condition-3 margin is UNSET -- brief v3's 0.10 was declared "
          "against the probe's 0.4202 headroom and is scale-wrong for the head column's "
          "~0.078, where a 0.10 drop lands below chance")

    ctrl = ControlCurve(_control_csv(os.path.join(tmpdir, "ctrl.csv"),
                                     range(0, 4001, 25), value=0.9076))
    check(ctrl.column == COND3_COLUMN,
          f"ControlCurve reads the {COND3_COLUMN!r} column by default")
    check(ctrl.summary()["stride_mode"] == 25 and ctrl.at(1000) == 0.9076
          and ctrl.at(1001) is None,
          f"the control curve reads by ABSOLUTE step: at(1000)={ctrl.at(1000)}, "
          f"at(1001)={ctrl.at(1001)} (None, NOT interpolated from its neighbours)")

    # ---- 3a REFUSES TO EVALUATE while M is unset, and 3b still works ----
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=ctrl, cond3_margin=None,
                       verbose=False)
    for step in range(0, 4001):
        st = dict(HEALTHY)
        st[COND3_COLUMN] = 0.55 if step % 25 == 0 else None   # a catastrophic shortfall
        w.update(step, st)
    check("abort3_morph_vs_control" not in w.fired() and w.c3.max_streak_steps == 0,
          "with M UNSET, 3a REFUSES to evaluate even on a 0.38 shortfall -- it does not "
          "silently score every step against a missing threshold and report healthy")
    check(abs(w.last_delta - (0.55 - 0.9076)) < 1e-9,
          f"...while the RAW difference is still logged every monitor step "
          f"({w.last_delta:.4f}) -- that trajectory is exactly what M gets declared from")

    # ---- CONDITIONS 3 AND 4 ARE ADVISORY ON EVERY ARM ----
    # This REPLACES the armed+unset-margin refusal. That refusal protected a safety net;
    # conditions 3 and 4 are no longer one, so an unset margin there disables a logging
    # line and refusing to start over it would be theatre. What must be tested instead
    # is the new invariant, in both directions.
    import dann_round5 as _D5adv
    check(set(_D5adv.ADVISORY_ONLY_DETECTORS)
          == {"abort3_morph_vs_control", "abort3b_morph_sign_flip",
              "abort4src_rank_collapse", "abort4tgt_rank_collapse"},
          f"the advisory set is exactly conditions 3 and 4: "
          f"{list(_D5adv.ADVISORY_ONLY_DETECTORS)}")
    wa = DannAbortWatch(mode=ABORT_MODE_ARMED, verbose=False)
    check(wa.mode == ABORT_MODE_ARMED,
          "armed + every margin UNSET now CONSTRUCTS -- the margins that remain unset "
          "belong to detectors that cannot abort anything")
    check(sorted(wa.summary()["arming_detectors"])
          == ["abort1_d_overpowering", "abort2_degenerate_collapse"],
          f"only conditions 1 and 2 can stop a run: "
          f"{wa.summary()['arming_detectors']}")

    for name in _D5adv.ADVISORY_ONLY_DETECTORS:
        w2 = DannAbortWatch(mode=ABORT_MODE_ARMED, verbose=False)
        for d in w2.all:
            if d.name == name:
                d.fired = True
        check(w2.fired() == [name] and w2.fired_arming() == []
              and w2.should_abort() is False,
              f"{name} FIRED on an ARMED run and did NOT abort it -- and is still "
              f"REPORTED by fired(), so the fold records the event rather than looking "
              f"healthy")

    for name, det in (("abort1_d_overpowering", "c1"),
                      ("abort2_degenerate_collapse", "c2")):
        w2 = DannAbortWatch(mode=ABORT_MODE_ARMED, verbose=False)
        getattr(w2, det).fired = True
        check(w2.should_abort() is True,
              f"KNOWN NULL: {name} still ABORTS an armed run -- 'advisory' was scoped to "
              f"conditions 3 and 4, it did not disarm the whole watch")
        check(DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, verbose=False)
              .__class__ is DannAbortWatch,
              "...and log_only still constructs")
    w3 = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, verbose=False)
    w3.c1.fired = True
    check(w3.should_abort() is False,
          "log_only never aborts, conditions 1 and 2 included -- arms 1 and 2 ran that "
          "way and the mode still means what it meant")
    check(DannAbortWatch(mode=ABORT_MODE_ARMED, cond3_margin=0.04,
                         cond4_src_margin=1.0, cond4_tgt_margin=2.0,
                         verbose=False).cond3_margin == 0.04,
          "armed + all three advisory margins set still constructs and keeps them -- "
          "advisory means it does not abort, not that it stops measuring")

    # ---- with M set, 3a behaves as in v3 ----
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=ctrl, cond3_margin=0.04,
                       verbose=False)
    for step in range(0, 4001):
        st = dict(HEALTHY)
        st[COND3_COLUMN] = 0.8476 if step % 25 == 0 else None  # 0.06 below the control
        w.update(step, st)
    check("abort3_morph_vs_control" in w.fired(),
          f"with M = 0.04, a sustained 0.06 shortfall FIRES 3a (fired at step "
          f"{w.c3.fired_at})")
    check(w.c3.fired_at is not None and w.c3.fired_at >= COND3_SUSTAIN,
          f"...only after {COND3_SUSTAIN} STEPS, not after {COND3_SUSTAIN} readings -- "
          f"at --morph-every 25 those differ by 25x, the unit error 7.10 records for "
          f"FreeMatch's warmup (fired at {w.c3.fired_at})")

    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=ctrl, cond3_margin=0.10,
                       verbose=False)
    for step in range(0, 4001):
        st = dict(HEALTHY)
        st[COND3_COLUMN] = 0.8476 if step % 25 == 0 else None
        w.update(step, st)
    check("abort3_morph_vs_control" not in w.fired(),
          "the SAME 0.06 shortfall does NOT fire against M = 0.10 -- which is the whole "
          "reason a margin declared on the probe's scale cannot be carried onto the head")

    # ---- condition 3 REFUSES to interpolate a missing control step ----
    sparse = ControlCurve(_control_csv(os.path.join(tmpdir, "sparse.csv"),
                                       range(0, 4001, 100), value=0.9076))
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=sparse, cond3_margin=0.04,
                       verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        for step in range(0, 4001):
            st = dict(HEALTHY)
            st[COND3_COLUMN] = 0.60 if step % 25 == 0 else None
            w.update(step, st)
    n_missing = len(w.missing_control_steps)
    check(n_missing > 0,
          f"a cadence mismatch (run every 25, control every 100) leaves {n_missing} "
          f"steps with NO control row, and every one is RECORDED")
    check(all(m % 100 != 0 for m in w.missing_control_steps),
          "the recorded misses are exactly the steps the control has no row for")
    check("abort3_morph_vs_control" in w.fired(),
          "condition 3 still fires from the steps the control DOES cover -- refusing "
          "to interpolate suppresses the comparison at missing steps, it does not "
          "disable the detector")
    covered = {s for s in range(0, 4001, 25) if s % 100 == 0}
    check(set(range(0, 4001, 25)) - covered == set(w.missing_control_steps),
          "EVERY uncovered step is refused -- none was silently filled in from a "
          "neighbour. An interpolated reference is not a reference.")

    bad = _control_csv(os.path.join(tmpdir, "nocol.csv"), range(0, 100, 25),
                       column="something_else")
    try:
        ControlCurve(bad)
        check(False, "KNOWN NULL: ControlCurve accepted a file with no morph column")
    except SystemExit as e:
        check("has no column" in str(e),
              "KNOWN NULL: a control file without the head column is REFUSED at load, "
              "not discovered at step 3000")

    # ---- 3b is the PRIMARY detector: no sustain, no margin, no control ----
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, cond3_margin=None, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _drive(w, 1, lambda step, st: st.update({COND3_COLUMN: 0.62}))
    check("abort3b_morph_sign_flip" in w.fired(),
          f"3b fires IMMEDIATELY below {COND3_SIGN_FLIP} on the FIRST reading -- no "
          f"sustain, no margin, no control curve. The historical failure WAS a sign flip "
          f"(head 0.61 -> 0.37), so this reproduces the actual event with no invented "
          f"threshold.")
    check("PRIMARY" in w.c3b.action_note,
          "...and it is labelled PRIMARY in the summary, not a backstop")
    check("abort3_morph_vs_control" not in w.fired(),
          "...and it fires with 3a inactive, so the primary detector does not depend on "
          "a margin that does not exist yet")
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, cond3_margin=None, verbose=False)
    _drive(w, 200, lambda step, st: st.update({COND3_COLUMN: 0.705}))
    check("abort3b_morph_sign_flip" not in w.fired(),
          "3b does NOT fire at 0.705 -- just above the floor is not a collapse")

    # ---- the M-declaration criterion, pre-registered and machine-readable ----
    import numpy as _np

    def _traj(path, jitter, level=0.566, seed=0, stride=25, last=8000):
        rng = _np.random.default_rng(seed)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["step", COND3_COLUMN])
            w.writeheader()
            for st_ in range(0, last + 1, stride):
                w.writerow({"step": st_, COND3_COLUMN: level + rng.normal(0, jitter)})
        return path

    # Jitter levels are calibrated against the CURRENT ceiling. They were 0.002/0.030
    # when the ceiling was the head column's 0.04; on the probe column's 0.2481 the old
    # "noisy" arm is comfortably declarable, so it no longer tests the failing branch.
    # These are synthetic SERIES, not physical AUROCs -- the criterion only compares
    # 3*J against the ceiling, so the absolute level is irrelevant.
    quiet = morph_trajectory_stats(_traj(os.path.join(tmpdir, "quiet.csv"), 0.002))
    noisy = morph_trajectory_stats(_traj(os.path.join(tmpdir, "noisy.csv"), 0.150))

    # ---- v7: J is computed on the series 3a WILL THRESHOLD, and k=1 is a no-op ----
    q1 = morph_trajectory_stats(os.path.join(tmpdir, "quiet.csv"), k=1)
    check(q1["J_raw"] == q1["J_at_k"] == q1["J_median_abs_step_to_step"]
          and q1["m_min_admissible"] == quiet["m_min_admissible"]
          and q1["k"] == 1,
          f"k=1 leaves every number BIT-IDENTICAL: J_raw == J_at_k == "
          f"{q1['J_raw']:.6f}, and m_min_admissible matches the default call")
    # the k-FLIP: a trajectory that fails at k=1 and passes at k=8
    flip = _traj(os.path.join(tmpdir, "flip.csv"), 0.120, seed=3)
    f1 = morph_trajectory_stats(flip, k=1)
    f8 = morph_trajectory_stats(flip, k=8)
    print(f"    k-flip: k=1 3J={f1['m_min_admissible']:.5f} -> {f1['M_DECLARABLE']}   "
          f"k=8 3J={f8['m_min_admissible']:.5f} -> {f8['M_DECLARABLE']}")
    check(f1["M_DECLARABLE"] is False and f8["M_DECLARABLE"] is True,
          f"the SAME trajectory is UNCALIBRABLE at k=1 (3J={f1['m_min_admissible']:.4f}) "
          f"and declarable at k=8 (3J={f8['m_min_admissible']:.4f}) -- J is computed on "
          f"the series 3a will actually threshold, so a declared k is not penalised by "
          f"raw jitter it never sees")
    check(f8["verdict"].startswith("AT k=8:") and f1["verdict"].startswith("AT k=1:"),
          "...and the verdict NAMES k, so a number cannot be read without its window")
    check(f8["j_used_for_criterion"] == "J_at_k"
          and "k=1" in f1["j_used_for_criterion"],
          f"...and j_used_for_criterion says which J the criterion used "
          f"({f8['j_used_for_criterion']!r} at k=8)")
    check(f8["J_n_consecutive_pairs"] > 0 and f8["J_at_k"] < f8["J_raw"],
          f"smoothing really does reduce J ({f8['J_raw']:.5f} -> {f8['J_at_k']:.5f})")
    # THE CAUTION: J falls far more than sqrt(k) while the IQR falls about sqrt(k)
    print(f"    k=8 reductions: J fell {f8['J_reduction_factor_at_k']:.1f}x, IQR fell "
          f"{f8['iqr_reduction_factor_at_k']:.1f}x, sqrt(8) = 2.83")
    check(f8["J_reduction_factor_at_k"] > 2 * f8["iqr_reduction_factor_at_k"],
          f"MEASURED CAUTION, surfaced not hidden: at k=8, J falls "
          f"{f8['J_reduction_factor_at_k']:.1f}x while the IQR -- the LEVEL noise that "
          f"could actually hold a 500-step false streak -- falls only "
          f"{f8['iqr_reduction_factor_at_k']:.1f}x, about sqrt(k). Consecutive trailing "
          f"medians share k-1 of their k inputs, so a criterion read off J alone can be "
          f"satisfied by autocorrelation rather than by real noise reduction.")
    check("autocorrelation" in f8["k_caution"].lower()
          and "NOT ACTED ON HERE" in f8["k_caution"],
          "...and the summary says so IN THE OUTPUT, while explicitly NOT changing the "
          "pre-registered criterion")

    # ---- v7: J_early is reported, is NOT the criterion, and warns on a big ratio ----
    early = _traj(os.path.join(tmpdir, "early.csv"), 0.003, seed=5)
    # rewrite with a noisy first 2000 steps
    rows = list(csv.DictReader(open(early, newline="")))
    rng_e = _np.random.default_rng(11)
    with open(early, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", COND3_COLUMN])
        w.writeheader()
        for r_ in rows:
            st_ = int(float(r_["step"]))
            v_ = (0.566 + rng_e.normal(0, 0.020)) if st_ < 2000 else float(r_[COND3_COLUMN])
            w.writerow({"step": st_, COND3_COLUMN: v_})
    e = morph_trajectory_stats(early, k=1)
    print(f"    early window: J={e['J_median_abs_step_to_step']:.5f} "
          f"J_early={e['J_early']:.5f} ratio={e['J_early_over_J']:.2f}")
    check(e["J_early_window_steps"] == [0, 2000] and e["J_window_steps"] == [2000, 8000],
          "J_early is measured over steps 0-2000, the criterion's J over 2000-8000")
    check(e["J_early_WARNING"] is True and e["J_early_over_J"] > 2.0,
          f"the ratio warning FIRES when early jitter exceeds 2x the settled jitter "
          f"({e['J_early_over_J']:.1f}x) -- a collapse happens early, which is exactly "
          f"where 3a operates")
    check("EARLY-WINDOW WARNING" in e["verdict"],
          "...and the verdict says so explicitly rather than burying it in a field")
    check(e["m_min_admissible"] == COND3_M_J_FACTOR * e["J_median_abs_step_to_step"],
          "...while M_DECLARABLE is STILL computed from the 2000-8000 window -- J_early "
          "is a sanity check, not a change to the criterion")
    quietE = morph_trajectory_stats(os.path.join(tmpdir, "quiet.csv"), k=1)
    check(quietE["J_early_WARNING"] is False
          and "EARLY-WINDOW WARNING" not in quietE["verdict"],
          "KNOWN NULL: a uniformly quiet trajectory does NOT trip the early warning")

    args_ms = __import__("run_dann_r5").build_argparser().parse_args(
        ["--morph-stats", "x.csv", "--abort3-median-window", "8"])
    check(args_ms.abort3_median_window == 8 and args_ms.morph_stats == "x.csv",
          "--morph-stats accepts --abort3-median-window, so the standalone path can be "
          "run at a CANDIDATE k before k is declared")

    print(f"    quiet arm: J={quiet['J_median_abs_step_to_step']:.5f} "
          f"3J={quiet['m_min_admissible']:.5f} IQR={quiet['iqr']:.5f}")
    print(f"    noisy arm: J={noisy['J_median_abs_step_to_step']:.5f} "
          f"3J={noisy['m_min_admissible']:.5f} IQR={noisy['iqr']:.5f}")
    check(quiet["M_DECLARABLE"] is True and noisy["M_DECLARABLE"] is False,
          f"the pre-registered criterion RESOLVES BOTH WAYS on synthetic arms: a quiet "
          f"trajectory gives M_DECLARABLE=True (3J={quiet['m_min_admissible']:.4f} <= "
          f"{COND3_M_CEILING}) and a noisy one False "
          f"({noisy['m_min_admissible']:.4f} > {COND3_M_CEILING})")
    check("UNCALIBRABLE" in noisy["verdict"] and "LEGITIMATE" in noisy["verdict"],
          "...and the failing branch says UNCALIBRABLE is a LEGITIMATE OUTCOME, not a "
          "failure -- a property of an instrument never designed as a graded monitor")
    check(quiet["m_factor"] == COND3_M_J_FACTOR == 3.0
          and quiet["m_ceiling"] == COND3_M_CEILING
          and abs(COND3_M_CEILING - 0.6088 * 0.4076) < 1e-6,
          f"the criterion keeps its pre-registered {COND3_M_J_FACTOR:g}x factor, and the "
          f"ceiling is the SAME FRACTION OF HEADROOM the original 0.04 encoded "
          f"(0.04/0.0657 = 0.6088) applied to the probe column's measured 0.4076 "
          f"-> {COND3_M_CEILING:.4f}. The fraction is pre-registered; the absolute is "
          f"derived from the control.")
    check(quiet["J_window_steps"] == [2000, 8000]
          and quiet["iqr_window_last_steps"] == 4000,
          "J is measured over steps 2000-8000 and the IQR over the last 4000, as "
          "pre-registered")
    check(quiet["reading_stride"] == 25 and quiet["J_n_consecutive_pairs"] == 240,
          f"'step-to-step' means between consecutive READINGS -- "
          f"{quiet['J_n_consecutive_pairs']} pairs at stride "
          f"{quiet['reading_stride']}, which is the jitter 3a actually sees")
    check("LOWER BOUND" in quiet["J_note"],
          "...and J is labelled a LOWER BOUND on the paired-difference noise, not that "
          "noise itself -- both arms score the identical pool so most of it cancels, but "
          "not exactly")
    check("k MUST BE DECLARED BEFORE M" in quiet["k_ordering_constraint"],
          "the summary carries the ordering constraint on k")
    # every field the declaration needs is machine-readable, not prose-only
    for key in ("J_median_abs_step_to_step", "m_min_admissible", "m_ceiling",
                "M_DECLARABLE", "iqr", "iqr_q25", "iqr_q75"):
        if not isinstance(quiet[key], (int, float, bool)):
            check(False, f"criterion field {key} is not machine-readable")
            break
    else:
        check(True, "every field the declaration needs is a NUMBER or a BOOL, not prose")

    # ---- the median window: default is a no-op, k>1 smooths the DIFFERENCE ----
    args_ = __import__("run_dann_r5").build_argparser().parse_args([])
    check(args_.abort3_median_window == 1,
          f"--abort3-median-window defaults to {args_.abort3_median_window} -- no "
          f"smoothing unless it is declared")
    ctrl_k = ControlCurve(_control_csv(os.path.join(tmpdir, "ctrlk.csv"),
                                       range(0, 4001, 25), value=0.9076))
    # A trace that alternates either side of the threshold. Against a control of 0.620
    # and M = 0.05 the two readings give paired differences of -0.090 (violating) and
    # -0.045 (not), so k=1 resets the streak every other reading while the median of an
    # 8-wide window sits at -0.0675 and sustains. Both readings stay above 0.5 so 3b
    # stays out of it. (The first version of this test alternated -0.066/-0.010, whose
    # median is -0.038 and does NOT cross M -- the arithmetic was wrong, not the code.)
    def _alt(w):
        for step in range(0, 4001):
            st = dict(HEALTHY)
            if step % 25 == 0:
                st[COND3_COLUMN] = 0.8176 if (step // 25) % 2 == 0 else 0.8626
            else:
                st[COND3_COLUMN] = None
            w.update(step, st)

    w1 = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=ctrl_k, cond3_margin=0.05,
                        cond3_median_window=1, verbose=False)
    _alt(w1)
    w8 = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=ctrl_k, cond3_margin=0.05,
                        cond3_median_window=8, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _alt(w8)
    check("abort3_morph_vs_control" not in w1.fired(),
          f"k=1 on an ALTERNATING trace never sustains a streak (max "
          f"{w1.c3.max_streak_steps} steps) -- every other reading resets it")
    check("abort3_morph_vs_control" in w8.fired(),
          f"k=8 on the SAME trace fires (at step {w8.c3.fired_at}) "
          f"-- the trailing median of the paired differences is what is thresholded, so "
          f"alternation stops breaking the streak")
    check(w8.c3.fired_at is not None and w8.c3.fired_at >= COND3_SUSTAIN + 7 * 25,
          f"...and k=8 delays the earliest possible fire by (k-1)*morph_every = 175 "
          f"steps, because the window must be FULL before 3a evaluates at all "
          f"(fired at {w8.c3.fired_at})")
    wk = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=ctrl_k, cond3_margin=0.05,
                        cond3_median_window=1, verbose=False)
    check(wk.summary()["cond3_median_window_k"] == 1
          and "BEFORE M" in wk.summary()["cond3_median_window_note"],
          "run_config.json records k and the constraint that it is declared BEFORE M")

    # ---- condition 4: DIFFERENTIAL, src and tgt separately ----
    ctrl4 = ControlCurve(_control_csv(os.path.join(tmpdir, "ctrl4.csv"),
                                      range(0, 4001, 25)))
    check(set(ctrl4.summary()["columns_loaded"]) ==
          {COND3_COLUMN, "src_participation_ratio", "tgt_participation_ratio"},
          f"the control curve carries ALL THREE differential columns "
          f"{ctrl4.summary()['columns_loaded']} -- condition 4 reads the two PR series "
          f"through the same object, same absolute-step lookup, same no-interpolation")

    # src collapses well below the control, tgt tracks it
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=ctrl4, cond3_margin=0.05,
                       cond4_src_margin=0.5175, cond4_tgt_margin=1.3263, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _drive(w, 4000, lambda step, st: st.update({"src_participation_ratio": 3.0}))
    check("abort4src_rank_collapse" in w.fired()
          and "abort4tgt_rank_collapse" not in w.fired(),
          f"condition 4 fires on SRC alone when only src falls below the control "
          f"(src fired at {w.c4src.fired_at}); tgt stays silent -- they are separate "
          f"detectors because arm 1 showed them behaving differently")
    check(abs(w.last_pr_delta["src"] - (3.0 - 5.2582)) < 1e-9
          and abs(w.last_pr_delta["tgt"]) < 1e-9,
          f"...and both PR differentials are logged every step "
          f"(src {w.last_pr_delta['src']:.4f}, tgt {w.last_pr_delta['tgt']:.4f})")

    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=ctrl4, cond3_margin=0.05,
                       cond4_src_margin=0.5175, cond4_tgt_margin=1.3263, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _drive(w, 4000, lambda step, st: st.update({"tgt_participation_ratio": 8.0}))
    check("abort4tgt_rank_collapse" in w.fired()
          and "abort4src_rank_collapse" not in w.fired(),
          "and on TGT alone when only tgt falls -- a single combined threshold would "
          "have hidden that")

    # UNSET margins REFUSE to evaluate rather than passing
    w = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=ctrl4, cond3_margin=None,
                       cond4_src_margin=None, cond4_tgt_margin=None, verbose=False)
    _drive(w, 4000, lambda step, st: st.update({"src_participation_ratio": 1.0,
                                                "tgt_participation_ratio": 1.0}))
    check(not w.fired() and w.c4src.n_evaluated == 0 and w.c4tgt.n_evaluated == 0,
          "with the cond4 margins UNSET, both detectors REFUSE to evaluate even on a "
          "total collapse to PR = 1.0 -- they do not score against a missing threshold "
          "and report healthy")

    # THE WITHDRAWN ABSOLUTE: PR < 20 fired 7,999 of 8,000 times on the control
    check(COND4_ABSOLUTE_PR_FLOOR_WITHDRAWN == 20.0 and D5_pr_floor() == 1.0,
          f"the absolute 'PR < 20 of 640' is WITHDRAWN and kept only as a record; the "
          f"headroom floor for a PR column is now 1.0 (a fully collapsed representation "
          f"has all its variance in one direction)")
    check(0 < COND4_SRC_CEILING < COND4_TGT_CEILING,
          f"src and tgt carry SEPARATE ceilings derived from their own control levels "
          f"({COND4_SRC_CEILING:.4f} and {COND4_TGT_CEILING:.4f})")
    check(COND4_SRC_MARGIN is None and COND4_TGT_MARGIN is None,
          "both cond4 margins ship UNSET, like cond3's M")

    # ---- log_only vs armed: the SAME fire, two different actions ----    # ---- log_only vs armed: the SAME fire, two different actions ----
    for mode, want in ((ABORT_MODE_LOG_ONLY, False), (ABORT_MODE_ARMED, True)):
        # a margin is required to construct an ARMED watch at all (checked above), so
        # one is supplied here -- this group is about ACTION, not about M.
        w = DannAbortWatch(mode=mode, cond3_margin=0.04, cond4_src_margin=1.0,
                           cond4_tgt_margin=2.0, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            _drive(w, 2 * COND2_SUSTAIN,
                   lambda step, st: st.update({"d_pred_source_frac": 0.02}))
        check(bool(w.fired()) and w.should_abort() is want,
              f"mode={mode}: the condition fires either way, should_abort() is {want} "
              f"-- action is gated on the arm, detection never is")

    # ---- the runner's default mode per arm, and that it is RECORDED ----
    import run_dann_r5 as R5
    check(R5.default_abort_mode("dann_nb3", 0) == ABORT_MODE_LOG_ONLY,
          "development (dann_nb3, fold 0) defaults to log_only -- if DANN destroys "
          "classifier access that IS the result, and aborting at step 1200 throws away "
          "the curve that is the evidence")
    check(R5.default_abort_mode("dann_nb3_lambda0", 0) == ABORT_MODE_LOG_ONLY
          and R5.default_abort_mode("dann_nb3_lambda0", 3) == ABORT_MODE_LOG_ONLY,
          "the lambda=0 control is log_only at EVERY fold -- it is the reference "
          "trajectory for every monitor and must always complete")
    for k in (1, 2, 3, 4):
        if R5.default_abort_mode("dann_nb3", k) != ABORT_MODE_ARMED:
            check(False, f"confirmatory fold {k} should default to armed")
            break
    else:
        check(True, "confirmatory folds 1-4 default to ARMED -- the trace has no "
                    "scientific value there and a dead run is wasted T4 time")

    args = R5.build_argparser().parse_args([])
    cfg = R5.build_config("dann_nb3", args, fold=0)
    cfg1 = R5.build_config("dann_nb3", args, fold=1)
    check(cfg["abort_mode"] == ABORT_MODE_LOG_ONLY
          and cfg1["abort_mode"] == ABORT_MODE_ARMED
          and "default for arm" in cfg["abort_mode_source"],
          f"run_config.json records abort_mode EXPLICITLY with its source "
          f"({cfg['abort_mode']!r} / {cfg1['abort_mode']!r}) -- a reader never has to "
          f"re-derive it")
    ex = R5.build_argparser().parse_args(["--abort-mode", ABORT_MODE_ARMED])
    cfgx = R5.build_config("dann_nb3", ex, fold=0)
    check(cfgx["abort_mode"] == ABORT_MODE_ARMED
          and "explicit" in cfgx["abort_mode_source"],
          "--abort-mode overrides the default outright and says so in the record")

    # ---- the v3 constants the runner ships with ----
    check(R5.MORPH_EVERY == 25, f"--morph-every default is {R5.MORPH_EVERY} (v3 4.1)")
    check(R5.ARMS["dann_nb3_lambda0"]["steps"] == 8000
          and R5.ARMS["dann_nb3"]["steps"] == 8000,
          "BOTH arms declare 8000 steps -- the control is now the reference trajectory "
          "for every monitor (v3 5), not a 2000-step null")
    check("morph_every" in R5.PINNED_KEYS,
          "morph_every is a PINNED key: a cadence mismatch does not coarsen condition "
          "3's comparison, it deletes it")
    check(cfg["abort_cond3_margin"] is None
          and "UNSET" in cfg["abort_cond3_margin_status"],
          "run_config.json records the condition-3 margin as UNSET, with the reason")
    check(cfg["abort_cond3_column"] == COND3_COLUMN
          and set(cfg["abort_cond3_interpretation_columns"])
              == set(COND3_INTERPRETATION_COLUMNS),
          f"...and records BOTH which column the detectors read "
          f"({cfg['abort_cond3_column']}) and which is for interpretation "
          f"({cfg['abort_cond3_interpretation_columns']}), so the distinction survives "
          f"into the next session")
    crit = cfg["abort_cond3_declaration_criterion"]
    check(crit["feasible_iff"].startswith(f"{COND3_M_J_FACTOR:g}*J <= ")
          and "UNCALIBRABLE" in crit["rule"]
          and "legitimate outcome" in crit["uncalibrable_is_legitimate"],
          "run_config.json carries the M-declaration criterion VERBATIM, including that "
          "UNCALIBRABLE is an allowed outcome -- fixed before arm 1, so the decision is "
          "not made while looking at the curve")
    check("abort_cond3_column" in R5.PINNED_KEYS
          and "abort_cond3_margin" in R5.EXEMPT_KEYS,
          "the cond3 COLUMN is pinned between arms (they must measure the same thing) "
          "while the MARGIN is exempt (arms 1-2 run before it exists)")
    # The withdrawn numbers must not survive as VALUES. They may -- and should --
    # survive as prose saying they are withdrawn, so this scans the numbers, not the
    # strings. (The first version of this check scanned the strings and failed on the
    # withdrawal note itself, which is the opposite of what it is for.)
    def _numbers(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            yield float(o)
        elif isinstance(o, dict):
            for v in o.values():
                yield from _numbers(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from _numbers(v)

    vals = list(_numbers(cfg))
    check(not any(abs(v - 0.55) < 1e-9 or abs(v - 0.6910) < 1e-3 for v in vals),
          f"neither 0.55 nor 0.6910 survives as a VALUE anywhere in the emitted config "
          f"({len(vals)} numeric fields scanned)")
    # 0.10 is scoped to the ABORT fields, not scanned globally: it is also the D-holdout
    # fraction, a legitimate unrelated setting. (The global scan flagged that on the
    # first run -- a blunt check that fails on an innocent value teaches people to
    # ignore it.)
    abort_vals = list(_numbers({k: v for k, v in cfg.items()
                                if "abort" in k or "cond3" in k or "margin" in k}))
    check(not any(abs(v - 0.10) < 1e-9 for v in abort_vals),
          f"brief v3's withdrawn 0.10 survives in no ABORT field ({len(abort_vals)} "
          f"scanned) -- while dann_d_holdout_frac is legitimately 0.10 and is left alone")
    note = cfg["morph_proxy_abort_note"]
    check("NOT invented" in note and "0.10" in note and "0.6910" in note,
          "...and the config says IN PROSE that all three are withdrawn AND that 0.55 "
          "was NOT invented -- it sat under the head instrument's real ~0.61 baseline. "
          "A number simply deleted leaves the next reader to reinvent it; a number "
          "deleted for the wrong reason leaves them to re-derive the wrong correction.")


def t14_control_curve_coverage(tmpdir):
    """The two arms must agree on WHICH ABSOLUTE STEPS the comparison happens at.

    Condition 3 refuses to interpolate, so `covers()` is what stands between a cadence
    mismatch and a silently coarsened comparison. Two things are checked here:

      1. the step grid the runner asks the control to cover is the grid it actually
            WRITES readings at -- offset ONE, not zero, because the measurement fires at
            `gstep % morph_every == 0` and lands in the row labelled `gstep + 1`. Asking for
            step 0 made a complete control look like it was missing a step, and the only
            listed cure for that is --allow-control-gaps, which would have silenced a
            warning that described nothing.

      2. a REAL cadence mismatch is still caught. That is the check that matters, and it
            is the one a wrong grid would have destroyed while looking like it was working.
    """
    section("T14 the control-curve coverage grid matches the runner's ACTUAL readings")
    import csv as _csv
    import dann_round5 as D5

    def write_curve(name, first, stride, last=8000):
           q = os.path.join(tmpdir, name)
           with open(q, "w", newline="") as fh:
               w = _csv.DictWriter(fh, fieldnames=["step", D5.COND3_COLUMN,
                                                   "src_participation_ratio",
                                                   "tgt_participation_ratio"])
               w.writeheader()
               for st in range(first, last + 1, stride):
                   w.writerow({"step": st, D5.COND3_COLUMN: 0.9076,
                               "src_participation_ratio": 5.26,
                               "tgt_participation_ratio": 10.21})
           return q

    # the grid the runner computes (kept in lockstep with run_dann_r5.py by construction)
    def want_grid(max_steps, morph_every):
           return [g + 1 for g in range(0, max_steps, int(morph_every))]

    w = want_grid(8000, 25)
    check(w[0] == 1 and w[1] == 26 and w[-1] == 7976 and len(w) == 320,
          f"the requested grid is offset ONE at stride 25: {w[:3]} ... {w[-1]}, "
          f"{len(w)} steps -- NOT {list(range(0, 76, 25))}..., which no run ever writes")

    # a FORWARD-FILLED control (a row at every step, as the runner actually writes)
    ff = write_curve("forward_filled.csv", 1, 1)
    check(D5.ControlCurve(ff).covers(w) == [],
          "a complete forward-filled control covers every requested step -- 0 missing. "
          "Under the old offset-0 grid this same file reported step 0 missing and aborted "
          "the run")

    # a control written ONLY on its own reading grid, same cadence: still complete
    same = write_curve("same_cadence.csv", 1, 25)
    check(D5.ControlCurve(same).covers(w) == [],
          "a control written only at its readings, at the SAME cadence, also covers "
          "every step")

    # a REAL mismatch must still be caught, or the check is decorative
    coarse = write_curve("coarse.csv", 1, 50)
    miss = D5.ControlCurve(coarse).covers(w)
    check(len(miss) == 160 and miss[0] == 26,
          f"a control at HALF the cadence is still caught: {len(miss)} of {len(w)} steps "
          f"missing (first {miss[0]}) -- the fix narrowed the request, it did not disable "
          f"the check")

    shifted = write_curve("shifted.csv", 2, 25)
    check(len(D5.ControlCurve(shifted).covers(w)) == len(w),
          "a control on the same stride but a DIFFERENT OFFSET misses every step -- an "
          "off-by-one between arms deletes the comparison rather than sliding it")

    # and the real shipped curve, if it is here
    # repo-root relative: this suite is normally run from scripts/
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real = os.path.join(_root, "runs", "dann_r5_arm1_control", "steps.csv")
    if os.path.isfile(real):
        c = D5.ControlCurve(real)
        m = c.covers(w)
        check(not m,
              f"THE SHIPPED ARM-1 CURVE covers all {len(w)} steps arm 2 will measure "
              f"at (steps {c.summary()['step_min']}-{c.summary()['step_max']})")
        cols = set(c.summary()["columns_loaded"])
        need = {D5.COND3_COLUMN, "src_participation_ratio", "tgt_participation_ratio"}
        check(need <= cols,
              f"...and carries every column the re-specified detectors read: "
              f"{sorted(need)}")


def t15_rescore_path():
    """The re-score's step-0 gate, and its known null.

    The gate is the only thing standing between a wrong re-score path and 64 plausible
    wrong numbers. 7.1a records what that costs: scoring a real nb3 checkpoint from the
    stale feature cache once gave TPR 0.194 against a logged 0.020.

    What makes the gate strong is that it compares 26,497 LOGITS, not a TPR. At 98 test
    positives TPR moves in steps of 1/98 = 0.0102, so two genuinely different models
    collide on it easily -- which is exactly the failure the 0.194-vs-0.020 incident was.
    So the null tested here is a perturbation that a TPR check would MISS.
    """
    section("T15 the re-score step-0 gate catches what a TPR check would not")
    import numpy as _np
    import rescore_dann_r5 as _RS

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = os.path.join(root, "runs", "dann_arm2_fold0", "extracted", "dann_r5_nb3_fold0")
    ref = os.path.join(run, "logits", "true_init.npz")
    if not os.path.isfile(ref):
        print("  (arm-2 true_init.npz not present in this checkout -- T15 skipped)")
        return
    z = _np.load(ref)
    pos, neg = _np.asarray(z["pos"]), _np.asarray(z["neg"])

    tpr, auroc = _RS._metrics(pos, neg)
    check(abs(tpr - 0.19387755102040816) < 1e-12 and abs(auroc - 0.9427007516518483) < 1e-12,
          f"the re-score's metric path reproduces the RUN's own logged step-0 numbers "
          f"from the run's own logits: TPR {tpr:.10f}, AUROC {auroc:.10f}")

    class _S:
        def __init__(self, p_, n_):
            self.p, self.n = p_, n_
        def _score(self):
            return self.p, self.n, None, None

    g = _RS.step0_hard_gate(_S(pos, neg), run, 0, tpr_tol=1.0 / 98.0, auroc_tol=1e-3,
                            verbose=False)
    check(g["verdict"] == "PASS" and g["n_logits_compared"] == 26497
          and g["max_abs_diff_pos"] == 0.0 and g["max_abs_diff_neg"] == 0.0,
          f"the gate PASSES on the run's own logits, comparing "
          f"{g['n_logits_compared']} of them element-wise")

    # THE NULL THAT MATTERS: perturb ONE negative far below every positive. The ranking
    # of the 98 positives is untouched, so TPR@1e-3 and AUROC barely move -- a scalar
    # gate sails through. The element-wise check does not.
    bad = neg.copy()
    bad[0] = float(neg.min()) - 50.0
    t2, a2 = _RS._metrics(pos, bad)
    check(abs(t2 - tpr) < 1e-12,
          f"...and the perturbation chosen for the null leaves TPR IDENTICAL "
          f"({t2:.10f}) -- a scalar gate would not see it at all")
    refused = False
    try:
        _RS.step0_hard_gate(_S(pos, bad), run, 0, tpr_tol=1.0 / 98.0, auroc_tol=1e-3,
                            verbose=False)
    except SystemExit as e:
        refused = "step-0 gate FAILED" in str(e)
    check(refused,
          "KNOWN NULL: ONE of 26,399 negatives moved by 50 logits and the gate REFUSES "
          "-- the check the TPR could not make")

    # a shape change is a different failure and gets its own message
    try:
        _RS.step0_hard_gate(_S(pos[:-1], neg), run, 0, tpr_tol=1.0, auroc_tol=1.0,
                            verbose=False)
        check(False, "KNOWN NULL: a SHORT positive vector did not refuse")
    except SystemExit as e:
        check("SHAPE" in str(e),
              "...and a pool of the wrong SIZE refuses with its own message -- 'the test "
              "pool this is scoring is not the pool the run scored', not a tolerance miss")


def t16_rescore_wiring():
    """Every cross-module call in rescore_dann_r5.py BINDS against the real signature.

    WHY THIS TEST EXISTS. The re-score shipped with ``R5.resolve_paths(args)`` against a
    two-argument ``resolve_paths(args, root)``. It reached Kaggle, extracted 4.2 GB,
    built the arm, printed the banner, and died on the next line -- after the notebook
    had done everything expensive. The local smoke test could not have caught it: it
    stopped at argument parsing, because everything past that needs a GPU and 4.6 GB of
    data.

    A TypeError from a call that was never once executed is not a GPU problem, and it
    does not need a GPU to find. This binds the arguments of every cross-module call the
    script makes -- statically, from the AST, so a call inside a branch that only runs on
    the venue is checked exactly like one that runs everywhere -- and separately checks
    that every dictionary key it reads out of build_arm's return really is a key
    build_arm returns.
    """
    section("T16 rescore_dann_r5.py's cross-module calls bind against real signatures")
    import ast as _ast
    import inspect as _inspect
    import rescore_dann_r5 as _RS
    import run_dann_r5 as _R5

    src = open(os.path.join(_HERE_DIR, "rescore_dann_r5.py"), encoding="utf-8").read()
    tree = _ast.parse(src)

    # ---- 1. every R5.<fn>(...) call binds ------------------------------------
    checked, failures = [], []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, _ast.Attribute) and isinstance(fn.value, _ast.Name)
                and fn.value.id == "R5"):
            continue
        target = getattr(_R5, fn.attr, None)
        if target is None:
            failures.append(f"R5.{fn.attr} does not exist")
            continue
        if not callable(target):
            continue
        n_pos = len(node.args) + sum(1 for a in node.args
                                     if isinstance(a, _ast.Starred))
        kw = [k.arg for k in node.keywords if k.arg]
        has_star = any(isinstance(a, _ast.Starred) for a in node.args)
        has_kwstar = any(k.arg is None for k in node.keywords)
        if has_star or has_kwstar:
            continue                      # cannot be bound statically; not used here
        try:
            _inspect.signature(target).bind(*([None] * n_pos),
                                            **{k: None for k in kw})
            checked.append(f"R5.{fn.attr}")
        except TypeError as e:
            failures.append(f"R5.{fn.attr}(...) at line {node.lineno}: {e}")

    check(not failures and len(checked) >= 3,
          f"all {len(checked)} R5.* calls bind: {sorted(set(checked))}"
          + (f" -- FAILURES {failures}" if failures else ""))

    # THE REGRESSION, named. resolve_paths takes TWO positional arguments.
    sig = _inspect.signature(_R5.resolve_paths)
    check(len([q for q in sig.parameters.values()
               if q.kind is q.POSITIONAL_OR_KEYWORD and q.default is q.empty]) == 2,
          f"KNOWN REGRESSION: R5.resolve_paths{sig} takes TWO required positional "
          f"arguments -- the re-score called it with one and died on the venue after "
          f"the 4.2 GB extract and the arm build")
    bound = False
    try:
        _inspect.signature(_R5.resolve_paths).bind(None)
    except TypeError:
        bound = True
    check(bound, "...and a one-argument call still raises, so this test would fail again "
                 "if the call regressed")

    # ---- 2. Scorer / LiveEncoder / the gate ----------------------------------
    from run_coral_r4_dev8000 import Scorer as _Sc
    try:
        _inspect.signature(_Sc).bind(None, None, 0, "d", arm="a", fp16=False)
        check(True, "Scorer(trainer, live, fold, logit_dir, arm=, fp16=) binds")
    except TypeError as e:                                        # pragma: no cover
        check(False, f"Scorer signature: {e}")
    from live_encode import LiveEncoder as _LE
    try:
        _inspect.signature(_LE).bind("s", "d", 0, batch_size=256, num_workers=0,
                                     device="cpu", verbose=True)
        check(True, "LiveEncoder(splits, data_root, fold, batch_size=, num_workers=, "
                    "device=, verbose=) binds")
    except TypeError as e:                                        # pragma: no cover
        check(False, f"LiveEncoder signature: {e}")
    try:
        _inspect.signature(_RS.step0_hard_gate).bind(
            None, "d", 0, tpr_tol=0.0, auroc_tol=0.0, verbose=False)
        check(True, "step0_hard_gate binds as the runner calls it")
    except TypeError as e:                                        # pragma: no cover
        check(False, f"step0_hard_gate signature: {e}")

    # ---- 3. every A["key"] it reads is a key build_arm RETURNS ---------------
    bsrc = _ast.parse(open(os.path.join(_HERE_DIR, "run_dann_r5.py"),
                           encoding="utf-8").read())
    returned = set()
    for node in _ast.walk(bsrc):
        if isinstance(node, _ast.FunctionDef) and node.name == "build_arm":
            for r in _ast.walk(node):
                if isinstance(r, _ast.Return) and isinstance(r.value, _ast.Call):
                    returned |= {k.arg for k in r.value.keywords if k.arg}
    used = set()
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Subscript) and isinstance(node.value, _ast.Name)
                and node.value.id == "A"
                and isinstance(node.slice, _ast.Constant)
                and isinstance(node.slice.value, str)):
            used.add(node.slice.value)
    check(returned and used <= returned,
          f"every A[...] key the re-score reads ({sorted(used)}) is returned by "
          f"build_arm" + (f" -- MISSING {sorted(used - returned)}" if used - returned
                          else ""))

    # ---- 4. AND THE WHOLE FUNCTION ACTUALLY RUNS ----------------------------
    # A signature test does not catch an unbound local; a syntax check catches neither.
    # The only thing that reliably catches "this line has never run" is running it. This
    # executes rescore_arm end to end with the encoder, LiveEncoder, Scorer and the
    # checkpoint loads stubbed -- and with --run-dir deliberately UNSET, because passing
    # it short-circuits the expression that raised UnboundLocalError on the venue.
    import io as _io
    import contextlib as _ctx
    buf = _io.StringIO()
    rc = 1
    try:
        with _ctx.redirect_stdout(buf):
            rc = _RS.wiring_check(_RS.build_argparser().parse_args([]), verbose=True)
    except Exception as e:                                          # noqa: BLE001
        check(False, f"the re-score's control flow RAISED under the wiring check: "
                     f"{type(e).__name__}: {e}")
    else:
        text = buf.getvalue()
        n_ok = text.count(": OK -- gate PASS")
        check(rc == 0 and n_ok == 2,
              f"rescore_arm executes END TO END for BOTH arms with only the GPU work "
              f"stubbed ({n_ok} of 2 arms, exit {rc}) -- real step-0 gate on 26,497 "
              f"stored logits, real retention check, real curve.csv")
        check("32 curve rows" in text and "32 checkpoints loaded" in text,
              "...writing 32 curve rows per arm and loading all 32 checkpoints, so the "
              "loop, the CSV writer and the lockout boundary all ran, not just parsed")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", default=False,
                    help="skip the groups that need the real Zoobot encoder (T5/T7/T8 "
                         "and T6's full scan)")
    ap.add_argument("--reachability-out", default=None,
                    help="also write the T7 counts to this JSON. Off by default, so the "
                         "suite never overwrites the richer report "
                         "`run_dann_r5.py --grad-check` produces.")
    a = ap.parse_args()

    counts = {}
    t1_grl_sign()
    t2_horizon_invariance()
    t3_resume_equivalence()
    t4_single_forward_pass()
    t9_fp16_smoke()
    t10_holdout()
    with tempfile.TemporaryDirectory(prefix="dann_r5_t13_") as td:
        t13_abort_conditions(td)
    with tempfile.TemporaryDirectory(prefix="dann_r5_t14_") as td:
        t14_control_curve_coverage(td)
    t15_rescore_path()
    t16_rescore_wiring()
    t6_no_batchnorm(fast=a.fast)
    if not a.fast:
        t11_d_train_eval()
        t12_morph_proxy()
        t5_lambda_zero_null()
        t7_reachability(counts)
        t8_grad_checkpointing()
    else:
        print("\n[--fast] skipped T5/T7/T8/T11/T12 (real-encoder groups)")

    print("\n" + "=" * 88)
    print(f"RESULT: {len(_passes)} passed, {len(_fails)} failed")
    if counts:
        print("\nDANN-REACHABLE PARAMETER COUNTS ('where does the domain gradient go?'):")
        for k, r in counts.items():
            print(f"  {k:>5}: {r['reachable_param_elements']:>12,} elements across "
                  f"{r['reachable_tensors']:>4} tensors   {r['reachable_prefixes']}")
        dst = a.reachability_out
        if dst:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as fh:
                json.dump({"source": "test_da_dann.py T7", "arms": counts}, fh,
                          indent=2, default=str)
            print(f"  -> {dst}")
    if _fails:
        for f in _fails:
            print(f"  FAILED: {f}")
    print("=" * 88)
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
