"""
run_coral_round4.py
===================
ROUND 4 -- Deep CORAL, IMAGE-SPACE, in two arms that differ ONLY in ``n_blocks``.

    coral_nb0_control   n_blocks=0   the frozen control
    coral_nb3           n_blocks=3   the unfrozen arm (PROJECT.md §7.1a)

Both arms: image-space (encoder in the loop), ``aug="weak"``, fp16 autocast + GradScaler,
true batch 256 source + 256 target, LP->FT initialised from round-1 CORAL's fold-0
step-8000 head, a step-0 hard gate against that head's own value, weight checkpoints every
500 steps, derived-logit dumps at every curve point, the read-only agreement column, and the
post-hoc collapse gate run as part of the run.

**round_1 IS NOT AN ARM.** ``runs/bakeoff_coral/`` is read for exactly two things: the
phase-1 head to initialise from, and the step-0 known-null that proves it loaded. No number
produced here is ever differenced against it -- round-1 CORAL was cached-feature, fp32,
un-augmented, gamma-scheduled and lr=1e-3/wd=1e-4, so a difference against it would be a
sum of five changes. The comparison this round supports is **nb3 minus nb0**, which differs
in one thing. ``run_coral.py`` is not imported, not called and not modified.

WHAT DIFFERS FROM ROUND 1, AND WHY (all declared here, before any run)
---------------------------------------------------------------------
1. **Image-space instead of cached features.** Forced: the cache is invalid the moment the
   encoder moves (§7.1a), and augmentation happens pre-encoder so there is no deterministic
   prefix to cache. The control arm pays the same cost so the arms stay matched -- §7.3
   measured aug+fp16 at -0.0714 TPR on a frozen head, which is why the control is
   image-space too rather than reusing a cheap cached run.
2. **CONSTANT gamma; ``warmup_cosine`` is gone from this path.** Round 1 decayed gamma to
   ~0.100 by its step-8000 read-off, suppressing the CORAL term exactly where it was
   measured. Sun & Saenko (arXiv:1607.01719 §4) set lambda so classification and CORAL are
   of comparable magnitude at the END of training. ``--calibrate`` measures that value from
   loss magnitudes alone.
3. **Zoobot's published optimiser protocol (§7.1a), not round-1 CORAL's.** head_lr =
   backbone_lr = 1e-4, lr_decay = 0.75, weight_decay = 0.05, no LR schedule. §7.1a pins
   these from ``FinetuneableZoobotAbstract.configure_optimizers`` and notes head LR and
   top-block LR are EQUAL in Zoobot, which is why ``backbone_lr`` defaults to None (->
   equals head_lr) rather than to head_lr/10.

   > **FLAGGED, not hidden.** This is 10x lower head LR and 500x higher weight decay than
   > round-1 CORAL used. It is adopted because the brief says to match §7.1a exactly and
   > because both arms get it identically, so it cannot bias nb3-vs-nb0. It DOES mean the
   > frozen arm here is not a re-run of round 1 and must not be read as one.
4. **fp32 covariance under fp16 autocast** -- ``coral_round4.Fp32CoralLoss``. See that
   module; the 640x640 second moment and its 409,600-term Frobenius reduction are the
   obvious half-precision hazard, so they are pinned to fp32 inside an autocast-disabled
   block while the encoder forward stays fp16.

THE GRADIENT PATH -- VERIFIED, NOT ASSUMED
------------------------------------------
``train_skeleton.encode_features``'s docstring flags, as an OPEN ITEM, that at
``n_blocks>0`` a DA loss on adapter features backprops into the unfrozen encoder, and
records that this was reasoned through for FreeMatch and never for CORAL. Round 4 does not
reason about it either: ``coral_round4.coral_reachable_params`` runs the real forward,
backprops the CORAL term ALONE, and reads the gradients. ``test_coral_round4.py`` asserts
the measured reachable set on both arms. That asymmetry -- 640 LayerNorm gains on nb0 versus
gains + 14.8M encoder parameters on nb3 -- IS the experiment.

GAMMA MUST BE THE SAME ON BOTH ARMS
-----------------------------------
``--calibrate`` reports an equalising gamma PER ARM, because the raw CORAL magnitude differs
between a frozen and an unfrozen encoder. Using each arm's own value would make the arms
differ in TWO things and the comparison would no longer isolate unfreezing. ``coral_weight``
is therefore a PINNED key: the precondition check REFUSES a pair of runs whose gamma
differs. Pick one value, write it down, and pass it to both.

Run:
    python scripts/run_coral_round4.py --calibrate                    # both arms, 500 steps
    python scripts/run_coral_round4.py --arm coral_nb0_control --gamma G --emit-config ref.json
    python scripts/run_coral_round4.py --arm coral_nb3 --gamma G --reference-config ref.json
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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import curve_hook                                              # noqa: E402
import evaluate as ev                                          # noqa: E402
from heads import LINEAR                                       # noqa: E402
from lens_dataset import LensDataset, UNLABELED                # noqa: E402
from live_encode import LiveEncoder                            # noqa: E402
from metrics import DEFAULT_TARGET_FPR                         # noqa: E402
from samplers import make_balanced_loader, PairedDomainLoader  # noqa: E402
from train_skeleton import TrainConfig, Trainer, set_seed      # noqa: E402
from coral_round4 import (Fp32CoralLoss, install_fp32_coral,   # noqa: E402
                          CoralCurveInstrument, coral_reachable_params,
                          equalising_gamma, test_label_lockout,
                          check_feature_norm_ceiling, DEFAULT_FEAT_NORM_CEILING,
                          DEFAULT_FEAT_NORM_FLOOR)
# Method-agnostic protocol machinery, IMPORTED (never modified). PROJECT.md's recon of
# run_unfreeze_freematch.py found these reusable verbatim across methods.
from run_unfreeze_freematch import (expected_checkpoint_steps,  # noqa: E402
                                    assert_checkpoint_retention,
                                    check_step0_gate, run_collapse_gate,
                                    evaluate_r1, head_l2_norm)
# NOTE -- run_unfreeze_freematch.check_preconditions is deliberately NOT imported. It closes
# over that module's OWN PINNED_KEYS, so calling it from here would silently check
# FreeMatch's key list: CORAL's keys would never be compared, and every FreeMatch-only key
# would read "<absent>" on both sides and pass. The round-4 test suite caught exactly that
# (a gamma mismatch sailed through), which is why round 4 owns its check below. The other
# four imports above were each read and confirmed free of module-level coupling.
from run_screen_unfreeze import assert_no_bf16                 # noqa: E402


# ---------------------------------------------------------------------------
# ROUND-4 CONSTANTS -- the swapped constant block
# ---------------------------------------------------------------------------
BUDGET_STEPS = 8000        # see the note below; BOTH arms, no truncated control
CURVE_EVERY = 125          # 64 curve points, matching §7.10's density (blind-spot width)
CKPT_EVERY = 500           # 16 retained weight checkpoints
BATCH_SIZE = 256           # TRUE batch, never micro-batched (§7.1a: accumulation is
                           # INVALID for CORAL -- it would estimate a 640x640 covariance
                           # at rank 32 instead of 256)
POS_FRACTION = 0.5
SEED = 42
HEAD_TYPE = LINEAR

# ---- §7.1a layer-wise LR settings, matched EXACTLY (Zoobot v2.0.1 defaults) ----
LR = 1e-4                  # head LR
BACKBONE_LR = None         # None -> EQUALS head LR. Zoobot's head group gets `lr` and
                           # encoder block i=0 gets `lr * lr_decay**0` -- the SAME value.
LR_DECAY = 0.75
WEIGHT_DECAY = 0.05
# There is no LR scheduler anywhere in this path (§7.10 verified none is constructed or
# stepped, ZOOBOT_COSINE_SCHEDULE=False). Round 4 adds none.

# ---- CORAL knobs: round-1 values EXCEPT the schedule and the target EMA ----
CORAL_INCLUDE_MEAN = True
CORAL_SHRINKAGE = 0.1
CORAL_EMA_MOMENTUM = 0.9
CORAL_TARGET_EMA = False        # ---- CHANGED: OFF by default on BOTH arms ----
# WHY THE TARGET EMA IS NOW OFF.
#   FROZEN, the EMA had a well-defined fixed point: the target features are constant, so
#   the buffer converges to the target pool's own covariance and the reference it supplies
#   is a better estimate than any single 256-draw batch. That is a real benefit and it is
#   why round 1 used it.
#   UNFROZEN, there is NO fixed point. The encoder moves every step, so the buffered
#   reference is computed from a model that no longer exists -- stale by construction, not
#   by lag. Two measured consequences, neither of which is Deep CORAL:
#     (a) it DESTROYS the objective's translation invariance. Deep CORAL is shift-invariant
#         (covariance is shift-invariant; the first-moment term is a difference of means).
#         Measured response to a constant shift of the representation: 8.6e-07 relative
#         without the EMA, 6.0e+03 relative with it against a stale reference.
#     (b) it ATTENUATES the target-side gradient to 10% (the reference is
#         0.9*history(detached) + 0.1*current(grad)), making the loss asymmetric between
#         the two domains when Deep CORAL treats them symmetrically.
#   --coral-target-ema re-enables it; the default is off and it is PINNED, so the two arms
#   cannot silently disagree.
#
# KNOWN LIMITATION OF TURNING IT OFF, recorded and deliberately NOT fixed.
#   Without the EMA the target covariance is a single-batch estimate: 256 samples in 640
#   dimensions, so the sample covariance is SINGULAR (rank <= 255 < 640). Deep CORAL's
#   validated regime (Sun & Saenko arXiv:1607.01719) had d << n, which ours does not.
#   The loss stays well-defined -- the Frobenius distance between two singular matrices is
#   finite, and the 0.1 shrinkage toward (tr/d)I plus the 1e-6 jitter keep the shrunk
#   matrices positive-definite -- and, crucially, the SOURCE covariance is estimated from
#   the same 256 draws in the same 640 dimensions, so the rank deficiency is SYMMETRIC
#   ACROSS DOMAINS. A symmetric deficiency inflates the variance of the estimated distance;
#   it does not bias it toward either domain. This is therefore a NOISE cost, not a bias,
#   and it is accepted rather than patched.

# WHY 8000 STEPS FOR BOTH ARMS.
# §7.14 leaves the per-method development maximum open and asks whether a shorter cap still
# reveals a method's shape. runs/devmax_truncation.json answers that on FreeMatch's own
# 8000-step curve: the "late is worse" decay is detectable ONLY at a cap of 8000 -- every
# cap from 1500 to 6000 returns not-distinguishable, and the permutation readout that would
# license 6000 is non-monotone in the cap. So there is no evidence-backed shorter maximum to
# adopt, and 8000 stands. Both arms run it: §7.10's frozen control was truncated to a
# fraction of the unfrozen run and could therefore say nothing about the later steps, an
# acknowledged residual risk. A matched-length control closes it.
CONTROL_STEPS = BUDGET_STEPS

# ---- phase-1 head: round-1 CORAL's own per-fold step-8000 checkpoints ----
ROUND1_CORAL_DIR = os.path.normpath(os.path.join(_HERE, "..", "runs", "bakeoff_coral"))

# ---- PER-FOLD STEP-0 GATE TABLE ----
# Re-scored from runs/bakeoff_coral/fold{k}/checkpoints/step_8000.pt against
# data/feature_cache; the TPRs match that run's own results.json to 16 digits. Baked in for
# the same reason §7.11 baked in FreeMatch's: pointing --fold at one fold while gating on
# another's number would not abort, it would train happily and produce a plausible, wrong
# result. With the table, a wrong --fold fails at the gate in minutes.
FOLD_GATE = {
    0: {"tpr": 0.21428571428571427, "head_l2": 66.268356, "auroc": 0.947701},
    1: {"tpr": 0.17346938775510204, "head_l2": 67.536934, "auroc": 0.907398},
    2: {"tpr": 0.21428571428571427, "head_l2": 72.597260, "auroc": 0.904965},
    3: {"tpr": 0.15306122448979592, "head_l2": 68.220108, "auroc": 0.904980},
    4: {"tpr": 0.20618556701030927, "head_l2": 66.015175, "auroc": 0.902787},
}

# ---- GATE TOLERANCES (§7.11's measured values; the reasoning is metric-level, not
# method-level, so it transfers unchanged) ----
ONE_POSITIVE_TOL = 1.0 / 98.0      # 0.010204 -- the finest difference 98 positives express
STEP0_AUROC_TOL = 1e-3             # ~100x the measured live-encode drift

# ---- CORAL's OWN collapse-gate anchors, measured from round 1's five curve files ----
# Spearman(checkpoint, Baseline A step_8000) on the fold's target pool. Round-1 CORAL's
# agreement RISES 0.90 -> 0.986 over its 8000 steps, on all five folds -- the OPPOSITE
# direction to FreeMatch, whose agreement FELL from ~0.876 to ~0.77 frozen and to
# 0.767-0.811 unfrozen (§7.12). That is the trace to read round 4 against: a frozen
# marginal aligner on a 640-d LayerNorm converges TOWARD the source ranking, which is
# consistent with §7.4 finding CORAL at baseline. A round-4 arm that instead pulls AWAY is
# doing something round-1 CORAL could not.
CORAL_AGREEMENT_ANCHORS = {
    "round1_coral_first": 0.897, "round1_coral_last": 0.986,
    "round1_coral_range_all_folds": [0.8969, 0.9869],
    "round1_coral_direction": "RISING on all five folds",
    "freematch_frozen_floor": 0.87,
    "freematch_unfrozen_nb3_minima": [0.760, 0.796],
    "note": ("read the TRACE, never the verdict (§7.12). The gate's cliff detector uses a "
             "W=1000 persistence window; at any budget where that is a large fraction of "
             "the run it cannot fire, and §7.10 showed a 0.52 cliff that recovered did not "
             "fire it either. Silence carries no information."),
}

DEFAULT_AGREEMENT_REF = os.path.normpath(
    os.path.join(_HERE, "..", "runs", "bakeoff_baseline_A", "checkpoints", "step_8000.pt"))

RUNS = {
    "coral_nb3":         {"n_blocks": 3, "steps": BUDGET_STEPS, "dir": "coral_r4_nb3"},
    "coral_nb0_control": {"n_blocks": 0, "steps": CONTROL_STEPS, "dir": "coral_r4_nb0_control"},
}

# ---------------------------------------------------------------------------
# CONFOUND PINNING -- everything that must MATCH between the two arms
# ---------------------------------------------------------------------------
# §7.8's 15-key check, re-derived for CORAL. FreeMatch's eight pseudo-label keys are gone
# (no SAT, no mu schedule, no local threshold) and CORAL's loss knobs are added. coral_weight
# is PINNED: --calibrate produces a different equalising gamma per arm, and letting each arm
# take its own would make the pair differ in two things at once.
PINNED_KEYS = ("augmentation", "precision", "seed", "batch_size",
               "lr", "lr_decay", "weight_decay", "pos_fraction", "head_type", "fold",
               "curve_every", "ckpt_every", "true_batch", "gradient_accumulation",
               "init_head_from", "init_head_l2", "max_steps", "budget_steps",
               "method", "coral_weight", "coral_weight_schedule", "coral_include_mean",
               "coral_shrinkage", "coral_target_ema", "coral_ema_momentum",
               "coral_loss_precision", "coral_attachment", "eval_precision",
               "feat_norm_ceiling", "feat_norm_floor")

# EXEMPT, each for a stated reason -- never widen this list to make a run pass.
EXEMPT_KEYS = ("n_blocks",              # THE variable under test
               "grad_checkpointing",    # memory<->recompute, numerically exact (§7.1a)
               "backbone_lr",           # exists only in the unfrozen arm
               "backbone_lr_effective",
               "encoder_eval",          # follows n_blocks (cache valid iff frozen)
               "epochs_planned",        # derived from max_steps, which is pinned
               "run", "run_dir",        # naming
               "device", "num_workers") # throughput only; the seeded sampler is unaffected


def check_preconditions(cfg_json, reference_config, verbose=True,
                        pinned=PINNED_KEYS, exempt=EXEMPT_KEYS):
    """REFUSE to start unless every PINNED key matches the declared reference arm.

    Round 4 owns this rather than importing §7.8's version, because that one closes over
    FreeMatch's ``PINNED_KEYS``. The key list is a parameter here so the coupling cannot
    recur silently.

    A key that is ABSENT FROM BOTH configs is reported as a mismatch, not a match: a pinned
    key that no longer exists in either config means the guard has stopped guarding
    anything, which is how a check rots into a rubber stamp.
    """
    if not os.path.isfile(reference_config):
        raise SystemExit(
            f"[coral-r4] ABORT: reference config not found: {reference_config}\n"
            f"  This run cannot verify that only n_blocks (and the exempt keys) differ, so "
            f"it will not start. Emit the other arm's config first:\n"
            f"    python scripts/run_coral_round4.py --arm coral_nb0_control --gamma G "
            f"--emit-config <path>")
    with open(reference_config) as fh:
        ref = json.load(fh)

    rows, bad, vacuous = [], [], []
    for k in pinned:
        rv, cv = ref.get(k, "<absent>"), cfg_json.get(k, "<absent>")
        ok = (rv == cv)
        if rv == "<absent>" and cv == "<absent>":
            ok = False
            vacuous.append(k)
        rows.append((k, rv, cv, ok))
        if not ok:
            bad.append(k)
    for k in exempt:
        rows.append((k, ref.get(k, "<absent>"), cfg_json.get(k, "<absent>"), None))

    if verbose:
        print("\n" + "=" * 92)
        print("PRECONDITION CHECK -- only n_blocks and its dependents may differ")
        print(f"  reference: {reference_config}")
        print("=" * 92)
        print(f"  {'key':<26} | {'reference':<26} | {'this run':<22} | status")
        print("  " + "-" * 88)
        for k, rv, cv, ok in rows:
            status = "EXEMPT" if ok is None else ("ok" if ok else "*** MISMATCH ***")
            print(f"  {k:<26} | {str(rv)[:26]:<26} | {str(cv)[:22]:<22} | {status}")
        print("  " + "-" * 88)
    if vacuous:
        raise SystemExit(
            f"\n[coral-r4] ABORT: {len(vacuous)} pinned key(s) are absent from BOTH "
            f"configs: {vacuous}. A guard that compares nothing is not a guard -- fix the "
            f"key list or the config builder.")
    if bad:
        raise SystemExit(
            f"\n[coral-r4] ABORT: {len(bad)} pinned setting(s) differ from the reference "
            f"arm: {bad}.\n"
            f"  The two arms may differ ONLY in n_blocks and its dependents. "
            f"Augmentation+precision alone is worth -0.0714 TPR (§7.3), and a per-arm "
            f"gamma would make nb3-minus-nb0 measure two changes instead of one.")
    if verbose:
        print("  ALL PINNED SETTINGS MATCH.\n")
    return rows


def round1_fold_ckpt(fold: int) -> str:
    """That fold's own round-1 CORAL step-8000 checkpoint (READ-ONLY, never written)."""
    return os.path.join(ROUND1_CORAL_DIR, f"fold{int(fold)}", "checkpoints", "step_8000.pt")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def build_config(run_key, args, root, fold=None):
    spec = RUNS[run_key]
    fold = int(args.fold if fold is None else fold)
    if fold not in FOLD_GATE:
        raise SystemExit(f"[coral-r4] --fold must be one of {sorted(FOLD_GATE)}, got {fold}")
    n_blocks = spec["n_blocks"]
    max_steps = int(args.max_steps) if args.max_steps else spec["steps"]
    grad_ckpt = bool(args.grad_checkpointing) and n_blocks > 0
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    head_ckpt = os.path.abspath(args.init_head_from or round1_fold_ckpt(fold))
    batch_size = int(args.smoke_batch_size or BATCH_SIZE)
    fp16 = bool(args.fp16)

    cfg = {
        "round": 4, "run": run_key, "run_dir": f'{spec["dir"]}_fold{fold}',
        "n_blocks": n_blocks, "fold": fold,
        "method": "deep_coral", "domain_adaptation": "coral",
        "data_path": "image_space",
        "lp_ft": True, "init_head_from": head_ckpt, "init_head_l2": None,
        "init_head_provenance": ("round-1 CORAL fold-%d step_8000 (runs/bakeoff_coral). "
                                 "Read-only: used as the LP->FT phase-1 head and as the "
                                 "step-0 known-null. Round 1 is NOT an arm and no round-4 "
                                 "number is differenced against it." % fold),
        "max_steps": max_steps, "budget_steps": max_steps,
        "is_preregistered_budget": (max_steps == spec["steps"]
                                    and not args.smoke_batch_size),
        "curve_every": int(args.curve_every), "ckpt_every": int(args.ckpt_every),
        "batch_size": batch_size, "true_batch": True, "gradient_accumulation": False,
        "batch_size_is_protocol": (batch_size == BATCH_SIZE),
        "grad_checkpointing": grad_ckpt,
        # ---- §7.1a layer-wise LR, matched exactly ----
        "lr": LR, "lr_decay": LR_DECAY, "weight_decay": WEIGHT_DECAY,
        "backbone_lr": (args.backbone_lr if n_blocks > 0 else None),
        "backbone_lr_effective": ((args.backbone_lr if args.backbone_lr is not None else LR)
                                  if n_blocks > 0 else None),
        "lr_protocol": ("zoobot_v2.0.1 defaults per PROJECT.md §7.1a: head_lr == "
                        "top-block lr, lr_decay 0.75, weight_decay 0.05, no LR schedule. "
                        "NOT round-1 CORAL's lr=1e-3/wd=1e-4 -- see the module docstring."),
        "pos_fraction": POS_FRACTION, "seed": SEED, "head_type": HEAD_TYPE,
        "selection_rule": f"R1_last_checkpoint_step{max_steps}",
        # ---- the CORAL loss ----
        "coral_weight": (None if args.gamma is None else float(args.gamma)),
        "coral_weight_schedule": "constant",
        "coral_weight_schedule_note": (
            "warmup_cosine is REMOVED from this path. Round 1 decayed gamma to ~0.100*peak "
            "by its step-8000 read-off, suppressing the CORAL term exactly where it was "
            "measured; Sun & Saenko arXiv:1607.01719 §4 set lambda so the two losses are "
            "comparable at the END of training. gamma here is constant for all steps."),
        "coral_include_mean": CORAL_INCLUDE_MEAN,
        "coral_shrinkage": CORAL_SHRINKAGE,
        "coral_target_ema": bool(args.coral_target_ema),
        "coral_ema_momentum": CORAL_EMA_MOMENTUM,
        "coral_attachment": "head.forward_with_features[1] (post-adapter, post-LayerNorm)",
        "coral_attachment_note": (
            "The SAME tensor round 1 used, identical on both arms and deliberately not "
            "moved. At head_type=linear the adapter is nn.LayerNorm(640) over the "
            "source-fitted standardizer output, so the tensor is 640-d. On nb0 the CORAL "
            "gradient reaches only the 640 LayerNorm GAINS (the bias cancels: covariance "
            "is shift-invariant and the first-moment term is a difference of means). On "
            "nb3 it flows through into the encoder. That asymmetry is the experiment."),
        "coral_loss_precision": "fp32_autocast_disabled",
        "feat_norm_ceiling": float(getattr(args, "feat_norm_ceiling",
                                           DEFAULT_FEAT_NORM_CEILING)),
        "feat_norm_floor": float(getattr(args, "feat_norm_floor",
                                         DEFAULT_FEAT_NORM_FLOOR)),
        "feat_norm_floor_note": (
            "HARD ABORT if either domain's mean ||f|| falls BELOW this. The ceiling "
            "catches the fp16 overflow direction only; the eps^4 degenerate solution "
            "drives the norm DOWN and nothing caught it. Shrink pressure is measured, "
            "monotone in gamma (-1.39/-1.69/-2.50% at 1/4/16 over 2000 steps) and still "
            "accelerating at the selected gamma=16, while the confirmatory runs are 4x "
            "longer. 40 against a ~66 init is a ~40%% collapse -- far outside anything "
            "observed; it catches a runaway, not ordinary drift."),
        "feat_norm_ceiling_note": (
            "HARD ABORT if either domain's mean post-LayerNorm ||f|| exceeds this. The "
            "unguarded fp16 covariance path is MEASURED to go non-finite at ||f|| ~= 126; "
            "LP->FT init sits at ~66 and the 500-step calibration reached 82 by step 400, "
            "so this is a live trajectory. An 8000-step run must not silently produce NaNs "
            "hours in. The value is an engineering choice, placed below the measured "
            "failure point with room above the observed trajectory."),
        "encoder_eval": ("frozen_feature_cache" if n_blocks == 0 else "live_encode"),
        "augmentation": "weak (+-3px translate, PROJECT.md §7.3)",
        "fp16": fp16, "precision": ("fp16" if fp16 else "fp32"),
        "eval_precision": ("fp16" if args.eval_fp16 else "fp32"),
        "device": device, "num_workers": int(args.num_workers),
        "agreement_ref": os.path.abspath(args.agreement_ref),
        "expect_step0_tpr": float(FOLD_GATE[fold]["tpr"]),
        "expect_step0_auroc": float(FOLD_GATE[fold]["auroc"]),
        "expect_head_l2": float(FOLD_GATE[fold]["head_l2"]),
        "step0_tpr_tolerance": (0.0 if n_blocks == 0 else ONE_POSITIVE_TOL),
        "step0_auroc_tolerance": STEP0_AUROC_TOL,
        "hard_gate": ("step-0 TPR@1e-3 + AUROC must reproduce the phase-1 checkpoint's own "
                      "fold value; abort otherwise"),
        "collapse_gate_reporting": (
            "log the agreement TRACE, never the verdict. §7.12: the cliff detector's "
            "W=1000 persistence window can make the gate structurally inoperative at short "
            "budgets, and §7.10 showed a 0.52 cliff that recovered did not fire it. "
            "Silence carries no information; the trace is the report."),
        "feature_scale_monitor": (
            "PRE-REGISTERED, recorded before either arm ran. Scaling features by eps scales "
            "both covariances by eps^2 and the squared-Frobenius CORAL objective by eps^4, "
            "so the loss can fall by SHRINKING the representation rather than aligning it. "
            "Falling post-LayerNorm feature norm alongside falling raw CORAL loss is the "
            "collapse signature; falling CORAL with a flat norm is genuine alignment. nb0 "
            "is the near-null (only 640 gains can rescale). Logged per domain at every "
            "curve point in diagnostics.csv. Not to be adjusted after the runs."),
        "reporting_note": ("fold 0 is DEVELOPMENT (§7.11/§7.14). Report TPR@FPR=1e-3 as a "
                           "RATE; never counts (§9)."),
    }
    return cfg, spec, max_steps, grad_ckpt, device, head_ckpt, fold, fp16


def gate_or_abort_coral(step0_row, expected, tol, run_key, fold, expected_auroc):
    info = check_step0_gate(step0_row, expected, tol, run_key, fold=fold,
                            expected_auroc=expected_auroc, auroc_tol=STEP0_AUROC_TOL)
    if info["passed"] is not True:
        raise SystemExit(
            f"[coral-r4] ABORT at the step-0 gate: TPR@1e-3 = {info['step0_tpr']} does not "
            f"reproduce round-1 CORAL fold-{fold}'s {expected}.\n"
            f"  Step 0 is the phase-1 model itself, so this is a plumbing failure, not a "
            f"result. Likely causes, in order: wrong --init-head-from (check ||head||_2 "
            f"against the pinned {FOLD_GATE[fold]['head_l2']}), the feature standardizer "
            f"not fitted from the cached source features, an fp16 eval encode against an "
            f"fp32 reference, or venue drift (§7.7: local Ada vs Kaggle T4 differ by "
            f"0.0204 = two positives).")
    return info


# ---------------------------------------------------------------------------
# the training loop
# ---------------------------------------------------------------------------

def build_arm(run_key, args, root, *, for_calibration=False):
    """Construct (trainer, coral, paired, src_loader, tgt_loader, cfg_json, ...).

    Shared by the real run and by --calibrate so the calibration measures the loss
    magnitudes of the SAME configuration that will be trained, not an approximation of it.
    """
    (cfg_json, spec, max_steps, grad_ckpt, device,
     head_ckpt, fold, fp16) = build_config(run_key, args, root)
    n_blocks = spec["n_blocks"]
    batch_size = int(cfg_json["batch_size"])

    splits_path = os.path.abspath(args.splits)
    data_root = os.path.abspath(args.data_root)
    cache_dir = os.path.abspath(args.cache_dir)

    assert_no_bf16(cfg_json["precision"], device)

    if not os.path.isfile(head_ckpt):
        raise SystemExit(
            f"[coral-r4] ABORT: phase-1 head not found: {head_ckpt}\n"
            f"  Both arms initialise from round-1 CORAL's fold-{fold} step_8000 head. "
            f"runs/ is not packed into data.zip, so on a fresh remote session point "
            f"--init-head-from straight at it.")
    cfg_json["init_head_l2"] = round(head_l2_norm(head_ckpt), 6)
    if abs(cfg_json["init_head_l2"] - cfg_json["expect_head_l2"]) > 1e-4:
        raise SystemExit(
            f"[coral-r4] ABORT: phase-1 head ||.||_2 = {cfg_json['init_head_l2']} but the "
            f"pinned value for fold {fold} is {cfg_json['expect_head_l2']}. This is not "
            f"fold {fold}'s round-1 CORAL head -- check --fold and --init-head-from.")

    # ---- datasets: ONE view per domain (CORAL needs no weak/strong pair) ----
    src_ds = LensDataset(splits_path, data_root, role="source_train",
                         aug="weak", n_views=1)
    tgt_ds = LensDataset(splits_path, data_root, role="target", fold=fold,
                         aug="weak", n_views=1)
    assert np.all(tgt_ds.labels == UNLABELED), "target labels must all be UNLABELED(-1)"

    steps = len(src_ds) // batch_size
    epochs_planned = max(1, int(math.ceil(max_steps / max(steps, 1))))
    cfg_json["epochs_planned"] = epochs_planned
    cfg_json["steps_per_epoch"] = steps

    src_loader = make_balanced_loader(src_ds, batch_size, use_latent=False,
                                      num_batches=steps, pos_fraction=POS_FRACTION,
                                      seed=SEED + fold, num_workers=int(args.num_workers),
                                      pin_memory=True,
                                      persistent_workers=int(args.num_workers) > 0)
    tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True,
                                      num_batches=steps, pos_fraction=POS_FRACTION,
                                      seed=SEED + fold + 1000,
                                      num_workers=int(args.num_workers),
                                      pin_memory=True,
                                      persistent_workers=int(args.num_workers) > 0)
    paired = PairedDomainLoader(src_loader, tgt_loader)

    from zoobot_encoder import ZoobotEncoder
    print(f"[load] building the Zoobot encoder (n_blocks={n_blocks}, "
          f"grad_checkpointing={grad_ckpt}) ...", flush=True)
    encoder = ZoobotEncoder(encoder_mode="frozen", n_blocks=n_blocks, device=device,
                            verbose=False, grad_checkpointing=grad_ckpt)

    gamma = 1.0 if for_calibration else float(cfg_json["coral_weight"])
    tcfg = TrainConfig(
        splits_path=splits_path, data_root=data_root, fold=fold, head_type=HEAD_TYPE,
        encoder_mode="frozen", lr=LR, weight_decay=WEIGHT_DECAY, batch_size=batch_size,
        pos_fraction=POS_FRACTION,
        use_coral=True, coral_weight=gamma,
        coral_include_mean=CORAL_INCLUDE_MEAN, coral_shrinkage=CORAL_SHRINKAGE,
        coral_target_ema=bool(args.coral_target_ema),
        coral_ema_momentum=CORAL_EMA_MOMENTUM,
        total_steps=max_steps, epochs=epochs_planned, steps_per_epoch=steps,
        max_steps=max_steps, seed=SEED + fold, device=device,
        num_workers=int(args.num_workers), amp=fp16,
        n_blocks=n_blocks, backbone_lr=args.backbone_lr, lr_decay=LR_DECAY,
        init_head_from=head_ckpt, grad_checkpointing=grad_ckpt)
    trainer = Trainer(tcfg, encoder=encoder)

    # THE round-4 loss: fp32 moments + the pre-registered feature-scale record.
    coral = install_fp32_coral(trainer, include_mean=CORAL_INCLUDE_MEAN,
                               shrinkage=CORAL_SHRINKAGE,
                               target_ema=bool(args.coral_target_ema),
                               ema_momentum=CORAL_EMA_MOMENTUM)

    # ---- standardizer from the CACHED source features (train/eval consistency) ----
    cache = ev.FeatureCache(cache_dir)
    if not cache.has("source_train"):
        raise FileNotFoundError(
            f"source_train.npz not found under {cache_dir}; run cache_features.py first. "
            f"(The standardizer is fitted from the cached SOURCE features so that training "
            f"and evaluation standardize identically -- this is the one cache read that is "
            f"valid at n_blocks>0, because it is a fixed constant of the source domain.)")
    src_feats = torch.from_numpy(
        np.asarray(cache.load("source_train")["features"], dtype=np.float32)).to(trainer.device)
    trainer.standardizer.fit(src_feats)
    del src_feats

    return dict(cfg_json=cfg_json, spec=spec, max_steps=max_steps, grad_ckpt=grad_ckpt,
                device=device, head_ckpt=head_ckpt, fold=fold, fp16=fp16,
                n_blocks=n_blocks, batch_size=batch_size, steps=steps,
                epochs_planned=epochs_planned, trainer=trainer, coral=coral,
                paired=paired, src_loader=src_loader, tgt_loader=tgt_loader,
                encoder=encoder, splits_path=splits_path, data_root=data_root,
                cache_dir=cache_dir)


def train_one(run_key, args):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    if args.gamma is None:
        raise SystemExit(
            "[coral-r4] ABORT: --gamma is REQUIRED for a training run and has no default.\n"
            "  gamma is a pre-registered constant for this round, not an inherited value. "
            "Measure it first:\n"
            "      python scripts/run_coral_round4.py --calibrate\n"
            "  then pass the SAME value to both arms (it is a PINNED key -- a pair whose "
            "gamma differs is refused, because the arms would then differ in two things).\n"
            "  --gamma 1.0 is a legitimate choice: it is the un-tuned RLSbench-style "
            "fixed-budget value round 1 nominally used.")

    A = build_arm(run_key, args, root)
    cfg_json, trainer, coral = A["cfg_json"], A["trainer"], A["coral"]
    fold, n_blocks, max_steps = A["fold"], A["n_blocks"], A["max_steps"]

    suffix = args.dir_suffix or ""
    if not cfg_json["is_preregistered_budget"] and not suffix:
        suffix = f"_SMOKE{max_steps}"
    out_dir = os.path.abspath(os.path.join(root, "runs", cfg_json["run_dir"] + suffix))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    curve_csv = os.path.join(out_dir, "curve.csv")
    agreement_ref = os.path.abspath(args.agreement_ref)

    print("=" * 92)
    print(f"ROUND-4 CORAL -- {cfg_json['run_dir'] + suffix}  (fold {fold}, {max_steps} "
          f"steps, n_blocks={n_blocks}, gamma={cfg_json['coral_weight']})")
    print("=" * 92)
    for k, v in cfg_json.items():
        print(f"  {k:<28}: {v}")
    print("=" * 92, flush=True)

    if args.emit_config:
        p = os.path.abspath(args.emit_config)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as fh:
            json.dump(cfg_json, fh, indent=2)
        print(f"[emit-config] declared reference for {run_key} -> {p}\n"
              f"[emit-config] nothing was trained. Pass it to the other arm as "
              f"--reference-config.")
        return None

    if args.reference_config:
        check_preconditions(cfg_json, os.path.abspath(args.reference_config))
    else:
        print("[pin] no --reference-config given -- this run DECLARES the reference. Emit "
              "it with --emit-config and pass it to the other arm.\n")

    if not os.path.isfile(agreement_ref):
        raise SystemExit(
            f"[coral-r4] ABORT: agreement reference not found: {agreement_ref}\n"
            f"  The collapse gate's agreement column is Spearman(this checkpoint, Baseline "
            f"A step_8000) on the fold's target pool (§7.6) and is comparable across runs "
            f"only if every run uses the SAME reference file.")

    want = RUNS[run_key]["steps"]
    cfg_json["preregistered_budget"] = want
    cfg_json["on_preregistered_budget"] = (max_steps == want)
    if max_steps != want and not args.allow_off_budget:
        raise SystemExit(
            f"[coral-r4] ABORT: effective budget is {max_steps} steps but the declared "
            f"budget for {run_key} is {want}.\n"
            f"  If this is a deliberate diagnostic, pass --allow-off-budget; the config "
            f"records on_preregistered_budget=false so it can never be aggregated as a "
            f"confirmatory result.")

    if args.dry_run:
        print("[dry-run] config validated; nothing trained.")
        return None

    if os.path.exists(curve_csv):
        raise SystemExit(f"[coral-r4] ABORT: {curve_csv} already exists -- appending would "
                         f"corrupt the curve. Move/delete {out_dir} first.")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- agreement reference (Baseline A on this fold's target pool, from the FROZEN
    # cache -- correct: the reference is a fixed vector from a frozen model; only the
    # CANDIDATE side is live-encoded) ----
    from run_bakeoff_coral import compute_agreement_refs
    print(f"[ref] scoring Baseline A on target_fold{fold} for the agreement column ...",
          flush=True)
    agreement_refs = compute_agreement_refs(A["cache_dir"], agreement_ref, [fold],
                                            device="cpu")

    def live_factory():
        if n_blocks == 0:
            return None                 # the frozen cache is valid and byte-identical
        return LiveEncoder(A["splits_path"], A["data_root"], fold,
                           batch_size=A["batch_size"], num_workers=0,
                           device=A["device"], verbose=True)

    instrument = CoralCurveInstrument(out_dir, fold, live_encoder_factory=live_factory,
                                      fp16_eval=bool(args.eval_fp16))

    ckpt_stride = int(args.ckpt_every)
    scratch_ckpt = os.path.join(ckpt_dir, "_curve_scratch.pt")

    def curve_point(gstep, full_state=False):
        retain = (gstep % ckpt_stride == 0)
        spath = os.path.join(ckpt_dir, f"step_{gstep}.pt") if retain else scratch_ckpt
        # Trainer.save on BOTH arms: it always carries the DA buffers (CoralLoss's EMA
        # mean/cov) and the schedule position, which round-1 checkpoints did not -- so a
        # round-4 checkpoint is continuable without the reconstruction §7.10 needed.
        trainer.save(spath, full_state=full_state,
                     schedule={"schedule_total_steps": int(max_steps),
                               "global_step": int(gstep),
                               "coral_weight": float(cfg_json["coral_weight"])})
        loader_hook = instrument.split_loader(trainer.encoder)
        if loader_hook is not None:
            loader_hook.reset()
        scores = {}
        row = curve_hook.append_curve_row(
            curve_csv, gstep, spath, A["cache_dir"], folds=(fold,),
            target_fpr=args.target_fpr, device="cpu",
            agreement_refs=agreement_refs, split_loader=loader_hook,
            collect_scores=scores)
        memo = instrument.on_curve_point(gstep, scores)
        trainer.head.train()
        return row, memo, (spath if retain else None)

    def log_curve(gstep, row, memo):
        d = instrument.last_diag()
        print(f"      [step {gstep:>6}] tpr@1e-3 f{fold}="
              f"{_cf(row.get(f'fold{fold}_tpr_1e-3'))}  "
              f"agr={_cf(row.get(f'agreement_fold{fold}'))}  "
              f"coral_raw={_cf(d.get('coral_raw_mean'))}  "
              f"src|f|={_cf(d.get('src_feat_l2_mean'))} "
              f"tgt|f|={_cf(d.get('tgt_feat_l2_mean'))} "
              f"(unit-gain ref {math.sqrt(640):.3f})  "
              f"mem={_cf(memo.get('target_minus_test_matched'))}", flush=True)

    # ---- STEP-0 HARD GATE, before a single optimizer step ----
    print("  [step0] evaluating the loaded model BEFORE training (hard gate) ...", flush=True)
    step0_row, step0_memo, _ = curve_point(0)
    log_curve(0, step0_row, step0_memo)
    gate = gate_or_abort_coral(step0_row, cfg_json["expect_step0_tpr"],
                               max(args.step0_tol, cfg_json["step0_tpr_tolerance"]),
                               run_key, fold, cfg_json["expect_step0_auroc"])

    # ---- train ----
    if A["device"] == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    global_step = 0
    epoch_logs = []
    stop = False
    for epoch in range(1, A["epochs_planned"] + 1):
        A["src_loader"].batch_sampler.set_epoch(epoch)
        A["tgt_loader"].batch_sampler.set_epoch(epoch)
        trainer.head.train()
        t_ep = time.time()
        for sbatch, tbatch in A["paired"]:
            scale_before = (trainer.scaler.get_scale() if trainer._amp_enabled else None)
            stats = trainer.train_step(sbatch, tbatch, step=global_step,
                                       total_steps=max_steps)
            skipped = bool(trainer._amp_enabled
                           and trainer.scaler.get_scale() < scale_before)
            cstats = coral.step_stats()
            stats = {**stats, **cstats,
                     "step_skipped": float(skipped),
                     "amp_scale": (float(trainer.scaler.get_scale())
                                   if trainer._amp_enabled else float("nan"))}
            instrument.observe_step(stats)
            global_step += 1
            # CHANGE 2 -- the hard feature-norm abort, checked EVERY step. Deliberately not
            # deferred to the curve point: at curve_every=125 a run could travel a long way
            # past the ceiling (and into the fp16 failure region) before anyone looked.
            check_feature_norm_ceiling(global_step, coral.last_scale,
                                       cfg_json["feat_norm_ceiling"], cstats["coral_raw"],
                                       floor=cfg_json["feat_norm_floor"])
            if global_step % int(args.curve_every) == 0:
                is_last = global_step >= max_steps
                row, memo, _ = curve_point(global_step, full_state=is_last)
                log_curve(global_step, row, memo)
            if global_step >= max_steps:
                stop = True
                break
        epoch_logs.append({"epoch": epoch, "secs": time.time() - t_ep,
                           "steps_done": global_step})
        if stop:
            break
    train_secs = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 2 ** 30
               if (A["device"] == "cuda" and torch.cuda.is_available()) else None)
    cfg_json["peak_gpu_gb"] = peak_gb

    if os.path.exists(scratch_ckpt):
        try:
            os.remove(scratch_ckpt)
        except OSError:
            pass

    r1 = os.path.join(ckpt_dir, f"step_{max_steps}.pt")
    if not os.path.isfile(r1):
        raise SystemExit(f"[coral-r4] ABORT: R1 checkpoint {r1} was not written "
                         f"({max_steps} must be a multiple of --ckpt-every).")
    kept = assert_checkpoint_retention(ckpt_dir, curve_csv, max_steps,
                                       int(args.ckpt_every), int(args.curve_every))
    print(f"[train] done in {train_secs/60:.1f} min", flush=True)

    res = evaluate_r1(r1, n_blocks, A["encoder"], instrument, A["cache_dir"],
                      args.target_fpr, bool(args.eval_fp16), out_dir,
                      cfg_json["run_dir"], fold=fold)
    res.update({"round": 4, "run": run_key, "run_dir": cfg_json["run_dir"],
                "n_blocks": n_blocks, "fold": fold, "coral_weight": cfg_json["coral_weight"],
                "train_minutes": train_secs / 60.0, "steps": max_steps,
                "step0_gate": gate, "epochs": epoch_logs,
                "n_checkpoints_saved": len(kept), "peak_gpu_gb": peak_gb,
                "instrument": instrument.summary()})
    res["collapse_gate"] = run_collapse_gate(out_dir, fold, agreement_ref,
                                             A["cache_dir"], n_blocks)
    res["collapse_gate"]["reporting_rule"] = cfg_json["collapse_gate_reporting"]
    # run_collapse_gate's inherited `anchors` are FreeMatch's (self-training floor 0.87,
    # Lever-1 lows). Replace them with CORAL's OWN, measured from round 1's five curve
    # files, or the trace would be read against the wrong reference.
    res["collapse_gate"]["anchors"] = CORAL_AGREEMENT_ANCHORS

    cfg_json["n_checkpoints_expected"] = len(
        expected_checkpoint_steps(max_steps, int(args.ckpt_every)))
    cfg_json["n_checkpoints_saved"] = len(kept)
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(cfg_json, fh, indent=2)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"[out] run_config.json + results.json + curve.csv + diagnostics.csv + "
          f"memorization.csv + logits/ + scores/ -> {out_dir}")
    return res


def _cf(x):
    try:
        return "n/a" if x is None else f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------------------
# --calibrate : gamma from LOSS MAGNITUDES ONLY
# ---------------------------------------------------------------------------

def calibrate_arm(run_key, args, root) -> dict:
    """Run ``--calibrate-steps`` steps at gamma=1.0 and report the equalising gamma.

    THE HARD RULE, enforced not merely stated: this function runs entirely inside
    :func:`coral_round4.test_label_lockout`, which makes TPR / AUROC / AUPRC / the metric
    suite / the curve hook and every ``test_*`` cache split raise on contact. gamma is
    derived from two loss series and nothing else, and it is not adjustable after seeing a
    metric because no metric is reachable from here.

    Note gamma=1.0 during calibration is a MEASUREMENT setting, not a training choice: the
    reported ``coral_raw`` series is the UNWEIGHTED loss, so the equalising gamma it yields
    does not depend on the weight used while measuring. It does, however, depend on the
    trajectory the weight induces -- at gamma=1.0 the CORAL term is actively being minimised
    while the series is recorded, which is the honest condition to measure under and is
    stated here rather than hidden.
    """
    n_steps = int(args.calibrate_steps)
    tail = int(args.calibrate_tail)
    print("\n" + "=" * 92)
    print(f"GAMMA CALIBRATION -- {run_key}  ({n_steps} steps at gamma=1.0, fold "
          f"{args.fold})")
    print("=" * 92)
    print("  Sun & Saenko arXiv:1607.01719 §4: set lambda so the classification and CORAL")
    print("  losses are of roughly the SAME MAGNITUDE at the end of training.")
    print("  LOSS MAGNITUDES ONLY -- no TPR, no AUROC, no test split is reachable below.")
    print("=" * 92, flush=True)

    A = build_arm(run_key, args, root, for_calibration=True)
    trainer, coral = A["trainer"], A["coral"]
    assert float(trainer.cfg.coral_weight) == 1.0, "calibration must run at gamma=1.0"

    sup, raw, cov, mean_t = [], [], [], []
    sl2, tl2, ssd, tsd, skips = [], [], [], [], 0
    t0 = time.time()
    step = 0
    for epoch in range(1, A["epochs_planned"] + 1):
        A["src_loader"].batch_sampler.set_epoch(epoch)
        A["tgt_loader"].batch_sampler.set_epoch(epoch)
        trainer.head.train()
        for sbatch, tbatch in A["paired"]:
            scale_before = (trainer.scaler.get_scale() if trainer._amp_enabled else None)
            st = trainer.train_step(sbatch, tbatch, step=step, total_steps=n_steps)
            if trainer._amp_enabled and trainer.scaler.get_scale() < scale_before:
                skips += 1
            cs = coral.step_stats()
            sup.append(st["sup_loss"]); raw.append(cs["coral_raw"])
            cov.append(cs["cov_term"]); mean_t.append(cs["mean_term"])
            sl2.append(cs["src_feat_l2"]); tl2.append(cs["tgt_feat_l2"])
            ssd.append(cs["src_perdim_std"]); tsd.append(cs["tgt_perdim_std"])
            step += 1
            if step % max(n_steps // 10, 1) == 0:
                print(f"    step {step:>5}/{n_steps}  sup={sup[-1]:.5f}  "
                      f"coral_raw={raw[-1]:.6g}  src|f|={sl2[-1]:.3f}  "
                      f"tgt|f|={tl2[-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            if step >= n_steps:
                break
        if step >= n_steps:
            break

    g = equalising_gamma(np.array(sup), np.array(raw), tail=tail)
    out = {
        "arm": run_key, "n_blocks": A["n_blocks"], "fold": A["fold"],
        "calibration_steps": step, "tail_window": tail,
        "gamma_measured_at": 1.0,
        "precision": A["cfg_json"]["precision"],
        "coral_loss_precision": "fp32_autocast_disabled",
        "fp16_steps_skipped": int(skips),
        "fp16_skip_rate": skips / max(step, 1),
        **g,
        "cov_term_tail_mean": float(np.mean(cov[-tail:])),
        "mean_term_tail_mean": float(np.mean(mean_t[-tail:])),
        "src_feat_l2_tail_mean": float(np.mean(sl2[-tail:])),
        "tgt_feat_l2_tail_mean": float(np.mean(tl2[-tail:])),
        "src_perdim_std_tail_mean": float(np.mean(ssd[-tail:])),
        "tgt_perdim_std_tail_mean": float(np.mean(tsd[-tail:])),
        "feat_l2_unit_gain_reference": math.sqrt(640.0),
        "src_feat_l2_first10_mean": float(np.mean(sl2[:10])),
        "tgt_feat_l2_first10_mean": float(np.mean(tl2[:10])),
        "wall_minutes": (time.time() - t0) / 60.0,
        "series": {"sup_loss": [float(x) for x in sup],
                   "coral_raw": [float(x) for x in raw],
                   "src_feat_l2": [float(x) for x in sl2],
                   "tgt_feat_l2": [float(x) for x in tl2]},
    }

    print("\n  " + "-" * 88)
    print(f"  EQUALISING GAMMA ({run_key}, n_blocks={A['n_blocks']}): "
          f"{g['gamma_equalising']:.6g}")
    print(f"    trailing-{g['tail_steps']} means: sup_loss {g['sup_loss_tail_mean']:.6g}  "
          f"coral_raw {g['coral_raw_tail_mean']:.6g}")
    print(f"    per-step gamma: median {g['per_step_gamma_median']:.6g}  "
          f"sd {g['per_step_gamma_sd']:.3g}  CV {g['per_step_gamma_cv']:.3f}  "
          f"p10-p90 [{g['per_step_gamma_p10']:.6g}, {g['per_step_gamma_p90']:.6g}]")
    print(f"    stability across the tail: first half {g['gamma_first_half']:.6g} vs "
          f"second half {g['gamma_second_half']:.6g}  (ratio {g['half_ratio']:.3f})")
    print(f"    feature norm: {out['src_feat_l2_first10_mean']:.3f} (first 10 steps) -> "
          f"{out['src_feat_l2_tail_mean']:.3f} (tail), unit-gain reference "
          f"{out['feat_l2_unit_gain_reference']:.3f}")
    print("  " + "-" * 88, flush=True)
    return out


def run_calibration(args, root) -> dict:
    arms = ([args.arm] if args.arm else list(RUNS))
    report = {"mode": "calibrate",
              "rule": ("Sun & Saenko arXiv:1607.01719 §4 -- lambda equalises the "
                       "classification and CORAL loss magnitudes at the end of training."),
              "hard_rule": ("LOSS MAGNITUDES ONLY. TPR/AUROC/AUPRC/metric_suite/purity/"
                            "curve_hook and every test_* cache split were made unreachable "
                            "(coral_round4.test_label_lockout) for the whole calibration. "
                            "gamma is never adjusted after seeing a metric."),
              "arms": {}}
    with test_label_lockout():
        for a in arms:
            report["arms"][a] = calibrate_arm(a, args, root)

    if len(report["arms"]) > 1:
        gs = {a: r["gamma_equalising"] for a, r in report["arms"].items()}
        report["gamma_by_arm"] = gs
        report["gamma_ratio_nb3_over_nb0"] = (
            gs.get("coral_nb3", float("nan")) / gs.get("coral_nb0_control", float("nan")))
        report["one_gamma_rule"] = (
            "coral_weight is a PINNED key: BOTH arms must run the same gamma or they "
            "differ in two things and nb3-minus-nb0 stops isolating unfreezing. These "
            "per-arm values are diagnostics; ONE of them (or a declared compromise) must "
            "be chosen, written down, and passed to both --gamma flags.")
    out = os.path.join(root, "runs", "coral_r4_gamma_calibration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[out] {out}")
    return report


# ---------------------------------------------------------------------------
# --grad-check : the reachable-parameter report (cheap, no training)
# ---------------------------------------------------------------------------

def run_grad_check(args, root) -> dict:
    """Backprop the CORAL term alone on each arm and report which parameters it reaches."""
    print("\n" + "=" * 92)
    print("CORAL GRADIENT REACHABILITY -- which parameters can the CORAL term actually move?")
    print("=" * 92)
    print("  train_skeleton.encode_features flags this as an OPEN ITEM: reasoned through")
    print("  for FreeMatch, never for CORAL. Measured here rather than argued.")
    print("=" * 92, flush=True)
    rep = {}
    for arm in ([args.arm] if args.arm else list(RUNS)):
        A = build_arm(arm, args, root, for_calibration=True)
        b = min(8, A["batch_size"])
        src = torch.rand(b, 1, 224, 224)
        tgt = torch.rand(b, 1, 224, 224)
        r = coral_reachable_params(A["trainer"], src, tgt)
        rep[arm] = r
        print(f"\n  [{arm}] n_blocks={r['n_blocks']}")
        print(f"    reachable parameter ELEMENTS : {r['reachable_param_elements']:,}")
        print(f"    reachable tensors            : {r['reachable_tensors']}")
        print(f"    module prefixes reached      : {r['reachable_prefixes']}")
        print(f"    received a grad but ALL-ZERO : "
              f"{sorted(r['zero_grad_tensors'])[:6]}{' ...' if len(r['zero_grad_tensors'])>6 else ''}")
        del A
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    out = os.path.join(root, "runs", "coral_r4_grad_reachability.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"\n[out] {out}")
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=sorted(RUNS), default=None,
                   help="which arm to run (required for a training run)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--gamma", type=float, default=None,
                   help="CONSTANT CORAL weight. Required for a training run; no default. "
                        "Must be identical on both arms (PINNED).")
    p.add_argument("--data-root", default=os.path.join(root, "data"))
    p.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    p.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    p.add_argument("--agreement-ref", default=DEFAULT_AGREEMENT_REF)
    p.add_argument("--init-head-from", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--allow-off-budget", action="store_true")
    p.add_argument("--curve-every", type=int, default=CURVE_EVERY)
    p.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    p.add_argument("--backbone-lr", type=float, default=BACKBONE_LR,
                   help="default None == head LR, which is what Zoobot does (§7.1a)")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--eval-fp16", action="store_true", default=False)
    p.add_argument("--grad-checkpointing", action="store_true", default=True,
                   help="MANDATORY at n_blocks=3 (§7.1a); a refused no-op at n_blocks=0")
    p.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                   action="store_false")
    p.add_argument("--coral-target-ema", dest="coral_target_ema", action="store_true",
                   default=CORAL_TARGET_EMA)
    p.add_argument("--coral-no-target-ema", dest="coral_target_ema", action="store_false")
    p.add_argument("--feat-norm-ceiling", type=float, default=DEFAULT_FEAT_NORM_CEILING,
                   help="HARD ABORT if mean ||f|| on either domain exceeds this "
                        "(measured fp16 covariance failure point is ~126)")
    p.add_argument("--feat-norm-floor", type=float, default=DEFAULT_FEAT_NORM_FLOOR,
                   help="HARD ABORT if mean ||f|| on either domain falls below this "
                        "(the eps^4 collapse direction; init is ~66)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    p.add_argument("--step0-tol", type=float, default=0.0)
    p.add_argument("--smoke-batch-size", type=int, default=None)
    p.add_argument("--dir-suffix", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--emit-config", default=None)
    p.add_argument("--reference-config", default=None)
    # ---- the two no-training diagnostic modes ----
    p.add_argument("--calibrate", action="store_true",
                   help="measure the equalising gamma per arm from LOSS MAGNITUDES ONLY")
    p.add_argument("--calibrate-steps", type=int, default=500)
    p.add_argument("--calibrate-tail", type=int, default=100)
    p.add_argument("--grad-check", action="store_true",
                   help="report the CORAL-reachable parameter set per arm; no training")
    args = p.parse_args(argv)

    if args.calibrate:
        run_calibration(args, root)
        return 0
    if args.grad_check:
        run_grad_check(args, root)
        return 0
    if not args.arm:
        p.error("--arm is required for a training run (or pass --calibrate / --grad-check)")
    train_one(args.arm, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
