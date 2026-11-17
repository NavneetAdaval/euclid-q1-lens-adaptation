"""
run_unfreeze_freematch.py
=========================
FOLD-0 DIAGNOSTIC: FreeMatch x n_blocks=3 (LP->FT), with its own null control.
Implements BRIEF_freematch_nb3_fold0.md against PROJECT.md.

WHAT THIS RUN IS. An EXPLORATORY fold-0 run asking whether self-training gains more from
unfreezing the encoder than source-only training did. PROJECT.md §7.8 screened Lever 1
twice -- both times on **Baseline A**, which has no target-domain training signal at all --
and recorded a documented negative. FreeMatch x nb3 has never been run, so the Lever-1
negative says nothing about it.

WHAT THIS RUN IS NOT. A result for the paper. Fold 0's paired-bootstrap CI half-width is
~0.10-0.12 (§7.8) and cannot resolve anything smaller; §7.8 further showed fold 0 cannot
resolve the unfreeze effect EVEN WITH ORACLE SELECTION. No number from here is reported as
a finding. The output is a DECISION about the diagnostics and step budget for a later
5-fold confirmatory run.

THE TWO RUNS
------------
``nb3_unfrozen``  n_blocks=3, grad-checkpointing ON (mandatory), 8000 steps.
``nb0_control``   n_blocks=0 (frozen), 2000 steps, otherwise identical.

WHY THE CONTROL IS NOT OPTIONAL. The unfrozen run differs from round-1 FreeMatch in TWO
ways: the encoder is unfrozen AND it receives 8000 further steps under a changed optimizer
setup. A confound is not a breakage -- it looks exactly like a result -- so it must be
controlled, not caught. The control isolates the extra-steps leg by taking the same head
through the same further steps with the encoder still frozen. If the control's curve is
flat across 250/500/1000/1500/2000, the extra-steps confound is dead and the unfrozen
result may be compared to round 1 with a citable justification.

  SCHEDULE, NOT BUDGET, IS WHAT MUST MATCH. The control runs 2000 steps but keeps the
  8000-step SCHEDULE (``--schedule-steps 8000``): FreeMatch's mu cosine warmup and its
  hard consistency-warmup gate are both defined as fractions of the planned run, so a
  control with a 2000-step schedule would have a COMPRESSED mu curve and its step 250
  would not be comparable to the unfrozen run's step 250. With the schedule pinned, the
  control is a true TRUNCATION -- identical training, stopped early.

THE PHASE-1 HEAD (BRIEF §2)
---------------------------
Both runs initialise from the converged frozen FreeMatch fold-0 step-8000 head, NOT the
Baseline A linear probe §7.8 used. Kumar et al. (arXiv:2202.10054) prescribe fine-tuning
from a converged linear probe FOR THE OBJECTIVE BEING FINE-TUNED, and FreeMatch's
self-adaptive threshold is an EMA over the model's own recent confidence -- state
calibrated to one particular decision boundary. Starting from Baseline A's head would force
that threshold to recalibrate at exactly the moment 14.8M parameters begin moving and
generating pseudo-labels. [UNVERIFIED -- this is reasoning about how SAT interacts with
LP->FT, not a cited result.]

  ****  THE HARD GATE  ****
  That init buys a FREE KNOWN-NULL: at step 0 the head is the round-1 head and the encoder
  is still the published Zoobot checkpoint, so the model IS the round-1 fold-0 model and
  must reproduce its TPR@1e-3 exactly. Both runs evaluate at step 0 BEFORE training and
  ABORT on a mismatch. Cost of the gate: ~2 minutes. Cost of skipping it: four hours.

  ****  THE GATE VALUE IS 0.387755, NOT THE 0.357 IN THE BRIEF  ****
  The brief cites 0.357 from PROJECT.md §6's Drive listing (``freematch_fold0_standard``),
  which is the earlier fold-0 VARIANT SCREEN -- a different run, on Drive, at a different
  protocol. The matched round-1 run whose fold mean is the reported 0.348 +- 0.024 is
  ``runs/bakeoff_freematch_5fold/``, and re-scoring its fold-0 step_8000 checkpoint against
  the frozen feature cache gives TPR@1e-3 = 0.387755 (AUROC 0.973624, 98 positives vs
  26,399 shared negatives), matching that run's own results.json. Since THAT is the
  checkpoint this run initialises from, 0.387755 is the value step 0 must reproduce.
  Override with --expect-step0-tpr if a different phase-1 head is ever used.

EVAL PATH. At n_blocks>0 the frozen feature cache is invalid by construction (§7.1a:
scoring a real nb3 checkpoint from the stale cache gave 0.194 against a logged 0.020), so
every curve point re-encodes val + test_pos_fold0 + test_negatives_shared + target_fold0
LIVE with the current encoder. The live encode runs in **fp32** by default, which is what
makes the step-0 gate an exact comparison (live_encode reproduces the cache byte-for-byte
at n_blocks=0 in fp32). Training precision is unaffected and stays fp16.

Run:
    python scripts/run_unfreeze_freematch.py --run nb3_unfrozen --fp16 --grad-checkpointing
    python scripts/run_unfreeze_freematch.py --run nb0_control  --fp16
    python scripts/run_unfreeze_freematch.py --run nb3_unfrozen --dry-run
    python scripts/run_unfreeze_freematch.py --run nb0_control --emit-config runs/ref_nb0.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import evaluate as ev                                       # noqa: E402
from heads import LINEAR                                    # noqa: E402
from live_encode import LiveEncoder                         # noqa: E402
from metrics import (metric_suite, DEFAULT_TARGET_FPR)      # noqa: E402
from unfreeze_instrument import CurveInstrument             # noqa: E402
from run_screen_unfreeze import assert_no_bf16, _normalise_cfg   # noqa: E402  (reuse)

# ---- ROUND-1 FREEMATCH SETTINGS -- inherited UNCHANGED (BRIEF §6) ----
# "Keep round-1 FreeMatch settings unchanged so the encoder LR is the ONLY new
# hyperparameter introduced." These are run_bakeoff_freematch.py's values verbatim; they
# are NOT the Zoobot-defaults protocol match §7.8 adopted for Baseline A. §7.8 measured
# weight_decay 0.05 vs 1e-4 (500x) as changing nothing at 14.8M trainable params, so there
# is no evidence-backed reason to revisit them -- and that evidence is from Baseline A, not
# FreeMatch, so revisiting them would introduce a second uncontrolled variable.
BUDGET_STEPS = 8000
# The control arm tracks the confirmatory budget so a paired control is §1-comparable.
# The fold-3 spot check ALREADY RUN at 2000 is historical and stays valid -- it validated
# continuation-equivalence, which §7.11 records as budget-independent.
CONTROL_STEPS = 1500
CURVE_EVERY = 125          # 64 curve points (BRIEF §5.2/§5.3)
CKPT_EVERY = 500           # 16 retained weight checkpoints (BRIEF §5.1)
BATCH_SIZE = 256           # TRUE batch, never micro-batched (§7.1a)
LR = 1e-3                  # head LR, round-1 FreeMatch
WEIGHT_DECAY = 1e-4        # round-1 FreeMatch
POS_FRACTION = 0.5
SEED = 42
HEAD_TYPE = LINEAR
DEFAULT_FOLD = 0
LR_DECAY = 0.75            # Zoobot default (unchanged)

# FreeMatch published-default knobs -- identical to run_bakeoff_freematch.py
MU_MAX = 1.0
SAF_WEIGHT = 0.01
SAT_EMA_MOMENTUM = 0.999
CONSISTENCY_WARMUP_FRAC = 0.1
FAIRNESS_TARGET = None
POS_CLASS = 1
FREEMATCH_LOCAL_MODE = "standard"      # the screen-winning variant; "floor" is a negative

# ---- THE ONE NEW HYPERPARAMETER ----
# LP->FT wants the encoder to move slowly relative to the (already converged) head, so the
# pretrained features are refined rather than distorted (Kumar arXiv:2202.10054). 1e-4 is
# head_lr/10 and is exactly the backbone LR §7.8's screen 1 used, so it is an inherited
# value rather than a fresh guess. It is not tuned: no validated label-free selection
# signal exists here (§7.6), and any sweep would be tuned on fold-0 test data (§7.8).
BACKBONE_LR = 1e-4

# ---- phase-1 head: the MATCHED round-1 FreeMatch fold-0 run (see the module docstring) ----
ROUND1_FM_DIR = os.path.normpath(os.path.join(_HERE, "..", "runs",
                                              "bakeoff_freematch_5fold", "fold0"))
ROUND1_FM_CKPT = os.path.join(ROUND1_FM_DIR, "checkpoints", "step_8000.pt")

# ---- PER-FOLD GATE TABLE -- PRE-REGISTERED (PROJECT.md §7.11, frozen 2026-08-04) ----
# Each fold continues from ITS OWN round-1 checkpoint, so each has its own step-0 known-null.
# Baking the table in is a defence against the failure mode a single shared constant invites:
# pointing the run at fold 1's head while gating on fold 0's number would not abort -- it
# would train happily and produce a plausible, wrong result. With the table, a wrong --fold
# fails at the gate in two minutes.
# Values re-scored from runs/bakeoff_freematch_5fold/fold{k}/checkpoints/step_8000.pt against
# data/feature_cache; TPRs match that run's own results.json.
FOLD_GATE = {
    0: {"tpr": 0.3877551020408163, "head_l2": 29.414692, "auroc": 0.973624},
    1: {"tpr": 0.336734693877551,  "head_l2": 29.280975, "auroc": 0.963311},
    2: {"tpr": 0.35714285714285715, "head_l2": 29.292334, "auroc": 0.963068},
    3: {"tpr": 0.3163265306122449, "head_l2": 29.465052, "auroc": 0.940989},
    4: {"tpr": 0.3402061855670103, "head_l2": 29.237820, "auroc": 0.963559},
}

# ---- GATE TOLERANCES, set from MEASUREMENT (2026-08-06) ----
# The gate originally demanded TPR@1e-3 reproduce EXACTLY. That premise is false for the
# LIVE-ENCODE path and it cost a fold-2 session to discover:
#
#   * the encoder forward is not bit-reproducible across a live re-encode vs the cached
#     features -- MEASURED on folds 0 and 1's own step-0 dumps, |live - cached| score
#     differences reach ~1e-2 (mean ~2-3e-3);
#   * TPR@FPR=1e-3 is a THRESHOLD-CROSSING metric, so any positive sitting within that
#     envelope of the threshold can flip, moving the metric by exactly one positive;
#   * fold 2's closest positive sits 3.06e-03 above its threshold -- a relative margin 30x
#     tighter than any other fold, i.e. squarely inside the drift envelope. Folds 0 and 1
#     passed at tolerance 0 only because their margins are 1.0e-1 and 2.4e-1.
#
# PROJECT.md §7.7 already recorded venue non-portability at 0.0204 = TWO positives, so a
# tolerance-0 gate contradicted the project's own measurement. ONE POSITIVE is the metric's
# own quantum and the tightest defensible slack.
#
# The gate is nonetheless made STRONGER, not weaker, by adding AUROC -- which is
# rank-based and NOT threshold-sensitive. MEASURED separation:
#     live-encode drift in AUROC : 1.2e-05 and 7.3e-06  (folds 0, 1)
#     WRONG head on this fold    : 7.6e-03              (650x larger)
# so 1e-3 passes legitimate drift and catches the wrong-model failure the TPR gate exists
# for. A wrong head can even hit a plausible TPR by chance (fold 1's head on fold 2's data
# gives exactly 0.336735); it cannot fake the AUROC.
ONE_POSITIVE_TOL = 1.0 / 98.0     # 0.010204 -- the finest difference 98 positives express
STEP0_AUROC_TOL = 1e-3            # ~100x the measured drift, ~7x tighter than a wrong head
# round-1 folds-1-4 mean TPR@1e-3 = 0.337603 -- the confirmatory comparison arm (§7.11)
ROUND1_FOLDS_1_4_MEAN = 0.337603
EXPECT_STEP0_TPR = FOLD_GATE[DEFAULT_FOLD]["tpr"]
EXPECT_HEAD_L2 = FOLD_GATE[DEFAULT_FOLD]["head_l2"]


def round1_fold_ckpt(fold):
    """That fold's own round-1 FreeMatch step-8000 checkpoint."""
    return os.path.normpath(os.path.join(
        _HERE, "..", "runs", "bakeoff_freematch_5fold", f"fold{int(fold)}",
        "checkpoints", "step_8000.pt"))

# Baseline A step_8000 -- the agreement column's reference vector (§7.6). Read-only.
DEFAULT_AGREEMENT_REF = os.path.normpath(
    os.path.join(_HERE, "..", "runs", "bakeoff_baseline_A", "checkpoints", "step_8000.pt"))

# CONFIRMATORY_STEPS is the PRE-REGISTERED budget (PROJECT.md §7.11). It is the run's
# BUILT-IN default, deliberately: routing it through --max-steps triggers the _SMOKE
# directory suffix (a guard against a short run occupying a real run's dir) and would stamp
# "not the pre-registered budget" provenance onto every confirmatory fold.
CONFIRMATORY_STEPS = 1500   # AMENDMENT 1, 2026-08-06 (was 2000)

RUNS = {
    "nb3_unfrozen": {"n_blocks": 3, "steps": CONFIRMATORY_STEPS,
                     "dir": "freematch_nb3_unfrozen"},
    "nb0_control":  {"n_blocks": 0, "steps": CONTROL_STEPS,
                     "dir": "freematch_nb0_control"},
}

# ---------------------------------------------------------------------------
# §1 -- CONFOUND PINNING
# ---------------------------------------------------------------------------
# Everything here must MATCH between the two runs or the comparison measures something
# other than unfreezing. This is §7.8's 15-key check, re-derived for FreeMatch: the four
# Baseline-A-only keys are gone and every FreeMatch knob is added, because a silent change
# to local_threshold_mode alone is worth ~0.18 TPR (0.357 standard vs 0.174 floor, §7.4).
PINNED_KEYS = ("augmentation", "precision", "seed", "batch_size",
               "lr", "lr_decay", "weight_decay", "pos_fraction", "head_type", "fold",
               "curve_every", "ckpt_every", "true_batch", "gradient_accumulation",
               "init_head_from", "init_head_l2", "schedule_total_steps",
               "method", "local_threshold_mode", "mu_max", "saf_weight",
               "sat_ema_momentum", "consistency_warmup_frac", "fairness_target",
               "pos_class", "eval_precision",
               # the continuation-state reconstruction must be identical in both arms, or
               # they resumed from different training states -- the very confound the
               # control exists to remove.
               "mu_step_offset", "sat_rebuild", "sat_passes", "adam_warmup_steps",
               # PINNED as of 2026-08-05 (see the EXEMPT_KEYS audit note)
               "max_steps", "budget_steps")

# EXEMPT, each for a stated reason -- never widen this list to make a run pass.
#   n_blocks           THE variable under test.
#   grad_checkpointing pure memory<->recompute, verified numerically exact (§7.1a test 8);
#                      mandatory at nb3, a refused no-op at nb0.
#   max_steps          NEW THIS ROUND (BRIEF §6). 8000 vs 2000 is the control's whole
#                      point. It is safe to exempt ONLY because schedule_total_steps is
#                      PINNED above: the two runs share one mu/warmup schedule and differ
#                      solely in where they stop.
#   backbone_lr*       exists only in the unfrozen run; the frozen run has no encoder to
#                      give an LR to, so there is no counterpart to match (BRIEF §6).
#   encoder_eval       follows n_blocks (cache is valid iff frozen), not an independent knob.
# AUDITED 2026-08-05. `max_steps` / `budget_steps` were exempt while the control ran 2000
# against the unfrozen arm's 8000 (§7.10). Both arms now run the pre-registered 2000, so
# that exemption is STALE and has been REMOVED -- they are PINNED below. Every remaining
# entry is justified inline; an exemption with no live reason is how a guard rots into a
# rubber stamp.
EXEMPT_KEYS = ("n_blocks",              # THE variable under test
               "grad_checkpointing",    # memory<->recompute, numerically exact (§7.1a)
               "backbone_lr",           # exists only in the unfrozen arm
               "backbone_lr_effective",
               "encoder_eval",          # follows n_blocks (cache valid iff frozen)
               "epochs_planned",        # derived from max_steps, which is now pinned
               "run", "run_dir",        # naming
               "device", "num_workers") # throughput only; seeded sampler is unaffected


def check_preconditions(cfg_json, reference_config, verbose=True):
    """REFUSE to start unless every pinned key matches the declared reference run."""
    if not os.path.isfile(reference_config):
        raise SystemExit(
            f"[unfreeze-fm] ABORT: reference config not found: {reference_config}\n"
            f"  This run cannot verify that only n_blocks (and the exempt keys) differ, so "
            f"it will not start. Emit the other run's config first:\n"
            f"    python scripts/run_unfreeze_freematch.py --run nb0_control "
            f"--emit-config <path>")
    with open(reference_config) as fh:
        ref = _normalise_cfg(json.load(fh))
    cfg_json = _normalise_cfg(cfg_json)

    rows, bad = [], []
    for k in PINNED_KEYS:
        rv, cv = ref.get(k, "<absent>"), cfg_json.get(k, "<absent>")
        ok = (rv == cv)
        rows.append((k, rv, cv, ok))
        if not ok:
            bad.append(k)
    for k in EXEMPT_KEYS:
        rows.append((k, ref.get(k, "<absent>"), cfg_json.get(k, "<absent>"), None))

    if verbose:
        print("\n" + "=" * 92)
        print("§1 PRECONDITION CHECK -- only n_blocks, grad_checkpointing and max_steps "
              "may differ")
        print(f"  reference: {reference_config}")
        print("=" * 92)
        print(f"  {'key':<24} | {'reference':<28} | {'this run':<22} | status")
        print("  " + "-" * 88)
        for k, rv, cv, ok in rows:
            status = "EXEMPT" if ok is None else ("ok" if ok else "*** MISMATCH ***")
            print(f"  {k:<24} | {str(rv)[:28]:<28} | {str(cv)[:22]:<22} | {status}")
        print("  " + "-" * 88)
    if bad:
        raise SystemExit(
            f"\n[unfreeze-fm] ABORT: {len(bad)} pinned setting(s) differ from the "
            f"reference run: {bad}.\n"
            f"  The unfrozen run and its control may differ ONLY in n_blocks, "
            f"grad_checkpointing and max_steps. Augmentation+precision alone is worth "
            f"-0.0714 TPR on this metric (§7.3) and local_threshold_mode is worth ~0.18 "
            f"(§7.4), either of which would swamp the effect under test.")
    if verbose:
        print("  ALL PINNED SETTINGS MATCH.\n")
    return rows


# ---------------------------------------------------------------------------
# checkpoint retention (Task-5 guard, re-parameterised for ckpt_every != curve_every)
# ---------------------------------------------------------------------------

def expected_checkpoint_steps(budget, ckpt_every, include_step0=True):
    """The steps at which a RETAINED checkpoint must exist.

    Unlike §7.8's version this is keyed to ``ckpt_every`` rather than the curve density:
    BRIEF §5.1 retains weights every 500 steps while the curve runs every 125, because the
    derived-logit dump (§5.2) now supplies what the per-curve-point weights were being kept
    for. Step 0 is included -- it is the loaded phase-1 model and the gate's evidence.
    """
    steps = list(range(int(ckpt_every), int(budget) + 1, int(ckpt_every)))
    return ([0] + steps) if include_step0 else steps


def assert_checkpoint_retention(ckpt_dir, curve_csv, budget, ckpt_every, curve_every,
                                verbose=True):
    """FAIL LOUDLY unless every retained checkpoint AND every curve row is present.

    PROJECT.md §7.7: ``nb1``/``nb3``/``nb0_image_kaggle`` came home with only step_8000.pt
    and that cost the A1 study its unfrozen checkpoints. The trainer was never at fault --
    the loss happened in the Kaggle COLLECTION cell. So this runs at the END OF TRAINING,
    where it proves the files exist on the training box, and the packaging step re-checks by
    content hash that they survived into the archives. One check alone would not have caught
    the failure that actually occurred.
    """
    want = expected_checkpoint_steps(budget, ckpt_every)
    have = sorted(int(m.group(1)) for m in
                  (re.fullmatch(r"step_(\d+)\.pt", f)
                   for f in os.listdir(ckpt_dir) if f.endswith(".pt")) if m)
    missing = [s for s in want if s not in set(have)]

    n_curve = 0
    if os.path.isfile(curve_csv):
        with open(curve_csv, newline="") as fh:
            n_curve = max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    want_curve = len(range(int(curve_every), int(budget) + 1, int(curve_every))) + 1  # +step0

    if verbose:
        print(f"[retention] checkpoints {len(have)}/{len(want)}  "
              f"curve rows {n_curve}/{want_curve}")
    if missing:
        raise SystemExit(
            f"[unfreeze-fm] ABORT: checkpoint retention failed -- {len(missing)} of "
            f"{len(want)} retained checkpoints missing from {ckpt_dir}: "
            f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if n_curve and n_curve != want_curve:
        raise SystemExit(
            f"[unfreeze-fm] ABORT: {n_curve} curve rows but {want_curve} expected -- "
            f"curve.csv and the run's step schedule disagree, so no per-step series can be "
            f"aligned to the TPR series.")
    return have


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def build_config(run_key, args, root, fold=None):
    """The run_config.json dict, built ONCE so --emit-config and the real run cannot drift.

    (§7.7: --emit-config exists so a matched pair can run CONCURRENTLY on T4 x2 -- the
    second rung validates against the first's declared config at minute 2 instead of after
    four hours. It is not a hole in §1: emitting trains nothing, and the packaging step
    re-checks the declared reference against what each run actually emitted.)
    """
    spec = RUNS[run_key]
    fold = int(args.fold if fold is None else fold)
    if fold not in FOLD_GATE:
        raise SystemExit(f"[unfreeze-fm] --fold must be one of {sorted(FOLD_GATE)}, got {fold}")
    n_blocks = spec["n_blocks"]
    max_steps = int(args.max_steps) if args.max_steps else spec["steps"]
    grad_ckpt = bool(args.grad_checkpointing) and n_blocks > 0
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    head_ckpt = os.path.abspath(args.init_head_from or round1_fold_ckpt(fold))
    batch_size = int(args.smoke_batch_size or BATCH_SIZE)

    cfg = {
        "run": run_key, "run_dir": f'{spec["dir"]}_fold{fold}', "n_blocks": n_blocks,
        "fold": fold,
        "lp_ft": True, "init_head_from": head_ckpt,
        "init_head_l2": None,                       # filled after the head is loaded
        "max_steps": max_steps, "budget_steps": max_steps,
        "schedule_total_steps": int(args.schedule_steps or BUDGET_STEPS),
        "is_preregistered_budget": (max_steps == spec["steps"]
                                    and not args.smoke_batch_size),
        "curve_every": int(args.curve_every), "ckpt_every": int(args.ckpt_every),
        "batch_size": batch_size, "true_batch": True, "gradient_accumulation": False,
        "batch_size_is_protocol": (batch_size == BATCH_SIZE),
        "grad_checkpointing": grad_ckpt,
        "lr": LR, "backbone_lr": (BACKBONE_LR if n_blocks > 0 else None),
        "backbone_lr_effective": (args.backbone_lr if args.backbone_lr is not None
                                  else BACKBONE_LR) if n_blocks > 0 else None,
        "lr_decay": LR_DECAY, "weight_decay": WEIGHT_DECAY,
        "pos_fraction": POS_FRACTION, "seed": SEED, "head_type": HEAD_TYPE, "fold": fold,
        "selection_rule": f"R1_last_checkpoint_step{max_steps}",
        "method": "freematch", "domain_adaptation": "freematch",
        "local_threshold_mode": FREEMATCH_LOCAL_MODE,
        "freematch_variant": f"FreeMatch-{FREEMATCH_LOCAL_MODE}",
        "mu_max": MU_MAX, "saf_weight": SAF_WEIGHT,
        "sat_ema_momentum": SAT_EMA_MOMENTUM,
        "consistency_warmup_frac": CONSISTENCY_WARMUP_FRAC,
        "fairness_target": FAIRNESS_TARGET, "pos_class": POS_CLASS,
        # ---- continuation-state reconstruction: PINNED, identical in both arms ----
        # These three are what a weights-only round-1 checkpoint failed to carry. They are
        # pinned (not exempt) because a difference between the arms would mean the two runs
        # resumed from different training states, which is exactly the confound the control
        # exists to remove.
        "mu_step_offset": int(args.mu_step_offset),
        "sat_rebuild": (not args.no_rebuild_sat),
        "sat_passes": int(args.sat_passes),
        "adam_warmup_steps": int(args.adam_warmup_steps),
        "continuation_note": (
            "reconstruction, NOT restoration: round 1 saved head+standardizer only, so "
            "tau_global/pbar and AdamW's moments were never on disk. Each is rebuilt to "
            "its fixed point for THIS model on THIS data; whether that matches round 1's "
            "exact step-8000 state is UNVERIFIED."),
        "encoder_eval": ("frozen_feature_cache" if n_blocks == 0 else "live_encode"),
        "augmentation": "weak (+-3px translate, PROJECT.md 7.3)",
        "fp16": bool(args.fp16), "precision": ("fp16" if args.fp16 else "fp32"),
        # eval precision is PINNED (not exempt): the live encode and the cached path must
        # agree numerically or the step-0 gate is not an exact comparison.
        "eval_precision": ("fp16" if args.eval_fp16 else "fp32"),
        "device": device, "num_workers": int(args.num_workers),
        "agreement_ref": os.path.abspath(args.agreement_ref),
        "expect_step0_tpr": float(args.expect_step0_tpr
                                  if args.expect_step0_tpr is not None
                                  else FOLD_GATE[fold]["tpr"]),
        "expect_step0_auroc": FOLD_GATE[fold]["auroc"],
        # the eval path decides the tolerance: the cached path is deterministic, the live
        # re-encode is not (MEASURED drift ~1e-2 in score -> up to one positive on a
        # threshold metric). Set here, not by a flag, so it cannot be loosened by habit.
        "step0_tpr_tolerance": (0.0 if n_blocks == 0 else ONE_POSITIVE_TOL),
        "step0_auroc_tolerance": STEP0_AUROC_TOL,
        "expect_head_l2": float(args.expect_head_l2
                                if args.expect_head_l2 is not None
                                else FOLD_GATE[fold]["head_l2"]),
        "hard_gate": ("step-0 TPR@1e-3 must reproduce the phase-1 checkpoint's own "
                      "fold-0 number; abort otherwise (BRIEF §2)"),
        "brief": "BRIEF_freematch_nb3_fold0.md",
        "exploratory": True,
        "reporting_note": ("fold-0 CI half-width ~0.10-0.12 (§7.8): this run cannot "
                           "resolve the effect and reports a decision, not a finding"),
    }
    return cfg, spec, max_steps, grad_ckpt, device, head_ckpt, fold


def head_l2_norm(ckpt_path):
    """L2 norm of the phase-1 head -- recorded in run_config.json and asserted identical
    across the two runs, as §7.8 did with ||head||_2 = 28.475603 for the Baseline A probe."""
    head, _std, _m = ev.load_checkpoint(ckpt_path, "cpu")
    return float(torch.sqrt(sum((p.detach().float() ** 2).sum()
                                for p in head.parameters())))


def train_one(run_key, args):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    cfg_json, spec, max_steps, grad_ckpt, device, head_ckpt, fold = build_config(
        run_key, args, root)
    n_blocks = spec["n_blocks"]
    batch_size = int(cfg_json["batch_size"])

    suffix = args.dir_suffix or ""
    if not cfg_json["is_preregistered_budget"] and not suffix:
        suffix = f"_SMOKE{max_steps}"      # a short run may never occupy the real run's dir
    out_dir = os.path.abspath(os.path.join(root, "runs",
                                          cfg_json["run_dir"] + suffix))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    curve_csv = os.path.join(out_dir, "curve.csv")
    splits_path = os.path.abspath(args.splits)
    data_root = os.path.abspath(args.data_root)
    cache_dir = os.path.abspath(args.cache_dir)
    agreement_ref = os.path.abspath(args.agreement_ref)

    assert_no_bf16(cfg_json["precision"], device)

    # ---- phase-1 head must exist and load BEFORE anything else (BRIEF §2 precondition) ----
    if not os.path.isfile(head_ckpt):
        raise SystemExit(
            f"[unfreeze-fm] ABORT: phase-1 head not found: {head_ckpt}\n"
            f"  Both runs initialise from the converged round-1 FreeMatch fold-0 step_8000\n"
            f"  head. §7.7 records a packaging step that copied only step_8000.pt for some\n"
            f"  runs and an A1 study that lost its checkpoints outright -- so this file is\n"
            f"  verified, never assumed. runs/ is not packed into data.zip, so on a fresh\n"
            f"  Kaggle session point --init-head-from straight at it.")
    cfg_json["init_head_l2"] = round(head_l2_norm(head_ckpt), 6)

    print("=" * 92)
    print(f"FREEMATCH x UNFREEZE -- {spec['dir'] + suffix}  (fold {fold}, {max_steps} steps,"
          f" n_blocks={n_blocks})")
    print("=" * 92)
    for k, v in cfg_json.items():
        print(f"  {k:<24}: {v}")
    print("=" * 92, flush=True)

    # ---- --emit-config: declare this run's config WITHOUT training (returns before §1) ----
    if args.emit_config:
        p = os.path.abspath(args.emit_config)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as fh:
            json.dump(cfg_json, fh, indent=2)
        print(f"[emit-config] declared reference for {run_key} -> {p}")
        print("[emit-config] nothing was trained. Pass it to the other run as "
              "--reference-config.")
        return None

    # ---- §1 -- runs BEFORE the dry-run return so --dry-run validates the real config ----
    if args.reference_config:
        check_preconditions(cfg_json, os.path.abspath(args.reference_config))
    else:
        print("[§1] no --reference-config given -- this run DECLARES the reference. Emit it "
              "with --emit-config and pass it to the other run.\n")

    if not os.path.isfile(agreement_ref):
        raise SystemExit(
            f"[unfreeze-fm] ABORT: agreement reference not found: {agreement_ref}\n"
            f"  The collapse gate's agreement column is Spearman(this checkpoint, Baseline "
            f"A step_8000) on the fold's target pool (§7.6) and is only comparable across "
            f"runs if every run uses the SAME reference file.")
    # ALWAYS checked, against the PRE-REGISTERED per-fold value (§7.11) -- never against a
    # CLI flag that defaults to None. This is the check that catches "--fold 1 but fold 0's
    # head", which would otherwise train happily and produce a plausible wrong number.
    if abs(cfg_json["init_head_l2"] - cfg_json["expect_head_l2"]) > 1e-4:
        raise SystemExit(
            f"[unfreeze-fm] ABORT: phase-1 head ||.||_2 = {cfg_json['init_head_l2']} but "
            f"the PRE-REGISTERED value for fold {fold} is {cfg_json['expect_head_l2']}. "
            f"This is not fold {fold}'s phase-1 head -- check --fold and --init-head-from.")

    # ---- BUDGET PRECONDITION -- ABORTS, it does not merely display ----
    # On fold 1 the progress counters showed ckpt 8/5 and curve 28/17 for three and a half
    # hours because the effective budget was 8000 while the pre-registration says 2000, and
    # nothing stopped: a passive display is not a guard. This runs BEFORE training and
    # refuses. --allow-off-budget exists for a deliberate diagnostic and says so in the
    # config, so an off-budget run can never be mistaken for a confirmatory one.
    want_budget = CONFIRMATORY_STEPS if run_key == "nb3_unfrozen" else CONTROL_STEPS
    cfg_json["preregistered_budget"] = want_budget
    cfg_json["on_preregistered_budget"] = (max_steps == want_budget)
    if max_steps != want_budget and not args.allow_off_budget:
        raise SystemExit(
            f"[unfreeze-fm] ABORT: effective budget is {max_steps} steps but the "
            f"PRE-REGISTERED budget for {run_key} is {want_budget} (PROJECT.md §7.11).\n"
            f"  A confirmatory fold must run exactly {want_budget}. Fold 1 ran 8000 by "
            f"accident and only the wall clock revealed it.\n"
            f"  If this is a deliberate diagnostic, pass --allow-off-budget; the config "
            f"will record on_preregistered_budget=false so it can never be aggregated as a "
            f"confirmatory result.")
    if max_steps != want_budget:
        print(f"[budget] WARNING: {max_steps} steps against a pre-registered {want_budget}. "
              f"--allow-off-budget given, so this is DECLARED off-budget and is recorded as "
              f"such. It is not a confirmatory result.")

    if args.dry_run:
        print("[dry-run] config validated; nothing trained.")
        return None

    if os.path.exists(curve_csv):
        raise SystemExit(f"[unfreeze-fm] ABORT: {curve_csv} already exists -- appending "
                         f"would corrupt the curve. Move/delete {out_dir} first.")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- the agreement reference vector (Baseline A on this fold's target pool) ----
    # Computed from the FROZEN cache, which is correct: the reference is a FIXED vector
    # produced by a frozen model, and re-encoding it with THIS run's drifting encoder would
    # make the agreement column measure two moving things at once. Only the CANDIDATE side
    # is live-encoded (§7.6, §7.7's --from-curve fix).
    from run_bakeoff_coral import compute_agreement_refs
    print(f"[ref] scoring Baseline A on target_fold{fold} for the agreement column ...",
          flush=True)
    agreement_refs = compute_agreement_refs(cache_dir, agreement_ref, [fold], device="cpu")

    # ---- encoder built HERE with the run's n_blocks (TrainConfig refuses a mismatch) ----
    from zoobot_encoder import ZoobotEncoder
    print(f"[load] building the Zoobot encoder (n_blocks={n_blocks}, "
          f"grad_checkpointing={grad_ckpt}) ...", flush=True)
    encoder = ZoobotEncoder(encoder_mode="frozen", n_blocks=n_blocks, device=device,
                            verbose=False, grad_checkpointing=grad_ckpt)

    # ---- the instrumentation (BRIEF §5) ----
    def live_factory():
        if n_blocks == 0:
            return None                     # frozen cache is valid and byte-identical
        return LiveEncoder(splits_path, data_root, fold, batch_size=batch_size,
                           num_workers=0, device=device, verbose=True)

    instrument = CurveInstrument(out_dir, fold, live_encoder_factory=live_factory,
                                 fp16_eval=bool(args.eval_fp16))

    steps_per_epoch = None                  # len(src)//batch, as round 1
    # 48000 source images / 256 -> 187 steps/epoch (PROJECT.md §7.4). Rounded UP, and the
    # loop stops on max_steps anyway, so an over-estimate costs nothing.
    epochs_planned = max(1, int(math.ceil(max_steps / max(48000 // batch_size, 1))))
    cfg_json["epochs_planned"] = epochs_planned

    from run_freematch import train_freematch_fold
    # peak memory must be measured INSIDE the training process -- a notebook cell that
    # launched this via subprocess has its own CUDA context and would read 0. §7.1a's
    # 4.76 GB anchor is only checkable if the number travels in results.json.
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    checkpoints, stability = train_freematch_fold(
        splits_path, data_root, cache_dir, fold, ckpt_dir, encoder=encoder,
        head_type=HEAD_TYPE, epochs=(epochs_planned,), batch_size=batch_size,
        pos_fraction=POS_FRACTION, lr=LR, weight_decay=WEIGHT_DECAY,
        steps_per_epoch=steps_per_epoch,
        mu_max=MU_MAX, saf_weight=SAF_WEIGHT, sat_ema_momentum=SAT_EMA_MOMENTUM,
        consistency_warmup_frac=CONSISTENCY_WARMUP_FRAC,
        local_threshold_mode=FREEMATCH_LOCAL_MODE, pos_class=POS_CLASS,
        fairness_target=FAIRNESS_TARGET, seed=SEED, device=device,
        num_workers=int(args.num_workers), fp16=bool(args.fp16),
        eval_every_steps=int(args.curve_every), curve_csv=curve_csv, curve_folds=(fold,),
        agreement_refs={fold: agreement_refs[fold]}, eval_target_fpr=args.target_fpr,
        eval_device="cpu", verbose=True,
        # ---- the new arguments ----
        n_blocks=n_blocks, backbone_lr=(BACKBONE_LR if n_blocks > 0 else None),
        lr_decay=LR_DECAY, init_head_from=head_ckpt, grad_checkpointing=grad_ckpt,
        schedule_steps=cfg_json["schedule_total_steps"], max_steps=max_steps,
        ckpt_every=int(args.ckpt_every), instrument=instrument, eval_at_step0=True,
        # ---- CONTINUATION-STATE RECONSTRUCTION (identical in both arms) ----
        mu_step_offset=int(args.mu_step_offset),
        rebuild_sat=(not args.no_rebuild_sat), sat_passes=int(args.sat_passes),
        sat_ema_iters=int(args.sat_ema_iters),
        adam_warmup_steps=int(args.adam_warmup_steps),
        # THE HARD GATE -- raises inside the trainer, before the first optimizer step.
        step0_check=lambda row: gate_or_abort(
            row, cfg_json["expect_step0_tpr"],
            max(args.step0_tol, cfg_json["step0_tpr_tolerance"]), run_key, fold=fold,
            expected_auroc=cfg_json["expect_step0_auroc"]))
    train_secs = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 2 ** 30
               if (device == "cuda" and torch.cuda.is_available()) else None)
    if peak_gb is not None:
        print(f"[mem] peak GPU allocated = {peak_gb:.2f} GB   "
              f"(§7.1a anchor for FreeMatch nb3 at true batch 256: 4.76 GB with "
              f"gradient checkpointing; OOM on a 14.56 GiB T4 without)")
    cfg_json["peak_gpu_gb"] = peak_gb
    cfg_json["step0_gate"] = gate = check_step0_gate(
        stability.get("step0_row"), cfg_json["expect_step0_tpr"],
        max(args.step0_tol, cfg_json["step0_tpr_tolerance"]), run_key, fold=fold,
        expected_auroc=cfg_json["expect_step0_auroc"], verbose=False)                    # already printed by the in-loop gate

    r1 = os.path.join(ckpt_dir, f"step_{max_steps}.pt")
    if not os.path.isfile(r1):
        raise SystemExit(f"[unfreeze-fm] ABORT: R1 checkpoint {r1} was not written "
                         f"({max_steps} must be a multiple of --ckpt-every).")
    kept = assert_checkpoint_retention(ckpt_dir, curve_csv, max_steps,
                                       int(args.ckpt_every), int(args.curve_every))
    print(f"[train] done in {train_secs/60:.1f} min", flush=True)

    # ---- final evaluation at R1 (single candidate => R1 enforced structurally) ----
    res = evaluate_r1(r1, n_blocks, encoder, instrument, cache_dir, args.target_fpr,
                      bool(args.eval_fp16), out_dir, cfg_json["run_dir"], fold=fold)
    res.update({"run": run_key, "run_dir": f'{spec["dir"]}_fold{fold}', "n_blocks": n_blocks,
        "fold": fold,
                "train_minutes": train_secs / 60.0, "steps": max_steps,
                "step0_gate": gate, "freematch_stability": stability,
                "n_checkpoints_saved": len(kept), "peak_gpu_gb": peak_gb})

    # §5.5 -- required output #5, produced by the run itself (see run_collapse_gate)
    res["collapse_gate"] = run_collapse_gate(out_dir, fold, agreement_ref, cache_dir,
                                             n_blocks)

    cfg_json["n_checkpoints_expected"] = len(
        expected_checkpoint_steps(max_steps, int(args.ckpt_every)))
    cfg_json["n_checkpoints_saved"] = len(kept)
    cfg_json["instrument"] = instrument.summary()
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(cfg_json, fh, indent=2)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"[out] run_config.json + results.json + curve.csv + diagnostics.csv + "
          f"memorization.csv + logits/ + scores/ -> {out_dir}")
    return res


# ---------------------------------------------------------------------------
# the step-0 hard gate
# ---------------------------------------------------------------------------

def check_step0_gate(step0_row, expected, tol, run_key, verbose=True, fold=0,
                     expected_auroc=None, auroc_tol=STEP0_AUROC_TOL):
    """Compare step 0 against the phase-1 checkpoint's own numbers, on TWO metrics.

    TPR@1e-3 is a THRESHOLD-CROSSING metric and is therefore knife-edge sensitive: a
    positive sitting within the encoder's live-vs-cached drift envelope (~1e-2 in score,
    MEASURED) flips, moving the metric by exactly one positive. Fold 2's closest positive
    sits 3.06e-03 above its threshold, so it flips; folds 0 and 1 sit 1.0e-1 and 2.4e-1
    clear, so they do not. A tolerance-0 TPR gate therefore fails on model identity it
    cannot actually distinguish -- and contradicts §7.7's own recorded venue
    non-portability of 0.0204.

    AUROC is rank-based and NOT threshold-sensitive, so it carries the identity check the
    TPR gate was meant to provide. MEASURED: live-encode drift moves it 1.2e-05, the WRONG
    fold's head moves it 7.6e-03. Checking both makes the gate STRICTLY STRONGER than the
    original -- a wrong head can hit a plausible TPR by chance (fold 1's head on fold 2's
    data gives exactly 0.336735) but cannot fake the AUROC.
    """
    if not step0_row:
        return {"passed": None, "reason": "no step-0 row was computed"}
    got = step0_row.get(f"fold{fold}_tpr_1e-3")
    got_auc = step0_row.get(f"fold{fold}_auroc")
    # +1e-9: the tolerance IS one positive and the observed diff IS one positive, but
    # 35/98 - 34/98 does not land exactly on 1.0/98.0 in binary floating point. Without the
    # epsilon the gate rejects the very case it was widened to admit.
    tpr_ok = (got is not None) and (abs(float(got) - float(expected)) <= float(tol) + 1e-9)
    if expected_auroc is None:
        auc_ok, auc_d = None, None
    else:
        auc_d = None if got_auc is None else abs(float(got_auc) - float(expected_auroc))
        auc_ok = (auc_d is not None) and (auc_d <= float(auroc_tol))
    ok = bool(tpr_ok and (auc_ok is not False))
    info = {"passed": ok, "tpr_ok": bool(tpr_ok), "auroc_ok": auc_ok,
            "step0_tpr": (None if got is None else float(got)),
            "expected": float(expected), "tolerance": float(tol),
            "tpr_diff_positives": (None if got is None
                                   else abs(float(got) - float(expected)) / ONE_POSITIVE_TOL),
            "step0_auroc": (None if got_auc is None else float(got_auc)),
            "expected_auroc": expected_auroc, "auroc_tolerance": float(auroc_tol),
            "auroc_abs_diff": auc_d,
            "source_risk_auroc": step0_row.get("source_risk_auroc")}
    if verbose:
        print("\n" + "=" * 78)
        print(f"HARD GATE (BRIEF §2) -- {run_key} step-0 known-null, fold {fold}")
        print("=" * 78)
        print(f"  TPR@1e-3  {info['step0_tpr']}  vs {expected}   tol {tol:.6f} "
              f"({tol/ONE_POSITIVE_TOL:.1f} positives)  -> {'ok' if tpr_ok else 'FAIL'}")
        if expected_auroc is not None:
            print(f"  AUROC     {info['step0_auroc']}  vs {expected_auroc}   "
                  f"|diff| {auc_d:.2e}  tol {auroc_tol:.0e}  "
                  f"-> {'ok' if auc_ok else 'FAIL'}")
        print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
        print("=" * 78, flush=True)
    return info


def gate_or_abort(step0_row, expected, tol, run_key, fold=0, expected_auroc=None):
    info = check_step0_gate(step0_row, expected, tol, run_key, fold=fold,
                            expected_auroc=expected_auroc)
    if info["passed"] is not True:
        raise SystemExit(
            f"[unfreeze-fm] ABORT at the step-0 gate: TPR@1e-3 = {info['step0_tpr']} "
            f"does not reproduce {expected}.\n"
            f"  Step 0 is the phase-1 model itself, so this is a plumbing failure, not a "
            f"result. Likely causes, in order: head norm mismatch (wrong "
            f"--init-head-from), feature standardizer state not restored, wrong "
            f"local_threshold_mode, an fp16 eval encode against an fp32 reference, or "
            f"venue drift (§7.7: local Ada vs Kaggle T4 differ by 0.0204).")
    return info


# ---------------------------------------------------------------------------
# §5.5 -- the collapse gate, run AS PART OF the run
# ---------------------------------------------------------------------------

def run_collapse_gate(out_dir, fold, ref_ckpt, cache_dir, n_blocks, verbose=True):
    """Run the post-hoc collapse gate and write ``gate_fold{k}.json`` into the run dir.

    WHY IT IS WIRED IN RATHER THAN LEFT AS A FOLLOW-UP COMMAND. BRIEF §9.5 makes the
    agreement trace and the firing decision REQUIRED OUTPUTS, and this configuration --
    self-training with 14.8M trainable parameters against a 782-image target pool -- is the
    most likely in the project to produce a real collapse. The gate has been UNCALIBRATED
    across three rounds for want of exactly one positive case (PROJECT.md §7.6), so a firing
    here is itself a result. A required output that depends on someone remembering to run a
    second command is one that goes missing.

    STILL POST-HOC, STILL NOT A STOPPER. This runs AFTER training completes, reading the
    finished curve. §7.6 is explicit that the gate is Job 1 (post-hoc validator) and NEVER a
    live early-stopper: a false halt is unrecoverable and the threshold cannot be calibrated
    without an observed collapse. Nothing here can end a run.

    ``--from-curve`` is mandatory at n_blocks>0: recomputing agreement from
    ``data/feature_cache`` scores a trained head against features from a DIFFERENT (frozen)
    encoder (§7.6, §7.7). At n_blocks=0 the cache is valid, so the frozen control ALSO runs
    the cached recompute -- giving an inline-vs-post-hoc lock-step check on the very column
    the unfrozen run has no way to cross-check. That known-null is the only evidence the
    inline column is trustworthy for nb3.
    """
    import collapse_gate as cg
    out = {"run_dir": out_dir, "fold": int(fold), "n_blocks": int(n_blocks),
           "from_curve": True, "ref": ref_ckpt}
    print("\n" + "=" * 78)
    print(f"§5.5 COLLAPSE GATE (post-hoc; never a stopper) -- {os.path.basename(out_dir)}")
    print("=" * 78, flush=True)
    try:
        rep = cg.run_report(out_dir, ref_ckpt, cache_dir, f"target_fold{fold}",
                            device="cpu", expect_healthy=False,
                            tpr_col=f"fold{fold}_tpr_1e-3", from_curve=True)
    except Exception as e:                       # never let the gate kill a finished run
        out["error"] = f"{type(e).__name__}: {e}"
        print(f"[gate] ERROR (reported, not swallowed): {out['error']}")
        rep = None
    if rep:
        rows = rep["rows"]
        vals = [float(r["agreement"]) for r in rows]
        out.update({
            "n_points": len(rows),
            "agreement_first": vals[0], "agreement_last": vals[-1],
            "agreement_min": min(vals), "agreement_max": max(vals),
            "fired": bool(any(t.get("fired") for t in rep["gate"].values())),
            "triggers": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                         for k, v in rep["gate"].items()},
            "anchors": {"round1_selftraining_floor": 0.87,
                        "lever1_unfrozen_lows": {"nb1": 0.74, "nb3": 0.79}},
        })

    # the lock-step known-null, frozen runs only
    if int(n_blocks) == 0:
        try:
            posthoc = cg.agreement_curve(out_dir, ref_ckpt, cache_dir,
                                         f"target_fold{fold}", "cpu")[0]
            inline = {int(r["step"]): float(r["agreement"]) for r in (rep or {}).get("rows", [])}
            diffs = [abs(inline[int(r["step"])] - float(r["agreement"]))
                     for r in posthoc if int(r["step"]) in inline]
            out["lockstep_check"] = {
                "n_compared": len(diffs), "max_abs_diff": (max(diffs) if diffs else None),
                "ok": bool(diffs) and max(diffs) < 1e-9,
                "note": ("frozen run only: the INLINE agreement column and an independent "
                         "post-hoc recompute from the feature cache must agree exactly. "
                         "This is the only evidence that the inline column -- which the "
                         "unfrozen run has no way to cross-check -- is correct.")}
            print(f"[gate] lock-step inline vs post-hoc: "
                  f"max|diff| = {out['lockstep_check']['max_abs_diff']} over "
                  f"{len(diffs)} points -> "
                  f"{'MATCH' if out['lockstep_check']['ok'] else 'MISMATCH'}")
        except Exception as e:
            out["lockstep_check"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[gate] lock-step check unavailable: {out['lockstep_check']['error']}")
    else:
        out["lockstep_check"] = {
            "skipped": True,
            "note": ("n_blocks>0: a post-hoc recompute from the frozen feature cache would "
                     "score this head against a different encoder's features, so round-1's "
                     "inline-vs-post-hoc check cannot hold here BY CONSTRUCTION (§7.6).")}

    p = os.path.join(out_dir, f"gate_fold{int(fold)}.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[gate] -> {p}")
    return out


# ---------------------------------------------------------------------------
# R1 evaluation
# ---------------------------------------------------------------------------

def evaluate_r1(ckpt_path, n_blocks, encoder, instrument, cache_dir, target_fpr, fp16,
                out_dir, label, fold=0):
    """Score R1 on fold-0 test positives vs the shared negatives; dump raw scores."""
    cache = ev.FeatureCache(cache_dir)
    head, std, _ = ev.load_checkpoint(ckpt_path, "cpu")

    if n_blocks > 0:
        # the encoder is at its FINAL trained state here, which is the state that produced
        # this checkpoint -- so a live encode is both required (stale cache) and correct.
        live = instrument.live()
        def get(name):
            return live.encode(encoder, name, fp16=fp16)
    else:
        def get(name):
            return cache.load(name)

    pos = get(f"test_pos_fold{fold}")
    neg = get("test_negatives_shared")
    ps = ev.score_features(head, std, pos["features"], "cpu")
    ns = ev.score_features(head, std, neg["features"], "cpu")

    y = np.concatenate([np.ones(ps.size, np.int64), np.zeros(ns.size, np.int64)])
    s = np.concatenate([ps, ns])
    suite = metric_suite(y, s, target_fpr=target_fpr)

    sdir = os.path.join(out_dir, "scores")
    os.makedirs(sdir, exist_ok=True)
    # ids travel with the scores so the paired comparison against the control can VERIFY
    # row alignment rather than assume it.
    np.savez(os.path.join(sdir, f"{label}_fold{fold}.npz"),
             pos_scores=ps.astype(np.float64), neg_scores=ns.astype(np.float64),
             pos_ids=np.asarray(pos["ids"], dtype="<U80"),
             neg_ids=np.asarray(neg["ids"], dtype="<U80"),
             target_fpr=np.asarray(target_fpr))

    print(f"\n[R1] {label}  fold{fold}  TPR@{target_fpr:g} = {suite['tpr']:.4f}   "
          f"AUROC = {suite['auroc']:.4f}   AUPRC = {suite['auprc']:.4f}")
    return {"tpr": float(suite["tpr"]), "auroc": float(suite["auroc"]),
            "auprc": float(suite["auprc"]),
            "purity_corr": (None if suite["purity_corr"] is None
                            else float(suite["purity_corr"])),
            "comp_at_purity": (None if suite["comp_at_purity"] is None
                               else float(suite["comp_at_purity"])),
            "n_pos": int(ps.size), "n_neg": int(ns.size), "ckpt": ckpt_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--fold", type=int, default=DEFAULT_FOLD,
                    choices=sorted(FOLD_GATE),
                    help="which fold to run. Everything derives from it: the phase-1 "
                         "head, the step-0 gate value, the head-norm check, the run "
                         "directory. A wrong value fails at the gate rather than "
                         "producing a plausible wrong number (PROJECT.md §7.11).")
    ap.add_argument("--data-root", default=os.path.join(root, "data"))
    ap.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=2,
                    help="2 (not 4): on Kaggle the two concurrent training processes SHARE "
                         "4 CPU cores. Use 0 for a local Windows smoke (§7.7 spawn).")
    ap.add_argument("--fp16", action="store_true",
                    help="autocast the TRAINING forwards (round-1 image-space precision)")
    ap.add_argument("--eval-fp16", action="store_true",
                    help="also autocast the LIVE EVAL encode. Default OFF -- fp32 is what "
                         "makes the step-0 gate an exact comparison against the fp32 "
                         "feature cache. Turning it on requires --step0-tol.")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="MANDATORY at n_blocks=3 (§7.1a: FreeMatch at true batch 256 peaks "
                         "4.76 GB with it and OOMs on a 14.56 GiB T4 without). Numerically "
                         "exact; refused as a no-op at n_blocks=0.")
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help=f"LR of the TOP unfrozen block (default {BACKBONE_LR:g} = head/10)")
    ap.add_argument("--curve-every", type=int, default=CURVE_EVERY)
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="SMOKE-TEST ONLY. The run's budget is fixed per --run; a different "
                         "value writes to a _SMOKE-suffixed dir so it can never be mistaken "
                         "for the real run.")
    ap.add_argument("--schedule-steps", type=int, default=BUDGET_STEPS,
                    help="span of the mu/warmup schedule. PINNED to 8000 for BOTH runs so "
                         "the control is a truncation, not a compressed copy.")
    ap.add_argument("--dir-suffix", default="")
    ap.add_argument("--smoke-batch-size", type=int, default=None,
                    help="SMOKE-TEST ONLY, and REFUSED without --max-steps. Batch 256 is a "
                         "pre-registered protocol constraint (§7.4, a known result-flipping "
                         "confound) and FreeMatch is batch-coupled through its SAT EMA, so "
                         "a smaller batch is a different method -- usable only to prove the "
                         "code path runs on a small card, never to produce a number.")
    ap.add_argument("--init-head-from", default=None,
                    help=f"phase-1 head (default: {ROUND1_FM_CKPT}). runs/ is not packed "
                         f"into data.zip, so on Kaggle point this straight at the file.")
    ap.add_argument("--expect-head-l2", type=float, default=None,
                    help="assert the loaded head's L2 norm (identity check across runs)")
    ap.add_argument("--expect-step0-tpr", type=float, default=None,
                    help="the hard gate's target value (default: the measured fold-0 "
                         "TPR@1e-3 of the phase-1 checkpoint)")
    ap.add_argument("--step0-tol", type=float, default=0.0,
                    help="FLOOR on the TPR gate tolerance. The real tolerance is set by the "
                         "eval path: 0 on the deterministic cached path, ONE POSITIVE "
                         "(1/98) on the live-encode path, where the encoder forward is not "
                         "bit-reproducible (MEASURED). The AUROC half of the gate is what "
                         "actually checks model identity and is not loosened.")
    ap.add_argument("--agreement-ref", default=DEFAULT_AGREEMENT_REF)
    # ---- CONTINUATION-STATE RECONSTRUCTION (continuation_state.py) ----
    ap.add_argument("--mu-step-offset", type=int, default=BUDGET_STEPS,
                    help="added to the schedule position the LOSS reads, so a continuation "
                         "resumes at mu_eff = full value instead of restarting the cosine "
                         "warmup (which holds mu at EXACTLY 0 for the first 10%% of the "
                         "schedule). Default 8000 = round 1's end. Affects mu and the "
                         "warmup gate ONLY -- there is no LR scheduler in this path.")
    ap.add_argument("--sat-passes", type=int, default=40,
                    help="passes over the target pool when reconstructing tau_global/pbar")
    ap.add_argument("--sat-ema-iters", type=int, default=5000,
                    help="length of the EMA cross-check on the rebuilt SAT buffers")
    ap.add_argument("--no-rebuild-sat", action="store_true",
                    help="skip the SAT reconstruction (reproduces the un-fixed behaviour: "
                         "the threshold restarts at 1/C)")
    ap.add_argument("--adam-warmup-steps", type=int, default=200,
                    help="steps of AdamW moment warm-up before training. Gradients are "
                         "computed under the corrected objective and only the MOMENTS are "
                         "updated; the weights are restored every step and the encoder is "
                         "left cold. 0 disables it.")
    ap.add_argument("--reference-config", default=None,
                    help="the other run's emitted run_config.json. Every PINNED key must "
                         "match it (§1).")
    ap.add_argument("--emit-config", default=None,
                    help="write this run's run_config.json WITHOUT training, then exit")
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--allow-off-budget", action="store_true",
                    help="permit a budget other than the PRE-REGISTERED one (§7.11). The "
                         "run is then recorded with on_preregistered_budget=false and can "
                         "never be aggregated as a confirmatory result.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke_batch_size and not args.max_steps:
        raise SystemExit(
            "[unfreeze-fm] --smoke-batch-size requires --max-steps. Batch 256 is pinned by "
            "the matched protocol; a reduced batch is only ever a code-path smoke, and "
            "--max-steps is what forces the output into a _SMOKE-suffixed directory where "
            "it cannot be mistaken for the run.")
    if args.eval_fp16 and args.step0_tol == 0.0:
        raise SystemExit(
            "[unfreeze-fm] --eval-fp16 with --step0-tol 0 will fail the gate on rounding "
            "alone: an fp16 live encode does not reproduce the fp32 feature cache "
            "bit-for-bit. Either drop --eval-fp16 (recommended) or pass an explicit "
            "--step0-tol; the largest defensible slack is one positive, 1/98 = 0.0102.")
    if args.run == "nb3_unfrozen" and not (args.grad_checkpointing or args.dry_run
                                           or args.emit_config or args.max_steps):
        raise SystemExit(
            "[unfreeze-fm] REFUSING to launch nb3_unfrozen without --grad-checkpointing. "
            "§7.1a MEASURED FreeMatch at n_blocks=3, true batch 256: 4.76 GB with "
            "checkpointing, OOM on a 14.56 GiB T4 without. Gradient accumulation is not a "
            "substitute and the path is removed -- FreeMatch's threshold is an EMA over "
            "per-batch confidence statistics, so micro-batching changes the method.")

    train_one(args.run, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
