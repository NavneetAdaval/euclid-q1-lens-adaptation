"""
run_dann_r5.py
==============
ROUND 5 -- **DANN x n_blocks=3**, image-space, per-fold runner.

    ONE ARM PER PROCESS.  dann_nb3 (or its lambda=0 control), fold k, image-space,
    weak aug, fp16, gradient checkpointing, true batch 256 + 256, FRESH FROM STEP 0.

Structurally this is ``run_coral_r4_dev8000.py`` with the CORAL objective replaced:
same ``Paths`` / resume / logit-persistence / step-0-gate / config-diff / retention
machinery, same cadences, same single-GPU rule. Where it differs from round 4, it
differs because DANN differs, and each difference is stated below.

-------------------------------------------------------------------------------
BRIEF v1's THREE FLAGGED DISAGREEMENTS -- ALL RESOLVED IN BRIEF v2
-------------------------------------------------------------------------------
1. **FILE NAME.** ``run_dann.py`` is round 1's fold-0 cached-feature probe and stays.
   This file follows round 4's precedent (``run_coral.py`` kept;
   ``run_coral_round4.py`` added). **Approved as-is.**

2. **``morph_proxy_auroc`` is now BUILT** -- ``morph_proxy.py``, a RECOVERY of the
   historical ML-FP-vs-P1-reserve instrument (the role and ``linear_probe.py`` were
   dropped at the splits_v6 rewrite; the DATA survived). Step-0 reading, both arms:
   **0.9202** on the adapter, **0.9214** on raw frozen encoder features against the
   historical 0.914. It arms abort condition 3, whose threshold is calibrated in
   ``runs/dann_r5_morph_step0.json`` -- **0.55 is not hardcoded anywhere**; it belongs
   to an instrument with a 0.61 baseline.

3. **Head init: Baseline A's ``step_8000`` confirmed** (``head_l2 = 28.475603``).
   Baseline A trains on source only and the source does not vary by fold, so for the
   ``LayerNorm(640) + Linear(640->1)`` architecture the source-only linear probe is
   UNIQUE -- a single head file is the correct artifact, not a shortcut. Because the
   head is fold-independent, the per-fold TPR/AUROC step-0 gate is the ONLY thing
   that proves fold identity, so it is **non-skippable: there is no flag that can
   bypass it** and it ABORTS rather than warning. See ``step0_gate``.

-------------------------------------------------------------------------------
ONE DISAGREEMENT WITH BRIEF v2 -- STATED, NOT RESOLVED SILENTLY
-------------------------------------------------------------------------------
Brief v2 asks for the step-0 gate "at tolerance 0". **PROJECT.md 7.7 measured that
this cannot be done on the live-encode path** and paid a session to find out: the
FreeMatch nb3 gate demanded exact reproduction, passed on folds 0/1/3 and failed
fold 2 by exactly one positive after a correct 16-minute setup, because
live-encode-vs-cached SCORE drift is ~1e-2 while fold 2's closest positive sat
3.06e-03 above its threshold. The pinned table here comes from Baseline A's CACHED
curve; an nb3 run scores through ``live_encode``. So tolerance 0 on TPR would fire
on arithmetic, not on a wrong fold.

Implemented as 7.7 prescribes, which is STRICTER than v1 was and is chosen by the
EVAL PATH rather than by a flag:

    TPR@1e-3   tolerance 0 on the cached path (n_blocks=0), ONE POSITIVE on the
               live-encode path (n_blocks>0).  ABORT, not warn.
    AUROC      tolerance 1e-3, always.  ABORT, not warn.

The AUROC half is what actually proves fold identity: it is rank-based, ~100x the
measured live-encode drift (1.2e-05) and ~7x tighter than a wrong head (7.6e-03).
7.7's worked example is that fold 1's head on fold 2's data gives TPR **exactly
0.336735** -- a legitimate-looking value that only AUROC catches. Tolerance 0 on TPR
would add no identification power that the AUROC half does not already have, and
would add false aborts.

-------------------------------------------------------------------------------
THE LAMBDA SCHEDULE -- THE ONE THING MOST LIKELY TO GO WRONG SILENTLY
-------------------------------------------------------------------------------
    lambda(step) = 2 / (1 + exp(-10 p)) - 1 = tanh(5p) ,  p = min(step/HORIZON, 1)

``LAMBDA_HORIZON`` is a **declared absolute constant**, **default 2000** (brief v2;
it was 8000 in v1), passed by flag, recorded in ``run_config.json``, and **identical
in the development run and in every confirmatory run**. It is NOT ``total_steps``.

WHY 2000. At horizon 2000, lambda >= 0.95 at every budget from 750 to 8000
(measured: 0.954 / 0.9989 / 0.9999 at 750 / 1500 / 2000), so the reported number does
not depend on a read-off step that 7.11 and 7.15 both establish is undeterminable. At
horizon 8000 the same budget window spans lambda in [0.44, 0.99] and lambda would
never reach 1.0 in any reported run -- not a faithful reading of Ganin. Precedent for
2000: the two unfrozen nb3 rounds declared 1500 (FreeMatch) and 2000 (CORAL). Computing it from run
length makes the 8000-step development run and the 2000-step confirmatory runs see
different objectives at the same step -- both still train, both still produce
plausible numbers, and the development curve silently stops predicting the
confirmatory result. See ``da_losses.dann_lambda_at_step``; the horizon-invariance
test in ``test_da_dann.py`` is the load-bearing check.

-------------------------------------------------------------------------------
ONE OPTIMISER
-------------------------------------------------------------------------------
D's ~1.7M parameters are ONE EXTRA PARAM GROUP on the AdamW the Trainer already
built, at the head LR (1e-4), no ``lr_decay``, weight decay 0.05. No second
optimiser, no separate D learning rate: gradient reversal is what removes the need
for a D/encoder balance, and that is the whole reason DANN was chosen over ADDA.
See ``dann_round5.install_dann``.

-------------------------------------------------------------------------------
NON-NEGOTIABLES CARRIED OVER FROM PROJECT.md (all previously paid for)
-------------------------------------------------------------------------------
* **Gradient checkpointing is MANDATORY at nb3** (7.1a): batch 256 image-space OOMs
  a 14.56 GiB T4 without it.
* **Gradient accumulation is NOT a substitute and the path is removed** (7.1a). It
  invalidates every DA loss, DANN included -- D would see a rank-32 view of a batch
  whose whole point is that it holds 256 of each domain.
* **The frozen feature cache is INVALID at n_blocks > 0** (7.1a). Every checkpoint
  here is scored through ``live_encode.LiveEncoder``. Scoring an nb3 checkpoint from
  the stale cache once produced TPR 0.194 against a logged 0.020.
* **Checkpoint retention guard** -- all 32 checkpoints written, counted, and the
  count recorded in ``run_config.json`` (7.7: this was a COLLECTION bug that cost
  the A1 study its unfrozen checkpoints).
* **Single GPU per fold.** No DataParallel, no DDP, no device sharding: halving the
  per-device batch would halve what D sees per step.
* **Timing only within one process, first configuration discarded** (7.7).

METRIC HYGIENE. The only test-set metric computed is the step-0 plumbing gate, a
KNOWN NULL (it must reproduce the phase-1 head's own per-fold number). After that
the run computes no metric at all: every training segment runs inside
``coral_round4.test_label_lockout()``, and checkpoints are SCORED (logits
persisted) but never MEASURED. Fold 0 is DEVELOPMENT and no number from it enters
a headline.

Run:
    python -u scripts/run_dann_r5.py --out-dir runs/dann_r5_nb3_dev8000_fold0
    python -u scripts/run_dann_r5.py --dry-run              # mock data, 3 steps, exits
    python -u scripts/run_dann_r5.py --emit-config ref.json # config only, nothing trained
    python -u scripts/run_dann_r5.py --arm dann_nb3_lambda0 --reference-config ref.json
    python -u scripts/run_dann_r5.py --grad-check           # nb3 reachability probe
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import evaluate as ev                                              # noqa: E402
from coral_round4 import test_label_lockout                        # noqa: E402
from da_losses import DANN_LAMBDA_HORIZON, dann_lambda_at_step     # noqa: E402
from da_losses import DANN_DROPOUT                                  # noqa: E402
from dann_round5 import (ABORT_COLUMNS, ABORT_MODE_ARMED,          # noqa: E402
                         ABORT_MODE_LOG_ONLY, CHECKPOINT_COLUMNS,
                         COND3_COLUMN, COND3_INTERPRETATION_COLUMNS,
                         COND3_MARGIN, COND3_M_CEILING, COND3_M_J_FACTOR,
                         ADVISORY_ONLY_DETECTORS, COND3B_SUSTAIN,
                         COND3_SIGN_FLIP, COND4_SRC_CEILING, COND4_SRC_MARGIN,
                         COND4_SUSTAIN, COND4_TGT_CEILING, COND4_TGT_MARGIN,
                         DANN_HEALTH_ANCHORS,
                         DEFAULT_HOLDOUT_FRAC, STEP_COLUMNS, TRACE_COLUMNS,
                         ControlCurve, DannAbortWatch, DannStepWindow,
                         assert_no_batchnorm, build_target_holdout,
                         dann_reachable_params, holdout_domain_stats, install_dann,
                         materialise_holdout_images, morph_trajectory_stats,
                         print_morph_trajectory_stats)
from morph_proxy import (MORPH_N_EVAL_PER_CLASS,                   # noqa: E402
                         MORPH_N_FIT_PER_CLASS, MORPH_RIDGE_ALPHA,
                         MorphProxy, build_morph_pool)
from heads import LINEAR                                           # noqa: E402
from lens_dataset import LensDataset, UNLABELED                    # noqa: E402
from samplers import (ClassBalancedBatchSampler, PairedDomainLoader,   # noqa: E402
                      make_balanced_loader)
from train_skeleton import TrainConfig, Trainer                    # noqa: E402
# Method-agnostic protocol machinery, IMPORTED VERBATIM from the round-4 runner --
# path resolution helpers, the RNG-preserving scoring context, the atomic CSV
# append, the banner, the logit Scorer. Nothing CORAL-specific is imported: round
# 4's `check_preconditions` closes over CORAL's key list and this module owns its
# own (the same trap round 4 recorded and avoided).
from run_coral_gamma_sweep import _append                          # noqa: E402
from run_coral_r4_dev8000 import (Paths, Scorer, _find_under,      # noqa: E402
                                  _path_tail, _rng_preserved, _rng_state,
                                  _set_rng_state, _sha256, _same,
                                  _truncate_csv_after, banner)
from run_unfreeze_freematch import head_l2_norm                    # noqa: E402
from run_screen_unfreeze import assert_no_bf16                     # noqa: E402


# ===========================================================================
# ROUND-5 CONSTANTS -- the swapped constant block
# ===========================================================================
ROUND = 5
METHOD = "dann"
DEV_STEPS = 8000           # 7.14's development maximum, on the POWER-MATCHING argument:
                           # FreeMatch's decay needs 8000 to detect, so a null at 4000
                           # cannot separate "DANN does not decay" from "4000 lacks the
                           # power". runs/devmax_truncation.json is the evidence.
CONFIRM_STEPS = 4000       # DECIDED 2026-09-02 and HASHED before any confirmatory fold
                           # exists (7.14). THE RULE: budget = 2 x the lambda horizon, so
                           # HALF of each confirmatory run trains at fixed adversarial
                           # strength. That rule reads DANN's OWN CONFIGURATION and no
                           # metric -- it is statable without opening a curve file, which
                           # is how a reader can check it was not chosen to select a
                           # value. The fold-0 curve then CORROBORATES it: the settled
                           # regime begins ~2250, and the reading at 4000 (0.4633) is
                           # 0.27 settled-sds from the settled mean (0.4787) -- i.e.
                           # unremarkable, which is the point.
                           # History: 2000 declared 2026-09-01 (v11), WITHDRAWN
                           # 2026-09-02 (v12), and NOT re-chosen.
CONFIRM_BUDGET_FILE = "runs/dann_r5_budget.json"
CONFIRM_BUDGET_DECLARED = True
CONFIRM_BUDGET_SHA256 = ("abf0dae0e313ae5388b833e7aad86171285299774a18af14f4c44309f0231fc1")
CONFIRM_BUDGET_RULE = ("Budget = 2 x the lambda horizon, so that half of each "
                       "confirmatory run trains at fixed adversarial strength. Derived "
                       "from DANN's own configuration; reads no metric.")
# PRE-DECLARED 2026-09-02, before any confirmatory fold exists. ckpt_every is 250 and the
# budget is 4000, so step 2000 is CHECKPOINT 8 OF 16 on every fold and its logits are
# written natively -- the sensitivity readout costs nothing.
SENSITIVITY_STEP = 2000
HEADLINE_DECLARATION = (
    "The headline DANN x nb3 figure is the 5-fold mean at the declared budget of 4000. "
    "The 5-fold mean at step 2000 is reported as a BUDGET-SENSITIVITY CHECK and does not "
    "replace it, WHICHEVER IS HIGHER.")
LOG_EVERY = 125            # 64 trace points at 8000 -- round 4's cadence, kept
CKPT_EVERY = 250           # 32 retained weight checkpoints -- the brief's count
BATCH_SIZE = 256           # TRUE batch, never micro-batched
POS_FRACTION = 0.5
SEED = 42
HEAD_TYPE = LINEAR

# ---- 7.1a layer-wise LR settings (Zoobot v2.0.1 defaults), matched to round 4 ----
LR = 1e-4                  # head LR; D's group takes the SAME value, no lr_decay
BACKBONE_LR = None         # None -> EQUALS head LR (Zoobot: block i=0 gets lr*decay**0)
LR_DECAY = 0.75
WEIGHT_DECAY = 0.05
# 7.14a: tuning budget ZERO, same as CORAL and FreeMatch. Nothing here was searched.

# ---- DANN knobs ----
LAMBDA_HORIZON = DANN_LAMBDA_HORIZON        # 2000. DECLARED ABSOLUTE. Not total_steps.
D_HOLDOUT_FRAC = DEFAULT_HOLDOUT_FRAC       # 0.10 of the fold's target pool
D_DROPOUT = DANN_DROPOUT                    # 0.2, Belfiore et al. 2024 (A&A); untuned
D_DROPOUT_LAYERS = 2                        # after EACH hidden ReLU -- see da_losses
SPECTRUM_EVERY = 1                          # rank monitors every step (~10 ms/domain)
MORPH_EVERY = 25                            # brief v3 4.1 -- was 1 in v2
# WHY 25. The proxy is a no-grad forward of 256 images (half a training batch) plus two
# 640x640 solves, EVERY call; at cadence 1 that is real wall clock on top of the
# ~78-image D holdout, over 8000 steps, on two arms. At 25 an 8000-step run still gets
# 320 readings and condition 3 reads over a 500-step window, i.e. ~20 readings per
# window -- which establishes "sustained" exactly as well as 500 do.
#
# BOTH ARMS MUST USE THE SAME VALUE. Condition 3 compares against the lambda=0 control
# AT THE SAME ABSOLUTE STEP and refuses to interpolate, so a cadence mismatch does not
# degrade the comparison, it DELETES it. `morph_every` is therefore a PINNED key.

# ---- abort-condition action, gated on the arm (brief v3 4.2) ----
ABORT_MARGIN_COND3 = COND3_MARGIN           # None -- UNSET BY DESIGN, see dann_round5
ABORT_SUSTAIN_COND4 = COND4_SUSTAIN         # 500; a DECLARED addition, see dann_round5
ABORT_COND3_COLUMN = COND3_COLUMN           # morph_head_auroc, NOT the frozen probe

# ---- phase-1 head: see disagreement 3 in the module docstring ----
BASELINE_A_DIR = os.path.normpath(os.path.join(_HERE, "..", "runs", "bakeoff_baseline_A"))
DEFAULT_INIT_HEAD = os.path.join(BASELINE_A_DIR, "checkpoints", "step_8000.pt")

# ---- PER-FOLD STEP-0 GATE TABLE ----
# Read from runs/bakeoff_baseline_A/curve.csv, row step=8000 -- the same checkpoint
# DEFAULT_INIT_HEAD points at. Baked in for the reason 7.11 baked FreeMatch's in:
# pointing --fold at one fold while gating on another's number would not abort, it
# would train happily and produce a plausible, wrong result.
#
# head_l2 is the SAME on every fold because Baseline A is source-only -- ONE model
# scored on five different test splits. That is a real difference from round 4,
# where each fold had its own CORAL head, and it means the head-norm check proves
# the right FILE but not the right FOLD. The per-fold TPR/AUROC gate is what proves
# the fold, so it is not optional here.
BASELINE_A_HEAD_L2 = 28.475603103637695
FOLD_GATE = {
    0: {"tpr": 0.19387755102040816, "auroc": 0.9426918613954919},
    1: {"tpr": 0.14285714285714285, "auroc": 0.8922624620134809},
    2: {"tpr": 0.19387755102040816, "auroc": 0.9031282879453535},
    3: {"tpr": 0.15306122448979592, "auroc": 0.9016486400613506},
    4: {"tpr": 0.19587628865979381, "auroc": 0.8908307601467254},
}
# Folds 0-3 hold 98 test positives; fold 4 holds 97 (489 lenses over 5 folds).
N_TEST_POS = {0: 98, 1: 98, 2: 98, 3: 98, 4: 97}

# ---- GATE TOLERANCES (7.11's measured values; the reasoning is metric-level) ----
def one_positive_tol(fold: int) -> float:
    return 1.0 / float(N_TEST_POS[int(fold)])


STEP0_AUROC_TOL = 1e-3         # ~100x the measured live-encode drift (7.7)


def default_abort_mode(arm_key: str, fold: int) -> str:
    """log_only on development and on the control; armed on confirmatory folds.

    A DEFAULT, not an inference: whatever this returns is written into
    ``run_config.json`` as ``abort_mode`` with ``abort_mode_source`` next to it, so a
    reader never has to re-derive it, and ``--abort-mode`` overrides it outright
    (brief v3 4.2).
    """
    if ARMS[arm_key]["lambda_zero"]:
        return ABORT_MODE_LOG_ONLY          # the control is the reference; it must finish
    if int(fold) == 0:
        return ABORT_MODE_LOG_ONLY          # development: the full curve IS the evidence
    return ABORT_MODE_ARMED                 # confirmatory


def step0_tpr_tolerance(fold: int, n_blocks: int) -> float:
    """THE gate tolerance on TPR, chosen by the EVAL PATH rather than by a flag.

    ``n_blocks == 0``  -> the frozen feature cache is valid and the eval is
                          deterministic, so the pinned value must reproduce EXACTLY.
    ``n_blocks > 0``   -> the eval runs through ``live_encode``, whose SCORE drift
                          against the cached path is ~1e-2 (7.7). One positive is the
                          finest difference the fold's positives can express and is
                          the calibrated floor; 0 here would fire on arithmetic.

    7.7 records what happens otherwise: FreeMatch's nb3 gate demanded exact
    reproduction, passed folds 0/1/3 and failed fold 2 by exactly one positive after a
    correct 16-minute setup, because fold 2's closest positive sat 3.06e-03 above its
    threshold -- inside the drift envelope. The run was fine; the gate's premise was
    false.
    """
    return 0.0 if int(n_blocks) == 0 else one_positive_tol(fold)

ARMS = {
    "dann_nb3": {"n_blocks": 3, "lambda_zero": False, "dir": "dann_r5_nb3",
                 "steps": DEV_STEPS},
    # THE CONTROL RUNS THE FULL 8000, not the design doc's 2000 (brief v3 5). It
    # acquired a second job: it is the REFERENCE TRAJECTORY for every monitor, not just
    # a null for the DANN loss. A control that stops at 2000 leaves condition 3 with no
    # reference past step 2000, and the same gap opens for feature std, participation
    # ratio and the morphology probe.
    "dann_nb3_lambda0": {"n_blocks": 3, "lambda_zero": True,
                         "dir": "dann_r5_nb3_lambda0", "steps": DEV_STEPS},
    # nb0 is NOT offered. 7.15 closed round 4 WITHOUT its nb0 control and recorded
    # the consequence (no unfreezing effect is attributable for CORAL). Adding an
    # nb0 arm here is a decision with a GPU-hour price and it belongs to whoever
    # writes the paper, not to a default in a runner.
}

LOGIT_ARM_FMT = "{arm}_fold{fold}"
SCRIPT_HASH_FILES = ("run_dann_r5.py", "dann_round5.py", "da_losses.py",
                     "morph_proxy.py", "calibrate_morph_r5.py",
                     "train_skeleton.py", "lens_dataset.py", "samplers.py",
                     "live_encode.py", "evaluate.py", "zoobot_encoder.py", "heads.py",
                     "metrics.py", "coral_round4.py")


# ===========================================================================
# CONFOUND PINNING -- everything that must MATCH between two arms
# ===========================================================================
# Derived from round 4's 30-key list: CORAL's seven loss knobs are gone (no
# covariance, no shrinkage, no target EMA) and DANN's are added. `dann_lambda_zero`
# is the variable under test between the two arms and is therefore EXEMPT; every
# other DANN key is PINNED, including the horizon -- an arm pair whose lambda
# horizons differ is comparing two schedules, not one method against its control.
PINNED_KEYS = ("augmentation", "precision", "seed", "batch_size",
               "lr", "lr_decay", "weight_decay", "pos_fraction", "head_type", "fold",
               "log_every", "ckpt_every", "true_batch", "gradient_accumulation",
               "init_head_from_tail", "init_head_l2", "max_steps",
               "method", "dann_lambda_horizon", "dann_lambda_schedule",
               "dann_d_hidden", "dann_d_norm", "dann_d_lr", "dann_d_weight_decay",
               "dann_d_dropout", "dann_d_dropout_layers",
               "dann_d_holdout_frac", "dann_d_holdout_n", "dann_domain_label_source",
               "dann_attachment", "dann_single_forward_pass", "dann_optimizer_groups",
               "morph_proxy_n_fit_per_class", "morph_proxy_n_eval_per_class",
               "morph_proxy_ridge_alpha", "morph_proxy_instrument",
               "abort_cond3_column", "abort_cond3_interpretation_columns",
               # PINNED, not a cadence nicety (brief v3 5): condition 3 step-matches
               # against the control and refuses to interpolate, so a mismatch here
               # deletes the comparison rather than coarsening it.
               "morph_every",
               "eval_precision", "n_blocks")

# EXEMPT, each for a stated reason -- never widen this list to make a run pass.
EXEMPT_KEYS = ("dann_lambda_zero",       # THE variable under test between the arms
               "arm", "run_dir", "out_dir", "logit_arm",   # naming
               "grad_checkpointing",     # memory<->recompute, numerically exact (7.1a)
               "backbone_lr", "backbone_lr_effective",     # only exist at n_blocks>0
               "encoder_eval",           # follows n_blocks (cache valid iff frozen)
               "epochs_planned", "steps_per_epoch",        # derived from max_steps
               "device", "num_workers",  # throughput only; the seeded sampler is unaffected
               "spectrum_every", "holdout_every",   # monitor cadences (morph_every is PINNED)
               "morph_proxy_step0",      # a measurement, not a setting
               "abort_mode", "abort_mode_source",   # gated on the ARM, by design
               "abort_cond3_margin", "abort_cond3_margin_status",  # unset on arms 1-2
               "abort_cond4_src_margin", "abort_cond4_tgt_margin",
               "abort_cond3_median_window_k",   # declared before M, not between arms
               "control_curve", "abort_conditions"
               "init_head_from",         # absolute path differs between venues
               "is_preregistered_budget")


def check_preconditions(cfg_json, reference_config, verbose=True,
                        pinned=PINNED_KEYS, exempt=EXEMPT_KEYS):
    """REFUSE to start unless every PINNED key matches the declared reference arm.

    The key lists are PARAMETERS, never module globals closed over. Round 4 recorded
    the near-miss this prevents: a check imported from another module silently
    compared THAT module's key list, so a gamma mismatch sailed through while every
    absent key read ``<absent>`` on both sides and "matched".

    A key absent from BOTH configs is a MISMATCH, not a match -- a pinned key that no
    longer exists anywhere means the guard has stopped guarding, which is how a check
    rots into a rubber stamp.
    """
    if not os.path.isfile(reference_config):
        raise SystemExit(
            f"[dann-r5] ABORT: reference config not found: {reference_config}\n"
            f"  This run cannot verify that only the exempt keys differ, so it will not "
            f"start. Emit the other arm's config first:\n"
            f"    python scripts/run_dann_r5.py --arm dann_nb3 --emit-config <path>")
    with open(reference_config) as fh:
        ref = json.load(fh)

    rows, bad, vacuous = [], [], []
    for k in pinned:
        rv, cv = ref.get(k, "<absent>"), cfg_json.get(k, "<absent>")
        ok = _same(rv, cv) if not (rv == "<absent>" and cv == "<absent>") else False
        if rv == "<absent>" and cv == "<absent>":
            vacuous.append(k)
        rows.append({"key": k, "reference": rv, "this_run": cv, "status":
                     "MATCH" if ok else "MISMATCH"})
        if not ok:
            bad.append(k)
    for k in exempt:
        rows.append({"key": k, "reference": ref.get(k, "<absent>"),
                     "this_run": cfg_json.get(k, "<absent>"), "status": "EXEMPT"})

    if verbose:
        print("\n" + "=" * 100)
        print("PRECONDITION CHECK -- only the exempt keys may differ between arms")
        print(f"  reference: {reference_config}")
        print("=" * 100)
        print(f"  {'key':<30} | {'reference':<26} | {'this run':<22} | status")
        print("  " + "-" * 96)
        for r in rows:
            print(f"  {r['key']:<30} | {str(r['reference'])[:26]:<26} | "
                  f"{str(r['this_run'])[:22]:<22} | "
                  f"{'ok' if r['status'] == 'MATCH' else r['status']}")
        print("  " + "-" * 96)
    if vacuous:
        raise SystemExit(
            f"\n[dann-r5] ABORT: {len(vacuous)} pinned key(s) are absent from BOTH "
            f"configs: {vacuous}. A guard that compares nothing is not a guard -- fix the "
            f"key list or the config builder.")
    if bad:
        raise SystemExit(
            f"\n[dann-r5] ABORT: {len(bad)} pinned setting(s) differ from the reference "
            f"arm: {bad}.\n"
            f"  The arms may differ ONLY in dann_lambda_zero and its dependents. "
            f"Augmentation+precision alone is worth -0.0714 TPR (7.3), and a differing "
            f"lambda horizon would make the pair measure two changes instead of one.")
    if verbose:
        print("  ALL PINNED SETTINGS MATCH.\n")
    return rows


# ===========================================================================
# path resolution -- the Kaggle mount, or local, or explicit
# ===========================================================================

def resolve_paths(args, root):
    """Locate data / splits / cache / phase-1 head. Same SEARCH (never guess) pattern
    round 4 established: ``data.zip`` auto-extracts to ``data/data/`` so the pieces do
    not share one common root, and the enclosing directory has been
    ``/kaggle/input/<slug>`` in one session and
    ``/kaggle/input/datasets/<owner>/<slug>`` in another.
    """
    search_roots = [r for r in (args.input_root, "/kaggle/input", root) if r]
    out = {}

    def pick(name, explicit, pattern, required=True):
        if explicit:
            p = os.path.abspath(explicit)
        else:
            hit = _find_under(search_roots, pattern)
            p = os.path.abspath(hit) if hit else None
        if required and (p is None or not os.path.exists(p)):
            raise SystemExit(
                f"[dann-r5] ABORT: could not resolve {name}.\n"
                f"  searched {search_roots} for {pattern!r}\n"
                f"  pass --{name.replace('_', '-')} explicitly.")
        out[name] = p
        return p

    splits = pick("splits", args.splits, os.path.join("splits", "splits.json"))
    out["data_root"] = (os.path.abspath(args.data_root) if args.data_root
                        else os.path.dirname(os.path.dirname(splits)))
    cache_probe = (os.path.join(args.cache_dir, "source_train.npz") if args.cache_dir
                   else None)
    if cache_probe and os.path.isfile(cache_probe):
        out["cache_dir"] = os.path.abspath(args.cache_dir)
    else:
        p = pick("_cache_probe", None, os.path.join("feature_cache", "source_train.npz"))
        out["cache_dir"] = os.path.dirname(p)
    pick("init_head_from", args.init_head_from,
         os.path.join("bakeoff_baseline_A", "checkpoints", "step_8000.pt"))
    out.pop("_cache_probe", None)
    return out


# ===========================================================================
# the config record -- ONE construction path, shared by --emit-config and the run
# ===========================================================================

def build_config(arm_key, args, *, fold, paths=None, init_head_l2=None,
                 holdout=None, steps_per_epoch=None, epochs_planned=None,
                 device=None, morph_info=None) -> dict:
    """The run's declared configuration.

    ``--emit-config`` calls this and returns; the real run calls the SAME function.
    That is the whole point (7.7): a second concurrent arm can validate its config
    against the first at minute 2 instead of waiting hours for a run to finish.
    Fields the caller cannot know without touching data are passed in rather than
    recomputed, so there is exactly one definition of each.
    """
    spec = ARMS[arm_key]
    fold = int(fold)
    if fold not in FOLD_GATE:
        raise SystemExit(f"[dann-r5] --fold must be one of {sorted(FOLD_GATE)}, got {fold}")
    n_blocks = int(spec["n_blocks"])
    max_steps = int(args.max_steps)
    grad_ckpt = bool(args.grad_checkpointing) and n_blocks > 0
    if max_steps != int(spec.get("steps", max_steps)):
        print(f"[dann-r5] NOTE: --max-steps {max_steps} differs from {arm_key}'s "
              f"declared {spec['steps']}. Recorded as on_declared_steps=false; the two "
              f"arms MUST match (max_steps is a PINNED key) or condition 3 loses its "
              f"reference past the shorter one.", flush=True)
    lambda_zero = bool(spec["lambda_zero"] or args.dann_lambda_zero)
    head_ckpt = os.path.abspath((paths or {}).get("init_head_from")
                                or args.init_head_from or DEFAULT_INIT_HEAD)
    dev = device or args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {
        "round": ROUND, "method": METHOD, "arm": arm_key,
        "run_dir": f'{spec["dir"]}_fold{fold}',
        "logit_arm": LOGIT_ARM_FMT.format(arm=arm_key, fold=fold),
        "n_blocks": n_blocks, "fold": fold,
        "n_test_positives": N_TEST_POS[fold],
        "data_path": "image_space",
        "domain_adaptation": "dann",

        # ---- LP->FT init: SEE DISAGREEMENT 3 IN THE MODULE DOCSTRING ----
        "lp_ft": True,
        "init_head_from": head_ckpt,
        "init_head_from_tail": _path_tail(head_ckpt, 5),
        "init_head_l2": init_head_l2,
        "init_head_provenance": (
            "Baseline A step_8000 (runs/bakeoff_baseline_A) -- the SOURCE-ONLY linear "
            "probe, i.e. the 'LP' of LP->FT. DECLARED, not inherited: round-1 DANN ran "
            "fold 0 only and its head sits behind a 640->64->640 bottleneck adapter, so "
            "there is no DANN phase-1 head to load. Method-neutral and identical on every "
            "fold, so it cannot favour DANN. NEEDS SIGN-OFF -- it is the one place round 5 "
            "is not a copy of round 4's construction."),

        # ---- budget ----
        "max_steps": max_steps,
        "arm_declared_steps": int(spec.get("steps", max_steps)),
        "on_declared_steps": max_steps == int(spec.get("steps", max_steps)),
        "development_maximum": DEV_STEPS,
        # fold 0 IS the development curve and runs at 8000; folds 1-4 run at the budget.
        "is_preregistered_budget": (bool(CONFIRM_BUDGET_DECLARED) and int(fold) != 0
                                    and int(max_steps) == int(CONFIRM_STEPS)),
        "confirm_steps": int(CONFIRM_STEPS),
        "confirm_budget_file": CONFIRM_BUDGET_FILE,
        "confirm_budget_sha256": CONFIRM_BUDGET_SHA256,
        "confirm_budget_rule": CONFIRM_BUDGET_RULE,
        "sensitivity_step": int(SENSITIVITY_STEP),
        "headline_declaration": HEADLINE_DECLARATION,
        "sensitivity_step_is_on_the_grid": (int(SENSITIVITY_STEP) % int(args.ckpt_every) == 0
                                            and int(SENSITIVITY_STEP) <= int(max_steps)),
        "cross_method_budgets": {"dann_nb3": int(CONFIRM_STEPS), "coral_nb3": 2000,
                                 "freematch_nb3": 1500},
        "budget_status": (
            f"DECIDED 2026-09-02 as {CONFIRM_STEPS} steps and hashed into "
            f"{CONFIRM_BUDGET_FILE} (sha256 {CONFIRM_BUDGET_SHA256[:16]}...) before any "
            f"confirmatory fold existed, per 7.14. BASIS: declared under a NAMED RULE, "
            f"corroborated by the curve -- neither a pure cost declaration (CORAL's) nor "
            f"derived from the metric. The rule reads the lambda horizon and no metric, "
            f"so its post-fold-0 timing does not make it a selection: it would have "
            f"given the same number stated before fold 0 ran. FOLD 0 runs at 8000 as the "
            f"development curve, so is_preregistered_budget is False there by "
            f"construction and True on folds 1-4 at {CONFIRM_STEPS}."),
        "selection_rule": ("NONE in this path. The budget is the selector; there is no "
                           "argmax, no smoothing and no label-free rule anywhere here."),

        # ---- cadences / batch ----
        "log_every": int(args.log_every), "ckpt_every": int(args.ckpt_every),
        "batch_size": int(args.batch_size), "true_batch": True,
        "gradient_accumulation": False,
        "gradient_accumulation_note": (
            "REMOVED, not merely unused (7.1a). It is invalid for every DA loss and DANN "
            "is no exception: micro-batching to 32 would hand D a rank-32 view of a batch "
            "whose entire purpose is to hold 256 samples of each domain."),
        "grad_checkpointing": grad_ckpt,
        "grad_checkpointing_note": (
            "MANDATORY at n_blocks=3 (7.1a): true batch 256 image-space OOMs a 14.56 GiB "
            "T4 without it. Verified bit-exact, and re-verified WITH the GRL present in "
            "test_da_dann.py -- the GRL sits above the checkpointed region."),
        "steps_per_epoch": steps_per_epoch, "epochs_planned": epochs_planned,

        # ---- 7.1a layer-wise LR, matched to round 4 exactly ----
        "lr": LR, "lr_decay": LR_DECAY, "weight_decay": WEIGHT_DECAY,
        "backbone_lr": (args.backbone_lr if n_blocks > 0 else None),
        "backbone_lr_effective": ((args.backbone_lr if args.backbone_lr is not None else LR)
                                  if n_blocks > 0 else None),
        "lr_protocol": ("zoobot_v2.0.1 defaults per PROJECT.md 7.1a: head_lr == top-block "
                        "lr, lr_decay 0.75, weight_decay 0.05, no LR schedule. Same values "
                        "round-4 CORAL used. Tuning budget for DANN: ZERO (7.14a)."),
        "pos_fraction": POS_FRACTION, "seed": SEED, "head_type": HEAD_TYPE,

        # ---- the DANN objective ----
        "dann_lambda_schedule": "2/(1+exp(-10p))-1, p = min(step/horizon, 1)",
        "dann_lambda_horizon": int(args.dann_lambda_horizon),
        "dann_lambda_horizon_note": (
            "DECLARED ABSOLUTE CONSTANT. NOT total_steps. It must be IDENTICAL in this "
            "development run and in every confirmatory run, or the two see different "
            "objectives at the same step and this curve stops predicting them -- silently, "
            "since both would still train and both would still look plausible."),
        "dann_lambda_zero": lambda_zero,
        "dann_lambda_zero_note": (
            "the CONTROL arm. lambda is pinned to 0.0 at every step but the GRL and D "
            "still run, so the control exercises the identical code path, consumes "
            "identical RNG, and carries identical parameter counts and optimizer groups. "
            "With lambda = 0 the GRL returns exactly-zero gradient, so the encoder and "
            "head see the domain term not at all -- test_da_dann.py asserts that this is "
            "bit-identical to removing the term entirely."),
        "dann_lambda_at_step0": dann_lambda_at_step(0, int(args.dann_lambda_horizon)),
        "dann_lambda_at_max_steps": (0.0 if lambda_zero else
                                     dann_lambda_at_step(max_steps,
                                                         int(args.dann_lambda_horizon))),
        "dann_domain_label_source": 1, "dann_domain_label_target": 0,
        "dann_d_topology": "640 -> 1024 -> 1024 -> 1",
        "dann_d_hidden": 1024,
        "dann_d_norm": "LayerNorm",
        "dann_d_norm_note": (
            "DELIBERATE substitution for tllib's BatchNorm1d. D is fed ONE tensor holding "
            "both domains, so any batch-coupled normalisation lets D read the domain label "
            "out of the batch statistics instead of the features. Also keeps the assembled "
            "model BatchNorm-free end to end (7.1a measured the encoder at 0 BatchNorm), "
            "which is asserted at startup, not assumed."),
        "dann_d_output": "logits (no sigmoid); loss = BCEWithLogitsLoss",
        "dann_d_dropout": float(args.dann_d_dropout),
        "dann_d_dropout_layers": int(args.dann_d_dropout_layers),
        "dann_d_dropout_note": (
            "Dropout 0.2 after each hidden activation. Belfiore et al. (2024, A&A) apply "
            "dropout between hidden layers; we apply it after each, given a target pool "
            "roughly 20x smaller. Topology 1024/1024 from Ganin/tllib, unmodified. "
            "Stated as a DEVIATION rather than as inheritance: the rate is theirs, the "
            "placement is ours. --dann-d-dropout-layers 1 restores the literal "
            "between-hidden-layers reading. Neither value was tuned."),
        "dann_d_train_eval_semantics": (
            "D is the FIRST module in this pipeline for which .train()/.eval() means "
            "anything -- 7.1a measured the encoder at 0 BatchNorm and 0 active dropout, "
            "so train/eval was a no-op everywhere until now. D trains in train mode; "
            "d_acc_train is measured there, as trained. d_acc_holdout is measured with D "
            "in EVAL (dropout off) and the mode restored, because a memorisation monitor "
            "carrying a random mask is not comparable to one without. The pair is "
            "therefore NOT a like-for-like difference -- read each column's trend."),
        "dann_single_forward_pass": True,
        "dann_single_forward_pass_note": (
            "source and target features are CONCATENATED into one tensor and D is called "
            "ONCE per step. Two calls would hand D the domain label under any "
            "batch-dependent normalisation; the single-pass form removes the question "
            "permanently rather than relying on the LayerNorm choice holding forever."),
        "dann_d_lr": LR, "dann_d_weight_decay": WEIGHT_DECAY,
        "dann_optimizer_groups": "ONE AdamW; D is one extra param group at the head LR",
        "dann_optimizer_note": (
            "No second optimiser and no separate D learning rate. Gradient reversal is "
            "what removes the need for a D/encoder balance -- that is the whole reason "
            "DANN was chosen over ADDA -- and a tuned D LR would also be a method-specific "
            "search that 7.14a's zero-tuning-budget position does not permit."),
        "dann_attachment": "head.forward_with_features[1] (post-adapter, post-LayerNorm)",
        "dann_attachment_note": (
            "THE SAME 640-d tensor round-1 CORAL and round-4 CORAL used, deliberately not "
            "moved. On nb3 the adversarial gradient flows through the adapter into "
            "stages.1/2/3. Note one real difference from CORAL: CORAL is shift-invariant "
            "so its LayerNorm BIAS gradient is analytically zero; D is an arbitrary "
            "function of the features, so DANN genuinely reaches the bias too."),
        "dann_d_holdout_frac": float(args.dann_d_holdout_frac),
        "dann_d_holdout_n": (None if holdout is None else int(holdout["n_holdout"])),
        "dann_d_holdout_indices": (None if holdout is None
                                   else [int(i) for i in holdout["holdout"]]),
        "dann_d_holdout_note": (
            "a fixed ~10% slice of the fold's target pool, DETERMINISTIC given seed+fold "
            "and recorded above by dataset position. Excluded from D's loss by removal "
            "from the target SAMPLER'S POOL (not by a loss mask, which would keep the "
            "images in the batch and shrink D's effective target batch below the 256 that "
            "7.4 pins). Scored under no_grad every step as d_acc_holdout; the gap against "
            "d_acc_target on the training batch is the memorisation signal."),
        "spectrum_every": int(args.spectrum_every),
        "morph_every": int(args.morph_every),
        "holdout_every": int(args.holdout_every),

        # ---- monitors ----
        "monitors": ["d_acc_train", "d_acc_holdout", "d_pred_source_frac",
                     "morph_head_auroc (condition 3)", "morph_head_auroc_pool",
                     "morph_proxy_auroc (no detector)", "morph_proxy_auroc_encoder",
                     "feat_std_per_dim (= src/tgt_perdim_std)",
                     "participation_ratio", "dims_at_90", "dims_at_99",
                     "lambda_current", "loss_domain_raw",
                     "ratio_lambda_domain_over_sup"],
        "morph_proxy_instrument": (
            "ONE pool, ML-FP (=1) vs P1-reserve (=0), TWO instruments. HEAD: the trained "
            "head's own lens logit, no fit -- the ~0.61-scale instrument that MOVED in "
            "the v5 failure, and the one condition 3 reads. PROBE: ridge linear probe on "
            "the adapter, AUROC on a held-out half -- the ~0.90-scale instrument that "
            "'stayed ~0.90' through those same failures, logged with NO detector."),
        "morph_proxy_n_fit_per_class": int(args.morph_n_fit_per_class),
        "morph_proxy_n_eval_per_class": int(args.morph_n_eval_per_class),
        "morph_proxy_ridge_alpha": MORPH_RIDGE_ALPHA,
        "morph_proxy_pool": (None if morph_info is None else morph_info),
        "morph_proxy_pool_membership": {
            "mlfp_in_source_train_negatives": 4455,
            "mlfp_total_on_disk": 4455,
            "mlfp_fraction_seen_in_training": 1.0,
            "mlfp_in_target_pools": 0,
            "mlfp_in_val": 0,
            "mlfp_in_test_negatives": 0,
            "p1_reserve_seen_anywhere": 0,
            "note": (
                "THE PROBE IS HALF-SEEN, ASYMMETRICALLY, and that is recorded rather "
                "than corrected. ALL 4,455 ML-FP objects are source_train negatives "
                "(label 0, source domain); they appear in no target pool, no val pool "
                "and no test negatives. The P1-reserve side is seen by nothing. So the "
                "probe's POSITIVE class is 100% training data and its NEGATIVE class is "
                "0%. Consequences, stated: (a) the ABSOLUTE level is not a clean "
                "generalisation number -- part of it can be memorisation of images the "
                "supervised BCE has already shaped; (b) the monitor's JOB is to detect "
                "DESTRUCTION of linear access, and a drop is a drop whether the images "
                "were seen or not, so the comparison against the lambda=0 control at the "
                "same step is unaffected -- both arms see exactly the same images. "
                "(c) This is NOT lens-label leakage: ML-FP and P1-reserve are both class "
                "0 in the lens task, so the probe's label is not the model's label. The "
                "historical instrument had the same property."),
        },
        "morph_proxy_note": (
            "RECOVERED, not invented. PROJECT_STATE.md records a `morphology_probe` role "
            "(ML-FP=1 vs P1-reserve=0, both real) and a scripts/linear_probe.py that "
            "selected checkpoints on it; BOTH were dropped at the splits_v6 rewrite. The "
            "data survived: 4,455 ML-FP hard negatives, and 23,160 of partition 1 that "
            "source_train's neg_random did not consume -- which IS the 'P1-reserve' the "
            "old instrument named. Label-free by construction: ML-FP are graded X = NOT "
            "a lens, P1-reserve are ungraded random galaxies, and no simulated image, no "
            "real-lens label and no test-split image is touched. Step-0 reading on raw "
            "frozen encoder features: 0.9214, against the historical 0.914 -- evidence "
            "the recovery is faithful, not a reproduction (the original sample, seed and "
            "probe hyperparameters are gone with linear_probe.py)."),
        # ---- brief v3 3 + 4.2: the pre-registered abort conditions ----
        "abort_mode": default_abort_mode(arm_key, fold) if args.abort_mode is None
                      else str(args.abort_mode),
        "abort_mode_source": ("explicit --abort-mode flag" if args.abort_mode is not None
                              else f"default for arm={arm_key} fold={fold}"),
        "abort_mode_rule": (
            "log_only on the DEVELOPMENT fold-0 arm and on the lambda=0 CONTROL; armed "
            "on CONFIRMATORY folds 1-4. Reason (brief v3 4.2): if DANN destroys "
            "classifier access, that IS the paper result and the full 8000-step curve is "
            "the evidence -- aborting at step 1200 throws away exactly what you would be "
            "aborting because you found. On a confirmatory fold the trace has no "
            "scientific value and a dead run is wasted T4 time. The control must always "
            "complete: it is the reference trajectory for every monitor."),
        "control_curve": (None if not args.control_curve
                          else os.path.abspath(args.control_curve)),
        "allow_control_gaps": bool(args.allow_control_gaps),
        "abort4_sustain": int(args.abort4_sustain),
        "abort_conditions": {
            "1_d_overpowering": ("held-out-target domain accuracy > 0.95 for 1000 "
                                 "consecutive steps WITH the domain loss not falling "
                                 "-> the SDAT AdamW failure; action is to switch to "
                                 "ADDA (design doc 6.1)"),
            "2_degenerate_collapse": ("d_pred_source_frac outside [0.2, 0.8] for 500 "
                                      "consecutive steps (design doc 6.2)"),
            "3_morph_vs_control": (
                f"SECONDARY / graded early warning. {ABORT_COND3_COLUMN} below the "
                f"lambda=0 control's value AT THE SAME ABSOLUTE STEP by more than M, "
                f"sustained 500 steps. M is currently "
                f"{'UNSET -- 3a is INACTIVE' if args.abort_cond3_margin is None else args.abort_cond3_margin}. "
                f"Reworded from the design doc: 'adversarial pressure has degraded "
                f"linear access to morphology relative to unfreezing alone', NOT 'the "
                f"documented failure has recurred' -- see morph_proxy_abort_note."),
            "3b_morph_sign_flip": (
                f"**PRIMARY DETECTOR.** {ABORT_COND3_COLUMN} < {COND3_SIGN_FLIP}. The "
                f"historical failure WAS a sign flip -- the head instrument fell ~0.61 "
                f"to ~0.37 (plain DANN) / ~0.33 (IW-CDAN), decisively below chance -- so "
                f"on the head column this reproduces the actual event with NO invented "
                f"threshold and no calibration. No sustain."),
            "4_rank_collapse": (f"participation ratio below 20 of 640 (either domain) "
                                f"for {args.abort4_sustain} steps. The sustain is a "
                                f"DECLARED ADDITION; the design doc gives none."),
        },
        "abort_cond3_column": ABORT_COND3_COLUMN,
        "abort_cond3_interpretation_columns": list(COND3_INTERPRETATION_COLUMNS),
        "abort_cond3_interpretation_note": (
            "the two HEAD columns are logged with NO DETECTOR ATTACHED. Arm 1 showed "
            "what they actually measure: ML-FP and P1-reserve are BOTH class 0 in the "
            "lens task, so a head fine-tuned to give both low lens logits drives them "
            "together -- the column falls to chance because THE CLASSIFIER SUCCEEDED, "
            "not because morphology access was destroyed. Measured on the lambda=0 "
            "control: 0.5657 at step 0 -> median 0.4948, 54% of readings below 0.5, "
            "with J = 0.033 ~= the standard error of an at-chance AUROC at n=256."),
        "abort_cond3b_sustain": int(args.abort3b_sustain),
        "abort_advisory_only_detectors": list(ADVISORY_ONLY_DETECTORS),
        "abort_advisory_note": (
            "conditions 3 and 4 are ADVISORY ON EVERY ARM, confirmatory included: they "
            "compute, they log, they record when they WOULD have fired, and they never "
            "stop a run. Conditions 1 and 2 stay ARMED. Derived from arms 1 and 2 on "
            "2026-08-30 and NOT pre-registered -- see derived_from_control_and_"
            "development_arms in runs/dann_r5_prereg.json."),
        "abort_cond4_src_margin": (None if args.abort_cond4_src_margin is None
                                   else float(args.abort_cond4_src_margin)),
        "abort_cond4_tgt_margin": (None if args.abort_cond4_tgt_margin is None
                                   else float(args.abort_cond4_tgt_margin)),
        "abort_cond4_form": (
            "DIFFERENTIAL against the lambda=0 control at matched absolute steps, src "
            "and tgt SEPARATELY. The absolute 'PR < 20 of 640' is WITHDRAWN: it was "
            "ported from the SimSiam finding in a different feature space, and on arm 1 "
            "it fired 7,999 of 8,000 times -- the LP->FT init sits at PR ~9-10 before a "
            "single step and the run never exceeded 13.40."),
        "abort_respecified_from_control": {
            "status": "DERIVED FROM THE CONTROL ARM -- NOT a pre-registration",
            "date": "2026-08-28",
            "source_run": "runs/dann_arm1 (dann_nb3_lambda0, fold 0, 8000 steps)",
            "why": ("the pre-registered decision tree anticipated 3a being uncalibrable "
                    "with 3b carrying the condition. It did NOT anticipate 3b firing on "
                    "the control. The tree ran out."),
            "why_this_is_clean": (
                "arm 2 is log_only, so NO DETECTOR ACTS ON THE REPORTED DEVELOPMENT RUN. "
                "The thresholds were changed BEFORE any run they could influence; they "
                "bind on folds 1-4 only."),
            "withdrawn": {
                "condition_3_on_morph_head_auroc_pool": (
                    "WITHDRAWN. 3a was UNCALIBRABLE (3J = 0.0989 > the 0.04 ceiling) and "
                    "3b FIRED ON THE CONTROL (54% of readings below 0.5 at lambda = 0)."),
                "condition_4_absolute_PR_lt_20": (
                    "WITHDRAWN. Fired 7,999 of 8,000 steps on the control; the run's "
                    "maximum PR was 13.40, so the threshold was never reachable."),
            },
        },
        "abort_cond3_median_window_k": int(args.abort3_median_window),
        "abort_cond3_median_window_note": (
            "k=1 is no smoothing. k>1 makes 3a threshold the trailing median of the last "
            "k PAIRED DIFFERENCES (the difference is what 3a reads, so the difference is "
            "what is smoothed -- smoothing each side separately would need an assumption "
            "about the control's own smoothness that nothing supports). **k MUST BE "
            "DECLARED BEFORE M**: a window chosen after a candidate M is on the table has "
            "been selected to make that M work. With k>1 the window must be FULL before "
            "3a evaluates, so the earliest possible fire moves later by (k-1)*morph_every "
            "steps -- 175 at k=8, morph_every=25, against a 500-step sustain."),
        "abort_cond3_declaration_criterion": {
            "rule": ("Let J = median absolute step-to-step change in "
                     "morph_head_auroc_pool over arm 1, steps 2000-8000. If a candidate "
                     f"M >= {COND3_M_J_FACTOR:g}*J can be declared while "
                     f"M <= {COND3_M_CEILING}: declare M, hash it, proceed with 3a "
                     "active. Else: declare 3a UNCALIBRABLE at this pool size, record J "
                     "and the reason, and let 3b carry condition 3 alone."),
            "feasible_iff": f"{COND3_M_J_FACTOR:g}*J <= {COND3_M_CEILING}",
            "ceiling_rationale": (f"{COND3_M_CEILING} is roughly 60% of the whole-pool "
                                  f"headroom over chance (0.0657); an M above it would "
                                  f"only fire in territory 3b already covers"),
            "factor_status": f"the {COND3_M_J_FACTOR:g}x factor is a DECLARED judgement call",
            "uncalibrable_is_legitimate": (
                "declaring 3a UNCALIBRABLE is a legitimate outcome, NOT a failure. It is "
                "reported as a property of the historical instrument, which was never "
                "designed as a graded monitor."),
            "computed_by": ("run_dann_r5.py --morph-stats <arm1>/steps.csv, or "
                            "automatically at the end of every run"),
            "pre_registered": ("this criterion was fixed BEFORE arm 1 ran. PROJECT.md "
                               "7.17 / runs/dann_r5_prereg.json."),
        },
        "abort_cond3_column_note": (
            "BOTH DETECTORS read morph_head_auroc_pool -- the trained head's own lens "
            "logit on ALL 256 pool images. The head is NOT FITTED, so it needs no "
            "held-out half; and neither detector is the head-vs-probe contrast (3a is "
            "head-to-control across arms, 3b is head-to-0.5), so the eval half's "
            "same-images argument does not bind on them. Whole-pool step 0 is 0.565735, "
            "headroom 0.0657, against eval-half 0.578125 / 0.0781: 3b becomes marginally "
            "MORE sensitive, which is a better-estimated number rather than a looser "
            "threshold. morph_head_auroc (eval half) stays logged FOR INTERPRETATION -- "
            "paired with morph_proxy_auroc on the same 128 images, head down + probe held "
            "= classifier ACCESS destroyed, both down = the REPRESENTATION went. That is "
            "a by-eye diagnostic, not a streak counter."),
        "abort_cond3_margin": (None if args.abort_cond3_margin is None
                               else float(args.abort_cond3_margin)),
        "abort_cond3_margin_status": (
            ("UNSET. Condition 3a is INACTIVE and refuses to evaluate; an --abort-mode "
             "armed run REFUSES TO START. The margin is declared off arm 1's lambda=0 "
             "control trajectory ON THE HEAD COLUMN and hashed before arm 2. It is NOT "
             "defaulted to brief v3's 0.10: that was declared against the frozen probe "
             "(step 0 = 0.9202, headroom 0.4202) and is scale-wrong for the head column "
             "(step 0 ~= 0.5781, headroom ~0.078), where a 0.10 drop lands below chance.")
            if args.abort_cond3_margin is None else
            (f"SET to {args.abort_cond3_margin}. Must be the value hashed after arm 1 "
             f"and before arm 2, and must not move afterwards.")),
        "morph_proxy_abort_note": (
            "THREE thresholds are withdrawn, and the REASON matters. (1) The design "
            "doc's 0.55 was NOT invented -- PROJECT_STATE v2.md 6/8-F4 records the "
            "trained HEAD's morphology AUROC falling ~0.61 -> ~0.37 under plain DANN and "
            "~0.33 under IW-CDAN, and 0.55 sat just under that ~0.61 baseline. It is "
            "withdrawn because it is meaningless on the FROZEN-PROBE column, which reads "
            "~0.92. (2) Brief v2's 0.6910 rescaling of it is withdrawn because rescaling "
            "a threshold across a change of instrument assumes the two measure the same "
            "thing. (3) Brief v3's 0.10 control margin is withdrawn because it was "
            "declared against the frozen probe's 0.4202 headroom and is scale-wrong for "
            "the head column's ~0.078. What replaces them: condition 3 now READS THE "
            "HEAD COLUMN, 3b (AUROC < 0.5) is the primary detector and needs no "
            "threshold at all, and 3a's margin is declared off arm 1's own control "
            "trajectory and hashed before arm 2."),
        "feature_scale_monitor": (
            "mean PER-DIMENSION std is the headline indicator, NOT mean ||f||. 7.15: "
            "round-4 CORAL's total feature variance fell 56x while mean ||f|| moved 2.5%, "
            "because ||z|| is dominated by the mean vector while covariance depends only "
            "on the deviation. Mean ||f|| is still logged, as an indicator KNOWN to be "
            "blind to the thing it was originally chosen to catch."),
        "rank_monitor": (
            "participation_ratio / dims_at_90 / dims_at_99, per domain. Logged separately "
            "from the scale monitors because 7.15 established that round 4's collapse was "
            "SCALE, not RANK (dims@90 29->26, dims@99 164->168, no direction zeroed) and "
            "that distinction is unrecoverable if the two are merged."),
        "health_anchors": DANN_HEALTH_ANCHORS,

        # ---- eval / precision ----
        "encoder_eval": ("frozen_feature_cache" if n_blocks == 0 else "live_encode"),
        "encoder_eval_note": (
            "the frozen cache is INVALID at n_blocks>0 (7.1a). Scoring a real nb3 "
            "checkpoint from the stale cache once gave TPR 0.194 against a logged 0.020 -- "
            "a different model, not a rounding difference."),
        "augmentation": "weak (+-3px translate, PROJECT.md 7.3)",
        "fp16": bool(args.fp16), "precision": ("fp16" if args.fp16 else "fp32"),
        "eval_precision": ("fp16" if args.eval_fp16 else "fp32"),
        "device": dev, "num_workers": int(args.num_workers),
        "single_gpu": ("no DataParallel / DistributedDataParallel / device sharding "
                       "anywhere in this path -- halving the per-device batch would halve "
                       "what D sees per step"),

        # ---- gate ----
        "expect_step0_tpr": float(FOLD_GATE[fold]["tpr"]),
        "expect_step0_auroc": float(FOLD_GATE[fold]["auroc"]),
        "expect_head_l2": BASELINE_A_HEAD_L2,
        "step0_tpr_tolerance": step0_tpr_tolerance(fold, n_blocks),
        "step0_auroc_tolerance": STEP0_AUROC_TOL,
        "step0_gate_skippable": False,
        "hard_gate": (
            "NON-SKIPPABLE. Step-0 TPR@1e-3 AND AUROC must reproduce the phase-1 head's "
            "own per-fold value or the run ABORTS. There is no bypass flag, by design: "
            "the phase-1 head is fold-INDEPENDENT (Baseline A is source-only and the "
            "source does not vary by fold), so the head-norm check proves the right FILE "
            "and this gate is the only thing that proves the right FOLD. Tolerances are "
            "chosen by the EVAL PATH, not by a flag (7.7): TPR at 0 on the deterministic "
            "cached path and at ONE POSITIVE on the live-encode path, AUROC at 1e-3 "
            "always. Tolerance 0 on TPR across a re-encode is what cost a session on "
            "FreeMatch fold 2 -- one positive, on a correct run."),

        # ---- the da_loss hook trick, declared rather than discovered ----
        "da_hook_slot": ("Trainer.coral (see dann_round5.install_dann). train_skeleton.py "
                         "is UNMODIFIED; DannLoss is installed into the one attribute "
                         "train_step reaches its DA term through, at coral_weight=1.0. "
                         "coral_weight=1.0 is not a fudge: canonical DANN adds the domain "
                         "term unweighted and lambda lives in the GRL."),
        "round1_dann_fold0_reference": {
            "tpr_at_1e-3_last_checkpoint": 0.19387755102040816,
            "read_off": "step_15000 of 75 checkpoints (the R1 last-checkpoint rule)",
            "oracle_best": 0.21428571428571427, "oracle_step": 6800,
            "curve_range": [0.09183673469387756, 0.21428571428571427],
            "curve_mean": 0.1812244897959184,
            "source": "runs/curve_dann_fold0/results.json",
            "note": ("MEASURED, not the '~0.19' that had been quoted from memory -- it "
                     "happens to be right. NOT COMPARABLE step-for-step to this round: "
                     "that run was CACHED-FEATURE with a 640->64->640 bottleneck "
                     "adapter, SGD (lr 1e-3, momentum 0.9, wd 1e-4) rather than AdamW, a "
                     "BatchNorm discriminator, and a lambda computed from total_steps. "
                     "Worth one observation: 0.193878 is EXACTLY Baseline A's own fold-0 "
                     "step-8000 value to 16 digits, which is 7.4's 'marginal alignment "
                     "sits at baseline' in its sharpest form."),
        },
        "reporting_note": ("fold 0 is DEVELOPMENT (7.11 / 7.14) and no number from it "
                           "enters a headline. Report TPR@FPR=1e-3 as a RATE, never as "
                           "counts (PROJECT.md 9)."),
    }
    return cfg


# ===========================================================================
# arm construction
# ===========================================================================

def build_arm(arm_key, args, paths, *, verbose=True) -> dict:
    """Datasets, loaders (with the target holdout removed), encoder, Trainer, D."""
    spec = ARMS[arm_key]
    fold = int(args.fold)
    n_blocks = int(spec["n_blocks"])
    batch_size = int(args.batch_size)
    max_steps = int(args.max_steps)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(args.fp16)
    grad_ckpt = bool(args.grad_checkpointing) and n_blocks > 0
    head_ckpt = os.path.abspath(paths["init_head_from"])

    # ---- FIRST, before anything expensive: an ARMED run with an UNSET condition-3
    # margin is refused. The same check lives in DannAbortWatch (so a test can hit it
    # directly and so no caller can construct an armed watch without a margin); this
    # one exists purely so the refusal lands in SECONDS rather than after a ~1 GB
    # encoder build and a 256-image morphology pool. ----
    _mode = (str(args.abort_mode) if args.abort_mode is not None
             else default_abort_mode(arm_key, fold))
    # Conditions 3 and 4 are ADVISORY on every arm now, so an unset margin there no
    # longer refuses an armed run -- it disables a logging line, not a safety net. The
    # construction still happens (cheaply, before the ~1 GB encoder build) so that any
    # refusal the arming detectors DO carry lands in seconds rather than minutes.
    if _mode == ABORT_MODE_ARMED:
        DannAbortWatch(mode=ABORT_MODE_ARMED, cond3_margin=args.abort_cond3_margin,
                       cond4_src_margin=args.abort_cond4_src_margin,
                       cond4_tgt_margin=args.abort_cond4_tgt_margin, verbose=False)

    assert_no_bf16("fp16" if fp16 else "fp32", device)

    if not os.path.isfile(head_ckpt):
        raise SystemExit(
            f"[dann-r5] ABORT: phase-1 head not found: {head_ckpt}\n"
            f"  runs/ is not packed into data.zip, so on a fresh remote session point "
            f"--init-head-from straight at it.")
    l2 = round(head_l2_norm(head_ckpt), 6)
    if abs(l2 - BASELINE_A_HEAD_L2) > 1e-4:
        raise SystemExit(
            f"[dann-r5] ABORT: phase-1 head ||.||_2 = {l2} but the pinned Baseline A "
            f"value is {BASELINE_A_HEAD_L2}. This is not the declared phase-1 head -- "
            f"check --init-head-from.\n"
            f"  NON-SKIPPABLE, and there is deliberately no override flag: the head is "
            f"fold-independent, so this check plus the per-fold step-0 gate are the ONLY "
            f"two things standing between a mis-pointed run and a plausible wrong number.")

    # ---- datasets: ONE view per domain (DANN needs no weak/strong pair) ----
    src_ds = LensDataset(paths["splits"], paths["data_root"], role="source_train",
                         aug="weak", n_views=1)
    tgt_ds = LensDataset(paths["splits"], paths["data_root"], role="target", fold=fold,
                         aug="weak", n_views=1)
    assert np.all(tgt_ds.labels == UNLABELED), "target labels must all be UNLABELED(-1)"

    # ---- the held-out target slice: removed from the SAMPLER'S POOL ----
    holdout = build_target_holdout(tgt_ds, frac=float(args.dann_d_holdout_frac),
                                   seed=SEED + fold)
    if verbose:
        print(f"[holdout] target pool {holdout['n_pool']} -> D trains on "
              f"{holdout['n_train']}, {holdout['n_holdout']} held out "
              f"({holdout['frac_actual']*100:.1f}%), deterministic at seed "
              f"{holdout['seed']}", flush=True)

    steps = len(src_ds) // batch_size
    epochs_planned = max(1, int(math.ceil(max_steps / max(steps, 1))))

    src_loader = make_balanced_loader(src_ds, batch_size, use_latent=False,
                                      num_batches=steps, pos_fraction=POS_FRACTION,
                                      seed=SEED + fold, num_workers=int(args.num_workers),
                                      pin_memory=True,
                                      persistent_workers=int(args.num_workers) > 0)
    # Built directly rather than through make_balanced_loader, because the pool has
    # to be the HOLDOUT-FREE one. Same sampler class, same seed offset (+1000) round
    # 4 used, so the only difference from a full-pool run is the missing 10%.
    tgt_sampler = ClassBalancedBatchSampler(holdout["train_pos"], holdout["train_neg"],
                                            batch_size, num_batches=steps,
                                            pos_fraction=POS_FRACTION,
                                            seed=SEED + fold + 1000)
    tgt_loader = DataLoader(tgt_ds, batch_sampler=tgt_sampler,
                            num_workers=int(args.num_workers), pin_memory=True,
                            persistent_workers=int(args.num_workers) > 0)
    paired = PairedDomainLoader(src_loader, tgt_loader)

    from zoobot_encoder import ZoobotEncoder
    if verbose:
        print(f"[load] building the Zoobot encoder (n_blocks={n_blocks}, "
              f"grad_checkpointing={grad_ckpt}) ...", flush=True)
    encoder = ZoobotEncoder(encoder_mode="frozen", n_blocks=n_blocks, device=device,
                            verbose=False, grad_checkpointing=grad_ckpt)

    tcfg = TrainConfig(
        splits_path=paths["splits"], data_root=paths["data_root"], fold=fold,
        head_type=HEAD_TYPE, encoder_mode="frozen", lr=LR, weight_decay=WEIGHT_DECAY,
        batch_size=batch_size, pos_fraction=POS_FRACTION,
        use_coral=False,                     # install_dann flips the hook on
        total_steps=max_steps, epochs=epochs_planned, steps_per_epoch=steps,
        max_steps=max_steps, seed=SEED + fold, device=device,
        num_workers=int(args.num_workers), amp=fp16,
        n_blocks=n_blocks, backbone_lr=args.backbone_lr, lr_decay=LR_DECAY,
        init_head_from=head_ckpt, grad_checkpointing=grad_ckpt)
    trainer = Trainer(tcfg, encoder=encoder)

    dann = install_dann(trainer,
                        lambda_horizon=int(args.dann_lambda_horizon),
                        lambda_zero=bool(spec["lambda_zero"] or args.dann_lambda_zero),
                        holdout_frac=float(args.dann_d_holdout_frac),
                        spectrum_every=int(args.spectrum_every),
                        dropout=float(args.dann_d_dropout),
                        n_dropout_layers=int(args.dann_d_dropout_layers),
                        d_lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- NO BATCHNORM ANYWHERE. Asserted at startup, not assumed. ----
    bn = assert_no_batchnorm(trainer.head, trainer.encoder.model, dann,
                             context="head + encoder + discriminator")
    if verbose:
        print(f"[bn] {bn['n_modules_scanned']} modules scanned, 0 BatchNorm "
              f"(head + encoder + D)", flush=True)

    # ---- standardizer from the CACHED source features (train/eval consistency) ----
    cache = ev.FeatureCache(paths["cache_dir"])
    if not cache.has("source_train"):
        raise FileNotFoundError(
            f"source_train.npz not found under {paths['cache_dir']}; run cache_features.py "
            f"first. (The standardizer is fitted from the cached SOURCE features so that "
            f"training and evaluation standardize identically -- this is the one cache read "
            f"that stays valid at n_blocks>0, because it is a fixed constant of the source "
            f"domain, not a function of the encoder being trained.)")
    src_feats = torch.from_numpy(
        np.asarray(cache.load("source_train")["features"],
                   dtype=np.float32)).to(trainer.device)
    trainer.standardizer.fit(src_feats)
    del src_feats

    holdout_images = materialise_holdout_images(tgt_ds, holdout["holdout"])

    # ---- the morphology proxy: the recovered ML-FP vs P1-reserve instrument ----
    morph_pool = build_morph_pool(paths["splits"], paths["data_root"],
                                  n_fit_per_class=int(args.morph_n_fit_per_class),
                                  n_eval_per_class=int(args.morph_n_eval_per_class),
                                  seed=SEED + fold, verbose=verbose)
    morph = MorphProxy(morph_pool, alpha=MORPH_RIDGE_ALPHA)

    # ---- the lambda=0 control curve, for abort condition 3 ----
    # Loaded HERE so a missing / mismatched control fails in seconds, not at step 3000.
    control = None
    if args.control_curve:
        control = ControlCurve(os.path.abspath(args.control_curve))
        # The step LABELS this run will actually write morph readings at. The measurement
        # fires when gstep % morph_every == 0 and the row it lands in is labelled
        # gstep + 1, so the grid is offset ONE, not zero. Computing want as
        # range(0, max_steps + 1, morph_every) made step 0 -- which no run ever writes --
        # look like a missing control step, and would have pushed arm 2 onto
        # --allow-control-gaps to clear an abort that described nothing real.
        want = [g + 1 for g in range(0, max_steps, int(args.morph_every))]
        missing = control.covers(want)
        if missing and not args.allow_control_gaps:
            raise SystemExit(
                f"[dann-r5] ABORT: the control curve {args.control_curve} is missing "
                f"{len(missing)} of the {len(want)} steps this run will measure "
                f"morphology at (first: {missing[:6]}).\n"
                f"  Condition 3 compares step for step and REFUSES TO INTERPOLATE, so a "
                f"cadence mismatch does not coarsen the comparison -- it deletes it. The "
                f"control's stride is {control.summary()['stride_mode']} and this run's "
                f"--morph-every is {args.morph_every}; make them equal, or pass "
                f"--allow-control-gaps to run with the comparison refused at those steps "
                f"(recorded in run_config.json).")
        if verbose:
            cs = control.summary()
            print(f"[abort3] control curve: {cs['n_steps']} readings, steps "
                  f"{cs['step_min']}-{cs['step_max']}, stride {cs['stride_mode']}; "
                  f"{len(missing)} of this run's {len(want)} morph steps missing",
                  flush=True)
    elif verbose:
        print("[abort3] no --control-curve given: condition 3 (morph vs the lambda=0 "
              "control) is INACTIVE. Conditions 1, 2, 3b and 4 still run. Expected for "
              "the control arm itself, which IS the reference.", flush=True)

    cfg_json = build_config(arm_key, args, fold=fold, paths=paths, init_head_l2=l2,
                            holdout=holdout, steps_per_epoch=steps,
                            epochs_planned=epochs_planned, device=device,
                            morph_info=morph_pool["info"])
    cfg_json["no_batchnorm_check"] = bn

    # Constructed BEFORE the encoder forward that would cost real time: an armed run
    # with an unset condition-3 margin refuses here, in seconds.
    watch = DannAbortWatch(mode=cfg_json["abort_mode"], control=control,
                           cond3_margin=args.abort_cond3_margin,
                           cond4_sustain=int(args.abort4_sustain),
                           cond3_median_window=int(args.abort3_median_window),
                           cond4_src_margin=args.abort_cond4_src_margin,
                           cond4_tgt_margin=args.abort_cond4_tgt_margin,
                           cond3b_sustain=int(args.abort3b_sustain),
                           verbose=verbose)
    if verbose:
        m = ("UNSET -- 3a INACTIVE, 3b (sign flip) still live"
             if args.abort_cond3_margin is None else str(args.abort_cond3_margin))
        m4 = (f"src {args.abort_cond4_src_margin} / tgt {args.abort_cond4_tgt_margin}")
        print(f"[abort] cond4 DIFFERENTIAL vs control, margins {m4}", flush=True)
        print(f"[abort] mode={cfg_json['abort_mode']} "
              f"({cfg_json['abort_mode_source']}); conditions 1/2/3/3b/4 evaluated every "
              f"step on column {ABORT_COND3_COLUMN}; cond3 margin {m}; median window k="
              f"{args.abort3_median_window}; cond4 sustain {args.abort4_sustain}",
              flush=True)

    return dict(cfg_json=cfg_json, spec=spec, arm=arm_key, fold=fold, n_blocks=n_blocks,
                batch_size=batch_size, max_steps=max_steps, steps=steps,
                epochs_planned=epochs_planned, device=device, fp16=fp16,
                grad_ckpt=grad_ckpt, trainer=trainer, dann=dann, paired=paired,
                src_loader=src_loader, tgt_loader=tgt_loader, encoder=encoder,
                src_ds=src_ds, tgt_ds=tgt_ds, holdout=holdout,
                holdout_images=holdout_images, morph=morph, control=control,
                watch=watch, head_ckpt=head_ckpt)


# ===========================================================================
# checkpoints: D rides along, because Trainer.save does not know about it
# ===========================================================================

def dann_schedule_payload(cfg, dann, gstep, **extra) -> dict:
    """What rides in ``Trainer.save``'s ``schedule`` slot.

    ``Trainer.save`` stores the head, the standardizer and the unfrozen encoder
    blocks. **D is none of those**, and it holds ~1.7M trained parameters. A
    checkpoint without ``discriminator_state`` is not continuable and not
    re-scorable as the model that produced it -- the same silent-restart class
    PROJECT.md 7.10 paid a round to find (FreeMatch's SAT buffers, the schedule
    position, AdamW's moments). ``lambda`` itself needs no rescue: it is a pure
    function of ``global_step`` and the declared horizon.
    """
    return {"method": METHOD, "global_step": int(gstep),
            "dann_lambda_horizon": int(dann.lambda_horizon),
            "dann_lambda_zero": bool(dann.lambda_zero),
            "dann_lambda_at_step": float(dann.lambda_at(gstep)),
            "discriminator_state": dann.state_dict_d(),
            "arm": cfg.get("arm"), "fold": cfg.get("fold"),
            **extra}


def save_resume(trainer, dann, P, cfg, *, gstep, epoch, batch_in_epoch,
                rng_epoch_start):
    """Weights + AdamW moments (including D's group) + GradScaler + D + RNG, atomically.

    HONEST LIMIT, stated rather than buried (inherited verbatim from round 4): the
    SAMPLE ORDER is restored exactly -- the batch sampler is a pure function of
    seed+epoch -- and the RNG state captured at the START of the interrupted epoch is
    restored before the loader iterator is rebuilt. But DataLoader worker seeds are
    derived when an iterator is created and this run uses persistent workers, so the
    AUGMENTATION OFFSETS drawn after a resume are not guaranteed to equal those an
    uninterrupted run would have drawn. That is a +-3px translate on individual
    images; it restarts no schedule, no buffer and no moment. Every resume is
    recorded in run.json so a resumed curve is never mistaken for an unresumed one.
    """
    tmp = P.resume_pt + ".tmp"
    trainer.save(tmp, full_state=True, schedule=dann_schedule_payload(
        cfg, dann, gstep,
        epoch=int(epoch), batch_in_epoch=int(batch_in_epoch),
        scaler=(trainer.scaler.state_dict() if trainer._amp_enabled else None),
        rng_epoch_start=rng_epoch_start, rng_now=_rng_state(),
        saved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    os.replace(tmp, P.resume_pt)


def load_resume(trainer, dann, P):
    if not os.path.isfile(P.resume_pt):
        return None
    ck = trainer.load(P.resume_pt, map_location=trainer.device, load_training_state=True)
    sch = ck.get("schedule") or {}
    if sch.get("scaler") and trainer._amp_enabled:
        trainer.scaler.load_state_dict(sch["scaler"])
    dstate = sch.get("discriminator_state")
    if not dstate:
        raise SystemExit(
            f"[dann-r5] ABORT: {P.resume_pt} carries no discriminator_state. D's ~1.7M "
            f"parameters would restart from a fresh init while the encoder and head "
            f"continued -- a silent restart of exactly the kind 7.10 documented. Refusing "
            f"to continue from this checkpoint.")
    dann.discriminator.load_state_dict(
        {k: v.to(trainer.device) for k, v in dstate.items()})
    if int(sch.get("dann_lambda_horizon", -1)) != int(dann.lambda_horizon):
        raise SystemExit(
            f"[dann-r5] ABORT: resume checkpoint declares lambda_horizon "
            f"{sch.get('dann_lambda_horizon')} but this run is configured for "
            f"{dann.lambda_horizon}. Continuing would splice two different objectives "
            f"into one curve.")
    return sch


def assert_checkpoint_retention(ckpt_dir, budget, ckpt_every, verbose=True):
    """Every retained checkpoint must exist, and the count is recorded.

    Round 4's ``run_unfreeze_freematch.assert_checkpoint_retention`` is NOT reused: it
    hardcodes ``include_step0=True`` and demands a ``step_0.pt``, which this path
    deliberately does not write (step 0 is captured as a LOGIT dump, ``true_init.npz``,
    which is what the gate actually reads). Calling it here would abort every healthy
    run. The check it performs -- fail loudly on a missing file, at the end of
    training, on the training box -- is reproduced exactly.
    """
    want = list(range(int(ckpt_every), int(budget) + 1, int(ckpt_every)))
    have = sorted(int(m.group(1)) for m in
                  (re.fullmatch(r"step_(\d+)\.pt", f)
                   for f in os.listdir(ckpt_dir) if f.endswith(".pt")) if m)
    missing = [s for s in want if s not in set(have)]
    if verbose:
        print(f"[retention] checkpoints {len(have)}/{len(want)}")
    if missing:
        raise SystemExit(
            f"[dann-r5] ABORT: checkpoint retention failed -- {len(missing)} of "
            f"{len(want)} retained checkpoints missing from {ckpt_dir}: "
            f"{missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  7.7: this was a COLLECTION bug last time, not a training bug, and it cost "
            f"the A1 study its unfrozen checkpoints. Packaging must refuse an incomplete "
            f"run; so does this.")
    return have, want


# ===========================================================================
# the step-0 gate
# ===========================================================================

def step0_gate(scorer, record, fold, *, tpr_tol, auroc_tol):
    """Score the LP->FT head itself. A KNOWN NULL: it must reproduce Baseline A's number.

    **NON-SKIPPABLE, AND IT ABORTS.** There is no warn tier and no bypass flag. Brief
    v2 makes this binding for a specific reason: the phase-1 head is
    fold-INDEPENDENT -- Baseline A is source-only and the source does not vary by fold,
    so for a ``LayerNorm(640) + Linear(640->1)`` head the source-only linear probe is
    unique and one head file serves all five folds. The head-norm check in
    ``build_arm`` therefore proves the right FILE and proves nothing about the fold.
    This gate is the only thing that proves the fold.

    THE TOLERANCES ARE CHOSEN BY THE EVAL PATH, NOT BY A FLAG (7.7). ``tpr_tol`` comes
    from :func:`step0_tpr_tolerance`: 0 on the deterministic cached path, one positive
    on the live-encode path. ``auroc_tol`` is 1e-3 always.

    Brief v2 asked for tolerance 0 on both. On the live-encode path that is the
    measured mistake 7.7 documents: the FreeMatch nb3 gate demanded exact
    reproduction, passed folds 0/1/3 and failed fold 2 by exactly one positive after a
    correct 16-minute setup, because live-encode SCORE drift is ~1e-2 and fold 2's
    closest positive sat 3.06e-03 above its threshold. The AUROC half is what carries
    the identification anyway: it is rank-based, ~100x the measured live-encode drift
    (1.2e-05) and ~7x tighter than a wrong head (7.6e-03). 7.7's worked example is
    that fold 1's head on fold 2's data gives TPR **exactly 0.336735** -- a
    legitimate-looking value that only AUROC catches.

    RESIDUAL RISK, STATED. The pinned table was scored from ``data/feature_cache`` on
    this machine; the run will be on a T4. 7.7 measured local-Ada-vs-T4 at 0.0204 (two
    positives) on the frozen ``nb0_image`` arm, but ALSO measured it to be
    arm-specific: identical CORAL nb3 weights gave identical TPR at all 9 shared
    checkpoints across venues, and that arm's step-0 gate came in exact. If this gate
    ever fires on venue drift rather than on a wrong fold, the fix is to re-score the
    pinned table ON the venue -- not to widen the gate.
    """
    from metrics import auroc as auroc_fn, tpr_at_fpr
    exp = FOLD_GATE[int(fold)]
    dst, secs, ps, ns = scorer.dump(0, name="true_init")
    y = np.concatenate([np.ones(ps.size, np.int64), np.zeros(ns.size, np.int64)])
    s = np.concatenate([ps, ns])
    tpr, au = float(tpr_at_fpr(y, s)), float(auroc_fn(y, s))
    dt, da = abs(tpr - exp["tpr"]), abs(au - exp["auroc"])
    print("\n" + "=" * 92)
    print(f"STEP-0 GATE (known null: this IS the phase-1 model) -- fold {fold}, "
          f"{int(ps.size)} test positives, scored in {secs:.0f}s")
    print(f"  TPR@1e-3 : {tpr:.6f}   expected {exp['tpr']:.6f}   |diff| {dt:.6f}  "
          f"(ABORT > {tpr_tol:.6f}"
          f"{' = exact, cached path' if tpr_tol == 0 else ' = one positive, live-encode path'})")
    print(f"  AUROC    : {au:.6f}   expected {exp['auroc']:.6f}   |diff| {da:.6f}  "
          f"(ABORT > {auroc_tol:.1e})")
    print(f"  ||head||2: {record.get('init_head_l2')}   pinned {BASELINE_A_HEAD_L2:.6f}  "
          f"(already hard-gated at 1e-4 in build_arm)")
    print("  NON-SKIPPABLE: the phase-1 head is fold-independent, so this gate is the "
          "only proof of fold identity.")
    print("=" * 92, flush=True)
    failed = [n for n, d, t in (("TPR@1e-3", dt, tpr_tol), ("AUROC", da, auroc_tol))
              if d > t]
    record["step0_gate"] = {"tpr": tpr, "auroc": au, "expected_tpr": exp["tpr"],
                            "expected_auroc": exp["auroc"], "tpr_abs_diff": dt,
                            "auroc_abs_diff": da,
                            "verdict": ("ABORT" if failed else "PASS"),
                            "failed_halves": failed,
                            "n_test_positives": int(ps.size),
                            "tpr_tolerance": tpr_tol, "auroc_tolerance": auroc_tol,
                            "skippable": False,
                            "true_init_logits": os.path.basename(dst)}
    if failed:
        raise SystemExit(
            f"[dann-r5] ABORT at the step-0 gate ({', '.join(failed)}): TPR@1e-3 = "
            f"{tpr:.6f} against the pinned {exp['tpr']:.6f} and AUROC {au:.6f} against "
            f"{exp['auroc']:.6f}.\n"
            f"  Step 0 IS the phase-1 model, so this is a plumbing failure, not a "
            f"result, and there is no flag to wave it through.\n"
            f"  Likely causes, in order: the WRONG --fold (the head is fold-independent, "
            f"so nothing upstream of here would notice), the wrong split/cache mount, an "
            f"fp16 eval encode against an fp32 reference, or a standardizer not fitted "
            f"from the cached source features.\n"
            f"  If and only if every one of those is excluded, suspect venue drift (7.7: "
            f"0.0204 local-Ada vs T4 on the frozen nb0_image arm, but MEASURED to be "
            f"arm-specific -- CORAL nb3 was identical across venues). The fix for venue "
            f"drift is to RE-SCORE the pinned table on the venue, never to widen this "
            f"gate.")
    print("[gate] step-0 known null reproduces Baseline A on this fold. Proceeding.\n",
          flush=True)
    return record["step0_gate"]


# ===========================================================================
# the training loop -- shared by the real run and by --dry-run
# ===========================================================================

def _print_peak_memory(record):
    alloc = torch.cuda.max_memory_allocated() / 2 ** 30
    resv = torch.cuda.max_memory_reserved() / 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print("\n" + "-" * 92)
    print(f"[gpu] {torch.cuda.get_device_name(0)}  |  {total:.2f} GB total  |  "
          f"{torch.cuda.device_count()} visible device(s), using cuda:0 only "
          f"(no DataParallel anywhere in this path)")
    print(f"[gpu] peak so far: {alloc:.2f} GB allocated / {resv:.2f} GB reserved")
    print(f"[gpu] reference:   3.29 GB on round-4 CORAL nb3 (same encoder, same batch, "
          f"same checkpointing, Kaggle T4). DANN adds D (~1.7M params) and one extra "
          f"512x640 activation, so a little above that is expected; well above it is not.")
    print("-" * 92 + "\n", flush=True)
    record["peak_gpu_gb_early"] = float(alloc)
    record["gpu_name"] = torch.cuda.get_device_name(0)
    record["gpu_total_gb"] = float(total)
    record["gpu_visible_devices"] = int(torch.cuda.device_count())


def train_loop(*, A, P, cfg, record, scorer, resume_sched=None, verbose=True):
    """The one loop. ``cfg`` is the run_config dict; ``A`` the built arm."""
    trainer, dann = A["trainer"], A["dann"]
    max_steps = int(cfg["max_steps"])
    log_every = int(cfg["log_every"])
    ckpt_every = int(cfg["ckpt_every"])
    holdout_every = max(1, int(cfg.get("holdout_every", 1)))
    morph_every = max(1, int(cfg.get("morph_every", 1)))
    morph = A.get("morph")
    watch = A.get("watch")
    steps_per_epoch = max(1, int(A["steps"]))
    t_start = time.time()

    gstep = int(resume_sched.get("global_step", 0)) if resume_sched else 0
    start_epoch = int(resume_sched.get("epoch", 1)) if resume_sched else 1
    skip_in_epoch = max(0, gstep - (start_epoch - 1) * steps_per_epoch) if resume_sched else 0
    if resume_sched:
        _set_rng_state(resume_sched.get("rng_epoch_start"))
        for p in (P.steps_csv, P.trace_csv, P.ckpt_csv):
            n = _truncate_csv_after(p, gstep)
            if n:
                print(f"[resume] dropped {n} row(s) past step {gstep} from "
                      f"{os.path.basename(p)}", flush=True)
        print(f"[resume] restored at step {gstep} (epoch {start_epoch}, "
              f"{skip_in_epoch} batch(es) to replay in this epoch)", flush=True)

    window = DannStepWindow()           # reset every log_every
    ckpt_window = DannStepWindow()      # reset every ckpt_every
    skipped_cum = float(record.get("fp16_skipped_cumulative", 0.0))
    peak_reported = False
    stop = False
    aborted, abort_reason = False, None
    last_holdout, last_morph = {}, {}

    epochs_planned = max(int(A["epochs_planned"]), math.ceil(max_steps / steps_per_epoch))
    for epoch in range(start_epoch, epochs_planned + 1):
        A["src_loader"].batch_sampler.set_epoch(epoch)
        A["tgt_loader"].batch_sampler.set_epoch(epoch)
        rng_epoch_start = _rng_state()
        trainer.head.train()
        seen_in_epoch = 0
        for sbatch, tbatch in A["paired"]:
            if seen_in_epoch < skip_in_epoch:
                seen_in_epoch += 1
                if verbose and seen_in_epoch % 50 == 0:
                    print(f"[resume]   replayed {seen_in_epoch}/{skip_in_epoch} batches",
                          flush=True)
                continue
            seen_in_epoch += 1

            # ---- lambda for THIS step. Recomputed from `gstep` and the DECLARED
            # horizon; nothing is incremented, nothing is buffered, so a resume at
            # step k gets exactly what an uninterrupted run would have. ----
            dann.set_step(gstep)

            scale_before = (trainer.scaler.get_scale() if trainer._amp_enabled else None)
            with test_label_lockout(verbose=False):
                st = trainer.train_step(sbatch, tbatch, step=gstep, total_steps=max_steps)
            skipped = bool(trainer._amp_enabled
                           and trainer.scaler.get_scale() < scale_before)

            # ---- the held-out-target monitor. NO grad, NO loss, never trained on.
            # D goes to EVAL (dropout off) inside holdout_domain_stats and the mode is
            # restored; the assertion below is cheap and catches a future edit that
            # forgets. ----
            if gstep % holdout_every == 0:
                with _rng_preserved():
                    last_holdout = holdout_domain_stats(
                        trainer, dann, A["holdout_images"], amp=trainer._amp_enabled)
                assert dann.discriminator.training, (
                    "the holdout monitor left D in eval mode -- every subsequent "
                    "training step would run without dropout")
            dann.set_holdout_stats(last_holdout)

            # ---- the morphology proxy. NO grad, NO loss, no lens label anywhere. ----
            if morph is not None and gstep % morph_every == 0:
                with _rng_preserved():
                    last_morph = morph.measure(trainer, amp=trainer._amp_enabled)
            dann.set_morph_stats(last_morph)

            stats = {**st, **dann.step_stats(sup_loss=st.get("sup_loss")),
                     "step_skipped": float(skipped),
                     "amp_scale": (float(trainer.scaler.get_scale())
                                   if trainer._amp_enabled else float("nan"))}
            gstep += 1
            # ---- the four pre-registered abort conditions. Evaluated EVERY step; the
            # raw morph-minus-control difference is logged whether or not anything
            # fires (brief v3 3). Whether a fire STOPS the run is `abort_mode`, which
            # is a recorded setting, not something derived here. ----
            if watch is not None:
                stats.update(watch.update(gstep, stats))
            skipped_cum += float(skipped)
            window.update(stats)
            ckpt_window.update(stats)
            _append(P.steps_csv, STEP_COLUMNS, {"step": gstep, **stats})

            if not peak_reported and gstep >= 5 and torch.cuda.is_available():
                _print_peak_memory(record)
                peak_reported = True

            if gstep % log_every == 0:
                row = window.row(gstep)
                row["wall_secs"] = round(time.time() - t_start, 1)
                _append(P.trace_csv, TRACE_COLUMNS, row)
                window.reset()
                if verbose:
                    eta = ((time.time() - t_start) / max(gstep, 1)
                           * (max_steps - gstep) / 60.0)
                    print(f"    [step {gstep:>5}/{max_steps}] "
                          f"sup={_f(row['sup_loss_mean'])} "
                          f"dom={_f(row['loss_domain_raw_mean'], 5)} "
                          f"lam={_f(row['lambda_current_mean'], 4)} "
                          f"d_acc={_f(row['d_acc_train_mean'], 3)} "
                          f"d_hold={_f(row['d_acc_holdout_mean'], 3)} "
                          f"d_src%={_f(row['d_pred_source_frac_mean'], 3)} "
                          f"sd_s={_f(row['src_perdim_std_mean'], 4)} "
                          f"sd_t={_f(row['tgt_perdim_std_mean'], 4)} "
                          f"PR_t={_f(row['tgt_participation_ratio_mean'], 1)} "
                          f"mhead={_f(row['morph_head_auroc_mean'], 4)} "
                          f"mprobe={_f(row['morph_proxy_auroc_mean'], 4)} "
                          f"dctl={_f(row.get('morph_minus_control_mean'), 4)} "
                          f"({row['wall_secs']:.0f}s, ~{eta:.0f} min left)", flush=True)

            if (gstep % ckpt_every == 0) or (gstep >= max_steps):
                crow = ckpt_window.row(gstep)
                ckpt_window.reset()
                trainer.save(os.path.join(P.ckpt, f"step_{gstep}.pt"), full_state=False,
                             schedule=dann_schedule_payload(cfg, dann, gstep))
                dst, score_secs, _, _ = scorer.dump(gstep)
                logit_file = os.path.relpath(dst, P.out).replace("\\", "/")
                save_resume(trainer, dann, P, cfg, gstep=gstep, epoch=epoch,
                            batch_in_epoch=seen_in_epoch,
                            rng_epoch_start=rng_epoch_start)
                crow_out = {k: crow.get(k) for k in CHECKPOINT_COLUMNS if k in crow}
                crow_out.update({"step": gstep,
                                 "wall_secs": round(time.time() - t_start, 1),
                                 "score_secs": round(score_secs, 1),
                                 "skipped_cumulative": skipped_cum,
                                 "logit_file": logit_file})
                _append(P.ckpt_csv, CHECKPOINT_COLUMNS, crow_out)
                if verbose:
                    print(f"    [ckpt {gstep:>5}] weights + D + logits written "
                          f"({score_secs:.0f}s to score)", flush=True)

            if watch is not None and watch.should_abort():
                aborted = True
                abort_reason = f"abort conditions fired: {watch.fired()}"
                banner(f"ARMED ABORT AT STEP {gstep}",
                       [abort_reason,
                        "Writing a final checkpoint and a final logit dump so the abort "
                        "step is scorable, then stopping.",
                        "This arm is CONFIRMATORY: the trace has no scientific value "
                        "here and a dead run is wasted T4 time. On the development or "
                        "control arms the mode is log_only and this branch is dead."])
                trainer.save(os.path.join(P.ckpt, f"step_{gstep}_ABORT.pt"),
                             full_state=False,
                             schedule=dann_schedule_payload(cfg, dann, gstep,
                                                            aborted=True))
                with contextlib.suppress(Exception):
                    scorer.dump(gstep)
                stop = True
                break

            if gstep >= max_steps:
                stop = True
                break
        skip_in_epoch = 0
        if stop:
            break

    record.update({
        "aborted": bool(aborted),
        "abort_reason": abort_reason,
        "abort_watch": (None if watch is None else watch.summary()),
        "steps_completed": int(gstep),
        "wall_minutes": (time.time() - t_start) / 60.0,
        "fp16_skipped_cumulative": skipped_cum,
        "peak_gpu_gb_allocated": (torch.cuda.max_memory_allocated() / 2 ** 30
                                  if torch.cuda.is_available() else None),
        "peak_gpu_gb_reserved": (torch.cuda.max_memory_reserved() / 2 ** 30
                                 if torch.cuda.is_available() else None),
    })
    return record


def _f(x, nd=5):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(v) else f"{v:.{nd}f}"


# ===========================================================================
# --grad-check : the nb3 reachability report (cheap, no training)
# ===========================================================================

def grad_check(args, root) -> dict:
    """Backprop the DOMAIN term alone at n_blocks 0 and 3 and report what it reaches.

    Reproduces round 4's method exactly (``coral_round4.coral_reachable_params`` ->
    ``dann_round5.dann_reachable_params``). Writes ``runs/dann_r5_reachability.json``.
    Uses a small random probe batch and the REAL encoder -- no data mount needed.
    """
    from zoobot_encoder import ZoobotEncoder
    torch.manual_seed(3)
    src = torch.rand(int(args.probe_batch), 1, 224, 224)
    tgt = torch.rand(int(args.probe_batch), 1, 224, 224)
    device = args.device or "cpu"

    out = {"purpose": ("round-5 DANN gradient reachability at n_blocks 0 and 3, measured "
                       "the way round 4 measured CORAL's -- backprop the domain term "
                       "ALONE, fp32, no autocast, and read the gradients."),
           "probe_batch": int(args.probe_batch),
           "lambda_used": 1.0,
           "expected_nb3_encoder_elements": 14_835_520,
           "expected_note": (
               "PROJECT.md 7.1a's measured count for n_blocks=3 (stages.1/2/3). Expect a "
               "small shortfall in the exactly-zero direction: dead GELU units in "
               "stages.3 mlp.fc2 that no image in a small probe batch activates. Round 4 "
               "measured 27,524 (0.19%). The reachable TENSOR SET is the structural claim; "
               "the element count is reported with its shortfall attributed."),
           "difference_from_round4": (
               "CORAL's LayerNorm BIAS gradient is analytically zero (covariance is "
               "shift-invariant, the first-moment term is a difference of means), and "
               "round 4 had to attribute a 1e-11 residual to float cancellation. D is an "
               "arbitrary function of the features, so DANN GENUINELY reaches the bias. "
               "Expect ~1,280 head elements here where round 4 expected 640."),
           "arms": {}}

    for nb in (0, 3):
        enc = ZoobotEncoder(encoder_mode="frozen", n_blocks=nb, device=device,
                            verbose=False)
        tcfg = TrainConfig(head_type=HEAD_TYPE, encoder_mode="frozen", n_blocks=nb,
                           backbone_lr=BACKBONE_LR, lr_decay=LR_DECAY, lr=LR,
                           weight_decay=WEIGHT_DECAY, batch_size=int(args.probe_batch),
                           seed=SEED, device=device, amp=False, use_coral=False)
        with contextlib.redirect_stdout(io.StringIO()):
            tr = Trainer(tcfg, encoder=enc)
            dann = install_dann(tr, lambda_horizon=int(args.dann_lambda_horizon),
                                lambda_zero=False, d_lr=LR, weight_decay=WEIGHT_DECAY)
        tr.standardizer.set_stats(torch.zeros(640), torch.ones(640))
        assert_no_batchnorm(tr.head, tr.encoder.model, dann,
                            context=f"nb{nb} head + encoder + discriminator")
        r = dann_reachable_params(tr, src, tgt, lambda_=1.0)
        touched = sorted({".".join(k.split(".")[1:3])
                          for k in r["reachable_by_tensor"] if k.startswith("encoder.")})
        r["encoder_prefixes_touched"] = touched
        r["encoder_shortfall_vs_7_1a"] = (14_835_520 - r["encoder_elements_reachable"]
                                          if nb == 3 else None)
        out["arms"][f"nb{nb}"] = r
        print(f"  n_blocks={nb}: {r['reachable_param_elements']:,} elements with a "
              f"non-zero grad across {r['reachable_tensors']} tensors")
        print(f"    head {r['head_elements_reachable']:,} | "
              f"encoder {r['encoder_elements_reachable']:,} | "
              f"D {r['discriminator_elements_reachable']:,}")
        print(f"    encoder prefixes touched: {touched or '(none)'}")
        if nb == 3:
            print(f"    encoder elements in the graph: "
                  f"{r['encoder_elements_with_grad']:,} (7.1a: 14,835,520); "
                  f"exactly zero on this probe batch: "
                  f"{r['encoder_elements_exactly_zero']:,}")
        del tr, enc

    dst = os.path.abspath(args.grad_check_out
                          or os.path.join(root, "runs", "dann_r5_reachability.json"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[out] {dst}")
    return out


# ===========================================================================
# --dry-run : mock data shaped like the real splits, 3 steps, exit
# ===========================================================================

def dry_run(args):
    """Construct model + data + optimizer on MOCK data and run 3 steps through the
    REAL ``train_loop``, then exercise the resume round-trip and exit.

    The mock uses ``test_lens_dataset.build_mock`` / ``build_splits`` -- the real
    ``splits_v6_label_free_da`` schema, the real ``LensDataset``, the real balanced
    samplers, the real Zoobot encoder, the real ``DannLoss`` and the real loop. What
    it deliberately does NOT exercise is the phase-1-head gate (no mock head can have
    ||.||_2 = 28.4756) or the 4.4 GB mount; ``--emit-config`` and the path-resolution
    check cover those.
    """
    from live_encode import LiveEncoder
    from test_lens_dataset import build_mock, build_splits
    from zoobot_encoder import ZoobotEncoder

    dev = args.device or "cpu"
    root = tempfile.mkdtemp(prefix="dann_r5_dry_")
    out_dir = args.out_dir or os.path.join(root, "out")
    print("=" * 92)
    print("DRY RUN -- mock data on the real schema, the real loop, 3 steps, then exit")
    print(f"  device   : {dev}   (nothing is trained on real data here)")
    print(f"  mock root: {root}")
    print(f"  out dir  : {out_dir}")
    print("=" * 92, flush=True)
    ok = False
    try:
        build_mock(root)
        splits = os.path.join(root, "splits.json")
        with open(splits, "w") as fh:
            json.dump(build_splits(), fh)

        src = LensDataset(splits, root, role="source_train", aug="weak", n_views=1)
        tgt = LensDataset(splits, root, role="target", fold=0, aug="weak", n_views=1)
        assert np.all(tgt.labels == UNLABELED), "mock target labels must be UNLABELED(-1)"
        bs, nsteps, nb = 4, 3, int(args.dry_run_n_blocks)

        # ``test_lens_dataset.build_mock`` writes CONSTANT-VALUED images, which the
        # per-image stretch maps to identical features -- every per-dimension std and
        # every eigenvalue comes back exactly 0, so the representation monitors would
        # be exercised only in their degenerate branch. Replace the pixels (in memory,
        # deterministically) so the scale and rank columns carry real numbers.
        rng = np.random.default_rng(0)
        for ds in (src, tgt):
            ds.images = rng.standard_normal(ds.images.shape).astype(ds.images.dtype)

        # The mock target pool holds 5 items, so the real 10% would round to a holdout
        # of ZERO and d_acc_holdout would never be exercised. The fraction is raised
        # HERE ONLY, in the dry run, to make the monitor path run.
        dry_holdout_frac = 0.34
        holdout = build_target_holdout(tgt, frac=dry_holdout_frac, seed=SEED)
        print(f"[dry] target pool {holdout['n_pool']} -> train {holdout['n_train']}, "
              f"holdout {holdout['n_holdout']}; holdout positions "
              f"{list(holdout['holdout'])}", flush=True)
        # determinism of the slice, checked rather than asserted in prose
        again = build_target_holdout(tgt, frac=dry_holdout_frac, seed=SEED)
        holdout_deterministic = np.array_equal(holdout["holdout"], again["holdout"])
        overlap = np.intersect1d(holdout["holdout"],
                                 np.concatenate([holdout["train_pos"],
                                                 holdout["train_neg"]]))
        print(f"[dry] holdout deterministic at the same seed: {holdout_deterministic}; "
              f"overlap with D's training pool: {overlap.size} (must be 0)", flush=True)

        sl = make_balanced_loader(src, bs, use_latent=False, num_batches=nsteps,
                                  pos_fraction=0.5, seed=SEED)
        ts = ClassBalancedBatchSampler(holdout["train_pos"], holdout["train_neg"], bs,
                                       num_batches=nsteps, pos_fraction=0.5,
                                       seed=SEED + 1000)
        tl = DataLoader(tgt, batch_sampler=ts, num_workers=0)

        gc_on = (nb > 0)
        enc = ZoobotEncoder(encoder_mode="frozen", n_blocks=nb, device=dev, verbose=False,
                            grad_checkpointing=gc_on)
        tcfg = TrainConfig(splits_path=splits, data_root=root, fold=0, head_type=HEAD_TYPE,
                           encoder_mode="frozen", lr=LR, weight_decay=WEIGHT_DECAY,
                           batch_size=bs, pos_fraction=0.5, use_coral=False,
                           total_steps=nsteps, epochs=1, steps_per_epoch=nsteps,
                           max_steps=nsteps, seed=SEED, device=dev, num_workers=0,
                           amp=False, n_blocks=nb, backbone_lr=BACKBONE_LR,
                           lr_decay=LR_DECAY, init_head_from=None,
                           grad_checkpointing=gc_on)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = Trainer(tcfg, encoder=enc)
            dann = install_dann(trainer,
                                lambda_horizon=int(args.dann_lambda_horizon),
                                lambda_zero=bool(args.dann_lambda_zero),
                                holdout_frac=dry_holdout_frac,
                                spectrum_every=1,
                                dropout=float(args.dann_d_dropout),
                                n_dropout_layers=int(args.dann_d_dropout_layers),
                                d_lr=LR, weight_decay=WEIGHT_DECAY)
        trainer.standardizer.set_stats(torch.zeros(640), torch.ones(640))
        bn = assert_no_batchnorm(trainer.head, trainer.encoder.model, dann,
                                 context="dry-run model")
        n_groups = len(trainer.opt.param_groups)
        group_names = [g.get("name", "?") for g in trainer.opt.param_groups]
        n_d = sum(p.numel() for p in dann.discriminator.parameters())
        print(f"[dry] optimizer: {type(trainer.opt).__name__}, {n_groups} LR group(s) "
              f"{group_names}; D contributes {n_d:,} params at lr="
              f"{trainer.opt.param_groups[-1]['lr']:g} wd="
              f"{trainer.opt.param_groups[-1]['weight_decay']:g}", flush=True)
        import torch.nn as _nn
        n_drop = sum(1 for m in dann.discriminator.modules()
                     if isinstance(m, _nn.Dropout))
        print(f"[dry] BatchNorm modules anywhere in the model: {bn['n_batchnorm']} "
              f"({bn['n_modules_scanned']} scanned); Dropout layers in D: {n_drop} "
              f"at p={args.dann_d_dropout}; D.training={dann.discriminator.training}",
              flush=True)

        P = Paths(out_dir)
        live = LiveEncoder(splits, root, 0, batch_size=bs, num_workers=0, device=dev,
                           verbose=False)
        scorer = Scorer(trainer, live, 0, P.logits, arm="dry", fp16=False)
        init_dst, init_secs, ipos, ineg = scorer.dump(0, name="true_init")
        print(f"[dry] step-0 true_init logits: {os.path.basename(init_dst)} "
              f"({ipos.size} pos / {ineg.size} neg, {init_secs:.1f}s)", flush=True)

        holdout_images = materialise_holdout_images(tgt, holdout["holdout"])

        # The morphology proxy needs the REAL ML-FP / P1-reserve shards, which the mock
        # tree does not contain. Substituted here with a pool built from the mock's own
        # target images and an arbitrary deterministic split, purely so the WIRING is
        # exercised end to end -- the NUMBER it produces in a dry run is meaningless and
        # is not checked for anything but finiteness. The real instrument is measured by
        # calibrate_morph_r5.py, which runs against the real mount.
        morph = _mock_morph_proxy(tgt, seed=SEED)
        print(f"[dry] morphology proxy: MOCK pool of {morph.images.shape[0]} images "
              f"(the real ML-FP / P1-reserve shards are not in the mock tree); wiring "
              f"only, the value is meaningless here", flush=True)

        # A synthetic control curve, so condition 3's plumbing runs end to end in the
        # dry run: one row at every step the loop will measure morphology at, holding a
        # value the run cannot plausibly fall 0.10 below. It exists to prove the file is
        # read and step-matched, not to test the threshold.
        ctrl_csv = os.path.join(root, "mock_control_steps.csv")
        with open(ctrl_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["step", ABORT_COND3_COLUMN,
                                               "src_participation_ratio",
                                               "tgt_participation_ratio"])
            w.writeheader()
            for st_ in range(0, nsteps + 1):
                w.writerow({"step": st_, ABORT_COND3_COLUMN: 0.5,
                            "src_participation_ratio": 5.0,
                            "tgt_participation_ratio": 10.0})
        control = ControlCurve(ctrl_csv)
        # The dry run PINS a margin so condition 3a's arithmetic is exercised end to
        # end. The real arms 1 and 2 run with it UNSET, which is a different code path
        # (3a refuses to evaluate) and is covered by test_da_dann.py's T13.
        watch = DannAbortWatch(mode=ABORT_MODE_LOG_ONLY, control=control,
                               cond3_margin=(0.05 if args.abort_cond3_margin is None
                                             else float(args.abort_cond3_margin)),
                               cond4_src_margin=1.0, cond4_tgt_margin=2.0,
                               cond4_sustain=int(args.abort4_sustain), verbose=False)
        print(f"[dry] control curve: {control.summary()['n_steps']} rows, "
              f"missing steps for this run: "
              f"{len(control.covers(range(0, nsteps + 1)))}; abort mode "
              f"{watch.mode}", flush=True)

        A = {"paired": PairedDomainLoader(sl, tl), "src_loader": sl, "tgt_loader": tl,
             "steps": nsteps, "epochs_planned": 1, "trainer": trainer, "dann": dann,
             "holdout_images": holdout_images, "morph": morph, "watch": watch}
        cfg = {"max_steps": nsteps, "log_every": 1, "ckpt_every": 1, "holdout_every": 1,
               "morph_every": 1, "arm": "dry", "fold": 0}
        record = {"dry_run": True}
        train_loop(A=A, P=P, cfg=cfg, record=record, scorer=scorer)

        # ---- lambda: horizon-invariance and resume-equivalence, on the real object ----
        lam_here = [dann.lambda_at(s) for s in (0, 1, 2, 3)]
        fresh = type(dann)(trainer.head.feat_dim,
                           lambda_horizon=int(args.dann_lambda_horizon),
                           lambda_zero=bool(args.dann_lambda_zero))
        lam_fresh = [fresh.lambda_at(s) for s in (0, 1, 2, 3)]
        lam_ok = all(a == b for a, b in zip(lam_here, lam_fresh))
        print(f"[dry] lambda reconstructible from step alone (fresh object == trained "
              f"object): {lam_ok}  {['%.6f' % x for x in lam_here]}", flush=True)

        # ---- resume round-trip: D and the moments must come back, not just weights ----
        save_resume(trainer, dann, P, cfg, gstep=nsteps, epoch=1, batch_in_epoch=nsteps,
                    rng_epoch_start=_rng_state())
        moments_before = _moment_fingerprint(trainer)
        with contextlib.redirect_stdout(io.StringIO()):
            enc2 = ZoobotEncoder(encoder_mode="frozen", n_blocks=nb, device=dev,
                                 verbose=False, grad_checkpointing=gc_on)
            t2 = Trainer(tcfg, encoder=enc2)
            d2 = install_dann(t2, lambda_horizon=int(args.dann_lambda_horizon),
                              lambda_zero=bool(args.dann_lambda_zero),
                              holdout_frac=dry_holdout_frac,
                              spectrum_every=1, d_lr=LR, weight_decay=WEIGHT_DECAY)
        d_before = {k: v.clone() for k, v in dann.discriminator.state_dict().items()}
        d_fresh_same = all(torch.equal(d_before[k].cpu(), v.cpu())
                           for k, v in d2.discriminator.state_dict().items())
        sch = load_resume(t2, d2, P)
        moments_after = _moment_fingerprint(t2)
        same_w = all(torch.equal(a.cpu(), b.cpu()) for a, b in
                     zip(trainer.head.state_dict().values(), t2.head.state_dict().values()))
        same_d = all(torch.equal(d_before[k].cpu(), v.cpu())
                     for k, v in d2.discriminator.state_dict().items())
        enc_a, enc_b = trainer.encoder_trainable_state(), t2.encoder_trainable_state()
        same_enc = (set(enc_a) == set(enc_b)
                    and all(torch.equal(enc_a[k].cpu(), enc_b[k].cpu()) for k in enc_a))
        ok_m = (moments_before is not None and moments_after is not None
                and abs(moments_before - moments_after)
                <= 1e-12 * max(1.0, abs(moments_before)))
        print("\n" + "-" * 92)
        print(f"[dry] resume round-trip: step={sch.get('global_step')} "
              f"epoch={sch.get('epoch')} batch_in_epoch={sch.get('batch_in_epoch')}")
        print(f"[dry]   head weights identical after reload      : {same_w}")
        print(f"[dry]   {len(enc_a)} trainable encoder tensors identical : {same_enc}")
        print(f"[dry]   DISCRIMINATOR identical after reload     : {same_d}  "
              f"(a FRESH D would differ: {not d_fresh_same})")
        print(f"[dry]   AdamW moment fingerprint restored        : {ok_m} "
              f"({moments_before} -> {moments_after})")
        print(f"[dry]   RNG state present in the payload         : "
              f"{bool(sch.get('rng_epoch_start')) and bool(sch.get('rng_now'))}")
        print("-" * 92, flush=True)

        wrote = sorted(os.path.relpath(p, P.out).replace("\\", "/")
                       for p in glob.glob(os.path.join(P.out, "**", "*"), recursive=True)
                       if os.path.isfile(p))
        print("[dry] files written:")
        for f in wrote:
            print(f"        {f}")
        with open(P.steps_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
        cols_present = set(rows[0].keys()) if rows else set()
        need = {"d_acc_train", "d_acc_holdout", "d_pred_source_frac", "lambda_current",
                "loss_domain_raw", "ratio_lambda_domain_over_sup", "src_perdim_std",
                "tgt_perdim_std", "src_participation_ratio", "tgt_dims_at_90",
                "tgt_dims_at_99", "morph_head_auroc", "morph_head_auroc_pool",
                "morph_proxy_auroc", "morph_proxy_auroc_encoder",
                "morph_minus_control", "control_morph_proxy_auroc",
                "abort1_d_overpowering_streak", "abort2_degenerate_collapse_fired",
                "abort3_morph_vs_control_streak", "abort3b_morph_sign_flip_fired",
                "abort4src_rank_collapse_streak", "abort4tgt_rank_collapse_streak",
                "src_pr_minus_control", "tgt_pr_minus_control"}
        cols_ok = need.issubset(cols_present)
        def _finite_every_row(col):
            try:
                return all(math.isfinite(float(r.get(col, "nan"))) for r in rows)
            except (TypeError, ValueError):
                return False

        holdout_logged = _finite_every_row("d_acc_holdout")
        morph_logged = (_finite_every_row("morph_proxy_auroc")
                        and _finite_every_row("morph_head_auroc")
                        and _finite_every_row("morph_head_auroc_pool"))
        ctrl_logged = _finite_every_row("morph_minus_control")
        d_mode_ok = bool(dann.discriminator.training)
        spectrum_logged = all(_finite_every_row(c) for c in
                              ("src_participation_ratio", "tgt_participation_ratio",
                               "src_dims_at_90", "tgt_dims_at_99"))
        std_logged = all(_finite_every_row(c) for c in
                         ("src_perdim_std", "tgt_perdim_std"))
        print(f"[dry] steps.csv: {len(rows)} rows, every brief-required column present: "
              f"{cols_ok}; d_acc_holdout finite every step: {holdout_logged}; "
              f"morph_proxy_auroc finite every step: {morph_logged}; "
              f"rank monitors finite: {spectrum_logged}; per-dim std finite: {std_logged}")
        print(f"[dry] D restored to train mode after the monitors ran: {d_mode_ok}; "
              f"morph_minus_control finite every step: {ctrl_logged}; abort watch "
              f"fired: {watch.fired() or 'none'}", flush=True)

        ok = bool(same_w and same_enc and same_d and (not d_fresh_same) and ok_m
                  and lam_ok and cols_ok and holdout_logged and spectrum_logged
                  and std_logged and morph_logged and ctrl_logged and d_mode_ok
                  and n_drop == 2 and not record["aborted"]
                  and holdout_deterministic and overlap.size == 0
                  and record["steps_completed"] == nsteps
                  and bn["n_batchnorm"] == 0
                  and os.path.isfile(P.ckpt_csv))
        print("\n" + ("[dry] DRY RUN PASSED -- model, data, optimizer, GRL, D, holdout "
                      "monitor, loop, logit persistence and resume all exercised."
                      if ok else "[dry] DRY RUN FAILED -- see above."), flush=True)
        return 0 if ok else 1
    finally:
        if not args.keep_dry_run:
            shutil.rmtree(root, ignore_errors=True)


def _mock_morph_proxy(tgt_ds, *, seed: int):
    """A MorphProxy over mock images, for the dry run only.

    The real instrument needs ``data/ml_fp_cutouts`` and the partition-1 reserve, and
    ``test_lens_dataset.build_mock`` contains neither. This substitutes the mock
    target images with an arbitrary deterministic 1/0 split so that the wiring -- the
    encode, the ridge solve, the AUROC, the column plumbing -- runs end to end. The
    VALUE is meaningless and the dry run checks only that it is finite. Never used
    outside ``--dry-run``.
    """
    n = len(tgt_ds)
    imgs = torch.stack([tgt_ds[i]["image"] for i in range(n)], dim=0)
    y = np.array([1 if i % 2 == 0 else 0 for i in range(n)], dtype=np.int64)
    idx = np.arange(n)
    pool = {"images": imgs, "y": y,
            "fit_idx": idx, "eval_idx": idx,      # mock only: no held-out half exists
            "info": {"instrument": "MOCK -- dry run only, value is meaningless",
                     "seed": int(seed), "n_total": int(n)}}
    return MorphProxy(pool, alpha=MORPH_RIDGE_ALPHA)


def _moment_fingerprint(trainer):
    """Sum of AdamW exp_avg_sq entries -- zero only if the moments were never restored."""
    tot, seen = 0.0, 0
    for st in trainer.opt.state.values():
        v = st.get("exp_avg_sq")
        if v is not None:
            tot += float(v.double().sum())
            seen += 1
    return tot if seen else None


# ===========================================================================
# main
# ===========================================================================

def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="dann_nb3", choices=sorted(ARMS))
    p.add_argument("--fold", type=int, default=0,
                   help="0 is the DEVELOPMENT fold (7.11/7.14); no number from it enters "
                        "a headline")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--input-root", default=None,
                   help="where to search for the dataset (default: /kaggle/input, then "
                        "the repo root)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--init-head-from", default=None,
                   help=f"phase-1 head; default {_path_tail(DEFAULT_INIT_HEAD, 4)} "
                        f"(see disagreement 3 in the module docstring)")
    p.add_argument("--max-steps", type=int, default=DEV_STEPS)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)

    # ---- the DANN flags the brief names ----
    p.add_argument("--dann-lambda-horizon", type=int, default=LAMBDA_HORIZON,
                   help="DECLARED ABSOLUTE constant for the lambda schedule. NOT the run "
                        "length. Must be identical in development and confirmatory runs.")
    p.add_argument("--dann-lambda-zero", action="store_true", default=False,
                   help="pin lambda to 0.0 at every step; the GRL and D still run "
                        "(the control arm, exercising the identical code path)")
    p.add_argument("--dann-d-holdout-frac", type=float, default=D_HOLDOUT_FRAC,
                   help="fraction of the fold's target pool held out of D's loss and "
                        "used for the d_acc_holdout memorisation monitor")
    p.add_argument("--dann-d-dropout", type=float, default=D_DROPOUT,
                   help="dropout inside D (Belfiore et al. 2024; inherited, not tuned)")
    p.add_argument("--dann-d-dropout-layers", type=int, default=D_DROPOUT_LAYERS,
                   choices=(0, 1, 2),
                   help="2 = after each hidden ReLU (default); 1 = only between the two "
                        "hidden layers, the brief's literal reading; 0 = no dropout")
    p.add_argument("--morph-every", type=int, default=MORPH_EVERY,
                   help="measure the morphology proxy every N steps (default 25). MUST "
                        "MATCH between the two arms: condition 3 step-matches against "
                        "the control and refuses to interpolate")
    p.add_argument("--morph-n-fit-per-class", type=int, default=MORPH_N_FIT_PER_CLASS)
    p.add_argument("--morph-n-eval-per-class", type=int, default=MORPH_N_EVAL_PER_CLASS)

    # ---- brief v3 3 + 4.2: the pre-registered abort conditions ----
    p.add_argument("--abort-mode", choices=(ABORT_MODE_LOG_ONLY, ABORT_MODE_ARMED),
                   default=None,
                   help="log_only = fire is banner-printed and recorded, run continues; "
                        "armed = run stops after writing a final scorable checkpoint. "
                        "DEFAULT depends on the arm: log_only on the development fold 0 "
                        "and on the lambda=0 control, armed on confirmatory folds 1-4. "
                        "Whatever is used is RECORDED in run_config.json as abort_mode.")
    p.add_argument("--control-curve", default=None, metavar="STEPS_CSV",
                   help="the lambda=0 arm's steps.csv, for abort condition 3. Read by "
                        "ABSOLUTE step; a missing step refuses the comparison rather "
                        "than interpolating. Omit on the control arm itself.")
    p.add_argument("--allow-control-gaps", action="store_true", default=False,
                   help="start even though the control curve does not cover every step "
                        "this run will measure morphology at (recorded)")
    p.add_argument("--abort-cond3-margin", type=float, default=ABORT_MARGIN_COND3,
                   help="condition 3a's margin below the control on the "
                        f"{ABORT_COND3_COLUMN} column. UNSET BY DEFAULT and deliberately "
                        "not defaulted: brief v3's 0.10 was declared against the FROZEN "
                        "PROBE (headroom 0.4202) and is scale-wrong for the head column "
                        "(headroom ~0.078). Declared off arm 1 and hashed before arm 2. "
                        "While unset, 3a REFUSES to evaluate and an --abort-mode armed "
                        "run REFUSES TO START; 3b (sign flip) needs no margin.")
    p.add_argument("--abort3-median-window", type=int, default=1, metavar="K",
                   help="condition 3a thresholds the trailing median of the last K "
                        "PAIRED differences instead of the instantaneous one. Default 1 "
                        "= no smoothing. K MUST BE DECLARED BEFORE M -- a window chosen "
                        "after a candidate M is on the table has been selected to make "
                        "that M work (PROJECT.md 7.17).")
    p.add_argument("--morph-stats", default=None, metavar="STEPS_CSV",
                   help="compute J, the trajectory IQR and the pre-registered "
                        "M-declaration verdict from a finished arm's steps.csv, print "
                        "them and exit. Trains nothing. This is what arm 1 is read with.")
    p.add_argument("--abort-cond4-src-margin", type=float, default=COND4_SRC_MARGIN,
                   help="condition 4's margin below the control on "
                        "src_participation_ratio. UNSET by default; declared off arm 1 "
                        f"and hashed. Admissible band [0.5175, {COND4_SRC_CEILING:.4f}].")
    p.add_argument("--abort-cond4-tgt-margin", type=float, default=COND4_TGT_MARGIN,
                   help="condition 4's margin below the control on "
                        "tgt_participation_ratio. UNSET by default; declared off arm 1 "
                        f"and hashed. Admissible band [1.3263, {COND4_TGT_CEILING:.4f}].")
    p.add_argument("--abort3b-sustain", type=int, default=COND3B_SUSTAIN,
                   help="condition 3b's sustain in steps. Default 0 = fire on a SINGLE "
                        "reading below the floor -- correct when 3b was a sign flip at "
                        "0.5, and a DECLARED CHOICE now that it is a floor at "
                        f"{COND3_SIGN_FLIP}. On arm 2 it fired at step 451 and held for "
                        "7,550 of 8,000 steps, so an armed fold aborts at 451 unless "
                        "this is raised deliberately (PROJECT.md 7.17).")
    p.add_argument("--abort4-sustain", type=int, default=ABORT_SUSTAIN_COND4,
                   help="condition 4's sustain in steps (default 500, a DECLARED "
                        "addition; the design doc gives none -- 0 = fire instantly)")
    p.add_argument("--holdout-every", type=int, default=1,
                   help="score the held-out slice every N steps (1 = every step, as the "
                        "brief asks; raise it to trade monitor resolution for throughput)")
    p.add_argument("--spectrum-every", type=int, default=SPECTRUM_EVERY,
                   help="recompute participation_ratio / dims_at_90 / dims_at_99 every N "
                        "steps (~10 ms per domain per step)")

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--eval-fp16", action="store_true", default=False,
                   help="scoring precision; leave OFF -- the reference logits are fp32")
    p.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                   action="store_false", default=True,
                   help="MANDATORY at nb3 on a T4 (7.1a); off only for a CPU smoke test")
    p.add_argument("--target-fpr", type=float, default=1e-3)

    p.add_argument("--emit-config", default=None,
                   help="write run_config.json from the SAME construction code and exit "
                        "without training, so a second arm validates at minute 2")
    p.add_argument("--reference-config", default=None,
                   help="the other arm's emitted config; every pinned key must match")
    p.add_argument("--no-resume", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--dry-run-n-blocks", type=int, default=3)
    p.add_argument("--keep-dry-run", action="store_true", default=False)
    p.add_argument("--grad-check", action="store_true", default=False,
                   help="reachability probe at nb0 and nb3; writes "
                        "runs/dann_r5_reachability.json and exits")
    p.add_argument("--grad-check-out", default=None)
    p.add_argument("--probe-batch", type=int, default=4)
    return p


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    args = build_argparser().parse_args(argv)

    if args.morph_stats:
        # k comes from --abort3-median-window so the standalone path can be run at a
        # CANDIDATE k before k is declared. J is then computed on the series 3a would
        # actually threshold at that k (PROJECT.md 7.17).
        st = morph_trajectory_stats(os.path.abspath(args.morph_stats),
                                    k=int(args.abort3_median_window),
                                    morph_every=int(args.morph_every))
        print_morph_trajectory_stats(st)
        dst = os.path.splitext(os.path.abspath(args.morph_stats))[0] + "_morph_stats.json"
        with open(dst, "w") as fh:
            json.dump(st, fh, indent=2, default=str)
        print(f"[out] {dst}")
        return 0
    if args.dry_run:
        return dry_run(args)
    if args.grad_check:
        grad_check(args, root)
        return 0

    paths = resolve_paths(args, root)

    # ---- --emit-config: the SAME construction code, run to completion, NO TRAINING ----
    # It BUILDS THE ARM rather than short-cutting to build_config, exactly as round 4's
    # runner does. A reference emitted without touching the data would carry null for
    # init_head_l2 and the holdout size, and every one of those pinned keys would then
    # compare null-against-real on the second rung -- i.e. the guard would be reporting
    # mismatches on its own shortcut instead of on a real disagreement.
    if args.emit_config:
        A = build_arm(args.arm, args, paths)
        dst = os.path.abspath(args.emit_config)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w") as fh:
            json.dump(A["cfg_json"], fh, indent=2, default=str)
        print(f"[emit-config] declared reference for {args.arm} fold {args.fold} -> {dst}")
        print(f"[emit-config] nothing was trained (arm fully constructed: "
              f"{A['steps']} steps/epoch, holdout {A['holdout']['n_holdout']}, "
              f"||head||_2 {A['cfg_json']['init_head_l2']}). Pass it to the other arm as "
              f"--reference-config.")
        # If a reference was ALSO given, run the pin check here. Without this, a
        # preflight that passes both flags emits a config and exits without comparing
        # anything -- so the second arm's minute-2 validation would silently skip the one
        # check that makes the two arms a controlled comparison, and the mismatch would
        # not surface until the real run four hours later.
        if args.reference_config:
            check_preconditions(A["cfg_json"], os.path.abspath(args.reference_config))
        return 0

    out_dir = args.out_dir or os.path.join(
        "/kaggle/working" if os.path.isdir("/kaggle/working") else root,
        "runs", f'{ARMS[args.arm]["dir"]}_fold{int(args.fold)}')
    P = Paths(out_dir)

    print("=" * 92)
    print(f"ROUND-5 DANN -- {args.arm}, fold {args.fold}, {args.max_steps} steps, "
          f"lambda_horizon={args.dann_lambda_horizon}"
          f"{' (lambda PINNED TO 0 -- control arm)' if args.dann_lambda_zero or ARMS[args.arm]['lambda_zero'] else ''}")
    print("=" * 92)
    for k in ("data_root", "splits", "cache_dir", "init_head_from"):
        print(f"  {k:<16}: {paths.get(k)}")
    print(f"  {'out_dir':<16}: {P.out}")
    print(f"  {'torch':<16}: {torch.__version__}  cuda={torch.cuda.is_available()}")
    print("=" * 92, flush=True)

    existing = os.path.isfile(P.steps_csv) or os.path.isfile(P.resume_pt)
    if existing and args.no_resume:
        raise SystemExit(
            f"[dann-r5] ABORT: {P.out} already holds a run and --no-resume was passed. "
            f"Move it aside or point --out-dir elsewhere -- appending would interleave "
            f"two runs' rows into one series.")

    A = build_arm(args.arm, args, paths)
    cfg = A["cfg_json"]
    cfg["out_dir"] = P.out
    cfg["holdout_every"] = int(args.holdout_every)

    print("\n" + "-" * 92)
    for k, v in cfg.items():
        if k in ("dann_d_holdout_indices", "health_anchors", "no_batchnorm_check"):
            continue
        s = str(v)
        print(f"  {k:<34}: {s[:120]}{' ...' if len(s) > 120 else ''}")
    print("-" * 92 + "\n", flush=True)

    if args.reference_config:
        rows = check_preconditions(cfg, os.path.abspath(args.reference_config))
        cfg["precondition_check"] = {"reference": os.path.abspath(args.reference_config),
                                     "rows": rows}
    else:
        print("[pin] no --reference-config given -- this run DECLARES the reference. "
              "Emit it with --emit-config and pass it to the other arm.\n")

    record = dict(cfg)
    record.update({
        "purpose": ("PROJECT.md 7.14 fold-0 DEVELOPMENT run for DANN: fix DANN's own "
                    "budget from its own fold-0 curve, before any confirmatory fold "
                    "exists. Fold 0 is development; no number here enters a headline."),
        "continuation": ("NONE. This is a FRESH trajectory from step 0. 7.11 documented "
                         "three silent restarts (threshold buffers, schedule position, "
                         "AdamW moments) when continuing from a weights-only checkpoint; "
                         "this run does not continue from anything, and its own resume "
                         "payload carries D as well."),
        "metric_hygiene": ("the ONLY test-set metric computed is the step-0 known-null "
                           "gate; every training segment runs inside test_label_lockout(); "
                           "checkpoints are SCORED (logits persisted) but never MEASURED"),
        "paths": paths,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "script_sha256_16": {f: _sha256(os.path.join(_HERE, f))
                             for f in SCRIPT_HASH_FILES
                             if os.path.isfile(os.path.join(_HERE, f))},
        "steps_completed": 0, "fp16_skipped_cumulative": 0.0, "resume_events": [],
    })

    from live_encode import LiveEncoder
    live = LiveEncoder(paths["splits"], paths["data_root"], int(args.fold),
                       batch_size=A["batch_size"], num_workers=0, device=A["device"],
                       verbose=True)
    scorer = Scorer(A["trainer"], live, int(args.fold), P.logits,
                    arm=cfg["logit_arm"], fp16=bool(args.eval_fp16))

    resume_sched = None
    if existing and not args.no_resume and os.path.isfile(P.resume_pt):
        resume_sched = load_resume(A["trainer"], A["dann"], P)
        record["resume_events"].append(
            {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "resumed_at_step": int((resume_sched or {}).get("global_step", 0)),
             "saved_utc": (resume_sched or {}).get("saved_utc")})
        ids_path = os.path.join(P.logits, "ids.npz")
        if os.path.isfile(ids_path):
            z = np.load(ids_path, allow_pickle=True)
            scorer.ids = (z["pos_ids"], z["neg_ids"])
        banner("RESUMED",
               [f"restored from {P.resume_pt} at step "
                f"{(resume_sched or {}).get('global_step')}",
                "weights, AdamW moments, GradScaler state, THE DISCRIMINATOR and RNG "
                "restored.",
                "lambda needs no rescue -- it is a pure function of global_step and the "
                "declared horizon, and the horizon is checked against the checkpoint's.",
                "Sample ORDER is exact; augmentation offsets after a resume are not "
                "guaranteed to equal an uninterrupted run's. Recorded in run_config.json."],
               "=")
    else:
        step0_gate(scorer, record, int(args.fold),
                   tpr_tol=cfg["step0_tpr_tolerance"],
                   auroc_tol=cfg["step0_auroc_tolerance"])

    with open(P.run_json, "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    print(f"\n[train] {args.arm} fold {args.fold}: {cfg['max_steps']} steps, "
          f"checkpoint+logits every {cfg['ckpt_every']}, trace every {cfg['log_every']}, "
          f"holdout monitor every {cfg['holdout_every']}\n", flush=True)
    if A["device"] == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    record = train_loop(A=A, P=P, cfg=cfg, record=record, scorer=scorer,
                        resume_sched=resume_sched)

    have, want = assert_checkpoint_retention(P.ckpt, cfg["max_steps"], cfg["ckpt_every"])
    record["n_checkpoints_expected"] = len(want)
    record["n_checkpoints_saved"] = len(have)

    # ---- the arm-completion summary: J, the trajectory IQR, and the PRE-REGISTERED
    # verdict on whether M is declarable. Emitted on EVERY arm, not just arm 1 -- the
    # criterion names arm 1, but a development arm's own jitter is worth having beside
    # it, and computing it only where it is "supposed" to matter is how a number ends
    # up unavailable at the moment it is wanted. ----
    try:
        record["morph_trajectory_summary"] = morph_trajectory_stats(
            P.steps_csv, k=int(args.abort3_median_window),
            morph_every=int(args.morph_every))
        print_morph_trajectory_stats(record["morph_trajectory_summary"])
    except SystemExit as e:
        record["morph_trajectory_summary"] = {"unavailable": str(e)}
        print(f"[morph-stats] not computed: {e}", flush=True)
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(P.run_json, "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    # run_config.json is the same record; written under both names so the packaging
    # and reporting tools that look for either find it.
    with open(os.path.join(P.out, "run_config.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    peak = record.get("peak_gpu_gb_allocated")
    print("\n" + "=" * 92)
    print(f"[done] steps {record['steps_completed']}/{cfg['max_steps']}  "
          f"{record['wall_minutes']:.1f} min  checkpoints "
          f"{record['n_checkpoints_saved']}/{record['n_checkpoints_expected']}  "
          f"peak {peak if peak is None else round(peak, 2)} GB allocated")
    print(f"[out] {P.out}")
    print("[next] download the run directory, drop it into runs/ locally, and read the "
          "budget off the persisted logits -- CPU work, no GPU session time. The budget "
          "must be DECLARED AND HASHED before any confirmatory fold runs (7.14).")
    print("=" * 92, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
