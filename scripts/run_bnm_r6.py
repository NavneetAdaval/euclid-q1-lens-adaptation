"""
run_bnm_r6.py
=============
ROUND 6 -- **BNM x n_blocks=3**, image-space, per-fold runner. CORAL x nb3's pattern on
the **coral-hook branch**: a single auxiliary term, no self-training machinery, no
discriminator, no ramp.

    python -u scripts/run_bnm_r6.py --fold 0 --gamma 4.0 --max-steps 8000
    python -u scripts/run_bnm_r6.py --fold 0 --measure-step0        # exits before training
    python -u scripts/run_bnm_r6.py --fold 1 --gamma 4.0 --max-steps <budget>

THREE THINGS THIS RUNNER DOES THAT ITS PREDECESSORS DID NOT
---------------------------------------------------------------------------
**1. It enforces "no horizon" as a RUNTIME INVARIANT, not a token scan.** The arm's
defining property -- that gamma never varies with the step -- is what makes BNM the
minimal-signal control rather than a fourth ranking entry. A static scan for
``set_step / lambda_at / total_steps / horizon / step / warmup / ramp / schedule`` is
defeated by arithmetic:

    w = min(1.0, i / 2000.0)          # a ramp, and none of those tokens appear
    loss = ce + w * gamma * bnm       # written in good faith by someone tidying the loop

So :class:`bnm_round6.GammaInvariant` records the gamma **actually multiplied into the
loss** at every step -- recovered at the assembly site as ``da_loss / raw_bnm``, which
sees any extra factor introduced anywhere between the hook's return and the total loss --
asserts it against the step-0 value, and **fails loudly on the first divergence**. The
full sequence is written to ``run.json`` so the invariant is verifiable after the fact,
not merely enforced during. ``test_bnm_round6.py`` injects a ramp and asserts it fires:
a guard never seen to fail is not known to work.

The token scan stays as a SECOND layer in ``audit_lambda0_r6.py --runner run_bnm_r6.py``.
It fails earlier and reads better in the audit output; the invariant is the one doing the
work.

**2. ``--measure-step0`` is a mode, not a side effect.** Stage 2's gamma grid comes from
the BNM term's gradient-norm contribution relative to source CE at step 0, and that
measurement needs the runner -- which risks it being taken from a run that has already
started training. This mode does one forward/backward, writes
``runs/bnm_r6_step0_gradnorms.json``, and **exits before any optimiser step**. It is
metric-blind by construction: ``evaluate``/``metrics`` are never imported on that path and
no test-set tensor is touched, which ``test_bnm_round6.py`` asserts rather than assumes.

**3. It keeps DANN's target holdout, deliberately, and the reason is not the one you would
guess.** BNM has no discriminator and needs no held-out slice. It is kept because
**the lambda=0 arm (7.22, folds 1-4 at budget 4000) IS this runner's gamma=0 known-null**,
and reusing it requires the target sampler to match field-for-field -- brief 5's rule. The
78 held-out images are a 10% reduction in target data that applies identically to the arm
BNM is compared against. It also turns out to be a genuine diagnostic here: 7.12 measured
the 782 target images being memorised, BNM is the objective most directly aligned with that
pathology, and a target slice the loss never saw is the only way to see confidence gains
generalise rather than memorise.

WHAT IS DELIBERATELY ABSENT: no ramp, no EMA, no thresholds, no mu, no LR scheduler, no
gradient clipping, no second optimiser, no discriminator. ``total_steps`` reaches nothing
on this path (the coral branch contains zero occurrences, audited).
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
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from bnm_round6 import (BNMHook, GammaInvariant, GammaHorizonDetected,   # noqa: E402
                        build_trainedon_probe, collapse_band, collapse_check,
                        holdout_gap, install_bnm, probe_stats, stage0a_anchors)
from coral_round4 import test_label_lockout                              # noqa: E402
from dann_round5 import (build_target_holdout,                            # noqa: E402
                         materialise_holdout_images)
from heads import LINEAR                                                 # noqa: E402
from lens_dataset import LensDataset, UNLABELED                          # noqa: E402
from morph_proxy import (MORPH_N_EVAL_PER_CLASS, MORPH_N_FIT_PER_CLASS,  # noqa: E402
                         MORPH_RIDGE_ALPHA, MorphProxy, build_morph_pool)
from run_coral_gamma_sweep import _append                                # noqa: E402
from run_coral_r4_dev8000 import Paths, Scorer                           # noqa: E402
from run_dann_r5 import (FOLD_GATE, N_TEST_POS, one_positive_tol,        # noqa: E402
                         resolve_paths, step0_tpr_tolerance)
from run_screen_unfreeze import assert_no_bf16                           # noqa: E402
from run_unfreeze_freematch import head_l2_norm                          # noqa: E402
from samplers import (ClassBalancedBatchSampler, PairedDomainLoader,     # noqa: E402
                      make_balanced_loader)
from train_skeleton import TrainConfig, Trainer                          # noqa: E402

REPO = os.path.dirname(_HERE)
ROUND, METHOD, ARM = 6, "bnm", "bnm_nb3"

# ---- FIXED, and none of it is swept (7.14a: the tuning budget is ZERO) ----
N_BLOCKS = 3
HEAD_TYPE = LINEAR
LR = 1e-4                 # Zoobot v2.0.1 defaults, per 7.1a and CORAL x nb3
LR_DECAY = 0.75
WEIGHT_DECAY = 0.05
BATCH_SIZE = 256
POS_FRACTION = 0.5
SEED = 42
CKPT_EVERY = 250
LOG_EVERY = 125

# ---------------------------------------------------------------------------
# THE HYBRID CADENCE (Stage-3 brief 2): every 25 steps to 500, then every 250.
# ---------------------------------------------------------------------------
# Stage 2 measured the crossing inside [0, 250] -- the budget rule reads the nuclear-norm
# trajectory, and at the selected gamma the first read was already past the ceiling. A
# finer grid resolves the crossing where the rule actually reads it.
#
# WHAT MUST NOT HAPPEN is a change of grid under a cross-arm comparison. 7.22 already
# killed one stability claim that lived on 250-8000 and dissolved on 250-4000; a second
# would be self-inflicted. So EVERY 250-MULTIPLE SURVIVES UNCHANGED and keeps its full
# evaluation, and the extra points are a strictly additional sub-grid that no cross-arm
# comparison reads.
#
# The extra points are STATISTICS-ONLY: nuclear norm, mean prediction entropy,
# predicted-positive fraction, and the 704/78 split. No TPR, no AUROC, no evaluation of
# any kind. This is enforced BY CONSTRUCTION rather than by discipline -- `stats_only_row`
# below does not take the scorer as an argument, so it cannot score even by mistake -- and
# it keeps ~18 new points out of the read-off pool that 7.16 would otherwise consume.
STATS_ONLY_EVERY = 25       # the sub-grid cadence
STATS_ONLY_UNTIL = 500      # ...applied only this far

STATS_COLUMNS = ["step", "wall_secs", "gamma_effective", "bnm_nuclear_norm",
                 "bnm_target_entropy", "bnm_pred_pos_frac", "bnm_mean_p_pos",
                 "collapse_gate_ok", "collapse_advisory_ceiling_ok",
                 "trainedon_nuclear", "heldout_nuclear", "nuclear_gap",
                 "trainedon_entropy", "heldout_entropy", "entropy_gap",
                 "trainedon_pred_pos", "heldout_pred_pos", "pred_pos_frac_gap"]


def checkpoint_kind(step: int, *, max_steps: int) -> str | None:
    """``"full"``, ``"stats"`` or ``None`` for this step.

    ``"full"``  -- a 250-multiple (or the final step): weights, logits, the diagnostic and
                   both morphology instruments. The grid every other arm is read on.
    ``"stats"`` -- a sub-grid point: statistics only, never evaluated.
    """
    if step % CKPT_EVERY == 0 or step >= max_steps:
        return "full"
    if step <= STATS_ONLY_UNTIL and step % STATS_ONLY_EVERY == 0:
        return "stats"
    return None


def stats_only_row(trainer, *, step, wall_secs, hook_stats, gamma_effective,
                   collapse, probe_images, held_images) -> dict:
    """A sub-grid reading. TAKES NO SCORER, SO IT CANNOT EVALUATE.

    That absence is the point: metric-blindness here is structural, not a promise. The
    only model-touching calls are ``probe_stats`` on two fixed 78-image slices, which
    compute the nuclear norm, the entropy and the predicted-positive fraction, and read
    no label of any kind.
    """
    gap = holdout_gap(probe_stats(trainer, probe_images),
                      probe_stats(trainer, held_images))
    return {"step": step, "wall_secs": round(wall_secs, 1),
            "gamma_effective": gamma_effective,
            "bnm_nuclear_norm": hook_stats["bnm_nuclear_norm"],
            "bnm_target_entropy": hook_stats["bnm_target_entropy"],
            "bnm_pred_pos_frac": hook_stats["bnm_pred_pos_frac"],
            "bnm_mean_p_pos": hook_stats["bnm_mean_p_pos"],
            "collapse_gate_ok": int(collapse["gate_ok"]),
            "collapse_advisory_ceiling_ok": int(collapse["advisory_ceiling_ok"]),
            "trainedon_nuclear": gap["trained_on"]["nuclear_norm"],
            "heldout_nuclear": gap["held_out"]["nuclear_norm"],
            "nuclear_gap": gap["nuclear_gap"],
            "trainedon_entropy": gap["trained_on"]["entropy"],
            "heldout_entropy": gap["held_out"]["entropy"],
            "entropy_gap": gap["entropy_gap"],
            "trainedon_pred_pos": gap["trained_on"]["pred_pos_frac"],
            "heldout_pred_pos": gap["held_out"]["pred_pos_frac"],
            "pred_pos_frac_gap": gap["pred_pos_frac_gap"]}
DEV_STEPS = 8000          # 7.14's development maximum, on power-matching grounds
D_HOLDOUT_FRAC = 0.10     # matched to the lambda=0 arm -- see the module docstring
BASELINE_A_HEAD_L2 = 28.475603
SPECTRUM_EVERY = 1
MORPH_EVERY = 25

DEFAULT_INIT_HEAD = os.path.join(REPO, "runs", "bakeoff_baseline_A",
                                 "checkpoints", "step_8000.pt")
STEP0_GRADNORMS = os.path.join(REPO, "runs", "bnm_r6_step0_gradnorms.json")
LOGIT_ARM_FMT = "bnm_nb3_fold{fold}"

TRACE_COLUMNS = ["step", "wall_secs", "loss", "sup_loss", "da_loss",
                 "gamma_effective", "bnm_nuclear_norm", "bnm_target_entropy",
                 "bnm_pred_pos_frac", "bnm_mean_p_pos",
                 "collapse_gate_ok", "collapse_advisory_ceiling_ok"]


# ===========================================================================
# config -- emitted by --emit-config for the known-null field comparison
# ===========================================================================

def build_config(args, *, fold, paths, init_head_l2, device) -> dict:
    max_steps = int(args.max_steps)
    steps_per_epoch = None       # filled by build_arm
    anchors = stage0a_anchors()
    band = collapse_band(anchors["init_pred_pos_frac"])
    return {
        "round": ROUND, "method": METHOD, "arm": ARM,
        "run_dir": f"bnm_r6_nb3_fold{fold}",
        "logit_arm": LOGIT_ARM_FMT.format(fold=fold),
        "n_blocks": N_BLOCKS, "fold": fold,
        "n_test_positives": N_TEST_POS[fold],
        "data_path": "image_space",
        "domain_adaptation": "bnm",

        "lp_ft": True,
        "init_head_from": os.path.abspath(paths["init_head_from"]),
        "init_head_l2": init_head_l2,
        "init_head_provenance": (
            "Baseline A step_8000 -- the SOURCE-ONLY linear probe, the 'LP' of LP->FT. "
            "Method-neutral and identical on every fold. The same head DANN x nb3 and the "
            "lambda=0 arm both use. DISCLOSURE (7.18): the four nb3 arms now initialise "
            "from THREE different heads -- CORAL from round-1 CORAL heads, FreeMatch from "
            "its own, DANN and BNM from Baseline A's. Pairing across folds removes fold "
            "difficulty and does not remove this."),

        "gamma": float(args.gamma),
        "gamma_ramp": None,
        "gamma_is_constant": True,
        "gamma_invariant": (
            "ENFORCED AT RUNTIME, not by a token scan. The gamma actually multiplied into "
            "the loss is recovered at the assembly site every step as da_loss / raw_bnm "
            "and asserted bit-stable against step 0; the full sequence rides in run.json. "
            "A static scan cannot see `w = min(1, i/2000)`, and this arm's whole claim is "
            "that no such w exists."),

        "max_steps": max_steps,
        "development_maximum": DEV_STEPS,
        "budget_rule": (
            "2 x the first fold-0 step at which normalised ||A||_* reaches the "
            "sharpening-only ceiling 10.8413 (Stage 0a), NEVER exceeding the declared "
            "ceiling of 8000 (7.14's development cap: a confirmatory budget never exceeds "
            "the development budget). Two refusal branches -- never crossed, or above the "
            "ceiling -- and NEITHER is clamped."),
        "selection_rule": "NONE. The budget is the selector; no argmax, no label-free rule.",

        "log_every": LOG_EVERY, "ckpt_every": CKPT_EVERY,
        "batch_size": BATCH_SIZE, "true_batch": True,
        "gradient_accumulation": False,
        "grad_checkpointing": bool(args.grad_checkpointing),
        "steps_per_epoch": steps_per_epoch,
        "lr": LR, "lr_decay": LR_DECAY, "weight_decay": WEIGHT_DECAY,
        "backbone_lr_effective": LR,
        "lr_protocol": ("zoobot_v2.0.1 defaults per 7.1a: head_lr == top-block lr, "
                        "lr_decay 0.75, weight_decay 0.05, NO LR schedule. Same values "
                        "round-4 CORAL and the lambda=0 arm used. Tuning budget: ZERO."),
        "pos_fraction": POS_FRACTION, "seed": SEED, "head_type": HEAD_TYPE,

        "target_holdout_frac": D_HOLDOUT_FRAC,
        "target_holdout_note": (
            "BNM has no discriminator and needs no holdout. It is KEPT because the "
            "lambda=0 arm is this runner's gamma=0 known-null and reusing it requires the "
            "target sampler to match field-for-field (brief 5). It is also a genuine "
            "diagnostic here: 7.12 measured the 782 target images being memorised and BNM "
            "is the objective most aligned with that, so a slice the loss never saw is the "
            "only way to tell generalisation from memorisation."),

        "collapse_band": band,
        "stage0a_anchors": anchors,

        "da_hook_slot": ("Trainer.coral, via bnm_round6.install_bnm at coral_weight=gamma. "
                         "train_skeleton.py is UNMODIFIED. BNMHook adapts the hook's "
                         "(src_adapter, tgt_adapter) signature to BNM's target-only "
                         "objective; the source tensor is ignored and test_bnm_round6.T1 "
                         "asserts the loss is bit-identical under a different source "
                         "batch, with CoralLoss as the known-null."),
        "source_in_loss": False,
        "encoder_eval": "live_encode",
        # byte-identical to the lambda=0 arm's string, so the known-null field comparison
        # is exact rather than "differs only in wording"
        "augmentation": "weak (+-3px translate, PROJECT.md 7.3)",
        "fp16": bool(args.fp16), "precision": "fp16" if args.fp16 else "fp32",
        "eval_precision": "fp32", "device": device,
        "num_workers": int(args.num_workers),
        "single_gpu": "no DataParallel / DDP / device sharding anywhere on this path",

        "expect_step0_tpr": float(FOLD_GATE[fold]["tpr"]),
        "expect_step0_auroc": float(FOLD_GATE[fold]["auroc"]),
        "step0_tpr_tolerance": step0_tpr_tolerance(fold, N_BLOCKS),
        "step0_auroc_tolerance": 1e-3,
        "step0_gate_skippable": False,
        "hard_gate": (
            "NON-SKIPPABLE. The phase-1 head is fold-INDEPENDENT, so the head-norm check "
            "proves the right FILE and only the per-fold gate proves the right FOLD."),

        "abort_mode": "log_only",
        "abort_note": ("ADVISORY, NEVER ABORTING, on every fold. Round 5's arrangement "
                       "would have killed all four confirmatory folds at ~step 450."),
        "metric_hygiene": ("the ONLY test-set metric computed is the step-0 gate; every "
                           "training segment runs inside test_label_lockout(); checkpoints "
                           "are SCORED (logits persisted) but never MEASURED"),
        "reporting_note": ("fold 0 is DEVELOPMENT (7.14) and enters no headline. "
                           "TPR@FPR=1e-3 is a RATE, never counts."),
    }


# ===========================================================================
# the arm
# ===========================================================================

def build_arm(args, paths, *, verbose=True) -> dict:
    fold = int(args.fold)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(args.fp16)
    grad_ckpt = bool(args.grad_checkpointing) and N_BLOCKS > 0
    max_steps = int(args.max_steps)
    head_ckpt = os.path.abspath(paths["init_head_from"])

    assert_no_bf16("fp16" if fp16 else "fp32", device)

    if not os.path.isfile(head_ckpt):
        raise SystemExit(f"[bnm-r6] ABORT: phase-1 head not found: {head_ckpt}")
    l2 = round(head_l2_norm(head_ckpt), 6)
    if abs(l2 - BASELINE_A_HEAD_L2) > 1e-4:
        raise SystemExit(
            f"[bnm-r6] ABORT: phase-1 head ||.||_2 = {l2}, pinned Baseline A value is "
            f"{BASELINE_A_HEAD_L2}. NON-SKIPPABLE and there is no override flag.")

    src_ds = LensDataset(paths["splits"], paths["data_root"], role="source_train",
                         aug="weak", n_views=1)
    tgt_ds = LensDataset(paths["splits"], paths["data_root"], role="target", fold=fold,
                         aug="weak", n_views=1)
    assert np.all(tgt_ds.labels == UNLABELED), "target labels must all be UNLABELED(-1)"

    # Matched to the lambda=0 arm, NOT because BNM needs it -- see the module docstring.
    holdout = build_target_holdout(tgt_ds, frac=D_HOLDOUT_FRAC, seed=SEED + fold)
    if verbose:
        print(f"[holdout] target pool {holdout['n_pool']} -> BNM trains on "
              f"{holdout['n_train']}, {holdout['n_holdout']} held out "
              f"(matched to the lambda=0 arm's sampler, seed {holdout['seed']})",
              flush=True)

    steps = len(src_ds) // BATCH_SIZE
    epochs_planned = max(1, int(math.ceil(max_steps / max(steps, 1))))

    src_loader = make_balanced_loader(src_ds, BATCH_SIZE, use_latent=False,
                                      num_batches=steps, pos_fraction=POS_FRACTION,
                                      seed=SEED + fold, num_workers=int(args.num_workers),
                                      pin_memory=True,
                                      persistent_workers=int(args.num_workers) > 0)
    tgt_sampler = ClassBalancedBatchSampler(holdout["train_pos"], holdout["train_neg"],
                                            BATCH_SIZE, num_batches=steps,
                                            pos_fraction=POS_FRACTION,
                                            seed=SEED + fold + 1000)
    tgt_loader = DataLoader(tgt_ds, batch_sampler=tgt_sampler,
                            num_workers=int(args.num_workers), pin_memory=True,
                            persistent_workers=int(args.num_workers) > 0)
    paired = PairedDomainLoader(src_loader, tgt_loader)

    from zoobot_encoder import ZoobotEncoder
    if verbose:
        print(f"[load] Zoobot encoder (n_blocks={N_BLOCKS}, "
              f"grad_checkpointing={grad_ckpt}) ...", flush=True)
    encoder = ZoobotEncoder(encoder_mode="frozen", n_blocks=N_BLOCKS, device=device,
                            verbose=False, grad_checkpointing=grad_ckpt)

    tcfg = TrainConfig(
        splits_path=paths["splits"], data_root=paths["data_root"], fold=fold,
        head_type=HEAD_TYPE, encoder_mode="frozen", lr=LR, weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE, pos_fraction=POS_FRACTION,
        use_coral=False,                     # install_bnm flips the hook on
        total_steps=max_steps, epochs=epochs_planned, steps_per_epoch=steps,
        max_steps=max_steps, seed=SEED + fold, device=device,
        num_workers=int(args.num_workers), amp=fp16,
        n_blocks=N_BLOCKS, backbone_lr=None, lr_decay=LR_DECAY,
        init_head_from=head_ckpt, grad_checkpointing=grad_ckpt)
    trainer = Trainer(tcfg, encoder=encoder)

    # ---- THE STANDARDIZER. Fitted from the CACHED SOURCE features, exactly as
    # run_dann_r5 does. This is the one cache read that stays valid at n_blocks>0,
    # because it is a fixed constant of the source domain rather than a function of the
    # encoder being trained -- and fitting it the same way is what makes training and
    # evaluation standardize identically.
    #
    # It is done HERE, in build_arm, rather than in the training loop: --measure-step0
    # never enters that loop, and a calibration run on an unfitted standardizer would
    # either crash (it did) or, worse, silently measure gradient norms on differently
    # scaled features than training will see.
    import evaluate as _ev
    cache = _ev.FeatureCache(paths["cache_dir"])
    if not cache.has("source_train"):
        raise SystemExit(
            f"[bnm-r6] source_train.npz not found under {paths['cache_dir']}; run "
            f"cache_features.py first. The standardizer is fitted from the cached SOURCE "
            f"features so that training and evaluation standardize identically.")
    src_feats = torch.from_numpy(
        np.asarray(cache.load("source_train")["features"],
                   dtype=np.float32)).to(device)
    trainer.standardizer.fit(src_feats)
    del src_feats
    if verbose:
        print(f"[standardizer] fitted from cached source_train "
              f"({paths['cache_dir']})", flush=True)

    hook = install_bnm(trainer, gamma=float(args.gamma))
    if verbose:
        print(f"[bnm] installed: {hook.extra_repr()}", flush=True)

    cfg = build_config(args, fold=fold, paths=paths, init_head_l2=l2, device=device)
    cfg["steps_per_epoch"] = steps
    cfg["epochs_planned"] = epochs_planned
    cfg["target_holdout_n"] = holdout["n_holdout"]
    cfg["target_holdout_indices"] = list(map(int, holdout["holdout_idx"])) \
        if "holdout_idx" in holdout else None

    # ---- THE 704/78 MEMORISATION DIAGNOSTIC, size-matched ----
    # The held-out 78 are a slice the BNM loss never sees. The trained-on probe is the
    # SAME SIZE, drawn the same stratified way, and materialised once -- because
    # normalised ||A||_* is sqrt(B/C) at a confident state, so an unmatched 704-vs-78
    # comparison reads 3.00x higher at IDENTICAL behaviour and would report sqrt(B) as
    # memorisation.
    # ---- BOTH MORPHOLOGY INSTRUMENTS. 7.19: they read ~0.34 apart at step 0
    # (trained-head ~0.57, frozen probe ~0.92) and NEITHER may be quoted as the other.
    # The frozen probe is what Stage 2's condition 3 reads, so it must be a real
    # measurement -- an absent column with a default would make that condition pass
    # silently, which is worse than not having it.
    morph_pool = build_morph_pool(paths["splits"], paths["data_root"],
                                  n_fit_per_class=MORPH_N_FIT_PER_CLASS,
                                  n_eval_per_class=MORPH_N_EVAL_PER_CLASS,
                                  seed=SEED + fold, verbose=verbose)
    morph = MorphProxy(morph_pool, alpha=MORPH_RIDGE_ALPHA)

    probe = build_trainedon_probe(tgt_ds, holdout, seed=SEED + fold)
    held_images = materialise_holdout_images(tgt_ds, holdout["holdout"])
    probe_images = materialise_holdout_images(tgt_ds, probe["probe"])
    if verbose:
        print(f"[diagnostic] size-matched target probes: {probe['n_probe']} trained-on vs "
              f"{held_images.shape[0]} held-out (disjoint: "
              f"{probe['disjoint_from_holdout']})", flush=True)
    cfg["trainedon_probe"] = probe

    return dict(cfg=cfg, trainer=trainer, hook=hook, paired=paired, encoder=encoder,
                src_ds=src_ds, tgt_ds=tgt_ds, holdout=holdout, steps=steps,
                held_images=held_images, probe_images=probe_images, probe=probe,
                morph=morph,
                epochs_planned=epochs_planned, device=device, fp16=fp16,
                max_steps=max_steps, head_ckpt=head_ckpt, fold=fold)


# ===========================================================================
# --measure-step0 : one forward/backward, no optimiser step, METRIC-BLIND
# ===========================================================================

def measure_step0(args, paths, *, out_path=STEP0_GRADNORMS, verbose=True) -> dict:
    """Stage 2's grid input. Exits before any optimiser step.

    METRIC-BLIND BY CONSTRUCTION: nothing on this path computes a metric or scores the
    test pool. ``build_arm`` does import ``evaluate.FeatureCache`` -- a cache ACCESSOR,
    used only to read ``source_train`` for the standardizer -- and ``test_bnm_round6.T11``
    both whitelists exactly that symbol and asserts that no test-set cache key is named
    anywhere on the path, with the scoring path as the known-null proving the check bites.
    """
    from bnm_round6 import bnm_gradient_calibration

    A = build_arm(args, paths, verbose=verbose)
    trainer = A["trainer"]

    src_batch, tgt_batch = next(iter(A["paired"]))
    cal = bnm_gradient_calibration(trainer, src_batch, tgt_batch)

    anchors = stage0a_anchors()
    out = {
        "round": ROUND, "arm": ARM, "fold": A["fold"],
        "purpose": ("Stage 2's gamma grid input: the BNM term's gradient-norm "
                    "contribution relative to source cross-entropy AT STEP 0."),
        "exited_before_any_optimiser_step": True,
        "reads_a_metric": False,
        "metric_blindness": ("no metric is computed and the test pool is never scored on "
                             "this path; the only `evaluate` symbol used is FeatureCache, "
                             "reading source_train for the standardizer. test_bnm_round6.T11 "
                             "asserts both, with known-nulls."),
        "device": A["device"], "fp16": A["fp16"],
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "batch_size": BATCH_SIZE,
        "stage0a_anchors": anchors,
        **cal,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    if verbose:
        print("\n" + "=" * 78)
        print("STEP-0 GRADIENT NORMS  (metric-blind; no optimiser step was taken)")
        print("=" * 78)
        print(f"  ||grad CE||            {cal['ce_grad_norm']:.6e}")
        print(f"  ||grad BNM(gamma=1)||  {cal['bnm_grad_norm_at_gamma_1']:.6e}")
        print(f"  ratio BNM/CE           {cal['ratio_bnm_over_ce_at_gamma_1']:.6e}")
        print(f"  parameters in the norm {cal['n_params_in_norm']}")
        print(f"\n  gamma grid at {{10%, 50%, 200%}} of the CE gradient norm:")
        for f, g in zip(cal["fractions"], cal["gamma_grid"]):
            print(f"    {f*100:>5.0f}%  ->  gamma = {g:.6g}")
        print(f"\nwrote {out_path}")
    return out


# ===========================================================================
# the step-0 gate -- NON-SKIPPABLE
# ===========================================================================

def step0_gate(scorer, fold, *, tpr_tol, auroc_tol, verbose=True) -> dict:
    import evaluate as ev
    import metrics as me

    _, _, pos, neg = scorer.dump(0, name="true_init")
    y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
    s = np.concatenate([pos, neg])
    tpr = float(me.tpr_at_fpr(y, s, target_fpr=1e-3))
    auroc = float(me.auroc(y, s))
    exp = FOLD_GATE[int(fold)]
    dt, da = abs(tpr - exp["tpr"]), abs(auroc - exp["auroc"])
    ok = dt <= tpr_tol and da <= auroc_tol
    rec = {"tpr": tpr, "auroc": auroc, "expected_tpr": exp["tpr"],
           "expected_auroc": exp["auroc"], "tpr_abs_diff": dt, "auroc_abs_diff": da,
           "tpr_tolerance": tpr_tol, "auroc_tolerance": auroc_tol,
           "verdict": "PASS" if ok else "FAIL", "n_test_positives": N_TEST_POS[int(fold)]}
    if verbose:
        print(f"[gate] fold {fold}: TPR {tpr:.8f} vs {exp['tpr']:.8f} (|d|={dt:.2e}, "
              f"tol {tpr_tol:.2e}) | AUROC {auroc:.8f} vs {exp['auroc']:.8f} "
              f"(|d|={da:.2e}, tol {auroc_tol:.0e}) -> {rec['verdict']}", flush=True)
    if not ok:
        raise SystemExit(
            f"[bnm-r6] STEP-0 GATE FAILED on fold {fold}. The phase-1 head is "
            f"fold-independent, so this is the only artifact proving fold identity. "
            f"Fix by re-scoring the fold table on the venue, NEVER by widening the gate.")
    return rec


# ===========================================================================
# training
# ===========================================================================

def train_loop(*, A, P, cfg, scorer, verbose=True) -> dict:
    trainer, hook, paired = A["trainer"], A["hook"], A["paired"]
    max_steps, steps = A["max_steps"], A["steps"]
    band = cfg["collapse_band"]
    inv = GammaInvariant(declared_gamma=float(cfg["gamma"]))

    t0 = time.time()
    gstep, epoch = 0, 0
    ckpt_rows = []
    n_stats_rows = 0
    # a SEPARATE file, so checkpoints.csv still holds exactly the 250 grid and no
    # cross-arm comparison can pick up a sub-grid row by accident (7.22 lesson).
    # Paths is shared with the other arms, so this is derived here rather than added
    # to that class.
    stats_csv = os.path.join(P.out, "stats_only.csv")
    trainer.head.train()

    while gstep < max_steps:
        epoch += 1
        for src_batch, tgt_batch in paired:
            if gstep >= max_steps:
                break
            gstep += 1
            st = trainer.train_step(src_batch, tgt_batch, step=gstep,
                                    total_steps=max_steps)
            hs = hook.last_stats

            # ---- THE RUNTIME INVARIANT. Fails loudly on the first divergence. ----
            obs = inv.observe(gstep, cfg_coral_weight=trainer.cfg.coral_weight,
                              hook_gamma=hook.gamma, da_loss=st["da_loss"],
                              raw_loss=hs["bnm_raw_loss"])

            cc = collapse_check(hs["bnm_pred_pos_frac"], band)

            if gstep % LOG_EVERY == 0:
                _append(P.trace_csv, TRACE_COLUMNS, {
                    "step": gstep, "wall_secs": round(time.time() - t0, 1),
                    "loss": st["loss"], "sup_loss": st["sup_loss"],
                    "da_loss": st["da_loss"],
                    "gamma_effective": obs["gamma_effective"],
                    "bnm_nuclear_norm": hs["bnm_nuclear_norm"],
                    "bnm_target_entropy": hs["bnm_target_entropy"],
                    "bnm_pred_pos_frac": hs["bnm_pred_pos_frac"],
                    "bnm_mean_p_pos": hs["bnm_mean_p_pos"],
                    "collapse_gate_ok": int(cc["gate_ok"]),
                    "collapse_advisory_ceiling_ok": int(cc["advisory_ceiling_ok"]),
                })
                if verbose:
                    eta = ((time.time() - t0) / max(gstep, 1) * (max_steps - gstep) / 60.0)
                    print(f"    [step {gstep:>5}/{max_steps}] sup={st['sup_loss']:.5f} "
                          f"bnm={hs['bnm_nuclear_norm']:.4f} "
                          f"ent={hs['bnm_target_entropy']:.4f} "
                          f"pos%={hs['bnm_pred_pos_frac']:.4f} "
                          f"g_eff={obs['gamma_effective']:.6g} "
                          f"gate={'ok' if cc['gate_ok'] else 'FIRED'} "
                          f"adv={'ok' if cc['advisory_ceiling_ok'] else 'FIRED'} "
                          f"(~{eta:.0f} min left)", flush=True)

            kind = checkpoint_kind(gstep, max_steps=max_steps)

            if kind == "stats":
                # SUB-GRID: statistics only. No weights, no logits, no evaluation.
                row = stats_only_row(
                    trainer, step=gstep, wall_secs=time.time() - t0, hook_stats=hs,
                    gamma_effective=obs["gamma_effective"], collapse=cc,
                    probe_images=A["probe_images"], held_images=A["held_images"])
                _append(stats_csv, STATS_COLUMNS, row)
                n_stats_rows += 1

            if kind == "full":
                trainer.save(os.path.join(P.ckpt, f"step_{gstep}.pt"), full_state=False)
                dst, secs, _, _ = scorer.dump(gstep)
                # THE MEMORISATION DIAGNOSTIC, pre-registered before Stage 3 runs.
                gap = holdout_gap(probe_stats(trainer, A["probe_images"]),
                                  probe_stats(trainer, A["held_images"]))
                mp = A["morph"].measure(trainer)
                ckpt_rows.append({"step": gstep,
                                  "logit_file": os.path.relpath(dst, P.out).replace("\\", "/"),
                                  "score_seconds": round(secs, 1),
                                  "bnm_nuclear_norm": hs["bnm_nuclear_norm"],
                                  "bnm_pred_pos_frac": hs["bnm_pred_pos_frac"],
                                  "gamma_effective": obs["gamma_effective"],
                                  "trainedon_nuclear": gap["trained_on"]["nuclear_norm"],
                                  "heldout_nuclear": gap["held_out"]["nuclear_norm"],
                                  "nuclear_gap": gap["nuclear_gap"],
                                  "trainedon_entropy": gap["trained_on"]["entropy"],
                                  "heldout_entropy": gap["held_out"]["entropy"],
                                  "entropy_gap": gap["entropy_gap"],
                                  "trainedon_pred_pos": gap["trained_on"]["pred_pos_frac"],
                                  "heldout_pred_pos": gap["held_out"]["pred_pos_frac"],
                                  "pred_pos_frac_gap": gap["pred_pos_frac_gap"],
                                  # BOTH instruments, never one quoted as the other
                                  "morph_head_auroc": mp["morph_head_auroc"],
                                  "morph_head_auroc_pool": mp["morph_head_auroc_pool"],
                                  "frozen_probe_auroc": mp["morph_proxy_auroc"],
                                  "frozen_probe_auroc_encoder":
                                      mp["morph_proxy_auroc_encoder"]})
                _append(P.ckpt_csv, list(ckpt_rows[-1].keys()), ckpt_rows[-1])
                if verbose:
                    print(f"    [ckpt {gstep:>5}] weights + logits written "
                          f"({secs:.0f}s to score)", flush=True)

    return {"steps_completed": gstep, "epochs": epoch,
            "wall_minutes": (time.time() - t0) / 60.0,
            "n_checkpoints_saved": len(ckpt_rows),
            "n_stats_only_rows": n_stats_rows,
            "cadence": (f"every {STATS_ONLY_EVERY} steps to "
                        f"{STATS_ONLY_UNTIL} (STATISTICS ONLY), then every "
                        f"{CKPT_EVERY} (full evaluation)"),
            "gamma_invariant": inv.to_artifact()}


# ===========================================================================

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0, choices=sorted(FOLD_GATE))
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=DEV_STEPS)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--input-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--init-head-from", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                   action="store_false", default=True)
    p.add_argument("--emit-config", default=None,
                   help="write run_config.json from the SAME construction code and exit")
    p.add_argument("--measure-step0", action="store_true", default=False,
                   help="one forward/backward, write the gradient norms, exit BEFORE any "
                        "optimiser step")
    p.add_argument("--step0-out", default=STEP0_GRADNORMS)
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    paths = resolve_paths(args, REPO)
    if args.init_head_from:
        paths["init_head_from"] = args.init_head_from
    paths.setdefault("init_head_from", DEFAULT_INIT_HEAD)

    if args.measure_step0:
        measure_step0(args, paths, out_path=args.step0_out)
        return 0

    if args.emit_config:
        A = build_arm(args, paths, verbose=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_config)) or ".",
                    exist_ok=True)
        with open(args.emit_config, "w", encoding="utf-8") as fh:
            json.dump(A["cfg"], fh, indent=2)
        print(f"[emit-config] {ARM} fold {args.fold} -> {args.emit_config}")
        print("[emit-config] nothing was trained.")
        return 0

    out_dir = args.out_dir or os.path.join(REPO, "runs", f"bnm_r6_nb3_fold{args.fold}")
    P = Paths(out_dir)
    A = build_arm(args, paths, verbose=True)
    cfg = A["cfg"]
    cfg["out_dir"] = P.out
    cfg["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cfg["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    from live_encode import LiveEncoder
    # fold is POSITIONAL and required -- LiveEncoder serves one fold's test positives and
    # target pool, so a call that omits it is scoring nothing in particular. There is no
    # cache_dir parameter: the whole point of live encoding is that it re-encodes with the
    # CURRENT encoder rather than reading a cache. Matches run_dann_r5.py:2254 exactly, so
    # the lambda=0 arm and this one score through identical machinery.
    live = LiveEncoder(paths["splits"], paths["data_root"], int(args.fold),
                       batch_size=A["cfg"]["batch_size"], num_workers=0,
                       device=A["device"], verbose=True)
    scorer = Scorer(A["trainer"], live, int(args.fold), P.logits,
                    arm=LOGIT_ARM_FMT.format(fold=args.fold), fp16=False)

    gate = step0_gate(scorer, args.fold,
                      tpr_tol=cfg["step0_tpr_tolerance"],
                      auroc_tol=cfg["step0_auroc_tolerance"])
    cfg["step0_gate"] = gate

    with test_label_lockout():
        result = train_loop(A=A, P=P, cfg=cfg, scorer=scorer)

    cfg.update(result)
    cfg["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cfg["n_checkpoints_expected"] = int(args.max_steps) // CKPT_EVERY
    with open(os.path.join(P.out, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    with open(P.run_json, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    inv = result["gamma_invariant"]
    print(f"\n[bnm-r6] done: {result['steps_completed']} steps, "
          f"{result['n_checkpoints_saved']} checkpoints, "
          f"{result['wall_minutes']:.1f} min")
    print(f"[bnm-r6] gamma invariant: {inv['n_observations']} observations, "
          f"HELD={inv['held']}, gamma_effective range "
          f"[{inv['gamma_effective_min']:.8g}, {inv['gamma_effective_max']:.8g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
