"""
run_coral_r4_dev8000.py
=======================
CORAL's PROJECT.md 7.14 FOLD-0 DEVELOPMENT RUN -- the one the method never got.

    ONE ARM.  coral_nb3, gamma = 16, fold 0, 8000 steps, image-space, FRESH FROM STEP 0.

7.14 gives every method its own fold-0 development run to fix its own budget before its
confirmatory folds exist. FreeMatch got one (7.10, 8000 steps). CORAL never did: round 4
went straight from a 500-step calibration to three 2000-step gamma-sweep arms, and gamma=16
was selected from those (runs/coral_r4_gamma_selection.json, branch C). This run extends the
SELECTED arm to the 8000-step development maximum that runs/devmax_truncation.json licenses
-- the only cap at which FreeMatch's decay was detectable at all.

WHAT THIS SCRIPT IS AND IS NOT
------------------------------
It is the 2000-step gamma=16 sweep arm, RUN AGAIN FROM STEP 0 FOR 8000 STEPS, with logits
persisted. It is NOT a continuation of that arm. 7.11 documented three pieces of training
state that restart silently when a run is continued from a checkpoint -- threshold buffers,
the consistency-weight schedule position, and AdamW's moments -- and that cost a full round
to find. This run therefore starts clean: same seed, same LP->FT init, same everything, step
0 to step 8000 in one trajectory. The first 2000 steps are a REPRODUCTION of the existing
arm and are checked as such (see THE REPRODUCTION CHECK below).

CONFIGURATION PROVENANCE -- read from the stored arm, not retyped
------------------------------------------------------------------
The arm is constructed through ``run_coral_round4.build_arm("coral_nb3", ...)`` with
``for_calibration=True`` and ``trainer.cfg.coral_weight`` set afterwards -- the SAME two
lines ``run_coral_gamma_sweep.py`` uses, so the configuration is the pinned one by
construction rather than by transcription. It is then DIFFED FIELD BY FIELD against the
stored config of the existing arm
(``kaggle/coral_r4/sweep_run/runs/coral_r4_sweep/g16/sweep.json``), and the run REFUSES TO
START on any mismatch outside the declared expected-difference list (``--allow-config-drift``
overrides, and says so in run.json).

Keys the stored file does not carry (it predates them, or they live in the shared code path)
take their reference from ``REFERENCE_SUPPLEMENT`` below, each with its provenance printed
in the diff table. ``num_workers`` is in that list and is NOT treated as cosmetic: the
DataLoader's worker count determines the augmentation RNG stream, so a different value gives
a different (still valid, but not comparable) trajectory.

THE REPRODUCTION CHECK (steps 250 and 2000)
-------------------------------------------
At each of those steps the run compares its own per-domain per-dimension std, per-domain
mean ||f||, raw CORAL and sup_loss against the stored arm's ``steps.csv``, both at the exact
step and as a window mean over the preceding ``--repro-window`` steps. It prints the table
either way and CONTINUES either way -- an 8000-step session is not worth throwing away over
a check result, and the logits are persisted regardless -- but a failure is printed inside a
banner that cannot be missed and recorded in ``run.json`` and ``repro_check.json``.

    WHY THE TOLERANCES ARE WHAT THEY ARE. Bit-exactness is NOT expected and is not the bar:
    cuDNN's convolution backward is non-deterministic by default, so two identical runs drift
    apart at some rate no one here has measured. What the check must separate is that drift
    from a CONFIGURATION change. The sweep itself calibrates that: the smallest configuration
    change available in this design is gamma 16 -> 4, and at step 250 it moves per-dimension
    std by +74% (0.202 -> 0.352) and raw CORAL by 4.6x (5.7e-4 -> 2.7e-3); at step 2000 by
    +58% and 4.7x. The tolerances below (15% on per-dimension std, 2.5x on raw CORAL) are set
    inside those margins with roughly 4x and 2.5x of room, so a real config change is caught
    at both check steps. The verdict is the RELATIVE difference alone; the window-mean
    z-score is printed as context but decides nothing, because the reference window is short
    and the quantity drifts monotonically inside it, which makes its sd an unreliable noise
    band (the suite caught exactly that -- the gamma=4 arm scored z = 5.8 at step 2000 while
    being 58% away). ``sup_loss`` is reported but is never an arbiter -- its per-step CV in
    the tail is ~0.67, and the feature NORM is nearly gamma-invariant, so it is a scale
    sanity check rather than a discriminator.

    The amp scale and the cumulative count of fp16-skipped steps are compared too. They are
    discrete and highly diagnostic: the reference arm skipped exactly 3 steps, with the scale
    walking 32768 -> 16384 -> 8192 and staying there. A different early-scale search shows up
    immediately.

WHAT IT WRITES  (all under --out-dir; the tree drops straight into runs/ locally)
--------------------------------------------------------------------------------
    run.json            config, provenance, script hashes, peak memory, timings, the
                        config-diff verdict, the repro verdict, every resume event, and the
                        abort record if the norm band is breached
    config_diff.json    the field-by-field diff against the stored arm
    repro_check.json    the reproduction tables at 250 and 2000
    trace.csv           one row per --log-every steps  (the sweep's exact columns)
    steps.csv           the per-STEP series              (the sweep's exact columns)
    checkpoints.csv     one row per checkpoint: per-domain mean ||f||, per-domain mean
                        per-dimension std, raw CORAL, sup_loss -- window means over the
                        checkpoint interval, plus the instantaneous values
    checkpoints/step_*.pt          lean weight checkpoints, every --ckpt-every steps
    logits/ids.npz                 pos_ids / neg_ids, the stable ids, written once
    logits/true_init.npz           the LP->FT head at step 0
    logits/g16dev8000/step_*.npz   pos / neg logits at EVERY checkpoint
    resume/latest.pt               weights + AdamW moments + GradScaler + DA buffers + RNG

The ``logits/`` tree is the format of ``runs/coral_r4_logits/`` (ids.npz + true_init.npz at
the top, one directory per arm holding ``step_*.npz`` with ``pos`` and ``neg``), and the id
order is CHECKED against that directory's ``ids.npz`` at step 0. A previous pass kept only
scalar TPR/AUROC and had to be re-scored from scratch; this one keeps the logits, so every
later question is a CPU question.

METRIC HYGIENE. The only test-set metric this script ever computes is the step-0 plumbing
gate (TPR@1e-3 + AUROC of the LP->FT head, which is a KNOWN NULL -- it must reproduce round-1
CORAL fold-0's own number, and that is why the gate exists). After step 0 it computes no
metric at all: every training segment runs inside ``coral_round4.test_label_lockout()``, and
the checkpoints are only SCORED, never MEASURED. The budget read-off happens locally,
afterwards, from the persisted logits, under the rule 7.14 requires be fixed in advance.
Fold 0 is DEVELOPMENT and no number from this run enters a headline.

SINGLE GPU. There is no DataParallel, no DistributedDataParallel and no device sharding
anywhere in this path, deliberately: training is sequential and halving the per-device batch
would change the 640x640 covariance CORAL estimates from 256 draws to 128 (7.1a -- the same
reason gradient accumulation is refused). If more than one GPU is visible the run uses
cuda:0 and says so.

Run:
    python -u scripts/run_coral_r4_dev8000.py --out-dir /kaggle/working/runs/coral_r4_nb3_dev8000_fold0
    python -u scripts/run_coral_r4_dev8000.py --dry-run          # mock data, 3 steps, exits
    python -u scripts/run_coral_r4_dev8000.py --config-diff-only # build + diff + exit
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from coral_round4 import (DEFAULT_FEAT_NORM_CEILING, DEFAULT_FEAT_NORM_FLOOR,   # noqa: E402
                          CoralStepWindow, FeatureNormCeiling,
                          check_feature_norm_ceiling, install_fp32_coral,
                          test_label_lockout)
import run_coral_round4 as R4                                                   # noqa: E402
from run_coral_gamma_sweep import STEP_COLUMNS, TRACE_COLUMNS, _append          # noqa: E402

# ---------------------------------------------------------------------------
# THE ARM
# ---------------------------------------------------------------------------
RUN_NAME = "coral_r4_nb3_dev8000_fold0"
ARM_KEY = "coral_nb3"
LOGIT_ARM = "g16dev8000"        # the arm folder inside logits/, coral_r4_logits-shaped
GAMMA = 16.0                    # runs/coral_r4_gamma_selection.json, branch C -- PINNED
FOLD = 0                        # DEVELOPMENT fold (7.11 / 7.14)
DEV_STEPS = 8000                # runs/devmax_truncation.json: the only cap that resolved
LOG_EVERY = 125                 # the sweep arm's cadence -- keep it, the traces are compared
CKPT_EVERY = 250
REPRO_STEPS = (250, 2000)
REPRO_WINDOW = 25

# ---------------------------------------------------------------------------
# THE CONFIG DIFF
# ---------------------------------------------------------------------------
# Every key compared against the stored arm. Order is the printing order.
DIFF_KEYS = (
    "gamma", "n_blocks", "fold", "max_steps", "log_every", "ckpt_every",
    "feat_norm_ceiling", "feat_norm_floor", "coral_target_ema",
    "init_head_from_tail", "init_head_l2", "batch_size", "steps_per_epoch",
    "lr", "backbone_lr_effective", "lr_decay", "weight_decay",
    "coral_shrinkage", "coral_include_mean", "coral_weight_schedule",
    "coral_attachment", "coral_ema_momentum",
    "precision", "coral_loss_precision", "eval_precision",
    "seed", "augmentation", "head_type", "pos_fraction", "true_batch",
    "gradient_accumulation", "grad_checkpointing", "num_workers",
)

# The ONE difference this run is allowed to have, and why.
EXPECTED_DIFFS = {
    "max_steps": (2000, DEV_STEPS,
                  "THE point of the run: 7.14's fold-0 development maximum, licensed by "
                  "runs/devmax_truncation.json (8000 is the only cap that resolved a decay)"),
}

# Reference values for keys the stored sweep.json does not carry, each with provenance.
# A key here is NOT a key waived -- it is a key whose reference had to be sourced elsewhere.
REFERENCE_SUPPLEMENT = {
    "num_workers": (2, "the g16 arm's own command line, logs/g16.log header: --num-workers 2"),
    "feat_norm_floor": (None,
                        "DID NOT EXIST when the g16 arm ran -- the two-sided floor was added "
                        "2026-08-19, after it. This run adds it (40.0). It is an ABORT "
                        "GUARD, not a training term: it changes no gradient and no update, "
                        "and it can only fire if mean ||f|| falls below 40 against a ~63-66 "
                        "trajectory. Declared, not hidden."),
    "seed": (R4.SEED, "shared code constant run_coral_round4.SEED"),
    "augmentation": ("weak (+-3px translate, PROJECT.md §7.3)",
                     "shared code path: build_config sets it for both runs"),
    "head_type": (R4.HEAD_TYPE, "shared code constant run_coral_round4.HEAD_TYPE"),
    "pos_fraction": (R4.POS_FRACTION, "shared code constant run_coral_round4.POS_FRACTION"),
    "true_batch": (True, "shared code path; 7.1a forbids accumulation for CORAL"),
    "gradient_accumulation": (False, "shared code path; 7.1a forbids it for CORAL"),
    "grad_checkpointing": (True, "shared code path: mandatory at n_blocks=3 (7.1a)"),
    "coral_weight_schedule": ("constant",
                              "shared code constant: warmup_cosine is removed from this path"),
    "coral_attachment": ("head.forward_with_features[1] (post-adapter, post-LayerNorm)",
                         "shared code constant, pinned in build_config"),
    "coral_ema_momentum": (R4.CORAL_EMA_MOMENTUM,
                           "shared code constant (inert while coral_target_ema is False)"),
    "eval_precision": ("fp32",
                       "the g16 arm computed no eval at all (it ran inside the lockout). "
                       "fp32 is the precision runs/coral_r4_logits was scored at, which is "
                       "what this run's logits must be comparable to."),
    "init_head_from_tail": ("runs/bakeoff_coral/fold0/checkpoints/step_8000.pt",
                            "derived from the stored init_head_from (an absolute Kaggle path)"),
}

# Quantities compared by the reproduction check, and the tolerance each is judged at.
#   rel     : |new-ref| / |ref| -- THE verdict criterion
#   z       : |new-ref| / sd(reference window) -- reported as CONTEXT, never the verdict
#   arbiter : does a failure here flip the verdict?
#
# THE VERDICT IS RELATIVE-DIFFERENCE ONLY, and that is a deliberate correction. An earlier
# version also passed a quantity whose window-mean z-score was small, on the reasoning that
# agreement within the reference's own scatter is agreement. The suite's discrimination test
# showed why that is wrong: at step 2000 the gamma=4 arm -- a REAL config change, 58% away
# on per-dimension std -- came in at z = 5.8 and was waved through, because the reference
# window's sd is small while the quantity is drifting monotonically inside it. A noise band
# built from a drifting window is not a noise band. z stays in the table because it is
# useful to see; it no longer decides anything.
#
# The tolerances are calibrated on the config-change margin, and the asymmetry is
# intentional: a false alarm costs a printed warning, a miss costs a curve wrongly treated
# as comparable to work already analysed. Every number is printed, so the reader judges.
REPRO_TOL = {
    "src_perdim_std": dict(rel=0.15, z=6.0, arbiter=True),
    "tgt_perdim_std": dict(rel=0.15, z=6.0, arbiter=True),
    "coral_raw":      dict(rel=1.50, z=6.0, arbiter=True),
    "src_feat_l2":    dict(rel=0.02, z=6.0, arbiter=True),
    "tgt_feat_l2":    dict(rel=0.02, z=6.0, arbiter=True),
    "sup_loss":       dict(rel=1.00, z=6.0, arbiter=False),
}
BIT_EXACT_RTOL = 1e-6

# The margin the tolerances are calibrated against -- measured, not asserted. Printed with
# the table so a reader can see what the check can and cannot separate.
REPRO_MARGIN_NOTE = (
    "calibration: the smallest CONFIG change available in this design (gamma 16 -> 4, "
    "measured in the sweep's own arms) moves per-dimension std by +74% at step 250 and +58% "
    "at step 2000, and raw CORAL by 4.6x and 4.7x -- against tolerances of 15% and 2.5x, so "
    "it is caught at both steps with ~4x and ~2.5x of margin. NOTE that the feature NORM is "
    "nearly gamma-invariant (0.3% between those two arms at step 2000), so ||f|| is a scale "
    "sanity check, NOT a discriminator -- which is the round-4 variance-collapse finding "
    "restated: covariance moved 56x while the norm barely moved."
)

PEAK_GB_REFERENCE = {
    "kaggle_t4_g16_2000steps": 3.286,     # sweep.json peak_gpu_gb: the same arm, same GPU
    "local_rtx4050_nb3_batch256": 5.86,   # runs/coral_round4_design.md 8, item 3
}

SCRIPT_HASH_FILES = ("run_coral_r4_dev8000.py", "coral_round4.py", "run_coral_round4.py",
                     "run_coral_gamma_sweep.py", "train_skeleton.py", "da_losses.py",
                     "lens_dataset.py", "samplers.py", "live_encode.py", "evaluate.py",
                     "zoobot_encoder.py", "heads.py")

CHECKPOINT_COLUMNS = [
    "step", "wall_secs", "score_secs", "n_steps_in_window",
    "sup_loss_mean", "coral_raw_mean", "cov_term_mean", "mean_term_mean",
    "src_feat_l2_mean", "src_perdim_std_mean", "tgt_feat_l2_mean", "tgt_perdim_std_mean",
    "coral_raw_last", "src_feat_l2_last", "tgt_feat_l2_last", "amp_scale_last",
    "skipped_cumulative", "logit_file",
]


# ===========================================================================
# small utilities
# ===========================================================================

def _sha256(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _path_tail(p, n=5):
    parts = os.path.normpath(str(p)).replace("\\", "/").split("/")
    return "/".join(parts[-n:])


def _rng_state():
    return {"python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)}


def _set_rng_state(s):
    if not s:
        return
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


@contextlib.contextmanager
def _rng_preserved():
    """Hold the RNG streams still across a block.

    Scoring builds its own DataLoader, and a DataLoader draws its workers' base seed from
    the global torch RNG. Without this, inserting a scoring pass every 250 steps would shift
    every subsequent augmentation offset -- and the reproduction check at step 2000 would
    then report a divergence caused by the check's own instrumentation. The reference arm
    scored nothing; this makes the training stream behave as if this one did not either.
    """
    s = _rng_state()
    try:
        yield
    finally:
        _set_rng_state(s)


def _spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, dtype=np.float64))).astype(np.float64)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=np.float64))).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _truncate_csv_after(path, step):
    """Drop any row whose ``step`` exceeds ``step``.

    A session that dies between two checkpoints leaves per-step rows the resumed run is
    about to write again. Left in place they appear as duplicated / non-monotone steps in a
    series whose whole purpose is a slope.
    """
    if not os.path.isfile(path):
        return 0
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return 0
    cols = list(rows[0].keys())
    keep = [r for r in rows if int(float(r["step"])) <= int(step)]
    if len(keep) == len(rows):
        return 0
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(keep)
    return len(rows) - len(keep)


def banner(title, lines, char="!"):
    w = 92
    print("\n" + char * w, flush=True)
    print(f"{char * 3} {title}")
    for ln in lines:
        print(f"{char * 3} {ln}")
    print(char * w + "\n", flush=True)


# ===========================================================================
# path resolution -- the Kaggle mount, or local, or explicit
# ===========================================================================

def _find_under(roots, pattern):
    for root in roots:
        if not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]          # shallowest match
    return None


def resolve_paths(args, root):
    """Locate data / splits / cache / phase-1 head / reference artifacts.

    The Kaggle mount is SEARCHED, never guessed: ``data.zip`` auto-extracts to ``data/data/``
    so the pieces do not share one common root, and the enclosing directory has been
    ``/kaggle/input/<slug>`` in one session and ``/kaggle/input/datasets/<owner>/<slug>`` in
    another. This is the pattern KAGGLE_CORAL_R4_SWEEP_CELLS.md Cell 1 established; it is
    reused rather than reinvented.
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
                f"[dev8000] ABORT: could not resolve {name}.\n"
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
         os.path.join("bakeoff_coral", "fold0", "checkpoints", "step_8000.pt"))
    pick("agreement_ref", args.agreement_ref,
         os.path.join("bakeoff_baseline_A", "checkpoints", "step_8000.pt"), required=False)

    # ---- reference artifacts (the existing 2000-step arm) ----
    ref_dir = args.reference_dir
    if not ref_dir:
        hit = _find_under(search_roots, os.path.join("coral_r4_g16_2000", "sweep.json"))
        if not hit:
            hit = _find_under(search_roots,
                              os.path.join("coral_r4_sweep", "g16", "sweep.json"))
        ref_dir = os.path.dirname(hit) if hit else None
    out["reference_dir"] = os.path.abspath(ref_dir) if ref_dir else None

    ref_ids = args.reference_ids or _find_under(
        search_roots, os.path.join("coral_r4_logits", "ids.npz"))
    out["reference_ids"] = os.path.abspath(ref_ids) if ref_ids else None
    out.pop("_cache_probe", None)
    return out


# ===========================================================================
# the config diff
# ===========================================================================

def build_record(A, args, paths):
    """The sweep-shaped configuration record, so it can be diffed key-for-key."""
    c = A["cfg_json"]
    return {
        "run": RUN_NAME,
        "arm": ARM_KEY,
        "gamma": float(args.gamma),
        "n_blocks": 3,
        "fold": int(args.fold),
        "max_steps": int(args.max_steps),
        "log_every": int(args.log_every),
        "ckpt_every": int(args.ckpt_every),
        "feat_norm_ceiling": float(args.feat_norm_ceiling),
        "feat_norm_floor": float(args.feat_norm_floor),
        "coral_target_ema": bool(args.coral_target_ema),
        "init_head_from": paths["init_head_from"],
        "init_head_from_tail": _path_tail(paths["init_head_from"], 5),
        "init_head_l2": c["init_head_l2"],
        "batch_size": A["batch_size"],
        "steps_per_epoch": A["steps"],
        "epochs_planned": A["epochs_planned"],
        "lr": R4.LR,
        "backbone_lr_effective": c["backbone_lr_effective"],
        "lr_decay": R4.LR_DECAY,
        "weight_decay": R4.WEIGHT_DECAY,
        "coral_shrinkage": R4.CORAL_SHRINKAGE,
        "coral_include_mean": R4.CORAL_INCLUDE_MEAN,
        "coral_weight_schedule": c["coral_weight_schedule"],
        "coral_attachment": c["coral_attachment"],
        "coral_ema_momentum": R4.CORAL_EMA_MOMENTUM,
        "precision": c["precision"],
        "coral_loss_precision": "fp32_autocast_disabled",
        "eval_precision": c["eval_precision"],
        "seed": R4.SEED,
        "augmentation": c["augmentation"],
        "head_type": c["head_type"],
        "pos_fraction": c["pos_fraction"],
        "true_batch": c["true_batch"],
        "gradient_accumulation": c["gradient_accumulation"],
        "grad_checkpointing": c["grad_checkpointing"],
        "num_workers": int(args.num_workers),
        "device": A["device"],
    }


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def config_diff(record, ref_path, verbose=True):
    """Field-by-field against the stored arm. Returns (rows, mismatches, ref)."""
    ref = {}
    if ref_path and os.path.isfile(ref_path):
        with open(ref_path) as fh:
            ref = json.load(fh)
        if "init_head_from" in ref:
            ref["init_head_from_tail"] = _path_tail(ref["init_head_from"], 5)
    rows, mismatches = [], []
    for k in DIFF_KEYS:
        if k in ref:
            rv, prov = ref[k], "stored sweep.json"
        elif k in REFERENCE_SUPPLEMENT:
            rv, prov = REFERENCE_SUPPLEMENT[k]
        else:
            rv, prov = "<absent>", ("NO REFERENCE -- a guard that compares nothing is not "
                                    "a guard")
        cv = record.get(k, "<absent>")
        if k in EXPECTED_DIFFS:
            want_ref, want_new, why = EXPECTED_DIFFS[k]
            status = ("EXPECTED" if (_same(rv, want_ref) and _same(cv, want_new))
                      else "MISMATCH")
            prov = f"{prov}; expected difference: {why}"
        elif rv is None:
            status = "NEW"          # declared-absent in the reference, see the supplement
        elif rv == "<absent>":
            status = "MISMATCH"
        else:
            status = "MATCH" if _same(rv, cv) else "MISMATCH"
        if status == "MISMATCH":
            mismatches.append(k)
        rows.append({"key": k, "reference": rv, "this_run": cv,
                     "status": status, "provenance": prov})

    if verbose:
        print("\n" + "=" * 100)
        print("CONFIG DIFF -- this run vs the stored 2000-step gamma=16 sweep arm")
        print(f"  reference: {ref_path}")
        print("=" * 100)
        print(f"  {'key':<26} {'reference':<34} {'this run':<24} status")
        print("  " + "-" * 96)
        for r in rows:
            mark = {"MATCH": " ", "EXPECTED": "*", "NEW": "+", "MISMATCH": "X"}[r["status"]]
            print(f"{mark} {r['key']:<26} {str(r['reference'])[:33]:<34} "
                  f"{str(r['this_run'])[:23]:<24} {r['status']}")
        print("  " + "-" * 96)
        print("  legend:  (blank) identical   * declared expected difference   "
              "+ new key, absent from the reference   X MISMATCH")
        for r in rows:
            if r["status"] in ("EXPECTED", "NEW", "MISMATCH"):
                print(f"    {r['key']}: {r['provenance']}")
        print(f"  => {len(mismatches)} mismatch(es)"
              + (f": {mismatches}" if mismatches else ""))
        print("=" * 100, flush=True)
    return rows, mismatches, ref


# ===========================================================================
# the reproduction check
# ===========================================================================

class ReproCheck:
    """Compare this run's own trajectory against the stored 2000-step arm's steps.csv."""

    QUANTS = ("src_perdim_std", "tgt_perdim_std", "coral_raw",
              "src_feat_l2", "tgt_feat_l2", "sup_loss")

    def __init__(self, steps_csv, at_steps=REPRO_STEPS, window=REPRO_WINDOW):
        self.path = steps_csv
        self.at = tuple(int(s) for s in at_steps)
        self.window = int(window)
        self.rows = {}
        self.available = False
        if steps_csv and os.path.isfile(steps_csv):
            with open(steps_csv, newline="") as fh:
                for r in csv.DictReader(fh):
                    self.rows[int(float(r["step"]))] = r
            self.available = bool(self.rows)
        self.results = []

    def _ref(self, step):
        exact = self.rows.get(step)
        if exact is None:
            return None
        lo = max(1, step - self.window + 1)
        win = [self.rows[s] for s in range(lo, step + 1) if s in self.rows]
        out = {"step": step, "n_window": len(win), "exact": {}, "win_mean": {}, "win_sd": {}}
        for q in self.QUANTS:
            out["exact"][q] = _num(exact.get(q))
            v = [x for x in (_num(r.get(q)) for r in win) if x is not None]
            out["win_mean"][q] = float(np.mean(v)) if v else None
            out["win_sd"][q] = float(np.std(v)) if v else None
        out["amp_scale"] = _num(exact.get("amp_scale"))
        out["skipped_cumulative"] = float(sum(
            _num(self.rows[s].get("step_skipped")) or 0.0
            for s in range(1, step + 1) if s in self.rows))
        return out

    def compare(self, step, exact_now, win_mean_now, amp_scale_now, skipped_now):
        ref = self._ref(step)
        if ref is None:
            res = {"step": step, "status": "NO_REFERENCE",
                   "note": f"the stored arm has no row at step {step} ({self.path})"}
            self.results.append(res)
            self._print(res)
            return res

        table, failed, bit_exact = [], [], True
        for q in self.QUANTS:
            tol = REPRO_TOL[q]
            rv, nv = ref["exact"][q], _num(exact_now.get(q))
            rm, nm = ref["win_mean"][q], _num(win_mean_now.get(q))
            sd = ref["win_sd"][q]
            if rv is None or nv is None:
                table.append({"quantity": q, "status": "MISSING", "arbiter": tol["arbiter"]})
                continue
            d = nv - rv
            # d == 0 is EXACT agreement and must pass whatever the reference value is.
            # Without this an exactly-zero reference gives rel = inf and a checker that
            # calls identical values divergent -- which is what the dry run's known-null
            # (this run vs itself) exists to catch, and did.
            rel = 0.0 if d == 0 else (abs(d) / abs(rv) if rv else float("inf"))
            dm = (nm - rm) if (nm is not None and rm is not None) else float("nan")
            z = (abs(dm) / sd) if (sd and sd > 0 and math.isfinite(dm)) else float("nan")
            ok = (d == 0) or (rel <= tol["rel"])       # z is context, never the verdict
            if abs(d) > BIT_EXACT_RTOL * max(1.0, abs(rv)):
                bit_exact = False
            if not ok and tol["arbiter"]:
                failed.append(q)
            table.append({"quantity": q, "ref_exact": rv, "this_exact": nv,
                          "abs_diff": d, "rel_diff": rel,
                          "ref_win_mean": rm, "this_win_mean": nm, "ref_win_sd": sd,
                          "win_mean_z": z, "rel_tol": tol["rel"], "z_tol": tol["z"],
                          "arbiter": tol["arbiter"], "status": "PASS" if ok else "DIVERGED"})

        amp_ok = (amp_scale_now is None or ref["amp_scale"] is None
                  or float(amp_scale_now) == float(ref["amp_scale"]))
        skip_ok = (skipped_now is None or ref["skipped_cumulative"] is None
                   or float(skipped_now) == float(ref["skipped_cumulative"]))
        res = {"step": step,
               "status": "REPRODUCED" if not failed else "DIVERGED",
               "bit_exact": bool(bit_exact),
               "failed_arbiters": failed,
               "amp_scale": {"reference": ref["amp_scale"], "this_run": amp_scale_now,
                             "match": bool(amp_ok)},
               "fp16_skipped_cumulative": {"reference": ref["skipped_cumulative"],
                                           "this_run": skipped_now, "match": bool(skip_ok)},
               "window": self.window, "n_window_reference": ref["n_window"],
               "table": table,
               "tolerance_basis": REPRO_MARGIN_NOTE,
               "reference": self.path}
        self.results.append(res)
        self._print(res)
        return res

    @staticmethod
    def _print(res):
        step = res["step"]
        if res["status"] == "NO_REFERENCE":
            banner(f"REPRODUCTION CHECK at step {step}: NO REFERENCE", [res["note"]], "?")
            return
        print("\n" + "=" * 100)
        print(f"REPRODUCTION CHECK at step {step} -- vs the stored 2000-step gamma=16 arm")
        print("=" * 100)
        print(f"  {'quantity':<16} {'reference':>14} {'this run':>14} {'rel diff':>10} "
              f"{'win-mean z':>11}  {'tol rel/z':>12}  status")
        print("  " + "-" * 96)
        for r in res["table"]:
            if r.get("status") == "MISSING":
                print(f"  {r['quantity']:<16} {'--':>14} {'--':>14}  MISSING")
                continue
            arb = "" if r["arbiter"] else "  (informational)"
            print(f"  {r['quantity']:<16} {r['ref_exact']:>14.6g} {r['this_exact']:>14.6g} "
                  f"{r['rel_diff']:>10.3%} {r['win_mean_z']:>11.2f}  "
                  f"{r['rel_tol']:>6.0%}/{r['z_tol']:<5.0f} {r['status']}{arb}")
        print("  " + "-" * 96)
        a, s = res["amp_scale"], res["fp16_skipped_cumulative"]
        print(f"  amp scale          reference {a['reference']}  this run {a['this_run']}  "
              f"{'MATCH' if a['match'] else 'DIFFERS'}")
        print(f"  fp16 steps skipped reference {s['reference']}  this run {s['this_run']}  "
              f"{'MATCH' if s['match'] else 'DIFFERS'}")
        print(f"  bit-exact vs the stored arm: {res['bit_exact']}  (NOT expected -- cuDNN "
              f"conv backward is non-deterministic; informational only)")
        print(f"  {res['tolerance_basis']}")
        print("=" * 100, flush=True)
        if res["status"] != "REPRODUCED":
            banner(
                f"REPRODUCTION CHECK FAILED AT STEP {step}",
                [f"diverged on: {', '.join(res['failed_arbiters'])}",
                 "The run CONTINUES -- an 8000-step session is not worth throwing away over",
                 "a check result, and the logits are being persisted either way.",
                 "BUT: a divergent curve is not comparable to the 2000-step arm, to",
                 "runs/coral_r4_outcome.json, or to runs/coral_r4_logits/. Read the table",
                 "above and config_diff.json BEFORE using this run for anything. Suspect,",
                 "in order: a different --num-workers (it sets the augmentation RNG",
                 "stream), a different phase-1 head, target_ema left on, a different",
                 "gamma, or a different GPU architecture (7.7)."])
        else:
            print(f"[repro] step {step}: REPRODUCED (within tolerance). Continuing.\n",
                  flush=True)


# ===========================================================================
# scoring + logit persistence
# ===========================================================================

class Scorer:
    """Score the fold's test pool and persist logits in runs/coral_r4_logits/ format."""

    def __init__(self, trainer, live, fold, logit_dir, arm=LOGIT_ARM, fp16=False):
        self.tr, self.live, self.fold = trainer, live, int(fold)
        self.root = logit_dir
        self.arm_dir = os.path.join(logit_dir, arm)
        self.fp16 = bool(fp16)
        os.makedirs(self.arm_dir, exist_ok=True)
        self.ids = None

    @torch.no_grad()
    def _score(self):
        import evaluate as ev
        was_train = self.tr.head.training
        self.tr.head.eval()
        self.tr.encoder.eval()
        p = self.live.encode(self.tr.encoder, f"test_pos_fold{self.fold}", fp16=self.fp16)
        n = self.live.encode(self.tr.encoder, "test_negatives_shared", fp16=self.fp16)
        dev = str(self.tr.device)
        ps = ev.score_features(self.tr.head, self.tr.standardizer, p["features"], dev)
        ns = ev.score_features(self.tr.head, self.tr.standardizer, n["features"], dev)
        if was_train:
            self.tr.head.train()
        return (np.asarray(ps, dtype=np.float64), np.asarray(ns, dtype=np.float64),
                np.asarray(p["ids"], dtype="<U80"), np.asarray(n["ids"], dtype="<U80"))

    def dump(self, step, name=None):
        """Score the CURRENT state and write pos/neg logits. Returns (path, secs, pos, neg)."""
        t0 = time.time()
        with _rng_preserved():
            ps, ns, pid, nid = self._score()
        if self.ids is None:
            self.ids = (pid, nid)
            np.savez_compressed(os.path.join(self.root, "ids.npz"),
                                pos_ids=pid, neg_ids=nid)
        else:
            assert np.array_equal(pid, self.ids[0]) and np.array_equal(nid, self.ids[1]), \
                f"step {step}: id order changed mid-run -- the pairing would be invalid"
        fn = (name or f"step_{int(step)}") + ".npz"
        dst = os.path.join(self.root, fn) if name else os.path.join(self.arm_dir, fn)
        np.savez_compressed(dst, pos=ps, neg=ns)
        return dst, (time.time() - t0), ps, ns


def check_ids_against_reference(ids_npz, ref_ids_npz):
    """Does the id ORDER match runs/coral_r4_logits/ids.npz? Warn, never abort."""
    out = {"reference": ref_ids_npz, "checked": False}
    if not ref_ids_npz or not os.path.isfile(ref_ids_npz):
        out["note"] = "reference ids.npz not present in this session; check skipped"
        print(f"[ids] reference ids.npz not found -- id-order check SKIPPED ({ref_ids_npz})",
              flush=True)
        return out
    a = np.load(ids_npz, allow_pickle=True)
    b = np.load(ref_ids_npz, allow_pickle=True)
    pos_ok = bool(np.array_equal(a["pos_ids"], b["pos_ids"]))
    neg_ok = bool(np.array_equal(a["neg_ids"], b["neg_ids"]))
    out.update({"checked": True, "pos_ids_match": pos_ok, "neg_ids_match": neg_ok,
                "n_pos": int(a["pos_ids"].size), "n_neg": int(a["neg_ids"].size),
                "n_pos_reference": int(b["pos_ids"].size),
                "n_neg_reference": int(b["neg_ids"].size)})
    if pos_ok and neg_ok:
        print(f"[ids] id order MATCHES runs/coral_r4_logits/ids.npz "
              f"({out['n_pos']} pos / {out['n_neg']} neg) -- these logits pair row-for-row "
              f"with the existing arms.", flush=True)
    else:
        banner("ID ORDER DOES NOT MATCH runs/coral_r4_logits/ids.npz",
               [f"pos_ids match: {pos_ok}   neg_ids match: {neg_ok}",
                f"this run: {out['n_pos']} pos / {out['n_neg']} neg;  "
                f"reference: {out['n_pos_reference']} / {out['n_neg_reference']}",
                "The run CONTINUES and the ids ARE saved, so the pairing can still be",
                "rebuilt locally by joining on the id strings. But a paired bootstrap that",
                "assumes row alignment (7.12's scheme) is INVALID until that join is done."])
    return out


# ===========================================================================
# the training loop -- shared by the real run and by --dry-run
# ===========================================================================

class Paths:
    def __init__(self, out_dir):
        self.out = os.path.abspath(out_dir)
        self.ckpt = os.path.join(self.out, "checkpoints")
        self.logits = os.path.join(self.out, "logits")
        self.resume = os.path.join(self.out, "resume")
        self.steps_csv = os.path.join(self.out, "steps.csv")
        self.trace_csv = os.path.join(self.out, "trace.csv")
        self.ckpt_csv = os.path.join(self.out, "checkpoints.csv")
        self.run_json = os.path.join(self.out, "run.json")
        self.diff_json = os.path.join(self.out, "config_diff.json")
        self.repro_json = os.path.join(self.out, "repro_check.json")
        self.resume_pt = os.path.join(self.resume, "latest.pt")
        for d in (self.out, self.ckpt, self.logits, self.resume):
            os.makedirs(d, exist_ok=True)


def save_resume(trainer, P, *, gstep, epoch, batch_in_epoch, rng_epoch_start, record):
    """Weights + AdamW moments + GradScaler + DA buffers + RNG, written atomically.

    7.11 is explicit that a weights-only checkpoint restarts training state silently and
    that it cost a round to find. So a resume here restores the OPTIMIZER MOMENTS (via
    ``Trainer.save(full_state=True)``), the DA buffers, the GradScaler's scale and growth
    tracker, and the RNG streams -- not just the weights.

    HONEST LIMIT, stated rather than buried: the SAMPLE ORDER is restored exactly (the batch
    sampler is a pure function of seed+epoch, so epoch E replays identically), and the RNG
    state captured at the START of the interrupted epoch is restored before the loader
    iterator is rebuilt. But DataLoader worker seeds are derived when an iterator is created
    and this run uses persistent workers, so the AUGMENTATION OFFSETS drawn after a resume
    are not guaranteed to equal those an uninterrupted run would have drawn. That is a +-3px
    translate on individual images; it restarts no schedule, no buffer and no moment. Every
    resume is recorded in run.json so a curve that was resumed is never mistaken for one
    that was not.
    """
    tmp = P.resume_pt + ".tmp"
    trainer.save(tmp, full_state=True, schedule={
        "global_step": int(gstep),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "coral_weight": float(record["gamma"]),
        "scaler": (trainer.scaler.state_dict() if trainer._amp_enabled else None),
        "rng_epoch_start": rng_epoch_start,
        "rng_now": _rng_state(),
        "run": RUN_NAME,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    os.replace(tmp, P.resume_pt)


def load_resume(trainer, P):
    if not os.path.isfile(P.resume_pt):
        return None
    ck = trainer.load(P.resume_pt, map_location=trainer.device, load_training_state=True)
    sch = ck.get("schedule") or {}
    if sch.get("scaler") and trainer._amp_enabled:
        trainer.scaler.load_state_dict(sch["scaler"])
    return sch


def _print_peak_memory(record):
    alloc = torch.cuda.max_memory_allocated() / 2 ** 30
    resv = torch.cuda.max_memory_reserved() / 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    ref = PEAK_GB_REFERENCE
    print("\n" + "-" * 92)
    print(f"[gpu] {torch.cuda.get_device_name(0)}  |  {total:.2f} GB total  |  "
          f"{torch.cuda.device_count()} visible device(s), using cuda:0 only "
          f"(no DataParallel anywhere in this path)")
    print(f"[gpu] peak so far: {alloc:.2f} GB allocated / {resv:.2f} GB reserved")
    print(f"[gpu] reference:   {ref['kaggle_t4_g16_2000steps']:.2f} GB allocated on the SAME "
          f"arm's 2000-step Kaggle T4 run (sweep.json peak_gpu_gb); "
          f"{ref['local_rtx4050_nb3_batch256']:.2f} GB observed locally at batch 256")
    d = alloc - ref["kaggle_t4_g16_2000steps"]
    if abs(d) > 0.5:
        print(f"[gpu] NOTE: {d:+.2f} GB against the matched Kaggle reference -- materially "
              f"different, worth knowing about.")
    print("-" * 92 + "\n", flush=True)
    record["peak_gpu_gb_early"] = float(alloc)
    record["gpu_name"] = torch.cuda.get_device_name(0)
    record["gpu_total_gb"] = float(total)
    record["gpu_visible_devices"] = int(torch.cuda.device_count())


def train_loop(*, trainer, coral, A, P, cfg, record, scorer, repro, resume_sched=None,
               verbose=True):
    """The one loop. ``cfg`` carries max_steps / log_every / ckpt_every / band / gamma."""
    max_steps = int(cfg["max_steps"])
    log_every = int(cfg["log_every"])
    ckpt_every = int(cfg["ckpt_every"])
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

    window = CoralStepWindow()          # reset every log_every
    ckpt_window = CoralStepWindow()     # reset every ckpt_every
    skipped_cum = float(record.get("fp16_skipped_cumulative", 0.0))
    peak_reported = False
    aborted, abort_reason = False, None
    stop = False

    epochs_planned = max(int(A["epochs_planned"]), math.ceil(max_steps / steps_per_epoch))
    for epoch in range(start_epoch, epochs_planned + 1):
        A["src_loader"].batch_sampler.set_epoch(epoch)
        A["tgt_loader"].batch_sampler.set_epoch(epoch)
        rng_epoch_start = _rng_state()
        trainer.head.train()
        seen_in_epoch = 0
        try:
            for sbatch, tbatch in A["paired"]:
                if seen_in_epoch < skip_in_epoch:
                    seen_in_epoch += 1
                    if verbose and seen_in_epoch % 50 == 0:
                        print(f"[resume]   replayed {seen_in_epoch}/{skip_in_epoch} "
                              f"batches", flush=True)
                    continue
                seen_in_epoch += 1

                scale_before = (trainer.scaler.get_scale() if trainer._amp_enabled else None)
                with test_label_lockout(verbose=False):
                    st = trainer.train_step(sbatch, tbatch, step=gstep,
                                            total_steps=max_steps)
                skipped = bool(trainer._amp_enabled
                               and trainer.scaler.get_scale() < scale_before)
                cs = coral.step_stats()
                stats = {**st, **cs, "step_skipped": float(skipped),
                         "amp_scale": (float(trainer.scaler.get_scale())
                                       if trainer._amp_enabled else float("nan"))}
                gstep += 1
                skipped_cum += float(skipped)
                window.update(stats)
                ckpt_window.update(stats)
                _append(P.steps_csv, STEP_COLUMNS, {"step": gstep, **stats})

                check_feature_norm_ceiling(gstep, coral.last_scale,
                                           cfg["feat_norm_ceiling"], cs["coral_raw"],
                                           floor=cfg["feat_norm_floor"])

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
                              f"sup={row['sup_loss_mean']:.5f} "
                              f"coral_raw={row['coral_raw_mean']:.6g} "
                              f"src|f|={row['src_feat_l2_mean']:.3f} "
                              f"tgt|f|={row['tgt_feat_l2_mean']:.3f} "
                              f"src_sd={row['src_perdim_std_mean']:.4f} "
                              f"tgt_sd={row['tgt_perdim_std_mean']:.4f} "
                              f"({row['wall_secs']:.0f}s, ~{eta:.0f} min left)", flush=True)

                at_ckpt = (gstep % ckpt_every == 0) or (gstep >= max_steps)
                at_repro = gstep in repro.at
                if at_ckpt or at_repro:
                    crow = ckpt_window.row(gstep)
                    ckpt_window.reset()
                    if at_ckpt:
                        trainer.save(os.path.join(P.ckpt, f"step_{gstep}.pt"),
                                     full_state=False,
                                     schedule={"coral_weight": float(cfg["gamma"]),
                                               "global_step": int(gstep)})
                        dst, score_secs, _, _ = scorer.dump(gstep)
                        logit_file = os.path.relpath(dst, P.out).replace("\\", "/")
                        save_resume(trainer, P, gstep=gstep, epoch=epoch,
                                    batch_in_epoch=seen_in_epoch,
                                    rng_epoch_start=rng_epoch_start, record=record)
                        crow_out = {k: crow.get(k) for k in CHECKPOINT_COLUMNS if k in crow}
                        crow_out.update({"step": gstep,
                                         "wall_secs": round(time.time() - t_start, 1),
                                         "score_secs": round(score_secs, 1),
                                         "skipped_cumulative": skipped_cum,
                                         "logit_file": logit_file})
                        _append(P.ckpt_csv, CHECKPOINT_COLUMNS, crow_out)
                        if verbose:
                            print(f"    [ckpt {gstep:>5}] weights + logits written "
                                  f"({score_secs:.0f}s to score)", flush=True)
                    if at_repro:
                        exact_now = {k: stats.get(k) for k in ReproCheck.QUANTS}
                        win_now = {k: crow.get(f"{k}_mean") for k in ReproCheck.QUANTS}
                        repro.compare(gstep, exact_now, win_now,
                                      stats.get("amp_scale"), skipped_cum)
                        with open(P.repro_json, "w") as fh:
                            json.dump({"results": repro.results,
                                       "reference_steps_csv": repro.path,
                                       "window": repro.window,
                                       "tolerances": REPRO_TOL,
                                       "basis": REPRO_MARGIN_NOTE}, fh, indent=2)

                if gstep >= max_steps:
                    stop = True
                    break
            skip_in_epoch = 0
            if stop:
                break
        except FeatureNormCeiling as e:
            aborted, abort_reason = True, str(e)
            banner(f"FEATURE-NORM BAND BREACHED AT STEP {gstep}",
                   [str(e),
                    "Saving a final checkpoint and a final logit dump so the abort step is "
                    "scorable, then stopping."])
            trainer.save(os.path.join(P.ckpt, f"step_{gstep}_ABORT.pt"), full_state=False,
                         schedule={"coral_weight": float(cfg["gamma"]),
                                   "global_step": int(gstep), "aborted": True})
            with contextlib.suppress(Exception):
                scorer.dump(gstep)
            break

    record.update({
        "steps_completed": int(gstep),
        "aborted": bool(aborted),
        "abort_reason": abort_reason,
        "wall_minutes": (time.time() - t_start) / 60.0,
        "fp16_skipped_cumulative": skipped_cum,
        "peak_gpu_gb_allocated": (torch.cuda.max_memory_allocated() / 2 ** 30
                                  if torch.cuda.is_available() else None),
        "peak_gpu_gb_reserved": (torch.cuda.max_memory_reserved() / 2 ** 30
                                 if torch.cuda.is_available() else None),
    })
    return record


# ===========================================================================
# the step-0 gate
# ===========================================================================

def step0_gate(scorer, record, fold, tol_warn, tol_abort, auroc_tol_warn, auroc_tol_abort):
    """Score the LP->FT head itself. A KNOWN NULL: it must reproduce round-1 CORAL's number.

    WHY THIS IS WARN-AND-CONTINUE, NOT THE DESIGN DOC'S HARD ABORT AT ONE POSITIVE.
    The pinned table was re-scored LOCALLY from data/feature_cache; 7.7 measured local-Ada
    vs Kaggle-T4 differing by 0.0204 -- TWO positives -- on the same checkpoint. A one-
    positive abort would therefore fire on venue drift alone and cost a session. The
    deterministic half of the gate is not weakened: ``build_arm`` already ABORTS if the
    phase-1 head's ||.||_2 differs from the pinned value by more than 1e-4, and that is the
    check that catches a wrong head or a wrong fold. This one catches a wrong SCORING path,
    at a tolerance wide enough to survive the venue and narrow enough to catch a real fault.
    """
    from metrics import auroc as auroc_fn, tpr_at_fpr
    exp = R4.FOLD_GATE[int(fold)]
    dst, secs, ps, ns = scorer.dump(0, name="true_init")
    y = np.concatenate([np.ones(ps.size, np.int64), np.zeros(ns.size, np.int64)])
    s = np.concatenate([ps, ns])
    tpr, au = float(tpr_at_fpr(y, s)), float(auroc_fn(y, s))
    dt, da = abs(tpr - exp["tpr"]), abs(au - exp["auroc"])
    print("\n" + "=" * 92)
    print(f"STEP-0 GATE (known null: this IS the phase-1 model) -- fold {fold}, "
          f"scored in {secs:.0f}s")
    print(f"  TPR@1e-3 : {tpr:.6f}   expected {exp['tpr']:.6f}   |diff| {dt:.6f}  "
          f"(warn > {tol_warn:.6f}, abort > {tol_abort:.6f})")
    print(f"  AUROC    : {au:.6f}   expected {exp['auroc']:.6f}   |diff| {da:.6f}  "
          f"(warn > {auroc_tol_warn:.1e}, abort > {auroc_tol_abort:.1e})")
    print(f"  ||head||2: {record['init_head_l2']}   pinned {exp['head_l2']:.6f}  "
          f"(already hard-gated at 1e-4 by build_arm)")
    print("=" * 92, flush=True)
    verdict = "PASS"
    if dt > tol_abort or da > auroc_tol_abort:
        verdict = "ABORT"
    elif dt > tol_warn or da > auroc_tol_warn:
        verdict = "WARN"
    record["step0_gate"] = {"tpr": tpr, "auroc": au, "expected_tpr": exp["tpr"],
                            "expected_auroc": exp["auroc"], "tpr_abs_diff": dt,
                            "auroc_abs_diff": da, "verdict": verdict,
                            "warn_tol": tol_warn, "abort_tol": tol_abort,
                            "true_init_logits": os.path.basename(dst)}
    if verdict == "WARN":
        banner("STEP-0 GATE: OUTSIDE THE ONE-POSITIVE TOLERANCE, INSIDE THE ABORT BAND",
               [f"TPR {tpr:.6f} vs the pinned {exp['tpr']:.6f} (|diff| {dt:.6f})",
                f"AUROC {au:.6f} vs {exp['auroc']:.6f} (|diff| {da:.6f})",
                "7.7's measured local-vs-T4 drift is 0.0204 on this metric, so this is",
                "consistent with venue drift and the run continues. If it is much larger",
                "than that, stop and check the scoring path before trusting the curve."])
    elif verdict == "ABORT":
        raise SystemExit(
            f"[dev8000] ABORT at the step-0 gate: TPR@1e-3 = {tpr:.6f} against the pinned "
            f"{exp['tpr']:.6f} and AUROC {au:.6f} against {exp['auroc']:.6f}.\n"
            f"  Step 0 IS the phase-1 model, so this is a plumbing failure, not a result. "
            f"Likely causes, in order: the wrong split/cache mount, an fp16 eval encode "
            f"against an fp32 reference, a standardizer not fitted from the cached source "
            f"features, or the wrong fold.")
    else:
        print("[gate] step-0 known null reproduces round-1 CORAL fold-0. Proceeding.\n",
              flush=True)
    return record["step0_gate"]


# ===========================================================================
# --dry-run : mock data shaped like the real splits, 3 steps, exit
# ===========================================================================

def _moment_fingerprint(trainer):
    """Sum of AdamW exp_avg_sq entries -- zero only if the moments were never restored."""
    tot, seen = 0.0, 0
    for st in trainer.opt.state.values():
        v = st.get("exp_avg_sq")
        if v is not None:
            tot += float(v.double().sum())
            seen += 1
    return tot if seen else None


def dry_run(args):
    """Construct model + data + optimizer on MOCK data, run 3 steps through the REAL loop.

    The mock uses ``test_lens_dataset.build_mock`` / ``build_splits`` -- the real
    ``splits_v6_label_free_da`` schema, the real ``LensDataset``, the real balanced samplers,
    the real Zoobot encoder at n_blocks=3, the real ``Fp32CoralLoss`` and the real
    ``train_loop`` above. What it deliberately does NOT exercise is ``build_arm``'s
    phase-1-head gate (no mock head can have ||.||_2 = 66.268356) or the real 4.4 GB mount.
    Those are covered by ``--config-diff-only`` and by the path-resolution test respectively.
    """
    from lens_dataset import LensDataset, UNLABELED
    from live_encode import LiveEncoder
    from samplers import PairedDomainLoader, make_balanced_loader
    from test_lens_dataset import build_mock, build_splits
    from train_skeleton import TrainConfig, Trainer
    from zoobot_encoder import ZoobotEncoder

    dev = args.device or "cpu"
    root = tempfile.mkdtemp(prefix="coral_r4_dev8000_dry_")
    out_dir = args.out_dir or os.path.join(root, "out")
    print("=" * 92)
    print("DRY RUN -- mock data on the real schema, the real loop, 3 steps, then exit")
    print(f"  device   : {dev}   (nothing is trained on real data here)")
    print(f"  mock root: {root}")
    print(f"  out dir  : {out_dir}")
    print("=" * 92, flush=True)
    try:
        build_mock(root)
        splits = os.path.join(root, "splits.json")
        with open(splits, "w") as fh:
            json.dump(build_splits(), fh)

        src = LensDataset(splits, root, role="source_train", aug="weak", n_views=1)
        tgt = LensDataset(splits, root, role="target", fold=0, aug="weak", n_views=1)
        assert np.all(tgt.labels == UNLABELED), "mock target labels must be UNLABELED(-1)"
        bs, nsteps = 4, 3
        sl = make_balanced_loader(src, bs, use_latent=False, num_batches=nsteps,
                                  pos_fraction=0.5, seed=R4.SEED)
        tl = make_balanced_loader(tgt, bs, use_latent=True, num_batches=nsteps,
                                  pos_fraction=0.5, seed=R4.SEED + 1000)
        print(f"[dry] datasets: source {len(src)} rows, target {len(tgt)} rows; "
              f"batch {bs}, {nsteps} steps", flush=True)

        gc_on = (dev != "cpu")
        enc = ZoobotEncoder(encoder_mode="frozen", n_blocks=3, device=dev, verbose=False,
                            grad_checkpointing=gc_on)
        tcfg = TrainConfig(splits_path=splits, data_root=root, fold=0,
                           head_type=R4.HEAD_TYPE, encoder_mode="frozen", lr=R4.LR,
                           weight_decay=R4.WEIGHT_DECAY, batch_size=bs, pos_fraction=0.5,
                           use_coral=True, coral_weight=1.0,
                           coral_include_mean=R4.CORAL_INCLUDE_MEAN,
                           coral_shrinkage=R4.CORAL_SHRINKAGE, coral_target_ema=False,
                           coral_ema_momentum=R4.CORAL_EMA_MOMENTUM, total_steps=nsteps,
                           epochs=1, steps_per_epoch=nsteps, max_steps=nsteps,
                           seed=R4.SEED, device=dev, num_workers=0, amp=False, n_blocks=3,
                           backbone_lr=None, lr_decay=R4.LR_DECAY, init_head_from=None,
                           grad_checkpointing=gc_on)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = Trainer(tcfg, encoder=enc)
            coral = install_fp32_coral(trainer, include_mean=R4.CORAL_INCLUDE_MEAN,
                                       shrinkage=R4.CORAL_SHRINKAGE, target_ema=False,
                                       ema_momentum=R4.CORAL_EMA_MOMENTUM)
        trainer.standardizer.set_stats(torch.zeros(640), torch.ones(640))
        trainer.cfg.coral_weight = float(args.gamma)
        n_opt = sum(len(g["params"]) for g in trainer.opt.param_groups)
        print(f"[dry] optimizer: {type(trainer.opt).__name__}, "
              f"{len(trainer.opt.param_groups)} LR group(s), {n_opt} param tensor(s); "
              f"gamma={trainer.cfg.coral_weight}", flush=True)

        P = Paths(out_dir)
        live = LiveEncoder(splits, root, 0, batch_size=bs, num_workers=0, device=dev,
                           verbose=False)
        scorer = Scorer(trainer, live, 0, P.logits, fp16=False)

        # a synthetic reference so the ReproCheck plumbing is exercised end to end
        ref_csv = os.path.join(root, "ref_steps.csv")
        with open(ref_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=STEP_COLUMNS)
            w.writeheader()
            for s in range(1, nsteps + 1):
                w.writerow({"step": s, "sup_loss": 0.5, "da_loss": 0.1, "coral_raw": 0.1,
                            "cov_term": 0.05, "mean_term": 0.05, "src_feat_l2": 25.3,
                            "src_perdim_std": 1.0, "tgt_feat_l2": 25.3,
                            "tgt_perdim_std": 1.0, "step_skipped": 0.0, "amp_scale": 0.0})
        repro = ReproCheck(ref_csv, at_steps=(2,), window=2)

        # step 0: the true_init dump, the same call the real run's step-0 gate makes
        init_dst, init_secs, init_pos, init_neg = scorer.dump(0, name="true_init")
        print(f"[dry] step-0 true_init logits: {os.path.basename(init_dst)} "
              f"({init_pos.size} pos / {init_neg.size} neg, {init_secs:.1f}s)", flush=True)

        A = {"paired": PairedDomainLoader(sl, tl), "src_loader": sl, "tgt_loader": tl,
             "steps": nsteps, "epochs_planned": 1}
        record = {"gamma": float(args.gamma), "dry_run": True}
        cfg = {"max_steps": nsteps, "log_every": 1, "ckpt_every": 1,
               "gamma": float(args.gamma), "feat_norm_ceiling": 0.0, "feat_norm_floor": 0.0}
        train_loop(trainer=trainer, coral=coral, A=A, P=P, cfg=cfg, record=record,
                   scorer=scorer, repro=repro)
        diverged_ok = (repro.results and repro.results[0]["status"] == "DIVERGED")
        print(f"[dry] ReproCheck FAILURE path exercised (synthetic reference that cannot "
              f"match): {diverged_ok}", flush=True)

        # ---- KNOWN NULL (PROJECT.md 9): compare the run against ITSELF ----
        # A checker that never returns REPRODUCED would pass the test above by always
        # failing. Re-reading this run's own steps.csv and comparing step nsteps to itself
        # MUST come back REPRODUCED and bit-exact; if it does not, the checker is broken.
        self_ref = ReproCheck(P.steps_csv, at_steps=(nsteps,), window=2)
        with open(P.steps_csv, newline="") as fh:
            last = [r for r in csv.DictReader(fh)][-1]
        self_res = self_ref.compare(
            nsteps, {k: _num(last.get(k)) for k in ReproCheck.QUANTS},
            {k: _num(last.get(k)) for k in ReproCheck.QUANTS},
            _num(last.get("amp_scale")), float(record["fp16_skipped_cumulative"]))
        null_ok = (self_res["status"] == "REPRODUCED" and self_res["bit_exact"])
        print(f"[dry] ReproCheck KNOWN NULL (this run vs itself): "
              f"{self_res['status']}, bit_exact={self_res['bit_exact']} -> {null_ok}",
              flush=True)

        # ---- the resume round-trip: moments must come back, not just weights ----
        save_resume(trainer, P, gstep=nsteps, epoch=1, batch_in_epoch=nsteps,
                    rng_epoch_start=_rng_state(), record=record)
        moments_before = _moment_fingerprint(trainer)
        with contextlib.redirect_stdout(io.StringIO()):
            enc2 = ZoobotEncoder(encoder_mode="frozen", n_blocks=3, device=dev,
                                 verbose=False, grad_checkpointing=gc_on)
            t2 = Trainer(tcfg, encoder=enc2)
            install_fp32_coral(t2, include_mean=R4.CORAL_INCLUDE_MEAN,
                               shrinkage=R4.CORAL_SHRINKAGE, target_ema=False,
                               ema_momentum=R4.CORAL_EMA_MOMENTUM)
        sch = load_resume(t2, P)
        moments_after = _moment_fingerprint(t2)
        same_w = all(torch.equal(a.cpu(), b.cpu()) for a, b in
                     zip(trainer.head.state_dict().values(), t2.head.state_dict().values()))
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
        print(f"[dry]   AdamW moment fingerprint restored        : {ok_m} "
              f"({moments_before} -> {moments_after})")
        print(f"[dry]   RNG state present in the payload         : "
              f"{bool(sch.get('rng_epoch_start')) and bool(sch.get('rng_now'))}")
        print("-" * 92, flush=True)

        # ---- what landed on disk ----
        wrote = sorted(os.path.relpath(p, P.out).replace("\\", "/")
                       for p in glob.glob(os.path.join(P.out, "**", "*"), recursive=True)
                       if os.path.isfile(p))
        print("[dry] files written:")
        for f in wrote:
            print(f"        {f}")
        z = np.load(os.path.join(P.logits, LOGIT_ARM, f"step_{nsteps}.npz"))
        ids = np.load(os.path.join(P.logits, "ids.npz"), allow_pickle=True)
        init = np.load(os.path.join(P.logits, "true_init.npz"))
        print(f"[dry] logit npz keys {sorted(z.files)}  pos {z['pos'].shape} "
              f"neg {z['neg'].shape}  dtype {z['pos'].dtype}")
        print(f"[dry] ids.npz keys {sorted(ids.files)}  pos_ids {ids['pos_ids'].shape} "
              f"{ids['pos_ids'].dtype}  neg_ids {ids['neg_ids'].shape}")
        ok = (set(z.files) == {"pos", "neg"} and set(ids.files) == {"pos_ids", "neg_ids"}
              and set(init.files) == {"pos", "neg"}
              and z["pos"].dtype == np.float64 and same_w and same_enc and ok_m
              and record["steps_completed"] == nsteps and not record["aborted"]
              and os.path.isfile(P.ckpt_csv) and os.path.isfile(P.repro_json)
              and diverged_ok and null_ok)
        print("\n" + ("[dry] DRY RUN PASSED -- model, data, optimizer, loop, logit "
                      "persistence and resume all exercised."
                      if ok else "[dry] DRY RUN FAILED -- see above."), flush=True)
        return 0 if ok else 1
    finally:
        if not args.keep_dry_run:
            shutil.rmtree(root, ignore_errors=True)


# ===========================================================================
# main
# ===========================================================================

def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=None, help=f"default: <working>/runs/{RUN_NAME}")
    p.add_argument("--input-root", default=None,
                   help="where to search for the dataset (default: /kaggle/input, then the "
                        "repo root)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--init-head-from", default=None)
    p.add_argument("--agreement-ref", default=None)
    p.add_argument("--reference-dir", default=None,
                   help="directory holding the stored g16 arm's sweep.json + steps.csv")
    p.add_argument("--reference-ids", default=None,
                   help="runs/coral_r4_logits/ids.npz, for the id-order check")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--fold", type=int, default=FOLD)
    p.add_argument("--max-steps", type=int, default=DEV_STEPS)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    p.add_argument("--repro-steps", default=",".join(str(s) for s in REPRO_STEPS))
    p.add_argument("--repro-window", type=int, default=REPRO_WINDOW)
    p.add_argument("--num-workers", type=int, default=2,
                   help="MUST be 2 to reproduce the stored arm: the worker count sets the "
                        "augmentation RNG stream")
    p.add_argument("--feat-norm-ceiling", type=float, default=DEFAULT_FEAT_NORM_CEILING)
    p.add_argument("--feat-norm-floor", type=float, default=DEFAULT_FEAT_NORM_FLOOR)
    p.add_argument("--coral-target-ema", dest="coral_target_ema", action="store_true",
                   default=False, help="OFF in round 4; PINNED -- see the design record")
    p.add_argument("--device", default=None)
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--eval-fp16", action="store_true", default=False,
                   help="scoring precision; leave OFF -- runs/coral_r4_logits is fp32")
    p.add_argument("--allow-config-drift", action="store_true", default=False,
                   help="start even though the config diff found a mismatch (recorded)")
    p.add_argument("--no-resume", action="store_true", default=False,
                   help="ignore resume/latest.pt and refuse to start over an existing run")
    p.add_argument("--config-diff-only", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--keep-dry-run", action="store_true", default=False)
    return p


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    args = build_argparser().parse_args(argv)

    if args.dry_run:
        return dry_run(args)

    out_dir = args.out_dir or os.path.join(
        "/kaggle/working" if os.path.isdir("/kaggle/working") else root, "runs", RUN_NAME)
    P = Paths(out_dir)
    paths = resolve_paths(args, root)

    print("=" * 92)
    print(f"CORAL 7.14 FOLD-0 DEVELOPMENT RUN -- {ARM_KEY}, gamma={args.gamma:g}, "
          f"fold {args.fold}, {args.max_steps} steps, FRESH FROM STEP 0")
    print("=" * 92)
    for k in ("data_root", "splits", "cache_dir", "init_head_from", "reference_dir",
              "reference_ids"):
        print(f"  {k:<16}: {paths.get(k)}")
    print(f"  {'out_dir':<16}: {P.out}")
    print(f"  {'torch':<16}: {torch.__version__}  cuda={torch.cuda.is_available()}")
    print("=" * 92, flush=True)

    resume_sched = None
    existing = os.path.isfile(P.steps_csv) or os.path.isfile(P.resume_pt)
    if existing and args.no_resume:
        raise SystemExit(
            f"[dev8000] ABORT: {P.out} already holds a run and --no-resume was passed. "
            f"Move it aside or point --out-dir elsewhere -- appending would interleave two "
            f"runs' rows into one series.")

    # ---- build the arm through the SAME two lines the sweep used ----
    sa = argparse.Namespace(
        arm=ARM_KEY, fold=int(args.fold), gamma=float(args.gamma),
        data_root=paths["data_root"], splits=paths["splits"], cache_dir=paths["cache_dir"],
        agreement_ref=(paths.get("agreement_ref") or R4.DEFAULT_AGREEMENT_REF),
        init_head_from=paths["init_head_from"],
        max_steps=int(args.max_steps), allow_off_budget=True,
        curve_every=int(args.log_every), ckpt_every=int(args.ckpt_every),
        backbone_lr=R4.BACKBONE_LR, fp16=bool(args.fp16), eval_fp16=bool(args.eval_fp16),
        grad_checkpointing=True, coral_target_ema=bool(args.coral_target_ema),
        num_workers=int(args.num_workers), device=args.device,
        target_fpr=1e-3, step0_tol=0.0, smoke_batch_size=None, dir_suffix=None,
        dry_run=False, emit_config=None, reference_config=None,
        calibrate=False, calibrate_steps=0, calibrate_tail=0, grad_check=False,
        feat_norm_ceiling=float(args.feat_norm_ceiling),
        feat_norm_floor=float(args.feat_norm_floor))
    A = R4.build_arm(ARM_KEY, sa, root, for_calibration=True)
    trainer, coral = A["trainer"], A["coral"]
    trainer.cfg.coral_weight = float(args.gamma)
    assert float(trainer.cfg.coral_weight) == float(args.gamma)
    assert coral.target_ema == bool(args.coral_target_ema), \
        "the CORAL module's target_ema does not match the requested setting"

    record = build_record(A, args, paths)
    record.update({
        "purpose": ("PROJECT.md 7.14 fold-0 DEVELOPMENT run for CORAL: fix CORAL's own "
                    "budget from its own fold-0 curve, before any confirmatory fold exists. "
                    "Fold 0 is development; no number here enters a headline."),
        "gamma_provenance": ("runs/coral_r4_gamma_selection.json, branch C -- the largest "
                             "swept value with measured non-collapsing, strictly-aligning "
                             "behaviour; hashed before any confirmatory fold exists"),
        "budget_provenance": ("runs/devmax_truncation.json: 8000 is the only development "
                              "cap at which FreeMatch's decay was detectable; every cap "
                              "1500-6000 was not distinguishable"),
        "continuation": ("NONE. This is a FRESH trajectory from step 0. 7.11 documented "
                         "three silent restarts (threshold buffers, schedule position, "
                         "AdamW moments) when continuing from a weights-only checkpoint; "
                         "this run does not continue from anything."),
        "metric_hygiene": ("the ONLY test-set metric computed is the step-0 known-null "
                           "gate; every training segment runs inside test_label_lockout(); "
                           "checkpoints are SCORED (logits persisted) but never MEASURED"),
        "single_gpu": ("no DataParallel / DistributedDataParallel / device sharding "
                       "anywhere in this path -- halving the per-device batch would change "
                       "CORAL's 640x640 covariance estimate from 256 draws to 128 (7.1a)"),
        "paths": paths,
        "out_dir": P.out,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "script_sha256_16": {f: _sha256(os.path.join(_HERE, f))
                             for f in SCRIPT_HASH_FILES
                             if os.path.isfile(os.path.join(_HERE, f))},
        "script_delta_vs_reference_run": (
            "the g16 arm ran the bundle-v4 copies of coral_round4.py / run_coral_round4.py "
            "/ run_coral_gamma_sweep.py. The ONLY difference in the current copies is the "
            "two-sided feature-norm FLOOR (added 2026-08-19): a new abort condition, no "
            "change to any gradient, update or loss term. Every other module in the "
            "training path is byte-identical."),
        "steps_completed": 0, "aborted": False, "abort_reason": None,
        "fp16_skipped_cumulative": 0.0, "resume_events": [],
    })

    # ---- the config diff, before anything expensive ----
    ref_cfg = (os.path.join(paths["reference_dir"], "sweep.json")
               if paths.get("reference_dir") else None)
    rows, mismatches, _ = config_diff(record, ref_cfg, verbose=True)
    with open(P.diff_json, "w") as fh:
        json.dump({"reference_config": ref_cfg, "rows": rows, "mismatches": mismatches,
                   "expected_differences": {k: {"reference": v[0], "this_run": v[1],
                                                "why": v[2]}
                                            for k, v in EXPECTED_DIFFS.items()}},
                  fh, indent=2, default=str)
    record["config_diff"] = {"reference_config": ref_cfg, "mismatches": mismatches,
                             "allowed_drift": bool(args.allow_config_drift)}
    if ref_cfg is None:
        banner("NO REFERENCE CONFIG FOUND",
               ["The stored 2000-step gamma=16 arm's sweep.json was not located, so the "
                "config could not be diffed and the reproduction check has nothing to "
                "compare against.",
                "Pass --reference-dir. The run continues, but it is UNVERIFIED that it "
                "matches the arm it is supposed to extend."], "?")
    if mismatches and not args.allow_config_drift:
        raise SystemExit(
            f"[dev8000] ABORT: the config differs from the stored arm on {mismatches}.\n"
            f"  This run's whole claim is that it IS that arm, extended to 8000 steps -- a "
            f"mismatch means it is not, and its first 2000 steps would not be comparable to "
            f"anything already analysed. See {P.diff_json}.\n"
            f"  Fix the mismatch, or pass --allow-config-drift to start anyway (recorded in "
            f"run.json as a deliberate deviation).")
    if args.config_diff_only:
        print("[dev8000] --config-diff-only: diff emitted, nothing trained.")
        with open(P.run_json, "w") as fh:
            json.dump(record, fh, indent=2, default=str)
        return 0

    # ---- scorer + reproduction check ----
    from live_encode import LiveEncoder
    live = LiveEncoder(paths["splits"], paths["data_root"], int(args.fold),
                       batch_size=A["batch_size"], num_workers=0, device=A["device"],
                       verbose=True)
    scorer = Scorer(trainer, live, int(args.fold), P.logits, fp16=bool(args.eval_fp16))
    ref_steps = (os.path.join(paths["reference_dir"], "steps.csv")
                 if paths.get("reference_dir") else None)
    repro = ReproCheck(ref_steps, at_steps=[int(s) for s in args.repro_steps.split(",")],
                       window=int(args.repro_window))
    record["repro_reference"] = {"steps_csv": ref_steps, "available": repro.available,
                                 "at_steps": list(repro.at)}
    if not repro.available:
        banner("NO REPRODUCTION REFERENCE",
               [f"{ref_steps} was not found, so steps {repro.at} cannot be compared against "
                f"the stored arm. The run continues and everything is logged; the "
                f"comparison can be done locally afterwards from steps.csv."], "?")

    # ---- resume, or step-0 gate on a fresh run ----
    ids_path = os.path.join(P.logits, "ids.npz")
    if existing and not args.no_resume and os.path.isfile(P.resume_pt):
        resume_sched = load_resume(trainer, P)
        record["resume_events"].append(
            {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "resumed_at_step": int((resume_sched or {}).get("global_step", 0)),
             "saved_utc": (resume_sched or {}).get("saved_utc")})
        if os.path.isfile(ids_path):
            z = np.load(ids_path, allow_pickle=True)
            scorer.ids = (z["pos_ids"], z["neg_ids"])
        banner("RESUMED",
               [f"restored from {P.resume_pt} at step "
                f"{(resume_sched or {}).get('global_step')}",
                "weights, AdamW moments, GradScaler state, CORAL buffers and RNG restored.",
                "Sample ORDER is exact; augmentation offsets after a resume are not "
                "guaranteed to equal an uninterrupted run's. Recorded in run.json."], "=")
    else:
        step0_gate(scorer, record, int(args.fold),
                   tol_warn=R4.ONE_POSITIVE_TOL, tol_abort=0.05,
                   auroc_tol_warn=R4.STEP0_AUROC_TOL, auroc_tol_abort=0.01)
        record["ids_check"] = check_ids_against_reference(ids_path,
                                                          paths.get("reference_ids"))

    with open(P.run_json, "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    # ---- train ----
    cfg = {"max_steps": int(args.max_steps), "log_every": int(args.log_every),
           "ckpt_every": int(args.ckpt_every), "gamma": float(args.gamma),
           "feat_norm_ceiling": float(args.feat_norm_ceiling),
           "feat_norm_floor": float(args.feat_norm_floor)}
    print(f"\n[train] gamma={args.gamma:g} nb3 fold {args.fold}: {cfg['max_steps']} steps, "
          f"checkpoint+logits every {cfg['ckpt_every']}, trace every {cfg['log_every']}\n",
          flush=True)
    record = train_loop(trainer=trainer, coral=coral, A=A, P=P, cfg=cfg, record=record,
                        scorer=scorer, repro=repro, resume_sched=resume_sched)
    record["repro_results"] = repro.results
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(P.run_json, "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    peak = record.get("peak_gpu_gb_allocated")
    print("\n" + "=" * 92)
    print(f"[done] steps {record['steps_completed']}/{cfg['max_steps']}  "
          f"aborted={record['aborted']}  {record['wall_minutes']:.1f} min  "
          f"peak {peak if peak is None else round(peak, 2)} GB allocated")
    for r in repro.results:
        print(f"[repro] step {r['step']}: {r['status']}")
    print(f"[out] {P.out}")
    print("[next] download the run directory, drop it into runs/ locally, and read the "
          "budget off the persisted logits -- CPU work, no GPU session time.")
    print("=" * 92, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
