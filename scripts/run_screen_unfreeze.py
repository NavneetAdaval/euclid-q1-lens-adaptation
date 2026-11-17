"""
run_screen_unfreeze.py
======================
LEVER 1 fold-0 SCREEN: does partially unfreezing the Zoobot encoder (LP->FT) beat the
frozen encoder on Baseline A, at fold 0, under the matched 8000-step protocol?

THE FOUR RUNS (names are deliberate -- the old A0/A1/A3 labels collided with "Baseline A",
the R1/R2/R3 selection rules, and fold indices):

  baselineA_fold0_nb0_cached   round-1 frozen cached-feature run  (ALREADY EXISTS; reused)
  baselineA_fold0_nb0_image    frozen, image-space path           <- THE CONTROL
  baselineA_fold0_nb1          n_blocks=1, LP->FT
  baselineA_fold0_nb3          n_blocks=3, LP->FT (Lines et al.'s exact setting)

There is no nb2 / nb4 / nb5 run: n_blocks=2 is within ~480k params of n_blocks=3
(14,355,840 vs 14,835,520, ~3% apart -- not separable on ~98 fold positives), and 4/5 add
only 113k params over 3. The informative rungs are 0, 1, 3.

****  THE CONTROL IS NOT OPTIONAL  ****
baselineA_fold0_nb0_image must reproduce baselineA_fold0_nb0_cached's fold-0 TPR@FPR=1e-3
within the paired-bootstrap CI. If it does not, the image-space code path is wrong and
every nb1/nb3 number is uninterpretable -- this script REFUSES to run nb1/nb3 until the
control passes. There is NO bypass flag by design. On a fresh Kaggle session with no local
run history, satisfy the gate with --control-scores pointing at the uploaded dumps
(which must carry pos_ids/neg_ids, or the gate fails closed on row alignment).

RESULT (Task 2, both controls complete): the ISOLATING control -- aug=none + fp32, i.e.
matched to the cached run in both respects the protocol run differs in -- reproduced it
EXACTLY: TPR 0.193878 vs 0.193878, AUROC 0.942761 vs 0.942692, paired diff +0.0000
CI [-0.031, +0.031]. The image-space code path is verified. The protocol run (weak
aug + fp16) sits at 0.122449, and that -0.0714 gap is a REAL effect of augmentation
and/or precision (CI [-0.1327, -0.0102], excludes 0) -- NOT a code defect. The split
between augmentation and precision is [UNVERIFIED]; the --diagnostic decomposition run
(weak aug + fp32) separates them.

WHY THE TWO SHOULD AGREE BUT NEED NOT BE IDENTICAL. Both train a linear head on frozen
640-d Zoobot features with the same optimizer/budget/seed. Two REAL differences remain,
both stated rather than hidden:
  (a) AUGMENTATION. The cached run reads aug="none" features (cache_features.py caches the
      deterministic path); the image-space run applies the +-3px weak augmentation
      (PROJECT.md 7.3) because that is what an unfrozen run must do. This is a genuine
      protocol difference between the two controls and is the main reason to expect
      agreement-within-CI rather than equality.
  (b) fp16 vs fp32. The image-space path autocasts the forwards under --fp16.
The frozen ENCODER itself is provably identical: live_encode.py reproduces the feature
cache byte-for-byte at n_blocks=0 (verified), so the features are not a source of drift.

LP->FT (nb1/nb3 only). Phase 1 is already done -- the round-1 frozen run IS the linear
probe -- so the head is initialised from its step_8000 checkpoint rather than randomly
(Kumar et al. arXiv:2202.10054: fine-tuning from a random head distorts pretrained features
and underperforms OOD; LP-FT recovers ~10% OOD over full FT). nb0_image is NOT LP->FT
initialised: it is the control for the from-scratch cached run and must match its setup.

EVAL PATH. At n_blocks=0 the feature cache is valid and is used. At n_blocks>0 the cache is
stale by construction, so every curve point re-encodes val + test_pos_fold0 +
test_negatives_shared + target_fold0 LIVE with the current encoder (live_encode.py) and
feeds them to the SAME curve_hook code via its split_loader hook -- so curve.csv keeps the
identical schema and analyze_fold0_selection.py / collapse_gate.py consume it unchanged.

Run:
    python scripts/run_screen_unfreeze.py --run nb0_image --fp16
    python scripts/run_screen_unfreeze.py --run nb1 --fp16
    python scripts/run_screen_unfreeze.py --run nb3 --fp16 --grad-checkpointing
    python scripts/run_screen_unfreeze.py --run nb1 --dry-run
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

import curve_hook                                          # noqa: E402
import evaluate as ev                                      # noqa: E402
from heads import LINEAR                                   # noqa: E402
from lens_dataset import LensDataset                       # noqa: E402
from live_encode import LiveEncoder                        # noqa: E402
from metrics import (tpr_at_fpr, metric_suite, paired_bootstrap, default_tpr_metric,
                     DEFAULT_TARGET_FPR)                   # noqa: E402
from samplers import make_balanced_loader                  # noqa: E402
from train_skeleton import (Trainer, TrainConfig, ZOOBOT_LR, ZOOBOT_LR_DECAY,
                            ZOOBOT_WEIGHT_DECAY,
                            zoobot_protocol_provenance)    # noqa: E402
from run_baseline import _cf                               # noqa: E402

# ---- PRE-REGISTERED matched settings (identical to round 1 where applicable) ----
BUDGET_STEPS = 8000
CURVE_EVERY = 250          # 32 curve points, same as round 1
BATCH_SIZE = 256           # matched (RLSbench); TRUE batch -- never micro-batched
# ---- C1/C2 PROTOCOL MATCH: Zoobot v2.0.1 published defaults, ADOPTED WITHOUT TUNING ----
# Was: LR=1e-3 (head) with backbone_lr defaulting to LR/10 = 1e-4, WEIGHT_DECAY=1e-4.
# Now: head AND top block at 1e-4 (Zoobot creates the head group with `lr` and gives block
# i=0 `lr * lr_decay**0` -- the same value), weight_decay 0.05, cosine_schedule off.
# See train_skeleton.ZOOBOT_PROTOCOL_RATIONALE for why these are adopted, not swept.
LR = ZOOBOT_LR                     # 1e-4, head AND (via lr_decay**0) the top block
WEIGHT_DECAY = ZOOBOT_WEIGHT_DECAY  # 0.05
POS_FRACTION = 0.5
SEED = 42
HEAD_TYPE = LINEAR
FOLD = 0
LR_DECAY = ZOOBOT_LR_DECAY         # 0.75, Zoobot default (unchanged)

# round-1 frozen cached run: the control reference AND the LP->FT phase-1 head
ROUND1_DIR = os.path.normpath(os.path.join(_HERE, "..", "runs", "bakeoff_baseline_A"))
ROUND1_CKPT = os.path.join(ROUND1_DIR, "checkpoints", f"step_{BUDGET_STEPS}.pt")


def round1_ckpt(args=None):
    """Path to the round-1 Baseline A checkpoint, overridable with --round1-ckpt.

    Every rung needs this file -- lp_ft rungs load it as the LP->FT phase-1 head
    (Kumar arXiv:2202.10054), and ALL rungs score it to build the agreement column's
    reference vector. The default location is inside the repo's runs/ tree, which is
    convenient locally and wrong anywhere else: data.zip does not pack runs/, so on a
    fresh Kaggle session the file is wherever the operator's dataset put it. Hence the
    flag -- point it straight at the file, no copying and no directory scaffolding.

    Nothing here is pinned by the §1 check, so overriding the path cannot affect whether
    nb0/nb3 count as matched. The IDENTITY of the file still matters (the agreement column
    is only comparable across runs if every run uses the same reference), which is what
    --round1-sha256 is for.
    """
    p = getattr(args, "round1_ckpt", None) if args is not None else None
    return os.path.abspath(p) if p else ROUND1_CKPT

RUNS = {
    "nb0_image": {"n_blocks": 0, "lp_ft": False, "dir": "baselineA_fold0_nb0_image"},
    "nb1":       {"n_blocks": 1, "lp_ft": True,  "dir": "baselineA_fold0_nb1"},
    "nb3":       {"n_blocks": 3, "lp_ft": True,  "dir": "baselineA_fold0_nb3"},
}

# ---------------------------------------------------------------------------
# THE AMENDED DECISION RULE -- recorded BEFORE any screen result existed.
# ---------------------------------------------------------------------------
DECISION_RULE = {
    "rule": ("Proceed to a full 5-fold commit only if the winning rung beats "
             "baselineA_fold0_nb0_image on fold-0 TPR@FPR=1e-3 by MORE than the fold-0 "
             "paired-bootstrap CI half-width. On a tie between nb1 and nb3, prefer nb3."),
    "amended_from": ("Original rule preferred nb1 on a tie, justified half on compute cost "
                     "and half on 'partial unfreeze' being a cleaner claim."),
    "amendment_rationale": (
        "Compute is no longer the binding constraint (Kaggle/Colab available), so the cost "
        "leg of the original tie-break is gone. Protocol match then wins: n_blocks=3 is "
        "Lines et al.'s exact setting, and matching it is what makes our Baseline A "
        "comparable to their published simulation-only result -- which is the entire "
        "purpose of this lever."),
    "prerequisite": ("An nb3 commit requires gradient checkpointing implemented and tested "
                     "(numerically exact; test 8 in test_encoder_unfreeze.py)."),
    "preregistered": True,
    "recorded_before_any_screen_result_existed": True,
    "frozen": "Do not modify this rule after seeing numbers.",
}


def _steps_per_epoch(n_src):
    return int(math.ceil(n_src / BATCH_SIZE))


# ---------------------------------------------------------------------------
# §1 -- CONFOUND PINNING. Only n_blocks (and its memory knob) may differ.
# ---------------------------------------------------------------------------

# Everything here must MATCH the reference nb0_image run, or the screen is measuring
# something other than the unfreeze. Task 2 MEASURED augmentation+precision at -0.0714 TPR
# on this exact metric -- the same order as the effect being screened -- so this is an
# assertion, not a convention.
PINNED_KEYS = ("augmentation", "precision", "seed", "batch_size", "budget_steps",
               "lr", "backbone_lr", "lr_decay", "weight_decay", "pos_fraction",
               "head_type", "fold", "curve_every", "true_batch",
               "gradient_accumulation")
# Exempt: n_blocks is THE variable under test; grad_checkpointing is a pure
# memory<->recompute trade that is numerically exact (test 8) and changes no result.
EXEMPT_KEYS = ("n_blocks", "grad_checkpointing")


def assert_no_bf16(precision, device):
    """T4 is sm_75 and has NO bfloat16 support -- fail loudly rather than downgrade."""
    if str(precision).lower() in ("bf16", "bfloat16"):
        raise SystemExit(
            "[screen] ABORT: precision resolved to bfloat16. T4 (sm_75) has no bf16 "
            "support, and the screen's precision is pinned to fp16/fp32. Refusing to run "
            "-- a silent precision downgrade is indistinguishable from a real effect "
            "(measured: precision+aug is worth -0.0714 TPR here).")
    if device == "cuda" and torch.cuda.is_available():
        major = torch.cuda.get_device_properties(0).major
        if str(precision) == "bf16" and major < 8:
            raise SystemExit("[screen] ABORT: bf16 requested on a pre-Ampere GPU.")


def _normalise_cfg(cfg):
    """Fill keys that older run_configs recorded under a different name.

    ``precision`` was added in Task 3; the completed Task-2 runs recorded the same fact as
    the boolean ``fp16``. Deriving one from the other is a faithful restatement of a value
    ALREADY IN THE FILE -- it is not a relaxation of the §1 check, and it is the correct fix
    rather than dropping ``precision`` from PINNED_KEYS (which would stop pinning the very
    thing Task 2 measured at -0.0714 TPR).
    """
    cfg = dict(cfg)
    if "precision" not in cfg or cfg.get("precision") in (None, "<absent>"):
        if "fp16" in cfg:
            cfg["precision"] = "fp16" if cfg["fp16"] else "fp32"
    return cfg


def check_preconditions(cfg_json, reference_config, verbose=True):
    """REFUSE to start unless every pinned key matches the reference nb0_image run.

    Returns the comparison table (list of rows). Raises SystemExit on any mismatch.
    """
    if not os.path.isfile(reference_config):
        raise SystemExit(
            f"[screen] ABORT: reference config not found: {reference_config}\n"
            f"  The screen cannot verify that only n_blocks differs, so it will not start. "
            f"Pass --reference-config pointing at the completed nb0_image run's "
            f"run_config.json (upload it with the Kaggle dataset).")
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
        print("\n" + "=" * 84)
        print("§1 PRECONDITION CHECK -- only n_blocks may differ from the reference run")
        print(f"  reference: {reference_config}")
        print("=" * 84)
        print(f"  {'key':<22} | {'reference':<26} | {'this run':<20} | status")
        print("  " + "-" * 80)
        for k, rv, cv, ok in rows:
            status = "EXEMPT" if ok is None else ("ok" if ok else "*** MISMATCH ***")
            print(f"  {k:<22} | {str(rv):<26} | {str(cv):<20} | {status}")
        print("  " + "-" * 80)
    if bad:
        raise SystemExit(
            f"\n[screen] ABORT: {len(bad)} pinned setting(s) differ from the reference "
            f"run: {bad}.\n"
            f"  The screen compares nb0_image / nb1 / nb3 and ONLY n_blocks may vary. "
            f"Task 2 measured augmentation+precision at -0.0714 TPR on this metric, which "
            f"is comparable to the effect being screened, so a mismatch here would not be "
            f"a nuisance -- it would invalidate the comparison.")
    if verbose:
        print("  ALL PINNED SETTINGS MATCH -- the screen is measuring n_blocks alone.\n")
    return rows


def expected_checkpoint_steps(budget, curve_every):
    """The steps at which a checkpoint MUST exist: one per curve point."""
    return [s for s in range(int(curve_every), int(budget) + 1, int(curve_every))]


def assert_checkpoint_retention(ckpt_dir, curve_csv, budget, curve_every, verbose=True):
    """FAIL LOUDLY unless a checkpoint survives for EVERY curve point.

    WHY THIS EXISTS (Task 5). ``nb1``, ``nb3`` and ``nb0_image_kaggle`` came home with only
    ``step_8000.pt``, which made IM/SND/ATC uncomputable on exactly the two runs the A1
    verdict rested on. The trainer was never at fault -- it has always written a checkpoint
    at every curve point, and the local runs have all 32. The loss happened in the Kaggle
    COLLECTION cell, which copied only the R1 checkpoint into the downloadable directory.

    So this guard is deliberately placed at the END OF TRAINING, where it proves the files
    exist on the training box, and it is paired with a second retention check inside the
    packaging step (``package_screen_results.py``) which proves they survived into the
    archives. One check alone would not have caught the failure that actually occurred.

    Raises SystemExit on any gap; returns the list of steps found.
    """
    want = expected_checkpoint_steps(budget, curve_every)
    have = sorted(int(m.group(1)) for m in
                  (re.fullmatch(r"step_(\d+)\.pt", f)
                   for f in os.listdir(ckpt_dir) if f.endswith(".pt")) if m)
    missing = [s for s in want if s not in set(have)]

    n_curve = 0
    if os.path.isfile(curve_csv):
        with open(curve_csv, newline="") as fh:
            n_curve = max(sum(1 for _ in csv.reader(fh)) - 1, 0)

    if verbose:
        print(f"[retention] checkpoints on disk: {len(have)}/{len(want)}  "
              f"(curve rows: {n_curve})")
    if missing:
        raise SystemExit(
            f"[screen] ABORT: checkpoint retention failed -- {len(missing)} of "
            f"{len(want)} curve-point checkpoints are missing from {ckpt_dir}: "
            f"{missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  Every curve point needs its checkpoint or the label-free validator study "
            f"(IM/SND/ATC) cannot be run on this run -- that is exactly what went wrong "
            f"with nb1/nb3 in Task 4.")
    if n_curve and n_curve != len(want):
        raise SystemExit(
            f"[screen] ABORT: {n_curve} curve rows but {len(want)} expected checkpoint "
            f"steps -- curve.csv and the checkpoint set disagree, so a validator series "
            f"could not be aligned to the TPR series.")
    return have


def _make_loader_with_fallback(src_ds, steps, want_workers):
    """Build the source loader, degrading num_workers if Windows spawn cannot pickle it.

    MEASURED failure mode (twice, non-deterministically): on Windows the DataLoader spawns
    workers and PICKLES the dataset to each child. source_train holds ~48k 64x64 float32
    images (~790 MB), so 4 workers means ~3 GB of simultaneous pickle traffic. When free RAM
    is tight this dies with "OSError: [Errno 22] Invalid argument" in ForkingPickler.dump
    and "UnpicklingError: pickle data was truncated" in the child. It is RAM-dependent, so
    it succeeded once and failed later on the SAME configuration -- i.e. flaky, not a clean
    pass/fail. With several multi-hour runs queued, a silent mid-run crash is the worst
    outcome, so we degrade deterministically instead of hoping.

    num_workers only affects THROUGHPUT, never the trained model: the sampler is seeded and
    yields the same index batches regardless of how many processes materialise them. So
    falling back is a wall-clock cost, not a protocol change.
    """
    order = [w for w in (want_workers, max(want_workers // 2, 1), 0)
             if w <= want_workers]
    seen, tries = set(), []
    for w in order:
        if w in seen:
            continue
        seen.add(w)
        tries.append(w)
    last = None
    for w in tries:
        loader = make_balanced_loader(
            src_ds, BATCH_SIZE, use_latent=False, num_batches=steps,
            pos_fraction=POS_FRACTION, seed=SEED, num_workers=w,
            pin_memory=True, persistent_workers=bool(w))
        try:
            it = iter(loader)          # this is where spawn actually happens
            next(it)                   # and this is where the pickle completes
            del it
            if w != want_workers:
                print(f"[loader] NOTE: num_workers={want_workers} failed to spawn; using "
                      f"{w}. Throughput only -- the seeded sampler yields identical "
                      f"batches, so the trained model is unaffected.", flush=True)
            else:
                print(f"[loader] num_workers={w}", flush=True)
            return loader, w
        except (OSError, EOFError, RuntimeError) as e:
            last = e
            print(f"[loader] num_workers={w} failed to spawn ({type(e).__name__}: "
                  f"{str(e)[:90]}); retrying with fewer workers ...", flush=True)
            del loader
    raise RuntimeError(f"could not build the source loader at any worker count: {last}")


def build_split_loader(n_blocks, encoder, live: LiveEncoder | None, cache_dir, fp16):
    """Return the curve_hook split_loader for this rung (None -> use the frozen cache)."""
    if n_blocks == 0:
        return None                    # cache is VALID and byte-identical (verified)
    cacheable = {}

    def loader(name):
        # re-encode live with the CURRENT encoder; memoised per curve point by the caller
        if name not in cacheable:
            cacheable[name] = live.encode(encoder, name, fp16=fp16)
        return cacheable[name]

    loader.reset = cacheable.clear       # type: ignore[attr-defined]
    return loader


def train_one(run_key, args):
    spec = RUNS[run_key]
    n_blocks = spec["n_blocks"]
    root = os.path.normpath(os.path.join(_HERE, ".."))
    budget = int(getattr(args, "budget_steps", BUDGET_STEPS))
    suffix = getattr(args, "dir_suffix", "") or ""
    if budget != BUDGET_STEPS and not suffix:
        suffix = f"_SMOKE{budget}"      # never let a short run occupy the real run's dir
    out_dir = os.path.abspath(os.path.join(root, "runs", spec["dir"] + suffix))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    curve_csv = os.path.join(out_dir, "curve.csv")
    r1_ckpt = os.path.join(ckpt_dir, f"step_{budget}.pt")
    round1_path = round1_ckpt(args)   # round-1 Baseline A -- NOT this run's output
    splits_path = os.path.abspath(args.splits)
    data_root = os.path.abspath(args.data_root)
    cache_dir = os.path.abspath(args.cache_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # checkpointing is a no-op at n_blocks=0 and only NEEDED at n_blocks=3 (measured: at
    # true batch 256 nb3 peaks 6.31 GB without it -> host-memory spill on a 6 GB card;
    # 2.02 GB with it). It is numerically exact, so enabling it changes cost, not results.
    grad_ckpt = bool(args.grad_checkpointing)

    cfg_json = {
        "run": run_key, "run_dir": spec["dir"], "n_blocks": n_blocks,
        "lp_ft": spec["lp_ft"],
        "init_head_from": (round1_path if spec["lp_ft"] else None),
        "budget_steps": budget, "is_preregistered_budget": (budget == BUDGET_STEPS),
        "curve_every": args.curve_every,
        "batch_size": BATCH_SIZE, "true_batch": True, "gradient_accumulation": False,
        "grad_checkpointing": grad_ckpt, "lr": LR, "backbone_lr": args.backbone_lr,
        # what the optimizer ACTUALLY uses once the None sentinel is resolved: under the
        # Zoobot protocol match that is the head LR itself, not LR/10.
        "backbone_lr_effective": (args.backbone_lr if args.backbone_lr is not None else LR),
        "lr_decay": LR_DECAY, "weight_decay": WEIGHT_DECAY,
        "pos_fraction": POS_FRACTION, "seed": SEED, "head_type": HEAD_TYPE,
        "fold": FOLD, "selection_rule": "R1_last_checkpoint_step8000",
        "method": "baseline_A_source_only", "domain_adaptation": None,
        "encoder_eval": ("frozen_feature_cache" if n_blocks == 0 else "live_encode"),
        "augmentation": (f"{args.aug} (+-3px translate, PROJECT.md 7.3)"
                         if args.aug == "weak" else
                         "none (matches the round-1 cached run; code-path control only)"),
        "fp16": bool(args.fp16),
        # resolved precision as a STRING -- what §1 pins and what the bf16 guard checks.
        "precision": ("fp16" if args.fp16 else "fp32"),
        "device": device,
        "control_reference": round1_path,
        "decision_rule": DECISION_RULE,
        # C2 deliverable 2: the write-up must be able to state that these hyperparameters
        # were ADOPTED from the published benchmark, not tuned by us.
        **zoobot_protocol_provenance(),
    }
    assert_no_bf16(cfg_json["precision"], device)

    print("=" * 80)
    print(f"LEVER 1 SCREEN -- {spec['dir'] + suffix}  (fold {FOLD}, {budget} steps)")
    print("=" * 80)
    for k, v in cfg_json.items():
        if k != "decision_rule":
            print(f"  {k:<20}: {v}")
    print("=" * 80, flush=True)

    # ---- --emit-config: DECLARE the reference up front so runs can go CONCURRENTLY ----
    # run_config.json is normally written only AFTER training, so a matched pair would have
    # to run sequentially (nb0 finishes -> nb3 validates against its config). Emitting the
    # config from the SAME construction code lets both start at once.
    #
    # This deliberately returns BEFORE the §1 check: at the Zoobot defaults the new nb0
    # config does not match the OLD (pre-protocol-match) reference, which is the whole
    # point -- we are declaring the new reference, not conforming to the old one.
    #
    # It is NOT a hole in §1. Emitting a config trains nothing, and
    # package_screen_results.py re-checks the declared reference against the config nb0
    # ACTUALLY emitted, on every pinned key, before any result is packaged. If someone
    # hand-edits the declared reference, that post-hoc check catches it.
    if getattr(args, "emit_config", None):
        p = os.path.abspath(args.emit_config)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as fh:
            json.dump(cfg_json, fh, indent=2)
        print(f"[emit-config] wrote the DECLARED reference config for {run_key} -> {p}")
        print("[emit-config] nothing was trained. Pass this to the other rung as "
              "--reference-config, and let package_screen_results.py verify it post-hoc.")
        return None

    # §1 -- refuse to start unless every pinned setting matches the reference run. Runs
    # BEFORE the dry-run return so --dry-run also validates the config it would use.
    ref_cfg = args.reference_config or os.path.join(
        root, "runs", RUNS["nb0_image"]["dir"], "run_config.json")
    if run_key == "nb0_image" and not os.path.isfile(ref_cfg):
        print("[§1] no reference config yet -- this IS the reference run (nb0_image); "
              "precondition check skipped.\n")
    elif args.diagnostic:
        # A DECLARED diagnostic deliberately varies a confound (e.g. the aug-vs-precision
        # decomposition run). It is NOT a screen rung and is never compared as one, so the
        # table is printed for the record but a mismatch is not fatal. This scopes the §1
        # check to matched runs; it does NOT touch the control gate, which stays absolute
        # and has no bypass. nb1/nb3 may never declare it (asserted below).
        print("[§1] --diagnostic: this run DELIBERATELY varies a pinned setting and is not "
              "part of the matched screen. Table printed for the record; mismatches are "
              "expected and not fatal.")
        try:
            check_preconditions(cfg_json, ref_cfg)
        except SystemExit as e:
            print(str(e))
            print("[§1] (tolerated: --diagnostic)\n")
    else:
        check_preconditions(cfg_json, ref_cfg)

    # ---- round-1 Baseline A checkpoint: needed by EVERY rung, checked BEFORE the dry-run
    # return so --dry-run catches a missing one in seconds instead of minutes into a run.
    #   * lp_ft rungs (nb1/nb3) load it as the LP->FT phase-1 head (Kumar arXiv:2202.10054)
    #   * ALL rungs score it on the target pool to build the agreement column's reference
    # It is 16 KB and lives in runs/, which data.zip does NOT pack -- so on a fresh Kaggle
    # session it has to be uploaded explicitly. That is a documented footgun, not a bug.
    if not os.path.isfile(round1_path):
        why = ("the LP->FT phase-1 head AND the agreement-column reference"
               if spec["lp_ft"] else "the agreement-column reference")
        raise SystemExit(
            f"[screen] ERROR: round-1 Baseline A checkpoint not found: {round1_path}\n"
            f"  Every rung needs it as {why}.\n"
            f"  It is ~16 KB and lives under runs/, which data.zip does not pack, so on a\n"
            f"  fresh session it is wherever your dataset put it. Point --round1-ckpt at it\n"
            f"  directly -- no copying, no directory scaffolding needed.")
    if args.round1_sha256:
        import hashlib
        got = hashlib.sha256(open(round1_path, "rb").read()).hexdigest()
        if got != args.round1_sha256:
            raise SystemExit(
                f"[screen] ERROR: --round1-ckpt sha256 mismatch\n"
                f"  file    : {round1_path}\n"
                f"  expected: {args.round1_sha256}\n"
                f"  got     : {got}\n"
                f"  This is not the round-1 Baseline A checkpoint. LP->FT would start from "
                f"the wrong head and the agreement column would not be comparable to any "
                f"previous run.")
        print(f"[ref] round-1 checkpoint sha256 verified: {got[:16]}")

    if args.dry_run:
        return None

    if os.path.exists(curve_csv):
        raise SystemExit(f"[screen] ERROR: {curve_csv} already exists -- appending would "
                         f"corrupt the curve. Move/delete {out_dir} first.")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- data: source_train only (Baseline A forms NO target batch) ----
    # aug="weak" is the default and is what an unfrozen run MUST use (PROJECT.md 7.3).
    # --aug none exists for ONE purpose: a pure code-path control. The round-1 cached run
    # trained on aug="none" cached features, so a weak-aug image-space run differs from it
    # in TWO ways (code path + augmentation). If the control is marginal, re-running it with
    # --aug none isolates the code path, separating "the new path is wrong" from
    # "augmentation moved the number".
    src_ds = LensDataset(splits_path, data_root, role="source_train", aug=args.aug,
                         n_views=1)
    steps = _steps_per_epoch(len(src_ds))
    epochs = int(math.ceil(budget / steps))
    print(f"[data] source_train n={len(src_ds)}  {steps} steps/epoch  -> {epochs} epochs "
          f"({steps * epochs} steps; R1 = step_{budget})", flush=True)

    # ---- trainer ----
    cfg = TrainConfig(
        splits_path=splits_path, data_root=data_root, fold=FOLD, head_type=HEAD_TYPE,
        n_blocks=n_blocks, backbone_lr=args.backbone_lr, lr_decay=LR_DECAY,
        init_head_from=(round1_path if spec["lp_ft"] else None),
        grad_checkpointing=grad_ckpt, lr=LR, weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE, pos_fraction=POS_FRACTION, total_steps=budget,
        epochs=epochs, seed=SEED, amp=bool(args.fp16), device=device,
        num_workers=args.num_workers, verbose_encoder=False)
    trainer = Trainer(cfg)

    # ---- standardizer: fit from the CACHED source features (as every round-1 runner
    # does), i.e. from the encoder's INITIAL state. DEVIATION NOTE: at n_blocks>0 the
    # encoder drifts away from those statistics during training. This is inherited from the
    # frozen protocol (fit once, then frozen -- PROJECT.md 7.1) and is applied IDENTICALLY
    # at train and eval time, so it is a consistent affine reparameterisation the head can
    # absorb, not a train/eval mismatch. Stated, not hidden.
    cache = ev.FeatureCache(cache_dir)
    sf = torch.from_numpy(np.asarray(cache.load("source_train")["features"],
                                     dtype=np.float32)).to(trainer.device)
    trainer.standardizer.fit(sf)
    del sf

    src_loader, n_workers_used = _make_loader_with_fallback(
        src_ds, steps, args.num_workers)

    # ---- live encoder (only when the cache is invalid) -- built LAZILY, on purpose ----
    # MEASURED failure: constructing it up front made the process hold the source dataset
    # (~800 MB) AND the eval datasets (~640 MB) at the moment the training DataLoader
    # spawned its workers. On Windows (spawn, not fork) the dataset is PICKLED to each
    # child, and that write died with "OSError: [Errno 22] Invalid argument" /
    # "UnpicklingError: pickle data was truncated". Deferring construction to the first
    # curve point means the workers are already up (persistent_workers=True spawns them on
    # the first iteration) and the big pickle happens while memory is low.
    live_box: dict = {}

    def get_live():
        if n_blocks == 0:
            return None
        if "live" not in live_box:
            live_box["live"] = LiveEncoder(splits_path, data_root, FOLD,
                                           batch_size=BATCH_SIZE, num_workers=0,
                                           device=device, verbose=True)
        return live_box["live"]

    from run_bakeoff_coral import compute_agreement_refs
    print(f"[ref] scoring round-1 Baseline A on target_fold{FOLD} for the agreement column ...",
          flush=True)
    agreement_refs = compute_agreement_refs(cache_dir, round1_path, [FOLD], device="cpu")

    cfg_extra = {"encoder_mode": trainer.encoder.effective_mode, "n_blocks": n_blocks,
                 "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                 "pos_fraction": POS_FRACTION, "seed": SEED, "steps_per_epoch": steps,
                 "method": "baseline_A_source_only", "fold": FOLD}

    # ---- train ----
    t0 = time.time()
    gstep = 0
    hist = []
    done = False
    for epoch in range(1, epochs + 1):
        src_loader.batch_sampler.set_epoch(epoch)
        for batch in src_loader:
            stats = trainer.train_step(batch)          # target_batch=None -> source-only
            hist.append(stats["loss"])
            gstep += 1

            if gstep % args.curve_every == 0:
                spath = os.path.join(ckpt_dir, f"step_{gstep}.pt")
                trainer.save(spath)
                loader_hook = build_split_loader(n_blocks, trainer.encoder, get_live(),
                                                 cache_dir, args.fp16)
                if loader_hook is not None:
                    loader_hook.reset()                 # fresh encode at THIS checkpoint
                row = curve_hook.append_curve_row(
                    curve_csv, gstep, spath, cache_dir, folds=(FOLD,),
                    target_fpr=args.target_fpr, device="cpu",
                    agreement_refs={FOLD: agreement_refs[FOLD]},
                    split_loader=loader_hook)
                el = time.time() - t0
                print(f"    [step {gstep:>5}/{budget}] "
                      f"loss={stats['loss']:.4f}  "
                      f"val_auroc={_cf(row['source_risk_auroc'])} "
                      f"val_tpr={_cf(row['source_risk_tpr'])} | "
                      f"fold{FOLD} tpr@1e-3={_cf(row[f'fold{FOLD}_tpr_1e-3'])} "
                      f"agr={_cf(row.get(f'agreement_fold{FOLD}'))} | "
                      f"{el/60:.1f} min", flush=True)

            if gstep >= budget:
                done = True
                break
        if done:
            break

    train_secs = time.time() - t0
    if not os.path.isfile(r1_ckpt):
        raise SystemExit(f"[screen] ERROR: R1 checkpoint {r1_ckpt} was not written "
                         f"({budget} must be a multiple of --curve-every).")
    # Task 5: every curve point must still have its checkpoint. Checked HERE (on the
    # training box) and again in package_screen_results.py (in the archives).
    kept_steps = assert_checkpoint_retention(ckpt_dir, curve_csv, budget, args.curve_every)
    print(f"[train] done in {train_secs/60:.1f} min ({gstep} steps)", flush=True)

    # ---- final evaluation at R1 (single candidate => R1 enforced structurally) ----
    res = evaluate_r1(r1_ckpt, n_blocks, trainer.encoder, get_live(), cache_dir,
                      args.target_fpr, args.fp16, out_dir, spec["dir"])
    res.update({"run": run_key, "run_dir": spec["dir"], "n_blocks": n_blocks,
                "train_minutes": train_secs / 60.0, "steps": gstep})

    cfg_json["num_workers_used"] = n_workers_used     # throughput provenance only
    # Task 5: record the retention fact in the config so the downloaded bundle is
    # self-describing -- a consumer can tell a complete run from a truncated one.
    cfg_json["n_checkpoints_expected"] = len(expected_checkpoint_steps(budget,
                                                                      args.curve_every))
    cfg_json["n_checkpoints_saved"] = len(kept_steps)
    res["n_checkpoints_saved"] = len(kept_steps)
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(cfg_json, fh, indent=2)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[out] run_config.json + results.json + curve.csv + scores/ -> {out_dir}")
    return res


def evaluate_r1(ckpt_path, n_blocks, encoder, live, cache_dir, target_fpr, fp16,
                out_dir, label):
    """Score R1 on fold-0 test positives vs the shared negatives; dump raw scores."""
    cache = ev.FeatureCache(cache_dir)
    head, std, _ = ev.load_checkpoint(ckpt_path, "cpu")

    def get(name):
        if n_blocks == 0:
            return cache.load(name)
        return live.encode(encoder, name, fp16=fp16)

    pos = get(f"test_pos_fold{FOLD}")
    neg = get("test_negatives_shared")
    ps = ev.score_features(head, std, pos["features"], "cpu")
    ns = ev.score_features(head, std, neg["features"], "cpu")

    y = np.concatenate([np.ones(ps.size, np.int64), np.zeros(ns.size, np.int64)])
    s = np.concatenate([ps, ns])
    suite = metric_suite(y, s, target_fpr=target_fpr)

    sdir = os.path.join(out_dir, "scores")
    os.makedirs(sdir, exist_ok=True)
    # ids travel with the scores so downstream paired comparisons can VERIFY row alignment
    # against round 1 rather than assume it (round-1 dumps carry pos_ids/neg_ids too).
    np.savez(os.path.join(sdir, f"{label}_fold{FOLD}.npz"),
             pos_scores=ps.astype(np.float64), neg_scores=ns.astype(np.float64),
             pos_ids=np.asarray(pos["ids"], dtype="<U80"),
             neg_ids=np.asarray(neg["ids"], dtype="<U80"),
             target_fpr=np.asarray(target_fpr))

    print(f"\n[R1] {label}  fold{FOLD}  TPR@{target_fpr:g} = {suite['tpr']:.4f}   "
          f"AUROC = {suite['auroc']:.4f}   AUPRC = {suite['auprc']:.4f}")
    return {"tpr": float(suite["tpr"]), "auroc": float(suite["auroc"]),
            "auprc": float(suite["auprc"]),
            "purity_corr": (None if suite["purity_corr"] is None
                            else float(suite["purity_corr"])),
            "n_pos": int(ps.size), "n_neg": int(ns.size), "ckpt": ckpt_path}


# ---------------------------------------------------------------------------
# the control check: nb0_image vs the round-1 cached run
# ---------------------------------------------------------------------------

def load_scores(path):
    """Return (pos_scores, neg_scores, pos_ids, neg_ids); ids are None if not stored."""
    with np.load(path, allow_pickle=False) as z:
        ids_p = z["pos_ids"] if "pos_ids" in z.files else None
        ids_n = z["neg_ids"] if "neg_ids" in z.files else None
        return z["pos_scores"], z["neg_scores"], ids_p, ids_n


def control_check(cache_dir, target_fpr=DEFAULT_TARGET_FPR, B=1000, seed=0, verbose=True,
                  control_scores=None):
    """Compare baselineA_fold0_nb0_image against baselineA_fold0_nb0_cached on fold 0.

    PASS iff the paired-bootstrap CI for (image - cached) contains 0, i.e. the two agree
    within the CI. Returns (passed, info).

    ``control_scores`` (a DIRECTORY) makes the gate satisfiable on a fresh Kaggle session,
    where there is no local run history. It must contain the uploaded score dumps:
        nb0_image_fold0.npz     (weak-aug/fp16 control)
        nb0_cached_fold0.npz    (round-1 reference)
    each WITH pos_ids/neg_ids -- without the ids the row-alignment check cannot run and the
    gate refuses, which is correct but wastes a session to rediscover.

    The gate FAILS CLOSED and there is deliberately no bypass flag.
    """
    root = os.path.normpath(os.path.join(_HERE, ".."))
    if control_scores:
        cs = os.path.abspath(control_scores)
        img_scores = os.path.join(cs, "nb0_image_fold0.npz")
        cached_override = os.path.join(cs, "nb0_cached_fold0.npz")
        if not os.path.isfile(img_scores) or not os.path.isfile(cached_override):
            return None, {"error":
                          f"--control-scores {cs} must contain BOTH nb0_image_fold0.npz "
                          f"and nb0_cached_fold0.npz (with pos_ids/neg_ids). "
                          f"Found: {sorted(os.listdir(cs)) if os.path.isdir(cs) else 'dir missing'}"}
    else:
        cached_override = None
        img_dir = os.path.join(root, "runs", RUNS["nb0_image"]["dir"])
        img_scores = os.path.join(img_dir, "scores",
                                  f"{RUNS['nb0_image']['dir']}_fold{FOLD}.npz")
    # round 1 dumped its per-fold scores as scores/step_8000_fold{k}.npz
    cached_scores = cached_override or os.path.join(
        ROUND1_DIR, "scores", f"step_{BUDGET_STEPS}_fold{FOLD}.npz")
    if not os.path.exists(cached_scores):
        sdir = os.path.join(ROUND1_DIR, "scores")
        cands = sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []
        fold0 = [f for f in cands if f"fold{FOLD}" in f]
        cached_scores = os.path.join(sdir, fold0[0]) if fold0 else None
    if cached_scores is None or not os.path.exists(cached_scores) \
            or not os.path.exists(img_scores):
        return None, {"error": f"missing scores (image={img_scores}, cached={cached_scores})"}

    p_i, n_i, pid_i, nid_i = load_scores(img_scores)
    p_c, n_c, pid_c, nid_c = load_scores(cached_scores)

    # the shared negative pool is identical across both runs by construction (PROJECT.md
    # 4.4), and both score the SAME fold-0 positives -> a paired bootstrap is valid.
    if p_i.size != p_c.size or n_i.size != n_c.size:
        return False, {"error": f"score vectors differ in length: image "
                                f"({p_i.size},{n_i.size}) vs cached ({p_c.size},{n_c.size})"}
    # VERIFY ROW ALIGNMENT, don't assume it. paired_bootstrap applies the same resampled
    # row indices to both vectors, so a permutation would silently corrupt the CI.
    if pid_i is None or pid_c is None or nid_i is None or nid_c is None:
        return None, {"error":
                      "score dumps are missing pos_ids/neg_ids, so row alignment cannot be "
                      "verified and the paired bootstrap would be unsound. The gate FAILS "
                      "CLOSED. Re-export the control dumps with ids "
                      "(run_screen_unfreeze saves them; round-1 dumps already carry them)."}
    aligned = bool(np.array_equal(pid_i, pid_c) and np.array_equal(nid_i, nid_c))
    if not aligned:
        return False, {"error": "score vectors are NOT row-aligned (stable ids differ) "
                                "-- a paired bootstrap would be invalid."}

    y = np.concatenate([np.ones(p_i.size, np.int64), np.zeros(n_i.size, np.int64)])
    s_img = np.concatenate([p_i, n_i])
    s_cac = np.concatenate([p_c, n_c])

    tpr_img = tpr_at_fpr(y, s_img, target_fpr)
    tpr_cac = tpr_at_fpr(y, s_cac, target_fpr)
    bs = paired_bootstrap(y, s_img, s_cac, metric_fn=default_tpr_metric(target_fpr),
                          B=B, seed=seed)
    info = {"tpr_nb0_image": float(tpr_img), "tpr_nb0_cached": float(tpr_cac),
            "diff": float(tpr_img - tpr_cac), "bootstrap": bs,
            "row_aligned_by_ids": aligned,
            "image_scores": img_scores, "cached_scores": cached_scores}

    lo, hi = _diff_ci(bs)
    passed = (lo is not None and lo <= 0.0 <= hi)
    info["diff_ci"] = [lo, hi]
    info["passed"] = bool(passed)

    if verbose:
        print("\n" + "=" * 80)
        print("CONTROL: baselineA_fold0_nb0_image  vs  baselineA_fold0_nb0_cached (fold 0)")
        print("=" * 80)
        print(f"  TPR@1e-3  nb0_image  = {tpr_img:.4f}")
        print(f"  TPR@1e-3  nb0_cached = {tpr_cac:.4f}")
        print(f"  paired diff (image - cached) = {tpr_img - tpr_cac:+.4f}  "
              f"CI [{_f(lo)}, {_f(hi)}]")
        print(f"  VERDICT: {'PASS' if passed else 'FAIL'} -- CI "
              f"{'contains' if passed else 'EXCLUDES'} 0")
        if not passed:
            print("  The image-space path does NOT reproduce the cached run. nb1/nb3 would "
                  "be uninterpretable; refusing to proceed.")
        print("=" * 80)
    return passed, info


def _diff_ci(bs):
    """(lo, hi) of the paired difference -- metrics.paired_bootstrap returns 'lo'/'hi'."""
    if not isinstance(bs, dict):
        return None, None
    lo, hi = bs.get("lo"), bs.get("hi")
    if lo is None or hi is None:
        return None, None
    return float(lo), float(hi)


def _f(x):
    return "n/a" if x is None else f"{x:+.4f}"


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    choices=list(RUNS) + ["control", "all"],
                    help="which screen run to execute ('control' just re-checks nb0_image "
                         "vs nb0_cached; 'all' runs nb0_image -> control -> nb1 -> nb3)")
    ap.add_argument("--data-root", default=os.path.join(root, "data"))
    ap.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--device", default=None)
    # 2 (not 4) is the default: on Kaggle two training processes SHARE 4 CPU cores, and on
    # Windows 4 workers intermittently blew up spawn-pickling the ~790 MB source dataset.
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--reference-config", default=None,
                    help="run_config.json of the completed nb0_image run. Every pinned "
                         "setting must match it (§1). Upload it with the Kaggle dataset.")
    ap.add_argument("--control-scores", default=None,
                    help="directory holding the uploaded control score dumps "
                         "(nb0_image_fold0.npz + nb0_cached_fold0.npz, WITH pos_ids/"
                         "neg_ids). Lets the control gate run on a fresh Kaggle session "
                         "with no local run history. The gate still fails closed.")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="LR of the TOP unfrozen block (default: head lr / 10)")
    ap.add_argument("--curve-every", type=int, default=CURVE_EVERY)
    ap.add_argument("--budget-steps", type=int, default=BUDGET_STEPS,
                    help="SMOKE-TEST ONLY. The screen result must use the pre-registered "
                         "8000; a smaller value writes to a suffixed dir so it can never be "
                         "mistaken for the real run.")
    ap.add_argument("--dir-suffix", default="",
                    help="append to the run dir name (smoke tests / control variants)")
    ap.add_argument("--aug", default="weak", choices=("weak", "none"),
                    help="source augmentation. 'weak' (default, PROJECT.md 7.3) is the "
                         "protocol. 'none' matches the round-1 cached run exactly and is "
                         "for isolating the code path in the control.")
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--eval-seed", type=int, default=0)
    # NO BYPASS FLAG, deliberately. The Task-2 build had --no-require-control; it is
    # REMOVED. The control is what establishes that the image-space path reproduces the
    # cached run, and without it nb1/nb3 are uninterpretable rather than merely unverified.
    # A gate with an escape hatch gets used under time pressure at 2am on a Kaggle session
    # that is about to expire, which is exactly when it matters most.
    ap.add_argument("--diagnostic", action="store_true",
                    help="declare that this run deliberately varies a pinned setting (e.g. "
                         "the aug-vs-precision decomposition run). Scopes the §1 confound "
                         "check to matched runs only -- the table is still printed. Has NO "
                         "effect on the control gate, and is REFUSED for nb1/nb3.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--round1-ckpt", default=None, metavar="PATH",
                    help="Path to the round-1 Baseline A step_8000 checkpoint. EVERY rung "
                         "needs it (nb1/nb3 as the LP->FT phase-1 head; all rungs as the "
                         "agreement-column reference). Defaults to "
                         "runs/bakeoff_baseline_A/checkpoints/step_8000.pt, which only "
                         "exists locally -- data.zip does not pack runs/. Point this "
                         "straight at the file wherever it actually is.")
    ap.add_argument("--round1-sha256", default=None, metavar="HEX",
                    help="If given, verify --round1-ckpt hashes to this. Use it to be sure "
                         "you got the right file: the agreement column is only comparable "
                         "across runs when every run uses the same reference.")
    ap.add_argument("--emit-config", default=None, metavar="PATH",
                    help="Build this rung's run_config.json and WRITE IT WITHOUT TRAINING, "
                         "then exit. Lets a matched pair run CONCURRENTLY: declare nb0's "
                         "config up front and pass it to nb3 as --reference-config. "
                         "package_screen_results.py verifies the declared reference against "
                         "the config nb0 actually emitted, so §1 is not weakened.")
    args = ap.parse_args(argv)

    if args.diagnostic and args.run in ("nb1", "nb3", "all"):
        raise SystemExit(
            f"[screen] --diagnostic is refused for {args.run!r}: nb1/nb3 ARE the matched "
            f"screen rungs, and the whole point of §1 is that only n_blocks may differ "
            f"between them and nb0_image. Use it only for declared nb0 diagnostics.")

    if args.run == "control":
        passed, info = control_check(os.path.abspath(args.cache_dir), args.target_fpr,
                                     args.bootstrap, args.eval_seed,
                                     control_scores=args.control_scores)
        if info.get("error"):
            # say WHY it failed closed -- discovering this by silence costs a session
            print("\n" + "=" * 78)
            print("CONTROL GATE: FAILED CLOSED")
            print("=" * 78)
            print(f"  {info['error']}")
            print("=" * 78)
        out = os.path.join(root, "runs", "baselineA_fold0_control.json")
        with open(out, "w") as fh:
            json.dump(info, fh, indent=2, default=str)
        print(f"[out] {out}")
        return 0 if passed else 1

    order = ["nb0_image", "nb1", "nb3"] if args.run == "all" else [args.run]

    # --dry-run and --emit-config both train nothing, so neither needs the control gate to
    # have passed. The gate still guards every real nb1/nb3 launch, with no bypass.
    no_train = bool(args.dry_run or args.emit_config)

    for key in order:
        if key in ("nb1", "nb3") and not no_train:
            passed, info = control_check(os.path.abspath(args.cache_dir), args.target_fpr,
                                         args.bootstrap, args.eval_seed, verbose=True,
                                         control_scores=args.control_scores)
            if passed is not True:
                raise SystemExit(
                    f"[screen] REFUSING to run {key}: the nb0_image control has not passed "
                    f"({info.get('error', 'CI excludes 0')}). nb1/nb3 are uninterpretable "
                    f"without it. There is NO bypass flag by design -- fix the control, or "
                    f"pass --control-scores pointing at the uploaded dumps.")
        train_one(key, args)
        if key == "nb0_image" and not no_train:
            control_check(os.path.abspath(args.cache_dir), args.target_fpr,
                          args.bootstrap, args.eval_seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
