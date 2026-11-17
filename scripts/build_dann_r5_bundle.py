"""
build_dann_r5_bundle.py
=======================
Assemble (and VERIFY) the Kaggle upload bundle for **round 5, DANN x n_blocks=3**, into
``kaggle/dann_r5/`` -- and package the results on the way back.

Modelled on ``build_coral_r4_bundle.py``: same layout, same discipline, same
verify-before-upload principle, round 5's contents. Two modes:

    (default)          build + verify the UPLOAD bundle
    --pack-results     build the DOWNLOAD tars + MANIFEST from finished run dirs
    --verify-results   check a downloaded set against its MANIFEST

WHAT A FRESH KAGGLE BOX DOES NOT HAVE
-------------------------------------
    data.zip                                             images + splits + feature cache
    scripts/                                             the WHOLE folder
    runs/bakeoff_baseline_A/checkpoints/step_8000.pt      the LP->FT phase-1 head
    runs/bakeoff_baseline_A/curve.csv                     the step-0 gate table's SOURCE
    runs/dann_r5_morph_step0.json                         the morph step-0 reading
    runs/dann_r5_reachability.json                        the nb3 reachability report
    MANIFEST.json / VERSION.txt / README.md / dataset-metadata.json

**runs/ is NOT inside data.zip**, which is why those ship explicitly.

ONE HEAD, FIVE FOLDS -- and why ``curve.csv`` travels with it. Baseline A is source-only
and the source does not vary by fold, so for a ``LayerNorm(640) + Linear(640->1)`` head
the source-only linear probe is UNIQUE: one 28.475603-norm file serves every fold. That
makes the head-norm check a proof of the right FILE and no proof at all of the right
FOLD, so the per-fold TPR/AUROC step-0 gate is load-bearing. ``verify`` below RE-DERIVES
``run_dann_r5.FOLD_GATE`` from ``curve.csv`` row ``step=8000`` rather than trusting the
table -- a mistyped digit there would abort a correct run, or worse, pass a wrong one.

**The WHOLE scripts/ folder is copied, not the import closure** (PROJECT.md 7.7: the
closure has grown twice mid-project and "stale scripts silently run old behavior" is a
documented failure mode). The closure is still computed and recorded so the bundle can
prove it is complete.

THE MORPHOLOGY PROXY NEEDS RAW SHARDS, NOT THE FEATURE CACHE. ``morph_proxy.py`` reads
``data/ml_fp_cutouts/shards/*.npz`` and every ``data/q1_unlabeled/shards/*.npz`` (for the
``partitions`` array that defines the P1 reserve). Both are inside ``data.zip`` --
measured, 256 files / 0.067 GB and 250 files / 3.435 GB -- and ``verify`` checks for them
by name, because a bundle that mounts fine and then cannot build the monitor arming abort
condition 3 is a wasted session.

VERIFICATION (runs by default)
------------------------------
  * every script the cells invoke is present, and the import closure is complete;
  * ``data.zip`` holds splits, the feature cache, AND the two raw shard trees the
    morphology proxy needs, with no backslash arcnames (Kaggle rejects those, 7.7);
  * the phase-1 head loads and its L2 is exactly the pinned 28.475603;
  * ``FOLD_GATE`` re-derives from ``curve.csv`` for all five folds;
  * the round-5 constants are what they claim: lambda horizon 2000 (NOT 8000, NOT the run
    length), D dropout 0.2 in two layers, zero BatchNorm anywhere in D, holdout 0.10,
    true batch 256, Zoobot's 1e-4 / 0.75 / 0.05, and ONE optimiser with D as a fifth
    param group at nb3.

Run:
    python scripts/build_dann_r5_bundle.py --kaggle-user YOUR_HANDLE
    python scripts/build_dann_r5_bundle.py --verify-only
    python scripts/build_dann_r5_bundle.py --pack-results /kaggle/working/runs/dann_r5_nb3_fold0 \\
        --results-out /kaggle/working/dann_r5_out --logs /kaggle/working/logs
    python scripts/build_dann_r5_bundle.py --verify-results /path/to/downloaded/dann_r5_out
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import sys
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from package_kaggle_screen import closure          # noqa: E402  (the ONE closure impl)

BUNDLE = "dann_r5"
CELLS = "KAGGLE_DANN_R5_CELLS.md"                  # optional; shipped if present

# entry points the session invokes, as import roots for the closure check
BUNDLE_ROOTS = ["run_dann_r5", "dann_round5", "morph_proxy", "calibrate_morph_r5",
                "test_da_dann", "package_screen_results", "rescore_dann_r5"]

# the cells assert each of these is in MANIFEST["scripts"] before copying anything
CELL_REQUIRES = ("run_dann_r5.py", "dann_round5.py", "da_losses.py", "morph_proxy.py",
                 "calibrate_morph_r5.py", "write_dann_r5_prereg.py",
                 "test_da_dann.py", "train_skeleton.py",
                 "zoobot_encoder.py", "lens_dataset.py", "samplers.py", "heads.py",
                 "evaluate.py", "metrics.py", "live_encode.py",
                 "package_screen_results.py", "build_dann_r5_bundle.py",
                 "rescore_dann_r5.py",
                 # ---- round 6: the lambda=0 baseline on folds 1-4 ----
                 # Same runner, same arm, no new code path. These three are the PRE-RUN
                 # audits and the pre-registration writer; they ride in the bundle so the
                 # notebook can re-verify on the venue rather than trusting a laptop.
                 "audit_lambda0_r6.py", "lambda0_r6_step0_conditions.py",
                 "write_lambda0_r6_prereg.py",
                 # ---- round 6: BNM (Stage 2 runs these on the venue) ----
                 "bnm_round6.py", "run_bnm_r6.py", "test_bnm_round6.py",
                 "test_da_bnm.py", "bnm_stage0_expressivity.py")

CHECKPOINTS = {
    "runs/bakeoff_baseline_A/checkpoints/step_8000.pt":
        ("the LP->FT phase-1 head, for EVERY fold. Baseline A is source-only and the "
         "source does not vary by fold, so the source-only linear probe is unique and "
         "one file is the correct artifact, not a shortcut."),
}

# small text artifacts that must travel with the bundle
EXTRA_FILES = {
    "runs/bakeoff_baseline_A/curve.csv":
        ("the SOURCE of run_dann_r5.FOLD_GATE. verify() re-derives the pinned per-fold "
         "TPR/AUROC from row step=8000 rather than trusting the table."),
    "runs/dann_r5_morph_step0.json":
        ("BOTH morphology instruments at step 0, on both arms: HEAD 0.565735 (WHOLE "
         "POOL -- what both detectors read, and the scale M is declared against, "
         "headroom 0.0657) / 0.578125 (eval half, interpretation only), and PROBE "
         "0.9202 (adapter) / 0.9214 (raw encoder), which has no detector. Plus the "
         "record of the withdrawn thresholds and the corrected reason for each."),
    "runs/dann_r5_reachability.json":
        ("the nb3 gradient-reachability report -- 14,835,520 encoder elements in the "
         "graph, stages.1/2/3 only, head 1,280 (gains AND biases)."),
    "runs/dann_r5_arm1_control/steps.csv":
        ("THE ARM-1 CONTROL CURVE. Arm 2 is compared against it at matched absolute "
         "steps, so it must be ON the venue -- --control-curve cannot reach a file that "
         "only exists on the laptop. Carries morph_proxy_auroc (condition 3) and "
         "src/tgt_participation_ratio (condition 4). 8000 rows, forward-filled between "
         "measurements; the readings sit on a stride-25 grid at offset 1."),
    "runs/dann_r5_arm1_control/run_config.json":
        ("arm 1's provenance next to its curve: 8000/8000 steps, 32/32 checkpoints, "
         "267.8 min on a T4, step-0 gate PASS, no aborts, no resumes. Without this a "
         "control curve is a bare CSV with no record of whether the run that made it "
         "was clean."),
    "runs/dann_r5_arm1_control/steps_morph_stats.json":
        ("the measured J / IQR / declarability verdicts that conditions 3 and 4 were "
         "RE-SPECIFIED from (PROJECT.md 7.17)."),
    "runs/dann_r5_budget.json":
        ("THE BUDGET RECORD -- currently a WITHDRAWN declaration, self-certifying. A "
         "declaration of 2000 was made 2026-09-01 and withdrawn 2026-09-02: the budget "
         "is the run owner's decision, made from runs/dann_r5_budget_packet.md. "
         "is_preregistered_budget is False everywhere until one is chosen and hashed."),
    "runs/dann_r5_budget_packet.md":
        ("the budget DECISION PACKET: both arms' full 32-checkpoint curves, what each "
         "candidate rule selects, local scatter per candidate, lambda coverage per "
         "budget, the d_acc cross-check, and the protocol conflict stated with no "
         "position taken. Contains NO recommendation."),
    "runs/dann_r5_curves/arm1_lambda0_curve.csv":
        ("the re-scored lambda=0 control curve, 32 checkpoints."),
    "runs/dann_r5_curves/arm2_dann_curve.csv":
        ("the re-scored development curve the budget was declared against."),
    "runs/dann_r5_prereg.json":
        ("THE PRE-REGISTRATION (PROJECT.md 7.17), self-certifying. It must travel with "
         "the run: a pre-registration that lives only on the laptop that wrote it "
         "cannot be checked against what actually ran."),

    # ---------------- round 6: the lambda=0 baseline, folds 1-4 ----------------
    "runs/lambda0_r6_prereg.json":
        ("ROUND 6's PRE-REGISTRATION (PROJECT.md 7.21), self-certifying. Budget 4000 "
         "under the rule 'the control is read where its treatment is read'; sensitivity "
         "readouts at 1500 and 2000; conditions ADVISORY on every fold. It is the "
         "AUTHORITY for round 6's budget -- run_config.json's confirm_budget_rule field "
         "carries round 5's DANN wording, because the runner is deliberately not edited."),
    "runs/lambda0_r6_audit.json":
        ("the three PRE-RUN audits: script provenance as a cross-run hash matrix, the "
         "field-for-field config comparison (38/38 training-relevant keys identical "
         "across 10 runs), and the 11-component schedule audit RE-RUN on the current "
         "bytes -- which is what licenses the 1500 and 2000 readouts."),
    # ---------------- round 6: BNM ----------------
    "runs/bnm_r6_stage0a.json":
        ("STAGE 0a's RESULT, and it is LOAD-BEARING AT RUNTIME, not documentation: "
         "bnm_round6.stage0a_anchors() READS this file for the sharpening ceiling "
         "(10.8413, which the budget rule crosses), the init nuclear norm (9.7704, which "
         "the gamma selection's condition 1 measures movement from) and the init "
         "predicted-positive fraction (0.2292, which the collapse band's floor is drawn "
         "from). Without it every round-6 script raises on import of its own anchors -- "
         "there is deliberately no fallback literal, so the file cannot silently drift "
         "from the rules that read it (test_bnm_round6.T8 corrupts it to prove that)."),

    "runs/lambda0_r6_step0_conditions.json":
        ("every declared condition evaluated AT STEP 0 and over arm 1's 8000 steps, "
         "BEFORE it was written into the pre-registration. Records that THREE conditions "
         "fire on a healthy control -- 3b on the head column at step 1, and condition 4 "
         "on both domains for 100% of the run -- so a firing on folds 1-4 is a disclosed "
         "expectation about the threshold, not a finding about the run."),
}

DATA_REQUIRES = (("data/splits/splits.json",
                  "data/feature_cache/source_train.npz",
                  "data/feature_cache/val.npz",
                  "data/feature_cache/test_negatives_shared.npz",
                  "data/q1_lenses_processed/lens_cutouts.npz")
                 + tuple(f"data/feature_cache/test_pos_fold{k}.npz" for k in range(5))
                 + tuple(f"data/feature_cache/target_fold{k}.npz" for k in range(5)))

# raw shard TREES (prefixes, not single files) the morphology proxy reads
DATA_TREES_REQUIRED = {
    "data/ml_fp_cutouts/shards/": "morph proxy: the ML-FP (=1) class",
    "data/q1_unlabeled/shards/": ("morph proxy: the P1-reserve (=0) class, and the "
                                  "`partitions` array that defines it"),
}

# pinned round-5 values the bundle asserts
BASELINE_A_HEAD_L2 = 28.475603103637695

DEFAULT_MAX_PART_BYTES = 480 * 1000 * 1000     # 7.7: ~475 MB is where browser DLs flake
SMALL_CKPT_LIMIT = 200 * 1000 * 1000

# ---------------------------------------------------------------------------
# dataset-metadata.json -- constraints ENFORCED, not hoped for
# ---------------------------------------------------------------------------
DATASET_SLUG = "lens-paper-dann-r5"                                        # 18 chars
DATASET_TITLE = "lens-paper dann round 5"                                  # 23 chars
DATASET_SUBTITLE = "Round-5 DANN x n_blocks=3 for Euclid Q1 lens finding"    # 51 chars


def build_metadata(kaggle_user: str | None) -> dict:
    user = (kaggle_user or "USERNAME").strip().strip("/")
    slug = DATASET_SLUG
    bad = [c for c in slug if not (c.isalnum() or c == "-")]
    if bad:
        raise SystemExit(f"dataset slug {slug!r} contains illegal characters {set(bad)}; "
                         f"Kaggle allows alphanumeric and '-' only")
    if not (3 <= len(slug) <= 50):
        raise SystemExit(f"dataset slug {slug!r} is {len(slug)} chars; Kaggle requires 3-50")
    if not (6 <= len(DATASET_TITLE) <= 50):
        raise SystemExit(f"title is {len(DATASET_TITLE)} chars; Kaggle requires 6-50")
    if not (20 <= len(DATASET_SUBTITLE) <= 80):
        raise SystemExit(f"subtitle is {len(DATASET_SUBTITLE)} chars; Kaggle requires 20-80")
    return {"title": DATASET_TITLE, "subtitle": DATASET_SUBTITLE,
            "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def head_l2(path):
    import torch
    import evaluate as ev
    head, _std, _m = ev.load_checkpoint(str(path), "cpu")
    return float(torch.sqrt(sum((p.detach().float() ** 2).sum()
                                for p in head.parameters())))


def link_or_copy(src: Path, dst: Path, force_copy=False):
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "already present"
        dst.unlink()
    if not force_copy:
        try:
            os.link(src, dst)
            return "hardlink (0 bytes extra)"
        except OSError:
            pass
    t0 = time.time()
    shutil.copy2(src, dst)
    return f"copied in {time.time() - t0:.0f}s"


def fold_gate_from_curve(curve_csv, step=8000) -> dict:
    """Re-derive the per-fold step-0 gate table from Baseline A's own curve.

    The pinned ``run_dann_r5.FOLD_GATE`` was read out of this file by hand. A mistyped
    digit would abort a correct run or, worse, wave through a run pointed at the wrong
    fold -- and the head-norm check cannot catch that, because the head is
    fold-independent. So the table is re-derived here and compared.
    """
    with open(curve_csv, newline="") as fh:
        rows = {int(float(r["step"])): r for r in csv.DictReader(fh)}
    row = rows.get(int(step))
    if row is None:
        raise SystemExit(f"{curve_csv} has no row at step {step} "
                         f"(have {sorted(rows)[:3]} ... {sorted(rows)[-3:]})")
    return {k: {"tpr": float(row[f"fold{k}_tpr_1e-3"]),
                "auroc": float(row[f"fold{k}_auroc"])} for k in range(5)}


README = """# lens-paper — round-5 DANN x n_blocks=3 (Kaggle dataset)

Everything a fresh Kaggle **T4 x 2** session needs.

## Name

* slug  `lens-paper-dann-r5`
* title `lens-paper dann round 5`
* id    `<your-kaggle-handle>/lens-paper-dann-r5`

```
python scripts/build_dann_r5_bundle.py --kaggle-user YOUR_HANDLE --version 1
kaggle datasets create  -p kaggle/dann_r5 -r zip
kaggle datasets version -p kaggle/dann_r5 -r zip -m "round-5 dann v2"
```

Bump the version on every re-upload — Kaggle silently serves the previous one otherwise.

## What it looks like once mounted

Kaggle auto-extracts archives uploaded to a dataset, so there is no `data.zip` to unzip:

```
/kaggle/input/lens-paper-dann-r5/
    data/splits/splits.json
    data/feature_cache/...
    data/ml_fp_cutouts/shards/...        <- the morphology proxy's ML-FP class
    data/q1_unlabeled/shards/...         <- the P1-reserve class + `partitions`
    scripts/                             <- copied to /kaggle/working and imported
    runs/bakeoff_baseline_A/checkpoints/step_8000.pt   <- LP->FT phase-1 head (all folds)
    runs/bakeoff_baseline_A/curve.csv                  <- the step-0 gate table's source
    runs/dann_r5_morph_step0.json  runs/dann_r5_reachability.json
    runs/dann_r5_arm1_control/steps.csv                <- arm 2's --control-curve
    MANIFEST.json  VERSION.txt  README.md  dataset-metadata.json
```

## What the session runs

Validate in minutes, before committing a GPU:

```
python scripts/test_da_dann.py                       # the full local suite
python scripts/run_dann_r5.py --dry-run              # mock data, 3 steps, no GPU
python scripts/run_dann_r5.py --arm dann_nb3 --emit-config /kaggle/working/ref.json     --input-root /kaggle/input/lens-paper-dann-r5
```

## RUN ORDER — and it is not arbitrary

Condition 3 measures the development arm against the **λ=0 control at the same absolute
step**, so the control has to exist first.

**1. The λ=0 control, fold 0, 8000 steps, log-only.** Both arms run the full 8000 now
(brief v3 §5): the control is the reference trajectory for every monitor, not just a
null for the DANN loss, and a control that stops at 2000 leaves condition 3 — and
feature std, and participation ratio — with no reference past step 2000.

```
python -u scripts/run_dann_r5.py --arm dann_nb3_lambda0 --fold 0     --input-root /kaggle/input/lens-paper-dann-r5     --out-dir /kaggle/working/runs/dann_r5_nb3_lambda0_fold0
```

**2. Navneet hashes the 0.10 margin** from the control's curve, before step 3.

**3. The development run, fold 0, 8000 steps, log-only**, pointed at the control:

```
python -u scripts/run_dann_r5.py --arm dann_nb3 --fold 0     --control-curve /kaggle/working/runs/dann_r5_nb3_lambda0_fold0/steps.csv     --reference-config /kaggle/working/ref.json     --input-root /kaggle/input/lens-paper-dann-r5     --out-dir /kaggle/working/runs/dann_r5_nb3_fold0
```

`--morph-every` must be **identical** on both arms — it is a PINNED key, and a mismatch
does not coarsen condition 3's comparison, it deletes it (the run refuses to start).

**4. Budget declared and hashed** → `runs/dann_r5_budget.json`, mirroring
`coral_r4_budget.json`. Expect **declared, not derived**: §7.13 and §7.14 both establish
that these curves do not locate a budget, so decline the argmax and write "declared" in
the file. Until then `CONFIRM_STEPS = 2000` stays a placeholder with
`is_preregistered_budget = false`.

**5. Confirmatory, folds 1–4, at the declared budget, detectors ARMED.** `abort_mode`
defaults to `armed` there and to `log_only` on fold 0 and on the control; whatever is
used is recorded in `run_config.json` with its source.

If two arms are run concurrently, one per device, the launching cell must BLOCK while
both run or Kaggle may time the session out (§7.7).

## Getting it home

```
python scripts/build_dann_r5_bundle.py --pack-results /kaggle/working/runs/dann_r5_nb3_fold0     /kaggle/working/runs/dann_r5_nb3_lambda0_fold0     --results-out /kaggle/working/dann_r5_out --logs /kaggle/working/logs
```

Independent self-contained tars, not `tar | split` — a part that fails to download can be
re-fetched and verified on its own (7.7). Verify locally after downloading:

```
python scripts/build_dann_r5_bundle.py --verify-results /path/to/dann_r5_out
```
"""


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _make_bundle_writable(out: Path) -> int:
    """Clear the read-only bit across the bundle tree before rebuilding.

    ``integrity.protect()`` makes every hashed artifact under ``runs/`` read-only, and
    ``shutil.copy2`` PRESERVES that bit onto the bundle's copies. The next rebuild then
    dies with PermissionError partway through -- after the scripts are refreshed but
    before VERSION.txt and MANIFEST.json are, leaving a bundle whose CODE IS NEW AND WHOSE
    LABEL IS OLD. That is the stale-mount failure inverted, and far harder to notice.

    The bundle is a DERIVED artifact, never the protected original, so clearing the bit
    here is safe: the originals under ``runs/`` stay read-only.
    """
    n = 0
    for root, _d, files in os.walk(out):
        for f in files:
            p = os.path.join(root, f)
            if not os.access(p, os.W_OK):
                os.chmod(p, 0o644)
                n += 1
    return n


def build(out: Path, version: int, force_copy: bool, kaggle_user=None):
    out.mkdir(parents=True, exist_ok=True)
    _n_unlocked = _make_bundle_writable(out)
    if _n_unlocked:
        print(f"  [build] cleared the read-only bit on {_n_unlocked} bundle file(s) "
              f"inherited from the protected originals")
    print("=" * 78)
    print("ROUND-5 DANN x nb3 -- KAGGLE UPLOAD BUNDLE")
    print("=" * 78)
    print(f"  out: {out}")

    sdir = out / "scripts"
    sdir.mkdir(exist_ok=True)
    names = sorted(p.name for p in _HERE.glob("*.py"))
    for n in names:
        shutil.copy2(_HERE / n, sdir / n)          # COPY, never a link
    print(f"\n  scripts/        {len(names)} .py files "
          f"({sum((sdir / n).stat().st_size for n in names) / 1e6:.1f} MB, copied)")

    clo = closure(BUNDLE_ROOTS)
    missing_clo = [f"{m}.py" for m in clo if f"{m}.py" not in names]
    print(f"  import closure  {len(clo)} modules from {len(BUNDLE_ROOTS)} roots"
          + (f"  *** MISSING {missing_clo} ***" if missing_clo else "  (all present)"))

    ck_info = {}
    print()
    for rel, why in {**CHECKPOINTS, **EXTRA_FILES}.items():
        src = _ROOT / rel
        dst = out / rel
        if not src.is_file():
            print(f"  MISSING  {rel}\n           needed as: {why}")
            ck_info[rel] = {"present": False, "why": why}
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        ck_info[rel] = {"present": True, "bytes": dst.stat().st_size,
                        "sha256": sha256_file(dst), "why": why}
        print(f"  ok       {rel}  ({dst.stat().st_size / 1024:.1f} KB, "
              f"sha256 {ck_info[rel]['sha256'][:16]})")

    dz = _ROOT / "data.zip"
    print()
    if dz.is_file():
        mode = link_or_copy(dz, out / "data.zip", force_copy=force_copy)
        print(f"  ok       data.zip  ({dz.stat().st_size / 1e9:.2f} GB, {mode})")
    else:
        print("  MISSING  data.zip -- build it with: python scripts/make_data_zip.py")

    # ---- the round-5 constants this bundle depends on, recorded AND asserted ----
    import da_losses as DL
    import dann_round5 as D5
    import run_dann_r5 as R5
    probe = DL.DomainDiscriminator(640)
    settings = {
        "lambda_horizon_default": int(R5.LAMBDA_HORIZON),
        "lambda_at_750_1500_2000_8000": [DL.dann_lambda_at_step(b, R5.LAMBDA_HORIZON)
                                         for b in (750, 1500, 2000, 8000)],
        "d_topology": [640, probe.hidden, probe.hidden, 1],
        "d_dropout": float(probe.dropout),
        "d_dropout_layers": int(probe.n_dropout_layers),
        "d_norm": "LayerNorm",
        "d_params": sum(p.numel() for p in probe.parameters()),
        "d_holdout_frac": float(D5.DEFAULT_HOLDOUT_FRAC),
        "domain_labels": {"source": DL.DANN_SOURCE_LABEL, "target": DL.DANN_TARGET_LABEL},
        "lr": R5.LR, "lr_decay": R5.LR_DECAY, "weight_decay": R5.WEIGHT_DECAY,
        "batch_size": R5.BATCH_SIZE, "pos_fraction": R5.POS_FRACTION, "seed": R5.SEED,
        "dev_steps": int(R5.DEV_STEPS), "ckpt_every": int(R5.CKPT_EVERY),
        "log_every": int(R5.LOG_EVERY),
        "n_checkpoints_expected": int(R5.DEV_STEPS // R5.CKPT_EVERY),
        "morph_n_fit_per_class": int(R5.MORPH_N_FIT_PER_CLASS),
        "morph_n_eval_per_class": int(R5.MORPH_N_EVAL_PER_CLASS),
        "morph_ridge_alpha": float(R5.MORPH_RIDGE_ALPHA),
        "baseline_a_head_l2": BASELINE_A_HEAD_L2,
        "fold_gate": {str(k): v for k, v in R5.FOLD_GATE.items()},
        "arms": sorted(R5.ARMS),
        # ---- brief v3 ----
        "morph_every_default": int(R5.MORPH_EVERY),
        "arm_steps": {k: int(v["steps"]) for k, v in R5.ARMS.items()},
        "pinned_keys": list(R5.PINNED_KEYS),
        "abort_cond3_margin": R5.ABORT_MARGIN_COND3,
        "abort_cond3_column": R5.ABORT_COND3_COLUMN,
        "abort_cond3_interpretation_columns": list(R5.COND3_INTERPRETATION_COLUMNS),
        "abort_cond3_median_window_default": 1,
        "abort_cond3_m_ceiling": R5.COND3_M_CEILING,
        "abort_cond3_m_j_factor": R5.COND3_M_J_FACTOR,
        "abort_cond3_floor": R5.COND3_SIGN_FLIP,
        "abort_cond4_form": "differential vs the lambda=0 control, src and tgt SEPARATELY",
        "abort_cond4_src_ceiling": R5.COND4_SRC_CEILING,
        "abort_cond4_tgt_ceiling": R5.COND4_TGT_CEILING,
        "abort_cond4_sustain": int(R5.ABORT_SUSTAIN_COND4),
        "respecified_from_control_arm": (
            "conditions 3 and 4 were RE-SPECIFIED from arm 1 on 2026-08-28 and are NOT "
            "pre-registered; see derived_from_control_arm in runs/dann_r5_prereg.json "
            "and PROJECT.md 7.17. Condition 3 moved from morph_head_auroc_pool to "
            "morph_proxy_auroc (the head column fell to chance AT lambda = 0, because "
            "ML-FP and P1-reserve are both class 0 in the lens task); condition 4's "
            "absolute PR < 20 became differential (it fired 7,999 of 8,000 control "
            "steps). Clean because arm 2 is log_only: no detector acts on the run this "
            "was decided before."),
        "abort_modes": {"dann_nb3/fold0": R5.default_abort_mode("dann_nb3", 0),
                        "dann_nb3/fold1": R5.default_abort_mode("dann_nb3", 1),
                        "dann_nb3_lambda0/fold0":
                            R5.default_abort_mode("dann_nb3_lambda0", 0)},
        "withdrawn": ("the design doc's 'morphology-proxy AUROC below 0.55' and brief "
                      "v2's 0.6910 port of it. Both rested on a 0.61 -> 0.33-0.46 "
                      "collapse attributed to DANN; PROJECT.md attributes that failure "
                      "to IMPORTANCE WEIGHTING and records no numbers for it. Condition "
                      "3 is now measured against the lambda=0 control step for step."),
    }

    # the re-score's checkpoint tars, INSIDE this bundle and in THIS manifest
    ckpt_parts = stage_checkpoints(out, _ROOT)
    manifest = {
        "bundle": BUNDLE, "cells": CELLS,
        "design_record": "ROUND5_DANN_DESIGN.md (not in the repo tree; ships if present)",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": int(version),
        "scripts": names, "n_scripts": len(names),
        "import_closure": [f"{m}.py" for m in clo],
        "cell_requires": list(CELL_REQUIRES),
        "checkpoints": ck_info,
        "rescore_checkpoints": ckpt_parts,
        "n_rescore_checkpoint_parts": len(ckpt_parts),
        "rescore_checkpoints_note": (
            "both finished arms' checkpoint tars, under "
            f"{CKPT_SUBDIR}/, for the re-score that rebuilds the per-checkpoint "
            "TPR/AUROC curves a collection defect dropped. They ride in THIS bundle "
            "and in THIS manifest so the notebook resolves ONE mount: a second dataset "
            "alongside the first is one more thing that can be half-uploaded while "
            "looking attached."),
        "data_requires": list(DATA_REQUIRES),
        "data_trees_required": DATA_TREES_REQUIRED,
        "round5_settings": settings,
        "docs": [],
    }
    docs = []
    for doc in (CELLS, "ROUND5_DANN_DESIGN.md"):
        src = _ROOT / doc
        if src.is_file():
            dst = out / "docs" / Path(doc).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            docs.append(Path(doc).name)
            print(f"  ok       docs/{Path(doc).name}  ({dst.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  note     {doc} not in the repo tree -- not shipped")
    manifest["docs"] = docs

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (out / "VERSION.txt").write_text(
        f"{BUNDLE} v{version}  built {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    (out / "README.md").write_text(README, encoding="utf-8")
    meta = build_metadata(kaggle_user)
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  dataset id      {meta['id']}"
          + ("   <-- rebuild with --kaggle-user to use the Kaggle CLI"
             if meta["id"].startswith("USERNAME/") else ""))
    print(f"\n  MANIFEST.json / VERSION.txt (v{version}) / README.md / "
          f"dataset-metadata.json written")
    return manifest


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify(out: Path) -> int:
    print("\n" + "=" * 78)
    print("VERIFY")
    print("=" * 78)
    fails = []

    def chk(cond, msg):
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        if not cond:
            fails.append(msg)

    mpath = out / "MANIFEST.json"
    chk(mpath.is_file(), "MANIFEST.json present")
    if not mpath.is_file():
        return 1
    m = json.loads(mpath.read_text())

    for n in CELL_REQUIRES:
        chk(n in m["scripts"] and (out / "scripts" / n).is_file(), f"script present: {n}")
    chk(not [c for c in m["import_closure"] if c not in m["scripts"]],
        f"import closure complete ({len(m['import_closure'])} modules)")

    for rel, info in m["checkpoints"].items():
        chk(bool(info.get("present")) and (out / rel).is_file(),
            f"artifact present: {rel}")

    head = out / "runs/bakeoff_baseline_A/checkpoints/step_8000.pt"
    if head.is_file():
        got = head_l2(head)
        chk(abs(got - BASELINE_A_HEAD_L2) < 1e-4,
            f"phase-1 head L2 = {got:.6f} (pinned {BASELINE_A_HEAD_L2:.6f})")

    # ---- the step-0 gate table, RE-DERIVED rather than trusted ----
    curve = out / "runs/bakeoff_baseline_A/curve.csv"
    if curve.is_file():
        import run_dann_r5 as R5
        derived = fold_gate_from_curve(curve)
        bad = [k for k in range(5)
               if abs(derived[k]["tpr"] - R5.FOLD_GATE[k]["tpr"]) > 1e-12
               or abs(derived[k]["auroc"] - R5.FOLD_GATE[k]["auroc"]) > 1e-12]
        chk(not bad,
            f"FOLD_GATE re-derives from curve.csv step=8000 on all 5 folds"
            + (f" -- MISMATCH on {bad}" if bad else
               f" (fold 0 TPR {derived[0]['tpr']:.6f}, AUROC {derived[0]['auroc']:.6f})"))
        chk(len({round(v['tpr'], 12) for v in derived.values()}) > 1,
            "the five folds do NOT share a TPR value -- so the gate can actually "
            "distinguish folds, which is the whole reason it is load-bearing here")
    else:
        chk(False, "runs/bakeoff_baseline_A/curve.csv present (the gate table's source)")

    # ---- THE ARM-1 CONTROL CURVE: usable by arm 2, or it is not shipped ----
    cc = out / "runs/dann_r5_arm1_control/steps.csv"
    if cc.is_file():
        import run_dann_r5 as R5
        import dann_round5 as D5
        rows = list(csv.DictReader(cc.open(newline="")))
        need = ("step", D5.COND3_COLUMN) + D5.ControlCurve.EXTRA_COLUMNS
        have = set(rows[0].keys()) if rows else set()
        miss = [c for c in need if c not in have]
        chk(not miss,
            "the arm-1 control curve carries every column the detectors read"
            + (f" -- MISSING {miss}" if miss else
               f" ({', '.join(need)})"))
        chk(len(rows) == 8000,
            f"the control curve is a COMPLETE 8000-step run ({len(rows)} rows) -- a "
            f"truncated control silently turns matched-step comparison into "
            f"extrapolation")
        # the curve must LOAD through the same class arm 2 will load it with
        try:
            ctrl = D5.ControlCurve(str(cc))
            sm = ctrl.summary()
            chk(True, f"it loads through ControlCurve: steps {sm['step_min']}-"
                      f"{sm['step_max']}, stride {sm['stride_mode']}, "
                      f"{sm['n_steps']} steps, columns "
                      f"{','.join(sm['columns_loaded'])}")
        except Exception as exc:                                    # pragma: no cover
            chk(False, f"ControlCurve FAILED to load the shipped curve: {exc}")
        rc = out / "runs/dann_r5_arm1_control/run_config.json"
        if rc.is_file():
            a1 = json.loads(rc.read_text())
            clean = (a1.get("steps_completed") == 8000
                     and a1.get("n_checkpoints_saved") == 32
                     and not a1.get("aborted")
                     and (a1.get("step0_gate") or {}).get("verdict") == "PASS")
            chk(clean,
                "the run that PRODUCED the control was clean (8000/8000, 32/32, no "
                "abort, gate PASS) -- conditions 3 and 4 were re-specified from it, so "
                "a dirty control would have contaminated both")
    else:
        chk(False, "runs/dann_r5_arm1_control/steps.csv present (arm 2 cannot run "
                   "differential detectors without it)")

    # ---- the re-score's checkpoint tars: listed, present, and correct ----
    parts = m.get("rescore_checkpoints") or []
    if parts:
        by_arm = {}
        bad_p = []
        for rec in parts:
            if rec.get("kind") == "checkpoint_file":
                continue
            if rec.get("kind") != "run_meta":
                by_arm.setdefault(rec["arm"], []).append(rec)
            q = out / rec["path"]
            if not q.is_file():
                bad_p.append(f"{rec['path']}: MISSING")
            elif q.stat().st_size != rec["bytes"]:
                bad_p.append(f"{rec['path']}: WRONG SIZE")
        chk(not bad_p,
            f"all {len(parts)} re-score checkpoint parts are present at the sizes the "
            f"manifest records" + (f" -- {bad_p[:3]}" if bad_p else ""))
        chk(sorted(by_arm) == ["dann_nb3", "dann_nb3_lambda0"],
            f"BOTH arms' checkpoints ship ({ {k: len(v) for k, v in by_arm.items()} }) "
            f"-- arm 1 is the unfreezing-only reference and its curve is not optional")
        cfiles = [r for r in parts if r.get("kind") == "checkpoint_file"]
        by_arm_f = {}
        for r in cfiles:
            by_arm_f.setdefault(r["arm"], []).append(r)
        chk(sorted((k, len(v)) for k, v in by_arm_f.items())
            == [("dann_nb3", 32), ("dann_nb3_lambda0", 32)],
            f"every checkpoint is hashed INDIVIDUALLY, 32 per arm "
            f"({ {k: len(v) for k, v in by_arm_f.items()} }) -- Kaggle auto-extracts "
            f"archives on upload, so the .pt files are what actually land and a per-tar "
            f"checksum cannot be verified on the venue")
        chk(len({r["sha256"] for r in cfiles}) == len(cfiles),
            f"all {len(cfiles)} checkpoint digests are distinct -- 64 identical hashes "
            f"would mean the same file was read 64 times")
        for arm, rows in by_arm_f.items():
            steps = sorted(int(r["basename"][5:-3]) for r in rows)
            chk(steps == list(range(250, 8001, 250)),
                f"{arm}: the 32 steps are exactly 250..8000 by 250, none missing or "
                f"duplicated across the 5 parts")
        metas = [r for r in parts if r.get("kind") == "run_meta"]
        chk(len(metas) == 2 * len(RESCORE_META_FILES),
            f"both arms' {len(RESCORE_META_FILES)} small run files ship too "
            f"({len(metas)} of {2 * len(RESCORE_META_FILES)}) -- nothing has to be "
            f"extracted by hand to run the gate")
        ti = [r for r in metas if r["path"].endswith("true_init.npz")]
        chk(len(ti) == 2,
            "...including BOTH arms' true_init.npz, the step-0 gate's element-wise "
            "reference -- without it the gate cannot run at all")
        parts = [r for r in parts
                 if r.get("kind") not in ("run_meta", "checkpoint_file")]
        tot = sum(r["bytes"] for r in parts)
        chk(tot > 4e9,
            f"re-score checkpoints total {tot / 1e9:.2f} GB (2.3 GB per arm, 32 "
            f"checkpoints each)")
        # names must not collide under a per-arm glob
        for arm in by_arm:
            hits = [r for r in parts
                    if r["path"].split("/")[-1].startswith(arm + "__")]
            chk(len(hits) == len(by_arm[arm]),
                f"a glob for {arm!r} matches exactly its own {len(hits)} parts -- the "
                f"arm prefix keeps 'nb3_fold0' from also matching "
                f"'nb3_lambda0_fold0'")
    else:
        chk(False,
            "runs/dann_arm1 and runs/dann_arm2_fold0 checkpoint tars are staged under "
            f"{CKPT_SUBDIR}/ (the re-score cannot run without them)")

    pre = out / "runs/dann_r5_prereg.json"
    if pre.is_file():
        body = json.loads(pre.read_text())
        got = body.pop("sha256_of_body_without_this_field", None)
        again = hashlib.sha256(json.dumps(body, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        chk(got == again,
            f"the shipped pre-registration SELF-CERTIFIES ({str(got)[:16]}...) -- a "
            f"prereg whose hash does not check is not a prereg")
        chk(str(body.get("two_morphology_instruments", {}).get("status", ""))
            .startswith("SETTLED"),
            "...and the instrument decision is SETTLED: condition 3 reads "
            f"{body.get('abort_conditions', {}).get('3_morph_vs_control', {}).get('detector_column')}")

    # ---- THE ARMED-RUN GUARD. An unset condition-3 margin with abort_mode=armed must
    # refuse. This is checked by EXERCISING the shipped code, not by inspecting a value:
    # the margin is CORRECTLY unset in a pre-arm-1 bundle, so a value check would either
    # always pass or always fail and prove nothing either way.
    try:
        import dann_round5 as _D5
        chk(_D5.COND3_MARGIN is None and _D5.COND4_SRC_MARGIN is None
            and _D5.COND4_TGT_MARGIN is None,
            "all THREE shipped margins are UNSET (cond3 M, cond4 src, cond4 tgt) -- "
            "each is declared within its arm-1 band and hashed before the confirmatory "
            "folds")
        chk(_D5.COND3_COLUMN == "morph_proxy_auroc",
            f"condition 3 reads {_D5.COND3_COLUMN!r}, the FROZEN-FEATURE PROBE. It was "
            f"moved here from morph_head_auroc_pool after arm 1: the head column fell "
            f"to chance AT lambda = 0 (median 0.4948, 54% of readings below 0.5), "
            f"because ML-FP and P1-reserve are BOTH class 0 in the lens task")
        chk(tuple(_D5.COND3_INTERPRETATION_COLUMNS)
            == ("morph_head_auroc_pool", "morph_head_auroc"),
            f"...while the two HEAD columns {list(_D5.COND3_INTERPRETATION_COLUMNS)!r} "
            f"are logged with NO DETECTOR ATTACHED -- kept as a by-eye diagnostic of "
            f"whether the model learned the lens task")
        # the ceilings are FRACTIONS of measured headroom, so a re-specified column
        # carries its ceiling with it. The column has now changed twice; check the
        # fraction, not the number it currently evaluates to.
        f_ = _D5.COND_CEILING_FRACTION
        chk(abs(_D5.COND3_M_CEILING - f_ * _D5.COND3_HEADROOM) < 1e-9
            and abs(_D5.COND4_SRC_CEILING
                    - f_ * (_D5.COND4_SRC_CONTROL_LEVEL - _D5.COND4_PR_FLOOR)) < 1e-9
            and abs(_D5.COND4_TGT_CEILING
                    - f_ * (_D5.COND4_TGT_CONTROL_LEVEL - _D5.COND4_PR_FLOOR)) < 1e-9,
            f"every ceiling is the SAME FRACTION ({f_}) of its measured headroom -- "
            f"cond3 {_D5.COND3_M_CEILING:.4f}, cond4 src {_D5.COND4_SRC_CEILING:.4f}, "
            f"cond4 tgt {_D5.COND4_TGT_CEILING:.4f}. Stated as a fraction so it travels "
            f"if a column is re-specified again")
        chk(_D5.COND4_PR_FLOOR == 1.0 and _D5.COND4_ABSOLUTE_PR_FLOOR_WITHDRAWN == 20.0,
            "condition 4's absolute 'PR < 20' is WITHDRAWN (it fired 7,999 of 8,000 "
            "control steps; the run maximum was 13.40) and headroom is measured to "
            "PR = 1, a fully collapsed representation, not to chance")
        import csv as _csv, tempfile as _tf, numpy as _np
        with _tf.TemporaryDirectory() as _d:
            def _traj(name, jit):
                q = os.path.join(_d, name)
                rng = _np.random.default_rng(0)
                with open(q, "w", newline="") as fh:
                    w_ = _csv.DictWriter(fh, fieldnames=["step", _D5.COND3_COLUMN])
                    w_.writeheader()
                    for st_ in range(0, 8001, 25):
                        w_.writerow({"step": st_,
                                     _D5.COND3_COLUMN: 0.9076 + rng.normal(0, jit)})
                return q
            q1 = _D5.morph_trajectory_stats(_traj("quiet.csv", 0.002))
            q2 = _D5.morph_trajectory_stats(_traj("noisy.csv", 0.150))
        chk(q1["M_DECLARABLE"] is True and q2["M_DECLARABLE"] is False,
            f"the M-declaration criterion is shipped and RESOLVES BOTH WAYS (3J = "
            f"{q1['m_min_admissible']:.4f} vs {q2['m_min_admissible']:.4f} against a "
            f"{_D5.COND3_M_CEILING:.4f} ceiling)")
        chk("UNCALIBRABLE" in q2["verdict"],
            "...and the failing branch reports UNCALIBRABLE as a legitimate outcome")
        # CONDITIONS 3 AND 4 ARE ADVISORY ON EVERY ARM (derived from arms 1+2,
        # 2026-08-30). The armed+unset-margin refusal that used to be checked here
        # protected a safety net that no longer exists, so what is checked now is the
        # invariant that replaced it -- in BOTH directions, because "nothing aborts" and
        # "the right things abort" are different claims.
        chk(set(_D5.ADVISORY_ONLY_DETECTORS)
            == {"abort3_morph_vs_control", "abort3b_morph_sign_flip",
                "abort4src_rank_collapse", "abort4tgt_rank_collapse"},
            f"the advisory set ships as exactly conditions 3 and 4: "
            f"{list(_D5.ADVISORY_ONLY_DETECTORS)}")
        _wa = _D5.DannAbortWatch(mode=_D5.ABORT_MODE_ARMED, verbose=False)
        chk(sorted(_wa.summary()["arming_detectors"])
            == ["abort1_d_overpowering", "abort2_degenerate_collapse"],
            "only conditions 1 and 2 can stop a run; 3 and 4 compute, log, and record "
            "when they WOULD have fired")
        for _n in _D5.ADVISORY_ONLY_DETECTORS:
            _w = _D5.DannAbortWatch(mode=_D5.ABORT_MODE_ARMED, verbose=False)
            for _d in _w.all:
                if _d.name == _n:
                    _d.fired = True
            chk(_w.should_abort() is False and _w.fired() == [_n],
                f"{_n} fires on an ARMED run WITHOUT aborting it, and is still reported")
        _w = _D5.DannAbortWatch(mode=_D5.ABORT_MODE_ARMED, verbose=False)
        _w.c1.fired = True
        chk(_w.should_abort() is True,
            "KNOWN NULL: condition 1 still ABORTS an armed run -- 'advisory' was scoped "
            "to conditions 3 and 4, it did not disarm the whole watch")
        chk(_D5.DannAbortWatch(mode=_D5.ABORT_MODE_LOG_ONLY, cond3_margin=None,
                               verbose=False).cond3_margin is None,
            "...while log_only + unset constructs fine: arms 1 and 2 run BEFORE the "
            "margins exist, which is the point of them")
    except Exception as e:                                        # noqa: BLE001
        chk(False, f"armed-run guard check raised {type(e).__name__}: {e}")

    dz = out / "data.zip"
    if dz.is_file():
        with zipfile.ZipFile(dz) as z:
            arcs = set(z.namelist())
        chk(not any("\\" in a for a in arcs),
            "no backslash arcnames in data.zip (Kaggle rejects those)")
        for need in DATA_REQUIRES:
            chk(need in arcs or any(a.endswith(need) for a in arcs),
                f"data.zip contains {need}")
        for tree, why in DATA_TREES_REQUIRED.items():
            n = sum(1 for a in arcs if a.startswith(tree) and a.endswith(".npz"))
            chk(n > 0, f"data.zip contains {n} shard(s) under {tree} -- {why}")
    else:
        chk(False, "data.zip present")

    s = m["round5_settings"]
    chk(s["lambda_horizon_default"] == 2000,
        f"lambda horizon is {s['lambda_horizon_default']} (brief v2; NOT 8000, and NOT "
        f"the run length)")
    chk(min(s["lambda_at_750_1500_2000_8000"]) >= 0.95,
        f"lambda >= 0.95 at every budget from 750 to 8000 "
        f"(min {min(s['lambda_at_750_1500_2000_8000']):.4f}) -- the read-off step does "
        f"not move the objective")
    chk(s["d_topology"] == [640, 1024, 1024, 1], f"D topology {s['d_topology']}")
    chk(s["d_dropout"] == 0.2 and s["d_dropout_layers"] == 2,
        f"D dropout {s['d_dropout']} in {s['d_dropout_layers']} layers")
    chk(s["d_norm"] == "LayerNorm", "D normalises with LayerNorm, not BatchNorm")
    chk(s["domain_labels"] == {"source": 1.0, "target": 0.0},
        "domain labels are source=1, target=0")
    chk(s["d_holdout_frac"] == 0.10, "D holdout fraction is 0.10")
    chk((s["lr"], s["lr_decay"], s["weight_decay"]) == (1e-4, 0.75, 0.05),
        "Zoobot's published optimiser protocol: lr 1e-4, lr_decay 0.75, wd 0.05 (7.1a)")
    chk(s["batch_size"] == 256, "true batch 256 (7.4 pins it; accumulation is removed)")
    chk(s["n_checkpoints_expected"] == 32,
        f"{s['n_checkpoints_expected']} retained checkpoints at the development budget")
    chk(s.get("morph_every_default") == 25,
        f"--morph-every default is {s.get('morph_every_default')} (brief v3 4.1)")
    chk(s.get("arm_steps") == {"dann_nb3": 8000, "dann_nb3_lambda0": 8000},
        f"BOTH arms declare 8000 steps {s.get('arm_steps')} -- the control is the "
        f"reference trajectory for every monitor (brief v3 5)")
    chk("morph_every" in s.get("pinned_keys", []),
        "morph_every is PINNED: condition 3 step-matches and refuses to interpolate, so "
        "a cadence mismatch deletes the comparison rather than coarsening it")
    chk(s.get("abort_cond3_margin") is None
        and s.get("abort_cond4_src_margin", None) is None
        and s.get("abort_cond4_tgt_margin", None) is None,
        "all three margins are UNSET in the manifest -- declared within their arm-1 "
        "bands and hashed before the confirmatory folds")
    chk(s.get("abort_cond3_column") == "morph_proxy_auroc",
        f"both condition-3 detectors read {s.get('abort_cond3_column')!r}, the FROZEN "
        f"PROBE -- re-specified from arm 1, where the head column fell to chance at "
        f"lambda = 0")
    chk("RE-SPECIFIED" in str(s.get("respecified_from_control_arm", "")),
        "...and the manifest records IN PROSE that conditions 3 and 4 were re-specified "
        "from the control arm and are NOT pre-registered")
    chk(s.get("abort_cond3_median_window_default") == 1,
        f"--abort3-median-window defaults to "
        f"{s.get('abort_cond3_median_window_default')} -- no smoothing unless declared, "
        f"and k is declared BEFORE M")
    chk(s.get("abort_modes") == {"dann_nb3/fold0": "log_only",
                                 "dann_nb3/fold1": "armed",
                                 "dann_nb3_lambda0/fold0": "log_only"},
        f"abort_mode defaults per arm: {s.get('abort_modes')}")
    # Scan the VALUES, not the prose: the manifest SHOULD say in words that 0.55 is
    # withdrawn and why, and a substring scan would fail on the withdrawal note itself.
    def _nums(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            yield float(o)
        elif isinstance(o, dict):
            for v in o.values():
                yield from _nums(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from _nums(v)

    live = list(_nums(s))
    chk(not any(abs(v - 0.55) < 1e-9 or abs(v - 0.6910) < 1e-3 for v in live),
        f"the withdrawn 0.55 (and brief v2's 0.6910 port) survive as no live SETTING "
        f"anywhere -- {len(live)} numeric fields scanned")
    chk("IMPORTANCE WEIGHTING" in str(s.get("withdrawn", "")),
        "...and the manifest records IN PROSE why they are withdrawn -- a number simply "
        "deleted leaves the next reader to reinvent it")

    # ---- the assembled model really is BatchNorm-free, and has ONE optimiser ----
    try:
        import torch
        import dann_round5 as D5
        import run_dann_r5 as R5
        from train_skeleton import TrainConfig, Trainer
        from zoobot_encoder import ZoobotEncoder
        with contextlib.redirect_stdout(io.StringIO()):
            enc = ZoobotEncoder(encoder_mode="frozen", n_blocks=3, device="cpu",
                                verbose=False)
            tr = Trainer(TrainConfig(head_type="linear", encoder_mode="frozen",
                                     n_blocks=3, lr=R5.LR, lr_decay=R5.LR_DECAY,
                                     weight_decay=R5.WEIGHT_DECAY, batch_size=4,
                                     device="cpu", use_coral=False), encoder=enc)
            dann = D5.install_dann(tr, d_lr=R5.LR, weight_decay=R5.WEIGHT_DECAY)
            bn = D5.assert_no_batchnorm(tr.head, tr.encoder.model, dann,
                                        context="bundle verify")
        groups = [g.get("name") for g in tr.opt.param_groups]
        chk(bn["n_batchnorm"] == 0,
            f"the assembled nb3 model has 0 BatchNorm across {bn['n_modules_scanned']} "
            f"modules (head + encoder + D)")
        chk(groups == ["head", "stages.3", "stages.2", "stages.1", "domain_discriminator"],
            f"ONE optimiser, {len(groups)} param groups {groups} -- D is the fifth, at "
            f"the head LR with no lr_decay")
        chk(tr.opt.param_groups[-1]["lr"] == R5.LR
            and tr.opt.param_groups[-1]["weight_decay"] == R5.WEIGHT_DECAY,
            f"D's group: lr {tr.opt.param_groups[-1]['lr']:g}, wd "
            f"{tr.opt.param_groups[-1]['weight_decay']:g}")
        del tr, enc
    except Exception as e:                                        # noqa: BLE001
        chk(False, f"assembled-model check raised {type(e).__name__}: {e}")

    meta_p = out / "dataset-metadata.json"
    chk(meta_p.is_file(), "dataset-metadata.json present")
    if meta_p.is_file():
        md = json.loads(meta_p.read_text())
        uid, _, slug = md.get("id", "").partition("/")
        # NOT a verify failure: dataset-metadata.json matters only to `kaggle datasets
        # create`; a manual upload through the web UI ignores it entirely.
        if not uid or uid == "USERNAME":
            print("  NOTE  dataset-metadata.json still says USERNAME/. That file is used "
                  "ONLY by the Kaggle CLI.\n"
                  "        IGNORE THIS if you upload through the web UI; otherwise "
                  "rebuild with --kaggle-user <handle>.")
        else:
            chk(True, f"dataset id names a Kaggle handle ({md.get('id')!r})")
        chk(3 <= len(slug) <= 50 and all(c.isalnum() or c == "-" for c in slug),
            f"slug {slug!r} is 3-50 chars of [alphanumeric-]")
        chk(6 <= len(md.get("title", "")) <= 50, "title is 6-50 chars")
        chk(len(md.get("licenses", [])) == 1, "exactly one license entry")

    print("\n" + "=" * 78)
    print(f"VERIFY: {len(fails)} failure(s)")
    for f in fails:
        print(f"  FAILED: {f}")
    print("=" * 78)
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# results packaging (the way home)
# ---------------------------------------------------------------------------

# checkpoint tars the RE-SCORE needs on the venue. The bundle proper does not carry
# them: it is the inputs a run needs, and these are the outputs of two runs that already
# happened. They travel as their own dataset so the 4.4 GB bundle is not re-uploaded.
CKPT_SUBDIR = "checkpoints_rescore"

RESCORE_ARM_DIRS = {
    "dann_nb3_lambda0": "runs/dann_arm1",
    "dann_nb3": "runs/dann_arm2_fold0",
}

# the run directory each arm's tars extract INTO, and the small files that must sit
# beside the checkpoints for the re-score to run. Together these are ~350 KB per arm --
# there is no reason to make anyone extract a 186 MB screen_core.tar.gz by hand to get
# them, and a manual step in a paste-and-run notebook is a defect, not an instruction.
RESCORE_RUN_SUBDIR = {
    "dann_nb3_lambda0": "dann_r5_nb3_lambda0_fold0",
    "dann_nb3": "dann_r5_nb3_fold0",
}
RESCORE_META_FILES = (
    ("run_config.json", "n_checkpoints_expected -- the retention check that stops a "
                        "curve being rebuilt from a partial checkpoint set"),
    ("checkpoints.csv", "names one logit file per checkpoint; the packaging "
                        "completeness check reads it"),
    ("logits/ids.npz", "the run's own test-pool id ORDER, pinned so the re-scored "
                       "logits pair row-for-row with the run's"),
    ("logits/true_init.npz", "THE STEP-0 GATE'S REFERENCE -- 26,497 natively produced "
                             "logits, compared element-wise. Without this the gate "
                             "cannot run, and the gate is what makes the other 63 "
                             "checkpoints trustworthy."),
)


def stage_checkpoints(out, root, verbose=True) -> list:
    """Copy both arms' checkpoint tars into ``<bundle>/checkpoints/`` and return records.

    ONE FOLDER, ONE DATASET, ONE MANIFEST. These parts live inside the bundle and are
    listed in the bundle's own MANIFEST.json, so the notebook resolves a single mount.
    A second dataset alongside the first is one more thing that can be half-uploaded
    while looking attached -- which is exactly how the re-score failed at cell 3 with
    every part reported missing.

    Names are ARM-PREFIXED. The two arms' parts are already distinctly named
    (``ckpt_dann_r5_nb3_fold0_*`` vs ``ckpt_dann_r5_nb3_lambda0_fold0_*``), but one is a
    substring of the other, so a glob for one arm would also match the other.

    Every sha256 is carried from the arm's own results MANIFEST and RE-VERIFIED against
    the copied bytes, so a bad copy is caught here rather than after an 8.6 GB upload.
    """
    dest = Path(out) / CKPT_SUBDIR
    entries = []
    missing_arms = []
    for arm, rel in RESCORE_ARM_DIRS.items():
        src = Path(root) / rel
        man_p = src / "MANIFEST.json"
        if not man_p.is_file():
            missing_arms.append(f"{arm} ({man_p})")
            continue
        man = json.loads(man_p.read_text())
        by_name = {q["name"]: q for q in man.get("parts", [])}
        tars = sorted(src.glob("ckpt_*.tar"))
        if not tars:
            missing_arms.append(f"{arm} (no ckpt_*.tar in {src})")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for t in tars:
            rec = by_name.get(t.name)
            if not rec:
                raise SystemExit(f"[stage] ABORT: {t.name} is not listed in {man_p} -- "
                                 f"an unlisted part has no checksum to verify against")
            dst = dest / f"{arm}__{t.name}"
            if not (dst.is_file() and dst.stat().st_size == rec["bytes"]):
                shutil.copy2(t, dst)
            got = sha256_file(dst)
            if got != rec["sha256"]:
                raise SystemExit(f"[stage] ABORT: {dst.name} sha256 {got[:16]} != "
                                 f"{rec['sha256'][:16]} from {man_p}. The copy is bad.")
            entries.append({"arm": arm, "kind": "checkpoint_tar",
                            "path": f"{CKPT_SUBDIR}/{dst.name}",
                            "bytes": rec["bytes"], "sha256": rec["sha256"]})
            if verbose:
                print(f"  ok       {CKPT_SUBDIR}/{dst.name}  "
                      f"({rec['bytes'] / 1e6:.1f} MB)")

        # ---- EVERY CHECKPOINT, HASHED INDIVIDUALLY -------------------------------
        # KAGGLE AUTO-EXTRACTS ARCHIVES ON DATASET UPLOAD. That is not a quirk to work
        # around later -- it is how data.zip has always become <mount>/data/... in every
        # round. It does the same to these tars: the .tar files do not survive, their
        # CONTENTS do. A per-tar checksum is therefore unverifiable on the venue, and
        # cell 2 reporting "10 parts MISSING FROM THE MOUNT" was the check being wrong,
        # not the upload.
        #
        # So each step_N.pt gets its own sha256, streamed out of the tar without
        # extracting it. Then the notebook can verify every checkpoint whether it arrives
        # as a tar or as loose files, and the manifest describes what is actually there.
        for t in tars:
            with tarfile.open(t) as tf:
                for mem in tf.getmembers():
                    if not mem.isfile():
                        continue
                    fh = tf.extractfile(mem)
                    h = hashlib.sha256()
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                    entries.append({
                        "arm": arm, "kind": "checkpoint_file",
                        "member": mem.name,                     # <run_sub>/checkpoints/step_N.pt
                        "basename": os.path.basename(mem.name),
                        "bytes": int(mem.size), "sha256": h.hexdigest(),
                        "from_tar": f"{CKPT_SUBDIR}/{arm}__{t.name}"})
        n_ck = len([e for e in entries
                    if e.get("kind") == "checkpoint_file" and e["arm"] == arm])
        if verbose:
            print(f"  ok       {arm}: {n_ck} checkpoints hashed individually "
                  f"(Kaggle extracts the tars, so the .pt files are what land)")

        # ---- the small run files, so nothing has to be extracted by hand ----
        sub = RESCORE_RUN_SUBDIR[arm]
        ex = src / "extracted" / sub
        for rel, why in RESCORE_META_FILES:
            q = ex / rel
            if not q.is_file():
                raise SystemExit(
                    f"[stage] ABORT: {q} is missing.\n"
                    f"  Needed because: {why}\n"
                    f"  Extract {src / 'screen_core.tar.gz'} into {src / 'extracted'} "
                    f"and re-run the build.")
            d2 = dest / f"{arm}__meta" / rel
            d2.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(q, d2)
            entries.append({"arm": arm, "kind": "run_meta",
                            "path": f"{CKPT_SUBDIR}/{arm}__meta/{rel}",
                            "extract_to": f"{sub}/{rel}",
                            "bytes": d2.stat().st_size,
                            "sha256": sha256_file(d2), "why": why})
        if verbose:
            print(f"  ok       {CKPT_SUBDIR}/{arm}__meta/  "
                  f"({len(RESCORE_META_FILES)} run files)")
    if missing_arms and verbose:
        print(f"  note     re-score checkpoints NOT staged for: {missing_arms}")
        print(f"           (the bundle is still valid for RUNNING an arm; it just "
              f"cannot feed the re-score)")
    return entries


def pack_results(run_dirs, out_dir, logs_dir, max_bytes, extra_files=()) -> int:
    """Independent self-contained tars + a MANIFEST with per-part size and sha256.

    Reuses ``package_screen_results``'s builders verbatim rather than re-implementing
    them -- they already glob (never enumerate by name), which is 7.7's
    stale-filename-list fix, and they already stream-verify reassembly without writing
    a second copy. The core archive keeps that module's fixed ``screen_core.tar.gz``
    name; renaming it here would fork a format two other rounds already read.

    REFUSES an incomplete run. ``run_config.json`` records
    ``n_checkpoints_expected`` / ``n_checkpoints_saved``; if a run dir carries those and
    they disagree, packaging stops. 7.7: the A1 study lost its unfrozen checkpoints to a
    COLLECTION cell that copied one file, and only a check on THIS side would have
    caught it.
    """
    import package_screen_results as PSR

    os.makedirs(out_dir, exist_ok=True)
    run_dirs = [os.path.abspath(d) for d in run_dirs]
    print("=" * 78)
    print("ROUND-5 DANN -- RESULTS PACKAGING")
    print("=" * 78)

    for rd in run_dirs:
        if not os.path.isdir(rd):
            raise SystemExit(f"[pack] ABORT: {rd} is not a directory")
        cfg_p = os.path.join(rd, "run_config.json")
        if os.path.isfile(cfg_p):
            with open(cfg_p) as fh:
                cfg = json.load(fh)
            exp, got = cfg.get("n_checkpoints_expected"), cfg.get("n_checkpoints_saved")
            if exp is not None and got is not None and int(exp) != int(got):
                raise SystemExit(
                    f"[pack] ABORT: {os.path.basename(rd)} saved {got} of {exp} expected "
                    f"checkpoints. Packaging an incomplete run is exactly the failure "
                    f"7.7 records -- fix the run, do not archive it.")
            print(f"  {os.path.basename(rd)}: checkpoints {got}/{exp}, "
                  f"arm={cfg.get('arm')}, steps={cfg.get('steps_completed')}")
        else:
            print(f"  {os.path.basename(rd)}: no run_config.json -- retention UNCHECKED")

    parts, source_hashes = [], {}
    for rd in run_dirs:
        rn = os.path.basename(rd)
        ck = os.path.join(rd, "checkpoints")
        if os.path.isdir(ck):
            tot = sum(os.path.getsize(os.path.join(ck, f)) for f in os.listdir(ck))
            if tot > SMALL_CKPT_LIMIT:
                parts += PSR.build_checkpoint_parts(rd, rn, out_dir, max_bytes)
            for f in os.listdir(ck):
                p = os.path.join(ck, f)
                if os.path.isfile(p):
                    source_hashes[f"{rn}/checkpoints/{f}"] = PSR.sha256_file(p)

    core = PSR.build_core_archive(run_dirs, out_dir, logs_dir, list(extra_files),
                                  SMALL_CKPT_LIMIT)

    # ---- EVERY ARTIFACT THE RUN NAMED MUST BE IN THE ARCHIVE -------------------
    # checkpoints.csv names one logit file per checkpoint, and those logits are the
    # ONLY way the task metric is ever computed: run.json's metric_hygiene says
    # checkpoints are "SCORED (logits persisted) but never MEASURED", so TPR@1e-3 is
    # derived offline from these files and from nothing else.
    #
    # Arms 1 and 2 both packaged cleanly, verified every checksum, and arrived with all
    # 32 logit files MISSING -- build_core_archive listed subdirectories one level deep
    # and the runner writes logits/<arm>_fold<k>/step_N.npz two levels deep. The walk is
    # fixed, but a walk cannot notice what it was never told to expect. This check reads
    # the run's OWN record of what it wrote and refuses to package a run whose named
    # artifacts are not in the tar. 7.7's rule, applied on the side that can act on it.
    members = set(core.get("members") or [])
    for pt in parts:
        members |= set(pt.get("members") or [])
    missing_named = []
    for rd in run_dirs:
        rn = os.path.basename(rd)
        ck_csv = os.path.join(rd, "checkpoints.csv")
        if not os.path.isfile(ck_csv):
            continue
        with open(ck_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                lf = (row.get("logit_file") or "").strip()
                if not lf:
                    continue
                arc = f"{rn}/{lf}"
                if arc not in members:
                    missing_named.append(
                        (arc, "NOT ON DISK" if not os.path.isfile(
                            os.path.join(rd, lf)) else "on disk, NOT ARCHIVED"))
    if missing_named:
        on_disk = [a for a, why in missing_named if why != "NOT ON DISK"]
        raise SystemExit(
            f"[pack] ABORT: {len(missing_named)} artifact(s) named in checkpoints.csv "
            f"are not in the archive.\n  "
            + "\n  ".join(f"{a}  ({why})" for a, why in missing_named[:8])
            + (f"\n  ... and {len(missing_named) - 8} more" if len(missing_named) > 8
               else "")
            + (f"\n\n  {len(on_disk)} of these EXIST ON DISK and were simply not "
               f"collected -- re-run packaging with the fixed archiver rather than "
               f"shipping without them. These logits are the only way the task metric "
               f"is ever computed; a run that arrives without them has to be re-scored "
               f"from its checkpoints, or re-run."
               if on_disk else
               "\n\n  None of these exist on disk, so the RUN did not write them. That "
               "is a runner problem, not a packaging one."))
    n_named = sum(1 for m in members if "/logits/" in m and m.endswith(".npz"))
    print(f"  every artifact named in checkpoints.csv is present "
          f"({n_named} logit file(s) archived)")

    all_parts = parts + [core]
    if parts:
        PSR.verify_parts_against_source(parts, out_dir, source_hashes)

    total = sum(p["bytes"] for p in all_parts)
    manifest = {
        "bundle": f"{BUNDLE}_results",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # `created` and `total_mb` are the key names
        # package_screen_results.verify_downloaded prints. Emitting BOTH spellings keeps
        # that reader working unchanged -- the alternative is a manifest that verifies
        # fine while reporting "packaged None  total None MB".
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_mb": round(total / 1e6, 1),
        "run_dirs": [os.path.basename(d) for d in run_dirs],
        "max_part_bytes": int(max_bytes),
        "parts": all_parts,
        "n_parts": len(all_parts),
        "total_bytes": total,
        "note": ("INDEPENDENT self-contained tars, not `tar | split`. A part that fails "
                 "to download can be re-fetched and verified on its own, and the parts "
                 "extract in any order (7.7). Checkpoints are stored UNCOMPRESSED -- "
                 "float32 weights barely deflate."),
    }
    mp = os.path.join(out_dir, "MANIFEST.json")
    with open(mp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n  {len(all_parts)} part(s), {manifest['total_bytes'] / 1e9:.2f} GB total")
    print(f"  {mp}")
    return 0


def verify_results(out_dir) -> int:
    import package_screen_results as PSR
    return PSR.verify_downloaded(os.path.abspath(out_dir))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(_ROOT / "kaggle" / "dann_r5"))
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--copy-data", action="store_true",
                   help="force a real copy of data.zip instead of a hardlink")
    p.add_argument("--kaggle-user", default=None,
                   help="your Kaggle handle; writes id=<handle>/lens-paper-dann-r5")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--pack-results", nargs="+", default=None, metavar="RUN_DIR",
                   help="package finished run directories for download")
    p.add_argument("--results-out", default=None)
    p.add_argument("--logs", default=None)
    p.add_argument("--max-part-bytes", type=int, default=DEFAULT_MAX_PART_BYTES)
    p.add_argument("--verify-results", default=None, metavar="DIR",
                   help="check a downloaded results set against its MANIFEST")
    a = p.parse_args(argv)

    if a.verify_results:
        return verify_results(a.verify_results)
    if a.pack_results:
        if not a.results_out:
            raise SystemExit("--pack-results requires --results-out")
        return pack_results(a.pack_results, os.path.abspath(a.results_out), a.logs,
                            int(a.max_part_bytes))

    out = Path(a.out).resolve()
    if not a.verify_only:
        build(out, a.version, a.copy_data, a.kaggle_user)
    return 0 if a.no_verify else verify(out)


if __name__ == "__main__":
    sys.exit(main())
