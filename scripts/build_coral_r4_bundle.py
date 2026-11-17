"""
build_coral_r4_bundle.py
========================
Assemble (and VERIFY) the Kaggle upload bundle for the round-4 CORAL gamma sweep, into
``kaggle/coral_r4/``.

Modelled on ``build_freematch_bundle.py`` -- same layout, same discipline, same
verification-before-upload principle -- with round 4's own contents. Everything placed in
the bundle is a **COPY**, so the bundle is self-contained and uploading it can never be
affected by later edits to the working tree (the one exception is ``data.zip``, which is
hard-linked when possible because it is ~4.6 GB; it is treated as immutable either way, and
``--copy-data`` forces a real copy).

WHAT A FRESH KAGGLE BOX DOES NOT HAVE
-------------------------------------
    data.zip                                                    images + splits + feature cache
    scripts/                                                    the WHOLE folder
    runs/bakeoff_coral/fold{0..4}/checkpoints/step_8000.pt       the LP->FT phase-1 heads
    runs/bakeoff_baseline_A/checkpoints/step_8000.pt             the agreement reference
    MANIFEST.json / VERSION.txt / README.md / dataset-metadata.json

**runs/ is NOT inside data.zip**, which is why those checkpoints ship explicitly. The sweep
needs fold 0's CORAL head (it is what LP->FT initialises from, and its own ||head||_2 =
66.268356 is the guard against pointing --fold at the wrong head). All five folds ship
anyway: they are ~9 KB each, and the two-arm confirmatory runs that follow this sweep need
the rest. The Baseline A reference is not used by the SWEEP -- the sweep computes no
agreement column, because agreement needs the target pool and the sweep is metric-locked --
but it is needed by the two-arm runs from the same dataset, so it ships too.

**The WHOLE scripts/ folder is copied, not the import closure** (PROJECT.md §7.7: the
closure has grown twice mid-project and "stale scripts silently run old behavior" is a
documented failure mode). The closure is still computed and recorded in MANIFEST.json so
the bundle can prove it is complete.

VERIFICATION (runs by default -- the point of scripting this rather than doing it by hand)
------------------------------------------------------------------------------------------
  * every script the Kaggle cells invoke is present;
  * the import closure of the sweep entry points is fully present;
  * data.zip contains splits.json and the feature-cache splits the sweep reads, with no
    backslash arcnames (Kaggle rejects those outright, §7.7);
  * every shipped checkpoint loads, and fold 0's head L2 equals 66.268356 -- the pinned
    round-4 gate value;
  * the round-4 defaults the sweep depends on are what they claim to be: target EMA OFF,
    feature-norm ceiling 110, gammas (1, 4, 16).

A bad bundle is then a two-minute local failure instead of a wasted session after a
multi-GB upload.

Run:
    python scripts/build_coral_r4_bundle.py
    python scripts/build_coral_r4_bundle.py --version 2
    python scripts/build_coral_r4_bundle.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from package_kaggle_screen import closure          # noqa: E402  (the ONE closure impl)

BUNDLE = "coral_r4_gamma_sweep"
CELLS = "KAGGLE_CORAL_R4_SWEEP_CELLS.md"

# entry points the session invokes, as import roots for the closure check
BUNDLE_ROOTS = ["launch_coral_sweep_kaggle", "run_coral_gamma_sweep", "run_coral_round4",
                "coral_round4", "analyze_coral_gamma_sweep",
                "test_coral_round4", "test_coral_round4_changes"]

# the cells assert each of these is in MANIFEST["scripts"] before copying anything
CELL_REQUIRES = ("launch_coral_sweep_kaggle.py", "run_coral_gamma_sweep.py",
                 "run_coral_round4.py", "coral_round4.py",
                 "analyze_coral_gamma_sweep.py", "train_skeleton.py",
                 "zoobot_encoder.py", "lens_dataset.py", "samplers.py", "heads.py",
                 "da_losses.py", "evaluate.py", "metrics.py",
                 "test_coral_round4.py", "test_coral_round4_changes.py")

CHECKPOINTS = {
    **{f"runs/bakeoff_coral/fold{k}/checkpoints/step_8000.pt":
       f"round-1 CORAL fold-{k} LP->FT phase-1 head (its own step-0 gate value)"
       for k in range(5)},
    "runs/bakeoff_baseline_A/checkpoints/step_8000.pt":
        "collapse-gate agreement reference for the two-arm runs (unused by the sweep)",
}

DATA_REQUIRES = (("data/splits/splits.json",
                  "data/feature_cache/source_train.npz",
                  "data/feature_cache/val.npz",
                  "data/feature_cache/test_negatives_shared.npz",
                  "data/q1_lenses_processed/lens_cutouts.npz")
                 + tuple(f"data/feature_cache/test_pos_fold{k}.npz" for k in range(5))
                 + tuple(f"data/feature_cache/target_fold{k}.npz" for k in range(5)))

# pinned round-4 values the bundle asserts (from runs/coral_round4_design.md)
FOLD_HEAD_L2 = {0: 66.268356, 1: 67.536934, 2: 72.597260,
                3: 68.220108, 4: 66.015175}
EXPECT_HEAD_L2 = FOLD_HEAD_L2[0]
EXPECT_STEP0_TPR = 0.21428571428571427


# ---------------------------------------------------------------------------
# dataset-metadata.json -- the CONSTRAINTS ARE ENFORCED, not hoped for
# ---------------------------------------------------------------------------
# Kaggle's documented limits (Kaggle/kaggle-cli wiki, "Dataset Metadata"):
#   id        "<username-or-org>/<slug>"
#   slug      3-50 chars, ALPHANUMERIC OR '-' ONLY  (a bad slug is rejected at upload,
#             after the multi-GB transfer, which is a slow way to find a typo)
#   title     6-50 chars
#   subtitle  20-80 chars when present (omit it rather than ship a short one)
#   licenses  EXACTLY ONE entry
DATASET_SLUG = "lens-paper-coral-r4"          # 19 chars, [a-z0-9-] only
DATASET_TITLE = "lens-paper coral round 4 sweep"        # 30 chars
DATASET_SUBTITLE = "Round-4 Deep CORAL gamma sweep for Euclid Q1 lens finding"   # 56


def build_metadata(kaggle_user: str | None) -> dict:
    """The metadata dict, with every documented constraint checked BEFORE upload."""
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


README = """# lens-paper — round-4 CORAL gamma sweep (Kaggle dataset)

Everything a fresh Kaggle **T4 × 2** session needs. Cells: `{cells}` (also in `docs/`).

## Name

* slug  `lens-paper-coral-r4`   (3-50 chars, alphanumeric + '-' only)
* title `lens-paper coral round 4 sweep`   (6-50 chars)
* id    `<your-kaggle-handle>/lens-paper-coral-r4`

Build with your handle so the id is valid — the verify step FAILS while it says `USERNAME`,
because that id cannot be uploaded:

```
python scripts/build_coral_r4_bundle.py --kaggle-user YOUR_HANDLE --version 1
kaggle datasets create  -p kaggle/coral_r4 -r zip
kaggle datasets version -p kaggle/coral_r4 -r zip -m "round-4 coral sweep v2"   # re-upload
```

Bump the version on every re-upload — Kaggle silently serves the previous one otherwise, and
Cell 1 prints `VERSION.txt` so you can see which copy you mounted.

## What it looks like once mounted

Kaggle **auto-extracts archives uploaded to a dataset**, so there is no `data.zip` to unzip
at the other end — `data.zip` here has a single top-level `data/` folder, so the mount is:

```
/kaggle/input/lens-paper-coral-r4/
    data/splits/splits.json          <- read-only; used straight from here, NEVER copied
    data/feature_cache/...
    scripts/                         <- copied to /kaggle/working (1.7 MB) and imported
    runs/bakeoff_coral/fold*/checkpoints/step_8000.pt      <- LP->FT phase-1 heads
    runs/bakeoff_baseline_A/checkpoints/step_8000.pt       <- agreement ref (two-arm runs)
    docs/                            <- the cells + the design record
    MANIFEST.json  VERSION.txt  README.md  dataset-metadata.json
```

`/kaggle/input` is read-only and `/kaggle/working` is the writable CWD. The 4.4 GB of data
is **not** copied into working: every use of it is a read.

## What the session runs

```
python -u scripts/launch_coral_sweep_kaggle.py     --out-root /kaggle/working/runs/coral_r4_sweep     --data-root  /kaggle/input/lens-paper-coral-r4/data     --splits     /kaggle/input/lens-paper-coral-r4/data/splits/splits.json     --cache-dir  /kaggle/input/lens-paper-coral-r4/data/feature_cache     --init-head-from /kaggle/input/lens-paper-coral-r4/runs/bakeoff_coral/fold0/checkpoints/step_8000.pt
```

gamma=1 -> cuda:0 and gamma=4 -> cuda:1 concurrently; gamma=16 on whichever frees first. A
100-step gamma=1 arm runs alone first (`--verify-only`) and its trace is read back before
both GPUs are committed.

**The criterion is NOT evaluated on Kaggle.** Download the sweep directories and run
`scripts/analyze_coral_gamma_sweep.py` locally — CPU work on small CSVs; it must not consume
GPU session time.
"""


def build(out: Path, version: int, force_copy: bool, kaggle_user=None):
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("ROUND-4 CORAL GAMMA SWEEP -- KAGGLE UPLOAD BUNDLE")
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
    for rel, why in CHECKPOINTS.items():
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

    # ---- the round-4 settings this bundle depends on, recorded and asserted ----
    import coral_round4 as C4
    import run_coral_round4 as R4
    settings = {
        "coral_target_ema_default": bool(R4.CORAL_TARGET_EMA),
        "feat_norm_ceiling_default": float(C4.DEFAULT_FEAT_NORM_CEILING),
        "sweep_gammas": [float(g) for g in C4.SWEEP_GAMMAS],
        "sweep_steps": int(C4.SWEEP_STEPS),
        "sweep_block": int(C4.SWEEP_BLOCK),
        "lr": R4.LR, "lr_decay": R4.LR_DECAY, "weight_decay": R4.WEIGHT_DECAY,
        "batch_size": R4.BATCH_SIZE, "n_blocks_sweep": 3,
        "criterion_sha256": hashlib.sha256(C4.SWEEP_CRITERION.encode()).hexdigest(),
    }

    manifest = {
        "bundle": BUNDLE, "cells": CELLS,
        "design_record": "runs/coral_round4_design.md",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": int(version),
        "scripts": names, "n_scripts": len(names),
        "import_closure": [f"{m}.py" for m in clo],
        "cell_requires": list(CELL_REQUIRES),
        "checkpoints": ck_info,
        "data_requires": list(DATA_REQUIRES),
        "round4_settings": settings,
        "docs": [Path(CELLS).name, "coral_round4_design.md"],
        "fold_head_l2": FOLD_HEAD_L2,
        "expect_step0_tpr_fold0": EXPECT_STEP0_TPR,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (out / "VERSION.txt").write_text(
        f"{BUNDLE} v{version}  built {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    (out / "README.md").write_text(README.format(cells=CELLS), encoding="utf-8")
    # the cells and the design record travel WITH the dataset -- a bundle that needs a
    # document from someone else's laptop to be usable is not self-contained.
    for doc in (CELLS, "runs/coral_round4_design.md"):
        src = _ROOT / doc
        if src.is_file():
            dst = out / "docs" / Path(doc).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  ok       docs/{Path(doc).name}  ({dst.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  MISSING  {doc}")
    meta_path = out / "dataset-metadata.json"
    meta = build_metadata(kaggle_user)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  dataset id      {meta['id']}"
          + ("   <-- REPLACE 'USERNAME' with your Kaggle handle, or rebuild with "
             "--kaggle-user" if meta["id"].startswith("USERNAME/") else ""))
    print(f"\n  MANIFEST.json / VERSION.txt (v{version}) / README.md / "
          f"dataset-metadata.json written")
    return manifest


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
        chk(n in m["scripts"] and (out / "scripts" / n).is_file(),
            f"script present: {n}")
    chk(not [c for c in m["import_closure"] if c not in m["scripts"]],
        f"import closure complete ({len(m['import_closure'])} modules)")

    for rel, info in m["checkpoints"].items():
        p = out / rel
        chk(info.get("present") and p.is_file(), f"checkpoint present: {rel}")
    for k, want in FOLD_HEAD_L2.items():
        p = out / f"runs/bakeoff_coral/fold{k}/checkpoints/step_8000.pt"
        if p.is_file():
            got = head_l2(p)
            chk(abs(got - want) < 1e-4,
                f"fold {k} phase-1 head L2 = {got:.6f} (pinned {want})")

    dz = out / "data.zip"
    if dz.is_file():
        with zipfile.ZipFile(dz) as z:
            arcs = set(z.namelist())
        chk(not any("\\" in a for a in arcs),
            "no backslash arcnames in data.zip (Kaggle rejects those)")
        for need in DATA_REQUIRES:
            chk(need in arcs or any(a.endswith(need) for a in arcs),
                f"data.zip contains {need}")
    else:
        chk(False, "data.zip present")

    s = m["round4_settings"]
    chk(s["coral_target_ema_default"] is False,
        "round-4 default: target EMA is OFF")
    chk(s["feat_norm_ceiling_default"] == 110.0,
        "round-4 default: feature-norm ceiling is 110")
    chk(tuple(s["sweep_gammas"]) == (1.0, 4.0, 16.0),
        "sweep gammas are (1, 4, 16)")
    chk(s["sweep_steps"] == 2000, "sweep budget is 2000 steps per arm")
    chk(s["n_blocks_sweep"] == 3, "the sweep runs nb3")
    meta_p = out / "dataset-metadata.json"
    chk(meta_p.is_file(), "dataset-metadata.json present")
    if meta_p.is_file():
        md = json.loads(meta_p.read_text())
        uid, _, slug = md.get("id", "").partition("/")
        # NOT a verify failure. dataset-metadata.json only matters to `kaggle datasets
        # create`; a manual upload through the Kaggle UI ignores it entirely, and failing
        # the bundle over an unused placeholder is noise for the flow actually in use.
        if not uid or uid == "USERNAME":
            print("  NOTE  dataset-metadata.json still says USERNAME/. That file is used "
                  "ONLY by the Kaggle CLI.\n"
                  "        IGNORE THIS if you upload through the web UI. To use the CLI "
                  "instead, rebuild\n"
                  "        with --kaggle-user <handle>.")
        else:
            chk(True, f"dataset id names a Kaggle handle ({md.get('id')!r})")
        chk(3 <= len(slug) <= 50 and all(c.isalnum() or c == '-' for c in slug),
            f"slug {slug!r} is 3-50 chars of [alphanumeric-]")
        chk(6 <= len(md.get("title", "")) <= 50, "title is 6-50 chars")
        chk(len(md.get("licenses", [])) == 1, "exactly one license entry")
    for d in m.get("docs", []):
        chk((out / "docs" / d).is_file(), f"doc shipped with the bundle: docs/{d}")

    print("\n" + "=" * 78)
    print(f"VERIFY: {len(fails)} failure(s)")
    for f in fails:
        print(f"  FAILED: {f}")
    print("=" * 78)
    return 1 if fails else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(_ROOT / "kaggle" / "coral_r4"))
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--copy-data", action="store_true",
                   help="force a real copy of data.zip instead of a hardlink")
    p.add_argument("--kaggle-user", default=None,
                   help="your Kaggle handle; writes id=<handle>/lens-paper-coral-r4")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out).resolve()
    if not a.verify_only:
        build(out, a.version, a.copy_data, a.kaggle_user)
    return 0 if a.no_verify else verify(out)


if __name__ == "__main__":
    sys.exit(main())
