"""
build_freematch_bundle.py
=========================
Assemble the Kaggle upload bundle for the FreeMatch x n_blocks=3 fold-0 diagnostic
(`KAGGLE_FREEMATCH_NB3_CELLS.md`), and VERIFY it before 4.6 GB goes over the wire.

WHAT THE SESSION NEEDS THAT A FRESH KAGGLE BOX DOES NOT HAVE
------------------------------------------------------------
    data.zip                                            the images + splits + feature cache
    scripts/                                            the whole folder (see below)
    runs/bakeoff_freematch_5fold/fold0/checkpoints/step_8000.pt   the LP->FT phase-1 head
    runs/bakeoff_baseline_A/checkpoints/step_8000.pt              the agreement reference
    MANIFEST.json                                       what Cell 1 copies and asserts on
    VERSION.txt                                         proof the mount is the fresh upload

**runs/ is NOT packed into data.zip**, which is exactly why those two checkpoints have to be
shipped explicitly. They are ~17 KB each and the run cannot start without either: the first
is the head both arms initialise from (and whose own fold-0 score, 0.387755, is the step-0
hard gate), the second is the fixed vector the collapse-gate agreement column is measured
against -- and agreement is only comparable across runs if every run uses the SAME file.

**The WHOLE scripts/ folder is copied, not the import closure.** PROJECT.md §7.7: the
closure has already grown twice mid-project (16 -> 20 -> more files) and "stale scripts
silently run old behavior" is a documented failure mode. The closure is still COMPUTED here
and recorded in MANIFEST.json as `import_closure`, so the bundle can prove it is complete --
but what ships is everything, because 2.7 MB against 4.6 GB is not a trade worth making.

**data.zip is HARD-LINKED, not copied**, when the bundle lands on the same volume. It is
4.64 GB and this machine has ~33 GB free; a hardlink is a second directory entry for the
same bytes, so any reader (Explorer, the Kaggle CLI, a browser upload) sees an ordinary
file while costing zero extra space. `--copy-data` forces a real copy if the bundle is being
built onto another drive. data.zip is treated as immutable either way.

VERIFICATION (runs by default; the point of building this with a script rather than by hand)
--------------------------------------------------------------------------------------------
Replays Cell 1's assertions against the assembled bundle: every MANIFEST script present, the
five feature-cache splits and splits.json inside data.zip, no backslash arcnames (§7.7 --
Kaggle rejects those outright), both checkpoints loadable, and the phase-1 head's L2 norm
equal to 29.414692. A bad bundle is then a two-minute local failure instead of a wasted
session after a multi-GB upload.

Run:
    python scripts/build_freematch_bundle.py
    python scripts/build_freematch_bundle.py --out kaggle/freematch --version 2
    python scripts/build_freematch_bundle.py --verify-only
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

# the driver + everything the notebook cells invoke, as import roots
BUNDLE_ROOTS = ["run_unfreeze_freematch", "unfreeze_instrument",
                "report_unfreeze_freematch", "pseudolabel_precision",
                "collapse_gate", "package_screen_results",
                "test_unfreeze_freematch", "probe_t4_freematch"]

# Cell 1 asserts each of these is in MANIFEST["scripts"] before it copies anything
CELL1_REQUIRES = ("run_unfreeze_freematch.py", "unfreeze_instrument.py",
                  "report_unfreeze_freematch.py", "pseudolabel_precision.py",
                  "run_freematch.py", "live_encode.py", "curve_hook.py",
                  "package_screen_results.py")

# ALL FIVE folds' phase-1 heads, at the paths the per-fold Cell 1 globs for.
# Each fold continues from ITS OWN round-1 checkpoint (PROJECT.md §7.11), so shipping only
# fold 0's -- which this bundle did until v5 -- strands every other fold at Cell 1.
CHECKPOINTS = {
    **{f"runs/bakeoff_freematch_5fold/fold{k}/checkpoints/step_8000.pt":
       f"LP->FT phase-1 head for FOLD {k} (its own step-0 gate value)"
       for k in range(5)},
    "runs/bakeoff_baseline_A/checkpoints/step_8000.pt":
        "collapse-gate agreement reference (fixed vector; identity matters across runs)",
}

# what data.zip must contain for the cells to run -- every fold's eval splits, since the
# session that runs fold k reads test_pos_fold{k} / target_fold{k}
DATA_REQUIRES = ("data/splits/splits.json",
                 "data/feature_cache/source_train.npz",
                 "data/feature_cache/val.npz",
                 "data/feature_cache/test_negatives_shared.npz",
                 "data/q1_lenses_processed/lens_cutouts.npz"
                 ) + tuple(f"data/feature_cache/test_pos_fold{k}.npz" for k in range(5)
                           ) + tuple(f"data/feature_cache/target_fold{k}.npz"
                                     for k in range(5))

# PRE-REGISTERED per-fold gate table (§7.11) -- the bundle verifies every head it ships
FOLD_HEAD_L2 = {0: 29.414692, 1: 29.280975, 2: 29.292334, 3: 29.465052, 4: 29.237820}
EXPECT_HEAD_L2 = FOLD_HEAD_L2[0]
EXPECT_STEP0_TPR = 0.3877551020408163


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
    """Hard-link when possible; fall back to a real copy. Returns the mode used."""
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "already present"
        dst.unlink()
    if not force_copy:
        try:
            os.link(src, dst)
            return "hardlink (0 bytes extra)"
        except OSError:
            pass                      # different volume, or FS without hardlinks
    t0 = time.time()
    shutil.copy2(src, dst)
    return f"copied in {time.time() - t0:.0f}s"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(out: Path, version: int, force_copy: bool):
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("FREEMATCH x nb3 -- KAGGLE UPLOAD BUNDLE")
    print("=" * 78)
    print(f"  out: {out}")

    # ---- scripts: the WHOLE folder (see the module docstring) ----
    sdir = out / "scripts"
    sdir.mkdir(exist_ok=True)
    names = sorted(p.name for p in _HERE.glob("*.py"))
    for n in names:
        shutil.copy2(_HERE / n, sdir / n)
    print(f"\n  scripts/        {len(names)} .py files "
          f"({sum((sdir / n).stat().st_size for n in names) / 1e6:.1f} MB)")

    clo = closure(BUNDLE_ROOTS)
    missing_clo = [f"{m}.py" for m in clo if f"{m}.py" not in names]
    print(f"  import closure  {len(clo)} modules from {len(BUNDLE_ROOTS)} roots"
          + (f"  *** MISSING {missing_clo} ***" if missing_clo else "  (all present)"))

    # ---- the two run checkpoints, at the paths Cell 1 globs for ----
    ck_info = {}
    print()
    for rel, why in CHECKPOINTS.items():
        src = _ROOT / rel
        dst = out / rel
        if not src.is_file():
            print(f"  MISSING  {rel}\n           needed as: {why}")
            ck_info[rel] = {"present": False}
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        ck_info[rel] = {"present": True, "bytes": dst.stat().st_size,
                        "sha256": sha256_file(dst), "why": why}
        print(f"  ok       {rel}  ({dst.stat().st_size / 1024:.1f} KB, "
              f"sha256 {ck_info[rel]['sha256'][:16]})")

    # ---- data.zip ----
    dz = _ROOT / "data.zip"
    print()
    if dz.is_file():
        mode = link_or_copy(dz, out / "data.zip", force_copy=force_copy)
        print(f"  ok       data.zip  ({dz.stat().st_size / 1e9:.2f} GB, {mode})")
    else:
        print(f"  MISSING  data.zip -- build it with: python scripts/make_data_zip.py")

    # ---- MANIFEST / VERSION / dataset metadata / README ----
    manifest = {
        "bundle": "freematch_nb3_fold0",
        "brief": "BRIEF_freematch_nb3_fold0.md",
        "cells": "KAGGLE_FREEMATCH_NB3_CELLS.md",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": int(version),
        "scripts": names,
        "n_scripts": len(names),
        "import_closure": [f"{m}.py" for m in clo],
        "roots": BUNDLE_ROOTS,
        "checkpoints": ck_info,
        "expect_head_l2": EXPECT_HEAD_L2,
        "expect_step0_tpr": EXPECT_STEP0_TPR,
        "note": ("The whole scripts/ folder ships, not just the import closure: the closure "
                 "has grown twice mid-project and stale scripts silently running old "
                 "behaviour is a documented failure mode (PROJECT.md 7.7)."),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "VERSION.txt").write_text(
        f"freematch_nb3_fold0 v{version}  built {manifest['built_utc']}\n")
    print(f"\n  ok       MANIFEST.json ({len(names)} scripts)  +  VERSION.txt (v{version})")

    meta = {"title": "lens-paper freematch nb3 fold0",
            "id": "USERNAME/lens-paper-freematch-nb3",
            "licenses": [{"name": "CC0-1.0"}]}
    mp = out / "dataset-metadata.json"
    if not mp.exists():
        mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  ok       dataset-metadata.json  "
              f"(EDIT 'id': replace USERNAME with your Kaggle username)")
    else:
        print(f"  kept     dataset-metadata.json (already present, not overwritten)")

    (out / "README.md").write_text(README.format(version=version), encoding="utf-8")
    print(f"  ok       README.md")
    return manifest


# ---------------------------------------------------------------------------
# verify -- replays Cell 1's assertions against the assembled bundle
# ---------------------------------------------------------------------------

def verify(out: Path):
    print("\n" + "=" * 78)
    print("VERIFY -- replaying Cell 1's assertions locally")
    print("=" * 78)
    problems = []

    def check(ok, msg):
        print(f"  [{'ok  ' if ok else 'FAIL'}] {msg}")
        if not ok:
            problems.append(msg)

    mp = out / "MANIFEST.json"
    check(mp.is_file(), "MANIFEST.json present")
    if not mp.is_file():
        return problems
    man = json.loads(mp.read_text(encoding="utf-8"))

    for n in CELL1_REQUIRES:
        check(n in man["scripts"], f"MANIFEST lists {n}")
    missing = [n for n in man["scripts"] if not (out / "scripts" / n).is_file()]
    check(not missing, f"every MANIFEST script is on disk"
                       + (f" (missing {missing[:5]})" if missing else ""))

    check((out / "VERSION.txt").is_file(), "VERSION.txt present")

    dz = out / "data.zip"
    check(dz.is_file(), "data.zip present")
    if dz.is_file():
        with zipfile.ZipFile(dz) as z:
            names = set(z.namelist())
        # §7.7: Windows os.path.join-built archives embed backslashes as literal filename
        # characters, which Kaggle rejects outright and Linux unzip mangles.
        bad = [n for n in names if "\\" in n]
        check(not bad, f"no backslash arcnames in data.zip ({len(bad)} found)")
        for n in DATA_REQUIRES:
            check(n in names, f"data.zip contains {n}")

    for rel, why in CHECKPOINTS.items():
        p = out / rel
        check(p.is_file(), f"{rel} present ({why[:40]}...)")
        if not p.is_file():
            continue
        src = _ROOT / rel
        if src.is_file():
            check(sha256_file(p) == sha256_file(src),
                  f"{Path(rel).parent.parent.name} checkpoint matches the repo copy")

    # EVERY fold's head, against the pre-registered per-fold norm. Checking only fold 0 --
    # which this did until v5 -- would pass a bundle that strands folds 1-4 at Cell 1.
    seen = {}
    for k, want in sorted(FOLD_HEAD_L2.items()):
        head = out / f"runs/bakeoff_freematch_5fold/fold{k}/checkpoints/step_8000.pt"
        if not head.is_file():
            check(False, f"fold {k} phase-1 head MISSING from the bundle")
            continue
        try:
            l2 = head_l2(head)
            check(abs(l2 - want) < 1e-4,
                  f"fold {k} phase-1 head ||.||_2 = {l2:.6f} (pre-registered {want})")
            seen[k] = round(l2, 6)
        except Exception as e:
            check(False, f"fold {k} head failed to load: {type(e).__name__}: {e}")
    check(len(set(seen.values())) == len(seen),
          f"all {len(seen)} shipped heads are DISTINCT -- no fold can pass another's gate")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\n  bundle size: {total / 1e9:.2f} GB "
          f"({sum(1 for p in out.rglob('*') if p.is_file())} files)")
    return problems


README = """\
# Kaggle upload bundle — FreeMatch × n_blocks=3, fold-0 diagnostic (v{version})

Everything `KAGGLE_FREEMATCH_NB3_CELLS.md` needs. Built by
`scripts/build_freematch_bundle.py`, which also verifies it — rebuild rather than
hand-editing, so MANIFEST.json and the contents cannot drift apart.

```
MANIFEST.json     what Cell 1 reads and asserts on
VERSION.txt       bump every re-upload; Cell 1 prints it to prove the mount is fresh
data.zip          images + splits + feature cache  (~4.6 GB)
scripts/          the whole folder, not just the import closure
runs/
  bakeoff_freematch_5fold/fold0/checkpoints/step_8000.pt   LP→FT phase-1 head
  bakeoff_baseline_A/checkpoints/step_8000.pt              agreement reference
```

`runs/` is deliberately outside `data.zip` and therefore shipped explicitly — both
checkpoints are ~17 KB and the session cannot start without either.

## Upload

Edit `dataset-metadata.json` and replace `USERNAME` with your Kaggle username, then:

```bash
kaggle datasets create -p kaggle/freematch -r zip
```

`-r zip` preserves `scripts/` and `runs/` as directories, which the notebook's globs
depend on. For a re-upload after changing anything:

```bash
kaggle datasets version -p kaggle/freematch -r zip -m "v{version}"
```

Bump `--version` on the builder each time so `VERSION.txt` changes — Kaggle will serve
the previous version silently if the bump did not take, and stale scripts running old
behaviour is a documented failure mode (PROJECT.md §7.7).

Browser upload works too (drag the folder in), but it is slow and flaky at ~4.6 GB.

## Session settings

Accelerator **GPU T4 × 2**, Internet **ON**. Only one GPU is used — the runs are
sequential, control first — but the two-GPU sessions carry the larger CPU/RAM
allocation, and Cell 2 reports `os.cpu_count()` so the worker count is grounded in what
the session actually has.

## Verify before uploading

```bash
python scripts/build_freematch_bundle.py --verify-only
```
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_ROOT / "kaggle" / "freematch"))
    ap.add_argument("--version", type=int, default=1,
                    help="stamp written to VERSION.txt; BUMP IT on every re-upload")
    ap.add_argument("--copy-data", action="store_true",
                    help="force a real copy of data.zip instead of a hard link (needed "
                         "only when the bundle is on a different volume)")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    if not args.verify_only:
        build(out, args.version, args.copy_data)
    problems = verify(out)

    print("\n" + "=" * 78)
    if problems:
        print(f"BUNDLE INCOMPLETE -- {len(problems)} problem(s). Do NOT upload:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("BUNDLE VERIFIED -- ready to upload.")
    print(f"  edit {out / 'dataset-metadata.json'} (set your Kaggle username), then")
    print(f"  kaggle datasets create -p {out} -r zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
