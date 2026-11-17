"""
package_kaggle_screen.py
========================
Build the Kaggle upload bundle for the LEVER-1 fold-0 screen (Task 3).

WHAT KAGGLE NEEDS THAT A FRESH SESSION DOES NOT HAVE
----------------------------------------------------
``run_screen_unfreeze.py`` refuses to start nb1/nb3 until (a) every pinned setting matches
the reference nb0_image run and (b) the control gate passes. Both checks read local run
history, which a fresh Kaggle session has none of. This packages exactly what those two
checks need:

    control_scores/nb0_image_fold0.npz    weak-aug/fp16 control  (WITH pos_ids/neg_ids)
    control_scores/nb0_cached_fold0.npz   round-1 reference      (WITH pos_ids/neg_ids)
    control_scores/nb0_image_noaug_fp32_fold0.npz   isolating control (context)
    reference/run_config.json             the nb0_image config the §1 check pins against

The ids are NOT optional: without them the row-alignment check cannot run and the gate
fails closed (correct, but a wasted session to rediscover).

ALSO writes a manifest listing the exact script files the Kaggle dataset version must
contain, computed from the real import closure rather than guessed.

Run:
    python scripts/package_kaggle_screen.py --out kaggle_screen_bundle
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# roots whose transitive local-import closure must be uploaded
KAGGLE_ROOTS = ["run_screen_unfreeze", "report_screen_unfreeze", "collapse_gate",
                "analyze_fold0_selection", "probe_t4_freematch",
                # the pre-launch timing cell measures REAL T4 step times before committing
                # 8000-step runs, instead of extrapolating from the local Ada card.
                "bench_unfreeze_steptime",
                # Task 5: the end-of-session packager. It runs ON Kaggle, so it must be in
                # the upload closure -- the whole point is that all 32 checkpoints per run
                # come home in verifiable parts.
                "package_screen_results"]


def local_modules():
    return {p.stem for p in _HERE.glob("*.py")}


def imports_of(mod, local):
    tree = ast.parse((_HERE / f"{mod}.py").read_text(encoding="utf-8"), f"{mod}.py")
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in local:
                    out.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module and n.level == 0 and n.module.split(".")[0] in local:
                out.add(n.module.split(".")[0])
    return out


def closure(roots):
    local = local_modules()
    seen, stack = set(), list(roots)
    while stack:
        m = stack.pop()
        if m in seen or m not in local:
            continue
        seen.add(m)
        stack.extend(imports_of(m, local) - seen)
    return sorted(seen)


def _check_ids(path):
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("pos_scores", "neg_scores", "pos_ids", "neg_ids")
                   if k not in z.files]
        n = (z["pos_scores"].size, z["neg_scores"].size) if not missing else None
    return missing, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_ROOT / "kaggle_screen_bundle"))
    ap.add_argument("--version", type=int, default=1,
                    help="stamp written to VERSION.txt; bump it every re-upload so Cell 1 "
                         "can prove the mount is fresh")
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "control_scores").mkdir(parents=True, exist_ok=True)
    (out / "reference").mkdir(parents=True, exist_ok=True)

    runs = _ROOT / "runs"
    sources = {
        "nb0_image_fold0.npz":
            runs / "baselineA_fold0_nb0_image" / "scores"
                 / "baselineA_fold0_nb0_image_fold0.npz",
        "nb0_cached_fold0.npz":
            runs / "bakeoff_baseline_A" / "scores" / "step_8000_fold0.npz",
        "nb0_image_noaug_fp32_fold0.npz":
            runs / "baselineA_fold0_nb0_image_noaug_fp32" / "scores"
                 / "baselineA_fold0_nb0_image_fold0.npz",
    }

    print("=" * 78)
    print("KAGGLE SCREEN BUNDLE")
    print("=" * 78)
    problems = []
    for name, src in sources.items():
        if not src.is_file():
            print(f"  MISSING  {name}  <- {src}")
            problems.append(name)
            continue
        missing, n = _check_ids(src)
        if missing:
            print(f"  BAD      {name}: missing keys {missing} -- the control gate will "
                  f"fail closed without ids")
            problems.append(name)
            continue
        shutil.copy2(src, out / "control_scores" / name)
        print(f"  ok       {name}  (pos={n[0]}, neg={n[1]}, ids present)")

    ref_cfg = runs / "baselineA_fold0_nb0_image" / "run_config.json"
    if ref_cfg.is_file():
        shutil.copy2(ref_cfg, out / "reference" / "run_config.json")
        cfg = json.loads(ref_cfg.read_text())
        print(f"  ok       reference/run_config.json  "
              f"(precision={cfg.get('precision')}, aug={cfg.get('augmentation')}, "
              f"seed={cfg.get('seed')}, batch={cfg.get('batch_size')})")
        if "precision" not in cfg:
            print("  NOTE: the reference config predates the 'precision' key; the §1 check "
                  "will compare it as '<absent>'. Re-emit it or accept the mismatch "
                  "explicitly -- do NOT loosen the check.")
    else:
        print(f"  MISSING  reference/run_config.json <- {ref_cfg}")
        problems.append("run_config.json")

    # copy the script closure INTO the bundle so the whole directory is uploadable as-is
    files = closure(KAGGLE_ROOTS)
    sdir = out / "scripts"
    sdir.mkdir(parents=True, exist_ok=True)
    for m in files:
        shutil.copy2(_HERE / f"{m}.py", sdir / f"{m}.py")

    manifest = {"scripts": [f"{m}.py" for m in files],
                "n_scripts": len(files),
                "control_scores": sorted(p.name for p in
                                         (out / "control_scores").glob("*.npz")),
                "roots": KAGGLE_ROOTS}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    # VERSION stamp: Cell 1 prints it to PROVE the mounted copy is the one just uploaded.
    # Kaggle silently serves the previous version if the bump did not take, and stale
    # scripts running old behaviour is a documented failure mode here (PROJECT.md 7.7).
    ver = int(args.version) if args.version is not None else 1
    (out / "VERSION.txt").write_text(f"v{ver}\n")

    print(f"\n  import closure for {KAGGLE_ROOTS}: {len(files)} script files")
    for m in files:
        print(f"     {m}.py")
    print(f"\n[out] {out}")
    if problems:
        print(f"\nINCOMPLETE: {problems}")
        return 1
    print("\nBundle complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
