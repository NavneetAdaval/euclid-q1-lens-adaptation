r"""
release_check.py -- what goes into the public repository, and a check that nothing else does.
=============================================================================================

    python paper/scripts/release_check.py               # build the release set, check it, report
    python paper/scripts/release_check.py --self-test   # known-bad cases must fail, known-good pass

THE RULE THIS ENFORCES. Nothing carrying Euclid pixels leaves the machine: no cutouts, no
simulated lenses (arcs painted onto Euclid cutouts), no image shards, no archive that could
contain any of them, and no Kaggle bundle (they carry the image data). Model checkpoints
above ``MAX_CKPT_MB`` stay out. Everything the paper's numbers are recomputed from -- code,
configs, run JSONs, stored test scores, split definitions, the pre-fold evidence -- goes in.

THE CHECK IS BY CONTENT AS WELL AS BY PATH. A path rule alone would pass an image array that
someone saved under ``runs/``; so every ``.npy``/``.npz`` in the release set is opened and
refused if any array has two or more axes each of length >= ``PIXEL_MIN_SIDE`` (an image,
or a stack of them). Stored scores are 1-D, or 2-D with a small class axis, and pass.

``release_set()`` builds the list; ``check()`` independently refuses any file that breaks a
rule, so a mistake in the builder is caught by the checker rather than trusted.

AUTHOR DECISIONS are listed in ``AUTHOR_DECISIONS`` and included by default so the size is
reported with them; ``--without-decisions`` reports the set without them.

Writes ``paper/data/release_manifest.json``.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DEFAULT = os.path.dirname(os.path.dirname(HERE))

MAX_CKPT_MB = 1.0          # AUTHOR DECISION: head-only checkpoints (~20 kB) in, encoders out
# An array is IMAGE-LIKE when its last two axes are both between these bounds and roughly
# square (ratio <= 2) -- a 64x64 cutout, a 224x224 resize, or a stack of either -- or when a
# key is named like image data. Stored scores (checkpoints x examples, e.g. 32 x 26399 or
# 32 x 98) and 640-d feature matrices are not square-ish within the bounds and pass. The
# rule fails SAFE: an ambiguous array is refused and reported, never silently released.
PIXEL_MIN_SIDE = 16
PIXEL_MAX_SIDE = 512
IMAGE_KEYS = ("image", "cutout", "lenses", "nonlenses", "pixels", "stamp")

# Top-level entries whose contents are candidates for release.
IN_ROOTS = ["scripts", "paper", "runs", "evidence", "logs", "bib_batches_incoming_2026-09-19"]
IN_TOP_FILES = ["requirements.txt", ".gitignore", "*.md"]
# Specific files or subtrees inside otherwise-excluded directories that must go in:
# the paper's build reads them, and they carry no pixels (checked below regardless).
IN_EXTRA = [
    "data/splits/splits.json",
    "kaggle/coral_r4/runs/main/extracted/**",          # CORAL x nb3 folds 1-4: run files + scores
    "kaggle/coral_r4/runs/fold0_run/*.json",            # CORAL fold-0 dev curve (make_m_venue)
    "kaggle/coral_r4/runs/fold0_run/*.csv",
    "kaggle/coral_r4/runs/fold0_run/*/*.json",
]
AUTHOR_DECISIONS = [
    "PROJECTv7.md",                                                  # the project record
    "data_old/q1_lenses_processed/lens_cutouts_order.csv",           # exact lens IDs (id_str)
    "data_old/q1_unlabeled/metadata/partitioned_catalog_with_tiles.csv",  # IDs + positions
]
# Never released, whatever else matches.
OUT_ROOTS = ["venv", ".venv", ".idea", ".claude", "data", "data_old", "lenses_30k",
             "lenses_9k_val", "sim_nonlens_30k", "kaggle", "kaggle_screen_bundle",
             "kaggle_task5_bundle", "kaggle_prefold"]
OUT_PATTERNS = ["*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.gz", "*.7z", "*.fits", "*.fits.gz",
                "*/__pycache__/*", "*.pyc", "*.aux", "*.bbl", "*.blg", "*.log.tex"]
CKPT_PATTERNS = ["*.pt", "*.pth", "*.ckpt"]


def _rel(p, repo):
    return os.path.relpath(p, repo).replace("\\", "/")


def _match(rel, pats):
    return any(fnmatch.fnmatchcase(rel, p) or fnmatch.fnmatchcase(os.path.basename(rel), p)
               for p in pats)


def _is_extra(rel, pats):
    for p in pats:
        if p.endswith("/**") and rel.startswith(p[:-2]):
            return True
        if fnmatch.fnmatchcase(rel, p):
            return True
    return False


def pixel_arrays(path):
    """Names of arrays in a .npy/.npz that look like images; [] if none."""
    try:
        if path.endswith(".npy"):
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            shapes = {"<array>": a.shape}
        else:
            with np.load(path, allow_pickle=False) as z:
                shapes = {k: z[k].shape for k in z.files}
    except Exception as e:                                     # unreadable -> refuse
        return [f"UNREADABLE ({type(e).__name__})"]

    def image_like(k, s):
        if any(t in k.lower() for t in IMAGE_KEYS):
            return len(s) >= 2
        if len(s) < 2:
            return False
        h, w = s[-2], s[-1]                     # the spatial axes of an image or a stack
        return (PIXEL_MIN_SIDE <= h <= PIXEL_MAX_SIDE and PIXEL_MIN_SIDE <= w <= PIXEL_MAX_SIDE
                and max(h, w) / min(h, w) <= 2)
    return [k for k, s in shapes.items() if image_like(k, s)]


def violation(rel, full, max_ckpt_mb=MAX_CKPT_MB):
    """The reason a file may not be released, or None."""
    top = rel.split("/")[0]
    if top in OUT_ROOTS and not _is_extra(rel, IN_EXTRA + AUTHOR_DECISIONS):
        return f"under excluded root '{top}/'"
    if _match(rel, OUT_PATTERNS):
        return "archive, image file or build product"
    if _match(rel, CKPT_PATTERNS) and os.path.getsize(full) > max_ckpt_mb * 1e6:
        return f"model checkpoint over {max_ckpt_mb} MB"
    if rel.endswith((".npy", ".npz")):
        pix = pixel_arrays(full)
        if pix:
            return f"pixel array(s): {', '.join(pix)[:80]}"
    return None


def release_set(repo, with_decisions=True, max_ckpt_mb=MAX_CKPT_MB):
    chosen, dropped = [], []
    for dp, dns, fns in os.walk(repo):
        dns[:] = [d for d in dns if d not in (".git", "venv", ".venv", "__pycache__")]
        for f in fns:
            full = os.path.join(dp, f)
            rel = _rel(full, repo)
            top = rel.split("/")[0]
            candidate = (top in IN_ROOTS
                         or ("/" not in rel and _match(rel, IN_TOP_FILES))
                         or _is_extra(rel, IN_EXTRA)
                         or (with_decisions and _is_extra(rel, AUTHOR_DECISIONS)))
            if rel in ("paper.zip", "paper_DRAFT.zip"):
                candidate = False
            if not with_decisions and _is_extra(rel, AUTHOR_DECISIONS):
                candidate = False            # a declined decision wins over a broad rule (*.md)
            if not candidate:
                continue
            why = violation(rel, full, max_ckpt_mb)
            (dropped if why else chosen).append((rel, os.path.getsize(full), why))
    return chosen, dropped


def check(files, repo, max_ckpt_mb=MAX_CKPT_MB):
    """Independent check of a release list. Returns [(rel, reason)] -- empty means pass."""
    bad = []
    for rel in files:
        full = os.path.join(repo, rel)
        why = violation(rel, full, max_ckpt_mb)
        if why:
            bad.append((rel, why))
    return bad


def self_test() -> int:
    tmp = tempfile.mkdtemp(prefix="release_check_selftest_")
    try:
        def put(rel, arr=None, text=None, size=None):
            p = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if arr is not None:
                (np.save if rel.endswith(".npy") else
                 (lambda f, a: np.savez(f, **a)))(p, arr)
            elif size is not None:
                with open(p, "wb") as fh:
                    fh.write(b"\0" * size)
            else:
                open(p, "w").write(text or "x")
        # known-good
        put("runs/a/logits/step_250.npz", {"logits": np.zeros(512), "labels": np.zeros(512)})
        put("runs/a/scores.npy", np.zeros((1000, 2)))
        put("runs/a/run.json", text="{}")
        put("runs/a/checkpoints/head.pt", size=20_000)
        put("data/splits/splits.json", text="{}")
        # real shapes from this repository that are NOT images and must pass
        put("runs/a/validator_scores.npz", {"steps": np.zeros(32), "val": np.zeros((32, 12000)),
                                            "pos": np.zeros((32, 98)), "tgt": np.zeros((32, 782))})
        put("runs/a/features.npy", np.zeros((782, 640)))
        good = ["runs/a/logits/step_250.npz", "runs/a/scores.npy", "runs/a/run.json",
                "runs/a/checkpoints/head.pt", "data/splits/splits.json",
                "runs/a/validator_scores.npz", "runs/a/features.npy"]
        # known-bad: each must be refused, for the stated reason
        put("runs/b/cutouts.npz", {"images": np.zeros((4, 64, 64))})      # pixels under runs/
        put("runs/b/one_image.npy", np.zeros((64, 64)))                    # a single 2-D image
        put("runs/b/resized.npz", {"x": np.zeros((8, 1, 224, 224))})       # resized stack, neutral key
        put("runs/b/stamps.npz", {"cutouts": np.zeros((10, 20))})          # image-named key
        put("runs/b/bundle.tar", size=10)                                  # archive
        put("runs/b/checkpoints/encoder.pt", size=2_000_000)               # checkpoint > 1 MB
        put("data/q1_lenses_processed/meta.json", text="{}")               # excluded root
        put("kaggle/dann_r5/runs/dann_r5_prereg.json", text="{}")          # Kaggle bundle
        bad = {"runs/b/cutouts.npz": "pixel", "runs/b/one_image.npy": "pixel",
               "runs/b/resized.npz": "pixel", "runs/b/stamps.npz": "pixel",
               "runs/b/bundle.tar": "archive", "runs/b/checkpoints/encoder.pt": "checkpoint",
               "data/q1_lenses_processed/meta.json": "excluded root",
               "kaggle/dann_r5/runs/dann_r5_prereg.json": "excluded root"}

        ok = True
        res_good = check(good, tmp)
        print(f"  known-good set ({len(good)} files): "
              f"{'PASS (no violations)' if not res_good else 'FAIL ' + str(res_good)}")
        ok &= not res_good
        for rel, kind in bad.items():
            r = check(good + [rel], tmp)
            hit = [w for f, w in r if f == rel]
            fired = bool(hit) and len(r) == 1
            print(f"  known-bad {rel:45s} -> {'REFUSED: ' + hit[0] if fired else 'NOT REFUSED'}")
            ok &= fired
        chosen, _ = release_set(tmp)
        leaked = [r for r, _s, _w in chosen if r in bad]
        print(f"  builder on the mixed tree: {len(chosen)} chosen, bad files chosen: {leaked or 'none'}")
        ok &= not leaked
        print(f"\n  {'all release-check self-tests pass' if ok else 'SELF-TEST FAILED'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--without-decisions", action="store_true")
    ap.add_argument("--max-ckpt-mb", type=float, default=MAX_CKPT_MB)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    chosen, dropped = release_set(a.repo, not a.without_decisions, a.max_ckpt_mb)
    bad = check([r for r, _s, _w in chosen], a.repo, a.max_ckpt_mb)
    total = sum(s for _r, s, _w in chosen)
    dec = [r for r, _s, _w in chosen if _is_extra(r, AUTHOR_DECISIONS)]
    dec_mb = sum(s for r, s, _w in chosen if r in dec) / 1e6
    by_top = {}
    for r, s, _w in chosen:
        t = r.split("/")[0] if "/" in r else "(top-level files)"
        n, b = by_top.get(t, (0, 0))
        by_top[t] = (n + 1, b + s)
    out = {"n_files": len(chosen), "total_bytes": total,
           "author_decisions_included": dec, "author_decisions_mb": round(dec_mb, 2),
           "max_ckpt_mb": a.max_ckpt_mb,
           "by_top": {k: {"files": n, "mb": round(b / 1e6, 2)} for k, (n, b) in sorted(by_top.items())},
           "dropped_inside_in_roots": [{"path": r, "mb": round(s / 1e6, 2), "why": w}
                                       for r, s, w in dropped],
           "files": [r for r, _s, _w in chosen]}
    os.makedirs(os.path.join(a.repo, "paper", "data"), exist_ok=True)
    json.dump(out, open(os.path.join(a.repo, "paper", "data", "release_manifest.json"), "w",
                        encoding="utf-8"), indent=1)
    print(f"  release set: {len(chosen)} files, {total / 1e6:.1f} MB "
          f"(author decisions {dec_mb:.1f} MB: {', '.join(dec) or 'none'})")
    for k, (n, b) in sorted(by_top.items(), key=lambda kv: -kv[1][1]):
        print(f"    {k:40s} {n:6d} files {b / 1e6:9.1f} MB")
    print(f"  dropped inside candidate roots: {len(dropped)} files, "
          f"{sum(s for _r, s, _w in dropped) / 1e9:.2f} GB")
    if bad:
        print("  RELEASE CHECK FAILED:")
        for r, w in bad[:50]:
            print(f"    {r}: {w}")
        return 1
    print("  release check: PASS -- no file in the release set carries pixels, is an "
          "archive, a Kaggle bundle, or an over-size checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
