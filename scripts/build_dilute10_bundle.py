"""
build_dilute10_bundle.py
========================
The Kaggle bundle for the 1:10 diluted-target experiment (brief_dilute10_launch section 2),
in the pattern of ``build_dann_r5_bundle.py``: one directory the Kaggle dataset is made from,
a ``MANIFEST.json`` the cells find BY CONTENT (``bundle == "dilute10"``) and verify file by
file before anything runs.

Layout (paths with forward slashes, ``arcname`` explicit wherever an archive is written):

  MANIFEST.json  VERSION.txt  README.md
  scripts/                       the whole scripts/ folder (*.py)
  data.zip                       HARD-LINKED from kaggle/dann_r5/data.zip after verification:
                                 splits.json byte-identical to the repo's; every image tree
                                 the runs read (source sims, val, test, grade-X and P1
                                 probe shards, the 250 unlabeled shards incl. the P2 shards
                                 holding the 14,085 fresh galaxies, lens cutouts) and the
                                 frozen feature cache
  bundle_data/splits/            dilute10_pool.json (the diluted pool; protected; hash =
                                 part 1) and d1_test_neg_remove.json. NOT ``data/``: Kaggle
                                 extracts every uploaded zip into a folder named after its
                                 stem, so ``data.zip`` becomes ``data/`` and a plain ``data/``
                                 folder collides ("Directory already exists: data", 2026-10-14).
                                 The builder refuses any top-level folder named like a zip stem.
  runs/                          dilute10_prereg_part1/2.json, protected_artifacts.json,
                                 Baseline A's head, the round-5/6 artifacts the DANN and BNM
                                 runners read, and runs/dilute10/coral_frozen/fold{1..4}/
                                 {checkpoints/step_8000.pt, gate_values.json, run_config.json}
  reference/                     what run_coral_r4_confirm.py resolves by search: the hashed
                                 coral_r4_budget.json, the fold-0 development run.json, the
                                 reference ids

The Zoobot encoder is NOT shipped: no earlier bundle shipped it; the runners fetch it from
the Hugging Face hub (internet on).

    python -u scripts/build_dilute10_bundle.py --out kaggle/dilute10 --version 1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

BUNDLE = "dilute10"
CELLS = "KAGGLE_DILUTE10.md"
DATA_ZIP_SRC = os.path.join(REPO, "kaggle", "dann_r5", "data.zip")
PART1 = "bd537a025e81833e73713649c9eed68fd6a0fd3152ee29240f9a333f7c649d85"
PART2 = "b8e09197a204be835fa089c858b08c7eb3e7887bfa19afd296d544d04f394503"
PART3 = "1b583ea2ae5056edb410040c32613d9388e5a994bf90fc366fc93d293efb596a"   # PROJECTv7 7.48
POOL = "18c73d6c1cbad99e61d43fa5bc1c972e6347f328253904fa03a64cc8d68153be"

# files the runs read inside data.zip (single files) and trees (prefixes)
ZIP_FILES = (["data/splits/splits.json", "data/q1_lenses_processed/lens_cutouts.npz",
              "data/feature_cache/source_train.npz", "data/feature_cache/val.npz",
              "data/feature_cache/test_negatives_shared.npz"]
             + [f"data/feature_cache/test_pos_fold{k}.npz" for k in range(5)]
             + [f"data/feature_cache/target_fold{k}.npz" for k in range(5)])
ZIP_TREES = {"data/sim_arc_on_real/": 60, "data/q1_unlabeled/shards/": 250,
             "data/ml_fp_cutouts/shards/": 256}

RUN_FILES = [
    "runs/dilute10_prereg_part1.json", "runs/dilute10_prereg_part2.json",
    "runs/dilute10_prereg_part3.json",
    "runs/protected_artifacts.json", "runs/bakeoff_baseline_A/checkpoints/step_8000.pt",
    "runs/bnm_r6_stage0a.json", "runs/bnm_r6_budget.json", "runs/bnm_r6_prereg.json",
    "runs/bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json", "runs/dann_r5_budget.json",
    "runs/dann_r5_prereg.json", "runs/lambda0_r6_prereg.json", "runs/coral_r4_budget.json",
    "runs/dilute10/feature_cache_manifest.json",
] + [   # dilute10_pool.pinned_holdout reads the 78 recorded holdout positions from here
    f"runs/dann_main/fold{k}/extracted/fold{k}/run.json" for k in (1, 2, 3, 4)]
# the bundle's own small data files; never named after a shipped zip's stem (see
# check_zip_stem_collisions)
BUNDLE_DATA = "bundle_data"
CORAL_FROZEN = ["checkpoints/step_8000.pt", "gate_values.json", "run_config.json"]
# from v3: the FreeMatch frozen stage's final heads (filed PROJECTv7 7.49), same three files
FREEMATCH_FROZEN = CORAL_FROZEN
FREEMATCH_FROZEN_FROM_VERSION = 3
REFERENCE = {   # bundle path -> repo source
    "reference/coral_r4_budget.json": "runs/coral_r4_budget.json",
    "reference/coral_r4_nb3_dev8000_fold0/run.json":
        "kaggle/coral_r4/dataset/reference/coral_r4_nb3_dev8000_fold0/run.json",
    "reference/coral_r4_logits/ids.npz":
        "kaggle/coral_r4/dataset/reference/coral_r4_logits/ids.npz",
}
CELL_REQUIRES = ("run_dilute10_dann.py", "run_dilute10_bnm.py",
                 "run_dilute10_freematch_frozen.py", "run_dilute10_freematch_nb3.py",
                 "run_dilute10_coral_nb3.py", "dilute10_common.py", "dilute10_pool.py",
                 "uniform_target_sampler.py", "dilute10_batch_log.py",
                 "pack_dilute10_results.py", "build_dann_r5_bundle.py",
                 "package_screen_results.py", "run_dann_r5.py", "run_bnm_r6.py",
                 "run_unfreeze_freematch.py", "run_freematch.py", "run_coral_r4_confirm.py",
                 "run_coral_round4.py", "run_coral_r4_dev8000.py")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def self_cert(p):
    d = json.load(open(p, encoding="utf-8"))
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def verify_data_zip(path):
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    miss = [f for f in ZIP_FILES if f not in names]
    trees = {t: sum(1 for n in names if n.startswith(t) and n.endswith(".npz"))
             for t in ZIP_TREES}
    short = {t: (n, ZIP_TREES[t]) for t, n in trees.items() if n < ZIP_TREES[t]}
    sp_zip = hashlib.sha256(z.read("data/splits/splits.json")).hexdigest()
    sp_repo = sha256_file(os.path.join(REPO, "data", "splits", "splits.json"))
    if miss or short or sp_zip != sp_repo:
        raise SystemExit(f"[bundle] data.zip unusable: missing {miss}, short trees {short}, "
                         f"splits.json match {sp_zip == sp_repo}")
    # the P2 shards holding every fresh galaxy are inside the zip
    pool = json.load(open(os.path.join(REPO, "data", "splits", "dilute10_pool.json")))
    shards = {loc[0] for fd in pool["folds"] for loc in fd["target_neg_locations"]}
    absent = [s for s in shards if f"data/q1_unlabeled/shards/{s}" not in names]
    if absent:
        raise SystemExit(f"[bundle] data.zip lacks {len(absent)} pool shards: {absent[:4]}")
    return {"files_checked": len(ZIP_FILES), "files_checked_list": list(ZIP_FILES),
            "trees": trees, "splits_json_sha256": sp_zip,
            "splits_json_matches_repo": True, "pool_shards_present": len(shards),
            "note": ("Kaggle EXTRACTS an uploaded zip, so the cells verify the extracted "
                     "contents (files_checked_list, tree counts, splits.json sha256) instead "
                     "of the zip's own sha256")}


def check_zip_stem_collisions(out):
    """Kaggle extracts every uploaded ``X.zip`` into a folder ``X/`` and drops the zip. Refuse
    the build if any top-level folder (``data/``, ``reference/``, ``runs/``, ``scripts/``,
    ...) or file has the same name as a shipped zip's stem."""
    top = os.listdir(out)
    stems = {os.path.splitext(f)[0] for f in top
             if f.lower().endswith(".zip") and os.path.isfile(os.path.join(out, f))}
    clash = sorted(t for t in top if t in stems and not t.lower().endswith(".zip"))
    if clash:
        raise SystemExit(f"[bundle] refusing: top-level {clash} would collide with Kaggle's "
                         f"auto-extraction of {[c + '.zip' for c in clash]}")
    return sorted(stems)


def build(out, version):
    if os.path.exists(out):
        raise SystemExit(f"[bundle] refusing: {out} exists (bump --version into a new dir)")
    # preconditions: the hashed artifacts are what parts 1/2 declare
    assert self_cert(os.path.join(REPO, "runs", "dilute10_prereg_part1.json")) == PART1
    assert self_cert(os.path.join(REPO, "runs", "dilute10_prereg_part2.json")) == PART2
    assert self_cert(os.path.join(REPO, "runs", "dilute10_prereg_part3.json")) == PART3
    assert sha256_file(os.path.join(REPO, "data", "splits", "dilute10_pool.json")) == POOL
    zinfo = verify_data_zip(DATA_ZIP_SRC)
    os.makedirs(out)
    # scripts/ (the whole folder)
    scripts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "*.py")))
    for f in scripts:
        put(os.path.join(HERE, f), os.path.join(out, "scripts", f))
    miss = [f for f in CELL_REQUIRES if f not in scripts]
    assert not miss, miss
    # data.zip: hard link (0 bytes extra) or copy
    try:
        os.link(DATA_ZIP_SRC, os.path.join(out, "data.zip"))
        how = "hardlink"
    except OSError:
        shutil.copy2(DATA_ZIP_SRC, os.path.join(out, "data.zip"))
        how = "copy"
    for f in ("dilute10_pool.json", "d1_test_neg_remove.json"):
        put(os.path.join(REPO, "data", "splits", f),
            os.path.join(out, BUNDLE_DATA, "splits", f))
    for r in RUN_FILES:
        put(os.path.join(REPO, r), os.path.join(out, r))
    heads = {}
    for k in (1, 2, 3, 4):
        base = f"runs/dilute10/coral_frozen/fold{k}"
        for f in CORAL_FROZEN:
            src = os.path.join(REPO, base, f)
            if not os.path.isfile(src):
                raise SystemExit(f"[bundle] missing {base}/{f} -- run section 1 first")
            put(src, os.path.join(out, base, f))
        gv = json.load(open(os.path.join(REPO, base, "gate_values.json")))
        hs = sha256_file(os.path.join(REPO, base, "checkpoints", "step_8000.pt"))
        if gv["head_sha256"] != hs:
            raise SystemExit(f"[bundle] fold {k}: gate_values.json describes another head")
        heads[str(k)] = hs
    fm_heads = {}
    if int(version) >= FREEMATCH_FROZEN_FROM_VERSION:
        for k in (1, 2, 3, 4):
            base = f"runs/dilute10/freematch_frozen/fold{k}"
            for f in FREEMATCH_FROZEN:
                src = os.path.join(REPO, base, f)
                if not os.path.isfile(src):
                    raise SystemExit(f"[bundle] missing {base}/{f} -- ingest FreeMatch frozen first")
                put(src, os.path.join(out, base, f))
            gv = json.load(open(os.path.join(REPO, base, "gate_values.json")))
            hs = sha256_file(os.path.join(REPO, base, "checkpoints", "step_8000.pt"))
            if gv["head_sha256"] != hs or int(gv["fold"]) != k:
                raise SystemExit(f"[bundle] fold {k}: FreeMatch gate_values.json describes "
                                 f"another head or fold")
            fm_heads[str(k)] = hs
    for dst, src in REFERENCE.items():
        put(os.path.join(REPO, src), os.path.join(out, dst))
    if os.path.isfile(os.path.join(REPO, CELLS)):
        put(os.path.join(REPO, CELLS), os.path.join(out, CELLS))
    zip_stems = check_zip_stem_collisions(out)
    with open(os.path.join(out, "VERSION.txt"), "w") as fh:
        fh.write(f"dilute10 bundle v{version} -- built "
                 f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write("# lens-paper -- 1:10 diluted-target experiment (Kaggle bundle)\n\n"
                 f"Bundle `dilute10` v{version}. The cells are in `{CELLS}`. Cell 2 finds this\n"
                 "bundle by its MANIFEST.json (bundle == \"dilute10\") and verifies every file\n"
                 "by sha256 before anything runs. Parts 1, 2 and 3 of the pre-registration govern.\n")
    files = {}
    for root, _d, fs in os.walk(out):
        for f in fs:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out).replace("\\", "/")
            if rel == "MANIFEST.json":
                continue
            files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
    man = {"bundle": BUNDLE, "version": int(version),
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "cells": CELLS, "part1_sha256": PART1, "part2_sha256": PART2,
           "part3_sha256": PART3,
           "pool_sha256": POOL, "scripts": scripts, "n_scripts": len(scripts),
           "cell_requires": list(CELL_REQUIRES), "data_zip": {"how": how, **zinfo},
           "bundle_data_dir": BUNDLE_DATA, "zip_stems_checked": zip_stems,
           "coral_frozen_heads_sha256": heads,
           "freematch_frozen_heads": (fm_heads if fm_heads else
                                      f"NOT IN v{int(version)} -- produced by session "
                                      "S4/S5; bundle v3"),
           "freematch_frozen_heads_sha256": fm_heads or None,
           "v2_note": "v2 = the provenance fix: runs record part 2, bundle version, manifest",
           "v3_note": ("v3 = the FreeMatch frozen heads (7.49), part 3 (7.48) and the 7.47 "
                       "code (guard log-only, band check by completion)"),
           "zoobot_weights": "not shipped (as in every earlier bundle); fetched from the hub",
           "n_files": len(files), "total_bytes": sum(v["bytes"] for v in files.values()),
           "files": files}
    with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    print(f"[bundle] {out}: {len(files)} files, {man['total_bytes'] / 1e9:.2f} GB, "
          f"MANIFEST sha256 {sha256_file(os.path.join(out, 'MANIFEST.json'))}")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "kaggle", "dilute10"))
    ap.add_argument("--version", type=int, default=1)
    a = ap.parse_args(argv)
    build(os.path.abspath(a.out), a.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
