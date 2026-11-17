"""
stage_dilute10.py
=================
Cell 2 of KAGGLE_DILUTE10.md, as a script, so the bundle self-test runs EXACTLY what the
cells run.

1. MOUNT BY CONTENT. Find every MANIFEST.json under ``--input-root`` (depth <= 6); keep
   those with ``bundle == "dilute10"``; take the highest version.
2. VERIFY. Every file the manifest lists is checked by sha256. ``data.zip`` is the one
   exception, because Kaggle extracts an uploaded zip: if it arrives as a FILE it is hashed;
   if it arrives EXTRACTED, the directory holding ``splits/splits.json`` is located and every
   file in the manifest's ``data_zip.files_checked`` list is required, the image trees are
   counted, and ``splits.json`` is hashed against the manifest.
3. STAGE into ``--work`` (the repo root the runners expect): ``scripts/`` copied,
   ``data/splits/splits.json`` copied from the extracted data, ``data/splits/
   {dilute10_pool.json, d1_test_neg_remove.json}`` copied from the bundle's
   ``bundle_data/splits/`` (the manifest's ``bundle_data_dir``; not ``data/``, which Kaggle's
   extraction of ``data.zip`` occupies), every other ``data/<tree>`` LINKED to the mount
   (symlink; a junction on Windows), ``runs/`` copied. The staged destinations are unchanged.
4. RE-VERIFY in place: part 1 and part 2 self-certify to their recorded digests; the pool
   hashes to part 1's value.
Writes ``<work>/stage.json`` with the resolved paths, the bundle version and the bundle's
``MANIFEST.json`` sha256 (every run records these; v2).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

DATA_TREES = ("feature_cache", "q1_unlabeled", "ml_fp_cutouts", "sim_arc_on_real",
              "q1_lenses_processed")


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


def link(src, dst):
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except (OSError, NotImplementedError):
        import _winapi
        _winapi.CreateJunction(src, dst)


def find_bundle(input_root):
    found = []
    for d in range(0, 7):
        for mp in glob.glob(os.path.join(input_root, *(["*"] * d), "MANIFEST.json")):
            try:
                m = json.load(open(mp, encoding="utf-8"))
            except Exception:                                            # noqa: BLE001
                continue
            if m.get("bundle") == "dilute10":
                found.append((os.path.dirname(mp), m))
    if not found:
        raise SystemExit("[stage] no dilute10 bundle (MANIFEST.json with bundle=='dilute10') "
                         f"under {input_root}. Add Data -> the dilute10 dataset.")
    found.sort(key=lambda rm: int(rm[1].get("version") or 0))
    return found[-1]


def find_data_dir(root):
    for d in range(0, 6):
        for p in glob.glob(os.path.join(root, *(["*"] * d), "splits", "splits.json")):
            dd = os.path.dirname(os.path.dirname(p))
            if os.path.isdir(os.path.join(dd, "feature_cache")):
                return dd
    raise SystemExit(f"[stage] no extracted data (splits/splits.json + feature_cache) "
                     f"under {root}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    bundle, man = find_bundle(os.path.abspath(a.input_root))
    print(f"[stage] bundle {man['bundle']} v{man['version']} built {man['built_utc']} at "
          f"{bundle}")

    # ---- verify every file ----
    bad, n_ok = [], 0
    for rel, info in man["files"].items():
        p = os.path.join(bundle, rel)
        if rel == "data.zip" and not os.path.isfile(p):
            continue                                   # extracted by Kaggle: checked below
        if not os.path.isfile(p):
            bad.append(f"{rel}: MISSING")
        elif os.path.getsize(p) != info["bytes"] or sha256_file(p) != info["sha256"]:
            bad.append(f"{rel}: CHECKSUM MISMATCH")
        else:
            n_ok += 1
    if bad:
        raise SystemExit("[stage] BUNDLE VERIFICATION FAILED:\n  " + "\n  ".join(bad[:20]))
    print(f"[stage] {n_ok} files verified by sha256")

    if os.path.isfile(os.path.join(bundle, "data.zip")):
        raise SystemExit("[stage] data.zip arrived as a FILE, not extracted. Kaggle extracts "
                         "uploaded zips; extract it under the bundle and re-run.")
    data_dir = find_data_dir(bundle)
    req = [f[len("data/"):] for f in man["data_zip"]["files_checked_list"]]
    miss = [f for f in req if not os.path.isfile(os.path.join(data_dir, f))]
    trees = {t: len(glob.glob(os.path.join(data_dir, t[len("data/"):], "*.npz")))
             for t in man["data_zip"]["trees"]}
    short = {t: n for t, n in trees.items() if n < man["data_zip"]["trees"][t]}
    sp = sha256_file(os.path.join(data_dir, "splits", "splits.json"))
    if miss or short or sp != man["data_zip"]["splits_json_sha256"]:
        raise SystemExit(f"[stage] EXTRACTED DATA INCOMPLETE: missing {miss[:6]}, short "
                         f"trees {short}, splits.json matches {sp == man['data_zip']['splits_json_sha256']}")
    print(f"[stage] extracted data at {data_dir}: {len(req)} files, trees {trees}, "
          f"splits.json verified")

    # ---- stage ----
    work = os.path.abspath(a.work)
    os.makedirs(os.path.join(work, "scripts"), exist_ok=True)
    for f in man["scripts"]:
        shutil.copy2(os.path.join(bundle, "scripts", f), os.path.join(work, "scripts", f))
    os.makedirs(os.path.join(work, "data", "splits"), exist_ok=True)
    shutil.copy2(os.path.join(data_dir, "splits", "splits.json"),
                 os.path.join(work, "data", "splits", "splits.json"))
    bdata = man.get("bundle_data_dir")
    if not bdata or bdata == "data":
        raise SystemExit("[stage] manifest has no bundle_data_dir (old layout whose data/ "
                         "collides with Kaggle's extraction of data.zip); rebuild the bundle")
    for f in ("dilute10_pool.json", "d1_test_neg_remove.json"):
        shutil.copy2(os.path.join(bundle, bdata, "splits", f),
                     os.path.join(work, "data", "splits", f))
    for t in DATA_TREES:
        link(os.path.join(data_dir, t), os.path.join(work, "data", t))
    for root, _d, fs in os.walk(os.path.join(bundle, "runs")):
        for f in fs:
            src = os.path.join(root, f)
            dst = os.path.join(work, os.path.relpath(src, bundle))
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    # ---- re-verify in place ----
    p1 = self_cert(os.path.join(work, "runs", "dilute10_prereg_part1.json"))
    p2 = self_cert(os.path.join(work, "runs", "dilute10_prereg_part2.json"))
    pl = sha256_file(os.path.join(work, "data", "splits", "dilute10_pool.json"))
    if (p1, p2, pl) != (man["part1_sha256"], man["part2_sha256"], man["pool_sha256"]):
        raise SystemExit(f"[stage] PRE-REGISTRATION MISMATCH: part1 {p1[:16]} part2 "
                         f"{p2[:16]} pool {pl[:16]}")
    # part 3 (PROJECTv7 7.48) ships from bundle v3 on: a manifest that names it must carry a
    # staged file with exactly that self-certifying digest; CORAL nb3 and FreeMatch nb3 refuse
    # to start without it (dilute10_common.verify_part3)
    p3 = None
    if man.get("part3_sha256"):
        p3p = os.path.join(work, "runs", "dilute10_prereg_part3.json")
        if not os.path.isfile(p3p):
            raise SystemExit("[stage] PRE-REGISTRATION MISMATCH: the manifest names part 3 but "
                             "runs/dilute10_prereg_part3.json was not staged")
        p3 = self_cert(p3p)
        if p3 != man["part3_sha256"]:
            raise SystemExit(f"[stage] PRE-REGISTRATION MISMATCH: part3 {p3[:16]} != manifest "
                             f"{man['part3_sha256'][:16]}")
    st = {"bundle": bundle, "version": man["version"], "data_dir": data_dir, "work": work,
          "part1": p1, "part2": p2, "part3": p3, "pool": pl,
          # read by dilute10_common.provenance() into every run's own record (bundle v2)
          "manifest_sha256": sha256_file(os.path.join(bundle, "MANIFEST.json"))}
    with open(os.path.join(work, "stage.json"), "w") as fh:
        json.dump(st, fh, indent=1)
    print(f"[stage] part 1 {p1[:16]}..., part 2 {p2[:16]}..., "
          f"part 3 {(p3[:16] + '...') if p3 else '(not in this bundle)'}, "
          f"pool {pl[:16]}... verified")
    print(f"[stage] OK -- staged into {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
