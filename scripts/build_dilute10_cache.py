"""
build_dilute10_cache.py
=======================
The frozen-feature cache for CORAL's diluted FROZEN stage (brief_dilute10_code.md
section 6): ``data/feature_cache_dilute10/target_fold{k}.npz`` for k = 1..4, each the
fold's whole diluted target pool in pool order (lenses, existing negatives, fresh draws).

* Rows of the ORIGINAL pool are COPIED byte for byte from ``data/feature_cache`` (never
  re-encoded into the new file); only the fresh galaxies are encoded, with
  ``cache_features.encode_indices`` -- the cache's own path -- frozen encoder, aug="none",
  batch 256.
* KNOWN-NULL, before anything is written: every existing target item of the fold is
  re-encoded through the same path with the cache's own batch geometry (the whole original
  pool, in order, batch 256) and must be BYTE-IDENTICAL to ``data/feature_cache``. Measured
  while building this script: on the local GPU that holds for all 782 rows, while a CPU
  encode never does (max |diff| ~1e-3) and a 50-item batch on the GPU differs by ~7e-4 --
  the features depend on device and batch composition. Any non-identical row refuses.
* ``data/feature_cache`` is opened read-only and never written.
* Writes ``runs/dilute10/feature_cache_manifest.json``: every new file's sha256, the device,
  the known-null result, the pool and builder hashes.

    python -u scripts/build_dilute10_cache.py --out data/feature_cache_dilute10 \
        --manifest runs/dilute10/feature_cache_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_pool as DP                                   # noqa: E402

ORIG_CACHE = os.path.join(REPO, "data", "feature_cache")
FOLDS = (1, 2, 3, 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the dilute10 target feature cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args(argv)

    import torch
    import integrity
    from cache_features import _save_split, dataset_ids, encode_indices
    from lens_dataset import LensDataset, UNLABELED
    from zoobot_encoder import ZoobotEncoder

    out = os.path.abspath(a.out)
    if os.path.abspath(out) == os.path.abspath(ORIG_CACHE):
        raise SystemExit("[dilute10-cache] refusing to write into data/feature_cache")
    for p in (out, os.path.abspath(a.manifest)):
        if os.path.exists(p) and not a.force:
            raise SystemExit(f"[dilute10-cache] refusing to overwrite {p} (pass --force)")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[dilute10-cache] no CUDA device: a CPU encode does not reproduce "
                         "the cache byte for byte (measured), so the known-null would fail")

    integrity.verify(where="(build_dilute10_cache start)")
    pool = DP.load_pool()
    sp = DP.load_splits()
    data_root = os.path.join(REPO, "data")
    dev_name = torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu"
    enc = ZoobotEncoder(encoder_mode="frozen", device=a.device, verbose=False)

    staged, report = {}, {}
    for k in FOLDS:
        fd = DP.pool_fold(pool, k)
        n_l, n_ex = int(fd["n_lenses"]), int(fd["n_existing_negatives"])
        n_orig, n_tot = n_l + n_ex, n_l + int(fd["n_negatives"])
        with np.load(os.path.join(ORIG_CACHE, f"target_fold{k}.npz"),
                     allow_pickle=False) as z:
            old = {key: z[key].copy() for key in z.files}
        if old["features"].shape[0] != n_orig:
            raise SystemExit(f"fold {k}: original cache has {old['features'].shape[0]} rows, "
                             f"pool says {n_orig}")
        ds = LensDataset(DP.diluted_splits(sp, pool, k), data_root, role="target", fold=k,
                         aug="none")
        assert len(ds) == n_tot and np.all(ds.labels == UNLABELED)
        ids = dataset_ids(ds)
        if not np.array_equal(ids[:n_orig], old["ids"]):
            raise SystemExit(f"fold {k}: existing item ids/order differ from the cache")

        # ---- KNOWN-NULL: the existing items, same path, same batch geometry ----
        t0 = time.time()
        re = encode_indices(enc, ds, np.arange(n_orig), a.device,
                            batch_size=a.batch_size, progress_every=0)
        same = np.all(re == old["features"], axis=1)
        kn = {"n_rows": int(n_orig), "n_byte_identical": int(same.sum()),
              "byte_identical_all": bool(re.tobytes() == old["features"].tobytes()),
              "max_abs_diff": float(np.abs(re - old["features"]).max()),
              "first_50_byte_identical": bool(re[:50].tobytes()
                                              == old["features"][:50].tobytes())}
        print(f"[known-null] fold {k}: {kn['n_byte_identical']}/{n_orig} rows byte-identical "
              f"(max|diff| {kn['max_abs_diff']:.3g})", flush=True)
        if not kn["byte_identical_all"]:
            raise SystemExit(f"[dilute10-cache] KNOWN-NULL FAILED on fold {k}: the encode "
                             f"path does not reproduce data/feature_cache. Nothing written.")

        # ---- the fresh galaxies ----
        fresh = encode_indices(enc, ds, np.arange(n_orig, n_tot), a.device,
                               batch_size=a.batch_size, progress_every=0)
        feats = np.concatenate([old["features"], fresh]).astype(np.float32)
        assert feats[:n_orig].tobytes() == old["features"].tobytes()
        staged[k] = dict(features=feats, labels=ds.labels, latent=ds.latent_labels,
                         domains=ds.domains, ids=ids, kinds=np.asarray(ds.kinds),
                         meta={"split": f"target_fold{k}", "role": "target", "fold": k,
                               "n": n_tot, "encoder_mode": "frozen",
                               "slug": json.loads(str(old["meta"]))["slug"],
                               "embed_dim": int(feats.shape[1]),
                               "dilute10": {"n_lenses": n_l, "n_existing": n_orig,
                                            "n_fresh": n_tot - n_orig,
                                            "existing_rows": "copied from data/feature_cache",
                                            "fresh_rows_device": dev_name}})
        report[k] = {"known_null": kn, "n_fresh_encoded": int(n_tot - n_orig),
                     "seconds": time.time() - t0}

    os.makedirs(out, exist_ok=True)
    files = {}
    for k, s in staged.items():
        p = os.path.join(out, f"target_fold{k}.npz")
        _save_split(p, s["features"], s["labels"], s["latent"], s["domains"], s["ids"],
                    s["kinds"], s["meta"])
        files[os.path.relpath(p, REPO).replace("\\", "/")] = DP.sha256_file(p)
    manifest = {
        "what": "dilute10 target feature cache for CORAL's diluted frozen stage, folds 1-4",
        "files_sha256": files,
        "device": dev_name, "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version() if a.device == "cuda" else None,
        "batch_size": a.batch_size, "aug": "none", "encoder_mode": "frozen",
        "known_null": {str(k): r["known_null"] for k, r in report.items()},
        "known_null_rule": ("every existing target item re-encoded through "
                            "cache_features.encode_indices with the cache's own batch "
                            "geometry must be byte-identical to data/feature_cache"),
        "n_fresh_encoded": {str(k): r["n_fresh_encoded"] for k, r in report.items()},
        "pool_sha256": pool["_verified"]["pool_sha256"],
        "part1_sha256": DP.PART1_SHA256,
        "builder": "scripts/build_dilute10_cache.py",
        "builder_sha256": DP.sha256_file(os.path.abspath(__file__)),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.manifest)), exist_ok=True)
    with open(a.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    integrity.verify(where="(build_dilute10_cache end)")
    print(json.dumps({"files": files, "manifest_sha256": DP.sha256_file(a.manifest)},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
