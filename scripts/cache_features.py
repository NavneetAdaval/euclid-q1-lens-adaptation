"""
cache_features.py
=================
ONE-TIME frozen-feature extraction for the label-free sim-to-real strong-lens
pipeline (Euclid Q1). This is the compute-saver: the Zoobot encoder is FROZEN, so
its 640-dim features for a given image never change. We run the encoder over every
needed split ONCE (deterministic, aug="none") and write the features to disk. Every
downstream step (evaluate.py, the DA bake-off comparisons) then reads this cache and
runs on CPU/4050 -- the encoder is never called inside an eval loop.

This is the ONLY step that needs a GPU.

WHAT IS CACHED (one .npz per split, under ``cache_dir``)
--------------------------------------------------------
  source_train.npz              labeled source (sim lenses + 3 negative sources)
  val.npz                       labeled source (sim lenses + 2 negative sources)
  test_negatives_shared.npz     the shared P4 negative pool -- encoded ONCE and
                                reused across all 5 test folds (the key dedup;
                                PROJECT.md 4.4: test negatives are deliberately shared)
  test_pos_fold{k}.npz          fold k's held-out real-lens POSITIVES (k=0..4)
  target_fold{k}.npz            fold k's UNLABELED target pool (real lenses from the
                                other folds + fresh target negatives; k=0..4)

Each .npz holds, row-aligned:
  features        (N, 640) float32   frozen Zoobot features
  labels          (N,)     int64     the SERVED label (test/source/val: 0/1;
                                      target: UNLABELED=-1)
  latent_labels   (N,)     int64     true identity 0/1 (diagnostics only)
  domains         (N,)     int64     0=source, 1=target
  ids             (N,)     <U…       a STABLE, machine-independent id per row
  kinds           (N,)     <U…       provenance kind (real_lens/random/sim_lens/…)
  meta            0-d json string     {split, role, fold, n, encoder_mode, slug}

DESIGN NOTES (tags: RB=research-backed, EC=engineering-choice, FLAG=forced deviation)

  * DETERMINISTIC (RB) -- every split is built with ``LensDataset(aug="none")`` (the
      loader's deterministic stretch+resize path). No augmentation RNG touches the
      cache, so the features are reproducible.

  * SHARED NEGATIVES ENCODED ONCE (RB) -- role="test" serves fold-positives + the
      shared negatives together. We encode the negative pool for fold 0 only, save it
      as ``test_negatives_shared.npz``, and for folds 1-4 we ENCODE ONLY the positives
      (an index subset) -- the negative rows are never pushed through the GPU again.
      We assert the folds' negative ids match the cached shared ids, so the "shared"
      contract is verified, not assumed.
      FLAG: LensDataset(role="test") still LOADS the ~26k negative pixels from the
      shards for folds 1-4 (it has no positives-only mode) -- that is cheap CPU/IO,
      not GPU compute. The expensive part (the encoder forward pass) is what we
      deduplicate, which is what the request asked for.

  * RESUMABLE (RB) -- each split is an independent file; a split whose .npz already
      exists (and is loadable) is skipped. A flaky Colab/Kaggle session can restart
      and pick up where it left off. ``--force`` re-encodes everything.

  * STABLE IDS (EC) -- ids are built from each row's provenance descriptor, NOT from
      an absolute path (paths differ across machines). Formats:
        sim_lens       -> "sim_lens:<global_row>"
        bare_deflector -> "bare_deflector:<shard_fn>:<idx>"
        file (others)  -> "<kind>:<basename>:<row>"
      so the same physical image gets the same id whether it appears as a test
      positive or an unlabeled target positive -- which lets downstream code line up
      shared negatives row-for-row for the paired bootstrap.

Run (needs GPU + the real data):
    python scripts\\cache_features.py --data-root data --cache-dir data\\feature_cache
    python scripts\\cache_features.py ... --force          # re-encode everything
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lens_dataset import LensDataset, UNLABELED
from zoobot_encoder import ZoobotEncoder, EMBED_DIM, MODEL_SLUG


# ---------------------------------------------------------------------------
# stable per-row id (machine-independent)
# ---------------------------------------------------------------------------

def stable_id(kind: str, ref: tuple) -> str:
    """Build a stable id from a LensDataset descriptor's ``ref`` (see module docstring)."""
    t = ref[0]
    if t == "sim_lens":
        return f"sim_lens:{int(ref[1])}"
    if t == "sim_nonlens":
        return f"bare_deflector:{ref[1]}:{int(ref[2])}"
    if t == "file":
        return f"{kind}:{os.path.basename(ref[1])}:{int(ref[3])}"
    raise AssertionError(f"unknown ref kind {ref!r}")


def dataset_ids(ds: LensDataset) -> np.ndarray:
    ids = [stable_id(d["kind"], d["ref"]) for d in ds._descriptors]
    return np.asarray(ids, dtype="<U80")


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_indices(encoder: ZoobotEncoder, ds: LensDataset, indices, device,
                   batch_size: int = 256, num_workers: int = 0,
                   progress_every: int = 20) -> np.ndarray:
    """Run the frozen encoder over ``ds`` restricted to ``indices`` (in that order).

    Returns an ``(len(indices), 640)`` float32 array aligned to ``indices``.
    """
    indices = np.asarray(indices, dtype=np.int64)
    subset = Subset(ds, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    out = np.empty((indices.size, EMBED_DIM), dtype=np.float32)
    pos = 0
    n_batches = len(loader)
    for bi, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True)
        feats = encoder(imgs).float().cpu().numpy()
        out[pos:pos + feats.shape[0]] = feats
        pos += feats.shape[0]
        if progress_every and (bi % progress_every == 0 or bi == n_batches - 1):
            print(f"      batch {bi + 1}/{n_batches}  ({pos}/{indices.size} rows)",
                  flush=True)
    assert pos == indices.size, (pos, indices.size)
    return out


def _save_split(path, features, labels, latent, domains, ids, kinds, meta):
    tmp = path + ".tmp.npz"
    np.savez(tmp,
             features=features.astype(np.float32),
             labels=np.asarray(labels, dtype=np.int64),
             latent_labels=np.asarray(latent, dtype=np.int64),
             domains=np.asarray(domains, dtype=np.int64),
             ids=np.asarray(ids, dtype="<U80"),
             kinds=np.asarray(kinds, dtype="<U40"),
             meta=np.asarray(json.dumps(meta)))
    os.replace(tmp, path)  # atomic: a killed process never leaves a half-written .npz


def _already_cached(path: str) -> bool:
    """True iff ``path`` exists and is a loadable .npz with a 'features' key."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            return "features" in d.files
    except Exception as e:  # corrupt / partial -> re-encode
        print(f"    [warn] {os.path.basename(path)} exists but is unreadable "
              f"({e}); re-encoding")
        return False


def cache_indices(encoder, ds, indices, device, path, split_name, role, fold,
                  batch_size, num_workers):
    """Encode ``ds[indices]`` and write ``path`` (atomic)."""
    indices = np.asarray(indices, dtype=np.int64)
    feats = encode_indices(encoder, ds, indices, device,
                           batch_size=batch_size, num_workers=num_workers)
    ids = dataset_ids(ds)[indices]
    kinds = np.asarray(ds.kinds, dtype="<U40")[indices]
    meta = {"split": split_name, "role": role, "fold": fold, "n": int(indices.size),
            "encoder_mode": encoder.encoder_mode, "slug": MODEL_SLUG,
            "embed_dim": EMBED_DIM}
    _save_split(path, feats, ds.labels[indices], ds.latent_labels[indices],
                ds.domains[indices], ids, kinds, meta)
    print(f"    saved {os.path.basename(path)}  (n={indices.size})", flush=True)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run(splits_path, data_root, cache_dir, folds, device, batch_size=256,
        num_workers=0, force=False, aug="none", n_blocks=0):
    # ****  THE CACHE IS VALID ONLY AT n_blocks=0  ****
    # Caching rests on ONE property: the encoder is frozen, so a given image's 640-d
    # feature never changes. Partial unfreezing breaks that twice over -- (a) the encoder
    # weights move every step, so any cached feature is stale the moment training starts,
    # and (b) augmentation (the +-3px translate; weak/strong views for AdaMatch/FreeMatch)
    # is applied BEFORE the encoder, so there is no deterministic prefix to cache in the
    # first place. Raise rather than silently emit features that look fine and are wrong.
    if int(n_blocks) != 0:
        raise ValueError(
            f"cache_features is valid ONLY for a frozen encoder (n_blocks=0), got "
            f"n_blocks={n_blocks}. With unfrozen blocks the encoder weights change every "
            f"step and augmentation happens pre-encoder, so there is no deterministic "
            f"feature to cache -- an unfrozen run must encode images live, not read this "
            f"cache. Refusing to write stale features.")
    os.makedirs(cache_dir, exist_ok=True)
    print("=" * 72)
    print("cache_features: one-time frozen-feature extraction")
    print(f"  splits    : {splits_path}")
    print(f"  data_root : {data_root}")
    print(f"  cache_dir : {cache_dir}")
    print(f"  device    : {device}   folds : {folds}")
    print("=" * 72, flush=True)

    def path_for(name):
        return os.path.join(cache_dir, f"{name}.npz")

    def maybe(name):
        """Return the path if this split still needs encoding, else None (skip)."""
        p = path_for(name)
        if not force and _already_cached(p):
            print(f"[skip] {name} (already cached)")
            return None
        return p

    print("\n[load] building the frozen Zoobot encoder (once) ...", flush=True)
    encoder = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=False)
    encoder.eval()

    t0 = time.time()

    # ---- TEST: per-fold positives + the shared negatives (encoded once) -------
    shared_ids = None  # populated once the shared-negatives split is available
    shared_path = path_for("test_negatives_shared")
    if not force and _already_cached(shared_path):
        with np.load(shared_path, allow_pickle=False) as d:
            shared_ids = d["ids"].copy()

    for k in folds:
        print(f"\n[test fold {k}] loading role='test' ...", flush=True)
        ds = LensDataset(splits_path, data_root, role="test", fold=k, aug=aug)
        pos_idx = np.where(ds.labels == 1)[0]
        neg_idx = np.where(ds.labels == 0)[0]
        ids = dataset_ids(ds)

        # shared negatives: encode ONCE (first fold that needs them)
        if shared_ids is None:
            p = maybe("test_negatives_shared")
            if p is not None:
                print(f"  encoding {neg_idx.size} SHARED test negatives (once) ...",
                      flush=True)
                cache_indices(encoder, ds, neg_idx, device, p,
                              "test_negatives_shared", "test", None,
                              batch_size, num_workers)
            shared_ids = ids[neg_idx].copy()
        else:
            # verify the shared contract without re-encoding
            got = ids[neg_idx]
            if not np.array_equal(got, shared_ids):
                raise AssertionError(
                    f"fold {k} test negatives differ from the shared pool "
                    f"(n={got.size} vs {shared_ids.size}); the 'shared negatives' "
                    f"contract (PROJECT.md 4.4) is violated -- refusing to cache.")

        # per-fold positives
        p = maybe(f"test_pos_fold{k}")
        if p is not None:
            print(f"  encoding {pos_idx.size} test positives ...", flush=True)
            cache_indices(encoder, ds, pos_idx, device, p,
                          f"test_pos_fold{k}", "test", k, batch_size, num_workers)
        del ds

    # ---- TARGET: per-fold unlabeled pool --------------------------------------
    for k in folds:
        p = maybe(f"target_fold{k}")
        if p is None:
            continue
        print(f"\n[target fold {k}] loading role='target' ...", flush=True)
        ds = LensDataset(splits_path, data_root, role="target", fold=k, aug=aug)
        assert np.all(ds.labels == UNLABELED), "target labels must all be UNLABELED(-1)"
        cache_indices(encoder, ds, np.arange(len(ds)), device, p,
                      f"target_fold{k}", "target", k, batch_size, num_workers)
        del ds

    # ---- SOURCE_TRAIN and VAL -------------------------------------------------
    for name, role in (("source_train", "source_train"), ("val", "val")):
        p = maybe(name)
        if p is None:
            continue
        print(f"\n[{name}] loading role='{role}' ...", flush=True)
        ds = LensDataset(splits_path, data_root, role=role, aug=aug)
        cache_indices(encoder, ds, np.arange(len(ds)), device, p,
                      name, role, None, batch_size, num_workers)
        del ds

    print(f"\n[done] feature cache up to date in {time.time() - t0:.1f}s -> {cache_dir}",
          flush=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.normpath(os.path.join(here, "..", "data"))
    p = argparse.ArgumentParser(description="One-time frozen-feature cache (GPU step).")
    p.add_argument("--data-root", default=os.environ.get("LENS_DATA_ROOT", default_root))
    p.add_argument("--splits", default=None,
                   help="splits.json path (default: <data-root>/splits/splits.json)")
    p.add_argument("--cache-dir", default=None,
                   help="output dir (default: <data-root>/feature_cache)")
    p.add_argument("--folds", default="0,1,2,3,4",
                   help="comma-separated fold ids to cache (default all)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="re-encode even if cached")
    p.add_argument("--n-blocks", type=int, default=0,
                   help="accepted ONLY as 0. Present so a caller that is unfreezing gets a "
                        "loud error instead of a silently stale cache (see run()).")
    args = p.parse_args()

    splits_path = args.splits or os.path.join(args.data_root, "splits", "splits.json")
    cache_dir = args.cache_dir or os.path.join(args.data_root, "feature_cache")
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(splits_path):
        print(f"SKIP: real splits not found at {splits_path}")
        return 3

    run(splits_path, args.data_root, cache_dir, folds, device,
        batch_size=args.batch_size, num_workers=args.num_workers, force=args.force,
        n_blocks=args.n_blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
