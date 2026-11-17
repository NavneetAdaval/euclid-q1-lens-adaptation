"""
live_encode.py
==============
LIVE (uncached) frozen-feature extraction for the eval splits, for LEVER 1.

WHY THIS EXISTS
---------------
Every evaluation path in this repo -- ``evaluate.py``, ``curve_hook.py``,
``collapse_gate.py`` -- reads ``data/feature_cache/*.npz``. That cache is valid on exactly
one assumption: the encoder is FROZEN, so an image's 640-d feature never changes
(cache_features.py, PROJECT.md 7.1). At ``n_blocks > 0`` the encoder weights move every
step, so the cache is stale the moment training starts and every downstream number
computed from it would be silently wrong.

So an unfrozen run must ENCODE THE EVAL SPLITS LIVE, with the current encoder, at each
curve point. This module does that and nothing else: it reproduces the cached ``.npz``
row order, labels and ids EXACTLY, so the arrays it returns are drop-in replacements for
``FeatureCache.load(name)`` and every metric/consumer downstream is unchanged.

ROW-ORDER CONTRACT (must match cache_features.py exactly, or A0-image vs A0-cached is
comparing two different orderings and the control is meaningless):
    val                    -> LensDataset(role="val"),           arange(len(ds))
    source_train           -> LensDataset(role="source_train"),  arange(len(ds))
    test_negatives_shared  -> LensDataset(role="test", fold=k),  where(labels == 0)
    test_pos_fold{k}       -> LensDataset(role="test", fold=k),  where(labels == 1)
    target_fold{k}         -> LensDataset(role="target", fold=k),arange(len(ds))
All with ``aug="none"`` -- the deterministic stretch+resize path, no augmentation RNG.

COST DISCIPLINE. Datasets are built ONCE and reused across every curve point (constructing
them re-reads ~26k negative cutouts from the shards, which is pure waste at 32 curve
points). Only the encoder forward is repeated, which is the part that actually changes.

VERIFICATION. ``verify_against_cache()`` encodes with a FROZEN (n_blocks=0) encoder and
compares to the existing ``feature_cache/*.npz`` element-wise. If the live path reproduces
the cache bit-for-bit on the frozen encoder, then the ONLY difference in an unfrozen run is
the encoder weights -- which is exactly the claim the A0-image control has to rest on.

Run the self-verification:
    python scripts/live_encode.py --verify --splits data/splits/splits.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cache_features import dataset_ids     # the ONE stable-id impl (reused, not rewritten)
from lens_dataset import LensDataset, UNLABELED
from zoobot_encoder import EMBED_DIM

# the splits an unfrozen fold-k run needs to score itself, in the cached naming
EVAL_SPLITS = ("val", "test_negatives_shared", "test_pos_fold{k}", "target_fold{k}")


class LiveEncoder:
    """Builds the eval datasets once; re-encodes them on demand with the CURRENT encoder.

    Parameters
    ----------
    splits_path, data_root : str
        As for LensDataset.
    fold : int
        Which fold's test positives / target pool to serve.
    batch_size, num_workers : int
        DataLoader settings for the encode pass (inference only).

        ``num_workers=0`` is the MEASURED optimum here, which is the opposite of the
        training loader's setting -- keep it. On Windows (spawn, not fork) each worker
        re-pickles the dataset's resident image arrays, and the eval loader is built and
        torn down once per split per curve point, so the spawn cost is paid ~128 times a
        fold and never amortised. MEASURED on test_negatives_shared (26,399 rows, fp16):
            num_workers=0 -> 39.0 s | 4 -> 49.2 s | 8 -> 71.1 s
        The training loader is the opposite case (long-lived, persistent_workers) and
        correctly keeps num_workers=4.
    """

    def __init__(self, splits_path, data_root, fold: int, *, batch_size: int = 256,
                 num_workers: int = 0, device="cuda", verbose: bool = True):
        self.splits_path, self.data_root, self.fold = splits_path, data_root, int(fold)
        self.batch_size, self.num_workers = batch_size, num_workers
        self.device = torch.device(device)
        self.verbose = verbose

        if verbose:
            print(f"[live] building eval datasets once (fold {fold}, aug='none') ...",
                  flush=True)
        # aug="none" == the deterministic stretch+resize path (matches cache_features.py)
        self._val = LensDataset(splits_path, data_root, role="val", aug="none")
        self._test = LensDataset(splits_path, data_root, role="test", fold=self.fold,
                                 aug="none")
        self._target = LensDataset(splits_path, data_root, role="target", fold=self.fold,
                                   aug="none")
        assert np.all(self._target.labels == UNLABELED), \
            "target labels must all be UNLABELED(-1) -- the label-free contract"

        self._neg_idx = np.where(self._test.labels == 0)[0]
        self._pos_idx = np.where(self._test.labels == 1)[0]
        self._ids: dict = {}          # memoised stable ids per split (built on first use)
        if verbose:
            print(f"[live]   val={len(self._val)}  test_neg={self._neg_idx.size}  "
                  f"test_pos={self._pos_idx.size}  target={len(self._target)}", flush=True)

    # -- the per-split (dataset, indices) map, mirroring cache_features.py ----

    def _spec(self, name: str):
        k = self.fold
        if name == "val":
            return self._val, np.arange(len(self._val))
        if name == "test_negatives_shared":
            return self._test, self._neg_idx
        if name in (f"test_pos_fold{k}", "test_pos"):
            return self._test, self._pos_idx
        if name in (f"target_fold{k}", "target"):
            return self._target, np.arange(len(self._target))
        raise KeyError(
            f"live_encode does not serve split {name!r} for fold {k}; known: "
            f"val / test_negatives_shared / test_pos_fold{k} / target_fold{k}")

    @torch.no_grad()
    def encode(self, encoder, name: str, *, fp16: bool = False) -> dict:
        """Encode one split with the CURRENT encoder state. Returns a cache-shaped dict.

        Always under ``no_grad`` and with the encoder in eval() -- this is scoring, never
        training, so no graph is built even when n_blocks>0 makes the params trainable.
        """
        ds, idx = self._spec(name)
        was_training = encoder.training
        encoder.eval()
        loader = DataLoader(Subset(ds, idx.tolist()), batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers,
                            pin_memory=(self.device.type == "cuda"))
        out = np.empty((idx.size, EMBED_DIM), dtype=np.float32)
        pos = 0
        amp = torch.amp.autocast("cuda") if (fp16 and self.device.type == "cuda") \
            else _null()
        for batch in loader:
            imgs = batch["image"].to(self.device, non_blocking=True)
            with amp:
                f = encoder(imgs)
            f = f.float().cpu().numpy()
            out[pos:pos + f.shape[0]] = f
            pos += f.shape[0]
        assert pos == idx.size, (pos, idx.size)
        if was_training:
            encoder.train()
        # ids come from cache_features.dataset_ids -- the SAME stable-id builder the cache
        # used -- so a live-encoded split can be checked row-for-row against a cached one
        # (the paired bootstrap requires row alignment, not merely equal lengths).
        if name not in self._ids:
            self._ids[name] = dataset_ids(ds)[idx]
        return {"features": out,
                "labels": np.asarray(ds.labels, dtype=np.int64)[idx],
                "latent_labels": np.asarray(ds.latent_labels, dtype=np.int64)[idx],
                "domains": np.asarray(ds.domains, dtype=np.int64)[idx],
                "ids": self._ids[name]}

    def encode_all(self, encoder, *, fp16: bool = False) -> dict:
        """Encode every split this fold's curve row needs, keyed by CACHED split name."""
        k = self.fold
        return {n.format(k=k): self.encode(encoder, n.format(k=k), fp16=fp16)
                for n in EVAL_SPLITS}


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ---------------------------------------------------------------------------
# verification against the frozen feature cache
# ---------------------------------------------------------------------------

def verify_against_cache(splits_path, data_root, cache_dir, fold=0, device="cuda",
                         batch_size=256, num_workers=0, atol=0.0, fp16=False):
    """Encode LIVE with a FROZEN encoder and compare element-wise to the cached .npz.

    A bit-for-bit match proves the live path reproduces cache_features.py exactly -- same
    row order, same transform, same encoder call -- so in an unfrozen run the ONLY
    difference is the encoder weights themselves.
    """
    import evaluate as ev
    from zoobot_encoder import ZoobotEncoder

    print("=" * 74)
    print(f"live_encode verification vs the frozen feature cache (fold {fold})")
    print(f"  cache : {cache_dir}\n  device: {device}  fp16={fp16}")
    print("=" * 74, flush=True)

    enc = ZoobotEncoder(n_blocks=0, device=device, verbose=False)
    live = LiveEncoder(splits_path, data_root, fold, batch_size=batch_size,
                       num_workers=num_workers, device=device)
    cache = ev.FeatureCache(cache_dir)

    fails = []
    for name in (n.format(k=fold) for n in EVAL_SPLITS):
        if not cache.has(name):
            print(f"  SKIP  {name} (not in cache)")
            continue
        got = live.encode(enc, name, fp16=fp16)
        want = cache.load(name)
        gf, wf = got["features"], np.asarray(want["features"], dtype=np.float32)
        if gf.shape != wf.shape:
            print(f"  FAIL  {name}: shape {gf.shape} != cached {wf.shape}")
            fails.append(name)
            continue
        same_labels = np.array_equal(got["labels"], np.asarray(want["labels"], np.int64))
        d = np.abs(gf - wf)
        exact = bool((d == 0).all())
        ok = exact if atol == 0.0 else bool(d.max() <= atol)
        tag = "PASS" if (ok and same_labels) else "FAIL"
        print(f"  {tag}  {name:<24} n={gf.shape[0]:>6}  max|diff|={d.max():.3e}  "
              f"{'byte-identical' if exact else 'NOT exact'}  labels_match={same_labels}")
        if not (ok and same_labels):
            fails.append(name)

    print("=" * 74)
    if fails:
        print(f"FAILED: {len(fails)} split(s) differ from the cache: {fails}")
        return 1
    print("LIVE ENCODE REPRODUCES THE FROZEN CACHE EXACTLY")
    return 0


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(root, "data"))
    ap.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="0 (default) demands a byte-identical match")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not args.verify:
        print("nothing to do; pass --verify")
        return 0
    return verify_against_cache(args.splits, args.data_root, args.cache_dir,
                                fold=args.fold, device=device,
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                atol=args.atol, fp16=args.fp16)


if __name__ == "__main__":
    sys.exit(main())
