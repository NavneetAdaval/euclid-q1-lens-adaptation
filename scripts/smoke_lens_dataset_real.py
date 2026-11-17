"""
smoke_lens_dataset_real.py
==========================
REAL-DATA smoke test for `LensDataset`. Unlike `test_lens_dataset.py` (which
fabricates a tiny dataset whose pixels encode their own identity, so index
resolution can be checked exactly), this loads the ACTUAL splits.json and the
ACTUAL Euclid data on disk. Real pixels do not encode identity, so the identity
trick is impossible here; instead we check the properties that must hold on real
data regardless of content:

  1. COUNTS   -- len(dataset) and its positive/negative composition match exactly
                 what splits.json declares, for all four roles
                 (source_train, val, target fold 0, test fold 0).
  2. SHAPE    -- every served "image" is (1, 224, 224) float32.
  3. FINITE   -- no NaN / inf survives the stretch+resize pipeline.
  4. RANGE    -- values sit in ~[0, 0.9]. The Zoobot stretch maps to
                 [0, asinh(q)=asinh(1)=0.8814], so the real-data ceiling is 0.881;
                 we allow a small tolerance. A violation means a pixel escaped the
                 percentile clamp (i.e. a bug), which is exactly what we want caught.

Also asserts the label-free contract on real data: the target role's __getitem__
returns UNLABELED (-1) for every item, so no real lens label can leak into a DA loss.

Run:
    python scripts\\smoke_lens_dataset_real.py
Exits 0 on success, 1 on any failed check, 3 if the real data is not present
(so it is safe to run in an environment without the dataset).
"""

import os
import sys
import json
import traceback

import numpy as np
import torch

from lens_dataset import LensDataset, UNLABELED, SOURCE, TARGET

# ---------------------------------------------------------------------------
# where the real data lives (repo_root/data), overridable via env var
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LENS_DATA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "data")))
SPLITS_PATH = os.path.join(DATA_ROOT, "splits", "splits.json")

# how many items to actually push through the transform per role (a smoke test,
# not an exhaustive sweep). All items' COUNTS are checked; RANGE/SHAPE/FINITE are
# checked on a deterministic random sample plus a few guaranteed edge indices.
SAMPLE_PER_ROLE = 200
RANGE_LO, RANGE_HI, TOL = 0.0, 0.9, 1e-3

# ---------------------------------------------------------------------------
# result tracking (same convention as test_lens_dataset.py)
# ---------------------------------------------------------------------------
_fails = []
def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _fails.append(msg)


def expected_counts(splits, role, fold=None):
    """(total, n_pos, n_neg) as DECLARED by splits.json, derived not hardcoded.
    For 'target', pos/neg is by latent identity (labels are withheld)."""
    if role == "source_train":
        n = splits[role]
        pos = len(n["sim_lens_rows"])
        neg = (len(n["neg_mlfp"]["locations"])
               + len(n["neg_bare_deflector"]["locations"])
               + len(n["neg_random"]["locations"]))
    elif role == "val":
        n = splits[role]
        pos = len(n["sim_lens_rows"])
        neg = (len(n["neg_bare_deflector"]["locations"])
               + len(n["neg_random"]["locations"]))
    elif role == "target":
        f = splits["real_lenses_kfold"]["folds"][fold]
        pos = len(f["target_lens_rows"])
        neg = len(f["target_neg_locations"])
    elif role == "test":
        f = splits["real_lenses_kfold"]["folds"][fold]
        pos = len(f["test_lens_rows"])
        neg = len(splits["test_negatives_shared"]["locations"])
    else:
        raise ValueError(role)
    return pos + neg, pos, neg


def sample_indices(n, k, seed=0):
    """Deterministic sample of up to k indices, always including 0 and n-1."""
    rng = np.random.default_rng(seed)
    if n <= k:
        return list(range(n))
    idx = set(rng.choice(n, size=k, replace=False).tolist())
    idx.update((0, n - 1))
    return sorted(idx)


def check_role(splits, role, fold=None):
    tag = role + (f"[fold {fold}]" if fold is not None else "")
    print(f"\n[{tag}]")

    ds = LensDataset(SPLITS_PATH, DATA_ROOT, role=role, fold=fold, aug="none")

    # ---- 1. counts match splits.json exactly ----
    total, exp_pos, exp_neg = expected_counts(splits, role, fold)
    check(len(ds) == total, f"count: len(ds)={len(ds)} == splits declares {total}")
    check(ds.images.shape[0] == total,
          f"count: {ds.images.shape[0]} raw images loaded == {total}")

    # positive/negative composition (target uses latent, labels are withheld)
    if role == "target":
        got_pos = int((ds.latent_labels == 1).sum())
        got_neg = int((ds.latent_labels == 0).sum())
    else:
        got_pos = int((ds.labels == 1).sum())
        got_neg = int((ds.labels == 0).sum())
    check(got_pos == exp_pos and got_neg == exp_neg,
          f"composition: pos={got_pos}=={exp_pos}, neg={got_neg}=={exp_neg}")

    # ---- domain code sanity ----
    exp_domain = SOURCE if role in ("source_train", "val") else TARGET
    check(all(int(d) == exp_domain for d in ds.domains),
          f"domain: all items domain=={'SOURCE' if exp_domain==SOURCE else 'TARGET'}")

    # ---- label-free contract for the target role ----
    if role == "target":
        check(all(int(l) == UNLABELED for l in ds.labels),
              "label-free: every stored label is UNLABELED (-1)")

    # ---- 2/3/4. shape / dtype / finite / range on a real sample ----
    idxs = sample_indices(len(ds), SAMPLE_PER_ROLE, seed=fold or 0)
    bad_shape = bad_dtype = 0
    gmin, gmax = np.inf, -np.inf
    n_nonfinite = 0
    leak = 0
    for i in idxs:
        item = ds[i]
        img = item["image"]
        if tuple(img.shape) != (1, 224, 224):
            bad_shape += 1
        if img.dtype != torch.float32:
            bad_dtype += 1
        if not torch.isfinite(img).all():
            n_nonfinite += 1
        gmin = min(gmin, float(img.min()))
        gmax = max(gmax, float(img.max()))
        if role == "target" and int(item["label"]) != UNLABELED:
            leak += 1

    check(bad_shape == 0, f"shape: all {len(idxs)} sampled images are (1,224,224)")
    check(bad_dtype == 0, f"dtype: all {len(idxs)} sampled images are float32")
    check(n_nonfinite == 0,
          f"finite: no NaN/inf in {len(idxs)} sampled images")
    check(gmin >= RANGE_LO - TOL and gmax <= RANGE_HI + TOL,
          f"range: sampled values in [{RANGE_LO},{RANGE_HI}] "
          f"(observed min={gmin:.4f} max={gmax:.4f}; stretch ceiling=asinh(1)=0.881)")
    if role == "target":
        check(leak == 0, "label-free: __getitem__ never returned a real label")

    return ds


def run():
    torch.manual_seed(0)

    if not os.path.exists(SPLITS_PATH):
        print(f"SKIP: real splits not found at {SPLITS_PATH}")
        print("      (set LENS_DATA_ROOT to point at the data root)")
        return 3

    with open(SPLITS_PATH) as fh:
        splits = json.load(fh)

    print("=" * 68)
    print("REAL-DATA smoke test for LensDataset")
    print(f"  data_root : {DATA_ROOT}")
    print(f"  schema    : {splits.get('schema')}  (seed={splits.get('seed')}, "
          f"n_folds={splits.get('n_folds')})")
    print("=" * 68)

    check_role(splits, "source_train")
    check_role(splits, "val")
    check_role(splits, "target", fold=0)
    check_role(splits, "test", fold=0)

    print("\n" + "=" * 68)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL REAL-DATA CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
