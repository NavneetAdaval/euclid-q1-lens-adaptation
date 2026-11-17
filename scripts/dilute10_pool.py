"""
dilute10_pool.py
================
The pool loader for the 1:10 diluted-target experiment (PROJECTv7.md 7.36-7.37,
brief_dilute10_code.md section 1.1).

* ``load_pool`` verifies ``data/splits/dilute10_pool.json`` against PART 1 before anything
  reads it: part 1's self-certifying digest must be the pinned ``PART1_SHA256``, and the pool
  file's whole-file sha256 must equal the digest part 1 records. Either mismatch refuses.
* ``diluted_splits`` returns a copy of ``splits.json`` in which fold k's target
  (``target_lens_rows``, ``target_neg_object_ids``, ``target_neg_locations``) is the pool's.
  ``LensDataset`` accepts a splits dict, so the target dataset is served by the UNCHANGED
  ``LensDataset`` -- same transform, same ``n_views``, same item ordering (lenses, then
  negatives); the existing members keep their positions and the fresh negatives follow.
* ``lens_dataset_factory`` wraps a ``LensDataset`` class so that only ``role="target"`` is
  redirected to the diluted pool; every other role is passed through untouched. The runners
  bind it in place of the name ``LensDataset`` inside the module they wrap.
* ``pinned_holdout`` returns, for DANN and BNM, EXACTLY the 78 holdout objects the original
  runs recorded, mapped by object identity to their positions in the diluted pool, in the
  shape ``dann_round5.build_target_holdout`` returns.

Latent labels exist on the dataset object (``LensDataset.latent_labels``) for logging; this
module never passes them to a sampler or a loss. Lens identity here is POSITIONAL, from the
pool file (position < n_lenses), which is what the lens-fraction logger reads.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

POOL_PATH = os.path.join(REPO, "data", "splits", "dilute10_pool.json")
PART1_PATH = os.path.join(REPO, "runs", "dilute10_prereg_part1.json")
SPLITS_PATH = os.path.join(REPO, "data", "splits", "splits.json")
PART1_SHA256 = "bd537a025e81833e73713649c9eed68fd6a0fd3152ee29240f9a333f7c649d85"
FOLDS = (1, 2, 3, 4)


class PoolVerificationError(RuntimeError):
    """The pool file or part 1 does not hash to what part 1 records."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _self_cert(d):
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def verify_part1(part1_path=PART1_PATH):
    with open(part1_path, encoding="utf-8") as fh:
        p1 = json.load(fh)
    got = _self_cert(p1)
    if got != PART1_SHA256 or p1.get("sha256_of_body_without_this_field") != PART1_SHA256:
        raise PoolVerificationError(f"part 1 digest {got} != pinned {PART1_SHA256}")
    return p1


def load_pool(pool_path=POOL_PATH, part1_path=PART1_PATH):
    """The pool dict, after verifying it against part 1. Refuses on any mismatch."""
    p1 = verify_part1(part1_path)
    want = p1["pool"]["whole_file_sha256"]
    got = sha256_file(pool_path)
    if got != want:
        raise PoolVerificationError(
            f"{pool_path} hashes to {got}; part 1 records {want}. Refusing to serve it.")
    with open(pool_path, encoding="utf-8") as fh:
        pool = json.load(fh)
    pool["_verified"] = {"pool_sha256": got, "part1_sha256": PART1_SHA256,
                         "pool_path": os.path.abspath(pool_path)}
    return pool


def pool_fold(pool, fold):
    for fd in pool["folds"]:
        if int(fd["fold"]) == int(fold):
            return fd
    raise KeyError(f"fold {fold} is not in the diluted pool (folds {FOLDS})")


def load_splits(splits=SPLITS_PATH):
    if isinstance(splits, dict):
        return splits
    with open(splits, encoding="utf-8") as fh:
        return json.load(fh)


def diluted_splits(splits, pool, fold):
    """A copy of splits.json whose fold-k target is the diluted pool's."""
    sp = load_splits(splits)
    out = dict(sp)
    out["real_lenses_kfold"] = copy.deepcopy(sp["real_lenses_kfold"])
    fd = pool_fold(pool, fold)
    for f in out["real_lenses_kfold"]["folds"]:
        if int(f["fold"]) == int(fold):
            if list(f["target_lens_rows"]) != list(fd["target_lens_rows"]):
                raise PoolVerificationError(f"fold {fold}: pool lens rows differ from splits")
            n_ex = int(fd["n_existing_negatives"])
            if [int(o) for o in f["target_neg_object_ids"]] != \
                    [int(o) for o in fd["target_neg_object_ids"][:n_ex]]:
                raise PoolVerificationError(f"fold {fold}: existing negatives moved")
            f["target_neg_object_ids"] = [int(o) for o in fd["target_neg_object_ids"]]
            f["target_neg_locations"] = [list(x) for x in fd["target_neg_locations"]]
            return out
    raise KeyError(f"fold {fold} not in splits.json")


def lens_dataset_factory(orig_cls, pool, fold):
    """A drop-in for ``LensDataset`` that serves fold k's target from the diluted pool."""
    fold = int(fold)

    def factory(splits, data_root, *args, **kw):
        role = kw.get("role", args[0] if args else None)
        if role == "target":
            f = kw.get("fold", args[1] if len(args) > 1 else None)
            if int(f) != fold:
                raise PoolVerificationError(f"target fold {f} requested; this run is {fold}")
            ds = orig_cls(diluted_splits(splits, pool, fold), data_root, *args, **kw)
            fd = pool_fold(pool, fold)
            if len(ds) != fd["n_lenses"] + fd["n_negatives"]:
                raise PoolVerificationError(f"diluted target has {len(ds)} items")
            ds.dilute10 = {"fold": fold, "n_lenses": int(fd["n_lenses"]),
                           "n_negatives": int(fd["n_negatives"]),
                           "pool_sha256": pool["_verified"]["pool_sha256"]}
            return ds
        return orig_cls(splits, data_root, *args, **kw)

    factory.__wrapped__ = orig_cls
    factory.dilute10_fold = fold
    return factory


# --------------------------------------------------------------------------------------
# the DANN/BNM holdout: the 78 objects the original runs recorded
# --------------------------------------------------------------------------------------
def recorded_holdout_positions(fold):
    """Positions in the ORIGINAL pool, from DANN x nb3's run.json (the runs that record them)."""
    p = os.path.join(REPO, "runs", "dann_main", f"fold{fold}", "extracted", f"fold{fold}",
                     "run.json")
    with open(p, encoding="utf-8") as fh:
        j = json.load(fh)
    return [int(i) for i in j["dann_d_holdout_indices"]], p


class _LatentStub:
    """Just enough of a target LensDataset for build_target_holdout: lenses then negatives."""

    def __init__(self, n_lenses, n_neg):
        self._n = n_lenses + n_neg
        self.indices_by_latent = {1: np.arange(n_lenses, dtype=np.int64),
                                  0: np.arange(n_lenses, self._n, dtype=np.int64)}

    def __len__(self):
        return self._n


def _identity(fold_entry, pos):
    n_l = len(fold_entry["target_lens_rows"])
    if pos < n_l:
        return ("lens_row", int(fold_entry["target_lens_rows"][pos]))
    return ("neg", int(fold_entry["target_neg_object_ids"][pos - n_l]))


def pinned_holdout(pool, fold, splits=SPLITS_PATH, *, frac=0.10, seed=None):
    """The original runs' 78 holdout objects, at their positions in the diluted pool.

    Verified three ways before it is returned:
      1. the recorded positions reproduce from ``dann_round5.build_target_holdout`` on the
         ORIGINAL pool's latent structure at the original seed (which is also how BNM, which
         recorded no indices, drew its slice);
      2. the 78 objects, mapped by identity into the diluted pool, are byte-identical (as
         a JSON list of identities) to the original's;
      3. the training pool is everything else: 352:3871 on folds 1-3, 353:3881 on fold 4.
    """
    import dann_round5
    fold = int(fold)
    seed = (42 + fold) if seed is None else int(seed)
    sp = load_splits(splits)
    orig = next(f for f in sp["real_lenses_kfold"]["folds"] if int(f["fold"]) == fold)
    fd = pool_fold(pool, fold)
    n_l = len(orig["target_lens_rows"])

    rec, rec_path = recorded_holdout_positions(fold)
    redrawn = dann_round5.build_target_holdout(
        _LatentStub(n_l, len(orig["target_neg_object_ids"])), frac=frac, seed=seed)
    if sorted(int(i) for i in redrawn["holdout"]) != sorted(rec):
        raise PoolVerificationError(f"fold {fold}: recorded holdout does not reproduce")

    ident_orig = [_identity(orig, p) for p in sorted(rec)]
    n_pool = int(fd["n_lenses"]) + int(fd["n_negatives"])
    where = {_identity(fd, p): p for p in range(n_pool)}
    dpos = [where[i] for i in ident_orig]
    ident_new = [_identity(fd, p) for p in dpos]
    b_orig = json.dumps(ident_orig).encode()
    b_new = json.dumps(ident_new).encode()
    if b_orig != b_new or len(dpos) != len(rec):
        raise PoolVerificationError(f"fold {fold}: holdout objects differ")

    held = set(dpos)
    train_pool = np.array([p for p in range(n_pool) if p not in held], dtype=np.int64)
    n_lens = int(fd["n_lenses"])
    hold = np.sort(np.asarray(dpos, dtype=np.int64))
    return {
        "frac_requested": float(frac), "frac_actual": float(hold.size / n_pool),
        "seed": seed, "n_pool": n_pool,
        # positional, from the pool file -- recorded, never handed to the sampler
        "train_pos": train_pool[train_pool < n_lens],
        "train_neg": train_pool[train_pool >= n_lens],
        "holdout": hold, "holdout_pos": hold[hold < n_lens], "holdout_neg": hold[hold >= n_lens],
        "n_train": int(train_pool.size), "n_holdout": int(hold.size),
        "train_pool": train_pool,
        "pinned_from": os.path.relpath(rec_path, REPO).replace("\\", "/"),
        "holdout_identity_sha256": hashlib.sha256(b_new).hexdigest(),
        "training_pool_counts": f"{int((train_pool < n_lens).sum())}:"
                                f"{int((train_pool >= n_lens).sum())}",
        "note": ("the original runs' 78 holdout objects, mapped by identity to the diluted "
                 "pool (dilute10_pool.pinned_holdout); held out of the target sampler's pool"),
    }
