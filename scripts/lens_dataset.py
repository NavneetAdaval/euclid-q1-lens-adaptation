"""
lens_dataset.py
===============
`LensDataset` for the label-free sim-to-real strong-lens finding project on Euclid Q1.

Reads `data/splits/splits.json` (schema ``splits_v6_label_free_da``) as the SINGLE
SOURCE OF TRUTH and serves the four data roles:

    role="source_train"   labeled source domain  (sim lenses + 3 negative sources)
    role="val"            labeled source domain  (sim lenses + 2 negative sources)
    role="target"         UNLABELED target domain, per fold k   (needs fold=k)
    role="test"           labeled evaluation,      per fold k   (needs fold=k)

Design decisions (grounded in the research memo / PROJECT.md; tags: RB=research-backed,
EC=engineering-choice):

  * INPUT STRETCH  (RB) — byte-for-byte Walmsley `galaxy-datasets.fits_to_jpg_like_scaling`:
      per-image clamp(quantile(p_min), quantile(p_max)) -> rescale to [0,1] -> asinh(x*q),
      with (q=1.0, p_min=0, p_max=99.7). Output ~[0, asinh(1)=0.881], single channel,
      resized to 224, NO ImageNet mean/std. This is exactly what the frozen Zoobot
      greyscale ConvNeXT-Nano was trained on. A constant-image guard (returns zeros) is
      added because the upstream code does not handle p_max==p_min.

  * POSITIONAL SHIFT (RB) — translate=(0.05,0.05) ~= +-3px at native 64px. Euclid VIS is
      0.1"/px, so +-3px = 0.3" ~= 0.33 x median theta_E (0.92"). Matches HOLISMOKES' +-3px
      prescription, stays inside Zoobot's own training-augmentation range, and is applied
      at native resolution BEFORE the 224 resize, EQUALLY to all classes/domains so it can
      never become a class or domain shortcut.

  * n_views (RB) — 1 for DANN/WDGRL/CORAL, 2 (weak+strong) for AdaMatch. When n_views==2,
      __getitem__ returns an extra key "image_strong"; "image" is always the primary/weak
      view.

  * UNLABELED TARGET (RB) — the target role's items carry label = UNLABELED (-1). Their
      true identities live in `latent_labels` for DIAGNOSTICS ONLY and are never returned
      by __getitem__, so the DA loss cannot accidentally use them.

  * FROZEN ENCODER (EC) — this loader always serves IMAGES (never cached features), so it
      is unchanged by the encoder_mode switch (frozen / lp_then_lastblock / lp_then_lora).
      Deterministic eval features are cached downstream using aug="none".

The loader CONFORMS to the splits; it never re-derives assignments. Positional indices
are load-bearing: `sim_lens_rows` index into the CONCATENATION of every sim shard's
`lenses` in `sim_shards` order, and every `[shard_filename, row]` location is resolved
against the array in the directory implied by its reference type.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA = "splits_v6_label_free_da"
UNLABELED = -1                      # label sentinel for the (unlabeled) target role
SOURCE, TARGET = 0, 1              # domain codes

# npz array keys, per data source
SIM_LENS_KEY = "lenses"           # sim shards: (N,64,64) simulated positives
SIM_NONLENS_KEY = "nonlenses"     # sim shards: (N,64,64) matched bare-deflector negatives
IMG_KEY = "images"                # unlabeled shards / ml_fp shards / lens_cutouts

# directory layout relative to data_root (matches PROJECT.md section 6);
# overridable so tests can point at temporary directories.
DEFAULT_LAYOUT = {
    "sim_dir": "sim_arc_on_real",
    "unlabeled_dir": "q1_unlabeled/shards",
    "mlfp_dir": "ml_fp_cutouts/shards",
    "lens_cutouts": "q1_lenses_processed/lens_cutouts.npz",
}


# ---------------------------------------------------------------------------
# Input stretch (Zoobot / galaxy-datasets convention)
# ---------------------------------------------------------------------------

def zoobot_stretch(img: torch.Tensor,
                   arcsinh_q: float = 1.0,
                   percentile_min: float = 0.0,
                   percentile_max: float = 99.7) -> torch.Tensor:
    """Per-image percentile-clip -> [0,1] -> arcsinh, exactly as Walmsley's
    ``fits_to_jpg_like_scaling`` (galaxy-datasets), plus a constant-image guard.

    Expects a float32 tensor of shape (C,H,W); returns the same shape.
    """
    if img.dtype != torch.float32:
        img = img.float()
    p_min = torch.quantile(img, percentile_min / 100.0)
    p_max = torch.quantile(img, percentile_max / 100.0)
    denom = (p_max - p_min)
    if float(denom) <= 0.0:            # degenerate/constant image (upstream code crashes here)
        return torch.zeros_like(img)
    img = torch.clamp(img, p_min, p_max)
    img = (img - p_min) / denom
    img = torch.asinh(img * arcsinh_q)
    return img


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class LensTransform:
    """Stretch -> (optional geometric augmentation at native res) -> resize to out_size.

    aug levels
      "none"   : deterministic (stretch + resize). Used for val/test/diagnostics/caching.
      "weak"   : flips + rot90 + translate(+-3px). AdaMatch weak view.
      "strong" : weak + mild zoom + a little random erasing. AdaMatch strong view.

    All geometric ops run at the native resolution (before the resize), so the +-3px
    shift is a true native-pixel shift.
    """

    def __init__(self, aug: str = "none", out_size: int = 224,
                 translate=(0.05, 0.05), degrees: float = 90.0,
                 strong_scale=(0.9, 1.3), erase_scale=(0.002, 0.02),
                 stretch_kwargs: dict | None = None):
        assert aug in ("none", "weak", "strong"), aug
        self.aug = aug
        self.out_size = out_size
        self.stretch_kwargs = stretch_kwargs or {}
        self._resize = v2.Resize((out_size, out_size),
                                 interpolation=InterpolationMode.BILINEAR, antialias=True)

        geom = []
        if aug in ("weak", "strong"):
            geom += [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
            scale = strong_scale if aug == "strong" else None
            geom.append(v2.RandomAffine(
                degrees=degrees, translate=translate, scale=scale,
                interpolation=InterpolationMode.BILINEAR, fill=0.0))
        self._geom = v2.Compose(geom) if geom else None
        self._erase = (v2.RandomErasing(p=0.5, scale=erase_scale, ratio=(0.5, 2.0), value=0.0)
                       if aug == "strong" else None)

    def __call__(self, img2d: np.ndarray | torch.Tensor) -> torch.Tensor:
        # -> float32 (1,H,W)
        if isinstance(img2d, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(img2d)).float()
        else:
            t = img2d.float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[0] != 1:
            raise ValueError(f"expected single-channel image, got shape {tuple(t.shape)}")

        t = zoobot_stretch(t, **self.stretch_kwargs)     # stretch at native res
        if self._geom is not None:
            t = self._geom(t)                            # geometric aug at native res
        t = self._resize(t)                              # -> out_size
        if self._erase is not None:
            t = self._erase(t)
        return t.contiguous()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LensDataset(Dataset):
    """Serves one role of ``splits.json`` (schema splits_v6_label_free_da).

    Parameters
    ----------
    splits : str | dict           path to splits.json, or an already-loaded dict.
    data_root : str               repo data root (dirs resolved via `layout`).
    role : str                    one of {"source_train","val","target","test"}.
    fold : int | None             required for role in {"target","test"}.
    aug : str                     "none" | "weak" | "strong" (view 0 / weak view).
    n_views : int                 1, or 2 to also emit a "strong" view (AdaMatch).
    out_size : int                encoder input size (224 for Zoobot ConvNeXT).
    layout : dict | None          directory overrides (defaults to PROJECT.md layout).
    strict_schema : bool          error (else warn) if splits schema != expected.
    """

    ROLES = ("source_train", "val", "target", "test")

    def __init__(self, splits, data_root: str, role: str, fold: int | None = None,
                 aug: str = "none", n_views: int = 1, out_size: int = 224,
                 layout: dict | None = None, strict_schema: bool = True,
                 weak_aug: str | None = None, strong_aug: str = "strong",
                 stretch_kwargs: dict | None = None):
        super().__init__()
        if role not in self.ROLES:
            raise ValueError(f"role must be one of {self.ROLES}, got {role!r}")
        if role in ("target", "test") and fold is None:
            raise ValueError(f"role={role!r} requires fold=k")
        if n_views not in (1, 2):
            raise ValueError("n_views must be 1 or 2")

        self.role = role
        self.fold = fold
        self.n_views = n_views
        self.data_root = data_root
        self.layout = {**DEFAULT_LAYOUT, **(layout or {})}

        splits = self._load_splits(splits, strict_schema)
        self.splits = splits

        # view transforms. view0 uses `aug` (default depends on role); view1 = strong.
        if weak_aug is None:
            weak_aug = aug
        self._tf_view0 = LensTransform(weak_aug, out_size, stretch_kwargs=stretch_kwargs)
        self._tf_view1 = (LensTransform(strong_aug, out_size, stretch_kwargs=stretch_kwargs)
                          if n_views == 2 else None)

        # ---- assemble the item list for this role, then load pixels -----------
        self._descriptors: list[dict] = []
        self._build_descriptors()

        n = len(self._descriptors)
        self.images = np.empty((n, 64, 64), dtype=np.float32)
        self.labels = np.array([d["label"] for d in self._descriptors], dtype=np.int64)
        self.domains = np.array([d["domain"] for d in self._descriptors], dtype=np.int64)
        self.latent_labels = np.array([d["latent"] for d in self._descriptors], dtype=np.int64)
        self.kinds = [d["kind"] for d in self._descriptors]

        self._load_pixels()

        # convenience index maps for balanced samplers built downstream
        self.indices_by_label = self._group_indices(self.labels)
        self.indices_by_latent = self._group_indices(self.latent_labels)

    # -- splits loading / validation ---------------------------------------

    @staticmethod
    def _load_splits(splits, strict_schema):
        if isinstance(splits, str):
            with open(splits) as fh:
                splits = json.load(fh)
        schema = splits.get("schema")
        if schema != SCHEMA:
            msg = f"splits schema is {schema!r}, expected {SCHEMA!r}"
            if strict_schema:
                raise ValueError(msg)
            print(f"[LensDataset] WARNING: {msg}")
        return splits

    # -- descriptor assembly (one dict per served item) --------------------

    def _add(self, kind, label, domain, latent, ref):
        """ref describes where the raw 64x64 image comes from:
             ("sim_lens", global_row)
             ("sim_nonlens", shard_filename, idx)
             ("file", abs_path, array_key, row)
        """
        self._descriptors.append(
            dict(kind=kind, label=label, domain=domain, latent=latent, ref=ref))

    def _unlabeled_path(self, shard_fn):
        return os.path.join(self.data_root, self.layout["unlabeled_dir"], shard_fn)

    def _mlfp_path(self, shard_fn):
        return os.path.join(self.data_root, self.layout["mlfp_dir"], shard_fn)

    def _lens_cutouts_path(self):
        return os.path.join(self.data_root, self.layout["lens_cutouts"])

    def _build_descriptors(self):
        s = self.splits
        if self.role in ("source_train", "val"):
            node = s[self.role]
            # positives: simulated lenses (domain=source, label=1)
            for g in node["sim_lens_rows"]:
                self._add("sim_lens", 1, SOURCE, 1, ("sim_lens", int(g)))
            # negatives: ML-FP (source_train only)
            if self.role == "source_train" and "neg_mlfp" in node:
                for shard_fn, idx in node["neg_mlfp"]["locations"]:
                    self._add("mlfp", 0, SOURCE, 0,
                              ("file", self._mlfp_path(shard_fn), IMG_KEY, int(idx)))
            # negatives: bare deflectors (sim shards' nonlenses)
            for shard_fn, idx in node["neg_bare_deflector"]["locations"]:
                self._add("bare_deflector", 0, SOURCE, 0,
                          ("sim_nonlens", shard_fn, int(idx)))
            # negatives: random real galaxies (unlabeled pool)
            for shard_fn, idx in node["neg_random"]["locations"]:
                self._add("random", 0, SOURCE, 0,
                          ("file", self._unlabeled_path(shard_fn), IMG_KEY, int(idx)))

        elif self.role == "target":
            fold = self._get_fold()
            lc = self._lens_cutouts_path()
            # unlabeled positives: real lenses from the OTHER four folds (label withheld)
            for row in fold["target_lens_rows"]:
                self._add("real_lens", UNLABELED, TARGET, 1,
                          ("file", lc, IMG_KEY, int(row)))
            # unlabeled negatives: fresh per-fold random galaxies (label withheld)
            for shard_fn, idx in fold["target_neg_locations"]:
                self._add("random", UNLABELED, TARGET, 0,
                          ("file", self._unlabeled_path(shard_fn), IMG_KEY, int(idx)))

        elif self.role == "test":
            fold = self._get_fold()
            lc = self._lens_cutouts_path()
            # positives: this fold's held-out real lenses (labels used at EVAL only)
            for row in fold["test_lens_rows"]:
                self._add("real_lens", 1, TARGET, 1, ("file", lc, IMG_KEY, int(row)))
            # negatives: the SHARED P4 pool (identical every fold)
            for shard_fn, idx in s["test_negatives_shared"]["locations"]:
                self._add("random", 0, TARGET, 0,
                          ("file", self._unlabeled_path(shard_fn), IMG_KEY, int(idx)))

    def _get_fold(self):
        folds = self.splits["real_lenses_kfold"]["folds"]
        for f in folds:
            if int(f["fold"]) == int(self.fold):
                return f
        raise KeyError(f"fold {self.fold} not found (have {[f['fold'] for f in folds]})")

    # -- pixel loading (each file opened exactly once) ---------------------

    def _load_pixels(self):
        sim_lens_targets = []          # (global_row, item_pos)
        bare_by_shard = defaultdict(list)   # shard_fn -> [(idx, item_pos)]
        file_refs = []                 # (abs_path, array_key, row, item_pos)

        for pos, d in enumerate(self._descriptors):
            ref = d["ref"]
            if ref[0] == "sim_lens":
                sim_lens_targets.append((ref[1], pos))
            elif ref[0] == "sim_nonlens":
                bare_by_shard[ref[1]].append((ref[2], pos))
            elif ref[0] == "file":
                file_refs.append((ref[1], ref[2], ref[3], pos))
            else:
                raise AssertionError(ref)

        self._load_sim(sim_lens_targets, bare_by_shard)
        self._load_grouped(file_refs)

    def _load_sim(self, sim_lens_targets, bare_by_shard):
        """Single pass over the sim shards in `sim_shards` order (the load-bearing
        concatenation order for `sim_lens_rows`)."""
        if not sim_lens_targets and not bare_by_shard:
            return
        sim_shards = self.splits["sim_shards"]
        targets = sorted(sim_lens_targets, key=lambda t: t[0])
        ptr = 0
        offset = 0
        for shard_fn in sim_shards:
            path = os.path.join(self.data_root, self.layout["sim_dir"], shard_fn)
            need_lens = ptr < len(targets) and targets[ptr][0] < offset + 10**12
            need_bare = shard_fn in bare_by_shard
            with np.load(path) as npz:
                L = int(npz[SIM_LENS_KEY].shape[0])
                # assign any sim_lens global rows that fall inside [offset, offset+L)
                if ptr < len(targets) and targets[ptr][0] < offset + L:
                    lenses = npz[SIM_LENS_KEY]
                    while ptr < len(targets) and targets[ptr][0] < offset + L:
                        g, item_pos = targets[ptr]
                        if g < offset:
                            raise IndexError(f"sim_lens_row {g} unresolved (below offset {offset})")
                        self.images[item_pos] = lenses[g - offset].astype(np.float32)
                        ptr += 1
                if need_bare:
                    nonlenses = npz[SIM_NONLENS_KEY]
                    nN = int(nonlenses.shape[0])
                    for idx, item_pos in bare_by_shard[shard_fn]:
                        if not (0 <= idx < nN):
                            raise IndexError(
                                f"bare_deflector idx {idx} out of range in {shard_fn} (n={nN})")
                        self.images[item_pos] = nonlenses[idx].astype(np.float32)
            offset += L
        if ptr != len(targets):
            raise IndexError(
                f"{len(targets) - ptr} sim_lens_rows exceed total sim lenses ({offset}); "
                f"largest requested row = {targets[-1][0]}")
        # any bare-deflector shards referenced that were not in sim_shards?
        missing = set(bare_by_shard) - set(sim_shards)
        if missing:
            raise KeyError(f"bare_deflector references shards not in sim_shards: {sorted(missing)}")

    def _load_grouped(self, file_refs):
        by_file = defaultdict(list)
        for path, key, row, pos in file_refs:
            by_file[(path, key)].append((row, pos))
        for (path, key), rows in by_file.items():
            with np.load(path) as npz:
                arr = npz[key]
                n = int(arr.shape[0])
                for row, pos in rows:
                    if not (0 <= row < n):
                        raise IndexError(f"row {row} out of range for {key} in {path} (n={n})")
                    self.images[pos] = arr[row].astype(np.float32)

    @staticmethod
    def _group_indices(labels):
        out = defaultdict(list)
        for i, v in enumerate(labels.tolist()):
            out[int(v)].append(i)
        return {k: np.array(v, dtype=np.int64) for k, v in out.items()}

    # -- access ------------------------------------------------------------

    def __len__(self):
        return len(self._descriptors)

    def __getitem__(self, i):
        raw = self.images[i]
        item = {
            "image": self._tf_view0(raw),
            "label": int(self.labels[i]),
            "domain": int(self.domains[i]),
            "kind": self.kinds[i],
            "index": i,
        }
        if self.n_views == 2:
            item["image_strong"] = self._tf_view1(raw)   # independent aug draw
        return item

    def get_raw(self, i):
        """Return the untransformed 64x64 image and its provenance. For diagnostics /
        tests: lets index-resolution be verified independently of the transform."""
        return self.images[i].copy(), dict(self._descriptors[i])

    def summary(self):
        from collections import Counter
        c = Counter(self.kinds)
        pos = int((self.labels == 1).sum())
        neg = int((self.labels == 0).sum())
        unl = int((self.labels == UNLABELED).sum())
        return (f"LensDataset(role={self.role}, fold={self.fold}, n={len(self)}, "
                f"pos={pos}, neg={neg}, unlabeled={unl}, kinds={dict(c)})")

    def __repr__(self):
        return self.summary()
