"""
probe_size_contrast.py  --  does field-of-view (64 vs wider) limit lens signal,
and is any gain real FOV rather than a denoising artifact of downsampling?

Every image entering the encoder is 64x64 (lenses AND negatives) -> no size
shortcut. Four arms, differing only in what each 64x64 lens image represents:
  64       : native 64px crop                              (baseline)
  96to64   : 96px crop, area-downsampled to 64   (more FOV + incidental denoise)
  128to64  : 128px crop, area-downsampled to 64  (even more FOV + denoise)
  64blur   : 64px crop, area-downsampled 64->43 then back to 64
             (the SAME low-pass/denoise as 96to64, but NO extra field of view)

Read:
  FOV gain  ~ (96to64 - 64) - (64blur - 64) = 96to64 - 64blur
  If 64blur ~ 64, the 96to64 gain is real field of view -> recut at 96.
  If 64blur also gains, part of the effect is denoising, not FOV.

Negatives: partition_1_reserve (NOT the sealed test-neg), same set for all arms.
Run from repo root:   python scripts\\probe_size_contrast.py
(zoobot_encoder.py must sit alongside this file.)
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from astropy.io import fits
except ImportError:
    sys.exit("Need astropy:  pip install astropy")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------- config ----------------
ROOT = Path(".")
ARCH_DIR = ROOT / "data" / "q1_lenses"
MANIFEST = ARCH_DIR / "lens_manifest_positive.csv"
SHARD_DIR = ROOT / "data" / "q1_unlabeled" / "shards"
SPLITS = ROOT / "data" / "splits" / "splits_v5.json"
OUT_SIZE = 64
MAX_CROP = 128
NEG_CAP = 3000
N_SPLITS = 5
FPR_TARGETS = [1e-2, 1e-3]
SEED = 0
# arm name -> (crop_px, resample mode to reach 64)
ARMS = {
    "64":      (64, None),
    "96to64":  (96, "area"),
    "128to64": (128, "area"),
    "64blur":  (64, "blur43"),   # denoise-only control: 64->43->64, no extra FOV
}
BASELINE = "64"
# ----------------------------------------


def centre_crop(a, n):
    h, w = a.shape
    y0, x0 = (h - n) // 2, (w - n) // 2
    return a[y0:y0 + n, x0:x0 + n]


def _resize(a, n, mode):
    t = torch.from_numpy(a[None, None].astype(np.float32))
    if mode == "area":
        t = F.interpolate(t, size=(n, n), mode="area")
    else:  # bilinear for upsampling
        t = F.interpolate(t, size=(n, n), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def make_view(v, crop, mode):
    c = centre_crop(v, crop).astype(np.float32)
    if mode is None:
        return c                                   # crop is already 64
    if mode == "area":
        return _resize(c, OUT_SIZE, "area")
    if mode == "blur43":
        return _resize(_resize(c, 43, "area"), OUT_SIZE, "bilinear")
    raise ValueError(mode)


def read_vis(zf, idn):
    cand = [n for n in zf.namelist() if Path(n).stem == idn and n.lower().endswith(".fits")]
    if not cand:
        return None
    try:
        with fits.open(io.BytesIO(zf.read(cand[0]))) as h:
            return np.asarray(h["VIS_FLUX"].data, dtype=np.float32)
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def cut_lenses():
    man = pd.read_csv(MANIFEST, dtype={"id_str": str})
    avail = man[man["available"]]
    views = {a: [] for a in ARMS}
    ids, skipped = [], []
    for arch, grp in avail.groupby("source"):
        zp = ARCH_DIR / arch
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp) as zf:
            for idn in grp["id_str"]:
                v = read_vis(zf, idn)
                if v is None or isinstance(v, str) or min(v.shape) < MAX_CROP:
                    skipped.append(idn)
                    continue
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                for a, (crop, mode) in ARMS.items():
                    views[a].append(make_view(v, crop, mode))
                ids.append(idn)
    return ids, {a: np.stack(views[a]).astype(np.float32) for a in ARMS}, skipped


def load_reserve_negs(cap):
    with open(SPLITS) as f:
        splits = json.load(f)
    try:
        locs = splits["unlabeled_euclid"]["partition_1_reserve"]
    except KeyError:
        sys.exit("splits_v5.json has no unlabeled_euclid/partition_1_reserve key.")
    rng = np.random.default_rng(SEED)
    locs = [tuple(x) for x in locs]
    rng.shuffle(locs)
    locs = locs[:cap]
    by_shard = {}
    for name, idx in locs:
        by_shard.setdefault(name, []).append(int(idx))
    negs = []
    for name, idxs in by_shard.items():
        with np.load(SHARD_DIR / name) as d:
            arr = d["images"]
            for i in idxs:
                negs.append(np.nan_to_num(arr[i], nan=0.0, posinf=0.0, neginf=0.0))
    return np.stack(negs).astype(np.float32)


def tpr_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target
    return float(tpr[ok][-1]) if ok.any() else 0.0


def cv_probe(feats, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fa = []
    for tr, va in skf.split(feats, y):
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(feats[tr]), y[tr])
        p = clf.decision_function(sc.transform(feats[va]))
        oof[va] = p
        fa.append(roc_auc_score(y[va], p))
    res = {"AUROC": roc_auc_score(y, oof), "fold_std": float(np.std(fa))}
    for t in FPR_TARGETS:
        res[f"TPR@{t:g}"] = tpr_at_fpr(y, oof, t)
    return res


def run(encoder):
    from zoobot_encoder import stretch_batch
    print("Cutting lens arms", list(ARMS), "...")
    ids, views, skipped = cut_lenses()
    print(f"  {len(ids)} lenses, {len(skipped)} skipped")
    negs = load_reserve_negs(NEG_CAP)
    print(f"  {len(negs)} reserve negatives\n")

    neg_feats = encoder.encode(stretch_batch(negs))
    y = np.r_[np.ones(len(ids)), np.zeros(len(negs))].astype(int)

    print(f"{'arm':>9} {'AUROC':>8} {'±fold':>7} {'TPR@1e-2':>9} {'TPR@1e-3':>9}")
    out = {}
    for a in ARMS:
        lf = encoder.encode(stretch_batch(views[a]))
        m = cv_probe(np.concatenate([lf, neg_feats]), y)
        out[a] = m
        print(f"{a:>9} {m['AUROC']:>8.4f} {m['fold_std']:>7.4f} "
              f"{m['TPR@0.01']:>9.4f} {m['TPR@0.001']:>9.4f}")

    b = out[BASELINE]["AUROC"]
    fov = out["96to64"]["AUROC"] - out["64blur"]["AUROC"]
    denoise = out["64blur"]["AUROC"] - b
    print(f"\n96to64 vs 64 (raw gain) : {out['96to64']['AUROC']-b:+.4f}")
    print(f"64blur vs 64 (denoise)  : {denoise:+.4f}")
    print(f"isolated FOV (96to64 - 64blur): {fov:+.4f}   fold_std~{out['64']['fold_std']:.4f}")
    print("\nIf isolated-FOV >> fold_std and denoise ~ 0  -> recut at 96 (real FOV gain).")
    print("If denoise carries most of the gain          -> FOV is not the lever; reconsider.")


if __name__ == "__main__":
    from zoobot_encoder import FrozenZoobotEncoder
    run(FrozenZoobotEncoder())
