"""
probe_size_native.py  --  the clean 64-vs-96 size test, native at both sizes.

Lenses are cut NATIVE at 64 and 96 px from the already-centred 300x300 VIS_FLUX
archive stamps (no mosaic loading, no 10" offset, no downsampling). Negatives are
the NATIVE 64 and 96 px cutouts made on ESA Datalabs (neg_test_64px.npz /
neg_test_96px.npz). Each arm pairs same-size lenses with same-size negatives, so
there is no size shortcut AND no resampling artifact (the thing that broke the
earlier 64blur control). Everything gets the identical Zoobot [0,1] stretch.

Two arms:
  64 : lens@64  vs  neg@64
  96 : lens@96  vs  neg@96
If 96 beats 64 by more than the fold std -> field of view carries real signal,
recut the pipeline at 96. If within fold std -> 64 is adequate.

Run from repo root:   python scripts\\probe_size_native.py
(zoobot_encoder.py alongside this file.)
"""
from __future__ import annotations
import io, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

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
NEG = {64: ARCH_DIR / "neg_test_64px.npz", 96: ARCH_DIR / "neg_test_96px.npz"}
SIZES = [64, 96]
ARCH_SRC = 300                      # archive stamp size
VIS_EXT = "VIS_FLUX"
RMS_EXT = "VIS_RMS"
RMS_SENTINEL = 1e15
CENTROID_SKIP_PIX = 20              # skip lenses whose light is this far off centre (recenter set)
N_SPLITS = 5
FPR_TARGETS = [1e-2, 1e-3]
SEED = 0
# ----------------------------------------


def centre_crop(a, n):
    h, w = a.shape
    y0, x0 = (h - n) // 2, (w - n) // 2
    return a[y0:y0 + n, x0:x0 + n]


def centroid_offset(v):
    h, w = v.shape
    half = h // 4
    sub = v[h//2-half:h//2+half, w//2-half:w//2+half]
    wts = np.clip(np.nan_to_num(sub), 0, None)
    if wts.sum() <= 0:
        return np.inf
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    my = (yy*wts).sum()/wts.sum(); mx = (xx*wts).sum()/wts.sum()
    return float(np.hypot(my-(sub.shape[0]-1)/2, mx-(sub.shape[1]-1)/2))


def read_vis(zf, idn):
    cand = [n for n in zf.namelist() if Path(n).stem == idn and n.lower().endswith(".fits")]
    if not cand:
        return None
    try:
        with fits.open(io.BytesIO(zf.read(cand[0]))) as h:
            vis = np.asarray(h[VIS_EXT].data, dtype=np.float32)
            if RMS_EXT in [x.name for x in h]:
                rms = np.asarray(h[RMS_EXT].data, dtype=np.float32)
                vis = np.where(rms >= RMS_SENTINEL, 0.0, vis)
            return np.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def cut_lenses():
    man = pd.read_csv(MANIFEST, dtype={"id_str": str})
    avail = man[man["available"]]
    out = {s: [] for s in SIZES}
    ids, skipped = [], 0
    for arch, grp in avail.groupby("source"):
        zp = ARCH_DIR / arch
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp) as zf:
            for idn in grp["id_str"]:
                v = read_vis(zf, idn)
                if v is None or min(v.shape) < max(SIZES):
                    skipped += 1; continue
                if centroid_offset(v) > CENTROID_SKIP_PIX:    # drop mis-centred (recenter) lenses
                    skipped += 1; continue
                for s in SIZES:
                    out[s].append(centre_crop(v, s).astype(np.float32))
                ids.append(idn)
    return ids, {s: np.stack(out[s]) for s in SIZES}, skipped


def load_negs(s):
    with np.load(NEG[s]) as d:
        im = np.asarray(d["images"], dtype=np.float32)
    assert im.shape[1] == s, f"{NEG[s]} is {im.shape[1]}px, expected {s}px"
    return im


def tpr_at_fpr(y, sc, t):
    fpr, tpr, _ = roc_curve(y, sc)
    ok = fpr <= t
    return float(tpr[ok][-1]) if ok.any() else 0.0


def cv_probe(feats, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan); fa = []
    for tr, va in skf.split(feats, y):
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(feats[tr]), y[tr])
        oof[va] = clf.decision_function(sc.transform(feats[va]))
        fa.append(roc_auc_score(y[va], oof[va]))
    r = {"AUROC": roc_auc_score(y, oof), "fold_std": float(np.std(fa))}
    for t in FPR_TARGETS:
        r[f"TPR@{t:g}"] = tpr_at_fpr(y, oof, t)
    return r


def run(encoder):
    from zoobot_encoder import stretch_batch
    ids, lens, skipped = cut_lenses()
    print(f"lenses: {len(ids)} cut, {skipped} skipped (corrupt/too-small/mis-centred)")
    res = {}
    print(f"\n{'size':>5} {'n_lens':>7} {'n_neg':>6} {'AUROC':>8} {'±fold':>7} {'TPR@1e-2':>9} {'TPR@1e-3':>9}")
    for s in SIZES:
        negs = load_negs(s)
        lf = encoder.encode(stretch_batch(lens[s]))
        nf = encoder.encode(stretch_batch(negs))
        feats = np.concatenate([lf, nf]); y = np.r_[np.ones(len(lf)), np.zeros(len(nf))].astype(int)
        m = cv_probe(feats, y); res[s] = m
        print(f"{s:>5} {len(lf):>7} {len(nf):>6} {m['AUROC']:>8.4f} {m['fold_std']:>7.4f} "
              f"{m['TPR@0.01']:>9.4f} {m['TPR@0.001']:>9.4f}")
    d = res[96]["AUROC"] - res[64]["AUROC"]
    fs = max(res[64]["fold_std"], res[96]["fold_std"])
    print(f"\n96 vs 64 (native, AUROC): {d:+.4f}   fold_std~{fs:.4f}")
    if d > 2*fs:
        print("-> 96 clearly beats 64: field of view carries real signal. Recut the pipeline at 96.")
    elif d > fs:
        print("-> 96 modestly ahead (1-2x fold std): suggestive, consider 96 but not decisive.")
    else:
        print("-> within fold std: 64 is adequate. Keep 64; FOV is not the bottleneck.")
    print("(AUROC is the contrast read; TPR@1e-2 supports it. TPR@1e-3 is noisy at this neg count.)")


if __name__ == "__main__":
    from zoobot_encoder import FrozenZoobotEncoder
    run(FrozenZoobotEncoder())
