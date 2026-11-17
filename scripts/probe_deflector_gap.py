"""
probe_deflector_gap.py -- is the SYNTHETIC deflector the sim->real transfer blocker?

Compares your sim NON-lenses (source_add=False renders = synthetic Sersic deflectors)
against real Euclid negatives (real bright galaxies, no arc). Both are "a galaxy, no
arc", so any separability is purely the synthetic-vs-real DEFLECTOR gap, isolated
from the arc. Three read-outs:
  1. 3-stat AUROC  : can mean/std/p99 alone separate them? (the brightness shortcut)
  2. encoder AUROC : can frozen Zoobot features separate them? (full deflector gap)
  3. neighbour test: for each synthetic deflector, fraction of k-NN that are REAL.

Verdict:
  encoder AUROC ~1.0  -> synthetic deflector is trivially non-real -> arc-on-real JUSTIFIED.
  ~0.7-0.85           -> real gap -> arc-on-real helps.
  ~0.5-0.6            -> synthetic deflector already looks real -> gap is the ARC, not the
                         deflector; arc-on-real buys little.

Run from repo root:  python scripts\\probe_deflector_gap.py
(zoobot_encoder.py alongside.)
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(".")
SIM_NONLENS_DIR = ROOT / "data" / "sim_nonlens" / "shards"
REAL_NEG_DIR = ROOT / "data" / "q1_unlabeled" / "shards"
N_PER_CLASS = 1000
K_NN = 10
SEED = 0


def load_stamps(d, n, avoid_partition=4):
    files = sorted(glob.glob(str(d / "*.npz")))
    if not files:
        sys.exit(f"no shards in {d}")
    rng = np.random.default_rng(SEED)
    out = []
    for f in files:
        with np.load(f) as z:
            im = z["images"]
            if "partitions" in z.files and avoid_partition is not None:
                im = im[z["partitions"] != avoid_partition]
            out.append(np.asarray(im, dtype=np.float32))
        if sum(len(x) for x in out) >= n * 3:
            break
    allim = np.concatenate(out, axis=0)
    idx = rng.choice(len(allim), size=min(n, len(allim)), replace=False)
    return allim[idx]


def three_stats(imgs):
    flat = imgs.reshape(len(imgs), -1)
    return np.stack([flat.mean(1), flat.std(1), np.percentile(flat, 99, axis=1)], axis=1)


def cv_auroc(feats, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, va in skf.split(feats, y):
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(feats[tr]), y[tr])
        oof[va] = clf.decision_function(sc.transform(feats[va]))
    return roc_auc_score(y, oof)


def run(encoder):
    from zoobot_encoder import stretch_batch
    sim = load_stamps(SIM_NONLENS_DIR, N_PER_CLASS, avoid_partition=None)
    real = load_stamps(REAL_NEG_DIR, N_PER_CLASS, avoid_partition=4)
    print(f"synthetic deflectors (sim_nonlens): {len(sim)}   real galaxies (neg): {len(real)}\n")

    # raw-stat distributions
    for name, s in [("synthetic", sim), ("real", real)]:
        f = s.reshape(len(s), -1)
        print(f"{name:>10}: mean {np.median(f.mean(1)):.4g}  std {np.median(f.std(1)):.4g}  "
              f"p99 {np.median(np.percentile(f,99,axis=1)):.4g}")

    y = np.r_[np.ones(len(sim)), np.zeros(len(real))].astype(int)

    # 1) 3-stat AUROC (brightness shortcut)
    stat_auc = cv_auroc(np.concatenate([three_stats(sim), three_stats(real)]), y)

    # 2) encoder AUROC (full deflector gap)
    sf = encoder.encode(stretch_batch(sim))
    rf = encoder.encode(stretch_batch(real))
    feats = np.concatenate([sf, rf])
    enc_auc = cv_auroc(feats, y)

    # 3) neighbour test: for synthetic stamps, fraction of k-NN that are real
    scaler = StandardScaler().fit(feats)
    fz = scaler.transform(feats)
    nn = NearestNeighbors(n_neighbors=K_NN + 1).fit(fz)
    _, idx = nn.kneighbors(fz[:len(sim)])
    is_real = (np.arange(len(feats)) >= len(sim))
    frac_real_nbr = is_real[idx[:, 1:]].mean()    # drop self

    print(f"\n3-stat AUROC (brightness shortcut) : {stat_auc:.4f}")
    print(f"encoder AUROC (full deflector gap) : {enc_auc:.4f}")
    print(f"synthetic->real k-NN fraction      : {frac_real_nbr:.3f} "
          f"(0.5 = well mixed; ~0 = synthetic clusters apart)")
    print()
    if enc_auc > 0.9:
        print("VERDICT: synthetic deflector is trivially non-real -> arc-on-real JUSTIFIED.")
    elif enc_auc > 0.7:
        print("VERDICT: real deflector gap -> arc-on-real should help.")
    else:
        print("VERDICT: synthetic deflector already looks ~real -> gap is the ARC, not the")
        print("         deflector; arc-on-real buys little. Reconsider before rebuilding.")


if __name__ == "__main__":
    from zoobot_encoder import FrozenZoobotEncoder
    run(FrozenZoobotEncoder())
