"""precheck_thetae.py — does fixing the Einstein-radius distribution make sims
sit closer to REAL lens-like objects in the frozen encoder's feature space?

Cheap go/no-go before a full regeneration. Encodes (frozen Zoobot):
  - current sims        (your existing ~1.3" clump, from sim_lenses/shards)
  - corrected sims      (the small --num batch with the new sigma_v draw)
  - ML-FP   (real lens-like)  and  P1-reserve (real ordinary)  -> the lens axis

Builds the real lens-morphology axis (ML-FP vs reserve, domain-orthogonal) and
reports where each group lands, plus k-NN lensy-fraction (how lens-like each
group looks by its nearest REAL neighbours). Corrected sims are also split by
their replayed theta_E (small <1.0" vs large) so we can see if the small arcs
specifically move toward the real-lens region.

GO  : corrected sims (esp. small-theta_E) sit HIGHER on the lens axis / have more
      lensy neighbours than current sims  -> theta_E is a transfer lever, regenerate.
NO-GO: no closer than current  -> size isn't the lever, pivot to texture/DA.

Run (Windows, local 4050):
  python precheck_thetae.py --corrected_dir ./val_corrected --data_dir data ^
      --encoder_cache ./zoobot_convnext_nano.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import cached_zoobot_encoder
from lens_dataset import LensDataset, StandardizeAsinh

from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def replay_theta_E(corrected_dir):
    """Replay the corrected sigma_v draw per file -> theta_E, using the batch's
    own generation_manifest.json (seed + good-galaxy count)."""
    man = json.load(open(Path(corrected_dir) / "generation_manifest.json"))
    n_good = len(man["good_galaxy_indices"])
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0)
    lc = LensCosmo(z_lens=0.5, z_source=2.5, cosmo=cosmo)
    files = sorted(glob.glob(os.path.join(corrected_dir, "*.npy")))
    tE = {}
    for f in files:
        m = re.search(r"seed(\d+)_(\d+)\.npy$", os.path.basename(f))
        if not m:
            continue
        seed, i = int(m.group(1)), int(m.group(2))
        rng = np.random.default_rng([seed, i])
        _ = rng.integers(0, n_good)                          # gal index draw (consume)
        sigma_v = float(np.sqrt(rng.uniform(160.0**2, 320.0**2)))  # NEW corrected draw
        tE[f] = lc.sis_sigma_v2theta_E(sigma_v)
    return tE, files


@torch.no_grad()
def encode_npy(encoder, files, norm, device, bs=256):
    out = []
    buf = []
    for f in files:
        img = np.load(f)
        if img.ndim == 3:
            img = img[..., 0] if img.shape[-1] in (1, 3) else img[0]
        buf.append(norm(img))                                # (1,64,64)
        if len(buf) == bs:
            out.append(encoder(torch.from_numpy(np.stack(buf)).to(device)).cpu().numpy()); buf = []
    if buf:
        out.append(encoder(torch.from_numpy(np.stack(buf)).to(device)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_role(encoder, role, data_dir, device, bs=256, cap=None):
    ds = LensDataset(role, data_dir, augment=nn.Identity(), verbose=False)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    F, Y = [], []
    for x, y in ld:
        F.append(encoder(x.to(device)).cpu().numpy()); Y.append(y.numpy())
    F, Y = np.concatenate(F), np.concatenate(Y)
    if cap and len(F) > cap:
        idx = np.random.default_rng(0).choice(len(F), cap, replace=False)
        F, Y = F[idx], Y[idx]
    return F, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--encoder_cache", default="./zoobot_convnext_nano.pt")
    ap.add_argument("--n_current", type=int, default=1500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = cached_zoobot_encoder(args.encoder_cache, freeze=True).to(device).eval()
    norm = StandardizeAsinh()

    # --- real anchors: ML-FP (lensy) vs reserve (ordinary) ---
    Fm, Ym = encode_role(enc, "morphology_probe", args.data_dir, device)
    mlfp, reserve = Fm[Ym == 1], Fm[Ym == 0]

    # --- current sims (sample from sim_lenses shards) ---
    sim_shards = sorted(glob.glob(os.path.join(args.data_dir, "sim_lenses", "shards", "sim_train_*.npz")))
    cur = []
    rng = np.random.default_rng(0)
    for sp in sim_shards:
        with np.load(sp) as d:
            arr = d["images"]
            take = min(len(arr), max(1, args.n_current // max(1, len(sim_shards))))
            for j in rng.choice(len(arr), take, replace=False):
                cur.append(norm(arr[j]))
    with torch.no_grad():
        cur_feat = enc(torch.from_numpy(np.stack(cur)).to(device)).cpu().numpy() if len(cur) <= 256 else \
            np.concatenate([enc(torch.from_numpy(np.stack(cur[i:i+256])).to(device)).cpu().numpy()
                            for i in range(0, len(cur), 256)])

    # --- corrected sims + their theta_E ---
    tE_map, cfiles = replay_theta_E(args.corrected_dir)
    cor_feat = encode_npy(enc, cfiles, norm, device)
    tE = np.array([tE_map[f] for f in cfiles])
    cor_small = cor_feat[tE < 1.0]
    cor_large = cor_feat[tE >= 1.0]
    print(f"corrected batch: {len(cfiles)} imgs, theta_E median {np.median(tE):.2f}\", "
          f"{(tE<1.0).mean():.0%} below 1.0\"")

    # --- domain-orthogonal lens-morphology axis (ML-FP vs reserve) ---
    scaler = StandardScaler().fit(np.concatenate([Fm, cur_feat, cor_feat]))
    Z = lambda X: scaler.transform(X)
    wm = LogisticRegression(max_iter=2000).fit(Z(Fm), Ym).coef_.ravel()
    Xd = np.concatenate([cur_feat, reserve]); yd = np.r_[np.ones(len(cur_feat)), np.zeros(len(reserve))]
    wd = LogisticRegression(max_iter=2000).fit(Z(Xd), yd).coef_.ravel()   # sim-vs-real axis
    wmu, wdu = wm/np.linalg.norm(wm), wd/np.linalg.norm(wd)
    wperp = wmu - (wmu @ wdu) * wdu; wperp /= np.linalg.norm(wperp)

    print("\nposition on DOMAIN-ORTHOGONAL lens axis (higher = more lens-like):")
    for name, X in [("reserve (ord. real)", reserve), ("ML-FP (lensy real)", mlfp),
                    ("CURRENT sims (~1.3\")", cur_feat),
                    ("CORRECTED sims small <1.0\"", cor_small),
                    ("CORRECTED sims large >=1.0\"", cor_large)]:
        if len(X):
            print(f"  {name:30s} mean={ (Z(X) @ wperp).mean():+.3f}  (n={len(X)})")

    # --- k-NN lensy fraction against the real anchor pool ---
    ref = np.concatenate([mlfp, reserve]); ref_lensy = np.r_[np.ones(len(mlfp)), np.zeros(len(reserve))]
    base = ref_lensy.mean()
    knn = NearestNeighbors(n_neighbors=20, metric="cosine").fit(Z(ref))
    print(f"\nk-NN lensy-fraction among 20 nearest REAL neighbours (chance={base:.0%}):")
    for name, X in [("CURRENT sims (~1.3\")", cur_feat),
                    ("CORRECTED sims small <1.0\"", cor_small),
                    ("CORRECTED sims large >=1.0\"", cor_large)]:
        if len(X):
            _, idx = knn.kneighbors(Z(X))
            print(f"  {name:30s} {ref_lensy[idx].mean(axis=1).mean():.1%}")

    print("\nGO if CORRECTED (esp. small) sits higher on the axis / has more lensy neighbours "
          "than CURRENT. NO-GO if no closer -> size isn't the lever, pivot to texture/DA.")


if __name__ == "__main__":
    main()
