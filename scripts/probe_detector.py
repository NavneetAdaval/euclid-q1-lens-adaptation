"""probe_detector.py — is the 0.90 lens direction a real DETECTOR, or a diagnostic artifact?

No supervised sim-task head. Just fit simple linear probes on the FROZEN encoder
features and cross-test whether the real-lens-morphology direction and the sim-arc
direction are the SAME direction.

Four numbers:
  (1) REAL probe, CV   : ML-FP (real lensy) vs reserve (real ordinary), 5-fold CV.
                         This is the 0.90 we keep seeing, made honest (out-of-fold).
  (2) SIM-ARC probe, CV: sim-lens vs sim-nonlens, 5-fold CV. Texture-CONTROLLED
                         (both simulated) so it measures pure arc separability.
  (3) CROSS real->sim  : fit on ML-FP/reserve, score sim-lens vs sim-nonlens.
                         High => the REAL-morphology direction fires on ARCS
                         (texture held constant => it's the arc, not the domain).
  (4) CROSS sim->real  : fit on sim arcs, score ML-FP vs reserve.
                         High => the SIM-arc direction fires on REAL lens-like objects.

READ:
  (3) and (4) BOTH high  -> a single transferable lens direction exists in the frozen
      features; every supervised head we trained walked away from it. Plan (b) is real:
      read this direction out directly. (Then, and only then, spend the sealed 326.)
  (3)/(4) low            -> "real lens-like morphology" and "arc" are different
      directions; the 0.90 was ML-FP-specific (rings/mergers), not a lens detector,
      and our morph steering metric was measuring the wrong thing.

Run (local 4050 or Colab):
  python probe_detector.py --data_dir <data> --splits <data>/splits/splits_v5.json \
      --encoder_cache <cache>.pt
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import cached_zoobot_encoder
from lens_dataset import LensDataset

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def encode(enc, role, data_dir, splits, device, bs=256):
    ds = LensDataset(role, data_dir, splits_path=splits, augment=nn.Identity(), verbose=False)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    F, Y = [], []
    for x, y in ld:
        F.append(enc(x.to(device)).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(F), np.concatenate(Y)


def tpr_at_fpr(pos, neg, fpr=1e-3):
    k = max(1, int(np.ceil(fpr * len(neg))))
    thr = np.sort(neg)[::-1][k - 1]
    return float((pos >= thr).mean())


def cv_oof(X, y, n=5):
    """Out-of-fold probability scores (honest, no train=test leakage)."""
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def fit_dir(X, y):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), y)
    return sc, lr


def score(scaler_lr, X):
    sc, lr = scaler_lr
    return lr.predict_proba(sc.transform(X))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits", default=None, help="use splits_v5.json (val = sim-lens vs sim-nonlens)")
    ap.add_argument("--encoder_cache", default="./zoobot_convnext_nano.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = cached_zoobot_encoder(args.encoder_cache, freeze=True).to(device).eval()

    Xm, ym = encode(enc, "morphology_probe", args.data_dir, args.splits, device)  # 1 ML-FP / 0 reserve
    Xv, yv = encode(enc, "val", args.data_dir, args.splits, device)               # 1 sim-lens / 0 sim-nonlens
    print(f"REAL morphology set: ML-FP={int((ym==1).sum())} reserve={int((ym==0).sum())}")
    print(f"SIM arc set (texture-controlled): sim-lens={int((yv==1).sum())} sim-nonlens={int((yv==0).sum())}\n")

    # (1) real-vs-real CV
    oof_m = cv_oof(Xm, ym)
    auc1 = roc_auc_score(ym, oof_m); tpr1 = tpr_at_fpr(oof_m[ym == 1], oof_m[ym == 0])

    # (2) sim-arc CV (texture-controlled)
    oof_v = cv_oof(Xv, yv)
    auc2 = roc_auc_score(yv, oof_v); tpr2 = tpr_at_fpr(oof_v[yv == 1], oof_v[yv == 0])

    # (3) cross: real-morphology direction -> sim arcs
    dir_m = fit_dir(Xm, ym)
    s_v = score(dir_m, Xv)
    auc3 = roc_auc_score(yv, s_v); tpr3 = tpr_at_fpr(s_v[yv == 1], s_v[yv == 0])

    # (4) cross: sim-arc direction -> real lens-like
    dir_v = fit_dir(Xv, yv)
    s_m = score(dir_v, Xm)
    auc4 = roc_auc_score(ym, s_m); tpr4 = tpr_at_fpr(s_m[ym == 1], s_m[ym == 0])

    print("================  PROBE-AS-DETECTOR  ================")
    print(f" (1) REAL probe  ML-FP vs reserve (CV)     AUROC={auc1:.3f}  TPR@1e-3={tpr1:.3f}")
    print(f" (2) SIM-ARC     sim-lens vs sim-nonlens(CV) AUROC={auc2:.3f}  TPR@1e-3={tpr2:.3f}")
    print(f" (3) CROSS real->sim (does real dir fire on ARCS?)  AUROC={auc3:.3f}  TPR@1e-3={tpr3:.3f}")
    print(f" (4) CROSS sim->real (does sim-arc dir fire on real?) AUROC={auc4:.3f}  TPR@1e-3={tpr4:.3f}")
    print("====================================================")
    verdict = "BOTH cross-tests high -> one transferable LENS direction exists; plan (b) is real." \
        if (auc3 > 0.75 and auc4 > 0.75) else \
        ("cross-tests LOW -> real-morphology != arc; the 0.90 was not a lens detector."
         if (auc3 < 0.65 or auc4 < 0.65) else
         "MIXED -> partial overlap; arc direction is partly but not cleanly in the real-morphology probe.")
    print(f" VERDICT: {verdict}")


if __name__ == "__main__":
    main()
