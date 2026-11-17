"""
test_coral_fisher_ci.py
=======================
Put an interval on the only two quantities that track round-4's TPR dose-response.

Test 2 reported a Fisher-ratio ordering (1.56 / 1.99 / 2.02 / 2.13 x init for gamma
0/1/4/16) and a rotation ordering (75.1 / 76.0 / 82.6 / 86.0 degrees from the init Fisher
direction). Four points, no error bar. This project has already had a Spearman of -0.90
collapse to -0.002 under a partial correlation, so an unmeasured four-point ordering is not
something to write a sentence about. This bootstraps it.

WHAT IS RESAMPLED. The SOURCE IMAGES used to estimate the class means and the pooled
within-class covariance -- a class-stratified resample with replacement, so each draw keeps
the 1:1 composition of the fixed sample. That is the estimation uncertainty that matters:
the Fisher direction and its ratio are sample statistics of ~2048 images in 640 dimensions,
and the question is whether the four arms' values are distinguishable given that.

PAIRING. All arms share seed, data order and initialisation, and here they additionally
share the SAME resampled row indices within a draw. The per-draw difference against gamma=0
therefore cancels the common-mode part -- which is exactly what made the TPR gate resolvable
where an unpaired comparison would not have been. Differences are formed per draw and the
CI is the percentile interval of those differences.

THE REFERENCE DIRECTION IS HELD FIXED. The init Fisher direction v0 is computed ONCE on the
full sample and reused for every draw and every arm, so "rotation" always means rotation
from the same reference. Bootstrapping v0 too would confound the rotation being measured
with noise in the thing it is measured from.

REGULARISATION. 640 dimensions on ~2048 samples makes the pooled within-class covariance
severely ill-conditioned, so the Fisher solve needs shrinkage. Primary lam = 0.1, the
coefficient da_losses._shrink already uses, reused rather than tuned. The whole analysis is
repeated at lam = 0.01 and lam = 0.5: if the ORDERING flips between settings, the ordering
is regularisation-dependent and must be reported as unestablished.

KNOWN-NULL. gamma=0 against ITSELF under independent resamples. It must straddle zero, and
its half-width is the RESOLUTION FLOOR -- any arm difference smaller than that is not a
difference. Reported first.

CPU only. Reads the deterministic (aug='none') features already extracted. Launches nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("g0", "g1", "g4", "g16")
ALL = ("init",) + ARMS


def batch_fisher(Zb, yb, lam):
    """Fisher direction + between/within ratio for EVERY arm at once.

    The pooled within-class covariance is formed as one batched matmul over the arm axis --
    that is the whole cost of the bootstrap, and batching it is what makes B=2000 across
    three shrinkage settings a ten-minute job instead of a many-hour one. The 5 SPD solves
    are looped rather than batched because this environment's MKL raises on batched float32
    ``linalg.solve`` (SLASWP parameter error); Cholesky per arm is exact, SPD-safe and
    negligible next to the Gram.
    """
    A, N, D = Zb.shape
    Sw = torch.zeros(A, D, D, dtype=Zb.dtype)
    n = 0
    means = {}
    for c in (0, 1):
        m = (yb == c)
        Zc = Zb[:, m, :]
        mu = Zc.mean(1, keepdim=True)
        means[c] = mu.squeeze(1)
        Zc = Zc - mu
        Sw += torch.bmm(Zc.transpose(1, 2), Zc)
        n += int(m.sum()) - 1
    Sw /= max(n, 1)
    tr = torch.diagonal(Sw, dim1=1, dim2=2).sum(1) / D
    I = torch.eye(D, dtype=Zb.dtype)
    d = means[1] - means[0]
    v = torch.empty(A, D, dtype=Zb.dtype)
    for i in range(A):
        M = (1 - lam) * Sw[i] + lam * tr[i] * I
        v[i] = torch.cholesky_solve(d[i].unsqueeze(1),
                                    torch.linalg.cholesky(M)).squeeze(1)
    v = v / v.norm(dim=1, keepdim=True)
    p = torch.bmm(Zb, v.unsqueeze(2)).squeeze(2)
    a1, b1 = p[:, yb == 1], p[:, yb == 0]
    gap = (a1.mean(1) - b1.mean(1)).abs()
    ss = (((a1 - a1.mean(1, keepdim=True)) ** 2).sum(1)
          + ((b1 - b1.mean(1, keepdim=True)) ** 2).sum(1))
    within = (ss / max(a1.shape[1] + b1.shape[1] - 2, 1)).sqrt()
    return v, (gap / within)


def angle_deg(u, v):
    return float(np.degrees(np.arccos(abs(float(np.clip(np.dot(u, v), -1, 1))))))


def ci(v, alpha=0.05):
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run(Z, yt, lam, B, seed):
    """Bootstrap per-arm Fisher ratio and rotation, PAIRED across arms within each draw."""
    v_full, r_full = batch_fisher(Z, yt, lam)
    v0 = v_full[0].numpy()                      # init direction, held FIXED across draws
    r0 = float(r_full[0])
    pos = torch.where(yt == 1)[0]
    neg = torch.where(yt == 0)[0]
    g = torch.Generator().manual_seed(seed)
    ratio = {a: np.empty(B) for a in ALL}
    rot = {a: np.empty(B) for a in ARMS}
    for b in range(B):
        idx = torch.cat([pos[torch.randint(0, pos.numel(), (pos.numel(),), generator=g)],
                         neg[torch.randint(0, neg.numel(), (neg.numel(),), generator=g)]])
        v, r = batch_fisher(Z[:, idx, :], yt[idx], lam)   # SAME idx for every arm -> paired
        for i, a in enumerate(ALL):
            ratio[a][b] = float(r[i])
            if a in ARMS:
                rot[a][b] = angle_deg(v[i].numpy(), v0)
    return v0, r0, ratio["init"], ratio, rot


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(root, "runs", "_feat_noaug"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lams", default="0.1,0.01,0.5")
    ap.add_argument("--out", default=os.path.join(root, "runs", "coral_r4_fisher_ci.json"))
    a = ap.parse_args(argv)

    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    y = np.load(os.path.join(a.cache, "S_y.npy"))
    yt = torch.from_numpy(y)
    Z = torch.from_numpy(np.stack([np.load(os.path.join(a.cache, f"Zs_{t}.npy"))
                                   for t in ALL])).double()
    lams = [float(x) for x in a.lams.split(",")]

    out = {"resampled": "source images, class-stratified, with replacement",
           "paired": "same resampled indices applied to every arm within a draw",
           "reference_direction": "init Fisher direction, computed once on the full sample "
                                  "and held fixed across draws and arms",
           "B": a.bootstrap, "seed": a.seed, "lams": lams,
           "n_source": int(Z.shape[1]), "n_dims": int(Z.shape[2]),
           "by_lam": {}}

    for lam in lams:
        print("\n" + "=" * 96)
        print(f"SHRINKAGE lam = {lam}"
              + ("   [PRIMARY -- da_losses._shrink's coefficient]" if lam == 0.1 else ""))
        print("=" * 96, flush=True)
        v0, r0, init_ratio, ratio, rot = run(Z, yt, lam, a.bootstrap, a.seed)
        rec = {"init_fisher_ratio_point": r0, "per_arm": {}, "vs_g0": {}}

        print(f"  init Fisher ratio (point) {r0:.5f}")
        print(f"\n  {'arm':>5} {'ratio med':>10} {'95% CI':>22} {'ratio/init':>11} "
              f"{'rot med':>9} {'95% CI':>20}")
        for arm in ARMS:
            rl, rh = ci(ratio[arm])
            al, ah = ci(rot[arm])
            rr = ratio[arm] / init_ratio
            rec["per_arm"][arm] = {
                "ratio_median": float(np.median(ratio[arm])), "ratio_ci": [rl, rh],
                "ratio_over_init_median": float(np.median(rr)),
                "ratio_over_init_ci": list(ci(rr)),
                "rotation_median_deg": float(np.median(rot[arm])), "rotation_ci": [al, ah]}
            print(f"  {arm:>5} {np.median(ratio[arm]):>10.4f} "
                  f"[{rl:>8.4f},{rh:>8.4f}] {np.median(rr):>11.4f} "
                  f"{np.median(rot[arm]):>9.2f} [{al:>7.2f},{ah:>7.2f}]")

        print(f"\n  PAIRED differences vs gamma=0")
        print(f"  {'arm':>5} {'d ratio':>9} {'95% CI':>22} {'excl0':>6} "
              f"{'d rot deg':>10} {'95% CI':>20} {'excl0':>6}")
        for arm in ("g1", "g4", "g16"):
            dr = ratio[arm] - ratio["g0"]
            da = rot[arm] - rot["g0"]
            rl, rh = ci(dr)
            al, ah = ci(da)
            rec["vs_g0"][arm] = {
                "d_ratio_median": float(np.median(dr)), "d_ratio_ci": [rl, rh],
                "d_ratio_excludes_zero": bool(rl > 0 or rh < 0),
                "d_rotation_median_deg": float(np.median(da)), "d_rotation_ci": [al, ah],
                "d_rotation_excludes_zero": bool(al > 0 or ah < 0)}
            print(f"  {arm:>5} {np.median(dr):>+9.4f} [{rl:>+8.4f},{rh:>+8.4f}] "
                  f"{str(bool(rl>0 or rh<0)):>6} {np.median(da):>+10.3f} "
                  f"[{al:>+8.3f},{ah:>+8.3f}] {str(bool(al>0 or ah<0)):>6}")

        # ---- KNOWN-NULL: g0 vs itself, INDEPENDENT resamples ----
        _, _, _, ratio2, rot2 = run(Z, yt, lam, a.bootstrap, a.seed + 4242)
        dn_r = ratio["g0"] - ratio2["g0"]
        dn_a = rot["g0"] - rot2["g0"]
        rl, rh = ci(dn_r)
        al, ah = ci(dn_a)
        rec["known_null_g0_vs_g0_independent"] = {
            "d_ratio_ci": [rl, rh], "d_ratio_half_width": float((rh - rl) / 2),
            "d_ratio_excludes_zero": bool(rl > 0 or rh < 0),
            "d_rotation_ci": [al, ah], "d_rotation_half_width": float((ah - al) / 2),
            "d_rotation_excludes_zero": bool(al > 0 or ah < 0)}
        print(f"\n  KNOWN-NULL g0 vs g0 (independent resamples):")
        print(f"    ratio    CI [{rl:+.4f}, {rh:+.4f}]  half-width {(rh-rl)/2:.4f}  "
              f"excl0 {bool(rl>0 or rh<0)}")
        print(f"    rotation CI [{al:+.3f}, {ah:+.3f}]  half-width {(ah-al)/2:.3f} deg  "
              f"excl0 {bool(al>0 or ah<0)}")

        rec["ordering_ratio"] = [float(np.median(ratio[x])) for x in ARMS]
        rec["ordering_rotation"] = [float(np.median(rot[x])) for x in ARMS]
        rec["ratio_monotone_in_gamma"] = bool(
            all(rec["ordering_ratio"][i] <= rec["ordering_ratio"][i + 1] for i in range(3)))
        rec["rotation_monotone_in_gamma"] = bool(
            all(rec["ordering_rotation"][i] <= rec["ordering_rotation"][i + 1]
                for i in range(3)))
        print(f"\n  monotone in gamma?  ratio {rec['ratio_monotone_in_gamma']}   "
              f"rotation {rec['rotation_monotone_in_gamma']}")
        out["by_lam"][str(lam)] = rec

    # ---- stability of the ORDERING across shrinkage settings ----
    orders = {k: (v["ratio_monotone_in_gamma"], v["rotation_monotone_in_gamma"])
              for k, v in out["by_lam"].items()}
    out["ordering_stable_across_lam"] = bool(len(set(orders.values())) == 1)
    print("\n" + "=" * 96)
    print(f"ORDERING STABLE ACROSS lam {lams}: {out['ordering_stable_across_lam']}   {orders}")
    print("=" * 96)

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
