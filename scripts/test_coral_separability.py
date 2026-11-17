"""
test_coral_separability.py
==========================
Resolve the contradiction: the mean-difference direction w lost 99.2% of its variance at
gamma=16 -- and collapses in EVERY arm including gamma=0 -- yet fold-0 TPR@FPR=1e-3 rises
monotonically with gamma (+0.2857 at gamma=16 vs gamma=0, CI [+0.1735, +0.4082]).

Those cannot both be describing discriminability. Two candidate resolutions, tested directly.

TEST 1 -- DECOMPOSE w's COLLAPSE.
  w is UNWHITENED: it says where the class centroids sit, not how separable the classes are.
  Total variance along w is (roughly) between-class + within-class. If the WITHIN term fell
  faster than the BETWEEN term, w's collapse is a DENOISING and separability along w
  improved while its variance fell. If the BETWEEN term fell faster, discriminability along
  w genuinely degraded and the signal must live somewhere else -- which is test 2.

TEST 2 -- WHERE DOES SEPARABILITY LIVE NOW?
  Recompute the discriminative direction AT EACH CHECKPOINT, whitened (Fisher/LDA), and
  measure how far it has ROTATED from init's w. Large, gamma-growing rotation => the signal
  MIGRATED and w was simply the wrong place to look. Small rotation with a rising Fisher
  ratio => test 1's denoising answer.

BASIS / DIRECTION CONVENTIONS, fixed before computing:
  * w is computed ONCE at init and HELD CONSTANT across all checkpoints for test 1. That is
    the point of test 1 -- a like-for-like decomposition of the same direction. Test 2 is
    the deliberate opposite: the direction is recomputed per checkpoint.
  * WHITENING for test 2: Fisher direction  v = (Sw + lam*tr(Sw)/D*I)^-1 (mu_pos - mu_neg),
    with Sw the pooled within-class covariance and lam = 0.1 -- the SAME Ledoit-Wolf-style
    shrinkage coefficient CoralLoss itself uses (da_losses._shrink), chosen for that
    consistency rather than tuned here. 640 dims on 2048 samples makes Sw rank-deficient
    without it (rank <= 2046 nominally, but severely ill-conditioned in practice), so some
    regularisation is mandatory; picking the project's existing constant avoids introducing
    a new free parameter.
  * Angles are reported as the acute angle between undirected axes (sign of an eigenvector /
    discriminant is arbitrary), i.e. arccos|cos|.

KNOWN-NULL FIRST. gamma=0 has no CORAL gradient, w collapses there too (x0.2765), and its
TPR is the LOWEST. It is therefore the reference for what all these quantities look like
when nothing helpful is happening. If the decomposition looks as favourable at gamma=0 as at
gamma=16, the decomposition does NOT explain the TPR difference and this is another
refutation.

CPU only. Reads the deterministic (aug='none') features already extracted. Launches nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("g0", "g1", "g4", "g16")
LAM = 0.1                       # matches da_losses._shrink's shrinkage coefficient


def pooled_within(Z, y):
    """Pooled within-class covariance."""
    D = Z.shape[1]
    Sw = np.zeros((D, D))
    n = 0
    for c in (0, 1):
        Zc = Z[y == c]
        if Zc.shape[0] < 2:
            continue
        Zc = Zc - Zc.mean(0)
        Sw += Zc.T @ Zc
        n += Zc.shape[0] - 1
    return Sw / max(n, 1)


def fisher_dir(Z, y, lam=LAM):
    """Shrinkage-regularised Fisher direction, unit norm."""
    Sw = pooled_within(Z, y)
    D = Sw.shape[0]
    Sw_r = (1 - lam) * Sw + lam * (np.trace(Sw) / D) * np.eye(D)
    d = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    v = np.linalg.solve(Sw_r, d)
    return v / np.linalg.norm(v)


def along(Z, y, u):
    """(between gap, pooled within std, ratio, total var) of the projection onto u."""
    p = Z @ u
    mp, mn = p[y == 1].mean(), p[y == 0].mean()
    gap = abs(float(mp - mn))
    vp, vn = p[y == 1], p[y == 0]
    within = float(np.sqrt(((vp - vp.mean()) ** 2).sum() + ((vn - vn.mean()) ** 2).sum())
                   / np.sqrt(max(vp.size + vn.size - 2, 1)))
    return gap, within, (gap / within if within > 0 else float("nan")), float(p.var(ddof=1))


def angle(u, v):
    c = abs(float(np.clip(u @ v, -1, 1)))
    return float(np.degrees(np.arccos(c)))


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(root, "runs", "_feat_noaug"))
    ap.add_argument("--out", default=os.path.join(root, "runs",
                                                  "coral_r4_separability.json"))
    a = ap.parse_args(argv)
    L = lambda n: np.load(os.path.join(a.cache, n)).astype(np.float64)

    Z0 = L("Zs_init.npy")
    y = np.load(os.path.join(a.cache, "S_y.npy"))
    w = Z0[y == 1].mean(0) - Z0[y == 0].mean(0)
    w = w / np.linalg.norm(w)
    v0 = fisher_dir(Z0, y)

    out = {"whitening": (f"Fisher v = (Sw + {LAM}*tr(Sw)/D*I)^-1 (mu_pos-mu_neg); lam={LAM} "
                         "is da_losses._shrink's coefficient, reused rather than tuned"),
           "w": "mean-difference direction, computed ONCE at init and held fixed (test 1)",
           "angle_convention": "acute angle between undirected axes, arccos|cos|",
           "known_null": "gamma=0 -- no CORAL gradient, lowest TPR; read everything against it",
           "init": {}, "test1": {}, "test2": {}}

    g, wi, r, tv = along(Z0, y, w)
    out["init"]["w"] = {"between_gap": g, "within_std": wi, "ratio": r, "total_var": tv}
    gv, wv, rv, tvv = along(Z0, y, v0)
    out["init"]["fisher"] = {"between_gap": gv, "within_std": wv, "ratio": rv,
                             "total_var": tvv,
                             "angle_to_w_deg": angle(v0, w)}
    tot0 = float(np.trace(np.cov(Z0, rowvar=False)))
    out["init"]["total_var_all_dims"] = tot0
    out["init"]["w_frac_of_total"] = tv / tot0
    out["init"]["fisher_frac_of_total"] = tvv / tot0

    print("=" * 96)
    print("INIT reference")
    print("=" * 96)
    print(f"  along w      : gap {g:.5f}  within {wi:.5f}  ratio {r:.5f}  "
          f"var {tv:.5f} ({100*tv/tot0:.3f}% of total)")
    print(f"  along Fisher : gap {gv:.5f}  within {wv:.5f}  ratio {rv:.5f}  "
          f"var {tvv:.5f} ({100*tvv/tot0:.3f}% of total)   angle to w "
          f"{out['init']['fisher']['angle_to_w_deg']:.1f} deg")

    print("\n" + "=" * 96)
    print("TEST 1 -- decompose w's collapse (w FIXED at init).  KNOWN-NULL g0 FIRST")
    print("=" * 96)
    print(f"{'arm':>5} {'gap':>10} {'gap/init':>9} {'within':>10} {'within/init':>12} "
          f"{'ratio':>9} {'ratio/init':>11} {'var/init':>9}")
    for arm in ARMS:
        Z = L(f"Zs_{arm}.npy")
        gg, ww, rr, vv = along(Z, y, w)
        rec = {"between_gap": gg, "within_std": ww, "ratio": rr, "total_var": vv,
               "gap_over_init": gg / g, "within_over_init": ww / wi,
               "ratio_over_init": rr / r, "var_over_init": vv / tv}
        out["test1"][arm] = rec
        print(f"{arm:>5} {gg:>10.5f} {rec['gap_over_init']:>9.4f} {ww:>10.5f} "
              f"{rec['within_over_init']:>12.4f} {rr:>9.5f} {rec['ratio_over_init']:>11.4f} "
              f"{rec['var_over_init']:>9.5f}")

    print("\n" + "=" * 96)
    print("TEST 2 -- per-checkpoint Fisher direction: rotation and where variance lives")
    print("=" * 96)
    print(f"{'arm':>5} {'angle vs w':>11} {'angle vs v0':>12} {'fisher ratio':>13} "
          f"{'ratio/init':>11} {'fisher var %':>13} {'w var %':>9}")
    for arm in ARMS:
        Z = L(f"Zs_{arm}.npy")
        v = fisher_dir(Z, y)
        gg, ww, rr, vv = along(Z, y, v)
        gw, wwx, rw, vw = along(Z, y, w)
        tot = float(np.trace(np.cov(Z, rowvar=False)))
        rec = {"angle_to_w_deg": angle(v, w), "angle_to_init_fisher_deg": angle(v, v0),
               "between_gap": gg, "within_std": ww, "ratio": rr,
               "ratio_over_init_fisher": rr / rv,
               "fisher_var_frac_of_total": vv / tot,
               "w_var_frac_of_total": vw / tot,
               "total_var_all_dims": tot}
        out["test2"][arm] = rec
        print(f"{arm:>5} {rec['angle_to_w_deg']:>11.1f} {rec['angle_to_init_fisher_deg']:>12.1f} "
              f"{rr:>13.5f} {rec['ratio_over_init_fisher']:>11.4f} "
              f"{100*rec['fisher_var_frac_of_total']:>12.4f}% {100*rec['w_var_frac_of_total']:>8.4f}%")

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
