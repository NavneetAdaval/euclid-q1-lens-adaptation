"""
test_coral_mechanism.py
=======================
Direct test of the PROPOSED MECHANISM behind round-4 CORAL's fold-0 TPR gain.

THE HYPOTHESIS
--------------
CORAL penalises directions where the two domains' variances DISAGREE and spares directions
where they AGREE. Class structure survives because the ~1:1 balanced target pool gives the
lens-vs-not direction real variance in BOTH domains.

THE DIRECT PREDICTION ON EXISTING CHECKPOINTS
---------------------------------------------
Per-direction shrinkage from init to step 2000 should be PREDICTED BY how well the two
domains' variances agreed along that direction AT INIT. Directions where source and target
agreed should shrink LESS. If the mechanism is real, the relationship should STRENGTHEN
with gamma, matching the measured dose-response.

EVERYTHING BELOW IS DEFINED BEFORE ANY NUMBER IS COMPUTED.

BASIS. The **init SOURCE eigenbasis** -- eigenvectors of the source-domain feature
covariance at the LP->FT init checkpoint. Fixed once, reused for every checkpoint and every
arm, so "direction d" means the same thing throughout. Chosen because it is defined BEFORE
any CORAL training and WITHOUT reference to the target domain, so neither the agreement
measure nor the shrinkage measure can be biased by the basis having been fitted to them.
(The alternative -- a per-checkpoint eigenbasis -- makes directions incomparable across
checkpoints and is unusable for this question.)

AGREEMENT AT INIT, per direction d, with s_d and t_d the source and target variances:

  PRIMARY   A_d = -|log(t_d / s_d)|          (log-ratio agreement; 0 = perfect, more
                                              negative = worse)
  SECONDARY D_d = -|s_d - t_d| / (s_d + t_d) (normalised absolute difference, in [-1, 0])

Both are declared here and BOTH are reported, because they are not equivalent and choosing
after seeing which correlates better would be exactly the error this project keeps
guarding against. The log-ratio is primary because the hypothesis is about whether the two
domains AGREE along a direction, which is a scale-free, per-direction property; the raw
CORAL penalty (s_d - t_d)^2 is dominated by the few high-variance directions and would
conflate "disagrees" with "is large".

SHRINKAGE, per direction d: r_d = s_d(step 2000) / s_d(init), source domain. Correlated
against agreement as log(r_d), since shrinkage spans orders of magnitude.

THE CONFOUND, AND HOW IT IS CONTROLLED. High-variance directions might shrink differently
for reasons having nothing to do with CORAL. Two controls:
  * the gamma=0 KNOWN-NULL (no CORAL gradient at all -- reported FIRST, it is the baseline
    everything else is read against);
  * a PARTIAL Spearman of agreement vs shrinkage, controlling for log s_d at init.

FEATURES ARE EXTRACTED WITH aug="none". The cached features from the earlier spectrum run
used the stochastic weak augmentation, so mixing them with a fresh extraction would put
augmentation noise into a geometry measurement. Deterministic extraction also makes this
analysis exactly reproducible. Same sample indices for every checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ARMS = {"g0": ("runs/coral_r4_gamma0/g0", 2000),
        "g1": ("kaggle/coral_r4/sweep_run/runs/coral_r4_sweep/g1", 2000),
        "g4": ("kaggle/coral_r4/sweep_run/runs/coral_r4_sweep/g4", 2000),
        "g16": ("kaggle/coral_r4/sweep_run/runs/coral_r4_sweep/g16", 2000)}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def partial_spearman(x, y, z):
    """Spearman(x, y) with the rank-linear effect of z removed from both."""
    r = lambda v: np.argsort(np.argsort(v)).astype(float)
    rx, ry, rz = r(x), r(y), r(z)
    def resid(a):
        b = np.polyfit(rz, a, 1)
        return a - (b[0] * rz + b[1])
    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


def extract(root, cache_dir, fold, n_source, seed):
    """Deterministic (aug='none') adapter features for every checkpoint, same indices."""
    import argparse as _ap
    import run_coral_round4 as R4
    from diagnose_coral_floor import adapter
    from lens_dataset import LensDataset, UNLABELED
    from samplers import POS_LABEL, NEG_LABEL

    os.makedirs(cache_dir, exist_ok=True)
    sa = _ap.Namespace(
        arm="coral_nb3", fold=fold, gamma=16.0, data_root=os.path.join(root, "data"),
        splits=os.path.join(root, "data", "splits", "splits.json"),
        cache_dir=os.path.join(root, "data", "feature_cache"),
        agreement_ref=R4.DEFAULT_AGREEMENT_REF, init_head_from=None, max_steps=2000,
        allow_off_budget=True, curve_every=125, ckpt_every=250, backbone_lr=R4.BACKBONE_LR,
        fp16=False, eval_fp16=False, grad_checkpointing=False, coral_target_ema=False,
        num_workers=0, device=None, target_fpr=1e-3, step0_tol=0.0, smoke_batch_size=None,
        dir_suffix=None, dry_run=False, emit_config=None, reference_config=None,
        calibrate=False, calibrate_steps=0, calibrate_tail=0, grad_check=False,
        feat_norm_ceiling=110.0, feat_norm_floor=40.0)
    A = R4.build_arm("coral_nb3", sa, root, for_calibration=True)
    tr = A["trainer"]

    sd = LensDataset(sa.splits, sa.data_root, role="source_train", aug="none", n_views=1)
    td = LensDataset(sa.splits, sa.data_root, role="target", fold=fold, aug="none",
                     n_views=1)
    assert np.all(td.labels == UNLABELED)
    sp = np.where(sd.labels == POS_LABEL)[0]
    sn = np.where(sd.labels == NEG_LABEL)[0]
    tp = np.where(td.latent_labels == POS_LABEL)[0]
    tn = np.where(td.latent_labels == NEG_LABEL)[0]
    rng = np.random.default_rng(seed)
    k = n_source // 2
    S_IDX = np.concatenate([rng.permutation(sp)[:k], rng.permutation(sn)[:k]])
    T_IDX = np.concatenate([tp, tn])
    np.save(os.path.join(cache_dir, "S_y.npy"), np.array([1] * k + [0] * k))
    np.save(os.path.join(cache_dir, "T_y.npy"),
            np.array([1] * tp.size + [0] * tn.size))

    todo = [("init", None)] + [(a, os.path.join(root, d, "checkpoints", f"step_{s}.pt"))
                               for a, (d, s) in ARMS.items()]
    for tag, ck in todo:
        fs = os.path.join(cache_dir, f"Zs_{tag}.npy")
        if os.path.exists(fs):
            continue
        if ck:
            tr.load(ck, map_location=tr.device)
        np.save(fs, adapter(tr, sd, S_IDX).numpy())
        np.save(os.path.join(cache_dir, f"Zt_{tag}.npy"), adapter(tr, td, T_IDX).numpy())
        print(f"  extracted {tag}", flush=True)


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(root, "runs", "_feat_noaug"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-source", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--out", default=os.path.join(root, "runs",
                                                  "coral_r4_mechanism.json"))
    a = ap.parse_args(argv)

    if not a.skip_extract:
        print("extracting deterministic (aug=none) features ...", flush=True)
        extract(root, a.cache, a.fold, a.n_source, a.seed)

    L = lambda n: np.load(os.path.join(a.cache, n))
    Zs0, Zt0 = L("Zs_init.npy").astype(np.float64), L("Zt_init.npy").astype(np.float64)
    S_y, T_y = L("S_y.npy"), L("T_y.npy")

    # ---- the fixed basis: init SOURCE eigenbasis ----
    Cs0 = np.cov(Zs0, rowvar=False)
    ev, V = np.linalg.eigh(Cs0)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]           # V[:, d] is direction d
    s0 = ev.clip(min=1e-300)                 # source variance along d at init (= eigenvalue)
    Ct0 = np.cov(Zt0, rowvar=False)
    t0 = np.einsum("ij,jk,ki->i", V.T, Ct0, V).clip(min=1e-300)

    A_log = -np.abs(np.log(t0 / s0))                       # PRIMARY agreement
    A_abs = -np.abs(s0 - t0) / (s0 + t0)                   # SECONDARY agreement

    out = {"basis": "init SOURCE eigenbasis (fixed across all checkpoints and arms)",
           "agreement_primary": "A_d = -|log(t_d/s_d)|  (0 = perfect agreement)",
           "agreement_secondary": "D_d = -|s_d-t_d|/(s_d+t_d)",
           "shrinkage": "r_d = s_d(step2000)/s_d(init), source domain; correlated as log r_d",
           "aug": "none (deterministic); see module docstring",
           "n_dims": int(s0.size), "n_source": int(Zs0.shape[0]),
           "n_target": int(Zt0.shape[0]),
           "confound_control": ("gamma=0 known-null reported first; plus a partial Spearman "
                                "controlling for log s_d at init"),
           "arms": {}}

    print("\n" + "=" * 92)
    print("PART 1 -- does agreement-at-init predict shrinkage?   (KNOWN-NULL g0 FIRST)")
    print("=" * 92)
    print(f"{'arm':>5} {'Spearman':>10} {'Pearson':>9} {'partial rho':>12} "
          f"{'(secondary rho)':>16} {'median shrink':>14}")
    for arm in ("g0", "g1", "g4", "g16"):
        Zs = L(f"Zs_{arm}.npy").astype(np.float64)
        Cs = np.cov(Zs, rowvar=False)
        s1 = np.einsum("ij,jk,ki->i", V.T, Cs, V).clip(min=1e-300)
        lr = np.log(s1 / s0)
        rec = {"spearman_primary": spearman(A_log, lr),
               "pearson_primary": pearson(A_log, lr),
               "partial_spearman_primary_ctrl_logs0": partial_spearman(A_log, lr, np.log(s0)),
               "spearman_secondary": spearman(A_abs, lr),
               "median_shrink_factor": float(np.median(np.exp(lr))),
               "total_var_ratio": float(s1.sum() / s0.sum())}
        out["arms"][arm] = rec
        print(f"{arm:>5} {rec['spearman_primary']:>+10.4f} {rec['pearson_primary']:>+9.4f} "
              f"{rec['partial_spearman_primary_ctrl_logs0']:>+12.4f} "
              f"{rec['spearman_secondary']:>+16.4f} {rec['median_shrink_factor']:>14.5f}")

    # ---- PART 2: the class-discriminative direction ----
    print("\n" + "=" * 92)
    print("PART 2 -- the class-discriminative direction (source, REAL labels)")
    print("=" * 92)
    mu_p = Zs0[S_y == 1].mean(0)
    mu_n = Zs0[S_y == 0].mean(0)
    w = mu_p - mu_n
    w = w / np.linalg.norm(w)
    out["discriminative_direction"] = {
        "method": ("class-mean-difference direction w = (mu_pos - mu_neg)/||.||, computed on "
                   "the SOURCE domain at init with REAL source labels. Unwhitened: with 640 "
                   "dims and 2048 samples the pooled within-class covariance is "
                   "ill-conditioned, and an LDA whitening would make w depend on the very "
                   "variance structure under test.")}
    sw0 = float(w @ Cs0 @ w)
    tw0 = float(w @ Ct0 @ w)
    Aw = -abs(np.log(tw0 / sw0))
    pct = float((A_log < Aw).mean() * 100)
    out["discriminative_direction"].update({
        "src_var_init": sw0, "tgt_var_init": tw0,
        "agreement_primary": Aw,
        "agreement_percentile_among_all_directions": pct,
        "median_agreement_all_directions": float(np.median(A_log))})
    print(f"  agreement along w: {Aw:+.4f}   ->  {pct:.1f}th percentile of all 640 "
          f"directions (median {np.median(A_log):+.4f})")
    for arm in ("g0", "g1", "g4", "g16"):
        Zs = L(f"Zs_{arm}.npy").astype(np.float64)
        Cs = np.cov(Zs, rowvar=False)
        sw1 = float(w @ Cs @ w)
        s1 = np.einsum("ij,jk,ki->i", V.T, Cs, V).clip(min=1e-300)
        med = float(np.median(s1 / s0))
        rw = sw1 / sw0
        out["discriminative_direction"][arm] = {
            "shrink_factor_w": rw, "median_shrink_all_directions": med,
            "w_over_median": rw / med,
            "shrink_percentile_among_all_directions":
                float(((s1 / s0) < rw).mean() * 100)}
        print(f"  {arm:>4}: w shrank x{rw:.5f}   median direction x{med:.5f}   "
              f"ratio {rw/med:>7.2f}   -> {out['discriminative_direction'][arm]['shrink_percentile_among_all_directions']:.1f}th "
              f"percentile of shrinkage")

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
