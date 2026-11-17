"""
diagnose_coral_spectrum.py
==========================
B: uniform shrinkage or DIMENSIONAL COLLAPSE?  and  C: the pooled-floor inconsistency.

A (read separately from steps.csv) established that the per-dimension std fell 57-69%
while the mean feature norm moved only 1.4-2.9%. That is the mu/delta decomposition:
z = mu + delta, ||z|| is dominated by ||mu||, but the covariance depends ONLY on delta.
So delta can collapse while the norm monitor sees nothing.

B asks the question that decides whether information was destroyed. A covariance can
shrink two ways:
  * UNIFORMLY -- every direction scales by the same factor. The cloud shrinks but keeps its
    shape; the participation ratio is unchanged; a rescaled classifier recovers everything.
  * NON-UNIFORMLY -- some directions collapse to ~0 while others hold. The feature space
    loses rank, the participation ratio drops, and the collapsed directions are gone.

PARTICIPATION RATIO. No prior definition was found anywhere in this repo (grepped for
"participation ratio" / "rank collapse" / "SimSiam": no hits in scripts/ or the .md files),
so the standard one is used and stated here rather than silently matched to something:

    PR  = (sum_i lambda_i)^2 / sum_i lambda_i^2          [effective number of dimensions]
    sPR = PR / D                                          [standardized to (0, 1]]

PR = D for a perfectly isotropic spectrum and PR = 1 when a single eigenvalue carries
everything. sPR is scale-INVARIANT: multiplying every eigenvalue by any constant leaves it
unchanged, which is exactly the property needed here -- it separates "the cloud shrank"
(sPR flat) from "the cloud lost directions" (sPR falls).

C re-measures the pooled same-domain floor, which came back 4-6x HIGHER than the
256-sample batch floor. More samples must give a LOWER floor, so the two were not
comparable. Both candidate causes are tested directly.

Read-only: loads two existing checkpoints, forward passes under no_grad, trains nothing.
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

import run_coral_round4 as R4
from coral_round4 import Fp32CoralLoss
from diagnose_coral_floor import adapter, coral_of
from lens_dataset import LensDataset, UNLABELED
from samplers import POS_LABEL, NEG_LABEL


def spectrum(Z):
    """Eigen-spectrum of the 640x640 sample covariance of Z (rows = samples)."""
    Z = Z.double()
    C = torch.cov(Z.T)                                  # unbiased, (D, D)
    ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)  # descending, non-negative
    tot = float(ev.sum())
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    cum = torch.cumsum(ev, 0) / max(tot, 1e-300)
    return {"eigs": ev, "total_var": tot,
            "participation_ratio": pr, "sPR": pr / ev.numel(),
            "n_dims_90pct": int((cum < 0.90).sum()) + 1,
            "n_dims_99pct": int((cum < 0.99).sum()) + 1,
            "top10": [float(x) for x in ev[:10]],
            "per_dim_var": torch.diagonal(C).clone()}


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g16-ckpt", default=os.path.join(
        root, "kaggle", "coral_r4", "sweep_run", "runs", "coral_r4_sweep",
        "g16", "checkpoints", "step_2000.pt"))
    ap.add_argument("--n-source", type=int, default=2048)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default=os.path.join(root, "data"))
    ap.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--out", default=os.path.join(root, "runs",
                                                  "coral_r4_spectrum_diagnostics.json"))
    a = ap.parse_args(argv)

    sa = argparse.Namespace(
        arm="coral_nb3", fold=a.fold, gamma=16.0, data_root=a.data_root, splits=a.splits,
        cache_dir=a.cache_dir, agreement_ref=R4.DEFAULT_AGREEMENT_REF,
        init_head_from=None, max_steps=2000, allow_off_budget=True, curve_every=125,
        ckpt_every=250, backbone_lr=R4.BACKBONE_LR, fp16=False, eval_fp16=False,
        grad_checkpointing=False, coral_target_ema=False, num_workers=0, device=None,
        target_fpr=1e-3, step0_tol=0.0, smoke_batch_size=None, dir_suffix=None,
        dry_run=False, emit_config=None, reference_config=None, calibrate=False,
        calibrate_steps=0, calibrate_tail=0, grad_check=False, feat_norm_ceiling=110.0,
        feat_norm_floor=40.0)
    A = R4.build_arm("coral_nb3", sa, root, for_calibration=True)
    tr = A["trainer"]

    src_ds = LensDataset(a.splits, a.data_root, role="source_train", aug="weak", n_views=1)
    tgt_ds = LensDataset(a.splits, a.data_root, role="target", fold=a.fold,
                         aug="weak", n_views=1)
    assert np.all(tgt_ds.labels == UNLABELED)
    s_pos = np.where(src_ds.labels == POS_LABEL)[0]
    s_neg = np.where(src_ds.labels == NEG_LABEL)[0]
    t_pos = np.where(tgt_ds.latent_labels == POS_LABEL)[0]
    t_neg = np.where(tgt_ds.latent_labels == NEG_LABEL)[0]

    # FIXED index sets, drawn ONCE and reused at both checkpoints, so every ratio below is
    # a like-for-like comparison of the same images through two models.
    rng = np.random.default_rng(a.seed)
    k = a.n_source // 2
    S_IDX = np.concatenate([rng.permutation(s_pos)[:k], rng.permutation(s_neg)[:k]])
    # target: shuffle WITHIN class, then split -- this is the C fix (see below)
    tp_sh, tn_sh = rng.permutation(t_pos), rng.permutation(t_neg)
    T_IDX = np.concatenate([tp_sh, tn_sh])
    h = len(tp_sh) // 2
    T_A = np.concatenate([tp_sh[:h], tn_sh[:h]])          # stratified, shuffled halves
    T_B = np.concatenate([tp_sh[h:2 * h], tn_sh[h:2 * h]])
    # the ORDERED split the previous run used, kept so the bug can be demonstrated
    tp_o, tn_o = t_pos, t_neg
    T_A_ord = np.concatenate([tp_o[:h], tn_o[:h]])
    T_B_ord = np.concatenate([tp_o[h:2 * h], tn_o[h:2 * h]])
    S_A = np.concatenate([rng.permutation(s_pos)[:h], rng.permutation(s_neg)[:h]])
    S_B = np.concatenate([rng.permutation(s_pos)[h:2 * h], rng.permutation(s_neg)[h:2 * h]])
    print(f"  source sample {S_IDX.size}   target pool {T_IDX.size}   "
          f"matched-n halves {T_A.size}/{T_B.size}   device {tr.device}", flush=True)

    out, feats = {}, {}
    for tag, ckpt in (("lp_ft_init", None), ("g16_step2000", a.g16_ckpt)):
        if ckpt:
            tr.load(ckpt, map_location=tr.device)
        print(f"\n[{tag}] encoding ...", flush=True)
        Zs = adapter(tr, src_ds, S_IDX)
        Zt = adapter(tr, tgt_ds, T_IDX)
        feats[tag] = (Zs, Zt)
        rec = {"checkpoint": ckpt or "LP->FT init"}
        for dom, Z in (("source", Zs), ("target", Zt)):
            sp = spectrum(Z)
            rec[dom] = {kk: sp[kk] for kk in
                        ("total_var", "participation_ratio", "sPR",
                         "n_dims_90pct", "n_dims_99pct", "top10")}
            rec[dom]["n_samples"] = int(Z.shape[0])
            rec[dom]["mean_feat_l2"] = float(Z.norm(dim=1).mean())
            rec[dom]["mean_perdim_std"] = float(Z.std(dim=0, unbiased=True).mean())
            rec[dom]["_pdv"] = sp["per_dim_var"]
            print(f"  [{tag}/{dom}] sPR {sp['sPR']:.4f}  PR {sp['participation_ratio']:.1f}"
                  f"  dims@90% {sp['n_dims_90pct']}  dims@99% {sp['n_dims_99pct']}"
                  f"  totvar {sp['total_var']:.4g}  ||f|| {rec[dom]['mean_feat_l2']:.2f}")

        # ---------- C: pooled floor, ordered vs stratified-shuffled, matched n ----------
        rec["C_pooled"] = {
            "cross_n782_vs_n782": coral_of(Zs[:len(T_IDX)], Zt)[0],
            "cross_matched_n391": coral_of(adapter(tr, src_ds, S_A),
                                           adapter(tr, tgt_ds, T_A))[0],
            "target_floor_ORDERED_n391": coral_of(adapter(tr, tgt_ds, T_A_ord),
                                                  adapter(tr, tgt_ds, T_B_ord))[0],
            "target_floor_SHUFFLED_n391": coral_of(adapter(tr, tgt_ds, T_A),
                                                   adapter(tr, tgt_ds, T_B))[0],
            "source_floor_SHUFFLED_n391": coral_of(adapter(tr, src_ds, S_A),
                                                   adapter(tr, src_ds, S_B))[0],
            "n_per_side": int(T_A.size),
        }
        out[tag] = rec

    # ---------- B: per-dimension variance ratio ----------
    for dom in ("source", "target"):
        v0 = out["lp_ft_init"][dom].pop("_pdv")
        v1 = out["g16_step2000"][dom].pop("_pdv")
        r = (v1 / v0.clamp_min(1e-30)).numpy()
        e0 = np.array(out["lp_ft_init"][dom]["top10"])
        stats = {"median": float(np.median(r)), "mean": float(r.mean()),
                 "p05": float(np.percentile(r, 5)), "p25": float(np.percentile(r, 25)),
                 "p75": float(np.percentile(r, 75)), "p95": float(np.percentile(r, 95)),
                 "min": float(r.min()), "max": float(r.max()),
                 "iqr_over_median": float((np.percentile(r, 75) - np.percentile(r, 25))
                                          / np.median(r)),
                 "log10_spread_p95_over_p05": float(np.log10(np.percentile(r, 95)
                                                             / max(np.percentile(r, 5), 1e-30))),
                 "frac_below_10x_median": float((r < 0.1 * np.median(r)).mean())}
        out.setdefault("B_per_dim_var_ratio_g16_over_init", {})[dom] = stats
        s0, s1 = out["lp_ft_init"][dom]["sPR"], out["g16_step2000"][dom]["sPR"]
        out.setdefault("B_sPR", {})[dom] = {
            "init": s0, "g16": s1, "ratio": s1 / s0,
            "total_var_ratio": (out["g16_step2000"][dom]["total_var"]
                                / out["lp_ft_init"][dom]["total_var"])}

    # ---------- verdict ----------
    v = {}
    for dom in ("source", "target"):
        b = out["B_sPR"][dom]
        s = out["B_per_dim_var_ratio_g16_over_init"][dom]
        uniform = (abs(b["ratio"] - 1.0) < 0.25) and (s["iqr_over_median"] < 1.0)
        v[dom] = ("UNIFORM SHRINKAGE -- the cloud shrank but kept its shape "
                  f"(sPR {b['init']:.4f} -> {b['g16']:.4f}, x{b['ratio']:.3f}; per-dim "
                  f"variance ratio IQR/median {s['iqr_over_median']:.2f}). No information "
                  "destroyed; a rescaled classifier recovers it."
                  if uniform else
                  "DIMENSIONAL COLLAPSE -- the feature space lost rank "
                  f"(sPR {b['init']:.4f} -> {b['g16']:.4f}, x{b['ratio']:.3f}; per-dim "
                  f"variance ratio IQR/median {s['iqr_over_median']:.2f}). Collapsed "
                  "directions are unrecoverable.")
    out["B_verdict"] = v
    g = out["g16_step2000"]["C_pooled"]
    out["C_verdict"] = {
        "ordered_vs_shuffled_target_floor": (g["target_floor_ORDERED_n391"]
                                             / max(g["target_floor_SHUFFLED_n391"], 1e-30)),
        "corrected_cross_over_floor": (g["cross_matched_n391"]
                                       / max(g["target_floor_SHUFFLED_n391"], 1e-30)),
    }

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print("\n" + "=" * 92)
    for dom in ("source", "target"):
        print(f"  B [{dom}]: {out['B_verdict'][dom]}")
    print(f"\n  C: ordered target floor is "
          f"{out['C_verdict']['ordered_vs_shuffled_target_floor']:.1f}x the shuffled one; "
          f"corrected cross/floor at matched n=391 = "
          f"{out['C_verdict']['corrected_cross_over_floor']:.2f}x")
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
