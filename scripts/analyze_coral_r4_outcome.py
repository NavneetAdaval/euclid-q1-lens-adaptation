"""
analyze_coral_r4_outcome.py
===========================
Round-4 CORAL closing diagnostics: did the alignment objective cost DETECTION
PERFORMANCE, and if so by what mechanism?

Inference only. Loads existing checkpoints, scores them, trains nothing.

    A  TPR@FPR=1e-3 and AUROC at every checkpoint of every arm, against the
       per-dimension std at the same step.
    B  between-class / within-class variance (Fisher-style separation) at the ends,
       source domain PRIMARY (real labels), target domain SECONDARY and diagnostic-only.
    C  dose-response across gamma, all referenced to the gamma=0 control.
    D  known-null: the same B computation on two disjoint samples of ONE domain.

WHY A UNIFORM RESCALE CANNOT MOVE THE METRIC, AND WHAT THAT BUYS US
--------------------------------------------------------------------
The score is w.z + b. Scaling z by eps gives eps*(w.z) + b, a monotone transform of the
score, and TPR at a fixed FPR is a function of the RANKING only -- so is AUROC. The 56x
variance collapse is therefore INVISIBLE to both metrics by construction, and no TPR change
may be attributed to the global scale factor.

That is not a limitation, it is the design. It means any movement in A must come from the
ANISOTROPY (lambda_1 fell 24x against the bulk's ~115x) or from structural change in the
representation. And B's between/within ratio is likewise scale-invariant -- (c*B)/(c*W) = B/W
-- so A and B test the same hypothesis by two independent routes. Agreement between them is
evidence; disagreement would mean one of them is not measuring what it claims.

THE REFERENCE IS gamma=0, NOT THE LP->FT INIT
---------------------------------------------
Both the init and the trained arms differ in TWO things at once (the encoder trained, AND
the alignment objective was on), so a delta against init confounds them. gamma=0 is the same
pipeline with the objective off: identical data, identical optimiser, identical step count,
target batch still drawn, CORAL still computed for logging but multiplied by exactly zero.
Every number below is reported as a delta against it. The init is still evaluated, but only
as context.

TARGET-DOMAIN DISCLOSURE. B's target-domain arm uses ``latent_labels`` -- the true lens /
non-lens identity of the UNLABELED target pool. It is a DIAGNOSTIC computed after the fact
on finished checkpoints. It never entered any loss, any sampler decision that a training run
depended on for its objective, or any selection. The source-domain arm, which uses
legitimate source labels, is the primary result.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import evaluate as ev
import run_coral_round4 as R4
from diagnose_coral_floor import adapter
from lens_dataset import LensDataset, UNLABELED
from live_encode import LiveEncoder
from metrics import tpr_at_fpr, auroc, DEFAULT_TARGET_FPR
from samplers import POS_LABEL, NEG_LABEL


def fisher(Z, y):
    """(between-class variance, within-class variance, ratio) in feature space.

    Between = sum_c n_c ||mu_c - mu||^2 / N   (variance of the class means about the grand
    mean, weighted by class size); Within = sum_c sum_{i in c} ||z_i - mu_c||^2 / N.
    Both are total (trace) quantities over the 640 dims, so their RATIO is invariant to any
    uniform rescale of Z -- which is exactly the property that makes it able to separate
    'the cloud shrank' from 'the classes stopped being separable'.
    """
    Z = Z.double()
    mu = Z.mean(0)
    N = Z.shape[0]
    b = w = 0.0
    for c in (0, 1):
        m = (y == c)
        n = int(m.sum())
        if n == 0:
            continue
        Zc = Z[m]
        muc = Zc.mean(0)
        b += n * float(((muc - mu) ** 2).sum())
        w += float(((Zc - muc) ** 2).sum())
    b, w = b / N, w / N
    return b, w, (b / w if w > 0 else float("nan"))


@torch.no_grad()
def score_ckpt(trainer, live, fold, fp16=False):
    """TPR@1e-3 and AUROC for the CURRENT trainer state, live-encoded (cache invalid at nb3)."""
    trainer.head.eval()
    trainer.encoder.eval()
    pos = live.encode(trainer.encoder, f"test_pos_fold{fold}", fp16=fp16)
    neg = live.encode(trainer.encoder, "test_negatives_shared", fp16=fp16)
    # score on the TRAINER's device: head and standardizer live there, and
    # evaluate.score_features moves the features to whatever device it is handed --
    # passing "cpu" put the inputs and the module on different devices.
    dev = str(trainer.device)
    ps = ev.score_features(trainer.head, trainer.standardizer, pos["features"], dev)
    ns = ev.score_features(trainer.head, trainer.standardizer, neg["features"], dev)
    y = np.concatenate([np.ones(ps.size, np.int64), np.zeros(ns.size, np.int64)])
    s = np.concatenate([ps, ns])
    return float(tpr_at_fpr(y, s)), float(auroc(y, s)), int(ps.size), int(ns.size)


def std_at_step(steps_csv, step):
    if not os.path.isfile(steps_csv):
        return None, None
    with open(steps_csv) as fh:
        for r in csv.DictReader(fh):
            if int(r["step"]) == int(step):
                return float(r["src_perdim_std"]), float(r["coral_raw"])
    return None, None


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default=os.path.join(
        root, "kaggle", "coral_r4", "sweep_run", "runs", "coral_r4_sweep"))
    ap.add_argument("--g0", default=os.path.join(root, "runs", "coral_r4_gamma0", "g0"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-source", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(root, "runs", "coral_r4_outcome.json"))
    a = ap.parse_args(argv)

    sa = argparse.Namespace(
        arm="coral_nb3", fold=a.fold, gamma=16.0,
        data_root=os.path.join(root, "data"),
        splits=os.path.join(root, "data", "splits", "splits.json"),
        cache_dir=os.path.join(root, "data", "feature_cache"),
        agreement_ref=R4.DEFAULT_AGREEMENT_REF, init_head_from=None, max_steps=2000,
        allow_off_budget=True, curve_every=125, ckpt_every=250,
        backbone_lr=R4.BACKBONE_LR, fp16=False, eval_fp16=False, grad_checkpointing=False,
        coral_target_ema=False, num_workers=0, device=None, target_fpr=DEFAULT_TARGET_FPR,
        step0_tol=0.0, smoke_batch_size=None, dir_suffix=None, dry_run=False,
        emit_config=None, reference_config=None, calibrate=False, calibrate_steps=0,
        calibrate_tail=0, grad_check=False, feat_norm_ceiling=110.0, feat_norm_floor=40.0)
    A = R4.build_arm("coral_nb3", sa, root, for_calibration=True)
    tr = A["trainer"]
    live = LiveEncoder(sa.splits, sa.data_root, a.fold, batch_size=256,
                       num_workers=0, device=A["device"], verbose=False)

    src_ds = LensDataset(sa.splits, sa.data_root, role="source_train", aug="weak", n_views=1)
    tgt_ds = LensDataset(sa.splits, sa.data_root, role="target", fold=a.fold,
                         aug="weak", n_views=1)
    assert np.all(tgt_ds.labels == UNLABELED)
    sp = np.where(src_ds.labels == POS_LABEL)[0]
    sn = np.where(src_ds.labels == NEG_LABEL)[0]
    tp = np.where(tgt_ds.latent_labels == POS_LABEL)[0]
    tn = np.where(tgt_ds.latent_labels == NEG_LABEL)[0]
    rng = np.random.default_rng(a.seed)
    k = a.n_source // 2
    S_IDX = np.concatenate([rng.permutation(sp)[:k], rng.permutation(sn)[:k]])
    S_Y = torch.tensor([1] * k + [0] * k)
    T_IDX = np.concatenate([tp, tn])
    T_Y = torch.tensor([1] * tp.size + [0] * tn.size)
    # D known-null: two disjoint SOURCE halves, class-stratified
    S_A = np.concatenate([rng.permutation(sp)[:k // 2], rng.permutation(sn)[:k // 2]])
    S_B = np.concatenate([rng.permutation(sp)[k // 2:k], rng.permutation(sn)[k // 2:k]])
    S_AY = torch.tensor([1] * (k // 2) + [0] * (k // 2))

    arms = {"g0": a.g0}
    for g in (1, 4, 16):
        arms[f"g{g}"] = os.path.join(a.sweep, f"g{g}")

    out = {"reference_arm": "g0", "fold": a.fold,
           "metric": "TPR@FPR=1e-3 as a RATE (PROJECT.md sec.2/sec.9); never counts",
           "rescale_note": ("a uniform rescale of z cannot move TPR@1e-3 or AUROC -- both "
                            "are rank-based and w.z+b scales monotonically. Any movement "
                            "must come from anisotropy or structural change."),
           "target_disclosure": ("B's target arm uses latent_labels, a post-hoc DIAGNOSTIC "
                                 "on finished checkpoints. It never entered any loss or "
                                 "selection. The source arm is primary."),
           "A": {}, "B": {}, "D": {}}

    # ---------- A: TPR/AUROC curve, plus the true init ----------
    print("\n[A] true LP->FT init (before any CORAL step)", flush=True)
    t, u, npos, nneg = score_ckpt(tr, live, a.fold)
    Zs0 = adapter(tr, src_ds, S_IDX)
    Zt0 = adapter(tr, tgt_ds, T_IDX)
    out["A"]["true_init"] = {"step": 0, "tpr": t, "auroc": u,
                             "src_perdim_std": float(Zs0.std(0).mean()),
                             "n_pos": npos, "n_neg": nneg}
    print(f"    TPR {t:.4f}  AUROC {u:.4f}  std {out['A']['true_init']['src_perdim_std']:.5f}")
    b0s, w0s, r0s = fisher(Zs0, S_Y)
    b0t, w0t, r0t = fisher(Zt0, T_Y)
    out["B"]["true_init"] = {"source": {"between": b0s, "within": w0s, "ratio": r0s,
                                        "total_var": float(Zs0.var(0).sum())},
                             "target": {"between": b0t, "within": w0t, "ratio": r0t,
                                        "total_var": float(Zt0.var(0).sum())}}
    print(f"    Fisher src B/W {r0s:.5f}   tgt B/W {r0t:.5f}")

    for name, d in arms.items():
        cks = sorted(glob.glob(os.path.join(d, "checkpoints", "step_*.pt")),
                     key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
        if not cks:
            print(f"\n[A] {name}: NO checkpoints at {d} -- skipped", flush=True)
            continue
        steps_csv = os.path.join(d, "steps.csv")
        rows = []
        print(f"\n[A] {name}: {len(cks)} checkpoints", flush=True)
        for p in cks:
            step = int(re.search(r"step_(\d+)", p).group(1))
            tr.load(p, map_location=tr.device)
            t, u, _, _ = score_ckpt(tr, live, a.fold)
            sd, cr = std_at_step(steps_csv, step)
            rows.append({"step": step, "tpr": t, "auroc": u,
                         "src_perdim_std": sd, "coral_raw": cr})
            print(f"    step {step:>5}  TPR {t:.4f}  AUROC {u:.4f}  std "
                  f"{'n/a' if sd is None else f'{sd:.5f}'}", flush=True)
        out["A"][name] = rows

        # B at this arm's FINAL checkpoint
        tr.load(cks[-1], map_location=tr.device)
        Zs = adapter(tr, src_ds, S_IDX)
        Zt = adapter(tr, tgt_ds, T_IDX)
        bs, ws, rs = fisher(Zs, S_Y)
        bt, wt, rt = fisher(Zt, T_Y)
        out["B"][name] = {
            "final_step": int(re.search(r"step_(\d+)", cks[-1]).group(1)),
            "source": {"between": bs, "within": ws, "ratio": rs,
                       "total_var": float(Zs.var(0).sum())},
            "target": {"between": bt, "within": wt, "ratio": rt,
                       "total_var": float(Zt.var(0).sum())}}
        print(f"    [B] src B/W {rs:.5f}  totvar {out['B'][name]['source']['total_var']:.4g}"
              f"   |  tgt B/W {rt:.5f}", flush=True)

        # D known-null on this arm's final checkpoint (source domain, two disjoint halves)
        Za, Zb = adapter(tr, src_ds, S_A), adapter(tr, src_ds, S_B)
        ra = fisher(Za, S_AY)[2]
        rb = fisher(Zb, S_AY)[2]
        out["D"][name] = {"half_A_ratio": ra, "half_B_ratio": rb,
                          "rel_diff": abs(ra - rb) / max((ra + rb) / 2, 1e-30)}
        print(f"    [D] same-domain halves B/W {ra:.5f} vs {rb:.5f}  "
              f"rel diff {out['D'][name]['rel_diff']:.3%}", flush=True)

    # ---------- C: deltas against gamma=0 ----------
    if "g0" in out["A"] and out["A"]["g0"]:
        ref_t = out["A"]["g0"][-1]["tpr"]
        ref_u = out["A"]["g0"][-1]["auroc"]
        ref_bs = out["B"]["g0"]["source"]["ratio"]
        ref_bt = out["B"]["g0"]["target"]["ratio"]
        ref_tv = out["B"]["g0"]["source"]["total_var"]
        C = {}
        for name in ("g1", "g4", "g16"):
            if name not in out["A"] or name not in out["B"]:
                continue
            t_end = out["A"][name][-1]["tpr"]
            u_end = out["A"][name][-1]["auroc"]
            bs = out["B"][name]["source"]["ratio"]
            bt = out["B"][name]["target"]["ratio"]
            tv = out["B"][name]["source"]["total_var"]
            C[name] = {
                "tpr_end": t_end, "delta_tpr_vs_g0": t_end - ref_t,
                "auroc_end": u_end, "delta_auroc_vs_g0": u_end - ref_u,
                "sep_source": bs, "sep_source_over_g0": bs / ref_bs if ref_bs else None,
                "sep_target": bt, "sep_target_over_g0": bt / ref_bt if ref_bt else None,
                "total_var_source": tv,
                "total_var_over_g0": tv / ref_tv if ref_tv else None,
                "separation_fell_Nx": ref_bs / bs if bs else None,
                "total_var_fell_Mx": ref_tv / tv if tv else None,
            }
            if C[name]["separation_fell_Nx"] and C[name]["total_var_fell_Mx"]:
                C[name]["N_over_M"] = (C[name]["separation_fell_Nx"]
                                       / C[name]["total_var_fell_Mx"])
        out["C"] = {"reference": {"tpr": ref_t, "auroc": ref_u,
                                  "sep_source": ref_bs, "sep_target": ref_bt,
                                  "total_var_source": ref_tv},
                    "arms": C}

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
