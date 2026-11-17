"""
test_coral_r4_resolvability.py
==============================
THE GATE on the round-4 confirmatory runs: is the fold-0 effect resolvable at all?

The fold-0 outcome pass reported gamma=16 above gamma=0 at 8/8 checkpoints, ending
+0.2857 in TPR@FPR=1e-3 (a RATE; PROJECT.md sec.2/sec.9). But the WITHIN-arm scatter is
larger than the between-arm effect -- gamma=4 spans 0.1735 to 0.5000 across four
consecutive checkpoints -- so the ordering could be noise with a trend painted on it.
This script answers that with the project's own machinery, not by eye.

PHASE 1 (GPU inference, once) re-scores every checkpoint and SAVES the per-checkpoint
logits + ids. The previous pass kept only scalars, so every question below would have
required paying the ~25 min again. Saved once, reused forever.

PHASE 2 (CPU) is the analysis:

  2  PAIRED BOOTSTRAP, sec.7.12's fold-aware scheme. One shared-negative resample per draw
     applied to EVERY checkpoint of BOTH arms (they share the 26,399 pool, so a
     negative-tail fluctuation moves every threshold together), one shared-positive
     resample likewise, each checkpoint deriving its OWN FPR=1e-3 threshold. The arms share
     seed, data order and initialisation, so the comparison is PAIRED: the same resampled
     rows score both arms within a draw and the difference is taken per draw. That is
     strictly more powerful than overlapping two independent CIs.
  3  Is the ORDERING significant? A paired sign test across checkpoints and a permutation
     test shuffling arm labels within each checkpoint. Both are reported with the caveat
     that checkpoints inside one run are NOT independent draws -- they are 250 steps apart
     on one trajectory -- so these are suggestive, never confirmatory.
  4  The gamma=0 degradation, which carries much of the headline gap and therefore needs
     its own account rather than being assumed.
  5  KNOWN-NULL (PROJECT.md sec.9): the same machinery on two arms that cannot differ.

Rates only. No counts of lenses anywhere, including for effect sizes and CI half-widths.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from analyze_devmax_truncation import TailROC, validate_fast_roc     # exact, validated
from metrics import DEFAULT_TARGET_FPR

ARMS = ("g0", "g1", "g4", "g16")
STEPS = tuple(range(250, 2001, 250))


# ---------------------------------------------------------------------------
# PHASE 1 -- score + persist logits
# ---------------------------------------------------------------------------

def phase1(root, out_dir, fold, sweep, g0dir):
    import torch
    import argparse as _ap
    import evaluate as ev
    import run_coral_round4 as R4
    from live_encode import LiveEncoder

    os.makedirs(out_dir, exist_ok=True)
    sa = _ap.Namespace(
        arm="coral_nb3", fold=fold, gamma=16.0,
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
    live = LiveEncoder(sa.splits, sa.data_root, fold, batch_size=256, num_workers=0,
                       device=A["device"], verbose=False)
    dev = str(tr.device)

    def score_now():
        tr.head.eval(); tr.encoder.eval()
        p = live.encode(tr.encoder, f"test_pos_fold{fold}", fp16=False)
        n = live.encode(tr.encoder, "test_negatives_shared", fp16=False)
        return (ev.score_features(tr.head, tr.standardizer, p["features"], dev),
                ev.score_features(tr.head, tr.standardizer, n["features"], dev),
                np.asarray(p["ids"], dtype="<U80"), np.asarray(n["ids"], dtype="<U80"))

    dirs = {"g0": g0dir, **{f"g{g}": os.path.join(sweep, f"g{g}") for g in (1, 4, 16)}}
    ps, ns, pid, nid = score_now()
    np.savez_compressed(os.path.join(out_dir, "true_init.npz"),
                        pos=ps.astype(np.float64), neg=ns.astype(np.float64))
    np.savez_compressed(os.path.join(out_dir, "ids.npz"), pos_ids=pid, neg_ids=nid)
    print(f"  true_init saved  ({ps.size} pos / {ns.size} neg)", flush=True)

    for arm, d in dirs.items():
        os.makedirs(os.path.join(out_dir, arm), exist_ok=True)
        for p in sorted(glob.glob(os.path.join(d, "checkpoints", "step_*.pt")),
                        key=lambda q: int(re.search(r"step_(\d+)", q).group(1))):
            step = int(re.search(r"step_(\d+)", p).group(1))
            dst = os.path.join(out_dir, arm, f"step_{step}.npz")
            if os.path.exists(dst):
                continue
            tr.load(p, map_location=tr.device)
            a, b, pi, ni = score_now()
            assert np.array_equal(pi, pid) and np.array_equal(ni, nid), \
                f"{arm}/step_{step}: id order differs -- the pairing would be invalid"
            np.savez_compressed(dst, pos=a.astype(np.float64), neg=b.astype(np.float64))
            print(f"  {arm}/step_{step} saved", flush=True)
    return out_dir


# ---------------------------------------------------------------------------
# PHASE 2 -- CPU analysis
# ---------------------------------------------------------------------------

def load_all(logit_dir):
    keys, pos, neg = [], [], []
    z = np.load(os.path.join(logit_dir, "true_init.npz"))
    keys.append(("init", 0)); pos.append(z["pos"]); neg.append(z["neg"])
    for arm in ARMS:
        for s in STEPS:
            f = os.path.join(logit_dir, arm, f"step_{s}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            keys.append((arm, s)); pos.append(z["pos"]); neg.append(z["neg"])
    return keys, np.stack(pos), np.stack(neg)


def bootstrap_matrix(pos, neg, B, seed, paired=True):
    """T[b, k] -- TPR@1e-3 of checkpoint k under draw b.

    ONE shared negative resample and ONE shared positive resample per draw, applied to every
    checkpoint (sec.7.12). With ``paired=False`` each half of the checkpoint list gets its own
    independent resample -- used only by the known-null to show what breaking the pairing
    costs.
    """
    roc = TailROC(pos, neg)
    K, Np, Nn = roc.K, roc.Np, roc.Nn
    rng = np.random.default_rng(seed)
    T = np.empty((B, K))
    for b in range(B):
        nidx = rng.integers(0, Nn, size=Nn)
        pidx = rng.integers(0, Np, size=Np)
        nc = np.bincount(nidx, minlength=Nn).astype(np.float64)
        pc = np.bincount(pidx, minlength=Np).astype(np.float64)
        T[b] = roc.tpr(pc, nc)
    return T, roc


def ci(v, alpha=0.05):
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logit-dir", default=os.path.join(root, "runs", "coral_r4_logits"))
    ap.add_argument("--sweep", default=os.path.join(
        root, "kaggle", "coral_r4", "sweep_run", "runs", "coral_r4_sweep"))
    ap.add_argument("--g0", default=os.path.join(root, "runs", "coral_r4_gamma0", "g0"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--out", default=os.path.join(root, "runs",
                                                  "coral_r4_resolvability.json"))
    a = ap.parse_args(argv)

    if not a.skip_score:
        print("PHASE 1 -- scoring and saving per-checkpoint logits", flush=True)
        phase1(root, a.logit_dir, a.fold, a.sweep, a.g0)

    print("\nPHASE 2 -- CPU analysis", flush=True)
    keys, pos, neg = load_all(a.logit_dir)
    idx = {k: i for i, k in enumerate(keys)}
    print(f"  {len(keys)} checkpoints, {pos.shape[1]} positives, {neg.shape[1]} negatives")

    T, roc = bootstrap_matrix(pos, neg, a.bootstrap, a.seed)
    worst = validate_fast_roc(roc, pos, neg, n_checks=100)
    print(f"  fast-ROC vs metrics.tpr_at_fpr: max|diff| = {worst:.3e}")
    if worst != 0.0:
        raise SystemExit("fast ROC does not reproduce metrics.tpr_at_fpr exactly")
    point = T.mean(axis=0) * 0 + np.array(
        [float(TailROC(pos[k:k + 1], neg[k:k + 1]).tpr(np.ones(pos.shape[1]),
                                                       np.ones(neg.shape[1]))[0])
         for k in range(len(keys))])

    out = {"metric": "TPR@FPR=1e-3 as a RATE (PROJECT.md sec.2/sec.9)",
           "n_pos": int(pos.shape[1]), "n_neg": int(neg.shape[1]),
           "B": a.bootstrap, "seed": a.seed,
           "scheme": ("sec.7.12 fold-aware: one shared-negative and one shared-positive "
                      "resample per draw applied to EVERY checkpoint of BOTH arms; each "
                      "checkpoint derives its own FPR=1e-3 threshold; PAIRED differences"),
           "point_tpr": {f"{k[0]}@{k[1]}": float(point[i]) for k, i in idx.items()},
           "per_checkpoint": {}, "arm_level": {}, "ordering": {},
           "gamma0": {}, "known_null": {}}

    # ---- 2: paired bootstrap, each arm vs g0 ----
    print("\n2 -- PAIRED BOOTSTRAP vs gamma=0")
    for arm in ("g1", "g4", "g16"):
        rows = []
        print(f"  {arm} - g0")
        print(f"    {'step':>5} {'g0':>8} {arm:>8} {'delta':>9} {'95% CI':>24} {'excl 0':>7}")
        for s in STEPS:
            if (arm, s) not in idx or ("g0", s) not in idx:
                continue
            d = T[:, idx[(arm, s)]] - T[:, idx[("g0", s)]]
            lo, hi = ci(d)
            pt = point[idx[(arm, s)]] - point[idx[("g0", s)]]
            excl = bool(lo > 0 or hi < 0)
            rows.append({"step": s, "g0": float(point[idx[("g0", s)]]),
                         "arm": float(point[idx[(arm, s)]]), "delta": float(pt),
                         "ci": [lo, hi], "excludes_zero": excl,
                         "frac_draws_le_0": float((d <= 0).mean())})
            print(f"    {s:>5} {rows[-1]['g0']:>8.4f} {rows[-1]['arm']:>8.4f} "
                  f"{pt:>+9.4f} [{lo:>+8.4f}, {hi:>+8.4f}] {str(excl):>7}")
        out["per_checkpoint"][arm] = rows

        # arm-level: mean over the 8 checkpoints, and the R1 (final) checkpoint alone
        ai = [idx[(arm, s)] for s in STEPS if (arm, s) in idx]
        gi = [idx[("g0", s)] for s in STEPS if ("g0", s) in idx]
        dmean = T[:, ai].mean(axis=1) - T[:, gi].mean(axis=1)
        lo, hi = ci(dmean)
        pt = float(point[ai].mean() - point[gi].mean())
        dR1 = T[:, idx[(arm, 2000)]] - T[:, idx[("g0", 2000)]]
        lo1, hi1 = ci(dR1)
        out["arm_level"][arm] = {
            "mean_over_checkpoints": {"delta": pt, "ci": [lo, hi],
                                      "excludes_zero": bool(lo > 0 or hi < 0)},
            "R1_final_step2000": {
                "delta": float(point[idx[(arm, 2000)]] - point[idx[("g0", 2000)]]),
                "ci": [lo1, hi1], "excludes_zero": bool(lo1 > 0 or hi1 < 0)}}
        print(f"    ARM MEAN  delta {pt:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"excl0 {bool(lo>0 or hi<0)}")
        print(f"    R1 (2000) delta {out['arm_level'][arm]['R1_final_step2000']['delta']:+.4f}"
              f"  CI [{lo1:+.4f}, {hi1:+.4f}]  excl0 {bool(lo1>0 or hi1<0)}")

    # ---- 3: ordering ----
    print("\n3 -- ORDERING (suggestive only; checkpoints in one run are NOT independent)")
    from math import comb
    for arm in ("g1", "g4", "g16"):
        d = np.array([point[idx[(arm, s)]] - point[idx[("g0", s)]]
                      for s in STEPS if (arm, s) in idx])
        n = d.size
        wins = int((d > 0).sum())
        p_sign = 2 * sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n
        p_sign = min(1.0, p_sign)
        rng = np.random.default_rng(7)
        obs = float(d.mean())
        perm = np.array([float((d * rng.choice([-1, 1], size=n)).mean())
                         for _ in range(20000)])
        p_perm = float((np.abs(perm) >= abs(obs)).mean())
        out["ordering"][arm] = {"wins": wins, "n": n, "mean_delta": obs,
                                "sign_test_p_two_sided": p_sign,
                                "permutation_p_two_sided": p_perm,
                                "min_attainable_p": 2.0 / 2 ** n}
        print(f"  {arm}: {wins}/{n} checkpoints above g0   mean delta {obs:+.4f}   "
              f"sign p={p_sign:.4f}   perm p={p_perm:.4f}   (floor {2.0/2**n:.4f})")

    # ---- 4: the gamma=0 degradation ----
    print("\n4 -- gamma=0 vs the TRUE INIT")
    g0v = np.array([point[idx[("g0", s)]] for s in STEPS])
    rho = float(np.corrcoef(np.arange(g0v.size), g0v)[0, 1])
    for s in STEPS:
        d = T[:, idx[("g0", s)]] - T[:, idx[("init", 0)]]
        lo, hi = ci(d)
        out["gamma0"][f"step_{s}"] = {
            "tpr": float(point[idx[("g0", s)]]),
            "delta_vs_init": float(point[idx[("g0", s)]] - point[idx[("init", 0)]]),
            "ci": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}
    dm = T[:, [idx[("g0", s)] for s in STEPS]].mean(axis=1) - T[:, idx[("init", 0)]]
    lo, hi = ci(dm)
    out["gamma0"]["mean_vs_init"] = {
        "init_tpr": float(point[idx[("init", 0)]]),
        "g0_mean_tpr": float(g0v.mean()),
        "delta": float(g0v.mean() - point[idx[("init", 0)]]),
        "ci": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
        "monotone_rho_step_vs_tpr": rho,
        "g0_min": float(g0v.min()), "g0_max": float(g0v.max()),
        "g0_range": float(g0v.max() - g0v.min())}
    print(f"  init {point[idx[('init',0)]]:.4f}   g0 mean {g0v.mean():.4f}   "
          f"delta {g0v.mean()-point[idx[('init',0)]]:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"excl0 {bool(lo>0 or hi<0)}")
    print(f"  g0 range {g0v.min():.4f}-{g0v.max():.4f} (span {g0v.max()-g0v.min():.4f})  "
          f"rho(step,TPR) {rho:+.3f}  -> {'monotone' if abs(rho)>0.7 else 'SCATTER, not monotone'}")

    # ---- 5: known-null ----
    print("\n5 -- KNOWN-NULL")
    d0 = T[:, idx[("g0", 2000)]] - T[:, idx[("g0", 2000)]]
    lo0, hi0 = ci(d0)
    T2, _ = bootstrap_matrix(pos, neg, a.bootstrap, a.seed + 991)
    dind = T[:, idx[("g0", 2000)]] - T2[:, idx[("g0", 2000)]]
    loi, hii = ci(dind)
    out["known_null"] = {
        "paired_same_arm": {"delta": 0.0, "ci": [lo0, hi0],
                            "excludes_zero": bool(lo0 > 0 or hi0 < 0),
                            "note": "identical scores, shared resample -> exactly 0 by "
                                    "construction; a plumbing check"},
        "unpaired_same_arm_diff_seeds": {
            "delta": 0.0, "ci": [loi, hii],
            "excludes_zero": bool(loi > 0 or hii < 0),
            "half_width": float((hii - loi) / 2),
            "note": "SAME checkpoint, INDEPENDENT resamples. Must straddle zero. Its width "
                    "is the price of breaking the pairing -- compare it against the paired "
                    "CIs above to see what the paired design buys."}}
    print(f"  paired, same arm:            delta 0  CI [{lo0:+.4f}, {hi0:+.4f}]  "
          f"excl0 {bool(lo0>0 or hi0<0)}")
    print(f"  UNPAIRED, same arm, 2 seeds: CI [{loi:+.4f}, {hii:+.4f}]  half-width "
          f"{(hii-loi)/2:.4f}  excl0 {bool(loi>0 or hii<0)}")

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
