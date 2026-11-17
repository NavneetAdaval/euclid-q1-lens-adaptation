"""
diagnose_coral_floor.py
=======================
Two cheap diagnostics that give the round-4 CORAL number a REFERENCE SCALE.

The sweep drove raw CORAL from 0.0500747 (step 1) to 0.0001095 (gamma=16 tail) -- a 457x
reduction. On its own that number means nothing: there is no scale for how small CORAL
*can* get. Both diagnostics supply one.

D1  SAME-DOMAIN NOISE FLOOR (a known-null, PROJECT.md sec.9)
    Compute the CORAL loss between two DISJOINT batches drawn from the SAME domain. Their
    true covariances are identical, so whatever comes back is pure 256-sample estimation
    noise -- the floor below which no cross-domain value can go. Run for both domains, at
    both ends of training (the floor scales with the feature scale, and the scale moved).

D2  IS THE SMALL VALUE AN ESTIMATOR ARTEFACT?
    With target_ema off, every step matches a 640x640 covariance estimated from 256 samples
    -- rank <= 255, badly underdetermined. An encoder could therefore match the SAMPLE
    covariances of small batches without matching the POPULATION covariances: fooling the
    estimator rather than aligning the distributions. Recomputing on the full 782-image
    target pool against a matched-size source sample, in ONE pass, distinguishes the two:
        pooled ~= batch-level  -> alignment is real at the population level
        pooled >> batch-level  -> the batch estimate was gamed; 457x is an artefact
    Also computed at the LP->FT init checkpoint, so the reduction can be quoted
    population-to-population rather than batch-to-batch.

Read-only. Loads two existing checkpoints, runs forward passes under no_grad, trains
nothing, writes runs/coral_r4_floor_diagnostics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_coral_round4 as R4
from coral_round4 import Fp32CoralLoss
from lens_dataset import LensDataset, UNLABELED
from samplers import POS_LABEL, NEG_LABEL
from train_skeleton import encode_features


@torch.no_grad()
def adapter(trainer, ds, idx, batch=128):
    """Post-LayerNorm adapter features for dataset rows ``idx`` -- the exact tensor CORAL
    consumes. eval() + no_grad: this is measurement, never training."""
    trainer.head.eval()
    trainer.encoder.eval()
    out = []
    for i in range(0, len(idx), batch):
        imgs = torch.stack([ds[int(j)]["image"] for j in idx[i:i + batch]])
        f = trainer.standardizer(
            encode_features(trainer.encoder, imgs.to(trainer.device),
                            trainer.cfg.encoder_mode))
        out.append(trainer.head.trunk(f).float().cpu())
    return torch.cat(out)


def coral_of(a, b, shrinkage=R4.CORAL_SHRINKAGE, include_mean=R4.CORAL_INCLUDE_MEAN):
    """Raw (unweighted) CORAL between two adapter blocks, target_ema OFF -- the round-4
    configuration exactly."""
    loss = Fp32CoralLoss(a.shape[1], include_mean=include_mean, shrinkage=shrinkage,
                         target_ema=False).eval()
    with torch.no_grad():
        v = float(loss(a, b))
    return v, float(loss.last_cov_term), float(loss.last_mean_term)


def balanced_draw(pos, neg, n_each, rng, taken=None):
    """n_each positives + n_each negatives, disjoint from ``taken`` if given."""
    taken = taken or set()
    p = [i for i in rng.permutation(pos) if i not in taken][:n_each]
    n = [i for i in rng.permutation(neg) if i not in taken][:n_each]
    assert len(p) == n_each and len(n) == n_each, "pool too small for a disjoint draw"
    return np.array(p + n)


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g16-ckpt", default=os.path.join(
        root, "kaggle", "coral_r4", "sweep_run", "runs", "coral_r4_sweep",
        "g16", "checkpoints", "step_2000.pt"))
    ap.add_argument("--pairs", type=int, default=60, help=">=50 required by the brief")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=os.path.join(root, "data"))
    ap.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--out", default=os.path.join(root, "runs",
                                                  "coral_r4_floor_diagnostics.json"))
    args = ap.parse_args(argv)

    half = args.batch // 2
    print("=" * 92)
    print("D1/D2 -- CORAL noise floor and pooled-vs-batch, gamma=16 final vs LP->FT init")
    print("=" * 92, flush=True)

    # ---- build the nb3 arm exactly as the sweep did (LP->FT init state) ----
    sa = argparse.Namespace(
        arm="coral_nb3", fold=args.fold, gamma=16.0,
        data_root=args.data_root, splits=args.splits, cache_dir=args.cache_dir,
        agreement_ref=R4.DEFAULT_AGREEMENT_REF, init_head_from=None,
        max_steps=2000, allow_off_budget=True, curve_every=125, ckpt_every=250,
        backbone_lr=R4.BACKBONE_LR, fp16=False, eval_fp16=False,
        grad_checkpointing=False, coral_target_ema=False, num_workers=0,
        device=args.device, target_fpr=1e-3, step0_tol=0.0, smoke_batch_size=None,
        dir_suffix=None, dry_run=False, emit_config=None, reference_config=None,
        calibrate=False, calibrate_steps=0, calibrate_tail=0, grad_check=False,
        feat_norm_ceiling=110.0)
    A = R4.build_arm("coral_nb3", sa, root, for_calibration=True)
    trainer = A["trainer"]

    # ---- datasets: same roles / same aug the sweep trained under ----
    src_ds = LensDataset(args.splits, args.data_root, role="source_train",
                         aug="weak", n_views=1)
    tgt_ds = LensDataset(args.splits, args.data_root, role="target", fold=args.fold,
                         aug="weak", n_views=1)
    assert np.all(tgt_ds.labels == UNLABELED)
    s_pos = np.where(src_ds.labels == POS_LABEL)[0]
    s_neg = np.where(src_ds.labels == NEG_LABEL)[0]
    t_pos = np.where(tgt_ds.latent_labels == POS_LABEL)[0]
    t_neg = np.where(tgt_ds.latent_labels == NEG_LABEL)[0]
    print(f"  source {len(src_ds)} ({s_pos.size}+/{s_neg.size}-)   "
          f"target {len(tgt_ds)} ({t_pos.size}+/{t_neg.size}-)   "
          f"device {trainer.device}", flush=True)

    results = {}
    for tag, ckpt in (("lp_ft_init", None), ("g16_step2000", args.g16_ckpt)):
        if ckpt:
            print(f"\n[{tag}] loading {ckpt}", flush=True)
            trainer.load(ckpt, map_location=trainer.device)
        else:
            print(f"\n[{tag}] LP->FT init: round-1 CORAL fold-0 head, published encoder",
                  flush=True)
        rng = np.random.default_rng(args.seed)

        # ---------- D1: same-domain floor + cross-domain, batch level ----------
        floor_s, floor_t, cross = [], [], []
        norms = {"src": [], "tgt": []}
        for k in range(args.pairs):
            sa1 = balanced_draw(s_pos, s_neg, half, rng)
            sa2 = balanced_draw(s_pos, s_neg, half, rng, taken=set(sa1.tolist()))
            ta1 = balanced_draw(t_pos, t_neg, half, rng)
            ta2 = balanced_draw(t_pos, t_neg, half, rng, taken=set(ta1.tolist()))
            S1, S2 = adapter(trainer, src_ds, sa1), adapter(trainer, src_ds, sa2)
            T1, T2 = adapter(trainer, tgt_ds, ta1), adapter(trainer, tgt_ds, ta2)
            floor_s.append(coral_of(S1, S2)[0])
            floor_t.append(coral_of(T1, T2)[0])
            cross.append(coral_of(S1, T1)[0])
            norms["src"].append(float(S1.norm(dim=1).mean()))
            norms["tgt"].append(float(T1.norm(dim=1).mean()))
            if (k + 1) % 20 == 0:
                print(f"    {k+1}/{args.pairs} pairs", flush=True)

        # ---------- D2: POOLED, full target pool vs matched source ----------
        n_t = min(t_pos.size, t_neg.size)
        tp = np.concatenate([t_pos, t_neg])
        sp = balanced_draw(s_pos, s_neg, n_t, rng)
        Tall = adapter(trainer, tgt_ds, tp)
        Sall = adapter(trainer, src_ds, sp)
        pooled, pooled_cov, pooled_mean = coral_of(Sall, Tall)
        # Same-domain POOLED floor. The split must be CLASS-STRATIFIED: `tp` is
        # concat([t_pos, t_neg]), so slicing it in half would compare all-lenses against
        # all-non-lenses -- two genuinely different distributions, not two samples of one.
        # (Measured while writing this: that mistake returns a "floor" of 2.3e-01, three
        # orders above the cross-domain value, which is how it was caught.) Split each
        # class in half instead, so both halves carry the same 1:1 composition.
        def halves(n_pos, n_tot):
            a = list(range(0, n_pos // 2)) + list(range(n_pos, n_pos + (n_tot - n_pos) // 2))
            b = (list(range(n_pos // 2, n_pos))
                 + list(range(n_pos + (n_tot - n_pos) // 2, n_tot)))
            return a, b
        ta_i, tb_i = halves(t_pos.size, len(tp))
        sa_i, sb_i = halves(n_t, len(sp))
        pooled_floor_t = coral_of(Tall[ta_i], Tall[tb_i])[0]
        pooled_floor_s = coral_of(Sall[sa_i], Sall[sb_i])[0]

        def st(v):
            v = np.asarray(v, float)
            return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max()),
                    "p05": float(np.percentile(v, 5)),
                    "p50": float(np.percentile(v, 50)),
                    "p95": float(np.percentile(v, 95)), "n": int(v.size)}

        r = {"checkpoint": ckpt or "LP->FT init (round-1 CORAL fold-0 head)",
             "n_pairs": args.pairs, "batch": args.batch,
             "D1_floor_source_vs_source": st(floor_s),
             "D1_floor_target_vs_target": st(floor_t),
             "D1_cross_source_vs_target": st(cross),
             "src_feat_l2_mean": float(np.mean(norms["src"])),
             "tgt_feat_l2_mean": float(np.mean(norms["tgt"])),
             "D2_pooled_cross": pooled,
             "D2_pooled_cov_term": pooled_cov, "D2_pooled_mean_term": pooled_mean,
             "D2_pooled_n_target": int(len(tp)), "D2_pooled_n_source": int(len(sp)),
             "D2_pooled_floor_target": pooled_floor_t,
             "D2_pooled_floor_source": pooled_floor_s}
        r["cross_over_floor_target"] = r["D1_cross_source_vs_target"]["mean"] / st(floor_t)["mean"]
        r["pooled_over_batch"] = pooled / r["D1_cross_source_vs_target"]["mean"]
        r["pooled_over_pooled_floor"] = pooled / max(pooled_floor_t, 1e-30)
        results[tag] = r

        print(f"  [{tag}] floor src {r['D1_floor_source_vs_source']['mean']:.3e} "
              f"+-{r['D1_floor_source_vs_source']['sd']:.1e} | "
              f"floor tgt {r['D1_floor_target_vs_target']['mean']:.3e} "
              f"+-{r['D1_floor_target_vs_target']['sd']:.1e}")
        print(f"  [{tag}] cross batch {r['D1_cross_source_vs_target']['mean']:.3e} "
              f"= {r['cross_over_floor_target']:.2f}x the target floor")
        print(f"  [{tag}] POOLED cross {pooled:.3e} ({r['pooled_over_batch']:.2f}x batch) "
              f"| pooled floor {pooled_floor_t:.3e} "
              f"({r['pooled_over_pooled_floor']:.2f}x)", flush=True)

    # ---------- verdicts ----------
    g = results["g16_step2000"]
    i = results["lp_ft_init"]
    at_floor = g["cross_over_floor_target"] < 2.0
    gamed = g["pooled_over_batch"] > 3.0
    results["verdict"] = {
        "D1": ("cross-domain CORAL is AT the same-domain estimation floor "
               f"({g['cross_over_floor_target']:.2f}x); alignment is complete to the "
               "estimator's resolution" if at_floor else
               f"cross-domain CORAL sits {g['cross_over_floor_target']:.2f}x ABOVE the "
               "same-domain floor -- a real residual remains and 'nearly complete' would "
               "overclaim"),
        "D2": ("pooled ~= batch-level -- alignment holds at the POPULATION level, not just "
               "the 256-sample estimate" if not gamed else
               f"pooled is {g['pooled_over_batch']:.1f}x the batch-level value -- the batch "
               "estimator was being gamed and the batch-level reduction must NOT be "
               "reported as achieved alignment"),
        "population_reduction": (i["D2_pooled_cross"] / g["D2_pooled_cross"]
                                 if g["D2_pooled_cross"] else float("inf")),
        "batch_reduction_reference": 0.0500747 / 0.00010948017248665564,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print("  D1:", results["verdict"]["D1"])
    print("  D2:", results["verdict"]["D2"])
    print(f"  population-to-population reduction (pooled init / pooled g16): "
          f"{results['verdict']['population_reduction']:.1f}x   "
          f"[batch-to-batch reference was "
          f"{results['verdict']['batch_reduction_reference']:.0f}x]")
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
