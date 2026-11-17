"""
test_da_freematch.py
====================
Tests for FreeMatch (da_losses.FreeMatchLoss + SAT/SAF helpers) + its wiring into the
image-space training loop and the per-fold run_freematch runner. Layers:

  PART 1 -- FreeMatch unit tests (pure tensors; no data, no encoder):
    * SAT global/local formula pinned against a HAND example (known batch confidences ->
      known tau_t, pbar_t, and local tau_t(c)).
    * the "floor" mode clamps the lens-class local threshold up to the global tau_t
      (never lowers the minority bar), while "standard" leaves it lowered.
    * SAF penalty is ~0 (minimized) at a balanced [0.5,0.5] confident-marginal and
      strictly positive when it collapses to one class.
    * FreeMatchLoss forward (momentum=0 -> deterministic SAT): supervised BCE uses the
      SOURCE logits; SAT masking accepts the confident target samples; and switching
      standard->floor DROPS a weak lens pseudo-label (the make-or-break behaviour).
    * mu schedule shape (shared with AdaMatch): 0 at start, mu_max at half, holds.
    * ConfirmationBiasDetector FIRES on an all-one-class (and an acceptance~0) stream.

  PART 2 -- gradient routing (image-space Trainer, real frozen encoder):
    * one FreeMatch step leaves EVERY encoder param grad None + requires_grad False, while
      head params carry gradients, and consistency + SAF are finite. (FreeMatch cannot
      backprop into the frozen encoder.)

  PART 3 -- end-to-end per-fold FreeMatch on the REAL data (fold 0), BOTH modes:
    * train fold 0 image-space with a tiny schedule for local_threshold_mode in
      {"standard","floor"}, print the acc_pos_frac trajectory + TPR@1e-3 for each, next to
      Baseline A (0.184), CORAL (0.201) and AdaMatch (~0.06) -- the make-or-break screen.

Convention: exit 0 on success, 1 on a failed check. PART 1 always runs; PART 2 needs the
real splits/encoder, PART 3 the splits + encoder + feature cache -- each SKIPS if absent.

Run:
    python scripts\\test_da_freematch.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import torch

from da_losses import (FreeMatchLoss, ConfirmationBiasDetector, adamatch_mu_at_step,
                       binary_prob2, freematch_sat_update, freematch_local_thresholds,
                       self_adaptive_fairness_penalty)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LENS_DATA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "data")))
SPLITS_PATH = os.path.join(DATA_ROOT, "splits", "splits.json")
CACHE_DIR = os.environ.get("LENS_CACHE_DIR", os.path.join(DATA_ROOT, "feature_cache"))
SCRATCH = os.environ.get("LENS_SCRATCH",
                         os.path.join(os.environ.get("TEMP", _HERE), "freematch_test"))

# reference numbers to print PART 3 next to (from the memo / build request)
BASELINE_A_TPR = 0.184
CORAL_TPR = 0.201
ADAMATCH_TPR = 0.06

_fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


def _f(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# PART 1 -- FreeMatch unit tests (pure tensors)
# ---------------------------------------------------------------------------

def part1_unit_tests():
    print("\n" + "=" * 72)
    print("PART 1 -- FreeMatch unit tests (pure tensors)")
    print("=" * 72)

    # --- SAT global/local formula vs a HAND example -----------------------------------
    # momentum=0.9, tau_prev=0.5, pbar_prev=[0.5,0.5]; batch prob_w=[[0.8,0.2],[0.6,0.4]]:
    #   max_probs=[0.8,0.6] mean=0.7 -> tau = 0.9*0.5 + 0.1*0.7 = 0.52
    #   batch pbar mean=[0.7,0.3]    -> pbar = 0.9*[0.5,0.5] + 0.1*[0.7,0.3] = [0.52,0.48]
    #   local standard = (pbar/max(pbar))*tau = ([0.52,0.48]/0.52)*0.52 = [0.52, 0.48]
    prob_w = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    tau_prev = torch.tensor(0.5)
    pbar_prev = torch.tensor([0.5, 0.5])
    tau_new, pbar_new = freematch_sat_update(prob_w, tau_prev, pbar_prev, momentum=0.9)
    check(abs(float(tau_new) - 0.52) < 1e-6,
          f"SAT global tau_t = m*tau + (1-m)*mean(max_prob) = 0.52 (got {float(tau_new):.4f})")
    check(torch.allclose(pbar_new, torch.tensor([0.52, 0.48]), atol=1e-6),
          f"SAT per-class pbar_t = [0.52,0.48] (got {pbar_new.tolist()})")
    local_std = freematch_local_thresholds(tau_new, pbar_new, pos_class=1, mode="standard")
    check(torch.allclose(local_std, torch.tensor([0.52, 0.48]), atol=1e-4),
          f"local tau_t(c) standard = [0.52,0.48] (got {[round(x,4) for x in local_std.tolist()]})")

    # --- floor mode clamps the (under-learned) lens class up to the global tau_t --------
    # here pbar[pos=1]=0.48 < pbar[neg]=0.52, so the lens local threshold (0.48) is BELOW
    # tau_t (0.52) -- exactly the minority-lowering. floor must raise it back to 0.52.
    local_floor = freematch_local_thresholds(tau_new, pbar_new, pos_class=1, mode="floor")
    check(abs(float(local_floor[1]) - 0.52) < 1e-4 and abs(float(local_floor[0]) - 0.52) < 1e-4,
          f"floor clamps lens local threshold 0.48 -> tau_t=0.52 "
          f"(got neg={float(local_floor[0]):.4f}, pos={float(local_floor[1]):.4f})")
    check(float(local_std[1]) < float(local_floor[1]) - 1e-3,
          f"standard KEEPS the lens bar lowered ({float(local_std[1]):.3f}) below floor "
          f"({float(local_floor[1]):.3f}) -- the make-or-break difference")

    # --- SAF penalty: minimized at balanced, positive when collapsed -------------------
    ft = torch.tensor([0.5, 0.5])
    saf_bal = self_adaptive_fairness_penalty(torch.tensor([0.5, 0.5]), ft)
    saf_skew = self_adaptive_fairness_penalty(torch.tensor([0.7, 0.3]), ft)
    saf_coll = self_adaptive_fairness_penalty(torch.tensor([0.99, 0.01]), ft)
    check(abs(float(saf_bal)) < 1e-6,
          f"SAF penalty == 0 at a balanced [0.5,0.5] confident-marginal (got {float(saf_bal):.2e})")
    check(float(saf_skew) > 1e-3 and float(saf_coll) > float(saf_skew),
          f"SAF penalty > 0 when skewed ({float(saf_skew):.3f}) and larger when collapsed "
          f"to one class ({float(saf_coll):.3f})")
    # SAF carries gradient into the (strong-aug) prediction, not the encoder
    mp = torch.tensor([0.8, 0.2], requires_grad=True)
    self_adaptive_fairness_penalty(mp, ft).backward()
    check(mp.grad is not None and torch.isfinite(mp.grad).all(),
          "SAF penalty is differentiable w.r.t. the mean prediction (finite grad)")

    # --- FreeMatchLoss forward: sup BCE + SAT masking + standard-vs-floor drop ----------
    # momentum=0 -> SAT jumps to the batch stats deterministically. Target weak logits:
    #   [-4, -4, +1, 0] -> confidences [0.982(neg), 0.982(neg), 0.731(pos), 0.5(neg-tie)].
    # tau ~ 0.799; local standard = [neg 0.799, pos 0.370]; local floor = [0.799, 0.799].
    #   standard: accepts s0,s1 (neg) + s2 (weak lens, conf 0.731 > 0.370)  -> 3, pos_frac 1/3
    #   floor   : the weak lens conf 0.731 < 0.799 is DROPPED                -> 2, pos_frac 0
    src_logits = torch.tensor([[2.0], [-1.0]])
    y = torch.tensor([1.0, 0.0])
    tw = torch.tensor([[-4.0], [-4.0], [1.0], [0.0]])
    ts = torch.zeros(4, 1)
    ref_bce = float(torch.nn.functional.binary_cross_entropy_with_logits(
        src_logits.reshape(-1), y))

    fm_std = FreeMatchLoss(sat_ema_momentum=0.0, consistency_warmup_frac=0.0,
                           local_threshold_mode="standard").eval()
    _, i_std = fm_std(src_logits, tw, ts, y, step=50, total_steps=100)
    check(abs(i_std["sup_loss"] - ref_bce) < 1e-5,
          f"supervised loss == BCE(source logits, y) ({i_std['sup_loss']:.4f} vs {ref_bce:.4f})")
    check(abs(i_std["tau_global"] - 0.79877) < 1e-2,
          f"SAT global tau_t self-estimated from the target confidences (~0.799; "
          f"got {i_std['tau_global']:.4f})")
    check(i_std["n_accepted"] == 3 and abs(i_std["accepted_pos_fraction"] - 1 / 3) < 1e-6,
          f"standard mode accepts the weak lens (n={int(i_std['n_accepted'])}, "
          f"acc_pos_frac={i_std['accepted_pos_fraction']:.3f})")

    fm_flr = FreeMatchLoss(sat_ema_momentum=0.0, consistency_warmup_frac=0.0,
                           local_threshold_mode="floor").eval()
    _, i_flr = fm_flr(src_logits, tw, ts, y, step=50, total_steps=100)
    check(i_flr["n_accepted"] == 2 and (i_flr["accepted_pos_fraction"] == 0.0
                                        or np.isnan(i_flr["accepted_pos_fraction"]) is False),
          f"floor mode DROPS the weak lens pseudo-label (n={int(i_flr['n_accepted'])}, "
          f"acc_pos_frac={i_flr['accepted_pos_fraction']:.3f})")
    check(i_flr["local_pos"] > i_std["local_pos"] + 1e-2,
          f"floor raises the lens local threshold ({i_flr['local_pos']:.3f}) above standard "
          f"({i_std['local_pos']:.3f})")

    # --- mu schedule shape (shared helper with AdaMatch) --------------------------------
    MU, TOTAL = 1.0, 1000
    g = [adamatch_mu_at_step(s, TOTAL, MU) for s in range(TOTAL)]
    peak_at = int(np.argmax(g))
    check(g[0] < 1e-9 and abs(max(g) - MU) < 1e-9 and abs(peak_at - TOTAL // 2) <= 1,
          f"mu starts at 0, reaches mu_max at the half-way step (peak {max(g):.3f} at {peak_at})")
    check(abs(g[-1] - MU) < 1e-9 and abs(g[3 * TOTAL // 4] - MU) < 1e-9,
          "mu HOLDS at mu_max through the second half")

    # --- confirmation-bias detector: healthy vs collapse streams (reused) --------------
    det_ok = ConfirmationBiasDetector(patience=20)
    check(not any(det_ok.update(0.5, 0.5) for _ in range(200)),
          "detector does NOT fire on a healthy stream (accept~0.5, pos~0.5)")
    det_cls = ConfirmationBiasDetector(patience=20)
    fired_cls = any(det_cls.update(0.8, 1.0) for _ in range(200))
    check(fired_cls and det_cls.reason == "class_collapse",
          f"detector FIRES on an all-one-class pseudo-label stream (reason={det_cls.reason})")
    det_acc = ConfirmationBiasDetector(patience=20)
    fired_acc = any(det_acc.update(0.0, float("nan")) for _ in range(200))
    check(fired_acc and det_acc.reason == "acceptance_collapse",
          f"detector FIRES on an acceptance~0 stream (reason={det_acc.reason})")


# ---------------------------------------------------------------------------
# PART 2 -- gradient routing through the image-space Trainer (real encoder)
# ---------------------------------------------------------------------------

def part2_gradient_routing(encoder=None):
    print("\n" + "=" * 72)
    print("PART 2 -- gradient routing (FreeMatch on; frozen encoder must stay grad-free)")
    print("=" * 72)
    if not os.path.exists(SPLITS_PATH):
        print(f"SKIP PART 2: real splits not found at {SPLITS_PATH}")
        return
    from torch.utils.data import DataLoader
    from lens_dataset import LensDataset
    from zoobot_encoder import ZoobotEncoder
    from samplers import ClassBalancedBatchSampler
    from train_skeleton import Trainer, TrainConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, BATCH = 6, 6
    if encoder is None:
        print(f"  [load] frozen Zoobot encoder ({device}) ...")
        encoder = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=False)

    src_ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="source_train", aug="weak", n_views=2)
    tgt_ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="target", fold=0, aug="weak", n_views=2)
    ssamp = ClassBalancedBatchSampler(src_ds.indices_by_label[1][:K],
                                      src_ds.indices_by_label[0][:K],
                                      batch_size=BATCH, num_batches=4, seed=0)
    tsamp = ClassBalancedBatchSampler(tgt_ds.indices_by_latent[1][:K],
                                      tgt_ds.indices_by_latent[0][:K],
                                      batch_size=BATCH, num_batches=4, seed=1)
    src_loader = DataLoader(src_ds, batch_sampler=ssamp, num_workers=0)
    tgt_loader = DataLoader(tgt_ds, batch_sampler=tsamp, num_workers=0)

    cfg = TrainConfig(splits_path=SPLITS_PATH, data_root=DATA_ROOT, fold=0,
                      head_type="linear", batch_size=BATCH, lr=1e-2,
                      use_freematch=True, freematch_mu_max=1.0,
                      freematch_local_threshold_mode="floor", total_steps=10,
                      device=device, seed=0)
    trainer = Trainer(cfg, encoder=encoder)
    trainer.fit_standardizer(src_loader, max_batches=2)

    # step 5 of 10 -> past the consistency warmup, mu_eff == mu_max (consistency active)
    stats = trainer.train_step(next(iter(src_loader)), next(iter(tgt_loader)),
                               step=5, total_steps=10)

    enc_rg = sum(1 for p in trainer.encoder.parameters() if p.requires_grad)
    enc_grad = sum(1 for p in trainer.encoder.parameters() if p.grad is not None)
    head_grad = sum(1 for p in trainer.head.parameters() if p.grad is not None)
    head_total = sum(1 for _ in trainer.head.parameters())

    check(enc_rg == 0, f"no encoder param requires_grad ({enc_rg} do)")
    check(enc_grad == 0, f"no encoder param accumulated a .grad ({enc_grad} did)")
    check(head_grad == head_total and head_total > 0,
          f"all {head_total} head params carry gradients ({head_grad} do)")
    check(np.isfinite(stats["consistency"]) and stats["consistency"] >= 0,
          f"consistency term is finite and >= 0 (got {stats['consistency']:.4f})")
    check(np.isfinite(stats["saf"]) and stats["saf"] >= 0,
          f"SAF penalty is finite and >= 0 (got {stats['saf']:.4f})")
    check(abs(stats["mu_eff"] - 1.0) < 1e-6,
          f"mu_eff == mu_max at the half-way step ({stats['mu_eff']:.3f})")
    print(f"    step: loss={stats['loss']:.4f}  sup={stats['sup_loss']:.4f}  "
          f"cons={stats['consistency']:.4f}  saf={stats['saf']:.4f}  "
          f"tau={stats['tau_global']:.3f}  accept={stats['acceptance']:.3f}")
    return encoder


# ---------------------------------------------------------------------------
# PART 3 -- end-to-end per-fold FreeMatch on the real data, BOTH modes (fold 0)
# ---------------------------------------------------------------------------

def part3_end_to_end(encoder=None):
    print("\n" + "=" * 72)
    print("PART 3 -- end-to-end per-fold FreeMatch on the REAL data (fold 0, BOTH modes)")
    print("=" * 72)
    if not os.path.exists(SPLITS_PATH):
        print(f"SKIP PART 3: real splits not found at {SPLITS_PATH}")
        return
    import evaluate as ev
    import run_freematch as rf
    from zoobot_encoder import ZoobotEncoder

    cache = ev.FeatureCache(CACHE_DIR)
    need = ["source_train", "test_negatives_shared", "target_fold0", "test_pos_fold0"]
    missing = [n for n in need if not cache.has(n)]
    if missing:
        print(f"SKIP PART 3: feature cache incomplete (missing {missing}). "
              f"Run cache_features.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if encoder is None:
        print(f"  [load] frozen Zoobot encoder ({device}) ...")
        encoder = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=False)

    mode_results = {}
    for mode in ("standard", "floor"):
        print(f"\n----- local_threshold_mode = {mode} -----")
        out_dir = os.path.join(SCRATCH, f"freematch_fold0_{mode}")
        # short schedule 1,3,5 + small batch/steps -> a fast but genuine number for fold 0
        results = rf.run_freematch(
            SPLITS_PATH, DATA_ROOT, CACHE_DIR, out_dir, folds=[0], head_type="linear",
            epochs=(1, 3, 5), batch_size=64, steps_per_epoch=6, mu_max=1.0,
            saf_weight=0.01, sat_ema_momentum=0.999, consistency_warmup_frac=0.1,
            local_threshold_mode=mode, seed=42, train_device=device, eval_device="cpu",
            bootstrap=200, eval_seed=0, val_select_metric="auroc", encoder=encoder,
            verbose=True, fp16=(device == "cuda"))
        mode_results[mode] = results

        sr = results["source_risk"]
        check(os.path.exists(os.path.join(out_dir, "results.json")),
              f"[{mode}] results.json written")
        check("per_fold" in sr and 0 in sr["per_fold"], f"[{mode}] source_risk has fold-0 entry")
        pf0 = sr["per_fold"][0]
        tpr0 = pf0["tpr"]
        check(tpr0 is not None and 0.0 <= tpr0 <= 1.0, f"[{mode}] fold-0 TPR@1e-3 in [0,1] ({tpr0})")
        check("metric_suite" in results and 0 in results["metric_suite"]["per_fold"],
              f"[{mode}] metric suite (AUPRC/purity) computed on the dumped scores")

    print("\n" + "=" * 72)
    print("FIRST FREEMATCH NUMBERS (fold 0, tiny schedule, SOURCE-RISK/label-free)")
    print("      the make-or-break standard-vs-floor comparison")
    print("=" * 72)
    for mode in ("standard", "floor"):
        results = mode_results[mode]
        pf0 = results["source_risk"]["per_fold"][0]
        suite0 = results["metric_suite"]["per_fold"][0]
        st0 = results["freematch_stability"][0]
        traj = [(e["epoch"], e["mean_accepted_pos_fraction"], e["mean_acceptance"])
                for e in st0["epochs"]]
        print(f"\n  [{mode}]  selected epoch {pf0['epoch']} (val_auroc={_f(pf0['val_metric'])})"
              f"   collapse={'DETECTED('+str(st0['cb_reason'])+')' if st0['cb_detected'] else 'none'}")
        print(f"    acc_pos_frac trajectory (epoch: pos_frac / accept):")
        print("      " + "  ".join(f"e{ep}:{_f(pf,2)}/{_f(ac,2)}" for ep, pf, ac in traj))
        print(f"    fold-0 TPR@1e-3 = {_f(pf0['tpr'])}   AUROC = {_f(pf0['auroc'])}   "
              f"AUPRC = {_f(suite0['auprc'])}   purity@50%(corr) = {_f(suite0['purity_corr'])}")

    print("\n" + "-" * 72)
    print("  REFERENCE (from the memo / build request):")
    print(f"    Baseline A (pooled, source-only, to beat) : {BASELINE_A_TPR:.4f}")
    print(f"    Deep CORAL (mean, first DA method)        : {CORAL_TPR:.4f}")
    print(f"    AdaMatch   (collapsed)                    : ~{ADAMATCH_TPR:.2f}")
    for mode in ("standard", "floor"):
        tpr0 = mode_results[mode]["source_risk"]["per_fold"][0]["tpr"]
        print(f"    FreeMatch  ({mode:<8})  fold-0 TPR@1e-3    : {_f(tpr0)}")
    print("  NOTE: these fold-0 tiny-schedule numbers are a PLUMBING/behaviour smoke, not "
          "the\n        reportable 5-fold FreeMatch result (run run_freematch.py over all "
          "folds for that).")


# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("FreeMatch tests")
    print(f"  data_root : {DATA_ROOT}")
    print("=" * 72)
    part1_unit_tests()

    encoder = None
    if os.path.exists(SPLITS_PATH):
        try:
            from zoobot_encoder import ZoobotEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\n[load] frozen Zoobot encoder once ({device}) for PART 2 + 3 ...")
            encoder = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=False)
        except Exception as e:
            print(f"[warn] could not load encoder ({e}); PART 2/3 will SKIP")
            encoder = None

    part2_gradient_routing(encoder)
    part3_end_to_end(encoder)

    print("\n" + "=" * 72)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL FREEMATCH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
