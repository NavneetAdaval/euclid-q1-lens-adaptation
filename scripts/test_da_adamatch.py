"""
test_da_adamatch.py
===================
Tests for AdaMatch (da_losses.AdaMatchLoss + helpers) + its wiring into the image-space
training loop and the per-fold run_adamatch runner. Layers:

  PART 1 -- AdaMatch unit tests (pure tensors; no data, no encoder):
    * distribution alignment rectifies a SKEWED target prediction toward the reference
      (a hand example -- target under-predicts positives, alignment boosts them)
    * relative confidence threshold masks correctly (aligned max-prob >= tau*src-conf)
    * pseudo-label masking DROPS low-confidence target samples
    * mu schedule shape: 0 at start, mu_max at half, ramps up then holds
    * supervised BCE uses the SOURCE STRONG view (source weak does not change it)
    * confirmation-bias detector FIRES on an all-one-class (and on an acceptance~0)
      pseudo-label stream, and does NOT fire on a healthy stream

  PART 2 -- gradient routing (image-space Trainer, real frozen encoder):
    * one AdaMatch step leaves EVERY encoder param grad None + requires_grad False, while
      head params carry gradients, and the consistency term is finite. (AdaMatch cannot
      backprop into the frozen encoder.)

  PART 3 -- end-to-end per-fold AdaMatch on the REAL data (fold 0):
    * train fold 0 image-space (encoder in the loop) with a tiny schedule, evaluate on
      fold 0's held-out positives + shared negatives (cached features), and PRINT the
      first AdaMatch TPR@1e-3 next to Baseline A (0.184) and CORAL (0.201).

Convention: exit 0 on success, 1 on a failed check. PART 1 always runs; PART 2 needs the
real splits/encoder, PART 3 the splits + encoder + feature cache -- each SKIPS if absent.

Run:
    python scripts\\test_da_adamatch.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import torch

from da_losses import (AdaMatchLoss, ConfirmationBiasDetector, adamatch_mu_at_step,
                       align_target_distribution)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LENS_DATA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "data")))
SPLITS_PATH = os.path.join(DATA_ROOT, "splits", "splits.json")
CACHE_DIR = os.environ.get("LENS_CACHE_DIR", os.path.join(DATA_ROOT, "feature_cache"))
SCRATCH = os.environ.get("LENS_SCRATCH",
                         os.path.join(os.environ.get("TEMP", _HERE), "adamatch_test"))

# reference numbers to print PART 3 next to (from the memo / build request)
BASELINE_A_TPR = 0.184
CORAL_TPR = 0.201

_fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


def _f(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# PART 1 -- AdaMatch unit tests (pure tensors)
# ---------------------------------------------------------------------------

def part1_unit_tests():
    print("\n" + "=" * 72)
    print("PART 1 -- AdaMatch unit tests (pure tensors)")
    print("=" * 72)

    # --- distribution alignment: skewed target rectified toward reference (hand ex.) ---
    ref = torch.tensor([0.5, 0.5])            # balanced source reference
    tgt = torch.tensor([0.9, 0.1])            # target UNDER-predicts positives
    p = torch.tensor([[0.6, 0.4]])            # a target-weak prediction (p_pos=0.4)
    aligned = align_target_distribution(p, ref, tgt)
    # ratio = [0.5/0.9, 0.5/0.1] = [0.556, 5.0]; unnorm = [0.333, 2.0]; norm p_pos = 0.857
    check(float(aligned[0, 1]) > float(p[0, 1]) + 0.3 and abs(float(aligned.sum()) - 1.0) < 1e-5,
          f"alignment boosts under-predicted positives 0.40 -> {float(aligned[0,1]):.3f} "
          f"(reference [0.5,0.5], target dist [0.9,0.1])")
    # a symmetric reference/target leaves predictions unchanged (ratio [1,1])
    same = align_target_distribution(p, torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]))
    check(torch.allclose(same, p, atol=1e-6),
          "alignment is identity when reference == target distribution")

    # --- relative threshold masking + pseudo-label drop of low-confidence ---
    # fixed reference [0.5,0.5]; a balanced target batch so alignment ~ identity, then:
    #   3 confident-pos (logit +4), 3 confident-neg (-4), 2 uncertain (0) => 6/8 accepted.
    am = AdaMatchLoss(tau=0.9, align_reference=[0.5, 0.5]).eval()
    src_weak = torch.full((8, 1), 4.0)        # source confidence ~ sigmoid(4)=0.982
    src_strong = torch.zeros(8, 1)
    y = torch.tensor([1., 0., 1., 0., 1., 0., 1., 0.])
    tw = torch.tensor([[4.], [4.], [4.], [-4.], [-4.], [-4.], [0.], [0.]])   # 3 pos,3 neg,2 unc
    ts = torch.zeros(8, 1)
    loss, info = am(src_weak, src_strong, tw, ts, y, step=100, total_steps=100)
    thr = info["threshold"]
    check(0.85 < thr < 0.90,
          f"relative threshold = tau*src_conf ~ 0.9*0.982 = {thr:.3f}")
    check(info["n_accepted"] == 6 and abs(info["acceptance"] - 6 / 8) < 1e-6,
          f"masking accepts the 6 confident target samples, drops the 2 uncertain "
          f"(n_accepted={info['n_accepted']}, acceptance={info['acceptance']:.3f})")
    check(abs(info["accepted_pos_fraction"] - 0.5) < 1e-6,
          f"accepted pseudo-labels are balanced (pos fraction "
          f"{info['accepted_pos_fraction']:.3f})")
    # lowering tau lets the uncertain (0.5) samples clear the threshold too
    am_low = AdaMatchLoss(tau=0.3, align_reference=[0.5, 0.5]).eval()
    _, info_low = am_low(src_weak, src_strong, tw, ts, y, step=100, total_steps=100)
    check(info_low["n_accepted"] == 8,
          f"a lower tau accepts all 8 target samples (n_accepted={info_low['n_accepted']})")

    # --- mu schedule shape: 0 at start, mu_max at half, ramps up then holds ---
    MU, TOTAL = 1.0, 1000
    g = [adamatch_mu_at_step(s, TOTAL, MU) for s in range(TOTAL)]
    peak_at = int(np.argmax(g))
    check(g[0] < 1e-9, f"mu starts at 0 (mu[0]={g[0]:.2e})")
    check(abs(max(g) - MU) < 1e-9 and abs(peak_at - TOTAL // 2) <= 1,
          f"mu reaches mu_max at the half-way step (peak {max(g):.4f} at {peak_at})")
    ramp = g[:peak_at + 1]
    check(all(ramp[i] <= ramp[i + 1] + 1e-12 for i in range(len(ramp) - 1)),
          "mu ramps up (non-decreasing) over the first half")
    check(abs(g[-1] - MU) < 1e-9 and abs(g[3 * TOTAL // 4] - MU) < 1e-9,
          "mu HOLDS at mu_max through the second half")

    # --- supervised BCE uses the SOURCE STRONG view (weak view does not change it) ---
    yy = torch.tensor([1., 0.])
    ss = torch.tensor([[2.0], [-1.0]])
    ref_bce = float(torch.nn.functional.binary_cross_entropy_with_logits(ss.view(-1), yy))
    m1 = AdaMatchLoss().eval()
    _, i1 = m1(torch.tensor([[0.5], [0.5]]), ss, torch.zeros(2, 1), torch.zeros(2, 1),
               yy, step=0, total_steps=10)
    m2 = AdaMatchLoss().eval()
    _, i2 = m2(torch.tensor([[9.0], [-9.0]]), ss, torch.zeros(2, 1), torch.zeros(2, 1),
               yy, step=0, total_steps=10)   # very different SOURCE WEAK logits, same strong
    check(abs(i1["sup_loss"] - ref_bce) < 1e-5,
          f"supervised loss == BCE(source STRONG, y) ({i1['sup_loss']:.4f} vs {ref_bce:.4f})")
    check(abs(i1["sup_loss"] - i2["sup_loss"]) < 1e-5,
          "supervised loss is unchanged by the source WEAK logits (weak = reference only)")

    # --- confirmation-bias detector: healthy vs collapse streams ---
    det_ok = ConfirmationBiasDetector(patience=20)
    fired_ok = any(det_ok.update(0.5, 0.5) for _ in range(200))
    check(not fired_ok, "detector does NOT fire on a healthy stream (accept~0.5, pos~0.5)")

    det_cls = ConfirmationBiasDetector(patience=20)
    fired_cls = any(det_cls.update(0.8, 1.0) for _ in range(200))
    check(fired_cls and det_cls.reason == "class_collapse",
          f"detector FIRES on an all-one-class pseudo-label stream (reason={det_cls.reason})")

    det_acc = ConfirmationBiasDetector(patience=20)
    fired_acc = any(det_acc.update(0.0, float("nan")) for _ in range(200))
    check(fired_acc and det_acc.reason == "acceptance_collapse",
          f"detector FIRES on an acceptance~0 stream (reason={det_acc.reason})")

    det_warm = ConfirmationBiasDetector(patience=5, warmup_steps=50)
    fired_warm = [i for i, _ in enumerate(range(60)) if det_warm.update(0.9, 1.0)]
    check(bool(fired_warm) and fired_warm[0] >= 50,
          f"detector never fires during its warmup window (first fire at "
          f"{fired_warm[0] if fired_warm else None} >= 50)")


# ---------------------------------------------------------------------------
# PART 2 -- gradient routing through the image-space Trainer (real encoder)
# ---------------------------------------------------------------------------

def part2_gradient_routing(encoder=None):
    print("\n" + "=" * 72)
    print("PART 2 -- gradient routing (AdaMatch on; frozen encoder must stay grad-free)")
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

    # weak+strong views for BOTH domains (n_views=2)
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
                      use_adamatch=True, adamatch_mu_max=1.0, total_steps=10,
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
    check(abs(stats["mu_eff"] - 1.0) < 1e-6,
          f"mu_eff == mu_max at the half-way step ({stats['mu_eff']:.3f})")
    check(np.isfinite(stats["da_loss"]) and stats["da_loss"] >= 0,
          f"weighted consistency (da_loss) finite and >= 0 ({stats['da_loss']:.4f})")
    print(f"    step: loss={stats['loss']:.4f}  sup={stats['sup_loss']:.4f}  "
          f"cons={stats['consistency']:.4f}  mu_eff={stats['mu_eff']:.3f}  "
          f"accept={stats['acceptance']:.3f}")
    return encoder


# ---------------------------------------------------------------------------
# PART 3 -- end-to-end per-fold AdaMatch on the real data (fold 0 number)
# ---------------------------------------------------------------------------

def part3_end_to_end(encoder=None):
    print("\n" + "=" * 72)
    print("PART 3 -- end-to-end per-fold AdaMatch on the REAL data (fold 0)")
    print("=" * 72)
    if not os.path.exists(SPLITS_PATH):
        print(f"SKIP PART 3: real splits not found at {SPLITS_PATH}")
        return
    import evaluate as ev
    import run_adamatch as ra
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

    out_dir = os.path.join(SCRATCH, "adamatch_fold0")
    # tiny schedule + small batch/steps -> a fast but genuine AdaMatch number for fold 0
    results = ra.run_adamatch(
        SPLITS_PATH, DATA_ROOT, CACHE_DIR, out_dir, folds=[0], head_type="linear",
        epochs=(1, 3), batch_size=32, steps_per_epoch=6, mu_max=1.0, tau=0.9,
        consistency_warmup_frac=0.1, seed=42, train_device=device, eval_device="cpu",
        bootstrap=200, eval_seed=0, val_select_metric="auroc", encoder=encoder,
        verbose=True)

    sr = results["source_risk"]
    check(os.path.exists(os.path.join(out_dir, "results.json")), "results.json written")
    check("per_fold" in sr and 0 in sr["per_fold"], "source_risk has fold-0 entry")
    pf0 = sr["per_fold"][0]
    tpr0 = pf0["tpr"]
    check(tpr0 is not None and 0.0 <= tpr0 <= 1.0, f"fold-0 TPR@1e-3 in [0,1] ({tpr0})")
    check(pf0["auroc"] is not None and 0.0 <= pf0["auroc"] <= 1.0,
          f"fold-0 AUROC in [0,1] ({pf0['auroc']})")
    check(os.path.isdir(os.path.join(out_dir, "scores")), "raw per-fold scores dumped")
    check("metric_suite" in results and 0 in results["metric_suite"]["per_fold"],
          "metric suite (AUPRC/purity) computed on the dumped scores")

    suite0 = results["metric_suite"]["per_fold"][0]
    print("\n" + "=" * 72)
    print("FIRST ADAMATCH NUMBER (fold 0, tiny schedule, SOURCE-RISK/label-free)")
    print("=" * 72)
    print(f"  selected epoch    : {pf0['epoch']}   (val_auroc={_f(pf0['val_metric'])})")
    print(f"  fold-0 n_pos      : {pf0['n_pos']}")
    print(f"  fold-0 TPR@1e-3   : {_f(tpr0)}")
    print(f"  fold-0 AUROC      : {_f(pf0['auroc'])}")
    print(f"  fold-0 AUPRC      : {_f(suite0['auprc'])}")
    print(f"  fold-0 purity@50% : raw {_f(suite0['purity_raw'])}  "
          f"corr(1:2200) {_f(suite0['purity_corr'])}")
    print("-" * 72)
    print(f"  AdaMatch fold-0 TPR@1e-3 : {_f(tpr0)}")
    print(f"  Baseline A  (pooled)     : {BASELINE_A_TPR:.4f}   (source-only, to beat)")
    print(f"  Deep CORAL  (mean)       : {CORAL_TPR:.4f}   (first DA method)")
    print("  NOTE: this fold-0 tiny-schedule number is a PLUMBING smoke, not the "
          "reportable\n        5-fold AdaMatch result (run run_adamatch.py over all folds "
          "for that).")


# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("AdaMatch tests")
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
    print("ALL ADAMATCH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
