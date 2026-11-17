"""
test_da_coral.py
================
Tests for Deep CORAL (da_losses.CoralLoss) + its wiring into the training loop and
the per-fold run_coral runner. Three layers:

  PART 1 -- CORAL loss unit tests (pure tensors; no data, no encoder):
    * zero when source == target features (identical moments -> exact 0)
    * strictly positive when the two domains differ
    * finite under small batches (n=8, even n=2)
    * invariant to ROW ORDER (covariance/mean are permutation-invariant)
    * SHRINKAGE keeps it STABLE for a tiny (n=8) noisy target: the loss varies LESS
      across random tiny target batches with shrinkage than without.

  PART 2 -- gradient routing (image-space Trainer, real frozen encoder):
    * one CORAL training step leaves EVERY encoder param with grad None +
      requires_grad False, while head params carry gradients, and the CORAL term is
      finite and > 0. (CORAL cannot backprop into the frozen encoder.)

  PART 3 -- end-to-end per-fold CORAL on the REAL feature cache:
    * train fold 0 with a tiny schedule, evaluate it on fold 0's held-out positives +
      shared negatives, and PRINT the first CORAL TPR@1e-3 number.

Convention (matches the other real-slice tests): exit 0 on success, 1 on a failed
check. PART 1 always runs; PART 2 needs the real splits/encoder, PART 3 the feature
cache -- each SKIPS (not fails) if its inputs are absent.

Run:
    python scripts\\test_da_coral.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import torch

from da_losses import CoralLoss, BlowupDetector, _mean_cov

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LENS_DATA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "data")))
SPLITS_PATH = os.path.join(DATA_ROOT, "splits", "splits.json")
CACHE_DIR = os.environ.get("LENS_CACHE_DIR", os.path.join(DATA_ROOT, "feature_cache"))
SCRATCH = os.environ.get("LENS_SCRATCH",
                         os.path.join(os.environ.get("TEMP", _HERE), "coral_test"))

_fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


# ---------------------------------------------------------------------------
# PART 1 -- CORAL loss unit tests (pure tensors)
# ---------------------------------------------------------------------------

def part1_loss_unit_tests():
    print("\n" + "=" * 72)
    print("PART 1 -- CORAL loss unit tests (pure tensors)")
    print("=" * 72)
    torch.manual_seed(0)
    D = 16

    # --- zero when source == target (EMA off so it is unambiguously per-batch) ---
    coral = CoralLoss(D, target_ema=False).eval()
    z = torch.randn(64, D)
    loss_eq = coral(z, z.clone())
    check(torch.isfinite(loss_eq) and abs(float(loss_eq)) < 1e-6,
          f"loss == 0 when source == target features (got {float(loss_eq):.2e})")

    # --- strictly positive when domains differ ---
    src = torch.randn(64, D)
    tgt = torch.randn(48, D) * 2.5 + 1.0            # different mean AND covariance
    loss_diff = coral(src, tgt)
    check(float(loss_diff) > 1e-6,
          f"loss > 0 when source != target (got {float(loss_diff):.4f})")

    # --- mean-term toggle: with-mean >= without-mean on a mean-shifted target ---
    coral_nomean = CoralLoss(D, include_mean=False, target_ema=False).eval()
    shifted = src + 3.0                              # pure first-moment shift
    with_mean = float(CoralLoss(D, include_mean=True, target_ema=False).eval()(src, shifted))
    no_mean = float(coral_nomean(src, shifted))
    check(with_mean > no_mean + 1e-6,
          f"mean-term ON catches a pure mean shift that mean-term OFF misses "
          f"(with={with_mean:.4f} > without={no_mean:.4f})")

    # --- finite under small batches (n=8 and the n=2 minimum) ---
    finite_small = True
    for n in (8, 4, 2):
        l = coral(torch.randn(n, D), torch.randn(n, D))
        finite_small &= bool(torch.isfinite(l))
    check(finite_small, "loss finite for tiny batches (n=8, 4, 2)")

    # --- invariant to row order ---
    perm_s = torch.randperm(src.shape[0]); perm_t = torch.randperm(tgt.shape[0])
    loss_perm = coral(src[perm_s], tgt[perm_t])
    check(torch.allclose(loss_diff, loss_perm, atol=1e-6),
          f"loss invariant to row order ({float(loss_diff):.6f} vs {float(loss_perm):.6f})")

    # --- shrinkage keeps a tiny (n=8) noisy target STABLE (lower variance) ---
    torch.manual_seed(1)
    src_big = torch.randn(512, D)                    # a stable source reference
    shrink = CoralLoss(D, shrinkage=0.1, target_ema=False).eval()
    noshr = CoralLoss(D, shrinkage=0.0, jitter=0.0, target_ema=False).eval()
    ls, ln = [], []
    for _ in range(40):
        t = torch.randn(8, D)                        # a fresh noisy tiny target
        ls.append(float(shrink(src_big, t)))
        ln.append(float(noshr(src_big, t)))
    std_shr, std_noshr = float(np.std(ls)), float(np.std(ln))
    check(all(np.isfinite(ls)) and all(np.isfinite(ln)),
          "shrinkage: all tiny-target losses finite (both settings)")
    check(std_shr < std_noshr,
          f"shrinkage stabilises the tiny (n=8) target: std {std_shr:.4f} < "
          f"no-shrink std {std_noshr:.4f}")

    # --- covariance helper matches torch.cov (unbiased sample covariance) ---
    _, cov = _mean_cov(src)
    ref = torch.cov(src.t())                          # torch.cov: rows=vars -> transpose
    check(torch.allclose(cov, ref, atol=1e-5),
          "covariance formula matches torch.cov (unbiased sample covariance)")

    # --- EMA on: still finite, and gradient flows to the input (head) ---
    coral_ema = CoralLoss(D, target_ema=True, ema_momentum=0.9).train()
    a = torch.randn(64, D, requires_grad=True)
    b = torch.randn(32, D, requires_grad=True)
    le = coral_ema(a, b)                              # first step seeds the EMA
    le2 = coral_ema(a, torch.randn(32, D, requires_grad=True))   # second uses EMA blend
    le2.backward()
    check(torch.isfinite(le) and torch.isfinite(le2)
          and a.grad is not None and torch.isfinite(a.grad).all(),
          "EMA target: finite over steps and gradient reaches the (head) input")


# ---------------------------------------------------------------------------
# PART 2 -- gradient routing through the image-space Trainer (real encoder)
# ---------------------------------------------------------------------------

def part2_gradient_routing():
    print("\n" + "=" * 72)
    print("PART 2 -- gradient routing (CORAL on; frozen encoder must stay grad-free)")
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
    K, BATCH = 8, 8
    print(f"  [load] frozen Zoobot encoder ({device}) ...")
    encoder = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=False)

    src_ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="source_train", aug="none")
    tgt_ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="target", fold=0, aug="none")
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
                      use_coral=True, coral_weight=1.0, device=device, seed=0)
    trainer = Trainer(cfg, encoder=encoder)
    trainer.fit_standardizer(src_loader, max_batches=2)

    stats = trainer.train_step(next(iter(src_loader)), next(iter(tgt_loader)))

    enc_rg = sum(1 for p in trainer.encoder.parameters() if p.requires_grad)
    enc_grad = sum(1 for p in trainer.encoder.parameters() if p.grad is not None)
    head_grad = sum(1 for p in trainer.head.parameters() if p.grad is not None)
    head_total = sum(1 for _ in trainer.head.parameters())

    check(enc_rg == 0, f"no encoder param requires_grad ({enc_rg} do)")
    check(enc_grad == 0, f"no encoder param accumulated a .grad ({enc_grad} did)")
    check(head_grad == head_total and head_total > 0,
          f"all {head_total} head params carry gradients ({head_grad} do)")
    check(np.isfinite(stats["da_loss"]) and stats["da_loss"] > 0,
          f"CORAL da_loss is finite and > 0 (got {stats['da_loss']:.4f})")
    print(f"    step: loss={stats['loss']:.4f}  sup={stats['sup_loss']:.4f}  "
          f"coral={stats['da_loss']:.4f}")


# ---------------------------------------------------------------------------
# PART 3 -- end-to-end per-fold CORAL on the real feature cache (fold 0 number)
# ---------------------------------------------------------------------------

def part3_end_to_end():
    print("\n" + "=" * 72)
    print("PART 3 -- end-to-end per-fold CORAL on the REAL feature cache")
    print("=" * 72)
    import evaluate as ev
    import run_coral as rc

    cache = ev.FeatureCache(CACHE_DIR)
    need = ["source_train", "test_negatives_shared", "target_fold0", "test_pos_fold0"]
    missing = [n for n in need if not cache.has(n)]
    if missing:
        print(f"SKIP PART 3: feature cache incomplete (missing {missing}). "
              f"Run cache_features.py first.")
        return

    out_dir = os.path.join(SCRATCH, "coral_fold0")
    # tiny schedule + few steps -> a fast but genuine CORAL number for fold 0
    results = rc.run_coral(
        CACHE_DIR, out_dir, folds=[0], head_type="linear", epochs=(1, 3, 8),
        batch_size=256, steps_per_epoch=40, coral_weight=1.0, seed=42,
        train_device="cpu", eval_device="cpu", bootstrap=300, eval_seed=0,
        val_select_metric="auroc", verbose=True)

    sr = results["source_risk"]
    check(os.path.exists(os.path.join(out_dir, "results.json")), "results.json written")
    check("per_fold" in sr and 0 in sr["per_fold"], "source_risk has fold-0 entry")
    pf0 = sr["per_fold"][0]
    tpr0 = pf0["tpr"]
    check(tpr0 is not None and 0.0 <= tpr0 <= 1.0, f"fold-0 TPR@1e-3 in [0,1] ({tpr0})")
    check(pf0["auroc"] is not None and 0.0 <= pf0["auroc"] <= 1.0,
          f"fold-0 AUROC in [0,1] ({pf0['auroc']})")
    check(os.path.isdir(os.path.join(out_dir, "scores")), "raw per-fold scores dumped")

    print("\n" + "=" * 72)
    print("FIRST DEEP-CORAL NUMBER (fold 0, tiny schedule, SOURCE-RISK/label-free)")
    print("=" * 72)
    print(f"  selected epoch    : {pf0['epoch']}   (val_auroc={_f(pf0['val_metric'])})")
    print(f"  fold-0 n_pos      : {pf0['n_pos']}")
    print(f"  fold-0 TPR@1e-3   : {_f(tpr0)}")
    print(f"  fold-0 AUROC      : {_f(pf0['auroc'])}")
    print(f"  (Baseline A to beat: pooled TPR@1e-3 ~= 0.18, AUROC ~= 0.90)")


def _f(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# PART 4 -- the stability fix: gamma schedule shape + blow-up detector (pure)
# ---------------------------------------------------------------------------

def part4_schedule_and_detector():
    print("\n" + "=" * 72)
    print("PART 4 -- gamma schedule shape + cov-term blow-up detector (pure)")
    print("=" * 72)
    from run_coral import coral_gamma_at_step

    PEAK, WF, MF, TOTAL = 1.0, 0.1, 0.1, 1000
    g = [coral_gamma_at_step(s, TOTAL, PEAK, schedule="warmup_cosine",
                             warmup_frac=WF, min_frac=MF) for s in range(TOTAL)]
    peak_at = int(np.argmax(g))

    # starts near 0
    check(g[0] < 1e-6, f"warmup_cosine starts near 0 (gamma[0]={g[0]:.3e})")
    # peaks AT coral_weight, at the end of warmup (~WF*TOTAL)
    check(abs(max(g) - PEAK) < 1e-6, f"warmup_cosine peak == coral_weight (max={max(g):.4f})")
    check(abs(peak_at - WF * TOTAL) <= 1, f"peak at end of warmup (~step {int(WF*TOTAL)}, "
                                          f"got {peak_at})")
    # ramps UP over warmup then DECAYS after it (monotone each side)
    ramp = g[:peak_at + 1]; decay = g[peak_at:]
    check(all(ramp[i] <= ramp[i + 1] + 1e-9 for i in range(len(ramp) - 1)),
          "gamma ramps up (non-decreasing) through warmup")
    check(all(decay[i] >= decay[i + 1] - 1e-9 for i in range(len(decay) - 1)),
          "gamma decays (non-increasing) after the peak")
    # ends near min_frac * coral_weight
    check(abs(g[-1] - MF * PEAK) < 5e-3,
          f"warmup_cosine ends near min_frac*peak ({g[-1]:.4f} ~ {MF*PEAK:.4f})")

    # warmup_hold: ramp then flat AT peak
    gh = [coral_gamma_at_step(s, TOTAL, PEAK, schedule="warmup_hold",
                              warmup_frac=WF) for s in range(TOTAL)]
    check(gh[0] < 1e-6 and abs(gh[-1] - PEAK) < 1e-9 and abs(max(gh) - PEAK) < 1e-9,
          f"warmup_hold ramps 0->peak then holds (start={gh[0]:.3e}, end={gh[-1]:.4f})")
    # constant: peak everywhere (backward-compat default)
    gc = [coral_gamma_at_step(s, TOTAL, PEAK, schedule="constant") for s in (0, 500, 999)]
    check(all(abs(v - PEAK) < 1e-12 for v in gc), "constant schedule == peak at every step")

    # --- blow-up detector: FIRES on a 20x spike, NOT on a flat/gently-rising series ---
    rng = np.random.default_rng(0)
    base_level = 1.0
    flat = [base_level + 0.02 * rng.standard_normal() for _ in range(200)]
    det_flat = BlowupDetector(baseline_window=30, blowup_factor=5.0)
    fired_flat = any(det_flat.update(c) for c in flat)
    check(not fired_flat, f"detector does NOT fire on a flat series (baseline="
                          f"{det_flat.baseline:.3g})")

    gentle = [base_level * (1.0 + 1.5 * i / 200.0) for i in range(200)]   # rises to ~2.5x
    det_gentle = BlowupDetector(baseline_window=30, blowup_factor=5.0)
    fired_gentle = any(det_gentle.update(c) for c in gentle)
    check(not fired_gentle, "detector does NOT fire on a gently-rising (<5x) series")

    spike = list(flat[:120])
    spike[120:] = [base_level * 20.0 for _ in range(80)]                  # late 20x blow-up
    det_spike = BlowupDetector(baseline_window=30, blowup_factor=5.0)
    fired_step = [i for i, c in enumerate(spike) if det_spike.update(c)]
    check(bool(fired_step) and det_spike.detected,
          f"detector FIRES on a 20x late spike (first at step {fired_step[0] if fired_step else None})")
    check(bool(fired_step) and fired_step[0] >= 120,
          "detector only fires AT/after the spike (never during the healthy window)")


# ---------------------------------------------------------------------------
# PART 5 -- stop_on_blowup prevents post-blow-up checkpoints (real cache)
# ---------------------------------------------------------------------------

class _StubDetector:
    """A BlowupDetector stand-in that fires deterministically after `fire_after`
    steps, so the runner's SAVE/STOP logic can be tested without a real pathology."""
    def __init__(self, fire_after, **_kw):
        self.fire_after = int(fire_after)
        self.n = 0
        self.ema = 0.0
        self.baseline = 1.0
        self.detected = False

    def update(self, cov_term):
        self.n += 1
        self.ema = float(cov_term)
        if self.n > self.fire_after:
            self.detected = True
            return True
        return False


def part5_stop_on_blowup_checkpoints():
    print("\n" + "=" * 72)
    print("PART 5 -- stop_on_blowup drops post-blow-up checkpoints (real cache)")
    print("=" * 72)
    import evaluate as ev
    import run_coral as rc

    cache = ev.FeatureCache(CACHE_DIR)
    need = ["source_train", "val", "target_fold0"]
    missing = [n for n in need if not cache.has(n)]
    if missing:
        print(f"SKIP PART 5: feature cache incomplete (missing {missing}).")
        return

    STEPS, EPOCHS = 5, (1, 2, 3)          # blow-up will be forced inside epoch 2
    ckpt_dir = os.path.join(SCRATCH, "coral_blowup_ckpts")
    if os.path.isdir(ckpt_dir):
        for fn in os.listdir(ckpt_dir):
            os.remove(os.path.join(ckpt_dir, fn))

    orig = rc.BlowupDetector
    try:
        # fire on the FIRST step of epoch 2 (steps 1..5 = epoch 1, step 6 = epoch 2)
        rc.BlowupDetector = lambda **kw: _StubDetector(fire_after=STEPS)
        ckpts, stab = rc.train_coral_fold(
            cache, 0, ckpt_dir, head_type="linear", epochs=EPOCHS, batch_size=64,
            steps_per_epoch=STEPS, coral_weight=1.0, coral_weight_schedule="warmup_cosine",
            stop_on_blowup=True, seed=42, device="cpu", verbose=True)
    finally:
        rc.BlowupDetector = orig

    saved_epochs = sorted(int(e["epoch"]) for e in stab["epochs"] if e["saved"])
    ckpt_files = sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []
    check(stab["blowup_epoch"] == 2, f"blow-up recorded at epoch 2 (got {stab['blowup_epoch']})")
    check(saved_epochs == [1], f"only the pre-blow-up epoch 1 was saved (saved {saved_epochs})")
    check(len(ckpts) == 1 and "ep01" in os.path.basename(ckpts[0]),
          f"exactly one stable checkpoint, epoch 1 ({[os.path.basename(c) for c in ckpts]})")
    check(all("ep02" not in f and "ep03" not in f for f in ckpt_files),
          f"no epoch-2/3 checkpoint on disk ({ckpt_files})")
    check(not any(int(e["epoch"]) == 3 for e in stab["epochs"]),
          "training stopped -- epoch 3 never ran")

    # sanity: with stop_on_blowup=False the same forced blow-up does NOT drop checkpoints
    ckpt_dir2 = os.path.join(SCRATCH, "coral_blowup_nostop")
    if os.path.isdir(ckpt_dir2):
        for fn in os.listdir(ckpt_dir2):
            os.remove(os.path.join(ckpt_dir2, fn))
    try:
        rc.BlowupDetector = lambda **kw: _StubDetector(fire_after=STEPS)
        ckpts2, stab2 = rc.train_coral_fold(
            cache, 0, ckpt_dir2, head_type="linear", epochs=EPOCHS, batch_size=64,
            steps_per_epoch=STEPS, coral_weight=1.0, coral_weight_schedule="warmup_cosine",
            stop_on_blowup=False, seed=42, device="cpu", verbose=False)
    finally:
        rc.BlowupDetector = orig
    saved2 = sorted(int(e["epoch"]) for e in stab2["epochs"] if e["saved"])
    check(saved2 == [1, 2, 3] and stab2["blowup_epoch"] is None,
          f"stop_on_blowup=False keeps all scheduled checkpoints (saved {saved2}, "
          f"blowup_epoch={stab2['blowup_epoch']})")


# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Deep CORAL tests")
    print(f"  data_root : {DATA_ROOT}")
    print("=" * 72)
    part1_loss_unit_tests()
    part2_gradient_routing()
    part3_end_to_end()
    part4_schedule_and_detector()
    part5_stop_on_blowup_checkpoints()

    print("\n" + "=" * 72)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL CORAL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
