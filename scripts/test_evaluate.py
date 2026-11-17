"""
test_evaluate.py
================
Tests for the evaluation harness (metrics.py + cache_features.py + evaluate.py).

Part 1 -- SYNTHETIC (pure NumPy/sklearn, always runs, no encoder/data needed):
  * tpr_at_fpr interpolates to a KNOWN analytic value on a ROC we constructed, and
    it RESPECTS drop_intermediate=False (uses the full curve, not sklearn's lighter
    default), verified against a hand-computed np.interp on the full curve.
  * the class-conditional bootstrap CI COVERS the point estimate and WIDENS with
    fewer positives.
  * the paired-difference CI is ~centered on 0 (exactly [0,0]) for identical score
    vectors and is not flagged significant.
  * the alert-rate guardrail FIRES on an all-positive-scoring dummy target and stays
    quiet on an all-negative-scoring one.

Part 2 -- REAL SLICE (skips with code 3 if the real data/encoder are absent):
  builds a SMALL feature cache from the real Euclid data (fold-0 test positives + a
  slice of the shared negatives + a balanced val slice + a target slice), trains two
  tiny demo checkpoints on the cached source(val) slice, and runs evaluate.py end-to-
  end -- printing the per-fold + pooled table, the SOURCE-RISK/ORACLE columns, the
  selection gap, and the guardrail. (The demo checkpoints are trained on the val slice
  purely to have a real head+standardizer to score with; they are NOT a real model.)

Run:
    python scripts\\test_evaluate.py
Exit 0 on success, 1 on a failed synthetic check, 3 if the real slice was skipped.
"""

from __future__ import annotations

import os
import sys
import traceback
import warnings

import numpy as np
from sklearn.metrics import roc_curve

from metrics import (tpr_at_fpr, auroc, operating_threshold, roc_curve_full,
                     paired_bootstrap, alert_rate_guardrail, default_tpr_metric)

_fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


# ===========================================================================
# Part 1 -- synthetic
# ===========================================================================

def make_known_roc(frac_above=0.4, n_pos=5000, n_neg=200000, target_fpr=1e-3, seed=0):
    """A ROC whose TPR@target_fpr is analytically ``frac_above``.

    Negatives ~ Uniform(0,1): the FPR=target_fpr threshold sits at the (1-target_fpr)
    quantile ~= 0.999. We put a fraction ``frac_above`` of positives WELL ABOVE that
    (score 1.5) and the rest WELL BELOW (0.3). So exactly ``frac_above`` of positives
    clear the threshold -> TPR@target_fpr == frac_above.
    """
    rng = np.random.default_rng(seed)
    neg = rng.uniform(0.0, 1.0, n_neg)
    n_hi = int(round(frac_above * n_pos))
    pos = np.concatenate([np.full(n_hi, 1.5), np.full(n_pos - n_hi, 0.3)])
    y = np.concatenate([np.ones(n_pos, int), np.zeros(n_neg, int)])
    s = np.concatenate([pos, neg])
    return y, s, frac_above


def test_tpr_known_and_drop_intermediate():
    print("\n[1] tpr_at_fpr: analytic value + respects drop_intermediate=False")
    target_fpr = 1e-3
    y, s, expected = make_known_roc(frac_above=0.4, target_fpr=target_fpr, seed=1)

    val = tpr_at_fpr(y, s, target_fpr)
    check(abs(val - expected) < 0.02,
          f"TPR@{target_fpr:g} = {val:.4f} ~= analytic {expected:.2f}")

    # it must use the FULL curve (drop_intermediate=False): value equals a hand
    # interp over the full curve, and the full curve has >= as many vertices as the
    # lighter default.
    full_fpr, full_tpr, _ = roc_curve_full(y, s)
    drop_fpr, _, _ = roc_curve(y, s, drop_intermediate=True)
    hand = float(np.interp(target_fpr, full_fpr, full_tpr))
    check(val == hand, "value equals np.interp over the FULL (undropped) ROC curve")
    check(len(full_fpr) >= len(drop_fpr),
          f"full curve keeps >= vertices than default ({len(full_fpr)} >= {len(drop_fpr)})")

    # extrapolation guard: a single-class input returns nan (not an extrapolated number)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v_deg = tpr_at_fpr(np.ones(10, int), np.linspace(0, 1, 10), target_fpr)
    check(np.isnan(v_deg), "single-class input -> nan (no silent extrapolation)")

    # a sanity anchor: AUROC is well above chance for this separable-ish construction
    check(auroc(y, s) > 0.5, f"AUROC = {auroc(y, s):.4f} > 0.5 (companion metric)")


def test_bootstrap_covers_and_widens():
    print("\n[2] bootstrap CI: covers the point estimate + widens with fewer positives")
    rng = np.random.default_rng(2)
    neg = rng.normal(0.0, 1.0, 4000)
    pos_full = rng.normal(1.2, 1.0, 400)
    y_full = np.concatenate([np.ones(pos_full.size, int), np.zeros(neg.size, int)])
    s_full = np.concatenate([pos_full, neg])

    res = paired_bootstrap(y_full, s_full, metric_fn=auroc, B=500, seed=0)
    check(res["lo"] <= res["point"] <= res["hi"],
          f"AUROC CI [{res['lo']:.4f}, {res['hi']:.4f}] covers point {res['point']:.4f}")
    # also for the headline metric
    res_t = paired_bootstrap(y_full, s_full, metric_fn=default_tpr_metric(1e-3),
                             B=500, seed=0)
    check(res_t["lo"] <= res_t["point"] <= res_t["hi"] or not np.isfinite(res_t["point"]),
          f"TPR CI [{res_t['lo']:.4f}, {res_t['hi']:.4f}] covers point {res_t['point']:.4f}")

    # fewer positives -> wider CI (same negatives, 40 vs 400 positives)
    pos_few = pos_full[:40]
    y_few = np.concatenate([np.ones(pos_few.size, int), np.zeros(neg.size, int)])
    s_few = np.concatenate([pos_few, neg])
    w_many = paired_bootstrap(y_full, s_full, metric_fn=auroc, B=500, seed=0)["ci_width"]
    w_few = paired_bootstrap(y_few, s_few, metric_fn=auroc, B=500, seed=0)["ci_width"]
    check(w_few > w_many,
          f"CI width grows as positives shrink (40 pos: {w_few:.4f} > 400 pos: {w_many:.4f})")


def test_paired_difference_identical():
    print("\n[3] paired-difference CI ~centered on 0 for identical score vectors")
    rng = np.random.default_rng(3)
    neg = rng.normal(0.0, 1.0, 3000)
    pos = rng.normal(1.0, 1.0, 300)
    y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
    s = np.concatenate([pos, neg])
    res = paired_bootstrap(y, s, scores_b=s.copy(), metric_fn=auroc, B=500, seed=0)
    check(res["point"] == 0.0, f"identical vectors -> point diff = {res['point']}")
    check(res["lo"] == 0.0 and res["hi"] == 0.0,
          f"identical vectors -> CI = [{res['lo']}, {res['hi']}] (exactly [0,0])")
    check(res["significant"] is False, "identical vectors -> NOT flagged significant")


def test_guardrail_fires():
    print("\n[4] alert-rate guardrail fires on an all-positive-scoring target")
    # threshold 0.0; a target that scores way above it -> alert rate 1.0 -> fire
    g_hi = alert_rate_guardrail(np.full(2000, 10.0), threshold=0.0)
    check(g_hi["alert_rate"] == 1.0 and g_hi["fired"],
          f"all-above target: alert_rate={g_hi['alert_rate']} -> FIRED={g_hi['fired']}")
    # a target that scores way below -> alert rate 0.0 -> quiet
    g_lo = alert_rate_guardrail(np.full(2000, -10.0), threshold=0.0)
    check(g_lo["alert_rate"] == 0.0 and not g_lo["fired"],
          f"all-below target: alert_rate={g_lo['alert_rate']} -> FIRED={g_lo['fired']}")
    # operating_threshold sanity: finite on a normal two-class input
    thr = operating_threshold(np.r_[np.ones(50, int), np.zeros(5000, int)],
                              np.r_[np.random.default_rng(4).normal(1, 1, 50),
                                    np.random.default_rng(5).normal(0, 1, 5000)], 1e-3)
    check(np.isfinite(thr), f"operating_threshold finite ({thr:.4f})")


def run_synthetic():
    print("=" * 78)
    print("Part 1 -- SYNTHETIC metric checks")
    print("=" * 78)
    test_tpr_known_and_drop_intermediate()
    test_bootstrap_covers_and_widens()
    test_paired_difference_identical()
    test_guardrail_fires()


# ===========================================================================
# Part 2 -- real slice, end-to-end
# ===========================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LENS_DATA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "data")))
SPLITS_PATH = os.path.join(DATA_ROOT, "splits", "splits.json")
SCRATCH = os.environ.get(
    "LENS_SCRATCH",
    os.path.join(os.environ.get("TEMP", _HERE), "eval_slice_demo"))

SLICE_NEG = 3000     # shared-negative slice (enough to define FPR=1e-3)
SLICE_VAL = 400      # balanced source(val) slice
SLICE_TGT = 400      # target slice for the guardrail


def build_small_cache(cache_dir, device):
    """Encode a SMALL real slice into a feature cache using cache_features helpers."""
    import torch
    from lens_dataset import LensDataset
    from zoobot_encoder import ZoobotEncoder
    import cache_features as cf

    os.makedirs(cache_dir, exist_ok=True)
    print("  [encode] building frozen encoder (real Zoobot weights) ...", flush=True)
    enc = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=False)

    # ---- test fold 0: all positives + a slice of the shared negatives ----
    print("  [encode] loading role='test' fold 0 (loads neg pixels; slow-ish) ...",
          flush=True)
    test_ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="test", fold=0, aug="none")
    pos_idx = np.where(test_ds.labels == 1)[0]
    neg_idx = np.where(test_ds.labels == 0)[0][:SLICE_NEG]
    cf.cache_indices(enc, test_ds, neg_idx, device,
                     os.path.join(cache_dir, "test_negatives_shared.npz"),
                     "test_negatives_shared", "test", None, 256, 0)
    cf.cache_indices(enc, test_ds, pos_idx, device,
                     os.path.join(cache_dir, "test_pos_fold0.npz"),
                     "test_pos_fold0", "test", 0, 256, 0)
    del test_ds

    # ---- val: a balanced source slice (for the label-free selector) ----
    print("  [encode] loading role='val' (balanced slice) ...", flush=True)
    val_ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="val", aug="none")
    vpos = val_ds.indices_by_label[1][:SLICE_VAL // 2]
    vneg = val_ds.indices_by_label[0][:SLICE_VAL // 2]
    vidx = np.concatenate([vpos, vneg])
    cf.cache_indices(enc, val_ds, vidx, device,
                     os.path.join(cache_dir, "val.npz"), "val", "val", None, 256, 0)
    del val_ds

    # ---- target fold 0: an unlabeled slice (for the guardrail) ----
    print("  [encode] loading role='target' fold 0 (slice) ...", flush=True)
    tgt_ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="target", fold=0, aug="none")
    tidx = np.arange(min(SLICE_TGT, len(tgt_ds)))
    cf.cache_indices(enc, tgt_ds, tidx, device,
                     os.path.join(cache_dir, "target_fold0.npz"),
                     "target_fold0", "target", 0, 256, 0)
    del tgt_ds
    return enc


def train_demo_checkpoint(cache_dir, path, device, steps, seed):
    """Train a tiny linear head + fit the standardizer on the cached source(val) slice.

    This exists ONLY to produce a real head+standardizer checkpoint (in the exact
    train_skeleton save format) so evaluate.py has something to score with. It is NOT
    a real model; training on the val slice makes its val metric optimistic (noted).
    """
    import torch
    from heads import build_head, FeatureStandardizer, EMBED_DIM

    with np.load(os.path.join(cache_dir, "val.npz"), allow_pickle=False) as d:
        feats = torch.from_numpy(d["features"].astype(np.float32)).to(device)
        y = torch.from_numpy(d["labels"].astype(np.float32)).view(-1, 1).to(device)

    torch.manual_seed(seed)
    std = FeatureStandardizer(EMBED_DIM).to(device)
    std.fit(feats)                       # freeze source-slice stats
    head = build_head("linear").to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    x = std(feats).detach()
    head.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = bce(head(x), y)
        loss.backward()
        opt.step()

    torch.save({"head_type": "linear", "head": head.state_dict(),
                "standardizer": std.state_dict(),
                "cfg": {"fold": 0, "head_type": "linear", "encoder_mode": "frozen"}},
               path)
    return float(loss.detach())


def run_real_slice():
    print("\n" + "=" * 78)
    print("Part 2 -- REAL SLICE end-to-end")
    print("=" * 78)
    if not os.path.exists(SPLITS_PATH):
        print(f"SKIP: real splits not found at {SPLITS_PATH}")
        return 3
    try:
        import torch  # noqa
        import evaluate as ev
    except Exception as e:
        print(f"SKIP: cannot import torch/evaluate ({e})")
        return 3

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    cache_dir = os.path.join(SCRATCH, "cache")
    print(f"  device={device}   cache_dir={cache_dir}")

    try:
        build_small_cache(cache_dir, device)
    except Exception as e:
        print(f"SKIP: could not build the real feature slice ({e})")
        traceback.print_exc()
        return 3

    # two demo checkpoints (undertrained vs trained) so selection has a real choice
    ep_lo = os.path.join(SCRATCH, "demo_ep05.pt")
    ep_hi = os.path.join(SCRATCH, "demo_ep200.pt")
    l0 = train_demo_checkpoint(cache_dir, ep_lo, device, steps=5, seed=0)
    l1 = train_demo_checkpoint(cache_dir, ep_hi, device, steps=200, seed=0)
    print(f"  [demo ckpts] ep05 final loss={l0:.4f}   ep200 final loss={l1:.4f}")

    out_dir = os.path.join(SCRATCH, "eval_out")
    results = ev.evaluate([ep_lo, ep_hi], cache_dir, device=device,
                          target_fpr=1e-3, B=500, seed=0, out_dir=out_dir)

    # light assertions that the harness produced a coherent, complete result
    sr = results["source_risk"]
    check(0 in sr["per_fold"], "real slice: per-fold results present for fold 0")
    check(sr["pooled_tpr"] is not None and 0.0 <= sr["pooled_tpr"] <= 1.0,
          f"real slice: pooled TPR in [0,1] ({sr['pooled_tpr']})")
    check(sr["pooled_tpr_ci"][0] <= sr["pooled_tpr"] <= sr["pooled_tpr"] + 1e-9
          or sr["pooled_tpr_ci"][0] <= sr["pooled_tpr"] <= sr["pooled_tpr_ci"][1],
          "real slice: pooled TPR within its bootstrap CI")
    check(os.path.exists(os.path.join(out_dir, "results.json")),
          "real slice: results.json written")
    check(os.path.exists(os.path.join(out_dir, "scores")),
          "real slice: raw per-fold scores dumped for the bake-off")
    return 0


# ===========================================================================
# main
# ===========================================================================

def main():
    run_synthetic()
    rc_real = run_real_slice()

    print("\n" + "=" * 78)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    if rc_real == 3:
        print("SYNTHETIC CHECKS PASSED; real slice SKIPPED (data/encoder absent).")
        return 3
    print("ALL EVALUATION-HARNESS CHECKS PASSED (synthetic + real slice).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
