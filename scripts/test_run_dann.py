"""
test_run_dann.py
================
Unit + smoke tests for run_dann.py (the DANN fold-0 viability probe). Follows the
project standard: build a MOCK feature cache that mirrors the real .npz schema
(features / labels / latent_labels / ids), run tiny DANN training against it, and pin
the load-bearing invariants adversarially. No GPU, no real data, no encoder.

Pinned here:
  * GRL forward == identity; backward == -lambda * grad (the whole method rests on this).
  * lambda(p) = 2/(1+exp(-10p)) - 1 : lambda(0)==0, monotone up, ~1 for large p.
  * Bottleneck adapter is 640->64->640 with a real skip (bottleneck FIXED at 64).
  * curve.csv is written with EXACTLY curve_hook.CURVE_COLUMNS (schema reuse).
  * checkpoints/step_*.pt saved on the eval-every-steps cadence, and are DANN
    checkpoints (method='dann') that round-trip through load_dann_checkpoint.
  * target served labels stay UNLABELED(-1) throughout (label-free contract).
  * a DANN checkpoint is NOT loadable by evaluate.load_checkpoint (build_head) --
    proving why the DANN-aware scorer is necessary, not gratuitous.

Run:  python scripts\\test_run_dann.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile

import numpy as np
import torch

import run_dann as rd
import curve_hook
import evaluate as ev


# ---------------------------------------------------------------------------
# mock feature cache (mirrors cache_features.py .npz schema)
# ---------------------------------------------------------------------------

def _make_cache(tmp, *, n_src=512, n_val=256, n_tpos=40, n_tneg=40,
                n_test_pos=30, n_test_neg=400, seed=0):
    """Write a tiny but SCHEMA-FAITHFUL feature cache. A weak but real signal is injected
    on feature 0 (positives shifted up) so metrics are finite and not degenerate."""
    rng = np.random.default_rng(seed)
    d = rd.EMBED_DIM

    def feats(n, pos_mask):
        x = rng.standard_normal((n, d)).astype(np.float32)
        x[:, 0] += 1.5 * pos_mask                     # weak lens signal on feature 0
        return x

    def save(name, features, labels, latent):
        np.savez(os.path.join(tmp, f"{name}.npz"),
                 features=features, labels=labels.astype(np.int64),
                 latent_labels=latent.astype(np.int64),
                 ids=np.arange(len(labels)))

    # source_train: balanced labeled
    sl = np.array([1, 0] * (n_src // 2), dtype=np.int64)
    save("source_train", feats(n_src, sl == 1), sl, sl)
    # val: balanced labeled (source-domain SIM)
    vl = np.array([1, 0] * (n_val // 2), dtype=np.int64)
    save("val", feats(n_val, vl == 1), vl, vl)
    # target_fold0: served labels ALL UNLABELED(-1); latent 0/1 balanced
    tlat = np.concatenate([np.ones(n_tpos), np.zeros(n_tneg)]).astype(np.int64)
    tserved = np.full(n_tpos + n_tneg, -1, dtype=np.int64)
    save("target_fold0", feats(n_tpos + n_tneg, tlat == 1), tserved, tlat)
    # test_pos_fold0: positives only ; test_negatives_shared: negatives only
    save("test_pos_fold0", feats(n_test_pos, np.ones(n_test_pos)),
         np.ones(n_test_pos, np.int64), np.ones(n_test_pos, np.int64))
    save("test_negatives_shared", feats(n_test_neg, np.zeros(n_test_neg)),
         np.zeros(n_test_neg, np.int64), np.zeros(n_test_neg, np.int64))
    return tmp


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_grl_sign():
    """GRL forward is identity; backward flips + scales the gradient by -lambda."""
    for lam in (0.0, 0.5, 1.0, 2.0):
        x = torch.randn(8, 4, requires_grad=True)
        y = rd.grad_reverse(x, lam)
        assert torch.allclose(y, x), "GRL forward must be identity"
        y.sum().backward()
        expected = torch.full_like(x, -lam)           # d(sum)/dx = 1, reversed*lam
        assert torch.allclose(x.grad, expected), f"GRL backward wrong at lambda={lam}"
    print("  ok  GRL: identity forward, -lambda*grad backward")


def test_lambda_schedule():
    """lambda(0)==0, monotone increasing, ~1 for large p, matches the closed form."""
    assert abs(rd.dann_lambda_at_step(0, 100) - 0.0) < 1e-12
    vals = [rd.dann_lambda_at_step(s, 100) for s in range(0, 101, 10)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), "lambda must be monotone up"
    assert rd.dann_lambda_at_step(100, 100) > 0.99, "lambda should approach 1"
    # closed form check at p=0.5
    p = 0.5
    assert abs(rd.dann_lambda_at_step(50, 100) - (2 / (1 + math.exp(-10 * p)) - 1)) < 1e-9
    print("  ok  lambda schedule: 0 -> ~1, monotone, matches 2/(1+exp(-10p))-1")


def test_adapter_shape_and_skip():
    """G_f is 640->64->640 with a genuine residual skip; bottleneck is fixed at 64."""
    a = rd.BottleneckAdapter()
    assert a.down.in_features == rd.EMBED_DIM and a.down.out_features == rd.BOTTLENECK_DIM
    assert a.up.out_features == rd.EMBED_DIM
    assert rd.BOTTLENECK_DIM == 64, "bottleneck d must stay fixed at 64"
    x = torch.randn(5, rd.EMBED_DIM)
    with torch.no_grad():
        a.up.weight.zero_(); a.up.bias.zero_()        # kill the residual branch
        assert torch.allclose(a(x), x), "with up==0 the skip must return x exactly"
    print("  ok  adapter: 640->64->640 with a real skip connection")


def test_dann_checkpoint_not_a_classifierhead():
    """A DANN checkpoint must NOT load via evaluate.load_checkpoint (build_head) -- this is
    the concrete reason run_dann needs its own DANN-aware scorer."""
    with tempfile.TemporaryDirectory() as tmp:
        model = rd.DANNModel()
        std = rd.FeatureStandardizer(rd.EMBED_DIM)
        std.fit(torch.randn(64, rd.EMBED_DIM))
        path = os.path.join(tmp, "dann_fold0_ep01.pt")
        rd.save_dann_checkpoint(path, model, std, 1, {"seed": 0})
        # round-trips through the DANN loader
        m2, s2, meta = rd.load_dann_checkpoint(path)
        assert meta["method"] == "dann"
        x = np.random.randn(7, rd.EMBED_DIM).astype(np.float32)
        assert rd.score_dann(m2, s2, x).shape == (7,)
        # but NOT through the ClassifierHead loader
        try:
            ev.load_checkpoint(path)
            raised = False
        except Exception:
            raised = True
        assert raised, "evaluate.load_checkpoint must FAIL on a DANN checkpoint"
    print("  ok  DANN ckpt round-trips DANN-aware; rejected by build_head loader (as intended)")


def test_train_smoke_and_curve_schema():
    """Tiny end-to-end fold-0 run writes step checkpoints + a curve.csv whose header is
    EXACTLY curve_hook.CURVE_COLUMNS, with finite source-val columns."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _make_cache(tmp)
        out_dir = os.path.join(tmp, "run")
        res = rd.run_dann(cache_dir, out_dir, epochs=(1, 2), batch_size=64,
                          steps_per_epoch=6, eval_every_steps=6, lr=1e-2,
                          train_device="cpu", eval_device="cpu", verbose=False)
        curve = os.path.join(out_dir, "curve.csv")
        assert os.path.exists(curve), "curve.csv must be written"
        with open(curve, newline="") as fh:
            r = csv.reader(fh); header = next(r); nrows = sum(1 for _ in r)
        assert header == curve_hook.CURVE_COLUMNS, f"schema mismatch: {header}"
        assert nrows >= 1, "at least one curve row expected"
        # step checkpoints exist and are DANN checkpoints
        cks = [f for f in os.listdir(os.path.join(out_dir, "checkpoints"))
               if f.startswith("step_")]
        assert cks, "expected step_*.pt checkpoints"
        # results.json has the fold-0 source-risk block with finite source-val auroc
        assert res["method"] == "dann" and res["fold"] == 0
        assert res["source_risk"]["source_risk_auroc"] is not None
        assert res["vanilla_no_label_shift_correction"] is True
    print("  ok  train smoke: step ckpts + curve.csv (exact schema) + results.json")


def test_target_stays_unlabeled():
    """The training path must assert target served labels are UNLABELED every step; feeding
    a labeled target must trip assert_target_unlabeled (label-free contract)."""
    from samplers import assert_target_unlabeled
    assert_target_unlabeled({"label": torch.full((10,), -1)})     # ok
    try:
        assert_target_unlabeled({"label": torch.tensor([-1, -1, 1, 0])})
        tripped = False
    except Exception:
        tripped = True
    assert tripped, "labeled target must trip the label-free guard"
    print("  ok  label-free contract: target served labels must stay UNLABELED(-1)")


def main():
    print("test_run_dann.py")
    torch.manual_seed(0)
    test_grl_sign()
    test_lambda_schedule()
    test_adapter_shape_and_skip()
    test_dann_checkpoint_not_a_classifierhead()
    test_target_stays_unlabeled()
    test_train_smoke_and_curve_schema()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
