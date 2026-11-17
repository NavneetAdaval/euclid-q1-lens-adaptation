"""
test_train_skeleton.py
======================
REAL-DATA plumbing test for the label-free DA training SKELETON. It does NOT check
accuracy (there is no real DA loss yet) -- it proves the wiring is correct. It runs
on a small slice of the ACTUAL Euclid data and, for BOTH head_type="linear" and
head_type="mlp", asserts:

  1. BATCH SHAPE + BALANCE -- source batches are (B,1,224,224) with the requested
     positive fraction (within tolerance).
  2. LABEL-FREE TARGET     -- every target batch carries ONLY label == UNLABELED(-1);
     LensDataset's -1 survives collation (checked directly + via assert_target_unlabeled).
  3. LEARNING IS WIRED     -- on a deliberately tiny, fixed overfit subset the loss
     DECREASES over a few steps (last-window mean < first-window mean).
  4. GRADIENT ROUTING      -- after a step, EVERY encoder param has grad None and
     requires_grad False, while head params carry gradients. (The optimizer touches
     head/standardizer only, never the encoder.)
  5. CHECKPOINT ROUND-TRIP -- save head+standardizer, reload into a freshly-initialized
     model, and get identical outputs on the same cached features.

The encoder is loaded ONCE and shared across both head types (real Zoobot weights).

Run:
    python scripts\\test_train_skeleton.py
Exits 0 on success, 1 on any failed check, 3 if the real splits/data are absent.
"""

import os
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from lens_dataset import LensDataset, UNLABELED
from zoobot_encoder import ZoobotEncoder
from heads import build_head, FeatureStandardizer
from samplers import ClassBalancedBatchSampler, PairedDomainLoader, assert_target_unlabeled
from train_skeleton import Trainer, TrainConfig, encode_features

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LENS_DATA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "data")))
SPLITS_PATH = os.path.join(DATA_ROOT, "splits", "splits.json")
SCRATCH = os.environ.get("LENS_SCRATCH", _HERE)

# tiny fixed overfit subset: K of each class, small batches, high LR, a few steps
K_PER_CLASS = 8
BATCH = 8
N_STEPS = 40
POS_FRACTION = 0.5
BALANCE_TOL = 0.13          # |observed - requested| positive fraction
LR = 2e-2

_fails = []
def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _fails.append(msg)


def build_tiny_loaders(seed):
    """Two tiny class-balanced loaders (source + target) over fixed index subsets,
    plus their datasets. Deterministic so the head can genuinely overfit them."""
    src_ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="source_train", aug="none")
    tgt_ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="target", fold=0, aug="none")

    pos = src_ds.indices_by_label[1][:K_PER_CLASS]
    neg = src_ds.indices_by_label[0][:K_PER_CLASS]
    src_sampler = ClassBalancedBatchSampler(pos, neg, batch_size=BATCH,
                                            num_batches=N_STEPS,
                                            pos_fraction=POS_FRACTION, seed=seed)
    src_loader = DataLoader(src_ds, batch_sampler=src_sampler, num_workers=0)

    # target balances on LATENT identity (served labels are all UNLABELED)
    tpos = tgt_ds.indices_by_latent[1][:K_PER_CLASS]
    tneg = tgt_ds.indices_by_latent[0][:K_PER_CLASS]
    tgt_sampler = ClassBalancedBatchSampler(tpos, tneg, batch_size=BATCH,
                                            num_batches=N_STEPS,
                                            pos_fraction=POS_FRACTION, seed=seed + 100)
    tgt_loader = DataLoader(tgt_ds, batch_sampler=tgt_sampler, num_workers=0)
    return src_ds, tgt_ds, src_loader, tgt_loader


def check_batch_shape_and_balance(src_loader):
    fracs, shapes_ok = [], True
    for i, batch in enumerate(src_loader):
        if tuple(batch["image"].shape) != (BATCH, 1, 224, 224):
            shapes_ok = False
        labels = batch["label"]
        fracs.append(float((labels == 1).float().mean()))
        if i >= 4:
            break
    mean_frac = float(np.mean(fracs))
    check(shapes_ok, f"shape: source batches are ({BATCH},1,224,224)")
    check(abs(mean_frac - POS_FRACTION) <= BALANCE_TOL,
          f"balance: source positive fraction {mean_frac:.3f} ~= {POS_FRACTION} "
          f"(tol {BALANCE_TOL})")


def check_target_label_free(tgt_loader):
    leaks, guard_ok = 0, True
    for i, batch in enumerate(tgt_loader):
        leaks += int((batch["label"] != UNLABELED).sum())
        try:
            assert_target_unlabeled(batch)
        except AssertionError:
            guard_ok = False
        if i >= 4:
            break
    check(leaks == 0, "label-free: every target label == UNLABELED(-1)")
    check(guard_ok, "label-free: assert_target_unlabeled() accepts the target batches")


def check_learning_wired(trainer, src_loader, tgt_loader):
    trainer.fit_standardizer(src_loader, max_batches=3)
    paired = PairedDomainLoader(src_loader, tgt_loader)
    history = trainer.fit(paired)
    w = max(3, N_STEPS // 5)
    first = float(np.mean(history[:w]))
    last = float(np.mean(history[-w:]))
    check(all(np.isfinite(history)), f"loss finite over all {len(history)} steps")
    check(last < first,
          f"learning: loss decreases (first-{w} mean {first:.4f} -> last-{w} mean {last:.4f})")
    return history


def check_gradient_routing(trainer, src_loader, tgt_loader):
    # one fresh step, then inspect grads BEFORE any zeroing
    src_batch = next(iter(src_loader))
    tgt_batch = next(iter(tgt_loader))
    trainer.train_step(src_batch, tgt_batch)

    enc_bad_rg = sum(1 for p in trainer.encoder.parameters() if p.requires_grad)
    enc_bad_grad = sum(1 for p in trainer.encoder.parameters() if p.grad is not None)
    head_with_grad = sum(1 for p in trainer.head.parameters() if p.grad is not None)
    head_total = sum(1 for _ in trainer.head.parameters())

    check(enc_bad_rg == 0, f"grad: no encoder param has requires_grad ({enc_bad_rg} do)")
    check(enc_bad_grad == 0, f"grad: no encoder param accumulated a .grad ({enc_bad_grad} did)")
    check(head_with_grad == head_total and head_total > 0,
          f"grad: all {head_total} head params carry gradients ({head_with_grad} do)")


def check_checkpoint_roundtrip(trainer, encoder, cfg, src_ds):
    # cache raw encoder features for a fixed set of images (deterministic, aug=none)
    imgs = torch.stack([src_ds[i]["image"] for i in range(BATCH)]).to(trainer.device)
    with torch.no_grad():
        raw = encode_features(encoder, imgs, cfg.encoder_mode)

    trainer.head.eval()
    with torch.no_grad():
        out_a = trainer.head(trainer.standardizer(raw))

    path = os.path.join(SCRATCH, f"_ckpt_roundtrip_{cfg.head_type}.pt")
    trainer.save(path)

    # fresh, randomly-initialized model (shares the frozen encoder), then load.
    # Its head starts from a different random init and its standardizer is unfitted,
    # so identical outputs after load() prove the checkpoint fully restored both.
    fresh = Trainer(cfg, encoder=encoder)
    fresh.load(path)
    fresh.head.eval()
    with torch.no_grad():
        out_b = fresh.head(fresh.standardizer(raw))

    identical = torch.allclose(out_a, out_b, atol=1e-6)
    check(identical, "checkpoint: save->load reproduces identical head outputs")

    # element-wise: prove the standardizer stats were actually restored, not that
    # outputs coincidentally matched
    check(torch.equal(fresh.standardizer.mean, trainer.standardizer.mean),
          "checkpoint: standardizer.mean restored element-wise")
    check(torch.equal(fresh.standardizer.std, trainer.standardizer.std),
          "checkpoint: standardizer.std restored element-wise")
    check(torch.equal(fresh.standardizer.fitted, trainer.standardizer.fitted),
          "checkpoint: standardizer.fitted buffer restored")
    try:
        os.remove(path)
    except OSError:
        pass


def run_for_head(head_type, encoder):
    print(f"\n{'='*68}\nhead_type = {head_type!r}\n{'='*68}")
    cfg = TrainConfig(splits_path=SPLITS_PATH, data_root=DATA_ROOT, fold=0,
                      head_type=head_type, encoder_mode="frozen", lr=LR,
                      batch_size=BATCH, pos_fraction=POS_FRACTION, epochs=1,
                      device=str(next(encoder.parameters()).device), seed=0)

    src_ds, tgt_ds, src_loader, tgt_loader = build_tiny_loaders(seed=0)

    print("\n[1] batch shape + balance")
    check_batch_shape_and_balance(src_loader)

    print("\n[2] label-free target")
    check_target_label_free(tgt_loader)

    print("\n[3] learning is wired (tiny overfit subset)")
    trainer = Trainer(cfg, encoder=encoder)
    check_learning_wired(trainer, src_loader, tgt_loader)

    print("\n[4] gradient routing (encoder frozen, head trains)")
    check_gradient_routing(trainer, src_loader, tgt_loader)

    print("\n[5] checkpoint round-trip")
    check_checkpoint_roundtrip(trainer, encoder, cfg, src_ds)


def run():
    torch.manual_seed(0)
    if not os.path.exists(SPLITS_PATH):
        print(f"SKIP: real splits not found at {SPLITS_PATH}")
        return 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 68)
    print("REAL-DATA plumbing test for the DA training skeleton")
    print(f"  data_root : {DATA_ROOT}")
    print(f"  device    : {device}")
    print("=" * 68)

    print("\n[load] building the frozen encoder once (real Zoobot weights) ...")
    encoder = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=False)

    for head_type in ("linear", "mlp"):
        run_for_head(head_type, encoder)

    print("\n" + "=" * 68)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL SKELETON PLUMBING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
