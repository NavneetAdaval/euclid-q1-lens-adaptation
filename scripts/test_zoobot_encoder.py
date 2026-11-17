"""
test_zoobot_encoder.py
======================
REAL-weights test for `ZoobotEncoder`. Downloads the actual Zoobot greyscale
ConvNeXT-Nano checkpoint, runs a REAL batch of stretched images from
`LensDataset(role="source_train")` through it, and asserts the frozen-encoder
contract:

  1. output shape is exactly (B, 640)
  2. output is finite (no NaN/inf)
  3. every encoder parameter has requires_grad == False
  4. gradients do NOT flow: the frozen forward detaches (out.requires_grad is False,
     out.grad_fn is None), and even with a grad-tracking input no parameter accumulates
     a .grad and the input receives no gradient.

It also prints the pretrained_cfg mean/std the loader found, so we can confirm by eye
that the ImageNet metadata is present-but-unused.

Run:
    python scripts\\test_zoobot_encoder.py
Exits 0 on success, 1 on any failed check, 3 if the real splits/data are absent.
"""

import os
import sys
import json
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from lens_dataset import LensDataset
from zoobot_encoder import ZoobotEncoder, EMBED_DIM, IN_CHANS, INPUT_SIZE

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LENS_DATA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "data")))
SPLITS_PATH = os.path.join(DATA_ROOT, "splits", "splits.json")
BATCH = 8

_fails = []
def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _fails.append(msg)


def run():
    torch.manual_seed(0)

    if not os.path.exists(SPLITS_PATH):
        print(f"SKIP: real splits not found at {SPLITS_PATH}")
        return 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 68)
    print("REAL-weights test for ZoobotEncoder")
    print(f"  data_root : {DATA_ROOT}")
    print(f"  device    : {device}")
    print("=" * 68)

    # ---- load the encoder (downloads real weights; prints cfg mean/std) ----
    print("\n[load] building encoder (downloads checkpoint on first run)")
    enc = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=True)
    print(f"\n  found pretrained_cfg mean = {enc.cfg_mean}")
    print(f"  found pretrained_cfg std  = {enc.cfg_std}")
    print(f"  found pretrained_cfg input_size = {enc.cfg_input_size}")

    # ---- pull one real, already-stretched batch from LensDataset ----
    print("\n[data] loading a real source_train batch")
    ds = LensDataset(SPLITS_PATH, DATA_ROOT, role="source_train", aug="none")
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    imgs = batch["image"].to(device)
    check(tuple(imgs.shape) == (BATCH, IN_CHANS, INPUT_SIZE, INPUT_SIZE),
          f"input batch is {tuple(imgs.shape)} == ({BATCH},{IN_CHANS},{INPUT_SIZE},{INPUT_SIZE})")
    check(float(imgs.min()) >= -1e-3 and float(imgs.max()) <= 0.9 + 1e-3,
          f"input already stretched to ~[0,0.9] (min={imgs.min():.4f} max={imgs.max():.4f})")

    # ---- 1 & 2. shape + finite ----
    print("\n[forward] frozen forward on the real batch")
    out = enc(imgs)
    check(tuple(out.shape) == (BATCH, EMBED_DIM),
          f"output shape {tuple(out.shape)} == ({BATCH},{EMBED_DIM})")
    check(torch.isfinite(out).all().item(), "output is finite (no NaN/inf)")
    print(f"  feature stats: mean={out.mean():.4f} std={out.std():.4f} "
          f"min={out.min():.4f} max={out.max():.4f}")

    # ---- 3. all encoder params frozen ----
    n_params = sum(p.numel() for p in enc.parameters())
    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    check(n_train == 0,
          f"all encoder params requires_grad=False ({n_train}/{n_params} trainable)")

    # ---- 4. gradients do not flow ----
    print("\n[grad] verifying no gradients flow through the frozen encoder")
    check(out.requires_grad is False and out.grad_fn is None,
          "frozen output is detached (requires_grad=False, grad_fn=None)")

    # even with a grad-tracking input, nothing should accumulate a gradient
    enc.zero_grad(set_to_none=True)
    x = imgs.clone().requires_grad_(True)
    out2 = enc(x)
    check(out2.requires_grad is False,
          "output has requires_grad=False even when input.requires_grad=True")
    # out2 is detached, so it has no graph to backprop; confirm that is the case
    could_backward = out2.grad_fn is not None
    check(not could_backward, "no autograd graph exists to backprop through")
    # and no parameter or input gradient materialized
    any_param_grad = any(p.grad is not None for p in enc.parameters())
    check(not any_param_grad and x.grad is None,
          "no encoder param grad and no input grad accumulated")

    print("\n" + "=" * 68)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL ENCODER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
