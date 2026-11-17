"""
bench_unfreeze_steptime.py
==========================
LEVER 1, spec §4 GATE: measure the REAL per-step wall clock for n_blocks 0/1/3 at the
matched batch size (256) with --fp16 on the actual GPU, and project the wall clock for
one 8000-step fold and for a full 4-method x 5-fold commit.

WHY THIS RUNS FIRST. The relative CPU ratios quoted in the spec (1.00 / 1.38 / 2.84) are
INDICATIVE ONLY -- they were measured on CPU at batch 8, where the arithmetic intensity and
the memory traffic are nothing like a batch-256 fp16 GPU step. The budget is ~50-60
GPU-hr/week, so the projection decides whether the screen is affordable BEFORE any
8000-step run starts.

TWO TIMING MODES
----------------
  --mode synthetic (default)  Random in-contract tensors, no DataLoader. Isolates the pure
                              GPU fwd+bwd+step cost and finds the OOM ceiling fast. This is
                              the number that scales with n_blocks.
  --mode real                 The genuine LensDataset -> balanced loader -> Trainer path
                              (aug="weak", n_views=1), so CPU augmentation + collation +
                              host->device copy are included. This is the number that
                              actually predicts wall clock.

MEMORY: GRADIENT CHECKPOINTING, **NOT** ACCUMULATION (this replaced an earlier approach)
-----------------------------------------------------------------------------------------
At n_blocks=3 the stored activations for a 256-image 224x224 backward exceed a 6 GB laptop
card. The fix is ``--grad-checkpointing`` (recompute activations in backward), which is
numerically exact AND leaves batch composition untouched, so TRUE batch 256 is preserved.

Gradient accumulation was tried first and REMOVED. It is exact only for a
per-sample-decomposable loss like Baseline A's BCE; it is NOT valid for three of the four
bake-off methods, which are batch-coupled by construction (CORAL's within-batch 640x640
covariance becomes a rank-32 estimate; FreeMatch's self-adaptive threshold EMA sees 8x more,
8x noisier updates; AdaMatch's distribution alignment uses per-batch statistics). PROJECT.md
7.4 pins batch size because it is a known result-flipping confound. Since this benchmark
exists to cost the FULL bake-off, timing it under a scheme that is invalid for 3/4 methods
would have produced a projection for runs we cannot legitimately do.

Run:
    python scripts/bench_unfreeze_steptime.py --fp16
    python scripts/bench_unfreeze_steptime.py --fp16 --grad-checkpointing
    python scripts/bench_unfreeze_steptime.py --fp16 --mode real --num-workers 4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from zoobot_encoder import ZoobotEncoder, EMBED_DIM, IN_CHANS, INPUT_SIZE
from heads import build_head, FeatureStandardizer
from train_skeleton import Trainer, TrainConfig

BATCH = 256
BUDGET_STEPS = 8000


# ---------------------------------------------------------------------------
# one timed configuration
# ---------------------------------------------------------------------------

def time_synthetic(n_blocks, device, *, batch=BATCH, fp16=True, grad_checkpointing=False,
                   warmup=8, iters=50, lr=1e-3, backbone_lr=1e-4, seed=42):
    """Time fwd+bwd+opt.step on random in-contract inputs at TRUE batch. Returns stats."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    enc = ZoobotEncoder(n_blocks=n_blocks, device=dev, verbose=False, self_check=False,
                        grad_checkpointing=grad_checkpointing)
    cfg = TrainConfig(head_type="linear", n_blocks=n_blocks, lr=lr,
                      backbone_lr=backbone_lr, batch_size=batch,
                      device=str(dev), amp=bool(fp16), seed=seed,
                      grad_checkpointing=grad_checkpointing)
    tr = Trainer(cfg, encoder=enc)
    # the standardizer is fitted in a real run; here any finite stats give the same cost
    tr.standardizer.set_stats(torch.zeros(EMBED_DIM), torch.ones(EMBED_DIM))

    x = (torch.rand(batch, IN_CHANS, INPUT_SIZE, INPUT_SIZE, device=dev) * 0.881)
    y = torch.randint(0, 2, (batch,), device=dev).float()

    def one_step():
        tr.opt.zero_grad(set_to_none=True)
        with tr._autocast():
            loss = tr.bce(tr.head(tr.standardizer(enc(x))), y.view(-1, 1))
        tr.scaler.scale(loss).backward()
        tr.scaler.step(tr.opt)
        tr.scaler.update()

    for _ in range(warmup):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0

    peak = (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else float("nan")
    gc_on = bool(getattr(enc, "grad_checkpointing", False))
    del tr, enc, x, y
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"ms_per_step": 1000.0 * secs / iters, "peak_gb": peak,
            "grad_checkpointing": gc_on, "batch": batch, "iters": iters}


def time_real(n_blocks, device, splits_path, data_root, cache_dir, *, batch=BATCH,
              fp16=True, grad_checkpointing=False, warmup=8, iters=50, num_workers=4,
              lr=1e-3, backbone_lr=1e-4, seed=42):
    """Time the genuine LensDataset -> balanced loader -> source-only Trainer step."""
    from lens_dataset import LensDataset
    from samplers import make_balanced_loader
    import evaluate as ev

    torch.manual_seed(seed)
    dev = torch.device(device)

    src_ds = LensDataset(splits_path, data_root, role="source_train", aug="weak", n_views=1)
    loader = make_balanced_loader(src_ds, batch, use_latent=False,
                                  num_batches=warmup + iters + 2, pos_fraction=0.5,
                                  seed=seed, num_workers=num_workers,
                                  pin_memory=True, persistent_workers=bool(num_workers))

    enc = ZoobotEncoder(n_blocks=n_blocks, device=dev, verbose=False, self_check=False,
                        grad_checkpointing=grad_checkpointing)
    cfg = TrainConfig(head_type="linear", n_blocks=n_blocks, lr=lr,
                      backbone_lr=backbone_lr, batch_size=batch, device=str(dev),
                      amp=bool(fp16), seed=seed, grad_checkpointing=grad_checkpointing)
    tr = Trainer(cfg, encoder=enc)
    cache = ev.FeatureCache(cache_dir)
    sf = torch.from_numpy(np.asarray(cache.load("source_train")["features"],
                                     dtype=np.float32)).to(dev)
    tr.standardizer.fit(sf)
    del sf

    def one_step(batch_dict):
        imgs = batch_dict["image"].to(dev, non_blocking=True)
        ys = batch_dict["label"].float().to(dev).view(-1, 1)
        tr.opt.zero_grad(set_to_none=True)
        with tr._autocast():
            loss = tr.bce(tr.head(tr.standardizer(enc(imgs))), ys)
        tr.scaler.scale(loss).backward()
        tr.scaler.step(tr.opt)
        tr.scaler.update()

    it = iter(loader)
    for _ in range(warmup):
        one_step(next(it))
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    done = 0
    for _ in range(iters):
        one_step(next(it))
        done += 1
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else float("nan")
    gc_on = bool(getattr(enc, "grad_checkpointing", False))

    del tr, enc, loader, it, src_ds
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"ms_per_step": 1000.0 * secs / done, "peak_gb": peak,
            "grad_checkpointing": gc_on, "batch": batch, "iters": done}


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------

def project(ms_per_step, steps=BUDGET_STEPS):
    fold_hr = ms_per_step * steps / 1000.0 / 3600.0
    return {"fold_hours": fold_hr,
            "screen_fold0_hours": fold_hr,                 # one fold, one method
            "commit_4x5_hours": fold_hr * 20}              # 4 methods x 5 folds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap.add_argument("--mode", default="synthetic", choices=("synthetic", "real"))
    ap.add_argument("--data-root", default=os.path.join(root, "data"))
    ap.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    ap.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    ap.add_argument("--n-blocks", default="0,1,3",
                    help="comma list. Each entry is 'N' (use the global "
                         "--grad-checkpointing) or 'N:0'/'N:1' to pin checkpointing OFF/ON "
                         "for that entry. Pinning lets every config be timed in ONE "
                         "process, which is REQUIRED for an on-vs-off comparison: GPU "
                         "clock/thermal state drifts ~30% BETWEEN processes (measured on a "
                         "config where the flag is a provable no-op), so cross-process "
                         "timings are not comparable. Repeat the first entry to absorb the "
                         "cuDNN-autotune warmup and discard it.")
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute encoder activations in backward (exact; preserves TRUE "
                         "batch). No-op at n_blocks=0.")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write the measured table as JSON")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # parse "N" or "N:0"/"N:1" -> (n_blocks, grad_checkpointing)
    configs = []
    for tok in args.n_blocks.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            a, b = tok.split(":", 1)
            configs.append((int(a), bool(int(b))))
        else:
            configs.append((int(tok), bool(args.grad_checkpointing)))

    print("=" * 78)
    print("LEVER 1 §4 GATE -- measured step time (TRUE batch "
          f"{args.batch_size}, fp16={bool(args.fp16)}, mode={args.mode}, "
          f"grad_checkpointing={bool(args.grad_checkpointing)})")
    if device == "cuda":
        print(f"  device : {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")
    else:
        print(f"  device : {device}")
    print(f"  iters  : {args.iters} timed steps after {args.warmup} warmup steps")
    print("=" * 78, flush=True)

    rows = []
    for i, (n, gc) in enumerate(configs):
        tag = f"n_blocks={n} gc={'on' if gc else 'off'}"
        print(f"\n[{tag}] timing ...", flush=True)
        try:
            if args.mode == "synthetic":
                r = time_synthetic(n, device, batch=args.batch_size, fp16=args.fp16,
                                   grad_checkpointing=gc,
                                   warmup=args.warmup, iters=args.iters)
            else:
                r = time_real(n, device, args.splits, args.data_root, args.cache_dir,
                              batch=args.batch_size, fp16=args.fp16,
                              grad_checkpointing=gc,
                              warmup=args.warmup, iters=args.iters,
                              num_workers=args.num_workers)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at {tag} TRUE batch={args.batch_size}: {str(e)[:160]}")
            torch.cuda.empty_cache()
            rows.append({"n_blocks": n, "gc_requested": gc, "oom": True})
            continue
        r.update(project(r["ms_per_step"]))
        r["n_blocks"], r["gc_requested"], r["order"] = n, gc, i
        rows.append(r)
        print(f"  {r['ms_per_step']:8.1f} ms/step   peak {r['peak_gb']:.2f} GB   "
              f"-> {r['fold_hours']:.2f} h per 8000-step fold", flush=True)

    print("\n" + "=" * 78)
    print("MEASURED STEP TIME + WALL-CLOCK PROJECTION  (TRUE batch "
          f"{args.batch_size}; all rows timed in ONE process)")
    print("=" * 78)
    print(f"{'n_blocks':>8} | {'ckpt':>5} | {'ms/step':>9} | {'ratio':>6} | {'peak GB':>8} | "
          f"{'1 fold (8k)':>12}")
    print("-" * 78)
    # baseline for the ratio = the LAST n_blocks=0 row (the warmup duplicate is discarded)
    zeros = [r for r in rows if r.get("n_blocks") == 0 and not r.get("oom")]
    base = zeros[-1]["ms_per_step"] if zeros else None
    for j, r in enumerate(rows):
        n = r["n_blocks"]
        note = "  <- WARMUP, discard" if j == 0 else ""
        if r.get("oom"):
            print(f"{n:>8} | {'on' if r['gc_requested'] else 'off':>5} | {'OOM':>9} | "
                  f"{'-':>6} | {'-':>8} | {'-':>12}{note}")
            continue
        ratio = (r["ms_per_step"] / base) if base else float("nan")
        print(f"{n:>8} | {'on' if r['grad_checkpointing'] else 'off':>5} | "
              f"{r['ms_per_step']:9.1f} | {ratio:6.2f} | {r['peak_gb']:8.2f} | "
              f"{r['fold_hours']:10.2f} h{note}")
    print("-" * 78)
    print(f"  Card: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB"
          if device == "cuda" else "")
    print("  ckpt=on at n_blocks=0 is reported as 'off': the flag is a no-op with a frozen "
          "encoder.")
    print("  Budget reference: ~50-60 GPU-hr/week (PROJECT.md §6).")
    ns = [r["n_blocks"] for r in rows]

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"device": device, "batch": args.batch_size, "fp16": bool(args.fp16),
                       "grad_checkpointing": bool(args.grad_checkpointing),
                       "mode": args.mode, "rows": rows}, fh, indent=2)
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
