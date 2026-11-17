"""
probe_t4_freematch.py
=====================
LEVER 1 §5 -- the KAGGLE T4 MEMORY/STEP-TIME PROBE.  This is a PROBE, NOT a training run:
it measures, it trains nothing, it writes no checkpoints, it produces no result that could
enter the paper.

WHY IT EXISTS. The Lever-1 step-time table was measured on an RTX 4050 (Ada, 6 GB). Those
numbers do NOT transfer to a Kaggle T4 (Turing, 16 GB) -- different architecture, different
memory ceiling, no Ada tensor-core throughput. The 5-fold commit cannot be costed without
T4 numbers.

WHY FREEMATCH SPECIFICALLY. It is the most memory-hungry method in the bake-off: the step
runs THREE encoder passes (source weak + target weak + target strong) versus Baseline A's
one. Peak memory is what decides whether an n_blocks=3 commit is possible at all.

THE EXTRAPOLATION THIS PROBE EXISTS TO TEST  [UNVERIFIED until run]
------------------------------------------------------------------
  * At n_blocks=1 the local peak was 0.42 GB vs 0.33 GB frozen, because backward only
    reaches stages[3] at 7x7 resolution. Three or four encoder passes should therefore
    still be small and fit anywhere.
  * At n_blocks=3, source-only at true batch 256 peaked at 6.31 GB locally. Scaling that
    toward three passes extrapolates to roughly ~19 GB, which would EXCEED a 16 GB T4.
    That figure is scaled from a SINGLE-forward measurement and is exactly the kind of
    unbacked extrapolation this project forbids acting on -- hence this probe.
  If it holds, n_blocks=3 DA runs need gradient checkpointing on EVERY venue, and moving to
  Kaggle does not rescue you from it.

VENUE RULES (PROJECT.md 7.7 -- do not "fix" these)
--------------------------------------------------
  * T4 ONLY. NOT P100: sm_60 is incompatible with Kaggle's PyTorch build (it ships sm_70+
    kernels), and the P100 has no tensor cores for fp16.
  * --fp16, --num-workers 4.
  * data.zip MUST be built with forward-slash arcname (Windows os.path.join embeds
    backslashes as literal filename characters, which Kaggle rejects).
  * Upload the whole scripts/ folder.

METHOD. 30 timed steps after warmup, TRUE batch 256 (never micro-batched -- accumulation is
invalid for FreeMatch, whose self-adaptive threshold is an EMA over per-batch confidence
statistics). ALL configurations are timed IN ONE PROCESS and the FIRST one is discarded:
cuDNN autotune / clock ramp inflates the first config, and cross-process GPU clock drift was
MEASURED locally at ~30% on a configuration where the varied flag was a provable no-op.

DEFAULT MODE IS SYNTHETIC -- random in-contract tensors, no data needed, so the probe runs
on a bare Kaggle session without unpacking data.zip. Memory and step time are set by tensor
shapes and the autograd graph, both of which are identical to a real batch. Pass
--mode real to include the genuine loader.

Run on Kaggle (T4):
    python scripts/probe_t4_freematch.py --fp16
    python scripts/probe_t4_freematch.py --fp16 --mode real --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lens_dataset import UNLABELED                          # noqa: E402
from zoobot_encoder import ZoobotEncoder, EMBED_DIM, IN_CHANS, INPUT_SIZE   # noqa: E402
from train_skeleton import Trainer, TrainConfig             # noqa: E402

BATCH = 256
BUDGET_STEPS = 8000

# the four configurations the spec asks for, plus a discarded warmup duplicate first
DEFAULT_CONFIGS = "1:0,1:0,1:1,3:0,3:1"


def _synth_batches(batch, device, labelled=True):
    """One source batch (weak) and one target batch (weak+strong), in-contract."""
    def img():
        return torch.rand(batch, IN_CHANS, INPUT_SIZE, INPUT_SIZE, device=device) * 0.881
    src = {"image": img(),
           "label": torch.randint(0, 2, (batch,), device=device)}
    tgt = {"image": img(), "image_strong": img(),
           # the label-free contract: assert_target_unlabeled() runs EVERY step
           "label": torch.full((batch,), UNLABELED, dtype=torch.long, device=device)}
    return src, tgt


def time_freematch(n_blocks, grad_checkpointing, device, *, batch=BATCH, fp16=True,
                   warmup=8, iters=30, lr=1e-3, backbone_lr=1e-4, seed=42):
    torch.manual_seed(seed)
    dev = torch.device(device)

    enc = ZoobotEncoder(n_blocks=n_blocks, device=dev, verbose=False, self_check=False,
                        grad_checkpointing=grad_checkpointing)
    cfg = TrainConfig(
        head_type="linear", n_blocks=n_blocks, lr=lr, backbone_lr=backbone_lr,
        batch_size=batch, device=str(dev), amp=bool(fp16), seed=seed,
        grad_checkpointing=grad_checkpointing,
        use_freematch=True, total_steps=BUDGET_STEPS)
    tr = Trainer(cfg, encoder=enc)
    tr.standardizer.set_stats(torch.zeros(EMBED_DIM), torch.ones(EMBED_DIM))

    src, tgt = _synth_batches(batch, dev)

    for i in range(warmup):
        tr.train_step(src, tgt, step=i, total_steps=BUDGET_STEPS)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for i in range(iters):
        tr.train_step(src, tgt, step=warmup + i, total_steps=BUDGET_STEPS)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0

    peak = (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else float("nan")
    gc_on = bool(getattr(enc, "grad_checkpointing", False))
    del tr, enc, src, tgt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"ms_per_step": 1000.0 * secs / iters, "peak_gb": peak,
            "grad_checkpointing": gc_on, "n_blocks": n_blocks, "batch": batch}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS,
                    help="comma list of 'n_blocks:grad_checkpointing(0|1)'. The FIRST entry "
                         "is discarded as warmup -- repeat it.")
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    configs = []
    for tok in args.configs.split(","):
        tok = tok.strip()
        if tok:
            a, b = tok.split(":")
            configs.append((int(a), bool(int(b))))

    print("=" * 82)
    print("LEVER 1 §5 -- KAGGLE T4 PROBE: FreeMatch, TRUE batch "
          f"{args.batch_size}, fp16={bool(args.fp16)}")
    print("  PROBE ONLY -- measures step time + peak memory. Trains nothing, saves nothing.")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {props.name}  {props.total_memory / 2**30:.1f} GB  "
              f"sm_{props.major}{props.minor}")
        if props.major == 6:
            print("  *** WARNING: sm_60 = P100. PROJECT.md 7.7 says T4 ONLY -- Kaggle's "
                  "PyTorch build ships sm_70+ kernels and the P100 has no fp16 tensor "
                  "cores. These numbers would not be usable. ***")
    else:
        print(f"  device : {device}  (CPU -- memory figures are meaningless here)")
    print(f"  torch  : {torch.__version__}")
    print("=" * 82, flush=True)

    rows = []
    for i, (n, gc) in enumerate(configs):
        tag = f"n_blocks={n} ckpt={'on' if gc else 'off'}"
        print(f"\n[{tag}] timing ...", flush=True)
        try:
            r = time_freematch(n, gc, device, batch=args.batch_size, fp16=args.fp16,
                               warmup=args.warmup, iters=args.iters)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Exhaustion surfaces TWO different ways and only one is an OutOfMemoryError:
            #   * torch.cuda.OutOfMemoryError -- the caching allocator refused; recoverable.
            #   * RuntimeError("CUDA error: out of memory") -- a driver-level failure raised
            #     from inside a kernel (MEASURED: F.layer_norm at n_blocks=3, ckpt off).
            # The second is NOT reliably recoverable: the CUDA context can be left unusable,
            # so every later configuration would report a bogus failure. We record the OOM
            # (it IS the result the probe is looking for) and then STOP, rather than emit
            # numbers from a poisoned context.
            msg = str(e)
            if "out of memory" not in msg.lower():
                raise
            hard = not isinstance(e, torch.cuda.OutOfMemoryError)
            print(f"  OOM -- RECORDED AS A RESULT, not an error "
                  f"({'driver-level' if hard else 'allocator'}): {msg.splitlines()[0][:110]}")
            rows.append({"n_blocks": n, "grad_checkpointing": gc, "oom": True,
                         "oom_kind": ("driver" if hard else "allocator"), "order": i})
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if hard:
                print("  [stop] a driver-level CUDA OOM can leave the context unusable, so "
                      "no further configurations are timed in this process. Put the "
                      "expected-OOM config LAST (it already is) and re-run the rest "
                      "separately if needed.")
                break
            continue
        r["order"] = i
        rows.append(r)
        print(f"  {r['ms_per_step']:8.1f} ms/step   peak {r['peak_gb']:.2f} GB", flush=True)

    print("\n" + "=" * 82)
    print("T4 PROBE RESULTS -- FreeMatch, TRUE batch 256 (all rows timed in ONE process)")
    print("=" * 82)
    print(f"{'n_blocks':>8} | {'ckpt':>5} | {'ms/step':>9} | {'peak GB':>8} | "
          f"{'1 fold (8k)':>12} | {'5-fold':>9}")
    print("-" * 82)
    for j, r in enumerate(rows):
        note = "   <- WARMUP, discard" if j == 0 else ""
        gc = "on" if r["grad_checkpointing"] else "off"
        if r.get("oom"):
            print(f"{r['n_blocks']:>8} | {gc:>5} | {'OOM':>9} | {'OOM':>8} | {'-':>12} | "
                  f"{'-':>9}{note}")
            continue
        fold_h = r["ms_per_step"] * BUDGET_STEPS / 1000.0 / 3600.0
        print(f"{r['n_blocks']:>8} | {gc:>5} | {r['ms_per_step']:9.1f} | "
              f"{r['peak_gb']:8.2f} | {fold_h:10.2f} h | {fold_h * 5:7.1f} h{note}")
    print("-" * 82)
    print("  '5-fold' = this method, 5 folds, 8000 steps. FreeMatch only -- the full")
    print("  commit also needs Baseline A / CORAL / AdaMatch (fewer encoder passes each).")
    print("  Budget reference: ~50-60 GPU-hr/week (PROJECT.md §6).")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"device": device, "batch": args.batch_size,
                       "fp16": bool(args.fp16), "rows": rows,
                       "gpu": (torch.cuda.get_device_name(0) if device == "cuda" else None)},
                      fh, indent=2)
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
