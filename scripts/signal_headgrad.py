"""
signal_headgrad.py
==================
LABEL-FREE SELECTION-SIGNAL TEST #2 -- the HEAD-GRADIENT-NORM signal
(method: arXiv:2601.16874). For every checkpoint on the fold-0 learning curve of a
run, compute the Frobenius norm of the gradient of the final classifier-head weight
under the SOURCE supervised BCE loss, and write it as ``headgrad_bce_balanced`` into
``curve_with_signals.csv`` (the same file signal_purity.py writes). No retraining.

WHAT THE SIGNAL IS (the loss-term decision, confirmed).
  * The gradient loss is the SOURCE BCE-HEAD term ONLY:  BCE(head(source features),
    source labels) -- Option A. This is the paper's setting (a SINGLE supervised head
    loss, not a multi-term self-training loss) and it is the SAME quantity the baseline
    optimizes, so the signal is apples-to-apples across baseline AND FreeMatch. For
    FreeMatch we deliberately do NOT include its consistency / SAF terms (those are
    step-scheduled + confidence-masked, so their gradient tracks the schedule, not head
    convergence).
  * The "head weight" whose gradient we measure is the FINAL Linear(feat_dim -> 1)
    weight (``head.classifier.weight``) -- the literal classifier head weight, identical
    in structure across both methods' linear heads.
  * ENCODER-FREE: the source features are read from the FROZEN-FEATURE CACHE
    (``source_train``), forwarded through the (frozen) standardizer + head. The Zoobot
    encoder is never called -- one forward + one backward through the HEAD per batch.
  * VARIANCE CONTROL (paper's caveat): the norm is AVERAGED over several FIXED seeded
    balanced batches (default 6, in the paper's 4-8 range), not one, to cut single-batch
    variance. The SAME batches (same seeds, same drawn indices) are reused across ALL of
    a method's checkpoints, so the signal moves only with the checkpoint. Seeds + batch
    composition are printed.
  * NAME: ``headgrad_bce_balanced`` -- "bce" (the loss term) + "balanced" (the batch is
    class-balanced 1:1). The name leaves room for a future prevalence-weighted variant
    (``headgrad_bce_prevweighted``) if this balanced one fails to track the oracle.

WORKFLOW (one script, two environments -- NOT forked).
  * BASELINE: run LOCAL (cached-feature run dir + local source cache).
  * FREEMATCH: run on COLAB against the Drive-mounted run dir, pointing --cache-dir at
    the SAME source-feature cache on Drive that the FreeMatch curve run used. The probe
    is encoder-free either way; only the file locations differ, supplied by args. See the
    Colab command handed over with this script.

REUSE, DON'T REIMPLEMENT. Head/standardizer reconstruction = evaluate.load_checkpoint;
BCE = torch's binary_cross_entropy_with_logits (the exact baseline/FreeMatch supervised
term). This module owns only the batch draw + the gradient-norm plumbing.

READ-ONLY guarantee. The only file written is ``curve_with_signals.csv`` in the run dir
(the ``headgrad_bce_balanced`` column added/replaced; any sibling column such as
``purity_50comp`` is preserved). curve.csv / results.json / scores/ / checkpoints/ are
never modified. Any missing expected artifact (curve.csv, checkpoints/, a per-step
checkpoint, or the source-feature cache) is a LOUD error naming the file -- no silent skips.

USAGE
-----
    # baseline, local:
    python scripts/signal_headgrad.py runs/curve_baseline_fold0
    # FreeMatch, on Colab (see the handed-over command):
    python scripts/signal_headgrad.py <drive_run_dir> --cache-dir <drive_source_cache>
    # optional: --n-batches 6  --batch-size 256  --base-seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from evaluate import FeatureCache, load_checkpoint  # noqa: E402

SOURCE_SPLIT = "source_train"          # labeled source features (the BCE term's data)
NEW_COLUMN = "headgrad_bce_balanced"   # source-BCE head-weight grad norm, balanced batches
DEFAULT_N_BATCHES = 6                   # in the paper's 4-8 range
DEFAULT_BATCH_SIZE = 256               # class-balanced -> 128 lens / 128 non-lens
DEFAULT_BASE_SEED = 42


def _die(msg: str):
    raise SystemExit(f"[signal_headgrad] ERROR: {msg}")


def _require_file(path: str, what: str) -> str:
    if not os.path.isfile(path):
        _die(f"missing {what}: {path}")
    return path


def _require_dir(path: str, what: str) -> str:
    if not os.path.isdir(path):
        _die(f"missing {what}: {path}")
    return path


def _checkpoint_for_step(run_dir: str, step: int) -> str:
    path = os.path.join(run_dir, "checkpoints", f"step_{int(step)}.pt")
    return _require_file(path, f"checkpoint for curve step {int(step)}")


def build_fixed_batches(labels: np.ndarray, *, n_batches: int, batch_size: int,
                        base_seed: int) -> list:
    """Draw ``n_batches`` FIXED class-balanced index batches from the source labels.

    Batch b uses its own seed ``base_seed + b`` so the composition is reproducible and
    reportable; each batch is half positives / half negatives (sampled WITHOUT replacement
    within a batch). Returns a list of (seed, np.ndarray[int]) -- reused across every
    checkpoint so the signal reflects the checkpoint, not the batch draw."""
    if batch_size % 2 != 0:
        _die(f"batch_size must be even for a 1:1 balanced batch, got {batch_size}")
    per = batch_size // 2
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if len(pos_idx) < per or len(neg_idx) < per:
        _die(f"source cache has too few samples for a balanced batch of {batch_size} "
             f"(pos={len(pos_idx)}, neg={len(neg_idx)}, need {per} each)")
    batches = []
    for b in range(n_batches):
        seed = base_seed + b
        rng = np.random.default_rng(seed)
        sel = np.concatenate([rng.choice(pos_idx, size=per, replace=False),
                              rng.choice(neg_idx, size=per, replace=False)])
        batches.append((seed, sel))
    return batches


@torch.no_grad()
def _finite_check(t: torch.Tensor, what: str):
    if not torch.isfinite(t).all():
        _die(f"non-finite {what}")


def headgrad_norm(head, std, feats: torch.Tensor, labels: torch.Tensor,
                  batches: list, device: str) -> float:
    """Mean over the fixed batches of ||d/dW_classifier BCE(head(std(x)), y)||_F.

    One forward + one backward through the HEAD per batch (encoder never touched); the
    frozen standardizer is applied but carries no trainable params. Returns the averaged
    Frobenius norm of ``head.classifier.weight.grad``."""
    head.eval().to(device)          # eval -> deterministic (no MLP dropout); LayerNorm is identical
    std.eval().to(device)
    W = head.classifier.weight       # the final Linear(feat_dim->1) weight = "head weight"
    norms = []
    for _seed, idx in batches:
        x = feats[idx].to(device)
        y = labels[idx].to(device).float().reshape(-1)
        head.zero_grad(set_to_none=True)
        logits = head(std(x)).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)   # the source BCE-head term
        loss.backward()
        if W.grad is None:
            _die("classifier head weight received no gradient (unexpected graph break)")
        g = W.grad.detach()
        _finite_check(g, "head-weight gradient")
        norms.append(float(torch.linalg.norm(g).cpu()))   # Frobenius norm of the weight grad
    head.zero_grad(set_to_none=True)
    return float(np.mean(norms))


def compute_headgrad_column(run_dir: str, cache_dir: str, *, n_batches: int,
                            batch_size: int, base_seed: int, device: str = "cpu") -> pd.DataFrame:
    run_dir = os.path.abspath(run_dir)
    _require_dir(run_dir, "run dir")
    curve_csv = _require_file(os.path.join(run_dir, "curve.csv"), "curve.csv")
    _require_dir(os.path.join(run_dir, "checkpoints"), "checkpoints/ dir")
    _require_dir(cache_dir, "feature cache dir")

    cache = FeatureCache(cache_dir)
    _require_file(cache.path(SOURCE_SPLIT), f"feature-cache split '{SOURCE_SPLIT}'")

    # append onto an existing curve_with_signals.csv (preserve purity_50comp etc.)
    signals_csv = os.path.join(run_dir, "curve_with_signals.csv")
    df = pd.read_csv(signals_csv if os.path.isfile(signals_csv) else curve_csv)
    if "step" not in df.columns:
        _die(f"curve has no 'step' column (found {list(df.columns)})")

    src = cache.load(SOURCE_SPLIT)
    feats = torch.from_numpy(np.asarray(src["features"], dtype=np.float32))
    labels_np = np.asarray(src["labels"]).astype(np.int64)
    labels = torch.from_numpy(labels_np)

    batches = build_fixed_batches(labels_np, n_batches=n_batches, batch_size=batch_size,
                                  base_seed=base_seed)
    seeds = [s for s, _ in batches]
    print(f"[signal_headgrad] batches: {n_batches} x balanced batch_size={batch_size} "
          f"({batch_size//2} lens / {batch_size//2} non-lens each), WITHOUT replacement")
    print(f"[signal_headgrad] seeds (reused across all checkpoints): {seeds}")

    vals = []
    for _, row in df.iterrows():
        step = int(row["step"])
        ckpt = _checkpoint_for_step(run_dir, step)
        head, std, _meta = load_checkpoint(ckpt, device)
        val = headgrad_norm(head, std, feats, labels, batches, device)
        vals.append(val)
        print(f"  step {step:>7}: {NEW_COLUMN}={val:.6f}")

    out = df.copy()
    out[NEW_COLUMN] = vals
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run dir containing curve.csv + checkpoints/ (baseline "
                                    "local, or a Drive-mounted FreeMatch run dir on Colab)")
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(_HERE),
                                                        "data", "feature_cache"),
                    help="frozen-feature cache holding source_train (default: "
                         "<repo>/data/feature_cache; on Colab point at the Drive source cache)")
    ap.add_argument("--n-batches", type=int, default=DEFAULT_N_BATCHES,
                    help="number of fixed seeded batches to average over (default 6; 4-8)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="balanced batch size, must be even (default 256 = 128/128)")
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                    help="base seed; batch b uses base_seed+b (default 42)")
    args = ap.parse_args(argv)

    if not (4 <= args.n_batches <= 8):
        print(f"[signal_headgrad] NOTE: --n-batches={args.n_batches} is outside the paper's "
              f"4-8 range; proceeding as requested.")

    print(f"[signal_headgrad] run_dir   = {os.path.abspath(args.run_dir)}")
    print(f"[signal_headgrad] cache_dir = {os.path.abspath(args.cache_dir)}")
    print(f"[signal_headgrad] signal    = {NEW_COLUMN} (source BCE-head grad norm of the "
          f"final Linear weight; Option A)")

    out = compute_headgrad_column(args.run_dir, args.cache_dir, n_batches=args.n_batches,
                                  batch_size=args.batch_size, base_seed=args.base_seed)

    dest = os.path.join(os.path.abspath(args.run_dir), "curve_with_signals.csv")
    out.to_csv(dest, index=False)
    print(f"[signal_headgrad] wrote {NEW_COLUMN} for {len(out)} checkpoints -> {dest}")


if __name__ == "__main__":
    main()
