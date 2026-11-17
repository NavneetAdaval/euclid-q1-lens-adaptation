"""
signal_purity.py
================
LABEL-FREE SELECTION-SIGNAL TEST #1 (post-process, no encoder, runs local on BOTH
methods). For every checkpoint on the fold-0 learning curve of a run, compute
``purity @ 50% completeness`` PREVALENCE-CORRECTED to Lines' N_N/N_P = 1:2200, and
write it as a new column ``purity_50comp`` into ``curve_with_signals.csv`` (a COPY of
the run's ``curve.csv`` plus that column). No retraining, no encoder.

WHY THIS SIGNAL. TPR@1e-3 (already in curve.csv, the fold-0 ORACLE) is our selection
TARGET; purity@50%-completeness is a *same-operating-regime candidate signal* -- a
label-free-ish number computed from the SAME fold-0 score arrays -- that we test as a
stand-in for the oracle. report_signals.py (Script 3) then measures how much TPR@1e-3
you lose if you pick the checkpoint by this signal instead of the oracle. (AUPRC/AUROC
are NOT selection targets and are not touched here.) The correction to 1:2200 is Lines
et al.'s own conversion purity = TPR / (TPR + FPR*2200) -- see metrics.NN_NP_Q1.

HOW SCORES ARE OBTAINED (READ THIS -- a deliberate, flagged design choice).
The run dirs do NOT store a per-checkpoint score array: ``scores/`` holds only the ONE
final SELECTED checkpoint's per-fold arrays (e.g. baseline_linear_ep133_fold0.npz),
whereas the curve has ~50 rows (one per saved step). So "purity for every curve row"
cannot come from ``scores/`` alone. Instead this script RECOMPUTES the fold-0 logits
for each curve step from the FROZEN-FEATURE CACHE (test_pos_fold0 + test_negatives_
shared) applied through that step's checkpoint head -- byte-for-byte the same path
curve_hook.py used to fill curve.csv (evaluate.load_checkpoint + evaluate.score_
features). This is encoder-free (cached features -> head only; the Zoobot encoder is
never called) for BOTH the cached-feature baseline AND the image-space FreeMatch run,
so the script runs LOCAL on either, exactly as requested.

  SELF-CHECK (fail-loud): for every step the recomputed TPR@1e-3 is compared against
  the ``fold0_tpr_1e-3`` value already in curve.csv. If they disagree, the cache /
  checkpoints do NOT correspond to this curve and the script ABORTS -- guaranteeing the
  purity is computed on the SAME scores that produced the oracle column.

REUSE, DON'T REIMPLEMENT. Scoring = evaluate.load_checkpoint + evaluate.score_features;
the purity correction = metrics.metric_suite(...)["purity_corr"]; the TPR self-check =
metrics.tpr_at_fpr. This module owns only plumbing (which rows, which column), never math.

READ-ONLY guarantee. The only file written is ``curve_with_signals.csv`` in the run
dir. curve.csv / results.json / scores/ / checkpoints/ are never modified. Any missing
expected artifact (curve.csv, checkpoints/, a per-step checkpoint, or a cache file) is
a LOUD error naming the file -- no silent skips.

USAGE
-----
    python scripts/signal_purity.py runs/curve_baseline_fold0
    python scripts/signal_purity.py /content/drive/MyDrive/lens_paper/runs/freematch_fold0_standard
    # optional: --cache-dir <path>  (default: <repo>/data/feature_cache)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# make the sibling modules importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from evaluate import FeatureCache, load_checkpoint, score_features  # noqa: E402
from metrics import metric_suite, tpr_at_fpr, DEFAULT_TARGET_FPR, NN_NP_Q1  # noqa: E402

# the fold-0 curve is a fold-0-only diagnostic (curve_hook.py); these are the cache
# splits it scores. Kept explicit so a wrong fold can never be silently used.
POS_SPLIT = "test_pos_fold0"
NEG_SPLIT = "test_negatives_shared"
TPR_COLUMN = "fold0_tpr_1e-3"          # the oracle column we self-check against
NEW_COLUMN = "purity_50comp"           # purity@50%-completeness, corrected to 1:2200
TARGET_COMPLETENESS = 0.5              # "50% completeness" (metrics.DEFAULT_TARGET_COMPLETENESS)
TPR_SELFCHECK_ATOL = 1e-6             # recomputed vs stored fold0 TPR must agree this closely


def _die(msg: str) -> "None":
    raise SystemExit(f"[signal_purity] ERROR: {msg}")


def _require_file(path: str, what: str) -> str:
    if not os.path.isfile(path):
        _die(f"missing {what}: {path}")
    return path


def _require_dir(path: str, what: str) -> str:
    if not os.path.isdir(path):
        _die(f"missing {what}: {path}")
    return path


def _checkpoint_for_step(run_dir: str, step: int) -> str:
    """The curve row at ``step`` was written for checkpoints/step_{step}.pt (curve_hook)."""
    path = os.path.join(run_dir, "checkpoints", f"step_{int(step)}.pt")
    return _require_file(path, f"checkpoint for curve step {int(step)}")


def compute_purity_column(run_dir: str, cache_dir: str, *,
                          target_fpr: float = DEFAULT_TARGET_FPR,
                          device: str = "cpu") -> pd.DataFrame:
    """Return curve.csv as a DataFrame with the ``purity_50comp`` column appended.

    Recomputes fold-0 logits per curve step from the frozen-feature cache + that step's
    checkpoint head, self-checks the TPR against the stored oracle column, then writes
    purity@50%-completeness corrected to 1:2200 via metric_suite (no reimplementation)."""
    run_dir = os.path.abspath(run_dir)
    _require_dir(run_dir, "run dir")
    curve_csv = _require_file(os.path.join(run_dir, "curve.csv"), "curve.csv")
    _require_dir(os.path.join(run_dir, "checkpoints"), "checkpoints/ dir")
    _require_dir(cache_dir, "feature cache dir")

    cache = FeatureCache(cache_dir)
    for split in (POS_SPLIT, NEG_SPLIT):
        _require_file(cache.path(split), f"feature-cache split '{split}'")

    # Append onto an existing curve_with_signals.csv if one is there (so a sibling signal
    # column, e.g. headgrad_bce_balanced, is preserved), else start from curve.csv. Either
    # base has identical rows/steps -- we only add/replace our own column.
    signals_csv = os.path.join(run_dir, "curve_with_signals.csv")
    df = pd.read_csv(signals_csv if os.path.isfile(signals_csv) else curve_csv)
    if "step" not in df.columns:
        _die(f"curve.csv has no 'step' column (found {list(df.columns)})")
    if TPR_COLUMN not in df.columns:
        _die(f"curve.csv has no '{TPR_COLUMN}' column to self-check against "
             f"(found {list(df.columns)})")

    # frozen features loaded ONCE (immutable across checkpoints)
    pos_feat = cache.load(POS_SPLIT)["features"]
    neg_feat = cache.load(NEG_SPLIT)["features"]
    y = np.concatenate([np.ones(pos_feat.shape[0], dtype=np.int64),
                        np.zeros(neg_feat.shape[0], dtype=np.int64)])

    purities: list = []
    for _, row in df.iterrows():
        step = int(row["step"])
        ckpt = _checkpoint_for_step(run_dir, step)
        head, std, _meta = load_checkpoint(ckpt, device)
        ps = score_features(head, std, pos_feat, device)
        ns = score_features(head, std, neg_feat, device)
        s = np.concatenate([ps, ns])

        # -- self-check: recomputed TPR@1e-3 must match the stored oracle column --
        tpr_now = tpr_at_fpr(y, s, target_fpr)
        tpr_ref = row[TPR_COLUMN]
        if pd.notna(tpr_ref) and np.isfinite(tpr_now):
            if not np.isclose(tpr_now, float(tpr_ref), atol=TPR_SELFCHECK_ATOL, rtol=0.0):
                _die(f"step {step}: recomputed fold0 TPR@{target_fpr:g}={tpr_now:.6f} does "
                     f"NOT match curve.csv {TPR_COLUMN}={float(tpr_ref):.6f}. The feature "
                     f"cache / checkpoints do not correspond to this curve -- refusing to "
                     f"write a purity computed on the wrong scores.")

        # -- the signal: purity@50%-completeness corrected to 1:2200 (metric_suite) --
        suite = metric_suite(y, s, target_fpr=target_fpr,
                             target_completeness=TARGET_COMPLETENESS,
                             nn_np_ratio=NN_NP_Q1)
        purities.append(suite["purity_corr"])
        print(f"  step {step:>7}: fold0_tpr={tpr_now:.4f}  {NEW_COLUMN}={suite['purity_corr']:.4f}")

    out = df.copy()
    out[NEW_COLUMN] = purities
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run dir containing curve.csv + checkpoints/ (baseline "
                                    "local, or a Drive-mounted FreeMatch run dir)")
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(_HERE),
                                                        "data", "feature_cache"),
                    help="frozen-feature cache dir (default: <repo>/data/feature_cache)")
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR,
                    help="operating FPR for the TPR self-check (default 1e-3)")
    args = ap.parse_args(argv)

    print(f"[signal_purity] run_dir   = {os.path.abspath(args.run_dir)}")
    print(f"[signal_purity] cache_dir = {os.path.abspath(args.cache_dir)}")
    print(f"[signal_purity] signal    = {NEW_COLUMN} "
          f"(purity@{int(TARGET_COMPLETENESS*100)}%-completeness, corrected to 1:{int(NN_NP_Q1)})")

    out = compute_purity_column(args.run_dir, args.cache_dir, target_fpr=args.target_fpr)

    dest = os.path.join(os.path.abspath(args.run_dir), "curve_with_signals.csv")
    out.to_csv(dest, index=False)
    print(f"[signal_purity] wrote {NEW_COLUMN} for {len(out)} checkpoints -> {dest}")


if __name__ == "__main__":
    main()
