"""
curve_hook.py
=============
Shared checkpoint-and-eval hook for the fold-0 LEARNING-CURVE diagnostic (PROJECT.md
7.4 "locate where the real-lens metric flattens"). It is invoked by run_baseline.py and
run_freematch.py ONLY when their ``--eval-every-steps N`` flag is set, so default runs are
byte-for-byte unchanged.

WHAT IT DOES (per trigger, every N optimizer steps)
---------------------------------------------------
Given a checkpoint that the runner just saved (``checkpoints/step_{step}.pt``), it appends
one row to ``curve.csv`` in the run dir with EXACTLY these columns:

    step, source_risk_auroc, source_risk_tpr, fold0_tpr_1e-3, fold0_auprc, fold0_auroc

  * source_risk_auroc / source_risk_tpr -- the LABEL-FREE selector, logged BOTH ways:
                      source-val AUROC and source-val TPR@1e-3 on held-out SIM. We log both
                      and decide which is the trustworthy stopping signal FROM the curves
                      (source-val saturates on sims -- PROJECT.md 7.4), rather than fixing
                      the selector now. Computed on ``val.npz`` ONLY -- never the target pool
                      or the fold-0 real-lens labels (PROJECT.md 4.2).
  * fold0_tpr_1e-3 -- DIAGNOSTIC-ONLY real metric: held-out fold-0 real-lens positives vs
                      the SHARED negative pool, via the SAME metrics.tpr_at_fpr the
                      evaluate.py/Candidate path uses.
  * fold0_auroc    -- ditto, via metrics.auroc.
  * fold0_auprc    -- via metrics.metric_suite (no metric is reimplemented here).

FOLD-0 BY DEFAULT (cost). By default the hook touches ``val`` + ``test_pos_fold0`` +
``test_negatives_shared`` and nothing else -- not the other four folds, not the guardrail
target pools -- so the per-trigger overhead stays tiny (feature-space CPU) even for the
expensive image-space FreeMatch run (PROJECT.md 7.4 "keep the eval on fold-0 only").

ALL-FOLD MODE (``folds=[0,1,2,3,4]``; the matched 5-fold bake-off, SPEC 3/4). Baseline A
is SOURCE-ONLY, so one training run is evaluated against all five folds and the per-fold
flatness re-check needs a per-fold TPR column, not just fold 0. Passing ``folds`` widens
the schema to ``fold{k}_tpr_1e-3 / fold{k}_auprc / fold{k}_auroc`` for each k. Cost stays
near-flat: the ~26k SHARED negatives are scored ONCE per checkpoint and reused by every
fold (only the ~98 extra positives per fold are new work). ``folds=[0]`` reproduces the
legacy schema EXACTLY, so existing runs/readers (run_dann.py, plot_curves.py) are
unaffected.

REUSE, DON'T REIMPLEMENT. Scoring is evaluate.load_checkpoint + evaluate.score_features;
every metric is a metrics.py function. This module owns only the plumbing (which splits to
read, which columns to write), never the math.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from evaluate import FeatureCache, load_checkpoint, score_features
from metrics import tpr_at_fpr, auroc, metric_suite, DEFAULT_TARGET_FPR
from analyze_fold0_selection import _spearman   # the ONE Spearman impl (reused, not re-written)

# the curve.csv schema -- fixed, in this order (the runners write nothing else).
# BOTH source-val selectors are logged; the selector is chosen from the curves, not now.
SOURCE_RISK_COLUMNS = ["step", "source_risk_auroc", "source_risk_tpr"]

# the DEFAULT fold set: fold 0 only (legacy behavior, unchanged for every existing caller).
DEFAULT_FOLDS = (0,)


def fold_columns(k: int) -> list[str]:
    """The three per-fold metric columns for fold ``k``, in schema order."""
    return [f"fold{k}_tpr_1e-3", f"fold{k}_auprc", f"fold{k}_auroc"]


def agreement_columns(agreement_folds=()) -> list[str]:
    """The collapse-gate agreement column(s), one per fold in ``agreement_folds``.

    Empty by default -> the legacy schema is unchanged. The matched CORAL bake-off
    (SPEC spec_bakeoff_2_coral.md §4) passes the trained fold k so ``agreement_fold{k}``
    is appended AFTER the metric columns (never interleaved, so existing readers that
    key by name are unaffected)."""
    return [f"agreement_fold{int(k)}" for k in agreement_folds]


def curve_columns(folds=DEFAULT_FOLDS, agreement_folds=()) -> list[str]:
    """The full curve.csv schema for a given fold set, in order.

    ``curve_columns([0])`` is byte-identical to the legacy CURVE_COLUMNS below -- that
    identity is what keeps run_dann.py / plot_curves.py / existing curve.csv files valid.
    Any ``agreement_fold{k}`` columns are appended last, so the legacy prefix is intact.
    """
    cols = list(SOURCE_RISK_COLUMNS)
    for k in folds:
        cols += fold_columns(int(k))
    cols += agreement_columns(agreement_folds)
    return cols


# legacy constant -- fold-0 schema. Still the exact schema run_dann.py writes and
# test_run_dann.py pins; do NOT widen it in place.
CURVE_COLUMNS = curve_columns(DEFAULT_FOLDS)

# memoize the (immutable, frozen-feature) splits so a 50-point baseline curve does not
# re-read the ~67 MB shared-negative cache from disk 50 times. Keyed by (cache_dir, name).
_ARRCACHE: dict = {}


def _cached_split(cache: FeatureCache, name: str) -> dict:
    key = (cache.dir, name)
    if key not in _ARRCACHE:
        _ARRCACHE[key] = cache.load(name)
    return _ARRCACHE[key]


def eval_checkpoint_folds(ckpt_path, cache: FeatureCache, *, folds=DEFAULT_FOLDS,
                          target_fpr: float = DEFAULT_TARGET_FPR,
                          val_select_metric: str = "auroc", device: str = "cpu",
                          agreement_refs=None, split_loader=None,
                          collect_scores=None) -> dict:
    """Score one saved checkpoint on source-val (both selectors) + each fold's test set.

    Reads ONLY val + test_pos_fold{k} (for k in ``folds``) + test_negatives_shared. The
    SHARED negatives are scored ONCE and reused across folds -- which is both the cheap
    path and the correct one: PROJECT.md 4.4 requires the FPR=1e-3 threshold to come from
    the same negative scores for every fold, so re-scoring per fold would be wasted work
    that must produce identical numbers anyway.

    ``val_select_metric`` is accepted for caller-API symmetry but does NOT gate the curve:
    BOTH source-val AUROC and source-val TPR@1e-3 are always logged (selector chosen later).

    ``agreement_refs`` (dict {fold: reference_scores_on_target_fold}) turns on the
    collapse-gate agreement column (SPEC spec_bakeoff_2_coral.md §4): for each fold k it
    scores THIS checkpoint on ``target_fold{k}`` (the fold's UNLABELED ~1:1 pool) and logs
    ``agreement_fold{k}`` = Spearman(these scores, the passed Baseline A reference scores).
    Reference scores are precomputed ONCE by the caller (Baseline A step_8000 on the same
    pool) and passed in -- read-only, never fed back into any loss.

    ``split_loader`` (default None -> read the frozen feature cache) is the ONE hook that
    makes this usable for a PARTIALLY UNFROZEN encoder. The cache is valid only while the
    encoder is frozen (cache_features.py raises at n_blocks>0), so an unfrozen run passes a
    callable ``name -> {"features","labels",...}`` that encodes the split LIVE with the
    current encoder (live_encode.LiveEncoder). Everything else -- which splits are read,
    which columns are written, and every metric -- is unchanged, so the curve.csv schema
    stays byte-identical and analyze_fold0_selection.py / collapse_gate.py consume it
    without modification.

    ``collect_scores`` (default None) is an OUT-parameter: pass a dict and it is filled with
    ``{split_name: head-logit vector}`` for every split this call already scored (val,
    test_negatives_shared, test_pos_fold{k}, target_fold{k}), plus ``{split}__ids``. It
    exists so the DERIVED-LOGIT DUMP (BRIEF 5.2, the fix for PROJECT.md 7.7's 38 GB scaling
    flag) is free: the scores are a by-product of the row this hook computes anyway, so
    dumping them costs no extra encoder forward. It is READ-ONLY plumbing -- no metric, no
    column and no returned value changes when it is passed."""
    load = split_loader if split_loader is not None else (lambda n: _cached_split(cache, n))
    head, std, _meta = load_checkpoint(ckpt_path, device)

    def _collect(name, split, scores):
        if collect_scores is None:
            return
        collect_scores[name] = np.asarray(scores, dtype=np.float32)
        ids = split.get("ids")
        if ids is not None:
            collect_scores[f"{name}__ids"] = np.asarray(ids)

    # ---- source_risk: label-free selector, source-domain SIM val ONLY (log BOTH) ----
    v = load("val")
    vs = score_features(head, std, v["features"], device)
    _collect("val", v, vs)
    out = {"source_risk_auroc": _num(auroc(v["labels"], vs)),
           "source_risk_tpr": _num(tpr_at_fpr(v["labels"], vs, target_fpr))}

    # ---- shared negatives: scored ONCE, reused by every fold (identical threshold) ----
    neg = load("test_negatives_shared")
    ns = score_features(head, std, neg["features"], device)
    _collect("test_negatives_shared", neg, ns)
    yneg = np.zeros(ns.size, dtype=np.int64)

    # ---- per-fold diagnostic: that fold's held-out real positives vs the shared negatives ----
    for k in folds:
        pos = load(f"test_pos_fold{int(k)}")
        ps = score_features(head, std, pos["features"], device)
        _collect(f"test_pos_fold{int(k)}", pos, ps)
        y = np.concatenate([np.ones(ps.size, dtype=np.int64), yneg])
        s = np.concatenate([ps, ns])

        tpr = tpr_at_fpr(y, s, target_fpr)                      # evaluate.py/Candidate path
        au = auroc(y, s)                                        # evaluate.py/Candidate path
        ap = metric_suite(y, s, target_fpr=target_fpr)["auprc"]  # metric_suite (no reimpl)

        c_tpr, c_ap, c_au = fold_columns(int(k))
        out[c_tpr], out[c_ap], out[c_au] = _num(tpr), _num(ap), _num(au)

    # ---- collapse-gate agreement column (label-free; read-only) ----
    # Spearman(this ckpt's scores on target_fold{k}, Baseline A's scores on the same
    # pool). Reuses the SAME score_features path above and the ONE _spearman helper.
    if agreement_refs:
        for k, ref in agreement_refs.items():
            tgt = load(f"target_fold{int(k)}")
            ts = score_features(head, std, tgt["features"], device)
            _collect(f"target_fold{int(k)}", tgt, ts)
            rho, _n = _spearman(ts, np.asarray(ref, dtype=float))
            out[f"agreement_fold{int(k)}"] = _num(rho)
    elif collect_scores is not None:
        # No agreement column requested, but the logit dump still wants the target pool --
        # it is the split the memorization probe (BRIEF 5.4) and the post-hoc pseudo-label
        # precision (BRIEF 5.6) are computed on. Scored here rather than skipped, so the
        # dump is complete regardless of whether the gate column is on.
        for k in folds:
            tgt = load(f"target_fold{int(k)}")
            _collect(f"target_fold{int(k)}", tgt,
                     score_features(head, std, tgt["features"], device))
    return out


def eval_checkpoint_fold0(ckpt_path, cache: FeatureCache, *,
                          target_fpr: float = DEFAULT_TARGET_FPR,
                          val_select_metric: str = "auroc", device: str = "cpu") -> dict:
    """Legacy fold-0-only entry point -- ``eval_checkpoint_folds`` with folds=[0]. Kept so
    existing callers keep the exact five-column result they were written against."""
    return eval_checkpoint_folds(ckpt_path, cache, folds=DEFAULT_FOLDS,
                                 target_fpr=target_fpr,
                                 val_select_metric=val_select_metric, device=device)


def append_curve_row(curve_csv, step, ckpt_path, cache_dir, *, folds=DEFAULT_FOLDS,
                     target_fpr: float = DEFAULT_TARGET_FPR,
                     val_select_metric: str = "auroc", device: str = "cpu",
                     agreement_refs=None, split_loader=None,
                     collect_scores=None) -> dict:
    """Compute the curve row for ``step`` and append it to ``curve_csv`` (writing the header
    first if the file is new). Returns the row dict.

    ``folds`` defaults to (0,) -- the legacy fold-0 schema, so existing callers are
    unchanged. Pass folds=[0,1,2,3,4] for the 5-fold bake-off's per-fold curve.

    ``agreement_refs`` (dict {fold: reference_scores}) appends the ``agreement_fold{k}``
    collapse-gate column(s) for those folds -- see ``eval_checkpoint_folds``."""
    cache = FeatureCache(cache_dir)
    m = eval_checkpoint_folds(ckpt_path, cache, folds=folds, target_fpr=target_fpr,
                              val_select_metric=val_select_metric, device=device,
                              agreement_refs=agreement_refs, split_loader=split_loader,
                              collect_scores=collect_scores)
    row = {"step": int(step), **m}
    agreement_folds = sorted(agreement_refs) if agreement_refs else ()
    cols = curve_columns(folds, agreement_folds=agreement_folds)
    os.makedirs(os.path.dirname(os.path.abspath(curve_csv)), exist_ok=True)
    write_header = not os.path.exists(curve_csv)
    with open(curve_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow({k: row[k] for k in cols})
    return row


def _num(x):
    """None for non-finite so the CSV cell is blank rather than 'nan' garbage."""
    return None if (x is None or not np.isfinite(x)) else float(x)
