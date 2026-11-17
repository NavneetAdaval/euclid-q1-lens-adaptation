"""
evaluate.py
===========
Evaluation harness for the label-free sim-to-real strong-lens pipeline (Euclid Q1).
It scores a trained head + standardizer against the REAL test set (reading the frozen
feature cache written by cache_features.py -- the encoder is never called here), and
produces the reportable metrics with uncertainty.

READS CACHED FEATURES ONLY (RB). The Zoobot encoder is frozen, so its features never
change; cache_features.py encoded them once (the single GPU step). evaluate.py loads
those .npz files and runs entirely on CPU/4050.

WHAT IT REPORTS (PROJECT.md 2 + 4.4/4.5)
----------------------------------------
  * Per-fold TPR@FPR=1e-3 and AUROC (each fold's held-out real-lens positives against
    the SHARED negative pool), plus the across-fold mean.
  * A pooled aggregate: all 489 test positives pooled across folds against the shared
    negatives (the negatives are shared, so pooling is legitimate -- PROJECT.md 4.4),
    with class-conditional paired-bootstrap CIs on both TPR@1e-3 and AUROC.
  * Raw per-fold scores dumped to disk (with stable ids) for later PAIRED method-vs-
    method comparisons in the bake-off.

MODEL SELECTION -- two columns + one guardrail, clearly labeled
---------------------------------------------------------------
  (a) SOURCE-RISK  (headline, honest label-free): pick the checkpoint/epoch by the best
      SOURCE-val metric (val is labeled held-out SIM -- the only label-free-legal
      selector; PROJECT.md 4.2), then report THAT checkpoint's test metrics.
  (b) ORACLE       (upper bound, NOT a valid UDA result): pick the checkpoint by the
      best real-TEST metric and report it. Printed only to expose the selection gap;
      it peeks at test labels and is labeled as invalid.
  (c) ALERT-RATE GUARDRAIL (logged, never a selector): the fraction of the UNLABELED
      target pool clearing the FPR=1e-3 threshold. Flags checkpoints where it explodes
      far above the ~0.002 lens prevalence (the "everything fires" collapse). SND is
      deliberately NOT implemented -- it is inapplicable under this extreme imbalance
      (a target-density selector would just measure the 99.95% negative majority).

SCOPE / FLAG (honest limitation)
--------------------------------
This harness evaluates ONE checkpoint across ALL folds -- which is exactly what makes
PROJECT.md 4.4's "threshold identical across folds" property hold (a single model scores
the shared negatives once) and matches the request's singular "a trained head +
standardizer checkpoint". Multiple checkpoints supplied on the CLI are treated as
model-SELECTION candidates (epochs), each evaluated across all folds; SOURCE-RISK /
ORACLE pick among them. If the DA bake-off ever yields genuinely PER-FOLD models (a
different head per fold), evaluate each with its own fold's positives and report the
across-fold mean +- CI rather than the single pooled ROC (whose shared-negative pooling
assumes one model). That extension is intentionally left for the bake-off.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

from heads import build_head, FeatureStandardizer, EMBED_DIM
from metrics import (auroc, tpr_at_fpr, operating_threshold, paired_bootstrap,
                     alert_rate_guardrail, default_tpr_metric, LENS_PREVALENCE,
                     DEFAULT_TARGET_FPR)


# ---------------------------------------------------------------------------
# feature cache access
# ---------------------------------------------------------------------------

class FeatureCache:
    """Thin reader over the .npz files written by cache_features.py."""

    def __init__(self, cache_dir: str):
        self.dir = cache_dir

    def path(self, name: str) -> str:
        return os.path.join(self.dir, f"{name}.npz")

    def has(self, name: str) -> bool:
        return os.path.exists(self.path(name))

    def load(self, name: str) -> dict:
        with np.load(self.path(name), allow_pickle=False) as d:
            out = {k: d[k] for k in d.files if k != "meta"}
        return out

    def folds(self) -> list[int]:
        ks = []
        for p in glob.glob(os.path.join(self.dir, "test_pos_fold*.npz")):
            m = re.search(r"test_pos_fold(\d+)\.npz$", os.path.basename(p))
            if m:
                ks.append(int(m.group(1)))
        return sorted(ks)


# ---------------------------------------------------------------------------
# checkpoint loading + scoring
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: str = "cpu"):
    """Reconstruct (head, standardizer, meta) from a train_skeleton checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head_type = ckpt.get("head_type", "linear")
    head = build_head(head_type)
    head.load_state_dict(ckpt["head"])
    head.eval().to(device)
    std = FeatureStandardizer(EMBED_DIM)
    std.load_state_dict(ckpt["standardizer"])
    std.eval().to(device)
    cfg = ckpt.get("cfg", {}) or {}
    meta = {"path": path, "head_type": head_type,
            "fold": cfg.get("fold"), "epoch": _epoch_from_name(path)}
    return head, std, meta


def _epoch_from_name(path: str):
    m = re.search(r"(?:ep|epoch)[_-]?(\d+)", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else None


@torch.no_grad()
def score_features(head, std, feats: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return the head LOGIT per row (monotonic in probability -> valid ranking score
    for TPR@FPR / AUROC, and threshold-comparable to the guardrail's logit threshold)."""
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(device)
    logits = head(std(x)).view(-1)
    return logits.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# a single evaluated candidate (one checkpoint, across all cached folds)
# ---------------------------------------------------------------------------

class Candidate:
    """One checkpoint evaluated across every cached test fold + val + targets."""

    def __init__(self, path, cache: FeatureCache, device="cpu",
                 target_fpr=DEFAULT_TARGET_FPR, val_select_metric="auroc"):
        self.path = path
        self.target_fpr = target_fpr
        self.val_select_metric = val_select_metric
        head, std, meta = load_checkpoint(path, device)
        self.meta = meta
        self.tag = os.path.splitext(os.path.basename(path))[0]

        # shared negatives scored ONCE by this checkpoint
        neg = cache.load("test_negatives_shared")
        self.neg_ids = neg["ids"]
        self.neg_scores = score_features(head, std, neg["features"], device)

        # per-fold positives
        self.folds = cache.folds()
        self.pos_scores = {}
        self.pos_ids = {}
        for k in self.folds:
            p = cache.load(f"test_pos_fold{k}")
            self.pos_ids[k] = p["ids"]
            self.pos_scores[k] = score_features(head, std, p["features"], device)

        # source-val metric (the label-free selector). May be absent in a tiny cache.
        self.val_metric = np.nan
        if cache.has("val"):
            v = cache.load("val")
            vs = score_features(head, std, v["features"], device)
            self.val_metric = (auroc(v["labels"], vs) if val_select_metric == "auroc"
                               else tpr_at_fpr(v["labels"], vs, target_fpr))

        # per-fold + pooled test metrics (this checkpoint scoring the shared negatives)
        self._compute_test_metrics()

        # unlabeled-target alert-rate guardrail (uses the pooled operating threshold)
        self._compute_guardrail(cache, head, std, device)

    # -- metrics -----------------------------------------------------------

    def _compute_test_metrics(self):
        neg = self.neg_scores
        yneg = np.zeros(neg.size, dtype=np.int64)
        self.per_fold = {}
        pooled_pos = []
        for k in self.folds:
            ps = self.pos_scores[k]
            y = np.concatenate([np.ones(ps.size, dtype=np.int64), yneg])
            s = np.concatenate([ps, neg])
            self.per_fold[k] = {
                "n_pos": int(ps.size),
                "tpr": tpr_at_fpr(y, s, self.target_fpr),
                "auroc": auroc(y, s),
            }
            pooled_pos.append(ps)
        # pooled aggregate: all folds' positives vs the shared negatives
        if pooled_pos:
            ppos = np.concatenate(pooled_pos)
            y = np.concatenate([np.ones(ppos.size, dtype=np.int64), yneg])
            s = np.concatenate([ppos, neg])
            self.pooled_y = y
            self.pooled_s = s
            self.pooled_pos_scores = ppos
            self.agg_tpr = tpr_at_fpr(y, s, self.target_fpr)
            self.agg_auroc = auroc(y, s)
            self.agg_threshold = operating_threshold(y, s, self.target_fpr)
        else:
            self.pooled_y = self.pooled_s = self.pooled_pos_scores = None
            self.agg_tpr = self.agg_auroc = self.agg_threshold = np.nan

        vals_tpr = [self.per_fold[k]["tpr"] for k in self.folds]
        vals_au = [self.per_fold[k]["auroc"] for k in self.folds]
        self.mean_tpr = float(np.nanmean(vals_tpr)) if vals_tpr else np.nan
        self.std_tpr = float(np.nanstd(vals_tpr)) if vals_tpr else np.nan
        self.mean_auroc = float(np.nanmean(vals_au)) if vals_au else np.nan
        self.std_auroc = float(np.nanstd(vals_au)) if vals_au else np.nan

    def _compute_guardrail(self, cache, head, std, device):
        self.guardrail = {}
        thr = self.agg_threshold
        rates = []
        for k in self.folds:
            name = f"target_fold{k}"
            if not cache.has(name):
                continue
            t = cache.load(name)
            ts = score_features(head, std, t["features"], device)
            g = alert_rate_guardrail(ts, thr, prevalence=LENS_PREVALENCE)
            self.guardrail[k] = g
            if np.isfinite(g["alert_rate"]):
                rates.append(g["alert_rate"])
        self.mean_alert_rate = float(np.nanmean(rates)) if rates else np.nan
        self.any_fired = any(g["fired"] for g in self.guardrail.values())

    # -- bootstrap CIs on the pooled aggregate -----------------------------

    def aggregate_cis(self, B=1000, seed=0):
        if self.pooled_y is None:
            return {}
        tpr_ci = paired_bootstrap(self.pooled_y, self.pooled_s,
                                  metric_fn=default_tpr_metric(self.target_fpr),
                                  B=B, seed=seed)
        au_ci = paired_bootstrap(self.pooled_y, self.pooled_s,
                                 metric_fn=auroc, B=B, seed=seed)
        return {"tpr": tpr_ci, "auroc": au_ci}

    # -- raw score dump (for the bake-off's paired comparisons) ------------

    def dump_scores(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for k in self.folds:
            np.savez(os.path.join(out_dir, f"{self.tag}_fold{k}.npz"),
                     pos_ids=self.pos_ids[k], pos_scores=self.pos_scores[k],
                     neg_ids=self.neg_ids, neg_scores=self.neg_scores,
                     target_fpr=np.asarray(self.target_fpr))


# ---------------------------------------------------------------------------
# selection + reporting
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


def select_and_report(candidates, target_fpr=DEFAULT_TARGET_FPR, B=1000, seed=0,
                      out_dir=None):
    """Pick the SOURCE-RISK and ORACLE checkpoints among candidates and print the
    reportable table. Returns a results dict (also written to out_dir/results.json)."""
    if not candidates:
        raise ValueError("no candidates to evaluate")

    # (a) SOURCE-RISK: best source-val metric (label-free-legal selector)
    have_val = any(np.isfinite(c.val_metric) for c in candidates)
    if have_val:
        source_risk = max(candidates, key=lambda c: (np.nan_to_num(c.val_metric, nan=-np.inf)))
    else:
        source_risk = candidates[0]   # only one epoch / no val cache -> nothing to select
    # (b) ORACLE: best real-test aggregate metric (peeks at labels -> NOT a valid UDA result)
    oracle = max(candidates, key=lambda c: (np.nan_to_num(c.agg_tpr, nan=-np.inf),
                                            np.nan_to_num(c.agg_auroc, nan=-np.inf)))

    folds = source_risk.folds
    print("\n" + "=" * 78)
    print(f"EVALUATION  (metric: TPR@FPR={target_fpr:g} + AUROC;  "
          f"{len(candidates)} candidate checkpoint(s))")
    print("=" * 78)

    def report_candidate(c, label, note=""):
        cis = c.aggregate_cis(B=B, seed=seed)
        print(f"\n[{label}]  {os.path.basename(c.path)}"
              f"   (val_{c.val_select_metric}={_fmt(c.val_metric)}){note}")
        print("  per-fold:")
        hdr = "    fold |  n_pos |  TPR@fpr |   AUROC"
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for k in folds:
            pf = c.per_fold[k]
            print(f"    {k:>4} | {pf['n_pos']:>6} | {_fmt(pf['tpr']):>8} | {_fmt(pf['auroc']):>7}")
        print(f"    across-fold mean   TPR = {_fmt(c.mean_tpr)} +- {_fmt(c.std_tpr)}"
              f"    AUROC = {_fmt(c.mean_auroc)} +- {_fmt(c.std_auroc)}")
        if cis:
            t, a = cis["tpr"], cis["auroc"]
            print(f"    POOLED (489 pos vs shared negs):")
            print(f"      TPR@{target_fpr:g} = {_fmt(c.agg_tpr)}  "
                  f"[95% CI {_fmt(t['lo'])}, {_fmt(t['hi'])}]  (median {_fmt(t['median'])})")
            print(f"      AUROC      = {_fmt(c.agg_auroc)}  "
                  f"[95% CI {_fmt(a['lo'])}, {_fmt(a['hi'])}]  (median {_fmt(a['median'])})")
        # guardrail
        gl = "FIRED" if c.any_fired else "ok"
        print(f"    ALERT-RATE GUARDRAIL: mean target alert rate = {_fmt(c.mean_alert_rate)}"
              f"  (prevalence ~{LENS_PREVALENCE}) -> {gl}")
        return cis

    sr_cis = report_candidate(
        source_risk, "SOURCE-RISK  (headline, label-free selection)")
    print("\n" + "-" * 78)
    or_cis = report_candidate(
        oracle, "ORACLE  (upper bound -- peeks at test labels; NOT a valid UDA result)",
        note="")

    # selection gap (oracle - source-risk), on the pooled aggregate
    gap_tpr = (oracle.agg_tpr - source_risk.agg_tpr
               if np.isfinite(oracle.agg_tpr) and np.isfinite(source_risk.agg_tpr) else np.nan)
    gap_au = (oracle.agg_auroc - source_risk.agg_auroc
              if np.isfinite(oracle.agg_auroc) and np.isfinite(source_risk.agg_auroc) else np.nan)
    same = os.path.abspath(oracle.path) == os.path.abspath(source_risk.path)
    print("\n" + "=" * 78)
    print(f"SELECTION GAP (oracle - source-risk):  TPR {_fmt(gap_tpr)}   AUROC {_fmt(gap_au)}"
          + ("   [same checkpoint selected]" if same else ""))
    print("=" * 78)

    results = {
        "target_fpr": target_fpr, "folds": folds,
        "source_risk": _candidate_summary(source_risk, sr_cis),
        "oracle": _candidate_summary(oracle, or_cis),
        "selection_gap": {"tpr": _num(gap_tpr), "auroc": _num(gap_au),
                          "same_checkpoint": same},
        "n_candidates": len(candidates),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        # dump raw per-fold scores for the SOURCE-RISK (headline) checkpoint
        source_risk.dump_scores(os.path.join(out_dir, "scores"))
        print(f"\n[out] results.json + raw per-fold scores -> {out_dir}")
    return results


def _num(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _candidate_summary(c, cis):
    out = {
        "checkpoint": c.path, "val_metric": _num(c.val_metric),
        "per_fold": {int(k): {"n_pos": c.per_fold[k]["n_pos"],
                              "tpr": _num(c.per_fold[k]["tpr"]),
                              "auroc": _num(c.per_fold[k]["auroc"])}
                     for k in c.folds},
        "mean_tpr": _num(c.mean_tpr), "std_tpr": _num(c.std_tpr),
        "mean_auroc": _num(c.mean_auroc), "std_auroc": _num(c.std_auroc),
        "pooled_tpr": _num(c.agg_tpr), "pooled_auroc": _num(c.agg_auroc),
        "operating_threshold": _num(c.agg_threshold),
        "mean_alert_rate": _num(c.mean_alert_rate), "guardrail_fired": bool(c.any_fired),
    }
    if cis:
        out["pooled_tpr_ci"] = [_num(cis["tpr"]["lo"]), _num(cis["tpr"]["hi"])]
        out["pooled_auroc_ci"] = [_num(cis["auroc"]["lo"]), _num(cis["auroc"]["hi"])]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def evaluate(checkpoints, cache_dir, device="cpu", target_fpr=DEFAULT_TARGET_FPR,
             B=1000, seed=0, out_dir=None, val_select_metric="auroc"):
    cache = FeatureCache(cache_dir)
    if not cache.has("test_negatives_shared") or not cache.folds():
        raise FileNotFoundError(
            f"feature cache at {cache_dir} is missing test_negatives_shared / "
            f"test_pos_fold*.npz; run cache_features.py first.")
    candidates = [Candidate(p, cache, device=device, target_fpr=target_fpr,
                            val_select_metric=val_select_metric)
                  for p in checkpoints]
    return select_and_report(candidates, target_fpr=target_fpr, B=B, seed=seed,
                             out_dir=out_dir)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.normpath(os.path.join(here, "..", "data"))
    p = argparse.ArgumentParser(description="Evaluate a trained head against the test set.")
    p.add_argument("checkpoints", nargs="+",
                   help="one or more head+standardizer checkpoints (epochs = selection "
                        "candidates)")
    p.add_argument("--cache-dir", default=os.path.join(default_root, "feature_cache"))
    p.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-select-metric", default="auroc", choices=("auroc", "tpr"))
    p.add_argument("--out-dir", default=None, help="write results.json + raw scores here")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    missing = [c for c in args.checkpoints if not os.path.exists(c)]
    if missing:
        print(f"SKIP: checkpoint(s) not found: {missing}")
        return 3
    if not os.path.isdir(args.cache_dir):
        print(f"SKIP: feature cache dir not found: {args.cache_dir}")
        return 3

    evaluate(args.checkpoints, args.cache_dir, device=device,
             target_fpr=args.target_fpr, B=args.bootstrap, seed=args.seed,
             out_dir=args.out_dir, val_select_metric=args.val_select_metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
