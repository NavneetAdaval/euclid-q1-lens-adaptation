"""
compare_methods.py
==================
PAIRED statistical comparison between two ALREADY-RUN methods (e.g. Baseline A vs
Deep CORAL), reading ONLY the raw per-fold scores each runner dumped to its
``runs/<method>/scores/`` directory. It RETRAINS NOTHING -- it is a pure post-hoc
significance test on saved score vectors.

WHY A DEDICATED PAIRED TEST (and not just "0.21 > 0.18")
-------------------------------------------------------
Comparing two headline point estimates is not a result. Two methods evaluated on the
SAME test positives and the SAME shared negatives are PAIRED -- the honest question is
the *signed difference* per fold and whether its bootstrap CI excludes 0. Pairing
cancels the shared fold-to-fold and negative-tail noise, so a paired test is far more
sensitive than comparing two independent CIs (which can overlap even when every fold
improves).

THE STRUCTURAL MISMATCH, HANDLED HONESTLY
-----------------------------------------
Baseline A is ONE model scored across all folds; Deep CORAL is a PER-FOLD model (a
different model per fold). There is therefore no single pooled ROC that is valid for
both. So we compare STRICTLY PER FOLD:

  * fold k: baseline's scores on fold k   vs   CORAL's model-k scores on fold k,
    using the SAME fold-k test positives and the SAME shared negative pool.

The negatives are the shared pool, dumped with stable ``neg_ids`` by both runners. We
ASSERT the two runs' ``neg_ids`` match exactly, per fold -- if they don't, we FAIL
LOUDLY rather than silently compare misaligned rows. (Positives come from the same
cached ``test_pos_fold{k}`` in identical row order for both runs, so they are aligned by
construction; where both runs dumped ``pos_ids`` we assert those too.)

THE PAIRED BOOTSTRAP (aggregate)
--------------------------------
Class-conditional resampling (positives and negatives resampled independently, WITH
replacement), and -- critically -- the SAME resampled indices are applied to BOTH
methods on each replicate (paired). The shared negative pool is resampled ONCE per
replicate and reused across folds (it is literally one pool shared by every fold, so
resampling it independently per fold would fake 5 independent negative draws). Each
fold's positives are resampled independently. The metric is averaged across folds per
replicate; we report the 95% CI of the mean signed difference (CORAL - baseline).

  * VERDICT: CI excludes 0  -> "distinguishable" (improvement if +, regression if -).
             CI straddles 0  -> "not distinguishable".

metrics.paired_bootstrap is REUSED verbatim for the PER-FOLD difference CIs (its
paired-difference mode is exactly this test on one fold). The ACROSS-FOLD aggregate
needs the extra cross-fold averaging, so it uses the identical class-conditional paired
scheme extended over folds (there is no pooling -- pooling per-fold-model scores under a
single threshold is the invalid thing we are avoiding).

THE REPORTED METRIC SUITE (field-standard low-FPR completeness+purity family)
----------------------------------------------------------------------------
For each method and the paired difference (with paired-bootstrap CIs), per fold and in
aggregate:
  * TPR@1e-3        -- HEADLINE operating metric (Lines' TPR@0.1%; PROJECT.md 2).
  * purity@50%-comp -- purity (=precision) at 50% completeness, reported BOTH raw-pool
                       (internal) AND prevalence-corrected to 1:2200 (Lines-comparable).
  * AUPRC           -- threshold-free HEADLINE summary (replaces AUROC in that role).
  * completeness@fixed-purity -- the dual (at Lines' ~24% purity reference point).
  * AUROC           -- kept, but SECONDARY / sanity only (over-optimistic under skew).
Per metric we state whether the paired-difference CI excludes 0. The verdict LEADS with
TPR@1e-3 and purity, not AUROC.

Run:
    python scripts\\compare_methods.py runs\\baseline_A runs\\coral --out-dir runs\\compare
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

from metrics import (metric_suite, DEFAULT_TARGET_FPR, DEFAULT_TARGET_COMPLETENESS,
                     DEFAULT_TARGET_PURITY, NN_NP_Q1, LINES_SOURCE_ONLY_COMPLETENESS,
                     LINES_SOURCE_ONLY_PURITY)


# ---------------------------------------------------------------------------
# the reported metric family (ordered; leads with TPR + purity, AUROC last/secondary)
# ---------------------------------------------------------------------------
# key -> (label, role). The keys match metrics.metric_suite's output. HEADLINE metrics
# lead the verdict; AUROC is kept but explicitly labeled secondary/sanity (PROJECT.md 2:
# TPR@1e-3 is the operating metric, AUPRC the threshold-free headline, AUROC over-
# optimistic under extreme skew).
SUITE_KEYS = ("tpr", "purity_corr", "purity_raw", "auprc", "comp_at_purity", "auroc")
HEADLINE_KEYS = ("tpr", "purity_corr", "auprc")


def _metric_labels(target_fpr, target_completeness, nn_np_ratio, target_purity):
    tc = f"{target_completeness * 100:g}%"
    r = f"1:{int(round(nn_np_ratio))}"
    return {
        "tpr":            (f"TPR@FPR={target_fpr:g}",                    "HEADLINE"),
        "purity_corr":    (f"purity@{tc}-comp (corr {r})",              "HEADLINE"),
        "purity_raw":     (f"purity@{tc}-comp (raw pool)",              "context"),
        "auprc":          ("AUPRC",                                     "HEADLINE (threshold-free)"),
        "comp_at_purity": (f"completeness@purity={target_purity:g} (corr {r})", "context"),
        "auroc":          ("AUROC",                                     "secondary/sanity"),
    }


def _resample_idx(pos_idx, neg_idx, rng):
    """Class-conditional WITH-REPLACEMENT resample (positives and negatives independently;
    never pooled) -- the same scheme as metrics.paired_bootstrap, applied to a metric
    VECTOR so every metric in the suite shares one replicate's row indices."""
    p = pos_idx[rng.integers(0, pos_idx.size, pos_idx.size)]
    n = neg_idx[rng.integers(0, neg_idx.size, neg_idx.size)]
    return np.concatenate([p, n])


# ---------------------------------------------------------------------------
# loading dumped per-fold scores
# ---------------------------------------------------------------------------

def _fold_file(run_dir: str, k: int) -> str:
    """The single ``scores/*_fold{k}.npz`` file in a run dir (error if 0 or >1)."""
    hits = sorted(glob.glob(os.path.join(run_dir, "scores", f"*_fold{k}.npz")))
    if not hits:
        raise FileNotFoundError(
            f"no scores/*_fold{k}.npz under {run_dir} (did the runner dump raw scores?)")
    if len(hits) > 1:
        raise ValueError(
            f"ambiguous fold {k} in {run_dir}: {len(hits)} matching score files "
            f"{[os.path.basename(h) for h in hits]} -- expected exactly one.")
    return hits[0]


def _available_folds(run_dir: str) -> list[int]:
    ks = []
    for p in glob.glob(os.path.join(run_dir, "scores", "*_fold*.npz")):
        m = re.search(r"_fold(\d+)\.npz$", os.path.basename(p))
        if m:
            ks.append(int(m.group(1)))
    return sorted(set(ks))


def _load_fold(run_dir: str, k: int) -> dict:
    with np.load(_fold_file(run_dir, k), allow_pickle=False) as z:
        out = {key: z[key] for key in z.files}
    for key in ("pos_scores", "neg_scores", "neg_ids"):
        if key not in out:
            raise KeyError(f"{_fold_file(run_dir, k)} is missing required key '{key}'")
    return out


# ---------------------------------------------------------------------------
# per-fold pairing (+ the alignment guard)
# ---------------------------------------------------------------------------

def _pair_fold(a: dict, b: dict, k: int):
    """Return (y, sA, sB) for fold k on identical rows, after the alignment guards.

    y: 1 for the fold-k positives, 0 for the shared negatives.
    sA/sB: the two methods' scores on those SAME rows (baseline=A, method=B).
    """
    # -- NEGATIVES: exact-id alignment or fail loudly ----------------------
    na, nb = a["neg_ids"], b["neg_ids"]
    if na.shape != nb.shape or not np.array_equal(na, nb):
        raise AssertionError(
            f"fold {k}: the two runs' shared negatives are NOT aligned "
            f"(neg_ids differ: {na.shape} vs {nb.shape}"
            + ("" if na.shape != nb.shape else
               f", first mismatch at index {int(np.argmax(na != nb))}")
            + "). Refusing to compare misaligned rows -- the pairing would be invalid.")

    # -- POSITIVES: same count (same cached fold); cross-check ids if both have them --
    pa, pb = a["pos_scores"], b["pos_scores"]
    if pa.shape != pb.shape:
        raise AssertionError(
            f"fold {k}: positive counts differ ({pa.shape[0]} vs {pb.shape[0]}) -- the "
            f"two runs did not score the same fold-{k} test positives.")
    if "pos_ids" in a and "pos_ids" in b and not np.array_equal(a["pos_ids"], b["pos_ids"]):
        raise AssertionError(
            f"fold {k}: pos_ids differ between the two runs -- positives are not aligned.")

    npos, nneg = pa.shape[0], a["neg_scores"].shape[0]
    y = np.concatenate([np.ones(npos, np.int64), np.zeros(nneg, np.int64)])
    sA = np.concatenate([pa, a["neg_scores"]]).astype(float)
    sB = np.concatenate([pb, b["neg_scores"]]).astype(float)
    return y, sA, sB


# ---------------------------------------------------------------------------
# aggregate cross-fold paired bootstrap
# ---------------------------------------------------------------------------

def _paired_suite_fold(y, sA, sB, skw, B, seed):
    """PER-FOLD paired difference (B-A) + 95% CI for EVERY metric in the suite, sharing
    one class-conditional resample per replicate across all metrics.

    Returns {key: {"A","B","diff","lo","hi","sig"}} where sig = (CI excludes 0).
    """
    y = np.asarray(y).ravel()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    a0 = metric_suite(y, sA, **skw)
    b0 = metric_suite(y, sB, **skw)
    reps = {k: np.empty(B, float) for k in SUITE_KEYS}
    rng = np.random.default_rng(seed)
    for bi in range(B):
        idx = _resample_idx(pos_idx, neg_idx, rng)
        yr = y[idx]
        a = metric_suite(yr, sA[idx], **skw)
        b = metric_suite(yr, sB[idx], **skw)
        for k in SUITE_KEYS:
            reps[k][bi] = b[k] - a[k]
    out = {}
    for k in SUITE_KEYS:
        diff = b0[k] - a0[k]
        if not np.isfinite(diff):
            # UNDEFINED difference: the metric is n/a for A and/or B on this fold (e.g.
            # completeness@target-purity when a model never reaches that purity). Report
            # diff = n/a with NO CI. A nanpercentile CI here is reachability-conditional --
            # built only from the replicates where BOTH methods happened to be defined --
            # which overstates a difference that is genuinely undefined. (Same rule as
            # report_bakeoff._metric_cell.)
            out[k] = {"A": a0[k], "B": b0[k], "diff": np.nan, "lo": np.nan, "hi": np.nan,
                      "sig": False}
            continue
        lo = float(np.nanpercentile(reps[k], 2.5))
        hi = float(np.nanpercentile(reps[k], 97.5))
        out[k] = {"A": a0[k], "B": b0[k], "diff": diff, "lo": lo, "hi": hi,
                  "sig": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0))}
    return out


def _aggregate_paired_bootstrap(fold_data, skw, B, seed):
    """fold_data: list of dicts {k, npos, nneg, posA, posB, negA, negB}.

    Class-conditional PAIRED bootstrap over folds for the WHOLE metric suite: resample
    the shared negatives ONCE per replicate (reused across folds -- it is literally one
    pool), resample each fold's positives independently, apply the SAME indices to A and
    B, and average each metric across folds per replicate. Returns, per metric key,
    per-method aggregate means + the mean signed difference (B-A) with its 95% CI and an
    excludes_zero verdict.
    """
    nneg = fold_data[0]["nneg"]
    for fd in fold_data:                      # shared pool -> identical length every fold
        assert fd["nneg"] == nneg, "shared-negative length differs across folds"

    def _suite_mean(use_b, pidx_by_fold, nidx):
        acc = {k: [] for k in SUITE_KEYS}
        for fd, pidx in zip(fold_data, pidx_by_fold):
            pos = fd["posB"] if use_b else fd["posA"]
            neg = fd["negB"] if use_b else fd["negA"]
            y = np.concatenate([np.ones(pidx.size, np.int64), np.zeros(nidx.size, np.int64)])
            s = np.concatenate([pos[pidx], neg[nidx]])
            su = metric_suite(y, s, **skw)
            for k in SUITE_KEYS:
                acc[k].append(su[k])
        return {k: float(np.nanmean(acc[k])) for k in SUITE_KEYS}

    # point estimates (no resampling): full-sample per-fold mean per metric
    ident_p = [np.arange(fd["npos"]) for fd in fold_data]
    ident_n = np.arange(nneg)
    A = _suite_mean(False, ident_p, ident_n)
    Bp = _suite_mean(True, ident_p, ident_n)

    reps = {k: np.empty(B, float) for k in SUITE_KEYS}
    rng = np.random.default_rng(seed)
    for bi in range(B):
        nidx = rng.integers(0, nneg, nneg)                       # shared neg resample
        pidx_by_fold = [rng.integers(0, fd["npos"], fd["npos"])  # per-fold pos resample
                        for fd in fold_data]
        a = _suite_mean(False, pidx_by_fold, nidx)
        b = _suite_mean(True, pidx_by_fold, nidx)
        for k in SUITE_KEYS:
            reps[k][bi] = b[k] - a[k]

    agg = {}
    for k in SUITE_KEYS:
        diff = Bp[k] - A[k]
        if not np.isfinite(diff):
            # UNDEFINED mean difference -> n/a with NO CI (see _paired_suite_fold).
            agg[k] = {"A": A[k], "B": Bp[k], "diff": np.nan, "median": np.nan,
                      "lo": np.nan, "hi": np.nan, "excludes_zero": False}
            continue
        lo = float(np.nanpercentile(reps[k], 2.5))
        hi = float(np.nanpercentile(reps[k], 97.5))
        agg[k] = {"A": A[k], "B": Bp[k], "diff": diff,
                  "median": float(np.nanmedian(reps[k])), "lo": lo, "hi": hi,
                  "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                        and (lo > 0.0 or hi < 0.0))}
    return {"metrics": agg, "B": int(B)}


# ---------------------------------------------------------------------------
# top-level comparison
# ---------------------------------------------------------------------------

def compare(baseline_dir, method_dir, *, folds=None, target_fpr=None,
            target_completeness=DEFAULT_TARGET_COMPLETENESS, nn_np_ratio=NN_NP_Q1,
            target_purity=DEFAULT_TARGET_PURITY, B=1000, seed=0,
            name_a=None, name_b=None, out_dir=None, verbose=True) -> dict:
    """Compare method B (method_dir) against baseline A (baseline_dir); diff = B - A.

    Reports the full low-FPR completeness+purity suite per fold and in aggregate, each
    with a paired-bootstrap CI of the mean difference: TPR@target_fpr (headline), AUPRC,
    purity@target_completeness (raw pool AND corrected to nn_np_ratio), completeness@
    target_purity, and AUROC (secondary/sanity). Purity is prevalence-corrected to
    nn_np_ratio (default NN_NP_Q1=2200) so it is comparable to Lines et al.
    """
    name_a = name_a or os.path.basename(os.path.normpath(baseline_dir))
    name_b = name_b or os.path.basename(os.path.normpath(method_dir))

    fa, fb = _available_folds(baseline_dir), _available_folds(method_dir)
    common = sorted(set(fa) & set(fb)) if folds is None else sorted(folds)
    if not common:
        raise ValueError(f"no common folds to compare (A has {fa}, B has {fb})")
    if folds is None and (fa != fb) and verbose:
        print(f"[warn] fold sets differ (A={fa}, B={fb}); comparing the intersection {common}")

    # resolve target_fpr from the dumped scores if not given
    tfpr_seen = set()
    paired = {}
    fold_data = []
    for k in common:
        a, b = _load_fold(baseline_dir, k), _load_fold(method_dir, k)
        for d in (a, b):
            if "target_fpr" in d:
                tfpr_seen.add(float(np.asarray(d["target_fpr"])))
        y, sA, sB = _pair_fold(a, b, k)
        npos = int((y == 1).sum()); nneg = int((y == 0).sum())
        paired[k] = {"y": y, "sA": sA, "sB": sB, "npos": npos, "nneg": nneg}
        fold_data.append({"k": k, "npos": npos, "nneg": nneg,
                          "posA": sA[:npos], "posB": sB[:npos],
                          "negA": sA[npos:], "negB": sB[npos:]})

    if target_fpr is None and len(tfpr_seen) == 1:
        target_fpr = tfpr_seen.pop()
    elif target_fpr is None:
        target_fpr = DEFAULT_TARGET_FPR
        if verbose and len(tfpr_seen) > 1:
            print(f"[warn] runs disagree on target_fpr {sorted(tfpr_seen)}; using {target_fpr}")

    skw = dict(target_fpr=target_fpr, target_completeness=target_completeness,
               nn_np_ratio=nn_np_ratio, target_purity=target_purity)
    labels = _metric_labels(target_fpr, target_completeness, nn_np_ratio, target_purity)

    # per-fold: full suite + paired-diff CI per metric
    per_fold = {}
    for k in common:
        pk = paired[k]
        m = _paired_suite_fold(pk["y"], pk["sA"], pk["sB"], skw, B, seed)
        per_fold[k] = {"n_pos": pk["npos"], "n_neg": pk["nneg"],
                       "metrics": {mk: {"A": _num(m[mk]["A"]), "B": _num(m[mk]["B"]),
                                        "diff": _num(m[mk]["diff"]),
                                        "ci": [_num(m[mk]["lo"]), _num(m[mk]["hi"])],
                                        "sig": bool(m[mk]["sig"])}
                                   for mk in SUITE_KEYS}}

    agg = _aggregate_paired_bootstrap(fold_data, skw, B, seed)["metrics"]
    aggregate = {mk: {"A": _num(agg[mk]["A"]), "B": _num(agg[mk]["B"]),
                      "diff": _num(agg[mk]["diff"]),
                      "ci": [_num(agg[mk]["lo"]), _num(agg[mk]["hi"])],
                      "distinguishable": bool(agg[mk]["excludes_zero"])}
                 for mk in SUITE_KEYS}

    results = {
        "name_A": name_a, "name_B": name_b, "diff": f"{name_b} - {name_a}",
        "target_fpr": target_fpr, "target_completeness": target_completeness,
        "nn_np_ratio": float(nn_np_ratio), "target_purity": target_purity,
        "folds": common, "B": B, "seed": seed,
        "metric_labels": {mk: labels[mk][0] for mk in SUITE_KEYS},
        "metric_roles": {mk: labels[mk][1] for mk in SUITE_KEYS},
        # Lines et al. reference point -- CONTEXT ONLY, never a tuning target (see metrics.py)
        "lines_reference": {"nn_np_q1": float(NN_NP_Q1),
                            "source_only_completeness": LINES_SOURCE_ONLY_COMPLETENESS,
                            "source_only_purity": LINES_SOURCE_ONLY_PURITY},
        "per_fold": {int(k): per_fold[k] for k in common},
        "aggregate": aggregate,
    }
    if verbose:
        _report(results)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "comparison.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        if verbose:
            print(f"\n[out] comparison.json -> {out_dir}")
    return results


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _num(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _f(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


def _sf(x, nd=4):    # signed
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:+.{nd}f}"


def _ci_str(ci):
    """CI bracket, or the explicit undefined marker when the diff carried no CI (both
    ends None -> the metric was n/a for a method, so no reachability-conditional CI)."""
    if ci[0] is None and ci[1] is None:
        return "(undefined -- no CI)"
    return f"[{_sf(ci[0])}, {_sf(ci[1])}]"


def _verdict(distinguishable, diff):
    if not distinguishable or diff is None:
        return "NOT distinguishable (CI straddles 0)"
    return ("DISTINGUISHABLE improvement (CI excludes 0, +)" if diff > 0
            else "DISTINGUISHABLE regression (CI excludes 0, -)")


# metric display order: HEADLINE (TPR, corrected purity, AUPRC) first, then context
# (raw purity, completeness@purity), AUROC last and explicitly secondary.
_REPORT_ORDER = ("tpr", "purity_corr", "auprc", "purity_raw", "comp_at_purity", "auroc")


def _report(r):
    a, b = r["name_A"], r["name_B"]
    tf = r["target_fpr"]
    labels = r["metric_labels"]; roles = r["metric_roles"]
    lref = r["lines_reference"]
    print("\n" + "=" * 92)
    print(f"PAIRED COMPARISON   {b}  vs  {a}   (diff = {b} - {a})")
    print(f"HEADLINE: TPR@FPR={tf:g} + purity  |  AUPRC (threshold-free)  |  AUROC SECONDARY/sanity")
    print(f"purity corrected to N_N/N_P = 1:{int(round(r['nn_np_ratio']))}  (Lines et al. Q1 deployment "
          f"rarity)   paired bootstrap B={r['B']}, seed={r['seed']}")
    print(f"Lines source-only reference (CONTEXT, not a target): ~{lref['source_only_completeness']*100:g}% "
          f"completeness @ ~{lref['source_only_purity']*100:g}% purity on real Q1")
    print("=" * 92)

    # ---- per-fold tables, one per metric (headline first) ----
    for mk in _REPORT_ORDER:
        lab, role = labels[mk], roles[mk]
        print(f"\nPER-FOLD  {lab}   [{role}]")
        print(f"    fold |  n_pos | {a[:12]:>12} | {b[:12]:>12} |     diff  | 95% CI of diff       | sig")
        print("    " + "-" * 80)
        for k in r["folds"]:
            mm = r["per_fold"][k]["metrics"][mk]
            print(f"    {k:>4} | {r['per_fold'][k]['n_pos']:>6} | {_f(mm['A']):>12} | {_f(mm['B']):>12} | "
                  f"{_sf(mm['diff']):>9} | {_ci_str(mm['ci']):>20} | "
                  f"{'yes' if mm['sig'] else ' no'}")

    # ---- aggregate (across-fold mean; CI of the mean difference) ----
    ag = r["aggregate"]
    print("\n" + "=" * 92)
    print("AGGREGATE (across-fold mean; paired bootstrap CI of the mean difference, B-A)")
    print("=" * 92)
    print(f"    {'metric':<34} | {a[:11]:>11} | {b[:11]:>11} |     diff  | 95% CI of diff       | excl. 0")
    print("    " + "-" * 92)
    for mk in _REPORT_ORDER:
        m = ag[mk]
        tag = "  <<" if mk in ("tpr", "purity_corr") else ""
        print(f"    {labels[mk]:<34} | {_f(m['A']):>11} | {_f(m['B']):>11} | {_sf(m['diff']):>9} | "
              f"{_ci_str(m['ci']):>20} | {'YES' if m['distinguishable'] else ' no'}{tag}")

    # ---- verdict: leads with TPR + purity, NOT AUROC ----
    print("\n" + "=" * 92)
    print("VERDICT  (headline = TPR@1e-3 and purity; AUROC is secondary/sanity only)")
    print("=" * 92)
    for mk in ("tpr", "purity_corr", "auprc"):
        m = ag[mk]
        print(f"  {labels[mk]:<34}: diff {_sf(m['diff'])}  ->  {_verdict(m['distinguishable'], m['diff'])}")
    au = ag["auroc"]
    print(f"  {'AUROC (secondary/sanity)':<34}: diff {_sf(au['diff'])}  ->  "
          f"{_verdict(au['distinguishable'], au['diff'])}")
    print("=" * 92)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Paired statistical comparison of two already-run methods "
                    "(reads dumped scores; retrains nothing). diff = method - baseline.")
    p.add_argument("baseline_dir", help="run dir of the BASELINE method (A)")
    p.add_argument("method_dir", help="run dir of the METHOD under test (B)")
    p.add_argument("--folds", default=None, help="comma-separated folds (default: intersection)")
    p.add_argument("--target-fpr", type=float, default=None,
                   help="default: read from the dumped scores")
    p.add_argument("--target-completeness", type=float, default=DEFAULT_TARGET_COMPLETENESS,
                   help="completeness at which purity is reported (default 0.5)")
    p.add_argument("--nn-np-ratio", type=float, default=NN_NP_Q1,
                   help="N_N/N_P for prevalence-corrected purity (default 2200 = Lines Q1)")
    p.add_argument("--target-purity", type=float, default=DEFAULT_TARGET_PURITY,
                   help="purity at which completeness is reported (default 0.24 = Lines ref)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name-a", default=None); p.add_argument("--name-b", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    for d in (args.baseline_dir, args.method_dir):
        if not os.path.isdir(os.path.join(d, "scores")):
            print(f"SKIP: no scores/ dir under {d} (run the method first so it dumps scores)")
            return 3
    folds = ([int(x) for x in args.folds.split(",") if x.strip() != ""]
             if args.folds else None)
    compare(args.baseline_dir, args.method_dir, folds=folds, target_fpr=args.target_fpr,
            target_completeness=args.target_completeness, nn_np_ratio=args.nn_np_ratio,
            target_purity=args.target_purity, B=args.bootstrap, seed=args.seed,
            name_a=args.name_a, name_b=args.name_b, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
