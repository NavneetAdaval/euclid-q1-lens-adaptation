"""
test_run_baseline.py
====================
End-to-end test for Baseline A (run_baseline.py) on the REAL feature cache.

Unlike the metric unit tests (test_evaluate.py Part 1, which are synthetic), Baseline
A has nothing to fake: it is a data-driven number. So this test runs the WHOLE thing
on the real cache with a tiny epoch schedule and asserts the result is well-formed and
label-free, then prints the actual Baseline A numbers.

REQUIRES the real feature cache (built once by cache_features.py). If the cache is
absent/incomplete it SKIPS with exit code 3 (same convention as the other real-slice
tests) -- it never fabricates a cache, because the point is the real number.

Checks
------
  * results.json is produced and carries source_risk + oracle + selection_gap.
  * pooled TPR is in [0,1] and its paired-bootstrap CI brackets it.
  * per-fold entries exist for EVERY cached fold.
  * the reported SOURCE-RISK checkpoint is the one with the best SOURCE-VAL metric
    (label-free selection) -- re-derived here independently -- and NOT simply the best
    TEST checkpoint (that is the ORACLE, computed separately). This is the leakage-
    critical assertion: selection used val, never the test lenses.

Run:
    python scripts\\test_run_baseline.py
Exit 0 on success, 1 on a failed check, 3 if the real cache was absent.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LENS_DATA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "data")))
CACHE_DIR = os.environ.get("LENS_CACHE_DIR", os.path.join(DATA_ROOT, "feature_cache"))
SCRATCH = os.environ.get(
    "LENS_SCRATCH",
    os.path.join(os.environ.get("TEMP", _HERE), "baseline_A_test"))

# tiny schedule -> a few SOURCE-RISK candidates, fast on CPU
TINY_EPOCHS = (1, 3, 8)

_fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


def cache_ready(cache) -> tuple[bool, str]:
    """The full Baseline-A cache needs source_train + val + shared negs + every fold."""
    if not cache.has("source_train"):
        return False, "source_train.npz missing"
    if not cache.has("val"):
        return False, "val.npz missing"
    if not cache.has("test_negatives_shared"):
        return False, "test_negatives_shared.npz missing"
    if not cache.folds():
        return False, "no test_pos_fold*.npz"
    return True, f"folds={cache.folds()}"


def val_ranking(checkpoints, cache, device="cpu", metric="auroc"):
    """Independently recompute each checkpoint's SOURCE-VAL metric and return the
    best-by-val checkpoint. This mirrors evaluate.py's label-free selector, so we can
    assert the harness selected by VAL (not by the test lenses)."""
    import evaluate as ev
    from metrics import auroc, tpr_at_fpr, DEFAULT_TARGET_FPR
    v = cache.load("val")
    scores_by_ckpt = {}
    for path in checkpoints:
        head, std, _ = ev.load_checkpoint(path, device)
        s = ev.score_features(head, std, v["features"], device)
        scores_by_ckpt[path] = (auroc(v["labels"], s) if metric == "auroc"
                                else tpr_at_fpr(v["labels"], s, DEFAULT_TARGET_FPR))
    best = max(scores_by_ckpt, key=lambda p: np.nan_to_num(scores_by_ckpt[p], nan=-np.inf))
    return best, scores_by_ckpt


def main():
    print("=" * 78)
    print("Baseline A -- source-only, end-to-end on the REAL feature cache")
    print("=" * 78)

    try:
        import evaluate as ev
        import run_baseline as rb
    except Exception as e:
        print(f"SKIP: cannot import run_baseline/evaluate ({e})")
        return 3

    cache = ev.FeatureCache(CACHE_DIR)
    ready, why = cache_ready(cache)
    print(f"  cache_dir = {CACHE_DIR}   ({why})")
    if not ready:
        print(f"SKIP: real feature cache incomplete -- {why}. Run cache_features.py first.")
        return 3

    folds = cache.folds()
    out_dir = os.path.join(SCRATCH, "eval_out")

    # ---- run the WHOLE Baseline A with a tiny schedule (train on CPU for the test) ----
    try:
        results = rb.run_baseline(
            CACHE_DIR, out_dir, head_type="linear", epochs=TINY_EPOCHS,
            batch_size=256, seed=42, train_device="cpu", eval_device="cpu",
            bootstrap=500, eval_seed=0, val_select_metric="auroc", verbose=True)
    except Exception as e:
        print(f"FAIL: run_baseline raised ({e})")
        traceback.print_exc()
        _fails.append("run_baseline raised")
        return 1

    print("\n" + "=" * 78)
    print("ASSERTIONS")
    print("=" * 78)

    # ---- results.json written, with the three reporting blocks ----
    rj = os.path.join(out_dir, "results.json")
    check(os.path.exists(rj), f"results.json written -> {rj}")
    for key in ("source_risk", "oracle", "selection_gap"):
        check(key in results, f"results carries '{key}'")
    check(os.path.isdir(os.path.join(out_dir, "scores")),
          "raw per-fold scores dumped (for the bake-off's paired comparisons)")

    sr = results["source_risk"]

    # ---- per-fold entries for EVERY cached fold ----
    sr_folds = sorted(int(k) for k in sr["per_fold"])
    check(sr_folds == list(folds),
          f"per-fold entries for all cached folds (got {sr_folds}, cache {list(folds)})")

    # ---- pooled TPR in [0,1] with a CI that brackets it ----
    ptpr = sr["pooled_tpr"]
    check(ptpr is not None and 0.0 <= ptpr <= 1.0, f"pooled TPR in [0,1] ({ptpr})")
    ci = sr.get("pooled_tpr_ci")
    check(ci is not None and ci[0] is not None and ci[1] is not None
          and ci[0] <= ptpr <= ci[1],
          f"pooled TPR {ptpr} within its bootstrap CI {ci}")
    pau = sr["pooled_auroc"]
    check(pau is not None and 0.0 <= pau <= 1.0, f"pooled AUROC in [0,1] ({pau})")

    # ---- THE leakage-critical check: SOURCE-RISK was chosen by SOURCE-VAL, not test ----
    checkpoints = [c["checkpoint"] if isinstance(c, dict) else c
                   for c in _candidate_paths(out_dir)]
    best_val_ckpt, val_scores = val_ranking(checkpoints, cache, device="cpu", metric="auroc")
    sel = os.path.abspath(sr["checkpoint"])
    check(sel == os.path.abspath(best_val_ckpt),
          "SOURCE-RISK checkpoint == the best-by-SOURCE-VAL checkpoint (label-free selection)")
    # and expose that this is a val ranking, not a test ranking
    print("    val-AUROC per candidate (the label-free selector evaluate.py used):")
    for p in sorted(val_scores, key=lambda q: os.path.basename(q)):
        tag = "  <- selected" if os.path.abspath(p) == sel else ""
        print(f"      {os.path.basename(p):<26} val_auroc={val_scores[p]:.4f}{tag}")

    # ---- report the actual Baseline A numbers ----
    print("\n" + "=" * 78)
    print("BASELINE A -- actual numbers (SOURCE-RISK headline, label-free)")
    print("=" * 78)
    print(f"  reported checkpoint : {os.path.basename(sr['checkpoint'])}"
          f"   (val_auroc={sr['val_metric']:.4f})")
    print("  per-fold TPR@1e-3 / AUROC:")
    for k in folds:
        pf = sr["per_fold"][str(k)] if str(k) in sr["per_fold"] else sr["per_fold"][k]
        print(f"    fold {k}: n_pos={pf['n_pos']:>4}  TPR={_f(pf['tpr'])}  AUROC={_f(pf['auroc'])}")
    print(f"  across-fold mean : TPR={_f(sr['mean_tpr'])} +- {_f(sr['std_tpr'])}   "
          f"AUROC={_f(sr['mean_auroc'])} +- {_f(sr['std_auroc'])}")
    print(f"  POOLED (489 pos vs shared negs):")
    print(f"    TPR@1e-3 = {_f(sr['pooled_tpr'])}  CI {_ci(sr.get('pooled_tpr_ci'))}")
    print(f"    AUROC    = {_f(sr['pooled_auroc'])}  CI {_ci(sr.get('pooled_auroc_ci'))}")
    print(f"  alert-rate guardrail: mean={_f(sr['mean_alert_rate'])}  "
          f"fired={sr['guardrail_fired']}")
    gap = results["selection_gap"]
    print(f"  SELECTION GAP (oracle - source-risk): TPR={_f(gap['tpr'])}  "
          f"AUROC={_f(gap['auroc'])}  same_ckpt={gap['same_checkpoint']}")

    print("\n" + "=" * 78)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL BASELINE-A CHECKS PASSED.")
    return 0


def _candidate_paths(out_dir):
    """The epoch-checkpoints run_baseline just wrote (the SOURCE-RISK candidates)."""
    import glob
    ck = os.path.join(out_dir, "checkpoints")
    return sorted(glob.glob(os.path.join(ck, "baseline_*_ep*.pt")))


def _f(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(ci):
    if not ci or ci[0] is None or ci[1] is None:
        return "[nan, nan]"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
