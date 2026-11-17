"""
analyze_fold0_selection.py
==========================
FOLD-0 CHECKPOINT-SELECTION ANALYSIS (protocol-setting, fold 0 ONLY).

Reads a run's saved fold-0 learning curve (``curve.csv`` written by curve_hook.py) and
answers three questions for the upcoming matched 5-fold bake-off (SPEC
spec_fold0_selection_check.md; PROJECT.md 7.4/7.6):

  Task A  Flatness   -- is the TPR@1e-3 curve flat inside the final-25% window
                        (steps 6000-8000 at the locked 8000-step budget)?
  Task B  Rules      -- what fold-0 TPR would each blind selection rule pick
                        (R1 last / R2 fixed-plateau / R3 median), vs the oracle?
  Task C  Saturation -- does LABEL-FREE source-val purity@50%-completeness saturate
                        (dead selector) or track the real-test TPR (alive)?

NO NEW TRAINING. Reads saved curves + checkpoints + the frozen-feature cache only.
Every metric comes from metrics.py; scoring reuses evaluate.load_checkpoint +
evaluate.score_features (the exact path curve_hook.py used). This module owns only
plumbing + the selection-rule logic, never the metric math.

Claim tags used in the printed report:  [RB]=research-backed  [EC]=engineering-choice
[UNVERIFIED]=assumption/open.

USAGE
-----
    python scripts/analyze_fold0_selection.py baseline          # shorthand
    python scripts/analyze_fold0_selection.py runs/curve_dann_fold0
    python scripts/analyze_fold0_selection.py freematch --cache-dir data/feature_cache
    python scripts/analyze_fold0_selection.py --self-test       # mock + adversarial checks
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---- reuse the pipeline's own code (no metric is reimplemented here) ----
from evaluate import FeatureCache, load_checkpoint, score_features          # noqa: E402
from metrics import (paired_bootstrap, purity_at_completeness, tpr_at_fpr,   # noqa: E402
                     DEFAULT_TARGET_FPR, DEFAULT_TARGET_COMPLETENESS)

# ---- locked protocol constants (PROJECT.md 7.4; SPEC 2) ----
BUDGET = 8000                     # [RB] locked step budget (PROJECT.md 7.4)
WINDOW_FRAC = 0.25                # [RB] final 25% of budget = window (SPEC A2)
WINDOW_LO = int(round(BUDGET * (1.0 - WINDOW_FRAC)))   # 6000, inclusive
WINDOW_HI = BUDGET                                     # 8000, inclusive
TPR_COL = "fold0_tpr_1e-3"        # DEFAULT column: fold-0 test TPR@1e-3 (curve_hook.py).
                                  # Overridden per fold via --fold/--tpr-col; task_a/b/c
                                  # take the column as an argument, so nothing below reads
                                  # this constant directly.
STEP_COL = "step"
REAL_PURITY_COL = "purity_50comp"  # existing REAL-domain purity (signal_purity.py) -- reference only

SHORTHAND = {
    "baseline": "runs/curve_baseline_fold0",
    "dann": "runs/curve_dann_fold0",
    "freematch": "runs/curve_freematch_fold0",
}


# ===========================================================================
# small helpers
# ===========================================================================

def _resolve_run_dir(arg: str) -> str:
    d = SHORTHAND.get(arg.lower(), arg)
    d = os.path.abspath(d)
    if not os.path.isdir(d):
        raise SystemExit(f"[analyze] ERROR: run dir not found: {d}")
    if not os.path.isfile(os.path.join(d, "curve.csv")):
        raise SystemExit(f"[analyze] ERROR: no curve.csv in {d}")
    return d


def _load_curve(run_dir: str) -> pd.DataFrame:
    """curve.csv, plus the REAL-domain purity column merged in if curve_with_signals.csv
    exists (Task C reference). Sorted by step, deduped."""
    df = pd.read_csv(os.path.join(run_dir, "curve.csv"))
    sig = os.path.join(run_dir, "curve_with_signals.csv")
    if os.path.isfile(sig) and REAL_PURITY_COL not in df.columns:
        s = pd.read_csv(sig)
        if REAL_PURITY_COL in s.columns and STEP_COL in s.columns:
            df = df.merge(s[[STEP_COL, REAL_PURITY_COL]], on=STEP_COL, how="left")
    return df.sort_values(STEP_COL).drop_duplicates(STEP_COL).reset_index(drop=True)


def _ols_slope_per_1000(steps, tpr):
    """OLS slope of TPR vs step, expressed in TPR units per 1000 steps (SPEC A4)."""
    steps = np.asarray(steps, float)
    tpr = np.asarray(tpr, float)
    if steps.size < 2 or np.ptp(steps) == 0:
        return np.nan
    slope = np.polyfit(steps, tpr, 1)[0]      # TPR per 1 step
    return slope * 1000.0


def _median_checkpoint_idx(tpr):
    """Index (into the given window arrays) of the checkpoint whose TPR is the MEDIAN of
    the window -- the checkpoint at the lower-median order statistic (floor((n-1)/2) of the
    TPR-sorted list). [EC] We return an ACTUAL checkpoint (not the median VALUE) because
    downstream compare_methods.py needs a real score vector to paired-bootstrap (SPEC 0,
    R3 note). For even n the two central checkpoints tie as 'the median'; taking the lower
    one is a fixed, documented tie-break so the pick is deterministic."""
    tpr = np.asarray(tpr, float)
    order = np.argsort(tpr, kind="stable")    # ascending; stable => ties keep step order
    return int(order[(tpr.size - 1) // 2])


def _score_vector_at_step(run_dir, step, cache, pos_split, neg_split, target_fpr,
                          tpr_ref=None):
    """Recompute the fold-0 (y, scores) at ``step`` from the frozen-feature cache + that
    step's checkpoint head -- the exact curve_hook.py path. If ``tpr_ref`` is given, verify
    the recomputed TPR@fpr matches it (guarantees checkpoint<->curve correspondence, same
    fail-loud self-check signal_purity.py uses)."""
    ckpt = os.path.join(run_dir, "checkpoints", f"step_{int(step)}.pt")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"[analyze] ERROR: missing checkpoint for step {step}: {ckpt}")
    head, std, _ = load_checkpoint(ckpt, "cpu")
    pos = cache.load(pos_split)["features"]
    neg = cache.load(neg_split)["features"]
    ps = score_features(head, std, pos, "cpu")
    ns = score_features(head, std, neg, "cpu")
    y = np.concatenate([np.ones(ps.size, np.int64), np.zeros(ns.size, np.int64)])
    s = np.concatenate([ps, ns])
    if tpr_ref is not None and np.isfinite(tpr_ref):
        tpr_now = tpr_at_fpr(y, s, target_fpr)
        if not np.isclose(tpr_now, float(tpr_ref), atol=1e-6, rtol=0.0):
            raise SystemExit(
                f"[analyze] ERROR: step {step} recomputed TPR={tpr_now:.6f} != curve "
                f"{tpr_ref:.6f}; cache/checkpoints do not match this curve.")
    return y, s


def _saved_fold0_scores(run_dir):
    """(y, scores) from the run's SAVED final-checkpoint score vector in scores/ (keys
    pos_scores/neg_scores). This is SPEC A5's 'saved score vectors' path -- used as the
    noise yardstick when the checkpoint head cannot be re-loaded by evaluate.load_checkpoint
    (e.g. DANN's dann_adapter). Returns (None, None) if no usable npz is found."""
    import glob
    sdir = os.path.join(run_dir, "scores")
    if not os.path.isdir(sdir):
        return None, None, None
    cands = sorted(glob.glob(os.path.join(sdir, "*fold0*.npz"))) or \
        sorted(glob.glob(os.path.join(sdir, "*.npz")))
    for p in cands:
        with np.load(p, allow_pickle=False) as d:
            if "pos_scores" in d.files and "neg_scores" in d.files:
                ps, ns = d["pos_scores"], d["neg_scores"]
                y = np.concatenate([np.ones(ps.size, np.int64), np.zeros(ns.size, np.int64)])
                s = np.concatenate([ps, ns]).astype(float)
                return y, s, os.path.basename(p)
    return None, None, None


def _spearman(x, y):
    """Spearman rank correlation with average-rank tie handling (n and rho). Dependency-
    free (numpy only) so the script needs no scipy."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3:
        return np.nan, n

    def _avg_rank(a):
        order = np.argsort(a, kind="stable")
        ranks = np.empty(a.size, float)
        sa = a[order]
        i = 0
        while i < a.size:
            j = i
            while j + 1 < a.size and sa[j + 1] == sa[i]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # average rank (1-based)
            i = j + 1
        return ranks

    rx, ry = _avg_rank(x), _avg_rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, n
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho, n


# ===========================================================================
# Task A -- flatness
# ===========================================================================

def task_a(df, cache, run_dir, pos_split, neg_split, target_fpr, tpr_col=TPR_COL,
           saved_scores_only=False):
    print("\n" + "=" * 78)
    print(f"TASK A -- FLATNESS CHECK  (is {tpr_col} flat in the final-25% window?)")
    print("=" * 78)
    if tpr_col not in df.columns:
        raise SystemExit(f"[analyze] ERROR: column {tpr_col!r} not in curve.csv "
                         f"(have: {list(df.columns)})")

    # A1 truncate to <= budget (INCLUSIVE)
    full_n = len(df)
    trunc = df[df[STEP_COL] <= BUDGET].reset_index(drop=True)
    print(f"[A1] truncate to steps <= {BUDGET} (inclusive): "
          f"{len(trunc)} of {full_n} checkpoints kept "
          f"({full_n - len(trunc)} discarded beyond budget).")
    if len(trunc) == 0:
        print("     no checkpoints within budget -- cannot analyse."); return None

    # A2 window
    win = trunc[(trunc[STEP_COL] >= WINDOW_LO) & (trunc[STEP_COL] <= WINDOW_HI)]
    win = win.reset_index(drop=True)
    print(f"[A2] window = steps {WINDOW_LO}-{WINDOW_HI} (inclusive): "
          f"{len(win)} checkpoints inside.")

    max_step = int(trunc[STEP_COL].max())
    if len(win) == 0:
        # FreeMatch case: curve never reaches the window. Descriptive-only, verdict PENDING.
        print(f"     [FLAG] curve ends at step {max_step} < {WINDOW_LO}: the locked window "
              f"has ZERO checkpoints. Per protocol the window is NOT relocated/rescaled.")
        print(f"     -> reporting descriptive stats over the AVAILABLE range only; no "
              f"window verdict, no selection-rule comparison.")
        _describe(trunc[tpr_col].values, trunc[STEP_COL].values, label="available range", tpr_col=tpr_col)
        print(f"\n[A6] VERDICT: PENDING-8K -- flatness not testable; curve stops at "
              f"{max_step}. Re-run this method to the full {BUDGET}-step budget "
              f"(the 5-fold bake-off) before a flatness verdict. [EC]")
        return {"window_n": 0, "max_step": max_step, "trunc": trunc, "win": win,
                "verdict": "PENDING-8K"}

    if len(win) < 3:
        print(f"     [FLAG] fewer than 3 in-window checkpoints -> the R3 median rule is "
              f"not well-resolved at this spacing; continuing with the rest.")

    # A3 window stats
    stats = _describe(win[tpr_col].values, win[STEP_COL].values, label="window", tpr_col=tpr_col)

    # A4 trend
    slope_k = _ols_slope_per_1000(win[STEP_COL].values, win[tpr_col].values)
    total_change = slope_k * (WINDOW_HI - WINDOW_LO) / 1000.0
    print(f"[A4] OLS trend across window: slope = {slope_k:+.5f} TPR per 1000 steps; "
          f"predicted total change over {WINDOW_LO}-{WINDOW_HI} = {total_change:+.4f} TPR.")

    # A5 noise yardstick -- bootstrap CI on TPR@1e-3. Prefer recomputing at the R1
    # (last-in-budget) checkpoint; if that checkpoint's head can't be loaded by the generic
    # linear/mlp loader (e.g. DANN's dann_adapter), fall back to the saved final-checkpoint
    # score vector in scores/ (SPEC A5's 'saved score vectors'). Noise scale is dominated by
    # the ~98 positives, so it is ~checkpoint-invariant -- the fallback is a valid yardstick.
    r1_step = int(trunc[STEP_COL].iloc[-1])
    r1_tpr = float(trunc[tpr_col].iloc[-1])
    if saved_scores_only:
        # REQUIRED for a partially-unfrozen encoder (n_blocks>0). The recompute path scores
        # the saved head against the FROZEN FEATURE CACHE, which is stale once the encoder
        # trains -- it pairs this run's head with a different encoder's features and yields
        # a number the model never produced (MEASURED on nb1: recompute 0.2245 vs the
        # curve's true 0.0918, and the tool's own guard correctly refused). The saved
        # scores/*.npz IS the R1 vector, produced with the live encoder, so here it is not a
        # fallback but the exact right input.
        y, s, npz = _saved_fold0_scores(run_dir)
        if y is None:
            raise SystemExit("[analyze] ERROR: --saved-scores given but no usable "
                             "scores/*.npz found.")
        ci_src = (f"SAVED R1 score vector ({npz}) -- forced via --saved-scores "
                  f"(the feature cache is invalid for an unfrozen encoder)")
        bs = paired_bootstrap(y, s, B=1000, seed=0, target_fpr=target_fpr)
        half = 0.5 * (bs["hi"] - bs["lo"])
        ratio = stats["range"] / half if half > 0 else np.nan
        print(f"[A5] noise yardstick: paired-bootstrap CI of TPR@{target_fpr:g} "
              f"[{bs['lo']:.3f}, {bs['hi']:.3f}] (half-width {half:.3f}); source: {ci_src}.")
        print(f"     RATIO  window-range / CI-half-width = {stats['range']:.4f} / "
              f"{half:.4f} = {ratio:.2f}")
        verdict = _verdict(slope_k, total_change, stats["range"], half, ratio)
        print(f"\n[A6] VERDICT: {verdict['tag']}")
        print(f"     reasoning: {verdict['why']}")
        return {"window_n": len(win), "trunc": trunc, "win": win, "stats": stats,
                "slope_k": slope_k, "total_change": total_change, "ci_half": half,
                "ratio": ratio, "r1_step": r1_step, "verdict": verdict["tag"]}
    try:
        y, s = _score_vector_at_step(run_dir, r1_step, cache, pos_split, neg_split,
                                     target_fpr, tpr_ref=r1_tpr)
        ci_src = f"recomputed at in-budget checkpoint step {r1_step}"
    except (ValueError, KeyError, RuntimeError):
        y, s, npz = _saved_fold0_scores(run_dir)
        if y is None:
            raise SystemExit("[analyze] ERROR: cannot load checkpoint head AND no saved "
                             "scores/*.npz for the A5 yardstick.")
        ci_src = (f"SAVED final-checkpoint scores ({npz}) -- checkpoint head not loadable "
                  f"by the generic linear/mlp loader")
    bs = paired_bootstrap(y, s, B=1000, seed=0, target_fpr=target_fpr)
    half = 0.5 * (bs["hi"] - bs["lo"])
    ratio = stats["range"] / half if half > 0 else np.nan
    print(f"[A5] noise yardstick: paired-bootstrap CI of TPR@{target_fpr:g} "
          f"[{bs['lo']:.3f}, {bs['hi']:.3f}] (half-width {half:.3f}); source: {ci_src}.  "
          f"[RB] metrics.paired_bootstrap, class-conditional.")
    print(f"     RATIO  window-range / CI-half-width = {stats['range']:.4f} / {half:.4f} "
          f"= {ratio:.2f}")

    # A6 verdict (judgement, no hard-coded threshold -- SPEC A6)
    verdict = _verdict(slope_k, total_change, stats["range"], half, ratio)
    print(f"\n[A6] VERDICT: {verdict['tag']}")
    print(f"     reasoning: {verdict['why']}")
    print(f"     (numbers shown above; the tag is a judgement, not a hard threshold -- "
          f"Navneet decides. [EC])")
    return {"window_n": len(win), "trunc": trunc, "win": win, "stats": stats,
            "slope_k": slope_k, "total_change": total_change, "ci_half": half,
            "ratio": ratio, "r1_step": r1_step, "verdict": verdict["tag"]}


def _describe(tpr, steps, label, tpr_col=TPR_COL):
    tpr = np.asarray(tpr, float)
    d = {"n": tpr.size, "mean": float(np.mean(tpr)), "median": float(np.median(tpr)),
         "min": float(np.min(tpr)), "max": float(np.max(tpr)),
         "range": float(np.ptp(tpr)), "std": float(np.std(tpr, ddof=1)) if tpr.size > 1 else 0.0}
    print(f"[A3] {label} stats on {tpr_col}: n={d['n']}  mean={d['mean']:.4f}  "
          f"median={d['median']:.4f}  min={d['min']:.4f}  max={d['max']:.4f}  "
          f"range={d['range']:.4f}  std={d['std']:.4f}")
    return d


def _verdict(slope_k, total_change, rng, half, ratio):
    """Map (slope, range/CI) to one of the four SPEC A6 tags. Thresholds here are
    DISPLAY heuristics only -- the printed numbers are the evidence. [EC]"""
    slope_big = abs(total_change) > half            # trend moves more than a CI half-width
    if slope_big and total_change < 0:
        return {"tag": "DEGRADING",
                "why": f"predicted window change {total_change:+.4f} is negative and "
                       f"exceeds the CI half-width {half:.3f} -> late degradation."}
    if slope_big and total_change > 0:
        return {"tag": "NOT-CONVERGED",
                "why": f"predicted window change {total_change:+.4f} is positive and "
                       f"exceeds the CI half-width {half:.3f} -> still rising at 6000."}
    if np.isfinite(ratio) and ratio <= 1.0:
        return {"tag": "FLAT-TIGHT",
                "why": f"near-flat slope (|change|={abs(total_change):.4f} < CI half "
                       f"{half:.3f}) and range {rng:.4f} <= CI half {half:.3f} "
                       f"(ratio {ratio:.2f})."}
    return {"tag": "FLAT-JITTERY",
            "why": f"near-flat slope (|change|={abs(total_change):.4f} < CI half "
                   f"{half:.3f}) but range {rng:.4f} is comparable to / above the CI half "
                   f"{half:.3f} (ratio {ratio:.2f})."}


# ===========================================================================
# Task B -- selection-rule comparison
# ===========================================================================

def task_b(a_out, target_fpr, ci_half, tpr_col=TPR_COL):
    print("\n" + "=" * 78)
    print("TASK B -- SELECTION-RULE COMPARISON  (which fold-0 TPR each blind rule picks)")
    print("=" * 78)
    if a_out is None or a_out.get("window_n", 0) == 0:
        print("[B] SKIPPED: no in-window checkpoints (curve stops before the window). "
              "Per protocol, no selection-rule comparison for this method until its "
              "8000-step run exists.")
        return

    trunc, win = a_out["trunc"], a_out["win"]
    steps = trunc[STEP_COL].values
    tprs = trunc[tpr_col].values

    # R1 last -- largest step <= budget
    i_r1 = int(np.argmax(steps))
    # R2 fixed-plateau -- nearest available <= 6000
    le6000 = trunc[trunc[STEP_COL] <= WINDOW_LO]
    if len(le6000):
        r2_step = int(le6000[STEP_COL].iloc[-1])
        i_r2 = int(np.where(steps == r2_step)[0][0])
    else:
        i_r2 = None
    # R3 median -- median checkpoint of the window
    j_med = _median_checkpoint_idx(win[tpr_col].values)
    r3_step = int(win[STEP_COL].iloc[j_med])
    i_r3 = int(np.where(steps == r3_step)[0][0])
    # ORACLE -- max TPR over all <= budget
    i_or = int(np.argmax(tprs))

    oracle_tpr = float(tprs[i_or])
    rows = [("R1 last", i_r1), ("R2 fixed-plateau", i_r2),
            ("R3 median", i_r3), ("ORACLE", i_or)]
    print(f"{'rule':<18}{'step':>8}{'TPR@1e-3':>11}{'gap_to_oracle':>15}")
    print("-" * 52)
    picks = {}
    for name, i in rows:
        if i is None:
            print(f"{name:<18}{'--':>8}{'n/a':>11}{'':>15}  (no checkpoint <= 6000)")
            continue
        tp = float(tprs[i]); gap = tp - oracle_tpr
        picks[name] = {"step": int(steps[i]), "tpr": tp, "gap": gap}
        print(f"{name:<18}{int(steps[i]):>8}{tp:>11.4f}{gap:>+15.4f}")

    print("\n[guardrail -- SPEC 3] The ORACLE is the best-test checkpoint on ~98 positives, "
          "so it overfits test noise; every gap-to-oracle OVERSTATES what a real rule loses. "
          "The comparison that matters is RULE vs RULE, not rule vs oracle.")

    # pairwise diffs vs CI half-width
    def _d(a, b):
        if a in picks and b in picks:
            return picks[a]["tpr"] - picks[b]["tpr"]
        return np.nan
    print("\n[pairwise rule differences]  (compare against CI half-width "
          f"{ci_half:.3f} from A5)")
    for lbl, a, b in [("R1 - R3", "R1 last", "R3 median"),
                      ("R1 - R2", "R1 last", "R2 fixed-plateau"),
                      ("R2 - R3", "R2 fixed-plateau", "R3 median")]:
        dv = _d(a, b)
        tag = ("indistinguishable" if np.isfinite(dv) and abs(dv) <= ci_half
               else "exceeds CI half" if np.isfinite(dv) else "n/a")
        print(f"   {lbl:<9} = {dv:+.4f}   [{tag}]")
    print("   [RB] If all pairwise |diff| <= CI half-width, the rules are indistinguishable "
          "and R1 (last) wins on simplicity (SPEC 3; PROJECT.md 7.4).")


# ===========================================================================
# Task C -- source-val purity saturation
# ===========================================================================

def task_c(df, cache, run_dir, target_fpr, tpr_col=TPR_COL):
    print("\n" + "=" * 78)
    print("TASK C -- SOURCE-VAL PURITY@50%-COMPLETENESS SATURATION  (label-free selector)")
    print("=" * 78)
    if not cache.has("val"):
        print("[C] SKIPPED: no 'val' split in the feature cache -> cannot compute the "
              "label-free source-val purity. (Real-domain purity_50comp is a DIFFERENT, "
              "non-label-free quantity; see note below.)")
        return

    val = cache.load("val")
    yv = val["labels"].astype(np.int64)
    feat = val["features"]
    steps = df[STEP_COL].values
    print(f"[C] computing source-val purity@{int(DEFAULT_TARGET_COMPLETENESS*100)}%-comp "
          f"(raw pool, sim-lens vs sim-nonlens) over {len(steps)} checkpoints; "
          f"val pool: {int((yv==1).sum())} pos / {int((yv==0).sum())} neg.")

    sv_pur, real_pur, tpr_real = [], [], []
    for _, row in df.iterrows():
        step = int(row[STEP_COL])
        ckpt = os.path.join(run_dir, "checkpoints", f"step_{step}.pt")
        if not os.path.isfile(ckpt):
            raise SystemExit(f"[analyze] ERROR: missing checkpoint step {step}: {ckpt}")
        try:
            head, std, _ = load_checkpoint(ckpt, "cpu")
        except (ValueError, KeyError) as e:
            print(f"[C] SKIPPED: checkpoint head is not loadable by the generic linear/mlp "
                  f"loader ({e}). Source-val purity needs to score the 'val' split through "
                  f"this method's head, which this script cannot reconstruct -- so Task C is "
                  f"not computable for this method here. NOT fabricating a value. [EC]")
            return
        sv = score_features(head, std, feat, "cpu")
        p = purity_at_completeness(yv, sv, DEFAULT_TARGET_COMPLETENESS, nn_np_ratio=None)
        sv_pur.append(p["purity"])
        tpr_real.append(float(row[tpr_col]) if pd.notna(row[tpr_col]) else np.nan)
        real_pur.append(float(row[REAL_PURITY_COL])
                        if REAL_PURITY_COL in df.columns and pd.notna(row.get(REAL_PURITY_COL))
                        else np.nan)

    sv_pur = np.array(sv_pur, float)
    rng = float(np.nanmax(sv_pur) - np.nanmin(sv_pur))
    print(f"[C] source-val purity: n={np.isfinite(sv_pur).sum()}  "
          f"min={np.nanmin(sv_pur):.4f}  max={np.nanmax(sv_pur):.4f}  "
          f"range={rng:.4f}  mean={np.nanmean(sv_pur):.4f}")

    rho, n = _spearman(sv_pur, tpr_real)
    print(f"[C] Spearman(source-val purity, fold-0 test TPR@1e-3) = "
          f"{rho:.3f}  (n={n})")

    have_real = np.isfinite(np.array(real_pur, float)).any()
    if have_real:
        real_pur = np.array(real_pur, float)
        rho_rr, n_rr = _spearman(sv_pur, real_pur)
        print(f"[C] REFERENCE (non-label-free): real-domain purity_50comp range = "
              f"{np.nanmax(real_pur)-np.nanmin(real_pur):.4f} "
              f"[{np.nanmin(real_pur):.4f}, {np.nanmax(real_pur):.4f}].")
        print(f"    Spearman(source-val purity, REAL-domain purity) = {rho_rr:.3f} (n={n_rr}) "
              f"-- does the label-free version track the real one?")
    else:
        print("[C] (no real-domain purity_50comp column found for the reference overlay.)")

    # verdict
    pinned = np.nanmin(sv_pur) >= 0.95 and rng <= 0.05
    print("\n[C] VERDICT:", end=" ")
    if pinned:
        print("DEAD SELECTOR -- source-val purity is pinned near 1.0 (range "
              f"{rng:.4f}), so on the easy sim domain the 50%-completeness point sits at "
              "near-zero FPR and carries no selection information. Drop it as a selector. "
              "[RB] confirms the 7.6 saturation hypothesis (same mechanism as source-val "
              "AUROC ~0.99).")
    elif np.isfinite(rho) and rho >= 0.4:
        print(f"ALIVE -- source-val purity VARIES (range {rng:.4f}) and tracks the real-test "
              f"TPR (Spearman {rho:.2f}); the earlier 'it must saturate' reasoning would be "
              "wrong. Worth carrying as a label-free selector candidate. [UNVERIFIED until "
              "confirmed across folds]")
    else:
        print(f"INCONCLUSIVE/UNUSABLE -- source-val purity varies (range {rng:.4f}) but does "
              f"NOT track real-test TPR (Spearman {rho:.2f}); moving but uninformative, so "
              "not a usable selector. [EC]")


# ===========================================================================
# self-test (mock + adversarial) -- SPEC 6
# ===========================================================================

def self_test():
    print("SELF-TEST (mock + adversarial) ...")
    ok = True

    # 1. median rule picks the correct checkpoint on a hand-built curve.
    #    window TPRs (step-ordered): 0.10, 0.50, 0.30, 0.20, 0.40
    #    sorted ascending: 0.10,0.20,0.30,0.40,0.50 -> median value 0.30 at the 3rd row.
    tpr = np.array([0.10, 0.50, 0.30, 0.20, 0.40])
    j = _median_checkpoint_idx(tpr)
    assert j == 2 and tpr[j] == 0.30, f"median idx {j} tpr {tpr[j]}"
    print("  [PASS] R3 median picks the median-TPR checkpoint (odd n).")

    #    even n: 0.10,0.20,0.30,0.40 -> lower-median = 0.20 (index 1). documented tie-break.
    j2 = _median_checkpoint_idx(np.array([0.10, 0.20, 0.30, 0.40]))
    assert j2 == 1, f"even-n median idx {j2}"
    print("  [PASS] R3 median lower-median tie-break on even n (picks 0.20).")

    #    ties keep step order (stable): [0.3,0.3,0.3] -> index 1 (middle).
    j3 = _median_checkpoint_idx(np.array([0.3, 0.3, 0.3]))
    assert j3 == 1, f"tie median idx {j3}"
    print("  [PASS] R3 median stable under all-tie window.")

    # 2. truncation at 8000 is INCLUSIVE; window 6000-8000 inclusive at both ends.
    df = pd.DataFrame({STEP_COL: [5500, 6000, 7000, 8000, 8500],
                       TPR_COL: [0.1, 0.2, 0.3, 0.4, 0.9]})
    trunc = df[df[STEP_COL] <= BUDGET]
    assert list(trunc[STEP_COL]) == [5500, 6000, 7000, 8000], "8000 must be kept, 8500 dropped"
    print("  [PASS] truncation <=8000 is inclusive (keeps 8000, drops 8500).")
    win = trunc[(trunc[STEP_COL] >= WINDOW_LO) & (trunc[STEP_COL] <= WINDOW_HI)]
    assert list(win[STEP_COL]) == [6000, 7000, 8000], "window must include both 6000 and 8000"
    print("  [PASS] window is inclusive at 6000 and 8000.")

    # 3. oracle excludes the out-of-budget spike (0.9 at 8500 must NOT be the oracle).
    oracle = float(trunc[TPR_COL].max())
    assert oracle == 0.4, f"oracle leaked out-of-budget point: {oracle}"
    print("  [PASS] oracle is computed over <=8000 only (ignores 0.9 at 8500).")

    # 4. slope sign: a rising line gives positive per-1000 slope.
    sk = _ols_slope_per_1000([6000, 7000, 8000], [0.20, 0.30, 0.40])
    assert abs(sk - 0.10) < 1e-9, f"slope {sk}"
    print("  [PASS] OLS slope = +0.100 TPR/1000 steps on a known rising line.")

    # 5. spearman on a monotone relation with ties = 1.0.
    rho, n = _spearman([1, 2, 2, 3, 4], [10, 20, 20, 30, 40])
    assert abs(rho - 1.0) < 1e-9 and n == 5, f"rho {rho} n {n}"
    print("  [PASS] Spearman handles ties (perfect monotone -> 1.0).")

    print("SELF-TEST: ALL PASSED" if ok else "SELF-TEST: FAILURES ABOVE")
    return ok


# ===========================================================================
# closing narrative (Outputs 3 + 4)
# ===========================================================================

def closing(a_out):
    print("\n" + "=" * 78)
    print("OUTPUT 3 -- RECOMMENDED CHECKPOINT SPACING FOR THE 5-FOLD RUN")
    print("=" * 78)
    print(f"""The fold-0 curves are at ~300-500 step spacing, giving only ~5 checkpoints in
the {WINDOW_LO}-{WINDOW_HI} window -- enough for a flatness verdict, NOT for a 20-point
median (R3 as adapted from FixMatch/FlexMatch). [RB]
  * If R1 (last) or R2 (fixed-plateau) is chosen: current ~500-step spacing is already
    sufficient (a single checkpoint), no change needed. [EC]
  * If R3 (median) is chosen: to resolve a ~20-checkpoint window median, log every ~100
    steps across {WINDOW_LO}-{WINDOW_HI} (=> ~21 checkpoints), or every ~250 steps across
    the whole 0-{BUDGET} budget (=> ~32 checkpoints, ~8 in-window). [EC]
  * COST: the fold-0 hook eval is feature-space CPU (val + fold-0 pos + shared negatives
    only; curve_hook.py). Going from ~16 to ~32 evals/fold roughly DOUBLES the curve-eval
    overhead per fold, which is minutes -- negligible next to training. The recommendation
    is ~250-step spacing over the full budget so whichever rule wins is well-resolved at
    trivial cost. [EC]  [UNVERIFIED] exact minutes not benchmarked here.""")

    print("\n" + "=" * 78)
    print("OUTPUT 4 -- ONE-PARAGRAPH STATEMENT FOR PROJECT.md 7.6")
    print("=" * 78)
    tag = a_out["verdict"] if a_out else "n/a"
    print(f"""This analysis tests, on fold 0 ONLY (a single disclosed use of fold-0 test
labels, then frozen for folds 1-4), whether the blind checkpoint-selection rules R1/R2/R3
are valid -- i.e. whether the TPR@1e-3 curve is flat inside the locked {WINDOW_LO}-{WINDOW_HI}
window. It establishes a per-method flatness verdict (this run: {tag}), the fold-0 TPR each
rule would select and their rule-vs-rule differences relative to the fold-0 bootstrap CI,
and whether label-free source-val purity@50%-completeness is a live selector or a saturated
dead one. It does NOT establish anything about folds 1-4 (untouched), does NOT tune any
hyperparameter, and does NOT license using fold-0 labels again. Rule differences are judged
against the ~98-positive bootstrap CI, so 'indistinguishable' means within that fold's
noise, not identical.""")


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run dir or shorthand {baseline,dann,freematch}")
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(_HERE),
                                                        "data", "feature_cache"))
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--fold", type=int, default=0,
                    help="which fold's curve column + test-positive split to analyse "
                         "(default 0). Sets both --tpr-col (fold{k}_tpr_1e-3) and the A5 "
                         "yardstick split (test_pos_fold{k}) together, so they can never "
                         "disagree -- a fold-1 column judged against fold-0 positives "
                         "would be a silent cross-fold mismatch.")
    ap.add_argument("--tpr-col", default=None,
                    help="explicit curve column override (default: fold{--fold}_tpr_1e-3)")
    ap.add_argument("--self-test", action="store_true", help="run mock/adversarial checks and exit")
    ap.add_argument("--saved-scores", action="store_true",
                    help="take the A5 noise yardstick from the run's SAVED scores/*.npz "
                         "instead of recomputing from the feature cache. REQUIRED for a "
                         "partially-unfrozen encoder (n_blocks>0), where the cache is stale "
                         "and the recompute would pair this head with a different encoder's "
                         "features. Task C is skipped for the same reason.")
    args = ap.parse_args(argv)

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not args.run:
        ap.error("provide a run dir/shorthand, or --self-test")

    run_dir = _resolve_run_dir(args.run)
    cache = FeatureCache(os.path.abspath(args.cache_dir))
    df = _load_curve(run_dir)
    fold = int(args.fold)
    tpr_col = args.tpr_col or f"fold{fold}_tpr_1e-3"
    pos_split = f"test_pos_fold{fold}"

    print("#" * 78)
    print(f"# CHECKPOINT-SELECTION ANALYSIS  (fold {fold}: {tpr_col} vs {pos_split})")
    print(f"# run      : {run_dir}")
    print(f"# cache    : {cache.dir}")
    print(f"# curve    : {len(df)} rows, steps {int(df[STEP_COL].min())}-"
          f"{int(df[STEP_COL].max())}, target_fpr={args.target_fpr:g}")
    print(f"# budget   : {BUDGET} steps; window {WINDOW_LO}-{WINDOW_HI} (final {int(WINDOW_FRAC*100)}%)")
    print("#" * 78)

    a_out = task_a(df, cache, run_dir, pos_split, "test_negatives_shared",
                   args.target_fpr, tpr_col=tpr_col,
                   saved_scores_only=args.saved_scores)
    ci_half = a_out.get("ci_half", np.nan) if a_out else np.nan
    task_b(a_out, args.target_fpr, ci_half, tpr_col=tpr_col)
    if args.saved_scores:
        # Task C recomputes source-val purity from the CACHED val features + each saved
        # head. With an unfrozen encoder those features belong to a different model, so the
        # numbers would be meaningless. Skipped explicitly rather than silently reported.
        print("\n" + "=" * 78)
        print("TASK C -- SKIPPED (--saved-scores): source-val purity is recomputed from the")
        print("frozen feature cache, which is INVALID for an unfrozen encoder. Reporting")
        print("nothing here rather than a number computed against the wrong features.")
        print("=" * 78)
    else:
        task_c(df, cache, run_dir, args.target_fpr, tpr_col=tpr_col)
    closing(a_out)


if __name__ == "__main__":
    main()
