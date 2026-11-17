"""
plot_curves.py
==============
Read the fold-0 LEARNING-CURVE diagnostic (curve.csv, written by run_baseline.py /
run_freematch.py under --eval-every-steps) and, per method, overlay on a shared STEP axis:

    * fold0_tpr_1e-3     -- the diagnostic real metric (raw, faint) + a ROLLING MEAN
                            (window 3-5 checkpoints; fold-0 TPR on ~98 positives is noisy).
    * source_risk_auroc  -- label-free selector A (source-val AUROC).
    * source_risk_tpr    -- label-free selector B (source-val TPR@1e-3).

CANDIDATE FLATTEN POINT (surfaced, NEVER auto-locked; PROJECT.md 7.4 / Step 4).
For each curve we smooth with the rolling mean, then mark the FIRST step whose smoothed
value comes within 5% of that curve's maximum smoothed value over the whole run
(threshold = 0.95 * max). For the noisy fold0-TPR this is the candidate "the real metric
has plateaued" step. We ALSO compute it for each selector, and report the step of each
selector's raw max, so we can read whether a label-free selector flattens at the SAME step
as the real metric (a usable stopping signal) or much earlier (saturates -> unreliable;
this is exactly the open question in PROJECT.md 7.4). Nothing is written back to configs.

Works on an ARBITRARY run dir -- pass one or more run dirs (or curve.csv paths); the same
code plots the FreeMatch curve unchanged once runs/curve_freematch_fold0/curve.csv lands:

    python scripts/plot_curves.py runs/curve_baseline_fold0
    python scripts/plot_curves.py runs/curve_baseline_fold0 runs/curve_freematch_fold0 \\
        --labels baseline freematch --window 5 --out runs/learning_curves.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # headless: save a PNG, never open a window
import matplotlib.pyplot as plt

# curve.csv columns we consume (must match curve_hook.CURVE_COLUMNS)
STEP = "step"
TPR = "fold0_tpr_1e-3"
SELECTORS = ("source_risk_auroc", "source_risk_tpr")
FLATTEN_FRAC = 0.05                         # "within 5% of the run's max"


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def _resolve_csv(path: str) -> str:
    """Accept a run dir (-> <dir>/curve.csv) or a direct curve.csv path."""
    if os.path.isdir(path):
        return os.path.join(path, "curve.csv")
    return path


def load_curve(csv_path: str) -> dict:
    """Load curve.csv -> {col: np.ndarray}, sorted by step, blanks -> nan."""
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")
    cols = rows[0].keys()

    def col(name):
        return np.array([_f(r.get(name, "")) for r in rows], dtype=float)

    out = {c: col(c) for c in cols}
    order = np.argsort(out[STEP])
    return {c: out[c][order] for c in out}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# smoothing + flatten detection
# ---------------------------------------------------------------------------

def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean, window shrinks at the edges (nan-aware). window>=1."""
    n = len(y)
    h = window // 2
    out = np.full(n, np.nan)
    for i in range(n):
        seg = y[max(0, i - h):min(n, i + h + 1)]
        if np.any(np.isfinite(seg)):
            out[i] = np.nanmean(seg)
    return out


def flatten_step(steps: np.ndarray, y_smooth: np.ndarray, frac: float = FLATTEN_FRAC):
    """First step whose smoothed value is >= (1-frac) * max(smoothed over the whole run).
    Returns (step, value_at_step, run_max) or (None, None, run_max) if no finite data."""
    finite = np.isfinite(y_smooth)
    if not np.any(finite):
        return None, None, np.nan
    run_max = float(np.nanmax(y_smooth))
    thr = (1.0 - frac) * run_max
    hit = np.where(finite & (y_smooth >= thr))[0]
    if hit.size == 0:
        return None, None, run_max
    i = int(hit[0])
    return int(steps[i]), float(y_smooth[i]), run_max


def argmax_step(steps: np.ndarray, y: np.ndarray):
    """Step of the FIRST occurrence of the raw maximum (and that max value)."""
    finite = np.isfinite(y)
    if not np.any(finite):
        return None, None
    m = np.nanmax(y)
    i = int(np.where(finite & (y >= m))[0][0])
    return int(steps[i]), float(y[i])


# ---------------------------------------------------------------------------
# per-method analysis
# ---------------------------------------------------------------------------

def analyze(curve: dict, window: int) -> dict:
    steps = curve[STEP]
    tpr_raw = curve[TPR]
    tpr_smooth = rolling_mean(tpr_raw, window)
    tpr_flat_step, tpr_flat_val, tpr_max = flatten_step(steps, tpr_smooth)

    # typical checkpoint interval -> coincidence tolerance (2 checkpoints)
    dsteps = np.diff(steps)
    interval = float(np.median(dsteps)) if dsteps.size else 0.0
    tol = 2.0 * interval

    selectors = {}
    for name in SELECTORS:
        if name not in curve:
            continue
        sel_smooth = rolling_mean(curve[name], window)
        f_step, f_val, s_max = flatten_step(steps, sel_smooth)
        mx_step, mx_val = argmax_step(steps, curve[name])
        gap = (None if (f_step is None or tpr_flat_step is None)
               else int(f_step - tpr_flat_step))
        coincides = (gap is not None and abs(gap) <= tol)
        selectors[name] = {"flatten_step": f_step, "max_step": mx_step,
                           "max_val": mx_val, "gap_vs_tpr": gap,
                           "coincides": coincides}
    return {"steps": steps, "tpr_raw": tpr_raw, "tpr_smooth": tpr_smooth,
            "tpr_flatten_step": tpr_flat_step, "tpr_flatten_val": tpr_flat_val,
            "tpr_run_max": tpr_max, "interval": interval, "tol": tol,
            "selectors": selectors}


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

_COLORS = {TPR: "#1f77b4", "source_risk_auroc": "#d62728",
           "source_risk_tpr": "#2ca02c"}


def plot_methods(methods: list[tuple[str, dict, dict]], window: int, out_path: str):
    """methods = [(label, curve, analysis), ...]; one subplot per method, shared step axis."""
    n = len(methods)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4.2 * n), squeeze=False)
    for ax, (label, curve, a) in zip(axes[:, 0], methods):
        steps = a["steps"]
        ax.plot(steps, a["tpr_raw"], color=_COLORS[TPR], alpha=0.25, lw=1,
                marker="o", ms=3, label="fold0_tpr_1e-3 (raw)")
        ax.plot(steps, a["tpr_smooth"], color=_COLORS[TPR], lw=2.2,
                label=f"fold0_tpr_1e-3 (rolling w={window})")
        for name in SELECTORS:
            if name in curve:
                ax.plot(steps, curve[name], color=_COLORS[name], lw=1.6,
                        alpha=0.9, label=name)
        # candidate flatten line (fold0-TPR)
        fs = a["tpr_flatten_step"]
        if fs is not None:
            ax.axvline(fs, color=_COLORS[TPR], ls="--", lw=1.4)
            ax.annotate(f"TPR flatten\n@ step {fs}", xy=(fs, a["tpr_flatten_val"]),
                        xytext=(6, -4), textcoords="offset points", fontsize=8,
                        color=_COLORS[TPR], va="top")
        # each selector's flatten (dotted, its own color)
        for name in SELECTORS:
            sel = a["selectors"].get(name)
            if sel and sel["flatten_step"] is not None:
                ax.axvline(sel["flatten_step"], color=_COLORS[name], ls=":", lw=1.2,
                           alpha=0.7)
        ax.set_title(f"{label}  (fold 0; window={window}, flatten=within "
                     f"{int(FLATTEN_FRAC*100)}% of run-max)")
        ax.set_xlabel("gradient step")
        ax.set_ylabel("metric (all in [0,1])")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# reporting (surface numbers; DO NOT auto-lock)
# ---------------------------------------------------------------------------

def _s(x):
    return "n/a" if x is None else str(x)


def report(methods: list[tuple[str, dict, dict]]):
    print("\n" + "=" * 78)
    print("LEARNING-CURVE FLATTEN READ  (fold 0; diagnostic only -- nothing is locked)")
    print("=" * 78)
    for label, _curve, a in methods:
        print(f"\n[{label}]  checkpoint interval ~{a['interval']:.0f} steps  "
              f"(coincidence tolerance +-{a['tol']:.0f} steps)")
        print(f"  fold0-TPR candidate FLATTEN step : {_s(a['tpr_flatten_step'])}"
              f"   (rolling-mean run-max = {a['tpr_run_max']:.4f})")
        for name in SELECTORS:
            sel = a["selectors"].get(name)
            if not sel:
                print(f"  {name:18s}: (column absent)")
                continue
            coin = ("COINCIDES with fold0-TPR flatten" if sel["coincides"]
                    else "does NOT coincide (earlier/later -> weak stopping signal)")
            print(f"  {name:18s}: max @ step {_s(sel['max_step'])} "
                  f"(={sel['max_val']:.4f}) | flatten @ step {_s(sel['flatten_step'])} "
                  f"| gap vs TPR flatten = {_s(sel['gap_vs_tpr'])} steps -> {coin}")
    print("\n" + "=" * 78)
    print("Read these by eye against the plot; the single fixed step budget is a SEPARATE")
    print("decision (Step 4) after seeing BOTH methods' curves. No config/splits touched.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot + flatten-read fold-0 learning curves.")
    p.add_argument("runs", nargs="+",
                   help="one or more run dirs (uses <dir>/curve.csv) or curve.csv paths")
    p.add_argument("--labels", nargs="*", default=None,
                   help="per-run labels (default: run dir basename)")
    p.add_argument("--window", type=int, default=3,
                   help="rolling-mean window in checkpoints (recommended 3-5; default 3)")
    p.add_argument("--out", default=None,
                   help="output PNG (default: <first run dir>/learning_curves.png)")
    args = p.parse_args()

    if args.window < 1:
        print("--window must be >= 1")
        return 2
    if args.labels and len(args.labels) != len(args.runs):
        print(f"--labels ({len(args.labels)}) must match number of runs ({len(args.runs)})")
        return 2

    methods = []
    for i, run in enumerate(args.runs):
        csv_path = _resolve_csv(run)
        if not os.path.exists(csv_path):
            print(f"SKIP: no curve.csv at {csv_path}")
            continue
        label = (args.labels[i] if args.labels
                 else os.path.basename(os.path.normpath(run)) or run)
        curve = load_curve(csv_path)
        if TPR not in curve:
            print(f"SKIP: {csv_path} lacks '{TPR}' column")
            continue
        methods.append((label, curve, analyze(curve, args.window)))

    if not methods:
        print("no usable curve.csv found")
        return 3

    first_dir = args.runs[0]
    first_dir = first_dir if os.path.isdir(first_dir) else os.path.dirname(first_dir)
    out_path = args.out or os.path.join(first_dir or ".", "learning_curves.png")
    plot_methods(methods, args.window, out_path)
    report(methods)
    print(f"\n[out] plot -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
