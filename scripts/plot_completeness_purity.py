"""
plot_completeness_purity.py
===========================
The completeness-vs-purity curve for every arm, with Lines et al.'s operating point marked.

    python -u scripts/plot_completeness_purity.py --out runs/round7/completeness_purity.png

WHY A CURVE AND NOT A POINT. `completeness@24%-purity` is **undefined on 6 of 12 arm x
readout cells** (2) -- including frozen Baseline A on 3 of its 5 folds and DANN x nb3 at its
own declared budget -- and where it is defined it swings by 4x across folds of one arm and
inverts under the read-off choice. **No single operating point can carry it.** A curve lets a
reader compare at any operating point instead of at one chosen after the fact, and the n/a
cells stop being holes in a table: they become **curve endpoints that fall short of the 24%
line**, which is what they actually are.

WHAT IS PLOTTED. For each arm, at its quoted readout, per fold:

    x = purity, PREVALENCE-CORRECTED to N_N/N_P = 2200 (Lines' Q1 rarity)
    y = completeness (= TPR = recall)

The curve is the **precision-recall ENVELOPE** -- env(c) = max purity attainable at
completeness >= c -- which is the same object `metrics._completeness_at_purity_from_curve`
integrates to produce the tabulated number, so the plot and the table cannot disagree. The
envelope helper is IMPORTED, not reimplemented.

The vertical reference at purity = 0.24 and the marker at (0.24, 0.50) are Lines'
sim-only operating point. Where an arm's fold curve never reaches x = 0.24 its endpoint sits
left of that line and the tabulated value for that fold is n/a -- the same fact, drawn.

TPR@FPR=1e-3 remains the project's primary claim; this figure and
`runs/round7/lines_ratios.json` are the secondary Lines comparison.

Writes NOTHING outside --out (the .png, and a .json of the plotted summary beside it).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

import metrics as me                                   # noqa: E402
from analyze_window_average import ARMS                # noqa: E402
from integrity import verify as _verify_integrity      # noqa: E402
from write_lines_ratios import QUOTED                  # noqa: E402

NN_NP = 2200.0
TARGET_PURITY = me.LINES_SOURCE_ONLY_PURITY            # 0.24
LINES_COMPLETENESS = me.LINES_SOURCE_ONLY_COMPLETENESS  # 0.50

COLOURS = {"CORAL x nb3": "#1f77b4", "FreeMatch x nb3": "#2ca02c",
           "DANN x nb3": "#d62728", "BNM x nb3": "#9467bd",
           "lambda=0": "#8c564b", "Baseline A (frozen)": "#7f7f7f"}
LABELS = {"lambda=0": r"$\lambda=0$ (unfrozen, no DA)",
          "Baseline A (frozen)": "Baseline A (frozen)"}


def envelope(pos, neg):
    """(completeness, corrected-purity) along the precision-recall ENVELOPE.

    The suffix-max construction is the one `metrics._completeness_at_purity_from_curve`
    uses; taking it from the same ROC pass is what keeps the curve and the tabulated
    number consistent by construction.
    """
    y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
    s = np.concatenate([pos, neg])
    fpr, tpr, _ = me.roc_curve_full(y, s)
    pur = me._purity_from_rates(tpr, fpr, NN_NP)
    p = np.where(np.isfinite(pur), pur, -np.inf)
    env = np.maximum.accumulate(p[::-1])[::-1]
    ok = np.isfinite(env) & (env > -np.inf)
    return tpr[ok], env[ok]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    _verify_integrity(where="before plotting the completeness-purity curves")

    n = len(ARMS)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    summary = {"target_purity": TARGET_PURITY, "nn_np_ratio": NN_NP,
               "lines_completeness": LINES_COMPLETENESS,
               "x": "purity, prevalence-corrected to 1:2200",
               "y": "completeness (= TPR = recall)",
               "curve": "precision-recall envelope, same construction as the tabulated value",
               "arms": {}}

    print("=" * 96)
    print("COMPLETENESS vs PURITY -- every arm at its quoted readout")
    print("=" * 96)
    print(f"  {'arm':<22}{'step':>6}  per-fold MAX attainable purity (the curve's right edge)")

    for ax, (name, spec) in zip(axes, ARMS.items()):
        ro = QUOTED[name]
        folds = list(spec["folds"])
        col = COLOURS[name]
        rec = {"readout": ro, "folds": folds, "max_purity_per_fold": {},
               "reaches_24pct_per_fold": {}}

        # The fold-mean is the mean COMPLETENESS at fixed PURITY -- the same quantity
        # the table reports. Averaging purity at fixed completeness is a DIFFERENT
        # statistic whose value at the dashed line would NOT equal the tabulated
        # fold-mean, and a figure that disagreed with its own table would be worse
        # than no figure.
        grid = np.geomspace(1e-3, 1.0, 600)   # common PURITY grid
        stacked = []
        maxes = []
        exact = []      # the exact per-fold completeness@24%, as the table computes it
        for f in folds:
            scores = spec["load"](f)
            if ro not in scores:
                continue
            comp, pur = envelope(*scores[ro])
            ax.plot(pur, comp, color=col, alpha=0.38, lw=1.1)
            mx = float(np.nanmax(pur))
            maxes.append(mx)
            rec["max_purity_per_fold"][str(f)] = mx
            rec["reaches_24pct_per_fold"][str(f)] = bool(mx >= TARGET_PURITY)
            # the EXACT tabulated value for this fold, from the same routine
            # write_lines_ratios.py uses -- a grid interpolation through a step
            # function does not reproduce it and must not be quoted as if it did
            pos_f, neg_f = scores[ro]
            yf = np.concatenate([np.ones(pos_f.size, int), np.zeros(neg_f.size, int)])
            ex = me.completeness_at_purity(yf, np.concatenate([pos_f, neg_f]),
                                           TARGET_PURITY, nn_np_ratio=NN_NP)
            exact.append(float(ex["completeness"]))
            # completeness as a function of purity. The envelope is monotone
            # non-increasing in completeness, so purity is too and the inverse is
            # well defined; np.interp needs increasing x, hence the sort.
            o = np.argsort(pur)
            stacked.append(np.interp(grid, pur[o], comp[o], left=np.nan, right=np.nan))

        if stacked:
            M = np.vstack(stacked)
            # the fold-mean curve is only drawn where EVERY fold is defined -- an average
            # over a shrinking subset of folds is the same error the n/a rule forbids
            full = ~np.isnan(M).any(axis=0)
            if full.any():
                mean_curve = M[:, full].mean(axis=0)
                ax.plot(grid[full], mean_curve, color=col, lw=2.6,
                        label="fold-mean (all folds defined)")
                lo, hi = grid[full].min(), grid[full].max()
                # NOTE: this is the DRAWN curve read at 0.24. It is a grid
                # interpolation through a step function and is close to, but not
                # identical with, the tabulated value marked separately below.
                rec["drawn_curve_at_24pct"] = (
                    float(np.interp(TARGET_PURITY, grid[full], mean_curve))
                    if lo <= TARGET_PURITY <= hi else None)
            else:
                rec["drawn_curve_at_24pct"] = None

        # the TABULATED fold-mean, marked exactly where the table says it is
        rec["exact_completeness_at_24pct_per_fold"] = exact
        if exact and all(np.isfinite(exact)):
            fm = float(np.mean(exact))
            rec["exact_fold_mean_at_24pct"] = fm
            ax.plot([TARGET_PURITY], [fm], marker="o", ms=7, mfc="white", mec=col,
                    mew=2.0, zorder=6, label=f"tabulated fold-mean {fm:.3f}")
        else:
            rec["exact_fold_mean_at_24pct"] = None

        ax.axvline(TARGET_PURITY, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.plot([TARGET_PURITY], [LINES_COMPLETENESS], marker="*", ms=13,
                color="k", zorder=5, label="Lines et al. sim-only")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.05)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25, lw=0.5)
        reach = sum(rec["reaches_24pct_per_fold"].values())
        ax.set_title(f"{LABELS.get(name, name)}  (step {ro})\n"
                     f"{reach}/{len(folds)} folds reach 24% purity", fontsize=9.5)
        ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
        ax.tick_params(labelbottom=True)   # every panel carries an xlabel

        summary["arms"][name] = rec
        print(f"  {name:<22}{ro:>6}  " +
              " ".join(f"{m:.3f}" for m in maxes) +
              f"   -> {reach}/{len(folds)} reach {TARGET_PURITY}")

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        ax.set_xlabel("purity (corrected to $N_N/N_P=2200$)", fontsize=9)
    for i in range(0, n, ncol):
        axes[i].set_ylabel("completeness", fontsize=9)

    fig.suptitle(
        "Completeness vs purity, per fold, at each arm's quoted readout\n"
        "dashed line = Lines et al.'s 24% purity; star = their sim-only "
        "(24% purity, 50% completeness)\n"
        "SECONDARY comparison -- the primary claim is TPR@FPR=1e-3. A fold whose "
        "curve ends left of the dashed line is an n/a cell in the table.",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    side = os.path.splitext(out)[0] + ".json"
    with open(side, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)

    print(f"\n[out] {os.path.relpath(out, REPO)}")
    print(f"[out] {os.path.relpath(side, REPO)}")
    _verify_integrity(where="after plotting the completeness-purity curves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
