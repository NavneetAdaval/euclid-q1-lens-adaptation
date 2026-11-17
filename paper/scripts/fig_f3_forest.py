r"""
fig_f3_forest.py -- F3, the paired contrasts as a forest plot.
==============================================================

    python paper/scripts/fig_f3_forest.py

T2 rendered on one axis. It reads the SAME declaration T2 does
(``_common.CONTRAST_ROWS``) and recomputes from the same logits, so the figure and the
table are two renderings of one set of numbers and cannot drift apart.

WHAT THE ENCODING CARRIES:

  * **filled marker, solid interval** = budget-matched (both arms read at the same
    step); **open marker, dashed interval** = budget-unmatched, kept only where the
    project record keeps it. A match removes the budget confound and nothing else.
  * the **zero line** is drawn, because ``interval excludes zero'' is the whole verdict;
  * the **one-positive resolution** ($1/98$) is drawn as a shaded band about zero. An
    interval whose bound falls inside it excludes zero but is consistent with an effect
    below the metric's own granularity -- "distinguishable" and "large" are different
    claims and the figure keeps them apart;
  * the last row uses a **different estimator** (fold-aware paired bootstrap, the one it
    was published under) and is separated by a rule and labelled, never mixed in with
    the $t$-intervals above it.

Writes ``paper/figures/F3_forest.pdf`` and nothing else.
"""
from __future__ import annotations

import json
import os

import _common as C

PLAIN = {  # LaTeX-free labels for the figure
    r"$\lambda=0$ $-$ Baseline A": r"$\lambda$=0 $-$ Baseline A",
    r"DANN $-$ $\lambda=0$": r"DANN $-$ $\lambda$=0",
    r"CORAL $-$ $\lambda=0$": r"CORAL $-$ $\lambda$=0",
    r"FreeMatch $-$ $\lambda=0$": r"FreeMatch $-$ $\lambda$=0",
    r"BNM $-$ $\lambda=0$": r"BNM $-$ $\lambda$=0",
    r"CORAL $-$ $\lambda=0$ (at 4000)": r"CORAL $-$ $\lambda$=0 (at 4000)",
    r"FreeMatch $-$ $\lambda=0$ (at 4000)": r"FreeMatch $-$ $\lambda$=0 (at 4000)",
}
ONE_POSITIVE = 1.0 / 98.0


def main() -> int:
    plt = C.use_paper_style()

    items = []          # (label, mean, lo, hi, matched, colour, kind)
    positions = []      # one y per item
    panels = []         # (y, panel title) for the italic panel headers
    y = 0.0
    last_panel = None
    for panel, label, a, ra, b, rb, match, _ in C.CONTRAST_ROWS:
        if panel != last_panel:
            y -= 0.85
            panels.append((y, C.PANEL_TITLE[panel]))
            last_panel = panel
        r = C.contrast(a, ra, b, rb)
        y -= 1.0
        positions.append(y)
        items.append((PLAIN.get(label, label) + f"   ({ra} / {rb})",
                      r["mean"], r["ci_lo"], r["ci_hi"],
                      match == C.MATCHED, C.ARM_COLOUR[a], "t"))

    # The one across-round contrast the project licenses, on the estimator it was
    # published under -- separated by a rule so it is never read as another t-interval.
    fb = json.load(open(C.art("foldmean_bootstrap"), encoding="utf-8"))
    sep_y = y - 0.42
    y -= 0.85
    panels.append((y, "Within-method, frozen $\\rightarrow$ unfrozen "
                      "(different estimator: fold-aware paired bootstrap)"))
    y -= 1.0
    positions.append(y)
    items.append(("FreeMatch nb3 $-$ FreeMatch frozen   (1500 / 8000)",
                  fb["diff"], fb["ci"][0], fb["ci"][1], False,
                  C.ARM_COLOUR["freematch"], "boot"))

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    ax.axvspan(-ONE_POSITIVE, ONE_POSITIVE, color="#BBBBBB", alpha=0.35, lw=0, zorder=0)
    ax.axvline(0.0, color="#444444", lw=0.9, zorder=1)

    for (label, m, lo, hi, matched, colour, kind), yp in zip(items, positions):
        ax.plot([lo, hi], [yp, yp], color=colour, lw=1.4, zorder=3,
                linestyle="-" if matched else (0, (3, 2)),
                solid_capstyle="butt")
        for xe in (lo, hi):
            ax.plot([xe, xe], [yp - 0.16, yp + 0.16], color=colour, lw=1.1, zorder=3)
        ax.plot([m], [yp], marker="D" if kind == "boot" else "o", ms=4.4,
                markerfacecolor=colour if matched else "white",
                markeredgecolor=colour, markeredgewidth=1.2, zorder=4)

    ax.set_yticks(positions)
    ax.set_yticklabels([it[0] for it in items], fontsize=7.0)
    for (py, title) in panels:
        ax.text(-0.155, py, title, fontsize=7.2, fontstyle="italic",
                fontweight="bold", va="center", ha="left", color="#333333")
    ax.axhline(sep_y, color="#999999", lw=0.6, linestyle=(0, (4, 3)))

    ax.set_xlim(-0.16, 0.56)
    ax.set_ylim(min(positions) - 0.9, 0.2)
    ax.set_xlabel(u"paired difference in TPR at FPR${=}10^{-3}$   (folds 1–4, paired)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], color="#555555", lw=1.4, marker="o", ms=4.4,
                   markerfacecolor="#555555", label="budget-matched"),
        plt.Line2D([], [], color="#555555", lw=1.4, linestyle=(0, (3, 2)), marker="o",
                   ms=4.4, markerfacecolor="white", label="budget-unmatched"),
        plt.Line2D([], [], color="#BBBBBB", lw=7, alpha=0.6,
                   label="one positive ($1/98$)"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.46),
              frameon=False, fontsize=7.0, handlelength=2.2, borderaxespad=0.0)

    C.save_fig(fig, os.path.join(C.FIGURES, "F3_forest.pdf"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
