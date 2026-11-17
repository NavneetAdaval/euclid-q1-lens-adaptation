r"""
fig_f3_forest_v2.py -- Figure 3 (rewrite v1): the paired contrasts on one axis.
==============================================================================

    python paper/scripts/fig_f3_forest_v2.py

The same contrasts and intervals as ``fig_f3_forest.py`` (``_common.CONTRAST_ROWS`` through
``_common.contrast``), in the row order of Table 3 (``make_main_tables_v2.ORDER``), with the
pre-registered rows marked by a dagger and the pre-declared sensitivity readouts by a double
dagger (``_common.CONTRAST_PREREG``). Filled markers are budget-matched; open markers are
not. The grey band is one positive (1/98). The last row is the frozen-to-unfrozen FreeMatch
comparison under its own estimator (fold-aware paired bootstrap), set apart by a rule.

``fig_f3_forest.py`` is unchanged. Writes ``paper/figures/F3_forest_v2.pdf`` and nothing else.
"""
from __future__ import annotations

import json
import os

import _common as C
from make_main_tables_v2 import ORDER

PLAIN = {0: r"$\lambda$=0 $-$ Baseline A", 1: r"DANN $-$ $\lambda$=0",
         2: r"CORAL $-$ $\lambda$=0", 3: r"FreeMatch $-$ $\lambda$=0", 4: r"BNM $-$ $\lambda$=0",
         5: r"CORAL $-$ $\lambda$=0", 6: r"FreeMatch $-$ $\lambda$=0", 7: "BNM $-$ CORAL",
         8: "BNM $-$ DANN", 9: "BNM $-$ FreeMatch", 10: "DANN $-$ CORAL",
         11: "DANN $-$ FreeMatch", 12: "DANN $-$ CORAL"}
MARK = {C.PREREG: r"$^{\dagger}$", C.SENSITIVITY: r"$^{\ddagger}$", C.NEITHER: ""}
TITLE = {"A": "Frozen against unfrozen, no target signal",
         "B": "A target signal against none", "C": "Objective against objective"}
ONE_POSITIVE = 1.0 / 98.0


def main() -> int:
    plt = C.use_paper_style()
    items, positions, panels = [], [], []
    y, last = 0.0, None
    with C.pool("D1"):
        for i in ORDER:
            panel, _l, a, ra, b, rb, match, _k = C.CONTRAST_ROWS[i]
            if panel != last:
                y -= 0.85
                panels.append((y, TITLE[panel]))
                last = panel
            r = C.contrast(a, ra, b, rb)
            y -= 1.0
            positions.append(y)
            items.append((f"{PLAIN[i]}{MARK[C.CONTRAST_PREREG[i]]}   ({ra} / {rb})",
                          r["mean"], r["ci_lo"], r["ci_hi"], match == C.MATCHED,
                          C.ARM_COLOUR[a], "t"))
        fb = json.load(open(C.art("foldmean_bootstrap"), encoding="utf-8"))
    sep_y = y - 0.42
    y -= 0.85
    panels.append((y, "Frozen against unfrozen, same objective (fold-aware paired bootstrap)"))
    y -= 1.0
    positions.append(y)
    items.append(("FreeMatch unfrozen $-$ frozen   (1500 / 8000)", fb["diff"], fb["ci"][0],
                  fb["ci"][1], False, C.ARM_COLOUR["freematch"], "boot"))

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axvspan(-ONE_POSITIVE, ONE_POSITIVE, color="#BBBBBB", alpha=0.35, lw=0, zorder=0)
    ax.axvline(0.0, color="#444444", lw=0.9, zorder=1)
    for (label, m, lo, hi, matched, colour, kind), yp in zip(items, positions):
        ax.plot([lo, hi], [yp, yp], color=colour, lw=1.4, zorder=3,
                linestyle="-" if matched else (0, (3, 2)), solid_capstyle="butt")
        for xe in (lo, hi):
            ax.plot([xe, xe], [yp - 0.16, yp + 0.16], color=colour, lw=1.1, zorder=3)
        ax.plot([m], [yp], marker="D" if kind == "boot" else "o", ms=4.4,
                markerfacecolor=colour if matched else "white",
                markeredgecolor=colour, markeredgewidth=1.2, zorder=4)
    ax.set_yticks(positions)
    ax.set_yticklabels([it[0] for it in items], fontsize=7.0)
    for (py, title) in panels:
        ax.text(-0.345, py, title, fontsize=7.2, fontstyle="italic", fontweight="bold",
                va="center", ha="left", color="#333333")
    ax.axhline(sep_y, color="#999999", lw=0.6, linestyle=(0, (4, 3)))
    ax.set_xlim(-0.35, 0.6)
    ax.set_ylim(min(positions) - 0.9, 0.2)
    ax.set_xlabel(u"paired difference in TPR at FPR${=}10^{-3}$   (folds 1–4, cleaned pool)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], color="#555555", lw=1.4, marker="o", ms=4.4,
                   markerfacecolor="#555555", label="budget-matched"),
        plt.Line2D([], [], color="#555555", lw=1.4, linestyle=(0, (3, 2)), marker="o",
                   ms=4.4, markerfacecolor="white", label="budget-unmatched"),
        plt.Line2D([], [], color="#BBBBBB", lw=7, alpha=0.6, label="one positive ($1/98$)"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.46),
              frameon=False, fontsize=7.0, handlelength=2.2, borderaxespad=0.0)
    C.save_fig(fig, os.path.join(C.FIGURES, "F3_forest_v2.pdf"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
