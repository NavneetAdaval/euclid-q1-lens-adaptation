r"""
fig_f7_dilute10.py -- the new figure of paper v6: the gain over the control, 1:1 against 1:10.
===============================================================================================

    python paper/scripts/fig_f7_dilute10.py

Per arm, two paired t-intervals against the same lambda=0 control (read at 4000): balanced -
lambda=0 (the paper's primary contrast, Table 2) and diluted - lambda=0 (G, the 1:10
experiment's primary contrast). Zero and the one-positive band are marked, in the style of
Figure 3. Both come from ``runs/dilute10_analysis/results.json`` (``balanced_minus_lambda0``,
whose reproduction of Table 2 the analysis checked exactly, and ``G``).

Writes ``paper/figures/F7_dilute10.pdf`` and nothing else.
"""
from __future__ import annotations

import os

import _common as C
import _dilute10 as X

ONE_POSITIVE = 1.0 / 98.0


def main() -> int:
    plt = C.use_paper_style()
    assert X.results()["reproduction"]["passed"]
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    ax.axvspan(-ONE_POSITIVE, ONE_POSITIVE, color="#BBBBBB", alpha=0.35, lw=0, zorder=0)
    ax.axvline(0.0, color="#444444", lw=0.9, zorder=1)
    ticks, labels = [], []
    y = 0.0
    for arm, _n, label, _m in X.ARMS:
        b, g = X.bal_minus_l0(arm), X.G(arm)
        ro = X.read_at(arm)
        assert g["lambda0_at"] == 4000 and g["diluted_at"] == ro
        colour = C.ARM_COLOUR[arm]
        for r, yy, filled, tag in ((b, y, True, "1:1 target"), (g, y - 0.9, False, "1:10 target")):
            ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=colour, lw=1.4, zorder=3,
                    solid_capstyle="butt")
            for xe in (r["ci_lo"], r["ci_hi"]):
                ax.plot([xe, xe], [yy - 0.16, yy + 0.16], color=colour, lw=1.1, zorder=3)
            ax.plot([r["mean"]], [yy], marker="o", ms=4.6,
                    markerfacecolor=colour if filled else "white",
                    markeredgecolor=colour, markeredgewidth=1.2, zorder=4)
            ticks.append(yy)
            labels.append(f"{label} ({ro} / 4000), {tag}")
        y -= 2.4
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_xlim(-0.35, 0.65)
    ax.set_ylim(min(ticks) - 0.8, 0.8)
    ax.set_xlabel(u"arm $-$ $\\lambda$=0 in TPR at FPR${=}10^{-3}$   (folds 1–4, cleaned pool)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], color="#555555", lw=1.4, marker="o", ms=4.6,
                   markerfacecolor="#555555", label="1:1 target, balanced with the labels"),
        plt.Line2D([], [], color="#555555", lw=1.4, marker="o", ms=4.6,
                   markerfacecolor="white", label="1:10 target, batches drawn without the labels"),
        plt.Line2D([], [], color="#BBBBBB", lw=7, alpha=0.6, label="one positive ($1/98$)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, fontsize=7.0, handlelength=2.2, borderaxespad=0.2)
    C.save_fig(fig, os.path.join(C.FIGURES, "F7_dilute10.pdf"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
