r"""
make_F5_resolution_v2.py -- Figure 5 (rewrite v1): gaps between methods against the noise
a single checkpoint carries.
=========================================================================================

    python paper/scripts/make_F5_resolution_v2.py

For every gap between two adaptation arms whose two steps are declared readouts (the rule of
Section 6.4), one row: the bar is noise_b = sqrt(sigma_A^2 + sigma_B^2) / 2, the contribution
of checkpoint-to-checkpoint scatter to a difference of four-fold means, with sigma the folds
1-4 mean of each arm's linear-trend residual sd (estimator b); the point is |gap|, the
difference of the two fold-means. Log axis. The gaps among CORAL, DANN and FreeMatch are
grouped above the gaps between BNM and each of them.

Source: audit/review_r1/out/A1_resolution.json (gaps_and_ratios), read through
make_review_macros.review, which re-checks every recorded input hash. No window-averaged
quantity is used. Writes ``paper/figures/F5_resolution_v2.pdf`` and nothing else.
"""
from __future__ import annotations

import os

import _common as C
import make_review_macros as RV

NAME = {"dann": "DANN", "coral": "CORAL", "freematch": "FreeMatch", "bnm": "BNM"}


def rows():
    a1, three, bnm, nb = RV.a1_sets()
    out = []
    for group, d in (("three", three), ("bnm", bnm)):
        seen = set()
        for k, g in d.items():
            pair = k.split("|")[0]
            steps = tuple(g["steps"])
            if (pair, steps) in seen:
                continue
            seen.add((pair, steps))
            a, b = pair.split("-")
            out.append((group, f"{NAME[a]} $-$ {NAME[b]}   ({steps[0]} / {steps[1]})",
                        abs(g["gap"]), nb(g), a))
    return out


def main() -> int:
    plt = C.use_paper_style()
    rr = rows()
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ys, labels = [], []
    y = 0.0
    last = None
    sep = None
    for group, label, gap, noise, arm in rr:
        if last is not None and group != last:
            y -= 0.6
            sep = y + 0.3
        last = group
        y -= 1.0
        ys.append(y)
        labels.append(label)
        ax.plot([0.002, noise], [y, y], color="#BBBBBB", lw=6.0, solid_capstyle="butt",
                zorder=1)
        ax.plot([gap], [y], marker="o", ms=5.0, color=C.ARM_COLOUR[arm],
                markeredgecolor="#222222", markeredgewidth=0.6, zorder=3)
    if sep is not None:
        ax.axhline(sep, color="#999999", lw=0.6, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xlim(0.002, 0.3)
    ax.set_ylim(min(ys) - 0.8, 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("difference in TPR at FPR$=10^{-3}$ (folds 1–4, cleaned pool)")
    ax.grid(axis="x", color="#EEEEEE", lw=0.5, which="both")
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color="#BBBBBB", lw=6.0,
                          label="noise from checkpoint scatter"),
               plt.Line2D([], [], color="#555555", marker="o", lw=0, ms=5.0,
                          label="|difference of fold-means|")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.0)
    C.save_fig(fig, os.path.join(C.FIGURES, "F5_resolution_v2.pdf"))
    plt.close(fig)
    print(f"  {len(rr)} gap rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
