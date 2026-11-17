r"""
fig_f1_setup.py -- F1, the setup schematic.
===========================================

    python paper/scripts/fig_f1_setup.py

What the four data roles are, how they rotate across folds, and which arm sees what.
**Every count in the figure is read from ``data/splits/splits.json``** -- the single
source of truth the builder writes and self-audits -- so the schematic cannot drift
from the splits it describes.

The three things the figure has to make unambiguous, because the paper's central claim
depends on all three:

  1. **No real lens label ever enters a training loss.** The 489 graded lenses appear in
     the target pool as *unlabelled images* and in the test pool as *labels used at
     evaluation only*. That is the whole meaning of ``label-free'' here.
  2. **The target pool is balanced ~1:1 on purpose**, and that is a theory-driven choice,
     not a convenience: under label shift, marginal-alignment methods provably cannot
     minimise target error, so the target is subsampled to match the source's 1:1 ratio.
     It is also what caps the pool at ~782 images and creates the memorisation exposure
     the paper reports.
  3. **Target negatives are fresh per fold; test negatives are shared.** Opposite choices
     for opposite reasons -- fresh target negatives keep the fold models uncorrelated so
     the error bars are honest, while one shared negative pool keeps the FPR${=}10^{-3}$
     threshold identical across folds, so fold-to-fold variance reflects positive
     recovery rather than negative-tail noise.

Vector PDF, colour-blind-safe, no title inside the figure. Writes
``paper/figures/F1_setup.pdf`` and nothing else.
"""
from __future__ import annotations

import json
import os

import _common as C

TITLE_FS, BODY_FS = 8.0, 6.9
LINE_H = 4.4          # y-units per body line
TITLE_H = 5.6         # y-units for the title row
PAD_TOP, PAD_BOT = 3.0, 2.6


def main() -> int:
    plt = C.use_paper_style()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    sp = json.load(open(C.ART["splits"], encoding="utf-8"))
    st, va = sp["source_train"], sp["val"]
    n_src_pos = len(st["sim_lens_rows"])
    n_mlfp = len(st["neg_mlfp"]["object_ids"])
    n_bare = len(st["neg_bare_deflector"]["deflector_object_ids"])
    n_rand = len(st["neg_random"]["object_ids"])
    n_val_pos = len(va["sim_lens_rows"])
    n_val_neg = (len(va["neg_bare_deflector"]["deflector_object_ids"])
                 + len(va["neg_random"]["object_ids"]))
    n_testneg = len(sp["test_negatives_shared"]["object_ids"])
    folds = sp["real_lenses_kfold"]["folds"]
    n_test = [len(f["test_lens_rows"]) for f in folds]
    n_tgt_pos = [len(f["target_lens_rows"]) for f in folds]
    n_tgt_neg = [len(f["target_neg_object_ids"]) for f in folds]
    n_lenses = sum(n_test)

    fig = plt.figure(figsize=(6.6, 3.55))
    ax = fig.add_axes([0.0, 0.415, 1.0, 0.585])
    bx = fig.add_axes([0.0, 0.0, 1.0, 0.40])
    for a in (ax, bx):
        a.set_xlim(0, 100)
        a.set_ylim(0, 100)
        a.axis("off")

    O = C.OKABE_ITO

    def box(a, x, w, top, title, lines, edge, face):
        h = TITLE_H + LINE_H * len(lines) + PAD_TOP + PAD_BOT
        a.add_patch(FancyBboxPatch(
            (x, top - h), w, h, boxstyle="round,pad=0,rounding_size=1.6",
            linewidth=1.0, edgecolor=edge, facecolor=face, zorder=1))
        a.text(x + w / 2, top - PAD_TOP, title, ha="center", va="top",
               fontsize=TITLE_FS, fontweight="bold", color=edge, zorder=3)
        for i, ln in enumerate(lines):
            a.text(x + w / 2, top - PAD_TOP - TITLE_H - LINE_H * i, ln,
                   ha="center", va="top", fontsize=BODY_FS, color="#222222", zorder=3)
        return top - h

    def arrow(a, x0, y0, x1, y1, colour, ls="-"):
        a.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", linestyle=ls, mutation_scale=8,
            linewidth=1.0, color=colour, shrinkA=0, shrinkB=0, zorder=2))

    TOP = 98.0
    # ---------------- labelled source --------------------------------------------
    src_bot = box(ax, 0, 31, TOP, "SOURCE — labelled", [
        f"sim lens positives   {n_src_pos:,}",
        f"grade-X hard negatives   {n_mlfp:,}",
        f"bare deflectors   {n_bare:,}",
        f"random galaxies   {n_rand:,}",
        "balanced 1:1, fixed across folds",
    ], O["blue"], "#EAF2FA")

    # ---------------- unlabelled target ------------------------------------------
    box(ax, 36.5, 28, TOP, "TARGET — unlabelled", [
        f"{n_tgt_pos[0]} real lenses,",
        "their labels NOT used",
        f"{n_tgt_neg[0]} fresh random galaxies",
        f"$\\approx${n_tgt_pos[0] + n_tgt_neg[0]} images, $\\sim$1:1",
        "drawn fresh for each fold",
    ], O["vermillion"], "#FCF0E8")

    # ---------------- held-out evaluation ----------------------------------------
    # On the paper's pool (D1) the shared test negatives are shown after the exact-ID
    # exclusion of Section 3, with the as-drawn count beside it; on the pre-registered pool
    # (PAPER_POOL=orig) the pool is the one the splits file draws.
    if C.POOL == "orig":
        test_lines = [f"{n_testneg:,} shared random galaxies",
                      "the ONE place real lens", "labels are used —", "at evaluation only"]
    else:
        n_removed = len(C.pool_sets()["neg_remove"])
        n_kept = sum(1 for o in sp["test_negatives_shared"]["object_ids"]
                     if int(o) not in C.pool_sets()["neg_remove"])
        assert n_testneg - n_kept == n_removed, (n_testneg, n_kept, n_removed)
        test_lines = [f"{n_kept:,} shared random galaxies",
                      f"({n_testneg:,} drawn, less {n_removed} excluded)",
                      "the ONE place real lens labels",
                      "are used — at evaluation only"]
    box(ax, 69.5, 30.5, TOP, "TEST — per fold", [
        f"{n_test[0]} held-out real lenses",
        *test_lines,
    ], O["green"], "#E8F5F0")

    arrow(ax, 31.4, 72, 36.1, 72, O["blue"])
    ax.text(33.75, 74.5, "train", ha="center", fontsize=6.4, color=O["blue"])
    arrow(ax, 64.9, 72, 69.1, 72, "#7A7A7A", ls=(0, (2, 2)))
    ax.text(67.0, 74.5, "score", ha="center", fontsize=6.4, color="#7A7A7A")

    # ---------------- source-val + who sees what ---------------------------------
    row2 = src_bot - 9.0
    box(ax, 0, 31, row2, "SOURCE-VAL — labelled", [
        f"{n_val_pos:,} sim positives, {n_val_neg:,} negatives",
        "(no grade X); the label-free proxy",
        "for selection — and it saturates",
    ], O["blue"], "#F4F8FC")
    box(ax, 36.5, 63.5, row2, "WHAT EACH ARM SEES", [
        "every arm: source images and source labels",
        r"$\lambda=0$ control: target images are loaded, but NO target",
        "gradient reaches the encoder or the head",
        "DA arms: a loss term computed on the target images",
    ], "#444444", "#F2F2F2")
    arrow(ax, 15.5, src_bot, 15.5, row2, O["blue"])
    # Trim the axes to what was actually drawn, so no blank band opens up below.
    row2_h = TITLE_H + LINE_H * 4 + PAD_TOP + PAD_BOT     # the taller of the two row-2 boxes
    ax.set_ylim(row2 - row2_h - 2.0, 100.0)

    # ---------------- fold rotation ----------------------------------------------
    bx.text(0, 96, f"FOLD ROTATION over the {n_lenses} graded real lenses — "
                   f"stratified by grade, each lens tested exactly once",
            fontsize=7.4, fontweight="bold", va="top", color="#222222")

    x0, cw, gap, rh = 11.0, 16.6, 1.2, 11.0
    for k in range(5):
        yy = 76.0 - k * (rh + 1.7)
        bx.text(x0 - 1.6, yy + rh / 2, f"fold {k}", ha="right", va="center", fontsize=6.9)
        for j in range(5):
            xx = x0 + j * (cw + gap)
            is_test = (j == k)
            bx.add_patch(FancyBboxPatch(
                (xx, yy), cw, rh, boxstyle="square,pad=0", linewidth=0.5,
                edgecolor="white",
                facecolor=O["green"] if is_test else O["vermillion"],
                alpha=1.0 if is_test else 0.42, zorder=1))
            bx.text(xx + cw / 2, yy + rh / 2,
                    f"test {n_test[j]}" if is_test else "target (unlabelled)",
                    ha="center", va="center", fontsize=6.2,
                    color="white" if is_test else "#5A3218", zorder=2)
    bx.text(x0 + 2.5 * (cw + gap) - gap / 2, 0.0,
            "Target negatives are fresh per fold (honest error bars); test negatives are "
            "the SAME shared pool every fold\n(one common FPR$\\,{=}\\,10^{-3}$ threshold). "
            "Opposite choices, opposite reasons.",
            ha="center", va="bottom", fontsize=6.4, color="#555555", linespacing=1.4)

    C.save_fig(fig, os.path.join(C.FIGURES, "F1_setup.pdf"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
