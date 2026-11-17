r"""
fig_f1_setup_v6.py -- Figure 1 (paper v6): the setup schematic, with the 1:10 target.
====================================================================================

    python paper/scripts/fig_f1_setup_v6.py

``fig_f1_setup_v2.py`` (unchanged; it still writes ``F1_setup_v2.pdf``) with one addition: a
third row holding the 1:10 target of the second experiment, its counts read from the pool
file ``data/splits/dilute10_pool.json`` (391 lenses + 3910 galaxies; 392 + 3920 on fold 4).
Its text says what the labels do there: they choose the pool, and the batches are drawn
uniformly, without them. The figure grows by that row; every length unit keeps its physical
size, so the first two rows are drawn exactly as in v2.

Writes ``paper/figures/F1_setup_v6.pdf`` and nothing else.
"""
from __future__ import annotations

import itertools
import json
import os

import _common as C
import _dilute10 as X

TITLE_FS, BODY_FS = 8.0, 6.9
LINE_H = 4.4          # y-units per body line
V2_FIG_H = 3.55       # v2's figure height (inches); v6 grows from it
FIG_H = V2_FIG_H
TITLE_H = 5.6         # y-units for the title row
PAD_TOP, PAD_BOT = 3.0, 2.6


def overlap_pct(folds) -> int:
    """Mean share of a fold's target lenses that another fold's target pool also holds,
    rounded to the nearest 5 per cent (the text says 'about')."""
    tl = {f["fold"]: set(f["target_lens_rows"]) for f in folds}
    shares = [len(tl[a] & tl[b]) / len(tl[a]) for a, b in itertools.permutations(tl, 2)]
    pct = 100 * sum(shares) / len(shares)
    out = int(5 * round(pct / 5))
    assert out == 75, pct
    return out


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
    ov = overlap_pct(folds)

    fig = plt.figure(figsize=(6.6, FIG_H))
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
    src_bot = box(ax, 0, 31, TOP, "SOURCE — labelled", [
        f"sim lens positives   {n_src_pos:,}",
        f"grade-X hard negatives   {n_mlfp:,}",
        f"bare deflectors   {n_bare:,}",
        f"random galaxies   {n_rand:,}",
        "balanced 1:1, fixed across folds",
    ], O["blue"], "#EAF2FA")

    box(ax, 36.5, 28, TOP, "TARGET — no label as target", [
        f"{n_tgt_pos[0]} real lenses +",
        f"{n_tgt_neg[0]} fresh random galaxies",
        "labels pick the pool and",
        "balance each batch 1:1",
        "galaxies drawn fresh per fold",
    ], O["vermillion"], "#FCF0E8")

    if C.POOL == "orig":
        test_lines = [f"{n_testneg:,} shared random galaxies",
                      "labels used to score", "the held-out fold"]
    else:
        n_removed = len(C.pool_sets()["neg_remove"])
        n_kept = sum(1 for o in sp["test_negatives_shared"]["object_ids"]
                     if int(o) not in C.pool_sets()["neg_remove"])
        assert n_testneg - n_kept == n_removed, (n_testneg, n_kept, n_removed)
        test_lines = [f"{n_kept:,} shared random galaxies",
                      f"({n_testneg:,} drawn, less {n_removed} excluded)",
                      "labels used to score",
                      "the held-out fold"]
    box(ax, 69.5, 30.5, TOP, "TEST — per fold", [
        f"{n_test[0]} held-out real lenses",
        *test_lines,
    ], O["green"], "#E8F5F0")

    arrow(ax, 31.4, 72, 36.1, 72, O["blue"])
    ax.text(33.75, 74.5, "train", ha="center", fontsize=6.4, color=O["blue"])
    arrow(ax, 64.9, 72, 69.1, 72, "#7A7A7A", ls=(0, (2, 2)))
    ax.text(67.0, 74.5, "score", ha="center", fontsize=6.4, color="#7A7A7A")

    row2 = src_bot - 9.0
    box(ax, 0, 31, row2, "SOURCE-VAL — labelled", [
        f"{n_val_pos:,} sim positives, {n_val_neg:,} negatives",
        "(no grade X); the only selection",
        "signal without target labels",
    ], O["blue"], "#F4F8FC")
    # Rewrite v5b (item 11): the adaptation-arms line now says the target batches are 1:1. At
    # 76.7 x-units it does not fit the 63.5-unit box on one line, so it wraps onto a fifth line;
    # the box (and the panel's lower y-limit) grow by one LINE_H. _check_geometry asserts that
    # nothing else moved.
    sees = [
        "every arm: source images and source labels",
        r"$\lambda=0$ control: target images are loaded, but no target",
        "gradient reaches the encoder or the head",
        "adaptation arms: a loss term computed on the target images,",
        "whose batches are 1:1 lenses and galaxies",
    ]
    box(ax, 36.5, 63.5, row2, "WHAT EACH ARM SEES", sees, "#444444", "#F2F2F2")
    arrow(ax, 15.5, src_bot, 15.5, row2, O["blue"])
    row2_h = TITLE_H + LINE_H * len(sees) + PAD_TOP + PAD_BOT
    v2_low = row2 - row2_h - 2.0                      # v2's lower y-limit

    # ---- paper v6: the 1:10 target of the second experiment, a third row ----
    P = X.pool()
    pf = {f["fold"]: f for f in P["folds"]}
    nl = [pf[k]["n_lenses"] for k in X.FOLDS]
    ngal = [pf[k]["n_negatives"] for k in X.FOLDS]
    assert nl[:3] == [nl[0]] * 3 and ngal[:3] == [ngal[0]] * 3
    assert nl[0] == n_tgt_pos[1] and all(g == P["ratio_galaxies_per_lens"] * l
                                         for l, g in zip(nl, ngal))
    row3 = row2 - row2_h - 7.0
    dil_lines = [
        f"{nl[0]} real lenses + {ngal[0]:,} random galaxies "
        f"({nl[3]} + {ngal[3]:,} on fold 4): the balanced pool's",
        "galaxies plus fresh draws, up to exactly ten per lens",
        "labels choose the pool; batches drawn uniformly, without the labels",
    ]
    box(ax, 36.5, 63.5, row3, "TARGET, 1:10 — second experiment", dil_lines,
        O["vermillion"], "#FDF6F1")
    row3_h = TITLE_H + LINE_H * len(dil_lines) + PAD_TOP + PAD_BOT
    new_low = row3 - row3_h - 2.0
    ax.set_ylim(new_low, 100.0)
    # keep every y-unit at v2's physical size: the top axes grow by the added range
    v2_ax_in = 0.585 * V2_FIG_H
    ax_in = v2_ax_in * (100.0 - new_low) / (100.0 - v2_low)
    bx_in, gap_in = 0.40 * V2_FIG_H, 0.015 * V2_FIG_H
    H = bx_in + gap_in + ax_in
    fig.set_size_inches(6.6, H)
    ax.set_position([0.0, (bx_in + gap_in) / H, 1.0, ax_in / H])
    bx.set_position([0.0, 0.0, 1.0, bx_in / H])

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
                    f"test {n_test[j]}" if is_test else "target",
                    ha="center", va="center", fontsize=6.2,
                    color="white" if is_test else "#5A3218", zorder=2)
    bx.text(x0 + 2.5 * (cw + gap) - gap / 2, 0.0,
            "Target negatives are drawn fresh for each fold; the target lenses of any two "
            f"folds overlap by about {ov}%.\nTest negatives are the same shared pool in every "
            "fold (one FPR$\\,{=}\\,10^{-3}$ threshold).",
            ha="center", va="bottom", fontsize=6.4, color="#555555", linespacing=1.4)

    C.save_fig(fig, os.path.join(C.FIGURES, "F1_setup_v6.pdf"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
