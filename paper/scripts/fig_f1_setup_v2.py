r"""
fig_f1_setup_v2.py -- Figure 1, the setup schematic (rewrite v1).
==================================================================

    python paper/scripts/fig_f1_setup_v2.py

The same schematic as ``fig_f1_setup.py``, drawn from the same counts in
``data/splits/splits.json``. Only the wording changes, to match the text:

  * the target box says what the labels do there: they are not training targets, but they
    choose which images form the pool and they balance each target batch 1:1;
  * the test box no longer calls evaluation the only place labels are used;
  * the fold-rotation note states the fold overlap instead of claiming the fold models are
    uncorrelated: "Target negatives are drawn fresh for each fold; the target lenses of any
    two folds overlap by about 75%." The 75% is computed here from the splits file
    (shared target lenses of two folds over one fold's target lenses, rounded to 5%).

``fig_f1_setup.py`` is left unchanged and still writes ``F1_setup.pdf``. Writes
``paper/figures/F1_setup_v2.pdf`` and nothing else.
"""
from __future__ import annotations

import itertools
import json
import os

import _common as C

TITLE_FS, BODY_FS = 8.0, 6.9
LINE_H = 4.4          # y-units per body line
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
    ax.set_ylim(row2 - row2_h - 2.0, 100.0)

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

    _check_geometry(fig)
    C.save_fig(fig, os.path.join(C.FIGURES, "F1_setup_v2.pdf"))
    plt.close(fig)
    return 0


GEOMETRY_REF = os.path.join(C.DATA, "F1_setup_v2_geometry_ref.json")
OLD_SEES_LINE = "adaptation arms: a loss term computed on the target images"
NEW_SEES_LINES = ("adaptation arms: a loss term computed on the target images,",
                  "whose batches are 1:1 lenses and galaxies")


def _geometry(fig):
    """Every patch, arrow and text of the figure in data coordinates, plus each axes' limits."""
    from matplotlib.patches import FancyArrowPatch
    out = []
    for ai, ax in enumerate(fig.axes):
        out.append({"axes": ai, "kind": "limits", "xlim": [round(v, 6) for v in ax.get_xlim()],
                    "ylim": [round(v, 6) for v in ax.get_ylim()]})
        for p in ax.patches:
            if isinstance(p, FancyArrowPatch):
                (x0, y0), (x1, y1) = p._posA_posB
                out.append({"axes": ai, "kind": "arrow", "xy": [round(x0, 6), round(y0, 6),
                                                                 round(x1, 6), round(y1, 6)]})
            else:
                out.append({"axes": ai, "kind": "box", "xywh": [round(p.get_x(), 6), round(p.get_y(), 6),
                                                                round(p.get_width(), 6),
                                                                round(p.get_height(), 6)]})
        for t in ax.texts:
            x, y = t.get_position()
            out.append({"axes": ai, "kind": "text", "xy": [round(x, 6), round(y, 6)],
                        "text": t.get_text(), "fontsize": t.get_fontsize()})
    return out


def _check_geometry(fig):
    """Rewrite v5b, item 11: only the text changed. Against the geometry captured from the
    generator before the change (paper/data/F1_setup_v2_geometry_ref.json, same pool), the
    figure must be identical except that (a) the "WHAT EACH ARM SEES" box is one LINE_H taller,
    its top fixed; (b) its last line reads NEW_SEES_LINES[0] at the old line's position and
    NEW_SEES_LINES[1] sits one LINE_H below; (c) the top panel's lower y-limit is one LINE_H
    lower. Every other patch, arrow, text and limit is unchanged."""
    import copy
    ref = json.load(open(GEOMETRY_REF, encoding="utf-8"))
    if ref["pool"] != C.POOL:
        return                       # the reference was captured on the default (D1) pool
    exp = copy.deepcopy(ref["geometry"])
    box_i = [i for i, r in enumerate(exp) if r["kind"] == "box" and r["axes"] == 0
             and r["xywh"][0] == 36.5 and r["xywh"][2] == 63.5]
    txt_i = [i for i, r in enumerate(exp) if r["kind"] == "text" and r["text"] == OLD_SEES_LINE]
    lim_i = [i for i, r in enumerate(exp) if r["kind"] == "limits" and r["axes"] == 0]
    assert len(box_i) == len(txt_i) == len(lim_i) == 1, (box_i, txt_i, lim_i)
    b = exp[box_i[0]]["xywh"]
    exp[box_i[0]]["xywh"] = [b[0], round(b[1] - LINE_H, 6), b[2], round(b[3] + LINE_H, 6)]
    t = exp[txt_i[0]]
    t["text"] = NEW_SEES_LINES[0]
    exp.insert(txt_i[0] + 1, dict(t, text=NEW_SEES_LINES[1],
                                  xy=[t["xy"][0], round(t["xy"][1] - LINE_H, 6)]))
    lim = exp[lim_i[0]]["ylim"]
    exp[lim_i[0]]["ylim"] = [round(lim[0] - LINE_H, 6), lim[1]]
    got = _geometry(fig)
    key = lambda r: json.dumps(r, sort_keys=True)            # noqa: E731
    assert sorted(map(key, got)) == sorted(map(key, exp)), \
        "Figure 1 geometry changed beyond the one wrapped line (rewrite v5b, item 11)"
    print("  F1_setup_v2: geometry asserted identical to the pre-v5b reference except the "
          "wrapped line, its box and the panel's lower y-limit")


if __name__ == "__main__":
    raise SystemExit(main())
