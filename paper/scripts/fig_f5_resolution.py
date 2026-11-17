r"""
fig_f5_resolution.py -- F5, the resolution limit.
=================================================

    python paper/scripts/fig_f5_resolution.py

The project's most transferable finding, in one figure and two panels. Both read the
round-7 artifacts (``window_fold0.json``, ``window_average.json``); nothing is recomputed
here, because §7.16's statistic is a statement about a *specific* fold-0 grid and
recomputing it on a different one would test a different quantity.

**(a) The scatter a single run shows against itself, against the difference the ranking
table is asked to resolve.** The within-run checkpoint sd on fold 0's identical
32-checkpoint grid, divided by $|\text{DANN}-\text{CORAL}| = 0.018$. At $k=1$ it is
2.8--5.1$\times$; under the declared 8-checkpoint score average it falls to
1.4--3.3$\times$ and **goes below $1\times$ on no arm.** The $1\times$ line is the whole
point of the panel: above it, a ranking built from one point estimate per method has no
resolving power. The fix is a larger effect or more positives, not a different read-off
convention and not another arm.

**(b) Why averaging helps less than it should.** Achieved noise reduction
$\mathrm{sd}_1/\mathrm{sd}_k$ against the $\sqrt{k}$ an independence model predicts.
$\lambda=0$ and the frozen Baseline A reach only 0.56--0.59 of it, so **the checkpoint
noise is autocorrelated** -- eight checkpoints are worth nearer two or three independent
draws. CORAL has no $k=8$ point because its grid holds exactly eight checkpoints, so
there is one window position and no spread to measure; that absence is drawn as an
absence. FreeMatch exceeds $\sqrt{k}$, which an independence model cannot produce; on six
window positions that is one unstable estimator and is not read as a result.

Writes ``paper/figures/F5_resolution.pdf`` and nothing else.
"""
from __future__ import annotations

import json
import math
import os
import re

import _common as C

KS = [1, 2, 4, 8]

# fold-0 arm names in window_fold0.json -> (display label, colour key)
F0 = {
    "DANN arm 2 (lambda live)": (r"DANN, fold-0 dev ($\lambda$ live)", "dann"),
    "FreeMatch x nb3": ("FreeMatch", "freematch"),
    "BNM x nb3": ("BNM", "bnm"),
    "CORAL x nb3": ("CORAL", "coral"),
    "DANN arm 1 (lambda=0)": (r"$\lambda{=}0$ control", "lambda0"),
}
SR = {
    "FreeMatch x nb3": ("FreeMatch", "freematch"),
    "BNM x nb3": ("BNM", "bnm"),
    "DANN x nb3": ("DANN", "dann"),
    "CORAL x nb3": ("CORAL", "coral"),
    "lambda=0": (r"$\lambda{=}0$", "lambda0"),
    "Baseline A (frozen)": ("Baseline A", "baseline"),
}


def main() -> int:
    plt = C.use_paper_style()
    w0 = json.load(open(C.art("window_fold0"), encoding="utf-8"))
    wa = json.load(open(C.art("window_average"), encoding="utf-8"))
    # Baseline A's scatter needs its per-step scores before step 8000, which exist only as
    # head checkpoints and are not recomputed per pool: its (b) line is read from the
    # pre-registered-pool artifact, and the caption says so.
    wa_orig = json.load(open(C.ART["window_average"], encoding="utf-8"))
    sr = dict(wa["scatter_reduction"])
    sr.setdefault("Baseline A (frozen)", wa_orig["scatter_reduction"]["Baseline A (frozen)"])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.75))

    # ---- (a) within-run scatter as a multiple of the between-method difference ----
    at8 = []
    for name, (label, ck) in F0.items():
        rec = w0["arms"][name]
        ys = [rec[str(k)]["multiple_of_dann_minus_coral"] for k in KS]
        at8.append(ys[-1])
        ax.plot(KS, ys, marker=C.ARM_MARKER[ck], ms=4.0, lw=1.2,
                color=C.ARM_COLOUR[ck], label=label)
    lo8, hi8 = min(at8), max(at8)
    # The band's label prints the range the prose quotes (\wzEightMultMinTwo{},
    # \wzEightMultMaxTwo{} in tables/M_prose.tex, written earlier in make_all.py), at the
    # same two decimals; a disagreement between the figure and the text stops the build.
    mp = open(os.path.join(C.TABLES, "M_prose.tex"), encoding="utf-8").read()
    for name, val in (("wzEightMultMinTwo", lo8), ("wzEightMultMaxTwo", hi8)):
        got = re.search(r"\\newcommand\{\\" + name + r"\}\{\$?([0-9.]+)\$?\}", mp).group(1)
        assert got == f"{val:.2f}", (name, got, val)
    ax.axhspan(lo8, hi8, color="#BBBBBB", alpha=0.22, lw=0, zorder=0)
    ax.axhline(1.0, color="#000000", lw=1.0, linestyle=(0, (4, 2)), zorder=2)
    # Under the line, clear of the arms' markers at k = 8.
    ax.text(1.05, 0.80, "the between-method difference itself", fontsize=6.6,
            color="#333333", va="top")
    ax.text(8.0, (lo8 + hi8) / 2, f"  {lo8:.2f}–{hi8:.2f}$\\times$\n  at $k{{=}}8$",
            fontsize=6.8, color="#555555", va="center", ha="left")

    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(0.92, 13.5)
    ax.set_ylim(0, 5.6)
    ax.set_xlabel("checkpoints averaged, $k$")
    ax.set_ylabel(r"within-run sd $/\ |$DANN $-$ CORAL$|$")
    ax.grid(color="#EEEEEE", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.6,
              labelspacing=0.28, borderaxespad=0.2)
    # The title states the reading only where it holds on this pool: every arm's scatter
    # above the between-method difference at every k.
    allk = [w0["arms"][n][str(k)]["multiple_of_dann_minus_coral"] for n in F0 for k in KS]
    # Kept short enough not to run into panel (b)'s title.
    title = ("(a)  the ranking is below its own noise" if min(allk) > 1.0 else
             "(a)  within-run scatter over the method gap")
    ax.set_title(title, fontsize=7.6, loc="left", pad=4)

    # ---- (b) achieved reduction vs the sqrt(k) an independence model predicts -------
    kk = [1, 2, 4, 8]
    bx.plot(kk, [math.sqrt(k) for k in kk], color="#000000", lw=1.0,
            linestyle=(0, (4, 2)), zorder=2, label=r"$\sqrt{k}$ (independent draws)")
    for name, (label, ck) in SR.items():
        rec = sr[name]
        xs = [k for k in kk if str(k) in rec]
        ys = [rec[str(k)]["ratio"] for k in xs]
        bx.plot(xs, ys, marker=C.ARM_MARKER[ck], ms=4.0, lw=1.2,
                color=C.ARM_COLOUR[ck], label=label)

    worst = sorted((sr[n]["8"]["achieved_over_expected"], lab)
                   for n, (lab, _) in SR.items()
                   if "8" in sr[n])[:2]
    bx.text(9.6, 1.15,
            "achieved $/\\ \\sqrt{k}$ at $k{=}8$:\n"
            + "\n".join(f"{lab}  {v:.2f}" for v, lab in worst)
            + "\n$\\Rightarrow$ the noise is\nautocorrelated",
            fontsize=6.6, color="#333333", va="bottom", ha="left")

    bx.set_xscale("log", base=2)
    bx.set_xticks(kk)
    bx.set_xticklabels([str(k) for k in kk])
    bx.set_xlim(0.92, 22)
    bx.set_ylim(0.6, 3.7)
    bx.set_xlabel("checkpoints averaged, $k$")
    bx.set_ylabel(r"noise reduction $\mathrm{sd}_1/\mathrm{sd}_k$")
    bx.grid(color="#EEEEEE", lw=0.5)
    bx.set_axisbelow(True)
    bx.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.6,
              labelspacing=0.28, borderaxespad=0.2)
    bx.set_title("(b)  averaging under-delivers", fontsize=7.6, loc="left", pad=4)

    fig.subplots_adjust(wspace=0.33)
    C.save_fig(fig, os.path.join(C.FIGURES, "F5_resolution.pdf"))
    plt.close(fig)
    print(f"  k=8 ratio band {lo8:.2f}-{hi8:.2f}x; sqrt-k shortfall "
          + ", ".join(f"{lab} {v:.2f}" for v, lab in worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
