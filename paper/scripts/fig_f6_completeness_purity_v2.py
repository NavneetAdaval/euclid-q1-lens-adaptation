r"""
fig_f6_completeness_purity_v2.py -- Figure 5 (completeness against purity), rewrite v2 text.
=============================================================================================

    python paper/scripts/fig_f6_completeness_purity_v2.py

The old generator (fig_f6_completeness_purity.py) wraps ``scripts/plot_completeness_purity.py``
unchanged, and that script draws its own titles and legend in the build's vocabulary
("SECONDARY comparison -- ...", "CORAL x nb3"). This generator draws the same data with the
same encoding and changes only the text:

  * no figure-level title (the caption carries the explanation);
  * panel titles "CORAL (step 2000)", ..., each followed by "· n/m folds reach 24% purity";
  * legend entries "fold mean (all folds defined)", "fold mean, as tabulated",
    "benchmark simulation-only point".

How. The plotter is run twice in this process. The first run is untouched and only records the
data of every plotted line (its figure is discarded, written nowhere). The second run's figure
is intercepted at ``savefig``: its texts are replaced and it is written to
``paper/figures/F6_completeness_purity_v2.pdf``. Every plotted array of the second figure is
asserted identical to the first run's, so the change is text only. The old PDF stays on disk.
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure                    # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402
import numpy as np                                      # noqa: E402

import _common as C                                     # noqa: E402

TITLE = {"CORAL x nb3": "CORAL", "FreeMatch x nb3": "FreeMatch", "DANN x nb3": "DANN",
         "lambda=0": r"$\lambda = 0$ control", "BNM x nb3": "BNM",
         "Baseline A (frozen)": "Baseline A, frozen"}
LEGEND = [(re.compile(r"^fold-mean \(all folds defined\)$"), "fold mean (all folds defined)"),
          (re.compile(r"^tabulated fold-mean \d\.\d{3}$"), "fold mean, as tabulated"),
          (re.compile(r"^Lines et al\. sim-only$"), "benchmark simulation-only point")]


def line_data(fig):
    out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for ln in ax.get_lines():
            x, y = ln.get_data()
            out.append((np.asarray(x, float).copy(), np.asarray(y, float).copy(),
                        ln.get_color(), ln.get_marker(), ln.get_linestyle(), ln.get_linewidth(),
                        ln.get_alpha()))
    return out


def run(tmp_out, hook):
    import plot_completeness_purity as pcp
    orig = Figure.savefig

    def patched(self, fname, *a, **k):
        target = hook(self)
        return orig(self, target or fname, *a, **k)
    Figure.savefig = patched
    try:
        rc = pcp.main(["--out", tmp_out])
    finally:
        Figure.savefig = orig
    assert not rc, rc
    return pcp


def main() -> int:
    import tempfile
    os.makedirs(C.FIGURES, exist_ok=True)
    out = os.path.join(C.FIGURES, "F6_completeness_purity_v2.pdf")
    tmp = tempfile.mkdtemp(prefix="f6v2_")
    ref = {}

    def record(fig):
        ref["data"] = line_data(fig)
        return None
    run(os.path.join(tmp, "ref.pdf"), record)

    def retext(fig):
        import plot_completeness_purity as pcp
        from analyze_window_average import ARMS
        from write_lines_ratios import QUOTED
        assert fig._suptitle is not None
        fig._suptitle.remove()
        fig._suptitle = None
        axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(axes) == len(ARMS)
        for ax, name in zip(axes, ARMS):
            m = re.search(r"\n(\d+)/(\d+) folds reach 24% purity$", ax.get_title())
            assert m and f"(step {QUOTED[name]})" in ax.get_title(), ax.get_title()
            ax.set_title(f"{TITLE[name]} (step {QUOTED[name]}) · {m.group(1)}/{m.group(2)} "
                         f"folds reach 24% purity", fontsize=9.5)
            handles = []
            for ln in ax.get_lines():
                lab = ln.get_label()
                if lab.startswith("_"):
                    continue
                new = [n for pat, n in LEGEND if pat.match(lab)]
                assert len(new) == 1, lab
                ln.set_label(new[0])
                handles.append(ln)
            ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)
        fig.tight_layout()
        assert all(len(ax.get_title()) for ax in axes)
        assert len(line_data(fig)) == len(ref["data"])
        for a, b in zip(line_data(fig), ref["data"]):
            assert np.array_equal(a[0], b[0], equal_nan=True) and \
                np.array_equal(a[1], b[1], equal_nan=True) and a[2:] == b[2:], \
                "a plotted array differs from the old generator's"
        return out
    run(out, retext)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f"  wrote {os.path.relpath(out, C.REPO)} (data asserted identical to the old generator's)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
