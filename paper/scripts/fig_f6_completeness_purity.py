r"""
fig_f6_completeness_purity.py -- F6, completeness vs purity, as vector PDF.
===========================================================================

    python paper/scripts/fig_f6_completeness_purity.py

**A WRAPPER, NOT A REIMPLEMENTATION.** The figure already exists as
``runs/round7/completeness_purity.png``; this invokes the project's own
``scripts/plot_completeness_purity.py`` with a ``.pdf`` destination, which is all that
was needed -- matplotlib picks the vector backend from the extension. **The underlying
computation is untouched**, which matters because the plotted fold-mean marker is
asserted equal to the tabulated value by the project's own test (``test_round7.T28``),
with ``T29`` asserting that an arm whose metric is n/a carries no marker at all. A second
plotting path could disagree with T4 and nobody would see it.

What the figure is for, and why it replaced a point estimate: **the arms' curves cross.**
FreeMatch $\times$ nb3 sits above CORAL $\times$ nb3 across most of the purity range and
falls below it near 24\% purity, so the tabulated ordering at that one operating point is
not a ranking -- it is a read of where one cliff falls relative to one line. That is a
different claim from the resolution limit: it is a real property of the arms and needs no
error bar. The n/a cells stop being holes and become what they are, curve endpoints:
$\lambda=0$'s and frozen Baseline A's fold-mean curves terminate well to the left of the
dashed line, because those arms never reach 24\% purity at any threshold. And every
fold-mean MARKER sits below the star: **no arm's fold-mean reaches Lines et al.'s sim-only
operating point.** CORRECTED 2026-09-19 (PROJECT.md 7.32): this used to read "the star sits
above every arm's CURVE", which is false of the per-fold curves this same figure draws at
alpha 0.38 -- DANN's fold-2 curve reaches 0.6122 at 24\% purity at its declared budget and
0.5510 at its sensitivity readout, both above the 0.50 star. The fold-mean statement stands;
the per-fold one never did.

The wrapped script also writes a ``.json`` of the plotted summary beside its ``--out``,
so ``paper/figures/F6_completeness_purity.json`` appears alongside the PDF. That is the
script's documented behaviour and it is left alone.

A WORKAROUND THAT USED TO LIVE HERE HAS BEEN REMOVED (2026-09-19). ``integrity.py`` used
to resolve the project record as ``PROJECTv6.md``, a filename that no longer existed, so
``verify()`` raised ``FileNotFoundError`` and every analysis script that calls it at
start-up -- including the plotter this wraps -- could not run. This wrapper rebound the
record path for its own call. ``integrity.py`` now DISCOVERS the record instead of naming
it (PROJECT.md 7.31), so the workaround is dead code and is gone; the wrapper is a plain
call again.

Writes into ``paper/figures/`` only.
"""
from __future__ import annotations

import os

import _common as C


def main() -> int:
    out = os.path.join(C.FIGURES, "F6_completeness_purity.pdf")
    os.makedirs(C.FIGURES, exist_ok=True)

    import plot_completeness_purity as pcp
    rc = pcp.main(["--out", out])
    print(f"  wrote {os.path.relpath(out, C.REPO)}")
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
