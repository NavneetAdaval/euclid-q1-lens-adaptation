r"""
make_rewrite_v5_macros.py -- macros for the values rewrite v5 adds to the text.
===============================================================================

    python paper/scripts/make_rewrite_v5_macros.py

Section 6.3's head-start sentences (brief_rewrite_v5.md, B3; Part A2):

  * ``\rwFiveFmStart``        -- the folds 1-4 cleaned-pool TPR@1e-3 fold-mean of the frozen
                                 FreeMatch head the unfrozen FreeMatch arm starts from (the
                                 round-1 run's step-8000 head, each unfrozen fold's
                                 ``init_head_from``), from its stored score dumps through
                                 _common.round1_summary;
  * ``\rwFiveFmFrozenPart``   -- that start minus the lambda=0 control (4000): the part of
                                 FreeMatch - lambda=0 gained with the encoder frozen;
  * ``\rwFiveFmUnfrozenPart`` -- unfrozen FreeMatch (1500) minus that start: the part gained
                                 after unfreezing (the licensed pair's fold-mean difference);
  * ``\rwFiveCoralStart``     -- the same start for CORAL (round-1 frozen CORAL, step 8000).

Each also has a four-decimal form (suffix ``Four``). Asserted before writing: the starts are the
heads every unfrozen fold's config names; the two parts sum exactly to FreeMatch - lambda=0;
the unfrozen part equals the licensed pair's difference; and the two-decimal parts sum to the
two-decimal pre-registered contrast the sentence prints (\dCtFmLamP).

Writes ``paper/tables/M_rewrite5.tex`` and nothing else. Opens no checkpoint.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

import _common as C

FOLDS = (1, 2, 3, 4)


def _fm(arm, ro):
    r = C.fold_mean(arm, "tpr", ro)
    return float(r[0] if isinstance(r, tuple) else r)


def _start(run):
    s = C.round1_summary(run, folds=FOLDS)
    return float(np.mean([s["per_fold"][f]["tpr"] for f in FOLDS]))


def _check_init_heads():
    fm = sorted(glob.glob(C.R("runs", "freematch_nb3_unfrozen_fold[1-4]", "run_config.json")))
    co = sorted(glob.glob(C.R("kaggle", "coral_r4", "runs", "main", "extracted",
                              "coral_r4_confirm", "fold*", "run.json")))
    assert len(fm) == len(co) == 4, (fm, co)
    for f, p in zip(FOLDS, fm):
        h = json.load(open(p, encoding="utf-8"))["init_head_from"]
        assert h.endswith(f"runs/bakeoff_freematch_5fold/fold{f}/checkpoints/step_8000.pt"), h
    for f, p in zip(FOLDS, co):
        h = json.load(open(p, encoding="utf-8"))["init_head_from"]
        assert h.endswith(f"runs/bakeoff_coral/fold{f}/checkpoints/step_8000.pt"), h


def _printed(macro):
    for p in glob.glob(os.path.join(C.TABLES, "M_*.tex")):
        m = re.search(r"\\newcommand\{\\" + macro + r"\}\{([^}]*)\}", open(p, encoding="utf-8").read())
        if m:
            return m.group(1)
    raise AssertionError(f"\\{macro} not found in tables/M_*.tex")


def build():
    _check_init_heads()
    fm_start = _start("bakeoff_freematch_5fold")
    co_start = _start("bakeoff_coral")
    u, lam = _fm("freematch", 1500), _fm("lambda0", 4000)
    frozen_part, unfrozen_part = fm_start - lam, u - fm_start
    assert abs((u - lam) - (frozen_part + unfrozen_part)) < 1e-12
    assert round(unfrozen_part, 4) == round(u - fm_start, 4)
    two = lambda x: f"{x:.2f}"                                    # noqa: E731
    four = lambda x: f"{x:.4f}"                                   # noqa: E731
    # The sentence prints "<frozen part> of its <dCtFmLamP> ... and <unfrozen part>": the
    # two printed parts must add up to the printed whole.
    whole = float(_printed("dCtFmLamP"))
    assert abs(float(two(frozen_part)) + float(two(unfrozen_part)) - whole) < 1e-9, \
        (two(frozen_part), two(unfrozen_part), whole)
    return [("rwFiveFmStart", two(fm_start)), ("rwFiveFmStartFour", four(fm_start)),
            ("rwFiveFmFrozenPart", two(frozen_part)), ("rwFiveFmFrozenPartFour", four(frozen_part)),
            ("rwFiveFmUnfrozenPart", two(unfrozen_part)),
            ("rwFiveFmUnfrozenPartFour", four(unfrozen_part)),
            ("rwFiveCoralStart", two(co_start)), ("rwFiveCoralStartFour", four(co_start))]


# Rewrite v5b, item 12: the paper's number style. Four-digit numbers take no separator; five or
# more digits are grouped with a thin space. These macros' generators either open a checkpoint
# (make_a5_restart.py; make_all.py keeps its committed output) or are protected byte for byte by
# verify_orig_tables.py (make_a4_augprec.py), so the prose uses a style-conforming copy, read
# from the committed file and asserted to be the same integer.
NUMSTYLE = [("M_restart.tex", "restartScheduleSteps"), ("M_restart.tex", "restartControlSteps"),
            ("M_restart.tex", "restartEmaIters"), ("M_restart.tex", "restartMuOffset"),
            ("M_augprec.tex", "augprecTrainable")]


def paper_number(n: int) -> str:
    return str(n) if abs(n) < 10000 else f"{n:,}".replace(",", r"\,")


def numstyle():
    out = []
    for fname, macro in NUMSTYLE:
        txt = open(os.path.join(C.TABLES, fname), encoding="utf-8").read()
        m = re.search(r"\\newcommand\{\\" + macro + r"\}\{((?:\{,\}|[^{}])*)\}", txt)
        assert m, (fname, macro)
        raw = m.group(1)
        n = int(raw.replace("{,}", "").replace(r"\,", ""))
        assert re.fullmatch(r"\d{1,3}(\{,\}\d{3})*", raw), (macro, raw)
        out.append((macro + "Ns", paper_number(n)))
    return out


def main() -> int:
    C.write_table(os.path.join(C.TABLES, "M_rewrite5.tex"),
                  "\n".join(rf"\newcommand{{\{n}}}{{{v}}}" for n, v in build() + numstyle()),
                  header_comment="Macros added by rewrite v5 (make_rewrite_v5_macros.py).",
                  size="")
    print("  wrote tables/M_rewrite5.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
