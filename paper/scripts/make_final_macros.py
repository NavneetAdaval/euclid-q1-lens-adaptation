r"""
make_final_macros.py -- the cross-check total the acknowledgements quote.
=========================================================================

    python paper/scripts/make_final_macros.py

The final build (brief_final_build.md, PROJECTv7 7.61) quotes the number of automated checks
in the acknowledgements:

  * ``\nCrossChecks`` -- the number of checks ``crosscheck.py`` runs, read from the last
                         ``paper/data/crosscheck.json``.

The count includes the check that verifies this macro (``crosscheck.build_final``, section
``FINAL``), so where the stored report predates that check the count is the stored total plus
one. The cross-check then asserts that the printed value equals its own total, so a stale
value fails the build's cross-check rather than printing.

Writes ``paper/tables/M_final.tex`` and nothing else.
"""
from __future__ import annotations

import json
import os

import _common as C

SECTION = "FINAL"


def main() -> int:
    cc = json.load(open(os.path.join(C.DATA, "crosscheck.json"), encoding="utf-8"))
    n = cc["n_checks"] + (0 if any(r["section"] == SECTION for r in cc["checks"]) else 1)
    C.write_table(os.path.join(C.TABLES, "M_final.tex"),
                  rf"\newcommand{{\nCrossChecks}}{{{n}}}",
                  header_comment=("The acknowledgements' cross-check total, from "
                                  "paper/data/crosscheck.json by "
                                  "paper/scripts/make_final_macros.py; crosscheck.py "
                                  "asserts it equals its own total."), size="")
    print(f"  \\nCrossChecks = {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
