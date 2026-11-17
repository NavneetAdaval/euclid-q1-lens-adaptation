r"""
make_rewrite_v4_macros.py -- macros for the values rewrite v4 adds to the text.
===============================================================================

    python paper/scripts/make_rewrite_v4_macros.py

  * ``\rwAboveN``   -- the negatives above the FPR=1e-3 threshold in any readout-by-fold
                       cell (Section 6.6), from audit/review_r1/out/A2_magnitude_bias.json
                       (read through make_review_macros.review, which re-checks the recorded
                       input hashes); the same file's frac_fainter_22p5 is \rvAboveFaintPct;
  * ``\rwDTwoWord`` -- the number of test lenses that also appear in training or validation
                       as negatives, as a capitalised word (Section 5.5), from
                       _common.pool_sets() (the exact-ID audit's D2 removal set).

Writes ``paper/tables/M_rewrite4.tex`` and nothing else.
"""
from __future__ import annotations

import os

import _common as C
import make_review_macros as RV

WORDS = {20: "Twenty"}


def build():
    mb = RV.review(RV.R1, "A2_magnitude_bias.json")
    above = mb["above_threshold_unique_objects"]
    assert above["n"] == above["n_with_mag"], "a magnitude is missing for an above-threshold negative"
    assert abs(100 * above["frac_fainter_22p5"] - 34) < 0.5, "the 34% no longer belongs to this set"
    n_d2 = len(C.pool_sets()["pos_remove"])
    return [("rwAboveN", str(above["n"])), ("rwDTwoWord", WORDS[n_d2])]


def main() -> int:
    C.write_table(os.path.join(C.TABLES, "M_rewrite4.tex"),
                  "\n".join(rf"\newcommand{{\{n}}}{{{v}}}" for n, v in build()),
                  header_comment="Macros added by rewrite v4 (make_rewrite_v4_macros.py).",
                  size="")
    print("  wrote tables/M_rewrite4.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
