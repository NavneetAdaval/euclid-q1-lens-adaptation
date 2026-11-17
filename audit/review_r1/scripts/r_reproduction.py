"""
r_reproduction.py -- review round 1: reproduce every arm's Table 2 value before anything new.
POST HOC AND EXPLORATORY (the check itself is exact).

    python -B audit/review_r1/scripts/r_reproduction.py

Table 2 in the compiled paper is tab:main (paper/tables/T1_main.tex); its per-fold values
are paper/tables/T5_perfold.tex. Each arm's headline readout (and DANN's 2000 sensitivity
readout) is recomputed on D1 from the loaders the review scripts use, asserted bit-equal to
``_common.suite``, and compared to the printed tables at 4 decimals.

Writes out/R_reproduction_table2.json.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_common as RC                                   # noqa: E402

SCRIPT = os.path.abspath(__file__)


def main():
    I = RC.Inputs()
    res = RC.reproduce_headlines(I)
    assert res["all_match"], res
    RC.write_json("R_reproduction_table2.json", res, script=SCRIPT, inputs=I, pool="D1",
                  folds=RC.CONF_FOLDS)
    print("  all match:", res["all_match"])


if __name__ == "__main__":
    main()
