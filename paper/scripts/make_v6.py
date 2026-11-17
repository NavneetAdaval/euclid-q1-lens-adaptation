r"""
make_v6.py -- rebuild the paper-v6 additions, then cross-check everything.
=========================================================================

    python paper/scripts/make_all.py      # runs everything, ending with this script
    python paper/scripts/make_v6.py       # or alone: the 1:10 experiment's macros, tables, figures

``make_all.py`` calls this script as its last step (PROJECTv7 7.56), so one command rebuilds
the whole paper. The v6 generators run after the rest because they read tables ``make_all.py``
writes (T3v2_contrasts.tex, A2_hashes_v2.tex, A7_prereg_pool_v2.tex, C_k8_pairs.tex,
M_batch.tex). The last step is ``crosscheck.py``, which now also recomputes every v6 number
(crosscheck_dilute10.py). Nothing here trains anything; the only GPU-free heavy step is
re-scoring stored logits in the cross-check.

Figures carry no build date: SOURCE_DATE_EPOCH (default 0) fixes the CreationDate that
matplotlib writes, so two builds give byte-identical PDFs (PROJECTv7 7.57).
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# reproducible figure PDFs: every generator inherits this (see the docstring)
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
V6 = ["make_dilute10_macros.py", "make_dilute10_tables.py",
      "fig_f1_setup_v6.py", "fig_f3_forest_v6.py", "fig_f7_dilute10.py"]


def main() -> int:
    for s in V6:
        print(f"\n=== {s} " + "=" * (72 - len(s)))
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, s)], cwd=REPO)
        if r.returncode:
            raise SystemExit(f"{s} failed with exit code {r.returncode}")
    print("\n=== crosscheck.py " + "=" * 61)
    return subprocess.run([sys.executable, "-u", os.path.join(HERE, "crosscheck.py")],
                          cwd=REPO).returncode


if __name__ == "__main__":
    raise SystemExit(main())
