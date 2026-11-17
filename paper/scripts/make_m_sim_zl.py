r"""
make_m_sim_zl.py -- the simulated lenses' deflector-redshift summary, as data and as macros.
===========================================================================================

    python paper/scripts/make_m_sim_zl.py

Reads ONLY the ``z_l`` array from each of the 60 arc-on-real shards (``data/sim_arc_on_real``;
``np.load`` on an ``.npz`` is lazy, so no image array is decompressed) and writes:

  * ``paper/data/sim_zl.json`` -- the artifact: count, percentiles and range over all
    simulated lenses, plus the shard count, so the numbers can be audited without the shards;
  * ``paper/tables/M_sim_zl.tex`` -- LaTeX macros read from that JSON, input by ``main.tex``,
    so prose quotes the number through a macro and never retypes it.

The statistic is over simulated LENSES (30,000), not unique deflectors: a deflector reused
for several lenses counts once per lens, which is the population the model is trained on.
``z_l`` is the deflector's photometric-redshift median, as the generator stored it.
"""
from __future__ import annotations

import json
import os

import numpy as np

import _common as C

OUT_JSON = os.path.join(C.DATA, "sim_zl.json")
OUT_TEX = os.path.join(C.TABLES, "M_sim_zl.tex")
EXPECTED_SHARDS = 60
EXPECTED_LENSES = 30000


def main() -> int:
    paths = C.sim_shards(["z_l"])           # the shards, or their public_inputs extract
    if len(paths) != EXPECTED_SHARDS:
        raise SystemExit(f"expected {EXPECTED_SHARDS} shards, found {len(paths)}")
    zl = []
    for _name, d in paths:
        zl.append(np.asarray(d["z_l"], dtype=float))
    zl = np.concatenate(zl)
    if zl.size != EXPECTED_LENSES:
        raise SystemExit(f"expected {EXPECTED_LENSES} lenses, found {zl.size}")

    p10, p50, p90 = np.percentile(zl, [10, 50, 90])
    rec = {
        "source": "data/sim_arc_on_real/shard_*.npz, key z_l only",
        "n_shards": len(paths),
        "n_lenses": int(zl.size),
        "z_l_p10": float(p10),
        "z_l_median": float(p50),
        "z_l_p90": float(p90),
        "z_l_min": float(zl.min()),
        "z_l_max": float(zl.max()),
        "note": "over simulated lenses, not unique deflectors; z_l is the stored "
                "photometric-redshift median",
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print(f"  wrote {os.path.relpath(OUT_JSON, C.REPO)}")

    body = "\n".join([
        rf"\newcommand{{\simZlMedian}}{{{rec['z_l_median']:.2f}}}",
        rf"\newcommand{{\simZlPTen}}{{{rec['z_l_p10']:.2f}}}",
        rf"\newcommand{{\simZlPNinety}}{{{rec['z_l_p90']:.2f}}}",
    ])
    C.write_table(OUT_TEX, body, header_comment=(
        "Macros: deflector-redshift summary of the 30,000 simulated lenses.\n"
        "Source: paper/data/sim_zl.json (written by this script from the shards)."),
        size="")
    print(f"  z_l over {rec['n_lenses']} lenses: p10 {p10:.3f}, median {p50:.3f}, "
          f"p90 {p90:.3f} (range {rec['z_l_min']:.3f}-{rec['z_l_max']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
