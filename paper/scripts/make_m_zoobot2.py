r"""
make_m_zoobot2.py -- the Zoobot 2.0 pretraining-set size, as a macro.
=====================================================================

    python paper/scripts/make_m_zoobot2.py

The number is a LITERATURE value, not a run artifact, so it cannot come from ``runs/`` and
PROJECT.md does not quote it (the cross-check has nothing to compare it against). It is
therefore transcribed once, with its source, into ``paper/data/zoobot2_pretraining.json``
(walmsley2024scaling, Appendix C.1), and this script asserts before emitting:

  * the six per-campaign counts (each rounded to the nearest thousand) sum to the reported
    total within the rounding of six counts (|sum - total| <= 3k);
  * no campaign is from Euclid.

Writes ``paper/tables/M_zoobot2.tex``.
"""
from __future__ import annotations

import json
import os

import _common as C

SRC = os.path.join(C.DATA, "zoobot2_pretraining.json")


def main() -> int:
    d = json.load(open(SRC, encoding="utf-8"))
    parts = d["campaigns_k"]
    total = int(d["total_reported_k"])
    assert len(parts) == 6, parts
    assert abs(sum(parts.values()) - total) <= 0.5 * len(parts), (sum(parts.values()), total)
    assert d["euclid_campaigns"] == 0
    assert not any("euclid" in (k + v).lower() for k, v in d["telescopes"].items())
    macros = "\n".join([
        rf"\newcommand{{\zoobotPretrainImages}}{{{total}\,000}}",
        rf"\newcommand{{\zoobotPretrainCampaigns}}{{{len(parts)}}}",
    ])
    C.write_table(os.path.join(C.TABLES, "M_zoobot2.tex"), macros,
                  header_comment="Macros for Section 4's pretraining sentence; transcribed "
                                 "literature values in paper/data/zoobot2_pretraining.json.",
                  size="")
    print(f"  Zoobot 2.0 pretraining: {total}k images reported, per-campaign sum "
          f"{sum(parts.values())}k over {len(parts)} campaigns, none Euclid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
