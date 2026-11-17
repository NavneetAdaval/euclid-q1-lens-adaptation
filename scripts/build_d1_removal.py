"""
build_d1_removal.py
===================
Writes ``data/splits/d1_test_neg_remove.json``: the CLEANED test pool's removal list
(``audit/decontam/rule.json``, D1, body sha256 b83fa019...), as the cache/test ids the scoring
code sees (``random:<shard>:<idx>``). It is the same set the paper removes
(``paper/scripts/_common.pool_sets``: exact_ids.contaminant_sets -> d1_neg_remove, 28 lens
positives + 237 simulation hosts = 265), precomputed so a Kaggle run can apply it without
``data_old/``. Reads identities only; no score, label or metric.

    python -u scripts/build_d1_removal.py --out data/splits/d1_test_neg_remove.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and not a.force:
        raise SystemExit(f"refusing to overwrite {a.out}")
    import exact_ids as X
    rule = json.load(open(os.path.join(REPO, "audit", "decontam", "rule.json"), encoding="utf-8"))
    I = X.identities()
    S = X.contaminant_sets(I)
    assert len(S["lens_in_test"]) == 28 and len(S["defl_in_test"]) == 237
    assert len(S["d1_neg_remove"]) == 265
    tn = I["splits"]["test_negatives_shared"]
    locs = [f"random:{s}:{i}" for s, i in tn["locations"]]
    remove = sorted(l for l in locs if I["test_neg_loc2id"][l] in S["d1_neg_remove"])
    assert len(remove) == 265, len(remove)
    body = {"what": "D1 removal list for the cleaned test pool, as cache ids",
            "rule": "audit/decontam/rule.json D1_primary_test_side",
            "rule_body_sha256": rule["sha256_of_body_without_this_field"],
            "n_test_negatives": len(locs), "n_remove": len(remove),
            "n_lens_positives": len(S["lens_in_test"]),
            "n_simulation_hosts": len(S["defl_in_test"]),
            "ids": remove,
            "builder_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read())
            .hexdigest()}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1)
    print(f"[d1] {a.out}: {len(remove)} of {len(locs)} test negatives removed "
          f"(rule {body['rule_body_sha256'][:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
