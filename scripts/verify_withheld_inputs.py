r"""
verify_withheld_inputs.py -- check your own copies of the inputs the release does not redistribute.
===================================================================================================

    python scripts/verify_withheld_inputs.py FOLDER [--list paper/data/not_redistributed.json]

``paper/data/not_redistributed.json`` lists every input that the post hoc review analyses recorded
by sha256 and that the public release leaves out (Euclid Q1 cutouts, the simulated lenses painted
onto them, the deflector cutouts, catalogue files, two superseded drafts). FOLDER is a directory
laid out like this repository (``FOLDER/data/...``, ``FOLDER/data_old/...``); each listed path is
looked up under it, hashed, and reported as

    match     the file is there and its sha256 equals the recorded one
    differs   the file is there but its bytes are not the ones the analysis read
    missing   the file is not there

Read-only. Exit code 0 only if every entry matches. Copying the matching files into the
repository at the same paths makes ``paper/scripts/make_review_macros.py`` verify them instead of
accepting them as withheld.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("folder", help="your copy of the withheld files, laid out like this repository")
    ap.add_argument("--list", default=os.path.join(REPO, "paper", "data", "not_redistributed.json"))
    ap.add_argument("--quiet", action="store_true", help="print only the summary and non-matches")
    a = ap.parse_args(argv)
    entries = json.load(open(a.list, encoding="utf-8"))["entries"]
    status = collections.Counter()
    by_cat = collections.defaultdict(collections.Counter)
    for e in entries:
        p = os.path.join(a.folder, *e["path"].split("/"))
        s = "missing" if not os.path.isfile(p) else ("match" if sha256(p) == e["sha256"] else "differs")
        status[s] += 1
        by_cat[e["category"]][s] += 1
        if s != "match" or not a.quiet:
            print(f"{s:8s} {e['path']}")
    print(f"\n{len(entries)} withheld inputs: {status['match']} match, {status['differs']} differ, "
          f"{status['missing']} missing")
    for c in sorted(by_cat):
        print(f"  {c:17s} " + ", ".join(f"{k} {v}" for k, v in sorted(by_cat[c].items())))
    return 0 if status["match"] == len(entries) else 1


if __name__ == "__main__":
    sys.exit(main())
