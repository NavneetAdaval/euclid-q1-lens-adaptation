"""ctrl_scan.py -- scan every new text file under audit/review_r1 for control characters.

    python -B audit/review_r1/scripts/ctrl_scan.py [<out.json>]

Flags any byte < 0x20 other than \\n (0x0A), \\t (0x09), and \\r (0x0D) when it is part of a
\\r\\n line ending. Binary figures (.png) are listed and skipped.
"""
import json
import os
import sys

RR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = {"status": "post_hoc_exploratory",
       "producing_script": "audit/review_r1/scripts/ctrl_scan.py",
       "test_pool": "n/a", "folds": "n/a",
       "rule": "flag bytes < 0x20 except 0x0A, 0x09, and 0x0D followed by 0x0A",
       "scanned": {}, "skipped_binary": [], "files_with_control_chars": {}}
for dp, dn, fn in os.walk(RR):
    for f in sorted(fn):
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, RR).replace("\\", "/")
        if f.lower().endswith(".png"):
            res["skipped_binary"].append(rel)
            continue
        data = open(p, "rb").read()
        bad = []
        for i, c in enumerate(data):
            if c < 0x20 and c not in (0x0A, 0x09):
                if c == 0x0D and i + 1 < len(data) and data[i + 1] == 0x0A:
                    continue
                bad.append((i, c))
        res["scanned"][rel] = len(data)
        if bad:
            res["files_with_control_chars"][rel] = [[i, hex(c)] for i, c in bad[:50]]
res["n_scanned"] = len(res["scanned"])
res["n_flagged"] = len(res["files_with_control_chars"])
print("scanned", res["n_scanned"], "text files; flagged", res["n_flagged"],
      "; skipped", len(res["skipped_binary"]), "png")
for k, v in res["files_with_control_chars"].items():
    print("  ", k, v[:5])
if len(sys.argv) > 1:
    json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
