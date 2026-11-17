"""snapdiff.py -- compare two snapshot.py outputs.

    python -B audit/review_r1/scripts/snapdiff.py <before.json> <after.json> [<out.json>]

Prints (and optionally writes as JSON) files added, removed, content-changed (sha256
differs) and touched-only (same sha256, different mtime).
"""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
added = sorted(set(b) - set(a))
removed = sorted(set(a) - set(b))
changed = sorted(k for k in set(a) & set(b) if a[k][2] != b[k][2])
touched = sorted(k for k in set(a) & set(b) if a[k][2] == b[k][2] and a[k][1] != b[k][1])
outside = [k for k in added if not k.startswith("audit/review_r1/")]
res = {"n_before": len(a), "n_after": len(b),
       "n_added": len(added), "n_added_inside_audit_review_r1": len(added) - len(outside),
       "added_outside_audit_review_r1": outside,
       "removed": removed, "content_changed": changed,
       "mtime_only_same_sha256": touched,
       "added_inside_audit_review_r1": [k for k in added if k.startswith("audit/review_r1/")]}
for k in ("n_before", "n_after", "n_added", "n_added_inside_audit_review_r1"):
    print(k, res[k])
for k in ("added_outside_audit_review_r1", "removed", "content_changed", "mtime_only_same_sha256"):
    print(k, len(res[k]))
    for x in res[k][:200]:
        print("   ", x)
if len(sys.argv) > 3:
    json.dump(res, open(sys.argv[3], "w", encoding="utf-8"), indent=1)
