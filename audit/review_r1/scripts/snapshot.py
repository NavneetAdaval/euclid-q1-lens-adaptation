"""snapshot.py -- sha256 of every file under a root.

    python -B audit/review_r1/scripts/snapshot.py <root> <out.json>

Writes {relpath: [size, mtime_ns, sha256]} to <out.json>. Used for the before/after
snapshots of the review-round-1 brief (rule zero). Reads only.
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

root, out = sys.argv[1], sys.argv[2]


def h(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            d.update(b)
    return d.hexdigest()


paths = []
for dp, dn, fn in os.walk(root):
    for f in fn:
        paths.append(os.path.join(dp, f))


def one(p):
    st = os.stat(p)
    try:
        return os.path.relpath(p, root).replace("\\", "/"), [st.st_size, st.st_mtime_ns, h(p)]
    except OSError as e:
        return os.path.relpath(p, root).replace("\\", "/"), [st.st_size, st.st_mtime_ns,
                                                             "ERR:" + str(e)]


with ThreadPoolExecutor(8) as ex:
    res = dict(ex.map(one, paths))
json.dump(res, open(out, "w"), indent=0, sort_keys=True)
print(len(res), "files", sum(v[0] for v in res.values()) / 1e9, "GB")
