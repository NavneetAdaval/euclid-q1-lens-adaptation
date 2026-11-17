"""
extract_neg_ids.py -- pull a random sample of test-negative object_ids to a CSV,
to upload to Datalabs and check against the MER catalogue (magnitude + star/spurious).
The whole pool shares ONE selection function (downloaded once, then hash-partitioned),
so a sample of the test partition characterises the magnitude floor of all of it.
Run:  python scripts\\extract_neg_ids.py
"""
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

SHARD_DIR = Path("data/q1_unlabeled/shards")
SPLITS = Path("data/splits/splits_v5.json")
ROLE = "partition_4_test_neg"
N = 5000
OUT = Path("data/q1_unlabeled/neg_sample_ids.csv")

splits = json.load(open(SPLITS))
locs = splits["unlabeled_euclid"][ROLE]
byf = defaultdict(list)
for nm, idx in locs:
    byf[nm].append(int(idx))
ids = []
for nm, idxs in byf.items():
    with np.load(SHARD_DIR / nm) as d:
        o = d["object_ids"]
        ids.extend(int(o[i]) for i in idxs)
ids = np.array(ids, dtype=np.int64)
rng = np.random.default_rng(0)
samp = rng.choice(ids, size=min(N, len(ids)), replace=False)
pd.DataFrame({"object_id": samp}).to_csv(OUT, index=False)
print(f"{len(ids):,} test-neg ids total; wrote {len(samp):,} sampled to {OUT}")
print("sample:", samp[:5].tolist())
