"""
audit_negatives.py  --  check the existing negative pool against the Lines standard.

Lines (2512.05899 Sect 3.2) requires:
  - TEST negatives = randomly selected Q1 non-lenses, representative, NO ML-FPs.
  - ML false positives (hard negatives) go in TRAINING only, disjoint from the
    random test negatives.
  - Search population (and so all negatives) screened at I_E < 22.5, stars removed.

This script checks the STRUCTURAL parts locally (no network):
  1. partition layout and duplicate object_ids in the shard pool
  2. that splits' partition lists agree with the shards' own 'partitions' array
  3. object_id disjointness: test-neg vs every training role, and ML-FP vs test
     and vs random negatives
  4. relative brightness of test vs train stamps (descriptive only)
The MAGNITUDE FLOOR (I_E < 22.5) is NOT in the shards -> needs a catalogue query
on the object_ids; this script prints the object_ids to check and explains.

Run from repo root:   python scripts\\audit_negatives.py
"""
from __future__ import annotations
import glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(".")
SHARD_DIR = ROOT / "data" / "q1_unlabeled" / "shards"
MLFP_DIR = ROOT / "data" / "ml_fp_cutouts" / "shards"
SPLITS = ROOT / "data" / "splits" / "splits_v5.json"
TEST_ROLE = "partition_4_test_neg"
TRAIN_ROLES = ["partition_0_ssl_pretrain", "partition_1_source_neg_random",
               "partition_1_reserve", "partition_2_uda_target", "partition_3_val_neg"]


def load_pool(shard_dir):
    """object_id -> partition (from the shards' own partitions array), + shard map."""
    oid2part, oid2loc, shard_oids = {}, {}, {}
    files = sorted(glob.glob(str(shard_dir / "*.npz")))
    if not files:
        sys.exit(f"no shards in {shard_dir}")
    dup = 0
    for f in files:
        with np.load(f) as d:
            oids = d["object_ids"]
            parts = d["partitions"] if "partitions" in d.files else np.full(len(oids), -1)
            shard_oids[Path(f).name] = oids
            for i, o in enumerate(oids):
                o = int(o)
                if o in oid2part:
                    dup += 1
                oid2part[o] = int(parts[i])
                oid2loc[o] = (Path(f).name, i)
    return oid2part, oid2loc, shard_oids, dup, len(files)


def resolve(splits, role, shard_oids):
    """role location list [shard, idx] -> set of object_ids."""
    out = set()
    locs = splits.get("unlabeled_euclid", {}).get(role)
    if locs is None:
        return None
    for name, idx in locs:
        arr = shard_oids.get(name)
        if arr is not None and idx < len(arr):
            out.add(int(arr[idx]))
    return out


def main():
    print(f"SHARD_DIR: {SHARD_DIR.resolve()}\n")
    oid2part, oid2loc, shard_oids, dup, nfiles = load_pool(SHARD_DIR)
    print(f"{nfiles} shards, {len(oid2part):,} unique object_ids, {dup} duplicate object_ids")
    pc = Counter(oid2part.values())
    print("partition counts (from shards' 'partitions' array):",
          {k: pc[k] for k in sorted(pc)})

    if not SPLITS.exists():
        sys.exit(f"\nmissing {SPLITS}")
    splits = json.load(open(SPLITS))
    print("\nsplits top-level keys:", list(splits.keys()))
    print("unlabeled_euclid roles:", list(splits.get("unlabeled_euclid", {}).keys()))

    # resolve roles to object_id sets
    test_ids = resolve(splits, TEST_ROLE, shard_oids)
    if test_ids is None:
        sys.exit(f"\nrole {TEST_ROLE} not in splits; tell me the right key.")
    print(f"\n{TEST_ROLE}: {len(test_ids):,} object_ids")

    # 2) do splits' test-role ids actually carry partition 4 in the shards?
    mismatched = sum(1 for o in test_ids if oid2part.get(o) != 4)
    print(f"  of which NOT tagged partition 4 in shards: {mismatched} "
          f"({'OK - consistent' if mismatched == 0 else 'MISMATCH - splits and shards disagree'})")

    # 3) disjointness: test vs each training role
    print("\nObject_id overlap of TEST negatives with training roles (want 0):")
    for r in TRAIN_ROLES:
        ids = resolve(splits, r, shard_oids)
        if ids is None:
            print(f"  {r:32s} (absent)")
            continue
        print(f"  {r:32s} n={len(ids):>7,}  overlap_with_test={len(test_ids & ids)}")

    # 4) ML-FPs: separate pool, must be disjoint from test (and ideally from random negs)
    mlfp_ids = set()
    mf_files = sorted(glob.glob(str(MLFP_DIR / "*.npz")))
    for f in mf_files:
        with np.load(f) as d:
            if "object_ids" in d.files:
                mlfp_ids |= set(int(x) for x in d["object_ids"])
    print(f"\nML-FP pool: {len(mlfp_ids):,} object_ids from {len(mf_files)} shards")
    if mlfp_ids:
        print(f"  ML-FP ∩ TEST negatives = {len(mlfp_ids & test_ids)}  "
              f"({'OK - Lines requires test has NO ML-FPs' if not (mlfp_ids & test_ids) else 'VIOLATION'})")
        p1 = resolve(splits, 'partition_1_source_neg_random', shard_oids) or set()
        print(f"  ML-FP ∩ random source negs = {len(mlfp_ids & p1)}  "
              f"({'OK - kept disjoint' if not (mlfp_ids & p1) else 'overlap - check'})")

    # 5) relative brightness (descriptive; NOT magnitude). sample test vs a train role.
    def sample_flux(ids, k=300):
        ids = list(ids)[:k]
        tot = []
        by_shard = defaultdict(list)
        for o in ids:
            nm, i = oid2loc[o]; by_shard[nm].append(i)
        for nm, idxs in by_shard.items():
            with np.load(SHARD_DIR / nm) as d:
                im = d["images"]
                for i in idxs:
                    tot.append(float(np.clip(im[i], 0, None).sum()))
        return np.array(tot)
    ft = sample_flux(test_ids)
    p1ids = resolve(splits, 'partition_1_source_neg_random', shard_oids) or set()
    fr = sample_flux(p1ids) if p1ids else np.array([])
    print("\nRelative brightness (total positive flux per stamp; descriptive only):")
    print(f"  test  p10/50/90 = {np.percentile(ft,[10,50,90]).round(3) if len(ft) else 'n/a'}")
    if len(fr):
        print(f"  train p10/50/90 = {np.percentile(fr,[10,50,90]).round(3)}")

    # what still needs the catalogue
    print("\n--- MAGNITUDE FLOOR (I_E < 22.5) NOT checkable here ---")
    print("Shards carry no magnitude. To verify the pool sits at I_E < 22.5 (Lines'")
    print("screening cut), query the MER catalogue for these object_ids and read I_E.")
    print("Sample test-neg object_ids to query:", list(test_ids)[:10])


if __name__ == "__main__":
    main()
