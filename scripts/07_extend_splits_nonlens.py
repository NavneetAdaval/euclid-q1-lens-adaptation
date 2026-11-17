"""07_extend_splits_nonlens.py — add sim NON-LENS negatives to the SOURCE task
AND rebuild the VAL set as a pure in-sim arc task.

EXTENDS the frozen splits.json; does NOT regenerate it. Every evaluation key
(real_lenses_kfold, partition_2/3/4_*, sim_val_shards, ...) is copied through
byte-identical. Two new keys are added:

  source_composition  -> source_train negatives (30k positives : 30k negatives):
      15,000 sim non-lenses + 10,545 random real + 4,455 ML-FP
      (kills most of the sim-vs-real shortcut; forces the head onto the arc)

  val_composition     -> val negatives = sim NON-LENSES (not real P3):
      sim-lens vs sim-non-lens is a PURE arc-detection task with no domain
      shortcut, so sim-val AUROC becomes an honest "did the head learn arcs?"
      signal instead of saturating at 1.0.

The sim non-lens pool is permuted once (seeded) and partitioned DISJOINTLY:
  [:15000]        -> source negatives
  [15000:15000+N] -> val negatives  (default N=9000, matching the 9k sim-val positives)
Re-running with the same --seed reproduces both selections exactly, and the
source selection is identical to the previous v4 run (same first-15k slice).

Writes splits_v5.json (new name so feature caches tagged by filename don't go
stale). Original frozen splits.json is untouched.

  python 07_extend_splits_nonlens.py \
      --splits data/splits/splits.json \
      --nonlens_shards_dir data/sim_nonlens/shards
"""
import argparse
import glob
import json
import os
import zipfile
from pathlib import Path

import numpy as np


def npz_count(path, key="images"):
    try:
        from numpy.lib.format import read_magic, _read_array_header
        with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as f:
            ver = read_magic(f)
            shape, _, _ = _read_array_header(f, ver)
        return int(shape[0])
    except Exception:
        with np.load(path) as d:
            return int(d[key].shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="path to existing (frozen) splits.json")
    ap.add_argument("--nonlens_shards_dir", required=True)
    ap.add_argument("--out", default=None, help="default: splits_v5.json next to input")
    ap.add_argument("--n_sim_nonlens", type=int, default=15000, help="source negatives")
    ap.add_argument("--n_val_nonlens", type=int, default=9000, help="val negatives")
    ap.add_argument("--n_random", type=int, default=10545)
    ap.add_argument("--n_mlfp", type=int, default=-1, help="-1 = use all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = json.load(open(args.splits))
    for k in ["simulated_lenses", "unlabeled_euclid", "ml_fp_cutouts", "real_lenses_kfold"]:
        assert k in splits, f"splits.json missing expected key {k!r} — wrong file?"
    assert "partition_1_source_neg_random" in splits["unlabeled_euclid"]
    assert "sim_train_shards" in splits["simulated_lenses"]
    assert "sim_val_shards" in splits["simulated_lenses"]
    assert "locations" in splits["ml_fp_cutouts"]

    rng = np.random.default_rng(args.seed)

    # --- enumerate sim non-lens shards -> flat [shard, idx] locations ---
    shards = sorted(glob.glob(os.path.join(args.nonlens_shards_dir, "*.npz")))
    assert shards, f"no .npz shards found in {args.nonlens_shards_dir}"
    nl_locs = []
    for sp in shards:
        name = os.path.basename(sp)
        for i in range(npz_count(sp)):
            nl_locs.append((name, i))
    need = args.n_sim_nonlens + args.n_val_nonlens
    assert len(nl_locs) >= need, \
        f"need {need} sim non-lenses (source {args.n_sim_nonlens} + val {args.n_val_nonlens}), " \
        f"only found {len(nl_locs)}"

    # one permutation -> disjoint source / val slices (source slice == previous v4 run)
    perm = rng.permutation(len(nl_locs))
    src_sel = perm[:args.n_sim_nonlens]
    val_sel = perm[args.n_sim_nonlens:args.n_sim_nonlens + args.n_val_nonlens]
    sim_nonlens_src = [[nl_locs[i][0], int(nl_locs[i][1])] for i in src_sel]
    sim_nonlens_val = [[nl_locs[i][0], int(nl_locs[i][1])] for i in val_sel]

    # --- random real subset for source ---
    p1 = splits["unlabeled_euclid"]["partition_1_source_neg_random"]
    assert len(p1) >= args.n_random, f"need {args.n_random} random, have {len(p1)}"
    ridx = rng.permutation(len(p1))[:args.n_random]
    random_sel = [p1[i] for i in ridx]

    # --- ml-fp for source (all by default) ---
    mlfp = splits["ml_fp_cutouts"]["locations"]
    mlfp_sel = list(mlfp) if args.n_mlfp < 0 else [mlfp[i] for i in rng.permutation(len(mlfp))[:args.n_mlfp]]

    out_splits = dict(splits)
    out_splits["schema_version"] = 5
    out_splits["source_composition"] = {
        "seed": args.seed,
        "negatives": {"sim_nonlens": sim_nonlens_src, "random_real": random_sel, "ml_fp": mlfp_sel},
        "counts": {"sim_nonlens": len(sim_nonlens_src), "random_real": len(random_sel),
                   "ml_fp": len(mlfp_sel)},
    }
    out_splits["val_composition"] = {
        "seed": args.seed,
        "negatives": {"sim_nonlens": sim_nonlens_val},
        "counts": {"sim_nonlens": len(sim_nonlens_val)},
        "note": "val = sim-pos vs sim-nonlens (pure arc task, no domain shortcut)",
    }

    out = args.out or str(Path(args.splits).with_name("splits_v5.json"))
    with open(out, "w") as f:
        json.dump(out_splits, f)

    # disjointness check
    overlap = set(map(tuple, sim_nonlens_src)) & set(map(tuple, sim_nonlens_val))
    assert not overlap, "BUG: source and val sim-nonlens overlap"

    print(f"wrote {out}")
    print(f"  source negatives: sim_nonlens={len(sim_nonlens_src)}  random={len(random_sel)}  "
          f"ml_fp={len(mlfp_sel)}  total={len(sim_nonlens_src)+len(random_sel)+len(mlfp_sel)}")
    print(f"  val negatives:    sim_nonlens={len(sim_nonlens_val)}  (disjoint from source)")
    print(f"  eval keys preserved byte-identical; original splits.json untouched")


if __name__ == "__main__":
    main()
