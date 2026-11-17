"""
Create the train/val/test splits for the lens-finding pipeline.

Outputs splits.json — the canonical definition consumed by all
downstream training scripts. Splits are defined once, never re-derived.

Design (final, post-Lines and post-ML-FP integration):
  - 5-fold CV across all 326 grade-A+B real lenses (mirrors Lines Sect. 4.5)
  - Sim val (9K) used for hyperparameter selection
  - Source negatives = 25,551 random (Partition 1) + 4,449 ML-FPs (mixed)
  - SSL pretraining = Partition 0 + ML-FP cutouts (unlabeled at this stage)
  - UDA target = Partition 2
  - Test negatives = Partition 4
  - Val negatives = Partition 3 (subsampled to 5K)

Real lens splits use stratified hash-based fold assignment (by grade).
Splits are deterministic: same inputs always produce the same splits.json.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------- Paths ----------
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SIM_SHARD_DIR = DATA_DIR / "sim_lenses" / "shards"
REAL_LENS_PATH = DATA_DIR / "q1_lenses_processed" / "lens_cutouts.npz"
UNLABELED_SHARD_DIR = DATA_DIR / "q1_unlabeled" / "shards"
ML_FP_SHARD_DIR = DATA_DIR / "ml_fp_cutouts" / "shards"

# ---------- Split parameters ----------
N_FOLDS = 5                          # k-fold CV across real lenses (Lines methodology)
N_SOURCE_NEGATIVES = 30_000          # 1:1 with simulated positives
ML_FP_FRACTION_OF_SOURCE = 4_449     # Use all confirmed ML-FPs we have
VAL_NEG_SUBSAMPLE = 5_000            # Partition 3 subsample for validation
RANDOM_SEED = 42                     # For all subsampling operations

# ---------- Real lens 5-fold CV ----------
EXPECTED_AB_LENSES = 326   # assert against this; only change if you've verified a new count

def split_real_lenses_kfold() -> dict[str, Any]:
    """Assign each eligible grade-A/B lens to one of N_FOLDS folds, stratified by
    grade so per-fold grade counts are balanced BY CONSTRUCTION, not in expectation.
    Deterministic: md5-sort within each grade, then round-robin into folds.

    Excluded (no fold assignment):
      - grade not in {A, B}           (the 'special' cutouts)
      - source in {group, recenter}   (different physics / uncertain processing)
    """
    if not REAL_LENS_PATH.exists():
        raise FileNotFoundError(f"Real lens cutouts not found at {REAL_LENS_PATH}")

    d = np.load(REAL_LENS_PATH, allow_pickle=False)
    id_strs, grades, source_zips = d["id_str"], d["grade"], d["source_zip"]

    by_grade: dict[str, list[tuple[str, int]]] = defaultdict(list)
    excluded_indices: list[int] = []
    for i in range(len(id_strs)):
        id_str, grade, source = str(id_strs[i]), str(grades[i]), str(source_zips[i])
        if source in ("group", "recenter") or grade not in ("A", "B"):
            excluded_indices.append(i)
            continue
        sort_key = hashlib.md5(f"{id_str}::{grade}".encode()).hexdigest()
        by_grade[grade].append((sort_key, i))

    fold_indices: dict[int, list[int]] = defaultdict(list)
    per_fold_grade_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for grade in sorted(by_grade):              # sorted for cross-run determinism
        for rank, (_, orig_i) in enumerate(sorted(by_grade[grade])):  # md5 order
            fold = rank % N_FOLDS               # round-robin => balanced within grade
            fold_indices[fold].append(orig_i)
            per_fold_grade_counts[fold][grade] += 1

    folds = {}
    for f in range(N_FOLDS):
        idxs = sorted(fold_indices.get(f, []))
        folds[str(f)] = {
            "indices": [int(idx) for idx in idxs],
            "n_total": len(idxs),
            "n_A": per_fold_grade_counts[f].get("A", 0),
            "n_B": per_fold_grade_counts[f].get("B", 0),
        }

    total_assigned = sum(folds[str(f)]["n_total"] for f in range(N_FOLDS))
    if total_assigned != EXPECTED_AB_LENSES:    # fail loud — silent miscount poisons everything downstream
        raise ValueError(
            f"Expected {EXPECTED_AB_LENSES} eligible grade-A/B lenses, assigned {total_assigned} "
            f"(A={sum(folds[str(f)]['n_A'] for f in range(N_FOLDS))}, "
            f"B={sum(folds[str(f)]['n_B'] for f in range(N_FOLDS))}, "
            f"excluded={len(excluded_indices)}). If the new count is correct, update EXPECTED_AB_LENSES."
        )

    return {
        "folds": folds,
        "excluded_indices": [int(i) for i in excluded_indices],
        "audit": {"n_folds": N_FOLDS, "total_assigned": total_assigned,
                  "total_excluded": len(excluded_indices)},
    }


# ---------- Unlabeled partition assignments ----------
def build_unlabeled_assignment(n_ml_fp: int) -> dict[str, Any]:
    """Assign each unlabeled cutout to a role based on its hash-partition.

    n_ml_fp is the ACTUAL number of ML-FP cutouts on disk; random negatives are
    sized to fill the remainder of the 1:1 source pool exactly.

    P0 -> SSL pretrain | P1 -> source neg (subsampled, remainder reserved)
    P2 -> UDA target   | P3 -> val neg (subsampled) | P4 -> test neg
    """
    if not UNLABELED_SHARD_DIR.exists():
        raise FileNotFoundError(f"Unlabeled shards not found at {UNLABELED_SHARD_DIR}")
    shards = sorted(UNLABELED_SHARD_DIR.glob("shard_tile*.npz"))
    if not shards:
        raise FileNotFoundError(f"No unlabeled shards in {UNLABELED_SHARD_DIR}")

    per_partition: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for shard_path in shards:
        with np.load(shard_path) as d:                 # context manager: no leaked handles
            for i, p in enumerate(d["partitions"]):
                per_partition[int(p)].append((shard_path.name, int(i)))

    rng = np.random.default_rng(RANDOM_SEED)

    p0_full = per_partition[0]                          # SSL pretrain (all)

    p1_full = per_partition[1]                          # source negatives
    n_random_negs = N_SOURCE_NEGATIVES - n_ml_fp        # exact remainder for 1:1
    if n_random_negs < 0:
        raise ValueError(
            f"{n_ml_fp} ML-FPs exceed the {N_SOURCE_NEGATIVES} source-negative budget; "
            f"reduce ML-FPs or raise N_SOURCE_NEGATIVES."
        )
    if len(p1_full) >= n_random_negs:
        idx = rng.choice(len(p1_full), size=n_random_negs, replace=False)
        chosen = set(idx.tolist())                      # precompute once -> O(n), not O(n^2)
        p1_subsample = [p1_full[i] for i in sorted(chosen)]
        p1_reserve = [p1_full[i] for i in range(len(p1_full)) if i not in chosen]
    else:
        print(f"WARNING: Partition 1 has only {len(p1_full)} negatives, need "
              f"{n_random_negs}; using all (the 1:1 assert in main will then fail).")
        p1_subsample, p1_reserve = p1_full, []

    p2_full = per_partition[2]                          # UDA target (all)

    p3_full = per_partition[3]                          # validation negatives, subsample
    if len(p3_full) > VAL_NEG_SUBSAMPLE:
        idx = rng.choice(len(p3_full), size=VAL_NEG_SUBSAMPLE, replace=False)
        p3_subsample = [p3_full[i] for i in sorted(idx.tolist())]
    else:
        p3_subsample = p3_full

    p4_full = per_partition[4]                          # test negatives (all)

    return {
        "partition_0_ssl_pretrain": p0_full,
        "partition_1_source_neg_random": p1_subsample,
        "partition_1_reserve": p1_reserve,
        "partition_2_uda_target": p2_full,
        "partition_3_val_neg": p3_subsample,
        "partition_4_test_neg": p4_full,
        "audit": {
            "p0_ssl_pretrain_count": len(p0_full),
            "p1_source_neg_count": len(p1_subsample),
            "p1_reserve_count": len(p1_reserve),
            "p2_uda_target_count": len(p2_full),
            "p3_val_neg_count": len(p3_subsample),
            "p3_dropped_count": len(p3_full) - len(p3_subsample),
            "p4_test_neg_count": len(p4_full),
            "total_unlabeled": sum(len(v) for v in per_partition.values()),
        },
    }


# ---------- ML-FP assignments ----------
def build_ml_fp_assignment() -> dict[str, Any]:
    """Enumerate ML-FP cutouts. They serve dual roles:
      - In SSL pretraining: contribute unlabeled examples
      - In supervised DA training: contribute confirmed non-lens labels
    """
    if not ML_FP_SHARD_DIR.exists():
        return {
            "shards": [],
            "audit": {"ml_fp_count": 0, "ml_fp_shards": 0},
        }

    shards = sorted(ML_FP_SHARD_DIR.glob("ml_fp_tile*.npz"))
    locations: list[tuple[str, int]] = []
    for shard_path in shards:
        d = np.load(shard_path)
        for i in range(len(d["object_ids"])):
            locations.append((shard_path.name, int(i)))

    return {
        "shards": [s.name for s in shards],
        "locations": locations,
        "audit": {
            "ml_fp_count": len(locations),
            "ml_fp_shards": len(shards),
        },
    }


# ---------- Simulated lens assignments ----------
def build_sim_assignment() -> dict[str, Any]:
    """Enumerate simulated lens shards for source-domain training and sim-val."""
    if not SIM_SHARD_DIR.exists():
        raise FileNotFoundError(f"Sim shards not found at {SIM_SHARD_DIR}")

    train_shards = sorted(SIM_SHARD_DIR.glob("sim_train_*.npz"))
    val_shards = sorted(SIM_SHARD_DIR.glob("sim_val_*.npz"))

    train_total = sum(int(np.load(s)["images"].shape[0]) for s in train_shards)
    val_total = sum(int(np.load(s)["images"].shape[0]) for s in val_shards)

    return {
        "sim_train_shards": [s.name for s in train_shards],
        "sim_val_shards": [s.name for s in val_shards],
        "audit": {
            "sim_train_count": train_total,
            "sim_val_count": val_total,
        },
    }


def check_split_hygiene() -> None:
    """Fail loud on the two leaks that hash-partitioning can't rule out by itself:
       (a) an object assigned to >1 partition (Euclid tile-overlap duplicate), and
       (b) an ML-FP that still sits inside a partition negative pool."""
    obj_to_parts: dict[Any, set[int]] = defaultdict(set)
    n_rows = 0
    for s in sorted(UNLABELED_SHARD_DIR.glob("shard_tile*.npz")):
        with np.load(s) as d:
            for o, p in zip(d["object_ids"].tolist(), d["partitions"].tolist()):
                obj_to_parts[o].add(int(p)); n_rows += 1

    cross = {o: ps for o, ps in obj_to_parts.items() if len(ps) > 1}
    if cross:
        sample = list(cross.items())[:3]
        raise ValueError(
            f"{len(cross)} object(s) span multiple partitions (e.g. {sample}). "
            f"Tile-overlap duplicate landed in different partitions -> train/test "
            f"negative leak. Re-partition by object_id hash, not per-cutout."
        )
    n_dup = n_rows - len(obj_to_parts)
    if n_dup:
        print(f"  note: {n_dup} duplicate cutouts (same object, same partition) "
              f"-- benign, but dedupe if you want clean negative counts")

    ml_fp_ids: set[Any] = set()
    for s in sorted(ML_FP_SHARD_DIR.glob("ml_fp_tile*.npz")):
        with np.load(s) as d:
            ml_fp_ids.update(d["object_ids"].tolist())
    role_ids = {p: {o for o, ps in obj_to_parts.items() if p in ps} for p in (1, 2, 3, 4)}
    for p in (1, 2, 3, 4):
        overlap = ml_fp_ids & role_ids[p]
        if overlap:
            raise ValueError(
                f"{len(overlap)} ML-FP object(s) also in partition {p}; "
                f"P4 overlap is a train/test negative leak."
            )
    print(f"  OK: no cross-partition objects; {len(ml_fp_ids):,} ML-FPs disjoint from P1-P4")


# ---------- Main ----------
def main():
    print("=" * 60)
    print("Creating splits — 5-fold stratified CV + ML-FP integration")
    print("=" * 60)

    print("\n[1/5] 5-fold stratified CV across real grade-A+B lenses...")
    real_split = split_real_lenses_kfold()              # self-asserts == EXPECTED_AB_LENSES
    print(f"  Total assigned across folds: {real_split['audit']['total_assigned']}")
    print(f"  Excluded (grade-C/blank, group, recenter): "
          f"{real_split['audit']['total_excluded']}")
    for f in range(N_FOLDS):
        info = real_split["folds"][str(f)]
        print(f"    Fold {f}: {info['n_total']:>3} total  "
              f"(A={info['n_A']:>3}, B={info['n_B']:>3})")

    print("\n[2/5] Enumerating ML-FP cutouts (counted first to size negatives)...")
    ml_fp_split = build_ml_fp_assignment()
    n_ml_fp = ml_fp_split["audit"]["ml_fp_count"]
    print(f"  ML-FP cutouts: {n_ml_fp:,}  (shards: {ml_fp_split['audit']['ml_fp_shards']:,})")
    if n_ml_fp == 0:
        raise ValueError("No ML-FP cutouts found; run the ML-FP download before splitting.")

    print("\n[3/5] Partitioning unlabeled Euclid galaxies...")
    unlabeled_split = build_unlabeled_assignment(n_ml_fp)
    a = unlabeled_split["audit"]
    print(f"  P0 SSL pretrain: {a['p0_ssl_pretrain_count']:,}")
    print(f"  P1 source neg:   {a['p1_source_neg_count']:,}  (reserve {a['p1_reserve_count']:,})")
    print(f"  P2 UDA target:   {a['p2_uda_target_count']:,}")
    print(f"  P3 val neg:      {a['p3_val_neg_count']:,}  (dropped {a['p3_dropped_count']:,})")
    print(f"  P4 test neg:     {a['p4_test_neg_count']:,}")

    print("\n[4/5] Enumerating simulated lens shards...")
    sim_split = build_sim_assignment()
    print(f"  Sim train: {sim_split['audit']['sim_train_count']:,}")
    print(f"  Sim val:   {sim_split['audit']['sim_val_count']:,}")

    print("\n[5/5] Split-hygiene checks (cross-partition leakage, ML-FP disjointness)...")
    check_split_hygiene()                               # raises on any leak

    # ---- Hard invariant: source pool is exactly 1:1 ----
    n_neg = a["p1_source_neg_count"] + n_ml_fp
    n_pos = sim_split["audit"]["sim_train_count"]
    if n_neg != n_pos:
        raise ValueError(
            f"Source pool not 1:1: {n_pos:,} sim positives vs {n_neg:,} negatives "
            f"({a['p1_source_neg_count']:,} random + {n_ml_fp:,} ML-FP). "
            f"Adjust N_SOURCE_NEGATIVES or check sim_train_count."
        )

    # ---- Hard invariant: every unlabeled cutout is accounted for ----
    n_placed = (a['p0_ssl_pretrain_count'] + a['p1_source_neg_count']
                + a['p1_reserve_count'] + a['p2_uda_target_count']
                + a['p3_val_neg_count'] + a['p4_test_neg_count'])
    if n_placed + a['p3_dropped_count'] != a['total_unlabeled']:
        raise ValueError(
            f"Unlabeled accounting mismatch: placed {n_placed:,} + dropped "
            f"{a['p3_dropped_count']:,} != total {a['total_unlabeled']:,}."
        )
    print(f"\n  Source pool 1:1: {n_pos:,} pos vs {n_neg:,} neg.")
    print(f"  Unlabeled fully accounted: {a['total_unlabeled']:,}.")

    # ---- Save ----
    splits = {
        "version": 3,
        "design": "5-fold stratified CV on real lenses; ML-FPs sized into source "
                  "negatives for exact 1:1; leakage-checked",
        "random_seed": RANDOM_SEED,
        "n_folds": N_FOLDS,
        "n_source_negatives_target": N_SOURCE_NEGATIVES,
        "n_ml_fp_actual": n_ml_fp,
        "real_lenses_kfold": real_split,
        "unlabeled_euclid": unlabeled_split,
        "ml_fp_cutouts": ml_fp_split,
        "simulated_lenses": sim_split,
    }
    out_json = SPLITS_DIR / "splits.json"
    with open(out_json, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved {out_json}")

    # ---- Human-readable audit ----
    audit_path = SPLITS_DIR / "audit.txt"
    with open(audit_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("LENS-FINDING SPLITS — version 3\n")
        f.write("Design: 5-fold stratified CV (Lines methodology) + ML-FP integration\n")
        f.write("=" * 60 + "\n\n")

        f.write("--- REAL LENSES (5-fold stratified CV) ---\n")
        for fold_i in range(N_FOLDS):
            info = real_split["folds"][str(fold_i)]
            f.write(f"  Fold {fold_i}: {info['n_total']:>3} lenses "
                    f"(A={info['n_A']:>3}, B={info['n_B']:>3})\n")
        f.write(f"  Excluded: {real_split['audit']['total_excluded']} "
                f"(grade-C/blank, group, recenter)\n\n")

        f.write("--- TRAINING ---\n")
        f.write("SSL pretraining inputs (unlabeled):\n")
        f.write(f"  - Partition 0 (unlabeled Euclid): {a['p0_ssl_pretrain_count']:,}\n")
        f.write(f"  - ML-FP cutouts (unlabeled at this stage): {n_ml_fp:,}\n")
        f.write(f"  Combined: {a['p0_ssl_pretrain_count'] + n_ml_fp:,}\n\n")
        f.write("Source-domain training (exact 1:1):\n")
        f.write(f"  - Positives: {n_pos:,} sim lenses\n")
        f.write(f"  - Negatives: {a['p1_source_neg_count']:,} random + {n_ml_fp:,} "
                f"ML-FP = {n_neg:,}\n\n")
        f.write("UDA target (unlabeled):\n")
        f.write(f"  - Partition 2: {a['p2_uda_target_count']:,}\n\n")

        f.write("--- MODEL / METHOD SELECTION ---\n")
        f.write(f"Sim-domain val: {sim_split['audit']['sim_val_count']:,} "
                f"(hyperparameters, checkpoint, DA-method choice)\n")
        f.write("NOTE: real lenses are NOT used for selection. Selection signals are\n")
        f.write("      sim-val performance + unlabeled-target diagnostics only, which\n")
        f.write("      keeps all 326 real lenses as a clean held-out test set.\n")
        f.write(f"Validation negatives: {a['p3_val_neg_count']:,}\n\n")

        f.write("--- TEST (held out; never in training or selection) ---\n")
        f.write("Each of the 5 folds is scored ONCE as a disjoint test subset against\n")
        f.write("the single trained model -> 5 independent TPR@FPR estimates for\n")
        f.write("cross-fold spread. Folds are NOT rotated as train/val (nothing trains\n")
        f.write("on real lenses).\n")
        f.write(f"Test negatives: {a['p4_test_neg_count']:,}\n\n")

        f.write("--- RESERVE / DROPPED / EXCLUDED ---\n")
        f.write(f"Partition 1 reserve: {a['p1_reserve_count']:,}\n")
        f.write(f"Partition 3 dropped (unused beyond val subsample): {a['p3_dropped_count']:,}\n")
        f.write(f"Excluded real lenses: {real_split['audit']['total_excluded']}\n\n")

        f.write("--- ACCOUNTING ---\n")
        n_real = (real_split['audit']['total_assigned']
                  + real_split['audit']['total_excluded'])
        n_sim = sim_split['audit']['sim_train_count'] + sim_split['audit']['sim_val_count']
        f.write(f"Real lenses total: {n_real}\n")
        f.write(f"Unlabeled Euclid total: {a['total_unlabeled']:,} "
                f"(placed {n_placed:,} + dropped {a['p3_dropped_count']:,})\n")
        f.write(f"Simulated lenses total: {n_sim:,}\n")
        f.write(f"ML-FP cutouts: {n_ml_fp:,}\n")

    print(f"Saved {audit_path}")


if __name__ == "__main__":
    main()