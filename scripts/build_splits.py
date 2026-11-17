r"""
build_splits.py  --  build the complete splits.json for the label-free sim-to-real
Euclid lens-finding pipeline (fresh design; supersedes any older splits).

DESIGN (all decisions research-locked in the design sessions):

  SOURCE (labeled, fixed -- does NOT rotate with folds)
    source_train : 80% of sim lenses (positives)
                   + negatives = ALL ML-FPs + UNIQUE bare-deflectors (deduped by
                     object_id) + random negatives, balanced 1:1 with positives.
    val          : 20% of sim lenses (positives)
                   + negatives = bare-deflectors + random (NO ML-FPs), 1:1.
    -> sim lenses split 80/20 GROUPED by deflector object_id (a deflector's <=10
       lens configs stay on one side); a deflector's bare-deflector negative goes
       to the SAME side as its lenses (no lens/bare leakage across train|val).

  REAL-LENS 5-FOLD  (grade-stratified, object_id-grouped)
    489 real lenses -> 5 folds via StratifiedGroupKFold(grade, groups=object_id).

  PER-FOLD ROTATION (target + test)
    target[k] : the OTHER four folds' real lenses (unlabeled positives)
                + FRESH per-fold random negatives from P2 (disjoint across folds),
                balanced ~1:1 with the target positives  (matches source label ratio).
    test[k]   : the held-out fold's real lenses (positives)
                + the SHARED full P4 test-negative pool (random only, NO ML-FPs),
                same pool every fold (fixed-threshold TPR@FPR=1e-3, benchmark-matched).

  NEGATIVE POOL ROLES (from the unlabeled shards' 'partitions' field, 0..4):
    P0 -> (SSL; not used here)   P1 -> source random negs
    P2 -> target random negs     P3 -> val random negs      P4 -> test negs
  (These partitions are one homogeneous pool of random Euclid galaxies, pre-split.)

  HARD RULES (audited at the end; script aborts if any fails):
    * source-train / val: sim lenses grouped by deflector object_id, no group split.
    * bare-deflectors deduped to unique object_id; a deflector's bare-neg on the
      same side as its lenses.
    * every negative used anywhere is unique to its role draw (no double-use within
      a role); target[k] random negs are disjoint ACROSS folds.
    * target random negs (P2)  disjoint from  test negs (P4)   [no test contamination]
    * a real lens is never in both target[k] and test[k].
    * grade balance preserved across folds.

Run from repo root:   python scripts\build_splits.py
"""
from __future__ import annotations
import glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

try:
    from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
except ImportError as e:
    sys.exit("need scikit-learn:  pip install scikit-learn")

# ----------------------------- config -----------------------------
ROOT          = Path(".")
SIM_DIR       = ROOT / "data" / "sim_arc_on_real"        # shard_*.npz: lenses, nonlenses, object_id
LENS_NPZ      = ROOT / "data" / "q1_lenses_processed" / "lens_cutouts.npz"
NEG_DIR       = ROOT / "data" / "q1_unlabeled" / "shards"  # *.npz: images, object_ids, partitions
MLFP_DIR      = ROOT / "data" / "ml_fp_cutouts" / "shards" # *.npz: images, object_ids
OUT_JSON      = ROOT / "data" / "splits" / "splits.json"

N_FOLDS       = 5
VAL_FRAC      = 0.20            # of sim lenses (grouped by deflector)
SEED          = 42

# partition role map (the unlabeled shards' integer 'partitions' field)
P_SOURCE_NEG  = 1              # train random negatives
P_TARGET_NEG  = 2              # per-fold target random negatives (fresh per fold)
P_VAL_NEG     = 3              # val random negatives
P_TEST_NEG    = 4              # shared test negatives

# how much of the negative composition is bare-deflector vs random in the SOURCE.
# All unique bare-deflectors are used; random fills the remainder to reach 1:1.
# ML-FPs: all of them, train only.


# ----------------------- load helpers -----------------------------
def load_sim():
    """Return per-lens deflector object_id array (row order == concatenated shard order)
    and, separately, the unique deflector object_ids that have a bare-deflector negative."""
    shards = sorted(glob.glob(str(SIM_DIR / "shard_*.npz")))
    if not shards:
        sys.exit(f"no sim shards in {SIM_DIR}")
    lens_defl = []          # deflector object_id per LENS row (positives)
    bare_defl = []          # deflector object_id per NONLENS row (bare-deflector negatives)
    per_shard = []
    for f in shards:
        with np.load(f) as d:
            oid = d["object_id"].astype("int64")
            n_lens = len(d["lenses"]); n_bare = len(d["nonlenses"])
            if n_lens != len(oid) or n_bare != len(oid):
                sys.exit(f"{f}: lenses/nonlenses/object_id length mismatch")
            lens_defl.append(oid)
            bare_defl.append(oid)            # nonlens i is the bare deflector of lens i (same object_id)
            per_shard.append((Path(f).name, n_lens))
    lens_defl = np.concatenate(lens_defl)
    bare_defl = np.concatenate(bare_defl)
    return lens_defl, bare_defl, shards, per_shard


def load_real_lenses():
    with np.load(LENS_NPZ, allow_pickle=True) as d:
        grade = np.asarray(d["grade"]).astype(str)
        oid   = d["object_id"].astype("int64")
        n     = len(grade)
    return n, grade, oid


def load_negatives():
    """object_id -> partition, plus a location map object_id -> (shard, idx)."""
    shards = sorted(glob.glob(str(NEG_DIR / "*.npz")))
    if not shards:
        sys.exit(f"no negative shards in {NEG_DIR}")
    oid2part, oid2loc = {}, {}
    for f in shards:
        with np.load(f) as d:
            oids = d["object_ids"].astype("int64")
            parts = d["partitions"].astype(int)
            for i, o in enumerate(oids):
                o = int(o)
                oid2part[o] = int(parts[i])
                oid2loc[o] = (Path(f).name, int(i))
    return oid2part, oid2loc


def load_mlfp():
    shards = sorted(glob.glob(str(MLFP_DIR / "*.npz")))
    if not shards:
        sys.exit(f"no ml-fp shards in {MLFP_DIR}")
    loc = {}
    for f in shards:
        with np.load(f) as d:
            oids = d["object_ids"].astype("int64")
            for i, o in enumerate(oids):
                loc[int(o)] = (Path(f).name, int(i))
    return loc


# ----------------------------- build ------------------------------
def main():
    rng = np.random.default_rng(SEED)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # ---------- SOURCE: split sim lenses 80/20 grouped by deflector object_id ----------
    lens_defl, bare_defl, sim_shards, per_shard = load_sim()
    n_sim = len(lens_defl)
    uniq_defl = np.unique(lens_defl)
    sim_defl_set = set(int(o) for o in uniq_defl)   # deflector galaxies used to build sim lenses
    print(f"sim: {n_sim} lens rows, {len(uniq_defl)} unique deflectors")

    # GroupKFold to get a grouped ~80/20 split (1 fold out of 5 = 20% val), grouped by deflector.
    gkf = GroupKFold(n_splits=int(round(1 / VAL_FRAC)))
    tr_idx, va_idx = next(gkf.split(np.zeros(n_sim), groups=lens_defl))
    train_defl = set(lens_defl[tr_idx].tolist())
    val_defl   = set(lens_defl[va_idx].tolist())
    # safety: grouped split => disjoint deflector sets
    assert train_defl.isdisjoint(val_defl), "sim train/val share a deflector (group leak)"
    print(f"  source split: train {len(tr_idx)} lens rows / val {len(va_idx)} lens rows "
          f"({len(train_defl)}|{len(val_defl)} deflectors)")

    # sim lens ROW positions per split (these index into the concatenated sim shard order)
    sim_train_rows = np.sort(tr_idx).tolist()
    sim_val_rows   = np.sort(va_idx).tolist()

    # ---------- bare-deflector negatives: DEDUPE to unique object_id, per side ----------
    # a deflector's bare-neg lives on the same side as its lenses (train_defl / val_defl).
    # dedupe: one bare-deflector per unique deflector object_id.
    bare_train_defl = sorted(train_defl)           # unique deflector ids, train side
    bare_val_defl   = sorted(val_defl)             # unique deflector ids, val side
    # map deflector object_id -> a representative (shard,row) of its bare-deflector (first occurrence)
    defl_first_bare = {}
    off = 0
    for (name, n) in per_shard:
        for i in range(n):
            o = int(bare_defl[off + i])
            if o not in defl_first_bare:
                defl_first_bare[o] = [name, i]     # (shard, row-in-shard) of a bare-deflector for this deflector
        off += n

    # ---------- negatives pool ----------
    oid2part, oid2loc = load_negatives()
    # EXCLUDE sim-deflector galaxies from the negative pool entirely: a galaxy used as
    # the host of a simulated POSITIVE lens must never appear as a NEGATIVE anywhere
    # (esp. P2 target / P4 test), or the same physical galaxy is both a positive host
    # and an evaluation negative -> label conflict / metric contamination.
    n_purged = 0
    for o in list(oid2part.keys()):
        if o in sim_defl_set:
            del oid2part[o]
            oid2loc.pop(o, None)
            n_purged += 1
    print(f"  purged {n_purged} sim-deflector galaxies from the negative pool")
    by_part = defaultdict(list)
    for o, p in oid2part.items():
        by_part[p].append(o)
    for p in by_part:
        by_part[p] = np.array(sorted(by_part[p]), dtype="int64")   # deterministic order
    print("  neg partitions (after purge):", {p: len(by_part[p]) for p in sorted(by_part)})

    mlfp_loc = load_mlfp()
    mlfp_oids = sorted(mlfp_loc.keys())
    n_mlfp = len(mlfp_oids)
    print(f"  ml-fp: {n_mlfp}")

    # ---------- counts: balance negatives 1:1 with positives on each side ----------
    n_pos_train = len(sim_train_rows)
    n_pos_val   = len(sim_val_rows)

    # TRAIN negatives = all ML-FP + all unique train bare-deflectors + random(P1) to fill 1:1
    n_bare_train = len(bare_train_defl)
    n_rand_train = n_pos_train - n_mlfp - n_bare_train
    if n_rand_train < 0:
        sys.exit(f"train: ML-FP ({n_mlfp}) + bare ({n_bare_train}) exceed positives ({n_pos_train}); "
                 f"reduce bare or raise positives.")
    # VAL negatives = all unique val bare-deflectors + random(P3) to fill 1:1 (no ML-FP)
    n_bare_val = len(bare_val_defl)
    n_rand_val = n_pos_val - n_bare_val
    if n_rand_val < 0:
        sys.exit(f"val: bare ({n_bare_val}) exceed positives ({n_pos_val}).")

    # draw random negatives WITHOUT replacement from the right partitions, all disjoint
    def draw(part, k, taken):
        pool = [o for o in by_part.get(part, []) if o not in taken]
        if k > len(pool):
            sys.exit(f"partition {part}: need {k}, only {len(pool)} free")
        chosen = rng.choice(pool, size=k, replace=False).tolist() if k > 0 else []
        taken.update(chosen)
        return chosen

    taken_negs = set()   # every random-negative object_id used anywhere (global disjointness ledger)
    train_rand = draw(P_SOURCE_NEG, n_rand_train, taken_negs)
    val_rand   = draw(P_VAL_NEG,   n_rand_val,   taken_negs)

    # ---------- REAL-LENS 5-FOLD (grade-stratified, object_id-grouped) ----------
    n_lens, grade, lens_oid = load_real_lenses()
    n_dupe = n_lens - len(set(lens_oid.tolist()))
    print(f"real lenses: {n_lens}  gradeA={int((grade=='A').sum())} gradeB={int((grade=='B').sum())}  "
          f"duplicate object_id rows={n_dupe}")

    if n_dupe > 0:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_iter = splitter.split(np.zeros(n_lens), y=grade, groups=lens_oid)
        fold_mode = "StratifiedGroupKFold (object_id groups)"
    else:
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_iter = splitter.split(np.zeros(n_lens), y=grade)
        fold_mode = "StratifiedKFold (no duplicate object_ids)"
    # test rows per fold = the held-out indices
    fold_test_rows = [np.sort(te).tolist() for _, te in fold_iter]
    print(f"  fold mode: {fold_mode}")

    # ---------- SHARED test negatives (full P4) ----------
    test_neg = sorted(int(o) for o in by_part.get(P_TEST_NEG, []))
    if not test_neg:
        sys.exit("no P4 test negatives found")

    # ---------- per-fold target + test ----------
    all_rows = set(range(n_lens))
    folds = []
    target_neg_ledger = set()   # target random negs across ALL folds (must stay disjoint from each other + from test)
    for k in range(N_FOLDS):
        test_rows = fold_test_rows[k]
        target_rows = sorted(all_rows - set(test_rows))    # other four folds' lenses = unlabeled target positives
        # fresh per-fold target random negatives from P2, ~1:1 with target positives, disjoint across folds
        n_tgt_pos = len(target_rows)
        # draw from P2 excluding anything already used as a target neg in a prior fold (fresh/disjoint)
        tgt_neg = draw(P_TARGET_NEG, n_tgt_pos, target_neg_ledger)  # target_neg_ledger also feeds `taken`? see note
        folds.append({
            "fold": k,
            "test_lens_rows": test_rows,                  # positions into lens_cutouts.npz
            "target_lens_rows": target_rows,              # unlabeled positives (positions into lens_cutouts.npz)
            "target_neg_object_ids": [int(o) for o in tgt_neg],
        })

    # ---------- ASSEMBLE splits.json ----------
    def loc_list(oids):
        return [list(oid2loc[int(o)]) for o in oids]     # [shard, idx] into unlabeled shards

    splits = {
        "schema": "splits_v6_label_free_da",
        "seed": SEED,
        "n_folds": N_FOLDS,
        "fold_mode": fold_mode,
        "sim_shards": [name for name, _ in per_shard],   # concat order defines sim row indices
        "source_train": {
            "sim_lens_rows": sim_train_rows,             # positions into concatenated sim 'lenses'
            "neg_mlfp": {"object_ids": [int(o) for o in mlfp_oids],
                          "locations": [list(mlfp_loc[o]) for o in mlfp_oids]},
            "neg_bare_deflector": {"deflector_object_ids": [int(o) for o in bare_train_defl],
                                    "locations": [defl_first_bare[o] for o in bare_train_defl]},
            "neg_random": {"object_ids": [int(o) for o in train_rand],
                            "locations": loc_list(train_rand)},
        },
        "val": {
            "sim_lens_rows": sim_val_rows,
            "neg_bare_deflector": {"deflector_object_ids": [int(o) for o in bare_val_defl],
                                    "locations": [defl_first_bare[o] for o in bare_val_defl]},
            "neg_random": {"object_ids": [int(o) for o in val_rand],
                            "locations": loc_list(val_rand)},
        },
        "test_negatives_shared": {"object_ids": test_neg, "locations": loc_list(test_neg)},
        "real_lenses_kfold": {"folds": folds},
    }

    # attach per-fold target-neg locations
    for fd in splits["real_lenses_kfold"]["folds"]:
        fd["target_neg_locations"] = loc_list(fd["target_neg_object_ids"])

    # ======================= AUDIT (abort on any failure) =======================
    audit(splits, lens_defl, bare_defl, train_defl, val_defl,
          oid2part, mlfp_oids, grade, n_lens, per_shard, sim_defl_set)

    with open(OUT_JSON, "w") as f:
        json.dump(splits, f)
    print(f"\nOK -> wrote {OUT_JSON}")
    _summary(splits, grade)


# ----------------------------- audit ------------------------------
def audit(splits, lens_defl, bare_defl, train_defl, val_defl,
          oid2part, mlfp_oids, grade, n_lens, per_shard, sim_defl_set):
    fail = []

    # 1) source sim train/val row disjoint + cover, and grouped-by-deflector (no shared deflector)
    tr = set(splits["source_train"]["sim_lens_rows"])
    va = set(splits["val"]["sim_lens_rows"])
    if tr & va: fail.append("sim train/val rows overlap")
    if train_defl & val_defl: fail.append("sim train/val share a deflector (group leak)")

    # 2) bare-deflectors deduped (unique object_ids) and on correct side
    bt = splits["source_train"]["neg_bare_deflector"]["deflector_object_ids"]
    bv = splits["val"]["neg_bare_deflector"]["deflector_object_ids"]
    if len(bt) != len(set(bt)): fail.append("train bare-deflectors not unique")
    if len(bv) != len(set(bv)): fail.append("val bare-deflectors not unique")
    if set(bt) - train_defl: fail.append("train bare-deflector from wrong side")
    if set(bv) - val_defl:   fail.append("val bare-deflector from wrong side")
    if set(bt) & set(bv):    fail.append("bare-deflector in both train and val")

    # 3) balance 1:1 on each source side
    n_pos_tr = len(splits["source_train"]["sim_lens_rows"])
    n_neg_tr = len(mlfp_oids) + len(bt) + len(splits["source_train"]["neg_random"]["object_ids"])
    if n_pos_tr != n_neg_tr: fail.append(f"train not 1:1 (pos {n_pos_tr} vs neg {n_neg_tr})")
    n_pos_va = len(splits["val"]["sim_lens_rows"])
    n_neg_va = len(bv) + len(splits["val"]["neg_random"]["object_ids"])
    if n_pos_va != n_neg_va: fail.append(f"val not 1:1 (pos {n_pos_va} vs neg {n_neg_va})")

    # 4) random negatives come from correct partitions.
    #    A foreign object_id (e.g. an ML-FP id, which is NOT in the negative-partition
    #    map) maps to sentinel -1 so it fails the partition check cleanly instead of
    #    raising KeyError. -1 is never a valid partition, so any leak is flagged.
    def parts_of(oids): return {oid2part.get(int(o), -1) for o in oids}
    if parts_of(splits["source_train"]["neg_random"]["object_ids"]) - {P_SOURCE_NEG}:
        fail.append("train random negs not all from P1")
    if parts_of(splits["val"]["neg_random"]["object_ids"]) - {P_VAL_NEG}:
        fail.append("val random negs not all from P3")
    if parts_of(splits["test_negatives_shared"]["object_ids"]) - {P_TEST_NEG}:
        fail.append("test negs not all from P4")

    # 5) global random-neg disjointness: train / val / test / all target folds mutually disjoint
    train_r = set(splits["source_train"]["neg_random"]["object_ids"])
    val_r   = set(splits["val"]["neg_random"]["object_ids"])
    test_r  = set(splits["test_negatives_shared"]["object_ids"])
    tgt_all = []
    for fd in splits["real_lenses_kfold"]["folds"]:
        tgt_all.append(set(fd["target_neg_object_ids"]))
    # pairwise across the named source/test sets
    for a, an, b, bn in [(train_r,"train",val_r,"val"), (train_r,"train",test_r,"test"),
                         (val_r,"val",test_r,"test")]:
        if a & b: fail.append(f"{an} and {bn} random negs overlap")
    # target folds: disjoint from each other AND from test (no test contamination), AND from train/val
    union_tgt = set()
    for i, tg in enumerate(tgt_all):
        if tg & test_r:  fail.append(f"target fold {i} negs overlap TEST negs (contamination)")
        if tg & train_r: fail.append(f"target fold {i} negs overlap train negs")
        if tg & val_r:   fail.append(f"target fold {i} negs overlap val negs")
        if tg & union_tgt: fail.append(f"target fold {i} negs overlap another fold's target (not fresh)")
        union_tgt |= tg
    # target negs must be P2
    for i, tg in enumerate(tgt_all):
        if tg and parts_of(tg) - {P_TARGET_NEG}:
            fail.append(f"target fold {i} negs not all from P2")

    # 6) per-fold real-lens rotation: test and target rows partition the 489, no overlap
    seen_test = []
    for fd in splits["real_lenses_kfold"]["folds"]:
        t = set(fd["test_lens_rows"]); g = set(fd["target_lens_rows"])
        if t & g: fail.append(f"fold {fd['fold']}: a lens is in both target and test")
        if t | g != set(range(n_lens)): fail.append(f"fold {fd['fold']}: target+test != all lenses")
        seen_test.append(t)
    # every lens tested exactly once across folds
    cnt = Counter()
    for t in seen_test: cnt.update(t)
    if set(cnt.values()) != {1} or len(cnt) != n_lens:
        fail.append("real lenses not each tested exactly once across folds")

    # 7) grade balance across folds (each fold's TEST positives keep ~grade ratio)
    for fd in splits["real_lenses_kfold"]["folds"]:
        g = grade[np.array(fd["test_lens_rows"])]
        a, b = int((g=="A").sum()), int((g=="B").sum())
        if a == 0 or b == 0:
            fail.append(f"fold {fd['fold']}: grade not represented (A={a} B={b})")

    # 8) mlfp only in train (never val/test/target)
    mlfp = set(int(o) for o in mlfp_oids)
    if mlfp & val_r or mlfp & test_r or (mlfp & union_tgt):
        fail.append("ML-FP leaked into val/test/target")

    # 9) NO sim-deflector galaxy appears as a negative anywhere (train/val/target/test).
    #    A sim deflector is the host of a simulated positive; it must never be a negative.
    for name, s in [("train", train_r), ("val", val_r), ("test", test_r), ("target(any fold)", union_tgt)]:
        bad = s & sim_defl_set
        if bad:
            fail.append(f"{len(bad)} sim-deflector galaxies appear in {name} negatives")

    if fail:
        print("\n=== AUDIT FAILED ===")
        for f in fail: print("  ✗", f)
        sys.exit(1)
    print("\n=== AUDIT PASSED ===  (all disjointness/balance/rotation/grade rules hold)")


def _summary(splits, grade):
    st = splits["source_train"]; va = splits["val"]
    print("source_train: pos", len(st["sim_lens_rows"]),
          "| neg =", len(st["neg_mlfp"]["object_ids"]), "mlfp +",
          len(st["neg_bare_deflector"]["deflector_object_ids"]), "bare +",
          len(st["neg_random"]["object_ids"]), "random")
    print("val:          pos", len(va["sim_lens_rows"]),
          "| neg =", len(va["neg_bare_deflector"]["deflector_object_ids"]), "bare +",
          len(va["neg_random"]["object_ids"]), "random")
    print("test negs (shared):", len(splits["test_negatives_shared"]["object_ids"]))
    for fd in splits["real_lenses_kfold"]["folds"]:
        g = grade[np.array(fd["test_lens_rows"])]
        print(f"  fold {fd['fold']}: test {len(fd['test_lens_rows'])} lenses "
              f"(A={int((g=='A').sum())} B={int((g=='B').sum())}) | "
              f"target {len(fd['target_lens_rows'])} lenses + {len(fd['target_neg_object_ids'])} negs")


if __name__ == "__main__":
    main()
