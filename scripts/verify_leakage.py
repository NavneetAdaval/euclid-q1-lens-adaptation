r"""
verify_leakage.py -- standalone checks for the two leakage vectors that
build_splits.py's audit is designed around, but verified INDEPENDENTLY here
against the actually-drawn splits (not the raw pool).

  (2) VISUAL DUPLICATES among the 489 real lenses (perceptual hash + pixel-corr
      confirmation). A confirmed dup with different object_ids straddling folds is
      flagged. Rows in CLEARED_DUP_ROWS were inspected (sky-separation matrix +
      montage) and confirmed to be DISTINCT lenses that merely share the canonical
      bright-elliptical+ring morphology -- they are whitelisted and reported as
      "inspected, distinct" rather than as leaks.

  (3) SIM-DEFLECTOR <-> NEGATIVE overlap, checked against the negatives ACTUALLY
      DRAWN in splits.json (train/val/target/test), NOT the raw on-disk pool.
      build_splits purges sim deflectors at draw time, so the raw shards still
      physically contain them -- but what matters is whether they entered a split.
      For context we also print the raw-pool overlap, clearly labelled as raw.

Run from repo root:  python scripts\verify_leakage.py
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT      = Path(".")
LENS_NPZ  = ROOT / "data" / "q1_lenses_processed" / "lens_cutouts.npz"
SIM_DIR   = ROOT / "data" / "sim_arc_on_real"
NEG_DIR   = ROOT / "data" / "q1_unlabeled" / "shards"
SPLITS    = ROOT / "data" / "splits" / "splits.json"

PHASH_SIZE   = 16
HAMMING_MAX  = 12
PIXCORR_MIN  = 0.92

# Rows inspected via inspect_dupes.py and CONFIRMED distinct (sky separations all
# >> 5"; different sky positions; shared morphology only). Whitelisted so the phash
# false-positive cluster is not re-reported as a hard leak. If lens_cutouts.npz is
# ever rebuilt (row order changes), RE-INSPECT and update this list.
CLEARED_DUP_ROWS = {238, 321, 388, 415, 427, 474, 479}


# ----------------------- perceptual hash -----------------------
def phash(img: np.ndarray, size: int = PHASH_SIZE) -> np.ndarray:
    """Robust downsample->median-threshold hash. Returns a flat bool array (size*size)."""
    x = np.nan_to_num(np.asarray(img, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    # asinh stretch so bright cores don't dominate the hash (match encoder-side intuition)
    x = np.arcsinh(x / (np.median(np.abs(x)) + 1e-8))
    h, w = x.shape
    # block-average to size x size
    hb, wb = h // size, w // size
    if hb == 0 or wb == 0:
        # tiny image; fall back to resize by simple striding
        idx_h = np.linspace(0, h - 1, size).astype(int)
        idx_w = np.linspace(0, w - 1, size).astype(int)
        small = x[np.ix_(idx_h, idx_w)]
    else:
        small = x[:hb * size, :wb * size].reshape(size, hb, size, wb).mean(axis=(1, 3))
    return (small > np.median(small)).ravel()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def load_lenses():
    with np.load(LENS_NPZ, allow_pickle=True) as d:
        imgs = d["images"].astype(np.float32)
        oid  = d["object_id"].astype("int64")
        ra   = d["ra"].astype("float64") if "ra" in d else None
        dec  = d["dec"].astype("float64") if "dec" in d else None
    return imgs, oid, ra, dec


def load_fold_of_row():
    """row index -> fold number it is TESTED in (from splits.json)."""
    s = json.load(open(SPLITS))
    row2fold = {}
    for fd in s["real_lenses_kfold"]["folds"]:
        for r in fd["test_lens_rows"]:
            row2fold[int(r)] = fd["fold"]
    return row2fold, s


# ---------------- (2) visual duplicate detection ----------------
def check_visual_dupes(imgs, oid, row2fold):
    n = len(imgs)
    hashes = [phash(imgs[i]) for i in range(n)]
    # brute-force pairwise (n=489 -> ~119k pairs, trivial)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(hashes[i], hashes[j]) <= HAMMING_MAX:
                candidates.append((i, j))

    confirmed = []
    for i, j in candidates:
        a = np.nan_to_num(imgs[i].ravel().astype(np.float64))
        b = np.nan_to_num(imgs[j].ravel().astype(np.float64))
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r >= PIXCORR_MIN:
            confirmed.append((i, j, r, int(oid[i]), int(oid[j]),
                              row2fold.get(i), row2fold.get(j)))
    return candidates, confirmed


def is_cleared(i, j):
    """True if both rows were inspected and confirmed distinct (whitelisted)."""
    return i in CLEARED_DUP_ROWS and j in CLEARED_DUP_ROWS


# ---------------- (3) sim-deflector / negative overlap ----------------
def load_sim_deflector_oids():
    oids = set()
    for f in sorted(glob.glob(str(SIM_DIR / "shard_*.npz"))):
        with np.load(f) as d:
            oids.update(int(o) for o in d["object_id"].astype("int64"))
    return oids


def load_neg_oids_by_part():
    by_part = defaultdict(set)
    for f in sorted(glob.glob(str(NEG_DIR / "*.npz"))):
        with np.load(f) as d:
            oids = d["object_ids"].astype("int64")
            parts = d["partitions"].astype(int)
            for o, p in zip(oids, parts):
                by_part[int(p)].add(int(o))
    return by_part


def main():
    fail, warn = [], []

    imgs, oid, ra, dec = load_lenses()
    row2fold, splits = load_fold_of_row()
    print(f"loaded {len(imgs)} real lenses; {len(set(oid.tolist()))} unique object_id")

    # -------- (2) visual dupes --------
    cand, conf = check_visual_dupes(imgs, oid, row2fold)
    print(f"\n(2) VISUAL DUPLICATES")
    print(f"    phash candidates (hamming<= {HAMMING_MAX}): {len(cand)}")
    print(f"    confirmed (pixel r>= {PIXCORR_MIN}): {len(conf)}")
    if conf:
        cleared, genuine = [], []
        for rec in conf:
            i, j, r, oi, oj, fi, fj = rec
            (cleared if is_cleared(i, j) else genuine).append(rec)
        if cleared:
            print(f"    {len(cleared)} confirmed pair(s) are WHITELISTED (inspected, distinct):")
            for i, j, r, oi, oj, fi, fj in cleared:
                print(f"      rows {i},{j}  r={r:.3f}  fold {fi} vs {fj}  [cleared: distinct look-alikes]")
        for i, j, r, oi, oj, fi, fj in genuine:
            same_oid = "SAME oid" if oi == oj else "diff oid"
            straddle = (fi is not None and fj is not None and fi != fj)
            tag = "CROSS-FOLD" if straddle else "same/na fold"
            print(f"      rows {i},{j}  r={r:.3f}  {same_oid}  fold {fi} vs {fj}  [{tag}]")
            if oi != oj and straddle:
                fail.append(f"NEW visual duplicate rows {i},{j} (diff object_id) straddle folds {fi},{fj}")
            elif oi != oj:
                warn.append(f"NEW visual duplicate rows {i},{j} (diff object_id) same/one fold")
        if not genuine:
            print("    -> no NEW duplicates beyond the whitelisted cluster. Clean.")
    else:
        print("    -> none confirmed. No visual-duplicate leakage.")

    # -------- (3) sim-deflector / negative overlap --------
    print(f"\n(3) SIM-DEFLECTOR <-> NEGATIVE OVERLAP")
    sim_defl = load_sim_deflector_oids()
    print(f"    sim deflectors: {len(sim_defl)} unique object_id")

    # (3a) THE ONE THAT MATTERS: negatives actually DRAWN into splits.json.
    drawn = {
        "train":  set(int(o) for o in splits["source_train"]["neg_random"]["object_ids"]),
        "val":    set(int(o) for o in splits["val"]["neg_random"]["object_ids"]),
        "test":   set(int(o) for o in splits["test_negatives_shared"]["object_ids"]),
    }
    tgt = set()
    for fd in splits["real_lenses_kfold"]["folds"]:
        tgt |= set(int(o) for o in fd["target_neg_object_ids"])
    drawn["target(all folds)"] = tgt
    print("    -- overlap with negatives DRAWN INTO splits.json (this is what matters) --")
    any_drawn_leak = False
    for role, s in drawn.items():
        ov = sim_defl & s
        print(f"      {role}: {len(s)} drawn negs, sim-deflector overlap = {len(ov)}")
        if ov:
            any_drawn_leak = True
            fail.append(f"{len(ov)} sim deflectors DRAWN into {role} negatives")
    if not any_drawn_leak:
        print("      -> no sim deflector entered any drawn negative role. Clean.")

    # (3b) CONTEXT ONLY: raw on-disk pool still physically contains sim deflectors
    #      (build_splits purges them at draw time, so 3a is what governs correctness).
    by_part = load_neg_oids_by_part()
    partname = {0: "P0(SSL)", 1: "P1(src)", 2: "P2(target)", 3: "P3(val)", 4: "P4(TEST)"}
    print("    -- raw on-disk pool overlap (CONTEXT ONLY; purged at draw time) --")
    for p in sorted(by_part):
        ov = sim_defl & by_part[p]
        if ov:
            print(f"      {partname.get(p, f'P{p}')}: {len(ov)} sim deflectors present in raw shards "
                  f"(excluded from splits by build_splits purge)")

    # -------- verdict --------
    print("\n================ VERDICT ================")
    if warn:
        print("WARNINGS (non-fatal, worth knowing):")
        for w in warn: print("  !", w)
    if fail:
        print("HARD LEAKS:")
        for f in fail: print("  ✗", f)
        sys.exit(1)
    print("No hard leakage found on the two vectors the split audit can't see. ✓")
    if not warn:
        print("No warnings either — both vectors clean.")


if __name__ == "__main__":
    main()
