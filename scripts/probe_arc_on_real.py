"""
probe_arc_on_real.py -- did arc-on-real close the sim->real gap?

Uses the SAME frozen Zoobot encoder + stretch as probe_deflector_gap.py, so the
numbers sit directly alongside the old synthetic-deflector result (which was 1.0).

Three encoder AUROCs:
  1. sim-lens vs REAL-neg   -- THE verdict. Old synthetic sims were ~1.0 (trivially
                               separable -> DA collapse). Now should drop to ~0.7-0.85
                               (legitimate morphology gap) if the rebuild worked.
  2. sim-lens vs bare-defl  -- arc detectability. Should be HIGH (the arc is real signal).
  3. bare-defl vs real-neg  -- deflector realism. Should be LOW ~0.5-0.7: the bare
                               deflectors are REAL Euclid galaxies, so hard to tell from
                               real negatives. This is the direct before/after on the
                               AUROC=1.0 deflector gap the whole rebuild targeted.

Run from repo root:  python scripts\\probe_arc_on_real.py
Requires: zoobot_encoder.py alongside, and the sim file downloaded from Drive to
          data\\sim_arc_on_real\\arc_on_real_debug.npz
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
SIM_NPZ = ROOT / "data" / "sim_arc_on_real" / "arc_on_real_debug.npz"
REAL_NEG_DIR = ROOT / "data" / "q1_unlabeled" / "shards"
N_PER_CLASS = 1000
SEED = 0


def cv_auroc(feats, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, va in skf.split(feats, y):
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(feats[tr]), y[tr])
        oof[va] = clf.decision_function(sc.transform(feats[va]))
    return roc_auc_score(y, oof)


def load_real_neg(d, n, avoid_partition=4):
    """Real Euclid negatives, EXCLUDING the P4 test-negative partition."""
    files = sorted(glob.glob(str(d / "*.npz")))
    if not files:
        sys.exit(f"no shards in {d}")
    rng = np.random.default_rng(SEED)
    out = []
    for f in files:
        with np.load(f) as z:
            im = z["images"]
            if "partitions" in z.files and avoid_partition is not None:
                im = im[z["partitions"] != avoid_partition]
            out.append(np.asarray(im, dtype=np.float32))
        if sum(len(x) for x in out) >= n * 3:
            break
    allim = np.concatenate(out, axis=0)
    idx = rng.choice(len(allim), size=min(n, len(allim)), replace=False)
    return allim[idx]


def sub(arr, n):
    if len(arr) <= n:
        return arr
    rng = np.random.default_rng(SEED)
    return arr[rng.choice(len(arr), n, replace=False)]


def run(encoder):
    from zoobot_encoder import stretch_batch

    if not SIM_NPZ.exists():
        sys.exit(f"sim file not found: {SIM_NPZ}\n(download arc_on_real_debug.npz from Drive)")
    d = np.load(SIM_NPZ)
    sim_lens = sub(d["lenses"].astype(np.float32), N_PER_CLASS)
    bare_defl = sub(d["nonlenses"].astype(np.float32), N_PER_CLASS)
    real_neg = load_real_neg(REAL_NEG_DIR, N_PER_CLASS, avoid_partition=4)
    print(f"sim lenses {len(sim_lens)}   bare deflectors {len(bare_defl)}   "
          f"real negatives {len(real_neg)}\n")

    Lf = encoder.encode(stretch_batch(sim_lens))
    Bf = encoder.encode(stretch_batch(bare_defl))
    Rf = encoder.encode(stretch_batch(real_neg))

    y_lr = np.r_[np.ones(len(Lf)), np.zeros(len(Rf))].astype(int)
    auc_lens_vs_real = cv_auroc(np.concatenate([Lf, Rf]), y_lr)

    y_lb = np.r_[np.ones(len(Lf)), np.zeros(len(Bf))].astype(int)
    auc_lens_vs_bare = cv_auroc(np.concatenate([Lf, Bf]), y_lb)

    y_br = np.r_[np.ones(len(Bf)), np.zeros(len(Rf))].astype(int)
    auc_bare_vs_real = cv_auroc(np.concatenate([Bf, Rf]), y_br)

    print(f"encoder AUROC  sim-lens vs REAL-neg   : {auc_lens_vs_real:.4f}   "
          f"<-- THE verdict (old synthetic sims were ~1.0)")
    print(f"encoder AUROC  sim-lens vs bare-defl  : {auc_lens_vs_bare:.4f}   "
          f"(arc detectability, should be HIGH)")
    print(f"encoder AUROC  bare-defl vs real-neg  : {auc_bare_vs_real:.4f}   "
          f"(deflector realism, should be LOW ~0.5-0.7)")
    print()
    if auc_lens_vs_real > 0.95:
        print("VERDICT: sim lenses STILL trivially separable from real negatives.")
        print("         The rebuild did NOT close the gap -- diagnose the arc-injection signature.")
    elif auc_lens_vs_real > 0.85:
        print("VERDICT: partial. Gap reduced from 1.0 but still high; DA will struggle.")
    else:
        print("VERDICT: gap CLOSED to the legitimate-morphology level.")
        print("         arc-on-real worked -- DA now has a real chance.")


if __name__ == "__main__":
    from zoobot_encoder import FrozenZoobotEncoder
    run(FrozenZoobotEncoder())
