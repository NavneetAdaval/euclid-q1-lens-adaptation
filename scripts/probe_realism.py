"""
probe_realism.py -- is the arc-on-real INJECTION realistic, controlling for deflector type?

The sim-lens vs random-neg number (0.94) is confounded: sim lenses sit on massive
ellipticals, random negatives are random galaxies, so the encoder separates on
deflector TYPE -- which REAL lenses do too. To isolate whether the simulated arc +
injection is realistic, we must compare against REAL lenses (same deflector type).

Read-outs:
  A. sim-lens  vs real-lens  -- THE realism test. LOW (~0.5-0.7) = sims look like real
                                lenses = injection realistic. HIGH = sims have an artifact.
  B. real-lens vs real-neg   -- THE control. This is the separability REAL lenses have
                                from random negatives (deflector type + real arc). Sets
                                what "correct" looks like.
  C. sim-lens  vs real-neg   -- (0.94 from before). Compare to B: if ~equal, the sim's
                                separability from negatives is the SAME as real lenses'
                                -> not a defect. If >> B, the sim has extra artifact.
  D. sim bare-defl vs real-lens -- do selected deflectors resemble real-lens hosts?

Verdict logic:
  A low AND C ~ B  -> sims behave like real lenses; earlier DA failure was about the
                      random negatives, not sim realism. Proceed (fix negatives, not sims).
  A high           -> the injection leaves an artifact; fix the sim.

Run from repo root:  python scripts\\probe_realism.py
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
REAL_LENS_NPZ = ROOT / "data" / "q1_lenses_processed" / "lens_cutouts.npz"
REAL_NEG_DIR = ROOT / "data" / "q1_unlabeled" / "shards"
N_NEG = 1000
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


def auc(a, b, enc, stretch):
    fa, fb = enc.encode(stretch(a)), enc.encode(stretch(b))
    y = np.r_[np.ones(len(fa)), np.zeros(len(fb))].astype(int)
    return cv_auroc(np.concatenate([fa, fb]), y)


def run(encoder):
    from zoobot_encoder import stretch_batch

    d = np.load(SIM_NPZ)
    sim_lens = d["lenses"].astype(np.float32)
    bare_defl = d["nonlenses"].astype(np.float32)
    real_lens = np.load(REAL_LENS_NPZ)["images"].astype(np.float32)
    real_neg = load_real_neg(REAL_NEG_DIR, N_NEG)
    print(f"sim lenses {len(sim_lens)}   real lenses {len(real_lens)}   "
          f"bare deflectors {len(bare_defl)}   real negatives {len(real_neg)}\n")

    A = auc(sim_lens, real_lens, encoder, stretch_batch)
    B = auc(real_lens, real_neg, encoder, stretch_batch)
    C = auc(sim_lens, real_neg, encoder, stretch_batch)
    D = auc(bare_defl, real_lens, encoder, stretch_batch)

    print(f"A. sim-lens   vs real-lens : {A:.4f}   <-- REALISM (low ~0.5-0.7 = sims look real)")
    print(f"B. real-lens  vs real-neg  : {B:.4f}   <-- CONTROL (real lenses' own separability)")
    print(f"C. sim-lens   vs real-neg  : {C:.4f}   (compare to B: ~equal = sim behaves like real)")
    print(f"D. bare-defl  vs real-lens : {D:.4f}   (deflector-host match)")
    print()
    print(f"   C - B = {C-B:+.3f}  (near 0 = sim's negative-separability matches real lenses')")
    print()
    if A < 0.72:
        print("VERDICT: sims are INDISTINGUISHABLE from real lenses (deflector-controlled).")
        if abs(C - B) < 0.06:
            print("         And sim-vs-neg matches real-vs-neg: the 0.94 was deflector-type,")
            print("         NOT a sim artifact. Earlier DA failure was the random negatives.")
        print("         Sims are realistic -> proceed; fix the negative set, not the sims.")
    elif A < 0.85:
        print("VERDICT: sims PARTIALLY realistic -- some residual injection artifact.")
    else:
        print("VERDICT: sims clearly separable from real lenses -> real injection artifact; fix sim.")


if __name__ == "__main__":
    from zoobot_encoder import FrozenZoobotEncoder
    run(FrozenZoobotEncoder())
