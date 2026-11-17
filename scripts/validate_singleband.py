"""validate_singleband.py  --  does the regenerated raw single-band sim close the
brightness shortcut? Run on a SMALL batch (~200) before committing to 30k.

Checks, in order:
  0. SELF-CHECK: StandardizeAsinh(c*x) == StandardizeAsinh(x). Confirms the
     normalizer is scale-invariant (so the /max was harmless and the fix is the
     un-clipped noise floor). If this fails, the whole diagnosis is suspect -- stop.
  1. Brightness-shortcut AUROC: can mean/std/p99 separate sim from real Euclid?
       ~0.99 -> still leaking (raise --lens-mag, regen a small batch, retry)
       ~0.50 -> brightness shortcut CLOSED -> safe to generate the full 30k
  2. Post-norm stat table vs known references.
  3. Montage saved to sim_vs_real_v2.png.

NOTE: this only tests the BRIGHTNESS shortcut, which is what the regeneration
fixes. A residual morphology gap (sims cleaner/rounder than real) is expected and
is domain adaptation's job, not the generator's -- it won't show up strongly in
mean/std/p99 and should NOT block the full generation.

Usage (locally, from your scripts dir):
  python validate_singleband.py --sim-dir ./val_batch --data-dir C:/Users/krris/ML/lens_paper/data
"""
import argparse
import glob
import os

import numpy as np
import torch

from lens_dataset import StandardizeAsinh, LensDataset


def normfn(img2d, norm):
    out = norm(torch.from_numpy(np.ascontiguousarray(img2d, dtype=np.float32)).unsqueeze(0))
    return np.asarray(out).reshape(-1)


def load_sim(path):
    a = np.asarray(np.load(path, allow_pickle=True))
    if a.dtype == object:
        a = np.asarray(a[0])
    a = a.astype(np.float32)
    if a.ndim == 3:                      # safety: collapse if somehow multi-channel
        a = a[..., 0] if a.shape[-1] in (1, 3) else a[0]
    return a


def feats(stack):
    return np.stack([stack.mean(1), stack.std(1),
                     np.percentile(stack, 99, axis=1)], axis=1)


def shortcut_auroc(pos, neg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    P, N = np.stack(pos), np.stack(neg)
    X = np.concatenate([feats(P), feats(N)])
    y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
    proba = cross_val_predict(LogisticRegression(max_iter=1000), X, y,
                              cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def table(name, flat):
    X = np.stack(flat)
    print(f"  {name:26s} mean={X.mean():.3f}  std={X.std():.3f}  p99={np.percentile(X,99):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True, help="dir of regenerated single-band .npy sims")
    ap.add_argument("--data-dir", required=True, help="data root with real Euclid shards")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    norm = StandardizeAsinh()

    # 0. scale-invariance self-check ----------------------------------------
    files = glob.glob(os.path.join(args.sim_dir, "*.npy"))
    if not files:
        raise SystemExit(f"no .npy found in {args.sim_dir}")
    probe = load_sim(files[0])
    a = normfn(probe, norm)
    b = normfn(probe * 7.0 + 3.0, norm)            # affine: scale 7, offset 3
    max_diff = float(np.max(np.abs(a - b)))
    print(f"[self-check] StandardizeAsinh affine-invariance max|diff| = {max_diff:.2e}")
    print("  (near 0 => normalizer ignores scale/offset, as the diagnosis assumed)")

    # 1./2. load sims and real, compare -------------------------------------
    rng = np.random.default_rng(0)
    files = list(rng.choice(files, min(args.n, len(files)), replace=False))
    sim_flat = [normfn(load_sim(f), norm) for f in files]
    print(f"\nloaded {len(sim_flat)} regenerated sims")

    ds = LensDataset("target_unlabeled", args.data_dir, verbose=False)
    idx = rng.choice(len(ds), min(args.n, len(ds)), replace=False)
    real_flat = [np.asarray(ds.images[i]).reshape(-1) for i in idx]

    print("\n== post-StandardizeAsinh stats ==")
    table("regenerated sim", sim_flat)
    table("real Euclid (target)", real_flat)
    print("  real Euclid (known ref)    mean=0.470  std=1.330  p99=4.71")
    print("  Model II / leaky (ref)     mean=0.036  std=0.850  p99=2.00")
    print("  Model IV /max+clip (ref)   mean=1.076  std=1.828  p99=6.40")

    auc = shortcut_auroc(sim_flat, real_flat)
    print("\n" + "=" * 62)
    print(f"  BRIGHTNESS-SHORTCUT AUROC (sim vs real): {auc:.4f}")
    print("  ~0.99 -> still leaking; raise --lens-mag, regen small batch, retry")
    print("  ~0.50 -> shortcut closed; safe to generate the full 30k")
    print("=" * 62)

    # cusp check: a too-sharp central pixel is a non-physical shortcut --------
    def core_stats(arr2d_list):
        ratios, central = [], []
        for a in arr2d_list:
            a = a.astype(np.float64)
            ratios.append(a.max() / (np.percentile(a, 99) + 1e-9))
            cy, cx = np.unravel_index(np.argmax(a), a.shape)
            central.append(abs(cy - 32) <= 3 and abs(cx - 32) <= 3)
        return np.median(ratios), np.mean(central)
    sim_imgs = [load_sim(f) for f in files[:300]]
    real_imgs = [np.asarray(ds.images[i]).squeeze() for i in idx[:300]]
    sr, sc = core_stats(sim_imgs)
    rr, rc = core_stats(real_imgs)
    print(f"\n  CORE CUSP  peak/p99: sim {sr:.1f}  vs real {rr:.1f}   "
          f"(want sim within ~0.3 of real)")
    print(f"  centered-peak frac: sim {sc:.2f} vs real {rc:.2f}")
    print("  sim >> real -> deflector core too cuspy; lower n_sersic / raise r_sersic")

    # 3. montage ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 8, figsize=(16, 4))
        for j in range(8):
            s = load_sim(files[j])
            ax[0, j].imshow(normfn(s, norm).reshape(s.shape)); ax[0, j].axis("off")
            ax[1, j].imshow(np.asarray(ds.images[idx[j]]).squeeze()); ax[1, j].axis("off")
        ax[0, 0].set_title("regenerated sim", loc="left")
        ax[1, 0].set_title("real Euclid", loc="left")
        fig.tight_layout(); fig.savefig("sim_vs_real_v2.png", dpi=110)
        print(f"montage -> {os.path.abspath('sim_vs_real_v2.png')}")
    except Exception as e:
        print(f"(montage skipped: {e})")


if __name__ == "__main__":
    main()