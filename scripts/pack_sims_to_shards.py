"""pack_sims_to_shards.py  --  pack loose single-band .npy lenses into .npz shards
matching the existing sim-shard schema, so 06_create_splits.py runs UNCHANGED.

Existing schema (from your sim_train_*.npz): images (N,64,64) float32, 5000/shard.
The dataset only ever reads `images` (subclass / axion_mass_log10 are dead weight
for lens-finding and are NOT written). Output shards are named to match the
splitter's globs: sim_train_*.npz / sim_val_*.npz.

Counts are forced EXACTLY to --target so the splitter's 1:1 invariant
(sim_train_count == 30000) passes on the nose even if generation dropped a few
images on simulation failures. If there are more than --target, it truncates;
if fewer, it errors (you must generate more -- silent padding would duplicate).

Usage:
  # train: 30000 -> 6 shards of 5000
  python pack_sims_to_shards.py --src ./lenses_30k --prefix sim_train --target 30000 \
      --out C:\\Users\\krris\\ML\\lens_paper\\data\\sim_lenses\\shards
  # val: 9000 -> 2 shards (5000 + 4000)
  python pack_sims_to_shards.py --src ./lenses_9k_val --prefix sim_val --target 9000 \
      --out C:\\Users\\krris\\ML\\lens_paper\\data\\sim_lenses\\shards
"""
import argparse
import glob
import os
import numpy as np

PER_SHARD = 5000   # match existing shard size


def load_one(path):
    a = np.asarray(np.load(path, allow_pickle=True))
    if a.dtype == object:
        a = np.asarray(a[0])
    a = a.astype(np.float32)
    if a.ndim == 3:                       # safety: collapse if multi-channel slipped in
        a = a[..., 0] if a.shape[-1] in (1, 3) else a[0]
    if a.shape != (64, 64):
        raise ValueError(f"{path}: expected (64,64), got {a.shape}")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of single-band .npy lenses")
    ap.add_argument("--out", required=True, help="sim_lenses/shards dir to write into")
    ap.add_argument("--prefix", required=True, choices=["sim_train", "sim_val", "sim_nonlens"])
    ap.add_argument("--target", type=int, required=True, help="exact image count to write")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.npy")))
    print(f"found {len(files)} .npy in {args.src}; target {args.target}")
    if len(files) < args.target:
        raise SystemExit(
            f"ONLY {len(files)} images, need {args.target}. Generate "
            f"{args.target - len(files)} more (don't pad -- it would duplicate)."
        )
    files = files[:args.target]           # deterministic truncate (sorted order)

    os.makedirs(args.out, exist_ok=True)
    # warn if shards with this prefix already exist (old Model II shards to replace)
    existing = sorted(glob.glob(os.path.join(args.out, f"{args.prefix}_*.npz")))
    if existing:
        print(f"  NOTE: {len(existing)} existing {args.prefix}_*.npz shards in --out.")
        print( "  Move/delete the OLD Model II shards first, or the splitter will read")
        print( "  both old and new and overcount. Existing:",
              [os.path.basename(e) for e in existing])

    n_shards = (args.target + PER_SHARD - 1) // PER_SHARD
    written = 0
    for s in range(n_shards):
        chunk = files[s * PER_SHARD:(s + 1) * PER_SHARD]
        imgs = np.stack([load_one(f) for f in chunk]).astype(np.float32)  # (n,64,64)
        out_path = os.path.join(args.out, f"{args.prefix}_{s:05d}.npz")
        np.savez(out_path, images=imgs)                                    # ONLY 'images'
        written += imgs.shape[0]
        print(f"  wrote {os.path.basename(out_path)}: {imgs.shape}  "
              f"[min {imgs.min():.4g} max {imgs.max():.4g} neg-frac {(imgs<0).mean():.3f}]")

    print(f"done: {written} images across {n_shards} shards (prefix {args.prefix}).")
    if written != args.target:
        raise SystemExit(f"BUG: wrote {written}, expected {args.target}")


if __name__ == "__main__":
    main()
