"""
Consolidate DeepLense Model II simulated lens images into .npz shards
matching the format used by Dataset 4 (unlabeled Euclid).

Source:  C:/Users/krris/DeepLenseSim/Model_II/train2/{no_sub,cdm,axion}/*.npy
         C:/Users/krris/DeepLenseSim/Model_II/val2/{no_sub,cdm,axion}/*.npy
Output:  data/sim_lenses/shards/sim_train_NNNNN.npz
         data/sim_lenses/shards/sim_val_NNNNN.npz

Each shard contains:
  images:           (N, 64, 64) float32   — pooled across the three subclasses
  subclass:         (N,) uint8            — 0=no_sub, 1=cdm, 2=axion
  axion_mass_log10: (N,) float32          — log10(axion mass) where applicable, NaN otherwise

For the binary lens-finding task, all three subclasses are treated as
positive (lens) examples. The subclass / axion_mass columns are retained
for stratified analysis later.
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------- Paths ----------
PROJECT_DIR = Path(__file__).parent.parent
SOURCE_DIR = Path("C:/Users/krris/DeepLenseSim/Model_II")
OUT_DIR = PROJECT_DIR / "data" / "sim_lenses" / "shards"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARD_SIZE = 5000
SUBCLASS_TO_INT = {"no_sub": 0, "cdm": 1, "axion": 2}


def load_one(path: Path, cls_name: str) -> tuple[np.ndarray, float]:
    """Load a single Model II .npy file, returning (image, axion_mass_or_nan).

    Handles the axion object-array format where each file is np.array([image, mass]).
    """
    raw = np.load(path, allow_pickle=True)
    if cls_name == "axion":
        img = raw[0]
        # raw[1] is the axion mass (10^-24 to 10^-22 in the script)
        mass = float(raw[1])
        return img.astype(np.float32), float(np.log10(mass))
    return raw.astype(np.float32), float("nan")


def process_split(split_name: str, split_dir_name: str) -> None:
    """Walk one split (train or val), build shards, save."""
    split_dir = SOURCE_DIR / split_dir_name
    if not split_dir.exists():
        print(f"WARNING: {split_dir} does not exist, skipping")
        return

    # Collect all (path, cls_name) tuples first so we know the total
    work = []
    for cls_name in SUBCLASS_TO_INT.keys():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            print(f"  WARNING: missing {cls_dir}")
            continue
        files = sorted(cls_dir.glob("*.npy"))
        work.extend([(f, cls_name) for f in files])
        print(f"  {split_name}/{cls_name}: {len(files)} files")

    if not work:
        print(f"  No files found for {split_name}")
        return

    print(f"  Total {split_name}: {len(work)} files. Shuffling for class mixing...")
    rng = np.random.default_rng(seed=42)
    rng.shuffle(work)  # so each shard has a mix of all three subclasses

    # Build shards
    images_buf: list[np.ndarray] = []
    subclass_buf: list[int] = []
    mass_buf: list[float] = []
    shard_idx = 0
    n_failures = 0

    pbar = tqdm(work, desc=f"  Loading {split_name}")
    for path, cls_name in pbar:
        try:
            img, mass = load_one(path, cls_name)
            if img.shape != (64, 64):
                n_failures += 1
                continue
            images_buf.append(img)
            subclass_buf.append(SUBCLASS_TO_INT[cls_name])
            mass_buf.append(mass)
        except Exception as e:
            n_failures += 1
            continue

        if len(images_buf) >= SHARD_SIZE:
            flush_shard(images_buf, subclass_buf, mass_buf,
                        split_name, shard_idx)
            shard_idx += 1
            images_buf, subclass_buf, mass_buf = [], [], []

    # Flush remaining partial shard
    if images_buf:
        flush_shard(images_buf, subclass_buf, mass_buf, split_name, shard_idx)
        shard_idx += 1

    print(f"  Saved {shard_idx} shards for {split_name}")
    if n_failures:
        print(f"  WARNING: {n_failures} files failed to load")


def flush_shard(images: list, subclass: list, mass: list,
                split_name: str, shard_idx: int) -> None:
    arr_images = np.stack(images).astype(np.float32)
    arr_subclass = np.array(subclass, dtype=np.uint8)
    arr_mass = np.array(mass, dtype=np.float32)
    out = OUT_DIR / f"sim_{split_name}_{shard_idx:05d}.npz"
    np.savez_compressed(out, images=arr_images,
                        subclass=arr_subclass,
                        axion_mass_log10=arr_mass)


def main():
    print("=" * 60)
    print("Consolidating DeepLense Model II simulations")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUT_DIR}")
    print()

    print("Processing TRAIN split:")
    process_split("train", "train")
    print()
    print("Processing VAL split:")
    process_split("val", "val")

    # Audit
    print("\n=== Audit ===")
    for pattern in ["sim_train_*.npz", "sim_val_*.npz"]:
        shards = sorted(OUT_DIR.glob(pattern))
        if not shards:
            print(f"  {pattern}: no shards")
            continue
        total = 0
        subclass_counts = np.zeros(3, dtype=int)
        for s in shards:
            d = np.load(s)
            total += len(d["images"])
            for sc in d["subclass"]:
                subclass_counts[sc] += 1
        size_mb = sum(s.stat().st_size for s in shards) / 1e6
        print(f"  {pattern}: {len(shards)} shards, {total:,} images, {size_mb:.1f} MB")
        print(f"    subclass counts: no_sub={subclass_counts[0]:,}  "
              f"cdm={subclass_counts[1]:,}  axion={subclass_counts[2]:,}")


if __name__ == "__main__":
    main()