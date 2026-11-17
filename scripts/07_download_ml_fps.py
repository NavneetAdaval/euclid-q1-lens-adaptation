"""
Download cutouts for the Walmsley grade-X ML-flagged false-positive catalog.

These are objects the Q1 discovery engine flagged as candidate lenses but
expert visual inspection rejected (score < 0.5). They are confirmed non-lenses,
specifically lens-resembling non-lenses — the hard negatives that capture
the model's false-positive failure mode.

Processes only tiles already in your tiles_done.txt (the 250 tiles you

Download the ~3,372 ML-FP cutouts not already recovered.

Filters the grade-X catalog to:
  - Only objects in the user's existing 250-tile pool (drops the 491 in new tiles)
  - Only objects not already in ml_fp_cutouts/shards/ (drops the 1,077 already recovered)

For each remaining tile, downloads the mosaic and extracts only the
specific missing object_ids. Appends to existing ml_fp_tile{N}.npz
files where they already exist from the recovery step.
"""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
from astropy.wcs import WCS
import astropy.units as u
from tqdm import tqdm

# ---------- Paths ----------
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
ML_FP_CATALOG = Path("C:/Users/krris/Downloads/q1_discovery_engine_lens_catalog_old.csv")
MOSAIC_CATALOG = DATA_DIR / "q1_unlabeled" / "metadata" / "vis_mosaics.csv"
EXISTING_TILES_DONE = DATA_DIR / "q1_unlabeled" / "metadata" / "tiles_done.txt"

SHARD_DIR = DATA_DIR / "ml_fp_cutouts" / "shards"
MOSAIC_DIR = DATA_DIR / "ml_fp_cutouts" / "_mosaics_tmp"
META_DIR = DATA_DIR / "ml_fp_cutouts" / "metadata"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MOSAIC_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

DONE_FILE = META_DIR / "download_tiles_done.txt"

# ---------- Parameters ----------
CUTOUT_SIZE = 64
N_PARALLEL_DOWNLOADS = 5
EXPECTED_MIN_MOSAIC_SIZE = 1_000_000_000  # 1 GB


def cleanup_temp_mosaics():
    """Delete any leftover FITS from prior runs."""
    count = 0
    total_size = 0
    for f in MOSAIC_DIR.glob("*.fits"):
        try:
            total_size += f.stat().st_size
            f.unlink()
            count += 1
        except OSError:
            pass
    if count > 0:
        print(f"Cleaned {count} stale mosaic files ({total_size / 1e9:.1f} GB)")


def load_recovered_ids() -> set:
    """Read all object_ids already in ml_fp_cutouts/shards/."""
    recovered = set()
    for shard_path in SHARD_DIR.glob("ml_fp_tile*.npz"):
        d = np.load(shard_path)
        for oid in d["object_ids"]:
            recovered.add(int(oid))
    return recovered


def load_done() -> set:
    if not DONE_FILE.exists():
        return set()
    return {int(line.strip()) for line in DONE_FILE.read_text().splitlines() if line.strip()}


def mark_done(tile: int):
    with open(DONE_FILE, "a") as f:
        f.write(f"{tile}\n")


def process_one_tile(args: tuple) -> dict:
    """Worker: download mosaic, extract specified ML-FPs, append to shard."""
    tile_index, ml_fp_records, file_name = args

    from astroquery.esa.euclid import Euclid

    stats = {
        "tile": tile_index, "attempted": len(ml_fp_records),
        "succeeded": 0, "failed": 0, "error": None,
        "download_s": 0.0, "size_gb": 0.0,
    }

    mosaic_path = MOSAIC_DIR / file_name

    try:
        if mosaic_path.exists():
            mosaic_path.unlink()

        t0 = time.time()
        Euclid.get_product(file_name=file_name, output_file=str(mosaic_path))
        stats["download_s"] = time.time() - t0

        if not mosaic_path.exists():
            stats["error"] = "Downloaded file does not exist"
            return stats

        size_bytes = mosaic_path.stat().st_size
        stats["size_gb"] = size_bytes / 1e9

        if size_bytes < EXPECTED_MIN_MOSAIC_SIZE:
            stats["error"] = f"File too small ({size_bytes/1e6:.1f} MB)"
            mosaic_path.unlink()
            return stats

        # Extract cutouts for this tile's missing ML-FPs
        new_images, new_ids, new_id_strs = [], [], []
        with fits.open(mosaic_path, memmap=True) as hdul:
            sci_idx = 0
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and hdu.data.ndim == 2:
                    sci_idx = i
                    break
            wcs = WCS(hdul[sci_idx].header)
            data = hdul[sci_idx].data

            for row in ml_fp_records:
                coord = SkyCoord(row["right_ascension"] * u.deg,
                                 row["declination"] * u.deg, frame="icrs")
                try:
                    cut = Cutout2D(data, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                                   wcs=wcs, mode="strict")
                    if cut.data.shape != (CUTOUT_SIZE, CUTOUT_SIZE):
                        stats["failed"] += 1
                        continue
                    new_images.append(cut.data.astype(np.float32))
                    new_ids.append(int(row["object_id"]))
                    new_id_strs.append(str(row["id_str"]))
                    stats["succeeded"] += 1
                except (NoOverlapError, ValueError):
                    stats["failed"] += 1

        # Append to existing shard file or create new
        if new_images:
            out_path = SHARD_DIR / f"ml_fp_tile{tile_index}.npz"
            new_images_arr = np.stack(new_images).astype(np.float32)
            new_ids_arr = np.array(new_ids, dtype=np.int64)
            new_id_strs_arr = np.array(new_id_strs)

            if out_path.exists():
                # Append to existing file
                existing = np.load(out_path)
                combined_images = np.concatenate([existing["images"],
                                                  new_images_arr])
                combined_ids = np.concatenate([existing["object_ids"],
                                               new_ids_arr])
                combined_id_strs = np.concatenate([existing["id_strs"],
                                                   new_id_strs_arr])
                np.savez_compressed(out_path,
                                    images=combined_images,
                                    object_ids=combined_ids,
                                    id_strs=combined_id_strs)
            else:
                np.savez_compressed(out_path,
                                    images=new_images_arr,
                                    object_ids=new_ids_arr,
                                    id_strs=new_id_strs_arr)

    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
    finally:
        if mosaic_path.exists():
            try:
                mosaic_path.unlink()
            except OSError:
                pass

    if stats["error"] is None:
        mark_done(tile_index)

    return stats


def main():
    cleanup_temp_mosaics()

    # Load ML-FP catalog
    catalog = pd.read_csv(ML_FP_CATALOG)
    ml_fp_df = catalog[catalog["grade"] == "X"].copy()
    ml_fp_df["object_id"] = ml_fp_df["object_id"].astype("int64")
    ml_fp_df["tile_index"] = ml_fp_df["tile_index"].astype(int)
    print(f"Total grade-X ML-FPs: {len(ml_fp_df)}")

    # Load tiles you've already processed for the unlabeled pool
    existing_tiles = set(int(line.strip()) for line in
                         EXISTING_TILES_DONE.read_text().splitlines() if line.strip())
    print(f"Tiles in your existing 250-tile pool: {len(existing_tiles)}")

    # Filter to ML-FPs in your existing tile set
    ml_fp_in_pool = ml_fp_df[ml_fp_df["tile_index"].isin(existing_tiles)]
    print(f"ML-FPs in your existing tile pool: {len(ml_fp_in_pool)}")

    # Subtract already-recovered IDs
    recovered_ids = load_recovered_ids()
    print(f"ML-FPs already recovered (in ml_fp_cutouts/): {len(recovered_ids)}")

    ml_fp_to_fetch = ml_fp_in_pool[~ml_fp_in_pool["object_id"].isin(recovered_ids)]
    print(f"ML-FPs to download: {len(ml_fp_to_fetch)}")

    # Tiles that still have missing ML-FPs
    tiles_needed = sorted(ml_fp_to_fetch["tile_index"].unique())
    print(f"Tiles to download: {len(tiles_needed)}")

    # Resume support
    done = load_done()
    if done:
        print(f"Resuming — {len(done)} tiles already downloaded this run")
    tiles_to_process = [t for t in tiles_needed if t not in done]
    print(f"Remaining tiles: {len(tiles_to_process)}")
    print(f"Estimated bandwidth: ~{len(tiles_to_process) * 1.47:.0f} GB\n")

    # Load mosaic file-name lookup
    mosaics = pd.read_csv(MOSAIC_CATALOG)
    file_name_by_tile = dict(zip(mosaics["tile_index"], mosaics["file_name"]))

    # Build per-tile work units (only ML-FPs not yet recovered)
    work_units = []
    for tile in tiles_to_process:
        tile_rows = ml_fp_to_fetch[ml_fp_to_fetch["tile_index"] == tile]
        file_name = file_name_by_tile.get(tile)
        if file_name is None:
            print(f"  [tile {tile}] no mosaic file_name found, skipping")
            continue
        records = tile_rows[["id_str", "object_id",
                             "right_ascension", "declination"]].to_dict("records")
        work_units.append((tile, records, file_name))

    total_succeeded = 0
    total_failed = 0
    pbar = tqdm(total=len(work_units), desc="Tiles")

    with ProcessPoolExecutor(max_workers=N_PARALLEL_DOWNLOADS) as pool:
        futures = {pool.submit(process_one_tile, wu): wu[0] for wu in work_units}

        for fut in as_completed(futures):
            tile = futures[fut]
            try:
                stats = fut.result()
            except Exception as e:
                tqdm.write(f"  tile {tile}: worker crashed — {e}")
                pbar.update(1)
                continue

            if stats["error"]:
                tqdm.write(f"  tile {stats['tile']:>9}: FAILED — {stats['error']}")
            else:
                total_succeeded += stats["succeeded"]
                total_failed += stats["failed"]
                speed = stats["size_gb"] * 1024 / max(stats["download_s"], 0.1)
                tqdm.write(
                    f"  tile {stats['tile']:>9}: "
                    f"{stats['succeeded']:>3} ok / {stats['failed']:>2} fail | "
                    f"{stats['size_gb']:.2f} GB in {stats['download_s']:.0f}s "
                    f"({speed:.1f} MB/s)"
                )
            pbar.update(1)

    pbar.close()
    print(f"\n=== Run summary ===")
    print(f"Total ML-FP cutouts downloaded this run: {total_succeeded}")
    print(f"Cutout failures: {total_failed}")

    # Final verification
    final_recovered = load_recovered_ids()
    print(f"\nTotal ML-FPs now on disk: {len(final_recovered)}")
    print(f"Target: 4,449 (4,940 - 491 sacrificed in new tiles)")


if __name__ == "__main__":
    main()