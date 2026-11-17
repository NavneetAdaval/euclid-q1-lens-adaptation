"""
Production cutout downloader using multiprocessing (not threading).

astroquery.esa.euclid is not thread-safe — concurrent get_product calls in
threads corrupt each other's downloads. Each process here gets its own
Euclid client, eliminating the race.

Resumable via tiles_done.txt.
"""
import argparse
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
DATA_DIR = Path(__file__).parent.parent / "data"
META_DIR = DATA_DIR / "q1_unlabeled" / "metadata"
SHARD_DIR = DATA_DIR / "q1_unlabeled" / "shards"
MOSAIC_DIR = DATA_DIR / "q1_unlabeled" / "_mosaics_tmp"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MOSAIC_DIR.mkdir(parents=True, exist_ok=True)

DONE_FILE = META_DIR / "tiles_done.txt"

# ---------- Parameters ----------
CUTOUT_SIZE = 64
TILE_GALAXY_THRESHOLD = 500
N_PARALLEL_DOWNLOADS = 3
EXPECTED_MIN_MOSAIC_SIZE = 1_000_000_000  # 1 GB sanity check; full file ~1.47 GB


def cleanup_temp_mosaics():
    """Delete any FITS files left over in the temp mosaic dir.
    Called on startup; also safe to call periodically."""
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
        print(f"Cleaned {count} stale mosaic files "
              f"({total_size / 1e9:.1f} GB reclaimed)")


def load_done() -> set:
    if not DONE_FILE.exists():
        return set()
    return {int(line.strip()) for line in DONE_FILE.read_text().splitlines() if line.strip()}


def mark_done(tile: int):
    with open(DONE_FILE, "a") as f:
        f.write(f"{tile}\n")


def process_one_tile(args: tuple) -> dict:
    """
    Worker (subprocess): download mosaic, extract cutouts, save shard, delete mosaic.
    astroquery is imported here so each process has its own client.
    """
    tile_index, galaxy_records, file_name = args

    from astroquery.esa.euclid import Euclid

    stats = {"tile": tile_index, "attempted": len(galaxy_records),
             "succeeded": 0, "failed": 0, "error": None,
             "download_s": 0.0, "extract_s": 0.0, "size_gb": 0.0}

    mosaic_path = MOSAIC_DIR / file_name

    try:
        # Remove stale partial files from previous interrupted runs
        if mosaic_path.exists():
            mosaic_path.unlink()

        # --- Download ---
        t0 = time.time()
        Euclid.get_product(file_name=file_name, output_file=str(mosaic_path))
        stats["download_s"] = time.time() - t0

        if not mosaic_path.exists():
            stats["error"] = "Downloaded file does not exist"
            return stats

        size_bytes = mosaic_path.stat().st_size
        stats["size_gb"] = size_bytes / 1e9

        # Sanity check: if the file is way too small, the download was incomplete
        if size_bytes < EXPECTED_MIN_MOSAIC_SIZE:
            stats["error"] = (f"File too small ({size_bytes/1e6:.1f} MB) — "
                              f"download likely incomplete")
            mosaic_path.unlink()
            return stats

        # --- Extract cutouts ---
        t0 = time.time()
        images, ids, partitions = [], [], []
        with fits.open(mosaic_path, memmap=True) as hdul:
            sci_idx = 0
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and hdu.data.ndim == 2:
                    sci_idx = i
                    break
            wcs = WCS(hdul[sci_idx].header)
            data = hdul[sci_idx].data

            for row in galaxy_records:
                coord = SkyCoord(row["right_ascension"] * u.deg,
                                 row["declination"] * u.deg, frame="icrs")
                try:
                    cut = Cutout2D(data, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                                   wcs=wcs, mode="strict")
                    if cut.data.shape != (CUTOUT_SIZE, CUTOUT_SIZE):
                        stats["failed"] += 1
                        continue
                    images.append(cut.data.astype(np.float32))
                    ids.append(int(row["object_id"]))
                    partitions.append(int(row["partition"]))
                    stats["succeeded"] += 1
                except (NoOverlapError, ValueError):
                    stats["failed"] += 1

        stats["extract_s"] = time.time() - t0

        # --- Save shard ---
        if images:
            out = SHARD_DIR / f"shard_tile{tile_index}.npz"
            np.savez_compressed(out,
                                images=np.stack(images),
                                object_ids=np.array(ids, dtype=np.int64),
                                partitions=np.array(partitions, dtype=np.int8))

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


def main(limit_tiles=None, n_parallel=N_PARALLEL_DOWNLOADS):
    cleanup_temp_mosaics()
    catalog = pd.read_csv(META_DIR / "partitioned_catalog_with_tiles.csv")
    mosaics = pd.read_csv(META_DIR / "vis_mosaics.csv")

    counts = catalog["tile_index"].value_counts()
    kept_tiles = counts[counts >= TILE_GALAXY_THRESHOLD].index
    catalog = catalog[catalog["tile_index"].isin(kept_tiles)]
    print(f"After threshold {TILE_GALAXY_THRESHOLD}: {len(catalog)} galaxies, "
          f"{catalog['tile_index'].nunique()} tiles")

    done = load_done()
    if done:
        print(f"Resuming — {len(done)} tiles already completed")

    file_name_by_tile = dict(zip(mosaics["tile_index"], mosaics["file_name"]))

    tiles_to_process = sorted(catalog["tile_index"].unique())
    tiles_to_process = [t for t in tiles_to_process if t not in done]
    if limit_tiles:
        tiles_to_process = tiles_to_process[:limit_tiles]

    print(f"{len(tiles_to_process)} tiles to process "
          f"with {n_parallel} parallel processes")
    print(f"Estimated total bandwidth: ~{len(tiles_to_process) * 1.47:.0f} GB\n")

    work_units = []
    for tile in tiles_to_process:
        tg = catalog[catalog["tile_index"] == tile]
        file_name = file_name_by_tile.get(tile)
        if file_name is None:
            print(f"  [tile {tile}] no mosaic file_name found, skipping")
            continue
        records = tg[["object_id", "right_ascension", "declination",
                      "partition"]].to_dict("records")
        work_units.append((tile, records, file_name))

    total_succeeded = 0
    total_failed = 0
    pbar = tqdm(total=len(work_units), desc="Tiles")

    with ProcessPoolExecutor(max_workers=n_parallel) as pool:
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
                    f"{stats['succeeded']:>4} ok / {stats['failed']:>3} fail | "
                    f"{stats['size_gb']:.2f} GB in {stats['download_s']:.0f}s "
                    f"({speed:.1f} MB/s)"
                )
            pbar.update(1)

    pbar.close()
    print(f"\n=== Run summary ===")
    print(f"Cutouts saved: {total_succeeded}")
    print(f"Cutout failures: {total_failed}")
    if total_failed + total_succeeded:
        print(f"Failure rate: "
              f"{total_failed/(total_succeeded+total_failed)*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-tiles", type=int, default=None,
                        help="Process only the first N tiles (for testing)")
    parser.add_argument("--parallel", type=int, default=N_PARALLEL_DOWNLOADS,
                        help="Number of parallel processes (default 3)")
    args = parser.parse_args()
    main(limit_tiles=args.limit_tiles, n_parallel=args.parallel)