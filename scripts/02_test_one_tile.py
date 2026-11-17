"""
Test the cutout pipeline end-to-end on a single tile.

Validates:
  - Mosaic download works via Euclid.get_product()
  - Actual mosaic file size (vs estimate)
  - Time per tile (download + cutout extraction)
  - Cutout failure rate (real, not estimated)
  - FITS HDU structure (which extension has the science image)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
from astropy.wcs import WCS
import astropy.units as u
from astroquery.esa.euclid import Euclid

DATA_DIR = Path(__file__).parent.parent / "data"
META_DIR = DATA_DIR / "q1_unlabeled" / "metadata"
SHARD_DIR = DATA_DIR / "q1_unlabeled" / "shards"
MOSAIC_DIR = DATA_DIR / "q1_unlabeled" / "_mosaics_tmp"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
MOSAIC_DIR.mkdir(parents=True, exist_ok=True)

CUTOUT_SIZE = 64
KEEP_MOSAIC = False  # set True if you want to inspect the FITS afterwards


def pick_test_tile(catalog: pd.DataFrame) -> int:
    """Pick a tile with a moderate galaxy count for a representative test."""
    counts = catalog["tile_index"].value_counts()
    # Pick a tile near the median of our kept tiles (~800-1000 galaxies)
    target_count = counts.median()
    closest_idx = (counts - target_count).abs().idxmin()
    print(f"Selected tile {closest_idx} with {counts[closest_idx]} galaxies "
          f"(median is {int(target_count)})")
    return int(closest_idx)


def download_mosaic(tile_index: int, file_name: str, file_path: str) -> Path:
    """Use astroquery to fetch the mosaic FITS into MOSAIC_DIR."""
    print(f"\nDownloading mosaic for tile {tile_index}...")
    t0 = time.time()
    # Euclid.get_product fetches by file_name; output goes to current dir by default
    Euclid.get_product(file_name=file_name, output_file=str(MOSAIC_DIR / file_name))
    dt = time.time() - t0
    out = MOSAIC_DIR / file_name
    size_gb = out.stat().st_size / 1e9
    print(f"  Downloaded in {dt:.0f}s, file size {size_gb:.2f} GB "
          f"({size_gb * 1024 / dt:.1f} MB/s)")
    return out


def extract_cutouts(fits_path: Path, galaxies: pd.DataFrame) -> tuple[int, int]:
    """Extract 64x64 cutouts for all galaxies in this tile."""
    print(f"\nOpening mosaic FITS and extracting cutouts...")
    t0 = time.time()
    successes, failures = 0, 0
    images, ids, partitions = [], [], []

    with fits.open(fits_path, memmap=True) as hdul:
        print(f"  HDU info:")
        hdul.info()
        # Pick the HDU with image data — usually [0], sometimes [1] for compressed FITS
        sci_idx = 0
        for i, hdu in enumerate(hdul):
            if hdu.data is not None and hdu.data.ndim == 2:
                sci_idx = i
                break
        print(f"  Using HDU {sci_idx} for science image")

        wcs = WCS(hdul[sci_idx].header)
        data = hdul[sci_idx].data

        for _, row in galaxies.iterrows():
            coord = SkyCoord(row["right_ascension"] * u.deg,
                             row["declination"] * u.deg, frame="icrs")
            try:
                cut = Cutout2D(data, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                               wcs=wcs, mode="strict")
                if cut.data.shape != (CUTOUT_SIZE, CUTOUT_SIZE):
                    failures += 1
                    continue
                images.append(cut.data.astype(np.float32))
                ids.append(int(row["object_id"]))
                partitions.append(int(row["partition"]))
                successes += 1
            except (NoOverlapError, ValueError):
                failures += 1

    dt = time.time() - t0
    rate = (successes + failures) / dt
    print(f"  Extracted {successes} cutouts, {failures} failures in {dt:.0f}s "
          f"({rate:.0f} galaxies/sec)")
    print(f"  Failure rate: {failures / (successes + failures) * 100:.1f}%")

    # Save shard
    if images:
        out = SHARD_DIR / f"shard_test_tile{galaxies['tile_index'].iloc[0]}.npz"
        np.savez_compressed(out,
                            images=np.stack(images),
                            object_ids=np.array(ids, dtype=np.int64),
                            partitions=np.array(partitions, dtype=np.int8))
        print(f"  Saved {out.name} ({out.stat().st_size / 1e6:.1f} MB)")

    return successes, failures


def main():
    catalog = pd.read_csv(META_DIR / "partitioned_catalog_with_tiles.csv")
    mosaics = pd.read_csv(META_DIR / "vis_mosaics.csv")

    # Apply threshold 500
    counts = catalog["tile_index"].value_counts()
    kept_tiles = counts[counts >= 500].index
    catalog = catalog[catalog["tile_index"].isin(kept_tiles)]
    print(f"After threshold 500: {len(catalog)} galaxies, "
          f"{catalog['tile_index'].nunique()} tiles")

    test_tile = pick_test_tile(catalog)
    tile_galaxies = catalog[catalog["tile_index"] == test_tile]
    tile_mosaic = mosaics[mosaics["tile_index"] == test_tile].iloc[0]

    fits_path = download_mosaic(
        test_tile,
        tile_mosaic["file_name"],
        tile_mosaic["file_path"],
    )

    successes, failures = extract_cutouts(fits_path, tile_galaxies)

    if not KEEP_MOSAIC:
        fits_path.unlink()
        print(f"\nDeleted mosaic file (set KEEP_MOSAIC=True to retain)")

    print(f"\n=== Test summary ===")
    print(f"Tile: {test_tile}")
    print(f"Galaxies attempted: {len(tile_galaxies)}")
    print(f"Cutouts succeeded:  {successes}")
    print(f"Cutouts failed:     {failures}")


if __name__ == "__main__":
    main()