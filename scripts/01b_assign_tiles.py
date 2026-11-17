"""
Assign each galaxy in partitioned_catalog.csv to a Q1 VIS mosaic tile.

Strategy: query the q1.mosaic_product table for all VIS mosaic centers,
then for each galaxy find the nearest mosaic center within the tile size.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.esa.euclid import Euclid

DATA_DIR = Path(__file__).parent.parent / "data"
META_DIR = DATA_DIR / "q1_unlabeled" / "metadata"


def fetch_vis_mosaics() -> pd.DataFrame:
    """All Q1 VIS mosaic products."""
    query = """
    SELECT tile_index, ra, dec, file_path, file_name
    FROM q1.mosaic_product
    WHERE instrument_name = 'VIS'
      AND product_type = 'DpdMerBksMosaic'
      AND category = 'SCIENCE'
    """
    print("Fetching VIS mosaic list...")
    job = Euclid.launch_job_async(query)
    df = job.get_results().to_pandas()
    print(f"  Got {len(df)} VIS mosaics")
    return df


def assign_tiles(catalog: pd.DataFrame, mosaics: pd.DataFrame) -> pd.DataFrame:
    """
    For each galaxy, find the nearest mosaic center.
    Q1 tile size is ~0.54° square; a galaxy is in a tile if it's within ~0.27°
    of the center (we use 0.30° as a conservative margin for overlaps).
    """
    print("Matching galaxies to mosaics...")
    gal_coords = SkyCoord(
        ra=catalog["right_ascension"].values * u.deg,
        dec=catalog["declination"].values * u.deg,
    )
    mosaic_coords = SkyCoord(
        ra=mosaics["ra"].values * u.deg,
        dec=mosaics["dec"].values * u.deg,
    )

    idx, sep, _ = gal_coords.match_to_catalog_sky(mosaic_coords)
    catalog = catalog.copy()
    catalog["tile_index"] = mosaics["tile_index"].values[idx]
    catalog["tile_separation_deg"] = sep.deg

    # Keep only matches within tile radius
    n_before = len(catalog)
    catalog = catalog[catalog["tile_separation_deg"] < 0.30].copy()
    print(f"  {len(catalog)} / {n_before} galaxies matched to a tile within 0.30°")
    print(f"  Galaxies span {catalog['tile_index'].nunique()} unique tiles")

    return catalog


if __name__ == "__main__":
    catalog = pd.read_csv(META_DIR / "partitioned_catalog.csv")
    print(f"Loaded {len(catalog)} galaxies")

    mosaics = fetch_vis_mosaics()
    mosaics.to_csv(META_DIR / "vis_mosaics.csv", index=False)

    catalog_with_tiles = assign_tiles(catalog, mosaics)
    catalog_with_tiles.to_csv(META_DIR / "partitioned_catalog_with_tiles.csv", index=False)

    print("\nPer-tile galaxy counts (top 20):")
    print(catalog_with_tiles["tile_index"].value_counts().head(20))
    print(f"\nMin/median/max galaxies per tile: "
          f"{catalog_with_tiles['tile_index'].value_counts().min()} / "
          f"{int(catalog_with_tiles['tile_index'].value_counts().median())} / "
          f"{catalog_with_tiles['tile_index'].value_counts().max()}")