import pandas as pd
from pathlib import Path

META_DIR = Path("data/q1_unlabeled/metadata")
df = pd.read_csv(META_DIR / "partitioned_catalog_with_tiles.csv")

counts = df["tile_index"].value_counts()

# How many galaxies do we keep if we drop tiles below threshold T?
print(f"{'Min/tile':>10} | {'Tiles kept':>10} | {'Galaxies kept':>14} | {'% galaxies':>10}")
print("-" * 55)
for thresh in [1, 10, 50, 100, 200, 300, 500, 700, 1000]:
    kept_tiles = (counts >= thresh).sum()
    kept_galaxies = counts[counts >= thresh].sum()
    pct = kept_galaxies / len(df) * 100
    print(f"{thresh:>10} | {kept_tiles:>10} | {kept_galaxies:>14} | {pct:>9.1f}%")


from astroquery.esa.euclid import Euclid

q = """
SELECT TOP 5 file_name, data_size, file_path
FROM q1.mosaic_product
WHERE instrument_name = 'VIS' AND product_type = 'DpdMerBksMosaic'
"""
job = Euclid.launch_job_async(q)
print(job.get_results().to_pandas())