"""
Query IRSA's Euclid MER catalog for candidate galaxies for the unlabeled SSL pool.
Saves a CSV with object_id, RA, Dec, magnitude, tile_index, and a deterministic
partition assignment per object.

Run once. Re-runs are idempotent — same partition assignments every time.
"""
import hashlib
from pathlib import Path

import pandas as pd
from astroquery.ipac.irsa import Irsa
from astroquery.esa.euclid import Euclid

import time
import pyvo

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "q1_unlabeled" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LENS_CATALOG_PATH = DATA_DIR / "q1_lenses" / "q1_discovery_engine_lens_catalog.csv"


def list_euclid_catalogs():
    """Run this first — confirms current catalog names in IRSA."""
    cats = Irsa.list_catalogs(filter="euclid")
    for name, desc in cats.items():
        print(f"  {name}: {desc}")
    return cats


def query_mer_catalog(top_n: int = 250_000):
    """
    ESA Euclid TAP query for unlabeled Q1 galaxy pool.

    Spatial filter: restricts to the four Q1 sky fields
      - Euclid Deep Field North  (RA ~269°, Dec ~+66°)
      - Euclid Deep Field Fornax (RA ~53°,  Dec ~-28°)
      - Euclid Deep Field South  (RA ~61°,  Dec ~-48°)
      - LDN1641                   (RA ~84°,  Dec ~-8°)

    Quality filter: detected in VIS + all NIR, galaxy-like, not spurious.
    """
    query = f"""
    SELECT TOP {top_n}
        object_id,
        right_ascension,
        declination,
        flux_vis_psf,
        point_like_prob,
        extended_prob,
        spurious_prob
    FROM catalogue.mer_catalogue
    WHERE vis_det = 1
      AND flux_vis_psf BETWEEN 2.3 AND 229
      AND flux_y_templfit > 0
      AND flux_j_templfit > 0
      AND flux_h_templfit > 0
      AND point_like_prob < 0.3
      AND extended_prob > 0.5
      AND spurious_prob < 0.3
      AND (
            (right_ascension BETWEEN 250 AND 290 AND declination BETWEEN  60 AND  70)
         OR (right_ascension BETWEEN  48 AND  60 AND declination BETWEEN -32 AND -24)
         OR (right_ascension BETWEEN  55 AND  70 AND declination BETWEEN -50 AND -45)
         OR (right_ascension BETWEEN  80 AND  88 AND declination BETWEEN -11 AND  -5)
      )
    """
    print(f"Submitting async ESA TAP query for up to {top_n} galaxies...")
    job = Euclid.launch_job_async(query)
    print(f"  Polling handled internally by astroquery...")
    results = job.get_results()
    df = results.to_pandas()
    print(f"  Got {len(df)} rows")
    return df


def partition_for(object_id, weights=(8, 3, 3, 1, 2)):
    """
    Weighted deterministic partitioning.
    Indices:  0=SSL pretrain, 1=source_neg, 2=uda_target, 3=sda_neg, 4=test_neg
    Share:    47%,             18%,         18%,          6%,         12%
    """
    total = sum(weights)
    h = int(hashlib.md5(str(object_id).encode()).hexdigest(), 16) % total
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if h < cumulative:
            return i
    return len(weights) - 1


def assign_partitions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["partition"] = df["object_id"].apply(partition_for)
    return df


def exclude_known_lenses(df: pd.DataFrame) -> pd.DataFrame:
    if not LENS_CATALOG_PATH.exists():
        print(f"WARNING: lens catalog not found at {LENS_CATALOG_PATH} — skipping exclusion")
        return df
    lens_cat = pd.read_csv(LENS_CATALOG_PATH)
    known = set(lens_cat["object_id"].dropna().astype("int64"))
    before = len(df)
    df = df[~df["object_id"].isin(known)]
    print(f"Excluded {before - len(df)} known lenses; {len(df)} remain")
    return df


if __name__ == "__main__":
    # Uncomment on first run to confirm catalog names
    # list_euclid_catalogs()

    df = query_mer_catalog(top_n=250_000)
    df = exclude_known_lenses(df)
    df = assign_partitions(df)

    print("\nPartition distribution:")
    print(df["partition"].value_counts().sort_index())

    out_path = OUT_DIR / "partitioned_catalog.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
