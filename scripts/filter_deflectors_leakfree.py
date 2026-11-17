"""
filter_deflectors_leakfree.py -- drop any deflector that is a known lens.

Deflectors are painted on to make sim lenses; a deflector that is itself a real
(test) lens would leak the test set into training. This drops any deflector
object_id present in the lens manifest, writing the final clean cutout file.

Run from repo root (after downloading the Datalabs deflector_cutouts.npz to
data/sim_deflectors/):  python scripts\\filter_deflectors_leakfree.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/sim_deflectors/deflector_cutouts_raw.npz")   # the Datalabs output
LENS_MANIFEST = Path("data/q1_lenses/lens_manifest_positive.csv")
OUT = Path("data/sim_deflectors/deflector_cutouts.npz")

d = np.load(RAW, allow_pickle=True)
oid = d["object_id"].astype("int64")
lens_ids = set(pd.read_csv(LENS_MANIFEST)["object_id"].dropna().astype("int64"))
is_lens = np.isin(oid, list(lens_ids))
keep = ~is_lens

print(f"deflectors in       : {len(oid):,}")
print(f"dropped (known lens) : {int(is_lens.sum())}")
print(f"deflectors out       : {int(keep.sum()):,}")

np.savez_compressed(
    OUT,
    images=d["images"][keep],
    object_id=oid[keep],
    ra=d["ra"][keep], dec=d["dec"][keep],
    phz_median=d["phz_median"][keep],
    stellarmass=d["stellarmass"][keep],
    mvis=d["mvis"][keep],
    tile_index=d["tile_index"][keep],
)
print(f"wrote {OUT}")
