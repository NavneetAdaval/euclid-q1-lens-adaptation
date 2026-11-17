"""
build_lens_cutouts.py -- build the real positive-class lens_cutouts.npz.

Cuts every AVAILABLE lens from its archive at its CATALOGUE RA/Dec via the stamp's
WCS -- the SAME centring rule the negatives used (02_download_cutouts.py:
Cutout2D at catalogue coord, mode='strict', raw float32). This makes positives and
negatives share one centring definition (verified: archive WCS maps catalogue
RA/Dec to stamp centre, d_centre~0.5px; lens light sits a few px off, which is
deployment-real and kept). Raw flux saved; normalization happens at load.

Output schema (matches LensDataset's lens_cutouts.npz, source_zip widened):
  images(N,64,64)f32, id_str, grade<U1, expert_score f32, source_zip,
  ra f64, dec f64, object_id i8.   NO label array (loader assigns positive by file).
Row order is recorded in a sidecar CSV so real_lenses_kfold can be regenerated
against it (splits are a SEPARATE later step -- this script does not touch them).

Run from repo root:  python scripts\\build_lens_cutouts.py
"""
from __future__ import annotations
import io, sys, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D, NoOverlapError
import astropy.units as u

ARCH_DIR = Path("data/q1_lenses")
MANIFEST = ARCH_DIR / "lens_manifest_positive.csv"
OUT_DIR = Path("data/q1_lenses_processed")
OUT_NPZ = OUT_DIR / "lens_cutouts.npz"
ORDER_CSV = OUT_DIR / "lens_cutouts_order.csv"     # row order -> for building splits
SKIP_CSV = OUT_DIR / "lens_cutouts_skipped.csv"
CUTOUT = 64
VIS_EXT = "VIS_FLUX"


def read_vis_wcs(zf, idn):
    cand = [n for n in zf.namelist() if Path(n).stem == idn and n.lower().endswith(".fits")]
    if not cand:
        return None, None, "no_fits_in_zip"
    try:
        with fits.open(io.BytesIO(zf.read(cand[0]))) as h:
            data = np.asarray(h[VIS_EXT].data, dtype=np.float32)
            w = WCS(h[VIS_EXT].header)
            if not w.has_celestial:
                return None, None, "no_celestial_wcs"
            return data, w, None
    except Exception as e:
        return None, None, f"{type(e).__name__}"


def main():
    if not MANIFEST.exists():
        sys.exit(f"missing manifest {MANIFEST}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MANIFEST, dtype={"id_str": str})
    avail = man[man["available"]].reset_index(drop=True)
    print(f"{len(avail)} available lenses in manifest")

    images, meta, skipped = [], [], []
    nonfinite_hits = 0
    for src, grp in avail.groupby("source"):
        zp = ARCH_DIR / src
        if not zp.exists():
            for idn in grp["id_str"]:
                skipped.append((idn, src, "zip_missing"))
            continue
        srctag = src[:-4] if src.endswith(".zip") else src   # 'lens.zip' -> 'lens'
        with zipfile.ZipFile(zp) as zf:
            for _, r in grp.iterrows():
                idn = r["id_str"]
                data, w, err = read_vis_wcs(zf, idn)
                if err is not None:
                    skipped.append((idn, src, err)); continue
                coord = SkyCoord(float(r["right_ascension"])*u.deg,
                                 float(r["declination"])*u.deg, frame="icrs")
                try:
                    cut = Cutout2D(data, coord, (CUTOUT, CUTOUT), wcs=w, mode="strict")
                except (NoOverlapError, ValueError) as e:
                    skipped.append((idn, src, f"cutout_{type(e).__name__}")); continue
                arr = cut.data.astype(np.float32)
                if arr.shape != (CUTOUT, CUTOUT):
                    skipped.append((idn, src, f"bad_shape_{arr.shape}")); continue
                if not np.isfinite(arr).all():        # negatives had none; instrument it
                    nonfinite_hits += 1
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                images.append(arr)
                meta.append((idn, str(r["grade"]), float(r["expert_score"]),
                             srctag, float(r["right_ascension"]),
                             float(r["declination"]), int(r["object_id"])))

    if not images:
        sys.exit("no lenses cut -- check archives/manifest")

    images = np.stack(images).astype(np.float32)
    id_str = np.array([m[0] for m in meta])
    grade = np.array([m[1] for m in meta])
    expert = np.array([m[2] for m in meta], dtype=np.float32)
    source_zip = np.array([m[3] for m in meta])           # widened naturally by numpy
    ra = np.array([m[4] for m in meta], dtype=np.float64)
    dec = np.array([m[5] for m in meta], dtype=np.float64)
    oid = np.array([m[6] for m in meta], dtype=np.int64)

    # back up any existing file (old splits' indices won't match the new order!)
    if OUT_NPZ.exists():
        bak = OUT_NPZ.with_name("lens_cutouts_OLD.npz")
        shutil.copy2(OUT_NPZ, bak)
        print(f"backed up existing -> {bak}")

    np.savez(OUT_NPZ, images=images, id_str=id_str, grade=grade,
             expert_score=expert, source_zip=source_zip, ra=ra, dec=dec, object_id=oid)

    pd.DataFrame({"row": np.arange(len(id_str)), "id_str": id_str, "grade": grade,
                  "source_zip": source_zip, "object_id": oid}).to_csv(ORDER_CSV, index=False)
    pd.DataFrame(skipped, columns=["id_str", "source", "reason"]).to_csv(SKIP_CSV, index=False)

    print(f"\nwrote {OUT_NPZ}: images {images.shape} {images.dtype}")
    print(f"  grade A={int((grade=='A').sum())}  B={int((grade=='B').sum())}")
    print(f"  source_zip counts: {dict(pd.Series(source_zip).value_counts())}")
    print(f"  raw flux: min={images.min():.4g} med={np.median(images):.4g} "
          f"max={images.max():.4g} frac<0={(images<0).mean():.3f}")
    print(f"  non-finite stamps cleaned: {nonfinite_hits} "
          f"({'identical handling to negatives' if nonfinite_hits == 0 else 'positives had NaNs negatives did not -- note this'})")
    print(f"  skipped {len(skipped)} (see {SKIP_CSV.name})")
    if skipped:
        from collections import Counter
        print("  skip reasons:", dict(Counter(s[2] for s in skipped)))
    print(f"\nrow order -> {ORDER_CSV.name}  (build real_lenses_kfold against this; splits are the next step)")


if __name__ == "__main__":
    main()
