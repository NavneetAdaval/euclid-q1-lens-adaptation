"""
inspect_lens_fits.py  --  look inside the Euclid Q1 lens archive products.

Before cutting any stamps we need to know, from your ACTUAL files:
  1. For each archive (lens / unsuccess / group / recenter), what files are in a
     lens folder, and is there a usable science FITS image at all? The failed-
     modelling archives may only contain PNGs.
  2. Inside the science FITS: how many HDUs, which one is VIS, the array size,
     pixel scale, dtype, and value range -- this fixes how the 64x64 stamp is made.

Reads zip members in memory (no extraction). Run from repo root:
    python inspect_lens_fits.py
"""

from pathlib import Path
import zipfile
import io
import sys

import pandas as pd

try:
    from astropy.io import fits
except ImportError:
    sys.exit("Need astropy:  pip install astropy")

# ---------------- config ----------------
ROOT = Path("data/q1_lenses")
MANIFEST = ROOT / "lens_manifest_positive.csv"
ARCHIVES = ["lens.zip", "unsuccess.zip", "group.zip", "recenter.zip"]
SAMPLES_PER_ARCHIVE = 3
# ----------------------------------------


def members_for_id(zf: zipfile.ZipFile, id_str: str) -> list:
    return [n for n in zf.namelist() if id_str in n and not n.endswith("/")]


def describe_fits(raw: bytes, label: str):
    try:
        with fits.open(io.BytesIO(raw)) as hdul:
            print(f"      {label}: {len(hdul)} HDU(s)")
            for i, hdu in enumerate(hdul):
                data = hdu.data
                name = hdu.header.get("EXTNAME", hdu.name or "")
                if data is None:
                    print(f"        [{i}] {hdu.__class__.__name__:12s} name={name!r:8} (no data)")
                    continue
                hdr = hdu.header
                # pixel scale: prefer CD/CDELT in deg -> arcsec
                scale = None
                for k in ("CDELT2", "CD2_2"):
                    if k in hdr:
                        scale = abs(float(hdr[k])) * 3600.0
                        break
                bunit = hdr.get("BUNIT", "")
                try:
                    import numpy as np
                    lo, mid, hi = (float(np.nanmin(data)), float(np.nanmedian(data)),
                                   float(np.nanmax(data)))
                    rng = f"min={lo:.3g} med={mid:.3g} max={hi:.3g}"
                except Exception:
                    rng = "stats n/a"
                sc = f"{scale:.4f}\"/pix" if scale else "scale?"
                print(f"        [{i}] {hdu.__class__.__name__:12s} name={name!r:8} "
                      f"shape={data.shape} dtype={data.dtype} {sc} BUNIT={bunit!r} {rng}")
    except Exception as e:
        print(f"      {label}: NOT READABLE as FITS ({e})")


def main():
    if not MANIFEST.exists():
        sys.exit(f"Run audit_lenses.py first; missing {MANIFEST}")
    man = pd.read_csv(MANIFEST, dtype={"id_str": str})
    avail = man[man["available"]]

    for arch in ARCHIVES:
        zp = ROOT / arch
        print("\n" + "=" * 70)
        print(f"ARCHIVE: {arch}")
        print("=" * 70)
        if not zp.exists():
            print("  (not present)")
            continue
        ids = avail.loc[avail["source"] == arch, "id_str"].head(SAMPLES_PER_ARCHIVE).tolist()
        if not ids:
            print("  (no positive-class lenses sourced from here)")
            continue
        with zipfile.ZipFile(zp) as zf:
            for idn in ids:
                print(f"\n  id_str = {idn}")
                mem = members_for_id(zf, idn)
                if not mem:
                    print("    (no files matched -- unexpected)")
                    continue
                exts = {}
                for m in mem:
                    e = Path(m).suffix.lower()
                    exts[e] = exts.get(e, 0) + 1
                print(f"    files: {exts}")
                fits_members = [m for m in mem if m.lower().endswith((".fits", ".fit"))]
                if not fits_members:
                    print("    NO .fits IN THIS FOLDER -> cannot cut a stamp from the archive;")
                    print("    these lenses must be cut from the Q1 tiles instead.")
                    continue
                # the primary science cutout should be <id_str>.fits; inspect it first,
                # then any others (lens_light, source_light, masks) briefly by name
                primary = [m for m in fits_members if Path(m).stem == idn]
                others = [m for m in fits_members if Path(m).stem != idn]
                for m in primary + others[:2]:
                    describe_fits(zf.read(m), Path(m).name)

    print("\nDone. Report this output back and I'll write the extractor to match.")


if __name__ == "__main__":
    main()
