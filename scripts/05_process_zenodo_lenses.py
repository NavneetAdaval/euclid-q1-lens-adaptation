"""
Process the Zenodo Q1 lens cutouts (lens.zip, group.zip, recenter.zip)
into 64x64 .npz shards matching the format of Dataset 4.

For lens.zip and group.zip: take a center crop (lens is at image center).
For recenter.zip: find the brightest peak and crop around it.

Output:
  data/q1_lenses_processed/lens_cutouts.npz   — all real lens cutouts
    images:         (N, 64, 64) float32
    id_str:         (N,) str — Zenodo lens identifier
    grade:          (N,) str — 'A', 'B', 'C', or '' if not in catalog
    expert_score:   (N,) float32
    source_zip:     (N,) str — 'lens', 'group', or 'recenter'
    ra:             (N,) float64
    dec:            (N,) float64
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent
ZENODO_DIR = PROJECT_DIR / "data" / "q1_lenses"
OUT_DIR = PROJECT_DIR / "data" / "q1_lenses_processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = ZENODO_DIR / "q1_discovery_engine_lens_catalog.csv"

CUTOUT_SIZE = 64
ORIG_SIZE = 300
CENTER_PIXEL = ORIG_SIZE // 2  # 150

# Smoothing scale for peak-finding in recenter.zip
RECENTER_SMOOTH_SIGMA = 1.5

# Path patterns inside each zip
ZIP_TO_SOURCE = {
    "lens.zip": "lens",
    "group.zip": "group",
    "recenter.zip": "recenter",
}


def find_lens_fits_files(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return list of (id_str, internal_zip_path) for each lens FITS.

    Each lens lives in a folder named {id_str}/, and the science FITS
    inside is {id_str}.fits. Other FITS files in the folder (lens_light,
    source_light, etc.) are skipped.
    """
    results = []
    for name in z.namelist():
        # Match patterns like: lens/102018665_NEG.../102018665_NEG....fits
        parts = name.split("/")
        if len(parts) != 3:
            continue
        if not parts[2].endswith(".fits"):
            continue
        # The science cutout has the same name as its folder
        folder = parts[1]
        filename = parts[2]
        if filename == f"{folder}.fits":
            results.append((folder, name))
    return results


def read_vis_flux(zip_bytes: bytes) -> np.ndarray:
    """Read the VIS_FLUX HDU from a FITS in memory. Returns (300,300) float32."""
    with fits.open(io.BytesIO(zip_bytes)) as hdul:
        # Find VIS_FLUX by name (more robust than fixed index)
        for hdu in hdul:
            if hdu.name == "VIS_FLUX":
                return np.array(hdu.data, dtype=np.float32, copy=True)
        raise ValueError("VIS_FLUX HDU not found")


def center_crop(image: np.ndarray, size: int = CUTOUT_SIZE) -> np.ndarray:
    """Crop the central size×size region of an image."""
    h, w = image.shape
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    return image[start_y:start_y + size, start_x:start_x + size]


def peak_centered_crop(image: np.ndarray, size: int = CUTOUT_SIZE,
                       smooth_sigma: float = RECENTER_SMOOTH_SIGMA) -> np.ndarray:
    """Find the brightest peak (in a smoothed image) and crop size×size around it.

    Restricts search to the central region of the image to avoid edge artifacts
    and neighbor sources that aren't the intended lens.
    """
    h, w = image.shape
    # Smooth to suppress noise spikes — we want the brightest *extended* source
    smoothed = gaussian_filter(image, sigma=smooth_sigma)

    # Restrict peak search to inner 70% of the image to avoid edge artifacts
    margin = int(h * 0.15)
    inner = smoothed[margin:h - margin, margin:w - margin]
    peak_y_inner, peak_x_inner = np.unravel_index(np.argmax(inner), inner.shape)
    peak_y = peak_y_inner + margin
    peak_x = peak_x_inner + margin

    # Crop around peak, clamping to image bounds
    half = size // 2
    start_y = max(0, min(h - size, peak_y - half))
    start_x = max(0, min(w - size, peak_x - half))
    return image[start_y:start_y + size, start_x:start_x + size]


def process_zip(zip_path: Path, catalog: pd.DataFrame,
                use_recenter: bool) -> list[dict]:
    """Process one zip file, returning a list of per-lens records."""
    print(f"\nProcessing {zip_path.name}...")
    results = []

    with zipfile.ZipFile(zip_path, "r") as z:
        targets = find_lens_fits_files(z)
        print(f"  Found {len(targets)} lens cutouts in {zip_path.name}")

        for id_str, internal_path in tqdm(targets, desc=f"  {zip_path.name}"):
            try:
                data = z.read(internal_path)
                vis_flux = read_vis_flux(data)

                if vis_flux.shape != (ORIG_SIZE, ORIG_SIZE):
                    print(f"    {id_str}: unexpected shape {vis_flux.shape}, skipping")
                    continue

                if use_recenter:
                    cutout = peak_centered_crop(vis_flux)
                else:
                    cutout = center_crop(vis_flux)

                # Look up catalog metadata by id_str
                cat_row = catalog[catalog["id_str"] == id_str]
                if len(cat_row) > 0:
                    cat_row = cat_row.iloc[0]
                    grade = str(cat_row.get("grade", "")) or ""
                    expert_score = float(cat_row.get("expert_score", float("nan")))
                    ra = float(cat_row.get("right_ascension", float("nan")))
                    dec = float(cat_row.get("declination", float("nan")))
                else:
                    grade = ""
                    expert_score = float("nan")
                    ra = float("nan")
                    dec = float("nan")

                results.append({
                    "image": cutout,
                    "id_str": id_str,
                    "grade": grade,
                    "expert_score": expert_score,
                    "source_zip": ZIP_TO_SOURCE[zip_path.name],
                    "ra": ra,
                    "dec": dec,
                })
            except Exception as e:
                print(f"    {id_str}: error — {e}")
                continue

    print(f"  Successfully processed {len(results)} from {zip_path.name}")
    return results


def main():
    # Load catalog for metadata join
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH}")
    catalog = pd.read_csv(CATALOG_PATH)
    print(f"Loaded catalog with {len(catalog)} entries")

    all_records = []

    # lens.zip — center crop
    lens_zip = ZENODO_DIR / "lens.zip"
    if lens_zip.exists():
        all_records.extend(process_zip(lens_zip, catalog, use_recenter=False))
    else:
        print(f"WARNING: {lens_zip} not found, skipping")

    # group.zip — center crop
    group_zip = ZENODO_DIR / "group.zip"
    if group_zip.exists():
        all_records.extend(process_zip(group_zip, catalog, use_recenter=False))
    else:
        print(f"WARNING: {group_zip} not found, skipping")

    # recenter.zip — peak-centered crop
    recenter_zip = ZENODO_DIR / "recenter.zip"
    if recenter_zip.exists():
        all_records.extend(process_zip(recenter_zip, catalog, use_recenter=True))
    else:
        print(f"WARNING: {recenter_zip} not found, skipping")

    if not all_records:
        raise RuntimeError("No lens cutouts processed!")

    # Stack into arrays
    print(f"\nStacking {len(all_records)} cutouts into arrays...")
    images = np.stack([r["image"] for r in all_records]).astype(np.float32)
    id_strs = np.array([r["id_str"] for r in all_records])
    grades = np.array([r["grade"] for r in all_records])
    expert_scores = np.array([r["expert_score"] for r in all_records], dtype=np.float32)
    source_zips = np.array([r["source_zip"] for r in all_records])
    ras = np.array([r["ra"] for r in all_records], dtype=np.float64)
    decs = np.array([r["dec"] for r in all_records], dtype=np.float64)

    out_path = OUT_DIR / "lens_cutouts.npz"
    np.savez_compressed(
        out_path,
        images=images,
        id_str=id_strs,
        grade=grades,
        expert_score=expert_scores,
        source_zip=source_zips,
        ra=ras,
        dec=decs,
    )

    print(f"\nSaved {out_path}")
    print(f"  Total: {len(all_records)} lenses")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")

    # Audit
    print("\n=== Audit ===")
    print(f"By source zip:")
    src_unique, src_counts = np.unique(source_zips, return_counts=True)
    for s, c in zip(src_unique, src_counts):
        print(f"  {s}: {c}")
    print(f"By grade:")
    grade_unique, grade_counts = np.unique(grades, return_counts=True)
    for g, c in zip(grade_unique, grade_counts):
        label = g if g else "(no catalog match)"
        print(f"  {label}: {c}")
    print(f"\nPixel range across all cutouts: {images.min():.3f} to {images.max():.3f}")
    print(f"Per-cutout mean range: {images.mean(axis=(1, 2)).min():.3f} to "
          f"{images.mean(axis=(1, 2)).max():.3f}")


if __name__ == "__main__":
    main()