"""
check_lens_wcs.py -- is the archive stamp's WCS valid enough to cut positives at
the catalogue RA/Dec (option 1, matching how negatives were cut)?

For each sampled lens it:
  1. reads the 300x300 VIS_FLUX stamp and its WCS
  2. maps the lens's catalogue (RA,Dec) -> pixel via the WCS
  3. measures (a) how far that predicted pixel is from the stamp centre, and
     (b) how far it is from the actual flux centroid (the light).
The decisive test is (b): if the WCS sends the catalogue coordinate to where the
light actually is -- even for off-centre stamps -- the WCS tracks the lens and
option 1 works. If the predicted pixel lands nowhere near the light, the WCS is
unreliable and we fall back to guarded centroid recentring.

Run from repo root:  python scripts\\check_lens_wcs.py
"""
from __future__ import annotations
import io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

ARCH_DIR = Path("data/q1_lenses")
MANIFEST = ARCH_DIR / "lens_manifest_positive.csv"
VIS_EXT, RMS_EXT = "VIS_FLUX", "VIS_RMS"
RMS_SENTINEL = 1e15
N_PER_SOURCE = 40
CENTRAL = 128
SEED = 0


def read_hdu(zf, idn):
    cand = [n for n in zf.namelist() if Path(n).stem == idn and n.lower().endswith(".fits")]
    if not cand:
        return None, None
    try:
        with fits.open(io.BytesIO(zf.read(cand[0]))) as h:
            vis = np.asarray(h[VIS_EXT].data, dtype=np.float32)
            names = [x.name for x in h]
            if RMS_EXT in names:
                rms = np.asarray(h[RMS_EXT].data, dtype=np.float32)
                vis = np.where(rms >= RMS_SENTINEL, 0.0, vis)
            vis = np.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
            # WCS from the VIS_FLUX header
            try:
                w = WCS(h[VIS_EXT].header)
                if not w.has_celestial:
                    w = None
            except Exception:
                w = None
            return vis, w
    except Exception:
        return None, None


def flux_centroid(v):
    h, w = v.shape
    cy, cx = h // 2, w // 2
    half = CENTRAL // 2
    sub = v[cy-half:cy+half, cx-half:cx+half]
    wts = np.clip(sub, 0, None)
    if wts.sum() <= 0:
        return None
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    my = (yy*wts).sum()/wts.sum(); mx = (xx*wts).sum()/wts.sum()
    return (cy-half+my, cx-half+mx)   # (y,x) in full-stamp pixels


def main():
    man = pd.read_csv(MANIFEST, dtype={"id_str": str})
    avail = man[man["available"]]
    rng = np.random.default_rng(SEED)
    print(f"{'source':>12} {'n':>4} {'noWCS':>6} "
          f"{'d_centre_p50':>12} {'d_centre_p90':>12} "
          f"{'d_light_p50':>11} {'d_light_p90':>11} {'frac_light<5':>12}")
    grand = []
    for src, grp in avail.groupby("source"):
        zp = ARCH_DIR / src
        if not zp.exists():
            continue
        ids = grp["id_str"].tolist()
        sub = grp.set_index("id_str")
        if len(ids) > N_PER_SOURCE:
            ids = rng.choice(ids, N_PER_SOURCE, replace=False).tolist()
        d_centre, d_light, nowcs, bad = [], [], 0, 0
        with zipfile.ZipFile(zp) as zf:
            for idn in ids:
                v, w = read_hdu(zf, idn)
                if v is None:
                    bad += 1; continue
                if w is None:
                    nowcs += 1; continue
                ra = float(sub.loc[idn, "right_ascension"])
                dec = float(sub.loc[idn, "declination"])
                try:
                    sk = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
                    px, py = w.world_to_pixel(sk)        # 0-based x,y
                except Exception:
                    nowcs += 1; continue
                cy = cx = (v.shape[0]-1)/2.0
                d_centre.append(float(np.hypot(py-cy, px-cx)))
                cen = flux_centroid(v)
                if cen is not None:
                    d_light.append(float(np.hypot(py-cen[0], px-cen[1])))
        dc = np.array(d_centre); dl = np.array(d_light)
        grand += list(dl)
        if len(dc) == 0:
            print(f"{src:>12} {0:>4} {nowcs:>6}  (no usable WCS)")
            continue
        fl5 = float((dl < 5).mean()) if len(dl) else float("nan")
        print(f"{src:>12} {len(dc):>4} {nowcs:>6} "
              f"{np.percentile(dc,50):>12.1f} {np.percentile(dc,90):>12.1f} "
              f"{(np.percentile(dl,50) if len(dl) else np.nan):>11.1f} "
              f"{(np.percentile(dl,90) if len(dl) else np.nan):>11.1f} {fl5:>12.2f}")
    if grand:
        g = np.array(grand)
        print(f"\nALL sources: WCS->catalogue pixel vs actual light  "
              f"p50={np.percentile(g,50):.1f} p90={np.percentile(g,90):.1f} "
              f"frac<5px={np.mean(g<5):.2f}")
    print("\nDecisive column is d_light (WCS-predicted pixel vs the lens light):")
    print(" small (p90 < ~5px, frac<5 high) -> WCS tracks the lens; OPTION 1 works,")
    print("   cut positives at catalogue RA/Dec exactly like the negatives.")
    print(" large -> WCS unreliable; fall back to guarded centroid recentring.")


if __name__ == "__main__":
    main()
