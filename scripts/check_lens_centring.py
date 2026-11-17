"""
check_lens_centring.py -- are the lens archive stamps centred on the lens light?

For a sample of lenses per source archive, computes the flux-weighted centroid in
the central region of the 300x300 VIS_FLUX stamp and reports its offset (in pixels)
from the geometric centre. This decides the positive cut:
  - if offsets are small (well within a 64px half-width of 32px) -> the archive is
    pre-centred on the light, a plain centre-crop matches how negatives were centred.
  - if offsets are large -> need to recentre on the centroid (and/or skip).
The 'recenter' source is expected to show large offsets (that's what the flag means).

Run from repo root:  python scripts\\check_lens_centring.py
"""
from __future__ import annotations
import io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

ARCH_DIR = Path("data/q1_lenses")
MANIFEST = ARCH_DIR / "lens_manifest_positive.csv"
VIS_EXT, RMS_EXT = "VIS_FLUX", "VIS_RMS"
RMS_SENTINEL = 1e15
N_PER_SOURCE = 40
CENTRAL = 128          # restrict centroid to central CENTRALxCENTRAL (avoid far neighbours)
SEED = 0


def read_vis(zf, idn):
    cand = [n for n in zf.namelist() if Path(n).stem == idn and n.lower().endswith(".fits")]
    if not cand:
        return None
    try:
        with fits.open(io.BytesIO(zf.read(cand[0]))) as h:
            vis = np.asarray(h[VIS_EXT].data, dtype=np.float32)
            names = [x.name for x in h]
            if RMS_EXT in names:
                rms = np.asarray(h[RMS_EXT].data, dtype=np.float32)
                vis = np.where(rms >= RMS_SENTINEL, 0.0, vis)
            return np.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def centroid_offset(v):
    h, w = v.shape
    cy, cx = h // 2, w // 2
    half = CENTRAL // 2
    sub = v[cy-half:cy+half, cx-half:cx+half]
    wts = np.clip(sub, 0, None)
    if wts.sum() <= 0:
        return np.inf
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    my = (yy * wts).sum() / wts.sum()
    mx = (xx * wts).sum() / wts.sum()
    # offset from the centre of the sub-window (which is the stamp centre)
    return float(np.hypot(my - (sub.shape[0]-1)/2, mx - (sub.shape[1]-1)/2))


def main():
    man = pd.read_csv(MANIFEST, dtype={"id_str": str})
    avail = man[man["available"]]
    rng = np.random.default_rng(SEED)
    print(f"{'source':>12} {'n':>4} {'off_p50':>8} {'off_p90':>8} {'off_max':>8} "
          f"{'frac>10':>8} {'frac>32':>8}   (pixels; 64px crop half-width=32)")
    for src, grp in avail.groupby("source"):
        zp = ARCH_DIR / src
        if not zp.exists():
            print(f"{src:>12}  (zip not found)")
            continue
        ids = grp["id_str"].tolist()
        if len(ids) > N_PER_SOURCE:
            ids = rng.choice(ids, N_PER_SOURCE, replace=False).tolist()
        offs, bad = [], 0
        with zipfile.ZipFile(zp) as zf:
            for idn in ids:
                v = read_vis(zf, idn)
                if v is None or isinstance(v, str):
                    bad += 1; continue
                offs.append(centroid_offset(v))
        offs = np.array([o for o in offs if np.isfinite(o)])
        if len(offs) == 0:
            print(f"{src:>12} {0:>4}  (no readable stamps; {bad} bad)")
            continue
        print(f"{src:>12} {len(offs):>4} "
              f"{np.percentile(offs,50):>8.1f} {np.percentile(offs,90):>8.1f} "
              f"{offs.max():>8.1f} {(offs>10).mean():>8.2f} {(offs>32).mean():>8.2f}"
              f"{'   ('+str(bad)+' bad)' if bad else ''}")
    print("\nRead: small offsets (p90 well under 32) -> plain centre-crop matches the")
    print("negatives' centring. Large offsets (esp. recenter) -> recentre on centroid.")


if __name__ == "__main__":
    main()
