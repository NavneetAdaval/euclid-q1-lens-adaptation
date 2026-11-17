"""
generate_arc_on_real.py  --  arc-on-real Euclid VIS lens generator (Colab).

Paints a lenstronomy-ray-traced, real HST/COSMOS source onto a REAL Euclid
deflector cutout. Only the arc is synthetic; the deflector is a real observation.

Physics chain (all parameters research-sourced, see project notes):
  deflector  : real Euclid VIS cutout (deflector_cutouts.npz), carries photo-z + stellar mass
  sigma_v    : abundance-matched to the Choi+2007 VDF (sigma*=161, alpha=2.32, beta=2.67)
  theta_E    : SIE,  4pi (sigma/c)^2 D_ls/D_s ; source z drawn so theta_E in [0.5, 2.0]"
  source     : GalSim COSMOS F814W (sample 23.5), COSMOSCatalog quality cuts + size/mag floor
  render     : lenstronomy SIE ray-trace of INTERPOL source -> Euclid VIS PSF -> 64px @ 0.1"/px
  paint      : scale arc to a drawn source magnitude, add to real cutout, Poisson noise on ARC ONLY
  non-lens   : the bare real deflector cutout (same galaxy, no arc)  <- matched negative

Output: checkpointed .npz shards on Google Drive. Debug run = 200 pairs first.

ENV: Colab, after  `pip install galsim lenstronomy`  and  `galsim_download_cosmos -s 23.5`.
Upload deflector_cutouts.npz to Colab and set DEFLECTOR_NPZ below.
"""
import os, time, warnings, glob
import numpy as np
import galsim
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util

# ----------------------------- config -----------------------------
REPO          = "/mnt/c/Users/krris/ML/lens_paper"           # WSL view of the Windows repo
DEFLECTOR_NPZ = REPO + "/data/sim_deflectors/deflector_cutouts_clean.npz"   # CLEANED deflectors
OUT_DIR       = REPO + "/data/sim_arc_on_real"               # sharded output dir
N_TARGET      = 30000         # DA source-positive scale (Parul 2410.01203)
CKPT_EVERY    = 500           # lenses per output shard (also the checkpoint granularity)
MAX_REUSE     = 10            # cap on how many accepted lenses reuse the same deflector
SUPERSAMPLE   = 2             # ray-trace supersampling (2 = ~half the cost of 3, still smooth)
PIX           = 64
PIX_SCALE     = 0.1           # Euclid VIS "/px
EUCLID_PSF_FWHM = 0.16        # Euclid VIS PSF FWHM, arcsec
ARC_SNR_RANGE = (5.0, 30.0)   # brightest-arc-pixel S/N vs deflector background RMS (HOLISMOKES-IV)
MU_MIN        = 5.0           # min total magnification to accept as a strong lens (HOLISMOKES VI/XI)
SHEAR_SIGMA   = 0.058         # external-shear strength, Gaussian sigma (Euclid Disc.Engine A includes ext. shear)
TE_FLOOR      = 0.75          # theta_E floor; NO upper cap -> natural Euclid falloff (Disc.Engine B, 2503.15325)
ZMAX          = 5.0           # max source redshift searched
SEED          = 42

cosmo  = FlatLambdaCDM(H0=70, Om0=0.3)
C_KMS  = 299792.458

# Precomputed comoving distance D_C(z) on a fine grid -> fast exact-enough interpolation.
# Angular-diameter distances derive from comoving: D_A = D_C/(1+z); D_A(z1,z2)=(D_C2-D_C1)/(1+z2)
# (flat universe). This replaces ~600 astropy calls per generation attempt with array interp.
_ZGRID = np.linspace(0.0, 6.0, 6001)
_DC = cosmo.comoving_distance(_ZGRID).value          # Mpc
def _Dc(z):
    return np.interp(z, _ZGRID, _DC)

# reusable SIE+SHEAR lens model + solver for the strong-lensing acceptance test
_lensmodel = LensModel(["SIE", "SHEAR"])
_solver = LensEquationSolver(_lensmodel)

# ----------------------- physics helpers --------------------------
def sample_choi_vdf(n, rng, smin=80, smax=400):
    s = np.linspace(smin, smax, 6000)
    a, b, ss = 2.32, 2.67, 161.0
    pdf = (s/ss)**a * np.exp(-(s/ss)**b) / s
    pdf /= pdf.sum()
    return rng.choice(s, size=n, p=pdf)

def theta_E_arcsec(sigma_v, z_l, z_s):
    """SIE Einstein radius. Accepts scalar or array z_s. Uses fast interpolated distances."""
    z_s = np.asarray(z_s, dtype=float)
    Dc_s = _Dc(z_s); Dc_l = _Dc(z_l)
    D_s  = Dc_s / (1.0 + z_s)                 # angular-diameter distance to source (flat)
    D_ls = (Dc_s - Dc_l) / (1.0 + z_s)        # lens->source ang-diam distance (flat)
    te = 4*np.pi*(sigma_v/C_KMS)**2*(D_ls/D_s) * (180/np.pi)*3600
    te = np.where(z_s <= z_l, 0.0, te)
    return te if te.ndim else float(te)

def pair_source_z(sigma_v, z_l, rng, te_lo=0.5, te_hi=2.0, zmax=4.0):
    zs_grid = np.linspace(z_l+0.05, zmax, 200)
    te = np.array([theta_E_arcsec(sigma_v, z_l, zs) for zs in zs_grid])
    ok = (te >= te_lo) & (te <= te_hi)
    if not ok.any():
        return None, None
    zs = float(rng.choice(zs_grid[ok]))
    return zs, theta_E_arcsec(sigma_v, z_l, zs)

def measure_shape(cutout):
    """deflector ellipticity (e1,e2) from adaptive moments; light traces mass for the SIE."""
    try:
        img = galsim.Image(np.ascontiguousarray(cutout, dtype=np.float64), scale=PIX_SCALE)
        res = galsim.hsm.FindAdaptiveMom(img, strict=False)
        if res.moments_status != 0:
            return 0.0, 0.0
        e1, e2 = res.observed_shape.e1, res.observed_shape.e2
        n = np.hypot(e1, e2)
        if n > 0.6:                      # clip implausible ellipticities
            e1, e2 = e1*0.6/n, e2*0.6/n
        return float(e1), float(e2)
    except Exception:
        return 0.0, 0.0

# ----------------------- source handling --------------------------
def make_cosmos_catalog():
    # COSMOSCatalog applies GalSim's built-in postage-stamp/fit quality selection
    return galsim.COSMOSCatalog(sample="23.5")

def draw_source_hires(cat, rng, src_scale=0.03, src_npix=128,
                      min_hlr=0.15, max_hlr=1.5):
    """
    Clean noiseless parametric COSMOS source as a SURFACE-BRIGHTNESS map for lenstronomy
    INTERPOL (per arcsec^2, not flux/pixel). Parametric fit -> no HST sky-noise to lens.
    FFT-size warning promoted to error so pathological compact fits are skipped, not hung.
    """
    gsp = galsim.GSParams(maximum_fft_size=8192, folding_threshold=1e-2)
    for _ in range(200):
        idx = int(rng.integers(cat.nobjects))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", galsim.errors.GalSimFFTSizeWarning)
                gal = cat.makeGalaxy(idx, gal_type="parametric", chromatic=False,
                                     gsparams=gsp)
                hlr = gal.calculateHLR()
                if not np.isfinite(hlr) or hlr < min_hlr or hlr > max_hlr:
                    continue
                obs = galsim.Convolve([gal, galsim.Gaussian(fwhm=0.09, gsparams=gsp)],
                                      gsparams=gsp)
                # INTERPOL expects surface brightness per arcsec^2, NOT flux/pixel:
                # draw with no_pixel (SB sampling), then divide by pixel area.
                img = obs.drawImage(nx=src_npix, ny=src_npix, scale=src_scale,
                                    method="no_pixel").array / (src_scale**2)
            if not np.isfinite(img).all() or img.sum() <= 0:
                continue
            return img.astype(np.float64), src_scale
        except Exception:
            continue
    return None, None

# ----------------------- lensing render ---------------------------
def render_arc(source_img, source_scale, kwargs_lens, src_dx, src_dy):
    """Ray-trace INTERPOL source through SIE+SHEAR, convolve Euclid PSF, 64px @ 0.1" (lenstronomy 1.14.1)."""
    numPix = PIX
    x, y, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = \
        util.make_grid_with_coordtransform(num_pix=numPix, delta_pix=PIX_SCALE,
                                           subgrid_res=1, left_lower=False, inverse=False)
    data_class = ImageData(image_data=np.zeros((numPix, numPix)),
                           ra_at_xy_0=ra_at_xy_0, dec_at_xy_0=dec_at_xy_0,
                           transform_pix2angle=Mpix2coord)
    psf_class  = PSF(psf_type="GAUSSIAN", fwhm=EUCLID_PSF_FWHM, pixel_size=PIX_SCALE)

    lens_model  = LensModel(["SIE", "SHEAR"])
    light_model = LightModel(["INTERPOL"])
    kwargs_src  = [{"image": source_img, "amp": 1.0, "scale": source_scale,
                    "phi_G": 0.0, "center_x": src_dx, "center_y": src_dy}]

    # supersample the ray-trace: pixel-center-only sampling (factor 1) is what rings/checkerboards
    kwargs_numerics = {"supersampling_factor": SUPERSAMPLE, "supersampling_convolution": False}
    image_model = ImageModel(data_class, psf_class, lens_model_class=lens_model,
                             source_model_class=light_model, kwargs_numerics=kwargs_numerics)
    arc = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_src,
                            source_add=True, lens_light_add=False, point_source_add=False)
    return np.asarray(arc, dtype=np.float64)


def pair_source_natural(sigma_v, z_l, rng, te_floor=TE_FLOOR, zmax=ZMAX):
    """
    Euclid Discovery Engine B method (2503.15325): pair so theta_E >= floor (0.75"),
    NO upper cap -- draw a source redshift uniformly above the minimum that reaches the
    floor. Gives the Collett/Euclid theta_E distribution. Vectorized cosmology.
    """
    zs_grid = np.linspace(z_l + 0.05, zmax, 300)
    te = theta_E_arcsec(sigma_v, z_l, zs_grid)     # vectorized over the whole grid
    ok = te >= te_floor
    if not ok.any():
        return None, None
    zs_min = zs_grid[ok][0]
    zs = float(rng.uniform(zs_min, zmax))
    return zs, float(theta_E_arcsec(sigma_v, z_l, zs))

# ----------------------------- main -------------------------------
def main(worker=0, n_workers=1):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED + 1000*worker)          # distinct stream per worker

    # this worker's share of the total, and its own shard-number band (10000 apart)
    my_target = N_TARGET // n_workers + (1 if worker < N_TARGET % n_workers else 0)
    shard_base = worker * 100000                              # worker w -> shards 100000*w ...
    tag = f"[w{worker}/{n_workers}]"

    d = np.load(DEFLECTOR_NPZ)
    defl_img = d["images"]; z_l_all = d["phz_median"]; mass_all = d["stellarmass"]
    oid_all = d["object_id"]
    Nd = len(defl_img)
    print(f"{tag} loaded {Nd} deflectors; my target {my_target}")

    # abundance-match sigma_v to Choi VDF, by stellar-mass rank (same for all workers -> deterministic)
    rng_ab = np.random.default_rng(SEED)                      # shared seed so sigma_v assignment matches
    sig_sorted = np.sort(sample_choi_vdf(Nd, rng_ab))
    order = np.argsort(mass_all)
    sigma_v_all = np.empty(Nd); sigma_v_all[order] = sig_sorted

    cat = make_cosmos_catalog()
    print(f"{tag} COSMOS sources: {cat.nobjects}")

    os.makedirs(OUT_DIR, exist_ok=True)
    # resume: count only THIS worker's existing shards (its own 100000-wide band)
    my_shards = [f for f in glob.glob(os.path.join(OUT_DIR, "shard_*.npz"))
                 if shard_base <= int(os.path.basename(f)[6:12]) < shard_base + 100000]
    shard_idx = shard_base + len(my_shards)
    made = len(my_shards) * CKPT_EVERY
    if made:
        print(f"{tag} resuming: {len(my_shards)} shards -> {made} lenses already done")

    use_count = np.zeros(Nd, dtype=np.int32)          # deflector reuse cap
    buf_L, buf_NL, buf_meta = [], [], []
    t0 = time.time(); attempted = 0
    order_idx = rng.permutation(Nd); ptr = 0

    while made < my_target:
        attempted += 1
        if attempted > my_target * 300:       # safety valve
            print(f"{tag} hit attempt ceiling"); break

        # iterate deflectors in permuted order, reshuffling when exhausted (reuse w/ new source/shear)
        if ptr >= Nd:
            order_idx = rng.permutation(Nd); ptr = 0
        di = int(order_idx[ptr]); ptr += 1
        if use_count[di] >= MAX_REUSE:        # reuse cap: skip over-used deflectors
            continue
        defl = defl_img[di].astype(np.float64)
        z_l = float(z_l_all[di]); sigma_v = float(sigma_v_all[di])

        # Euclid Discovery Engine B pairing: theta_E >= 0.75" floor, no upper cap (natural falloff)
        zs, theta_E = pair_source_natural(sigma_v, z_l, rng)
        if zs is None:                        # sigma too low to reach the floor
            continue

        src, src_scale = draw_source_hires(cat, rng)
        if src is None:
            continue

        e1, e2 = measure_shape(defl)
        # external shear: strength ~ N(0, SHEAR_SIGMA), random position angle (Euclid Disc.Engine A)
        g_ext = abs(rng.normal(0, SHEAR_SIGMA)); psi = rng.uniform(0, np.pi)
        g1, g2 = g_ext*np.cos(2*psi), g_ext*np.sin(2*psi)
        kwargs_lens = [{"theta_E": theta_E, "e1": e1, "e2": e2, "center_x": 0.0, "center_y": 0.0},
                       {"gamma1": g1, "gamma2": g2, "ra_0": 0.0, "dec_0": 0.0}]

        # source offset relative to the caustic (scales with theta_E)
        r = np.sqrt(rng.uniform(0, (0.5*theta_E)**2)); th = rng.uniform(0, 2*np.pi)
        src_dx, src_dy = r*np.cos(th), r*np.sin(th)

        try:
            arc = render_arc(src, src_scale, kwargs_lens, src_dx, src_dy)
        except Exception:
            continue
        if not np.isfinite(arc).all() or arc.sum() <= 0:
            continue

        # --- strong-lensing acceptance: solve lens equation (SIE+SHEAR), require multiple images ---
        try:
            xi, yi = _solver.image_position_from_source(
                src_dx, src_dy, kwargs_lens, min_distance=0.05, search_window=5)
            n_img = len(xi)
            mu_tot = float(np.sum(np.abs(_lensmodel.magnification(xi, yi, kwargs_lens))))
        except Exception:
            continue
        if n_img < 2 or not np.isfinite(mu_tot) or mu_tot < MU_MIN:
            continue                                     # not a genuine strong lens -> reject

        # --- HOLISMOKES-IV arc scaling: brightest arc pixel = (S/N) x background RMS ---
        # background RMS from the emptiest 10%x10% corner of THIS deflector cutout
        k = max(3, int(round(0.10 * PIX)))               # ~6px for 64px
        corners = [defl[:k, :k], defl[:k, -k:], defl[-k:, :k], defl[-k:, -k:]]
        bg_patch = min(corners, key=lambda c: c.max())   # lowest-max corner = emptiest
        bg_rms = float(np.std(bg_patch))
        if bg_rms <= 0 or not np.isfinite(bg_rms) or arc.max() <= 0:
            continue

        target_snr = float(rng.uniform(*ARC_SNR_RANGE))  # brightest-arc-pixel S/N vs background
        arc = arc * (target_snr * bg_rms / arc.max())

        # HOLISMOKES acceptance: brightest arc pixel must beat the deflector at that pixel
        peak_yx = np.unravel_index(np.argmax(arc), arc.shape)
        if arc[peak_yx] < defl[peak_yx]:
            continue                                     # arc drowned by deflector core -> reject

        # coadd onto the REAL cutout. NO added noise: the cutout already carries Euclid noise.
        lens_img = defl + arc
        buf_L.append(lens_img.astype(np.float32))
        buf_NL.append(defl.astype(np.float32))           # matched bare-deflector negative
        buf_meta.append((int(oid_all[di]), z_l, zs, sigma_v, theta_E, e1, e2, target_snr,
                         n_img, mu_tot, g_ext))
        use_count[di] += 1
        made += 1

        if len(buf_L) >= CKPT_EVERY:
            _save_shard(OUT_DIR, shard_idx, buf_L, buf_NL, buf_meta)
            print(f"{tag} shard {shard_idx:06d} | made {made}/{my_target} | attempted {attempted} | "
                  f"reuse<= {use_count.max()} | {time.time()-t0:.0f}s", flush=True)
            shard_idx += 1; buf_L, buf_NL, buf_meta = [], [], []

    if buf_L:                                            # final partial shard
        _save_shard(OUT_DIR, shard_idx, buf_L, buf_NL, buf_meta)
    print(f"{tag} DONE: {made} lenses ({time.time()-t0:.0f}s)", flush=True)


def _save_shard(out, idx, lenses, nonlenses, meta):
    m = np.array(meta)
    np.savez_compressed(os.path.join(out, f"shard_{idx:06d}.npz"),
        lenses=np.stack(lenses), nonlenses=np.stack(nonlenses),
        object_id=m[:,0].astype("int64"), z_l=m[:,1], z_s=m[:,2],
        sigma_v=m[:,3], theta_E=m[:,4], e1=m[:,5], e2=m[:,6], target_snr=m[:,7],
        n_img=m[:,8].astype("int64"), mu_tot=m[:,9], g_ext=m[:,10])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    a = ap.parse_args()
    main(worker=a.worker, n_workers=a.n_workers)
