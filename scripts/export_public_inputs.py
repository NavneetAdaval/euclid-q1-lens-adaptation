"""
export_public_inputs.py -- write ``public_inputs/``: everything the paper build reads from
files that cannot be redistributed, reduced to the columns and rows the build uses.
=====================================================================================

    python scripts/export_public_inputs.py

The paper build (``paper/scripts/make_all.py``) reads a handful of inputs that carry Euclid
pixels, catalogue copies, or the project record. This writes their pixel-free, minimal
extracts, so the build can run where those files are absent. ``paper/scripts/_common.py``
and ``scripts/integrity.py`` hold the loaders that switch between the originals and these
files (``PUBLIC_INPUTS=1`` forces the extracts).

WHAT IS WRITTEN (and what the build reads from it):

  lens_ids.npz            id_str, ra, dec of the 489 graded lenses, in row order
                          (from data/q1_lenses_processed/lens_cutouts.npz; no images)
  deflectors.npz          object_id_exact, ra, dec, stellarmass, phz_median, mvis of the
                          19,167-galaxy deflector sample, in row order
                          (from data_old/sim_deflectors/deflector_cutouts_clean.npz; no images)
  sim_meta.npz            shard, object_id_exact, z_l, sigma_v, theta_E, target_snr per
                          simulated lens, in shard order (from data/sim_arc_on_real/; no images)
  discovery_engine_v0.0.1_hard_negatives.csv
                          id_str, grade, right_ascension, declination of the 4,455 grade-X
                          hard negatives, copied token for token from the v0.0.1 catalogue
  discovery_engine_v0.0.3_hard_negative_rows.csv
                          the same columns for the v0.0.3 rows that are hard negatives
  mer_flux_test_negatives.csv, mer_flux_test_negatives_tiles.csv
                          object_id, flux_vis_psf of the 26,399 shared test negatives, copied
                          token for token from the two stored MER query outputs
  record_extract.json     NOT written here: scripts/write_record_extract.py writes it and its
                          manifest entry; this script leaves both alone
  MANIFEST.json           source, licence, rows, columns and sha256 of every file above

NO FLOAT-ROUNDED IDS. The stored object_id of the deflector file and of the simulation shards
is float-rounded (it equals float64 of the exact ID). Only the exact ID is written; the loaders
reconstruct the stored value as int64(float64(exact)), and this script asserts that this
reproduces every stored value.

Reads the originals; writes only under public_inputs/. Loads no model.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from exact_ids import from_idstr, rebuild          # noqa: E402
import integrity                                    # noqa: E402

OUT = os.path.join(REPO, "public_inputs")
MANIFEST = {}

CC_BY_NC = "CC BY-NC 3.0 IGO (Euclid Q1 data; credit ESA Euclid/Euclid Consortium/NASA/Q1-2025)"
CC_BY = "CC-BY-4.0"


def R(*p):
    return os.path.join(REPO, *p)


def stored_from_exact(e):
    """The float-rounded value the original files store for an exact ID."""
    return np.asarray(e, dtype=np.int64).astype(np.float64).astype(np.int64)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def note(name, source, licence, what, **extra):
    p = os.path.join(OUT, name)
    MANIFEST[name] = dict(source=source, licence=licence, contents=what,
                          bytes=os.path.getsize(p), sha256=sha(p), **extra)


def csv_subset(src, dst, cols, keep_row):
    """Copy the named columns of the rows ``keep_row(row)`` selects, token for token."""
    with open(src, newline="", encoding="utf-8") as fi:
        rows = list(csv.DictReader(fi))
    kept = [r for r in rows if keep_row(r)]
    with open(dst, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, lineterminator="\n")
        w.writerow(cols)
        w.writerows([[r[c] for c in cols] for r in kept])
    return len(kept)


def main() -> int:
    # the extracts are written FROM the originals; reading the extracts here would be circular
    assert os.environ.get("PUBLIC_INPUTS") != "1", "run without PUBLIC_INPUTS=1"
    os.makedirs(OUT, exist_ok=True)
    sp = json.load(open(R("data", "splits", "splits.json"), encoding="utf-8"))

    # ---- 1. lens IDs (the D1 filter) ----
    with np.load(R("data", "q1_lenses_processed", "lens_cutouts.npz"), allow_pickle=True) as z:
        id_str, ra, dec = np.asarray(z["id_str"]), np.asarray(z["ra"]), np.asarray(z["dec"])
    for s, a, b in zip(id_str, ra, dec):
        assert rebuild(float(a), float(b)) == from_idstr(str(s))
    np.savez_compressed(os.path.join(OUT, "lens_ids.npz"), id_str=id_str, ra=ra, dec=dec)
    note("lens_ids.npz", "data/q1_lenses_processed/lens_cutouts.npz (keys id_str, ra, dec)",
         CC_BY + " (Discovery Engine v0.0.3, doi:10.5281/zenodo.15025832)",
         "identifier and position of the 489 graded lenses, row order", rows=int(id_str.size))

    # ---- 2. deflector sample ----
    with np.load(R("data_old", "sim_deflectors", "deflector_cutouts_clean.npz")) as d:
        cols = {k: np.asarray(d[k]) for k in ("object_id", "ra", "dec", "stellarmass",
                                               "phz_median", "mvis")}
    exact = np.array([rebuild(float(a), float(b)) for a, b in zip(cols["ra"], cols["dec"])],
                     dtype=np.int64)
    assert np.array_equal(stored_from_exact(exact), cols["object_id"]), \
        "stored deflector IDs are not int64(float64(exact))"
    np.savez_compressed(os.path.join(OUT, "deflectors.npz"), object_id_exact=exact,
                        **{k: cols[k] for k in ("ra", "dec", "stellarmass", "phz_median", "mvis")})
    note("deflectors.npz", "data_old/sim_deflectors/deflector_cutouts_clean.npz "
         "(keys ra, dec, stellarmass, phz_median, mvis; object_id as exact ID)", CC_BY_NC,
         "the 19,167-galaxy deflector sample: exact ID, position, catalogue stellar mass, "
         "photometric redshift and mvis, row order", rows=int(exact.size))
    r2e = dict(zip(cols["object_id"].tolist(), exact.tolist()))

    # ---- 3. simulation metadata ----
    shards = sorted(glob.glob(R("data", "sim_arc_on_real", "shard_*.npz")))
    names = [os.path.basename(s) for s in shards]
    assert names == sp["sim_shards"], "shard order differs from splits.json sim_shards"
    keys = ("z_l", "sigma_v", "theta_E", "target_snr")
    acc = {k: [] for k in keys}
    sh, oid = [], []
    for p, n in zip(shards, names):
        with np.load(p) as s:
            o = np.asarray(s["object_id"], np.int64)
            for k in keys:
                acc[k].append(np.asarray(s[k]))
        sh += [n] * o.size
        oid.append(np.array([r2e[int(x)] for x in o], dtype=np.int64))
    oid = np.concatenate(oid)
    np.savez_compressed(os.path.join(OUT, "sim_meta.npz"), shard=np.array(sh),
                        object_id_exact=oid, **{k: np.concatenate(v) for k, v in acc.items()})
    note("sim_meta.npz", "data/sim_arc_on_real/shard_*.npz (keys object_id as exact ID, "
         + ", ".join(keys) + ")", "this work (GPL-3.0); deflector IDs from " + CC_BY_NC,
         "per simulated lens, in shard order: shard, deflector exact ID, z_l, sigma_v, "
         "theta_E, target arc S/N", rows=int(oid.size), shards=len(names))

    # ---- 4. Discovery Engine catalogue rows the build reads ----
    hn = {int(o) for o in sp["source_train"]["neg_mlfp"]["object_ids"]}

    def is_hn(r):
        s = r["id_str"]
        if "_" in s:
            t = s.rsplit("_", 1)[1]
            e = -int(t[3:]) if t.startswith("NEG") else int(t)
        else:
            e = rebuild(float(r["right_ascension"]), float(r["declination"]))
        return e in hn
    cols4 = ["id_str", "grade", "right_ascension", "declination"]
    n1 = csv_subset(R("data_old", "q1_lenses", "q1_discovery_engine_lens_catalog_old.csv"),
                    os.path.join(OUT, "discovery_engine_v0.0.1_hard_negatives.csv"), cols4, is_hn)
    assert n1 == len(hn) == 4455, n1
    note("discovery_engine_v0.0.1_hard_negatives.csv",
         "q1_discovery_engine_lens_catalog_old.csv = Discovery Engine v0.0.1 (doi:10.5281/zenodo.15003117)",
         CC_BY, "the rows of the 4,455 grade-X hard negatives; columns " + ", ".join(cols4),
         rows=n1)
    n3 = csv_subset(R("data_old", "q1_lenses", "q1_discovery_engine_lens_catalog.csv"),
                    os.path.join(OUT, "discovery_engine_v0.0.3_hard_negative_rows.csv"), cols4, is_hn)
    note("discovery_engine_v0.0.3_hard_negative_rows.csv",
         "q1_discovery_engine_lens_catalog.csv = Discovery Engine v0.0.3 (doi:10.5281/zenodo.15025832)",
         CC_BY, "the v0.0.3 rows whose ID is one of the 4,455 hard negatives; columns "
         + ", ".join(cols4), rows=n3)

    # ---- 5. MER PSF flux of the shared test negatives ----
    tn = {str(int(o)) for o in sp["test_negatives_shared"]["object_ids"]}
    for src, dst in (("partitioned_catalog.csv", "mer_flux_test_negatives.csv"),
                     ("partitioned_catalog_with_tiles.csv", "mer_flux_test_negatives_tiles.csv")):
        n = csv_subset(R("data_old", "q1_unlabeled", "metadata", src), os.path.join(OUT, dst),
                       ["object_id", "flux_vis_psf"], lambda r: r["object_id"] in tn)
        assert n == len(tn) == 26399, (src, n)
        note(dst, "data_old/q1_unlabeled/metadata/" + src + " (the stored MER query output)",
             CC_BY_NC, "object_id and flux_vis_psf (uJy) of the 26,399 shared test negatives",
             rows=n)

    # ---- 6. the record extract: NOT written here any more ----
    # public_inputs/record_extract.json and its MANIFEST.json entry are written by
    # scripts/write_record_extract.py (PROJECTv7 7.57-7.58). This script leaves the file alone
    # and carries its existing manifest entry over unchanged.
    old_manifest = os.path.join(OUT, "MANIFEST.json")
    if os.path.isfile(old_manifest):
        prev = json.load(open(old_manifest, encoding="utf-8")).get("record_extract.json")
        if prev:
            MANIFEST["record_extract.json"] = prev
    print("  record_extract.json: left alone (written by scripts/write_record_extract.py)")

    # ---- personal-information check on everything written ----
    # e-mail addresses, home-directory paths, and the account name of whoever runs this
    import getpass
    home = os.path.expanduser("~").replace("\\", "/").rstrip("/")
    terms = [r"[\w.+-]+@[\w-]+\.[A-Za-z][\w.-]+", r"/home/", r"/mnt/[a-z]/", re.escape(home),
             re.escape(home.replace("/", "\\")), re.escape(getpass.getuser())]
    pat = re.compile("|".join(terms), re.I)
    for name in MANIFEST:
        p = os.path.join(OUT, name)
        if name.endswith((".json", ".csv")):
            assert not pat.search(open(p, encoding="utf-8").read()), f"personal info in {name}"
        else:
            with np.load(p) as z:
                for k in z.files:
                    if z[k].dtype.kind == "U":
                        assert not any(pat.search(str(v)) for v in z[k]), f"{name}:{k}"
    big = [n for n, m in MANIFEST.items() if m["bytes"] > 25e6]
    assert not big, f"over 25 MB: {big}"

    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(MANIFEST, fh, indent=1)
    for n, m in MANIFEST.items():
        print(f"  {n:52s} {m['bytes']:>10,d} B  rows {m.get('rows')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
