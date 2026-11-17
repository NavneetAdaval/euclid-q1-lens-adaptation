"""
a2_top_negatives.py -- review round 1, Part A2: top-scoring test negatives.
POST HOC AND EXPLORATORY.

    python -B audit/review_r1/scripts/a2_top_negatives.py

For each arm at its headline readout (and DANN's 2000 sensitivity readout), folds 1-4,
D1 pool: the test negatives scoring at or above that fold's FPR=1e-3 threshold, plus the
top 30 by score. Cross-matched to the locally stored Discovery Engine catalogues
(v0.0.3 and v0.0.1) by exact ID and by position within 1 arcsec.

Writes out/A2_top_negatives.json, out/top_negatives.csv, out/A2_crossmatch.json,
out/A2_magnitude_bias.json and fig/top_neg_page_XX.png.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_common as RC                                   # noqa: E402

C = RC.C
SCRIPT = os.path.abspath(__file__)
POOL = "D1"
TOPN = 30
MATCH_ARCSEC = 1.0


def exact_from_idstr(s):
    """Exact Euclid ID from an id_str ending in '<tile>_<id>' or '<tile>_NEG<id>' (any
    prefix, e.g. 'OTF_RELEASE_'); 0 when the string carries no Euclid ID. Agrees with
    exact_ids.from_idstr on the '<tile>_...' form."""
    import re
    m = re.search(r"(\d+)_(NEG)?(\d+)$", s)
    if not m:
        return 0
    v = int(m.group(3))
    return -v if m.group(2) else v


def main():
    import exact_ids as X
    I = RC.Inputs()
    S = C.pool_sets()
    loc2id = S["loc2id"]
    mer = RC.mer_table(I)

    per = {}
    objs = {}                                     # exact id -> record
    with C.pool(POOL):
        for arm, step in RC.HEADLINE:
            t = RC.tag(arm, step)
            per[t] = {}
            for f in RC.CONF_FOLDS:
                v = RC.load_vec(RC.step_paths(arm, f)[step], f, arm, POOL, I)
                assert RC.suite_of(v)["tpr"] == float(C.suite(arm, f, step)["tpr"])
                thr = RC.threshold_of(v)
                order = np.argsort(-v["neg"], kind="mergesort")
                rank = np.empty(v["neg"].size, int)
                rank[order] = np.arange(1, v["neg"].size + 1)
                above = np.flatnonzero(v["neg"] >= thr)
                top = order[:TOPN]
                rec_above, rec_top = [], []
                for idx, dest in ((above, rec_above), (top, rec_top)):
                    for i in sorted(idx, key=lambda i: rank[i]):
                        loc = v["neg_locs"][i]
                        oid = loc2id[loc]
                        dest.append({"euclid_id": oid, "loc": loc, "score": float(v["neg"][i]),
                                     "rank": int(rank[i])})
                        o = objs.setdefault(oid, {"loc": loc, "above": set(), "top30": set(),
                                                  "best_rank": 10 ** 9, "ranks": {}})
                        o["best_rank"] = min(o["best_rank"], int(rank[i]))
                        o["ranks"][f"{t}|fold{f}"] = int(rank[i])
                        (o["above"] if dest is rec_above else o["top30"]).add(f"{t}|fold{f}")
                per[t][str(f)] = {"threshold": thr, "n_neg": int(v["neg"].size),
                                  "n_above": len(rec_above), "above_threshold": rec_above,
                                  "top30": rec_top}

    # ---------------- metadata + catalogue cross-match ----------------
    ids = sorted(objs)
    ra = mer["right_ascension"].reindex(ids).to_numpy(float)
    dec = mer["declination"].reindex(ids).to_numpy(float)
    mag = RC.ab_mag(mer["flux_vis_psf"].reindex(ids).to_numpy(float))
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    cats = {}
    for ver in ("v0.0.3", "v0.0.1"):
        I.add(C.ORIG[f"catalogue_{ver}"])
        t = C.catalogue(ver)
        # rows whose id_str carries no Euclid object ID (no "<tile>_" prefix) get 0 and are
        # matched by position only
        ex = np.array([exact_from_idstr(str(s)) for s in t["id_str"]], np.int64)
        n_noid = int((ex == 0).sum())
        cats[ver] = dict(exact=ex, grade=t["grade"].astype(str).to_numpy(),
                         ra=t["right_ascension"].to_numpy(float),
                         dec=t["declination"].to_numpy(float),
                         id_str=t["id_str"].astype(str).to_numpy(),
                         subset=(t["subset"].astype(str).to_numpy() if "subset" in t else None),
                         n_rows_without_euclid_id=n_noid)
    xm = {}
    have = np.isfinite(ra) & np.isfinite(dec)
    for ver, c in cats.items():
        e2i = {}
        for i, e in enumerate(c["exact"]):
            if e != 0:
                e2i.setdefault(int(e), []).append(i)
        sc_cat = SkyCoord(c["ra"] * u.deg, c["dec"] * u.deg)
        sc_obj = SkyCoord(ra[have] * u.deg, dec[have] * u.deg)
        idx, sep, _ = sc_obj.match_to_catalog_sky(sc_cat)
        pos_hit = {}
        for j, oi in enumerate(np.flatnonzero(have)):
            if sep[j].arcsec <= MATCH_ARCSEC:
                pos_hit[ids[oi]] = (int(idx[j]), float(sep[j].arcsec))
        xm[ver] = {"by_exact_id": {}, "by_position_1arcsec": {},
                   "n_catalogue_rows": int(c["exact"].size),
                   "n_rows_without_euclid_id_position_only": c["n_rows_without_euclid_id"]}
        for oid in ids:
            if oid in e2i:
                xm[ver]["by_exact_id"][str(oid)] = [
                    {"id_str": str(c["id_str"][i]), "grade": str(c["grade"][i])} for i in e2i[oid]]
            if oid in pos_hit:
                i, s = pos_hit[oid]
                xm[ver]["by_position_1arcsec"][str(oid)] = {
                    "id_str": str(c["id_str"][i]), "grade": str(c["grade"][i]), "sep_arcsec": s,
                    "catalogue_exact_id_equals": int(c["exact"][i]) == oid}
    # also against the 489 graded lenses (positions from lens_cutouts.npz)
    lm = RC.lens_meta(I)
    sc_l = SkyCoord(lm["ra"].astype(float) * u.deg, lm["dec"].astype(float) * u.deg)
    idx, sep, _ = SkyCoord(ra[have] * u.deg, dec[have] * u.deg).match_to_catalog_sky(sc_l)
    lens_pos = {}
    for j, oi in enumerate(np.flatnonzero(have)):
        if sep[j].arcsec <= MATCH_ARCSEC:
            lens_pos[str(ids[oi])] = {"lens_row": int(idx[j]), "id_str": str(lm["id_str"][idx[j]]),
                                      "grade": str(lm["grade"][idx[j]]),
                                      "sep_arcsec": float(sep[j].arcsec)}
    flagged = {}
    for ver in xm:
        for kind in ("by_exact_id", "by_position_1arcsec"):
            for oid, hit in xm[ver][kind].items():
                hits = hit if isinstance(hit, list) else [hit]
                for h in hits:
                    if h["grade"] in ("A", "B", "C", "X"):
                        flagged.setdefault(oid, []).append(
                            {"catalogue": ver, "match": kind, **h})
    combos_total = len(RC.HEADLINE) * len(RC.CONF_FOLDS)

    # ---------------- CSV ----------------
    rows = []
    for i, oid in enumerate(ids):
        o = objs[oid]
        arms_above = sorted({k.split("|")[0] for k in o["above"]})
        rows.append({
            "euclid_id": oid,
            "ra_deg": None if not np.isfinite(ra[i]) else round(float(ra[i]), 7),
            "dec_deg": None if not np.isfinite(dec[i]) else round(float(dec[i]), 7),
            "vis_psf_mag_ab": None if not np.isfinite(mag[i]) else round(float(mag[i]), 4),
            "n_armfold_above_threshold": len(o["above"]),
            "n_armfold_combinations": combos_total,
            "arms_above_threshold": ";".join(arms_above),
            "n_armfold_in_top30": len(o["top30"]),
            "best_rank": o["best_rank"],
            "catalogue_match_grades": ";".join(sorted({f"{h['catalogue']}:{h['grade']}:"
                                                       f"{'id' if h['match'] == 'by_exact_id' else 'pos'}"
                                                       for h in flagged.get(str(oid), [])})),
            "graded_lens_within_1arcsec": (lens_pos[str(oid)]["id_str"]
                                           if str(oid) in lens_pos else ""),
            "location": o["loc"],
        })
    rows.sort(key=lambda r: (-r["n_armfold_above_threshold"], r["best_rank"], r["euclid_id"]))
    os.makedirs(RC.OUT, exist_ok=True)
    csv_path = os.path.join(RC.OUT, "top_negatives.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("# status=post_hoc_exploratory; producing_script=" + RC.rel(SCRIPT) +
                 "; test_pool=D1; folds=1,2,3,4; arms=" +
                 ",".join(RC.tag(a, s) for a, s in RC.HEADLINE) +
                 "; provenance in A2_top_negatives.json\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {RC.rel(csv_path)}")

    # ---------------- magnitude bias ----------------
    import make_m_prose as MP
    with C.pool(POOL):
        paper_frac = MP.test_faint_fraction()
    sp = json.load(open(I.add(C.ART["splits"]), encoding="utf-8"))
    tn = sp["test_negatives_shared"]
    keep = C.pool_keep_neg([f"random:{a}:{b}" for a, b in tn["locations"]])
    pool_ids = np.array([int(o) for o in tn["object_ids"]], np.int64)[keep]
    pool_mag = RC.ab_mag(mer["flux_vis_psf"].reindex(pool_ids).to_numpy(float))
    assert np.isfinite(pool_mag).all()
    pool_frac = float((pool_mag > 22.5).mean())
    assert pool_frac == paper_frac
    above_ids = [r["euclid_id"] for r in rows if r["n_armfold_above_threshold"] > 0]
    am = RC.ab_mag(mer["flux_vis_psf"].reindex(above_ids).to_numpy(float))
    inst = []
    by_arm = {}
    for t, d in per.items():
        mm = []
        for f, rec in d.items():
            m = RC.ab_mag(mer["flux_vis_psf"].reindex(
                [x["euclid_id"] for x in rec["above_threshold"]]).to_numpy(float))
            inst.extend(m.tolist())
            mm.extend(m.tolist())
        mm = np.asarray(mm)
        by_arm[t] = {"n_instances": int(mm.size), "frac_fainter_22p5": float((mm > 22.5).mean()),
                     "median_mag": float(np.median(mm))}
    inst = np.asarray(inst)
    q = [5, 16, 50, 84, 95]
    bias = {
        "definition": "AB mag = 23.9 - 2.5 log10(MER flux_vis_psf / uJy); 'fainter than 22.5' "
                      "means mag > 22.5 (make_m_prose.test_faint_fraction, §3)",
        "D1_pool": {"n": int(pool_mag.size), "frac_fainter_22p5": pool_frac,
                    "reproduces_make_m_prose_test_faint_fraction": True,
                    "percentiles": dict(zip(map(str, q), np.percentile(pool_mag, q).tolist()))},
        "above_threshold_unique_objects": {
            "n": int(am.size), "n_with_mag": int(np.isfinite(am).sum()),
            "frac_fainter_22p5": float((am[np.isfinite(am)] > 22.5).mean()),
            "percentiles": dict(zip(map(str, q), np.percentile(am[np.isfinite(am)], q).tolist()))},
        "above_threshold_all_armfold_instances": {
            "n": int(inst.size), "frac_fainter_22p5": float((inst > 22.5).mean()),
            "percentiles": dict(zip(map(str, q), np.percentile(inst, q).tolist()))},
        "by_arm_instances": by_arm,
    }

    # ---------------- galleries ----------------
    multi = [r for r in rows if r["n_armfold_above_threshold"] >= 2]
    coral_above = []
    for f in RC.CONF_FOLDS:
        for x in per[RC.tag("coral", 2000)][str(f)]["above_threshold"]:
            coral_above.append((x["rank"], f, x["euclid_id"]))
    seen, coral_top = set(), []
    for rk, f, oid in sorted(coral_above, key=lambda t: (t[0], t[1], t[2])):
        if oid not in seen:
            seen.add(oid)
            coral_top.append(oid)
    coral_top = coral_top[:25]
    byid = {r["euclid_id"]: r for r in rows}
    seq = [("multi", r["euclid_id"]) for r in multi] + [("coral_top25", o) for o in coral_top]
    pages = []
    for p0 in range(0, len(seq), 25):
        chunk = seq[p0:p0 + 25]
        items = []
        for grp, oid in chunk:
            r = byid[oid]
            img, oid_img = RC.neg_image(r["location"], I)
            assert oid_img == oid
            magtxt = "n/a" if r["vis_psf_mag_ab"] is None else f"{r['vis_psf_mag_ab']:.2f}"
            items.append((img, f"{oid}\nn={r['n_armfold_above_threshold']}  m={magtxt}"
                               + ("  [CORAL top25]" if grp == "coral_top25" else "")))
        path = os.path.join(RC.FIG, f"top_neg_page_{len(pages) + 1:02d}.png")
        RC.gallery(items, path, f"Top-scoring D1 test negatives, page {len(pages) + 1} "
                                f"(left: encoder stretch; right: linear 1-99.5 pct). "
                                f"n = arm x fold combinations above threshold (of "
                                f"{combos_total}). POST HOC.")
        pages.append({"file": RC.rel(path), "objects": [[g, o] for g, o in chunk]})

    RC.write_json("A2_top_negatives.json", {
        "per_arm_fold": per,
        "n_unique_objects": len(rows),
        "n_unique_above_threshold_any": sum(r["n_armfold_above_threshold"] > 0 for r in rows),
        "n_unique_above_threshold_ge2": len(multi),
        "count_distribution": {str(k): sum(r["n_armfold_above_threshold"] == k for r in rows)
                               for k in range(combos_total + 1)},
        "csv": RC.rel(csv_path),
        "galleries": {"rule": "all objects above threshold in >= 2 arm x fold combinations, "
                              "sorted by that count (desc) then best rank, then the first 25 "
                              "distinct objects above CORAL@2000's threshold ordered by their "
                              "best CORAL rank (ties: fold, then ID); objects may appear in both "
                              "groups", "pages": pages},
    }, script=SCRIPT, inputs=I, pool=POOL, folds=RC.CONF_FOLDS)
    RC.write_json("A2_crossmatch.json", {
        "match_radius_arcsec": MATCH_ARCSEC,
        "coordinates_source": "MER query output right_ascension/declination (" +
                              C.INPUT_LABEL["mer_flux"] + ")",
        "catalogue_exact_id_source": "id_str (exact; the v0.0.3 object_id column is float-rounded)",
        "n_objects_with_coordinates": int(have.sum()), "n_objects": len(ids),
        "matches": xm,
        "graded_lens_set_within_1arcsec": lens_pos,
        "objects_matching_a_grade_A_B_C_X_entry": flagged,
        "n_objects_matching_A_B_C_X": len(flagged),
        "positional_match_sets_equal_exact_id_match_sets": {
            v: set(xm[v]["by_exact_id"]) == set(xm[v]["by_position_1arcsec"]) for v in xm},
        "every_positional_match_has_the_identical_exact_id": all(
            h["catalogue_exact_id_equals"] for v in xm
            for h in xm[v]["by_position_1arcsec"].values()),
    }, script=SCRIPT, inputs=I, pool=POOL, folds=RC.CONF_FOLDS)
    RC.write_json("A2_magnitude_bias.json", bias, script=SCRIPT, inputs=I, pool=POOL,
                  folds=RC.CONF_FOLDS)


if __name__ == "__main__":
    main()
