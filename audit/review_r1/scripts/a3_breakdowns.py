"""
a3_breakdowns.py -- review round 1, Part A3: astronomer-facing breakdowns.
POST HOC AND EXPLORATORY. DESCRIPTIVE ONLY: no contrast is formed.

    python -B audit/review_r1/scripts/a3_breakdowns.py

Recall at each fold's own FPR=1e-3 threshold (D1 pool), folds 1-4, every arm at its
headline readout (DANN's 2000 sensitivity readout listed separately): by grade, and by
the per-lens columns stored locally (expert score, Einstein radius, arc brightness).
"Recovered" = test positive with score >= metrics.operating_threshold.

Writes out/A3_recall_by_grade.json, out/A3_recall_by_property.json,
out/A3_lens_columns.json, out/A3_galleries.json and fig/lens_*.png.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_common as RC                                   # noqa: E402

C = RC.C
SCRIPT = os.path.abspath(__file__)
POOL = "D1"
HEADLINE6 = [(a, s) for a, s in RC.HEADLINE if (a, s) != ("dann", 2000)]
DA4 = [("bnm", 4000), ("freematch", 1500), ("dann", 4000), ("coral", 2000)]
CAT_FILES = ["q1_discovery_engine_lens_catalog.csv", "q1_discovery_engine_lens_catalog_old.csv",
             "lens_manifest_positive.csv", "modeling_lens_mass.csv", "modeling_lens_mge.csv",
             "modeling_lens_sersic.csv", "modeling_mge_magnitude.csv"]


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def main():
    import exact_ids as X
    import pandas as pd
    I = RC.Inputs()
    lm = RC.lens_meta(I)
    grade = lm["grade"].astype(str)
    exact = np.array([X.from_idstr(str(s)) for s in lm["id_str"]], np.int64)

    # ---------------- recovered flags per arm/fold ----------------
    rec = {}            # tag -> {row: bool}
    fold_of = {}
    checks = {}
    with C.pool(POOL):
        for arm, step in RC.HEADLINE:
            t = RC.tag(arm, step)
            rec[t] = {}
            checks[t] = {}
            for f in RC.CONF_FOLDS:
                v = RC.load_vec(RC.step_paths(arm, f)[step], f, arm, POOL, I)
                thr = RC.threshold_of(v)
                hit = v["pos"] >= thr
                for r, h in zip(v["pos_rows"], hit):
                    rec[t][r] = bool(h)
                    fold_of[r] = f
                checks[t][str(f)] = {"threshold": thr, "n_pos": int(v["pos"].size),
                                     "n_recovered_count": int(hit.sum()),
                                     "count_rate": float(hit.mean()),
                                     "interpolated_tpr": RC.tpr_of(v)}
    rows = sorted(fold_of)

    # ---------------- recall by grade ----------------
    by_grade = {}
    for t in rec:
        out = {}
        for g in ("A", "B"):
            per = {}
            K = N = 0
            for f in RC.CONF_FOLDS:
                rr = [r for r in rows if fold_of[r] == f and grade[r] == g]
                k = sum(rec[t][r] for r in rr)
                per[str(f)] = {"recovered": k, "total": len(rr)}
                K += k
                N += len(rr)
            lo, hi = clopper_pearson(K, N)
            out[g] = {"per_fold": per, "summed": {"recovered": K, "total": N, "rate": K / N,
                                                   "clopper_pearson_95": [lo, hi]}}
        other = sorted({grade[r] for r in rows} - {"A", "B"})
        out["grades_present_other_than_A_B"] = other
        by_grade[t] = out

    # ---------------- per-lens columns available ----------------
    cols = {}
    for fn in CAT_FILES:
        p = I.add(RC.R("data_old", "q1_lenses", fn))
        t = pd.read_csv(p)
        cols[RC.rel(p)] = {"n_rows": int(len(t)), "columns": list(t.columns)}
    lc = RC.R("data", "q1_lenses_processed", "lens_cutouts.npz")
    with np.load(lc, allow_pickle=True) as z:
        cols[RC.rel(lc)] = {"n_rows": int(len(z["id_str"])),
                            "columns": [k for k in z.files if k != "images"]}

    # ---------------- recall by property ----------------
    mass = pd.read_csv(RC.R("data_old", "q1_lenses", "modeling_lens_mass.csv"))
    mag = pd.read_csv(RC.R("data_old", "q1_lenses", "modeling_mge_magnitude.csv"))
    idx = {str(s): i for i, s in enumerate(lm["id_str"])}

    def by_id(df, col):
        v = np.full(len(lm["id_str"]), np.nan)
        for s, x in zip(df["id_str"].astype(str), df[col].astype(float)):
            if s in idx:
                v[idx[s]] = x
        return v
    props = {
        "expert_score (lens_cutouts.npz)": lm["expert_score"].astype(float),
        "einstein_radius_effective_median_pdf (modeling_lens_mass.csv)":
            by_id(mass, "einstein_radius_effective_median_pdf"),
        "vis_lensed_source_magnitude_ab_median_pdf (modeling_mge_magnitude.csv; arc brightness)":
            by_id(mag, "vis_lensed_source_magnitude_ab_median_pdf"),
        "vis_max_lensed_source_signal_to_noise_ratio (modeling_mge_magnitude.csv)":
            by_id(mag, "vis_max_lensed_source_signal_to_noise_ratio"),
    }
    by_prop = {}
    rows_a = np.array(rows)
    for pname, vals in props.items():
        x = vals[rows_a]
        ok = np.isfinite(x)
        edges = np.quantile(x[ok], [0, 1 / 3, 2 / 3, 1]).tolist()
        bins = np.digitize(x, edges[1:-1], right=True)          # 0,1,2 ; nan handled below
        rec_p = {"n_test_lenses_folds1_4": int(rows_a.size), "n_with_value": int(ok.sum()),
                 "bin_edges": edges, "bin_rule": "tertiles of the folds 1-4 test lenses "
                 "with a value; bin i = (edge_i, edge_i+1], first bin closed below",
                 "n_value_exactly_zero": int((x[ok] == 0).sum()),
                 "n_value_gt_30": int((x[ok] > 30).sum()),
                 "bin_counts": [int(((bins == b) & ok).sum()) for b in range(3)],
                 "bin_grade_counts": [{g: int(((bins == b) & ok & (grade[rows_a] == g)).sum())
                                       for g in ("A", "B")} for b in range(3)],
                 "recall": {}}
        for t in rec:
            rr = []
            for b in range(3):
                sel = rows_a[(bins == b) & ok]
                k = int(sum(rec[t][int(r)] for r in sel))
                lo, hi = clopper_pearson(k, len(sel))
                rr.append({"recovered": k, "total": int(len(sel)), "rate": k / len(sel),
                           "clopper_pearson_95": [lo, hi]})
            rec_p["recall"][t] = rr
        by_prop[pname] = rec_p

    # ---------------- galleries ----------------
    def missed_all(r):
        return not any(rec[RC.tag(a, s)][r] for a, s in HEADLINE6)
    sets = {
        "a_recovered_by_all_four_DA_arms": [r for r in rows
                                            if all(rec[RC.tag(a, s)][r] for a, s in DA4)],
        "b_missed_by_every_arm": [r for r in rows if missed_all(r)],
        "c_recovered_by_ge3_DA_arms_missed_by_lambda0": [
            r for r in rows if sum(rec[RC.tag(a, s)][r] for a, s in DA4) >= 3
            and not rec[RC.tag("lambda0", 4000)][r]],
    }
    gal = {}
    for name, rr in sets.items():
        chosen = sorted(rr, key=lambda r: int(exact[r]))[:16]
        items = []
        for r in chosen:
            img = RC.lens_image(r, I)
            items.append((img, f"{int(exact[r])}\n{grade[r]} es={float(lm['expert_score'][r]):.2f}"
                               f" f{fold_of[r]}"))
        path = os.path.join(RC.FIG, f"lens_{name}.png")
        RC.gallery(items, path, f"Lenses: {name} ({len(rr)} in set; first 16 by exact Euclid "
                                f"ID). Left: encoder stretch; right: linear 1-99.5 pct. "
                                f"Folds 1-4, D1 thresholds. POST HOC.")
        gal[name] = {"n_in_set": len(rr), "shown": [{"lens_row": int(r), "euclid_id": int(exact[r]),
                                                     "id_str": str(lm["id_str"][r]),
                                                     "grade": grade[r], "fold": fold_of[r]}
                                                    for r in chosen],
                     "grade_counts_in_set": {g: sum(grade[r] == g for r in rr) for g in ("A", "B")},
                     "file": RC.rel(path)}
    defs = {"DA arms": [RC.tag(a, s) for a, s in DA4],
            "every arm": [RC.tag(a, s) for a, s in HEADLINE6],
            "selection": "sort by exact Euclid object ID (integer, ascending), take the first 16"}

    RC.write_json("A3_recall_by_grade.json", {"by_arm": by_grade, "threshold_checks": checks},
                  script=SCRIPT, inputs=I, pool=POOL, folds=RC.CONF_FOLDS,
                  notes=["descriptive; no contrast formed",
                         "dann@2000 is DANN's pre-declared sensitivity readout"])
    RC.write_json("A3_lens_columns.json", {"columns_by_file": cols}, script=SCRIPT, inputs=I,
                  pool="n/a", folds="n/a")
    RC.write_json("A3_recall_by_property.json", by_prop, script=SCRIPT, inputs=I, pool=POOL,
                  folds=RC.CONF_FOLDS, notes=["descriptive; no contrast formed"])
    RC.write_json("A3_galleries.json", {"definitions": defs, "galleries": gal}, script=SCRIPT,
                  inputs=I, pool=POOL, folds=RC.CONF_FOLDS)


if __name__ == "__main__":
    main()
