r"""
make_m_prose.py -- macros for the result numbers the prose quotes.
==================================================================

    python paper/scripts/make_m_prose.py

The prose used to type its result numbers. Each one is now a macro defined here from the
same artifact and the same ``_common`` call that fills the table it comes from, on the
current test pool (``PAPER_POOL``). The macro prints exactly as the typed number did --
same decimals, same explicit sign -- so on the pre-registered pool every macro renders the
text that was typed before (``audit/decontam/verify_prose_macros.py`` checks that, number
by number).

Sources, by macro family:

  * ``\fm*``, ``\sd*``        -- ``_common.fold_mean`` (Table 1);
  * ``\ct*``                  -- ``_common.contrast`` (Table 2);
  * ``\lp*``                  -- the licensed pair's bootstrap artifact (Table 2, last row);
  * ``\rOne*``                -- round-1 score dumps and paired comparisons (Table 3);
  * ``\lr*``                  -- the Lines-ratio artifact (Table 4, Table 5 per fold);
  * ``\au*``                  -- ``_common.fold_mean(.., "auroc")`` (Table 4);
  * ``\wa*``                  -- the window-average artifact (Table 6; scatter reduction);
  * ``\wz*``                  -- the fold-0 window artifact (Figure 5a);
  * ``\fz*``                  -- the fold-0 controlled pair's curves (Figure 4).

Writes ``paper/tables/M_prose.tex`` and nothing else. Trains nothing, loads no model.
"""
from __future__ import annotations

import csv
import json
import math
import os

import _common as C

MACROS: list = []          # (name, value, nd, signed, source)


def m(name, value, nd=4, signed=False, source=""):
    MACROS.append((name, float(value), nd, signed, source))


def render(value, nd, signed):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return r"\text{n/a}"
    return f"{value:+.{nd}f}" if signed else f"{value:.{nd}f}"


def _load(key):
    return json.load(open(C.art(key), encoding="utf-8"))


def fold0_curves():
    """{arm: {step: tpr}} and the fold-0 init line, on the current pool."""
    if C.POOL == "orig":
        out = {}
        for key, art in (("arm1_lambda0", "dann_arm1_curve"), ("arm2_dann", "dann_arm2_curve")):
            rows = csv.DictReader(open(C.ART[art], newline="", encoding="utf-8"))
            out[key] = {r["step"]: float(r["tpr_at_1e-3"]) for r in rows}
        return out, float(C.suite("baseline", 0, 8000)["tpr"])
    d = json.load(open(os.path.join(C.DATA, f"fold0_curves_{C.POOL}.json"), encoding="utf-8"))
    return d["curves"], d["init_baseline_fold0"]


def build() -> list:
    MACROS.clear()
    T1, T2 = "Table 1 (fold_mean)", "Table 2 (contrast)"
    # ---------------- Table 1: fold-means and spreads ----------------
    for key, arm, ro in [("Baseline", "baseline", 8000), ("Lambda", "lambda0", 4000),
                         ("Bnm", "bnm", 4000), ("FreeMatch", "freematch", 1500),
                         ("Dann", "dann", 4000), ("DannSens", "dann", 2000),
                         ("Coral", "coral", 2000)]:
        mean, sd, _ = C.fold_mean(arm, "tpr", ro)
        m(f"fm{key}", mean, source=T1)
        m(f"sd{key}", sd, source=T1)
    m("auLambda", C.fold_mean("lambda0", "auroc", 4000)[0], source="Table 4 (AUROC)")

    # ---------------- Table 2: contrasts ----------------
    for key, a, ra, b, rb in [("LamBase", "lambda0", 4000, "baseline", 8000),
                              ("DannLam", "dann", 4000, "lambda0", 4000),
                              ("CoralLam", "coral", 2000, "lambda0", 2000),
                              ("FmLam", "freematch", 1500, "lambda0", 1500),
                              ("BnmLam", "bnm", 4000, "lambda0", 4000),
                              ("BnmCoral", "bnm", 2000, "coral", 2000),
                              ("BnmDann", "bnm", 4000, "dann", 4000),
                              ("BnmFm", "bnm", 1500, "freematch", 1500),
                              ("DannCoral", "dann", 4000, "coral", 2000),
                              ("DannFm", "dann", 4000, "freematch", 1500),
                              ("DannCoralMatched", "dann", 2000, "coral", 2000)]:
        r = C.contrast(a, ra, b, rb)
        m(f"ct{key}", r["mean"], signed=True, source=T2)
        m(f"ct{key}Lo", r["ci_lo"], signed=True, source=T2)
        m(f"ct{key}Hi", r["ci_hi"], signed=True, source=T2)
        m(f"ct{key}Abs", abs(r["mean"]), source=T2)
    # BNM's share of the distance, both readings of Section 6 (whole per cent)
    fb, fl, fc = (C.fold_mean(a, "tpr", ro)[0] for a, ro in
                  (("bnm", 4000), ("lambda0", 4000), ("coral", 2000)))
    m("bnmShareUnmatched", 100.0 * (fb - fl) / (fc - fl), nd=0, source="Table 1 ratio")
    m("bnmShareMatched", 100.0 * C.contrast("bnm", 4000, "lambda0", 4000)["mean"]
      / C.contrast("coral", 2000, "lambda0", 2000)["mean"], nd=0, source="Table 2 ratio")

    # ---------------- licensed pair ----------------
    lp = _load("foldmean_bootstrap")
    m("lpDiff", lp["diff"], signed=True, source="foldmean_bootstrap")
    m("lpLo", lp["ci"][0], signed=True, source="foldmean_bootstrap")
    m("lpHi", lp["ci"][1], signed=True, source="foldmean_bootstrap")

    # ---------------- Table 3: round 1 ----------------
    for key, run, tag in [("Base", "bakeoff_baseline_A", None), ("Coral", "bakeoff_coral", "coral"),
                          ("Ada", "bakeoff_adamatch_5fold", "adamatch"),
                          ("Fm", "bakeoff_freematch_5fold", "freematch")]:
        rs = C.round1_summary(run)
        m(f"rOne{key}", rs["across_fold"]["tpr"]["mean"], source="Table 3")
        m(f"rOne{key}Sd", rs["across_fold"]["tpr"]["std"], source="Table 3")
        m(f"rOne{key}Reach", sum(1 for v in rs["per_fold"].values()
                                  if math.isfinite(v["comp_at_purity"])), nd=0, source="Table 3")
        if tag:
            pname = (f"round1_paired_{tag}.json" if C.POOL == "orig"
                     else f"round1_paired_{tag}_{C.POOL}.json")
            agg = json.load(open(os.path.join(C.DATA, pname), encoding="utf-8"))["aggregate"]["tpr"]
            m(f"rOne{key}Diff", agg["diff"], signed=True, source="Table 3 paired")
            m(f"rOne{key}Lo", agg["ci"][0], signed=True, source="Table 3 paired")
            m(f"rOne{key}Hi", agg["ci"][1], signed=True, source="Table 3 paired")

    # ---------------- Table 4 / 5: completeness at 24% purity and X ----------------
    lr = _load("lines_ratios")["arms"]
    for key, arm, ro in [("Coral", "CORAL x nb3", 2000), ("FreeMatch", "FreeMatch x nb3", 1500),
                         ("Bnm", "BNM x nb3", 4000), ("Dann", "DANN x nb3", 4000),
                         ("DannSens", "DANN x nb3", 2000)]:
        rec = lr[arm]["readouts"][str(ro)]
        ok = rec["defined_on_all_folds"]
        m(f"lr{key}Comp", rec["fold_mean"] if ok else float("nan"), source="Table 4")
        m(f"lr{key}X", rec["pct_of_lines"] if ok else float("nan"), nd=1, source="Table 4")
        lo, hi = rec["pct_of_lines_ci95"] if ok else (float("nan"), float("nan"))
        m(f"lr{key}XLo", lo, nd=1, signed=True, source="Table 4")
        m(f"lr{key}XHi", hi, nd=1, signed=True, source="Table 4")
        m(f"lr{key}CompFoldTwo", rec["comp_at_purity_per_fold"]["2"] or float("nan"),
          source="Table 5 (fold 2)")

    # ---------------- Table 6: window averaging, k = 8, and scatter reduction ----------------
    wa = _load("window_average")["arms"]
    for key, arm, ro in [("FreeMatch", "FreeMatch x nb3", 1500), ("Coral", "CORAL x nb3", 2000),
                         ("Dann", "DANN x nb3", 4000), ("DannSens", "DANN x nb3", 2000)]:
        rec = wa[arm]["readouts"][str(ro)]
        m(f"wa{key}Logit", rec["mean_logit"]["8"]["tpr_mean"], source="Table 6 k=8")
        m(f"wa{key}Rank", rec["mean_rank"]["8"]["tpr_mean"], source="Table 6 k=8")
    sr = _load("window_average")["scatter_reduction"]
    m("waLambdaAchieved", sr["lambda=0"]["8"]["achieved_over_expected"], nd=2,
      source="window scatter reduction")
    m("waLambdaSdOne", sr["lambda=0"]["1"]["sd_k_foldmean"], source="window scatter reduction")
    m("waLambdaSdEight", sr["lambda=0"]["8"]["sd_k_foldmean"], source="window scatter reduction")
    m("waDannAchieved", sr["DANN x nb3"]["8"]["achieved_over_expected"], nd=2,
      source="window scatter reduction")
    m("waBnmAchieved", sr["BNM x nb3"]["8"]["achieved_over_expected"], nd=2,
      source="window scatter reduction")
    m("waFreeMatchAchieved", sr["FreeMatch x nb3"]["8"]["achieved_over_expected"], nd=2,
      source="window scatter reduction")

    # ---------------- Figure 5a: the fold-0 window analysis ----------------
    wz = _load("window_fold0")["arms"]
    names = {"Coral": "CORAL x nb3", "FreeMatch": "FreeMatch x nb3",
             "Dann": "DANN arm 2 (lambda live)", "Lambda": "DANN arm 1 (lambda=0)",
             "Bnm": "BNM x nb3"}
    for key, arm in names.items():
        for k, word in (("1", "One"), ("8", "Eight")):
            m(f"wz{key}Sd{word}", wz[arm][k]["sd"], source="fold-0 window")
            m(f"wz{key}Mult{word}", wz[arm][k]["multiple_of_dann_minus_coral"], nd=1,
              source="fold-0 window")
    ranked = [wz[names[k]]["1"]["multiple_of_dann_minus_coral"]
              for k in ("Coral", "FreeMatch", "Dann")]
    m("wzRankedMultMin", min(ranked), nd=1, source="fold-0 window")
    m("wzRankedMultMax", max(ranked), nd=1, source="fold-0 window")
    eight = [wz[a]["8"]["multiple_of_dann_minus_coral"] for a in names.values()]
    m("wzEightMultMin", min(eight), nd=1, source="fold-0 window")
    m("wzEightMultMax", max(eight), nd=1, source="fold-0 window")

    # ---------------- Figure 4: the fold-0 controlled pair ----------------
    cv, init = fold0_curves()
    lam = cv["arm1_lambda0"]
    last = lam[max(lam, key=int)]
    m("fzLambdaLast", last, source="fold-0 curves")
    m("fzInit", init, source="fold-0 curves")
    m("fzLambdaArgmax", int(max(lam, key=lambda s: (lam[s], -int(s)))), nd=0,
      source="fold-0 curves")

    # ---------------- added by the wording pass (prose_wording_brief.md) ----------------
    ONE = 1.0 / 98                                        # the one-positive granularity
    # A6 / H: Baseline A's fold-0 value on the pre-registered pool, on which the fold-0
    # probe was scored (and at which cell A matched it)
    with C.pool("orig"):
        m("baseFoldZeroPrereg", C.suite("baseline", 0, 8000)["tpr"],
          source="pre-registered pool (fold 0)")
    # B: the licensed pair's lower bound in one-positive units
    m("lpLoInPositives", lp["ci"][0] / ONE, nd=1, source="foldmean_bootstrap / (1/98)")
    # C3: BNM - CORAL's nearer bound in one-positive units
    m("ctBnmCoralHiInPositives", abs(C.contrast("bnm", 2000, "coral", 2000)["ci_hi"]) / ONE,
      nd=1, source="Table 2 (contrast) / (1/98)")
    # D1 / D6: Baseline A's benchmark ratio on the folds 1-4 basis (Table 4's row)
    b = lr["Baseline A (frozen)"]["readouts"]["8000"]
    b14 = b.get("folds_1_4") if b["defined_on_all_folds"] else None
    m("lrBaselineX", b14["pct_of_lines"] if b14 else float("nan"), nd=1, source="Table 4")
    m("lrBaselineXLo", 100 * b14["ci95"][0] / C.LINES_COMPLETENESS if b14 else float("nan"),
      nd=1, signed=True, source="Table 4")
    m("lrBaselineXHi", 100 * b14["ci95"][1] / C.LINES_COMPLETENESS if b14 else float("nan"),
      nd=1, signed=True, source="Table 4")
    # E1: the two combinations' DANN-CORAL separations at k = 8
    lg = {k: wa[a]["readouts"][str(r)]["mean_logit"]["8"]["tpr_mean"] for k, a, r in
          (("coral", "CORAL x nb3", 2000), ("d4", "DANN x nb3", 4000), ("d2", "DANN x nb3", 2000))}
    rk = {k: wa[a]["readouts"][str(r)]["mean_rank"]["8"]["tpr_mean"] for k, a, r in
          (("coral", "CORAL x nb3", 2000), ("d4", "DANN x nb3", 4000), ("d2", "DANN x nb3", 2000))}
    m("waLogitDannCoralMaxGap", max(lg["d4"] - lg["coral"], lg["d2"] - lg["coral"]),
      source="Table 6 k=8")
    m("waRankCoralDannMinGap", min(rk["coral"] - rk["d4"], rk["coral"] - rk["d2"]),
      source="Table 6 k=8")
    # F1: the k = 1 scatter against the second difference, |DANN - FreeMatch|
    dfm = abs(C.contrast("dann", 4000, "freematch", 1500)["mean"])
    for key in ("Coral", "FreeMatch", "Dann"):
        m(f"wz{key}MultFm", wz[names[key]]["1"]["sd"] / dfm, nd=1, source="fold-0 window")
    # F2 / F4: the k = 8 range at two decimals, and CORAL's shortfall
    m("wzEightMultMinTwo", min(eight), nd=2, source="fold-0 window")
    m("wzEightMultMaxTwo", max(eight), nd=2, source="fold-0 window")
    m("wzCoralEightGap", abs(C.contrast("dann", 4000, "coral", 2000)["mean"])
      - wz[names["Coral"]]["8"]["sd"], source="fold-0 window")

    # ---------------- added by the final wording pass (wording_final_brief.md) ----------------
    # D8: FreeMatch vs CORAL fold-mean completeness curves, built exactly as Figure 6 builds them
    d8 = d8_curves()
    m("dEightFmAbovePct", 100 * d8["frac_fm_above"], nd=0, source="Figure 6 curves (D8)")
    # Section 7.3: the grid each arm's scatter reduction was computed on -- the checkpoints the
    # analysis's own loader returns (step 0, the initialisation, is excluded by _scan_dir)
    for key, arm in (("Dann", "DANN x nb3"), ("Bnm", "BNM x nb3")):
        n = grid_points(arm)
        m(f"wa{key}GridPoints", n, nd=0, source="window scatter reduction grid")
        m(f"wa{key}EightPositions", n - 8 + 1, nd=0, source="window scatter reduction grid")

    # ---------------- added by the provenance-fixes pass (provenance_fixes_brief.md) ----------------
    # The deflector sample (Section 3, Section 9): the recoverable selection, read from the
    # 19,167-galaxy file the simulator draws from. Pool-independent.
    dd = deflectors()
    m("DeflMassFloor", math.floor(dd["stellarmass"].min() * 10) / 10, nd=1,
      source="deflector sample (deflector_cutouts_clean.npz)")
    m("DeflZmin", dd["phz_median"].min(), nd=2, source="deflector sample (deflector_cutouts_clean.npz)")
    m("DeflZmax", dd["phz_median"].max(), nd=2, source="deflector sample (deflector_cutouts_clean.npz)")
    m("DeflUnphysMass", DEFL_UNPHYS_MASS, nd=1, source="deflector sample (deflector_cutouts_clean.npz)")
    m("DeflUnphysMag", DEFL_UNPHYS_MVIS, nd=0, source="deflector sample (deflector_cutouts_clean.npz)")
    m("DeflUnphysN", int(((dd["stellarmass"] > DEFL_UNPHYS_MASS) | (dd["mvis"] < DEFL_UNPHYS_MVIS)).sum()),
      nd=0, source="deflector sample (deflector_cutouts_clean.npz)")
    # gradex_brief.md: the unphysical-value deflectors that were drawn, and the lenses they host
    m("DeflUnphysDrawn", len(dd["unphys_drawn"]), nd=0, source="deflectors drawn into the shards")
    m("DeflUnphysLenses", dd["unphys_lenses"], nd=0, source="deflectors drawn into the shards")
    # gradex_brief.md: the grade-X hard negatives graded C in version 0.0.3
    m("GradeXnowC", len(gradex()["now_c"]), nd=0, source="grade-X list vs catalogue v0.0.3")
    # the hard negatives used, and the grade-X rows of v0.0.1 (part 1's reference set c)
    m("GradeXUsed", len(gradex()["hn"]), nd=0, source="splits.json source_train.neg_mlfp")
    m("GradeXAll", grade_x_reference_n(), nd=0,
      source="data/splits/dilute10_pool.json reference set c (dilute10 part 1)")
    # screening_cut_brief.md: the applied MER selection, from the stored query's constants
    q = mer_query()
    m("FluxMin", q["flux_min"], nd=1, source="scripts/01_query_catalog.py (MER query)")
    m("FluxMax", q["flux_max"], nd=0, source="scripts/01_query_catalog.py (MER query)")
    m("MagFaint", ab_mag(q["flux_min"]), nd=1, source="scripts/01_query_catalog.py (MER query)")
    m("MagBright", ab_mag(q["flux_max"]), nd=1, source="scripts/01_query_catalog.py (MER query)")
    m("FracFaintTest", 100 * test_faint_fraction(), nd=0,
      source=f"test negatives on pool {C.POOL}, PSF magnitude > {BENCH_IE_LIMIT}")
    return MACROS


BENCH_IE_LIMIT = 22.5      # the benchmark's I_E limit (lines2025simstosky, Sect. 2.2)
QUERY_SCRIPT = ("scripts", "01_query_catalog.py")
# The query's other conditions, exactly as the script writes them; Section 3 states each.
QUERY_OTHER = ["vis_det = 1", "flux_y_templfit > 0", "flux_j_templfit > 0",
               "flux_h_templfit > 0", "point_like_prob < 0.3", "extended_prob > 0.5",
               "spurious_prob < 0.3"]
_Q: dict = {}


def ab_mag(f_ujy: float) -> float:
    """AB magnitude of a flux in microjansky (MER `flux_vis_psf` is in uJy per the ESA TAP schema)."""
    return 23.9 - 2.5 * math.log10(f_ujy)


def mer_query() -> dict:
    """The stored MER query's text, parsed: flux bounds, row cap, WHERE conditions, sky boxes."""
    if not _Q:
        import re
        src = open(C.R(*QUERY_SCRIPT), encoding="utf-8").read()
        body = re.search(r'query = f"""(.*?)"""', src, re.S).group(1)
        fl = re.search(r"flux_vis_psf BETWEEN ([\d.]+) AND ([\d.]+)", body)
        where = body.split("WHERE", 1)[1].split("AND (", 1)[0]     # the sky boxes follow
        where = re.sub(r"BETWEEN (\S+) AND (\S+)", r"BETWEEN \1 & \2", where)
        conds = [re.sub(r" & ", " AND ", c.strip()) for c in re.split(r"\bAND\b", where)
                 if c.strip()]
        _Q.update(body=body, flux_min=float(fl.group(1)), flux_max=float(fl.group(2)),
                  top=re.search(r"SELECT TOP \{?(\w+)\}?", body).group(1),
                  top_default=int(re.search(r"top_n: int = ([\d_]+)", src).group(1)),
                  conds=conds, n_boxes=body.count("right_ascension BETWEEN"))
    return _Q


def test_faint_fraction() -> float:
    """Fraction of the current pool's shared test negatives fainter than BENCH_IE_LIMIT in PSF
    magnitude, read from the stored query output (one row per object, flux_vis_psf)."""
    import numpy as np
    import pandas as pd
    sp = json.load(open(C.ART["splits"], encoding="utf-8"))
    tn = sp["test_negatives_shared"]
    ids = np.array([int(o) for o in tn["object_ids"]], np.int64)
    if C.POOL != "orig":
        ids = ids[C.pool_keep_neg([f"random:{a}:{b}" for a, b in tn["locations"]])]
    cat = C.mer_flux(usecols=["object_id", "flux_vis_psf"]).set_index("object_id")
    f = cat["flux_vis_psf"].reindex(ids)
    assert not f.isna().any(), "a test negative is missing from the stored query output"
    mags = 23.9 - 2.5 * np.log10(f.to_numpy(float))
    return float((mags > BENCH_IE_LIMIT).mean())


_GX: dict = {}


def gradex() -> dict:
    """The 4,455 hard negatives against the v0.0.1-matched file and catalogue v0.0.3, by exact ID."""
    if not _GX:
        import pandas as pd

        import numpy as np

        def exact(id_str, ra, dec):
            """[OTF_RELEASE_]<tile>_<id> or <tile>_NEG<id>: the last token. The 16 v0.0.3 rows with no subset carry a short
            non-Euclid id_str; their Euclid ID is rebuilt from RA/Dec (the encoding of
            scripts/exact_ids.py: sign(dec) * (floor(RA*1e7)*1e9 + floor(|Dec|*1e7)))."""
            s = str(id_str)
            if "_" in s:                         # also OTF_RELEASE_<tile>_<id>
                t = s.rsplit("_", 1)[1]
                return -int(t[3:]) if t.startswith("NEG") else int(t)
            v = int(np.floor(ra * 1e7 + 1e-6)) * 10**9 + int(np.floor(abs(dec) * 1e7 + 1e-6))
            return -v if dec < 0 else v
        cols = ["id_str", "grade", "right_ascension", "declination"]
        sp = json.load(open(C.ART["splits"], encoding="utf-8"))
        hn = {int(o) for o in sp["source_train"]["neg_mlfp"]["object_ids"]}
        old = C.catalogue("v0.0.1", usecols=cols)
        new = C.catalogue("v0.0.3", usecols=cols)
        g_old = {exact(s, ra, de): g for s, g, ra, de in old[cols].itertuples(index=False)}
        g_new = {exact(s, ra, de): g for s, g, ra, de in new[cols].itertuples(index=False)}
        _GX.update(hn=hn, old_grades={g_old.get(i) for i in hn},
                   now_c={i for i in hn if g_new.get(i) == "C"},
                   new_grades={g_new[i] for i in hn if i in g_new})
    return _GX


def grade_x_reference_n() -> int:
    """The grade-X rows of the v0.0.1 file, as counted by part 1's reference set c."""
    pool = json.load(open(C.R("data", "splits", "dilute10_pool.json"), encoding="utf-8"))
    return int(pool["reference_sets"]["c_grade_X_hard_negatives"]["n"])


# Catalogue values the paper calls outside the physical range (provenance_fixes_brief.md)
DEFL_UNPHYS_MASS = 12.5      # log10 stellar mass above this
DEFL_UNPHYS_MVIS = -26.0     # catalogue `mvis` below this
_DEFL: dict = {}


def deflectors() -> dict:
    """The deflector sample's catalogue columns (no images are read)."""
    if not _DEFL:
        import numpy as np
        d = C.deflector_table()
        _DEFL.update({k: np.asarray(d[k], float) for k in ("stellarmass", "phz_median", "mvis")})
        ids = np.asarray(d["object_id"], np.int64)
        bad = set(ids[(_DEFL["stellarmass"] > DEFL_UNPHYS_MASS)
                      | (_DEFL["mvis"] < DEFL_UNPHYS_MVIS)].tolist())
        # the shards and this file store the same float-rounded IDs, so they match directly
        drawn, lenses = set(), 0
        for _name, s in C.sim_shards(["object_id"]):
            o = np.asarray(s["object_id"], np.int64)
            hit = np.isin(o, list(bad))
            drawn |= set(o[hit].tolist())
            lenses += int(hit.sum())
        _DEFL.update(unphys=bad, unphys_drawn=drawn, unphys_lenses=lenses)
    return _DEFL


_GRID: dict = {}


def grid_points(arm) -> int:
    if arm not in _GRID:
        _GRID[arm] = _grid_points(arm)
    return _GRID[arm]


def _grid_points(arm) -> int:
    """Checkpoints per fold that analyze_window_average's scatter reduction slides over."""
    spec = C.ARMS[arm]
    counts = {len(spec["load"](f)) for f in spec["folds"]}
    assert len(counts) == 1, f"{arm}: folds differ in grid size {counts}"
    return counts.pop()


_D8: dict = {}


def d8_curves() -> dict:
    key = C.POOL
    if key not in _D8:
        _D8[key] = _d8_curves()
    return _D8[key]


def _d8_curves() -> dict:
    """Figure 6's FreeMatch and CORAL fold-mean curves on the current pool, and their crossings.

    Same construction as scripts/plot_completeness_purity.py: the per-fold envelope,
    interpolated onto np.geomspace(1e-3, 1, 600), averaged where every fold is defined.
    """
    import numpy as np
    import plot_completeness_purity as pcp
    grid = np.geomspace(1e-3, 1.0, 600)

    def curve(arm, ro):
        st = []
        for f in C.ARMS[arm]["folds"]:
            comp, pur = pcp.envelope(*C.ARMS[arm]["load"](f)[ro])
            o = np.argsort(pur)
            st.append(np.interp(grid, pur[o], comp[o], left=np.nan, right=np.nan))
        M = np.vstack(st)
        out = np.full(grid.size, np.nan)
        full = ~np.isnan(M).any(axis=0)
        out[full] = M[:, full].mean(axis=0)
        return out
    fm, co = curve("FreeMatch x nb3", 1500), curve("CORAL x nb3", 2000)
    both = ~np.isnan(fm) & ~np.isnan(co)
    g, d = grid[both], (fm - co)[both]
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    crossings = [float(np.sqrt(g[i] * g[i + 1])) for i in idx]
    at24 = float(np.interp(0.24, g, fm[both]) - np.interp(0.24, g, co[both]))
    return {"frac_fm_above": float((d > 0).mean()), "crossings": crossings,
            "fm_minus_coral_at_24": at24}


# ======================================================================================
#  prose_claims -- every condition the prose states in words, asserted on the current pool
# ======================================================================================
CLAIMS: list = []      # (id, statement, holds)


def claim(cid, statement, holds):
    CLAIMS.append((cid, statement, bool(holds)))


def prose_claims() -> list:
    """Evaluate every worded condition of the approved prose on the current pool.

    main() fails the build if any is false, so the text cannot silently become untrue.
    """
    CLAIMS.clear()
    ONE = 1.0 / 98
    v = {n: val for n, val, *_ in MACROS}

    # A1
    claim("A1a", "round-1 CORAL - Baseline A lower bound > 0 and < one positive / 5",
          0 < v["rOneCoralLo"] < ONE / 5)
    claim("A1b", "round-1 CORAL - Baseline A upper bound < both self-training lower bounds",
          v["rOneCoralHi"] < min(v["rOneAdaLo"], v["rOneFmLo"]))
    # A2
    claim("A2", "24% purity reachable on all 5 folds for all four round-1 arms",
          all(v[f"rOne{k}Reach"] == 5 for k in ("Base", "Coral", "Ada", "Fm")))
    # A3 (the phrase "excluding zero only at the margin" rests on A1a)
    # B / J3
    claim("B", "licensed-pair lower bound > one positive", v["lpLo"] > ONE)
    # C1 / C2
    diffs = C.contrast("bnm", 2000, "coral", 2000)["diffs"]
    claim("C1", "BNM - CORAL negative on all four folds", all(d < 0 for d in diffs))
    # C3
    claim("C3a", "BNM - CORAL nearer bound > 3 positives from zero",
          abs(v["ctBnmCoralHi"]) > 3 * ONE)
    wid = {n: C.contrast(a, ra, "lambda0", rb) for n, a, ra, rb in
           (("DANN", "dann", 4000, 4000), ("CORAL", "coral", 2000, 2000),
            ("FreeMatch", "freematch", 1500, 1500))}
    wid = {n: r["ci_hi"] - r["ci_lo"] for n, r in wid.items()}
    claim("C3b", "the two tightest adaptation-vs-control intervals are CORAL's and FreeMatch's",
          set(sorted(wid, key=wid.get)[:2]) == {"CORAL", "FreeMatch"})
    # D1 / D7
    lr = json.load(open(C.art("lines_ratios"), encoding="utf-8"))["arms"]

    def X(arm, ro):
        e = lr[arm]["readouts"][str(ro)]
        if not e["defined_on_all_folds"]:
            return None
        if "folds_1_4" in e:
            f = e["folds_1_4"]
            return (f["pct_of_lines"], 100 * f["ci95"][0] / 0.5, 100 * f["ci95"][1] / 0.5)
        return (e["pct_of_lines"], *e["pct_of_lines_ci95"])
    xs = {"CORAL": X("CORAL x nb3", 2000), "FreeMatch": X("FreeMatch x nb3", 1500),
          "BNM": X("BNM x nb3", 4000), "DANN4000": X("DANN x nb3", 4000),
          "DANN2000": X("DANN x nb3", 2000), "Baseline": X("Baseline A (frozen)", 8000),
          "lambda0": X("lambda=0", 4000)}
    claim("D1a", "CORAL and FreeMatch ratio intervals wider than 50 points",
          all(xs[a] and xs[a][2] - xs[a][1] > 50 for a in ("CORAL", "FreeMatch")))
    claim("D1b", "100% inside CORAL's and FreeMatch's intervals, outside BNM's",
          all(xs[a][1] <= 100 <= xs[a][2] for a in ("CORAL", "FreeMatch"))
          and not (xs["BNM"][1] <= 100 <= xs["BNM"][2]))
    lam = lr["lambda=0"]["readouts"]["4000"]
    claim("D1c", "lambda=0 is the only n/a arm, with exactly two unreachable folds",
          [a for a, x in xs.items() if x is None] == ["lambda0"]
          and len(lam["undefined_folds"]) == 2)
    # D2
    defined = {a: x for a, x in xs.items() if x}
    claim("D2a", "DANN@4000's ratio interval is the widest among defined ratios",
          max(defined, key=lambda a: defined[a][2] - defined[a][1]) == "DANN4000")
    claim("D2b", "DANN@2000 is the highest ratio point estimate",
          max(defined, key=lambda a: defined[a][0]) == "DANN2000")
    # D3 / D5
    und = {a: lr[k]["readouts"][str(ro)]["undefined_folds"] for a, k, ro in
           (("CORAL", "CORAL x nb3", 2000), ("FreeMatch", "FreeMatch x nb3", 1500),
            ("BNM", "BNM x nb3", 4000), ("DANN4000", "DANN x nb3", 4000),
            ("lambda0", "lambda=0", 4000), ("Baseline", "Baseline A (frozen)", 8000))}
    claim("D3a", "unreachable folds: two for lambda=0, none for any other arm",
          len(und["lambda0"]) == 2 and all(not u for a, u in und.items() if a != "lambda0"))
    comp = {a: v[f"lr{k}Comp"] for a, k in (("CORAL", "Coral"), ("FreeMatch", "FreeMatch"),
                                             ("DANN4000", "Dann"), ("BNM", "Bnm"))}
    base_comp = lr["Baseline A (frozen)"]["readouts"]["8000"].get("folds_1_4", {}).get("fold_mean", 0)
    claim("D3b", "no declared-readout fold-mean completeness reaches 0.50",
          all(c < 0.50 for c in comp.values()) and base_comp < 0.50)
    claim("D3c", "0 < 0.50 - CORAL < two positives on one fold (2 x 0.0102 / 4)",
          0 < 0.50 - comp["CORAL"] < 2 * ONE / 4)
    claim("D3d", "order CORAL > FreeMatch > DANN@4000 > BNM",
          comp["CORAL"] > comp["FreeMatch"] > comp["DANN4000"] > comp["BNM"])
    claim("D3e", "DANN@2000 fold-mean completeness > 0.50", v["lrDannSensComp"] > 0.50)
    # D4
    pf = {a: lr[k]["readouts"][str(ro)]["comp_at_purity_per_fold"] for a, k, ro in
          (("CORAL", "CORAL x nb3", 2000), ("FreeMatch", "FreeMatch x nb3", 1500),
           ("DANN4000", "DANN x nb3", 4000), ("DANN2000", "DANN x nb3", 2000),
           ("BNM", "BNM x nb3", 4000))}
    above = {a: sum(1 for x in f.values() if x is not None and x > 0.50) for a, f in pf.items()}
    claim("D4", "CORAL, FreeMatch and DANN (either readout) each exceed 0.50 on >= 1 fold; BNM on none",
          above["CORAL"] >= 1 and above["FreeMatch"] >= 1
          and (above["DANN4000"] + above["DANN2000"]) >= 1 and above["BNM"] == 0)
    # D6
    claim("D6", "Baseline A's ratio is defined", math.isfinite(v["lrBaselineX"]))
    # E1
    lg = {k: v[f"wa{k}Logit"] for k in ("FreeMatch", "Coral", "Dann", "DannSens")}
    rk = {k: v[f"wa{k}Rank"] for k in ("FreeMatch", "Coral", "Dann", "DannSens")}
    claim("E1a", "mean_logit: FreeMatch >= DANN >= CORAL at both DANN read-offs",
          all(lg["FreeMatch"] >= lg[d] >= lg["Coral"] for d in ("Dann", "DannSens")))
    claim("E1b", "mean_rank: FreeMatch >= CORAL >= DANN at both DANN read-offs",
          all(rk["FreeMatch"] >= rk["Coral"] >= rk[d] for d in ("Dann", "DannSens")))
    claim("E1c", "mean_logit DANN - CORAL max gap < one positive", v["waLogitDannCoralMaxGap"] < ONE)
    # F1: the ratios against the second difference use the same three arms, same order
    claim("F1", "F1 ratios are CORAL, FreeMatch, DANN (the order of the first set)",
          v["wzCoralMultOne"] <= v["wzFreeMatchMultOne"] <= v["wzDannMultOne"]
          and v["wzCoralMultFm"] <= v["wzFreeMatchMultFm"] <= v["wzDannMultFm"])
    # F2 / F3 / F4 / F5 / F8
    below = [k for k in ("Coral", "FreeMatch", "Dann", "Lambda", "Bnm")
             if v[f"wz{k}MultEight"] < 1.0]
    claim("F2a", "exactly one arm below 1x at k = 8, and it is CORAL", below == ["Coral"])
    claim("F2b", "CORAL's k = 8 shortfall < one positive on one fold (0.0102 / 4)",
          0 < v["wzCoralEightGap"] < ONE / 4)
    # F6
    ach = {k: v[f"wa{k}Achieved"] for k in ("Lambda", "FreeMatch", "Dann", "Bnm")}
    claim("F6a", "DANN and BNM exceed sqrt(k); FreeMatch and lambda=0 do not",
          ach["Dann"] > 1 and ach["Bnm"] > 1 and ach["FreeMatch"] <= 1 and ach["Lambda"] <= 1)
    claim("F6b", "DANN and BNM are the two highest achieved reductions",
          set(sorted(ach, key=ach.get)[-2:]) == {"Dann", "Bnm"})
    # F7
    t1 = [v["fmCoral"], v["fmFreeMatch"], v["fmDann"]]
    gap = max(t1) - min(t1)
    k1 = [v[f"wz{k}SdOne"] for k in ("Coral", "FreeMatch", "Dann", "Lambda", "Bnm")]
    k8 = [v[f"wz{k}SdEight"] for k in ("Coral", "FreeMatch", "Dann", "Lambda", "Bnm")]
    claim("F7a", "largest T1 gap among CORAL/FreeMatch/DANN < the minimum k = 1 scatter",
          gap < min(k1))
    claim("F7b", "exactly four arms' k = 8 scatter exceeds that gap",
          sum(1 for s in k8 if s > gap) == 4)
    # G
    claim("G", "|DANN - FreeMatch spread difference| > |point difference|",
          abs(v["sdDann"] - v["sdFreeMatch"]) > abs(v["fmDann"] - v["fmFreeMatch"]))
    # H
    ap = json.load(open(C.art("lever1_aug_precision"), encoding="utf-8"))
    base0 = float(C.suite("baseline", 0, 8000)["tpr"])
    ap_orig = json.load(open(C.ART["lever1_aug_precision"], encoding="utf-8"))
    claim("Ha", "Baseline A fold 0 - cell A = exactly one positive (1/98)",
          abs((base0 - ap["cells"]["A_none_fp32"]["tpr"]) - ONE) < 1e-12)
    claim("Hb", "cell A = Baseline A fold 0 on the pre-registered pool",
          abs(ap_orig["cells"]["A_none_fp32"]["tpr"] - v["baseFoldZeroPrereg"]) < 1e-12)
    # I
    claim("I", "|lambda=0 - Baseline A| < 0.0001", abs(v["ctLamBase"]) < 0.0001)
    # D8 (final wording pass)
    d8 = d8_curves()
    claim("D8a", "FreeMatch above CORAL on > 50% of Figure 6's purity grid",
          d8["frac_fm_above"] > 0.50)
    claim("D8b", "CORAL above FreeMatch at 24% purity (tabulated fold-means and drawn curves)",
          v["lrCoralComp"] > v["lrFreeMatchComp"] and d8["fm_minus_coral_at_24"] < 0)
    claim("D8c", "at least 3 crossings, at least two of them between purity 0.15 and 0.30",
          len(d8["crossings"]) >= 3
          and sum(1 for c in d8["crossings"] if 0.15 <= c <= 0.30) >= 2)
    # Section 7.3 continuation (final wording pass)
    claim("F6c", "DANN's and BNM's achieved reductions exceed sqrt(k) (ratio > 1)",
          v["waDannAchieved"] > 1 and v["waBnmAchieved"] > 1)
    claim("F6d", "DANN's and BNM's scatter grids have the same size (the text names one size)",
          v["waDannGridPoints"] == v["waBnmGridPoints"])
    # Figure 6 caption (credits_provenance_brief.md): the fold lines drawn above the star are
    # exactly the four the caption names. Read at each arm's quoted readout -- the one the
    # figure plots, DANN at 4000 -- from the same per-fold values Table 5 prints.
    from write_lines_ratios import QUOTED
    named = {("DANN x nb3", "2"), ("CORAL x nb3", "2"), ("FreeMatch x nb3", "2"),
             ("FreeMatch x nb3", "4")}
    drawn = {(arm, f) for arm, ro in QUOTED.items()
             for f, x in lr[arm]["readouts"][str(ro)]["comp_at_purity_per_fold"].items()
             if x is not None and x > C.LINES_COMPLETENESS}
    claim("F6e", "F6 fold lines above the star = DANN f2, CORAL f2, FreeMatch f2 and f4",
          QUOTED["DANN x nb3"] == 4000 and drawn == named)
    # Deflector sample (provenance-fixes pass): the selection the text states holds for every
    # deflector, at the printed precision, and the sample is the 19,167 the text names
    dd = deflectors()
    claim("DFa", "deflector sample holds the 19,167 galaxies Section 3 names",
          dd["stellarmass"].size == 19167)
    claim("DFb", "every deflector has log M* >= the printed floor, and the floor is the minimum to 0.1 dex",
          float(render(v["DeflMassFloor"], 1, False)) <= dd["stellarmass"].min()
          < float(render(v["DeflMassFloor"], 1, False)) + 0.1)
    claim("DFc", "every deflector's photometric redshift lies in the printed range",
          float(render(v["DeflZmin"], 2, False)) - 1e-6 <= dd["phz_median"].min()
          and dd["phz_median"].max() <= float(render(v["DeflZmax"], 2, False)) + 1e-6)
    claim("DFd", "the unphysical-value count is non-zero (the Section 9 clause is kept)",
          v["DeflUnphysN"] > 0)
    claim("DFe", "17 unphysical-value deflectors were drawn, hosting 132 simulated lenses",
          v["DeflUnphysDrawn"] == 17 and v["DeflUnphysLenses"] == 132
          and dd["unphys_drawn"] <= dd["unphys"])
    # grade-X hard negatives (gradex_brief.md)
    gx = gradex()
    claim("GXa", "all 4,455 hard negatives are grade X in the version-0.0.1-matched file",
          len(gx["hn"]) == 4455 and gx["old_grades"] == {"X"})
    claim("GXb", "exactly 8 of them are graded C in version 0.0.3; none is graded A or B there",
          v["GradeXnowC"] == 8 and gx["new_grades"] == {"C"})
    # screening cut (screening_cut_brief.md): the macros are the query's constants, and the
    # conditions Section 3 states in words are exactly the query's
    q = mer_query()
    claim("SCa", "flux macros print the query's constants (2.3 and 229 uJy)",
          render(v["FluxMin"], 1, False) == "2.3" and render(v["FluxMax"], 0, False) == "229"
          and q["flux_min"] == 2.3 and q["flux_max"] == 229.0)
    claim("SCb", "PSF magnitudes are 23.9 - 2.5 log10(f) of those constants (23.0, 18.0)",
          render(v["MagFaint"], 1, False) == "23.0" and render(v["MagBright"], 1, False) == "18.0")
    claim("SCc", "the query's other conditions are exactly the seven Section 3 lists, in four sky boxes, capped at 250,000 rows",
          q["conds"][:1] == ["vis_det = 1"] and sorted(q["conds"][2:]) == sorted(QUERY_OTHER[1:])
          and q["conds"][1].startswith("flux_vis_psf BETWEEN") and len(q["conds"]) == 8
          and q["n_boxes"] == 4 and q["top"] == "top_n" and q["top_default"] == 250000
          and "ORDER BY" not in q["body"])
    claim("SCd", "the faint fraction is between 0 and 1 and the faint limit is fainter than 22.5",
          0 < v["FracFaintTest"] < 100 and ab_mag(q["flux_min"]) > BENCH_IE_LIMIT)
    # Section 3: the PSF the simulator convolves with, read from its source (not imported)
    src = open(C.R("scripts", "generate_arc_on_real.py"), encoding="utf-8").read()
    import re
    claim("PSF", "simulator PSF is a Gaussian of FWHM 0.16 arcsec, as Section 3 states",
          re.search(r"^EUCLID_PSF_FWHM\s*=\s*0\.16\b", src, re.M) is not None
          and 'PSF(psf_type="GAUSSIAN", fwhm=EUCLID_PSF_FWHM' in src
          and src.count("PSF(") == 1)
    rewrite_v1_claims(v, ONE)
    rewrite_v2_claims()
    rewrite_v3_claims()
    rewrite_v4_claims()
    rewrite_v5_claims()
    return CLAIMS


def rewrite_v1_claims(v, ONE):
    """The worded claims added by rewrite v1 (brief_rewrite_v1.md and its addenda): every
    sentence the brief marks [assert], and the other new wording claims. Values come from
    _common on the current pool, from the review-round JSONs (make_review_macros, whose
    recorded input hashes are re-checked), and from the display macros (make_m_display)."""
    import make_review_macros as RV
    import make_m_display as MD
    rv = {n: val for n, _p, val, _s in RV.build()}
    dv = {n: val for n, _p, val, _s in MD.build()}
    ck = RV.checks()
    ct = {i: C.contrast(a, ra, b, rb) for i, (_p, _l, a, ra, b, rb, _m, _k)
          in enumerate(C.CONTRAST_ROWS)}
    # Contributions bullet 1 [assert] and the abstract: the four target-signal rows
    claim("RW1", "rows 1, 4, 5, 6 (each DA arm against lambda=0): all four intervals exclude zero",
          all(ct[i]["excludes_zero"] and ct[i]["mean"] > 0 for i in (1, 4, 5, 6)))
    claim("RW2", "abstract: rows 1, 2, 3, 4, 5, 6 exclude zero with all four folds agreeing in sign",
          all(ct[i]["excludes_zero"] and ct[i]["n_same_sign"] == 4 for i in (1, 2, 3, 4, 5, 6)))
    # Appendix C, the independence paragraph and Figure 4 [assert] (rewrite v5: moved there from
    # contributions bullet 2 and Section 6.4)
    claim("RW3", "every three-arm gap's ratio_b (noise_b / |gap|) is > 1, declared readouts only",
          all(r > 1 for r in ck["three_ratio_b"].values()) and len(ck["three_ratio_b"]) == 4)
    claim("RW4", "every BNM gap's ratio_b is < 1, declared readouts only",
          all(r < 1 for r in ck["bnm_ratio_b"].values()) and len(ck["bnm_ratio_b"]) == 6)
    claim("RW5", "Appendix C: the largest three-arm gap is below the smallest three-arm noise_b",
          max(ck["three_gaps"].values()) < min(ck["three_noise"].values()))
    claim("RW6", "Section 6.3 and Appendix C: BNM - CORAL is the one objective-vs-objective row whose interval "
          "excludes zero (rows 7-12)",
          ct[7]["excludes_zero"] and not any(ct[i]["excludes_zero"] for i in (8, 9, 10, 11, 12)))
    claim("RW7", "Section 6.4: lag-1 autocorrelations (folds 1-4 means) all within +-0.2 "
          "('close to uncorrelated')", -0.2 < rv["rvLagMin"] and rv["rvLagMax"] < 0.2)
    # Contributions bullet 3 [assert] / Section 6.6
    claim("RW8", "grade A/B recall ratio is between 1.9 and 3.3 for every readout ('two to three times')",
          all(1.9 <= r <= 3.3 for r in ck["grade_ab_ratio"].values()))
    claim("RW9", "Section 6.6: recall rises with expert score (three bins) in every readout",
          all(ck["expert_score_rises"].values()))
    claim("RW10", "Section 6.6: for all four DA headlines the smallest Einstein-radius bin has "
          "the lowest recall", all(ck["theta_small_lowest_da"].values()))
    claim("RW11", "Section 6.6: the lenses every model misses are mostly grade B",
          rv["rvMissedAllB"] > rv["rvMissedAllN"] / 2)
    xm = RV.review(RV.R1, "A2_crossmatch.json")["objects_matching_a_grade_A_B_C_X_entry"]
    claim("RW12", "Section 6.6: every catalogue match among the top negatives is grade C",
          all(h["grade"] == "C" for hits in xm.values() for h in hits) and len(xm) == rv["rvTopNegC"])
    claim("RW13", "Section 6.6: the above-threshold negatives are brighter than the pool",
          rv["rvAboveFaintPct"] < rv["rvPoolFaintPct"])
    # Section 6.6 / Appendix H [assert]
    claim("RW14", "the bright-only pool leaves all eight contrasts' zero-inclusion unchanged",
          all(ck["l2_zero_inclusion_same"]) and len(ck["l2_zero_inclusion_same"]) == 8)
    l2 = RV.review(RV.R1B, "L2_bright_pool.json")["readouts"]
    drops = {k: d["lines_ratio_bright"] - d["lines_ratio_D1"] for k, d in l2.items()
             if d["lines_ratio_bright"] is not None and d["lines_ratio_D1"] is not None}
    claim("RW15", "Appendix H: TPR falls on the bright pool for every readout, and BNM's benchmark "
          "ratio falls the most", all(d["tpr_change_mean"] < 0 for d in l2.values())
          and min(drops, key=drops.get) == "bnm@4000")
    # Section 5.5 [assert]
    with pool("orig"):
        ct_o = {i: C.contrast(a, ra, b, rb) for i, (_p, _l, a, ra, b, rb, _m, _k)
                in enumerate(C.CONTRAST_ROWS)}
    claim("RW16", "Section 5.5: zero-inclusion is the same on the original and cleaned pools for "
          "every contrast", all(ct_o[i]["excludes_zero"] == ct[i]["excludes_zero"] for i in ct))
    claim("RW17", "Section 5.5: the change of pool raises every arm's TPR, and the DA arms the most",
          dv["dPoolUpMin"] > 0 and DA_RAISE_MOST())
    claim("RW18", "Section 5.5: of the 20 D2 lenses, 14 were training negatives and 6 validation "
          "negatives ('Fourteen' / 'six'); 19 random galaxies",
          rv["rvDTwoTrainNeg"] == 14 and rv["rvDTwoValNeg"] == 6 and rv["rvDTwoRandom"] == 19)
    claim("RW19", "Section 5.5: on the original pool the DA arms place more of the 28 above "
          "threshold than Baseline A and lambda=0", rv["rvFsevenDaMin"] > rv["rvFsevenRefMax"])
    # Abstract / Section 6.2 / conclusion
    claim("RW20", "Baseline A and lambda=0 print the same two-decimal TPR (one value quoted for both)",
          f"{dv['dFmBaseline']:.2f}" == f"{dv['dFmLambda']:.2f}")
    claim("RW21", "Section 6.3: DANN - lambda=0 has the widest interval of rows 1-6",
          max((ct[i]["ci_hi"] - ct[i]["ci_lo"], i) for i in range(1, 7))[1] == 1)
    claim("RW22", "Section 6.3: lambda=0 - Baseline A admits zero", not ct[0]["excludes_zero"])
    claim("RW23", "Section 6.3 / Figure: lambda=0 ends the development fold below its start",
          dv["dFzLambdaLast"] < dv["dFzInit"])
    # Section 6.1
    fz = {k: dv[k] for k in ("dFrozenBase", "dFrozenAda", "dFrozenFm")}
    claim("RW24", "Section 6.1: frozen self-training arms reach about twice Baseline A (1.8-2.4x)",
          all(1.8 <= fz[k] / fz["dFrozenBase"] <= 2.4 for k in ("dFrozenAda", "dFrozenFm")))
    claim("RW25", "Section 6.1: frozen CORAL's difference excludes zero only at the margin "
          "(0 < lower bound < one positive)", 0 < dv["dFrozenCoralLo"] < ONE)
    claim("RW26", "Section 6.1: the frozen-to-unfrozen FreeMatch interval excludes zero",
          dv["dLpLo"] > 0)
    probe = json.load(open(C.ART["r1_dann_probe"], encoding="utf-8"))
    with pool("orig"):
        b0 = C.suite("baseline", 0, 8000)["tpr"]
    claim("RW28", "Section 6.1: the DANN development run's last checkpoint equals Baseline A's "
          "fold-0 TPR on the original pool", probe["candidates"][-1]["fold0_tpr_1e-3"] == b0)
    # Section 6.5
    claim("RW27", "Section 6.5: DANN's sensitivity-readout ratio exceeds 100%, no declared DA "
          "headline does", dv["dLrDannSens"] > 100 and dv["dLrDaMax"] < 100)


def rewrite_v2_claims():
    """The literal numbers and worded procedural facts rewrite v2 adds (brief_rewrite_v2.md
    sections 1.1-1.4 with Addenda 1, 3 and 4), each against its stored source."""
    import glob
    import re as _re

    def cfg(*p):
        return json.load(open(C.R(*p), encoding="utf-8"))
    # The lab log's part of each claim: record_extract.prose() tests the phrases on the log, or
    # without the log returns the verdict recorded at extraction (scripts/record_extract.py).
    import record_extract as RX
    # Section 3.1
    claim("RW2-1", "Section 3.1: 30 000 lenses generated; 24 000 training lenses",
          RX.prose("RW2-1") and
          RV_F5()["populations"]["a"].startswith("24000 source-train simulated positives"))
    # Section 4.5, the budgets as stated, from the run configs and budget files
    dann = [cfg(*os.path.relpath(p, C.REPO).split(os.sep)) for p in
            glob.glob(C.R("runs", "dann_main", "fold*", "extracted", "fold*", "run_config.json"))]
    lam = [cfg(*os.path.relpath(p, C.REPO).split(os.sep)) for p in
           glob.glob(C.R("runs", "baseline_unfrozen", "fold*", "extracted", "lambda0_r6_fold*",
                         "run_config.json"))]
    claim("RW2-2", "Appendix B (budget rule): DANN read at 4000, twice its lambda horizon of 2000 "
          "(lambda = tanh(5s/2000))", len(dann) == 4 and all(c["max_steps"] == 4000 for c in dann)
          and RX.prose("RW2-2"))
    claim("RW2-3", "Appendix B (budget rule): the lambda=0 control is read at 4000",
          len(lam) == 4 and all(c["max_steps"] == 4000 for c in lam))
    bnm = cfg("runs", "bnm_r6_budget.json")
    claim("RW2-4", "Appendix B (budget rule): BNM's budget is 1000, twice its crossing step",
          bnm["selected_budget_steps"] == 1000 == 2 * bnm["crossing_step"])
    cb = cfg("runs", "coral_r4_budget.json")
    coral = [cfg(*os.path.relpath(p, C.REPO).split(os.sep)) for p in
             glob.glob(C.R("kaggle", "coral_r4", "runs", "main", "extracted", "coral_r4_confirm",
                           "fold*", "run.json"))]
    claim("RW2-5", "Appendix B (budget rule): CORAL declared 2000 on cost, declining the development maximum "
          "at 2500", cb["derived_or_declared"] == "declared"
          and cb["resolvability_evidence"]["task2_pairwise"]["argmax_step"] == 2500
          and len(coral) == 4 and all(c["max_steps"] == 2000 for c in coral))
    claim("RW2-6", "Appendix B (budget rule): Baseline A uses 8000", cfg("runs", "bakeoff_baseline_A",
                                                         "run_config.json")["budget_steps"] == 8000)
    fm = [cfg("runs", f"freematch_nb3_unfrozen_fold{f}", "run_config.json") for f in (1, 2, 3, 4)]
    claim("RW2-7", "Appendix B (budget rule and amendments): FreeMatch first declared 2000, amended to 1500 where "
          "the development curve peaks; every confirmatory fold ran at 1500",
          RX.prose("RW2-7")
          and all(c["max_steps"] == 1500 for c in fm))
    claim("RW2-8", "Section 4.3: FreeMatch's local-threshold variant chosen by a fold-0 TPR screen "
          "(standard 0.357 against floor 0.174)",
          RX.prose("RW2-8")
          and all(c["local_threshold_mode"] == "standard" for c in fm))


def rewrite_v3_claims():
    """The procedural facts rewrite v3 states (brief_rewrite_v3.md section 1), against the
    record and the run configs."""
    import glob
    import re as _re
    import record_extract as RX          # the lab log's part of RW3-2..4 (see rewrite_v2_claims)
    pats = {"coral": ("kaggle", "coral_r4", "runs", "main", "extracted", "coral_r4_confirm",
                      "fold*", "run.json"),
            "freematch": ("runs", "freematch_nb3_unfrozen_fold[1-4]", "run_config.json"),
            "dann": ("runs", "dann_main", "fold*", "extracted", "fold*", "run_config.json"),
            "bnm": ("runs", "bnm_r6_folds", "bnm_r6_fold*", "run_config.json"),
            "lambda0": ("runs", "baseline_unfrozen", "fold*", "extracted", "lambda0_r6_fold*",
                        "run_config.json")}
    runs = {a: [json.load(open(p, encoding="utf-8")) for p in glob.glob(C.R(*q))]
            for a, q in pats.items()}
    base = json.load(open(C.R("runs", "bakeoff_baseline_A", "run_config.json"), encoding="utf-8"))
    claim("RW3-1", "Section 8: every reported result is a single run per fold with one seed "
          "(one config per fold 1-4 for each unfrozen arm; seed 42 everywhere)",
          all(len(v) == 4 and sorted(c["fold"] for c in v) == [1, 2, 3, 4] for v in runs.values())
          and all(c["seed"] == 42 for v in runs.values() for c in v) and base["seed"] == 42)
    claim("RW3-2", "Section 4.3 / Table 10: BNM's weight from a fold-0 sweep over a grid set by "
          "step-0 gradient norms, read the nuclear norm and the predicted-positive fraction, "
          "never TPR", RX.prose("RW3-2"))
    claim("RW3-3", "Appendix J: GalSim unpinned in the WSL2 conda environment used for simulation",
          RX.prose("RW3-3"))
    claim("RW3-4", "Section 4.5 and Appendix B: Baseline A's 8000 fixed before any results; BNM's budget rule "
          "declared before its development curve",
          RX.prose("RW3-4"))
    d1 = [json.load(open(p, encoding="utf-8"))["folds"]
          for p in glob.glob(os.path.join(C.DATA, "round1_paired_*_D1.json"))]
    claim("RW3-5", "Appendix D (tab:frozen): frozen-comparison means and paired differences use folds 0-4",
          len(d1) == 3 and all(f == [0, 1, 2, 3, 4] for f in d1))


def rewrite_v5_claims():
    """The new sentences of rewrite v5 (brief_rewrite_v5.md B1-B8, C4, D2; addendum 1), against
    the sampler, the run configs, the stored scores and the window artifacts."""
    import glob
    import hashlib
    import numpy as np
    import samplers as SM
    import make_review_macros as RV

    def runcfgs(*pat):
        fs = sorted(glob.glob(C.R(*pat)))
        assert len(fs) == 4, (pat, fs)
        return [json.load(open(p, encoding="utf-8")) for p in fs]

    arms = {
        "coral": runcfgs("kaggle", "coral_r4", "runs", "main", "extracted", "coral_r4_confirm",
                         "fold*", "run.json"),
        "dann": runcfgs("runs", "dann_main", "fold*", "extracted", "fold*", "run_config.json"),
        "lambda0": runcfgs("runs", "baseline_unfrozen", "fold*", "extracted", "lambda0_r6_fold*",
                           "run_config.json"),
        "bnm": runcfgs("runs", "bnm_r6_folds", "bnm_r6_fold*", "run_config.json"),
        "freematch": runcfgs("runs", "freematch_nb3_unfrozen_fold[1-4]", "run_config.json"),
    }
    # B1 / B2 / abstract: every target batch is exactly half lenses. The sampler takes a fixed
    # count per class; one batch drawn from small pools must be exactly 128/128, and every
    # adaptation arm ran at batch 256, pos_fraction 0.5, no gradient accumulation, through
    # the latent-class target sampler.
    smp = SM.ClassBalancedBatchSampler(np.arange(0, 40), np.arange(100, 160), 256,
                                       num_batches=3, pos_fraction=0.5, seed=0)
    batches = list(iter(smp))
    exact = all(sum(i < 100 for i in b) == 128 and sum(i >= 100 for i in b) == 128
                for b in batches)
    src = {n: open(C.R("scripts", n), encoding="utf-8").read()
           for n in ("run_coral_round4.py", "run_freematch.py", "run_dann_r5.py", "run_bnm_r6.py")}
    latent = ("tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True"
              in src["run_coral_round4.py"] and
              "tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True"
              in src["run_freematch.py"] and
              all('tgt_sampler = ClassBalancedBatchSampler(holdout["train_pos"], '
                  'holdout["train_neg"],' in src[n] for n in ("run_dann_r5.py", "run_bnm_r6.py")))
    cfg_ok = all(c["batch_size"] == 256 and c["pos_fraction"] == 0.5
                 and c["gradient_accumulation"] is False
                 for k in ("coral", "dann", "bnm", "freematch") for c in arms[k])
    claim("RW5-1", "Sections 4.4 and 8, abstract: every target batch of every adaptation arm is "
          "exactly half lenses and half galaxies (128/128)",
          smp.n_pos == 128 == smp.n_neg and exact and latent and cfg_ok)
    dl = open(C.R("scripts", "da_losses.py"), encoding="utf-8").read()
    claim("RW5-2", "Section 8: FreeMatch's fairness term targets a uniform class marginal (no "
          "fairness_target set, so the module default 1/C per class applies)",
          all(c.get("fairness_target") is None for c in arms["freematch"])
          and "if fairness_target is None:\n            ft = torch.full((self.num_classes,), init)"
          in dl)
    # B3: the heads each unfrozen arm starts from, and the FreeMatch decomposition.
    heads = {k: [c["init_head_from"].split("/runs/", 1)[1] for c in v] for k, v in arms.items()}
    starts_ok = (heads["freematch"] == [f"bakeoff_freematch_5fold/fold{f}/checkpoints/step_8000.pt"
                                        for f in (1, 2, 3, 4)]
                 and heads["coral"] == [f"bakeoff_coral/fold{f}/checkpoints/step_8000.pt"
                                        for f in (1, 2, 3, 4)]
                 and all(set(heads[k]) == {"bakeoff_baseline_A/checkpoints/step_8000.pt"}
                         for k in ("dann", "bnm", "lambda0")))
    claim("RW5-3", "Section 6.3: FreeMatch and CORAL start from their frozen-comparison heads; "
          "DANN, BNM and the control start from the same (Baseline A) head", starts_ok)
    with pool("D1"):
        def r1(run):
            s = C.round1_summary(run, folds=(1, 2, 3, 4))
            return float(np.mean([s["per_fold"][f]["tpr"] for f in (1, 2, 3, 4)]))
        fm_s, co_s = r1("bakeoff_freematch_5fold"), r1("bakeoff_coral")
        u = C.fold_mean("freematch", "tpr", 1500)
        lam = C.fold_mean("lambda0", "tpr", 4000)
        u = float(u[0] if isinstance(u, tuple) else u)
        lam = float(lam[0] if isinstance(lam, tuple) else lam)
    fb = json.load(open(os.path.join(C.DATA, "foldmean_bootstrap_D1.json"), encoding="utf-8"))
    claim("RW5-4", "Section 6.3: FreeMatch - lambda=0 splits into (start - lambda=0) gained frozen "
          "plus (unfrozen - start) gained after unfreezing, the licensed pair's difference",
          abs((u - lam) - ((fm_s - lam) + (u - fm_s))) < 1e-12
          and abs((u - fm_s) - fb["diff"]) < 1e-9 and abs(fm_s - fb["round1_fold_mean"]) < 1e-9)
    claim("RW5-5", "Section 6.3: CORAL's frozen head starts a little above the control "
          "(above it, by less than five positives in 98)", 0 < co_s - lam < 5 / 98)
    # B4: BNM - CORAL was not pre-registered, and \contrastCount counts the main table's intervals.
    cc = int(_printed_macro("contrastCount"))
    claim("RW5-6", "Section 6.3: BNM - CORAL (row 7) was not pre-registered; it is one of "
          "\\contrastCount intervals, the main contrast table's rows",
          C.CONTRAST_PREREG[7] == C.NEITHER and cc == len(C.CONTRAST_ROWS))
    # B7: DANN's gain, and the other three objectives' declared contrasts against the control.
    ct = {i: C.contrast(a, ra, b, rb) for i, (_p, _l, a, ra, b, rb, _m, _k)
          in enumerate(C.CONTRAST_ROWS)}
    others = {4: ct[4]["mean"], 5: ct[5]["mean"], 6: ct[6]["mean"]}
    claim("RW5-7", "Contributions bullet 1: the other three objectives' declared contrasts against "
          "the control run from BNM's (min) to CORAL's at 2000/4000 (max)",
          min(others, key=others.get) == 4 and max(others, key=others.get) == 5
          and C.CONTRAST_ROWS[4][2] == "bnm" and C.CONTRAST_ROWS[5][2] == "coral"
          and C.CONTRAST_ROWS[6][2] == "freematch")
    # Addendum 1: the ranking sentences (Section 6.4, Appendix C, contributions bullet 2,
    # the abstract and the conclusion).
    pairs = [("dann", 4000, "coral", 2000), ("dann", 2000, "coral", 2000),
             ("dann", 4000, "freematch", 1500), ("dann", 2000, "freematch", 1500),
             ("coral", 2000, "freematch", 1500)]
    with pool("D1"):
        k1 = [C.contrast(*p) for p in pairs]
    claim("RW5-8", "Section 6.4 and abstract: at each arm's declared read-off every paired "
          "interval among CORAL, DANN and FreeMatch includes zero",
          all(not r["excludes_zero"] for r in k1))
    wraw = open(os.path.join(C.DATA, "window_average_D1.json"), "rb").read()
    w = json.loads(wraw)
    cells = {"fm": ("FreeMatch x nb3", "1500"), "d4": ("DANN x nb3", "4000"),
             "d2": ("DANN x nb3", "2000"), "co": ("CORAL x nb3", "2000")}
    top = all(max(cells, key=lambda n: w["arms"][cells[n][0]]["readouts"][cells[n][1]][comb]["8"]
                  ["tpr_mean"]) == "fm" for comb in ("mean_logit", "mean_rank"))
    claim("RW5-9", "Section 6.4, contributions bullet 2: at k=8 FreeMatch has the highest fold-mean "
          "under both pre-declared score combinations",
          top and w["k_primary"] == 8 and w["test_pool"]["pool"] == "D1")
    a3 = json.load(open(C.R("audit", "rewrite_v5", "out", "A3_window_pairs.json"), encoding="utf-8"))
    fmr = [r for r in a3["rows"] if "FreeMatch" in r["contrast"] and r["k"] == 8]
    claim("RW5-10", "Section 6.4, abstract, conclusion: FreeMatch's k=8 lead over DANN (both "
          "read-offs) and CORAL excludes zero under mean rank, and under mean logit it does not",
          a3["source_sha256"] == hashlib.sha256(wraw).hexdigest() and len(fmr) == 6
          and all(r["ci"][1] < 0 for r in fmr if r["combination"] == "mean_rank")
          and all(r["includes_zero"] for r in fmr if r["combination"] == "mean_logit"))
    prereg = json.load(open(C.R("runs", "round7", "window_prereg.json"), encoding="utf-8"))
    claim("RW5-11", "Section 6.4: the window average is the pre-declared secondary analysis, "
          "with mean logit primary and mean rank secondary",
          prereg["status"].startswith("DECLARED SECONDARY ANALYSIS")
          and prereg["score_combination"]["primary"]["name"] == "mean_logit"
          and prereg["score_combination"]["secondary"]["name"] == "mean_rank"
          and prereg["declared_before_any_windowed_number_was_computed"] is True)
    _a1, three, _bnm, nb = RV.a1_sets()
    exceed = []
    for k, g in three.items():
        a, b = k.split("|")[0].split("-")
        ra, rb = g["steps"]
        with pool("D1"):
            se = C.contrast(a, ra, b, rb)["se"]
        exceed.append(nb(g) > se)
    claim("RW5-12", "Section 6.4: for at least one pair among CORAL, DANN and FreeMatch the "
          "independent noise figure exceeds the standard error the paired interval implies",
          any(exceed))
    with pool("D1"):
        npos = [C.suite("freematch", f, 1500)["n_pos"] for f in (1, 2, 3, 4)]
    claim("RW5-13", "Abstract: about 98 test lenses per confirmatory fold",
          round(float(np.mean(npos))) == 98)


def _printed_macro(name):
    import glob
    import re as _re
    for p in glob.glob(os.path.join(C.TABLES, "M_*.tex")):
        m = _re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", open(p, encoding="utf-8").read())
        if m:
            return m.group(1)
    raise AssertionError(f"\\{name} is not defined in tables/M_*.tex")


def rewrite_v4_claims():
    """The procedural facts rewrite v4 states (brief_rewrite_v4.md), against the configs, the
    split builder and the record."""
    import glob
    import re as _re
    dann = [json.load(open(p, encoding="utf-8")) for p in
            glob.glob(C.R("runs", "dann_main", "fold*", "extracted", "fold*", "run_config.json"))]
    lam = [json.load(open(p, encoding="utf-8")) for p in
           glob.glob(C.R("runs", "baseline_unfrozen", "fold*", "extracted", "lambda0_r6_fold*",
                         "run_config.json"))]
    claim("RW4-1", "Section 4.3: DANN's folds ran with abort conditions armed, the control's only "
          "logged them; no arming condition fired on DANN; every DANN fold ran 4000 steps",
          len(dann) == 4 == len(lam)
          and all(c["abort_mode"] == "armed" and c["abort_watch"]["fired_arming"] == []
                  and c["steps_completed"] == 4000 for c in dann)
          and all(c["abort_mode"] == "log_only" for c in lam))
    bs = open(C.R("scripts", "build_splits.py"), encoding="utf-8").read()
    claim("RW4-2", "Appendix F rule 9: the builder checks only the random-galaxy draws (train, val, "
          "test, target) against simulation hosts",
          '[("train", train_r), ("val", val_r), ("test", test_r), ("target(any fold)", union_tgt)]'
          in bs and "bad = s & sim_defl_set" in bs)
    claim("RW4-3", "Section 3.4: the test negatives are the whole hash partition reserved for them",
          "test_neg = sorted(int(o) for o in by_part.get(P_TEST_NEG, []))" in bs)
    mr = open(os.path.join(C.TABLES, "M_restart.tex"), encoding="utf-8").read()
    steps = int(_re.search(r"\\restartControlSteps\}\{([\d{},]+)\}", mr).group(1)
                .replace("{,}", ""))
    claim("RW4-4", "Section 6.1: the frozen continuation of Appendix E ran 2000 steps, at least the "
          "1500 further steps of the unfrozen arm", steps == 2000 >= 1500)
    import record_extract as RX          # the lab log's part of RW4-5 (see rewrite_v2_claims)
    claim("RW4-5", "Section 4.2: two optimiser settings compared on the development fold with the "
          "source-only arm; the published defaults adopted, and they did not score higher",
          RX.prose("RW4-5"))


def RV_F5():
    import make_review_macros as RV
    return RV.review(RV.R1, "F5_host_properties.json")


def DA_RAISE_MOST() -> bool:
    """Every adaptation headline gains more from the pool change than Baseline A and lambda=0."""
    gain = {}
    for arm, ro in (("coral", 2000), ("freematch", 1500), ("dann", 4000), ("bnm", 4000),
                    ("lambda0", 4000), ("baseline", 8000)):
        with pool("orig"):
            o = C.fold_mean(arm, "tpr", ro)[0]
        with pool("D1"):
            d = C.fold_mean(arm, "tpr", ro)[0]
        gain[arm] = d - o
    return min(gain[a] for a in ("coral", "freematch", "dann", "bnm")) > \
        max(gain["lambda0"], gain["baseline"])


pool = C.pool


def used_names() -> set:
    """Macro names the paper's .tex files reference (\\name{})."""
    import glob
    txt = ""
    for p in glob.glob(os.path.join(C.PAPER, "sections", "*.tex")) + [os.path.join(C.PAPER, "main.tex")]:
        txt += open(p, encoding="utf-8").read()
    return {n for n, *_ in MACROS if ("\\" + n + "{}") in txt}


def main() -> int:
    import sys
    build()
    # The worded conditions are the text's claims about the pool the paper reports. They are
    # checked on every build; --values-only skips them for one purpose only: rendering the
    # macros on the pre-registered pool to prove the number mapping
    # (audit/decontam/prose_macros.py), where the approved wording is not expected to hold.
    if "--values-only" not in sys.argv[1:]:
        prose_claims()
        with open(os.path.join(C.DATA, f"prose_claims_{C.POOL}.json"), "w", encoding="utf-8",
                  newline="\n") as fh:
            json.dump([{"id": i, "claim": s, "holds": h} for i, s, h in CLAIMS], fh, indent=1)
        failed = [c for c in CLAIMS if not c[2]]
        for i, s, h in CLAIMS:
            print(f"  claim {i:<4} {'holds' if h else 'FALSE'}  {s}")
        if failed:
            raise SystemExit(f"{len(failed)} prose claim(s) false on pool {C.POOL}: "
                             + ", ".join(c[0] for c in failed))
    # Only the macros the text uses are written, so every emitted macro is one the
    # cross-check covers. Before the prose was converted (audit/decontam/prose_macros.py
    # verify-plan) every macro is written.
    used = ({n for n, *_ in MACROS} if "--values-only" in sys.argv[1:]
            else (used_names() or {n for n, *_ in MACROS}))
    lines, last = [], None
    for name, value, nd, signed, source in MACROS:
        if name not in used:
            continue
        if source != last:                      # the source, once per group, on its own line
            lines.append(f"% {source}")
            last = source
        lines.append(rf"\newcommand{{\{name}}}{{{render(value, nd, signed)}}}")
    C.write_table(os.path.join(C.TABLES, "M_prose.tex"), "\n".join(lines), size="",
                  header_comment=("Macros for the result numbers quoted in the prose, "
                                  f"test pool {C.POOL}.\nEach prints as the typed number "
                                  "did: same decimals, same explicit sign."))
    print(f"  {sum(1 for l in lines if not l.startswith('%'))} prose macros written ({len(MACROS)} defined) on pool {C.POOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
