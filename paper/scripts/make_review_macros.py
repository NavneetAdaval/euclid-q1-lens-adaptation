r"""
make_review_macros.py -- macros for the numbers the review rounds added to the text.
====================================================================================

    python paper/scripts/make_review_macros.py

Every value comes from one of three read-only sources and nowhere else:

  * the review-round JSONs, ``audit/review_r1/out/*.json`` and ``audit/review_r1b/out/*.json``.
    Before any value is read, every input file each JSON records under ``inputs_sha256`` is
    re-hashed and must match, so a macro cannot outlive a change in what produced it. An
    input that is absent is accepted only if ``paper/data/not_redistributed.json`` lists it
    with the same recorded hash (the Euclid cutouts and catalogues the public release does not
    redistribute); anything else absent fails;
  * ``PROJECTv7.md``, for the four early-run numbers of the Methods subsection "Where labels
    enter" (lines 491, 3159 and 4123); ``crosscheck.py`` re-reads them independently;
  * the run files of the DANN, lambda=0 and BNM folds, for the size of the held-out target
    slice (``dann_d_holdout_n`` / ``target_holdout_n``), checked against the review-round-1b
    lookup that reports it.

Writes ``paper/tables/M_review.tex`` and ``paper/data/review_macros.json`` (name, value as
printed, unrounded value, source) and nothing else. Trains nothing, loads no model.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re

import _common as C

R1 = C.R("audit", "review_r1", "out")
R1B = C.R("audit", "review_r1b", "out")
MACROS: list = []          # (name, printed, value, source)
_SHA: dict = {}


def _sha(path):
    if path not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


_LOADED: dict = {}
# The review rounds read a few of the paper's own generated files (paper/...) only to check
# that they reproduced the paper's numbers; rewrite v1 regenerates those files. For such an
# input, the recorded hash is verified against the pre-rewrite copy kept in the rewrite's
# backup, and the substitution is recorded in BACKUP_VERIFIED.
BACKUP = C.R("backup", "rewrite_v1_20261010_182859")
BACKUP_VERIFIED: list = []
# Inputs restored after a review run read them: rel -> (hash the review recorded, hash expected
# now). run_coral_r4_confirm.py: review r1b hashed the half-edited file on disk (0ec4f3c6...,
# kept at audit/coral_confirm_halfedit/); PROJECTv7 7.38 restored the version that ran the CORAL
# x nb3 confirmatory folds (2853805195bd88bf...). The one line r1b quotes from it (575) is
# identical in both.
RESTORED = {
    "scripts/run_coral_r4_confirm.py": (
        "0ec4f3c6cd05a854f53e3ae8d66e2fb2f3777b21abc23cfa5a6b54f34c0b9eae",
        "2853805195bd88bf4c1f4df8e32a4ff16458ba2845844ff17a4eea2323a7bd2d"),
}


# Inputs the public release deliberately leaves out (Euclid pixels, catalogues, superseded
# drafts), each with the hash its review file recorded. An absent input passes only if it is
# listed here with that same hash, so a missing file is never skipped silently. In the project
# every input is present and this list is never read.
WITHHELD_LIST = os.path.join(C.PAPER, "data", "not_redistributed.json")
_WITHHELD: dict = {}
VERIFIED: set = set()
WITHHELD: set = set()


def _withheld() -> dict:
    if not _WITHHELD:
        d = json.load(open(WITHHELD_LIST, encoding="utf-8"))
        _WITHHELD.update({e["path"]: e["sha256"] for e in d["entries"]})
    return _WITHHELD


def review(folder, name):
    """A review-round JSON's ``result``, after every recorded input hash is re-checked."""
    p = os.path.join(folder, name)
    if p not in _LOADED:
        d = json.load(open(p, encoding="utf-8"))
        assert d.get("status") == "post_hoc_exploratory", p
        for rel, want in d.get("inputs_sha256", {}).items():
            path = C.R(*rel.split("/"))
            if not os.path.exists(path):
                assert os.path.exists(WITHHELD_LIST) and _withheld().get(rel) == want, (
                    f"{name}: input {rel} is absent and not listed with its recorded hash in "
                    f"{os.path.relpath(WITHHELD_LIST, C.REPO)}")
                WITHHELD.add(rel)
                continue
            got = _sha(path)
            if rel in RESTORED and want == RESTORED[rel][0]:
                want = RESTORED[rel][1]
            if got != want and rel.startswith("paper/"):
                got = _sha(os.path.join(BACKUP, *rel.split("/")))
                if got == want:
                    BACKUP_VERIFIED.append((name, rel))
            assert got == want, f"{name}: input {rel} changed since the review run"
            VERIFIED.add(rel)
        _LOADED[p] = d
    return _LOADED[p]["result"]


def src(folder, name, key):
    return f"{os.path.relpath(os.path.join(folder, name), C.REPO).replace(os.sep, '/')} -> {key}"


def m(name, printed, value, source):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    MACROS.append((name, printed, float(value) if value is not None else None, source))


def f(x, nd, signed=False):
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def thousands(n):
    """The paper's number style (rewrite v5b, item 12): four-digit numbers take no separator;
    five or more digits are grouped with a thin space."""
    n = int(n)
    return str(n) if abs(n) < 10000 else f"{n:,}".replace(",", r"\,")


def declared_rows(gaps):
    """A1 gap rows whose steps are all declared readouts (the rule of the rewrite brief)."""
    return {k: g for k, g in gaps.items()
            if g.get("available") and all(g["steps_are_declared_readouts"])}


THREE = ("dann-coral", "dann-freematch", "coral-freematch")
BNMS = ("bnm-coral", "bnm-dann", "bnm-freematch")


def a1_sets():
    a1 = review(R1, "A1_resolution.json")
    rows = declared_rows(a1["gaps_and_ratios"])
    three = {k: g for k, g in rows.items() if k.split("|")[0] in THREE}
    bnm = {k: g for k, g in rows.items() if k.split("|")[0] in BNMS}
    nb = lambda g: g["ratio_detrended_b_linfit_resid"] * abs(g["gap"])   # noqa: E731
    return a1, three, bnm, nb


def project_early():
    """The four early-run numbers, each read from the PROJECTv7.md line the brief names.

    Line numbers re-pointed 2026-10-12 (rewrite v4b): the record's §7.35 corrections inserted
    marker lines above these, moving 491 -> 494, 3159/3162 -> 3178/3181 and 4123 -> 4143. The
    patterns live in ``scripts/record_extract.py`` (``rv.*``), which reads the log, or without
    it the extract; the line-3178 table header is checked the same way.
    """
    import record_extract as RX
    assert RX.check("rv.early_table_header"), "PROJECTv7.md:3178 moved"
    return {k: float(RX.value(f"rv.early_{k}")[0]) for k in ("noda", "iw", "dann", "probe")}


def holdout_n():
    vals = []
    for pat, key in ((("runs", "dann_main", "fold*", "extracted", "fold*", "run.json"), "dann_d_holdout_n"),
                     (("runs", "baseline_unfrozen", "fold*", "extracted", "*", "run.json"), "dann_d_holdout_n"),
                     (("runs", "bnm_r6_folds", "bnm_r6_fold*", "run.json"), "target_holdout_n")):
        for p in sorted(glob.glob(C.R(*pat))):
            vals.append(int(json.load(open(p, encoding="utf-8"))[key]))
    assert len(vals) == 12 and len(set(vals)) == 1, vals
    l1 = review(R1B, "L1_target_labels.json")
    assert f"({vals[0]} images)" in json.dumps(l1), "r1b L1 no longer reports the slice size"
    return vals[0]


def build() -> list:
    MACROS.clear()
    # ---------------- F4: fold overlap ----------------
    f4 = review(R1, "F4_fold_overlap.json")
    off = [v["count"] for a, row in f4["shared_target_lenses"].items()
           for b, v in row.items() if a != b]
    S = lambda k: src(R1, "F4_fold_overlap.json", k)          # noqa: E731
    m("rvSharedTargetMin", str(min(off)), min(off), S("shared_target_lenses (off-diagonal min)"))
    m("rvSharedTargetMax", str(max(off)), max(off), S("shared_target_lenses (off-diagonal max)"))
    tl = f4["pool_sizes"]["target_lens_rows"]["1"]
    m("rvTargetLensPool", str(tl), tl, S("pool_sizes.target_lens_rows.1"))
    fr = [v["fraction_of_row_fold_pool"] for a, row in f4["shared_target_lenses"].items()
          for b, v in row.items() if a != b]
    ov = 100 * sum(fr) / len(fr)
    m("rvTargetOverlapPct", f"{5 * round(ov / 5):.0f}", ov,
      S("shared_target_lenses fraction_of_row_fold_pool, off-diagonal mean, nearest 5%"))

    # ---------------- F5: host properties ----------------
    f5 = review(R1, "F5_host_properties.json")
    au = []
    for var, d in f5["single_variable_auroc_positive_vs_negative"].items():
        au += [x["auroc"] for x in d["vs_each_negative_type"].values() if isinstance(x, dict)]
        au.append(d["vs_all_source_negatives_having_it"]["auroc"])
    S = lambda k: src(R1, "F5_host_properties.json", k)       # noqa: E731
    m("rvHostAurocMin", f(min(au), 2), min(au), S("single_variable_auroc (min)"))
    m("rvHostAurocMax", f(max(au), 2), max(au), S("single_variable_auroc (max)"))
    tot = {p: int(v[next(iter(v))].split("/")[1]) for p, v in f5["availability"].items()}
    n_neg = tot["b1_bare_deflectors"] + tot["b2_random_galaxies"] + tot["b3_grade_X_ml_fp"]
    m("rvBareDeflN", thousands(tot["b1_bare_deflectors"]), tot["b1_bare_deflectors"],
      S("availability.b1_bare_deflectors"))
    m("rvSourceNegN", thousands(n_neg), n_neg, S("availability (b1 + b2 + b3)"))
    m("rvSourcePosN", thousands(tot["a_training_positive_hosts"]), tot["a_training_positive_hosts"],
      S("availability.a_training_positive_hosts"))

    # ---------------- F6: the 20 D2 lenses ----------------
    f6 = review(R1, "F6_d2_lenses.json")
    S = lambda k: src(R1, "F6_d2_lenses.json", k)             # noqa: E731
    roles = [x["roles"] for x in f6["lenses"]]
    n_train = sum(any(r.startswith("training negative") for r in rr) for rr in roles)
    n_val = sum(any(r.startswith("validation negative") for r in rr) and
                not any(r.startswith("training negative") for r in rr) for rr in roles)
    n_random = sum(any("(random galaxy)" in r for r in rr) for rr in roles)
    assert n_train + n_val == f6["n"]
    m("rvDTwoTrainNeg", str(n_train), n_train, S("lenses[].roles (training negatives)"))
    m("rvDTwoValNeg", str(n_val), n_val, S("lenses[].roles (validation negatives only)"))
    m("rvDTwoRandom", str(n_random), n_random, S("lenses[].roles (random galaxy)"))

    # ---------------- F7: the 28 lenses in the original pool ----------------
    f7 = review(R1, "F7_lenses_in_test_pool.json")["per_arm"]
    S = lambda k: src(R1, "F7_lenses_in_test_pool.json", k)   # noqa: E731
    own = [f7["coral@2000"][k]["n28_that_are_also_this_folds_test_positives"] for k in "1234"]
    for t in f7:
        assert [f7[t][k]["n28_that_are_also_this_folds_test_positives"] for k in "1234"] == own
    da = [f7[t][k]["n28_at_or_above_threshold"] for t in
          ("bnm@4000", "freematch@1500", "dann@4000", "coral@2000") for k in "1234"]
    ref = [f7[t][k]["n28_at_or_above_threshold"] for t in ("baseline@8000", "lambda0@4000")
           for k in "1234"]
    above = {f7[t][k]["n_negatives_at_or_above_threshold"] for t in f7 for k in "1234"}
    assert len(above) == 1
    m("rvOwnPosMin", str(min(own)), min(own), S("n28_that_are_also_this_folds_test_positives (min)"))
    m("rvOwnPosMax", str(max(own)), max(own), S("n28_that_are_also_this_folds_test_positives (max)"))
    m("rvFsevenDaMin", str(min(da)), min(da), S("n28_at_or_above_threshold, DA headlines (min)"))
    m("rvFsevenDaMax", str(max(da)), max(da), S("n28_at_or_above_threshold, DA headlines (max)"))
    m("rvFsevenRefMin", str(min(ref)), min(ref), S("n28_at_or_above_threshold, Baseline A and lambda=0 (min)"))
    m("rvFsevenRefMax", str(max(ref)), max(ref), S("n28_at_or_above_threshold, Baseline A and lambda=0 (max)"))
    a = above.pop()
    m("rvAboveThrN", str(a), a, S("n_negatives_at_or_above_threshold"))

    # ---------------- the exclusion rule's digest ----------------
    rule = C.pool_rule()
    m("rvRuleSha", rule["sha256_of_body_without_this_field"][:8], None,
      "audit/decontam/rule.json -> sha256_of_body_without_this_field (first 8 hex digits)")

    # ---------------- A1: the resolution scale (estimator b, declared readouts only) ----------------
    a1, three, bnm, nb = a1_sets()
    S = lambda k: src(R1, "A1_resolution.json", k)            # noqa: E731
    n3 = [nb(g) for g in three.values()]
    g3 = [abs(g["gap"]) for g in three.values()]
    gb = [abs(g["gap"]) for g in bnm.values()]
    m("rvNoiseMin", f(min(n3), 3), min(n3), S("gaps_and_ratios, three-arm declared rows: noise_b min"))
    m("rvNoiseMax", f(max(n3), 3), max(n3), S("gaps_and_ratios, three-arm declared rows: noise_b max"))
    m("rvScale", f(max(n3), 2), max(n3), S("gaps_and_ratios, three-arm declared rows: noise_b max"))
    m("rvGapThreeMin", f(min(g3), 3), min(g3), S("gaps_and_ratios, three-arm declared rows: |gap| min"))
    m("rvGapThreeMax", f(max(g3), 3), max(g3), S("gaps_and_ratios, three-arm declared rows: |gap| max"))
    m("rvGapBnmMin", f(min(gb), 2), min(gb), S("gaps_and_ratios, BNM declared rows: |gap| min"))
    m("rvGapBnmMax", f(max(gb), 2), max(gb), S("gaps_and_ratios, BNM declared rows: |gap| max"))
    sig = [a1["sigma_by_arm"][k]["folds1_4_mean_detrended_b_linfit_resid"]
           for k in ("lambda0", "bnm", "freematch", "dann", "coral")]
    lag = [a1["sigma_by_arm"][k]["folds1_4_mean_lag1_autocorr_resid"]
           for k in ("lambda0", "bnm", "freematch", "dann", "coral")]
    m("rvSigmaMin", f(min(sig), 2), min(sig), S("sigma_by_arm folds1_4_mean_detrended_b (min)"))
    m("rvSigmaMax", f(max(sig), 2), max(sig), S("sigma_by_arm folds1_4_mean_detrended_b (max)"))
    m("rvLagMin", f(min(lag), 2, True), min(lag), S("sigma_by_arm folds1_4_mean_lag1 (min)"))
    m("rvLagMax", f(max(lag), 2, True), max(lag), S("sigma_by_arm folds1_4_mean_lag1 (max)"))

    # ---------------- A2: top-scoring negatives ----------------
    a2 = review(R1, "A2_top_negatives.json")
    xm = review(R1, "A2_crossmatch.json")
    mb = review(R1, "A2_magnitude_bias.json")
    m("rvTopNegN", str(a2["n_unique_objects"]), a2["n_unique_objects"],
      src(R1, "A2_top_negatives.json", "n_unique_objects"))
    m("rvTopNegC", str(xm["n_objects_matching_A_B_C_X"]), xm["n_objects_matching_A_B_C_X"],
      src(R1, "A2_crossmatch.json", "n_objects_matching_A_B_C_X"))
    # r3: grade A / grade B recall ratio per readout, from Table 4's summed counts
    g3 = review(R1, "A3_recall_by_grade.json")["by_arm"]
    gr = [(d["A"]["summed"]["recovered"] / d["A"]["summed"]["total"])
          / (d["B"]["summed"]["recovered"] / d["B"]["summed"]["total"]) for d in g3.values()]
    m("rvGradeRatioMin", f(min(gr), 1), min(gr),
      src(R1, "A3_recall_by_grade.json", "by_arm.*.{A,B}.summed recovered/total (min ratio)"))
    m("rvGradeRatioMax", f(max(gr), 1), max(gr),
      src(R1, "A3_recall_by_grade.json", "by_arm.*.{A,B}.summed recovered/total (max ratio)"))
    af = 100 * mb["above_threshold_unique_objects"]["frac_fainter_22p5"]
    pf = 100 * mb["D1_pool"]["frac_fainter_22p5"]
    m("rvAboveFaintPct", f(af, 0), af, src(R1, "A2_magnitude_bias.json",
                                           "above_threshold_unique_objects.frac_fainter_22p5"))
    m("rvPoolFaintPct", f(pf, 0), pf, src(R1, "A2_magnitude_bias.json", "D1_pool.frac_fainter_22p5"))

    # ---------------- A3: lenses found and missed ----------------
    pr = review(R1, "A3_recall_by_property.json")
    gal = review(R1, "A3_galleries.json")["galleries"]
    th = pr["einstein_radius_effective_median_pdf (modeling_lens_mass.csv)"]
    m("rvThetaN", str(th["n_with_value"]), th["n_with_value"],
      src(R1, "A3_recall_by_property.json", "einstein_radius...n_with_value"))
    m("rvConfTestN", str(th["n_test_lenses_folds1_4"]), th["n_test_lenses_folds1_4"],
      src(R1, "A3_recall_by_property.json", "einstein_radius...n_test_lenses_folds1_4"))
    mb_ = gal["b_missed_by_every_arm"]
    m("rvMissedAllN", str(mb_["n_in_set"]), mb_["n_in_set"],
      src(R1, "A3_galleries.json", "galleries.b_missed_by_every_arm.n_in_set"))
    m("rvMissedAllB", str(mb_["grade_counts_in_set"]["B"]), mb_["grade_counts_in_set"]["B"],
      src(R1, "A3_galleries.json", "galleries.b_missed_by_every_arm.grade_counts_in_set.B"))

    # ---------------- the held-out target slice (r1b L1, read from the run files) ----------------
    h = holdout_n()
    m("rvHoldoutN", str(h), h, "runs/{dann_main,baseline_unfrozen,bnm_r6_folds}/*/run.json "
                               "(dann_d_holdout_n / target_holdout_n), all 12 folds; "
                               "audit/review_r1b/out/L1_target_labels.json")

    # ---------------- L2: the bright-only pool ----------------
    l2 = review(R1B, "L2_bright_pool.json")
    m("rvBrightN", thousands(l2["pool"]["n_bright"]), l2["pool"]["n_bright"],
      src(R1B, "L2_bright_pool.json", "pool.n_bright"))
    br = l2["readouts"]["bnm@4000"]["lines_ratio_bright"]
    m("rvBnmBrightRatio", f(br, 2), br, src(R1B, "L2_bright_pool.json",
                                           "readouts.bnm@4000.lines_ratio_bright"))
    bd = l2["readouts"]["bnm@4000"]["lines_ratio_D1"]
    m("rvBnmDOneRatio", f(bd, 2), bd, src(R1B, "L2_bright_pool.json",
                                         "readouts.bnm@4000.lines_ratio_D1"))

    # ---------------- run file: the cross-architecture gradient-norm move (Appendix J) ----------------
    bp = json.load(open(C.R("runs", "bnm_r6_prereg.json"), encoding="utf-8"))["provenance"]
    assert abs(100 * (bp["cpu_to_t4_ratio_move"] - 1) - bp["cpu_to_t4_ratio_move_pct"]) < 0.05
    m("rvGradShiftPct", f(bp["cpu_to_t4_ratio_move_pct"], 1), bp["cpu_to_t4_ratio_move_pct"],
      "runs/bnm_r6_prereg.json -> provenance.cpu_to_t4_ratio_move_pct")

    # ---------------- PROJECTv7.md: the early natural-imbalance runs ----------------
    e = project_early()
    m("rvEarlyNoDa", f(e["noda"], 2), e["noda"], "PROJECTv7.md:494 (no-DA head AUROC)")
    m("rvEarlyDann", f(e["dann"], 2), e["dann"], "PROJECTv7.md:4143 (plain DANN head AUROC)")
    m("rvEarlyIw", f(e["iw"], 2), e["iw"], "PROJECTv7.md:494 (IW-CDAN head AUROC)")
    m("rvEarlyProbe", f(e["probe"], 2), e["probe"], "PROJECTv7.md:3178-3181 (frozen linear probe)")
    return MACROS


# ---------------- quantities the prose claims assert (no macro) ----------------
def checks() -> dict:
    """Booleans and values make_m_prose.prose_claims turns into asserted claims."""
    a1, three, bnm, nb = a1_sets()
    a3 = review(R1, "A3_recall_by_grade.json")["by_arm"]
    ratios = {t: d["A"]["summed"]["rate"] / d["B"]["summed"]["rate"] for t, d in a3.items()}
    prop = review(R1, "A3_recall_by_property.json")
    th = prop["einstein_radius_effective_median_pdf (modeling_lens_mass.csv)"]["recall"]
    es = prop["expert_score (lens_cutouts.npz)"]["recall"]
    l2 = review(R1B, "L2_bright_pool.json")
    return {
        "three_ratio_b": {k: g["ratio_detrended_b_linfit_resid"] for k, g in three.items()},
        "bnm_ratio_b": {k: g["ratio_detrended_b_linfit_resid"] for k, g in bnm.items()},
        "three_gaps": {k: abs(g["gap"]) for k, g in three.items()},
        "three_noise": {k: nb(g) for k, g in three.items()},
        "grade_ab_ratio": ratios,
        "theta_small_lowest_da": {t: th[t][0]["rate"] < min(th[t][1]["rate"], th[t][2]["rate"])
                                  for t in ("bnm@4000", "freematch@1500", "dann@4000",
                                            "coral@2000")},
        "expert_score_rises": {t: es[t][0]["rate"] < es[t][1]["rate"] < es[t][2]["rate"]
                               for t in es},
        "l2_zero_inclusion_same": [c["zero_inclusion_same"] for c in l2["contrasts"]],
    }


def main() -> int:
    build()
    lines, last = [], None
    for name, printed, _v, source in MACROS:
        if source.split(" -> ")[0] != last:
            last = source.split(" -> ")[0]
            lines.append(f"% {last}")
        lines.append(rf"\newcommand{{\{name}}}{{{printed}}}")
    C.write_table(os.path.join(C.TABLES, "M_review.tex"), "\n".join(lines), size="",
                  header_comment="Macros for the review-round numbers (rewrite v1). Sources:\n"
                                 "audit/review_r1/out, audit/review_r1b/out (input hashes "
                                 "re-checked), PROJECTv7.md, run files.")
    os.makedirs(C.DATA, exist_ok=True)
    with open(os.path.join(C.DATA, "review_macros.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump([{"name": n, "printed": p, "value": v, "source": s}
                   for n, p, v, s in MACROS], fh, indent=1)
    print(f"  {len(MACROS)} review macros written; "
          f"{len(_SHA)} recorded input files re-hashed and matched")
    for name, rel in sorted(set(BACKUP_VERIFIED)):
        print(f"    {name}: {rel} regenerated by this rewrite; recorded hash matched the "
              f"pre-rewrite copy in {os.path.relpath(BACKUP, C.REPO)}")
    print(f"  {len(VERIFIED)} inputs verified, {len(WITHHELD)} not redistributed "
          f"({os.path.relpath(WITHHELD_LIST, C.REPO).replace(os.sep, '/')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
