r"""
make_appendix_tables_v2.py -- the appendix tables added by rewrite v1.
=====================================================================

    python paper/scripts/make_appendix_tables_v2.py

  * (``AC_orderings_v2.tex`` moved to make_orderings_v3.py in rewrite v3.)
  * ``T2v2_contrasts4.tex`` -- Appendix A, Table 5 (rewrite v2), from tables/T2_contrasts.tex;
  * ``AH_bright_pool.tex``   -- Appendix H, the bright-only test pool, from
        audit/review_r1b/out/L2_bright_pool.json;
  * ``AI_grade_perfold.tex``, ``AI_recall_property.tex`` -- Appendix I, from
        audit/review_r1/out/A3_recall_by_grade.json and A3_recall_by_property.json;
  * ``AI_top_negatives.tex`` -- Appendix I, the 25 test negatives placed above threshold most
        often, from audit/review_r1/out/top_negatives.csv, each row's count re-derived from
        A2_top_negatives.json before it is printed.

Review-round JSONs are read through ``make_review_macros.review``, which re-checks every
recorded input hash first. Writes those files and nothing else.
"""
from __future__ import annotations

import csv
import os
import re

import _common as C
import make_review_macros as RV

HERE = os.path.dirname(os.path.abspath(__file__))


def f4(x, signed=False):
    if x is None:
        return C.NA
    return f"${x:+.4f}$" if signed else f"${x:.4f}$"


# ------------------------------------------------------------------ Appendix C
# Rewrite v3: Table 11 (AC_orderings_v2.tex) is built by make_orderings_v3.py, which reads
# the rows' sources directly and never runs the protected make_a3_orderings.py (that script
# reads runs/fold_average.json, which the public release excludes).


# ------------------------------------------------------------------ Appendix H
NAMES = {"baseline@8000": "Baseline A (8000)", "lambda0@4000": r"$\lambda=0$ (4000)",
         "bnm@4000": "BNM (4000)", "freematch@1500": "FreeMatch (1500)",
         "dann@4000": "DANN (4000)", "dann@2000": "DANN (2000, sens.)",
         "coral@2000": "CORAL (2000)"}
CNAME = {"DANN - lambda=0": r"DANN $-$ $\lambda=0$", "CORAL - lambda=0": r"CORAL $-$ $\lambda=0$",
         "FreeMatch - lambda=0": r"FreeMatch $-$ $\lambda=0$", "BNM - lambda=0": r"BNM $-$ $\lambda=0$",
         "BNM - CORAL": r"BNM $-$ CORAL"}


def bright_pool():
    l2 = RV.review(RV.R1B, "L2_bright_pool.json")
    n = l2["pool"]
    a = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{\textbf{The bright-only test pool.} The cleaned test negatives with VIS PSF "
         rf"magnitude $\le 22.5$ (${RV.thousands(n['n_bright'])}$ of "
         rf"${RV.thousands(n['n_D1'])}$; the magnitude limit of "
         r"Section~\ref{sec:splits}); positives unchanged. (a) TPR at FPR${}=10^{-3}$, folds 1--4. "
         r"(b) Paired contrasts, paired $t$, df${}=3$. (c) Completeness at 24\% purity and its "
         r"ratio to 0.50; n/a where any fold is undefined.}",
         r"\label{tab:bright}", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"\multicolumn{9}{l}{\textit{(a) TPR@$10^{-3}$}} \\",
         r"Readout & f1 & f2 & f3 & f4 & mean & sd & cleaned pool & change \\", r"\midrule"]
    for k, d in l2["readouts"].items():
        b = d["tpr_bright"]
        a.append(" & ".join([NAMES[k]] + [f4(b["per_fold"][f]) for f in "1234"] +
                            [f4(b["mean"]), f4(b["sd"]), f4(d["tpr_D1"]["mean"]),
                             f4(d["tpr_change_mean"], True)]) + r" \\")
    a += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{6pt}",
          r"\begin{tabular}{llrrrc}", r"\toprule",
          r"\multicolumn{6}{l}{\textit{(b) Paired contrasts, $\Delta$ and 95\% CI}} \\",
          r"Contrast & Steps & bright pool & 95\% CI & cleaned pool & 0 excluded in both \\",
          r"\midrule"]
    for c in l2["contrasts"]:
        b = c["bright"]
        a.append(f"{CNAME[c['contrast']]} & {c['steps'][0]} / {c['steps'][1]} & "
                 f"{f4(b['mean'], True)} & $[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]$ & "
                 f"{f4(c['D1']['mean'], True)} & "
                 f"{'yes' if c['zero_inclusion_same'] and not c['includes_zero']['bright'] else 'no'} \\\\")
    a += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{6pt}",
          r"\begin{tabular}{lrrrrrrr}", r"\toprule",
          r"\multicolumn{8}{l}{\textit{(c) Completeness at 24\% purity}} \\",
          r"Readout & f1 & f2 & f3 & f4 & mean & ratio & ratio, cleaned pool \\", r"\midrule"]
    for k, d in l2["readouts"].items():
        c = d["comp_at_purity_bright"]
        a.append(" & ".join([NAMES[k]] + [f4(c["per_fold"][f]) for f in "1234"] +
                            [f4(c["mean"]), f4(d["lines_ratio_bright"]),
                             f4(d["lines_ratio_D1"])]) + r" \\")
    a += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    C.write_table(os.path.join(C.TABLES, "AH_bright_pool.tex"), "\n".join(a),
                  header_comment="Appendix H (rewrite v1): audit/review_r1b/out/L2_bright_pool.json.",
                  placement="htbp")


# ------------------------------------------------------------------ Appendix I
def grade_perfold():
    g = RV.review(RV.R1, "A3_recall_by_grade.json")["by_arm"]
    a = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{\textbf{Recall by grade, per fold}, at each fold's FPR${}=10^{-3}$ threshold "
         r"on the cleaned pool: recovered/total. Sums and intervals: Table~\ref{tab:graderecall}.}",
         r"\label{tab:gradeperfold}", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r" & \multicolumn{4}{c}{grade A} & \multicolumn{4}{c}{grade B} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Readout & f1 & f2 & f3 & f4 & f1 & f2 & f3 & f4 \\", r"\midrule"]
    for k, d in g.items():
        cells = [f"{d[gr]['per_fold'][f]['recovered']}/{d[gr]['per_fold'][f]['total']}"
                 for gr in "AB" for f in "1234"]
        a.append(" & ".join([NAMES[k]] + cells) + r" \\")
    a += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    C.write_table(os.path.join(C.TABLES, "AI_grade_perfold.tex"), "\n".join(a),
                  header_comment="Appendix I (rewrite v1): audit/review_r1/out/A3_recall_by_grade.json.",
                  placement="htbp")


PROPS = [("expert_score (lens_cutouts.npz)", "Expert score"),
         ("einstein_radius_effective_median_pdf (modeling_lens_mass.csv)",
          r"Einstein radius ($''$)"),
         ("vis_lensed_source_magnitude_ab_median_pdf (modeling_mge_magnitude.csv; arc brightness)",
          "Arc magnitude (VIS, AB)"),
         ("vis_max_lensed_source_signal_to_noise_ratio (modeling_mge_magnitude.csv)", "Arc S/N")]


def recall_property():
    p = RV.review(RV.R1, "A3_recall_by_property.json")
    a = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{\textbf{Recall by lens property}, folds 1--4, cleaned pool. Bins are tertiles "
         r"of the confirmatory test lenses that have a value (edges given); cells are "
         r"recovered/total and rate. Einstein radius and arc values come from the lens-modelling "
         r"tables stored with the Discovery Engine catalogue and exist for a subset of lenses; "
         r"stored values of exactly 0 and one arc magnitude above 30 are kept as stored.}",
         r"\label{tab:recallprop}", r"\scriptsize", r"\begin{tabular}{lrrr}", r"\toprule"]
    for key, label in PROPS:
        d = p[key]
        e = d["bin_edges"]
        a += [rf"\multicolumn{{4}}{{l}}{{\textit{{{label}}}: {d['n_with_value']} of "
              rf"{d['n_test_lenses_folds1_4']} lenses; edges {e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}, "
              rf"{e[3]:.2f}}} \\", r"\midrule",
              r"Readout & low & middle & high \\"]
        for k, rr in d["recall"].items():
            a.append(" & ".join([NAMES[k]] + [f"{x['recovered']}/{x['total']} (${x['rate']:.2f}$)"
                                              for x in rr]) + r" \\")
        a.append(r"\midrule")
    a[-1] = r"\bottomrule"
    a += [r"\end{tabular}", r"\end{table}"]
    C.write_table(os.path.join(C.TABLES, "AI_recall_property.tex"), "\n".join(a), size="",
                  header_comment="Appendix I (rewrite v1): audit/review_r1/out/A3_recall_by_property.json.",
                  placement="htbp")


def top_negatives():
    a2 = RV.review(RV.R1, "A2_top_negatives.json")
    counts = {}
    for t, d in a2["per_arm_fold"].items():
        for f, rec in d.items():
            for x in rec["above_threshold"]:
                counts[int(x["euclid_id"])] = counts.get(int(x["euclid_id"]), 0) + 1
    path = C.R("audit", "review_r1", "out", "top_negatives.csv")
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    rows.sort(key=lambda r: (-int(r["n_armfold_above_threshold"]), int(r["best_rank"]),
                             int(r["euclid_id"])))
    top = rows[:25]
    a = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{\textbf{The test negatives most often above threshold.} The 25 cleaned-pool "
         r"negatives placed above the FPR${}=10^{-3}$ threshold in the most of the 28 "
         r"readout-by-fold cells (seven readouts, four folds); position and VIS PSF magnitude "
         r"from the MER catalogue; best rank among the negatives of any cell; grade where the "
         r"object is in the Discovery Engine catalogue. We make no claim about their nature.}",
         r"\label{tab:topneg}", r"\begin{tabular}{rrrrrrc}", r"\toprule",
         r"Euclid ID & RA ($^\circ$) & Dec ($^\circ$) & VIS mag & cells & best rank & grade \\",
         r"\midrule"]
    for r in top:
        oid = int(r["euclid_id"])
        assert counts.get(oid, 0) == int(r["n_armfold_above_threshold"]), oid
        g = sorted({x.split(":")[1] for x in r["catalogue_match_grades"].split(";") if x})
        a.append(f"${oid}$ & {float(r['ra_deg']):.5f} & ${float(r['dec_deg']):+.5f}$ & "
                 f"{float(r['vis_psf_mag_ab']):.2f} & {r['n_armfold_above_threshold']} & "
                 f"{r['best_rank']} & {'/'.join(g) if g else '---'} \\\\")
    a += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    C.write_table(os.path.join(C.TABLES, "AI_top_negatives.tex"), "\n".join(a),
                  header_comment="Appendix I (rewrite v1): audit/review_r1/out/top_negatives.csv, "
                                 "counts re-derived from A2_top_negatives.json.",
                  placement="htbp")


# ------------------------------------------------------------------ Appendix A, Table 5
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\{,\}\d{3})*(?:\.\d+)?")


def contrasts4_v2():
    """Rewrite v2 section 3.1: the four-decimal contrasts (make_t2_contrasts.py's protected
    output, tables/T2_contrasts.tex) reordered and marked as Table 3, with the paper's arm
    names. Every row is parsed from the protected file; the Steps / Delta / CI / Signs /
    Budget cells are carried over unchanged, and their number tokens are asserted identical."""
    import json
    import make_main_tables_v2 as MT
    src = open(os.path.join(C.TABLES, "T2_contrasts.tex"), encoding="utf-8").read()
    body = src.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = [l.strip() for l in body.split("\n")
            if " & " in l and not l.strip().startswith(r"\multicolumn")]
    assert len(rows) == len(C.CONTRAST_ROWS) + 1, len(rows)
    cells = []
    for i, r in enumerate(rows):
        c = [x.strip() for x in r.rstrip("\\").strip().split(" & ")]
        assert len(c) == 6, r
        if i < len(C.CONTRAST_ROWS):
            _p, _l, a, ra, b, rb, _m, _k = C.CONTRAST_ROWS[i]
            assert c[1] == f"{ra} / {rb}", (i, c[1])
        cells.append(c)
    old_tokens = sorted(t for c in cells for t in NUM.findall(" ".join(c[1:5])))

    def clean(c):
        delta = c[2].replace("$^{f}$", "$^{a}$")
        assert not re.search(r"\$\^\{[b-e]\}\$", delta), delta
        return [c[1], delta, c[3], c[4], c[5].replace(r"\textbf{unmatched}", "unmatched")]

    fb = json.load(open(C.art("foldmean_bootstrap"), encoding="utf-8"))
    assert fb["B"] == 5000
    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{\textbf{Paired contrasts at four decimals}, folds 1--4, cleaned pool. "
             r"Paired $t$-interval on the four per-fold differences (df${}=3$); the last row uses "
             r"the fold-aware paired bootstrap ($B=5000$). Signs counts the folds agreeing in "
             r"direction with the mean. $^{\dagger}$Registered with the $\lambda=0$ control, before it "
             r"ran; $^{\ddagger}$sensitivity readout declared with those contrasts. The arms differ in head initialisation and tuning "
             r"(Section~\ref{sec:arms}), and a budget match removes only the budget difference. "
             r"\contrastCount{} intervals at 95\%, with no multiplicity correction; several share "
             r"the control arm. $^{*}$Interval excludes zero.}",
             r"\label{tab:contrasts4}",
             r"\begin{tabular}{llrlcl}", r"\toprule",
             r"Contrast & Steps & $\Delta$ TPR@$10^{-3}$ & 95\% CI & Signs & Budget \\"]
    last, new = None, []
    for i in MT.ORDER:
        panel = C.CONTRAST_ROWS[i][0]
        if panel != last:
            lines += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{{MT.PANEL[panel]}}} \\"]
            last = panel
        c = clean(cells[i])
        new.append(c)
        lines.append(" & ".join([MT.LABEL[i] + MT.MARK[C.CONTRAST_PREREG[i]]] + c) + r" \\")
    c = clean(cells[-1])
    new.append(c)
    lines += [r"\midrule",
              r"\multicolumn{6}{l}{\textit{Frozen against unfrozen, same objective "
              r"(fold-aware paired bootstrap)}} \\",
              " & ".join([r"FreeMatch unfrozen $-$ frozen"] + c) + r" \\",
              r"\bottomrule", r"\end{tabular}", "", r"\vspace{2pt}",
              r"\footnotesize $^{a}$$-0.00003$ before rounding.", r"\end{table}"]
    assert sorted(t for c in new for t in NUM.findall(" ".join(c[:4]))) == old_tokens
    assert "$-0.00003$ before rounding" in src
    C.write_table(os.path.join(C.TABLES, "T2v2_contrasts4.tex"), "\n".join(lines),
                  header_comment="Appendix A (rewrite v2): tables/T2_contrasts.tex's rows "
                                 "(make_t2_contrasts.py, protected), reordered and marked as "
                                 "Table 3; number tokens asserted identical.",
                  placement="htbp")


def main() -> int:
    bright_pool()
    grade_perfold()
    recall_property()
    top_negatives()
    contrasts4_v2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
