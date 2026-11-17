r"""
make_main_tables_v2.py -- the compact main-text tables of rewrite v1.
====================================================================

    python paper/scripts/make_main_tables_v2.py

Four tables at main-text precision (two decimals for rates and differences). Each is
computed from the same ``_common`` calls and artifacts as its four-decimal counterpart,
which stays in the appendices unchanged (``verify_orig_tables.py`` checks those):

  * ``T1v2_frozen.tex``    -- the frozen comparison (four-decimal: T3_round1.tex), with one
                              method taxonomy: marginal alignment by moment matching (CORAL)
                              or adversarially (DANN); self-training (AdaMatch, FreeMatch);
  * ``T2v2_main.tex``      -- the main table (four-decimal: T1_main.tex);
  * ``T3v2_contrasts.tex`` -- the paired contrasts (four-decimal: T2_contrasts.tex), with
                              the pre-registered rows marked by a dagger and the pre-declared
                              sensitivity rows by a double dagger (``_common.CONTRAST_PREREG``),
                              the target-signal block ordered 1, 5, 6, then 2, 3, then 4;
  * ``TX_grade_recall.tex`` -- recall by grade, from audit/review_r1/out/A3_recall_by_grade.json
                              (input hashes re-checked by make_review_macros.review).

Writes those four files and nothing else.
"""
from __future__ import annotations

import json
import math
import os

import _common as C
import make_review_macros as RV


def f2(x, signed=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return C.NA
    if round(x, 2) == 0.0:
        return "$0.00$"
    return f"${x:+.2f}$" if signed else f"${x:.2f}$"


def ci2(lo, hi):
    return f"$[{lo:+.2f}, {hi:+.2f}]$".replace("[-0.00", "[0.00").replace(", -0.00]", ", 0.00]")


# ------------------------------------------------------------------ Table 1
FROZEN = [("Baseline A", "source only", "bakeoff_baseline_A", None),
          ("CORAL", "marginal alignment (moment matching)", "bakeoff_coral", "coral"),
          ("AdaMatch", "self-training", "bakeoff_adamatch_5fold", "adamatch"),
          ("FreeMatch", "self-training", "bakeoff_freematch_5fold", "freematch")]


def table1():
    body = [r"\begin{table}[tbp]", r"\centering",
            r"\caption{\textbf{The frozen comparison.} Means over folds 0--4, the development "
            r"fold included (with across-fold sd), of TPR at FPR${}=10^{-3}$, encoder frozen, "
            r"8000 steps, last checkpoint. Differences against Baseline A are paired by fold over "
            r"the same five folds, with 95\% intervals from the "
            r"class-conditional paired bootstrap ($B=1000$). DANN ran on the development "
            r"fold only, with a different adapter and budget, and on the original test pool; "
            r"every other row uses the cleaned pool.}",
            r"\label{tab:frozen}",
            r"\begin{tabular}{l>{\TRAG}p{3.6cm}rr}", r"\toprule",
            r"Method & Family & TPR@$10^{-3}$ & vs.\ Baseline A, 95\% CI \\", r"\midrule"]
    with C.pool("D1"):
        for label, fam, run, tag in FROZEN:
            rs = C.round1_summary(run)["across_fold"]["tpr"]
            cell = f"${rs['mean']:.2f} \\pm {rs['std']:.2f}$"
            if tag:
                agg = json.load(open(os.path.join(C.DATA, f"round1_paired_{tag}_D1.json"),
                                     encoding="utf-8"))["aggregate"]["tpr"]
                lo, hi = agg["ci"]
                # a bound that rounds to zero at two decimals is shown at four, so that a
                # bound just above zero stays visible as such
                diff = f2(agg["diff"], True) + " " + (
                    f"$[{lo:+.4f}, {hi:+.2f}]$" if round(lo, 2) == 0.0 else ci2(lo, hi))
            else:
                diff = "---"
            body.append(f"{label} & {fam} & {cell} & {diff} \\\\")
    probe = json.load(open(C.ART["r1_dann_probe"], encoding="utf-8"))
    last = probe["candidates"][-1]["fold0_tpr_1e-3"]
    body += [r"\midrule",
             f"DANN (fold 0) & marginal alignment (adversarial) & ${last:.2f}$ & --- \\\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    C.write_table(os.path.join(C.TABLES, "T1v2_frozen.tex"), "\n".join(body),
                  header_comment="Table 1 (rewrite v1): the frozen comparison at two decimals. "
                                 "Sources as T3_round1.tex.")


# ------------------------------------------------------------------ Table 2
MAIN = [("baseline", "Baseline A", "frozen"), ("lambda0", r"$\lambda=0$ control", "unfrozen"),
        ("bnm", "BNM", "unfrozen"), ("freematch", "FreeMatch", "unfrozen"),
        ("dann", "DANN", "unfrozen"), ("coral", "CORAL", "unfrozen")]


def table2():
    body = [r"\begin{table}[tbp]", r"\centering",
            r"\caption{\textbf{Main results.} TPR at FPR${}=10^{-3}$ on the cleaned test pool, "
            r"mean and sample sd over the confirmatory folds 1--4; each arm is read at its "
            r"final budget except BNM, whose rule gave 1000 steps and which is read at "
            r"4000, where its control is read (Appendix~\ref{sec:budget-detail}). Per-fold "
            r"and four-decimal values: Appendix~\ref{app:perfold}.}",
            r"\label{tab:main2}",
            r"\begin{tabular}{lrrrl}", r"\toprule",
            r"Arm & TPR@$10^{-3}$ & sd & Read at & Encoder \\", r"\midrule"]
    with C.pool("D1"):
        for arm, label, enc in MAIN:
            ro = C.ARM_META[arm]["readout"]
            mean, sd, _ = C.fold_mean(arm, "tpr", ro)
            body.append(f"{label} & ${mean:.2f}$ & ${sd:.2f}$ & {ro} & {enc} \\\\")
        mean, sd, _ = C.fold_mean("dann", "tpr", 2000)
        body.append(f"DANN, sensitivity readout & ${mean:.2f}$ & ${sd:.2f}$ & 2000 & unfrozen \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    C.write_table(os.path.join(C.TABLES, "T2v2_main.tex"), "\n".join(body),
                  header_comment="Table 2 (rewrite v1): the main table at two decimals. "
                                 "_common.fold_mean on pool D1.")


# ------------------------------------------------------------------ Table 3
ORDER = [0, 1, 5, 6, 2, 3, 4, 7, 8, 9, 10, 11, 12]
MARK = {C.PREREG: r"$^{\dagger}$", C.SENSITIVITY: r"$^{\ddagger}$", C.NEITHER: ""}
LABEL = {0: r"$\lambda=0$ $-$ Baseline A", 1: r"DANN $-$ $\lambda=0$",
         2: r"CORAL $-$ $\lambda=0$", 3: r"FreeMatch $-$ $\lambda=0$", 4: r"BNM $-$ $\lambda=0$",
         5: r"CORAL $-$ $\lambda=0$", 6: r"FreeMatch $-$ $\lambda=0$", 7: r"BNM $-$ CORAL",
         8: r"BNM $-$ DANN", 9: r"BNM $-$ FreeMatch", 10: r"DANN $-$ CORAL",
         11: r"DANN $-$ FreeMatch", 12: r"DANN $-$ CORAL"}
PANEL = {"A": r"\textit{Frozen against unfrozen, no target signal}",
         "B": r"\textit{A target signal against none}",
         "C": r"\textit{Objective against objective}"}


def table3():
    assert sorted(ORDER) == list(range(len(C.CONTRAST_ROWS)))
    body = [r"\begin{table}[tbp]", r"\centering",
            r"\caption{\textbf{Paired contrasts} in TPR at FPR${}=10^{-3}$, folds 1--4, cleaned "
            r"pool: mean difference, paired $t$-interval (df${}=3$), and the number of folds "
            r"agreeing in sign. $^{\dagger}$Registered with the $\lambda=0$ control, before it ran "
            r"but after the adaptation arms' results were known (Section~\ref{sec:prereg}). "
            r"$^{\ddagger}$Sensitivity readout declared with those contrasts. Steps give where "
            r"each arm is "
            r"read. Four-decimal values: Table~\ref{tab:contrasts4}.}",
            r"\label{tab:contrasts2}",
            r"\begin{tabular}{llrrc}", r"\toprule",
            r"Contrast & Steps & $\Delta$ & 95\% CI & Signs \\"]
    last = None
    with C.pool("D1"):
        for i in ORDER:
            panel, _l, a, ra, b, rb, _m, _k = C.CONTRAST_ROWS[i]
            if panel != last:
                body += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{{PANEL[panel]}}} \\"]
                last = panel
            r = C.contrast(a, ra, b, rb)
            body.append(f"{LABEL[i]}{MARK[C.CONTRAST_PREREG[i]]} & {ra} / {rb} & "
                        f"{f2(r['mean'], True)} & {ci2(r['ci_lo'], r['ci_hi'])} & "
                        f"{r['n_same_sign']}/4 \\\\")
        fb = json.load(open(C.art("foldmean_bootstrap"), encoding="utf-8"))
    body += [r"\midrule",
             r"\multicolumn{5}{l}{\textit{Frozen against unfrozen, same objective "
             r"(fold-aware paired bootstrap)}} \\",
             f"FreeMatch unfrozen $-$ frozen & 1500 / 8000 & {f2(fb['diff'], True)} & "
             f"{ci2(fb['ci'][0], fb['ci'][1])} & 4/4 \\\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    C.write_table(os.path.join(C.TABLES, "T3v2_contrasts.tex"), "\n".join(body),
                  header_comment="Table 3 (rewrite v1): contrasts at two decimals, "
                                 "_common.contrast over CONTRAST_ROWS (pool D1); "
                                 "marks from _common.CONTRAST_PREREG; last row "
                                 "paper/data/foldmean_bootstrap_D1.json.")


# ------------------------------------------------------------------ grade recall
READ = [("baseline@8000", "Baseline A"), ("lambda0@4000", r"$\lambda=0$ control"),
        ("bnm@4000", "BNM"), ("freematch@1500", "FreeMatch"), ("dann@4000", "DANN"),
        ("dann@2000", "DANN, 2000"), ("coral@2000", "CORAL")]


def table_x():
    g = RV.review(RV.R1, "A3_recall_by_grade.json")["by_arm"]
    body = [r"\begin{table}[tbp]", r"\centering",
            r"\caption{\textbf{Recall by grade} at each fold's FPR${}=10^{-3}$ threshold on the "
            r"cleaned pool, summed over folds 1--4: recovered/total, rate, and Clopper--Pearson "
            r"95\% interval. Descriptive only; no contrast is formed. Per-fold counts: "
            r"Appendix~\ref{app:breakdowns}.}",
            r"\label{tab:graderecall}",
            r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r" & \multicolumn{3}{c}{grade A} & \multicolumn{3}{c}{grade B} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"Readout & found & rate & 95\% CI & found & rate & 95\% CI \\", r"\midrule"]
    for key, label in READ:
        cells = []
        for gr in "AB":
            s = g[key][gr]["summed"]
            lo, hi = s["clopper_pearson_95"]
            cells += [f"{s['recovered']}/{s['total']}", f"${s['rate']:.2f}$",
                      f"$[{lo:.2f}, {hi:.2f}]$"]
        body.append(" & ".join([label] + cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    C.write_table(os.path.join(C.TABLES, "TX_grade_recall.tex"), "\n".join(body),
                  header_comment="Table (rewrite v1): recall by grade, from "
                                 "audit/review_r1/out/A3_recall_by_grade.json.")


def main() -> int:
    table1()
    table2()
    table3()
    table_x()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
