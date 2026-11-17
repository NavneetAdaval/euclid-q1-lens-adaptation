"""
make_t2_contrasts.py -- T2, the paired contrasts.
=================================================

    python paper/scripts/make_t2_contrasts.py

Every contrast is a **paired** difference on the same four confirmatory folds, so fold
difficulty cancels. The interval is a paired t-interval on n = 4 differences, df = 3,
as fixed in the round-5/6 pre-registration reporting block -- never a bootstrap at
n = 4, where 4**4 = 256 distinct resamples make the tails a lattice artifact.

THE MATCHED COLUMN IS NOT DECORATION. A budget match removes the BUDGET confound and
nothing else: head initialisation and each arm's own tuning budget still travel with
every row. Three heads initialise the six arms -- CORAL from round-1 CORAL heads,
FreeMatch from its own, DANN / BNM / lambda=0 from Baseline A's -- so
**DANN - lambda=0 is the only fully controlled pair in the table** (same head, same
budget, differing in lambda alone), and it carries the widest interval. Precision and
cleanliness point in opposite directions here and the table says so.

BUDGET-UNMATCHED ROWS ARE KEPT ONLY WHERE PROJECT.md KEEPS THEM, and each is labelled
unmatched in its own row: the CORAL / FreeMatch forms against the 4000 lambda=0
headline (7.22 keeps them for continuity) and the DANN - CORAL / DANN - FreeMatch
forms at declared budgets (7.18).

MULTIPLICITY IS DISCLOSED, NOT CORRECTED -- the caption says how many intervals the
table carries and at what level.

THE LAST PANEL IS A DIFFERENT ESTIMATOR AND A DIFFERENT KIND OF CLAIM. The frozen ->
unfrozen FreeMatch comparison is the one within-method contrast the project licenses
across rounds (7.15: same head LR, weight decay, augmentation, precision, batch
composition and literally the same training function; only ``n_blocks`` changed). It
was published under the earlier fold-aware paired bootstrap, so it is REPRODUCED FROM
ITS OWN ARTIFACT (``runs/foldmean_bootstrap.json``) rather than silently re-estimated
under the t-interval, and the estimator is named in the row. **No other round-1 number
is differenced against any nb3 number anywhere in this paper.**

Writes ``paper/tables/T2_contrasts.tex`` and nothing else.
"""
from __future__ import annotations

import json
import os

import _common as C

M, U, NAB = C.MATCHED, C.UNMATCHED, C.NOT_APPLICABLE

# The contrast set and the panel titles live in _common so that T2 and F3 are two
# renderings of ONE declaration and cannot drift apart.
ROWS = C.CONTRAST_ROWS

PANEL = {
    "A": r"\textit{Frozen against unfrozen treatment}",
    "B": r"\textit{A target-domain training signal vs.\ none} (each DA arm against the "
         r"unfrozen no-signal control)",
    "C": r"\textit{Objective against objective}",
}


def _width_clause():
    """The caption's comparison of interval widths, checked on the current pool.

    Printed as written only where it holds: DANN - lambda=0 has the widest of all the
    intervals, and of the three adaptation arms against the matched control the two
    tightest are CORAL's and FreeMatch's (initialised from their own heads). Always three
    lines, so the table's line layout does not depend on the pool.
    """
    w = {(a, ra, b, rb): C.contrast(a, ra, b, rb) for _, _, a, ra, b, rb, _, _ in ROWS}
    wid = {k: r["ci_hi"] - r["ci_lo"] for k, r in w.items()}
    dl = ("dann", 4000, "lambda0", 4000)
    widest = wid[dl] == max(wid.values())
    three = {"DANN": dl, "CORAL": ("coral", 2000, "lambda0", 2000),
             "FreeMatch": ("freematch", 1500, "lambda0", 1500)}
    tight = set(sorted(three, key=lambda n: wid[three[n]])[:2]) == {"CORAL", "FreeMatch"}
    if widest and tight:
        return [r"(same head, same budget, differing in $\lambda$ alone) --- and it carries the",
                r"widest interval, while the two tightest belong to arms initialised from a",
                r"different head. \textbf{Multiplicity is disclosed, not corrected}: 13 intervals"]
    if widest:
        return [r"(same head, same budget, differing in $\lambda$ alone) --- and it carries the",
                r"widest interval.",
                r"\textbf{Multiplicity is disclosed, not corrected}: 13 intervals"]
    return [r"(same head, same budget, differing in $\lambda$ alone).",
            r"%",
            r"\textbf{Multiplicity is disclosed, not corrected}: 13 intervals"]


def _lower_bound_clause(lb, one_pos):
    """The licensed pair's footnote, stating only the comparison that holds (three lines)."""
    if lb <= 0:
        return [rf"(${lb:+.4f}$) does not exclude zero, so ``distinguishable from",
                r"zero'' is not supported."]
    if lb < one_pos:
        return [rf"(${lb:+.4f}$) is smaller than one positive (${one_pos:.4f}$), so "
                r"``distinguishable from",
                r"zero'' is supported and ``large'' is not."]
    return [rf"(${lb:+.4f}$) exceeds one positive (${one_pos:.4f}$), so "
            r"``distinguishable from",
            r"zero'' and ``larger than one positive'' are both supported."]


def main() -> int:
    width_lines = _width_clause()
    body = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Paired contrasts in TPR@$10^{-3}$, folds 1--4.}",
        r"Paired $t$-interval on the four per-fold differences, $\mathrm{df}=3$; never a",
        r"bootstrap at $n=4$. ``Signs'' counts the folds agreeing in the direction of the",
        r"point estimate. \textbf{A budget match removes the budget confound and nothing",
        r"else}: head initialisation and each arm's own tuning budget travel with every",
        r"row, so \textbf{DANN $-$ $\lambda=0$ is the only fully controlled pair here}",
        *width_lines,
        r"at the 95\% level with no correction, sharing control arms and therefore not",
        r"independent.",
        r"$^{a}$Baseline A is frozen and converged; its 8000-step budget is not",
        r"commensurable with an unfrozen step count, so no budget-match state is claimed.",
        r"The comparison is nonetheless the most tightly paired in the project: the frozen",
        r"value is measured by the step-0 gate \emph{inside} each unfrozen run, on the same",
        r"fold, the same head and the same evaluation path.",
        r"$^{b}$The only fully controlled pair.",
        r"$^{c}$Budget-unmatched forms kept for continuity with the $\lambda=0$ headline",
        r"readout; never quoted as matched.",
        r"$^{d}$Read at each arm's own declared budget, which differ (4000 / 2000 / 1500);",
        r"unmatched by construction.}",
        r"\label{tab:contrasts}",
        r"\begin{tabular}{>{\TRAG}p{4.2cm}lrlcl}",
        r"\toprule",
        r"Contrast & Steps & $\Delta$ TPR@$10^{-3}$ & 95\% CI & Signs & Budget \\",
        r"\midrule",
    ]

    last_panel = None
    n_intervals = 0
    zero_notes = []      # a point estimate that rounds to zero: printed 0.0000, footnoted
    for panel, label, a, ra, b, rb, match, mark in ROWS:
        if panel != last_panel:
            body.append(r"\multicolumn{6}{l}{%s} \\" % PANEL[panel])
            last_panel = panel
        r = C.contrast(a, ra, b, rb)
        n_intervals += 1
        star = r"$^{*}$" if r["excludes_zero"] else ""
        if C.rounds_to_zero(r["mean"]):
            # "-0.0000" beside a sign count reads as a typo; print 0.0000 and give the
            # unrounded difference, whose sign the "Signs" column counts against.
            assert not zero_notes, "a second rounded-to-zero row needs its own footnote mark"
            est = C.f4(0.0) + star + r"$^{f}$"
            zero_notes.append(r"$^{f}$" + C.unrounded(r["mean"]) + " before rounding.")
        else:
            est = C.f4(r["mean"], signed=True) + star
        body.append(" & ".join([
            label + mark,
            f"{ra} / {rb}",
            est,
            C.ci(r["ci_lo"], r["ci_hi"]),
            f"{r['n_same_sign']}/4",
            match,
        ]) + r" \\")

    # ---- the licensed within-method, across-round contrast, on its own estimator ----
    fb = json.load(open(C.art("foldmean_bootstrap"), encoding="utf-8"))
    # Everything the row and its footnote state is read from the artifact: the star, the
    # sign count, the lower bound, and one positive at the largest per-fold positive count.
    fb_star = r"$^{*}$" if fb["ci_excludes_zero"] else ""
    fb_signs = sum(1 for v in fb["per_fold"].values()
                   if v["diff"] * (1 if fb["diff"] >= 0 else -1) > 0)
    one_pos = 1.0 / max(C.suite("freematch", f, 1500)["n_pos"] for f in C.CONFIRMATORY_FOLDS)
    body += [
        r"\midrule",
        # The one panel heading longer than the table: a paragraph cell of the text width
        # (no digits in the column spec, so the row's audited numbers keep their positions).
        r"\multicolumn{6}{@{\hspace{\tabcolsep}}p{\dimexpr\textwidth-\tabcolsep-\tabcolsep\relax}@{}}"
        r"{\textit{Within-method, frozen $\rightarrow$ unfrozen "
        r"(the one comparison licensed across rounds$^{e}$); fold-aware paired "
        r"bootstrap, $B=5000$}} \\",
        " & ".join([
            r"FreeMatch $\times$ nb3 $-$ FreeMatch frozen",
            "1500 / 8000",
            C.f4(fb["diff"], signed=True) + fb_star,
            C.ci(fb["ci"][0], fb["ci"][1]),
            f"{fb_signs}/{len(fb['per_fold'])}",
            NAB,
        ]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{2pt}",
        r"\footnotesize $^{*}$interval excludes zero." + "".join(" " + n for n in zero_notes),
        r"$^{e}$Round-1 FreeMatch and FreeMatch $\times$ nb3 share head LR, weight decay,",
        r"augmentation, precision, batch composition and the same training function; only",
        r"\texttt{n\_blocks} changed. \textbf{No other round-1 number is differenced against",
        r"any unfrozen number in this paper} --- eight configuration fields separate round-1",
    ] + ([
        r"CORAL from CORAL $\times$ nb3. This row is reproduced from its own published",
        r"artifact under the \textbf{fold-aware paired bootstrap} it was estimated with, and",
        r"is not re-estimated under the $t$-interval used above; its lower bound",
    ] if C.POOL == "orig" else [
        r"CORAL from CORAL $\times$ nb3. This row is recomputed on the test pool of Section~"
        r"\ref{sec:data}",
        r"under the \textbf{fold-aware paired bootstrap} it was published with, and is not",
        r"re-estimated under the $t$-interval used above; its lower bound",
    ]) + _lower_bound_clause(fb["ci"][0], one_pos) + [
        r"\end{table*}",
    ]

    C.write_table(os.path.join(C.TABLES, "T2_contrasts.tex"), "\n".join(body),
                  header_comment=(
                      "T2 -- paired contrasts, folds 1-4, TPR@FPR=1e-3.\n"
                      "Estimator: paired t on 4 differences, df=3 (analyze_lambda0_r6.paired_t),\n"
                      + ("except the final row, reproduced from runs/foldmean_bootstrap.json\n"
                         if C.POOL == "orig" else
                         "except the final row, paper/data/foldmean_bootstrap_" + C.POOL
                         + ".json (bootstrap_foldmean.py on this pool),\n")
                      + "under the fold-aware paired bootstrap it was published with."))
    print(f"  {n_intervals} t-intervals + 1 reproduced bootstrap interval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
