r"""
make_t4_secondary.py -- T4, the secondary readouts.
===================================================

    python paper/scripts/make_t4_secondary.py

completeness@24\%-purity (Lines et al.'s operating point) and purity@50\%-completeness,
prevalence-corrected to $N_N/N_P = 1{:}2200$, plus AUROC as the saturating sanity
metric. Fold-means, folds 1--4.

THE LINES RATIO IS A SECONDARY AND THE TABLE IS BUILT TO SAY SO. The ratios and their
intervals are read from ``runs/round7/lines_ratios.json`` -- the artifact that persists
them -- and nothing is recomputed here that would let this table disagree with it. The
column carries its interval in the same row, never a bare percentage:

  * the metric is **undefined on 6 of the 12 arm x readout cells**, including frozen
    Baseline A on 3 of its 5 folds and DANN at its own declared budget. An arm with any
    undefined fold is n/a -- no fold-mean, no interval, never an average over the
    reachable subset;
  * where it is defined it swings ~4x across folds of one arm, and the intervals span
    [4.3\%, 92.9\%];
  * it **inverts under the read-off choice** -- DANN reads n/a at 4000 and 53.1\%, above
    every other arm, at the pre-declared 2000 sensitivity readout. Both DANN rows are in
    the table for exactly that reason.

purity@50\%-completeness and AUROC are computed from each arm's own stored logits via
``metrics.metric_suite``, the same pass the headline TPR comes from.

Writes ``paper/tables/T4_secondary.tex`` and nothing else.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st

import _common as C

# (row label, arm key, readout, tag)
ROWS = [
    (r"Baseline A (frozen)", "baseline", 8000, ""),
    (r"$\lambda=0$ unfrozen control", "lambda0", 4000, ""),
    (r"BNM $\times$ nb3", "bnm", 4000, ""),
    (r"FreeMatch $\times$ nb3", "freematch", 1500, ""),
    (r"DANN $\times$ nb3", "dann", 4000, ""),
    (r"\quad DANN, sensitivity readout", "dann", 2000, r"$^{\ddagger}$"),
    (r"CORAL $\times$ nb3", "coral", 2000, ""),
]


def _dagger_lines(lr):
    """The note on DANN's sensitivity row, stating only what holds on this pool.

    "Inverts" is printed only where the benchmark ratio is n/a at the declared budget and,
    at 2000, above every other arm's ratio at its quoted readout. The last line ends
    mid-sentence exactly as the original did, so the caption keeps its line layout.
    """
    d = lr["arms"][C.ARM_KEY["dann"]]["readouts"]
    d4, d2 = d["4000"], d["2000"]
    others = []
    for arm, ro in ROWS_X:
        rec = lr["arms"][C.ARM_KEY[arm]]["readouts"][str(ro)]
        if rec["defined_on_all_folds"] and "folds_1_4" not in rec:
            others.append(rec["pct_of_lines"])
        elif rec["defined_on_all_folds"]:
            others.append(rec["folds_1_4"]["pct_of_lines"])
    top = d2["defined_on_all_folds"] and all(d2["pct_of_lines"] > x for x in others)
    if top and not d4["defined_on_all_folds"]:
        return [r"$^{\ddagger}$DANN's pre-declared sensitivity readout. It is in the table because",
                r"the secondary \emph{inverts} between the two defensible read-off points: n/a at",
                r"the declared budget, and above every other arm at 2000. Ratios and intervals are"]
    if top:
        return [r"$^{\ddagger}$DANN's pre-declared sensitivity readout. It is in the table as",
                r"the second defensible read-off point; the secondary is defined at both, and",
                r"at 2000 it is above every other arm. Ratios and intervals are"]
    return [r"$^{\ddagger}$DANN's pre-declared sensitivity readout. It is in the table as",
            r"the second defensible read-off point.",
            r"Ratios and intervals are"]


# the arms whose ratio DANN's sensitivity readout is compared against (quoted readouts)
ROWS_X = [("baseline", 8000), ("lambda0", 4000), ("bnm", 4000), ("freematch", 1500),
          ("coral", 2000)]


def main() -> int:
    lr = json.load(open(C.art("lines_ratios"), encoding="utf-8"))
    lr_src = (r"\texttt{runs/round7/lines\_ratios.json}" if C.POOL == "orig" else
              r"\texttt{paper/data/lines\_ratios\_" + C.POOL + ".json}")

    body = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Secondary readouts, folds 1--4.} Completeness at 24\% purity is",
        r"Lines et al.'s operating point; $X$ expresses it as a fraction of their",
        r"label-free sim-only completeness of 0.50. \textbf{$X$ is a secondary and is never",
        r"quoted without this interval.} \textbf{That reference point (24\% purity at 50\%",
        r"completeness) is itself a mean over ten random initialisations with a reported",
        r"spread of $\pm 0.10$ in purity; the intervals below come only from our own",
        r"sampling variability at the fixed 0.50 denominator and do not propagate the",
        r"reference point's own uncertainty, so they are if anything too narrow.} An arm",
        r"with 24\% purity unreachable on \emph{any}",
        r"fold is \emph{n/a} --- no fold-mean, no interval, and never an average over the",
        r"reachable subset. Purity is prevalence-corrected to $N_N/N_P = 1{:}2200$, Euclid",
        r"Q1's rarity, so it is comparable with the benchmark. AUROC is a saturating sanity",
        r"metric and never an arbiter: the $\lambda=0$ row is the demonstration, a globally",
        r"competent ranker that is nearly useless at the operating point.",
        *_dagger_lines(lr),
        r"read from " + lr_src + r"; purity and AUROC are computed",
        r"from the same stored logits as the headline TPR.}",
        r"\label{tab:secondary}",
        r"\begin{tabular}{>{\TRAG}p{3.3cm}lrlrr}",
        r"\toprule",
        r"Arm & Read at & comp@24\%-pur. & $X = {\cdot}/0.50$, 95\% CI & pur.@50\%-comp. "
        r"& AUROC \\",
        r"\midrule",
    ]

    for label, arm, ro, tag in ROWS:
        rec = lr["arms"][C.ARM_KEY[arm]]["readouts"][str(ro)]
        comp = rec["fold_mean"]
        if rec["defined_on_all_folds"] and "folds_1_4" in rec:
            # Baseline A's record is on folds 0-4; this table's basis is folds 1-4.
            f14 = rec["folds_1_4"]
            comp = f14["fold_mean"]
            x = C.pct(f14["pct_of_lines"])
            lo, hi = (100.0 * v / C.LINES_COMPLETENESS for v in f14["ci95"])
            xcell = f"{x}, $[{lo:+.1f}, {hi:+.1f}]\\%$"
        elif rec["defined_on_all_folds"]:
            x = C.pct(rec["pct_of_lines"])
            lo, hi = rec["pct_of_lines_ci95"]
            xcell = f"{x}, $[{lo:+.1f}, {hi:+.1f}]\\%$"
        else:
            xcell = C.NA + r"\,\footnotesize(" + \
                ",".join(str(f) for f in rec["undefined_folds"]) + r")"
        pur, _, _ = C.fold_mean(arm, "purity_corr", ro)
        au, _, _ = C.fold_mean(arm, "auroc", ro)
        body.append(" & ".join([
            label + tag, str(ro),
            C.f4(comp) if comp is not None else C.NA,
            xcell, C.f4(pur), C.f4(au),
        ]) + r" \\")

    # Baseline A on its own 5-fold basis, for continuity with the round-1 table.
    pur5 = st.mean(C.per_fold("baseline", "purity_corr", 8000,
                              folds=(0, 1, 2, 3, 4)).values())
    au5 = st.mean(C.per_fold("baseline", "auroc", 8000, folds=(0, 1, 2, 3, 4)).values())
    b5 = lr["arms"][C.ARM_KEY["baseline"]]["readouts"]["8000"]
    assert b5["n_folds"] == 5, b5["n_folds"]
    if b5["defined_on_all_folds"]:
        # X is never quoted without its interval; the project's paired t is defined at
        # n = 4 only, so on this 5-fold basis the interval is n/a and printed as such.
        lo5, hi5 = b5["pct_of_lines_ci95"]
        ci5 = (f"$[{lo5:+.1f}, {hi5:+.1f}]\\%$"
               if all(isinstance(v, float) and math.isfinite(v) for v in (lo5, hi5))
               else C.NA)
        c5, x5 = C.f4(b5["fold_mean"]), C.pct(b5["pct_of_lines"]) + ", " + ci5
    else:
        c5 = C.NA
        x5 = C.NA + r"\,\footnotesize(" + ",".join(str(f) for f in b5["undefined_folds"]) + ")"
    body += [
        r"\midrule",
        " & ".join([r"\quad Baseline A, 5-fold basis$^{\S}$", "8000", c5,
                    x5, C.f4(pur5), C.f4(au5)]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{2pt}",
        r"\footnotesize \emph{n/a} in the completeness columns is followed by the folds on",
        r"which 24\% purity is unreachable. $^{\S}$Baseline A is a single source-only model,",
        r"so a folds-0--4 mean is legitimate for it; \textbf{only its folds-1--4 values are",
        r"admissible comparators} against the per-fold-model arms above.",
        r"\end{table*}",
    ]

    C.write_table(os.path.join(C.TABLES, "T4_secondary.tex"), "\n".join(body),
                  header_comment=("T4 -- secondary readouts, folds 1-4.\n"
                                  "comp@24%-purity + X ratio: " + lr_src.replace(
                                      r"\texttt{", "").replace("}", "").replace("\\", "")
                                  + ".\n"
                                  "purity@50%-completeness (1:2200) and AUROC: metric_suite on "
                                  "the stored logits."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
