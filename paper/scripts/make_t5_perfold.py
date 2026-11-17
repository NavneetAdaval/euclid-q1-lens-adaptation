r"""
make_t5_perfold.py -- T5 (appendix), per-fold values for every arm and every readout.
====================================================================================

    python paper/scripts/make_t5_perfold.py

Four panels -- TPR@$10^{-3}$, AUROC, completeness@24\%-purity, purity@50\%-completeness
-- each with one row per (arm, readout) and one column per fold. This is the table that
makes every aggregate in the paper checkable by hand, and it is where the project's two
standing reporting rules are visible at once:

  * **fold 0 is development.** It exists on disk only for Baseline A among the arms
    reported here (the unfrozen arms' fold-0 runs are development runs at different
    budgets and are not part of this grid), and where it appears it is marked and
    excluded from the mean;
  * **undefined is n/a, and n/a propagates.** completeness@24\%-purity is unreachable on
    several folds; an arm with any undefined fold gets no fold-mean and no interval.

Rates throughout, never counts. The metric quantises at $1/98$ on folds 1--3 and $1/97$
on fold 4, which is why the per-fold cells are exact multiples of those.

Every value is ``metrics.metric_suite`` on that arm's own stored logits for that fold and
step. Writes ``paper/tables/T5_perfold.tex`` and nothing else.
"""
from __future__ import annotations

import math
import os

import _common as C

PANELS = [
    ("tpr", r"TPR at FPR${}=10^{-3}$ (primary)", "tab:perfold-tpr", 4),
    ("auroc", r"AUROC (secondary, saturating sanity metric)", "tab:perfold-auroc", 4),
    ("comp_at_purity", r"completeness at 24\% purity (secondary; Lines et al.'s "
                       r"operating point)", "tab:perfold-comp", 4),
    ("purity_corr", r"purity at 50\% completeness, corrected to $1{:}2200$",
     "tab:perfold-purity", 4),
]


def rows_for(metric: str):
    out = []
    for arm in C.MAIN_ORDER:
        meta = C.ARM_META[arm]
        for ro in sorted(C.READOUTS[arm]):
            declared = (ro == meta["readout"])
            if declared:
                tag = ""
            elif ro == meta["budget"]:
                # BNM's rule-derived budget of 1000 fired `no_matched_arm_budget`: it is
                # that arm's own-objective reading and is never cross-arm comparable.
                tag = r"\,\footnotesize(own budget)"
            else:
                tag = r"\,\footnotesize(sens.)"
            label = meta["short"] + tag
            vals = C.per_fold(arm, metric, ro)
            mean, sd, _ = C.fold_mean(arm, metric, ro)
            f0 = None
            if arm == "baseline":
                f0 = C.suite(arm, 0, ro)[metric]
            out.append((label, ro, declared, f0, vals, mean, sd))
    return out


def main() -> int:
    body = []
    for metric, title, lab, nd in PANELS:
        body += [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{\textbf{Per-fold values: " + title + r".} Rows marked",
            r"\footnotesize(sens.)\normalsize{} are pre-declared sensitivity readouts and do",
            r"not replace the arm's headline, whichever is higher; the row marked",
            r"\footnotesize(own budget)\normalsize{} is BNM's rule-derived budget, which fired",
            r"\texttt{no\_matched\_arm\_budget} and is its own-objective reading only. The fold-0 column is",
            r"\textbf{development} and enters no mean; it is populated only for the single",
            r"frozen Baseline A model. \emph{n/a} marks a value the metric leaves undefined,",
            r"and an arm with any undefined fold gets no fold-mean.}",
            r"\label{" + lab + r"}",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"Arm & Step & fold 0$^{\dagger}$ & fold 1 & fold 2 & fold 3 & fold 4 & mean "
            r"& sd \\",
            r"\midrule",
        ]
        for label, ro, declared, f0, vals, mean, sd in rows_for(metric):
            cells = [label, str(ro),
                     C.NA if f0 is None else C.f4(f0, nd)]
            cells += [C.f4(vals[f], nd) for f in (1, 2, 3, 4)]
            cells += [C.f4(mean, nd), C.f4(sd, nd)]
            body.append(" & ".join(cells) + r" \\")
        body += [
            r"\bottomrule",
            r"\end{tabular}",
            r"",
            r"\vspace{2pt}",
            r"\footnotesize $^{\dagger}$fold 0 is the development fold and is excluded from",
            r"every mean in this paper; \emph{n/a} in that column means the arm has no",
            r"fold-0 run on this grid, not that the value is undefined.",
            r"\end{table*}",
            r"",
        ]

    C.write_table(os.path.join(C.TABLES, "T5_perfold.tex"), "\n".join(body),
                  header_comment=("T5 (appendix) -- per-fold values, every arm, every "
                                  "readout, four metrics.\nSource: metric_suite on each "
                                  "arm's stored per-checkpoint logits."),
                  placement="htbp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
