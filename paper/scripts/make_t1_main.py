"""
make_t1_main.py -- T1, the main results table (folds 1-4).
==========================================================

    python paper/scripts/make_t1_main.py

One row per arm: frozen Baseline A, the lambda=0 unfrozen control, and the four
unfrozen arms. Columns: fold-mean TPR@1e-3, sd across folds, declared budget, the
step the arm is read at, and the encoder regime.

EVERY NUMBER IS COMPUTED FROM THE ARM'S OWN STORED LOGITS by ``_common.suite`` ->
``metrics.metric_suite``. Nothing is read from PROJECT.md; ``crosscheck.py`` does that
comparison separately.

THREE THINGS THE CAPTION HAS TO CARRY, and they are generated with the numbers:

  * **Baseline A's row is its folds-1-4 mean**, not the 5-fold 0.176 that the round-1
    table quotes. Both are computed here and the difference is entirely fold 0. Only
    the folds-1-4 value is an admissible comparator on the confirmatory basis.
  * **Fold 0 is development and enters no mean.**
  * **BNM is read at 4000**, the mandatory readout, not at its own rule-derived budget
    of 1000 -- which fired ``no_matched_arm_budget`` and is its own-objective reading
    only.

Writes ``paper/tables/T1_main.tex`` and nothing else.
"""
from __future__ import annotations

import os
import statistics as st

import _common as C


def main() -> int:
    rows = []
    for arm in C.MAIN_ORDER:
        meta = C.ARM_META[arm]
        mean, sd, _ = C.fold_mean(arm, "tpr")
        budget = meta["budget"]
        readout = meta["readout"]
        bud = f"{budget}" if budget == readout else rf"{budget}\,$^{{\dagger}}$"
        rows.append((meta["label"], C.f4(mean), C.f4(sd), bud, f"{readout}",
                     meta["regime"]))

    # Baseline A's two bases, both computed, so the caption can state the difference.
    b14 = st.mean(C.per_fold("baseline", "tpr").values())
    b04 = st.mean(C.per_fold("baseline", "tpr", folds=(0, 1, 2, 3, 4)).values())

    body = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Main results: label-free TPR at FPR${}=10^{-3}$, folds 1--4.}",
        r"Each arm is read at the step fixed by its own budget rule, declared and hashed",
        r"before that arm's confirmatory folds existed; what differs between arms is only",
        r"\emph{where} each is read. \textbf{Fold 0 is development and enters no mean.}",
        r"Values are rates; the metric quantises at $1/98$ on folds 1--3 and $1/97$ on",
        r"fold 4. The sd is the across-fold sample sd of four values, not a standard error.",
        rf"\textbf{{Baseline A's row is its folds-1--4 mean ({b14:.4f})}}, the only",
        rf"admissible comparator on the confirmatory basis; its 5-fold mean is {b04:.4f}",
        r"and the difference is entirely fold~0.",
        r"$^{\dagger}$BNM's rule-derived budget of 1000 fired \texttt{no\_matched\_arm\_budget}",
        r"and is its own-objective reading only, so the arm is reported at the mandatory",
        r"4000 readout shared with the $\lambda=0$ known-null.",
        r"\label{tab:main}}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Arm & TPR@$10^{-3}$ & sd & Budget & Read at & Encoder \\",
        r"\midrule",
    ]
    for r in rows:
        body.append(" & ".join(r) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    C.write_table(os.path.join(C.TABLES, "T1_main.tex"), "\n".join(body),
                  header_comment="T1 -- main results, folds 1-4. TPR@FPR=1e-3, fold-mean.\n"
                                 "Source: each arm's per-checkpoint logits via "
                                 "scripts/analyze_window_average.ARMS + metrics.metric_suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
