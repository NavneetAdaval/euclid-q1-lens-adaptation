r"""
make_t6_window.py -- T6 (appendix), checkpoint-window score averaging.
======================================================================

    python paper/scripts/make_t6_window.py

$k \in \{1, 2, 4, 8\}$, both declared score combinations, every arm at every mandatory
readout, from ``runs/round7/window_average.json``.

WHAT MAKES THIS TABLE READABLE RATHER THAN A SECOND SET OF RESULTS:

  * **It is a declared secondary analysis and replaces no hashed headline.** $k$, the
    grid, both combinations and the window-placement rule were declared and hashed
    (\texttt{a572b81e\ldots}) before any windowed number existed, and the analysis module
    contains no $k$ of its own -- it reads that file.
  * **$k = 1$ IS the existing single-point read-off**, not a new estimator, so the $k=1$
    column must reproduce each arm's hashed headline exactly. This script checks that
    independently: it compares the $k=1$ column against the headline recomputed from the
    same arm's logits by ``_common``, and prints a FAIL if any arm disagrees beyond
    $10^{-9}$. That check is what licenses reading the rest of the row.
  * **The window ends AT the pre-registered read-off step, so the rule reads no metric.**
    Where fewer than $k$ trained checkpoints exist at or before the read-off, the window
    truncates and the achieved $k$ is reported rather than silently padded.
  * **Both combinations are reported everywhere.** ``mean_logit`` is not scale-invariant,
    so one wide-scale checkpoint can dominate and turn the window back into a
    single-point read-off in disguise; ``mean_rank`` exists for that reason. They do
    \emph{not} agree in magnitude, and declaring both in advance is what makes that
    visible instead of a hidden analyst choice. They agree on every qualitative verdict.

Writes ``paper/tables/T6_window.tex`` and nothing else.
"""
from __future__ import annotations

import json
import os

import _common as C

KS = ["1", "2", "4", "8"]
ORDER = [("coral", 2000), ("freematch", 1500), ("dann", 4000), ("dann", 2000),
         ("lambda0", 4000), ("lambda0", 2000), ("lambda0", 1500),
         ("bnm", 1000), ("bnm", 1500), ("bnm", 2000), ("bnm", 4000),
         ("baseline", 8000)]


def _agreement_line(w):
    """The caption's last clause, checked on the current pool.

    The two combinations "agree on every qualitative verdict" is printed only where, at
    k = 8, they order every pair of the three unfrozen adaptation arms (DANN at both of its
    read-offs) the same way; otherwise the clause names the pairs they order differently.
    """
    rows = {"CORAL": ("coral", 2000), "FreeMatch": ("freematch", 1500),
            "DANN (4000)": ("dann", 4000), "DANN (2000)": ("dann", 2000)}
    v = {n: tuple(w["arms"][C.ARM_KEY[a]]["readouts"][str(ro)][c]["8"]["tpr_mean"]
                  for c in ("mean_logit", "mean_rank")) for n, (a, ro) in rows.items()}
    names = list(rows)
    flips = [(p, q) for i, p in enumerate(names) for q in names[i + 1:]
             if rows[p][0] != rows[q][0]
             and (v[p][0] - v[q][0]) * (v[p][1] - v[q][1]) < 0]
    if not flips:
        return r"but they agree on every qualitative verdict.}"
    # An arm whose order against DANN flips at BOTH of DANN's read-offs is named once.
    dann = ("DANN (4000)", "DANN (2000)")
    both = [p for p in names if p not in dann
            and all((p, d) in flips or (d, p) in flips for d in dann)]
    rest = [(p, q) for p, q in flips if not ({p, q} & set(both) and {p, q} & set(dann))]
    parts = ([f"{p} and DANN differently at both of DANN's read-off points" for p in both]
             + [f"{p} and {q} differently" for p, q in rest])
    return r"and at $k=8$ they order " + "; ".join(parts) + r".}"


def main() -> int:
    w = json.load(open(C.art("window_average"), encoding="utf-8"))
    # Baseline A's windows need its per-step scores before step 8000, which come from its
    # head checkpoints and are not recomputed per pool, so its row is read from the
    # pre-registered-pool artifact and labelled as such.
    w_orig = json.load(open(C.ART["window_average"], encoding="utf-8"))
    prereg = json.load(open(C.ART["window_prereg"], encoding="utf-8"))
    assert w["prereg_sha256"] == prereg.get("sha256", w["prereg_sha256"])
    assert w_orig["prereg_sha256"] == w["prereg_sha256"]

    def src(arm):
        return w_orig if arm == "baseline" else w

    def pool_of(arm):
        return "orig" if arm == "baseline" else C.POOL

    # ---- the k=1 reproduction check, run here rather than asserted ----
    problems = []
    for arm, ro in ORDER:
        rec = src(arm)["arms"][C.ARM_KEY[arm]]["readouts"][str(ro)]["mean_logit"]["1"]
        folds = (0, 1, 2, 3, 4) if arm == "baseline" else C.CONFIRMATORY_FOLDS
        with C.pool(pool_of(arm)):
            mine, _, _ = C.fold_mean(arm, "tpr", ro, folds=folds)
        if abs(rec["tpr_mean"] - mine) > 1e-9:
            problems.append(f"{arm}@{ro}: window k=1 {rec['tpr_mean']!r} vs "
                            f"recomputed {mine!r}")
    if problems:
        print("  !! k=1 REPRODUCTION FAILED:")
        for p in problems:
            print("     " + p)
    else:
        print(f"  k=1 reproduces the recomputed headline on all {len(ORDER)} "
              f"arm x readout cells")

    body = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Checkpoint-window score averaging (appendix).} Fold-mean",
        r"TPR@$10^{-3}$ under an average of the $k$ consecutive checkpoints' \emph{scores}",
        r"ending at the arm's pre-registered read-off step --- scores, never weights, and",
        r"the window is placed by the read-off rule, so \textbf{the rule reads no metric}.",
        r"$k$, the grid, both combinations and the placement rule were declared and hashed",
        r"(\texttt{a572b81e\ldots}) before any windowed number existed. \textbf{This is a",
        # On the pre-registered pool the headline values are the hashed ones; on any other
        # pool the hashed files fix the read-off, not the value, so the caption says that.
        (r"declared secondary analysis and replaces no hashed headline.} \textbf{$k=1$ is"
         if C.POOL == "orig" else
         r"declared secondary analysis and replaces no pre-registered read-off.} "
         r"\textbf{$k=1$ is"),
        r"the existing single-point read-off, not a new estimator}: that column reproduces",
        r"every arm's headline, checked here against the value recomputed independently",
        r"from the same logits, which is what licenses reading the rest of each row.",
        r"$k_{\mathrm{ach}}$ is the achieved window where fewer than $k$ trained",
        r"checkpoints exist at or before the read-off; the window truncates rather than",
        r"padding. The two combinations \textbf{do not agree in magnitude} --- declaring",
        r"both in advance is what makes that visible rather than an analyst's choice ---",
        _agreement_line(w),
        r"\label{tab:window}",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{4}{c}{\texttt{mean\_logit} (primary)} "
        r"& \multicolumn{4}{c}{\texttt{mean\_rank} (secondary)} \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
        r"Arm & Read at & $k_{\mathrm{ach}}$ & $k{=}1$ & $k{=}2$ & $k{=}4$ & $k{=}8$ "
        r"& $k{=}1$ & $k{=}2$ & $k{=}4$ & $k{=}8$ \\",
        r"\midrule",
    ]

    for arm, ro in ORDER:
        rec = src(arm)["arms"][C.ARM_KEY[arm]]["readouts"][str(ro)]
        ach = max(rec["mean_logit"]["8"]["achieved_k"])
        label = C.ARM_META[arm]["short"]
        if ro != C.ARM_META[arm]["readout"]:
            label += r"\,\footnotesize" + (
                "(own budget)" if ro == C.ARM_META[arm]["budget"] else "(sens.)")
        if pool_of(arm) != C.POOL:
            label += r"$^{\dagger}$"
        cells = [label, str(ro), str(ach)]
        for comb in ("mean_logit", "mean_rank"):
            for k in KS:
                cells.append(C.f4(rec[comb][k]["tpr_mean"]))
        body.append(" & ".join(cells) + r" \\")

    body += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{2pt}",
        r"\footnotesize Baseline A's row is a folds-0--4 mean --- it is a single",
        r"source-only model, so fold-0 inclusion is legitimate for it; every other row is",
        r"folds 1--4. Its per-step score vectors are recomputed on CPU from the frozen",
        r"640-d feature cache and the retained per-step head weights, which is exact",
        r"because the encoder is frozen.",
    ] + ([
        r"$^{\dagger}$Computed on the pre-registered test pool",
        r"(Appendix~\ref{app:prereg_pool}); every other row uses the test pool of",
        r"Section~\ref{sec:data}.",
    ] if C.POOL != "orig" else []) + [
        r"\end{table*}",
    ]

    C.write_table(os.path.join(C.TABLES, "T6_window.tex"), "\n".join(body),
                  header_comment=("T6 (appendix) -- checkpoint-window averaging.\n"
                                  + ("Source: runs/round7/window_average.json, declared and "
                                     if C.POOL == "orig" else
                                     "Source: paper/data/window_average_" + C.POOL +
                                     ".json (Baseline A row: runs/round7/"
                                     "window_average.json), declared and ")
                                  + "hashed at a572b81e...\n"
                                  "k=1 cross-checked against metric_suite on the same logits."),
                  placement="htbp")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
