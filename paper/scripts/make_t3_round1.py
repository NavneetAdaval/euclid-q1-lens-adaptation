r"""
make_t3_round1.py -- T3, the round-1 bake-off under the FROZEN protocol, 5-fold basis.
======================================================================================

    python paper/scripts/build_round1_comparisons.py   # once; slow
    python paper/scripts/make_t3_round1.py

THE BASIS IS DIFFERENT FROM EVERY OTHER RESULTS TABLE AND THE CAPTION SAYS SO. Round 1
is a **5-fold** mean under a **frozen** encoder; the confirmatory basis everywhere else
is folds 1--4 under an unfrozen one. **No round-1 number is differenced against any
unfrozen number anywhere in this paper** -- eight configuration fields separate round-1
CORAL from CORAL x nb3 (head LR, weight decay, cached-feature vs image-space,
augmentation, precision, gamma schedule, gamma, target EMA), so the obvious difference
would confound ``encoder unfrozen'' with ``protocol changed''. The one exception the
project licenses is FreeMatch's own frozen -> unfrozen comparison, which appears in T2
and nowhere else.

TWO COLUMNS ARE THERE TO SHOW A REPORTING RULE WORKING, not to add information:

  * **pooled.** A single pooled ROC is valid only for Baseline A, which is one
    source-only model scoring all five folds under one common threshold. Every DA arm
    trains a different model per fold, so there is no common threshold and a pooled
    number would be invalid. ``report_bakeoff`` detects this from the differing per-fold
    negatives and suppresses it at source -- the n/a cells here are the artifact's own
    nulls, not a formatting choice made in this script.
  * **comp@24\% defined.** The count of folds on which Lines et al.'s operating point is
    reachable at all. It rises with the tier, and on the source-only baseline it is 2 of
    5 -- the first sign of what section~5 reports at length.

The paired column is the class-conditional paired bootstrap ($B=1000$, seed 0) round 1
published under, regenerated into JSON by ``build_round1_comparisons.py``; the estimator
is named in the caption because it is **not** the $t$-interval used on folds 1--4.

Writes ``paper/tables/T3_round1.tex`` and nothing else.
"""
from __future__ import annotations

import json
import math
import os
import sys

import _common as C

DATA = C.DATA

ROWS = [
    ("Baseline A (source-only)", "reference", "r1_baseline", None),
    ("CORAL (warmup-cosine)", "marginal alignment", "r1_coral", "coral"),
    ("AdaMatch", "self-training", "r1_adamatch", "adamatch"),
    ("FreeMatch (standard)", "self-training", "r1_freematch", "freematch"),
]


def main() -> int:
    missing = [t for _, _, _, t in ROWS if t and
               not os.path.exists(os.path.join(
                   DATA, f"round1_paired_{t}.json" if C.POOL == "orig"
                   else f"round1_paired_{t}_{C.POOL}.json"))]
    if missing:
        print("  !! missing paired comparisons for: " + ", ".join(missing))
        print("     run:  python paper/scripts/build_round1_comparisons.py")
        return 1

    body = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Round 1: the domain-adaptation bake-off under the frozen",
        r"protocol.} \textbf{Basis: 5-fold means, frozen encoder} --- deliberately",
        r"different from every other results table in this paper, whose basis is folds",
        r"1--4 with three encoder blocks unfrozen. \textbf{No number in this table is",
        r"differenced against any unfrozen number}; eight configuration fields separate",
        r"round-1 CORAL from CORAL $\times$ nb3 alone. All methods ran a matched protocol:",
        r"frozen Zoobot encoder, batch 256, 8000 steps, selection rule R1 (last",
        r"checkpoint) pre-registered and frozen before results. The paired column is the",
        r"\textbf{class-conditional paired bootstrap} ($B=1000$, seed 0) round 1 published",
        r"under, not the $t$-interval used on the confirmatory folds. A \emph{pooled} entry",
        r"exists only for Baseline A: it is one source-only model scoring all five folds",
        r"under one common threshold, whereas every DA arm trains a different model per",
        r"fold, so no common threshold exists and the pooled figure is suppressed at source.",
        r"The last column counts the folds on which 24\% purity is reachable at all.}",
        r"\label{tab:round1}",
        r"\begin{tabular}{>{\TRAG}p{1.8cm}>{\TRAG}p{1.4cm}rrl>{\TRAG}p{2.65cm}l}",
        r"\toprule",
        r"Method & Family & TPR@$10^{-3}$ & pooled & vs.\ Baseline A, 95\% CI & Verdict "
        r"& comp@24\% \\",
        r"\midrule",
    ]

    for label, family, art, tag in ROWS:
        # Recomputed from the run's stored score dumps on the current pool; on the
        # pre-registered pool it must reproduce the run's own metric_suite.json.
        run = os.path.basename(os.path.dirname(C.ART[art]))
        rs = C.round1_summary(run)
        af = rs["across_fold"]["tpr"]
        pooled = rs["pooled"]
        pcell = C.f4(pooled["tpr"]) if pooled else C.NA
        ndef = sum(1 for v in rs["per_fold"].values() if math.isfinite(v["comp_at_purity"]))
        with C.pool("orig"):
            ro = C.round1_summary(run)
        ms = json.load(open(C.ART[art], encoding="utf-8"))
        assert abs(ro["across_fold"]["tpr"]["mean"] - ms["across_fold"]["tpr"]["mean"]) < 1e-12
        assert abs(ro["across_fold"]["tpr"]["std"] - ms["across_fold"]["tpr"]["std"]) < 1e-12
        assert (ro["pooled"] is None) == (ms.get("pooled") is None), run
        if ro["pooled"]:
            assert abs(ro["pooled"]["tpr"] - ms["pooled"]["tpr"]["value"]) < 1e-12, run
        assert sum(1 for v in ro["per_fold"].values() if math.isfinite(v["comp_at_purity"])) \
            == sum(1 for v in ms["per_fold"].values()
                   if v["comp_at_purity"]["value"] is not None
                   and math.isfinite(v["comp_at_purity"]["value"])), run
        if tag is None:
            diff, verdict = "---", "reference"
        else:
            pname = (f"round1_paired_{tag}.json" if C.POOL == "orig"
                     else f"round1_paired_{tag}_{C.POOL}.json")
            cmp_ = json.load(open(os.path.join(DATA, pname),
                                  encoding="utf-8"))["aggregate"]["tpr"]
            diff = (C.f4(cmp_["diff"], signed=True) + " " +
                    C.ci(cmp_["ci"][0], cmp_["ci"][1]))
            verdict = (r"\textbf{distinguishable}~$\uparrow$" if cmp_["distinguishable"]
                       else "not distinguishable")
        body.append(" & ".join([
            label, family,
            f"${af['mean']:.4f} \\pm {af['std']:.4f}$",
            pcell, diff, verdict, f"{ndef}/5",
        ]) + r" \\")

    # ---- the DANN fold-0 probe, with every caveat attached, from its own artifact ----
    probe = json.load(open(C.ART["r1_dann_probe"], encoding="utf-8"))
    last = probe["candidates"][-1]
    body += [
        r"\midrule",
        " & ".join([
            r"DANN$^{\dagger}$", "marginal alignment",
            f"${last['fold0_tpr_1e-3']:.4f}$", C.NA, "---",
            "documented negative", C.NA,
        ]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{2pt}",
        r"\footnotesize $^{\dagger}$\textbf{Fold-0 probe only, not a matched 5-fold run}",
        r"--- a cached-feature run with a different adapter, optimiser and budget, entered",
        r"here as a tier placement and nothing more. Its value is the R1 last checkpoint",
        rf"({last['tag']}); even oracle selection over all {probe['n_candidates']} of its",
        rf"checkpoints leaves it at ${probe['oracle']['fold0_tpr_1e-3']:.4f}$, against",
        r"Baseline A's own fold-0 value, so the placement does not turn on the read-off rule.",
        r"Because its basis is one fold it carries no across-fold spread and no pooled value.",
    ] + ([
        r"Its value is computed on the pre-registered test pool",
        r"(Appendix~\ref{app:prereg_pool}); every other row uses the test pool of",
        r"Section~\ref{sec:data}.",
    ] if C.POOL != "orig" else []) + [
        r"\end{table*}",
    ]

    orig = C.POOL == "orig"
    C.write_table(os.path.join(C.TABLES, "T3_round1.tex"), "\n".join(body),
                  header_comment=("T3 -- round 1 bake-off, FROZEN protocol, 5-FOLD basis.\n"
                                  + ("Means/sd: runs/bakeoff_*/metric_suite.json.\n" if orig
                                     else "Means/sd: runs/bakeoff_*/scores/*.npz, test pool "
                                     + C.POOL + " (orig asserted equal to metric_suite.json).\n")
                                  + ("Paired column: paper/data/round1_paired_*.json " if orig
                                     else "Paired column: paper/data/round1_paired_*_"
                                     + C.POOL + ".json ")
                                  +
                                  "(compare_methods.py, B=1000, seed 0).\n"
                                  "DANN row: runs/curve_dann_fold0/results.json (fold-0 probe)."),
                  size="scriptsize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
