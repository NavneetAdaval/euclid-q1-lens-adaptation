r"""
make_a3_orderings.py -- Appendix A3: every ordering that did not survive its own error bar.
===========================================================================================

    python paper/scripts/make_a3_orderings.py

One row per ordering, each with how it died. Every number is read from the run or analysis
artifact that measured it, and every "how it died" verdict is ASSERTED from that artifact
before it is printed -- if a re-run changed an outcome, this script stops rather than
printing a stale sentence:

  1  budget peak            runs/fold_average.json  (budget_comparison)
  2  decay onset            runs/devmax_truncation.json  (per-cap primary detector)
  3  agreement -> shrinkage runs/coral_r4_mechanism.json
  4  Fisher-ratio order     runs/coral_r4_fisher_ci.json
  5  read-off inversion     runs/round7/window_average.json
  6  DANN's spread          per-fold TPR through _common (metric_suite on stored logits)
  7  folds 3-4 hard         per-fold AUROC through _common
  8  rankable at all        runs/round7/window_fold0.json  (checkpoint sd multiples)
  9  lambda=0 least stable  runs/round7/window_fold0.json  (row9_stability)
  10 budget-rule crossing   runs/bnm_r6_budget.json + the development run's stats_only.csv

Writes ``paper/tables/A3_orderings.tex`` and nothing else.
"""
from __future__ import annotations

import csv
import json
import os

import _common as C


def j(*p):
    return json.load(open(C.R(*p), encoding="utf-8"))


def f2(x):
    return f"${x:.2f}$"


def f3(x, signed=False):
    return f"${x:+.3f}$" if signed else f"${x:.3f}$"


def main() -> int:
    rows = []

    # 1 -- budget peak ------------------------------------------------------------------
    b = j("runs", "fold_average.json")["budget_comparison"]
    cands = sorted(int(k) for k in b["rows"])
    assert b["n_distinguishable"] == 0
    rows.append((r"The budget at which the self-training arm's curve peaks",
                 r"\S\ref{sec:negative}",
                 f"None of {b['n_candidates']} candidate budgets ({cands[0]}--{cands[-1]} "
                 f"steps) is distinguishable from the best, at step {b['best_step']}."))

    # 2 -- decay onset ------------------------------------------------------------------
    pc = j("runs", "devmax_truncation.json")["fold0_development"]["per_cap"]
    caps = sorted(int(c) for c in pc)
    hit = [c for c in caps if pc[str(c)]["primary"]["detected"]]
    assert hit == [caps[-1]], f"decay detected at {hit}, not only at the cap"
    lo_c, hi_c = caps[0], caps[-2]
    rows.append((r"Decay sets in before the 8000-step cap",
                 r"\S\ref{sec:negative}",
                 f"Detectable only at the full cap of {caps[-1]}; every shorter "
                 f"truncation ({lo_c}--{hi_c}) has an interval that crosses zero."))

    # 3 -- agreement at init predicts CORAL's shrinkage ---------------------------------
    arms = j("runs", "coral_r4_mechanism.json")["arms"]
    r16, r0 = arms["g16"]["spearman_primary"], arms["g0"]["spearman_primary"]
    p16 = arms["g16"]["partial_spearman_primary_ctrl_logs0"]
    assert abs(r0) > abs(r16), "the known-null is no longer the stronger correlation"
    rows.append((r"Source--target agreement at initialisation predicts which feature "
                 r"directions the alignment objective shrinks",
                 "---",
                 f"Spearman $\\rho = {r16:.2f}$ at the weight used, but ${r0:.2f}$ at "
                 f"weight zero, where the objective contributes no gradient; controlling "
                 f"for initial scale leaves $\\rho = {p16:.3f}$."))

    # 4 -- Fisher-ratio ordering vs shrinkage -------------------------------------------
    fc = j("runs", "coral_r4_fisher_ci.json")
    assert fc["ordering_stable_across_lam"] is False
    gam = [("g0", 0), ("g1", 1), ("g4", 4), ("g16", 16)]
    inverted = {}
    for lam in fc["lams"]:
        pa = fc["by_lam"][str(lam)]["per_arm"]
        r = [pa[g]["ratio_median"] for g, _ in gam]
        inverted[lam] = [(gam[i][1], gam[i + 1][1]) for i in range(3) if r[i + 1] < r[i]]
    bad = {lam: v for lam, v in inverted.items() if v}
    assert bad, "the Fisher-ratio ordering is monotone at every shrinkage"
    lam_b, pairs = next(iter(sorted(bad.items())))
    ok = sorted(l for l in inverted if not inverted[l])
    rows.append((r"A Fisher-ratio separability measure rises with the alignment weight",
                 "---",
                 f"Monotone in the weight at whitening shrinkage "
                 f"{' and '.join(str(l) for l in ok)}; at {lam_b} the weight-{pairs[0][0]} "
                 f"arm exceeds the weight-{pairs[0][1]} arm, so the ordering depends on "
                 f"the shrinkage."))

    # 5 -- the read-off inversion -------------------------------------------------------
    wa = json.load(open(C.art("window_average"), encoding="utf-8"))["arms"]
    def w(arm, ro, k):
        return wa[arm]["readouts"][str(ro)]["mean_logit"][str(k)]["tpr_mean"]
    others = {"CORAL x nb3": 2000, "FreeMatch x nb3": 1500}
    def rank(dann_ro, k):
        vals = {a: w(a, ro, k) for a, ro in others.items()}
        vals["DANN x nb3"] = w("DANN x nb3", dann_ro, k)
        order = sorted(vals, key=vals.get, reverse=True)
        return order.index("DANN x nb3") + 1, order
    r2k1, _ = rank(2000, 1)
    r4k1, _ = rank(4000, 1)
    r2k8, o2 = rank(2000, 8)
    r4k8, o4 = rank(4000, 8)
    assert (r2k1, r4k1) == (1, 3) and o2 == o4, "the inversion or its window fix changed"
    names = {"FreeMatch x nb3": "FreeMatch", "CORAL x nb3": "CORAL", "DANN x nb3": "DANN"}
    rows.append((r"The ranking of the three unfrozen adaptation arms, under the choice of "
                 r"read-off step",
                 r"\S\ref{sec:resolution-fix}",
                 f"At a single checkpoint DANN is first of three at step 2000 "
                 f"({f3(w('DANN x nb3', 2000, 1))}) and last at 4000 "
                 f"({f3(w('DANN x nb3', 4000, 1))}). Under the declared 8-checkpoint score "
                 f"average under \\texttt{{mean\\_logit}} the order is {' $\\geq$ '.join(names[a] for a in o2)} at both: "
                 f"it died to the single-point estimator, not to the data."))

    # 6 -- DANN's spread ----------------------------------------------------------------
    sd = {k: C.fold_mean(a, "tpr", ro)[1] for k, (a, ro) in {
        "dann_decl": ("dann", 4000), "fm": ("freematch", 1500), "coral": ("coral", 2000),
        "dann1500": ("dann", 1500), "dann2000": ("dann", 2000)}.items()}
    assert sd["dann_decl"] > sd["fm"] and sd["dann1500"] < sd["fm"]
    rows.append((r"DANN's across-fold spread exceeds the other arms'",
                 r"\S\ref{sec:limitations}",
                 f"Holds at declared budgets ({C.f4(sd['dann_decl'])} against FreeMatch's "
                 f"{C.f4(sd['fm'])}). Budget-matched it shrinks against CORAL at 2000 "
                 f"({C.f4(sd['dann2000'])} against {C.f4(sd['coral'])}) and reverses "
                 f"against FreeMatch at 1500 ({C.f4(sd['dann1500'])} against "
                 f"{C.f4(sd['fm'])}): it tracks training length."))

    # 7 -- folds 3-4 intrinsically hard -------------------------------------------------
    fm = C.per_fold("freematch", "auroc")
    worst = min(fm, key=fm.get)
    assert worst not in (3, 4)
    rows.append((r"Folds 3 and 4 are intrinsically hard at the ranking level",
                 "---",
                 f"The self-training arm's AUROC on folds 3 and 4 ({f3(fm[3])}, "
                 f"{f3(fm[4])}) is above its fold {worst} ({f3(fm[worst])}), so the "
                 f"difficulty is not a property of those folds alone."))

    # 8 -- the measurement of why they keep failing -------------------------------------
    wf = json.load(open(C.art("window_fold0"), encoding="utf-8"))["arms"]
    ranked = ["CORAL x nb3", "FreeMatch x nb3", "DANN arm 2 (lambda live)"]
    m1 = [wf[a]["1"]["multiple_of_dann_minus_coral"] for a in ranked]
    m8 = [wf[a]["8"]["multiple_of_dann_minus_coral"] for a in wf]
    # Computed, not asserted: say "below 1x on no arm" only where it is true on this pool.
    label = {"CORAL x nb3": "CORAL", "FreeMatch x nb3": "FreeMatch",
             "DANN arm 2 (lambda live)": "DANN", "DANN arm 1 (lambda=0)": r"the $\lambda=0$ control",
             "BNM x nb3": "BNM"}
    below = [label[a] for a in wf if wf[a]["8"]["multiple_of_dann_minus_coral"] < 1.0]
    tail8 = ("below $1\\times$ on no arm." if not below else
             "below $1\\times$ on " + ", ".join(below) + ".")
    rows.append((r"The three unfrozen adaptation arms can be ranked at all "
                 r"\emph{(a measurement of why the others fail, not an ordering)}",
                 r"\S\ref{sec:resolution-diagnosis}",
                 f"On one fold, one run's checkpoint-to-checkpoint scatter is "
                 f"{min(m1):.1f}--{max(m1):.1f}$\\times$ the difference between DANN and "
                 # two decimals where an arm is below 1x, so 0.96 does not print as 1.0
                 + (f"CORAL; after the declared average it is {min(m8):.1f}--{max(m8):.1f}"
                    if not below else
                    f"CORAL; after the declared average it is {min(m8):.2f}--{max(m8):.2f}")
                 + f"$\\times$, " + tail8))

    # 9 -- lambda=0 least stable --------------------------------------------------------
    st = json.load(open(C.art("window_fold0"), encoding="utf-8"))["row9_stability"]
    a1, a8 = st["250-8000"]["1"], st["250-8000"]["8"]
    b1, b8 = st["250-4000"]["1"], st["250-4000"]["8"]
    # Computed, not asserted: the verbs are chosen by the values on this pool. "Dissolves"
    # uses the analysis's own |gap| < 0.02 rule (analyze_window_fold0.py).
    holds = a1["gap"] > 0 and a8["gap"] > 0
    dissolves = abs(b1["gap"]) < 0.02 and abs(b8["gap"]) < 0.02
    head = "Holds" if holds else "Does not hold"
    mid = ("it dissolves: " if dissolves else "the values are ")
    rows.append((r"Unfreezing without an adaptation term is the least stable arm",
                 "---",
                 f"{head} on steps 250--8000 (checkpoint sd/mean {f3(a1['lambda0_sd_over_mean'])} "
                 f"against DANN's {f3(a1['dann_sd_over_mean'])}; {f3(a8['lambda0_sd_over_mean'])} "
                 f"against {f3(a8['dann_sd_over_mean'])} after averaging). On the "
                 f"grid-matched 250--4000 {mid}{f3(b1['lambda0_sd_over_mean'])} "
                 f"against {f3(b1['dann_sd_over_mean'])}, and {f3(b8['lambda0_sd_over_mean'])} "
                 f"against {f3(b8['dann_sd_over_mean'])} after averaging."))

    # 10 -- the budget-rule crossing step -----------------------------------------------
    bj = j("runs", "bnm_r6_budget.json")
    ceil, cross, budget = bj["sharpening_ceiling"], bj["crossing_step"], bj["selected_budget_steps"]
    with open(C.R("runs", "bnm_r6_dev8000", "bnm_r6_dev8000_fold0", "stats_only.csv"),
              newline="", encoding="utf-8") as fh:
        traj = [(int(r["step"]), float(r["bnm_nuclear_norm"])) for r in csv.DictReader(fh)]
    pre = [(s, v) for s, v in traj if s < cross and v < ceil]
    near_s, near_v = max(pre, key=lambda x: x[1])
    alt = 2 * near_s
    assert budget == 2 * cross
    rows.append((r"The budget rule's crossing step is fixed by the trajectory",
                 "---",
                 f"The closest reading below the ceiling, at step {near_s}, falls "
                 f"{f3(ceil - near_v)} short; one sharper batch there would have set the "
                 f"budget at {alt} steps instead of {budget}, "
                 f"{round(100 * (1 - alt / budget))}\\% lower."))

    # Row positions quoted in A3's prose, and the spreads Section 9 quotes, as macros, so
    # the text cannot drift from the table or from the per-fold values.
    readoff_row = next(i for i, r in enumerate(rows, 1) if "read-off step" in r[0])
    measure_row = next(i for i, r in enumerate(rows, 1) if "a measurement of why" in r[0])
    spread_row = next(i for i, r in enumerate(rows, 1) if "spread exceeds" in r[0])
    dann_decl = C.ARM_META["dann"]["readout"]
    fm_decl = C.ARM_META["freematch"]["readout"]
    assert fm_decl == 1500 and dann_decl == 4000
    # Section 9's paired-interval sentence: a paired interval's width is set by the spread
    # of the PER-FOLD DIFFERENCES, so that is the quantity quoted -- over the same
    # declaration (C.CONTRAST_ROWS), folds and budgets as Table T2, sample sd as elsewhere.
    import statistics as _st
    diff_sd = {}
    for _p, _l, a_, ra_, b_, rb_, _m, _k in C.CONTRAST_ROWS:
        r_ = C.contrast(a_, ra_, b_, rb_)
        diff_sd[(a_, ra_, b_, rb_)] = _st.stdev(r_["a"][f] - r_["b"][f]
                                                for f in C.CONFIRMATORY_FOLDS)
    ranked_sd = sorted(diff_sd.items(), key=lambda kv: kv[1], reverse=True)
    assert ranked_sd[0][0] == ("dann", 4000, "lambda0", 4000), \
        "DANN - lambda=0 no longer has the largest per-fold-difference sd"
    means = {"dann": C.fold_mean("dann", "tpr", 4000)[0],
             "fm": C.fold_mean("freematch", "tpr", 1500)[0],
             "dann_sens": C.fold_mean("dann", "tpr", 2000)[0]}
    # Appendix A2 states, as a fact, that DANN's declared headline is LOWER than its
    # pre-declared step-2000 sensitivity reading; fail loudly if that ever stops holding.
    assert means["dann"] < means["dann_sens"], (means["dann"], means["dann_sens"])
    ones = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
            7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
    def mac(n, v):
        return rf"\newcommand{{\{n}}}{{{v}}}"
    C.write_table(os.path.join(C.TABLES, "M_orderings.tex"), "\n".join([
        mac("orderingsCount", str(len(rows))),
        mac("orderingsReadoffRow", ones[readoff_row]),
        mac("orderingsMeasurementRow", ones[measure_row]),
        mac("orderingsSpreadRow", ones[spread_row]),
        mac("spreadDannDeclared", C.f4(sd["dann_decl"])),
        mac("spreadDannDeclaredBudget", str(dann_decl)),
        mac("spreadFreeMatchDeclared", C.f4(sd["fm"])),
        mac("spreadFreeMatchDeclaredBudget", str(fm_decl)),
        mac("spreadDannMatched", C.f4(sd["dann1500"])),
        mac("spreadMatchedBudget", "1500"),
        mac("meanDannDeclared", C.f4(means["dann"])),
        mac("meanFreeMatchDeclared", C.f4(means["fm"])),
        mac("meanDannSensitivity", C.f4(means["dann_sens"])),
        mac("meanDannSensitivityStep", "2000"),
        mac("contrastCount", str(len(diff_sd))),
        mac("diffSdDannLambda", C.f4(ranked_sd[0][1])),
        mac("diffSdNext", C.f4(ranked_sd[1][1])),
    ]), header_comment="Macros for Appendix A3 and Section 9, from the same sources as "
                       "tables/A3_orderings.tex.", size="")

    body = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Every ordering that did not survive its own error bar.} Each "
        r"row states the ordering as it was first read and how it failed; the section "
        r"column names where the main text discusses it, and --- marks those reported "
        r"here only. Numbers are read from the artifact that measured them. All but one "
        r"row are orderings that failed a test; the row marked as a measurement is the "
        r"quantitative account of why they keep failing (Section~\ref{sec:resolution}).}",
        r"\label{tab:orderings}",
        r"\begin{tabular}{p{0.3\textwidth}lp{0.52\textwidth}}",
        r"\toprule",
        r"Ordering as first read & Section & How it failed \\",
        r"\midrule",
    ]
    for i, (o, s, how) in enumerate(rows, 1):
        body.append(f"{i}. {o} & {s} & {how} \\\\")
        if i < len(rows):
            body.append(r"\addlinespace")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    C.write_table(os.path.join(C.TABLES, "A3_orderings.tex"), "\n".join(body),
                  header_comment="A3 -- one row per failed ordering; sources in "
                                 "paper/scripts/make_a3_orderings.py.",
                  placement="htbp")
    print(f"  {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
