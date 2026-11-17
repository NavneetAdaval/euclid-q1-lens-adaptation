r"""
make_prereg_fixes.py -- the numbers the pre-registration fixes add (PROJECTv7 7.65).
====================================================================================

    python paper/scripts/make_prereg_fixes.py

Writes ``paper/tables/M_prereg.tex`` (macros) and ``paper/tables/C_window_secondary.tex``
(Appendix C's window secondaries) and nothing else. Every value is read from a stored file or
computed from stored logits with the paper's own functions (``_common``, ``metrics``); nothing
is trained, and nothing is typed. ``crosscheck.build_prereg_fixes`` recomputes each one
independently.

  * DANN's five-fold means (fold 0 included), cleaned pool, at 4000 and 2000 -- the DANN budget
    file's declaration says "5-fold mean" while the same file excludes fold 0 from every
    headline (verification item 2);
  * FreeMatch's adaptive threshold: the first fold-0 step at which tau_global has covered 90% of
    its travel, from the run's diagnostics.csv, by the definition of
    scripts/analyze_turn_timing.describe (asserted equal to runs/turn_timing.json where that
    file exists; it is not in the public release) (item 4);
  * BNM's two carried-over predictions: the decay statistic from
    runs/bnm_r6_budget_analysis.json task1, and the predicted-positive fraction from
    runs/bnm_r6_folds/*/trace.csv (item 5);
  * BNM's step-zero gate as run (item 16a) and its budget preview (item 16d);
  * the window declaration's secondaries, its sqrt(k) scatter test and the ordering verdicts,
    from paper/data/window_average_D1.json (and runs/round7/window_average.json for the
    original pool) (item 9);
  * the lambda=0 control's fold-4 completeness at 24% purity, asserted to be one lens above
    every negative (G2).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics as st
import sys

import numpy as np

import _common as C

sys.path.insert(0, C.REPO_SCRIPTS)
import metrics as me                                            # noqa: E402

DANN_F0 = "runs/dann_arm2_fold0/rescore/dann_r5_nb3_fold0/logits/dann_nb3_fold0/step_{}.npz"
FM_DIAG = "runs/freematch_fold0_nb3_unfrozen/diagnostics.csv"
TURN = "runs/turn_timing.json"
BNM_PREREG = "runs/bnm_r6_prereg.json"
BNM_BUDGET = "runs/bnm_r6_budget.json"
BNM_ANALYSIS = "runs/bnm_r6_budget_analysis.json"
BNM_FOLD = "runs/bnm_r6_folds/bnm_r6_fold{}"
LATE_FROM = 3000            # "about 0.34": the mean over the logged steps after this one
WIN_D1 = "window_average_D1.json"
WIN_ORIG = "runs/round7/window_average.json"
WORD = {1: "one", 2: "two", 3: "three"}
WORD10 = {10: "ten"}
PART2 = "runs/dilute10_prereg_part2.json"
PART3 = "runs/dilute10_prereg_part3.json"
CORAL_CONFIRM = "scripts/run_coral_r4_confirm.py"   # step0_gate unchanged by its later half-edit


def J(rel):
    return json.load(open(C.R(rel), encoding="utf-8"))


def dann_five_fold():
    """{step: five-fold mean TPR} on the pool _common selects: fold 0 from its stored logits
    through _common._pooled_npz + metric_suite, folds 1-4 through _common.fold_mean."""
    out = {}
    for step in (4000, 2000):
        pos, neg = C._pooled_npz(C.R(DANN_F0.format(step)), 0)
        y = np.r_[np.ones(pos.size, int), np.zeros(neg.size, int)]
        t0 = float(me.metric_suite(y, np.r_[pos, neg], target_fpr=C.TARGET_FPR)["tpr"])
        _m, _sd, v = C.fold_mean("dann", "tpr", step)
        assert sorted(v) == [1, 2, 3, 4]
        out[step] = (t0 + sum(v.values())) / 5
    return out


def tau_ninety():
    """First step at which tau_global_last has covered 90% of its first-to-last travel."""
    rows = [r for r in csv.DictReader(open(C.R(FM_DIAG), encoding="utf-8"))
            if r["tau_global_last"]]
    s = [int(r["step"]) for r in rows]
    v = [float(r["tau_global_last"]) for r in rows]
    assert v[-1] > v[0]
    target = v[0] + 0.90 * (v[-1] - v[0])
    step = next(a for a, b in zip(s, v) if b >= target)
    if os.path.isfile(C.R(TURN)):
        assert J(TURN)["0"]["label_free"]["tau_global"]["saturation_step_90pct"] == step
    return step


def bnm_facts():
    pre = J(BNM_PREREG)
    p = pre["predictions_from_7_20"]
    assert p["carried_verbatim"] and not p["amended"] and p["scored_not_revised"]
    m = re.search(r"from (0\.\d+) toward ~(0\.\d+)", p["prediction_2_collapse"])
    assert m, p["prediction_2_collapse"]
    start, target = float(m.group(1)), float(m.group(2))
    t1 = J(BNM_ANALYSIS)
    assert t1["n_pos"] == 98 and t1["n_neg"] == 26399          # the original test pool, fold 0
    t1 = t1["task1"]
    assert t1["cap"] == 8000 and t1["verdict"] == "not distinguishable" and not t1["detected"]
    assert t1["delta"] < 0 < t1["ci"][1] and t1["ci"][0] < 0    # opposite sign, undetected
    # predicted-positive fraction on the target batches, folds 1-4
    first, late, allv = set(), [], []
    for f in (1, 2, 3, 4):
        rows = list(csv.DictReader(open(C.R(BNM_FOLD.format(f), "trace.csv"), encoding="utf-8")))
        steps = [int(r["step"]) for r in rows]
        vals = [float(r["bnm_pred_pos_frac"]) for r in rows]
        first.add(steps[0])
        assert vals[0] > start, (f, vals[0])                    # risen by the first logged step
        assert all(0.01 < x < 0.99 for x in vals)               # no collapse to 0 or 1
        late += [x for a, x in zip(steps, vals) if a > LATE_FROM]
        allv += vals
    assert len(first) == 1
    # step-zero gate as run
    tpr_d, au_d, au_tol, tpr_tol_pos = [], [], set(), set()
    for f in (1, 2, 3, 4):
        g = J(BNM_FOLD.format(f) + "/run.json")["step0_gate"]
        assert g["verdict"] == "PASS"
        tpr_d.append(g["tpr_abs_diff"])
        au_d.append(g["auroc_abs_diff"])
        au_tol.add(g["auroc_tolerance"])
        tpr_tol_pos.add(round(g["tpr_tolerance"] * g["n_test_positives"], 9))
    assert pre["step0_gate"]["required_abs_diff"] == 0.0
    assert all(x == 0.0 for x in tpr_d) and au_tol == {0.001} and tpr_tol_pos == {1.0}
    bound = 10 ** math.floor(math.log10(max(au_d)))
    mult = math.ceil(max(au_d) / bound)
    assert max(au_d) < mult * bound
    bud = J(BNM_BUDGET)
    assert pre["budget"]["stage2_preview_budget"] == bud["stage2_preview_was"]
    return {"start": start, "target": target, "t1": t1, "first": first.pop(),
            "late": st.mean(late), "min": min(allv), "max": max(allv),
            "gate_mult": mult, "gate_exp": int(round(math.log10(bound))),
            "preview": bud["stage2_preview_was"], "budget": bud["selected_budget_steps"]}


ROWS = [("CORAL", "CORAL x nb3", "2000", ""), ("FreeMatch", "FreeMatch x nb3", "1500", ""),
        ("DANN", "DANN x nb3", "4000", ""), ("DANN", "DANN x nb3", "2000", r"\,\footnotesize(sens.)"),
        (r"$\lambda=0$", "lambda=0", "4000", ""), ("BNM", "BNM x nb3", "4000", "")]


def window_cell(w, arm, ro, comb, k, metric):
    """Fold-mean of a per-fold window value; n/a if any fold is undefined."""
    r = w["arms"][arm]["readouts"][ro][comb][str(k)]
    v = r[f"{metric}_per_fold"]
    assert len(v) == 4
    if any(x is None or not math.isfinite(x) for x in v):
        return float("nan")
    return st.mean(v)


def window_table(w):
    body = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{The window average's secondary metrics.} Fold-mean AUROC and "
        r"completeness at 24\% purity, folds 1--4, cleaned pool, for a single checkpoint "
        r"($k=1$, each arm's headline) and for the declared eight-checkpoint average under "
        r"each score combination. The declaration named both metrics as secondaries "
        r"(Table~\ref{tab:window2} gives TPR). \emph{n/a}: 24\% purity is unreachable on at "
        r"least one fold.}",
        r"\label{tab:window-secondary}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{AUROC} & \multicolumn{3}{c}{completeness at 24\% purity} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Arm & Read at & $k{=}1$ & $k{=}8$, logit & $k{=}8$, rank & $k{=}1$ & $k{=}8$, logit "
        r"& $k{=}8$, rank \\",
        r"\midrule",
    ]
    cells = {}
    for name, arm, ro, tag in ROWS:
        row = [name + tag, ro]
        for metric in ("auroc", "comp_at_purity"):
            k1 = window_cell(w, arm, ro, "mean_logit", 1, metric)
            assert (math.isnan(k1) and math.isnan(window_cell(w, arm, ro, "mean_rank", 1, metric))
                    ) or k1 == window_cell(w, arm, ro, "mean_rank", 1, metric)
            for comb, k in (("mean_logit", 1), ("mean_logit", 8), ("mean_rank", 8)):
                v = k1 if k == 1 else window_cell(w, arm, ro, comb, k, metric)
                cells[(arm, ro, metric, comb, k)] = v
                row.append(C.f4(v))
        body.append(" & ".join(row) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body), cells


def build():
    out = []

    def add(name, val):
        assert re.fullmatch(r"[A-Za-z]+", name), name
        out.append((name, val))

    five = dann_five_fold()
    add("dDannFiveFoldFourK", f"{five[4000]:.2f}")
    add("dDannFiveFoldTwoK", f"{five[2000]:.2f}")
    add("fmTauNinetyStep", str(tau_ninety()))

    b = bnm_facts()
    t1 = b["t1"]
    add("bnmDecayDelta", f"{t1['delta']:+.3f}")
    add("bnmDecayLo", f"{t1['ci'][0]:+.3f}")
    add("bnmDecayHi", f"{t1['ci'][1]:+.3f}")
    add("bnmDecayCap", str(t1["cap"]))
    add("bnmPredPosStart", f"{b['start']:.2f}")
    add("bnmPredPosTarget", f"{b['target']:g}")
    add("bnmPredPosFirstStep", str(b["first"]))
    add("bnmPredPosLate", f"{b['late']:.2f}")
    add("bnmPredPosLateFrom", str(LATE_FROM))
    add("bnmPredPosMin", f"{b['min']:.2f}")
    add("bnmPredPosMax", f"{b['max']:.2f}")
    add("bnmGateAurocTol", r"10^{-3}")
    add("bnmGateAurocBound", rf"{b['gate_mult']} \times 10^{{{b['gate_exp']}}}")
    add("bnmBudgetPreview", str(b["preview"]))
    add("bnmBudgetSteps", str(b["budget"]))

    # ---- the window declaration (paper/data/window_average_D1.json) ----
    w = json.load(open(os.path.join(C.DATA, WIN_D1), encoding="utf-8"))
    assert w["test_pool"]["pool"] == "D1"
    sr = w["scatter_reduction"]
    lam8 = sr["lambda=0"]["8"]["achieved_over_expected"]
    others8 = [sr[a]["8"]["achieved_over_expected"] for a in ("FreeMatch x nb3", "DANN x nb3",
                                                              "BNM x nb3")]
    assert "8" not in sr["CORAL x nb3"] and max(int(k) for k in sr["CORAL x nb3"]) == 4
    add("winSqrtkLam", f"{lam8:.2f}")
    add("winSqrtkMin", f"{min(others8):.2f}")
    add("winSqrtkMax", f"{max(others8):.2f}")
    add("winSqrtkCoral", f"{sr['CORAL x nb3']['4']['achieved_over_expected']:.2f}")
    ov = w["ordering_verdicts"]
    r5 = ov["row5_readoff_inversion"]
    assert r5["1"]["dann_position_flips"] and not r5["8"]["dann_position_flips"]
    r6 = ov["row6_dann_spread"]["8"]
    assert r6["dann_is_widest"]
    sd6 = r6["across_fold_sd"]
    runner = max((v, k) for k, v in sd6.items() if k != "DANN@4000")
    assert runner[1] == "CORAL@2000"
    add("winSixDann", f"{sd6['DANN@4000']:.3f}")
    add("winSixCoral", f"{runner[0]:.3f}")
    r6o = J(WIN_ORIG)["ordering_verdicts"]["row6_dann_spread"]["8"]
    assert not r6o["dann_is_widest"] and r6o["widest"] == "BNM@4000"
    add("winSixBnmOrig", f"{r6o['across_fold_sd']['BNM@4000']:.3f}")
    add("winSixDannOrig", f"{r6o['across_fold_sd']['DANN@4000']:.3f}")
    r8 = ov["row8_resolution_argument"]
    add("winEightDiff", f"{r8['between_method_differences_7_18']['DANN - CORAL']:.3f}")
    k8 = r8["by_k"]["8"]
    add("winEightSdLo", f"{k8['within_run_sd_lo']:.3f}")
    add("winEightSdHi", f"{k8['within_run_sd_hi']:.3f}")
    add("winEightMultLo", f"{k8['multiple_of_dann_minus_coral_lo']:.1f}")
    add("winEightMultHi", f"{k8['multiple_of_dann_minus_coral_hi']:.1f}")
    r9 = ov["row9_lambda0_stability"]
    assert not r9["1"]["lambda0_is_widest"] and not r9["8"]["lambda0_is_widest"]
    table, cells = window_table(w)
    # "averaging raises both secondaries for every arm but DANN@2000's completeness" -- asserted
    for _n, arm, ro, _t in ROWS:
        a1 = cells[(arm, ro, "auroc", "mean_logit", 1)]
        a8 = cells[(arm, ro, "auroc", "mean_logit", 8)]
        assert a8 > a1, (arm, ro)

    # ---- the 1:10 pre-registration's parts 2 and 3, and the step-zero gates (items 7, 8) ----
    p2 = J(PART2)
    m = re.search(r"at most (\d+) training steps", p2["scope"])
    assert m, p2["scope"]
    add("dlPartTwoMaxStepsWord", WORD10[int(m.group(1))])
    stop = p2["stop_rules_nb3_coral_freematch"]["only_stop_conditions"][0]
    assert "one positive plus |AUROC diff| <= 1e-3" in stop and "CORAL's original warn-only band" in stop
    add("dlGateAurocTol", r"10^{-3}")
    src = open(C.R(CORAL_CONFIRM), encoding="utf-8").read()
    m = re.findall(r"def step0_gate\(scorer, record, fold, n_pos, tol_abort=(0\.\d+), "
                   r"auroc_tol_abort=(0\.\d+)\):", src)
    assert len(m) == 1 and "WARN, not abort, at the one-positive tolerance" in src, m
    add("coralGateWarnTpr", m[0][0])
    ev = J(PART3)["evidence"]["coral_nb3_300_step_summary"]["per_fold"]
    applied = [ev[str(k)]["applied_updates_of_300"] for k in (1, 2, 3, 4)]
    add("dlCoralCheckSteps", "300")
    add("dlCoralCheckAppliedMin", str(min(applied)))
    add("dlCoralCheckAppliedMax", str(max(applied)))

    # ---- G2: the lambda=0 control's fold-4 completeness is one lens ----
    s = C.suite("lambda0", 4, 4000)
    pos, neg = C.ARMS[C.ARM_KEY["lambda0"]]["load"](4)[4000]
    above = int((pos > neg.max()).sum())
    assert above == 1 and abs(s["comp_at_purity"] - 1 / pos.size) < 1e-12, (above, s)
    add("lamFourCompLenses", WORD[above])

    C.write_table(os.path.join(C.TABLES, "M_prereg.tex"),
                  "\n".join(rf"\newcommand{{\{n}}}{{{v}}}" for n, v in out),
                  header_comment=("The pre-registration fixes' numbers (PROJECTv7 7.65), read "
                                  "from stored files by paper/scripts/make_prereg_fixes.py;\n"
                                  "crosscheck.build_prereg_fixes recomputes each."), size="")
    C.write_table(os.path.join(C.TABLES, "C_window_secondary.tex"), table,
                  header_comment=("Appendix C: the window declaration's secondaries, from "
                                  "paper/data/window_average_D1.json."),
                  size="footnotesize", placement="htbp")
    for n, v in out:
        print(f"  \\{n} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
