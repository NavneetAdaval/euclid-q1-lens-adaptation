r"""
make_orderings_v3.py -- the paper's Table 11 (Appendix C), self-contained (rewrite v3).
=====================================================================================

    python paper/scripts/make_orderings_v3.py

Writes ``paper/tables/AC_orderings_v2.tex`` (the file Appendix C inputs, label
``tab:orderings2``) and ``paper/tables/M_orderings_v3.tex`` (``\contrastCount``), and nothing
else. Until rewrite v3 the table was built by running the protected ``make_a3_orderings.py``
into a temporary folder and editing its rows. That generator reads ``runs/fold_average.json``
for its row 1, which averages the development fold with the abandoned FreeMatch fold-1 run, and
which the public release excludes. This generator never reads that file, directly or through
another script: it re-reads, here, the same sources the protected generator reads for the rows
the paper keeps, with the same arithmetic and the same assertions.

  row 1  budget peak            audit/curation_fold1/out/fold0_only.json (via \rwFz* macros,
                                make_rewrite_v2_macros.py, which re-checks its input hashes)
  row 2  decay onset            runs/devmax_truncation.json  (fold0_development.per_cap)
  row 3  agreement -> shrinkage runs/coral_r4_mechanism.json
  row 4  Fisher-ratio order     runs/coral_r4_fisher_ci.json
  row 5  read-off inversion     window_average (C.art: paper/data/window_average_D1.json)
  row 6  DANN's spread          per-fold TPR through _common (pool D1)
  row 7  folds 3-4 hard         per-fold AUROC through _common (pool D1)
  row 8  lambda=0 least stable  window_fold0 (C.art: paper/data/window_fold0_D1.json)
  row 9  budget-rule crossing   runs/bnm_r6_budget.json + the development run's stats_only.csv

When the current ``AC_orderings_v2.tex`` exists, the number tokens of rows 2-9 are asserted
identical to it before it is overwritten. The protected generator and its outputs
(``A3_orderings.tex``, ``M_orderings.tex``) are left as they are; ``verify_orig_tables.py``
still regenerates and checks them.
"""
from __future__ import annotations

import csv
import json
import os
import re

import _common as C

NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\{,\}\d{3})*(?:\.\d+)?")
OUT = os.path.join(C.TABLES, "AC_orderings_v2.tex")
# rewrite v4 (D4): the grid-matched k=1 pair leaves row 8's sentence
DROPPED_V4_ROW8 = ["0.246", "0.225"]


def j(*p):
    return json.load(open(C.R(*p), encoding="utf-8"))


def f3(x, signed=False):
    return f"${x:+.3f}$" if signed else f"${x:.3f}$"


def rows_2_to_9():
    rows = []
    # 2 -- decay onset
    pc = j("runs", "devmax_truncation.json")["fold0_development"]["per_cap"]
    caps = sorted(int(c) for c in pc)
    hit = [c for c in caps if pc[str(c)]["primary"]["detected"]]
    assert hit == [caps[-1]], f"decay detected at {hit}, not only at the cap"
    lo_c, hi_c = caps[0], caps[-2]
    rows.append((r"Decay sets in before the 8000-step cap", r"\S\ref{sec:negative}",
                 f"Detectable only at the full cap of {caps[-1]}; every shorter "
                 f"truncation ({lo_c}--{hi_c}) has an interval that crosses zero."))
    # 3 -- agreement at init predicts CORAL's shrinkage
    arms = j("runs", "coral_r4_mechanism.json")["arms"]
    r16, r0 = arms["g16"]["spearman_primary"], arms["g0"]["spearman_primary"]
    p16 = arms["g16"]["partial_spearman_primary_ctrl_logs0"]
    assert abs(r0) > abs(r16), "the known-null is no longer the stronger correlation"
    rows.append((r"Source--target agreement at initialisation predicts which feature "
                 r"directions the alignment objective shrinks", "---",
                 f"Spearman $\\rho = {r16:.2f}$ at the weight used, but ${r0:.2f}$ at "
                 f"weight zero, where the objective contributes no gradient; controlling "
                 f"for initial scale leaves $\\rho = {p16:.3f}$."))
    # 4 -- Fisher-ratio ordering vs shrinkage
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
    rows.append((r"A Fisher-ratio separability measure rises with the alignment weight", "---",
                 f"Monotone in the weight at whitening shrinkage "
                 f"{' and '.join(str(l) for l in ok)}; at {lam_b} the weight-{pairs[0][0]} "
                 f"arm exceeds the weight-{pairs[0][1]} arm, so the ordering depends on "
                 f"the shrinkage."))
    # 5 -- the read-off inversion (rewrite v2 wording)
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
    _r2k8, o2 = rank(2000, 8)
    _r4k8, o4 = rank(4000, 8)
    assert (r2k1, r4k1) == (1, 3) and o2 == o4, "the inversion or its window fix changed"
    names = {"FreeMatch x nb3": "FreeMatch", "CORAL x nb3": "CORAL", "DANN x nb3": "DANN"}
    rows.append((r"The ranking of the three unfrozen adaptation arms, under the choice of "
                 r"read-off step", r"App.~\ref{sec:resolution-fix}",
                 f"At a single checkpoint DANN is first of three at step 2000 "
                 f"({f3(w('DANN x nb3', 2000, 1))}) and last at 4000 "
                 f"({f3(w('DANN x nb3', 4000, 1))}). Under the declared 8-checkpoint score "
                 f"average of logits the order is {' $\\geq$ '.join(names[a] for a in o2)} "
                 f"at both, so the inversion came from reading a single checkpoint."))
    # 6 -- DANN's spread
    sd = {k: C.fold_mean(a, "tpr", ro)[1] for k, (a, ro) in {
        "dann_decl": ("dann", 4000), "fm": ("freematch", 1500), "coral": ("coral", 2000),
        "dann1500": ("dann", 1500), "dann2000": ("dann", 2000)}.items()}
    assert sd["dann_decl"] > sd["fm"] and sd["dann1500"] < sd["fm"]
    # rewrite v4 (A7): Section 8 no longer discusses DANN's spread, so the section column is ---
    rows.append((r"DANN's across-fold spread exceeds the other arms'", "---",
                 f"Holds at declared budgets ({C.f4(sd['dann_decl'])} against FreeMatch's "
                 f"{C.f4(sd['fm'])}). Budget-matched it shrinks against CORAL at 2000 "
                 f"({C.f4(sd['dann2000'])} against {C.f4(sd['coral'])}) and reverses "
                 f"against FreeMatch at 1500 ({C.f4(sd['dann1500'])} against "
                 f"{C.f4(sd['fm'])}): it tracks training length."))
    # 7 -- folds 3-4 intrinsically hard
    fm = C.per_fold("freematch", "auroc")
    worst = min(fm, key=fm.get)
    assert worst not in (3, 4)
    rows.append((r"Folds 3 and 4 are intrinsically hard at the ranking level", "---",
                 f"The self-training arm's AUROC on folds 3 and 4 ({f3(fm[3])}, "
                 f"{f3(fm[4])}) is above its fold {worst} ({f3(fm[worst])}), so the "
                 f"difficulty is not a property of those folds alone."))
    # 8 -- lambda=0 least stable (the protected generator's row 9)
    st = json.load(open(C.art("window_fold0"), encoding="utf-8"))["row9_stability"]
    a1, a8 = st["250-8000"]["1"], st["250-8000"]["8"]
    b1, b8 = st["250-4000"]["1"], st["250-4000"]["8"]
    # rewrite v4 (D4): the outcome stated plainly. The wording holds only for this pattern:
    # the control less stable on 250-8000 at both k, and more stable than DANN on the
    # grid-matched 250-4000 after averaging (recomputed independently in the v4 report).
    assert a1["gap"] > 0 and a8["gap"] > 0 and b8["gap"] < 0, (a1["gap"], a8["gap"], b8["gap"])
    rows.append((r"Unfreezing without an adaptation term is the least stable arm", "---",
                 f"Holds on steps 250--8000 ({f3(a1['lambda0_sd_over_mean'])} against DANN's "
                 f"{f3(a1['dann_sd_over_mean'])}; {f3(a8['lambda0_sd_over_mean'])} against "
                 f"{f3(a8['dann_sd_over_mean'])} after averaging) but not on the grid-matched "
                 f"250--4000 after averaging ({f3(b8['lambda0_sd_over_mean'])} against "
                 f"{f3(b8['dann_sd_over_mean'])}), so the order depends on the step range."))
    # 9 -- the budget-rule crossing step (the protected generator's row 10)
    bj = j("runs", "bnm_r6_budget.json")
    ceil, cross, budget = bj["sharpening_ceiling"], bj["crossing_step"], bj["selected_budget_steps"]
    with open(C.R("runs", "bnm_r6_dev8000", "bnm_r6_dev8000_fold0", "stats_only.csv"),
              newline="", encoding="utf-8") as fh:
        traj = [(int(r["step"]), float(r["bnm_nuclear_norm"])) for r in csv.DictReader(fh)]
    pre = [(s, v) for s, v in traj if s < cross and v < ceil]
    near_s, near_v = max(pre, key=lambda x: x[1])
    alt = 2 * near_s
    assert budget == 2 * cross
    rows.append((r"The budget rule's crossing step is fixed by the trajectory", "---",
                 f"The closest reading below the ceiling, at step {near_s}, falls "
                 f"{f3(ceil - near_v)} short; one sharper batch there would have set the "
                 f"budget at {alt} steps instead of {budget}, "
                 f"{round(100 * (1 - alt / budget))}\\% lower."))
    return rows


ROW1 = (r"The budget at which the self-training arm's curve peaks", r"\S\ref{sec:negative}",
        r"On the development fold the peak cannot be located: the steps within two positives "
        r"of the best are \rwFzSteps{}, and \rwFzWithinSd{} of \rwFzPoints{} checkpoints lie "
        r"within one sd of it.")


def row_tokens(text):
    """{row number: number tokens of that row}, from a written table body."""
    out = {}
    for line in text.split("\n"):
        m = re.match(r"^(\d+)\. ", line)
        if m:
            out[int(m.group(1))] = NUM.findall(line[m.end():])
    return out


def main() -> int:
    assert C.POOL == "D1"
    rows = [ROW1] + rows_2_to_9()
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{\textbf{Orderings that did not survive their own error bars.} Each row "
             r"gives an ordering as we first read it and how it failed. The section column "
             r"points to where it is discussed; --- marks those reported here only. Rows 1 and "
             r"5--8 are computed on the cleaned test pool and row 2 on the original test pool "
             r"(Section~\ref{sec:testpool}); rows 3, 4 and 9 use no test data.}",
             r"\label{tab:orderings2}",
             r"\begin{tabular}{p{0.3\textwidth}lp{0.52\textwidth}}", r"\toprule",
             r"Ordering as first read & Section & How it failed \\", r"\midrule"]
    for i, (o, s, how) in enumerate(rows, 1):
        lines.append(f"{i}. {o} & {s} & {how} \\\\")
        if i < len(rows):
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    new = "\n".join(lines)
    checked = "no current file to compare"
    if os.path.exists(OUT):
        old = row_tokens(open(OUT, encoding="utf-8").read())
        mine = row_tokens(new)
        assert sorted(old) == sorted(mine) == list(range(1, 10)), (sorted(old), sorted(mine))
        for k in range(2, 10):
            if k == 8 and old[k] != mine[k]:
                # rewrite v4 (D4): row 8 drops the k=1 grid-matched pair; every other token
                # must be carried over unchanged and in order
                kept = [t for t in old[k]]
                for t in DROPPED_V4_ROW8:
                    kept.remove(t)
                assert mine[k] == kept, f"row 8: {old[k]} -> {mine[k]}"
                continue
            assert old[k] == mine[k], f"row {k}: number tokens differ: {old[k]} vs {mine[k]}"
        checked = ("rows 2-9 number tokens identical to the current file"
                   + ("" if old[8] == mine[8] else
                      f", except row 8's declared drop of {DROPPED_V4_ROW8}"))
    C.write_table(OUT, new,
                  header_comment="Appendix C, Table 11 (rewrite v3): make_orderings_v3.py, "
                                 "self-contained; reads no fold_average.json.",
                  placement="htbp")
    n = len(C.CONTRAST_ROWS)
    C.write_table(os.path.join(C.TABLES, "M_orderings_v3.tex"),
                  rf"\newcommand{{\contrastCount}}{{{n}}}",
                  header_comment="The contrast count (rewrite v3), len(_common.CONTRAST_ROWS); "
                                 "replaces M_orderings.tex in main.tex.", size="")
    print(f"  wrote AC_orderings_v2.tex ({len(rows)} rows; {checked}) and M_orderings_v3.tex "
          f"(contrastCount {n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
