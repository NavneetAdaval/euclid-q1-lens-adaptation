r"""
make_a7_prereg_pool.py -- the appendix "Results on the pre-registered test pool".
================================================================================

    python paper/scripts/make_a7_prereg_pool.py

The main text reports every score-based number on the test pool of Section 3, which
excludes from the shared test negatives each object sharing an exact Euclid ID with a
catalogued lens or a simulation host galaxy (``audit/decontam/rule.json``; D1 in
``_common``). This appendix gives the same headline quantities on the pre-registered pool,
where the test negatives are exactly as the splits file draws them, and on D2, which also
drops the test lenses that appear among training or validation negatives or as a
simulation deflector.

Everything is computed here from the stored scores through ``_common.pool_filter``, by
the same ``fold_mean`` / ``contrast`` calls T1 and T2 use; the licensed FreeMatch pair is
read from its pool's own bootstrap artifact (``runs/foldmean_bootstrap.json`` and
``paper/data/foldmean_bootstrap_D1.json``). Nothing is typed.

Writes ``paper/tables/A7_prereg_pool.tex`` (the table) and ``paper/tables/M_pool.tex``
(the macros Sections 3 and 9 use).
"""
from __future__ import annotations

import json
import os

import _common as C

# (label, arm, readout) -- every arm of the core table, DANN at both budgets.
CORE = [
    (r"CORAL $\times$ nb3", "coral", 2000),
    (r"FreeMatch $\times$ nb3", "freematch", 1500),
    (r"DANN $\times$ nb3", "dann", 4000),
    (r"DANN $\times$ nb3, sensitivity readout", "dann", 2000),
    (r"$\lambda=0$ unfrozen control", "lambda0", 4000),
    (r"BNM $\times$ nb3", "bnm", 4000),
    (r"Baseline A (frozen)", "baseline", 8000),
]
POOLS = ("orig", "D1", "D2")


def fm(arm, ro, p):
    with C.pool(p):
        return C.fold_mean(arm, "tpr", ro)[0]


def contains_zero(lo, hi):
    return lo <= 0.0 <= hi


def main() -> int:
    S = C.pool_sets()
    n_removed = len(S["neg_remove"])
    n_d2 = len(S["pos_remove"])

    # Simulation hosts among the negatives OTHER than the test pool, by exact ID (the
    # training, validation and target negatives; bare-deflector negatives are simulation
    # hosts by construction and are not counted). Appendix A6's rule 9 cites this count.
    sp = S["splits"]
    other = set(int(o) for o in sp["source_train"]["neg_random"]["object_ids"])
    other |= set(int(o) for o in sp["source_train"]["neg_mlfp"]["object_ids"])
    other |= set(int(o) for o in sp["val"]["neg_random"]["object_ids"])
    for fd in sp["real_lenses_kfold"]["folds"]:
        other |= set(int(o) for o in fd["target_neg_object_ids"])
    n_defl_other = len(S["defl_used"] & other)
    n_defl_all = len(S["defl_used"] & (other | set(S["loc2id"].values())))
    assert n_defl_all == n_defl_other + S["n_defl_in_test"], (n_defl_all, n_defl_other)

    # The shared test-negative pool as drawn (the splits file) and after the D1 exclusion
    # (Section 3 quotes both, and Figure 1 the second). Every removed object must be in the
    # drawn pool, so the two counts differ by exactly the number removed.
    drawn = [int(o) for o in sp["test_negatives_shared"]["object_ids"]]
    n_testneg_d1 = sum(1 for o in drawn if o not in S["neg_remove"])
    assert len(drawn) - n_testneg_d1 == n_removed == 265, (len(drawn), n_testneg_d1)

    # ---------------- (a) fold-mean TPR under the three pools ----------------
    core = {(arm, ro): {p: fm(arm, ro, p) for p in POOLS} for _, arm, ro in CORE}

    # The largest |D1 -> D2| change over the arms of the core table (T1: each arm at its
    # own read-off step).
    shifts = {arm: abs(core[(arm, C.ARM_META[arm]["readout"])]["D2"]
                       - core[(arm, C.ARM_META[arm]["readout"])]["D1"])
              for arm in C.MAIN_ORDER}
    max_arm = max(shifts, key=shifts.get)
    max_shift = shifts[max_arm]

    body = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Fold-mean TPR@$10^{-3}$ (folds 1--4) and the paired contrasts on the",
        r"pre-registered test pool (test negatives exactly as drawn), on the pool of the",
        r"main text (D1, which excludes the " + str(n_removed) + r" objects of",
        r"Section~\ref{sec:data}), and on D2 (D1 also without the " + str(n_d2)
        + r" test lenses noted in Section~\ref{sec:limitations}).}",
        r"\label{tab:prereg_pool}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\multicolumn{5}{l}{\textit{(a) Fold-mean TPR@$10^{-3}$}} \\",
        r"Arm & Read at & Pre-registered & D1 (main text) & D2 \\",
        r"\midrule",
    ]
    for label, arm, ro in CORE:
        v = core[(arm, ro)]
        body.append(" & ".join([label, str(ro)] + [C.f4(v[p]) for p in POOLS]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"", r"\vspace{6pt}",
             r"\begin{tabular}{>{\TRAG}p{3.0cm}lllc}",
             r"\toprule",
             r"\multicolumn{5}{l}{\textit{(b) Paired contrasts, $\Delta$ TPR@$10^{-3}$ "
             r"and 95\% CI}} \\",
             r"Contrast & Steps & Pre-registered & D1 (main text) & Zero-inclusion \\",
             r"\midrule"]

    same_all = True
    rows_json = []
    zero_notes = []      # a point estimate that rounds to zero: printed 0.0000, footnoted

    def est(x):
        if C.rounds_to_zero(x):
            assert not zero_notes, "a second rounded-to-zero cell needs its own footnote mark"
            zero_notes.append(r"$^{\dagger}$" + C.unrounded(x) + " before rounding.")
            return C.f4(0.0) + r"$^{\dagger}$"
        return C.f4(x, signed=True)

    for panel, label, a, ra, b, rb, match, mark in C.CONTRAST_ROWS:
        cells = {}
        for p in ("orig", "D1"):
            with C.pool(p):
                r = C.contrast(a, ra, b, rb)
            cells[p] = r
        z = [contains_zero(cells[p]["ci_lo"], cells[p]["ci_hi"]) for p in ("orig", "D1")]
        same = z[0] == z[1]
        same_all &= same
        rows_json.append(dict(label=label, orig=[cells["orig"][k] for k in
                                                  ("mean", "ci_lo", "ci_hi")],
                              D1=[cells["D1"][k] for k in ("mean", "ci_lo", "ci_hi")],
                              same_zero_inclusion=same))
        body.append(" & ".join([
            label, f"{ra} / {rb}",
            est(cells["orig"]["mean"]) + " "
            + C.ci(cells["orig"]["ci_lo"], cells["orig"]["ci_hi"]),
            est(cells["D1"]["mean"]) + " "
            + C.ci(cells["D1"]["ci_lo"], cells["D1"]["ci_hi"]),
            "same" if same else r"\textbf{differs}",
        ]) + r" \\")

    fb = {"orig": json.load(open(C.ART["foldmean_bootstrap"], encoding="utf-8"))}
    with C.pool("D1"):
        fb["D1"] = json.load(open(C.art("foldmean_bootstrap"), encoding="utf-8"))
    z = [contains_zero(*fb[p]["ci"]) for p in ("orig", "D1")]
    same = z[0] == z[1]
    same_all &= same
    rows_json.append(dict(label="FreeMatch nb3 - FreeMatch frozen (licensed pair)",
                          orig=[fb["orig"]["diff"], *fb["orig"]["ci"]],
                          D1=[fb["D1"]["diff"], *fb["D1"]["ci"]], same_zero_inclusion=same))
    body += [
        r"\midrule",
        " & ".join([
            r"FreeMatch $\times$ nb3 $-$ FreeMatch frozen$^{\ast}$", "1500 / 8000",
            C.f4(fb["orig"]["diff"], signed=True) + " " + C.ci(*fb["orig"]["ci"]),
            C.f4(fb["D1"]["diff"], signed=True) + " " + C.ci(*fb["D1"]["ci"]),
            "same" if same else r"\textbf{differs}",
        ]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{2pt}",
        r"\footnotesize Intervals as in Table~\ref{tab:contrasts}: paired $t$, df $=3$;",
        r"$^{\ast}$fold-aware paired bootstrap, $B=5000$." + "".join(" " + n for n in zero_notes),
        r"\end{table}",
    ]

    C.write_table(os.path.join(C.TABLES, "A7_prereg_pool.tex"), "\n".join(body),
                  header_comment=("A7 -- results on the pre-registered test pool, D1 and D2.\n"
                                  "Stored scores through _common.pool_filter; rule "
                                  + S["rule_sha256"][:16] + "...\n"
                                  "Licensed pair: runs/foldmean_bootstrap.json (orig), "
                                  "paper/data/foldmean_bootstrap_D1.json (D1)."),
                  placement="htbp")

    macros = "\n".join([
        r"% Test-pool macros for Sections 3 and 9, from audit/decontam/rule.json via",
        r"% scripts/exact_ids.contaminant_sets and the stored scores.",
        r"\newcommand{\DOneRemoved}{" + f"{n_removed:,}".replace(",", "{,}") + "}",
        r"\newcommand{\DOneLensNeg}{" + str(S["n_lens_in_test"]) + "}",
        r"\newcommand{\DOneDeflNeg}{" + str(S["n_defl_in_test"]) + "}",
        r"\newcommand{\DTwoPositives}{" + str(n_d2) + "}",
        r"\newcommand{\DeflOtherNeg}{" + str(n_defl_other) + "}",
        r"\newcommand{\DeflAllNeg}{" + str(n_defl_all) + "}",
        r"\newcommand{\DTwoMaxShift}{" + f"{max_shift:.4f}" + "}",
        r"\newcommand{\DTwoMaxShiftArm}{" + C.ARM_META[max_arm]["short"] + "}",
        r"\newcommand{\TestNegDrawn}{" + f"{len(drawn):,}".replace(",", r"\,") + "}",
        r"\newcommand{\TestNegDOne}{" + f"{n_testneg_d1:,}".replace(",", r"\,") + "}",
    ])
    C.write_table(os.path.join(C.TABLES, "M_pool.tex"), macros, size="",
                  header_comment="Macros: the test pool (Sections 3 and 9).")

    with open(os.path.join(C.DATA, "prereg_pool_summary.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump({"rule_sha256": S["rule_sha256"], "n_removed_negatives": n_removed,
                   "n_d2_positives": n_d2,
                   "core_fold_mean_tpr": {f"{arm}@{ro}": core[(arm, ro)]
                                          for _, arm, ro in CORE},
                   "d1_to_d2_shift": shifts, "max_shift": [max_arm, max_shift],
                   "contrasts": rows_json, "all_same_zero_inclusion": same_all},
                  fh, indent=1)
    print(f"  D1 removes {n_removed} test negatives; D2 {n_d2} test positives; "
          f"max |D1->D2| = {max_shift:.4f} ({max_arm}); zero-inclusion unchanged on all "
          f"{len(rows_json)} rows: {same_all}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
