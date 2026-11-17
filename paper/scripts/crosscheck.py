r"""
crosscheck.py -- diff every generated number against PROJECT.md.
================================================================

    python paper/scripts/crosscheck.py

**PROJECT.md IS THE CHECK, NEVER THE SOURCE.** Every number in every table and figure is
computed from a run artifact by the scripts in this directory. This script is the other
half of that rule: it holds what PROJECT.md *says*, with the section it says it in, and
reports where the recomputation disagrees -- rather than silently preferring one.

HOW A COMPARISON IS JUDGED. PROJECT.md quotes numbers at the precision it quotes them at,
so a check passes when the recomputed value **rounds to the quoted value at the quoted
number of decimals**. That is the honest test of "does the paper's table agree with the
record", and it will not manufacture a failure out of a fourth decimal the record never
claimed. Where a disagreement survives that test it is reported with both values and the
artifact path, and it goes into ``STAGE1_REPORT.md``.

TWO BLOCKS. PROJECT.md quotes the pre-registered test pool, so the PROJECT.md checks run
with ``_common.pool("orig")``. The tables now report the D1 pool; a second block checks
every table cell the exact-ID audit recomputed on D1 (``audit/decontam/numbers.csv``) against
the generated tables, and the recomputed artifacts and Appendix A7 against the audit's
``diagnostics.json``.

It writes ``paper/data/crosscheck.json`` and prints a report. It returns non-zero if any
check FAILS, so it can be wired into a rebuild, but it is a reporting tool: a failure
here means the doc and the artifacts disagree, and that disagreement is the finding.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics as st

import _common as C

DATA = C.DATA
CHECKS = []


def check(section: str, what: str, quoted, computed, source: str):
    """Record one comparison. ``quoted`` is PROJECT.md's value as PROJECT.md prints it."""
    CHECKS.append(dict(section=section, what=what, quoted=quoted,
                       computed=(None if computed is None
                                 or (isinstance(computed, float)
                                     and not math.isfinite(computed))
                                 else float(computed)),
                       source=_rel(source)))


def _rel(source: str) -> str:
    """Provenance as repo-relative paths, so crosscheck.json does not depend on where the
    repository sits or on a redirected paper/data (PAPER_DATA_DIR)."""
    orig = source
    for base, name in ((DATA, "paper/data"), (C.REPO, "")):
        for b in {base, base.replace("\\", "/")}:
            source = source.replace(b + os.sep, (name + "/") if name else "")
            source = source.replace(b + "/", (name + "/") if name else "")
    return source.replace("\\", "/") if source != orig else source


def _decimals(q: str) -> int:
    return len(q.split(".")[1]) if "." in q else 0


def judge(rec) -> str:
    q, c = rec["quoted"], rec["computed"]
    if q == "n/a":
        return "PASS" if c is None else "FAIL"
    if c is None:
        return "FAIL"
    nd = _decimals(q)
    got, want = f"{round(c, nd):.{nd}f}", f"{float(q):.{nd}f}"
    # -0 and 0 are one number: a table that printed "-0.0000" now prints "0.0000" with the
    # unrounded value in a footnote (checked separately, build_zero_footnotes).
    return "PASS" if got == want or float(got) == 0.0 == float(want) else "FAIL"


# =====================================================================================
#  the declarations -- each is "PROJECT.md says X in section Y"
# =====================================================================================
def build() -> None:
    LOG = "per-checkpoint logits -> metrics.metric_suite"

    # ---- headline fold-means, folds 1-4 (state block, 7.22, 7.28) -------------------
    for arm, quoted, sd_quoted, sec in [
        ("baseline", "0.1714", "", "7.22"),
        ("lambda0", "0.1355", "0.0364", "7.22"),
        ("bnm", "0.3274", "0.0357", "7.28"),
        ("freematch", "0.4169", "0.0595", "7.12/7.18"),
        ("dann", "0.4167", "0.1525", "7.18"),
        ("coral", "0.4348", "0.0885", "7.15/7.18"),
    ]:
        m, sd, per = C.fold_mean(arm, "tpr")
        check(sec, f"{arm}: folds 1-4 fold-mean TPR@1e-3", quoted, m, LOG)
        if sd_quoted:
            check(sec, f"{arm}: folds 1-4 across-fold sd", sd_quoted, sd, LOG)

    # ---- per-fold TPR at each arm's declared readout (7.18, 7.22, 7.28) -------------
    PERFOLD = {
        "dann": ("7.18", ["0.4286", "0.6224", "0.2653", "0.3505"]),
        "coral": ("7.15", ["0.3571", "0.5612", "0.3980", "0.4227"]),
        "freematch": ("7.12", ["0.4184", "0.4796", "0.3367", "0.4330"]),
        "lambda0": ("7.22", ["0.1837", "0.1429", "0.1122", "0.1031"]),
        "bnm": ("7.28", ["0.3367", "0.3571", "0.2755", "0.3402"]),
        "baseline": ("7.22", ["0.142857", "0.193878", "0.153061", "0.195876"]),
    }
    for arm, (sec, vals) in PERFOLD.items():
        per = C.per_fold(arm, "tpr")
        for f, q in zip((1, 2, 3, 4), vals):
            check(sec, f"{arm}: fold {f} TPR@1e-3", q, per[f], LOG)

    # ---- sensitivity readouts (7.18, 7.22, 7.28) ------------------------------------
    for arm, ro, q, sec in [("dann", 2000, "0.4551", "7.18"),
                            ("lambda0", 1500, "0.1457", "7.22"),
                            ("lambda0", 2000, "0.1740", "7.22"),
                            ("bnm", 1000, "0.3351", "7.28"),
                            ("bnm", 1500, "0.2990", "7.28"),
                            ("bnm", 2000, "0.3427", "7.28")]:
        m, sd, _ = C.fold_mean(arm, "tpr", ro)
        check(sec, f"{arm} @{ro}: fold-mean TPR@1e-3", q, m, LOG)
    check("7.18", "dann @2000: across-fold sd", "0.1102",
          C.fold_mean("dann", "tpr", 2000)[1], LOG)

    # ---- Baseline A's two bases (7.22) ----------------------------------------------
    b04 = st.mean(C.per_fold("baseline", "tpr", folds=(0, 1, 2, 3, 4)).values())
    check("7.22", "Baseline A: folds 0-4 mean (the 0.176 the round-1 table quotes)",
          "0.17591", b04, LOG)
    ms = json.load(open(C.ART["r1_baseline"], encoding="utf-8"))
    check("2/7.4", "Baseline A: POOLED TPR (valid only for the single-model arm)",
          "0.17587", ms["pooled"]["tpr"]["value"], "runs/bakeoff_baseline_A/metric_suite.json")

    # ---- paired contrasts, folds 1-4 (7.22, 7.28, 7.18) -----------------------------
    CON = [
        ("7.22", "lambda0", 4000, "baseline", 8000, "-0.0360", "-0.1249", "+0.0530"),
        ("7.22", "dann", 4000, "lambda0", 4000, "+0.2812", "+0.0595", "+0.5029"),
        ("7.22", "coral", 2000, "lambda0", 2000, "+0.2607", "+0.0976", "+0.4238"),
        ("7.22", "freematch", 1500, "lambda0", 1500, "+0.2712", "+0.1649", "+0.3776"),
        # These two upper/lower bounds were published as +0.2560 and +0.1385 and were
        # CORRECTED in the record on 2026-09-19 (PROJECT.md 7.31): two analysis scripts
        # held 4-dp literals where the exact per-fold values belonged, so the arithmetic
        # ran on rounded inputs. The values below track what PROJECT.md says NOW -- the
        # record moved to meet the artifacts, not the other way round.
        ("7.28", "bnm", 4000, "lambda0", 4000, "+0.1919", "+0.1278", "+0.2561"),
        ("7.22", "coral", 2000, "lambda0", 4000, "+0.2993", "+0.1386", "+0.4600"),
        ("7.22", "freematch", 1500, "lambda0", 4000, "+0.2815", "+0.1858", "+0.3771"),
        ("7.28", "bnm", 2000, "coral", 2000, "-0.0920", "-0.1680", "-0.0160"),
        ("7.28", "bnm", 4000, "dann", 4000, "-0.0893", "-0.2887", "+0.1101"),
        ("7.28", "bnm", 1500, "freematch", 1500, "-0.1180", "-0.2578", "+0.0219"),
        ("7.18", "dann", 4000, "coral", 2000, "-0.0180", "-0.178", "+0.142"),
        ("7.18", "dann", 4000, "freematch", 1500, "-0.0002", "-0.166", "+0.165"),
        ("7.30", "dann", 2000, "coral", 2000, "+0.0203", "", ""),
    ]
    for sec, a, ra, b, rb, qm, qlo, qhi in CON:
        r = C.contrast(a, ra, b, rb)
        name = f"{a}({ra}) - {b}({rb})"
        check(sec, f"{name}: point estimate", qm, r["mean"], LOG + " + paired t, df=3")
        if qlo:
            check(sec, f"{name}: 95% CI lower", qlo, r["ci_lo"], LOG + " + paired t, df=3")
            check(sec, f"{name}: 95% CI upper", qhi, r["ci_hi"], LOG + " + paired t, df=3")

    # ---- the "recovers ~64% of the distance" figure (7.28) --------------------------
    l0 = C.fold_mean("lambda0", "tpr")[0]
    bn = C.fold_mean("bnm", "tpr")[0]
    co = C.fold_mean("coral", "tpr")[0]
    check("7.28", "BNM recovers (BNM - lambda0)/(CORAL - lambda0), quoted against CORAL",
          "0.64", (bn - l0) / (co - l0), LOG)

    # ---- round 1, frozen, 5-fold (7.4) ----------------------------------------------
    R1 = [("r1_baseline", "Baseline A", "0.176", "0.023"),
          ("r1_coral", "CORAL", "0.192", "0.025"),
          ("r1_adamatch", "AdaMatch", "0.331", "0.037"),
          ("r1_freematch", "FreeMatch", "0.348", "0.024")]
    for key, name, qm, qsd in R1:
        d = json.load(open(C.ART[key], encoding="utf-8"))["across_fold"]["tpr"]
        src = f"runs/.../{os.path.basename(os.path.dirname(C.ART[key]))}/metric_suite.json"
        check("7.4", f"round-1 {name}: 5-fold mean TPR@1e-3", qm, d["mean"], src)
        check("7.4", f"round-1 {name}: 5-fold sd", qsd, d["std"], src)
    for key, name, q in [("r1_coral", "CORAL", "0.1868"),
                         ("r1_freematch", "FreeMatch", "0.3376")]:
        pf = json.load(open(C.ART[key], encoding="utf-8"))["per_fold"]
        m = st.mean(pf[str(f)]["tpr"]["value"] for f in (1, 2, 3, 4))
        check("7.15/7.12", f"round-1 {name}: folds 1-4 mean (the confirmatory basis)",
              q, m, "metric_suite.json per-fold")

    for tag, name, qm, qlo, qhi in [
        ("coral", "CORAL", "+0.016", "-0.006", "+0.043"),
        ("adamatch", "AdaMatch", "+0.155", "+0.092", "+0.229"),
        ("freematch", "FreeMatch", "+0.172", "+0.100", "+0.229"),
    ]:
        p = os.path.join(DATA, f"round1_paired_{tag}.json")
        if not os.path.exists(p):
            continue
        agg = json.load(open(p, encoding="utf-8"))["aggregate"]["tpr"]
        s = f"paper/data/round1_paired_{tag}.json (compare_methods, B=1000, seed 0)"
        check("7.4", f"round-1 {name} - Baseline A: point estimate", qm, agg["diff"], s)
        check("7.4", f"round-1 {name} - Baseline A: CI lower", qlo, agg["ci"][0], s)
        check("7.4", f"round-1 {name} - Baseline A: CI upper", qhi, agg["ci"][1], s)

    fb = json.load(open(C.ART["foldmean_bootstrap"], encoding="utf-8"))
    for lab, q, got in [("point estimate", "+0.0793", fb["diff"]),
                        ("CI lower", "+0.0075", fb["ci"][0]),
                        ("CI upper", "+0.1690", fb["ci"][1])]:
        check("7.12", f"FreeMatch nb3 - round-1 FreeMatch (fold-aware bootstrap): {lab}",
              q, got, "runs/foldmean_bootstrap.json")

    # ---- the Lines secondary (2, 7.15, 7.12, 7.28) ----------------------------------
    lr = json.load(open(C.ART["lines_ratios"], encoding="utf-8"))["arms"]
    S = "runs/round7/lines_ratios.json"
    for arm, ro, qc, qx, sec in [("CORAL x nb3", "2000", "0.2428", "48.6", "7.15"),
                                 ("FreeMatch x nb3", "1500", "0.1327", "26.5", "7.12"),
                                 ("DANN x nb3", "2000", "0.2655", "53.1", "2"),
                                 ("BNM x nb3", "4000", "0.1328", "26.6", "2")]:
        rec = lr[arm]["readouts"][ro]
        check(sec, f"{arm} @{ro}: completeness@24%-purity", qc, rec["fold_mean"], S)
        check(sec, f"{arm} @{ro}: X = comp/0.50 (%)", qx, rec["pct_of_lines"], S)
    cor = lr["CORAL x nb3"]["readouts"]["2000"]
    check("2", "CORAL X: 95% CI lower (%)", "4.3", cor["pct_of_lines_ci95"][0], S)
    check("2", "CORAL X: 95% CI upper (%)", "92.9", cor["pct_of_lines_ci95"][1], S)
    for arm, ro, sec in [("DANN x nb3", "4000", "2"), ("lambda=0", "4000", "7.22"),
                         ("Baseline A (frozen)", "8000", "2")]:
        rec = lr[arm]["readouts"][ro]
        check(sec, f"{arm} @{ro}: completeness@24%-purity is undefined (n/a)",
              "n/a", rec["fold_mean"], S)
    r1fm = json.load(open(C.ART["r1_freematch"], encoding="utf-8"))["per_fold"]
    check("2", "round-1 FreeMatch: folds 1-4 comp@24%-purity (the like-for-like basis)",
          "0.1101", st.mean(r1fm[str(f)]["comp_at_purity"]["value"] for f in (1, 2, 3, 4)),
          "runs/bakeoff_freematch_5fold/metric_suite.json")
    check("2", "round-1 FreeMatch: 5-fold comp@24%-purity (NOT comparable to folds 1-4)",
          "0.1412",
          st.mean(r1fm[str(f)]["comp_at_purity"]["value"] for f in range(5)),
          "runs/bakeoff_freematch_5fold/metric_suite.json")

    # ---- other secondaries at the declared readouts ---------------------------------
    check("7.22", "lambda=0: AUROC at 4000 (competent ranker, useless at the point)",
          "0.8800", C.fold_mean("lambda0", "auroc")[0], LOG)
    check("7.22", "lambda=0: purity@50%-completeness (1:2200) at 4000",
          "0.0091", C.fold_mean("lambda0", "purity_corr")[0], LOG)
    check("7.12", "FreeMatch x nb3: purity@50%-completeness (1:2200) at 1500",
          "0.1335", C.fold_mean("freematch", "purity_corr")[0], LOG)
    check("7.28", "BNM x nb3: AUROC at 4000", "0.9083",
          C.fold_mean("bnm", "auroc")[0], LOG)
    check("7.18", "DANN x nb3: AUROC at 4000", "0.9066",
          C.fold_mean("dann", "auroc")[0], LOG)
    check("7.18", "CORAL x nb3: AUROC at 2000", "0.9088",
          C.fold_mean("coral", "auroc")[0], LOG)
    check("7.18", "FreeMatch x nb3: AUROC at 1500", "0.9657",
          C.fold_mean("freematch", "auroc")[0], LOG)

    # ---- window averaging (7.30) ----------------------------------------------------
    wa = json.load(open(C.ART["window_average"], encoding="utf-8"))
    W = "runs/round7/window_average.json"
    for arm, ro, q1, q8l, q8r in [("CORAL x nb3", "2000", "0.4348", "0.4577", "0.4194"),
                                  ("FreeMatch x nb3", "1500", "0.4169", "0.4604", "0.4348"),
                                  ("DANN x nb3", "4000", "0.4167", "0.4577", "0.3964"),
                                  ("DANN x nb3", "2000", "0.4551", "0.4271", "0.3709"),
                                  ("lambda=0", "4000", "0.1355", "0.1715", "0.1689"),
                                  ("BNM x nb3", "4000", "0.3274", "0.3810", "0.3427"),
                                  ("Baseline A (frozen)", "8000", "0.1759", "0.1637", "0.1657")]:
        rec = wa["arms"][arm]["readouts"][ro]
        check("7.30", f"{arm} @{ro}: k=1 (must reproduce the hashed headline)",
              q1, rec["mean_logit"]["1"]["tpr_mean"], W)
        check("7.30", f"{arm} @{ro}: k=8 mean_logit", q8l,
              rec["mean_logit"]["8"]["tpr_mean"], W)
        check("7.30", f"{arm} @{ro}: k=8 mean_rank", q8r,
              rec["mean_rank"]["8"]["tpr_mean"], W)
    for arm, q in [("lambda=0", "0.59"), ("Baseline A (frozen)", "0.56")]:
        check("7.30", f"{arm}: achieved / sqrt(k) at k=8 (the autocorrelation shortfall)",
              q, wa["scatter_reduction"][arm]["8"]["achieved_over_expected"], W)

    # ---- 7.16's eighth entry, fold 0 ------------------------------------------------
    w0 = json.load(open(C.ART["window_fold0"], encoding="utf-8"))["arms"]
    W0 = "runs/round7/window_fold0.json"
    for arm, q1, q8 in [("CORAL x nb3", "0.0513", "0.0250"),
                        ("FreeMatch x nb3", "0.0606", "0.0468"),
                        ("DANN arm 2 (lambda live)", "0.0914", "0.0590"),
                        ("DANN arm 1 (lambda=0)", "0.0402", "0.0285"),
                        ("BNM x nb3", "0.0717", "0.0290")]:
        check("7.16/7.30", f"{arm}: fold-0 checkpoint sd at k=1", q1, w0[arm]["1"]["sd"], W0)
        check("7.16/7.30", f"{arm}: fold-0 checkpoint sd at k=8", q8, w0[arm]["8"]["sd"], W0)
    mult8 = [w0[a]["8"]["multiple_of_dann_minus_coral"] for a in w0]
    check("7.16/7.30", "within-run sd at k=8, LOWEST multiple of |DANN-CORAL|",
          "1.4", min(mult8), W0)
    check("7.16/7.30", "within-run sd at k=8, HIGHEST multiple of |DANN-CORAL|",
          "3.3", max(mult8), W0)

    # ---- the fold-0 controlled pair (7.18) ------------------------------------------
    import csv as _csv

    def curve(p):
        with open(p, newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        return [float(r["tpr_at_1e-3"]) for r in rows]

    a1, a2 = curve(C.ART["dann_arm1_curve"]), curve(C.ART["dann_arm2_curve"])
    D = "runs/dann_r5_curves/*.csv"
    check("7.18", "fold 0: lambda=0 arm, mean over 32 checkpoints", "0.1454",
          st.mean(a1), D)
    check("7.18", "fold 0: DANN arm, mean over 32 checkpoints", "0.4534", st.mean(a2), D)
    check("7.18", "fold 0: lambda=0 arm, last checkpoint", "0.1531", a1[-1], D)
    check("7.18", "fold 0: DANN arm, last checkpoint", "0.4796", a2[-1], D)
    check("7.18", "fold 0: lambda=0 arm, maximum", "0.2041", max(a1), D)
    check("7.18", "fold 0: DANN arm, maximum", "0.5816", max(a2), D)
    check("7.18", "fold 0: shared step-0 initialisation (frozen LP->FT head)", "0.1939",
          C.suite("baseline", 0, 8000)["tpr"], LOG)
    check("7.18", "fold 0: steps where DANN is above the control (of 32)", "32",
          sum(1 for x, y in zip(a2, a1) if x > y), D)

    # ---- Appendix A4: the round-1 augmentation / precision cells (7.3) --------------
    ap = json.load(open(C.R("runs", "lever1_aug_precision.json"), encoding="utf-8"))
    A4 = "runs/lever1_aug_precision.json"
    for key, cell, qt, qa in [("A_none_fp32", "A", "0.193878", "0.942761"),
                              ("B_weak_fp32", "B", "0.142857", "0.921417"),
                              ("C_weak_fp16", "C", "0.122449", "0.924291")]:
        check("7.3", f"A4 cell {cell}: TPR@1e-3", qt, ap["cells"][key]["tpr"], A4)
        check("7.3", f"A4 cell {cell}: AUROC", qa, ap["cells"][key]["auroc"], A4)
    for leg, name, qd, qlo, qhi in [("augmentation_effect_at_fp32", "aug leg B-A",
                                     "-0.0510", "-0.1224", "0.0000"),
                                    ("precision_effect_at_weak_aug", "precision leg C-B",
                                     "-0.0204", "-0.0510", "+0.0306"),
                                    ("total", "total C-A", "-0.0714", "-0.1327", "-0.0102")]:
        check("7.3", f"A4 {name}: point estimate", qd, ap[leg]["diff"], A4)
        check("7.3", f"A4 {name}: CI lower", qlo, ap[leg]["ci"][0], A4)
        check("7.3", f"A4 {name}: CI upper", qhi, ap[leg]["ci"][1], A4)

    # Baseline A's fold-0 value must be the same number wherever the paper prints it: the
    # main per-fold table (T5) and Appendix A4, which cites it to say cell C is not that run.
    # Compared as PRINTED, so a formatting or source drift in either file is caught.
    import re as _re
    t5 = open(os.path.join(C.TABLES, "T5_perfold.tex"), encoding="utf-8").read()
    t5_f0 = _re.search(r"^Baseline A & 8000 & \$([0-9.]+)\$", t5, _re.M).group(1)
    ma = open(os.path.join(C.TABLES, "M_augprec.tex"), encoding="utf-8").read()
    a4_f0 = _re.search(r"\\augprecBaselineFoldZero\}\{\$([0-9.]+)\$\}", ma).group(1)
    check("7.3", "Baseline A fold 0: T5 prints == A4 cites", t5_f0, float(a4_f0),
          "tables/T5_perfold.tex vs tables/M_augprec.tex")
    check("7.3", "Baseline A fold 0 (the run cell A reproduces)", "0.193878",
          C.suite("baseline", 0, 8000)["tpr"], LOG)
    vp = json.load(open(C.R("runs", "lever1_screen_report.json"), encoding="utf-8"))
    check("7.7", "cell C's configuration, local vs Kaggle: difference", "0.0204",
          vp["venue_portability"]["diff"], "runs/lever1_screen_report.json")
    check("7.7", "venue difference: CI lower", "-0.0306", vp["venue_portability"]["ci"][0],
          "runs/lever1_screen_report.json")
    check("7.7", "venue difference: CI upper", "0.0510", vp["venue_portability"]["ci"][1],
          "runs/lever1_screen_report.json")
    cr = json.load(open(os.path.join(DATA, "venue_coral_repro.json"), encoding="utf-8"))
    check("7.7", "CORAL x nb3 re-scoring: checkpoints with identical TPR (of 9)", "9",
          cr["identical_tpr"], "paper/data/venue_coral_repro.json")

    # ---- Appendix A5: the restart evidence and the reconstructed control (7.10) ------
    def fold0_curve(run):
        with open(C.R("runs", run, "curve.csv"), newline="", encoding="utf-8") as fh:
            return [float(r["fold0_tpr_1e-3"]) for r in _csv.DictReader(fh)]
    uc = fold0_curve("freematch_fold0_nb0_control")
    fc = fold0_curve("freematch_fold0_nb0_control_FIXED")
    A5 = "runs/freematch_fold0_nb0_control{,_FIXED}/curve.csv"
    check("7.10", "A5 unfixed control: step-0 TPR", "0.3878", uc[0], A5)
    check("7.10", "A5 unfixed control: trough", "0.2347", min(uc), A5)
    check("7.10", "A5 unfixed control: final", "0.3367", uc[-1], A5)
    check("7.10", "A5 unfixed control: max drift from step 0", "0.1531",
          max(abs(x - uc[0]) for x in uc), A5)
    check("7.10", "A5 unfixed control: trough-to-final recovery ('spread')", "0.102",
          uc[-1] - min(uc), A5)
    check("7.12", "A5 reconstructed control: mean over steps > 0", "0.3686",
          st.mean(fc[1:]), A5)
    check("7.12", "A5 reconstructed control: sd over steps > 0 (population)", "0.0107",
          st.pstdev(fc[1:]), A5)
    tl = json.load(open(C.R("runs", "round1_tail_stats.json"), encoding="utf-8"))["0"]
    check("7.12", "A5 round-1 fold-0 tail 6000-8000: mean", "0.3696", tl["tail_mean"],
          "runs/round1_tail_stats.json")
    check("7.12", "A5 round-1 fold-0 tail 6000-8000: sd", "0.0105", tl["tail_sd"],
          "runs/round1_tail_stats.json")

    # ---- the splits (4, 7.17) --------------------------------------------------------
    sp = json.load(open(C.ART["splits"], encoding="utf-8"))
    S2 = "data/splits/splits.json"
    check("4.1", "source-train sim lens positives", "24000",
          len(sp["source_train"]["sim_lens_rows"]), S2)
    check("4.1", "ML-FP hard negatives", "4455",
          len(sp["source_train"]["neg_mlfp"]["object_ids"]), S2)
    check("7.17", "source-train bare deflectors", "2881",
          len(sp["source_train"]["neg_bare_deflector"]["deflector_object_ids"]), S2)
    check("7.17", "source-train random P1 negatives", "16664",
          len(sp["source_train"]["neg_random"]["object_ids"]), S2)
    check("4.2", "source-val sim lens positives", "6000",
          len(sp["val"]["sim_lens_rows"]), S2)
    check("4.4", "shared test negatives", "26399",
          len(sp["test_negatives_shared"]["object_ids"]), S2)
    f0, f4 = sp["real_lenses_kfold"]["folds"][0], sp["real_lenses_kfold"]["folds"][4]
    check("7.13", "fold 0 test positives", "98", len(f0["test_lens_rows"]), S2)
    check("7.13", "fold 4 test positives (the fold that is NOT 98)", "97",
          len(f4["test_lens_rows"]), S2)
    check("4.3", "fold 0 unlabelled target lenses", "391",
          len(f0["target_lens_rows"]), S2)
    check("3.2", "graded real lenses in total",
          "489", sum(len(f["test_lens_rows"])
                     for f in sp["real_lenses_kfold"]["folds"]), S2)


# =====================================================================================
#  D1 -- the pool the tables now use -- against the exact-ID audit's numbers.csv
# =====================================================================================
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\{,\}\d{3})*(?:\.\d+)?")

# Cells the audit could only copy from original-pool summary files (its D1 column holds
# the original value there) and that this pipeline now recomputes on D1 from stored
# scores. They are checked against the recomputed artifact instead, below.
RECOMPUTED_SINCE_AUDIT = {"T2_contrasts.tex": ("FreeMatch frozen", "one positive"),
                         "A3_orderings.tex": ("5. The ranking", "8. The three",
                                              "9. Unfreezing")}
# Rewrite v2 (Addendum 4), the one permitted removal of checks: A3 row 1, "None of 24
# candidate budgets (500--3375 steps) ... at step 750", averaged the development fold with the
# abandoned FreeMatch fold-1 run (runs/fold_average.json; audit/curation_fold1/REPORT.md).
# The paper no longer uses it, so its four numbers.csv checks (rows 6-9: 24, 500, 3375, 750)
# are retired here; build_rewrite_v2() checks the fold-0-only replacement instead.
RETIRED_BY_REWRITE_V2 = {"A3_orderings.tex": ("1. The budget at which",)}


def _table_body(name):
    p = os.path.join(C.TABLES, name)
    return [l.rstrip("\r\n") for l in open(p, encoding="utf-8") if not l.startswith("%")]


def _cells(line):
    return re.split(r"(?<!\\)&", line)


def build_d1() -> None:
    """Every table cell the audit recomputed on D1 must be reproduced by this pipeline."""
    NCSV = "audit/decontam/numbers.csv (D1)"
    rows = list(csv.DictReader(open(C.ART["pool_numbers"], encoding="utf-8")))
    bodies = {}
    for r in rows:
        if r["D1"] in ("", "None") or not r["provenance"].startswith("recomputed"):
            continue
        f = r["file"]
        skip = RECOMPUTED_SINCE_AUDIT.get(f, ())
        if any(k in r["row"] or k in r["text"] for k in skip):
            continue
        # only the row's value cell (column 2) is retired; its row-number cell stays checked
        if r["col"] == "2" and any(k in r["row"] for k in RETIRED_BY_REWRITE_V2.get(f, ())):
            RETIRED.append(f"{f}:{r['line']} c{r['col']}t{r['tok']} = {r['D1']}")
            continue
        if f not in bodies:
            bodies[f] = _table_body(f)
        line = bodies[f][int(r["line"]) - 1]
        parts = _cells(line) if "&" in line and "\\\\" in line else [line]
        col, t = int(r["col"]), int(r["tok"])
        toks = NUM.findall(parts[col]) if col < len(parts) else []
        got = float(toks[t].replace("{,}", "")) if t < len(toks) else None
        check("D1", f"{f}:{r['line']} {r['row'][:26]} c{col}t{t}",
              _quoted(r["D1"]), got, NCSV)

    diag = json.load(open(C.R("audit", "decontam", "diagnostics.json"), encoding="utf-8"))
    core = {(d["arm"], int(d["readout"])): d for d in diag["core"]}
    DIAG = "audit/decontam/diagnostics.json"

    # the licensed pair, recomputed on D1 by bootstrap_foldmean itself: its two fold-means
    # are quantities the audit measured on D1 (FreeMatch x nb3 core, round-1 per-fold)
    fb = json.load(open(os.path.join(DATA, "foldmean_bootstrap_D1.json"), encoding="utf-8"))
    fbo = json.load(open(C.ART["foldmean_bootstrap"], encoding="utf-8"))
    SRC = "paper/data/foldmean_bootstrap_D1.json"
    check("D1", "licensed pair: FreeMatch x nb3 fold-mean",
          f"{core[('freematch', 1500)]['D1']['mean']:.12f}", fb["unfrozen_fold_mean"], SRC)
    fm_r1 = diag["round1"]["FreeMatch (standard)"]["D1"]["per_fold"][1:5]
    check("D1", "licensed pair: round-1 FreeMatch folds 1-4 mean",
          f"{st.mean(fm_r1):.12f}", fb["round1_fold_mean"], SRC)
    check("D1", "licensed pair: diff = unfrozen - round-1",
          f"{core[('freematch', 1500)]['D1']['mean'] - st.mean(fm_r1):.12f}", fb["diff"], SRC)
    check("D1", "licensed pair: same B as the original", f"{fbo['B']}", fb["B"], SRC)
    check("D1", "licensed pair: shared negatives = original - 265",
          f"{fbo['n_shared_negatives'] - 265}", fb["n_shared_negatives"], SRC)

    # T3 means: this pipeline's round-1 D1 5-fold means against the audit's
    for label, art in [("Baseline A", "r1_baseline"), ("CORAL (warmup-cosine)", "r1_coral"),
                       ("AdaMatch", "r1_adamatch"), ("FreeMatch (standard)", "r1_freematch")]:
        run = os.path.basename(os.path.dirname(C.ART[art]))
        with C.pool("D1"):
            mine = C.round1_summary(run)["across_fold"]["tpr"]["mean"]
        check("D1", f"T3 {label}: 5-fold mean TPR@1e-3",
              f"{diag['round1'][label]['D1']['mean']:.12f}", mine, DIAG + " round1")

    # A7: the appendix's three pools against the audit, arm by arm
    a7 = json.load(open(os.path.join(DATA, "prereg_pool_summary.json"), encoding="utf-8"))
    for key, v in a7["core_fold_mean_tpr"].items():
        arm, ro = key.split("@")
        d = core[(arm, int(ro))]
        for p in ("orig", "D1", "D2"):
            check("A7", f"A7 {key}: fold-mean TPR@1e-3, pool {p}",
                  f"{d[p]['mean']:.12f}", v[p], DIAG + " core")
    check("A7", "A7: every contrast keeps its zero-inclusion", "1",
          float(a7["all_same_zero_inclusion"]), "paper/data/prereg_pool_summary.json")

    # Section 3 and Figure 1 state the shared test pool after the exclusion, with the
    # as-drawn count beside it: both macros against PROJECT.md 4.4's 26399 and the 265
    # objects the exclusion removes.
    mp = open(os.path.join(C.TABLES, "M_pool.tex"), encoding="utf-8").read()
    def pool_macro(name):
        v = re.search(r"\\newcommand\{\\" + name + r"\}\{([0-9\\,]+)\}", mp).group(1)
        return float(v.replace("\\,", ""))
    check("4.4", "Section 3: shared test negatives as drawn (TestNegDrawn)", "26399",
          pool_macro("TestNegDrawn"), "tables/M_pool.tex")
    check("4.4", "Section 3: shared test negatives on D1 (TestNegDOne)", f"{26399 - 265}",
          pool_macro("TestNegDOne"), "tables/M_pool.tex")


RETIRED: list = []


def build_rewrite_v2() -> None:
    """Rewrite v2: the macros of tables/M_rewrite2.tex against their sources, read here
    independently of make_rewrite_v2_macros.py; the fold-0-only peak reading recomputed from
    the stored logits; the budget values the budget paragraph states; and Table 11 row 1."""
    import glob
    import numpy as np
    import metrics as me
    txt = open(os.path.join(C.TABLES, "M_rewrite2.tex"), encoding="utf-8").read()
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", txt, re.M))
    num = {k: v.replace("\\,", "") for k, v in mac.items()}
    REC = "PROJECTv7.md"
    # The lab log's values come through scripts/record_extract.py (cc.* keys): from the log,
    # or without it from public_inputs/record_extract.json. The patterns are the ones used here
    # before; the source label stays the log's name.
    import record_extract as RX

    def rx(key):
        return RX.value(key)[0]
    # Section 3.1: the deflector pool, from a different line of the record than the generator
    check("3.1", "deflector pool (rwDeflPool), record's data inventory line",
          rx("cc.defl_pool").replace(",", ""), float(num["rwDeflPool"]), REC)
    check("3.1", "arc-capable deflectors (rwDeflArc)", rx("cc.defl_arc").replace(",", ""),
          float(num["rwDeflArc"]), REC)
    check("4.1", "deflector reuse limit (rwDeflReuse = 'ten')", rx("cc.defl_reuse"),
          {"ten": 10.0}.get(mac["rwDeflReuse"]), REC)
    f5 = json.load(open(C.R("audit", "review_r1", "out", "F5_host_properties.json"),
                        encoding="utf-8"))["result"]
    check("F5", "distinct deflectors of the training positives (rwDeflDistinct) = bare-deflector "
          "negatives, one per distinct deflector",
          f5["distributions"]["b1_bare_deflectors"]["stellarmass"]["n_total"] and
          str(f5["distributions"]["b1_bare_deflectors"]["stellarmass"]["n_total"]),
          float(num["rwDeflDistinct"]), "audit/review_r1/out/F5_host_properties.json")
    # Section 4.2: optimiser settings, from one fold's config per arm and the record
    def cfg(*p):
        return json.load(open(C.R(*p), encoding="utf-8"))
    # powers of ten are compared by exponent: judge() rounds to the quoted decimals, which
    # would make "1e-4" against 0.0001 pass vacuously
    sci = {"10^{-4}": -4.0, "10^{-3}": -3.0}

    def ex(v):
        e = math.log10(float(v))
        assert abs(e - round(e)) < 1e-9, v
        return f"{round(e):d}"
    d2 = cfg("runs", "dann_main", "fold2", "extracted", "fold2", "run_config.json")
    check("7.14a", "unfrozen default learning rate (rwOptLr): Zoobot learning_rate, exponent",
          ex(rx("cc.opt_lr")), sci[mac["rwOptLr"]], REC)
    check("7.14a", "unfrozen default layer-wise decay (rwOptDecay): Zoobot lr_decay",
          rx("cc.opt_lr_decay"), float(mac["rwOptDecay"]), REC)
    check("7.14a", "unfrozen default weight decay (rwOptWd): Zoobot weight_decay",
          rx("cc.opt_wd"), float(mac["rwOptWd"]), REC)
    check("cfg", "DANN fold 2 run config: lr = rwOptLr, exponent", ex(d2["lr"]),
          sci[mac["rwOptLr"]], "runs/dann_main/fold2/extracted/fold2/run_config.json")
    f3 = cfg("runs", "freematch_nb3_unfrozen_fold3", "run_config.json")
    for key, m in (("lr", "rwFmHeadLr"), ("backbone_lr", "rwFmEncLr"), ("weight_decay", "rwFmWd")):
        check("7.14a", f"unfrozen FreeMatch fold 3 {key} ({m}), exponent", ex(f3[key]),
              sci[mac[m]], "runs/freematch_nb3_unfrozen_fold3/run_config.json")
    check("7.14a", "record: unfrozen FreeMatch kept the frozen run's lr, exponent",
          ex(rx("cc.fm_lr")), sci[mac["rwFmHeadLr"]], REC)
    check("7.14a", "record: unfrozen FreeMatch kept the frozen run's wd, exponent",
          ex(rx("cc.fm_wd")), sci[mac["rwFmWd"]], REC)
    b = cfg("runs", "bakeoff_baseline_A", "run_config.json")
    check("cfg", "Baseline A lr (rwBaseLr), exponent", ex(b["lr"]), sci[mac["rwBaseLr"]],
          "runs/bakeoff_baseline_A/run_config.json")
    check("cfg", "Baseline A weight decay (rwBaseWd), exponent", ex(b["weight_decay"]),
          sci[mac["rwBaseWd"]], "runs/bakeoff_baseline_A/run_config.json")
    check("cfg", "batch size (rwBatch)", f"{b['batch_size']}", float(num["rwBatch"]),
          "runs/bakeoff_baseline_A/run_config.json")
    # Section 4.5: the budgets the paragraph states, each against its own source
    bnm = cfg("runs", "bnm_r6_budget.json")
    check("7.26", "BNM budget = 2 x crossing step (1000)", "1000",
          float(bnm["selected_budget_steps"]), "runs/bnm_r6_budget.json")
    check("7.26", "BNM budget is twice the crossing step", "2",
          bnm["selected_budget_steps"] / bnm["crossing_step"], "runs/bnm_r6_budget.json")
    check("7.15", "CORAL development curve maximum declined (step 2500)", "2500",
          float(rx("cc.coral_argmax")), REC)
    check("7.11", "FreeMatch first budget (2000), Amendment 1 old value", "2000",
          float(rx("cc.fm_budget_old")), REC)
    check("7.11", "FreeMatch amended budget (1500), Amendment 1 new value", "1500",
          float(rx("cc.fm_budget_new")), REC)
    for f in (1, 2, 3, 4):
        c = cfg("runs", f"freematch_nb3_unfrozen_fold{f}", "run_config.json")
        check("7.11", f"FreeMatch confirmatory fold {f} ran at the final budget 1500", "1500",
              float(c["max_steps"]), f"runs/freematch_nb3_unfrozen_fold{f}/run_config.json")
    # Table 11 row 1 and Section 7: the fold-0-only peak reading, recomputed on D1 from the
    # stored logits with the repo's metric routine, against fold0_only.json and the macros
    F0 = C.R("audit", "curation_fold1", "out", "fold0_only.json")
    j = json.load(open(F0, encoding="utf-8"))
    run = C.R("runs", "freematch_fold0_nb3_unfrozen")
    files = sorted(glob.glob(os.path.join(run, "logits", "step_*.npz")),
                   key=lambda p: int(os.path.basename(p)[5:-4]))
    ids = np.load(os.path.join(run, "logits", "ids.npz"))
    pos_ids, neg_ids = list(ids["test_pos_fold0"]), list(ids["test_negatives_shared"])
    steps, tpr = [], []
    for p in files:
        z = np.load(p)
        a, n = C.pool_filter(z["test_pos_fold0"], z["test_negatives_shared"], pos_ids, neg_ids,
                             where=p, pool_name="D1")
        steps.append(int(z["step"]))
        tpr.append(me.tpr_at_fpr(np.r_[np.ones(a.size), np.zeros(n.size)], np.r_[a, n], 1e-3))
    steps, tpr = np.array(steps), np.array(tpr)
    pk = tpr.max()
    within2 = [int(s) for s in steps[tpr >= pk - 2 / 98 - 1e-12]]
    n_sd = int((tpr >= pk - tpr.std() - 1e-12).sum())
    SRC = "runs/freematch_fold0_nb3_unfrozen/logits (D1, recomputed)"
    jd = j["D1"]
    check("Add.4", "fold-0 D1: number of curve points (rwFzPoints)", str(jd["n_points"]),
          float(num["rwFzPoints"]), SRC)
    check("Add.4", "fold-0 D1: recomputed points = JSON", str(jd["n_points"]), float(len(tpr)), SRC)
    check("Add.4", "fold-0 D1: within one sd (rwFzWithinSd)", str(jd["b_within_one_sd"]["ddof0"]["n"]),
          float(num["rwFzWithinSd"]), F0)
    check("Add.4", "fold-0 D1: within one sd, recomputed", str(jd["b_within_one_sd"]["ddof0"]["n"]),
          float(n_sd), SRC)
    listed = [int(x) for x in mac["rwFzSteps"].split(", ")]
    for k, (s_json, s_mac, s_re) in enumerate(zip(jd["a_within_two_positives_steps"], listed,
                                                  within2 + [None] * 9)):
        check("Add.4", f"fold-0 D1: step {k + 1} within two positives (rwFzSteps)", str(s_json),
              float(s_mac), F0)
        check("Add.4", f"fold-0 D1: step {k + 1} within two positives, recomputed", str(s_json),
              None if s_re is None else float(s_re), SRC)
    check("Add.4", "fold-0 D1: number of steps within two positives", str(len(within2)),
          float(len(listed)), SRC)
    check("Add.4", "Section 7: spread starts at rwFzStepMin", str(min(within2)),
          float(num["rwFzStepMin"]), SRC)
    check("Add.4", "Section 7: spread ends at rwFzStepMax", str(max(within2)),
          float(num["rwFzStepMax"]), SRC)
    row1 = [l for l in _table_body("AC_orderings_v2.tex") if l.startswith("1. ")]
    check("Add.4", "Table 11 row 1 prints the three fold-0 macros and no fold-averaged value", "1",
          float(len(row1) == 1 and all(f"\\{m}{{}}" in row1[0] for m in
                                       ("rwFzSteps", "rwFzWithinSd", "rwFzPoints"))
                and "candidate budgets" not in row1[0]), "tables/AC_orderings_v2.tex")


def _quoted(s: str) -> str:
    """numbers.csv prints floats with repr; quote them at the table's own precision."""
    v = float(s)
    return f"{v:.0f}" if v == int(v) and "." not in s.rstrip("0").rstrip(".") else s


# =====================================================================================
#  The prose macros (tables/M_prose.tex): every one checked on both pools
# =====================================================================================
def build_prose() -> None:
    """(1) Pre-registered pool: each replaced number's macro renders the number that was
    typed there before (paper/PROSE_MACRO_MAP.csv). (2) D1: each macro written to
    tables/M_prose.tex equals an independent reference -- the exact-ID audit's
    diagnostics.json where it measured the quantity, otherwise the D1 artifact read here
    directly (not through make_m_prose)."""
    import make_m_prose as MP
    rows = list(csv.DictReader(open(os.path.join(C.PAPER, "PROSE_MACRO_MAP.csv"),
                                    encoding="utf-8")))
    with C.pool("orig"):
        orig = {n: MP.render(v, nd, sg) for n, v, nd, sg, _ in MP.build()}
    for r in rows:
        name = r["macro"].lstrip("\\")
        got = orig.get(name)
        check("PR-orig", f"{r['file'].split('/')[-1]}:{r['line']} {r['macro']}",
              r["typed"], None if got is None else float(got), "typed original")

    text = open(os.path.join(C.TABLES, "M_prose.tex"), encoding="utf-8").read()
    written = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text))
    ref = _prose_refs()
    for name, shown in sorted(written.items()):
        if name not in ref:
            check("PR-D1", f"{name}: no independent reference", "n/a", float(shown), "none")
            continue
        value, src = ref[name]
        nd = len(shown.split(".")[1]) if "." in shown else 0
        quoted = f"{value:+.{nd}f}" if shown[0] in "+-" else f"{value:.{nd}f}"
        check("PR-D1", f"{name} = {shown}", quoted, float(shown), src)


def _prose_refs() -> dict:
    """name -> (reference value on D1, source), read independently of make_m_prose."""
    R = {}
    diag = json.load(open(C.R("audit", "decontam", "diagnostics.json"), encoding="utf-8"))
    DG = "audit diagnostics.json"
    core = {(d["arm"], int(d["readout"])): d["D1"] for d in diag["core"]}
    for key, arm, ro in [("Baseline", "baseline", 8000), ("Lambda", "lambda0", 4000),
                         ("Bnm", "bnm", 4000), ("FreeMatch", "freematch", 1500),
                         ("Dann", "dann", 4000), ("DannSens", "dann", 2000),
                         ("Coral", "coral", 2000)]:
        R[f"fm{key}"] = (core[(arm, ro)]["mean"], DG)
        R[f"sd{key}"] = (core[(arm, ro)]["sd"], DG)
    ct = {c["contrast"]: c["new"] for c in diag["contrasts"] if c["mode"] == "D1"}

    def find(label):
        hits = [v for k, v in ct.items() if k.split("$^{")[0].strip() == label]
        assert len(hits) == 1, label
        return hits[0]
    labels = {"LamBase": r"$\lambda=0$ $-$ Baseline A", "DannLam": r"DANN $-$ $\lambda=0$",
              "CoralLam": r"CORAL $-$ $\lambda=0$", "FmLam": r"FreeMatch $-$ $\lambda=0$",
              "BnmLam": r"BNM $-$ $\lambda=0$", "BnmCoral": r"BNM $-$ CORAL",
              "BnmDann": r"BNM $-$ DANN", "BnmFm": r"BNM $-$ FreeMatch",
              "DannCoral": r"DANN $-$ CORAL", "DannFm": r"DANN $-$ FreeMatch",
              "DannCoralMatched": r"DANN $-$ CORAL (both at 2000)"}
    for key, lab in labels.items():
        m_, lo, hi = find(lab)
        R[f"ct{key}"], R[f"ct{key}Lo"], R[f"ct{key}Hi"] = (m_, DG), (lo, DG), (hi, DG)
        R[f"ct{key}Abs"] = (abs(m_), DG)
    R["bnmShareUnmatched"] = (100 * (core[("bnm", 4000)]["mean"] - core[("lambda0", 4000)]["mean"])
                              / (core[("coral", 2000)]["mean"] - core[("lambda0", 4000)]["mean"]), DG)
    R["bnmShareMatched"] = (100 * find(labels["BnmLam"])[0] / find(labels["CoralLam"])[0],
                            DG + " (4-dp contrasts)")
    r1 = diag["round1"]
    for key, lab in [("Base", "Baseline A"), ("Coral", "CORAL (warmup-cosine)"),
                     ("Ada", "AdaMatch"), ("Fm", "FreeMatch (standard)")]:
        R[f"rOne{key}"] = (r1[lab]["D1"]["mean"], DG)
        # Table 3 prints the population sd (np.nanstd); diagnostics.json stores the
        # sample sd, so the reference is the population sd of the audit's own per-fold values
        R[f"rOne{key}Sd"] = (st.pstdev(r1[lab]["D1"]["per_fold"]), DG + " per_fold (pop. sd)")
    for key, tag in [("Coral", "coral"), ("Ada", "adamatch"), ("Fm", "freematch")]:
        p = os.path.join(DATA, f"round1_paired_{tag}_D1.json")
        a = json.load(open(p, encoding="utf-8"))["aggregate"]["tpr"]
        R[f"rOne{key}Diff"], R[f"rOne{key}Lo"], R[f"rOne{key}Hi"] = (
            (a["diff"], p), (a["ci"][0], p), (a["ci"][1], p))
    p = os.path.join(DATA, "foldmean_bootstrap_D1.json")
    lp = json.load(open(p, encoding="utf-8"))
    R["lpDiff"], R["lpLo"], R["lpHi"] = (lp["diff"], p), (lp["ci"][0], p), (lp["ci"][1], p)
    p = os.path.join(DATA, "lines_ratios_D1.json")
    lr = json.load(open(p, encoding="utf-8"))["arms"]
    for key, arm, ro in [("Coral", "CORAL x nb3", 2000), ("FreeMatch", "FreeMatch x nb3", 1500),
                         ("Bnm", "BNM x nb3", 4000), ("Dann", "DANN x nb3", 4000),
                         ("DannSens", "DANN x nb3", 2000)]:
        e = lr[arm]["readouts"][str(ro)]
        if e["defined_on_all_folds"]:
            R[f"lr{key}Comp"] = (e["fold_mean"], p)
            R[f"lr{key}X"] = (e["pct_of_lines"], p)
            R[f"lr{key}XLo"], R[f"lr{key}XHi"] = ((e["pct_of_lines_ci95"][0], p),
                                                   (e["pct_of_lines_ci95"][1], p))
        R[f"lr{key}CompFoldTwo"] = (e["comp_at_purity_per_fold"]["2"], p)
    nums = list(csv.DictReader(open(C.ART["pool_numbers"], encoding="utf-8")))
    au = [r for r in nums if r["file"] == "T4_secondary.tex" and "lambda=0" in r["row"]
          and r["col"] == "5"]
    R["auLambda"] = (float(au[0]["D1"]), "audit numbers.csv (T4 AUROC)")
    p = os.path.join(DATA, "window_average_D1.json")
    wa = json.load(open(p, encoding="utf-8"))
    for key, arm, ro in [("FreeMatch", "FreeMatch x nb3", 1500), ("Coral", "CORAL x nb3", 2000),
                         ("Dann", "DANN x nb3", 4000), ("DannSens", "DANN x nb3", 2000)]:
        e = wa["arms"][arm]["readouts"][str(ro)]
        R[f"wa{key}Logit"] = (e["mean_logit"]["8"]["tpr_mean"], p)
        R[f"wa{key}Rank"] = (e["mean_rank"]["8"]["tpr_mean"], p)
    sr = wa["scatter_reduction"]
    R["waLambdaAchieved"] = (sr["lambda=0"]["8"]["achieved_over_expected"], p)
    R["waLambdaSdOne"] = (sr["lambda=0"]["1"]["sd_k_foldmean"], p)
    R["waLambdaSdEight"] = (sr["lambda=0"]["8"]["sd_k_foldmean"], p)
    R["waDannAchieved"] = (sr["DANN x nb3"]["8"]["achieved_over_expected"], p)
    R["waBnmAchieved"] = (sr["BNM x nb3"]["8"]["achieved_over_expected"], p)
    R["waFreeMatchAchieved"] = (sr["FreeMatch x nb3"]["8"]["achieved_over_expected"], p)
    p = os.path.join(DATA, "window_fold0_D1.json")
    wz = json.load(open(p, encoding="utf-8"))["arms"]
    names = {"Coral": "CORAL x nb3", "FreeMatch": "FreeMatch x nb3",
             "Dann": "DANN arm 2 (lambda live)", "Lambda": "DANN arm 1 (lambda=0)",
             "Bnm": "BNM x nb3"}
    for key, arm in names.items():
        for k, word in (("1", "One"), ("8", "Eight")):
            R[f"wz{key}Sd{word}"] = (wz[arm][k]["sd"], p)
            R[f"wz{key}Mult{word}"] = (wz[arm][k]["multiple_of_dann_minus_coral"], p)
    m1 = [wz[names[k]]["1"]["multiple_of_dann_minus_coral"] for k in ("Coral", "FreeMatch", "Dann")]
    m8 = [wz[a]["8"]["multiple_of_dann_minus_coral"] for a in names.values()]
    R["wzRankedMultMin"], R["wzRankedMultMax"] = (min(m1), p), (max(m1), p)
    R["wzEightMultMin"], R["wzEightMultMax"] = (min(m8), p), (max(m8), p)
    p = os.path.join(DATA, "fold0_curves_D1.json")
    fz = json.load(open(p, encoding="utf-8"))
    lam = {int(s): v for s, v in fz["curves"]["arm1_lambda0"].items()}
    R["fzLambdaLast"] = (lam[max(lam)], p)
    R["fzLambdaArgmax"] = (float(max(lam, key=lambda s: (lam[s], -s))), p)
    R["fzInit"] = (r1["Baseline A"]["D1"]["per_fold"][0], DG + " (round-1 Baseline A fold 0)")

    # ---- macros added by the wording pass ----
    ONE = 1.0 / 98
    R["baseFoldZeroPrereg"] = (r1["Baseline A"]["orig"]["per_fold"][0],
                               DG + " (round-1 Baseline A fold 0, pre-registered pool)")
    R["lpLoInPositives"] = (lp["ci"][0] / ONE, "foldmean_bootstrap_D1.json / (1/98)")
    R["ctBnmCoralHiInPositives"] = (abs(find(labels["BnmCoral"])[2]) / ONE, DG + " / (1/98)")
    p = os.path.join(DATA, "lines_ratios_D1.json")
    b14 = json.load(open(p, encoding="utf-8"))["arms"]["Baseline A (frozen)"]["readouts"]["8000"]["folds_1_4"]
    R["lrBaselineX"] = (b14["pct_of_lines"], p)
    R["lrBaselineXLo"] = (100 * b14["ci95"][0] / 0.5, p)
    R["lrBaselineXHi"] = (100 * b14["ci95"][1] / 0.5, p)
    wl = {k: wa["arms"][a]["readouts"][str(ro)] for k, a, ro in
          (("c", "CORAL x nb3", 2000), ("d4", "DANN x nb3", 4000), ("d2", "DANN x nb3", 2000))}
    lgt = {k: e["mean_logit"]["8"]["tpr_mean"] for k, e in wl.items()}
    rnk = {k: e["mean_rank"]["8"]["tpr_mean"] for k, e in wl.items()}
    p = os.path.join(DATA, "window_average_D1.json")
    R["waLogitDannCoralMaxGap"] = (max(lgt["d4"], lgt["d2"]) - lgt["c"], p)
    R["waRankCoralDannMinGap"] = (rnk["c"] - max(rnk["d4"], rnk["d2"]), p)
    p = os.path.join(DATA, "window_fold0_D1.json")
    dfm = abs(find(labels["DannFm"])[0])
    for key in ("Coral", "FreeMatch", "Dann"):
        R[f"wz{key}MultFm"] = (wz[names[key]]["1"]["sd"] / dfm, p + " / audit |DANN-FreeMatch|")
    R["wzEightMultMinTwo"], R["wzEightMultMaxTwo"] = (min(m8), p), (max(m8), p)
    # (the final wording pass removed \wzCoralEightGap from the text; its value is still
    # asserted by make_m_prose's claim F2b, and a macro not written needs no reference)

    # ---- macros added by the final wording pass ----
    # D8: Figure 6's FreeMatch/CORAL fold-mean curves, rebuilt here from the envelope
    import numpy as np
    import plot_completeness_purity as pcp
    grid = np.geomspace(1e-3, 1.0, 600)

    def fold_mean_curve(arm, ro):
        rows = []
        for f in C.ARMS[arm]["folds"]:
            comp, pur = pcp.envelope(*C.ARMS[arm]["load"](f)[ro])
            o = np.argsort(pur)
            rows.append(np.interp(grid, pur[o], comp[o], left=np.nan, right=np.nan))
        return np.nanmean(np.where(np.isnan(np.vstack(rows)).any(axis=0), np.nan,
                                   np.vstack(rows)), axis=0)
    with C.pool("D1"):
        fm, co = fold_mean_curve("FreeMatch x nb3", 1500), fold_mean_curve("CORAL x nb3", 2000)
    ok = ~np.isnan(fm) & ~np.isnan(co)
    R["dEightFmAbovePct"] = (100 * float((fm[ok] > co[ok]).mean()),
                             "Figure 6 construction, recomputed in crosscheck")
    # Section 7.3: grid sizes, counted from the step files the analysis's loader returns
    with C.pool("D1"):
        n = {len(C.ARMS["DANN x nb3"]["load"](f)) for f in C.ARMS["DANN x nb3"]["folds"]}
    assert len(n) == 1
    R["waDannGridPoints"] = (float(n.pop()), "analyze_window_average loaders (step 0 excluded)")
    R["waDannEightPositions"] = (R["waDannGridPoints"][0] - 7, "grid points - k + 1, k = 8")

    # ---- macros added by the provenance-fixes pass: the deflector sample, read here directly ----
    p = C.INPUT_LABEL["deflectors"]
    z = C.deflector_table()
    sm, zp, mv = (z[k].astype(np.float64) for k in ("stellarmass", "phz_median", "mvis"))
    R["DeflMassFloor"] = (np.floor(10 * sm.min()) / 10, p + " (min log M*, floored to 0.1)")
    R["DeflZmin"], R["DeflZmax"] = (zp.min(), p), (zp.max(), p)
    R["DeflUnphysMass"] = (12.5, "provenance_fixes_brief.md threshold")
    R["DeflUnphysMag"] = (-26.0, "provenance_fixes_brief.md threshold")
    R["DeflUnphysN"] = (float(np.count_nonzero((sm > 12.5) | (mv < -26.0))), p)

    # ---- macros added by the grade-X pass: matched by EXACT ID rebuilt from RA/Dec ----
    import glob
    import sys
    import pandas as pd
    sys.path.insert(0, C.R("scripts"))
    from exact_ids import rebuild
    ex = np.array([rebuild(float(a), float(b)) for a, b in zip(z["ra"], z["dec"])])
    r2e = dict(zip(z["object_id"].astype(np.int64).tolist(), ex.tolist()))
    bad = set(ex[(sm > 12.5) | (mv < -26.0)].tolist())
    hosts, n_lens = set(), 0
    for _name, s in C.sim_shards(["object_id"]):
        e = [r2e[int(o)] for o in s["object_id"].astype(np.int64)]
        hits = [x for x in e if x in bad]
        hosts |= set(hits)
        n_lens += len(hits)
    R["DeflUnphysDrawn"] = (float(len(hosts)), "shards x deflector file, exact IDs from RA/Dec")
    R["DeflUnphysLenses"] = (float(n_lens), "shards x deflector file, exact IDs from RA/Dec")
    q = C.INPUT_LABEL["catalogue_v0.0.3"]
    cat = C.catalogue("v0.0.3")
    c_ids = {rebuild(a, b) for a, b, g in zip(cat["right_ascension"], cat["declination"],
                                                cat["grade"]) if g == "C"}
    sp = json.load(open(C.ART["splits"], encoding="utf-8"))
    hn = {int(o) for o in sp["source_train"]["neg_mlfp"]["object_ids"]}
    R["GradeXnowC"] = (float(len(hn & c_ids)), q + " (grade C, exact IDs from RA/Dec)")
    R["GradeXUsed"] = (float(len(hn)), "splits.json source_train.neg_mlfp, read here")
    pool = json.load(open(C.R("data", "splits", "dilute10_pool.json"), encoding="utf-8"))
    assert "grade X" in pool["reference_set_definitions"]["c_grade_X_hard_negatives"]
    n_gx = pool["reference_sets"]["c_grade_X_hard_negatives"]["n"]
    assert len(hn) <= n_gx
    R["GradeXAll"] = (float(n_gx), "dilute10_pool.json reference set c (part 1), read here")

    # ---- macros added by the screening-cut pass: the query constants, read here directly ----
    qs = C.R("scripts", "01_query_catalog.py")
    lo, hi = (float(x) for x in re.findall(r"flux_vis_psf\s+BETWEEN\s+([\d.]+)\s+AND\s+([\d.]+)",
                                             open(qs, encoding="utf-8").read())[0])
    R["FluxMin"], R["FluxMax"] = (lo, qs), (hi, qs)
    R["MagFaint"] = (23.9 - 2.5 * math.log10(lo), qs + " (AB, uJy)")
    R["MagBright"] = (23.9 - 2.5 * math.log10(hi), qs + " (AB, uJy)")
    # D1 test negatives (exact-ID removal set), PSF flux from the tile-assigned query output
    keep = [int(o) for o in sp["test_negatives_shared"]["object_ids"]
            if int(o) not in C.pool_sets()["neg_remove"]]
    qt = C.INPUT_LABEL["mer_flux_tiles"]
    fl = C.mer_flux(tiles=True, usecols=["object_id", "flux_vis_psf"]).set_index("object_id")["flux_vis_psf"]
    mags = 23.9 - 2.5 * np.log10(fl.reindex(keep).to_numpy(float))
    assert not np.isnan(mags).any() and len(keep) == 26399 - 265
    R["FracFaintTest"] = (100 * float((mags > 22.5).mean()), qt + " (D1 test negatives)")
    return R


def build_claims() -> None:
    """Every worded condition of the prose (make_m_prose.prose_claims), on D1."""
    p = os.path.join(DATA, "prose_claims_D1.json")
    for c in json.load(open(p, encoding="utf-8")):
        check("CLAIM", f"{c['id']}: {c['claim']}"[:61], "1", float(c["holds"]),
              "paper/data/prose_claims_D1.json")


def _printed(name: str) -> dict:
    """{macro: printed value} of a generated macro file, '\\,' and '$' stripped."""
    txt = open(os.path.join(C.TABLES, name), encoding="utf-8").read()
    return {n: v.replace("\\,", "").replace("$", "")
            for n, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", txt)}


def build_review() -> None:
    """Rewrite v1: every macro in tables/M_review.tex and tables/M_display.tex against its
    source, read here directly -- the review-round JSONs, PROJECTv7.md, the run files and
    _common -- not through make_review_macros.py or make_m_display.py; and
    _common.CONTRAST_PREREG against the review's pre-registration lookup (F1)."""
    R1 = C.R("audit", "review_r1", "out")
    R1B = C.R("audit", "review_r1b", "out")

    def rj(folder, name):
        return json.load(open(os.path.join(folder, name), encoding="utf-8"))["result"]
    ref = {}
    f4 = rj(R1, "F4_fold_overlap.json")
    off = [(v["count"], v["fraction_of_row_fold_pool"]) for a, row in
           f4["shared_target_lenses"].items() for b, v in row.items() if a != b]
    ref["rvSharedTargetMin"] = min(c for c, _ in off)
    ref["rvSharedTargetMax"] = max(c for c, _ in off)
    ref["rvTargetLensPool"] = f4["pool_sizes"]["target_lens_rows"]["1"]
    ref["rvTargetOverlapPct"] = 5 * round(100 * sum(fr for _, fr in off) / len(off) / 5)
    f5 = rj(R1, "F5_host_properties.json")
    au = [x["auroc"] for d in f5["single_variable_auroc_positive_vs_negative"].values()
          for x in list(d["vs_each_negative_type"].values()) + [d["vs_all_source_negatives_having_it"]]
          if isinstance(x, dict)]
    ref["rvHostAurocMin"], ref["rvHostAurocMax"] = min(au), max(au)
    av = f5["availability"]
    ref["rvBareDeflN"] = int(av["b1_bare_deflectors"]["stellarmass"].split("/")[1])
    ref["rvSourceNegN"] = sum(int(next(iter(av[k].values())).split("/")[1])
                              for k in ("b1_bare_deflectors", "b2_random_galaxies", "b3_grade_X_ml_fp"))
    ref["rvSourcePosN"] = int(av["a_training_positive_hosts"]["stellarmass"].split("/")[1])
    tally = rj(R1, "F6_d2_lenses.json")["role_tally"]
    ref["rvDTwoTrainNeg"] = tally.get("training negative (random galaxy)", 0)
    ref["rvDTwoRandom"] = (tally.get("training negative (random galaxy)", 0)
                           + tally.get("validation negative (random galaxy)", 0))
    ref["rvDTwoValNeg"] = rj(R1, "F6_d2_lenses.json")["n"] - ref["rvDTwoTrainNeg"]
    f7 = rj(R1, "F7_lenses_in_test_pool.json")["per_arm"]
    own = [f7["coral@2000"][k]["n28_that_are_also_this_folds_test_positives"] for k in "1234"]
    da = [f7[t][k]["n28_at_or_above_threshold"] for t in
          ("bnm@4000", "freematch@1500", "dann@4000", "coral@2000") for k in "1234"]
    rf = [f7[t][k]["n28_at_or_above_threshold"] for t in ("baseline@8000", "lambda0@4000")
          for k in "1234"]
    ref.update(rvOwnPosMin=min(own), rvOwnPosMax=max(own), rvFsevenDaMin=min(da),
               rvFsevenDaMax=max(da), rvFsevenRefMin=min(rf), rvFsevenRefMax=max(rf),
               rvAboveThrN=f7["coral@2000"]["1"]["n_negatives_at_or_above_threshold"])
    a1 = rj(R1, "A1_resolution.json")
    rows = [(k.split("|")[0], g) for k, g in a1["gaps_and_ratios"].items()
            if g.get("available") and all(g["steps_are_declared_readouts"])]
    three = [g for p, g in rows if p in ("dann-coral", "dann-freematch", "coral-freematch")]
    bnm = [g for p, g in rows if p.startswith("bnm-")]
    nb = [math.sqrt(sum(a1["sigma_by_arm"][x]["folds1_4_mean_detrended_b_linfit_resid"] ** 2
                        for x in p.split("-"))) / 2
          for p, g in rows if p in ("dann-coral", "dann-freematch", "coral-freematch")]
    ref.update(rvNoiseMin=min(nb), rvNoiseMax=max(nb), rvScale=max(nb),
               rvGapThreeMin=min(abs(g["gap"]) for g in three),
               rvGapThreeMax=max(abs(g["gap"]) for g in three),
               rvGapBnmMin=min(abs(g["gap"]) for g in bnm),
               rvGapBnmMax=max(abs(g["gap"]) for g in bnm))
    arms5 = ("lambda0", "bnm", "freematch", "dann", "coral")
    sg = [a1["sigma_by_arm"][x]["folds1_4_mean_detrended_b_linfit_resid"] for x in arms5]
    lg = [a1["sigma_by_arm"][x]["folds1_4_mean_lag1_autocorr_resid"] for x in arms5]
    ref.update(rvSigmaMin=min(sg), rvSigmaMax=max(sg), rvLagMin=min(lg), rvLagMax=max(lg))
    ref["rvTopNegN"] = rj(R1, "A2_top_negatives.json")["n_unique_objects"]
    ref["rvTopNegC"] = rj(R1, "A2_crossmatch.json")["n_objects_matching_A_B_C_X"]
    mbias = rj(R1, "A2_magnitude_bias.json")
    ref["rvAboveFaintPct"] = 100 * mbias["above_threshold_unique_objects"]["frac_fainter_22p5"]
    ref["rvPoolFaintPct"] = 100 * mbias["D1_pool"]["frac_fainter_22p5"]
    th = rj(R1, "A3_recall_by_property.json")[
        "einstein_radius_effective_median_pdf (modeling_lens_mass.csv)"]
    ref["rvThetaN"], ref["rvConfTestN"] = th["n_with_value"], th["n_test_lenses_folds1_4"]
    gb = rj(R1, "A3_galleries.json")["galleries"]["b_missed_by_every_arm"]
    ref["rvMissedAllN"], ref["rvMissedAllB"] = gb["n_in_set"], gb["grade_counts_in_set"]["B"]
    # r3: grade A / grade B recall ratio over Table 4's seven readouts, from the summed counts
    g3 = rj(R1, "A3_recall_by_grade.json")["by_arm"]
    assert len(g3) == 7
    gr = [(v["A"]["summed"]["recovered"] / v["A"]["summed"]["total"])
          / (v["B"]["summed"]["recovered"] / v["B"]["summed"]["total"]) for v in g3.values()]
    ref["rvGradeRatioMin"], ref["rvGradeRatioMax"] = min(gr), max(gr)
    run = json.load(open(C.R("runs", "dann_main", "fold1", "extracted", "fold1", "run.json"),
                         encoding="utf-8"))
    ref["rvHoldoutN"] = run["dann_d_holdout_n"]
    l2 = rj(R1B, "L2_bright_pool.json")
    ref["rvBrightN"] = l2["pool"]["n_bright"]
    ref["rvBnmBrightRatio"] = l2["readouts"]["bnm@4000"]["lines_ratio_bright"]
    ref["rvBnmDOneRatio"] = l2["readouts"]["bnm@4000"]["lines_ratio_D1"]
    ref["rvGradShiftPct"] = json.load(open(C.R("runs", "bnm_r6_prereg.json"), encoding="utf-8"))[
        "provenance"]["cpu_to_t4_ratio_move_pct"]
    import record_extract as RX      # the log's lines, or the extract (cc.early_* keys)
    # Record lines 494 / 4143 / 3181 (re-pointed 2026-10-12, rewrite v4b, after the record's
    # §7.35 marker lines moved them from 491 / 4123 / 3162).
    for name, key in (("rvEarlyNoDa", "noda"), ("rvEarlyIw", "iw"), ("rvEarlyDann", "dann"),
                      ("rvEarlyProbe", "probe")):
        ref[name] = float(RX.value(f"cc.early_{key}")[0])
    printed = _printed("M_review.tex")
    sha = C.pool_rule()["sha256_of_body_without_this_field"][:8]
    check("RV", "rvRuleSha = rule.json digest (first 8)", "1",
          float(printed.pop("rvRuleSha") == sha), "audit/decontam/rule.json")
    for name, shown in sorted(printed.items()):
        check("RV", f"{name} = {shown}", shown, ref.get(name),
              "audit/review_r1(b)/out, PROJECTv7.md, run files")

    # ---------------- display variants (M_display.tex) ----------------
    dref = {}
    heads = [("Baseline", "baseline", 8000), ("Lambda", "lambda0", 4000), ("Bnm", "bnm", 4000),
             ("FreeMatch", "freematch", 1500), ("Dann", "dann", 4000),
             ("DannSens", "dann", 2000), ("Coral", "coral", 2000)]
    with C.pool("D1"):
        for key, arm, ro in heads:
            mean, sd, _ = C.fold_mean(arm, "tpr", ro)
            dref[f"dFm{key}"], dref[f"dSd{key}"] = mean, sd
        da4 = [dref[f"dFm{k}"] for k in ("Bnm", "FreeMatch", "Dann", "Coral")]
        dref["dHeadMin"], dref["dHeadMax"] = min(da4), max(da4)
        names = ["LamBase", "DannLam", "CoralLamM", "FmLamM", "BnmLam", "CoralLamP", "FmLamP",
                 "BnmCoral", "BnmDann", "BnmFm", "DannCoral", "DannFm", "DannCoralM"]
        means = {}
        for i, (_p, _l, a, ra, b, rb, _m, _k) in enumerate(C.CONTRAST_ROWS):
            r = C.contrast(a, ra, b, rb)
            n = names[i]
            means[i] = r["mean"]
            dref.update({f"dCt{n}": r["mean"], f"dCt{n}Lo": r["ci_lo"], f"dCt{n}Hi": r["ci_hi"],
                         f"dCt{n}Abs": abs(r["mean"]),
                         f"dCt{n}AbsLo": min(abs(r["ci_lo"]), abs(r["ci_hi"])),
                         f"dCt{n}AbsHi": max(abs(r["ci_lo"]), abs(r["ci_hi"]))})
        dref["dGainMin"] = min(means[i] for i in (1, 4, 5, 6))
        dref["dGainMax"] = max(means[i] for i in (1, 4, 5, 6))
        dref["dAuLambda"] = C.fold_mean("lambda0", "auroc", 4000)[0]
    p = os.path.join(DATA, "lines_ratios_D1.json")
    lr = json.load(open(p, encoding="utf-8"))["arms"]
    pct = {}
    for key, arm, ro in [("Coral", "CORAL x nb3", 2000), ("FreeMatch", "FreeMatch x nb3", 1500),
                         ("Bnm", "BNM x nb3", 4000), ("Dann", "DANN x nb3", 4000),
                         ("DannSens", "DANN x nb3", 2000)]:
        e = lr[arm]["readouts"][str(ro)]
        pct[key] = e["pct_of_lines"]
        dref[f"dLr{key}"] = e["pct_of_lines"]
        dref[f"dLr{key}Lo"], dref[f"dLr{key}Hi"] = e["pct_of_lines_ci95"]
        dref[f"dComp{key}"] = e["fold_mean"]
    b14 = lr["Baseline A (frozen)"]["readouts"]["8000"]["folds_1_4"]
    dref["dLrBaseline"] = b14["pct_of_lines"]
    dref["dLrBaselineLo"] = 100 * b14["ci95"][0] / C.LINES_COMPLETENESS
    dref["dLrBaselineHi"] = 100 * b14["ci95"][1] / C.LINES_COMPLETENESS
    dref["dLrDaMin"] = min(pct[k] for k in ("Coral", "FreeMatch", "Bnm", "Dann"))
    dref["dLrDaMax"] = max(pct[k] for k in ("Coral", "FreeMatch", "Bnm", "Dann"))
    # Appendix G: the A7 table's own printed fold-means (four decimals) bound the pool changes
    a7 = "\n".join(_table_body("A7_prereg_pool.tex"))
    sect = a7.split(r"\textit{(a)", 1)[1].split(r"\bottomrule", 1)[0]
    up, d2 = [], []
    for ln in sect.split("\n"):
        nums_ = [float(x) for x in re.findall(r"\$([-+]?\d\.\d{4})\$", ln)]
        if len(nums_) == 3:
            up.append(nums_[1] - nums_[0])
            d2.append(abs(nums_[2] - nums_[1]))
    assert len(up) == 7, up
    dref["dPoolUpMin"], dref["dPoolUpMax"], dref["dPoolDTwoMax"] = min(up), max(up), max(d2)
    for key, tag, lab in [("Coral", "coral", "CORAL (warmup-cosine)"),
                          ("Ada", "adamatch", "AdaMatch"), ("Fm", "freematch", "FreeMatch (standard)")]:
        a = json.load(open(os.path.join(DATA, f"round1_paired_{tag}_D1.json"),
                           encoding="utf-8"))["aggregate"]["tpr"]
        dref[f"dFrozen{key}Diff"], dref[f"dFrozen{key}Lo"], dref[f"dFrozen{key}Hi"] = (
            a["diff"], a["ci"][0], a["ci"][1])
    diag = json.load(open(C.R("audit", "decontam", "diagnostics.json"), encoding="utf-8"))["round1"]
    for key, lab in [("Base", "Baseline A"), ("Coral", "CORAL (warmup-cosine)"),
                     ("Ada", "AdaMatch"), ("Fm", "FreeMatch (standard)")]:
        dref[f"dFrozen{key}"] = diag[lab]["D1"]["mean"]
    lp = json.load(open(os.path.join(DATA, "foldmean_bootstrap_D1.json"), encoding="utf-8"))
    dref["dLpDiff"], dref["dLpLo"], dref["dLpHi"] = lp["diff"], lp["ci"][0], lp["ci"][1]
    fz = json.load(open(os.path.join(DATA, "fold0_curves_D1.json"), encoding="utf-8"))
    lam = fz["curves"]["arm1_lambda0"]
    dref["dFzLambdaLast"] = lam[max(lam, key=int)]
    dref["dFzInit"] = fz["init_baseline_fold0"]
    for name, shown in sorted(_printed("M_display.tex").items()):
        check("DISP", f"{name} = {shown}", shown, dref.get(name),
              "_common (D1), lines_ratios_D1.json, A7 table, round-1 JSONs, fold0_curves_D1.json")

    # ---------------- the pre-registration marks ----------------
    f1 = rj(R1, "F1_prereg_contrasts.json")["reading_of_each_contrast"]
    pre = {i for rd in f1.values() for i in rd["headline_form_rows"]}
    sens = {i for rd in f1.values() for s in rd["sensitivity_forms"] for i in s["rows"]}
    got_pre = {i for i, v in C.CONTRAST_PREREG.items() if v == C.PREREG}
    got_sens = {i for i, v in C.CONTRAST_PREREG.items() if v == C.SENSITIVITY}
    check("RV", "CONTRAST_PREREG pre-registered rows = F1 headline forms", "1",
          float(got_pre == pre), "audit/review_r1/out/F1_prereg_contrasts.json")
    check("RV", "CONTRAST_PREREG sensitivity rows = F1 sensitivity forms", "1",
          float(got_sens == sens), "audit/review_r1/out/F1_prereg_contrasts.json")


def build_a3_windows() -> None:
    """A3 rows 5, 8 and 9 now read the pool's window artifacts. The audit's copy read the
    pre-registered-pool files there, so those cells are checked here, against the D1
    artifacts, rather than against numbers.csv."""
    lines = _table_body("A3_orderings.tex")
    wa = json.load(open(os.path.join(DATA, "window_average_D1.json"), encoding="utf-8"))["arms"]
    wz = json.load(open(os.path.join(DATA, "window_fold0_D1.json"), encoding="utf-8"))
    SRC = "paper/data/window_*_D1.json"

    def row(prefix):
        hit = [l for l in lines if l.startswith(prefix)]
        assert len(hit) == 1, prefix
        return [t for t in NUM.findall(hit[0].split("&", 2)[2])]
    t5 = [t for t in row("5. The ranking") if "." in t]
    d = wa["DANN x nb3"]["readouts"]
    for q, v in zip(t5, [d["2000"]["mean_logit"]["1"]["tpr_mean"],
                         d["4000"]["mean_logit"]["1"]["tpr_mean"]]):
        check("D1-A3", "A3 row 5: DANN at k=1", f"{v:.3f}", abs(float(q)), SRC)
    t8 = [t for t in row("8. The three") if "." in t]
    ranked = ["CORAL x nb3", "FreeMatch x nb3", "DANN arm 2 (lambda live)"]
    m1 = [wz["arms"][a]["1"]["multiple_of_dann_minus_coral"] for a in ranked]
    m8 = [wz["arms"][a]["8"]["multiple_of_dann_minus_coral"] for a in wz["arms"]]
    for q, v in zip(t8, [min(m1), max(m1), min(m8), max(m8)]):
        nd = len(q.split(".")[1])
        check("D1-A3", "A3 row 8: scatter / |DANN - CORAL|", f"{v:.{nd}f}", abs(float(q)), SRC)
    t9 = [t for t in row("9. Unfreezing") if "." in t]
    st_ = wz["row9_stability"]
    vals = []
    for g in ("250-8000", "250-4000"):
        for k in ("1", "8"):
            vals += [st_[g][k]["lambda0_sd_over_mean"], st_[g][k]["dann_sd_over_mean"]]
    for q, v in zip(t9, vals):
        check("D1-A3", "A3 row 9: checkpoint sd/mean", f"{v:.3f}", abs(float(q)), SRC)


def build_k8_pairs() -> None:
    """Appendix C's k=1 / k=8 pairs table (rewrite v5, make_k8_pairs.py). Every printed
    number is recomputed here independently of the audit file the generator reads: k=1 from
    the stored scores (_common.contrast, D1), k=8 from the per-fold windowed TPRs in
    window_average_D1.json, paired t with df=3. Signs count folds agreeing with the mean."""
    from scipy import stats as _st
    t975 = float(_st.t.ppf(0.975, 3))
    w = json.load(open(os.path.join(DATA, "window_average_D1.json"), encoding="utf-8"))["arms"]
    ARM = {"DANN": ("dann", "DANN x nb3"), "CORAL": ("coral", "CORAL x nb3"),
           "FreeMatch": ("freematch", "FreeMatch x nb3")}
    SRC_W = "paper/data/window_average_D1.json (per-fold windowed TPR, k=8)"
    SRC_S = "stored scores -> _common.contrast (D1)"

    def ptt(diffs):
        m = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1))
        hw = t975 * sd / math.sqrt(len(diffs))
        s = 1.0 if m >= 0 else -1.0
        return m, m - hw, m + hw, sum(1 for d in diffs if d * s > 0)

    # Each pair is two table lines: the first carries the pair, the steps, the three deltas
    # and the three sign counts; the next carries the three intervals.
    lines = _table_body("C_k8_pairs.tex")
    firsts = [i for i, l in enumerate(lines) if " $-$ " in l and "/" in l.split("&")[1]]
    assert len(firsts) == 5, firsts
    for i in firsts:
        cells = [c.strip() for c in lines[i].rstrip("\\ ").split("&")]
        nxt = [c.strip() for c in lines[i + 1].rstrip("\\ ").split("&")]
        na, nb = [x.strip() for x in cells[0].split("$-$")]
        ra, rb = [int(x) for x in cells[1].split("/")]
        deltas = [NUM.findall(cells[j])[0] for j in (2, 4, 6)]
        cis = [[t for t in NUM.findall(nxt[j]) if "." in t] for j in (2, 4, 6)]
        signs = [re.fullmatch(r"(\d)/4", cells[j]).group(1) for j in (3, 5, 7)]
        assert all(len(c) == 2 for c in cis), (lines[i + 1], cis)
        nums = [x for d, c in zip(deltas, cis) for x in (d, c[0], c[1])]
        with C.pool("D1"):
            c = C.contrast(ARM[na][0], ra, ARM[nb][0], rb)
            k1 = (c["mean"], c["ci_lo"], c["ci_hi"], c["n_same_sign"])
        blocks = [("k=1", k1, SRC_S)]
        for comb in ("mean_logit", "mean_rank"):
            x = w[ARM[na][1]]["readouts"][str(ra)][comb]["8"]["tpr_per_fold"]
            y = w[ARM[nb][1]]["readouts"][str(rb)][comb]["8"]["tpr_per_fold"]
            blocks.append((f"k=8 {comb}", ptt([p - q for p, q in zip(x, y)]), SRC_W))
        for bi, (lab, (m, lo, hi, sg), src) in enumerate(blocks):
            for q, v, nm in zip(nums[3 * bi:3 * bi + 3], (m, lo, hi), ("delta", "CI lo", "CI hi")):
                check("V5-K8", f"C_k8_pairs {na}-{nb} {ra}/{rb} {lab} {nm}", q, v, src)
            check("V5-K8", f"C_k8_pairs {na}-{nb} {ra}/{rb} {lab} signs", signs[bi], sg, src)


def build_zero_footnotes() -> None:
    """Tables 3 and 15 print the D1 lambda=0 - Baseline A difference as 0.0000 with a
    footnote giving it before rounding; the footnote must equal the stored-score value at
    the precision it prints."""
    with C.pool("D1"):
        x = C.contrast("lambda0", 4000, "baseline", 8000)["mean"]
    for name, mark in (("T2_contrasts.tex", r"$^{f}$"), ("A7_prereg_pool.tex", r"$^{\dagger}$")):
        text = "\n".join(_table_body(name))
        m = re.search(re.escape(mark) + r"\$([-+]\d\.\d+)\$ before rounding", text)
        assert m, f"{name}: the rounded-to-zero footnote is missing"
        check("D1", f"{name}: lambda=0 - Baseline A before rounding (footnote)",
              m.group(1), x, "stored scores -> _common.contrast (D1)")


def build_dilute10() -> None:
    """Paper v6: the 1:10 diluted-target experiment and the DANN - lambda=0 memorisation
    diagnostic. Every macro of tables/M_dilute10.tex and every number of the v6 tables,
    recomputed from its artifact in crosscheck_dilute10.py (stored logits through the pool
    filter, the analysis files, the head norms, the run logs and the pool file)."""
    import crosscheck_dilute10
    crosscheck_dilute10.build(check)


def build_prereg_fixes() -> None:
    """PROJECTv7 7.65: every macro of tables/M_prereg.tex, every cell of
    tables/C_window_secondary.tex, the new \\dannPrefoldSha and Table 10's two DANN
    pre-registration rows, each recomputed here from its file by a path separate from
    make_prereg_fixes.py / make_a2_hashes.py."""
    import numpy as np
    import sys
    sys.path.insert(0, C.REPO_SCRIPTS)
    import integrity
    import metrics as me
    P = _printed("M_prereg.tex")
    E = {}
    J = lambda rel: json.load(open(C.R(rel), encoding="utf-8"))
    # DANN five-fold means: fold 0 by tpr_at_fpr on the pooled stored logits, folds 1-4 by suite
    for step, name in ((4000, "dDannFiveFoldFourK"), (2000, "dDannFiveFoldTwoK")):
        pos, neg = C._pooled_npz(C.R("runs", "dann_arm2_fold0", "rescore", "dann_r5_nb3_fold0",
                                     "logits", "dann_nb3_fold0", f"step_{step}.npz"), 0)
        y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
        t0 = me.tpr_at_fpr(y, np.r_[pos, neg], C.TARGET_FPR)
        t0 = float(t0["tpr"] if isinstance(t0, dict) else t0)
        E[name] = (t0 + sum(C.suite("dann", f, step)["tpr"] for f in (1, 2, 3, 4))) / 5
    # FreeMatch's adaptive threshold: 90% of its first-to-last travel, numpy over the CSV
    rows = list(csv.DictReader(open(C.R("runs", "freematch_fold0_nb3_unfrozen", "diagnostics.csv"),
                                    encoding="utf-8")))
    s = np.array([int(r["step"]) for r in rows if r["tau_global_last"]])
    v = np.array([float(r["tau_global_last"]) for r in rows if r["tau_global_last"]])
    E["fmTauNinetyStep"] = float(s[np.argmax(v >= v[0] + 0.9 * (v[-1] - v[0]))])
    # BNM: the decay statistic (delta re-derived from the stored curve), predicted-positive
    # fraction, gate and budget
    ba = J("runs/bnm_r6_budget_analysis.json")
    t1 = ba["task1"]
    curve = {int(k): x for k, x in ba["curve"].items()}
    early = st.mean(curve[k] for k in t1["early_steps"])
    late = st.mean(curve[k] for k in t1["late_steps"])
    assert abs((early - late) - t1["delta"]) < 1e-12
    E.update(bnmDecayDelta=early - late, bnmDecayLo=t1["ci"][0], bnmDecayHi=t1["ci"][1],
             bnmDecayCap=float(max(t1["late_steps"])))
    pp = J("runs/bnm_r6_prereg.json")["predictions_from_7_20"]["prediction_2_collapse"]
    a, b = re.search(r"from (\S+) toward ~(\S+),", pp).groups()
    E.update(bnmPredPosStart=float(a), bnmPredPosTarget=float(b))
    tr = [np.genfromtxt(C.R("runs", "bnm_r6_folds", f"bnm_r6_fold{k}", "trace.csv"), delimiter=",",
                        names=True) for k in (1, 2, 3, 4)]
    allv = np.concatenate([t["bnm_pred_pos_frac"] for t in tr])
    latev = np.concatenate([t["bnm_pred_pos_frac"][t["step"] > 3000] for t in tr])
    E.update(bnmPredPosFirstStep=float(min(t["step"][0] for t in tr)),
             bnmPredPosLate=float(latev.mean()), bnmPredPosLateFrom=3000.0,
             bnmPredPosMin=float(allv.min()), bnmPredPosMax=float(allv.max()))
    check("PRF", "BNM predicted-positive fraction never collapses (0.01, 0.99)", "1",
          float(((allv > 0.01) & (allv < 0.99)).all()), "runs/bnm_r6_folds/*/trace.csv")
    gates = [J(f"runs/bnm_r6_folds/bnm_r6_fold{k}/run.json")["step0_gate"] for k in (1, 2, 3, 4)]
    check("PRF", "BNM step-0 gate: TPR diff 0 on every fold", "1",
          float(all(g["tpr_abs_diff"] == 0 for g in gates)), "runs/bnm_r6_folds/*/run.json")
    mx = max(g["auroc_abs_diff"] for g in gates)
    want = {"bnmGateAurocTol": "10^{-3}", "dlGateAurocTol": "10^{-3}",
            "bnmGateAurocBound": f"{math.ceil(mx / 1e-5)} \\times 10^{{-5}}"}
    check("PRF", "BNM step-0 gate: AUROC tolerance 1e-3 on every fold", "1",
          float(all(g["auroc_tolerance"] == 1e-3 for g in gates)), "runs/bnm_r6_folds/*/run.json")
    bud = J("runs/bnm_r6_budget.json")
    E.update(bnmBudgetPreview=float(bud["stage2_preview_was"]),
             bnmBudgetSteps=float(bud["selected_budget_steps"]))
    # the 1:10 parts 2 and 3, and the main design's CORAL gate
    p2 = J("runs/dilute10_prereg_part2.json")
    want["dlPartTwoMaxStepsWord"] = {"10": "ten"}[re.search(r"at most (\d+)", p2["scope"]).group(1)]
    check("PRF", "part 2 supersedes CORAL's warn-only band at 1e-3 AUROC", "1",
          float("<= 1e-3" in p2["stop_rules_nb3_coral_freematch"]["only_stop_conditions"][0]),
          "runs/dilute10_prereg_part2.json")
    src = open(C.R("scripts", "run_coral_r4_confirm.py"), encoding="utf-8").read()
    E["coralGateWarnTpr"] = float(re.search(r"tol_abort=([\d.]+), auroc_tol_abort", src).group(1))
    ap = J("runs/dilute10_prereg_part3.json")["evidence"]["coral_nb3_300_step_summary"]["per_fold"]
    ap = [ap[k]["applied_updates_of_300"] for k in ("1", "2", "3", "4")]
    E.update(dlCoralCheckSteps=300.0, dlCoralCheckAppliedMin=float(min(ap)),
             dlCoralCheckAppliedMax=float(max(ap)))
    # the window declaration
    w = json.load(open(os.path.join(DATA, "window_average_D1.json"), encoding="utf-8"))
    sr = w["scatter_reduction"]

    def aoe(arm, k):    # achieved / expected, re-derived from the per-fold sds
        x = sr[arm][str(k)]
        return (st.mean(x["sd_1_per_fold"]) / st.mean(x["sd_k_per_fold"])) / math.sqrt(k)
    o8 = [aoe(a, 8) for a in ("FreeMatch x nb3", "DANN x nb3", "BNM x nb3")]
    E.update(winSqrtkLam=aoe("lambda=0", 8), winSqrtkMin=min(o8), winSqrtkMax=max(o8),
             winSqrtkCoral=aoe("CORAL x nb3", 4))
    arms8 = {n: w["arms"][a]["readouts"][ro]["mean_logit"]["8"]["tpr_per_fold"]
             for n, a, ro in (("DANN", "DANN x nb3", "4000"), ("CORAL", "CORAL x nb3", "2000"),
                              ("FM", "FreeMatch x nb3", "1500"), ("L0", "lambda=0", "4000"),
                              ("BNM", "BNM x nb3", "4000"))}
    sd8 = {n: st.stdev(x) for n, x in arms8.items()}
    check("PRF", "window k=8, cleaned pool: DANN's across-fold sd is the widest", "1",
          float(max(sd8, key=sd8.get) == "DANN"), "paper/data/window_average_D1.json")
    E.update(winSixDann=sd8["DANN"], winSixCoral=sd8["CORAL"])
    wo = J("runs/round7/window_average.json")["arms"]
    so = {a: st.stdev(wo[a]["readouts"][ro]["mean_logit"]["8"]["tpr_per_fold"])
          for a, ro in (("DANN x nb3", "4000"), ("BNM x nb3", "4000"), ("CORAL x nb3", "2000"),
                        ("FreeMatch x nb3", "1500"), ("lambda=0", "4000"))}
    check("PRF", "window k=8, original pool: BNM's across-fold sd is the widest", "1",
          float(max(so, key=so.get) == "BNM x nb3"), "runs/round7/window_average.json")
    E.update(winSixBnmOrig=so["BNM x nb3"], winSixDannOrig=so["DANN x nb3"])
    r8 = w["ordering_verdicts"]["row8_resolution_argument"]
    per8 = list(r8["by_k"]["8"]["per_arm"].values())
    d = r8["between_method_differences_7_18"]["DANN - CORAL"]
    E.update(winEightDiff=d, winEightSdLo=min(per8), winEightSdHi=max(per8),
             winEightMultLo=min(per8) / d, winEightMultHi=max(per8) / d)
    r9 = w["ordering_verdicts"]["row9_lambda0_stability"]
    check("PRF", "window: the control's range is narrower than DANN's at k=1 and k=8", "1",
          float(all(r9[k]["mean_within_fold_range"]["lambda=0"]
                    < r9[k]["mean_within_fold_range"]["DANN x nb3"] for k in ("1", "8"))),
          "paper/data/window_average_D1.json")
    # G2: the control's fold-4 completeness, as a count of lenses
    s4 = C.suite("lambda0", 4, 4000)
    want["lamFourCompLenses"] = {1: "one"}.get(round(s4["comp_at_purity"] * s4["n_pos"]), "?")
    for name, shown in sorted(P.items()):
        if name in want:
            check("PRF", f"{name} = {shown}", "1",
                  float(want[name] in (shown, shown + "}")), "file")   # _printed stops at "}"
        else:
            check("PRF", f"{name} = {shown}", shown, E[name], "recomputed (crosscheck)")
    missing = sorted(set(P) - set(E) - set(want))
    assert not missing, missing
    # Appendix C's window secondaries table: every cell from the per-fold lists
    lines = [l for l in _table_body("C_window_secondary.tex") if l.count("&") == 7
             and not l.startswith("Arm") and not l.startswith(" &")]
    keys = [("CORAL x nb3", "2000"), ("FreeMatch x nb3", "1500"), ("DANN x nb3", "4000"),
            ("DANN x nb3", "2000"), ("lambda=0", "4000"), ("BNM x nb3", "4000")]
    assert len(lines) == len(keys), lines
    for line, (arm, ro) in zip(lines, keys):
        cells = [c.strip() for c in _cells(line.rstrip("\\ "))][2:]
        exp = []
        for metric in ("auroc", "comp_at_purity"):
            for comb, k in (("mean_logit", 1), ("mean_logit", 8), ("mean_rank", 8)):
                x = w["arms"][arm]["readouts"][ro][comb][str(k)][f"{metric}_per_fold"]
                exp.append(None if any(t is None or not math.isfinite(t) for t in x) else st.mean(x))
        for c, e in zip(cells, exp):
            q = "n/a" if "n/a" in c else c.strip("$")
            check("PRF", f"C_window_secondary {arm} {ro}", q, e, "paper/data/window_average_D1.json")
    # Table 10: the DANN pre-registration's two rows, and the pre-fold digest macro
    ev = "evidence/prefold/dann_r5_prereg_kaggle_v2_20260902.json"
    sc = integrity.self_certifying_sha(C.R(ev))
    cur = J("runs/dann_r5_prereg.json")
    body = _table_body("A2_hashes_v6.tex")
    cp = [l for l in body if l.startswith("DANN pre-registration, pre-fold copy &")]
    rg = [l for l in body if l.startswith("DANN pre-registration, regenerated file &")]
    assert len(cp) == 1 and len(rg) == 1
    cp = [c.strip() for c in _cells(cp[0].rstrip("\\ "))]
    rg = [c.strip() for c in _cells(rg[0].rstrip("\\ "))]
    rec = J(ev)["recorded_utc"].replace("T", " ")[:16]
    check("PRF", "Table 10 pre-fold copy: digest, date, before, archived", "1",
          float(cp[6] == r"\texttt{" + sc[:16] + "}" and cp[1] == rec and cp[4] == "before"
                and cp[3] > rec and cp[7] == "no (archived)"), ev)
    check("PRF", "Table 10 regenerated file: digest, date, after", "1",
          float(rg[6] == r"\texttt{" + cur["sha256_of_body_without_this_field"][:16] + "}"
                and rg[1] == cur["recorded_utc"].replace("T", " ")[:16] and rg[1] > rg[3]
                and rg[4] == r"\textbf{after}"), "runs/dann_r5_prereg.json")
    bl = [c.strip() for c in _cells([l for l in body if l.startswith("DANN budget")][0])]
    check("PRF", "Table 10 DANN budget: a midnight record prints as a date", "1",
          float(J("runs/dann_r5_budget.json")["declared_utc"] == bl[1] + "T00:00:00Z"),
          "runs/dann_r5_budget.json")
    a2 = _printed("M_a2.tex")
    check("PRF", "\\dannPrefoldSha = the copy's self-certifying digest", "1",
          float(a2["dannPrefoldSha"] == sc[:8]), ev)
    # the four-decimal branch sentences print exactly part 1's template, filled
    sec = open(os.path.join(C.PAPER, "sections", "06_results.tex"), encoding="utf-8").read()
    check("PRF", "Section 6.4 prints no 'G =' or 'D =' in the branch sentences", "1",
          float("$G = " not in sec and "$D = " not in sec and "{$G = " not in sec),
          "paper/sections/06_results.tex")


def build_code_review() -> None:
    """PROJECTv7 7.66: every macro of tables/M_codereview.tex, recomputed here by running the
    code that ran (the transform, the losses) and reading the per-step logs, not through
    make_code_review_fixes.py; plus the qualitative facts the new sentences state."""
    import glob
    import hashlib
    import sys
    import numpy as np
    import torch
    sys.path.insert(0, C.REPO_SCRIPTS)
    from torchvision.transforms import v2
    import da_losses as DLS
    from lens_dataset import LensTransform
    P = _printed("M_codereview.tex")
    E, want = {}, {}
    J = lambda rel: json.load(open(C.R(rel), encoding="utf-8"))
    F = (1, 2, 3, 4)
    # augmentation: the transform objects themselves, and the largest shift actually drawn
    w, s = LensTransform("weak"), LensTransform("strong")
    tw = [type(t).__name__ for t in w._geom.transforms]
    check("CRV", "weak view = H-flip, V-flip, RandomAffine (no erase)", "1",
          float(tw == ["RandomHorizontalFlip", "RandomVerticalFlip", "RandomAffine"]
                and w._erase is None and all(t.p == 0.5 for t in w._geom.transforms[:2])),
          "scripts/lens_dataset.py")
    aff, saff = w._geom.transforms[2], s._geom.transforms[2]
    check("CRV", "rotation fills with zeros; 'none' has no geometry (test images unaugmented)",
          "1", float(aff.fill == 0.0 and LensTransform("none")._geom is None),
          "scripts/lens_dataset.py")
    torch.manual_seed(0)
    img = [torch.zeros(1, 64, 64)]
    sh = [max(abs(x) for x in aff.make_params(img)["translate"]) for _ in range(20000)]
    E["augShiftPx"] = float(max(sh))
    E["augRotDeg"] = float(aff.degrees[1])
    check("CRV", "rotation range is symmetric", "1", float(aff.degrees[0] == -aff.degrees[1]),
          "scripts/lens_dataset.py")
    E.update(augZoomLo=saff.scale[0], augZoomHi=saff.scale[1],
             augEraseLoPct=100 * s._erase.scale[0], augEraseHiPct=100 * s._erase.scale[1])
    check("CRV", "weak view has no zoom; the strong view erases with p = 0.5", "1",
          float(aff.scale is None and s._erase.p == 0.5), "scripts/lens_dataset.py")
    check("CRV", "Baseline A trains on cached features (no image loader)", "1",
          float("READS CACHED FEATURES ONLY" in open(C.R("scripts", "run_baseline.py"),
                                                      encoding="utf-8").read()),
          "scripts/run_baseline.py")
    # CORAL shrinkage: every record under the CORAL run directories that names it
    vals = set()
    for f in (glob.glob(C.R("kaggle", "coral_r4", "runs", "main", "extracted", "coral_r4_confirm",
                            "fold[1-4]", "run.json"))
              + glob.glob(C.R("runs", "dilute10", "coral_*", "fold[1-4]", "run*.json"))
              + [C.R("runs", "bakeoff_coral", "run_config.json")]):
        j = json.load(open(f, encoding="utf-8"))
        if "coral_shrinkage" in j:
            vals.add(j["coral_shrinkage"])
    assert len(vals) == 1, vals
    E["coralShrinkPct"] = 100 * vals.pop()
    c = DLS.CoralLoss(640, shrinkage=0.1, target_ema=False)
    x = torch.randn(64, 640, dtype=torch.float64).requires_grad_()
    y = torch.randn(64, 640, dtype=torch.float64).requires_grad_()
    cv = lambda z: torch.cov(z.detach().T)
    sk = lambda m: 0.9 * m + 0.1 * torch.trace(m) / 640 * torch.eye(640, dtype=m.dtype) \
        + c.jitter * torch.eye(640, dtype=m.dtype)
    ref = (x.detach().mean(0) - y.detach().mean(0)).pow(2).mean() + (sk(cv(x)) - sk(cv(y))).pow(2).mean()
    check("CRV", "CORAL = mean-of-entries terms after 10% shrinkage (not 1/(4d^2))", "1",
          float(abs(float(c(x, y).detach()) - float(ref)) < 1e-9 * float(ref)), "scripts/da_losses.py")
    # FreeMatch: the records, and the loss's own defaults
    fm = [J(f"runs/freematch_nb3_unfrozen_fold{k}/run_config.json") for k in F] + \
         [J(f"runs/dilute10/freematch_nb3/fold{k}/run_config.json") for k in F]
    for key, name, mult in (("sat_ema_momentum", "fmSatMomentum", 1),
                            ("saf_weight", "fmSafWeight", 1),
                            ("consistency_warmup_frac", "fmWarmupPct", 100)):
        v = {r[key] for r in fm}
        assert len(v) == 1, (key, v)
        E[name] = mult * v.pop()
    L = DLS.FreeMatchLoss()
    check("CRV", "FreeMatch fairness target: fixed uniform [0.5, 0.5]", "1",
          float(all(r["fairness_target"] is None for r in fm)
                and torch.equal(L.fairness_target.cpu(), torch.tensor([0.5, 0.5]))),
          "scripts/da_losses.py")
    check("CRV", "FreeMatch unfrozen stage: offset 8000 of an 8000 schedule -> mu = 1", "1",
          float(all(r["mu_step_offset"] == r["schedule_total_steps"] == 8000 for r in fm)
                and DLS.adamatch_mu_at_step(8000, 8000) == 1.0
                and DLS.adamatch_mu_at_step(4000, 8000) == 1.0
                and DLS.adamatch_mu_at_step(3999, 8000) < 1.0), "runs/*freematch*")
    # BNM: the loss on a batch equals -(nuclear norm)/2
    lg = torch.randn(256, 1, dtype=torch.float64)
    p = torch.sigmoid(lg.view(-1))
    nuc = torch.linalg.svdvals(torch.stack([1 - p, p], 1)).sum()
    E["bnmNucDenom"] = round(float(-nuc / DLS.bnm_loss(lg)), 6)  # bnm_loss is float32
    # DANN discriminator settings, from every DANN-path record; no learning-rate schedule
    recs = [J(f"runs/dann_main/fold{k}/extracted/fold{k}/run_config.json") for k in F] + \
           [J(f"runs/baseline_unfrozen/fold{k}/extracted/lambda0_r6_fold{k}/run_config.json")
            for k in F] + [J(f"runs/dilute10/dann/fold{k}/run_config.json") for k in F]
    want["dannDiscLr"] = rf"10^{{{int(round(math.log10(recs[0]['lr'])))}}}"
    assert len({r["lr"] for r in recs}) == len({r["weight_decay"] for r in recs}) == 1
    E["dannDiscWd"] = recs[0]["weight_decay"]
    ts = open(C.R("scripts", "train_skeleton.py"), encoding="utf-8").read()
    check("CRV", "no learning-rate scheduler in the training loop", "1",
          float("lr_scheduler" not in ts and "LambdaLR" not in ts), "scripts/train_skeleton.py")
    # skipped updates: summed from the per-step logs (the run.json totals of the control are
    # stale, written at steps_completed 0)
    def steps_sum(rel):
        rows = list(csv.DictReader(open(C.R(rel, "steps.csv"), encoding="utf-8")))
        return int(sum(float(r["step_skipped"]) for r in rows))
    lam = [steps_sum(f"runs/baseline_unfrozen/fold{k}/extracted/lambda0_r6_fold{k}") for k in F]
    dan = [steps_sum(f"runs/dann_main/fold{k}/extracted/fold{k}") for k in F]
    cor = [steps_sum("kaggle/coral_r4/runs/main/extracted/coral_r4_confirm/fold%d" % k) for k in F]
    want.update(skipLamList=", ".join(map(str, lam)), skipDannList=", ".join(map(str, dan)),
                skipCoralList=", ".join(map(str, cor)))
    E.update(lamSkipFoldTwo=lam[1], dannSkipFoldTwo=dan[1])
    fms = {round(J(f"runs/freematch_nb3_unfrozen_fold{k}/results.json")["freematch_stability"]
                 ["instrument"]["skip_rate_fp16"] * 1500) for k in F}
    assert len(fms) == 1, fms
    E["skipFmEach"] = fms.pop()
    # checkpoint spacing, from the logit file names on disk
    def spacing(pattern):
        out = set()
        for d in filter(os.path.isdir, glob.glob(C.R(*pattern))):
            st_ = sorted(int(f[5:-4]) for f in os.listdir(d)
                         if f.startswith("step_") and f.endswith(".npz") and f != "step_0.npz")
            out |= {b - a for a, b in zip(st_, st_[1:])}
        return out
    fmg = spacing(("runs", "freematch_nb3_unfrozen_fold[1-4]", "logits"))
    oth = (spacing(("runs", "dann_main", "fold[1-4]", "extracted", "fold[1-4]", "logits", "*"))
           | spacing(("runs", "bnm_r6_folds", "bnm_r6_fold[1-4]", "logits", "*"))
           | spacing(("kaggle", "coral_r4", "runs", "main", "extracted", "coral_r4_confirm",
                      "fold[1-4]", "logits", "*")))
    assert len(fmg) == len(oth) == 1, (fmg, oth)
    E.update(ckptFm=fmg.pop(), ckptOther=oth.pop())
    E.update(winEightSpanFm=8 * E["ckptFm"], winEightSpanOther=8 * E["ckptOther"])
    # CORAL's gate, by calling the band logic's thresholds as defined
    import inspect
    import run_coral_r4_confirm as RC
    sig = inspect.signature(RC.step0_gate)
    E["coralGateAurocAbort"] = sig.parameters["auroc_tol_abort"].default
    check("CRV", "CORAL gate: abort beyond 0.05 TPR (the existing macro)", "1",
          float(sig.parameters["tol_abort"].default == 0.05), "scripts/run_coral_r4_confirm.py")
    want["coralGateAurocWarn"] = rf"10^{{{int(round(math.log10(RC.R4.STEP0_AUROC_TOL)))}}}"
    # FreeMatch fold 1's record
    k1 = J("runs/freematch_nb3_unfrozen_fold1/run_config.json")
    k2 = J("runs/freematch_nb3_unfrozen_fold2/run_config.json")
    want["fmFoldOneMissingWord"] = {3: "three"}.get(len(set(k2) - set(k1)), "?")
    check("CRV", "FreeMatch fold 1 gated at tolerance 0", "1",
          float(k1["step0_gate"]["tolerance"] == 0.0), "runs/freematch_nb3_unfrozen_fold1")
    # the warm-up probe rolls back the optimiser only
    cs = open(C.R("scripts", "continuation_state.py"), encoding="utf-8").read()
    body = cs[cs.index("def warm_adam_moments"):]
    body = body[:body.index("\ndef ", 10)]
    check("CRV", "warm-up probe: 500 steps, optimiser rolled back to 200, nothing else", "1",
          float("DEFAULT_PROBE_STEPS = (50, 200, 500)" in cs and "opt.load_state_dict(state_at_n)"
                in body and "tau_global" not in body and "scaler.load_state_dict" not in body),
          "scripts/continuation_state.py")
    # 1:10: arms with a recorded helper digest that no shipped script matches
    lost = {}
    for f in glob.glob(C.R("runs", "dilute10", "*", "fold[1-4]", "run_config.json")) + \
            glob.glob(C.R("runs", "dilute10", "coral_nb3", "fold[1-4]", "run.json")):
        arm = f.replace("\\", "/").split("/")[-3]
        for name, pre in json.load(open(f, encoding="utf-8"))["dilute10"][
                "imported_module_sha256"].items():
            h = hashlib.sha256(open(C.R("scripts", name), "rb").read()).hexdigest()
            if not h.startswith(pre[:16]):
                lost.setdefault(arm, set()).add(name)
    want["dlLostArmsWord"] = {4: "four"}.get(len(lost), "?")
    want["dlLostFilesWord"] = {3: "three"}.get(min(len(v) for v in lost.values()), "?")
    check("CRV", "1:10 lost versions: DANN, BNM, CORAL and FreeMatch frozen stages", "1",
          float(set(lost) == {"dann", "bnm", "coral_frozen", "freematch_frozen"}
                and all(len(v) == 3 for v in lost.values())), "runs/dilute10/*")
    for name, shown in sorted(P.items()):
        if name in want:
            check("CRV", f"{name} = {shown}", "1",
                  float(want[name] in (shown, shown + "}")), "file")   # _printed stops at "}"
        else:
            check("CRV", f"{name} = {shown}", shown, E[name], "recomputed (crosscheck)")
    missing = sorted(set(P) - set(E) - set(want))
    assert not missing, missing


def build_final() -> None:
    """Final build (PROJECTv7 7.61): the acknowledgements quote the number of checks this
    script runs, as \\nCrossChecks (make_final_macros.py). Run last, so the total it is
    compared with is this run's own, this check included."""
    p = os.path.join(C.TABLES, "M_final.tex")
    m = re.findall(r"\\newcommand\{\\nCrossChecks\}\{(\d+)\}", open(p, encoding="utf-8").read())
    assert len(m) == 1, m
    check("FINAL", "acknowledgements: \\nCrossChecks = the cross-check total", m[0],
          len(CHECKS) + 1, "paper/data/crosscheck.json (this run)")


def main() -> int:
    with C.pool("orig"):
        build()
    n_orig = len(CHECKS)
    build_d1()
    build_zero_footnotes()
    n_d1 = len(CHECKS)
    build_a3_windows()
    build_k8_pairs()
    build_prose()
    build_claims()
    build_review()
    build_rewrite_v2()
    n_pre_v6 = len(CHECKS)
    build_dilute10()
    n_v6 = len(CHECKS) - n_pre_v6
    build_prereg_fixes()
    build_code_review()
    build_final()
    os.makedirs(DATA, exist_ok=True)
    fails = []
    print("=" * 100)
    print("CROSS-CHECK: every generated number against PROJECT.md "
          "(the doc is the CHECK, never the source)")
    print("=" * 100)
    print(f"  {'sec':<10}{'quantity':<62}{'PROJECT.md':>12}{'computed':>12}  ")
    for rec in CHECKS:
        rec["verdict"] = judge(rec)
        c = rec["computed"]
        shown = "n/a" if c is None else (f"{c:.6g}")
        flag = "" if rec["verdict"] == "PASS" else "   <<< MISMATCH"
        if rec["verdict"] != "PASS":
            fails.append(rec)
        print(f"  {rec['section']:<10}{rec['what'][:61]:<62}"
              f"{rec['quoted']:>12}{shown:>12}{flag}")

    out = os.path.join(DATA, "crosscheck.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"n_checks": len(CHECKS), "n_mismatch": len(fails),
                   "n_checks_prereg_pool_vs_project_md": n_orig,
                   "n_checks_d1_vs_audit": n_d1 - n_orig,
                   "n_checks_a3_windows_and_prose_macros": len(CHECKS) - n_d1 - n_v6,
                   "n_checks_dilute10_v6": n_v6,
                   "checks": CHECKS}, fh, indent=1)
    print("-" * 100)
    print(f"  {len(CHECKS)} checks, {len(fails)} mismatch(es) -> "
          f"{os.path.relpath(out, C.REPO)}")
    print(f"    pre-registered pool vs PROJECT.md: {n_orig}; "
          f"D1 / A7 vs the exact-ID audit: {n_d1 - n_orig}; "
          f"A3 window rows + prose macros: {len(CHECKS) - n_d1 - n_v6}; "
          f"1:10 experiment (paper v6): {n_v6}")
    print(f"    retired by rewrite v2 (Addendum 4; A3 row 1, fold-averaged): {len(RETIRED)}")
    for r in RETIRED:
        print(f"      retired  {r}")
    for rec in fails:
        c = rec["computed"]
        print(f"    MISMATCH  sec {rec['section']}  {rec['what']}\n"
              f"              PROJECT.md {rec['quoted']}   computed "
              f"{'n/a' if c is None else repr(c)}\n"
              f"              artifact: {rec['source']}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
