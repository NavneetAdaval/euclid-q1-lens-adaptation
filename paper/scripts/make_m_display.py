r"""
make_m_display.py -- main-text display variants of numbers the tables already carry.
====================================================================================

    python paper/scripts/make_m_display.py

The main text prints rates and differences at two decimals and percentages as whole
numbers; the appendix tables keep four decimals. This script computes each displayed
quantity from the SAME ``_common`` call (or artifact) that fills its four-decimal table and
rounds it once. No existing macro is redefined and no stored value changes, so
``audit/decontam/prose_macros.py verify`` is unaffected; every macro here is a new name.

Families:

  * ``\dFm*``, ``\dSd*``   -- fold-means and spreads, ``_common.fold_mean`` (Table 2);
  * ``\dCt*``              -- the paired contrasts, ``_common.contrast`` over
                              ``_common.CONTRAST_ROWS`` (Table 3), with ``Abs`` variants;
  * ``\dHead*``, ``\dGain*`` -- ranges over the headline readouts and the target-signal rows;
  * ``\dLr*``              -- benchmark ratios (whole per cent), the Lines-ratio artifact;
  * ``\dPool*``            -- the original-pool and D2 changes (Appendix G), via ``_common``
                              on the ``orig``, ``D1`` and ``D2`` pools;
  * ``\dFrozen*``          -- the frozen comparison (Table 1), ``_common.round1_summary`` and
                              the round-1 paired artifacts.

Writes ``paper/tables/M_display.tex`` and ``paper/data/display_macros.json``.
"""
from __future__ import annotations

import json
import math
import os

import _common as C

MACROS: list = []          # (name, printed, value, source)


def m(name, value, nd=2, signed=False, source="", pct=False):
    v = float(value)
    shown = (f"{v:.0f}" if pct else (f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"))
    if not pct and math.isfinite(v) and round(v, nd) == 0.0:
        shown = f"{0.0:.{nd}f}"            # a value that rounds to zero prints unsigned
    if not math.isfinite(v):
        shown = r"\text{n/a}"
    MACROS.append((name, shown, v, source))


# CONTRAST_ROWS index -> display name
ROWNAME = {0: "LamBase", 1: "DannLam", 2: "CoralLamM", 3: "FmLamM", 4: "BnmLam",
           5: "CoralLamP", 6: "FmLamP", 7: "BnmCoral", 8: "BnmDann", 9: "BnmFm",
           10: "DannCoral", 11: "DannFm", 12: "DannCoralM"}
HEAD = [("Baseline", "baseline", 8000), ("Lambda", "lambda0", 4000), ("Bnm", "bnm", 4000),
        ("FreeMatch", "freematch", 1500), ("Dann", "dann", 4000), ("DannSens", "dann", 2000),
        ("Coral", "coral", 2000)]
DA_HEAD = ["Bnm", "FreeMatch", "Dann", "Coral"]


def build() -> list:
    MACROS.clear()
    T2, T3 = "Table 2 (_common.fold_mean, D1)", "Table 3 (_common.contrast, D1)"
    with C.pool("D1"):
        means = {}
        for key, arm, ro in HEAD:
            mean, sd, _ = C.fold_mean(arm, "tpr", ro)
            means[key] = mean
            m(f"dFm{key}", mean, source=T2)
            m(f"dSd{key}", sd, source=T2)
        da = [means[k] for k in DA_HEAD]
        m("dHeadMin", min(da), source=T2 + ": min over BNM, FreeMatch, DANN, CORAL headlines")
        m("dHeadMax", max(da), source=T2 + ": max over BNM, FreeMatch, DANN, CORAL headlines")
        cts = {}
        for i, (_p, _l, a, ra, b, rb, _m, _k) in enumerate(C.CONTRAST_ROWS):
            r = C.contrast(a, ra, b, rb)
            cts[i] = r
            n = ROWNAME[i]
            m(f"dCt{n}", r["mean"], signed=True, source=T3 + f" row {i}")
            m(f"dCt{n}Lo", r["ci_lo"], signed=True, source=T3 + f" row {i}")
            m(f"dCt{n}Hi", r["ci_hi"], signed=True, source=T3 + f" row {i}")
            m(f"dCt{n}Abs", abs(r["mean"]), source=T3 + f" row {i}")
            m(f"dCt{n}AbsLo", min(abs(r["ci_lo"]), abs(r["ci_hi"])), source=T3 + f" row {i}")
            m(f"dCt{n}AbsHi", max(abs(r["ci_lo"]), abs(r["ci_hi"])), source=T3 + f" row {i}")
        gains = [cts[i]["mean"] for i in (1, 4, 5, 6)]
        m("dGainMin", min(gains), source=T3 + ": min over rows 1, 4, 5, 6")
        m("dGainMax", max(gains), source=T3 + ": max over rows 1, 4, 5, 6")

    # ---------------- benchmark ratios (whole per cent), the Lines-ratio artifact ----------------
    with C.pool("D1"):
        lr = json.load(open(C.art("lines_ratios"), encoding="utf-8"))["arms"]
    LR = "lines_ratios_D1.json"
    pcts = {}
    for key, arm, ro in [("Coral", "CORAL x nb3", 2000), ("FreeMatch", "FreeMatch x nb3", 1500),
                         ("Bnm", "BNM x nb3", 4000), ("Dann", "DANN x nb3", 4000),
                         ("DannSens", "DANN x nb3", 2000)]:
        rec = lr[arm]["readouts"][str(ro)]
        assert rec["defined_on_all_folds"], arm
        pcts[key] = rec["pct_of_lines"]
        m(f"dLr{key}", rec["pct_of_lines"], pct=True, source=LR + f" {arm} {ro} pct_of_lines")
        lo, hi = rec["pct_of_lines_ci95"]
        m(f"dLr{key}Lo", lo, pct=True, source=LR + f" {arm} {ro} pct_of_lines_ci95")
        m(f"dLr{key}Hi", hi, pct=True, source=LR + f" {arm} {ro} pct_of_lines_ci95")
        m(f"dComp{key}", rec["fold_mean"], source=LR + f" {arm} {ro} fold_mean completeness")
    b = lr["Baseline A (frozen)"]["readouts"]["8000"]["folds_1_4"]
    m("dLrBaseline", b["pct_of_lines"], pct=True, source=LR + " Baseline A folds_1_4 pct_of_lines")
    m("dLrBaselineLo", 100 * b["ci95"][0] / C.LINES_COMPLETENESS, pct=True,
      source=LR + " Baseline A folds_1_4 ci95 / 0.50")
    m("dLrBaselineHi", 100 * b["ci95"][1] / C.LINES_COMPLETENESS, pct=True,
      source=LR + " Baseline A folds_1_4 ci95 / 0.50")
    da_pct = [pcts[k] for k in DA_HEAD]
    m("dLrDaMin", min(da_pct), pct=True, source=LR + ": min over the four DA headlines")
    m("dLrDaMax", max(da_pct), pct=True, source=LR + ": max over the four DA headlines")

    # ---------------- Appendix G: original pool -> cleaned pool, and D2 ----------------
    rows = [("coral", 2000), ("freematch", 1500), ("dann", 4000), ("dann", 2000),
            ("lambda0", 4000), ("bnm", 4000), ("baseline", 8000)]
    up, d2 = [], []
    for arm, ro in rows:
        with C.pool("orig"):
            o = C.fold_mean(arm, "tpr", ro)[0]
        with C.pool("D1"):
            d1 = C.fold_mean(arm, "tpr", ro)[0]
        with C.pool("D2"):
            dd = C.fold_mean(arm, "tpr", ro)[0]
        up.append(d1 - o)
        d2.append(abs(dd - d1))
    assert all(x > 0 for x in up), up
    G = "Appendix G (_common.fold_mean on pools orig, D1, D2)"
    m("dPoolUpMin", min(up), source=G + ": min D1 - orig over arms")
    m("dPoolUpMax", max(up), source=G + ": max D1 - orig over arms")
    m("dPoolDTwoMax", max(d2), source=G + ": max |D2 - D1| over arms")

    # ---------------- Table 1: the frozen comparison ----------------
    with C.pool("D1"):
        F = "Table 1 (_common.round1_summary, D1)"
        for key, run, tag in [("Base", "bakeoff_baseline_A", None), ("Coral", "bakeoff_coral", "coral"),
                              ("Ada", "bakeoff_adamatch_5fold", "adamatch"),
                              ("Fm", "bakeoff_freematch_5fold", "freematch")]:
            rs = C.round1_summary(run)
            m(f"dFrozen{key}", rs["across_fold"]["tpr"]["mean"], source=F)
            if tag:
                agg = json.load(open(os.path.join(C.DATA, f"round1_paired_{tag}_D1.json"),
                                     encoding="utf-8"))["aggregate"]["tpr"]
                m(f"dFrozen{key}Diff", agg["diff"], signed=True, source=F + " paired")
                m(f"dFrozen{key}Lo", agg["ci"][0], signed=True, source=F + " paired")
                m(f"dFrozen{key}Hi", agg["ci"][1], signed=True, source=F + " paired")
        lp = json.load(open(C.art("foldmean_bootstrap"), encoding="utf-8"))
        m("dLpDiff", lp["diff"], signed=True, source="foldmean_bootstrap_D1.json")
        m("dLpLo", lp["ci"][0], signed=True, source="foldmean_bootstrap_D1.json")
        m("dLpHi", lp["ci"][1], signed=True, source="foldmean_bootstrap_D1.json")
        m("dAuLambda", C.fold_mean("lambda0", "auroc", 4000)[0], source="Table A (AUROC)")

    # ---------------- the development-fold controlled pair (Figure: fold-0 curves) ----------------
    import make_m_prose as MP
    with C.pool("D1"):
        cv, init = MP.fold0_curves()
    lam = cv["arm1_lambda0"]
    m("dFzLambdaLast", lam[max(lam, key=int)], source="fold0_curves_D1.json (lambda=0, last step)")
    m("dFzInit", init, source="fold0_curves_D1.json (frozen-probe start, Baseline A fold 0)")
    return MACROS


def main() -> int:
    build()
    lines, last = [], None
    for name, shown, _v, source in MACROS:
        s = source.split(":")[0]
        if s != last:
            lines.append(f"% {s}")
            last = s
        lines.append(rf"\newcommand{{\{name}}}{{{shown}}}")
    C.write_table(os.path.join(C.TABLES, "M_display.tex"), "\n".join(lines), size="",
                  header_comment="Main-text display variants (two decimals, whole per cent), "
                                 "computed from the same calls as the four-decimal tables.")
    os.makedirs(C.DATA, exist_ok=True)
    with open(os.path.join(C.DATA, "display_macros.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump([{"name": n, "printed": s, "value": v, "source": src}
                   for n, s, v, src in MACROS], fh, indent=1)
    print(f"  {len(MACROS)} display macros written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
