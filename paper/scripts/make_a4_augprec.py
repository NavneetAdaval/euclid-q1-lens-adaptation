r"""
make_a4_augprec.py -- Appendix A4: the round-1 augmentation / precision 2x2, three cells.
=========================================================================================

    python paper/scripts/make_a4_augprec.py

Reads the persisted analysis artifact ``runs/lever1_aug_precision.json`` (written by
``scripts/decompose_aug_precision.py`` from each cell's saved score dump: class-conditional
paired bootstrap, B = 2000, seed 0, 95% percentile interval) and checks every cell against
that cell's own ``results.json``, so the appendix and the runs cannot disagree silently.
The trainable-parameter count is counted from cell A's saved checkpoint, not typed.

Writes ``paper/tables/A4_augprec.tex`` (the table) and ``paper/tables/M_augprec.tex``
(macros the prose in Section 4 and Appendix A4 quote). Nothing else.

Cell D (aug = none at fp16) was never run; the table says so rather than leaving a gap.
"""
from __future__ import annotations

import json
import math
import os

import re
import sys

import _common as C

ART = C.R("runs", "lever1_aug_precision.json")
CELLS = [  # key in the artifact, run directory, label
    ("A_none_fp32", "baselineA_fold0_nb0_image_noaug_fp32", "A", "none", "fp32"),
    ("B_weak_fp32", "baselineA_fold0_nb0_image_weak_fp32", "B", "weak", "fp32"),
    ("C_weak_fp16", "baselineA_fold0_nb0_image", "C", "weak", "fp16"),
]


def _macro(name, value):
    return rf"\newcommand{{\{name}}}{{{value}}}"


def main() -> int:
    # On the pre-registered pool: the project's artifact. On D1/D2: the same script's
    # recomputation on that pool (paper/data/lever1_aug_precision_<pool>.json), which
    # build_pool_recomputes.py writes only after the pre-registered pool reproduced ART.
    a = json.load(open(C.art("lever1_aug_precision"), encoding="utf-8"))

    # every cell must agree with its own run to 1e-9 -- the appendix quotes the runs
    # (each run's results.json was scored on the pre-registered pool)
    for key, run, *_ in CELLS:
        r = json.load(open(C.R("runs", run, "results.json"), encoding="utf-8"))
        if C.POOL == "orig":
            for m in ("tpr", "auroc"):
                assert abs(a["cells"][key][m] - r[m]) < 1e-9, f"{key} {m} disagrees with {run}"
        assert r["n_blocks"] == 0 and r["n_pos"] == a["n_pos"], f"{run} is not the frozen fold-0 cell"

    # Cell C is NOT the Baseline A run reported elsewhere. That run is the cached-feature,
    # un-augmented fp32 model (T5's fold-0 column, via the same C.suite call); cell A
    # reproduces it. Cell C's own configuration was also run on Kaggle, and the two venues
    # differ -- both numbers come from the screen report that measured the difference.
    base_f0 = C.suite("baseline", 0, 8000)["tpr"]
    if C.POOL == "orig":
        assert abs(base_f0 - a["cells"]["A_none_fp32"]["tpr"]) < 1e-9, \
            "cell A no longer reproduces Baseline A's fold-0 value"
    elif abs(base_f0 - a["cells"]["A_none_fp32"]["tpr"]) >= 1e-9:
        # Appendix A4's prose states this difference (one positive) on this pool; that
        # statement is asserted by make_m_prose.py's prose_claims (claim Ha).
        print(f"  note: on pool {C.POOL} cell A is {a['cells']['A_none_fp32']['tpr']:.4f} and "
              f"Baseline A's fold-0 value is {base_f0:.4f}, as Appendix A4 states "
              f"(asserted in make_m_prose.py, claim Ha)")
    venue = (json.load(open(C.R("runs", "lever1_screen_report.json"),
                            encoding="utf-8"))["venue_portability"]
             if C.POOL == "orig" else a["venue_portability"])
    assert abs(venue["tpr_local"] - a["cells"]["C_weak_fp16"]["tpr"]) < 1e-9, \
        "the screen report's local run is not cell C"
    venue_lenses = round(venue["diff"] / a["one_lens"])
    # A4 says the venue difference is as large as the precision leg; hold it to that.
    assert abs(abs(a["precision_effect_at_weak_aug"]["diff"]) - abs(venue["diff"])) < 1e-9, \
        "the venue difference and the precision leg are no longer the same size"

    if "--count-from-checkpoint" in sys.argv[1:]:
        import torch
        ck = torch.load(C.R("runs", CELLS[0][1], "checkpoints", "step_8000.pt"),
                        map_location="cpu", weights_only=False)
        n_trainable = sum(int(t.numel()) for t in ck["head"].values())
    else:
        # The count does not depend on the test pool. A default rebuild opens no
        # checkpoint and keeps the value this script last counted from it.
        m = re.search(r"\\newcommand\{\\augprecTrainable\}\{([0-9{},]+)\}",
                      open(os.path.join(C.PAPER, "tables", "M_augprec.tex"),
                           encoding="utf-8").read())
        n_trainable = int(m.group(1).replace("{,}", ""))

    one = a["one_lens"]
    legs = {"aug": a["augmentation_effect_at_fp32"],
            "prec": a["precision_effect_at_weak_aug"],
            "tot": a["total"]}
    for leg in legs.values():                                    # the three verdicts
        lo, hi = leg["ci"]
        leg["excludes_zero"] = (lo > 0) or (hi < 0)
    # "the total's upper bound sits one positive from zero" -- checked, not asserted
    tot_hi_lenses = legs["tot"]["ci"][1] / one
    assert legs["tot"]["excludes_zero"], "the total no longer excludes zero"
    assert not legs["aug"]["excludes_zero"] and not legs["prec"]["excludes_zero"], \
        "a leg now resolves on its own; the appendix prose would be wrong"
    assert abs(tot_hi_lenses + 1.0) < 1e-6, \
        f"total's upper bound is {tot_hi_lenses:.3f} positives from zero, not exactly one"
    resid = legs["tot"]["diff"] - (legs["aug"]["diff"] + legs["prec"]["diff"])
    assert abs(resid) < 1e-12, "the A->B->C path no longer decomposes exactly"
    lens = {k: round(v["diff"] / one) for k, v in legs.items()}   # whole lenses on 98

    rows = []
    for key, _run, lab, aug, prec in CELLS:
        c = a["cells"][key]
        rows.append(f"{lab} & {aug} & {prec} & {C.f4(c['tpr'])} & {C.f4(c['auroc'])} \\\\")
    rows.append(r"D & none & fp16 & \emph{not run} & \emph{not run} \\")

    def legrow(name, path, leg):
        lo, hi = legs[leg]["ci"]
        return (f"{name} & {path} & {C.f4(legs[leg]['diff'], signed=True)} & "
                f"{C.ci(lo, hi)} & {'yes' if legs[leg]['excludes_zero'] else 'no'} \\\\")

    body = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Round-1 augmentation and precision, three cells of a $2\times2$.} "
        r"Frozen encoder (Baseline A), development fold 0, "
        + f"{n_trainable:,}".replace(",", "{,}")
        + r" trainable parameters, 98 test positives. Cell D was never run, so the "
        r"interaction between the two factors is not identified. Differences are paired "
        r"class-conditional bootstrap estimates (2000 replicates, 95\% percentile "
        r"interval); the one-positive resolution is $1/98$. The two legs sum to the total "
        r"exactly because $A\to B\to C$ is a chain, which is arithmetic and not evidence "
        r"that the effects add.}",
        r"\label{tab:augprec}",
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Cell & Augmentation & Precision & TPR@$10^{-3}$ & AUROC \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{6pt}",
        r"\begin{tabular}{llrlr}",
        r"\toprule",
        r"Leg & Path & $\Delta$TPR@$10^{-3}$ & 95\% interval & Excludes 0 \\",
        r"\midrule",
        legrow("Augmentation (held at fp32)", "$B-A$", "aug"),
        legrow("Precision (held at weak augmentation)", "$C-B$", "prec"),
        legrow("Total", "$C-A$", "tot"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    C.write_table(os.path.join(C.TABLES, "A4_augprec.tex"), body,
                  header_comment=("A4 -- runs/lever1_aug_precision.json, checked against each "
                                  "cell's results.json;\nparameter count from cell A's "
                                  "checkpoint." if C.POOL == "orig" else
                                  "A4 -- paper/data/lever1_aug_precision_" + C.POOL
                                  + ".json (the same script on test pool " + C.POOL
                                  + ");\nparameter count from cell A's checkpoint."),
                  placement="htbp")

    tlo, thi = legs["tot"]["ci"]
    macros = "\n".join([
        _macro("augprecTrainable", f"{n_trainable:,}".replace(",", "{,}")),
        _macro("augprecTotal", C.f4(legs["tot"]["diff"], signed=True)),
        _macro("augprecTotalCI", C.ci(tlo, thi)),
        _macro("augprecTotalLenses", str(abs(lens["tot"]))),
        _macro("augprecAug", C.f4(legs["aug"]["diff"], signed=True)),
        _macro("augprecAugCI", C.ci(*legs["aug"]["ci"])),
        _macro("augprecAugLenses", str(abs(lens["aug"]))),
        _macro("augprecPrec", C.f4(legs["prec"]["diff"], signed=True)),
        _macro("augprecPrecCI", C.ci(*legs["prec"]["ci"])),
        _macro("augprecPrecLenses", str(abs(lens["prec"]))),
        _macro("augprecCellA", C.f4(a["cells"]["A_none_fp32"]["tpr"])),
        _macro("augprecCellC", C.f4(a["cells"]["C_weak_fp16"]["tpr"])),
        _macro("augprecOnePos", C.f4(one)),
        _macro("augprecBaselineFoldZero", C.f4(base_f0)),
        _macro("augprecCellCKaggle", C.f4(venue["tpr_kaggle"])),
        _macro("augprecVenueDiff", C.f4(venue["diff"])),
        _macro("augprecVenueLenses", str(venue_lenses)),
        _macro("augprecVenueCI", C.ci(*venue["ci"])),
    ])
    C.write_table(os.path.join(C.TABLES, "M_augprec.tex"), macros,
                  header_comment="Macros for Section 4 and Appendix A4, from "
                                 + ("runs/lever1_aug_precision.json." if C.POOL == "orig" else
                                    "paper/data/lever1_aug_precision_" + C.POOL + ".json."),
                  size="")
    print(f"  total {legs['tot']['diff']:+.4f} CI {tlo:+.4f},{thi:+.4f} "
          f"(upper bound {tot_hi_lenses:+.3f} positives); trainable {n_trainable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
