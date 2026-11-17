r"""
make_m_venue.py -- macros for Section 4's venue-and-precision paragraph.
=========================================================================

    python paper/scripts/make_m_venue.py

Reads, and asserts before emitting:

  * every unfrozen arm's recorded training and evaluation precision, and which arms record
    a GPU name -- via ``audit_venue.record`` on each fold's own run files;
  * the one cross-venue re-scoring that was checked: CORAL x nb3 fold 0's development run
    (``kaggle/coral_r4/runs/fold0_run/dev8000_metric_curve.json``, whose run records a
    Tesla T4) against a second scoring of the same weights stored in
    ``runs/coral_r4_logits`` (whose files record no device). TPR@1e-3 and AUROC are
    recomputed here from the second scoring's logits, step by step.

The measured venue difference itself (the frozen baseline, weak augmentation, fp16) is
already emitted by ``make_a4_augprec.py`` from ``runs/lever1_screen_report.json`` and is not
duplicated here.

Writes ``paper/tables/M_venue.tex`` and ``paper/data/venue_coral_repro.json``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import _common as C
import audit_venue as AV

sys.path.insert(0, C.REPO_SCRIPTS)
from metrics import auroc, tpr_at_fpr  # noqa: E402

NB3 = {k: v for k, v in [(a[0], a[2]) for a in AV.ARMS]
       if k in ("lambda=0 (nb3)", "BNM x nb3", "FreeMatch x nb3", "DANN x nb3",
                "CORAL x nb3")}


def sci(x, nd=1):
    mant, exp = f"{float(x):.{nd}e}".split("e")
    return rf"${mant} \times 10^{{{int(exp)}}}$"


def main() -> int:
    recs = {arm: [AV.record(d(f)) for f in (1, 2, 3, 4)] for arm, d in NB3.items()}
    train = {r["precision"] for rs in recs.values() for r in rs}
    evalp = {r["eval_precision"] for rs in recs.values() for r in rs}
    assert train == {"fp16"} and evalp == {"fp32"}, (train, evalp)
    gpu_arms = sorted(a for a, rs in recs.items() if all(r["gpu_recorded"] for r in rs))
    gpus = {r["gpu_recorded"] for a in gpu_arms for r in recs[a]}
    no_gpu = sorted(a for a, rs in recs.items() if not any(r["gpu_recorded"] for r in rs))
    assert gpus == {"Tesla T4"} and no_gpu == ["FreeMatch x nb3"], (gpus, no_gpu)
    assert all("cuda" in r["venue"] for r in recs["FreeMatch x nb3"])

    # the CORAL x nb3 cross-scoring check, recomputed
    dev = C.R("kaggle", "coral_r4", "runs", "fold0_run")
    curve = {r["step"]: r for r in json.load(open(os.path.join(dev, "dev8000_metric_curve.json"),
                                                  encoding="utf-8"))}
    t4 = json.load(open(os.path.join(dev, "coral_r4_nb3_dev8000_fold0", "run.json"),
                        encoding="utf-8"))
    assert t4.get("gpu_name") == "Tesla T4"
    steps, rows = [0] + list(range(250, 2001, 250)), []
    for s in steps:
        p = C.R("runs", "coral_r4_logits",
                "true_init.npz" if s == 0 else os.path.join("g16", f"step_{s}.npz"))
        z = np.load(p)
        y = np.r_[np.ones(z["pos"].size), np.zeros(z["neg"].size)]
        sc = np.r_[z["pos"], z["neg"]]
        rows.append({"step": s, "tpr_second": tpr_at_fpr(y, sc, 1e-3),
                     "tpr_t4": curve[s]["tpr"], "auroc_second": auroc(y, sc),
                     "auroc_t4": curve[s]["auroc"]})
    same = sum(abs(r["tpr_second"] - r["tpr_t4"]) < 1e-12 for r in rows)
    max_au = max(abs(r["auroc_second"] - r["auroc_t4"]) for r in rows)
    assert same == len(rows), "the CORAL re-scoring no longer matches at every checkpoint"
    json.dump({"t4_run_gpu": t4["gpu_name"], "second_scoring_venue": "not recorded",
               "rows": rows, "identical_tpr": same, "max_auroc_diff": max_au},
              open(os.path.join(C.DATA, "venue_coral_repro.json"), "w",
                   encoding="utf-8", newline="\n"), indent=1)

    names = {"lambda=0 (nb3)": r"$\lambda{=}0$", "BNM x nb3": "BNM", "DANN x nb3": "DANN",
             "CORAL x nb3": "CORAL"}
    listed = [names[a] for a in ("CORAL x nb3", "DANN x nb3", "BNM x nb3", "lambda=0 (nb3)")
              if a in gpu_arms]
    macros = "\n".join([
        rf"\newcommand{{\venueTrainPrecision}}{{{train.pop()}}}",
        rf"\newcommand{{\venueEvalPrecision}}{{{evalp.pop()}}}",
        rf"\newcommand{{\venueGpu}}{{Tesla~T4}}",
        rf"\newcommand{{\venueGpuArms}}{{{', '.join(listed[:-1])} and {listed[-1]}}}",
        rf"\newcommand{{\venueCoralCheckpoints}}{{{len(rows)}}}",
        rf"\newcommand{{\venueCoralMaxAuroc}}{{{sci(max_au)}}}",
    ])
    C.write_table(os.path.join(C.TABLES, "M_venue.tex"), macros,
                  header_comment="Macros for Section 4's venue paragraph; sources in "
                                 "paper/scripts/make_m_venue.py.", size="")
    print(f"  nb3 precision {train or 'fp16'}/{evalp or 'fp32'}; GPU recorded for {gpu_arms}; "
          f"CORAL re-scoring identical at {same}/{len(rows)}, max AUROC diff {max_au:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
