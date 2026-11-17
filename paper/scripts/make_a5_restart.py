r"""
make_a5_restart.py -- Appendix A5: reconstructing training state across a checkpoint restart.
=============================================================================================

    python paper/scripts/make_a5_restart.py

Every number Appendix A5 and Section 8 quote about the restart comes from here:

  * the four keys a round-1 checkpoint stores -- read from a round-1 checkpoint itself;
  * the V-shaped excursion -- from the UNFIXED frozen control's ``curve.csv``;
  * the reconstructed control and round 1's own tail -- the FIXED control's ``curve.csv``
    and ``runs/round1_tail_stats.json`` (population sd, the convention that file uses);
  * the fix settings -- the FIXED control's ``run_config.json`` and the defaults in
    ``scripts/continuation_state.py``;
  * the measured reconstruction -- parsed from the FIXED control's own log, each line
    asserted present so a changed log format fails loudly rather than printing nothing;
  * the step-0 gate -- every FreeMatch x nb3 fold's recorded ``step0_gate``.

Writes ``paper/tables/M_restart.tex`` (macros) and ``paper/tables/A5_gate.tex`` (table).
"""
from __future__ import annotations

import csv
import inspect
import json
import os
import re
import sys

import numpy as np
import torch

import _common as C

UNFIXED = C.R("runs", "freematch_fold0_nb0_control")
FIXED = C.R("runs", "freematch_fold0_nb0_control_FIXED")
ROUND1_CKPT = C.R("runs", "bakeoff_freematch_5fold", "fold0", "checkpoints", "step_8000.pt")
TAIL = C.R("runs", "round1_tail_stats.json")
GATE_RUNS = [(0, "freematch_fold0_nb3_unfrozen", "development")] + \
            [(f, f"freematch_nb3_unfrozen_fold{f}", "confirmatory") for f in (1, 2, 3, 4)]

sys.path.insert(0, C.REPO_SCRIPTS)
import continuation_state as CS  # noqa: E402


def curve(run_dir):
    with open(os.path.join(run_dir, "curve.csv"), newline="", encoding="utf-8") as fh:
        return [(int(r["step"]), float(r["fold0_tpr_1e-3"])) for r in csv.DictReader(fh)]


def grab(pattern, text, what):
    m = re.search(pattern, text)
    assert m, f"log line for {what} not found -- has the log format changed?"
    return m


def sci(x, nd=1):
    """Scientific notation that renders in LaTeX, unlike a bare 1.2e-06."""
    mant, exp = f"{float(x):.{nd}e}".split("e")
    return rf"${mant} \times 10^{{{int(exp)}}}$"


def m_(name, value):
    return rf"\newcommand{{\{name}}}{{{value}}}"


def main() -> int:
    one = 1.0 / json.load(open(os.path.join(UNFIXED, "results.json"), encoding="utf-8"))["n_pos"]

    keys = list(torch.load(ROUND1_CKPT, map_location="cpu", weights_only=False).keys())
    assert len(keys) == 4, f"round-1 checkpoint holds {keys}, not four keys"

    cfg_u = json.load(open(os.path.join(UNFIXED, "run_config.json"), encoding="utf-8"))
    mu_zero = int(round(cfg_u["consistency_warmup_frac"] * cfg_u["schedule_total_steps"]))

    u = curve(UNFIXED)
    s0 = u[0][1]
    trough_step, trough = min(u, key=lambda x: x[1])
    end = u[-1][1]
    maxdrift = max(abs(t - s0) for _s, t in u)
    assert trough_step < mu_zero, "the trough no longer sits inside the mu = 0 window"

    f = curve(FIXED)
    f_tail = np.array([t for s, t in f if s > 0])
    f_maxdrift = max(abs(t - f[0][1]) for _s, t in f)
    tail = json.load(open(TAIL, encoding="utf-8"))["0"]

    cfg_f = json.load(open(os.path.join(FIXED, "run_config.json"), encoding="utf-8"))
    log = open(os.path.join(FIXED, "nb0_control_v2.log"), encoding="utf-8",
               errors="replace").read()
    sat = grab(r"\[sat\] reconstructing .*?: (\d+) passes over a (\d+)-image pool at batch "
               r"(\d+) -> (\d+) batches", log, "SAT reconstruction")
    tau = grab(r"\[sat\] tau_global : direct ([0-9.]+) \| EMA\((\d+)\)", log, "tau")
    agree = grab(r"bleed-corrected \|diff\| ([0-9.eE+-]+)\n", log, "tau agreement")
    pbar = grab(r"bleed-corrected max\|diff\| ([0-9.eE+-]+)\)", log, "pbar agreement")
    carried = grab(r"step counter 0 -> (\d+) \(carried into training: (True|False)\)", log,
                   "Adam step counter")
    debiased = grab(r"DEBIASED second moment moved ([+-][0-9.]+)%", log, "debiased move")
    # Name the interval that percentage is measured over, and prove it: the logged change
    # must equal (debiased at the probe's end - debiased at the warm-up's end) / the latter.
    probe = grab(r"warming moments for (\d+) steps \(probing to (\d+)\)", log, "probe range")
    warm_n, probe_end = int(probe.group(1)), int(probe.group(2))
    dv = {int(s): float(v) for s, v in re.findall(
        r"\[adam\]\s+step\s+(\d+): .*?debiased = ([0-9.eE+-]+)", log)}
    pct = 100.0 * (dv[probe_end] - dv[warm_n]) / dv[warm_n]
    assert abs(pct - float(debiased.group(1))) < 0.01, \
        f"logged {debiased.group(1)}% is not the step {warm_n} -> {probe_end} change ({pct:.2f}%)"
    assert int(sat.group(1)) == cfg_f["sat_passes"] == CS.DEFAULT_PASSES
    assert int(tau.group(2)) == CS.DEFAULT_EMA_ITERS
    assert int(carried.group(1)) == cfg_f["adam_warmup_steps"] == CS.DEFAULT_WARM_STEPS
    assert carried.group(2) == "True"
    tol = inspect.signature(CS.sat_check_or_raise).parameters["tol"].default
    assert float(agree.group(1)) <= tol and float(pbar.group(1)) <= tol

    # the per-fold step-0 gate, as each run recorded it
    rows, gate = [], {}
    for fold, run, role in GATE_RUNS:
        g = json.load(open(C.R("runs", run, "run_config.json"), encoding="utf-8"))["step0_gate"]
        assert g["passed"]
        diff_pos = abs(g["step0_tpr"] - g["expected"]) / one if fold != 4 else \
            abs(g["step0_tpr"] - g["expected"]) * 97
        tol_pos = round(g["tolerance"] / one) if fold != 4 else round(g["tolerance"] * 97)
        au = g.get("auroc_abs_diff")
        rule = (r"TPR exact" if g["tolerance"] == 0 else
                rf"TPR $\le$ 1 positive; AUROC $\le 10^{{-3}}$")
        gate[fold] = dict(diff=round(diff_pos), tol=tol_pos, au=au, rule=rule)
        rows.append(f"{fold} ({role}) & {rule} & {round(diff_pos)} & "
                    f"{sci(au) if au is not None else '---'} \\\\")
    assert gate[2]["diff"] == 1 and gate[2]["tol"] == 1, "fold 2's gate record changed"

    table = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{The per-fold step-0 gate, as each run recorded it} "
        r"(FreeMatch $\times$ nb3). The gate re-scores the loaded head before any "
        r"training and aborts the fold on a mismatch. Folds 0 and 1 ran under the "
        r"original rule, exact reproduction of TPR@$10^{-3}$; fold 2 differed by exactly "
        r"one positive, which that rule refuses, and folds 2--4 ran under the recalibrated "
        r"rule. TPR differences are in positives; the AUROC half was added with the "
        r"recalibration.}",
        r"\label{tab:restartgate}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Fold & Rule & $|\Delta\mathrm{TPR}|$ (positives) & $|\Delta\mathrm{AUROC}|$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    C.write_table(os.path.join(C.TABLES, "A5_gate.tex"), table,
                  header_comment="A5 -- step0_gate from each FreeMatch x nb3 fold's "
                                 "run_config.json.")

    keytex = ", ".join(r"\texttt{" + k.replace("_", r"\_") + "}" for k in keys)
    macros = "\n".join([
        m_("restartKeys", keytex),
        m_("restartMuZeroSteps", f"{mu_zero:,}".replace(",", "{,}")),
        m_("restartScheduleSteps", f"{cfg_u['schedule_total_steps']:,}".replace(",", "{,}")),
        m_("restartControlSteps", f"{u[-1][0]:,}".replace(",", "{,}")),
        m_("restartStart", C.f4(s0)),
        m_("restartTrough", C.f4(trough)),
        m_("restartTroughStep", str(trough_step)),
        m_("restartEnd", C.f4(end)),
        m_("restartMaxDrift", C.f4(maxdrift)),
        m_("restartMaxDriftPos", str(round(maxdrift / one))),
        m_("restartRecovery", C.f4(end - trough)),
        m_("restartOnePos", C.f4(one)),
        m_("restartFixedMaxDrift", C.f4(f_maxdrift)),
        m_("restartFixedMaxDriftPos", str(round(f_maxdrift / one))),
        m_("restartFixedMean", C.f4(float(f_tail.mean()))),
        m_("restartFixedSD", C.f4(float(f_tail.std()))),
        m_("restartFixedN", str(f_tail.size)),
        m_("restartTailMean", C.f4(tail["tail_mean"])),
        m_("restartTailSD", C.f4(tail["tail_sd"])),
        m_("restartTailN", str(tail["n"])),
        m_("restartSatPasses", sat.group(1)),
        m_("restartPool", sat.group(2)),
        m_("restartSatBatches", sat.group(4)),
        m_("restartEmaIters", f"{int(tau.group(2)):,}".replace(",", "{,}")),
        m_("restartMomentum", str(cfg_f["sat_ema_momentum"])),
        m_("restartTauDirect", C.f4(float(tau.group(1)))),
        m_("restartTauAgree", sci(agree.group(1))),
        m_("restartPbarAgree", sci(pbar.group(1))),
        m_("restartSatTol", sci(tol, 0)),
        m_("restartMuOffset", f"{cfg_f['mu_step_offset']:,}".replace(",", "{,}")),
        m_("restartAdamSteps", str(cfg_f["adam_warmup_steps"])),
        m_("restartAdamDebiased", debiased.group(1) + r"\%"),
        m_("restartAdamProbeEnd", str(probe_end)),
    ])
    C.write_table(os.path.join(C.TABLES, "M_restart.tex"), macros,
                  header_comment="Macros for Section 8 and Appendix A5 (see "
                                 "paper/scripts/make_a5_restart.py for sources).", size="")
    print(f"  unfixed {s0:.4f} -> {trough:.4f}@{trough_step} -> {end:.4f}, max drift "
          f"{maxdrift:.4f} ({maxdrift/one:.0f} pos); fixed max drift {f_maxdrift:.4f}; "
          f"gate diffs {[gate[k]['diff'] for k in sorted(gate)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
