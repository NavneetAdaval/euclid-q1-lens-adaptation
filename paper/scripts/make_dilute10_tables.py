r"""
make_dilute10_tables.py -- the tables of the 1:10 diluted-target experiment (paper v6).
======================================================================================

    python paper/scripts/make_dilute10_tables.py

Main text:
  * ``T_dilute10_main.tex``  -- per arm: diluted TPR@1e-3 mean (sd), read-off step, G with its
                                interval and signs, D with its interval, and the branch;
  * ``T3v6_contrasts.tex``   -- Table 2 (``T3v2_contrasts.tex``, read and left unchanged) with a
                                new block, "A target signal against none, 1:10 target": the four
                                primary G rows, marked with a section sign.
Appendix K:
  * ``AK_perfold.tex``       -- per-fold diluted values, four metrics x four arms;
  * ``AK_contrasts4.tex``    -- the ten paired t-intervals at four decimals;
  * ``AK_frozen.tex``        -- the frozen stages against round 1 (two bootstrap intervals) and
                                the FreeMatch frozen/unfrozen decomposition, diluted vs balanced;
  * ``AK_head_norms.tex``    -- the frozen-stage head norms (runs/dilute10_head_norms/);
  * ``AK_predictions.tex``   -- every scored component of part 1's four predictions;
  * ``AK_sensitivity.tex``   -- the ten contrasts on the original and the reduced pools;
  * ``AK_bookkeeping.tex``   -- GradScaler skips and applied fractions, lens fraction against
                                expected, advisory firings;
  * ``AK_memorisation.tex``  -- the DANN - lambda=0 memorisation diagnostic (PROJECTv7 7.54).
Appendix B:
  * ``A2_hashes_v6.tex``     -- ``A2_hashes_v2.tex`` (read, left unchanged) with the dilution
                                experiment's declarations added, digests re-derived from disk.

Every value comes from an artifact through ``_dilute10``. A bound that rounds to zero at two
decimals is printed at three, so that its side of zero stays visible.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import _common as C
import _dilute10 as X

sys.path.insert(0, C.REPO_SCRIPTS)
import integrity  # noqa: E402  (scripts/integrity.py: the digest functions the build re-checks)

NA = C.NA
SEC = r"$^{\S}$"
METRIC = [("tpr", r"TPR@$10^{-3}$"), ("auroc", "AUROC"),
          ("comp_at_purity", r"completeness at 24\% purity"),
          ("purity_corr", r"purity at 50\% completeness")]


def m2(x, signed=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return NA
    if round(x, 2) == 0.0 and x != 0.0:
        return f"${x:+.3f}$" if signed else f"${x:.3f}$"
    return f"${X.s2(x)}$" if signed else f"${x:.2f}$"


def ci2(lo, hi):
    return f"$[{m2(lo)[1:-1]}, {m2(hi)[1:-1]}]$"


def m4(x, signed=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return NA
    return f"${X.s4(x)}$" if signed else f"${x:.4f}$"


def ci4(lo, hi):
    return f"$[{X.s4(lo)}, {X.s4(hi)}]$"


def write(name, body, comment, **kw):
    C.write_table(os.path.join(C.TABLES, name), body, header_comment=comment, **kw)


# ------------------------------------------------------------------ main-text table
def table_main():
    body = [r"\begin{table}[tbp]", r"\centering",
            r"\caption{\textbf{The 1:10 target, with batches drawn without the labels.} Per arm, folds 1--4, "
            r"cleaned pool: the diluted run's TPR at FPR${}=10^{-3}$ (mean and sample sd) at the "
            r"arm's read-off step; $G$, diluted $-$ $\lambda=0$, with the control read at 4000, "
            r"the primary contrast; $D$, balanced $-$ diluted at the same read-off, secondary and "
            r"without a verdict; and the pre-registered branch. Paired $t$-intervals (df${}=3$); "
            r"Signs counts the folds agreeing with $G$'s sign. A bound that rounds to zero at two "
            r"decimals is shown at three. The $\lambda=0$ control ($\dlLamTpr{}$) is shared with "
            r"Table~\ref{tab:main2}. Matched-step forms, four-decimal values and per-fold values: "
            r"Appendix~\ref{app:dilute10}.}",
            r"\label{tab:dilute}",
            r"\begin{tabular}{lrrrrcrl}", r"\toprule",
            r"Arm & Read at & TPR@$10^{-3}$ & sd & $G$, 95\% CI & Signs & $D$, 95\% CI & Branch \\",
            r"\midrule"]
    for arm, _n, label, _m in X.ARMS:
        t = X.per_arm(arm)
        g, d = X.G(arm), X.D(arm)
        body.append(f"{label} & {X.read_at(arm)} & {m2(t['mean'], False)} & {m2(t['sd'], False)} & "
                    f"{m2(g['mean'])} {ci2(g['ci_lo'], g['ci_hi'])} & {g['n_same_sign']}/4 & "
                    f"{m2(d['mean'])} {ci2(d['ci_lo'], d['ci_hi'])} & "
                    f"{X.verdict(arm)['branch'].lower()} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("T_dilute10_main.tex", "\n".join(body),
          "Main-text table of the 1:10 experiment. runs/dilute10_analysis/results.json "
          "(per_arm, G primary, D, verdicts).")


# ------------------------------------------------------------------ Table 2 with the new block
def table_contrasts():
    src = open(os.path.join(C.TABLES, "T3v2_contrasts.tex"), encoding="utf-8").read()
    body = src.split("\n", 3)[3] if src.startswith("% GENERATED") else src
    body = "\n".join(l for l in body.splitlines() if not l.startswith("%"))
    anchor = "\\midrule\n\\multicolumn{5}{l}{\\textit{Objective against objective}} \\\\"
    assert body.count(anchor) == 1, "Table 2's layout changed"
    cap_old = "Four-decimal values: Table~\\ref{tab:contrasts4}.}"
    assert body.count(cap_old) == 1
    rows = [r"\midrule",
            r"\multicolumn{5}{l}{\textit{A target signal against none, 1:10 target}} \\"]
    for arm, _n, label, _m in X.ARMS:
        g = X.G(arm)
        rows.append(f"{label}{SEC} $-$ $\\lambda=0$ & {g['diluted_at']} / {g['lambda0_at']} & "
                    f"{m2(g['mean'])} & {ci2(g['ci_lo'], g['ci_hi'])} & {g['n_same_sign']}/4 \\\\")
    body = body.replace(anchor, "\n".join(rows) + "\n" + anchor)
    body = body.replace(cap_old,
                        r"$^{\S}$From the second experiment, with a 1:10 target whose batches are drawn "
                        r"without the labels (Section~\ref{sec:results-dilute}): the arm's diluted run against "
                        r"the same $\lambda=0$ control, pre-registered as that experiment's "
                        r"primary contrast; a bound that rounds to zero at two decimals is shown at "
                        r"three, and the matched-step forms and the balanced $-$ diluted contrasts "
                        r"are in Table~\ref{tab:dl-contrasts4}. Four-decimal values: "
                        r"Tables~\ref{tab:contrasts4} and~\ref{tab:dl-contrasts4}.}")
    write("T3v6_contrasts.tex", body,
          "Table 2 (paper v6): tables/T3v2_contrasts.tex unchanged, plus the 1:10 block from "
          "runs/dilute10_analysis/results.json (G primary).", size="")


# ------------------------------------------------------------------ Appendix K
def table_perfold():
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{\textbf{Per-fold values of the diluted runs}, cleaned pool, at each arm's "
            r"read-off step, with the fold-mean and sample sd. Completeness at 24\% purity and "
            r"purity at 50\% completeness use the prevalence correction of "
            r"Equation~\eqref{eq:purity}.}",
            r"\label{tab:dl-perfold}",
            r"\begin{tabular}{llrrrrrrr}", r"\toprule",
            r"Metric & Arm & Step & fold 1 & fold 2 & fold 3 & fold 4 & mean & sd \\"]
    for key, mlabel in METRIC:
        body.append(r"\midrule")
        for i, (arm, _n, label, _m) in enumerate(X.ARMS):
            r = X.per_arm(arm, key)
            cells = [m4(r["per_fold"][str(k)], False) for k in X.FOLDS]
            body.append(f"{mlabel if i == 0 else ''} & {label} & {X.read_at(arm)} & "
                        + " & ".join(cells) + f" & {m4(r['mean'], False)} & {m4(r['sd'], False)} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("AK_perfold.tex", "\n".join(body),
          "Appendix K: per-fold diluted values. runs/dilute10_analysis/results.json per_arm.",
          placement="htbp")


def table_contrasts4():
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{\textbf{The ten paired $t$-intervals of the 1:10 experiment} at four "
            r"decimals, folds 1--4, cleaned pool (df${}=3$). $G$ is diluted $-$ $\lambda=0$; its "
            r"primary form reads the control at 4000 and decides the branch, and its "
            r"matched-step form, a pre-declared sensitivity readout, reads the control at the "
            r"arm's own step. $D$ is balanced $-$ diluted at the same read-off; it carries no "
            r"verdict and includes the noise of a changed batch order. No multiplicity "
            r"correction. $^{*}$Interval excludes zero.}",
            r"\label{tab:dl-contrasts4}",
            r"\begin{tabular}{llrlcl}", r"\toprule",
            r"Contrast & Steps & $\Delta$ TPR@$10^{-3}$ & 95\% CI & Signs & Role \\"]

    def row(name, r, steps, role):
        star = "$^{*}$" if r["excludes_zero"] else ""
        return (f"{name} & {steps} & {m4(r['mean'])}{star} & {ci4(r['ci_lo'], r['ci_hi'])} & "
                f"{r['n_same_sign']}/4 & {role} \\\\")
    body += [r"\midrule", r"\multicolumn{6}{l}{\textit{$G$: diluted $-$ $\lambda=0$}} \\"]
    for arm, _n, label, _m in X.ARMS:
        g = X.G(arm)
        body.append(row(f"{label} $-$ $\\lambda=0$", g, f"{g['diluted_at']} / {g['lambda0_at']}",
                        "primary"))
        if arm in X.MATCHED_G:
            gm = X.G(arm, "sensitivity")
            body.append(row(f"{label} $-$ $\\lambda=0$", gm,
                            f"{gm['diluted_at']} / {gm['lambda0_at']}", "sensitivity"))
    body += [r"\midrule", r"\multicolumn{6}{l}{\textit{$D$: balanced $-$ diluted}} \\"]
    for arm, _n, label, _m in X.ARMS:
        d = X.D(arm)
        ro = X.read_at(arm)
        body.append(row(f"{label}", d, f"{ro} / {ro}", "secondary"))
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("AK_contrasts4.tex", "\n".join(body),
          "Appendix K: the 10 t-intervals. runs/dilute10_analysis/results.json G and D.",
          placement="htbp")


def _agg(key):
    a = X.results()["descriptive"]["frozen_vs_round1"][key]
    pf = a["per_fold"]
    agg = a["aggregate"]
    return (json.loads(pf) if isinstance(pf, str) else pf,
            json.loads(agg) if isinstance(agg, str) else agg)


def table_frozen():
    R = X.results()["descriptive"]
    fz = R["frozen_stages"]
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{\textbf{The frozen stages on the 1:10 target against the balanced target}, "
            r"and where FreeMatch's contrast is gained. Top: TPR at FPR${}=10^{-3}$ of each frozen "
            r"stage's step-8000 head per fold (1:1 is the main design), cleaned pool; the "
            r"difference is diluted $-$ balanced under "
            r"the class-conditional paired bootstrap ($B=1000$), as in the frozen comparison. "
            r"Bottom: FreeMatch's contrast against the control ($\lambda=0$ at 4000) split into "
            r"the part reached by its frozen head (start $-$ $\lambda=0$) and the part gained "
            r"after unfreezing (unfrozen $-$ start). AUROC: the mean over the four folds of the "
            r"same heads' AUROC on the same pool.}",
            r"\label{tab:dl-frozen}",
            r"\begin{tabular}{llrrrrrrrl}", r"\toprule",
            r"Stage & Target & fold 1 & fold 2 & fold 3 & fold 4 & mean & AUROC & 1:10 $-$ 1:1 "
            r"& 95\% CI \\",
            r"\midrule"]
    # r3: the balanced heads' AUROC, from the main design's stored score dumps (the same
    # summary analyze_dilute10.frozen_stages reads their TPR from), on the cleaned pool
    with C.pool("D1"):
        r1s = {"FreeMatch": C.round1_summary("bakeoff_freematch_5fold", folds=X.FOLDS),
               "CORAL": C.round1_summary("bakeoff_coral", folds=X.FOLDS)}
    for key, run in (("FreeMatch", "freematch_frozen"), ("CORAL", "coral_frozen")):
        r1 = fz["round1_tpr"][key]
        dl = fz[run]["tpr"]
        _pf, agg = _agg(key)
        t = agg["tpr"]
        assert abs(t["B"] - dl["mean"]) < 1e-12 and abs(t["A"] - fz["round1_start_mean"][key]) < 1e-12
        pf = r1s[key]["per_fold"]
        assert all(abs(float(pf[k]["tpr"]) - r1[str(k)]) < 1e-12 for k in X.FOLDS), key
        auc_r1 = sum(float(pf[k]["auroc"]) for k in X.FOLDS) / len(X.FOLDS)
        body.append(f"{key} & 1:1 (main design) & "
                    + " & ".join(m4(r1[str(k)], False) for k in X.FOLDS)
                    + f" & {m4(t['A'], False)} & {m4(auc_r1, False)} & & \\\\")
        body.append(f" & 1:10 & " + " & ".join(m4(dl["per_fold"][str(k)], False) for k in X.FOLDS)
                    + f" & {m4(t['B'], False)} & {m4(fz[run]['auroc']['mean'], False)} "
                    + f"& {m4(t['diff'])} & {ci4(*t['ci'])} \\\\")
    dec = R["freematch_decomposition"]
    body += [r"\midrule",
             r" & Target & start & frozen & unfrozen & total & & & & \\", r"\midrule"]
    for tgt, k, tot in (("1:1 (main design)", "balanced", "total"), ("1:10", "diluted", "G_primary")):
        v = dec[k]
        body.append(f"FreeMatch & {tgt} & {m4(v['start'], False)} & {m4(v['frozen_part'])} & "
                    f"{m4(v['unfrozen_part'])} & {m4(v[tot])} & & & & \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("AK_frozen.tex", "\n".join(body),
          "Appendix K: frozen stages vs round 1 and the FreeMatch decomposition. "
          "runs/dilute10_analysis/results.json descriptive.", placement="htbp", colsep="1.5pt")


def table_heads():
    H = X.heads()["heads"]
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{\textbf{The frozen-stage heads} that initialise the unfrozen CORAL and "
            r"FreeMatch arms, read from their stored step-8000 state dicts: the $L_2$ norm of the "
            r"layer-normalisation gain and of its bias, and $\|\mathrm{head}\|_2$ over every head "
            r"parameter. A gain of one in every dimension has norm $\sqrt{640}=\dlUnitGain{}$. "
            r"Each $\|\mathrm{head}\|_2$ equals the value its run recorded when it was used.}",
            r"\label{tab:dl-heads}",
            r"\begin{tabular}{lllrrrr}", r"\toprule",
            r"Head & Target & Norm & fold 1 & fold 2 & fold 3 & fold 4 \\"]
    for arm, lab in (("coral", "CORAL"), ("freematch", "FreeMatch")):
        body.append(r"\midrule")
        for stage, tl in (("round1", "main design (1:1)"), ("diluted", "1:10")):
            for q, ql in (("ln_gain_norm", "gain"), ("ln_bias_norm", "bias"),
                          ("head_l2", r"$\|\mathrm{head}\|_2$")):
                first = (stage == "round1" and q == "ln_gain_norm")
                cells = [f"${H[arm][stage][str(k)][q]:.1f}$" for k in X.FOLDS]
                body.append(f"{lab if first else ''} & {tl if q == 'ln_gain_norm' else ''} & {ql} & "
                            + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("AK_head_norms.tex", "\n".join(body),
          "Appendix K: frozen-stage head norms. runs/dilute10_head_norms/head_norms.json.",
          placement="htbp")


def _tex(s: str) -> str:
    """A prediction component, as results.json prints it, set in LaTeX."""
    s = s.replace("lambda=0", r"$\lambda=0$").replace("%", r"\%")
    s = re.sub(r"(?<![\w$])-(?=\d)", "$-$", s)
    s = s.replace(" < ", " $<$ ").replace(" > ", " $>$ ").replace(" - ", " $-$ ")
    s = s.replace("'", "'")
    # v6e: the paper says "main design" where results.json says "round 1"
    s = s.replace("round-1", "main-design").replace("round 1", "main design")
    return s


def table_predictions():
    P = X.results()["predictions"]
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{\textbf{Our four predictions for the 1:10 experiment, scored as written.} "
            r"Each prediction, recorded before any run, is split into its components; where its "
            r"wording admits two literal readings, both are scored. \emph{Rests on} gives the "
            r"stored value each score reads; relative loss is $D$ / (balanced $-$ $\lambda=0$). "
            r"No prediction was amended. Component 3d(ii) takes a maximum over \dlBnmPpN{} "
            r"single-batch values and would be met by batch noise alone; we do not count it as "
            r"support (Section~\ref{sec:results-dilute}).}",
            r"\label{tab:dl-predictions}",
            r"\begin{tabular}{l>{\TRAG}p{5.4cm}l>{\TRAG}p{5.0cm}}", r"\toprule",
            r"ID & Component & Score & Rests on \\", r"\midrule"]
    last = None
    for c in P["components"]:
        if last is not None and c["id"][0] != last:
            body.append(r"\midrule")
        last = c["id"][0]
        body.append(f"{c['id']} & {_tex(c['component'])} & {c['score']} & {_tex(c['rests_on'])} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("AK_predictions.tex", "\n".join(body),
          "Appendix K: prediction scoring. runs/dilute10_analysis/results.json predictions.",
          placement="htbp", size="scriptsize")


def table_sensitivity():
    S = X.results()["sensitivities"]
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{\textbf{The ten contrasts of the 1:10 experiment on the original and the "
            r"reduced test pools}, beside whether the cleaned pool's interval excludes zero. The "
            r"original pool keeps every drawn test negative (Section~\ref{sec:testpool}); the "
            r"reduced pool also removes the test lenses that appear in training or validation. "
            r"Match: the same zero-inclusion as on the cleaned pool. Paired $t$-intervals "
            r"(df${}=3$).}",
            r"\label{tab:dl-sensitivity}",
            r"\begin{tabular}{llcrcrc}", r"\toprule",
            r"Contrast & Steps & Cleaned & Original pool & Match & Reduced pool & Match \\",
            r"\midrule"]
    keys = []
    for arm, n, label, _m in X.ARMS:
        g = X.G(arm)
        keys.append((f"$G$ {label}", f"{n}|primary", "G", f"G {label} primary",
                     f"{g['diluted_at']} / {g['lambda0_at']}", X.G(arm)["excludes_zero"]))
        if arm in X.MATCHED_G:
            gm = X.G(arm, "sensitivity")
            keys.append((f"$G$ {label}", f"{n}|sensitivity", "G", f"G {label} sensitivity",
                         f"{gm['diluted_at']} / {gm['lambda0_at']}", gm["excludes_zero"]))
    for arm, n, label, _m in X.ARMS:
        ro = X.read_at(arm)
        keys.append((f"$D$ {label}", n, "D", f"D {label}", f"{ro} / {ro}", X.D(arm)["excludes_zero"]))
    yn = {True: "yes", False: "\\textbf{no}"}
    for lab, key, blk, zk, steps, cl in keys:
        o, r = S["orig"][blk][key], S["D2"][blk][key]
        body.append(f"{lab} & {steps} & {'excl.' if cl else 'incl.'} & "
                    f"{m4(o['mean'])} {ci4(o['ci_lo'], o['ci_hi'])} & "
                    f"{yn[S['orig']['zero_inclusion_matches_cleaned'][zk]]} & "
                    f"{m4(r['mean'])} {ci4(r['ci_lo'], r['ci_hi'])} & "
                    f"{yn[S['D2']['zero_inclusion_matches_cleaned'][zk]]} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("AK_sensitivity.tex", "\n".join(body),
          "Appendix K: pool sensitivities. runs/dilute10_analysis/results.json sensitivities "
          "(orig, D2).", placement="htbp", colsep="2.5pt")


def table_bookkeeping():
    B = X.results()["descriptive"]["bookkeeping"]
    sk, lf = B["skips"], B["lens_fraction"]
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{\textbf{Run bookkeeping of the 1:10 experiment.} GradScaler skips and the "
            r"fraction of the budget's optimiser updates applied (half-precision runs; the CORAL "
            r"frozen stage ran in full precision on CPU, and the BNM runs and the FreeMatch frozen runs did "
            r"not record skips); the lens fraction of the target batches, its run mean $\pm$ sd "
            r"over the run's logged batches, against the pool's expected value; and CORAL's "
            r"advisory feature-norm guard: the norm it reads (the mean $L_2$ norm of a batch's "
            r"feature vectors, source and target) at step 1 and its smallest value over the run, "
            r"source or target, against the guard's threshold, and its firings.}",
            r"\label{tab:dl-bookkeeping}",
            r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Run & Quantity & fold 1 & fold 2 & fold 3 & fold 4 \\", r"\midrule"]
    LAB = {"dann": "DANN", "bnm": "BNM", "coral_nb3": "CORAL", "freematch_nb3": "FreeMatch",
           "coral_frozen": "CORAL frozen", "freematch_frozen": "FreeMatch frozen"}
    for run in ("dann", "coral_nb3", "freematch_nb3"):
        body.append(f"{LAB[run]} & skips / applied & "
                    + " & ".join(f"{sk[f'{run}/fold{k}']['skips']} / "
                                 f"${sk[f'{run}/fold{k}']['applied_fraction']:.4f}$"
                                 for k in X.FOLDS) + r" \\")
    body.append(r"\midrule")
    for run in ("dann", "bnm", "coral_nb3", "freematch_nb3", "coral_frozen", "freematch_frozen"):
        # v6e: the sd per run beside its mean (part 1's "mean and sd per run")
        body.append(f"{LAB[run]} & lens fraction, mean $\\pm$ sd & "
                    + " & ".join(f"${lf[f'{run}/fold{k}']['mean']:.4f} \\pm "
                                 f"{lf[f'{run}/fold{k}']['sd']:.4f}$" for k in X.FOLDS)
                    + r" \\")
        body.append(r" & expected & "
                    + " & ".join(f"${lf[f'{run}/fold{k}']['expected']:.4f}$" for k in X.FOLDS)
                    + r" \\")
    body.append(r"\midrule")
    # r3: the guard's input and threshold, per fold, from steps.csv and the advisory log
    s1, lo, ceil = [], [], []
    for k in X.FOLDS:
        rows = X.steps_csv(k)
        assert int(rows[0]["step"]) == 1
        s1.append((float(rows[0]["src_feat_l2"]), float(rows[0]["tgt_feat_l2"])))
        lo.append(min(min(float(r["src_feat_l2"]), float(r["tgt_feat_l2"])) for r in rows))
        c = {float(r["ceiling"]) for r in X.advisory_rows(k)}
        assert len(c) == 1
        ceil.append(c.pop())
    body.append(r"CORAL & norm at step 1, source & " + " & ".join(f"${s:.1f}$" for s, _t in s1)
                + r" \\")
    body.append(r" & norm at step 1, target & " + " & ".join(f"${t:.1f}$" for _s, t in s1)
                + r" \\")
    body.append(r" & smallest norm & " + " & ".join(f"${v:.1f}$" for v in lo) + r" \\")
    body.append(r" & guard threshold & " + " & ".join(f"${v:.0f}$" for v in ceil) + r" \\")
    body.append(r" & advisory firings & "
                + " & ".join(f"{X.advisory(k)['n_firings']}" for k in X.FOLDS) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("AK_bookkeeping.tex", "\n".join(body),
          "Appendix K: run bookkeeping. runs/dilute10_analysis/results.json descriptive.bookkeeping; "
          "runs/dilute10/coral_nb3/fold*/dilute10_feature_norm_advisory.json.", placement="htbp",
          colsep="2.5pt")


def table_memorisation():
    Gp = X.gap()
    rp = Gp["readings_per_fold"]
    c = Gp["contrast_dann_minus_lambda0"]
    mx = Gp["any_checkpoint_max_descriptive"]["values"]
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{\textbf{The memorisation diagnostic for DANN and its control}, a "
            r"retrospective analysis declared before it was computed. Each value is the gap in "
            r"the normalised nuclear norm of the predictions between the target slice the arm "
            r"trains on and the held-out slice, \mzHoldout{} images each, computed without "
            r"augmentation on the stored checkpoints: at the read-off step (primary) and averaged "
            r"over the \mzNckpt{} checkpoints (secondary). The maxima over checkpoints are "
            r"descriptive only, because a maximum of noisy reads is positive even when no gap "
            r"exists. These values are not comparable with BNM's logged gap, which used one "
            r"augmented view per image.}",
            r"\label{tab:dl-memorisation}",
            r"\begin{tabular}{llrrrrrl}", r"\toprule",
            r"Reading & Arm & fold 1 & fold 2 & fold 3 & fold 4 & mean & 95\% CI \\", r"\midrule"]
    for key, lab, ck in (("at_4000", "step 4000", "primary_at_4000"),
                         ("within_fold_mean", "mean over steps", "secondary_within_fold_mean")):
        for arm, al in (("dann", "DANN"), ("lambda0", r"$\lambda=0$")):
            vals = [rp[arm][str(k)][key] for k in X.FOLDS]
            body.append(f"{lab if arm == 'dann' else ''} & {al} & "
                        + " & ".join(m4(v) for v in vals) + f" & {m4(sum(vals) / 4)} & \\\\")
        r = c[ck]
        body.append(r" & DANN $-$ $\lambda=0$ & " + " & ".join(m4(v) for v in r["diffs"])
                    + f" & {m4(r['mean'])} & {ci4(r['ci_lo'], r['ci_hi'])} \\\\")
        body.append(r"\midrule")
    for arm, al in (("dann", "DANN"), ("lambda0", r"$\lambda=0$")):
        body.append(f"{'maximum' if arm == 'dann' else ''} & {al} & "
                    + " & ".join(m4(mx[arm][str(k)]["value"]) for k in X.FOLDS) + " & & \\\\")
        body.append(r" & \quad at step & "
                    + " & ".join(str(mx[arm][str(k)]["step"]) for k in X.FOLDS) + " & & \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("AK_memorisation.tex", "\n".join(body),
          "Appendix K: DANN - lambda=0 memorisation diagnostic. "
          "runs/dann_lambda0_holdout_gap/results.json.", placement="htbp")


# ------------------------------------------------------------------ Appendix B, Table of hashes
def _utc(s):
    return s.replace("T", " ")[:16] if s else "---"


def _first_run(arms):
    """Earliest start over the named runs, from the run records themselves."""
    times = []
    for arm in arms:
        for k in X.FOLDS:
            rr = X.run_record(arm, k)
            t = rr.get("started_utc") or (rr.get("dilute10") or {}).get("recorded_utc")
            assert t, (arm, k)
            times.append((t, arm, k))
    return min(times)


def table_hashes():
    src = open(os.path.join(C.TABLES, "A2_hashes_v2.tex"), encoding="utf-8").read()
    body = "\n".join(l for l in src.splitlines() if not l.startswith("%"))
    inv = {r["path"]: r for r in json.load(open(integrity.INVENTORY, encoding="utf-8"))["artifacts"]}

    def digest(rel):
        r = inv.get(rel)
        flav = r["flavour"] if r else ("self_certifying" if rel.endswith(".json") else "whole_file")
        p = C.R(*rel.split("/"))
        got = integrity.self_certifying_sha(p) if flav == "self_certifying" else integrity.whole_file_sha(p)
        verified = bool(r) and got == r["sha256"]
        return got, {"whole_file": "whole file", "self_certifying": "self-certifying"}[flav], verified

    all_runs = ("coral_frozen", "freematch_frozen", "dann", "bnm", "coral_nb3", "freematch_nb3")
    t_all = _first_run(all_runs)
    t_p3 = _first_run(("coral_nb3", "freematch_nb3"))
    p1, p2, p3 = X.load(X.PART1), X.load(X.PART2), X.load(X.PART3)
    plan = X.load(X.PLAN)
    rows = [
        ("1:10 pool definition", "data/splits/dilute10_pool.json", None,
         r"digest in part 1", t_all),
        ("1:10 pre-registration, part 1", "runs/dilute10_prereg_part1.json", p1["recorded_utc"],
         "file", t_all),
        ("1:10 pre-registration, part 2", "runs/dilute10_prereg_part2.json", p2["recorded_utc"],
         "file", t_all),
        ("1:10 pre-registration, part 3", "runs/dilute10_prereg_part3.json", p3["recorded_utc"],
         "file", t_p3),
    ]
    out = [r"\midrule",
           r"\multicolumn{8}{l}{\emph{The 1:10 diluted-target experiment (Section~\ref{sec:dilute-methods}):}} \\"]
    for label, rel, rec, srcw, first in rows:
        got, flav, ok = digest(rel)
        assert ok, rel
        rec_t = rec or p1["recorded_utc"]
        vs = "before" if rec_t < first[0] else "\\textbf{after}"
        out.append(f"{label} & {_utc(rec)} & {srcw} & {_utc(first[0])} & {vs} & {flav} & "
                   f"\\texttt{{{got[:16]}}} & yes \\\\")
    out.append(r"\multicolumn{8}{l}{\emph{Analysis declarations (govern analyses, not runs):}} \\")
    got, flav, ok = digest("runs/dilute10_analysis/plan.json")
    assert ok
    out.append(f"1:10 analysis plan & {_utc(plan['recorded_utc'])} & file & --- & n/a & {flav} & "
               f"\\texttt{{{got[:16]}}} & yes \\\\")
    got, flav, ok = digest("runs/dann_lambda0_holdout_gap/plan.json")
    assert ok
    out.append(f"DANN $-$ $\\lambda=0$ memorisation plan (retrospective) & --- & run logs only & --- & "
               f"n/a & {flav} & \\texttt{{{got[:16]}}} & yes \\\\")
    anchor = "\\midrule\n\\multicolumn{8}{l}{\\emph{Protected, not a pre-registration (governs no run):}} \\\\"
    assert body.count(anchor) == 1
    body = body.replace(anchor, "\n".join(out) + "\n" + anchor)
    old = ("Eight artifacts are hashed; seven are re-checked by every analysis run, and the "
           "window-averaging declaration is hashed but not enrolled.")
    assert body.count(old) == 1
    n_new = len(rows) + 2
    words = {14: "Fourteen", 13: "thirteen"}
    body = body.replace(old, f"{words[8 + n_new]} artifacts are hashed; {words[7 + n_new]} are re-checked by "
                             "every analysis run, and the window-averaging declaration is hashed "
                             "but not enrolled. The 1:10 rows are dated against that experiment's "
                             "first run.")
    old3 = "because that would date its protection to the writing of this paper."
    assert body.count(old3) == 1
    body = body.replace(old3, old3 + " The first 1:10 run was CORAL's frozen stage on fold 1, "
                              "run locally.")
    old2 = "so it has neither the re-check nor the read-only protection of the other seven."
    assert body.count(old2) == 1
    body = body.replace(old2, "so it has neither the re-check nor the read-only protection of "
                              f"the other {words[7 + n_new]}.")
    write("A2_hashes_v6.tex", body,
          "Appendix B (paper v6): tables/A2_hashes_v2.tex unchanged, plus the 1:10 experiment's "
          "declarations, digests re-derived from disk through scripts/integrity.py.",
          size="", placement="htbp")


def relabel_v6():
    """The v6 copies of Table 2 and of the hash table carry new labels (tab:contrasts6,
    tab:hashes6), so that no label is defined twice among the table files; the two generated
    tables that point at Table 2 get v6 copies whose only change is that reference. Each copy
    is asserted identical to its source apart from the swapped label or reference."""
    for name, old, new in (("T3v6_contrasts.tex", r"\label{tab:contrasts2}", r"\label{tab:contrasts6}"),
                           ("A2_hashes_v6.tex", r"\label{tab:hashes2}", r"\label{tab:hashes6}"),
                           ("A2_hashes_v6.tex", r"\ref{tab:hashes2}", r"\ref{tab:hashes6}")):
        p = os.path.join(C.TABLES, name)
        s = open(p, encoding="utf-8").read()
        assert s.count(old) == 1, (name, old)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(s.replace(old, new))
    for src, dst in (("A7_prereg_pool_v2.tex", "A7_prereg_pool_v6.tex"),
                     ("C_k8_pairs.tex", "C_k8_pairs_v6.tex")):
        s = open(os.path.join(C.TABLES, src), encoding="utf-8").read()
        body = "\n".join(l for l in s.splitlines() if not l.startswith("%"))
        n = body.count(r"\ref{tab:contrasts2}")
        assert n >= 1, src
        lab = {"A7_prereg_pool_v2.tex": ("tab:prereg_pool2", "tab:prereg_pool6"),
               "C_k8_pairs.tex": ("tab:k8pairs", "tab:k8pairs6")}[src]
        assert body.count(r"\label{" + lab[0] + "}") == 1, src
        new = body.replace(r"\ref{tab:contrasts2}", r"\ref{tab:contrasts6}") \
                  .replace(r"\label{" + lab[0] + "}", r"\label{" + lab[1] + "}")
        back = new.replace(r"\ref{tab:contrasts6}", r"\ref{tab:contrasts2}") \
                  .replace(r"\label{" + lab[1] + "}", r"\label{" + lab[0] + "}")
        assert back == body, src
        place = re.search(r"\\begin\{table\}\[([a-zA-Z!]*)\]", body).group(1)
        write(dst, new, f"Paper v6: tables/{src} unchanged except {n} reference(s) to Table 2 "
                        f"(tab:contrasts2 -> tab:contrasts6) and its label "
                        f"({lab[0]} -> {lab[1]}).", size="", placement=place)


def main() -> int:
    table_main()
    table_contrasts()
    table_perfold()
    table_contrasts4()
    table_frozen()
    table_heads()
    table_predictions()
    table_sensitivity()
    table_bookkeeping()
    table_memorisation()
    table_hashes()
    relabel_v6()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
