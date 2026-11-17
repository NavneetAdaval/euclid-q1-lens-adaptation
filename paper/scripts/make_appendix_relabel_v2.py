r"""
make_appendix_relabel_v2.py -- appendix tables relabelled into the paper's vocabulary
(rewrite v1, extended in rewrite v2).
=====================================================================================

    python paper/scripts/make_appendix_relabel_v2.py

Several generated appendix tables carry the build's internal labels ("CORAL $\times$ nb3",
"D1", raw JSON keys) or captions in the old voice. Their generators are left unchanged
(several are protected by verify_orig_tables.py, and A5's opens a checkpoint, so make_all.py
keeps its committed output). This script reads each generated file as make_all.py has just
left it, applies a declared list of substitutions, and writes a ``*_v2.tex`` copy with its
own labels. The source files are not modified.

Number safety. The number tokens (crosscheck.py's regex, comment lines excluded) inside every
``tabular`` are asserted identical before and after, so no table value can change. Outside the
tabulars (captions and notes), where rewrite v2 replaces whole sentences, the multiset of
tokens removed and added is asserted equal to the job's declared ``drops`` and ``adds`` --
so every number that leaves or enters a caption is named here and nowhere implied.

  tables/A2_hashes.tex       -> tables/A2_hashes_v2.tex        (Table 10, tab:hashes2)
  tables/A4_augprec.tex      -> tables/A4_augprec_v2.tex       (Table 13, tab:augprec2)
  tables/A5_gate.tex         -> tables/A5_gate_v2.tex          (tab:restartgate2)
  tables/A7_prereg_pool.tex  -> tables/A7_prereg_pool_v2.tex   (Table 16, tab:prereg_pool2)
  tables/T5_perfold.tex      -> tables/T5_perfold_v2.tex       (Tables 6--9, tab:perfold-*2)
  tables/T6_window.tex       -> tables/T6_window_v2.tex        (Table 12, tab:window2)
"""
from __future__ import annotations

import os
import re
from collections import Counter

import _common as C

NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\{,\}\d{3})*(?:\.\d+)?")

COMMON = [(r"FreeMatch $\times$ nb3 $-$ FreeMatch frozen", r"FreeMatch unfrozen $-$ frozen"),
          (r" $\times$ nb3", ""),
          ("the project record", "our run logs")]

# ------------------------------------------------------------------ rewrite v2 texts
A2_CAPTION = (
    r"\textbf{Protected artifacts and when each was recorded.} Each enrolled file is made read-only "
    r"once written and is checked by digest at every build. \emph{Recorded} is the file's own "
    r"timestamp; where it has none, \emph{Source} says whether its timing rests on a flag "
    r"inside the file, on a dated Kaggle dataset version, or on our run logs alone. "
    r"\emph{vs.\ runs} compares it with the start of "
    r"the first confirmatory run it governs. Two kinds of digest are checked: a whole-file "
    r"digest, and a self-certifying digest of the file body with its own hash field removed. "
    r"\emph{Enrolled} says whether the file is in the inventory that every analysis run "
    r"re-checks; every row's digest, enrolled or not, matches its file when the table is built. "
    r"Eight artifacts are hashed; seven are re-checked by every analysis run, and the "
    r"window-averaging declaration is hashed but not enrolled. The DANN pre-registration has "
    r"two rows, its archived pre-fold copy and the file as regenerated after the folds. A date "
    r"without a time is a day-only record. The last row is protected but governs no run.")

A2_NOTES = (
    r"\begingroup\footnotesize\noindent\textbf{Notes to Table~\ref{tab:hashes2}.} What each "
    r"file fixed: "
    "\n"
    r"\emph{BNM budget}: budget 1000, BNM's rule-derived budget. The file carries no timestamp "
    r"and no before-flag, so its "
    r"timing rests on our run logs alone."
    "\n"
    r"\emph{BNM pre-registration}: the weight $\gamma$. BNM's weight came from a "
    r"development-fold sweep over a grid set by gradient norms at step zero; the sweep read the "
    r"nuclear norm and the predicted-positive fraction, never TPR. The budget was left "
    r"undecided, because its rule reads a crossing only "
    r"the development run produces."
    "\n"
    r"\emph{CORAL budget}: budget 2000, declared on cost, declining the development curve's "
    r"maximum."
    "\n"
    r"\emph{DANN budget and sensitivity readout}: budget 4000, $2 \times$ the $\lambda$ "
    r"horizon, and the step-2000 sensitivity declaration, which we call the "
    r"headline-and-sensitivity rule: the headline is the mean at 4000, and the mean at step "
    r"2000 is a budget-sensitivity check reported beside the headline, not instead of it. Every "
    r"confirmatory fold records this file's self-certifying digest when it starts, so the rule "
    r"is verified as fixed before the folds. The declaration says ``5-fold mean'', but the same "
    r"file excludes fold 0 from every headline, and we report folds 1--4; with the development "
    r"fold, the five-fold means on the cleaned pool are \dDannFiveFoldFourK{} at 4000 and "
    r"\dDannFiveFoldTwoK{} at 2000."
    "\n"
    r"\emph{DANN pre-registration}: the arms, the abort conditions, the rule for declaring the "
    r"margin $M$, the budget decision, the headline-and-sensitivity rule, and the predictions, "
    r"among them the arc-suppression prediction and the spread prediction. We regenerated the "
    r"file after the folds; the first row is the copy that predates them, and every one of its "
    r"\dannPrefoldIdenticalKeys{} pre-fold keys is unchanged in the second. The copy is entry "
    r"\dannPrefoldEntry{} of the file's hash history, and we archive it in the repository; the "
    r"regeneration, on 2026-09-05, added \dannPrefoldAddedKeys{} keys reporting fold results. The "
    r"copy does not establish that the arc-suppression prediction preceded its own "
    r"development-fold test, whose outcome it already contains; entry \dannArcEntry{} of the "
    r"hash history was not recovered."
    "\n"
    r"\emph{$\lambda=0$ control pre-registration}: budget 4000, under the rule that the control "
    r"is read where its treatment is read; three outcome branches (low, near the adaptation "
    r"arms, in between), which decided whether BNM would run, with no run by default; and the "
    r"four paired contrasts against the control. The file's description of two other arms' "
    r"budgets is wrong: it attributes FreeMatch's 1500 to the saturation of its adaptive "
    r"threshold, which completes 90\% of its rise on the development fold only at step "
    r"\fmTauNinetyStep{}, and CORAL's 2000 to its configuration. "
    r"Appendix~\ref{sec:budget-detail} gives the bases recorded at the time (FreeMatch: the "
    r"development curve's highest point; CORAL: cost). The sentence governs nothing in the "
    r"control's run."
    "\n"
    r"\emph{Checkpoint-window averaging declaration}: $k=8$ primary, the grid $\{1,2,4,8\}$, "
    r"both score combinations, and a window that ends at each arm's read-off step; it governs "
    r"analyses, not training runs."
    "\n"
    r"\emph{BNM step-0 gradient norms, re-measured}: a second, later measurement. The $\gamma$ "
    r"grid is never re-derived from it, and its original measuring run is lost. Protected, but "
    r"not a pre-registration: it governs no run."
    "\n"
    r"A further digest in our run logs, \texttt{7356acda\ldots}, is the hash of a superseded "
    r"pre-registration text; the text is lost, and we keep its digest in "
    r"\texttt{public\_inputs/\allowbreak record\_extract.json}. The window-averaging row, marked "
    r"\emph{no} under \emph{Enrolled}, is hashed in our run logs, and its digest matches the file, "
    r"but it is absent from the inventory that every analysis run re-checks, so it has neither the "
    r"re-check nor the read-only protection of the other seven. We did not enrol it after the "
    r"fact, because that would date its protection to the writing of this paper."
    r"\par\endgroup")

T6_CAPTION = (
    r"\textbf{Checkpoint-window score averaging.} Fold-mean TPR@$10^{-3}$ when the per-example "
    r"scores of the $k$ consecutive checkpoints ending at each arm's read-off step are averaged. "
    r"We declared $k \in \{1, 2, 4, 8\}$, both score combinations and the window placement, and "
    r"hashed them (\texttt{a572b81e\ldots}), before computing any windowed number. This is a "
    r"secondary analysis and replaces no pre-registered read-off. $k=1$ reproduces each arm's "
    r"headline. $k_{\mathrm{ach}}$ is the achieved window where fewer than $k$ checkpoints exist "
    r"before the read-off; the window truncates rather than padding. "
    r"Checkpoints are \ckptFm{} steps apart for FreeMatch and \ckptOther{} for the other "
    r"arms, so a full $k=8$ window spans \winEightSpanFm{} training steps for FreeMatch and "
    r"\winEightSpanOther{} for the others. "
    r"At $k=8$ the two "
    r"combinations order CORAL and DANN differently. Rows are folds 1--4 except "
    r"Baseline~A$^{\dagger}$, a single model averaged over folds 0--4 and computed on the "
    r"original test pool.")

T6_FOOT_OLD_START = r"\footnotesize Baseline A's row is a folds-0--4 mean"
T6_FOOT_NEW = (r"\footnotesize $^{\dagger}$Baseline A's per-step scores are recomputed on CPU "
               r"from the frozen 640-d feature cache and the retained per-step head weights, "
               r"which is exact because the encoder is frozen.")

T5_SENS_OLD = ("are pre-declared sensitivity readouts and do\nnot replace the arm's headline, "
               "whichever is higher;")
T5_SENS_NEW = "are pre-declared readouts reported beside the headline, not instead of it;"
T5_OWN_OLD = ("which fired\n\\texttt{no\\_matched\\_arm\\_budget} and is its own-objective "
              "reading only.")
T5_OWN_NEW = "which matches no other arm's, so it is reported for BNM alone."

# job: labels [(old, new)], subs [(old, new)], special ops, adds, drops (number tokens allowed
# to enter / leave captions and notes; tabular tokens may not change at all)
JOBS = {
    "A2_hashes": dict(
        labels=[("tab:hashes", "tab:hashes2")],
        subs=[(r"whole\_\allowbreak{}file", "whole file"),
              (r"self\_\allowbreak{}certifying", "self-certifying"),
              ("& Flavour &", "& Digest &"),
              # final review (PROJECTv7 7.63): the column reports inventory membership
              (r"& Verified \\", r"& Enrolled \\"),
              # v6e: the Source column names the run logs
              ("& record only &", "& run logs only &")],
        caption=A2_CAPTION,
        notes=(r"\begingroup\footnotesize\noindent\textbf{Notes to Table~\ref{tab:hashes2}.}",
               r"\par\endgroup", A2_NOTES,
               ["budget 1000", "budget 2000, declared on cost", "budget 4000 = 2", r"\dannArcEntry",
                r"\dannPrefoldEntry", "7356acda", r"$k=8$ primary, the grid $\{1,2,4,8\}$"]),
        # the key name 3_spread_not_mean, "fold-0 test" -> "development-fold test"; 7.65 adds
        # the DANN budget note's "5-fold mean" sentence (5, fold 0, folds 1--4, 4000, 2000)
        # and the lambda=0 note's budget sentence (1500, 90%, 2000)
        adds=["5", "1", "-4", "4000", "2000", "2000", "1500", "90"], drops=["3"]),
    "A4_augprec": dict(
        labels=[("tab:augprec", "tab:augprec2")],
        subs=[(r"\caption{\textbf{Round-1 augmentation and precision,",
               r"\caption{\textbf{Augmentation and precision in the frozen comparison,"),
              # rewrite v5b, item 12: four-digit numbers take no separator in the prose
              ("1{,}921", "1921")],
        adds=["1921"], drops=["1", "1{,}921"]),  # the "1" of "Round-1"; 1{,}921 -> 1921
    "A5_gate": dict(
        labels=[("tab:restartgate", "tab:restartgate2")],
        subs=[(r"(FreeMatch $\times$ nb3)", r"(unfrozen FreeMatch)")],
        adds=[], drops=[]),
    "A7_prereg_pool": dict(
        labels=[("tab:prereg_pool", "tab:prereg_pool2")],
        subs=[
            ("on the pool of the\nmain text (D1, which excludes the 265 objects of\n"
             r"Section~\ref{sec:data}), and on D2 (D1 also without the 20 test lenses noted in "
             r"Section~\ref{sec:limitations}).",
             "on the cleaned pool of the\nmain text, which excludes the 265 objects of\n"
             r"Section~\ref{sec:testpool}, and on that pool also without the 20 test lenses of "
             r"Section~\ref{sec:testpool} (``reduced'')."),
            ("pre-registered test pool (test negatives exactly as drawn)",
             "original test pool (test negatives exactly as drawn)"),
            ("D1 (main text)", "cleaned pool"),
            ("& D2 \\\\", "& reduced \\\\"),
            ("& Pre-registered &", "& Original &"),
            (r"Intervals as in Table~\ref{tab:contrasts}", r"Intervals as in Table~\ref{tab:contrasts2}"),
        ],
        adds=[], drops=[]),
    "T5_perfold": dict(
        labels=[("tab:perfold-tpr", "tab:perfold-tpr2"), ("tab:perfold-auroc", "tab:perfold-auroc2"),
                ("tab:perfold-comp", "tab:perfold-comp2"),
                ("tab:perfold-purity", "tab:perfold-purity2")],
        subs=[(T5_SENS_OLD, T5_SENS_NEW), (T5_OWN_OLD, T5_OWN_NEW),
              # rewrite v3: Table 1's frozen comparison averages folds 0-4, so fold 0 is
              # excluded from every confirmatory mean, not from every mean
              ("excluded from\nevery mean in this paper", "excluded from\nevery confirmatory mean in "
                                                          "this paper"),
              # 7.65 (G2): the control's fold-4 completeness rests on a single lens, which
              # make_prereg_fixes.py asserts from the stored scores
              ("and an arm with any undefined fold gets no fold-mean.}\n\\label{tab:perfold-comp}",
               "and an arm with any undefined fold gets no fold-mean. $\\lambda=0$'s fold-4 value "
               "at step 4000 is \\lamFourCompLenses{} lens.}\n\\label{tab:perfold-comp}")],
        counts={T5_SENS_OLD: 4, T5_OWN_OLD: 4, "excluded from\nevery mean in this paper": 4},
        adds=["0", "4", "4000"], drops=[]),
    "T6_window": dict(
        labels=[("tab:window", "tab:window2")],
        subs=[(r"\texttt{mean\_logit} (primary)", "mean logit (primary)"),
              (r"\texttt{mean\_rank} (secondary)", "mean rank (secondary)")],
        caption=T6_CAPTION,
        foot=(T6_FOOT_OLD_START, r"\end{table}", T6_FOOT_NEW),
        # caption gains the declared grid {1, 2, 4, 8}; the footnote's "fold-0 inclusion"
        # phrase goes (the folds 0--4 and 1--4 statements move into the caption unchanged);
        # 7.66 adds the checkpoint spacing (its values are macros; "k=8" adds one "8")
        adds=["1", "2", "4", "8", "8"], drops=["0"]),
}


def tokens(text):
    body = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("%"))
    return NUM.findall(body)


def split_tabular(text):
    """(tokens inside every tabular, tokens outside)."""
    inside, outside, depth = [], [], 0
    for line in text.split("\n"):
        if line.lstrip().startswith("%"):
            continue
        if r"\begin{tabular}" in line:
            depth += 1
        (inside if depth else outside).extend(NUM.findall(line))
        if r"\end{tabular}" in line:
            depth -= 1
    return inside, outside


def replace_caption(text, new):
    a = text.index(r"\caption{")
    b = text.index(r"\label{", a)
    return text[:a] + r"\caption{" + new + "}\n" + text[b:]


def replace_block(text, start, end, new, must_contain=()):
    a = text.index(start)
    b = text.index(end, a) + len(end)
    old = text[a:b]
    for s in must_contain:
        assert s in old, f"block no longer contains {s!r}"
    return text[:a] + new + text[b:]


def main() -> int:
    for name, job in JOBS.items():
        src = open(os.path.join(C.TABLES, f"{name}.tex"), encoding="utf-8").read()
        body = "\n".join(l for l in src.split("\n") if not l.startswith("%"))
        out = body
        for a, b in job["subs"] + COMMON:
            n = out.count(a)
            assert n or (a, b) in COMMON, f"{name}: substitution target missing: {a[:50]}"
            if a in job.get("counts", {}):
                assert n == job["counts"][a], f"{name}: {a[:40]} found {n} times"
            out = out.replace(a, b)
        if "caption" in job:
            out = replace_caption(out, job["caption"])
        if "notes" in job:
            s, e, new, must = job["notes"]
            s_old = s.replace("tab:" + job["labels"][0][1].split(":")[1],
                              job["labels"][0][0])  # the source carries the old label
            out = replace_block(out, s_old, e, new, must)
        if "foot" in job:
            s, e, new = job["foot"]
            a = out.index(s)
            b = out.index(e, a)
            out = out[:a] + new + "\n" + out[b:]
        for old_lab, new_lab in job["labels"]:
            out = out.replace(r"\label{" + old_lab + "}", r"\label{" + new_lab + "}")
            out = out.replace(r"\ref{" + old_lab + "}", r"\ref{" + new_lab + "}")
        ti, to = split_tabular(body)
        ni, no = split_tabular(out)
        assert ni == ti, f"{name}: number tokens inside a tabular changed"
        added, dropped = Counter(no) - Counter(to), Counter(to) - Counter(no)
        assert added == Counter(job["adds"]) and dropped == Counter(job["drops"]), \
            f"{name}: caption/notes tokens added {dict(added)} dropped {dict(dropped)}"
        assert " nb3" not in out and "project" not in out, name
        dst = os.path.join(C.TABLES, f"{name}_v2.tex")
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("% GENERATED FILE -- do not edit by hand.\n"
                     "% Rebuild:  python make_all.py (generator: make_appendix_relabel_v2.py)\n"
                     f"% Relabelled copy of tables/{name}.tex; tabular number tokens asserted "
                     "identical, caption/notes token changes declared.\n"
                     + out.rstrip() + "\n")
        print(f"  wrote {os.path.relpath(dst, C.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
