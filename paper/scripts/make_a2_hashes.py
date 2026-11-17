r"""
make_a2_hashes.py -- appendix table: the protected pre-registration artifacts.
==============================================================================

    python paper/scripts/make_a2_hashes.py

Every artifact whose hash the project record carries, **re-derived from the file on disk
at build time** by the project's own ``scripts/integrity.py`` -- not transcribed. A hash
typed into a paper by hand certifies nothing; a hash recomputed by the build certifies
that the file the paper describes is the file on disk.

TWO HASH FLAVOURS, HANDLED SEPARATELY, because the project uses both and checking the
wrong one reports a false failure and trains the reader to ignore the check:

  * **whole-file** -- sha256 of the bytes;
  * **self-certifying** -- sha256 of the JSON body with its own hash field removed. These
    files verify themselves: the writer re-reads the file it just wrote and raises if the
    digest does not match.

HASHED IS NOT THE SAME AS VERIFIED, AND THE TABLE NOW SAYS WHICH IS WHICH.
``protected_files()`` DISCOVERS every file under ``runs/`` whose digest the record
carries, and finds eight. ``verify()`` iterates the inventory at
``runs/protected_artifacts.json`` instead, and that inventory holds seven: the
checkpoint-window declaration is recorded and hashed but was never added to it, so no
analysis run re-checks it and nothing makes it read-only. The final column reports
inventory membership so the table cannot imply a guarantee the repository does not
provide. See PROJECT.md 7.34.

DATES COME FROM THE FILES, NEVER FROM TYPED TEXT (2026-09-26, DANN_PREREG_REPORT.md).
Each row's date is the file's OWN top-level ``recorded_utc`` / ``declared_utc``. Where a
file carries no timestamp but carries its own before-flag
(``recorded_before_any_confirmatory_fold``, ``declared_before_any_windowed_number_was_computed``)
the flag is reported instead. Where it carries neither, the row says "record only" and
no date is printed. Each dated row is compared with the EARLIEST ``started_utc`` of the
confirmatory runs it governs, read from those runs' own files, and the table prints
before or after. The one protected file that governs no run is listed in its own group.

Writes ``paper/tables/A2_hashes.tex`` and nothing else.
"""
from __future__ import annotations

import glob
import json
import os
import re

import _common as C
import audit_venue as AV

RECORD = os.path.join(C.REPO, "PROJECTv7.md")

# What each protected artifact fixed, and which confirmatory arm it governs. NO DATES
# HERE: dates are read from the files by file_date() below.
ROLE = {
    "runs/dann_r5_prereg.json":
        ("DANN $\\times$ nb3 pre-registration", "DANN x nb3",
         "arms, abort conditions, the rule for declaring the margin M, the budget "
         "decision, the headline/sensitivity rule, and the predictions, including "
         "\\texttt{prediction\\_arc\\_suppression} and \\texttt{3\\_spread\\_not\\_mean}. "
         "The date and the \\emph{before} are those of a pre-fold copy of this file "
         "(history entry \\dannPrefoldEntry{}, archived as "
         "\\texttt{evidence/prefold/dann\\_r5\\_prereg\\_kaggle\\_v2\\_20260902.json}); the "
         "file itself was regenerated after the folds, on 2026-09-05, which added "
         "\\dannPrefoldAddedKeys{} post-fold keys and left all \\dannPrefoldIdenticalKeys{} "
         "pre-fold keys identical. The copy does \\emph{not} establish the arc-suppression "
         "prediction's precedence over its own fold-0 test, whose outcome it already "
         "contains (history entry \\dannArcEntry{} was not recovered)"),
    "runs/dann_r5_budget.json":
        ("DANN $\\times$ nb3 budget + sensitivity readout", "DANN x nb3",
         "budget 4000 = 2 $\\times$ the $\\lambda$ horizon, and the step-2000 "
         "sensitivity pre-declaration: the headline is the mean at 4000, and the mean at "
         "step 2000 is a budget-sensitivity check that \\emph{does not replace it, "
         "whichever is higher} (key \\texttt{PRE\\_DECLARED\\_headline\\_and\\_sensitivity}). "
         "Every confirmatory fold records this file's self-certifying digest when it "
         "starts, so this rule is verified as fixed before the folds"),
    "runs/coral_r4_budget.json":
        ("CORAL $\\times$ nb3 budget", "CORAL x nb3",
         "budget 2000, declared on cost and explicitly not the argmax"),
    "runs/lambda0_r6_prereg.json":
        ("$\\lambda=0$ control pre-registration", "lambda=0 (nb3)",
         "budget 4000 under the rule \\emph{the control is read where its treatment is "
         "read}; both decision branches, and the four pre-registered paired contrasts"),
    "runs/bnm_r6_prereg.json":
        ("BNM $\\times$ nb3 pre-registration", "BNM x nb3",
         "$\\gamma$ from a metric-blind rule; budget left OPEN because the budget rule "
         "reads a crossing only the development run produces"),
    "runs/bnm_r6_budget.json":
        ("BNM $\\times$ nb3 budget", "BNM x nb3",
         "budget 1000, outcome \\texttt{no\\_matched\\_arm\\_budget}: the arm's "
         "own-objective reading only, never cross-arm comparable. The file carries no "
         "timestamp and no before-flag, so its timing rests on the project record alone"),
    "runs/round7/window_prereg.json":
        ("Checkpoint-window averaging declaration", None,
         "$k=8$ primary, the grid $\\{1,2,4,8\\}$, both score combinations, and the "
         "window placed by the pre-registered read-off so the rule reads no metric; it "
         "governs analyses, not training runs"),
    "runs/bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json":
        ("BNM step-0 gradient norms, re-measured", "NONE",
         "a second, later measurement; the $\\gamma$ grid is never re-derived from it "
         "and its original measuring run is lost. Protected, but not a "
         "pre-registration: it governs no run"),
}

DATE_KEYS = ("recorded_utc", "declared_utc")
FLAG_KEYS = ("recorded_before_any_confirmatory_fold",
             "declared_before_any_windowed_number_was_computed")
TS = re.compile(r"20\d\d-\d\d-\d\dT\d\d:\d\d")


def file_date(path):
    """(date, source) from the file's OWN top-level fields -- never typed."""
    j = json.load(open(C.R(path), encoding="utf-8"))
    for k in DATE_KEYS:
        v = j.get(k)
        if isinstance(v, str) and TS.match(v):
            # a value written as midnight UTC is a day-only record (PROJECTv7 7.65): print the
            # date alone rather than a time nobody recorded
            if v.endswith("T00:00:00Z"):
                return v[:10], "file", k
            return v[:16].replace("T", " "), "file", k
    for k in FLAG_KEYS:
        if j.get(k) is True:
            return None, "file (flag only)", k
    return None, "record only", None


def first_start(arm):
    """Earliest started_utc across the arm's confirmatory folds, from their own files."""
    folds, dir_of = next((f, d) for a, f, d, _ in AV.ARMS if a == arm)
    starts = []
    for f in folds:
        for p in glob.glob(os.path.join(dir_of(f), "**", "*.json"), recursive=True):
            if os.path.getsize(p) > 3_000_000:
                continue
            try:
                j = json.load(open(p, encoding="utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            v = j.get("started_utc") if isinstance(j, dict) else None
            if isinstance(v, str) and TS.match(v):
                starts.append(v[:16].replace("T", " "))
    return min(starts) if starts else None


EVIDENCE = "evidence/prefold/dann_r5_prereg_kaggle_v2_20260902.json"
PACK = "runs/dann_main/results_pack.md"
SELF_DESCRIBING = {"recorded_utc", "sha256_of_body_without_this_field"}


def prefold_dann(integrity):
    """The DANN pre-registration's pre-fold copy: ASSERTED from the files, never typed.

    Asserts (1) the copy's recomputed self-certifying digest starts with the prefix the
    results pack prints for the history entry described as the pre-declaration before any
    confirmatory fold; (2) its own recorded_utc precedes the earliest DANN confirmatory
    started_utc; (3) every key of the copy is in the current file with an identical value,
    except the two self-describing keys. Returns the facts the table and macros use.
    """
    pack = open(C.R(PACK), encoding="utf-8").read()
    hist = re.findall(r"^\|\s*(\d+)\s*\|\s*`([0-9a-f]{16})`\s*\|\s*(.+?)\s*\|\s*$", pack, re.M)
    assert len(hist) == 11, len(hist)
    prefold = [(int(n), h) for n, h, d in hist if "before any confirmatory fold" in d]
    arc = [int(n) for n, h, d in hist if "prediction_arc_suppression" in d]
    assert len(prefold) == 1 and len(arc) == 1, (prefold, arc)
    entry, prefix = prefold[0]

    sc = integrity.self_certifying_sha(C.R(EVIDENCE))
    assert sc and sc.startswith(prefix), (sc, prefix)                       # (1)
    ev = json.load(open(C.R(EVIDENCE), encoding="utf-8"))
    cur = json.load(open(C.R("runs/dann_r5_prereg.json"), encoding="utf-8"))
    rec = ev["recorded_utc"][:16].replace("T", " ")
    first = first_start("DANN x nb3")
    assert first and rec < first, (rec, first)                              # (2)
    canon = lambda v: json.dumps(v, sort_keys=True)
    keys = [k for k in ev if k not in SELF_DESCRIBING]
    diff = [k for k in keys if k not in cur or canon(ev[k]) != canon(cur[k])]
    assert not diff, diff                                                   # (3)
    added = sorted(set(cur) - set(ev))
    return {"entry": entry, "prefix": prefix, "arc_entry": arc[0], "sc": sc,
            "recorded": rec, "first": first, "n_identical": len(keys),
            "added": added}


KAGGLE_V14 = "evidence/kaggle_bundle_v14"
MONTHS = {9: ("Sept", "September")}


def kaggle_lambda0(integrity):
    """The lambda=0 pre-registration's Kaggle copy (final build, PROJECTv7 7.61): ASSERTED.

    Asserts (1) the copy in evidence/kaggle_bundle_v14/ is byte-identical to the protected
    file; (2) the Kaggle version date that SOURCE.md records precedes the earliest lambda=0
    confirmatory started_utc. Returns the date as the Source column and Appendix B.2 print
    it, and that first start."""
    rel = "runs/lambda0_r6_prereg.json"
    copy = KAGGLE_V14 + "/lambda0_r6_prereg.json"
    assert integrity.whole_file_sha(C.R(copy)) == integrity.whole_file_sha(C.R(rel))   # (1)
    src = open(C.R(KAGGLE_V14 + "/SOURCE.md"), encoding="utf-8").read()
    m = re.findall(r"^\| Version date \| \*\*(20\d\d)-(\d\d)-(\d\d)\*\*", src, re.M)
    assert len(m) == 1, m
    y, mo, d = (int(x) for x in m[0])
    first = first_start("lambda=0 (nb3)")
    assert first and f"{y:04d}-{mo:02d}-{d:02d}" < first[:10], (m, first)       # (2)
    return f"{d} {MONTHS[mo][0]} {y}", f"{d} {MONTHS[mo][1]} {y}", first


def main() -> int:
    import integrity
    # Without the lab log (the public release) integrity.protected_files() reads the digests from
    # public_inputs/record_extract.json. With the log, the committed extract must be what the log
    # gives now, or the build stops: the extract cannot fall behind the log.
    if os.path.isfile(RECORD):
        import write_record_extract
        if write_record_extract.main(["--check"]):
            raise SystemExit("public_inputs/record_extract.json is stale; run "
                             "python scripts/write_record_extract.py")
    pf = prefold_dann(integrity)
    kl_date, kl_date_long, kl_first = kaggle_lambda0(integrity)

    # This table covers the eight artifacts in ROLE, looked up by name. Later enrolments (the
    # 1:10 experiment, from 2026-10-13) are in the record and the inventory too and are
    # ignored here; they have their own table (A2_hashes_v6, PROJECTv7 7.56).
    found = {rel: (rel, f, s) for rel, f, s in integrity.protected_files(md_path=RECORD)}
    missing = sorted(set(ROLE) - set(found))
    assert not missing, f"not found under a recorded hash: {missing}"
    rows = sorted(found[rel] for rel in ROLE)
    # Membership of the inventory verify() actually walks -- NOT re-derived from the
    # record, because that is the whole point: the two sets differ.
    inv = {rel: (f, s) for rel, f, s in integrity._load_inventory()}
    inventory = set(inv)
    for rel, f, s in rows:                       # an enrolled one must verify there as well
        assert rel not in inv or inv[rel] == (f, s), (rel, inv.get(rel), (f, s))

    body = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{The protected artifacts, and when each was recorded relative to "
        r"the runs it governs.} Each file is read-only once written and checked by digest, "
        r"so an overwrite fails loudly instead of succeeding quietly. \textbf{The digests "
        r"and the dates are read from the files at build time, not transcribed.} "
        r"\emph{Recorded} is the file's own timestamp; where a file carries none, "
        r"\emph{Source} says whether it carries its own before-flag or whether its timing "
        r"rests on the project record alone. \emph{vs.\ runs} compares that with the "
        r"earliest start time of the confirmatory runs the file governs, read from those "
        r"runs' own files. The DANN pre-registration's file was regenerated after DANN's "
        r"confirmatory folds, so its row is dated from an archived pre-fold copy whose "
        r"digest matches the file's recorded hash history, and the build asserts that the "
        r"copy's pre-registered keys are unchanged in the current file (see the notes and "
        r"Appendix~\ref{sec:hashes}). Two flavours of digest are checked separately: a "
        r"whole-file digest, and a self-certifying one taken over the JSON body with its "
        r"own hash field removed. \textbf{The last column separates hashed from "
        r"verified}: eight artifacts are hashed in the record, seven are in the inventory "
        r"every analysis run re-checks, and the eighth is recorded but unenrolled. The last "
        r"row is protected but is not a pre-registration: it governs no run.}",
        r"\label{tab:hashes}",
        # p{} columns wrap the long cells so the table fits the text width; the digest
        # and the verdict keep their own width.
        r"\begin{tabular}{>{\TRAG}p{2.6cm}>{\TRAG}p{1.3cm}>{\TRAG}p{1.75cm}>{\TRAG}p{1.3cm}>{\TRAG}p{1.3cm}>{\TRAG}p{1.4cm}l>{\TRAG}p{1.4cm}}",
        r"\toprule",
        r"Artifact & Recorded (UTC) & Source & First run (UTC) & vs.\ runs & Flavour & "
        r"sha256 (first 16) & Verified \\",
        r"\midrule",
    ]
    report = []
    ordered = sorted(rows, key=lambda r: ROLE[r[0]][1] == "NONE")   # governing rows first
    for i, (rel, flavour, sha) in enumerate(ordered):
        label, arm, _what = ROLE[rel]
        date, source, key = file_date(rel)
        if rel == "runs/dann_r5_prereg.json":
            # Two rows (PROJECTv7 7.65). The file was regenerated after the folds, so its own
            # digest and date are those of the regeneration: that row is dated from the file
            # and its verdict computed. The pre-fold state is the archived copy, ASSERTED in
            # prefold_dann() -- its digest, date, run and ordering all read; it is not in the
            # inventory.
            run = pf["first"]
            assert pf["recorded"] < run
            report.append((label + ", pre-fold copy", pf["recorded"], "archived copy",
                           "evidence recorded_utc", run, "before"))
            body.append(" & ".join([
                label + ", pre-fold copy", pf["recorded"], "archived copy", run, "before",
                flavour.replace("_", r"\_\allowbreak{}"), r"\texttt{" + pf["sc"][:16] + r"}",
                "no (archived)",
            ]) + r" \\")
            verdict = "before" if date < run else r"\textbf{after}"
            assert source == "file" and verdict != "before", (date, run)
            report.append((label + ", regenerated file", date, source, key, run, verdict))
            body.append(" & ".join([
                label + ", regenerated file", date, source, run, verdict,
                flavour.replace("_", r"\_\allowbreak{}"), r"\texttt{" + sha[:16] + r"}",
                "yes" if rel in inventory else r"\textbf{no}",
            ]) + r" \\")
            continue
        if rel == "runs/lambda0_r6_prereg.json":
            # the file carries no timestamp; its timing is the Kaggle version that holds a
            # byte-identical copy, ASSERTED in kaggle_lambda0()
            source, run, verdict = f"Kaggle dataset version, {kl_date}", kl_first, "before"
            report.append((label, date, source, key, run, verdict))
            body.append(" & ".join([
                label, date or "---", source, run, verdict,
                flavour.replace("_", r"\_\allowbreak{}"), r"\texttt{" + sha[:16] + r"}",
                "yes" if rel in inventory else r"\textbf{no}",
            ]) + r" \\")
            continue
        if arm == "NONE":
            if i and ROLE[ordered[i - 1][0]][1] != "NONE":
                body.append(r"\midrule")
                body.append(r"\multicolumn{8}{l}{\emph{Protected, not a pre-registration "
                            r"(governs no run):}} \\")
            run, verdict = "---", "n/a"
        elif arm is None:
            run = "---"
            verdict = "before (flag)" if source == "file (flag only)" else "not checkable"
        else:
            run = first_start(arm) or "---"
            if date and run != "---":
                verdict = "before" if date < run else r"\textbf{after}"
            elif source == "file (flag only)":
                verdict = "before (flag)"
            else:
                verdict = "not checkable"
        report.append((label, date, source, key, run, verdict))
        body.append(" & ".join([
            label, date or "---", source, run, verdict,
            flavour.replace("_", r"\_\allowbreak{}"), r"\texttt{" + sha[:16] + r"}",
            "yes" if rel in inventory else r"\textbf{no}",
        ]) + r" \\")
    # The notes are set as a paragraph under the float rather than inside it: with them
    # inside, the float is taller than a page.
    body += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
        r"\begingroup\footnotesize\noindent\textbf{Notes to Table~\ref{tab:hashes}.} "
        r"What each one fixed: ",
    ]
    for rel, _flavour, _sha in ordered:
        label, _arm, what = ROLE[rel]
        body.append(r"\emph{" + label + r"} --- " + what + ".")
    body += [
        r"A further digest in the project record, \texttt{7356acda\ldots}, is the hash of "
        r"a \emph{superseded} pre-registration text, kept deliberately and correctly "
        r"matching no live file. \textbf{The one row marked \emph{no} above is hashed in "
        r"the record but absent from \texttt{runs/protected\_artifacts.json}}, the "
        r"inventory \texttt{verify()} walks, so it receives neither the re-check nor the "
        r"read-only protection the other seven do. It is reported here rather than "
        r"quietly enrolled, because enrolling it now would date its protection to the "
        r"paper rather than to the declaration.\par\endgroup",
    ]

    C.write_table(os.path.join(C.TABLES, "A2_hashes.tex"), "\n".join(body),
                  header_comment=("Appendix: protected pre-registration artifacts.\n"
                                  "Digests RE-DERIVED at build time by "
                                  "scripts/integrity.protected_files();\n"
                                  "dates READ from each file's own fields; first-run "
                                  "times READ from the runs' own files."),
                  size="scriptsize",
                  placement="htbp")
    C.write_table(os.path.join(C.TABLES, "M_a2.tex"), "\n".join([
        rf"\newcommand{{\dannPrefoldEntry}}{{{pf['entry']}}}",
        rf"\newcommand{{\dannPrefoldSha}}{{{pf['sc'][:8]}}}",
        rf"\newcommand{{\dannArcEntry}}{{{pf['arc_entry']}}}",
        rf"\newcommand{{\dannPrefoldIdenticalKeys}}{{{pf['n_identical']}}}",
        rf"\newcommand{{\dannPrefoldAddedKeys}}{{{len(pf['added'])}}}",
        rf"\newcommand{{\lambdaKaggleDate}}{{{kl_date_long}}}",
    ]), header_comment=("Macros for Appendix A2's DANN pre-fold paragraph and the lambda=0 "
                        "Kaggle date, asserted from the evidence files by "
                        "paper/scripts/make_a2_hashes.py."), size="")
    print(f"  DANN pre-fold copy: self-cert {pf['sc'][:16]} = history entry {pf['entry']} "
          f"prefix {pf['prefix']}; recorded {pf['recorded']} < first fold {pf['first']}; "
          f"{pf['n_identical']} keys identical; added {pf['added']}; arc entry {pf['arc_entry']}")
    for r in report:
        print("  ", " | ".join(str(x) for x in r))
    print(f"  {len(rows)} protected artifacts, digests re-derived from disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
