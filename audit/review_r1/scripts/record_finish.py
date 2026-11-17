"""
record_finish.py -- store the snapshot diffs and the finish-sequence summary lines as JSON.

    python -B audit/review_r1/scripts/record_finish.py <scratch_dir>

<scratch_dir> holds snap_before.json, snap_mid.json, snap_final.json, diff_mid.json,
diff_final.json (from snapshot.py / snapdiff.py), and tools/*.log, makeall.log, the
byte-comparison results of the redirected make_all run and the tectonic log. Writes
out/SNAPSHOT_diff.json and out/FINISH_tools.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import zipfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_common as RC                                   # noqa: E402

SCRIPT = os.path.abspath(__file__)


def lines(path, pats):
    out = []
    for ln in open(path, encoding="utf-8", errors="replace").read().splitlines():
        if any(re.search(p, ln) for p in pats):
            out.append(ln.strip())
    return out


def main():
    S = sys.argv[1]
    I = RC.Inputs()
    snaps = {}
    for n in ("snap_before", "snap_mid", "snap_final"):
        p = os.path.join(S, n + ".json")
        d = json.load(open(p))
        snaps[n] = {"file_outside_repo": p.replace("\\", "/"), "sha256": RC.sha256(p),
                    "n_files": len(d), "total_bytes": sum(v[0] for v in d.values())}
    diffs = {n: json.load(open(os.path.join(S, n + ".json")))
             for n in ("diff_mid", "diff_final")}
    # optional last check, taken after the report and these record JSONs were written
    end = None
    if os.path.exists(os.path.join(S, "snap_end.json")):
        p = os.path.join(S, "snap_end.json")
        d = json.load(open(p))
        snaps["snap_end"] = {"file_outside_repo": p.replace("\\", "/"), "sha256": RC.sha256(p),
                             "n_files": len(d), "total_bytes": sum(v[0] for v in d.values())}
        end = json.load(open(os.path.join(S, "diff_end.json")))
    RC.write_json("SNAPSHOT_diff.json", {
        "snapshots": snaps,
        "after_analysis_vs_before": diffs["diff_mid"],
        "after_finish_sequence_vs_before": diffs["diff_final"],
        "final_check_vs_before": end,
        "final_check_note": "taken after REPORT_review_r1.md and the out/ record JSONs were "
                            "written; this file and the report's final edit postdate it "
                            "(both inside audit/review_r1/)",
        "method": "audit/review_r1/scripts/snapshot.py: sha256 of every file under the repo "
                  "root (venv and data included); snapdiff.py compares by path and sha256 and "
                  "separately lists same-sha256 files whose mtime changed",
    }, script=SCRIPT, inputs=I, pool="n/a", folds="n/a",
        notes=["snapshot files live in the session scratchpad, outside the repo; their sha256 "
               "is recorded here"])

    T = os.path.join(S, "tools")
    mk = os.path.join(S, "makeall.log")
    # make_all redirected run: byte comparison of every output against the tracked file
    cmp = {}
    for d in ("tables", "figures", "data"):
        for f in sorted(glob.glob(os.path.join(S, "makeall", d, "*"))):
            tracked = RC.R("paper", d, os.path.basename(f))
            a = open(f, "rb").read()
            b = open(tracked, "rb").read() if os.path.exists(tracked) else None
            same = a == b
            rec = {"byte_identical": same}
            if not same and b is not None and f.endswith(".pdf"):
                strip = lambda x: re.sub(rb"/CreationDate \(D:[^)]*\)", b"", x)
                rec["identical_except_CreationDate"] = strip(a) == strip(b)
                rec["creation_dates"] = {
                    "tracked": re.findall(rb"/CreationDate \(D:([^)]*)\)", b)[0].decode(),
                    "regenerated": re.findall(rb"/CreationDate \(D:([^)]*)\)", a)[0].decode()}
            cmp[f"{d}/{os.path.basename(f)}"] = rec
    zs = zipfile.ZipFile(os.path.join(T, "paper_DRAFT.zip"))
    zt = zipfile.ZipFile(RC.R("paper_DRAFT.zip"))
    I.add(RC.R("paper_DRAFT.zip"))
    zip_cmp = {"entries_scratch": len(zs.namelist()), "entries_tracked": len(zt.namelist()),
               "same_entry_names": set(zs.namelist()) == set(zt.namelist()),
               "entries_with_different_content": [n for n in zs.namelist()
                                                  if n in zt.namelist()
                                                  and zs.read(n) != zt.read(n)]}
    cc_same = (open(os.path.join(T, "data", "crosscheck.json"), "rb").read() ==
               open(RC.R("paper", "data", "crosscheck.json"), "rb").read())
    tlog = os.path.join(T, "tectonic.log")
    mlog = os.path.join(S, "compile", "main.log")
    res = {
        "environment": "every tool run with PYTHONDONTWRITEBYTECODE=1 and paper/data redirected "
                       "(PAPER_DATA_DIR) to a scratch copy; nothing under the repo was written",
        "1_make_all": {
            "how": "run with PAPER_TABLES_DIR / PAPER_FIGURES_DIR / PAPER_DATA_DIR pointed at "
                   "scratch copies of paper/{tables,figures,data}",
            "summary_lines": lines(mk, [r"ALL TABLES AND FIGURES", r"checks, \d+ mismatch",
                                        r"claims"]),
            "byte_comparison_with_tracked": cmp,
            "make_m_prose_claims": {
                "n": len(re.findall(r"(?m)^  claim \S+", open(mk, encoding="utf-8").read())),
                "n_holds": len(re.findall(r"(?m)^  claim \S+\s+holds ",
                                          open(mk, encoding="utf-8").read()))},
            "n_outputs": len(cmp),
            "n_byte_identical": sum(v["byte_identical"] for v in cmp.values()),
            "in_place_run": "SKIPPED: the six figure PDFs are not byte-identical (they differ "
                            "only in /CreationDate), so an in-place run would rewrite tracked "
                            "outputs"},
        "2_check_latex": lines(os.path.join(T, "check_latex.log"),
                               [r"\.tex files", r"DOES NOT", r"^rc="]),
        "3_crosscheck": {"summary_lines": lines(os.path.join(T, "crosscheck.log"),
                                                [r"checks, \d+ mismatch", r"pre-registered pool vs",
                                                 r"^rc="]),
                         "crosscheck_json_byte_identical_to_tracked": cc_same},
        "4_prose_macros_verify": lines(os.path.join(T, "prose_macros.log"),
                                       [r"replaced,", r"^rc="]),
        "5_verify_orig_tables": {
            "tokens": lines(os.path.join(T, "vot_tokens.log"), [r"tables regenerated", r"^rc="]),
            "bytes": lines(os.path.join(T, "vot_bytes.log"), [r"tables regenerated", r"^rc="])},
        "6_test_make_archive": lines(os.path.join(T, "test_make_archive.log"),
                                     [r"passed", r"^rc="]),
        "7_integrity_verify": lines(os.path.join(T, "integrity.log"), [r"verified", r"^rc="]),
        "8_make_archive_draft": {
            "how": "audit/review_r1/scripts/make_archive_draft_redirected.py: make_archive.main() "
                   "unchanged, OUT_DRAFT pointed at the scratch dir, so <repo>/paper_DRAFT.zip "
                   "was not rewritten",
            "summary_lines": lines(os.path.join(T, "make_archive.log"),
                                   [r"SUBMISSION GATE", r"WARNING", r"wrote .*paper_DRAFT",
                                    r"^rc="]),
            "scratch_zip_vs_tracked_zip": zip_cmp},
        "9_tectonic": {
            "how": "tectonic 0.17.0 from C:/Users/krris/ML/_tex_env, run on the unzipped scratch "
                   "paper_DRAFT.zip",
            "summary_lines": lines(tlog, [r"Writing `main.pdf`", r"^rc=", r"error", r"^Pages:"]),
            "overfull_boxes": len(re.findall(r"Overfull", open(mlog, encoding="utf-8",
                                                                errors="replace").read())),
            "underfull_boxes": len(re.findall(r"Underfull", open(mlog, encoding="utf-8",
                                                                  errors="replace").read())),
            "latex_warnings": len(re.findall(r"LaTeX Warning", open(mlog, encoding="utf-8",
                                                                    errors="replace").read())),
        },
    }
    RC.write_json("FINISH_tools.json", res, script=SCRIPT, inputs=I, pool="as each tool",
                  folds="as each tool")


if __name__ == "__main__":
    main()
