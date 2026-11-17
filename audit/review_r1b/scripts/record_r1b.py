"""
record_r1b.py -- snapshot diffs (prefix audit/review_r1b/) and finish-sequence summaries.

    python -B audit/review_r1b/scripts/record_r1b.py <scratch_dir>

<scratch_dir> holds snap_before.json and any of snap_mid.json / snap_final.json /
snap_end.json (from audit/review_r1/scripts/snapshot.py), plus makeall.log, makeall/,
tools/*.log, tools/paper_DRAFT.zip, tools/data/ and compile/main.log from the redirected
finish sequence. Writes out/SNAPSHOT_diff.json and out/FINISH_tools.json.

round-1's snapdiff.py hard-codes the audit/review_r1/ prefix, so the diff is recomputed
here with this folder's prefix; the log-line extraction reuses round-1's record_finish.lines.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "review_r1", "scripts"))
import rr_common as RC                                   # noqa: E402
from record_finish import lines                          # noqa: E402

SCRIPT = os.path.abspath(__file__)
OUT = os.path.join(os.path.dirname(HERE), "out")
PREFIX = "audit/review_r1b/"


def diff(a, b):
    added = sorted(set(b) - set(a))
    return {"n_before": len(a), "n_after": len(b), "n_added": len(added),
            "added_inside_audit_review_r1b": [k for k in added if k.startswith(PREFIX)],
            "added_outside_audit_review_r1b": [k for k in added if not k.startswith(PREFIX)],
            "removed": sorted(set(a) - set(b)),
            "content_changed": sorted(k for k in set(a) & set(b) if a[k][2] != b[k][2]),
            "mtime_only_same_sha256": sorted(k for k in set(a) & set(b)
                                             if a[k][2] == b[k][2] and a[k][1] != b[k][1])}


def main():
    RC.OUT = OUT
    S = sys.argv[1]
    I = RC.Inputs()
    snaps, diffs = {}, {}
    before = json.load(open(os.path.join(S, "snap_before.json")))
    for n in ("snap_before", "snap_mid", "snap_final", "snap_end"):
        p = os.path.join(S, n + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        snaps[n] = {"file_outside_repo": p.replace("\\", "/"), "sha256": RC.sha256(p),
                    "n_files": len(d), "total_bytes": sum(v[0] for v in d.values())}
        if n != "snap_before":
            diffs[n + "_vs_before"] = diff(before, d)
    RC.write_json("SNAPSHOT_diff.json", {"snapshots": snaps, "diffs": diffs,
                                         "prefix": PREFIX},
                  script=SCRIPT, inputs=I, pool="n/a", folds="n/a",
                  notes=["snap_mid: after L1/L2; snap_final: after the finish sequence; "
                         "snap_end: after the report and these record JSONs"])

    T = os.path.join(S, "tools")
    mk = os.path.join(S, "makeall.log")
    if not os.path.exists(mk):
        return
    cmp = {}
    for d in ("tables", "figures", "data"):
        for f in sorted(glob.glob(os.path.join(S, "makeall", d, "*"))):
            tracked = RC.R("paper", d, os.path.basename(f))
            a = open(f, "rb").read()
            b = open(tracked, "rb").read() if os.path.exists(tracked) else None
            rec = {"byte_identical": a == b}
            if not rec["byte_identical"] and b is not None and f.endswith(".pdf"):
                strip = lambda x: re.sub(rb"/CreationDate \(D:[^)]*\)", b"", x)
                rec["identical_except_CreationDate"] = strip(a) == strip(b)
            cmp[f"{d}/{os.path.basename(f)}"] = rec
    mtxt = open(mk, encoding="utf-8", errors="replace").read()
    zs = zipfile.ZipFile(os.path.join(T, "paper_DRAFT.zip"))
    zt = zipfile.ZipFile(RC.R("paper_DRAFT.zip"))
    I.add(RC.R("paper_DRAFT.zip"))
    mlog = open(os.path.join(S, "compile", "main.log"), encoding="utf-8", errors="replace").read()
    res = {
        "environment": "every tool run with PYTHONDONTWRITEBYTECODE=1 and outputs redirected to "
                       "scratch copies; nothing under paper/ or <repo>/paper_DRAFT.zip written",
        "1_make_all": {
            "summary_lines": lines(mk, [r"ALL TABLES AND FIGURES", r"checks, \d+ mismatch",
                                        r"^rc="]),
            "make_m_prose_claims": {"n": len(re.findall(r"(?m)^  claim \S+", mtxt)),
                                    "n_holds": len(re.findall(r"(?m)^  claim \S+\s+holds ",
                                                              mtxt))},
            "n_outputs": len(cmp),
            "n_byte_identical": sum(v["byte_identical"] for v in cmp.values()),
            "n_pdf_identical_except_CreationDate": sum(
                bool(v.get("identical_except_CreationDate")) for v in cmp.values()),
            "byte_comparison_with_tracked": cmp},
        "2_check_latex": lines(os.path.join(T, "check_latex.log"), [r"\.tex files", r"^rc="]),
        "3_crosscheck": {
            "summary_lines": lines(os.path.join(T, "crosscheck.log"),
                                   [r"checks, \d+ mismatch", r"pre-registered pool vs", r"^rc="]),
            "crosscheck_json_byte_identical_to_tracked":
                open(os.path.join(T, "data", "crosscheck.json"), "rb").read() ==
                open(RC.R("paper", "data", "crosscheck.json"), "rb").read()},
        "4_prose_macros_verify": lines(os.path.join(T, "prose_macros.log"),
                                       [r"replaced,", r"^rc="]),
        "5_verify_orig_tables": {
            "tokens": lines(os.path.join(T, "vot_tokens.log"), [r"tables regenerated", r"^rc="]),
            "bytes": lines(os.path.join(T, "vot_bytes.log"), [r"tables regenerated", r"^rc="])},
        "6_test_make_archive": lines(os.path.join(T, "test_make_archive.log"),
                                     [r"passed", r"^rc="]),
        "7_integrity_verify": lines(os.path.join(T, "integrity.log"), [r"verified", r"^rc="]),
        "8_make_archive_draft": {
            "how": "audit/review_r1/scripts/make_archive_draft_redirected.py (unchanged)",
            "summary_lines": lines(os.path.join(T, "make_archive.log"),
                                   [r"SUBMISSION GATE", r"^rc="]),
            "same_entry_names_as_tracked": set(zs.namelist()) == set(zt.namelist()),
            "n_entries": len(zs.namelist()),
            "entries_with_different_content": [n for n in zs.namelist() if n in zt.namelist()
                                               and zs.read(n) != zt.read(n)]},
        "9_tectonic": {
            "summary_lines": lines(os.path.join(T, "tectonic.log"),
                                   [r"Writing `main.pdf`", r"^rc=", r"^Pages:", r"error"]),
            "overfull_boxes": len(re.findall(r"Overfull", mlog)),
            "underfull_boxes": len(re.findall(r"Underfull", mlog)),
            "latex_warnings": len(re.findall(r"LaTeX Warning", mlog))},
    }
    RC.write_json("FINISH_tools.json", res, script=SCRIPT, inputs=I, pool="as each tool",
                  folds="as each tool")


if __name__ == "__main__":
    main()
