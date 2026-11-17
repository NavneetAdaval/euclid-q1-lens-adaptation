r"""
audit_venue.py -- venue, precision and code path behind every number in a main-text table.
===========================================================================================

    python paper/scripts/audit_venue.py

One row per (arm, fold) behind Tables T1-T4, plus Appendix A4's three cells and round-1
FreeMatch (the licensed within-method pair). Every field is read from the run's OWN files
and nothing is inferred:

  * venue      -- an explicit GPU name (``gpu_name`` / ``gpu`` in the run's JSON, or a
                  ``[gpu] <name>`` line in the run's own log). Otherwise "not recorded",
                  with ``device`` quoted if the run recorded one.
  * precision  -- ``precision``, else the ``fp16`` flag; the evaluation precision is
                  ``eval_precision`` where recorded.
  * code path  -- ``data_path`` / ``encoder_eval`` / ``n_blocks`` where recorded.
  * paths      -- the prefix of paths the run wrote into its files (``/kaggle/...`` or
                  ``C:\...``). Shown in its own column because it is NOT a venue record: it
                  says where a file was written, not what hardware computed the number.

Then every reported contrast is checked side against side: a property is "shared" only if
both sides RECORD it and the records agree.

Writes ``paper/data/venue_audit.json`` and prints the tables. It feeds no table or macro;
it is an audit, re-runnable, and changes nothing in the paper.
"""
from __future__ import annotations

import glob
import json
import os
import re

import _common as C

NR = "not recorded"


def _json_files(d):
    return [p for p in (os.path.join(d, n) for n in ("run.json", "run_config.json",
                                                      "results.json"))
            if os.path.isfile(p)]


def _find(obj, keys):
    """First value for any of ``keys`` anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and not isinstance(obj[k], (dict, list)):
                return obj[k]
        for v in obj.values():
            r = _find(v, keys)
            if r is not None:
                return r
    return None


def _path_prefix(texts):
    s = " ".join(texts)
    kag, loc = "/kaggle/" in s, bool(re.search(r"[A-Z]:\\\\Users", s))
    return ("/kaggle/" if kag else "") + (" + " if kag and loc else "") + \
           ("C:\\Users\\..." if loc else "") or "none"


def record(d):
    files = _json_files(d)
    objs = [json.load(open(p, encoding="utf-8")) for p in files]
    raw = [open(p, encoding="utf-8").read() for p in files]
    get = lambda *ks: next((v for v in (_find(o, ks) for o in objs) if v is not None), None)

    gpu = get("gpu_name", "gpu")
    if gpu is None:
        for lp in glob.glob(os.path.join(d, "*.log")):
            m = re.search(r"\[gpu\]\s+([^|\n]+?)\s+\|", open(lp, encoding="utf-8",
                                                          errors="replace").read())
            if m:
                gpu = m.group(1).strip()
                break
    device = get("device")
    venue = gpu if gpu else (f"{NR} (device: {device})" if device else NR)

    prec = get("precision")
    if prec is None:
        fp16 = get("fp16")
        prec = None if fp16 is None else ("fp16" if fp16 else "fp32")
    evalp = get("eval_precision")
    parts = [f"{k}={get(k)}" for k in ("data_path", "encoder_eval") if get(k) is not None]
    nb = get("n_blocks")
    if nb is not None:
        parts.append(f"n_blocks={nb}")
    code = ", ".join(parts) or NR
    return {"dir": os.path.relpath(d, C.REPO), "venue": venue, "gpu_recorded": gpu,
            "precision": prec or NR, "eval_precision": evalp or NR, "code_path": code,
            "path_prefix": _path_prefix(raw)}


# (arm label, folds, dir-for-fold function, which tables)
NB3 = (1, 2, 3, 4)
ARMS = [
    ("Baseline A (frozen)", (0, 1, 2, 3, 4), lambda f: C.R("runs", "bakeoff_baseline_A"),
     "T1 T2 T3 T4 T5"),
    ("lambda=0 (nb3)", NB3, lambda f: C.R("runs", "baseline_unfrozen", f"fold{f}",
                                          "extracted", f"lambda0_r6_fold{f}"), "T1 T2 T4 T5"),
    ("BNM x nb3", NB3, lambda f: C.R("runs", "bnm_r6_folds", f"bnm_r6_fold{f}"),
     "T1 T2 T4 T5"),
    ("FreeMatch x nb3", NB3, lambda f: C.R("runs", f"freematch_nb3_unfrozen_fold{f}"),
     "T1 T2 T4 T5"),
    ("DANN x nb3", NB3, lambda f: C.R("runs", "dann_main", f"fold{f}", "extracted",
                                      f"fold{f}"), "T1 T2 T4 T5"),
    ("CORAL x nb3", NB3, lambda f: C.R("kaggle", "coral_r4", "runs", "main", "extracted",
                                       "coral_r4_confirm", f"fold{f}"), "T1 T2 T4 T5"),
    ("round-1 CORAL (frozen)", (0, 1, 2, 3, 4), lambda f: C.R("runs", "bakeoff_coral"), "T3"),
    ("round-1 AdaMatch (frozen)", (0, 1, 2, 3, 4),
     lambda f: C.R("runs", "bakeoff_adamatch_5fold"), "T3"),
    ("round-1 FreeMatch (frozen)", (0, 1, 2, 3, 4),
     lambda f: C.R("runs", "bakeoff_freematch_5fold"), "T3, licensed pair"),
    ("round-1 DANN probe (frozen)", (0,), lambda f: C.R("runs", "curve_dann_fold0"), "T3"),
    ("A4 cell A", (0,), lambda f: C.R("runs", "baselineA_fold0_nb0_image_noaug_fp32"), "A4"),
    ("A4 cell B", (0,), lambda f: C.R("runs", "baselineA_fold0_nb0_image_weak_fp32"), "A4"),
    ("A4 cell C", (0,), lambda f: C.R("runs", "baselineA_fold0_nb0_image"), "A4"),
]
KEY = {"baseline": "Baseline A (frozen)", "lambda0": "lambda=0 (nb3)", "bnm": "BNM x nb3",
       "freematch": "FreeMatch x nb3", "dann": "DANN x nb3", "coral": "CORAL x nb3"}


def main() -> int:
    table = {}
    for arm, folds, dir_of, used in ARMS:
        for f in folds:
            r = record(dir_of(f))
            r.update(arm=arm, fold=f, tables=used)
            table[(arm, f)] = r

    def side(arm, folds):
        rs = [table[(arm, f)] for f in folds]
        def one(k):
            vals = {r[k] for r in rs}
            return vals.pop() if len(vals) == 1 else " / ".join(sorted(vals))
        return {k: one(k) for k in ("venue", "precision", "eval_precision", "path_prefix")}

    def verdict(a, b, k):
        va, vb = a[k], b[k]
        if NR in va or NR in vb:
            return f"cannot establish ({'both' if NR in va and NR in vb else 'one side'} not recorded)"
        return "shared" if va == vb else "DIFFERS"

    contrasts = []
    for panel, label, aa, ra, bb, rb, match, mark in C.CONTRAST_ROWS:
        a, b = side(KEY[aa], NB3), side(KEY[bb], NB3)
        contrasts.append(dict(contrast=f"{aa}@{ra} - {bb}@{rb}", a=a, b=b,
                              venue=verdict(a, b, "venue"),
                              precision=verdict(a, b, "precision"),
                              eval_precision=verdict(a, b, "eval_precision")))
    a, b = side("FreeMatch x nb3", NB3), side("round-1 FreeMatch (frozen)", NB3)
    contrasts.append(dict(contrast="freematch nb3@1500 - round-1 freematch@8000 (licensed)",
                          a=a, b=b, venue=verdict(a, b, "venue"),
                          precision=verdict(a, b, "precision"),
                          eval_precision=verdict(a, b, "eval_precision")))
    for name, x, y in (("A4 B - A (augmentation leg)", "A4 cell B", "A4 cell A"),
                       ("A4 C - B (precision leg)", "A4 cell C", "A4 cell B"),
                       ("A4 C - A (total)", "A4 cell C", "A4 cell A")):
        a, b = side(x, (0,)), side(y, (0,))
        contrasts.append(dict(contrast=name, a=a, b=b, venue=verdict(a, b, "venue"),
                              precision=verdict(a, b, "precision"),
                              eval_precision=verdict(a, b, "eval_precision")))

    out = {"runs": [table[k] for k in table], "contrasts": contrasts}
    path = os.path.join(C.DATA, "venue_audit.json")
    json.dump(out, open(path, "w", encoding="utf-8", newline="\n"), indent=1)

    print("| arm | fold | tables | venue (recorded) | train precision | eval precision | "
          "code path | path prefix (not a venue record) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in out["runs"]:
        print(f"| {r['arm']} | {r['fold']} | {r['tables']} | {r['venue']} | {r['precision']} | "
              f"{r['eval_precision']} | {r['code_path']} | {r['path_prefix']} |")
    print()
    print("| contrast | side A venue / precision | side B venue / precision | venue | "
          "train precision | eval precision |")
    print("|---|---|---|---|---|---|")
    for c in contrasts:
        print(f"| {c['contrast']} | {c['a']['venue']} / {c['a']['precision']} | "
              f"{c['b']['venue']} / {c['b']['precision']} | {c['venue']} | {c['precision']} | "
              f"{c['eval_precision']} |")
    print(f"\n  wrote {os.path.relpath(path, C.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
