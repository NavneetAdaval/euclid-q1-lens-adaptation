r"""
make_k8_pairs.py -- Appendix C's table of paired contrasts among CORAL, DANN and FreeMatch at
k = 1 and after the pre-declared k = 8 window average, plus the macros the text uses for it.
==============================================================================================

    python paper/scripts/make_k8_pairs.py

Rewrite v5 (brief_rewrite_v5.md, Part A3; addendum 1, items 1-4).

Input: ``audit/rewrite_v5/out/A3_window_pairs.json`` (audit/rewrite_v5/scripts/a3_window_pairs.py),
whose rows difference the per-fold windowed TPRs of ``paper/data/window_average_D1.json`` on the
cleaned (D1) pool, folds 1-4, paired t with df = 3. Before anything is written:

  * the audit file's recorded ``source_sha256`` must equal the current
    window_average_D1.json, so the table cannot describe stale windows;
  * at k = 1 the two score combinations must give identical intervals (a one-checkpoint
    window has nothing to combine), so the table prints a single k = 1 block;
  * the k = 1 rows must reproduce _common.contrast on the stored scores (the single-checkpoint
    read-offs the main contrast table uses);
  * the macros below must be consistent with the rows (FreeMatch's lead, the further-interval
    count).

Writes ``paper/tables/C_k8_pairs.tex`` (label tab:k8pairs) and ``paper/tables/M_k8pairs.tex``:

  * ``\rwFiveKWord``          -- the declared primary window length, as a word ("eight");
  * ``\rwFiveKPairsWord``     -- the number of pairs in the table, as a word;
  * ``\rwFiveFurtherKEight``  -- k = 8 intervals in the table (pairs x combinations), as a word;
  * ``\rwFiveFurtherKOne``    -- k = 1 intervals in the table that the main contrast table
                                 (_common.CONTRAST_ROWS) does not already report, as a word;
  * ``\rwFivePosPerFold``     -- test positives per confirmatory fold, rounded mean over
                                 folds 1-4 (stored scores, _common.suite n_pos).
"""
from __future__ import annotations

import hashlib
import json
import os

import _common as C

AUDIT = C.R("audit", "rewrite_v5", "out", "A3_window_pairs.json")
WINDOW = os.path.join(C.DATA, "window_average_D1.json")
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
        8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
NAME = {"DANN": "DANN", "CORAL": "CORAL", "FreeMatch": "FreeMatch"}
KEY = {"DANN": "dann", "CORAL": "coral", "FreeMatch": "freematch"}


def _parse(contrast):
    a, b = contrast.split(" - ")
    na, ra = a[:-1].split("(")
    nb, rb = b[:-1].split("(")
    return na, int(ra), nb, int(rb)


def signs_with_mean(r):
    """Folds agreeing with the sign of the mean -- the definition the main contrast table
    uses (_common.contrast n_same_sign), not the audit file's majority count."""
    s = 1.0 if r["mean"] >= 0 else -1.0
    return sum(1 for d in r["diffs_folds_1_4"] if d * s > 0)


def _cell(r):
    star = "" if r["includes_zero"] else r"$^{*}$"
    return (rf"${r['mean']:+.4f}${star}",
            rf"$[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]$",
            f"{signs_with_mean(r)}/4")


def load():
    a3 = json.load(open(AUDIT, encoding="utf-8"))
    raw = open(WINDOW, "rb").read()
    assert a3["source_sha256"] == hashlib.sha256(raw).hexdigest(), \
        "A3_window_pairs.json was computed from a different window_average_D1.json; rerun it"
    w = json.loads(raw)
    assert w["k_primary"] == 8 and w["test_pool"]["pool"] == "D1"
    return a3, w


def rows(a3):
    by = {}
    for r in a3["rows"]:
        by[(r["contrast"], r["combination"], r["k"])] = r
    pairs = []
    for r in a3["rows"]:
        if r["contrast"] not in pairs:
            pairs.append(r["contrast"])
    out = []
    for p in pairs:
        k1l, k1r = by[(p, "mean_logit", 1)], by[(p, "mean_rank", 1)]
        for f in ("mean", "half_width", "signs_agreeing"):
            assert abs(k1l[f] - k1r[f]) < 1e-12, (p, f)
        na, ra, nb, rb = _parse(p)
        with C.pool("D1"):
            c = C.contrast(KEY[na], ra, KEY[nb], rb)
        assert abs(c["mean"] - k1l["mean"]) < 1e-9 and abs(c["ci_lo"] - k1l["ci"][0]) < 1e-9, \
            (p, c, k1l)
        assert c["n_same_sign"] == signs_with_mean(k1l), (p, c["n_same_sign"])
        out.append((na, ra, nb, rb, k1l, by[(p, "mean_logit", 8)], by[(p, "mean_rank", 8)]))
    return out


def macros(a3, w, rr):
    k = int(w["k_primary"])
    n_pairs = len(rr)
    n_k8 = 2 * n_pairs
    main = {(a, ra, b, rb) for _g, _l, a, ra, b, rb, _m, _k in C.CONTRAST_ROWS}
    new_k1 = [r for r in rr if (KEY[r[0]], r[1], KEY[r[2]], r[3]) not in main
              and (KEY[r[2]], r[3], KEY[r[0]], r[1]) not in main]
    with C.pool("D1"):
        npos = [C.suite("freematch", f, 1500)["n_pos"] for f in (1, 2, 3, 4)]
    per_fold = round(sum(npos) / len(npos))
    # FreeMatch has the highest k=8 fold-mean under both combinations (the text says so).
    cells = {"FreeMatch": ("FreeMatch x nb3", "1500"), "DANN4000": ("DANN x nb3", "4000"),
             "DANN2000": ("DANN x nb3", "2000"), "CORAL": ("CORAL x nb3", "2000")}
    for comb in ("mean_logit", "mean_rank"):
        v = {n: w["arms"][a]["readouts"][ro][comb]["8"]["tpr_mean"] for n, (a, ro) in cells.items()}
        assert max(v, key=v.get) == "FreeMatch", (comb, v)
    # FreeMatch's lead excludes zero under mean rank only (the text says so).
    fm = [r for r in rr if "FreeMatch" in (r[0], r[2])]
    assert fm and all(r[6]["ci"][1] < 0 for r in fm), "mean rank: a FreeMatch lead admits zero"
    assert all(r[5]["includes_zero"] for r in fm), "mean logit: a FreeMatch lead excludes zero"
    # Every single-checkpoint interval among the three includes zero (the text says so).
    assert all(r[4]["includes_zero"] for r in rr)
    return [("rwFiveKWord", WORD[k]), ("rwFiveKPairsWord", WORD[n_pairs]),
            ("rwFiveFurtherKEight", WORD[n_k8]), ("rwFiveFurtherKOne", WORD[len(new_k1)]),
            ("rwFivePosPerFold", str(per_fold))]


def table(rr, mac):
    m = dict(mac)
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{\textbf{Paired contrasts among the three unfrozen adaptation arms} at each "
            r"arm's declared read-off ($k=1$) and after the pre-declared $k=8$ window average, "
            r"folds 1--4, cleaned pool, paired $t$ ($\mathrm{df}=3$). At $k=1$ the two score "
            r"combinations coincide. Computed after the results were known, from the declared "
            r"window rule; " + m["rwFiveFurtherKEight"] + r" further intervals at $k=8$, and "
            + m["rwFiveFurtherKOne"] + r" at $k=1$ that Table~\ref{tab:contrasts2} does not "
            r"report, all without multiplicity correction. Each pair's first line gives "
            r"$\Delta$ and the signs, the folds agreeing in direction with the mean; the second "
            r"line gives the 95\% interval. $^{*}$Interval excludes zero.}",
            r"\label{tab:k8pairs}",
            r"\begin{tabular}{llrcrcrc}", r"\toprule",
            r" & & \multicolumn{2}{c}{$k=1$} & \multicolumn{2}{c}{$k=8$, mean logit (primary)}"
            r" & \multicolumn{2}{c}{$k=8$, mean rank (secondary)} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
            r"Pair & Steps & $\Delta$ (95\% CI) & Signs & $\Delta$ (95\% CI) & Signs "
            r"& $\Delta$ (95\% CI) & Signs \\", r"\midrule"]
    for i, (na, ra, nb, rb, k1, k8l, k8r) in enumerate(rr):
        c1, c2, c3 = _cell(k1), _cell(k8l), _cell(k8r)
        first = [f"{NAME[na]} $-$ {NAME[nb]}", f"{ra} / {rb}",
                 c1[0], c1[2], c2[0], c2[2], c3[0], c3[2]]
        second = ["", "", c1[1], "", c2[1], "", c3[1], ""]
        if i:
            body.append(r"\addlinespace")
        body.append(" & ".join(first) + r" \\")
        body.append(" & ".join(second) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)


def main() -> int:
    a3, w = load()
    rr = rows(a3)
    mac = macros(a3, w, rr)
    C.write_table(os.path.join(C.TABLES, "C_k8_pairs.tex"), table(rr, mac),
                  header_comment="Appendix C (rewrite v5): k=1 and k=8 paired contrasts among "
                                 "CORAL, DANN and FreeMatch, from audit/rewrite_v5/out/"
                                 "A3_window_pairs.json.",
                  size="scriptsize", colsep="2pt")
    C.write_table(os.path.join(C.TABLES, "M_k8pairs.tex"),
                  "\n".join(rf"\newcommand{{\{n}}}{{{v}}}" for n, v in mac),
                  header_comment="Macros added by rewrite v5 addendum 1 (make_k8_pairs.py).",
                  size="")
    print("  wrote tables/C_k8_pairs.tex and tables/M_k8pairs.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
