"""a3_window_pairs.py -- k-window paired t-intervals among CORAL, DANN and FreeMatch, cleaned pool.

    python -B audit/rewrite_v5/scripts/a3_window_pairs.py

Rewrite v5, Part A3 (brief_rewrite_v5.md). Read-only. For every pair among CORAL, DANN and
FreeMatch, at every declared read-off (CORAL 2000, FreeMatch 1500, DANN 4000 and its
pre-declared sensitivity read-off 2000), under BOTH declared score combinations (mean_logit,
the primary; mean_rank, the secondary; runs/round7/window_prereg.json -> score_combination),
at k = 1 and the declared k = 8: the paired t-interval on the four per-fold differences
(df = 3), and the fold signs.

Input: paper/data/window_average_D1.json, the per-fold windowed TPRs on the cleaned (D1) pool,
built by paper/scripts/build_pool_recomputes.py under the declared window rule. No logit is
re-scored and no model is loaded. The one pair that file already stores as a contrast,
DANN(2000) - CORAL(2000), is re-derived here and asserted equal, under both combinations and
both k.

Writes audit/rewrite_v5/out/A3_window_pairs.json and nothing else.
"""
import hashlib
import json
import math
import os

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(REPO, "paper", "data", "window_average_D1.json")
OUT = os.path.join(os.path.dirname(HERE), "out", "A3_window_pairs.json")

PAIRS = [("DANN x nb3", 4000, "CORAL x nb3", 2000),
         ("DANN x nb3", 2000, "CORAL x nb3", 2000),
         ("DANN x nb3", 4000, "FreeMatch x nb3", 1500),
         ("DANN x nb3", 2000, "FreeMatch x nb3", 1500),
         ("CORAL x nb3", 2000, "FreeMatch x nb3", 1500)]
COMBS = ("mean_logit", "mean_rank")
KS = (1, 8)


def main() -> int:
    raw = open(SRC, "rb").read()
    d = json.loads(raw)
    assert d["k_primary"] == 8, d["k_primary"]
    assert d["test_pool"]["pool"] == "D1", d["test_pool"]
    arms = d["arms"]
    t975 = float(stats.t.ppf(0.975, 3))

    def per_fold(arm, ro, comb, k):
        cell = arms[arm]["readouts"][str(ro)][comb][str(k)]
        assert list(arms[arm]["folds"]) == [1, 2, 3, 4], arms[arm]["folds"]
        return [float(v) for v in cell["tpr_per_fold"]]

    rows = []
    for a, ra, b, rb in PAIRS:
        for comb in COMBS:
            for k in KS:
                x, y = per_fold(a, ra, comb, k), per_fold(b, rb, comb, k)
                diffs = [p - q for p, q in zip(x, y)]
                m = sum(diffs) / 4
                sd = math.sqrt(sum((v - m) ** 2 for v in diffs) / 3)
                hw = t975 * sd / 2
                rows.append({
                    "contrast": f"{a.split()[0]}({ra}) - {b.split()[0]}({rb})",
                    "combination": comb, "combination_role": "primary" if comb == "mean_logit"
                    else "secondary", "k": k, "mean": m, "ci": [m - hw, m + hw],
                    "half_width": hw, "includes_zero": bool(m - hw <= 0.0 <= m + hw),
                    "signs_agreeing": int(max(sum(v > 0 for v in diffs), sum(v < 0 for v in diffs))),
                    "diffs_folds_1_4": diffs})

    # Consistency: the one stored contrast among the three must be reproduced exactly.
    stored = d["contrasts_at_k_primary"]["DANN(2000) - CORAL(2000)"]
    for k in KS:
        mine = [r for r in rows if r["contrast"] == "DANN(2000) - CORAL(2000)"
                and r["combination"] == "mean_logit" and r["k"] == k][0]
        assert abs(mine["mean"] - stored[str(k)]["mean"]) < 1e-12, (k, mine["mean"])
        assert abs(mine["ci"][0] - stored[str(k)]["ci_lo"]) < 1e-9, (k, mine["ci"])

    summary = {}
    for comb in COMBS:
        k8 = [r for r in rows if r["combination"] == comb and r["k"] == 8]
        summary[comb] = {"n_pairs": len(k8),
                         "n_including_zero": sum(r["includes_zero"] for r in k8),
                         "excluding_zero": [r["contrast"] for r in k8 if not r["includes_zero"]]}

    out = {"what": "k-window paired t-intervals (df=3) among CORAL, DANN and FreeMatch, cleaned (D1) pool, "
                   "folds 1-4, every declared read-off, both declared score combinations",
           "status": "post_hoc_exploratory",
           "producing_script": "audit/rewrite_v5/scripts/a3_window_pairs.py",
           "source": "paper/data/window_average_D1.json",
           "source_sha256": hashlib.sha256(raw).hexdigest(),
           "declared_rule": "runs/round7/window_prereg.json (k grid 1,2,4,8; primary k=8; "
                            "primary combination mean_logit, secondary mean_rank)",
           "t975_df3": t975,
           "consistency_check": "DANN(2000) - CORAL(2000), mean_logit, k=1 and k=8: re-derived "
                                "mean and lower bound equal the stored contrast",
           "summary_k8": summary, "rows": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for r in rows:
        print(f"  {r['contrast']:30} {r['combination']:10} k={r['k']}  {r['mean']:+.4f} "
              f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  {r['signs_agreeing']}/4"
              f"{'' if r['includes_zero'] else '  EXCLUDES ZERO'}")
    print(f"  k=8 summary: {summary}")
    print(f"  wrote {os.path.relpath(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
