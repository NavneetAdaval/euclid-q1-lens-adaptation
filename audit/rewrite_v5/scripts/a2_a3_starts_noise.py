"""a2_a3_starts_noise.py -- rewrite v5 Part A2 (head starts) and A3 (independence vs paired SE).

    python -B audit/rewrite_v5/scripts/a2_a3_starts_noise.py

Read-only; cleaned (D1) pool; folds 1-4; no model is loaded.

A2. The frozen heads the unfrozen FreeMatch and CORAL arms start from are the round-1 frozen
    runs' step-8000 checkpoints (each unfrozen fold's run_config ``init_head_from``:
    runs/bakeoff_freematch_5fold/fold{k}/checkpoints/step_8000.pt and
    runs/bakeoff_coral/fold{k}/checkpoints/step_8000.pt). Their TPR@1e-3 is read from the
    stored score dumps through paper/scripts/_common.round1_summary. Checks:
      unfrozen FreeMatch (1500) - start == the licensed-pair row of Table 5 (+0.1049), and
      FreeMatch(1500) - lambda=0(4000) == (start - lambda=0) + (unfrozen - start).
    DANN, BNM and lambda=0 all start from Baseline A's step-8000 head; recorded per fold.

A3. For every pair in Figure 4 (declared read-offs, audit/review_r1/out/A1_resolution.json
    through make_review_macros.a1_sets) and every objective-against-objective row of Table 5:
      (a) the Section 6.4 independence figure sqrt(sigma_a^2 + sigma_b^2)/2, exactly as
          make_review_macros computes it (ratio_detrended_b_linfit_resid * |gap|), or from the
          per-arm sigma for a Table 5 row Figure 4 does not carry;
      (b) the paired t-interval's half-width / t(0.975, 3), on the cleaned pool;
    and the ratio a/b.

Writes audit/rewrite_v5/out/A2_head_starts.json and audit/rewrite_v5/out/A3_independence.json.
"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTDIR = os.path.join(os.path.dirname(HERE), "out")
os.environ.setdefault("PAPER_POOL", "D1")
sys.path.insert(0, os.path.join(REPO, "paper", "scripts"))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import _common as C  # noqa: E402
import make_review_macros as RV  # noqa: E402

FOLDS = (1, 2, 3, 4)


def fm(arm, ro):
    r = C.fold_mean(arm, "tpr", ro)
    return float(r[0] if isinstance(r, tuple) else r)


def init_heads():
    pats = {
        "freematch": ("runs", "freematch_nb3_unfrozen_fold[1-4]", "run_config.json"),
        "coral": ("kaggle", "coral_r4", "runs", "main", "extracted", "coral_r4_confirm", "fold*", "run.json"),
        "dann": ("runs", "dann_main", "fold*", "extracted", "fold*", "run_config.json"),
        "lambda0": ("runs", "baseline_unfrozen", "fold*", "extracted", "*", "run_config.json"),
        "bnm": ("runs", "bnm_r6_folds", "bnm_r6_fold*", "run_config.json"),
    }
    out = {}
    for arm, p in pats.items():
        fs = sorted(glob.glob(os.path.join(REPO, *p)))
        assert len(fs) == 4, (arm, fs)
        out[arm] = [json.load(open(f, encoding="utf-8"))["init_head_from"].split("/runs/", 1)[1] for f in fs]
    for k, f in enumerate(FOLDS):
        assert out["freematch"][k] == f"bakeoff_freematch_5fold/fold{f}/checkpoints/step_8000.pt"
        assert out["coral"][k] == f"bakeoff_coral/fold{f}/checkpoints/step_8000.pt"
    for arm in ("dann", "lambda0", "bnm"):
        assert set(out[arm]) == {"bakeoff_baseline_A/checkpoints/step_8000.pt"}, out[arm]
    return out


def a2():
    heads = init_heads()
    starts = {}
    for key, run in (("freematch", "bakeoff_freematch_5fold"), ("coral", "bakeoff_coral")):
        s = C.round1_summary(run, folds=FOLDS)
        per = [float(s["per_fold"][f]["tpr"]) for f in FOLDS]
        starts[key] = {"run": run, "per_fold": dict(zip(map(str, FOLDS), per)),
                       "fold_mean": float(np.mean(per))}
    u, lam = fm("freematch", 1500), fm("lambda0", 4000)
    unfreeze = u - starts["freematch"]["fold_mean"]
    frozen_part = starts["freematch"]["fold_mean"] - lam
    checks = {
        "unfrozen_freematch_1500": u, "lambda0_4000": lam,
        "unfrozen_minus_start": unfreeze, "table5_last_row_target": 0.1049,
        "unfrozen_minus_start_equals_table5_to_4dp": round(unfreeze, 4) == 0.1049,
        "freematch_minus_lambda0": u - lam,
        "start_minus_lambda0": frozen_part,
        "identity_holds": abs((u - lam) - (frozen_part + unfreeze)) < 1e-12,
        "freematch_minus_lambda0_equals_0_3325_to_4dp": round(u - lam, 4) == 0.3325,
        "coral_start_minus_lambda0": starts["coral"]["fold_mean"] - lam,
    }
    assert checks["unfrozen_minus_start_equals_table5_to_4dp"]
    assert checks["freematch_minus_lambda0_equals_0_3325_to_4dp"]
    return {"init_head_from": heads, "starts": starts, "checks": checks}


TABLE5_OBJECTIVE_ROWS = [("bnm", 2000, "coral", 2000), ("bnm", 4000, "dann", 4000),
                         ("bnm", 1500, "freematch", 1500), ("dann", 4000, "coral", 2000),
                         ("dann", 4000, "freematch", 1500), ("dann", 2000, "coral", 2000)]


def a3():
    a1, three, bnm, nb = RV.a1_sets()
    sig = {k: float(a1["sigma_by_arm"][k]["folds1_4_mean_detrended_b_linfit_resid"])
           for k in a1["sigma_by_arm"]}
    t975 = float(stats.t.ppf(0.975, 3))
    rows, seen = [], set()
    for group, d in (("figure4_three", three), ("figure4_bnm", bnm)):
        for k, g in d.items():
            pair = k.split("|")[0]
            steps = tuple(g["steps"])
            if (pair, steps) in seen:
                continue
            seen.add((pair, steps))
            a, b = pair.split("-")
            rows.append([group, a, steps[0], b, steps[1], float(nb(g))])
    have = {(r[1], r[2], r[3], r[4]) for r in rows}
    for a, ra, b, rb in TABLE5_OBJECTIVE_ROWS:
        if (a, ra, b, rb) not in have and (b, rb, a, ra) not in have:
            rows.append(["table5_only", a, ra, b, rb, math.sqrt(sig[a] ** 2 + sig[b] ** 2) / 2])
    out = []
    for group, a, ra, b, rb, ind in rows:
        A, B = C.per_fold(a, "tpr", ra), C.per_fold(b, "tpr", rb)
        d = np.array([A[f] - B[f] for f in FOLDS])
        m = float(d.mean())
        se = float(d.std(ddof=1) / 2)
        out.append({"group": group, "pair": f"{a}@{ra} - {b}@{rb}",
                    "in_table5": (a, ra, b, rb) in TABLE5_OBJECTIVE_ROWS,
                    "a_independence": ind, "b_paired_se": se, "ratio_a_over_b": ind / se,
                    "diff": m, "ci": [m - t975 * se, m + t975 * se], "half_width": t975 * se,
                    "signs_agreeing": int(max((d > 0).sum(), (d < 0).sum()))})
    return {"t975_df3": t975, "sigma_detrended_b": sig, "rows": out,
            "n_pairs_ratio_above_1": sum(r["ratio_a_over_b"] > 1 for r in out),
            "a_exceeds_half_width_anywhere": any(r["a_independence"] > r["half_width"] for r in out)}


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    meta = {"status": "post_hoc_exploratory", "pool": C.POOL,
            "producing_script": "audit/rewrite_v5/scripts/a2_a3_starts_noise.py"}
    r2 = dict(meta, **a2())
    r3 = dict(meta, **a3())
    json.dump(r2, open(os.path.join(OUTDIR, "A2_head_starts.json"), "w", encoding="utf-8"), indent=1)
    json.dump(r3, open(os.path.join(OUTDIR, "A3_independence.json"), "w", encoding="utf-8"), indent=1)
    s = r2["starts"]
    print(f"  A2 FreeMatch start {s['freematch']['fold_mean']:.4f}  CORAL start {s['coral']['fold_mean']:.4f}")
    for k, v in r2["checks"].items():
        print(f"     {k}: {v}")
    for r in r3["rows"]:
        print(f"  A3 {r['group']:13} {r['pair']:30} a={r['a_independence']:.4f} b={r['b_paired_se']:.4f} "
              f"a/b={r['ratio_a_over_b']:.2f}  {r['diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] "
              f"{r['signs_agreeing']}/4")
    print(f"  pairs with a/b > 1: {r3['n_pairs_ratio_above_1']}; a exceeds a half-width anywhere: "
          f"{r3['a_exceeds_half_width_anywhere']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
