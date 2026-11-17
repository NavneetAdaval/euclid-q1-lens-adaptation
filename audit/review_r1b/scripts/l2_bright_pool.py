"""
l2_bright_pool.py -- review round 1b, L2: the bright-only negative pool (sensitivity readout).
POST HOC AND EXPLORATORY.

    python -B audit/review_r1b/scripts/l2_bright_pool.py

Pool: the D1 test negatives with AB mag = 23.9 - 2.5 log10(MER flux_vis_psf / uJy) <= 22.5,
i.e. NOT "fainter than 22.5" in the sense of §3 / make_m_prose.test_faint_fraction /
round-1 A2. Positives are unchanged. Every metric is scripts/metrics.metric_suite; every
interval is analyze_lambda0_r6.paired_t (df = n - 1), both via paper/scripts/_common.
The D1 comparison values come from _common.suite / _common.contrast on the D1 pool.

Writes audit/review_r1b/out/L2_bright_pool.json.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "review_r1", "scripts"))
import rr_common as RC                                   # noqa: E402  (round-1 helpers)

C = RC.C
SCRIPT = os.path.abspath(__file__)
OUT = os.path.join(os.path.dirname(HERE), "out")
LIMIT = 22.5
FOLDS = RC.CONF_FOLDS

# (label, arm_a, step_a, arm_b, step_b, form, source)
CONTRASTS = [
    ("DANN - lambda=0", "dann", 4000, "lambda0", 4000, "pre-registered = budget-matched",
     "runs/lambda0_r6_prereg.json reporting.paired_contrasts[1]; budget 4000"),
    ("CORAL - lambda=0", "coral", 2000, "lambda0", 4000, "pre-registered (lambda=0 at 4000)",
     "runs/lambda0_r6_prereg.json reporting.paired_contrasts[2]; round-1 F1"),
    ("CORAL - lambda=0", "coral", 2000, "lambda0", 2000, "budget-matched",
     "lambda0 sensitivity readout 2000"),
    ("FreeMatch - lambda=0", "freematch", 1500, "lambda0", 4000,
     "pre-registered (lambda=0 at 4000)",
     "runs/lambda0_r6_prereg.json reporting.paired_contrasts[3]; round-1 F1"),
    ("FreeMatch - lambda=0", "freematch", 1500, "lambda0", 1500, "budget-matched",
     "lambda0 sensitivity readout 1500"),
    ("BNM - lambda=0", "bnm", 4000, "lambda0", 4000, "pre-registered = budget-matched",
     "runs/bnm_r6_prereg.json budget.why_4000 (not one of lambda0_r6_prereg's four)"),
    ("BNM - CORAL", "bnm", 4000, "coral", 2000, "declared budgets", "_common.CONTRAST_ROWS"),
    ("BNM - CORAL", "bnm", 2000, "coral", 2000, "budget-matched", "_common.CONTRAST_ROWS[7]"),
]
NEEDED = sorted({(a, s) for _, a, s, b, t, *_ in CONTRASTS} |
                {(b, t) for _, a, s, b, t, *_ in CONTRASTS} | set(RC.HEADLINE))


def fmean_sd(vals):
    if any(not math.isfinite(v) for v in vals):
        return float("nan"), float("nan")
    return st.mean(vals), st.stdev(vals)


def main():
    RC.OUT = OUT
    I = RC.Inputs()
    S = C.pool_sets()
    mer = RC.mer_table(I)
    I.add(C.ART["splits"], C.ART["pool_rule"])

    # ---------------- the pool ----------------
    sp = json.load(open(C.ART["splits"], encoding="utf-8"))
    tn = sp["test_negatives_shared"]
    locs = [f"random:{a}:{b}" for a, b in tn["locations"]]
    keep = C.pool_keep_neg(locs)
    d1_locs = [l for l, k in zip(locs, keep) if k]
    d1_ids = np.array([S["loc2id"][l] for l in d1_locs], np.int64)
    mag = RC.ab_mag(mer["flux_vis_psf"].reindex(d1_ids).to_numpy(float))
    assert np.isfinite(mag).all()
    bright_of = {l: bool(m <= LIMIT) for l, m in zip(d1_locs, mag)}
    n_bright = int(sum(bright_of.values()))
    import make_m_prose as MP
    with C.pool("D1"):
        faint_frac = MP.test_faint_fraction()
    assert abs((1 - n_bright / len(d1_locs)) - faint_frac) < 1e-12
    pool = {"definition": "D1 test negatives with VIS PSF AB mag <= 22.5 (complement of "
                          "make_m_prose.test_faint_fraction's 'fainter than 22.5')",
            "n_D1": len(d1_locs), "n_bright": n_bright, "n_removed_faint": len(d1_locs) - n_bright,
            "faint_fraction_D1_reproduces_make_m_prose": faint_frac,
            "n_negatives_per_fpr_1e-3": n_bright * 1e-3}

    # ---------------- per (arm, step, fold) ----------------
    suites = {}
    with C.pool("D1"):
        for arm, step in NEEDED:
            for f in FOLDS:
                v = RC.load_vec(RC.step_paths(arm, f)[step], f, arm, "D1", I)
                d1 = RC.suite_of(v)
                assert d1["tpr"] == float(C.suite(arm, f, step)["tpr"])
                bm = np.array([bright_of[l] for l in v["neg_locs"]])
                vb = dict(pos=v["pos"], neg=v["neg"][bm], pos_rows=v["pos_rows"],
                          neg_locs=[l for l, k in zip(v["neg_locs"], bm) if k])
                assert vb["neg"].size == n_bright
                b = RC.suite_of(vb)
                suites[(arm, step, f)] = {"D1": d1, "bright": b,
                                          "threshold_bright": RC.threshold_of(vb),
                                          "n_pos": int(v["pos"].size)}

    readouts = {}
    for arm, step in RC.HEADLINE:
        rec = {}
        for metric in ("tpr", "comp_at_purity", "auroc"):
            for p in ("bright", "D1"):
                vals = [float(suites[(arm, step, f)][p][metric]) for f in FOLDS]
                m, s = fmean_sd(vals)
                rec[f"{metric}_{p}"] = {"per_fold": dict(zip(map(str, FOLDS), vals)),
                                        "mean": m, "sd": s,
                                        "n_undefined_folds": sum(not math.isfinite(x)
                                                                 for x in vals)}
            rec[f"{metric}_change_mean"] = (rec[f"{metric}_bright"]["mean"] -
                                            rec[f"{metric}_D1"]["mean"])
        for p in ("bright", "D1"):
            cm = rec[f"comp_at_purity_{p}"]["mean"]
            rec[f"lines_ratio_{p}"] = (cm / C.LINES_COMPLETENESS
                                       if math.isfinite(cm) else None)
        rec["thresholds_bright"] = {str(f): suites[(arm, step, f)]["threshold_bright"]
                                    for f in FOLDS}
        readouts[RC.tag(arm, step)] = rec

    contrasts = []
    for label, a, sa, b, sb, form, src in CONTRASTS:
        out = {"contrast": label, "steps": [sa, sb], "form": form, "source": src}
        for p in ("bright", "D1"):
            diffs = [float(suites[(a, sa, f)][p]["tpr"]) - float(suites[(b, sb, f)][p]["tpr"])
                     for f in FOLDS]
            t = C.paired_t(diffs)
            sign = 1.0 if t["mean"] >= 0 else -1.0
            t["n_same_sign"] = sum(1 for d in diffs if d * sign > 0)
            t["diffs"] = dict(zip(map(str, FOLDS), diffs))
            out[p] = t
        with C.pool("D1"):
            ref = C.contrast(a, sa, b, sb)
        assert abs(ref["mean"] - out["D1"]["mean"]) < 1e-15
        inc = {p: not out[p]["excludes_zero"] for p in ("bright", "D1")}
        out["includes_zero"] = inc
        out["zero_inclusion_same"] = inc["bright"] == inc["D1"]
        out["change_mean"] = out["bright"]["mean"] - out["D1"]["mean"]
        contrasts.append(out)

    RC.write_json("L2_bright_pool.json", {
        "pool": pool, "readouts": readouts, "contrasts": contrasts,
        "lines_completeness_reference": C.LINES_COMPLETENESS,
        "n_a_rule": "an arm with any fold undefined gets no fold-mean and no ratio "
                    "(_common.fold_mean's rule)",
        "interval": "paired t on 4 differences, df = 3 (analyze_lambda0_r6.paired_t); no "
                    "multiplicity correction",
    }, script=SCRIPT, inputs=I, pool="D1 restricted to VIS PSF mag <= 22.5 (and D1 for "
                                      "comparison)", folds=FOLDS)


if __name__ == "__main__":
    main()
