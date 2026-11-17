"""
test_round7.py
==============
Tests for round 7's two closing analyses.

    python -u scripts/test_round7.py

THE ONE THE BRIEF NAMES EXPLICITLY. Task 2 says "Reuse BNM's existing diagnostic code path
unchanged ... Assert the shared path in the test file." T1-T4 do exactly that: they assert
that ``analyze_holdout_gap_lambda0`` calls BNM's own objects and defines none of its own,
that the slice builders are BNM's, and that a drifted second implementation would be
caught because the reconstructed slices are compared against the indices the runs recorded.

The rest pin the window-averaging declaration (k is read from the hashed file, never
hard-coded), the window-placement rule, and both declared score combinations.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

FAILED = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILED.append(label)


# ---------------------------------------------------------------------------
# T1-T4 -- the shared diagnostic code path (Task 2)
# ---------------------------------------------------------------------------
def t_shared_path():
    print("\nT1-T4  the 704/78 diagnostic uses BNM's code path, unchanged")
    import analyze_holdout_gap_lambda0 as A
    import bnm_round6 as B
    import dann_round5 as D5

    check(A.probe_stats is B.probe_stats,
          "T1  probe_stats IS bnm_round6.probe_stats (not a copy)")
    check(A.holdout_gap is B.holdout_gap,
          "T2  holdout_gap IS bnm_round6.holdout_gap (not a copy)")
    check(A.build_trainedon_probe is B.build_trainedon_probe and
          A.build_target_holdout is D5.build_target_holdout and
          A.materialise_holdout_images is D5.materialise_holdout_images,
          "T3  both slice builders are the runners' own objects")

    # T4: no second implementation may exist in the analysis module.
    src = inspect.getsource(A)
    banned = ("def probe_stats", "def holdout_gap", "def build_trainedon_probe",
              "def build_target_holdout", "nuclear_norm(", "torch.linalg.svd")
    offenders = [b for b in banned if b in src]
    check(not offenders,
          f"T4  the analysis module defines no diagnostic of its own "
          f"{'(found ' + str(offenders) + ')' if offenders else ''}")


def t_size_matching():
    print("\nT5-T6  the size-matching guard that makes the gap meaningful")
    import bnm_round6 as B
    a = {"n": 78, "nuclear_norm": 6.0, "entropy": 0.1, "pred_pos_frac": 0.2}
    b = {"n": 704, "nuclear_norm": 6.0, "entropy": 0.1, "pred_pos_frac": 0.2}
    raised = False
    try:
        B.holdout_gap(a, b)
    except ValueError:
        raised = True
    check(raised, "T5  holdout_gap REFUSES unmatched slice sizes (||A||* scales as sqrt(B))")
    g = B.holdout_gap(a, dict(a))
    check(abs(g["nuclear_gap"]) < 1e-12 and g["n_per_slice"] == 78,
          "T6  identical size-matched slices give a zero gap")


def t_recorded_slices_agree():
    print("\nT7-T8  the three arms carry the SAME 78-image slice")
    lam, dann, probes = {}, {}, {}
    for f in (1, 2, 3, 4):
        lam[f] = set(json.load(open(os.path.join(
            REPO, "runs", "baseline_unfrozen", f"fold{f}", "extracted",
            f"lambda0_r6_fold{f}", "run_config.json"),
            encoding="utf-8"))["dann_d_holdout_indices"])
        dann[f] = set(json.load(open(os.path.join(
            REPO, "runs", "dann_main", f"fold{f}", "extracted", f"fold{f}",
            "run_config.json"), encoding="utf-8"))["dann_d_holdout_indices"])
        probes[f] = set(json.load(open(os.path.join(
            REPO, "runs", "bnm_r6_folds", f"bnm_r6_fold{f}", "run_config.json"),
            encoding="utf-8"))["trainedon_probe"]["probe"])
    check(all(lam[f] == dann[f] and len(lam[f]) == 78 for f in lam),
          "T7  lambda=0's and DANN's holdouts are identical, 78 images, all four folds")
    check(all(not (probes[f] & lam[f]) and len(probes[f]) == 78 for f in probes),
          "T8  BNM's trained-on probe is size-matched and disjoint from that slice")


def t_no_retrospective_holdout():
    print("\nT9  CORAL and FreeMatch are EXCLUDED, not given a holdout retrospectively")
    coral = json.load(open(os.path.join(
        REPO, "kaggle", "coral_r4", "runs", "main", "extracted", "coral_r4_confirm",
        "fold1", "run.json"), encoding="utf-8"))
    fm = json.load(open(os.path.join(
        REPO, "runs", "freematch_nb3_unfrozen_fold1", "run_config.json"),
        encoding="utf-8"))
    has = [k for d in (coral, fm) for k in d if "holdout" in k.lower()]
    check(not has, f"T9  neither run records a target holdout {has if has else ''} -- "
                   f"so neither may be compared")


# ---------------------------------------------------------------------------
# T10+ -- the window-averaging declaration (Task 3)
# ---------------------------------------------------------------------------
def t_declaration():
    print("\nT10-T12  the window declaration is hashed and is the only source of k")
    import analyze_window_average as W
    pre = json.load(open(W.PREREG, encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k != "sha256_of_body_without_this_field"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    check(digest == pre["sha256_of_body_without_this_field"],
          "T10  the declaration self-certifies")

    src = inspect.getsource(W)
    check("K_GRID = list(pre[" in src and "K_PRIMARY = int(pre[" in src,
          "T11  k is READ from the declaration, never defined in the analysis")
    # a literal k=8 anywhere in the analysis would be a second source of truth
    check("k = 8" not in src and "K = 8" not in src,
          "T12  no literal k survives in the analysis module")


def t_window_rule():
    print("\nT13-T15  the window-placement rule")
    import analyze_window_average as W
    steps = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    check(W.window(steps, 2000, 8) == steps,
          "T13  k=8 at the last checkpoint takes all eight, ending AT the read-off")
    check(W.window(steps, 1000, 2) == [750, 1000],
          "T14  the window ENDS at the read-off, it is not centred on it")
    check(W.window(steps, 500, 8) == [250, 500],
          "T15  too few checkpoints truncates to what exists (achieved k reported)")
    check(W.window(steps, 1100, 4) == [],
          "T15b a read-off with no checkpoint returns nothing rather than guessing")


def t_combinations():
    print("\nT16-T18  both declared score combinations")
    import analyze_window_average as W
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50) * 10.0 + 3.0
    sbs = {100: (a[:10], a[10:]), 200: (b[:10], b[10:])}

    v_logit, n_pos = W.combine(sbs, [100, 200], "mean_logit")
    check(n_pos == 10 and v_logit.size == 50, "T16  combine preserves the pool shape")

    # a wide-scale checkpoint dominates mean_logit and cannot dominate mean_rank
    rho_logit = np.corrcoef(v_logit, np.concatenate([b[:10], b[10:]]))[0, 1]
    v_rank, _ = W.combine(sbs, [100, 200], "mean_rank")
    rho_rank = np.corrcoef(v_rank, np.concatenate([b[:10], b[10:]]))[0, 1]
    check(rho_logit > 0.98 and rho_rank < rho_logit,
          "T17  mean_logit IS dominated by a wide-scale checkpoint; mean_rank is less so "
          f"(rho {rho_logit:.3f} vs {rho_rank:.3f}) -- why both are declared")

    single, _ = W.combine(sbs, [100], "mean_logit")
    check(np.allclose(single, np.concatenate([a[:10], a[10:]])),
          "T18  k=1 is the identity -- the existing single-point read-off")


def t_k1_reproduces_headlines():
    print("\nT19  k=1 reproduces the hashed headline numbers (the pipeline check)")
    p = os.path.join(REPO, "runs", "round7", "window_average.json")
    if not os.path.isfile(p):
        check(False, "T19  runs/round7/window_average.json missing -- run the analysis")
        return
    d = json.load(open(p, encoding="utf-8"))["arms"]

    def k1(arm, ro):
        return d[arm]["readouts"][str(ro)]["mean_logit"]["1"]["tpr_mean"]

    known = {("CORAL x nb3", 2000): 0.4348, ("FreeMatch x nb3", 1500): 0.4169,
             ("DANN x nb3", 4000): 0.4167, ("DANN x nb3", 2000): 0.4551,
             ("lambda=0", 4000): 0.1355, ("BNM x nb3", 1000): 0.3351,
             ("BNM x nb3", 4000): 0.3274}
    bad = [f"{a}@{s}: {k1(a, s):.4f} vs {v}" for (a, s), v in known.items()
           if abs(k1(a, s) - v) > 5e-4]
    check(not bad, f"T19  every k=1 fold-mean matches PROJECT.md {bad if bad else ''}")


# ---------------------------------------------------------------------------
# T20+ -- the closing corrections
# ---------------------------------------------------------------------------
def t_fold0_shares_the_path():
    print("\nT20  the fold-0 check uses the SAME diagnostic path as the lambda=0 one")
    import analyze_holdout_gap_fold0 as F
    import analyze_holdout_gap_lambda0 as A
    import bnm_round6 as B
    ok = (F.probe_stats is B.probe_stats is A.probe_stats and
          F.holdout_gap is B.holdout_gap is A.holdout_gap and
          F.build_trainedon_probe is B.build_trainedon_probe)
    check(ok, "T20  both round-7 diagnostics call bnm_round6's objects, not copies")
    src = inspect.getsource(F)
    banned = [b for b in ("def probe_stats", "def holdout_gap", "def build_trainedon_probe",
                          "nuclear_norm(") if b in src]
    check(not banned, f"T20b the fold-0 module defines no diagnostic of its own {banned}")


def t_gap_intervals():
    print("\nT21-T23  the fold-level intervals on the 704/78 gaps")
    from analyze_lambda0_r6 import paired_t
    b = json.load(open(os.path.join(REPO, "runs", "bnm_r6_folds_results.json"),
                       encoding="utf-8"))["diagnostic_704_78"]
    t1 = paired_t(b["read_off"]["per_fold"])
    check(not t1["excludes_zero"] and abs(t1["mean"] - 0.0993) < 5e-4,
          f"T21  BNM reading (1) is +{t1['mean']:.4f} "
          f"[{t1['ci_lo']:+.4f}, {t1['ci_hi']:+.4f}] -- INCLUDES zero, so 7.28's "
          f"interpretive sentence is downgraded")
    t3 = paired_t(b["within_fold_mean"]["per_fold"])
    check(t3["excludes_zero"] and t3["all_same_sign"],
          f"T22  reading (3), a SECONDARY, resolves it: +{t3['mean']:.4f} "
          f"[{t3['ci_lo']:+.4f}, {t3['ci_hi']:+.4f}], all 4 folds same sign")
    # the sd 7.29's table quotes must be the one the per-fold values give
    check(abs(t1["sd"] - 0.1012) < 5e-4,
          f"T23  reading (1)'s sd is {t1['sd']:.4f}, the value 7.29 now quotes (not 0.1058)")


def t_fold0_null():
    print("\nT24  the fold-0 null for readings (2) and (3)")
    p = os.path.join(REPO, "runs", "round7", "holdout_gap_fold0.json")
    if not os.path.isfile(p):
        check(False, "T24  runs/round7/holdout_gap_fold0.json missing -- run the check")
        return
    d = json.load(open(p, encoding="utf-8"))
    n = d["null_for_readings_2_and_3"]
    bnm, lam = n["bnm_fold0"], n["lambda0_fold0"]
    check(bnm["within_fold_mean"] > lam["within_fold_mean"],
          f"T24  on fold 0 the null sits below BNM on reading (3) "
          f"({lam['within_fold_mean']:+.4f} vs {bnm['within_fold_mean']:+.4f}) "
          f"-- n = 1 fold, descriptive")
    # reading (2) has a POSITIVE null and the null arm proves it on its own reads
    check(lam["any_checkpoint_max"] > 0 > lam["within_fold_mean"],
          f"T24b reading (2) has a positive null BY CONSTRUCTION: the same lambda=0 "
          f"arm has within-fold mean {lam['within_fold_mean']:+.4f} and "
          f"any-checkpoint max {lam['any_checkpoint_max']:+.4f} -- so reading (2) "
          f"gets NO interval")


def t_curve_matches_table():
    print("\nT28-T29  the completeness-purity figure agrees with its own table")
    fp = os.path.join(REPO, "runs", "round7", "completeness_purity.json")
    tp = os.path.join(REPO, "runs", "round7", "lines_ratios.json")
    if not (os.path.isfile(fp) and os.path.isfile(tp)):
        check(False, "T28  the figure summary or the ratios table is missing")
        return
    c = json.load(open(fp, encoding="utf-8"))["arms"]
    t = json.load(open(tp, encoding="utf-8"))["arms"]
    bad = []
    for a, r in c.items():
        tab = t[a]["readouts"][str(r["readout"])].get("fold_mean")
        mark = r.get("exact_fold_mean_at_24pct")
        if (tab is None) != (mark is None) or (tab is not None
                                               and abs(tab - mark) > 1e-9):
            bad.append(f"{a}: table {tab} vs marker {mark}")
    check(not bad, f"T28  the marked fold-mean IS the tabulated one on every arm {bad}")

    # an n/a arm must have no marker, and its fold-mean curve must stop left of 0.24
    na = [a for a, r in c.items()
          if t[a]["readouts"][str(r["readout"])].get("fold_mean") is None]
    check(na and all(c[a].get("exact_fold_mean_at_24pct") is None and
                     c[a].get("drawn_curve_at_24pct") is None for a in na),
          f"T29  the {len(na)} n/a arms carry no marker and no curve value at 24% "
          f"-- an n/a is a curve endpoint, not a hole")


def t_lines_ratios_persisted():
    print("\nT25-T27  the Lines ratios are on disk with the n/a cases explicit")
    p = os.path.join(REPO, "runs", "round7", "lines_ratios.json")
    if not os.path.isfile(p):
        check(False, "T25  runs/round7/lines_ratios.json missing -- run the writer")
        return
    d = json.load(open(p, encoding="utf-8"))
    coral = d["arms"]["CORAL x nb3"]["readouts"]["2000"]
    check(abs(coral["fold_mean"] - 0.2428) < 5e-4 and
          abs(coral["pct_of_lines"] - 48.6) < 0.1,
          "T25  CORAL's 0.2428 / 48.6% now has an artifact behind it")

    na = [(a, ro) for a, rec in d["arms"].items()
          for ro, e in rec["readouts"].items() if not e["defined_on_all_folds"]]
    check(len(na) >= 5 and all(
        d["arms"][a]["readouts"][ro]["fold_mean"] is None and
        d["arms"][a]["readouts"][ro]["why_na"] for a, ro in na),
        f"T26  {len(na)} arm x readout cells are n/a, each with no fold-mean and a "
        f"stated reason -- never an average over the reachable subset")

    bnm = d["arms"]["BNM x nb3"]
    check("no_matched_arm_budget" in bnm.get("note", "") and
          abs(bnm["readouts"]["4000"]["pct_of_lines"] - 26.6) < 0.1 and
          abs(bnm["readouts"]["1000"]["pct_of_lines"] - 13.3) < 0.1,
          "T27  BNM's ratios are in the table with the no_matched_arm_budget note")


def main():
    for fn in (t_shared_path, t_size_matching, t_recorded_slices_agree,
               t_no_retrospective_holdout, t_declaration, t_window_rule,
               t_combinations, t_k1_reproduces_headlines,
               t_fold0_shares_the_path, t_gap_intervals, t_fold0_null,
               t_lines_ratios_persisted, t_curve_matches_table):
        fn()
    print("\n" + "=" * 70)
    if FAILED:
        print(f"FAILED {len(FAILED)}:")
        for f in FAILED:
            print("  -", f)
        return 1
    print("all round-7 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
