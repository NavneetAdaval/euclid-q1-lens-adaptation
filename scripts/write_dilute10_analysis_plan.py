"""
write_dilute10_analysis_plan.py
===============================
Writes ``runs/dilute10_analysis/plan.json`` -- the ANALYSIS PLAN of the 1:10 diluted-target
experiment (brief_dilute10_analysis; PROJECTv7.md 7.53), hashed BEFORE any value of the
experiment is computed.

It computes NO metric and loads NO score vector. Every logits, ids and score file is hashed
as bytes. The only numbers it derives from run files are optimiser bookkeeping (GradScaler
skip counts, for part 3's no-verdict rule), which are not test metrics and which PROJECTv7
7.51-7.52 already report.

What it fixes (brief section 1):
  * the diluted inputs at each arm's read-off, the frozen-stage heads' persisted logits and
    gate files, each with path and sha256;
  * the comparators -- balanced arms and lambda=0 at 4000/2000/1500 -- at the paths
    scripts/analyze_window_average.ARMS reads (asserted here), and round-1 frozen CORAL and
    FreeMatch step-8000 score dumps, each with path and sha256;
  * the pool (D1 rule body digest, rule file, removal list) and the code (metric, interval,
    pool-filter and bootstrap modules) by sha256;
  * part 1's contrast list, verdict rule, qualifier and predictions, COPIED from the part-1
    file (never retyped), part 3's no-verdict rule with each run's applied fraction;
  * the prediction-scoring rubric, the descriptive items, the sensitivities and the
    stop rules, declared before any value exists.

    python -u scripts/write_dilute10_analysis_plan.py --out runs/dilute10_analysis/plan.json \
        --brief C:/Users/krris/Downloads/brief_dilute10_analysis.md
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from integrity import verify as _verify_integrity            # noqa: E402

FOLDS = (1, 2, 3, 4)
PART_DIGESTS = {
    1: "bd537a025e81833e73713649c9eed68fd6a0fd3152ee29240f9a333f7c649d85",
    2: "b8e09197a204be835fa089c858b08c7eb3e7887bfa19afd296d544d04f394503",
    3: "1b583ea2ae5056edb410040c32613d9388e5a994bf90fc366fc93d293efb596a",
}
D1_RULE_BODY = "b83fa019abba5238ccfca593a3130b54be574a47359b4e2818b274efd3fd0e4b"
D1_REMOVAL_FILE = "3d3deafbc57a24242c54f63750c252d4b57377c4287b57cd9fb0518fb6ba838f"

# applied-update counts PROJECTv7 7.51 (CORAL nb3) and 7.52 (FreeMatch nb3) report; the
# writer re-derives them from the run files and refuses on any difference
RECORDED_APPLIED = {"coral_nb3": {1: 1991, 2: 1990, 3: 1991, 4: 1991},
                    "freematch_nb3": {1: 1498, 2: 1498, 3: 1497, 4: 1497}}
BUDGET = {"dann": 4000, "bnm": 4000, "coral_nb3": 2000, "freematch_nb3": 1500}


def R(*p):
    return os.path.join(REPO, *p)


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def self_cert(d):
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


def entry(*p):
    path = R(*p)
    if not os.path.isfile(path):
        raise SystemExit(f"REFUSING: planned input missing: {rel(path)}")
    return {"path": rel(path), "sha256": sha(path)}


# ------------------------------------------------------------------ the inputs
DILUTED = {
    # arm: (read-off step, logits dir relative to the fold folder)
    "DANN x nb3": ("dann", 4000, "logits/dann_nb3_fold{k}"),
    "BNM x nb3": ("bnm", 4000, "logits/bnm_nb3_fold{k}"),
    "CORAL x nb3": ("coral_nb3", 2000, "logits/g16_2000_fold{k}"),
    "FreeMatch x nb3": ("freematch_nb3", 1500, "logits"),
}
BALANCED_DIR = {      # the directories analyze_window_average.ARMS scans (asserted below)
    "DANN x nb3": "runs/dann_main/fold{k}/extracted/fold{k}/logits/dann_nb3_fold{k}",
    "BNM x nb3": "runs/bnm_r6_folds/bnm_r6_fold{k}/logits/bnm_nb3_fold{k}",
    "CORAL x nb3": ("kaggle/coral_r4/runs/main/extracted/coral_r4_confirm/fold{k}/logits/"
                    "g16_2000_fold{k}"),
    "FreeMatch x nb3": "runs/freematch_nb3_unfrozen_fold{k}/logits",
    "lambda=0": ("runs/baseline_unfrozen/fold{k}/extracted/lambda0_r6_fold{k}/logits/"
                 "dann_nb3_lambda0_fold{k}"),
}
BALANCED_STEPS = {"DANN x nb3": [4000], "BNM x nb3": [4000], "CORAL x nb3": [2000],
                  "FreeMatch x nb3": [1500], "lambda=0": [4000, 2000, 1500]}


def _ids_beside(d):
    for cand in (d, os.path.dirname(d)):
        p = os.path.join(cand, "ids.npz")
        if os.path.isfile(p):
            return p
    raise SystemExit(f"REFUSING: no ids.npz beside {rel(d)}")


def _assert_arms_paths():
    """The planned comparator directories ARE the ones the paper's loaders read."""
    import analyze_window_average as AWA
    import inspect
    for arm, tmpl in BALANCED_DIR.items():
        src = inspect.getsource(AWA.ARMS[arm]["load"])
        for part in tmpl.replace("{k}", "").split("/"):
            if part and part not in src:
                raise SystemExit(f"REFUSING: {arm}: planned path part {part!r} is not in "
                                 f"analyze_window_average's loader")


def inputs(part1):
    out = {"diluted": {}, "diluted_frozen": {}, "balanced": {}, "round1_frozen": {},
           "paper_reference_tables": {}, "descriptive_files": {}}
    for arm, (key, step, sub) in DILUTED.items():
        out["diluted"][arm] = {"read_at": step, "folds": {}}
        for k in FOLDS:
            d = R("runs", "dilute10", key, f"fold{k}", sub.format(k=k))
            out["diluted"][arm]["folds"][str(k)] = {
                "logits": entry(d, f"step_{step}.npz"),
                "ids": entry(_ids_beside(d))}
    for stage in ("coral_frozen", "freematch_frozen"):
        out["diluted_frozen"][stage] = {"read_at": 8000, "folds": {}}
        for k in FOLDS:
            f = R("runs", "dilute10", stage, f"fold{k}")
            out["diluted_frozen"][stage]["folds"][str(k)] = {
                "logits": entry(f, "logits", "step_8000.npz"),
                "ids": entry(f, "logits", "ids.npz"),
                "gate_values": entry(f, "gate_values.json")}
    _assert_arms_paths()
    p1art = part1["artifacts"]
    for arm, tmpl in BALANCED_DIR.items():
        out["balanced"][arm] = {"steps": BALANCED_STEPS[arm], "folds": {}}
        for k in FOLDS:
            d = R(tmpl.format(k=k))
            rec = {f"step_{s}": entry(d, f"step_{s}.npz") for s in BALANCED_STEPS[arm]}
            rec["ids"] = entry(_ids_beside(d))
            # every file part 1 names must be this file, at part 1's hash
            for name, e in rec.items():
                want = p1art[arm][str(k)].get(name)
                if want is None or want["path"] != e["path"] or want["sha256"] != e["sha256"]:
                    raise SystemExit(f"REFUSING: {arm} fold {k} {name} differs from part 1: "
                                     f"{e} vs {want}")
            out["balanced"][arm]["folds"][str(k)] = rec
    r1 = part1["artifacts"]["round1_frozen_scores (descriptive comparison, section 6)"]
    for run, tag in (("bakeoff_coral", "CORAL"), ("bakeoff_freematch_5fold", "FreeMatch")):
        out["round1_frozen"][tag] = {"run_dir": f"runs/{run}", "read_at": 8000, "folds": {}}
        for k in FOLDS:
            e = entry(r1[run][str(k)]["path"])
            if e["sha256"] != r1[run][str(k)]["sha256"]:
                raise SystemExit(f"REFUSING: round-1 {run} fold {k} differs from part 1")
            out["round1_frozen"][tag]["folds"][str(k)] = e
    out["paper_reference_tables"] = {
        "per_fold_tpr (Table 5)": entry("paper", "tables", "T5_perfold_v2.tex"),
        "contrasts_four_decimals (Table 4)": entry("paper", "tables", "T2v2_contrasts4.tex"),
        "frozen_decomposition_macros": entry("paper", "tables", "M_rewrite5.tex"),
    }
    desc = {}
    for key in ("dann", "bnm", "coral_nb3", "freematch_nb3", "coral_frozen", "freematch_frozen"):
        for k in FOLDS:
            f = R("runs", "dilute10", key, f"fold{k}")
            files = ["target_lens_fraction_summary.json", "dilute10_post_run_check.json"]
            files += {"dann": ["steps.csv", "run.json"], "bnm": ["trace.csv", "run.json"],
                      "coral_nb3": ["steps.csv", "run.json",
                                    "dilute10_feature_norm_advisory.json"],
                      "freematch_nb3": ["diagnostics.csv", "results.json"],
                      "coral_frozen": ["run_config.json"],
                      "freematch_frozen": ["run_config.json"]}[key]
            desc[f"{key}/fold{k}"] = {n: entry(f, n) for n in files}
    for k in FOLDS:   # balanced BNM trajectory, context for the predicted-positive item
        desc[f"bnm_balanced_context/fold{k}"] = {
            "trace.csv": entry("runs", "bnm_r6_folds", f"bnm_r6_fold{k}", "trace.csv")}
    out["descriptive_files"] = desc
    return out


# ------------------------------------------------------------------ part 3 bookkeeping
def applied_fractions():
    """GradScaler skips and applied updates of the two nb3 arms part 3 binds."""
    out = {}
    for k in FOLDS:
        rows = list(csv.DictReader(open(R("runs", "dilute10", "coral_nb3", f"fold{k}",
                                          "steps.csv"), encoding="utf-8")))
        assert len(rows) == 2000 and int(rows[-1]["step"]) == 2000
        skips = sum(int(float(r["step_skipped"])) for r in rows)
        out.setdefault("coral_nb3", {})[str(k)] = skips
        rows = list(csv.DictReader(open(R("runs", "dilute10", "freematch_nb3", f"fold{k}",
                                          "diagnostics.csv"), encoding="utf-8")))
        sk = sum(float(r["step_skipped_mean"]) * int(r["n_steps_in_window"])
                 for r in rows if r["n_steps_in_window"] not in ("", "0"))
        assert abs(sk - round(sk)) < 1e-6, sk
        assert sum(int(r["n_steps_in_window"] or 0) for r in rows) == 1500
        out.setdefault("freematch_nb3", {})[str(k)] = int(round(sk))
    res = {}
    for arm, by in out.items():
        res[arm] = {}
        for k, skips in by.items():
            applied = BUDGET[arm] - skips
            if applied != RECORDED_APPLIED[arm][int(k)]:
                raise SystemExit(f"REFUSING: {arm} fold {k}: {applied} applied, record says "
                                 f"{RECORDED_APPLIED[arm][int(k)]}")
            res[arm][k] = {"skips": skips, "applied": applied, "budget": BUDGET[arm],
                           "applied_fraction": applied / BUDGET[arm],
                           "fires_below_half": applied / BUDGET[arm] < 0.5}
    return res


# ------------------------------------------------------------------ the plan body
def build(brief_path):
    parts = {}
    for n in (1, 2, 3):
        p = R("runs", f"dilute10_prereg_part{n}.json")
        d = jload(p)
        got = self_cert(d)
        if got != PART_DIGESTS[n] or d["sha256_of_body_without_this_field"] != got:
            raise SystemExit(f"REFUSING: part {n} digest {got} != {PART_DIGESTS[n]}")
        parts[n] = d
    p1, p3 = parts[1], parts[3]

    rule = jload(R("audit", "decontam", "rule.json"))
    if self_cert(rule) != D1_RULE_BODY:
        raise SystemExit("REFUSING: rule.json body digest differs from b83fa019")
    rem = entry("data", "splits", "d1_test_neg_remove.json")
    if rem["sha256"] != D1_REMOVAL_FILE:
        raise SystemExit("REFUSING: d1_test_neg_remove.json differs from 3d3deafb")

    code = {n: entry(*n.split("/")) for n in (
        "scripts/metrics.py", "scripts/analyze_lambda0_r6.py",
        "scripts/analyze_window_average.py", "scripts/compare_methods.py",
        "scripts/exact_ids.py", "paper/scripts/_common.py",
        "scripts/analyze_dilute10.py", "scripts/write_dilute10_analysis_plan.py")}

    applied = applied_fractions()
    branches = p1["branches"]
    contrasts = p1["contrasts"]

    plan = {
        "experiment": "dilute10 -- analysis plan",
        "status": ("hashed and enrolled BEFORE any value of the 1:10 experiment is computed; "
                   "parts 1-3 govern, and where this plan and part 1 differ, part 1 wins"),
        "recorded_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "brief": {"path_basename": os.path.basename(brief_path), "sha256": sha(brief_path)},
        "governing": {f"part{n}": {"path": f"runs/dilute10_prereg_part{n}.json",
                                   "sha256": PART_DIGESTS[n], "verified": True}
                      for n in (1, 2, 3)},
        "inputs": inputs(p1),
        "pool": {
            "primary": "D1 (cleaned): the shared test negatives without the 265 objects that "
                       "share an exact Euclid ID with a catalogued lens or a simulation host",
            "rule": {"path": "audit/decontam/rule.json", "body_sha256": D1_RULE_BODY,
                     "file_sha256": sha(R("audit", "decontam", "rule.json"))},
            "removal_list": rem,
            "filter": "paper/scripts/_common.pool_filter, applied to every stored score "
                      "vector via its run's own ids.npz (asserts 265 negatives, 0 positives "
                      "removed under D1; negative order equal to splits.json)",
            "sensitivities": {
                "orig": "the pre-registered pool, unfiltered (_common pool 'orig')",
                "reduced": "D2 = D1 minus the 20 test lenses that also appear as training or "
                           "validation negatives (paper Appendix G, _common pool 'D2')"},
        },
        "code": code,
        "code_note": ("scripts/analyze_dilute10.py is the driver; it re-implements no metric "
                      "or interval. Any later edit to it is recorded in results.json with its "
                      "new sha256 and the reason. Metrics: metrics.metric_suite; intervals: "
                      "analyze_lambda0_r6.paired_t (df = 3) via _common.contrast's form; "
                      "frozen-stage bootstrap: compare_methods.compare."),
        "metrics": {
            "primary": "tpr = TPR at FPR 1e-3",
            "secondary_per_fold": {"auroc": "AUROC",
                                   "comp_at_purity": "completeness at 24% purity (1:2200)",
                                   "purity_corr": "purity at 50% completeness (1:2200)"},
            "fold_mean_rule": "mean and sample sd over folds 1-4 (_common.fold_mean); n/a with "
                              "no fold-mean or interval if any fold is undefined",
        },
        "reproduction_check": {
            "what": "through these inputs and code, on the cleaned pool, folds 1-4: the "
                    "per-fold TPR@1e-3 of balanced DANN 4000, BNM 4000, CORAL 2000, FreeMatch "
                    "1500 and lambda=0 at 4000/2000/1500, and the balanced - lambda=0 "
                    "contrasts (mean and 95% CI)",
            "against": "paper/tables/T5_perfold_v2.tex (Table 5) and "
                       "paper/tables/T2v2_contrasts4.tex (Table 4), at their hashes above",
            "tolerance": "exact at four decimals (the printed token)",
            "on_failure": "STOP: no diluted value is computed; report",
            "ids_check": "every diluted logits file: negatives equal splits.json's test "
                         "negatives in order, 265 removed by D1; positives equal that fold's "
                         "positives in the balanced and lambda=0 comparators' ids.npz",
        },
        "contrasts": {
            "part1_common_method": contrasts["common_method"],
            "primary_G_diluted_minus_lambda0": [
                {"arm": "DANN x nb3", "diluted_at": 4000, "lambda0_at": 4000, "form": "primary"},
                {"arm": "BNM x nb3", "diluted_at": 4000, "lambda0_at": 4000, "form": "primary"},
                {"arm": "CORAL x nb3", "diluted_at": 2000, "lambda0_at": 4000, "form": "primary"},
                {"arm": "CORAL x nb3", "diluted_at": 2000, "lambda0_at": 2000,
                 "form": "sensitivity"},
                {"arm": "FreeMatch x nb3", "diluted_at": 1500, "lambda0_at": 4000,
                 "form": "primary"},
                {"arm": "FreeMatch x nb3", "diluted_at": 1500, "lambda0_at": 1500,
                 "form": "sensitivity"},
            ],
            "part1_primary_list_VERBATIM": contrasts["primary"]["list"],
            "secondary_D_balanced_minus_diluted": [
                {"arm": "DANN x nb3", "at": 4000}, {"arm": "BNM x nb3", "at": 4000},
                {"arm": "CORAL x nb3", "at": 2000}, {"arm": "FreeMatch x nb3", "at": 1500}],
            "part1_secondary": contrasts["secondary"],
            "interval": "paired t on the four per-fold differences, df = 3 "
                        "(analyze_lambda0_r6.paired_t); report mean, 95% CI and the number of "
                        "folds agreeing with the sign of the mean (_common.contrast)",
            "excludes_zero": "paired_t's own excludes_zero: (lo * hi) > 0; a bound exactly at 0 "
                             "includes zero",
            "multiplicity": contrasts["multiplicity"],
            "multiplicity_brief": "10 t-intervals plus 2 bootstrap intervals, no correction; "
                                  "stated in the report",
        },
        "verdict": {
            "part1_branches_VERBATIM": branches,
            "form_used_for_the_branch": (
                "part 1 section 4.1 gives CORAL and FreeMatch two G forms: lambda=0 at 4000 "
                "'primary' and the matched step as 'sensitivity'. The branch verdict uses the "
                "PRIMARY form (DANN 4000/4000, BNM 4000/4000, CORAL 2000/4000, FreeMatch "
                "1500/4000); the sensitivity form is reported beside it with the branch it "
                "would give, labelled as such and never as the verdict"),
            "D_reading": "D's interval 'excludes zero (positive)' iff ci_lo > 0; otherwise it "
                         "'includes zero or is negative'",
            "mechanical_rule": [
                "if part 3's no-verdict rule fires for the arm: no branch; values descriptive",
                "else if G's interval includes zero: COLLAPSES",
                "else if G's interval lies wholly ABOVE zero and D's ci_lo > 0: SHRINKS",
                "else if G's interval lies wholly ABOVE zero: SURVIVES",
                "else (G's interval lies wholly BELOW zero): part 1 section 7 has no branch "
                "that fits; the analysis assigns NONE, stops before the prediction scoring "
                "and reports for the author"],
            "mandatory_qualifier_test": (
                "under SURVIVES only: fires iff D's ci_hi > 0.5 x (balanced - lambda=0), where "
                "balanced - lambda=0 is the paper's fold-mean contrast in the same form as the "
                "verdict (DANN 4000/4000, BNM 4000/4000, CORAL 2000/4000, FreeMatch 1500/4000), "
                "as reproduced in the reproduction check; the qualifier sentence is then "
                "appended with [D upper] = D's ci_hi and [balanced - lambda=0] = that value"),
            "licensed_sentence": "printed from part 1's table with [arm], G [CI] (primary form) "
                                 "and D [CI] filled in, at four decimals, and nothing else",
        },
        "no_verdict_rule_part3": {
            "VERBATIM": p3["no_verdict_rule"],
            "binding_on": p3["binding_on"],
            "applied_updates_from_run_files": applied,
            "fires_on_any_run": any(v["fires_below_half"] for a in applied.values()
                                    for v in a.values()),
        },
        "descriptive": {
            "part1_descriptive": contrasts["descriptive"],
            "retained_fraction": "G / (balanced - lambda=0), point estimate only, primary form "
                                 "for the verdict; for CORAL and FreeMatch also the matched "
                                 "form G_sens / (balanced - lambda=0 matched)",
            "freematch_decomposition": (
                "as paper 6.3 (make_rewrite_v5_macros): start = folds 1-4 cleaned-pool "
                "fold-mean TPR of the diluted FreeMatch frozen head (step 8000), RECOMPUTED "
                "from its persisted logits through pool_filter + metric_suite and "
                "cross-checked against gate_values.json's cleaned_pool block; frozen part = "
                "start - lambda=0 at 4000; unfrozen part = diluted FreeMatch at 1500 - start; "
                "the parts sum to G (primary). The balanced decomposition (0.4118 / 0.2276 / "
                "0.1049) is reproduced beside it; CORAL's diluted frozen start likewise"),
            "frozen_vs_round1": (
                "diluted frozen FreeMatch and CORAL (step 8000) against round-1 frozen "
                "FreeMatch and CORAL (step 8000), folds 1-4, cleaned pool, through "
                "compare_methods.compare -- the class-conditional paired bootstrap of the "
                "paper's frozen comparison (Table 13): shared negatives resampled once per "
                "replicate and reused across folds, each fold's positives resampled "
                "independently, B = 1000, seed 0 (the paper's values); difference = diluted "
                "- round 1; 2 intervals (TPR@1e-3); the suite's other metrics reported "
                "descriptively"),
            "coral_head_growth": "LayerNorm gain norm and ||head||_2 of the diluted CORAL "
                                 "frozen heads against round 1, as recorded in PROJECTv7 7.46; "
                                 "||head||_2 re-read from gate_values.json head_l2",
            "run_bookkeeping": "per run: GradScaler skips and applied fraction where recorded "
                               "(DANN, CORAL nb3: steps.csv; FreeMatch nb3: diagnostics.csv; "
                               "BNM and the frozen stages: not recorded), CORAL nb3's "
                               "feature-norm advisory count, lens-fraction run-means "
                               "(target_lens_fraction_summary.json)",
            "bnm_predicted_positive": "BNM's bnm_pred_pos_frac trajectory from each diluted "
                                      "fold's trace.csv (logged every 125 steps); the balanced "
                                      "BNM folds' trace.csv beside it as context",
        },
        "predictions": {
            "part1_VERBATIM": p1["predictions_VERBATIM"],
            "rule": "each component scored held / failed / not testable with the number it "
                    "rests on; where a component's wording admits two literal readings, BOTH "
                    "are scored and reported, neither is chosen; no prediction is amended",
            "relative_loss": "D / (balanced - lambda=0), primary form",
            "rubric": {
                "1 DANN": [
                    {"id": "1a", "text": "degrades", "test": "D point > 0"},
                    {"id": "1b", "text": "G's point estimate falls below the balanced +0.31",
                     "test": "G point < 0.31 (as stated) and G point < reproduced balanced - "
                             "lambda=0 (4000/4000); both reported"},
                    {"id": "1c", "text": "D is positive",
                     "test": "reading (i) D point > 0; reading (ii) D ci_lo > 0"},
                    {"id": "1d", "text": "G's interval probably includes zero",
                     "test": "G interval includes zero (the hedge 'probably' is recorded)"},
                    {"id": "1e", "text": "COLLAPSES, or SHRINKS with a wide D",
                     "test": "branch in {COLLAPSES, SHRINKS}; 'wide' is not operationalised "
                             "-> that qualifier is not testable"},
                    {"id": "1f", "text": "mechanism: marginal alignment pulls lens features "
                                         "toward galaxies; prior collapses",
                     "test": "not testable from stored logits"}],
                "2 CORAL": [
                    {"id": "2a", "text": "degrades", "test": "D point > 0"},
                    {"id": "2b", "text": "less than DANN",
                     "test": "reading (i) absolute D_CORAL < D_DANN; reading (ii) relative "
                             "loss CORAL < relative loss DANN"},
                    {"id": "2c", "text": "G's interval still excludes zero",
                     "test": "G interval (primary) excludes zero"},
                    {"id": "2d", "text": "SHRINKS or SURVIVES",
                     "test": "branch in {SHRINKS, SURVIVES}"},
                    {"id": "2e", "text": "mechanism: covariance alignment is a weaker pull",
                     "test": "not testable"}],
                "3 BNM": [
                    {"id": "3a", "text": "degrades", "test": "D point > 0"},
                    {"id": "3b", "text": "the most of the four in relative terms",
                     "test": "relative loss of BNM is the largest of the four arms"},
                    {"id": "3c", "text": "G's interval may include zero",
                     "test": "not testable (permissive: either outcome is consistent)"},
                    {"id": "3d", "text": "its predicted-positive fraction drifts above the "
                                         "true ~0.09",
                     "test": "per fold, from trace.csv bnm_pred_pos_frac: reading (i) the "
                             "value at the last logged step (4000) > 0.09; reading (ii) any "
                             "logged step > 0.09; held if true on 4/4 folds, failed if 0/4, "
                             "'mixed k/4' otherwise"},
                    {"id": "3e", "text": "mechanism: high-scoring negatives penalised at "
                                         "fixed FPR",
                     "test": "not testable without a new measurement"}],
                "4 FreeMatch": [
                    {"id": "4a", "text": "degrades", "test": "D point > 0"},
                    {"id": "4b", "text": "less than BNM",
                     "test": "reading (i) absolute D_FM < D_BNM; reading (ii) relative loss "
                             "FM < relative loss BNM"},
                    {"id": "4c", "text": "G's interval still excludes zero",
                     "test": "G interval (primary) excludes zero"},
                    {"id": "4d", "text": "its head start (0.41 on the balanced pool) is "
                                         "predicted to shrink",
                     "test": "reading (i) diluted frozen FreeMatch fold-mean TPR (cleaned, "
                             "folds 1-4) < round-1 frozen fold-mean; reading (ii) the "
                             "frozen-vs-round-1 bootstrap interval lies wholly below zero"},
                    {"id": "4e", "text": "mechanism: uniform fairness target; self-adaptive "
                                         "thresholds partly absorb the shift",
                     "test": "not testable"}],
            },
        },
        "sensitivities": {
            "part1": contrasts["sensitivities"],
            "what": "items 1-3 (per-fold metrics, the 6 G and 4 D intervals) repeated on the "
                    "orig and reduced pools; for each contrast, whether zero-inclusion matches "
                    "the cleaned pool's",
            "note": "appendix only; no verdict is taken from them",
        },
        "stop_rules": [
            "integrity.py --verify fails before or after",
            "any planned input's sha256 differs at analysis time",
            "the reproduction check differs at four decimals",
            "an ids check fails",
            "a G interval lies wholly below zero (no fitting branch)",
        ],
        "outputs": ["runs/dilute10_analysis/results.json", "runs/dilute10_analysis/REPORT.md"],
    }
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--brief", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out) and not a.force:
        raise SystemExit(f"REFUSING to overwrite {rel(out)} without --force")
    _verify_integrity(where="start")
    plan = build(a.brief)
    plan["sha256_of_body_without_this_field"] = self_cert(plan)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(plan, fh, indent=1, ensure_ascii=False, sort_keys=True)
    back = jload(out)
    if self_cert(back) != back["sha256_of_body_without_this_field"]:
        raise SystemExit("plan does not re-read to its own digest")
    _verify_integrity(where="end")
    print(f"wrote {rel(out)}  self-certifying sha256 "
          f"{back['sha256_of_body_without_this_field']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
