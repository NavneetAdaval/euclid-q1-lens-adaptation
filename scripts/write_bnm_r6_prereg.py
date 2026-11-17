"""
write_bnm_r6_prereg.py
======================
Build ROUND 6's BNM pre-registration, ``runs/bnm_r6_prereg.json``.

    python -u scripts/write_bnm_r6_prereg.py --draft     # writes the DRAFT, no hash
    python -u scripts/write_bnm_r6_prereg.py --final     # writes the real one + sha256

**--final REFUSES while the venue provenance is missing.** Stage-3 brief 1 makes
``runs/bnm_r6_step0_gradnorms_t4fp16.json`` a precondition: the gamma grid that was swept
came from a T4/fp16 measurement whose file did not come home, and the CPU file must not be
hashed in its place. That precondition is enforced HERE, in code, rather than left to
whoever runs this remembering it -- the same reason ``stage0a_anchors()`` has no fallback
literal.

Every number that a rule reads is LOADED from the artifact that produced it. Nothing that
a rule reads is retyped in this file.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

import bnm_round6 as B                                    # noqa: E402
import run_bnm_r6 as R                                    # noqa: E402
from run_dann_r5 import FOLD_GATE                         # noqa: E402

SWEEP_DIR = os.path.join(REPO, "runs", "bnm_r6_sweep")
VENUE_GRADNORMS = os.path.join(REPO, "runs", "bnm_r6_step0_gradnorms_t4fp16.json")

# PATH 2 (Stage-3 brief 1). The original T4/fp16 measurement's session is gone, so the grid's
# true provenance is UNRECOVERABLE. A re-measurement is a DIFFERENT RUN and is recorded as
# one -- under its own filename, so it can never be mistaken for the original by a later
# reader or by this script. The swept grid is NOT re-derived from it.
REMEASURED_GRADNORMS = os.path.join(
    REPO, "runs", "bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json")

# WHAT COUNTS AS A MATERIAL DIFFERENCE, DECLARED BEFORE THE RE-MEASUREMENT RUNS.
# Same architecture, same precision, same seed, same step 0: the only expected source of
# disagreement is fp16 reduction-order nondeterminism, which is far below this. The
# back-computed ratio is itself precise to ~0.1% (the three swept gammas imply 12.70648 /
# 12.70616 / 12.70600, a spread of 4e-5 relative), so 5% is well clear of rounding noise.
# Declaring it now is the point: a threshold chosen after seeing the number is not a
# threshold.
MATERIAL_RATIO_DELTA = 0.05
CPU_GRADNORMS = os.path.join(REPO, "runs", "bnm_r6_step0_gradnorms.json")
FINAL = os.path.join(REPO, "runs", "bnm_r6_prereg.json")
DRAFT = os.path.join(REPO, "runs", "bnm_r6_prereg.DRAFT.json")

SWEPT_GRID = [0.00787, 0.039351, 0.157406]
FRACTIONS = [0.10, 0.50, 2.00]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sweep_arms():
    """The three Stage-2 arms, read from what actually ran."""
    out = []
    for d in sorted(glob.glob(os.path.join(SWEEP_DIR, "bnm_r6_sweep_g*")),
                    key=lambda p: float(os.path.basename(p).split("_g")[1])):
        rows = list(csv.DictReader(open(os.path.join(d, "checkpoints.csv"))))
        cfg = json.load(open(os.path.join(d, "run_config.json")))
        out.append({"gamma": float(os.path.basename(d).split("_g")[1]),
                    "rows": rows, "cfg": cfg, "dir": os.path.basename(d)})
    return out


def _remeasurement_block(have_venue, have_remeasured, implied_ratio):
    """The path-2 record. Honest about what it is and is not."""
    if have_venue:
        return {"applies": False,
                "why": "the original T4/fp16 measurement was recovered (path 1)"}
    if not have_remeasured:
        return {"applies": False, "why": "no measurement of either kind is present"}

    d = json.load(open(REMEASURED_GRADNORMS))
    ratio = float(d["ratio_bnm_over_ce_at_gamma_1"])
    delta = ratio / implied_ratio - 1.0
    material = abs(delta) > MATERIAL_RATIO_DELTA

    # Does this file, put through the SAME formula and the same 6-dp rounding, regenerate
    # the grid that was actually swept? This is computed, never used to set the grid -- but
    # it is a far stronger statement than "inside 5%", so it is recorded rather than lost.
    regen = [round(f / ratio, 6) for f in FRACTIONS]
    exact = regen == list(SWEPT_GRID)

    return {
        "applies": True,
        "file": "runs/bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json",
        "sha256": sha256_file(REMEASURED_GRADNORMS),
        "WHAT_THIS_IS": (
            "A SECOND, LATER MEASUREMENT. It is NOT the provenance of the gamma grid that "
            "was swept. The original T4/fp16 run's session was lost before its file was "
            "downloaded, so the grid's true provenance is UNRECOVERABLE and this "
            "pre-registration says so rather than substituting a lookalike."),
        "WHAT_THE_SWEEP_ACTUALLY_USED": (
            "the ORIGINAL measurement, whose ratio is known only by back-computation from "
            f"the swept grid: {implied_ratio:.4f}. gamma = 0.00787 was selected under that "
            "grid and REMAINS SELECTED. The grid is NOT re-derived from this file."),
        "device": d.get("device"), "fp16": d.get("fp16"), "gpu_name": d.get("gpu_name"),
        "ce_grad_norm": d.get("ce_grad_norm"),
        "bnm_grad_norm_at_gamma_1": d.get("bnm_grad_norm_at_gamma_1"),
        "ratio_remeasured": ratio,
        "ratio_back_computed_from_swept_grid": round(implied_ratio, 4),
        "delta_vs_swept": round(delta, 5),
        "delta_vs_swept_pct": round(100 * delta, 2),
        "material_threshold": MATERIAL_RATIO_DELTA,
        "material_threshold_declared_before_the_measurement": True,
        "is_material": bool(material),
        "regenerates_the_swept_grid_exactly": exact,
        "grid_regenerated_from_this_file": regen,
        "grid_actually_swept": list(SWEPT_GRID),
        "what_exact_regeneration_does_and_does_not_mean": (
            "DOES: the second measurement, put through the same formula and the same 6-dp "
            "rounding, returns the identical grid -- so the swept grid is reproducible on "
            "this architecture and precision, and the numbers Stage 2 used were not a "
            "one-off artifact of a lost run. That is the strongest corroboration available "
            "short of the original file. DOES NOT: restore provenance. This is still a "
            "different run from the one that produced the grid, the original remains lost, "
            "and the grid is still not re-derived from this file."
            if exact else
            "the second measurement does NOT regenerate the swept grid at 6 dp; the "
            "difference is recorded and the swept grid stands unchanged."),
        "interpretation": (
            "MATERIAL. Same architecture, same precision, same seed, same step 0, yet the "
            "ratio moved more than the declared 5%. That is a WITHIN-ARCHITECTURE variance "
            "data point for 7.7, alongside the CPU->T4 +27.6% already recorded, and it "
            "means the gamma grid is less reproducible than a single measurement suggests."
            if material else
            "NOT MATERIAL. The re-measurement agrees with the back-computed ratio inside "
            "the declared 5%, which is consistent with fp16 reduction-order "
            "nondeterminism and adds no 7.7 finding. It corroborates the swept grid "
            "without being its provenance."),
        "what_this_does_NOT_license": (
            "it does not license re-deriving the grid, re-selecting gamma, or describing "
            "the swept grid as having verified provenance. The grid's own measurement is "
            "gone and stays gone in this record.")}


def build():
    anch = B.stage0a_anchors()
    band = B.collapse_band(anch["init_pred_pos_frac"])
    arms = sweep_arms()

    have_venue = os.path.isfile(VENUE_GRADNORMS)
    have_remeasured = os.path.isfile(REMEASURED_GRADNORMS)
    cpu = json.load(open(CPU_GRADNORMS))
    implied = [f / g for f, g in zip(FRACTIONS, SWEPT_GRID)]
    implied_ratio = sum(implied) / len(implied)

    # ---- the selection, RE-DERIVED from the returned curves, never retyped ----
    sel_input = [{"gamma": a["gamma"],
                  "nuclear_norm_final": float(a["rows"][-1]["bnm_nuclear_norm"]),
                  "pred_pos_frac_final": float(a["rows"][-1]["bnm_pred_pos_frac"]),
                  "frozen_probe_auroc_final": float(a["rows"][-1]["frozen_probe_auroc"]),
                  "frozen_probe_auroc_step0":
                      a["cfg"].get("frozen_probe_auroc_step0", 0.9202)} for a in arms]
    selection = B.select_gamma(sel_input, band=band, init_nuclear=anch["init"])

    diag = []
    for a in arms:
        for r in a["rows"]:
            diag.append({"gamma": a["gamma"], "step": int(r["step"]),
                         "nuclear_gap": float(r["nuclear_gap"]),
                         "entropy_gap": float(r["entropy_gap"]),
                         "pred_pos_frac_gap": float(r["pred_pos_frac_gap"])})
    ent_neg = sum(1 for d in diag if d["entropy_gap"] < 0)
    nuc_pos = sum(1 for d in diag if d["nuclear_gap"] > 0)
    both = sum(1 for d in diag if d["nuclear_gap"] > 0 and d["entropy_gap"] < 0)

    pre = {
        "round": 6,
        "arm": "bnm_nb3",
        "recorded_before_any_confirmatory_fold": True,
        "revision_rule": (
            "NOTHING here may be revised after any confirmatory fold runs. Anything "
            "re-specified from a control arm is labelled derived_from_control_arm and is "
            "NOT pre-registration."),

        # ---------------- 1. PROVENANCE (Stage-3 brief 1) ----------------
        "provenance": {
            "gamma_grid_swept": SWEPT_GRID,
            "fractions_of_ce_grad_norm": FRACTIONS,
            "grid_source": "T4/fp16 --measure-step0, run before Stage 2",
            "venue_gradnorm_file": "runs/bnm_r6_step0_gradnorms_t4fp16.json",
            "venue_gradnorm_file_present": have_venue,
            "venue_gradnorm_sha256": sha256_file(VENUE_GRADNORMS) if have_venue else None,
            "back_computed_ratio_from_swept_grid": round(implied_ratio, 4),
            "back_computed_ratio_per_fraction": [round(x, 5) for x in implied],
            "cpu_reference": {
                "file": "runs/bnm_r6_step0_gradnorms.json",
                "sha256": sha256_file(CPU_GRADNORMS),
                "ce_grad_norm": cpu["ce_grad_norm"],
                "bnm_grad_norm_at_gamma_1": cpu["bnm_grad_norm_at_gamma_1"],
                "ratio": cpu["ratio_bnm_over_ce_at_gamma_1"],
                "NOT_THE_GRID_SOURCE": (
                    "the CPU/fp32 grid was 0.010045/0.050225/0.200901 and was NOT swept; "
                    "Stage-2 brief 1 forbids hashing it as the grid's provenance")},
            "cpu_to_t4_ratio_move": round(implied_ratio /
                                          cpu["ratio_bnm_over_ce_at_gamma_1"], 4),
            "cpu_to_t4_ratio_move_pct": round(
                100 * (implied_ratio / cpu["ratio_bnm_over_ce_at_gamma_1"] - 1), 1),
            "section_7_7_note": (
                "a +27.6% move in the BNM/CE step-0 gradient ratio between CPU/fp32 and "
                "T4/fp16 is a 7.7 data point: results are not portable across "
                "architectures at this precision."),
            # ---- PATH 2: the original is gone; this records what replaced it ----
            "original_measurement_recovered": have_venue,
            "path_taken": ("PATH 1 (the original file)" if have_venue else
                           "PATH 2 (a second, later measurement)" if have_remeasured
                           else "NEITHER -- blocked"),
            "remeasurement": _remeasurement_block(have_venue, have_remeasured,
                                                  implied_ratio),
            "if_the_file_is_unrecoverable": (
                "re-run --measure-step0 on T4/fp16 and record it as a SECOND, LATER "
                "measurement -- never as the provenance for the grid already swept. "
                "Report both numbers, disclose that the sweep used the first, and treat "
                "any material difference as a WITHIN-ARCHITECTURE variance data point for "
                "7.7. DO NOT re-derive the grid from the new file.")},

        # ---------------- 2. gamma, and how it was selected ----------------
        "gamma": {
            "value": selection.get("selected"),
            "selected_from": SWEPT_GRID,
            "n_passing": selection["n_passing"],
            "basis": selection.get("basis"),
            "criterion": selection["criterion"],
            "rule_reads_a_metric": selection["reads_a_metric"],
            "rule_declared_before_the_sweep": True,
            "extension_rule": selection["extension_rule"],
            "extension_used": False,
            "advisory_ceiling_breached_by_selected_arm":
                "ADVISORY_CEILING_BREACHED_BY_SELECTED_ARM" in selection,
            "per_arm": [{k: v for k, v in a.items() if k != "rows"}
                        for a in selection["arms"]]},

        # ---------------- 3. cadence (Stage-3 brief 2) ----------------
        "cadence": {
            "full_evaluation_every": R.CKPT_EVERY,
            "stats_only_every": R.STATS_ONLY_EVERY,
            "stats_only_until": R.STATS_ONLY_UNTIL,
            "n_stats_only_points": 18,
            "statistics_only_property": (
                "sub-grid points record the nuclear norm, mean prediction entropy, "
                "predicted-positive fraction and the 704/78 split ONLY. No TPR, no AUROC, "
                "no evaluation of any kind. Enforced BY CONSTRUCTION: "
                "run_bnm_r6.stats_only_row does not take the scorer as an argument, so it "
                "cannot score even by mistake. test_bnm_round6.T19 asserts the signature, "
                "then the tokenised body, with the full path as the known-null."),
            "why_the_250_grid_is_untouched": (
                "every 250-multiple keeps its full evaluation, so the Task-A decay "
                "detector and every stability comparison against rounds 4, 5 and 7.22 run "
                "on the same grid as before. 7.22 already killed one stability claim that "
                "lived on 250-8000 and dissolved on 250-4000; changing the grid under a "
                "cross-arm comparison would manufacture a second."),
            "sub_grid_rows_go_to": "stats_only.csv, NOT checkpoints.csv",
            "why_a_separate_file": (
                "so no cross-arm comparison can pick up a sub-grid row by accident, and "
                "so ~18 new read-off points never enter the pool 7.16 consumes.")},

        # ---------------- 4. the budget rule, with its THREE named outcomes ----------
        "budget": {
            "rule": (f"budget = {B.BUDGET_MULTIPLIER} x the first fold-0 step at which "
                     f"normalised ||A||_* reaches the sharpening ceiling "
                     f"{anch['sharpening_ceiling']:.4f}"),
            "rule_reads_a_metric": False,
            "rule_unchanged_after_seeing_stage_2": True,
            "ceiling": B.BUDGET_CEILING,
            "multiplier": B.BUDGET_MULTIPLIER,
            "matched_arm_budgets": list(B.MATCHED_ARM_BUDGETS),
            "why_set_membership_and_not_a_floor": (
                "7.22 makes cross-arm contrasts by reading at a MATCHED STEP, not by "
                "clearing a threshold -- lambda=0 was read at 2000 for CORAL and 1500 for "
                "FreeMatch -- and 7.16 showed the read-off point matters enormously. A "
                "budget of 1600 is not 'comparable to FreeMatch's 1500' merely because it "
                "exceeds it. Membership of the declared set is the property that licenses "
                "a contrast; being above some number is not."),
            "why_this_could_change_before_hashing": (
                "OUTCOME-NEUTRAL for this round: the Stage-2 trajectory implies 1000, "
                "which is neither >= 1500 nor a member of {1500, 2000, 4000}, so the "
                "verdict is identical under either definition. A definition change that "
                "cannot alter the outcome it is applied to is a repair of the instrument, "
                "not a selection of the result. PROJECT.md 7.23 records this."),
            "a_match_removes_only_the_budget_confound": (
                "Head initialisation and the gamma-tuning budget still travel with every "
                "contrast (7.18) and are not addressed by reading at a matched step."),
            "named_outcomes": {
                "normal": "the budget is derived and quoted normally",
                "never_crossed": "REFUSES. budget None. Stop and report.",
                "exceeds_ceiling": (
                    "REFUSES. budget None. NEVER clamped to the ceiling -- a clamped "
                    "budget is indistinguishable from a derived one in this JSON."),
                "no_matched_arm_budget": (
                    "DOES NOT REFUSE. Fires when the rule-derived budget is not EXACTLY "
                    "one of the declared arm budgets {1500, 2000, 4000}. The budget is "
                    "derived and used, but the number is quoted ONLY as BNM's "
                    "own-objective reading -- 'BNM read where its own objective says it "
                    "is done'. All cross-arm contrasts use MATCHED READOUTS, with 4000 "
                    "mandatory as the lambda=0 known-null's budget.")},
            "why_declared_now": (
                "Stage 2 measured the saturation -- with C = 2 the confidence term is "
                "~99% spent inside 1000 steps -- so a budget far below every other arm is "
                "foreseeable. Data bearing on the rule has been seen, so THE RULE DOES "
                "NOT CHANGE; only which number may be quoted against other arms, and that "
                "is fixed before the development run rather than after."),

            # ---- Stage-3 brief 2: 1000 IS A PREVIEW, NOT THE DECLARATION ----
            "stage2_preview_budget": 1000,
            "stage2_preview_is_NOT_the_declaration": (
                "1000 is what the rule returns when fed the STAGE-2 trajectory, which was "
                "sampled at 250 cadence -- so its crossing is resolved only to +/-250 and "
                "the implied budget only to +/-500. THE BUDGET IS DETERMINED BY THE "
                "DEVELOPMENT RUN'S TRAJECTORY, not by this preview. The development run's "
                "hybrid cadence resolves the crossing to +/-25 and therefore the budget to "
                "+/-50, so the declared budget may plausibly land anywhere in 750-1050. A "
                "shift from 1000 at Stage 3 is the INTENDED REFINEMENT, not a change of "
                "rule -- recorded here so it cannot read as one afterwards."),
            "budget_resolution_stage2": "+/-500 (250 cadence)",
            "budget_resolution_stage3": "+/-50 (25 cadence to step 500)",
            "plausible_stage3_range": [750, 1050],
            "mandatory_readout": 4000,
            "why_4000": (
                "BNM's gamma=0 known-null is the lambda=0 arm at budget 4000 (7.22), so "
                "4000 is where the control is read and therefore where the treatment must "
                "also be read."),
            "CONFIRM_STEPS": None,
            "is_preregistered_budget": False,
            "budget_status": (
                "OPEN. The rule reads a crossing that only the Stage-3 development run "
                "produces. Following the round-5 precedent for runs/dann_r5_budget.json, "
                "this stays False until a budget is derived and hashed.")},

        # ---------------- 5. THE DISCLOSURE (Stage-3 brief 4) ----------------
        "disclosure_704_78": {
            "what_it_is": (
                "the round-6 analogue of lambda=0's 'three conditions fire on a healthy "
                "control': declared in advance so that seeing it on folds 1-4 is a "
                "disclosed expectation, not a discovery."),
            "signature": ("a POSITIVE nuclear gap with a NEGATIVE entropy gap is "
                          "MEMORISING; both slices moving together is GENERALISING"),
            "size_matched": "78 trained-on vs 78 held-out, drawn the same stratified way",
            "why_size_matching_is_not_optional": (
                "normalised ||A||_* is sqrt(B/C) at a confident state, so 704-vs-78 reads "
                "3.00x higher at IDENTICAL behaviour; an unmatched comparison would report "
                "sqrt(B) as memorisation"),
            "stage2_observation": {
                "n_reads": len(diag),
                "entropy_gap_negative": f"{ent_neg}/{len(diag)}",
                "nuclear_gap_positive": f"{nuc_pos}/{len(diag)}",
                "full_signature_present": f"{both}/{len(diag)}",
                "per_read": diag},
            "SUPPORTED": (
                "the entropy gap is negative in every read; the full signature is present "
                "in 10 of 12; magnitude is dose-responsive across the grid "
                "(+0.021 -> +0.069 -> +0.131); at the selected gamma it is absent early "
                "and emerges (-0.0085, -0.0232, +0.0410, +0.0729)."),
            "NOT_SUPPORTED_AND_NOT_CLAIMED": (
                "any gamma-ordering. Within-arm scatter is sd ~ 0.043 against a 0.111 "
                "separation, from four autocorrelated checkpoints on one fold."),
            "NO_SIGN_TEST": (
                "the 12 reads are 3 arms x 4 checkpoints and are autocorrelated within "
                "arm, so the effective unit is the ARM and n = 3. Reported descriptively "
                "-- 'consistent in sign across all three gamma and all four checkpoints' "
                "-- with NO p-value and NO confidence interval."),
            "PREDICTED_ON_FOLDS_1_4": (
                "the memorising signature PERSISTS: entropy gap negative and nuclear gap "
                "positive at the majority of full-grid checkpoints on each of folds 1-4, "
                "at magnitudes comparable to the selected gamma's fold-0 values "
                "(nuclear gap of order +0.02 to +0.08 on a 6.245 ceiling, i.e. ~0.3-1.3%)."),
            "PREDICTION_REASONING": (
                "7.12 measured 14.8M parameters memorising these same 782 target images on "
                "4/4 confirmatory folds, and BNM's confidence term (~70% of its pull, "
                "7.20) rewards confident predictions on exactly those images. The trained-"
                "on slice is inside that memorised set and the held-out 78 are not, so the "
                "gap is expected wherever the memorisation is. Fold 0 is one fold and the "
                "selected gamma is the weakest of the three, so folds 1-4 at the same gamma "
                "are the like-for-like test."),
            "SCORED_EITHER_WAY": (
                "this prediction is scored after folds 1-4, not revised. A negative result "
                "is reported as a failed prediction and is informative about BNM, not "
                "about the diagnostic."),

            # ---- Stage-3 brief 3: A PREDICTION WITH NO FALSIFIER SCORES ITSELF ----
            "FALSIFIER": {
                "tolerated": (
                    "ONE fold near zero or slightly negative. Within-arm scatter is "
                    "sd ~ 0.043, so a single off-sign fold is noise, not falsification."),
                "falsified_by_sign": (
                    "consistently negative gaps across folds -- i.e. the memorising "
                    "signature absent or reversed on the majority of folds 1-4."),
                "falsified_by_magnitude": (
                    "nuclear gaps above ~0.15. A gap that large is not the mechanism "
                    "described; it would indicate something other than the 7.12 "
                    "memorisation being read."),
                "why_a_falsifier_is_stated": (
                    "a prediction with no stated failure condition scores itself: any "
                    "outcome can be narrated as consistent with it after the fact.")},

            "TWO_INDEPENDENT_PREDICTIONS_RECORDED": {
                "note": (
                    "an independent prediction was written against this one BEFORE folds "
                    "1-4 ran. Both are recorded, with the disagreement, because a forecast "
                    "that survives having been contradicted in advance is worth more than "
                    "one nobody challenged. THE REGISTERED PREDICTION STANDS -- and it "
                    "stands ON THE MERITS, not merely because it was written first. The "
                    "three reasons are recorded below because the decision went AGAINST "
                    "the better-informed bound, which is the kind of choice that looks "
                    "arbitrary later unless its reasoning is written down now."),
                "why_the_registered_bound_is_the_better_one": {
                    "1_independence": (
                        "+0.04 was derived FROM fold 0 reaching +0.0729 at step 1000 -- one "
                        "fold, four autocorrelated checkpoints, within-arm sd 0.043. A "
                        "lower bound fitted that closely to a single observed trajectory is "
                        "halfway to a postdiction and inherits every bit of that "
                        "trajectory's noise. +0.02 was set BEFORE the crossover was seen, "
                        "so it is independent of the thing it predicts. That makes it the "
                        "more informative forecast even though it is the vaguer one."),
                    "2_different_reference_class_MEASURED_NOT_ASSUMED": (
                        "each of folds 1-4 has a different target composition, so fold 0's "
                        "gap magnitude is not automatically the right point estimate for "
                        "them. MEASURED FROM data/splits/splits.json rather than assumed: "
                        "there are 489 distinct real lenses split into five DISJOINT test "
                        "sets (98/98/98/98/97), and each fold's target pool is all 489 "
                        "minus its own test set (391/391/391/391/392). So fold 0's target "
                        "pool is a strict SUBSET of the union of folds 1-4's (489), not "
                        "equal to it, and it shares 293 of its 391 images -- 74.9% -- with "
                        "each of folds 1-4 individually."),
                    "2b_WHAT_THAT_OVERLAP_DOES_TO_THIS_ARGUMENT": (
                        "IT WEAKENS IT, and that is recorded rather than left out. At 74.9% "
                        "shared images fold 0 IS partly informative about folds 1-4, so "
                        "'tuned to the wrong reference' overstates it: the reference is "
                        "substantially the same, differing by ~25% of the pool. The honest "
                        "form is that fold 0 is a correlated but not interchangeable draw. "
                        "Reasons 1 and 3 do not depend on this and are unaffected; they are "
                        "the load-bearing ones."),
                    "2c_THE_OVERLAP_ALSO_BEARS_ON_THE_FALSIFIER": (
                        "because folds 1-4 share ~75% of their target images with each "
                        "other as well, their gaps are NOT independent draws. Cross-fold "
                        "scatter should therefore be SMALLER than independent sampling "
                        "would imply, which makes the 'one fold near zero is tolerated' "
                        "allowance more generous than it first looks -- a genuinely "
                        "off-sign fold under this much sharing is more surprising, not "
                        "less. The tolerance is kept as declared; this note exists so that "
                        "if exactly one fold does come back off-sign, the reading is made "
                        "with the correlation in view rather than discovered afterwards."),
                    "3_a_bound_noise_alone_can_break_is_not_testing_the_mechanism": (
                        "with within-arm sd ~ 0.043, a true value sitting AT a lower bound "
                        "of +0.04 can be carried to zero or below by ordinary fold scatter "
                        "-- the bound would be falsified by noise rather than by the "
                        "mechanism being wrong. The wider [0.02, 0.08] window is robust to "
                        "one-sd scatter in a way [0.04, 0.08] is not."),
                    "what_this_costs": (
                        "the registered bound is weaker: it is satisfied by outcomes the "
                        "independent one would call failures. That cost is accepted "
                        "deliberately -- a prediction that only survives because it was "
                        "fitted to the one trajectory available is not worth more for "
                        "being narrower.")},
                "registered_prediction_written_first": (
                    "nuclear gap of order +0.02 to +0.08 on a 6.245 ceiling (~0.3-1.3%), "
                    "sign positive with entropy gap negative, at the majority of full-grid "
                    "checkpoints on each of folds 1-4."),
                "independent_prediction": (
                    "sign: positive nuclear gap, negative entropy gap, on ALL FOUR folds. "
                    "magnitude: roughly +0.04 to +0.08 at a budget near 1000, from fold "
                    "0's crossover around steps 500-750 reaching +0.0729 by 1000."),
                "where_they_disagree": (
                    "the LOWER BOUND on magnitude: +0.02 (registered) vs +0.04 "
                    "(independent). The registered prediction is the more permissive, so "
                    "an outcome in [0.02, 0.04) confirms it and fails the independent one. "
                    "The upper bound (+0.08), the sign, the falsifier and the mechanism "
                    "are agreed."),
                "how_this_is_scored": (
                    "both are scored against the same folds 1-4 result. If the outcome "
                    "lands in [0.02, 0.04) that is reported explicitly as the registered "
                    "prediction surviving where the independent one did not, and vice "
                    "versa above 0.08. THE DISAGREEMENT IS DELIBERATELY LEFT LIVE: it was "
                    "not collapsed to a single number before the data arrived, because the "
                    "[0.02, 0.04) window is the genuinely diagnostic one and a quiet "
                    "convergence beforehand would have destroyed the only place where the "
                    "two forecasts can be told apart.")}},

        # ---------------- 6. 7.20's predictions, carried forward UNAMENDED ----------
        "predictions_from_7_20": {
            "carried_verbatim": True,
            "amended": False,
            "prediction_1_decay_direction": (
                "Delta POSITIVE (late-is-worse), point estimate nearer CORAL's +0.0278 "
                "than FreeMatch's +0.0951, and NOT distinguishable from zero at cap 8000."),
            "prediction_1_why_ambiguous": (
                "7.14's rule 'no pseudo-label loop => no confirmation-bias decay' does not "
                "resolve BNM: it has no pseudo-label loop (like CORAL) but does have a "
                "confidence-sharpening loop (like FreeMatch). The ambiguity is why this is "
                "pre-registered rather than assumed."),
            "prediction_2_collapse": (
                "target predicted-positive fraction DRIFTS UPWARD from 0.2292 toward ~0.5, "
                "and does NOT collapse to 0 or 1."),
            "prediction_2_stage2_status": (
                "CONSISTENT SO FAR on fold 0 development only: end-of-run predicted-positive "
                "fraction was 0.2617 / 0.3477 / 0.4062 across the grid, upward from 0.2292 "
                "and monotone in gamma, with no collapse. This is NOT the confirmatory "
                "scoring of the prediction, which happens on folds 1-4."),
            "scored_not_revised": True},

        # ---------------- 7. the collapse band ----------------
        "collapse_band": {
            "init_pred_pos_frac": anch["init_pred_pos_frac"],
            "floor": band["floor"],
            "floor_status": "GATES -- an arm below it does not pass selection",
            "ceiling": band["ceiling"],
            "ceiling_status": (
                "ADVISORY ONLY. It logs and ESCALATES in the result, and never selects. A "
                "detector that can only trigger on catastrophe is worse than an advisory "
                "one that triggers on drift."),
            "separate_flags_through_to_the_artifact": [
                "collapse_gate_ok", "collapse_advisory_ceiling_ok"],
            "at_step_0": {
                "pred_pos_frac": anch["init_pred_pos_frac"],
                "gate_ok": bool(anch["init_pred_pos_frac"] >= band["floor"]),
                "advisory_ceiling_ok": bool(
                    anch["init_pred_pos_frac"] <= band["ceiling"])},
            "the_standing_rule": (
                "the band must not fire at step 0. Earned three times: round 5's abort3b, "
                "condition 4, and the 0.25-0.75 absolute band this one replaces.")},

        # ---------------- 8. conditions ----------------
        "conditions": {
            "arming": "ADVISORY, NEVER ABORTING, on every fold.",
            "why": (
                "round 5's abort3b fired on all four confirmatory folds, and three "
                "conditions fire on a healthy lambda=0 control. A detector that fires on "
                "healthy runs must not be allowed to end them."),
            "derived_from_control_arm": [],
            "derived_from_control_arm_note": (
                "anything re-specified from a control arm is labelled here and is NOT "
                "pre-registration.")},

        # ---------------- 9. the step-0 gate, non-skippable ----------------
        "step0_gate": {
            "non_skippable": True,
            "per_fold_tpr": {str(k): FOLD_GATE[k]["tpr"] for k in sorted(FOLD_GATE)},
            "per_fold_auroc": {str(k): FOLD_GATE[k]["auroc"] for k in sorted(FOLD_GATE)},
            "confirmatory_folds": [1, 2, 3, 4],
            "required_abs_diff": 0.0,
            "what_it_asserts": (
                "at step 0 the arm reproduces Baseline A's per-fold TPR and AUROC exactly "
                "-- |diff| = 0.0 -- because before any optimiser step the model IS "
                "Baseline A. A non-zero diff means the arm is not starting where it "
                "claims to."),
            "observed_on_fold_0_in_stage_2": {
                "tpr": 0.19387755102040816, "auroc": 0.9427007516518483,
                "pinned_tpr": FOLD_GATE[0]["tpr"], "pinned_auroc": FOLD_GATE[0]["auroc"]}},

        # ---------------- 10. reporting ----------------
        "reporting": {
            "primary_metric": ("TPR@FPR=1e-3, interpolated, drop_intermediate=False, "
                               "ALWAYS a rate, never counts"),
            "quantisation": {"folds_1_3": "1/98", "fold_4": "1/97"},
            "interval": ("paired t-interval on 4 differences, df=3. NOT a bootstrap: at "
                         "n=4 a bootstrap draws from a 4^4=256 lattice."),
            "pooling": "INVALID for per-fold-model arms. Per-fold ROC curves only.",
            "fold0": "Development. Enters no mean.",
            "auprc": "NEVER a selection proxy",
            "auroc": "saturating secondary sanity metric only",
            "morphology": (
                "TWO instruments ~0.34 apart: the trained-head AUROC (~0.57) and the "
                "frozen-feature probe (~0.92). 7.19: never quote one as the other. "
                "Condition 3 reads the FROZEN PROBE."),
            "undefined_values": "report as n/a with NO CI",
            "decay_detector": (
                "Task-A, UNMODIFIED: same definition, same shared-resample paired "
                "bootstrap, B=5000, seed 0, on the 250-grid points ONLY, so it is "
                "comparable to CORAL's +0.0278 and FreeMatch's +0.0951."),
            "nothing_from_fold_0": "Nothing about ranking is concluded from fold 0."},

        "input_artifacts": {},
        "code_artifacts": {},
    }

    for rel in ("runs/bnm_r6_stage0a.json", "runs/bnm_r6_step0_gradnorms.json"):
        p = os.path.join(REPO, rel)
        if os.path.isfile(p):
            pre["input_artifacts"][rel] = sha256_file(p)
    if have_venue:
        pre["input_artifacts"]["runs/bnm_r6_step0_gradnorms_t4fp16.json"] = \
            sha256_file(VENUE_GRADNORMS)
    if have_remeasured:
        pre["input_artifacts"][
            "runs/bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json"] = \
            sha256_file(REMEASURED_GRADNORMS)
    for a in arms:
        for leaf in ("checkpoints.csv", "run_config.json"):
            rel = f"runs/bnm_r6_sweep/{a['dir']}/{leaf}"
            pre["input_artifacts"][rel] = sha256_file(os.path.join(REPO, rel))
    for name in ("bnm_round6.py", "run_bnm_r6.py", "test_bnm_round6.py",
                 "da_losses.py", "test_da_bnm.py"):
        pre["code_artifacts"][f"scripts/{name}"] = sha256_file(
            os.path.join(_HERE, name))

    return pre, have_venue


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--draft", action="store_true")
    g.add_argument("--final", action="store_true")
    args = ap.parse_args()

    pre, have_venue = build()
    have_remeasured = os.path.isfile(REMEASURED_GRADNORMS)

    # THE GRID IS NEVER RE-DERIVED. Asserted here, not trusted: path 2's whole risk is that
    # a second measurement quietly becomes the grid's source.
    assert pre["provenance"]["gamma_grid_swept"] == SWEPT_GRID, \
        "the swept grid changed -- path 2 must never re-derive it"
    assert pre["gamma"]["value"] == 0.00787, \
        "gamma changed -- the selection stands on the swept grid"

    if args.final and not (have_venue or have_remeasured):
        raise SystemExit(
            "REFUSING to write the final pre-registration.\n"
            f"  absent: {os.path.relpath(VENUE_GRADNORMS, REPO)}\n"
            "  The gamma grid that was swept came from a T4/fp16 --measure-step0 whose\n"
            "  file did not come home. Stage-3 brief 1 forbids hashing the CPU file in\n"
            "  its place, so the provenance record cannot be completed.\n"
            "  FIX: recover it from the notebook's /kaggle/working/runs/ (path 1), or\n"
            "  re-measure on T4/fp16 and save it as\n"
            "    runs/bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json\n"
            "  which is recorded as a SECOND, LATER measurement (path 2).\n"
            "  Until then: --draft.")

    out = FINAL if args.final else DRAFT
    pre["status"] = "FINAL" if args.final else (
        "DRAFT -- not authoritative, not hashed into PROJECT.md")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(pre, fh, indent=2)

    print(f"wrote {os.path.relpath(out, REPO)}")
    print(f"  gamma                     {pre['gamma']['value']}")
    print(f"  original measurement      {'RECOVERED' if have_venue else 'LOST'}")
    print(f"  path taken                {pre['provenance']['path_taken']}")
    rm = pre["provenance"]["remeasurement"]
    if rm.get("applies"):
        print(f"  re-measured ratio         {rm['ratio_remeasured']:.4f} vs "
              f"{rm['ratio_back_computed_from_swept_grid']:.4f} back-computed "
              f"({rm['delta_vs_swept_pct']:+.2f}%)")
        print(f"  material (>{int(100 * MATERIAL_RATIO_DELTA)}%)?            "
              f"{rm['is_material']}")
    print(f"  input artifacts hashed    {len(pre['input_artifacts'])}")
    print(f"  code artifacts hashed     {len(pre['code_artifacts'])}")
    if args.final:
        print(f"\n  sha256({os.path.basename(out)}) = {sha256_file(out)}")
        print("  put that line in PROJECT.md.")
    else:
        print("\n  DRAFT. Review it, then re-run with --final once provenance lands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
