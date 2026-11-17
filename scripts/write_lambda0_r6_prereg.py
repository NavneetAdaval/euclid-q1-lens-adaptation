"""
write_lambda0_r6_prereg.py
==========================
Writes ``runs/lambda0_r6_prereg.json`` -- the PRE-REGISTRATION for the **lambda=0
unfrozen source-only baseline, folds 1-4** (round 6).

Mirrors ``write_dann_r5_prereg.py``'s writer exactly: canonicalise the body, hash it,
store the hash under ``sha256_of_body_without_this_field``, then re-read the file from
disk and verify the hash reproduces. Self-certifying: anyone can recompute it.

**Nothing in the body may be revised after any confirmatory fold runs.** Anything
re-specified from a control arm is labelled ``derived_from_control_arm`` and is NOT
pre-registration.

    python -u scripts/write_lambda0_r6_prereg.py
"""
from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "runs", "lambda0_r6_prereg.json")
AUDIT = os.path.join(REPO, "runs", "lambda0_r6_audit.json")
COND = os.path.join(REPO, "runs", "lambda0_r6_step0_conditions.json")


def _canonical(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def build_body() -> dict:
    with open(AUDIT, encoding="utf-8") as fh:
        audit = json.load(fh)
    with open(COND, encoding="utf-8") as fh:
        cond = json.load(fh)

    return {
        "round": 6,
        "arm": "dann_nb3_lambda0",
        "what_it_is": (
            "The unfrozen source-only baseline: encoder unfrozen at n_blocks=3, NO "
            "domain-adaptation term. It is round 5's ARM 1 extended from fold 0 to "
            "folds 1-4. lambda is pinned to 0.0 at every step, so the discriminator "
            "sits in the graph and contributes no gradient; what remains is supervised "
            "training on source with three encoder blocks unfrozen."),
        "why_it_exists": (
            "The three nb3 DA arms are all compared against FROZEN Baseline A at 0.176 "
            "while they themselves are unfrozen. Two things differ across that "
            "comparison -- the encoder went from frozen to unfrozen, AND a DA term was "
            "added -- so nothing in the current data attributes the 0.176 -> ~0.42 "
            "movement to either one. This run fills the missing cell."),
        "recorded_before_any_confirmatory_fold": True,
        "revision_rule": (
            "NOTHING here may be revised after any confirmatory fold runs. Anything "
            "re-specified from a control arm is labelled derived_from_control_arm and "
            "is NOT pre-registration."),
        "no_new_code": (
            "No new loss and no new code path. "
            "run_dann_r5.py --arm dann_nb3_lambda0 --fold k --max-steps 4000, "
            "k in {1,2,3,4}."),

        # -------------------------------------------------------------- budget
        "budget": {
            "CONFIRM_STEPS": 4000,
            "rule": ("THE CONTROL IS READ WHERE ITS TREATMENT IS READ. Budget = 4000, "
                     "DANN x nb3's declared budget."),
            "rule_reads_a_metric": False,
            "why_this_rule": (
                "Arms 1 and 2 share an initialisation, a fold, a seed, a GPU and a step "
                "grid and differ in lambda alone -- the only fully controlled cross-arm "
                "contrast in the project. Reading the control at a different step than "
                "its treatment introduces a second difference and destroys the one "
                "clean contrast available. The rule states in one sentence, reads no "
                "metric, and is hashable in advance."),
            "why_no_rule_from_this_arm_s_own_configuration": (
                "This arm has NOTHING with a horizon: no lambda ramp, no EMA, no "
                "thresholds, no mu, no scheduler. The rule that produced FreeMatch's "
                "1500 (SAT saturation), CORAL's 2000 (gamma config with EMA disabled) "
                "and DANN's 4000 (2x the lambda horizon) has nothing to grab onto."),
            "why_not_from_the_fold0_curve": (
                "It peaks at step 250 and degrades, with the highest checkpoint scatter "
                "of any arm (sd/mean 0.276, absolute sd 0.0402). Declaring a budget "
                "from where that curve looks best is the winner's curse 7.11 forbids, "
                "and 7.13 established peak location is unmeasurable at ~98 positives "
                "regardless."),
            "what_this_rule_does_NOT_claim": (
                "It does not claim 4000 is optimal for this arm. It almost certainly is "
                "not -- fold 0 peaks at 250. 'Optimal' was never available (7.13). This "
                "is a DECLARED CHOICE WITH A STATED BASIS, as 7.14 requires of every "
                "per-method budget, not a derived optimum."),
            "alternative_considered_and_declined": {
                "alternative": ("read this arm at EACH DA arm's own budget "
                                "(1500 / 2000 / 4000), which is the coherent choice if "
                                "its primary role were 'method-neutral baseline for all "
                                "three DA arms' rather than 'DANN's control'"),
                "declined_because": ("7.18 treats the lambda contrast as this arm's most "
                                     "striking claim, and no single budget serves the "
                                     "method-neutral role. Recorded so a reader sees the "
                                     "choice was made, not assumed."),
                "partially_recovered_by": ("the pre-declared sensitivity readouts at "
                                           "1500 and 2000 below, which give the "
                                           "budget-matched comparators for free"),
            },
        },

        # ------------------------------------------------- sensitivity readouts
        "sensitivity_readouts": {
            "steps": [1500, 2000],
            "clause": ("Reported ALONGSIDE the headline; they do NOT replace it, "
                       "whichever is higher. 7.18's clause, verbatim."),
            "free": ("ckpt_every is 250 and the budget is 4000, so 1500 and 2000 are "
                     "checkpoints 6 and 8 of 16 on every fold. No extra compute."),
            "what_they_buy": ("budget-matched comparators against FreeMatch x nb3 "
                              "(1500) and CORAL x nb3 (2000), which the 4000 headline "
                              "does not provide"),
            "validity_precondition": (
                "They are valid ONLY IF no component of the schedule is a function of "
                "the run's total length. Re-audited on the CURRENT bytes of "
                "run_dann_r5.py / dann_round5.py / da_losses.py / train_skeleton.py "
                "rather than inherited from 7.18 -- see schedule_audit below."),
            "if_a_component_carries_a_horizon": (
                "DROP the sensitivity readouts and say they were dropped. Do NOT "
                "reinterpret them."),
        },

        # ------------------------------------------------------ schedule audit
        "schedule_audit": {
            "status": "RE-RUN, PASS" if audit["part_c_schedule_audit"]["PASS"]
                      else "RE-RUN, FAIL",
            "n_components": audit["part_c_schedule_audit"]["n_components"],
            "components": {k: v["finding"] for k, v in
                           audit["part_c_schedule_audit"]["components"].items()},
            "sensitivity_readouts_valid":
                audit["part_c_schedule_audit"]["sensitivity_readouts_valid"],
            "artifact": "runs/lambda0_r6_audit.json",
        },

        # ------------------------------------------------- script provenance
        "script_provenance": {
            "finding": (
                "THREE versions of run_dann_r5.py exist in the record -- arm 1 fold 0 "
                "(9b6e38669c), arm 2 fold 0 (4fd5964f37), and DANN folds 1-4 "
                "(320ed27d74, which is also the working tree). 7.19's re-specification "
                "of conditions 3 and 4 landed between them."),
            "what_these_folds_match": (
                "DANN's confirmatory folds 1-4, BYTE-FOR-BYTE on run_dann_r5.py, "
                "dann_round5.py and train_skeleton.py. That is the match the science "
                "needs, because DANN x nb3 - lambda=0 on folds 1-4 is the only fully "
                "controlled contrast."),
            "what_is_therefore_NOT_matched": (
                "Arm 1's fold-0 bytes. Byte-matching both is impossible. CONSEQUENCE, "
                "declared here rather than discovered later: under 7.15's rule, fold "
                "0's 0.1531 is a QUALITATIVE consistency check against the folds-1-4 "
                "mean and is NEVER differenced against it. Fold 0 is development and "
                "enters no mean anyway."),
            "da_losses_py": (
                "differs from every run because round 6 appended BNM to it. PROVEN "
                "append-only: sha256 of lines 1-1441 of the current file equals "
                "6b05cd66dd7ad345, the whole-file hash every DANN run recorded. Every "
                "byte those runs executed is bit-identical."),
            "training_relevant_config_keys_identical_across_all_runs":
                all(v["identical_everywhere"] for v in
                    audit["part_b_config_fields"]["training_relevant"].values()),
            "n_training_relevant_keys_checked":
                len(audit["part_b_config_fields"]["training_relevant"]),
            "n_runs_compared": audit["part_b_config_fields"]["n_runs_compared"],
            "head_init_sha256": audit["part_b_config_fields"]["head_init"]["sha256"],
            "artifact": "runs/lambda0_r6_audit.json",
        },

        # ------------------------------------------------------------- the gate
        "step0_gate": {
            "status": "NON-SKIPPABLE. There is no flag that bypasses it.",
            "why": ("The phase-1 head is fold-INDEPENDENT -- the same file initialises "
                    "all five folds -- so a mis-pointed fold trains happily and produces "
                    "a plausible wrong number. The gate is the only artifact proving "
                    "fold identity, and the AUROC half is what actually proves it: it is "
                    "rank-based and ~100x the resolution of the TPR half."),
            "expected": {
                "1": {"tpr": 0.14285714285714285, "auroc": 0.8922624620134809},
                "2": {"tpr": 0.19387755102040816, "auroc": 0.9031282879453535},
                "3": {"tpr": 0.15306122448979592, "auroc": 0.9016486400613506},
                "4": {"tpr": 0.19587628865979381, "auroc": 0.8908307601467254},
            },
            "tolerance_tpr": {"1": 1 / 98, "2": 1 / 98, "3": 1 / 98, "4": 1 / 97},
            "tolerance_auroc": 1e-3,
            "TOLERANCE_DISAGREEMENT_WITH_THE_BRIEF": {
                "brief_says": "Require |diff| = 0.0 exactly, as round 5 achieved.",
                "declared_here": (
                    "PASS/FAIL stays at the runner's CALIBRATED one-positive tolerance "
                    "(1/98, and 1/97 on fold 4). The observed |diff| is reported "
                    "separately and is expected to be 0.0, as arm 1 achieved."),
                "why": (
                    "Requiring exactly 0.0 at n_blocks=3 re-introduces the failure 7.7 "
                    "documents: the eval runs through live_encode, whose SCORE drift "
                    "against the cached path is ~1e-2. FreeMatch's nb3 gate demanded "
                    "exact reproduction, passed folds 0/1/3 and FAILED fold 2 by exactly "
                    "one positive after a correct 16-minute setup, because fold 2's "
                    "closest positive sat 3.06e-03 above its threshold -- inside the "
                    "drift envelope. The run was fine; the gate's premise was false. "
                    "Round 5 achieving 0.0 is evidence the drift did not bite there, not "
                    "evidence it cannot."),
                "status": "A DELIBERATE DEPARTURE FROM THE BRIEF, declared before any fold runs.",
            },
        },

        # --------------------------------------------------------- conditions
        "conditions": {
            "arming": "ADVISORY, NEVER ABORTING, on every fold.",
            "mechanism": ("run_dann_r5.default_abort_mode returns log_only for EVERY "
                          "fold of a lambda_zero arm -- verified on the emitted configs "
                          "for folds 1-4. No flag is needed and none is passed."),
            "why": ("Round 5's abort3b fired on all four confirmatory folds and under "
                    "the pre-registered arrangement would have killed every one at "
                    "~step 450. v10's decision to make conditions 3 and 4 advisory is "
                    "what preserved that data."),
            "derived_from_control_arm": [
                "condition 3a's margin M (UNSET, and 7.19 declared 3a UNCALIBRABLE: "
                "3J = 0.099 against a ceiling of 0.04)",
                "condition 4's src and tgt margins",
            ],
            "derived_from_control_arm_note": (
                "These are labelled derived_from_control_arm and are NOT "
                "pre-registration. 3a is additionally INAPPLICABLE here: this arm IS "
                "the control, so the condition would compare it against itself."),
            "STEP_0_EVALUATION": {
                "rule": ("Evaluate every declared threshold at step 0 BEFORE declaring "
                         "it, and print the step-0 value into the pre-registration. "
                         "Earned twice: round 5's abort3b, and round 6 Stage 0a's "
                         "0.25-0.75 collapse band, which would have fired at step 0 on "
                         "the initialisation's own 0.2292 predicted-positive fraction."),
                "artifact": "runs/lambda0_r6_step0_conditions.json",
                "per_condition": {
                    k: {"first_logged_step_value": v.get("first_logged_step_value"),
                        "true_step0_value": v.get("true_step0_value"),
                        "fires_at_first_logged_step": v.get("FIRES_AT_FIRST_LOGGED_STEP"),
                        "would_fire_on_arm1": v.get("would_fire_on_arm1")}
                    for k, v in cond["conditions"].items()},
                "THREE_CONDITIONS_FIRE_IMMEDIATELY_ON_A_HEALTHY_CONTROL": {
                    "which": cond["conditions_that_would_fire_on_arm1"],
                    "3b_head_column": (
                        "morph_head_auroc_pool is 0.5657 at the true step 0 but 0.4356 "
                        "at the first logged training step, so the < 0.5 detector fires "
                        "at step 1 on a run with NO adversary at all. Longest streak 375."),
                    "4_rank_collapse": (
                        "participation ratio ranges 2.03-9.69 across all 8000 steps of "
                        "arm 1, entirely below the declared threshold of 20 of 640. The "
                        "condition fires on 100% of a healthy control's steps and has "
                        "never had discriminative power in this pipeline."),
                    "consequence": (
                        "Both are recorded as EXPECTED FIRINGS. A firing here is a "
                        "property of the threshold, not evidence about the run, and it "
                        "must not be reported as a finding about folds 1-4. Neither can "
                        "stop anything: the arm is log_only on every fold."),
                    "the_one_that_is_actually_informative": (
                        "3b on the PROBE column (morph_proxy_auroc < 0.7) does NOT fire: "
                        "0.9202 at step 0, 0.9172 at step 1, longest streak 0 over 8000 "
                        "steps. It is the only member of this family with headroom, and "
                        "it is the one to watch."),
                },
            },
        },

        # ----------------------------------------------------------- reporting
        "reporting": {
            "primary_metric": ("TPR@FPR=1e-3, interpolated, drop_intermediate=False, "
                               "ALWAYS a rate, never counts"),
            "quantisation": {"folds_1_3": 1 / 98, "fold_4": 1 / 97,
                             "note": "fold 4 holds 97 positives"},
            "secondary": ["completeness@24%-purity (Lines' operating point)",
                          "purity@50%-completeness"],
            "operating_point": "FIXED, not selected",
            "auprc": "NEVER a selection proxy",
            "auroc": "saturating secondary sanity metric only",
            "pooling": "INVALID for per-fold-model arms. Per-fold ROC curves only.",
            "interval": ("paired t-interval on 4 differences, df=3. NOT a bootstrap: at "
                         "n=4 a bootstrap draws from 4^4=256 distinct resamples and its "
                         "tails are a lattice artifact."),
            "undefined_values": ("report as n/a with NO CI. Never a "
                                 "reachability-conditional bootstrap, never an aggregate "
                                 "fabricated from the reachable subset."),
            "round1": "No number here is differenced against any round-1 number.",
            "fold0": "Development. Enters no mean.",
            "paired_contrasts": [
                {"contrast": "lambda=0 - Baseline A (frozen)",
                 "isolates": "unfreezing alone"},
                {"contrast": "DANN x nb3 - lambda=0",
                 "isolates": "the adversarial term alone -- the only fully controlled "
                             "contrast"},
                {"contrast": "CORAL x nb3 - lambda=0",
                 "isolates": "DA vs no DA, CONFOUNDED"},
                {"contrast": "FreeMatch x nb3 - lambda=0",
                 "isolates": "DA vs no DA, CONFOUNDED"},
            ],
            "computed_at": ["the declared budget 4000", "step 1500", "step 2000"],
            "DISCLOSURE_that_travels_with_the_bottom_two_rows": (
                "This arm initialises from Baseline A's head, as DANN x nb3 does. CORAL "
                "x nb3 initialises from round-1 CORAL heads and FreeMatch x nb3 from its "
                "own. So it is a legitimate METHOD-NEUTRAL baseline for those two but "
                "NOT a matched control for them. Only the DANN row is a matched "
                "contrast. State this in the SAME SENTENCE as the numbers, not in a "
                "footnote."),
            "SECOND_DISCLOSURE_found_by_the_audit": (
                "DANN's confirmatory folds ran with abort_mode='armed' while this arm "
                "runs 'log_only' on every fold. An armed detector either aborts or does "
                "nothing, and DANN's folds did not abort, so this cannot have changed a "
                "training step -- but it is a real asymmetry in the 'only fully "
                "controlled contrast' and is disclosed rather than smoothed over."),
            "stability_table": (
                "Compute this arm's checkpoint-to-checkpoint sd on folds 1-4 and add it "
                "to 7.18's table alongside fold 0's sd/mean 0.276. If unfreezing without "
                "a DA term is the noisiest arm on four more folds, that becomes a claim "
                "rather than a single-fold observation."),
        },

        # ------------------------------------------------------------ decision
        "what_this_decides": {
            "declared_before_the_folds_exist": True,
            "note": "Both branches are declared NOW so that neither can be chosen afterwards.",
            "branches": [
                {"outcome": "folds 1-4 come in LOW (roughly consistent with fold 0's "
                            "0.1531, well below the DA arms)",
                 "reading": "unfreezing alone does nothing or slightly hurts, and the "
                            "0.15 -> 0.42 movement is attributable to the DA term. The "
                            "paper's headline comparison becomes sound and 7.18's "
                            "single-fold caveat is discharged.",
                 "bnm": "GO"},
                {"outcome": "folds 1-4 come in NEAR THE DA ARMS (~0.40+)",
                 "reading": "the DA terms are decorative and the finding is about "
                            "encoder capacity, not domain adaptation. A different paper, "
                            "and still a publishable one -- write that one.",
                 "bnm": "NO-GO, and the bake-off stops"},
                {"outcome": "IN BETWEEN, or the fold spread makes the call ambiguous",
                 "reading": "7.16 applies.",
                 "bnm": "NO-GO by default. Do not run BNM to break a tie that cannot be "
                        "broken."},
            ],
            "on_landing": ("Update PROJECT.md's 'Next, in order' and 7.18's limitation "
                           "note ('the effect size relative to unfreezing alone rests on "
                           "a single fold') to say what the four folds actually showed."),
        },

        # ------------------------------------------------------------- the run
        "run": {
            "command": ("python -u scripts/run_dann_r5.py --arm dann_nb3_lambda0 "
                        "--fold {k} --max-steps 4000"),
            "folds": [1, 2, 3, 4],
            "ckpt_every": 250, "log_every": 125,
            "checkpoints_per_fold": 16,
            "cost": "~153 min/fold on T4, ~10.2 GPU-h total, two Kaggle sessions",
            "on_declared_steps_will_read_false": (
                "ARMS['dann_nb3_lambda0']['steps'] is round 5's development maximum of "
                "8000, so passing --max-steps 4000 makes run_config record "
                "on_declared_steps=false. That is CORRECT and means 'not on round 5's "
                "development maximum'. 4000 is ROUND 6's declared confirmatory budget "
                "and this file is its authority. run_dann_r5.py is deliberately NOT "
                "edited to say so, because editing it would break the byte-match against "
                "DANN's confirmatory folds -- which is the match that matters."),
            "confirm_budget_rule_field_is_round5s": (
                "run_config.json's confirm_budget_rule / budget_status fields carry "
                "round 5's DANN wording ('2 x the lambda horizon'), a provenance "
                "artifact of reusing the runner unmodified. THIS FILE is authoritative "
                "for round 6's rule."),
            "instruments": {
                "log_both_morphology_instruments_at_every_checkpoint": True,
                "never_quote_one_as_the_other": True,
                "trained_head_auroc_scale_at_step0": 0.578125,
                "frozen_probe_auroc_scale_at_step0": 0.920166015625,
                "separation": "~0.34",
            },
            "record_gpu_architecture_for_every_fold": (
                "results are not portable across GPU architectures at this precision "
                "(7.7)"),
            "collection_defect_to_avoid": (
                "Round 5's per-checkpoint logits never came home (7.18). CONFIRM the "
                "logit files are present and complete BEFORE the Kaggle session ends."),
        },

        "anti_drift": [
            "The target is BALANCED: ~391 unlabeled real lenses : ~391 fresh P2 "
            "negatives, ~1:1, per fold. The 1:2200 rarity applies only at evaluation. "
            "This has been miscorrected repeatedly.",
            "Two morphology instruments, ~0.34 apart. Never quote one as the other; a "
            "monitor built on the frozen probe cannot see a failure that only moved the "
            "head.",
            "Compute the interval before writing the sentence, and always run the "
            "known-null -- in this project the known-null has changed the conclusion "
            "more often than the main analysis has.",
        ],

        "input_artifacts": {
            "runs/lambda0_r6_audit.json": _sha256_file(AUDIT),
            "runs/lambda0_r6_step0_conditions.json": _sha256_file(COND),
        },
    }


def main() -> int:
    body = json.loads(json.dumps(build_body()))
    h = hashlib.sha256(_canonical(body).encode()).hexdigest()
    body["sha256_of_body_without_this_field"] = h

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2, sort_keys=True)

    with open(OUT, encoding="utf-8") as fh:
        chk = json.load(fh)
    got = chk.pop("sha256_of_body_without_this_field")
    again = hashlib.sha256(_canonical(chk).encode()).hexdigest()

    print("=" * 78)
    print("ROUND 6 PRE-REGISTRATION -- lambda=0 unfrozen source-only, folds 1-4")
    print("=" * 78)
    print(f"  file   : {OUT}")
    print(f"  sha256 : {got}")
    print(f"  verify : {'OK -- re-read from disk and reproduces' if got == again else 'MISMATCH'}")
    print(f"  budget : {body['budget']['CONFIRM_STEPS']}  "
          f"rule: {body['budget']['rule']}")
    print(f"  sensitivity readouts: {body['sensitivity_readouts']['steps']}  "
          f"(valid: {body['schedule_audit']['sensitivity_readouts_valid']})")
    print(f"  conditions: {body['conditions']['arming']}")
    print(f"  expected firings on a healthy control: "
          f"{body['conditions']['STEP_0_EVALUATION']['THREE_CONDITIONS_FIRE_IMMEDIATELY_ON_A_HEALTHY_CONTROL']['which']}")
    return 0 if got == again else 1


if __name__ == "__main__":
    raise SystemExit(main())
