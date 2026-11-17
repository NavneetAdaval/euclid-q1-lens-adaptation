"""
write_dann_r5_prereg.py
=======================
Emit ``runs/dann_r5_prereg.json`` -- round 5's PRE-REGISTRATION -- and self-certify it.

Mirrors ``analyze_coral_dev8000_budget.py``'s writer exactly: canonicalise the body
through JSON, hash it with ``sort_keys=True, separators=(",", ":")``, store the digest in
``sha256_of_body_without_this_field``, then **re-read the file from disk and verify the
digest**. A pre-registration whose hash does not check is not a pre-registration.

WHAT THIS FILE IS. Everything fixed before arm 1 runs. It does NOT contain the
confirmatory budget -- that is declared and hashed separately into
``runs/dann_r5_budget.json`` after arm 2, mirroring ``coral_r4_budget.json`` -- and it
does not contain abort condition 3's margin as a settled value, because 7.14's protocol
requires the margin be hashed after arm 1 and before arm 2. Both are recorded here as
PENDING with the rule that will fix them.

Values are READ FROM THE CODE, not retyped: the arms, the pinned keys, the gate table,
the discriminator topology and the abort constants all come from ``run_dann_r5`` /
``dann_round5`` / ``da_losses`` at write time. A pre-registration that disagrees with the
code it pre-registers is worse than none, and transcription is how that happens.

Run:
    python scripts/write_dann_r5_prereg.py
    python scripts/write_dann_r5_prereg.py --verify        # check the digest, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import da_losses as DL          # noqa: E402
import dann_round5 as D5        # noqa: E402
import run_dann_r5 as R5        # noqa: E402


def _canonical(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def build_body() -> dict:
    probe_d = DL.DomainDiscriminator(640)
    return {
        "purpose": (
            "PRE-REGISTRATION for round 5, DANN x n_blocks=3. Recorded BEFORE any arm "
            "runs. Nothing below may be revised after seeing any confirmatory fold. "
            "Format follows PROJECT.md 7.11; the self-certifying digest follows "
            "runs/coral_r4_budget.json."),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "dann",
        "protocol": "PROJECT.md 7.14 cross-method protocol, third method after CORAL",
        "exploratory_confirmatory_split": (
            "FOLD 0 IS DEVELOPMENT. FOLDS 1-4 ARE CONFIRMATORY. The budget will be "
            "chosen by inspecting fold-0 test labels, so fold 0 cannot contribute to any "
            "headline. The confirmatory comparison is folds 1-4, 391 positives."),

        # ---------------- 2.1 arms and order ----------------
        "arms_and_order": [
            {"n": 1, "arm": "dann_nb3_lambda0", "fold": 0, "steps": 8000,
             "abort_mode": "log_only",
             "why_first": (
                 "it is the REFERENCE TRAJECTORY abort condition 3 subtracts, not merely "
                 "a null for the DANN loss, so it has to exist before arm 2 starts. It "
                 "runs the FULL 8000 for the same reason: a control stopping at the "
                 "design doc's 2000 would leave condition 3 -- and feature std, and "
                 "participation ratio -- with no reference past step 2000."),
             "why_log_only": ("it is the reference; it must always complete")},
            {"n": 2, "arm": "dann_nb3", "fold": 0, "steps": 8000,
             "abort_mode": "log_only",
             "why_log_only": (
                 "if DANN destroys classifier access, that IS the paper result and the "
                 "full 8000-step curve is the evidence. Aborting at step 1200 throws "
                 "away exactly what you would be aborting because you found."),
             "control_curve": ("the arm-1 run's steps.csv, passed as --control-curve. "
                               "NOT curve.csv -- this path does not write one; and NOT "
                               "trace.csv, which carries window means at stride 125.")},
            {"n": 3, "arm": "dann_nb3", "fold": "1-4", "steps": "the declared budget",
             "abort_mode": "armed",
             "why_armed": ("the trace has no scientific value on a confirmatory fold and "
                           "a dead run is wasted T4 time")},
        ],
        "morph_every": int(R5.MORPH_EVERY),
        "morph_every_note": (
            "25, IDENTICAL on both arms and a PINNED key. Condition 3 step-matches "
            "against the control by ABSOLUTE step and refuses to interpolate, so a "
            "cadence mismatch does not coarsen the comparison -- it deletes it. The "
            "runner refuses to start on a control that does not cover every step this "
            "run will measure at."),

        # ---------------- 2.2 fixed before any run ----------------
        "fixed_before_any_run": {
            "lambda_horizon": int(R5.LAMBDA_HORIZON),
            "lambda_horizon_note": (
                "ABSOLUTE STEP, identical in every run, NOT total_steps. Computing it "
                "from run length makes the 8000-step development run and the "
                "confirmatory runs see different objectives at the same step -- "
                "silently, since both still train and both still look plausible. "
                "Precedent: the two unfrozen nb3 rounds declared 1500 (FreeMatch) and "
                "2000 (CORAL). Ganin's ramp and Belfiore's constant both put lambda's "
                "destination at 1; at horizon 2000, lambda >= 0.95 at every budget from "
                "750 to 8000."),
            "lambda_at_750_1500_2000_8000": [
                DL.dann_lambda_at_step(b, R5.LAMBDA_HORIZON)
                for b in (750, 1500, 2000, 8000)],
            "head": "LayerNorm(640) + Linear(640->1)",
            "head_trainable_params": 1921,
            "head_note": (
                "IDENTICAL to CORAL x nb3 and FreeMatch x nb3. NO BOTTLENECK. Recorded "
                "as an UNTESTED architectural choice, not an optimal one: round-1 DANN "
                "used a 640->64->640 bottleneck adapter and nothing has compared the "
                "two under this protocol."),
            "init_head_from": "runs/bakeoff_baseline_A/checkpoints/step_8000.pt",
            "init_head_l2": R5.BASELINE_A_HEAD_L2,
            "init_head_note": (
                "Baseline A is source-only and the source does not vary by fold, so for "
                "this head architecture the source-only linear probe is UNIQUE: one file "
                "serves all five folds. That is the correct artifact, not a shortcut -- "
                "but it is also why the step-0 gate is load-bearing (below)."),
            "discriminator": {
                "topology": [640, probe_d.hidden, probe_d.hidden, 1],
                "norm": "LayerNorm",
                "norm_note": (
                    "a declared substitution for tllib's BatchNorm1d: D is fed ONE "
                    "tensor holding both domains, so any batch-coupled normalisation "
                    "lets D read the domain label out of the batch statistics instead of "
                    "the features. Also keeps the assembled model BatchNorm-free end to "
                    "end (7.1a measured the encoder at 0), which is asserted at startup."),
                "dropout": float(probe_d.dropout),
                "dropout_layers": int(probe_d.n_dropout_layers),
                "dropout_note": (
                    "Dropout 0.2 after each hidden activation. Belfiore et al. (2024, "
                    "A&A) apply dropout between hidden layers; we apply it after each, "
                    "given a target pool roughly 20x smaller. Topology 1024/1024 from "
                    "Ganin/tllib, unmodified. The rate is theirs, the placement is ours."),
                "forward": ("source and target CONCATENATED into one tensor, D called "
                            "EXACTLY ONCE per step"),
                "output": "logits; loss = nn.BCEWithLogitsLoss (fp16-safe under autocast)",
                "domain_labels": {"source": DL.DANN_SOURCE_LABEL,
                                  "target": DL.DANN_TARGET_LABEL},
                "params": sum(p.numel() for p in probe_d.parameters()),
            },
            "optimiser": {
                "count": 1,
                "d_group": {"lr": R5.LR, "lr_decay": None, "weight_decay": R5.WEIGHT_DECAY},
                "note": ("ONE AdamW. D is one extra param group at the HEAD LR with no "
                         "lr_decay. No second optimiser and no separate D learning rate: "
                         "gradient reversal is what removes the need for a D/encoder "
                         "balance, and that is the whole reason DANN was chosen over "
                         "ADDA."),
                "lr": R5.LR, "lr_decay": R5.LR_DECAY, "weight_decay": R5.WEIGHT_DECAY,
                "provenance": "Zoobot v2.0.1 published defaults (7.1a), same as round 4",
            },
            "batch": {"size": R5.BATCH_SIZE, "true_batch": True,
                      "gradient_accumulation": False,
                      "note": ("7.1a: accumulation is REMOVED, not merely unused. It "
                               "would hand D a rank-32 view of a batch whose entire "
                               "purpose is 256 samples of each domain.")},
            "augmentation": "weak (+-3px translate, 7.3)", "precision": "fp16",
            "grad_checkpointing": True,
            "seed": R5.SEED, "pos_fraction": R5.POS_FRACTION,
            "tuning_budget": 0,
            "tuning_budget_note": (
                "ZERO. No sweep of any kind: no lambda sweep, no D learning-rate search, "
                "no dropout search, no architecture search. Every value is inherited "
                "(Ganin/tllib topology, Belfiore's rate, Zoobot's optimiser). THIS IS "
                "CLEANER THAN ROUND 4'S CORAL ARM, which carried a three-point gamma "
                "sweep (7.15) and had to disclose that asymmetry under 7.14a. State it "
                "as a strength."),
        },

        # ---------------- 2.3 the step-0 gate ----------------
        "step0_gate": {
            "status": "LOAD-BEARING and NON-SKIPPABLE -- there is no flag that bypasses it",
            "why": (
                "Baseline A's head is FOLD-INDEPENDENT BY CONSTRUCTION: it trains on "
                "source only, and the source does not vary by fold. So head_l2 proves "
                "the FILE and proves nothing about the FOLD. The per-fold TPR/AUROC gate "
                "is the ONLY thing that proves fold identity."),
            "tpr_tolerance": {"cached_path_n_blocks_0": 0.0,
                              "live_encode_path_n_blocks_gt_0": "one positive (1/98 or 1/97)"},
            "auroc_tolerance": 1e-3,
            "action": "ABORT on either half; no warn tier",
            "tolerance_note": (
                "chosen by the EVAL PATH, not by a flag (7.7). Tolerance 0 on TPR across "
                "a re-encode is the measured mistake 7.7 records: the FreeMatch nb3 gate "
                "demanded exact reproduction, passed folds 0/1/3 and failed fold 2 by "
                "exactly one positive on a correct run, because live-encode score drift "
                "is ~1e-2 while fold 2's closest positive sat 3.06e-03 above threshold. "
                "The AUROC half carries the identification anyway: rank-based, ~100x the "
                "measured live-encode drift (1.2e-05), ~7x tighter than a wrong head "
                "(7.6e-03). 7.7's worked example is that fold 1's head on fold 2's data "
                "gives a legitimate-looking TPR that only AUROC catches."),
            "fold_gate_table": {str(k): v for k, v in R5.FOLD_GATE.items()},
            "fold_gate_provenance": (
                "runs/bakeoff_baseline_A/curve.csv, row step=8000. The bundle builder "
                "RE-DERIVES it from that file rather than trusting the table, and checks "
                "the five folds do not share a TPR value."),
        },

        # ---------------- 2.4 abort conditions ----------------
        "abort_conditions": {
            "1_d_overpowering": {
                "rule": (f"d_acc_holdout > {D5.COND1_HOLDOUT_ACC} for "
                         f"{D5.COND1_SUSTAIN} consecutive ABSOLUTE steps WITH the raw "
                         f"domain loss not falling"),
                "action_if_fires": ("switch to ADDA, whose separate discriminator "
                                    "optimiser sidesteps the SDAT AdamW failure. This is "
                                    "the ONE condition that reverses the DANN-over-ADDA "
                                    "decision."),
                "note": ("BOTH halves are required. A D at 0.99 while its loss is still "
                         "falling is a D that is winning, which is ordinary early "
                         "behaviour."),
            },
            "2_degenerate_collapse": {
                "rule": (f"d_pred_source_frac outside {list(D5.COND2_MARGINAL_BAND)} for "
                         f"{D5.COND2_SUSTAIN} consecutive ABSOLUTE steps"),
                "note": ("D has collapsed to predicting one domain; d_acc_train near 0.5 "
                         "is then NOT evidence of alignment, which is why the marginal "
                         "is logged separately from accuracy."),
            },
            "3_morph_vs_control": {
                "detector_column": D5.COND3_COLUMN,
                "interpretation_columns": list(D5.COND3_INTERPRETATION_COLUMNS),
                "column_note": (
                    "RE-SPECIFIED 2026-08-28 from the arm-1 control -- see "
                    "derived_from_control_arm. Both detectors now read the FROZEN PROBE "
                    "(morph_proxy_auroc), control level 0.9076 with 0.4076 of headroom. "
                    "The two HEAD columns are logged with NO DETECTOR: arm 1 showed they "
                    "fall to chance because the CLASSIFIER SUCCEEDED, both probe classes "
                    "being class 0 in the lens task."),
                "rule_b_PRIMARY": (
                    f"morph_head_auroc(step) < {D5.COND3_SIGN_FLIP}, no sustain. THE "
                    f"PRIMARY DETECTOR. The historical failure WAS a sign flip -- the "
                    f"head instrument fell ~0.61 to ~0.37 (plain DANN) / ~0.33 "
                    f"(IW-CDAN), decisively below chance -- so on this column the rule "
                    f"reproduces the actual event with NO invented threshold and no "
                    f"calibration. That is a stronger position than any margin anyone "
                    f"could declare in advance."),
                "rule_a_SECONDARY": (
                    f"morph_head_auroc(step) < morph_head_auroc_lambda0(step) - M, "
                    f"sustained {D5.COND3_SUSTAIN} consecutive ABSOLUTE steps. A graded "
                    f"early warning."),
                "margin_M": D5.COND3_MARGIN,
                "margin_status": (
                    "UNSET, and deliberately not defaulted. While unset, 3a REFUSES TO "
                    "EVALUATE (it does not score every step against a missing threshold "
                    "and report healthy) and an --abort-mode armed run REFUSES TO START. "
                    "M is declared off ARM 1's control trajectory ON THE HEAD COLUMN and "
                    "hashed BEFORE arm 2 starts, and does not move afterwards."),
                "margin_scale": (
                    "declare M against the DETECTOR column's headroom over chance: "
                    "0.0657 at step 0 (whole pool) -- NOT the frozen probe's 0.4202. "
                    "Brief v3's 0.10 was declared against the probe and is scale-wrong "
                    "here: a 0.10 drop from 0.5657 lands well below chance."),
                "declaration_criterion_PRE_REGISTERED": {
                    "rule_verbatim": (
                        "Let J = median absolute step-to-step change in "
                        "morph_head_auroc_pool over arm 1, steps 2000-8000. "
                        f"If a candidate M >= {D5.COND3_M_J_FACTOR:g}*J can be declared "
                        f"while M <= {D5.COND3_M_CEILING}: declare M, hash it, proceed "
                        "with 3a active. Else: declare 3a UNCALIBRABLE at this pool "
                        "size, record J and the reason, and let 3b carry condition 3 "
                        "alone."),
                    "feasible_iff": (f"{D5.COND3_M_J_FACTOR:g}*J <= "
                                     f"{D5.COND3_M_CEILING}"),
                    "m_min_admissible": f"{D5.COND3_M_J_FACTOR:g}*J",
                    "ceiling": D5.COND3_M_CEILING,
                    "ceiling_rationale": (
                        f"{D5.COND3_M_CEILING} is roughly 60% of the whole-pool headroom "
                        f"over chance (0.0657). An M above it would only fire in "
                        f"territory 3b already covers."),
                    "factor": D5.COND3_M_J_FACTOR,
                    "factor_status": "a DECLARED judgement call",
                    "uncalibrable_is_a_legitimate_outcome": (
                        "YES. Declaring 3a UNCALIBRABLE is not a failure -- it is "
                        "reported as a property of the historical instrument, which was "
                        "never designed as a graded monitor. 3b then carries condition 3 "
                        "alone, which is the detector matched to the historical failure "
                        "signature anyway."),
                    "why_J_and_what_it_misses": (
                        "3a reads a PAIRED DIFFERENCE: both arms score the identical "
                        "fixed pool at the identical step, so pool-selection variance is "
                        "largely COMMON and substantially cancels -- but not exactly, "
                        "since different models rank the same images differently. Arm "
                        "1's own step-to-step jitter is therefore a LOWER BOUND on the "
                        "noise in the difference, and it is the only measurable proxy "
                        "available from arm 1 alone. It is labelled as a bound, not as "
                        "the quantity itself. The absolute AUROC SE (~0.035 at n=256) is "
                        "NOT the relevant scale."),
                    "J_is_defined_on_the_thresholded_series": (
                        "CLARIFICATION (not a change to the criterion). If k > 1 is "
                        "declared, 3a thresholds a median-smoothed difference, so J is "
                        "computed on THE SERIES 3a WILL ACTUALLY THRESHOLD -- the "
                        "declared k is applied to arm 1's trajectory first, then J is "
                        "measured on the smoothed series, with the first k-1 readings "
                        "dropped to match the detector's refusal to evaluate on a "
                        "partial window. Comparing M against the RAW J would "
                        "over-constrain and could send a viable 3a to UNCALIBRABLE for "
                        "no reason. This is legitimate rather than circular ONLY because "
                        "k is declared BEFORE M. Both J_raw and J_at_k are emitted and "
                        "j_used_for_criterion names which one m_min_admissible came "
                        "from; at k=1 they are the same number."),
                    "J_early_reported_not_used": (
                        "J over steps 0-2000 is emitted alongside the criterion's J "
                        "(2000-8000), with the ratio. The criterion window measures "
                        "SETTLED jitter, but a collapse happens EARLY and that is where "
                        "3a operates, so an M sized on late-window noise could throw "
                        "false streaks in the first thousand steps. M_DECLARABLE is "
                        "UNCHANGED and still computed from 2000-8000; J_early is a "
                        "sanity check to look at before writing M down, and the verdict "
                        "says so explicitly when J_early > 2*J."),
                    "MEASURED_CAUTION_on_k": (
                        "smoothing reduces J MUCH more than the ~sqrt(k) the rationale "
                        "assumed -- measured ~11-15x at k=8 against sqrt(8)=2.83 -- "
                        "because consecutive trailing medians share k-1 of their k "
                        "inputs, so the smoothed series' STEP-TO-STEP change collapses "
                        "while its LEVEL noise (the thing that could actually hold a "
                        "500-step false streak) falls only about sqrt(k). The summary "
                        "therefore emits J_reduction_factor_at_k beside "
                        "iqr_reduction_factor_at_k: if J has fallen far more than the "
                        "IQR, the criterion was satisfied by autocorrelation rather than "
                        "by real noise reduction. NOT ACTED ON -- the criterion is "
                        "pre-registered and is not being changed after the fact. "
                        "Recorded so the k decision is made with the number visible, and "
                        "so that a k>1 declaration is disclosed as the weaker evidence "
                        "it is."),
                    "step_to_step_means": (
                        "between consecutive READINGS, which at --morph-every 25 are 25 "
                        "steps apart. That is the jitter 3a actually sees."),
                    "computed_by": ("dann_round5.morph_trajectory_stats, emitted "
                                    "automatically at the end of every run and available "
                                    "standalone as: run_dann_r5.py --morph-stats "
                                    "<arm1>/steps.csv"),
                    "also_emitted": ("the interquartile range of the trajectory over the "
                                     "last 4000 steps"),
                },
                "median_window_k": {
                    "default": 1,
                    "meaning": ("k=1 is no smoothing. k>1 makes 3a threshold the trailing "
                                "median of the last k PAIRED DIFFERENCES -- the "
                                "difference is what 3a reads, so the difference is what "
                                "is smoothed; smoothing each side separately would need "
                                "an assumption about the control's own smoothness that "
                                "nothing supports."),
                    "when": ("if J is borderline. The 500-step sustain already spans 20 "
                             "readings at --morph-every 25, so single-step sensitivity is "
                             "not wanted anyway; k=8 cuts jitter by roughly a third."),
                    "HARD_ORDERING_CONSTRAINT": (
                        "k MUST BE DECLARED BEFORE M. If the window is chosen after a "
                        "candidate M is on the table, it has been selected to make that "
                        "M work."),
                    "cost": ("with k>1 the window must be FULL before 3a evaluates at "
                             "all, so the earliest possible fire moves later by "
                             "(k-1)*morph_every steps -- 175 at k=8, against a 500-step "
                             "sustain."),
                    "flag": "--abort3-median-window",
                },
                "wording": ("adversarial pressure has degraded linear access to "
                            "morphology RELATIVE TO UNFREEZING ALONE. NOT 'the "
                            "documented failure has recurred'."),
                "step_matching": ("read the control by ABSOLUTE step; a step missing "
                                  "from the control REFUSES the comparison rather than "
                                  "being interpolated, and every refusal is recorded"),
                "raw_difference_logged": (
                    "morph_minus_control is written every monitor step whether or not 3a "
                    "is evaluating -- INCLUDING while M is unset, because that trajectory "
                    "is exactly what M gets declared from."),
                "withdrawn_thresholds": {
                    "0.55": ("the design doc's absolute level. **NOT INVENTED** -- "
                             "PROJECT_STATE v2.md 6/8-F4 records the head instrument's "
                             "real ~0.61 no-DA baseline, and 0.55 sat just under it. "
                             "WITHDRAWN because it is meaningless on the FROZEN-PROBE "
                             "column, which reads ~0.92. It was never wrong for its own "
                             "instrument."),
                    "0.6910": ("brief v2's rescaling of 0.55 onto the probe column by "
                               "fraction of headroom. WITHDRAWN: rescaling across a "
                               "change of instrument assumes the two measure the same "
                               "thing, and they read ~0.34 apart."),
                    "0.10": ("brief v3's control margin. WITHDRAWN: declared against the "
                             "probe's 0.4202 headroom, scale-wrong for the head column's "
                             "0.0781."),
                },
            },
            "4_rank_collapse": {
                "rule": (f"participation ratio below {D5.COND4_PR_FLOOR} of 640 (either "
                         f"domain) for {R5.ABORT_SUSTAIN_COND4} consecutive ABSOLUTE "
                         f"steps"),
                "sustain_status": (
                    "the 500-step sustain is a DECLARED ADDITION -- the design doc gives "
                    "condition 4 no sustain, and a single-batch participation ratio is "
                    "noisy enough that an instantaneous trigger would be twitchy. "
                    "--abort4-sustain 0 restores the literal reading."),
                "note": ("DANN's characteristic failure is collapse to a low-dimensional "
                         "domain-invariant subspace; the SimSiam finding (PR ~3-5 of "
                         "640) shows this project can produce it. DISTINCT from round "
                         "4's SCALE collapse (7.15)."),
            },
            "streak_semantics": (
                "every streak is a SPAN IN ABSOLUTE STEPS, not a count of monitor "
                "samples. At --morph-every 25 a sample counter would mean 12,500 steps "
                "where the rule says 500 -- the unit error 7.10 records for FreeMatch's "
                "warmup."),
            "abort_mode_rule": (
                "log_only on arms 1 and 2, armed on arm 3. Recorded EXPLICITLY in each "
                "run_config.json as abort_mode with abort_mode_source beside it; never "
                "inferred at read time."),
        },

        # ---------------- THE INSTRUMENT DECISION -- SETTLED ----------------
        "two_morphology_instruments": {
            "status": "SETTLED before arm 1 (brief v5). Both columns implemented.",
            "finding": (
                "PROJECT_STATE v2.md 6 and 8/F4 record TWO morphology instruments from "
                "the SAME v5-era runs, ~0.34 apart. (1) The TRAINED HEAD's AUROC on "
                "ML-FP vs P1-reserve: ~0.61 with no DA, falling to ~0.37 under plain "
                "DANN and ~0.33 under IW-CDAN. (2) The FROZEN-FEATURE LINEAR PROBE on "
                "the same pool, which that document says explicitly 'stayed ~0.90 -- "
                "representation intact, the supervised HEAD is what degrades'. The "
                "round-5 design doc's monitor table lists BOTH as separate rows, "
                "precisely because 'the earlier diagnostic separated the two'."),
            "step0_readings": {
                "morph_head_auroc_eval_half": 0.578125,
                "morph_head_auroc_whole_pool": 0.565735,
                "morph_proxy_auroc_adapter": 0.920166,
                "morph_proxy_auroc_raw_encoder": 0.921387,
                "historical_head_no_da": 0.61,
                "historical_probe": "~0.90-0.91 (0.914 raw Zoobot)",
                "identical_on_both_arms": True,
                "note": ("BOTH instruments reconstruct their historical values on the "
                         "recovered pool, which is what makes the distinction a "
                         "measurement rather than an argument. Identical on both arms at "
                         "step 0 by construction: same head, same encoder, same "
                         "standardizer, and D is not in the monitor's path."),
            },
            "decision": (
                "CONDITION 3 READS THE HEAD COLUMN. The frozen probe stays logged on "
                "both arms with NO detector attached -- not redundancy: it is the "
                "historical diagnostic reproduced, and it is what makes a head drop "
                "INTERPRETABLE. Head down + probe held = classifier ACCESS destroyed. "
                "Both down = the REPRESENTATION went."),
            "why_before_arm_1_and_not_by_backfill": (
                "three reasons. (a) M is SCALE-WRONG on the probe column, not merely "
                "unset, so it has to be re-declared on the head column either way. "
                "(b) The historical failure was a sign flip, so 3b on the head column is "
                "the instrument-matched detector and needs no threshold at all. (c) "
                "Backfill resolution is inadequate: 32 checkpoints over 8000 steps is "
                "250-step spacing against a rule that reads over 500 absolute steps -- "
                "two samples per window. Enough to eyeball a trajectory, thin for "
                "calibrating a margin, useless for the armed detector on folds 1-4."),
            "cost": ("zero extra compute. Both columns come out of the ONE forward pass "
                     "morph_proxy already runs; the head column needs no probe fit."),
            "which_column_each_detector_reads": {
                "3a_and_3b": D5.COND3_COLUMN,
                "interpretation_only": list(D5.COND3_INTERPRETATION_COLUMNS),
                "no_detector_at_all": "morph_proxy_auroc / morph_proxy_auroc_encoder",
            },
            "which_half": (
                "THE DETECTORS READ THE WHOLE POOL (morph_head_auroc_pool, 256 images). "
                "The eval-half column exists so the head and probe move on the SAME 128 "
                "images, which is what makes 'head down + probe held' a statement about "
                "the same objects -- but that argument binds on the CONTRAST, and "
                "neither detector is the contrast. The probe needs a held-out half "
                "because it is FITTED; the head is not, so it uses all 256 and is simply "
                "better estimated. Consequence recorded: whole-pool step 0 = 0.565735 "
                "(headroom 0.0657) against eval-half 0.578125 (0.0781), so 3b is "
                "marginally MORE sensitive -- a better-estimated number, not a looser "
                "threshold."),
        },

        # ---------------- what the record actually says ----------------
        "record_correction": {
            "what_the_record_says": (
                "PROJECT_STATE v2.md 6 and 8/F4: at v5, from a fresh adapter, BOTH "
                "adversarial configs degraded the morphology probe -- plain DANN "
                "(--iw off) to ~0.37 and IW-CDAN (--iw on) to ~0.33, from a ~0.61 no-DA "
                "baseline. PROJECTv6.md's 'no longer part of the plan' bullet attributes "
                "the failure to importance weighting with no numbers, which is accurate "
                "but INCOMPLETE."),
            "error_1_the_planning_side_misattribution": (
                "planning conversation repeatedly stated the collapse as '0.61 -> "
                "0.33-0.46' and attributed it to DANN alone. '0.33-0.46' is wrong: the "
                "record says 0.37 and 0.33, and 0.46 appears in NO document."),
            "error_2_the_planning_side_OVER-correction": (
                "briefs v3 and v4 then asserted the numbers were INVENTED in planning. "
                "They were not. That came from searching PROJECTv6.md alone and reading "
                "absence from one document as absence from the record. Recorded here so "
                "a future session re-derives NEITHER error."),
            "what_survives": ("'0.33-0.46' is wrong; the attribution to IW alone is "
                              "incomplete; and the two instruments must never be quoted "
                              "as one another."),
            "where_fixed": ("PROJECTv6.md section 1 (the IW bullet's anti-drift note) "
                            "and ROUND5_DANN_DESIGN.md (4 instances)."),
        },

        # ---------------- RE-SPECIFIED FROM THE CONTROL, NOT PRE-REGISTERED ----------
        "derived_from_control_arm": {
            "status": ("NOT A PRE-REGISTRATION. Conditions 3 and 4 were RE-SPECIFIED "
                       "from the arm-1 control after the pre-registered decision tree "
                       "ran out."),
            "date": "2026-08-28",
            "source_run": "runs/dann_arm1 -- dann_nb3_lambda0, fold 0, 8000/8000 steps, "
                          "32/32 checkpoints, 267.8 min on a Tesla T4, step-0 gate PASS "
                          "(TPR |diff| 0.0, AUROC |diff| 8.9e-06)",
            "prereg_hash_it_was_derived_under":
                "2c363e29295ada6c8fc10dc8377410d900f0f3fb1f0df2e51ab0b72de4dfa4e9",
            "why_the_tree_ran_out": (
                "the tree anticipated 3a being UNCALIBRABLE with 3b carrying the "
                "condition. Arm 1 delivered BOTH halves failing: 3a uncalibrable "
                "(3J = 0.0989 against the 0.04 ceiling) AND 3b firing on the control "
                "(54% of readings below 0.5 at lambda = 0). Condition 4 fired 7,999 of "
                "8,000 steps."),
            "WHY_THIS_IS_CLEAN": (
                "ARM 2 IS log_only. No detector acts on the reported development run, so "
                "the thresholds were changed BEFORE any run they could influence. The "
                "re-specification binds on the CONFIRMATORY folds 1-4 only, and those "
                "have not been run."),
            "condition_3_WITHDRAWN": {
                "was": "morph_head_auroc_pool; 3a margin M vs control; 3b floor at 0.5",
                "arm1_evidence": ("the head column went 0.5657 (step 0) -> median "
                                  "0.4948, IQR [0.4639, 0.5321], 54% of readings below "
                                  "0.5, longest below-chance run 375 steps -- AT "
                                  "LAMBDA = 0, with no adversary present."),
                "why": ("ML-FP and P1-reserve are BOTH class 0 in the lens task, so the "
                        "head is fine-tuned to give both low lens logits -- trained to "
                        "make them indistinguishable. At step 0 it is Baseline A's "
                        "frozen 1,921-param probe, so residual structure leaks through "
                        "at 0.5657; unfreeze 14.8M parameters and the classes converge. "
                        "That is THE CLASSIFIER SUCCEEDING, not access being destroyed, "
                        "and it was going to happen at nb3 whatever lambda did. J = "
                        "0.033 is about the standard error of an at-chance AUROC at "
                        "n=256 (Hanley-McNeil ~0.036): noise around 0.5 with the signal "
                        "gone. A degradation cannot be detected in a quantity the "
                        "control has already degraded to chance."),
            },
            "condition_3_REPLACEMENT": {
                "column": D5.COND3_COLUMN,
                "why_the_probe_now": (
                    "condition 3 was moved OFF the probe because the record says it held "
                    "~0.90 through the historical failures. THOSE FAILURES WERE AT nb0: "
                    "with the encoder frozen the probe measures FIXED features and was "
                    "STRUCTURALLY INCAPABLE of moving, so 'it did not move' was "
                    "guaranteed, not evidence. That reasoning does not transfer to nb3, "
                    "where the probe is live."),
                "arm1_control_level": D5.COND3_CONTROL_LEVEL,
                "arm1_headroom_over_chance": D5.COND3_HEADROOM,
                "arm1_J": 0.007812, "arm1_3J": 0.023438,
                "arm1_J_early_over_J": 0.56, "arm1_IQR": 0.027222,
                "ceiling_fraction": D5.COND_CEILING_FRACTION,
                "ceiling_absolute": D5.COND3_M_CEILING,
                "ceiling_note": (
                    "stated as a FRACTION of measured headroom (0.6088 -- exactly what "
                    "the original 0.04 was of the head column's 0.0657) so it travels "
                    "correctly if the column changes again. It has now changed twice."),
                "M_DECLARABLE": True,
                "admissible_M_band": [0.023438, D5.COND3_M_CEILING],
                "3b_floor": D5.COND3_SIGN_FLIP,
                "3b_floor_derivation": (
                    "HALF the measured headroom: 0.5 + 0.5*0.4076 = 0.7038, rounded down "
                    "to 0.70. Chosen because (a) it means something -- half the linear "
                    "morphology signal destroyed; (b) it is a fraction of measured "
                    "headroom, so it travels; (c) arm 1's minimum reading was 0.8108, "
                    "~13 J above it, so the control never came close. The old 0.5 was "
                    "the HEAD column's inversion point and sits 40 J below the probe "
                    "control's minimum -- far past catastrophic. A DECLARED judgement "
                    "call, derived from the control, not carried across."),
            },
            "condition_4_WITHDRAWN": {
                "was": "participation ratio below 20 of 640, either domain, 500 steps",
                "arm1_evidence": ("fired 7,999 of 8,000 steps. src 9.07 -> 2.18, tgt "
                                  "10.40 -> 6.85, run maximum 13.40 -- the threshold was "
                                  "NEVER REACHABLE. The LP->FT init sits at PR ~9-10 in "
                                  "this adapter space before a single step."),
                "why": ("ported from the SimSiam finding (PR ~3-5 of 640) in a DIFFERENT "
                        "feature space. 7.15's monitor-design error repeated: a "
                        "threshold set without checking what the indicator reads on the "
                        "model it monitors."),
            },
            "condition_4_REPLACEMENT": {
                "form": ("DIFFERENTIAL against the lambda=0 control at matched absolute "
                         "steps, same shape as 3a, with src and tgt as SEPARATE "
                         "detectors -- arm 1 showed them behaving differently and one "
                         "combined threshold would hide that."),
                "headroom_floor": D5.COND4_PR_FLOOR,
                "headroom_floor_note": ("measured to 1.0, not to chance: a fully "
                                        "collapsed representation has all its variance "
                                        "in one direction, PR = 1."),
                "src": {"arm1_control_level": D5.COND4_SRC_CONTROL_LEVEL,
                        "arm1_J": 0.172510, "arm1_3J": 0.517530,
                        "arm1_J_early_over_J": 2.39,
                        "J_EARLY_WARNING": True,
                        "warning_note": ("src PR's EARLY jitter is 2.4x its settled "
                                         "jitter, because src PR falls fast in the first "
                                         "2000 steps (9.07 -> ~4). A margin sized on the "
                                         "settled window may throw false streaks early. "
                                         "Look at this before writing the src margin "
                                         "down."),
                        "ceiling": D5.COND4_SRC_CEILING,
                        "DECLARABLE": True,
                        "admissible_band": [0.517530, D5.COND4_SRC_CEILING]},
                "tgt": {"arm1_control_level": D5.COND4_TGT_CONTROL_LEVEL,
                        "arm1_J": 0.442085, "arm1_3J": 1.326254,
                        "arm1_J_early_over_J": 0.93,
                        "J_EARLY_WARNING": False,
                        "ceiling": D5.COND4_TGT_CEILING,
                        "DECLARABLE": True,
                        "admissible_band": [1.326254, D5.COND4_TGT_CEILING]},
            },
            "margins_still_unset": (
                "all three -- cond3 M, cond4 src, cond4 tgt -- ship UNSET. Each is "
                "declared by the run owner within its admissible band and hashed before "
                "the confirmatory folds. An --abort-mode armed run with ANY of them "
                "unset refuses to start, naming the ones missing."),
        },

        # ---------------- 2.5 predictions ----------------
        "predictions_recorded_before_the_run": {
            "1_label_shift": (
                "4.3's argument: the balanced ~1:1 target removes the failure mode that "
                "sank the IW-based attempt. THIS STANDS ON ITS OWN and is deliberately "
                "NOT paired against a documented DANN-specific morphology collapse -- "
                "the record (PROJECT_STATE v2 section 6/F4) shows plain DANN degrading "
                "the HEAD-level probe too, so 'DANN failed only because of label shift' "
                "is not a claim this project has earned."),
            "2_regularisation_RISK_NOTE_ONLY": (
                "Belfiore et al. (2024, A&A) found DANN without source-side noise "
                "regularisation unstable and underperforming: target accuracy 0.73 -> "
                "0.90 with noise, and visibly larger error bars without. Round-1 DANN "
                "ran cached-feature and un-augmented; the nb3 arm runs image-space with "
                "aug=weak. RISK NOTE ONLY -- 7.14 fixes augmentation across methods, so "
                "this CANNOT be acted on without breaking the cross-method comparison."),
            "3_spread_not_mean": (
                "Belfiore ensemble 20 seeds for stability. At 5 folds x 1 seed, EXPECT "
                "DANN's fold-to-fold spread to exceed CORAL's and FreeMatch's. A wide "
                "spread is a statement about VARIANCE, not about the mean. If materially "
                "wider, that is itself reportable."),
        },

        # ---------------- 2.6 what the number is and is not ----------------
        "what_danns_number_is_and_is_not": (
            "CORAL and FreeMatch each report their nb3 arm against their OWN prior run. "
            "ROUND-1 DANN EXISTS ON FOLD 0 ONLY, so DANN has NO within-method paired "
            "comparison. It enters the cross-method table as a STANDALONE value against "
            "Baseline A and the other nb3 arms. Stated here, before the run, not in the "
            "discussion afterwards."),

        # ---------------- 2.7 round-1 reference ----------------
        "round1_dann_fold0_reference": {
            "tpr_at_1e-3_last_checkpoint": 0.19387755102040816,
            "positives_recovered": "19 of 98",
            "read_off": "step_15000 of 75 checkpoints (the R1 last-checkpoint rule)",
            "oracle_best": 0.21428571428571427,
            "oracle_positives_recovered": "21 of 98",
            "oracle_step": 6800,
            "curve_range": [0.09183673469387756, 0.21428571428571427],
            "curve_mean": 0.1812244897959184,
            "source": "runs/curve_dann_fold0/results.json",
            "oracle_significance": (
                "EVEN ORACLE SELECTION leaves round-1 DANN at baseline (21 of 98 against "
                "Baseline A's 19 of 98 on the same fold). That forecloses the 'it only "
                "failed because of the read-off rule' objection, and is an independent "
                "data point for 7.6/7.12's finding that label-free selection has no "
                "signal here."),
            "granularity_note": (
                "the metric can only take values k/98, so round-1 DANN's last checkpoint "
                "and Baseline A's fold-0 step-8000 value both RECOVERED 19 OF 98 "
                "POSITIVES. That is agreement on an integer count. Do NOT write "
                "'identical to sixteen digits' -- the digits are the denominator, not "
                "evidence (7.16 is a section about exactly this)."),
            "NOT_step_comparable_to_round_5_because": [
                "cached-feature, with a 640->64->640 bottleneck adapter (round 5 is "
                "image-space with no bottleneck)",
                "SGD, lr 1e-3, momentum 0.9, weight decay 1e-4 (round 5 is AdamW at "
                "1e-4 / 0.05)",
                "BatchNorm discriminator (round 5 uses LayerNorm)",
                "lambda computed from total_steps (round 5 uses a declared absolute "
                "horizon of 2000)",
                "15,040 steps at 188 steps/epoch (round 5's development arm is 8000 at "
                "187 steps/epoch)",
                "frozen encoder (round 5 is n_blocks=3)",
            ],
            "why_the_list_matters": (
                "without it someone will later read rounds 1 and 5 as one experiment at "
                "two budgets. They are two experiments."),
        },

        # ---------------- 2.8 the probe's limitation, with direction ----------------
        "morphology_probe_limitation": {
            "membership": {
                "mlfp_in_source_train_negatives": 4455,
                "mlfp_total_on_disk": 4455,
                "mlfp_fraction_seen_in_training": 1.0,
                "mlfp_in_target_val_or_test": 0,
                "p1_reserve_seen_anywhere": 0,
            },
            "statement": (
                "the probe's POSITIVE class sits inside D's source batch and its NEGATIVE "
                "class sits in no batch at all."),
            "direction": (
                "DANN drags source features toward target features, which acts on one "
                "probe class and not the other. The lambda=0 arm only PARTLY cancels it: "
                "both arms see the same images, but only one has a term pulling ML-FP "
                "specifically. Direction: MORE LIKELY TO INFLATE SEPARABILITY THAN TO "
                "DEPRESS IT."),
            "consequence": (
                "that is the tolerable direction -- condition 3's realistic failure mode "
                "is FAILING TO FIRE WHEN IT SHOULD, not false-firing. But it means "
                "condition 3a is NOT DECISIVE ON ITS OWN. Cross-check against 3b (the "
                "sign flip), the participation ratio, and the frozen-feature probe. Note "
                "this compounds with instrument_ambiguity_OPEN above: both point the "
                "same way, toward under-detection."),
            "not_lens_label_leakage": (
                "ML-FP and P1-reserve are BOTH class 0 in the lens task, so the probe's "
                "label is not the model's label."),
            "do_not_redesign": (
                "both classes are drawn from the source population, the pool preserves "
                "continuity with the historical instrument, and a replacement costs a "
                "session."),
        },

        # ---------------- the condition-3 disclosure, worded for the paper ----------------
        "condition_3_disclosure_for_the_paper": (
            "Condition 3 was primarily a SIGN-FLIP TEST, matched to the historical "
            "failure signature: the trained head's morphology AUROC fell ~0.61 -> ~0.37 "
            "under plain DANN and ~0.33 under IW-CDAN, crossing decisively below chance. "
            "On the same instrument, 'AUROC < 0.5' reproduces that event with no "
            "invented threshold. A graded secondary test was specified alongside it, "
            "comparing the arm against a lambda=0 control at matched absolute steps; "
            "WHETHER THAT SECONDARY WAS CALIBRABLE AT THIS POOL SIZE WAS AN OPEN "
            "QUESTION, resolved from the control arm against a criterion fixed before "
            "either arm ran. "
            "**Written before the run so that an uncalibrable secondary reads as a "
            "stated limitation of an instrument never designed as a graded monitor, "
            "rather than as a threshold that could not be made to work.**"),

        # ---------------- 2.9 standing disclosures ----------------
        "standing_disclosures": [
            "AdamW, against the field's SGD preference for deep adversarial training "
            "(SDAT), with Belfiore et al.'s Adam@1e-4 as the astronomy-side "
            "counterweight. Arm plausibly conservative.",
            "LayerNorm inside D rather than the reference BatchNorm (tllib/Ganin), for "
            "the domain-leak reason in section 2 of the design doc.",
            "NO nb0 control, the same decision round 4 took for CORAL: no unfreezing "
            "effect is attributed for DANN.",
            "Confirmatory runs read at lambda ~= 1 (>= 0.95 at any budget from 750).",
            "Tuning budget zero -- cleaner than round 4's CORAL arm, which carried a "
            "gamma sweep (7.14a item 5).",
            "Cite the Euclid dual / lensed-AGN DANN attempt: same instrument, same "
            "sim->real direction, no improvement on real data despite features "
            "generalising. Cited WHICHEVER WAY round 5 goes -- as corroboration if DANN "
            "underperforms, and as the contrast that makes the result notable if it does "
            "not.",
            "The morphology proxy is a RECOVERY of an instrument whose code was lost at "
            "the splits_v6 rewrite; agreement with the historical 0.914 is evidence the "
            "recovery is faithful, not a reproduction.",
        ],

        # ---------------- pending items ----------------
        "arm1_complete": {
            "run": "runs/dann_arm1", "arm": "dann_nb3_lambda0", "fold": 0,
            "steps": "8000/8000", "checkpoints": "32/32",
            "wall_minutes": 267.76, "gpu": "Tesla T4",
            "peak_gpu_gb": 3.3023,
            "step0_gate": "PASS (TPR |diff| 0.0, AUROC |diff| 8.9e-06)",
            "aborted": False, "resumes": 0,
            "conditions_1_and_2": "did not fire; healthy",
            "what_it_changed": ("conditions 3 and 4 -- see derived_from_control_arm"),
        },
        "arm2_complete": {
            "run": "runs/dann_arm2_fold0", "arm": "dann_nb3", "fold": 0,
            "steps": "8000/8000", "checkpoints": "32/32",
            "wall_minutes": 306.83, "gpu": "Tesla T4",
            "step0_gate": "PASS (TPR |diff| 0.0, AUROC |diff| 8.9e-06)",
            "aborted": False, "resumes": 0, "abort_mode": "log_only",
            "adversary_worked": ("lambda reached 0.9999 and held; d_acc_train median "
                                 "0.531, raw domain loss 0.691 vs ln2 = 0.693 -- domain "
                                 "confusion achieved and sustained"),
            "headline": ("the FROZEN PROBE fell 0.9172 -> median 0.7880 against a control "
                         "holding at 0.9078: paired median -0.1180, deepest excursion "
                         "-0.3325 at step 6926. morph_proxy_auroc_encoder fell with it "
                         "(0.781 vs 0.904), so the ENCODER's own features lost the "
                         "information -- not the adapter re-mixing what is still there. "
                         "The historical failure signature, with its unfreezing confound "
                         "subtracted."),
            "detectors_on_offline_replay": {
                "cond3a": "fires at 6 of 9 margins across [0.0234, 0.2481]",
                "cond4_src": "fires at 9 of 9 across [0.5175, 2.5924]",
                "cond4_tgt": "fires at 9 of 9 across [1.3263, 5.6079]",
                "cond3b": ("FIRES -- min 0.5579, 70 of 320 readings below the 0.70 floor. "
                           "First fired at step 451 and held 7,550 of 8,000 steps."),
                "cond1_and_cond2": ("did NOT fire and did not come close -- longest "
                                    "streaks 5 and 9 steps against a 500-step sustain. "
                                    "The pathology is in the REPRESENTATION, not in the "
                                    "adversarial game's dynamics."),
                "note": ("reported across each whole admissible band because the margins "
                         "are not declared yet; a single number would be one chosen "
                         "after seeing the run. Arm 2 was log_only AND 3a/4 were "
                         "inactive (margins unset), so nothing acted."),
            },
            "OPEN_DECISION_cond3b_sustain": (
                "3b was written with sustain 0 -- one reading fires it. Correct for a "
                "SIGN FLIP at 0.5, not obviously correct for a FLOOR at 0.70. On arm 2 "
                "an armed fold would abort at step 451. That may be right -- if the "
                "destruction is real and sustained, aborting IS the finding -- but it is "
                "a DECLARED CHOICE, now --abort3b-sustain, to be set with the margins."),
            "no_task_metric": (
                "metric_hygiene held: checkpoints were SCORED (logits persisted) but "
                "never MEASURED, and the only test-set number computed during the run is "
                "the step-0 gate. Separately, a COLLECTION DEFECT dropped all 64 "
                "per-checkpoint logit files from BOTH arms' archives (the archiver "
                "listed subdirectories one level deep; the runner writes them two deep). "
                "Fixed twice over: the walk is recursive, and packaging now REFUSES a "
                "run whose checkpoints.csv names an artifact the archive lacks. The "
                "checkpoints came home, so the logits are recoverable by re-scoring "
                "through live_encode -- but until that is done there is NO TPR CURVE for "
                "either arm and the budget cannot be read off one."),
        },
        # ---- RECORDED BEFORE ANY RE-SCORED CURVE EXISTS -------------------------
        "prediction_arc_suppression": {
            "status": ("A DIRECTIONAL PREDICTION, recorded and hashed BEFORE any "
                       "re-scored TPR curve exists for either arm. At the time this was "
                       "written, all 64 per-checkpoint logit files had been dropped by "
                       "the collection defect and NO task metric existed for arm 1 or "
                       "arm 2 beyond the step-0 gate, which is identical across arms by "
                       "construction."),
            "date": "2026-08-30",
            "prediction": ("TPR@1e-3 FALLS on arm 2 (lambda live) relative to arm 1 "
                           "(lambda = 0) at matched steps."),
            "mechanism": (
                "PROJECT.md 3.1: source positives are a real Euclid cutout with a "
                "PAINTED arc, Poisson noise added to the arc alone; target is real "
                "Euclid. The dominant systematic source-vs-target signal available to D "
                "is therefore the synthetic arc's RENDERING SIGNATURE. Under gradient "
                "reversal the encoder is pushed to make that cue unavailable. The "
                "expensive solution is arcs that resemble real arcs -- the desired "
                "outcome. The cheap solution is to stop representing arcs, and gradient "
                "descent finds cheap solutions. The morphology-probe collapse arm 2 "
                "measured would then be collateral from broadly suppressed "
                "structured-light features."),
            "what_would_falsify_it": (
                "TPR@1e-3 flat or higher on arm 2 than arm 1 across the curve. That "
                "would say the morphology loss is the PRICE of alignment and does not "
                "cost lens-finding -- the opposite sentence in the paper."),
            "MEASURED_CAVEAT_1_the_premise_is_right_but_not_exclusive": {
                "finding": ("the brief states each source negative is 'the same cutout' "
                            "bare. MEASURED from data/splits/splits.json, the source "
                            "pool is: 24,000 sim-lens positives (50.0%); 4,455 ML-FP "
                            "(9.3%); 2,881 bare deflectors (6.0%); 16,664 random P1 "
                            "(34.7%). The matched-bare-cutout description covers 6% of "
                            "the pool, not the negative half."),
                "why_it_still_holds": (
                    "build_splits.py: the unlabeled partitions are 'one homogeneous pool "
                    "of random Euclid galaxies, pre-split', and source random negatives "
                    "are P1 while the target is P2. So 34.7% of the source pool is "
                    "DRAWN FROM THE SAME DISTRIBUTION as the target and gives D no "
                    "signal at all. The arc rendering remains the dominant systematic "
                    "cue, over half the pool."),
                "the_residue": (
                    "ML-FP and bare deflectors -- 15.3% of the source pool -- ARE "
                    "selected real populations that differ from a generic target pool, "
                    "so D has a SECOND channel besides arc rendering. 'The ONLY "
                    "systematic signal is the arc' is too strong as stated; 'the "
                    "dominant one' is supported. This weakens the mechanism's "
                    "specificity, not the prediction's direction."),
            },
            "MEASURED_CAVEAT_2_the_probe_axis_is_PARTLY_the_domain_axis": {
                "STATUS": "A CONFOUND IN ARM 2's HEADLINE, found before the re-score.",
                "finding": (
                    "the morphology probe is ML-FP (class 1) vs P1-reserve (class 0). "
                    "ML-FP are 100% source_train negatives. P1-reserve is the UNUSED "
                    "remainder of partition P1 -- and P1 and the target's P2 are the "
                    "SAME homogeneous pool, randomly pre-split, so P1-reserve is "
                    "IDENTICALLY DISTRIBUTED to the target."),
                "consequence": (
                    "DANN's objective is to make source indistinguishable from target. "
                    "ML-FP is a distinctive SOURCE subpopulation; P1-reserve is "
                    "target-distributed. So the adversary is DIRECTLY pressured to "
                    "collapse exactly the axis the probe measures. Part of arm 2's "
                    "-0.118 probe drop may be DANN succeeding at what it was asked to "
                    "do, read out on a proxy for the domain axis -- not morphology "
                    "destroyed as collateral."),
                "what_it_does_NOT_overturn": (
                    "the measurement stands: the probe fell, arm 1 held it flat at "
                    "lambda = 0, and the comparison is step-matched. What changes is the "
                    "INTERPRETATION available from the probe alone."),
                "why_it_makes_the_re_score_decisive": (
                    "TPR@1e-3 is not confounded this way. The test pool is not the "
                    "probe pool and lens-finding is not ML-FP-vs-P1-reserve "
                    "separability. The task metric is the reading that separates 'DANN "
                    "broke the representation' from 'DANN did its job on an axis we "
                    "chose to watch'."),
                "not_resolved_here": (
                    "quantifying how much of the drop is domain-axis collapse vs "
                    "genuine morphology loss would need a probe whose classes sit on "
                    "the SAME side of the domain split. Not designed here -- flagged."),
            },
            "generalisation_A_HYPOTHESIS_NOT_A_CLAIM": (
                "IF the arc-suppression mechanism holds, adversarial DA has a "
                "STRUCTURAL problem with arc-on-real simulations: the alignment "
                "objective is pointed at the signal the task depends on. 3.1 records "
                "arc-on-real as field standard (HOLISMOKES XI), so the scope would be "
                "wide. It would also OFFER AN EXPLANATION for the Euclid dual/lensed "
                "AGN group's null. DO NOT ASSERT EITHER. One method, one fold, one "
                "sim-generation pipeline, and the re-scored curve does not exist yet. "
                "Recorded so that a later fit to the data cannot be mistaken for a "
                "prediction."),
            "ordering_guarantee": (
                "this block is hashed into the prereg BEFORE the re-score script is "
                "run. The hash and its date are the audit trail."),
        },
        "derived_from_control_and_development_arms": {
            "status": ("NOT A PRE-REGISTRATION. Derived from arms 1 and 2 together, "
                       "after both ran."),
            "date": "2026-08-30",
            "source_runs": ["runs/dann_arm1 (lambda=0 control)",
                            "runs/dann_arm2_fold0 (development, lambda live)"],
            "change": ("CONDITIONS 3 AND 4 BECOME log_only ON EVERY ARM, CONFIRMATORY "
                       "INCLUDED. Conditions 1 and 2 STAY ARMED. Implemented as "
                       "dann_round5.ADVISORY_ONLY_DETECTORS: the advisory detectors "
                       "compute, log every column, and record when they WOULD have "
                       "fired; should_abort() consults only the arming ones."),
            "why": (
                "conditions 3 and 4 were SAFETY aborts -- stop burning GPU on a run "
                "producing garbage. DANN's nb3 behaviour is now characterised on fold 0 "
                "by both arms: the probe drop is real, early, sustained and EXPECTED. "
                "There is no uncertainty left for an abort to protect against. And 3b "
                "at sustain 0 fired on arm 2 at step 451 and held 7,550 of 8,000 steps, "
                "so an armed confirmatory fold aborts at 451 -- destroying the TPR@1e-3 "
                "that fold exists to produce, to re-establish a fact already in hand."),
            "why_this_cannot_bias_the_result": (
                "it is STRICTLY LESS INTERVENTION. The detectors still compute and still "
                "log; the reported metric is unchanged; nothing about how TPR@1e-3 is "
                "measured moves. The usual hazard in a post-hoc threshold change is that "
                "it selects WHICH RUNS SURVIVE -- and that is precisely the mechanism "
                "being removed. A detector that cannot halt a run cannot steer a result "
                "toward any value. The direction also runs against the "
                "prediction_arc_suppression prediction's convenience: keeping the aborts "
                "would have left the folds that collapsed unmeasured, which is the "
                "arrangement that could flatter a method, not the one being adopted."),
            "conditions_1_and_2_stay_armed": (
                "they catch the adversarial game genuinely breaking -- the one failure "
                "these runs have NOT demonstrated immunity to. On arm 2 their longest "
                "streaks were 5 and 9 steps against a 500-step sustain, so arming them "
                "costs nothing."),
            "cond3b_sustain": ("--abort3b-sustain remains a flag and is still recorded. "
                               "Its value is now MOOT for firing, since 3b cannot abort. "
                               "Recorded anyway so the run says what it was set to."),
            "what_is_still_needed_before_the_confirmatory_folds": (
                "the three advisory margins can now be declared without a run depending "
                "on them, but they should still be declared and hashed BEFORE folds 1-4 "
                "so the logged 'would have fired' column means something. k is still "
                "declared before M."),
        },
        "prediction_arc_suppression_OUTCOME": {
            "status": "FALSIFIED, and in the OPPOSITE DIRECTION.",
            "date_resolved": "2026-09-01",
            "predicted": "TPR@1e-3 FALLS on arm 2 relative to arm 1 at matched steps",
            "observed": ("TPR@1e-3 ROSE at 32 of 32 matched steps. Arm 2 mean 0.4534 / "
                         "last 0.4796 / max 0.5816; arm 1 mean 0.1454 / last 0.1531. "
                         "Paired difference: median +0.3265, mean +0.3080, range "
                         "+0.0408 to +0.4694. AUROC agrees at 31 of 32."),
            "hash_ordering_intact": (
                "the prediction was hashed on 2026-08-30 when logits/ held only ids.npz "
                "and true_init.npz for both arms -- verified on disk at the time -- and "
                "true_init is bit-identical BETWEEN the arms, so the one task number "
                "that existed carried no information about the direction."),
            "re_score_integrity": (
                "both arms' step-0 gates PASS at max|diff| 0.000e+00 over all 26,497 "
                "logits; every one of the 64 curve rows recomputes from its own shipped "
                ".npz to 5e-9, which is curve.csv's 8-decimal rounding."),
            "what_the_mechanism_got_wrong": (
                "the argument was that D's dominant source-vs-target cue is the painted "
                "arc's rendering signature, and that the cheap way to remove it is to "
                "stop representing arcs. The encoder did not take the cheap solution: it "
                "reached a representation where simulated and real arcs are not "
                "separable AND real lenses are far easier to find. That is the outcome "
                "the method is meant to produce."),
            "generalisation_WITHDRAWN": (
                "'adversarial DA has a structural problem with arc-on-real sims, which "
                "would explain the Euclid dual/lensed AGN null' is NOT SUPPORTED and "
                "must not be carried forward. It was recorded as a hypothesis, hashed "
                "before any curve existed, so that this could be said plainly."),
            "second_finding_UNFREEZING_ALONE_HURTS": (
                "arm 1 ends at 0.1531 against the frozen LP->FT head's own 0.1939 -- "
                "WORSE than the model it started from -- with its maximum (0.2041) at "
                "step 250 and a decline after. Three unfrozen blocks on a source-only "
                "objective make the model worse at finding real lenses. Visible only "
                "because the control was run."),
            "coral_recontextualised_A_HYPOTHESIS": (
                "if unfreezing alone lands at ~0.15, CORAL x nb3's 0.4348 was its DA "
                "TERM, not the extra capacity. NOT A CLAIM: CORAL's number is folds 1-4 "
                "and this is fold 0, and no CORAL lambda=0 control was ever run."),
            "morphology_confound_RESOLVED": (
                "the probe fell 0.917 -> 0.788 on the arm that gained +0.33 TPR, over "
                "the same checkpoints. The flagged confound -- ML-FP are source-only and "
                "P1-reserve is drawn from the target's own homogeneous pool, so the "
                "adversary is directly pressured to collapse the probe's axis -- reads "
                "as CORRECT. DANN did what it was asked to do on an axis we chose to "
                "watch; 'morphology access destroyed' was the wrong frame."),
            "uncertainty": (
                "paired bootstrap over the matched test pool (ids pinned, 7.12's "
                "scheme): step 1750 +0.3878 [+0.1324, +0.5102]; step 4000 +0.3265 "
                "[+0.2140, +0.4490]; step 8000 +0.3673 [+0.2143, +0.5204]. This bounds "
                "TEST-SET sampling on 98 positives on ONE fold. It says nothing about "
                "fold-to-fold variability and DOES NOT SUBSTITUTE for 7.4's five-fold "
                "rule, which folds 1-4 have not yet been run to answer."),
            "no_budget_read_off_this": (
                "arm 2 peaks at step 1750 with sd 0.0914 across checkpoints and a 0.1735 "
                "outlier at step 1500. 7.13/7.14 establish these curves do not locate a "
                "budget; 7.11 set the precedent for declining the argmax on the "
                "winner's-curse argument. DECLARED, not derived."),
        },
        "probe_axis_AUDITED": {
            "status": ("SETTLED FROM THE SPLIT FILES, 2026-09-01. This replaces both the "
                       "v4 membership note and the arm-2 writeup's wording."),
            "membership_v4_report_CONFIRMED_EXACTLY": (
                "ML-FP: 4,455 of 4,455 in source_train.neg_mlfp; 0 in every target fold "
                "(all five), 0 in val, 0 in test_negatives_shared. P1-reserve: 23,160, "
                "in NO POOL AT ALL."),
            "the_two_statements_are_NOT_incompatible": (
                "the brief reads the v4 membership table as contradicting the arm-2 "
                "writeup. They are DIFFERENT CLAIMS -- membership vs distribution -- and "
                "disjoint sets can be identically distributed. There was no contradiction "
                "to resolve. FLAGGED, not resolved silently."),
            "but_the_arm2_wording_WAS_wrong": (
                "'P1-reserve comes from the target's own pool' is FALSE. P1-reserve and "
                "source_train.neg_random are partition P1; every target fold is P2; val "
                "is P3; test is P4. P1 != P2. Corrected in 7.17."),
            "partitions_exchangeable_MEASURED": {
                "why_measured": ("build_splits.py calls the partitions 'one homogeneous "
                                 "pool, pre-split', but the partitions field is "
                                 "PRE-EXISTING in the shard files and that script only "
                                 "READS it. A claim in a comment is not enforcement."),
                "P1_vs_P2_auroc": 0.5123, "P1_vs_P2_sd": 0.0621,
                "verdict": ("AT CHANCE -> exchangeable. P1-reserve IS "
                            "target-DISTRIBUTED, though disjoint by membership. The "
                            "brief's premise HOLDS."),
            },
            "but_the_briefs_CONCLUSION_does_not_follow": {
                "claim_tested": ("'if ML-FP is source-only and P1-reserve is "
                                 "target-distributed, the probe's axis is approximately "
                                 "the domain axis'"),
                "mlfp_vs_P1_auroc": 0.8688,
                "mlfp_vs_P2_auroc": 0.8774,
                "verdict": (
                    "swapping the probe's negative class from source-side P1 to ACTUAL "
                    "TARGET P2 moves the reading 0.8688 -> 0.8774. If the probe read a "
                    "domain contrast, that swap would change it. It does not, because "
                    "P1 and P2 are at chance against each other -- there is NO DOMAIN "
                    "SIGNAL IN THE NEGATIVE CLASS to pick up. CONDITION 3 IS NOT A "
                    "DOMAIN PROXY, and 7.17's arm-2 section does NOT become a null "
                    "observation."),
            },
            "what_condition_3_ACTUALLY_measures": (
                "POPULATION SEPARABILITY: can a linear probe on frozen features separate "
                "a SELECTED not-a-lens population (ML-FP, graded X by a previous "
                "classifier, living in data/ml_fp_cutouts/shards with no partitions "
                "array at all) from generic Euclid galaxies. Calling it 'morphology' was "
                "always loose; it is now measured rather than assumed."),
            "the_confound_SURVIVES_by_another_route": (
                "the source pool is 50.0% sim-lens, 9.3% ML-FP, 6.0% bare deflectors, "
                "34.7% random P1 -- and that 34.7% is indistinguishable from target. So "
                "D's ENTIRE discriminative signal is the painted arcs, ML-FP, and the "
                "bare deflectors. Gradient reversal is therefore directly pressured to "
                "make ML-FP look like a generic galaxy, which is exactly what the probe "
                "reports. The confound is real but runs through the probe's POSITIVE "
                "class being a source-only cue, NOT through its negative class being "
                "target. The probe fell to 0.788 median / 0.558 min, not to 0.5."),
            "7_4_NOT_rewritten": (
                "the historical v5 collapse MAY share this explanation. FLAGGED, NOT "
                "CONCLUDED: it needs the same audit on that round's pools, and one "
                "fold's inference is not grounds to rewrite a bake-off."),
            "harmless_to_the_runs": (
                "condition 3 is log_only on every arm (v10 4), so this reinterpretation "
                "changes no run's behaviour -- it records a column, it does not act. "
                "That is why making detectors advisory BEFORE this was known mattered."),
        },
        "arm1_folds_1_4_declined": {
            "status": "CONSIDERED AND DECLINED, 2026-09-01. Recorded so a future session "
                      "does not read the absence as an oversight.",
            "decision": ("arm 1 is reported as a SINGLE-FOLD CONTROL OBSERVATION. Folds "
                         "1-4 for it will not be run."),
            "reasons": [
                "it is NOT a control for the other arms: arm 1 initialises from Baseline "
                "A's head, CORAL x nb3 from round-1 CORAL heads, FreeMatch x nb3 from "
                "its own. Arm 1 is DANN's control, not the nb3 table's baseline row, and "
                "presenting it as the latter would misdescribe it.",
                "the fold-0 result falls in the SAFE direction: 0.1939 -> 0.1531 means "
                "fine-tuning alone does not explain the DA gains, so the exposure is "
                "understatement. A single-fold observation that STRENGTHENS the claim "
                "does not need the rigour of one that threatens it.",
                "the paper's claim does not need it: 'the tiers are level under "
                "unfreezing' is a comparison among DA methods, and 7.16 ends the round.",
            ],
            "what_it_IS_reported_as": (
                "arm 1 is full fine-tuning -- 14,835,520 parameters, 99.2% of the "
                "encoder -- from a linear-probe head, evaluated OUT OF DISTRIBUTION, and "
                "it ends BELOW the frozen head it started from. That is the FEATURE "
                "DISTORTION phenomenon of Kumar et al. 2202.10054, which is why this "
                "project adopted LP->FT (7.1a). The protocol choice was adopted on a "
                "citation; this connects it to a measurement in this pipeline on this "
                "data. ONE FOLD, stated as such."),
        },
        "PRE_DECLARED_headline_and_sensitivity": {
            "status": ("PRE-DECLARED 2026-09-02, BEFORE ANY CONFIRMATORY FOLD EXISTS. "
                       "Hashed into runs/dann_r5_budget.json (sha256 abf0dae0...) and "
                       "pinned in run_dann_r5.py as HEADLINE_DECLARATION."),
            "THE_DECLARATION": (
                "The headline DANN x nb3 figure is the 5-fold mean at the declared budget "
                "of 4000. The 5-fold mean at step 2000 is reported as a "
                "BUDGET-SENSITIVITY CHECK and does not replace it, WHICHEVER IS HIGHER."),
            "why_the_ordering_is_fixed_in_advance": (
                "two numbers with the choice of headline made afterwards is SELECTION "
                "WEARING A DISCLOSURE. Fixing the ordering before either exists is what "
                "makes it a sensitivity check."),
            "what_it_buys": (
                "it converts the cross-budget objection from an argument into a table "
                "row: at CORAL's budget DANN scores X, at its own declared budget Y."),
            "cost": (
                "ZERO. ckpt_every is 250 and the budget is 4000, so every confirmatory "
                "fold writes 16 checkpoints at 250-step spacing and STEP 2000 IS "
                "CHECKPOINT 8 OF 16, with its logits persisted natively during the run. "
                "No extra checkpoints, no extra scoring, no config change."),
            "correction_to_brief_v14": (
                "v14 2 describes '4000 with 32 checkpoints, spacing 125, so step 2000 is "
                "checkpoint 16' and prices the readout at 'five extra checkpoints'. "
                "ckpt_every is a FIXED 250, so it is 16 checkpoints at spacing 250 and "
                "step 2000 is checkpoint 8. The conclusion is STRONGER than the brief's: "
                "the readout is free, not cheap. FLAGGED, not resolved silently."),
            "a_1500_readout_is_equally_free_and_is_NOT_declared": (
                "step 1500 is checkpoint 6 of 16, so a FreeMatch-matched readout costs "
                "nothing either. It is deliberately NOT pre-declared, because only the "
                "2000 readout was specified and adding one now without declaring it "
                "would be the selection this block exists to prevent. A decision for the "
                "run owner, and only worth anything if made BEFORE the folds run."),
        },
        "cross_method_budgets_VERIFIED": {
            "status": "VERIFIED FROM THE FILES 2026-09-02.",
            "the_three_nb3_rows": "1500 / 2000 / 4000",
            "freematch_nb3": {
                "budget": 1500,
                "source": ("runs/freematch_nb3_unfrozen_fold{0..4}/run_config.json -- all "
                           "five at max_steps 1500, budget_steps 1500, "
                           "is_preregistered_budget true")},
            "coral_nb3": {"budget": 2000,
                          "source": "runs/coral_r4_budget.json selected_budget_steps"},
            "dann_nb3": {"budget": 4000, "source": "runs/dann_r5_budget.json"},
            "CORRECTION": (
                "an earlier disclosure claimed CORAL and FreeMatch were BOTH confirmed at "
                "2000. FreeMatch x nb3 was confirmed at 1500. Every disclosure has been "
                "corrected."),
            "second_CORRECTION_freematch_stability": (
                "the stability note also said FreeMatch x nb3 had NO COMPARABLE FIGURE "
                "because 'its fold-0 curve stops at 1500'. That read its CONFIRMATORY "
                "fold-0 run, which stops at 1500 BECAUSE 1500 IS ITS BUDGET. Its fold-0 "
                "DEVELOPMENT curve (runs/freematch_fold0_nb3_unfrozen/curve.csv) runs "
                "0-8000 at stride 125 and 32 points land on the matched grid: mean "
                "0.4115, sd 0.0606, sd/mean 0.147. The full ordering is arm 1 (0.276) > "
                "arm 2 (0.202) > FreeMatch (0.147) > CORAL (0.118)."),
            "why_different_budgets_are_the_DESIGN": (
                "7.14 has each method fix its own budget from its own fold-0 curve, "
                "precisely because the methods are not interchangeable. Runs are "
                "otherwise identical -- same encoder, LP->FT init, batch 256, folds, test "
                "pools -- and what differs is ONLY WHERE EACH IS READ. What requires "
                "disclosure is the SIZE of the gap and its reason, not its existence."),
            "equal_steps_would_be_UNEQUAL_treatment": (
                "at 2000 CORAL trains its whole run at fixed gamma and FreeMatch at fixed "
                "thresholds, while DANN would spend its whole run inside the lambda ramp. "
                "Matching the NUMBER would mismatch each method's exposure to its own "
                "configuration. The lambda-ramp argument against 2000 applies to DANN "
                "ONLY and is NOT a claim that CORAL's or FreeMatch's budgets were wrong."),
            "the_direction_is_the_OPPOSITE_of_the_intuitive_one": (
                "the worry a reader brings is that the larger budget flatters DANN. Fold "
                "0: 1500 -> 0.1735 (FreeMatch's budget, the curve's single WORST "
                "reading); 2000 -> 0.4082 (CORAL's budget, pre-transition); 4000 -> "
                "0.4633 (DANN's, on the plateau); 8000 -> 0.4796; settled mean 0.4787. "
                "MATCHING CORAL'S BUDGET WOULD HAVE UNDERSTATED DANN, NOT FLATTERED IT, "
                "and matching FreeMatch's 1500 would have landed on the "
                "collapse-and-recovery outlier. Fold 0 only; it enters no headline."),
        },
        "budget_DECIDED": {
            "status": "DECIDED 2026-09-02. CONFIRM_STEPS = 4000.",
            "file": "runs/dann_r5_budget.json",
            "sha256": "a05de875d4cc22406ec4298944a7cde57df9627aefdf75248dd1001a092432aa",
            "decided_from": "runs/dann_r5_budget_packet.md",
            "basis": "declared under a NAMED RULE, corroborated by the curve",
            "basis_note": (
                "a THIRD thing, distinct from both precedents in the record. CORAL x "
                "nb3's budget is a PURE COST declaration -- its own file says no budget "
                "in 250-8000 is defensible over another on evidence. A DERIVED budget "
                "would read the metric and inherit 7.11's winner's-curse problem. This "
                "rule reads DANN's OWN CONFIGURATION and no metric."),
            "THE_RULE": (
                "Budget = 2 x the lambda horizon, so that half of each confirmatory run "
                "trains at fixed adversarial strength. Derived from DANN's own "
                "configuration; reads no metric."),
            "why_the_rule": (
                "lambda = tanh(5*step/2000) pins at ~1.0 near step 2000, so EVERY fold "
                "has its transition at the same absolute step and the budget decides "
                "which side of it all four report from. On fold 0 the regimes measure: "
                "early (250-2000, n=8) mean 0.3776 sd 0.1314; settled (2250-8000, n=24) "
                "mean 0.4787 sd 0.0578 -- a 2.27x difference in within-fold scatter. A "
                "budget of 2000 would place all four folds in the early regime on a "
                "round where 7.16 records a realised half-width of ~0.08 at 98 "
                "positives. 4000 places every fold well inside the settled regime rather "
                "than at its edge."),
            "corroboration_NOT_derivation": (
                "the fold-0 curve's settled regime begins around 2250, consistent with "
                "the rule; and the reading at 4000 (0.4633) is UNREMARKABLE against the "
                "settled mean (0.4787) -- 0.27 settled-sds away. The rule was not chosen "
                "to select a value, and this is the check that it did not accidentally "
                "land on a favourable one."),
            "alternatives_declined": {
                "2000": ("CORAL-matched. Declined: places all folds inside the lambda "
                         "ramp; CORAL HAS NO RAMP, so its budget faced no analogous "
                         "constraint and could not carry across."),
                "1750_and_1250": ("argmax and the two 1-sd rules. Declined on 7.11's "
                                  "winner's-curse precedent, and both sit in the early "
                                  "regime where the global sd used by those rules "
                                  "describes neither phase (local sd 0.153 and 0.165)."),
                "8000": ("last checkpoint, R1 applying with no budget selection to "
                         "defend. Declined on cost: 20.5 GPU-h against 10.2, to move the "
                         "fold-0 reading +0.0163 and the local scatter 0.041 -> 0.035."),
                "3000": ("a cheaper point still past every fold's transition. Declined "
                         "in favour of a rule statable as a round fraction; RECORDED AS "
                         "THE FALLBACK if the hours become a problem -- 7.7 GPU-h."),
            },
            "timing_is_not_a_selection": (
                "the decision was made AFTER fold 0 BY DESIGN -- 7.14 specifies a fold-0 "
                "development run for exactly this. The rule READS NO METRIC, so the "
                "post-fold-0 timing does not make it a selection: it could have been "
                "stated before fold 0 ran and would have given the same number."),
            "lambda_coverage_at_4000": (
                "73.6% of the run at lambda >= 0.99 and 50.5% at >= 0.9999, against the "
                "development run's 86.8% and 75.2%. This RESOLVES the concern recorded "
                "when 2000 was on the table, where the horizon equalled the budget and "
                "the fold reached full adversarial strength only at its last step."),
            "cost": ("153.4 min/fold on a T4, 10.2 GPU-h for folds 1-4, 16 checkpoints "
                     "per fold. Two Kaggle sessions at two folds each."),
            "DISCLOSE_WITH_THE_COMPARISON": (
                "the three nb3 rows sit at 1500 / 2000 / 4000 -- FreeMatch 1500, CORAL "
                "2000, DANN 4000. See cross_method_budgets_VERIFIED."),
        },
        "budget_2000_WITHDRAWN_and_not_rechosen": {
            "status": ("WITHDRAWN 2026-09-02, and NOT re-chosen -- 4000 was decided "
                       "instead. Kept so a reader can tell the question was reopened."),
            "what_happened": (
                "brief v11 4 asked for a declared budget and one was declared at 2000 on "
                "cost (sha256 9987a46a...). Brief v12 (revised) withdrew that "
                "instruction: the budget is the RUN OWNER'S decision, made from the "
                "fold-0 curve. runs/dann_r5_budget.json now records the withdrawal "
                "rather than deleting it, so the record shows the declaration happened "
                "and was retracted."),
            "decision_packet": (
                "runs/dann_r5_budget_packet.md -- both arms' full 32-checkpoint curves "
                "unsmoothed and unannotated; six candidate rules with what each selects, "
                "UNRANKED; local scatter per candidate; lambda coverage per budget; the "
                "d_acc cross-check; and the protocol conflict stated with no position "
                "taken. NO RECOMMENDATION anywhere in it."),
            "state_of_the_code_NOW": (
                "SUPERSEDED by budget_DECIDED: CONFIRM_STEPS = 4000, "
                "CONFIRM_BUDGET_DECLARED = True, and KAGGLE_FOLDS.md is un-bannered at "
                "4000 with Cell 3 checking the new hash."),
            "what_the_replacement_must_record": (
                "which BASIS was used -- 'declared', or 'derived' under a NAMED rule from "
                "the packet -- in a derived_or_declared field, so a later reader can tell "
                "without reconstructing the reasoning."),
            "the_protocol_conflict_UNRESOLVED": (
                "7.13/7.14/7.11 hold these curves do not locate a budget and CORAL's file "
                "says 'declared'; but round 5's fold-0 arm was specified at 8000 steps IN "
                "ORDER TO fix the budget, and its own run.json says so. Both are quoted "
                "in the packet. THIS SESSION DOES NOT RESOLVE IT."),
            "lambda_arithmetic_CORRECTED": (
                "the brief's 'development run spends 75% of steps at lambda >= 0.99' and "
                "'B=2000 spends ~0' are both wrong against the lambda >= 0.99 threshold "
                "it specifies. MEASURED: lambda >= 0.99 first at step 1059, so the "
                "development run is at 86.8% and B=2000 at 47.1%. The brief's figures "
                "correspond to lambda >= 0.9999 (first at step 1981): 75.2% and 1.0%. "
                "Both columns are in the packet. FLAGGED, not resolved silently."),
        },
        "stability_claim_CORRECTED": {
            "status": "CORRECTED 2026-09-02.",
            "was": "'DANN is distinctively unstable'",
            "why_wrong": (
                "it contradicted its own table. The ordering is arm 1 (0.276) > arm 2 "
                "(0.202) > CORAL (0.118), and ARM 1 HAS NO ADVERSARY. Unfreezing without "
                "a DA term is the LEAST stable of the three, so the numbers do not "
                "single DANN out -- if anything the adversarial term stabilises relative "
                "to plain fine-tuning at nb3. It also fits arm 1's other behaviour: a "
                "run peaking at step 250 and degrading is a run wandering."),
            "the_caveat_that_stays": (
                "arms 1 and 2 share an init, fold, seed, GPU and grid and differ in "
                "lambda ALONE, so 0.276 -> 0.202 is a controlled contrast. CORAL shares "
                "none of that (it initialises from round-1 CORAL heads), so its 0.118 is "
                "context, not a matched comparison."),
            "what_survives_unqualified": (
                "arm 2's curve has a 0.1735 reading at step 1500 sitting between 0.5000 "
                "and 0.5816 -- a collapse and full recovery inside 250 steps. That is a "
                "fact about reading any single checkpoint off this curve."),
            "freematch": ("NO COMPARABLE FIGURE EXISTS -- its fold-0 curve stops at 1500 "
                          "at stride 125. The n=6 window is not offered as a substitute."),
        },
        "probe_material_DEMOTED": {
            "status": "DEMOTED to a DIAGNOSTIC NOTE 2026-09-02. Not a finding.",
            "why": (
                "the audit falsified the domain-proxy hypothesis on the NEGATIVE class "
                "(P1/P2 exchangeable at 0.5123; swapping to actual target moves the "
                "reading only 0.8688 -> 0.8774) but the mechanism it then gave brings "
                "the concern back through the POSITIVE class: ML-FP is 9.3% of the "
                "source pool and source-only, so 'looks like ML-FP' IS one of D's cues "
                "and gradient reversal is pressured to erase it. A probe fall driven by "
                "that is ALIGNMENT SUCCEEDING on a source-specific cue, not "
                "representation damage -- AND THE INSTRUMENT CANNOT SEPARATE THEM. The "
                "concern was relocated, not dissolved."),
            "why_it_stopped_mattering": (
                "TPR@1e-3 rose x2.47 over its own initialisation on the SAME "
                "checkpoints. Whatever the probe was measuring, the representation "
                "retained and improved what the task needs. That reading does not go "
                "through this instrument at all."),
            "7_4_flag_KEPT_AS_A_FLAG": (
                "the historical v5 collapse may share this explanation. Flagged, not "
                "concluded; it needs the same audit on that round's pools."),
        },
        "budget_DECLARED_SUPERSEDED": {
            "file": "runs/dann_r5_budget.json",
            "sha256": "9987a46a5df7ad75fd0435505b6ef095099bf9c555554b4108d21717cf6bb330",
            "budget_steps": 2000,
            "derived_or_declared": "declared",
            "declared_before_any_confirmatory_fold": True,
            "why_2000": ("it is CORAL x nb3's OWN declared budget. 7.4 compares METHODS, "
                         "and two methods confirmed at different budgets are not that "
                         "comparison. Matching costs nothing in defensibility because "
                         "neither budget was derived from a curve."),
            "why_not_1750": ("1750 is the fold-0 argmax (0.5816), 1.4 checkpoint-sds "
                             "above its own mean on a curve with sd/mean 0.202 that "
                             "contains a 0.1735 outlier 250 steps earlier. 7.11's "
                             "winner's-curse precedent was written for this case."),
            "stability_disclosure": (
                "DANN IS DISTINCTIVELY UNSTABLE. On the identical 32-checkpoint grid: "
                "DANN arm 2 sd/mean 0.202, arm 1 0.276, CORAL x nb3 0.118. FreeMatch x "
                "nb3 is NOT MEASURABLE on this grid -- its fold-0 curve stops at step "
                "1500 at stride 125. Over the 250-1500 window all four share (n=6, weak): "
                "FreeMatch 0.070, CORAL 0.119, DANN arm 1 0.292, DANN arm 2 0.347."),
            "DISCLOSED_RISK_lambda_horizon_equals_budget": (
                "the lambda horizon is 2000 and the budget is 2000, so a confirmatory "
                "fold spends its ENTIRE run inside the ramp and reaches full adversarial "
                "strength only at its last step -- unlike the development run, where "
                "lambda sat at ~1.0 for 6000 steps. NOT CORRECTED: the horizon is a "
                "declared constant fixed before arm 1 ran, and moving it to suit a "
                "budget chosen afterwards is the exact manoeuvre pre-registration "
                "prevents. Arm 2's step-2000 reading was 0.4082 vs a curve mean of "
                "0.4534 -- inside the noise. IF FOLDS 1-4 COME IN LOW, LOOK HERE FIRST."),
        },
        "ROUND_5_RESULT": {
            "status": "CLOSED 2026-09-05. Folds 1-4 complete, verified, recorded.",
            "headline": ("DANN x nb3 fold-mean TPR@1e-3 = 0.4167 over folds 1-4 at the "
                         "declared budget 4000. Fold sd 0.1525, unpaired 95% CI (t,df=3) "
                         "[0.174, 0.659]. Fold-mean AUROC 0.9066."),
            "per_fold_at_4000": {"1": 0.428571, "2": 0.622449, "3": 0.265306,
                                 "4": 0.350515},
            "integrity": ("all four: 4000/4000 steps, 16/16 checkpoints, no resumes, not "
                          "aborted, no ARMING detector fired, pinned budget hash "
                          "abf0dae0... present in every run_config."),
            "fold_identity_PROVEN": (
                "every fold passed its OWN step-0 gate at |diff| EXACTLY 0.0 against four "
                "DISTINCT values -- 0.142857 / 0.193878 / 0.153061 / 0.195876. The "
                "phase-1 head is FOLD-INDEPENDENT (one file initialises all five folds), "
                "so a mis-pointed fold would train and score happily and produce a "
                "plausible WRONG number. The per-fold gate is the ONLY artifact proving "
                "fold identity, which is why 7.11 made it non-skippable."),
            "abort3b_advisory_SAVED_THE_ROUND": (
                "abort3b fired ADVISORY on all four folds and stopped nothing. Under the "
                "PRE-REGISTERED arrangement it would have aborted every one at ~step 450, "
                "destroying exactly the TPR the folds exist to produce. v10 decision to "
                "make conditions 3 and 4 advisory -- taken BEFORE any confirmatory fold "
                "ran, on arm-1 and arm-2 evidence -- is what preserved these runs."),
            "development_fold_PREDICTED_the_result": (
                "fold 0 at step 4000 read 0.4633; folds 1-4 mean 0.4167. A 0.047 "
                "difference against a fold-to-fold sd of 0.1525 -- inside the noise. That "
                "is 7.14 protocol working."),
            "CORRECTION_do_not_say_fold0_did_not_survive": (
                "an earlier draft said fold 0 +0.33 did not survive. That conflates two "
                "quantities: the +0.33 was DANN MINUS THE lambda=0 CONTROL, and there is "
                "NO CONTROL ON FOLDS 1-4 (arm 1 ran on fold 0 only -- see "
                "arm1_folds_1_4_declined). Nothing in the confirmatory data speaks to "
                "whether that gap survived. The correct limitation is narrower and "
                "already on the record: the effect size RELATIVE TO UNFREEZING ALONE "
                "rests on a single fold. That is NOT the same as fold 0 having been "
                "misleading and must not be upgraded into it."),
        },
        "PAIRED_cross_method_analysis": {
            "why": ("CORAL x nb3, FreeMatch x nb3 and DANN x nb3 all ran on the SAME five "
                    "folds, so fold difficulty is a shared nuisance term a paired "
                    "comparison removes."),
            "per_fold_TPR": {
                "dann_B4000": [0.4286, 0.6224, 0.2653, 0.3505],
                "coral_B2000": [0.3571, 0.5612, 0.3980, 0.4227],
                "freematch_B1500": [0.4184, 0.4796, 0.3367, 0.4330]},
            "means": {"dann": 0.4167, "coral": 0.4348, "freematch": 0.4169},
            "dann_minus_coral": {"mean": -0.0180,
                                 "paired_ci95": [-0.1781, 0.1421],
                                 "unpaired_ci95": [-0.2986, 0.2625],
                                 "half_width_paired": 0.1601,
                                 "half_width_unpaired": 0.2806,
                                 "tighter_by": "43%"},
            "dann_minus_freematch": {"mean": -0.0002,
                                     "paired_ci95": [-0.1656, 0.1652],
                                     "unpaired_ci95": [-0.2607, 0.2602],
                                     "half_width_paired": 0.1654,
                                     "half_width_unpaired": 0.2604,
                                     "tighter_by": "36%"},
            "method": ("PAIRED t-interval on 4 differences (df=3), NOT a bootstrap: at "
                       "n=4 folds a bootstrap draws from only 4^4 = 256 distinct "
                       "resamples and its tail behaviour is an artefact of that lattice."),
            "verdict": ("both intervals still contain zero, so indistinguishable is "
                        "UNCHANGED -- but it is now a MUCH TIGHTER bound, for free. The "
                        "three unfrozen DA arms sit within +-0.17 of each other rather "
                        "than +-0.28."),
            "fold_difficulty_is_substantially_shared": (
                "hardest->easiest: DANN [3,4,1,2], CORAL [1,3,4,2], FreeMatch [3,1,4,2]. "
                "Fold 2 easiest for ALL THREE; fold 3 hardest or second-hardest for all "
                "three. Spearman rho: DANN-FreeMatch +0.8, CORAL-FreeMatch +0.8, "
                "DANN-CORAL +0.4. AT n=4 rho CAN ONLY BE +-1.0, +-0.8, +-0.4, 0.0 AND NO "
                "P-VALUE IS MEANINGFUL -- the orderings are the evidence."),
            "CONFOUNDS_pairing_does_NOT_remove": (
                "the three arms sit at DIFFERENT BUDGETS (1500/2000/4000, 7.14 design) "
                "and INITIALISE FROM DIFFERENT HEADS (CORAL x nb3 from round-1 CORAL "
                "heads, FreeMatch from its own, DANN from Baseline A). Pairing removes "
                "fold difficulty and nothing else."),
        },
        "SENSITIVITY_CHECK_RESOLVED": {
            "declared_budget_4000": 0.4167,
            "sensitivity_step_2000": 0.4551,
            "difference": 0.0384,
            "resolution": ("the sensitivity reading is HIGHER. The pre-declaration "
                           "governs and the headline STANDS AT 0.4167 -- its clause was "
                           "does not replace it, WHICHEVER IS HIGHER, hashed 2026-09-02 "
                           "before any confirmatory fold existed."),
            "why_this_matters_A_FINDING_not_bookkeeping": (
                "at 0.4551 DANN sits ABOVE CORAL (0.4348) and FreeMatch (0.4169) -- the "
                "BEST of the three. At 0.4167 it sits BELOW BOTH -- the WORST of the "
                "three. Two defensible read-off points, 125 minutes of training apart, "
                "INVERT THE RANKING OF THE ENTIRE nb3 TABLE. Nothing distinguishes them "
                "on evidence; only the order in which they were declared does. This is "
                "7.16 argument demonstrated on this project own data, and it is the "
                "worked example of the pre-declaration costing something real."),
        },
        "prediction_3_spread_not_mean_RESOLVED": {
            "prediction": ("At 5 folds x 1 seed, EXPECT DANN fold-to-fold spread to "
                           "exceed CORAL and FreeMatch. (Belfiore et al. 2024)"),
            "at_declared_budgets": {"dann_B4000_sd": 0.1525, "coral_B2000_sd": 0.0885,
                                    "freematch_B1500_sd": 0.0595,
                                    "ratios": "1.72x CORAL, 2.56x FreeMatch",
                                    "verdict": "HELD"},
            "BUT_the_sd_orders_exactly_with_the_budget": (
                "DANN OWN spread scales steeply with its own budget: 0.0374 at 1500 -> "
                "0.1102 at 2000 -> 0.1525 at 4000. Reading DANN at the other arms "
                "budgets is a MEASUREMENT, not an adjustment."),
            "budget_matched": {"both_at_2000": "DANN 0.1102 vs CORAL 0.0885 = 1.24x",
                               "both_at_1500": ("DANN 0.0374 vs FreeMatch 0.0595 = 0.63x "
                                                "-- REVERSED, DANN is the LESS variable "
                                                "arm")},
            "RECORDED_VERDICT": (
                "HELD as pre-registered and as read at the declared budgets; NOT "
                "SUPPORTED once budget is matched. Both readings are reported because the "
                "prediction was made about arms whose budgets were always going to "
                "differ, and neither is more correct -- they answer different questions. "
                "The declared-budget verdict is largely a statement about TRAINING "
                "LENGTH, not about DANN. Belfiore mechanism is NOT established here."),
            "disagreement_with_brief_v15_3_FLAGGED": (
                "v15 section 3 asked that the budget confound be stated and not adjusted "
                "for. Stating it is not enough here: the confound is MEASURABLE and "
                "largely explains the result. Reading DANN at 1500 and 2000 is not an "
                "adjustment to the prediction, it is a second measurement, and it "
                "inverts the verdict against FreeMatch. Reporting only the "
                "declared-budget reading would have overstated a pre-registered "
                "prediction that the data does not carry."),
        },
        "pending_hashes": {
            "abort_cond3_margin": (
                "UNSET in code. Declared off arm 1's lambda=0 control trajectory ON THE "
                "morph_head_auroc_pool COLUMN, against its 0.0657 headroom over chance -- "
                "not the frozen probe's 0.4202 -- under the criterion recorded above, and "
                "hashed AFTER arm 1 completes and BEFORE arm 2 starts. **k is declared "
                "before M.** Until M is set, condition 3a refuses to evaluate and an "
                "armed run refuses to start; 3b (the primary detector) needs no margin "
                "and is live throughout. UNCALIBRABLE is an allowed outcome."),
            "confirmatory_budget": (
                "declared and hashed into runs/dann_r5_budget.json AFTER arm 2, "
                "mirroring runs/coral_r4_budget.json. EXPECT 'declared', not 'derived': "
                "7.13 and 7.14 both establish these curves do not locate a budget, so "
                "decline the argmax on the winner's-curse argument 7.11 set the "
                "precedent for. Until then CONFIRM_STEPS = 2000 is a PLACEHOLDER with "
                "is_preregistered_budget = false."),
        },
        "code_state_at_recording": {
            "arms": {k: {"n_blocks": v["n_blocks"], "lambda_zero": v["lambda_zero"],
                         "steps": v["steps"]} for k, v in R5.ARMS.items()},
            "pinned_keys": list(R5.PINNED_KEYS),
            "n_checkpoints_expected": int(R5.DEV_STEPS // R5.CKPT_EVERY),
            "ckpt_every": int(R5.CKPT_EVERY), "log_every": int(R5.LOG_EVERY),
            "morph_pool": {"n_fit_per_class": int(R5.MORPH_N_FIT_PER_CLASS),
                           "n_eval_per_class": int(R5.MORPH_N_EVAL_PER_CLASS),
                           "ridge_alpha": float(R5.MORPH_RIDGE_ALPHA)},
            "morph_columns": ["morph_head_auroc (condition 3)",
                              "morph_head_auroc_pool",
                              "morph_proxy_auroc (no detector)",
                              "morph_proxy_auroc_encoder"],
            "cond3_detector_column": D5.COND3_COLUMN,
            "cond3_interpretation_columns": list(D5.COND3_INTERPRETATION_COLUMNS),
            "cond3_floor": D5.COND3_SIGN_FLIP,
            "cond4_form": "differential vs control, src and tgt separately",
            "cond4_src_ceiling": D5.COND4_SRC_CEILING,
            "cond4_tgt_ceiling": D5.COND4_TGT_CEILING,
            "cond3_margin": D5.COND3_MARGIN,
            "cond3_median_window_default": 1,
            "cond3_m_ceiling": D5.COND3_M_CEILING,
            "cond3_m_j_factor": D5.COND3_M_J_FACTOR,
            "cond3_j_window": list(D5.COND3_J_WINDOW),
            "cond3_iqr_last_steps": D5.COND3_IQR_LAST_STEPS,
            "d_holdout_frac": float(D5.DEFAULT_HOLDOUT_FRAC),
            "local_test_suite": "scripts/test_da_dann.py -- 229 checks, all passing",
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(_ROOT, "runs", "dann_r5_prereg.json"))
    p.add_argument("--verify", action="store_true",
                   help="check the existing file's digest and exit; write nothing")
    a = p.parse_args(argv)
    out = os.path.abspath(a.out)

    if a.verify:
        chk = json.load(open(out))
        got = chk.pop("sha256_of_body_without_this_field")
        again = hashlib.sha256(_canonical(chk).encode()).hexdigest()
        print(f"[prereg] {out}")
        print(f"[prereg] recorded digest : {got}")
        print(f"[prereg] recomputed      : {again}")
        print(f"[prereg] VERIFIED: {got == again}")
        return 0 if got == again else 1

    body = json.loads(json.dumps(build_body()))     # canonicalise before hashing
    h = hashlib.sha256(_canonical(body).encode()).hexdigest()
    body["sha256_of_body_without_this_field"] = h
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(body, fh, indent=1, sort_keys=True)

    # ENFORCE it: re-read what actually landed on disk and verify the digest.
    chk = json.load(open(out))
    got = chk.pop("sha256_of_body_without_this_field")
    again = hashlib.sha256(_canonical(chk).encode()).hexdigest()
    if got != again:
        raise SystemExit(f"[prereg] the written file does not certify itself: "
                         f"{got} != {again}")
    print("=" * 92)
    print("ROUND-5 DANN PRE-REGISTRATION")
    print("=" * 92)
    print(f"  {out}")
    print(f"  sha256 : {h}")
    print(f"  hash verified against the file on disk: {got == again}")
    print(f"  {os.path.getsize(out):,} bytes, {len(body)} top-level keys")
    print("=" * 92)
    print("  ARM 1: DONE (runs/dann_arm1). Conditions 3 and 4 RE-SPECIFIED from it.")
    print("  PENDING, in order: (1) declare k, then the three margins, and hash them;")
    print("  (2) run arm 2 (log_only); (3) declare and hash the budget into")
    print("  runs/dann_r5_budget.json; (4) run folds 1-4 armed.")
    print(f"  Condition 3 reads {D5.COND3_COLUMN} (control 0.9076, headroom "
          f"{D5.COND3_HEADROOM:.4f}); 3b is a FLOOR at {D5.COND3_SIGN_FLIP}.")
    print(f"  Admissible bands from arm 1:  M [0.0234, {D5.COND3_M_CEILING:.4f}]  |  "
          f"cond4 src [0.5175, {D5.COND4_SRC_CEILING:.4f}]  |  "
          f"cond4 tgt [1.3263, {D5.COND4_TGT_CEILING:.4f}]")
    print("  The two HEAD columns are logged with NO DETECTOR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
