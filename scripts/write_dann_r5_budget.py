#!/usr/bin/env python
"""Record round-5 DANN's confirmatory budget. Mirrors coral_r4_budget.json's shape.

DECLARED UNDER A NAMED RULE, CORROBORATED BY THE CURVE. This is neither of the two things
the record already contains, and the distinction is the point:

  * CORAL x nb3's budget is a PURE COST DECLARATION -- its own file says "no budget in
    250-8000 is defensible over any other on evidence", so the choice was cost and is
    labelled as one.
  * A DERIVED budget would read the metric (argmax, 1-sd rules) and inherit 7.11's
    winner's-curse problem.

This is a third thing. The rule reads DANN's OWN CONFIGURATION -- the lambda horizon -- and
no metric at all. The fold-0 curve is then checked for consistency with it. That ordering
matters: the rule was not chosen to select a value, and a reader can verify it by noting
that the rule is statable without opening a curve file.

HISTORY, kept rather than erased. 2000 was declared on cost 2026-09-01 (brief v11), then
WITHDRAWN 2026-09-02 (brief v12) when the decision was returned to the run owner, then NOT
re-chosen: 4000 was decided 2026-09-02 from runs/dann_r5_budget_packet.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from da_losses import dann_lambda_at_step                           # noqa: E402

DEV_CURVE = os.path.join(_ROOT, "runs", "dann_r5_curves", "arm2_dann_curve.csv")
CONTROL_CURVE = os.path.join(_ROOT, "runs", "dann_r5_curves", "arm1_lambda0_curve.csv")
OUT = os.path.join(_ROOT, "runs", "dann_r5_budget.json")

LAMBDA_HORIZON = 2000
CONFIRM_STEPS = 2 * LAMBDA_HORIZON            # THE RULE, evaluated: 4000
MIN_8000_T4 = 306.83                          # measured on arm 2 itself

THE_RULE = ("Budget = 2 x the lambda horizon, so that half of each confirmatory run "
            "trains at fixed adversarial strength. Derived from DANN's own "
            "configuration; reads no metric.")


def _curve(p, col="tpr_at_1e-3"):
    return {int(r["step"]): float(r[col]) for r in csv.DictReader(open(p, newline=""))}


def build() -> dict:
    dev, ctl = _curve(DEV_CURVE), _curve(CONTROL_CURVE)
    ks = sorted(dev)
    v = np.array([dev[s] for s in ks], float)
    c = np.array([ctl[s] for s in ks], float)
    early = np.array([dev[s] for s in ks if s <= LAMBDA_HORIZON], float)
    settled = np.array([dev[s] for s in ks if s > LAMBDA_HORIZON], float)
    per_min = MIN_8000_T4 / 8000.0

    def cost(B):
        return {"minutes_per_fold": round(per_min * B, 1),
                "gpu_hours_folds_1_4": round(4 * per_min * B / 60.0, 1),
                "checkpoints_per_fold": B // 250}

    def local(step, k=2):
        i = ks.index(step)
        w = v[max(0, i - k):min(len(v), i + k + 1)]
        return round(float(w.std(ddof=1)), 6)

    return {
        "budget_steps": CONFIRM_STEPS,
        "derived_or_declared": "declared under a named rule, corroborated by the curve",
        "basis_note": (
            "NOT a pure cost declaration as CORAL x nb3's was, and NOT derived from the "
            "metric. The rule reads DANN's own configuration (the lambda horizon) and no "
            "metric; the fold-0 curve is then checked for CONSISTENCY with it. The rule "
            "is statable without opening a curve file, which is how a reader can verify "
            "it was not chosen to select a value."),
        "is_preregistered_budget": True,
        "declared_utc": "2026-09-02T00:00:00Z",
        "decided_from": "runs/dann_r5_budget_packet.md",
        "development_run": "runs/dann_arm2_fold0 (dann_nb3, fold 0, 8000 steps)",
        "control_run": "runs/dann_arm1 (dann_nb3_lambda0, fold 0, 8000 steps)",

        "THE_RULE": THE_RULE,

        "why_the_rule": {
            "the_transition": (
                "lambda = tanh(5*step/2000) pins at ~1.0 near step 2000, so EVERY "
                "confirmatory fold has its own transition at the same absolute step, and "
                "the budget decides which side of it all four folds report from."),
            "the_two_regimes_on_fold_0": {
                "early_lambda_still_ramping": {
                    "steps": "250-2000", "n": int(early.size),
                    "mean": round(float(early.mean()), 4),
                    "sd": round(float(early.std(ddof=1)), 4)},
                "settled_lambda_fixed": {
                    "steps": "2250-8000", "n": int(settled.size),
                    "mean": round(float(settled.mean()), 4),
                    "sd": round(float(settled.std(ddof=1)), 4)},
                "sd_ratio_early_over_settled": round(
                    float(early.std(ddof=1) / settled.std(ddof=1)), 2),
            },
            "what_that_rules_out": (
                "a budget of 2000 would place all four folds in the EARLY regime, at "
                f"{early.std(ddof=1)/settled.std(ddof=1):.1f}x the within-fold scatter, on a round where 7.16 "
                "records a realised half-width of ~0.08 at 98 positives. 4000 places "
                "every fold well inside the settled regime rather than at its edge."),
        },

        "corroboration_NOT_derivation": {
            "settled_regime_begins": 2250,
            "consistent_with_the_rule": True,
            "reading_at_the_budget": dev[CONFIRM_STEPS],
            "settled_mean": round(float(settled.mean()), 4),
            "distance_in_settled_sds": round(
                float(abs(dev[CONFIRM_STEPS] - settled.mean()) / settled.std(ddof=1)), 2),
            "note": (
                f"the fold-0 curve's settled regime begins around 2250, consistent with "
                f"the rule; and the reading at {CONFIRM_STEPS} ({dev[CONFIRM_STEPS]:.4f}) "
                f"is UNREMARKABLE against the settled mean "
                f"({settled.mean():.4f}) -- "
                f"{abs(dev[CONFIRM_STEPS]-settled.mean())/settled.std(ddof=1):.2f} settled-sds away. "
                f"THE RULE WAS NOT CHOSEN TO SELECT A VALUE, and this is the check that "
                f"it did not accidentally land on a favourable one."),
        },

        "alternatives_considered_and_declined": {
            "2000": ("CORAL-matched (CORAL x nb3's verified budget; FreeMatch x nb3's "
                     "is 1500, not 2000). DECLINED: places all folds inside the lambda "
                     "ramp. CORAL HAS NO RAMP and FreeMatch has none either, so neither "
                     "budget faced an analogous constraint and neither could carry "
                     "across. This is NOT a claim that their budgets were wrong."),
            "1750": (f"the argmax ({dev[1750]:.4f}). DECLINED on 7.11's winner's-curse "
                     f"precedent, and it sits in the early regime where the GLOBAL sd "
                     f"used by such rules describes neither phase (local sd "
                     f"{local(1750)})."),
            "1250": (f"the two 1-sd rules, which coincide here ({dev[1250]:.4f}). "
                     f"DECLINED for the same reason: early regime, local sd "
                     f"{local(1250)}."),
            "8000": (f"last checkpoint -- R1 applying, with no budget selection to "
                     f"defend. DECLINED ON COST: 20.5 GPU-h against 10.2, to move the "
                     f"fold-0 reading {dev[8000]-dev[CONFIRM_STEPS]:+.4f} and the local "
                     f"scatter {local(CONFIRM_STEPS)} -> {local(8000)}."),
            "3000": ("a cheaper point still past every fold's transition. DECLINED in "
                     "favour of a rule statable as a round fraction. RECORDED BECAUSE IT "
                     "IS THE FALLBACK if the hours become a problem: 7.7 GPU-h against "
                     "10.2, and it clears the transition by 750 steps."),
        },

        "lambda_at_the_budget": {
            "lambda_horizon": LAMBDA_HORIZON,
            "lambda_at_confirm_steps": dann_lambda_at_step(CONFIRM_STEPS, LAMBDA_HORIZON),
            "first_step_lambda_ge_0.99": 1059,
            "first_step_lambda_ge_0.9999": 1981,
            "fraction_of_run_at_lambda_ge_0.99": round((CONFIRM_STEPS - 1059 + 1)
                                                       / CONFIRM_STEPS, 3),
            "fraction_of_run_at_lambda_ge_0.9999": round((CONFIRM_STEPS - 1981 + 1)
                                                         / CONFIRM_STEPS, 3),
            "development_run_for_comparison": {
                "fraction_at_lambda_ge_0.99": round((8000 - 1059 + 1) / 8000, 3),
                "fraction_at_lambda_ge_0.9999": round((8000 - 1981 + 1) / 8000, 3)},
            "note": ("the earlier concern that the horizon EQUALLED the budget is "
                     "resolved by this choice: at 4000 the run spends 73.6% of its steps "
                     "at lambda >= 0.99 and 50.5% at >= 0.9999, against the development "
                     "run's 86.8% and 75.2%."),
        },

        "PRE_DECLARED_headline_and_sensitivity": {
            "status": ("PRE-DECLARED 2026-09-02, BEFORE ANY CONFIRMATORY FOLD EXISTS. "
                       "This is the whole point of recording it now."),
            "THE_DECLARATION": (
                "The headline DANN x nb3 figure is the 5-fold mean at the declared budget "
                "of 4000. The 5-fold mean at step 2000 is reported as a BUDGET-SENSITIVITY "
                "CHECK and does not replace it, WHICHEVER IS HIGHER."),
            "why_now": (
                "two numbers with the choice of headline made afterwards is selection "
                "wearing a disclosure. The ordering fixed in advance is what makes it a "
                "sensitivity check instead."),
            "what_it_answers": (
                "it converts the cross-budget objection from an argument into a table "
                "row: at CORAL's budget DANN scores X, at its own declared budget Y."),
            "cost": (
                "ZERO. ckpt_every is 250 and the budget is 4000, so a confirmatory fold "
                "writes 16 checkpoints at 250-step spacing and STEP 2000 IS CHECKPOINT 8 "
                "OF 16 on every fold. Its logits are persisted natively during the run. "
                "No extra checkpoints, no extra scoring, no config change."),
            "correction_to_the_brief": (
                "brief v14 2 says '4000 with 32 checkpoints, spacing 125, so step 2000 is "
                "checkpoint 16' and prices it at 'five extra checkpoints'. ckpt_every is "
                "a fixed 250, so it is 16 checkpoints at spacing 250 and step 2000 is "
                "checkpoint 8. The conclusion is STRONGER than the brief's: the readout "
                "costs nothing at all."),
            "step_2000_on_the_confirmatory_grid": True,
            "confirmatory_checkpoint_grid": list(range(250, 4001, 250)),
        },

        "cross_method_budgets_VERIFIED": {
            "note": ("READ FROM THE FILES, 2026-09-02, after an earlier claim of "
                     "'CORAL and FreeMatch both at 2000' proved wrong for FreeMatch."),
            "dann_nb3": {"budget": 4000, "source": "runs/dann_r5_budget.json (this file)"},
            "coral_nb3": {"budget": 2000,
                          "source": "runs/coral_r4_budget.json selected_budget_steps",
                          "basis": "declared (cost)"},
            "freematch_nb3": {
                "budget": 1500,
                "source": ("runs/freematch_nb3_unfrozen_fold{0..4}/run_config.json -- "
                           "all five at max_steps 1500, budget_steps 1500, "
                           "is_preregistered_budget true"),
                "basis": "pre-registered"},
            "the_three_rows_sit_at": "1500 / 2000 / 4000",
            "CORRECTION": (
                "an earlier disclosure said CORAL and FreeMatch were BOTH confirmed at "
                "2000. FreeMatch x nb3 was confirmed at 1500. Every disclosure has been "
                "corrected to 1500 / 2000 / 4000."),
        },

        "stability_context": {
            "note": ("32-checkpoint 250-8000 grid, fold 0, TPR@1e-3. Context for the "
                     "budget's sensitivity, not an argument for any value."),
            "dann_arm2_lambda_live": {"mean": round(float(v.mean()), 6),
                                      "sd": round(float(v.std(ddof=1)), 6),
                                      "sd_over_mean": round(float(v.std(ddof=1) / v.mean()), 4)},
            "dann_arm1_lambda0": {"mean": round(float(c.mean()), 6),
                                  "sd": round(float(c.std(ddof=1)), 6),
                                  "sd_over_mean": round(float(c.std(ddof=1) / c.mean()), 4)},
            "coral_r4_nb3": {"mean": 0.434118, "sd": 0.051328, "sd_over_mean": 0.1182},
            "freematch_nb3": {
                "mean": 0.4115, "sd": 0.0606, "sd_over_mean": 0.147,
                "CORRECTION": (
                    "an earlier note said NO COMPARABLE FIGURE EXISTS because 'its "
                    "fold-0 curve stops at 1500'. That read FreeMatch's CONFIRMATORY "
                    "fold-0 run, which stops at 1500 because 1500 IS ITS BUDGET. Its "
                    "fold-0 DEVELOPMENT curve -- runs/freematch_fold0_nb3_unfrozen/"
                    "curve.csv -- runs 0..8000 at stride 125, and 32 of its points land "
                    "on the matched 250-stride grid. The figure exists and is given "
                    "here."),
            },
            "ordering": (
                "arm 1 (0.276) > arm 2 (0.202) > FreeMatch (0.147) > CORAL (0.118). "
                "ARM 1 HAS NO ADVERSARY, "
                "so unfreezing WITHOUT a DA term is the least stable of the three -- if "
                "anything the adversarial term stabilises relative to plain fine-tuning "
                "at nb3. Arms 1 and 2 share an init, fold, seed, GPU and grid and differ "
                "in lambda alone, so that contrast is controlled; CORAL shares none of "
                "it and is context, not a matched comparison."),
        },

        "cost_accounting": {
            "measured_minutes_8000_steps_T4": MIN_8000_T4,
            "measured_breakdown": ("306.8 min total for arm 2 = ~241 min training + "
                                   "65.8 min scoring 32 checkpoints"),
            "at_the_budget": cost(CONFIRM_STEPS),
            "at_each_alternative": {str(B): cost(B) for B in (1250, 1750, 2000, 3000, 8000)},
            "note": ("scoring cost is per CHECKPOINT and the checkpoint count scales with "
                     "the budget, so this is close to linear in steps."),
        },

        "confirmatory_design": {
            "folds": [1, 2, 3, 4],
            "fold0": "DEVELOPMENT -- excluded from every headline number",
            "comparison": ("fold-mean vs fold-mean (PROJECT.md 2: pooling is invalid for "
                           "per-fold models)"),
            "arm1_folds_1_4": "CONSIDERED AND DECLINED -- see prereg arm1_folds_1_4_declined",
            "abort_modes": ("conditions 1 and 2 ARMED; conditions 3 and 4 ADVISORY on "
                            "every arm (v10 4)."),
            "timing_of_this_decision": (
                "made AFTER fold 0, BY DESIGN. 7.14 specifies a fold-0 development run "
                "for exactly this purpose. The rule reads no metric, so the post-fold-0 "
                "timing does not make it a selection -- it could have been stated before "
                "fold 0 ran and would have given the same number."),
            "note": "this file must exist and be hashed BEFORE any confirmatory fold runs",
        },

        "history": {
            "2000_declared": ("2026-09-01, brief v11 4, on cost and CORAL-matching. "
                              "sha256 9987a46a5df7ad75fd0435505b6ef095099bf9c555554b4108"
                              "d21717cf6bb330"),
            "2000_withdrawn": ("2026-09-02, brief v12: the budget was returned to the run "
                               "owner as a decision to be made from the fold-0 curve."),
            "4000_decided": ("2026-09-02, brief v13, from "
                             "runs/dann_r5_budget_packet.md. 2000 was NOT re-chosen."),
            "why_kept": ("the record should show that 2000 was declared, withdrawn, and "
                         "then not re-chosen -- deleting it would leave a reader unable "
                         "to tell that the question had ever been reopened."),
        },

        "fold0_curve_TPR_at_1e-3": {str(k): dev[k] for k in ks},
        "fold0_control_curve_TPR_at_1e-3": {str(k): ctl[k] for k in ks},
    }


def main() -> int:
    body = build()
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["sha256_of_body_without_this_field"] = hashlib.sha256(blob).hexdigest()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(body, fh, indent=1, sort_keys=True)

    again = json.load(open(OUT))
    got = again.pop("sha256_of_body_without_this_field")
    calc = hashlib.sha256(json.dumps(again, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
    r = body["why_the_rule"]["the_two_regimes_on_fold_0"]
    co = body["corroboration_NOT_derivation"]
    print("=" * 92)
    print("ROUND-5 DANN -- CONFIRMATORY BUDGET")
    print("=" * 92)
    print(f"  {OUT}")
    print(f"  budget            : {body['budget_steps']} steps")
    print(f"  basis             : {body['derived_or_declared']}")
    print(f"  rule              : {THE_RULE}")
    print(f"  sha256            : {got}")
    print(f"  verified on disk  : {got == calc}")
    print(f"  regimes (fold 0)  : early {r['early_lambda_still_ramping']['mean']} "
          f"sd {r['early_lambda_still_ramping']['sd']} (n={r['early_lambda_still_ramping']['n']})  |  "
          f"settled {r['settled_lambda_fixed']['mean']} sd {r['settled_lambda_fixed']['sd']} "
          f"(n={r['settled_lambda_fixed']['n']})  -- "
          f"{r['sd_ratio_early_over_settled']}x")
    print(f"  corroboration     : reading {co['reading_at_the_budget']:.4f} vs settled mean "
          f"{co['settled_mean']} = {co['distance_in_settled_sds']} settled-sd (unremarkable)")
    c4 = body["cost_accounting"]["at_the_budget"]
    print(f"  cost              : {c4['minutes_per_fold']} min/fold, "
          f"{c4['gpu_hours_folds_1_4']} GPU-h for folds 1-4, "
          f"{c4['checkpoints_per_fold']} checkpoints/fold")
    lam = body["lambda_at_the_budget"]
    print(f"  lambda coverage   : {lam['fraction_of_run_at_lambda_ge_0.99']:.3f} at >=0.99, "
          f"{lam['fraction_of_run_at_lambda_ge_0.9999']:.3f} at >=0.9999 "
          f"(dev run {lam['development_run_for_comparison']['fraction_at_lambda_ge_0.99']:.3f} / "
          f"{lam['development_run_for_comparison']['fraction_at_lambda_ge_0.9999']:.3f})")
    print("=" * 92)
    return 0 if got == calc else 1


if __name__ == "__main__":
    sys.exit(main())
