"""
declare_window_average.py
=========================
ROUND 7, TASK 3 -- the CHECKPOINT-WINDOW AVERAGING declaration.

Written and hashed BEFORE any windowed number is computed. The point of the task is that
7.16 records nine orderings that dissolved under an analyst choice; an analysis invented
to fix that must not itself be an analyst choice made after seeing results.

    python -u scripts/declare_window_average.py --out runs/round7/window_prereg.json

Writes NOTHING outside --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

from integrity import verify as _verify_integrity  # noqa: E402

DECLARATION = {
    "round": 7,
    "task": "checkpoint-window averaging",
    "status": "DECLARED SECONDARY ANALYSIS -- replaces no hashed headline",
    "declared_before_any_windowed_number_was_computed": True,
    "reads_a_metric_to_place_the_window": False,

    "why": (
        "Single-point read-off is the noisiest estimator available. 7.16's eighth entry "
        "measures a within-run checkpoint sd of 0.0402-0.0914 on fold 0 against between-"
        "method differences of 0.018, and fold 0's BNM curve moves 0.5102 -> 0.2857 across "
        "one 250-step gap. Averaging k consecutive checkpoints' SCORES reduces that scatter "
        "by sqrt(k) IF the checkpoint noise is independent. Whether it actually does is the "
        "measurement."),

    "k": {
        "primary": 8,
        "grid": [1, 2, 4, 8],
        "declared_in_advance": True,
        "justification": (
            "k=8 from the checkpoint sd of 0.05-0.09 and a sqrt(k) noise reduction: 8 "
            "checkpoints take 0.05-0.09 to 0.018-0.032, which is the scale of the "
            "differences the nb3 table tries to resolve. k=1 IS the existing single-point "
            "read-off and is the comparison basis, not a new estimator. k=2 and k=4 are "
            "declared here so the sqrt(k) curve has interior points; they are reported "
            "whatever they show."),
        "k_may_not_be_varied_after_results_are_seen": True,
    },

    "score_combination": {
        "primary": {
            "name": "mean_logit",
            "rule": ("arithmetic mean, per image, of the RAW saved logits over the "
                     "window's checkpoints; the metric is then computed once on the "
                     "averaged score vector"),
        },
        "secondary": {
            "name": "mean_rank",
            "rule": ("per checkpoint, replace each image's logit by its rank over that "
                     "checkpoint's full eval pool divided by (N-1) so it lies in [0,1]; "
                     "average those across the window. Scale-free."),
            "why_declared": (
                "mean_logit is the literal reading of 'average scores', but it is not "
                "invariant to a checkpoint whose logit scale has drifted: one wide-scale "
                "checkpoint can dominate the mean and the window would then be a "
                "single-point read-off wearing a disguise. mean_rank cannot do that. Both "
                "are declared NOW and BOTH are reported for every arm; neither is "
                "promoted after the fact."),
        },
        "weights_are_never_averaged": (
            "scores only. Averaging weights across checkpoints of an unfrozen encoder is "
            "a different and unlicensed intervention."),
    },

    "window_placement": {
        "rule": ("the k consecutive AVAILABLE checkpoints ending AT the pre-registered "
                 "read-off step, inclusive of that step"),
        "placed_by": "the pre-registered read-off point -- so the rule reads no metric",
        "truncation": ("where fewer than k trained checkpoints exist at or before the "
                       "read-off, the window is truncated to all that do exist and the "
                       "ACHIEVED k is reported next to the value"),
        "step_0_excluded": ("step 0 is the initialisation, not a trained checkpoint; it "
                            "never enters a window"),
    },

    "scope": {
        "note": ("every arm and both baselines, at every mandatory readout. Read-off "
                 "steps are copied from each arm's own pre-registration/budget record; "
                 "none is chosen here."),
        "arms": {
            "CORAL x nb3": {"folds": [1, 2, 3, 4], "readouts": [2000],
                            "grid": "250..2000 step 250"},
            "FreeMatch x nb3": {"folds": [1, 2, 3, 4], "readouts": [1500],
                                "grid": "125..1500 step 125"},
            "DANN x nb3": {"folds": [1, 2, 3, 4], "readouts": [4000, 2000],
                           "grid": "250..4000 step 250",
                           "note": "2000 is 7.18's pre-declared sensitivity readout"},
            "lambda=0": {"folds": [1, 2, 3, 4], "readouts": [4000, 2000, 1500],
                         "grid": "250..4000 step 250"},
            "BNM x nb3": {"folds": [1, 2, 3, 4], "readouts": [1000, 1500, 2000, 4000],
                          "grid": "250..4000 step 250",
                          "note": "1000 is no_matched_arm_budget; 4000 is the mandatory one"},
            "Baseline A (frozen)": {"folds": [0, 1, 2, 3, 4], "readouts": [8000],
                                    "grid": "250..8000 step 250",
                                    "note": ("round-1 R1 last-checkpoint. Only step_8000 "
                                             "scores were retained, so the per-checkpoint "
                                             "score vectors are RECOMPUTED on CPU from the "
                                             "frozen 640-d feature cache and the retained "
                                             "per-step head weights. No GPU, no training.")},
        },
    },

    "metric": {
        "primary": "TPR@FPR=1e-3, interpolated, drop_intermediate=False, always a rate",
        "secondary_reported": ["auroc", "comp_at_purity (24% purity, corrected to 1:2200)"],
        "undefined_values": "n/a with no CI and no fold-mean, as everywhere else",
    },

    "scatter_test": {
        "question": "does the achieved scatter reduction match the sqrt(k) expectation?",
        "estimator": ("for each arm and fold, slide the k-window over EVERY valid "
                      "position on that arm's own checkpoint grid, compute the metric at "
                      "each position, and take the sd across positions. Compare against "
                      "the same fold's k=1 sd computed on the same set of positions, so "
                      "the two sds are over the same steps."),
        "expectation": "sd_k = sd_1 / sqrt(k)",
        "a_large_shortfall_means": ("the checkpoint noise is autocorrelated rather than "
                                    "independent, which is itself a reportable line"),
    },

    "what_this_is_not": {
        "not_a_replacement_for_any_hashed_headline": True,
        "hashed_headlines_stand": ("every pre-registered single-point number in 7.12, "
                                   "7.15, 7.18, 7.22 and 7.28 remains the arm's "
                                   "reported result"),
        "labelled_everywhere": "DECLARED SECONDARY ANALYSIS",
    },

    "the_verdict_it_owes": (
        "7.16 records nine orderings as dissolved. This analysis must state, for each "
        "ordering that a read-off choice could plausibly have caused, whether it SURVIVES "
        "window averaging or NOT. Orderings survived => the limit is in the data and 7.16 "
        "is stronger than nine anecdotes. Orderings did not survive => the read-off choice "
        "was the fault, the fix is located, and 7.16's affected entries get updated "
        "rather than left standing alongside a contradicting claim."),

    "no_significance_claims_beyond_the_declared_forms": (
        "the unit is the fold, n=4 (n=5 for Baseline A). Paired t on differences, df=n-1, "
        "exactly as 7.22 and 7.28. Never a bootstrap at n=4."),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    _verify_integrity(where="before writing the round-7 window declaration")

    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite a declaration")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    body = dict(DECLARATION)
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    body["sha256_of_body_without_this_field"] = digest
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1, sort_keys=True)

    back = json.load(open(out, encoding="utf-8"))
    chk = {k: v for k, v in back.items()
           if k != "sha256_of_body_without_this_field"}
    again = hashlib.sha256(
        json.dumps(chk, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if again != back["sha256_of_body_without_this_field"]:
        raise SystemExit("self-certification FAILED -- the file does not hash to itself")

    print(f"[declared] {os.path.relpath(out, REPO)}")
    print(f"[sha256]   {digest}")
    print(f"  k primary {DECLARATION['k']['primary']}, grid {DECLARATION['k']['grid']}")
    print("  score combinations: mean_logit (primary), mean_rank (secondary) -- BOTH reported")
    print("  window ends AT the pre-registered read-off step; the rule reads no metric")

    _verify_integrity(where="after writing the round-7 window declaration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
