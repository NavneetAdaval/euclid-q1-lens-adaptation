"""
write_bnm_r6_budget.py
======================
BNM's budget record, derived by the **pre-registered rule** from the fold-0 development
trajectory: ``budget = 2 x the first step at which normalised ||A||_* reaches the
sharpening ceiling``, read on the merged full + sub-grid nuclear-norm trajectory.

WHY THIS EXISTS AS ITS OWN SCRIPT. ``analyze_coral_dev8000_budget.py`` is the Task-A decay
detector and is applied UNMODIFIED, which is what keeps Delta comparable across arms. But
its section 4 writes a CORAL budget declaration -- ``method: deep_coral, gamma=16,
selected_budget_steps: 2000`` -- that is hardcoded and has nothing to do with BNM. Filing
that under a BNM name would be a trap for any later reader. The detector's useful output
is its analysis; BNM's budget comes from the rule below and is written here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

import bnm_round6 as B                     # noqa: E402
from integrity import verify as _verify_integrity   # noqa: E402

RUN = os.path.join(REPO, "runs", "bnm_r6_dev8000", "bnm_r6_dev8000_fold0")
OUT = os.path.join(REPO, "runs", "bnm_r6_budget.json")
PREREG = os.path.join(REPO, "runs", "bnm_r6_prereg.json")


def main():
    _verify_integrity(where="before writing the budget")
    anch = B.stage0a_anchors()
    full = list(csv.DictReader(open(os.path.join(RUN, "checkpoints.csv"))))
    stats = list(csv.DictReader(open(os.path.join(RUN, "stats_only.csv"))))
    cfg = json.load(open(os.path.join(RUN, "run_config.json")))

    merged = sorted(
        [(int(r["step"]), float(r["bnm_nuclear_norm"])) for r in stats] +
        [(int(r["step"]), float(r["bnm_nuclear_norm"])) for r in full])
    res = B.nuclear_norm_crossing([s for s, _ in merged], [v for _, v in merged],
                                  anch["sharpening_ceiling"])

    # How noisy is the trajectory near the crossing? The cadence resolves the SAMPLING to
    # +/-25, but the underlying quantity is a per-batch reading, so the crossing step is
    # only as well determined as the noise allows. Measured, not asserted.
    near = [(s, v) for s, v in merged if s <= 600]
    below = [v for _, v in near if v < anch["sharpening_ceiling"]]
    closest = max(below) if below else None

    body = {
        "round": 6, "arm": "bnm_nb3", "method": "BNM, n_blocks=3, image-space",
        "gamma": cfg.get("gamma"),
        "purpose": "the confirmatory budget for folds 1-4, derived by the rule "
                   "pre-registered in runs/bnm_r6_prereg.json",
        "prereg_sha256": hashlib.sha256(open(PREREG, "rb").read()).hexdigest(),
        "rule": res["rule"],
        "rule_reads_a_metric": res["reads_a_metric"],
        "development_run": {
            "dir": "runs/bnm_r6_dev8000/bnm_r6_dev8000_fold0",
            "fold": 0, "steps": cfg.get("steps_completed"),
            "wall_minutes": round(cfg.get("wall_minutes", 0), 1),
            "gpu": cfg.get("gpu_name"), "cadence": cfg.get("cadence"),
            "trajectory_points_read": len(merged),
            "step_range": [merged[0][0], merged[-1][0]]},
        "sharpening_ceiling": anch["sharpening_ceiling"],
        "crossing_step": res["crossing_step"],
        "selected_budget_steps": res["budget"],
        "outcome": res["outcome"],
        "fires": res["fires"],
        "cross_arm_comparable": res["cross_arm_comparable"],
        "matched_arm_budgets": res["matched_arm_budgets"],
        "which_number_is_quoted": res["which_number_is_quoted"],
        "budget_confound_removed_by_a_match": res["budget_confound_removed_by_a_match"],
        "read_off_rule": (
            "train to the budget and read the FINAL checkpoint. R1 -- the last checkpoint "
            "-- as pre-registered throughout the project. No label-free selector, no "
            "argmax, no smoothing: THE BUDGET IS THE SELECTOR."),
        "stage2_preview_was": 1000,
        "stage2_preview_matched_the_development_run": res["budget"] == 1000,
        "crossing_resolution": {
            "cadence_below_500": 25,
            "sampling_resolution_steps": "+/-25, so +/-50 on the budget",
            "BUT_THE_TRAJECTORY_IS_BATCH_NOISY": (
                "the nuclear norm is a per-batch reading and it is not monotone: over "
                "steps 25-500 it moves non-monotonically "
                f"(closest sub-ceiling approach {closest:.4f} against a ceiling of "
                f"{anch['sharpening_ceiling']:.4f}). So the finer cadence resolves WHERE WE "
                "LOOKED to +/-25, but the crossing step itself is only as well determined "
                "as that noise allows -- a single sharp batch could cross earlier. This is "
                "a limitation of the rule as written, recorded rather than corrected, "
                "because the rule is pre-registered and was applied as written."),
            "closest_approach_below_ceiling_before_crossing": closest},
        "NOT_FROM_THE_CORAL_SCRIPT": (
            "analyze_coral_dev8000_budget.py section 4 emits a CORAL budget declaration "
            "(deep_coral, gamma=16, 2000 steps) that is hardcoded and unrelated to this "
            "arm. It is the Task-A DECAY DETECTOR that is used from that script, applied "
            "unmodified; its budget section is not."),
    }
    body["sha256_of_body_without_this_field"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1, sort_keys=True)

    print(f"wrote {os.path.relpath(OUT, REPO)}")
    print(f"  crossing step          {body['crossing_step']}")
    print(f"  budget                 {body['selected_budget_steps']}")
    print(f"  outcome                {body['outcome']}")
    print(f"  cross-arm comparable   {body['cross_arm_comparable']}")
    print(f"  matched arm budgets    {body['matched_arm_budgets']}")
    print(f"  stage-2 preview held   {body['stage2_preview_matched_the_development_run']}")
    print(f"  sha256                 {body['sha256_of_body_without_this_field']}")
    _verify_integrity(where="after writing the budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
