"""
lambda0_r6_step0_conditions.py
==============================
ROUND 6 -- evaluate EVERY declared abort/advisory condition at STEP 0, and on the only
lambda=0 trajectory that exists, BEFORE the condition is written into a
pre-registration. CPU-only, reads no test-set metric.

WHY THIS EXISTS -- the standing rule, and it has now been earned twice.

  * Round 5: ``abort3b`` fired on all four confirmatory folds and, under the
    pre-registered arrangement, would have killed every one of them at ~step 450.
    v10's decision to make conditions 3 and 4 advisory is the only reason that data
    survived (7.19).
  * Round 6 Stage 0a: the collapse band proposed for BNM's selection criterion,
    0.25-0.75, would have fired at step 0 -- Baseline A's head predicts positive on
    0.2292 of the target before training does anything (7.20).

Two thresholds fired on their own starting state. So: **evaluate every declared
threshold at step 0 before declaring it, and print the step-0 value into the
pre-registration.** This script is that check, run mechanically rather than by
inspection.

WHAT IT READS. ``runs/dann_r5_arm1_control/steps.csv`` -- arm 1, fold 0, lambda=0,
8000 steps. It is the SAME ARM as the confirmatory folds, differing only in fold and
budget, so it is the best available prior on what these detectors will do. It is not a
prediction of folds 1-4 and is not used as one: the output is "this condition fired /
did not fire on the one trajectory we have", which is exactly what a pre-registration
should record and nothing more.

    python -u scripts/lambda0_r6_step0_conditions.py --out runs/lambda0_r6_step0_conditions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM1_STEPS = os.path.join(REPO, "runs", "dann_r5_arm1_control", "steps.csv")
MORPH_STEP0 = os.path.join(REPO, "runs", "dann_r5_morph_step0.json")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def longest_streak(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def evaluate(rows, step0_morph: dict) -> dict:
    col = {c: [_f(r.get(c)) for r in rows] for c in rows[0]}
    steps = [int(float(r["step"])) for r in rows]
    first = rows[0]
    out = {"trajectory": {"file": os.path.relpath(ARM1_STEPS, REPO).replace("\\", "/"),
                          "n_rows": len(rows),
                          "first_logged_step": steps[0], "last_step": steps[-1]},
           "conditions": {}}

    def cond(key, declaration, first_logged_value, fires_immediately, flags, sustain,
             fired_col=None, note=None, true_step0=None):
        streak = longest_streak(flags)
        fired_logged = None
        if fired_col and fired_col in col:
            fired_logged = any(bool(v) for v in col[fired_col] if v is not None)
        rec = {
            "declaration": declaration,
            # steps.csv's first row is step 1: the runner logs no step-0 training row.
            # Where a genuine step-0 reading exists (runs/dann_r5_morph_step0.json) it
            # is carried alongside rather than conflated -- the two differ, and for
            # condition 3b they differ across the threshold.
            "first_logged_step_value": first_logged_value,
            "true_step0_value": true_step0,
            "FIRES_AT_FIRST_LOGGED_STEP": bool(fires_immediately),
            "sustain_required_steps": sustain,
            "longest_streak_on_arm1": streak,
            "would_fire_on_arm1": bool(sustain is not None and streak >= sustain)
            if sustain else bool(streak > 0),
            "runner_logged_fired_flag": fired_logged,
        }
        if note:
            rec["note"] = note
        out["conditions"][key] = rec

    # ---- condition 1: D overpowering -------------------------------------
    dah = col.get("d_acc_holdout", [])
    cond("1_d_overpowering",
         "held-out-target domain accuracy > 0.95 for 1000 consecutive steps WITH the "
         "domain loss not falling",
         _f(first.get("d_acc_holdout")),
         (_f(first.get("d_acc_holdout")) or 0) > 0.95,
         [(v is not None and v > 0.95) for v in dah], 1000,
         "abort1_d_overpowering_fired",
         "on the lambda=0 arm D trains against a representation no adversarial gradient "
         "is shaping, so this condition is about D's own capacity, not about adaptation")

    # ---- condition 2: degenerate collapse --------------------------------
    dps = col.get("d_pred_source_frac", [])
    v0 = _f(first.get("d_pred_source_frac"))
    cond("2_degenerate_collapse",
         "d_pred_source_frac outside [0.2, 0.8] for 500 consecutive steps",
         v0, v0 is not None and not (0.2 <= v0 <= 0.8),
         [(v is not None and not (0.2 <= v <= 0.8)) for v in dps], 500,
         "abort2_degenerate_collapse_fired")

    # ---- condition 3a: control-referenced margin --------------------------
    out["conditions"]["3a_morph_vs_control"] = {
        "declaration": "morph AUROC below the lambda=0 control's value at the SAME "
                       "ABSOLUTE step by more than M, sustained 500 steps",
        "margin_M": None,
        "status": "INACTIVE and INAPPLICABLE. M is UNSET (7.19 declared 3a UNCALIBRABLE: "
                  "3J = 0.099 against a ceiling of 0.04). It is also structurally "
                  "meaningless HERE -- this arm IS the control, so the condition would "
                  "compare the control against itself.",
        "FIRES_AT_FIRST_LOGGED_STEP": False,
        "first_logged_step_value": None,
        "true_step0_value": None,
        "would_fire_on_arm1": False,
    }

    # ---- condition 3b: sign flip -- BOTH instruments ----------------------
    # 7.19's whole lesson is that these two columns are ~0.34 apart and a threshold
    # declared on one is meaningless on the other. So evaluate BOTH, explicitly.
    for label, column, thresh, scale in (
            ("3b_sign_flip_HEAD_column", "morph_head_auroc_pool", 0.5, "~0.57"),
            ("3b_sign_flip_PROBE_column", "morph_proxy_auroc", 0.7, "~0.92")):
        series = col.get(column, [])
        s0 = series[0] if series else None
        cond(label,
             f"{column} < {thresh} (no sustain)", s0,
             s0 is not None and s0 < thresh,
             [(v is not None and v < thresh) for v in series], None,
             "abort3b_morph_sign_flip_fired",
             f"the {scale}-scale instrument. NEVER quote one of these as the other "
             f"(7.19): they read ~0.34 apart at step 0.",
             true_step0=step0_morph.get(column))

    # ---- condition 4: rank collapse --------------------------------------
    for dom in ("src", "tgt"):
        series = col.get(f"{dom}_participation_ratio", [])
        s0 = series[0] if series else None
        cond(f"4_rank_collapse_{dom}",
             f"{dom}_participation_ratio < 20 of 640, sustained 500 steps", s0,
             s0 is not None and s0 < 20,
             [(v is not None and v < 20) for v in series], 500,
             "abort4_rank_collapse_fired")

    out["step0_morphology_instruments"] = step0_morph
    out["any_condition_fires_at_first_logged_step"] = any(
        c.get("FIRES_AT_FIRST_LOGGED_STEP") for c in out["conditions"].values())
    out["conditions_that_would_fire_on_arm1"] = sorted(
        k for k, c in out["conditions"].items() if c.get("would_fire_on_arm1"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO, "runs", "lambda0_r6_step0_conditions.json"))
    args = ap.parse_args()

    with open(ARM1_STEPS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    step0_morph = {}
    if os.path.exists(MORPH_STEP0):
        with open(MORPH_STEP0, encoding="utf-8") as fh:
            m = json.load(fh)
        step0_morph = m.get("readings_step0", {}).get("dann_nb3", {})

    res = evaluate(rows, step0_morph)

    print("=" * 80)
    print("ROUND 6 -- every declared condition, evaluated AT STEP 0 and on arm 1")
    print("=" * 80)
    print(f"\ntrajectory: {res['trajectory']['file']}  "
          f"({res['trajectory']['n_rows']} rows, steps "
          f"{res['trajectory']['first_logged_step']}-{res['trajectory']['last_step']})")
    print("\nstep-0 morphology instruments (never quote one as the other):")
    for k, v in sorted(step0_morph.items()):
        print(f"    {k:<32} {v}")

    print(f"\n{'condition':<30}{'true s0':>10}{'step 1':>10}{'fires@1':>9}"
          f"{'sustain':>9}{'streak':>9}{'fired':>8}")
    for k, c in res["conditions"].items():
        s1 = c.get("first_logged_step_value")
        t0 = c.get("true_step0_value")
        print(f"{k:<30}{('-' if t0 is None else f'{t0:.4f}'):>10}"
              f"{('n/a' if s1 is None else f'{s1:.4f}'):>10}"
              f"{str(c.get('FIRES_AT_FIRST_LOGGED_STEP')):>9}"
              f"{str(c.get('sustain_required_steps', '-')):>9}"
              f"{str(c.get('longest_streak_on_arm1', '-')):>9}"
              f"{str(c.get('would_fire_on_arm1')):>8}")

    print(f"\nANY condition fires at the first logged step: "
          f"{res['any_condition_fires_at_first_logged_step']}")
    print(f"conditions that would fire on arm 1: "
          f"{res['conditions_that_would_fire_on_arm1'] or 'none'}")
    print("\nAll conditions on this arm are ADVISORY (abort_mode=log_only on every "
          "fold, from run_dann_r5.default_abort_mode). None can stop a run; the table "
          "above is recorded so that a firing detector is a disclosed expectation "
          "rather than a surprise.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
