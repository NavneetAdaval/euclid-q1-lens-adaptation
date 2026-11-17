"""
report_unfreeze_freematch.py
============================
Assembles the BRIEF §9 required outputs for the FreeMatch x nb3 fold-0 diagnostic run:

  1. results.json   -- TPR@1e-3 at step 8000 (unfrozen) and step 2000 (control), plus the
                       paired diff with a class-conditional bootstrap CI.
  2. full curves for both runs (summarised here; the raw curve.csv travels with the run).
  3. the FreeMatch-internal diagnostics (§5.3) as a tidy table.
  4. the memorization probe (§5.4).
  5. the collapse-gate agreement trace, and whether it fired.
  6. THE EXPLICIT VERDICT ON THE CONFOUND -- is the control's curve flat across
     250/500/1000/1500/2000?
  8. run_config.json provenance for both runs (n_checkpoints_saved, loaded head norm).

Item 7 (the written 5-fold budget recommendation) is a JUDGEMENT to be argued from the
curve by a human and pre-registered before any confirmatory run; this script prints the
evidence it must be argued from -- argmax step, whether the curve is still climbing at the
end, the post-argmax decrease fraction and the whole-curve noise sd -- and deliberately
does not write the recommendation itself.

REPORTING DISCIPLINE (PROJECT.md §9, BRIEF §8). Everything is a RATE. TPR@FPR=1e-3 and
purity@50%-completeness only; never "N of 98", never "lenses found", not in effect sizes,
CI half-widths or regret. And nothing here writes a verdict sentence: it prints the numbers
and the mechanical flatness test, because "fails the significance bar" and "is worse" are
different claims and conflating them has caused real errors in this project.

Run:
    python scripts/report_unfreeze_freematch.py
    python scripts/report_unfreeze_freematch.py --unfrozen runs/... --control runs/...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import (tpr_at_fpr, paired_bootstrap, default_tpr_metric,
                     DEFAULT_TARGET_FPR)                   # noqa: E402
from analyze_fold0_selection import _spearman              # noqa: E402  (one impl)

FOLD = 0
# the steps BRIEF §9.6 names as the control's flatness probe
CONTROL_PROBE_STEPS = (250, 500, 1000, 1500, 2000)
# §7.8 measured per-checkpoint noise sd ~= 0.046 on unfrozen fold-0 curves; the control is
# called FLAT when its spread across the probe steps stays inside the one-positive
# resolution band the metric can actually distinguish.
ONE_POSITIVE = 1.0 / 98.0        # 0.0102 -- the finest difference 98 positives can express


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return [{k: _maybe_float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _maybe_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return v


def read_json(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def load_scores(path):
    with np.load(path, allow_pickle=False) as z:
        return (z["pos_scores"], z["neg_scores"],
                z["pos_ids"] if "pos_ids" in z.files else None,
                z["neg_ids"] if "neg_ids" in z.files else None)


def find_scores(run_dir):
    sdir = os.path.join(run_dir, "scores")
    if not os.path.isdir(sdir):
        return None
    cands = [f for f in sorted(os.listdir(sdir)) if f.endswith(f"fold{FOLD}.npz")]
    return os.path.join(sdir, cands[0]) if cands else None


# ---------------------------------------------------------------------------
# 6. THE CONFOUND VERDICT
# ---------------------------------------------------------------------------

# MEASURED per-fold: the range of round 1's own converged 6000-8000 tail, i.e. what "flat"
# looks like on that fold when nothing at all has changed. Recomputed by
# runs/round1_tail_stats.json; hardcoded here so the report needs no round-1 curves.
ROUND1_TAIL_RANGE_BY_FOLD = {0: 0.0306, 1: 0.0714, 2: 0.0714, 3: 0.0408, 4: 0.0825}


def round1_tail_range(fold):
    """That fold's own flatness null. Falls back to the WIDEST observed if unknown, so an
    unrecognised fold cannot be judged by an accidentally strict threshold."""
    return ROUND1_TAIL_RANGE_BY_FOLD.get(int(fold), max(ROUND1_TAIL_RANGE_BY_FOLD.values()))


def control_flatness(curve, probe_steps=CONTROL_PROBE_STEPS, tpr_col=None, fold=FOLD):
    """Is the control's TPR curve flat across the probe steps?

    WHAT THIS DECIDES. The unfrozen run differs from round-1 FreeMatch in two ways at once:
    the encoder is unfrozen AND the head gets thousands more steps under a changed optimizer
    setup. The control removes the first, so its curve isolates the second. Flat across
    250/500/1000/1500/2000 => the extra-steps leg moves nothing => the unfrozen result may
    be attributed to unfreezing and compared against round 1 with a citable justification.
    Not flat => it may not, and that is a finding about the design, not a failure.

    ``flat`` is decided mechanically: the spread across the probe steps must not exceed the
    one-positive resolution of 98 positives (1/98 = 0.0102). Anything smaller than one
    positive is not a difference this metric can express, and anything larger is a real
    movement of the head that the unfrozen run also carries.
    """
    tpr_col = tpr_col or f"fold{int(fold)}_tpr_1e-3"
    by_step = {int(r["step"]): r for r in curve if r.get("step") is not None}
    got = {s: by_step[s].get(tpr_col) for s in probe_steps if s in by_step}
    got = {s: v for s, v in got.items() if v is not None}
    missing = [s for s in probe_steps if s not in got]
    step0 = by_step.get(0, {}).get(tpr_col)

    vals = list(got.values())
    spread = (max(vals) - min(vals)) if vals else None
    drift_from_step0 = (None if (step0 is None or not vals)
                        else max(abs(v - step0) for v in vals))
    # THE CRITERION IS PER-FOLD, keyed to THAT fold's own round-1 tail. A single constant
    # is the error: 3 positives was measured on fold 0, but round 1's converged 6000-8000
    # tail ranges over 3 / 7 / 7 / 4 / 8 positives on folds 0-4. Applying fold 0's number
    # to fold 3 called a continuation non-flat that was statistically indistinguishable
    # from round 1 continuing to run. Same principle as §7.6: reference against the MATCHED
    # null, not a constant.
    ROUND1_TAIL_RANGE = round1_tail_range(fold)
    v_shaped = (None if not vals else
                bool(min(vals) < min(vals[0], vals[-1]) - ONE_POSITIVE))
    flat = (None if (spread is None or missing)
            else bool(spread <= ROUND1_TAIL_RANGE + 1e-9 and not v_shaped))
    return {"probe_steps": list(probe_steps), "tpr_at_probe_steps": got,
            "missing_probe_steps": missing, "step0_tpr": step0,
            "spread": spread, "max_drift_from_step0": drift_from_step0,
            "one_positive_resolution": ONE_POSITIVE,
            "criterion_range": ROUND1_TAIL_RANGE, "fold": int(fold),
            "criterion_basis": (
                f"fold {fold}'s OWN round-1 converged tail (6000-8000) ranges "
                f"{ROUND1_TAIL_RANGE:.4f} = {ROUND1_TAIL_RANGE/ONE_POSITIVE:.0f} positives. "
                f"Per-fold because that range is 3/7/7/4/8 positives across folds 0-4 -- a "
                f"constant calibrated on fold 0 rejects continuations on noisier folds that "
                f"are indistinguishable from round 1 continuing to run."),
            "v_shaped": v_shaped, "flat": flat,
            "verdict": _flatness_sentence(flat, missing)}


def _flatness_sentence(flat, missing):
    if missing:
        return (f"UNDECIDED -- the control curve is missing probe steps {missing}, so the "
                f"extra-steps confound cannot be assessed.")
    if flat is True:
        return ("CONTROLLED -- the control's TPR@1e-3 is flat across "
                "250/500/1000/1500/2000 to within the one-positive resolution, so the extra "
                "steps under the changed optimizer setup move nothing on their own and the "
                "unfrozen result may be attributed to unfreezing.")
    return ("NOT CONTROLLED -- the control's TPR@1e-3 moves by more than the one-positive "
            "resolution across 250/500/1000/1500/2000, so the unfrozen result cannot be "
            "attributed to unfreezing alone: the extra steps move the head by themselves.")


# ---------------------------------------------------------------------------
# curve shape (the evidence for the §9.7 budget argument)
# ---------------------------------------------------------------------------

def curve_shape(curve, tpr_col=f"fold{FOLD}_tpr_1e-3", exclude_step0=True):
    rows = [r for r in curve if r.get(tpr_col) is not None
            and not (exclude_step0 and int(r["step"]) == 0)]
    if not rows:
        return {}
    steps = np.array([int(r["step"]) for r in rows], float)
    tpr = np.array([float(r[tpr_col]) for r in rows], float)
    argmax_i = int(np.argmax(tpr))
    half = len(tpr) // 2
    post = tpr[argmax_i:]
    dec = float(np.mean(np.diff(post) < 0)) if post.size > 1 else float("nan")
    rho, _n = _spearman(steps, tpr)
    tail = tpr[-max(len(tpr) // 4, 1):]
    return {
        "n_points": len(tpr),
        "argmax_step": int(steps[argmax_i]), "argmax_tpr": float(tpr[argmax_i]),
        "final_step": int(steps[-1]), "final_tpr": float(tpr[-1]),
        "spearman_step_tpr": (None if rho is None else float(rho)),
        "first_half_mean": float(np.mean(tpr[:half])) if half else None,
        "second_half_mean": float(np.mean(tpr[half:])),
        "whole_curve_mean": float(np.mean(tpr)), "whole_curve_sd": float(np.std(tpr)),
        "max_z_above_own_mean": float((np.max(tpr) - np.mean(tpr)) / (np.std(tpr) + 1e-12)),
        "post_argmax_fraction_decreasing": dec,
        "final_quarter_mean": float(np.mean(tail)),
        "final_quarter_sd": float(np.std(tail)),
        # "still climbing at the end" is the §9.7 trigger for arguing a LONGER confirmatory
        # budget. Stated as evidence, never as a decision.
        "still_climbing_at_end": bool(steps[argmax_i] >= steps[-1] * 0.9),
    }


# ---------------------------------------------------------------------------
# 5. collapse-gate agreement trace
# ---------------------------------------------------------------------------

def agreement_trace(curve, run_dir, col=f"agreement_fold{FOLD}"):
    """The §9.5 required output: the agreement trace AND whether the gate fired.

    The firing decision is read from ``gate_fold{k}.json``, which the run itself writes
    (run_unfreeze_freematch.run_collapse_gate) rather than being deferred to a command
    someone has to remember. If that file is absent the report says the decision is MISSING
    -- it never infers a verdict from the trace, because the gate's triggers are a cliff
    test with a persistence clause plus a floor, not a threshold on the last value.
    """
    rows = [(int(r["step"]), r.get(col)) for r in curve if r.get(col) is not None]
    gate = read_json(os.path.join(run_dir, f"gate_fold{FOLD}.json"))
    out = {"available": bool(rows)}
    if not rows:
        out["note"] = f"no {col} column -- the agreement reference was not wired in"
    else:
        vals = np.array([v for _s, v in rows], float)
        out.update({
            "n": len(rows), "first": float(vals[0]), "last": float(vals[-1]),
            "min": float(vals.min()), "min_at_step": int(rows[int(np.argmin(vals))][0]),
            "max": float(vals.max()),
            "monotone_decline": bool(np.all(np.diff(vals) <= 0)),
            # anchors from PROJECT.md §7.6: round-1 self-training floor ~0.87; the unfrozen
            # Baseline A rungs reached 0.74 (nb1) / 0.79 (nb3) WITHOUT the gate firing.
            "round1_selftraining_floor": 0.87,
            "lever1_unfrozen_lows": {"nb1": 0.74, "nb3": 0.79},
            "note": ("The gate is a FLOOR ALARM ONLY and is never maximised -- maximum "
                     "agreement is the source-only model itself (TTYD's MeanBN failure "
                     "mode).")})
    if gate:
        out["gate_fired"] = gate.get("fired")
        out["gate_triggers"] = gate.get("triggers")
        out["lockstep_check"] = gate.get("lockstep_check")
        out["gate_verdict"] = _gate_sentence(gate)
    else:
        out["gate_fired"] = None
        out["gate_verdict"] = (
            f"MISSING -- no gate_fold{FOLD}.json in this run dir. Required output #5 is "
            f"not satisfied; rerun the gate: python scripts/collapse_gate.py --run "
            f"{run_dir} --from-curve --tpr-col fold{FOLD}_tpr_1e-3")
    return out


def _gate_sentence(gate):
    if gate.get("error"):
        return f"ERROR -- the gate did not complete: {gate['error']}"
    if gate.get("fired"):
        return ("FIRED. Across three previous rounds the gate never fired, so its "
                "collapse-CATCHING ability was UNVERIFIED (§7.6) -- a fire here is the "
                "first positive case and is itself a result. Inspect the trigger and the "
                "step before treating the run's TPR as a measurement of unfreezing.")
    lo = gate.get("agreement_min")
    extra = ""
    if lo is not None:
        extra = (f" Minimum agreement {lo:.4f} against a round-1 self-training floor of "
                 f"~0.87 and Lever-1 unfrozen lows of 0.74 (nb1) / 0.79 (nb3), neither of "
                 f"which fired either.")
    return ("DID NOT FIRE." + extra + " The gate remains validated against false alarms "
            "only; with still no observed collapse, whether agreement decline separates "
            "healthy divergence from failure stays UNVERIFIED.")


# ---------------------------------------------------------------------------
# 3/4. diagnostics + memorization summaries
# ---------------------------------------------------------------------------

DIAG_SHOW = ["step", "tau_global_last", "local_neg_last", "local_pos_last",
             "acceptance_mean", "mask_rate_pos_mean", "mask_rate_neg_mean",
             "accepted_pos_fraction_mean", "pseudo_pos_fraction_all_mean",
             "sup_loss_mean", "consistency_mean", "unlabeled_loss_weighted",
             "unlabeled_over_supervised", "saf_mean", "step_skipped_mean"]


def print_table(rows, cols, title, max_rows=24):
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    if not rows:
        print("  (no rows)")
        return
    show = rows if len(rows) <= max_rows else (
        rows[:max_rows // 2] + [None] + rows[-max_rows // 2:])
    print("  " + " | ".join(f"{c[:14]:>14}" for c in cols))
    print("  " + "-" * (17 * len(cols)))
    for r in show:
        if r is None:
            print("  " + " | ".join(f"{'...':>14}" for _ in cols))
            continue
        print("  " + " | ".join(
            f"{(str(int(r[c])) if c == 'step' and r.get(c) is not None else _fmt(r.get(c))):>14}"
            for c in cols))


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if math.isfinite(v) else "n/a"
    return str(v)


def memorization_summary(rows):
    """Does the target pool's confidence pull AWAY from the composition-matched test pool?"""
    rows = [r for r in rows if r.get("target_minus_test_matched") is not None]
    if not rows:
        return {"available": False}
    steps = np.array([int(r["step"]) for r in rows], float)
    gap = np.array([float(r["target_minus_test_matched"]) for r in rows], float)
    tgt = np.array([float(r["conf_target_mean"]) for r in rows], float)
    tst = np.array([float(r["conf_test_matched"]) for r in rows], float)
    rho, _n = _spearman(steps, gap)
    return {"available": True, "n": len(rows),
            "gap_first": float(gap[0]), "gap_last": float(gap[-1]),
            "gap_max": float(gap.max()), "gap_change": float(gap[-1] - gap[0]),
            "spearman_step_gap": (None if rho is None else float(rho)),
            "target_conf_first": float(tgt[0]), "target_conf_last": float(tgt[-1]),
            "test_conf_first": float(tst[0]), "test_conf_last": float(tst[-1]),
            "reading": ("target and test confidence RISING TOGETHER (gap flat) is the "
                        "learning-something-real signature; the target pulling away is the "
                        "BRIEF §4 target-pool-memorization signature. The gap uses the "
                        "COMPOSITION-MATCHED test statistic, so it is not moved by the "
                        "target pool being ~1:1 while the test pool is ~1:269.")}


# ---------------------------------------------------------------------------
# 1. the paired difference
# ---------------------------------------------------------------------------

def paired_diff(unfrozen_scores, control_scores, target_fpr, B, seed):
    """Unfrozen R1 minus control R1, class-conditional paired bootstrap.

    Both runs score the SAME 98 fold-0 positives against the SAME ~26k shared negative pool
    (PROJECT.md §4.4), so the two score vectors are aligned row-for-row and the pairing is
    valid. That alignment is VERIFIED against the stored ids rather than assumed -- the
    bootstrap applies the same resampled row indices to both vectors, so a permutation would
    silently corrupt the CI rather than fail.
    """
    p_u, n_u, pid_u, nid_u = load_scores(unfrozen_scores)
    p_c, n_c, pid_c, nid_c = load_scores(control_scores)
    if p_u.size != p_c.size or n_u.size != n_c.size:
        return {"error": f"score vectors differ in length: unfrozen ({p_u.size},{n_u.size})"
                         f" vs control ({p_c.size},{n_c.size})"}
    if pid_u is None or pid_c is None or nid_u is None or nid_c is None:
        return {"error": "score dumps lack pos_ids/neg_ids, so row alignment cannot be "
                         "verified and a paired bootstrap would be unsound."}
    if not (np.array_equal(pid_u, pid_c) and np.array_equal(nid_u, nid_c)):
        return {"error": "score vectors are NOT row-aligned (ids differ) -- a paired "
                         "bootstrap would be invalid."}

    y = np.concatenate([np.ones(p_u.size, np.int64), np.zeros(n_u.size, np.int64)])
    s_u = np.concatenate([p_u, n_u])
    s_c = np.concatenate([p_c, n_c])
    bs = paired_bootstrap(y, s_u, s_c, metric_fn=default_tpr_metric(target_fpr),
                          B=B, seed=seed)
    lo, hi = bs.get("lo"), bs.get("hi")
    half = None if (lo is None or hi is None) else (float(hi) - float(lo)) / 2.0
    return {"tpr_unfrozen": float(tpr_at_fpr(y, s_u, target_fpr)),
            "tpr_control": float(tpr_at_fpr(y, s_c, target_fpr)),
            "diff": float(tpr_at_fpr(y, s_u, target_fpr)
                          - tpr_at_fpr(y, s_c, target_fpr)),
            "ci": [None if lo is None else float(lo), None if hi is None else float(hi)],
            "ci_half_width": half, "bootstrap": bs, "row_aligned_by_ids": True,
            "significance_note": (
                "This is fold 0. §7.8 established that fold 0 cannot resolve the unfreeze "
                "effect EVEN WITH ORACLE SELECTION (oracle gap 0.1122 against a half-width "
                "of 0.1226). Read the interval, not a pass/fail: 'fails the significance "
                "bar' and 'is worse' are different claims.")}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_report(unfrozen_dir, control_dir, target_fpr, B, seed):
    rep = {"brief": "BRIEF_freematch_nb3_fold0.md", "fold": FOLD,
           "metric": f"TPR@FPR={target_fpr:g}",
           "status": "EXPLORATORY fold-0 diagnostic -- not a result for the paper"}

    for tag, d in (("unfrozen", unfrozen_dir), ("control", control_dir)):
        cfg = read_json(os.path.join(d, "run_config.json"))
        res = read_json(os.path.join(d, "results.json"))
        curve = read_csv(os.path.join(d, "curve.csv"))
        rep[tag] = {
            "dir": d, "present": bool(cfg or res),
            "n_blocks": cfg.get("n_blocks"), "max_steps": cfg.get("max_steps"),
            "schedule_total_steps": cfg.get("schedule_total_steps"),
            "init_head_from": cfg.get("init_head_from"),
            "init_head_l2": cfg.get("init_head_l2"),
            "n_checkpoints_saved": cfg.get("n_checkpoints_saved"),
            "n_checkpoints_expected": cfg.get("n_checkpoints_expected"),
            "step0_gate": cfg.get("step0_gate") or res.get("step0_gate"),
            "eval_precision": cfg.get("eval_precision"),
            "r1_tpr": res.get("tpr"), "r1_auroc": res.get("auroc"),
            "r1_auprc": res.get("auprc"), "r1_purity_corr": res.get("purity_corr"),
            "r1_comp_at_purity": res.get("comp_at_purity"),
            "train_minutes": res.get("train_minutes"),
            "fp16_health": ((res.get("freematch_stability") or {}).get("instrument")
                            or cfg.get("instrument")),
            "curve_shape": curve_shape(curve),
            "agreement": agreement_trace(curve, d),
            "memorization": memorization_summary(
                read_csv(os.path.join(d, "memorization.csv"))),
            "n_curve_rows": len(curve),
        }

    # 6 -- THE CONFOUND VERDICT
    rep["confound_verdict"] = control_flatness(read_csv(
        os.path.join(control_dir, "curve.csv")))

    # 1 -- the paired difference
    su, sc = find_scores(unfrozen_dir), find_scores(control_dir)
    if su and sc:
        rep["paired_diff_unfrozen_minus_control"] = paired_diff(su, sc, target_fpr, B, seed)
    else:
        rep["paired_diff_unfrozen_minus_control"] = {
            "error": f"missing R1 score dumps (unfrozen={su}, control={sc})"}

    # the round-1 anchor this whole run is positioned against
    rep["round1_reference"] = {
        "run": "runs/bakeoff_freematch_5fold (matched 5-fold, frozen)",
        "fold0_tpr_at_step8000": 0.3877551020408163,
        "fold_mean_tpr": 0.347633073848096, "fold_sd": 0.023892451232600975,
        "note": ("The 0.357 quoted in the brief is the earlier fold-0 VARIANT SCREEN "
                 "(freematch_fold0_standard, on Drive), not this matched run. The phase-1 "
                 "head both runs load comes from the matched run, so 0.387755 is the "
                 "number step 0 must -- and does -- reproduce.")}
    return rep


def print_report(rep, unfrozen_dir, control_dir):
    print("\n" + "#" * 118)
    print("# FREEMATCH x n_blocks=3, FOLD-0 DIAGNOSTIC -- BRIEF §9 REQUIRED OUTPUTS")
    print(f"# {rep['status']}")
    print("#" * 118)

    print("\n--- 8. RUN PROVENANCE ---")
    for tag in ("unfrozen", "control"):
        r = rep[tag]
        g = r.get("step0_gate") or {}
        print(f"  [{tag}] n_blocks={r['n_blocks']}  steps={r['max_steps']}  "
              f"schedule={r['schedule_total_steps']}  eval={r['eval_precision']}")
        print(f"          head ||.||_2 = {r['init_head_l2']}  from {r['init_head_from']}")
        print(f"          checkpoints {r['n_checkpoints_saved']}/"
              f"{r['n_checkpoints_expected']}   curve rows {r['n_curve_rows']}")
        print(f"          step-0 gate: {g.get('passed')}  "
              f"(got {g.get('step0_tpr')}, expected {g.get('expected')})")
        fh = r.get("fp16_health") or {}
        if fh.get("steps_observed"):
            print(f"          fp16 skipped-step rate = "
                  f"{fh.get('skip_rate_fp16', 0):.4f}  "
                  f"({fh.get('steps_skipped_fp16')}/{fh.get('steps_observed')} updates "
                  f"discarded; worst window {fh.get('max_window_skip_rate', 0):.3f})")

    print("\n--- 1. HEADLINE (rates only; never counts) ---")
    print(f"  {'run':<12} | {'TPR@1e-3':>10} | {'AUROC':>8} | {'AUPRC':>8} | "
          f"{'purity@50% (1:2200)':>20} | {'comp@24%purity':>15}")
    print("  " + "-" * 96)
    for tag in ("unfrozen", "control"):
        r = rep[tag]
        print(f"  {tag:<12} | {_fmt(r['r1_tpr']):>10} | {_fmt(r['r1_auroc']):>8} | "
              f"{_fmt(r['r1_auprc']):>8} | {_fmt(r['r1_purity_corr']):>20} | "
              f"{_fmt(r['r1_comp_at_purity']):>15}")
    ref = rep["round1_reference"]
    print(f"  {'round-1':<12} | {_fmt(ref['fold0_tpr_at_step8000']):>10} |"
          f"          |          |                      |"
          f"    (frozen, fold 0)")

    pd = rep["paired_diff_unfrozen_minus_control"]
    print("\n  paired diff (unfrozen - control), class-conditional bootstrap:")
    if pd.get("error"):
        print(f"    n/a -- {pd['error']}")
    else:
        lo, hi = pd["ci"]
        print(f"    {pd['diff']:+.4f}   CI [{_fmt(lo)}, {_fmt(hi)}]   "
              f"half-width {_fmt(pd['ci_half_width'])}")
        print(f"    {pd['significance_note']}")

    print("\n--- 6. VERDICT ON THE CONFOUND (the control's flatness) ---")
    cv = rep["confound_verdict"]
    for s in cv["probe_steps"]:
        print(f"    step {s:>5}: TPR@1e-3 = {_fmt(cv['tpr_at_probe_steps'].get(s))}")
    print(f"    step     0: TPR@1e-3 = {_fmt(cv['step0_tpr'])}  (the loaded phase-1 model)")
    print(f"    spread = {_fmt(cv['spread'])}   max drift from step 0 = "
          f"{_fmt(cv['max_drift_from_step0'])}   one-positive resolution = "
          f"{cv['one_positive_resolution']:.4f}")
    print(f"    {cv['verdict']}")

    print("\n--- 2/7. CURVE SHAPE (the evidence for the 5-fold budget argument) ---")
    print(f"  {'run':<12} | {'argmax step':>11} | {'argmax':>8} | {'final':>8} | "
          f"{'rho(step,TPR)':>13} | {'sd':>8} | {'post-argmax dec':>15} | climbing?")
    print("  " + "-" * 108)
    for tag in ("unfrozen", "control"):
        c = rep[tag]["curve_shape"] or {}
        print(f"  {tag:<12} | {str(c.get('argmax_step')):>11} | "
              f"{_fmt(c.get('argmax_tpr')):>8} | {_fmt(c.get('final_tpr')):>8} | "
              f"{_fmt(c.get('spearman_step_tpr'),3):>13} | "
              f"{_fmt(c.get('whole_curve_sd')):>8} | "
              f"{_fmt(c.get('post_argmax_fraction_decreasing'),2):>15} | "
              f"{c.get('still_climbing_at_end')}")
    print("  NOTE (§7.8): per-checkpoint noise sd was ~0.046 on the unfrozen Baseline A "
          "curves,\n  comparable to the effect size being measured. A single argmax ~2 sd "
          "above its own curve\n  mean is the top of N noisy draws, not a peak.")

    print("\n--- 5. COLLAPSE-GATE AGREEMENT TRACE + FIRING DECISION ---")
    for tag in ("unfrozen", "control"):
        a = rep[tag]["agreement"]
        if a.get("available"):
            print(f"  [{tag}] first={a['first']:.4f}  last={a['last']:.4f}  "
                  f"min={a['min']:.4f} @ step {a['min_at_step']}  "
                  f"(round-1 self-training floor ~{a['round1_selftraining_floor']}; "
                  f"Lever-1 unfrozen lows {a['lever1_unfrozen_lows']})")
        else:
            print(f"  [{tag}] {a.get('note')}")
        print(f"         FIRED: {a.get('gate_fired')}")
        print(f"         {a.get('gate_verdict')}")
        ls = a.get("lockstep_check") or {}
        if ls.get("ok") is not None:
            print(f"         inline-vs-post-hoc lock-step: "
                  f"{'MATCH' if ls['ok'] else 'MISMATCH'} "
                  f"(max|diff| {ls.get('max_abs_diff')}, {ls.get('n_compared')} points)")
        elif ls.get("skipped"):
            print(f"         lock-step check: skipped by construction (n_blocks>0)")
    print(f"  {rep['unfrozen']['agreement'].get('note', '')}")

    print("\n--- 4. MEMORIZATION PROBE (§5.4) ---")
    for tag in ("unfrozen", "control"):
        m = rep[tag]["memorization"]
        if not m.get("available"):
            print(f"  [{tag}] (no memorization.csv)")
            continue
        print(f"  [{tag}] target conf {m['target_conf_first']:.4f} -> "
              f"{m['target_conf_last']:.4f}   matched-test conf "
              f"{m['test_conf_first']:.4f} -> {m['test_conf_last']:.4f}")
        print(f"          gap (target - matched test) {m['gap_first']:+.4f} -> "
              f"{m['gap_last']:+.4f}   change {m['gap_change']:+.4f}   "
              f"rho(step,gap) = {_fmt(m['spearman_step_gap'],3)}")
    print(f"  {rep['unfrozen']['memorization'].get('reading', '')}")

    print("\n--- 3. FREEMATCH-INTERNAL DIAGNOSTICS (§5.3) ---")
    for tag, d in (("unfrozen", unfrozen_dir), ("control", control_dir)):
        rows = read_csv(os.path.join(d, "diagnostics.csv"))
        print_table(rows, DIAG_SHOW, f"  [{tag}] {os.path.join(d, 'diagnostics.csv')}")

    print("\n--- 7. THE 5-FOLD BUDGET RECOMMENDATION ---")
    print("  NOT written by this script. It is a judgement to be argued from the curve")
    print("  above and PRE-REGISTERED before any confirmatory run starts (§7.8's")
    print("  exploratory/confirmatory split: fold 0 is the development fold).")
    print("  Read: still-climbing-at-end => argue a longer budget; a genuine early peak")
    print("  followed by decay => the BRIEF §4 memorization signature; flat => keep 8000,")
    print("  which is also the only budget comparable to every existing paired result.")


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unfrozen", default=os.path.join(root, "runs",
                                                       "freematch_fold0_nb3_unfrozen"))
    ap.add_argument("--control", default=os.path.join(root, "runs",
                                                      "freematch_fold0_nb0_control"))
    ap.add_argument("--out", default=None, help="where to write results.json "
                                                "(default: <unfrozen>/../report_freematch_nb3_fold0.json)")
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    unf, ctl = os.path.abspath(args.unfrozen), os.path.abspath(args.control)
    rep = build_report(unf, ctl, args.target_fpr, args.bootstrap, args.seed)
    print_report(rep, unf, ctl)

    out = os.path.abspath(args.out or os.path.join(
        os.path.dirname(unf), "report_freematch_nb3_fold0.json"))
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
