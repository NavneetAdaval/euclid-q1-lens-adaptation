"""
aggregate_folds.py
==================
Aggregate whatever FreeMatch x nb3 fold archives are on disk, and run the LABEL-FREE
SELECTOR ANALYSIS. Runs locally, works on a PARTIAL set -- folds arrive one Kaggle session
at a time, so this reports on what is present and names what is missing rather than waiting
for all five.

Reads the pre-registration (PROJECT.md §7.11, frozen 2026-08-04) and holds to it:

  * FOLD 0 IS DEVELOPMENT. Its budget was chosen by inspecting fold-0 test labels, so it
    never enters the headline. Reported separately, labelled, and excluded from every
    aggregate.
  * The confirmatory comparison is FOLDS 1-4 unfrozen vs ROUND 1's folds 1-4 -- not the
    5-fold mean. 391 positives, half-width ~0.061.
  * Selector is R1 (last checkpoint). The R1/R2/R3 span is reported as a robustness note,
    because §7.8 measured those rules spanning 0.0816 on nb3 with the ordering flipping.
  * Primary TPR@FPR=1e-3; secondary completeness@24%-purity and purity@50%-completeness at
    1:2200. AUPRC is never a selector.
  * Rates only. No counts, including for effect sizes and CI half-widths (§9).

THE LABEL-FREE SELECTOR ANALYSIS (a PRIMARY output, §7.11)
----------------------------------------------------------
Two quantities computable with NO test labels moved monotonically alongside fold 0's decay:
the MEMORIZATION GAP (target-pool minus composition-matched test-pool confidence) and TAU
SATURATION (FreeMatch's self-adaptive threshold). The question this answers is:

    does a FIXED threshold on the memorization gap, or a FIXED tau value, pick a checkpoint
    within noise of the TPR-optimal one, CONSISTENTLY across folds 1-4?

If yes, that is a stopping rule needing no test labels -- and it would explain §7.6's
selector null structurally rather than as a failure of the validators: the frozen regime is
metric-degenerate (round 1's head moved 4.2% of its norm while TPR stayed within 3
positives), so there was nothing for any selector to track, whereas the unfrozen regime has
a range of 28 positives and Spearman(step, TPR) = -0.659.

Thresholds are fit on FOLD 0 ONLY (development) and applied UNCHANGED to folds 1-4. Fitting
them on the folds they are then evaluated on would be exactly the oracle selection §7.6
warns about.

Run:
    python scripts/aggregate_folds.py
    python scripts/aggregate_folds.py --runs-dir runs --glob "freematch_nb3_unfrozen_fold*"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import (metric_suite, paired_bootstrap, default_tpr_metric,
                     DEFAULT_TARGET_FPR, DEFAULT_TARGET_PURITY,
                     DEFAULT_TARGET_COMPLETENESS, NN_NP_Q1)      # noqa: E402

ALL_FOLDS = (0, 1, 2, 3, 4)
CONFIRMATORY = (1, 2, 3, 4)
DEV_FOLD = 0
ONE_POSITIVE = 1.0 / 98.0
LINES_SIMONLY_COMPLETENESS = 0.50          # arXiv:2512.05899

# PRE-REGISTERED (§7.11): round 1's per-fold R1 values, the comparison arm
ROUND1_TPR = {0: 0.3877551020408163, 1: 0.336734693877551, 2: 0.35714285714285715,
              3: 0.3163265306122449, 4: 0.3402061855670103}
ROUND1_FOLDS_1_4_MEAN = 0.337603
ROUND1_COMP24 = {0: 0.2653, 1: 0.0816, 2: 0.0816, 3: 0.1122, 4: 0.1649}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def find_folds(runs_dir, pattern):
    """Map fold -> run dir for whatever is present. Silent about nothing."""
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, pattern))):
        if not os.path.isdir(d):
            continue
        cfg = os.path.join(d, "run_config.json")
        k = None
        if os.path.isfile(cfg):
            try:
                k = json.load(open(cfg)).get("fold")
            except Exception:
                k = None
        if k is None:
            m = re.search(r"fold(\d+)", os.path.basename(d))
            k = int(m.group(1)) if m else None
        if k is not None:
            out[int(k)] = d
    return out


def read_curve(run_dir):
    p = os.path.join(run_dir, "curve.csv")
    if not os.path.isfile(p):
        return []
    return [{k: _f(v) for k, v in r.items()} for r in csv.DictReader(open(p))]


def read_csv_rows(path):
    return ([{k: _f(v) for k, v in r.items()} for r in csv.DictReader(open(path))]
            if os.path.isfile(path) else [])


def _f(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def suite_at(run_dir, fold, step):
    """Full metric suite at one curve point, from that point's logit dump."""
    p = os.path.join(run_dir, "logits", f"step_{int(step)}.npz")
    if not os.path.isfile(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        pos, neg = z[f"test_pos_fold{fold}"], z["test_negatives_shared"]
    y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
    s = np.concatenate([pos, neg]).astype(float)
    return metric_suite(y, s, target_fpr=DEFAULT_TARGET_FPR,
                        target_completeness=DEFAULT_TARGET_COMPLETENESS,
                        nn_np_ratio=NN_NP_Q1, target_purity=DEFAULT_TARGET_PURITY)


def scores_at(run_dir, fold, step):
    p = os.path.join(run_dir, "logits", f"step_{int(step)}.npz")
    if not os.path.isfile(p):
        return None, None
    with np.load(p, allow_pickle=False) as z:
        return z[f"test_pos_fold{fold}"], z["test_negatives_shared"]


# ---------------------------------------------------------------------------
# per-fold summary
# ---------------------------------------------------------------------------

def summarise_fold(fold, run_dir):
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    res = json.load(open(os.path.join(run_dir, "results.json")))
    curve = read_curve(run_dir)
    steps = [int(r["step"]) for r in curve]
    tpr = np.array([r.get(f"fold{fold}_tpr_1e-3") for r in curve], dtype=float)
    agr = np.array([r.get(f"agreement_fold{fold}") for r in curve], dtype=float)
    r1_step = int(cfg.get("max_steps") or max(steps))

    # R1 / R2 / R3 -- the robustness note §7.8 requires
    valid = [(s, t) for s, t in zip(steps, tpr) if s > 0 and np.isfinite(t)]
    r1 = dict(valid)[r1_step] if r1_step in dict(valid) else (valid[-1][1] if valid else None)
    r2 = float(np.median([t for _s, t in valid[len(valid)//2:]])) if valid else None
    r3 = float(np.median([t for _s, t in valid])) if valid else None

    gate_p = os.path.join(run_dir, f"gate_fold{fold}.json")
    gate = json.load(open(gate_p)) if os.path.isfile(gate_p) else {}
    inst = cfg.get("instrument") or {}
    st = res.get("freematch_stability") or {}

    return {
        "fold": fold, "dir": run_dir, "is_development": fold == DEV_FOLD,
        "n_blocks": cfg.get("n_blocks"), "max_steps": cfg.get("max_steps"),
        "step0_gate": cfg.get("step0_gate"),
        "init_head_l2": cfg.get("init_head_l2"),
        "n_checkpoints_saved": cfg.get("n_checkpoints_saved"),
        "n_checkpoints_expected": cfg.get("n_checkpoints_expected"),
        "peak_gpu_gb": cfg.get("peak_gpu_gb"), "train_minutes": res.get("train_minutes"),
        "fp16_skip_rate": inst.get("skip_rate_fp16"),
        "fp16_worst_window": inst.get("max_window_skip_rate"),
        "mu_eff_at_step0": st.get("mu_eff_at_step0"),
        "sat_tau_rebuilt": ((st.get("sat_reconstruction") or {}).get("tau_direct")),
        "r1_step": r1_step, "r1_tpr": r1, "r2_tpr": r2, "r3_tpr": r3,
        "selector_span": (None if None in (r1, r2, r3) else max(r1, r2, r3) - min(r1, r2, r3)),
        "r1_suite": suite_at(run_dir, fold, r1_step),
        "round1_tpr": ROUND1_TPR.get(fold),
        "curve_steps": steps, "curve_tpr": tpr.tolist(), "curve_agreement": agr.tolist(),
        "argmax_step": int(steps[int(np.nanargmax(tpr))]) if valid else None,
        "argmax_tpr": float(np.nanmax(tpr)) if valid else None,
        "gate_fired": gate.get("fired"),
        "agreement_first": gate.get("agreement_first"),
        "agreement_last": gate.get("agreement_last"),
        "agreement_min": gate.get("agreement_min"),
    }


# ---------------------------------------------------------------------------
# THE LABEL-FREE SELECTOR ANALYSIS (§7.11 primary output)
# ---------------------------------------------------------------------------

def selector_series(run_dir, fold):
    """The two label-free trajectories, aligned to the curve's steps."""
    memo = {int(r["step"]): r for r in read_csv_rows(os.path.join(run_dir,
                                                                 "memorization.csv"))}
    diag = {int(r["step"]): r for r in read_csv_rows(os.path.join(run_dir,
                                                                 "diagnostics.csv"))}
    curve = read_curve(run_dir)
    steps, gap, tau, tpr = [], [], [], []
    for r in curve:
        s = int(r["step"])
        g = (memo.get(s) or {}).get("target_minus_test_matched")
        t = (diag.get(s) or {}).get("tau_global_last")
        v = r.get(f"fold{fold}_tpr_1e-3")
        if g is None or t is None or v is None:
            continue
        steps.append(s); gap.append(float(g)); tau.append(float(t)); tpr.append(float(v))
    return (np.array(steps), np.array(gap), np.array(tau), np.array(tpr))


def turn_points(steps, gap, tau, tpr):
    """Where each trajectory turns, relative to where TPR peaks."""
    if steps.size == 0:
        return {}
    i_tpr = int(np.argmax(tpr))
    # the label-free series are monotone-ish, so "turns" = first crossing of the midpoint
    def first_cross(v):
        if v.size < 2:
            return None
        mid = 0.5 * (v[0] + v[-1])
        up = v[-1] >= v[0]
        idx = np.where(v >= mid)[0] if up else np.where(v <= mid)[0]
        return int(steps[idx[0]]) if idx.size else None
    return {"tpr_peak_step": int(steps[i_tpr]), "tpr_peak": float(tpr[i_tpr]),
            "gap_midpoint_step": first_cross(gap), "tau_midpoint_step": first_cross(tau),
            "gap_at_tpr_peak": float(gap[i_tpr]), "tau_at_tpr_peak": float(tau[i_tpr]),
            "gap_first": float(gap[0]), "gap_last": float(gap[-1]),
            "tau_first": float(tau[0]), "tau_last": float(tau[-1]),
            "spearman_step_tpr": _spear(steps, tpr),
            "spearman_gap_tpr": _spear(gap, tpr), "spearman_tau_tpr": _spear(tau, tpr)}


def _spear(x, y):
    if len(x) < 3:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def fit_thresholds_on_dev(steps, gap, tau, tpr):
    """Read the gap / tau values AT the TPR-optimal checkpoint on the DEVELOPMENT fold.

    That is the whole rule: "stop when the memorization gap first exceeds G" / "stop when
    tau first exceeds T", with G and T taken from where fold 0's TPR peaked. Fitting them
    anywhere else -- e.g. per fold -- would be oracle selection.
    """
    i = int(np.argmax(tpr))
    return {"gap_threshold": float(gap[i]), "tau_threshold": float(tau[i]),
            "fitted_on_fold": DEV_FOLD, "fitted_at_step": int(steps[i]),
            "note": ("thresholds are read off the DEVELOPMENT fold's TPR-optimal checkpoint "
                     "and applied UNCHANGED to the confirmatory folds")}


def apply_rule(steps, series, threshold, rising=True):
    """First step at which the series crosses the threshold -- the stopping rule."""
    idx = np.where(series >= threshold)[0] if rising else np.where(series <= threshold)[0]
    return int(steps[idx[0]]) if idx.size else None


def evaluate_rule(steps, tpr, picked_step):
    """Regret of a picked checkpoint against the oracle, in TPR units (a RATE)."""
    if picked_step is None:
        return None
    j = int(np.argmin(np.abs(steps - picked_step)))
    return {"picked_step": int(steps[j]), "picked_tpr": float(tpr[j]),
            "oracle_tpr": float(np.max(tpr)),
            "regret": float(np.max(tpr) - tpr[j])}


def selector_analysis(folds):
    dev = folds.get(DEV_FOLD)
    out = {"available": bool(dev), "per_fold_turns": {}, "rule": None, "results": {}}
    for k, d in sorted(folds.items()):
        s, g, t, v = selector_series(d["dir"], k)
        out["per_fold_turns"][k] = turn_points(s, g, t, v)
    if not dev:
        out["note"] = (f"fold {DEV_FOLD} (development) is not present, so the thresholds "
                       f"cannot be fitted where the pre-registration requires. Turn points "
                       f"are reported per fold; no rule is evaluated.")
        return out

    s0, g0, t0, v0 = selector_series(dev["dir"], DEV_FOLD)
    if s0.size == 0:
        out["note"] = "fold 0 has no aligned memorization/diagnostics series"
        return out
    rule = fit_thresholds_on_dev(s0, g0, t0, v0)
    out["rule"] = rule

    for k, d in sorted(folds.items()):
        s, g, t, v = selector_series(d["dir"], k)
        if s.size == 0:
            continue
        gp = apply_rule(s, g, rule["gap_threshold"], rising=True)
        tp = apply_rule(s, t, rule["tau_threshold"], rising=True)
        out["results"][k] = {
            "is_development": k == DEV_FOLD,
            "memorization_gap_rule": evaluate_rule(s, v, gp),
            "tau_rule": evaluate_rule(s, v, tp),
            # THE STEP CONTROL, and it is not optional. Both label-free series are close to
            # monotone in the step, so a rule keyed to either can look predictive while
            # carrying no information beyond "stop early". §7.6 established the discipline
            # after saturated source-val AUROC scored Spearman +0.54 against real-test TPR
            # purely by inheriting the shared training-time trend: THE NULL FOR A
            # TRAJECTORY-BASED SELECTOR IS THE STEP, NOT ZERO. A gap or tau rule earns its
            # keep only by beating "stop at the step the development fold peaked at".
            "fixed_step_control": evaluate_rule(s, v, rule["fitted_at_step"]),
            "r1_tpr": float(v[-1]), "oracle_tpr": float(np.max(v)),
            "r1_regret": float(np.max(v) - v[-1]),
        }
    conf = [r for k, r in out["results"].items() if k in CONFIRMATORY]
    if conf:
        def agg(key):
            xs = [c[key]["regret"] for c in conf if c.get(key)]
            return {"n": len(xs), "mean_regret": float(np.mean(xs)) if xs else None,
                    "max_regret": float(np.max(xs)) if xs else None}
        out["confirmatory_summary"] = {
            "memorization_gap_rule": agg("memorization_gap_rule"),
            "tau_rule": agg("tau_rule"),
            "fixed_step_control": agg("fixed_step_control"),
            "R1_incumbent": {"n": len(conf),
                             "mean_regret": float(np.mean([c["r1_regret"] for c in conf])),
                             "max_regret": float(np.max([c["r1_regret"] for c in conf]))},
            "one_positive": ONE_POSITIVE,
        }
    return out


# ---------------------------------------------------------------------------
# confirmatory aggregate
# ---------------------------------------------------------------------------

def confirmatory(folds, bootstrap=2000, seed=0):
    have = [k for k in CONFIRMATORY if k in folds]
    missing = [k for k in CONFIRMATORY if k not in folds]
    out = {"folds_present": have, "folds_missing": missing,
           "complete": not missing, "n_expected": len(CONFIRMATORY)}
    if not have:
        return out
    tprs = [folds[k]["r1_tpr"] for k in have if folds[k]["r1_tpr"] is not None]
    out["mean_tpr"] = float(np.mean(tprs)) if tprs else None
    out["sd_tpr"] = float(np.std(tprs)) if len(tprs) > 1 else None
    out["per_fold_tpr"] = {k: folds[k]["r1_tpr"] for k in have}
    out["round1_per_fold"] = {k: ROUND1_TPR[k] for k in have}
    out["round1_mean"] = float(np.mean([ROUND1_TPR[k] for k in have]))
    out["paired_diff_mean"] = (None if out["mean_tpr"] is None
                               else out["mean_tpr"] - out["round1_mean"])
    for key, label in (("comp_at_purity", "comp@24%-purity"),
                       ("purity_corr", "purity@50%-completeness")):
        vals = [folds[k]["r1_suite"][key] for k in have
                if folds[k].get("r1_suite") and folds[k]["r1_suite"].get(key) is not None
                and np.isfinite(folds[k]["r1_suite"][key])]
        out[key] = {"per_fold": {k: (folds[k]["r1_suite"] or {}).get(key) for k in have},
                    "mean": float(np.mean(vals)) if vals else None,
                    "n_defined": len(vals), "n_folds": len(have), "label": label}
    if out[  # completeness vs the Lines reference
            "comp_at_purity"]["mean"] is not None:
        out["comp_at_purity"]["fraction_of_lines"] = (
            out["comp_at_purity"]["mean"] / LINES_SIMONLY_COMPLETENESS)
    out["round1_comp24_mean"] = float(np.mean([ROUND1_COMP24[k] for k in have]))

    # per-fold paired bootstrap against that fold's round-1 scores
    diffs = {}
    for k in have:
        pos, neg = scores_at(folds[k]["dir"], k, folds[k]["r1_step"])
        r1p = os.path.join(_HERE, "..", "runs", "bakeoff_freematch_5fold", "scores",
                           f"freematch_fold{k}.npz")
        if pos is None or not os.path.isfile(r1p):
            diffs[k] = {"error": "missing scores for a paired bootstrap"}
            continue
        with np.load(r1p, allow_pickle=False) as z:
            rp, rn = z["pos_scores"], z["neg_scores"]
        if rp.size != pos.size or rn.size != neg.size:
            diffs[k] = {"error": f"length mismatch ({pos.size},{neg.size}) vs ({rp.size},{rn.size})"}
            continue
        y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
        bs = paired_bootstrap(y, np.concatenate([pos, neg]).astype(float),
                              np.concatenate([rp, rn]).astype(float),
                              metric_fn=default_tpr_metric(DEFAULT_TARGET_FPR),
                              B=bootstrap, seed=seed)
        diffs[k] = {"diff": float(bs.get("point", np.nan)), "lo": bs.get("lo"),
                    "hi": bs.get("hi")}
    out["per_fold_paired_diff_vs_round1"] = diffs
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def report(folds, conf, sel):
    print("\n" + "#" * 104)
    print("# FREEMATCH x nb3 -- 5-FOLD CONFIRMATORY AGGREGATE")
    print("# Pre-registration: PROJECT.md §7.11 (frozen 2026-08-04). Fold 0 is DEVELOPMENT.")
    print("#" * 104)

    present = sorted(folds)
    print(f"\nfolds present: {present}   missing: "
          f"{[k for k in ALL_FOLDS if k not in folds]}")
    if not conf["complete"]:
        print(f"*** PARTIAL SET -- confirmatory folds {conf['folds_missing']} not yet run. "
              f"Everything below is provisional. ***")

    print("\n--- PER-FOLD PROVENANCE ---")
    print(f"  {'fold':>4} {'gate':>6} {'step0 TPR':>10} {'||head||2':>10} {'ckpt':>7} "
          f"{'skip':>8} {'peak GB':>8} {'min':>7} {'mu@0':>5}")
    for k in present:
        f = folds[k]
        g = f.get("step0_gate") or {}
        print(f"  {k:>4}{' (dev)' if f['is_development'] else '     '} "
              f"{str(g.get('passed')):>6} {_fmt(g.get('step0_tpr'),6):>10} "
              f"{_fmt(f['init_head_l2'],6):>10} "
              f"{f['n_checkpoints_saved']}/{f['n_checkpoints_expected']:<4} "
              f"{_fmt(f['fp16_skip_rate'],5):>8} {_fmt(f['peak_gpu_gb'],2):>8} "
              f"{_fmt(f['train_minutes'],1):>7} {_fmt(f['mu_eff_at_step0'],2):>5}")

    print("\n--- PER-FOLD RESULT AT R1 (rates) ---")
    print(f"  {'fold':>4} {'R1 TPR':>9} {'round1':>9} {'diff':>9} {'argmax':>9} "
          f"{'@step':>7} {'comp@24%':>9} {'pur@50%':>9} {'R1/R2/R3 span':>14}")
    for k in present:
        f = folds[k]
        su = f.get("r1_suite") or {}
        d = (None if f["r1_tpr"] is None or f["round1_tpr"] is None
             else f["r1_tpr"] - f["round1_tpr"])
        tag = " (dev)" if f["is_development"] else ""
        print(f"  {k:>4} {_fmt(f['r1_tpr']):>9} {_fmt(f['round1_tpr']):>9} "
              f"{_fmt(d):>9} {_fmt(f['argmax_tpr']):>9} {str(f['argmax_step']):>7} "
              f"{_fmt(su.get('comp_at_purity')):>9} {_fmt(su.get('purity_corr')):>9} "
              f"{_fmt(f['selector_span']):>14}{tag}")

    print("\n--- CONFIRMATORY AGGREGATE (folds 1-4 only; fold 0 EXCLUDED by pre-registration) ---")
    if conf.get("mean_tpr") is None:
        print("  no confirmatory folds present yet")
    else:
        print(f"  folds present {conf['folds_present']}  (missing {conf['folds_missing']})")
        print(f"  unfrozen  mean TPR@1e-3 = {_fmt(conf['mean_tpr'])} "
              f"+- {_fmt(conf['sd_tpr'])}")
        print(f"  round 1   mean TPR@1e-3 = {_fmt(conf['round1_mean'])}")
        print(f"  paired diff (fold-mean)  = {_fmt(conf['paired_diff_mean'], 4)}")
        print("\n  per-fold paired bootstrap vs round 1 (class-conditional):")
        for k, v in (conf.get("per_fold_paired_diff_vs_round1") or {}).items():
            if v.get("error"):
                print(f"    fold {k}: n/a -- {v['error']}")
            else:
                print(f"    fold {k}: {v['diff']:+.4f}  CI [{_fmt(v['lo'])}, {_fmt(v['hi'])}]")
        c = conf["comp_at_purity"]
        print(f"\n  comp@24%-purity mean {_fmt(c['mean'])} "
              f"(defined {c['n_defined']}/{c['n_folds']}) vs round 1 "
              f"{_fmt(conf['round1_comp24_mean'])}")
        if c.get("fraction_of_lines"):
            print(f"    = {c['fraction_of_lines']:.1%} of Lines' sim-only "
                  f"{LINES_SIMONLY_COMPLETENESS:.2f}")
        p = conf["purity_corr"]
        print(f"  purity@50%-completeness mean {_fmt(p['mean'])} "
              f"(defined {p['n_defined']}/{p['n_folds']})")

    print("\n--- LABEL-FREE SELECTOR ANALYSIS (§7.11 PRIMARY OUTPUT) ---")
    if not sel.get("rule"):
        print(f"  {sel.get('note', 'unavailable')}")
    else:
        r = sel["rule"]
        print(f"  thresholds fitted on fold {r['fitted_on_fold']} (DEVELOPMENT) at its "
              f"TPR-optimal step {r['fitted_at_step']}:")
        print(f"    memorization gap >= {r['gap_threshold']:+.4f}   tau >= "
              f"{r['tau_threshold']:.4f}")
        print(f"\n  {'fold':>4} {'rule':>18} {'picks step':>11} {'TPR there':>10} "
              f"{'oracle':>8} {'regret':>8}")
        for k in sorted(sel["results"]):
            row = sel["results"][k]
            tag = " (dev, fitted here)" if row["is_development"] else ""
            for name, key in (("memorization gap", "memorization_gap_rule"),
                              ("tau", "tau_rule"),
                              ("fixed-step CONTROL", "fixed_step_control")):
                e = row.get(key)
                if not e:
                    print(f"  {k:>4} {name:>18} {'never crosses':>11}")
                    continue
                print(f"  {k:>4} {name:>18} {e['picked_step']:>11} "
                      f"{e['picked_tpr']:>10.4f} {e['oracle_tpr']:>8.4f} "
                      f"{e['regret']:>8.4f}{tag}")
            print(f"  {k:>4} {'R1 (incumbent)':>18} {'last':>11} "
                  f"{row['r1_tpr']:>10.4f} {row['oracle_tpr']:>8.4f} "
                  f"{row['r1_regret']:>8.4f}")
        cs = sel.get("confirmatory_summary")
        if cs:
            print(f"\n  CONFIRMATORY (folds 1-4), mean regret in TPR units "
                  f"(one positive = {cs['one_positive']:.4f}):")
            for key, lab in (("memorization_gap_rule", "memorization gap rule"),
                             ("tau_rule", "tau rule"),
                             ("fixed_step_control", "fixed-step CONTROL (the null)"),
                             ("R1_incumbent", "R1 (incumbent)")):
                v = cs[key]
                print(f"    {lab:<24} n={v['n']}  mean {_fmt(v['mean_regret'])}  "
                      f"max {_fmt(v['max_regret'])}")
            print("\n  A rule beats R1 here only if its mean regret is LOWER and it is "
                  "consistent\n  across folds -- a rule that wins on average by winning "
                  "hugely on one fold is not\n  a stopping rule. §7.6's random-selection "
                  "band is the reference, not zero.")

    print("\n--- TURN POINTS: where each label-free series moves vs where TPR peaks ---")
    print(f"  {'fold':>4} {'TPR peak':>9} {'@step':>7} {'gap mid @':>10} {'tau mid @':>10} "
          f"{'rho(gap,TPR)':>13} {'rho(tau,TPR)':>13}")
    for k, t in sorted((sel.get("per_fold_turns") or {}).items()):
        if not t:
            continue
        print(f"  {k:>4} {t['tpr_peak']:>9.4f} {t['tpr_peak_step']:>7} "
              f"{str(t['gap_midpoint_step']):>10} {str(t['tau_midpoint_step']):>10} "
              f"{_fmt(t['spearman_gap_tpr'],3):>13} {_fmt(t['spearman_tau_tpr'],3):>13}")

    print("\n--- COLLAPSE-GATE TRACE (read the TRACE, not the verdict) ---")
    print(f"  {'fold':>4} {'fired':>7} {'first':>8} {'last':>8} {'min':>8}")
    for k in present:
        f = folds[k]
        print(f"  {k:>4} {str(f['gate_fired']):>7} {_fmt(f['agreement_first']):>8} "
              f"{_fmt(f['agreement_last']):>8} {_fmt(f['agreement_min']):>8}")
    print("  A 0.52 cliff that recovers does NOT fire the gate (measured, §7.6). Silence is "
          "not\n  evidence of no collapse.")


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs"))
    ap.add_argument("--glob", default="freematch_nb3_unfrozen_fold*")
    ap.add_argument("--also", nargs="*", default=[],
                    help="extra run dirs to include (e.g. the original fold-0 run)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(root, "runs", "fold_aggregate.json"))
    args = ap.parse_args(argv)

    found = find_folds(os.path.abspath(args.runs_dir), args.glob)
    for extra in args.also:
        d = os.path.abspath(extra)
        cfgp = os.path.join(d, "run_config.json")
        if os.path.isfile(cfgp):
            k = json.load(open(cfgp)).get("fold", DEV_FOLD)
            found.setdefault(int(k), d)

    if not found:
        print(f"no fold runs matched {args.glob!r} under {args.runs_dir}")
        print(f"expected any of: {[f'freematch_nb3_unfrozen_fold{k}' for k in ALL_FOLDS]}")
        return 1

    folds = {}
    for k, d in sorted(found.items()):
        try:
            folds[k] = summarise_fold(k, d)
        except Exception as e:
            print(f"[warn] fold {k} at {d} could not be summarised: "
                  f"{type(e).__name__}: {e}")
    conf = confirmatory(folds, args.bootstrap, args.seed)
    sel = selector_analysis(folds)
    report(folds, conf, sel)

    payload = {"pre_registration": "PROJECT.md §7.11 (frozen 2026-08-04)",
               "folds": {k: {kk: vv for kk, vv in v.items()
                             if kk not in ("curve_tpr", "curve_agreement", "curve_steps")}
                         for k, v in folds.items()},
               "confirmatory": conf, "selector_analysis": sel}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
