"""
run_coral_r4_confirm.py
=======================
ROUND-4 CORAL **CONFIRMATORY** RUN -- one fold per process.

    coral_nb3, gamma = 16, budget 2000 steps, folds 1 / 2 / 3 / 4.
    FOLD 0 IS DEVELOPMENT AND CANNOT BE SELECTED. The runner refuses it at argument
    parsing, not deep in the run: the budget was chosen by inspecting fold 0's own curve,
    so letting fold 0 into the confirmatory set would be oracle selection.

THE PRE-REGISTRATION GOVERNS THIS SCRIPT, NOT THE OTHER WAY AROUND
------------------------------------------------------------------
`runs/coral_r4_budget.json` is hashed. At startup this script re-computes that hash and
**refuses to run if it does not check**. The budget is then read OUT of the verified file
rather than typed here, so the number that trains is the number that was pre-registered:

    budget            2000 steps
    read-off rule     train to 2000, read the FINAL checkpoint (step_2000.pt)
    derived/declared  DECLARED, on cost -- the fold-0 development curve could not locate a
                      budget (decay detector Delta = +0.0278, CI [-0.0144, +0.0702]; the
                      pairwise survivors were peak-vs-trough pairs of a wobbling curve)

**THE BUDGET IS THE SELECTOR.** There is no checkpoint-selection code in this path: no
argmax, no smoothing, no label-free rule, no "best" anything. Every training segment runs
inside `coral_round4.test_label_lockout()`, and the only test-set metric computed anywhere
is the step-0 plumbing gate, which is a KNOWN NULL (it must reproduce round-1 CORAL's own
per-fold number, and that is the entire reason it exists).

    The eight 250-step checkpoints are LAB RECORD AND DIAGNOSTICS ONLY.
    THE REPORTED RESULT IS step_2000 AND NOTHING ELSE.

CONFIGURATION -- READ FROM THE FOLD-0 ARM, NOT RETYPED
------------------------------------------------------
The arm is built through `run_coral_round4.build_arm("coral_nb3", ...)` with
`for_calibration=True` and `trainer.cfg.coral_weight` set afterwards -- the same two lines
the sweep and the fold-0 development run used, so the configuration is the pinned one by
construction. It is then diffed field by field against the fold-0 development arm's stored
`run.json`, and the run REFUSES TO START unless every difference is on this allowlist:

    fold                 the variable under test
    max_steps            8000 (development maximum) -> 2000 (the pre-registered budget)
    init_head_from       each fold LP->FT-inits from ITS OWN round-1 CORAL step-8000 head
    init_head_l2         ...and therefore from its own head norm (PROJECT.md 7.11)
    epochs_planned       derived from max_steps, which is itself allowlisted

Anything else is a hard refusal. The allowlist and the pinned-key list are PARAMETERS of the
check, never module globals closed over -- `runs/coral_round4_design.md` §5 records a
near-miss where a check imported from another module silently compared that module's key
list and let a gamma mismatch through.

PER-FOLD, NOT PER-PROJECT
-------------------------
Folds 0-3 hold 98 test positives; **fold 4 holds 97** (489 lenses over 5 folds, each tested
exactly once). So the one-positive resolution is 1/98 = 0.010204 on folds 1-3 and
1/97 = 0.010309 on fold 4, and each fold gates against its own `FOLD_GATE` entry. The count
is resolved from `splits.json` at startup, cross-checked against the live-encoded pool, and
printed in the per-fold header so a reader can check it from the log alone.

OUTPUTS (per fold, under --out-dir/fold{k}; the tree drops into runs/ unrenamed)
--------------------------------------------------------------------------------
    run.json            config, the verified pre-registration, the gate, timings, peak mem
    config_diff.json    field-by-field against the fold-0 arm
    trace.csv           per --log-every steps      steps.csv   per step
    checkpoints.csv     per checkpoint: per-domain mean ||f||, per-domain mean per-dimension
                        std, raw CORAL, sup_loss
    checkpoints/step_*.pt          every 250 steps
    logits/ids.npz                 pos_ids / neg_ids for THIS fold
    logits/true_init.npz           the LP->FT head at step 0
    logits/g16_2000_fold{k}/step_*.npz    pos / neg logits at every checkpoint
    resume/latest.pt               weights + AdamW moments + GradScaler + DA buffers + RNG

SINGLE GPU PER FOLD. The four folds are independent and the launcher runs two at a time,
one per device via CUDA_VISIBLE_DEVICES. A single fold is never split across devices: there
is no DataParallel/DDP anywhere in this path, because halving the per-device batch would
change CORAL's 640x640 covariance estimate from 256 draws to 128 (§7.1a, the same reason
gradient accumulation is refused).

Run (one fold per process):
    python -u scripts/run_coral_r4_confirm.py --fold 1 --out-dir /kaggle/working/runs/coral_r4_confirm
    python -u scripts/run_coral_r4_confirm.py --fold 1 --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from coral_round4 import (DEFAULT_FEAT_NORM_CEILING, DEFAULT_FEAT_NORM_FLOOR,   # noqa: E402
                          install_fp32_coral)
import run_coral_round4 as R4                                                   # noqa: E402
import run_coral_r4_dev8000 as DEV                                              # noqa: E402
from run_coral_r4_dev8000 import (Paths, ReproCheck, Scorer, _find_under,       # noqa: E402
                                  _path_tail, _rng_state, _sha256, _same,
                                  banner, build_record, load_resume, train_loop)

CONFIRMATORY_FOLDS = (1, 2, 3, 4)
ARM_KEY = "coral_nb3"
GAMMA = 16.0
LOG_EVERY = 125
CKPT_EVERY = 250
PREREG_BASENAME = "coral_r4_budget.json"
DEV_ARM_RUN_JSON = "coral_r4_nb3_dev8000_fold0"     # the folder its run.json sits in


def logit_arm(fold, budget):
    return f"g16_{int(budget)}_fold{int(fold)}"


# ===========================================================================
# the pre-registration gate
# ===========================================================================

def verify_prereg(path, fold):
    """Re-hash `runs/coral_r4_budget.json` and REFUSE to run if it does not check.

    The hash is computed the way the file itself specifies: pop the digest field, dump the
    remainder sorted and compact, sha256 that. The budget and the read-off rule are then
    read OUT of the verified file, so what trains is what was pre-registered -- not a
    constant retyped in this script that could drift away from it.
    """
    if not path or not os.path.isfile(path):
        raise SystemExit(
            f"[confirm] ABORT: the pre-registration was not found ({path}).\n"
            f"  PROJECT.md 7.14 requires it to exist and be hashed BEFORE any confirmatory "
            f"fold runs. Without it this run has no fixed read-off point and is not "
            f"confirmatory. Pass --prereg.")
    with open(path) as fh:
        body = json.load(fh)
    recorded = body.pop("sha256_of_body_without_this_field", None)
    if recorded is None:
        raise SystemExit(f"[confirm] ABORT: {path} carries no hash field. It is not a "
                         f"pre-registration.")
    recomputed = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if recorded != recomputed:
        raise SystemExit(
            f"[confirm] ABORT: THE PRE-REGISTRATION HASH DOES NOT CHECK.\n"
            f"  file      : {path}\n"
            f"  recorded  : {recorded}\n"
            f"  recomputed: {recomputed}\n"
            f"  The file has been altered since it was hashed. A pre-registration that does "
            f"not certify itself pre-registers nothing, and a confirmatory run under it "
            f"would be uninterpretable. Restore the original file; do NOT re-hash the "
            f"altered one.")

    budget = int(body["selected_budget_steps"])
    folds = [int(f) for f in body["confirmatory_design"]["folds"]]
    rule = str(body["read_off_rule"])
    if int(fold) not in folds:
        raise SystemExit(
            f"[confirm] ABORT: fold {fold} is not in the pre-registered confirmatory set "
            f"{folds}.")
    if "FINAL checkpoint" not in rule:
        raise SystemExit(
            f"[confirm] ABORT: the pre-registered read-off rule is not the final-checkpoint "
            f"rule this runner implements:\n  {rule}")
    return {"path": os.path.abspath(path), "sha256": recorded, "verified": True,
            "budget_steps": budget, "read_off_rule": rule,
            "derived_or_declared": body.get("derived_or_declared"),
            "confirmatory_folds": folds,
            "method": body.get("method")}


# ===========================================================================
# per-fold facts, resolved from the splits rather than assumed
# ===========================================================================

def fold_positive_count(splits_path, fold):
    """The fold's TEST positive count, read from splits_v6 -- 98 on folds 0-3, 97 on fold 4.

    Assumed-98 was wrong for fold 4 and it matters: the one-positive resolution and every
    gate tolerance are 1/n, and FOLD_GATE[4]'s 0.20618... is 20/97, not a 98ths value.
    """
    with open(splits_path) as fh:
        s = json.load(fh)
    for f in s["real_lenses_kfold"]["folds"]:
        if int(f["fold"]) == int(fold):
            return int(len(f["test_lens_rows"]))
    raise SystemExit(f"[confirm] ABORT: fold {fold} not found in {splits_path}")


def print_fold_header(fold, n_pos, gate, prereg, paths, budget):
    one_pos = 1.0 / n_pos
    print("\n" + "=" * 96)
    print(f"FOLD {fold} -- CONFIRMATORY   (coral_nb3, gamma={GAMMA:g}, {budget} steps)")
    print("=" * 96)
    print(f"  test positives (splits_v6)  : {n_pos}")
    print(f"  one-positive resolution     : {one_pos:.6f}   "
          f"(a RATE -- never reported as a count, PROJECT.md 9)")
    print(f"  step-0 gate (FOLD_GATE[{fold}])  : TPR@1e-3 {gate['tpr']:.6f}   "
          f"AUROC {gate['auroc']:.6f}   ||head||2 {gate['head_l2']:.6f}")
    print(f"  pre-registration            : {prereg['path']}")
    print(f"  pre-registration sha256     : {prereg['sha256']}  VERIFIED")
    print(f"  budget / read-off           : {prereg['budget_steps']} steps, FINAL "
          f"checkpoint (step_{prereg['budget_steps']}.pt) -- THE BUDGET IS THE SELECTOR")
    print(f"  budget provenance           : {prereg['derived_or_declared'].upper()} "
          f"(see the pre-registration for the grounds)")
    print(f"  LP->FT init                 : {_path_tail(paths['init_head_from'], 5)}")
    print("=" * 96, flush=True)


# ===========================================================================
# path resolution -- per fold
# ===========================================================================

def resolve_paths(args, root, fold):
    """Locate splits / data / cache / this fold's phase-1 head / the pre-registration.

    Searched, never guessed -- the Kaggle mount has appeared as /kaggle/input/<slug> and as
    /kaggle/input/datasets/<owner>/<slug>, and `data.zip` auto-extracts to `data/data`, so
    the pieces do not share one root. This is the pattern the existing Kaggle runners
    established (KAGGLE_CORAL_R4_SWEEP_CELLS.md Cell 1); it is reused, not reinvented.
    """
    roots = [r for r in (args.input_root, "/kaggle/input", root) if r]
    out = {}

    def pick(name, explicit, pattern, required=True):
        p = os.path.abspath(explicit) if explicit else None
        if p is None:
            hit = _find_under(roots, pattern)
            p = os.path.abspath(hit) if hit else None
        if required and (p is None or not os.path.exists(p)):
            raise SystemExit(f"[confirm] ABORT: could not resolve {name}.\n"
                             f"  searched {roots} for {pattern!r}\n"
                             f"  pass --{name.replace('_', '-')} explicitly.")
        out[name] = p
        return p

    splits = pick("splits", args.splits, os.path.join("splits", "splits.json"))
    out["data_root"] = (os.path.abspath(args.data_root) if args.data_root
                        else os.path.dirname(os.path.dirname(splits)))
    probe = os.path.join(args.cache_dir, "source_train.npz") if args.cache_dir else None
    if probe and os.path.isfile(probe):
        out["cache_dir"] = os.path.abspath(args.cache_dir)
    else:
        out["cache_dir"] = os.path.dirname(
            pick("_cache", None, os.path.join("feature_cache", "source_train.npz")))
    # THIS FOLD's round-1 CORAL head -- fold-specific, per 7.11
    pick("init_head_from", args.init_head_from,
         os.path.join("bakeoff_coral", f"fold{int(fold)}", "checkpoints", "step_8000.pt"))
    pick("agreement_ref", args.agreement_ref,
         os.path.join("bakeoff_baseline_A", "checkpoints", "step_8000.pt"), required=False)
    pick("prereg", args.prereg, PREREG_BASENAME)
    ref = args.dev_run_json or _find_under(roots, os.path.join(DEV_ARM_RUN_JSON, "run.json"))
    out["dev_run_json"] = os.path.abspath(ref) if ref else None
    ref_ids = args.reference_ids or _find_under(
        roots, os.path.join("coral_r4_logits", "ids.npz"))
    out["reference_ids"] = os.path.abspath(ref_ids) if ref_ids else None
    out.pop("_cache", None)
    return out


# ===========================================================================
# the config diff -- allowlist and pinned list are PARAMETERS, never globals
# ===========================================================================

CONFIRM_DIFF_KEYS = tuple(DEV.DIFF_KEYS) + ("epochs_planned",)


def expected_diffs_for(fold, budget, ref):
    """The ALLOWLIST, resolved for this fold: (reference value, required new value, why).

    Each entry pins the NEW value too, not merely "may differ". That is what catches fold 2
    being pointed at fold 3's phase-1 head -- a mismatch that would otherwise train happily
    and produce a plausible, wrong number.
    """
    spe = int(ref.get("steps_per_epoch") or 187)
    return {
        "fold": (0, int(fold), "THE variable under test; fold 0 is development"),
        "max_steps": (int(ref.get("max_steps", 8000)), int(budget),
                      "8000 was the development maximum; the confirmatory budget is the "
                      "pre-registered 2000 (runs/coral_r4_budget.json)"),
        "init_head_from_tail": (
            ref.get("init_head_from_tail"),
            f"runs/bakeoff_coral/fold{int(fold)}/checkpoints/step_8000.pt",
            "each fold LP->FT-inits from ITS OWN round-1 CORAL step-8000 head (7.11)"),
        "init_head_l2": (ref.get("init_head_l2"),
                         float(R4.FOLD_GATE[int(fold)]["head_l2"]),
                         "...and therefore from that fold's own head norm, pinned in "
                         "FOLD_GATE and hard-gated by build_arm at 1e-4"),
        "epochs_planned": (ref.get("epochs_planned"),
                           max(1, int(math.ceil(budget / max(spe, 1)))),
                           "derived from max_steps, which is itself allowlisted"),
    }


def config_diff(record, ref_path, expected, pinned=CONFIRM_DIFF_KEYS, verbose=True):
    """Field-by-field against the fold-0 development arm. Returns (rows, mismatches).

    Round 4 owns this rather than importing the development runner's, because that one
    closes over ITS OWN expected-difference dict. `runs/coral_round4_design.md` §5 records
    the near-miss that argument makes concrete: a check that silently compares the wrong key
    list is a check that compares nothing. Both lists are parameters here.

    A key absent from BOTH sides is a MISMATCH, not a match -- same rule, same reason.
    """
    ref = {}
    if ref_path and os.path.isfile(ref_path):
        with open(ref_path) as fh:
            ref = json.load(fh)
    rows, mismatches = [], []
    for k in pinned:
        rv = ref.get(k, "<absent>")
        cv = record.get(k, "<absent>")
        if k in expected:
            want_ref, want_new, why = expected[k]
            ok = _same(rv, want_ref) and _same(cv, want_new)
            status = "EXPECTED" if ok else "MISMATCH"
            note = why if ok else (f"ALLOWLISTED BUT WRONG: expected {want_ref!r} -> "
                                   f"{want_new!r}, got {rv!r} -> {cv!r}")
        elif rv == "<absent>" and cv == "<absent>":
            status, note = "MISMATCH", ("absent from BOTH configs -- a guard that compares "
                                        "nothing is not a guard")
        else:
            status = "MATCH" if _same(rv, cv) else "MISMATCH"
            note = "" if status == "MATCH" else "must be identical to the fold-0 arm"
        if status == "MISMATCH":
            mismatches.append(k)
        rows.append({"key": k, "reference": rv, "this_run": cv, "status": status,
                     "note": note})

    if verbose:
        print("\n" + "=" * 100)
        print("CONFIG DIFF -- this confirmatory fold vs the fold-0 DEVELOPMENT arm")
        print(f"  reference: {ref_path}")
        print("=" * 100)
        print(f"  {'key':<26} {'fold-0 arm':<32} {'this run':<26} status")
        print("  " + "-" * 96)
        for r in rows:
            mark = {"MATCH": " ", "EXPECTED": "*", "MISMATCH": "X"}[r["status"]]
            print(f"{mark} {r['key']:<26} {str(r['reference'])[:31]:<32} "
                  f"{str(r['this_run'])[:25]:<26} {r['status']}")
        print("  " + "-" * 96)
        print("  legend:  (blank) identical   * allowlisted difference   X MISMATCH")
        for r in rows:
            if r["status"] != "MATCH" and r["note"]:
                print(f"    {r['key']}: {r['note']}")
        print(f"  => {len(mismatches)} mismatch(es)"
              + (f": {mismatches}" if mismatches else ""))
        print("=" * 100, flush=True)
    return rows, mismatches


# ===========================================================================
# the step-0 gate, per fold
# ===========================================================================

def step0_gate(scorer, record, fold, n_pos, tol_abort=0.05, auroc_tol_abort=0.01):
    """Score the LP->FT head itself: a KNOWN NULL against round-1 CORAL's own fold value.

    WARN, not abort, at the one-positive tolerance. 7.7 measured a local-vs-T4 drift of
    0.0204 on this metric -- two positives -- so a one-positive abort would fire on the
    venue alone and cost a session. The deterministic half of the gate is untouched:
    build_arm ALREADY aborts if the phase-1 head's ||.||_2 differs from FOLD_GATE's pinned
    value by more than 1e-4, and that is what catches a wrong head or a wrong fold. (The
    fold-0 development run measured this drift at exactly 0.0 on all nine shared
    checkpoints, so the wide band is insurance, not an expectation.)
    """
    from metrics import auroc as auroc_fn, tpr_at_fpr
    exp = R4.FOLD_GATE[int(fold)]
    one_pos = 1.0 / int(n_pos)
    dst, secs, ps, ns = scorer.dump(0, name="true_init")
    if ps.size != int(n_pos):
        raise SystemExit(
            f"[confirm] ABORT: fold {fold} scored {ps.size} positives but splits_v6 says "
            f"{n_pos}. The eval pool is not this fold's pool.")
    y = np.concatenate([np.ones(ps.size, np.int64), np.zeros(ns.size, np.int64)])
    s = np.concatenate([ps, ns])
    tpr, au = float(tpr_at_fpr(y, s)), float(auroc_fn(y, s))
    dt, da = abs(tpr - exp["tpr"]), abs(au - exp["auroc"])
    print("\n" + "=" * 92)
    print(f"STEP-0 GATE, fold {fold} (known null: this IS the phase-1 model), {secs:.0f}s")
    print(f"  TPR@1e-3 : {tpr:.6f}   expected {exp['tpr']:.6f}   |diff| {dt:.6f}   "
          f"(one positive on THIS fold = {one_pos:.6f}; abort > {tol_abort})")
    print(f"  AUROC    : {au:.6f}   expected {exp['auroc']:.6f}   |diff| {da:.6f}   "
          f"(abort > {auroc_tol_abort})")
    print(f"  positives: {ps.size}   negatives: {ns.size}")
    print("=" * 92, flush=True)
    verdict = ("ABORT" if (dt > tol_abort or da > auroc_tol_abort)
               else ("WARN" if (dt > one_pos or da > R4.STEP0_AUROC_TOL) else "PASS"))
    record["step0_gate"] = {"fold": int(fold), "tpr": tpr, "auroc": au,
                            "expected_tpr": exp["tpr"], "expected_auroc": exp["auroc"],
                            "tpr_abs_diff": dt, "auroc_abs_diff": da,
                            "one_positive_resolution": one_pos,
                            "n_pos": int(ps.size), "n_neg": int(ns.size),
                            "verdict": verdict}
    if verdict == "ABORT":
        raise SystemExit(
            f"[confirm] ABORT at the step-0 gate on fold {fold}: TPR@1e-3 {tpr:.6f} against "
            f"the pinned {exp['tpr']:.6f}, AUROC {au:.6f} against {exp['auroc']:.6f}.\n"
            f"  Step 0 IS the phase-1 model, so this is plumbing, not a result: check the "
            f"mount, the fold, the standardizer source and the eval precision.")
    if verdict == "WARN":
        banner(f"STEP-0 GATE fold {fold}: outside one positive, inside the abort band",
               [f"TPR {tpr:.6f} vs {exp['tpr']:.6f} (|diff| {dt:.6f}, one positive "
                f"{one_pos:.6f})",
                "7.7's measured local-vs-T4 drift is 0.0204, so this is consistent with the "
                "venue. If it is much larger, stop and check the scoring path."])
    else:
        print(f"[gate] fold {fold} step-0 known null reproduces round-1 CORAL. Proceeding.\n",
              flush=True)
    return record["step0_gate"]


def check_shared_negatives(ids_npz, ref_ids_npz, fold):
    """The NEGATIVE pool is shared across folds; the positives are not.

    So fold k's neg_ids must equal the reference run's neg_ids exactly, while its pos_ids
    must NOT (they are a different fold's held-out lenses). Both directions are checked --
    a positives match would mean the fold argument did not take effect.
    """
    out = {"reference": ref_ids_npz, "checked": False, "fold": int(fold)}
    if not ref_ids_npz or not os.path.isfile(ref_ids_npz):
        print(f"[ids] no reference ids.npz in this session -- shared-negative check SKIPPED",
              flush=True)
        return out
    a = np.load(ids_npz, allow_pickle=True)
    b = np.load(ref_ids_npz, allow_pickle=True)
    neg_ok = bool(np.array_equal(a["neg_ids"], b["neg_ids"]))
    pos_same = bool(a["pos_ids"].shape == b["pos_ids"].shape
                    and np.array_equal(a["pos_ids"], b["pos_ids"]))
    out.update({"checked": True, "neg_ids_match_reference": neg_ok,
                "pos_ids_identical_to_reference": pos_same,
                "n_pos": int(a["pos_ids"].size), "n_neg": int(a["neg_ids"].size)})
    print(f"[ids] fold {fold}: {out['n_pos']} pos / {out['n_neg']} neg | shared negative "
          f"pool matches the reference: {neg_ok} | positives differ from fold 0's: "
          f"{not pos_same}", flush=True)
    if not neg_ok:
        banner(f"SHARED NEGATIVE POOL DIFFERS ON FOLD {fold}",
               ["fold k's negatives are supposed to be the SAME ~26k shared pool as every",
                "other fold's. They are not. The run continues and the ids ARE saved, but",
                "any cross-fold statement about the negatives is invalid until this is",
                "explained."])
    if pos_same:
        banner(f"FOLD {fold}'s POSITIVES ARE IDENTICAL TO THE REFERENCE FOLD'S",
               ["Each fold holds out a DIFFERENT set of real lenses, so this should be",
                "impossible. The most likely cause is that --fold did not take effect and",
                "this process is training the reference fold again. STOP and check."])
    return out


# ===========================================================================
# main
# ===========================================================================

def _confirmatory_fold(v):
    """argparse type: refuse fold 0 AT PARSE TIME, with the reason."""
    try:
        k = int(v)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--fold must be an integer, got {v!r}")
    if k == 0:
        raise argparse.ArgumentTypeError(
            "FOLD 0 IS DEVELOPMENT AND CANNOT BE RUN AS A CONFIRMATORY FOLD. The 2000-step "
            "budget was chosen by inspecting fold 0's own TPR curve (PROJECT.md 7.11/7.14), "
            "so including fold 0 in the confirmatory set would be oracle selection and "
            "would contaminate every downstream number. The confirmatory folds are 1, 2, 3, "
            "4. Fold 0's result is reported separately and labelled development.")
    if k not in CONFIRMATORY_FOLDS:
        raise argparse.ArgumentTypeError(
            f"--fold must be one of {list(CONFIRMATORY_FOLDS)}, got {k}")
    return k


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=_confirmatory_fold, required=True,
                   help="1, 2, 3 or 4. Fold 0 is DEVELOPMENT and is refused here.")
    p.add_argument("--out-dir", default=None,
                   help="parent directory; this fold writes to <out-dir>/fold{k}")
    p.add_argument("--input-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--init-head-from", default=None)
    p.add_argument("--agreement-ref", default=None)
    p.add_argument("--prereg", default=None,
                   help="runs/coral_r4_budget.json -- hashed; the run refuses if it fails")
    p.add_argument("--dev-run-json", default=None,
                   help="the fold-0 development arm's run.json (the config-diff reference)")
    p.add_argument("--reference-ids", default=None)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--max-steps", type=int, default=None,
                   help="NOT for the confirmatory budget -- that comes from the verified "
                        "pre-registration. Smoke use only, and it forces a _SMOKE suffix.")
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--feat-norm-ceiling", type=float, default=DEFAULT_FEAT_NORM_CEILING)
    p.add_argument("--feat-norm-floor", type=float, default=DEFAULT_FEAT_NORM_FLOOR)
    p.add_argument("--coral-target-ema", dest="coral_target_ema", action="store_true",
                   default=False)
    p.add_argument("--device", default=None)
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--eval-fp16", action="store_true", default=False)
    p.add_argument("--allow-config-drift", action="store_true", default=False,
                   help="start despite a config mismatch (recorded in run.json)")
    p.add_argument("--no-resume", action="store_true", default=False)
    p.add_argument("--config-diff-only", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--keep-dry-run", action="store_true", default=False)
    return p


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    args = build_argparser().parse_args(argv)
    fold = int(args.fold)

    if args.dry_run:
        return dry_run(args, root, fold)

    paths = resolve_paths(args, root, fold)
    prereg = verify_prereg(paths["prereg"], fold)
    budget = int(args.max_steps) if args.max_steps else int(prereg["budget_steps"])
    smoke = bool(args.max_steps) and int(args.max_steps) != int(prereg["budget_steps"])
    n_pos = fold_positive_count(paths["splits"], fold)
    gate = R4.FOLD_GATE[fold]

    parent = args.out_dir or os.path.join(
        "/kaggle/working" if os.path.isdir("/kaggle/working") else root,
        "runs", "coral_r4_confirm")
    P = Paths(os.path.join(parent, f"fold{fold}" + ("_SMOKE" if smoke else "")))

    print("=" * 96)
    print(f"ROUND-4 CORAL CONFIRMATORY -- fold {fold} of {list(CONFIRMATORY_FOLDS)}"
          + ("   [SMOKE -- NOT the pre-registered budget]" if smoke else ""))
    print("=" * 96)
    for k in ("splits", "data_root", "cache_dir", "init_head_from", "prereg",
              "dev_run_json", "reference_ids"):
        print(f"  {k:<16}: {paths.get(k)}")
    print(f"  {'out_dir':<16}: {P.out}")
    print(f"  {'torch':<16}: {torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f"  devices={torch.cuda.device_count()}" if torch.cuda.is_available() else ""))
    print_fold_header(fold, n_pos, gate, prereg, paths, budget)

    existing = os.path.isfile(P.steps_csv) or os.path.isfile(P.resume_pt)
    if existing and args.no_resume:
        raise SystemExit(f"[confirm] ABORT: {P.out} already holds a run and --no-resume was "
                         f"passed. Move it aside or point --out-dir elsewhere.")

    # ---- build the arm through the SAME two lines the development run used ----
    sa = argparse.Namespace(
        arm=ARM_KEY, fold=fold, gamma=float(args.gamma),
        data_root=paths["data_root"], splits=paths["splits"], cache_dir=paths["cache_dir"],
        agreement_ref=(paths.get("agreement_ref") or R4.DEFAULT_AGREEMENT_REF),
        init_head_from=paths["init_head_from"], max_steps=budget, allow_off_budget=True,
        curve_every=int(args.log_every), ckpt_every=int(args.ckpt_every),
        backbone_lr=R4.BACKBONE_LR, fp16=bool(args.fp16), eval_fp16=bool(args.eval_fp16),
        grad_checkpointing=True, coral_target_ema=bool(args.coral_target_ema),
        num_workers=int(args.num_workers), device=args.device, target_fpr=1e-3,
        step0_tol=0.0, smoke_batch_size=None, dir_suffix=None, dry_run=False,
        emit_config=None, reference_config=None, calibrate=False, calibrate_steps=0,
        calibrate_tail=0, grad_check=False,
        feat_norm_ceiling=float(args.feat_norm_ceiling),
        feat_norm_floor=float(args.feat_norm_floor))
    A = R4.build_arm(ARM_KEY, sa, root, for_calibration=True)
    trainer, coral = A["trainer"], A["coral"]
    trainer.cfg.coral_weight = float(args.gamma)
    assert float(trainer.cfg.coral_weight) == float(args.gamma)
    assert coral.target_ema == bool(args.coral_target_ema)

    record = build_record(A, argparse.Namespace(**{**vars(args), "max_steps": budget}),
                          paths)
    record.update({
        "run": f"coral_r4_confirm_fold{fold}",
        "role": "CONFIRMATORY",
        "purpose": ("PROJECT.md 7.14 confirmatory fold. The budget and the read-off point "
                    "were fixed and hashed BEFORE this fold existed. Fold 0 is development "
                    "and is excluded from every headline number."),
        "pre_registration": prereg,
        "reported_checkpoint": f"step_{budget}.pt",
        "reported_checkpoint_rule": (
            "R1 -- the FINAL checkpoint. THE BUDGET IS THE SELECTOR. The intermediate "
            "250-step checkpoints are LAB RECORD AND DIAGNOSTICS ONLY and must not be read "
            "as results: choosing among them by any metric would be the oracle selection "
            "the pre-registration exists to prevent."),
        "checkpoint_selection_code_paths": "NONE -- no argmax, no smoothing, no selector",
        "fold_facts": {"fold": fold, "n_test_positives": n_pos,
                       "one_positive_resolution": 1.0 / n_pos,
                       "source": "splits_v6 real_lenses_kfold[fold].test_lens_rows",
                       "note": ("folds 0-3 hold 98, fold 4 holds 97 -- 489 lenses over 5 "
                                "folds, each tested exactly once")},
        "metric_hygiene": ("the only test-set metric computed is the step-0 known-null "
                           "gate; every training segment runs inside test_label_lockout(); "
                           "checkpoints are SCORED (logits persisted) but never MEASURED"),
        "single_gpu": ("no DataParallel / DDP / device sharding: halving the per-device "
                       "batch would change CORAL's covariance estimate from 256 draws to "
                       "128 (7.1a)"),
        "paths": paths, "out_dir": P.out,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "script_sha256_16": {f: _sha256(os.path.join(_HERE, f))
                             for f in ("run_coral_r4_confirm.py",) + DEV.SCRIPT_HASH_FILES
                             if os.path.isfile(os.path.join(_HERE, f))},
        "is_smoke": smoke,
        "steps_completed": 0, "aborted": False, "abort_reason": None,
        "fp16_skipped_cumulative": 0.0, "resume_events": [],
    })

    # ---- the config diff, before anything expensive ----
    with open(paths["dev_run_json"]) as fh:
        ref_cfg = json.load(fh)
    expected = expected_diffs_for(fold, budget, ref_cfg)
    rows, mismatches = config_diff(record, paths["dev_run_json"], expected, verbose=True)
    with open(P.diff_json, "w") as fh:
        json.dump({"reference": paths["dev_run_json"], "rows": rows,
                   "mismatches": mismatches,
                   "allowlist": {k: {"reference": v[0], "required": v[1], "why": v[2]}
                                 for k, v in expected.items()}}, fh, indent=1, default=str)
    record["config_diff"] = {"reference": paths["dev_run_json"], "mismatches": mismatches,
                             "allowed_drift": bool(args.allow_config_drift)}
    if mismatches and not args.allow_config_drift:
        raise SystemExit(
            f"[confirm] ABORT: fold {fold} differs from the fold-0 development arm on "
            f"{mismatches}, which is not on the allowlist "
            f"{sorted(expected)}.\n"
            f"  The confirmatory claim is that these folds differ from the development arm "
            f"ONLY in fold, budget and the per-fold LP->FT head. A mismatch means they do "
            f"not. See {P.diff_json}.")
    if args.config_diff_only:
        print("[confirm] --config-diff-only: diff emitted, nothing trained.")
        with open(P.run_json, "w") as fh:
            json.dump(record, fh, indent=1, default=str)
        return 0

    # ---- scorer, step-0 gate, ids ----
    from live_encode import LiveEncoder
    live = LiveEncoder(paths["splits"], paths["data_root"], fold,
                       batch_size=A["batch_size"], num_workers=0, device=A["device"],
                       verbose=True)
    scorer = Scorer(trainer, live, fold, P.logits, arm=logit_arm(fold, budget),
                    fp16=bool(args.eval_fp16))
    repro = ReproCheck(None, at_steps=(), window=1)     # no reference arm for a new fold

    resume_sched = None
    ids_path = os.path.join(P.logits, "ids.npz")
    if existing and not args.no_resume and os.path.isfile(P.resume_pt):
        resume_sched = load_resume(trainer, P)
        record["resume_events"].append(
            {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "resumed_at_step": int((resume_sched or {}).get("global_step", 0))})
        if os.path.isfile(ids_path):
            z = np.load(ids_path, allow_pickle=True)
            scorer.ids = (z["pos_ids"], z["neg_ids"])
        banner(f"RESUMED fold {fold}",
               [f"from {P.resume_pt} at step {(resume_sched or {}).get('global_step')}",
                "weights, AdamW moments, GradScaler, CORAL buffers and RNG restored (7.11's "
                "three restarts are all handled).",
                "Sample ORDER is exact; augmentation offsets after a resume are not "
                "guaranteed to match an uninterrupted run's. Recorded in run.json."], "=")
    else:
        step0_gate(scorer, record, fold, n_pos)
        record["ids_check"] = check_shared_negatives(ids_path, paths.get("reference_ids"),
                                                     fold)

    with open(P.run_json, "w") as fh:
        json.dump(record, fh, indent=1, default=str)

    # ---- train ----
    cfg = {"max_steps": budget, "log_every": int(args.log_every),
           "ckpt_every": int(args.ckpt_every), "gamma": float(args.gamma),
           "feat_norm_ceiling": float(args.feat_norm_ceiling),
           "feat_norm_floor": float(args.feat_norm_floor)}
    print(f"\n[train] fold {fold}: {budget} steps, checkpoint+logits every "
          f"{cfg['ckpt_every']}, trace every {cfg['log_every']}\n", flush=True)
    record = train_loop(trainer=trainer, coral=coral, A=A, P=P, cfg=cfg, record=record,
                        scorer=scorer, repro=repro, resume_sched=resume_sched)
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    final = os.path.join(P.ckpt, f"step_{budget}.pt")
    final_logits = os.path.join(P.logits, logit_arm(fold, budget), f"step_{budget}.npz")
    record["reported_checkpoint_present"] = bool(os.path.isfile(final))
    record["reported_logits_present"] = bool(os.path.isfile(final_logits))
    with open(P.run_json, "w") as fh:
        json.dump(record, fh, indent=1, default=str)

    banner(f"FOLD {fold} COMPLETE -- THE REPORTED RESULT IS step_{budget}",
           [f"reported checkpoint : checkpoints/step_{budget}.pt  "
            f"(present: {record['reported_checkpoint_present']})",
            f"reported logits     : "
            f"logits/{logit_arm(fold, budget)}/step_{budget}.npz  "
            f"(present: {record['reported_logits_present']})",
            "",
            "The other checkpoints in this directory are LAB RECORD AND DIAGNOSTICS ONLY.",
            "Reading a result off any of them would be the oracle selection the hashed",
            "pre-registration exists to prevent. THE BUDGET IS THE SELECTOR.",
            "",
            f"steps {record['steps_completed']}/{budget}   aborted={record['aborted']}   "
            f"{record['wall_minutes']:.1f} min"], "=")
    print(f"[out] {P.out}", flush=True)
    return 0


# ===========================================================================
# --dry-run
# ===========================================================================

def dry_run(args, root, fold):
    """Mock data on the real schema through the REAL loop, 3 steps, then exit.

    Covers: the pre-registration hash gate, the fold-0 refusal (at parse time, so it is
    covered by the suite rather than here), model + data + optimizer construction, three
    real CORAL steps, the logit dump and the resume round-trip. It does NOT cover build_arm's
    phase-1-head gate (no mock head can carry a pinned ||.||_2) or the real mount.
    """
    from lens_dataset import LensDataset, UNLABELED
    from live_encode import LiveEncoder
    from samplers import PairedDomainLoader, make_balanced_loader
    from test_lens_dataset import build_mock, build_splits
    from train_skeleton import TrainConfig, Trainer
    from zoobot_encoder import ZoobotEncoder

    dev = args.device or "cpu"
    mock_fold = 0 if fold not in (0, 1) else fold     # the mock splits only define folds 0/1
    tmp = tempfile.mkdtemp(prefix="coral_r4_confirm_dry_")
    out_dir = args.out_dir or os.path.join(tmp, "out")
    print("=" * 96)
    print(f"DRY RUN -- confirmatory fold {fold} plumbing on mock data, 3 steps, then exit")
    print(f"  device      : {dev}   (nothing trains on real data here)")
    print(f"  mock root   : {tmp}")
    print(f"  mock fold   : {mock_fold}  (the mock splits define folds 0 and 1 only; the "
          f"REAL fold {fold} is exercised by --config-diff-only against the mount)")
    print("=" * 96, flush=True)
    try:
        # ---- the pre-registration gate, on the real file ----
        prereg_path = args.prereg or _find_under([root, "/kaggle/input"], PREREG_BASENAME)
        prereg = verify_prereg(prereg_path, fold)
        print(f"[dry] pre-registration VERIFIED: {prereg['sha256']}")
        print(f"[dry]   budget {prereg['budget_steps']} steps, folds "
              f"{prereg['confirmatory_folds']}, {prereg['derived_or_declared']}")

        # ---- real per-fold facts from the real splits, if the mount is present ----
        real_splits = args.splits or _find_under(
            [root, "/kaggle/input"], os.path.join("splits", "splits.json"))
        if real_splits:
            counts = {k: fold_positive_count(real_splits, k) for k in (0,) + CONFIRMATORY_FOLDS}
            print(f"[dry] per-fold test positives from splits_v6: {counts}")
            print(f"[dry]   fold {fold} one-positive resolution = "
                  f"{1.0 / counts[fold]:.6f}")

        build_mock(tmp)
        splits = os.path.join(tmp, "splits.json")
        with open(splits, "w") as fh:
            json.dump(build_splits(), fh)

        src = LensDataset(splits, tmp, role="source_train", aug="weak", n_views=1)
        tgt = LensDataset(splits, tmp, role="target", fold=mock_fold, aug="weak", n_views=1)
        assert np.all(tgt.labels == UNLABELED), "mock target labels must be UNLABELED(-1)"
        bs, nsteps = 4, 3
        sl = make_balanced_loader(src, bs, use_latent=False, num_batches=nsteps,
                                  pos_fraction=0.5, seed=R4.SEED + mock_fold)
        tl = make_balanced_loader(tgt, bs, use_latent=True, num_batches=nsteps,
                                  pos_fraction=0.5, seed=R4.SEED + mock_fold + 1000)
        print(f"[dry] datasets: source {len(src)} rows, target {len(tgt)} rows; batch {bs}",
              flush=True)

        gc_on = (dev != "cpu")
        enc = ZoobotEncoder(encoder_mode="frozen", n_blocks=3, device=dev, verbose=False,
                            grad_checkpointing=gc_on)
        tcfg = TrainConfig(splits_path=splits, data_root=tmp, fold=mock_fold,
                           head_type=R4.HEAD_TYPE, encoder_mode="frozen", lr=R4.LR,
                           weight_decay=R4.WEIGHT_DECAY, batch_size=bs, pos_fraction=0.5,
                           use_coral=True, coral_weight=1.0,
                           coral_include_mean=R4.CORAL_INCLUDE_MEAN,
                           coral_shrinkage=R4.CORAL_SHRINKAGE, coral_target_ema=False,
                           coral_ema_momentum=R4.CORAL_EMA_MOMENTUM, total_steps=nsteps,
                           epochs=1, steps_per_epoch=nsteps, max_steps=nsteps,
                           seed=R4.SEED + mock_fold, device=dev, num_workers=0, amp=False,
                           n_blocks=3, backbone_lr=None, lr_decay=R4.LR_DECAY,
                           init_head_from=None, grad_checkpointing=gc_on)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = Trainer(tcfg, encoder=enc)
            coral = install_fp32_coral(trainer, include_mean=R4.CORAL_INCLUDE_MEAN,
                                       shrinkage=R4.CORAL_SHRINKAGE, target_ema=False,
                                       ema_momentum=R4.CORAL_EMA_MOMENTUM)
        trainer.standardizer.set_stats(torch.zeros(640), torch.ones(640))
        trainer.cfg.coral_weight = float(args.gamma)
        n_opt = sum(len(g["params"]) for g in trainer.opt.param_groups)
        print(f"[dry] optimizer: {type(trainer.opt).__name__}, "
              f"{len(trainer.opt.param_groups)} LR groups, {n_opt} param tensors; "
              f"gamma={trainer.cfg.coral_weight}", flush=True)

        P = Paths(out_dir)
        liveenc = LiveEncoder(splits, tmp, mock_fold, batch_size=bs, num_workers=0,
                              device=dev, verbose=False)
        scorer = Scorer(trainer, liveenc, mock_fold, P.logits,
                        arm=logit_arm(fold, prereg["budget_steps"]), fp16=False)
        dst, secs, ip, ineg = scorer.dump(0, name="true_init")
        print(f"[dry] step-0 true_init logits: {ip.size} pos / {ineg.size} neg", flush=True)

        A = {"paired": PairedDomainLoader(sl, tl), "src_loader": sl, "tgt_loader": tl,
             "steps": nsteps, "epochs_planned": 1}
        record = {"gamma": float(args.gamma), "dry_run": True}
        cfg = {"max_steps": nsteps, "log_every": 1, "ckpt_every": 1,
               "gamma": float(args.gamma), "feat_norm_ceiling": 0.0, "feat_norm_floor": 0.0}
        train_loop(trainer=trainer, coral=coral, A=A, P=P, cfg=cfg, record=record,
                   scorer=scorer, repro=ReproCheck(None, at_steps=(), window=1))

        # ---- resume round-trip: moments must come back, not just weights ----
        DEV.save_resume(trainer, P, gstep=nsteps, epoch=1, batch_in_epoch=nsteps,
                        rng_epoch_start=_rng_state(), record=record)
        m_before = DEV._moment_fingerprint(trainer)
        with contextlib.redirect_stdout(io.StringIO()):
            enc2 = ZoobotEncoder(encoder_mode="frozen", n_blocks=3, device=dev,
                                 verbose=False, grad_checkpointing=gc_on)
            t2 = Trainer(tcfg, encoder=enc2)
            install_fp32_coral(t2, include_mean=R4.CORAL_INCLUDE_MEAN,
                               shrinkage=R4.CORAL_SHRINKAGE, target_ema=False,
                               ema_momentum=R4.CORAL_EMA_MOMENTUM)
        sch = load_resume(t2, P)
        m_after = DEV._moment_fingerprint(t2)
        same_w = all(torch.equal(a.cpu(), b.cpu()) for a, b in
                     zip(trainer.head.state_dict().values(), t2.head.state_dict().values()))
        ea, eb = trainer.encoder_trainable_state(), t2.encoder_trainable_state()
        same_enc = set(ea) == set(eb) and all(torch.equal(ea[k].cpu(), eb[k].cpu())
                                              for k in ea)
        ok_m = (m_before is not None and m_after is not None
                and abs(m_before - m_after) <= 1e-12 * max(1.0, abs(m_before)))
        print("\n" + "-" * 96)
        print(f"[dry] resume: step={sch.get('global_step')} epoch={sch.get('epoch')} "
              f"batch_in_epoch={sch.get('batch_in_epoch')}")
        print(f"[dry]   head weights identical              : {same_w}")
        print(f"[dry]   {len(ea)} trainable encoder tensors identical: {same_enc}")
        print(f"[dry]   AdamW moment fingerprint restored    : {ok_m} "
              f"({m_before} -> {m_after})")
        print("-" * 96, flush=True)

        arm = logit_arm(fold, prereg["budget_steps"])
        z = np.load(os.path.join(P.logits, arm, f"step_{nsteps}.npz"))
        ids = np.load(os.path.join(P.logits, "ids.npz"), allow_pickle=True)
        init = np.load(os.path.join(P.logits, "true_init.npz"))
        wrote = sorted(os.path.relpath(p, P.out).replace("\\", "/")
                       for p in glob.glob(os.path.join(P.out, "**", "*"), recursive=True)
                       if os.path.isfile(p))
        print("[dry] files written:")
        for f in wrote:
            print(f"        {f}")
        print(f"[dry] logit npz {sorted(z.files)} pos {z['pos'].shape} neg {z['neg'].shape} "
              f"dtype {z['pos'].dtype}")
        print(f"[dry] ids.npz {sorted(ids.files)} pos_ids {ids['pos_ids'].shape} "
              f"{ids['pos_ids'].dtype}")
        ok = (set(z.files) == {"pos", "neg"} and set(ids.files) == {"pos_ids", "neg_ids"}
              and set(init.files) == {"pos", "neg"} and z["pos"].dtype == np.float64
              and same_w and same_enc and ok_m and record["steps_completed"] == nsteps
              and not record["aborted"] and os.path.isfile(P.ckpt_csv))
        print("\n" + ("[dry] DRY RUN PASSED -- pre-registration gate, model, data, "
                      "optimizer, loop, logit persistence and resume all exercised."
                      if ok else "[dry] DRY RUN FAILED -- see above."), flush=True)
        return 0 if ok else 1
    finally:
        if not args.keep_dry_run:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
