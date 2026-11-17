#!/usr/bin/env python
"""Rebuild the per-checkpoint TPR@1e-3 / AUROC curves for round-5 DANN, both arms.

WHY THIS EXISTS
---------------
Arms 1 and 2 both finished clean and both arrived with all 32 per-checkpoint logit files
missing: the results archiver enumerated each run's subdirectories one level deep and the
runner writes ``logits/<arm>_fold<k>/step_N.npz`` two levels down. Fixed on both sides
(recursive walk; packaging now refuses a run whose ``checkpoints.csv`` names an artifact
the archive lacks) -- but the two runs already home need their curves rebuilt from the
checkpoints, which DID come home.

``run.json`` records the discipline this restores: checkpoints are **SCORED (logits
persisted) but never MEASURED**, and the only test-set number computed during a run is the
step-0 known-null gate. That is why no curve exists as a side effect of training, and why
rebuilding one is a separate, explicit act.

THE FROZEN CACHE IS INVALID HERE
--------------------------------
``n_blocks > 0`` means the encoder moved, so the cached features are a different model's.
PROJECT.md 7.1a: scoring a real nb3 checkpoint from the stale cache once gave TPR 0.194
against a logged 0.020. Everything below goes through ``live_encode``.

IDENTITY WITH THE RUN PATH, NOT AN IMITATION OF IT
--------------------------------------------------
The arm is built by calling ``run_dann_r5.build_arm`` -- the same function the run used,
not a re-implementation -- and scored with the same ``Scorer`` class, the same
``LiveEncoder``, the same ``logit_arm`` string, and therefore the same on-disk layout. A
re-implementation that agreed to four decimal places would still be a second code path
whose disagreements nobody would ever see.

THE STEP-0 GATE IS A HARD GATE, AND IT IS STRONGER THAN THE BRIEF ASKED FOR
--------------------------------------------------------------------------
Both arms shipped ``logits/true_init.npz`` -- the natively produced step-0 logits, 98
positives and 26,399 negatives, bit-identical between the arms. So step 0 is not checked
by reproducing a scalar TPR (two different logit vectors can share a TPR; at 98 positives
the metric moves in steps of 0.0102 and is nearly designed to collide). It is checked
ELEMENT BY ELEMENT against 26,497 stored numbers. If that matches, the re-score path is
the run path; if it does not, nothing downstream is worth computing and the script stops.

The scalar fold-gate check runs too, because it is the check the RUN made and the one the
pre-registration names.

THE LOCKOUT BOUNDARY, STATED
----------------------------
Scoring runs inside ``test_label_lockout()``: producing logits must not be able to consult
a metric, and here that is enforced rather than asserted. Metrics are computed afterwards,
outside the lockout, from the persisted files -- which is the whole point of this script
and is recorded as such in ``rescore.json``. The boundary is: **logits inside, numbers
outside.**
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_dann_r5 as R5                                            # noqa: E402
from run_coral_r4_dev8000 import Scorer                             # noqa: E402
from coral_round4 import test_label_lockout                         # noqa: E402

TARGET_FPR = 1e-3

# element-wise tolerance for the step-0 identity check. fp32 accumulation over a live
# encode is not bit-deterministic across cuDNN algorithm choices, so this is a float
# tolerance, not equality -- but it is 6 orders of magnitude tighter than the gate's
# AUROC tolerance and it is checked on every one of the 26,497 logits, not on a summary.
STEP0_LOGIT_ATOL = 1e-4


# ---------------------------------------------------------------------------
# the runs this knows how to rebuild
# ---------------------------------------------------------------------------

ARM_RUNS = {
    "dann_nb3_lambda0": {
        "run_dir": "runs/dann_arm1/extracted/dann_r5_nb3_lambda0_fold0",
        "what": "arm 1 -- the lambda=0 control. Its curve is the UNFREEZING-ONLY "
                "reference and is not optional: without it, a TPR difference on arm 2 "
                "cannot be attributed to the adversary rather than to unfreezing.",
    },
    "dann_nb3": {
        "run_dir": "runs/dann_arm2_fold0/extracted/dann_r5_nb3_fold0",
        "what": "arm 2 -- the development run, lambda live.",
    },
}


def _steps_from_ckpt_dir(ckpt_dir):
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
        m = re.match(r"step_(\d+)\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def _metrics(pos, neg, target_fpr=TARGET_FPR):
    """TPR@fpr and AUROC. Imported HERE, not at module scope, so the names cannot be
    bound before ``test_label_lockout`` patches them -- the lockout patches function
    objects wherever they are bound, and a module-level import would create a binding
    that predates it."""
    import metrics as me
    y = np.concatenate([np.ones(pos.size, dtype=np.int64),
                        np.zeros(neg.size, dtype=np.int64)])
    s = np.concatenate([np.asarray(pos, dtype=np.float64),
                        np.asarray(neg, dtype=np.float64)])
    tpr = me.tpr_at_fpr(y, s, target_fpr=target_fpr)
    if isinstance(tpr, (tuple, list)):
        tpr = tpr[0]
    return float(tpr), float(me.auroc(y, s))


# ---------------------------------------------------------------------------
# the step-0 gate
# ---------------------------------------------------------------------------

def step0_hard_gate(scorer, run_dir, fold, *, tpr_tol, auroc_tol, verbose=True):
    """Score the FRESH arm and check it two ways. Raises SystemExit on any failure.

    Check A -- element-wise against the run's own ``true_init.npz``. This is the strong
    one, and it is what makes the other 63 checkpoints trustworthy: it says the encoder
    build, the live encode, the standardizer and the head all reproduce the run's own
    numbers, not merely a summary of them.

    Check B -- the scalar fold gate the RUN made and the pre-registration names.
    """
    ps, ns, _, _ = scorer._score()
    out = {"n_pos": int(ps.size), "n_neg": int(ns.size)}

    ref_p = os.path.join(run_dir, "logits", "true_init.npz")
    if not os.path.isfile(ref_p):
        raise SystemExit(
            f"[rescore] ABORT: {ref_p} is missing, so step 0 cannot be checked against "
            f"the run's own logits. Do not proceed -- an unverified re-score path is "
            f"exactly what produced TPR 0.194 against a logged 0.020 (7.1a).")
    ref = np.load(ref_p)
    rp, rn = np.asarray(ref["pos"], np.float64), np.asarray(ref["neg"], np.float64)
    if rp.shape != ps.shape or rn.shape != ns.shape:
        raise SystemExit(
            f"[rescore] ABORT: step-0 logit SHAPE differs from the run's: "
            f"pos {ps.shape} vs {rp.shape}, neg {ns.shape} vs {rn.shape}. The test pool "
            f"this is scoring is not the pool the run scored.")
    dp = float(np.abs(ps - rp).max())
    dn = float(np.abs(ns - rn).max())
    out.update({"max_abs_diff_pos": dp, "max_abs_diff_neg": dn,
                "n_logits_compared": int(ps.size + ns.size),
                "atol": STEP0_LOGIT_ATOL})
    ok_elem = max(dp, dn) <= STEP0_LOGIT_ATOL
    if verbose:
        print(f"  [A] element-wise vs true_init.npz: {ps.size + ns.size} logits, "
              f"max|diff| pos {dp:.3e} / neg {dn:.3e}  (atol {STEP0_LOGIT_ATOL:.0e})  "
              f"-> {'MATCH' if ok_elem else 'MISMATCH'}", flush=True)

    tpr, auroc = _metrics(ps, ns)
    g = R5.FOLD_GATE[int(fold)]
    dt, da = abs(tpr - g["tpr"]), abs(auroc - g["auroc"])
    ok_gate = dt <= tpr_tol and da <= auroc_tol
    out.update({"tpr": tpr, "auroc": auroc, "expected_tpr": g["tpr"],
                "expected_auroc": g["auroc"], "tpr_abs_diff": dt, "auroc_abs_diff": da,
                "tpr_tolerance": tpr_tol, "auroc_tolerance": auroc_tol,
                "elementwise_pass": bool(ok_elem), "gate_pass": bool(ok_gate)})
    if verbose:
        print(f"  [B] fold-{fold} gate: TPR {tpr:.6f} (expect {g['tpr']:.6f}, "
              f"|diff| {dt:.6f}, tol {tpr_tol:.6f})  AUROC {auroc:.6f} "
              f"(|diff| {da:.2e}, tol {auroc_tol:.0e})  -> "
              f"{'PASS' if ok_gate else 'FAIL'}", flush=True)

    out["verdict"] = "PASS" if (ok_elem and ok_gate) else "FAIL"
    if out["verdict"] != "PASS":
        raise SystemExit(
            "[rescore] ABORT: the step-0 gate FAILED.\n"
            f"  element-wise: {'pass' if ok_elem else 'FAIL'}   "
            f"scalar gate: {'pass' if ok_gate else 'FAIL'}\n"
            "  The re-score path does not reproduce the run path, so every number it "
            "would produce for the other 63 checkpoints is suspect. STOP. Do not widen "
            "a tolerance to get past this -- 7.7's rule is to fix by re-scoring on the "
            "venue, never by widening the gate. Report the printed values.")
    return out


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------

def rescore_arm(arm_key, args, *, verbose=True):
    spec = ARM_RUNS[arm_key]
    root = os.path.normpath(os.path.join(_HERE, ".."))
    run_dir = os.path.abspath(args.run_dir or os.path.join(root, spec["run_dir"]))
    if not os.path.isdir(run_dir):
        raise SystemExit(f"[rescore] ABORT: run dir not found: {run_dir}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    steps = _steps_from_ckpt_dir(ckpt_dir)

    R5.banner(f"RE-SCORE {arm_key} -- fold {args.fold}",
              [spec["what"],
               f"run dir   : {run_dir}",
               f"checkpoints: {len(steps)} found"
               + (f"  ({steps[0][0]} .. {steps[-1][0]})" if steps else ""),
               "the frozen feature cache is INVALID at n_blocks=3; this goes through "
               "live_encode (7.1a)."], "=")

    # retention: rebuilding a curve from a partial checkpoint set produces a curve that
    # looks complete and is not.
    rec_p = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(rec_p):
        rec = json.load(open(rec_p))
        exp = rec.get("n_checkpoints_expected")
        if exp is not None and len(steps) != int(exp):
            raise SystemExit(
                f"[rescore] ABORT: {len(steps)} checkpoints on disk but the run recorded "
                f"{exp}. Extract every part before re-scoring -- a curve rebuilt from a "
                f"partial set is not a partial curve, it is a wrong one.")

    # ---- build the arm through the RUN's OWN construction code ----
    args.arm = arm_key
    paths = R5.resolve_paths(args, root)
    A = R5.build_arm(arm_key, args, paths, verbose=verbose)
    cfg = A["cfg_json"]

    from live_encode import LiveEncoder
    live = LiveEncoder(paths["splits"], paths["data_root"], int(args.fold),
                       batch_size=A["batch_size"], num_workers=0, device=A["device"],
                       verbose=verbose)
    out_logits = os.path.join(run_dir, "logits")
    scorer = Scorer(A["trainer"], live, int(args.fold), out_logits,
                    arm=cfg["logit_arm"], fp16=bool(args.eval_fp16))
    ids_p = os.path.join(out_logits, "ids.npz")
    if os.path.isfile(ids_p):
        z = np.load(ids_p, allow_pickle=True)
        scorer.ids = (z["pos_ids"], z["neg_ids"])       # pin the run's own id ORDER

    # ---- THE HARD GATE, before any of the other checkpoints are touched ----
    print("\n[gate] step 0 -- the fresh arm, before any checkpoint is loaded", flush=True)
    t0 = time.time()
    gate = step0_hard_gate(scorer, run_dir, int(args.fold),
                           tpr_tol=cfg["step0_tpr_tolerance"],
                           auroc_tol=cfg["step0_auroc_tolerance"], verbose=verbose)
    gate["seconds"] = round(time.time() - t0, 1)
    print(f"[gate] PASS in {gate['seconds']:.0f}s -- the re-score path IS the run path. "
          f"Proceeding to {len(steps)} checkpoints.\n", flush=True)
    if args.gate_only:
        return {"arm": arm_key, "run_dir": run_dir, "step0_gate": gate,
                "gate_only": True}

    # ---- PHASE A: logits, with metrics LOCKED OUT ----
    per_step, t_start = [], time.time()
    with test_label_lockout(verbose=False):
        for i, (st, cp) in enumerate(steps, 1):
            A["trainer"].load(cp, load_training_state=False)
            dst, secs, _, _ = scorer.dump(st)
            per_step.append({"step": st, "logit_file":
                             os.path.relpath(dst, run_dir).replace("\\", "/"),
                             "score_seconds": round(secs, 1)})
            done = time.time() - t_start
            if verbose:
                print(f"  [{i:>2}/{len(steps)}] step {st:>5}  {secs:5.1f}s   "
                      f"({done/60:.1f} min elapsed, "
                      f"~{done/i*(len(steps)-i)/60:.0f} min left)", flush=True)
    phase_a_min = (time.time() - t_start) / 60.0

    # ---- PHASE B: the numbers, OUTSIDE the lockout, from the persisted files ----
    rows = []
    for r in per_step:
        z = np.load(os.path.join(run_dir, r["logit_file"]))
        tpr, auroc = _metrics(np.asarray(z["pos"]), np.asarray(z["neg"]))
        rows.append({"step": r["step"], "tpr_at_1e-3": round(tpr, 8),
                     "auroc": round(auroc, 8), "n_pos": int(z["pos"].size),
                     "n_neg": int(z["neg"].size),
                     "logit_file": r["logit_file"],
                     "score_seconds": r["score_seconds"]})

    curve_p = os.path.join(run_dir, "curve.csv")
    with open(curve_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    best = max(rows, key=lambda r: r["tpr_at_1e-3"])
    last = rows[-1]
    print(f"\n[curve] {curve_p}")
    print(f"  {len(rows)} steps, TPR@1e-3 "
          f"{min(r['tpr_at_1e-3'] for r in rows):.6f} .. "
          f"{best['tpr_at_1e-3']:.6f} (max at step {best['step']})")
    print(f"  last checkpoint (step {last['step']}): TPR {last['tpr_at_1e-3']:.6f}, "
          f"AUROC {last['auroc']:.6f}")
    print(f"  phase A (logits, lockout ON) {phase_a_min:.1f} min")

    return {"arm": arm_key, "run_dir": run_dir, "curve_csv": curve_p,
            "n_checkpoints": len(rows), "step0_gate": gate,
            "phase_a_minutes": round(phase_a_min, 2),
            "rows": rows,
            "lockout_boundary": (
                "PHASE A produced every logit file with test_label_lockout() ACTIVE, so "
                "no scoring decision could consult a metric -- enforced, not asserted. "
                "PHASE B computed TPR/AUROC afterwards, outside the lockout, from the "
                "persisted files. The boundary is: logits inside, numbers outside. This "
                "restores the run's own discipline (run.json metric_hygiene): "
                "checkpoints are SCORED but never MEASURED while anything can still be "
                "chosen on the basis of the score."),
            "reproduces_run_path": (
                "the arm was built by calling run_dann_r5.build_arm -- the same function "
                "the run used -- and scored with the same Scorer, LiveEncoder, "
                "logit_arm and on-disk layout. Verified at step 0 element-wise against "
                "the run's own true_init.npz over "
                f"{gate['n_logits_compared']} logits, max|diff| "
                f"{max(gate['max_abs_diff_pos'], gate['max_abs_diff_neg']):.3e}."),
            }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# the wiring check -- run this function's WHOLE control flow without a GPU
# ---------------------------------------------------------------------------

def wiring_check(args, verbose=True) -> int:
    """Execute ``rescore_arm`` end to end locally, with only the GPU work stubbed.

    WHY THIS EXISTS. Two bugs in this file reached the venue and died AFTER a 4.2 GB
    extract and a full arm build: a call to ``resolve_paths`` with one argument instead
    of two, and then an ``UnboundLocalError`` from the fix for the first one. Neither
    needed a GPU to find. Both survived because the local check stopped at
    ``parse_args`` -- everything past it wanted a GPU and 4.6 GB of data, so the body of
    the function was never once executed anywhere before it was executed on Kaggle.

    A signature test does not catch an unbound local. A syntax check does not catch
    either. The only thing that reliably catches "this line has never run" is RUNNING IT,
    so this stubs the four expensive things -- ``build_arm``, ``LiveEncoder``, ``Scorer``
    and the checkpoint loads -- and lets every other line execute for real: the path
    logic, the retention check, the banner, the argument passing, the dict keys, the
    lockout boundary, the CSV writing.

    The stub scorer returns each arm's REAL ``true_init.npz`` logits, so the step-0 gate
    genuinely passes rather than being skipped, and Phase B computes real metrics from
    real files. What is faked is the encoder, not the plumbing.
    """
    import shutil
    import tempfile
    import types

    root = os.path.normpath(os.path.join(_HERE, ".."))
    real_build_arm = R5.build_arm
    real_resolve = R5.resolve_paths
    g = globals()
    real_scorer = g["Scorer"]
    ok_arms = []

    class _StubTrainer:
        def __init__(self):
            self.loaded = []

        def load(self, path, load_training_state=True):
            self.loaded.append((os.path.basename(path), load_training_state))
            return {}

    class _StubScorer:
        """Same constructor and same two methods the real Scorer exposes."""

        def __init__(self, trainer, live, fold, logit_dir, arm=None, fp16=False):
            self.tr, self.fold, self.root = trainer, int(fold), logit_dir
            self.arm_dir = os.path.join(logit_dir, arm or "stub")
            os.makedirs(self.arm_dir, exist_ok=True)
            self.ids = None
            z = np.load(os.path.join(logit_dir, "true_init.npz"))
            self.p0 = np.asarray(z["pos"], np.float64)
            self.n0 = np.asarray(z["neg"], np.float64)
            zi = np.load(os.path.join(logit_dir, "ids.npz"), allow_pickle=True)
            self.pid, self.nid = zi["pos_ids"], zi["neg_ids"]
            self.n_scored = 0

        def _score(self):
            # step 0 returns the run's own logits EXACTLY, so the gate is a real check
            k = len(self.tr.loaded)
            if k == 0:
                return self.p0, self.n0, self.pid, self.nid
            rng = np.random.default_rng(k)
            return (self.p0 + rng.normal(0, 0.01, self.p0.shape),
                    self.n0 + rng.normal(0, 0.01, self.n0.shape), self.pid, self.nid)

        def dump(self, step, name=None):
            ps, ns, _, _ = self._score()
            fn = (name or f"step_{int(step)}") + ".npz"
            dst = (os.path.join(self.root, fn) if name
                   else os.path.join(self.arm_dir, fn))
            np.savez_compressed(dst, pos=ps, neg=ns)
            self.n_scored += 1
            return dst, 0.01, ps, ns

    tmp = tempfile.mkdtemp(prefix="rescore_wiring_")
    try:
        for arm_key, spec in ARM_RUNS.items():
            live_run = os.path.join(root, spec["run_dir"])
            need = ("run_config.json", "logits/ids.npz", "logits/true_init.npz")
            if not all(os.path.isfile(os.path.join(live_run, q)) for q in need):
                if verbose:
                    print(f"[wiring] {arm_key}: run files not in this checkout -- skipped")
                continue

            # a run dir with the full 32-checkpoint set, so the retention check RUNS
            fake = os.path.join(tmp, arm_key)
            os.makedirs(os.path.join(fake, "checkpoints"))
            for q in need:
                d = os.path.join(fake, q)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(os.path.join(live_run, q), d)
            exp = json.load(open(os.path.join(fake, "run_config.json")))[
                "n_checkpoints_expected"]
            for st in range(250, 250 * exp + 1, 250):
                open(os.path.join(fake, "checkpoints", f"step_{st}.pt"), "wb").write(b"")

            stub_paths = {"splits": "SPLITS", "data_root": "DATA",
                          "cache_dir": "CACHE", "init_head_from": "HEAD"}
            tr = _StubTrainer()

            def _fake_build_arm(a_key, a, pth, *, verbose=True, _tr=tr):
                cfgj = {"logit_arm": R5.LOGIT_ARM_FMT.format(arm=a_key,
                                                             fold=int(a.fold)),
                        "step0_tpr_tolerance": 1.0 / 98.0,
                        "step0_auroc_tolerance": 1e-3}
                return {"cfg_json": cfgj, "batch_size": 256, "device": "cpu",
                        "trainer": _tr}

            R5.build_arm = _fake_build_arm
            R5.resolve_paths = lambda a, r: dict(stub_paths)
            g["Scorer"] = _StubScorer
            sys.modules["live_encode"] = types.ModuleType("live_encode")
            sys.modules["live_encode"].LiveEncoder = (
                lambda *a, **k: types.SimpleNamespace())
            # --run-dir is left UNSET on purpose. Passing it short-circuits
            # `args.run_dir or os.path.join(root, spec["run_dir"])` -- the exact
            # expression that raised UnboundLocalError on the venue, and which an earlier
            # version of this check HID by supplying run_dir. So the DEFAULT path runs
            # instead: _HERE is repointed so `root` lands on the temp tree.
            real_here = g["_HERE"]
            real_spec_dir = ARM_RUNS[arm_key]["run_dir"]
            try:
                g["_HERE"] = os.path.join(tmp, "scripts")
                ARM_RUNS[arm_key]["run_dir"] = arm_key
                a2 = argparse.Namespace(**vars(args))
                a2.run_dir = None
                a2.gate_only = False
                res = rescore_arm(arm_key, a2, verbose=False)
            finally:
                g["_HERE"] = real_here
                ARM_RUNS[arm_key]["run_dir"] = real_spec_dir
                R5.build_arm = real_build_arm
                R5.resolve_paths = real_resolve
                g["Scorer"] = real_scorer
                sys.modules.pop("live_encode", None)

            rows = list(csv.DictReader(open(res["curve_csv"], newline="")))
            assert len(rows) == exp, f"{arm_key}: curve has {len(rows)} rows, want {exp}"
            assert res["step0_gate"]["verdict"] == "PASS", f"{arm_key}: gate did not pass"
            assert res["step0_gate"]["n_logits_compared"] == 26497, arm_key
            assert [int(r["step"]) for r in rows] == list(range(250, 250 * exp + 1, 250))
            assert all(0.0 <= float(r["tpr_at_1e-3"]) <= 1.0 for r in rows)
            assert tr.loaded and all(ls is False for _, ls in tr.loaded), (
                f"{arm_key}: checkpoints must load with load_training_state=False")
            ok_arms.append(arm_key)
            if verbose:
                print(f"[wiring] {arm_key}: OK -- gate PASS "
                      f"({res['step0_gate']['n_logits_compared']} logits), "
                      f"{len(rows)} curve rows, {len(tr.loaded)} checkpoints loaded")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if verbose:
        print(f"[wiring] {len(ok_arms)} arm(s) executed end to end with the GPU work "
              f"stubbed: {ok_arms}")
    return 0 if ok_arms else 1


def build_argparser():
    p = R5.build_argparser()
    p.description = "rebuild the per-checkpoint TPR/AUROC curves for round-5 DANN"
    p.add_argument("--rescore-arms", nargs="+", default=list(ARM_RUNS),
                   choices=sorted(ARM_RUNS),
                   help="which arms to rebuild (default: both -- arm 1 is the "
                        "unfreezing-only reference and is not optional for the "
                        "comparison the round exists to make)")
    p.add_argument("--run-dir", default=None,
                   help="override the run directory for a SINGLE arm")
    p.add_argument("--gate-only", action="store_true", default=False,
                   help="run the step-0 hard gate and stop. ~3-5 min per arm, and it is "
                        "the check that decides whether the other 63 are worth running")
    p.add_argument("--wiring-check", action="store_true", default=False,
                   help="execute rescore_arm's WHOLE control flow locally with the GPU "
                        "work stubbed, and exit. No GPU, no data, ~2 s. This is what "
                        "catches a line that has never run.")
    p.add_argument("--rescore-out", default=None,
                   help="where to write rescore.json (default: alongside the runs)")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.wiring_check:
        return wiring_check(args)
    if args.run_dir and len(args.rescore_arms) != 1:
        raise SystemExit("[rescore] --run-dir overrides ONE arm; name exactly one in "
                         "--rescore-arms")
    args.fold = int(args.fold)

    results, t0 = [], time.time()
    for arm in args.rescore_arms:
        results.append(rescore_arm(arm, args, verbose=True))

    out = {
        "purpose": ("rebuild the per-checkpoint TPR@1e-3 / AUROC curves the collection "
                    "defect dropped from both arms' archives"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_minutes": round((time.time() - t0) / 60.0, 2),
        "target_fpr": TARGET_FPR,
        "encoder_eval": "live_encode",
        "encoder_eval_note": ("the frozen cache is INVALID at n_blocks>0 (7.1a): scoring "
                              "a real nb3 checkpoint from the stale cache once gave TPR "
                              "0.194 against a logged 0.020 -- a different model, not a "
                              "rounding difference."),
        "arms": results,
    }
    dst = args.rescore_out or os.path.join(
        os.path.normpath(os.path.join(_HERE, "..")), "runs", "dann_r5_rescore.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[out] {dst}")

    if len(results) == 2 and all(r.get("curve_csv") for r in results):
        R5.banner("BOTH CURVES BUILT", [
            f"{r['arm']:<18} last-step TPR {r['rows'][-1]['tpr_at_1e-3']:.6f}  "
            f"AUROC {r['rows'][-1]['auroc']:.6f}" for r in results
        ] + ["the comparison these were built for is arm 2 against arm 1 at MATCHED "
             "steps; a single last-step pair is not it."], "=")
    return 0


if __name__ == "__main__":
    sys.exit(main())
