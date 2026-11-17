"""
test_coral_round4_changes.py
============================
T10 -- the two round-4 changes and the gamma-sweep machinery.

Split into its own module rather than appended to ``test_coral_round4.py`` so the original
suite's 133 checks stay exactly as they were and this can be run on its own. It reuses that
module's ``check`` / ``raises`` / ``section`` / ``_trainer`` / ``_args`` helpers, and
``test_coral_round4.py`` calls it as its final group, so the full suite is still one command.

  CHANGE 1  target EMA OFF by default on both arms, PINNED, still re-enablable.
  CHANGE 2  a hard feature-norm abort at a configurable ceiling (default 110).
  SWEEP     the pre-registered selection criterion's arithmetic, the sweep runner's
            metric-hygiene and collision guards, and the Kaggle launcher's device plan
            and verification gate.

Run:  python scripts/test_coral_round4_changes.py
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_coral_round4 as T
from test_coral_round4 import check, raises, section, _trainer, _args
from coral_round4 import (DEFAULT_FEAT_NORM_CEILING, FeatureNormCeiling,
                          check_feature_norm_ceiling, SWEEP_CRITERION, SWEEP_GAMMAS,
                          slope_with_block_ci, norm_growth_with_ci, damping_verdict,
                          install_fp32_coral, shift_invariance_check)


def t10_change1_ema_off():
    section("T10a  CHANGE 1 -- target EMA OFF by default on both arms, and PINNED")
    import run_coral_round4 as R4
    root = os.path.normpath(os.path.join(_HERE, ".."))

    check(R4.CORAL_TARGET_EMA is False,
          "CORAL_TARGET_EMA default is now False")
    check("coral_target_ema" in R4.PINNED_KEYS,
          "coral_target_ema is PINNED -- a mismatch between arms fails the check")

    c0 = R4.build_config("coral_nb0_control", _args(arm="coral_nb0_control", gamma=0.5),
                         root)[0]
    c3 = R4.build_config("coral_nb3", _args(arm="coral_nb3", gamma=0.5), root)[0]
    check(c0["coral_target_ema"] is False and c3["coral_target_ema"] is False,
          "BOTH arms' configs carry coral_target_ema=False by default")

    d = tempfile.mkdtemp(prefix="coral_r4_ema_")
    try:
        ref = os.path.join(d, "ref.json")
        with open(ref, "w") as fh:
            json.dump(c0, fh)
        bad = dict(c3)
        bad["coral_target_ema"] = True
        raises(SystemExit, lambda: R4.check_preconditions(bad, ref, verbose=False),
               "pinning REFUSES an arm pair whose target-EMA setting differs")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    on = R4.build_config("coral_nb3", _args(arm="coral_nb3", gamma=0.5,
                                            coral_target_ema=True), root)[0]
    check(on["coral_target_ema"] is True,
          "--coral-target-ema still re-enables it when explicitly asked")

    tr = _trainer(0, use_coral=True, gamma=1.0)
    c = install_fp32_coral(tr, target_ema=R4.CORAL_TARGET_EMA)
    check(c.target_ema is False, "the installed CORAL module has target_ema=False")

    s = torch.randn(32, 640, requires_grad=True)
    t = torch.randn(32, 640, requires_grad=True) * 1.3 + 0.5
    c.train()
    v1, v2 = float(c(s, t)), float(c(s, t))
    check(v1 == v2,
          "with the EMA off the loss is a PURE FUNCTION of the batch (identical inputs -> "
          "identical value; no hidden state, no stale reference)")
    check(float(c.ema_ready) == 0.0,
          "no EMA buffer is ever written when target_ema=False")

    si = shift_invariance_check(tr, torch.rand(4, 1, 224, 224), torch.rand(4, 1, 224, 224))
    check(si["bias_rel_change"] < 1e-4,
          f"translation invariance is RESTORED with the EMA off ({si['bias_rel_change']:.1e} "
          f"relative response to a +1.0 shift, vs 6.0e+03 measured with a stale EMA)")

    # The known limitation this change accepts, asserted so it stays on the record.
    n_samples, dim = 256, 640
    check(n_samples - 1 < dim,
          f"KNOWN LIMITATION, accepted not fixed: a {n_samples}-sample covariance in {dim} "
          f"dims has rank <= {n_samples - 1} < {dim}, i.e. SINGULAR. Deep CORAL's validated "
          f"regime had d << n. The deficiency is SYMMETRIC across domains (both estimated "
          f"from the same 256 draws), so it is a noise cost, not a bias.")
    lo = Fp32 = None  # noqa: F841  (keep the namespace tidy)
    del tr


def t10_change2_norm_ceiling():
    section("T10b  CHANGE 2 -- the hard feature-norm abort")
    import run_coral_round4 as R4
    root = os.path.normpath(os.path.join(_HERE, ".."))

    check(DEFAULT_FEAT_NORM_CEILING == 110.0, "the default ceiling is 110")
    check_feature_norm_ceiling(123, {"src_feat_l2": 82.0, "tgt_feat_l2": 81.0}, 110.0, 0.05)
    check(True, "no abort below the ceiling (82/81 against 110 -- the observed trajectory)")

    for dom, bad in (("source", {"src_feat_l2": 111.0, "tgt_feat_l2": 80.0}),
                     ("target", {"src_feat_l2": 80.0, "tgt_feat_l2": 126.4})):
        raises(FeatureNormCeiling,
               lambda b=bad: check_feature_norm_ceiling(777, b, 110.0, 0.123),
               f"ABORTS when the {dom} domain exceeds the ceiling")

    msg = ""
    try:
        check_feature_norm_ceiling(777, {"src_feat_l2": 130.0, "tgt_feat_l2": 129.0},
                                   110.0, 0.4567)
    except FeatureNormCeiling as e:
        msg = str(e)
    for token in ("step 777", "130.0", "129.0", "0.4567", "110"):
        check(token in msg,
              f"the abort message carries {token!r} (step, BOTH norms, raw CORAL, ceiling)")

    check_feature_norm_ceiling(1, {"src_feat_l2": 1e9, "tgt_feat_l2": 1e9}, 0.0, 0.0)
    check(True, "ceiling=0 disables the guard -- an explicit opt-out, never silent")

    check("feat_norm_ceiling" in R4.PINNED_KEYS,
          "feat_norm_ceiling is PINNED so the arms cannot run different ceilings")
    c0 = R4.build_config("coral_nb0_control", _args(arm="coral_nb0_control", gamma=0.5),
                         root)[0]
    check(c0["feat_norm_ceiling"] == 110.0, "the config carries the ceiling")
    src = open(os.path.join(_HERE, "run_coral_round4.py"), encoding="utf-8").read()
    check("check_feature_norm_ceiling(global_step" in src,
          "the main runner checks the ceiling EVERY step, not once per curve point")


def t10_sweep_criterion():
    section("T10c  the PRE-REGISTERED sweep criterion and its arithmetic")
    check(tuple(SWEEP_GAMMAS) == (1.0, 4.0, 16.0), "the swept gammas are 1, 4, 16")
    for token in ("SMALLEST gamma", "MOVING-BLOCK BOOTSTRAP", "LOWER BOUND <= 0",
                  "ENTIRELY BELOW zero", "INAPPLICABLE", "EXTENDING THE SWEEP UPWARD",
                  "LOSS DYNAMICS and FEATURE STATISTICS ONLY"):
        check(token in SWEEP_CRITERION, f"the criterion text pins {token!r}")

    rng = np.random.default_rng(0)
    steps = np.arange(1000, 2001)

    rising = 0.05 + 1e-5 * (steps - 1000) + rng.normal(0, 2e-4, steps.size)
    r = slope_with_block_ci(steps, rising, B=400, seed=1)
    check(r["not_rising"] is False and r["strictly_rising"] is True,
          f"clause (i) FAILS a demonstrably rising CORAL series (slope {r['slope']:+.2e})")

    falling = 0.05 - 1e-5 * (steps - 1000) + rng.normal(0, 2e-4, steps.size)
    r = slope_with_block_ci(steps, falling, B=400, seed=1)
    check(r["not_rising"] is True and r["strictly_falling"] is True,
          f"clause (i) PASSES a falling series (slope {r['slope']:+.2e})")

    flat = 0.05 + rng.normal(0, 2e-4, steps.size)
    r = slope_with_block_ci(steps, flat, B=400, seed=1)
    check(r["not_rising"] is True and r["strictly_falling"] is False,
          "clause (i) PASSES a FLAT series and does not call it strictly falling -- "
          "'not rising' is the test, not 'falling'")

    ar = np.zeros(steps.size)
    ar[0] = 0.05
    for i in range(1, steps.size):
        ar[i] = 0.999 * ar[i - 1] + rng.normal(0, 1e-4)
    wide = slope_with_block_ci(steps, ar, block=50, B=400, seed=2)
    narrow = slope_with_block_ci(steps, ar, block=1, B=400, seed=2)
    w = wide["ci"][1] - wide["ci"][0]
    n = narrow["ci"][1] - narrow["ci"][0]
    check(w > n,
          f"the MOVING-BLOCK bootstrap gives a wider CI than an iid one on an "
          f"autocorrelated trace ({w:.2e} vs {n:.2e}) -- an iid bootstrap would understate "
          f"the slope's uncertainty, which is why block=50")

    ref = norm_growth_with_ci(steps, np.linspace(66, 90, steps.size), B=400, seed=3)
    damp = norm_growth_with_ci(steps, np.linspace(66, 70, steps.size), B=400, seed=4)
    same = norm_growth_with_ci(steps, np.linspace(66, 90, steps.size), B=400, seed=5)
    v = damping_verdict(damp, ref)
    check(v["passed"] is True,
          f"clause (ii) PASSES a measurably damped arm (diff {v['diff']:+.3f})")
    v = damping_verdict(same, ref)
    check(v["passed"] is False,
          "clause (ii) FAILS an arm that grows as much as the reference")
    flat_ref = norm_growth_with_ci(steps, np.full(steps.size, 66.0), B=200, seed=6)
    v = damping_verdict(damp, flat_ref)
    check(v["applicable"] is False and v["passed"] is False,
          "clause (ii) is INAPPLICABLE (not passed) when the reference arm did not grow -- "
          "there is nothing to damp")


def t10_sweep_runner_and_launcher():
    section("T10d  sweep runner hygiene + collision guard; launcher device plan + gate")
    import run_coral_gamma_sweep as SW
    import launch_coral_sweep_kaggle as L

    src = open(SW.__file__, encoding="utf-8").read()
    check("with test_label_lockout():" in src,
          "the sweep runner wraps its ENTIRE run in test_label_lockout()")
    check("curve_hook.append_curve_row" not in src,
          "the sweep runner never calls the curve hook (it would compute TPR)")
    check("dump_logits" not in src and "memorization_row" not in src,
          "no logit dump and no memorization probe in the sweep path (both read test splits)")
    check("check_feature_norm_ceiling" in src,
          "the sweep runner enforces the feature-norm ceiling")
    check("os.fsync" in src,
          "trace rows are fsync'd -- a Kaggle session kill must not lose the last rows")

    d = tempfile.mkdtemp(prefix="coral_r4_sweep_")
    try:
        with open(os.path.join(d, "trace.csv"), "w") as fh:
            fh.write("step\n1\n")
        raises(SystemExit,
               lambda: SW.main(["--gamma", "1", "--out-dir", d, "--dry-run"]),
               "the sweep REFUSES to start where a trace.csv already exists (the exact "
               "concurrent-write failure that corrupted the calibration)",
               match="already exists")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    check(L.CONCURRENT == ((1.0, 0), (4.0, 1)),
          "launcher pins gamma=1 -> cuda:0 and gamma=4 -> cuda:1, launched concurrently")
    check(L.QUEUED == (16.0,), "gamma=16 is queued for whichever device frees first")
    lsrc = open(L.__file__, encoding="utf-8").read()
    check('"-u"' in lsrc, "the launcher passes python -u")
    check("stderr=subprocess.STDOUT" in lsrc and "subprocess.PIPE" not in lsrc,
          "the launcher redirects stdout+stderr to a FILE and never uses a pipe")
    check("CUDA_VISIBLE_DEVICES" in lsrc,
          "the launcher pins each arm to a device via CUDA_VISIBLE_DEVICES")
    check('j["proc"].poll()' in lsrc or "proc.poll()" in lsrc,
          "liveness is Popen.poll() -- not `ps -p`, which cannot see native Windows PIDs")

    d = tempfile.mkdtemp(prefix="coral_r4_verify_")
    try:
        check(L.verify_trace(d)["ok"] is False, "verify_trace FAILS on a missing trace.csv")
        with open(os.path.join(d, "trace.csv"), "w") as fh:
            fh.write("step,nonsense\n1,2\n")
        check(L.verify_trace(d)["ok"] is False,
              "verify_trace FAILS on a trace with the wrong columns")
        with open(os.path.join(d, "trace.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(L.REQUIRED_TRACE_COLS))
            w.writeheader()
            w.writerow({c: "nan" for c in L.REQUIRED_TRACE_COLS})
        check(L.verify_trace(d)["ok"] is False,
              "verify_trace FAILS when the last row is non-finite")
        with open(os.path.join(d, "trace.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(L.REQUIRED_TRACE_COLS))
            w.writeheader()
            w.writerow({c: "1.0" for c in L.REQUIRED_TRACE_COLS})
        check(L.verify_trace(d)["ok"] is True, "verify_trace PASSES a well-formed trace")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t10_analyzer_end_to_end():
    section("T10e  the local analyzer, end to end on synthetic sweep directories")
    import analyze_coral_gamma_sweep as AN
    root = tempfile.mkdtemp(prefix="coral_r4_an_")
    try:
        steps = np.arange(1, 2001)
        # g1 grows and its CORAL rises; g4 is damped and flat; g16 damped and falling
        spec = {1: (np.linspace(66, 90, 2000), 0.02 + 1e-5 * steps),
                4: (np.linspace(66, 72, 2000), np.full(2000, 0.05)),
                16: (np.linspace(66, 67, 2000), 0.09 - 1e-5 * steps)}
        rng = np.random.default_rng(7)
        for g, (norm, coral) in spec.items():
            d = os.path.join(root, f"g{g}")
            os.makedirs(d)
            with open(os.path.join(d, "steps.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["step", "sup_loss", "coral_raw", "src_feat_l2",
                            "tgt_feat_l2", "src_perdim_std", "tgt_perdim_std"])
                for i, s in enumerate(steps):
                    w.writerow([s, 0.05, coral[i] + rng.normal(0, 1e-5),
                                norm[i] + rng.normal(0, 0.01),
                                norm[i] + rng.normal(0, 0.01), 0.7, 0.7])
            json.dump({"gamma": float(g), "steps_completed": 2000, "aborted": False},
                      open(os.path.join(d, "sweep.json"), "w"))
        out = os.path.join(root, "verdict.json")
        rc = AN.main(["--sweep-dir", root, "--bootstrap", "300", "--out", out])
        check(rc == 0, "the analyzer runs end to end on three synthetic arms")
        v = json.load(open(out))
        check(v["selected_gamma"] == 4.0,
              f"it selects the SMALLEST gamma passing both clauses (got "
              f"{v['selected_gamma']})")
        check(v["arms"]["1.0"]["criterion_i"]["not_rising"] is False,
              "gamma=1's rising CORAL fails clause (i), as constructed")
        check(v["arms"]["4.0"]["criterion_ii"]["passed"] is True,
              "gamma=4's damped norm passes clause (ii), as constructed")

        # and when NOTHING passes it must select nothing and say so
        root2 = tempfile.mkdtemp(prefix="coral_r4_an2_")
        try:
            for g in (1, 4, 16):
                d = os.path.join(root2, f"g{g}")
                os.makedirs(d)
                with open(os.path.join(d, "steps.csv"), "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["step", "sup_loss", "coral_raw", "src_feat_l2",
                                "tgt_feat_l2", "src_perdim_std", "tgt_perdim_std"])
                    for i, s in enumerate(steps):
                        w.writerow([s, 0.05, 0.02 + 1e-5 * s,
                                    66 + 0.012 * s, 66 + 0.012 * s, 0.7, 0.7])
            out2 = os.path.join(root2, "verdict.json")
            AN.main(["--sweep-dir", root2, "--bootstrap", "300", "--out", out2])
            v2 = json.load(open(out2))
            check(v2["selected_gamma"] is None and v2["passing_gammas"] == [],
                  "when no gamma passes, NOTHING is selected")
            check("EXTEND THE SWEEP UPWARD" in v2["recommendation"],
                  "and the recommendation is to extend the sweep upward, not to relax "
                  "the criterion")
        finally:
            shutil.rmtree(root2, ignore_errors=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_all():
    t10_change1_ema_off()
    t10_change2_norm_ceiling()
    t10_sweep_criterion()
    t10_sweep_runner_and_launcher()
    t10_analyzer_end_to_end()


if __name__ == "__main__":
    print("=" * 88)
    print("ROUND-4 CORAL -- T10: the two changes + the gamma sweep")
    print("=" * 88)
    run_all()
    print("\n" + "=" * 88)
    print(f"RESULT: {len(T._passes)} passed, {len(T._fails)} failed")
    for f in T._fails:
        print(f"  FAILED: {f}")
    print("=" * 88)
    sys.exit(1 if T._fails else 0)
