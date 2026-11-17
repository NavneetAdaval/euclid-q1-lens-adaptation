"""
test_validators.py
==================
Tests for ``validators.py`` (the six label-free validators) and for the analysis helpers in
``study_validators.py`` that the A1 verdict rests on.

Three layers, in the order they can break:
  1. the validator MATH          -- delegated to ``validators.run_tests()`` (known answers)
  2. the ANALYSIS helpers        -- regret, the R1/R2/R3 rules, partial Spearman
  3. the REAL-DATA contracts     -- the balanced target pool, the frozen-encoder SND
                                    constancy, and the checkpoint-reconstruction gate

Layer 3 is skipped with a clear message if the feature cache or the runs are absent, so this
file still runs on a machine that only has the code.

    python scripts/test_validators.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import validators as V                                    # noqa: E402
import study_validators as S                              # noqa: E402

CACHE = os.path.join(_ROOT, "data", "feature_cache")


# ---------------------------------------------------------------------------
# layer 2 -- the analysis helpers
# ---------------------------------------------------------------------------

def test_regret():
    tpr = np.array([0.10, 0.30, 0.20, 0.25])

    # A validator that ranks perfectly has zero regret.
    reg, pick = S.regret_for(np.array([1.0, 4.0, 2.0, 3.0]), tpr, +1)
    assert reg == 0.0 and pick == 1, (reg, pick)
    print("  [PASS] regret == 0 for a perfectly-ranking validator")

    # A validator that ranks perfectly BACKWARDS has maximal regret.
    reg, pick = S.regret_for(np.array([4.0, 1.0, 3.0, 2.0]), tpr, +1)
    assert abs(reg - 0.20) < 1e-12 and pick == 0, (reg, pick)
    print("  [PASS] regret == oracle-worst for an inverted validator")

    # Direction is honoured: the same scores read as argmin pick the other end.
    reg_min, pick_min = S.regret_for(np.array([4.0, 1.0, 3.0, 2.0]), tpr, -1)
    assert pick_min == 1 and reg_min == 0.0, (reg_min, pick_min)
    print("  [PASS] direction = -1 selects by argmin")

    # A constant validator degenerates to the documented tie-break (earliest checkpoint).
    reg, pick = S.regret_for(np.ones(4), tpr, +1)
    assert pick == 0, pick
    print("  [PASS] constant validator falls back to the earliest-checkpoint tie-break")

    # NaNs are ignored rather than poisoning the argmax.
    reg, pick = S.regret_for(np.array([np.nan, 4.0, 2.0, np.nan]), tpr, +1)
    assert pick == 1 and reg == 0.0, (reg, pick)
    print("  [PASS] NaN validator entries are skipped, not selected")

    # Too little data -> no pick, rather than a fake one.
    assert S.regret_for(np.array([np.nan, np.nan, np.nan, np.nan]), tpr, +1)[1] is None
    print("  [PASS] all-NaN validator returns no pick")


def test_rules():
    steps = np.array([250, 2000, 6000, 6500, 7000, 8000])
    tpr = np.array([0.10, 0.30, 0.20, 0.25, 0.15, 0.22])
    r = S._rule_indices(steps, tpr, 8000)
    assert steps[r["R1 last"]] == 8000
    assert steps[r["R2 fixed-plateau"]] == 6000          # last step <= 6000
    # window [6000, 8000] holds tprs 0.20, 0.25, 0.15, 0.22 (steps 6000/6500/7000/8000).
    # Sorted: 0.15, 0.20, 0.22, 0.25. n=4 is even, and the documented tie-break takes the
    # LOWER median = the 2nd smallest = 0.20, i.e. step 6000.
    assert steps[r["R3 median"]] == 6000, steps[r["R3 median"]]
    print("  [PASS] R1/R2/R3 pick the documented steps")

    # An empty [6000, budget] window must yield None, not a silent wrong pick.
    r2 = S._rule_indices(np.array([250, 500, 900]), np.array([0.1, 0.2, 0.3]), 8000)
    assert r2["R3 median"] is None and r2["R2 fixed-plateau"] is not None
    print("  [PASS] empty plateau window -> R3 is None (reported n/a)")

    # control_step's argmax IS R1, by construction -- the internal consistency check that
    # the report relies on when it shows control_step tying R1 on every run.
    reg_ctrl, pick = S.regret_for(steps.astype(float), tpr, +1)
    assert pick == r["R1 last"]
    print("  [PASS] argmax(control_step) is exactly R1 (report consistency check)")


def test_partial_spearman():
    rng = np.random.default_rng(3)
    z = np.arange(40.0)
    # x and y are independent GIVEN z, but both trend with z -> large raw rho, ~0 partial.
    x = z + rng.normal(0, 3, 40)
    y = z + rng.normal(0, 3, 40)
    raw, _ = S._spearman(x, y)
    par, _ = S.partial_spearman(x, y, z)
    assert raw > 0.75, raw
    assert abs(par) < 0.4, par
    print(f"  [PASS] partial_spearman removes a shared trend (raw {raw:+.3f} -> "
          f"partial {par:+.3f})")

    # A genuine relationship survives partialling.
    y2 = x + rng.normal(0, 0.5, 40)
    assert S.partial_spearman(x, y2, z)[0] > 0.7
    print("  [PASS] partial_spearman keeps a genuine relationship")

    # Degenerate z (constant) must not blow up.
    assert np.isfinite(S.partial_spearman(x, y, np.ones(40))[0])
    print("  [PASS] partial_spearman survives a constant covariate")


def test_fisher_pool():
    rho, lo, hi, k = S._fisher_pool([0.5, 0.5, 0.5], [30, 30, 30])
    assert abs(rho - 0.5) < 1e-9 and lo < 0.5 < hi and k == 3
    print("  [PASS] Fisher pooling of identical rhos returns that rho, with a CI")
    # rho = +/-1 is excluded (arctanh is infinite there) rather than poisoning the mean.
    rho2, *_ , k2 = S._fisher_pool([1.0, 0.4], [30, 30])
    assert k2 == 1 and abs(rho2 - 0.4) < 1e-9
    print("  [PASS] |rho| == 1 entries are dropped from the pool, not propagated as inf")


# ---------------------------------------------------------------------------
# layer 3 -- real-data contracts
# ---------------------------------------------------------------------------

def test_real_contracts():
    if not os.path.isdir(CACHE):
        print(f"  [SKIP] no feature cache at {CACHE}")
        return
    sp = S._Splits(CACHE, 0)          # its constructor asserts the Trap-1 balance itself
    assert sp.tgt["features"].shape[0] == 782
    assert abs(float(sp.tgt_latent.mean()) - 0.5) < 1e-9
    print(f"  [PASS] target pool is 782 rows, exactly 1:1 balanced (Trap 1 precondition)")
    assert np.all(np.asarray(sp.tgt["labels"]) == -1)
    print("  [PASS] target labels are all UNLABELED(-1) -- label-free contract held")

    # IM must REFUSE the imbalanced test-negative pool -- the exact Trap-1 mistake.
    fake = np.zeros(sp.neg["features"].shape[0], dtype=np.int64)
    try:
        V.information_maximization(np.zeros(fake.size), latent_labels=fake)
    except ValueError as e:
        assert "NOT BALANCED" in str(e)
        print("  [PASS] IM refuses the imbalanced 26k test-negative pool")
    else:
        raise AssertionError("IM accepted the imbalanced pool")

    # SND on backbone features is CONSTANT across checkpoints on a frozen encoder.
    # This is why the brief's "SND on target features" reading cannot select anything here.
    c1 = V.snd_from_features(sp.tgt["features"])
    c2 = V.snd_from_features(sp.tgt["features"].copy())
    assert c1 == c2
    print(f"  [PASS] feature-space SND is a constant ({c1:.6f}) -> a known-null, not a "
          f"candidate")


def test_reconstruction_gate():
    """The gate that everything downstream depends on: a checkpoint's recomputed real-test
    TPR must reproduce the value curve.csv logged for that step."""
    run = os.path.join(_ROOT, "runs", "baselineA_fold0_nb0_image")
    if not (os.path.isdir(CACHE) and os.path.isdir(run)):
        print("  [SKIP] cache or nb0_image run missing")
        return
    inv = S.inventory([("nb0_image", "baselineA_fold0_nb0_image", 0, "head", "frozen")],
                      verbose=False)
    assert inv and inv[0]["n_ckpt"] >= 3, inv
    sp = S._Splits(CACHE, 0)
    out = S.compute_run(inv[0], sp, None, verbose=False)
    assert out["n_fail"] == 0, out["verify"]
    mx = max(x["d_tpr"] for x in out["verify"])
    assert mx == 0.0, mx
    print(f"  [PASS] all {len(out['verify'])} nb0_image checkpoints reproduce curve.csv "
          f"EXACTLY (max |dTPR| = {mx:.1e})")

    # And the gate actually BITES: a corrupted logged value must be rejected, not absorbed.
    rec = dict(inv[0])
    curve = {k: v.copy() for k, v in rec["curve"].items()}
    curve[S.TPR_COL][3] += 0.05                      # 5 lenses of corruption on one row
    rec["curve"] = curve
    rec["steps_with_ckpt"] = rec["steps_with_ckpt"][:6]
    bad = S.compute_run(rec, sp, None, verbose=False)
    assert bad["n_fail"] == 1, bad["n_fail"]
    assert len(bad["steps"]) == 5
    print("  [PASS] the gate REJECTS a checkpoint whose logged TPR was corrupted "
          "(and drops it from the validator series)")


def main():
    print("=" * 74)
    print("test_validators.py")
    print("=" * 74)
    print("\n[1] validator math (delegated to validators.run_tests)")
    print("-" * 74)
    V.run_tests()
    print("\n[2] analysis helpers")
    print("-" * 74)
    test_regret()
    test_rules()
    test_partial_spearman()
    test_fisher_pool()
    print("\n[3] real-data contracts")
    print("-" * 74)
    test_real_contracts()
    test_reconstruction_gate()
    print("\n" + "=" * 74)
    print("ALL TESTS PASSED")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
