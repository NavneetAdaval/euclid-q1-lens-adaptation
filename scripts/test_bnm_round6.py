"""
test_bnm_round6.py
==================
Tests for ROUND 6's installer and rules (``bnm_round6``). The loss itself is covered by
``test_da_bnm.py``; this file covers the wiring, which is where the arm's *claim* lives.

  T1  THE SOURCE TENSOR IS NOT IN THE LOSS. Replacing ``src_adapter`` with completely
      different values leaves the returned loss BIT-IDENTICAL, and leaves no gradient on
      anything only the source touches. THIS IS THE POINT OF THE ARM: "no domain
      comparison in the loss at all" is a claim about the code, so it is tested as one.
      CoralLoss and DannLoss both fail this test by construction -- they must.

  T2  the hook produces exactly the head's own target logits (no reimplementation).
  T3  the head is NOT registered as a submodule -- zero parameters, zero buffers, so the
      optimiser cannot see head params twice and ``Trainer.save`` writes no BNM state.
  T4  install refuses to overwrite CORAL, refuses DANN, refuses a double install, and
      refuses when the AdaMatch/FreeMatch branch is selected.
  T5  install sets the hook, use_coral and coral_weight=gamma.
  T6  THE COLLAPSE BAND DOES NOT FIRE AT STEP 0 -- the standing rule, now earned three
      times (round 5's abort3b, condition 4, and the 0.25-0.75 absolute band this one
      replaces). Also: the band admits 7.20's PREDICTED drift toward ~0.5 while still
      excluding both degenerate solutions.
  T6b THE CEILING IS 0.75 AND ADVISORY-ONLY, the floor gates. A ceiling that can only
      trip on catastrophe is a formality; one tight enough to matter can fire on
      legitimate behaviour. Resolved by STATUS rather than by loosening.
  T7  the budget rule: fires on a crossing, and REFUSES (budget None) on EITHER branch --
      ceiling never reached, or the rule returning more than the declared 8000 cap. The
      refusal is never a silent clamp, and the two branches stay distinguishable.
  T8  the Stage-0a anchors are LOADED from the artifact, not retyped -- corrupting the
      file changes what the rules see.
  T9  gamma grid scaling: gamma_i is proportional to the requested fraction.
  T10 THE RUNTIME INVARIANT. A token scan for set_step/horizon/ramp/... is defeated by
      arithmetic -- `w = min(1, i/2000); loss = ce + w*gamma*bnm` contains none of those
      tokens. So gamma is checked at the LOSS-ASSEMBLY SITE, and the brief's own ramp is
      INJECTED here as the known-null: a guard never seen to fail is not known to work.
  T11 --measure-step0 is METRIC-BLIND by construction: an AST scan of the whole path
      finds no route to evaluate/metrics, with step0_gate as the known-null proving the
      scan can see one.
  T12 the gamma grid READS the --measure-step0 artifact; corrupting the file moves the
      grid, and an inert BNM term stops rather than dividing by zero.
  T13 the 704/78 memorisation diagnostic is SIZE-MATCHED. Normalised ||A||_* is
      sqrt(B/C) at a confident state, so 704-vs-78 reads 3.00x higher at IDENTICAL
      behaviour -- an unmatched comparison would report sqrt(B) as memorisation. The gap
      helper REFUSES a mismatched pair rather than returning a number.
  T14 Stage 2's selection and the extension rule, both declared before any sweep arm
      runs: smallest passing gamma; none passing -> one x4 extension; that failing too ->
      STOP AND REPORT with its own reason code. And the advisory ceiling genuinely does
      NOT gate -- an arm at pred-pos 0.95 is selectable, with the breach ESCALATED.

  T15-T18 exist because T1-T14 were all GREEN while the runner crashed twice on the
  venue, minutes into a sweep arm. Every one of them tests a path no unit test executed:

  T15 every call site BINDS against the callee's real signature. The runner called
      LiveEncoder(splits, data_root, cache_dir=...) -- a keyword it has never had, and
      missing the required positional `fold`. Known-null: that historical call.
  T16 the instruments are FINITE IN fp16, which is what the venue runs. BNM_EPS (1e-12)
      is below fp16's smallest subnormal, so clamp_min(BNM_EPS) is a NO-OP in half
      precision: log(0) = -inf, 0 * -inf = nan. Every venue entropy reading was nan while
      the nuclear-norm column looked healthy, because bnm_nuclear_norm already casts with
      .float(). Not cosmetic: entropy_gap is half of T13's memorisation signature.
  T17 probe_stats and holdout_gap are EXECUTED against a stub trainer, in both dtypes.
      T13 exercised holdout_gap on hand-built dicts -- the half that never touches a
      model -- so a NameError in probe_stats survived to the venue.
  T18 no module uses a name it never defines or imports. A missing import is invisible to
      any check that does not execute the line, and the two live sites need a GPU.

  T19 THE HYBRID CADENCE (Stage-3 brief 2). Every 250-multiple keeps its full
      evaluation, so no cross-arm comparison changes grid (7.22's lesson); the 18
      sub-grid points are STATISTICS ONLY. Metric-blindness is asserted first from
      the SIGNATURE -- stats_only_row never receives a scorer, so it cannot score
      even by mistake -- then from the body, with the full path as the known-null.
  T20 no_matched_arm_budget: a third NAMED outcome that does NOT refuse. The rule's
      budget must be EXACTLY one of the declared arm budgets {1500, 2000, 4000} --
      set membership, not a floor, because 7.22 contrasts at a MATCHED step and 1600
      is not comparable to FreeMatch's 1500 merely by exceeding it. The selected
      arm's own Stage-2 trajectory lands on it (budget 1000), and the two existing
      refusals stay distinguishable and un-clamped.

  T21 PATH 2. The original T4/fp16 measurement's session was lost, so a second
      measurement replaces it -- under its own filename, with a material
      threshold declared BEFORE it ran, and never re-deriving the swept grid or
      re-selecting gamma. The risk path 2 carries is a lookalike being
      substituted for real provenance, which the finished artifact could not
      show; these checks are that substitution's tripwire.

  T22 THE PROTECTED SET. Every artifact whose hash PROJECT.md records is read-only
      (an overwrite FAILS rather than succeeding quietly) and is re-hashed at the
      start AND end of every analysis script. Known-null three ways: unchanged
      passes, changed refuses, absent refuses. Plus: no script writes to a
      repo-anchored path that --out does not control.

    python -u scripts/test_bnm_round6.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bnm_round6 as B  # noqa: E402
from da_losses import binary_prob2, bnm_nuclear_norm  # noqa: E402
from heads import build_head  # noqa: E402

_fails: list[str] = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        _fails.append(msg)


_skips: list[str] = []


def skip(msg, why):
    """Record a check that CANNOT run here -- visibly, and never as a pass.

    A check that silently vanishes in one environment is worse than no check: the suite
    still prints green and nobody learns which assertions were actually made.
    """
    print(f"  SKIP  {msg}\n        ({why})")
    _skips.append(f"{msg} -- {why}")


def make_hook(gamma=1.0, in_dim=640):
    head = build_head("linear", in_dim=in_dim)
    return head, B.BNMHook(head, gamma)


# ---------------------------------------------------------------------------

def t1_source_not_in_loss():
    print("\nT1 -- THE SOURCE TENSOR IS NOT IN THE LOSS (the arm's central claim)")
    torch.manual_seed(0)
    head, hook = make_hook()
    hook.train()

    tgt = torch.randn(64, 640, requires_grad=True)
    src_a = torch.randn(64, 640, requires_grad=True)
    src_b = torch.randn(64, 640, requires_grad=True) * 17.0 + 3.0

    va = hook(src_a, tgt)
    vb = hook(src_b, tgt)
    check(float(va.detach()) == float(vb.detach()),
          f"T1 loss is BIT-IDENTICAL under a totally different source batch "
          f"({float(va.detach()):.10f})")

    va.backward()
    check(src_a.grad is None or float(src_a.grad.abs().sum()) == 0.0,
          "T1 no gradient reaches the source adapter tensor")
    check(tgt.grad is not None and float(tgt.grad.abs().sum()) > 0,
          "T1 gradient DOES reach the target adapter tensor")

    # The known-null: CoralLoss must FAIL the same test, or T1 proves nothing.
    from da_losses import CoralLoss
    coral = CoralLoss(640)
    coral.train()
    t2 = torch.randn(64, 640, requires_grad=True)
    ca = float(coral(torch.randn(64, 640, requires_grad=True), t2).detach())
    cb = float(coral(torch.randn(64, 640, requires_grad=True) * 17.0 + 3.0, t2).detach())
    check(ca != cb,
          "T1 KNOWN-NULL: CoralLoss DOES change with the source batch, so the test bites")


def t2_t3_hook_shape():
    print("\nT2-T3 -- the hook reproduces the head's logits and owns nothing")
    torch.manual_seed(1)
    head, hook = make_hook()
    x = torch.randn(32, 640)
    logits_ref, adapter = head.forward_with_features(x)
    check(torch.equal(hook.target_logits(adapter), logits_ref),
          "T2 hook.target_logits(adapter) == head.forward_with_features(x)[0], exactly")

    check(len(list(hook.parameters())) == 0,
          f"T3 hook owns NO parameters (got {len(list(hook.parameters()))})")
    check(len(list(hook.buffers())) == 0, "T3 hook owns NO buffers")
    check(len(list(hook.children())) == 0,
          "T3 the head is not registered as a child module")


class _FakeCfg:
    use_coral = False
    use_adamatch = False
    use_freematch = False
    coral_weight = 1.0


class _FakeTrainer:
    def __init__(self, head):
        self.head = head
        self.coral = torch.nn.Identity()
        self.cfg = _FakeCfg()
        self.dann = None


def t4_t5_install():
    print("\nT4-T5 -- install refuses the wrong situations and sets the right ones")
    head = build_head("linear", in_dim=640)

    tr = _FakeTrainer(head)
    hook = B.install_bnm(tr, gamma=4.0)
    check(tr.coral is hook, "T5 install puts the hook in the coral slot")
    check(tr.cfg.use_coral is True, "T5 install sets use_coral=True")
    check(tr.cfg.coral_weight == 4.0, "T5 install sets coral_weight = gamma")
    check(hook.gamma == 4.0, "T5 the hook records gamma")

    try:
        B.install_bnm(tr, gamma=1.0)
        check(False, "T4 refuses a DOUBLE install")
    except RuntimeError:
        check(True, "T4 refuses a DOUBLE install")

    from coral_round4 import Fp32CoralLoss
    tr2 = _FakeTrainer(build_head("linear", in_dim=640))
    tr2.coral = Fp32CoralLoss(640)
    try:
        B.install_bnm(tr2, gamma=1.0)
        check(False, "T4 refuses to overwrite an installed CORAL")
    except RuntimeError:
        check(True, "T4 refuses to overwrite an installed CORAL")

    tr3 = _FakeTrainer(build_head("linear", in_dim=640))
    tr3.dann = object()
    try:
        B.install_bnm(tr3, gamma=1.0)
        check(False, "T4 refuses when DANN is installed")
    except RuntimeError:
        check(True, "T4 refuses when DANN is installed")

    tr4 = _FakeTrainer(build_head("linear", in_dim=640))
    tr4.cfg.use_freematch = True
    try:
        B.install_bnm(tr4, gamma=1.0)
        check(False, "T4 refuses when the FreeMatch branch is selected")
    except RuntimeError:
        check(True, "T4 refuses when the FreeMatch branch is selected")


def t6_collapse_band():
    print("\nT6 -- the collapse band does NOT fire at step 0")
    a = B.stage0a_anchors()
    s0 = a["init_pred_pos_frac"]
    band = B.collapse_band(s0)
    check(band["fires_at_step0"] is False,
          f"T6 band does not fire at the step-0 value {s0:.4f} "
          f"(floor {band['floor']:.4f}, ceiling {band['ceiling']:.2f})")
    check(B.collapse_check(s0, band)["inside"], "T6 the step-0 value is inside")

    # It must admit 7.20's PREDICTION (drift up toward ~0.5) ...
    check(B.collapse_check(0.50, band)["inside"],
          "T6 admits the PREDICTED healthy endpoint ~0.50")
    # ... while still excluding both degenerate solutions.
    check(not B.collapse_check(0.0, band)["inside"],
          "T6 excludes the all-negative collapse (0.00)")
    check(not B.collapse_check(1.0, band)["inside"],
          "T6 excludes the all-positive collapse (1.00)")

    # THE CEILING IS 0.75 AND ADVISORY. A ceiling that only trips on catastrophe is a
    # formality; 0.75 is deliberately close enough to the predicted ~0.5 endpoint to be a
    # real detector, and is therefore ADVISORY rather than loosened -- round 5's abort3b
    # arrangement generalised. The FLOOR is what gates gamma selection.
    check(band["ceiling"] == 0.75, f"T6 ceiling is 0.75 (got {band['ceiling']})")
    check("ADVISORY ONLY" in band["ceiling_status"], "T6 the ceiling is ADVISORY ONLY")
    check("GATES" in band["floor_status"], "T6 the floor GATES gamma selection")

    # the two halves are reported separately, and only the floor gates
    hi = B.collapse_check(0.85, band)
    check(hi["gate_ok"] and not hi["advisory_ceiling_ok"],
          "T6 a reading of 0.85 passes the GATE and trips only the ADVISORY ceiling")
    lo_v = B.collapse_check(0.01, band)
    check(not lo_v["gate_ok"] and lo_v["advisory_ceiling_ok"],
          "T6 a reading of 0.01 fails the GATE (the degeneracy the floor is for)")
    check(B.collapse_check(0.50, band)["gate_ok"]
          and B.collapse_check(0.50, band)["advisory_ceiling_ok"],
          "T6 the predicted ~0.50 endpoint trips neither half")

    # And the band it replaces WOULD have fired -- the reason for the change.
    check(not (0.25 <= s0 <= 0.75),
          f"T6 KNOWN-NULL: the withdrawn ABSOLUTE 0.25-0.75 band DOES fire at "
          f"step 0 ({s0:.4f} < 0.25)")


def t7_budget_rule():
    print("\nT7 -- the budget rule fires on a crossing and REFUSES otherwise")
    a = B.stage0a_anchors()
    ceil = a["sharpening_ceiling"]
    steps = list(range(250, 4001, 250))

    # must actually reach 10.8413 -- an earlier draft of this test rose to only 10.35
    # and "failed" against a module that was behaving correctly.
    rising = [9.6 + 0.09 * i for i in range(len(steps))]
    r = B.nuclear_norm_crossing(steps, rising, ceil)
    check(r["crossed"] and r["budget"] == 2 * r["crossing_step"],
          f"T7 fires on a crossing: step {r['crossing_step']} -> budget {r['budget']}")
    check(r["reads_a_metric"] is False, "T7 the rule reads no metric")

    flat = [9.7 + 0.01 * i for i in range(len(steps))]
    r2 = B.nuclear_norm_crossing(steps, flat, ceil)
    check(r2["crossed"] is False and r2["budget"] is None,
          "T7 REFUSES when the ceiling is never reached (budget None, stop and report)")

    # the crossing is the FIRST one, not the last
    bumpy = [9.6, 10.9, 9.8, 10.95] + [11.0] * (len(steps) - 4)
    r3 = B.nuclear_norm_crossing(steps, bumpy, ceil)
    check(r3["crossing_step"] == steps[1],
          f"T7 takes the FIRST crossing ({r3['crossing_step']}), not a later one")

    # THE DECLARED BUDGET CEILING. 8000 is the development cap 7.14 already fixes; a
    # confirmatory budget never exceeds the development budget. Above it the rule REFUSES.
    check(B.BUDGET_CEILING == 8000, f"T7 the ceiling is 8000 (got {B.BUDGET_CEILING})")
    long_steps = list(range(250, 8001, 250))

    at_cap = [(11.0 if x >= 4000 else 9.7) for x in long_steps]
    r4 = B.nuclear_norm_crossing(long_steps, at_cap, ceil)
    check(r4["fires"] and r4["budget"] == 8000,
          f"T7 a crossing at 4000 fires exactly AT the ceiling (budget {r4['budget']})")

    over = [(11.0 if x >= 4250 else 9.7) for x in long_steps]
    r5 = B.nuclear_norm_crossing(long_steps, over, ceil)
    check(r5["fires"] is False and r5["budget"] is None,
          "T7 a crossing at 4250 REFUSES -- the rule does not fire above the ceiling")
    # THE POINT: refusal must not be a silent clamp.
    check(r5["budget"] != B.BUDGET_CEILING and r5["budget_before_ceiling"] == 8500,
          f"T7 refusal is NOT a clamp: budget is None, and the raw "
          f"{r5['budget_before_ceiling']} is preserved so the two cases stay "
          f"distinguishable in the JSON")
    check(r5["exceeds_ceiling"] is True and "above the declared ceiling" in
          (r5["refusal_reason"] or ""),
          "T7 the refusal records WHICH branch fired")

    never = B.nuclear_norm_crossing(long_steps, [9.7] * len(long_steps), ceil)
    check(never["refusal_reason"] != r5["refusal_reason"],
          "T7 the two refusal branches are distinguishable from each other")


def t8_anchors_from_artifact():
    print("\nT8 -- the anchors are loaded from the artifact, not retyped")
    a = B.stage0a_anchors()
    check(abs(a["balanced_confident"] - 11.3137) < 1e-3
          and abs(a["collapsed_confident"] - 8.0) < 1e-9
          and abs(a["sharpening_ceiling"] - 10.8413) < 1e-3,
          "T8 anchors match Stage 0a's recorded values")

    with open(B.STAGE0A, encoding="utf-8") as fh:
        blob = json.load(fh)
    blob["part_b"]["decomposition"]["confident_at_current_split"] = 99.0
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fake.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        check(B.stage0a_anchors(p)["sharpening_ceiling"] == 99.0,
              "T8 corrupting the artifact changes what the rules see (no hidden literal)")


def t9_grid_scaling():
    print("\nT9 -- the gamma grid scales with the requested fractions")
    ce, bnm = 4.0, 0.5
    grid = [round(f * ce / bnm, 6) for f in B.GRID_FRACTIONS]
    check(grid == [0.8, 4.0, 16.0],
          f"T9 gamma_i = fraction_i * ||CE|| / ||BNM|| -> {grid}")
    check(tuple(B.GRID_FRACTIONS) == (0.10, 0.50, 2.00),
          "T9 the declared fractions are {10%, 50%, 200%}")

    # The grid must be DERIVED from the measured norms, not a constant that happens to
    # look like CORAL's {1, 4, 16}. (At ce/bnm = 8 it coincidentally equals {0.8, 4, 16};
    # an earlier draft of this test mistook that coincidence for a property.)
    grid2 = [round(f * 7.0 / 0.3, 6) for f in B.GRID_FRACTIONS]
    # compared with a tolerance: the module rounds gamma to 6 dp, so exact float
    # equality here would fail on the rounding rather than on the property.
    ratio, expect = grid2[1] / grid[1], (7.0 / 0.3) / (ce / bnm)
    check(grid2 != grid and abs(ratio - expect) < 1e-6,
          f"T9 the grid MOVES with the measured norms -- derived, not copied "
          f"({ratio:.6f} vs {expect:.6f})")


def t10_gamma_invariant():
    print("\nT10 -- THE RUNTIME INVARIANT: inject a ramp and watch it fire")
    import random

    # healthy: gamma constant, the raw term varying as it does in a real run
    inv = B.GammaInvariant(4.0)
    random.seed(0)
    for s in range(1, 501):
        raw = random.uniform(-11.4, -5.6)
        inv.observe(s, cfg_coral_weight=4.0, hook_gamma=4.0,
                    da_loss=4.0 * raw, raw_loss=raw)
    a = inv.to_artifact()
    check(a["held"] and a["n_observations"] == 500,
          f"T10 holds over 500 healthy steps ({a['n_observations']} observations)")
    check(a["gamma_effective_min"] == a["gamma_effective_max"] == 4.0,
          "T10 the effective gamma is stable across a varying raw term")

    # THE KNOWN-NULL, and it is the brief's own example verbatim:
    #     w = min(1.0, i / 2000.0);  loss = ce + w * gamma * bnm
    # None of the scanned tokens (set_step / lambda_at / total_steps / horizon / step /
    # warmup / ramp / schedule) appear in that line, so a static scan cannot see it.
    inv2 = B.GammaInvariant(4.0)
    random.seed(1)
    fired_at = None
    for s in range(1, 4001):
        raw = random.uniform(-11.4, -5.6)
        w = min(1.0, s / 2000.0)
        try:
            inv2.observe(s, cfg_coral_weight=4.0, hook_gamma=4.0,
                         da_loss=w * 4.0 * raw, raw_loss=raw)
        except B.GammaHorizonDetected:
            fired_at = s
            break
    check(fired_at is not None,
          f"T10 KNOWN-NULL: the brief's ramp FIRES the invariant (step {fired_at})")
    check(fired_at is not None and fired_at <= 5,
          f"T10 ...and fires almost immediately (step {fired_at}), not after 2000")
    art = inv2.to_artifact()
    check(art["held"] is False and art["first_divergence"]["step"] == fired_at,
          "T10 the artifact records held=False and names the diverging step")
    check("gamma_effective" in art["first_divergence"]["quantity"],
          "T10 the divergence names WHICH quantity moved")

    # a ramp implemented by mutating the config instead is caught by the EXACT layer
    inv3 = B.GammaInvariant(4.0)
    inv3.observe(1, cfg_coral_weight=4.0, hook_gamma=4.0, da_loss=4.0, raw_loss=1.0)
    try:
        inv3.observe(2, cfg_coral_weight=4.0000001, hook_gamma=4.0,
                     da_loss=4.0000001, raw_loss=1.0)
        check(False, "T10 a mutated cfg.coral_weight fires the BIT-IDENTITY layer")
    except B.GammaHorizonDetected:
        check(inv3.first_divergence["comparison"] == "bit-identity",
              "T10 a mutated cfg.coral_weight fires the BIT-IDENTITY layer")

    # the sequence is persisted, so the guard is verifiable after the fact
    check(len(art["steps"]) == len(art["gamma_effective"]) > 0
          and "why_the_two_layers_differ" in art,
          "T10 the full sequence and the tolerance rationale ride in the artifact")


def t11_measure_step0_is_metric_blind():
    print("\nT11 -- --measure-step0 is METRIC-BLIND by construction, asserted not assumed")
    import ast as _ast

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_bnm_r6.py"), encoding="utf-8").read()
    tree = _ast.parse(src)
    funcs = {n.name: n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)}
    check("measure_step0" in funcs, "T11 run_bnm_r6.measure_step0 exists")

    # The metric-blind path is measure_step0 + what it calls in-module (build_arm) +
    # bnm_gradient_calibration. Scan all three for any route to a metric.
    # BANNED: anything that computes a metric or scores the test pool.
    #
    # `FeatureCache` is deliberately NOT banned. build_arm imports it from `evaluate` to
    # fit the feature standardizer from the cached SOURCE features -- the one cache read
    # that stays valid at n_blocks>0, because it is a constant of the source domain rather
    # than a function of the encoder. It is a cache accessor, not a metric. To keep the
    # claim honest rather than merely narrower, the SECOND check below asserts that no
    # test-set cache key is named anywhere on this path, so the allowance cannot be used
    # to reach test data.
    banned = {"metrics", "me", "tpr_at_fpr", "auroc", "metric_suite", "auprc",
              "roc_curve_full", "score_features", "Scorer", "step0_gate"}
    allowed_from_evaluate = {"FeatureCache"}
    on_path = [funcs[k] for k in ("measure_step0", "build_arm") if k in funcs]
    hits = []
    for fn in on_path:
        for node in _ast.walk(fn):
            if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                mod = getattr(node, "module", None) or ""
                names = [a.name for a in node.names]
                if mod.split(".")[0] in banned:
                    hits.append(f"{fn.name}: import {mod}")
                for nm in names:
                    root = nm.split(".")[0]
                    if root in banned:
                        hits.append(f"{fn.name}: import {nm}")
                    if mod == "evaluate" and root not in allowed_from_evaluate:
                        hits.append(f"{fn.name}: from evaluate import {nm}")
            elif isinstance(node, _ast.Name) and node.id in banned:
                hits.append(f"{fn.name}: name {node.id}")
            elif isinstance(node, _ast.Attribute):
                v = node.value
                if isinstance(v, _ast.Name) and v.id in banned:
                    hits.append(f"{fn.name}: {v.id}.{node.attr}")
                # `ev.<anything>` is allowed only for the whitelisted accessor
                if isinstance(v, _ast.Name) and v.id in ("ev", "_ev", "evaluate")                         and node.attr not in allowed_from_evaluate:
                    hits.append(f"{fn.name}: {v.id}.{node.attr}")
    check(not hits, f"T11 no route to a metric on the --measure-step0 path "
                    f"({[f.name for f in on_path]}); hits={hits}")

    # THE ALLOWANCE CANNOT REACH TEST DATA: no test-set cache key is named on this path.
    #
    # Matched as EXACT cache keys, not substrings, and with docstrings excluded. A
    # substring test flagged the words "value", "evaluate" and the prose of an error
    # message -- the same false-positive class as the audit's ramp scan firing on the
    # runner's own docstring. A scan over prose is not a scan over code.
    import re as _re

    def _cache_key(v):
        return v in ("val", "test_negatives_shared") or bool(
            _re.fullmatch(r"test_pos_fold\d+", v))

    key_hits = []
    for fn in on_path:
        body = fn.body[1:] if (fn.body and isinstance(fn.body[0], _ast.Expr)
                               and isinstance(getattr(fn.body[0], "value", None),
                                              _ast.Constant)) else fn.body
        for stmt in body:
            for n in _ast.walk(stmt):
                if isinstance(n, _ast.Constant) and isinstance(n.value, str)                         and _cache_key(n.value):
                    key_hits.append(f"{fn.name}: {n.value!r}")
    check(not key_hits,
          f"T11 no TEST-SET cache key is named on this path (only source_train); "
          f"hits={key_hits}")

    # KNOWN-NULL for that check too: the Scorer path DOES name test keys.
    scorer_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "run_coral_r4_dev8000.py"), encoding="utf-8").read()
    check(any(_cache_key(n.value) for n in _ast.walk(_ast.parse(scorer_src))
              if isinstance(n, _ast.Constant) and isinstance(n.value, str)),
          "T11 KNOWN-NULL: the scoring path DOES name test cache keys, so the check bites")

    cal = _ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "bnm_round6.py"), encoding="utf-8").read())
    calfn = next((n for n in _ast.walk(cal) if isinstance(n, _ast.FunctionDef)
                  and n.name == "bnm_gradient_calibration"), None)
    check(calfn is not None, "T11 bnm_gradient_calibration exists")
    cal_hits = [n.id for n in _ast.walk(calfn)
                if isinstance(n, _ast.Name) and n.id in banned]
    check(not cal_hits, f"T11 bnm_gradient_calibration touches no metric ({cal_hits})")

    # KNOWN-NULL: the scan must be able to SEE a metric. step0_gate uses one.
    gate = funcs.get("step0_gate")
    gate_hits = [n for n in _ast.walk(gate)
                 if isinstance(n, _ast.Name) and n.id in banned] if gate else []
    check(bool(gate_hits),
          "T11 KNOWN-NULL: the same scan DOES find metrics in step0_gate, so it bites")


def t12_grid_reads_the_file():
    print("\nT12 -- the gamma grid READS the measurement, no hidden literal")
    import tempfile as _tf

    blob = {"ce_grad_norm": 4.0, "bnm_grad_norm_at_gamma_1": 0.5,
            "fractions": [0.10, 0.50, 2.00]}
    with _tf.TemporaryDirectory() as d:
        p = os.path.join(d, "g.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        g = B.gamma_grid_from_file(p)
        check(g["gamma_grid"] == [0.8, 4.0, 16.0],
              f"T12 grid from file -> {g['gamma_grid']}")
        check(g["reads_a_metric"] is False, "T12 the reader reads no metric")

        # corrupt it: the grid must follow the file (the T8 discipline)
        blob["ce_grad_norm"] = 40.0
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        g2 = B.gamma_grid_from_file(p)
        check(g2["gamma_grid"] == [8.0, 40.0, 160.0],
              f"T12 corrupting the file MOVES the grid -> {g2['gamma_grid']}")

        # an inert BNM term must stop rather than produce an infinite gamma
        blob["bnm_grad_norm_at_gamma_1"] = 0.0
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        try:
            B.gamma_grid_from_file(p)
            check(False, "T12 a zero BNM gradient norm STOPS rather than dividing by zero")
        except SystemExit:
            check(True, "T12 a zero BNM gradient norm STOPS rather than dividing by zero")


def t13_size_matched_probe():
    print("\nT13 -- the 704/78 diagnostic is SIZE-MATCHED, and must be")
    import math as _m

    # THE TRAP, stated as a test: normalised ||A||_* is sqrt(B/C) at a confident state,
    # so an unmatched comparison reports sqrt(B) as memorisation.
    def conf(n):
        z = torch.full((n, 1), 8.0)
        z[: n // 2] = -8.0
        return float(bnm_nuclear_norm(binary_prob2(z)))

    a, b = conf(704), conf(78)
    check(abs(a / b - _m.sqrt(704 / 78)) < 0.01,
          f"T13 at IDENTICAL behaviour 704 reads {a/b:.2f}x higher than 78 "
          f"(= sqrt(704/78) = {_m.sqrt(704/78):.2f}) -- the confound the matching removes")

    # the gap helper REFUSES an unmatched pair rather than returning a number
    try:
        B.holdout_gap({"n": 704, "nuclear_norm": a, "entropy": 0.1, "pred_pos_frac": 0.3},
                      {"n": 78, "nuclear_norm": b, "entropy": 0.1, "pred_pos_frac": 0.3})
        check(False, "T13 holdout_gap REFUSES slices of different sizes")
    except ValueError:
        check(True, "T13 holdout_gap REFUSES slices of different sizes")

    # matched slices give a gap of exactly zero when behaviour is identical
    same = {"n": 78, "nuclear_norm": b, "entropy": 0.10, "pred_pos_frac": 0.30}
    g = B.holdout_gap(dict(same), dict(same))
    check(g["nuclear_gap"] == 0.0 and g["entropy_gap"] == 0.0,
          "T13 identical behaviour on matched slices gives a zero gap")

    # the MEMORISING signature is the one the diagnostic exists to name
    g2 = B.holdout_gap({"n": 78, "nuclear_norm": b + 1.5, "entropy": 0.02,
                        "pred_pos_frac": 0.55},
                       {"n": 78, "nuclear_norm": b, "entropy": 0.30,
                        "pred_pos_frac": 0.31})
    check(g2["nuclear_gap"] > 0 and g2["entropy_gap"] < 0,
          "T13 memorising signature: positive nuclear gap WITH negative entropy gap")


def t14_selection_and_extension():
    print("\nT14 -- Stage 2 selection, and the extension rule declared before the sweep")
    a = B.stage0a_anchors()
    band = B.collapse_band(a["init_pred_pos_frac"])

    def arm(g, nuc, pos, probe=0.90):
        return dict(gamma=g, nuclear_norm_final=nuc, pred_pos_frac_final=pos,
                    frozen_probe_auroc_final=probe, frozen_probe_auroc_step0=0.9202)

    r = B.select_gamma([arm(0.01, 10.5, 0.35), arm(0.05, 10.9, 0.42),
                        arm(0.20, 11.1, 0.55)], band=band, init_nuclear=a["init"])
    check(r["selected"] == 0.01 and r["n_passing"] == 3,
          "T14 three pass -> the SMALLEST is taken (7.15's winner's curse)")
    check(r["reads_a_metric"] is False, "T14 selection reads no metric")

    r2 = B.select_gamma([arm(0.01, 9.78, 0.24), arm(0.05, 9.80, 0.25),
                         arm(0.20, 9.82, 0.26)], band=band, init_nuclear=a["init"])
    check(r2["n_passing"] == 0 and r2["action"] == "EXTEND",
          "T14 none passes -> EXTEND, rather than an undeclared judgement call")
    check(r2["next_grid"] == [round(0.20 * B.GRID_EXTENSION_MULTIPLIER, 6)],
          f"T14 the extension multiplies the TOP gamma by "
          f"{B.GRID_EXTENSION_MULTIPLIER} -> {r2['next_grid']}")

    r3 = B.select_gamma([arm(0.80, 9.82, 0.26)], band=band, init_nuclear=a["init"],
                        n_extensions_used=1)
    check(r3["action"] == "STOP_AND_REPORT"
          and r3["reason_code"] == "no_gamma_passed_after_extension",
          "T14 the extension failing too -> STOP AND REPORT with its own reason code")
    check(B.MAX_GRID_EXTENSIONS == 1, "T14 exactly ONE extension is permitted")

    # THE CEILING NEVER GATES -- by decision. So an arm high up the all-positive
    # direction is SELECTABLE, and the breach must be escalated where a human sees it.
    r4 = B.select_gamma([arm(0.05, 10.9, 0.95)], band=band, init_nuclear=a["init"])
    check(r4["selected"] == 0.05,
          "T14 an arm at pred-pos 0.95 still PASSES -- the ceiling is advisory by design")
    check("ADVISORY_CEILING_BREACHED_BY_SELECTED_ARM" in r4,
          "T14 ...and the breach is ESCALATED in the result, not left in a column")
    r5 = B.select_gamma([arm(0.05, 10.9, 0.40)], band=band, init_nuclear=a["init"])
    check("ADVISORY_CEILING_BREACHED_BY_SELECTED_ARM" not in r5,
          "T14 a healthy arm raises no escalation (the key is not always present)")

    # the FLOOR does gate
    r6 = B.select_gamma([arm(0.05, 10.9, 0.01)], band=band, init_nuclear=a["init"])
    check(r6["n_passing"] == 0,
          "T14 an arm below the GATING floor does not pass")



def t15_call_sites_bind():
    print("\nT15 -- every call into a sibling module BINDS against its real signature")
    import ast as _ast, importlib as _il, inspect as _isp, io as _io

    # WHY THIS EXISTS. run_bnm_r6.main() called
    #     LiveEncoder(paths["splits"], paths["data_root"], cache_dir=...)
    # which omits the REQUIRED positional `fold` and passes a keyword LiveEncoder has
    # never had. Nothing caught it: the suites do not construct the scoring path, and
    # --measure-step0 exits before it. The venue found it instead -- after loading the
    # encoder, unfreezing 14.8M params, fitting the standardizer and building the morph
    # pool, ~2 minutes of GPU into the first sweep arm. A signature error should not cost
    # a GPU session, so it is now a static check.

    def bind_problems(path):
        src = _io.open(path, encoding="utf-8").read()
        tree = _ast.parse(src)
        imported = {}
        for n in _ast.walk(tree):
            if isinstance(n, _ast.ImportFrom) and n.module:
                for a in n.names:
                    try:
                        imported[a.asname or a.name] = getattr(
                            _il.import_module(n.module), a.name)
                    except Exception:
                        pass
        out, n_bound = [], 0
        for n in _ast.walk(tree):
            if not isinstance(n, _ast.Call) or not isinstance(n.func, _ast.Name):
                continue
            obj = imported.get(n.func.id)
            if obj is None or _isp.ismodule(obj):
                continue
            try:
                sig = _isp.signature(obj)
            except (TypeError, ValueError):
                continue
            # *args / **kwargs at the call site cannot be bound statically -- skip rather
            # than guess, so the check never reports something it did not actually verify.
            if any(isinstance(a, _ast.Starred) for a in n.args) \
                    or any(k.arg is None for k in n.keywords):
                continue
            n_bound += 1
            try:
                sig.bind(*[_isp.Parameter.empty] * len(n.args),
                         **{k.arg: _isp.Parameter.empty for k in n.keywords})
            except TypeError as e:
                out.append(f"{os.path.basename(path)}:{n.lineno} {n.func.id}(...) -- {e}")
        return out, n_bound

    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "run_bnm_r6.py")
    probs, n_bound = bind_problems(runner)
    check(n_bound >= 20,
          f"T15 the scan actually reached the call sites ({n_bound} bound)")
    check(not probs,
          "T15 every bindable call site in run_bnm_r6.py matches its real signature"
          + ("" if not probs else ":\n        " + "\n        ".join(probs)))

    # KNOWN-NULL: the exact call that failed on the venue must be REPORTED, or this
    # test proves nothing. Reproduced as a synthetic module rather than by mutating the
    # runner's text -- an in-place edit left dangling continuation lines, the file did not
    # parse, and the "known-null" silently checked nothing. A known-null that can fail to
    # be a known-null is the failure this project keeps writing tests about, so the parse
    # is now asserted (n_bad == 1) rather than assumed.
    import tempfile as _tf
    historical = (
        "from live_encode import LiveEncoder" "\n"
        "paths = {}" "\n"
        "live = LiveEncoder(paths['splits'], paths['data_root'],"
        " cache_dir=paths.get('cache_dir'))" "\n")
    with _tf.TemporaryDirectory() as td:
        pth = os.path.join(td, "historical_call.py")
        _io.open(pth, "w", encoding="utf-8").write(historical)
        bad, n_bad = bind_problems(pth)
    check(n_bad == 1,
          f"T15 the known-null module PARSED and one call was bound (got {n_bad})")
    check(any("LiveEncoder" in b and "fold" in b for b in bad),
          "T15 KNOWN-NULL: the historical call IS reported for the missing `fold` "
          "-- the check bites")




def t16_fp16_entropy_is_finite():
    print("\nT16 -- the instruments are FINITE IN fp16, which is what the venue runs")
    import ast as _ast
    import io as _io

    # WHY THIS EXISTS. The confirmatory runs are fp16. BNM_EPS is 1e-12, which is BELOW
    # fp16's smallest subnormal (5.96e-08), so on an fp16 tensor clamp_min(BNM_EPS) rounds
    # straight back to 0.0, log(0) = -inf, and 0 * -inf = nan. Every entropy reading on
    # the venue was nan while the nuclear-norm column looked perfectly healthy --
    # bnm_nuclear_norm already casts with .float(), the entropy path did not. The tests
    # ran in fp32, so they saw none of it.
    #
    # This is not a cosmetic log column: entropy_gap is one of the TWO components of the
    # memorisation signature in T13, so a nan there deletes the Stage-2 diagnostic while
    # leaving it looking like it ran.

    # the arithmetic itself, at the saturation that produces the underflow
    for dtype in (torch.float16, torch.float32):
        logits = torch.tensor([[-30.0], [30.0], [0.0]], dtype=dtype)
        prob2 = binary_prob2(logits)
        p32 = prob2.float()
        ent = -(p32.clamp_min(B.BNM_EPS).log() * p32).sum(1).mean()
        check(torch.isfinite(ent).item(),
              f"T16 the float32 form is finite on {str(dtype).split('.')[-1]} input "
              f"(ent={float(ent):.6f})")

    # KNOWN-NULL: the PRE-FIX form must produce nan on fp16, or the fix fixed nothing
    logits16 = torch.tensor([[-30.0], [30.0]], dtype=torch.float16)
    p16 = binary_prob2(logits16)
    bad = -(p16.clamp_min(B.BNM_EPS).log() * p16).sum(1).mean()
    check(bool(torch.isnan(bad).item()),
          "T16 KNOWN-NULL: the pre-fix fp16 form IS nan -- the bug was real and is "
          "reachable at the saturation BNM drives toward")
    check(float(p16.min()) == 0.0,
          "T16 ...because fp16 underflows the probability to exactly 0 before the clamp")

    # and the clamp genuinely cannot lift it in fp16
    check(float(p16.clamp_min(B.BNM_EPS).min()) == 0.0,
          f"T16 clamp_min({B.BNM_EPS}) is a NO-OP in fp16 -- the epsilon is not "
          f"representable, so the guard silently does nothing")

    # NO ENTROPY SITE MAY STILL COMPUTE THE LOG IN THE INPUT DTYPE. Static, because
    # both live sites need a trainer and a GPU to execute, so no unit test reaches them.
    #
    # Scoped to `ent = ...` assignments deliberately. A first version flagged every
    # clamp_min in both files and reported five pre-existing, unrelated ones in other
    # losses plus the two it had just fixed -- a check that shouts at everything tells you
    # nothing, and would have been switched off within a week.
    import bnm_round6 as _mod

    def entropy_sites_in_input_dtype(path):
        src = _io.open(path, encoding="utf-8").read()
        tree = _ast.parse(src)
        bad = []
        for fn in _ast.walk(tree):
            if not isinstance(fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            # names bound in this function from an expression containing .float()
            f32 = set()
            for node in _ast.walk(fn):
                if isinstance(node, _ast.Assign) and len(node.targets) == 1 \
                        and isinstance(node.targets[0], _ast.Name):
                    seg = _ast.get_source_segment(src, node.value) or ""
                    if ".float()" in seg:
                        f32.add(node.targets[0].id)
            for node in _ast.walk(fn):
                if not (isinstance(node, _ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], _ast.Name)
                        and node.targets[0].id == "ent"):
                    continue
                seg = _ast.get_source_segment(src, node.value) or ""
                if "clamp_min" not in seg:
                    continue
                if ".float()" in seg or any(n in seg for n in f32):
                    continue
                bad.append((node.lineno, seg.strip()))
        return bad

    for path in (_mod.__file__,
                 os.path.join(os.path.dirname(os.path.abspath(_mod.__file__)),
                              "da_losses.py")):
        bad = entropy_sites_in_input_dtype(path)
        check(not bad,
              f"T16 every `ent =` clamp_min site in {os.path.basename(path)} works in "
              f"float32"
              + ("" if not bad else
                 ": " + "; ".join(f"line {ln}: {seg}" for ln, seg in bad)))

    # and the check BITES: the pre-fix line, scanned the same way, is reported
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        q = os.path.join(td, "prefix_entropy.py")
        _io.open(q, "w", encoding="utf-8").write(
            "def f(prob2, BNM_EPS):\n"
            "    ent = -(prob2.clamp_min(BNM_EPS).log() * prob2).sum(1).mean()\n"
            "    return ent\n")
        check(len(entropy_sites_in_input_dtype(q)) == 1,
              "T16 KNOWN-NULL: the pre-fix entropy line IS reported by the scan")




def t17_probe_path_actually_runs():
    print("\nT17 -- probe_stats and holdout_gap are EXECUTED, not just imported")

    # WHY THIS EXISTS. probe_stats used BNM_EPS without importing it. Nothing caught the
    # NameError because no test ever CALLED probe_stats -- T13 exercised holdout_gap on
    # hand-built dicts, which is precisely the half of the diagnostic that does not touch
    # a model. The venue found it 14 minutes into a sweep arm, on the first checkpoint.
    #
    # So the path is now run against a stub trainer: no Zoobot, no GPU, but the real
    # probe_stats, the real holdout_gap, and the real dtype.

    class _StubEncoder(torch.nn.Module):
        n_blocks = 0

        def forward(self, x):                       # (B,1,8,8) -> (B,16)
            return x.flatten(1)[:, :16]

    class _StubStd(torch.nn.Module):
        def forward(self, f):
            return f

    class _StubCfg:
        encoder_mode = "frozen"

    class _StubTrainer:
        def __init__(self, dtype):
            self.encoder = _StubEncoder()
            self.standardizer = _StubStd()
            self.cfg = _StubCfg()
            self.device = torch.device("cpu")
            self.head = build_head("linear", in_dim=16)
            # the head must carry the SAME dtype as the features. On the venue autocast
            # does this; here it is explicit, and it is the point of the fp16 case --
            # an fp32 head would silently upcast and never reproduce the nan.
            self.head = self.head.to(dtype)
            self.head.eval()
            self.dtype = dtype

    for dtype in (torch.float32, torch.float16):
        tr = _StubTrainer(dtype)
        name = str(dtype).split(".")[-1]

        # a slice the "loss saw" (driven confident) and one it did not
        g = torch.Generator().manual_seed(11)
        trained_on = torch.randn(78, 1, 8, 8, generator=g, dtype=dtype) * 6.0
        held_out = torch.randn(78, 1, 8, 8, generator=g, dtype=dtype) * 0.5

        st_t = B.probe_stats(tr, trained_on)
        st_h = B.probe_stats(tr, held_out)

        for label, st in (("trained-on", st_t), ("held-out", st_h)):
            finite = all(math.isfinite(float(st[k])) for k in
                         ("nuclear_norm", "entropy", "pred_pos_frac", "mean_p_pos"))
            check(finite,
                  f"T17 probe_stats returns all-finite instruments on {name} "
                  f"({label}: ent={st['entropy']:.4f}, "
                  f"||A||_*={st['nuclear_norm']:.4f})")
            check(st["n"] == 78, f"T17 probe_stats reports n=78 on {name} ({label})")

        gap = B.holdout_gap(st_t, st_h)
        check(all(math.isfinite(float(gap[k])) for k in
                  ("nuclear_gap", "entropy_gap", "pred_pos_frac_gap")),
              f"T17 holdout_gap on real probe_stats output is finite on {name} "
              f"(nuclear {gap['nuclear_gap']:+.4f}, entropy {gap['entropy_gap']:+.4f})")
        check(gap["n_per_slice"] == 78,
              f"T17 the gap is reported per size-matched slice on {name}")

    # the head must be left in the mode it was found in -- the monitor must not train
    tr = _StubTrainer(torch.float32)
    tr.head.train()
    B.probe_stats(tr, torch.randn(78, 1, 8, 8))
    check(tr.head.training is True,
          "T17 probe_stats RESTORES head.train() -- a monitor that leaves the head in "
          "eval() would silently change the run it is measuring")
    tr.head.eval()
    B.probe_stats(tr, torch.randn(78, 1, 8, 8))
    check(tr.head.training is False, "T17 ...and leaves an eval() head in eval()")


def t18_no_undefined_names():
    print("\nT18 -- no module uses a name it never defines or imports")
    import ast as _ast
    import builtins as _bi
    import io as _io

    # The NameError in T17's story was a MISSING IMPORT, and a missing import is invisible
    # to every check that does not execute the line. This scan reads the whole module
    # statically, so it covers the paths the tests cannot reach: the ones needing a GPU.
    _BUILTINS = set(dir(_bi))

    def _bound(node):
        out = set()
        a = getattr(node, "args", None)
        if a is not None:
            for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
                out.add(arg.arg)
            if a.vararg:
                out.add(a.vararg.arg)
            if a.kwarg:
                out.add(a.kwarg.arg)
        for n in _ast.walk(node):
            if isinstance(n, _ast.Name) and isinstance(n.ctx, (_ast.Store, _ast.Del)):
                out.add(n.id)
            elif isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                out.add(n.name)
            elif isinstance(n, _ast.Import):
                for al in n.names:
                    out.add(al.asname or al.name.split(".")[0])
            elif isinstance(n, _ast.ImportFrom):
                for al in n.names:
                    out.add(al.asname or al.name)
            elif isinstance(n, _ast.ExceptHandler) and n.name:
                out.add(n.name)
            elif isinstance(n, (_ast.Global, _ast.Nonlocal)):
                out.update(n.names)
        return out

    def undefined_in(path, runtime_names=()):
        src = _io.open(path, encoding="utf-8").read()
        tree = _ast.parse(src)
        module_level = _bound(tree) | set(runtime_names)
        found = []

        def walk(node, enclosing):
            for child in _ast.iter_child_nodes(node):
                if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                      _ast.Lambda, _ast.ClassDef)):
                    walk(child, enclosing | _bound(child))
                else:
                    if isinstance(child, _ast.Name) and isinstance(child.ctx, _ast.Load) \
                            and child.id not in enclosing \
                            and child.id not in module_level \
                            and child.id not in _BUILTINS:
                        found.append((child.lineno, child.id))
                    walk(child, enclosing)

        for top in tree.body:
            if isinstance(top, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                walk(top, module_level | _bound(top))
            else:
                walk(top, module_level)
        return sorted(set(found))

    import run_bnm_r6 as _R
    for mod in (B, _R):
        bad = undefined_in(mod.__file__, runtime_names=vars(mod))
        check(not bad,
              f"T18 no undefined names in {os.path.basename(mod.__file__)}"
              + ("" if not bad else
                 ": " + ", ".join(f"line {ln}: {n}" for ln, n in bad)))

    # KNOWN-NULL: the exact historical defect must be reported
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        q = os.path.join(td, "missing_import.py")
        _io.open(q, "w", encoding="utf-8").write(
            "import torch\n"
            "def probe_stats(prob2):\n"
            "    return prob2.clamp_min(BNM_EPS).log()\n")
        bad = undefined_in(q)
        check(any(n == "BNM_EPS" for _, n in bad),
              "T18 KNOWN-NULL: a BNM_EPS used without importing it IS reported")




def t19_hybrid_cadence_is_metric_blind():
    print("\nT19 -- the hybrid cadence: grid-matched, and the sub-grid CANNOT evaluate")
    import ast as _ast
    import inspect as _isp
    import io as _io
    import run_bnm_r6 as _R

    # ---- (a) GRID-MATCHING. Every 250-multiple keeps its full evaluation. 7.22 killed a
    # stability claim by changing the grid under a comparison; a second would be
    # self-inflicted.
    multiples = [s for s in range(250, 8001, 250)]
    check(all(_R.checkpoint_kind(s, max_steps=8000) == "full" for s in multiples),
          f"T19 all {len(multiples)} of the 250-multiples are still FULL evaluations")

    sub = [s for s in range(1, 8001) if _R.checkpoint_kind(s, max_steps=8000) == "stats"]
    check(all(s % _R.CKPT_EVERY != 0 for s in sub),
          "T19 no sub-grid point is a 250-multiple -- the two sets are disjoint")
    check(all(s <= _R.STATS_ONLY_UNTIL for s in sub),
          f"T19 the sub-grid stops at {_R.STATS_ONLY_UNTIL}")
    check(len(sub) == 18,
          f"T19 the sub-grid adds exactly 18 points, not ~20 (got {len(sub)}) -- "
          f"7.16 consumes read-off points, so the count is asserted, not assumed")
    check(_R.checkpoint_kind(8000, max_steps=8000) == "full"
          and _R.checkpoint_kind(137, max_steps=8000) is None,
          "T19 the final step is full, and an ordinary step is neither")

    # ---- (b) METRIC-BLIND BY CONSTRUCTION. The strongest form of this claim is that the
    # function cannot score because it never receives a scorer -- not that it chooses not
    # to. Assert the SIGNATURE first, then the body.
    params = set(_isp.signature(_R.stats_only_row).parameters)
    forbidden = {"scorer", "score", "evaluator", "metrics"}
    check(not (params & forbidden),
          f"T19 stats_only_row takes NO scorer -- it cannot evaluate even by mistake "
          f"(params: {sorted(params)})")

    # THE SHARED SCANNER (prereg-edits brief 6). Three scans in this project have fired
    # on their own documentation, each fixed where it fired. code_scan.scan ends that by
    # (a) tokenising, so prose is invisible, and (b) taking known_null_src as a REQUIRED
    # argument -- a scan whose control does not fire raises instead of returning "clean".
    from code_scan import KnownNullDidNotFire, dedent_block, scan as code_scan_scan

    src = _io.open(_R.__file__, encoding="utf-8").read()
    tree = _ast.parse(src)

    def fn_src(name):
        fn = next(n for n in tree.body
                  if isinstance(n, _ast.FunctionDef) and n.name == name)
        return dedent_block(_ast.get_source_segment(src, fn) or "")

    EVAL_TOKENS = ("scorer", "dump", "evaluate", "metric_suite", "tpr", "auroc",
                   "auprc", "test_pos", "test_neg")
    res = code_scan_scan(fn_src("stats_only_row"), EVAL_TOKENS,
                         known_null_src=fn_src("train_loop"))
    check(res["clean"],
          "T19 stats_only_row's CODE reaches no evaluation symbol"
          + ("" if res["clean"] else f": {res['hits']}"))
    check("scorer" in res["known_null_hits"],
          f"T19 KNOWN-NULL: the FULL path's code DOES match "
          f"{res['known_null_hits']} -- the scan bites")

    # the tokenising is what makes it correct, not luck: the RAW text is not clean
    check("scorer" in fn_src("stats_only_row").lower(),
          "T19 the RAW text of stats_only_row contains 'scorer' (in its docstring), so "
          "tokenising is doing real work here")

    # and the helper REFUSES a scan whose control cannot fire
    try:
        code_scan_scan("x = 1", ("scorer",), known_null_src="y = 2")
        check(False, "T19 code_scan.scan REFUSES a known-null that matches nothing")
    except KnownNullDidNotFire:
        check(True, "T19 code_scan.scan REFUSES a known-null that matches nothing")

    # ---- (c) the sub-grid's columns carry no metric
    cols = [c.lower() for c in _R.STATS_COLUMNS]
    metric_cols = [c for c in cols
                   if any(t in c for t in ("tpr", "auroc", "auprc", "logit_file",
                                           "purity", "completeness"))]
    check(not metric_cols,
          f"T19 STATS_COLUMNS holds no metric column"
          + ("" if not metric_cols else f": {metric_cols}"))
    for needed in ("bnm_nuclear_norm", "bnm_target_entropy", "bnm_pred_pos_frac",
                   "nuclear_gap", "entropy_gap"):
        check(needed in cols,
              f"T19 the sub-grid records {needed} (brief 2's required statistic)")


def t20_matched_arm_budget():
    print("\nT20 -- no_matched_arm_budget: a third NAMED outcome that does NOT refuse")

    ceil_ = 10.8413

    # THE INSTRUMENT IS SET MEMBERSHIP, NOT A FLOOR. 7.22 makes cross-arm contrasts at a
    # MATCHED STEP -- lambda=0 was read at 2000 for CORAL and 1500 for FreeMatch -- and
    # 7.16 showed the read-off point matters enormously. So 1600 is NOT comparable to
    # FreeMatch's 1500 merely because it exceeds it. That case is the one a floor got
    # wrong, so it is tested first.
    over_old_floor = B.nuclear_norm_crossing([800], [10.9], ceil_)     # budget 1600
    check(over_old_floor["budget"] == 1600
          and over_old_floor["outcome"] == "no_matched_arm_budget",
          "T20 budget 1600 is NOT comparable despite exceeding the old 1500 floor -- "
          "the case set membership gets right and a floor got wrong")

    # EXACT MATCHES on each declared arm budget are comparable
    for step, want in ((750, 1500), (1000, 2000), (2000, 4000)):
        r = B.nuclear_norm_crossing([step], [10.9], ceil_)
        check(r["budget"] == want and r["outcome"] == "normal"
              and r["cross_arm_comparable"] is True,
              f"T20 an exact match on {want} is comparable "
              f"({'FreeMatch' if want == 1500 else 'CORAL' if want == 2000 else 'lambda=0'})")

    check(tuple(B.MATCHED_ARM_BUDGETS) == (1500, 2000, 4000),
          "T20 the declared set is {1500, 2000, 4000} -- FreeMatch, CORAL, lambda=0/DANN")

    # the Stage-2 trajectory at the SELECTED gamma: crossing between 250 and 500
    r = B.nuclear_norm_crossing([250, 500, 750, 1000],
                                [10.6077, 10.8559, 10.7807, 10.7744], ceil_)
    check(r["outcome"] == "no_matched_arm_budget",
          f"T20 the selected arm's own trajectory lands on no_matched_arm_budget "
          f"(budget {r['budget']})")
    check(r["fires"] is True and r["budget"] == 1000,
          "T20 ...and it FIRES -- the budget is derived and used, not refused")
    check(r["cross_arm_comparable"] is False,
          "T20 ...but it is marked NOT cross-arm comparable")
    check("4000" in r["which_number_is_quoted"],
          "T20 ...and the artifact says matched readouts carry the contrasts, 4000 "
          "mandatory as the lambda=0 known-null's budget")
    check("7.18" in r["budget_confound_removed_by_a_match"],
          "T20 the artifact carries the disclosure that a match removes the BUDGET "
          "confound only -- head init and gamma-tuning budget still travel (7.18)")

    # the two REFUSALS are unchanged and stay distinguishable from the new outcome
    never = B.nuclear_norm_crossing([1000, 2000], [10.0, 10.2], ceil_)
    check(never["outcome"] == "never_crossed" and never["fires"] is False
          and never["budget"] is None,
          "T20 never_crossed still REFUSES (budget None)")
    over = B.nuclear_norm_crossing([5000], [10.9], ceil_)
    check(over["outcome"] == "exceeds_ceiling" and over["fires"] is False
          and over["budget"] is None and over["budget_before_ceiling"] == 10000,
          "T20 exceeds_ceiling still REFUSES, and is NOT clamped (raw 10000 preserved)")

    check(len({never["outcome"], over["outcome"], r["outcome"],
               over_old_floor["outcome"], "normal"}) == 4,
          "T20 all four outcomes are distinguishable in the artifact")




def t21_path2_never_rederives_the_grid():
    print("\nT21 -- PATH 2: a second measurement is recorded, and NEVER becomes the grid")
    import importlib as _il
    import json as _j
    import tempfile as _tf

    # The original T4/fp16 measurement's session was lost, so the swept grid's provenance is
    # unrecoverable. Path 2 re-measures. THE RISK IS THAT THE SECOND MEASUREMENT QUIETLY
    # BECOMES THE GRID'S SOURCE -- a lookalike substituted for the real provenance. These
    # checks exist because that substitution would be invisible in the finished artifact.
    W = _il.import_module("write_bnm_r6_prereg")

    check(os.path.basename(W.REMEASURED_GRADNORMS)
          != os.path.basename(W.VENUE_GRADNORMS),
          "T21 the re-measurement has its OWN filename, so it cannot be mistaken for the "
          "original by a later reader or by the writer")
    check("REMEASURED" in os.path.basename(W.REMEASURED_GRADNORMS),
          "T21 ...and the filename says what it is")

    # the material threshold is DECLARED, and declared before any re-measurement exists
    check(W.MATERIAL_RATIO_DELTA == 0.05,
          f"T21 the material threshold is declared as {W.MATERIAL_RATIO_DELTA} -- a "
          f"threshold chosen after seeing the number is not a threshold")
    # The threshold must be a MODULE-LEVEL LITERAL, not something derived at call time from
    # the measurement it judges. Checked against the source, because a value computed from
    # the re-measurement would still read as 0.05 at runtime on the day it happened to.
    import ast as _a
    import io as _i2
    _wsrc = _i2.open(W.__file__, encoding="utf-8").read()
    _lit = [n for n in _a.parse(_wsrc).body
            if isinstance(n, _a.Assign)
            and any(isinstance(t, _a.Name) and t.id == "MATERIAL_RATIO_DELTA"
                    for t in n.targets)
            and isinstance(n.value, _a.Constant)]
    check(len(_lit) == 1,
          "T21 MATERIAL_RATIO_DELTA is a module-level CONSTANT literal, not computed from "
          "the measurement it judges")

    back = sum(f / g for f, g in zip(W.FRACTIONS, W.SWEPT_GRID)) / len(W.SWEPT_GRID)

    def block(ratio):
        with _tf.TemporaryDirectory() as td:
            p = os.path.join(td, "re.json")
            _j.dump({"device": "cuda", "fp16": True, "gpu_name": "Tesla T4",
                     "ce_grad_norm": 6.0, "bnm_grad_norm_at_gamma_1": 6.0 * ratio,
                     "ratio_bnm_over_ce_at_gamma_1": ratio}, open(p, "w"))
            orig = W.REMEASURED_GRADNORMS
            W.REMEASURED_GRADNORMS = p
            try:
                return W._remeasurement_block(False, True, back)
            finally:
                W.REMEASURED_GRADNORMS = orig

    agree = block(back * 1.005)
    check(agree["is_material"] is False and abs(agree["delta_vs_swept_pct"] - 0.5) < 0.01,
          "T21 a re-measurement 0.5% from the back-computed ratio is NOT material")
    check("corroborates" in agree["interpretation"],
          "T21 ...and is described as corroborating, not as provenance")

    big = block(back * 1.12)
    check(big["is_material"] is True,
          "T21 KNOWN-NULL: a 12% move IS material -- the threshold can fire")
    check("7.7" in big["interpretation"],
          "T21 ...and a material move is named as a WITHIN-ARCHITECTURE 7.7 data point")

    for b in (agree, big):
        check("NOT the provenance" in b["WHAT_THIS_IS"],
              "T21 every re-measurement block says it is NOT the grid's provenance")
        check("0.00787" in b["WHAT_THE_SWEEP_ACTUALLY_USED"],
              "T21 ...and records that gamma 0.00787 was selected under the SWEPT grid")
        check("does not license" in b["what_this_does_NOT_license"],
              "T21 ...and states what it does not license")

    # the grid and gamma are IMMUTABLE across the whole path-2 arrangement
    check(W.SWEPT_GRID == [0.00787, 0.039351, 0.157406],
          "T21 the swept grid is the one that was swept, unchanged")




def t22_integrity_guard():
    print("\nT22 -- the protected set: an overwrite FAILS, and a change is CAUGHT")
    import json as _j
    import tempfile as _tf
    import integrity as _I

    # WHY. Running the Task-A detector for round 6 silently overwrote CORAL's analysis
    # file, and the repair moved coral_r4_budget.json's self-hash away from what 7.14
    # records. Both survived only because a backup happened to exist and the detector
    # happened to be deterministic. Neither is a control. Folds 1-4 run the same scripts
    # four more times, so the guard is tested, not trusted.
    #
    # WHAT RUNS WHERE. The protected set is a property of the ANALYSIS MACHINE: it is the
    # local runs/ tree checked against the local PROJECT.md. The training venue holds
    # neither -- PROJECT.md is not in the bundle, and the venue never writes a hashed
    # artifact. So the live-record checks are SKIPPED there, visibly, while the
    # guard-LOGIC checks (which use synthetic files) run everywhere. An earlier version
    # required the inventory unconditionally and took the venue suite down with a
    # FileNotFoundError before any GPU work started.
    have_inventory = os.path.isfile(_I.INVENTORY)
    have_project = os.path.isfile(_I.PROJECT_MD)

    if have_inventory and have_project:
        d = _j.load(open(_I.INVENTORY, encoding="utf-8"))
        check(d["n"] >= 5, f"T22 the inventory lists {d['n']} protected artifact(s)")
        flavours = {r["flavour"] for r in d["artifacts"]}
        check(flavours <= {"whole_file", "self_certifying"},
              f"T22 both hash flavours are handled {sorted(flavours)} -- checking a "
              f"self-certifying file as whole-file would report a false failure")
        _I.verify(quiet=True)
        check(True, "T22 every protected artifact currently matches PROJECT.md")
    else:
        # The record's NAME is asked of integrity.py rather than repeated here. A second
        # hard-coded filename is how the guard went stale the first time, and a skip
        # message naming the wrong file is what made the staleness look like a venue.
        absent = ("runs/protected_artifacts.json" if not have_inventory
                  else os.path.basename(_I.PROJECT_MD))
        skip("T22 the LIVE protected set verifies against PROJECT.md",
             f"{absent} is not present -- this is the training venue, which holds "
             f"neither the record nor the artifacts it protects, and never writes one")

    # ---- GUARD LOGIC. Synthetic files only, so this runs everywhere. ----
    saved = _I.INVENTORY
    try:
        with _tf.TemporaryDirectory() as td:
            victim = os.path.join(td, "fake_artifact.json")
            _j.dump({"budget": 1000}, open(victim, "w"))
            good = _I.whole_file_sha(victim)
            rel = os.path.relpath(victim, _I.REPO).replace("\\", "/")
            ip = os.path.join(td, "inv.json")
            _j.dump({"n": 1, "artifacts": [{"path": rel, "flavour": "whole_file",
                                            "sha256": good}]}, open(ip, "w"))
            md = os.path.join(td, "FAKE_PROJECT.md")
            open(md, "w").write(f"hashes to `{good}`\n")
            _I.INVENTORY = ip

            _I.verify(md_path=md, quiet=True)
            check(True, "T22 KNOWN-NULL control: an unchanged artifact PASSES")

            _j.dump({"budget": 4000}, open(victim, "w"))   # a quiet, plausible corruption
            try:
                _I.verify(md_path=md, quiet=True)
                check(False, "T22 KNOWN-NULL: a CHANGED artifact is refused")
            except SystemExit as e:
                check("CHANGED" in str(e),
                      "T22 KNOWN-NULL: a CHANGED artifact is refused, naming the file")

            os.remove(victim)
            try:
                _I.verify(md_path=md, quiet=True)
                check(False, "T22 KNOWN-NULL: an ABSENT artifact is refused")
            except SystemExit as e:
                check("ABSENT" in str(e),
                      "T22 KNOWN-NULL: an ABSENT artifact is refused")
    finally:
        _I.INVENTORY = saved

    # the analysis scripts actually CALL it, at start AND end -- source-only, runs anywhere
    import ast as _a
    import io as _i
    for name, n_expected in (("analyze_coral_dev8000_budget.py", 2),
                             ("write_bnm_r6_budget.py", 2)):
        p = os.path.join(os.path.dirname(os.path.abspath(_I.__file__)), name)
        if not os.path.isfile(p):
            skip(f"T22 {name} calls the check at start and end",
                 "the script is not staged here")
            continue
        src = _i.open(p, encoding="utf-8").read()
        calls = [n for n in _a.walk(_a.parse(src))
                 if isinstance(n, _a.Call) and isinstance(n.func, _a.Name)
                 and n.func.id == "_verify_integrity"]
        check(len(calls) == n_expected,
              f"T22 {name} calls the check {len(calls)}x (start and end)")

    # and no repo-anchored write escapes --out any more
    from audit_write_paths import audit_file
    scripts_dir = os.path.dirname(os.path.abspath(_I.__file__))
    anchored = []
    for name in sorted(f for f in os.listdir(scripts_dir) if f.endswith(".py")):
        for ln, seg in audit_file(os.path.join(scripts_dir, name)):
            if any(t in seg for t in ("REPO", '"runs"', "'runs'", "runs/", '".."')):
                anchored.append(f"{name}:{ln}")
    check(not anchored,
          "T22 no script writes to a repo-anchored path that --out does not control"
          + ("" if not anchored else f": {anchored}"))


def main() -> int:
    print("=" * 74)
    print("ROUND 6 -- BNM installer and rules (CPU, no GPU)")
    print("=" * 74)
    t1_source_not_in_loss()
    t2_t3_hook_shape()
    t4_t5_install()
    t6_collapse_band()
    t7_budget_rule()
    t8_anchors_from_artifact()
    t9_grid_scaling()
    t10_gamma_invariant()
    t11_measure_step0_is_metric_blind()
    t12_grid_reads_the_file()
    t13_size_matched_probe()
    t14_selection_and_extension()
    t15_call_sites_bind()
    t16_fp16_entropy_is_finite()
    t17_probe_path_actually_runs()
    t18_no_undefined_names()
    t19_hybrid_cadence_is_metric_blind()
    t20_matched_arm_budget()
    t21_path2_never_rederives_the_grid()
    t22_integrity_guard()

    print("\n" + "=" * 74)
    if _skips:
        print(f"SKIPPED ({len(_skips)}) -- not runnable in this environment:")
        for m in _skips:
            print("   - " + m)
        print()
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL BNM ROUND-6 WIRING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
