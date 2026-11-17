"""
test_da_bnm.py
==============
Tests for ROUND 6 -- **BNM as the minimal-target-signal control**: ``da_losses``
(``bnm_nuclear_norm`` / ``bnm_loss`` / ``bnm_analytic_reference`` / ``BNMLoss``).

Pure tensors. No data, no encoder, no GPU -- by design, because Stage 0a's whole
purpose is to be answerable before any GPU time is spent (PROJECT.md 7.20).

  T1  ANALYTIC VALUES -- the three batch states against hand-computed numbers.
      THIS IS THE POINT OF THE SUITE. With C=2 the target matrix has rank <= 2, so
      BNM's multi-class "diversity" pressure reduces to rank-1-vs-rank-2 here, and
      whether the objective still separates the states it must separate is a
      question that has to be ANSWERED rather than assumed. Round 1's CORAL was
      attached to an adapter structurally incapable of expressing it (7.15); this is
      the check that the same mistake is not being repeated.

  T2  balanced-confident is the ARGMAX -- the objective prefers the state we want.

  T3  THE STRUCTURAL JUSTIFICATION FOR CHOOSING BNM. The Frobenius norm CANNOT
      distinguish collapsed-confident from balanced-confident (16.00 both); the
      nuclear norm CAN (8.00 vs 11.31). Without this, BNM is plain confidence
      maximisation with extra arithmetic.

  T4  THE LIMITATION, asserted so it cannot quietly stop being true: collapsed
      (8.00) scores ABOVE maximally-uncertain (5.66). BNM prefers collapse to
      uncertainty; the 41% collapsed->balanced margin is all gamma has.

  T5  the loss is NEGATED (minimising it maximises the nuclear norm).
  T6  it is differentiable with finite, non-zero gradients.
  T7  GRADIENT DIRECTION -- one descent step from a near-uniform batch RAISES the
      nuclear norm. This is the sign error that would silently invert the arm and
      still train.
  T8  row-order invariance (singular values do not care about batch order).
  T9  TARGET-ONLY -- forward takes target logits and nothing else, and rejects the
      multi-column logits that would mean it was wired to a C-way head.
  T10 STATELESS -- no parameters, no buffers, no set_step. Structurally exempt from
      7.10's weights-only-continuation problem and from 7.18's budget problem: there
      is no schedule here to be positioned wrongly on resume.
  T11 NO RAMP -- BNMLoss exposes no horizon, no step counter and no total_steps.
      7.18 spent a round on DANN's lambda horizon; the minimal control must have no
      analogue, and this test is what stops one being added later by accident.
  T12 the COLLAPSE MONITOR -- last_stats reports predicted-positive fraction, and it
      is right. It is the quantity BNM has no explicit term protecting.

    python -u scripts/test_da_bnm.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from da_losses import (BNMLoss, binary_prob2, bnm_analytic_reference,  # noqa: E402
                       bnm_loss, bnm_nuclear_norm)

_fails: list[str] = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        _fails.append(msg)


def near(a, b, tol=1e-3):
    return abs(float(a) - float(b)) < tol


# ---------------------------------------------------------------------------

def t1_t4_analytic():
    print("\nT1-T4 -- analytic batch states, B=256, C=2, normalised by min(B,C)=2")
    ref = bnm_analytic_reference(batch=256)

    # T1 -- the numbers themselves.
    check(near(ref["collapsed_confident"]["nuclear_over_min_bc"], 8.0000, 1e-3),
          "T1 collapsed-confident nuclear/2 == 8.0000")
    check(near(ref["collapsed_confident"]["frobenius"], 16.0000, 1e-3),
          "T1 collapsed-confident frobenius == 16.0000")
    check(near(ref["balanced_confident"]["nuclear_over_min_bc"], 11.3137, 1e-3),
          "T1 balanced-confident nuclear/2 == 11.3137")
    check(near(ref["balanced_confident"]["frobenius"], 16.0000, 1e-3),
          "T1 balanced-confident frobenius == 16.0000")
    check(near(ref["maximally_uncertain"]["nuclear_over_min_bc"], 5.6569, 1e-3),
          "T1 maximally-uncertain nuclear/2 == 5.6569")
    check(near(ref["maximally_uncertain"]["frobenius"], 11.3137, 1e-3),
          "T1 maximally-uncertain frobenius == 11.3137")

    nuc = {k: ref[k]["nuclear_over_min_bc"]
           for k in ("collapsed_confident", "balanced_confident", "maximally_uncertain")}
    fro = {k: ref[k]["frobenius"]
           for k in ("collapsed_confident", "balanced_confident", "maximally_uncertain")}

    # T2 -- the objective prefers the state we want.
    check(max(nuc, key=nuc.get) == "balanced_confident",
          "T2 balanced-confident is the argmax over the three states")

    # T3 -- the reason BNM is chosen over plain confidence maximisation.
    check(near(fro["collapsed_confident"], fro["balanced_confident"], 1e-4),
          "T3 FROBENIUS cannot separate collapsed from balanced (16.00 == 16.00)")
    check(nuc["balanced_confident"] - nuc["collapsed_confident"] > 3.0,
          "T3 NUCLEAR separates them by > 3 (11.31 vs 8.00)")

    # T4 -- the limitation, recorded as an assertion.
    check(nuc["collapsed_confident"] > nuc["maximally_uncertain"],
          "T4 BNM prefers collapse (8.00) to uncertainty (5.66) -- the known limitation")
    check(near(ref["_margin_collapsed_to_balanced"], 0.4142, 1e-3),
          "T4 collapsed->balanced margin is 41.4%, which is what gamma has to work with")


def t5_t7_gradients():
    print("\nT5-T7 -- sign, differentiability and gradient DIRECTION")

    z = torch.randn(64, 1, requires_grad=True)
    lv = bnm_loss(z)
    check(float(lv.detach()) < 0.0, "T5 bnm_loss returns a NEGATIVE scalar (it minimises)")

    lv.backward()
    check(z.grad is not None and torch.isfinite(z.grad).all() and float(z.grad.abs().sum()) > 0,
          "T6 gradients are finite and non-zero")

    # T7 -- from a near-uniform batch, a DESCENT step on the negated loss must RAISE
    # the nuclear norm. A sign error here trains happily and inverts the arm.
    torch.manual_seed(0)
    z2 = (torch.randn(256, 1) * 0.01).requires_grad_(True)
    before = float(bnm_nuclear_norm(binary_prob2(z2)).detach())
    bnm_loss(z2).backward()
    with torch.no_grad():
        z3 = z2 - 10.0 * z2.grad
    after = float(bnm_nuclear_norm(binary_prob2(z3)))
    check(after > before,
          f"T7 one descent step raises the nuclear norm ({before:.4f} -> {after:.4f})")


def t8_invariance():
    print("\nT8 -- row-order invariance")
    torch.manual_seed(1)
    z = torch.randn(128, 1)
    a = float(bnm_nuclear_norm(binary_prob2(z)))
    perm = torch.randperm(128)
    b = float(bnm_nuclear_norm(binary_prob2(z[perm])))
    check(near(a, b, 1e-5), f"T8 nuclear norm is permutation-invariant ({a:.6f} vs {b:.6f})")


def t9_target_only():
    print("\nT9 -- TARGET-ONLY call contract")
    loss = BNMLoss(gamma=1.0)
    loss.train()

    import inspect
    sig = inspect.signature(loss.forward)
    check(len(sig.parameters) == 1,
          f"T9 forward takes exactly ONE argument (target logits), got {list(sig.parameters)}")

    z = torch.randn(32, 1, requires_grad=True)
    v = loss(z)
    check(torch.isfinite(v).all(), "T9 accepts (B,1) target logits")

    v2 = loss(z.reshape(-1))
    check(near(float(v.detach()), float(v2.detach()), 1e-6),
          "T9 accepts (B,) target logits and agrees with (B,1)")

    bad = torch.randn(32, 2, requires_grad=True)
    try:
        loss(bad)
        check(False, "T9 REJECTS multi-column logits (wired to a C-way head)")
    except ValueError:
        check(True, "T9 REJECTS multi-column logits (wired to a C-way head)")


def t10_t11_stateless_no_ramp():
    print("\nT10-T11 -- stateless, parameter-free, and NO RAMP")
    loss = BNMLoss(gamma=4.0)

    check(len(list(loss.parameters())) == 0,
          "T10 BNMLoss owns NO trainable parameters (unlike DannLoss)")
    check(len(list(loss.buffers())) == 0,
          "T10 BNMLoss owns NO buffers (unlike FreeMatch's SAT EMAs)")

    loss.train()
    torch.manual_seed(2)
    z = torch.randn(64, 1, requires_grad=True)
    a = float(loss(z).detach())
    b = float(loss(z).detach())
    check(near(a, b, 1e-9),
          "T10 two identical calls give identical values -- no hidden step state")

    for attr in ("set_step", "lambda_at", "total_steps", "horizon", "step",
                 "warmup", "ramp", "schedule"):
        check(not hasattr(loss, attr),
              f"T11 BNMLoss exposes no `{attr}` -- there is no horizon to position on")


def t12_collapse_monitor():
    print("\nT12 -- the collapse monitor")
    loss = BNMLoss()
    loss.train()

    # 24 confidently positive, 40 confidently negative -> pred-pos fraction 0.375.
    z = torch.cat([torch.full((24, 1), 8.0), torch.full((40, 1), -8.0)])
    z.requires_grad_(True)
    loss(z)
    s = loss.last_stats

    for k in ("bnm_nuclear_norm", "bnm_target_entropy", "bnm_pred_pos_frac",
              "bnm_mean_p_pos"):
        check(k in s, f"T12 last_stats reports `{k}`")
    check(near(s["bnm_pred_pos_frac"], 24 / 64, 1e-6),
          f"T12 pred-pos fraction is correct ({s['bnm_pred_pos_frac']:.4f} == 0.3750)")
    check(s["bnm_target_entropy"] < 0.01,
          f"T12 entropy is near zero on a confident batch ({s['bnm_target_entropy']:.2e})")

    # A fully collapsed batch: the monitor must see 1.0 while the objective still
    # returns a perfectly respectable-looking value. That gap IS the reason the
    # monitor exists.
    zc = torch.full((64, 1), 8.0, requires_grad=True)
    loss(zc)
    check(near(loss.last_stats["bnm_pred_pos_frac"], 1.0, 1e-6),
          "T12 a fully collapsed batch reads pred-pos fraction 1.0")
    check(loss.last_stats["bnm_nuclear_norm"] > 3.9,
          f"T12 ...while the objective still reads "
          f"{loss.last_stats['bnm_nuclear_norm']:.4f} -- collapse is not self-announcing")


def main() -> int:
    print("=" * 72)
    print("ROUND 6 -- BNM loss tests (CPU, pure tensors)")
    print("=" * 72)
    t1_t4_analytic()
    t5_t7_gradients()
    t8_invariance()
    t9_target_only()
    t10_t11_stateless_no_ramp()
    t12_collapse_monitor()

    print("\n" + "=" * 72)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL BNM CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
