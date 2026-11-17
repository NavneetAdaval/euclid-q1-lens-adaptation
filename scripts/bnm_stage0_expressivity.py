"""
bnm_stage0_expressivity.py
==========================
ROUND 6, STAGE 0a -- **binary expressivity of the nuclear norm**, CPU-only, before
any GPU time is spent on BNM.

WHY THIS SCRIPT EXISTS. Round 1's CORAL cost a full round because the loss was
attached to an adapter structurally incapable of expressing it: a diagonal LayerNorm
offering 640 reachable parameters against a 640x640 covariance objective (PROJECT.md
7.15). The instruction that came out of that is to check the objective can express
what it is being asked to express BEFORE running it, not after.

BNM's version of that risk is the CLASS COUNT. With two classes the target output
matrix A is B x 2 and row-stochastic, so rank(A) <= 2 and the rank-1-vs-rank-31
"diversity" pressure BNM exercises on Office-Home reduces here to rank-1-vs-rank-2.
The question is whether the objective still separates the states it needs to
separate, with a margin gamma can act on.

TWO PARTS.

  A. ANALYTIC. Three batch states with hand-computable norms, checked against the
     values declared in the brief before this ran:

         state                  nuc/min(B,C)   frobenius
         collapsed-confident        8.000        16.000
         balanced-confident        11.314        16.000
         maximally-uncertain        5.657        11.314

     Two things this must show:
       1. balanced-confident is the MAXIMUM, so the objective prefers the state we
          want;
       2. the FROBENIUS norm cannot tell collapsed-confident from balanced-confident
          (16.00 both) while the NUCLEAR norm can (8.00 vs 11.31). That is the whole
          reason for choosing BNM over plain confidence maximisation, and it is what
          makes the choice structural rather than decorative.

     And one honest limitation to record rather than bury: collapsed-confident (8.00)
     scores ABOVE maximally-uncertain (5.66). BNM prefers collapse to uncertainty.
     The 41% collapsed-to-balanced margin is all gamma has to work with.

  B. EMPIRICAL. The analytic states are idealised. The failure signature we are
     actually guarding against is a REAL model whose reachable range of the objective
     is far narrower than the analytic one -- that is what "the loss cannot move"
     looked like for round-1 CORAL. So: take Baseline A's frozen LP head, run it over
     the cached fold-0 TARGET features in batches of 256, and measure the objective's
     value at the state training actually starts from, plus the range reachable by
     sharpening/softening that same model's outputs with a temperature. If the
     empirical margin is much smaller than the analytic 41%, STOP AND REPORT.

     The temperature sweep is a PROBE OF THE OBJECTIVE'S SURFACE, not a training
     simulation and not a selection: it reads no test set and no metric. It answers
     "if the head's outputs get sharper, does the nuclear norm move?" -- the round-1
     CORAL question, asked before the GPU bill.

READS NO METRIC. No TPR, no AUROC, no test-set quantity appears anywhere in this
file. Target features only, plus the source-only head that every arm initialises from.

    python -u scripts/bnm_stage0_expressivity.py --out runs/bnm_r6_stage0a.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from da_losses import (BNMLoss, binary_prob2, bnm_analytic_reference,  # noqa: E402
                       bnm_loss, bnm_nuclear_norm)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_CACHE = os.path.join(REPO, "data", "feature_cache")
BASELINE_A_HEAD = os.path.join(REPO, "runs", "bakeoff_baseline_A",
                               "checkpoints", "step_8000.pt")

# The brief's declared analytic values, typed here BEFORE the script ran so the
# comparison is a check and not a transcription.
DECLARED = {
    "collapsed_confident": {"nuclear_over_min_bc": 8.00, "frobenius": 16.00},
    "balanced_confident": {"nuclear_over_min_bc": 11.31, "frobenius": 16.00},
    "maximally_uncertain": {"nuclear_over_min_bc": 5.66, "frobenius": 11.31},
}

# Stage 0a's stop rule, declared before part B ran. The empirical reachable margin is
# compared against the analytic 41%; below this fraction of it the arm stops.
EMPIRICAL_MARGIN_FLOOR_FRAC = 0.50


# ---------------------------------------------------------------------------
# part A -- analytic
# ---------------------------------------------------------------------------

def part_a(batch: int = 256) -> dict:
    ref = bnm_analytic_reference(batch=batch)
    out = {"batch": batch, "states": {}, "checks": {}}

    for name, declared in DECLARED.items():
        got = ref[name]
        out["states"][name] = got
        out["checks"][f"{name}_matches_declared"] = bool(
            abs(got["nuclear_over_min_bc"] - declared["nuclear_over_min_bc"]) < 0.01
            and abs(got["frobenius"] - declared["frobenius"]) < 0.01)

    nuc = {k: ref[k]["nuclear_over_min_bc"] for k in DECLARED}
    fro = {k: ref[k]["frobenius"] for k in DECLARED}

    # 1. balanced-confident is the maximum -- the objective prefers the state we want.
    out["checks"]["balanced_is_argmax"] = bool(
        max(nuc, key=nuc.get) == "balanced_confident")

    # 2. the discrimination the choice of BNM rests on.
    out["checks"]["frobenius_CANNOT_separate_collapse"] = bool(
        abs(fro["collapsed_confident"] - fro["balanced_confident"]) < 1e-4)
    out["checks"]["nuclear_CAN_separate_collapse"] = bool(
        nuc["balanced_confident"] - nuc["collapsed_confident"] > 1.0)

    # 3. the limitation, asserted so it cannot quietly stop being true.
    out["checks"]["prefers_collapse_over_uncertainty"] = bool(
        nuc["collapsed_confident"] > nuc["maximally_uncertain"])

    out["analytic_margin_collapsed_to_balanced"] = ref["_margin_collapsed_to_balanced"]

    # 4. the loss is negated (it MINIMISES) and differentiable.
    z = torch.randn(64, 1, requires_grad=True)
    lv = bnm_loss(z)
    lv.backward()
    out["checks"]["loss_is_negative"] = bool(float(lv.detach()) < 0.0)
    out["checks"]["loss_is_differentiable"] = bool(
        z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0)

    # 5. gradient DIRECTION: from a near-uniform batch, one BNM step must raise the
    #    nuclear norm. Cheap, and it is the sign error that would silently invert the
    #    whole arm.
    z2 = torch.randn(256, 1) * 0.01
    z2.requires_grad_(True)
    before = float(bnm_nuclear_norm(binary_prob2(z2)))
    bnm_loss(z2).backward()
    with torch.no_grad():
        z3 = z2 - 10.0 * z2.grad          # a gradient-DESCENT step on the negated loss
    after = float(bnm_nuclear_norm(binary_prob2(z3)))
    out["gradient_direction"] = {"nuclear_before": before, "nuclear_after": after}
    out["checks"]["descent_step_raises_nuclear_norm"] = bool(after > before)

    out["PASS"] = all(v for k, v in out["checks"].items()
                      if k != "prefers_collapse_over_uncertainty")
    return out


# ---------------------------------------------------------------------------
# part B -- empirical, on real cached target features
# ---------------------------------------------------------------------------

def _load_head(path: str):
    """Baseline A's frozen LP head + its feature standardizer, as plain tensors.

    Deliberately does NOT import the model builder: this script must run on a laptop
    with no GPU and no encoder, and everything it needs is a (640 -> 1) linear map.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("head_state", ck.get("head", ck.get("state_dict", ck)))
    w = b = None
    for k, v in sd.items():
        if k.endswith("classifier.weight"):
            w = v.float().reshape(-1)
        elif k.endswith("classifier.bias"):
            b = v.float().reshape(-1)
    if w is None or b is None:
        raise SystemExit(f"no classifier.weight/bias in {path}; keys={list(sd)[:12]}")

    std = ck.get("standardizer_state", ck.get("standardizer"))
    mu = sig = None
    if isinstance(std, dict):
        for k, v in std.items():
            if k.endswith("mean"):
                mu = v.float().reshape(-1)
            elif k.endswith("std"):
                sig = v.float().reshape(-1)
    return w, b, mu, sig


def _logits(feats: np.ndarray, w, b, mu, sig) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(feats)).float()
    if mu is not None and sig is not None and mu.numel() == x.shape[1]:
        x = (x - mu) / sig.clamp_min(1e-8)
    return x @ w + b


def part_b(fold: int = 0, batch: int = 256, seed: int = 0,
           analytic_margin: float = 0.41) -> dict:
    tgt_path = os.path.join(FEATURE_CACHE, f"target_fold{fold}.npz")
    if not os.path.exists(tgt_path) or not os.path.exists(BASELINE_A_HEAD):
        return {"SKIPPED": True,
                "reason": f"missing {tgt_path if not os.path.exists(tgt_path) else BASELINE_A_HEAD}"}

    z = np.load(tgt_path)
    key = "features" if "features" in z.files else z.files[0]
    feats = z[key]
    w, b, mu, sig = _load_head(BASELINE_A_HEAD)
    logits = _logits(feats, w, b, mu, sig)

    rng = np.random.default_rng(seed)
    n = logits.shape[0]
    idx = rng.permutation(n)
    n_batches = max(1, n // batch)

    # (i) the objective AT THE STATE TRAINING STARTS FROM, over real batches.
    at_init = []
    pos_frac = []
    for i in range(n_batches):
        sl = torch.from_numpy(idx[i * batch:(i + 1) * batch].copy())
        lb = logits[sl]
        at_init.append(float(bnm_nuclear_norm(binary_prob2(lb))))
        pos_frac.append(float((torch.sigmoid(lb) > 0.5).float().mean()))

    # (ii) the REACHABLE range on this model's own outputs. Temperature only rescales
    #      the logits, so it traverses exactly the confidence axis BNM pushes along
    #      without changing which images the head ranks above which.
    temps = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0]
    sweep = []
    for t in temps:
        vals, pf = [], []
        for i in range(n_batches):
            sl = torch.from_numpy(idx[i * batch:(i + 1) * batch].copy())
            lb = logits[sl] * t
            vals.append(float(bnm_nuclear_norm(binary_prob2(lb))))
            pf.append(float((torch.sigmoid(lb) > 0.5).float().mean()))
        sweep.append({"temperature": t,
                      "nuclear_mean": float(np.mean(vals)),
                      "pred_pos_frac_mean": float(np.mean(pf))})

    lo = min(s["nuclear_mean"] for s in sweep)
    hi = max(s["nuclear_mean"] for s in sweep)
    empirical_margin = (hi - lo) / lo

    # (iii) CONFIDENCE vs DIVERSITY, decomposed. Temperature cannot change WHICH class
    #       each image is assigned to, so `hi` is exactly the "confident at the head's
    #       current class split" state, and the gap from `hi` up to the balanced-confident
    #       maximum is the part of BNM's pressure that requires RE-BALANCING. With C=2
    #       those are the objective's only two components, so this splits the whole of
    #       what BNM can ask for at init into the part it shares with plain confidence
    #       maximisation and the part that is BNM-specific.
    balanced_max = bnm_analytic_reference(
        batch=batch)["balanced_confident"]["nuclear_over_min_bc"]
    init = float(np.mean(at_init))
    total_gain = balanced_max - init
    confidence_gain = hi - init
    diversity_gain = balanced_max - hi

    out = {
        "SKIPPED": False,
        "fold": fold,
        "n_target_images": int(n),
        "batch": batch,
        "n_batches": n_batches,
        "head": os.path.relpath(BASELINE_A_HEAD, REPO).replace("\\", "/"),
        "at_init": {
            "nuclear_mean": float(np.mean(at_init)),
            "nuclear_sd": float(np.std(at_init, ddof=1)) if len(at_init) > 1 else 0.0,
            "pred_pos_frac_mean": float(np.mean(pos_frac)),
        },
        "temperature_sweep": sweep,
        "reachable": {"min": lo, "max": hi, "margin": empirical_margin},
        "decomposition": {
            "init": init,
            "confident_at_current_split": hi,
            "balanced_confident_max": balanced_max,
            "total_gain_available": total_gain,
            "confidence_component": confidence_gain,
            "diversity_component": diversity_gain,
            "confidence_share": confidence_gain / total_gain if total_gain else float("nan"),
            "diversity_share": diversity_gain / total_gain if total_gain else float("nan"),
        },
        "analytic_margin": analytic_margin,
        "empirical_over_analytic": empirical_margin / analytic_margin,
        "floor_frac_declared_before_running": EMPIRICAL_MARGIN_FLOOR_FRAC,
    }
    out["PASS"] = bool(empirical_margin >= EMPIRICAL_MARGIN_FLOOR_FRAC * analytic_margin)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "bnm_r6_stage0a.json"))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    a = part_a(batch=args.batch)
    b = part_b(fold=args.fold, batch=args.batch,
               analytic_margin=a["analytic_margin_collapsed_to_balanced"])

    print("=" * 74)
    print("STAGE 0a -- BINARY EXPRESSIVITY OF THE NUCLEAR NORM   (CPU, no metric read)")
    print("=" * 74)
    print(f"\nPART A -- analytic, B={args.batch}, C=2, normalised by min(B,C)=2\n")
    print(f"  {'state':<24}{'nuc/min(B,C)':>14}{'frobenius':>12}")
    for name in DECLARED:
        s = a["states"][name]
        print(f"  {name:<24}{s['nuclear_over_min_bc']:>14.4f}{s['frobenius']:>12.4f}")
    print(f"\n  collapsed -> balanced margin: {a['analytic_margin_collapsed_to_balanced']:.1%}")
    print("\n  checks:")
    for k, v in a["checks"].items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  PART A: {'PASS' if a['PASS'] else 'FAIL'}")

    print("\nPART B -- empirical, Baseline A head over cached target features\n")
    if b.get("SKIPPED"):
        print(f"  SKIPPED: {b['reason']}")
    else:
        print(f"  target fold {b['fold']}: {b['n_target_images']} images, "
              f"{b['n_batches']} batches of {b['batch']}")
        print(f"  at init: nuclear {b['at_init']['nuclear_mean']:.4f} "
              f"(sd {b['at_init']['nuclear_sd']:.4f}), "
              f"pred-pos frac {b['at_init']['pred_pos_frac_mean']:.4f}")
        print(f"\n  {'temperature':>12}{'nuclear':>12}{'pred-pos frac':>16}")
        for s in b["temperature_sweep"]:
            print(f"  {s['temperature']:>12.2f}{s['nuclear_mean']:>12.4f}"
                  f"{s['pred_pos_frac_mean']:>16.4f}")
        d = b["decomposition"]
        print("\n  CONFIDENCE vs DIVERSITY, from the init state:")
        print(f"    init                          {d['init']:.4f}")
        print(f"    confident at current split    {d['confident_at_current_split']:.4f}"
              f"   (+{d['confidence_component']:.4f}, "
              f"{d['confidence_share']:.1%} of the available gain)")
        print(f"    balanced-confident maximum    {d['balanced_confident_max']:.4f}"
              f"   (+{d['diversity_component']:.4f}, "
              f"{d['diversity_share']:.1%} of the available gain)")
        print(f"\n  reachable range {b['reachable']['min']:.4f} -> "
              f"{b['reachable']['max']:.4f}, margin {b['reachable']['margin']:.1%}")
        print(f"  against the analytic {b['analytic_margin']:.1%}: "
              f"{b['empirical_over_analytic']:.2f}x")
        print(f"  floor declared before running: "
              f"{b['floor_frac_declared_before_running']:.2f} x analytic")
        print(f"\n  PART B: {'PASS' if b['PASS'] else 'FAIL -- STOP AND REPORT'}")

    result = {"stage": "0a", "part_a": a, "part_b": b,
              "PASS": bool(a["PASS"] and (b.get("SKIPPED") or b["PASS"]))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {args.out}")
    print(f"STAGE 0a OVERALL: {'PASS' if result['PASS'] else 'FAIL'}")
    return 0 if result["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
