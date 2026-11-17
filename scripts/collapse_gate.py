"""
collapse_gate.py
================
POST-HOC collapse-gate VALIDATOR (Job 1) for the label-free sim-to-real strong-lens
bake-off (PROJECT.md).  See spec_collapse_gate.md.

WHAT THIS IS
------------
The matched 5-fold bake-off selects checkpoints with R1 = "last checkpoint", frozen
before the run.  R1 is only unsafe if a method COLLAPSES late in training -- a sharp
break, not gentle drift.  This script reads a finished run's saved checkpoints, builds
the label-free AGREEMENT curve against Baseline A, and reports whether (and where) the
run broke.  If it fires at step S, selection falls back to the last checkpoint before S.

WHAT THIS IS **NOT**
--------------------
NOT a live early-stopper (Job 2).  It never halts training.  A false halt is
unrecoverable, and the firing thresholds cannot be calibrated until we have watched at
least one REAL collapse curve -- which does not exist on disk yet.  Everything below
therefore reports raw curve statistics plus fire/no-fire at two or three TRANSPARENT,
UNCALIBRATED settings; the operating point is Navneet's later decision, made against
real CORAL collapse data.  [EC]

THE SIGNAL  (spec 1)
--------------------
    agreement(s) = Spearman( DA_ckpt_s scores on fold-k TARGET pool ,
                             BaselineA_final scores on the same pool )

  * Pool = the fold-k target pool (~391 unlabeled real lenses + ~391 matched negatives,
    PROJECT.md 4.3).  UNLABELED -> the gate is fully label-free.  [RB]
  * Spearman (rank) because the headline metric TPR@FPR=1e-3 depends only on score
    RANKING, so the gate should watch ranking too.  [EC]  TTYD (arXiv:2409.04409) used
    hard-label agreement / KL / L1 / L2 and found the divergence choice did not much
    matter; Spearman is our ranking-appropriate adaptation and is [UNVERIFIED] as the
    best choice.
  * DIRECTION -- agreement is a FLOOR ALARM, never a maximize target.  Maximum agreement
    means BEING Baseline A, i.e. zero adaptation; TTYD found a reference too close to the
    source-only model always selects under-trained checkpoints, and Baseline A *is* the
    source-only model.  The gate only ever looks for a DOWNWARD break.  [RB]
  * Expected HEALTHY shape: starts near 1.0, declines gradually as adaptation legitimately
    pulls away, then levels off.  A gentle decline to a floor is HEALTHY.  The gate must
    fire only on a CLIFF (fast, persistent local drop) or a SUSTAINED SLIDE with no floor.

REUSE (spec 8): scoring via evaluate.load_checkpoint / evaluate.score_features /
evaluate.FeatureCache; Spearman via analyze_fold0_selection._spearman.  No new math.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import FeatureCache, load_checkpoint, score_features   # noqa: E402
from analyze_fold0_selection import _spearman                        # noqa: E402

try:
    import pandas as pd
except Exception:                                                    # pragma: no cover
    pd = None


# ---------------------------------------------------------------------------
# defaults / shorthands
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

DEFAULT_CACHE = os.path.join(_ROOT, "data", "feature_cache")
# Baseline A's FINAL (R1) checkpoint -- the reported 5-fold Baseline A (PROJECT.md 7.4,
# pooled TPR 0.184).  Source-only, so it is fold-independent by construction.
DEFAULT_REF = os.path.join(_ROOT, "runs", "baseline_A", "checkpoints",
                           "baseline_linear_ep100.pt")

SHORTHAND = {
    "baseline": os.path.join(_ROOT, "runs", "curve_baseline_fold0"),
    "dann":     os.path.join(_ROOT, "runs", "curve_dann_fold0"),
    "freematch": os.path.join(_ROOT, "runs", "curve_freematch_fold0"),
}

# Two TRANSPARENT, UNCALIBRATED settings (spec 4).  These are NOT tuned thresholds.
# W = window in gradient steps over which a drop counts as "sudden".
# M = agreement drop within W that counts as a cliff.
# F = absolute agreement level whose sustained breach counts as a slide.
SETTINGS = {
    "strict": dict(W=1000, M=0.15, F=0.50),
    "loose":  dict(W=1000, M=0.30, F=0.20),
}


# ---------------------------------------------------------------------------
# G1 -- agreement-curve computation
# ---------------------------------------------------------------------------

def list_step_checkpoints(run_dir: str):
    """[(step, path), ...] for checkpoints/step_<N>.pt, ascending by step."""
    out = []
    for p in glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pt")):
        m = re.search(r"step_(\d+)\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def reference_scores(ref_ckpt: str, cache: FeatureCache, split: str, device="cpu"):
    """Score the target pool ONCE through Baseline A -- the fixed reference vector."""
    head, std, _ = load_checkpoint(ref_ckpt, device)
    feats = cache.load(split)["features"]
    return score_features(head, std, feats, device)


def agreement_curve_from_csv(run_dir: str, col: str = None, verbose: bool = True):
    """Read the agreement column that was LOGGED INLINE during training.

    ****  REQUIRED for a partially-unfrozen encoder (Lever 1)  ****
    ``agreement_curve()`` below recomputes agreement by scoring CACHED target features with
    each saved head. That is correct only while the encoder is frozen, because the cache is
    only valid then (cache_features.py raises at n_blocks>0). For an unfrozen run the
    checkpoint's head belongs to a DIFFERENT encoder than the one that produced the cached
    features, so the post-hoc recompute would score a mismatched pair and report an
    agreement the model never had.

    The inline column in curve.csv IS correct for those runs -- it was computed at each
    curve point by encoding the target pool LIVE with the encoder as it stood at that step.
    So for n_blocks>0 the gate must READ the column rather than recompute it.

    NOTE this breaks the round-1 lock-step property (inline == post-hoc recompute), which
    was verified for the frozen round. That equality cannot hold for an unfrozen run by
    construction; it is not a regression, but it does mean the independent-recompute
    cross-check is unavailable there. Stated, not hidden.
    """
    p = os.path.join(run_dir, "curve.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no curve.csv in {run_dir}")
    import csv as _csv
    with open(p, newline="") as fh:
        rd = list(_csv.DictReader(fh))
    if not rd:
        raise ValueError(f"{p} has no rows")
    if col is None:
        cols = [c for c in rd[0] if c.startswith("agreement_fold")]
        if not cols:
            raise KeyError(f"{p} has no agreement_fold* column; the run did not log one.")
        col = cols[0]
    rows = []
    for r in rd:
        v = r.get(col, "")
        if v == "" or v is None:
            continue
        rows.append({"step": int(r["step"]), "agreement": float(v), "n": None})
    for i, r in enumerate(rows):
        r["delta"] = np.nan if i == 0 else r["agreement"] - rows[i - 1]["agreement"]
    if verbose:
        print(f"  source    : INLINE column '{col}' from {os.path.relpath(p, _ROOT)}")
        print(f"              (live-encoded during training; the post-hoc recompute is "
              f"INVALID for an unfrozen encoder)")
        print(f"  DA run    : {os.path.relpath(run_dir, _ROOT)}  ({len(rows)} curve points)")
    return rows, []


def agreement_curve(run_dir: str, ref_ckpt: str, cache_dir: str, split: str,
                    device: str = "cpu", verbose: bool = True):
    """Return (rows, skipped).

    rows    : [{step, agreement, n, delta}]  one per loadable saved DA checkpoint
    skipped : [(step, path, reason)]  checkpoints whose head the generic loader rejects
              -- reported, never worked around (spec 8).

    VALID ONLY FOR A FROZEN ENCODER -- it scores CACHED target features. For n_blocks>0 use
    ``agreement_curve_from_csv`` (see its docstring).
    """
    cache = FeatureCache(cache_dir)
    if not cache.has(split):
        raise FileNotFoundError(
            f"target-pool split '{split}' not in the feature cache at {cache_dir}. "
            f"The gate's real-data validation cannot run, and the TEST pool must NOT be "
            f"substituted for it (spec 2).")

    ref = reference_scores(ref_ckpt, cache, split, device)
    feats = cache.load(split)["features"]

    rows, skipped = [], []
    for step, path in list_step_checkpoints(run_dir):
        try:
            head, std, _ = load_checkpoint(path, device)
        except Exception as exc:                       # e.g. DANN's dann_adapter head
            skipped.append((step, path, f"{type(exc).__name__}: {exc}"))
            continue
        s = score_features(head, std, feats, device)
        rho, n = _spearman(s, ref)
        rows.append({"step": step, "agreement": rho, "n": n})

    for i, r in enumerate(rows):
        r["delta"] = np.nan if i == 0 else r["agreement"] - rows[i - 1]["agreement"]

    if verbose:
        print(f"  reference : {os.path.relpath(ref_ckpt, _ROOT)}  (Baseline A, R1/final)")
        print(f"  pool      : {split}  (n={feats.shape[0]} unlabeled target rows)")
        print(f"  DA run    : {os.path.relpath(run_dir, _ROOT)}  "
              f"({len(rows)} loadable checkpoints, {len(skipped)} skipped)")
    return rows, skipped


def print_curve(rows, label="agreement curve"):
    print(f"\n  {label}:")
    print("      step |  agreement |     delta")
    print("    -------+------------+----------")
    for r in rows:
        d = "     --  " if not np.isfinite(r["delta"]) else f"{r['delta']:+9.4f}"
        print(f"    {r['step']:>6} | {r['agreement']:>10.4f} | {d}")


# ---------------------------------------------------------------------------
# G2 -- the drop detector (gate logic)
# ---------------------------------------------------------------------------

def detect_drop(steps, agreement, W: float, M: float, F: float):
    """Two independent triggers over an agreement curve.  Returns a dict.

    CLIFF  : the largest agreement drop a[i] - a[j] over any pair with
             0 < step[j]-step[i] <= W.  Fires if that drop exceeds M **and the drop
             persists** -- i.e. agreement never comes back up to within M of the
             pre-drop level a[i] afterwards.  The persistence clause is the
             anti-false-fire requirement of spec 4/5: a healthy early DIP that recovers
             is not a collapse.  [EC]
    FLOOR  : agreement goes below F and STAYS below for the rest of the run.
             Catches a sustained slide with no recovery.  [EC]

    Raw statistics (max window drop, curve minimum) are returned unconditionally so the
    numbers can be read without reference to the uncalibrated M/W/F.
    """
    steps = np.asarray(steps, float)
    a = np.asarray(agreement, float)
    n = a.size
    out = {"W": W, "M": M, "F": F, "n_points": int(n)}

    if n == 0:
        out.update(max_window_drop=np.nan, max_window_drop_from=None,
                   max_window_drop_at=None, min_agreement=np.nan, min_at=None,
                   cliff_fired=False, cliff_step=None, cliff_recovered=False,
                   floor_fired=False, floor_step=None, fired=False, fire_step=None)
        return out

    # --- raw window statistics -------------------------------------------
    best = (-np.inf, None, None)          # (drop, i, j)
    best_persist = (-np.inf, None, None)  # largest drop that does NOT recover
    for i in range(n):
        for j in range(i + 1, n):
            if steps[j] - steps[i] > W:
                break
            drop = a[i] - a[j]
            if drop > best[0]:
                best = (drop, i, j)
            # persistence: does agreement ever climb back to within M of a[i]?
            recovered = bool(np.nanmax(a[j:]) >= a[i] - M)
            if (not recovered) and drop > best_persist[0]:
                best_persist = (drop, i, j)

    out["max_window_drop"] = float(best[0]) if np.isfinite(best[0]) else np.nan
    out["max_window_drop_from"] = None if best[1] is None else int(steps[best[1]])
    out["max_window_drop_at"] = None if best[2] is None else int(steps[best[2]])
    imin = int(np.nanargmin(a))
    out["min_agreement"] = float(a[imin])
    out["min_at"] = int(steps[imin])

    # --- cliff trigger ----------------------------------------------------
    cliff_fired = bool(np.isfinite(best_persist[0]) and best_persist[0] > M)
    out["cliff_fired"] = cliff_fired
    out["cliff_step"] = int(steps[best_persist[2]]) if cliff_fired else None
    out["cliff_drop"] = float(best_persist[0]) if np.isfinite(best_persist[0]) else np.nan
    # was the biggest raw drop suppressed purely by the persistence clause?
    out["cliff_recovered"] = bool(
        np.isfinite(best[0]) and best[0] > M and not cliff_fired)

    # --- floor trigger ----------------------------------------------------
    below = a < F
    floor_step = None
    if below.any():
        # first index from which every later point is also below F
        suffix_all_below = np.cumprod(below[::-1])[::-1].astype(bool)
        idx = np.flatnonzero(suffix_all_below)
        if idx.size:
            floor_step = int(steps[idx[0]])
    out["floor_fired"] = floor_step is not None
    out["floor_step"] = floor_step

    fires = [s for s in (out["cliff_step"], out["floor_step"]) if s is not None]
    out["fired"] = bool(fires)
    out["fire_step"] = int(min(fires)) if fires else None
    return out


def print_gate_table(rows, settings=SETTINGS, label=""):
    """Evaluate every transparent setting on one curve and print the fire/no-fire table."""
    steps = [r["step"] for r in rows]
    agr = [r["agreement"] for r in rows]
    res = {name: detect_drop(steps, agr, **cfg) for name, cfg in settings.items()}
    any_cfg = next(iter(res.values()))
    print(f"\n  RAW curve statistics{(' -- ' + label) if label else ''} "
          f"(no thresholds involved):")
    print(f"    largest drop within W={any_cfg['W']} steps : "
          f"{any_cfg['max_window_drop']:+.4f}  "
          f"(step {any_cfg['max_window_drop_from']} -> {any_cfg['max_window_drop_at']})")
    print(f"    curve minimum                        : "
          f"{any_cfg['min_agreement']:.4f}  (at step {any_cfg['min_at']})")
    print("\n  GATE (uncalibrated settings -- fire/no-fire is a CANDIDATE, not a verdict):")
    hdr = "    setting | W     | M     | F     | cliff        | floor        | FIRE at"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name, r in res.items():
        c = f"FIRE @{r['cliff_step']}" if r["cliff_fired"] else "no"
        if r["cliff_recovered"]:
            c = "no (recovered)"
        f = f"FIRE @{r['floor_step']}" if r["floor_fired"] else "no"
        fire = str(r["fire_step"]) if r["fired"] else "-- no fire --"
        print(f"    {name:<7} | {r['W']:<5g} | {r['M']:<5.2f} | {r['F']:<5.2f} | "
              f"{c:<12} | {f:<12} | {fire}")
    return res


# ---------------------------------------------------------------------------
# G3 -- self-test on synthetic curves with known answers
# ---------------------------------------------------------------------------

def _synthetic_cases():
    """Hand-built agreement curves with known answers (spec 5)."""
    cases = []

    # 1. HEALTHY DECLINE -- gradual 1.00 -> 0.85 then flat.  Must NOT fire.
    #    This is the false-positive guard: the most important test.
    steps = list(range(300, 300 * 25 + 1, 300))
    a = [max(0.85, 1.00 - 0.15 * (i / 12.0)) for i in range(len(steps))]
    cases.append(("healthy decline (gradual 1.00->0.85, then flat)", steps, a, False, None))

    # 2. CLIFF COLLAPSE -- flat ~0.90, sharp break to ~0.20 at step 4200.
    steps = list(range(300, 300 * 25 + 1, 300))
    a = []
    for s in steps:
        if s < 4200:
            a.append(0.90)
        elif s == 4200:
            a.append(0.55)
        else:
            a.append(0.20)
    cases.append(("cliff collapse (0.90 -> 0.20 at step 4200)", steps, a, True, 4200))

    # 3. SLOW SLIDE, NO FLOOR -- steady decline that never levels.  Floor must fire.
    steps = list(range(300, 300 * 30 + 1, 300))
    a = [1.00 - 0.90 * (i / (len(steps) - 1.0)) for i in range(len(steps))]
    cases.append(("slow slide, no floor (1.00 -> 0.10, never levels)", steps, a, True, None))

    # 4. EARLY DIP THEN RECOVERY -- healthy early divergence.  Must NOT fire.
    steps = list(range(300, 300 * 25 + 1, 300))
    a = [1.00, 0.95, 0.72, 0.80, 0.88, 0.92] + [0.90] * (len(steps) - 6)
    cases.append(("early dip then recovery (dip to 0.72, back to 0.90)", steps, a,
                  False, None))

    return cases


def self_test(settings=SETTINGS, verbose=True):
    print("\n" + "=" * 78)
    print("G3 -- SELF-TEST on synthetic agreement curves (known answers)")
    print("=" * 78)
    print("  Expectation is stated per case.  A gate that FALSE-FIRES on the healthy")
    print("  cases is the worst outcome and must be caught here (spec 5).")

    ok = True
    for name, steps, a, expect_fire, expect_step in _synthetic_cases():
        res = {n: detect_drop(steps, a, **c) for n, c in settings.items()}
        fired_any = any(r["fired"] for r in res.values())
        detail = ", ".join(
            f"{n}: {'FIRE@' + str(r['fire_step']) if r['fired'] else 'no-fire'}"
            f"(cliff={'y' if r['cliff_fired'] else 'n'},floor={'y' if r['floor_fired'] else 'n'})"
            for n, r in res.items())

        if expect_fire:
            passed = fired_any
            if passed and expect_step is not None:
                # S must land AT or just after the break
                s = min(r["fire_step"] for r in res.values() if r["fired"])
                passed = expect_step <= s <= expect_step + 900
        else:
            passed = not fired_any

        ok &= passed
        tag = "PASS" if passed else "FAIL"
        want = ("must FIRE" + (f" at ~{expect_step}" if expect_step else "")
                if expect_fire else "must NOT fire")
        print(f"\n  [{tag}] {name}")
        print(f"         expected: {want}")
        print(f"         observed: {detail}")

    print("\n  " + ("ALL SELF-TESTS PASS -- the gate separates cliffs/slides from healthy "
                    "decline and dips." if ok else
                    "*** SELF-TEST FAILURE -- do not use the gate until fixed. ***"))
    return ok


# ---------------------------------------------------------------------------
# G4 -- real-data validation (the one curve buildable today)
# ---------------------------------------------------------------------------

def _load_tpr_curve(run_dir, tpr_col="fold0_tpr_1e-3", step_col="step"):
    """The run's known TPR@1e-3 curve, for the overlay.  None if unavailable."""
    p = os.path.join(run_dir, "curve.csv")
    if pd is None or not os.path.isfile(p):
        return None
    df = pd.read_csv(p).sort_values(step_col).drop_duplicates(step_col)
    if tpr_col not in df.columns:
        return None
    return dict(zip(df[step_col].astype(int), df[tpr_col].astype(float)))


def print_overlay(rows, tpr_by_step, label):
    if not tpr_by_step:
        print("\n  (no curve.csv TPR column found -- overlay skipped)")
        return
    print(f"\n  OVERLAY -- agreement vs the known {label} TPR@1e-3 curve:")
    print("      step |  agreement | TPR@1e-3 |  agreement bar")
    print("    -------+------------+----------+" + "-" * 34)
    for r in rows:
        t = tpr_by_step.get(r["step"], np.nan)
        bar = "#" * int(round(max(0.0, min(1.0, r["agreement"])) * 30))
        ts = "   n/a  " if not np.isfinite(t) else f"{t:8.4f}"
        print(f"    {r['step']:>6} | {r['agreement']:>10.4f} | {ts} | {bar}")


# ---------------------------------------------------------------------------
# report driver
# ---------------------------------------------------------------------------

def run_report(run_dir, ref_ckpt, cache_dir, split, device="cpu", expect_healthy=True,
               tpr_col="fold0_tpr_1e-3", label=None, from_curve=False):
    label = label or os.path.basename(run_dir)
    print("\n" + "=" * 78)
    print(f"G1 -- AGREEMENT CURVE   [{label}]")
    print("=" * 78)
    if from_curve:
        rows, skipped = agreement_curve_from_csv(run_dir)
    else:
        rows, skipped = agreement_curve(run_dir, ref_ckpt, cache_dir, split, device)

    if skipped:
        print(f"\n  SKIPPED (head not loadable by evaluate.load_checkpoint -- reported, "
              f"not worked around):")
        for step, path, reason in skipped[:5]:
            print(f"    step {step}: {os.path.basename(path)}  [{reason.splitlines()[0][:90]}]")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    if not rows:
        print("\n  NO usable checkpoints -- cannot build a curve for this run. "
              "Reporting nothing rather than fabricating one.")
        return None

    print_curve(rows)

    print("\n" + "=" * 78)
    print(f"G2 -- DROP DETECTOR   [{label}]")
    print("=" * 78)
    res = print_gate_table(rows, label=label)

    tpr = _load_tpr_curve(run_dir, tpr_col=tpr_col)
    print_overlay(rows, tpr, label)

    a = np.array([r["agreement"] for r in rows], float)
    fired = any(r["fired"] for r in res.values())
    imax = int(np.argmax(a))
    rng = float(a.max() - a.min())
    if imax > 0 and a[imax] - a[0] > 0.01:
        shape = (f"RISES to a peak {a.max():.4f} at step {rows[imax]['step']}, then "
                 f"settles -- NOT the monotone 'starts near 1.0 and declines' shape "
                 f"predicted in spec 1")
    elif a[0] >= a[-1]:
        shape = "starts high and declines gradually, then levels (as spec 1 predicted)"
    else:
        shape = "see curve above"
    print(f"\n  SHAPE: first={a[0]:.4f}  last={a[-1]:.4f}  min={a.min():.4f}  "
          f"max={a.max():.4f}  total range={rng:.4f}   -> {shape}")
    if rng < 0.15:
        print(f"  CAVEAT [important]: the healthy curve's DYNAMIC RANGE is only {rng:.4f}.")
        print(f"    Agreement barely moved while the run's real metric changed a lot, so")
        print(f"    this run gives an upper anchor ('healthy lives near {a.mean():.2f}')")
        print(f"    but NO evidence about how far a collapse would push it. The headroom")
        print(f"    between healthy and collapse is [UNVERIFIED] -- see the scope note.")
    if expect_healthy:
        verdict = ("AS EXPECTED: gate does NOT fire on this healthy run."
                   if not fired else
                   "*** UNEXPECTED FIRE on a run known not to have collapsed -- the gate "
                   "is over-sensitive at this setting; do not adopt it. ***")
        print(f"  VERDICT: {verdict}")
    return {"rows": rows, "skipped": skipped, "gate": res}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Post-hoc collapse-gate validator (agreement vs Baseline A).")
    p.add_argument("--run", default="freematch",
                   help="DA run dir or shorthand (baseline|dann|freematch)")
    p.add_argument("--ref", default=DEFAULT_REF,
                   help="Baseline A FINAL (R1) checkpoint = the fixed reference")
    p.add_argument("--split", default="target_fold0",
                   help="feature-cache split holding the fold's UNLABELED target pool")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--tpr-col", default="fold0_tpr_1e-3")
    p.add_argument("--device", default="cpu")
    p.add_argument("--self-test-only", action="store_true")
    p.add_argument("--also-run", action="append", default=[],
                   help="extra run dir/shorthand to report (e.g. a sensitivity check)")
    p.add_argument("--from-curve", action="store_true",
                   help="read the INLINE agreement_fold* column from the run's curve.csv "
                        "instead of recomputing from cached features. REQUIRED for a "
                        "partially-unfrozen encoder (n_blocks>0), where the feature cache "
                        "is stale and the post-hoc recompute would score a head against "
                        "features from a different encoder.")
    args = p.parse_args(argv)

    print("=" * 78)
    print("COLLAPSE-GATE VALIDATOR (Job 1) -- post-hoc only.  No training, no halting.")
    print("=" * 78)
    print("  Agreement = Spearman(DA_ckpt scores, BaselineA_final scores) on the fold's")
    print("  UNLABELED target pool.  FLOOR ALARM only -- never maximize agreement (that")
    print("  returns the source model).  Firing thresholds are UNCALIBRATED [UNVERIFIED].")

    ok = self_test()
    if args.self_test_only:
        return 0 if ok else 1
    if not ok:
        print("\nABORT: self-test failed; refusing to report a real curve from broken logic.")
        return 1

    run_dir = SHORTHAND.get(args.run.lower(), args.run)
    if not os.path.isdir(run_dir):
        print(f"\nSKIP: run dir not found: {run_dir}")
        return 3
    if not os.path.isfile(args.ref):
        print(f"\nSKIP: reference checkpoint not found: {args.ref}")
        return 3

    print("\n" + "=" * 78)
    print("G4 -- REAL-DATA VALIDATION (the ONE curve buildable from on-disk checkpoints)")
    print("=" * 78)
    print("  FreeMatch fold-0 did NOT collapse (its TPR was still climbing at 5400), so a")
    print("  HEALTHY curve / NO FIRE is the expected result.  This establishes the healthy")
    print("  reference SHAPE.  It does NOT prove the gate catches a real collapse.")
    run_report(run_dir, args.ref, args.cache_dir, args.split, args.device,
               expect_healthy=True, tpr_col=args.tpr_col, from_curve=args.from_curve)

    for extra in args.also_run:
        d = SHORTHAND.get(extra.lower(), extra)
        if not os.path.isdir(d):
            print(f"\nSKIP extra run (not found): {d}")
            continue
        run_report(d, args.ref, args.cache_dir, args.split, args.device,
                   expect_healthy=True, tpr_col=args.tpr_col, from_curve=args.from_curve)

    print("\n" + "=" * 78)
    print("SCOPE STATEMENT (do not let the numbers above imply more than this)")
    print("=" * 78)
    print("  SHOWN     : the gate does not false-fire on 2 synthetic healthy curves, and")
    print("              fires on 2 synthetic broken ones with S at/just after the break;")
    print("              it does not false-fire on the one REAL healthy curve on disk.")
    print("  UNVERIFIED: that it catches a REAL collapse.  No collapsed run exists yet;")
    print("              CORAL (the one method with a documented late-epoch covariance")
    print("              blow-up, PROJECT.md 7.4) is the calibration case, pending the")
    print("              bake-off.  Do NOT promote this to a live early-stopper (Job 2)")
    print("              until that evidence exists.")
    print("\n  BAKE-OFF HOOK REQUIREMENT (one line, for the run spec):")
    print("    At every curve step, log one extra column per fold --")
    print("      agreement = Spearman(DA_checkpoint scores, BaselineA_final scores)")
    print("      on that fold's UNLABELED target pool (target_fold{k}) --")
    print("    so this gate can read live-produced curves without re-loading checkpoints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
