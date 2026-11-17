"""
compare_fold_timing.py
======================
Why is one fold slower than another? Answers it from ``trace.csv`` rather than from
impressions, and distinguishes the two causes that need different responses:

  UNIFORM   -- every 125-step window is slower by roughly the same factor. The session
               is slower, not the run: a different physical T4, a busier host, thermal
               throttling. Nothing to fix and nothing to report.
  LOCALIZED -- the per-window rate is normal except for spikes. Those land at
               checkpoint steps, which means I/O contention (writing ~66 MB per
               checkpoint into a /kaggle/working that already holds another fold), or
               they land anywhere, which means the host was interrupted.

WHAT IS AND IS NOT FOLD-DEPENDENT HERE. The training work is IDENTICAL across folds by
construction: ``steps_per_epoch`` is 187 on every fold (the source pool does not vary by
fold -- Baseline A is source-only), batch is 256, the target pool is 782 images and the
discriminator holdout is 78. Only ``n_test_positives`` differs, 98 on folds 0-3 and 97 on
fold 4, and that touches scoring only. **So a fold-to-fold wall-time difference is a
property of the venue, not of the fold.**

THE REFERENCE SPREAD, measured on round 5's DANN confirmatory folds -- same runner, same
4000-step budget, same T4, no resumes on any of them:

    fold 1  152.6 min   (+5.8%)
    fold 2  137.9 min   (-4.4%)     <- the FASTEST of the four
    fold 3  147.1 min   (+1.9%)
    fold 4  139.6 min   (-3.3%)
    mean 144.3, sd 6.8, range 14.7 min = 10.2% of the mean

A 10% spread across folds is what this venue already produced with nothing wrong. Treat a
difference inside that band as noise; it is evidence about Kaggle, not about the run.

    python -u scripts/compare_fold_timing.py A/trace.csv B/trace.csv
    python -u scripts/compare_fold_timing.py --labels fold1 fold2 A/trace.csv B/trace.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st

CKPT_EVERY = 250

# round 5's DANN confirmatory folds: the venue's own fold-to-fold spread, no resumes.
REFERENCE = {"fold 1": 152.6, "fold 2": 137.9, "fold 3": 147.1, "fold 4": 139.6}


def read_trace(path: str) -> list:
    """(step, cumulative_wall_secs, n_steps_in_window) rows, in step order."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            out.append((int(float(r["step"])), float(r["wall_secs"]),
                        int(float(r.get("n_steps_in_window") or 0))))
        except (TypeError, ValueError, KeyError):
            continue
    return sorted(out, key=lambda t: t[0])


def windows(trace: list) -> list:
    """(step, seconds_per_step) for each logging window."""
    out = []
    for (s0, w0, _), (s1, w1, _) in zip(trace, trace[1:]):
        ds, dw = s1 - s0, w1 - w0
        if ds > 0 and dw >= 0:
            out.append((s1, dw / ds))
    return out


def describe(label: str, path: str) -> dict:
    tr = read_trace(path)
    if len(tr) < 3:
        raise SystemExit(f"{path}: only {len(tr)} usable rows")
    wins = windows(tr)
    rates = [r for _, r in wins]
    med = st.median(rates)
    at_ckpt = [r for s, r in wins if s % CKPT_EVERY == 0]
    off_ckpt = [r for s, r in wins if s % CKPT_EVERY != 0]
    spikes = [(s, r) for s, r in wins if r > 2.0 * med]
    return {
        "label": label, "path": path,
        "last_step": tr[-1][0], "wall_min": tr[-1][1] / 60.0,
        "n_windows": len(wins),
        "median_s_per_step": med,
        "p90_s_per_step": sorted(rates)[int(0.9 * (len(rates) - 1))],
        "median_at_ckpt": st.median(at_ckpt) if at_ckpt else None,
        "median_off_ckpt": st.median(off_ckpt) if off_ckpt else None,
        "spikes": spikes,
        "projected_4000_min": med * 4000 / 60.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="two or more trace.csv paths")
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(os.path.abspath(t)))
                             for t in args.traces]
    if len(labels) != len(args.traces):
        raise SystemExit("--labels must match the number of traces")

    runs = [describe(l, t) for l, t in zip(labels, args.traces)]

    print("=" * 78)
    print("FOLD TIMING -- is one run slower, or is the session slower?")
    print("=" * 78)
    print(f"\n{'run':<14}{'steps':>7}{'wall min':>10}{'s/step':>9}"
          f"{'p90':>8}{'proj 4000':>11}")
    for r in runs:
        print(f"{r['label']:<14}{r['last_step']:>7}{r['wall_min']:>10.1f}"
              f"{r['median_s_per_step']:>9.3f}{r['p90_s_per_step']:>8.3f}"
              f"{r['projected_4000_min']:>11.1f}")

    base = runs[0]
    print(f"\nagainst {base['label']}:")
    for r in runs[1:]:
        f = r["median_s_per_step"] / base["median_s_per_step"]
        print(f"  {r['label']}: median step rate {f:.3f}x  "
              f"({'slower' if f > 1 else 'faster'} by {abs(f - 1) * 100:.1f}%)")

    print("\ncheckpoint-step windows vs ordinary windows (I/O contention shows here):")
    for r in runs:
        a, b = r["median_at_ckpt"], r["median_off_ckpt"]
        if a and b:
            print(f"  {r['label']:<14} at ckpt {a:.3f} s/step   "
                  f"off ckpt {b:.3f} s/step   ratio {a / b:.2f}x")

    print("\nspikes (> 2x that run's median):")
    any_spike = False
    for r in runs:
        if r["spikes"]:
            any_spike = True
            at = sum(1 for s, _ in r["spikes"] if s % CKPT_EVERY == 0)
            print(f"  {r['label']:<14} {len(r['spikes'])} spike(s), "
                  f"{at} of them at a checkpoint step")
            for s, v in r["spikes"][:6]:
                print(f"      step {s:>5}  {v:.3f} s/step"
                      f"{'  (checkpoint)' if s % CKPT_EVERY == 0 else ''}")
    if not any_spike:
        print("  none -- the difference is UNIFORM, i.e. a slower session, not a stall")

    ref = list(REFERENCE.values())
    m, lo, hi = st.mean(ref), min(ref), max(ref)
    print(f"\nreference spread, round 5's DANN folds 1-4 on this venue "
          f"(same runner, same budget, no resumes):")
    print(f"  {'  '.join(f'{k}={v:.1f}' for k, v in REFERENCE.items())}")
    print(f"  mean {m:.1f} min, sd {st.stdev(ref):.1f}, "
          f"range {hi - lo:.1f} min = {100 * (hi - lo) / m:.1f}% of the mean")
    print(f"  DANN's fold 2 was the FASTEST of the four -- there is nothing intrinsic")
    print(f"  about any fold. steps_per_epoch is 187 on every fold and the training")
    print(f"  work is identical by construction.")

    spread = max(r["projected_4000_min"] for r in runs) - \
        min(r["projected_4000_min"] for r in runs)
    print(f"\nobserved spread here, projected to 4000 steps: {spread:.1f} min "
          f"({100 * spread / m:.1f}% of the reference mean)")
    if spread <= (hi - lo):
        print("  INSIDE the spread this venue already produced with nothing wrong.")
        print("  Not worth acting on. Record the wall time and move on.")
    else:
        print("  LARGER than round 5's fold-to-fold spread. Worth a look:")
        print("    - check `gpu_name` in each run_config.json (T4 vs anything else)")
        print("    - check `resume_events` -- a resume redoes work since the last save")
        print("    - check free space in /kaggle/working; a second fold's checkpoints")
        print("      sharing the disk slows every checkpoint write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
