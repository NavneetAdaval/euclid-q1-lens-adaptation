"""
report_signals.py
=================
LABEL-FREE SELECTION-SIGNAL REPORT (post-process, local). Reads one or more
``curve_with_signals.csv`` files (one per method, produced by signal_purity.py /
signal_headgrad.py) and, PER METHOD, reports:

  * ORACLE step   -- the checkpoint that maximises the fold-0 TPR@1e-3 curve, smoothed
                     with a centred rolling window of 3 (so a single-step spike cannot
                     win). TPR@1e-3 is the SELECTION TARGET; this is the checkpoint we
                     wish we could pick.
  * SIGNAL step(s) -- for each label-free signal column present, the checkpoint that
                     signal would select:
                       - ``purity_50comp``          -> the MAX (higher purity = better).
                       - ``headgrad_bce_balanced``  -> reported BOTH ways, the MIN-pick
                         AND the MAX-pick, side by side. We do NOT default to "min":
                         the baseline showed headgrad-MIN drifts to the OVERFIT late
                         checkpoint (the source-BCE gradient shrinks under sim-saturation),
                         so the paper's "min = converged" convention is suspect in THIS
                         regime -- surface both directions and let the user judge.
  * GAP in TPR@1e-3 -- raw fold0 TPR@1e-3 at the ORACLE-picked checkpoint MINUS that at
                     the SIGNAL-picked checkpoint. This is THE DELIVERABLE: "how much
                     headline metric you lose by trusting this signal." A positive gap =
                     the signal costs you that much TPR; a negative gap = the signal
                     happened to beat the smoothed oracle pick.

This tool SURFACES the numbers only. It does NOT crown a winning signal, and it does NOT
auto-pick a direction for headgrad -- both directions' step + gap are shown side by side.

Selection conventions (stated, not hidden):
  - The oracle uses the SMOOTHED (window-3) TPR curve; signals are picked on their RAW
    per-checkpoint values (only the oracle got a smoothing window in the spec).
  - Ties are broken by the EARLIEST step (first checkpoint reaching the best value).
  - purity_50comp: higher is better (max). headgrad_bce_balanced: shown as MIN and MAX.

READ-ONLY: writes nothing. Only reads the given curve_with_signals.csv files. A missing
file, a missing required column (step / fold0_tpr_1e-3), or a signals file with NO signal
column at all is a LOUD error -- no silent skips.

USAGE
-----
    python scripts/report_signals.py runs/curve_baseline_fold0/curve_with_signals.csv
    python scripts/report_signals.py runs/*/curve_with_signals.csv          # several methods
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

TPR_COLUMN = "fold0_tpr_1e-3"
HEADGRAD_COLUMN = "headgrad_bce_balanced"

# each signal column -> ordered list of (direction, tag) selection rules to REPORT.
# purity: only "higher=better". headgrad: BOTH min and max (don't auto-pick a direction --
# "min=converged" is suspect under sim-saturation; see the module docstring).
SIGNAL_SPECS = {
    "purity_50comp":  [("higher", "max")],
    HEADGRAD_COLUMN:  [("lower", "min"), ("higher", "max")],
}
DEFAULT_WINDOW = 3


def _die(msg: str):
    raise SystemExit(f"[report_signals] ERROR: {msg}")


def _method_label(csv_path: str) -> str:
    """Human label for a method = the run-dir (parent) folder name."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    return parent or os.path.basename(csv_path)


def _pick_extreme(df: pd.DataFrame, column: str, direction: str) -> int:
    """Row index (into df) of the best value in ``column``; ties -> earliest step.

    ``direction`` is "higher" (argmax) or "lower" (argmin). NaNs are ignored; earliest
    step wins ties because the frame is sorted by step first."""
    valid = df[df[column].notna()]
    if valid.empty:
        _die(f"column '{column}' is entirely NaN -- no checkpoint to pick from")
    best = valid[column].max() if direction == "higher" else valid[column].min()
    winners = valid[np.isclose(valid[column], best)]
    return int(winners.index[0])          # earliest step (df pre-sorted by step)


def analyse_method(csv_path: str, *, window: int) -> dict:
    if not os.path.isfile(csv_path):
        _die(f"missing curve_with_signals.csv: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ("step", TPR_COLUMN):
        if col not in df.columns:
            _die(f"{csv_path}: required column '{col}' absent (found {list(df.columns)})")
    df = df.sort_values("step").reset_index(drop=True)

    present = [c for c in SIGNAL_SPECS if c in df.columns]
    if not present:
        _die(f"{csv_path}: no signal column present (looked for {list(SIGNAL_SPECS)}). "
             f"Run signal_purity.py / signal_headgrad.py first.")

    # ORACLE: argmax of the window-smoothed TPR curve (spec: rolling window 3)
    smoothed = df[TPR_COLUMN].rolling(window=window, center=True, min_periods=1).mean()
    df_sm = df.assign(_tpr_smoothed=smoothed)
    oracle_idx = _pick_extreme(df_sm, "_tpr_smoothed", "higher")
    oracle_step = int(df.loc[oracle_idx, "step"])
    oracle_tpr = float(df.loc[oracle_idx, TPR_COLUMN])

    # one PICK per (column, direction) rule -- headgrad yields two (min + max)
    picks = []
    for col in present:
        for direction, tag in SIGNAL_SPECS[col]:
            idx = _pick_extreme(df, col, direction)
            pick_tpr = float(df.loc[idx, TPR_COLUMN])
            picks.append({
                "name": f"{col}[{tag}]",
                "column": col,
                "tag": tag,
                "step": int(df.loc[idx, "step"]),
                "value": float(df.loc[idx, col]),
                "tpr": pick_tpr,
                "gap": oracle_tpr - pick_tpr,      # +ve = TPR lost by trusting the signal
            })

    return {
        "label": _method_label(csv_path),
        "path": csv_path,
        "n_ckpts": int(len(df)),
        "window": window,
        "oracle_step": oracle_step,
        "oracle_tpr": oracle_tpr,
        "picks": picks,
    }


def _print_report(results: list, window: int):
    for r in results:
        print(f"\n=== {r['label']}  ({r['n_ckpts']} checkpoints)  [{r['path']}]")
        print(f"  ORACLE (TPR@1e-3, rolling window {window}): "
              f"step {r['oracle_step']}  ->  TPR@1e-3 = {r['oracle_tpr']:.4f}")
        for p in r["picks"]:
            print(f"  SIGNAL {p['name']}: step {p['step']}  "
                  f"[{p['column']}={p['value']:.4f}]  ->  TPR@1e-3 = {p['tpr']:.4f}   "
                  f"GAP = {p['gap']:+.4f} TPR")

    # compact cross-method table of the deliverable (gaps), one column per pick
    names = []                                   # first-seen order, stable across methods
    for r in results:
        for p in r["picks"]:
            if p["name"] not in names:
                names.append(p["name"])
    w = max([len(n) for n in names] + [12]) + 2

    print("\n--- GAP in TPR@1e-3 (oracle-pick minus signal-pick); +ve = metric LOST ---")
    header = f"{'method':<26}{'oracleTPR':>10}" + "".join(f"{n:>{w}}" for n in names)
    print(header)
    for r in results:
        gapmap = {p["name"]: p["gap"] for p in r["picks"]}
        line = f"{r['label'][:26]:<26}{r['oracle_tpr']:>10.4f}"
        line += "".join(f"{gapmap[n]:>+{w}.4f}" if n in gapmap else f"{'-':>{w}}"
                        for n in names)
        print(line)
    print("\n(Surface only -- no signal and no headgrad DIRECTION is auto-selected. Read "
          "each gap against the TPR@1e-3 CI for that method before trusting a signal.)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("curves", nargs="+", help="one or more curve_with_signals.csv (one per method)")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help="rolling window for the oracle TPR smoothing (default 3)")
    args = ap.parse_args(argv)

    results = [analyse_method(p, window=args.window) for p in args.curves]
    _print_report(results, args.window)


if __name__ == "__main__":
    main()
