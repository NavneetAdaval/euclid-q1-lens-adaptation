"""
pseudolabel_precision.py
========================
POST-HOC ONLY. Pseudo-label precision on the fold's unlabeled target pool, computed from
the DUMPED LOGITS after the run has finished (BRIEF §5.6).

  ****  THE DISCIPLINE THIS SCRIPT EXISTS TO KEEP  ****
  The ~391 target lenses have known A/B grades that never enter the loss. Comparing the
  model's confident target predictions against them afterwards is the same category of
  operation as evaluating on test labels -- legitimate, and the mechanism figure the paper
  currently lacks. But it must NEVER inform a checkpoint choice, a hyperparameter or a
  method selection. The moment it does, it is oracle selection and PROJECT.md §7.6's
  argument collapses.

  That is why this is a SEPARATE SCRIPT reading dumped artifacts rather than a hook inside
  the training loop, why the training run writes no target labels at all (they are derived
  here, from splits.json), and why nothing here returns a "best" step. It prints the series;
  reading a maximum off it and calling that the checkpoint to keep would be the exact error.

WHAT IS RECONSTRUCTED, AND THE ONE HONEST APPROXIMATION
------------------------------------------------------
FreeMatch accepts an unlabeled sample when its max class probability on the WEAKLY
AUGMENTED view exceeds that class's local self-adaptive threshold. This script replays that
decision using:
  * the per-curve-point target logits from ``logits/step_*.npz`` -- which were produced by
    the DETERMINISTIC eval encode (aug="none", fp32), not the weak-augmented fp16 training
    view, and
  * the local thresholds tau_t(c) recorded in ``diagnostics.csv`` at the same step.

So this is the pseudo-label decision this model WOULD make on the clean view, not a
transcript of the exact accept/reject calls the training loop made on its augmented views.
[UNVERIFIED that the two agree closely -- they are the same model and threshold on a
slightly different view of the same images.] The threshold-free columns (target-pool AUROC
and top-k precision) do not depend on that reconstruction at all, and are the more robust
read.

Run:
    python scripts/pseudolabel_precision.py --run runs/freematch_fold0_nb3_unfrozen
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import auroc                                   # noqa: E402


def target_truth(splits_path, fold):
    """The target pool's TRUE identities, in LensDataset row order, from splits.json alone.

    lens_dataset.py builds role="target" as ``target_lens_rows`` (real lenses, latent 1)
    followed by ``target_neg_locations`` (random galaxies, latent 0). No image is loaded --
    only the split manifest -- so this stays cheap and cannot disturb the run's artifacts.
    Grades come from lens_cutouts.npz's ``grade`` array, indexed by the same rows.
    """
    with open(splits_path) as fh:
        s = json.load(fh)
    node = None
    for f in s["real_lenses_kfold"]["folds"]:
        if int(f["fold"]) == int(fold):
            node = f
            break
    if node is None:
        raise KeyError(f"fold {fold} not in {splits_path}")
    lens_rows = [int(r) for r in node["target_lens_rows"]]
    n_neg = len(node["target_neg_locations"])
    y = np.concatenate([np.ones(len(lens_rows), np.int64), np.zeros(n_neg, np.int64)])
    return y, lens_rows, n_neg


def lens_grades(data_root, lens_rows):
    """A/B grade per target lens row, or None if lens_cutouts.npz is not to hand."""
    p = os.path.join(data_root, "q1_lenses_processed", "lens_cutouts.npz")
    if not os.path.isfile(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        if "grade" not in z.files:
            return None
        g = z["grade"]
    return np.asarray([str(g[r]) for r in lens_rows])


def read_diag(path):
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                out[int(float(r["step"]))] = {k: (float(v) if v not in ("", None) else None)
                                              for k, v in r.items()}
            except (TypeError, ValueError):
                continue
    return out


def binary_prob2(logit):
    p = 1.0 / (1.0 + np.exp(-np.asarray(logit, dtype=np.float64)))
    return np.stack([1.0 - p, p], axis=1)


def replay(logits, tau_neg, tau_pos):
    """The FreeMatch accept decision: argmax class, accepted iff p > that class's local tau."""
    prob = binary_prob2(logits)
    pseudo = prob.argmax(1)
    maxp = prob.max(1)
    thr = np.where(pseudo == 1, tau_pos, tau_neg)
    return pseudo, (maxp > thr)


def step_row(step, logits, y, grades, diag):
    d = diag.get(step) or {}
    tau_neg, tau_pos = d.get("local_neg_last"), d.get("local_pos_last")
    row = {"step": step, "n_target": int(y.size),
           "target_auroc": float(auroc(y, np.asarray(logits, float))),
           "local_neg": tau_neg, "local_pos": tau_pos}

    # threshold-free: precision of the top-k highest-scoring target images, at k = the
    # number of real lenses in the pool. A rate, per PROJECT.md §9 -- never a count.
    k = int(y.sum())
    order = np.argsort(-np.asarray(logits, float))
    row["precision_at_k_lenses"] = float(y[order[:k]].mean()) if k else None

    if tau_neg is None or tau_pos is None:
        row.update({"accept_rate": None, "pseudo_precision_lens": None,
                    "pseudo_recall_lens": None, "pseudo_precision_nonlens": None})
        return row

    pseudo, acc = replay(logits, tau_neg, tau_pos)
    row["accept_rate"] = float(acc.mean())
    lens_acc = acc & (pseudo == 1)
    non_acc = acc & (pseudo == 0)
    row["accepted_lens_rate"] = float(lens_acc.mean())
    # PRECISION of the accepted lens pseudo-labels: of the target images this model would
    # confidently call a lens, what fraction really are one? This is the confirmation-bias
    # number (Arazo et al. arXiv:1908.02983) -- it is what "the encoder can reshape features
    # until its own mistakes look confidently right" would degrade.
    row["pseudo_precision_lens"] = (float(y[lens_acc].mean()) if lens_acc.any() else None)
    row["pseudo_precision_nonlens"] = (float(1.0 - y[non_acc].mean()) if non_acc.any()
                                       else None)
    row["pseudo_recall_lens"] = (float(lens_acc[y == 1].mean()) if (y == 1).any() else None)

    if grades is not None:
        n_lens = int(y.sum())
        for g in ("A", "B"):
            m = np.zeros(y.size, bool)
            m[:n_lens] = (grades == g)
            row[f"recall_grade_{g}"] = float(lens_acc[m].mean()) if m.any() else None
    return row


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run dir containing logits/ + diagnostics.csv")
    ap.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    ap.add_argument("--data-root", default=os.path.join(root, "data"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    run = os.path.abspath(args.run)
    logit_dir = os.path.join(run, "logits")
    files = sorted(glob.glob(os.path.join(logit_dir, "step_*.npz")),
                   key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    if not files:
        raise SystemExit(f"[pseudolabel] no logit dumps under {logit_dir} -- this analysis "
                         f"reads the dumped logits (BRIEF §5.2) and cannot be run without "
                         f"them.")
    y, lens_rows, n_neg = target_truth(os.path.abspath(args.splits), args.fold)
    grades = lens_grades(os.path.abspath(args.data_root), lens_rows)
    diag = read_diag(os.path.join(run, "diagnostics.csv"))
    key = f"target_fold{int(args.fold)}"

    rows = []
    for p in files:
        step = int(re.search(r"step_(\d+)", p).group(1))
        with np.load(p, allow_pickle=False) as z:
            if key not in z.files:
                continue
            logits = z[key]
        if logits.size != y.size:
            raise SystemExit(
                f"[pseudolabel] {p}: {logits.size} target logits but splits.json says the "
                f"fold-{args.fold} target pool has {y.size} rows. These are different "
                f"pools -- refusing to align them.")
        rows.append(step_row(step, logits, y, grades, diag))

    print("=" * 118)
    print(f"PSEUDO-LABEL PRECISION on target_fold{args.fold}  --  POST-HOC ONLY (BRIEF §5.6)")
    print(f"  run: {run}")
    print(f"  target pool: {int(y.sum())} real lenses + {n_neg} random galaxies "
          f"(grades: {'yes' if grades is not None else 'unavailable'})")
    print("  THIS MUST NOT INFORM ANY CHECKPOINT, HYPERPARAMETER OR METHOD CHOICE.")
    print("=" * 118)
    cols = ["step", "target_auroc", "precision_at_k_lenses", "accept_rate",
            "accepted_lens_rate", "pseudo_precision_lens", "pseudo_recall_lens",
            "pseudo_precision_nonlens", "recall_grade_A", "recall_grade_B"]
    print("  " + " | ".join(f"{c[:13]:>13}" for c in cols))
    print("  " + "-" * (16 * len(cols)))
    for r in rows:
        print("  " + " | ".join(
            f"{(f'{r[c]:.4f}' if isinstance(r.get(c), float) else str(r.get(c) if r.get(c) is not None else 'n/a')):>13}"
            for c in cols))

    out = os.path.abspath(args.out or os.path.join(run, "pseudolabel_precision.json"))
    with open(out, "w") as fh:
        json.dump({"run": run, "fold": args.fold, "n_target": int(y.size),
                   "n_target_lenses": int(y.sum()), "rows": rows,
                   "discipline": ("POST-HOC ONLY -- must never inform a checkpoint, "
                                  "hyperparameter or method choice (BRIEF §5.6)"),
                   "approximation": ("replayed on the deterministic aug='none' eval encode, "
                                     "not the weak-augmented fp16 training view")},
                  fh, indent=2)
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
