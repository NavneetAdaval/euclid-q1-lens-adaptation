"""
analyze_holdout_gap_fold0.py
============================
ROUND 7 CLOSING, TASK 1's CHEAP CHECK -- the 704/78 diagnostic on FOLD 0, where the
round-5 development runs DID retain weights.
**RETROSPECTIVE AND UNREGISTERED. n = 1 fold. No ordering claim, no significance claim.**

    python -u scripts/analyze_holdout_gap_fold0.py --out runs/round7/holdout_gap_fold0.json

WHAT THE CHECK FOUND. §7.29 recorded that DANN x nb3's CONFIRMATORY folds retained zero
`.pt` files. The development runs are separate artifacts and they did retain some:

  * DANN arm 2 (lambda live), fold 0 -- ONE checkpoint, ``step_8000.pt``.
  * DANN arm 1 (lambda = 0),   fold 0 -- ALL 32 checkpoints, 250..8000.
  * BNM x nb3, fold 0 -- no weights, but its dev run WROTE the gap into
    ``checkpoints.csv`` at all 32 checkpoints, so its readings are already on disk.

That buys two things §7.29 could not have:

  1. **A matched single-fold cross-arm pair.** Arm 1 and arm 2 are the same run
     configuration differing in lambda and nothing else (§7.18), on the same fold, with the
     SAME 78-image holdout, read at the SAME step 8000. It is n = 1 and it is descriptive,
     but it is the only place in the project where the cross-arm question can be looked at
     at all.
  2. **A null for readings (2) and (3).** §7.27 declared three readings. lambda=0's
     confirmatory folds retained one checkpoint each, so §7.29 could only produce reading
     (1). Arm 1's full fold-0 grid supplies the any-checkpoint maximum and the within-fold
     mean for an arm with NO target objective -- the reference the other two readings have
     been missing.

THE CODE PATH IS BNM's, UNCHANGED, exactly as in ``analyze_holdout_gap_lambda0``:
``probe_stats``, ``holdout_gap`` and ``build_trainedon_probe`` from ``bnm_round6``, the
holdout builder from ``dann_round5``. The reconstructed slices are asserted equal to the
indices the runs recorded before anything is scored.

Inference only. No training, no optimizer step, no gradient.

Writes NOTHING outside --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

from bnm_round6 import (build_trainedon_probe, holdout_gap, probe_stats)   # noqa: E402
from dann_round5 import (build_target_holdout, materialise_holdout_images)  # noqa: E402
from integrity import verify as _verify_integrity                          # noqa: E402
from lens_dataset import LensDataset                                       # noqa: E402
from train_skeleton import TrainConfig, Trainer                            # noqa: E402

FOLD = 0
SEED = 42
D_HOLDOUT_FRAC = 0.10
MATCHED_STEP = 8000        # the only step arm 2 retained; arm 1 has it too

SPLITS = os.path.join(REPO, "data", "splits", "splits.json")
DATA_ROOT = os.path.join(REPO, "data")

ARM2 = os.path.join(REPO, "runs", "dann_arm2_fold0", "extracted", "dann_r5_nb3_fold0")
ARM1 = os.path.join(REPO, "runs", "dann_arm1", "extracted",
                    "dann_r5_nb3_lambda0_fold0")
BNM_DEV = os.path.join(REPO, "runs", "bnm_r6_dev8000", "bnm_r6_dev8000_fold0")


def steps_available(run_dir):
    d = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(d):
        return []
    return sorted(int(f[5:-3]) for f in os.listdir(d)
                  if f.startswith("step_") and f.endswith(".pt"))


def build_trainer(run_dir, step, device):
    path = os.path.join(run_dir, "checkpoints", f"step_{step}.pt")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    c = dict(raw["cfg"])
    cfg = TrainConfig(
        splits_path=SPLITS, data_root=DATA_ROOT, fold=FOLD,
        head_type=c["head_type"], encoder_mode=c["encoder_mode"],
        n_blocks=int(c["n_blocks"]), init_head_from=None,
        lr=c["lr"], weight_decay=c["weight_decay"], batch_size=c["batch_size"],
        seed=int(c["seed"]), device=str(device), amp=False,
        grad_checkpointing=False, verbose_encoder=False)
    tr = Trainer(cfg)
    tr.load(path, map_location=device, load_training_state=False)
    tr.head.eval()
    return tr


def summarise(label, gaps_by_step, budget=None):
    """The three readings §7.27 declared, on whatever steps exist."""
    steps = sorted(gaps_by_step)
    vals = [gaps_by_step[s] for s in steps]
    out = {"n_checkpoints": len(steps), "steps": steps,
           "any_checkpoint_max": max(vals) if len(vals) > 1 else None,
           "within_fold_mean": st.mean(vals) if len(vals) > 1 else None,
           "at_matched_step": gaps_by_step.get(MATCHED_STEP)}
    if budget is not None and budget in gaps_by_step:
        out["at_own_budget"] = gaps_by_step[budget]
        out["own_budget"] = budget
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    _verify_integrity(where="before the round-7 fold-0 holdout-gap check")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 100)
    print("ROUND 7 CLOSING -- THE 704/78 DIAGNOSTIC ON FOLD 0 (the development runs)")
    print("*** RETROSPECTIVE AND UNREGISTERED. n = 1 FOLD. NO ORDERING CLAIM. ***")
    print("=" * 100)
    print(f"  device {device}   inference only")
    print(f"  arm 2 (lambda live) retained: {steps_available(ARM2)}")
    print(f"  arm 1 (lambda = 0)  retained: {len(steps_available(ARM1))} checkpoints "
          f"{steps_available(ARM1)[:3]}...{steps_available(ARM1)[-1:]}")
    print("  BNM fold 0 retained no weights; its gaps come from its own checkpoints.csv")

    # ---- the slices, reconstructed and asserted ----
    tgt_ds = LensDataset(SPLITS, DATA_ROOT, role="target", fold=FOLD, aug="weak",
                         n_views=1)
    holdout = build_target_holdout(tgt_ds, frac=D_HOLDOUT_FRAC, seed=SEED + FOLD)
    probe = build_trainedon_probe(tgt_ds, holdout, seed=SEED + FOLD)

    got = set(int(i) for i in holdout["holdout"])
    for lab, p in (("arm 2", ARM2), ("arm 1", ARM1)):
        rec = set(json.load(open(os.path.join(p, "run_config.json"),
                                 encoding="utf-8"))["dann_d_holdout_indices"])
        if got != rec:
            raise SystemExit(f"{lab}: reconstructed holdout != the slice it ran with. STOP.")
    rec_probe = set(int(i) for i in json.load(
        open(os.path.join(BNM_DEV, "run_config.json"),
             encoding="utf-8"))["trainedon_probe"]["probe"])
    if set(int(i) for i in probe["probe"]) != rec_probe:
        raise SystemExit("reconstructed probe != BNM's fold-0 probe. STOP.")
    print(f"\n  slices verified: holdout {len(got)} matches BOTH DANN arms; trained-on "
          f"probe {len(rec_probe)} matches BNM's fold-0 probe; disjoint "
          f"{not (set(probe['probe']) & got)}")

    held_images = materialise_holdout_images(tgt_ds, holdout["holdout"])
    probe_images = materialise_holdout_images(tgt_ds, probe["probe"])

    out = {"round": 7, "task": "704_78_diagnostic_fold0_development_runs",
           "fold": FOLD, "status": "RETROSPECTIVE AND UNREGISTERED",
           "n_folds": 1, "matched_step": MATCHED_STEP,
           "code_path": "bnm_round6.probe_stats / bnm_round6.holdout_gap, unchanged",
           "arms": {}}

    # ---- the two DANN arms ----
    for label, run_dir in (("DANN arm 2 (lambda live)", ARM2),
                           ("DANN arm 1 (lambda = 0)", ARM1)):
        steps = steps_available(run_dir)
        gaps, detail = {}, {}
        print(f"\n  {label} -- {len(steps)} checkpoint(s)")
        for s in steps:
            tr = build_trainer(run_dir, s, device)
            with torch.no_grad():
                g = holdout_gap(probe_stats(tr, probe_images),
                                probe_stats(tr, held_images))
            del tr
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gaps[s] = g["nuclear_gap"]
            detail[str(s)] = g
            if s == MATCHED_STEP or len(steps) <= 4:
                print(f"    step {s:>5}  ||A||* gap {g['nuclear_gap']:+.4f}   "
                      f"entropy gap {g['entropy_gap']:+.4f}   "
                      f"pred-pos gap {g['pred_pos_frac_gap']:+.4f}")
        out["arms"][label] = {"per_step": detail,
                              "readings": summarise(label, gaps)}

    # ---- BNM fold 0, from its own record ----
    rows = list(csv.DictReader(open(os.path.join(BNM_DEV, "checkpoints.csv"),
                                    encoding="utf-8")))
    bnm_gaps = {int(r["step"]): float(r["nuclear_gap"]) for r in rows}
    out["arms"]["BNM x nb3 (from its own checkpoints.csv)"] = {
        "per_step": {str(k): {"nuclear_gap": v} for k, v in bnm_gaps.items()},
        "readings": summarise("BNM", bnm_gaps, budget=1000)}

    # ---- the report ----
    print("\n" + "=" * 100)
    print("THE MATCHED SINGLE-FOLD PAIR -- same fold, same holdout, same step 8000")
    print("=" * 100)
    a2 = out["arms"]["DANN arm 2 (lambda live)"]["readings"]["at_matched_step"]
    a1 = out["arms"]["DANN arm 1 (lambda = 0)"]["readings"]["at_matched_step"]
    bn = bnm_gaps.get(MATCHED_STEP)
    print(f"  DANN arm 2 (lambda live)   ||A||* gap at 8000: {a2:+.4f}")
    print(f"  DANN arm 1 (lambda = 0)    ||A||* gap at 8000: {a1:+.4f}   <- the null")
    print(f"  BNM x nb3                  ||A||* gap at 8000: {bn:+.4f}")
    print(f"\n  arm 2 minus its own control: {a2 - a1:+.4f}")
    print("  n = 1 FOLD. This is a description of one fold, not a test and not an ordering.")
    out["matched_pair_at_8000"] = {"arm2": a2, "arm1_null": a1, "bnm": bn,
                                   "arm2_minus_arm1": a2 - a1,
                                   "caveat": "n = 1 fold; descriptive only, no test"}

    print("\n" + "=" * 100)
    print("THE NULL FOR READINGS (2) AND (3) -- which §7.29 could not supply")
    print("=" * 100)
    r1 = out["arms"]["DANN arm 1 (lambda = 0)"]["readings"]
    rb = out["arms"]["BNM x nb3 (from its own checkpoints.csv)"]["readings"]
    print(f"  {'reading':<28}{'BNM fold 0':>14}{'lambda=0 fold 0 (null)':>26}")
    print(f"  {'(2) any-checkpoint maximum':<28}{rb['any_checkpoint_max']:>+14.4f}"
          f"{r1['any_checkpoint_max']:>+26.4f}")
    print(f"  {'(3) within-fold mean':<28}{rb['within_fold_mean']:>+14.4f}"
          f"{r1['within_fold_mean']:>+26.4f}")
    print(f"  {'at step 8000':<28}{bn:>+14.4f}{a1:>+26.4f}")
    print("\n  Both arms are read on the SAME 32-point grid, the same fold and the same")
    print("  slice. The null has NO target objective, so any gap it shows is slice")
    print("  heterogeneity plus whatever the source objective alone induces.")
    print("  n = 1 fold: descriptive, no test, no ordering claim.")
    out["null_for_readings_2_and_3"] = {"bnm_fold0": rb, "lambda0_fold0": r1}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n[out] {os.path.relpath(os.path.abspath(a.out), REPO)}")

    _verify_integrity(where="after the round-7 fold-0 holdout-gap check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
