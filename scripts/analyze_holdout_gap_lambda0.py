"""
analyze_holdout_gap_lambda0.py
==============================
ROUND 7, TASK 2 -- the 704/78 diagnostic on the lambda=0 arm.
**RETROSPECTIVE AND UNREGISTERED.** No prediction was made for it and none may be written
for it now; it cannot be scored against one.

    python -u scripts/analyze_holdout_gap_lambda0.py --out runs/round7/holdout_gap_lambda0.json

WHAT THIS IS, AND WHAT THE REST OF TASK 2 IS NOT. The brief asks for the 704/78 gap on
every arm that carries the same held-out target slice. The slice exists on exactly three
arms -- BNM (measured during its own run), DANN x nb3 and lambda=0, all three
byte-identical -- and CORAL x nb3 and FreeMatch x nb3 have no held-out target slice at
all, so they are excluded rather than given one retrospectively.

Of the three that have it, only BNM and lambda=0 can be MEASURED:

  * BNM wrote the gap into ``checkpoints.csv`` at every checkpoint while it trained.
  * DANN x nb3's confirmatory folds retained **zero** model checkpoints (0 ``.pt`` files
    across all four folds), and its per-checkpoint record carries ``d_acc_holdout`` -- the
    DISCRIMINATOR's accuracy -- not the nuclear norm, the entropy or the predicted-positive
    fraction. Those three cannot be recovered from a logit dump: they are functions of the
    model's output on TARGET images, and no arm but FreeMatch ever dumped target logits.
    Recovering them would mean re-running training, which round 7 forbids.
  * lambda=0 retained ONE checkpoint per fold, ``resume/latest.pt`` at step 4000, which
    is its pre-registered read-off point. So reading (1) of 7.27 is computable and
    readings (2) and (3) are not, on this arm, ever.

WHY lambda=0 IS THE ONE WORTH HAVING ANYWAY. It is the gamma=0 / lambda=0 floor: no target
gradient reaches the encoder or head at all, so NEITHER slice is trained on. Its gap is
therefore a pure slice-to-slice sampling difference under a model that saw neither -- the
null that says how much of BNM's +0.0993 is memorisation and how much is the two slices
simply differing. That is the single most useful number Task 2 can still produce.

THE CODE PATH IS BNM's, UNCHANGED. ``build_target_holdout``, ``build_trainedon_probe``,
``probe_stats``, ``materialise_holdout_images`` and ``holdout_gap`` are imported from
``dann_round5`` and ``bnm_round6``; nothing is reimplemented. The reconstructed slices are
asserted equal to the indices the runs recorded, so a drifted second implementation would
fail loudly rather than produce a comparable-looking number.

Inference only: no training, no optimizer step, no gradient.

Writes NOTHING outside --out.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

from bnm_round6 import (build_trainedon_probe, holdout_gap, probe_stats)   # noqa: E402
from dann_round5 import (build_target_holdout, materialise_holdout_images)  # noqa: E402
from integrity import verify as _verify_integrity                          # noqa: E402
from lens_dataset import LensDataset                                       # noqa: E402
from train_skeleton import TrainConfig, Trainer                            # noqa: E402

FOLDS = (1, 2, 3, 4)
SEED = 42
D_HOLDOUT_FRAC = 0.10
READ_OFF = 4000          # lambda=0's pre-registered budget; the only checkpoint retained

SPLITS = os.path.join(REPO, "data", "splits", "splits.json")
DATA_ROOT = os.path.join(REPO, "data")

# BNM's measured gaps, from runs/bnm_r6_folds_results.json, for the side-by-side.
BNM_RESULTS = os.path.join(REPO, "runs", "bnm_r6_folds_results.json")


def lambda0_dir(fold):
    return os.path.join(REPO, "runs", "baseline_unfrozen", f"fold{fold}", "extracted",
                        f"lambda0_r6_fold{fold}")


def build_trainer(fold, device):
    """Reconstruct the lambda=0 model at step 4000 from its retained resume checkpoint."""
    ckpt_path = os.path.join(lambda0_dir(fold), "resume", "latest.pt")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = dict(raw["cfg"])

    cfg = TrainConfig(
        splits_path=SPLITS, data_root=DATA_ROOT, fold=fold,
        head_type=c["head_type"], encoder_mode=c["encoder_mode"],
        n_blocks=int(c["n_blocks"]),
        init_head_from=None,          # the head comes from the checkpoint itself
        lr=c["lr"], weight_decay=c["weight_decay"], batch_size=c["batch_size"],
        seed=int(c["seed"]), device=str(device), amp=False,
        grad_checkpointing=False, verbose_encoder=False)
    tr = Trainer(cfg)
    tr.load(ckpt_path, map_location=device, load_training_state=False)
    tr.head.eval()
    return tr, raw


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    _verify_integrity(where="before the round-7 lambda=0 holdout-gap analysis")

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 100)
    print("ROUND 7, TASK 2 -- THE 704/78 DIAGNOSTIC ON THE lambda=0 ARM")
    print("*** RETROSPECTIVE AND UNREGISTERED -- no prediction exists and none is written ***")
    print("=" * 100)
    print(f"  device {device}   inference only, no training step of any kind")
    print(f"  read-off {READ_OFF} -- reading (1) of 7.27 ONLY. lambda=0 retained one")
    print("  checkpoint per fold, so readings (2) and (3) are not available on this arm.")
    print("  CODE PATH: bnm_round6.probe_stats / holdout_gap, dann_round5's holdout")
    print("  builder -- imported, not reimplemented; reconstructions are ASSERTED equal")
    print("  to the indices the runs recorded.")

    out = {"round": 7, "task": "704_78_diagnostic_cross_arm",
           "arm": "lambda=0 (gamma=0 / lambda=0 floor)",
           "status": "RETROSPECTIVE AND UNREGISTERED",
           "reading": "(1) read-off checkpoint only -- (2) and (3) unavailable, one "
                      "checkpoint retained per fold",
           "read_off_step": READ_OFF,
           "code_path": "bnm_round6.probe_stats / bnm_round6.holdout_gap, unchanged",
           "per_fold": {}}

    rows = []
    for f in FOLDS:
        seed = SEED + f
        tgt_ds = LensDataset(SPLITS, DATA_ROOT, role="target", fold=f, aug="weak",
                             n_views=1)
        holdout = build_target_holdout(tgt_ds, frac=D_HOLDOUT_FRAC, seed=seed)
        probe = build_trainedon_probe(tgt_ds, holdout, seed=seed)

        # --- the shared-path assertions ---
        rec_hold = set(int(i) for i in json.load(
            open(os.path.join(lambda0_dir(f), "run_config.json"),
                 encoding="utf-8"))["dann_d_holdout_indices"])
        got_hold = set(int(i) for i in holdout["holdout"])
        if got_hold != rec_hold:
            raise SystemExit(f"fold {f}: reconstructed holdout != the slice lambda=0 ran "
                             f"with. The comparison would be meaningless. STOP.")
        rec_probe = set(int(i) for i in json.load(
            open(os.path.join(REPO, "runs", "bnm_r6_folds", f"bnm_r6_fold{f}",
                              "run_config.json"),
                 encoding="utf-8"))["trainedon_probe"]["probe"])
        got_probe = set(int(i) for i in probe["probe"])
        if got_probe != rec_probe:
            raise SystemExit(f"fold {f}: reconstructed trained-on probe != BNM's. The two "
                             f"arms would be measured on different slices. STOP.")

        held_images = materialise_holdout_images(tgt_ds, holdout["holdout"])
        probe_images = materialise_holdout_images(tgt_ds, probe["probe"])

        tr, raw = build_trainer(f, device)
        with torch.no_grad():
            gap = holdout_gap(probe_stats(tr, probe_images),
                              probe_stats(tr, held_images))
        del tr
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"\n  fold {f}  (holdout and probe both byte-match the recorded slices)")
        print(f"    trained-on  ||A||* {gap['trained_on']['nuclear_norm']:.4f}   "
              f"entropy {gap['trained_on']['entropy']:.4f}   "
              f"pred-pos {gap['trained_on']['pred_pos_frac']:.4f}")
        print(f"    held-out    ||A||* {gap['held_out']['nuclear_norm']:.4f}   "
              f"entropy {gap['held_out']['entropy']:.4f}   "
              f"pred-pos {gap['held_out']['pred_pos_frac']:.4f}")
        print(f"    GAP         ||A||* {gap['nuclear_gap']:+.4f}   "
              f"entropy {gap['entropy_gap']:+.4f}   "
              f"pred-pos {gap['pred_pos_frac_gap']:+.4f}")
        out["per_fold"][str(f)] = gap
        rows.append(gap)

    nuc = [r["nuclear_gap"] for r in rows]
    ent = [r["entropy_gap"] for r in rows]
    pp = [r["pred_pos_frac_gap"] for r in rows]
    out["fold_means"] = {"nuclear_gap": st.mean(nuc), "nuclear_gap_sd": st.stdev(nuc),
                         "entropy_gap": st.mean(ent), "entropy_gap_sd": st.stdev(ent),
                         "pred_pos_frac_gap": st.mean(pp),
                         "pred_pos_frac_gap_sd": st.stdev(pp)}

    print("\n" + "=" * 100)
    print("FOLD-MEANS -- n = 4, the fold is the unit")
    print("=" * 100)
    print(f"  nuclear gap   per fold " + " ".join(f"{v:+.4f}" for v in nuc) +
          f"   mean {st.mean(nuc):+.4f}  sd {st.stdev(nuc):.4f}")
    print(f"  entropy gap   per fold " + " ".join(f"{v:+.4f}" for v in ent) +
          f"   mean {st.mean(ent):+.4f}  sd {st.stdev(ent):.4f}")
    print(f"  pred-pos gap  per fold " + " ".join(f"{v:+.4f}" for v in pp) +
          f"   mean {st.mean(pp):+.4f}  sd {st.stdev(pp):.4f}")

    if os.path.isfile(BNM_RESULTS):
        b = json.load(open(BNM_RESULTS, encoding="utf-8"))["diagnostic_704_78"]
        print("\n  SIDE BY SIDE with BNM, reading (1), read-off checkpoint:")
        print(f"    BNM x nb3   nuclear gap per fold " +
              " ".join(f"{v:+.4f}" for v in b["read_off"]["per_fold"]) +
              f"   mean {b['read_off']['mean']:+.4f}")
        print(f"    lambda=0    nuclear gap per fold " +
              " ".join(f"{v:+.4f}" for v in nuc) + f"   mean {st.mean(nuc):+.4f}")
        out["bnm_read_off_for_reference"] = b["read_off"]
        print("\n  NO SIGNIFICANCE CLAIM, NO CROSS-ARM ORDERING CLAIM. n = 4 per arm, the")
        print("  reads are retrospective, and the two arms sit at different read-off steps")
        print("  (BNM 1000, lambda=0 4000) because each is read where its own")
        print("  pre-registration says. Fold-means and per-fold values, and nothing more.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n[out] {os.path.relpath(os.path.abspath(a.out), REPO)}")

    _verify_integrity(where="after the round-7 lambda=0 holdout-gap analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
