"""
run_coral_gamma_sweep.py
========================
ONE gamma of the round-4 CORAL development sweep, as ONE process.

    nb3 (n_blocks=3), fold 0, 2000 steps, target EMA OFF, everything else identical to the
    pinned round-4 config. Three gammas -- 1, 4, 16 -- are three INDEPENDENT invocations of
    this script, each writing to its OWN output directory.

Separate processes and separate output paths are deliberate. The calibration was corrupted
by two processes racing on one output file (`runs/coral_round4_design.md` §9); this script
cannot repeat that, because the trace path is derived from the gamma and it REFUSES to start
if that path already holds a trace.

WHAT THIS IS FOR
----------------
The 500-step calibration showed magnitude-matching is ill-posed on the unfrozen arm: raw
CORAL scales as (feature scale)^4, the unfrozen encoder moves the scale, so "equalise the
losses at the end of training" has no fixed point. At gamma=1.0 the CORAL term is losing --
sup_loss falls, raw CORAL rises, the norm rises 25%. This sweep asks the only question that
makes an 8000-step run interpretable: **at what gamma does the CORAL term stop losing?**

The selection criterion is PRE-REGISTERED in ``coral_round4.SWEEP_CRITERION``, printed by
this script before the first optimizer step, and evaluated LOCALLY afterwards by
``analyze_coral_gamma_sweep.py``. It is not evaluated here, and nothing here can see it.

    ****  METRIC HYGIENE -- A HARD RULE, ENFORCED, NOT DECLARED  ****
    This sweep reads LOSS DYNAMICS and FEATURE STATISTICS ONLY. The whole run executes
    inside ``coral_round4.test_label_lockout()``, which makes TPR / AUROC / AUPRC /
    metric_suite / purity / the curve hook and every ``test_*`` cache split raise
    ``TestLabelAccess`` on contact. There is no curve_hook call, no logit dump and no
    memorization probe in this path -- those all read test splits. A gamma chosen here
    therefore cannot have been chosen on a test-set quantity.

    The ONE cache read is ``source_train``, for the feature standardizer, which is a fixed
    constant of the source domain and carries no target and no test information.

WHAT IT WRITES (all under ``--out-dir``, one directory per gamma)
-----------------------------------------------------------------
    trace.csv          one row per log interval: step, sup_loss, raw CORAL, cov/mean terms,
                       and PER DOMAIN mean ||f|| and mean per-dimension std -- window means
                       plus the instantaneous last value.
    steps.csv          the per-STEP series, unaggregated. The slope fit and the block
                       bootstrap want every step, not a 16-point summary; at 2000 rows this
                       is ~200 KB and there is no reason to throw it away.
    sweep.json         config + provenance + the abort record, written incrementally.
    checkpoints/       every ``--ckpt-every`` steps (default 250). Kaggle sessions time out;
                       a sweep arm that dies at step 1900 with no checkpoints is a total
                       loss, and at nb3 a checkpoint is ~59 MB, so 8 of them is ~475 MB.

Run (one gamma per process):
    python -u scripts/run_coral_gamma_sweep.py --gamma 1  --out-dir runs/coral_r4_sweep/g1
    python -u scripts/run_coral_gamma_sweep.py --gamma 4  --out-dir runs/coral_r4_sweep/g4
    python -u scripts/run_coral_gamma_sweep.py --gamma 16 --out-dir runs/coral_r4_sweep/g16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from coral_round4 import (SWEEP_CRITERION, SWEEP_STEPS, DEFAULT_FEAT_NORM_CEILING,
                          FeatureNormCeiling, check_feature_norm_ceiling,
                          DEFAULT_FEAT_NORM_FLOOR,
                          test_label_lockout, CoralStepWindow, CORAL_STEP_KEYS)
import run_coral_round4 as R4

TRACE_COLUMNS = (
    ["step", "n_steps_in_window", "wall_secs"]
    + [f"{k}_mean" for k in CORAL_STEP_KEYS]
    + [f"{k}_last" for k in ("coral_raw", "src_feat_l2", "tgt_feat_l2", "amp_scale")]
    + ["coral_over_sup", "src_l2_over_unit_gain"]
)

STEP_COLUMNS = ["step", "sup_loss", "da_loss", "coral_raw", "cov_term", "mean_term",
                "src_feat_l2", "src_perdim_std", "tgt_feat_l2", "tgt_perdim_std",
                "step_skipped", "amp_scale"]


def _append(path, columns, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        if new:
            w.writeheader()
        w.writerow({c: row.get(c) for c in columns})
        fh.flush()
        os.fsync(fh.fileno())          # a Kaggle session kill must not lose the last rows


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gamma", type=float, required=True,
                   help="the CONSTANT CORAL weight for this arm (sweep: 1, 4, 16)")
    p.add_argument("--out-dir", required=True,
                   help="OWN directory for this gamma -- never shared between processes")
    p.add_argument("--fold", type=int, default=0, help="development fold; 0 only")
    p.add_argument("--max-steps", type=int, default=SWEEP_STEPS)
    p.add_argument("--log-every", type=int, default=125)
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--feat-norm-ceiling", type=float, default=DEFAULT_FEAT_NORM_CEILING)
    p.add_argument("--feat-norm-floor", type=float, default=DEFAULT_FEAT_NORM_FLOOR)
    p.add_argument("--data-root", default=os.path.join(root, "data"))
    p.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    p.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    p.add_argument("--init-head-from", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--coral-target-ema", dest="coral_target_ema", action="store_true",
                   default=False, help="OFF by default in round 4 -- see the design doc")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    trace_csv = os.path.join(out_dir, "trace.csv")
    steps_csv = os.path.join(out_dir, "steps.csv")
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    print("=" * 92)
    print(f"ROUND-4 CORAL GAMMA SWEEP -- gamma = {args.gamma:g}, nb3, fold {args.fold}, "
          f"{args.max_steps} steps")
    print("=" * 92)
    print(SWEEP_CRITERION)
    print("=" * 92)
    print(f"  out-dir            : {out_dir}")
    print(f"  feature-norm ceiling: {args.feat_norm_ceiling:g}  (HARD ABORT above this)")
    print(f"  target EMA          : {'ON' if args.coral_target_ema else 'OFF (round-4 default)'}")
    print("=" * 92, flush=True)

    if os.path.exists(trace_csv):
        raise SystemExit(
            f"[sweep] ABORT: {trace_csv} already exists. Appending would interleave two "
            f"runs' rows into one file -- which is exactly the concurrent-write failure the "
            f"calibration hit. Move or delete {out_dir} first, or point --out-dir "
            f"somewhere else.")
    if args.dry_run:
        print("[dry-run] criterion printed, config validated; nothing trained.")
        return 0

    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- build the arm through the SAME code path the real runner uses ----
    # Reusing R4.build_arm is the point: the sweep must train the pinned configuration, not
    # an approximation of it. `for_calibration=True` only bypasses the --gamma requirement
    # in build_config; the weight actually used is set below.
    sweep_args = argparse.Namespace(
        arm="coral_nb3", fold=args.fold, gamma=args.gamma,
        data_root=args.data_root, splits=args.splits, cache_dir=args.cache_dir,
        agreement_ref=R4.DEFAULT_AGREEMENT_REF, init_head_from=args.init_head_from,
        max_steps=args.max_steps, allow_off_budget=True,
        curve_every=args.log_every, ckpt_every=args.ckpt_every,
        backbone_lr=R4.BACKBONE_LR, fp16=bool(args.fp16), eval_fp16=False,
        grad_checkpointing=True, coral_target_ema=bool(args.coral_target_ema),
        num_workers=int(args.num_workers), device=args.device,
        target_fpr=1e-3, step0_tol=0.0, smoke_batch_size=None, dir_suffix=None,
        dry_run=False, emit_config=None, reference_config=None,
        calibrate=False, calibrate_steps=0, calibrate_tail=0, grad_check=False,
        feat_norm_ceiling=args.feat_norm_ceiling)

    t_start = time.time()
    record = {"gamma": float(args.gamma), "n_blocks": 3, "fold": int(args.fold),
              "max_steps": int(args.max_steps), "log_every": int(args.log_every),
              "ckpt_every": int(args.ckpt_every),
              "feat_norm_ceiling": float(args.feat_norm_ceiling),
              "feat_norm_floor": float(args.feat_norm_floor),
              "coral_target_ema": bool(args.coral_target_ema),
              "criterion": SWEEP_CRITERION,
              "metric_hygiene": ("ran entirely inside test_label_lockout(); no TPR, AUROC, "
                                 "AUPRC, metric_suite, curve_hook or test_* cache split "
                                 "was reachable at any point"),
              "aborted": False, "abort_reason": None, "steps_completed": 0}

    # ****  THE HARD RULE  ****
    with test_label_lockout():
        A = R4.build_arm("coral_nb3", sweep_args, root, for_calibration=True)
        trainer, coral = A["trainer"], A["coral"]
        trainer.cfg.coral_weight = float(args.gamma)
        assert float(trainer.cfg.coral_weight) == float(args.gamma)
        assert coral.target_ema == bool(args.coral_target_ema), \
            "the CORAL module's target_ema does not match the requested setting"

        record.update({"init_head_from": A["cfg_json"]["init_head_from"],
                       "init_head_l2": A["cfg_json"]["init_head_l2"],
                       "batch_size": A["batch_size"], "lr": R4.LR,
                       "backbone_lr_effective": A["cfg_json"]["backbone_lr_effective"],
                       "lr_decay": R4.LR_DECAY, "weight_decay": R4.WEIGHT_DECAY,
                       "coral_shrinkage": R4.CORAL_SHRINKAGE,
                       "coral_include_mean": R4.CORAL_INCLUDE_MEAN,
                       "precision": A["cfg_json"]["precision"],
                       "coral_loss_precision": "fp32_autocast_disabled",
                       "device": A["device"], "steps_per_epoch": A["steps"]})
        with open(os.path.join(out_dir, "sweep.json"), "w") as fh:
            json.dump(record, fh, indent=2)

        window = CoralStepWindow()
        gstep = 0
        stop = False
        print(f"\n[train] gamma={args.gamma:g} starting ...", flush=True)
        try:
            for epoch in range(1, A["epochs_planned"] + 1):
                A["src_loader"].batch_sampler.set_epoch(epoch)
                A["tgt_loader"].batch_sampler.set_epoch(epoch)
                trainer.head.train()
                for sbatch, tbatch in A["paired"]:
                    scale_before = (trainer.scaler.get_scale()
                                    if trainer._amp_enabled else None)
                    st = trainer.train_step(sbatch, tbatch, step=gstep,
                                            total_steps=args.max_steps)
                    skipped = bool(trainer._amp_enabled
                                   and trainer.scaler.get_scale() < scale_before)
                    cs = coral.step_stats()
                    stats = {**st, **cs, "step_skipped": float(skipped),
                             "amp_scale": (float(trainer.scaler.get_scale())
                                           if trainer._amp_enabled else float("nan"))}
                    gstep += 1
                    window.update(stats)
                    _append(steps_csv, STEP_COLUMNS, {"step": gstep, **stats})

                    # ---- CHANGE 2: the hard feature-norm abort ----
                    check_feature_norm_ceiling(gstep, coral.last_scale,
                                               args.feat_norm_ceiling, cs["coral_raw"],
                                               floor=args.feat_norm_floor)

                    if gstep % int(args.ckpt_every) == 0:
                        trainer.save(os.path.join(ckpt_dir, f"step_{gstep}.pt"),
                                     full_state=False,
                                     schedule={"coral_weight": float(args.gamma),
                                               "global_step": int(gstep)})
                    if gstep % int(args.log_every) == 0:
                        row = window.row(gstep)
                        row["wall_secs"] = round(time.time() - t_start, 1)
                        _append(trace_csv, TRACE_COLUMNS, row)
                        window.reset()
                        print(f"    [step {gstep:>5}/{args.max_steps}] "
                              f"sup={row['sup_loss_mean']:.5f} "
                              f"coral_raw={row['coral_raw_mean']:.6g} "
                              f"src|f|={row['src_feat_l2_mean']:.3f} "
                              f"tgt|f|={row['tgt_feat_l2_mean']:.3f} "
                              f"src_sd={row['src_perdim_std_mean']:.4f} "
                              f"({row['wall_secs']:.0f}s)", flush=True)
                    if gstep >= args.max_steps:
                        stop = True
                        break
                if stop:
                    break
        except FeatureNormCeiling as e:
            record["aborted"] = True
            record["abort_reason"] = str(e)
            record["abort_step"] = int(gstep)
            print("\n" + "!" * 92, flush=True)
            print(f"[ABORT] {e}")
            print("!" * 92, flush=True)
            trainer.save(os.path.join(ckpt_dir, f"step_{gstep}_ABORT.pt"),
                         full_state=False,
                         schedule={"coral_weight": float(args.gamma),
                                   "global_step": int(gstep), "aborted": True})

        record["steps_completed"] = int(gstep)
        record["wall_minutes"] = (time.time() - t_start) / 60.0
        record["peak_gpu_gb"] = (torch.cuda.max_memory_allocated() / 2 ** 30
                                 if torch.cuda.is_available() else None)
        with open(os.path.join(out_dir, "sweep.json"), "w") as fh:
            json.dump(record, fh, indent=2)

    print(f"\n[done] gamma={args.gamma:g}  steps={record['steps_completed']}  "
          f"aborted={record['aborted']}  {record['wall_minutes']:.1f} min")
    print(f"[out] {trace_csv}\n[out] {steps_csv}\n[out] "
          f"{os.path.join(out_dir, 'sweep.json')}")
    print("[next] download the sweep directories and run LOCALLY:\n"
          "       python scripts/analyze_coral_gamma_sweep.py --sweep-dir runs/coral_r4_sweep")
    return 0


if __name__ == "__main__":
    sys.exit(main())
