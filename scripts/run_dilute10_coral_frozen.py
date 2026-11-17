"""
run_dilute10_coral_frozen.py
============================
The CORAL FROZEN stage rerun under the 1:10 diluted target (part 1 section 4.2;
brief_dilute10_code.md section 4). A WRAPPER around ``run_coral.train_coral_fold`` called with
EXACTLY the arguments of the round-1 run CORAL x nb3 started from -- ``run_bakeoff_coral``,
the ``warmup_cosine`` run (fresh linear head, seed 42, cached features, CPU, fp32, peak gamma
1.0, warm-up 0.1, floor 0.1, target EMA 0.9, shrinkage 0.1, mean included, 43 epochs, R1 =
step 8000) -- with these names rebound:

  run_coral._guarded_load            target_fold{k} read from data/feature_cache_dilute10/
                                     (the diluted pool's features)       (--target-pool dilute10)
  run_coral.ClassBalancedBatchSampler the TARGET call -> uniform sampler (--target-sampler uniform)
  curve_hook.append_curve_row        SCORING ONLY (logits persisted, no metric) -- always

Writes ``checkpoints/step_8000.pt`` and ``gate_values.json`` (TPR@1e-3 and AUROC of that head
on fold k, cached path; its L2 norm and sha256). Output: ``runs/dilute10/coral_frozen/fold{k}/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402

ARM = "coral_frozen"
DILUTE_CACHE = os.path.join(REPO, "data", "feature_cache_dilute10")
ROUND1_CONFIG = os.path.join(REPO, "runs", "bakeoff_coral", "run_config.json")


class _Constructed(Exception):
    pass


def round1_cfg(BC):
    return {"budget_steps": BC.BUDGET_STEPS, "epochs_trained": BC.EPOCHS,
            "steps_per_epoch": 188, "curve_every": BC.CURVE_EVERY,
            "batch_size": BC.BATCH_SIZE, "lr": BC.LR, "weight_decay": BC.WEIGHT_DECAY,
            "pos_fraction": BC.POS_FRACTION, "seed": BC.SEED, "head_type": BC.HEAD_TYPE,
            "selection_rule": "R1_last_checkpoint_step8000", "curve_folds": BC.ALL_FOLDS,
            "target_fpr": BC.DEFAULT_TARGET_FPR, "method": "deep_coral",
            "domain_adaptation": "coral", "coral_peak_gamma": BC.CORAL_PEAK_GAMMA,
            "coral_weight_schedule": BC.CORAL_SCHEDULE,
            "coral_warmup_frac": BC.CORAL_WARMUP_FRAC, "coral_min_frac": BC.CORAL_MIN_FRAC,
            "coral_shrinkage": BC.CORAL_SHRINKAGE,
            "coral_include_mean": BC.CORAL_INCLUDE_MEAN,
            "coral_target_ema": BC.CORAL_TARGET_EMA,
            "coral_ema_momentum": BC.CORAL_EMA_MOMENTUM,
            "stop_on_blowup": BC.STOP_ON_BLOWUP, "agreement_ref": None,
            "folds_this_invocation": None}


_LAST = {}


def run(args, *, dry=False):
    import curve_hook
    import evaluate as ev
    import run_bakeoff_coral as BC
    import run_coral as RC
    import metrics  # noqa: F401
    fold = int(args.fold)
    pool = C.get_pool(args)
    n_l = C.lens_count(pool, args, fold)
    out = C.out_dir_for(args, ARM)
    ckpt_dir = os.path.join(out, "checkpoints")
    if dry:                     # train_coral_fold makes ckpt_dir before the dry-run stop
        import tempfile
        ckpt_dir = os.path.join(tempfile.mkdtemp(prefix="dilute10_dry_"), "checkpoints")
    cache = ev.FeatureCache(os.path.abspath(args.cache_dir))
    rec = C.Recorder()
    _LAST["rec"] = rec          # for main's exception-path summary
    identity = {}
    with C.PatchSet() as ps:
        if args.target_pool == "dilute10":
            dil = ev.FeatureCache(os.path.abspath(args.dilute_cache_dir))
            orig_gl = RC._guarded_load
            want = f"target_fold{fold}"

            def guarded_load(c, name, f):
                if name == want:
                    d = orig_gl(dil, name, f)
                    if len(d["features"]) != n_l + int(C.DP.pool_fold(pool, fold)
                                                       ["n_negatives"]):
                        raise SystemExit(f"[dilute10-coral-frozen] {name} in "
                                         f"{dil.dir} is not the diluted pool")
                    if not dry:
                        identity.update(C.check_fold_identity(
                            ARM, fold, target_ids=d["ids"],
                            cache_dir=os.path.abspath(args.cache_dir)))
                    return d
                return orig_gl(c, name, f)
            ps.set(RC, "_guarded_load", guarded_load,
                   f"{want} from {os.path.relpath(dil.dir, REPO)}")
        if args.target_sampler == "uniform" or dry:
            ps.set(RC, "ClassBalancedBatchSampler", C.direct_sampler_ctor(
                RC.ClassBalancedBatchSampler, uniform=args.target_sampler == "uniform",
                target_seed=BC.SEED + fold + 1000,
                n_lenses=n_l if args.target_sampler == "uniform" else None,
                csv_path=None if dry else os.path.join(out, "target_lens_fraction.csv"),
                recorder=rec),
                "TARGET call -> uniform sampler" if args.target_sampler == "uniform"
                else "recording only")
        ps.set(curve_hook, "append_curve_row", C.scoring_only_curve_row(
            curve_hook.append_curve_row, gate_steps=(),
            dump_dir=None if dry else os.path.join(out, "logits"), fold=fold),
            "measurement lockout: scoring only")
        if dry:
            def stop(*a, **kw):
                raise _Constructed()
            ps.set(RC, "coral_gamma_at_step", stop, "dry run: stop after the samplers")
        cfg = round1_cfg(BC)
        cfg["folds_this_invocation"] = [fold]
        cfg["dilute10"] = C.provenance(
            args, arm=ARM, pool=pool, abort_mode=None, patches=ps,
            extra={"device": "cpu", "precision": "fp32",
                   "measurement_lockout": "curve rows scored, not measured; agreement "
                                          "column not computed",
                   "dilute_cache_dir": (os.path.relpath(os.path.abspath(
                       args.dilute_cache_dir), REPO).replace("\\", "/")
                       if args.target_pool == "dilute10" else None),
                   "gate_values_file": C.GATE_FILE})
        if not dry:
            os.makedirs(ckpt_dir, exist_ok=True)
        t0 = time.time()
        try:
            RC.train_coral_fold(
                cache, fold, ckpt_dir, head_type=BC.HEAD_TYPE, epochs=(BC.EPOCHS,),
                batch_size=BC.BATCH_SIZE, pos_fraction=BC.POS_FRACTION, lr=BC.LR,
                weight_decay=BC.WEIGHT_DECAY, coral_weight=BC.CORAL_PEAK_GAMMA,
                coral_include_mean=BC.CORAL_INCLUDE_MEAN,
                coral_shrinkage=BC.CORAL_SHRINKAGE, coral_target_ema=BC.CORAL_TARGET_EMA,
                coral_ema_momentum=BC.CORAL_EMA_MOMENTUM,
                coral_weight_schedule=BC.CORAL_SCHEDULE,
                coral_warmup_frac=BC.CORAL_WARMUP_FRAC, coral_min_frac=BC.CORAL_MIN_FRAC,
                stop_on_blowup=BC.STOP_ON_BLOWUP, eval_every_steps=BC.CURVE_EVERY,
                curve_csv=os.path.join(out, "curve.csv"), curve_folds=BC.ALL_FOLDS,
                agreement_refs=None, eval_target_fpr=BC.DEFAULT_TARGET_FPR,
                eval_device="cpu", seed=BC.SEED, device="cpu", verbose=True)
        except _Constructed:
            pass
    samplers = {seed: s for (_k, seed, s) in rec.built}
    tgt = samplers.get(BC.SEED + fold + 1000)
    cfg["dilute10"]["realised_training_pool"] = _counts(tgt, n_l) if tgt is not None else None
    if dry:
        return cfg, samplers
    cfg["wall_minutes"] = (time.time() - t0) / 60
    cfg["dilute10"]["fold_identity"] = identity or None
    r1 = os.path.join(ckpt_dir, f"step_{BC.BUDGET_STEPS}.pt")
    if not os.path.isfile(r1):
        raise SystemExit(f"[dilute10-coral-frozen] R1 checkpoint {r1} was not written")
    C.write_gate_values(out, fold, r1, os.path.abspath(args.cache_dir))
    cfg["gate_values_file"] = {"path": C.GATE_FILE,          # values stay in that file
                               "sha256": C.DP.sha256_file(os.path.join(out, C.GATE_FILE))}
    with open(os.path.join(out, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    C.write_logger_summary(rec, out)
    return cfg, samplers


def _counts(s, n_l):
    inner = getattr(s, "inner", s)
    pool = getattr(inner, "pool", None)
    if pool is None:
        pool = list(inner.pos) + list(inner.neg)
    pool = np.asarray(pool)
    return {"lenses": int((pool < n_l).sum()), "galaxies": int((pool >= n_l).sum()),
            "holdout": 0}


def build_parser():
    p = C.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]),
                          num_workers_default=0)
    p.add_argument("--cache-dir", default=os.path.join(REPO, "data", "feature_cache"))
    p.add_argument("--dilute-cache-dir", default=DILUTE_CACHE)
    # --num-workers is NOT APPLICABLE here: train_coral_fold builds no DataLoader (its
    # samplers index cached feature tensors directly), so there is no worker RNG to couple
    # (part 2 D.7, declared). The common flag is accepted and ignored.
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run or args.emit_config:
        cfg, _ = run(args, dry=True)
        if args.emit_config:
            with open(args.emit_config, "w") as fh:
                json.dump(cfg, fh, indent=2, default=str)
        print("[dilute10-coral-frozen] constructed; nothing trained.")
        return 0
    out = C.out_dir_for(args, ARM)
    C.refuse_existing(out, args.force)
    os.makedirs(out, exist_ok=True)
    # run() writes the summary itself, ENFORCED, only after its R1 checkpoint exists (a
    # completed budget); any exception before that writes it unenforced (PROJECTv7 7.47)
    with C.logger_summary_on_exception(lambda: _LAST.get("rec"), lambda: out):
        run(args)
    C.post_run_check(arm=ARM, out_dir=out, budget=8000, ckpt_every=250, logit_subdir="logits",
                     r1_ckpt="checkpoints/step_8000.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
