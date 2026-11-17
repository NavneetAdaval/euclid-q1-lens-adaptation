"""
run_dilute10_freematch_frozen.py
================================
The FreeMatch FROZEN stage rerun under the 1:10 diluted target (part 1 section 4.2;
brief_dilute10_code.md section 4). A WRAPPER around ``run_freematch.train_freematch_fold``
called with EXACTLY round 1's arguments -- the constants of ``run_bakeoff_freematch``
(fresh linear head, seed 42, weak augmentation, fp16, image space with the frozen encoder in
the loop, 8000-step budget over 43 epochs, R1 = step 8000) -- with these names rebound:

  run_freematch.LensDataset          target role -> diluted pool        (--target-pool dilute10)
  run_freematch.make_balanced_loader TARGET loader -> uniform sampler   (--target-sampler uniform)
  curve_hook.append_curve_row        SCORING ONLY (logits persisted, no metric) -- always

The curve hook uses no RNG and its rows are only printed, so the scoring-only hook changes no
training quantity; the round-1 agreement column is not computed (it would need Baseline A on
the diluted target pool and never touches the loss).

At the end it writes the final head (``checkpoints/step_8000.pt``) and that head's own
per-fold step-0 gate values -- TPR@1e-3 and AUROC on the cached path, the head's L2 norm and
sha256 -- to ``gate_values.json``. That is this stage's one test-metric read, at run time.
Output: ``runs/dilute10/freematch_frozen/fold{k}/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402
import dilute10_pool as DP                                   # noqa: E402

ARM = "freematch_frozen"
ROUND1_CONFIG = os.path.join(REPO, "runs", "bakeoff_freematch_5fold", "run_config.json")


class _Constructed(Exception):
    """Raised by --dry-run right after the loaders exist, before any model is built."""


def round1_cfg(BF, args, device):
    """Round 1's run_config dict, from run_bakeoff_freematch's own constants."""
    return {"budget_steps": BF.BUDGET_STEPS, "epochs_trained": BF.EPOCHS,
            "steps_per_epoch": 187, "curve_every": BF.CURVE_EVERY,
            "batch_size": BF.BATCH_SIZE, "lr": BF.LR, "weight_decay": BF.WEIGHT_DECAY,
            "pos_fraction": BF.POS_FRACTION, "seed": BF.SEED, "head_type": BF.HEAD_TYPE,
            "selection_rule": "R1_last_checkpoint_step8000", "curve_folds": BF.ALL_FOLDS,
            "target_fpr": BF.DEFAULT_TARGET_FPR, "method": "freematch",
            "domain_adaptation": "freematch",
            "local_threshold_mode": BF.FREEMATCH_LOCAL_MODE,
            "freematch_variant": f"FreeMatch-{BF.FREEMATCH_LOCAL_MODE} (screen-winning; "
                                 f"floor is a documented negative)",
            "mu_max": BF.MU_MAX, "saf_weight": BF.SAF_WEIGHT,
            "sat_ema_momentum": BF.SAT_EMA_MOMENTUM,
            "consistency_warmup_frac": BF.CONSISTENCY_WARMUP_FRAC,
            "fairness_target": BF.FAIRNESS_TARGET, "pos_class": BF.POS_CLASS,
            "device": device, "fp16": bool(args.fp16), "num_workers": int(args.num_workers),
            "agreement_ref": None, "folds_this_invocation": [int(args.fold)]}


_LAST = {}


def run(args, *, dry=False):
    import curve_hook
    import run_bakeoff_freematch as BF
    import run_freematch as RF
    import torch
    import live_encode, zoobot_encoder, metrics, evaluate  # noqa: F401
    fold = int(args.fold)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pool = C.get_pool(args)
    n_l = C.lens_count(pool, args, fold)
    out = C.out_dir_for(args, ARM)
    ckpt_dir = os.path.join(out, "checkpoints")
    rec = C.Recorder()
    _LAST["rec"] = rec          # for main's exception-path summary
    with C.PatchSet() as ps:
        identity = {}
        if args.target_pool == "dilute10":
            fac = DP.lens_dataset_factory(RF.LensDataset, pool, fold)

            def checked(*a, **kw):
                ds = fac(*a, **kw)
                if getattr(ds, "dilute10", None) and not dry:
                    from cache_features import dataset_ids
                    identity.update(C.check_fold_identity(
                        ARM, fold, target_ids=dataset_ids(ds),
                        cache_dir=os.path.abspath(args.cache_dir)))
                return ds
            ps.set(RF, "LensDataset", checked,
                   "target role -> diluted pool; fold-identity check (part 2 D.6)")
        if args.target_sampler == "uniform" or dry:
            ps.set(RF, "make_balanced_loader", C.loader_ctor(
                RF.make_balanced_loader, uniform=args.target_sampler == "uniform",
                n_lenses=n_l if args.target_sampler == "uniform" else None,
                csv_path=None if dry else os.path.join(out, "target_lens_fraction.csv"),
                recorder=rec),
                "TARGET loader -> uniform sampler" if args.target_sampler == "uniform"
                else "recording only")
        ps.set(curve_hook, "append_curve_row", C.scoring_only_curve_row(
            curve_hook.append_curve_row, gate_steps=(),
            dump_dir=None if dry else os.path.join(out, "logits"), fold=fold),
            "measurement lockout: scoring only")
        if dry:
            def stop(*a, **kw):
                raise _Constructed()
            ps.set(RF, "TrainConfig", stop, "dry run: stop after the loaders")
        cfg = round1_cfg(BF, args, device)
        counts = None
        cfg["dilute10"] = C.provenance(
            args, arm=ARM, pool=pool, counts=counts, abort_mode=None, patches=ps,
            extra={"measurement_lockout": "curve rows scored, not measured; agreement "
                                          "column not computed",
                   "gate_values_file": C.GATE_FILE})
        encoder = None
        if not dry:
            from zoobot_encoder import ZoobotEncoder
            encoder = ZoobotEncoder(encoder_mode="frozen", device=device, verbose=False)
            os.makedirs(ckpt_dir, exist_ok=True)
        t0 = time.time()
        try:
            RF.train_freematch_fold(
                os.path.abspath(args.splits), os.path.abspath(args.data_root),
                os.path.abspath(args.cache_dir), fold, ckpt_dir, encoder=encoder,
                head_type=BF.HEAD_TYPE, epochs=(BF.EPOCHS,), batch_size=BF.BATCH_SIZE,
                pos_fraction=BF.POS_FRACTION, lr=BF.LR, weight_decay=BF.WEIGHT_DECAY,
                mu_max=BF.MU_MAX, saf_weight=BF.SAF_WEIGHT,
                sat_ema_momentum=BF.SAT_EMA_MOMENTUM,
                consistency_warmup_frac=BF.CONSISTENCY_WARMUP_FRAC,
                local_threshold_mode=BF.FREEMATCH_LOCAL_MODE, pos_class=BF.POS_CLASS,
                fairness_target=BF.FAIRNESS_TARGET, seed=BF.SEED, device=device,
                num_workers=int(args.num_workers), fp16=bool(args.fp16),
                eval_every_steps=BF.CURVE_EVERY, curve_csv=os.path.join(out, "curve.csv"),
                curve_folds=BF.ALL_FOLDS, agreement_refs=None,
                eval_target_fpr=BF.DEFAULT_TARGET_FPR, eval_device="cpu", verbose=True)
        except _Constructed:
            pass
    loaders = {k: v for (kind, k, v) in rec.built}
    if dry:
        tgt = loaders[True].batch_sampler
        cfg["dilute10"]["realised_training_pool"] = _counts(tgt, n_l)
        return cfg, loaders
    cfg["dilute10"]["realised_training_pool"] = (
        _counts(loaders[True].batch_sampler, n_l) if True in loaders else None)
    cfg["wall_minutes"] = (time.time() - t0) / 60
    cfg["dilute10"]["fold_identity"] = identity or None
    r1 = os.path.join(ckpt_dir, f"step_{BF.BUDGET_STEPS}.pt")
    if not os.path.isfile(r1):
        raise SystemExit(f"[dilute10-fm-frozen] R1 checkpoint {r1} was not written")
    C.write_gate_values(out, fold, r1, os.path.abspath(args.cache_dir))
    cfg["gate_values_file"] = {"path": C.GATE_FILE,          # values stay in that file
                               "sha256": C.DP.sha256_file(os.path.join(out, C.GATE_FILE))}
    cfg["n_logit_files"] = len([f for f in os.listdir(os.path.join(out, "logits"))
                                if f.startswith("step_")])
    with open(os.path.join(out, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    C.write_logger_summary(rec, out)
    return cfg, loaders


def _counts(batch_sampler, n_l):
    inner = getattr(batch_sampler, "inner", batch_sampler)
    pool = getattr(inner, "pool", None)
    if pool is None:                                   # balanced: both latent pools
        pool = list(inner.pos) + list(inner.neg)
    import numpy as np
    pool = np.asarray(pool)
    return {"lenses": int((pool < n_l).sum()), "galaxies": int((pool >= n_l).sum()),
            "holdout": 0}


def build_parser():
    p = C.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]),
                          num_workers_default=4)
    p.add_argument("--splits", default=os.path.join(REPO, "data", "splits", "splits.json"))
    p.add_argument("--data-root", default=os.path.join(REPO, "data"))
    p.add_argument("--cache-dir", default=os.path.join(REPO, "data", "feature_cache"))
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    C.refuse_zero_workers(args)
    if args.dry_run or args.emit_config:
        cfg, loaders = run(args, dry=True)
        if args.emit_config:
            with open(args.emit_config, "w") as fh:
                json.dump(cfg, fh, indent=2, default=str)
        print("[dilute10-fm-frozen] constructed; nothing trained.")
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
