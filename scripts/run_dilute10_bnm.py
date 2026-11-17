"""
run_dilute10_bnm.py
===================
BNM x nb3 under the 1:10 diluted target (brief_dilute10_code.md section 4). A WRAPPER:
it calls ``run_bnm_r6.main`` / ``run_bnm_r6.build_arm`` -- the round-6 folds' own
construction, hook, gamma invariant and training loop -- with these names rebound inside
``run_bnm_r6``:

  LensDataset               target role served from the diluted pool  (--target-pool dilute10)
  build_target_holdout      the original runs' 78 objects, pinned      (--target-pool dilute10)
  build_trainedon_probe     the ORIGINAL size-matched probe (same 78 objects; they are still
                            trained-on in the diluted pool)            (--target-pool dilute10)
  ClassBalancedBatchSampler the TARGET call -> UniformTargetBatchSampler (--target-sampler uniform)
  build_config              + the ``dilute10`` provenance block (always)

gamma 0.00787, trained to 4000, gamma invariant kept (the runner's own GammaInvariant).
Output: ``runs/dilute10/bnm/fold{k}/``; an existing directory is refused without --force.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402
import dilute10_pool as DP                                   # noqa: E402

ARM = "bnm"
GAMMA = 0.00787
BUDGET = 4000


def base_argv(args):
    a = ["--fold", str(args.fold), "--gamma", repr(GAMMA), "--max-steps", str(BUDGET),
         "--num-workers", str(args.num_workers)]
    if args.device:
        a += ["--device", args.device]
    if args.input_root:
        a += ["--input-root", args.input_root]
    return a


def _counts(holdout, n_lenses):
    tp = (holdout["train_pool"] if "train_pool" in holdout
          else np.concatenate([holdout["train_pos"], holdout["train_neg"]]))
    tp = np.asarray(tp)
    return {"lenses": int((tp < n_lenses).sum()), "galaxies": int((tp >= n_lenses).sum()),
            "holdout": int(holdout["n_holdout"])}


def patched(args, B, ps, *, recorder=None):
    import bnm_round6
    fold = int(args.fold)
    pool = C.get_pool(args)
    sp = DP.load_splits()
    orig = next(f for f in sp["real_lenses_kfold"]["folds"] if int(f["fold"]) == fold)
    n_l = len(orig["target_lens_rows"])
    stub = DP._LatentStub(n_l, len(orig["target_neg_object_ids"]))
    orig_holdout = B.build_target_holdout(stub, frac=B.D_HOLDOUT_FRAC, seed=42 + fold)
    holdout = None
    if args.target_pool == "dilute10":
        holdout = DP.pinned_holdout(pool, fold)
        if not np.array_equal(np.sort(holdout["holdout"]), np.sort(orig_holdout["holdout"])):
            raise AssertionError("pinned holdout positions differ from the original draw")
        ps.set(B, "LensDataset", DP.lens_dataset_factory(B.LensDataset, pool, fold),
               "target role -> diluted pool")
        ps.set(B, "build_target_holdout", lambda tgt_ds, frac=0.1, seed=None: holdout,
               "the original 78 objects, pinned")
        orig_probe = bnm_round6.build_trainedon_probe(stub, orig_holdout, seed=42 + fold)
        ps.set(B, "build_trainedon_probe",
               lambda tgt_ds, h, seed=None: dict(orig_probe),
               "the original probe (same objects; positions preserved)")
    counts = _counts(holdout if holdout is not None else orig_holdout, n_l)
    if args.target_sampler == "uniform" or recorder is not None:
        ps.set(B, "ClassBalancedBatchSampler", C.direct_sampler_ctor(
            B.ClassBalancedBatchSampler, uniform=args.target_sampler == "uniform",
            target_seed=42 + fold + 1000, n_lenses=n_l, csv_path=C.logger_csv(args, ARM),
            recorder=recorder,
            expect_pool=None if holdout is None else holdout["train_pool"]),
            "TARGET call -> uniform sampler" if args.target_sampler == "uniform"
            else "recording only")
    orig_bc = B.build_config

    def build_config(*a, **kw):
        cfg = orig_bc(*a, **kw)
        cfg["dilute10"] = C.provenance(
            args, arm="bnm_nb3", pool=pool, counts=counts, abort_mode=cfg.get("abort_mode"),
            patches=ps, extra={"budget": BUDGET, "gamma": GAMMA,
                               "holdout_identity_sha256": None if holdout is None
                               else holdout["holdout_identity_sha256"]})
        return cfg
    ps.set(B, "build_config", build_config, "+ dilute10 provenance")


def construct(args, *, recorder=None):
    import copy
    args = copy.copy(args)
    args.dry_run = True                       # never create a logger file when constructing
    import run_bnm_r6 as B
    import live_encode, zoobot_encoder  # noqa: F401
    bargs = B.build_argparser().parse_args(base_argv(args))
    paths = B.resolve_paths(bargs, os.path.dirname(HERE))
    paths.setdefault("init_head_from", B.DEFAULT_INIT_HEAD)
    with C.PatchSet() as ps:
        patched(args, B, ps, recorder=recorder)
        A = B.build_arm(bargs, paths, verbose=False)
    return A, B


def main(argv=None):
    p = C.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]),
                          num_workers_default=2)
    args = p.parse_args(argv)
    C.refuse_zero_workers(args)
    import run_bnm_r6 as B
    import live_encode, zoobot_encoder  # noqa: F401

    if args.dry_run:
        A, _ = construct(args, recorder=C.Recorder())
        print(json.dumps({"tgt_first_batch_head":
                          C.first_batches(A["paired"].target_loader.batch_sampler, 1)[0][:8]}))
        if args.emit_config:
            with open(args.emit_config, "w") as fh:
                json.dump(A["cfg"], fh, indent=2, default=str)
        print("[dilute10-bnm] dry run: arm constructed, nothing trained.")
        return 0

    bargv = base_argv(args)
    if args.emit_config:
        bargv += ["--emit-config", os.path.abspath(args.emit_config)]
    else:
        out = C.out_dir_for(args, ARM)
        C.refuse_existing(out, args.force)
        os.makedirs(out, exist_ok=True)
        bargv += ["--out-dir", out]
    rec = C.Recorder() if args.target_sampler == "uniform" else None
    out_or_none = None if args.emit_config else C.out_dir_for(args, ARM)
    with C.logger_summary_on_exception(rec, lambda: out_or_none):
        with C.PatchSet() as ps:
            patched(args, B, ps, recorder=rec)
            rc = B.main(bargv)
    if out_or_none is not None:
        completed, reason = C.run_completion(out_or_none, budget=BUDGET, steps_json="run.json")
        C.write_logger_summary(rec, out_or_none, completed=completed, reason=reason)
    if not args.emit_config:
        C.post_run_check(arm="bnm_nb3", out_dir=C.out_dir_for(args, ARM), budget=BUDGET,
                         ckpt_every=250, logit_subdir=f"logits/bnm_nb3_fold{args.fold}",
                         steps_json="run.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
