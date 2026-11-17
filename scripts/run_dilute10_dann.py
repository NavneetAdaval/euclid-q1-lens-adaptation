"""
run_dilute10_dann.py
====================
DANN x nb3 under the 1:10 diluted target (brief_dilute10_code.md section 4). A WRAPPER:
it calls ``run_dann_r5.main`` / ``run_dann_r5.build_arm`` -- the confirmatory folds' own
construction, loss and training loop -- with these names rebound inside ``run_dann_r5``:

  LensDataset               target role served from the diluted pool  (--target-pool dilute10)
  build_target_holdout      the original runs' 78 objects, pinned      (--target-pool dilute10)
  ClassBalancedBatchSampler the TARGET call -> UniformTargetBatchSampler (--target-sampler uniform)
  build_config              + the ``dilute10`` provenance block (always)

Arm ``dann_nb3``, budget 4000, ``--abort-mode armed`` exactly as DANN's confirmatory folds
ran (conditions 1-2 armed, 3-4 advisory by the runner's own construction). Output:
``runs/dilute10/dann/fold{k}/``; an existing directory is refused without ``--force``.

    python -u scripts/run_dilute10_dann.py --fold 1 --target-pool dilute10 --target-sampler uniform
    python -u scripts/run_dilute10_dann.py --fold 1 --emit-config cfg.json      # nothing trained
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402
import dilute10_pool as DP                                   # noqa: E402

ARM = "dann"
BUDGET = 4000
ABORT_MODE = "armed"


def base_argv(args):
    a = ["--arm", "dann_nb3", "--fold", str(args.fold), "--max-steps", str(BUDGET),
         "--abort-mode", ABORT_MODE, "--num-workers", str(args.num_workers)]
    if args.device:
        a += ["--device", args.device]
    if args.input_root:
        a += ["--input-root", args.input_root]
    return a


def patched(args, D, ps, *, recorder=None):
    """Install the rebindings on run_dann_r5 ``D``; returns the pinned holdout (or None)."""
    fold = int(args.fold)
    pool = C.get_pool(args)
    holdout = None
    if args.target_pool == "dilute10":
        holdout = DP.pinned_holdout(pool, fold)
        ps.set(D, "LensDataset", DP.lens_dataset_factory(D.LensDataset, pool, fold),
               "target role -> diluted pool")
        ps.set(D, "build_target_holdout", lambda tgt_ds, frac=0.1, seed=None: holdout,
               "the original 78 objects, pinned")
    n_l = C.lens_count(pool, args, fold) if args.target_sampler == "uniform" else None
    if args.target_sampler == "uniform" or recorder is not None:
        ps.set(D, "ClassBalancedBatchSampler", C.direct_sampler_ctor(
            D.ClassBalancedBatchSampler, uniform=args.target_sampler == "uniform",
            target_seed=42 + fold + 1000, n_lenses=n_l, csv_path=C.logger_csv(args, ARM),
            recorder=recorder,
            expect_pool=None if holdout is None else holdout["train_pool"]),
            "TARGET call -> uniform sampler" if args.target_sampler == "uniform"
            else "recording only")
    orig_bc = D.build_config

    def build_config(*a, **kw):
        cfg = orig_bc(*a, **kw)
        h = kw.get("holdout")
        counts = None
        if h is not None:
            nl = (len(next(f for f in DP.load_splits()["real_lenses_kfold"]["folds"]
                           if int(f["fold"]) == fold)["target_lens_rows"])
                  if pool is None else int(DP.pool_fold(pool, fold)["n_lenses"]))
            tp = [int(i) for i in (h.get("train_pool") if "train_pool" in h else
                                   list(h["train_pos"]) + list(h["train_neg"]))]
            counts = {"lenses": sum(1 for i in tp if i < nl),
                      "galaxies": sum(1 for i in tp if i >= nl), "holdout": int(h["n_holdout"])}
        cfg["dilute10"] = C.provenance(args, arm="dann_nb3", pool=pool, counts=counts,
                                       abort_mode=cfg.get("abort_mode"), patches=ps,
                                       extra={"budget": BUDGET,
                                              "holdout_identity_sha256":
                                                  None if holdout is None
                                                  else holdout["holdout_identity_sha256"]})
        return cfg
    ps.set(D, "build_config", build_config, "+ dilute10 provenance")
    return holdout


def construct(args, *, recorder=None):
    import copy
    args = copy.copy(args)
    args.dry_run = True                       # never create a logger file when constructing
    """Build the arm (no training); returns (A, D). For --dry-run and the tests."""
    import run_dann_r5 as D
    import live_encode, zoobot_encoder  # noqa: F401  (so their hashes are recorded)
    dargs = D.build_argparser().parse_args(base_argv(args))
    paths = D.resolve_paths(dargs, os.path.dirname(HERE))
    with C.PatchSet() as ps:
        patched(args, D, ps, recorder=recorder)
        A = D.build_arm(dargs.arm, dargs, paths, verbose=False)
    return A, D


def main(argv=None):
    p = C.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]),
                          num_workers_default=2)
    args = p.parse_args(argv)
    C.refuse_zero_workers(args)
    import run_dann_r5 as D
    import live_encode, zoobot_encoder  # noqa: F401

    if args.dry_run:
        rec = C.Recorder()
        A, _ = construct(args, recorder=rec)
        src = A["src_loader"].batch_sampler
        tgt = A["tgt_loader"].batch_sampler
        print(json.dumps({"src_first3": C.first_batches(src, 3)[:1][0][:8],
                          "tgt_first_batch_head": C.first_batches(tgt, 1)[0][:8]}))
        if args.emit_config:
            with open(args.emit_config, "w") as fh:
                json.dump(A["cfg_json"], fh, indent=2, default=str)
        print("[dilute10-dann] dry run: arm constructed, nothing trained.")
        return 0

    dargv = base_argv(args)
    if args.emit_config:
        dargv += ["--emit-config", os.path.abspath(args.emit_config)]
    else:
        out = C.out_dir_for(args, ARM)
        C.refuse_existing(out, args.force)
        os.makedirs(out, exist_ok=True)
        dargv += ["--out-dir", out, "--no-resume"] if not args.force else ["--out-dir", out]
    rec = C.Recorder() if args.target_sampler == "uniform" else None
    out_or_none = None if args.emit_config else C.out_dir_for(args, ARM)
    with C.logger_summary_on_exception(rec, lambda: out_or_none):
        with C.PatchSet() as ps:
            patched(args, D, ps, recorder=rec)
            rc = D.main(dargv)
    if out_or_none is not None:
        completed, reason = C.run_completion(out_or_none, budget=BUDGET, steps_json="run.json")
        C.write_logger_summary(rec, out_or_none, completed=completed, reason=reason)
    if not args.emit_config:
        C.post_run_check(arm="dann_nb3", out_dir=C.out_dir_for(args, ARM), budget=BUDGET,
                         ckpt_every=250, logit_subdir=f"logits/dann_nb3_fold{args.fold}",
                         steps_json="run.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
