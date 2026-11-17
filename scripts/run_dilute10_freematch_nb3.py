"""
run_dilute10_freematch_nb3.py
=============================
FreeMatch x nb3 under the 1:10 diluted target (brief_dilute10_code.md section 4). A WRAPPER
around ``run_unfreeze_freematch.main --run nb3_unfrozen`` -- the confirmatory folds' own
construction, Appendix E continuation (SAT rebuild, mu offset 8000, 200-step Adam warm-up),
loss and loop, 1500 steps -- with these names rebound:

  run_unfreeze_freematch.FOLD_GATE      fold k's gate = the NEW frozen head's gate_values.json
                                        (refused if missing or its head sha256 mismatches)
  run_unfreeze_freematch.build_config   run_dir -> runs/dilute10/freematch_nb3/fold{k}; + provenance
  run_freematch.LensDataset             target role -> diluted pool          (--target-pool dilute10)
  run_freematch.make_balanced_loader    TARGET loader -> uniform sampler     (--target-sampler uniform)
  continuation_state.rebuild_sat_buffers asserted to draw from the uniform loader over the
                                        diluted pool; its batch count is recorded
  measurement lockout (always): curve_hook.append_curve_row scores only, except the step-0
  gate; unfreeze_instrument.memorization_row label-free; evaluate_r1 scores only;
  run_collapse_gate skipped.

``--init-head-from`` defaults to the new frozen stage's ``step_8000.pt``
(``runs/dilute10/freematch_frozen/fold{k}/``) under ``--target-pool dilute10``, and to round
1's head otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402
from uniform_target_sampler import UniformTargetBatchSampler  # noqa: E402

ARM = "freematch_nb3"
FROZEN = "freematch_frozen"


def resolve_head(args):
    if args.init_head_from:
        return os.path.abspath(args.init_head_from)
    if args.target_pool == "dilute10":
        return os.path.join(C.frozen_dir_for(args, FROZEN), "checkpoints", "step_8000.pt")
    return None                                              # round 1's head (R's default)


def base_argv(args, head):
    a = ["--run", "nb3_unfrozen", "--fold", str(args.fold), "--fp16",
         "--grad-checkpointing", "--num-workers", str(args.num_workers)]
    if head:
        a += ["--init-head-from", head]
    if args.device:
        a += ["--device", args.device]
    return a


class _Constructed(Exception):
    """--dry-run: raised at TrainConfig, i.e. right after the loaders exist."""


def patched(args, R, ps, *, sat_record, recorder=None, dry=False):
    import continuation_state as cs
    import curve_hook
    import run_freematch as RF
    import unfreeze_instrument as UI
    fold = int(args.fold)
    pool = C.get_pool(args)
    n_l = C.lens_count(pool, args, fold)
    head = resolve_head(args)
    out = C.out_dir_for(args, ARM)
    gv = None
    if head is not None and (args.target_pool == "dilute10" or args.frozen_dir):
        frozen = os.path.abspath(args.frozen_dir) if args.frozen_dir else \
            C.frozen_dir_for(args, FROZEN)
        gv = C.read_gate_values(frozen, fold, head)          # refuses if missing / mismatch
        fg = {k: dict(v) for k, v in R.FOLD_GATE.items()}
        fg[fold] = {"tpr": float(gv["tpr"]), "auroc": float(gv["auroc"]),
                    "head_l2": float(gv["head_l2"])}
        ps.set(R, "FOLD_GATE", fg, f"fold {fold} gate <- {frozen}/gate_values.json")
    if args.target_pool == "dilute10":
        ps.set(RF, "LensDataset", C.DP.lens_dataset_factory(RF.LensDataset, pool, fold),
               "target role -> diluted pool")
    uniform = args.target_sampler == "uniform"
    if uniform or recorder is not None:
        ps.set(RF, "make_balanced_loader", C.loader_ctor(
            RF.make_balanced_loader, uniform=uniform, n_lenses=n_l if uniform else None,
            csv_path=None if dry else C.logger_csv(args, ARM), recorder=recorder),
            "TARGET loader -> uniform sampler" if uniform else "recording only")
    if dry:
        def stop(*a, **kw):
            raise _Constructed()
        ps.set(RF, "TrainConfig", stop, "dry run: stop after the loaders")
    orig_sat = cs.rebuild_sat_buffers

    def rebuild_sat_buffers(trainer, target_loader, **kw):
        bs = target_loader.batch_sampler
        inner = getattr(bs, "inner", bs)
        n_pool = len(target_loader.dataset)
        if uniform and not isinstance(inner, UniformTargetBatchSampler):
            raise AssertionError("SAT rebuild is not drawing from the uniform loader")
        if args.target_pool == "dilute10":
            want = n_l + int(C.DP.pool_fold(pool, fold)["n_negatives"])
            if n_pool != want or int(kw.get("pool_size", -1)) != want:
                raise AssertionError(f"SAT rebuild pool {n_pool}/{kw.get('pool_size')} "
                                     f"is not the diluted pool {want}")
        rep = orig_sat(trainer, target_loader, **kw)
        sat_record.update({"n_batches": rep["n_batches"], "pool_size": rep["pool_size"],
                           "sampler": type(inner).__name__})
        return rep
    ps.set(cs, "rebuild_sat_buffers", rebuild_sat_buffers,
           "asserts the uniform diluted loader; records the batch count")

    ps.set(curve_hook, "append_curve_row", C.scoring_only_curve_row(
        curve_hook.append_curve_row, gate_steps=(0,), dump_dir=None, fold=fold),
        "measurement lockout: scoring only except the step-0 gate")
    ps.set(UI, "memorization_row", C.label_free_memorization_row,
           "measurement lockout: label-free memorisation row")

    def evaluate_r1(ckpt_path, n_blocks, encoder, instrument, cache_dir, target_fpr, fp16,
                    out_dir, label, fold=0):
        import evaluate as ev
        head_, std, _ = ev.load_checkpoint(ckpt_path, "cpu")
        live = instrument.live()
        pos = live.encode(encoder, f"test_pos_fold{fold}", fp16=fp16)
        neg = live.encode(encoder, "test_negatives_shared", fp16=fp16)
        ps_ = ev.score_features(head_, std, pos["features"], "cpu")
        ns_ = ev.score_features(head_, std, neg["features"], "cpu")
        sdir = os.path.join(out_dir, "scores")
        os.makedirs(sdir, exist_ok=True)
        np.savez(os.path.join(sdir, f"freematch_nb3_fold{fold}.npz"),
                 pos_scores=ps_.astype(np.float64), neg_scores=ns_.astype(np.float64),
                 pos_ids=np.asarray(pos["ids"], dtype="<U80"),
                 neg_ids=np.asarray(neg["ids"], dtype="<U80"),
                 target_fpr=np.asarray(target_fpr))
        return {"scored_not_measured": True, "n_pos": int(ps_.size),
                "n_neg": int(ns_.size), "ckpt": ckpt_path}
    ps.set(R, "evaluate_r1", evaluate_r1, "measurement lockout: R1 scored, not measured")
    ps.set(R, "run_collapse_gate",
           lambda *a, **kw: {"skipped": "measurement lockout (dilute10); post-hoc only, "
                                        "never a stopper"},
           "measurement lockout: collapse gate skipped")

    orig_bc = R.build_config
    rel_run_dir = os.path.relpath(out, os.path.join(REPO, "runs")).replace("\\", "/")

    def build_config(*a, **kw):
        cfg, *rest = orig_bc(*a, **kw)
        cfg["run_dir"] = rel_run_dir
        counts = {"lenses": n_l,
                  "galaxies": (int(C.DP.pool_fold(pool, fold)["n_negatives"]) if pool
                               else len(next(f for f in C.DP.load_splits()
                                             ["real_lenses_kfold"]["folds"]
                                             if int(f["fold"]) == fold)
                                        ["target_neg_object_ids"])),
                  "holdout": 0}
        cfg["dilute10"] = C.provenance(
            args, arm=ARM, pool=pool, counts=counts, abort_mode=None, patches=ps,
            extra={"gate_values": gv, "sat_rebuild": sat_record,
                   "measurement_lockout": ("curve rows scored except step 0; memorisation "
                                           "row label-free; R1 scored; collapse gate "
                                           "skipped")})
        return (cfg, *rest)
    ps.set(R, "build_config", build_config, "+ dilute10 provenance; run_dir")
    return head


def build_parser():
    p = C.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]),
                          num_workers_default=4)
    p.add_argument("--init-head-from", default=None)
    p.add_argument("--frozen-dir", default=None,
                   help="where gate_values.json lives (default: the new frozen stage's dir)")
    return p


def construct(args):
    """Run train_one up to the loaders (pass a SCRATCH --out-root), stop, return them."""
    import run_unfreeze_freematch as R
    import live_encode, zoobot_encoder, run_freematch, continuation_state  # noqa: F401
    rec, sat_record = C.Recorder(), {}
    with C.PatchSet() as ps:
        head = patched(args, R, ps, sat_record=sat_record, recorder=rec, dry=True)
        try:
            R.main(base_argv(args, head))
        except _Constructed:
            pass
    return {k: v for (_kind, k, v) in rec.built}


def main(argv=None):
    args = build_parser().parse_args(argv)
    C.refuse_zero_workers(args)
    import run_unfreeze_freematch as R
    import live_encode, zoobot_encoder, run_freematch, continuation_state  # noqa: F401
    sat_record = {}
    out = C.out_dir_for(args, ARM)
    rec = C.Recorder() if args.target_sampler == "uniform" else None
    live = not (args.emit_config or args.dry_run)
    with C.logger_summary_on_exception(rec, lambda: out if live else None):
        with C.PatchSet() as ps:
            head = patched(args, R, ps, sat_record=sat_record, recorder=rec)
            rargv = base_argv(args, head)
            if args.emit_config:
                rc = R.main(rargv + ["--emit-config", os.path.abspath(args.emit_config)])
            elif args.dry_run:
                rc = R.main(rargv + ["--dry-run"])
            else:
                C.refuse_existing(out, args.force)
                rc = R.main(rargv)
    if live:
        completed, reason = C.run_completion(out, budget=1500, steps_json="results.json",
                                             steps_key="steps")
        C.write_logger_summary(rec, out, completed=completed, reason=reason)
        with open(os.path.join(out, "dilute10_sat_rebuild.json"), "w") as fh:
            json.dump(sat_record, fh, indent=2)
        C.post_run_check(arm="freematch_nb3", out_dir=out, budget=1500, ckpt_every=125,
                         first_step=0, logit_subdir="logits", steps_json="results.json",
                         steps_key="steps", r1_ckpt="checkpoints/step_1500.pt")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
