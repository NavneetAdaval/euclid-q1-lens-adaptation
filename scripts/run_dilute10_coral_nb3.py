"""
run_dilute10_coral_nb3.py
=========================
CORAL x nb3 under the 1:10 diluted target (brief_dilute10_code.md section 4). A WRAPPER
around ``run_coral_r4_confirm.main`` -- which builds the arm with
``run_coral_round4.build_arm`` and trains with ``run_coral_r4_dev8000.train_loop``, the
confirmatory folds' own code -- gamma 16, target EMA off, 2000 steps (the hashed
``runs/coral_r4_budget.json``). Rebound:

  run_coral_r4_confirm.expected_diffs_for  accepts the extra keyword arguments the CURRENT
        file's main passes but its 3-argument definition does not take (a half-applied edit
        made after the confirmatory folds ran -- see PROJECTv7.md 7.37). The keywords are
        dropped, which is exactly the call the confirmatory folds made. Under dilute10 the
        allowlisted head tail is the new frozen head's.
  run_coral_r4_confirm.step0_gate  the original gate, then the declared tolerance enforced:
        one positive (live-encode path) and |AUROC diff| <= 1e-3, else abort.
  run_coral_r4_confirm.build_record        + the ``dilute10`` provenance block (always)
  run_coral_round4.FOLD_GATE      fold k <- the new frozen head's gate_values.json (dilute10)
  run_coral_round4.LensDataset    target role -> diluted pool               (dilute10)
  run_coral_round4.make_balanced_loader TARGET loader -> uniform sampler    (uniform)

Output: ``runs/dilute10/coral_nb3/fold{k}/``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402

ARM = "coral_nb3"
FROZEN = "coral_frozen"
STEP0_AUROC_TOL = 1e-3
BUDGET = 2000
FEATURE_NORM_CSV = "dilute10_feature_norm_advisory.csv"
FEATURE_NORM_COLUMNS = ("step", "kind", "src_feat_l2", "tgt_feat_l2", "ceiling", "floor",
                        "coral_raw")


def advisory_feature_norm_guard(orig, csv_path, firings):
    """Log-only stand-in for ``coral_round4.check_feature_norm_ceiling``.

    Part 2 (stop_rules_nb3_coral_freematch) declares that CORAL nb3's ONLY stop conditions
    are the step-0 gate, a missed budget and missing logits, and that everything else it
    logs is advisory. The inherited round-4 guard raised ``FeatureNormCeiling`` (PROJECTv7
    7.46: Kaggle fold 1 aborted at step 1). This calls the original UNCHANGED -- same
    ceiling, same floor, same message -- catches its exception, records the firing (step,
    src/tgt ||f||, the band, raw CORAL) in ``firings`` and in ``csv_path``, and returns.
    No threshold value is changed and the frozen module is not edited."""
    from coral_round4 import FeatureNormCeiling

    def guard(step, scale, ceiling, raw_coral, *a, **kw):
        try:
            return orig(step, scale, ceiling, raw_coral, *a, **kw)
        except FeatureNormCeiling as e:
            floor = kw.get("floor", a[0] if a else None)
            row = {"step": int(step),
                   "kind": "ceiling" if "CEILING" in str(e) else "floor",
                   "src_feat_l2": float(scale.get("src_feat_l2", float("nan"))),
                   "tgt_feat_l2": float(scale.get("tgt_feat_l2", float("nan"))),
                   "ceiling": float(ceiling) if ceiling is not None else None,
                   "floor": float(floor) if floor is not None else None,
                   "coral_raw": float(raw_coral)}
            firings.append(row)
            if csv_path:
                import csv as _csv
                new = not os.path.isfile(csv_path)
                os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
                with open(csv_path, "a", newline="") as fh:
                    w = _csv.DictWriter(fh, fieldnames=FEATURE_NORM_COLUMNS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
            return None
    guard.__wrapped__ = orig
    return guard


def feature_norm_summary(firings):
    return {"status": "advisory (log-only), runs/dilute10_prereg_part2.json "
                      "stop_rules_nb3_coral_freematch",
            "n_firings": len(firings),
            "first_step": firings[0]["step"] if firings else None,
            "last_step": firings[-1]["step"] if firings else None,
            "kinds": sorted({f["kind"] for f in firings}),
            "csv": FEATURE_NORM_CSV if firings else None}


def resolve_head(args):
    if args.init_head_from:
        return os.path.abspath(args.init_head_from)
    if args.target_pool == "dilute10":
        return os.path.join(C.frozen_dir_for(args, FROZEN), "checkpoints", "step_8000.pt")
    return None


def base_argv(args, head, parent):
    a = ["--fold", str(args.fold), "--out-dir", parent, "--num-workers", str(args.num_workers)]
    if head:
        a += ["--init-head-from", head]
    if args.device:
        a += ["--device", args.device]
    if args.input_root:
        a += ["--input-root", args.input_root]
    return a


def patched(args, CF, R4, ps, *, recorder=None, capture=None, guard_csv=None,
            firings=None):
    fold = int(args.fold)
    import run_coral_r4_dev8000 as DEV
    ps.set(DEV, "check_feature_norm_ceiling", advisory_feature_norm_guard(
        DEV.check_feature_norm_ceiling, guard_csv, firings if firings is not None else []),
        "feature-norm guard LOG-ONLY: advisory under part 2 stop_rules_nb3_coral_freematch "
        "(PROJECTv7 7.47)")
    pool = C.get_pool(args)
    n_l = C.lens_count(pool, args, fold)
    head = resolve_head(args)
    gv = None
    if head is not None and (args.target_pool == "dilute10" or args.frozen_dir):
        frozen = os.path.abspath(args.frozen_dir) if args.frozen_dir else \
            C.frozen_dir_for(args, FROZEN)
        gv = C.read_gate_values(frozen, fold, head)
        fg = {k: dict(v) for k, v in R4.FOLD_GATE.items()}
        fg[fold] = {"tpr": float(gv["tpr"]), "auroc": float(gv["auroc"]),
                    "head_l2": float(gv["head_l2"])}
        ps.set(R4, "FOLD_GATE", fg, f"fold {fold} gate <- {frozen}/gate_values.json")
    if args.target_pool == "dilute10":
        ps.set(R4, "LensDataset", C.DP.lens_dataset_factory(R4.LensDataset, pool, fold),
               "target role -> diluted pool")
    uniform = args.target_sampler == "uniform"
    if uniform or recorder is not None:
        ps.set(R4, "make_balanced_loader", C.loader_ctor(
            R4.make_balanced_loader, uniform=uniform, n_lenses=n_l if uniform else None,
            csv_path=C.logger_csv(args, ARM) if capture is None else None,
            recorder=recorder),
            "TARGET loader -> uniform sampler" if uniform else "recording only")

    orig_ed = CF.expected_diffs_for

    def expected_diffs_for(f, budget, ref, **_ignored):
        exp = orig_ed(f, budget, ref)
        if head is not None and gv is not None:
            from run_coral_r4_dev8000 import _path_tail
            exp["init_head_from_tail"] = (ref.get("init_head_from_tail"),
                                          _path_tail(head, 5),
                                          "dilute10: the NEW frozen-stage head (part 1 4.2)")
        return exp
    ps.set(CF, "expected_diffs_for", expected_diffs_for,
           "3-argument call, as the confirmatory folds ran it")

    orig_gate = CF.step0_gate

    def step0_gate(scorer, record, f, n_pos, *a, **kw):
        g = orig_gate(scorer, record, f, n_pos, *a, **kw)
        one = 1.0 / int(n_pos)
        if g["tpr_abs_diff"] > one + 1e-12 or g["auroc_abs_diff"] > STEP0_AUROC_TOL:
            raise SystemExit(f"[dilute10-coral-nb3] ABORT at the step-0 gate: |TPR diff| "
                             f"{g['tpr_abs_diff']:.6f} (one positive {one:.6f}), |AUROC "
                             f"diff| {g['auroc_abs_diff']:.2e} (tol {STEP0_AUROC_TOL})")
        g["dilute10_tolerance"] = {"tpr": one, "auroc": STEP0_AUROC_TOL, "verdict": "PASS"}
        return g
    ps.set(CF, "step0_gate", step0_gate, "declared tolerance enforced")

    orig_br = CF.build_record

    def build_record(A, a2, paths):
        rec = orig_br(A, a2, paths)
        n_t = len(A["tgt_loader"].dataset)
        counts = {"lenses": n_l, "galaxies": n_t - n_l, "holdout": 0}
        rec["dilute10"] = C.provenance(
            args, arm=ARM, pool=pool, counts=counts, abort_mode=None, patches=ps,
            extra={"gate_values": gv, "step0_tolerance": {"tpr": "one positive",
                                                          "auroc": STEP0_AUROC_TOL}})
        return rec
    ps.set(CF, "build_record", build_record, "+ dilute10 provenance")

    if capture is not None:
        orig_ba = R4.build_arm

        def build_arm(*a, **kw):
            A = orig_ba(*a, **kw)
            capture["A"] = A
            return A
        ps.set(R4, "build_arm", build_arm, "capture the arm (construct / emit only)")
    return head


def build_parser():
    p = C.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]),
                          num_workers_default=2)
    p.add_argument("--init-head-from", default=None)
    p.add_argument("--frozen-dir", default=None)
    return p


def construct(args):
    """Run the confirm main with --config-diff-only into a scratch parent: the arm is built,
    the record and config diff are written there, nothing is trained."""
    import run_coral_r4_confirm as CF
    import run_coral_round4 as R4
    import live_encode, zoobot_encoder  # noqa: F401
    rec, cap = C.Recorder(), {}
    tmp = tempfile.mkdtemp(prefix="dilute10_coral_nb3_")
    try:
        with C.PatchSet() as ps:
            head = patched(args, CF, R4, ps, recorder=rec, capture=cap)
            CF.main(base_argv(args, head, tmp) + ["--config-diff-only"])
        with open(os.path.join(tmp, f"fold{int(args.fold)}", "run.json")) as fh:
            record = json.load(fh)
        with open(os.path.join(tmp, f"fold{int(args.fold)}", "config_diff.json")) as fh:
            diff = json.load(fh)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"record": record, "config_diff": diff, "A": cap.get("A"),
            "loaders": {k: v for (_kind, k, v) in rec.built}}


def main(argv=None):
    args = build_parser().parse_args(argv)
    C.refuse_zero_workers(args)
    if args.emit_config or args.dry_run:
        r = construct(args)
        if args.emit_config:
            with open(args.emit_config, "w") as fh:
                json.dump(r["record"], fh, indent=1, default=str)
        print(f"[dilute10-coral-nb3] constructed; config-diff mismatches vs the fold-0 "
              f"development arm: {r['config_diff']['mismatches']}. Nothing trained.")
        return 0
    import run_coral_r4_confirm as CF
    import run_coral_round4 as R4
    import live_encode, zoobot_encoder  # noqa: F401
    parent = os.path.join(os.path.abspath(args.out_root), ARM)
    out = os.path.join(parent, f"fold{int(args.fold)}")
    C.refuse_existing(out, args.force)
    rec = C.Recorder() if args.target_sampler == "uniform" else None
    firings = []
    with C.logger_summary_on_exception(rec, lambda: out):
        with C.PatchSet() as ps:
            head = patched(args, CF, R4, ps, recorder=rec,
                           guard_csv=os.path.join(out, FEATURE_NORM_CSV), firings=firings)
            # a fresh run refuses to touch an existing directory; with --force (a logged
            # resume, KAGGLE_DILUTE10.md Cell 5) the confirm runner resumes from
            # resume/latest.pt
            rc = CF.main(base_argv(args, head, parent)
                         + ([] if args.force else ["--no-resume"]))
    if os.path.isdir(out):
        with open(os.path.join(out, "dilute10_feature_norm_advisory.json"), "w") as fh:
            json.dump(feature_norm_summary(firings), fh, indent=2)
    completed, reason = C.run_completion(out, budget=BUDGET, steps_json="run.json")
    C.write_logger_summary(rec, out, completed=completed, reason=reason)
    C.post_run_check(arm="coral_nb3", out_dir=out, budget=BUDGET, ckpt_every=250,
                     logit_subdir=f"logits/{CF.logit_arm(int(args.fold), 2000)}",
                     steps_json="run.json", r1_ckpt="checkpoints/step_2000.pt")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
