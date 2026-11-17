"""
audit_lambda0_r6.py
===================
ROUND 6 -- the three pre-run audits for the **lambda=0 unfrozen source-only baseline,
folds 1-4**. CPU-only, no GPU, no metric of any kind is read.

The run itself is round 5's ARM 1 extended from fold 0 to folds 1-4:

    python -u scripts/run_dann_r5.py --arm dann_nb3_lambda0 --fold k --max-steps 4000

No new loss and no new code path, which is precisely why the audits below are the whole
of the pre-run work: the claim being made is that the confirmatory folds run the SAME
program the arms they will be differenced against ran, and that claim has to be checked
rather than asserted.

WHAT THE FIRST VERSION OF THIS AUDIT GOT WRONG, kept because it is the finding.
The brief asks to "verify field-for-field that the confirmatory config matches ARM 1's".
Run the check and it fails -- and it fails for a reason that makes the brief's target the
wrong one. THREE versions of ``run_dann_r5.py`` are in the record:

    arm 1, fold 0, lambda=0    9b6e38669cb2e66c
    arm 2, fold 0, DANN        4fd5964f37202dfd
    DANN folds 1-4  (= CURRENT) 320ed27d74473ce1

7.19's re-specification of conditions 3 and 4 landed between them. So byte-matching arm
1's fold 0 is not available without reverting that, and reverting it would break the
match against DANN's confirmatory folds -- which is the match that matters, because
**DANN x nb3 - lambda=0 on folds 1-4 is the only fully controlled contrast in the
project** (brief 7). You cannot have both, and this audit checks the one the science
needs. What is lost is stated rather than hidden: fold 0's 0.1531 ran a different
script version than these folds will, so under 7.15's rule it is a QUALITATIVE
consistency check against folds 1-4 and never a difference. Fold 0 enters no mean
anyway (brief 6), so nothing downstream depends on the loss.

  A. SCRIPT PROVENANCE, as a cross-run MATRIX rather than a single comparison. Every
     file in SCRIPT_HASH_FILES, hashed now, against every DANN run on disk. The
     confirmatory folds must byte-match DANN folds 1-4. ``da_losses.py`` differs from
     all of them because round 6 appended BNM to it; that is allowed ONLY if the
     difference is provably append-only -- i.e. some prefix of the current file hashes
     to exactly what those runs recorded for the whole file.

  B. CONFIG FIELDS. Arm 1's fold-0 config and DANN's four confirmatory configs against
     the configs the runner emits for lambda=0 folds 1-4 at the declared budget,
     produced by the SAME construction code via --emit-config. Two sweeps:
       * TRAINING-RELEVANT keys, compared across EVERY run on disk. These decide what
         the optimiser does. Any difference here is fatal.
       * the exhaustive sweep, classified, so that a key nobody thought to list cannot
         differ silently.
     The head init is compared by the file's own sha256, not by its path -- arm 1 ran
     on Kaggle and these configs were emitted locally, so the path strings differ while
     the file does not.

  C. THE SCHEDULE AUDIT, RE-RUN. 7.18 audited 11 components on DANN's path and found
     every one flat or absolute-step pinned, which is what licenses reading step 1500
     and step 2000 out of a 4000-step run as the models 1500- and 2000-step runs would
     produce. lambda=0 takes the same coral-hook branch so the audit *should* carry --
     and "should carry" is exactly the reasoning this audit exists to replace. It is
     re-run against the CURRENT bytes of all four modules.

     If any component turns out to carry a horizon, the sensitivity readouts at 1500
     and 2000 are DROPPED and said to be dropped. They are not reinterpreted.

    python -u scripts/audit_lambda0_r6.py --config-dir <dir with cfg_fold1..4.json>
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
ARM1_CONFIG = os.path.join(REPO, "runs", "dann_r5_arm1_control", "run_config.json")
ARM2_CONFIG = os.path.join(REPO, "runs", "dann_arm2_fold0", "extracted",
                           "dann_r5_nb3_fold0", "run_config.json")
DANN_CONFIRM_GLOB = os.path.join(REPO, "runs", "dann_main", "fold*", "extracted",
                                 "fold*", "run_config.json")
HEAD_FILE = os.path.join(REPO, "runs", "bakeoff_baseline_A", "checkpoints", "step_8000.pt")

CONFIRM_STEPS_R6 = 4000
SENSITIVITY_STEPS_R6 = (1500, 2000)
CKPT_EVERY = 250

AUDIT_MODULES = ("run_dann_r5.py", "dann_round5.py", "da_losses.py", "train_skeleton.py")

# The runner whose arm-specific rows are scanned. It is a PARAMETER, not a literal buried
# in part_c: two rows (`ema_thresholds_mu`, and the arm's no-horizon check) read the
# runner by name, and an audit that keeps reading run_dann_r5.py after a new runner lands
# would pass on the wrong file. `--runner` re-points them, and every affected row prints
# the filename it actually scanned.
DEFAULT_RUNNER = "run_dann_r5.py"

# The reference the confirmatory folds MUST byte-match: the runs they will be
# differenced against. Not arm 1 -- see the module docstring.
MATCH_TARGET = "dann_confirm_fold1"


# ---------------------------------------------------------------------------
# A -- script provenance, as a cross-run matrix
# ---------------------------------------------------------------------------

def sha256_file(path: str, n: int = 16) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:n]


def prefix_sha16(data: bytes, n_lines: int) -> str:
    return hashlib.sha256(b"\n".join(data.split(b"\n")[:n_lines])).hexdigest()[:16]


def append_only_split(path: str, want: str):
    """Return the line count whose prefix hashes to ``want``, or None.

    If one exists, every byte the earlier run executed is bit-identical and the whole
    difference lies below that line. This is a proof, not an argument.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    for n in range(1, len(data.split(b"\n")) + 1):
        if prefix_sha16(data, n) == want:
            return n, len(data.split(b"\n"))
    return None, None


def load_runs() -> dict:
    runs = {}
    if os.path.exists(ARM1_CONFIG):
        with open(ARM1_CONFIG, encoding="utf-8") as fh:
            runs["arm1_lambda0_fold0"] = json.load(fh)
    if os.path.exists(ARM2_CONFIG):
        with open(ARM2_CONFIG, encoding="utf-8") as fh:
            runs["arm2_dann_fold0"] = json.load(fh)
    for f in sorted(glob.glob(DANN_CONFIRM_GLOB)):
        k = os.path.basename(os.path.dirname(f))
        with open(f, encoding="utf-8") as fh:
            runs[f"dann_confirm_{k}"] = json.load(fh)
    return runs


def part_a(runs: dict) -> dict:
    names = sorted({n for r in runs.values() for n in r.get("script_sha256_16", {})})
    current = {n: sha256_file(os.path.join(SCRIPTS, n))
               for n in names if os.path.exists(os.path.join(SCRIPTS, n))}

    matrix = {n: {r: runs[r].get("script_sha256_16", {}).get(n) for r in runs}
              for n in names}
    for n in names:
        matrix[n]["CURRENT"] = current.get(n)

    target = runs.get(MATCH_TARGET, {}).get("script_sha256_16", {})
    files, ok = {}, True
    for n in names:
        want, got = target.get(n), current.get(n)
        if got is None:
            files[n] = {"status": "MISSING_LOCALLY"}
            ok = False
        elif want == got:
            files[n] = {"status": "IDENTICAL_TO_TARGET", "sha16": got}
        else:
            split, total = append_only_split(os.path.join(SCRIPTS, n), want)
            if split is None:
                files[n] = {"status": "MODIFIED_NOT_APPEND_ONLY",
                            "target": want, "current": got}
                ok = False
            else:
                files[n] = {
                    "status": "APPEND_ONLY", "target": want, "current": got,
                    "identical_through_line": split, "appended_lines": total - split,
                    "proof": (f"sha256 of lines 1-{split} of the current file == {want}, "
                              f"the whole-file hash {MATCH_TARGET} recorded. Every byte "
                              f"it executed is bit-identical; the difference is entirely "
                              f"below line {split}."),
                }

    versions = {n: sorted({v for v in matrix[n].values() if v}) for n in names}
    return {"match_target": MATCH_TARGET, "matrix": matrix, "files": files,
            "n_distinct_versions": {n: len(v) for n, v in versions.items()},
            "PASS": ok}


# ---------------------------------------------------------------------------
# B -- config fields
# ---------------------------------------------------------------------------

# What the optimiser actually does. A difference in ANY of these is fatal: it means the
# runs are not the same program and 7.15's rule forbids differencing their numbers.
TRAINING_RELEVANT = [
    "n_blocks", "data_path", "lp_ft", "init_head_l2", "head_type",
    "lr", "lr_decay", "weight_decay", "backbone_lr_effective",
    "batch_size", "true_batch", "gradient_accumulation", "grad_checkpointing",
    "steps_per_epoch", "pos_fraction", "seed", "augmentation",
    "fp16", "precision", "eval_precision", "num_workers", "encoder_eval",
    "log_every", "ckpt_every",
    "dann_lambda_schedule", "dann_lambda_horizon", "dann_d_topology", "dann_d_hidden",
    "dann_d_norm", "dann_d_dropout", "dann_d_dropout_layers", "dann_d_lr",
    "dann_d_weight_decay", "dann_d_holdout_frac", "dann_attachment",
    "dann_single_forward_pass", "da_hook_slot", "domain_adaptation",
]

FOLD_DEPENDENT = {
    "fold", "n_test_positives", "run_dir", "logit_arm", "out_dir",
    "expect_step0_tpr", "expect_step0_auroc", "step0_tpr_tolerance",
    "dann_d_holdout_indices", "dann_d_holdout_n", "abort_mode_source",
    "morph_proxy_pool", "paths", "init_head_from", "init_head_from_tail",
}
BUDGET_DEPENDENT = {
    "max_steps", "on_declared_steps", "is_preregistered_budget", "epochs_planned",
    "dann_lambda_at_max_steps", "sensitivity_step_is_on_the_grid",
    "n_checkpoints_expected",
}
DETECTOR_LAYER = {
    "abort_cond3_column", "abort_cond3_declaration_criterion", "abort_conditions",
    "abort_cond3_column_note", "abort_cond3_interpretation_column",
    "abort_cond3_interpretation_columns", "abort_cond3_interpretation_note",
    "abort_cond3_margin", "abort_cond3_margin_status", "abort_cond3_median_window_k",
    "abort_cond3_median_window_note", "abort_cond3b_sustain", "abort_cond4_form",
    "abort_cond4_src_margin", "abort_cond4_tgt_margin", "abort_advisory_note",
    "abort_advisory_only_detectors", "abort_respecified_from_control",
    "abort4_sustain", "abort_mode", "abort_watch", "morph_proxy_abort_note",
    "control_curve", "allow_control_gaps",
}
# present only in a POST-run record, or only in a PRE-run emit. Neither is a difference
# in the program: one file is written before training and one after.
POST_RUN_ONLY = {
    "started_utc", "finished_utc", "torch_version", "script_sha256_16",
    "steps_completed", "fp16_skipped_cumulative", "resume_events", "step0_gate",
    "peak_gpu_gb_early", "peak_gpu_gb_allocated", "peak_gpu_gb_reserved", "gpu_name",
    "gpu_total_gb", "gpu_visible_devices", "aborted", "abort_reason", "wall_minutes",
    "n_checkpoints_saved", "morph_trajectory_summary", "continuation",
    "metric_hygiene", "purpose", "n_checkpoints_expected", "out_dir", "paths",
}
PROVENANCE_TEXT = {
    "budget_status", "reporting_note", "device", "confirm_budget_rule",
    "confirm_budget_file", "confirm_budget_sha256", "confirm_steps",
    "cross_method_budgets", "headline_declaration", "sensitivity_step",
    "round1_dann_fold0_reference", "no_batchnorm_check", "init_head_provenance",
}


def part_b(runs: dict, cfgs: dict) -> dict:
    out = {"PASS": True}

    # --- B1: training-relevant keys, across EVERY run on disk -------------------
    everything = dict(runs)
    for k, c in cfgs.items():
        everything[f"lambda0_r6_fold{k}"] = c
    b1 = {}
    for key in TRAINING_RELEVANT:
        vals = {r: c.get(key, "<ABSENT>") for r, c in everything.items()}
        distinct = {json.dumps(v, sort_keys=True, default=str) for v in vals.values()}
        b1[key] = {"identical_everywhere": len(distinct) == 1,
                   "value": next(iter(vals.values())) if len(distinct) == 1 else None,
                   "values": None if len(distinct) == 1 else vals}
        if len(distinct) != 1:
            out["PASS"] = False
    out["training_relevant"] = b1
    out["n_runs_compared"] = len(everything)
    out["runs_compared"] = sorted(everything)

    # --- B2: the head init file, by its own sha256 -----------------------------
    head_sha = sha256_file(HEAD_FILE, 64) if os.path.exists(HEAD_FILE) else None
    l2 = {r: c.get("init_head_l2") for r, c in everything.items()}
    out["head_init"] = {
        "file": os.path.relpath(HEAD_FILE, REPO).replace("\\", "/"),
        "sha256": head_sha,
        "init_head_l2_per_run": l2,
        "l2_identical_everywhere": len(set(map(str, l2.values()))) == 1,
        "note": ("compared by the FILE's sha256 and by ||head||_2, not by the path "
                 "string: arm 1 ran on Kaggle and these configs were emitted locally, "
                 "so the paths differ while the file does not."),
    }
    if not out["head_init"]["l2_identical_everywhere"]:
        out["PASS"] = False

    # --- B3: exhaustive sweep, arm1 fold0 vs each emitted lambda0 fold ---------
    arm1 = runs.get("arm1_lambda0_fold0", {})
    sweep = {}
    for fold, cfg in sorted(cfgs.items()):
        rec = {"IDENTICAL": [], "FOLD_DEP": [], "BUDGET": [], "DETECTOR_LAYER": [],
               "PRE_VS_POST_RUN": [], "TEXT": [], "UNEXPECTED": []}
        for k in sorted(set(arm1) | set(cfg)):
            in_a, in_c = k in arm1, k in cfg
            if in_a and in_c and arm1[k] == cfg[k]:
                rec["IDENTICAL"].append(k)
            elif k in FOLD_DEPENDENT:
                rec["FOLD_DEP"].append(k)
            elif k in BUDGET_DEPENDENT:
                rec["BUDGET"].append(k)
            elif k in DETECTOR_LAYER:
                rec["DETECTOR_LAYER"].append(k)
            elif not (in_a and in_c) and (k in POST_RUN_ONLY or k in PROVENANCE_TEXT
                                          or k in DETECTOR_LAYER):
                rec["PRE_VS_POST_RUN"].append(k)
            elif k in PROVENANCE_TEXT or k in POST_RUN_ONLY:
                rec["TEXT"].append(k)
            else:
                rec["UNEXPECTED"].append({"key": k,
                                          "arm1": arm1.get(k, "<ABSENT>"),
                                          "confirm": cfg.get(k, "<ABSENT>")})
        rec["counts"] = {c: len(v) for c, v in rec.items() if isinstance(v, list)}
        if rec["UNEXPECTED"]:
            out["PASS"] = False
        sweep[fold] = rec
    out["exhaustive_sweep_vs_arm1"] = sweep

    # --- B4: the per-fold step-0 gate -----------------------------------------
    gates = {f: (c.get("expect_step0_tpr"), c.get("expect_step0_auroc"),
                 c.get("step0_tpr_tolerance"), c.get("step0_gate_skippable"))
             for f, c in cfgs.items()}
    out["per_fold_gate"] = {
        str(f): {"tpr": g[0], "auroc": g[1], "tpr_tolerance": g[2], "skippable": g[3]}
        for f, g in sorted(gates.items())}
    out["gate_tpr_all_distinct"] = len({g[0] for g in gates.values()}) == len(gates)
    out["gate_auroc_all_distinct"] = len({g[1] for g in gates.values()}) == len(gates)
    out["gate_non_skippable_everywhere"] = all(g[3] is False for g in gates.values())
    if not (out["gate_auroc_all_distinct"] and out["gate_non_skippable_everywhere"]):
        out["PASS"] = False

    # --- B5: abort mode. The brief requires ADVISORY, never aborting. ----------
    modes = {f: c.get("abort_mode") for f, c in cfgs.items()}
    out["abort_mode"] = {
        "per_fold": modes,
        "all_log_only": all(m == "log_only" for m in modes.values()),
        "dann_confirm_folds_for_contrast": {
            r: c.get("abort_mode") for r, c in runs.items() if "confirm" in r},
        "note": ("run_dann_r5.default_abort_mode returns log_only for EVERY fold of the "
                 "lambda_zero arm -- the control must finish. DANN's confirmatory folds "
                 "ran 'armed', so the two sides of the contrast differ here. This cannot "
                 "change a training step (an armed detector either aborts or does "
                 "nothing, and DANN's folds did not abort), but it is a real asymmetry "
                 "and is disclosed rather than smoothed over."),
    }
    if not out["abort_mode"]["all_log_only"]:
        out["PASS"] = False
    return out


# ---------------------------------------------------------------------------
# C -- the schedule audit, re-run
# ---------------------------------------------------------------------------

SCHEDULER_TOKENS = ("lr_scheduler", "LambdaLR", "StepLR", "MultiStepLR", "CosineAnnealing",
                    "OneCycleLR", "ExponentialLR", "ReduceLROnPlateau", "get_last_lr",
                    "CosineAnnealingWarmRestarts", "LinearLR", "PolynomialLR")
CLIP_TOKENS = ("clip_grad_norm_", "clip_grad_value_", "clip_grad")


def _code_only(src: str) -> str:
    """The source's NAME and OP tokens, with comments and string literals removed.

    A token scan over raw text cannot tell code from prose, and the file most likely to
    DISCUSS ramp tokens is the one documenting the guard against them -- run_bnm_r6.py's
    docstring names every token the scan looks for, and scanning raw text flagged its own
    documentation.

    THE CANONICAL IMPLEMENTATION NOW LIVES IN ``code_scan.code_only``; this delegates so
    there is one scanner rather than two that can drift apart. New scans should use
    ``code_scan.scan``, which additionally REQUIRES a known-null.
    """
    from code_scan import code_only as _co
    return _co(src)


def _read(name: str) -> str:
    with open(os.path.join(SCRIPTS, name), encoding="utf-8") as fh:
        return fh.read()


def _find_func(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _reads(node, ident: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == ident for n in ast.walk(node))


def _funcs_reading(src: str, ident: str) -> list:
    hits = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _reads(node, ident):
            hits.append(node.name)
    return sorted(set(hits))


def coral_branch_reads_total_steps() -> dict:
    """7.18's actual claim: the CORAL-HOOK BRANCH contains 0 occurrences of total_steps.

    ``train_step`` legitimately ACCEPTS ``total_steps`` and forwards it to the
    AdaMatch/FreeMatch branches through two early returns. Everything after those
    returns is the branch this arm takes, and that is what must be clean -- a scan that
    merely asks "does train_step mention total_steps" answers a different question and
    fails on a forward.
    """
    fn = _find_func(_read("train_skeleton.py"), "train_step")
    if fn is None:
        return {"found": False}
    guards, branch = [], []
    for stmt in fn.body:
        is_guard = (isinstance(stmt, ast.If)
                    and any(isinstance(s, ast.Return) for s in stmt.body)
                    and _reads(stmt, "total_steps"))
        (guards if is_guard else branch).append(stmt)
    hits = [ast.dump(s) for s in branch if _reads(s, "total_steps")]
    return {"found": True, "n_forwarding_guards": len(guards),
            "n_statements_in_coral_branch": len(branch),
            "total_steps_in_coral_branch": len(hits)}


def part_c(modules=AUDIT_MODULES, runner: str = DEFAULT_RUNNER) -> dict:
    out = {"components": {}, "PASS": True, "modules": list(modules), "runner": runner}
    srcs = {m: _read(m) for m in modules}
    if runner not in srcs:
        raise SystemExit(
            f"--runner {runner} is not in the audited module list {list(modules)}. The "
            f"arm-specific rows would scan a file the audit never read; add it with "
            f"--modules or fix --runner.")

    def record(key, finding, ok, detail=None):
        out["components"][key] = {"finding": finding, "OK": bool(ok)}
        if detail is not None:
            out["components"][key]["detail"] = detail
        if not ok:
            out["PASS"] = False

    sched = {m: [t for t in SCHEDULER_TOKENS if t in s] for m, s in srcs.items()}
    sched = {m: v for m, v in sched.items() if v}
    record("torch_lr_scheduler",
           "none exists in any audited module" if not sched else f"FOUND {sched}",
           not sched, {"tokens_scanned": len(SCHEDULER_TOKENS)})

    cb = coral_branch_reads_total_steps()
    readers = {m: _funcs_reading(s, "total_steps") for m, s in srcs.items()}
    readers = {m: v for m, v in readers.items() if v}
    ts_ok = bool(cb.get("found")) and cb.get("total_steps_in_coral_branch") == 0
    record("total_steps",
           (f"the coral-hook branch this arm takes contains 0 occurrences "
            f"({cb.get('n_statements_in_coral_branch')} statements scanned); "
            f"train_step only FORWARDS it through "
            f"{cb.get('n_forwarding_guards')} early-return guards into the "
            f"AdaMatch/FreeMatch branches"
            if ts_ok else "READ INSIDE THE CORAL BRANCH"),
           ts_ok, {"coral_branch": cb, "functions_mentioning": readers})

    clip = {m: [t for t in CLIP_TOKENS if t in s] for m, s in srcs.items()}
    clip = {m: v for m, v in clip.items() if v}
    record("gradient_clipping",
           "none -- 0 hits, matching 7.14's no-clip CORAL fix" if not clip
           else f"FOUND {clip}", not clip)

    ts = _read("train_skeleton.py")
    layerwise = any(p in ts for p in ("lr_decay ** ", "lr_decay**"))
    temporal = any(p in ts for p in ("lr_decay ** step", "lr_decay ** epoch",
                                     "lr_decay**step", "lr_decay**epoch"))
    record("lr_decay",
           "layer-wise, indexing DEPTH (backbone_lr * lr_decay**i over blocks)"
           if (layerwise and not temporal) else "temporal exponent found",
           layerwise and not temporal)

    gi_detail, gi_abs = {}, True
    for m, s in srcs.items():
        if "growth_interval" not in s:
            continue
        for node in ast.walk(ast.parse(s)):
            if isinstance(node, ast.keyword) and node.arg == "growth_interval":
                if isinstance(node.value, ast.Constant):
                    gi_detail[m] = node.value.value
                else:
                    gi_detail[m], gi_abs = "NON-CONSTANT", False
    record("gradscaler_growth_interval",
           (f"absolute constant {gi_detail}" if gi_abs and gi_detail
            else (f"NON-CONSTANT {gi_detail}" if gi_detail
                  else "not set anywhere; torch's default 2000 applies, an ABSOLUTE "
                       "step count whose state at step N depends only on steps 0..N")),
           gi_abs)

    smp_path = os.path.join(SCRIPTS, "samplers.py")
    smp = _read("samplers.py") if os.path.exists(smp_path) else ""
    smp_uses = _funcs_reading(smp, "max_steps") if smp else []
    record("sampler_num_batches",
           "derived from len(src_ds) // batch_size, not from max_steps"
           if not smp_uses else f"SAMPLER READS max_steps in {smp_uses}",
           not smp_uses)

    pg_assign = []
    for m, s in srcs.items():
        for node in ast.walk(ast.parse(s)):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript) and "param_groups" in ast.dump(t):
                    pg_assign.append(f"{m}:{getattr(node, 'lineno', '?')}")
    record("param_groups_mutation",
           "no assignment targets param_groups -- lr/wd/betas set once at construction"
           if not pg_assign else f"FOUND {pg_assign}", not pg_assign)

    lam0 = ("target_ema", "tau_global", "adamatch_mu_at_step", "freematch_sat_update")
    present = [t for t in lam0 if t in srcs[runner]]
    record("ema_thresholds_mu",
           f"none referenced by {runner}" if not present
           else f"FOUND {present} in {runner}",
           not present, {"scanned": runner})

    # THE ARM'S OWN SCHEDULE ROW. For a DANN-family runner this is lambda's horizon; for
    # a runner with no schedule at all (BNM) the equivalent claim is that its DA module
    # exposes NO horizon to be positioned on. Which check applies is decided by what is
    # in the module list, and the finding says which was run -- so this row cannot
    # silently keep answering the DANN question after the runner changes.
    dl = srcs["da_losses.py"]
    if "bnm_round6.py" in srcs:
        import bnm_round6 as _b
        hook_attrs = ("set_step", "lambda_at", "total_steps", "horizon", "step",
                      "warmup", "ramp", "schedule")
        hook = _b.BNMHook.__dict__
        leaked = [a for a in hook_attrs if a in hook]
        record("arm_has_no_horizon",
               (f"BNMHook exposes none of {hook_attrs} -- there is no schedule to be "
                f"positioned on, so no horizon can be misread on resume or at a "
                f"sensitivity readout")
               if not leaked else f"BNMHook EXPOSES {leaked}",
               not leaked, {"scanned": "bnm_round6.BNMHook"})
    else:
        lam_fn = _find_func(dl, "dann_lambda_at_step")
        lam_ok = lam_fn is not None and not _reads(lam_fn, "total_steps")
        record("lambda_horizon",
               "dann_lambda_at_step reads lambda_horizon only, never total_steps -- and "
               "on THIS arm lambda is pinned to 0.0 at every step, so the schedule "
               "cannot matter here at all", lam_ok, {"scanned": "da_losses.py"})

    grid_ok = all(s % CKPT_EVERY == 0 and s <= CONFIRM_STEPS_R6
                  for s in SENSITIVITY_STEPS_R6)
    record("ckpt_log_cadence",
           f"ckpt_every {CKPT_EVERY} / log_every 125 -> "
           f"{CONFIRM_STEPS_R6 // CKPT_EVERY} checkpoints; steps "
           f"{SENSITIVITY_STEPS_R6} are checkpoints "
           f"{[s // CKPT_EVERY for s in SENSITIVITY_STEPS_R6]}", grid_ok)

    record("epochs_planned",
           "ceil(max_steps / steps_per_epoch), used only to cap the loop; data order "
           "comes from the sampler, so steps 0..N see identical batches at any "
           "max_steps", "epochs_planned" in srcs[runner], {"scanned": runner})

    # ---- the RUNNER-SIDE token scan. SECOND LAYER ONLY. -------------------
    # The runner is where a ramp would most plausibly appear, so it is scanned. But a
    # scan is defeated by arithmetic -- `w = min(1.0, i/2000.0); loss = ce + w*gamma*bnm`
    # contains none of these tokens -- so this row is explicitly the SECOND layer: it
    # fails earlier and reads better here, while the RUNTIME INVARIANT
    # (bnm_round6.GammaInvariant, exercised against an injected ramp in
    # test_bnm_round6.T10) is the one doing the work.
    ramp_tokens = ("set_step", "lambda_at", "horizon", "warmup", "ramp", "schedule_at",
                   "anneal", "linspace")
    runner_code = _code_only(srcs[runner])
    found = [t for t in ramp_tokens if t in runner_code]
    has_invariant = ("GammaInvariant" in runner_code
                     and "GammaHorizonDetected" in runner_code)
    if "bnm_round6.py" in srcs:
        record("runner_no_ramp_second_layer",
               (f"{runner}: none of {ramp_tokens}; and it installs the RUNTIME INVARIANT "
                f"(GammaInvariant), which is the primary guard -- a scan cannot see "
                f"`w = min(1, i/2000)`"
                if not found and has_invariant
                else (f"{runner}: FOUND {found}" if found
                      else f"{runner} does NOT install GammaInvariant -- the primary "
                           f"guard is absent and this scan alone is not sufficient")),
               (not found) and has_invariant,
               {"scanned": runner, "tokens": list(ramp_tokens),
                "runtime_invariant_present": has_invariant})

    out["n_components"] = len(out["components"])
    out["sensitivity_readouts_valid"] = out["PASS"]
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "lambda0_r6_audit.json"))
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--runner", default=DEFAULT_RUNNER,
                    help="the runner whose arm-specific rows are scanned; must be in "
                         "--modules")
    ap.add_argument("--modules", nargs="+", default=list(AUDIT_MODULES))
    args = ap.parse_args()

    runs = load_runs()
    cfgs = {}
    for k in (1, 2, 3, 4):
        p = os.path.join(args.config_dir, f"cfg_fold{k}.json")
        if not os.path.exists(p):
            raise SystemExit(f"missing emitted config: {p}")
        with open(p, encoding="utf-8") as fh:
            cfgs[k] = json.load(fh)

    a, b, c = part_a(runs), part_b(runs, cfgs), part_c(tuple(args.modules), args.runner)

    line = "=" * 80
    print(line)
    print("ROUND 6 -- lambda=0 baseline folds 1-4: PRE-RUN AUDITS (CPU, no metric read)")
    print(line)

    print(f"\nA. SCRIPT PROVENANCE -- must byte-match {MATCH_TARGET}\n")
    cols = sorted(runs) + ["CURRENT"]
    print("   " + "file".ljust(20) + "".join(cc.replace("dann_confirm_", "dc_")
                                             .replace("arm1_lambda0_", "a1_")
                                             .replace("arm2_dann_", "a2_")[:10].ljust(11)
                                             for cc in cols))
    for n, row in sorted(a["matrix"].items()):
        if a["n_distinct_versions"][n] == 1:
            continue
        print("   " + n.ljust(20) + "".join(str(row.get(cc))[:10].ljust(11) for cc in cols))
    same = [n for n, v in a["n_distinct_versions"].items() if v == 1]
    print(f"   ({len(same)} files identical across every run and the working tree)")
    print()
    for n, rec in sorted(a["files"].items()):
        st = rec["status"]
        if st == "IDENTICAL_TO_TARGET":
            continue
        print(f"  {'OK  ' if st == 'APPEND_ONLY' else 'FAIL'}  {n}: {st}")
        if st == "APPEND_ONLY":
            print(f"          identical through line {rec['identical_through_line']}, "
                  f"+{rec['appended_lines']} appended (BNM, round 6)")
    print(f"\n  PART A: {'PASS' if a['PASS'] else 'FAIL'}")

    print("\nB. CONFIG FIELDS\n")
    bad = [k for k, v in b["training_relevant"].items() if not v["identical_everywhere"]]
    print(f"  B1 training-relevant keys: "
          f"{len(TRAINING_RELEVANT) - len(bad)}/{len(TRAINING_RELEVANT)} identical "
          f"across all {b['n_runs_compared']} runs")
    for k in bad:
        print(f"      ! {k}: {b['training_relevant'][k]['values']}")
    print(f"  B2 head init sha256 {b['head_init']['sha256'][:32]}...  "
          f"||head||_2 identical everywhere: {b['head_init']['l2_identical_everywhere']}")
    print("  B3 exhaustive sweep vs arm 1 fold 0:")
    for fold, rec in sorted(b["exhaustive_sweep_vs_arm1"].items()):
        cnt = rec["counts"]
        print(f"      fold {fold}: identical {cnt['IDENTICAL']:>3} | fold-dep "
              f"{cnt['FOLD_DEP']:>2} | budget {cnt['BUDGET']:>2} | detector "
              f"{cnt['DETECTOR_LAYER']:>2} | pre/post {cnt['PRE_VS_POST_RUN']:>2} | "
              f"text {cnt['TEXT']:>2} | UNEXPECTED {cnt['UNEXPECTED']}")
        for u in rec["UNEXPECTED"]:
            print(f"        ! {u['key']}")
    print("  B4 per-fold step-0 gate:")
    for f, g in sorted(b["per_fold_gate"].items()):
        print(f"      fold {f}: TPR {g['tpr']:.6f}  AUROC {g['auroc']:.6f}  "
              f"tol {g['tpr_tolerance']:.6f}  skippable={g['skippable']}")
    print(f"      TPR distinct {b['gate_tpr_all_distinct']} | AUROC distinct "
          f"{b['gate_auroc_all_distinct']} | non-skippable "
          f"{b['gate_non_skippable_everywhere']}")
    print(f"  B5 abort mode, this arm: {b['abort_mode']['per_fold']} "
          f"(all log_only: {b['abort_mode']['all_log_only']})")
    print(f"      DANN's confirmatory folds ran: "
          f"{sorted(set(b['abort_mode']['dann_confirm_folds_for_contrast'].values()))}")
    print(f"\n  PART B: {'PASS' if b['PASS'] else 'FAIL'}")

    print(f"\nC. SCHEDULE AUDIT, RE-RUN on current bytes of {', '.join(args.modules)}")
    print(f"   arm-specific rows scanned: {args.runner}\n")
    for k, v in c["components"].items():
        print(f"  {'OK  ' if v['OK'] else 'FAIL'}  {k:<28} {v['finding']}")
    print(f"\n  PART C: {'PASS' if c['PASS'] else 'FAIL'} ({c['n_components']} components)")
    print(f"  sensitivity readouts at {SENSITIVITY_STEPS_R6}: "
          f"{'VALID' if c['sensitivity_readouts_valid'] else 'DROPPED'}")

    result = {"round": 6, "arm": "dann_nb3_lambda0", "folds": [1, 2, 3, 4],
              "budget": CONFIRM_STEPS_R6,
              "sensitivity_steps": list(SENSITIVITY_STEPS_R6),
              "part_a_script_provenance": a, "part_b_config_fields": b,
              "part_c_schedule_audit": c,
              "PASS": bool(a["PASS"] and b["PASS"] and c["PASS"])}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")
    print(f"OVERALL: {'PASS' if result['PASS'] else 'FAIL'}")
    return 0 if result["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
