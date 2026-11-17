"""
write_dilute10_prereg.py
========================
Writes ``runs/dilute10_prereg_part1.json`` -- PART 1 OF 2 of the pre-registration for the
**1:10 diluted-target experiment** (DANN, BNM, CORAL, FreeMatch x nb3, folds 1-4;
brief_dilute10_prereg.md, PROJECTv7.md 7.36).

Part 1 is the DESIGN: what changes, the arms, budgets, read-offs, contrasts, branches,
predictions, stop conditions, the diluted pool's hashes and the cost from recorded run
times. Part 2 (a later brief) records the step-0 threshold values, their declared
statuses, the lens-fraction band and the RNG-coupling finding, and carries this file's
digest. Both are hashed and enrolled before any GPU run.

Mirrors ``write_lambda0_r6_prereg.py``'s writer: canonicalise the body, hash it, store the
hash under ``sha256_of_body_without_this_field``, re-read the file from disk and RAISE if
the digest does not reproduce.

**Every value taken from code or config is read here, at write time** -- budgets, gamma,
the lambda schedule, fairness_target, seeds, the holdout, gate tolerances, wall times, the
pool's hashes. The prose of brief sections 5, 7 and 8 is stored verbatim and checked
against the brief's own text (whitespace-normalised) before anything is written.

NO TEST LABEL, SCORE OR METRIC is read: logit files are hashed as bytes and never loaded,
and the step-0 gate VALUES (test metrics) are referenced by path and hash only -- part 2
records them. Asserted with ``code_scan.scan`` and a known-null that must fire.

    python -u scripts/write_dilute10_prereg.py --out runs/dilute10_prereg_part1.json \
        --brief C:/Users/krris/Downloads/brief_dilute10_prereg.md

Refuses to overwrite without ``--force``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import code_scan  # noqa: E402
import integrity  # noqa: E402

FOLDS = (1, 2, 3, 4)
POOL = os.path.join(REPO, "data", "splits", "dilute10_pool.json")
BUILDER = os.path.join(HERE, "build_dilute10_pool.py")
RULE = os.path.join(REPO, "audit", "decontam", "rule.json")


def R(*p):
    return os.path.join(REPO, *p)


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------------------
# the record's run artifacts (paths as analyze_window_average.ARMS / paper _common use)
# --------------------------------------------------------------------------------------
def dann_run(k):
    return R("runs", "dann_main", f"fold{k}", "extracted", f"fold{k}")


def bnm_run(k):
    return R("runs", "bnm_r6_folds", f"bnm_r6_fold{k}")


def coral_run(k):
    return R("kaggle", "coral_r4", "runs", "main", "extracted", "coral_r4_confirm", f"fold{k}")


def fm_run(k):
    return R("runs", f"freematch_nb3_unfrozen_fold{k}")


def l0_run(k):
    return R("runs", "baseline_unfrozen", f"fold{k}", "extracted", f"lambda0_r6_fold{k}")


LOGITS = {
    "DANN x nb3": lambda k, s: R(dann_run(k), "logits", f"dann_nb3_fold{k}", f"step_{s}.npz"),
    "BNM x nb3": lambda k, s: R(bnm_run(k), "logits", f"bnm_nb3_fold{k}", f"step_{s}.npz"),
    "CORAL x nb3": lambda k, s: R(coral_run(k), "logits", f"g16_2000_fold{k}", f"step_{s}.npz"),
    "FreeMatch x nb3": lambda k, s: R(fm_run(k), "logits", f"step_{s}.npz"),
    "lambda=0": lambda k, s: R(l0_run(k), "logits", f"dann_nb3_lambda0_fold{k}",
                               f"step_{s}.npz"),
}
IDS = {
    "DANN x nb3": lambda k: R(dann_run(k), "logits", "ids.npz"),
    "BNM x nb3": lambda k: R(bnm_run(k), "logits", "ids.npz"),
    "CORAL x nb3": lambda k: R(coral_run(k), "logits", "ids.npz"),
    "FreeMatch x nb3": lambda k: R(fm_run(k), "logits", "ids.npz"),
    "lambda=0": lambda k: R(l0_run(k), "logits", "ids.npz"),
}

# --------------------------------------------------------------------------------------
# no-test-label scan
# --------------------------------------------------------------------------------------
FORBIDDEN = ["metric_suite", "roc_auc_score", "roc_curve", "tpr_at_fpr", "auroc", "tpr",
             "pos_scores", "neg_scores", "score_features", "load_checkpoint", "np",
             "numpy", "test_lens_rows"]
KNOWN_NULL = ("import numpy as np\nz = np.load(p)\n"
              "m = metric_suite(y, np.concatenate([z['pos_scores'], z['neg_scores']]))\n")


def no_test_label_scan():
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    r = code_scan.scan(src, FORBIDDEN, known_null_src=KNOWN_NULL)
    if not r["clean"]:
        raise SystemExit(f"[dilute10-prereg] ABORT: the writer reads a score/metric token: "
                         f"{r['hits']}")
    return {"file": rel(os.path.abspath(__file__)), "clean": True,
            "forbidden_tokens": FORBIDDEN, "known_null_fired_on": r["known_null_hits"],
            "logit_files": "hashed as bytes, never loaded",
            "step0_gate_values": "referenced by path and hash; recorded in part 2"}


# --------------------------------------------------------------------------------------
# verbatim prose (brief sections 5, 7, 8) -- checked against the brief before writing
# --------------------------------------------------------------------------------------
S5 = ("Dilution changes three things at once: the target's class proportion (1:1 → 1:10), "
      "the number of real lenses per target batch (128 → about 23), and how often each "
      "target lens is seen (about 5.5× less often for a fixed step budget). The negative "
      "side also becomes about 10× more varied. A change in the gain therefore cannot be "
      "attributed to the proportion alone. This experiment is a first test of whether the "
      "gain survives a target far from balanced, toward survey prevalence; it does not "
      "separate adaptation to real images from the class proportion.")

S7_BRANCHES = [
    {"branch": "SURVIVES",
     "condition": "G's interval excludes zero, and D's interval includes zero or is negative",
     "licensed_sentence": (
         "\"With a 1:10 target drawn without the labels, [arm] still raises TPR@10⁻³ over "
         "the control by G [CI], and we detect no loss against the balanced target "
         "(D [CI]).\"")},
    {"branch": "SHRINKS",
     "condition": "G's interval excludes zero, and D's interval excludes zero (positive)",
     "licensed_sentence": (
         "\"With a 1:10 target, [arm] still raises TPR@10⁻³ over the control by G [CI], but "
         "by less than with a balanced target (D [CI]). Dilution also changes the number of "
         "real lenses per batch and how often each is seen, so we do not attribute the loss "
         "to the proportion alone.\"")},
    {"branch": "COLLAPSES",
     "condition": "G's interval includes zero",
     "licensed_sentence": "\"We cannot detect a gain for [arm] with a 1:10 target: G [CI].\""},
]
S7_QUALIFIER = ("**Mandatory qualifier.** Under SURVIVES, if D's upper bound exceeds half of "
                "(balanced − λ=0), add: \"…though the interval admits losing up to [D upper] "
                "of the original [balanced − λ=0].\"")
S7_ABSTRACT_RULE = ("**Abstract and conclusion rule.** Report every arm's branch. Lead with "
                    "DANN (the fully controlled pair) and BNM (whose objective depends most "
                    "directly on the class balance). Frame the experiment only as a first "
                    "test toward survey prevalence, matching §8 and the conclusion's stated "
                    "next experiment.")

S8 = [
    ("1. **DANN degrades.** G's point estimate falls below the balanced +0.31. D is positive. "
     "Given DANN's fold sd (0.17), G's interval probably includes zero (COLLAPSES, or SHRINKS "
     "with a wide D). *Reason:* under label shift, marginal alignment cannot minimise target "
     "error (Tachet des Combes et al. 2020; Tan et al. 2020), and aligning a 1:10 target with "
     "a 1:1 source pulls lens features toward galaxies. In early runs, DANN and IW-CDAN "
     "collapsed with an imbalanced target (paper §4.4). That evidence is **motivation, not "
     "measurement**: it used a frozen encoder, an earlier pipeline and a different imbalance, "
     "and its artifacts are not in the repository."),
    ("2. **CORAL degrades,** for the same reason through moment matching, but less than DANN. "
     "G's interval still excludes zero (SHRINKS or SURVIVES). *Reason:* its balanced interval "
     "is tighter ([+0.19, +0.49]), and covariance alignment is a weaker pull than an "
     "adversary."),
    ("3. **BNM degrades, the most of the four in relative terms.** G's interval may include "
     "zero. Its predicted-positive fraction drifts above the true ~0.09. *Reason:* its "
     "diversity term rewards a batch whose predictions split evenly, so at 1:10 it pushes "
     "galaxies toward the lens class. That creates high-scoring negatives, which is exactly "
     "what TPR at a fixed FPR penalises."),
    ("4. **FreeMatch degrades, less than BNM, and G's interval still excludes zero.** "
     "*Reason:* its fairness target stays uniform (no retuning), which is wrong at 1:10. Its "
     "self-adaptive per-class thresholds partly absorb the shift, and pseudo-labels still "
     "enter only above a confidence threshold. Its frozen stage is also rerun at 1:10, so its "
     "head start (0.41 on the balanced pool) is predicted to shrink."),
]


def _norm(s):
    s = re.sub(r"(?m)^\s*>\s?", "", s)            # blockquote markers
    s = s.replace("|", " ")                         # table cell separators
    return re.sub(r"\s+", " ", s).strip()


def verbatim_check(brief_path):
    txt = _norm(open(brief_path, encoding="utf-8").read())
    want = [S5, S7_QUALIFIER, S7_ABSTRACT_RULE] + S8
    for b in S7_BRANCHES:
        want += [b["condition"], b["licensed_sentence"]]
    missing = [w[:70] for w in want if _norm(w) not in txt]
    if missing:
        raise SystemExit("[dilute10-prereg] ABORT: verbatim text not found in the brief:\n  "
                         + "\n  ".join(missing))
    return {"brief_path_basename": os.path.basename(brief_path),
            "brief_sha256": sha256_file(brief_path),
            "n_passages_checked": len(want),
            "method": "each stored passage, whitespace-normalised, is a substring of the "
                      "whitespace-normalised brief (blockquote markers and table pipes removed)"}


# --------------------------------------------------------------------------------------
# values read from code and config
# --------------------------------------------------------------------------------------
def same_across_folds(get, what):
    vals = {k: get(k) for k in FOLDS}
    uniq = {json.dumps(v, sort_keys=True) for v in vals.values()}
    if len(uniq) != 1:
        raise SystemExit(f"[dilute10-prereg] ABORT: {what} differs across folds: {vals}")
    return vals[FOLDS[0]]


def read_unchanged():
    import dann_round5                                               # read, not transcribed
    dcfg = {k: jload(R(dann_run(k), "run_config.json")) for k in FOLDS}
    drun = {k: jload(R(dann_run(k), "run.json")) for k in FOLDS}
    bcfg = {k: jload(R(bnm_run(k), "run_config.json")) for k in FOLDS}
    crun = {k: jload(R(coral_run(k), "run.json")) for k in FOLDS}
    fcfg = {k: jload(R(fm_run(k), "run_config.json")) for k in FOLDS}
    fres = {k: jload(R(fm_run(k), "results.json")) for k in FOLDS}
    l0p = jload(R("runs", "lambda0_r6_prereg.json"))
    cb = jload(R("runs", "coral_r4_budget.json"))

    u = {
        "seed": same_across_folds(lambda k: [dcfg[k]["seed"], bcfg[k]["seed"],
                                             crun[k]["seed"], fcfg[k]["seed"]], "seed"),
        "coral_gamma": same_across_folds(lambda k: crun[k]["gamma"], "CORAL gamma"),
        "coral_weight_schedule": same_across_folds(lambda k: crun[k]["coral_weight_schedule"],
                                                   "CORAL schedule"),
        "coral_target_ema": same_across_folds(lambda k: crun[k]["coral_target_ema"],
                                              "CORAL target EMA"),
        "bnm_gamma": same_across_folds(lambda k: bcfg[k]["gamma"], "BNM gamma"),
        "dann_lambda_schedule": same_across_folds(lambda k: dcfg[k]["dann_lambda_schedule"],
                                                  "DANN lambda schedule"),
        "dann_lambda_horizon": same_across_folds(lambda k: dcfg[k]["dann_lambda_horizon"],
                                                 "DANN lambda horizon"),
        "freematch_fairness_target_config": same_across_folds(
            lambda k: fcfg[k]["fairness_target"], "FreeMatch fairness_target"),
        "freematch_fairness_target_effective": same_across_folds(
            lambda k: fres[k]["freematch_stability"]["sat_reconstruction"]
            ["fairness_target_after"], "FreeMatch effective fairness target"),
        "freematch_sat_passes": same_across_folds(lambda k: fcfg[k]["sat_passes"],
                                                  "FreeMatch SAT passes"),
        "holdout_frac_code": dann_round5.DEFAULT_HOLDOUT_FRAC,
        "holdout_n_dann": same_across_folds(lambda k: drun[k]["dann_d_holdout_n"],
                                            "DANN holdout n"),
        "holdout_n_bnm": same_across_folds(lambda k: bcfg[k]["target_holdout_n"],
                                           "BNM holdout n"),
        "budgets_as_run": {
            "DANN x nb3": same_across_folds(lambda k: dcfg[k]["max_steps"], "DANN budget"),
            "BNM x nb3": same_across_folds(lambda k: bcfg[k]["max_steps"], "BNM budget"),
            "CORAL x nb3": same_across_folds(lambda k: crun[k]["max_steps"], "CORAL budget"),
            "FreeMatch x nb3": same_across_folds(lambda k: fcfg[k]["max_steps"],
                                                 "FreeMatch budget"),
            "lambda=0 (prereg CONFIRM_STEPS)": l0p["budget"]["CONFIRM_STEPS"],
        },
        "coral_r4_budget_declared": cb["selected_budget_steps"],
        "step0_tolerances": {
            "DANN x nb3": same_across_folds(
                lambda k: [round(dcfg[k]["step0_tpr_tolerance"] * (97 if k == 4 else 98)),
                           dcfg[k]["step0_auroc_tolerance"]], "DANN gate tolerance"),
            "BNM x nb3": same_across_folds(
                lambda k: [round(bcfg[k]["step0_tpr_tolerance"] * (97 if k == 4 else 98)),
                           bcfg[k]["step0_auroc_tolerance"]], "BNM gate tolerance"),
        },
        "freematch_gate_tolerance_as_run": {
            str(k): {"positive_tolerance": fres[k]["step0_gate"].get("tolerance"),
                     "auroc_half": "auroc_tolerance" in fres[k]["step0_gate"],
                     "auroc_tolerance": fres[k]["step0_gate"].get("auroc_tolerance")}
            for k in FOLDS},
        "coral_gate_resolution_as_run": {
            str(k): crun[k]["step0_gate"].get("one_positive_resolution") for k in FOLDS},
    }
    # the brief's table, checked against what the runs actually used
    want = {"DANN x nb3": 4000, "BNM x nb3": 4000, "CORAL x nb3": 2000,
            "FreeMatch x nb3": 1500, "lambda=0 (prereg CONFIRM_STEPS)": 4000}
    bad = {a: (u["budgets_as_run"][a], w) for a, w in want.items()
           if u["budgets_as_run"][a] != w}
    if bad or u["coral_gamma"] != 16.0 or u["bnm_gamma"] != 0.00787 \
            or u["holdout_n_dann"] != 78 or u["holdout_n_bnm"] != 78:
        raise SystemExit(f"[dilute10-prereg] ABORT: the brief disagrees with the runs: {bad} "
                         f"gamma {u['coral_gamma']}/{u['bnm_gamma']} holdout "
                         f"{u['holdout_n_dann']}/{u['holdout_n_bnm']}")
    return u


def read_frozen_stages():
    """The round-1 frozen stages the unfrozen CORAL / FreeMatch arms started from."""
    crun = {k: jload(R(coral_run(k), "run.json")) for k in FOLDS}
    fcfg = {k: jload(R(fm_run(k), "run_config.json")) for k in FOLDS}
    c_tail = {k: crun[k]["init_head_from_tail"] for k in FOLDS}
    f_tail = {k: fcfg[k]["init_head_from"].split("/runs/", 1)[1] for k in FOLDS}
    for k in FOLDS:
        if c_tail[k] != f"runs/bakeoff_coral/fold{k}/checkpoints/step_8000.pt":
            raise SystemExit(f"[dilute10-prereg] ABORT: CORAL nb3 fold {k} head is {c_tail[k]}")
        if f_tail[k] != f"bakeoff_freematch_5fold/fold{k}/checkpoints/step_8000.pt":
            raise SystemExit(f"[dilute10-prereg] ABORT: FreeMatch nb3 fold {k} head is "
                             f"{f_tail[k]}")
    cc = jload(R("runs", "bakeoff_coral", "run_config.json"))
    fc = jload(R("runs", "bakeoff_freematch_5fold", "run_config.json"))

    # head initialisation of the round-1 frozen stages: read from the runners' code
    rb_c = open(R("scripts", "run_bakeoff_coral.py"), encoding="utf-8").read()
    rb_f = open(R("scripts", "run_bakeoff_freematch.py"), encoding="utf-8").read()
    rf = open(R("scripts", "run_freematch.py"), encoding="utf-8").read()
    head_init = {
        "run_bakeoff_coral.py passes init_head_from": "init_head_from" in
        code_scan.code_token_set(rb_c),
        "run_bakeoff_freematch.py passes init_head_from": "init_head_from" in
        code_scan.code_token_set(rb_f),
    }
    if any(head_init.values()):
        raise SystemExit(f"[dilute10-prereg] ABORT: a round-1 runner initialises its head "
                         f"from a file: {head_init}")
    cpu_coral = 'device="cpu"' in rb_c.replace(" ", "")
    fm_aug_weak = rf.count('aug="weak"')
    import train_skeleton                                            # read, not transcribed
    lr_kind = ("cosine-scheduled" if train_skeleton.ZOOBOT_COSINE_SCHEDULE
               else "constant (train_skeleton.ZOOBOT_COSINE_SCHEDULE is False)")

    keys_c = ["budget_steps", "batch_size", "lr", "weight_decay", "pos_fraction", "seed",
              "head_type", "selection_rule", "coral_peak_gamma", "coral_weight_schedule",
              "coral_warmup_frac", "coral_min_frac", "coral_shrinkage", "coral_include_mean",
              "coral_target_ema", "coral_ema_momentum", "stop_on_blowup"]
    keys_f = ["budget_steps", "batch_size", "lr", "weight_decay", "pos_fraction", "seed",
              "head_type", "selection_rule", "local_threshold_mode", "freematch_variant",
              "mu_max", "saf_weight", "sat_ema_momentum", "consistency_warmup_frac",
              "fairness_target", "pos_class", "device", "fp16"]
    return {
        "CORAL x nb3": {
            "init_head_from": {str(k): c_tail[k] for k in FOLDS},
            "init_head_from_sha256": {
                str(k): sha256_file(R(c_tail[k])) for k in FOLDS},
            "which_round1_coral_run": (
                "runs/bakeoff_coral -- the matched 8000-step round-1 CORAL bake-off, run with "
                f"coral_weight_schedule = '{cc['coral_weight_schedule']}' (peak gamma "
                f"{cc['coral_peak_gamma']}, warmup_frac {cc['coral_warmup_frac']}, min_frac "
                f"{cc['coral_min_frac']}, target EMA {cc['coral_target_ema']}). It is the "
                "warmup_cosine variant, NOT the original constant-gamma runs/coral."),
            "frozen_stage_config": {k: cc[k] for k in keys_c},
            "frozen_stage_head_init": ("fresh linear head, seed "
                                       f"{cc['seed']} (the runner passes no init_head_from)"),
            "frozen_stage_precision": "fp32" if cpu_coral else "UNKNOWN",
            "frozen_stage_device": "cpu (run_bakeoff_coral.py trains with device=\"cpu\")"
            if cpu_coral else "UNKNOWN",
            "frozen_stage_augmentation": ("none -- trains on the cached 640-d features, which "
                                          "cache_features.py builds with aug=\"none\""),
            "frozen_stage_input": "data/feature_cache (frozen encoder, n_blocks=0)",
            "run_config": "runs/bakeoff_coral/run_config.json",
            "run_config_sha256": sha256_file(R("runs", "bakeoff_coral", "run_config.json")),
        },
        "FreeMatch x nb3": {
            "init_head_from": {str(k): "runs/" + f_tail[k] for k in FOLDS},
            "init_head_from_sha256": {
                str(k): sha256_file(R("runs", f_tail[k])) for k in FOLDS},
            "frozen_stage_config": {k: fc[k] for k in keys_f},
            "frozen_stage_head_init": ("fresh linear head, seed "
                                       f"{fc['seed']} (the runner passes no init_head_from)"),
            "frozen_stage_precision": "fp16 autocast on the encoder forwards"
            if fc["fp16"] else "fp32",
            "frozen_stage_augmentation": (
                "image-space, encoder (n_blocks=0) in the loop; source = weak view only "
                "(n_views=1), target = weak + strong views (n_views=2); LensDataset "
                f"aug=\"weak\" ({fm_aug_weak} sites in run_freematch.py)"),
            "frozen_stage_schedule": (
                f"8000 steps, R1 = step 8000; consistency warm-up fraction "
                f"{fc['consistency_warmup_frac']}, mu_max {fc['mu_max']}, SAF weight "
                f"{fc['saf_weight']}, SAT EMA {fc['sat_ema_momentum']}, lr {fc['lr']} "
                f"{lr_kind}"),
            "run_config": "runs/bakeoff_freematch_5fold/run_config.json",
            "run_config_sha256": sha256_file(R("runs", "bakeoff_freematch_5fold",
                                               "run_config.json")),
        },
        "DANN x nb3": {"init_head_from": "runs/bakeoff_baseline_A/checkpoints/step_8000.pt "
                                         "(Baseline A, unchanged; no frozen stage is rerun)"},
        "BNM x nb3": {"init_head_from": "runs/bakeoff_baseline_A/checkpoints/step_8000.pt "
                                        "(Baseline A, unchanged; no frozen stage is rerun)"},
    }


# --------------------------------------------------------------------------------------
# cost, from recorded run times
# --------------------------------------------------------------------------------------
LANE_HOURS = 10.0      # usable hours per T4 lane: the 12 h Kaggle cap less a 2 h margin
LANES = 2              # T4 x2, two folds concurrently (PROJECTv7.md, dual-GPU runs)


def epoch_seconds_from_log(path):
    s = [float(x) for x in re.findall(r"\[ep +\d+\]\s+([0-9.]+)s", open(
        path, encoding="utf-8", errors="replace").read())]
    return len(s), sum(s)


def pack_sessions(minutes):
    """First-fit-decreasing of whole folds onto LANES x LANE_HOURS sessions."""
    lanes = []
    for m in sorted(minutes, reverse=True):
        for i, used in enumerate(lanes):
            if used + m <= LANE_HOURS * 60:
                lanes[i] += m
                break
        else:
            lanes.append(m)
    return math.ceil(len(lanes) / LANES) if lanes else 0


def read_cost(pool):
    rows, notes = {}, {}

    def put(arm, k, minutes, source, gpu):
        rows.setdefault(arm, {})[str(k)] = {"minutes": minutes, "gpu_hours": minutes / 60,
                                            "source": source, "gpu": gpu}

    for k in FOLDS:
        j = jload(R(dann_run(k), "run.json"))
        put("DANN x nb3 (rerun, 4000 steps)", k, j["wall_minutes"],
            rel(R(dann_run(k), "run.json")) + ":wall_minutes", j["gpu_name"])
        j = jload(R(bnm_run(k), "run.json"))
        put("BNM x nb3 (rerun, 4000 steps)", k, j["wall_minutes"],
            rel(R(bnm_run(k), "run.json")) + ":wall_minutes", j["gpu_name"])
        j = jload(R(coral_run(k), "run.json"))
        put("CORAL x nb3 (rerun, 2000 steps)", k, j["wall_minutes"],
            rel(R(coral_run(k), "run.json")) + ":wall_minutes", j["gpu_name"])

    # FreeMatch nb3: recorded train_minutes + the larger SAT rebuild
    rs = jload(R("runs", "dann_r5_curves", "dann_r5_rescore.json"))
    score_s = [row["score_seconds"] for arm in rs["arms"] for row in arm["rows"]]
    n_img = {arm["rows"][0]["n_pos"] + arm["rows"][0]["n_neg"] for arm in rs["arms"]}
    assert len(n_img) == 1, n_img
    n_img = n_img.pop()
    sec_per_img = statistics.median(score_s) / n_img
    notes["forward_throughput"] = {
        "source": "runs/dann_r5_curves/dann_r5_rescore.json rows[*].score_seconds "
                  "(nb3 checkpoints through live_encode on a T4)",
        "median_seconds_per_pass": statistics.median(score_s), "images_per_pass": n_img,
        "n_passes": len(score_s), "images_per_second": 1 / sec_per_img}

    pool_folds = {fd["fold"]: fd for fd in pool["folds"]}
    sat = {}
    for k in FOLDS:
        res = jload(R(fm_run(k), "results.json"))
        st = res["freematch_stability"]
        sr = st["sat_reconstruction"]
        n_pool = pool_folds[k]["n_lenses"] + pool_folds[k]["n_negatives"]
        nb_new = int(sr["n_passes"]) * math.ceil(n_pool / int(sr["batch_size"]))
        extra = nb_new - int(sr["n_batches"])
        epoch_secs = sum(e.get("secs", 0.0) for e in st["epochs"])
        step_s = epoch_secs / int(st["stopped_at_step"])
        central = extra * int(sr["batch_size"]) * sec_per_img / 60
        upper = extra * step_s / 60
        sat[str(k)] = {"pool_before": int(sr["pool_size"]), "pool_after": n_pool,
                       "batches_before": int(sr["n_batches"]), "batches_after": nb_new,
                       "extra_batches": extra,
                       "extra_minutes_central": central, "extra_minutes_upper": upper,
                       "central_basis": "extra batches x 256 images at the recorded forward "
                                        "throughput",
                       "upper_basis": "extra batches x this fold's recorded mean seconds per "
                                      "TRAINING step (forward+backward, all three streams)"}
        put("FreeMatch x nb3 (rerun, 1500 steps)", k, res["train_minutes"] + upper,
            rel(R(fm_run(k), "results.json")) + ":train_minutes + SAT rebuild upper bound",
            "Tesla T4 (PROJECTv7.md 7.12; not recorded in the run's files)")
        rows["FreeMatch x nb3 (rerun, 1500 steps)"][str(k)]["recorded_train_minutes"] = \
            res["train_minutes"]
    notes["freematch_sat_rebuild"] = sat
    notes["freematch_train_minutes_covers"] = (
        "time.time() around train_freematch_fold: SAT rebuild, Adam warm-up, step-0 "
        "evaluation and training; NOT encoder build or packaging")

    # round-1 frozen FreeMatch (image-space, T4): logged per-epoch seconds
    for k in FOLDS:
        p = R("runs", "bakeoff_freematch", f"fold{k}", f"bakeoff_freematch_fold{k}.log")
        n, s = epoch_seconds_from_log(p)
        put("FreeMatch frozen stage (rerun, 8000 steps)", k, s / 60,
            rel(p) + f": sum of {n} logged per-epoch seconds",
            "Tesla T4 (PROJECTv7.md: image-space methods ran on Kaggle T4)")
    notes["freematch_frozen_logged_time_covers"] = (
        "the 43 logged epoch durations only -- excludes encoder build, the curve-point "
        "evaluations if they fall outside the epoch timer, and packaging; the round-1 logs "
        "record no session start/end")

    # round-1 frozen CORAL: CPU, cached features -- no logged wall time exists
    mt = {k: os.path.getmtime(R("runs", "bakeoff_coral", f"fold{k}", "curve.csv"))
          for k in (0, 1, 2, 3, 4)}
    ck = sorted(glob.glob(R("runs", "bakeoff_coral", "fold1", "checkpoints", "step_*.pt")),
                key=os.path.getmtime)
    span_f1 = (os.path.getmtime(ck[-1]) - os.path.getmtime(ck[0])) / 60 if ck else None
    for k in FOLDS:
        put("CORAL frozen stage (rerun, 8000 steps, CPU)", k, 0.0,
            "no logged wall time exists (bakeoff_coral_folds1234.log has no timestamps); "
            "CPU, so 0 GPU-hours", "cpu")
        rows["CORAL frozen stage (rerun, 8000 steps, CPU)"][str(k)]["cpu_minutes_from_mtimes"] = \
            ((mt[k] - mt[k - 1]) / 60) if k > 1 else None
    notes["coral_frozen_mtimes"] = {
        "basis": "FILESYSTEM mtimes of runs/bakeoff_coral/fold*/curve.csv, NOT a logged time",
        "fold_to_fold_minutes": {f"{k - 1}->{k}": (mt[k] - mt[k - 1]) / 60 for k in (2, 3, 4)},
        "fold1_first_to_last_checkpoint_minutes": span_f1}

    # the frozen cache for the new galaxies (CORAL frozen stage)
    n_new = sum(fd["n_fresh_negatives"] for fd in pool["folds"])
    enc = n_new * sec_per_img / 60
    rows["CORAL frozen-cache encode (new galaxies)"] = {
        "all": {"minutes": enc, "gpu_hours": enc / 60, "n_images": n_new,
                "source": "n_fresh_negatives x the recorded forward throughput (derived, "
                          "one frozen forward per image, aug='none')",
                "gpu": "any (T4 rate used)"}}

    summary = {}
    total_gpu_h = 0.0
    for arm, per in rows.items():
        mins = [v["minutes"] for v in per.values()]
        g = sum(v["gpu_hours"] for v in per.values())
        gpu_mins = [m for m, v in zip(mins, per.values()) if v.get("gpu") != "cpu"]
        summary[arm] = {"gpu_hours": g, "kaggle_sessions_if_run_alone":
                        pack_sessions(gpu_mins) if g > 0 else 0}
        total_gpu_h += g
    gpu_jobs = [v["minutes"] for per in rows.values() for v in per.values()
                if v.get("gpu") != "cpu" and v["minutes"] > 0]
    return {
        "per_arm_per_fold": rows,
        "per_arm": summary,
        "total_gpu_hours": total_gpu_h,
        "total_kaggle_sessions_lower_bound": pack_sessions(gpu_jobs),
        "total_kaggle_sessions_if_each_arm_separate": sum(
            s["kaggle_sessions_if_run_alone"] for s in summary.values()),
        "session_rule": (f"a Kaggle session = T4 x{LANES}, two folds concurrently, whole folds "
                         f"only, at most {LANE_HOURS:g} h per lane (the 12 h cap less a 2 h "
                         "margin, PROJECTv7.md 7.11); first-fit-decreasing"),
        "ordering_constraint": ("each fold's FreeMatch / CORAL frozen stage must finish and "
                                "its head be home before that fold's unfrozen stage starts, so "
                                "the lower bound is not reachable in one pass"),
        "notes": notes,
    }


# --------------------------------------------------------------------------------------
# artifact paths and hashes
# --------------------------------------------------------------------------------------
def artifacts():
    out = {}
    readouts = {"DANN x nb3": [4000], "BNM x nb3": [4000], "CORAL x nb3": [2000],
                "FreeMatch x nb3": [1500], "lambda=0": [4000, 2000, 1500]}
    missing = []
    for arm, steps in readouts.items():
        out[arm] = {}
        for k in FOLDS:
            d = {}
            for s in steps:
                p = LOGITS[arm](k, s)
                if not os.path.isfile(p):
                    missing.append(rel(p))
                    continue
                d[f"step_{s}"] = {"path": rel(p), "sha256": sha256_file(p)}
            p = IDS[arm](k)
            d["ids"] = {"path": rel(p), "sha256": sha256_file(p)}
            out[arm][str(k)] = d
    out["round1_frozen_scores (descriptive comparison, section 6)"] = {
        run: {str(k): {"path": rel(p), "sha256": sha256_file(p)}
              for k in FOLDS
              for p in sorted(glob.glob(R("runs", run, "scores", f"*_fold{k}.npz")))}
        for run in ("bakeoff_freematch_5fold", "bakeoff_coral")}
    out["step0_gate_reference_values (read in part 2, not here)"] = {
        "DANN x nb3": {str(k): {"path": rel(R(dann_run(k), "run_config.json")),
                                "fields": ["expect_step0_tpr", "expect_step0_auroc"],
                                "sha256": sha256_file(R(dann_run(k), "run_config.json"))}
                       for k in FOLDS},
        "BNM x nb3": {str(k): {"path": rel(R(bnm_run(k), "run_config.json")),
                               "fields": ["expect_step0_tpr", "expect_step0_auroc"],
                               "sha256": sha256_file(R(bnm_run(k), "run_config.json"))}
                      for k in FOLDS},
    }
    if missing:
        raise SystemExit("[dilute10-prereg] ABORT (stop condition 7 shape): stored logits "
                         "missing:\n  " + "\n  ".join(missing))
    return out


# --------------------------------------------------------------------------------------
def build_body(brief_path):
    pool = jload(POOL)
    builder_now = sha256_file(BUILDER)
    if pool["builder_sha256"] != builder_now:
        raise SystemExit(f"[dilute10-prereg] ABORT: the pool was built by builder "
                         f"{pool['builder_sha256']}, but the builder on disk is {builder_now}")
    if not pool["audit"]["passed"]:
        raise SystemExit("[dilute10-prereg] ABORT: the pool's audit did not pass")
    rule = jload(RULE)
    u = read_unchanged()
    heads = read_frozen_stages()
    cost = read_cost(pool)
    arts = artifacts()
    vb = verbatim_check(brief_path)
    a = pool["audit"]

    return {
        "experiment": "dilute10",
        "status": "part 1 of 2 — design; step-0 threshold values pending (part 2)",
        "recorded_before_any_run": True,
        "revision_rule": ("Nothing here may be revised after any run of this experiment "
                          "starts. Part 2 adds, it does not amend."),
        "brief": vb,
        "no_test_label_scan": no_test_label_scan(),

        # ---------------------------------------------------------------- section 1
        "what_the_experiment_is": {
            "arms": ["DANN x nb3", "BNM x nb3", "CORAL x nb3", "FreeMatch x nb3"],
            "folds": list(FOLDS),
            "change_1_pool": ("The target pool is diluted to 1:10. Each fold's target lenses "
                              "are unchanged (391; 392 on fold 4). Its galaxies are the "
                              "fold's existing target negatives plus fresh P2 draws, up to "
                              "exactly 10x the number of lenses."),
            "change_2_sampler": ("Target batches are drawn uniformly without replacement from "
                                 "the training pool, cycling, with no use of the latent "
                                 "labels."),
            "unchanged": {
                "heads_and_head_initialisation": True,
                "optimiser_settings": True,
                "coral_gamma": u["coral_gamma"],
                "coral_weight_schedule": u["coral_weight_schedule"],
                "coral_target_ema": u["coral_target_ema"],
                "bnm_gamma": u["bnm_gamma"],
                "dann_lambda_schedule": u["dann_lambda_schedule"],
                "dann_lambda_horizon": u["dann_lambda_horizon"],
                "freematch_defaults": {
                    "fairness_target_config": u["freematch_fairness_target_config"],
                    "fairness_target_effective": u["freematch_fairness_target_effective"],
                    "sat_passes": u["freematch_sat_passes"],
                    "note": "not retuned for 1:10"},
                "seed": u["seed"],
                "budgets_read_offs_and_step0_gates": True,
                "source_loader": True,
                "dann_bnm_holdout": {
                    "n": u["holdout_n_dann"], "frac_in_code": u["holdout_frac_code"],
                    "requirement": "byte-identical to the original runs (the same 78 objects)",
                    "how_the_pool_supports_it": (
                        "existing negatives keep their dataset positions (lenses, then the "
                        "original negatives, then the fresh draws), so the recorded holdout "
                        "positions name the same objects; the code brief must pin the holdout "
                        "to those 78 rather than re-drawing 10% of the larger pool"),
                },
            },
            "frozen_stages_rerun": ("The frozen stages that the unfrozen CORAL and FreeMatch "
                                    "arms start from are also rerun under the new pool and "
                                    "sampler (section 4.2)."),
            "lambda0_control": "reused, not rerun",
            "author_declinable": ("The CORAL frozen-stage rerun may be declined by the author; "
                                  "if so, delete it here, in section 4.2, and its rows in "
                                  "sections 6 and 7 (brief 4.2). Recorded as INCLUDED."),
        },

        # ---------------------------------------------------------------- the pool
        "pool": {
            "path": rel(POOL),
            "whole_file_sha256": sha256_file(POOL),
            "builder": rel(BUILDER),
            "builder_sha256": builder_now,
            "seed": pool["seed"],
            "seed_namespacing": pool["seed_namespacing"],
            "per_fold": {str(fd["fold"]): {
                "n_lenses": fd["n_lenses"], "n_existing_negatives": fd["n_existing_negatives"],
                "n_fresh_negatives": fd["n_fresh_negatives"], "n_negatives": fd["n_negatives"]}
                for fd in pool["folds"]},
            "realised_ratios": a["realised_ratios"],
            "screened_out_of_P2": a["screened_out_of_P2"],
            "capacity": a["capacity"],
            "existing_negatives_screen_hits_NOT_REMOVED":
                a["existing_negatives_screen_hits_NOT_REMOVED"],
            "reference_sets": pool["reference_sets"],
            "inputs_sha256": pool["inputs_sha256"],
            "expected_lens_fraction_per_batch": {
                str(fd["fold"]): fd["n_lenses"] / (fd["n_lenses"] + fd["n_negatives"])
                for fd in pool["folds"]},
        },

        # ---------------------------------------------------------------- section 4
        "arms_folds_budgets_readoffs": {
            "unfrozen": [
                {"arm": "DANN x nb3", "starts_from": "Baseline A head (unchanged)",
                 "trained_to": u["budgets_as_run"]["DANN x nb3"], "read_at": 4000,
                 "contrast_against_lambda0_at": {"primary": 4000}},
                {"arm": "BNM x nb3", "starts_from": "Baseline A head (unchanged)",
                 "trained_to": u["budgets_as_run"]["BNM x nb3"], "read_at": 4000,
                 "contrast_against_lambda0_at": {"primary": 4000}},
                {"arm": "CORAL x nb3", "starts_from": "NEW frozen-stage head (section 4.2)",
                 "trained_to": u["budgets_as_run"]["CORAL x nb3"], "read_at": 2000,
                 "contrast_against_lambda0_at": {"primary": 4000, "sensitivity": 2000}},
                {"arm": "FreeMatch x nb3",
                 "starts_from": ("NEW frozen-stage head (section 4.2), with the Appendix E "
                                 "state reconstruction"),
                 "trained_to": u["budgets_as_run"]["FreeMatch x nb3"], "read_at": 1500,
                 "contrast_against_lambda0_at": {"primary": 4000, "sensitivity": 1500}},
            ],
            "budgets_checked_against_the_runs": u["budgets_as_run"],
            "step0_gates": {
                "DANN_and_BNM": ("keep their per-fold step-0 gates against Baseline A's "
                                 "values (the original runs' expect_step0_* fields, "
                                 "referenced under artifacts)"),
                "CORAL_and_FreeMatch": ("taken against THEIR NEW FROZEN HEAD'S OWN per-fold "
                                        "values, recorded when section 4.2 finishes"),
                "tolerance": ("unchanged: one positive on the live-encode path, plus "
                              "|AUROC diff| <= 1e-3"),
                "tolerance_as_run_dann_bnm_positives_and_auroc": u["step0_tolerances"],
                "tolerance_as_run_freematch": u["freematch_gate_tolerance_as_run"],
                "tolerance_as_run_coral_one_positive_resolution":
                    u["coral_gate_resolution_as_run"],
                "DISCLOSURE": ("FreeMatch x nb3 fold 1 originally ran its gate at an EXACT "
                               "tolerance with no AUROC half; folds 2-4 ran at one positive "
                               "+ AUROC 1e-3. The rerun uses one positive + AUROC 1e-3 on "
                               "every fold."),
            },
            "frozen_stages_rerun": {
                "FreeMatch frozen": ("an exact replica of round-1 frozen FreeMatch "
                                     "(image-space, weak augmentation, fp16, 8000 steps, R1 = "
                                     "step 8000), except for the pool and the sampler"),
                "CORAL frozen": ("an exact replica of the round-1 frozen CORAL run CORAL nb3 "
                                 "started from (runs/bakeoff_coral, warmup_cosine), except for "
                                 "the pool and the sampler"),
                "their_final_heads": "initialise section 4.1",
                "their_own_tpr": "a DESCRIPTIVE SECONDARY (section 6)",
                "round1_configs": {"CORAL": heads["CORAL x nb3"]["frozen_stage_config"],
                                   "FreeMatch": heads["FreeMatch x nb3"]["frozen_stage_config"]},
            },
            "lambda0_control": {
                "reused": "the existing lambda=0 folds 1-4",
                "why": ("they pass no target gradient, so the target pool and sampler do not "
                        "enter what the model learns"),
                "CAVEAT_declared_now": (
                    "If the target sampler shares a random number stream with the source "
                    "sampler or the augmentation, changing it reorders the source batches. In "
                    "that case the lambda=0 run is 'the same model up to batch order'. This is "
                    "disclosed, not rerun. Resolved by the code brief."),
            },
        },

        # ---------------------------------------------------------------- section 5
        "what_dilution_changes_VERBATIM": S5,

        # ---------------------------------------------------------------- section 6
        "contrasts": {
            "common_method": {
                "pool": ("cleaned test pool, rule audit/decontam/rule.json body sha256 "
                         f"{rule['sha256_of_body_without_this_field']}"),
                "metric": "TPR@1e-3",
                "folds": list(FOLDS),
                "interval": "paired t-interval on the four per-fold differences, df = 3",
                "computation": ("every value from stored logits through metrics.metric_suite, "
                                "never from printed tables (7.31)"),
            },
            "primary": {
                "what": "for each arm, diluted - lambda=0, at the steps in section 4.1",
                "list": ["DANN@4000 - l0@4000", "BNM@4000 - l0@4000",
                         "CORAL@2000 - l0@4000", "CORAL@2000 - l0@2000 (sensitivity)",
                         "FreeMatch@1500 - l0@4000", "FreeMatch@1500 - l0@1500 (sensitivity)"],
                "n_intervals": 6,
            },
            "secondary": {
                "what": ("for each arm, balanced - diluted at the same read-off; balanced "
                         "per-fold values from the stored logits of the paper's runs"),
                "n_intervals": 4,
                "verdict": "NONE -- reported as intervals",
                "caveat": "they include batch-order noise, because the sampler changes the "
                          "draw sequence",
            },
            "descriptive": [
                "retained fraction (diluted - lambda=0)/(balanced - lambda=0), point estimate "
                "only",
                "frozen-stage diluted FreeMatch and CORAL against their round-1 frozen heads, "
                "folds 1-4, class-conditional paired bootstrap as in the frozen comparison "
                "(2 intervals)",
                "realised lens fraction per target batch, mean and sd per run",
                "BNM's predicted-positive fraction trajectory",
            ],
            "multiplicity": ("10 new t-intervals plus 2 bootstrap intervals, NO correction. "
                             "Disclosed."),
            "sensitivities": "original and reduced pools in an appendix only",
        },

        # ---------------------------------------------------------------- section 7
        "branches": {
            "definitions": {"G": "diluted - lambda=0", "D": "balanced - diluted"},
            "per_arm": True,
            "table_VERBATIM": S7_BRANCHES,
            "mandatory_qualifier_VERBATIM": S7_QUALIFIER,
            "abstract_and_conclusion_rule_VERBATIM": S7_ABSTRACT_RULE,
        },

        # ---------------------------------------------------------------- section 8
        "predictions_VERBATIM": {
            "rule": "recorded verbatim; scored either way, never amended",
            "items": S8,
            "numbers_inside_NOT_checked_here": (
                "the balanced values quoted in the reasons (+0.31, sd 0.17, [+0.19, +0.49], "
                "0.41) are test metrics; this part reads none, so they are recorded as the "
                "brief states them"),
        },

        # ---------------------------------------------------------------- section 9
        "stop_conditions": [
            ("1. The code brief's known-null fails. With the option off, every arm's emitted "
             "config and first target and source batches must be bit-identical to the "
             "original runs. If not, run nothing."),
            "2. A step-0 gate fails: that fold stops.",
            ("3. A threshold evaluated at step 0 on the new pool would already fire, or is "
             "degenerate (for example, BNM's drift floor below 0). It is declared ADVISORY "
             "WITH ITS REASON, in part 2, before any run. It is never loosened until it "
             "cannot fire."),
            ("4. The realised mean lens fraction of a run's target batches falls outside the "
             "band derived in part 2 (expected about 0.091). This indicates a sampler bug: "
             "stop."),
            ("5. BNM's gamma invariant breaks, or DANN's armed conditions 1-2 fire: as in the "
             "original runs."),
            ("6. A fold misses its budget: stop and report. Never rerun a completed fold. A "
             "fold may be rerun only for a logged infrastructure failure, with the reason "
             "recorded."),
            ("7. Any per-checkpoint logits are missing at packaging: refuse, as in the round-6 "
             "cells."),
        ],

        # ---------------------------------------------------------------- section 3
        "cost": cost,
        "head_provenance": heads,
        "artifacts": arts,

        "part_2_will_record": [
            "the step-0 values of every threshold on the new pool and each one's declared "
            "status (armed / advisory with reason)",
            "the lens-fraction band for stop condition 4",
            "the RNG-coupling finding (section 4.3 caveat)",
            "the per-fold step-0 gate values of the new CORAL / FreeMatch frozen heads (when "
            "section 4.2 finishes)",
            "this file's digest",
        ],
        "recorded_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write the dilute10 pre-registration, part 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--brief", required=True, help="path to brief_dilute10_prereg.md")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    if os.path.exists(out) and not a.force:
        raise SystemExit(f"[dilute10-prereg] refusing to overwrite {out} (pass --force)")

    integrity.verify(where="(write_dilute10_prereg start)")
    body = json.loads(json.dumps(build_body(a.brief)))
    h = hashlib.sha256(_canonical(body).encode()).hexdigest()
    body["sha256_of_body_without_this_field"] = h

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2, sort_keys=True)

    with open(out, encoding="utf-8") as fh:
        chk = json.load(fh)
    got = chk.pop("sha256_of_body_without_this_field")
    again = hashlib.sha256(_canonical(chk).encode()).hexdigest()
    if got != again or got != h:
        raise SystemExit(f"[dilute10-prereg] DIGEST MISMATCH on re-read: stored {got}, "
                         f"recomputed {again}")
    integrity.verify(where="(write_dilute10_prereg end)")

    c = body["cost"]
    print("=" * 78)
    print("DILUTE10 PRE-REGISTRATION, PART 1 -- design")
    print("=" * 78)
    print(f"  file         : {out}")
    print(f"  sha256       : {got}")
    print(f"  verify       : OK -- re-read from disk and reproduces")
    print(f"  recorded_utc : {body['recorded_utc']}")
    print(f"  pool sha256  : {body['pool']['whole_file_sha256']}")
    print(f"  builder sha  : {body['pool']['builder_sha256']}")
    print("  cost (GPU-h, sessions if run alone):")
    for arm, s in c["per_arm"].items():
        per = c["per_arm_per_fold"][arm]
        fm = "  ".join(f"f{k}={v['minutes']:.1f}m" for k, v in per.items())
        print(f"    {arm:<45} {s['gpu_hours']:6.2f}  {s['kaggle_sessions_if_run_alone']}   {fm}")
    print(f"  total GPU-h {c['total_gpu_hours']:.2f}; sessions >= "
          f"{c['total_kaggle_sessions_lower_bound']} (packed), "
          f"{c['total_kaggle_sessions_if_each_arm_separate']} if each arm runs separately")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
