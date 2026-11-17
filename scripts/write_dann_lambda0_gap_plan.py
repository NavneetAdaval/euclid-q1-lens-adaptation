"""
write_dann_lambda0_gap_plan.py
==============================
Writes ``runs/dann_lambda0_holdout_gap/plan.json``: the declaration of the retrospective
DANN - lambda=0 704/78 diagnostic (PROJECTv7 7.54), self-certifying (sha256 of the body with
its own hash field removed, the dilute10 scheme), then re-reads it from disk and verifies it.

    python -u scripts/write_dann_lambda0_gap_plan.py

Hashes inputs AS BYTES ONLY. Opens no checkpoint as a model, reads no weights, computes no
statistic. Refuses if the output folder already exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
OUT_DIR = os.path.join(REPO, "runs", "dann_lambda0_holdout_gap")
PLAN = os.path.join(OUT_DIR, "plan.json")
FOLDS = (1, 2, 3, 4)
KEY = "sha256_of_body_without_this_field"


def R(*p):
    return os.path.join(REPO, *p)


def rel(p):
    return os.path.relpath(p, REPO).replace(os.sep, "/")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def self_cert(d):
    body = {k: v for k, v in d.items() if k != KEY}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def hashed(paths):
    out = {}
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"STOP: input missing: {rel(p)}")
        out[rel(p)] = sha(p)
    return out


def tar_paths():
    ps = []
    for k in FOLDS:
        if k == 1:
            ps += [R("runs", "dann_main", "fold1", f"ckpt_dann_r5_nb3_fold1_part0{p}of03.tar")
                   for p in (1, 2, 3)]
        else:
            ps += [R("runs", "dann_main", f"fold{k}", f"ckpt_fold{k}_0{p}.tar")
                   for p in range(6)]
    for k in FOLDS:
        ps += [R("runs", "baseline_unfrozen", f"fold{k}",
                 f"ckpt_lambda0_r6_fold{k}_part0{p}of03.tar") for p in (1, 2, 3)]
    return ps


def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f"REFUSING: {rel(OUT_DIR)} already exists.")

    dann_rc = [R("runs", "dann_main", f"fold{k}", "extracted", f"fold{k}", "run_config.json")
               for k in FOLDS]
    l0_rc = [R("runs", "baseline_unfrozen", f"fold{k}", "extracted", f"lambda0_r6_fold{k}",
               "run_config.json") for k in FOLDS]
    bnm_rc = [R("runs", "bnm_r6_folds", f"bnm_r6_fold{k}", "run_config.json") for k in FOLDS]
    latest = [R("runs", "baseline_unfrozen", f"fold{k}", "extracted", f"lambda0_r6_fold{k}",
                "resume", "latest.pt") for k in FOLDS]
    code = [R("scripts", n) for n in (
        "analyze_dann_lambda0_gap.py", "write_dann_lambda0_gap_plan.py", "bnm_round6.py",
        "dann_round5.py", "analyze_lambda0_r6.py", "train_skeleton.py", "lens_dataset.py",
        "integrity.py", "analyze_holdout_gap_lambda0.py", "analyze_holdout_gap_fold0.py")]

    print("hashing inputs (bytes only) ...", flush=True)
    plan = {
        "what": "the 704/78 memorisation diagnostic on DANN x nb3 against its lambda=0 control, "
                "folds 1-4, 16 checkpoints per fold per arm (PROJECTv7 7.54)",
        "status": "retrospective, unregistered: decided 2026-10-24 after all results were "
                  "known; in no pre-registration",
        "brief": "brief_dann_memorisation_diagnostic.md (author's Downloads; not in the repo)",
        "author_decisions_2026_10_24": [
            "Option A: the main analysis uses deterministic aug='none' views for both arms on "
            "all folds",
            "the known-null replays 7.29's exact random sequence; folds 2-4 must match at four "
            "decimals; fold 1 is excluded from the value check; weight equality is checked on "
            "all four folds",
            "the model is built from each checkpoint's own config via TrainConfig/Trainer, as "
            "7.29's drivers do (not the runner's build_arm)",
            "the trained-on probe is checked against BNM's recorded probe (DANN's and lambda=0's "
            "run configs record only the holdout); the holdout against both arms' run configs",
            "local RTX 4050, fp32 (amp=False); device, GPU name and torch version recorded",
        ],
        "inputs": {
            "tars": hashed(tar_paths()),
            "run_configs": hashed(dann_rc + l0_rc),
            "holdout_and_probe_records": hashed(bnm_rc),
            "resume_checkpoints": hashed(latest),
            "code": hashed(code),
            "data": hashed([R("data", "splits", "splits.json")]),
            "reference": hashed([R("runs", "round7", "holdout_gap_lambda0.json")]),
            "note": "holdout indices: dann_d_holdout_indices in each DANN and lambda=0 "
                    "run_config.json; trained-on probe: trainedon_probe.probe in each BNM "
                    "run_config.json. Tars are re-hashed by the driver before extraction.",
        },
        "views": {
            "main": "aug='none': one deterministic view per image (stretch + resize), "
                    "materialised once per fold and reused for every checkpoint of both arms",
            "why": "it adds no augmentation noise, it is exactly reproducible, and both arms "
                   "see identical images, so the DANN - lambda=0 contrast is unaffected",
            "known_null": "aug='weak': 7.29's one fixed weak-augmented view per image, "
                          "recreated by replaying 7.29's driver's exact call sequence",
        },
        "statistic": {
            "definition": "unchanged from 7.20: for each checkpoint, normalised nuclear norm "
                          "||A||*, mean prediction entropy and predicted-positive fraction on "
                          "the size-matched 78-image trained-on probe and on the 78-image "
                          "holdout, through bnm_round6.probe_stats; gap = trained-on - holdout "
                          "(bnm_round6.holdout_gap)",
            "primary_statistic": "the normalised nuclear-norm gap",
            "descriptive": "entropy gap and predicted-positive-fraction gap",
            "slices": "holdout = dann_round5.build_target_holdout(frac=0.10, seed=42+fold); "
                      "probe = bnm_round6.build_trainedon_probe(seed=42+fold); both asserted "
                      "equal to the recorded indices on every fold before anything is computed",
            "model": "TrainConfig/Trainer from the checkpoint's own cfg, init_head_from=None, "
                     "amp=False, grad_checkpointing=False; Trainer.load(load_training_state="
                     "False); head in eval mode; inference only",
            "per_checkpoint_checks": "schedule.global_step == step; schedule.fold and "
                                     "schedule.arm match the run; cfg fold, n_blocks, lr, "
                                     "weight_decay, batch_size, head_type, lr_decay equal the "
                                     "run_config; cfg.seed == 42 + fold",
        },
        "checkpoints": {"steps": list(range(250, 4001, 250)), "n_per_fold": 16,
                        "arms": ["DANN x nb3", "lambda=0"]},
        "readings": {
            "primary": {"what": "the gap at step 4000 (the pre-registered read-off of both "
                                "arms)",
                        "contrast": "DANN - lambda=0, paired t on the four per-fold "
                                    "differences, df = 3, 95% (analyze_lambda0_r6.paired_t)"},
            "secondary": {"what": "the within-fold mean of the gap over the 16 checkpoints",
                          "contrast": "same contrast and interval"},
            "descriptive_only": {
                "what": "the any-checkpoint maximum, per arm and per fold",
                "interval": None, "contrast": None,
                "reason": "a maximum over noisy reads has a positive null (7.28): the maximum "
                          "of 16 noisy reads is positive whether or not any gap exists"},
            "own_arm": "each arm's own fold-mean gap at step 4000 and as the within-fold mean, "
                       "with its paired-t interval against zero, labelled descriptive",
            "numbering_note": "7.27 numbers the readings (1) read-off, (2) any-checkpoint "
                              "maximum, (3) within-fold mean; here primary = 7.27 (1), "
                              "secondary = 7.27 (3), descriptive = 7.27 (2)",
            "excludes_zero": "paired_t's own excludes_zero: (lo * hi) > 0",
            "number_format": "[Δ] [CI] is filled as '+0.0000 [−0.0000, +0.0000]', four "
                             "decimals, Unicode minus",
        },
        "interpretation_VERBATIM": {
            "excludes_zero_positive": "Having a target objective is accompanied by a "
                                      "measurable gap between trained-on and held-out target "
                                      "images: DANN − λ=0 = [Δ] [CI] at the read-off step.",
            "excludes_zero_negative": "DANN's trained-on vs held-out gap is smaller than its "
                                      "control's: [Δ] [CI].",
            "includes_zero": "Whether part of the adaptation gain comes from fitting these "
                             "particular target images remains undetermined: DANN − λ=0 = "
                             "[Δ] [CI] at the read-off step.",
            "includes_zero_secondary_rule": "The secondary reading is then reported only "
                                            "together with the statement that it is the "
                                            "secondary.",
            "in_every_case": "In every case: no multiplicity correction, and the intervals are "
                             "probably too narrow because the folds' target pools overlap "
                             "(paper §5.3).",
        },
        "known_null": {
            "what": "lambda=0 at step 4000 must reproduce 7.29 (from resume/latest.pt): "
                    "+0.2128 / -0.1362 / +0.0624 / -0.0569 on folds 1-4",
            "procedure": "replay 7.29's driver's call sequence exactly: folds 1..4 in order; "
                         "per fold the aug='weak' dataset, the slices, both materialisations, "
                         "then Trainer(cfg from resume/latest.pt) (which seeds torch with "
                         "42 + fold, so it fixes the views of fold + 1), weights loaded from "
                         "the tar's step_4000.pt; and per fold check step_4000.pt against "
                         "resume/latest.pt by tensor equality (head, standardizer, trainable "
                         "encoder blocks)",
            "tolerance": "equal at four decimals (the printed token, and the stored json "
                         "value rounded) on folds 2, 3 and 4",
            "fold1_exclusion": "7.29 drew its fold-1 views before anything seeded torch, and "
                               "torch's default seed on this install is time-based (two fresh "
                               "processes draw different numbers), so those views cannot be "
                               "recreated. Fold 1's replay value is computed and reported, not "
                               "checked. Its weight equality IS checked.",
            "rules": {"weights identical, gap equal at 4 dp on folds 2-4": "pass",
                      "weights identical, a checked gap differs at 4 dp": "STOP and report "
                                                                         "(the pipeline "
                                                                         "differs from 7.29's)",
                      "weights not identical on any fold": "report the difference and the "
                                                           "gaps; STOP for the author"},
        },
        "disclosures": {
            "lambda0_views_differ": "The main tables' λ=0 step-4000 gaps differ from 7.29's "
                                    "because the views differ (no augmentation here vs one "
                                    "weak-augmented view in 7.29). Both sets are reported side "
                                    "by side. These values are not directly comparable with "
                                    "BNM's logged gaps (7.28), which used weak-augmented "
                                    "views.",
            "paired_t_import": "analyze_lambda0_r6 imports the metrics module at import time; "
                               "nothing in it is called, and no test logits, test labels or "
                               "scorer output are read",
        },
        "outputs": ["runs/dann_lambda0_holdout_gap/plan.json",
                    "runs/dann_lambda0_holdout_gap/known_null.json (not enrolled)",
                    "runs/dann_lambda0_holdout_gap/results.json",
                    "runs/dann_lambda0_holdout_gap/REPORT.md"],
        "driver_edit_rule": "if the driver is edited after this plan is written, the plan is "
                            "not rewritten; the edit goes into results.json as "
                            "driver_edits_after_plan with the reason",
    }
    plan[KEY] = self_cert(plan)
    os.makedirs(OUT_DIR)
    with open(PLAN, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1, ensure_ascii=False)

    back = json.load(open(PLAN, encoding="utf-8"))
    if self_cert(back) != back[KEY] or back[KEY] != plan[KEY]:
        raise SystemExit("STOP: plan.json does not re-verify from disk.")
    print(f"[plan] {rel(PLAN)}  self-certifying {back[KEY]}")
    print(f"[plan] file sha256 {sha(PLAN)}")
    print(f"[plan] driver sha256 {back['inputs']['code']['scripts/analyze_dann_lambda0_gap.py']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
