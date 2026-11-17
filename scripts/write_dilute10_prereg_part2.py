"""
write_dilute10_prereg_part2.py
==============================
Writes ``runs/dilute10_prereg_part2.json`` -- PART 2 OF 2 of the 1:10 diluted-target
pre-registration (brief_dilute10_part2 section D; PROJECTv7.md 7.38). Final: no
confirmatory run may start before this file is enrolled.

Mirrors the part-1 writer: canonical body, sha256 under
``sha256_of_body_without_this_field``, re-read from disk, RAISE on a digest mismatch.

Every value is READ at write time -- part 1's digest from part 1 itself, the step-0 values
from the scratch outputs of ``dilute10_step0_eval.py``, the worker counts from the original
runs' configs, the restoration hashes from the files, the audit summary from
``audit/script_provenance/report.json``, the RNG finding from the heavy-test reports.
Threshold statuses are DERIVED by the declared rule (a threshold that already fires at steps
0-10, or is degenerate, becomes advisory with its reason; every other keeps its original
status), not typed in.

    python -u scripts/write_dilute10_prereg_part2.py --out runs/dilute10_prereg_part2.json \
        --scratch C:/Users/krris/ML/scratch_dilute10_step0 --tests <dir with heavy_*.json>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_pool as DP                                   # noqa: E402

FOLDS = (1, 2, 3, 4)
LENS_BAND = 0.002


def R(*p):
    return os.path.join(REPO, *p)


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def scratch(sdir, what, k):
    p = os.path.join(sdir, f"{what}_fold{k}.json")
    if not os.path.isfile(p):
        raise SystemExit(f"[part2] ABORT: missing step-0 output {p}")
    return jload(p)


def dann_thresholds(sdir):
    import dann_round5 as D5
    out = {}
    for k in FOLDS:
        d = scratch(sdir, "dann", k)
        s = d["series_steps_1_10"]
        t0 = d["true_step0"]
        hold = [t0["d_acc_holdout"]] + s["d_acc_holdout"]
        psrc = s["d_pred_source_frac"]
        probe = [t0["morph"].get("morph_proxy_auroc")] + s["morph_proxy_auroc"]
        lo, hi = D5.COND2_MARGINAL_BAND
        fired_cols = {c: v for c, v in s.items() if c.endswith("_fired") and any(v)}
        out[str(k)] = {
            "venue": d["venue"], "steps": "true step 0 + logged steps 1-10",
            "num_workers": d.get("num_workers", 2),
            "num_workers_note": d.get("workers_note", "workers as declared (2); output "
                                      "written before the field existed, when 2 was fixed"),
            "cond1": {"name": "abort1_d_overpowering", "original_status": "armed",
                      "rule": f"d_acc_holdout > {D5.COND1_HOLDOUT_ACC} sustained "
                              f"{D5.COND1_SUSTAIN} steps",
                      "step0": t0["d_acc_holdout"], "step1": s["d_acc_holdout"][0],
                      "max_0_10": max(hold),
                      "domain_loss_step1": s["loss_domain_raw"][0],
                      "domain_loss_step10": s["loss_domain_raw"][-1],
                      "domain_loss_falling": d["domain_loss_falling"],
                      "breaches_at_0_10": max(hold) > D5.COND1_HOLDOUT_ACC},
            "cond2": {"name": "abort2_degenerate_collapse", "original_status": "armed",
                      "rule": f"d_pred_source_frac outside [{lo}, {hi}] sustained "
                              f"{D5.COND2_SUSTAIN} steps",
                      "step0": None, "step0_note": "not defined before the first D forward",
                      "step1": psrc[0], "min_1_10": min(psrc), "max_1_10": max(psrc),
                      "breaches_at_0_10": any(v < lo or v > hi for v in psrc)},
            "cond3b": {"name": "abort3b_morph_sign_flip", "original_status": "advisory",
                       "rule": f"morph_proxy_auroc < {D5.COND3_SIGN_FLIP} (sustain "
                               f"{D5.COND3B_SUSTAIN})",
                       "step0": probe[0], "step1": s["morph_proxy_auroc"][0],
                       "min_0_10": min(probe),
                       "breaches_at_0_10": min(probe) < D5.COND3_SIGN_FLIP},
            "cond3a": {"name": "abort3_morph_vs_control", "original_status": "advisory",
                       "active_as_confirmatory_folds_ran_it": False,
                       "why": ("margin M UNSET (7.19: uncalibrable) and no control curve; it "
                               "computes nothing"),
                       "margin": d["cond3_margin"], "control_curve": d["control_curve"]},
            "cond4": {"name": "abort4src/abort4tgt_rank_collapse",
                      "original_status": "advisory",
                      "active_as_confirmatory_folds_ran_it": False,
                      "why": "differential vs control; margins UNSET and no control curve",
                      "margins": d["cond4_margins"],
                      "src_participation_ratio_1_10": [min(s["src_participation_ratio"]),
                                                       max(s["src_participation_ratio"])],
                      "tgt_participation_ratio_1_10": [min(s["tgt_participation_ratio"]),
                                                       max(s["tgt_participation_ratio"])]},
            "detectors_fired_in_steps_1_10": sorted(fired_cols),
            "aborted_in_10_steps": d["aborted"],
            "holdout_composition": d["holdout_composition"],
            "training_target": d["training_target"],
            "holdout_note": ("the holdout is 39 lenses : 39 galaxies while D trains on ~1:11. "
                             "At 1:10, D may classify the holdout's lenses as source-like, "
                             "which would make condition 1 LESS likely to fire."),
            "lambda_steps_1_10": s["lambda_current"],
        }
    return out


def bnm_thresholds(sdir):
    out = {}
    for k in FOLDS:
        b = scratch(sdir, "bnm", k)
        band = b["collapse_band_on_diluted_pool"]
        g = b["gradnorms"]
        t4 = b["gradnorms_balanced_t4"]
        out[str(k)] = {
            "venue": b["venue"],
            "gradnorm_num_workers": b.get("gradnorm_num_workers", 2),
            "gradnorm_workers_note": b.get("gradnorm_workers_note", "workers as declared (2); "
                                           "output written before the field existed"),
            "step0_pred_pos_frac_diluted_training_pool":
                b["step0_pred_pos_frac_diluted_training_pool"],
            "step0_pred_pos_frac_balanced_training_pool_same_fold":
                b["step0_pred_pos_frac_balanced_training_pool_same_fold"],
            "original_anchor": b["original_anchor"],
            "collapse_floor": {"original_status": "gating (Stage 2 gamma selection; the "
                                                  "collapse band's gating half)",
                               "rule": "floor = step-0 predicted-positive fraction - 0.20",
                               "step0_value": band["step0_value"], "floor": band["floor"],
                               "degenerate": band["floor"] <= 0.0},
            "ceiling": {"original_status": "advisory", "value": band["ceiling"],
                        "step0_value": band["step0_value"],
                        "breaches_at_step0": band["ceiling_fires_at_step0"]},
            "gradnorms_diluted": g, "gradnorms_balanced_t4": t4,
            "ratio_diluted_over_balanced_t4": (
                g["ratio_bnm_over_ce_at_gamma_1"] / t4["ratio_bnm_over_ce_at_gamma_1"]
                if t4.get("ratio_bnm_over_ce_at_gamma_1") else None),
            "gamma_unchanged": b["gamma_unchanged"],
            "gradnorm_note": ("descriptive only; gamma stays 0.00787. Diluted values are on "
                              "a local Ada GPU, the balanced reference on a Kaggle T4 (7.7)"),
        }
    return out


def declare(threshold_id, value, original_status, fires, degenerate, reason_fire):
    if degenerate:
        return {"threshold": threshold_id, "value": value, "original_status": original_status,
                "declared_status": "advisory",
                "reason": "degenerate on the diluted pool: " + reason_fire}
    if fires:
        return {"threshold": threshold_id, "value": value, "original_status": original_status,
                "declared_status": "advisory",
                "reason": "already fires at steps 0-10: " + reason_fire}
    return {"threshold": threshold_id, "value": value, "original_status": original_status,
            "declared_status": original_status,
            "reason": "does not fire at steps 0-10 on any fold; original status kept"}


def statuses(dann, bnm):
    out = []
    for cid in ("cond1", "cond2", "cond3b"):
        fires = [k for k, v in dann.items() if v[cid]["breaches_at_0_10"]]
        v0 = dann["1"][cid]
        out.append(declare(f"DANN {cid} ({v0['name']})", v0["rule"], v0["original_status"],
                           bool(fires), False, f"breached on folds {fires}"))
    for cid in ("cond3a", "cond4"):
        v0 = dann["1"][cid]
        out.append({"threshold": f"DANN {cid} ({v0['name']})", "value": "margin UNSET",
                    "original_status": "advisory (inactive)",
                    "declared_status": "advisory (inactive)",
                    "reason": v0["why"] + "; unchanged"})
    deg = [k for k, v in bnm.items() if v["collapse_floor"]["degenerate"]]
    floors = {k: v["collapse_floor"]["floor"] for k, v in bnm.items()}
    out.append(declare("BNM collapse floor", floors,
                       bnm["1"]["collapse_floor"]["original_status"], False, bool(deg),
                       f"floor = step-0 pred-pos fraction - 0.20 <= 0 on folds {deg}; it can "
                       f"never fire, so it is not loosened but declared advisory"))
    fires = [k for k, v in bnm.items() if v["ceiling"]["breaches_at_step0"]]
    out.append(declare("BNM ceiling 0.75", 0.75, "advisory", bool(fires), False,
                       f"breached at step 0 on folds {fires}"))
    return out


def lens_band(sdir):
    out = {}
    for k in FOLDS:
        L = scratch(sdir, "lens", k)
        out[str(k)] = {a: {kk: L[a][kk] for kk in ("L", "G", "expected", "realised_run_mean",
                                                   "abs_diff", "n_batches_in_budget",
                                                   "pre_phase_batches",
                                                   "all_batches_256_no_duplicates")}
                       for a in ("dann", "bnm", "coral_nb3", "freematch_nb3",
                                 "coral_frozen", "freematch_frozen")}
        for a, v in out[str(k)].items():
            v["inside_band"] = (v["abs_diff"] <= LENS_BAND
                                and v["all_batches_256_no_duplicates"])
    return out


def fold_identity(sdir):
    out = {"freematch_frozen": {}, "coral_frozen": {}}
    for k in FOLDS:
        f = scratch(sdir, "frozen_ids", k)
        for a in out:
            out[a][str(k)] = {x: f[a][x] for x in ("target_ids_sha256", "n_target",
                                                   "test_ids_sha256", "test_pos_ids_sha256",
                                                   "test_neg_ids_sha256", "n_test_pos",
                                                   "n_test_neg", "served_by")}
    return out


def workers_declared():
    k = 1
    return {
        "dann": jload(R("runs", "dann_main", f"fold{k}", "extracted", f"fold{k}",
                        "run_config.json"))["num_workers"],
        "bnm": jload(R("runs", "bnm_r6_folds", f"bnm_r6_fold{k}", "run_config.json"))
        ["num_workers"],
        "coral_nb3": jload(R("kaggle", "coral_r4", "runs", "main", "extracted",
                             "coral_r4_confirm", f"fold{k}", "run.json"))["num_workers"],
        "freematch_nb3": jload(R("runs", f"freematch_nb3_unfrozen_fold{k}",
                                 "run_config.json"))["num_workers"],
        "freematch_frozen": jload(R("runs", "bakeoff_freematch_5fold", "run_config.json"))
        ["num_workers"],
        "coral_frozen": ("not applicable: train_coral_fold builds no DataLoader (samplers "
                         "index cached feature tensors), so no worker RNG exists"),
        "source": "each original run's recorded num_workers (Kaggle, with workers)",
        "enforced": "every DataLoader-based dilute10 runner refuses --num-workers 0",
    }


def rng_finding(tests_dir):
    rows = {}
    for p in sorted(glob.glob(os.path.join(tests_dir, "heavy_*.json"))):
        h = jload(p).get("heavy", {})
        for key, v in h.get("RNG", {}).items():
            if "identical_2000_steps" in v:
                rows[key] = {"identical_2000_steps": v["identical_2000_steps"],
                             "digest": v["digest_dilute10"],
                             "report": os.path.basename(p)}
    want = {f"{a}/fold{k}" for a in ("dann", "bnm", "coral_nb3", "freematch_nb3",
                                     "coral_frozen", "freematch_frozen") for k in FOLDS}
    missing = sorted(want - set(rows))
    allok = not missing and all(v["identical_2000_steps"] for v in rows.values())
    if not allok:
        raise SystemExit(f"[part2] ABORT: RNG check incomplete or failing: missing {missing}")
    return {"finding": ("target sampling shares no random stream with source sampling, "
                        "augmentation or torch's global RNG; the source batch index sequence "
                        "was bit-identical for 2000 steps on all six arms and folds 1-4"),
            "lambda0_reuse": "UNQUALIFIED",
            "per_arm_fold": rows,
            "residual": ("FreeMatch x nb3's larger SAT rebuild (680 vs 160 target batches) "
                         "advances torch's in-process RNG only when num_workers=0; it is "
                         "removed by declaring the original worker counts and refusing 0")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--tests", required=True,
                    help="directory holding the heavy-test reports heavy_*.json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    if os.path.exists(out) and not a.force:
        raise SystemExit(f"[part2] refusing to overwrite {out}")
    import integrity
    integrity.verify(where="(write_dilute10_prereg_part2 start)")

    p1 = DP.verify_part1()
    dann = dann_thresholds(a.scratch)
    bnm = bnm_thresholds(a.scratch)
    audit = jload(R("audit", "script_provenance", "report.json"))
    restored = R("scripts", "run_coral_r4_confirm.py")
    halfedit = R("audit", "coral_confirm_halfedit", "run_coral_r4_confirm.py.halfedit")
    body = {
        "experiment": "dilute10",
        "status": "part 2 of 2 — final; no confirmatory run may start before this file is "
                  "enrolled",
        "part1": {"path": "runs/dilute10_prereg_part1.json",
                  "sha256": p1["sha256_of_body_without_this_field"],
                  "verified": True, "recorded_utc": p1["recorded_utc"]},
        "scope": ("step-0 values on the DILUTED pool with the UNIFORM sampler, folds 1-4, "
                  "through the new runners; no test label read; at most 10 training steps"),
        "dann_step0": dann,
        "bnm_step0": bnm,
        "threshold_statuses": {
            "rule": ("a threshold that would already fire at steps 0-10, or is degenerate "
                     "(e.g. a BNM floor at or below 0), becomes ADVISORY with its reason; it "
                     "is never loosened until it can no longer fire. Every other threshold "
                     "keeps its original status. 'Fires' is judged on any single step, "
                     "ignoring the sustain window (the stricter reading)."),
            "declared": statuses(dann, bnm)},
        "lens_fraction_band": {
            "stop_condition": ("a run STOPS if its realised run-mean lens fraction differs "
                               f"from its exact expected value by more than {LENS_BAND}, or if "
                               "any target batch is not exactly 256 items with no duplicates"),
            "tolerance": LENS_BAND,
            "per_fold": lens_band(a.scratch),
            "correction_recorded_here_not_in_part1": (
                "part 1's 'about 0.091' holds for CORAL and FreeMatch (391/4301, 392/4312); "
                "DANN and BNM, whose holdout is removed, expect 0.0834 (352/4223, 353/4234)"),
        },
        "stop_rules_nb3_coral_freematch": {
            "only_stop_conditions": [
                ("the step-0 gate against the NEW frozen head's gate_values.json at one "
                 "positive plus |AUROC diff| <= 1e-3, identical on folds 1-4 (supersedes "
                 "FreeMatch fold 1's original exact tolerance and CORAL's original warn-only "
                 "band)"),
                "a missed budget",
                "missing per-checkpoint logits"],
            "everything_else": "advisory"},
        "stop_rules_frozen_stages": {
            "conditions": ["fold identity: the runner aborts unless the served target-ID and "
                           "test-ID hashes equal fold_identity below",
                           "a missed budget", "missing per-checkpoint logits"]},
        "fold_identity": fold_identity(a.scratch),
        "num_workers_declared": workers_declared(),
        "rng_coupling": rng_finding(a.tests),
        "measurement_lockout": (
            "the new frozen and FreeMatch x nb3 runners persist logits without computing test "
            "metrics; only the step-0 gates (and each frozen stage's end-of-run "
            "gate_values.json) measure. Stricter than the original runners; it cannot affect "
            "training, because the replaced curve hook uses no RNG and its output was only "
            "printed"),
        "restoration_section_A": {
            "file": "scripts/run_coral_r4_confirm.py",
            "restored_sha256": DP.sha256_file(restored),
            "source": "kaggle/coral_r4/dataset/scripts/run_coral_r4_confirm.py",
            "source_sha256": DP.sha256_file(R("kaggle", "coral_r4", "dataset", "scripts",
                                              "run_coral_r4_confirm.py")),
            "recorded_by_confirmatory_folds": "2853805195bd88bf",
            "half_edit_kept_at": "audit/coral_confirm_halfedit/run_coral_r4_confirm.py.halfedit",
            "half_edit_sha256": DP.sha256_file(halfedit),
            "T8_expected_updated": True},
        "script_provenance_audit_section_B": {
            "report": "audit/script_provenance/report.json",
            "report_sha256": DP.sha256_file(R("audit", "script_provenance", "report.json")),
            **{k: audit[k] for k in ("n_records", "n_rows", "n_match", "n_differs",
                                     "n_unexplained")},
            "unexplained_executed_versions": audit["unexplained_executed_versions"]},
        "inputs_sha256": {os.path.basename(p): DP.sha256_file(p) for p in sorted(
            glob.glob(os.path.join(a.scratch, "*_fold*.json")))},
        "writer_sha256": DP.sha256_file(os.path.abspath(__file__)),
        "recorded_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    body = json.loads(json.dumps(body, default=str))
    h = hashlib.sha256(_canonical(body).encode()).hexdigest()
    body["sha256_of_body_without_this_field"] = h
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2, sort_keys=True)
    chk = jload(out)
    got = chk.pop("sha256_of_body_without_this_field")
    if got != h or hashlib.sha256(_canonical(chk).encode()).hexdigest() != h:
        raise SystemExit("[part2] DIGEST MISMATCH on re-read")
    integrity.verify(where="(write_dilute10_prereg_part2 end)")
    print(f"[part2] {out}\n  sha256       {h}\n  recorded_utc {body['recorded_utc']}")
    for t in body["threshold_statuses"]["declared"]:
        print(f"  {t['threshold']:<48} {t['original_status']:<22} -> {t['declared_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
