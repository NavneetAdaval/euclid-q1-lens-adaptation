"""
analyze_dann_lambda0_gap.py
===========================
The 704/78 memorisation diagnostic on the paper's one fully controlled pair: DANN x nb3
against its lambda=0 control, folds 1-4, all 16 checkpoints per fold (PROJECTv7 7.54).
**RETROSPECTIVE AND UNREGISTERED.** Decided 2026-10-24 after every result was known; the rule
is declared and hashed in ``runs/dann_lambda0_holdout_gap/plan.json`` before any value exists.

    python -u scripts/analyze_dann_lambda0_gap.py --stage known_null
    python -u scripts/analyze_dann_lambda0_gap.py --stage full

WHY IT IS POSSIBLE NOW. 7.29 said DANN x nb3's confirmatory folds retained no checkpoints and
lambda=0 one per fold. Both were false (correction note, 2026-10-22): each fold of both arms
holds step_250 .. step_4000 inside unextracted Kaggle tars. The two arms share a byte-identical
78-image holdout (test_round7.T7) and differ in lambda alone.

THE CODE PATH IS 7.29's, UNCHANGED. ``probe_stats``, ``holdout_gap`` and
``build_trainedon_probe`` from ``bnm_round6``; ``build_target_holdout`` and
``materialise_holdout_images`` from ``dann_round5``; the model rebuilt from each checkpoint's
own cfg through TrainConfig/Trainer exactly as ``analyze_holdout_gap_lambda0.build_trainer``
does; paired t from ``analyze_lambda0_r6.paired_t``. Nothing is reimplemented
(test_round7.T1-T4).

TWO STAGES, IN ORDER.

  known_null  Replays 7.29's driver's exact sequence (weak-augmented views, lambda=0, folds
              1..4 in order, the Trainer of fold f-1 seeding the views of fold f) with the
              weights of the tar's ``step_4000.pt``. Checks step_4000.pt against the extracted
              ``resume/latest.pt`` for tensor equality on all four folds, and the nuclear gap
              against 7.29's at four decimals on folds 2-4. Fold 1 is computed and reported but
              not checked: 7.29 drew its fold-1 views before anything seeded torch, whose
              default seed is time-based on this install, so they cannot be recreated.
  full        Only after a passing known_null. Both arms x folds 1-4 x 16 checkpoints on
              deterministic ``aug="none"`` views, identical for both arms; the readings and
              contrasts the plan fixes.

Reads target images only (the recorded holdout and the trained-on probe). Never loads test
logits, test labels or a scorer, and never calls the step-0 gate. Inference only.

Tars are extracted one arm x fold at a time, ``step_*.pt`` members only, into a scratch folder
OUTSIDE the repo, and the folder is deleted before the next. Writes nothing in the repo outside
``runs/dann_lambda0_holdout_gap/``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import statistics as st
import sys
import tarfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

from analyze_lambda0_r6 import paired_t                                    # noqa: E402
from bnm_round6 import (build_trainedon_probe, holdout_gap, probe_stats)   # noqa: E402
from dann_round5 import (build_target_holdout, materialise_holdout_images)  # noqa: E402
from integrity import verify as _verify_integrity                          # noqa: E402
from lens_dataset import LensDataset                                       # noqa: E402
from train_skeleton import TrainConfig, Trainer                            # noqa: E402

FOLDS = (1, 2, 3, 4)
SEED = 42
D_HOLDOUT_FRAC = 0.10
STEPS = tuple(range(250, 4001, 250))      # 16 per fold
READ_OFF = 4000
ARMS = ("dann", "lambda0")
ARM_LABEL = {"dann": "DANN x nb3", "lambda0": "lambda=0"}
ARM_TAG = {"dann": "dann_nb3", "lambda0": "dann_nb3_lambda0"}

OUT_DIR = os.path.join(REPO, "runs", "dann_lambda0_holdout_gap")
PLAN = os.path.join(OUT_DIR, "plan.json")
KNOWN_NULL = os.path.join(OUT_DIR, "known_null.json")
RESULTS = os.path.join(OUT_DIR, "results.json")
REPORT = os.path.join(OUT_DIR, "REPORT.md")
SCRATCH = os.path.join(os.path.dirname(REPO), "scratch_dann_l0_gap")
MIN_FREE_GB = 4.0

SPLITS = os.path.join(REPO, "data", "splits", "splits.json")
DATA_ROOT = os.path.join(REPO, "data")

#: 7.29's lambda=0 nuclear gaps at step 4000, folds 1-4, as printed in the record.
S729_PRINTED = {1: "+0.2128", 2: "-0.1362", 3: "+0.0624", 4: "-0.0569"}
KNOWN_NULL_CHECKED_FOLDS = (2, 3, 4)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def run_dir(arm, fold):
    if arm == "dann":
        return os.path.join(REPO, "runs", "dann_main", f"fold{fold}", "extracted",
                            f"fold{fold}")
    return os.path.join(REPO, "runs", "baseline_unfrozen", f"fold{fold}", "extracted",
                        f"lambda0_r6_fold{fold}")


def bnm_run_config(fold):
    return os.path.join(REPO, "runs", "bnm_r6_folds", f"bnm_r6_fold{fold}", "run_config.json")


def tars(arm, fold):
    if arm == "dann" and fold == 1:
        names = [f"ckpt_dann_r5_nb3_fold1_part0{p}of03.tar" for p in (1, 2, 3)]
        base = os.path.join(REPO, "runs", "dann_main", "fold1")
    elif arm == "dann":
        names = [f"ckpt_fold{fold}_0{p}.tar" for p in range(6)]
        base = os.path.join(REPO, "runs", "dann_main", f"fold{fold}")
    else:
        names = [f"ckpt_lambda0_r6_fold{fold}_part0{p}of03.tar" for p in (1, 2, 3)]
        base = os.path.join(REPO, "runs", "baseline_unfrozen", f"fold{fold}")
    return [os.path.join(base, n) for n in names]


def rel(p):
    return os.path.relpath(p, REPO).replace(os.sep, "/")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def self_cert(d):
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# the plan gate
# ---------------------------------------------------------------------------

def load_plan():
    """Refuse unless the plan exists, self-certifies, and names THIS file's sha256."""
    if not os.path.isfile(PLAN):
        raise SystemExit(f"REFUSING: {rel(PLAN)} does not exist. The rule is declared first.")
    plan = json.load(open(PLAN, encoding="utf-8"))
    if self_cert(plan) != plan.get("sha256_of_body_without_this_field"):
        raise SystemExit("REFUSING: plan.json does not self-certify.")
    me = sha256_file(os.path.abspath(__file__))
    want = plan["inputs"]["code"][rel(os.path.abspath(__file__))]
    if me != want:
        raise SystemExit(f"REFUSING: this driver's sha256 {me} != the plan's {want}. An edit "
                         f"after the plan is recorded as driver_edits_after_plan, not run "
                         f"silently.")
    # every small input still hashes as declared (tars are checked at extraction time)
    for group in ("code", "run_configs", "holdout_and_probe_records", "data", "reference"):
        for p, h in plan["inputs"][group].items():
            if p == rel(os.path.abspath(__file__)):
                continue
            got = sha256_file(os.path.join(REPO, p))
            if got != h:
                raise SystemExit(f"REFUSING: {p} changed since the plan ({h} -> {got}).")
    return plan


# ---------------------------------------------------------------------------
# scratch extraction -- one arm x fold at a time, step_*.pt only, outside the repo
# ---------------------------------------------------------------------------

_MEMBER = re.compile(r"(?:^|/)checkpoints/step_(\d+)\.pt$")


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def extract(arm, fold, plan, only_steps=None):
    if os.path.exists(SCRATCH):
        raise SystemExit(f"REFUSING: scratch {SCRATCH} exists; it is deleted after each use.")
    fg = free_gb(os.path.dirname(REPO))
    if fg < MIN_FREE_GB:
        raise SystemExit(f"STOP: {fg:.2f} GB free < {MIN_FREE_GB} GB before extracting.")
    os.makedirs(SCRATCH)
    got = {}
    for t in tars(arm, fold):
        want = plan["inputs"]["tars"][rel(t)]
        h = sha256_file(t)
        if h != want:
            shutil.rmtree(SCRATCH, ignore_errors=True)
            raise SystemExit(f"STOP: {rel(t)} sha256 {h} != plan {want}.")
        with tarfile.open(t, "r:") as tf:
            for m in tf.getmembers():
                mm = _MEMBER.search(m.name)
                if not (m.isfile() and mm):
                    continue
                s = int(mm.group(1))
                if only_steps is not None and s not in only_steps:
                    continue
                if s in got:
                    raise SystemExit(f"STOP: step {s} appears twice in {arm} fold {fold}.")
                dst = os.path.join(SCRATCH, f"step_{s}.pt")
                with tf.extractfile(m) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 22)
                got[s] = dst
    want_steps = set(STEPS if only_steps is None else only_steps)
    if set(got) != want_steps:
        raise SystemExit(f"STOP: {arm} fold {fold} tars hold steps {sorted(got)}, "
                         f"expected {sorted(want_steps)}.")
    return got


def drop_scratch():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    if os.path.exists(SCRATCH):
        raise SystemExit(f"STOP: could not delete {SCRATCH}.")


# ---------------------------------------------------------------------------
# slices and model, exactly as 7.29's drivers build them
# ---------------------------------------------------------------------------

def slices(fold, aug):
    tgt_ds = LensDataset(SPLITS, DATA_ROOT, role="target", fold=fold, aug=aug, n_views=1)
    holdout = build_target_holdout(tgt_ds, frac=D_HOLDOUT_FRAC, seed=SEED + fold)
    probe = build_trainedon_probe(tgt_ds, holdout, seed=SEED + fold)
    got_hold = set(int(i) for i in holdout["holdout"])
    for arm in ARMS:
        rec = set(int(i) for i in json.load(open(os.path.join(run_dir(arm, fold),
                                                              "run_config.json"),
                                                 encoding="utf-8"))["dann_d_holdout_indices"])
        if got_hold != rec:
            raise SystemExit(f"STOP: fold {fold}: reconstructed holdout != the slice "
                             f"{ARM_LABEL[arm]} ran with.")
    rec_probe = set(int(i) for i in json.load(open(bnm_run_config(fold), encoding="utf-8"))
                    ["trainedon_probe"]["probe"])
    got_probe = set(int(i) for i in probe["probe"])
    if got_probe != rec_probe:
        raise SystemExit(f"STOP: fold {fold}: reconstructed trained-on probe != BNM's "
                         f"recorded probe.")
    if len(got_hold) != 78 or len(got_probe) != 78 or (got_hold & got_probe):
        raise SystemExit(f"STOP: fold {fold}: slices not 78/78 disjoint.")
    return tgt_ds, holdout, probe


def check_ckpt(raw, arm, fold, step, rc):
    """The checkpoint's recorded step and config must match the run it came from."""
    sch = raw.get("schedule") or {}
    c = raw["cfg"]
    bad = []
    if int(sch.get("global_step", -1)) != step:
        bad.append(f"schedule.global_step {sch.get('global_step')} != {step}")
    if sch.get("fold") is not None and int(sch["fold"]) != fold:
        bad.append(f"schedule.fold {sch.get('fold')}")
    if sch.get("arm") is not None and sch["arm"] != rc["arm"]:
        bad.append(f"schedule.arm {sch.get('arm')} != run {rc['arm']}")
    if "dann_lambda_zero" in sch and bool(sch["dann_lambda_zero"]) != bool(rc["dann_lambda_zero"]):
        bad.append("schedule.dann_lambda_zero")
    if rc["arm"] != ARM_TAG[arm]:
        bad.append(f"run_config arm {rc['arm']} != {ARM_TAG[arm]}")
    for k in ("fold", "n_blocks", "lr", "weight_decay", "batch_size", "head_type",
              "lr_decay"):
        if k in rc and rc[k] != c.get(k):
            bad.append(f"cfg.{k} {c.get(k)} != run_config {rc[k]}")
    # run_config records the base seed; the trainer ran with base + fold
    if int(rc["seed"]) != SEED or int(c["seed"]) != SEED + fold:
        bad.append(f"seed: run_config {rc['seed']} (want {SEED}), cfg {c['seed']} "
                   f"(want {SEED + fold})")
    if bad:
        raise SystemExit(f"STOP: {ARM_LABEL[arm]} fold {fold} step {step}: " + "; ".join(bad))


def build_trainer(cfg_src, weights_path, fold, device):
    """analyze_holdout_gap_lambda0.build_trainer, with cfg and weights possibly from two
    files (the known-null takes the cfg from resume/latest.pt, as 7.29 did, and the weights
    from step_4000.pt)."""
    c = dict(cfg_src["cfg"])
    cfg = TrainConfig(
        splits_path=SPLITS, data_root=DATA_ROOT, fold=fold,
        head_type=c["head_type"], encoder_mode=c["encoder_mode"],
        n_blocks=int(c["n_blocks"]),
        init_head_from=None,
        lr=c["lr"], weight_decay=c["weight_decay"], batch_size=c["batch_size"],
        seed=int(c["seed"]), device=str(device), amp=False,
        grad_checkpointing=False, verbose_encoder=False)
    tr = Trainer(cfg)
    tr.load(weights_path, map_location=device, load_training_state=False)
    tr.head.eval()
    return tr


def gap_of(tr, probe_images, held_images):
    with torch.no_grad():
        return holdout_gap(probe_stats(tr, probe_images), probe_stats(tr, held_images))


def weights_equal(a, b):
    """Tensor equality of every model tensor the reader loads: head, standardizer, the
    trainable encoder blocks."""
    out = {}
    for k in ("head", "standardizer", "encoder_trainable"):
        da, db = a.get(k) or {}, b.get(k) or {}
        if set(da) != set(db):
            out[k] = {"equal": False, "reason": "key sets differ",
                      "only_a": sorted(set(da) - set(db))[:5],
                      "only_b": sorted(set(db) - set(da))[:5]}
            continue
        diff = [n for n in da if not torch.equal(da[n].cpu(), db[n].cpu())]
        maxabs = max((float((da[n].float().cpu() - db[n].float().cpu()).abs().max())
                      for n in diff), default=0.0)
        out[k] = {"equal": not diff, "n_tensors": len(da), "n_differ": len(diff),
                  "max_abs_diff": maxabs, "differ_first": diff[:5]}
    return all(v["equal"] for v in out.values()), out


def venue(device):
    return {"device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "precision": "fp32 (amp=False)",
            "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def fmt4(x):
    return f"{x:+.4f}".replace("-", "−")


# ---------------------------------------------------------------------------
# stage 1 -- the known-null
# ---------------------------------------------------------------------------

def stage_known_null(plan, device):
    if os.path.exists(KNOWN_NULL):
        raise SystemExit(f"REFUSING: {rel(KNOWN_NULL)} exists.")
    ref_json = json.load(open(os.path.join(REPO, "runs", "round7", "holdout_gap_lambda0.json"),
                              encoding="utf-8"))["per_fold"]
    print("KNOWN-NULL: replaying 7.29's sequence (weak-aug views, lambda=0, folds 1..4)")
    out = {"stage": "known_null", "plan_sha256": plan["sha256_of_body_without_this_field"],
           "driver_sha256": sha256_file(os.path.abspath(__file__)), "venue": venue(device),
           "per_fold": {}}
    ok = True
    for f in FOLDS:
        # 7.29's order: dataset, slices, both materialisations, THEN the trainer.
        tgt_ds, holdout, probe = slices(f, aug="weak")
        held_images = materialise_holdout_images(tgt_ds, holdout["holdout"])
        probe_images = materialise_holdout_images(tgt_ds, probe["probe"])

        ck = extract("lambda0", f, plan, only_steps={READ_OFF})
        latest_p = os.path.join(run_dir("lambda0", f), "resume", "latest.pt")
        if sha256_file(latest_p) != plan["inputs"]["resume_checkpoints"][rel(latest_p)]:
            raise SystemExit(f"STOP: {rel(latest_p)} changed since the plan.")
        latest = torch.load(latest_p, map_location="cpu", weights_only=False)
        s4000 = torch.load(ck[READ_OFF], map_location="cpu", weights_only=False)
        rc = json.load(open(os.path.join(run_dir("lambda0", f), "run_config.json"),
                            encoding="utf-8"))
        check_ckpt(s4000, "lambda0", f, READ_OFF, rc)
        eq, eq_detail = weights_equal(latest, s4000)
        cfg_equal = dict(latest["cfg"]) == dict(s4000["cfg"])

        tr = build_trainer(latest, ck[READ_OFF], f, device)
        g = gap_of(tr, probe_images, held_images)
        del tr, latest, s4000
        if device.type == "cuda":
            torch.cuda.empty_cache()
        drop_scratch()

        got = f"{g['nuclear_gap']:+.4f}"
        checked = f in KNOWN_NULL_CHECKED_FOLDS
        match = (got == S729_PRINTED[f]
                 and got == f"{ref_json[str(f)]['nuclear_gap']:+.4f}")
        row = {"weights_identical": eq, "weights_detail": eq_detail,
               "cfg_identical": cfg_equal,
               "nuclear_gap": g["nuclear_gap"], "nuclear_gap_4dp": got,
               "s729_printed": S729_PRINTED[f],
               "s729_json": ref_json[str(f)]["nuclear_gap"],
               "value_checked": checked, "match_4dp": match,
               "gap": g}
        if not checked:
            row["why_not_checked"] = plan["known_null"]["fold1_exclusion"]
        out["per_fold"][str(f)] = row
        print(f"  fold {f}: weights identical {eq}  cfg identical {cfg_equal}  gap {got}  "
              f"7.29 {S729_PRINTED[f]}  "
              + (f"MATCH {match}" if checked else "(not checked -- unseeded 7.29 views)"))
        if not eq:
            ok = False
            out["stopped"] = f"fold {f}: step_4000.pt and resume/latest.pt differ"
            break
        if checked and not match:
            ok = False
            out["stopped"] = f"fold {f}: weights identical but gap {got} != 7.29 {S729_PRINTED[f]}"
            break
    out["pass"] = ok and len(out["per_fold"]) == 4
    with open(KNOWN_NULL, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n  KNOWN-NULL {'PASS' if out['pass'] else 'FAIL -- STOP'}  -> {rel(KNOWN_NULL)}")
    return out["pass"]


# ---------------------------------------------------------------------------
# stage 2 -- the full run
# ---------------------------------------------------------------------------

def readings(per_step):
    vals = [per_step[s]["nuclear_gap"] for s in STEPS]
    ent = [per_step[s]["entropy_gap"] for s in STEPS]
    pp = [per_step[s]["pred_pos_frac_gap"] for s in STEPS]
    return {"at_4000": per_step[READ_OFF]["nuclear_gap"],
            "within_fold_mean": st.mean(vals),
            "any_checkpoint_max": max(vals),
            "any_checkpoint_max_step": STEPS[vals.index(max(vals))],
            "entropy_gap_at_4000": per_step[READ_OFF]["entropy_gap"],
            "entropy_gap_within_fold_mean": st.mean(ent),
            "pred_pos_gap_at_4000": per_step[READ_OFF]["pred_pos_frac_gap"],
            "pred_pos_gap_within_fold_mean": st.mean(pp)}


def interpretation(plan, prim, sec):
    sent = plan["interpretation_VERBATIM"]
    filled = (f"{fmt4(prim['mean'])} [{fmt4(prim['ci_lo'])}, {fmt4(prim['ci_hi'])}]")
    if prim["excludes_zero"] and prim["mean"] > 0:
        key = "excludes_zero_positive"
    elif prim["excludes_zero"]:
        key = "excludes_zero_negative"
    else:
        key = "includes_zero"
    text = sent[key].replace("[Δ] [CI]", filled)
    sec_text = (f"Secondary reading (the within-fold mean over the 16 checkpoints; it is the "
                f"secondary, not the primary): DANN − λ=0 = {fmt4(sec['mean'])} "
                f"[{fmt4(sec['ci_lo'])}, {fmt4(sec['ci_hi'])}].")
    return {"branch": key, "sentence": text, "secondary_statement": sec_text,
            "always": sent["in_every_case"]}


def stage_full(plan, device):
    if os.path.exists(RESULTS) or os.path.exists(REPORT):
        raise SystemExit("REFUSING: results.json / REPORT.md exist.")
    if not os.path.isfile(KNOWN_NULL):
        raise SystemExit("REFUSING: run --stage known_null first.")
    kn = json.load(open(KNOWN_NULL, encoding="utf-8"))
    if not kn.get("pass"):
        raise SystemExit("REFUSING: the known-null did not pass.")

    per = {arm: {} for arm in ARMS}
    for f in FOLDS:
        tgt_ds, holdout, probe = slices(f, aug="none")
        held_images = materialise_holdout_images(tgt_ds, holdout["holdout"])
        probe_images = materialise_holdout_images(tgt_ds, probe["probe"])
        for arm in ARMS:
            rc = json.load(open(os.path.join(run_dir(arm, f), "run_config.json"),
                                encoding="utf-8"))
            ck = extract(arm, f, plan)
            rows = {}
            for s in STEPS:
                raw = torch.load(ck[s], map_location="cpu", weights_only=False)
                check_ckpt(raw, arm, f, s, rc)
                tr = build_trainer(raw, ck[s], f, device)
                rows[s] = gap_of(tr, probe_images, held_images)
                del tr, raw
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            drop_scratch()
            per[arm][f] = rows
            r = readings(rows)
            print(f"  {ARM_LABEL[arm]:<11} fold {f}: at 4000 {r['at_4000']:+.4f}  "
                  f"mean {r['within_fold_mean']:+.4f}  max {r['any_checkpoint_max']:+.4f}"
                  f" (step {r['any_checkpoint_max_step']})   free {free_gb(REPO):.1f} GB")

    rd = {arm: {f: readings(per[arm][f]) for f in FOLDS} for arm in ARMS}
    prim = paired_t([rd["dann"][f]["at_4000"] - rd["lambda0"][f]["at_4000"] for f in FOLDS])
    sec = paired_t([rd["dann"][f]["within_fold_mean"] - rd["lambda0"][f]["within_fold_mean"]
                    for f in FOLDS])
    own = {arm: {"at_4000_vs_zero": paired_t([rd[arm][f]["at_4000"] for f in FOLDS]),
                 "within_fold_mean_vs_zero": paired_t([rd[arm][f]["within_fold_mean"]
                                                       for f in FOLDS]),
                 "label": "descriptive"} for arm in ARMS}
    maxima = {arm: {str(f): {"value": rd[arm][f]["any_checkpoint_max"],
                             "step": rd[arm][f]["any_checkpoint_max_step"]} for f in FOLDS}
              for arm in ARMS}
    descr = {arm: {k: st.mean(rd[arm][f][k] for f in FOLDS)
                   for k in ("entropy_gap_at_4000", "entropy_gap_within_fold_mean",
                             "pred_pos_gap_at_4000", "pred_pos_gap_within_fold_mean")}
             for arm in ARMS}
    interp = interpretation(plan, prim, sec)
    s729_vs_main = {str(f): {"s729_weak_view": S729_PRINTED[f],
                             "known_null_weak_view_replay": kn["per_fold"][str(f)]["nuclear_gap_4dp"],
                             "main_no_aug_view": f"{rd['lambda0'][f]['at_4000']:+.4f}"}
                    for f in FOLDS}

    res = {"status": plan["status"],
           "plan_sha256": plan["sha256_of_body_without_this_field"],
           "driver_sha256": sha256_file(os.path.abspath(__file__)),
           "driver_edits_after_plan": [],
           "known_null_file_sha256": sha256_file(KNOWN_NULL),
           "known_null": {"pass": kn["pass"], "venue": kn["venue"],
                          "per_fold": {f: {k: v for k, v in r.items() if k != "gap"}
                                       for f, r in kn["per_fold"].items()}},
           "venue": venue(device),
           "views": "aug='none' (deterministic), identical for both arms",
           "per_checkpoint": {arm: {str(f): {str(s): per[arm][f][s] for s in STEPS}
                                    for f in FOLDS} for arm in ARMS},
           "readings_per_fold": {arm: {str(f): rd[arm][f] for f in FOLDS} for arm in ARMS},
           "contrast_dann_minus_lambda0": {"primary_at_4000": prim,
                                           "secondary_within_fold_mean": sec},
           "own_arm_vs_zero_descriptive": own,
           "any_checkpoint_max_descriptive": {
               "values": maxima,
               "no_interval_no_contrast": plan["readings"]["descriptive_only"]["reason"]},
           "entropy_and_pred_pos_fold_means_descriptive": descr,
           "lambda0_step4000_s729_vs_main": {
               "per_fold": s729_vs_main,
               "disclosure": plan["disclosures"]["lambda0_views_differ"]},
           "interpretation": interp,
           "multiplicity_and_overlap": plan["interpretation_VERBATIM"]["in_every_case"]}
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    write_report(res, rd, plan)
    print(f"\n  {interp['sentence']}\n  {interp['secondary_statement']}")
    print(f"[out] {rel(RESULTS)}  {rel(REPORT)}")


def write_report(res, rd, plan):
    L = []
    a = L.append
    a("# DANN − λ=0: the 704/78 memorisation diagnostic, folds 1–4 (PROJECTv7 §7.54)\n")
    a("> **RETROSPECTIVE AND UNREGISTERED.** " + plan["status"] + ". Rule declared in "
      f"`plan.json` (self-certifying `{res['plan_sha256']}`) before any value was computed.\n")
    a(f"Driver sha256 `{res['driver_sha256']}`. Venue: {res['venue']['gpu_name']}, torch "
      f"{res['venue']['torch_version']} (CUDA {res['venue']['cuda_version']}), "
      f"{res['venue']['precision']}. Views: {res['views']}.\n")

    a("## Known-null (λ=0, step 4000, §7.29's weak-aug views replayed)\n")
    a("| fold | step_4000.pt == resume/latest.pt | gap (replay) | §7.29 | checked | match |")
    a("|---|:--:|---:|---:|:--:|:--:|")
    for f, r in res["known_null"]["per_fold"].items():
        a(f"| {f} | {'yes' if r['weights_identical'] else 'NO'} | "
          f"{fmt4(r['nuclear_gap'])} | {r['s729_printed'].replace('-', '−')} | "
          f"{'yes' if r['value_checked'] else 'no'} | "
          f"{('yes' if r['match_4dp'] else 'NO') if r['value_checked'] else '—'} |")
    a(f"\nResult: **{'PASS' if res['known_null']['pass'] else 'FAIL'}**. Fold 1 is not "
      f"checked: {plan['known_null']['fold1_exclusion']}\n")

    def table(key, title):
        a(f"## {title}\n")
        a("| arm | fold 1 | fold 2 | fold 3 | fold 4 | fold-mean |")
        a("|---|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            v = [rd[arm][f][key] for f in FOLDS]
            a(f"| {ARM_LABEL[arm]} | " + " | ".join(fmt4(x) for x in v) +
              f" | {fmt4(st.mean(v))} |")
        d = [rd['dann'][f][key] - rd['lambda0'][f][key] for f in FOLDS]
        a("| DANN − λ=0 | " + " | ".join(fmt4(x) for x in d) + f" | {fmt4(st.mean(d))} |\n")

    table("at_4000", "Primary: normalised ‖A‖* gap at step 4000")
    table("within_fold_mean", "Secondary: within-fold mean of the gap over 16 checkpoints")

    a("## Contrasts DANN − λ=0 (paired t, df = 3, 95%)\n")
    a("| reading | mean | sd | 95% CI | excludes 0 | folds same sign |")
    a("|---|---:|---:|---|:--:|:--:|")
    for lab, k in (("**primary** (step 4000)", "primary_at_4000"),
                   ("secondary (within-fold mean)", "secondary_within_fold_mean")):
        c = res["contrast_dann_minus_lambda0"][k]
        a(f"| {lab} | {fmt4(c['mean'])} | {c['sd']:.4f} | [{fmt4(c['ci_lo'])}, "
          f"{fmt4(c['ci_hi'])}] | {'yes' if c['excludes_zero'] else 'no'} | "
          f"{'yes' if c['all_same_sign'] else 'no'} |")

    a("\n## Each arm's own gap against zero (descriptive)\n")
    a("| arm | reading | mean | 95% CI |")
    a("|---|---|---:|---|")
    for arm in ARMS:
        for lab, k in (("step 4000", "at_4000_vs_zero"),
                       ("within-fold mean", "within_fold_mean_vs_zero")):
            c = res["own_arm_vs_zero_descriptive"][arm][k]
            a(f"| {ARM_LABEL[arm]} | {lab} | {fmt4(c['mean'])} | "
              f"[{fmt4(c['ci_lo'])}, {fmt4(c['ci_hi'])}] |")

    a("\n## Any-checkpoint maximum (descriptive only: no interval, no contrast)\n")
    a(f"Reason: {plan['readings']['descriptive_only']['reason']}\n")
    a("| arm | fold 1 | fold 2 | fold 3 | fold 4 |")
    a("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        m = res["any_checkpoint_max_descriptive"]["values"][arm]
        a(f"| {ARM_LABEL[arm]} | " + " | ".join(
            f"{fmt4(m[str(f)]['value'])} (step {m[str(f)]['step']})" for f in FOLDS) + " |")

    a("\n## Entropy and predicted-positive gaps (descriptive, fold-means)\n")
    a("| arm | entropy @4000 | entropy, within-fold mean | pred-pos @4000 | "
      "pred-pos, within-fold mean |")
    a("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        d = res["entropy_and_pred_pos_fold_means_descriptive"][arm]
        a(f"| {ARM_LABEL[arm]} | {fmt4(d['entropy_gap_at_4000'])} | "
          f"{fmt4(d['entropy_gap_within_fold_mean'])} | {fmt4(d['pred_pos_gap_at_4000'])} | "
          f"{fmt4(d['pred_pos_gap_within_fold_mean'])} |")

    a("\n## λ=0 at step 4000: §7.29's views against this analysis's views\n")
    a(res["lambda0_step4000_s729_vs_main"]["disclosure"] + "\n")
    a("| fold | §7.29 (one weak-aug view) | replay (same views) | this analysis (no aug) |")
    a("|---|---:|---:|---:|")
    for f, r in res["lambda0_step4000_s729_vs_main"]["per_fold"].items():
        a(f"| {f} | {r['s729_weak_view'].replace('-', '−')} | "
          f"{r['known_null_weak_view_replay'].replace('-', '−')} | "
          f"{r['main_no_aug_view'].replace('-', '−')} |")

    a("\n## Interpretation (sentence fixed in the plan)\n")
    a(f"> {res['interpretation']['sentence']}\n")
    if res["interpretation"]["branch"] == "includes_zero":
        a(f"> {res['interpretation']['secondary_statement']}\n")
    else:
        a(f"{res['interpretation']['secondary_statement']}\n")
    a(f"{res['interpretation']['always']}\n")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("known_null", "full"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    plan = load_plan()
    _verify_integrity(where=f"before the DANN - lambda=0 gap analysis ({a.stage})")
    device = torch.device(a.device)
    print("=" * 100)
    print("DANN - lambda=0 704/78 DIAGNOSTIC -- RETROSPECTIVE AND UNREGISTERED")
    print(f"  plan {plan['sha256_of_body_without_this_field']}   stage {a.stage}   "
          f"{venue(device)}")
    print("=" * 100)
    try:
        if a.stage == "known_null":
            passed = stage_known_null(plan, device)
        else:
            stage_full(plan, device)
            passed = True
    finally:
        if os.path.exists(SCRATCH):
            shutil.rmtree(SCRATCH, ignore_errors=True)
    _verify_integrity(where=f"after the DANN - lambda=0 gap analysis ({a.stage})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
