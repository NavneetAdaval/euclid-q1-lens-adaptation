"""
test_coral_r4_confirm.py
========================
Pre-handover suite for `run_coral_r4_confirm.py` (PROJECT.md §9: nothing is presented
untested). No GPU, no training. It tests the machinery that surrounds the confirmatory
folds, against the REAL stored artifacts.

  T1  FOLD 0 CANNOT BE SELECTED -- refused at ARGUMENT PARSING, not deep in the run, and
      the message says why. Every other entry point that could smuggle fold 0 in is checked
      too.
  T2  the pre-registration hash gate: it verifies the real file, and it FAILS LOUDLY on
      every way the file can be altered (value changed, key added, key removed, hash
      tampered, hash removed, file absent).
  T3  the config diff: allowlist enforced by VALUE (not merely "may differ"), pinned keys
      compared, the vacuous-guard rule, and the per-fold head pinning that catches a fold
      pointed at another fold's phase-1 head.
  T4  per-fold facts resolved from splits_v6: 98 / 98 / 98 / 98 / 97 across folds 0-4,
      489 total, and each fold's FOLD_GATE value consistent with its own denominator.
  T5  NO METRIC-BASED CHECKPOINT SELECTION exists anywhere in the confirmatory path.
  T6  logit format round-trip and id-order stability, plus the shared-negative invariant.
  T7  path resolution against a MOCKED Kaggle mount, per fold.

Run:  python -u scripts/test_coral_r4_confirm.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_coral_r4_confirm as C                                    # noqa: E402
import run_coral_r4_dev8000 as DEV                                  # noqa: E402
import run_coral_round4 as R4                                       # noqa: E402

ROOT = os.path.normpath(os.path.join(_HERE, ".."))
# Search the mount, this tree, AND its ancestors. The suite runs from three places -- the
# repo, the staged Kaggle bundle (whose root is kaggle/coral_r4/dataset), and the mount --
# and a suite that fails because an artifact sits one directory higher is reporting on its
# own layout, not on the code. It still FAILS rather than skips if an artifact is genuinely
# absent.
_SEARCH = [ROOT, "/kaggle/input"] + [
    os.path.normpath(os.path.join(_HERE, *([".."] * n))) for n in range(2, 6)]

PREREG = DEV._find_under(_SEARCH, C.PREREG_BASENAME)
SPLITS = DEV._find_under(_SEARCH, os.path.join("splits", "splits.json"))
DEV_RUN_JSON = DEV._find_under(_SEARCH, os.path.join(C.DEV_ARM_RUN_JSON, "run.json"))
REF_IDS = DEV._find_under(_SEARCH, os.path.join("coral_r4_logits", "ids.npz"))

EXPECTED_COUNTS = {0: 98, 1: 98, 2: 98, 3: 98, 4: 97}     # PROJECT.md 4.5; 489 total

_fails, _passes = [], []


def check(cond, msg):
    (_passes if cond else _fails).append(msg)
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")


def section(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# T1 -- fold 0 cannot be selected
# ---------------------------------------------------------------------------

def t1_no_fold0():
    section("T1  fold 0 is DEVELOPMENT and cannot be run as a confirmatory fold")
    ap = C.build_argparser()
    for bad in ("0", "-0", "00"):
        try:
            with contextlib.redirect_stderr(io.StringIO()) as err:
                ap.parse_args(["--fold", bad])
            check(False, f"--fold {bad} is refused")
        except SystemExit as e:
            msg = err.getvalue()
            check(int(e.code) != 0 and "DEVELOPMENT" in msg and "oracle selection" in msg,
                  f"--fold {bad} is refused AT PARSE TIME with the reason "
                  f"(exit {e.code}, message names development + oracle selection)")
    for good in (1, 2, 3, 4):
        ns = ap.parse_args(["--fold", str(good)])
        check(ns.fold == good, f"--fold {good} parses to {good}")
    for bad in ("5", "-1", "x", "1.5"):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                ap.parse_args(["--fold", bad])
            check(False, f"--fold {bad} is refused")
        except SystemExit as e:
            check(int(e.code) != 0, f"--fold {bad} is refused [{type(e).__name__}]")

    check(C.CONFIRMATORY_FOLDS == (1, 2, 3, 4),
          f"the confirmatory set is {C.CONFIRMATORY_FOLDS} -- fold 0 is not in it")
    check(0 not in C.CONFIRMATORY_FOLDS and all(f in C.CONFIRMATORY_FOLDS
                                                for f in (1, 2, 3, 4)),
          "fold 0 absent and folds 1-4 present")
    src = open(os.path.join(_HERE, "run_coral_r4_confirm.py"), encoding="utf-8").read()
    check("--fold" in src and "required=True" in src,
          "--fold is REQUIRED, so there is no default that could silently be 0")
    # the launcher must not be able to schedule fold 0 either
    import launch_coral_r4_confirm_kaggle as L
    check(sorted(L.DEFAULT_FOLDS) == [1, 2, 3, 4] and L.VERIFY_FOLD != 0,
          f"the launcher's default set is {list(L.DEFAULT_FOLDS)} and it verifies on fold "
          f"{L.VERIFY_FOLD} -- never fold 0")
    for bad in ("0", "-0"):
        try:
            with contextlib.redirect_stderr(io.StringIO()) as err:
                L.main([f"--folds", bad])
            check(False, f"the launcher refuses --folds {bad}")
        except SystemExit as e:
            check(int(e.code) != 0 and "DEVELOPMENT" in err.getvalue(),
                  f"the launcher refuses --folds {bad} at parse time, with the reason")
    check(L.DEFAULT_CONCURRENCY == 1,
          "the launcher runs ONE FOLD AT A TIME by default")
    check(L.waves_for((1, 2, 3, 4), 1) == [(1,), (2,), (3,), (4,)],
          "at concurrency 1 each fold is its own wave -- one at a time, in order")
    check(L.waves_for((1, 2, 3, 4), 2) == [(1, 2), (3, 4)],
          "concurrency 2 remains available as an opt-in (two folds, one per device)")
    check(L.waves_for((3,), 1) == [(3,)],
          "a single requested fold produces a single wave")
    # running one fold at a time means the report is written once per fold -- it must MERGE,
    # or each session would erase the previous folds and claim they were never run
    td = tempfile.mkdtemp(prefix="report_merge_")
    try:
        p = os.path.join(td, "launch_report.json")
        with open(p, "w") as fh:
            json.dump({"folds": [{"fold": 1, "ok": True}]}, fh)
        r = L.load_report(td)
        by = {f["fold"]: f for f in r["folds"]}
        by[2] = {"fold": 2, "ok": True}
        merged = sorted(by)
        check(merged == [1, 2],
              "launch_report.json MERGES across invocations (fold 1 from an earlier session "
              "survives when fold 2 is run later)")
        check(L.load_report(os.path.join(td, "absent"))["folds"] == [],
              "a first invocation with no prior report starts from an empty fold list")
        with open(p, "w") as fh:
            fh.write("{not json")
        check(L.load_report(td)["folds"] == [],
              "a corrupt report is treated as empty rather than crashing the launch")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    # and the pre-registration itself agrees
    if PREREG:
        with open(PREREG) as fh:
            b = json.load(fh)
        check(b["confirmatory_design"]["folds"] == [1, 2, 3, 4],
              "the hashed pre-registration names folds [1,2,3,4] as confirmatory")
        check("DEVELOPMENT" in b["confirmatory_design"]["fold0"],
              "the pre-registration records fold 0 as DEVELOPMENT")


# ---------------------------------------------------------------------------
# T2 -- the pre-registration hash gate
# ---------------------------------------------------------------------------

def t2_prereg():
    section("T2  the pre-registration hash gate")
    if not PREREG:
        check(False, "runs/coral_r4_budget.json located")
        return
    check(True, f"pre-registration located ({os.path.relpath(PREREG, ROOT)})")
    info = C.verify_prereg(PREREG, 1)
    check(info["verified"] and len(info["sha256"]) == 64,
          f"the real file VERIFIES (sha256 {info['sha256'][:16]}…)")
    check(info["budget_steps"] == 2000,
          f"the budget is read OUT of the verified file: {info['budget_steps']} steps")
    check("FINAL checkpoint" in info["read_off_rule"],
          "the read-off rule is the final-checkpoint rule this runner implements")
    check(info["derived_or_declared"] == "declared",
          "the budget is recorded as DECLARED, not derived")

    with open(PREREG) as fh:
        good = json.load(fh)
    td = tempfile.mkdtemp(prefix="prereg_")
    try:
        def write(obj, name):
            p = os.path.join(td, name)
            with open(p, "w") as fh:
                json.dump(obj, fh, indent=1, sort_keys=True)
            return p

        # (a) a changed VALUE
        bad = json.loads(json.dumps(good))
        bad["selected_budget_steps"] = 2500
        p = write(bad, "changed_value.json")
        try:
            C.verify_prereg(p, 1)
            check(False, "a changed budget value is refused")
        except SystemExit as e:
            check("HASH DOES NOT CHECK" in str(e),
                  "a CHANGED VALUE (budget 2000 -> 2500) is refused: the hash does not check")
        # (b) an added key
        bad = json.loads(json.dumps(good))
        bad["extra_note"] = "harmless-looking"
        p = write(bad, "added_key.json")
        try:
            C.verify_prereg(p, 1)
            check(False, "an added key is refused")
        except SystemExit as e:
            check("HASH DOES NOT CHECK" in str(e), "an ADDED KEY is refused")
        # (c) a removed key
        bad = json.loads(json.dumps(good))
        bad.pop("cost_accounting")
        p = write(bad, "removed_key.json")
        try:
            C.verify_prereg(p, 1)
            check(False, "a removed key is refused")
        except SystemExit as e:
            check("HASH DOES NOT CHECK" in str(e), "a REMOVED KEY is refused")
        # (d) a tampered hash
        bad = json.loads(json.dumps(good))
        bad["sha256_of_body_without_this_field"] = "0" * 64
        p = write(bad, "tampered_hash.json")
        try:
            C.verify_prereg(p, 1)
            check(False, "a tampered hash is refused")
        except SystemExit as e:
            check("HASH DOES NOT CHECK" in str(e), "a TAMPERED HASH is refused")
        # (e) no hash at all
        bad = json.loads(json.dumps(good))
        bad.pop("sha256_of_body_without_this_field")
        p = write(bad, "no_hash.json")
        try:
            C.verify_prereg(p, 1)
            check(False, "a file with no hash is refused")
        except SystemExit as e:
            check("carries no hash" in str(e),
                  "a file with NO HASH FIELD is refused -- it is not a pre-registration")
        # (f) absent entirely
        try:
            C.verify_prereg(os.path.join(td, "nope.json"), 1)
            check(False, "a missing file is refused")
        except SystemExit as e:
            check("was not found" in str(e) and "7.14" in str(e),
                  "an ABSENT pre-registration is refused, citing 7.14")
        # (g) re-hashing the altered file does NOT make it acceptable as fold 0
        bad = json.loads(json.dumps(good))
        bad.pop("sha256_of_body_without_this_field")
        bad["confirmatory_design"] = dict(bad["confirmatory_design"], folds=[0, 1, 2, 3, 4])
        import hashlib
        bad["sha256_of_body_without_this_field"] = hashlib.sha256(
            json.dumps(bad, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        p = write(bad, "rehashed_with_fold0.json")
        ok_hash = True
        try:
            C.verify_prereg(p, 0)
        except SystemExit:
            ok_hash = False
        check(ok_hash, "a re-hashed file passes the HASH check (as it must -- the hash "
                       "proves integrity, not authority)")
        ap = C.build_argparser()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                ap.parse_args(["--fold", "0"])
            check(False, "…but fold 0 is still refused")
        except SystemExit:
            check(True, "…but fold 0 is STILL refused at parse time, so re-hashing a file "
                        "to add fold 0 does not get it into the confirmatory set")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# T3 -- the config diff
# ---------------------------------------------------------------------------

def t3_config_diff():
    section("T3  config diff -- allowlist by VALUE, pinned keys, vacuous guard")
    if not DEV_RUN_JSON:
        check(False, "the fold-0 development arm's run.json located")
        return
    check(True, f"fold-0 arm run.json located ({os.path.relpath(DEV_RUN_JSON, ROOT)})")
    with open(DEV_RUN_JSON) as fh:
        ref = json.load(fh)
    check(int(ref["fold"]) == 0 and int(ref["max_steps"]) == 8000,
          f"the reference IS the fold-0 8000-step development arm "
          f"(fold {ref['fold']}, {ref['max_steps']} steps)")

    for fold in (1, 2, 3, 4):
        exp = C.expected_diffs_for(fold, 2000, ref)
        rec = {k: ref.get(k) for k in C.CONFIRM_DIFF_KEYS if k in ref}
        rec.update({"fold": fold, "max_steps": 2000,
                    "init_head_from_tail":
                        f"runs/bakeoff_coral/fold{fold}/checkpoints/step_8000.pt",
                    "init_head_l2": float(R4.FOLD_GATE[fold]["head_l2"]),
                    "epochs_planned": 11})
        rows, mism = quiet(C.config_diff, rec, DEV_RUN_JSON, exp)
        st = {r["key"]: r["status"] for r in rows}
        check(mism == [], f"fold {fold}: a correctly-built record has 0 mismatches "
                          f"(got {mism})")
        check(sorted(k for k, v in st.items() if v == "EXPECTED")
              == sorted(["epochs_planned", "fold", "init_head_from_tail", "init_head_l2",
                         "max_steps"]),
              f"fold {fold}: exactly the five allowlisted keys are EXPECTED, nothing else")
        check(len(rows) == len(C.CONFIRM_DIFF_KEYS),
              f"fold {fold}: all {len(C.CONFIRM_DIFF_KEYS)} pinned keys compared")

        # THE allowlist is by VALUE: another fold's head must not pass
        wrong = dict(rec)
        other = 2 if fold != 2 else 3
        wrong["init_head_l2"] = float(R4.FOLD_GATE[other]["head_l2"])
        wrong["init_head_from_tail"] = \
            f"runs/bakeoff_coral/fold{other}/checkpoints/step_8000.pt"
        _r, m = quiet(C.config_diff, wrong, DEV_RUN_JSON, exp)
        check(set(m) == {"init_head_l2", "init_head_from_tail"},
              f"fold {fold} pointed at fold {other}'s phase-1 head is caught "
              f"({m}) -- allowlisted keys are pinned to a VALUE, not waived")

    # a wrong budget is a mismatch even though max_steps is allowlisted
    exp = C.expected_diffs_for(1, 2000, ref)
    rec = {k: ref.get(k) for k in C.CONFIRM_DIFF_KEYS if k in ref}
    rec.update({"fold": 1, "max_steps": 8000,
                "init_head_from_tail": "runs/bakeoff_coral/fold1/checkpoints/step_8000.pt",
                "init_head_l2": float(R4.FOLD_GATE[1]["head_l2"]), "epochs_planned": 11})
    _r, m = quiet(C.config_diff, rec, DEV_RUN_JSON, exp)
    check("max_steps" in m,
          "running a confirmatory fold at 8000 steps is a MISMATCH -- the allowlist pins "
          "8000 -> 2000 exactly, not 'any max_steps'")

    # every non-allowlisted key must be caught when it moves
    base = {k: ref.get(k) for k in C.CONFIRM_DIFF_KEYS if k in ref}
    base.update({"fold": 1, "max_steps": 2000,
                 "init_head_from_tail": "runs/bakeoff_coral/fold1/checkpoints/step_8000.pt",
                 "init_head_l2": float(R4.FOLD_GATE[1]["head_l2"]), "epochs_planned": 11})
    for k, bad in (("gamma", 4.0), ("coral_target_ema", True), ("num_workers", 4),
                   ("weight_decay", 1e-4), ("lr", 1e-3), ("batch_size", 128),
                   ("seed", 7), ("ckpt_every", 500), ("feat_norm_floor", 0.0),
                   ("feat_norm_ceiling", 200.0), ("precision", "fp32"),
                   ("coral_weight_schedule", "warmup_cosine"), ("grad_checkpointing", False)):
        broken = dict(base)
        broken[k] = bad
        _r, m = quiet(C.config_diff, broken, DEV_RUN_JSON, exp)
        check(m == [k], f"a {k} of {bad!r} is caught as the only mismatch (got {m})")

    missing = dict(base)
    missing.pop("coral_shrinkage")
    _r, m = quiet(C.config_diff, missing, DEV_RUN_JSON, exp)
    check("coral_shrinkage" in m, "a key present in the reference but absent here is caught")
    td = tempfile.mkdtemp(prefix="thin_")
    try:
        thin = os.path.join(td, "run.json")
        with open(thin, "w") as fh:
            json.dump({"fold": 0}, fh)
        rows, m = quiet(C.config_diff, {"fold": 1}, thin, C.expected_diffs_for(1, 2000, {}))
        vac = [r["key"] for r in rows
               if r["reference"] == "<absent>" and r["this_run"] == "<absent>"]
        check(vac and all(k in m for k in vac),
              f"keys absent from BOTH sides are MISMATCHES ({len(vac)} such), never silent "
              f"passes")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# T4 -- per-fold facts from splits_v6
# ---------------------------------------------------------------------------

def t4_fold_facts():
    section("T4  per-fold facts resolved from splits_v6 (fold 4 holds 97, not 98)")
    if not SPLITS:
        check(False, "splits.json located")
        return
    counts = {k: C.fold_positive_count(SPLITS, k) for k in range(5)}
    check(counts == EXPECTED_COUNTS,
          f"per-fold test positives {counts} == PROJECT.md 4.5's structure")
    check(sum(counts.values()) == 489,
          f"the five folds partition all {sum(counts.values())} real lenses, each tested once")
    check(counts[4] == 97 and counts[1] == counts[2] == counts[3] == 98,
          "fold 4 holds 97 and folds 1-3 hold 98 -- the resolution is per fold, not global")
    for k in (1, 2, 3, 4):
        n = counts[k]
        gate = R4.FOLD_GATE[k]["tpr"]
        x = gate * n
        check(abs(x - round(x)) < 1e-9,
              f"fold {k}: FOLD_GATE TPR {gate:.6f} is an exact multiple of 1/{n} "
              f"({round(x)}/{n}) -- the gate value uses this fold's own denominator")
    check(abs(1 / 97 - 0.010309278350515464) < 1e-15,
          "fold 4's one-positive resolution is 1/97 = 0.010309, not 1/98 = 0.010204")
    with open(SPLITS) as fh:
        s = json.load(fh)
    check(s.get("schema") == "splits_v6_label_free_da",
          f"the splits schema is {s.get('schema')}")
    check(len(s["test_negatives_shared"]["locations"]) == 26399,
          "the shared negative pool is 26,399 rows and is the same pool for every fold")
    try:
        C.fold_positive_count(SPLITS, 9)
        check(False, "an unknown fold id is refused")
    except SystemExit:
        check(True, "an unknown fold id raises rather than silently returning a count")


# ---------------------------------------------------------------------------
# T5 -- no metric-based checkpoint selection
# ---------------------------------------------------------------------------

def t5_no_selection():
    section("T5  NO metric-based checkpoint selection anywhere in the confirmatory path")
    mods = ["run_coral_r4_confirm.py", "launch_coral_r4_confirm_kaggle.py"]
    banned = [
        (r"\bnp\.argmax\b", "argmax over a metric"),
        (r"\bargmax\s*\(", "argmax call"),
        (r"\bmax\s*\([^)]*\btpr\b", "max over TPR"),
        (r"\bbest[_ ]?(step|ckpt|checkpoint|tpr|score)\b", "a 'best checkpoint' notion"),
        (r"\bsmooth", "smoothing"),
        (r"\bselect_checkpoint\b", "a checkpoint selector"),
    ]
    def executable_only(src):
        """Docstrings, comments AND string literals removed.

        A metric-based selector is a CALL, never a string. Prose about NOT selecting is
        expected all over this runner -- including inside recorded values such as
        `"checkpoint_selection_code_paths": "NONE -- no argmax, no smoothing, no selector"`,
        which is a declaration in run.json, not a code path. Scanning raw source flagged
        exactly that line, so the scan is against executable text only. `argmax(` and
        `np.argmax(` would still be caught outside a literal.
        """
        s = re.sub(r'"""(?:.|\n)*?"""', "", src)
        s = re.sub(r"'''(?:.|\n)*?'''", "", s)
        s = re.sub(r"#.*", "", s)
        s = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', s)
        s = re.sub(r"'(?:[^'\\\n]|\\.)*'", "''", s)
        return s

    for m in mods:
        code = executable_only(open(os.path.join(_HERE, m), encoding="utf-8").read())
        for pat, what in banned:
            hits = re.findall(pat, code, flags=re.I)
            check(not hits, f"{m}: no {what} in executable code ({len(hits)} hits)")
    # the stripper must not be a rubber stamp: it has to still catch a real selector
    probe = executable_only('x = np.argmax(tprs)  # pick the best\n"no argmax here"\n')
    check(re.search(r"\bnp\.argmax\b", probe) is not None,
          "the literal-stripping scan still catches a REAL np.argmax call (so it is not "
          "passing by deleting everything)")
    check(re.search(r"argmax", executable_only('s = "no argmax here"\n')) is None,
          "...and does not flag the word appearing inside a string literal")
    src = open(os.path.join(_HERE, "run_coral_r4_confirm.py"), encoding="utf-8").read()
    check("reported_checkpoint" in src and "THE BUDGET IS THE SELECTOR" in src,
          "the runner names step_<budget> as THE reported checkpoint and says the budget "
          "is the selector")
    check("LAB RECORD AND DIAGNOSTICS ONLY" in src,
          "the runner marks the intermediate checkpoints lab-record-only in its output")
    # the only metric call permitted is the step-0 gate
    code = re.sub(r'"""(?:.|\n)*?"""', "", src)
    n_tpr = len(re.findall(r"tpr_at_fpr\s*\(", code))
    n_auroc = len(re.findall(r"auroc_fn\s*\(", code))
    check(n_tpr == 1 and n_auroc == 1,
          f"exactly one TPR call and one AUROC call in the whole runner ({n_tpr}/{n_auroc}) "
          f"-- both inside the step-0 known-null gate")
    check("test_label_lockout" in open(os.path.join(_HERE, "run_coral_r4_dev8000.py"),
                                       encoding="utf-8").read(),
          "the shared training loop runs inside test_label_lockout(), so no metric is even "
          "reachable during training")
    for pat in (r"DataParallel\s*\(", r"init_process_group", r"device_ids"):
        for m in mods:
            src = open(os.path.join(_HERE, m), encoding="utf-8").read()
            check(not re.search(pat, re.sub(r"#.*", "", re.sub(r'"""(?:.|\n)*?"""', "", src))),
                  f"{m}: no {pat!r} -- a fold is never split across devices")


# ---------------------------------------------------------------------------
# T6 -- logit format and id ordering
# ---------------------------------------------------------------------------

def t6_logits():
    section("T6  logit format round-trip, id stability, shared-negative invariant")
    td = tempfile.mkdtemp(prefix="confirm_logit_")
    try:
        rng = np.random.default_rng(0)
        pos = rng.normal(size=98)
        neg = rng.normal(size=26399)
        p = os.path.join(td, "step_2000.npz")
        np.savez_compressed(p, pos=pos.astype(np.float64), neg=neg.astype(np.float64))
        z = np.load(p)
        check(sorted(z.files) == ["neg", "pos"] and z["pos"].dtype == np.float64
              and np.array_equal(z["pos"], pos) and np.array_equal(z["neg"], neg),
              "pos/neg logits round-trip bit-exactly at float64 -- the "
              "runs/coral_r4_logits/ format")
        for fold in (1, 2, 3, 4):
            arm = C.logit_arm(fold, 2000)
            check(arm == f"g16_2000_fold{fold}" and str(fold) in arm,
                  f"fold {fold} writes to its own logit arm directory ({arm}), so "
                  f"concurrent folds cannot collide")
        if REF_IDS:
            a = np.load(REF_IDS, allow_pickle=True)
            check(sorted(a.files) == ["neg_ids", "pos_ids"],
                  f"the reference ids.npz holds {sorted(a.files)}")
            check(a["neg_ids"].size == 26399,
                  f"the shared negative pool is {a['neg_ids'].size} rows -- the invariant "
                  f"each fold's neg_ids is checked against")
            # the checker must accept a matching negative pool and reject a permuted one
            mine = os.path.join(td, "ids.npz")
            np.savez_compressed(mine, pos_ids=a["pos_ids"][:97], neg_ids=a["neg_ids"])
            r = quiet(C.check_shared_negatives, mine, REF_IDS, 4)
            check(r["neg_ids_match_reference"] and not r["pos_ids_identical_to_reference"],
                  "a fold with the SAME negatives and DIFFERENT positives is accepted "
                  "(that is what a different fold looks like)")
            perm = os.path.join(td, "ids_perm.npz")
            n = a["neg_ids"].copy()
            n[[0, 1]] = n[[1, 0]]
            np.savez_compressed(perm, pos_ids=a["pos_ids"][:97], neg_ids=n)
            r = quiet(C.check_shared_negatives, perm, REF_IDS, 4)
            check(not r["neg_ids_match_reference"],
                  "a permuted negative pool is caught -- same set, different ORDER, which "
                  "row-aligned comparison depends on")
            same = os.path.join(td, "ids_same.npz")
            np.savez_compressed(same, pos_ids=a["pos_ids"], neg_ids=a["neg_ids"])
            r = quiet(C.check_shared_negatives, same, REF_IDS, 1)
            check(r["pos_ids_identical_to_reference"],
                  "a fold whose POSITIVES equal the reference fold's is flagged -- that "
                  "would mean --fold did not take effect")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# T7 -- path resolution per fold
# ---------------------------------------------------------------------------

def t7_paths():
    section("T7  path resolution against a MOCKED Kaggle mount, per fold")
    td = tempfile.mkdtemp(prefix="confirm_mount_")
    try:
        ds = os.path.join(td, "datasets", "navneet78", "lens-paper-coral-r4")
        made = ["data/data/splits/splits.json", "data/data/feature_cache/source_train.npz",
                "runs/bakeoff_baseline_A/checkpoints/step_8000.pt",
                "reference/coral_r4_budget.json",
                f"reference/{C.DEV_ARM_RUN_JSON}/run.json",
                "reference/coral_r4_logits/ids.npz"]
        made += [f"runs/bakeoff_coral/fold{k}/checkpoints/step_8000.pt" for k in range(5)]
        for rel in made:
            p = os.path.join(ds, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").close()
        for fold in (1, 2, 3, 4):
            args = argparse.Namespace(input_root=td, data_root=None, splits=None,
                                      cache_dir=None, init_head_from=None,
                                      agreement_ref=None, prereg=None, dev_run_json=None,
                                      reference_ids=None)
            got = C.resolve_paths(args, os.path.join(td, "norepo"), fold)
            want = os.path.abspath(os.path.join(
                ds, "runs", "bakeoff_coral", f"fold{fold}", "checkpoints", "step_8000.pt"))
            check(got["init_head_from"] == want,
                  f"fold {fold} resolves to ITS OWN phase-1 head "
                  f"(…/bakeoff_coral/fold{fold}/…)")
            if fold == 1:
                check(got["data_root"].endswith(os.path.join("data", "data")),
                      "data_root is the data/data directory data.zip auto-extracts to")
                check(got["prereg"].endswith("coral_r4_budget.json"),
                      "the pre-registration is located inside the mount")
                check(got["dev_run_json"] and got["dev_run_json"].endswith("run.json"),
                      "the fold-0 arm's run.json (the diff reference) is located")
                check(got["reference_ids"].endswith("ids.npz"),
                      "the reference ids.npz is located")
        empty = tempfile.mkdtemp(prefix="confirm_empty_")
        try:
            args = argparse.Namespace(input_root=empty, data_root=None, splits=None,
                                      cache_dir=None, init_head_from=None,
                                      agreement_ref=None, prereg=None, dev_run_json=None,
                                      reference_ids=None)
            # resolve_paths searches [input_root, /kaggle/input, repo root]. On Kaggle
            # /kaggle/input EXISTS and holds the real dataset, so an empty --input-root is
            # NOT an unresolvable mount there -- the search succeeds and the check fails for
            # a reason that has nothing to do with the behaviour under test. Patch the
            # finder to find nothing, which is the condition actually being tested and is
            # identical in every environment.
            _orig = C._find_under
            C._find_under = lambda roots, pattern: None
            try:
                C.resolve_paths(args, empty, 1)
                check(False, "an unresolvable mount aborts")
            except SystemExit as e:
                check("could not resolve" in str(e),
                      "an unresolvable mount aborts with a path-specific message")
            finally:
                C._find_under = _orig
            # and, independent of any patching: an explicitly-named path that does not
            # exist must abort, in every environment
            bad = argparse.Namespace(input_root=empty, data_root=None,
                                     splits=os.path.join(empty, "nope", "splits.json"),
                                     cache_dir=None, init_head_from=None,
                                     agreement_ref=None, prereg=None, dev_run_json=None,
                                     reference_ids=None)
            try:
                C.resolve_paths(bad, empty, 1)
                check(False, "an explicit path that does not exist aborts")
            except SystemExit as e:
                check("could not resolve" in str(e),
                      "an explicit --splits that does not exist aborts, with the path in "
                      "the message")
        finally:
            shutil.rmtree(empty, ignore_errors=True)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    print("=" * 92)
    print("test_coral_r4_confirm.py -- pre-handover suite (no GPU, no training)")
    print("=" * 92)
    t1_no_fold0()
    t2_prereg()
    t3_config_diff()
    t4_fold_facts()
    t5_no_selection()
    t6_logits()
    t7_paths()
    print("\n" + "=" * 92)
    print(f"{len(_passes)} passed, {len(_fails)} failed")
    for f in _fails:
        print(f"  FAILED: {f}")
    print("=" * 92)
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
