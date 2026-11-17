"""
test_coral_r4_dev8000.py
========================
Pre-handover suite for ``run_coral_r4_dev8000.py`` (PROJECT.md 9: nothing is presented
untested). No GPU, no training, no real 8000-step anything -- it tests the machinery that
surrounds the run, against the REAL stored artifacts wherever they exist.

    T1  the reproduction check, against the REAL stored g16 steps.csv
        - KNOWN NULL: the reference compared to itself is REPRODUCED and bit-exact
        - DISCRIMINATION: the g4 arm's own values at steps 250 / 2000 are DIVERGED
          (that is the config change the tolerances were calibrated to catch)
        - a within-tolerance perturbation passes; a beyond-tolerance one fails
        - sup_loss is never an arbiter
    T2  the config diff -- mismatch detection, the vacuous-guard rule, expected differences
    T3  path resolution against a MOCKED Kaggle mount in the flat layout the real dataset
        mounts at (datasets/<owner>/<slug>/...), including the data/data double directory
    T4  logit persistence: format round-trip and id-order agreement with the REAL
        runs/coral_r4_logits/ids.npz, plus the mismatch path
    T5  resume bookkeeping: rows past the resume step are dropped, earlier rows are not
    T6  the pinned constants agree with the artifacts they were read from

Run:  python -u scripts/test_coral_r4_dev8000.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_coral_r4_dev8000 as D                                    # noqa: E402
from run_coral_gamma_sweep import STEP_COLUMNS                      # noqa: E402

ROOT = os.path.normpath(os.path.join(_HERE, ".."))

# The reference artifacts live in different places depending on where this runs: in the repo
# they sit under kaggle/coral_r4/sweep_run/ and runs/; on Kaggle they are shipped inside the
# dataset under reference/. Locate them the same way the runner does rather than assuming
# one layout -- a suite that silently skips its own checks because a path moved is worse
# than no suite.
_SEARCH = [ROOT, "/kaggle/input", os.path.dirname(_HERE)]


def locate(*patterns):
    for pat in patterns:
        hit = D._find_under(_SEARCH, pat)
        if hit:
            return hit
    return None


_g16 = locate(os.path.join("coral_r4_g16_2000", "steps.csv"),
              os.path.join("coral_r4_sweep", "g16", "steps.csv"))
_g4 = locate(os.path.join("coral_r4_g4_2000", "steps.csv"),
             os.path.join("coral_r4_sweep", "g4", "steps.csv"))
_ids = locate(os.path.join("coral_r4_logits", "ids.npz"))
G16 = os.path.dirname(_g16) if _g16 else os.path.join(ROOT, "__missing_g16__")
G4 = os.path.dirname(_g4) if _g4 else os.path.join(ROOT, "__missing_g4__")
REF_LOGITS = os.path.dirname(_ids) if _ids else os.path.join(ROOT, "__missing_logits__")
GAMMA_SELECTION = locate("coral_r4_gamma_selection.json")
G16_LOG = locate("g16.log")

_fails, _passes = [], []


def check(cond, msg):
    (_passes if cond else _fails).append(msg)
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")


def section(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def row_at(steps_csv, step):
    with open(steps_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if int(float(r["step"])) == int(step):
                return r
    return None


def as_quants(row):
    return {q: D._num(row.get(q)) for q in D.ReproCheck.QUANTS}


# ---------------------------------------------------------------------------
# T1 -- the reproduction check, on the real stored traces
# ---------------------------------------------------------------------------

def t1_repro():
    section("T1  reproduction check vs the REAL stored g16 arm")
    ref_csv = os.path.join(G16, "steps.csv")
    if not os.path.isfile(ref_csv):
        check(False, f"stored g16 steps.csv present ({ref_csv})")
        return
    check(True, f"stored g16 steps.csv present ({os.path.relpath(ref_csv, ROOT)})")

    rc = D.ReproCheck(ref_csv, at_steps=D.REPRO_STEPS, window=D.REPRO_WINDOW)
    check(rc.available and len(rc.rows) == 2000,
          f"loaded {len(rc.rows)} reference rows, at_steps={rc.at}")

    for step in D.REPRO_STEPS:
        row = row_at(ref_csv, step)
        vals = as_quants(row)
        ref_win = rc._ref(step)["win_mean"]
        cum = sum(D._num(rc.rows[s].get("step_skipped")) or 0.0 for s in range(1, step + 1))

        # ---- KNOWN NULL: the reference against itself ----
        res = quiet(rc.compare, step, vals, ref_win, D._num(row.get("amp_scale")), cum)
        check(res["status"] == "REPRODUCED" and res["bit_exact"],
              f"step {step}: the reference compared to ITSELF is REPRODUCED and bit-exact "
              f"(known null)")
        check(res["amp_scale"]["match"] and res["fp16_skipped_cumulative"]["match"],
              f"step {step}: amp scale {res['amp_scale']['reference']} and cumulative fp16 "
              f"skips {res['fp16_skipped_cumulative']['reference']} both match")

        # ---- DISCRIMINATION: the g4 arm is a real config change ----
        g4row = row_at(os.path.join(G4, "steps.csv"), step)
        if g4row:
            res4 = quiet(rc.compare, step, as_quants(g4row), as_quants(g4row),
                         D._num(g4row.get("amp_scale")), cum)
            hit = set(res4["failed_arbiters"])
            check(res4["status"] == "DIVERGED"
                  and {"src_perdim_std", "tgt_perdim_std", "coral_raw"} <= hit,
                  f"step {step}: the gamma=4 arm's own values are DIVERGED on "
                  f"{sorted(hit)} -- the smallest available config change is caught")

        # ---- the tolerance boundary, in both directions ----
        for q in ("src_perdim_std", "coral_raw", "src_feat_l2"):
            tol = D.REPRO_TOL[q]["rel"]
            inside = dict(vals)
            inside[q] = vals[q] * (1 + 0.5 * tol)
            r_in = quiet(rc.compare, step, inside, ref_win, D._num(row.get("amp_scale")), cum)
            outside = dict(vals)
            outside[q] = vals[q] * (1 + 40 * tol)
            r_out = quiet(rc.compare, step, outside, outside,
                          D._num(row.get("amp_scale")), cum)
            check(q not in r_in["failed_arbiters"],
                  f"step {step}: {q} perturbed by {0.5 * tol:.1%} (inside tol) PASSES")
            check(q in r_out["failed_arbiters"],
                  f"step {step}: {q} perturbed by {40 * tol:.0%} (far outside tol) FAILS")

    # ---- sup_loss is never an arbiter ----
    step = D.REPRO_STEPS[0]
    row = row_at(ref_csv, step)
    vals = as_quants(row)
    vals["sup_loss"] = vals["sup_loss"] * 50.0
    res = quiet(rc.compare, step, vals, vals, D._num(row.get("amp_scale")), 0.0)
    check("sup_loss" not in res["failed_arbiters"],
          "a 50x sup_loss excursion does not flip the verdict (it is informational only)")

    # ---- a missing reference step is reported, not silently passed ----
    rc2 = D.ReproCheck(ref_csv, at_steps=(7777,), window=5)
    res = quiet(rc2.compare, 7777, vals, vals, 0.0, 0.0)
    check(res["status"] == "NO_REFERENCE",
          "a step the reference does not contain returns NO_REFERENCE, never REPRODUCED")

    # ---- no reference file at all ----
    rc3 = D.ReproCheck(os.path.join(ROOT, "does_not_exist.csv"))
    check(rc3.available is False and rc3.rows == {},
          "a missing reference file leaves the check unavailable rather than crashing")


# ---------------------------------------------------------------------------
# T2 -- the config diff
# ---------------------------------------------------------------------------

def t2_config_diff():
    section("T2  config diff -- mismatch detection, vacuous guard, expected differences")
    ref_cfg = os.path.join(G16, "sweep.json")
    with open(ref_cfg) as fh:
        ref = json.load(fh)

    base = {k: ref.get(k) for k in D.DIFF_KEYS if k in ref}
    for k, (v, _p) in D.REFERENCE_SUPPLEMENT.items():
        base.setdefault(k, v)
    base["init_head_from_tail"] = D._path_tail(ref["init_head_from"], 5)
    base["max_steps"] = D.DEV_STEPS
    base["feat_norm_floor"] = D.DEFAULT_FEAT_NORM_FLOOR

    rows, mism, _ = quiet(D.config_diff, base, ref_cfg)
    check(mism == [], f"a correctly-built record has 0 mismatches (got {mism})")
    st = {r["key"]: r["status"] for r in rows}
    check(st["max_steps"] == "EXPECTED",
          "max_steps 2000 -> 8000 is reported as the declared EXPECTED difference")
    check(st["feat_norm_floor"] == "NEW",
          "feat_norm_floor is reported as NEW (absent from the reference, declared here)")
    check(st["gamma"] == "MATCH" and st["coral_target_ema"] == "MATCH"
          and st["init_head_l2"] == "MATCH",
          "gamma, coral_target_ema and the phase-1 head norm all MATCH")
    check(len(rows) == len(D.DIFF_KEYS),
          f"every one of the {len(D.DIFF_KEYS)} declared keys is compared, none skipped")

    for k, bad in (("gamma", 4.0), ("coral_target_ema", True), ("num_workers", 4),
                   ("weight_decay", 1e-4), ("lr", 1e-3), ("batch_size", 128),
                   ("init_head_l2", 67.536934), ("seed", 7), ("ckpt_every", 500),
                   ("coral_weight_schedule", "warmup_cosine")):
        broken = dict(base)
        broken[k] = bad
        _r, m, _ = quiet(D.config_diff, broken, ref_cfg)
        check(m == [k], f"a {k} of {bad!r} is caught as the only mismatch (got {m})")

    # the vacuous-guard rule round 4 added: absent from BOTH sides is a mismatch
    missing = dict(base)
    missing.pop("coral_shrinkage")
    _r, m, _ = quiet(D.config_diff, missing, ref_cfg)
    check("coral_shrinkage" in m,
          "a key present in the reference but absent from the record is a mismatch")
    with tempfile.TemporaryDirectory() as td:
        thin = os.path.join(td, "sweep.json")
        with open(thin, "w") as fh:
            json.dump({"gamma": 16.0}, fh)
        rows2, m2, _ = quiet(D.config_diff, {"gamma": 16.0}, thin)
        vac = [r["key"] for r in rows2
               if r["reference"] == "<absent>" and r["this_run"] == "<absent>"]
        check(vac and all(k in m2 for k in vac),
              f"keys absent from BOTH configs are MISMATCHES, not silent passes "
              f"({len(vac)} such keys)")

    # a wrong expected-difference pair is a mismatch, not a free pass
    wrong = dict(base)
    wrong["max_steps"] = 4000
    _r, m, _ = quiet(D.config_diff, wrong, ref_cfg)
    check("max_steps" in m,
          "max_steps = 4000 is a MISMATCH -- the expected difference is 2000 -> 8000 "
          "exactly, not 'any change to max_steps'")

    # no reference file at all
    rows3, m3, ref3 = quiet(D.config_diff, base, os.path.join(ROOT, "nope.json"))
    check(ref3 == {} and len(m3) > 0,
          "with NO reference config every key falls to <absent> and the run is refused")


# ---------------------------------------------------------------------------
# T3 -- path resolution against a mocked Kaggle mount
# ---------------------------------------------------------------------------

def t3_paths():
    section("T3  path resolution against a MOCKED Kaggle mount (flat datasets/ layout)")
    import argparse
    td = tempfile.mkdtemp(prefix="coral_r4_mount_")
    try:
        ds = os.path.join(td, "datasets", "navneet78", "lens-paper-coral-r4")
        # data.zip auto-extracts to data/data -- the double directory the cells document
        made = [
            os.path.join(ds, "data", "data", "splits", "splits.json"),
            os.path.join(ds, "data", "data", "feature_cache", "source_train.npz"),
            os.path.join(ds, "data", "data", "feature_cache", "test_pos_fold0.npz"),
            os.path.join(ds, "runs", "bakeoff_coral", "fold0", "checkpoints", "step_8000.pt"),
            os.path.join(ds, "runs", "bakeoff_baseline_A", "checkpoints", "step_8000.pt"),
            os.path.join(ds, "reference", "coral_r4_g16_2000", "sweep.json"),
            os.path.join(ds, "reference", "coral_r4_g16_2000", "steps.csv"),
            os.path.join(ds, "reference", "coral_r4_logits", "ids.npz"),
        ]
        for p in made:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").close()

        args = argparse.Namespace(input_root=td, data_root=None, splits=None,
                                  cache_dir=None, init_head_from=None, agreement_ref=None,
                                  reference_dir=None, reference_ids=None)
        got = D.resolve_paths(args, os.path.join(td, "no_repo_here"))
        check(got["splits"] == os.path.abspath(made[0]),
              "splits.json found under datasets/<owner>/<slug>/data/data/splits/")
        check(got["data_root"] == os.path.abspath(os.path.join(ds, "data", "data")),
              "data_root is the parent of splits/ -- the double data/data directory, "
              "which is what LensDataset wants")
        check(got["cache_dir"] == os.path.abspath(os.path.join(ds, "data", "data",
                                                               "feature_cache")),
              "feature_cache located from source_train.npz, not assumed")
        check(got["init_head_from"] == os.path.abspath(made[3]),
              "the round-1 CORAL fold-0 phase-1 head is located")
        check(got["agreement_ref"] == os.path.abspath(made[4]),
              "the baseline-A agreement reference is located")
        check(got["reference_dir"] == os.path.abspath(os.path.dirname(made[5])),
              "the stored g16 reference directory is located")
        check(got["reference_ids"] == os.path.abspath(made[7]),
              "runs/coral_r4_logits/ids.npz is located for the id-order check")

        # the OTHER real layout: /kaggle/input/<slug>, no datasets/<owner> level
        td2 = tempfile.mkdtemp(prefix="coral_r4_mount2_")
        try:
            ds2 = os.path.join(td2, "lens-paper-coral-r4")
            for rel in ("data/splits/splits.json", "data/feature_cache/source_train.npz",
                        "runs/bakeoff_coral/fold0/checkpoints/step_8000.pt"):
                p = os.path.join(ds2, *rel.split("/"))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "wb").close()
            a2 = argparse.Namespace(input_root=td2, data_root=None, splits=None,
                                    cache_dir=None, init_head_from=None,
                                    agreement_ref=None, reference_dir=None,
                                    reference_ids=None)
            g2 = D.resolve_paths(a2, os.path.join(td2, "no_repo_here"))
            check(g2["data_root"] == os.path.abspath(os.path.join(ds2, "data")),
                  "the single-level /kaggle/input/<slug> layout also resolves")
        finally:
            shutil.rmtree(td2, ignore_errors=True)

        # explicit flags win over the search
        a3 = argparse.Namespace(input_root=td, data_root=None,
                                splits=os.path.abspath(made[0]), cache_dir=None,
                                init_head_from=os.path.abspath(made[3]),
                                agreement_ref=None, reference_dir=None,
                                reference_ids=None)
        g3 = D.resolve_paths(a3, td)
        check(g3["splits"] == os.path.abspath(made[0]),
              "an explicit --splits is honoured rather than searched over")

        # a missing required piece ABORTS rather than training against nothing
        empty = tempfile.mkdtemp(prefix="coral_r4_empty_")
        try:
            a4 = argparse.Namespace(input_root=empty, data_root=None, splits=None,
                                    cache_dir=None, init_head_from=None,
                                    agreement_ref=None, reference_dir=None,
                                    reference_ids=None)
            try:
                D.resolve_paths(a4, empty)
                check(False, "an unresolvable mount raises SystemExit")
            except SystemExit as e:
                check("could not resolve" in str(e),
                      f"an unresolvable mount aborts with a path-specific message "
                      f"[{type(e).__name__}]")
        finally:
            shutil.rmtree(empty, ignore_errors=True)
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# T4 -- logit persistence + id order
# ---------------------------------------------------------------------------

def t4_logits():
    section("T4  logit persistence format + id order vs the REAL runs/coral_r4_logits/")
    ref_ids = os.path.join(REF_LOGITS, "ids.npz")
    if not os.path.isfile(ref_ids):
        check(False, f"reference ids.npz present ({ref_ids})")
        return
    z = np.load(ref_ids, allow_pickle=True)
    check(sorted(z.files) == ["neg_ids", "pos_ids"],
          f"reference ids.npz holds exactly {sorted(z.files)}")
    check(z["pos_ids"].size == 98 and z["neg_ids"].size == 26399,
          f"reference pool is {z['pos_ids'].size} pos / {z['neg_ids'].size} neg")

    ref_step = (locate(os.path.join("coral_r4_logits", "g16", "step_2000.npz"))
                or os.path.join(REF_LOGITS, "g16", "step_2000.npz"))
    if os.path.isfile(ref_step):
        s = np.load(ref_step)
        check(sorted(s.files) == ["neg", "pos"] and s["pos"].dtype == np.float64,
              f"a reference step npz holds {sorted(s.files)} at dtype {s['pos'].dtype} -- "
              f"the format this run writes")
        check(s["pos"].size == z["pos_ids"].size and s["neg"].size == z["neg_ids"].size,
              "the reference logit vectors are row-aligned with ids.npz")

    td = tempfile.mkdtemp(prefix="coral_r4_logit_")
    try:
        # exact copy -> match
        mine = os.path.join(td, "ids.npz")
        np.savez_compressed(mine, pos_ids=z["pos_ids"], neg_ids=z["neg_ids"])
        r = quiet(D.check_ids_against_reference, mine, ref_ids)
        check(r["checked"] and r["pos_ids_match"] and r["neg_ids_match"],
              "an identical id array reports a MATCH")

        # a permuted order -> reported, not silently accepted
        perm = os.path.join(td, "ids_perm.npz")
        p = z["pos_ids"].copy()
        p[[0, 1]] = p[[1, 0]]
        np.savez_compressed(perm, pos_ids=p, neg_ids=z["neg_ids"])
        r = quiet(D.check_ids_against_reference, perm, ref_ids)
        check(r["checked"] and not r["pos_ids_match"] and r["neg_ids_match"],
              "swapping two positive ids is caught -- same set, different ORDER, which is "
              "what a row-aligned paired bootstrap depends on")

        # a missing reference is skipped, not fatal
        r = quiet(D.check_ids_against_reference, mine, os.path.join(td, "nope.npz"))
        check(r["checked"] is False and "note" in r,
              "a missing reference ids.npz skips the check instead of aborting the run")

        # round-trip of the format this run writes
        pos = np.random.default_rng(0).normal(size=98)
        neg = np.random.default_rng(1).normal(size=26399)
        dst = os.path.join(td, "step_250.npz")
        np.savez_compressed(dst, pos=pos.astype(np.float64), neg=neg.astype(np.float64))
        b = np.load(dst)
        check(sorted(b.files) == ["neg", "pos"]
              and np.array_equal(b["pos"], pos) and np.array_equal(b["neg"], neg)
              and b["pos"].dtype == np.float64,
              "pos/neg logits round-trip bit-exactly at float64")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# T5 -- resume bookkeeping
# ---------------------------------------------------------------------------

def t5_resume_csv():
    section("T5  resume bookkeeping -- rows past the resume step are dropped")
    td = tempfile.mkdtemp(prefix="coral_r4_trunc_")
    try:
        p = os.path.join(td, "steps.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=STEP_COLUMNS)
            w.writeheader()
            for s in range(1, 301):
                w.writerow({c: (s if c == "step" else 0.0) for c in STEP_COLUMNS})
        n = D._truncate_csv_after(p, 250)
        with open(p, newline="") as fh:
            rows = list(csv.DictReader(fh))
        check(n == 50 and len(rows) == 250 and int(rows[-1]["step"]) == 250,
              f"50 rows past step 250 dropped, 250 kept, last step = {rows[-1]['step']}")
        check(list(rows[0].keys()) == list(STEP_COLUMNS),
              "the header survives the rewrite unchanged")
        check(D._truncate_csv_after(p, 250) == 0,
              "truncating again is a no-op (idempotent)")
        check(D._truncate_csv_after(os.path.join(td, "absent.csv"), 250) == 0,
              "truncating a file that does not exist is a no-op, not an error")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# T6 -- the pinned constants agree with the artifacts they came from
# ---------------------------------------------------------------------------

def t6_constants():
    section("T6  pinned constants vs the artifacts they were read from")
    sel = GAMMA_SELECTION
    if sel and os.path.isfile(sel):
        with open(sel) as fh:
            s = json.load(fh)
        check(float(s["selected_gamma"]) == D.GAMMA,
              f"gamma {D.GAMMA} is the value coral_r4_gamma_selection.json selected "
              f"(branch {s.get('branch_taken')})")
        check(s["constraints_for_confirmatory_runs"]["coral_target_ema"] is False
              and float(s["constraints_for_confirmatory_runs"]["feat_norm_ceiling"]) ==
              D.DEFAULT_FEAT_NORM_CEILING,
              "target EMA OFF and ceiling 110 are the selection's own stated constraints")
    ref_cfg = os.path.join(G16, "sweep.json")
    if os.path.isfile(ref_cfg):
        with open(ref_cfg) as fh:
            r = json.load(fh)
        check(int(r["log_every"]) == D.LOG_EVERY and int(r["ckpt_every"]) == D.CKPT_EVERY,
              f"log_every {D.LOG_EVERY} and ckpt_every {D.CKPT_EVERY} match the stored arm")
        check(int(r["max_steps"]) == 2000 and D.DEV_STEPS == 8000,
              "the stored arm is 2000 steps and this run is 8000 -- the one declared "
              "difference")
        check(int(r["steps_completed"]) == 2000 and r["aborted"] is False,
              "the stored arm completed all 2000 steps without aborting, so it is a valid "
              "reference over the whole compared window")
    log = G16_LOG
    if log and os.path.isfile(log):
        with open(log, errors="ignore") as fh:
            head = fh.readline() + fh.readline()
        check("--num-workers 2" in head,
              "the reference arm's own command line confirms --num-workers 2, the value "
              "REFERENCE_SUPPLEMENT pins (it sets the augmentation RNG stream)")
    ap = D.build_argparser()
    defaults = vars(ap.parse_args([]))
    check(defaults["num_workers"] == 2 and defaults["gamma"] == 16.0
          and defaults["max_steps"] == 8000 and defaults["ckpt_every"] == 250
          and defaults["fold"] == 0 and defaults["coral_target_ema"] is False
          and defaults["eval_fp16"] is False and defaults["fp16"] is True,
          "the CLI defaults ARE the pinned arm (gamma 16, fold 0, 8000, ckpt 250, "
          "workers 2, target EMA off, fp16 train / fp32 eval)")
    check(defaults["feat_norm_floor"] == 40.0 and defaults["feat_norm_ceiling"] == 110.0,
          "the feature-norm band defaults are floor 40 / ceiling 110")
    check(tuple(int(x) for x in defaults["repro_steps"].split(",")) == D.REPRO_STEPS,
          f"the reproduction check defaults to steps {D.REPRO_STEPS}")
    # single-GPU by construction: look for an actual CALL, not a mention in prose. The
    # docstrings deliberately talk about DataParallel to say the run does not use it.
    import re as _re
    src = open(os.path.join(_HERE, "run_coral_r4_dev8000.py"), encoding="utf-8").read()
    calls = _re.findall(r"\b(?:nn\.)?(?:Distributed)?DataParallel\s*\(", src)
    check(not calls,
          f"no DataParallel/DDP is ever CONSTRUCTED in the script ({len(calls)} call "
          f"sites) -- single-GPU by construction, so the per-device batch, and with it "
          f"CORAL's 256-draw covariance estimate, is never halved")
    check(not _re.search(r"\b(?:device_ids|output_device|init_process_group|"
                         r"CUDA_VISIBLE_DEVICES\s*\]\s*=)", src),
          "no device sharding, no process-group init and no CUDA_VISIBLE_DEVICES rewrite")


def main():
    print("=" * 88)
    print("test_coral_r4_dev8000.py -- pre-handover suite (no GPU, no training)")
    print("=" * 88)
    t1_repro()
    t2_config_diff()
    t3_paths()
    t4_logits()
    t5_resume_csv()
    t6_constants()
    print("\n" + "=" * 88)
    print(f"{len(_passes)} passed, {len(_fails)} failed")
    for f in _fails:
        print(f"  FAILED: {f}")
    print("=" * 88)
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
