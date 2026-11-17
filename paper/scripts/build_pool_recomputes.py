r"""
build_pool_recomputes.py -- the pool-dependent JSON artifacts, recomputed on D1.
================================================================================

    python paper/scripts/build_pool_recomputes.py            # skip outputs that exist
    python paper/scripts/build_pool_recomputes.py --force    # recompute

Five artifacts behind the tables were written by the project's own analysis scripts on
the pre-registered pool. Each is recomputable from stored per-object scores, so each is
recomputed here on D1 by **the same code**, called in-process with only its score loader
routed through ``_common.pool_filter``:

  * ``lines_ratios``        -- ``scripts/write_lines_ratios.main`` (T4's comp@24% and X);
  * ``foldmean_bootstrap``  -- ``scripts/bootstrap_foldmean.main`` (the licensed
    FreeMatch frozen -> unfrozen pair; B = 5000, seed 0);
  * ``round1_paired_*``     -- ``scripts/compare_methods.compare`` (T3's paired column;
    B = 1000, seed 0);
  * ``window_average``      -- ``scripts/analyze_window_average.main`` (T6), for the five
    unfrozen arms only: Baseline A's per-checkpoint scores before step 8000 are not
    stored, and recomputing them would mean loading its head checkpoints, which this pass
    does not do. Baseline A's T6 row therefore stays on the pre-registered pool, labelled;
  * ``lever1_aug_precision`` -- ``scripts/decompose_aug_precision.main`` (A4's three cells
    and legs; B = 2000, seed 0), with the venue difference beside it.

**Before writing a D1 file each builder re-runs itself on the pre-registered pool and
asserts that it reproduces the project's existing artifact.** A recomputation path that
cannot reproduce the original is not trusted to produce the new value.

Reads stored scores only. Trains nothing, loads no model, touches no GPU. Writes only
``paper/data/*_D1.json``; the reproduction runs write to a temporary directory outside
the project.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

import _common as C

DATA = C.DATA
TMP = tempfile.mkdtemp(prefix="pool_recompute_")


def _close(a, b, where, tol=1e-12):
    """Recursive numeric equality of two JSON trees (keys, lengths and values)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{where}: keys differ"
        for k in a:
            _close(a[k], b[k], f"{where}.{k}", tol)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{where}: lengths differ"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{where}[{i}]", tol)
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            assert a is None and b is None, f"{where}: {a} vs {b}"
        elif math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), f"{where}: {a} vs {b}"
        else:
            assert abs(a - b) <= tol, f"{where}: {a} vs {b}"
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"


def _load(p):
    return json.load(open(p, encoding="utf-8"))


def _write(obj, name):
    dest = os.path.join(DATA, name)
    obj = dict(obj)
    obj["test_pool"] = {"pool": C.POOL, "rule": "audit/decontam/rule.json",
                        "rule_sha256": C.pool_sets()["rule_sha256"],
                        "built_by": "paper/scripts/build_pool_recomputes.py"}
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print(f"  wrote {os.path.relpath(dest, C.REPO)}")


def _need(name, force):
    if os.path.exists(os.path.join(DATA, name)) and not force:
        print(f"  have paper/data/{name} (use --force to recompute)")
        return False
    return True


# ------------------------------------------------------------------ lines ratios
def lines_ratios(force):
    name = f"lines_ratios_{C.POOL}.json"
    if not _need(name, force):
        return
    import write_lines_ratios as WLR
    with C.pool("orig"):
        WLR.main(["--out", os.path.join(TMP, "lr_orig.json")])
    a, b = _load(os.path.join(TMP, "lr_orig.json")), _load(C.ART["lines_ratios"])
    _close(a["arms"], b["arms"], "lines_ratios(orig).arms")
    print("  lines_ratios: the pre-registered pool reproduces runs/round7/lines_ratios.json")
    WLR.main(["--out", os.path.join(TMP, "lr_pool.json")])
    _write(_load(os.path.join(TMP, "lr_pool.json")), name)


# ------------------------------------------------------------------ licensed pair
def foldmean_bootstrap(force):
    name = f"foldmean_bootstrap_{C.POOL}.json"
    if not _need(name, force):
        return
    import bootstrap_foldmean as BF
    raw = BF.load_arm_scores

    def pooled(root, fold, r1_step=None):
        up, un, rp, rn, step = raw(root, fold, r1_step)
        d = os.path.join(root, "runs", f"freematch_nb3_unfrozen_fold{fold}", "logits")
        z = np.load(os.path.join(d, "ids.npz"))
        up, un = C.pool_filter(up, un, list(z[f"test_pos_fold{fold}"]),
                               list(z["test_negatives_shared"]),
                               where=f"licensed pair nb3 fold {fold} step {step}")
        rp, rn = C.round1_scores("bakeoff_freematch_5fold", fold)
        return up, un, rp, rn, step

    BF.load_arm_scores = pooled
    try:
        with C.pool("orig"):
            BF.main(["--out", os.path.join(TMP, "fb_orig.json")])
        a, b = _load(os.path.join(TMP, "fb_orig.json")), _load(C.ART["foldmean_bootstrap"])
        for k in ("diff", "ci", "unfrozen_fold_mean", "round1_fold_mean", "n_shared_negatives",
                  "B", "seed"):
            _close(a[k], b[k], f"foldmean_bootstrap(orig).{k}")
        print("  foldmean_bootstrap: the pre-registered pool reproduces "
              "runs/foldmean_bootstrap.json")
        BF.main(["--out", os.path.join(TMP, "fb_pool.json")])
    finally:
        BF.load_arm_scores = raw
    _write(_load(os.path.join(TMP, "fb_pool.json")), name)


# ------------------------------------------------------------------ round-1 paired
ROUND1 = [("coral", "bakeoff_coral", "CORAL"),
          ("adamatch", "bakeoff_adamatch_5fold", "AdaMatch"),
          ("freematch", "bakeoff_freematch_5fold", "FreeMatch")]


def round1_paired(force):
    import compare_methods as CM
    raw = CM._load_fold

    def pooled(run_dir, k):
        d = raw(run_dir, k)
        keep = (C.pool_keep_neg(d["neg_ids"]) if C.POOL != "orig"
                else np.ones(d["neg_ids"].size, bool))
        d["pos_scores"], d["neg_scores"] = C.pool_filter(
            d["pos_scores"], d["neg_scores"], d.get("pos_ids"), d["neg_ids"],
            where=f"round-1 {os.path.basename(run_dir)} fold {k}")
        d["neg_ids"] = d["neg_ids"][keep]
        return d

    CM._load_fold = pooled
    try:
        base = C.R("runs", "bakeoff_baseline_A")
        for tag, run, disp in ROUND1:
            name = f"round1_paired_{tag}_{C.POOL}.json"
            if not _need(name, force):
                continue
            kw = dict(name_a="Baseline A", name_b=disp, verbose=False)
            with C.pool("orig"):
                a = CM.compare(base, C.R("runs", run), **kw)
            b = _load(os.path.join(DATA, f"round1_paired_{tag}.json"))
            _close(json.loads(json.dumps(a["aggregate"])), b["aggregate"],
                   f"round1_paired_{tag}(orig).aggregate")
            print(f"  round1_paired_{tag}: the pre-registered pool reproduces "
                  f"paper/data/round1_paired_{tag}.json")
            r = CM.compare(base, C.R("runs", run), **kw)
            _write(json.loads(json.dumps(r)), name)
    finally:
        CM._load_fold = raw


# ------------------------------------------------------------------ window averaging
BASELINE = "Baseline A (frozen)"


def window_average(force):
    name = f"window_average_{C.POOL}.json"
    if not _need(name, force):
        return
    import analyze_window_average as AWA
    held = AWA.ARMS.pop(BASELINE)              # restored below, in the same order
    try:
        with C.pool("orig"):
            AWA.main(["--out", os.path.join(TMP, "wa_orig.json")])
        a, b = _load(os.path.join(TMP, "wa_orig.json")), _load(C.ART["window_average"])
        for arm in a["arms"]:
            _close(a["arms"][arm], b["arms"][arm], f"window_average(orig).{arm}")
        print("  window_average: the pre-registered pool reproduces runs/round7/"
              "window_average.json for the five unfrozen arms")
        AWA.main(["--out", os.path.join(TMP, "wa_pool.json")])
    finally:
        AWA.ARMS[BASELINE] = held
    out = _load(os.path.join(TMP, "wa_pool.json"))
    out["baseline_note"] = ("Baseline A is absent: its per-checkpoint scores before step "
                            "8000 are not stored and no head is loaded in this pass; its "
                            "window rows stay on the pre-registered pool.")
    _write(out, name)


# ------------------------------------------------------------------ A4 + venue
def aug_precision(force):
    """``scripts/decompose_aug_precision.main`` on the pool, plus the venue difference.

    The venue difference is ``scripts/report_screen_unfreeze``'s paired bootstrap (Kaggle
    cell C minus local cell C, B = 2000, seed 0), recomputed here from the two stored score
    dumps because that script's report mixes many other runs; the pre-registered-pool
    value must reproduce ``runs/lever1_screen_report.json``'s ``venue_portability``.
    """
    name = f"lever1_aug_precision_{C.POOL}.json"
    if not _need(name, force):
        return
    import decompose_aug_precision as DAP
    from metrics import paired_bootstrap, default_tpr_metric, tpr_at_fpr
    raw = DAP.load

    def pooled(run_dir):
        d = raw(run_dir)
        if d is None:
            return d
        keep = (C.pool_keep_neg(d["neg_ids"]) if C.POOL != "orig"
                else np.ones(d["neg_ids"].size, bool))
        d["pos"], d["neg"] = C.pool_filter(d["pos"], d["neg"], d["pos_ids"], d["neg_ids"],
                                           where=f"A4 {run_dir}")
        d["neg_ids"] = d["neg_ids"][keep]
        return d

    def venue():
        loc, kag = pooled(DAP.CELLS["C_weak_fp16"][0]), pooled("baselineA_fold0_nb0_image_kaggle")
        assert np.array_equal(loc["neg_ids"], kag["neg_ids"])
        assert np.array_equal(loc["pos_ids"], kag["pos_ids"])
        y = np.concatenate([np.ones(loc["pos"].size, np.int64),
                            np.zeros(loc["neg"].size, np.int64)])
        s_loc = np.concatenate([loc["pos"], loc["neg"]])
        s_kag = np.concatenate([kag["pos"], kag["neg"]])
        t_loc, t_kag = tpr_at_fpr(y, s_loc, 1e-3), tpr_at_fpr(y, s_kag, 1e-3)
        bs = paired_bootstrap(y, s_kag, s_loc, metric_fn=default_tpr_metric(1e-3),
                              B=2000, seed=0)
        one = 1.0 / loc["pos"].size
        return {"tpr_local": float(t_loc), "tpr_kaggle": float(t_kag),
                "diff": float(t_kag - t_loc), "lenses": abs(t_kag - t_loc) / one,
                "ci": [float(bs["lo"]), float(bs["hi"])],
                "exceeds_one_lens": bool(abs(t_kag - t_loc) > one)}

    DAP.load = pooled
    try:
        with C.pool("orig"):
            DAP.main(["--out", os.path.join(TMP, "ap_orig.json")])
            v0 = venue()
        a, b = _load(os.path.join(TMP, "ap_orig.json")), _load(C.R("runs",
                                                                   "lever1_aug_precision.json"))
        _close(a, b, "lever1_aug_precision(orig)")
        vb = _load(C.R("runs", "lever1_screen_report.json"))["venue_portability"]
        _close(v0, vb, "venue_portability(orig)", tol=1e-12)
        print("  aug_precision + venue: the pre-registered pool reproduces "
              "runs/lever1_aug_precision.json and lever1_screen_report.json")
        DAP.main(["--out", os.path.join(TMP, "ap_pool.json")])
        v1 = venue()
    finally:
        DAP.load = raw
    out = _load(os.path.join(TMP, "ap_pool.json"))
    out["venue_portability"] = v1
    _write(out, name)


# ------------------------------------------------------------------ fold-0 window analysis
def window_fold0(force):
    """``scripts/analyze_window_fold0.main`` (7.16's eighth entry and row 9) on the pool.

    Its ratios divide by ``DANN_MINUS_CORAL``, the between-method difference the results
    table is asked to resolve. The script types it as 0.018, the rounded pre-registered
    value of |DANN - CORAL| at declared budgets; on another pool it is set to that pool's
    own |DANN - CORAL|, computed by ``_common.contrast`` exactly as Table 2 computes it.
    """
    name = f"window_fold0_{C.POOL}.json"
    if not _need(name, force):
        return
    import analyze_window_fold0 as AWF
    typed = AWF.DANN_MINUS_CORAL
    with C.pool("orig"):
        assert round(abs(C.contrast("dann", 4000, "coral", 2000)["mean"]), 3) == typed
        AWF.main(["--out", os.path.join(TMP, "wf0_orig.json")])
    a, b = _load(os.path.join(TMP, "wf0_orig.json")), _load(C.ART["window_fold0"])
    for key in ("arms", "row9_stability", "adjacent_swing"):
        _close(a[key], b[key], f"window_fold0(orig).{key}")
    print("  window_fold0: the pre-registered pool reproduces runs/round7/window_fold0.json")
    denom = abs(C.contrast("dann", 4000, "coral", 2000)["mean"])
    AWF.DANN_MINUS_CORAL = denom
    try:
        AWF.main(["--out", os.path.join(TMP, "wf0_pool.json")])
    finally:
        AWF.DANN_MINUS_CORAL = typed
    out = _load(os.path.join(TMP, "wf0_pool.json"))
    out["dann_minus_coral_denominator"] = {
        "value": denom, "source": "|DANN(4000) - CORAL(2000)| fold-mean TPR, this pool",
        "pre_registered_pool_value_as_typed_in_the_script": typed}
    _write(out, name)


# ------------------------------------------------------------------ fold-0 controlled pair
def fold0_curves(force):
    """Figure 4's two fold-0 curves (DANN arm 1, lambda = 0; arm 2, lambda live), per step.

    The stored curves (``runs/dann_r5_curves/*.csv``) hold metrics only. The rescored
    per-checkpoint logits behind them (``runs/dann_arm2_fold0/rescore``) carry ids, so the
    curves are recomputed from those logits; on the pre-registered pool they must equal the
    CSVs (which store TPR rounded to 1e-8) at every one of the 32 steps. The reference line
    is Baseline A's fold-0 value on the same pool.
    """
    import csv as _csv
    import analyze_window_fold0 as AWF
    import metrics as me
    name = f"fold0_curves_{C.POOL}.json"
    if not _need(name, force):
        return
    arms = {"arm1_lambda0": ("DANN arm 1 (lambda=0)", "dann_arm1_curve"),
            "arm2_dann": ("DANN arm 2 (lambda live)", "dann_arm2_curve")}

    def curves():
        out = {}
        for key, (arm, _) in arms.items():
            sc = AWF._scan_dir(AWF.ARMS_F0[arm], 0)
            out[key] = {}
            for s in sorted(sc):
                pos, neg = sc[s]
                y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
                out[key][str(s)] = float(me.tpr_at_fpr(y, np.concatenate([pos, neg]), 1e-3))
        return out

    with C.pool("orig"):
        orig = curves()
    for key, (_, art) in arms.items():
        rows = list(_csv.DictReader(open(C.ART[art], newline="", encoding="utf-8")))
        assert [r["step"] for r in rows] == list(orig[key]), f"{key}: step grid differs"
        for r in rows:
            assert abs(float(r["tpr_at_1e-3"]) - orig[key][r["step"]]) < 1e-8, (key, r["step"])
    print("  fold0_curves: the pre-registered pool reproduces runs/dann_r5_curves/*.csv "
          "at all steps")
    out = {"what": "fold-0 TPR@1e-3 per checkpoint, DANN arm 1 (lambda=0) and arm 2",
           "curves": curves(),
           "init_baseline_fold0": float(C.suite("baseline", 0, 8000)["tpr"])}
    _write(out, name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["lines_ratios", "foldmean_bootstrap",
                                       "round1_paired", "window_average", "aug_precision",
                                       "window_fold0", "fold0_curves"])
    a = ap.parse_args()
    if C.POOL == "orig":
        print("  PAPER_POOL=orig: the project's own artifacts are the pre-registered values;"
              " nothing to build")
        return 0
    os.makedirs(DATA, exist_ok=True)
    steps = [("lines_ratios", lines_ratios), ("foldmean_bootstrap", foldmean_bootstrap),
             ("round1_paired", round1_paired), ("window_average", window_average),
             ("aug_precision", aug_precision), ("window_fold0", window_fold0),
             ("fold0_curves", fold0_curves)]
    try:
        for key, fn in steps:
            if a.only in (None, key):
                print(f"\n--- {key} ({C.POOL}) ---")
                fn(a.force)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
