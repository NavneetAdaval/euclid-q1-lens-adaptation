"""
test_dilute10.py
================
Tests for the 1:10 diluted-target code (brief_dilute10_code.md section 7). CPU only; no
training; no test label read (no metric is computed anywhere in this file).

    python -u scripts/test_dilute10.py --quick --out <report.json>   # T2-T10
    python -u scripts/test_dilute10.py --heavy --out <report.json>   # T1 + RNG check + diffs

T1  known-null, per arm x folds 1-4: original+balanced emits a config equal to the original
    run's field for field (named exemptions only) and the first 20 target and source batch
    index sequences equal the ORIGINAL construction's, built in-process
T2  uniform sampler (20 000 batches, fold 1 DANN training pool) + AST label scan
T3  pool loader: tampered pool refused; existing members keep their positions
T4  holdout: the 78 objects equal the original runs' recorded holdout, per fold
T5  FreeMatch SAT rebuild uses the uniform diluted loader; batch count (part 1: 680)
T6  gate wiring: the nb3 runners refuse without gate_values.json / on a head-hash mismatch
T7  lens-fraction logger never reaches train_step (known-null synthetic runner caught)
T8  no edits: every rule-zero module hashes as it did before this brief
T9  output safety: each runner refuses an existing output directory without --force
T10 integrity --build: data/splits discovered; old scope misses the pool (known-null)
T11 every DataLoader runner refuses --num-workers 0
T12 fold identity (frozen stages), with a known-null
T13 lens-fraction stop injected into every runner (--t13)
T13b (--t13) the band is enforced at each 187-batch check and at the end of a run that
    COMPLETED its budget; an aborted run's summary is written unenforced, without raising
    (PROJECTv7 7.47)
T14 part 2 provenance (bundle v2): provenance() refuses a missing or altered part 2 (incl. a
    re-self-certified forgery) and records the pinned digest otherwise; bundle_provenance() is
    null-with-reason locally, records version + manifest from stage.json, and refuses a
    stage.json without them, with another part 2, or whose manifest no longer hashes
T15 CORAL nb3's inherited feature-norm guard is log-only through the wrapper (known-null:
    the original raises on the Kaggle fold-1 step-1 values) (PROJECTv7 7.47)
T16 part 3 (PROJECTv7 7.48): the two nb3 arms refuse a missing, altered or forged part 3
    (directly, through provenance() and through their own --emit-config); other arms
    record it if present and never require it; stage.json's part 3 is checked
CFGID config identity across the v2 provenance change (--cfgid): every runner's emitted
    config, folds 1-4, dilute10 + uniform, equals the PRE-CHANGE snapshot field for field,
    except the new provenance fields, recorded_utc, started_utc and dilute10_common.py's own
    module hash; and every emitted config carries the pinned part2_sha256
RNG (section 2): source batch indices for 2000 steps, original vs dilute10+uniform, per arm
DIFF (section 7): fold 1, dilute10+uniform vs original, per arm -- allowed keys only
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402
import dilute10_pool as DP                                   # noqa: E402

FOLDS = (1, 2, 3, 4)

# rule zero: recorded at the start of brief_dilute10_code (2026-10-13), before any edit
FROZEN_MODULES = {
    "samplers.py": "32753fe586e077de2069070d77a795ea30210b5e8fc8d85b852e6c2b84fac96e",
    "lens_dataset.py": "87c3701b4eb8e76333b06938d3d3fda8a53e7f46afe59bbb7fee6074b566318d",
    "train_skeleton.py": "06834cac81d1fdcd9a560d508a080a51902ca11b33e4cc648da9a4e76dbc3691",
    "da_losses.py": "2d861ffc8649f4987756f5773c9e72e8bb2fba5cf790c7b227e7436ce1f89a98",
    "run_dann_r5.py": "320ed27d74473ce1b19cd3ee3f27375b2af3cd3ce58a748228116048302e4515",
    "dann_round5.py": "6884d4690aa1d58aaff876b1624de259028965a47e7a38a9437a8e477eff8c1e",
    "run_bnm_r6.py": "5cb8b3f8a0f22d2089f738f2cf5a0e4d0756c6fe4661328e55d3594e9f1f0220",
    "bnm_round6.py": "f1c71a7f6a70e80eadf89d137369c741dcceb313d6370eede70242b749241f05",
    "run_freematch.py": "8262b7cf37fe0c6aeb0ce8b7ed657a75175ff7e1dca471548b41d6a18708150e",
    "run_unfreeze_freematch.py": "e8f745890dbf4b349f1254f4a898d72cb33909108ca0f495c078df17823981b8",
    "continuation_state.py": "7b1ac7516f907137fc3d82d1ade3dab0dafd15f440d115faf69b71b5163e1033",
    "run_coral.py": "2a8821d38ab770ea4face2fbc323b5c0c3e0a8de84d770366575d5aeea2e8cb3",
    "run_coral_round4.py": "2dac8d4f8dbe4018ac3f987426bff180c50102c9ceaf5db7c17d99b75eb200d8",
    "run_bakeoff_adamatch.py": "b5e22a158818d79ad083ba799d1a2d9ad4a81cd2d68a573ccbb71b25890a9932",
    "run_bakeoff_baseline.py": "3dcb52cf5f0af9ee96595b6f71fb6674ddf31dc48c64f71f53526c91e1ec67c9",
    "run_bakeoff_coral.py": "9a10e548b3178b637de6e2bc849a314ad2064710e78a7da49772840223a08f82",
    "run_bakeoff_freematch.py": "6e35b3692cd1ce1fab48a12a96571bd1fefbe451fd96c8691ddee616f6ac16d6",
    # ADDED 2026-10-13 (brief_dilute10_part2 section A, PROJECTv7.md 7.38): restored to the
    # version all four CORAL x nb3 confirmatory folds recorded (2853805195bd88bf, copied byte
    # for byte from kaggle/coral_r4/dataset/scripts/). The half-edited file it replaced
    # (0ec4f3c6...) is kept in audit/coral_confirm_halfedit/.
    "run_coral_r4_confirm.py": "2853805195bd88bf4c1f4df8e32a4ff16458ba2845844ff17a4eea2323a7bd2d",
}
NEW_RUNNERS = ("run_dilute10_dann.py", "run_dilute10_bnm.py",
               "run_dilute10_freematch_frozen.py", "run_dilute10_freematch_nb3.py",
               "run_dilute10_coral_frozen.py", "run_dilute10_coral_nb3.py")

# ---- T1 exemptions, each named with its reason ----
EXEMPT_COMMON = {
    "dilute10": "the new provenance block",
    "device": "venue: the original ran on a Kaggle T4, this emit runs on the CPU",
    "init_head_from": "path (Kaggle mount vs local)",
    "init_head_from_tail": "path tail (Kaggle dataset slug vs local repo name)",
    "agreement_ref": "path",
    "paths": "paths", "out_dir": "path", "run_dir": "path (runs/dilute10/...)",
    "started_utc": "timestamp", "torch_version": "venue",
}
EXEMPT_ARM = {
    "bnm": {"gamma_invariant": "the stored copy is updated at run time with observations"},
    "coral_nb3": {
        "config_diff": "reference path to the fold-0 development run.json",
        "pre_registration": "path to runs/coral_r4_budget.json (digest identical)",
        "script_sha256_16": ("da_losses.py (BNM appended, append-only, 7.21); "
                             "run_coral_r4_confirm.py now matches (restored, 7.38)"),
        "steps_completed": "run-time counter", "fp16_skipped_cumulative": "run-time counter"},
    "freematch_frozen": {"folds_this_invocation": "invocation list (round 1: [0]; here [k])"},
    "coral_frozen": {"folds_this_invocation": "invocation list (round 1: [1,2,3,4]; here [k])"},
    "freematch_nb3": {},
}
# FreeMatch x nb3 fold 1 ran before the 2026-08-06 gate change (part 1 disclosure)
DECLARED_EXCEPTIONS = {
    ("freematch_nb3", 1): {
        "step0_tpr_tolerance": "fold 1 ran at tolerance 0 before the 2026-08-06 change",
        "step0_auroc_tolerance": "fold 1 ran with no AUROC half",
        "expect_step0_auroc": "fold 1 ran with no AUROC half"},
}

STORED = {
    "dann": lambda k: f"runs/dann_main/fold{k}/extracted/fold{k}/run_config.json",
    "bnm": lambda k: f"runs/bnm_r6_folds/bnm_r6_fold{k}/run_config.json",
    "coral_nb3": lambda k: f"kaggle/coral_r4/runs/main/extracted/coral_r4_confirm/fold{k}/run.json",
    "freematch_nb3": lambda k: f"runs/freematch_nb3_unfrozen_fold{k}/run_config.json",
    "freematch_frozen": lambda k: "runs/bakeoff_freematch_5fold/run_config.json",
    "coral_frozen": lambda k: "runs/bakeoff_coral/run_config.json",
}
ALLOWED_DILUTE_DIFF = {"dilute10"}               # + init_head_from(_tail/_l2) for nb3 arms


def ok(name, cond, detail=None):
    return {"test": name, "pass": bool(cond), "detail": detail}


def ns(fold, pool="original", sampler="balanced", **kw):
    p = C.add_common_args(argparse.ArgumentParser(), num_workers_default=kw.pop("nw", 2))
    p.add_argument("--init-head-from", default=None)
    p.add_argument("--frozen-dir", default=None)
    p.add_argument("--splits", default=os.path.join(REPO, "data", "splits", "splits.json"))
    p.add_argument("--data-root", default=os.path.join(REPO, "data"))
    p.add_argument("--cache-dir", default=os.path.join(REPO, "data", "feature_cache"))
    p.add_argument("--dilute-cache-dir",
                   default=os.path.join(REPO, "data", "feature_cache_dilute10"))
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    a = p.parse_args(["--fold", str(fold), "--target-pool", pool, "--target-sampler",
                      sampler, "--device", "cpu"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# ======================================================================================
# QUICK TESTS
# ======================================================================================
def t2_uniform_sampler():
    from uniform_target_sampler import UniformTargetBatchSampler, label_scan
    pool = DP.load_pool()
    h = DP.pinned_holdout(pool, 1)
    tp = h["train_pool"]
    n_l = int(DP.pool_fold(pool, 1)["n_lenses"])
    s = UniformTargetBatchSampler(tp, 256, 20000, seed=42 + 1 + 1000)
    batches = [np.asarray(b) for b in s]
    allidx = np.concatenate(batches)
    held = set(int(i) for i in h["holdout"])
    sizes = {b.size for b in batches}
    dup = sum(len(set(b.tolist())) != b.size for b in batches)
    N = tp.size
    n_full = allidx.size // N
    per_pass_equal = True
    for p_ in range(n_full):
        seg = allidx[p_ * N:(p_ + 1) * N]
        if not np.array_equal(np.sort(seg), np.sort(tp)):
            per_pass_equal = False
            break
    fr = np.array([(b < n_l).mean() for b in batches])
    p = 352 / 4223
    sd_hyp = np.sqrt(256 * p * (1 - p) * (N - 256) / (N - 1)) / 256
    scan = label_scan()
    det = {"n_batches": len(batches), "batch_sizes": sorted(sizes),
           "holdout_drawn": int(sum(int(i) in held for i in allidx)),
           "dup_within_batch": int(dup), "complete_passes_checked": int(n_full),
           "every_member_once_per_pass": per_pass_equal,
           "training_pool": f"{int((tp < n_l).sum())}:{int((tp >= n_l).sum())}",
           "mean_lens_fraction": float(fr.mean()), "target_mean": p,
           "sd_lens_fraction": float(fr.std(ddof=1)), "sd_hypergeometric": float(sd_hyp),
           "boundary_swaps": s.n_boundary_swaps, "label_scan": scan}
    cond = (sizes == {256} and det["holdout_drawn"] == 0 and dup == 0 and per_pass_equal
            and abs(fr.mean() - p) <= 0.002 and abs(fr.std(ddof=1) / sd_hyp - 1) <= 0.10
            and scan["clean"] and scan["known_null_hits"])
    return ok("T2 uniform sampler", cond, det)


def t3_pool_loader():
    det = {}
    tmp = tempfile.mkdtemp()
    try:
        bad = os.path.join(tmp, "pool.json")
        b = bytearray(open(DP.POOL_PATH, "rb").read())
        i = b.index(b'"seed": 20261013')
        b[i + len('"seed": 2026101')] = ord("4")            # 20261013 -> 20261014
        open(bad, "wb").write(bytes(b))
        try:
            DP.load_pool(bad)
            det["tampered_refused"] = False
        except DP.PoolVerificationError as e:
            det["tampered_refused"] = True
            det["tampered_message"] = str(e)[:120]
        det["genuine_accepted"] = DP.load_pool()["_verified"]["pool_sha256"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pool = DP.load_pool()
    sp = DP.load_splits()
    pos_ok = {}
    for k in FOLDS:
        d = DP.diluted_splits(sp, pool, k)
        o = next(f for f in sp["real_lenses_kfold"]["folds"] if f["fold"] == k)
        n = next(f for f in d["real_lenses_kfold"]["folds"] if f["fold"] == k)
        n_ex = len(o["target_neg_object_ids"])
        pos_ok[str(k)] = (n["target_lens_rows"] == o["target_lens_rows"]
                          and n["target_neg_object_ids"][:n_ex] == o["target_neg_object_ids"]
                          and n["target_neg_locations"][:n_ex] == o["target_neg_locations"]
                          and len(n["target_neg_object_ids"]) == 10 * len(n["target_lens_rows"]))
    # dataset-level, fold 1: the served descriptors of the original pool are a prefix
    from lens_dataset import LensDataset
    od = LensDataset(sp, os.path.join(REPO, "data"), role="target", fold=1)
    nd = DP.lens_dataset_factory(LensDataset, pool, 1)(DP.SPLITS_PATH, os.path.join(
        REPO, "data"), role="target", fold=1)
    pref = all(od._descriptors[i]["ref"][1:] == nd._descriptors[i]["ref"][1:]
               and od._descriptors[i]["kind"] == nd._descriptors[i]["kind"]
               for i in range(len(od)))
    img = bool(np.array_equal(od.images, nd.images[:len(od)]))
    det.update({"splits_level_positions_kept": pos_ok, "dataset_prefix_fold1": pref,
                "dataset_prefix_pixels_fold1": img, "n_original": len(od), "n_diluted": len(nd)})
    return ok("T3 pool loader", det["tampered_refused"] and all(pos_ok.values()) and pref
              and img, det)


def t4_holdout():
    pool = DP.load_pool()
    det, good = {}, True
    for k in FOLDS:
        h = DP.pinned_holdout(pool, k)
        rec, _ = DP.recorded_holdout_positions(k)
        sp = DP.load_splits()
        o = next(f for f in sp["real_lenses_kfold"]["folds"] if f["fold"] == k)
        ident_o = [DP._identity(o, p) for p in sorted(rec)]
        ident_n = [DP._identity(DP.pool_fold(pool, k), int(p)) for p in h["holdout"]]
        same = json.dumps(ident_o).encode() == json.dumps(ident_n).encode()
        want = "353:3881" if k == 4 else "352:3871"
        det[str(k)] = {"n_holdout": h["n_holdout"], "objects_byte_identical": same,
                       "training_pool": h["training_pool_counts"], "expected": want,
                       "identity_sha256": h["holdout_identity_sha256"]}
        good &= same and h["n_holdout"] == 78 and h["training_pool_counts"] == want
    return ok("T4 holdout", good, det)


def t5_sat_rebuild():
    import torch
    import continuation_state as cs
    import run_freematch as RF
    import run_unfreeze_freematch as R
    import run_dilute10_freematch_nb3 as W
    from uniform_target_sampler import UniformTargetBatchSampler

    class FM:
        sat_ema_momentum = 0.999
        tau_global = torch.tensor(0.5, dtype=torch.float64)
        pbar = torch.tensor([0.5, 0.5], dtype=torch.float64)
        fairness_target = torch.tensor([0.5, 0.5], dtype=torch.float64)
        last_tau_global = 0.5

    class Trainer:
        freematch = FM()
        _amp_enabled = False

        class cfg:
            batch_size = 256

    drawn = {}

    def fake_collect(trainer, target_loader, n_batches, verbose=True):
        it, seen = iter(target_loader.batch_sampler), []
        for _ in range(int(n_batches)):
            try:
                b = next(it)
            except StopIteration:
                it = iter(target_loader.batch_sampler)
                b = next(it)
            seen.append(len(b))
        drawn["n"] = len(seen)
        rng = np.random.default_rng(0)
        bmax = 0.9 + 0.01 * rng.random(int(n_batches))
        bp = np.tile([0.6, 0.4], (int(n_batches), 1))
        return bmax, bp

    det = {}
    pool = DP.load_pool()
    from lens_dataset import LensDataset
    ds = DP.lens_dataset_factory(LensDataset, pool, 1)(
        DP.SPLITS_PATH, os.path.join(REPO, "data"), role="target", fold=1, aug="weak",
        n_views=2)
    uni = C.loader_ctor(RF.make_balanced_loader, uniform=True)(
        ds, 256, use_latent=True, num_batches=187, seed=42 + 1 + 1000)
    bal = RF.make_balanced_loader(ds, 256, use_latent=True, num_batches=187,
                                  seed=42 + 1 + 1000)
    args = ns(1, "dilute10", "uniform", nw=4)
    with C.PatchSet() as ps0:
        ps0.set(cs, "collect_target_batch_stats", fake_collect, "test stub")
        sat = {}
        with C.PatchSet() as ps:
            # the wrapper's rebuild (FOLD_GATE untouched: no init-head given, pool dilute10
            # -> it would look for the frozen head; give it a stand-in frozen dir)
            ps.set(W, "resolve_head", lambda a: None)
            W.patched(args, R, ps, sat_record=sat)
            rep = cs.rebuild_sat_buffers(Trainer(), uni, n_passes=40, pool_size=len(ds),
                                         batch_size=256, ema_iters=50)
            det["uniform"] = {"n_batches": rep["n_batches"], "pool_size": rep["pool_size"],
                              "drawn": drawn["n"], "sampler": sat.get("sampler"),
                              "is_uniform": isinstance(uni.batch_sampler,
                                                       UniformTargetBatchSampler)}
            try:
                cs.rebuild_sat_buffers(Trainer(), bal, n_passes=40, pool_size=len(ds),
                                       batch_size=256, ema_iters=50)
                det["known_null_balanced_refused"] = False
            except AssertionError as e:
                det["known_null_balanced_refused"] = True
                det["known_null_message"] = str(e)
    good = (det["uniform"]["n_batches"] == 680 and det["uniform"]["drawn"] == 680
            and det["uniform"]["pool_size"] == 4301 and det["uniform"]["is_uniform"]
            and det["known_null_balanced_refused"])
    return ok("T5 FreeMatch SAT rebuild", good, det)


def t6_gate_wiring():
    import run_coral_r4_confirm as CF
    import run_coral_round4 as R4
    import run_unfreeze_freematch as R
    import run_dilute10_coral_nb3 as WC
    import run_dilute10_freematch_nb3 as WF
    det = {}
    tmp = tempfile.mkdtemp()
    try:
        head_src = os.path.join(REPO, "runs", "bakeoff_freematch_5fold", "fold1",
                                "checkpoints", "step_8000.pt")
        head = os.path.join(tmp, "step_8000.pt")
        shutil.copyfile(head_src, head)
        for name, fn in (("freematch_nb3", lambda a, ps: WF.patched(a, R, ps, sat_record={})),
                         ("coral_nb3", lambda a, ps: WC.patched(a, CF, R4, ps))):
            res = {}
            a = ns(1, "dilute10", "uniform", init_head_from=head, frozen_dir=tmp)
            for case in ("missing", "mismatch", "match"):
                gp = os.path.join(tmp, C.GATE_FILE)
                if os.path.exists(gp):
                    os.remove(gp)
                if case != "missing":
                    sha = DP.sha256_file(head) if case == "match" else "0" * 64
                    json.dump({"fold": 1, "head_sha256": sha, "tpr": float("nan"),
                               "auroc": float("nan"), "head_l2": 0.0}, open(gp, "w"))
                try:
                    with C.PatchSet() as ps:
                        fn(a, ps)
                    res[case] = "accepted"
                except SystemExit as e:
                    res[case] = "refused: " + str(e)[:90]
            det[name] = res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    good = all(r["missing"].startswith("refused") and r["mismatch"].startswith("refused")
               and r["match"] == "accepted" for r in det.values())
    return ok("T6 gate wiring", good, det)


def t7_logger():
    from dilute10_batch_log import LensFractionLogger, scan_runner
    from uniform_target_sampler import UniformTargetBatchSampler
    det = {}
    for f in NEW_RUNNERS + ("dilute10_common.py",):
        det[f] = scan_runner(open(os.path.join(HERE, f), encoding="utf-8").read())
    inner = UniformTargetBatchSampler(np.arange(4223), 256, 50, seed=7)
    ref = [list(b) for b in UniformTargetBatchSampler(np.arange(4223), 256, 50, seed=7)]
    got = [list(b) for b in LensFractionLogger(inner, 352, None)]
    det["logger_yields_inner_batches_unchanged"] = got == ref
    good = all(v["clean"] and v["known_null_hits"] for k, v in det.items()
               if isinstance(v, dict)) and det["logger_yields_inner_batches_unchanged"]
    return ok("T7 lens-fraction logger", good, det)


def t8_no_edits():
    now = {f: DP.sha256_file(os.path.join(HERE, f)) for f in FROZEN_MODULES}
    bad = {f: (FROZEN_MODULES[f], now[f]) for f in now if now[f] != FROZEN_MODULES[f]}
    return ok("T8 no edits", not bad, {"n_checked": len(now), "changed": bad})


def t9_output_safety():
    det = {}
    tmp = tempfile.mkdtemp()
    try:
        for f in NEW_RUNNERS:
            arm = f[len("run_dilute10_"):-3]
            os.makedirs(os.path.join(tmp, arm, "fold1"), exist_ok=True)
            mod = __import__(f[:-3])
            try:
                mod.main(["--fold", "1", "--out-root", tmp, "--device", "cpu"])
                det[arm] = "NOT refused"
            except SystemExit as e:
                det[arm] = ("refused" if "REFUSING" in str(e) else f"other exit: {e}")[:120]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok("T9 output safety", all(v == "refused" for v in det.values()), det)


#: T10's only allowed extra discoveries (PROJECTv7.md 7.44). window_prereg.json is hashed in the
#: record but deliberately NOT enrolled (7.34), so it is required. protected_artifacts.json
#: cannot enrol itself (its hash changes with every enrolment): allowed, not required.
T10_REQUIRED_EXTRA = ("runs/round7/window_prereg.json",)
T10_ALLOWED_EXTRA = ("runs/protected_artifacts.json",)


def t10_extras_ok(extra_paths):
    paths = sorted(extra_paths)
    return (all(p in paths for p in T10_REQUIRED_EXTRA)
            and all(p in T10_REQUIRED_EXTRA + T10_ALLOWED_EXTRA for p in paths)
            and len(paths) == len(set(paths)))


def t10_integrity_build():
    import subprocess
    import integrity
    live_path = os.path.join(REPO, "runs", "protected_artifacts.json")
    before = DP.sha256_file(live_path)
    tmp = tempfile.mkdtemp()
    try:
        dst = os.path.join(tmp, "inv.json")
        subprocess.run([sys.executable, os.path.join(HERE, "integrity.py"), "--build",
                        "--out", dst], check=True, capture_output=True)
        reb = json.load(open(dst))["artifacts"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    live = json.load(open(live_path))["artifacts"]
    key = lambda e: (e["path"], e["flavour"], e["sha256"])         # noqa: E731
    extra = sorted(set(map(key, reb)) - set(map(key, live)))
    missing = sorted(set(map(key, live)) - set(map(key, reb)))
    old = integrity.protected_files(scan_dirs=("runs",))
    det = {"live_n": len(live), "rebuilt_n": len(reb),
           "in_rebuilt_not_live": extra, "in_live_not_rebuilt": missing,
           "known_null_old_scope_misses_pool":
               "data/splits/dilute10_pool.json" not in [p for p, _, _ in old],
           "live_inventory_unchanged": DP.sha256_file(live_path) == before,
           "note": ("runs/round7/window_prereg.json is hashed in the record but deliberately "
                    "NOT enrolled (PROJECTv7.md 7.34); any rebuild that scans runs/ finds it. "
                    "runs/protected_artifacts.json is allowed, not required (7.44)")}
    extra_paths = [e[0] for e in extra]
    det["known_null_third_extra_fails"] = not t10_extras_ok(
        extra_paths + ["runs/dilute10/unexpected_known_null.json"])
    det["known_null_window_prereg_required"] = not t10_extras_ok(
        [p for p in extra_paths if p not in T10_REQUIRED_EXTRA])
    good = (not missing and t10_extras_ok(extra_paths)
            and det["known_null_third_extra_fails"] and det["known_null_window_prereg_required"]
            and det["known_null_old_scope_misses_pool"] and det["live_inventory_unchanged"])
    return ok("T10 integrity build", good, det)


def t11_refuse_zero_workers():
    """Part 2 D.7: every DataLoader-based runner refuses --num-workers 0."""
    det = {}
    for f in ("run_dilute10_dann.py", "run_dilute10_bnm.py",
              "run_dilute10_freematch_frozen.py", "run_dilute10_freematch_nb3.py",
              "run_dilute10_coral_nb3.py"):
        mod = __import__(f[:-3])
        try:
            mod.main(["--fold", "1", "--num-workers", "0", "--device", "cpu",
                      "--emit-config", os.path.join(tempfile.gettempdir(), "never.json")])
            det[f] = "NOT refused"
        except SystemExit as e:
            det[f] = "refused" if "num-workers 0" in str(e) else f"other: {str(e)[:80]}"
    det["run_dilute10_coral_frozen.py"] = "not applicable (no DataLoader; declared in part 2)"
    return ok("T11 refuse num_workers=0",
              all(v == "refused" for k, v in det.items() if "coral_frozen" not in k), det)


def t12_fold_identity():
    """Part 2 D.6: the frozen stages' fold-identity check passes on the real served IDs and
    aborts on a known-null (one target ID altered). Skipped until part 2 exists."""
    if not os.path.isfile(C.PART2_PATH):
        return ok("T12 fold identity", True, {"skipped": "part 2 not written yet"})
    import evaluate as ev
    from cache_features import dataset_ids
    from lens_dataset import LensDataset
    pool = DP.load_pool()
    cache = os.path.join(REPO, "data", "feature_cache")
    det, good = {}, True
    for k in FOLDS:
        ds = DP.lens_dataset_factory(LensDataset, pool, k)(
            DP.SPLITS_PATH, os.path.join(REPO, "data"), role="target", fold=k, aug="weak",
            n_views=2)
        fm = [str(x) for x in dataset_ids(ds)]
        cr = [str(x) for x in ev.FeatureCache(os.path.join(
            REPO, "data", "feature_cache_dilute10")).load(f"target_fold{k}")["ids"]]
        r = {}
        for arm, ids in (("freematch_frozen", fm), ("coral_frozen", cr)):
            try:
                C.check_fold_identity(arm, k, target_ids=ids, cache_dir=cache)
                real = "pass"
            except SystemExit as e:
                real = f"ABORTED: {str(e)[:80]}"
            try:
                C.check_fold_identity(arm, k, target_ids=ids[:-1] + ["tampered"],
                                      cache_dir=cache)
                null = "NOT caught"
            except SystemExit:
                null = "caught"
            r[arm] = {"real_ids": real, "known_null": null}
            good &= real == "pass" and null == "caught"
        det[str(k)] = r
    return ok("T12 fold identity", good, det)


class _Faulty:
    """A target sampler with a deliberate fault, for T13. ``kind``: 'short' yields a
    255-item batch; 'drift' yields well-formed batches whose lens fraction is +0.01 high."""

    def __init__(self, pool, n_lenses, kind, n_batches=400, seed=0):
        self.pool = np.asarray(pool)
        self.batch_size = 256
        self.kind, self.n_batches = kind, n_batches
        self.rng = np.random.default_rng(seed)
        self.lens = self.pool[self.pool < n_lenses]
        self.gal = self.pool[self.pool >= n_lenses]
        self.n_l = int(round(256 * ((self.lens.size / self.pool.size) + 0.01)))

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for i in range(self.n_batches):
            if self.kind == "short":
                yield self.rng.choice(self.pool, 255, replace=False).tolist()
            else:
                yield (self.rng.choice(self.lens, self.n_l, replace=False).tolist()
                       + self.rng.choice(self.gal, 256 - self.n_l, replace=False).tolist())


def _runner_logger(arm, scratch):
    """Build the arm in dilute10+uniform and return ITS lens-fraction logger."""
    from dilute10_batch_log import LensFractionLogger
    if arm in ("dann", "bnm"):
        W = __import__(f"run_dilute10_{arm}")
        A, _ = W.construct(ns(1, "dilute10", "uniform"))
        bs = (A.get("tgt_loader") or A["paired"].target_loader).batch_sampler
    elif arm == "coral_nb3":
        import run_dilute10_coral_nb3 as W
        bs = W.construct(ns(1, "dilute10", "uniform", **_stand_in(arm, 1, scratch)))[
            "loaders"][True].batch_sampler
    elif arm == "freematch_nb3":
        import run_dilute10_freematch_nb3 as W
        bs = W.construct(ns(1, "dilute10", "uniform", nw=4, out_root=scratch,
                            **_stand_in(arm, 1, scratch)))[True].batch_sampler
    else:
        W = __import__(f"run_dilute10_{arm}")
        _cfg, built = W.run(ns(1, "dilute10", "uniform", out_root=scratch), dry=True)
        bs = built[True].batch_sampler if arm == "freematch_frozen" else built[42 + 1 + 1000]
    assert isinstance(bs, LensFractionLogger), type(bs)
    return bs


def t13_lens_fraction_stop(arms=("dann", "bnm", "coral_nb3", "freematch_nb3",
                                  "freematch_frozen", "coral_frozen")):
    """Part 2 stop condition 4, enforced at run time in every runner."""
    from dilute10_batch_log import LensFractionStop
    det = {}
    scratch = tempfile.mkdtemp(prefix="t13_")
    try:
        for arm in arms:
            lg = _runner_logger(arm, scratch)
            pool = lg.inner.pool
            r = {"expected": lg.expected}
            for kind in ("short", "drift"):
                lg.inner = _Faulty(pool, lg.n_lenses, kind)
                lg.fractions, lg._pending = [], []
                try:
                    n = sum(1 for _ in lg)
                    r[kind] = f"NOT stopped after {n} batches"
                except LensFractionStop as e:
                    r[kind] = "stopped: " + str(e).split("]")[0].lstrip("[")
            det[arm] = r
            print(f"[T13] {arm}: {r}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    good = all(v["short"].startswith("stopped: dilute10 STOP: lens-fraction band, malformed")
               and v["drift"].startswith("stopped: dilute10 STOP: lens-fraction band, run-mean")
               for v in det.values())
    return ok("T13 lens-fraction stop", good, det)


def t13b_band_by_completion(arms=("dann", "bnm", "coral_nb3", "freematch_nb3",
                                   "freematch_frozen", "coral_frozen")):
    """T13 extension (PROJECTv7 7.47), on every runner's OWN logger:
    (i)   a +0.01 drift still stops AT THE FIRST 187-batch check;
    (ii)  an aborted run after 5 drifting batches: the run record says aborted at step 1,
          ``run_completion`` says not completed, and ``write_logger_summary`` writes the
          summary with band_enforced false and that reason WITHOUT raising;
    (iii) a completed run (record at its budget) outside the band, 150 drifting batches so no
          187-batch check has run: the end-of-run summary STOPS;
    (iv)  a short batch still stops at once (the per-batch check is unchanged)."""
    from dilute10_batch_log import LensFractionStop
    det = {}
    scratch = tempfile.mkdtemp(prefix="t13b_")
    try:
        for arm in arms:
            lg = _runner_logger(arm, scratch)
            pool, r = lg.inner.pool, {}

            def reset(kind, n):
                lg.inner = _Faulty(pool, lg.n_lenses, kind, n_batches=n)
                lg.fractions, lg._pending = [], []
                lg.csv_path = None

            reset("drift", 400)
            try:
                n = sum(1 for _ in lg)
                r["drift_first_check"] = f"NOT stopped after {n}"
            except LensFractionStop as e:
                r["drift_first_check"] = ("stopped at 187" if "after 187 batches" in str(e)
                                          else f"stopped elsewhere: {e}")
            rec = C.Recorder()
            rec.built.append(("t13b", True, lg))
            d = os.path.join(scratch, f"{arm}_aborted")
            os.makedirs(d)
            json.dump({"steps_completed": 1, "aborted": True,
                       "abort_reason": "synthetic step-1 abort"},
                      open(os.path.join(d, "run.json"), "w"))
            reset("drift", 5)
            list(lg)
            comp, why = C.run_completion(d, budget=2000, steps_json="run.json")
            try:
                s = C.write_logger_summary(rec, d, completed=comp, reason=why)
                r["aborted_5"] = ("summary written, no raise" if s and not s["band_enforced"]
                                  and s["reason"] == why and s["n_batches"] == 5
                                  and os.path.isfile(os.path.join(
                                      d, "target_lens_fraction_summary.json"))
                                  else f"unexpected summary {s}")
            except LensFractionStop as e:
                r["aborted_5"] = f"RAISED: {e}"
            d = os.path.join(scratch, f"{arm}_completed")
            os.makedirs(d)
            json.dump({"steps_completed": 2000, "aborted": False},
                      open(os.path.join(d, "run.json"), "w"))
            reset("drift", 150)
            list(lg)
            comp, why = C.run_completion(d, budget=2000, steps_json="run.json")
            try:
                C.write_logger_summary(rec, d, completed=comp, reason=why)
                r["completed_outside"] = "NOT stopped"
            except LensFractionStop as e:
                r["completed_outside"] = ("stopped at summary" if comp and "after 150 batches"
                                          in str(e) else f"stopped oddly: {e}")
            reset("short", 3)
            try:
                list(lg)
                r["short"] = "NOT stopped"
            except LensFractionStop as e:
                r["short"] = "stopped" if "malformed batch" in str(e) else str(e)
            det[arm] = r
            print(f"[T13b] {arm}: {r}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    good = all(v == {"drift_first_check": "stopped at 187",
                     "aborted_5": "summary written, no raise",
                     "completed_outside": "stopped at summary", "short": "stopped"}
               for v in det.values()) and len(det) == len(arms)
    return ok("T13b band enforced by completion", good, det)


def t15_feature_norm_advisory():
    """CORAL nb3's inherited feature-norm guard is LOG-ONLY through the wrapper (part 2
    stop_rules_nb3_coral_freematch; PROJECTv7 7.47).
    Known-null: the ORIGINAL guard raises on the Kaggle fold-1 step-1 values. The advisory
    guard, on the same values, returns, records the firing (step, src, tgt, band) in its list
    and its CSV; a floor breach is logged the same way; values inside the band log nothing.
    Wiring: train_loop resolves the name in run_coral_r4_dev8000's globals, the wrapper's
    PatchSet rebinds exactly that name and restores the original on exit."""
    import csv as _csv
    import coral_round4 as CR4
    import run_coral_r4_dev8000 as DEV
    import run_dilute10_coral_nb3 as W
    det = {}
    kaggle = {"src_feat_l2": 129.848, "tgt_feat_l2": 149.133}
    try:
        CR4.check_feature_norm_ceiling(1, kaggle, 110.0, 15.349, floor=40.0)
        det["known_null"] = "original did NOT raise"
    except CR4.FeatureNormCeiling:
        det["known_null"] = "original raised"
    scratch = tempfile.mkdtemp(prefix="t15_")
    try:
        csvp = os.path.join(scratch, "run", W.FEATURE_NORM_CSV)
        fir = []
        g = W.advisory_feature_norm_guard(CR4.check_feature_norm_ceiling, csvp, fir)
        r1 = g(1, kaggle, 110.0, 15.349, floor=40.0)
        g(2, {"src_feat_l2": 66.0, "tgt_feat_l2": 70.0}, 110.0, 0.04, floor=40.0)
        g(3, {"src_feat_l2": 30.0, "tgt_feat_l2": 70.0}, 110.0, 0.01, floor=40.0)
        rows = list(_csv.DictReader(open(csvp, newline="")))
        det["advisory"] = {"returned": r1, "n_firings": len(fir),
                           "steps": [f["step"] for f in fir],
                           "kinds": [f["kind"] for f in fir],
                           "csv_rows": [(x["step"], x["kind"], x["src_feat_l2"],
                                         x["tgt_feat_l2"]) for x in rows]}
        det["summary"] = W.feature_norm_summary(fir)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    names = DEV.train_loop.__code__.co_names
    det["train_loop_resolves_global"] = ("check_feature_norm_ceiling" in names
                                         and DEV.train_loop.__globals__ is vars(DEV))
    import run_coral_r4_confirm as CF
    import run_coral_round4 as R4
    before = DEV.check_feature_norm_ceiling
    args = ns(1, "dilute10", "uniform", **_stand_in("coral_nb3", 1, tempfile.mkdtemp(
        prefix="t15s_")))
    with C.PatchSet() as ps:
        W.patched(args, CF, R4, ps, firings=[])
        inside = DEV.check_feature_norm_ceiling
        det["rebound_inside"] = getattr(inside, "__wrapped__", None) is before
        det["rebound_logged"] = any("check_feature_norm_ceiling" in x for x in ps.log)
    det["restored_after"] = DEV.check_feature_norm_ceiling is before
    good = (det["known_null"] == "original raised"
            and det["advisory"]["returned"] is None
            and det["advisory"]["steps"] == [1, 3]
            and det["advisory"]["kinds"] == ["ceiling", "floor"]
            and len(det["advisory"]["csv_rows"]) == 2
            and det["advisory"]["csv_rows"][0][:2] == ("1", "ceiling")
            and det["train_loop_resolves_global"] and det["rebound_inside"]
            and det["rebound_logged"] and det["restored_after"])
    return ok("T15 feature-norm guard advisory", good, det)


def t14_part2_provenance():
    """brief_dilute10_part2_provenance, change 1 / T14. No metric is read."""
    import hashlib as _h
    det, scratch = {}, tempfile.mkdtemp(prefix="t14_")
    orig = C.verify_part2
    try:
        real = C.verify_part2()
        det["real_is_pinned"] = (real == C.PART2_SHA256 ==
                                 "b8e09197a204be835fa089c858b08c7eb3e7887bfa19afd296d544d04f394503")
        d = json.load(open(C.PART2_PATH, encoding="utf-8"))
        alt = dict(d, status=str(d.get("status")) + " (tampered)")
        forged = dict(alt)
        body = {k: v for k, v in forged.items() if k != "sha256_of_body_without_this_field"}
        forged["sha256_of_body_without_this_field"] = _h.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name, content in (("missing", None), ("altered", alt),
                              ("re_self_certified_forgery", forged)):
            path = os.path.join(scratch, f"part2_{name}.json")
            if content is not None:
                json.dump(content, open(path, "w", encoding="utf-8"))
            try:
                C.verify_part2(path)
                det[f"verify_{name}"] = "NOT refused"
            except SystemExit:
                det[f"verify_{name}"] = "refused"
            # ... and through provenance(), the path every runner takes before training
            C.verify_part2 = lambda p=path: orig(p)
            try:
                C.provenance(ns(1), arm="t14")
                det[f"provenance_{name}"] = "NOT refused"
            except SystemExit:
                det[f"provenance_{name}"] = "refused"
            finally:
                C.verify_part2 = orig
        rec = C.provenance(ns(1), arm="t14")
        det["provenance_records_pinned"] = rec.get("part2_sha256") == C.PART2_SHA256
        det["repo_has_no_stage_json"] = not os.path.exists(C.STAGE_JSON)
        det["local_provenance_null_with_reason"] = (rec.get("bundle_version") is None
                                                    and rec.get("bundle_manifest_sha256") is None
                                                    and bool(rec.get("bundle_note")))
        bdir = os.path.join(scratch, "bundle")
        os.makedirs(bdir)
        man = os.path.join(bdir, "MANIFEST.json")
        json.dump({"bundle": "dilute10", "version": 2}, open(man, "w"))
        msha = DP.sha256_file(man)
        good_stage = {"bundle": bdir, "version": 2, "part2": C.PART2_SHA256,
                      "manifest_sha256": msha}
        variants = {"good": good_stage,
                    "no_manifest_sha": {k: v for k, v in good_stage.items()
                                        if k != "manifest_sha256"},
                    "wrong_part2": dict(good_stage, part2="0" * 64),
                    "manifest_changed": dict(good_stage, manifest_sha256="f" * 64)}
        for name, st in variants.items():
            sp = os.path.join(scratch, f"stage_{name}.json")
            json.dump(st, open(sp, "w"))
            try:
                r = C.bundle_provenance(sp)
                det[f"stage_{name}"] = ("recorded" if r["bundle_version"] == 2
                                        and r["bundle_manifest_sha256"] == msha
                                        and r["bundle_manifest_rehashed"] is True
                                        else f"wrong record {r}")
            except SystemExit:
                det[f"stage_{name}"] = "refused"
    finally:
        C.verify_part2 = orig
        shutil.rmtree(scratch, ignore_errors=True)
    good = (det["real_is_pinned"] and det["provenance_records_pinned"]
            and all(det[f"{w}_{n}"] == "refused" for w in ("verify", "provenance")
                    for n in ("missing", "altered", "re_self_certified_forgery"))
            and det["repo_has_no_stage_json"] and det["local_provenance_null_with_reason"]
            and det["stage_good"] == "recorded"
            and all(det[f"stage_{n}"] == "refused"
                    for n in ("no_manifest_sha", "wrong_part2", "manifest_changed")))
    return ok("T14 part 2 provenance", good, det)


def t16_part3_provenance():
    """brief_dilute10_part3 section 2 / T16 (PROJECTv7 7.48). No metric is read.
    The two nb3 arms refuse a missing, an altered or a re-self-certified forged part 3, both
    in ``verify_part3`` and through ``provenance()``; every other arm is unaffected (records
    None, never refuses) and records the pinned digest when part 3 is the real one. A
    stage.json carrying another part 3 is refused, one without part 3 (bundle <= v2) is not.
    Runner level: both nb3 runners' --emit-config refuse with part 3 missing; DANN's does not."""
    import hashlib as _h
    det, scratch = {}, tempfile.mkdtemp(prefix="t16_")
    orig = C.verify_part3
    required = ("coral_nb3", "freematch_nb3")
    others = ("dann_nb3", "bnm_nb3", "coral_frozen", "freematch_frozen")
    try:
        det["real_is_pinned"] = (C.verify_part3("coral_nb3") == C.PART3_SHA256 ==
                                 "1b583ea2ae5056edb410040c32613d9388e5a994bf90fc366fc93d293efb596a")
        d = json.load(open(C.PART3_PATH, encoding="utf-8"))
        alt = dict(d, status=str(d.get("status")) + " (tampered)")
        forged = dict(alt)
        body = {k: v for k, v in forged.items() if k != "sha256_of_body_without_this_field"}
        forged["sha256_of_body_without_this_field"] = _h.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name, content in (("missing", None), ("altered", alt),
                              ("re_self_certified_forgery", forged)):
            path = os.path.join(scratch, f"part3_{name}.json")
            if content is not None:
                json.dump(content, open(path, "w", encoding="utf-8"))
            for arm in required + others:
                try:
                    r = C.verify_part3(arm, path)
                    det[f"verify_{name}_{arm}"] = f"returned {r}"
                except SystemExit:
                    det[f"verify_{name}_{arm}"] = "refused"
                C.verify_part3 = lambda a, p=path: orig(a, p)
                try:
                    rec = C.provenance(ns(1), arm=arm)
                    det[f"provenance_{name}_{arm}"] = f"recorded {rec.get('part3_sha256')}"
                except SystemExit:
                    det[f"provenance_{name}_{arm}"] = "refused"
                finally:
                    C.verify_part3 = orig
        for arm in required + others:
            det[f"real_{arm}"] = C.provenance(ns(1), arm=arm).get("part3_sha256") == \
                C.PART3_SHA256
        # stage.json's part 3 (bundle_provenance)
        bdir = os.path.join(scratch, "bundle")
        os.makedirs(bdir)
        man = os.path.join(bdir, "MANIFEST.json")
        json.dump({"bundle": "dilute10", "version": 3}, open(man, "w"))
        good = {"bundle": bdir, "version": 3, "part2": C.PART2_SHA256,
                "part3": C.PART3_SHA256, "manifest_sha256": DP.sha256_file(man)}
        for name, st in {"stage_good_v3": good,
                         "stage_v2_no_part3": dict(good, part3=None, version=2),
                         "stage_wrong_part3": dict(good, part3="0" * 64)}.items():
            sp = os.path.join(scratch, f"{name}.json")
            json.dump(st, open(sp, "w"))
            try:
                C.bundle_provenance(sp)
                det[name] = "accepted"
            except SystemExit:
                det[name] = "refused"
        # runner level: each runner's own --emit-config, in-process, part 3 missing
        root = os.path.join(scratch, "cfgroot")
        oroot = _cfgid_root_setup(root)
        missing = os.path.join(scratch, "part3_missing.json")
        C.verify_part3 = lambda a, p=missing: orig(a, p)
        try:
            for arm, mod in (("coral_nb3", "run_dilute10_coral_nb3"),
                             ("freematch_nb3", "run_dilute10_freematch_nb3"),
                             ("dann", "run_dilute10_dann")):
                W = __import__(mod)
                emit = os.path.join(scratch, f"emit_{arm}.json")
                try:
                    W.main(["--fold", "1", "--target-pool", "dilute10", "--target-sampler",
                            "uniform", "--device", "cpu", "--out-root", oroot,
                            "--emit-config", emit])
                    got = json.load(open(emit)).get("dilute10", {}).get("part3_sha256", "<absent>")
                    det[f"runner_{arm}_part3_missing"] = f"ran, recorded {got}"
                except SystemExit as e:
                    det[f"runner_{arm}_part3_missing"] = ("refused" if "part 3" in str(e)
                                                          else f"exited otherwise: {e}")
        finally:
            C.verify_part3 = orig
    finally:
        C.verify_part3 = orig
        shutil.rmtree(scratch, ignore_errors=True)
    names = ("missing", "altered", "re_self_certified_forgery")
    good = (det["real_is_pinned"]
            and all(det[f"{w}_{n}_{a}"] == "refused" for w in ("verify", "provenance")
                    for n in names for a in required)
            and all(det[f"verify_{n}_{a}"] == "returned None" for n in names for a in others)
            and all(det[f"provenance_{n}_{a}"] == "recorded None" for n in names
                    for a in others)
            and all(det[f"real_{a}"] for a in required + others)
            and det["stage_good_v3"] == "accepted" and det["stage_v2_no_part3"] == "accepted"
            and det["stage_wrong_part3"] == "refused"
            and det["runner_coral_nb3_part3_missing"] == "refused"
            and det["runner_freematch_nb3_part3_missing"] == "refused"
            and det["runner_dann_part3_missing"] == "ran, recorded None")
    return ok("T16 part 3 provenance", good, det)


# ---- CFGID: config identity across the v2 provenance change (brief change 1) ----
CFGID_SNAPSHOT = os.path.join(REPO, "audit", "dilute10_part2_provenance", "pre_change_configs")
CFGID_NEW_FIELDS = {"part2_sha256", "bundle_version", "bundle_manifest_sha256", "bundle_note",
                    "bundle_manifest_rehashed", "stage_json_sha256"}
# PROJECTv7 7.47 (brief_dilute10_coral_300step section 2): the modules edited there, whose own
# hashes in imported_module_sha256 necessarily change, and the ONE rebinding it adds (CORAL
# nb3's feature-norm guard made log-only). Excluded by name; every other field is compared.
CFGID_EDITED_7_47 = {"dilute10_batch_log.py", "run_dilute10_dann.py", "run_dilute10_bnm.py",
                     "run_dilute10_coral_nb3.py", "run_dilute10_coral_frozen.py",
                     "run_dilute10_freematch_frozen.py", "run_dilute10_freematch_nb3.py"}
CFGID_NEW_REBINDING_7_47 = "run_coral_r4_dev8000.check_feature_norm_ceiling -- "
# PROJECTv7 7.48 (brief_dilute10_part3 section 2): the one new provenance field. The only
# runner-imported module edited there is dilute10_common.py, whose own hash is already
# excluded above.
CFGID_NEW_FIELDS_7_48 = {"part3_sha256"}
CFGID_RUNNERS = {"dann": "run_dilute10_dann.py", "bnm": "run_dilute10_bnm.py",
                 "coral_nb3": "run_dilute10_coral_nb3.py",
                 "freematch_frozen": "run_dilute10_freematch_frozen.py",
                 "freematch_nb3": "run_dilute10_freematch_nb3.py"}


def _cfgid_strip(cfg):
    import re
    c = json.loads(json.dumps(cfg))
    c.pop("started_utc", None)
    # CORAL nb3's --emit-config builds in a FRESH tempfile.mkdtemp(prefix="dilute10_coral_nb3_")
    # (run_dilute10_coral_nb3.py), so out_dir differs between ANY two emits; only the random
    # component is normalised -- the rest of the path is still compared (T1 exempts out_dir)
    if isinstance(c.get("out_dir"), str):
        c["out_dir"] = re.sub(r"dilute10_coral_nb3_[^\\/]+", "dilute10_coral_nb3_<mkdtemp>",
                              c["out_dir"])
    d = c.get("dilute10") or {}
    for k in CFGID_NEW_FIELDS | CFGID_NEW_FIELDS_7_48 | {"recorded_utc"}:
        d.pop(k, None)
    (d.get("imported_module_sha256") or {}).pop("dilute10_common.py", None)
    for m in CFGID_EDITED_7_47:
        (d.get("imported_module_sha256") or {}).pop(m, None)
    if isinstance(d.get("rebound_names"), list):
        d["rebound_names"] = [x for x in d["rebound_names"]
                              if not x.startswith(CFGID_NEW_REBINDING_7_47)]
    return c


def _cfgid_root_setup(root):
    """The frozen-stage inputs the snapshot used, at the snapshot's own path: copies of the
    CORAL frozen heads + gate files, and a FreeMatch-frozen STUB (round 1's head and its
    FOLD_GATE row, read from code)."""
    import run_unfreeze_freematch as RU
    oroot = os.path.join(root, "out")
    for k in FOLDS:
        s = os.path.join(REPO, "runs", "dilute10", "coral_frozen", f"fold{k}")
        d = os.path.join(oroot, "coral_frozen", f"fold{k}")
        os.makedirs(os.path.join(d, "checkpoints"))
        shutil.copyfile(os.path.join(s, "checkpoints", "step_8000.pt"),
                        os.path.join(d, "checkpoints", "step_8000.pt"))
        shutil.copyfile(os.path.join(s, "gate_values.json"), os.path.join(d, "gate_values.json"))
        d = os.path.join(oroot, "freematch_frozen", f"fold{k}")
        os.makedirs(os.path.join(d, "checkpoints"))
        h = os.path.join(d, "checkpoints", "step_8000.pt")
        shutil.copyfile(os.path.join(REPO, "runs", "bakeoff_freematch_5fold", f"fold{k}",
                                     "checkpoints", "step_8000.pt"), h)
        g = RU.FOLD_GATE[k]
        json.dump({"fold": k, "head_sha256": DP.sha256_file(h), "tpr": g["tpr"],
                   "auroc": g["auroc"], "head_l2": g["head_l2"],
                   "STUB": "config-identity check only"},
                  open(os.path.join(d, "gate_values.json"), "w"))
    return oroot


def cfgid(snapshot=CFGID_SNAPSHOT):
    """Emit every runner's config with the CURRENT code at the snapshot's root and compare."""
    import subprocess
    meta = json.load(open(os.path.join(snapshot, "_snapshot.json")))
    root = meta["root"]
    if os.path.exists(root):
        raise SystemExit(f"CFGID: refusing, {root} exists")
    oroot = _cfgid_root_setup(root)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    det = {}
    try:
        for arm, script in CFGID_RUNNERS.items():
            for k in FOLDS:
                key = f"{arm}/fold{k}"
                cfgp = os.path.join(root, f"cfg_{arm}_f{k}.json")
                r = subprocess.run([sys.executable, "-u", f"scripts/{script}", "--fold", str(k),
                                    "--target-pool", "dilute10", "--target-sampler", "uniform",
                                    "--device", "cpu", "--out-root", oroot,
                                    "--emit-config", cfgp],
                                   cwd=REPO, env=env, capture_output=True, text=True)
                if r.returncode != 0 or not os.path.isfile(cfgp):
                    det[key] = {"pass": False, "error": r.stderr[-800:]}
                    print(f"[CFGID] {key}: EMIT FAILED", flush=True)
                    continue
                new = json.load(open(cfgp))
                old = json.load(open(os.path.join(snapshot, f"cfg_{arm}_f{k}.json")))
                a, b = _cfgid_strip(old), _cfgid_strip(new)
                keys = sorted(set(a) | set(b))
                diff = [x for x in keys if a.get(x, "<absent>") != b.get(x, "<absent>")]
                da, db = a.get("dilute10") or {}, b.get("dilute10") or {}
                dd = sorted(x for x in set(da) | set(db) if da.get(x) != db.get(x))
                nb = new.get("dilute10") or {}
                p2 = nb.get("part2_sha256") == C.PART2_SHA256
                det[key] = {"differing_fields": diff, "differing_dilute10_fields": dd,
                            "n_fields": len(keys), "n_dilute10_fields": len(set(da) | set(db)),
                            "new_fields_present": sorted(CFGID_NEW_FIELDS & set(nb)),
                            "part2_sha256_pinned": p2, "pass": not diff and p2}
                print(f"[CFGID] {key}: {det[key]['pass']}", flush=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return ok("CFGID config identity",
              len(det) == 20 and all(v["pass"] for v in det.values()), det)


# ======================================================================================
# HEAVY: T1, the RNG check, the dilute10 config diffs
# ======================================================================================
def _batches(bs, n=20):
    return C.first_batches(bs, n)


def construct_pair(arm, fold, mode, *, scratch, stand_in=None):
    """(cfg, src_batch_sampler, tgt_batch_sampler, sat_pre_batches) for one arm/mode.

    mode: 'wrapper_original' | 'module_original' | 'wrapper_dilute10'
    """
    pool_mode = "dilute10" if mode == "wrapper_dilute10" else "original"
    samp = "uniform" if mode == "wrapper_dilute10" else "balanced"
    if arm in ("dann", "bnm"):
        W = __import__(f"run_dilute10_{arm}")
        a = ns(fold, pool_mode, samp)
        if mode == "module_original":
            if arm == "dann":
                import run_dann_r5 as D
                dargs = D.build_argparser().parse_args(W.base_argv(a))
                A = D.build_arm(dargs.arm, dargs, D.resolve_paths(dargs, REPO), verbose=False)
                cfg = A["cfg_json"]
            else:
                import run_bnm_r6 as B
                bargs = B.build_argparser().parse_args(W.base_argv(a))
                paths = B.resolve_paths(bargs, REPO)
                paths.setdefault("init_head_from", B.DEFAULT_INIT_HEAD)
                A = B.build_arm(bargs, paths, verbose=False)
                cfg = A["cfg"]
        else:
            A, _ = W.construct(a)
            cfg = A["cfg_json"] if arm == "dann" else A["cfg"]
        src = (A["src_loader"] if "src_loader" in A else A["paired"].source_loader).batch_sampler
        tgt = (A["tgt_loader"] if "tgt_loader" in A else A["paired"].target_loader).batch_sampler
        return cfg, src, tgt, 0
    if arm == "coral_nb3":
        import run_coral_r4_confirm as CF
        import run_coral_round4 as R4
        import run_dilute10_coral_nb3 as W
        a = ns(fold, pool_mode, samp, **(stand_in or {}))
        if mode == "module_original":
            cap, err = {}, None
            with C.PatchSet() as ps:
                orig_ba = R4.build_arm
                ps.set(R4, "build_arm", lambda *x, **y: cap.setdefault("A", orig_ba(*x, **y)),
                       "capture only")
                orig_br = CF.build_record
                ps.set(CF, "build_record", lambda *x, **y: cap.setdefault(
                    "rec", orig_br(*x, **y)), "capture only")
                tmp = tempfile.mkdtemp(dir=scratch)
                try:
                    CF.main(W.base_argv(a, None, tmp) + ["--config-diff-only"])
                except TypeError as e:
                    err = f"TypeError: {e}"
            A, cfg = cap["A"], dict(cap["rec"])
            cfg["_on_disk_main_error"] = err
        else:
            r = W.construct(a)
            A, cfg = r["A"], r["record"]
        return cfg, A["src_loader"].batch_sampler, A["tgt_loader"].batch_sampler, 0
    if arm == "freematch_nb3":
        import run_freematch as RF
        import run_unfreeze_freematch as R
        import run_dilute10_freematch_nb3 as W
        a = ns(fold, pool_mode, samp, nw=4, out_root=scratch, **(stand_in or {}))
        cfg_path = os.path.join(scratch, f"fm_nb3_{mode}_{fold}.json")
        if mode == "module_original":
            R.main(W.base_argv(a, None) + ["--emit-config", cfg_path])
            rec = C.Recorder()
            with C.PatchSet() as ps:
                ps.set(RF, "make_balanced_loader", C.loader_ctor(
                    RF.make_balanced_loader, uniform=False, recorder=rec), "record only")
                ps.set(RF, "TrainConfig", lambda *x, **y: (_ for _ in ()).throw(
                    W._Constructed()), "stop")
                orig_bc = R.build_config

                def bc(*x, **y):
                    c, *rest = orig_bc(*x, **y)
                    c["run_dir"] = os.path.relpath(os.path.join(scratch, "orig_fm", str(fold)),
                                                   os.path.join(REPO, "runs"))
                    return (c, *rest)
                ps.set(R, "build_config", bc, "scratch out dir only")
                try:
                    R.main(W.base_argv(a, None))
                except W._Constructed:
                    pass
            loaders = {k: v for (_x, k, v) in rec.built}
        else:
            W.main(["--fold", str(fold), "--target-pool", pool_mode, "--target-sampler", samp,
                    "--device", "cpu", "--out-root", scratch, "--emit-config", cfg_path]
                   + (["--init-head-from", stand_in["init_head_from"], "--frozen-dir",
                       stand_in["frozen_dir"]] if stand_in else []))
            loaders = W.construct(a)
        cfg = json.load(open(cfg_path))
        pre = 40 * -(-len(loaders[True].dataset) // 256)
        return cfg, loaders[False].batch_sampler, loaders[True].batch_sampler, pre
    # frozen stages
    W = __import__(f"run_dilute10_{arm}")
    a = ns(fold, pool_mode, samp, nw=4 if arm == "freematch_frozen" else 0,
           out_root=scratch)
    if mode == "module_original":
        a.target_sampler = "balanced"
        cfg, built = W.run(a, dry=True)       # recording-only patches + stop; no rebinding
    else:
        cfg, built = W.run(a, dry=True)
    if arm == "freematch_frozen":
        return cfg, built[False].batch_sampler, built[True].batch_sampler, 0
    import run_bakeoff_coral as BC
    return cfg, built[BC.SEED + fold], built[BC.SEED + fold + 1000], 0


def compare_cfg(arm, fold, emitted, stored):
    ex = dict(EXEMPT_COMMON, **EXEMPT_ARM.get(arm, {}))
    decl = DECLARED_EXCEPTIONS.get((arm, fold), {})
    diffs, declared, exempt_hit = [], [], []
    for k in sorted(emitted):
        if k.startswith("_"):
            continue
        if k not in stored:
            if k in decl:
                declared.append(f"{k}: only in emitted")
            elif k not in ex:
                diffs.append(f"{k}: only in emitted")
            continue
        if emitted[k] != stored[k]:
            if k in ex:
                exempt_hit.append(k)
            elif k in decl:
                declared.append(k)
            else:
                diffs.append(k)
    return {"undeclared_differences": diffs, "declared_exceptions": declared,
            "exempt_fields_that_differ": exempt_hit,
            "n_fields_compared": len([k for k in emitted if k in stored])}


def heavy(arms, folds):
    rep = {"T1": {}, "RNG": {}, "DIFF": {}}
    scratch = tempfile.mkdtemp(prefix="dilute10_heavy_")
    try:
        for arm in arms:
            for k in folds:
                key = f"{arm}/fold{k}"
                try:
                    cw, sw, tw, _ = construct_pair(arm, k, "wrapper_original", scratch=scratch)
                    cm, sm, tm, _ = construct_pair(arm, k, "module_original", scratch=scratch)
                    stored = json.load(open(os.path.join(REPO, STORED[arm](k))))
                    cmp_ = compare_cfg(arm, k, cw, stored)
                    skip = set(EXEMPT_COMMON) | set(EXEMPT_ARM.get(arm, {}))
                    same_inproc = {kk: v for kk, v in cw.items() if kk not in skip
                                   and not kk.startswith("_")} == \
                        {kk: v for kk, v in cm.items() if kk not in skip
                         and not kk.startswith("_")}
                    sb_w, sb_m = _batches(sw), _batches(sm)
                    tb_w, tb_m = _batches(tw), _batches(tm)
                    rep["T1"][key] = {
                        "config_vs_stored": cmp_,
                        "config_equals_in_process_original": same_inproc,
                        "first20_source_equal": sb_w == sb_m,
                        "first20_target_equal": tb_w == tb_m,
                        "on_disk_main_error": cm.get("_on_disk_main_error"),
                        "pass": (not cmp_["undeclared_differences"] and same_inproc
                                 and sb_w == sb_m and tb_w == tb_m)}
                except Exception as e:                                   # noqa: BLE001
                    rep["T1"][key] = {"pass": False, "error": repr(e),
                                      "trace": traceback.format_exc()[-1500:]}
                print(f"[T1] {key}: {rep['T1'][key].get('pass')}", flush=True)
            # ---- RNG check + DIFF on fold 1 (RNG on every fold requested in --folds) ----
            for k in folds:
                key = f"{arm}/fold{k}"
                try:
                    stand = _stand_in(arm, k, scratch)
                    co, so, to_, pre_o = construct_pair(arm, k, "module_original",
                                                        scratch=scratch)
                    cd, sd, td, pre_d = construct_pair(arm, k, "wrapper_dilute10",
                                                       scratch=scratch, stand_in=stand)
                    se = arm != "bnm"
                    pp = 500 if arm == "freematch_nb3" else 0
                    s_o = C.paired_source_sequence(so, to_, 2000, pre_target_batches=pre_o,
                                                   pre_paired_steps=pp, set_epoch=se)
                    s_d = C.paired_source_sequence(sd, td, 2000, pre_target_batches=pre_d,
                                                   pre_paired_steps=pp, set_epoch=se)
                    rep["RNG"][key] = {"identical_2000_steps": s_o == s_d,
                                       "digest_original": C.seq_digest(s_o),
                                       "digest_dilute10": C.seq_digest(s_d),
                                       "sat_pre_target_batches": [pre_o, pre_d],
                                       "adam_warmup_paired_steps": pp,
                                       "target_sampler_dilute10": type(
                                           getattr(td, "inner", td)).__name__}
                    if k == 1:
                        allowed = set(ALLOWED_DILUTE_DIFF)
                        if arm in ("coral_nb3", "freematch_nb3"):
                            allowed |= {"init_head_from", "init_head_from_tail",
                                        "config_diff", "paths", "run_dir",
                                        "started_utc"}
                        if arm == "freematch_nb3":
                            allowed |= {"run_dir"}
                        keys = sorted(set(co) | set(cd))
                        d = [x for x in keys if not x.startswith("_")
                             and co.get(x, "<absent>") != cd.get(x, "<absent>")]
                        rep["DIFF"][arm] = {"differing_keys": d,
                                            "disallowed": [x for x in d if x not in allowed],
                                            "dilute10_block": cd.get("dilute10")}
                except Exception as e:                                   # noqa: BLE001
                    rep["RNG"][key] = {"error": repr(e),
                                       "trace": traceback.format_exc()[-1500:]}
                print(f"[RNG] {key}: {rep['RNG'][key].get('identical_2000_steps')}",
                      flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return rep


def _stand_in(arm, fold, scratch):
    """For the nb3 runners' dilute10 construction: a COPY of round 1's head as the stand-in
    for the new frozen head (which does not exist yet), with gate values taken from the
    runner module's own FOLD_GATE table for that head -- read from code, no metric computed.
    """
    if arm not in ("coral_nb3", "freematch_nb3"):
        return None
    if arm == "coral_nb3":
        import run_coral_round4 as M
        src = os.path.join(REPO, "runs", "bakeoff_coral", f"fold{fold}", "checkpoints",
                           "step_8000.pt")
    else:
        import run_unfreeze_freematch as M
        src = os.path.join(REPO, "runs", "bakeoff_freematch_5fold", f"fold{fold}",
                           "checkpoints", "step_8000.pt")
    d = os.path.join(scratch, f"standin_{arm}_{fold}")
    os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
    head = os.path.join(d, "checkpoints", "step_8000.pt")
    shutil.copyfile(src, head)
    g = M.FOLD_GATE[fold]
    json.dump({"fold": fold, "head_sha256": DP.sha256_file(head), "tpr": g["tpr"],
               "auroc": g["auroc"], "head_l2": g["head_l2"],
               "note": "STAND-IN for the test: round 1's head, its FOLD_GATE row"},
              open(os.path.join(d, C.GATE_FILE), "w"))
    return {"init_head_from": head, "frozen_dir": d}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--t13", action="store_true",
                    help="T13: inject faulty target samplers into every runner")
    ap.add_argument("--cfgid", action="store_true",
                    help="CFGID: emitted configs vs the pre-change snapshot (v2 provenance)")
    ap.add_argument("--arms", default="dann,bnm,coral_nb3,freematch_nb3,freematch_frozen,"
                                      "coral_frozen")
    ap.add_argument("--folds", default="1,2,3,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and not a.force:
        raise SystemExit(f"refusing to overwrite {a.out}")
    import integrity
    integrity.verify(where="(test_dilute10 start)")
    rep = {}
    if a.quick:
        rep["quick"] = []
        for t in (t2_uniform_sampler, t3_pool_loader, t4_holdout, t5_sat_rebuild,
                  t6_gate_wiring, t7_logger, t8_no_edits, t9_output_safety,
                  t10_integrity_build, t11_refuse_zero_workers, t12_fold_identity,
                  t14_part2_provenance, t15_feature_norm_advisory, t16_part3_provenance):
            try:
                r = t()
            except Exception as e:                                       # noqa: BLE001
                r = ok(t.__name__, False, {"error": repr(e),
                                           "trace": traceback.format_exc()[-2000:]})
            rep["quick"].append(r)
            print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['test']}", flush=True)
    if a.heavy:
        rep["heavy"] = heavy(a.arms.split(","), [int(x) for x in a.folds.split(",")])
    if a.t13:
        r = t13_lens_fraction_stop(tuple(a.arms.split(",")))
        rep["t13"] = r
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['test']}", flush=True)
        r = t13b_band_by_completion(tuple(a.arms.split(",")))
        rep["t13b"] = r
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['test']}", flush=True)
    if a.cfgid:
        r = cfgid()
        rep["cfgid"] = r
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['test']}", flush=True)
    integrity.verify(where="(test_dilute10 end)")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
