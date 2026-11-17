"""
dilute10_identity_check.py
==========================
Pre-launch dataset-identity check (brief_dilute10_launch section 0). Fold 1, one process
per arm, BOTH modes (original+balanced, then dilute10+uniform):

  identity  Every dataset the arm constructs is recorded -- every ``LensDataset`` instance
            (a recorder on ``LensDataset.__init__``, so ANY import path is caught), every
            cached split read through ``evaluate.FeatureCache.load``, plus the DANN/BNM
            holdout, the morphology-probe pool, BNM's trained-on probe, and the live
            encoder's test/val/target sets built inside the runner's patch context. Each is
            fingerprinted by its FULL served id list and the sha256 of its first 64 served
            image (or feature) arrays. The two modes are compared entry by entry; the ONLY
            entry allowed to differ is the target TRAINING pool. A known-null (one id swapped
            in the probe pool) must make the comparison fail.
  probe10   DANN fold 1, both modes, local GPU, num_workers=0, inside test_label_lockout():
            the morphology probe before any optimiser step, then 10 training steps; per-step
            morph_proxy_auroc and source/target participation ratios.

The recorders wrap; they change nothing that is served. Writes JSON to ``--out`` (a
scratch folder outside the repo). No test label, no metric.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402
import dilute10_pool as DP                                   # noqa: E402

FOLD = 1


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _sha_list(xs):
    return _sha("\n".join(str(x) for x in xs).encode())


# --------------------------------------------------------------------------------------
# recorders
# --------------------------------------------------------------------------------------
class Recorders:
    def __init__(self):
        self.log = []

    def __enter__(self):
        import evaluate as ev
        import lens_dataset as LD
        from cache_features import dataset_ids
        self._ld, self._ev = LD, ev
        self._init = LD.LensDataset.__init__
        self._load = ev.FeatureCache.load
        log = self.log
        orig_init, orig_load = self._init, self._load

        def init(obj, *a, **kw):
            orig_init(obj, *a, **kw)
            ids = [str(x) for x in dataset_ids(obj)]
            log.append({"kind": "LensDataset", "role": obj.role, "fold": obj.fold,
                        "n": len(obj), "aug": obj._tf_view0.aug, "n_views": obj.n_views,
                        "ids_sha256": _sha_list(ids), "ids_first": ids[:3],
                        "images64_sha256": _sha(np.ascontiguousarray(obj.images[:64])
                                                .tobytes())})

        def load(cache, name):
            d = orig_load(cache, name)
            ids = [str(x) for x in d.get("ids", [])]
            log.append({"kind": "FeatureCache", "dir": os.path.relpath(cache.dir, REPO)
                        .replace("\\", "/"), "name": name, "n": len(d["features"]),
                        "ids_sha256": _sha_list(ids),
                        "features64_sha256": _sha(np.ascontiguousarray(d["features"][:64])
                                                  .tobytes())})
            return d
        LD.LensDataset.__init__ = init
        ev.FeatureCache.load = load
        return self

    def __exit__(self, *exc):
        self._ld.LensDataset.__init__ = self._init
        self._ev.FeatureCache.load = self._load
        return False


def fp_tensor(t):
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return _sha(np.ascontiguousarray(a[:64]).tobytes())


def fp_morph(m):
    info = m.info
    return {"kind": "morph_pool",
            "ids_sha256": _sha_list([tuple(x) for x in info.get("mlfp_locations", [])]
                                    + ["--"] + [tuple(x) for x in
                                                info.get("reserve_locations", [])]),
            "images64_sha256": fp_tensor(m.images), "n": int(len(m.y)),
            "y_sha256": _sha(np.asarray(m.y).tobytes()),
            "fit_eval_sha256": _sha(np.asarray(m.fit_idx).tobytes()
                                    + np.asarray(m.eval_idx).tobytes())}


def fp_holdout(A, tgt_ds):
    from cache_features import dataset_ids
    ids = [str(x) for x in dataset_ids(tgt_ds)]
    h = A["holdout"]["holdout"]
    return {"kind": "holdout", "n": int(len(h)),
            "ids_sha256": _sha_list([ids[int(i)] for i in h]),
            "images64_sha256": fp_tensor(A["holdout_images"] if "holdout_images" in A
                                         else A["held_images"])}


def live_encoder_inside(patch_fn, args):
    """Build the LiveEncoder the runner builds, INSIDE the runner's patch context."""
    from live_encode import LiveEncoder
    with C.PatchSet() as ps:
        patch_fn(ps)
        LiveEncoder(DP.SPLITS_PATH, os.path.join(REPO, "data"), FOLD, batch_size=256,
                    num_workers=0, device="cpu", verbose=False)


# --------------------------------------------------------------------------------------
# per arm, per mode
# --------------------------------------------------------------------------------------
def ns(W, mode, nw, scratch, extra=()):
    pool, samp = ("dilute10", "uniform") if mode == "dilute10" else ("original", "balanced")
    p = W.build_parser() if hasattr(W, "build_parser") else \
        C.add_common_args(argparse.ArgumentParser(), num_workers_default=nw)
    return p.parse_args(["--fold", str(FOLD), "--target-pool", pool, "--target-sampler",
                         samp, "--device", "cpu", "--num-workers", str(nw), "--out-root",
                         scratch, *extra])


def build(arm, mode, scratch, standin):
    """Return (recorded log, explicit extras) for one arm in one mode."""
    extras = []
    with Recorders() as R:
        if arm in ("dann", "bnm"):
            W = __import__(f"run_dilute10_{arm}")
            a = ns(W, mode, 2, scratch)
            A, M = W.construct(a)
            tgt = (A["tgt_ds"])
            extras.append(fp_holdout(A, tgt))
            extras.append(fp_morph(A["morph"]))
            if arm == "bnm":
                extras.append({"kind": "bnm_trainedon_probe",
                               "ids_sha256": _sha_list(A["probe"]["probe"]),
                               "images64_sha256": fp_tensor(A["probe_images"])})
            a2 = copy.copy(a)
            a2.dry_run = True
            live_encoder_inside(lambda ps: W.patched(a2, M, ps), a2)
        elif arm == "coral_nb3":
            import run_coral_r4_confirm as CF
            import run_coral_round4 as R4
            W = __import__("run_dilute10_coral_nb3")
            a = ns(W, mode, 2, scratch,
                   standin if mode == "dilute10" else ())
            W.construct(a)
            live_encoder_inside(lambda ps: W.patched(a, CF, R4, ps), a)
        elif arm == "freematch_nb3":
            import run_unfreeze_freematch as RU
            W = __import__("run_dilute10_freematch_nb3")
            a = ns(W, mode, 4, scratch, standin if mode == "dilute10" else ())
            W.construct(a)
            live_encoder_inside(lambda ps: W.patched(a, RU, ps, sat_record={}, dry=True), a)
        else:                                            # frozen stages
            W = __import__(f"run_dilute10_{arm}")
            a = ns(W, mode, 4 if arm == "freematch_frozen" else 1, scratch)
            W.run(a, dry=True)
            import evaluate as ev
            c = ev.FeatureCache(os.path.abspath(a.cache_dir))
            for name in ("val", "test_negatives_shared", f"test_pos_fold{FOLD}"):
                c.load(name)                             # what the scoring stub reads
        log = list(R.log)
    return log + extras


def _stand_in(arm, scratch):
    import test_dilute10 as T
    s = T._stand_in(arm, FOLD, scratch)
    return ("--init-head-from", s["init_head_from"], "--frozen-dir", s["frozen_dir"]) \
        if s else ()


def is_target_training(e, mode_n):
    return (e.get("kind") == "LensDataset" and e.get("role") == "target"
            and e.get("n") in mode_n) or \
        (e.get("kind") == "FeatureCache" and e.get("name") == f"target_fold{FOLD}"
         and e.get("n") in mode_n)


def compare(orig, dil):
    """Entry-by-entry comparison; only the target training pool may differ."""
    pool = DP.load_pool()
    fd = DP.pool_fold(pool, FOLD)
    dil_n = {int(fd["n_lenses"]) + int(fd["n_negatives"])}
    rows, bad = [], []

    def key(e):
        return (e["kind"], e.get("role"), e.get("name"), e.get("fold"), e.get("aug"),
                e.get("n_views"))
    # pair entries in order within each key
    from collections import defaultdict
    qo, qd = defaultdict(list), defaultdict(list)
    for e in orig:
        qo[key(e)].append(e)
    for e in dil:
        qd[key(e)].append(e)
    for k in sorted(set(qo) | set(qd), key=str):
        lo, ld = qo.get(k, []), qd.get(k, [])
        for i in range(max(len(lo), len(ld))):
            eo = lo[i] if i < len(lo) else None
            ed = ld[i] if i < len(ld) else None
            target_train = ed is not None and is_target_training(ed, dil_n)
            same = (eo is not None and ed is not None and
                    {x: v for x, v in eo.items() if x not in ("dir",)} ==
                    {x: v for x, v in ed.items() if x not in ("dir",)})
            label = "/".join(str(x) for x in k if x is not None)
            rows.append({"dataset": label, "occurrence": i, "identical": same,
                         "allowed_to_differ": target_train,
                         "n_original": None if eo is None else eo.get("n"),
                         "n_dilute10": None if ed is None else ed.get("n")})
            if not same and not target_train:
                # a diluted-mode target entry with no original twin is ALSO the training pool
                bad.append(rows[-1])
    return {"rows": rows, "unexpected_differences": bad, "pass": not bad}


def run_identity(arm, out):
    scratch = tempfile.mkdtemp(prefix=f"d10id_{arm}_")
    try:
        standin = _stand_in(arm, scratch) if arm in ("coral_nb3", "freematch_nb3") else ()
        orig = build(arm, "original", scratch, standin)
        dil = build(arm, "dilute10", scratch, standin)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    res = compare(orig, dil)
    # known-null: the same comparison with one probe-pool id swapped must FAIL
    tampered = copy.deepcopy(orig)
    hit = False
    for e in tampered:
        if e.get("kind") == "morph_pool":
            e["ids_sha256"] = _sha_list(["swapped"]) ; hit = True
    what = "morph_pool ids (one id swapped)"
    if not hit:                       # arms without a probe pool: swap the source set's id
        for e in tampered:
            if (e.get("kind") == "LensDataset" and e.get("role") == "source_train") or \
                    (e.get("kind") == "FeatureCache" and e.get("name") == "source_train"):
                e["ids_sha256"] = _sha_list(["swapped"]); hit = True
                what = f"{e['kind']} source_train ids (one swapped)"
                break
    if not hit:
        raise SystemExit(f"[identity] {arm}: the known-null found nothing to tamper -- the "
                         f"check cannot be shown to work on this arm")
    kn = compare(tampered, dil)
    res["known_null"] = {"tampered": what, "comparison_failed": not kn["pass"]}
    res["entries_original"], res["entries_dilute10"] = orig, dil
    res["arm"], res["fold"] = arm, FOLD
    return res


# --------------------------------------------------------------------------------------
def run_probe10(out):
    import gc
    import torch
    import run_dann_r5 as D
    import run_dilute10_dann as W
    from coral_round4 import test_label_lockout
    res = {"fold": FOLD, "venue": torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "cpu", "num_workers": 0}
    for mode in ("original", "dilute10"):
        pool, samp = ("dilute10", "uniform") if mode == "dilute10" else ("original", "balanced")
        p = C.add_common_args(argparse.ArgumentParser(), num_workers_default=0)
        a = p.parse_args(["--fold", str(FOLD), "--target-pool", pool, "--target-sampler",
                          samp, "--device", "cuda", "--num-workers", "0"])
        A, _ = W.construct(a)
        sub = os.path.join(out, f"probe10_{mode}")
        P = D.Paths(sub)

        class NoScorer:
            def dump(self, step, *x, **k):
                return os.path.join(P.out, "logits", "NOT_SCORED"), 0.0, None, None
        tr = A["trainer"]
        with test_label_lockout(verbose=False):
            m0 = A["morph"].measure(tr, amp=tr._amp_enabled)
            cfg = dict(A["cfg_json"])
            cfg["max_steps"] = 10
            D.train_loop(A=A, P=P, cfg=cfg, record={}, scorer=NoScorer(), verbose=False)
        rows = list(csv.DictReader(open(P.steps_csv, newline="")))
        res[mode] = {"pre_update_morph_proxy_auroc": m0.get("morph_proxy_auroc"),
                     "pre_update_morph_head_auroc": m0.get("morph_head_auroc"),
                     "per_step": [{c: float(r[c]) for c in ("step", "morph_proxy_auroc",
                                                            "src_participation_ratio",
                                                            "tgt_participation_ratio")}
                                  for r in rows]}
        del A, tr
        gc.collect()
        torch.cuda.empty_cache()
    orig_rows = list(csv.DictReader(open(os.path.join(
        REPO, "runs", "dann_main", f"fold{FOLD}", "extracted", f"fold{FOLD}", "steps.csv"),
        newline="")))
    res["original_run_recorded"] = {
        "note": ("steps.csv has NO pre-update row: the loop measures the probe after the "
                 "first optimiser step and labels it step 1"),
        "step1_morph_proxy_auroc": float(orig_rows[0]["morph_proxy_auroc"]),
        "steps_1_10": [float(r["morph_proxy_auroc"]) for r in orig_rows[:10]],
        "src_pr_1_10": [float(r["src_participation_ratio"]) for r in orig_rows[:10]],
        "tgt_pr_1_10": [float(r["tgt_participation_ratio"]) for r in orig_rows[:10]],
        "venue": "Tesla T4, num_workers 2"}
    res["pre_update_equal_between_modes"] = (
        res["original"]["pre_update_morph_proxy_auroc"]
        == res["dilute10"]["pre_update_morph_proxy_auroc"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    if out.startswith(REPO + os.sep):
        raise SystemExit("--out must be outside the repo")
    os.makedirs(out, exist_ok=True)
    import integrity
    integrity.verify(where="(identity_check start)")
    r = run_probe10(out) if a.what == "probe10" else run_identity(a.what, out)
    dst = os.path.join(out, f"{a.what}.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1, default=str)
    if a.what != "probe10":
        print(f"[identity] {a.what}: pass={r['pass']} unexpected={len(r['unexpected_differences'])}"
              f" known_null_failed={r['known_null']['comparison_failed']}"
              f" entries={len(r['entries_original'])}/{len(r['entries_dilute10'])}")
    else:
        print(f"[probe10] pre-update equal: {r['pre_update_equal_between_modes']}")
    integrity.verify(where="(identity_check end)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
