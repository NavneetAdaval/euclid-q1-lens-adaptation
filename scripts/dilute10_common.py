"""
dilute10_common.py
==================
Shared plumbing for the six dilute10 runners (brief_dilute10_code.md section 4).

A runner WRAPS an existing runner: it imports that module and, for the duration of one
call, rebinds a handful of NAMES inside it (``PatchSet``) -- the target ``LensDataset``, the
target sampler, the DANN/BNM holdout, the config builder (to add provenance) -- and then
calls the module's own ``main`` / ``build_arm`` / training function. No file is edited, no
loss and no training step is re-implemented, and every binding is restored on exit.

With ``--target-pool original --target-sampler balanced`` (the defaults) NOTHING that
reaches the training path is rebound: only the provenance fields are added, plus (for the
FreeMatch / frozen runners) the measurement lockout below, which touches no training
quantity.

MEASUREMENT LOCKOUT. The brief permits exactly one test-metric read per run: the step-0
gate (and, for a frozen stage, its final head's gate values). The round-1 frozen trainers
and the FreeMatch x nb3 runner measure test TPR at every curve point and at R1; here those
measurements are replaced by SCORING ONLY (logits persisted, no metric), which changes no
training quantity: the curve hook uses no RNG and its rows are only printed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dilute10_pool as DP                                   # noqa: E402
from dilute10_batch_log import LensFractionLogger           # noqa: E402
from uniform_target_sampler import (NAMESPACE, UniformTargetBatchSampler,  # noqa: E402
                                    make_uniform_target_loader)

OUT_ROOT = os.path.join(REPO, "runs", "dilute10")
FOLDS = (1, 2, 3, 4)
NEW_PROVENANCE_FIELDS = ("dilute10",)                       # the one added top-level key


# --------------------------------------------------------------------------------------
# patching
# --------------------------------------------------------------------------------------
class PatchSet(contextlib.AbstractContextManager):
    """Rebind attributes on modules/objects; restore every one on exit."""

    def __init__(self):
        self._saved = []
        self.log = []

    def set(self, obj, name, value, why=""):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
        self.log.append(f"{getattr(obj, '__name__', type(obj).__name__)}.{name}"
                        + (f" -- {why}" if why else ""))
        return value

    def __exit__(self, *exc):
        for obj, name, val in reversed(self._saved):
            setattr(obj, name, val)
        self._saved.clear()
        return False


# --------------------------------------------------------------------------------------
# arguments, output directories, provenance
# --------------------------------------------------------------------------------------
def add_common_args(p: argparse.ArgumentParser, *, num_workers_default: int):
    p.add_argument("--fold", type=int, required=True, choices=FOLDS)
    p.add_argument("--target-pool", choices=("original", "dilute10"), default="original")
    p.add_argument("--target-sampler", choices=("balanced", "uniform"), default="balanced")
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--force", action="store_true",
                   help="write into an existing output directory")
    p.add_argument("--emit-config", default=None,
                   help="construct the run, write its run_config.json here, train nothing")
    p.add_argument("--dry-run", action="store_true",
                   help="construct the run, draw the first batches, train nothing, write "
                        "nothing except --emit-config if given")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=num_workers_default)
    p.add_argument("--input-root", default=None)
    return p


def refuse_zero_workers(args):
    """Part 2 (D.7): with ``num_workers=0`` augmentation runs in-process on torch's global
    RNG, which FreeMatch's larger SAT rebuild would advance. Every DataLoader-based runner
    therefore refuses 0; the declared values are the original runs' (Kaggle, with workers)."""
    if int(getattr(args, "num_workers", 1)) == 0:
        raise SystemExit("[dilute10] REFUSING --num-workers 0 (runs/dilute10_prereg_part2.json "
                         "D.7): declare the original run's worker count")


PART2_PATH = os.path.join(REPO, "runs", "dilute10_prereg_part2.json")
#: part 2's self-certifying digest (runs/dilute10_prereg_part2.json, PROJECTv7 7.38). Every run
#: from bundle v2 on records it in its own files (brief_dilute10_part2_provenance, 7.43).
PART2_SHA256 = "b8e09197a204be835fa089c858b08c7eb3e7887bfa19afd296d544d04f394503"
#: Cell 2's record (scripts/stage_dilute10.py) in the working directory -- the repo root
STAGE_JSON = os.path.join(REPO, "stage.json")


def verify_part2(part2_path=PART2_PATH):
    """Re-derive part 2's self-certifying digest from the staged file; refuse to start unless
    it is the pinned ``PART2_SHA256``."""
    import hashlib as _h
    if not os.path.isfile(part2_path):
        raise SystemExit(f"[dilute10] REFUSING: {part2_path} (part 2) is missing")
    d = json.load(open(part2_path, encoding="utf-8"))
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    got = _h.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if got != PART2_SHA256 or d.get("sha256_of_body_without_this_field") != PART2_SHA256:
        raise SystemExit(f"[dilute10] REFUSING: part 2 digest {got} != pinned {PART2_SHA256}")
    return got


PART3_PATH = os.path.join(REPO, "runs", "dilute10_prereg_part3.json")
#: part 3's self-certifying digest (runs/dilute10_prereg_part3.json, PROJECTv7 7.48): the CORAL
#: nb3 decision, the no-verdict rule and the FreeMatch nb3 rule. Binding on the two nb3 arms.
PART3_SHA256 = "1b583ea2ae5056edb410040c32613d9388e5a994bf90fc366fc93d293efb596a"
#: the arms part 3 binds; they refuse to start without it. Every other arm records it if
#: present and verified, and never requires it.
PART3_REQUIRED_ARMS = ("coral_nb3", "freematch_nb3")


def _part3_digest(part3_path):
    import hashlib as _h
    d = json.load(open(part3_path, encoding="utf-8"))
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    got = _h.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return got, d.get("sha256_of_body_without_this_field")


def verify_part3(arm, part3_path=PART3_PATH):
    """Re-derive part 3's self-certifying digest from the staged file. For the arms part 3
    binds (``PART3_REQUIRED_ARMS``) refuse to start unless it is present and equals the pinned
    ``PART3_SHA256``. For every other arm return the digest if present and verified, else
    None -- never refuse."""
    required = arm in PART3_REQUIRED_ARMS
    if not os.path.isfile(part3_path):
        if required:
            raise SystemExit(f"[dilute10] REFUSING: {part3_path} (part 3) is missing; it binds "
                             f"{arm} and must be staged before it runs")
        return None
    got, claimed = _part3_digest(part3_path)
    if got != PART3_SHA256 or claimed != PART3_SHA256:
        if required:
            raise SystemExit(f"[dilute10] REFUSING: part 3 digest {got} != pinned "
                             f"{PART3_SHA256}")
        return None
    return got


def bundle_provenance(stage_path=STAGE_JSON):
    """Bundle version and MANIFEST.json sha256 from Cell 2's ``stage.json`` (Kaggle); null with
    a reason when there is none (a local run). Where the staged bundle is readable, its
    manifest is re-hashed and must match."""
    if not os.path.isfile(stage_path):
        return {"bundle_version": None, "bundle_manifest_sha256": None,
                "bundle_note": f"local run: no Cell 2 stage.json at {stage_path}"}
    st = json.load(open(stage_path, encoding="utf-8"))
    ver, msha = st.get("version"), st.get("manifest_sha256")
    if ver is None or not msha:
        raise SystemExit(f"[dilute10] REFUSING: {stage_path} lacks version/manifest_sha256 "
                         f"(staged from a bundle older than v2)")
    if st.get("part2") != PART2_SHA256:
        raise SystemExit(f"[dilute10] REFUSING: stage.json part 2 {st.get('part2')} != "
                         f"{PART2_SHA256}")
    if st.get("part3") is not None and st.get("part3") != PART3_SHA256:
        raise SystemExit(f"[dilute10] REFUSING: stage.json part 3 {st.get('part3')} != "
                         f"{PART3_SHA256}")
    mp = os.path.join(st.get("bundle") or "", "MANIFEST.json")
    verified = None
    if os.path.isfile(mp):
        verified = DP.sha256_file(mp) == msha
        if not verified:
            raise SystemExit(f"[dilute10] REFUSING: {mp} no longer hashes to stage.json's "
                             f"manifest_sha256 {msha}")
    return {"bundle_version": int(ver), "bundle_manifest_sha256": msha,
            "bundle_manifest_rehashed": verified,
            "stage_json_sha256": DP.sha256_file(stage_path)}


def fold_identity_expected(arm, fold, part2_path=PART2_PATH):
    """Part 2's fold-identity hashes for a frozen stage, from the VERIFIED part-2 file."""
    import hashlib as _h
    if not os.path.isfile(part2_path):
        raise SystemExit(f"[dilute10] REFUSING: {part2_path} is missing -- the fold-identity "
                         f"check has nothing to check against")
    d = json.load(open(part2_path, encoding="utf-8"))
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    got = _h.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if got != d.get("sha256_of_body_without_this_field"):
        raise SystemExit(f"[dilute10] REFUSING: {part2_path} fails its self-certifying digest")
    return d["fold_identity"][arm][str(int(fold))]


def ids_sha(ids):
    import hashlib as _h
    return _h.sha256("\n".join(str(x) for x in ids).encode()).hexdigest()


def check_fold_identity(arm, fold, *, target_ids, cache_dir):
    """Abort unless the served target IDs and the test IDs this run would score hash to
    part 2's values for this fold."""
    from evaluate import FeatureCache
    exp = fold_identity_expected(arm, fold)
    c = FeatureCache(cache_dir)
    tpos = [str(x) for x in c.load(f"test_pos_fold{int(fold)}")["ids"]]
    tneg = [str(x) for x in c.load("test_negatives_shared")["ids"]]
    got = {"target_ids_sha256": ids_sha([str(x) for x in target_ids]),
           "test_ids_sha256": ids_sha(tpos + ["--"] + tneg)}
    bad = {k: (got[k], exp[k]) for k in got if got[k] != exp[k]}
    if bad:
        raise SystemExit(f"[dilute10] FOLD-IDENTITY ABORT ({arm} fold {fold}): {bad}")
    return {"checked": True, **got}


def post_run_check(*, arm, out_dir, budget, ckpt_every, logit_subdir, steps_json=None,
                   steps_key="steps_completed", first_step=None, r1_ckpt=None):
    """Run-time stop conditions 'a missed budget' and 'missing per-checkpoint logits'
    (part 2), checked when the wrapped runner returns. Stops with a named reason; writes
    ``dilute10_post_run_check.json`` either way."""
    first = int(ckpt_every) if first_step is None else int(first_step)
    grid = list(range(first, int(budget) + 1, int(ckpt_every)))
    ldir = os.path.join(out_dir, logit_subdir)
    missing = [s for s in grid if not os.path.isfile(os.path.join(ldir, f"step_{s}.npz"))]
    named_missing = []
    ck = os.path.join(out_dir, "checkpoints.csv")
    if os.path.isfile(ck):
        import csv as _csv
        for r in _csv.DictReader(open(ck, newline="")):
            f = (r.get("logit_file") or "").strip()
            if f and not os.path.isfile(os.path.join(out_dir, f)):
                named_missing.append(f)
    steps = None
    if steps_json and os.path.isfile(os.path.join(out_dir, steps_json)):
        steps = json.load(open(os.path.join(out_dir, steps_json), encoding="utf-8")).get(steps_key)
    r1_ok = None if r1_ckpt is None else os.path.isfile(os.path.join(out_dir, r1_ckpt))
    rep = {"arm": arm, "budget": int(budget), "steps_completed": steps,
           "r1_checkpoint_present": r1_ok, "logit_grid": [grid[0], grid[-1], int(ckpt_every)],
           "n_expected_logits": len(grid), "missing_logits": missing,
           "named_in_checkpoints_csv_but_missing": named_missing}
    with open(os.path.join(out_dir, "dilute10_post_run_check.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    if (steps_json and (steps is None or int(steps) != int(budget))) or r1_ok is False:
        raise SystemExit(f"[dilute10 STOP: missed budget] {arm}: steps_completed={steps}, "
                         f"budget {budget}, R1 present={r1_ok} (part 1 stop condition 6)")
    if missing or named_missing:
        raise SystemExit(f"[dilute10 STOP: logit coverage] {arm}: missing per-checkpoint "
                         f"logits {missing[:6]} {named_missing[:6]} (part 1 stop condition 7)")
    return rep


def out_dir_for(args, arm):
    return os.path.join(os.path.abspath(args.out_root), arm, f"fold{int(args.fold)}")


def refuse_existing(path, force):
    if os.path.exists(path) and not force:
        raise SystemExit(f"[dilute10] REFUSING: {path} already exists. Never overwrite a run; "
                         f"pass --force only for a logged infrastructure rerun.")


def module_sha256s() -> dict:
    """sha256 of every loaded module whose file lives in scripts/."""
    out = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f or not os.path.abspath(f).startswith(HERE + os.sep):
            continue
        if f.endswith(".py"):
            out[os.path.basename(f)] = DP.sha256_file(f)
    return dict(sorted(out.items()))


def provenance(args, *, arm, pool=None, counts=None, abort_mode=None, extra=None,
               patches=None) -> dict:
    p1 = DP.verify_part1()
    d = {
        "experiment": "dilute10",
        "arm": arm,
        "part1_sha256": DP.PART1_SHA256,
        "part2_sha256": verify_part2(),
        "part3_sha256": verify_part3(arm),
        "pool_sha256": p1["pool"]["whole_file_sha256"],
        "pool_path": "data/splits/dilute10_pool.json",
        "target_pool": args.target_pool,
        "target_sampler": args.target_sampler,
        "uniform_sampler_namespace": NAMESPACE if args.target_sampler == "uniform" else None,
        "realised_training_pool": counts,
        "abort_mode": abort_mode,
        "rebound_names": list(patches.log) if patches is not None else [],
        "imported_module_sha256": module_sha256s(),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **bundle_provenance(),
    }
    if pool is not None:
        d["pool_verified"] = pool.get("_verified")
    if extra:
        d.update(extra)
    return d


def get_pool(args):
    return DP.load_pool() if args.target_pool == "dilute10" else None


def logger_csv(args, arm):
    if args.emit_config or args.dry_run:
        return None
    return os.path.join(out_dir_for(args, arm), "target_lens_fraction.csv")


# --------------------------------------------------------------------------------------
# sampler replacements
# --------------------------------------------------------------------------------------
class Recorder:
    """Records every sampler/loader a constructor builds (used by --dry-run and tests)."""

    def __init__(self):
        self.built = []


def direct_sampler_ctor(orig_cls, *, uniform: bool, target_seed: int, n_lenses=None,
                        csv_path=None, recorder=None, expect_pool=None):
    """Replacement for ``ClassBalancedBatchSampler(pos, neg, batch_size, ...)`` calls.

    Calls whose ``seed`` is not the TARGET seed (SEED + fold + 1000) are passed straight to
    the original class. For the target call: with ``uniform``, the positions of the two
    lists are merged into ONE label-free training pool and handed to
    ``UniformTargetBatchSampler`` -- the sampler receives positions only.
    """
    def ctor(pos_indices, neg_indices, batch_size, num_batches=None, pos_fraction=0.5,
             seed=0):
        if int(seed) != int(target_seed) or not uniform:
            s = orig_cls(pos_indices, neg_indices, batch_size, num_batches=num_batches,
                         pos_fraction=pos_fraction, seed=seed)
        else:
            pool = np.sort(np.concatenate([np.asarray(pos_indices, np.int64),
                                           np.asarray(neg_indices, np.int64)]))
            if expect_pool is not None and not np.array_equal(pool, np.sort(expect_pool)):
                raise AssertionError("target training pool differs from the pinned one")
            s = UniformTargetBatchSampler(pool, batch_size, num_batches, seed)
            if n_lenses is not None:
                s = LensFractionLogger(s, n_lenses, csv_path)
        if recorder is not None:
            recorder.built.append(("seed", int(seed), s))
        return s
    ctor.__wrapped__ = orig_cls
    return ctor


def loader_ctor(orig_fn, *, uniform: bool, n_lenses=None, csv_path=None, recorder=None):
    """Replacement for ``make_balanced_loader``: the TARGET call (``use_latent=True``) gets
    the uniform sampler over every position of the dataset; source calls pass through."""
    def fn(dataset, batch_size, use_latent=False, num_batches=None, pos_fraction=0.5,
           seed=0, num_workers=0, pin_memory=False, persistent_workers=False):
        if use_latent and uniform:
            wrap = ((lambda inner: LensFractionLogger(inner, n_lenses, csv_path))
                    if n_lenses is not None else None)
            dl = make_uniform_target_loader(
                dataset, np.arange(len(dataset), dtype=np.int64), batch_size,
                num_batches if num_batches is not None else -(-len(dataset) // batch_size),
                seed, num_workers=num_workers, pin_memory=pin_memory,
                persistent_workers=persistent_workers, wrap=wrap)
        else:
            dl = orig_fn(dataset, batch_size, use_latent=use_latent, num_batches=num_batches,
                         pos_fraction=pos_fraction, seed=seed, num_workers=num_workers,
                         pin_memory=pin_memory, persistent_workers=persistent_workers)
        if recorder is not None:
            recorder.built.append(("use_latent", bool(use_latent), dl))
        return dl
    fn.__wrapped__ = orig_fn
    return fn


def write_logger_summary(recorder, out_dir, *, completed: bool = True,
                         reason: str | None = None):
    """After a run: the lens-fraction logger's mean/sd, next to its CSV (never to a loss).

    ``completed`` says whether the run completed its budget, read by the caller from the
    run's own record (``run_completion``). Part 2's band is enforced only then; for an
    aborted or crashed run the summary records ``band_enforced: false`` and ``reason`` and
    does not raise (PROJECTv7 7.46/7.47)."""
    if recorder is None or out_dir is None or not os.path.isdir(out_dir):
        return None
    for item in recorder.built:
        obj = item[2]
        cand = getattr(obj, "batch_sampler", obj)
        if isinstance(cand, LensFractionLogger):
            s = cand.summary(enforce=bool(completed), reason=reason)
            with open(os.path.join(out_dir, "target_lens_fraction_summary.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(s, fh, indent=2)
            return s
    return None


def run_completion(out_dir, *, budget, steps_json, steps_key="steps_completed"):
    """(completed, reason) from a wrapped runner's OWN record after it returned normally:
    completed iff the record shows ``steps_key >= budget`` and no ``aborted`` flag. A run
    whose record is missing is treated as not completed, with that as the reason."""
    p = os.path.join(out_dir, steps_json)
    if not os.path.isfile(p):
        return False, f"run record {steps_json} not written"
    rec = json.load(open(p, encoding="utf-8"))
    steps = rec.get(steps_key)
    if rec.get("aborted"):
        return False, (f"run aborted at step {steps}"
                       + (f" ({rec.get('abort_reason')})" if rec.get("abort_reason") else ""))
    if steps is None or int(steps) < int(budget):
        return False, f"run stopped at step {steps} of {budget}"
    return True, None


@contextlib.contextmanager
def logger_summary_on_exception(recorder, out_dir_fn):
    """Around a wrapped ``main``: if it raises (a declared stop, a crash or an interrupt),
    write the lens-fraction summary UNENFORCED with the exception as the reason, then
    re-raise the original exception unchanged. ``out_dir_fn`` is called lazily (the
    directory may not exist yet when construction fails); ``recorder`` may be a callable
    returning the recorder (the frozen runners create theirs inside ``run``)."""
    try:
        yield
    except BaseException as e:
        try:
            out = out_dir_fn()
            rec = recorder() if callable(recorder) else recorder
            write_logger_summary(rec, out, completed=False,
                                 reason=f"run raised {type(e).__name__}: {str(e)[:400]}")
        except Exception:                                            # noqa: BLE001
            pass
        raise


def lens_count(pool, args, fold):
    if args.target_pool == "dilute10":
        return int(DP.pool_fold(pool, fold)["n_lenses"])
    sp = DP.load_splits()
    return len(next(f for f in sp["real_lenses_kfold"]["folds"]
                    if int(f["fold"]) == int(fold))["target_lens_rows"])


# --------------------------------------------------------------------------------------
# measurement lockout: scoring without measuring
# --------------------------------------------------------------------------------------
def scoring_only_curve_row(orig_fn, *, gate_steps=(), dump_dir=None, fold=None):
    """Replacement for ``curve_hook.append_curve_row``.

    At steps in ``gate_steps`` (the step-0 gate) it calls the original, which measures.
    Everywhere else it SCORES the same splits the original reads, fills
    ``collect_scores`` exactly as the original would, writes a curve row whose metric
    columns are blank, and (``dump_dir``) persists the logits -- it computes no metric.
    """
    import curve_hook as CH
    from evaluate import FeatureCache, load_checkpoint, score_features

    def fn(curve_csv, step, ckpt_path, cache_dir, *, folds=CH.DEFAULT_FOLDS,
           target_fpr=None, val_select_metric="auroc", device="cpu", agreement_refs=None,
           split_loader=None, collect_scores=None):
        if int(step) in set(int(s) for s in gate_steps):
            return orig_fn(curve_csv, step, ckpt_path, cache_dir, folds=folds,
                           target_fpr=target_fpr if target_fpr is not None
                           else CH.DEFAULT_TARGET_FPR,
                           val_select_metric=val_select_metric, device=device,
                           agreement_refs=agreement_refs, split_loader=split_loader,
                           collect_scores=collect_scores)
        cache = FeatureCache(cache_dir)
        load = split_loader if split_loader is not None else (
            lambda n: CH._cached_split(cache, n))
        head, std, _ = load_checkpoint(ckpt_path, device)
        got = {}
        names = ["val", "test_negatives_shared"] + [f"test_pos_fold{int(k)}" for k in folds]
        if collect_scores is not None:
            names += [f"target_fold{int(k)}" for k in folds]
        for n in names:
            sp = load(n)
            got[n] = np.asarray(score_features(head, std, sp["features"], device),
                                dtype=np.float32)
            if sp.get("ids") is not None:
                got[f"{n}__ids"] = np.asarray(sp["ids"])
        if collect_scores is not None:
            collect_scores.update(got)
        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True)
            k = int(fold if fold is not None else list(folds)[0])
            keep = ["val", "test_negatives_shared", f"test_pos_fold{k}"]
            np.savez_compressed(os.path.join(dump_dir, f"step_{int(step)}.npz"),
                                step=np.asarray(int(step)),
                                **{n: got[n] for n in keep})
            ids_p = os.path.join(dump_dir, "ids.npz")
            if not os.path.exists(ids_p):
                np.savez_compressed(ids_p, **{n: got[f"{n}__ids"] for n in keep
                                              if f"{n}__ids" in got})
        row = {"step": int(step)}
        agreement_folds = sorted(agreement_refs) if agreement_refs else ()
        cols = CH.curve_columns(folds, agreement_folds=agreement_folds)
        for c in cols:
            row.setdefault(c, None)
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(curve_csv)), exist_ok=True)
        new = not os.path.exists(curve_csv)
        with open(curve_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow({c: row[c] for c in cols})
        row["_scored_not_measured"] = True
        return row
    fn.__wrapped__ = orig_fn
    return fn


def label_free_memorization_row(step, scores, fold):
    """``unfreeze_instrument.memorization_row`` without the test-label split: the target and
    val confidences only. The test columns are withheld (None)."""
    from unfreeze_instrument import sigmoid
    tgt = scores.get(f"target_fold{int(fold)}")
    val = scores.get("val")
    return {"step": int(step),
            "conf_target_mean": None if tgt is None else float(np.mean(sigmoid(tgt))),
            "conf_test_matched": None, "conf_test_pos_mean": None,
            "conf_test_neg_mean": None, "conf_test_raw_mean": None,
            "target_minus_test_matched": None,
            "conf_val_mean": None if val is None else float(np.mean(sigmoid(val)))}


# --------------------------------------------------------------------------------------
# gate values (frozen stages write them; the nb3 runners read them)
# --------------------------------------------------------------------------------------
GATE_FILE = "gate_values.json"


D1_REMOVE = os.path.join(REPO, "data", "splits", "d1_test_neg_remove.json")


def write_gate_values(out_dir, fold, head_ckpt, cache_dir, step=8000):
    """RUN TIME ONLY: the final frozen head's gate values on fold k, from PERSISTED logits.

    The one permitted test-metric read of a frozen stage. Read from ``logits/step_<step>.npz``
    (the scoring-only hook's dump) and its ``ids.npz``:

      * top level (``tpr``, ``auroc``) -- the ORIGINAL test pool (all 26,399 shared negatives):
        what the nb3 runner's step-0 gate measures on its own path, so the gate compares like
        with like (part 2: one positive + AUROC <= 1e-3);
      * ``cleaned_pool`` -- the same logits with the D1 removal list applied
        (``data/splits/d1_test_neg_remove.json``, rule b83fa019...): the cleaned-pool values
        the launch brief asks for.

    The persisted logits are cross-checked against this head re-scored on the cached features
    (max |diff| <= 1e-5), so the file describes the head it names.
    """
    from evaluate import FeatureCache, load_checkpoint, score_features
    from metrics import auroc, tpr_at_fpr
    from run_unfreeze_freematch import head_l2_norm
    k = int(fold)
    L = np.load(os.path.join(out_dir, "logits", f"step_{int(step)}.npz"))
    I = np.load(os.path.join(out_dir, "logits", "ids.npz"))
    ps, ns = L[f"test_pos_fold{k}"].astype(np.float64), L["test_negatives_shared"].astype(np.float64)
    nid = [str(x) for x in I["test_negatives_shared"]]
    cache = FeatureCache(cache_dir)
    head, std, _ = load_checkpoint(head_ckpt, "cpu")
    rs = score_features(head, std, cache.load(f"test_pos_fold{k}")["features"], "cpu")
    rn = score_features(head, std, cache.load("test_negatives_shared")["features"], "cpu")
    drift = float(max(np.abs(rs - ps).max(), np.abs(rn - ns).max()))
    if drift > 1e-5:
        raise SystemExit(f"[dilute10] gate_values: persisted logits are not this head's "
                         f"scores (max |diff| {drift:.2e})")

    def suite(pos, neg):
        y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
        s = np.concatenate([pos, neg])
        return {"tpr": float(tpr_at_fpr(y, s)), "auroc": float(auroc(y, s)),
                "n_pos": int(pos.size), "n_neg": int(neg.size)}
    d1 = json.load(open(D1_REMOVE, encoding="utf-8"))
    rm = set(d1["ids"])
    keep = np.array([i not in rm for i in nid])
    orig, clean = suite(ps, ns), suite(ps, ns[keep])
    gv = {"fold": k, "head": os.path.abspath(head_ckpt),
          "head_sha256": DP.sha256_file(head_ckpt),
          "head_l2": round(float(head_l2_norm(head_ckpt)), 6),
          **orig, "pool": "original (all shared negatives) -- what the nb3 step-0 gate measures",
          "cleaned_pool": {**clean, "rule_body_sha256": d1["rule_body_sha256"],
                           "n_removed": int((~keep).sum())},
          "from": f"logits/step_{int(step)}.npz (persisted), cross-checked against the head "
                  f"re-scored on cached features (max |diff| {drift:.1e})",
          "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = os.path.join(out_dir, GATE_FILE)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(gv, fh, indent=2)
    return gv


class GateRefused(SystemExit):
    pass


def read_gate_values(frozen_dir, fold, head_ckpt):
    """The nb3 runners refuse to start without the frozen stage's gate values, or when the
    head they would initialise from is not the head those values describe."""
    p = os.path.join(frozen_dir, GATE_FILE)
    if not os.path.isfile(p):
        raise GateRefused(f"[dilute10] REFUSING: {p} is missing -- the new frozen stage has "
                          f"not finished, so there is no gate to check against.")
    with open(p, encoding="utf-8") as fh:
        gv = json.load(fh)
    if int(gv["fold"]) != int(fold):
        raise GateRefused(f"[dilute10] REFUSING: {p} is fold {gv['fold']}, not {fold}")
    got = DP.sha256_file(head_ckpt)
    if got != gv["head_sha256"]:
        raise GateRefused(f"[dilute10] REFUSING: head {head_ckpt} hashes to {got}, but "
                          f"{p} describes {gv['head_sha256']}.")
    return gv


def frozen_dir_for(args, arm_frozen):
    return os.path.join(os.path.abspath(args.out_root), arm_frozen, f"fold{int(args.fold)}")


# --------------------------------------------------------------------------------------
# batch-sequence helpers (dry-run, the RNG check, tests)
# --------------------------------------------------------------------------------------
def first_batches(batch_sampler, n):
    out = []
    while len(out) < n:
        for b in batch_sampler:
            out.append([int(i) for i in b])
            if len(out) >= n:
                break
    return out


def paired_source_sequence(src_bs, tgt_bs, n_steps, *, pre_target_batches=0,
                           pre_paired_steps=0, set_epoch=True):
    """Emulate PairedDomainLoader on the SAMPLERS alone for ``n_steps`` training steps.

    ``pre_target_batches`` target-only draws (FreeMatch's SAT rebuild) and
    ``pre_paired_steps`` paired draws (its Adam warm-up) come first, as in the runner.
    Returns the source batches of the training steps.
    """
    def target_draws(k):
        it = iter(tgt_bs)
        for _ in range(k):
            try:
                next(it)
            except StopIteration:
                it = iter(tgt_bs)
                next(it)

    target_draws(int(pre_target_batches))
    done = 0
    while done < int(pre_paired_steps):            # iter(paired) re-made on exhaustion
        tit = iter(tgt_bs)
        for sb in src_bs:
            try:
                next(tit)
            except StopIteration:
                tit = iter(tgt_bs)
                next(tit)
            done += 1
            if done >= pre_paired_steps:
                break
    seq, epoch = [], 0
    while len(seq) < n_steps:
        epoch += 1
        if set_epoch:
            for s in (src_bs, tgt_bs):
                if hasattr(s, "set_epoch"):
                    s.set_epoch(epoch)
        tit = iter(tgt_bs)
        for sb in src_bs:
            try:
                next(tit)
            except StopIteration:
                tit = iter(tgt_bs)
                next(tit)
            seq.append([int(i) for i in sb])
            if len(seq) >= n_steps:
                break
    return seq


def seq_digest(seq) -> str:
    return hashlib.sha256(json.dumps(seq).encode()).hexdigest()
