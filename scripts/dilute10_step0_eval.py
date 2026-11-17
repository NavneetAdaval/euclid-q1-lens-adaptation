"""
dilute10_step0_eval.py
======================
Step-0 threshold evaluation on the 1:10 DILUTED pool with the UNIFORM sampler
(brief_dilute10_part2 section C). Folds 1-4, through the new runners. Writes JSON into
``--out`` (a SCRATCH folder outside the repo); ``write_dilute10_prereg_part2.py`` reads it.

NO TEST LABEL, NO TEST METRIC. The step-0 gate is never called. Every training or
measurement step runs inside ``coral_round4.test_label_lockout()``, which makes the metric
functions and every ``test_*`` cache split unreachable. DANN's checkpoint scorer is a stub
that scores nothing. The frozen stages' TEST IDs are read as identities (the ids arrays of
the cached test splits), outside the lockout, for the fold-identity check; no label, score
or metric is involved.

  dann        <= 10 training steps on the GPU (dilute10 + uniform, the runner's arm):
              true step 0 (holdout D accuracy, morphology probe) + steps 1-10 from steps.csv
  bnm         Baseline A's step-0 predicted-positive fraction on the diluted TRAINING pool
              (cached frozen features, bnm_stage0_expressivity's own batching), and the
              step-0 gradient norms through run_bnm_r6.measure_step0 under the wrapper
  frozen_ids  sha256 of the target-pool IDs each frozen runner serves, and of the test IDs
              it would score
  lens        exact expected lens fraction, and the run-mean the uniform sampler realises
              over each arm's full budget (deterministic; no training)

    python -u scripts/dilute10_step0_eval.py --what all --fold 1 --out C:/Users/krris/ML/scratch_dilute10_step0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402
import dilute10_pool as DP                                   # noqa: E402


def _sha_list(xs):
    return hashlib.sha256("\n".join(str(x) for x in xs).encode()).hexdigest()


def _venue():
    import torch
    return {"device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_arch": "Ada (sm_89)" if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "note": ("local RTX 4050, not a Kaggle T4; PROJECTv7.md 7.7 records that the "
                     "venue can matter at this precision")}


def _runner_args(parser_mod, fold, nw, extra=()):
    p = parser_mod.build_parser() if hasattr(parser_mod, "build_parser") else \
        C.add_common_args(argparse.ArgumentParser(), num_workers_default=nw)
    return p.parse_args(["--fold", str(fold), "--target-pool", "dilute10",
                         "--target-sampler", "uniform", "--device", "cuda",
                         "--num-workers", str(nw), *extra])


# --------------------------------------------------------------------------------------
def eval_dann(fold, out, workers=2):
    import run_dann_r5 as D
    import run_dilute10_dann as W
    from coral_round4 import test_label_lockout
    from dann_round5 import holdout_domain_stats
    args = _runner_args(W, fold, workers)
    A, _ = W.construct(args)
    trainer, dann = A["trainer"], A["dann"]
    sub = os.path.join(out, f"dann_fold{fold}_run")
    P = D.Paths(sub)

    class NoScorer:
        def dump(self, step, *a, **k):
            return os.path.join(P.out, "logits", "NOT_SCORED"), 0.0, None, None

    with test_label_lockout(verbose=False):
        h0 = holdout_domain_stats(trainer, dann, A["holdout_images"],
                                  amp=trainer._amp_enabled)
        m0 = A["morph"].measure(trainer, amp=trainer._amp_enabled)
        cfg = dict(A["cfg_json"])
        cfg["max_steps"] = 10
        rec = D.train_loop(A=A, P=P, cfg=cfg, record={}, scorer=NoScorer(), verbose=False)
    import csv
    rows = list(csv.DictReader(open(P.steps_csv, newline="")))

    def col(name):
        out_ = []
        for r in rows:
            v = r.get(name)
            out_.append(None if v in (None, "") else float(v))
        return out_
    cols = ["d_acc_holdout", "loss_domain_raw", "d_pred_source_frac",
            "morph_proxy_auroc", "morph_head_auroc", "src_participation_ratio",
            "tgt_participation_ratio", "d_acc_train", "lambda_current"]
    det = [c for c in (rows[0] if rows else {}) if c.endswith(("_fired", "_streak"))]
    series = {c: col(c) for c in cols + det if rows and c in rows[0]}
    steps = [int(float(r["step"])) for r in rows]

    def mx(c):
        v = [x for x in series.get(c, []) if x is not None]
        return max(v) if v else None

    def mn(c):
        v = [x for x in series.get(c, []) if x is not None]
        return min(v) if v else None
    hd = A["holdout"]
    nl = int(DP.pool_fold(DP.load_pool(), fold)["n_lenses"])
    res = {
        "fold": fold, "venue": _venue(), "steps_logged": steps,
        "true_step0": {"d_acc_holdout": h0.get("d_acc_holdout"),
                       "d_holdout_logit_mean": h0.get("d_holdout_logit_mean"),
                       "morph": {k: v for k, v in (m0 or {}).items()
                                 if isinstance(v, (int, float))}},
        "series_steps_1_10": series,
        "step1": {c: (series[c][0] if series.get(c) else None) for c in series},
        "max_steps_1_10": {c: mx(c) for c in series},
        "min_steps_1_10": {c: mn(c) for c in series},
        "domain_loss_falling": (None if not series.get("loss_domain_raw") else
                                bool(series["loss_domain_raw"][-1]
                                     < series["loss_domain_raw"][0])),
        "abort_watch": rec.get("abort_watch"), "aborted": rec.get("aborted"),
        "abort_mode": A["cfg_json"].get("abort_mode"),
        "cond3_margin": A["cfg_json"].get("abort_cond3_margin"),
        "cond4_margins": [A["cfg_json"].get("abort_cond4_src_margin"),
                          A["cfg_json"].get("abort_cond4_tgt_margin")],
        "control_curve": A["cfg_json"].get("control_curve"),
        "holdout_composition": {"lenses": int(len(hd["holdout_pos"])),
                                "galaxies": int(len(hd["holdout_neg"]))},
        "training_target": {"lenses": int((hd["train_pool"] < nl).sum()),
                            "galaxies": int((hd["train_pool"] >= nl).sum())},
        "num_workers": workers,
        "workers_note": ("num_workers=0 used ONLY because the Windows worker spawn failed "
                         "twice at 2 (OSError 22 / truncated pickle; PROJECTv7.md 7.7); these "
                         "10 diagnostic steps then draw augmentation in-process"
                         if workers == 0 else "workers as declared (2)"),
        "steps_completed": rec.get("steps_completed"),
        "wall_minutes": rec.get("wall_minutes"),
    }
    return res


# --------------------------------------------------------------------------------------
def eval_bnm(fold, out, bnm_workers=2):
    import torch
    import bnm_stage0_expressivity as X
    import evaluate as ev
    from coral_round4 import test_label_lockout
    pool = DP.load_pool()
    h = DP.pinned_holdout(pool, fold)
    feats = ev.FeatureCache(os.path.join(REPO, "data", "feature_cache_dilute10")).load(
        f"target_fold{fold}")["features"]
    w, b, mu, sig = X._load_head(X.BASELINE_A_HEAD)

    def ppf(rows):
        lg = X._logits(feats[rows], w, b, mu, sig)
        rng = np.random.default_rng(0)                       # part_b's seed and batching
        idx = rng.permutation(lg.shape[0])
        nb = max(1, lg.shape[0] // 256)
        pf = [float((torch.sigmoid(lg[torch.from_numpy(idx[i * 256:(i + 1) * 256].copy())])
                     > 0.5).float().mean()) for i in range(nb)]
        whole = float((torch.sigmoid(lg) > 0.5).float().mean())
        return {"batched_mean": float(np.mean(pf)), "n_batches": nb, "whole_pool": whole,
                "n_images": int(lg.shape[0])}
    nl = int(DP.pool_fold(pool, fold)["n_lenses"])
    sp = DP.load_splits()
    orig = next(f for f in sp["real_lenses_kfold"]["folds"] if f["fold"] == fold)
    n_orig = len(orig["target_lens_rows"]) + len(orig["target_neg_object_ids"])
    held = set(int(i) for i in h["holdout"])
    orig_train = np.array([i for i in range(n_orig) if i not in held])
    pp_dil = ppf(h["train_pool"])
    pp_bal = ppf(orig_train)
    import bnm_round6 as BR
    band = BR.collapse_band(pp_dil["batched_mean"])

    # step-0 gradient norms, through the wrapper (dilute10 + uniform), metric-blind
    import run_bnm_r6 as B
    import run_dilute10_bnm as W
    args = _runner_args(W, fold, bnm_workers)
    gpath = os.path.join(out, f"bnm_gradnorms_fold{fold}.json")
    bargs = B.build_argparser().parse_args(W.base_argv(args))
    paths = B.resolve_paths(bargs, REPO)
    paths.setdefault("init_head_from", B.DEFAULT_INIT_HEAD)
    a2 = copy.copy(args)
    a2.dry_run = True
    with test_label_lockout(verbose=False):
        with C.PatchSet() as ps:
            W.patched(a2, B, ps)
            g = B.measure_step0(bargs, paths, out_path=gpath, verbose=False)
    t4 = json.load(open(os.path.join(REPO, "runs",
                                     "bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json")))
    return {"fold": fold, "venue": _venue(),
            "step0_pred_pos_frac_diluted_training_pool": pp_dil,
            "step0_pred_pos_frac_balanced_training_pool_same_fold": pp_bal,
            "original_anchor": {"value": BR.stage0a_anchors()["init_pred_pos_frac"]
                                if hasattr(BR, "stage0a_anchors") else None,
                                "source": "runs/bnm_r6_stage0a.json part_b (fold 0, 782)"},
            "collapse_band_on_diluted_pool": band,
            "ceiling_step0_value": pp_dil["batched_mean"],
            "gradnorms": {k: g.get(k) for k in ("ce_grad_norm", "bnm_grad_norm_at_gamma_1",
                                                "ratio_bnm_over_ce_at_gamma_1",
                                                "n_params_in_norm", "gpu_name", "fp16")},
            "gradnorms_balanced_t4": {k: t4.get(k) for k in (
                "ce_grad_norm", "bnm_grad_norm_at_gamma_1", "ratio_bnm_over_ce_at_gamma_1",
                "gpu_name", "fold")},
            "gamma_unchanged": 0.00787,
            "gradnorm_num_workers": bnm_workers,
            "gradnorm_workers_note": (
                "num_workers=0 used ONLY for this descriptive measurement: on this Windows "
                "host the DataLoader worker spawn fails pickling the arm (OSError 22 / "
                "truncated pickle, reproduced; PROJECTv7.md 7.7). Linux/Kaggle forks and is "
                "unaffected. The value is descriptive; gamma is not re-derived from it."
                if bnm_workers == 0 else "workers as declared")}


# --------------------------------------------------------------------------------------
def eval_frozen_ids(fold, out):
    import evaluate as ev
    from cache_features import dataset_ids
    from lens_dataset import LensDataset
    pool = DP.load_pool()
    ds = DP.lens_dataset_factory(LensDataset, pool, fold)(
        DP.SPLITS_PATH, os.path.join(REPO, "data"), role="target", fold=fold, aug="weak",
        n_views=2)
    fm_tgt = [str(x) for x in dataset_ids(ds)]
    dil = ev.FeatureCache(os.path.join(REPO, "data", "feature_cache_dilute10"))
    cr_tgt = [str(x) for x in dil.load(f"target_fold{fold}")["ids"]]
    orig = ev.FeatureCache(os.path.join(REPO, "data", "feature_cache"))
    tpos = [str(x) for x in orig.load(f"test_pos_fold{fold}")["ids"]]
    tneg = [str(x) for x in orig.load("test_negatives_shared")["ids"]]
    test = {"test_pos_ids_sha256": _sha_list(tpos), "n_test_pos": len(tpos),
            "test_neg_ids_sha256": _sha_list(tneg), "n_test_neg": len(tneg),
            "test_ids_sha256": _sha_list(tpos + ["--"] + tneg)}
    return {"fold": fold,
            "freematch_frozen": {"target_ids_sha256": _sha_list(fm_tgt),
                                 "n_target": len(fm_tgt),
                                 "served_by": "LensDataset (diluted splits), cache_features."
                                              "dataset_ids", **test},
            "coral_frozen": {"target_ids_sha256": _sha_list(cr_tgt), "n_target": len(cr_tgt),
                             "served_by": "data/feature_cache_dilute10 target ids", **test},
            "same_target_ids_both_runners": fm_tgt == cr_tgt,
            "hash_rule": "sha256 of the ids joined by newlines, in served order"}


# --------------------------------------------------------------------------------------
def eval_lens(fold, out):
    from uniform_target_sampler import UniformTargetBatchSampler
    pool = DP.load_pool()
    fd = DP.pool_fold(pool, fold)
    nl, ng = int(fd["n_lenses"]), int(fd["n_negatives"])
    h = DP.pinned_holdout(pool, fold)
    full = np.arange(nl + ng)
    arms = {   # arm: (training pool, seed, pre-phase target batches, budget batches)
        "dann": (h["train_pool"], 42 + fold + 1000, 0, 4000),
        "bnm": (h["train_pool"], 42 + fold + 1000, 0, 4000),
        "coral_nb3": (full, 42 + fold + 1000, 0, 2000),
        "freematch_nb3": (full, 42 + fold + 1000, 40 * -(-(nl + ng) // 256) + 500, 1500),
        "coral_frozen": (full, 42 + fold + 1000, 0, 8000),
        "freematch_frozen": (full, 42 + fold + 1000, 0, 8000),
    }
    res = {"fold": fold}
    for arm, (tp, seed, pre, budget) in arms.items():
        L = int((tp < nl).sum())
        G = int(tp.size - L)
        s = UniformTargetBatchSampler(tp, 256, pre + budget, seed)
        fr, ok = [], True
        for i, bt in enumerate(s):
            bt = np.asarray(bt)
            if bt.size != 256 or len(set(bt.tolist())) != 256:
                ok = False
            if i >= pre:
                fr.append(float((bt < nl).mean()))
        res[arm] = {"L": L, "G": G, "expected": L / (L + G),
                    "realised_run_mean": float(np.mean(fr)),
                    "abs_diff": abs(float(np.mean(fr)) - L / (L + G)),
                    "n_batches_in_budget": budget, "pre_phase_batches": pre,
                    "all_batches_256_no_duplicates": ok}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=("dann", "bnm", "frozen_ids", "lens", "all"),
                    required=True)
    ap.add_argument("--fold", type=int, required=True, choices=(1, 2, 3, 4))
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dann-workers", type=int, default=2)
    ap.add_argument("--bnm-workers", type=int, default=2,
                    help="DataLoader workers for the descriptive BNM gradient-norm batch")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    if out.startswith(REPO + os.sep):
        raise SystemExit("--out must be outside the repo (a scratch folder)")
    os.makedirs(out, exist_ok=True)
    import integrity
    integrity.verify(where="(dilute10_step0_eval start)")
    todo = ("dann", "bnm", "frozen_ids", "lens") if a.what == "all" else (a.what,)
    fn = {"dann": eval_dann, "bnm": eval_bnm, "frozen_ids": eval_frozen_ids,
          "lens": eval_lens}
    for w in todo:
        dst = os.path.join(out, f"{w}_fold{a.fold}.json")
        if os.path.exists(dst) and not a.force:
            raise SystemExit(f"refusing to overwrite {dst}")
        t0 = time.time()
        r = (fn[w](a.fold, out, a.bnm_workers) if w == "bnm" else
             fn[w](a.fold, out, a.dann_workers) if w == "dann" else fn[w](a.fold, out))
        r["_script_sha256"] = DP.sha256_file(os.path.abspath(__file__))
        r["_seconds"] = time.time() - t0
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, default=lambda o: o.tolist()
                      if hasattr(o, "tolist") else str(o))
        print(f"[step0-eval] {w} fold {a.fold} -> {dst} ({r['_seconds']:.0f}s)", flush=True)
    integrity.verify(where="(dilute10_step0_eval end)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
