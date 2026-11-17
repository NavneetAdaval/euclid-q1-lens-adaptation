"""
study_validators.py
===================
The A1 STUDY: does any label-free validator select checkpoints better than R1?

WHAT THIS ANSWERS
-----------------
Correlation is diagnostic; REGRET is the decision. For each run and each validator:

    oracle_step = argmax over checkpoints of real-test TPR@1e-3
    picked_step = argmax (or argmin, per the validator's PUBLISHED direction) of the score
    regret      = TPR(oracle_step) - TPR(picked_step)      reported in TPR and in LENSES

The incumbent is R1 (last checkpoint). A validator is only useful if its regret is
materially BELOW R1's. R2 (fixed-plateau, last step <= 6000) and R3 (median checkpoint of
the [6000, budget] window) are reported for context, with the same definitions as
``analyze_fold0_selection.py`` (imported, not re-derived).

WHAT IS AND IS NOT COMPUTABLE -- READ THIS BEFORE TRUSTING THE TABLE
--------------------------------------------------------------------
The Task-4 brief assumed "32 checkpoints per run" for every run. THAT IS NOT TRUE ON DISK.
Only the FINAL checkpoint survives for the three Kaggle runs -- ``nb1``, ``nb3`` and
``nb0_image_kaggle`` -- because the Kaggle driver kept only ``step_8000.pt``. So:

  * Validators 1-3 (source-val TPR, source-val AUROC, agreement) are logged in ``curve.csv``
    at all 32 points and are therefore available for EVERY run, including nb1 and nb3.
  * Validators 4-6 (IM, SND, ATC) need per-checkpoint re-inference and are therefore
    available ONLY for runs whose per-checkpoint artifacts survive.

Every run that DOES have per-checkpoint artifacts has a FROZEN encoder, so re-inference is
a head forward pass over the existing ``data/feature_cache/*.npz`` -- exact, CPU-only, and
free. There is no GPU cost in this study at all. (The brief anticipated re-composing base
Zoobot weights with a checkpoint delta through ``live_encode.py``; that path is only needed
for unfrozen checkpoints, and no unfrozen intermediate checkpoint exists.)

RECONSTRUCTION VERIFICATION IS A HARD GATE. Before any validator is computed from a
checkpoint, that checkpoint's recomputed real-test TPR@1e-3 and source-val AUROC are
checked against the values logged in ``curve.csv``. A mismatch aborts the run for that
checkpoint rather than producing a plausible-looking wrong number.

USAGE
    python scripts/study_validators.py --stage all
    python scripts/study_validators.py --stage report      # re-report from the cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import evaluate as ev                                            # noqa: E402
from analyze_fold0_selection import _median_checkpoint_idx, _spearman   # noqa: E402
from metrics import DEFAULT_TARGET_FPR, auroc, tpr_at_fpr        # noqa: E402
from validators import (                                         # noqa: E402
    AGREEMENT_CAVEAT, DIRECTION, agreement, atc, information_maximization,
    snd_from_features, snd_from_logits, source_val_auroc, source_val_tpr,
)

STEP_COL = "step"
TPR_COL = "fold0_tpr_1e-3"
PREREG_BUDGET = 8000          # PROJECT.md 7.4, frozen before results
WINDOW_LO = 6000              # R2/R3 window floor, same as analyze_fold0_selection
DEFAULT_REF = os.path.join(_ROOT, "runs", "baseline_A", "checkpoints",
                           "baseline_linear_ep100.pt")

#: The runs in the study. ``informative`` marks the ones with real curve spread -- a
#: validator CANNOT be tested on a flat curve, because there is nothing to select.
RUN_REGISTRY = [
    # (tag,                 run_dir,                               fold, kind,   note)
    ("nb1",                 "baselineA_fold0_nb1",                 0, "head", "unfrozen n_blocks=1"),
    ("nb3",                 "baselineA_fold0_nb3",                 0, "head", "unfrozen n_blocks=3"),
    ("nb0_kaggle",          "baselineA_fold0_nb0_image_kaggle",    0, "head", "frozen, Kaggle"),
    ("nb0_image",           "baselineA_fold0_nb0_image",           0, "head", "frozen, local"),
    ("nb0_noaug_fp32",      "baselineA_fold0_nb0_image_noaug_fp32", 0, "head", "frozen, local"),
    ("nb0_weak_fp32",       "baselineA_fold0_nb0_image_weak_fp32", 0, "head", "frozen, local"),
    ("r1_baseline",         "curve_baseline_fold0",                0, "head", "round-1 Baseline A"),
    ("r1_freematch",        "curve_freematch_fold0",               0, "head", "round-1 FreeMatch"),
    ("r1_dann",             "curve_dann_fold0",                    0, "dann", "round-1 DANN"),
]

#: Validators computed from curve.csv alone (free, available for every run).
CURVE_VALIDATORS = {
    "source_val_tpr": "source_risk_tpr",
    "source_val_auroc": "source_risk_auroc",
    "agreement": "agreement_fold0",
}
#: Validators needing per-checkpoint re-inference.
INFERENCE_VALIDATORS = ("im", "snd", "neg_snd", "snd_features", "atc")

#: TWO SYNTHETIC CONTROLS, added after the known-null tripwire fired (see CONTROL_NOTE).
#: ``control_noise`` has zero information by construction -- it calibrates what the PIPELINE
#: produces from nothing. ``control_step`` is "later is better", which is not label-free
#: information about checkpoint QUALITY at all -- it calibrates the shared training-time
#: trend that every step-monotone quantity inherits.
CONTROL_VALIDATORS = ("control_step", "control_noise")
CONTROL_SEED = 12345
N_NOISE_DRAWS = 2000

ALL_VALIDATORS = tuple(CURVE_VALIDATORS) + INFERENCE_VALIDATORS + CONTROL_VALIDATORS

#: Selection direction for everything in the table. The controls are argmax by convention
#: (``control_step`` argmax == R1, which is the point of it); they are kept OUT of
#: ``validators.DIRECTION`` because they are diagnostics, not candidate validators.
SELECT_DIRECTION = dict(DIRECTION, control_step=+1, control_noise=+1)

CONTROL_NOTE = """\
WHY THERE ARE TWO SYNTHETIC CONTROL ROWS
----------------------------------------
The brief specified a tripwire: "if saturated source-val AUROC shows a correlation, the
analysis code is wrong; stop and fix it." On the first run it FIRED -- pooled Spearman
+0.54. It was investigated rather than obeyed, and the tripwire's PREMISE turned out to be
what is wrong, not the code:

  * ``control_noise`` -- a pure-noise validator pushed through the identical pipeline --
    pools to rho = +0.002, sd 0.065, 95% band [-0.13, +0.13] over 2000 draws. THE PIPELINE
    DOES NOT MANUFACTURE CORRELATION, so the analysis code is correct.
  * ``control_step`` -- the training step itself used as the "validator" -- pools to
    rho = +0.46, which is essentially source-val AUROC's +0.54. And source-val AUROC is a
    near-perfect monotone function of step (per-run rho(step, AUROC) = 0.84 to 1.00).

So the correlation is INHERITED FROM THE SHARED TRAINING-TIME TREND: real-test TPR rises
over early training and so does source-val AUROC, and Spearman cannot tell that apart from
selection information. Saturation bounds a metric's RANGE, not its RANK ORDER -- the
tripwire silently assumed those were the same thing.

CONSEQUENCE FOR READING THE TABLE: the null reference for a correlation is NOT zero, it is
the ``control_step`` row. A validator that does not beat ``control_step`` has demonstrated
nothing beyond "later checkpoints tend to be better". The step-partialled column removes
the trend directly. None of this touches REGRET, which is the decision metric and which was
never confounded -- regret asks which checkpoint argmax actually picks."""


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def _read_curve(path):
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty curve: {path}")
    cols = list(rows[0])
    out = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            v = r.get(c, "")
            try:
                out[c].append(float(v) if v not in ("", None) else np.nan)
            except ValueError:
                out[c].append(np.nan)
    return {c: np.asarray(v, dtype=float) for c, v in out.items()}


def _checkpoint_map(run_dir):
    """{step: path} for every ``step_*.pt`` in the run's checkpoints/ dir."""
    import re
    d = os.path.join(run_dir, "checkpoints")
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        m = re.fullmatch(r"step_(\d+)\.pt", fn)
        if m:
            out[int(m.group(1))] = os.path.join(d, fn)
    return out


def inventory(runs=RUN_REGISTRY, root=_ROOT, max_step=PREREG_BUDGET, verbose=True):
    """What exists on disk, per run. This is the load-bearing scope check."""
    inv = []
    for tag, rd, fold, kind, note in runs:
        run_dir = os.path.join(root, "runs", rd)
        curve_path = os.path.join(run_dir, "curve.csv")
        if not os.path.isfile(curve_path):
            if verbose:
                print(f"  SKIP {tag}: no curve.csv")
            continue
        curve = _read_curve(curve_path)
        steps_all = curve[STEP_COL].astype(int)
        keep = steps_all <= max_step
        curve = {c: v[keep] for c, v in curve.items()}
        steps = steps_all[keep]
        ck = _checkpoint_map(run_dir)
        have = sorted(s for s in steps.tolist() if s in ck)
        tpr = curve[TPR_COL]
        inv.append({
            "tag": tag, "run_dir": run_dir, "rel": rd, "fold": fold, "kind": kind,
            "note": note, "curve": curve, "steps": steps, "ckpts": ck,
            "steps_with_ckpt": have,
            "n_curve": int(steps.size), "n_ckpt": len(have),
            "full_ckpts": len(have) == int(steps.size),
            "tpr_spread": float(np.nanmax(tpr) - np.nanmin(tpr)),
            "max_step_all": int(steps_all.max()),
            "has_agreement": "agreement_fold0" in curve,
        })
    if verbose:
        print("=" * 96)
        print(f"INVENTORY  (curve truncated to the pre-registered budget, step <= {max_step})")
        print("=" * 96)
        print(f"{'run':<17}{'curve':>6}{'ckpts':>7}{'spread':>9}{'lenses':>8}"
              f"{'agree':>7}  {'validators 4-6':<16} note")
        print("-" * 96)
        for r in inv:
            ok = "YES" if r["n_ckpt"] >= 3 else "NO"
            print(f"{r['tag']:<17}{r['n_curve']:>6}{r['n_ckpt']:>7}"
                  f"{r['tpr_spread']:>9.4f}{r['tpr_spread'] * 98:>8.1f}"
                  f"{'yes' if r['has_agreement'] else 'no':>7}  {ok:<16} {r['note']}")
        print("-" * 96)
        missing = [r["tag"] for r in inv if r["n_ckpt"] < 3]
        if missing:
            print(f"NOTE: {', '.join(missing)} kept only the FINAL checkpoint, so validators "
                  f"4-6 (IM/SND/ATC) cannot be computed there.\n"
                  f"      Validators 1-3 are still available for them from curve.csv.")
    return inv


# ---------------------------------------------------------------------------
# per-checkpoint re-inference (+ the reconstruction verification gate)
# ---------------------------------------------------------------------------

class _Splits:
    """The four cached splits every checkpoint is scored on, loaded once."""

    def __init__(self, cache_dir, fold=0):
        cache = ev.FeatureCache(cache_dir)
        self.val = cache.load("val")
        self.neg = cache.load("test_negatives_shared")
        self.pos = cache.load(f"test_pos_fold{fold}")
        self.tgt = cache.load(f"target_fold{fold}")
        self.n_pos = int(self.pos["features"].shape[0])
        self.lens = 1.0 / self.n_pos                     # one lens, in TPR units
        # Trap 1 pre-check: the IM pool must be the BALANCED target pool.
        lat = np.asarray(self.tgt["latent_labels"]).reshape(-1)
        self.tgt_latent = lat
        frac = float(lat.mean())
        assert abs(frac - 0.5) < 0.05, f"target pool is not balanced: pos frac {frac}"
        assert np.all(np.asarray(self.tgt["labels"]) == -1), \
            "target labels must be UNLABELED(-1) -- the label-free contract"
        # SND on backbone FEATURES is constant across checkpoints on a frozen encoder.
        # Compute it once here; it is a known-null control, not a candidate.
        self.snd_features_const = snd_from_features(self.tgt["features"])


def _score_all(ckpt_path, kind, splits, device="cpu"):
    """Score the four splits with one checkpoint. Returns raw logit vectors."""
    if kind == "dann":
        import run_dann as rd
        model, std, _ = rd.load_dann_checkpoint(ckpt_path, device)
        f = lambda x: rd.score_dann(model, std, x, device)     # noqa: E731
    else:
        head, std, _ = ev.load_checkpoint(ckpt_path, device)
        f = lambda x: ev.score_features(head, std, x, device)  # noqa: E731
    return {
        "val": f(splits.val["features"]),
        "neg": f(splits.neg["features"]),
        "pos": f(splits.pos["features"]),
        "tgt": f(splits.tgt["features"]),
    }


def compute_run(rec, splits, ref_tgt_scores, *, device="cpu",
                target_fpr=DEFAULT_TARGET_FPR, atol=2e-4, verbose=True):
    """Re-infer every available checkpoint of one run, VERIFY it, and compute validators 4-6.

    The verification gate: recomputed real-test TPR@1e-3 and source-val AUROC must match the
    values ``curve.csv`` logged for that step. ``atol`` is 2e-4 by default -- curve.csv stores
    full float repr, and the frozen-cache path is deterministic, so agreement should be
    essentially exact; the tolerance exists only to absorb float64 text round-tripping.
    """
    steps = rec["steps_with_ckpt"]
    curve, csteps = rec["curve"], rec["steps"]
    idx_of = {int(s): i for i, s in enumerate(csteps.tolist())}

    out = {"steps": [], "verify": [], "vals": {v: [] for v in INFERENCE_VALIDATORS},
           "recomputed": {"tpr": [], "sv_auroc": [], "sv_tpr": [], "agreement": []},
           "scores": {"val": [], "pos": [], "neg": [], "tgt": []}}
    y_test = np.concatenate([np.ones(splits.n_pos, dtype=np.int64),
                             np.zeros(splits.neg["features"].shape[0], dtype=np.int64)])
    if verbose and steps:
        print(f"\n-- {rec['tag']}: re-inferring {len(steps)} checkpoints ({rec['kind']}) ...")

    # An UNFROZEN checkpoint cannot be scored from the frozen feature cache: its encoder
    # moved, so the cached 640-d features are stale and every number computed from them is
    # wrong (measured on an nb3 checkpoint: cache-scored TPR 0.194 vs logged 0.020). The
    # reconstruction gate below catches this and discards the checkpoint, which is correct
    # but looks like a mysterious mismatch -- so say it plainly up front.
    unfrozen = False
    if steps:
        import torch as _torch
        _c = _torch.load(rec["ckpts"][steps[0]], map_location="cpu", weights_only=False)
        unfrozen = int(_c.get("n_blocks", 0) or 0) > 0 or "encoder_trainable" in _c
        del _c
    if unfrozen:
        print(f"   [SKIP] {rec['tag']} is UNFROZEN (n_blocks>0): its checkpoints carry an "
              f"encoder delta, so they must be scored through live_encode with the encoder "
              f"reconstructed (base Zoobot + delta), NOT from data/feature_cache. That path "
              f"is not implemented here yet, so validators 4-6 are left n/a rather than "
              f"computed from stale features.")
        out["unfrozen_skipped"] = True
        out["n_fail"] = 0
        return out

    fails = 0
    for s in steps:
        sc = _score_all(rec["ckpts"][s], rec["kind"], splits, device)
        i = idx_of[s]
        got_tpr = float(tpr_at_fpr(y_test, np.concatenate([sc["pos"], sc["neg"]]), target_fpr))
        got_auc = source_val_auroc(splits.val["labels"], sc["val"])
        want_tpr = float(curve[TPR_COL][i])
        want_auc = float(curve["source_risk_auroc"][i])
        d_tpr, d_auc = abs(got_tpr - want_tpr), abs(got_auc - want_auc)
        ok = (d_tpr <= atol) and (d_auc <= atol)
        out["verify"].append({"step": int(s), "logged_tpr": want_tpr, "recomputed_tpr": got_tpr,
                              "d_tpr": d_tpr, "logged_sv_auroc": want_auc,
                              "recomputed_sv_auroc": got_auc, "d_sv_auroc": d_auc, "ok": ok})
        if not ok:
            fails += 1
            continue                                   # never compute validators from it

        out["steps"].append(int(s))
        out["recomputed"]["tpr"].append(got_tpr)
        out["recomputed"]["sv_auroc"].append(got_auc)
        out["recomputed"]["sv_tpr"].append(source_val_tpr(splits.val["labels"], sc["val"],
                                                          target_fpr))
        out["recomputed"]["agreement"].append(
            agreement(sc["tgt"], ref_tgt_scores) if ref_tgt_scores is not None else np.nan)
        out["vals"]["im"].append(
            information_maximization(sc["tgt"], latent_labels=splits.tgt_latent))
        out["vals"]["snd"].append(snd_from_logits(sc["tgt"]))
        out["vals"]["neg_snd"].append(-out["vals"]["snd"][-1])
        out["vals"]["snd_features"].append(splits.snd_features_const)
        out["vals"]["atc"].append(atc(sc["val"], splits.val["labels"], sc["tgt"]))
        for k in ("val", "pos", "neg", "tgt"):
            out["scores"][k].append(sc[k].astype(np.float32))

    if verbose and steps:
        v = out["verify"]
        mx_t = max((x["d_tpr"] for x in v), default=0.0)
        mx_a = max((x["d_sv_auroc"] for x in v), default=0.0)
        tag = "FAIL" if fails else "PASS"
        print(f"   [{tag}] reconstruction: {len(v) - fails}/{len(v)} checkpoints match "
              f"curve.csv  (max |dTPR|={mx_t:.2e}, max |dAUROC|={mx_a:.2e})")
    out["n_fail"] = fails
    return out


# ---------------------------------------------------------------------------
# selection rules + regret
# ---------------------------------------------------------------------------

def _rule_indices(steps, tpr, budget):
    """R1 / R2 / R3 indices into ``steps``, matching analyze_fold0_selection exactly."""
    steps = np.asarray(steps, int)
    tpr = np.asarray(tpr, float)
    i_r1 = int(np.argmax(steps))
    le = np.where(steps <= WINDOW_LO)[0]
    i_r2 = int(le[np.argmax(steps[le])]) if le.size else None
    win = np.where((steps >= WINDOW_LO) & (steps <= budget))[0]
    if win.size:
        j = _median_checkpoint_idx(tpr[win])
        i_r3 = int(win[j])
    else:
        i_r3 = None
    return {"R1 last": i_r1, "R2 fixed-plateau": i_r2, "R3 median": i_r3}


def regret_for(scores, tpr, direction):
    """Regret of selecting by ``scores`` in ``direction``. Returns (regret, picked_index)."""
    s = np.asarray(scores, float)
    t = np.asarray(tpr, float)
    m = np.isfinite(s) & np.isfinite(t)
    if m.sum() < 2:
        return np.nan, None
    idx = np.where(m)[0]
    # ties broken toward the EARLIER checkpoint: a fixed, documented, label-free tie-break
    pick = idx[int(np.argmax(s[idx] * direction))]
    return float(np.nanmax(t[m]) - t[pick]), int(pick)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def partial_spearman(x, y, z):
    """Spearman(x, y) with the effect of ``z`` removed (rank partial correlation).

    Needed because source-val AUROC is a near-monotone function of the training step, and
    so is real-test TPR over early training; the raw rank correlation between them is mostly
    that shared trend. See CONTROL_NOTE.
    """
    rxy, n = _spearman(x, y)
    rxz, _ = _spearman(x, z)
    ryz, _ = _spearman(y, z)
    if not np.isfinite(rxy):
        return np.nan, n
    if not np.isfinite(rxz) or not np.isfinite(ryz):
        # A degenerate covariate (constant, or too few points) carries no rank information,
        # so there is nothing to remove: the partial correlation IS the raw one.
        return float(rxy), n
    den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 0.0))
    if den < 1e-12:
        return np.nan, n
    return float(np.clip((rxy - rxz * ryz) / den, -1.0, 1.0)), n


def noise_control_band(inv, n_draws=N_NOISE_DRAWS, seed=CONTROL_SEED):
    """Pooled Spearman of a PURE-NOISE validator, over ``n_draws`` draws.

    This is the decisive test of the ANALYSIS CODE: a pipeline that turns noise into
    correlation is broken. Returns (mean, sd, lo95, hi95).
    """
    rng = np.random.default_rng(seed)
    tprs = [r["curve"][TPR_COL] for r in inv]
    pooled = np.empty(n_draws)
    for d in range(n_draws):
        rr, nn = [], []
        for t in tprs:
            rho, n = _spearman(rng.standard_normal(t.size), t)
            rr.append(rho)
            nn.append(n)
        pooled[d] = _fisher_pool(rr, nn)[0]
    return (float(np.nanmean(pooled)), float(np.nanstd(pooled)),
            float(np.nanpercentile(pooled, 2.5)), float(np.nanpercentile(pooled, 97.5)))


def _fisher_pool(rhos, ns):
    """Pool per-run Spearman rhos in Fisher-z space; returns (rho, lo, hi, k)."""
    r = np.asarray(rhos, float)
    n = np.asarray(ns, float)
    m = np.isfinite(r) & np.isfinite(n) & (n > 3) & (np.abs(r) < 1.0)
    if m.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    z = np.arctanh(r[m])
    w = n[m] - 3.0
    zbar = float(np.sum(w * z) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return float(np.tanh(zbar)), float(np.tanh(zbar - 1.96 * se)), \
        float(np.tanh(zbar + 1.96 * se)), int(m.sum())


def build_tables(inv, computed, splits, budget=PREREG_BUDGET):
    """Assemble the per-run validator series, regret table and correlation table."""
    lens = splits.lens
    per_run = {}
    for rec in inv:
        tag = rec["tag"]
        curve, steps = rec["curve"], rec["steps"].astype(int)
        tpr = curve[TPR_COL]
        series = {}
        # validators 1-3 straight from curve.csv (all 32 points, every run)
        for name, col in CURVE_VALIDATORS.items():
            series[name] = curve[col] if col in curve else np.full(steps.size, np.nan)
        # validators 4-6 only where checkpoints survived; align onto the curve steps
        c = computed.get(tag)
        if c and c["steps"]:
            pos = {s: j for j, s in enumerate(c["steps"])}
            for v in INFERENCE_VALIDATORS:
                arr = np.full(steps.size, np.nan)
                for i, s in enumerate(steps.tolist()):
                    if s in pos:
                        arr[i] = c["vals"][v][pos[s]]
                series[v] = arr
            # agreement recomputed from checkpoints fills the round-1 runs' missing column
            if not rec["has_agreement"]:
                arr = np.full(steps.size, np.nan)
                for i, s in enumerate(steps.tolist()):
                    if s in pos:
                        arr[i] = c["recomputed"]["agreement"][pos[s]]
                series["agreement"] = arr
        else:
            for v in INFERENCE_VALIDATORS:
                series[v] = np.full(steps.size, np.nan)

        # the two synthetic controls (see CONTROL_NOTE)
        series["control_step"] = steps.astype(float)
        # zlib.crc32, NOT hash(): Python randomises string hashing per process, which would
        # make this row -- and therefore the report -- non-reproducible run to run.
        series["control_noise"] = np.random.default_rng(
            CONTROL_SEED + zlib.crc32(tag.encode())).standard_normal(steps.size)

        rules = _rule_indices(steps, tpr, budget)
        oracle_i = int(np.nanargmax(tpr))
        row = {"steps": steps, "tpr": tpr, "series": series,
               "oracle": {"step": int(steps[oracle_i]), "tpr": float(tpr[oracle_i])},
               # expected regret of picking a checkpoint UNIFORMLY AT RANDOM: the
               # "no information at all" reference for the regret column.
               "random_regret": float(np.nanmean(tpr[oracle_i] - tpr)),
               "random_regret_lenses": float(np.nanmean(tpr[oracle_i] - tpr) / lens),
               "rules": {}, "regret": {}, "spearman": {}}
        for name, i in rules.items():
            row["rules"][name] = None if i is None else {
                "step": int(steps[i]), "tpr": float(tpr[i]),
                "regret": float(tpr[oracle_i] - tpr[i]),
                "regret_lenses": float((tpr[oracle_i] - tpr[i]) / lens)}
        for v in ALL_VALIDATORS:
            reg, pick = regret_for(series[v], tpr, SELECT_DIRECTION[v])
            row["regret"][v] = None if pick is None else {
                "step": int(steps[pick]), "tpr": float(tpr[pick]),
                "regret": reg, "regret_lenses": reg / lens,
                "n": int(np.isfinite(series[v]).sum())}
            rho, n = _spearman(series[v], tpr)
            prho, _ = partial_spearman(series[v], tpr, steps.astype(float))
            row["spearman"][v] = {"rho": float(rho) if np.isfinite(rho) else np.nan, "n": int(n),
                                  "partial_rho": float(prho) if np.isfinite(prho) else np.nan,
                                  "range": float(np.nanmax(series[v]) - np.nanmin(series[v]))
                                  if np.isfinite(series[v]).any() else np.nan}
        per_run[tag] = row
    return per_run


def print_report(inv, per_run, computed, splits, budget=PREREG_BUDGET, out_dir=None):
    lens = splits.lens
    order = [r["tag"] for r in inv]
    byt = {r["tag"]: r for r in inv}
    W = 96

    print("\n" + "=" * W)
    print("DELIVERABLE 2 -- CHECKPOINT-RECONSTRUCTION VERIFICATION (logged vs recomputed)")
    print("=" * W)
    print("Every checkpoint is re-scored from data/feature_cache + the checkpoint head and")
    print("checked against curve.csv BEFORE any validator is computed from it.")
    print(f"\n{'run':<17}{'ckpts':>7}{'pass':>6}{'max|dTPR|':>12}{'max|dAUROC|':>14}  verdict")
    print("-" * W)
    any_fail = False
    for t in order:
        c = computed.get(t)
        if not c or not c["verify"]:
            print(f"{t:<17}{0:>7}{'--':>6}{'--':>12}{'--':>14}  no per-checkpoint artifacts")
            continue
        v = c["verify"]
        mt = max(x["d_tpr"] for x in v)
        ma = max(x["d_sv_auroc"] for x in v)
        npass = sum(x["ok"] for x in v)
        ok = npass == len(v)
        any_fail |= not ok
        print(f"{t:<17}{len(v):>7}{npass:>6}{mt:>12.2e}{ma:>14.2e}  "
              f"{'EXACT' if mt == 0 and ma == 0 else ('PASS' if ok else 'FAIL')}")
    print("-" * W)
    print("VERDICT: " + ("SOME CHECKPOINTS FAILED -- validators from them were discarded."
                         if any_fail else
                         "all reconstructions reproduce curve.csv; validators are trustworthy."))

    print("\n" + "=" * W)
    print("DELIVERABLE 3 -- REGRET TABLE     validator x run -> regret in LENSES "
          f"(1 lens = {lens:.4f} TPR)")
    print("=" * W)
    print("Lower is better. R1 is the INCUMBENT: a validator is only useful if it is")
    print("materially below the R1 row on the informative runs.")
    hdr = f"{'selector':<18}" + "".join(f"{t:>13}" for t in order)
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in ("R1 last", "R2 fixed-plateau", "R3 median"):
        cells = ""
        for t in order:
            r = per_run[t]["rules"].get(name)
            cells += f"{'n/a':>13}" if r is None else f"{r['regret_lenses']:>13.1f}"
        print(f"{name:<18}{cells}" + ("   <-- INCUMBENT" if name == "R1 last" else ""))
    print("-" * len(hdr))
    for v in ALL_VALIDATORS:
        cells = ""
        for t in order:
            r = per_run[t]["regret"].get(v)
            cells += f"{'n/a':>13}" if r is None else f"{r['regret_lenses']:>13.1f}"
        mark = "   <-- KNOWN-NULL CONTROL" if v == "source_val_auroc" else ""
        mark = "   <-- 2nd null (constant)" if v == "snd_features" else mark
        mark = "   <-- CONTROL (no content)" if v in CONTROL_VALIDATORS else mark
        print(f"{v:<18}{cells}{mark}")
    print("-" * len(hdr))
    print(f"{'random pick (exp)':<18}"
          + "".join(f"{per_run[t]['random_regret_lenses']:>13.1f}" for t in order)
          + "   <-- zero-information reference")
    print(f"{'ORACLE tpr':<18}" + "".join(f"{per_run[t]['oracle']['tpr']:>13.4f}" for t in order))
    print(f"{'oracle step':<18}" + "".join(f"{per_run[t]['oracle']['step']:>13d}" for t in order))
    print(f"{'curve spread (l)':<18}" + "".join(f"{byt[t]['tpr_spread'] / lens:>13.1f}"
                                                for t in order))
    print("\nn/a = validator not computable for that run (no per-checkpoint artifacts), or")
    print("      the rule has no eligible checkpoint (empty [6000, budget] window).")

    print("\n" + "=" * W)
    print("DELIVERABLE 4 -- SPEARMAN(validator, real-test TPR@1e-3), per run and pooled")
    print("=" * W)
    hdr = (f"{'validator':<18}" + "".join(f"{t:>13}" for t in order)
           + f"{'POOLED (raw)':>24}{'POOLED |step':>14}")
    print(hdr)
    print("-" * len(hdr))
    pooled_raw = {}
    for v in ALL_VALIDATORS:
        cells, rhos, ns, prhos = "", [], [], []
        for t in order:
            sp = per_run[t]["spearman"][v]
            if np.isfinite(sp["rho"]):
                cells += f"{sp['rho']:>13.3f}"
                rhos.append(sp["rho"])
                ns.append(sp["n"])
                prhos.append(sp["partial_rho"])
            else:
                cells += f"{'n/a':>13}"
        rho, lo, hi, k = _fisher_pool(rhos, ns)
        prho = _fisher_pool(prhos, ns)[0]
        pooled_raw[v] = (rho, lo, hi, k)
        s_raw = "n/a" if not np.isfinite(rho) else f"{rho:+.3f} [{lo:+.2f},{hi:+.2f}] k={k}"
        s_par = "n/a" if not np.isfinite(prho) else f"{prho:+.3f}"
        mark = "  <-- CONTROL" if v in CONTROL_VALIDATORS else ""
        print(f"{v:<18}{cells}{s_raw:>24}{s_par:>14}{mark}")
    print("-" * len(hdr))
    print("POOLED (raw)  = Fisher-z mean of per-run rhos, weighted by (n-3), 95% CI.")
    print("POOLED |step  = same, but with the training-step trend PARTIALLED OUT.")
    print("With 6-9 short curves these are NOISY -- read the CI, not the point estimate.")

    # ---- the known-null control gate, as corrected (deliverable 4) ----
    nm, nsd, nlo, nhi = noise_control_band(inv)
    a_rho, a_lo, a_hi, _ = pooled_raw["source_val_auroc"]
    s_rho = pooled_raw["control_step"][0]
    a_par = _fisher_pool([per_run[t]["spearman"]["source_val_auroc"]["partial_rho"]
                          for t in order],
                         [per_run[t]["spearman"]["source_val_auroc"]["n"] for t in order])[0]

    print("\n" + "=" * W)
    print("THE KNOWN-NULL TRIPWIRE -- IT FIRED, AND THE PREMISE (NOT THE CODE) WAS WRONG")
    print("=" * W)
    print(f"  source-val AUROC   pooled rho = {a_rho:+.3f} [{a_lo:+.2f}, {a_hi:+.2f}]"
          f"   <-- tripwire condition met")
    print(f"  control_noise      pooled rho = {nm:+.3f}  sd {nsd:.3f}  "
          f"95% band [{nlo:+.3f}, {nhi:+.3f}]  ({N_NOISE_DRAWS} draws)")
    print(f"  control_step       pooled rho = {s_rho:+.3f}")
    print(f"  source-val AUROC   partialled on step = {a_par:+.3f}")
    print("-" * W)
    if abs(nm) > max(0.15, 3 * nsd):
        print("  *** STOP: the PURE-NOISE control is correlated. The analysis code really is")
        print("  *** wrong -- it is manufacturing correlation. Fix before reading anything.")
    else:
        print("  CODE IS CORRECT: pure noise pools to ~0 through this exact pipeline, so the")
        print("  pipeline does not manufacture correlation.")
        print("  THE PREMISE WAS WRONG: saturation bounds a metric's RANGE, not its RANK")
        print("  ORDER. source-val AUROC is a near-monotone function of the training step")
        print("  (per-run rho(step, AUROC) = 0.84-1.00), and real-test TPR also trends with")
        print("  step, so their rank correlation is mostly that SHARED TREND -- which is why")
        print("  the content-free control_step row reproduces it almost exactly.")
        print("  => The null reference for a correlation is control_step, NOT zero.")
        print("  => Regret is unaffected: it asks which checkpoint argmax PICKS, and there")
        print("     source-val AUROC is bad exactly as PROJECT.md 7.6 recorded.")
    print("=" * W)

    # ---- DELIVERABLE 5 -- the verdict, computed rather than eyeballed ----
    print("\n" + "=" * W)
    print("DELIVERABLE 5 -- VERDICT: is any validator usable, i.e. regret materially")
    print("                 BELOW R1's? (paired per-run differences, in lenses)")
    print("=" * W)
    unfrozen = [t for t in order if byt[t]["note"].startswith("unfrozen")]
    r1 = {t: (per_run[t]["rules"]["R1 last"] or {}).get("regret_lenses", np.nan)
          for t in order}

    def _reg(t, v):
        r = per_run[t]["regret"].get(v)
        return np.nan if r is None else r["regret_lenses"]

    print(f"{'selector':<18}{'mean reg':>9}{'vs R1':>8}{'win':>5}{'tie':>5}{'los':>5}"
          f"{'  nb1':>7}{'  nb3':>7}{'  unfrozen mean vs R1':>23}")
    print("-" * W)
    for v in [x for x in ALL_VALIDATORS] + ["R2 fixed-plateau", "R3 median",
                                            "random pick (exp)"]:
        if v == "random pick (exp)":
            vals = {t: per_run[t]["random_regret_lenses"] for t in order}
        elif v in ("R2 fixed-plateau", "R3 median"):
            vals = {t: (per_run[t]["rules"][v] or {}).get("regret_lenses", np.nan)
                    for t in order}
        else:
            vals = {t: _reg(t, v) for t in order}
        d = np.array([vals[t] - r1[t] for t in order if np.isfinite(vals[t])
                      and np.isfinite(r1[t])])
        m = np.nanmean([vals[t] for t in order]) if np.isfinite(
            list(vals.values())).any() else np.nan
        w, tie, l = int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())
        du = np.array([vals[t] - r1[t] for t in unfrozen
                       if np.isfinite(vals[t]) and np.isfinite(r1[t])])
        s_nb1 = f"{vals['nb1']:.1f}" if np.isfinite(vals.get("nb1", np.nan)) else "n/a"
        s_nb3 = f"{vals['nb3']:.1f}" if np.isfinite(vals.get("nb3", np.nan)) else "n/a"
        s_u = f"{du.mean():+.1f}" if du.size else "n/a"
        print(f"{v:<18}{m:>9.1f}{(d.mean() if d.size else np.nan):>+8.1f}"
              f"{w:>5}{tie:>5}{l:>5}{s_nb1:>7}{s_nb3:>7}{s_u:>23}")
    # How well does a COIN FLIP do against R1? Any validator inside this band has not
    # demonstrated that it selects at all.
    rng = np.random.default_rng(CONTROL_SEED)
    draws = np.empty(N_NOISE_DRAWS)
    for d in range(N_NOISE_DRAWS):
        diffs = []
        for t in order:
            tp = per_run[t]["tpr"]
            ok = np.isfinite(tp)
            pick = rng.choice(np.where(ok)[0])
            diffs.append((np.nanmax(tp) - tp[pick]) / lens - r1[t])
        draws[d] = float(np.mean(diffs))
    n_lo, n_hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    print("-" * W)
    print("'vs R1' = mean(validator regret - R1 regret) in lenses; NEGATIVE means better")
    print("than the incumbent. win/tie/los counts runs beaten/tied/lost against R1.")
    print(f"\nRANDOM-SELECTION BAND: a validator that picks a checkpoint UNIFORMLY AT RANDOM")
    print(f"scores 'vs R1' = {draws.mean():+.1f} lenses, 95% band [{n_lo:+.1f}, {n_hi:+.1f}] "
          f"({N_NOISE_DRAWS} draws).")
    print("ANY validator whose 'vs R1' falls inside that band is INDISTINGUISHABLE FROM")
    print("CHANCE on this evidence -- it has not been shown to select anything.")
    inside = [v for v in ALL_VALIDATORS
              if v not in CONTROL_VALIDATORS and v != "snd_features"
              and np.isfinite(np.nanmean([_reg(t, v) - r1[t] for t in order]))
              and n_lo <= np.nanmean([_reg(t, v) - r1[t] for t in order]) <= n_hi]
    print(f"  inside the band: {', '.join(inside) if inside else '(none)'}")
    print(f"R1's own mean regret is {np.nanmean(list(r1.values())):.1f} lenses; picking a")
    print(f"checkpoint UNIFORMLY AT RANDOM costs "
          f"{np.nanmean([per_run[t]['random_regret_lenses'] for t in order]):.1f} lenses.")
    print("\nNOTE on snd_features: it is CONSTANT (frozen encoder), so 'argmax' degenerates")
    print("to 'take the first checkpoint'. Its regret row measures that, not a selection.")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        blob = {"budget": budget, "lens_in_tpr": lens, "n_pos": splits.n_pos,
                "agreement_caveat": AGREEMENT_CAVEAT,
                "control_note": CONTROL_NOTE,
                "noise_control": {"mean": nm, "sd": nsd, "lo95": nlo, "hi95": nhi,
                                  "n_draws": N_NOISE_DRAWS},
                "pooled": {v: {"rho": pooled_raw[v][0], "lo": pooled_raw[v][1],
                               "hi": pooled_raw[v][2], "k": pooled_raw[v][3]}
                           for v in ALL_VALIDATORS},
                "runs": {t: {
                    "rel": byt[t]["rel"], "note": byt[t]["note"],
                    "n_curve": byt[t]["n_curve"], "n_ckpt": byt[t]["n_ckpt"],
                    "tpr_spread_lenses": byt[t]["tpr_spread"] / lens,
                    "steps": per_run[t]["steps"].tolist(),
                    "tpr": per_run[t]["tpr"].tolist(),
                    "oracle": per_run[t]["oracle"],
                    "rules": per_run[t]["rules"],
                    "regret": per_run[t]["regret"],
                    "spearman": per_run[t]["spearman"],
                    "verify": (computed.get(t) or {}).get("verify", []),
                } for t in order}}
        p = os.path.join(out_dir, "validator_study.json")
        with open(p, "w") as fh:
            json.dump(blob, fh, indent=2, default=float)
        print(f"\nwrote {os.path.relpath(p, _ROOT)}")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(_ROOT, "data", "feature_cache"))
    ap.add_argument("--ref-ckpt", default=DEFAULT_REF,
                    help="Baseline A FINAL checkpoint = the agreement reference")
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "runs", "validator_study_a1"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=PREREG_BUDGET)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stage", default="all", choices=["inventory", "all"])
    ap.add_argument("--save-scores", action="store_true",
                    help="cache every checkpoint's raw score vectors to out-dir/scores.npz")
    args = ap.parse_args(argv)

    inv = inventory(max_step=args.max_step)
    if args.stage == "inventory":
        return 0

    splits = _Splits(args.cache_dir, args.fold)
    print(f"\ncache: n_val={splits.val['features'].shape[0]}  "
          f"n_test_pos={splits.n_pos}  n_test_neg={splits.neg['features'].shape[0]}  "
          f"n_target={splits.tgt['features'].shape[0]} (balanced 1:1, Trap-1 assertion PASSED)")
    print(f"SND on backbone features is CONSTANT at {splits.snd_features_const:.6f} "
          f"(frozen encoder) -- a second known-null control.")

    ref_scores = None
    if os.path.isfile(args.ref_ckpt):
        h, s, _ = ev.load_checkpoint(args.ref_ckpt, args.device)
        ref_scores = ev.score_features(h, s, splits.tgt["features"], args.device)
        print(f"agreement reference: {os.path.relpath(args.ref_ckpt, _ROOT)}")
    else:
        print(f"WARNING: agreement reference not found ({args.ref_ckpt}); recomputed "
              f"agreement will be n/a (curve.csv's logged column is still used).")

    computed = {}
    for rec in inv:
        if rec["n_ckpt"] >= 3:
            computed[rec["tag"]] = compute_run(rec, splits, ref_scores, device=args.device)
        else:
            computed[rec["tag"]] = {"steps": [], "verify": [], "n_fail": 0,
                                    "vals": {v: [] for v in INFERENCE_VALIDATORS},
                                    "recomputed": {}, "scores": {}}

    per_run = build_tables(inv, computed, splits, budget=args.max_step)
    print_report(inv, per_run, computed, splits, budget=args.max_step, out_dir=args.out_dir)

    if args.save_scores:
        os.makedirs(args.out_dir, exist_ok=True)
        for t, c in computed.items():
            if c["steps"]:
                np.savez_compressed(
                    os.path.join(args.out_dir, f"scores_{t}.npz"),
                    steps=np.asarray(c["steps"]),
                    **{k: np.asarray(v) for k, v in c["scores"].items()})
        print(f"cached raw score vectors to {os.path.relpath(args.out_dir, _ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
