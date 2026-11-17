"""
morph_proxy.py
==============
THE MORPHOLOGY-PROXY AUROC MONITOR -- recovered, not invented.

It answers one question, every monitor step: **can the model still tell one kind of
real Euclid galaxy from another?** -- asked twice, of two different things (see TWO
INSTRUMENTS below). It is a recovery of the instrument family that caught the v5
adversarial-DA failure (``PROJECT_STATE v2.md`` 6 / 8-F4: the trained HEAD's
morphology AUROC fell ~0.61 -> ~0.37 under plain DANN and ~0.33 under IW-CDAN, while
the frozen-feature probe "stayed ~0.90"), and it arms round 5's abort condition 3.

-------------------------------------------------------------------------------
PROVENANCE -- WHAT THIS IS A RECOVERY OF
-------------------------------------------------------------------------------
PROJECT_STATE.md records two things that pin the instrument exactly:

    scripts/lens_dataset.py ... role `morphology_probe` (ML-FP=1 vs P1-reserve=0,
    both real).
    scripts/linear_probe.py -- checkpoint selection by (PR effective rank on P2) +
    (morphology AUROC ML-FP vs reserve).

and the number it produced:

    generic SimSiam SSL ... slightly *hurt* vs raw Zoobot (morphAUROC 0.898 vs 0.914).

**Both `linear_probe.py` and the `morphology_probe` role are GONE** -- the role was
dropped when the splits moved to `splits_v6_label_free_da`, and neither
`linear_probe.py` nor `encoder_probe.py` survives in `scripts/`. What survives is
everything needed to rebuild it exactly:

  * **ML-FP (label 1)** -- all 4,455 hard negatives in
    ``data/ml_fp_cutouts/shards/*.npz``: objects a prior lens-finding CNN flagged as
    lenses and experts graded X. Arc-mimics: chance alignments, ring galaxies
    (PROJECT.md 3.4).
  * **P1-reserve (label 0)** -- the partition-1 random Euclid galaxies that
    ``source_train.neg_random`` did NOT consume. Measured from the current splits:
    **39,824 in P1, 16,664 used, 23,160 in reserve.** That residual IS the
    "P1-reserve" the old instrument named.

So this module is a RECOVERY of the historical instrument on the surviving data. What
is NOT recoverable is the exact sample, seed and probe hyperparameters of the run that
printed 0.914, so the step-0 readings here are reported on their own terms.

**No absolute threshold is hardcoded anywhere in this file.** The design doc's 0.55
sat just under the HEAD instrument's real ~0.61 baseline; it is withdrawn because it
is meaningless on the frozen-probe column, which reads ~0.92, not because it was
invented. Brief v2's 0.6910 rescaling of it is withdrawn too: rescaling a threshold
across a change of instrument assumes the two measure the same thing, and they do
not. Abort condition 3's margin is declared off the lambda=0 control's own head-column
trajectory and hashed before the development arm runs (PROJECT.md 7.17).

-------------------------------------------------------------------------------
LABEL-FREE BY CONSTRUCTION
-------------------------------------------------------------------------------
Neither class is a lens label. ML-FP are objects graded **X = not a lens**;
P1-reserve are random galaxies with no grade at all. Both are real Euclid VIS
cutouts. No simulated image, no real-lens label and no test-split image is touched.
The 489 real lenses appear nowhere in this file.

The reserve is also disjoint from every training and evaluation role by construction:
P1's used remainder goes to ``source_train`` negatives, P2 to the per-fold target, P3
to val negatives, P4 to the shared test negatives (PROJECT.md 4). The reserve is the
part of P1 nothing else uses. **The ML-FP side is not disjoint** -- all 4,455 ML-FP
are source_train negatives -- and that is faithful to the original instrument, which
had the same property, but it is stated here rather than discovered later.

-------------------------------------------------------------------------------
THE PROBE, AND WHY IT IS BUILT THIS WAY
-------------------------------------------------------------------------------
Fixed pool, drawn once, deterministic given the seed. Each monitor step:

    encode (no_grad, current encoder) -> standardizer -> head.trunk adapter
      -> z-score using the FIT half's statistics
      -> ridge closed form on the FIT half
      -> AUROC on the held-out EVAL half

Four choices worth defending:

1. **Held-out evaluation is not optional.** 640 features against a few hundred
   samples is separable by construction; a probe fitted and scored on the same rows
   returns ~1.0 forever and monitors nothing.
2. **Z-scoring on the fit half makes the probe SCALE-INVARIANT.** Round 4 measured a
   56x feature-scale collapse (7.15). Without this, a shrinking representation would
   drag the morphology reading down for a reason that has nothing to do with
   morphology -- and there are already separate monitors for scale
   (``*_perdim_std``) and rank (``participation_ratio``). Each monitor should detect
   ONE thing; 7.16's standing instruction is exactly that.
3. **Ridge in closed form, alpha DECLARED at 1.0 on z-scored features, never tuned.**
   One 641x641 solve, ~10 ms, no optimiser, no iteration count, nothing that could
   drift between steps. Tuning alpha would make the monitor's own hyperparameter a
   free variable in a round whose tuning budget is zero (7.14a).
4. **The adapter is the attachment point, and the raw encoder features are logged
   too.** ``morph_proxy_auroc`` reads the 640-d post-LayerNorm adapter -- the tensor
   the classifier actually sees and the tensor DANN's gradient shapes. The historical
   0.914 was measured on raw frozen encoder features, so
   ``morph_proxy_auroc_encoder`` is logged alongside it for comparability. They
   answer different questions and both are cheap once the forward has run.

-------------------------------------------------------------------------------
TWO INSTRUMENTS, NOT ONE -- and the one condition 3 reads is the HEAD
-------------------------------------------------------------------------------
``PROJECT_STATE v2.md`` 6 and 8/F4 record TWO morphology numbers from the SAME
v5-era runs, and they read ~0.34 apart:

    trained HEAD, AUROC on ML-FP vs P1-reserve   ~0.61 no-DA -> ~0.37 plain DANN
                                                            -> ~0.33 IW-CDAN
    frozen-feature linear probe, same pool       "stayed ~0.90 -- representation
                                                 intact, the supervised HEAD is
                                                 what degrades"

Measured at round 5's step 0 on this pool, BOTH reconstruct their historical
values. So they are not two views of one quantity; they are two quantities, and
only the first one moved when the historical failure happened.

  * ``morph_head_auroc`` -- the AUROC of the TRAINED HEAD'S OWN LENS LOGIT at
    separating ML-FP from P1-reserve. No probe, no fit: it is the model's own
    decision function, read on a morphology task. **This is the instrument abort
    condition 3 reads** (PROJECT.md 7.17). It is the ~0.61-scale one.
  * ``morph_proxy_auroc`` -- the ridge probe above. **Kept, with no detector
    attached**, because it is the historical diagnostic reproduced and it is what
    makes a drop INTERPRETABLE: head down + probe held = classifier ACCESS
    destroyed; both down = the REPRESENTATION went. The round-5 design doc's
    monitor row 5 always wanted this pair.

WHICH HALF THE HEAD COLUMN IS READ ON. Both are emitted. ``morph_head_auroc`` is
the **eval half**, matching the frozen-probe column exactly, so the two move on
THE SAME 128 IMAGES and "head down + probe held" is a statement about the same
objects rather than about two different samples. ``morph_head_auroc_pool`` is the
whole 256 and is the lower-variance estimate -- the head needs no fitting, so
nothing forbids using all of it. The eval half is preferred for the PAIR; the
pool column is there if the margin ever turns out to be hard to declare against
eval-half noise, and switching would be one line plus a stated reason.

AUROC is rank-based, so any monotone transform of the probe scores is invisible to
it. That is deliberate: the monitor asks whether the information is ACCESSIBLE to a
linear read-out, not how well calibrated that read-out is.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lens_dataset import DEFAULT_LAYOUT, IMG_KEY, LensTransform   # noqa: E402

# ---- declared, untuned ----
MORPH_N_FIT_PER_CLASS = 64      # 128 rows to fit a 640-d ridge probe on
MORPH_N_EVAL_PER_CLASS = 64     # 128 held-out rows to score AUROC on
MORPH_RIDGE_ALPHA = 1.0         # on z-scored features; NOT tuned
MORPH_AUG = "none"              # a FIXED deterministic view -- see build_morph_pool
MORPH_SEED_OFFSET = 90210       # keeps the morph draw independent of the holdout draw


# ---------------------------------------------------------------------------
# 1. the pool
# ---------------------------------------------------------------------------

def _unlabeled_shards(data_root, layout):
    d = os.path.join(data_root, layout["unlabeled_dir"])
    return sorted(glob.glob(os.path.join(d, "*.npz")))


def _mlfp_shards(data_root, layout):
    d = os.path.join(data_root, layout["mlfp_dir"])
    return sorted(glob.glob(os.path.join(d, "*.npz")))


def p1_reserve_locations(splits: dict, data_root: str, layout: dict,
                         *, partition: int = 1) -> list:
    """Partition-``partition`` unlabeled galaxies that ``source_train`` did NOT consume.

    Returns ``[(shard_basename, row), ...]`` sorted, so the enumeration is
    deterministic before any RNG is involved. This is the "P1-reserve" the historical
    instrument named; the current splits leave 23,160 of P1's 39,824 unused.
    """
    used = defaultdict(set)
    for shard_fn, idx in splits["source_train"]["neg_random"]["locations"]:
        used[os.path.basename(str(shard_fn))].add(int(idx))
    out = []
    for path in _unlabeled_shards(data_root, layout):
        fn = os.path.basename(path)
        with np.load(path) as npz:
            if "partitions" not in npz:
                continue
            parts = np.asarray(npz["partitions"])
        u = used.get(fn, set())
        for row in np.where(parts == int(partition))[0].tolist():
            if int(row) not in u:
                out.append((fn, int(row)))
    return sorted(out)


def mlfp_locations(splits: dict, data_root: str, layout: dict) -> list:
    """Every ML-FP hard negative, ``[(shard_basename, row), ...]``, sorted.

    Taken from the shards rather than from ``source_train.neg_mlfp`` so the pool is a
    property of the DATA, not of a training split -- if a future splits build stops
    using all 4,455, the monitor should not silently change with it.
    """
    out = []
    for path in _mlfp_shards(data_root, layout):
        fn = os.path.basename(path)
        with np.load(path) as npz:
            n = int(npz[IMG_KEY].shape[0])
        out += [(fn, i) for i in range(n)]
    return sorted(out)


def _load_rows(paths_and_rows, base_dir, transform):
    """Load and transform the selected rows, one file opened once."""
    by_file = defaultdict(list)
    for pos, (fn, row) in enumerate(paths_and_rows):
        by_file[fn].append((int(row), pos))
    out = [None] * len(paths_and_rows)
    for fn, rows in by_file.items():
        with np.load(os.path.join(base_dir, fn)) as npz:
            arr = npz[IMG_KEY]
            for row, pos in rows:
                out[pos] = transform(np.asarray(arr[row], dtype=np.float32))
    return torch.stack(out, dim=0)


def build_morph_pool(splits, data_root, *, n_fit_per_class=MORPH_N_FIT_PER_CLASS,
                     n_eval_per_class=MORPH_N_EVAL_PER_CLASS, seed=42,
                     layout=None, aug=MORPH_AUG, verbose=True) -> dict:
    """The fixed ML-FP vs P1-reserve pool, materialised once as a tensor.

    ``aug="none"`` on purpose: this is a FIXED probe. If the pool were re-augmented
    every step, a move in ``morph_proxy_auroc`` could be the representation changing
    or the augmentation drawing differently, and the monitor exists to say which. The
    same argument the held-out target slice uses.

    Deterministic given ``seed``: the candidate locations are sorted before any draw,
    and the draw is ``numpy.random.default_rng(seed + MORPH_SEED_OFFSET)`` -- offset
    so the morphology pool and the D-holdout slice cannot accidentally share a
    sampling stream.
    """
    if isinstance(splits, str):
        with open(splits) as fh:
            splits = json.load(fh)
    layout = {**DEFAULT_LAYOUT, **(layout or {})}
    data_root = os.path.abspath(data_root)

    pos_loc = mlfp_locations(splits, data_root, layout)          # ML-FP  -> 1
    neg_loc = p1_reserve_locations(splits, data_root, layout)    # reserve -> 0
    n_need = int(n_fit_per_class) + int(n_eval_per_class)
    if len(pos_loc) < n_need or len(neg_loc) < n_need:
        raise SystemExit(
            f"[morph] ABORT: need {n_need} per class, have {len(pos_loc)} ML-FP and "
            f"{len(neg_loc)} P1-reserve. The morphology proxy cannot be built, and it "
            f"arms abort condition 3 -- refusing to run without it rather than logging "
            f"NaN and calling the round monitored.")

    rng = np.random.default_rng(int(seed) + MORPH_SEED_OFFSET)
    pick_pos = [pos_loc[i] for i in
                sorted(rng.choice(len(pos_loc), size=n_need, replace=False).tolist())]
    pick_neg = [neg_loc[i] for i in
                sorted(rng.choice(len(neg_loc), size=n_need, replace=False).tolist())]

    tf = LensTransform(aug=aug)
    imgs = torch.cat([
        _load_rows(pick_pos, os.path.join(data_root, layout["mlfp_dir"]), tf),
        _load_rows(pick_neg, os.path.join(data_root, layout["unlabeled_dir"]), tf)], 0)
    y = np.concatenate([np.ones(n_need, np.int64), np.zeros(n_need, np.int64)])

    # fit / eval split: the FIRST n_fit of each class fits, the rest scores. Both
    # halves are class-balanced by construction, so AUROC has an equal-n null.
    fit_idx = np.concatenate([np.arange(n_fit_per_class),
                              n_need + np.arange(n_fit_per_class)])
    eval_idx = np.concatenate([np.arange(n_fit_per_class, n_need),
                               n_need + np.arange(n_fit_per_class, n_need)])

    info = {
        "instrument": "morphology proxy: ML-FP (=1) vs P1-reserve (=0), both real Euclid",
        "provenance": ("RECOVERED from PROJECT_STATE.md's `morphology_probe` role and "
                       "scripts/linear_probe.py, both of which were dropped at the "
                       "splits_v6 rewrite. The DATA survives; the code did not. The "
                       "historical readings this instrument produced were morphAUROC "
                       "0.914 (raw frozen Zoobot) and 0.898 (SimSiam)."),
        "label_free": ("neither class is a lens label -- ML-FP are graded X = NOT a "
                       "lens, P1-reserve are ungraded random galaxies. No simulated "
                       "image, no real-lens label and no test-split image is touched."),
        "n_mlfp_available": len(pos_loc), "n_reserve_available": len(neg_loc),
        "n_fit_per_class": int(n_fit_per_class),
        "n_eval_per_class": int(n_eval_per_class),
        "n_total": int(2 * n_need), "seed": int(seed),
        "seed_effective": int(seed) + MORPH_SEED_OFFSET,
        "aug": aug, "ridge_alpha": MORPH_RIDGE_ALPHA,
        "mlfp_locations": [[a, int(b)] for a, b in pick_pos],
        "reserve_locations": [[a, int(b)] for a, b in pick_neg],
        "disjointness": ("P1-reserve is the part of partition 1 that source_train's "
                         "neg_random did not consume, so it is disjoint from every "
                         "training and evaluation role. The ML-FP side is NOT disjoint "
                         "-- all 4,455 are source_train negatives -- which is faithful "
                         "to the original instrument and is stated, not hidden."),
    }
    if verbose:
        print(f"[morph] pool built: {n_need} ML-FP + {n_need} P1-reserve "
              f"(of {len(pos_loc)} / {len(neg_loc)} available), "
              f"fit {2*n_fit_per_class} / eval {2*n_eval_per_class}, "
              f"aug={aug}, seed {info['seed_effective']}", flush=True)
    return {"images": imgs, "y": y, "fit_idx": fit_idx, "eval_idx": eval_idx,
            "info": info}


# ---------------------------------------------------------------------------
# 2. the probe
# ---------------------------------------------------------------------------

def auroc_np(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC with tie handling. Local so the monitor has no import cost."""
    y = np.asarray(y_true).astype(np.int64).ravel()
    s = np.asarray(score, dtype=np.float64).ravel()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0        # average rank for ties
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ridge_probe_auroc(feats: torch.Tensor, y: np.ndarray, fit_idx, eval_idx,
                      *, alpha: float = MORPH_RIDGE_ALPHA) -> float:
    """Fit a ridge linear probe on ``fit_idx``, return AUROC on ``eval_idx``.

    Z-scored on the FIT half's own statistics, which is what makes the whole reading
    invariant to a uniform rescaling of the representation. 7.15's variance collapse
    would otherwise show up here as a morphology loss, and this monitor is supposed to
    detect exactly one thing.
    """
    X = feats.detach().double()
    Xf = X[fit_idx]
    mu = Xf.mean(0, keepdim=True)
    sd = Xf.std(0, unbiased=True, keepdim=True).clamp_min(1e-8)
    Z = (X - mu) / sd
    Zf = Z[fit_idx]
    yf = torch.from_numpy(np.asarray(y)[fit_idx]).double().to(Z.device)
    yf = yf - yf.mean()

    d = Zf.shape[1]
    A = Zf.T @ Zf + float(alpha) * Zf.shape[0] * torch.eye(d, dtype=Z.dtype,
                                                           device=Z.device)
    w = torch.linalg.solve(A, Zf.T @ yf)
    scores = (Z[eval_idx] @ w).cpu().numpy()
    return auroc_np(np.asarray(y)[eval_idx], scores)


class MorphProxy:
    """The per-step morphology-proxy monitor.

    Holds the fixed pool and evaluates it against whatever the trainer's encoder and
    head currently are. Costs ONE no-grad forward of ``n_total`` images plus two
    640x640 solves.

    Four columns come out of ONE forward pass: the head's own logit read on the eval
    half and on the whole pool, and the ridge probe on the adapter and on the raw
    encoder features. The head columns are free -- the logit was computed anyway.

    COST, MEASURED AND STATED. At the declared 256-image pool this is half a training
    batch of forward-only work per call, on top of the ~78-image D holdout. The build
    brief asks for it every step; ``--morph-every`` exists so that can be traded. Abort
    condition 3 reads "sustained over 500 steps", which 20 readings at
    ``--morph-every 25`` establish just as well as 500 readings do, at ~1/25th the
    cost. The default follows the brief; the flag is there because the tradeoff is
    real and the number above was not known when the brief was written.
    """

    def __init__(self, pool: dict, *, device=None, alpha: float = MORPH_RIDGE_ALPHA):
        self.images = pool["images"]
        self.y = np.asarray(pool["y"])
        self.fit_idx = np.asarray(pool["fit_idx"])
        self.eval_idx = np.asarray(pool["eval_idx"])
        self.info = dict(pool["info"])
        self.alpha = float(alpha)
        self.device = device
        self.last = {}

    @torch.no_grad()
    def measure(self, trainer, *, amp: bool = False, batch_size: int = 128) -> dict:
        """AUROC on the adapter features AND on the raw encoder features.

        ``morph_head_auroc``           -- THE HEAD'S OWN LENS LOGIT on the eval half.
                                          No probe, no fit. This is the ~0.61-scale
                                          instrument and the one abort condition 3
                                          reads (PROJECT.md 7.17).
        ``morph_head_auroc_pool``      -- the same, on all 256 images: lower variance,
                                          not paired with the probe column.
        ``morph_proxy_auroc``          -- ridge probe on the 640-d post-standardizer,
                                          post-LayerNorm adapter. NO detector attached;
                                          it is what makes a head drop interpretable.
        ``morph_proxy_auroc_encoder``  -- ridge probe on the raw encoder output,
                                          comparable to the historical 0.914 / 0.898.

        Runs with the head in ``eval()`` and restores the previous mode. It never
        touches D, so D's dropout state is irrelevant here -- but the head mode is
        restored anyway, because a monitor that leaves a module in the wrong mode is
        the failure this round already had to design around once.
        """
        from train_skeleton import encode_features
        was_head_train = trainer.head.training
        trainer.head.eval()
        trainer.encoder.eval()
        dev = trainer.device
        enc_chunks, adp_chunks, logit_chunks = [], [], []
        ctx = trainer._autocast() if amp else _nullctx()
        with ctx:
            for i in range(0, self.images.shape[0], int(batch_size)):
                imgs = self.images[i:i + int(batch_size)].to(dev)
                raw = encode_features(trainer.encoder, imgs, trainer.cfg.encoder_mode)
                lg, adapter = trainer.head.forward_with_features(
                    trainer.standardizer(raw))
                enc_chunks.append(raw.detach().float().cpu())
                adp_chunks.append(adapter.detach().float().cpu())
                logit_chunks.append(lg.detach().float().cpu().view(-1))
        if was_head_train:
            trainer.head.train()

        enc = torch.cat(enc_chunks, 0)
        adp = torch.cat(adp_chunks, 0)
        # The head's own lens logit, used AS IS. No fitting happens here, so there is
        # no fit/eval leakage to guard against and no held-out half is required --
        # the eval-half read exists to pair with the probe column on the same images,
        # not to protect against overfitting.
        logit = torch.cat(logit_chunks, 0).numpy()
        out = {
            "morph_head_auroc": auroc_np(self.y[self.eval_idx], logit[self.eval_idx]),
            "morph_head_auroc_pool": auroc_np(self.y, logit),
            "morph_proxy_auroc": ridge_probe_auroc(adp, self.y, self.fit_idx,
                                                   self.eval_idx, alpha=self.alpha),
            "morph_proxy_auroc_encoder": ridge_probe_auroc(enc, self.y, self.fit_idx,
                                                           self.eval_idx,
                                                           alpha=self.alpha),
            "morph_n_eval": float(self.eval_idx.size),
            "morph_n_pool": float(self.y.size),
        }
        self.last = out
        return out


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
