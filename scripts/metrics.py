"""
metrics.py
==========
The metric computations for the label-free sim-to-real strong-lens pipeline
(Euclid Q1). Pure NumPy + scikit-learn: NO torch, NO encoder, NO data loading --
so it imports fast, tests fast, and can be reused anywhere a (y_true, scores)
pair is available.

Everything here is grounded in PROJECT.md section 2 and the research notes in the
build request (tags: RB=research-backed, EC=engineering-choice, FLAG=deviation
forced by a real interface / a numerical subtlety).

  * TPR @ FPR=1e-3 (RB) -- the headline, prevalence-invariant operating metric that
      Lines et al. report (they call it TPR@0.1%). Computed via
      ``sklearn.metrics.roc_curve(..., drop_intermediate=False)`` then
      ``np.interp(target_fpr, fpr, tpr)``.

      drop_intermediate=False is MANDATORY here (RB): the sklearn default drops
      "suboptimal" ROC vertices to lighten the curve. In the current sklearn that
      drop is collinear-only (it removes points that lie exactly on the segment
      between their neighbours, so np.interp is unchanged), but the heuristic is a
      version-dependent implementation detail we do NOT want our low-FPR operating
      point to depend on. Passing drop_intermediate=False pins the FULL curve --
      every tail vertex that defines the 1e-3 operating point is retained -- so the
      metric is correct regardless of the sklearn version. (This is the one place
      the request is emphatic, and we follow it exactly.)

  * AUROC (RB) -- ``roc_auc_score``; kept available but now SECONDARY / sanity only
      (it is prevalence-invariant and over-optimistic under extreme skew). The
      threshold-free HEADLINE summary is now AUPRC.

  * AUPRC (RB) -- ``average_precision_score`` (NOT trapezoidal ``auc`` on a PR curve;
      trapezoidal PR-AUC is biased under extreme class skew). Average precision is the
      exact (non-interpolated) area under the PR curve. Its random-score baseline equals
      the positive prevalence, so a bare AUPRC number is only meaningful next to the pool
      prevalence it was measured at.

  * purity_at_completeness / completeness_at_purity (RB) -- the field-standard low-FPR
      completeness (=recall/TPR) + purity (=precision) family, made comparable to Lines
      et al. (arXiv:2512.05899). Purity is reported at a fixed completeness (and vice
      versa), computed by interpolating on the completeness / purity axis exactly as
      ``tpr_at_fpr`` interpolates on the FPR axis, with the same nan-not-extrapolation
      guard. Two purity modes: raw-pool (internal comparison) and PREVALENCE-CORRECTED
      to a chosen N_N/N_P (Lines' own conversion, purity = TPR/(TPR + FPR*(N_N/N_P))),
      so our lens-rich eval pool can be rescaled to Lines' 1:2200 deployment rarity.

  * NO trapezoidal AUC on a PR curve (RB) -- we never call ``sklearn.metrics.auc``
      on precision/recall; AUPRC above uses ``average_precision_score``.

  * paired_bootstrap (RB) -- class-conditional resampling: positives and negatives
      are resampled independently WITH REPLACEMENT (the class-conditional structure
      is preserved; the two classes are never pooled). B=1000, fixed seed, returns
      median + a 2.5/97.5 percentile CI. For method-vs-method it uses the SAME
      resampled row indices for both methods (paired) and returns the CI of the
      per-replicate difference; "significant" iff that CI excludes 0.

  * alert-rate guardrail (EC/RB) -- the fraction of an UNLABELED target pool that
      clears the FPR=1e-3 threshold. Logged, never a selector. It "fires" when that
      fraction explodes far above the ~0.002 lens prevalence (the "everything fires"
      collapse). This is a diagnostic, not SND: SND is inapplicable under this
      extreme imbalance (target-density selectors just measure the negative majority).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# Deployment lens prevalence in Euclid Q1 (~1 in 2186 ~= 4.6e-4; PROJECT.md 2).
# The guardrail flags an alert rate FAR above this. We use ~0.002 as the reference
# scale named in the build request (a conservative upper bound on prevalence).
LENS_PREVALENCE = 0.002
DEFAULT_TARGET_FPR = 1e-3

# Default operating points for the completeness/purity family (EC -- reporting knobs,
# never tuning targets). 50% completeness is the standard reference completeness Lines
# et al. report their purity at; 24% purity is Lines' source-only purity point.
DEFAULT_TARGET_COMPLETENESS = 0.5
DEFAULT_TARGET_PURITY = 0.24


# ---------------------------------------------------------------------------
# Lines et al. (arXiv:2512.05899) REFERENCE BLOCK  (context, NOT a tuning target)
# ---------------------------------------------------------------------------
#
# These constants exist ONLY so a purity number can be contextualized against the
# published Euclid Q1 benchmark. They are documented, never hardcoded silently into a
# metric, and -- per PROJECT.md's one hard rule -- MUST NOT be used as an optimization
# target. Nothing in the pipeline selects a checkpoint or tunes a knob to move a number
# toward these values.
#
#   NN_NP_Q1 = 2200
#       The negative-to-positive ratio (N_N / N_P) of REAL Q1 that Lines et al. quote
#       for their purity conversion: 497 expert lenses in 1,086,554 screened images
#       (~1:2186; they round to 2200). Our evaluation pool is deliberately lens-RICHER
#       than this (~1:270 = 26,399 shared negatives : ~98 fold positives), which INFLATES
#       raw-pool purity relative to deployment. Correcting purity to NN_NP_Q1 rescales it
#       to Lines' deployment rarity so the two are comparable (see purity_at_completeness,
#       mode (b)). Because we correct to RARER positives, corrected purity is LOWER than
#       raw-pool purity -- that direction is the sanity check in the tests.
#
#   LINES_SOURCE_ONLY_COMPLETENESS ~ 0.50   at   LINES_SOURCE_ONLY_PURITY ~ 0.24
#       Lines' own SIM-ONLY (source-only) reference operating point on real Q1: roughly
#       50% completeness at ~24% purity. This is the "no domain adaptation" number our
#       label-free UDA is trying to recover/beat -- a comparison anchor, not a goal to
#       optimize toward.
NN_NP_Q1 = 2200.0
LINES_SOURCE_ONLY_COMPLETENESS = 0.50
LINES_SOURCE_ONLY_PURITY = 0.24


# ---------------------------------------------------------------------------
# ROC curve (full -- never drop tail vertices)
# ---------------------------------------------------------------------------

def roc_curve_full(y_true, scores):
    """``sklearn.roc_curve`` with ``drop_intermediate=False`` (the FULL curve).

    Returns ``(fpr, tpr, thresholds)`` with every vertex retained, so the low-FPR
    operating point can never be an interpolation across a dropped tail point.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    return roc_curve(y_true, scores, drop_intermediate=False)


def _two_class(y_true):
    return np.unique(np.asarray(y_true).ravel()).size >= 2


# ---------------------------------------------------------------------------
# TPR @ FPR
# ---------------------------------------------------------------------------

def tpr_at_fpr(y_true, scores, target_fpr: float = DEFAULT_TARGET_FPR,
               return_status: bool = False):
    """True-positive rate at a fixed false-positive rate (default 1e-3).

    Uses ``roc_curve_full`` (drop_intermediate=False) then linear interpolation
    ``np.interp(target_fpr, fpr, tpr)``.

    Guards against silent extrapolation: ``target_fpr`` MUST lie within
    ``[fpr.min(), fpr.max()]``. With the full curve those bounds are [0, 1] whenever
    both classes are present, so extrapolation only happens on a degenerate fold
    (a single class, or an out-of-range request) -- in which case we return
    ``np.nan`` and a status string, never an extrapolated number.

    Parameters
    ----------
    return_status : bool
        If True, return ``(value, status)`` where status is one of
        {"ok", "single_class", "extrapolation"}; else return just the float.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    def _out(val, status):
        return (float(val), status) if return_status else float(val)

    if not _two_class(y_true):
        warnings.warn("tpr_at_fpr: only one class present; ROC undefined -> nan")
        return _out(np.nan, "single_class")

    fpr, tpr, _ = roc_curve_full(y_true, scores)
    lo, hi = float(fpr.min()), float(fpr.max())
    if not (lo <= target_fpr <= hi):
        warnings.warn(
            f"tpr_at_fpr: target_fpr={target_fpr} outside observed FPR range "
            f"[{lo}, {hi}] -> nan (refusing to extrapolate)")
        return _out(np.nan, "extrapolation")

    return _out(np.interp(target_fpr, fpr, tpr), "ok")


def operating_threshold(y_true, scores, target_fpr: float = DEFAULT_TARGET_FPR) -> float:
    """The score threshold that realises FPR=``target_fpr`` on ``(y_true, scores)``.

    Derived from the SAME full ROC curve as ``tpr_at_fpr`` (drop_intermediate=False),
    so the reported TPR@FPR and this threshold are mutually consistent. The threshold
    is what the alert-rate guardrail applies to the unlabeled target pool.

    sklearn sets ``thresholds[0] = inf`` (the "predict nothing positive" vertex at
    fpr=0). We replace that single infinity with a finite value just above the max
    score before interpolating, so the threshold near fpr=0 is finite and usable.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if not _two_class(y_true):
        return np.nan
    fpr, tpr, thr = roc_curve_full(y_true, scores)
    thr = thr.astype(float).copy()
    if np.isinf(thr[0]):
        # a threshold strictly above every score -> nothing predicted positive (fpr=0)
        thr[0] = np.nextafter(scores.max(), np.inf)
    lo, hi = float(fpr.min()), float(fpr.max())
    if not (lo <= target_fpr <= hi):
        return np.nan
    return float(np.interp(target_fpr, fpr, thr))


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def auroc(y_true, scores) -> float:
    """Area under the ROC curve (``sklearn.roc_auc_score``); nan if one class.

    SECONDARY / sanity metric (see module docstring): prevalence-invariant, but
    over-optimistic under the extreme lens:non-lens skew, so it is NOT the headline.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if not _two_class(y_true):
        return np.nan
    return float(roc_auc_score(y_true, scores))


# ---------------------------------------------------------------------------
# AUPRC (average precision) -- the threshold-free HEADLINE summary
# ---------------------------------------------------------------------------

def auprc(y_true, scores) -> float:
    """Average precision = area under the precision-recall curve (``sklearn
    ``average_precision_score``); nan if one class.

    This uses the exact step-function AP, NOT ``sklearn.metrics.auc`` on a PR curve
    (trapezoidal PR-AUC is biased under extreme class skew). AP replaces AUROC as the
    threshold-free summary because, unlike AUROC, it is sensitive to the low-FPR / high-
    skew regime that matters for lens finding.

    IMPORTANT (interpretation): the random-baseline AUPRC equals the POSITIVE PREVALENCE
    of the pool it is measured on (``n_pos / (n_pos + n_neg)``). An AUPRC of e.g. 0.3 is
    excellent at 1:2200 prevalence but poor at 1:1 -- so AUPRC is only meaningful beside
    the prevalence it was computed at. It is therefore an INTERNAL-comparison metric
    (same pool for both methods), not directly comparable across different prevalences.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if not _two_class(y_true):
        return np.nan
    return float(average_precision_score(y_true, scores))


def prevalence(y_true) -> float:
    """Positive prevalence ``n_pos / n_total`` -- the random-baseline for ``auprc``."""
    y = np.asarray(y_true).ravel()
    return float(np.mean(y == 1)) if y.size else np.nan


# ---------------------------------------------------------------------------
# completeness (=recall/TPR) + purity (=precision) family
# ---------------------------------------------------------------------------
#
# purity (precision) = TP / (TP + FP). At a threshold with true-positive rate TPR and
# false-positive rate FPR, TP = TPR*N_P and FP = FPR*N_N, so
#
#       purity = TPR*N_P / (TPR*N_P + FPR*N_N) = TPR / (TPR + FPR*(N_N/N_P)).
#
# The ONLY pool-dependent quantity is the ratio r = N_N/N_P. Using the pool's actual
# ratio gives raw-pool purity; substituting a chosen r (e.g. NN_NP_Q1=2200) rescales
# purity to that deployment rarity -- this is Lines et al.'s own prevalence correction.
# Because our eval pool is lens-RICHER than 1:2200, correcting to the rarer 1:2200
# LOWERS purity (more negatives per positive => more false positives per true positive).

def _pool_ratio(y):
    """Actual negative-to-positive ratio N_N/N_P of the label vector (nan if no pos)."""
    n_pos = int(np.count_nonzero(y == 1))
    n_neg = int(np.count_nonzero(y == 0))
    return (n_neg / n_pos) if n_pos > 0 else np.nan


def _purity_from_rates(tpr, fpr, ratio):
    """purity = TPR / (TPR + FPR*ratio); nan where the denominator is 0 (no predicted
    positives). Works on scalars or arrays."""
    tpr = np.asarray(tpr, dtype=float)
    fpr = np.asarray(fpr, dtype=float)
    denom = tpr + fpr * float(ratio)
    with np.errstate(divide="ignore", invalid="ignore"):
        pur = np.where(denom > 0, tpr / denom, np.nan)
    return pur if pur.ndim else float(pur)


def _finite_thresholds(thr, scores):
    """A copy of the ROC thresholds with sklearn's leading +inf (the 'predict nothing
    positive' vertex) replaced by a finite value just above the max score, so a threshold
    interpolated near completeness=0 is usable (mirrors ``operating_threshold``)."""
    thr = np.asarray(thr, dtype=float).copy()
    if thr.size and np.isinf(thr[0]):
        thr[0] = np.nextafter(float(np.max(scores)), np.inf)
    return thr


def _operating_point_at_completeness(target, fpr, tpr, thr_fin):
    """Interpolate the (fpr, threshold) of the BEST operating point that just achieves
    completeness = ``target`` (i.e. the SMALLEST fpr / HIGHEST threshold reaching that
    TPR).

    Why not a plain ``np.interp(target, tpr, fpr)``: a TPR plateau (the same completeness
    at several thresholds, because lowering the threshold caught extra NEGATIVES but no
    new positive) makes the fpr multi-valued at ``target``, and ``np.interp`` silently
    returns the RIGHT (highest-fpr, worst-purity) end of the plateau. The operating point
    at a fixed completeness is the FIRST threshold that reaches it -- you would never
    lower the threshold past it, adding false positives for no extra recall. So we reduce
    each distinct TPR level to its MINIMUM fpr (``np.unique`` returns the first index, and
    fpr is non-decreasing along the ROC curve, so the first vertex at a TPR has the
    smallest fpr) and interpolate on those breakpoints. This mirrors ``tpr_at_fpr``, which
    likewise reads the BEST (highest) TPR within its FPR budget.
    """
    tpr = np.asarray(tpr, dtype=float)
    fpr = np.asarray(fpr, dtype=float)
    uniq_tpr, first = np.unique(tpr, return_index=True)     # first index => min fpr / max thr
    fpr_at = float(np.interp(target, uniq_tpr, fpr[first]))
    thr_at = float(np.interp(target, uniq_tpr, np.asarray(thr_fin, float)[first]))
    return fpr_at, thr_at


def purity_at_completeness(y_true, scores,
                           target_completeness: float = DEFAULT_TARGET_COMPLETENESS,
                           nn_np_ratio=None) -> dict:
    """Purity (=precision) at a fixed completeness (=recall/TPR).

    Finds the threshold achieving ``target_completeness`` by interpolating on the
    completeness axis of the FULL ROC curve (drop_intermediate=False) -- exactly the way
    ``tpr_at_fpr`` interpolates on the FPR axis -- then reports the purity there.

    Two modes
    ---------
    ``nn_np_ratio=None``  (raw-pool): purity on the pool AS-IS, using the pool's own
        N_N/N_P. This is the internal-comparison purity.
    ``nn_np_ratio=r`` (e.g. 2200): PREVALENCE-CORRECTED purity to a negative-to-positive
        ratio ``r``. purity = TPR / (TPR + FPR*r) at the chosen threshold -- Lines' own
        conversion (they state N_N/N_P=2200 for Q1). Correcting a lens-rich pool to the
        rarer 1:2200 LOWERS purity.

    Returns
    -------
    dict with keys ``purity`` (nan if the target completeness is not reachable -- the
    guard REFUSES to extrapolate, never returns an out-of-range number), ``completeness``
    (the achieved TPR = target when ok), ``fpr``, ``threshold``, ``nn_np_ratio`` (the
    ratio actually used), and ``status`` in {"ok","single_class","extrapolation"}.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    ratio = float(nn_np_ratio) if nn_np_ratio is not None else _pool_ratio(y)

    def _out(purity, comp, fpr, thr, status):
        return {"purity": float(purity) if np.isfinite(purity) else np.nan,
                "completeness": float(comp) if np.isfinite(comp) else np.nan,
                "fpr": float(fpr) if np.isfinite(fpr) else np.nan,
                "threshold": float(thr) if np.isfinite(thr) else np.nan,
                "nn_np_ratio": float(ratio), "status": status}

    if not _two_class(y):
        warnings.warn("purity_at_completeness: only one class present -> nan")
        return _out(np.nan, np.nan, np.nan, np.nan, "single_class")

    fpr, tpr, thr = roc_curve_full(y, s)
    lo, hi = float(tpr.min()), float(tpr.max())
    if not (lo <= target_completeness <= hi):
        warnings.warn(
            f"purity_at_completeness: target_completeness={target_completeness} outside "
            f"observed completeness range [{lo}, {hi}] -> nan (refusing to extrapolate)")
        return _out(np.nan, np.nan, np.nan, np.nan, "extrapolation")

    thr_fin = _finite_thresholds(thr, s)
    fpr_at, thr_at = _operating_point_at_completeness(target_completeness, fpr, tpr, thr_fin)
    purity = _purity_from_rates(target_completeness, fpr_at, ratio)
    return _out(purity, target_completeness, fpr_at, thr_at, "ok")


def completeness_at_purity(y_true, scores, target_purity: float,
                           nn_np_ratio=None) -> dict:
    """Completeness (=recall/TPR) at a fixed (optionally prevalence-corrected) purity --
    the dual of ``purity_at_completeness``.

    Reports the MAXIMUM completeness at which purity is still >= ``target_purity`` (the
    standard "recall at fixed precision" operating point), with linear interpolation at
    the crossing.

    FLAG (a deviation forced by the metric, documented not hidden): precision is NOT
    monotone along the ROC/PR curve (a single extra false positive can dip it), so a bare
    ``np.interp`` on the purity axis is ill-defined. We therefore interpolate on the
    PRECISION-RECALL ENVELOPE -- ``env(c) = max purity achievable at completeness >= c``,
    which IS monotone non-increasing in completeness -- the standard convention behind
    "recall at precision". This preserves the same interpolate + nan-guard discipline as
    ``purity_at_completeness`` while being well-defined under the zig-zag.

    ``nn_np_ratio`` behaves exactly as in ``purity_at_completeness`` (None => raw pool;
    a value => prevalence-corrected purity for the crossing test).

    Returns a dict with ``completeness`` (nan if ``target_purity`` is NOT reachable at
    ANY operating point -- never extrapolated), ``purity`` (>= target at the returned
    point), ``fpr``, ``threshold``, ``nn_np_ratio``, ``status`` in
    {"ok","single_class","unreachable"}.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    ratio = float(nn_np_ratio) if nn_np_ratio is not None else _pool_ratio(y)

    if not _two_class(y):
        warnings.warn("completeness_at_purity: only one class present -> nan")
        return {"completeness": np.nan, "purity": np.nan, "fpr": np.nan,
                "threshold": np.nan, "nn_np_ratio": float(ratio), "status": "single_class"}

    fpr, tpr, thr = roc_curve_full(y, s)
    return _completeness_at_purity_from_curve(fpr, tpr, thr, s, ratio, target_purity)


def _completeness_at_purity_from_curve(fpr, tpr, thr, scores, ratio, target_purity,
                                       warn=True):
    """Core of ``completeness_at_purity`` on an ALREADY-computed ROC curve (so callers
    that already have the curve -- e.g. ``metric_suite`` -- do not pay for a second
    ``roc_curve``). ``ratio`` is the final N_N/N_P to use (raw or corrected)."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    comp = tpr

    def _out(comp_v, purity, fpr_v, thr_v, status):
        return {"completeness": float(comp_v) if np.isfinite(comp_v) else np.nan,
                "purity": float(purity) if np.isfinite(purity) else np.nan,
                "fpr": float(fpr_v) if np.isfinite(fpr_v) else np.nan,
                "threshold": float(thr_v) if np.isfinite(thr_v) else np.nan,
                "nn_np_ratio": float(ratio), "status": status}

    pur = _purity_from_rates(tpr, fpr, ratio)          # nan at the tpr=fpr=0 vertex

    # Precision-recall ENVELOPE: suffix-max of purity along increasing completeness
    # (roc_curve returns vertices in order of non-decreasing tpr). env[i] = best purity
    # attainable at completeness >= comp[i]; monotone non-increasing in completeness.
    # Vectorized suffix-max (nan -> -inf): reverse, cumulative max, reverse back -- O(n)
    # in numpy, NOT a Python loop (this runs thousands of times under the bootstrap).
    p = np.where(np.isfinite(pur), pur, -np.inf)
    env = np.maximum.accumulate(p[::-1])[::-1]

    reachable = env >= target_purity
    if not np.any(reachable):
        if warn:
            best = float(np.nanmax(pur)) if np.any(np.isfinite(pur)) else float("nan")
            warnings.warn(
                f"completeness_at_purity: target_purity={target_purity} exceeds the best "
                f"attainable purity {best} (ratio={ratio:g}) -> nan (unreachable)")
        return _out(np.nan, np.nan, np.nan, np.nan, "unreachable")

    i = int(np.max(np.where(reachable)[0]))            # last vertex with env >= target
    if i >= env.size - 1:
        comp_star = float(comp[i])                     # target met across the whole curve
    else:
        e0, e1 = env[i], env[i + 1]
        c0, c1 = comp[i], comp[i + 1]
        if e0 == e1 or not np.isfinite(e1):
            comp_star = float(c1)
        else:
            frac = (e0 - target_purity) / (e0 - e1)
            comp_star = float(c0 + frac * (c1 - c0))

    # operating rates AT comp_star (same interpolation as purity_at_completeness)
    thr_fin = _finite_thresholds(thr, scores)
    fpr_at, thr_at = _operating_point_at_completeness(comp_star, fpr, tpr, thr_fin)
    purity_at = _purity_from_rates(comp_star, fpr_at, ratio)
    return _out(comp_star, purity_at, fpr_at, thr_at, "ok")


# ---------------------------------------------------------------------------
# metric suite -- all low-FPR metrics from ONE shared ROC pass (for the bake-off's
# multi-metric paired bootstrap, which recomputes every metric thousands of times)
# ---------------------------------------------------------------------------

def metric_suite(y_true, scores, *, target_fpr: float = DEFAULT_TARGET_FPR,
                 target_completeness: float = DEFAULT_TARGET_COMPLETENESS,
                 nn_np_ratio: float = NN_NP_Q1,
                 target_purity: float = DEFAULT_TARGET_PURITY) -> dict:
    """Compute the whole reportable suite as SCALARS in one pass (shares a single
    ``roc_curve_full`` across the ROC-based metrics; ``auprc``/``auroc`` use their own
    sklearn passes). Returns a dict of floats:

        tpr           -- TPR @ target_fpr                         (HEADLINE)
        purity_corr   -- purity @ target_completeness, corrected to nn_np_ratio (HEADLINE)
        purity_raw    -- purity @ target_completeness, raw pool
        auprc         -- average precision                        (threshold-free headline)
        comp_at_purity-- completeness @ target_purity (corrected to nn_np_ratio)
        auroc         -- AUROC                                    (SECONDARY / sanity)

    This is what ``compare_methods`` bootstraps; the individual public functions above are
    the single source of truth for the math and are what the unit tests pin.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    out = {"tpr": np.nan, "purity_corr": np.nan, "purity_raw": np.nan,
           "auprc": np.nan, "comp_at_purity": np.nan, "auroc": np.nan}
    if not _two_class(y):
        return out

    fpr, tpr, thr = roc_curve_full(y, s)

    # TPR @ FPR (headline) -- same interpolation as tpr_at_fpr, on the shared curve
    lo_f, hi_f = float(fpr.min()), float(fpr.max())
    out["tpr"] = (float(np.interp(target_fpr, fpr, tpr))
                  if lo_f <= target_fpr <= hi_f else np.nan)

    # purity @ completeness (raw + corrected) -- shared curve, best-operating-point fpr
    lo_c, hi_c = float(tpr.min()), float(tpr.max())
    if lo_c <= target_completeness <= hi_c:
        thr_fin = _finite_thresholds(thr, s)
        fpr_c, _ = _operating_point_at_completeness(target_completeness, fpr, tpr, thr_fin)
        out["purity_raw"] = _purity_from_rates(target_completeness, fpr_c, _pool_ratio(y))
        out["purity_corr"] = _purity_from_rates(target_completeness, fpr_c, nn_np_ratio)

    # completeness @ purity (corrected) -- reuse the SAME curve (envelope core). This is
    # the bulk/bootstrap path: silence the unreachable-target warning (nan is aggregated
    # by nanmean upstream); the public completeness_at_purity still warns for direct use.
    out["comp_at_purity"] = _completeness_at_purity_from_curve(
        fpr, tpr, thr, s, float(nn_np_ratio), target_purity, warn=False)["completeness"]

    out["auprc"] = float(average_precision_score(y, s))
    out["auroc"] = float(roc_auc_score(y, s))
    return out


def default_tpr_metric(target_fpr: float = DEFAULT_TARGET_FPR):
    """Return a ``metric_fn(y, s) -> float`` computing TPR@target_fpr (for bootstrap)."""
    def _fn(y, s):
        return tpr_at_fpr(y, s, target_fpr=target_fpr)
    return _fn


# ---------------------------------------------------------------------------
# alert-rate guardrail
# ---------------------------------------------------------------------------

def alert_rate(target_scores, threshold) -> float:
    """Fraction of ``target_scores`` at or above ``threshold`` (the "alert" fraction)."""
    ts = np.asarray(target_scores, dtype=float).ravel()
    if ts.size == 0 or not np.isfinite(threshold):
        return np.nan
    return float(np.mean(ts >= threshold))


def alert_rate_guardrail(target_scores, threshold,
                         prevalence: float = LENS_PREVALENCE,
                         factor: float = 10.0) -> dict:
    """Alert-rate collapse guardrail.

    Computes the fraction of the unlabeled target pool clearing ``threshold`` (the
    FPR=1e-3 operating threshold) and FLAGS ("fires") when it exceeds
    ``factor * prevalence`` -- i.e. the model is flagging far more of the target than
    the ~0.002 lens prevalence can explain ("everything fires" collapse).

    EC: ``factor=10`` (fire above ~2% alerts) is a diagnostic threshold, not a tuned
    selector. The guardrail is logged; it never selects a checkpoint.
    """
    ar = alert_rate(target_scores, threshold)
    limit = float(factor) * float(prevalence)
    fired = bool(np.isfinite(ar) and ar > limit)
    return {"alert_rate": ar, "prevalence": float(prevalence),
            "factor": float(factor), "limit": limit, "fired": fired}


# ---------------------------------------------------------------------------
# paired class-conditional bootstrap
# ---------------------------------------------------------------------------

def _resample(pos_idx, neg_idx, rng):
    """Independent WITH-REPLACEMENT resample of the positive and negative index sets
    (class-conditional: the two pools are never merged before resampling)."""
    p = pos_idx[rng.integers(0, pos_idx.size, pos_idx.size)]
    n = neg_idx[rng.integers(0, neg_idx.size, neg_idx.size)]
    return np.concatenate([p, n])


def paired_bootstrap(y_true, scores, scores_b=None, metric_fn=None,
                     B: int = 1000, seed: int = 0, alpha: float = 0.05,
                     target_fpr: float = DEFAULT_TARGET_FPR) -> dict:
    """Class-conditional bootstrap CI for a metric, or for a paired method difference.

    Positives and negatives are resampled independently with replacement (their
    class-conditional structure is preserved -- they are NOT pooled). The metric is
    recomputed on each replicate.

    Single-method mode (``scores_b is None``)
        Returns the point estimate, the bootstrap median, and the
        [alpha/2, 1-alpha/2] percentile CI of the metric.

    Paired-difference mode (``scores_b`` given)
        The SAME resampled row indices are applied to both score vectors on every
        replicate (paired). Returns the difference point estimate / median / CI and
        ``significant = (CI excludes 0)``. Use this for method-vs-method in the
        bake-off, where the shared negatives make the two methods' scores aligned
        row-for-row.

    Parameters
    ----------
    metric_fn : callable(y, s) -> float
        Defaults to TPR@``target_fpr``. AUROC can be passed for an AUROC CI.
    """
    if metric_fn is None:
        metric_fn = default_tpr_metric(target_fpr)

    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError(
            f"paired_bootstrap needs >=1 of each class, got pos={pos_idx.size} "
            f"neg={neg_idx.size}")

    diff = scores_b is not None
    if diff:
        sb = np.asarray(scores_b, dtype=float).ravel()
        if sb.shape != s.shape:
            raise ValueError(
                f"scores_b shape {sb.shape} != scores shape {s.shape} (paired mode "
                f"requires row-aligned score vectors)")
        point = metric_fn(y, s) - metric_fn(y, sb)
    else:
        point = metric_fn(y, s)

    rng = np.random.default_rng(seed)
    reps = np.empty(B, dtype=float)
    for b in range(B):
        idx = _resample(pos_idx, neg_idx, rng)
        yr = y[idx]
        if diff:
            reps[b] = metric_fn(yr, s[idx]) - metric_fn(yr, sb[idx])
        else:
            reps[b] = metric_fn(yr, s[idx])

    n_valid = int(np.isfinite(reps).sum())
    lo = float(np.nanpercentile(reps, 100.0 * (alpha / 2.0)))
    hi = float(np.nanpercentile(reps, 100.0 * (1.0 - alpha / 2.0)))
    med = float(np.nanmedian(reps))
    out = {"point": float(point) if np.isfinite(point) else np.nan,
           "median": med, "lo": lo, "hi": hi,
           "ci_width": hi - lo, "B": int(B), "n_valid": n_valid,
           "alpha": float(alpha)}
    if diff:
        # significant iff the CI of the difference excludes 0
        out["significant"] = bool(np.isfinite(lo) and np.isfinite(hi)
                                  and (lo > 0.0 or hi < 0.0))
    return out
