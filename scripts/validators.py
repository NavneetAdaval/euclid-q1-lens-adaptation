"""
validators.py
=============
Label-free CHECKPOINT VALIDATORS for the A1 study (Task 4, Part 1).

WHY THIS EXISTS
---------------
In the unfrozen regime (n_blocks > 0) we have no defensible way to pick a checkpoint.
Source-val AUROC is saturated (~0.999) and the three pre-registered rules disagree badly
(`nb3` gives 0.2143 / 0.1327 / 0.1327 under R1 / R2 / R3 -- an 8-lens swing on one run).
Selecting the peak requires test labels and is therefore illegal under PROJECT.md 2.

This module implements the six candidate validators so they can be scored by REGRET
against the real-test oracle. Expectations are calibrated low on purpose: Musgrave et al.,
*Unsupervised Domain Adaptation: A Reality Check* (arXiv:2111.15672), find that SOTA
validation methods are not well-correlated with accuracy and that the accuracy drop caused
by validators exceeds the differences between UDA algorithms. A negative result here is a
genuine finding.

THE SIX VALIDATORS
------------------
  1. source-val TPR@FPR=1e-3   -- NOT saturated (~0.90). The closest legal analogue to the
                                  ID-validation early stopping LP-FT depends on.
  2. source-val AUROC          -- the KNOWN-NULL CONTROL. Saturated ~0.999. If this shows a
                                  correlation, the analysis code is wrong (deliverable 4).
  3. agreement                 -- Spearman vs Baseline A's final scores on the target pool.
  4. IM (InfoMax)              -- H(mean p) - mean H(p), nats. Higher better.
  5. SND / NegSND              -- soft neighborhood density (Saito et al., ICCV 2021).
  6. ATC                       -- Garg et al., arXiv:2201.04234, negative-entropy variant.

DIRECTION IS NOT OURS TO CHOOSE (Trap 3). Each validator is tested in its PUBLISHED
direction, recorded in ``DIRECTION``. Musgrave found SND *inversely* correlated with
accuracy and introduced NegSND; we report NegSND but picking the sign by which correlates
better with real-test TPR is itself a use of test labels and CANNOT be carried into a
selection rule.

TWO IMPLEMENTATION POINTS THAT ARE EASY TO GET WRONG
----------------------------------------------------
* **IM must use the BALANCED target pool** (Trap 1). IM's diversity term rewards a balanced
  predicted marginal; on the 1:2200 unlabeled Q1 pool it would reward predicting 50/50,
  which is catastrophically wrong. ``information_maximization`` therefore takes a
  ``latent_labels`` argument and FAILS LOUDLY if the pool is not ~balanced. The latent
  labels are used ONLY for that construction-time assertion -- they never enter the
  returned value, so the validator stays label-free.

* **SND operates on the network OUTPUT, not the backbone features.** Saito et al. compute
  the pairwise cosine similarity of the *softmax output* vectors, and Musgrave's
  ``pytorch-adapt`` ``SNDValidator`` defaults to ``layer="preds"``. The Task-4 brief says
  "target features"; on a FROZEN encoder the backbone features are identical at every
  checkpoint, so a feature-space SND is EXACTLY CONSTANT and cannot select anything. We
  implement the generic kernel (``snd``) and expose both entry points --
  ``snd_from_logits`` (published, primary) and feature-space SND (which doubles as a second
  known-null control on the frozen runs). See the module test for the constancy proof.

Run the unit tests:
    python scripts/validators.py --test
    python scripts/test_validators.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from analyze_fold0_selection import _spearman            # noqa: E402  the ONE spearman impl
from metrics import DEFAULT_TARGET_FPR, auroc, tpr_at_fpr  # noqa: E402

# Saito et al.'s default softmax temperature for the SND similarity rows. Stated, not tuned.
SND_TEMPERATURE = 0.05

# Balance tolerance for the IM pool assertion (Trap 1). The fold target pool is 391:391 by
# construction (PROJECT.md 4.3), so anything outside a few percent means the wrong pool.
IM_BALANCE_TOL = 0.05

#: Published direction of each validator: +1 = argmax selects, -1 = argmin selects.
#: This is a REGISTRY, not a tunable. Flipping a sign because it correlates better with
#: real-test TPR is a use of test labels (Trap 3).
DIRECTION = {
    "source_val_tpr": +1,
    "source_val_auroc": +1,     # known-null control
    "agreement": +1,            # see the caveat in AGREEMENT_CAVEAT
    "im": +1,
    "snd": +1,                  # Saito's published direction
    "neg_snd": -1,              # Musgrave's inversion, reported not adopted
    "snd_features": +1,         # brief's literal reading; constant on a frozen encoder
    "atc": +1,
}

AGREEMENT_CAVEAT = (
    "Agreement is included because the brief lists it, but PROJECT.md 7.6 has already "
    "ruled it out AS A SELECTOR: max agreement is the source-only model itself (TTYD's "
    "MeanBN failure mode, which always selects an under-trained checkpoint). It is a FLOOR "
    "ALARM only. Its regret row is reported for completeness and is expected to be bad."
)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def binary_probs(logits) -> np.ndarray:
    """(N,) head logits -> (N,2) probability vectors ``[1-s, s]``."""
    s = _sigmoid(np.asarray(logits, dtype=np.float64).reshape(-1))
    return np.stack([1.0 - s, s], axis=1)


def _entropy_rows(p: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy in NATS. 0 log 0 := 0."""
    p = np.asarray(p, dtype=np.float64)
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=-1)


# ---------------------------------------------------------------------------
# validators 1-3 -- free, recomputable from curve.csv
# ---------------------------------------------------------------------------

def source_val_tpr(labels, logits, target_fpr: float = DEFAULT_TARGET_FPR) -> float:
    """Validator 1. Source-val TPR@FPR=1e-3 -- the ~0.90, NOT-saturated legal analogue of
    the ID-validation early stopping Kumar et al.'s LP-FT protocol depends on."""
    return float(tpr_at_fpr(np.asarray(labels), np.asarray(logits), target_fpr))


def source_val_auroc(labels, logits) -> float:
    """Validator 2. THE KNOWN-NULL CONTROL. Saturated ~0.999 (PROJECT.md 7.6): it carries no
    selection information, so it must show a near-zero correlation with real-test TPR. If it
    does not, the analysis code is wrong -- stop and fix it (deliverable 4)."""
    return float(auroc(np.asarray(labels), np.asarray(logits)))


def agreement(logits, reference_logits) -> float:
    """Validator 3. Spearman rank correlation between this checkpoint's target-pool scores
    and Baseline A's final-checkpoint scores on the same pool (TTYD, arXiv:2409.04409).

    See ``AGREEMENT_CAVEAT``: this is a floor alarm in this project, not a selector.
    """
    rho, _ = _spearman(np.asarray(logits, float).reshape(-1),
                       np.asarray(reference_logits, float).reshape(-1))
    return float(rho)


# ---------------------------------------------------------------------------
# validator 4 -- IM (InfoMax)
# ---------------------------------------------------------------------------

def information_maximization(target_logits, *, latent_labels=None,
                             require_balanced: bool = True,
                             tol: float = IM_BALANCE_TOL) -> float:
    """Validator 4. ``IM = H(mean_i p_i) - mean_i H(p_i)`` in NATS. Higher is better.

    First term is *diversity* (is the predicted marginal spread over both classes), second
    is *confidence* (is each individual prediction sharp).

    TRAP 1 -- THE POOL. This must be computed on the fold's BALANCED target training pool
    (391 lenses + 391 negatives), never the 225k unlabeled Q1 pool: at 1:2200 the diversity
    term would reward predicting 50/50, which is catastrophically wrong. Musgrave computes
    IM on the target *train* set and ours is the balanced pool, so this maps directly.

    ``latent_labels`` is used ONLY to assert the pool is balanced and is never read into the
    returned number -- the validator itself stays label-free.
    """
    p = binary_probs(target_logits)
    if require_balanced:
        if latent_labels is None:
            raise ValueError(
                "IM refuses to run without `latent_labels` for the balance assertion "
                "(Trap 1). Pass the pool's latent labels, or set require_balanced=False "
                "and justify it in writing.")
        lat = np.asarray(latent_labels).reshape(-1)
        if lat.size != p.shape[0]:
            raise ValueError(f"latent_labels size {lat.size} != n_scores {p.shape[0]}")
        known = lat[(lat == 0) | (lat == 1)]
        if known.size != lat.size:
            raise ValueError("IM balance assertion needs 0/1 latent labels for every row; "
                             f"got values {sorted(set(lat.tolist()))}")
        frac_pos = float(known.mean())
        if abs(frac_pos - 0.5) > tol:
            raise ValueError(
                f"IM POOL IS NOT BALANCED: positive fraction {frac_pos:.4f} (tol {tol}). "
                "IM's diversity term rewards a balanced predicted marginal; computing it "
                "on an imbalanced pool rewards predicting 50/50 and is invalid (Trap 1).")
    marginal = p.mean(axis=0)                      # (2,)
    diversity = float(_entropy_rows(marginal[None, :])[0])
    confidence = float(_entropy_rows(p).mean())
    return diversity - confidence


# ---------------------------------------------------------------------------
# validator 5 -- SND / NegSND
# ---------------------------------------------------------------------------

def snd(vectors, *, temperature: float = SND_TEMPERATURE) -> float:
    """Validator 5 kernel. Soft Neighborhood Density (Saito et al., ICCV 2021).

    Rows are L2-normalised, the full pairwise cosine-similarity matrix is formed, the
    SELF-SIMILARITY DIAGONAL IS DROPPED (each row becomes n-1 long, as in Musgrave's
    ``pytorch-adapt`` SNDValidator), each row goes through a softmax at ``temperature``, and
    the mean row entropy (nats) is returned. Published direction: higher = better-adapted.

    ``temperature`` defaults to Saito et al.'s 0.05 -- stated, not tuned.
    """
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"snd expects a 2-D (n, d) array, got shape {v.shape}")
    n = v.shape[0]
    if n < 3:
        raise ValueError(f"snd needs at least 3 rows to have a neighborhood, got {n}")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("snd: at least one row has zero norm; cosine similarity undefined")
    v = v / norms
    sim = v @ v.T
    # drop the diagonal -> (n, n-1); a point is never its own neighbour
    off = sim[~np.eye(n, dtype=bool)].reshape(n, n - 1) / float(temperature)
    off -= off.max(axis=1, keepdims=True)           # softmax in a numerically safe form
    e = np.exp(off)
    p = e / e.sum(axis=1, keepdims=True)
    return float(_entropy_rows(p).mean())


def snd_from_logits(target_logits, *, temperature: float = SND_TEMPERATURE) -> float:
    """SND on the network OUTPUT vectors ``[1-s, s]`` -- Saito's published formulation and
    ``pytorch-adapt``'s ``layer="preds"`` default. This is the PRIMARY SND entry point."""
    return snd(binary_probs(target_logits), temperature=temperature)


def snd_from_features(target_features, *, temperature: float = SND_TEMPERATURE) -> float:
    """SND on the backbone features -- the Task-4 brief's literal reading.

    NOTE: on a FROZEN encoder these features are identical at every checkpoint, so this is
    exactly constant and cannot select anything. That makes it a useful SECOND known-null
    control on the frozen runs rather than a real candidate.
    """
    return snd(target_features, temperature=temperature)


def neg_snd(target_logits, *, temperature: float = SND_TEMPERATURE) -> float:
    """Musgrave's NegSND. Reported, NOT adopted -- see Trap 3 / ``DIRECTION``."""
    return -snd_from_logits(target_logits, temperature=temperature)


# ---------------------------------------------------------------------------
# validator 6 -- ATC
# ---------------------------------------------------------------------------

def negative_entropy_score(logits) -> np.ndarray:
    """ATC's score function: the NEGATIVE ENTROPY of the softmax output, per Garg et al.'s
    reference implementation (the ATC-NE variant). Higher = more confident."""
    return -_entropy_rows(binary_probs(logits))


def atc_threshold(source_scores, source_correct) -> float:
    """Pick ``t`` such that ``#(score > t) / N`` equals the source-val accuracy.

    Implemented as the order statistic directly: with ``n_err`` errors, reject the ``n_err``
    lowest-scoring points, i.e. ``t`` = the largest rejected score. With distinct scores this
    makes the pass-fraction EXACTLY the accuracy. Ties (which saturation produces in bulk)
    can only push the realised pass-fraction below the accuracy; ``atc`` reports the realised
    value so the shortfall is visible rather than silent.
    """
    s = np.asarray(source_scores, dtype=np.float64).reshape(-1)
    c = np.asarray(source_correct).reshape(-1).astype(bool)
    if s.size != c.size:
        raise ValueError(f"scores size {s.size} != correctness size {c.size}")
    n_err = int(s.size - int(c.sum()))
    if n_err <= 0:
        return -np.inf                              # accept everything: accuracy is 1.0
    if n_err >= s.size:
        return np.inf                               # accept nothing: accuracy is 0.0
    return float(np.sort(s)[n_err - 1])


def atc(source_val_logits, source_val_labels, target_logits, *,
        return_detail: bool = False):
    """Validator 6. Average Thresholded Confidence (Garg et al., arXiv:2201.04234).

    Predicts target ACCURACY as the fraction of target points whose negative-entropy score
    exceeds a threshold calibrated so the source-val pass-fraction equals source-val
    accuracy. Higher = better.

    TRAP 2 -- THIS IS EXPECTED TO FAIL, AND THAT IS A RESULT. At source accuracy ~0.999 the
    threshold sits near the bottom of the score range, so the target estimate saturates near
    1 and carries no information. We run it anyway: "confidence-transfer methods are defeated
    by source saturation" is reportable, and it is the SAME ROOT CAUSE as source-val AUROC
    being dead. Do not silently drop it.
    """
    sv = np.asarray(source_val_logits, dtype=np.float64).reshape(-1)
    y = np.asarray(source_val_labels).reshape(-1)
    pred = (sv >= 0.0).astype(np.int64)             # logit >= 0  <=>  sigmoid >= 0.5
    correct = (pred == y.astype(np.int64))
    acc = float(correct.mean())

    s_src = negative_entropy_score(sv)
    t = atc_threshold(s_src, correct)
    s_tgt = negative_entropy_score(target_logits)
    est = float(np.mean(s_tgt > t))

    if not return_detail:
        return est
    return {
        "estimate": est,
        "threshold": t,
        "source_accuracy": acc,
        "source_pass_fraction": float(np.mean(s_src > t)),
        "n_source": int(sv.size),
        "n_target": int(np.asarray(target_logits).size),
    }


# ---------------------------------------------------------------------------
# unit tests (deliverable 1)
# ---------------------------------------------------------------------------

def _test_im_known_answer():
    """IM on a synthetic balanced set with a KNOWN answer."""
    # Perfectly confident and perfectly balanced: half predict class 1, half class 0.
    #   marginal = [0.5, 0.5] -> H = ln 2;  every row entropy = 0  ->  IM = ln 2.
    big = 40.0                                       # sigmoid(40) == 1 to float64
    logits = np.array([big] * 50 + [-big] * 50)
    lat = np.array([1] * 50 + [0] * 50)
    got = information_maximization(logits, latent_labels=lat)
    assert abs(got - np.log(2.0)) < 1e-9, got
    print(f"  [PASS] IM confident+balanced == ln2 ({got:.12f})")

    # All-confident but COLLAPSED to one class: marginal = [0,1] -> H = 0; rows 0 -> IM = 0.
    got = information_maximization(np.array([big] * 100), latent_labels=lat)
    assert abs(got - 0.0) < 1e-9, got
    print(f"  [PASS] IM collapsed-to-one-class == 0 ({got:.12f})")

    # Maximally UNCERTAIN (all logits 0 -> p = [.5,.5]): marginal H = ln2, row H = ln2 -> 0.
    got = information_maximization(np.zeros(100), latent_labels=lat)
    assert abs(got - 0.0) < 1e-9, got
    print(f"  [PASS] IM all-uncertain == 0 ({got:.12f})")

    # Closed form on a non-degenerate case: 100 rows at s=0.8, so marginal = [0.2,0.8]
    # and every row has the same entropy -> diversity == confidence -> IM = 0 exactly.
    s = 0.8
    logit = float(np.log(s / (1 - s)))
    got = information_maximization(np.full(100, logit), latent_labels=lat)
    assert abs(got) < 1e-12, got
    print(f"  [PASS] IM identical-rows == 0 (diversity cancels confidence) ({got:.2e})")

    # Trap 1: an imbalanced pool must FAIL LOUDLY, not silently return a number.
    bad = np.array([1] * 90 + [0] * 10)
    try:
        information_maximization(logits, latent_labels=bad)
    except ValueError as e:
        assert "NOT BALANCED" in str(e), str(e)
        print("  [PASS] IM refuses an imbalanced pool (Trap 1)")
    else:
        raise AssertionError("IM accepted an imbalanced pool -- Trap 1 guard is missing")

    # ...and refuses to run at all with no labels to check against.
    try:
        information_maximization(logits)
    except ValueError as e:
        assert "require_balanced" in str(e)
        print("  [PASS] IM refuses to run without the balance assertion")
    else:
        raise AssertionError("IM ran without a balance assertion")


def _test_snd_orthogonal():
    """SND on an ORTHOGONAL-features case, where the answer is known in closed form."""
    # Orthonormal rows: every off-diagonal cosine similarity is exactly 0, so each row's
    # softmax is uniform over its n-1 neighbours and the entropy is exactly ln(n-1) --
    # the maximum a row can have. This is the maximal-density case.
    n = 8
    v = np.eye(n)
    got = snd(v)
    assert abs(got - np.log(n - 1)) < 1e-12, (got, np.log(n - 1))
    print(f"  [PASS] SND orthogonal rows == ln(n-1) ({got:.12f} vs {np.log(n-1):.12f})")

    # Scale invariance: SND is defined on cosine similarity, so scaling rows changes nothing.
    got2 = snd(v * 17.3)
    assert abs(got2 - got) < 1e-12, (got, got2)
    print("  [PASS] SND is scale-invariant (cosine, not dot product)")

    # Two tight, well-separated clusters: each point's mass concentrates on its own cluster,
    # so at T=0.05 the entropy approaches ln(n/2 - 1), well BELOW the orthogonal case.
    half = np.zeros((n, 3))
    half[: n // 2] = [1.0, 0.0, 0.0]
    half[n // 2:] = [0.0, 1.0, 0.0]
    half += 1e-6 * np.random.default_rng(0).standard_normal(half.shape)
    clustered = snd(half)
    assert clustered < got, (clustered, got)
    assert abs(clustered - np.log(n // 2 - 1)) < 1e-6, (clustered, np.log(n // 2 - 1))
    print(f"  [PASS] SND two tight clusters == ln(n/2-1) ({clustered:.9f}) < orthogonal")

    # The diagonal really is excluded. The ORTHOGONAL case is where this bites hardest: a
    # self-similarity of 1.0 against neighbours at 0.0 is a 1/T = 20-nat gap, so keeping the
    # diagonal would put essentially all the row mass on the point itself and collapse the
    # entropy to ~0 instead of the correct ln(n-1).
    # (In the tight-cluster case above the diagonal would NOT dominate -- self-similarity is
    # 1.0 and so is same-cluster similarity, so it would merely add one more member. That is
    # why the check is done here and not there.)
    sim_full = (v @ v.T) / SND_TEMPERATURE
    sim_full -= sim_full.max(axis=1, keepdims=True)
    e = np.exp(sim_full)
    p = e / e.sum(axis=1, keepdims=True)
    with_diag = float(_entropy_rows(p).mean())
    assert with_diag < 1e-6, with_diag
    assert abs(got - np.log(n - 1)) < 1e-12
    print(f"  [PASS] self-similarity diagonal is excluded (with-diag would collapse to "
          f"{with_diag:.2e} vs the correct {got:.4f})")

    # Degenerate inputs fail loudly rather than returning a nan.
    for bad, why in [(np.eye(2), "n<3"), (np.zeros((4, 3)), "zero-norm row")]:
        try:
            snd(bad)
        except ValueError:
            print(f"  [PASS] SND rejects {why}")
        else:
            raise AssertionError(f"SND accepted {why}")


def _test_atc_same_distribution():
    """ATC reproduces the accuracy EXACTLY when source and target are the same distribution."""
    rng = np.random.default_rng(7)
    n = 1000
    y = rng.integers(0, 2, n)
    # A deliberately imperfect model, so accuracy is genuinely < 1 and the test has content.
    logits = np.where(y == 1, rng.normal(1.2, 1.5, n), rng.normal(-1.2, 1.5, n))
    d = atc(logits, y, logits, return_detail=True)          # target IS source
    acc = d["source_accuracy"]
    assert 0.6 < acc < 0.95, f"test is vacuous at accuracy {acc}"
    assert abs(d["estimate"] - acc) < 1e-12, (d["estimate"], acc)
    assert abs(d["source_pass_fraction"] - acc) < 1e-12, d
    print(f"  [PASS] ATC(target==source) == accuracy exactly "
          f"({d['estimate']:.12f} vs {acc:.12f})")

    # A genuinely shifted, less-confident target must be estimated BELOW the source accuracy.
    shifted = rng.normal(0.0, 0.3, n)                       # near-zero logits = unconfident
    est_shift = atc(logits, y, shifted)
    assert est_shift < acc, (est_shift, acc)
    print(f"  [PASS] ATC on an unconfident shifted target drops ({est_shift:.4f} < {acc:.4f})")

    # Perfect source accuracy -> threshold accepts everything -> estimate pinned at 1.0.
    # This is TRAP 2 in miniature: the saturated regime is exactly where ATC goes blind.
    y2 = np.array([0, 1] * 200)
    perfect = np.where(y2 == 1, 30.0, -30.0)
    d2 = atc(perfect, y2, rng.normal(0, 0.1, 500), return_detail=True)
    assert d2["source_accuracy"] == 1.0 and d2["estimate"] == 1.0, d2
    print("  [PASS] ATC saturates to 1.0 at perfect source accuracy (Trap 2 mechanism)")


def _test_helpers():
    """The pieces the three headline tests stand on."""
    # binary_probs rows are normalised probability vectors.
    p = binary_probs([-3.0, 0.0, 3.0])
    assert np.allclose(p.sum(axis=1), 1.0)
    assert abs(p[1, 1] - 0.5) < 1e-12 and p[0, 1] < 0.05 and p[2, 1] > 0.95
    print("  [PASS] binary_probs -> normalised [1-s, s]")

    # entropy in NATS, with 0 log 0 == 0.
    assert abs(_entropy_rows(np.array([[0.5, 0.5]]))[0] - np.log(2)) < 1e-12
    assert abs(_entropy_rows(np.array([[1.0, 0.0]]))[0]) < 1e-12
    print("  [PASS] entropy is in nats and 0 log 0 == 0")

    # sigmoid does not overflow at extreme logits (the confident-prediction regime).
    assert np.all(np.isfinite(_sigmoid([-800.0, 800.0])))
    print("  [PASS] sigmoid is overflow-safe at extreme logits")

    # agreement of a vector with itself is 1, and with its negation is -1.
    x = np.random.default_rng(1).standard_normal(200)
    assert abs(agreement(x, x) - 1.0) < 1e-12
    assert abs(agreement(x, -x) + 1.0) < 1e-12
    print("  [PASS] agreement self == +1, negated == -1")

    # NegSND is exactly the negation, and its DIRECTION entry is the mirror of SND's.
    lg = np.random.default_rng(2).standard_normal(60)
    assert abs(neg_snd(lg) + snd_from_logits(lg)) < 1e-12
    assert DIRECTION["neg_snd"] == -DIRECTION["snd"]
    print("  [PASS] NegSND == -SND and the direction registry mirrors it")


def run_tests() -> int:
    print("=" * 74)
    print("validators.py unit tests")
    print("=" * 74)
    print("helpers:")
    _test_helpers()
    print("IM (validator 4):")
    _test_im_known_answer()
    print("SND (validator 5):")
    _test_snd_orthogonal()
    print("ATC (validator 6):")
    _test_atc_same_distribution()
    print("=" * 74)
    print("ALL VALIDATOR UNIT TESTS PASSED")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="run the unit tests")
    args = ap.parse_args(argv)
    if not args.test:
        print("nothing to do; pass --test")
        return 0
    return run_tests()


if __name__ == "__main__":
    sys.exit(main())
