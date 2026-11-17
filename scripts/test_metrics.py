"""
test_metrics.py
===============
Unit tests for the NEW low-FPR completeness + purity metric family in metrics.py
(auprc / purity_at_completeness / completeness_at_purity / prevalence / metric_suite).
The existing tpr_at_fpr / auroc / paired_bootstrap are unchanged and exercised elsewhere.

Every check is against a KNOWN answer -- a hand-computed confusion matrix, a
hand-computed average precision, or an analytic reachability limit -- so the tests catch
the correction-DIRECTION bug (raw vs 1:2200 prevalence-corrected purity) and the
interpolation / nan-guard discipline (unreachable target -> nan, never extrapolation).

Run:
    python scripts\\test_metrics.py
Exit 0 on success, 1 on a failed check.
"""

from __future__ import annotations

import sys
import traceback
import warnings

import numpy as np

import metrics as m

_fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


def approx(a, b, tol=1e-9):
    return np.isfinite(a) and np.isfinite(b) and abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# 1. AUPRC: hand cross-check + separable + random==prevalence
# ---------------------------------------------------------------------------

def test_auprc():
    print("\n[1] AUPRC (average precision)")
    # y=[1,0,1,0] ranked by score desc -> positives retrieved at ranks 1 and 3.
    #   AP = (P@rank1 * dR) + (P@rank3 * dR) = 1.0*0.5 + (2/3)*0.5 = 0.833333...
    ap = m.auprc([1, 0, 1, 0], [4.0, 3.0, 2.0, 1.0])
    check(approx(ap, (1.0 + 2.0 / 3.0) / 2.0), f"hand AP = 0.83333 (got {ap:.6f})")

    # perfectly separable -> AP = 1.0
    check(approx(m.auprc([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8]), 1.0),
          "separable set -> AP = 1.0")

    # random scores -> AP ~ positive prevalence (the documented random baseline)
    rng = np.random.default_rng(0)
    n = 200_000
    y = (rng.random(n) < 0.1).astype(int)
    s = rng.random(n)
    ap_rand = m.auprc(y, s)
    prev = m.prevalence(y)
    check(abs(ap_rand - prev) < 0.01,
          f"random AP {ap_rand:.4f} ~= prevalence {prev:.4f}")

    # single class -> nan
    check(np.isnan(m.auprc([1, 1, 1], [1.0, 2.0, 3.0])), "single class -> nan")


# ---------------------------------------------------------------------------
# 2. purity_at_completeness: hand-known confusion matrix, raw AND 1:2200-corrected
# ---------------------------------------------------------------------------

def test_purity_at_completeness_hand():
    print("\n[2] purity_at_completeness on a hand-known confusion matrix")
    # 10 positives scored 10..1; 2 'high' negatives (8.5, 6.5) score above the 5th
    # positive (=6), the other 8 negatives score below 1. At completeness 0.5 the
    # threshold sits at the 5th positive:
    #   TP = 5, FP = 2, N_P = 10, N_N = 10  => tpr=0.5, fpr=0.2, pool ratio N_N/N_P = 1.
    pos = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    neg = [8.5, 6.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03]
    y = [1] * 10 + [0] * 10
    s = pos + neg

    raw = m.purity_at_completeness(y, s, target_completeness=0.5, nn_np_ratio=None)
    check(raw["status"] == "ok", "raw: status ok")
    check(approx(raw["completeness"], 0.5), "raw: completeness = 0.5 (target achieved)")
    check(approx(raw["fpr"], 0.2), f"raw: fpr = 0.2 (got {raw['fpr']})")
    # raw-pool purity = TP/(TP+FP) = 5/7
    check(approx(raw["purity"], 5.0 / 7.0), f"raw purity = 5/7 = 0.714286 (got {raw['purity']:.6f})")

    corr = m.purity_at_completeness(y, s, target_completeness=0.5, nn_np_ratio=2200)
    # corrected: purity = tpr/(tpr + fpr*2200) = 0.5/(0.5 + 0.2*2200) = 0.5/440.5
    check(approx(corr["purity"], 0.5 / 440.5),
          f"1:2200-corrected purity = 0.5/440.5 = 0.0011351 (got {corr['purity']:.7f})")

    # DIRECTION SANITY: our pool (1:1) is lens-RICHER than 1:2200, so correcting to the
    # rarer positives must LOWER purity (catches the correction-direction bug).
    check(corr["purity"] < raw["purity"],
          f"corrected purity {corr['purity']:.6f} < raw purity {raw['purity']:.6f} "
          "(lens-rich pool corrected to rarer 1:2200 lowers purity)")
    check(approx(corr["nn_np_ratio"], 2200.0) and approx(raw["nn_np_ratio"], 1.0),
          "reported nn_np_ratio: 2200 (corrected) vs 1.0 (raw pool)")


# ---------------------------------------------------------------------------
# 3. completeness_at_purity: monotone example + inverse consistency + unreachable nan
# ---------------------------------------------------------------------------

def test_completeness_at_purity():
    print("\n[3] completeness_at_purity (dual) + reachability nan-guard")
    # Monotone construction: 5 positives (50..10), 4 negatives interleaved (45,35,25,15)
    # + 6 low negatives (<10). N_P=5, N_N=10 => raw pool ratio = 2.
    #   completeness 0.6 <=> TP=3 (thr=30), FP=2 (45,35) => tpr=0.6, fpr=0.2
    #   raw purity there = 0.6/(0.6+0.2*2) = 0.6
    pos = [50, 40, 30, 20, 10]
    neg = [45, 35, 25, 15, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5]
    y = [1] * 5 + [0] * 10
    s = pos + neg

    # forward: purity@0.6-completeness (raw) = 0.6
    fwd = m.purity_at_completeness(y, s, 0.6, nn_np_ratio=None)
    check(approx(fwd["purity"], 0.6), f"forward purity@0.6-comp = 0.6 (got {fwd['purity']})")

    # dual: completeness at raw purity 0.6 = 0.6 (inverse consistency)
    dual = m.completeness_at_purity(y, s, target_purity=0.6, nn_np_ratio=None)
    check(dual["status"] == "ok", "dual: status ok")
    check(approx(dual["completeness"], 0.6),
          f"completeness@purity=0.6 (raw) = 0.6 (got {dual['completeness']})")

    # UNREACHABLE: put all negatives ABOVE all positives so the best attainable purity is
    # bounded well under 1 -> a high target purity is unreachable -> nan (NOT extrapolated).
    posU = [5, 4, 3, 2, 1]
    negU = [9, 8, 7]                       # 3 negatives above every positive
    yU = [1] * 5 + [0] * 3
    sU = posU + negU
    # best purity along the curve is 5/8 = 0.625 (full completeness); target 0.9 is unreachable
    unr = m.completeness_at_purity(yU, sU, target_purity=0.9, nn_np_ratio=None)
    check(np.isnan(unr["completeness"]) and unr["status"] == "unreachable",
          "unreachable target purity -> nan (status 'unreachable')")

    # target purity > 1 is trivially unreachable
    check(np.isnan(m.completeness_at_purity(y, s, 1.5)["completeness"]),
          "target purity 1.5 (>1) -> nan")

    # single class -> nan
    check(np.isnan(m.completeness_at_purity([0, 0, 0], [1.0, 2.0, 3.0], 0.5)["completeness"]),
          "single class -> nan")


# ---------------------------------------------------------------------------
# 4. interpolation + nan-guards for purity_at_completeness
# ---------------------------------------------------------------------------

def test_nan_guards():
    print("\n[4] nan-guards: unreachable completeness -> nan, never extrapolation")
    pos = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    neg = [8.5, 6.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03]
    y = [1] * 10 + [0] * 10
    s = pos + neg
    # completeness > 1 is out of the observed [0,1] range -> nan (refuse to extrapolate)
    r = m.purity_at_completeness(y, s, target_completeness=1.5)
    check(np.isnan(r["purity"]) and r["status"] == "extrapolation",
          "target completeness 1.5 -> nan (status 'extrapolation')")
    # single class -> nan
    r2 = m.purity_at_completeness([1, 1, 1, 1], [1.0, 2.0, 3.0, 4.0], 0.5)
    check(np.isnan(r2["purity"]) and r2["status"] == "single_class",
          "single class -> nan (status 'single_class')")


# ---------------------------------------------------------------------------
# 5. metric_suite matches the individual public functions (single source of truth)
# ---------------------------------------------------------------------------

def test_suite_consistency():
    print("\n[5] metric_suite agrees with the individual public functions")
    rng = np.random.default_rng(7)
    n_pos, n_neg = 120, 6000
    s_pos = rng.normal(1.5, 1.0, n_pos)
    s_neg = rng.normal(0.0, 1.0, n_neg)
    y = np.concatenate([np.ones(n_pos, int), np.zeros(n_neg, int)])
    s = np.concatenate([s_pos, s_neg])

    suite = m.metric_suite(y, s, target_fpr=1e-3, target_completeness=0.5,
                           nn_np_ratio=2200, target_purity=0.24)
    check(approx(suite["tpr"], m.tpr_at_fpr(y, s, 1e-3)), "suite tpr == tpr_at_fpr")
    check(approx(suite["auroc"], m.auroc(y, s)), "suite auroc == auroc")
    check(approx(suite["auprc"], m.auprc(y, s)), "suite auprc == auprc")
    check(approx(suite["purity_raw"],
                 m.purity_at_completeness(y, s, 0.5, None)["purity"]),
          "suite purity_raw == purity_at_completeness(raw)")
    check(approx(suite["purity_corr"],
                 m.purity_at_completeness(y, s, 0.5, 2200)["purity"]),
          "suite purity_corr == purity_at_completeness(2200)")
    csuite = suite["comp_at_purity"]
    cfunc = m.completeness_at_purity(y, s, 0.24, 2200)["completeness"]
    check((np.isnan(csuite) and np.isnan(cfunc)) or approx(csuite, cfunc),
          "suite comp_at_purity == completeness_at_purity(0.24, 2200)")
    # corrected purity < raw purity on this lens-rich synthetic pool (prevalence >> 1:2200)
    check(suite["purity_corr"] < suite["purity_raw"],
          "corrected purity < raw purity (synthetic pool is lens-richer than 1:2200)")


def main():
    warnings.simplefilter("ignore")            # nan-guards warn by design; tests assert nan
    print("=" * 72)
    print("metrics.py  low-FPR completeness+purity family unit tests")
    print("=" * 72)
    test_auprc()
    test_purity_at_completeness_hand()
    test_completeness_at_purity()
    test_nan_guards()
    test_suite_consistency()

    print("\n" + "=" * 72)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for f in _fails:
            print("   - " + f)
        return 1
    print("ALL METRICS CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
