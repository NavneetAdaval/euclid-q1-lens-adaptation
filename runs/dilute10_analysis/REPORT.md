# 1:10 diluted-target experiment — analysis report

Plan `runs/dilute10_analysis/plan.json`, self-certifying sha256 `68367e25c784334c4bdbc2097c53a8cca8141636f097d60f5ae0bf7329e416ed`. Parts 1–3 govern. Stored logits only; every value through `metrics.metric_suite`, `_common.pool_filter`, `analyze_lambda0_r6.paired_t` and `compare_methods.compare`. Cleaned pool (D1, rule `b83fa019…`) unless stated. Folds 1–4.

## Reproduction check (§2)

**PASSED** — Table 5 per-fold TPR rows and Table 4 balanced − λ=0 rows, exact at four decimals.

| row | printed (f1, f2, f3, f4, mean, sd) | match |
|---|---|---|
| lambda=0 @1500 | 0.2245, 0.1735, 0.1837, 0.1959, 0.1944, 0.0221 | True |
| lambda=0 @2000 | 0.2245, 0.2245, 0.1735, 0.2371, 0.2149, 0.0282 | True |
| lambda=0 @4000 | 0.2245, 0.1735, 0.1429, 0.1959, 0.1842, 0.0346 | True |
| BNM x nb3 @4000 | 0.3980, 0.4592, 0.3776, 0.3814, 0.4040, 0.0378 | True |
| FreeMatch x nb3 @1500 | 0.5000, 0.5918, 0.4286, 0.5464, 0.5167, 0.0697 | True |
| DANN x nb3 @4000 | 0.5408, 0.7143, 0.3163, 0.4227, 0.4985, 0.1706 | True |
| CORAL x nb3 @2000 | 0.4694, 0.6327, 0.5000, 0.4845, 0.5216, 0.0751 | True |

| contrast | steps | printed (Δ, lo, hi, signs) | match |
|---|---|---|---|
| DANN x nb3 − λ=0 | 4000/4000 | +0.3144, +0.0565, +0.5722, 4/4 | True |
| CORAL x nb3 − λ=0 | 2000/4000 | +0.3375, +0.1889, +0.4860, 4/4 | True |
| FreeMatch x nb3 − λ=0 | 1500/4000 | +0.3325, +0.2272, +0.4378, 4/4 | True |
| CORAL x nb3 − λ=0 | 2000/2000 | +0.3068, +0.1834, +0.4301, 4/4 | True |
| FreeMatch x nb3 − λ=0 | 1500/1500 | +0.3223, +0.1984, +0.4463, 4/4 | True |
| BNM x nb3 − λ=0 | 4000/4000 | +0.2199, +0.1383, +0.3014, 4/4 | True |

IDs check: **PASSED** (24 files: negatives = splits.json order, 265 removed by D1, positives = the fold's positives in the balanced and λ=0 comparators).

## Per arm × fold (§3.1), diluted, cleaned pool

**TPR@10⁻³**

| arm | step | f1 | f2 | f3 | f4 | mean | sd |
|---|---|---|---|---|---|---|---|
| DANN × nb3 | 4000 | 0.5306 | 0.5204 | 0.5102 | 0.5155 | 0.5192 | 0.0087 |
| BNM × nb3 | 4000 | 0.3980 | 0.5408 | 0.2857 | 0.2887 | 0.3783 | 0.1203 |
| CORAL × nb3 | 2000 | 0.2449 | 0.2041 | 0.1531 | 0.1753 | 0.1943 | 0.0397 |
| FreeMatch × nb3 | 1500 | 0.2959 | 0.6327 | 0.3469 | 0.4742 | 0.4374 | 0.1502 |

**AUROC**

| arm | step | f1 | f2 | f3 | f4 | mean | sd |
|---|---|---|---|---|---|---|---|
| DANN × nb3 | 4000 | 0.9510 | 0.8971 | 0.8808 | 0.7972 | 0.8815 | 0.0637 |
| BNM × nb3 | 4000 | 0.9445 | 0.9804 | 0.9218 | 0.9096 | 0.9391 | 0.0311 |
| CORAL × nb3 | 2000 | 0.7813 | 0.8089 | 0.7990 | 0.7610 | 0.7876 | 0.0211 |
| FreeMatch × nb3 | 1500 | 0.9660 | 0.9874 | 0.9731 | 0.9781 | 0.9762 | 0.0090 |

**completeness @24% purity**

| arm | step | f1 | f2 | f3 | f4 | mean | sd |
|---|---|---|---|---|---|---|---|
| DANN × nb3 | 4000 | 0.5102 | 0.5204 | 0.4490 | 0.4639 | 0.4859 | 0.0348 |
| BNM × nb3 | 4000 | 0.3163 | 0.5204 | 0.2449 | 0.1134 | 0.2988 | 0.1700 |
| CORAL × nb3 | 2000 | 0.0408 | 0.1020 | 0.0408 | 0.0619 | 0.0614 | 0.0289 |
| FreeMatch × nb3 | 1500 | 0.1735 | 0.6327 | 0.2551 | 0.4227 | 0.3710 | 0.2030 |

**purity @50% completeness (1:2200)**

| arm | step | f1 | f2 | f3 | f4 | mean | sd |
|---|---|---|---|---|---|---|---|
| DANN × nb3 | 4000 | 0.2979 | 0.2382 | 0.1984 | 0.1889 | 0.2308 | 0.0495 |
| BNM × nb3 | 4000 | 0.0915 | 0.2707 | 0.0571 | 0.0179 | 0.1093 | 0.1117 |
| CORAL × nb3 | 2000 | 0.0194 | 0.0147 | 0.0072 | 0.0072 | 0.0121 | 0.0060 |
| FreeMatch × nb3 | 1500 | 0.0991 | 0.3726 | 0.0803 | 0.1776 | 0.1824 | 0.1336 |

## Contrasts (§3.2–3.3), paired t, df = 3

| contrast | steps | mean | 95% CI | excl. 0 | signs |
|---|---|---|---|---|---|
| G DANN (primary) | 4000/4000 | +0.3350 | [+0.2913, +0.3787] | True | 4/4 |
| G BNM (primary) | 4000/4000 | +0.1941 | [+0.0029, +0.3854] | True | 4/4 |
| G CORAL (primary) | 2000/4000 | +0.0102 | [-0.0251, +0.0454] | False | 3/4 |
| G CORAL (sensitivity) | 2000/2000 | -0.0206 | [-0.0740, +0.0329] | False | 3/4 |
| G FreeMatch (primary) | 1500/4000 | +0.2533 | [-0.0042, +0.5107] | False | 4/4 |
| G FreeMatch (sensitivity) | 1500/1500 | +0.2431 | [-0.0229, +0.5090] | False | 4/4 |
| D DANN (balanced − diluted) | 4000/4000 | -0.0206 | [-0.2840, +0.2427] | False | 2/4 |
| D BNM (balanced − diluted) | 4000/4000 | +0.0257 | [-0.1076, +0.1591] | False | 2/4 |
| D CORAL (balanced − diluted) | 2000/2000 | +0.3273 | [+0.1925, +0.4621] | True | 4/4 |
| D FreeMatch (balanced − diluted) | 1500/1500 | +0.0793 | [-0.0800, +0.2385] | False | 3/4 |

**Multiplicity:** 10 t-intervals plus 2 bootstrap intervals, no correction.

## Branches (§3.4)

Part 3's no-verdict rule: applied fractions CORAL nb3 0.9955, 0.9950, 0.9955, 0.9955; FreeMatch nb3 0.9987, 0.9987, 0.9980, 0.9980 — fires on none.

| arm | G (primary) | D | branch | qualifier | sensitivity-form branch |
|---|---|---|---|---|---|
| DANN × nb3 | +0.3350 [+0.2913, +0.3787] | -0.0206 [-0.2840, +0.2427] | **SURVIVES** | fires | — |
| BNM × nb3 | +0.1941 [+0.0029, +0.3854] | +0.0257 [-0.1076, +0.1591] | **SURVIVES** | fires | — |
| CORAL × nb3 | +0.0102 [-0.0251, +0.0454] | +0.3273 [+0.1925, +0.4621] | **COLLAPSES** | n/a | COLLAPSES |
| FreeMatch × nb3 | +0.2533 [-0.0042, +0.5107] | +0.0793 [-0.0800, +0.2385] | **COLLAPSES** | n/a | COLLAPSES |

- **DANN × nb3:** With a 1:10 target drawn without the labels, DANN × nb3 still raises TPR@10⁻³ over the control by G = +0.3350 [+0.2913, +0.3787], and we detect no loss against the balanced target (D = -0.0206 [-0.2840, +0.2427]). …though the interval admits losing up to 0.2427 of the original 0.3144.
- **BNM × nb3:** With a 1:10 target drawn without the labels, BNM × nb3 still raises TPR@10⁻³ over the control by G = +0.1941 [+0.0029, +0.3854], and we detect no loss against the balanced target (D = +0.0257 [-0.1076, +0.1591]). …though the interval admits losing up to 0.1591 of the original 0.2199.
- **CORAL × nb3:** We cannot detect a gain for CORAL × nb3 with a 1:10 target: G = +0.0102 [-0.0251, +0.0454].
- **FreeMatch × nb3:** We cannot detect a gain for FreeMatch × nb3 with a 1:10 target: G = +0.2533 [-0.0042, +0.5107].

## Descriptive (§3.5)

Retained fraction G / (balanced − λ=0), point estimates: DANN (primary) 1.0657; BNM (primary) 0.8829; CORAL (primary) 0.0301; FreeMatch (primary) 0.7616; CORAL (matched) -0.0670; FreeMatch (matched) 0.7541.

**FreeMatch decomposition** (recomputed from the persisted step_8000 logits through pool_filter + metric_suite (cleaned pool); equal to gate_values.json cleaned_pool: True):

| | start (frozen head) | frozen part | unfrozen part | total |
|---|---|---|---|---|
| diluted | 0.3019 | +0.1177 | +0.1355 | +0.2533 |
| balanced (paper §6.3) | 0.4118 | +0.2276 | +0.1049 | +0.3325 |

CORAL frozen start: diluted 0.1944, round 1 0.2098. λ=0 @4000 = 0.1842.

**Diluted frozen heads, step 8000, TPR per fold** (round 1 beside):

| stage | f1 | f2 | f3 | f4 | mean | round-1 f1–f4 | round-1 mean |
|---|---|---|---|---|---|---|---|
| FreeMatch | 0.2551 | 0.3367 | 0.2653 | 0.3505 | 0.3019 | 0.4184, 0.4184, 0.3776, 0.4330 | 0.4118 |
| CORAL | 0.1939 | 0.1939 | 0.1735 | 0.2165 | 0.1944 | 0.1837, 0.2245, 0.1837, 0.2474 | 0.2098 |

**Frozen stages against round 1** (compare_methods, class-conditional paired bootstrap, shared negatives resampled once per replicate, B = 1000, seed 0):

| stage | round 1 | diluted | diluted − round 1 | 95% CI |
|---|---|---|---|---|
| FreeMatch | 0.4118 | 0.3019 | -0.1099 | [-0.1634, -0.0689] |
| CORAL | 0.2098 | 0.1944 | -0.0154 | [-0.0461, +0.0076] |

**CORAL frozen-stage head growth** (PROJECTv7 7.46 (LayerNorm gain norm trunk.0.weight; ||head||_2), unit gain sqrt(640) = 25.3; head_l2 re-read from gate_values.json): LayerNorm gain norm diluted 97.0/103.7/98.2/102.2 vs round 1 22.4/33.1/24.0/23.4; ‖head‖₂ (gate_values) 134.16/139.05/133.42/137.10 vs round 1 67.5/72.6/68.2/66.0. FreeMatch frozen ‖head‖₂ 28.25/28.26/28.44/28.19.

**Run bookkeeping**

| run | f1 | f2 | f3 | f4 |
|---|---|---|---|---|
| dann skips / applied fraction | 3 / 0.9992 | 6 / 0.9985 | 3 / 0.9992 | 3 / 0.9992 |
| coral_nb3 skips / applied fraction | 9 / 0.9955 | 10 / 0.9950 | 9 / 0.9955 | 9 / 0.9955 |
| freematch_nb3 skips / applied fraction | 2 / 0.9987 | 2 / 0.9987 | 3 / 0.9980 | 3 / 0.9980 |
| bnm, frozen stages skips | not recorded (BNM, FreeMatch frozen); CORAL frozen is fp32 CPU | | | |
| coral_nb3 feature-norm advisory firings | 2000 | 2000 | 2000 | 2000 |
| dann lens fraction run-mean (expected) | 0.083362 (0.083353) | 0.083340 (0.083353) | 0.083337 (0.083353) | 0.083366 (0.083373) |
| bnm lens fraction run-mean (expected) | 0.083366 (0.083353) | 0.083345 (0.083353) | 0.083342 (0.083353) | 0.083365 (0.083373) |
| coral_nb3 lens fraction run-mean (expected) | 0.090918 (0.090909) | 0.090914 (0.090909) | 0.090906 (0.090909) | 0.090908 (0.090909) |
| freematch_nb3 lens fraction run-mean (expected) | 0.090913 (0.090909) | 0.090903 (0.090909) | 0.090897 (0.090909) | 0.090911 (0.090909) |
| coral_frozen lens fraction run-mean (expected) | 0.090909 (0.090909) | 0.090915 (0.090909) | 0.090908 (0.090909) | 0.090912 (0.090909) |
| freematch_frozen lens fraction run-mean (expected) | 0.090905 (0.090909) | 0.090909 (0.090909) | 0.090905 (0.090909) | 0.090912 (0.090909) |

**BNM predicted-positive fraction** (trace.csv, target batches; balanced run as context), steps 125 / 500 / 1000 / 2000 / 3000 / 4000:

| fold | diluted | balanced (context) |
|---|---|---|
| 1 | 0.059 / 0.043 / 0.078 / 0.066 / 0.051 / 0.105 | 0.352 / 0.312 / 0.355 / 0.273 / 0.363 / 0.359 |
| 2 | 0.066 / 0.070 / 0.066 / 0.074 / 0.031 / 0.059 | 0.277 / 0.355 / 0.316 / 0.324 / 0.289 / 0.367 |
| 3 | 0.074 / 0.102 / 0.086 / 0.086 / 0.078 / 0.074 | 0.273 / 0.336 / 0.320 / 0.297 / 0.348 / 0.316 |
| 4 | 0.086 / 0.047 / 0.086 / 0.059 / 0.051 / 0.035 | 0.312 / 0.324 / 0.359 / 0.348 / 0.328 / 0.336 |

## Prediction scoring (§3.6)

Relative loss D / (balanced − λ=0): DANN -0.0657, BNM 0.1171, CORAL 0.9699, FreeMatch 0.2384.

| id | component | score | rests on |
|---|---|---|---|
| 1a | DANN degrades (D point > 0) | **failed** | D = -0.0206 |
| 1b(i) | G point < +0.31 (as stated) | **failed** | G = +0.3350 |
| 1b(ii) | G point < reproduced balanced - lambda=0 | **failed** | G = +0.3350 vs +0.3144 |
| 1c(i) | D is positive: point | **failed** | D = -0.0206 |
| 1c(ii) | D is positive: interval excludes zero above | **failed** | D CI [-0.2840, +0.2427] |
| 1d | G's interval (probably) includes zero | **failed** | G CI [+0.2913, +0.3787] |
| 1e | branch COLLAPSES or SHRINKS | **failed** | branch SURVIVES |
| 1e' | 'with a wide D' | **not testable** | 'wide' is not operationalised in part 1 |
| 1f | mechanism (marginal alignment pulls lenses toward galaxies; prior collapses) | **not testable** | no stored quantity measures it |
| 2a | CORAL degrades (D point > 0) | **held** | D = +0.3273 |
| 2b(i) | less than DANN, absolute | **failed** | D CORAL +0.3273 vs DANN -0.0206 |
| 2b(ii) | less than DANN, relative | **failed** | relative loss CORAL 0.9699 vs DANN -0.0657 |
| 2c | G's interval excludes zero | **failed** | G CI [-0.0251, +0.0454] |
| 2d | branch SHRINKS or SURVIVES | **failed** | branch COLLAPSES |
| 2e | mechanism (moment matching a weaker pull) | **not testable** | no stored quantity |
| 3a | BNM degrades (D point > 0) | **held** | D = +0.0257 |
| 3b | the most of the four in relative terms | **failed** | relative loss DANN -0.0657, BNM 0.1171, CORAL 0.9699, FreeMatch 0.2384 |
| 3c | G's interval may include zero | **not testable** | permissive wording; G CI [+0.0029, +0.3854] |
| 3d(i) | predicted-positive fraction > 0.09 at step 4000 | **mixed 1/4** | step-4000 values f1 0.1055, f2 0.0586, f3 0.0742, f4 0.0352 |
| 3d(ii) | predicted-positive fraction > 0.09 at any logged step | **held** | max per fold f1 0.1055, f2 0.0938, f3 0.1016, f4 0.0938 |
| 3e | mechanism (high-scoring negatives) | **not testable** | needs a new measurement |
| 4a | FreeMatch degrades (D point > 0) | **held** | D = +0.0793 |
| 4b(i) | less than BNM, absolute | **failed** | D FreeMatch +0.0793 vs BNM +0.0257 |
| 4b(ii) | less than BNM, relative | **failed** | relative loss FreeMatch 0.2384 vs BNM 0.1171 |
| 4c | G's interval still excludes zero | **failed** | G CI [-0.0042, +0.5107] |
| 4d(i) | head start shrinks: point | **held** | diluted frozen 0.3019 vs round-1 frozen 0.4118 |
| 4d(ii) | head start shrinks: bootstrap interval wholly below zero | **held** | diluted - round 1 = -0.1099 [-0.1634, -0.0689] |
| 4e | mechanism (uniform fairness target; adaptive thresholds) | **not testable** | no stored quantity |

No prediction was amended; the four predictions are part 1's, verbatim.

## Sensitivities (§3.8, appendix only)

| contrast | cleaned excl. 0 | original | matches | reduced | matches |
|---|---|---|---|---|---|
| G DANN (primary) 4000/4000 | True | +0.2814 [+0.2388, +0.3240] | True | +0.3336 [+0.2846, +0.3826] | True |
| G BNM (primary) 4000/4000 | True | +0.1610 [+0.0471, +0.2749] | True | +0.2015 [+0.0036, +0.3994] | True |
| G CORAL (primary) 2000/4000 | False | +0.0333 [-0.0006, +0.0672] | True | -0.0009 [-0.0601, +0.0584] | True |
| G CORAL (sensitivity) 2000/2000 | False | -0.0052 [-0.0561, +0.0457] | True | -0.0248 [-0.0901, +0.0405] | True |
| G FreeMatch (primary) 1500/4000 | False | +0.2099 [+0.0051, +0.4147] | False | +0.2457 [-0.0035, +0.4948] | True |
| G FreeMatch (sensitivity) 1500/1500 | False | +0.1997 [+0.0071, +0.3923] | False | +0.2491 [-0.0187, +0.5169] | True |
| D DANN | False | -0.0002 [-0.2233, +0.2230] | True | -0.0191 [-0.2788, +0.2406] | True |
| D BNM | False | +0.0310 [-0.0915, +0.1534] | True | +0.0220 [-0.1083, +0.1524] | True |
| D CORAL | True | +0.2660 [+0.1065, +0.4255] | True | +0.3364 [+0.2066, +0.4662] | True |
| D FreeMatch | False | +0.0715 [-0.0542, +0.1972] | True | +0.0805 [-0.0782, +0.2392] | True |

Per-fold values on both pools are in `results.json` → `sensitivities`.

## Inputs

Every planned input re-hashed at run time (219 files) and equal to the plan. Driver `scripts/analyze_dilute10.py` sha256 `2296fc388187f21bd053d207e5d84ce5c94a0b8fdefd8e49b3db167b77cf0000` (plan recorded `74f3876f9baa0fe31a9ad771986443f5d91cfc97fd5f5956e1d1a5dc0b6cf12e`). Edits since the plan: score_predictions: D unpacked as (DANN, CORAL, BNM, FM) from ARMS, whose order is (DANN, BNM, CORAL, FM), so components 2a/2b(i)/3a/4b(i) read BNM's D as CORAL's and vice versa in the first full run; fixed to (DANN, BNM, CORAL, FM). No other line changed; every contrast, branch and descriptive value is unaffected.

