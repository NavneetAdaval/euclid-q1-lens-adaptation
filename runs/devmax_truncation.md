# Development-budget maximum: at what truncation cap is the decay unambiguous?

PROJECT.md §7.14, open item. Read-only analysis of artifacts already on disk; no GPU, no training, nothing existing modified.

All numbers are **TPR@FPR=1e-3 as a rate** (§2/§9). The metric's one-positive resolution is **0.0102** as a rate.

## 1. Files used

| role | file | curve points |
|---|---|---|
| fold 0 development curve (full 8000) | `runs/freematch_fold0_nb3_unfrozen/curve.csv` | 65 |
| fold 0 per-checkpoint logit dumps | `runs/freematch_fold0_nb3_unfrozen/logits/step_*.npz` | 65 |
| fold 1 partial curve (secondary) | `runs/freematch_nb3_5folds/_x/freematch_nb3_unfrozen_fold1/curve.csv` | 28 |
| fold 1 per-checkpoint logit dumps | `runs/freematch_nb3_5folds/_x/freematch_nb3_unfrozen_fold1/logits/step_*.npz` | 28 |

Fold 0: steps 0–8000 every 125, 98 positives against 26399 shared negatives.
Fold 1: steps 0–3375 every 125 — PARTIAL -- run killed mid-flight (PROJECT.md 7.11); lab-record only, not cited in any paper (7.13 lab-record note)

Provenance: recomputing TPR@1e-3 from the logit dumps reproduces `curve.csv` with max|diff| = 0.0e+00 (fold 0) and 0.0e+00 (fold 1). The fast tail-ROC used inside the bootstrap reproduces `metrics.tpr_at_fpr` with max|diff| = 0.0e+00.

## 2. The detector (fixed before any truncated result was looked at)

```
------------------------------------------------------------------------------------
Stated here, printed to stdout, and written to the JSON *before* the numbers, so a
later reader can check the definition did not move to fit a tidy answer.

CANDIDATE SET.  For a cap C, the checkpoints in the window are
    S(C) = { s in curve steps : 0 < s <= C }.
Step 0 is EXCLUDED. It is the LP->FT initialisation -- round 1's own frozen head --
not a product of the unfrozen run, and its value is the pre-registered step-0 gate
constant (fold 0: 0.387755, PROJECT.md 7.11). Including it would let a fixed constant
that no training produced drive the "early" arm. (Sensitivity variant V3 includes it.)

PARTITION (primary, "P1-quartile").
    LATE(C)  = { s in S(C) : s >  0.75 * C }      -- the last quartile of the window
    EARLY(C) = { s in S(C) : s <= 0.75 * C }
Both are defined by fractions of the window, not by absolute steps, so the detector
asks the same question of every cap.

STATISTIC.
    Delta(C) = mean_{EARLY(C)} TPR@1e-3  -  mean_{LATE(C)} TPR@1e-3
Delta > 0 means "late is worse", the 7.13 structure. The EARLY arm is a MEAN, not a
max: taking the best earlier region would import a winner's-curse bias that inflates
Delta under the null and would make the known-null control fire by construction.
(The "best earlier window" form the brief offers as an example is run anyway, as
sensitivity variant V2, reported WITH its own permutation-null false-positive rate.)

DECISION RULE.
    "decay detectable at cap C"  <=>  the 95% bootstrap CI of Delta(C) has lo > 0.
    "late is BETTER at cap C"    <=>  that CI has hi < 0.
    otherwise: not distinguishable.
"Not distinguishable" is neither "no effect" nor "no decay" (PROJECT.md 9).

BOOTSTRAP -- the 7.12 fold-aware class-conditional scheme, reduced to one fold.
The compared arms here are CHECKPOINTS OF ONE FOLD, so the shared-pool structure that
7.12 handles across folds applies across checkpoints: every checkpoint scores the SAME
~26k negative pool, so a negative-tail fluctuation moves every checkpoint's FPR=1e-3
threshold together, and the checkpoints are positively correlated, not independent.
Each draw b therefore:
  1. resamples the shared negative pool ONCE (with replacement, same size) -- one
     negative-tail realisation applied to EVERY checkpoint in the window;
  2. resamples the ~98 fold positives ONCE (with replacement, same size) -- the SAME
     positive indices applied to EVERY checkpoint, which is what makes the EARLY-vs-LATE
     comparison PAIRED;
  3. lets each checkpoint derive its OWN FPR=1e-3 threshold from that one negative
     realisation (correct: each checkpoint is a different model scoring a common pool);
  4. forms Delta_b from those per-checkpoint TPRs.
B = 5000 draws, seed 0, percentile interval, alpha = 0.05 -- the 7.12 settings.
Because the draws are shared across caps, the per-cap results are directly comparable
and differ only in which checkpoints enter the two arms.

KNOWN-NULL CONTROLS (PROJECT.md 9). Two, because the first one is degenerate.
  N1  The window [0, 1500] on its own -- the brief's suggestion. Fold 0's argmax sits
      at 1250/1500, so this window is the RISE, not the decay, and the detector must
      not report decay in it. Note this window IS the C = 1500 cap row: at that cap the
      null control and the measurement are the same computation, which is itself the
      reason no cap at or below 1500 can be "defensible" on this evidence.
  N2  A step-permutation null, which is the non-degenerate one. Within each cap the
      EARLY/LATE membership is permuted at random (destroying step-locked structure but
      preserving the checkpoints, the scores and the correlation between them), the full
      bootstrap is re-run, and the fraction of permutations whose CI excludes zero is
      measured. A correctly calibrated detector fires on ~2.5% of permutations at the
      one-sided level implied by alpha = 0.05. Materially more than that means the
      detector is broken and no cap result from it may be believed.
      This is 7.13's own null: "averaging suppressed noise exactly as if the folds
      shared no step-locked structure" -- here we impose no step-locked structure and
      check the detector agrees.

SENSITIVITY VARIANTS (pre-registered, reported, NOT used to pick the headline):
  V2 "best-earlier": EARLY arm = the best contiguous earlier window of the same width
      as LATE. Winner's-curse biased; reported only beside its own N2 rate.
  V3 "with-step0":   step 0 admitted to the EARLY arm.
  V4 "last-third":   LATE = last third of the window instead of the last quartier.

MULTIPLICITY. Eight caps are tested on one curve and the windows are nested, so the
smallest cap that fires is a selected minimum over eight strongly dependent tests. No
correction is applied to the per-cap CIs (they are reported as what they are); the N2
permutation rate is what bounds the family-wise false-alarm behaviour, and the
smallest-defensible-cap statement is made against it, not against the per-cap p-value.

REPORTING. TPR@FPR=1e-3 as a RATE throughout (PROJECT.md 9). No counts of lenses.
The one-positive resolution of the metric, 1/98 = 0.0102, is quoted as a rate scale.
```
definition sha256 `301826021baf9dac28009328615b10d31b6b9e50886fbfb248439388d8bdd4ed`

## 3. Result per cap — fold 0 (development)

| cap | ckpts | EARLY mean | LATE mean | Δ | 95% CI | verdict |
|---|---|---|---|---|---|---|
| 1500 | 12 | 0.4494 | 0.5000 | -0.0506 | [-0.1029, +0.0283] | not distinguishable |
| 2000 | 16 | 0.4621 | 0.4337 | +0.0284 | [-0.0434, +0.0952] | not distinguishable |
| 2500 | 20 | 0.4574 | 0.4163 | +0.0411 | [-0.0265, +0.0897] | not distinguishable |
| 3000 | 24 | 0.4520 | 0.4252 | +0.0269 | [-0.0340, +0.0697] | not distinguishable |
| 4000 | 32 | 0.4453 | 0.4375 | +0.0078 | [-0.0473, +0.0565] | not distinguishable |
| 5000 | 40 | 0.4464 | 0.4153 | +0.0311 | [-0.0147, +0.0766] | not distinguishable |
| 6000 | 48 | 0.4400 | 0.4056 | +0.0344 | [-0.0068, +0.0865] | not distinguishable |
| 8000 | 64 | 0.4314 | 0.3364 | +0.0951 | [+0.0442, +0.1405] | DECAY DETECTABLE (late worse) |

## 4. Known-null controls

**N1 — first 1500 steps only.** Δ = -0.0506, 95% CI [-0.1029, +0.0283] → **not distinguishable**.

The brief's control window. Fold 0's argmax is at 1250/1500, so this window is the RISE; the detector must NOT report decay in it. It is the SAME computation as the cap=1500 row -- which is exactly why no cap at or below 1500 can be defended from this evidence.

**N2 — step-permutation null (the non-degenerate control).** EARLY/LATE membership is permuted within the window, destroying step-locked structure while preserving the checkpoints, their scores and their correlations.

| cap | primary fires on | V2 'best earlier' fires on | nominal | permutation p (primary) |
|---|---|---|---|---|
| 1500 | 0.0% | 0.4% | 2.5% | 0.968 |
| 2000 | 0.6% | 3.9% | 2.5% | 0.145 |
| 2500 | 0.3% | 3.0% | 2.5% | 0.045 |
| 3000 | 0.1% | 2.1% | 2.5% | 0.092 |
| 4000 | 0.5% | 0.7% | 2.5% | 0.331 |
| 5000 | 0.4% | 0.2% | 2.5% | 0.018 |
| 6000 | 1.2% | 2.7% | 2.5% | 0.008 |
| 8000 | 6.2% | 49.5% | 2.5% | 0.000 |

## 5. Sensitivity variants

| cap | V2 best-earlier | V3 with step 0 | V4 last-third | V5 endpoint (post-hoc) |
|---|---|---|---|---|
| 1500 | -0.0170 [-0.0871, +0.0558] | -0.0567 [-0.1088, +0.0207] | -0.0071 [-0.0625, +0.0587] | -0.0636 [-0.1271, +0.0649] |
| 2000 | +0.0332 [-0.0383, +0.0944] | +0.0227 [-0.0489, +0.0885] | +0.0041 [-0.0529, +0.0738] | +0.0390 [-0.0517, +0.1470] |
| 2500 | +0.0510 [-0.0286, +0.1061] | +0.0367 [-0.0324, +0.0861] | +0.0308 [-0.0371, +0.0813] | +0.0733 [-0.0312, +0.1407] |
| 3000 | +0.0442 [-0.0255, +0.0908] | +0.0235 [-0.0376, +0.0655] | +0.0290 [-0.0344, +0.0702] | +0.0175 [-0.0816, +0.0967] |
| 4000 | +0.0281 [-0.0485, +0.0791] | +0.0055 [-0.0501, +0.0539] | +0.0056 [-0.0439, +0.0492] | +0.0153 [-0.0677, +0.1091] |
| 5000 | +0.0484 [-0.0258, +0.1082] | +0.0292 [-0.0172, +0.0744] | +0.0244 [-0.0252, +0.0635] | -0.0211 [-0.1062, +0.0542] |
| 6000 | +0.0565 [-0.0128, +0.1250] | +0.0330 [-0.0093, +0.0845] | +0.0358 [-0.0064, +0.0820] | +0.0238 [-0.0597, +0.1222] |
| 8000 | +0.1186 [+0.0459, +0.1767] | +0.0942 [+0.0430, +0.1390] | +0.0908 [+0.0431, +0.1376] | +0.0409 [-0.0476, +0.1365] |

V5 was added **after** the primary table was seen and is labelled post-hoc; it is reported because under R1 the budget *is* the selector, so the decision-relevant quantity at a cap is the endpoint. It does not answer the question — the pre-registered primary does.

## 6. Secondary — fold 1 partial curve

PARTIAL -- run killed mid-flight (PROJECT.md 7.11); lab-record only, not cited in any paper (7.13 lab-record note)

| cap | ckpts | EARLY mean | LATE mean | Δ | 95% CI | verdict |
|---|---|---|---|---|---|---|
| 1500 | 12 | 0.4048 | 0.4218 | -0.0170 | [-0.0918, +0.0476] | not distinguishable |
| 2000 | 16 | 0.4090 | 0.4286 | -0.0196 | [-0.0918, +0.0476] | not distinguishable |
| 2500 | 20 | 0.4143 | 0.3939 | +0.0204 | [-0.0469, +0.0741] | not distinguishable |
| 3000 | 24 | 0.4104 | 0.4065 | +0.0040 | [-0.0533, +0.0607] | not distinguishable |

## 7. The answer

**The smallest cap at which the pre-registered detector fires is 8000, and it is the only cap that fires.** Every cap from 1500 to 6000 returns *not distinguishable*.

So the honest answer to §7.14's open question is: **this analysis does not support any reduction below 8000.** §7.14 asked "whether the decay is unambiguous by step 3000" — on fold 0's own development curve it is not, and on fold 1's partial curve nothing is detectable at any cap it reaches either.

Three things keep that from being over-read:

- The **permutation readout is more permissive than the pre-registered rule** and would put the onset earlier — p = 1500→0.968, 2000→0.145, 2500→0.045, 3000→0.092, 4000→0.331, 5000→0.018, 6000→0.008, 8000→0.000. But that series is **not monotone in the cap**, which is the signature of noise wandering across the threshold rather than of a real earlier onset. A criterion that says "yes at 2500, no at 3000, no at 4000, yes at 5000" cannot be used to choose a cap.
- **The gap between the two readouts is itself informative.** The bootstrap CI is conditional on the checkpoint set: it prices evaluation-data sampling error and treats the trained models as fixed. The permutation null prices checkpoint-to-checkpoint scatter instead. They disagree at 5000–6000, and where two defensible null models disagree, "unambiguous" is not the right word.
- **Even the reduction the permissive readout would license does not solve the affordability problem.** At the measured throughput (431.6 min for 8000 steps plus 65 curve points) a 6000-step cap is ≈5.4 h/fold against 7.2 h — a ~25% saving. The caps that would actually make a per-method development run cheap (≤3000 ≈ ≤2.7 h/fold) are precisely the caps at which **nothing** is detectable, on either readout, on either fold.

**Consequence for §7.14.** The per-method development-run maximum cannot be set from evidence at 98 positives per fold. It has to be set as a declared cost/coverage choice, disclosed as such, rather than justified as "long enough to see the shape". This is the same negative that §7.13 reports one level down: §7.13 found the peak location unmeasurable; this finds the decay ONSET unmeasurable below the full budget. The one structure that survives at 8000 survives only at 8000.

**Detector integrity.** The known-null control passes: on the first-1500 window the detector reports no decay (it correctly sees the rise instead). The step-permutation null fires on ≤1.2% of permutations for every cap up to 6000 against a 2.5% nominal — conservative, not broken. At cap 8000 it fires on 6.25% of permutations, above nominal because the 8000-step checkpoint set has real non-step-locked scatter that a conditional CI does not price; the observed permutation p at 8000 is below 1/2001, far under that floor, so the cap-8000 detection survives its own false-alarm rate. The pre-registered warning about the "best earlier region" form was also borne out: V2 fires on 49.5% of permutations at cap 8000 — it is a broken detector, and it is the operationalisation the brief offered as an example.
