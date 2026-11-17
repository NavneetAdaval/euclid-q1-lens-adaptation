# DANN − λ=0: the 704/78 memorisation diagnostic, folds 1–4 (PROJECTv7 §7.54)

> **RETROSPECTIVE AND UNREGISTERED.** retrospective, unregistered: decided 2026-10-24 after all results were known; in no pre-registration. Rule declared in `plan.json` (self-certifying `9dee7d6a02c17be0aece2a9439f2989ab28d40122588080e90c1bf18538f5ffa`) before any value was computed.

Driver sha256 `c4fc514623da6a9acf7b159a608624ee33b56cd8ffa026251c76ed66834734dd`. Venue: NVIDIA GeForce RTX 4050 Laptop GPU, torch 2.7.1+cu118 (CUDA 11.8), fp32 (amp=False). Views: aug='none' (deterministic), identical for both arms.

## Known-null (λ=0, step 4000, §7.29's weak-aug views replayed)

| fold | step_4000.pt == resume/latest.pt | gap (replay) | §7.29 | checked | match |
|---|:--:|---:|---:|:--:|:--:|
| 1 | yes | +0.0282 | +0.2128 | no | — |
| 2 | yes | −0.1362 | −0.1362 | yes | yes |
| 3 | yes | +0.0624 | +0.0624 | yes | yes |
| 4 | yes | −0.0569 | −0.0569 | yes | yes |

Result: **PASS**. Fold 1 is not checked: 7.29 drew its fold-1 views before anything seeded torch, and torch's default seed on this install is time-based (two fresh processes draw different numbers), so those views cannot be recreated. Fold 1's replay value is computed and reported, not checked. Its weight equality IS checked.

## Primary: normalised ‖A‖* gap at step 4000

| arm | fold 1 | fold 2 | fold 3 | fold 4 | fold-mean |
|---|---:|---:|---:|---:|---:|
| DANN x nb3 | +0.0237 | +0.0250 | +0.1580 | +0.2288 | +0.1089 |
| lambda=0 | +0.2119 | −0.1226 | −0.0396 | −0.0089 | +0.0102 |
| DANN − λ=0 | −0.1882 | +0.1476 | +0.1977 | +0.2377 | +0.0987 |

## Secondary: within-fold mean of the gap over 16 checkpoints

| arm | fold 1 | fold 2 | fold 3 | fold 4 | fold-mean |
|---|---:|---:|---:|---:|---:|
| DANN x nb3 | +0.0803 | +0.0641 | +0.0874 | +0.1247 | +0.0891 |
| lambda=0 | +0.1050 | −0.0671 | +0.0748 | −0.0676 | +0.0113 |
| DANN − λ=0 | −0.0248 | +0.1313 | +0.0126 | +0.1924 | +0.0779 |

## Contrasts DANN − λ=0 (paired t, df = 3, 95%)

| reading | mean | sd | 95% CI | excludes 0 | folds same sign |
|---|---:|---:|---|:--:|:--:|
| **primary** (step 4000) | +0.0987 | 0.1948 | [−0.2113, +0.4086] | no | no |
| secondary (within-fold mean) | +0.0779 | 0.1012 | [−0.0832, +0.2390] | no | no |

## Each arm's own gap against zero (descriptive)

| arm | reading | mean | 95% CI |
|---|---|---:|---|
| DANN x nb3 | step 4000 | +0.1089 | [−0.0531, +0.2708] |
| DANN x nb3 | within-fold mean | +0.0891 | [+0.0483, +0.1300] |
| lambda=0 | step 4000 | +0.0102 | [−0.2170, +0.2374] |
| lambda=0 | within-fold mean | +0.0113 | [−0.1346, +0.1571] |

## Any-checkpoint maximum (descriptive only: no interval, no contrast)

Reason: a maximum over noisy reads has a positive null (7.28): the maximum of 16 noisy reads is positive whether or not any gap exists

| arm | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---:|---:|---:|---:|
| DANN x nb3 | +0.1932 (step 2500) | +0.1732 (step 3000) | +0.1841 (step 750) | +0.2739 (step 2250) |
| lambda=0 | +0.2894 (step 3000) | +0.0698 (step 1000) | +0.1994 (step 500) | +0.0943 (step 2000) |

## Entropy and predicted-positive gaps (descriptive, fold-means)

| arm | entropy @4000 | entropy, within-fold mean | pred-pos @4000 | pred-pos, within-fold mean |
|---|---:|---:|---:|---:|
| DANN x nb3 | −0.0060 | −0.0169 | +0.0705 | +0.0469 |
| lambda=0 | −0.0205 | −0.0123 | +0.0064 | −0.0058 |

## λ=0 at step 4000: §7.29's views against this analysis's views

The main tables' λ=0 step-4000 gaps differ from 7.29's because the views differ (no augmentation here vs one weak-augmented view in 7.29). Both sets are reported side by side. These values are not directly comparable with BNM's logged gaps (7.28), which used weak-augmented views.

| fold | §7.29 (one weak-aug view) | replay (same views) | this analysis (no aug) |
|---|---:|---:|---:|
| 1 | +0.2128 | +0.0282 | +0.2119 |
| 2 | −0.1362 | −0.1362 | −0.1226 |
| 3 | +0.0624 | +0.0624 | −0.0396 |
| 4 | −0.0569 | −0.0569 | −0.0089 |

## Interpretation (sentence fixed in the plan)

> Whether part of the adaptation gain comes from fitting these particular target images remains undetermined: DANN − λ=0 = +0.0987 [−0.2113, +0.4086] at the read-off step.

> Secondary reading (the within-fold mean over the 16 checkpoints; it is the secondary, not the primary): DANN − λ=0 = +0.0779 [−0.0832, +0.2390].

In every case: no multiplicity correction, and the intervals are probably too narrow because the folds' target pools overlap (paper §5.3).
