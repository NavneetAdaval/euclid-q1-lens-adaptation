# Round-5 DANN — budget decision packet

**This packet contains no recommendation.** It reports the fold-0 development curve and
the quantities that bear on choosing a confirmatory budget from it. The choice is the
run owner's.

Sources, all already on disk and independently recomputed for this packet:
`runs/dann_r5_curves/arm2_dann_curve.csv` (DANN, λ live) and
`arm1_lambda0_curve.csv` (the λ=0 control), both re-scored 2026-09-01 with step-0 gates
passing at `max|diff| = 0.000e+00` over 26,497 logits; and arm 2's `steps.csv` for the
discriminator monitors. Fold 0 only — folds 1–4 do not exist.

---

## 1. The curve

All 32 checkpoints, both arms, unsmoothed and unannotated. λ is DANN's adversarial
coefficient at that step, `tanh(5·step/2000)`.

| step | arm 2 TPR@1e-3 | arm 2 AUROC | arm 1 TPR@1e-3 | arm 1 AUROC | λ | paired Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.346939 | 0.972747 | 0.204082 | 0.938346 | 0.5546 | +0.142857 |
| 500 | 0.285714 | 0.966643 | 0.173469 | 0.896166 | 0.8483 | +0.112245 |
| 750 | 0.438776 | 0.941022 | 0.204082 | 0.923163 | 0.9540 | +0.234694 |
| 1000 | 0.285714 | 0.956772 | 0.081633 | 0.906782 | 0.9866 | +0.204082 |
| 1250 | 0.500000 | 0.954983 | 0.173469 | 0.894611 | 0.9961 | +0.326531 |
| 1500 | 0.173469 | 0.828100 | 0.132653 | 0.871314 | 0.9989 | +0.040816 |
| 1750 | 0.581633 | 0.974449 | 0.193878 | 0.897803 | 0.9997 | +0.387755 |
| 2000 | 0.408163 | 0.940503 | 0.081633 | 0.867775 | 0.9999 | +0.326531 |
| 2250 | 0.438776 | 0.948205 | 0.204082 | 0.888524 | 0.9999 | +0.234694 |
| 2500 | 0.530612 | 0.939301 | 0.153061 | 0.900734 | 0.9999 | +0.377551 |
| 2750 | 0.510204 | 0.940819 | 0.163265 | 0.883737 | 0.9999 | +0.346939 |
| 3000 | 0.500000 | 0.938259 | 0.153061 | 0.898135 | 0.9999 | +0.346939 |
| 3250 | 0.520408 | 0.923609 | 0.122449 | 0.874174 | 0.9999 | +0.397959 |
| 3500 | 0.530612 | 0.955949 | 0.142857 | 0.818131 | 0.9999 | +0.387755 |
| 3750 | 0.510204 | 0.919709 | 0.204082 | 0.882399 | 0.9999 | +0.306122 |
| 4000 | 0.463255 | 0.939555 | 0.132653 | 0.864903 | 0.9999 | +0.330602 |
| 4250 | 0.500000 | 0.941234 | 0.163265 | 0.887658 | 0.9999 | +0.336735 |
| 4500 | 0.428571 | 0.949250 | 0.132653 | 0.828497 | 0.9999 | +0.295918 |
| 4750 | 0.295918 | 0.901698 | 0.132653 | 0.875809 | 0.9999 | +0.163265 |
| 5000 | 0.469388 | 0.927948 | 0.163265 | 0.876577 | 0.9999 | +0.306122 |
| 5250 | 0.453051 | 0.964999 | 0.081633 | 0.855674 | 0.9999 | +0.371418 |
| 5500 | 0.520408 | 0.964768 | 0.153061 | 0.880586 | 0.9999 | +0.367347 |
| 5750 | 0.530612 | 0.922638 | 0.061224 | 0.847387 | 0.9999 | +0.469388 |
| 6000 | 0.561224 | 0.960233 | 0.153061 | 0.892690 | 0.9999 | +0.408163 |
| 6250 | 0.520408 | 0.959368 | 0.112245 | 0.886160 | 0.9999 | +0.408163 |
| 6500 | 0.438776 | 0.945062 | 0.122449 | 0.871937 | 0.9999 | +0.316327 |
| 6750 | 0.428571 | 0.955864 | 0.142857 | 0.873770 | 0.9999 | +0.285714 |
| 7000 | 0.418367 | 0.946430 | 0.061224 | 0.888787 | 0.9999 | +0.357143 |
| 7250 | 0.418367 | 0.971744 | 0.183673 | 0.874438 | 0.9999 | +0.234694 |
| 7500 | 0.540816 | 0.988256 | 0.153061 | 0.880444 | 0.9999 | +0.387755 |
| 7750 | 0.479592 | 0.976982 | 0.163265 | 0.911669 | 0.9999 | +0.316327 |
| 8000 | 0.479592 | 0.971969 | 0.153061 | 0.913433 | 0.9999 | +0.326531 |

arm 2: mean 0.453379, sd 0.091381, min 0.173469, max 0.581633  
arm 1: mean 0.145408, sd 0.040153, min 0.061224, max 0.204082

---

## 2. What each candidate rule selects

Listed in the brief's order. **Not ranked.**

| rule | step | arm 2 TPR@1e-3 there |
|---|---:|---:|
| Last checkpoint | 8000 | 0.479592 |
| Argmax | 1750 | 0.581633 |
| First step within 1 sd of the maximum | 1250 | 0.500000 |
| First step after which no remaining improvement exceeds 1 sd | 1250 | 0.500000 |
| Matched to CORAL x nb3's budget | 2000 | 0.408163 |
| Twice the lambda horizon | 4000 | 0.463255 |

1 sd of arm 2's curve = **0.091381**. Maximum 0.581633 at step 1750.

The third and fourth rules coincide on this curve — both select step 1250.
They are distinct rules that happen to agree here; that is a property of this curve,
not of the rules.

---

## 3. Stability context

Checkpoint-to-checkpoint sd on the identical 32-point 250→8000 grid, fold 0, TPR@1e-3:

| arm | n | mean | sd | sd/mean |
|---|---:|---:|---:|---:|
| DANN arm 2 (λ live) | 32 | 0.4534 | 0.0914 | 0.202 |
| DANN arm 1 (λ = 0) | 32 | 0.1454 | 0.0402 | 0.276 |
| CORAL × nb3 | 32 | 0.4341 | 0.0513 | 0.118 |

**FreeMatch × nb3 has no comparable figure.** Its fold-0 curve stops at step 1500 and
is logged at stride 125, so there is no 8000-step counterpart. No substitute is offered
here.

### Local scatter around each candidate step

The five checkpoints centred on each candidate (±2), so a budget landing beside a large
local excursion is distinguishable from one landing in a quiet stretch.

| candidate step | the 5 checkpoints | local sd | local min | local max |
|---:|---|---:|---:|---:|
| 8000 | 0.5408, 0.4796, 0.4796 | 0.035348 | 0.4796 | 0.5408 |
| 1750 | 0.5000, 0.1735, 0.5816, 0.4082, 0.4388 | 0.153129 | 0.1735 | 0.5816 |
| 1250 | 0.4388, 0.2857, 0.5000, 0.1735, 0.5816 | 0.164915 | 0.1735 | 0.5816 |
| 2000 | 0.1735, 0.5816, 0.4082, 0.4388, 0.5306 | 0.157652 | 0.1735 | 0.5816 |
| 4000 | 0.5306, 0.5102, 0.4633, 0.5000, 0.4286 | 0.040577 | 0.4286 | 0.5306 |

The global sd is 0.0914; the local sd ranges from 0.035 to 0.165 depending on where a
budget lands.

---

## 4. λ at each candidate budget

λ = `tanh(5·step/2000)`. Thresholds, computed rather than assumed:

- λ ≥ 0.99 is first reached at step **1059**
- λ ≥ 0.9999 is first reached at step **1981**

| budget B | steps at λ ≥ 0.99 | fraction of run | fraction at λ ≥ 0.9999 |
|---:|---:|---:|---:|
| 1250 | 192 | **0.154** | 0.000 |
| 1750 | 692 | **0.395** | 0.000 |
| 2000 | 942 | **0.471** | 0.010 |
| 4000 | 2942 | **0.736** | 0.505 |
| 8000 | 6942 | **0.868** | 0.752 |

Development run (8000 steps): **0.868** at λ ≥ 0.99, 0.752 at λ ≥ 0.9999.

> **Two corrections to the brief's framing of this section, flagged rather than applied**
> **silently.** The brief gives the development run's fraction as "75% (6000 of 8000)"
> and says the fraction at B = 2000 is "~0". Measured against the λ ≥ 0.99 threshold the
> brief itself specifies, neither holds: the development run spends **86.8%** of its steps
> at λ ≥ 0.99 (6,942 of 8,000, from step 1,059), and B = 2000 spends **47.1%** there, not
> ~0. The brief's two figures correspond to the *λ ≥ 0.9999* threshold (first reached at
> step 1,981), where the development run is at 75.2% and B = 2000 at 1.0%. Both columns
> are given above so the choice of threshold is visible rather than buried. **No
> conclusion is drawn from either.**

---

## 5. The `d_acc` convergence cross-check

Label-free, and independent of TPR. 250-step rolling means of the discriminator's
training accuracy and the raw domain loss on arm 2. Chance is 0.5; ln 2 = 0.693147.

| step | rolling `d_acc_train` | rolling `loss_domain_raw` |
|---:|---:|---:|
| 250 | 0.7014 | 0.5363 |
| 500 | 0.5734 | 0.6806 |
| 750 | 0.5528 | 0.6876 |
| 1000 | 0.5354 | 0.6904 |
| 1250 | 0.5319 | 0.6909 |
| 1500 | 0.5237 | 0.6922 |
| 1750 | 0.5205 | 0.6919 |
| 2000 | 0.5300 | 0.6910 |
| 2500 | 0.5240 | 0.6915 |
| 3000 | 0.5313 | 0.6913 |
| 4000 | 0.5181 | 0.6927 |
| 6000 | 0.5352 | 0.6913 |
| 8000 | 0.5135 | 0.6929 |

First step at which both are within tolerance of chance **and remain so for the rest of
the run**:

| tolerance on `d_acc` | tolerance on domain loss | first such step |
|---:|---:|---:|
| 0.02 | 0.005 | 7894 |
| 0.03 | 0.01 | 7585 |
| 0.05 | 0.02 | 908 |

The bulk of the convergence happens early — `d_acc` is at 0.535 and the domain loss at
0.690 by step 1,000 — but the series never settles tightly enough to satisfy the two
narrower tolerances until very late, because it continues to fluctuate around chance for
the whole run. Both readings are given; which is the relevant one depends on what the
criterion is meant to capture.

---

## 6. The protocol conflict

Two things in the record point different ways. Both are quoted; **no position is taken**,
and the run owner resolves it.

**Against deriving a budget from this curve —** §7.13 and §7.14 hold that these curves do
not locate a budget, and §7.11 set the precedent for declining the argmax on
winner's-curse grounds. CORAL × nb3's own budget file records the basis explicitly:

> "DECLARED ON COST, NOT DERIVED FROM THE CURVE. The fold-0 development curve does not
> locate a budget, so no budget in 250-8000 is defensible over any other on evidence. The
> choice is therefore an explicit cost decision and is labelled as one."
> — `runs/coral_r4_budget.json`, `grounds.summary`

**For deriving one —** round 5's fold-0 arm was specified at 8000 steps *in order to*
determine the budget for folds 1–4. From `run.json`:

> "PROJECT.md 7.14 fold-0 DEVELOPMENT run for DANN: fix DANN's own budget from its own
> fold-0 curve, before any confirmatory fold exists. Fold 0 is development; no number
> here enters a headline."
> — `runs/dann_arm2_fold0/.../run.json`, `purpose`

A run specified to fix a budget from its curve, under a protocol that holds such curves
cannot fix one. **This packet does not resolve that.**

**Whatever is chosen, the budget file must record which basis was used** — `declared`, or
`derived` under a named rule from §2 — so that a later reader can tell which it was
without reconstructing the reasoning. `coral_r4_budget.json` uses the field
`derived_or_declared` for exactly this.

---

## 7. Status

`CONFIRM_STEPS` is **unset as a decision**: it holds the placeholder value 2000 and
`is_preregistered_budget` is **false** everywhere. `runs/dann_r5_budget.json` records a
**withdrawn** declaration — an earlier session declared 2000 on cost, and that
declaration is withdrawn pending this decision rather than deleted, so the record shows
it happened. `KAGGLE_FOLDS.md` is not updated and will take the budget once there is one.
