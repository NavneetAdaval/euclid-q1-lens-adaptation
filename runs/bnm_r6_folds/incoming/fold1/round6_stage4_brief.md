# BRIEF — Round 6, Stage 4: confirmatory folds 1–4

**Paste this whole file to Claude Code. The development run is complete and the pre-registration
is hashed (`d205f678…`). Nothing already built or run is to be redone.**

Repo: `C:\Users\krris\ML\lens_paper\`. Budget **1000**, outcome `no_matched_arm_budget`,
γ = 0.00787, hybrid cadence, read-off rule R1 (final checkpoint).

---

## 1. BEFORE ANY FOLD — CLOSE THE OVERWRITE CLASS

The CORAL incident is fixed at the instance, not the class. A hashed artifact was silently
overwritten by a script run for a different arm and survived only because a backup happened to
exist and the detector happened to be deterministic. Neither is a control, and folds 1–4 run the
same scripts four more times.

1. **`chmod 444` every file whose hash appears in PROJECT.md.** An overwrite must fail loudly
   rather than succeed silently — the same principle as `stats_only_row` not taking the scorer.
   List the protected set in PROJECT.md so it can be audited.
2. **Automatic manifest verification** at the start *and* end of every analysis script: every
   hashed declaration's self-hash checked against PROJECT.md, non-zero exit on mismatch. The
   integrity sweep you ran by hand becomes something that cannot be skipped.
3. **Grep every script for hardcoded output paths** not derived from `--out`, the same way the
   `run_dann_r5.py` literal was found by AST scan rather than by re-reading. Report the count,
   even if zero.

---

## 2. THE FALSIFIER AMBIGUITY — RESOLVED, WITH BOTH READINGS REPORTED

**Primary: the gap at the final checkpoint (step = budget), one value per fold, fold-mean over
four.** Consistent with R1, which already governs every other quantity in the project. Neither a
single fold nor a single checkpoint falsifies.

Rationale, which predates the fold-0 data: a falsifier ordinary noise can break is not testing
the mechanism. Checkpoint sd is 0.0696, so fold 0's +0.2003 excursion sits 2.4 sd above its own
mean — unremarkable across 32 autocorrelated checkpoints. This is the identical argument used
against the +0.04 lower bound before any fold-0 trajectory existed.

**Hazard, to be recorded in the §7.26 annotation:** this reading was settled with fold-0 data
visible, and it is the reading fold 0 survives. The argument predating the data is a defence, not
a proof. Therefore:

> **Report both readings on folds 1–4** — the fold-mean-of-read-off-values (primary) and the
> any-checkpoint maximum (secondary) — and state what each would have concluded.

Also report, as a third figure for comparability with fold 0's preview, the within-fold mean over
checkpoints. All three declared here, before the folds run.

The hashed pre-registration is **not** rewritten. §7.26 annotates it, per §7.21.

---

## 3. A NINTH §7.16 ENTRY — WRITE IT NOW, IT DOES NOT DEPEND ON FOLDS 1–4

The closest sub-ceiling approach was **10.8162** against a ceiling of **10.8413** — a 0.025 gap,
against batch-to-batch scatter of roughly 0.15–0.2 in the same window. So the crossing step is
set by which batch landed where, not by the underlying trajectory: a marginally sharper batch at
step 325 yields a budget of 650 instead of 1000, a **35% swing from noise alone**.

The transferable finding: **a metric-blind budget rule reading a per-batch statistic inherits
that statistic's noise into the budget. Finer checkpointing improves where you looked, not what
you are looking at.** The hybrid cadence's value here was exposing the limitation, not changing
the number.

Record it as a limitation of the rule, applied as written and not corrected — that is what
pre-registration means.

---

## 4. RUN — FOLDS 1–4

Budget 1000, R1 read-off, hybrid cadence, γ = 0.00787. Per-fold step-0 gate non-skippable:
**0.142857 / 0.193878 / 0.153061 / 0.195876**, |diff| = 0.0 exactly.

**Mandatory readouts per fold**, all pre-declared:

| step | role |
|---:|---|
| **1000** | headline — BNM read where its own objective says it is done. Not cross-arm comparable. |
| **4000** | **mandatory** — the λ=0 known-null's budget. Carries every cross-arm contrast. |
| 1500 | matched to FreeMatch |
| 2000 | matched to CORAL |

Folds run to 4000 so all four readouts exist from one run. The budget of 1000 selects the
headline checkpoint; it does not shorten the run.

Log throughout: both morphology instruments (never one quoted as the other, §7.19), normalised
‖A‖\*, mean target prediction entropy, target predicted-positive fraction, the 704/78 split, and
the γ invariant. Collapse floor 0.0292 **gates**; ceiling 0.75 **advisory only**. Conditions
advisory, never aborting. Record GPU architecture per fold. **Verify all logit files come home
before any session ends.**

---

## 5. ANALYSIS

- Primary **TPR@FPR=1e-3**, interpolated, `drop_intermediate=False`, always a rate. Quantisation
  1/98 folds 1–3, **1/97 fold 4**. Off-lattice values are legitimate: FPR=1e-3 falls between 26
  and 27 false positives out of 26,399, forcing an interpolation weight of 0.39898. Any
  off-lattice value should sit at exactly that fraction between adjacent lattice points — a
  cheap integrity check, and one that passed on the development run.
- Per-fold ROC. **Pooling is invalid.**
- Paired **t-interval on 4 differences (df=3)**, not a bootstrap.
- **Contrasts:** BNM(4000) − λ=0(4000) is the headline comparison and the only budget-matched
  one against the known-null. Also BNM(1500) − FreeMatch(1500), BNM(2000) − CORAL(2000),
  BNM(4000) − DANN(4000). Every one carries the §7.18 disclosure: **a budget match removes the
  budget confound only** — head init and γ-tuning budget still travel.
- **No BNM number is differenced against any round-1 number.**
- State results as a **bound**, not a null.
- Disclose multiplicity: count the intervals, state no correction was applied.

---

## 6. WHAT GOES INTO PROJECT.md

1. The BNM row, framed as the **minimal-target-signal control** — target predictions only, no
   domain comparison, no pseudo-labels — not as a fourth ranking entry.
2. §7.20's Prediction 1 scored **failed** on sign, with the CI recorded so the nuance survives.
   Prediction 2 scored directionally held. §2's 704/78 prediction scored under both readings.
3. §3's ninth §7.16 entry.
4. Whichever branch of the round's framing fired — *the minimal signal also buys the gain* or
   *the floor is located above it* — written as the interpretation declared in advance.
5. The §1 protection measures and the hardcoded-path audit count.

---

## 7. ANTI-DRIFT

- **The target is BALANCED**, ~391 unlabeled real lenses : ~391 fresh P2 negatives, ~1:1, per
  fold. The 1:2200 rarity applies only at evaluation.
- **Two morphology instruments, ~0.34 apart.** Never quote one as the other.
- **Results are not portable across GPU architectures** at this precision (§7.7).
- **Compute the interval before writing the sentence.** Fold 0 reads 0.306 at the budget and
  0.398 at 4000; both will be in the table, and the sentence must be written after the folds, not
  anchored to whichever preview reads better.
