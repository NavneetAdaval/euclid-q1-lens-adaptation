# Round-4 CORAL — build record and pre-registered predictions

Built 2026-08-16. **Nothing has been trained.** This records what exists, what was verified,
and what was predicted *before* either arm ran, so a later reader can check that nothing
moved.

`run_coral.py` and every round-1 artifact are untouched. Round-1 CORAL
(`runs/bakeoff_coral/`) is read for exactly two things — the LP→FT phase-1 head, and the
step-0 known-null that proves it loaded — and **no round-4 number is ever differenced
against it.** The comparison this round supports is **nb3 − nb0**, which differs in one thing.

| file | role |
|---|---|
| [coral_round4.py](scripts/coral_round4.py) | `Fp32CoralLoss`, the feature-scale monitor, gradient-reachability probes, the test-label lockout, the CORAL curve instrument |
| [run_coral_round4.py](scripts/run_coral_round4.py) | the two-arm runner, `--calibrate`, `--grad-check`, confound pinning |
| [test_coral_round4.py](scripts/test_coral_round4.py) | the pre-handover suite — **133 checks, 0 failures** |

---

## 1. The two arms

Identical in every respect except `n_blocks`. Both image-space, `aug="weak"`, fp16 autocast
+ GradScaler, true batch 256 source + 256 target, LP→FT from round-1 CORAL fold-0
step-8000, step-0 hard gate, checkpoints every 500, logit dumps every curve point (125),
agreement column, collapse-gate trace.

```
coral_nb0_control   n_blocks=0
coral_nb3           n_blocks=3   (grad checkpointing ON — mandatory, §7.1a)
```

**§7.1a layer-wise LR, matched exactly:** head_lr = backbone_lr = **1e-4**, lr_decay
**0.75**, weight_decay **0.05**, no LR schedule. `backbone_lr` defaults to `None`, which
resolves to head_lr — Zoobot's head group gets `lr` and encoder block *i*=0 gets
`lr · lr_decay⁰`, the same value.

> **Flagged, not hidden.** That is 10× lower head LR and 500× higher weight decay than
> round-1 CORAL used (1e-3 / 1e-4). It is adopted because the brief says match §7.1a
> exactly, and because both arms get it identically so it cannot bias nb3 − nb0. It does
> mean **the frozen arm is not a re-run of round 1** and must not be read as one.

**Budget: 8000 steps, both arms.** §7.14 left the development maximum open;
`runs/devmax_truncation.json` answers it on FreeMatch's own curve — the decay is detectable
only at a cap of 8000, and every cap from 1500 to 6000 is not distinguishable. There is no
evidence-backed shorter maximum to adopt. Both arms run the full length, which also closes
§7.10's acknowledged gap: that control was truncated and could say nothing about the later
steps.

---

## 2. The gradient path — VERIFIED, with the evidence

`train_skeleton.encode_features:89-93` flags this as an open item: *"Combining n_blocks>0
with a DA loss is a SEPARATE decision that must be made (and measured) explicitly."* It was
reasoned through for FreeMatch and never for CORAL. It is now measured, not argued.

**Yes — at `n_blocks=3` the CORAL loss backprops into the encoder, and that is intended.**

The chain: `ZoobotEncoder.forward` sets `_frozen = (n_blocks == 0)`, so at nb3 the forward
runs *without* `no_grad` and the features carry grad → `standardizer` (buffers only,
differentiable pass-through) → `head.forward_with_features` → `src_adapter`/`tgt_adapter`
→ `Fp32CoralLoss`. Both domain branches reach the encoder.

Measured by backpropagating **the CORAL term alone**, with no supervised term
(`coral_reachable_params`):

| arm | elements with a non-zero gradient | tensors | module prefixes |
|---|---|---|---|
| `nb0` | **1,032** | 2 | `head.trunk` |
| `nb3` | **14,809,028** | 122 | `head.trunk`, `encoder.stages` |

Asserted structurally, and all of it passes:

- **All 640 LayerNorm gains** carry gradient on both arms.
- **The classifier is unreachable** on both arms — CORAL consumes the adapter, which is
  upstream of it.
- `nb0`: **nothing in the encoder** is reachable.
- `nb3`: **exactly `stages.1`, `stages.2`, `stages.3`** — the stem and `stages.0` are not,
  matching §7.1a's block convention. All **14,835,520** unfrozen encoder elements are in the
  graph, reproducing §7.1a's count exactly.

### Two measured caveats, recorded because the raw counts do not land on the round numbers

**(a) The 1,032 on nb0 is not 640.** The extra 392 are LayerNorm *bias* entries with a
bitwise non-zero gradient. They are **float cancellation around an analytically zero value**,
not a path: a bias shifts every sample of *both* domains identically, covariance is
shift-invariant, and the first-moment term is a *difference* of means, so the shift cancels
twice over. Settled functionally rather than by choosing an epsilon —

- bias grad rms / gain grad rms = **4.7e-06**
- shifting the bias by **+1.0** moves the CORAL loss by **8.6e-07** relative (fp32
  cancellation in the one-pass covariance formula), while scaling the gain by 1.10 moves it
  by **2.1e-01** — a factor of **2×10⁵** apart.

So the honest statement is: **the reachable knob on nb0 is the 640 gains.**

**(b) The 14,809,028 on nb3 is 27,524 short of 640 + 14,835,520.** Those elements have an
*exactly* zero gradient, concentrated in `stages.3.blocks.*.mlp.fc2.weight` — GELU units no
sample in the 4-image probe batch activated. That is 0.19%, a property of the probe batch,
not of the graph, which is why the **tensor set** is the structural claim and the element
count is reported with its shortfall attributed.

---

## 3. The CORAL loss

**Attachment point unchanged from round 1** and identical on both arms:
`head.forward_with_features(...)[1]` — post-adapter, post-LayerNorm, 640-d. Pinned as a
config key so a future edit that moves it fails the precondition check.

**Constant gamma. `warmup_cosine` is gone from this path** and
`coral_gamma_at_step` is deliberately not imported. Round 1 decayed gamma to ≈0.100·peak by
its step-8000 read-off, suppressing the CORAL term exactly where it was measured; Sun &
Saenko (arXiv:1607.01719 §4) set λ so the classification and CORAL losses are comparable at
the **end** of training.

**fp32 covariance under fp16 autocast** — `Fp32CoralLoss` wraps the whole moment computation
in an autocast-disabled block. Verified bit-exact against the shipped `CoralLoss` at fp32
(forward, gradient, and both component terms), so the guard changes nothing except under
autocast.

> **Measured, and it corrects the assumption behind the requirement.** At the *nominal*
> post-LayerNorm scale the unguarded fp16 path is **not** worse: rel err 7.7e-06 against the
> guarded 8.0e-06. torch accumulates the matmul in fp32 and the entries of zᵀz stay ~3×10².
> The guard earns its place at **larger** feature scales, which an unfrozen encoder is free
> to reach: sweeping the scale, the unguarded path stays fine at 2× and 4× and goes
> **non-finite at 5× (‖z‖ ≈ 126)**, where zᵀz overflows fp16 and `CoralLoss`'s own finite
> assert aborts the run. The guarded path is exact at every scale tested through 8×.
> So the guard is **insurance against a scale change, not a fix for a live bug** — and the
> scale change is precisely what §4 predicts might happen.

### CHANGE 1 — the target EMA is now OFF by default, on both arms

**Superseding the "kept at round 1's value" decision below**, which was made before the
measurement was available. `--coral-target-ema` re-enables it; the default is off, both arms
must match, and `coral_target_ema` is a **PINNED** key so a mismatch fails the precondition
check.

**The reasoning turns on whether the EMA has a fixed point, and unfreezing removes it.**

*Frozen*, it does. The target features are constant, so the buffer converges to the target
pool's own covariance — a strictly better estimate than any single 256-draw batch. That is a
real benefit and it is why round 1 used it.

*Unfrozen*, it does not. The encoder moves every step, so the buffered reference was
computed by a model that no longer exists: **stale by construction, not by lag.** Two
measured consequences, neither of which is Deep CORAL:

- **It destroys the objective's translation invariance.** Deep CORAL is shift-invariant —
  covariance is shift-invariant and the first-moment term is a difference of means. Measured
  response to a constant shift of the representation: **8.6e-07** relative without the EMA,
  **6.0e+03** with it against a stale reference. Verified restored with the EMA off
  (9.9e-08).
- **It attenuates the target-side gradient to 10%.** The reference is
  `0.9·history(detached) + 0.1·current(grad)`, so the two domains contribute asymmetrically
  to a loss that treats them symmetrically.

> **KNOWN LIMITATION OF TURNING IT OFF — recorded, and deliberately NOT fixed.**
> Without the EMA the target covariance is a single-batch estimate: **256 samples in 640
> dimensions**, so the sample covariance is **singular** (rank ≤ 255 < 640). Deep CORAL's
> validated regime (Sun & Saenko arXiv:1607.01719) had d ≪ n; ours does not.
>
> The loss nonetheless remains well-defined — the Frobenius distance between two singular
> matrices is finite, and the 0.1 shrinkage toward (tr Σ/d)·I plus the 1e-6 jitter keep both
> shrunk matrices positive-definite. And the **source** covariance is estimated from the
> same 256 draws in the same 640 dimensions, so the rank deficiency is **symmetric across
> domains**. A symmetric deficiency inflates the variance of the estimated distance; it does
> not bias it toward either domain.
>
> It is therefore a **noise cost, not a bias**, and it is accepted rather than patched.

### CHANGE 2 — a hard feature-norm abort

`--feat-norm-ceiling`, **default 110**, PINNED, checked **every step** on both arms (not
once per curve point: at `curve_every=125` a run could travel far past the ceiling before
anyone looked). The abort message carries the **step number, both domains' norms, the raw
CORAL value and the ceiling** — everything needed to read the failure without re-running.

The measured basis: the unguarded fp16 covariance path goes **non-finite at ‖z‖ ≈ 126**;
LP→FT init sits at **~66**; the 500-step calibration reached **82 by step 400**. That is a
live trajectory, not a hypothetical, and an 8000-step run must not silently produce NaNs
hours in. The value 110 is an **engineering choice** — below the measured failure point,
with room above the observed trajectory — not a measurement. `--feat-norm-ceiling 0`
disables the guard as an explicit opt-out.

### The original `target_ema` finding (superseded by Change 1, kept for the record)

Round 1's `target_ema=True, momentum=0.9` was initially kept (so the loss configuration was
not quietly re-tuned), but it has a consequence worth stating:

**Deep CORAL's objective is shift-invariant. This implementation, with the EMA on, is not.**
The alignment reference is `0.9·history(detached) + 0.1·current(grad)`, and the buffered
half was computed with an *earlier* model. Shift the representation and the source mean
moves by the full δ while the target reference moves by only 0.1·δ, so the first-moment term
acquires a `(0.9·δ)²` penalty pure CORAL does not have. Measured: the same +1.0 bias shift
that moves the pure objective by 8.6e-07 moves the EMA'd one by **6.0e+03** relative.

Two further consequences, both flagged as open:

- Only **10% of the target-side gradient** reaches the model, so the two domains contribute
  asymmetrically.
- At `n_blocks=3` the encoder moves every step, so the buffer is **stale by construction** —
  a problem round 1 never had, because its encoder was fixed.

`--coral-no-target-ema` existed so this could be answered by measurement. **It has been
answered: see Change 1 above. The default is now OFF.**

---

## 4. PRE-REGISTERED PREDICTION — the feature-scale monitor

*Written into the code before either arm ran. Not to be adjusted afterwards.*

Scaling the representation by ε scales both covariances by ε², their difference by ε², and
the **squared** Frobenius objective by **ε⁴**; the mean term falls as ε². So the CORAL loss
can be driven toward zero by **shrinking the representation** rather than by aligning
anything — a degenerate solution the objective cannot distinguish from success.

Verified numerically (T2), not just asserted: at ε = 0.5 / 0.25 / 0.1 the covariance term
comes back at exactly ε⁴ and the mean term at exactly ε², to <1e-4 relative.

**What differs between the arms is the capacity available to do it.** On `nb0` only the 640
gains can rescale, and shrinking them also shrinks the logit, which the supervised BCE
resists — this arm is the **near-null**. On `nb3`, 14.8M encoder parameters can reshape the
representation with the classifier free to re-scale afterwards, and the BCE resistance is
much weaker.

> **THE PREDICTION.** A **falling** post-LayerNorm feature norm alongside a **falling** raw
> CORAL loss is the collapse signature. A falling CORAL loss with a **flat** norm is genuine
> alignment. The two are distinguishable only if the norm is logged.

Logged at every curve point, on both arms, **per domain**, in `diagnostics.csv`:
`src_feat_l2`, `src_perdim_std`, `tgt_feat_l2`, `tgt_perdim_std`, `coral_raw`, `cov_term`,
`mean_term`, plus `src_l2_over_unit_gain` and the live Sun & Saenko ratio `coral_over_sup`.
Computed inside `Fp32CoralLoss.forward` from the two tensors it already receives — zero
extra forward passes, and the domains are unambiguously labelled.

**Reading note.** `src_l2_over_unit_gain` divides by √640 = 25.298, the norm an *untrained*
LayerNorm produces. The LP→FT head does not start there: round-1 CORAL's trained gains give
a step-0 norm of **≈66**, so the column starts near 2.6. The signature to watch is a
**decline from the run's own step-0 value**, which is recorded in the first row.

---

## 5. Reused unchanged vs. owned by round 4

**Reused, method-agnostic, unmodified:** `train_skeleton.Trainer` (LP→FT init with the
provenance ‖head‖₂ log, layer-wise LR groups, `save`/`load` with DA buffers),
`ZoobotEncoder` (the unfreeze itself, grad checkpointing), `curve_hook`, `live_encode`,
`collapse_gate`, and from `run_unfreeze_freematch`: `expected_checkpoint_steps`,
`assert_checkpoint_retention`, `check_step0_gate`, `run_collapse_gate`, `evaluate_r1`,
`head_l2_norm`. `unfreeze_instrument`'s logit dump, memorization probe and live-encoder
plumbing are inherited by subclassing.

**Owned by round 4:** the constant block, the CORAL `FOLD_GATE` table, `Fp32CoralLoss`, the
feature-scale monitor, the CORAL diagnostics columns, the calibration mode — and the
precondition check.

> **A near-miss the test suite caught, worth recording.**
> `run_unfreeze_freematch.check_preconditions` closes over **that module's own**
> `PINNED_KEYS`. Called from a CORAL runner it would have checked FreeMatch's key list:
> CORAL's keys would never be compared, and every FreeMatch-only key would read `<absent>`
> on both sides and pass. **A gamma mismatch between the two arms sailed straight through
> it.** Round 4 therefore owns its check, with the key list as a parameter so the coupling
> cannot recur, and with a rule that a pinned key absent from *both* configs is a
> **mismatch** — a guard that compares nothing is not a guard. A regression test pins the
> near-miss so it stays on the record.

**Step-0 gate table**, re-scored from round 1's five checkpoints against `data/feature_cache`
(TPRs match that run's own `results.json`):

| fold | TPR@1e-3 | AUROC | ‖head‖₂ |
|---|---|---|---|
| 0 | 0.214286 | 0.947701 | 66.268356 |
| 1 | 0.173469 | 0.907398 | 67.536934 |
| 2 | 0.214286 | 0.904965 | 72.597260 |
| 3 | 0.153061 | 0.904980 | 68.220108 |
| 4 | 0.206186 | 0.902787 | 66.015175 |

Tolerances follow §7.11's measured values: **0** on nb0 (cached path, deterministic), **one
positive = 0.010204** on nb3 (live encode drifts ~1e-2 in score and TPR@1e-3 is a
threshold-crossing metric), with AUROC at **1e-3** on both — rank-based, so it catches a
wrong head that a plausible TPR would not.

**Collapse-gate anchors are CORAL's own**, measured from round 1's five curve files:
agreement **rises 0.897 → 0.986 on all five folds**. That is the *opposite* direction to
FreeMatch (0.876 → ~0.77 frozen, minima 0.760–0.796 unfrozen). A frozen marginal aligner on
a 640-d LayerNorm converges *toward* the source ranking, consistent with §7.4 finding CORAL
at baseline. An arm that instead pulls away is doing something round-1 CORAL could not.
Per §7.12 the **trace is the report** — the gate's verdict is not.

---

## 6. Gamma calibration — loss magnitudes only

`--calibrate` runs fold 0 for 500 steps at gamma = 1.0 and logs the raw classification and
raw (unweighted) CORAL loss per step; the reported gamma equalises their trailing-100-step
means.

**The hard rule is enforced in code, not merely stated.** The whole calibration runs inside
`test_label_lockout()`, which seals `tpr_at_fpr`, `auroc`, `auprc`, `metric_suite`,
`purity_at_completeness`, `completeness_at_purity`, `curve_hook.append_curve_row`,
`curve_hook.eval_checkpoint_folds` and every `test_*` cache split — patched at **every
binding inside this project's modules**, because `from metrics import tpr_at_fpr` creates an
independent binding that patching the module attribute would miss. Each raises
`TestLabelAccess` on contact; the lockout releases cleanly on exit. Twenty bindings are
sealed in practice, and seven adversarial tests confirm each one fires. `source_train`
remains loadable — the standardizer is fitted from the cached source features, which carry
no target-domain and no test information.

gamma is a **PINNED key**: `--calibrate` reports a value per arm, but the two arms must run
the *same* gamma or they differ in two things and nb3 − nb0 stops isolating unfreezing. The
runner refuses a training run without an explicit `--gamma` — it is a recorded decision, not
an inherited default.

### Results — fold 0, 500 steps at gamma = 1.0, trailing-100 window

| | `coral_nb0_control` | `coral_nb3` |
|---|---|---|
| **equalising gamma** | **2.0366** | **0.8183** |
| trailing-100 sup_loss | 0.09425 | 0.05459 |
| trailing-100 coral_raw | 0.04628 | 0.06671 |
| per-step gamma median | 1.983 | 1.061 |
| per-step gamma CV | **0.298** | **0.868** |
| per-step p10–p90 | [1.355, 2.873] | [0.323, 3.463] |
| first half → second half | 2.064 → 2.010 (**ratio 0.974**) | 1.138 → 0.629 (**ratio 0.553**) |
| cov term / mean term | 0.01296 / 0.03332 | 0.02510 / 0.04161 |
| fp16 steps discarded | 0.000 | 0.008 |
| wall time (local RTX 4050) | 12.4 min | 68.3 min (contended) |

**The nb0 calibration converged; the nb3 one did not.** nb0's gamma is flat across the tail
(2.064 → 2.010, a 2.6% drift) with CV 0.30. nb3's *halves* by 1.138 → 0.629 across the same
window, with CV 0.87 and a p10–p90 spanning an order of magnitude. **0.8183 is a snapshot of
a quantity still moving at step 500**, not a converged value, and it should not be quoted as
though it were.

**Why nb3's gamma is falling, and why that matters.** The equalising gamma is
sup_loss / coral_raw, and on nb3 both terms move the wrong way for stability: the supervised
loss falls (0.0546 vs nb0's 0.0942) while the raw CORAL loss *rises* (0.0667 vs 0.0463).
The rise is coupled to the representation growing — see below — because the objective scales
as (feature scale)⁴. So nb3's gamma is a function of a scale that is itself still drifting,
which is a structural reason to prefer the control arm's value.

**gamma = 1.0** (round 1's nominal un-tuned value) sits inside nb3's p10–p90 band but
**just below** nb0's p10 of 1.355.

**A limitation of this style of calibration, stated rather than buried.** The series were
recorded while training at gamma = 1.0, so the trajectory was already shaped by a CORAL term
of that weight. Training at a different gamma gives a different trajectory, and the
"equalising" property will not hold exactly. Sun & Saenko's rule also refers to the **end**
of training; 500 of 8000 steps is an early-training snapshot on both arms.

### The pre-registered monitor already separates the arms

| | src ‖f‖ step 0–10 | mid (240–260) | tail (400–500) | change |
|---|---|---|---|---|
| `coral_nb0_control` | 66.20 | 66.29 | 66.34 | **+0.2%** |
| `coral_nb3` | 65.60 | 75.95 | 82.34 | **+25.5%** |

Target-domain norms track the same pattern (nb0 69.56 → 69.79; nb3 68.31 → 82.44).

Two things follow, and neither is a result about the method:

1. **The near-null behaved as a near-null.** With only 640 gains available, nb0's
   representation scale is flat to 0.2% over 500 steps while its raw CORAL loss is likewise
   flat (0.0465 → 0.0463). The monitor's control condition works.
2. **nb3 moved, and it moved UP.** The pre-registered collapse signature is a *falling* norm
   alongside a *falling* CORAL loss. Over this window the norm rises 25% and the raw CORAL
   loss rises with it (0.0353 → 0.0667). **That is the opposite of the collapse signature.**
   The unfrozen encoder is growing the representation, not shrinking it.

This is a 500-step calibration at gamma = 1.0 on the development fold, not the run, so it is
evidence that the instrument discriminates the arms — not evidence about what an 8000-step
arm will do. The prediction stands as written and is not amended.

### Cost, measured

nb3 took 68.3 min for 500 steps locally, but for most of that it was contending with a
duplicate process (see §9); the uncontended log shows ~8.9 s/step contended. nb0 took
12.4 min. Extrapolating nb3 to 8000 steps plus 64 curve points is **not** a local-GPU job on
a 6 GB card. The relevant anchor is §7.11's measured 431.6 min for FreeMatch nb3 at 8000
steps + 65 curve points on a **T4**, with three encoder forwards per step; CORAL uses two,
so a T4 should be comparable or faster. Run these on Kaggle, not locally.

---

## 7. Test suite — 133 checks, 0 failures

| group | what it establishes |
|---|---|
| T1 | `Fp32CoralLoss` matches the shipped `CoralLoss` bit-exactly at fp32 (forward, gradient, components) |
| T2 | the ε⁴ / ε² scaling law the §4 prediction rests on, verified numerically |
| T3 | fp16 autocast: fp32 covariance, matches an fp32 reference, and the scale at which the unguarded path breaks |
| T4 | gradient reachability on both arms + functional shift-invariance + the `target_ema` finding |
| T5 | **known-null:** gamma = 0 reproduces a source-only run **bit-for-bit**, on both arms — head weights *and* all 120 trainable encoder tensors |
| T6 | **mock data** on the real `splits_v6_label_free_da` schema — real `LensDataset`, real encoder, real CORAL steps, target labels verified all `-1` |
| T7 | the curve instrument writes diagnostics, the feature-scale columns, the memorization row and the four-split logit dump |
| T8 | **adversarial suite** — 21 checks: leaked target label, detached adapter, wrong width, mismatched widths, 1-row batch, non-finite feature, missing gamma, four step-0 gate cases, six pinning cases, seven lockout cases, three degenerate-series cases |
| T9 | checkpoint round-trip carries the CORAL EMA buffers and the optimizer state — the round-1 omission §7.10 documents, so a round-4 run is continuable without reconstruction |

---

## 8. Open, and needing a decision before either arm runs

1. **One gamma for both arms.** nb0 gives **2.0366** (converged, CV 0.30, half-ratio 0.97);
   nb3 gives **0.8183** (not converged — halves across the tail window, CV 0.87). Since
   `coral_weight` is PINNED and must be identical on both arms, and since only one of the
   two measurements has settled, **the defensible choice is nb0's 2.0366** — it is measured
   at a representation scale that is not drifting, whereas nb3's value is a function of a
   scale that grew 25% during the measurement. gamma = 1.0 (round 1's un-tuned value)
   remains a legitimate alternative and sits inside nb3's p10–p90 but just below nb0's.
   **This is Navneet's call**, and per §7.14 it must be written down and hashed before the
   confirmatory folds exist. Re-calibrating nb3 for longer would sharpen it, at ~1 GPU-hour
   per additional 500 steps locally.
2. **`target_ema`.** Keep round 1's 0.9 (default) or turn it off. The staleness argument
   above is a reason to think it matters more at nb3 than it ever did at nb0, and it is
   currently UNVERIFIED either way.
3. **Peak memory.** Observed 5.86 / 6.14 GB on the local RTX 4050 at nb3, batch 256, grad
   checkpointing on. It fits, but with little headroom; a T4 (14.56 GiB) is comfortable.

---

## 9. Operational record — two process-handling errors, for the lab record

Neither touched the code, the tests, or any measurement, but both are worth recording
because they cost wall-clock and one of them produced a false report.

1. **An orphaned duplicate ran for ~1 hour.** The first calibration was launched with
   `nohup … &`; the wrapper returned exit 0 and the log went quiet, so it was assumed dead.
   It was not. A second launch then ran concurrently with it on the same 6 GB GPU — both
   at ~8.9 s/step instead of ~4-5, GPU utilisation 36%, and both headed for the same output
   file. Killed once identified. **Lesson: verify the process is gone before relaunching,
   and never launch a GPU job with `nohup &` from a wrapper that reports its own exit.**
2. **A liveness check that could not see the process.** A monitor used `ps -p <pid>`; Git
   Bash's `ps` sees only MSYS processes, not native Windows PIDs, so it reported the
   calibration had died while it was running at 100% GPU. Corrected to `tasklist`.
   **Lesson: on Windows, check process liveness with `tasklist` or PowerShell.**
3. **Stdout was invisible for the whole tracked run.** Piping `python … | grep -v` makes
   grep block-buffer its output, so the task log stayed at 0 bytes until exit even though
   the code calls `print(..., flush=True)`. **For the real 8000-step runs: use `python -u`
   and redirect straight to a file with no pipe**, or the run is unobservable for hours.

---

## 10. The gamma sweep — development, fold 0, nb3 only (BUILT, NOT YET RUN)

Added after the calibration. Runs on **Kaggle T4 × 2**, never locally.

**Why.** The calibration showed magnitude-matching is ill-posed on the unfrozen arm — raw
CORAL scales as (feature scale)⁴, the unfrozen encoder moves the scale, so "equalise at the
end of training" has no fixed point (nb3's gamma halved across its own tail, CV 0.868). And
at gamma = 1.0 the CORAL term is **losing**: `sup_loss` falls, raw CORAL rises, the norm
rises 25%. An 8000-step run in that state would yield a null nobody could interpret.

**The sweep.** gamma ∈ {1, 4, 16}, nb3, fold 0, **2000 steps each**, target EMA off,
everything else identical to the pinned config. Three independent processes, one per gamma.
Per-log-interval traces of, **per domain**: mean ‖f‖, mean per-dimension std, raw CORAL and
`sup_loss` — plus the full per-step series (`steps.csv`), because the slope fit and the
block bootstrap want every step, not a 16-point summary.

**Metric hygiene is enforced, not declared.** The whole run executes inside
`test_label_lockout()`. There is no `curve_hook` call, no logit dump and no memorization
probe in this path — all three read test splits. The only cache read is `source_train`, for
the standardizer. A gamma chosen here therefore *cannot* have been chosen on a test-set
quantity.

**The pre-registered selection criterion** lives in `coral_round4.SWEEP_CRITERION`, is
printed by the runner before the first optimizer step, and is evaluated **locally** by
`analyze_coral_gamma_sweep.py` on the downloaded traces (CPU work on small CSVs — it must
not consume GPU session time):

> Choose the **smallest** gamma satisfying both:
> **(i)** raw CORAL is not monotonically rising over the second half (steps 1000–2000) —
> OLS slope, moving-block bootstrap (block 50, B = 5000); **PASS iff the 95% CI's lower
> bound ≤ 0**, i.e. fail only when demonstrably rising. The stricter "demonstrably falling"
> reading is reported as `strictly_falling` but is **not** the criterion.
> **(ii)** source feature-norm growth over the full window — mean(last 10%) − mean(first
> 10%) — is measurably damped vs the gamma = 1.0 arm; **PASS iff the CI of the difference
> lies entirely below zero**. **INAPPLICABLE** (not passed) if the reference arm did not
> grow.
>
> If no gamma satisfies both: **report it, select nothing, and extend the sweep upward**
> (64, then 256). Never relax either clause.

**One statistic had to be rebuilt, and the test caught it.** `growth` is a statement about
the *order* of a series. The first implementation block-resampled the whole window, which
shuffles late blocks into the head and early blocks into the tail — destroying exactly the
structure being measured. On a clean ramp it returned a difference of −18.02 with a CI of
[−12, +13]: an interval that does not contain its own point estimate, the signature of a
statistic and a resampling scheme that disagree. The head and tail windows are now
block-resampled **separately**, since the uncertainty that matters is in estimating each
window's mean.

**Execution.** `launch_coral_sweep_kaggle.py` pins gamma=1 → `cuda:0` and gamma=4 →
`cuda:1` concurrently and starts gamma=16 on whichever frees first. Every operational rule
in §9 is encoded in it: `python -u`, stdout to a **file** never a pipe, one output directory
per gamma with a refuse-if-exists guard, liveness via `Popen.poll()` rather than `ps -p`,
checkpoints every 250 steps, and a **100-step gamma=1 verification arm that is read back**
before both GPUs are committed.

**Kaggle dataset:** `kaggle/coral_r4/` — built and verified by
`scripts/build_coral_r4_bundle.py` (0 failures). Cells: `KAGGLE_CORAL_R4_SWEEP_CELLS.md`.

**Test coverage:** `test_coral_round4_changes.py`, **60 checks, 0 failures**, on top of the
original suite's 133 (which still passes unchanged after the EMA default flip).
