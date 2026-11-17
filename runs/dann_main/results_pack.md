# Round 5 — results pack

Every number the results section needs, with its provenance. **Nothing here is newly
computed** — this is collection, so that writing does not mean re-deriving anything from
eleven sessions of logs. Assembled 2026-09-05, after round 5 closed.

Companion files: `fold_curves.json` (all 16 checkpoints × 4 folds),
`paired_analysis.json`, `../dann_r5_budget.json`, `../dann_r5_prereg.json`,
`../dann_r5_budget_packet.md`.

---

## 1. The nb3 table

| method | fold-mean TPR@1e-3 | fold sd | fold-mean AUROC | budget | head init |
|---|---:|---:|---:|---:|---|
| **DANN × nb3** | **0.4167** | 0.1525 | 0.9066 | 4000 | Baseline A phase-1 head |
| CORAL × nb3 | 0.4348 | 0.0885 | 0.9088 | 2000 | round-1 CORAL heads |
| FreeMatch × nb3 | 0.4169 | 0.0595 | 0.9657 | 1500 | its own round-1 heads |
| Baseline A (frozen, source-only) | ~0.18 pooled | — | ~0.90 | — | linear probe |

Per fold, TPR@1e-3, folds 1–4 (fold 0 is development and enters no mean):

| fold | DANN | CORAL | FreeMatch |
|---:|---:|---:|---:|
| 1 | 0.4286 | 0.3571 | 0.4184 |
| 2 | 0.6224 | 0.5612 | 0.4796 |
| 3 | 0.2653 | 0.3980 | 0.3367 |
| 4 | 0.3505 | 0.4227 | 0.4330 |
| **mean** | **0.4167** | **0.4348** | **0.4169** |

**Per-fold AUROC, all three arms at their own budgets:**

| fold | DANN (4000) | CORAL (2000) | FreeMatch (1500) | n_pos |
|---:|---:|---:|---:|---:|
| 1 | 0.9534 | 0.9158 | 0.9488 | 98 |
| 2 | 0.9809 | 0.9394 | 0.9760 | 98 |
| 3 | 0.8657 | 0.8913 | 0.9678 | 98 |
| 4 | 0.8262 | 0.8886 | 0.9701 | **97** |
| **mean** | **0.9066** | **0.9088** | **0.9657** | |
| **drop (3,4)−(1,2)** | **−0.1211** | −0.0376 | **+0.0066** | |

**The folds-3/4 AUROC drop is DANN's alone, and only at 4000.** FreeMatch's folds 3–4 are
its *best*. Read budget-matched, DANN's drop is −0.0223 at 1500 and −0.0179 at 2000 — the
latter *smaller than CORAL's* (−0.0376) at the same budget. The collapse emerges between
step 2000 and step 4000. **Folds 3 and 4 are not intrinsically hard at the ranking level**,
and any claim that they are is withdrawn (claims map G7).

DANN's per-fold TPR moves 2000 → 4000 by −0.020 / **+0.010** / **−0.102** / −0.041, and its
mean AUROC falls 0.9315 → 0.9066. That divergence — folds moving in opposite directions
with continued training — is the mechanism behind the spread finding in §6.

**CORAL source:** `kaggle/coral_r4/runs/main/extracted/coral_r4_confirm/fold{1..4}/logits/`,
recomputed here; its per-fold TPR reproduces the recorded 0.4348 fold-mean exactly, which
validates the extraction. **FreeMatch source:**
`runs/freematch_nb3_unfrozen_fold{1..4}/curve.csv` at each fold's own budget step.

**Sources.** DANN: `runs/dann_main/fold{1..4}/core_fold*.tar.gz` → per-checkpoint
logits, recomputed into `fold_curves.json`. CORAL: PROJECT.md §7.15 *THE REPORTED
RESULT — folds 1–4*, per fold at step 2000. FreeMatch: `runs/foldmean_bootstrap.json`,
`per_fold[k].unfrozen`. Baseline A: §7.4.

**Unpaired 95% CI for DANN** (t, df=3): [0.1741, 0.6593], half-width 0.2426.

---

## 2. The paired analysis — a bound

> **Any true difference between the three nb3 methods is smaller than ±0.17 in
> TPR@1e-3**, on the same folds, with fold difficulty removed.

| contrast | mean diff | paired 95% CI | unpaired 95% CI | half-width |
|---|---:|---|---|---|
| DANN − CORAL | -0.0180 | [−0.1781, +0.1421] | [−0.2986, +0.2625] | 0.160 vs 0.281 (**43% tighter**) |
| DANN − FreeMatch | -0.0002 | [−0.1656, +0.1652] | [−0.2607, +0.2602] | 0.165 vs 0.260 (**36% tighter**) |

Per-fold differences — DANN − CORAL: +0.0715  +0.0612  -0.1327  -0.0722

Per-fold differences — DANN − FreeMatch: +0.0102  +0.1429  -0.0714  -0.0825

**Why a paired *t*-interval and not a bootstrap.** At n = 4 folds a bootstrap draws from
only 4⁴ = 256 distinct resamples; its tail behaviour is an artefact of that lattice, not
of the data. The *t*-interval's assumption is the weaker one here.

**Why pairing is justified — fold difficulty is substantially shared.** Hardest →
easiest: DANN **[3, 4, 1, 2]**, CORAL **[1, 3, 4, 2]**, FreeMatch **[3, 1, 4, 2]**.
Fold 2 is easiest for all three; fold 3 is hardest or second-hardest for all three.
Spearman ρ: DANN–FreeMatch +0.8, CORAL–FreeMatch +0.8, DANN–CORAL +0.4 — **but at
n = 4, ρ can only take ±1.0, ±0.8, ±0.4, 0.0 and no p-value is meaningful. Present the
orderings, not the coefficient.**

DANN's folds 3 and 4 are weak at the **ranking** level, not just at threshold: AUROC
0.8657 and 0.8262 against 0.9534 and 0.9809 on folds 1 and 2.

> **Standing caveat.** Pairing removes fold difficulty and **nothing else**. The three
> arms sit at different budgets (1500 / 2000 / 4000) and initialise from different
> heads. Both confounds survive the pairing.

---

## 3. The read-off inversion

| read at | fold-mean | 95% CI | DANN's rank of three |
|---|---:|---|---|
| step 4000 — the declared budget | **0.4167** | [0.174, 0.659] | **3rd of 3** (below CORAL 0.4348 and FreeMatch 0.4169) |
| step 2000 — the sensitivity check | 0.4551 | [0.280, 0.630] | **1st of 3** (above both) |

**Two defensible read-off points, 125 minutes of training apart, invert the ranking of
the entire nb3 table.** Nothing distinguishes them on evidence; only the order in which
they were declared does.

The pre-declaration, verbatim:

> The headline DANN x nb3 figure is the 5-fold mean at the declared budget of 4000. The 5-fold mean at step 2000 is reported as a BUDGET-SENSITIVITY CHECK and does not replace it, WHICHEVER IS HIGHER.

Recorded **PRE-DECLARED 2026-09-02, BEFORE ANY CONFIRMATORY FOLD EXISTS. Hashed into runs/dann_r5_budget.json (sha256 abf0dae0...) and pinned in run_dann_r5.py as HEADLINE_DECLARATION.**  
Hashed into `runs/dann_r5_budget.json`, sha256 `abf0dae0e313ae5388b833e7aad86171285299774a18af14f4c44309f0231fc1`, and pinned in `run_dann_r5.py` as
`HEADLINE_DECLARATION`. The budget file's `declared_utc` is **2026-09-02T00:00:00Z**; the first confirmatory fold ran on 2026-09-03. The
ordering is auditable from those two timestamps and the hash.

**Cost of the readout: zero.** `ckpt_every` is 250 and the budget 4000, so step 2000 is
checkpoint 8 of 16 on every fold and its logits were written natively during the run.

---

## 4. The budget derivation

**B = 4000**, `derived_or_declared` = *declared under a named rule, corroborated by the curve*.

> Budget = 2 x the lambda horizon, so that half of each confirmatory run trains at fixed adversarial strength. Derived from DANN's own configuration; reads no metric.

The rule reads DANN's own configuration and **no metric**. Why it: λ = tanh(5·step/2000)
pins at ~1.0 near step 2000, so every fold has its transition at the same absolute step
and the budget decides which side of it all four report from. On fold 0:

| regime | steps | n | mean | sd |
|---|---|---:|---:|---:|
| early — λ still ramping | 250-2000 | 8 | 0.3776 | **0.1314** |
| settled — λ fixed | 2250-8000 | 24 | 0.4787 | **0.0578** |

A 2.27× difference in within-fold scatter.

**Corroboration, not derivation:** the settled regime begins ~2250, and the reading at 4000 (0.4633) is 0.27 settled-sds from the settled mean (0.4787) — unremarkable, which is the check that the rule did not land on a favourable value.

**Declined alternatives:**

- **1250** — the two 1-sd rules, which coincide here (0.5000). DECLINED for the same reason: early regime, local sd 0.164915.
- **1750** — the argmax (0.5816). DECLINED on 7.11's winner's-curse precedent, and it sits in the early regime where the GLOBAL sd used by such rules describes neither phase (local sd 0.153129).
- **2000** — CORAL-matched (CORAL x nb3's verified budget; FreeMatch x nb3's is 1500, not 2000). DECLINED: places all folds inside the lambda ramp. CORAL HAS NO RAMP and FreeMatch has none either, so neither budget faced an analogous constraint and neither could carry across. This is NOT a claim that their budgets were wrong.
- **3000** — a cheaper point still past every fold's transition. DECLINED in favour of a rule statable as a round fraction. RECORDED BECAUSE IT IS THE FALLBACK if the hours become a problem: 7.7 GPU-h against 10.2, and it clears the transition by 750 steps.
- **8000** — last checkpoint -- R1 applying, with no budget selection to defend. DECLINED ON COST: 20.5 GPU-h against 10.2, to move the fold-0 reading +0.0163 and the local scatter 0.040577 -> 0.035348.

---

## 5. Fold-0 control — SINGLE FOLD

| arm | fold-0 mean over 32 ckpts | at step 4000 | last (8000) |
|---|---:|---:|---:|
| DANN (λ live) | 0.4534 | 0.4633 | 0.4796 |
| λ = 0 control | 0.1454 | — | 0.1531 |

DANN above the control at **32 of 32 matched steps**, paired median +0.3265. The two
arms differ in **λ and nothing else** — same fold, seed, init, GPU, step grid,
bit-identical at step 0.

**The λ=0 arm ends at 0.1531, below its own frozen LP→FT init of 0.1939.** That is full
fine-tuning — 14,835,520 parameters, 99.2% of the encoder — from a linear-probe head,
evaluated out of distribution, ending worse than it started: the **feature-distortion**
phenomenon of Kumar et al. (2202.10054) that motivated adopting LP→FT (§7.1a). The
protocol choice was adopted on a citation; this connects it to a measurement in this
pipeline, on this data.

> **ONE FOLD.** Arm 1 ran on fold 0 only; folds 1–4 were **considered and declined**
> (prereg `arm1_folds_1_4_declined`), so there is **no control on the confirmatory
> folds**. Nothing in the confirmatory data speaks to whether the +0.33 gap survived.
> The limitation is that the effect size *relative to unfreezing alone* rests on a
> single fold — **not** that fold 0 was misleading. DANN's absolute level held: fold 0
> at 4000 read 0.4633 against folds 1–4's 0.4167, a 0.047 difference on a fold sd of
> 0.1525.

---

## 6. Both pre-registered predictions, and their outcomes

### `prediction_arc_suppression` — FALSIFIED, opposite direction

Predicted: TPR@1e-3 **falls** on the λ-live arm relative to λ=0, because the dominant
source-vs-target cue is the painted arc's rendering signature and gradient reversal's
cheap solution is to stop representing arcs.

Observed: TPR **rose at 32 of 32 matched steps**, paired median **+0.3265**. The encoder
did not take the cheap solution. **The arc-on-real generalisation — including the
proposed explanation of the Euclid dual/lensed AGN null — is withdrawn and must not be
carried forward.**

Hashed before any curve existed: when it was recorded, `logits/` held only `ids.npz`
and `true_init.npz`, verified on disk, and `true_init` is bit-identical between the two
arms — so the one task number in existence carried no directional information.

### `3_spread_not_mean` — HELD as declared, NOT supported budget-matched

Predicted (Belfiore et al. 2024, who ensemble 20 seeds for stability): DANN's
fold-to-fold spread exceeds CORAL's and FreeMatch's.

| reading | DANN sd | comparator | ratio |
|---|---:|---:|---:|
| at declared budgets | 0.1525 (B=4000) | CORAL 0.0885 (B=2000) | 1.72× |
| at declared budgets | 0.1525 (B=4000) | FreeMatch 0.0595 (B=1500) | 2.56× |
| **budget-matched** | 0.1102 (B=2000) | CORAL 0.0885 | **1.24×** |
| **budget-matched** | 0.0374 (B=1500) | FreeMatch 0.0595 | **0.63× — reversed** |

DANN's own spread scales steeply with its own budget: **0.0374 → 0.1102 → 0.1525** at
1500 / 2000 / 4000. **The declared-budget verdict is largely a statement about training
length, not about DANN**, and Belfiore's mechanism is not established by this data.
Both readings are recorded; neither is more correct — they answer different questions.

---

## 7. Disclosures that travel with every number

- **Different budgets are §7.14's design, not a departure from it.** Each method fixes
  its own budget from its own fold-0 curve because the methods are not interchangeable.
  Runs are otherwise identical — same encoder, LP→FT protocol, batch 256, folds, test
  pools. What differs is *only where each is read*. Equal step counts would be unequal
  treatment: at 2000 CORAL trains wholly at fixed γ and FreeMatch at fixed thresholds,
  while DANN would spend its whole run inside the λ ramp. **The λ-ramp argument applies
  to DANN only and is not a claim that the other budgets were wrong.**
- **The morphology probe is a diagnostic note, not a finding.** Its fall is consistent
  with alignment succeeding on a source-specific cue (ML-FP is 9.3% of the source pool
  and source-only, so "looks like ML-FP" is one of D's cues) and **cannot be separated
  from representation damage with this instrument**. What answers the round's question
  is the TPR curve. It measures *population separability* — a selected not-a-lens class
  against generic galaxies — not morphology and not a domain proxy (P1/P2 exchangeable
  at AUROC 0.5123; swapping the negative class to actual target moves the reading only
  0.8688 → 0.8774).
- **AdamW, against the field's SGD preference** for DANN. Adopted to match Zoobot's
  published protocol (§7.1a) and fixed across methods by §7.14; untested against SGD.
- **LayerNorm in the discriminator**, not BatchNorm — a deliberate substitution for
  tllib's BatchNorm1d, since the whole model is BatchNorm-free (asserted at startup,
  0 of 218 modules).
- **No bottleneck layer** between encoder and discriminator. Untested.
- **No nb0 DANN control** at this protocol: round-1 DANN was cached-feature, adapter,
  SGD, BatchNorm-D and λ from `total_steps`, so it is not step-comparable.
- **DANN has no within-method paired comparison** — no round-1 DANN counterpart at a
  matched protocol to difference against, unlike CORAL and FreeMatch.
- **Zero tuning budget for DANN**, against CORAL's γ sweep. λ's horizon and the
  discriminator's topology were declared, never searched.
- **98 test positives on folds 0-3, 97 on fold 4.** TPR granularity is **1/98 = 0.0102 or 1/97 = 0.0103 depending on the fold** -- it is NOT uniform. Fold 4's 0.350515 is 34/97, not x/98.
- **Per-fold models**: pooling is invalid (§7.2); every mean here is a fold-mean.

---

## 8. Provenance — the pre-registration hash sequence

Each hash is a self-certifying `runs/dann_r5_prereg.json`; what changed at each is what
makes the record of *what was fixed before what* reconstructible.

| # | sha256 (first 16) | what changed |
|---:|---|---|
| 1 | `2c363e29295ada6c` | the original pre-registration: arms, abort conditions 1–4, the M-declaration rule, the two morphology instruments |
| 2 | `93caa2b2882fde0f` | condition 3 re-specified onto the frozen probe after arm 1; conditions 3/4 ceilings as fractions of measured headroom |
| 3 | `bbdf407ba6e34bed` | arm-1 completion record and the admissible margin bands |
| 4 | `9212cdb4f4c0c3e3` | arm-2 completion; the two-of-four-pathologies finding |
| 5 | `e28adf000ee127ab` | **`prediction_arc_suppression` hashed — before any TPR curve existed** |
| 6 | `18e0f1bc3c5a7a0f` | conditions 3 and 4 made ADVISORY on every arm (v10) |
| 7 | `7def3680d6342b8a` | the re-score wiring check |
| 8 | `9d525f30fa87a603` | budget declaration of 2000 WITHDRAWN; decision returned to the run owner |
| 9 | `465e51872dbde17c` | budget DECIDED at 4000 under the named rule |
| 10 | `51351308196ad332` | **the step-2000 sensitivity readout pre-declared — before any confirmatory fold** |
| 11 | `3ad1fc361a150b95` | round 5 closed: folds 1–4, the paired analysis, both predictions resolved |

Current: `3ad1fc361a150b95567fdfc1abd3a5145e3cf06ef22d857ef32bbfbb68dcabca`, 41 top-level keys.

Also hashed: `runs/dann_r5_budget.json` (`abf0dae0e313ae53`), pinned in `run_dann_r5.py` as
`CONFIRM_BUDGET_SHA256` and re-checked by every confirmatory fold before it starts.

---

## 9. Corrections made during the round

*A project that reports its own corrections is making a claim about the numbers it did
not correct.* These belong in methods or an appendix.

1. **The DANN/importance-weighting misattribution, and its over-correction.** The design
   record attributed a historical morphology collapse to plain DANN; PROJECT.md
   attributed it to importance weighting. Four instances corrected. Then the correction
   itself over-reached — a later brief claimed the 0.61→0.33 numbers were "invented in
   planning"; they were not, and are in `PROJECT_STATE v2` §6/§8-F4.
2. **The morphology instrument was the wrong one.** Condition 3 was built on the
   trained-head column. Arm 1 showed it falls to chance **at λ = 0** — ML-FP and
   P1-reserve are both class 0 in the lens task, so a head fine-tuned to give both low
   lens logits drives them together. The column was the problem, not the criterion: the
   pre-registered rule correctly returned UNCALIBRABLE. Condition 3 moved to the frozen
   probe.
3. **Condition 4's threshold was ported across feature spaces.** "Participation ratio
   below 20 of 640" came from a SimSiam finding in a different space; on arm 1 it fired
   **7,999 of 8,000 steps** and the run's maximum PR was 13.40 — the threshold was never
   reachable. Re-specified as a differential against the λ=0 control, src and tgt
   separately.
4. **The logit collection defect.** Both fold-0 arms packaged cleanly, verified every
   checksum, and arrived with **all 64 per-checkpoint logit files missing**: the
   archiver listed subdirectories one level deep and the runner writes them two levels
   down. Fixed twice over — the walk is recursive, and packaging now **refuses** any run
   whose `checkpoints.csv` names an artifact the archive lacks. The curves were rebuilt
   by re-scoring, with step-0 gates passing at `max|diff| = 0.000e+00` over 26,497
   logits.
5. **A stability claim that contradicted its own table.** "DANN is distinctively
   unstable" was written while the table showed arm 1 (**no adversary**) as the least
   stable of three. Rewritten around the ordering; and FreeMatch's sd, previously
   reported as "not measurable", was measurable all along — the earlier check read its
   *confirmatory* curve, which stops at 1500 because 1500 is its budget.

---

## 10. Verification status

- All four confirmatory folds: 4000/4000 steps, 16/16 checkpoints, no resumes, not
  aborted, **no ARMING detector fired**, pinned budget hash present in every
  `run_config.json`.
- **Every fold passed its own step-0 gate at |diff| exactly 0.0 against four DISTINCT
  values** — 0.142857 / 0.193878 / 0.153061 / 0.195876. The phase-1 head is
  fold-independent, so this gate is the only artifact proving fold identity.
- `abort3b` fired **advisory** on all four folds and stopped nothing. Under the
  pre-registered arrangement it would have aborted every one at ~step 450. **v10's
  decision to make conditions 3 and 4 advisory — taken before any confirmatory fold ran
  — is what preserved these runs.**
- **Schedule audit**: 11 components checked; every one flat or absolute-step pinned, so
  step 2000 of a 4000-step run is the model a 2000-step run would produce.
- Local test suite: 229 checks, all passing.
- **Companion files:** `claims_map.md` (every claim with a strength label) and `paper_skeleton.md` (sections, claims carried, figure/table inventory).
