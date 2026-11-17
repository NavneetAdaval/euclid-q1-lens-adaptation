# Claims map — what the paper may assert, and at what strength

One row per claim. **Every row carries a strength label**, so the writing cannot
accidentally promote a hypothesis to a finding — the failure this project has corrected
twice (the DANN/IW misattribution, and "DANN is distinctively unstable" written against a
table that said otherwise).

**Strength vocabulary**

| label | means |
|---|---|
| **established** | multi-fold, controlled, survives the caveats in its own row |
| **single-fold observation** | one fold. Reportable, never a finding. |
| **hypothesis** | consistent with the data; the test that would settle it does not exist |
| **withdrawn** | was asserted, is falsified, must not reappear |

---

## A. The headline result

| # | claim | strength | evidence | file | caveats that must travel |
|---|---|---|---|---|---|
| A1 | The three unfrozen DA methods land within **0.018** of each other on fold-mean TPR@1e-3 (DANN 0.4167, CORAL 0.4348, FreeMatch 0.4169) | **established** | 4 folds each, same folds, same encoder/protocol/test pools | `results_pack.md` §1, `paired_analysis.json` | different budgets (1500/2000/4000) and different head inits; neither is removed by anything below |
| A2 | Any true difference between them is **smaller than ±0.17** in TPR@1e-3, fold difficulty removed | **established** | paired *t*-interval, df=3, on the same 4 folds | `results_pack.md` §2 | paired removes fold difficulty **only**; the bound is wide because n=4 |
| A3 | Pairing is justified: fold difficulty is substantially shared | **established** | fold 2 easiest for all three; fold 3 hardest or near-hardest for all three | `results_pack.md` §2 | at n=4, ρ takes only ±1.0/±0.8/±0.4/0 — present orderings, never a p-value |
| A4 | All three unfrozen arms roughly **double** frozen Baseline A (~0.18 pooled) | **established** | §7.4 | `results_pack.md` §1 | Baseline A is pooled, the others are fold-means; not a like-for-like interval |

## B. The methodological findings — the strongest material in the round

| # | claim | strength | evidence | file | caveats |
|---|---|---|---|---|---|
| B1 | **The read-off point inverts the ranking.** At step 2000 DANN is 1st of three (0.4551); at its declared 4000 it is 3rd (0.4167). Two defensible points, 125 min apart | **established** | same four folds, both readings from natively-written checkpoints | `results_pack.md` §3 | — |
| B2 | The pre-declaration is what makes B1 reportable rather than a choice | **established** | hashed `51351308…`, `declared_utc` 2026-09-02; first fold ran 2026-09-03 | `results_pack.md` §3, §8 | the ordering is auditable from the hash and timestamps, not from testimony |
| B3 | Reading a mid-run checkpoint is a valid budget-matched comparison here | **established** | 11-component schedule audit: no LR scheduler, `total_steps` unread on the DANN path, λ absolute, `lr_decay` layer-wise, no EMA/clip, sampler independent of run length | PROJECT §7.17 audit table | holds **for DANN**; not asserted for methods with μ or threshold schedules |
| B4 | Different budgets are §7.14's **design**, not a departure | **established** | each method fixes its own budget from its own fold-0 curve; runs otherwise identical | `results_pack.md` §7 | equal step counts would be *unequal* treatment; the λ-ramp argument applies to DANN only |

## C. DANN's behaviour

| # | claim | strength | evidence | file | caveats |
|---|---|---|---|---|---|
| C1 | DANN reaches **domain confusion**: `d_acc` at chance (median 0.531), domain loss 0.691 vs ln 2 = 0.693 | **established** (fold 0, but a training-dynamics fact, not a test metric) | arm 2 `steps.csv` | — |
| C2 | **DANN degrades on folds 3–4 with continued training while improving on 1–2.** AUROC drop (3,4)−(1,2) is −0.121 at 4000, −0.018 at 2000; FreeMatch's is **+0.007**, CORAL's −0.038 | **established** | all three arms' per-fold AUROC, plus DANN read budget-matched | `results_pack.md` §1, PROJECT §7.17 | the collapse emerges **between 2000 and 4000**; it is not a constant property of DANN, and folds 3–4 are **not** intrinsically hard at the ranking level |
| C3 | That divergence is the mechanism behind DANN's fold spread (sd 0.0374 → 0.1102 → 0.1525 across 1500/2000/4000) | **established** | same per-fold curves | `results_pack.md` §6 | descriptive, not causal — no intervention test |
| C4 | DANN's spread exceeds CORAL's and FreeMatch's **at the declared budgets** (1.72×, 2.56×) | **established** | fold sds at each arm's own budget | `results_pack.md` §6 | **and reverses budget-matched**: 1.24× vs CORAL at 2000, **0.63× vs FreeMatch at 1500**. Both readings are the result |
| C5 | The pre-registered spread prediction is **HELD as declared, NOT supported budget-matched** | **established, both readings** | prereg `3_spread_not_mean` | `results_pack.md` §6 | Belfiore's mechanism is **not** established by this data; the declared-budget verdict is largely about training length |
| C6 | The fold-0 development run **predicted** the confirmatory level (0.4633 vs 0.4167, on a fold sd of 0.1525) | **established** | §7.14's protocol | `results_pack.md` §5 | — |

## D. Single-fold observations — reportable, never findings

| # | claim | strength | evidence | file | caveats |
|---|---|---|---|---|---|
| D1 | **Unfreezing alone underperforms its own initialisation**: λ=0 ends at 0.1531 against the frozen LP→FT head's 0.1939 | **single-fold observation** | arm 1, fold 0, 32 checkpoints | `results_pack.md` §5 | fold 0 only. Folds 1–4 **considered and declined** (prereg `arm1_folds_1_4_declined`) with reasoning |
| D2 | That reproduces Kumar et al.'s **feature distortion** — 99.2% of the encoder fine-tuned from a linear probe, evaluated OOD, ending below its start | **single-fold observation** | same | `results_pack.md` §5 | connects a protocol choice (LP→FT, §7.1a) to a measurement in this pipeline rather than a citation |
| D3 | DANN beats its λ=0 control at **32 of 32** matched steps, paired median +0.3265 | **single-fold observation** | arms differ in λ alone — same fold, seed, init, GPU, bit-identical at step 0 | `results_pack.md` §5 | **there is no control on folds 1–4**, so nothing in the confirmatory data speaks to whether this gap survived. Do **not** write "it did not survive" — that conflates two quantities |
| D4 | The frozen probe falls under DANN (0.917 → 0.788 median) and **not** under λ=0 (holds ~0.908) | **single-fold, DIAGNOSTIC NOTE ONLY** | arm 1 vs arm 2, step-matched | PROJECT §7.17 | **cannot separate "alignment succeeding on a source-specific cue" from "representation damage"** — ML-FP is 9.3% of the source pool and source-only, so "looks like ML-FP" is one of D's cues. The instrument does not resolve it, and the TPR curve is what answers the question |
| D5 | The probe measures **population separability**, not morphology and not a domain proxy | **established** (a fact about the instrument, measured) | P1/P2 exchangeable at AUROC 0.5123; swapping the negative class to actual target moves the reading only 0.8688 → 0.8774 | PROJECT §7.17 | corrects two earlier descriptions of the same column |

## E. Already on the record from earlier rounds

| # | claim | strength | evidence | caveats |
|---|---|---|---|---|
| E1 | Round-1 tier placement was substantially an **adapter artifact** | **established** (§7.15) | — | as recorded; not re-litigated here |
| E2 | The two-tier structure collapses under unfreezing | **established** (§7.4 amendment) | DANN now makes it three for three, with the most mechanistically different method | budgets and head inits differ across the three |

## F. Hypotheses — the test does not exist

| # | claim | strength | why it stays a hypothesis |
|---|---|---|---|
| F1 | A low-rank task-specific subspace under unfreezing explains why methods converge | **hypothesis** | participation ratio was **never logged** for the CORAL or FreeMatch nb3 runs, so the comparison that would test it does not exist. Their checkpoints are retained, so it is recoverable offline |
| F2 | CORAL × nb3's 0.4348 was its **DA term**, not the extra capacity | **hypothesis** | needs a CORAL λ=0 control. Arm 1 says source-only nb3 fine-tuning ends below its own frozen init, but that is DANN's control on one fold |
| F3 | The historical v5 morphology collapse shares the probe's population-separability explanation | **hypothesis** | needs the same pool audit on that round. **Do not rewrite §7.4 on one fold's inference** |

## G. Withdrawn — must not reappear

| # | claim | why withdrawn |
|---|---|---|
| G1 | **Arc suppression is DANN's mechanism**: the encoder stops representing arcs to hide the sim's rendering signature | **FALSIFIED, decisively and in the opposite direction.** TPR rose at 32 of 32 matched steps, median +0.3265. Hashed as `prediction_arc_suppression` before any curve existed (`e28adf00…`) |
| G2 | Adversarial DA has a structural problem with **arc-on-real** simulations; this would explain the Euclid dual/lensed AGN null | **NOT SUPPORTED.** Recorded as a hypothesis riding on G1; G1 is falsified, so this goes with it |
| G3 | "DANN is distinctively unstable" | contradicted its own table — arm 1, **with no adversary**, was the least stable of three. Replaced by the ordering and the budget-matched reading (C4) |
| G4 | "Fold 0's +0.33 did not survive" | conflates DANN-minus-control with DANN's level. There is no control on folds 1–4. DANN's absolute level *held* (C6) |
| G5 | CORAL and FreeMatch were both confirmed at 2000 | FreeMatch × nb3 was confirmed at **1500**, verified from its five `run_config.json` files |
| G6 | FreeMatch × nb3 has no comparable 8000-step curve | it does — that check read its *confirmatory* curve, which stops at 1500 because 1500 is its budget. Its development curve runs 0→8000 |
| G7 | Folds 3–4 are hard at the **ranking** level | FreeMatch's folds 3–4 are its **best** (AUROC 0.968, 0.970). The drop is DANN's, and only at 4000 (C2) |

---

## Disclosures that attach to every row

Not claims; conditions on all of the above.

- **98 test positives on folds 0–3, 97 on fold 4.** TPR granularity is 1/98 = 0.0102 or
  1/97 = 0.0103 — **not uniform**. §7.16 is a section about granularity and must not state
  it as though it were.
- Per-fold models throughout; **pooling is invalid** (§7.2). Every mean is a fold-mean.
- **AdamW**, against the field's SGD preference for DANN — Zoobot's published protocol,
  fixed across methods by §7.14, untested against SGD.
- **LayerNorm in the discriminator**, not BatchNorm — deliberate; the model is
  BatchNorm-free, asserted at startup (0 of 218 modules).
- **No bottleneck** between encoder and discriminator. Untested.
- **No nb0 DANN control at this protocol** — round 1 was cached-feature, adapter, SGD,
  BatchNorm-D, λ from `total_steps`. Not step-comparable.
- **DANN has no within-method paired comparison**, unlike CORAL and FreeMatch.
- **Zero tuning budget for DANN** against CORAL's γ sweep. λ's horizon and D's topology
  were declared, never searched.
- Conditions 3 and 4 were **advisory** on the confirmatory folds; `abort3b` fired on all
  four and stopped nothing. Under the pre-registered arrangement every fold would have
  aborted at ~step 450.
