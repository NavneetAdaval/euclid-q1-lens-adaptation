# Paper skeleton — headings, claims, figures. **No prose.**

Target: **A&A**. The paper answers Lines et al. §4.4, and the audience is lens-finding, so
astronomy framing leads and the ML is method rather than contribution. Claim IDs refer to
`claims_map.md`; **nothing may be written that is not a row there**, and every row's
strength label governs the verb used.

An **ML-workshop variant** is noted per section where it would differ: shorter, methodology
forward, astronomy background compressed to a paragraph.

---

## 1. Introduction

**Carries:** the problem, not results.

- Strong lensing in Euclid Q1; why sim-to-real is the binding constraint
- The label-free framing: no target labels, so no target-supervised selection is available
- Lines et al. §4.4 as the open question this answers
- What the paper contributes: a **controlled, pre-registered comparison** of three DA
  families under an identical protocol, and a **bound** (A2) rather than a winner

*Workshop variant: compress to one paragraph; lead instead with the pre-registration and
read-off-inversion material, which is the transferable content.*

---

## 2. Data

- Euclid Q1; the 489 real lenses; the 5-fold rotation
- **Source domain**: sim arcs painted on real cutouts (arc-on-real, HOLISMOKES XI standard).
  Composition matters and is load-bearing later: 50.0% sim-lens, 9.3% ML-FP, 6.0% bare
  deflector, 34.7% random P1
- **Target**: real Euclid, per-fold, unlabeled
- **Test pools**: 98 positives on folds 0–3, **97 on fold 4** — state here, once, because
  §6 depends on it
- Partition scheme P0–P4 and its disjointness guarantees

---

## 3. Method

- Zoobot ConvNeXt-Nano; **LP→FT** and why (§7.1a), with D2 as the in-pipeline evidence
- `n_blocks=3`: 14,835,520 trainable encoder parameters, 99.2% of the encoder
- The three DA families: CORAL (moment matching), FreeMatch (self-training), DANN
  (adversarial) — one paragraph each, mechanism only
- **Protocol fixed across methods** (§7.14): encoder, init protocol, batch 256, folds,
  augmentation, test pools. **What differs is only where each is read** — B4
- DANN specifics: GRL, discriminator topology 640→1024→1024→1, **LayerNorm not
  BatchNorm**, λ = tanh(5·step/H) with **H = 2000 declared absolute, never `total_steps`**

*Workshop variant: this becomes §2 and expands; the λ-horizon and the fixed-protocol
argument are the parts an ML audience will actually test.*

---

## 4. Experimental design and pre-registration

**This section is the methodological contribution. It should not be an appendix.**

- Development/confirmatory split: fold 0 develops, folds 1–4 confirm, **no fold-0 number
  enters a headline** (§7.14)
- Per-method budgets from per-method fold-0 curves — B4. The three sit at 1500 / 2000 /
  4000 and **that is the design**
- **DANN's budget rule**, stated verbatim: 2 × the λ horizon, so half of each confirmatory
  run trains at fixed adversarial strength. **Reads no metric.** Corroborated, not derived
- The **pre-declared sensitivity readout** and its ordering — B2
- Abort conditions, and the decision to make conditions 3–4 **advisory before any
  confirmatory fold ran**. `abort3b` fired on all four and stopped nothing; under the
  pre-registered arrangement every fold would have aborted at ~step 450
- The per-fold step-0 gate as the **only** proof of fold identity, given a fold-independent
  head — four distinct values, |diff| exactly 0.0

**Table 1** — the pre-registration timeline: hash, date, what was fixed, what had not yet
run. Source: `results_pack.md` §8 (11 rows). The two load-bearing rows are `e28adf00…`
(arc-suppression prediction, before any curve) and `51351308…` (sensitivity readout, before
any confirmatory fold).

---

## 5. Results

**Carries:** A1, A2, A3, A4, C1, C6.

- **Table 2 — the nb3 table.** Three methods × 4 folds: TPR@1e-3 and AUROC per fold, plus
  fold-mean, fold sd, budget, head init. Frozen Baseline A as the reference row.
  Source: `results_pack.md` §1. *Every cell exists; nothing is a placeholder.*
- **Figure 1 — paired differences.** DANN−CORAL and DANN−FreeMatch: per-fold points, with
  the paired interval and the unpaired interval drawn alongside so the 43% / 36% narrowing
  is visible. Source: `paired_analysis.json`. Caption states the bound (A2) and that
  pairing removes fold difficulty **only**
- The bound stated as a **quantitative claim**, not as a null result (A2)
- Fold-difficulty orderings as the justification for pairing (A3) — orderings, **never** ρ
  with a p-value at n = 4
- Domain confusion reached (C1); the development fold predicted the confirmatory level (C6)

---

## 6. The read-off problem

**Carries:** B1, B2, B3, C2, C3, C4, C5. *This is the section a referee will remember.*

- **Table 3 — the inversion.** The same four folds read at step 2000 and step 4000:
  per-fold and mean, with DANN's rank of three at each. 1st at 2000, 3rd at 4000
- Why both points are defensible and nothing on the evidence separates them — only the
  declared order does (B2)
- **The schedule audit** (B3) as what licenses reading a mid-run checkpoint at all: 11
  components, all flat or absolute-step pinned. One table or a compact paragraph
- **C2** — the AUROC evidence that DANN degrades on folds 3–4 with continued training while
  improving on 1–2, and that FreeMatch shows **no** such drop (+0.007). Explicitly: folds
  3–4 are **not** intrinsically hard at the ranking level
- **C3/C4/C5** — the spread finding, both readings, and the pre-registered prediction
  resolving as *held-as-declared, not-supported-budget-matched*
- Granularity: 1/98 or 1/97 **by fold**, not uniform

**Figure 2 — the fold-0 development curves.** Both arms (λ live, λ = 0), 32 checkpoints,
λ overlaid on a second axis. This one figure carries the two-regime structure, the budget
derivation, and D3's control comparison simultaneously. Source: `dann_r5_budget_packet.md`
§1 and `dann_r5_curves/`.

---

## 7. What the control shows, and what it cannot

**Carries:** D1, D2, D3 — all **single-fold**. The verbs must say so.

- λ = 0 ends **below** its own frozen initialisation (D1); the LP→FT connection (D2)
- DANN above its control at 32/32 matched steps (D3)
- **Explicit:** there is no control on folds 1–4; it was considered and declined, with
  reasoning, and the effect size *relative to unfreezing alone* therefore rests on one fold
- **Do not write** that the fold-0 gap "did not survive" (G4)

---

## 8. Diagnostics that did not resolve

**Carries:** D4, D5. Framed as instrument limits, not as findings.

- What the frozen probe measures — **population separability**, measured (D5)
- Its fall under DANN and not under λ = 0 (D4), and why the instrument **cannot** separate
  alignment-succeeding-on-a-source-cue from representation damage
- Why the TPR curve is what answers the question instead
- F1 as an explicitly labelled hypothesis: PR was never logged for CORAL or FreeMatch nb3

---

## 9. Limitations

Every disclosure from `claims_map.md`, in one place: different budgets and head inits;
AdamW vs SGD; LayerNorm in D; no bottleneck; no nb0 DANN control at this protocol; no
within-method paired comparison for DANN; zero tuning budget against CORAL's γ sweep;
n = 4 folds; per-fold models so pooling is invalid.

---

## 10. Conclusions

- The bound (A2), not a winner
- The methodological result: **where you read matters as much as what you run** — B1 with
  its own numbers
- F2 as the named next experiment (a CORAL λ = 0 control)

---

## Appendix A — corrections made during the round

The five from `results_pack.md` §9. *A project that reports its own corrections is making a
claim about the numbers it did not correct.* In A&A this is an appendix; **in the workshop
variant it belongs in the main body.**

## Appendix B — pre-registration hashes

`results_pack.md` §8 in full, with what changed at each of the 11 emissions.

---

## Figure/table inventory

| # | what | source | carries |
|---|---|---|---|
| T1 | pre-registration timeline, 11 hashes | `results_pack.md` §8 | B2 |
| T2 | the nb3 table — 3 methods × 4 folds, TPR + AUROC, budgets, head inits | `results_pack.md` §1 | A1, A4, C2 |
| T3 | the read-off inversion — same folds at 2000 and 4000, with ranks | `results_pack.md` §3 | B1 |
| F1 | paired differences, paired vs unpaired intervals | `paired_analysis.json` | A2, A3 |
| F2 | fold-0 development curves, both arms, λ overlaid | `dann_r5_budget_packet.md` §1 | B4, C3, D1, D3 |

**All five exist in data today.** Nothing in this skeleton requires a new run.
