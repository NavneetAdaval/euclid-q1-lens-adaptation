# Fold-0 checkpoint-selection analysis — report

**Script:** `scripts/analyze_fold0_selection.py` (standalone; `--self-test` for mock/adversarial checks).
**Raw output:** `runs/fold0_selection_analysis_stdout.txt`.
**Scope:** fold 0 only, one disclosed use of fold-0 test labels, then frozen for folds 1–4.
**No new training.** Reads saved `curve.csv` + `checkpoints/` + frozen-feature cache only. Every metric via `metrics.py`; scoring via `evaluate.load_checkpoint`/`score_features`. Tags: `[RB]` research-backed, `[EC]` engineering-choice, `[UNVERIFIED]`.

---

## Discovery (what exists on disk)

Three fold-0 curves exist locally — **Baseline A, DANN, FreeMatch**. No AdaMatch/CORAL curve. Drive not separately mounted; these local files are the genuine runs.

| method | rows | step range | spacing | ≤8000 | in window 6000–8000 |
|---|---|---|---|---|---|
| Baseline A | 50 | 500 → 25000 | 500 | 16 | 5 |
| DANN | 75 | 200 → 15000 | 200 | 40 | 11 |
| FreeMatch | 18 | 300 → **5400** | 300 | 18 | **0** |

- **`fold0_tpr_1e-3`** is TPR@FPR=1e-3 on the fold-0 test set (98 real lenses vs shared 26 399 negatives). Source-val cols: `source_risk_auroc/tpr` (sim val).
- Every curve step has a matching `step_*.pt` (full coverage for Task C).
- **FreeMatch stops at step 5400** — never reaches the locked window. Handled descriptive-only, verdict pending its 8k run. Window is **not** rescaled to fit it.

---

## Task A — flatness (window 6000–8000, budget 8000)

| method | window n | mean | range | OLS slope (TPR/1000) | pred. Δ over window | bootstrap CI half-width | range / CI-half | **verdict** |
|---|---|---|---|---|---|---|---|---|
| Baseline A | 5 | 0.206 | 0.051 | −0.018 | −0.037 | 0.092 | 0.56 | **FLAT-TIGHT** |
| DANN | 11 | 0.197 | 0.031 | −0.002 | −0.005 | 0.087 | 0.35 | **FLAT-TIGHT** |
| FreeMatch | 0 | — (0.307 over 0–5400) | 0.296 over 0–5400 | — | — | — | — | **PENDING-8K** |

- Both in-budget methods are **flat inside the window**: predicted window change (−0.037, −0.005) is smaller than the ~98-positive bootstrap CI half-width (~0.09), and the checkpoint-to-checkpoint range is ≤ CI half-width. The R1/R2/R3 flatness assumption holds. `[RB]` noise scale = `metrics.paired_bootstrap`, class-conditional.
- **Judgment nuance:** Baseline's window slope is mildly negative (−0.018/1000 → −0.037 over the window). It is *within* the CI half-width, so **FLAT-TIGHT**, not DEGRADING — but it is the least-flat of the two. Navneet's call; the numbers are shown, not thresholded. `[EC]`
- **FreeMatch:** its available range is steeply rising (0.09 → 0.39), i.e. clearly **not yet converged at 5400** — consistent with a method that needs its full 8k run before any flatness verdict. `[EC]`

## Task B — selection rules (fold-0 TPR each blind rule would pick)

**Baseline A** — CI half-width 0.092:

| rule | step | TPR | gap→oracle |
|---|---|---|---|
| R1 last | 8000 | 0.194 | −0.041 |
| R2 fixed-plateau | 6000 | 0.235 | 0.000 |
| R3 median | 7500 | 0.204 | −0.031 |
| ORACLE | 6000 | 0.235 | — |

Pairwise: R1−R3 = −0.010, R1−R2 = −0.041, R2−R3 = +0.031 — **all within CI half-width → indistinguishable.**

**DANN** — CI half-width 0.087:

| rule | step | TPR | gap→oracle |
|---|---|---|---|
| R1 last | 8000 | 0.204 | −0.010 |
| R2 fixed-plateau | 6000 | 0.204 | −0.010 |
| R3 median | 7800 | 0.194 | −0.020 |
| ORACLE | 6800 | 0.214 | — |

Pairwise: R1−R3 = +0.010, R1−R2 = 0.000, R2−R3 = +0.010 — **all within CI half-width → indistinguishable.**

**FreeMatch:** skipped — no in-window checkpoints (per protocol, no rule comparison until its 8k run).

> **Guardrail (printed alongside):** the oracle is the best-test checkpoint on ~98 positives, so it overfits test noise and every gap-to-oracle *overstates* what a real rule loses. **Rule-vs-rule is the comparison that matters.** On both in-budget methods, the three rules are indistinguishable within fold noise → **R1 (last checkpoint) wins on simplicity.** `[RB]` SPEC §3 / PROJECT §7.4.

## Task C — source-val purity@50%-completeness (label-free selector)

| method | n | source-val purity range | Spearman(sv-purity, fold0 TPR) | real-domain purity range (ref) | **verdict** |
|---|---|---|---|---|---|
| Baseline A | 50 | 0.9990–1.0000 (0.001) | +0.13 | 0.009–0.020 | **DEAD** |
| FreeMatch | 18 | 0.9980–0.9990 (0.001) | −0.14 | 0.008–0.111 | **DEAD** |
| DANN | — | — | — | — | skipped (dann_adapter head not reconstructable) |

- **Confirmed:** the *label-free* source-val purity is **pinned at ~1.0** (range 0.001 on both methods) while the *real-domain* purity actually varies (up to range 0.11 on FreeMatch). Source-val purity does **not** track real-test TPR (|Spearman| ≤ 0.14). This is the saturation Navneet asked to test for: on the easy sim domain the 50%-completeness point sits at near-zero FPR, so purity pins at 1.0. **Drop source-val purity as a selector.** `[RB]` — same mechanism as the dead source-val AUROC (§7.6).
- **Important scope correction:** the existing `purity_50comp` column in `curve_with_signals.csv` is **real-domain** purity (`test_pos_fold0` + shared negs, corrected to 1:2200) — *not* the label-free quantity. It is the non-label-free *reference* here, not a usable selector (it needs real labels).
- **DANN:** its `dann_adapter` head can't be scored through the generic linear/mlp loader, so Task C is not computable for it here — reported, not fabricated. (Baseline+FreeMatch already settle the source-val-purity question.)

---

## Output 3 — recommended checkpoint spacing for the 5-fold run

Current ~300–500-step spacing gives only ~5 in-window checkpoints — enough for a flatness verdict, **not** for a 20-point median.
- **R1/R2 chosen:** current spacing already sufficient, no change. `[EC]`
- **R3 chosen:** log every ~100 steps in 6000–8000 (~21 pts) or every ~250 steps over the whole 0–8000 budget (~32 pts, ~8 in-window). `[EC]`
- **Cost:** curve-eval is feature-space CPU (val + fold-0 pos + shared negs). ~16 → ~32 evals/fold ≈ doubles curve overhead = minutes, negligible vs training. **Recommend ~250-step spacing over the full budget.** `[UNVERIFIED]` exact minutes not benchmarked.

## Output 4 — paragraph for PROJECT.md §7.6

This analysis tests, on fold 0 only (a single disclosed use of fold-0 test labels, then frozen for folds 1–4), whether the blind checkpoint-selection rules R1/R2/R3 are valid — i.e. whether the TPR@1e-3 curve is flat inside the locked 6000–8000 window. On the two in-budget fold-0 curves (Baseline A, DANN) the curve is **FLAT-TIGHT**: the predicted window trend (−0.037, −0.005 TPR) and the checkpoint-to-checkpoint range (0.051, 0.031) both sit at or below the ~98-positive bootstrap CI half-width (~0.09), and all three rules (R1 last / R2 fixed-plateau / R3 median) pick fold-0 TPRs that differ by less than that half-width — i.e. they are **indistinguishable within fold noise**, so **R1 (last checkpoint) is chosen on simplicity**. Label-free **source-val purity@50%-completeness is a dead selector** (pinned at ~1.0, range 0.001, does not track real-test TPR), confirming the §7.6 saturation hypothesis; the varying `purity_50comp` on disk is the *real-domain* (non-label-free) quantity and is not usable for selection. **FreeMatch's fold-0 curve ends at step 5400 and never reaches the window, so its flatness verdict is pending its full 8000-step run in the 5-fold bake-off.** This analysis establishes nothing about folds 1–4 (untouched), tunes no hyperparameter, and does not license re-using fold-0 labels.

## Bottom line

- **Rule to adopt: R1 (last checkpoint at 8000).** All rules are within fold noise on both in-budget methods; R1 is simplest and defensible. `[RB]`
- **Source-val purity: drop it** as a selector (dead/saturated). `[RB]`
- **Before locking the rule for the bake-off:** the flatness verdict is only confirmed for Baseline A + DANN. **FreeMatch (a self-training contender) must be re-run to 8000 steps** to confirm it also plateaus in-window — its 0–5400 curve is still steeply rising. `[EC]`
