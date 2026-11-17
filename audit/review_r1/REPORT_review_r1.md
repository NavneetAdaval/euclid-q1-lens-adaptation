# Review round 1 — analysis report (stored scores only, no paper edits)

**Status: post hoc and exploratory.** Every number below is stored in a JSON (or the CSV) under
`audit/review_r1/out/`, each carrying `"status": "post_hoc_exploratory"`, the producing script, the
sha256 of every input file, the test pool and the folds. A pointer `out/X.json → key` follows every
table. Default pool **D1**; fold 0 is development.

Scripts (all in `audit/review_r1/scripts/`, run with `python -B`):

| script | produces |
|---|---|
| `rr_common.py` | shared loaders (imports `paper/scripts/_common.py` and `scripts/metrics.py`; nothing re-implemented) |
| `r_reproduction.py` | `out/R_reproduction_table2.json` |
| `f_facts.py` | `out/F1_…` to `out/F7_…` |
| `a1_resolution.py` | `out/A1_availability.json`, `out/A1_series.json`, `out/A1_resolution.json`, `fig/fig5_detrended_draft.png` |
| `a2_top_negatives.py` | `out/A2_top_negatives.json`, `out/top_negatives.csv`, `out/A2_crossmatch.json`, `out/A2_magnitude_bias.json`, `fig/top_neg_page_01..06.png` |
| `a3_breakdowns.py` | `out/A3_recall_by_grade.json`, `out/A3_recall_by_property.json`, `out/A3_lens_columns.json`, `out/A3_galleries.json`, `fig/lens_*.png` |
| `snapshot.py`, `snapdiff.py`, `ctrl_scan.py` | snapshots, their diff, the control-character scan (`out/CTRL_scan.json`) |
| `record_finish.py` | `out/SNAPSHOT_diff.json`, `out/FINISH_tools.json` |
| `make_archive_draft_redirected.py` | runs `make_archive.py --draft` unchanged with its zip written outside the repo (section 8) |

Path note: the brief's `reproduce/_common.py` is `paper/scripts/_common.py` in this repository (the
`reproduce/` layout exists only in the release build). `CONTRAST_ROWS` is at
`paper/scripts/_common.py:561`.

---

## 1. Snapshot diff

sha256 of every file under the repo root (115,088 files, 88.6 GB, `venv/` and data included) was
taken before any file was created, and compared by path and sha256 (mtime changes listed
separately):

| comparison | added | added outside `audit/review_r1/` | removed | content changed | mtime-only changes |
|---|---|---|---|---|---|
| after the analysis vs before | 38 | 0 | 0 | 0 | 0 |
| after the finish sequence (section 8) vs before | 40 | 0 | 0 | 0 | 0 |
| final, after this report and the record JSONs vs before | 44 | 0 | 0 | 0 | 0 |

**No pre-existing file changed**; every addition is inside `audit/review_r1/`.
`out/SNAPSHOT_diff.json → result.after_analysis_vs_before`, `result.after_finish_sequence_vs_before`,
`result.final_check_vs_before`. The final check was taken after this report and the record JSONs
were written. Only `SNAPSHOT_diff.json`, `CTRL_scan.json`, `scripts/ctrl_scan.py` (provenance
fields added to its JSON) and this table row were rewritten after it, all inside
`audit/review_r1/`. The snapshot files themselves are kept outside the repo (session scratchpad); their
sha256 are in `out/SNAPSHOT_diff.json → result.snapshots`. Every script ran with `python -B`, and
every finish-sequence tool with `PYTHONDONTWRITEBYTECODE=1`, so no bytecode cache was written beside
the imported repo modules.

---

## 2. Reproduction checks

**Table 2** of the compiled paper is `tab:main` (`paper/tables/T1_main.tex`); its per-fold values are
`paper/tables/T5_perfold.tex`. Each readout was recomputed on D1 by the review loaders, asserted
**bit-equal** to `_common.suite`, and compared with both printed tables at 4 dp.

| readout | f1 | f2 | f3 | f4 | fold-mean | T5 printed | Table 2 printed | bit-equal to `_common.suite` | match |
|---|---|---|---|---|---|---|---|---|---|
| baseline@8000 | 0.1429 | 0.2143 | 0.1735 | 0.2062 | 0.1842 | 0.1842 | 0.1842 | True | True |
| lambda0@4000 | 0.2245 | 0.1735 | 0.1429 | 0.1959 | 0.1842 | 0.1842 | 0.1842 | True | True |
| bnm@4000 | 0.3980 | 0.4592 | 0.3776 | 0.3814 | 0.4040 | 0.4040 | 0.4040 | True | True |
| freematch@1500 | 0.5000 | 0.5918 | 0.4286 | 0.5464 | 0.5167 | 0.5167 | 0.5167 | True | True |
| dann@4000 | 0.5408 | 0.7143 | 0.3163 | 0.4227 | 0.4985 | 0.4985 | 0.4985 | True | True |
| dann@2000 (sens.) | 0.5714 | 0.7041 | 0.5000 | 0.4433 | 0.5547 | 0.5547 | not in Table 2 | True | True |
| coral@2000 | 0.4694 | 0.6327 | 0.5000 | 0.4845 | 0.5216 | 0.5216 | 0.5216 | True | True |

`out/R_reproduction_table2.json → result.rows`, `result.table2_T1_main`, `result.all_match` (= true).
The same per-fold equality is re-asserted inside A2 and A3 before any new number is computed, and
in A3 the count-based recall (positives ≥ threshold) equals the interpolated TPR in all 28
arm×fold cells (`out/A3_recall_by_grade.json → result.threshold_checks`).

**§7 raw sd (fold 0, 32-point 250–8000 grid, D1):**

| arm | paper macro | recomputed | equal to stored `window_fold0_D1.json` | matches paper (4 dp) |
|---|---|---|---|---|
| CORAL | `\wzCoralSdOne` = 0.0529 | 0.052911 | True (exact) | True |
| FreeMatch | `\wzFreeMatchSdOne` = 0.0588 | 0.058832 | True (exact) | True |
| DANN (arm 2) | `\wzDannSdOne` = 0.1123 | 0.112342 | True (exact) | True |
| λ=0 (arm 1) | `\wzLambdaSdOne` = 0.0493 | 0.049296 | True (exact) | True |

`out/A1_resolution.json → result.reproduction_fold0_raw_sd`.

Factual line: every Table 2 value and every §7 raw sd reproduces exactly.

---

## 3. Part F

### F1. Pre-registered contrasts (`runs/lambda0_r6_prereg.json`, read-only)

Key `reporting.paired_contrasts`, verbatim:

| line | contrast | isolates |
|---|---|---|
| 112 | `lambda=0 - Baseline A (frozen)` | unfreezing alone |
| 116 | `DANN x nb3 - lambda=0` | the adversarial term alone -- the only fully controlled contrast |
| 120 | `CORAL x nb3 - lambda=0` | DA vs no DA, CONFOUNDED |
| 124 | `FreeMatch x nb3 - lambda=0` | DA vs no DA, CONFOUNDED |

**None of the four entries names a step** (`out/F1_prereg_contrasts.json → result.contrast_entries_name_a_step` = false). The steps come from elsewhere in the file:

- `budget.CONFIRM_STEPS` = 4000 (line 9), rule "THE CONTROL IS READ WHERE ITS TREATMENT IS READ".
- `reporting.computed_at` = ["the declared budget 4000", "step 1500", "step 2000"] (line 102).
- `sensitivity_readouts.steps` = [1500, 2000] (line 203); `clause` (line 200): "Reported ALONGSIDE the headline; they do NOT replace it, whichever is higher."; `what_they_buy` (line 208): "budget-matched comparators against FreeMatch x nb3 (1500) and CORAL x nb3 (2000), which the 4000 headline does not provide".
- `budget.alternative_considered_and_declined`: reading λ=0 at each DA arm's own budget was considered and declined, "partially recovered by the pre-declared sensitivity readouts at 1500 and 2000".

Reading of each contrast (λ=0 at its declared 4000; the other arm at its own declared read-off,
`_common.ARM_META`, which `crosscheck.py` verifies against each arm's budget file):

| contrast | headline form in the file | pre-declared sensitivity form | `CONTRAST_ROWS` index (headline) | index (sensitivity) |
|---|---|---|---|---|
| λ=0 − Baseline A | λ=0 4000 / Baseline A 8000 | — | 0 | — |
| DANN − λ=0 | 4000 / 4000 | — | 1 | — |
| CORAL − λ=0 | **2000 / 4000** | 2000 / 2000 (sensitivity readout) | 5 (labelled `unmatched`, footnote c) | 2 (labelled `matched`) |
| FreeMatch − λ=0 | **1500 / 4000** | 1500 / 1500 (sensitivity readout) | 6 (labelled `unmatched`, footnote c) | 3 (labelled `matched`) |

Rows 4, 7, 8, 9, 10, 11, 12 (BNM − λ=0; BNM − CORAL; BNM − DANN; BNM − FreeMatch; DANN − CORAL;
DANN − FreeMatch; DANN − CORAL at 2000) correspond to none of the four.
`out/F1_prereg_contrasts.json → result.reading_of_each_contrast`, `result.contrast_rows`,
`result.contrast_rows_not_corresponding_to_any_preregistered_contrast`.

Factual line: the file fixes λ=0's headline at 4000 and lists 1500/2000 as sensitivity readouts; the
contrast entries carry no step.

### F2. Class priors inside the objectives

**FreeMatch.** The fairness (SAF) target marginal is **uniform [0.5, 0.5]** over [non-lens, lens].

- Config key `fairness_target` = `null` in every stored run config, `runs/freematch_nb3_unfrozen_fold{0..4}/run_config.json:37` (`saf_weight` 0.01).
- Set by `scripts/run_unfreeze_freematch.py:133` `FAIRNESS_TARGET = None`, passed at `:625`; `scripts/train_skeleton.py:284` `freematch_fairness_target: tuple | None = None  # None -> uniform [0.5,0.5] (balanced pool)`, passed at `:375`.
- `scripts/da_losses.py:874-875` (docstring: "None -> uniform [1/C]*C"), `:918` `init = 1.0 / self.num_classes`, `:922` `if fairness_target is None:`, `:923` `ft = torch.full((self.num_classes,), init)`, `:930` `self.register_buffer("fairness_target", ft)`, used at `:1019` `saf = self_adaptive_fairness_penalty(mean_pred, self.fairness_target, self.eps)`; comment at `:727`.

**BNM.** **No class-prior parameter.** `BNMLoss.__init__(self, gamma: float = 1.0)` (`da_losses.py:1578`) takes only γ; the objective is `-nuclear_norm(prob2)/min(B,C)` (`:1453`, `:1598-1599`, `:1616`). No line in `BNMLoss` (`:1555-1619`) mentions a prior, marginal or fairness term, and no `runs/bnm_r6_folds/*/run_config.json` key matches prior/fair/marginal.
Related, but not in the objective: the BNM runner's **target batch sampler** is class-balanced on the
target lens/non-lens split, `scripts/run_bnm_r6.py:104` `POS_FRACTION = 0.5`, `:350`
`tgt_sampler = ClassBalancedBatchSampler(holdout["train_pos"], holdout["train_neg"],`, `:352`
`pos_fraction=POS_FRACTION,`.

`out/F2_class_priors.json → result.freematch`, `result.bnm`.

Factual line: FreeMatch's SAF target is uniform [0.5, 0.5] by default; BNM has no prior parameter.

### F3. The §3.1 development probes

Paper text: `paper/sections/03_data.tex:31` (AUROC 0.995) and `:33` (≈ 0.84). Record:
`PROJECTv7.md:789`, `:791`. An earlier pass reached the same result:
`paper/RELEASE_CURATION_REPORT.md:31`.

| number | producer | evidence |
|---|---|---|
| AUROC 0.995, per-image brightness | **not identified** | no log, output file or note links 0.995 to a script; both candidates print their AUROC to stdout only; `validate_singleband.py`'s one file output (`sim_vs_real_v2.png`, `:143`) is not on disk |
| residual morphology gap ≈ 0.84 AUROC | **not identified** | `probe_arc_on_real.py` and `probe_realism.py` print to stdout only and no captured output exists; their input `data/sim_arc_on_real/arc_on_real_debug.npz` is on disk, but re-running them would load the Zoobot encoder |

What each candidate's code measures (for reference only, not an attribution):

| script | features | classifier | classes / data |
|---|---|---|---|
| `validate_singleband.py` | per-image mean, std, 99th percentile after `StandardizeAsinh` (`:47-48`) | logistic regression, 5-fold `cross_val_predict` (`:59`) | regenerated single-band sims (`--sim-dir`) vs real Euclid `target_unlabeled` (`:95`); prints "BRIGHTNESS-SHORTCUT AUROC" (`:108`) |
| `probe_deflector_gap.py` | (1) mean/std/p99 of raw stamps (`:58`); (2) frozen Zoobot features | balanced logistic regression, 5-fold CV (`:68`) | synthetic deflectors `data/sim_nonlens/shards` (`:32`; on disk only under `data_old/`) vs real negatives `data/q1_unlabeled/shards` (`:33`) |
| `probe_arc_on_real.py` | frozen Zoobot features | logistic regression, 5-fold CV | sim-lens vs real-neg, sim-lens vs bare deflector, bare deflector vs real-neg (`:92`, `:95`, `:98`) |
| `probe_realism.py` | frozen Zoobot features | logistic regression, 5-fold CV | (A) sim-lens vs real-lens, (B) real-lens vs real-neg, (C) sim-lens vs real-neg, (D) bare deflector vs real-lens (`:10-18`) |

`out/F3_dev_probes.json`.

Factual line: neither number can be tied to a script from the record.

### F4. Fold overlap (`data/splits/splits.json`)

Shared **target lenses**, count (fraction of the row fold's own target pool):

| | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| fold 0 | 391 (1.000) | 293 (0.749) | 293 (0.749) | 293 (0.749) | 294 (0.752) |
| fold 1 | 293 (0.749) | 391 (1.000) | 293 (0.749) | 293 (0.749) | 294 (0.752) |
| fold 2 | 293 (0.749) | 293 (0.749) | 391 (1.000) | 293 (0.749) | 294 (0.752) |
| fold 3 | 293 (0.749) | 293 (0.749) | 293 (0.749) | 391 (1.000) | 294 (0.752) |
| fold 4 | 294 (0.750) | 294 (0.750) | 294 (0.750) | 294 (0.750) | 392 (1.000) |

- **Shared target negatives:** 0 for every pair of distinct folds (391/391/391/391/392 on the diagonal).
- **Shared source rows:** all of them. `splits.json` has one `source_train` block: 24,000 simulated positives + 4,455 grade-X + 2,881 bare deflectors + 16,664 random galaxies = 47,000 rows, identical for every pair.
- **Shared test negatives:** all of them. One `test_negatives_shared` block: 26,399 per pair on the pre-registered pool, 26,134 on D1.
- Within each fold, target lenses and test lenses are disjoint (all 5 folds); test lenses are disjoint across folds.

`out/F4_fold_overlap.json → result.shared_target_lenses`, `shared_target_negatives`,
`shared_source_rows`, `shared_test_negatives`, `lens_target_vs_test_disjoint_within_fold`.

Factual line: any two folds share 293–294 of their ~391 target lenses (≈ 75%).

### F5. Host properties as a class cue in training

Available locally: the deflector table (`data_old/sim_deflectors/deflector_cutouts_clean.npz`: stellar
mass, photo-z, `mvis`) for deflector-derived objects only; the MER query output (`flux_vis_psf`,
→ AB mag with the §3 definition) for any object it contains; and catalogue v0.0.1's
`flux_vis_1fwhm_aper` (a different flux definition) for grade-X objects. The `mvis` values are
negative (median −23.8), and the query that produced them is not stored.

| population | variable | available | p05 | p50 | p95 | mean |
|---|---|---|---|---|---|---|
| (a) hosts of the 24,000 training positives | stellar mass | 24000/24000 | 11.266 | 11.404 | 11.718 | 11.442 |
| | photo-z (`phz_median`) | 24000/24000 | 0.410 | 0.650 | 0.770 | 0.629 |
| | `mvis` (deflector table) | 24000/24000 | −24.612 | −23.794 | −23.116 | −23.827 |
| | VIS PSF mag (MER) | 15628/24000 | 21.306 | 22.286 | 22.891 | 22.215 |
| (b1) bare deflectors | stellar mass | 2881/2881 | 11.249 | 11.383 | 11.703 | 11.421 |
| | photo-z | 2881/2881 | 0.380 | 0.610 | 0.760 | 0.596 |
| | `mvis` | 2881/2881 | −24.549 | −23.761 | −23.097 | −23.795 |
| | VIS PSF mag (MER) | 1935/2881 | 21.015 | 22.169 | 22.868 | 22.084 |
| (b2) random galaxies | VIS PSF mag (MER) | 16664/16664 | 20.771 | 22.387 | 22.949 | 22.191 |
| (b3) grade-X | VIS PSF mag (MER) | 1120/4455 | 20.936 | 22.074 | 22.915 | 22.018 |
| | VIS 1-FWHM aperture mag (cat. v0.0.1) | 3987/4455 | 20.827 | 22.697 | 24.077 | 22.574 |

Stellar mass, photo-z and `mvis` are **not available** for random galaxies or grade-X objects. The
24,000 positives use 2,881 distinct deflectors.

Single-variable AUROC, training positives (label 1) vs training negatives, score = the variable
itself:

| variable | vs bare deflectors | vs random galaxies | vs grade-X | vs all negatives that have it |
|---|---|---|---|---|
| stellar mass | 0.5438 (24000/2881) | not available | not available | 0.5438 (24000/2881) |
| photo-z | 0.5798 (24000/2881) | not available | not available | 0.5798 (24000/2881) |
| `mvis` | 0.4801 (24000/2881) | not available | not available | 0.4801 (24000/2881) |
| VIS PSF mag (MER) | 0.5640 (15628/1935) | 0.4648 (15628/16664) | 0.5839 (15628/1120) | 0.4813 (15628/19719) |

`out/F5_host_properties.json → result.availability`, `result.distributions`,
`result.single_variable_auroc_positive_vs_negative` (also `auroc_direction_free`).

Factual line: the single-variable AUROCs range from 0.4648 to 0.5839.

### F6. The 20 D2 lenses

Match method for all 20: exact Euclid object ID (lens ID from `id_str`; bare-deflector and
simulation-host IDs rebuilt from RA/Dec because the stored ones are float-rounded), via
`scripts/exact_ids.contaminant_sets` (`d2_pos_remove`). Target-negative lists were also checked: no
D2 lens is a target negative.

| lens row | id_str | fold | grade | role(s) | matched in (`splits.json` key) |
|---|---|---|---|---|---|
| 62 | 102020064_NEG635872258496841682 | 4 | A | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 97 | 102021503_NEG655073934479995542 | 2 | A | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 163 | 102157958_2727137773639721420 | 1 | A | validation negative (bare deflector); simulation host | `val.neg_bare_deflector.deflector_object_ids`; sim shard `object_id` → deflector table |
| 172 | 102158276_2727570223645896962 | 3 | A | validation negative (random galaxy) | `val.neg_random.object_ids` |
| 174 | 102158584_2687900321650859511 | 2 | A | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 192 | 102158894_2730612507653143373 | 4 | A | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 209 | 102159486_2695397736666928563 | 0 | A | validation negative (random galaxy) | `val.neg_random.object_ids` |
| 224 | 102160337_2701254133678933511 | 4 | A | validation negative (random galaxy) | `val.neg_random.object_ids` |
| 257 | 102021013_NEG606090281487314103 | 0 | B | validation negative (random galaxy) | `val.neg_random.object_ids` |
| 266 | 102021497_NEG611694459481354870 | 3 | B | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 311 | 102045468_NEG535850797268396974 | 1 | B | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 314 | 102046111_NEG522631915266742195 | 4 | B | validation negative (random galaxy) | `val.neg_random.object_ids` |
| 323 | 102157957_2711813162637737494 | 4 | B | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 340 | 102158896_2760272555655712464 | 0 | B | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 382 | 102022479_NEG623367075470669448 | 4 | A | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 384 | 102022973_NEG606454307466806991 | 1 | A | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 393 | 102046761_NEG532020950261969455 | 1 | A | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 461 | 102046761_NEG534264558260612632 | 2 | B | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 479 | 102159771_2643073721671053537 | 4 | B | training negative (random galaxy) | `source_train.neg_random.object_ids` |
| 488 | 102160610_2728596478683519590 | 4 | B | training negative (random galaxy) | `source_train.neg_random.object_ids` |

Tallies: roles 14 training negative (random galaxy), 5 validation negative (random galaxy), 1
validation negative (bare deflector) + simulation host (same lens); grades 11 A, 9 B; folds 0:3, 1:4,
2:3, 3:2, 4:8. `out/F6_d2_lenses.json → result.lenses`, `role_tally`, `grade_tally`, `fold_tally`.

Factual line: 19 of the 20 are random-galaxy training or validation negatives.

### F7. The 28 catalogued lenses in the pre-registered test pool

The 28 = graded lenses sharing an exact Euclid ID with a shared test negative (`lens_in_test`); 19
grade A, 9 grade B. Of the 28, 5 / 8 / 6 / 4 are also that fold's own test positives (folds 1–4).
Threshold = `metrics.operating_threshold` at FPR = 1e-3 on that fold's **pre-registered-pool** vector
(26,399 negatives); 26 negatives score at or above it in every arm×fold.

| readout | f1 | f2 | f3 | f4 | sum (of 112) | median rank f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|---|---|---|
| baseline@8000 | 7 | 7 | 7 | 7 | 28 | 70.0 | 70.0 | 70.0 | 70.0 |
| lambda0@4000 | 9 | 7 | 8 | 7 | 31 | 84.5 | 159.5 | 134.0 | 67.5 |
| bnm@4000 | 13 | 13 | 15 | 17 | 58 | 29.5 | 30.5 | 25.5 | 17.0 |
| freematch@1500 | 15 | 15 | 16 | 17 | 63 | 23.0 | 25.5 | 19.5 | 20.0 |
| dann@4000 | 16 | 17 | 13 | 13 | 59 | 19.5 | 16.5 | 27.5 | 31.0 |
| dann@2000 (sens.) | 18 | 16 | 18 | 15 | 67 | 21.5 | 15.0 | 17.0 | 22.5 |
| coral@2000 | 16 | 17 | 18 | 17 | 68 | 21.5 | 22.0 | 17.5 | 21.0 |

Rank = position among the 26,399 pre-registered-pool negatives, 1 = highest score.
`out/F7_lenses_in_test_pool.json → result.per_arm.<readout>.<fold>.n28_at_or_above_threshold`,
`median_rank_of_28`, `ranks_of_28`, `n_negatives_at_or_above_threshold`,
`n28_that_are_also_this_folds_test_positives`.

Factual line: on the pre-registered pool, 13–18 of the ~26 above-threshold negatives per DA-arm fold
are catalogued lenses (7–9 for Baseline A and λ=0).

---

## 4. Part A

### A1. §7 resolution limit, redone

**Availability** (`out/A1_availability.json → result.grids`):

| arm | fold 0 | folds 1–4 (each) |
|---|---|---|
| λ=0 | arm 1 dev run, 32 of 32 grid steps (250–8000) | 16 steps, 250–4000, every 250 |
| BNM | dev8000 run, 32/32 | 16 steps, 250–4000, every 250 |
| FreeMatch | 8000-step diagnostic run, 32/32 (64 stored, subsampled to the grid as §7 does) | 12 steps, 125–1500, every 125 |
| DANN | arm 2 dev run, 32/32 | 16 steps, 250–4000, every 250 |
| CORAL | dev8000 run, 32/32 | 8 steps, 250–2000, every 250 |
| Baseline A | step 8000 only | step 8000 only — **no series** (earlier steps exist only as head checkpoints, which this brief may not load) |

Series values: `out/A1_series.json → result.series."<arm>|fold<k>"`.

**Per-series estimators** (D1). (a) = sd of successive differences / √2; (b) = residual sd after a
least-squares line in step (ddof 2); lag-1 = autocorrelation of the (b) residuals.

| arm | scope | n | raw sd | (a) | (b) | lag-1 |
|---|---|---|---|---|---|---|
| λ=0 | fold 0 | 32 | 0.0493 | 0.0539 | 0.0468 | −0.4017 |
| λ=0 | fold 1 | 16 | 0.0453 | 0.0425 | 0.0398 | −0.1680 |
| λ=0 | fold 2 | 16 | 0.0448 | 0.0311 | 0.0313 | −0.0790 |
| λ=0 | fold 3 | 16 | 0.0441 | 0.0388 | 0.0368 | −0.1164 |
| λ=0 | fold 4 | 16 | 0.0487 | 0.0395 | 0.0441 | 0.1920 |
| λ=0 | **folds 1–4 mean** | | 0.0457 | 0.0380 | 0.0380 | −0.0428 |
| BNM | fold 0 | 32 | 0.0841 | 0.0714 | 0.0750 | 0.0839 |
| BNM | fold 1 | 16 | 0.0633 | 0.0680 | 0.0651 | −0.1358 |
| BNM | fold 2 | 16 | 0.0657 | 0.0517 | 0.0636 | 0.2307 |
| BNM | fold 3 | 16 | 0.0504 | 0.0581 | 0.0486 | −0.4272 |
| BNM | fold 4 | 16 | 0.0505 | 0.0527 | 0.0522 | −0.0474 |
| BNM | **folds 1–4 mean** | | 0.0575 | 0.0576 | 0.0574 | −0.0949 |
| FreeMatch | fold 0 | 32 | 0.0588 | 0.0413 | 0.0430 | 0.0543 |
| FreeMatch | fold 1 | 12 | 0.0425 | 0.0330 | 0.0445 | 0.3467 |
| FreeMatch | fold 2 | 12 | 0.0390 | 0.0325 | 0.0409 | 0.1801 |
| FreeMatch | fold 3 | 12 | 0.0382 | 0.0368 | 0.0387 | −0.0365 |
| FreeMatch | fold 4 | 12 | 0.0394 | 0.0399 | 0.0413 | 0.0665 |
| FreeMatch | **folds 1–4 mean** | | 0.0398 | 0.0356 | 0.0414 | 0.1392 |
| DANN | fold 0 | 32 | 0.1123 | 0.0942 | 0.0968 | 0.0501 |
| DANN | fold 1 | 16 | 0.1220 | 0.0931 | 0.1023 | 0.0876 |
| DANN | fold 2 | 16 | 0.1194 | 0.1080 | 0.1069 | −0.0883 |
| DANN | fold 3 | 16 | 0.1054 | 0.0573 | 0.0938 | 0.4629 |
| DANN | fold 4 | 16 | 0.0575 | 0.0550 | 0.0580 | 0.0219 |
| DANN | **folds 1–4 mean** | | 0.1011 | 0.0784 | 0.0903 | 0.1210 |
| CORAL | fold 0 | 32 | 0.0529 | 0.0472 | 0.0493 | 0.0763 |
| CORAL | fold 1 | 8 | 0.0599 | 0.0501 | 0.0639 | 0.2688 |
| CORAL | fold 2 | 8 | 0.1060 | 0.0598 | 0.0721 | 0.1825 |
| CORAL | fold 3 | 8 | 0.0514 | 0.0357 | 0.0433 | 0.0047 |
| CORAL | fold 4 | 8 | 0.0460 | 0.0349 | 0.0402 | −0.0994 |
| CORAL | **folds 1–4 mean** | | 0.0658 | 0.0451 | 0.0549 | 0.0892 |

`out/A1_resolution.json → result.per_series`, `result.sigma_by_arm`.

**Like-for-like ratios.** σ = folds 1–4 mean of the per-fold detrended sd (fold-1–4 σ is available
for every DA arm, so fold 0 is not used for σ). `noise = sqrt(σ_A² + σ_B²)/2`; ratio = noise/|gap|.
Gap = fold-mean TPR difference, folds 1–4, D1. Old ratio = each arm's raw fold-0 sd / |gap|.

| gap | form | steps A/B | both steps declared readouts? | gap | ratio (a) | ratio (b) | old ratio, A | old ratio, B |
|---|---|---|---|---|---|---|---|---|
| DANN − CORAL | declared budget | 4000/2000 | yes/yes | −0.0231 | 1.956 | 2.284 | DANN 4.860 | CORAL 2.289 |
| DANN − CORAL | budget-matched | 2000/2000 | yes/yes | +0.0331 | 1.368 | 1.597 | DANN 3.398 | CORAL 1.601 |
| DANN − FreeMatch | declared budget | 4000/1500 | yes/yes | −0.0182 | 2.368 | 2.731 | DANN 6.182 | FreeMatch 3.237 |
| DANN − FreeMatch | budget-matched | 1500/1500 | **no**/yes | −0.0922 | 0.467 | 0.539 | DANN 1.219 | FreeMatch 0.638 |
| CORAL − FreeMatch | declared budget | 2000/1500 | yes/yes | +0.0049 | 5.809 | 6.948 | CORAL 10.702 | FreeMatch 11.899 |
| CORAL − FreeMatch | budget-matched | 1500/1500 | **no**/yes | +0.0024 | 11.743 | 14.045 | CORAL 21.633 | FreeMatch 24.054 |
| BNM − CORAL | declared budget | 4000/2000 | yes/yes | −0.1176 | 0.311 | 0.337 | BNM 0.715 | CORAL 0.450 |
| BNM − CORAL | budget-matched | 2000/2000 | yes/yes | −0.1176 | 0.311 | 0.338 | BNM 0.715 | CORAL 0.450 |
| BNM − DANN | declared = matched | 4000/4000 | yes/yes | −0.0945 | 0.515 | 0.566 | BNM 0.890 | DANN 1.189 |
| BNM − FreeMatch | declared budget | 4000/1500 | yes/yes | −0.1127 | 0.300 | 0.314 | BNM 0.746 | FreeMatch 0.522 |
| BNM − FreeMatch | budget-matched | 1500/1500 | yes/yes | −0.1334 | 0.254 | 0.265 | BNM 0.630 | FreeMatch 0.441 |

"no" = the stored checkpoint exists (DANN 1500, CORAL 1500) but is not a declared readout of that
arm. CORAL − FreeMatch is a ratio only, not a contrast interval.
`out/A1_resolution.json → result.gaps_and_ratios."<A>-<B>|<form>"` (`gap`,
`ratio_detrended_a_diff`, `ratio_detrended_b_linfit_resid`, `old_ratio_raw_fold0_sd_over_gap`).

**Window averaging on detrended series.** Windows = k consecutive stored checkpoints, `mean_logit`,
every valid position (the positions `analyze_window_average` uses); σ₁ at the same positions.
Reduction = σ₁/σ_k with each estimator, folds 1–4 = ratio of the fold-means of σ. For reference,
estimator (a) applied to an overlapping k-window mean of independent noise shrinks by **k**, not √k
(successive window means differ by (x_new − x_old)/k); that analytic value is listed, not measured.

| arm | scope | k | positions per fold | reduction (a) | √k | k (white noise, est. a) | reduction (b) | reduction, raw sd | (a) > √k? | lag-1 if so |
|---|---|---|---|---|---|---|---|---|---|---|
| λ=0 | fold 0 | 2 | 31 | 1.869 | 1.414 | 2 | 1.537 | 1.188 | yes | −0.4017 |
| λ=0 | fold 0 | 4 | 29 | 4.386 | 2.000 | 4 | 1.860 | 1.253 | yes | −0.4017 |
| λ=0 | fold 0 | 8 | 25 | 5.456 | 2.828 | 8 | 2.734 | 1.494 | yes | −0.4017 |
| λ=0 | folds 1–4 | 2 | 15 | 1.404 | 1.414 | 2 | 1.082 | 0.994 | no | |
| λ=0 | folds 1–4 | 4 | 13 | 2.295 | 2.000 | 4 | 1.784 | 1.101 | yes | −0.0428 |
| λ=0 | folds 1–4 | 8 | 9 | 2.820 | 2.828 | 8 | 2.029 | 1.354 | no | |
| BNM | fold 0 | 2 | 31 | 1.716 | 1.414 | 2 | 1.414 | 1.335 | yes | 0.0839 |
| BNM | fold 0 | 4 | 29 | 2.223 | 2.000 | 4 | 1.539 | 1.345 | yes | 0.0839 |
| BNM | fold 0 | 8 | 25 | 5.473 | 2.828 | 8 | 4.475 | 1.840 | yes | 0.0839 |
| BNM | folds 1–4 | 2 | 15 | 1.721 | 1.414 | 2 | 1.470 | 1.405 | yes | −0.0949 |
| BNM | folds 1–4 | 4 | 13 | 2.948 | 2.000 | 4 | 2.232 | 1.873 | yes | −0.0949 |
| BNM | folds 1–4 | 8 | 9 | 4.222 | 2.828 | 8 | 3.573 | 3.100 | yes | −0.0949 |
| FreeMatch | fold 0 | 2 | 31 | 1.645 | 1.414 | 2 | 1.192 | 1.140 | yes | 0.0543 |
| FreeMatch | fold 0 | 4 | 29 | 2.454 | 2.000 | 4 | 1.600 | 1.160 | yes | 0.0543 |
| FreeMatch | fold 0 | 8 | 25 | 3.326 | 2.828 | 8 | 2.400 | 1.194 | yes | 0.0543 |
| FreeMatch | folds 1–4 | 2 | 11 | 1.444 | 1.414 | 2 | 1.166 | 1.204 | yes | 0.1392 |
| FreeMatch | folds 1–4 | 4 | 9 | 1.817 | 2.000 | 4 | 1.274 | 1.422 | no | |
| FreeMatch | folds 1–4 | 8 | 5 | 2.298 | 2.828 | 8 | 1.873 | 2.249 | no | |
| DANN | fold 0 | 2 | 31 | 1.741 | 1.414 | 2 | 1.244 | 1.243 | yes | 0.0501 |
| DANN | fold 0 | 4 | 29 | 2.405 | 2.000 | 4 | 1.606 | 1.448 | yes | 0.0501 |
| DANN | fold 0 | 8 | 25 | 5.270 | 2.828 | 8 | 2.730 | 1.913 | yes | 0.0501 |
| DANN | folds 1–4 | 2 | 15 | 1.618 | 1.414 | 2 | 1.250 | 1.218 | yes | 0.1210 |
| DANN | folds 1–4 | 4 | 13 | 2.589 | 2.000 | 4 | 1.895 | 1.576 | yes | 0.1210 |
| DANN | folds 1–4 | 8 | 9 | 5.190 | 2.828 | 8 | 4.887 | 3.241 | yes | 0.1210 |
| CORAL | fold 0 | 2 | 31 | 1.488 | 1.414 | 2 | 1.276 | 1.252 | yes | 0.0763 |
| CORAL | fold 0 | 4 | 29 | 2.483 | 2.000 | 4 | 2.049 | 1.898 | yes | 0.0763 |
| CORAL | fold 0 | 8 | 25 | 4.061 | 2.828 | 8 | 3.128 | 2.510 | yes | 0.0763 |
| CORAL | folds 1–4 | 2 | 7 | 1.335 | 1.414 | 2 | 1.175 | 0.898 | no | |
| CORAL | folds 1–4 | 4 | 5 | 1.993 | 2.000 | 4 | 1.891 | 1.241 | no | |
| CORAL | folds 1–4 | 8 | 1 | n/a | 2.828 | 8 | n/a | n/a | n/a (one position) | |

Lag-1 shown is the arm's k = 1 linear-fit-residual autocorrelation (fold 0, or the folds 1–4 mean;
per-fold values in `sigma_by_arm`). `out/A1_resolution.json → result.window_averaging.<arm>.<fold0|folds1_4>.<k>`.

Like-for-like ratio against k (σ_k = folds 1–4 mean of estimator (a) on the k-window series), used
in panel (a) of the draft figure:

| gap / form | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| DANN − CORAL, declared | 1.956 | 1.193 | 0.685 | n/a |
| DANN − CORAL, matched | 1.368 | 0.834 | 0.479 | n/a |
| DANN − FreeMatch, declared | 2.368 | 1.509 | 0.981 | 0.642 |
| DANN − FreeMatch, matched | 0.467 | 0.298 | 0.194 | 0.127 |
| CORAL − FreeMatch, declared | 5.809 | 3.555 | 2.354 | n/a |
| CORAL − FreeMatch, matched | 11.743 | 7.186 | 4.758 | n/a |
| BNM − CORAL, declared | 0.311 | 0.178 | 0.099 | n/a |
| BNM − CORAL, matched | 0.311 | 0.178 | 0.099 | n/a |
| BNM − DANN | 0.515 | 0.314 | 0.189 | 0.107 |
| BNM − FreeMatch, declared | 0.300 | 0.184 | 0.126 | 0.101 |
| BNM − FreeMatch, matched | 0.254 | 0.155 | 0.107 | 0.086 |

`out/A1_resolution.json → result.ratio_vs_k_estimator_a_window_averaged`.

Factual line: at k = 1 the like-for-like ratio is above 1 for the three gaps among DANN, CORAL and
FreeMatch at declared budgets and below 1 for every gap involving BNM.

### A2. Top-scoring test negatives (D1)

Per readout (Baseline A 8000, λ=0 4000, BNM 4000, FreeMatch 1500, DANN 4000, DANN 2000, CORAL 2000)
and fold 1–4: 26 negatives at or above the threshold in every one of the 28 arm×fold cells, plus the
top 30. Union: **191 unique objects**; 168 above threshold in ≥ 1 cell; 104 in ≥ 2 cells.
Per-cell lists: `out/A2_top_negatives.json → result.per_arm_fold.<readout>.<fold>.above_threshold`,
`.top30`; count distribution `result.count_distribution`. One row per object:
`out/top_negatives.csv` (Euclid ID, RA/Dec and VIS PSF mag from the stored MER query output,
number of cells above threshold, arms, cells in top 30, best rank, catalogue match, location).

**Cross-match** (`out/A2_crossmatch.json`). All 191 have stored coordinates. Discovery Engine
v0.0.3 (2,584 rows; 21 rows carry no Euclid ID and are matchable by position only) and v0.0.1
(7,354 rows): 22 objects match by exact ID in each, and the same 22 by position within 1″ (every
positional match has the identical exact ID; positional matching added no object). None is within
1″ of a graded (A/B) lens in `lens_cutouts.npz`. **All 22 matches are grade C in both versions**;
none is grade A, B or X.

| Euclid ID | VIS PSF mag | cells above threshold (of 28) | best rank | readouts above threshold |
|---|---|---|---|---|
| −614740525474565457 | 20.9963 | 21 | 1 | bnm, coral, dann@2000, dann@4000, freematch, lambda0 |
| 2721660958665742075 | 22.6091 | 20 | 5 | all seven |
| 2674724083652311112 | 21.6466 | 11 | 1 | baseline, bnm, dann@2000, dann@4000, lambda0 |
| −652726602487221405 | 22.7864 | 10 | 3 | coral, dann@2000, dann@4000, freematch |
| 2691532629662288078 | 21.6631 | 9 | 10 | coral, dann@2000, dann@4000, freematch, lambda0 |
| −582908879476330138 | 22.6574 | 6 | 4 | bnm, coral, dann@4000, freematch, lambda0 |
| 2664946780647342201 | 20.9821 | 5 | 12 | coral, dann@2000, dann@4000 |
| 2745097475664476790 | 22.2178 | 4 | 13 | coral, dann@2000, dann@4000 |
| 2745692749646546479 | 22.8054 | 4 | 13 | dann@2000, dann@4000, freematch, lambda0 |
| −519232663294154195 | 22.3629 | 3 | 2 | bnm, dann@2000, freematch |
| −608230289466641729 | 22.5018 | 3 | 3 | dann@4000 |
| −569626044493122214 | 21.5420 | 3 | 7 | dann@4000, freematch |
| 2672567324677621558 | 22.1702 | 2 | 1 | freematch |
| −607331071464380944 | 22.5872 | 1 | 3 | coral |
| −620897273485969972 | 22.3351 | 1 | 11 | bnm |
| −573907589479199428 | 20.8857 | 1 | 19 | freematch |
| −651332364480516764 | 21.0530 | 1 | 20 | coral |
| 2665236167644831980 | 21.5294 | 1 | 25 | freematch |
| −529964798269418977 | 22.6534 | 1 | 26 | freematch |
| −650712882464619007 | 21.8920 | 0 (top 30 only) | 28 | — |
| −519281943288837491 | 21.4822 | 0 (top 30 only) | 28 | — |
| −532406933273857282 | 21.8465 | 0 (top 30 only) | 30 | — |

`out/A2_crossmatch.json → result.objects_matching_a_grade_A_B_C_X_entry`; the same flags are the
`catalogue_match_grades` column of `out/top_negatives.csv`.

**Galleries:** `fig/top_neg_page_01.png` … `fig/top_neg_page_06.png` (pages 1–5 hold 25 objects,
page 6 holds 4): the 104 objects above threshold in ≥ 2 cells sorted by that count (then best rank),
then the first 25 distinct objects above CORAL@2000's threshold by best CORAL rank (some objects
appear in both groups). Each object is drawn with the encoder stretch (`lens_dataset.zoobot_stretch`,
as Figure 2) and a linear clip at the 1st/99.5th percentiles; labels give ID, cell count and VIS PSF
mag. `out/A2_top_negatives.json → result.galleries`.

**Magnitude bias** (AB mag = 23.9 − 2.5 log₁₀ `flux_vis_psf`[µJy], fainter = mag > 22.5, as §3 /
`make_m_prose.test_faint_fraction`, which this value reproduces exactly):

| set | n | fraction fainter than 22.5 |
|---|---|---|
| whole D1 pool | 26,134 | 0.4279 |
| above-threshold negatives, unique objects | 168 | 0.3393 |
| above-threshold negatives, all arm×fold instances | 728 | 0.3187 |
| baseline@8000 instances | 104 | 0.4231 |
| lambda0@4000 instances | 104 | 0.2596 |
| bnm@4000 instances | 104 | 0.3462 |
| freematch@1500 instances | 104 | 0.2596 |
| dann@4000 instances | 104 | 0.3077 |
| dann@2000 instances | 104 | 0.3269 |
| coral@2000 instances | 104 | 0.3077 |

`out/A2_magnitude_bias.json` (also percentiles and per-readout medians).

Factual line: 22 of the 191 top-scoring negatives are catalogued, all grade C; none is grade A, B or
X.

### A3. Astronomer-facing breakdowns (descriptive; no contrasts)

**Recall by grade**, folds 1–4, each fold's own FPR = 1e-3 threshold on D1 (recovered / total;
Clopper–Pearson 95% on the sums). No grade-C lens is a test positive.

| readout | grade | f1 | f2 | f3 | f4 | summed | rate | CP 95% |
|---|---|---|---|---|---|---|---|---|
| baseline@8000 | A | 10/49 | 14/49 | 12/49 | 16/49 | 52/196 | 0.2653 | [0.2049, 0.3329] |
| baseline@8000 | B | 4/49 | 7/49 | 5/49 | 4/48 | 20/195 | 0.1026 | [0.0638, 0.1540] |
| lambda0@4000 | A | 16/49 | 13/49 | 11/49 | 15/49 | 55/196 | 0.2806 | [0.2189, 0.3491] |
| lambda0@4000 | B | 6/49 | 4/49 | 3/49 | 4/48 | 17/195 | 0.0872 | [0.0516, 0.1359] |
| bnm@4000 | A | 27/49 | 32/49 | 28/49 | 29/49 | 116/196 | 0.5918 | [0.5195, 0.6613] |
| bnm@4000 | B | 12/49 | 13/49 | 9/49 | 8/48 | 42/195 | 0.2154 | [0.1599, 0.2798] |
| freematch@1500 | A | 32/49 | 34/49 | 32/49 | 38/49 | 136/196 | 0.6939 | [0.6242, 0.7576] |
| freematch@1500 | B | 17/49 | 24/49 | 10/49 | 15/48 | 66/195 | 0.3385 | [0.2724, 0.4095] |
| dann@4000 | A | 32/49 | 43/49 | 22/49 | 32/49 | 129/196 | 0.6582 | [0.5872, 0.7243] |
| dann@4000 | B | 21/49 | 27/49 | 9/49 | 9/48 | 66/195 | 0.3385 | [0.2724, 0.4095] |
| dann@2000 (sens.) | A | 35/49 | 42/49 | 35/49 | 32/49 | 144/196 | 0.7347 | [0.6671, 0.7951] |
| dann@2000 (sens.) | B | 21/49 | 27/49 | 14/49 | 11/48 | 73/195 | 0.3744 | [0.3063, 0.4463] |
| coral@2000 | A | 32/49 | 38/49 | 35/49 | 34/49 | 139/196 | 0.7092 | [0.6402, 0.7717] |
| coral@2000 | B | 14/49 | 24/49 | 14/49 | 13/48 | 65/195 | 0.3333 | [0.2676, 0.4042] |

`out/A3_recall_by_grade.json → result.by_arm.<readout>.<A|B>`.

**Per-lens columns available locally** (`out/A3_lens_columns.json`): catalogue v0.0.3 (`subset`,
`id_str`, `object_id`, `tile_index`, RA/Dec, `segmentation_area`, `flux_vis_1fwhm_aper`,
`expert_score`, `grade`, `expert_total_votes`, `notes`, `references`); v0.0.1 (same core plus
`score_judges`, `duplicate_group*`, `expert_total-votes`); `lens_manifest_positive.csv`;
`lens_cutouts.npz` (`id_str`, `grade`, `expert_score`, `source_zip`, RA/Dec, `object_id`); and four
lens-modelling tables beside the catalogues in `data_old/q1_lenses/` covering 335 lenses
(`modeling_lens_mass.csv` with `einstein_radius_effective_*` and `einstein_radius_*`;
`modeling_lens_sersic.csv`; `modeling_lens_mge.csv`; `modeling_mge_magnitude.csv` with lensed-source
magnitude, flux and S/N in VIS and NIR). Full column lists in the JSON.

**Recall by lens property** (tertiles of the folds 1–4 test lenses that have a value; recovered /
total, rate):

*Expert score* (`lens_cutouts.npz`, 391/391; edges 1.5029 / 1.7575 / 2.2438 / 3.1584; counts 131 /
130 / 130; grade mix B131 | A66+B64 | A130):

| readout | low | mid | high |
|---|---|---|---|
| baseline@8000 | 10/131 (0.076) | 22/130 (0.169) | 40/130 (0.308) |
| lambda0@4000 | 10/131 (0.076) | 19/130 (0.146) | 43/130 (0.331) |
| bnm@4000 | 23/131 (0.176) | 51/130 (0.392) | 84/130 (0.646) |
| freematch@1500 | 36/131 (0.275) | 67/130 (0.515) | 99/130 (0.762) |
| dann@4000 | 36/131 (0.275) | 68/130 (0.523) | 91/130 (0.700) |
| dann@2000 (sens.) | 39/131 (0.298) | 76/130 (0.585) | 102/130 (0.785) |
| coral@2000 | 36/131 (0.275) | 67/130 (0.515) | 101/130 (0.777) |

*Einstein radius* (`einstein_radius_effective_median_pdf`, 245/391; edges 0.0 / 0.7117 / 1.0114 /
4.3089; counts 82 / 81 / 82; two stored values are exactly 0):

| readout | small | mid | large |
|---|---|---|---|
| baseline@8000 | 8/82 (0.098) | 22/81 (0.272) | 24/82 (0.293) |
| lambda0@4000 | 9/82 (0.110) | 28/81 (0.346) | 27/82 (0.329) |
| bnm@4000 | 25/82 (0.305) | 47/81 (0.580) | 48/82 (0.585) |
| freematch@1500 | 39/82 (0.476) | 53/81 (0.654) | 56/82 (0.683) |
| dann@4000 | 38/82 (0.463) | 59/81 (0.728) | 48/82 (0.585) |
| dann@2000 (sens.) | 48/82 (0.585) | 57/81 (0.704) | 57/82 (0.695) |
| coral@2000 | 46/82 (0.561) | 56/81 (0.691) | 55/82 (0.671) |

*Arc brightness* (`vis_lensed_source_magnitude_ab_median_pdf`, 245/391; edges 18.1118 / 21.9493 /
22.5742 / 41.0771; counts 82 / 81 / 82; one stored value > 30; bin 1 = brightest):

| readout | bright | mid | faint |
|---|---|---|---|
| baseline@8000 | 22/82 (0.268) | 15/81 (0.185) | 17/82 (0.207) |
| lambda0@4000 | 25/82 (0.305) | 20/81 (0.247) | 19/82 (0.232) |
| bnm@4000 | 44/82 (0.537) | 37/81 (0.457) | 39/82 (0.476) |
| freematch@1500 | 48/82 (0.585) | 47/81 (0.580) | 53/82 (0.646) |
| dann@4000 | 50/82 (0.610) | 47/81 (0.580) | 48/82 (0.585) |
| dann@2000 (sens.) | 52/82 (0.634) | 56/81 (0.691) | 54/82 (0.659) |
| coral@2000 | 54/82 (0.659) | 48/81 (0.593) | 55/82 (0.671) |

*Arc S/N* (`vis_max_lensed_source_signal_to_noise_ratio`, 245/391; edges 0.0 / 8.6683 / 13.835 /
121.3568; counts 82 / 81 / 82):

| readout | low | mid | high |
|---|---|---|---|
| baseline@8000 | 24/82 (0.293) | 18/81 (0.222) | 12/82 (0.146) |
| lambda0@4000 | 26/82 (0.317) | 23/81 (0.284) | 15/82 (0.183) |
| bnm@4000 | 43/82 (0.524) | 42/81 (0.519) | 35/82 (0.427) |
| freematch@1500 | 54/82 (0.659) | 51/81 (0.630) | 43/82 (0.524) |
| dann@4000 | 50/82 (0.610) | 46/81 (0.568) | 49/82 (0.598) |
| dann@2000 (sens.) | 55/82 (0.671) | 57/81 (0.704) | 50/82 (0.610) |
| coral@2000 | 58/82 (0.707) | 50/81 (0.617) | 49/82 (0.598) |

`out/A3_recall_by_property.json → result."<column>"` (edges, counts, grade mix per bin, recall
with Clopper–Pearson intervals per bin).

**Galleries** (16 each; selection: sort the set by exact Euclid object ID ascending, take the first
16; DA arms = BNM 4000, FreeMatch 1500, DANN 4000, CORAL 2000; "every arm" = those four plus
Baseline A 8000 and λ=0 4000):

| gallery | set size | grade mix of set | file |
|---|---|---|---|
| (a) recovered by all four DA arms | 114 | A 91, B 23 | `fig/lens_a_recovered_by_all_four_DA_arms.png` |
| (b) missed by every arm | 136 | A 37, B 99 | `fig/lens_b_missed_by_every_arm.png` |
| (c) recovered by ≥ 3 DA arms, missed by λ=0 | 105 | A 71, B 34 | `fig/lens_c_recovered_by_ge3_DA_arms_missed_by_lambda0.png` |

`out/A3_galleries.json → result.galleries` (IDs, grades and folds of the 16 shown).

Factual line: summed over folds 1–4, every readout recovers a larger fraction of grade-A than of
grade-B lenses.

---

## 5. Figures produced

- `fig/fig5_detrended_draft.png` — draft replacement of Figure 5: (a) like-for-like ratio against
  k for the six gaps (solid declared budgets, dashed budget-matched); (b) achieved reduction with
  estimator (a) against √k and against k. The existing `paper/figures/F5_resolution.pdf` is untouched.
- `fig/top_neg_page_01.png` … `fig/top_neg_page_06.png` — top-scoring D1 negatives.
- `fig/lens_a_recovered_by_all_four_DA_arms.png`, `fig/lens_b_missed_by_every_arm.png`,
  `fig/lens_c_recovered_by_ge3_DA_arms_missed_by_lambda0.png`.

---

## 6. Not available

- **F3:** the producers of AUROC 0.995 and ≈ 0.84 — not identified (section 3, F3).
- **F5:** stellar mass, photo-z and `mvis` for random-galaxy and grade-X training negatives (no
  stored values); MER VIS PSF flux beyond the objects the stored MER query output contains
  (available for 15,628/24,000 positive hosts, 1,935/2,881 bare deflectors, 1,120/4,455 grade-X;
  `out/F5_host_properties.json → result.availability`); the definition of the deflector table's
  `mvis` (its query is not stored).
- **A1:** Baseline A per-checkpoint series (only step 8000's scores are stored; earlier steps would
  need loading head checkpoints). CORAL's folds 1–4 k = 8 point (one window position per fold).
  DANN 1500 and CORAL 1500 are stored checkpoints but not declared readouts (flagged in the table).
- **A3:** Einstein radius and arc brightness exist for 245 of the 391 folds 1–4 test lenses
  (modelling tables cover 335 of 489 lenses).

---

## 7. Control-character scan

`scripts/ctrl_scan.py` checks every text file under `audit/review_r1/` for bytes < 0x20 other than
`\n`, `\t` and `\r` inside `\r\n`; the PNGs are listed and skipped.

- **One hit, fixed during the work:** a literal backspace (0x08) in `scripts/record_finish.py`. A
  regex `\b` had been turned into 0x08 by a Python patch sent through a shell heredoc, the failure
  the brief warns about. It was replaced with a space, the regex re-checked (51/51 claims counted),
  and the JSON regenerated.
- **Final scan, after this report was finished:** 34 text files scanned (every `.py`, `.json`,
  `.csv` and this `.md` under `audit/review_r1/`), **0 flagged**; 10 PNGs skipped as binary.

`out/CTRL_scan.json`.

---

## 8. Finish: standard tool sequence

Rule zero forbids touching existing files, and several tools write into the repo
(`paper/{tables,figures,data}`, `paper/data/crosscheck.json`, `<repo>/paper_DRAFT.zip`). Every tool
was therefore run with its outputs redirected to scratch copies (the mechanisms the scripts already
provide: `PAPER_TABLES_DIR`, `PAPER_FIGURES_DIR`, `PAPER_DATA_DIR`), and each output was compared
byte-for-byte with the tracked file. `out/FINISH_tools.json` holds every line quoted below.

1. **`make_all.py`** (redirected): rc 0. 51/51 `make_m_prose` claims hold.
   `1314 checks, 0 mismatch(es)`. `ALL TABLES AND FIGURES REBUILT. No number disagrees with PROJECT.md.`
   Of 49 outputs, **43 are byte-identical** to the tracked files (every table, macro file and data
   JSON). **The 6 figure PDFs (F1–F6) differ, only in `/CreationDate`** (tracked 2026-10-04,
   regenerated 2026-10-08; identical once that field is removed).
   **In-place run: skipped.** It would rewrite the six tracked figure PDFs with new timestamps, which
   is not byte-identical reproduction (the brief's condition).
   `out/FINISH_tools.json → result.1_make_all`.
2. **`check_latex.py`**: rc 0. `40 .tex files: inputs and graphics resolve, …`
3. **`crosscheck.py`** (`PAPER_DATA_DIR` → scratch copy): rc 0. `1314 checks, 0 mismatch(es)`.
   `pre-registered pool vs PROJECT.md: 209; D1 / A7 vs the exact-ID audit: 696; A3 window rows + prose macros: 409`.
   The `crosscheck.json` it wrote is byte-identical to the tracked `paper/data/crosscheck.json`.
4. **`audit/decontam/prose_macros.py verify`**: rc 0.
   `192 replaced, 192 verified against the typed original, 0 mismatches; 185 of the 192 still in the text, 7 removed by approved wording`.
5. **`audit/decontam/verify_orig_tables.py`**:
   - `--mode tokens`: rc 0, `10 tables regenerated on the pre-registered pool; tokens: 10 identical, 0 differ`.
   - `--mode bytes`: rc 0, `10 tables regenerated on the pre-registered pool; bytes: 10 identical, 0 differ`.
6. **`test_make_archive.py`**: rc 0, `25/25 passed`.
7. **`integrity.py --verify`**: rc 0, `[integrity] 7 protected artifact(s) verified`.
8. **`make_archive.py --draft`**: run unchanged through
   `scripts/make_archive_draft_redirected.py`, which only points `make_archive.OUT_DRAFT` at the
   scratch dir so the tracked `paper_DRAFT.zip` is not rewritten. rc 0.
   `SUBMISSION GATE -- 8 violation(s)` (**gate 8, unchanged**). The archive self-verified.
   It has the same 48 entries as the tracked `paper_DRAFT.zip`, and 0 entries differ in content.
9. **tectonic 0.17.0** (`ML\_tex_env`), compiling the unzipped scratch archive: rc 0, 52 pages,
   **0 overfull boxes**, 5 underfull boxes, 0 `LaTeX Warning` lines.

Factual line: every tool passes; the paper build did not move, apart from the figure PDFs'
creation timestamps in the redirected rebuild.
