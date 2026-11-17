# Review round 1b — target-label lookup and bright-only pool (read-only)

**Status: post hoc and exploratory.** Every number below is in a JSON under
`audit/review_r1b/out/` carrying `"status": "post_hoc_exploratory"`, the producing script, the input
sha256s, the pool and the folds.

| script (`audit/review_r1b/scripts/`) | output |
|---|---|
| `l1_target_labels.py` | `out/L1_target_labels.json` |
| `l2_bright_pool.py` | `out/L2_bright_pool.json` |
| `record_r1b.py` | `out/SNAPSHOT_diff.json`, `out/FINISH_tools.json` |
| `ctrl_scan_r1b.py` | `out/CTRL_scan.json` |

The round-1 helpers (`rr_common`, `f_facts.cite`, `record_finish.lines`, `snapshot.py`,
`make_archive_draft_redirected.py`) are imported or run unchanged; none was copied or edited.
`record_r1b.py` and `ctrl_scan_r1b.py` exist only because round-1's `snapdiff.py` and
`ctrl_scan.py` hard-code the `audit/review_r1/` folder.

---

## 1. Snapshot diff

sha256 of every file under the repo root, taken with `audit/review_r1/scripts/snapshot.py` before
the folder existed (115,132 files). Comparison by path and sha256, with the prefix
`audit/review_r1b/`:

| comparison | added | added outside `audit/review_r1b/` | removed | content changed | mtime-only |
|---|---|---|---|---|---|
| after L1 + L2 vs before | 4 | 0 | 0 | 0 | 0 |
| after the finish sequence vs before | 7 | 0 | 0 | 0 | 0 |
| final, after this report and the record JSONs vs before | 10 | 0 | 0 | 0 | 0 |

**No pre-existing file changed.** `out/SNAPSHOT_diff.json → result.diffs.snap_mid_vs_before`,
`snap_final_vs_before`, `snap_end_vs_before`. The last row was measured after everything else was
written. Only `SNAPSHOT_diff.json`, `FINISH_tools.json`, `CTRL_scan.json` (rewritten by the
recorder and scanner) and this row postdate it, all inside `audit/review_r1b/`.

---

## 2. L1 — does any arm use target labels during training?

**Shared mechanism** (every runner below reaches it). Target items are served with label −1
(`scripts/lens_dataset.py:298`, `:302`: `self._add("real_lens", UNLABELED, TARGET, 1,` /
`self._add("random", UNLABELED, TARGET, 0,`). Their true lens/non-lens identity is kept in
`latent_labels` (`:228`) and grouped by `indices_by_latent` (`:235`). The docstring at `:35` describes
these as "for DIAGNOSTICS ONLY". The shared sampler balances on it:

- `scripts/samplers.py:23`: "we balance on ``indices_by_latent`` (latent identity {0,1})".
- `:177`: `index_map = dataset.indices_by_latent if use_latent else dataset.indices_by_label`.
- `:186`: `def make_balanced_loader(`.

The DANN/λ=0/BNM runners build their own sampler from a latent-stratified split,
`scripts/dann_round5.py:341` `def build_target_holdout(target_ds`. It reads `by_latent =
target_ds.indices_by_latent` (`:371`), is documented as "Stratified by LATENT class" (`:361`), and
removes the held-out slice from the training pool at `:386`. Every Kaggle-bundle copy of
`samplers.py`, `lens_dataset.py`, `dann_round5.py`, `run_freematch.py`, `run_coral_round4.py`,
`bnm_round6.py`, `run_bnm_r6.py` and `run_dann_r5.py` is byte-identical to `scripts/`
(`out/L1_target_labels.json → result.kaggle_bundle_copies_identical_to_scripts`).

Mode key: (a) uniform from the target pool, ignoring labels; (b) class-balanced/stratified on
lens/non-lens identity; (c) something else.

| arm | sampling | target labels read during training, beyond the sampler | target labels read for diagnostics only |
|---|---|---|---|
| CORAL × nb3 | **(b)** `run_coral_r4_confirm.py:575` `A = R4.build_arm(...)` → `run_coral_round4.py:513` `tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True,` | none found. The step-0 gate (`run_coral_round4.py:446`) uses the test fold; the memorisation probe (`unfreeze_instrument.py:180`) reads target scores, not labels | none |
| FreeMatch × nb3 | **(b)** `run_unfreeze_freematch.py:610` `from run_freematch import train_freematch_fold` → `run_freematch.py:224` `use_latent=True` | **yes, thresholds:** the SAT buffer rebuild (`continuation_state.py:83` `def collect_target_batch_stats(trainer, target_loader`) averages over batches from the same latent-balanced loader (`:86`). The SAF class marginal is the constant uniform [0.5, 0.5] (`da_losses.py:923`); it reads no label | none |
| DANN × nb3 | **(b)** `run_dann_r5.py:1043` `holdout = build_target_holdout(...)`, `:1062` `tgt_sampler = ClassBalancedBatchSampler(holdout["train_pos"], holdout["train_neg"],` | **yes, pool composition:** a latent-stratified ~10% slice (78 images) is removed from the training pool (`dann_round5.py:371`). Abort condition 1, armed on DANN's folds, reads D's *domain* accuracy on that slice (`dann_round5.py:461`, `run_dann_r5.py:1121`); it reads domain labels, not lens labels. Condition 3 uses an ML-FP vs P1-reserve pool, not target labels (`dann_round5.py:614`, `:877`) | none |
| λ = 0 | **(b)** same code path, `run_dann_r5.py:1062` | **yes, pool composition:** the same latent-stratified holdout (`dann_round5.py:371`); aborts are log-only on this arm | none |
| BNM × nb3 | **(b)** `run_bnm_r6.py:335` `holdout = build_target_holdout(...)`, `:350` `tgt_sampler = ClassBalancedBatchSampler(holdout["train_pos"], holdout["train_neg"],` | **yes, pool composition:** the same latent-stratified holdout (`dann_round5.py:371`) | **yes:** a latent-stratified "trained-on" probe (`bnm_round6.py:658` `def build_trainedon_probe(...)`, `:668`; built at `run_bnm_r6.py:432`) whose nuclear norm, entropy and predicted-positive fraction are logged (the §8 memorisation diagnostic) |
| round-1 CORAL (frozen) | **(b)** `run_bakeoff_coral.py:60` → `run_coral.py:245` `tgt_latent = np.asarray(tgt["latent_labels"], ...)   # 0/1 identity (balancing only)`, `:250` `tpos_idx = np.where(tgt_latent == POS_LABEL)[0]` | none found; the blow-up stop (`run_coral.py:205`) reads the covariance term | none |
| round-1 AdaMatch (frozen) | **(b)** `run_bakeoff_adamatch.py:61` → `run_adamatch.py:129` `use_latent=True` | none found | none |
| round-1 FreeMatch (frozen) | **(b)** `run_bakeoff_freematch.py:65` → `run_freematch.py:224` `use_latent=True` | SAF marginal = constant uniform (`da_losses.py:923`); no label read | none |
| round-1 DANN probe (fold 0) | **(b)** `run_dann.py:384` `tgt_latent = np.asarray(tgt["latent_labels"], ...)   # 0/1 identity (balancing)`, `:389` | none found | none |

**Diagnostic or post-hoc reads outside any training loop** (no effect on gradients or on which
steps run): `cache_features.py:28` (stores `latent_labels` in the feature cache; the frozen round-1
arms take their sampler input from it), `live_encode.py:162`, `study_validators.py:244`,
`validators.py:171`, `pseudolabel_precision.py:63`. Excluded as the brief directs: evaluation on the
held-out test fold (step-0 gates, curve scoring, collapse gate).

`out/L1_target_labels.json → result.arms.<arm>` (`sampling`, `other_training_reads`,
`diagnostic_only_reads`, `checked_and_label_free`), `result.shared_mechanism`,
`result.post_hoc_only_reads`.

Factual line: all nine arms draw target batches class-balanced on lens/non-lens identity, at
`pos_fraction` 0.5.

---

## 3. L2 — bright-only negative pool

**Pool.** The D1 test negatives with VIS PSF AB mag ≤ 22.5 (mag = 23.9 − 2.5 log₁₀
`flux_vis_psf`[µJy], the §3 and round-1 A2 definition; the complement of "fainter than 22.5").
**14,950 of 26,134** remain and 11,184 are removed. The D1 faint fraction 0.4279 reproduces
`make_m_prose.test_faint_fraction` exactly. Positives are unchanged.
`out/L2_bright_pool.json → result.pool`.

**TPR at FPR = 1e-3, folds 1–4** (D1 values bit-equal to `_common.suite`):

| readout | f1 | f2 | f3 | f4 | mean | sd | D1 mean | change |
|---|---|---|---|---|---|---|---|---|
| baseline@8000 | 0.1429 | 0.2143 | 0.1633 | 0.2062 | 0.1816 | 0.0342 | 0.1842 | −0.0026 |
| lambda0@4000 | 0.2245 | 0.1429 | 0.1122 | 0.1031 | 0.1457 | 0.0552 | 0.1842 | −0.0385 |
| bnm@4000 | 0.3673 | 0.4388 | 0.3367 | 0.3814 | 0.3811 | 0.0428 | 0.4040 | −0.0230 |
| freematch@1500 | 0.4796 | 0.5510 | 0.3776 | 0.5361 | 0.4861 | 0.0786 | 0.5167 | −0.0306 |
| dann@4000 | 0.5000 | 0.7041 | 0.2857 | 0.3608 | 0.4627 | 0.1838 | 0.4985 | −0.0359 |
| dann@2000 (sens.) | 0.5408 | 0.7041 | 0.4694 | 0.4227 | 0.5342 | 0.1232 | 0.5547 | −0.0205 |
| coral@2000 | 0.4694 | 0.6122 | 0.5000 | 0.4742 | 0.5140 | 0.0669 | 0.5216 | −0.0077 |

`out/L2_bright_pool.json → result.readouts.<readout>.tpr_bright`, `tpr_D1`, `tpr_change_mean`
(thresholds in `thresholds_bright`).

**Paired contrasts** (paired t, df = 3, no multiplicity correction; D1 values equal
`_common.contrast`):

| contrast | steps | form | bright mean | bright 95% CI | folds same sign | D1 mean | D1 95% CI | change | includes 0 (bright / D1) | same |
|---|---|---|---|---|---|---|---|---|---|---|
| DANN − λ=0 | 4000/4000 | pre-registered = budget-matched | +0.3170 | [+0.0484, +0.5856] | 4/4 | +0.3144 | [+0.0565, +0.5722] | +0.0026 | no / no | yes |
| CORAL − λ=0 | 2000/4000 | pre-registered | +0.3683 | [+0.2206, +0.5160] | 4/4 | +0.3375 | [+0.1889, +0.4860] | +0.0308 | no / no | yes |
| CORAL − λ=0 | 2000/2000 | budget-matched | +0.3399 | [+0.1893, +0.4906] | 4/4 | +0.3068 | [+0.1834, +0.4301] | +0.0332 | no / no | yes |
| FreeMatch − λ=0 | 1500/4000 | pre-registered | +0.3404 | [+0.1920, +0.4888] | 4/4 | +0.3325 | [+0.2272, +0.4378] | +0.0079 | no / no | yes |
| FreeMatch − λ=0 | 1500/1500 | budget-matched | +0.3276 | [+0.1768, +0.4784] | 4/4 | +0.3223 | [+0.1984, +0.4463] | +0.0053 | no / no | yes |
| BNM − λ=0 | 4000/4000 | pre-registered = budget-matched | +0.2354 | [+0.1260, +0.3448] | 4/4 | +0.2199 | [+0.1383, +0.3014] | +0.0155 | no / no | yes |
| BNM − CORAL | 4000/2000 | declared budgets | −0.1329 | [−0.1987, −0.0671] | 4/4 | −0.1176 | [−0.1857, −0.0496] | −0.0153 | no / no | yes |
| BNM − CORAL | 2000/2000 | budget-matched | −0.1329 | [−0.1870, −0.0788] | 4/4 | −0.1176 | [−0.1772, −0.0580] | −0.0153 | no / no | yes |

Step pairs: DANN, CORAL and FreeMatch − λ=0 as in round 1 F1 (`runs/lambda0_r6_prereg.json`). BNM −
λ=0 at 4000/4000 per `runs/bnm_r6_prereg.json → budget.why_4000`; it is not one of λ=0's four. BNM −
CORAL in both forms of `_common.CONTRAST_ROWS`. `out/L2_bright_pool.json → result.contrasts[i]`
(`bright`, `D1`, `includes_zero`, `zero_inclusion_same`, `change_mean`).

**Completeness at 24% prevalence-corrected purity** (`metric_suite`'s `comp_at_purity`; n/a rule:
any undefined fold → no mean and no ratio):

| readout | f1 | f2 | f3 | f4 | mean (bright) | ratio to 0.50 (bright) | mean (D1) | ratio (D1) |
|---|---|---|---|---|---|---|---|---|
| baseline@8000 | 0.1224 | 0.0714 | 0.1020 | 0.0928 | 0.0972 | 0.194 | 0.0921 | 0.184 |
| lambda0@4000 | 0.0714 | n/a | n/a | 0.0103 | n/a | n/a | n/a | n/a |
| bnm@4000 | 0.1939 | 0.2857 | 0.2551 | 0.3608 | 0.2739 | 0.548 | 0.3555 | 0.711 |
| freematch@1500 | 0.4592 | 0.5510 | 0.3265 | 0.5258 | 0.4656 | 0.931 | 0.4835 | 0.967 |
| dann@4000 | 0.4286 | 0.7143 | 0.2245 | 0.3093 | 0.4192 | 0.838 | 0.4550 | 0.910 |
| dann@2000 (sens.) | 0.5408 | 0.7041 | 0.3878 | 0.4124 | 0.5113 | 1.023 | 0.5291 | 1.058 |
| coral@2000 | 0.3571 | 0.6122 | 0.4796 | 0.4330 | 0.4705 | 0.941 | 0.4961 | 0.992 |

`out/L2_bright_pool.json → result.readouts.<readout>.comp_at_purity_bright`, `comp_at_purity_D1`,
`lines_ratio_bright`, `lines_ratio_D1`.

Factual line: on the bright pool every DA-arm TPR falls by 0.008–0.036. All eight intervals exclude
zero, as they do on D1.

---

## 4. Control-character scan

`scripts/ctrl_scan_r1b.py` checks every text file under `audit/review_r1b/` for bytes < 0x20
other than `\n`, `\t`, and `\r` inside `\r\n`. Every file was written with the Write/Edit tools; no
file was patched through a shell. **Final scan: 10 text files, 0 flagged.**
`out/CTRL_scan.json`.

---

## 5. Finish sequence

Same approach as round 1. Every tool ran with `PYTHONDONTWRITEBYTECODE=1` and its outputs
redirected to scratch copies (`PAPER_TABLES_DIR`, `PAPER_FIGURES_DIR`, `PAPER_DATA_DIR`; the
archive through round-1's wrapper). Nothing was written into `paper/` or to `paper_DRAFT.zip`.
`out/FINISH_tools.json` holds every line below.

1. **`make_all.py`** (redirected): rc 0; 51/51 claims hold; `1314 checks, 0 mismatch(es)`;
   `ALL TABLES AND FIGURES REBUILT. No number disagrees with PROJECT.md.` Of 49 outputs, 43 are
   byte-identical. **The 6 figure PDFs (F1–F6) differ only in `/CreationDate`**, so they count as
   identical under this brief's rule.
2. **`check_latex.py`**: rc 0, `40 .tex files: inputs and graphics resolve, …`.
3. **`crosscheck.py`** (redirected data dir): rc 0, `1314 checks, 0 mismatch(es)`;
   `pre-registered pool vs PROJECT.md: 209; D1 / A7 vs the exact-ID audit: 696; A3 window rows + prose macros: 409`.
   Its `crosscheck.json` is byte-identical to the tracked file.
4. **`prose_macros.py verify`**: rc 0,
   `192 replaced, 192 verified against the typed original, 0 mismatches; 185 of the 192 still in the text, 7 removed by approved wording`.
5. **`verify_orig_tables.py`**:
   - tokens: rc 0, `10 tables regenerated on the pre-registered pool; tokens: 10 identical, 0 differ`.
   - bytes: rc 0, `… bytes: 10 identical, 0 differ`.
6. **`test_make_archive.py`**: rc 0, `25/25 passed`.
7. **`integrity.py --verify`**: rc 0, `[integrity] 7 protected artifact(s) verified`.
8. **`make_archive.py --draft`** (via `audit/review_r1/scripts/make_archive_draft_redirected.py`):
   rc 0, `SUBMISSION GATE -- 8 violation(s)` (**gate 8**). The archive has the same 48 entries as the
   tracked `paper_DRAFT.zip`, and 0 entries differ in content.
9. **tectonic** (`ML\_tex_env`) on the unzipped scratch archive: rc 0, 52 pages, **0 overfull
   boxes**, 5 underfull, 0 `LaTeX Warning`.

Factual line: every tool passes; the paper build is unchanged.
