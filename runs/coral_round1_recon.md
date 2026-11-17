# Round-1 CORAL — reconnaissance

Read-only. No file was modified; no training code was written. Everything below is quoted
from the repo as it stands.

**The run under examination:** `runs/bakeoff_coral/` — the matched 5-fold bake-off run
(PROJECT.md §7.4, run 2/4), fold-mean TPR@FPR=1e-3 **0.1923 ± 0.0247**, per fold
0.2143 / 0.1735 / 0.2143 / 0.1531 / 0.2062, R1 = `step_8000.pt` in every fold.

**Code path:** `scripts/run_bakeoff_coral.py` (a thin driver that pins the matched protocol)
→ `run_coral.train_coral_fold` (all the CORAL math) → `da_losses.CoralLoss`.
`runs/coral/` and `runs/coral_warmup_cosine/` are the earlier epoch-schedule runs from the
same `train_coral_fold`; the bake-off driver only fixes the knobs and turns on the per-step
curve hook.

---

## 1. Where the CORAL loss attaches

`scripts/run_coral.py:333-336` — the whole attachment, verbatim:

```python
src_logits, src_adapter = head.forward_with_features(x_src[sidx])
_, tgt_adapter = head.forward_with_features(x_tgt[tidx])
sup_loss = bce(src_logits, y[sidx])
da_loss = gamma_eff * coral(src_adapter, tgt_adapter)
```

`forward_with_features` is `scripts/heads.py:101-103`:

```python
def forward_with_features(self, x: torch.Tensor):
    f = self.trunk(x)
    return self.classifier(f), f                   # (logits, adapter_features)
```

and for the `linear` head that round 1 used (`heads.py:78-80`):

```python
if head_type == LINEAR:
    self.trunk = nn.Sequential(nn.LayerNorm(in_dim))
    feat_dim = in_dim
```

### The answer, and it is the single most consequential finding here

The CORAL loss is computed on the **adapter output, which at `head_type="linear"` is
640-dimensional** — but the "adapter" is nothing but a `LayerNorm(640)`. There is no
projection. The tensor CORAL sees is:

```
frozen Zoobot 640-d feature
  → FeatureStandardizer  (per-dimension (x−μ)/σ, μ/σ fitted ONCE on source_train, FROZEN,
                          zero trainable parameters — heads.py:119-161)
  → nn.LayerNorm(640)    (per-SAMPLE normalisation across the 640 dims, then a trainable
                          per-dimension gain γ and shift β)
  → this is `src_adapter` / `tgt_adapter`
```

So it is **not** the 640-d frozen Zoobot output raw, and it is **not** a learned bottleneck.
It is the frozen output put through one frozen affine rescale and one trainable
diagonal affine.

Verified against the artifact — `runs/bakeoff_coral/fold0/checkpoints/step_8000.pt`:

| tensor | shape |
|---|---|
| `trunk.0.weight` | (640,) |
| `trunk.0.bias` | (640,) |
| `classifier.weight` | (1, 640) |
| `classifier.bias` | (1,) |

**Total trainable parameters: 1,921.** (This is the same 1,921 PROJECT.md §7.3 quotes.)
Of those, **1,280 — the LayerNorm gain and shift — are the *only* parameters CORAL can
move**; the remaining 641 are the classifier, which CORAL does not touch directly (it
receives gradient only through the shared BCE term).

**Consequence for specifying the unfrozen version.** Round-1 CORAL was asked to align two
640-d covariance matrices using a **per-dimension diagonal gain and shift**. That
transformation cannot rotate the feature space, cannot mix dimensions, and therefore cannot
change any off-diagonal correlation structure except by rescaling rows and columns —
γᵢγⱼ·Cov(i,j) up to the LayerNorm's per-sample centring. The objective is
`||Σ_s − Σ_t||²_F`, a *full* matrix distance, over a parameterisation that only reaches a
diagonal similarity transform of it. The alignment problem posed was largely unsolvable by
the model given to solve it. That is a much better explanation of "CORAL sits at baseline"
than "marginal alignment doesn't work here", and it is *not* what §7.4 currently concludes —
so it is worth stating explicitly before the unfrozen variant is specified, because the
unfrozen variant removes exactly this constraint.

**Also confirmed:** the encoder cannot receive CORAL gradient in round 1. The features
arrive already detached (they are read from the cache as constants,
`run_coral.py:239` `x_src = standardizer(src_feats).detach()`), and `CoralLoss.forward`
asserts the adapter features carry grad specifically to catch the reverse mistake
(`da_losses.py`, "Head-only-graph guard").

### The loss itself (`da_losses.CoralLoss`)

- First + second moment (DomainBed form). Covariance is `(zᵀz − (1ᵀz)ᵀ(1ᵀz)/n)/(n−1)`,
  i.e. **640×640, estimated from 256 samples** → rank ≤ 255, structurally rank-deficient.
- Shrinkage `Σ ← (1−λ)Σ + λ·(tr Σ/d)·I + jitter·I` with **λ = 0.1**, jitter 1e-6, applied to
  *both* domains before the distance.
- `cov_term = (src_cov − tgt_cov).pow(2).mean()` = `||·||²_F / d²`, and
  `mean_term = (src_mean − tgt_mean).pow(2).mean()` = `||·||² / d`. Note the canonical
  Sun & Saenko normalisation is `1/(4d²)`; this is `1/d²`. Fixed constant, stated here
  because it changes the effective scale of γ by 4×.
- **Target statistics are EMA'd**: `tgt_ref = 0.9·history(detached) + 0.1·current(grad)`.
  So 90% of the alignment reference on any step is a frozen constant and only 10% of it
  carries gradient. With a 782-image target pool this is a deliberate variance reduction,
  but it also means the per-step gradient signal through the target branch is attenuated 10×
  relative to the source branch.

---

## 2. The CORAL loss weight (lambda / gamma)

**Peak γ = 1.0**, and it is *not* constant across the run.

Set in three places, all consistent for the bake-off run:

- `run_bakeoff_coral.py:75-82` — the pre-registered block:
  ```python
  CORAL_PEAK_GAMMA = 1.0            # FIXED peak, not tuned on target (RLSbench)
  CORAL_SCHEDULE = "warmup_cosine"  # the blow-up fix -- ON (SPEC §1 / constraint)
  CORAL_WARMUP_FRAC = 0.1
  CORAL_MIN_FRAC = 0.1
  CORAL_SHRINKAGE = 0.1
  ```
- `run_coral.py:722` — the CLI default (`--coral-weight`, default 1.0), whose help string is
  `"gamma -- FIXED, not tuned on target (RLSbench fixed-budget)"`.
- `runs/bakeoff_coral/run_config.json` and the checkpoint's own `cfg` both record
  `coral_peak_gamma: 1.0`, `coral_weight_schedule: "warmup_cosine"`, `coral_warmup_frac: 0.1`,
  `coral_min_frac: 0.1`.

The **effective** weight is `coral_gamma_at_step` (`run_coral.py:105-131`): linear ramp
0 → 1.0 over the first 10% of the *planned* total steps, then cosine decay 1.0 → 0.1 over
the remainder. Planned total = `steps_per_epoch × max_epoch` = 188 × 43 = **8,084**, so:

- warmup runs to step ≈ 808 — **γ is below 1.0 for the first ~10% of training and is
  exactly 0 at step 0**;
- by the R1 read-off at step 8000, γ has decayed to ≈ **0.100**.

So the model that produced the reported 0.1923 was, at the moment it was read, being
trained under a CORAL weight one tenth of its nominal value. This was a deliberate stability
fix (`run_coral.py:79-100` documents it as a response to the late-training Frobenius blow-up)
and the schedule *shape* is declared fixed-not-tuned — but for specifying the unfrozen
version it matters that "λ = 1.0" describes a peak that the run spends almost none of its
time at.

`stop_on_blowup=False` in the bake-off (`run_bakeoff_coral.py:83`), so the `BlowupDetector`
only logged; training was never halted.

---

## 3. Batch composition

**Batch size 256** for both domains (`run_bakeoff_coral.py:66`, and
`run_config.json: batch_size 256`). **Steps per epoch = 188**, 43 epochs, R1 at step 8000.

Source and target are drawn by **two independent `ClassBalancedBatchSampler`s with equal
batch counts**, consumed by `zip` — not by `PairedDomainLoader` (`run_coral.py:262-271, 322`):

```python
src_sampler = ClassBalancedBatchSampler(
    pos_idx, neg_idx, batch_size=batch_size, num_batches=steps_per_epoch,
    pos_fraction=pos_fraction, seed=seed + fold)
steps = len(src_sampler)
tgt_sampler = ClassBalancedBatchSampler(
    tpos_idx, tneg_idx, batch_size=batch_size, num_batches=steps,
    pos_fraction=pos_fraction, seed=seed + fold + 1000)
...
for sbatch, tbatch in zip(src_sampler, tgt_sampler):
```

**Is the source batch class-balanced? Yes — exactly, every step.** See §8 below for the
proof; at batch 256 / `pos_fraction=0.5` it is exactly 128 sim-lens positives + 128 negatives
in every batch.

**The target batch is also exactly balanced — on latent identity.** `tpos_idx`/`tneg_idx`
come from `tgt["latent_labels"]` (`run_coral.py:250-251`), the true 0/1 identity of the
unlabeled target rows. Measured on `data/feature_cache/target_fold0.npz`: served `labels`
are all `-1` (UNLABELED, and `assert_target_unlabeled` re-checks this every step), while
`latent_labels` are 391 / 391. So each covariance step sees **exactly 128 real lenses and
128 random galaxies** on the target side.

> **Flag, stated neutrally because it bears on the label-free claim.** The target *pool* is
> ~1:1 by construction (PROJECT.md §4.3), so random sampling would give ≈1:1 in
> expectation. But the sampler does not sample randomly — it uses the latent lens/non-lens
> identity to force the ratio to be *exact* in every mini-batch. That identity never enters
> the loss, and the effect is small given an already-balanced pool, but it is a use of
> target identity information at sampling time and should be disclosed as such rather than
> described as "the pool happens to be balanced".

**Target reuse is heavy.** The pool is 782 rows; 188 batches × 256 = 48,128 target draws per
epoch, so each target image is seen ~62× per epoch and ~2,650× over 8000 steps. The
`_CyclingPool` reshuffles and cycles whenever a class pool is exhausted, so this is sampling
with replacement across the epoch. The covariance on the target side is therefore computed
from 256 draws out of only 391 distinct images per class.

---

## 4. Data path, precision, augmentation

- **Cached features, not image-space.** `run_coral.py` never constructs an encoder and never
  imports `ZoobotEncoder` or `LensDataset` (beyond the `UNLABELED` constant). It reads
  `data/feature_cache/{source_train,val,target_fold{k}}.npz` through a guarded loader that
  *refuses* any other split (`_train_allowed`, `_guarded_load`, `run_coral.py:139-152`).
- **fp32 throughout.** Cache arrays are `float32` (verified: `source_train` (48000, 640)
  float32, `target_fold0` (782, 640) float32). The training loop has **no `autocast`, no
  `GradScaler`, no `--fp16` flag** — `loss.backward(); opt.step()` in plain precision.
  This differs from AdaMatch/FreeMatch, which run `--fp16` because the encoder is in the loop.
- **Augmentation OFF.** The cache was built with `LensDataset(aug="none")` —
  `cache_features.py`: *"every split is built with `LensDataset(aug="none")` … No
  augmentation RNG touches the cache"*. There is no augmentation anywhere downstream,
  because there are no images downstream. So round-1 CORAL is the `aug=none, fp32` cell of
  §7.3's 2×2 (cell A) — the *best* cell of that table, worth noting when comparing against
  the image-space methods which paid the measured −0.0714.
- The standardizer is fit once on `source_train` and applied unchanged to the target
  (`run_coral.py:236-249`), so both domains share one frozen normalisation.

---

## 5. Optimizer, LR, schedule

`run_coral.py:255`:

```python
opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
```

- **AdamW**, `lr = 1e-3`, `weight_decay = 1e-4`, betas at torch defaults (0.9, 0.999).
- **Parameter set = `head.parameters()` only** — 1,921 params. The `FeatureStandardizer` is
  buffers-only, and `CoralLoss` owns only EMA buffers, so neither is (or could be) in the
  optimizer.
- **No learning-rate scheduler of any kind.** None is constructed and none is stepped —
  consistent with §7.10's independently verified statement that there is no LR scheduler
  anywhere in this training path. The LR is flat 1e-3 for all 8000 steps.
- **The only schedule in the run is the CORAL γ schedule** described in §2.
- Seed 42, decorrelated per fold via `set_seed(seed + fold)`.

---

## 6. Can the round-1 trainer run image-space?

**`run_coral.train_coral_fold` cannot — it is cached-feature-only.** It has no `encoder`
argument, no `n_blocks`, no `backbone_lr`, no `instrument`, no live-encode path, and its
inputs are the cached feature tensors held in memory as constants.

**But a working image-space CORAL path already exists elsewhere and would not need to be
written from scratch.** `train_skeleton.Trainer.train_step` (`train_skeleton.py:507-546`)
encodes *images* for both domains and applies the identical `CoralLoss` to the identical
adapter tensors:

```python
src_feat = self.standardizer(
    encode_features(self.encoder, source_batch["image"].to(self.device),
                    self.cfg.encoder_mode))
tgt_feat = self.standardizer(
    encode_features(self.encoder, target_batch["image"].to(self.device),
                    self.cfg.encoder_mode))
src_logits, src_adapter = self.head.forward_with_features(src_feat)
_, tgt_adapter = self.head.forward_with_features(tgt_feat)
...
if self.cfg.use_coral:
    da_loss = self.cfg.coral_weight * self.coral(src_adapter, tgt_adapter)
```

and `TrainConfig` already carries `use_coral`, `coral_weight`, `coral_include_mean`,
`coral_shrinkage`, `coral_target_ema`, `coral_ema_momentum` **alongside** `n_blocks`,
`backbone_lr`, `lr_decay`, `grad_checkpointing` and `init_head_from`. The
`Trainer._build_optimizer` layer-decayed LR groups and the LP→FT head init are in that same
class and are method-agnostic.

**Three gaps between "exists" and "usable", all real:**

1. **No γ schedule on that path.** `train_skeleton` multiplies by the constant
   `cfg.coral_weight`. `coral_gamma_at_step` lives in `run_coral.py` and is not wired in.
   Round-1 CORAL's stability fix would silently not apply.
2. **No runner.** There is no `run_bakeoff_coral`-equivalent that drives the image-space
   path — no per-fold driver, no curve hook, no R1 enforcement, no gates.
3. **`encode_features` explicitly flags the interaction as an open decision**
   (`train_skeleton.py:89-93`): at `n_blocks>0` the encoder features carry grad, so a DA loss
   on adapter features *will* backprop into the encoder. For FreeMatch that was reasoned
   through and declared correct (`run_freematch.py:140-146`). **For CORAL it has never been
   reasoned through**, and it is exactly the semantics change the unfrozen variant is about:
   CORAL would go from moving 1,280 diagonal parameters to reshaping the 14.8M-parameter
   feature extractor that produces the covariances it is trying to match.

So: **not a new path, but a new runner plus one wiring decision and one unexamined semantic
change.**

---

## 7. How much of `run_unfreeze_freematch.py` is method-agnostic?

Short answer: **most of the unfreezing machinery you listed is not in that file at all** —
it is in `train_skeleton.Trainer` and `zoobot_encoder.ZoobotEncoder`, which are already
method-agnostic and already reachable from a CORAL config. What
`run_unfreeze_freematch.py` (1,031 lines) contributes is the *run protocol*, and that is
roughly **70–75% reusable with only a swapped constant block**.

### Where each item on your list actually lives

| machinery | lives in | method-agnostic? |
|---|---|---|
| LP→FT head init | `train_skeleton.Trainer.__init__` (`cfg.init_head_from`, +provenance ‖head‖₂ log) | **fully** |
| layer-wise LR (`backbone_lr · lr_decay^i`) | `train_skeleton.Trainer._build_optimizer` | **fully** |
| the unfreeze itself + grad checkpointing | `zoobot_encoder.ZoobotEncoder(n_blocks=…)` | **fully** |
| step-0 gate | `run_unfreeze_freematch.check_step0_gate` / `gate_or_abort` (75 lines) | mechanism yes; **the `FOLD_GATE` value table is FreeMatch's** |
| checkpointing every 500 | `expected_checkpoint_steps` + `assert_checkpoint_retention` (55 lines) | **fully** |
| logit dumps | `unfreeze_instrument.dump_logits` (35 lines) + `CurveInstrument` | **fully** |
| agreement logging | `curve_hook` + `run_bakeoff_coral.compute_agreement_refs` — note `run_unfreeze_freematch.py:582` **already imports it from the CORAL runner** | **fully** |
| collapse gate | `run_collapse_gate` (91 lines) → `collapse_gate.py` | **fully** |
| bundle packaging / R1 eval | `evaluate_r1` (50 lines) | **fully** |
| confound pinning | `check_preconditions` (52 lines) | mechanism yes; **8 of the ~30 `PINNED_KEYS` are FreeMatch knobs** |
| memorization probe | `unfreeze_instrument.memorization_row` (37 lines) | **fully** (it is sigmoid-of-logit on cached splits) |

### What is genuinely FreeMatch-only

- The constants block (`MU_MAX`, `SAF_WEIGHT`, `SAT_EMA_MOMENTUM`,
  `CONSISTENCY_WARMUP_FRAC`, `FREEMATCH_LOCAL_MODE`, `POS_CLASS`) and the ~13 config keys
  built from it in `build_config` (89 lines, of which ~20 are FreeMatch-specific).
- `FOLD_GATE` — five FreeMatch round-1 TPR/‖head‖₂/AUROC triples. A CORAL variant needs its
  own table read off `runs/bakeoff_coral/fold{k}/checkpoints/step_8000.pt`; the per-fold
  R1 TPRs are already known (0.2143 / 0.1735 / 0.2143 / 0.1531 / 0.2062) but ‖head‖₂ and
  AUROC would have to be re-derived.
- The `train_freematch_fold(...)` call in `train_one` — ~35 keyword arguments, most of them
  FreeMatch knobs.
- **The continuation-state reconstruction** (`mu_step_offset`, `rebuild_sat`, `sat_passes`,
  `sat_ema_iters`, `adam_warmup_steps`, and all of `continuation_state.py`). The *concept*
  transfers exactly — round-1 CORAL checkpoints are the same four-key weights-only format
  (verified above: `['head_type','head','standardizer','cfg']`), so they are among §7.10's
  "17 of 22 legacy" runs and a continuation from one would restart training state. But
  CORAL's state is a **different set of objects**: `CoralLoss.ema_mean` (640,),
  `CoralLoss.ema_cov` (640,640), `ema_ready`, plus the γ-schedule position and AdamW's
  moments. The FreeMatch SAT-rebuild code does not apply; the AdamW moment warm-up does.
  Worth noting the CORAL EMA reconstruction is *easier* than FreeMatch's — with weights
  held fixed, the EMA fixed point of the target mean/cov is just the pool mean/cov over the
  782 target rows, computable in one pass with no sampling.
- `unfreeze_instrument.STEP_KEYS` / `DIAG_COLUMNS` — all 20 per-step diagnostic columns are
  FreeMatch internals (`tau_global`, `local_pos/neg`, `pbar_*`, `acceptance`, `mask_rate_*`,
  `saf`, `mu_eff`). A CORAL variant would log `cov_term`, `mean_term`, `gamma_eff` instead —
  `StepWindow`'s accumulate/mean/last mechanics (60 lines) carry over unchanged, only the
  key list changes. `MEMO_COLUMNS` and the logit dump carry over untouched.

### The honest bottom line

A CORAL × unfreeze variant does not need new unfreezing machinery. It needs:
(a) a runner that drives `train_skeleton`'s existing image-space CORAL step instead of
`run_coral`'s cached loop; (b) the γ schedule wired into that path; (c) a CORAL `FOLD_GATE`
table; (d) `PINNED_KEYS`/`STEP_KEYS` edited to CORAL's knobs; (e) a CORAL continuation-state
reconstruction, which is the genuinely new code and is smaller than FreeMatch's was.
Everything else — gates, retention, logit dumps, agreement, memorization probe, collapse
gate, R1 eval, packaging — is reusable as written.

**One thing that is not a code question but will decide the result.** §7.14 exists because a
budget derived from FreeMatch's decay may not fit CORAL. Two independent reasons CORAL's
curve may look nothing like FreeMatch's once unfrozen: γ is *decaying* to 0.1 by the end of
a run (so the DA pressure falls off by construction, unlike SAT which saturates *upward*),
and the loss now backprops into the encoder for the first time. Neither is a reason to
prejudge the budget — they are reasons the fold-0 development run cannot be skipped.

---

## 8. `samplers.py` — the load-bearing Arazo claim, answered exactly

**Question:** does the class-balanced source sampler *guarantee* a fixed number of labeled
samples in every mini-batch, or does it only balance in expectation?

**Answer: it is a HARD GUARANTEE. It is not weighted random sampling, and it is not balance
in expectation.** PROJECT.md §7.12's `(UNVERIFIED …)` caveat can be discharged.

The count is fixed at construction (`samplers.py:134-135`):

```python
# at least one of each class so every batch is genuinely two-class
self.n_pos = int(min(batch_size - 1, max(1, round(pos_fraction * batch_size))))
self.n_neg = self.batch_size - self.n_pos
```

and every batch is assembled by taking exactly those counts (`samplers.py:159-163`):

```python
for _ in range(self.num_batches):
    batch = np.array(pos_pool.take(self.n_pos) + neg_pool.take(self.n_neg),
                     dtype=np.int64)
    rng.shuffle(batch)                        # interleave pos/neg within the batch
    yield batch.tolist()
```

`_CyclingPool.take(k)` is contractually total — it returns exactly `k` indices or raises
(`samplers.py:70-85`). There is no path that returns fewer.

At the round-1 CORAL setting (batch 256, `pos_fraction=0.5`): **`n_pos = 128`, `n_neg = 128`,
every batch, every epoch.**

### `drop_last`

**There is no `drop_last` anywhere, and it cannot arise.** `ClassBalancedBatchSampler` is a
`batch_sampler` (`make_balanced_loader` passes it as `batch_sampler=`, which makes
`batch_size`/`shuffle`/`sampler`/`drop_last` DataLoader arguments inapplicable). Batches are
*constructed* to size rather than carved out of a permutation, so a short final batch is not
representable: `__iter__` runs `for _ in range(self.num_batches)` and each iteration emits
exactly `n_pos + n_neg = batch_size` indices. There is no ragged tail to drop.

### What happens when the source pool is exhausted mid-run

It never stalls or shrinks; it **reshuffles and cycles** (`samplers.py:78-84`):

```python
while k > 0:
    if self._cur >= self._perm.size:          # exhausted -> reshuffle (cycle)
        self._perm = self.rng.permutation(self.items)
        self._cur = 0
    take = min(k, self._perm.size - self._cur)
    out.extend(self._perm[self._cur:self._cur + take].tolist())
```

with a documented fallback to a direct with-replacement draw if a single request exceeds the
whole pool (`samplers.py:72-74`). So exhaustion changes *which* rows appear, never *how
many*.

### Verified empirically, not just read

Running the real class against the real round-1 configuration (24,000 source positives /
24,000 negatives, batch 256, `pos_fraction=0.5`, seed 42) across three epochs:

- `num_batches` (default) = **188** — matching `run_config.json`'s `steps_per_epoch: 188`;
- distinct batch sizes observed: **{256}**;
- distinct labeled-positive counts per batch: **{128}** — a single value, over all 564
  batches;
- target pool (391/391, 188 batches): batch sizes **{256}**, latent-positives per batch
  **{128}**;
- adversarial probe — a 5+5 pool asked for 50 batches of 4: returns 50 batches, all size 4,
  all with exactly 2 positives. No stall, no short batch, no exception.

### One precise caveat, in the interest of stating it exactly

The guarantee is on the **count of labeled samples**, not on their distinctness. Within one
epoch the source positive pool supplies 188 × 128 = **24,064** draws from 24,000 rows, so it
cycles once near the end of the epoch and the reshuffled tail can repeat a row already used
in the same batch. Measured: the minimum number of distinct indices in a source batch is
**254 of 256** — i.e. at most two duplicated rows, and only in the last batch or two of an
epoch. Every batch still carries exactly 128 labeled positives and 128 labeled negatives.

**So the Arazo et al. (arXiv:1908.02983) framing is safe to state as a fact.** Their first
mitigation — enforcing a minimum number of labeled samples per mini-batch — is satisfied
here by construction, at a fixed 128 of 256 per step, for every method in the project
(`run_baseline`, `run_coral`, `run_dann` use `ClassBalancedBatchSampler` directly;
`run_freematch`, `run_adamatch` and `train_skeleton` use it through `make_balanced_loader`).
The paper may say "guaranteed", not "in expectation". The one nuance worth a clause is that
this was inherited from the label-shift argument of PROJECT.md §4.3 rather than adopted as
an Arazo mitigation — §7.12 already frames it as "structurally and by accident", which is
the accurate description.
