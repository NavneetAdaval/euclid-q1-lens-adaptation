# Kaggle cells — CORAL's §7.14 fold-0 development run (nb3, γ=16, 8000 steps)

**One arm.** `coral_nb3`, gamma = 16, fold 0, **8000 steps, fresh from step 0**. It is the
existing 2000-step gamma=16 sweep arm re-run four times as long, with **logits persisted at
every checkpoint**. It is *not* a continuation of that arm — §7.11's three silent restarts
(threshold buffers, schedule position, AdamW moments) are the reason.

**Session settings:** Accelerator **GPU T4 ×1** (T4 ×2 also works — the script uses `cuda:0`
only and never builds a DataParallel; halving the per-device batch would change CORAL's
covariance estimate from 256 draws to 128, §7.1a). Internet **ON** (Zoobot pulls from
HuggingFace on first use). Persistence: **Files only** is enough.

**Dataset:** `lens-paper-coral-r4`, re-uploaded with the new version (bump the version or
Kaggle silently serves the old copy). It mounts at

```
/kaggle/input/datasets/navneet78/lens-paper-coral-r4
```

but nothing below hard-codes that — every path is **searched**, so the mount layout does not
matter.

| what the run needs inside the dataset | why |
|---|---|
| `data/…/splits/splits.json`, `data/…/feature_cache/source_train.npz` | the splits and the standardizer's source features |
| `runs/bakeoff_coral/fold0/checkpoints/step_8000.pt` | the LP→FT phase-1 head **and** the step-0 known null |
| `reference/coral_r4_g16_2000/{sweep.json,steps.csv}` | the config diff and the reproduction check |
| `reference/coral_r4_logits/ids.npz` | the id-order check |
| `scripts/` | copied to `/kaggle/working` and imported |

**Cost.** ~200 min of training (measured: 50.2 min per 2000 steps for this exact arm on a
Kaggle T4) plus ~30–50 min of scoring across 32 checkpoints ⇒ **≈ 4 h against the 12 h cap.**
Checkpoints every 250 steps, and resume restores AdamW's moments, so a dead session costs at
most 250 steps.

---

## Cell 1 — resolve the dataset, stage scripts

```python
import os, glob, shutil

INP, WORK = '/kaggle/input', '/kaggle/working'

def find(pat):
    hits = glob.glob(os.path.join(INP, '**', pat), recursive=True)
    assert hits, f'NOT FOUND: {pat}'
    return sorted(hits, key=len)[0]          # shallowest match

SPLITS = find('splits/splits.json')
DATA   = os.path.dirname(os.path.dirname(SPLITS))
CACHE  = os.path.dirname(find('feature_cache/source_train.npz'))
HEAD0  = find('bakeoff_coral/fold0/checkpoints/step_8000.pt')
REFDIR = os.path.dirname(find('coral_r4_g16_2000/steps.csv'))
REFIDS = find('coral_r4_logits/ids.npz')

SCRIPTS = os.path.join(WORK, 'scripts')
shutil.copytree(os.path.dirname(find('scripts/run_coral_r4_dev8000.py')), SCRIPTS,
                dirs_exist_ok=True)
os.chdir(WORK)
OUT = f'{WORK}/runs/coral_r4_nb3_dev8000_fold0'

for n, p in [('DATA', DATA), ('SPLITS', SPLITS), ('CACHE', CACHE), ('HEAD0', HEAD0),
             ('REFDIR', REFDIR), ('REFIDS', REFIDS), ('SCRIPTS', SCRIPTS)]:
    assert os.path.exists(p), f'MISSING {n}: {p}'
    print(f'{n:8s} {p}')
for sub in ('sim_arc_on_real', 'q1_unlabeled', 'ml_fp_cutouts', 'q1_lenses_processed'):
    assert os.path.isdir(os.path.join(DATA, sub)), f'MISSING {sub} under {DATA}'
print(open(find('VERSION.txt')).read().strip())
print('data subdirs ok |', len(glob.glob(SCRIPTS + '/*.py')), 'scripts |',
      round(shutil.disk_usage(WORK).free / 1e9, 1), 'GB free')
```

`/kaggle/input` is read-only, so the 4.4 GB of data is **read from the mount, never copied**.
Only `scripts/` (~1.8 MB) goes into `/kaggle/working`.

## Cell 2 — the pre-handover suite (~1 min, CPU)

```python
!python -u scripts/test_coral_r4_dev8000.py 2>&1 | tail -12
```

Must end **`75 passed, 0 failed`**. It checks the reproduction check against the real stored
traces (including a known null and the γ=4 discrimination case), the config diff, path
resolution, the logit format and id order, and the resume bookkeeping. A stale script copy or
a bad mount fails here, in a minute, instead of four hours in.

## Cell 3 — dry run on this machine (~2 min, GPU)

```python
!python -u scripts/run_coral_r4_dev8000.py --dry-run 2>&1 | tail -30
```

Mock data on the real `splits_v6_label_free_da` schema through the **real** loop: real
`LensDataset`, real samplers, real Zoobot encoder at `n_blocks=3`, real `Fp32CoralLoss`, real
logit dump, real resume round-trip. Must end
**`DRY RUN PASSED`**. It is the cheapest possible proof that this GPU can run the loop at all.

## Cell 4 — the run (~4 h)

```python
import subprocess, sys, os
os.makedirs(f'{OUT}/logs', exist_ok=True)
LOG = f'{OUT}/logs/dev8000.log'
cmd = [sys.executable, '-u', 'scripts/run_coral_r4_dev8000.py',
       '--out-dir', OUT,
       '--data-root', DATA, '--splits', SPLITS, '--cache-dir', CACHE,
       '--init-head-from', HEAD0,
       '--reference-dir', REFDIR, '--reference-ids', REFIDS]
print(' '.join(cmd), '\n')
with open(LOG, 'w', buffering=1) as fh:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:            # tee: the notebook sees it AND the file keeps it
        fh.write(line)
        print(line, end='')
print('\nreturncode', p.wait())
```

Everything else is a default and every default **is** the pinned arm: γ=16, fold 0, 8000
steps, target EMA off, feature-norm band [40, 110], batch 256/256, fp16 train + fp32 eval,
`--num-workers 2`, checkpoints and logits every 250, trace every 125.

> **Why a tee and not a pipe.** §9's rule is "`python -u`, stdout to a file, never a pipe",
> and the failure it came from was `python … | grep`, where **grep** block-buffered and the
> log sat at 0 bytes for an hour. Reading the child's lines in Python and writing them
> through is not that failure: `-u` keeps the child unbuffered, and the file is opened
> line-buffered. You get both the live view and the durable log.

**What to watch for, in the order it prints:**

1. the **config diff** — 31 keys MATCH, `max_steps` EXPECTED, `feat_norm_floor` NEW,
   `0 mismatch(es)`. Any mismatch **aborts before training** (that is deliberate).
2. the **step-0 gate** — TPR@1e-3 should come back at 0.214286 and AUROC at 0.947701. It
   *warns* rather than aborting outside one positive, because §7.7 measured a 0.0204
   local-vs-T4 drift on this very metric; the deterministic ‖head‖₂ = 66.268356 check has
   already hard-gated before this point.
3. the **peak GPU memory** line, printed once at step 5. The matched reference is **3.29 GB
   allocated** on this arm's own 2000-step Kaggle T4 run; 5.86 GB was the local RTX 4050
   figure. A difference over 0.5 GB from the Kaggle reference is called out explicitly.
4. the **reproduction check at step 250 and step 2000** — the whole point. `REPRODUCED`
   means this trajectory is the stored arm's trajectory and the extension is comparable to
   everything already analysed. `DIVERGED` prints a banner and **the run keeps going** — but
   read the table and `config_diff.json` before using the result.

## Cell 5 — if the session died: resume

Re-run **Cell 1 and Cell 4 exactly as they are.** The script finds `resume/latest.pt` and
restores weights, **AdamW's first and second moments**, the GradScaler's scale and growth
tracker, CORAL's buffers and the RNG, then replays the interrupted epoch's already-seen
batches so the sample order is exact. Every resume is recorded in `run.json`.

Honest limit, printed by the script too: DataLoader worker seeds are drawn when an iterator
is created, so **augmentation offsets after a resume are not guaranteed to match an
uninterrupted run's** (±3px translate on individual images). No schedule, buffer or moment
restarts — those are §7.11's three, and all three are handled.

## Cell 6 — package for download

```python
import tarfile, os
base, name = os.path.dirname(OUT), os.path.basename(OUT)

light = f'{WORK}/{name}_light.tar.gz'      # everything except the .pt files
with tarfile.open(light, 'w:gz') as t:
    t.add(OUT, arcname=name,
          filter=lambda ti: None if ti.name.endswith('.pt') else ti)
print(light, round(os.path.getsize(light) / 1e6, 1), 'MB')

full = f'{WORK}/{name}_checkpoints.tar'    # the 32 weight checkpoints, ~1.9 GB
with tarfile.open(full, 'w') as t:
    t.add(os.path.join(OUT, 'checkpoints'), arcname=f'{name}/checkpoints')
print(full, round(os.path.getsize(full) / 1e9, 2), 'GB')
```

Download **`…_light.tar.gz` first** — it holds `run.json`, `config_diff.json`,
`repro_check.json`, `trace.csv`, `steps.csv`, `checkpoints.csv` and **all the logits**, which
is everything the budget read-off needs. Untar it straight into `runs/` locally; the tree is
already named `coral_r4_nb3_dev8000_fold0`, no renaming.

The checkpoint tarball is only needed to re-score something the logits cannot answer. It
should not be needed at all — that is the point of persisting logits this time.

---

## Then, LOCALLY — the read-off

Not on Kaggle. CPU work on small files; it must not consume GPU session time.

```bash
tar xzf coral_r4_nb3_dev8000_fold0_light.tar.gz -C runs/
cp -r runs/coral_r4_nb3_dev8000_fold0/logits/g16dev8000 runs/coral_r4_logits/
```

The second line puts the new arm alongside `g0/g1/g4/g16` in `runs/coral_r4_logits/`, which
is why the id order was checked at step 0: the rows line up, so the paired bootstrap in
`test_coral_r4_resolvability.py` (phase 2 — `TailROC`, the shared-resample scheme from
§7.12) applies to it unchanged **once its `ARMS` and `STEPS` constants are widened** to
include `g16dev8000` and steps 250…8000. That is a two-constant edit, not a new tool.

> **`sweep_metrics_from_logits.py` does NOT read this directory.** It expects the
> `CoralCurveInstrument` four-split dump (`test_pos_fold{k}` / `test_negatives_shared` /
> `val` / `target_fold{k}` keys in one npz). This run writes the `runs/coral_r4_logits/`
> two-key `pos` / `neg` format, which is what the round-4 analysis actually uses — deliberate,
> and worth knowing before reaching for the wrong script.

The run computes **no metric after step 0** — every training segment sits inside
`coral_round4.test_label_lockout()`, and checkpoints are *scored* (logits saved) but never
*measured*. §7.14 requires CORAL's read-off rule to be written down and hashed **before** its
confirmatory folds exist, and that decision belongs in a local analysis pass, not in a
training log you watched go by.

`logits/` is `runs/coral_r4_logits/`-shaped — `ids.npz` + `true_init.npz` at the top and
`g16dev8000/step_*.npz` holding `pos`/`neg` — and the id order was checked against the
existing `ids.npz` at step 0, so the checkpoints of this run pair row-for-row with the g0 /
g1 / g4 / g16 arms already analysed.

---

## Operational rules encoded here

| rule | what it prevents |
|---|---|
| every path **searched**, never guessed | the mount layout differing between sessions |
| the config is **diffed field by field** against the stored arm and a mismatch refuses to start | four hours spent producing a curve comparable to nothing |
| the reproduction check **warns and continues** | losing a session to a check result — while never letting a divergent curve pass as comparable |
| logits persisted at **every** checkpoint | the previous pass, which kept only scalars and had to be entirely re-scored |
| the RNG is held still across scoring | the instrumentation shifting the augmentation stream and faking a divergence at step 2000 |
| checkpoint every 250 + full-state resume | §7.11's three silent restarts, and a timeout at step 7900 |
| `python -u`, tee to a file | the calibration's log sitting at 0 bytes for an hour while healthy |
| single GPU, no DataParallel | a halved per-device batch changing CORAL's 640×640 covariance estimate |
