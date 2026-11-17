# Kaggle cells — round-4 CORAL **confirmatory** folds 1–4 (nb3, γ=16, 2000 steps)

**Four independent runs, ONE AT A TIME.** `coral_nb3`, γ = 16, budget **2000 steps**, folds
**1, 2, 3, 4** — one fold per invocation, ~75 min each. **Fold 0 is DEVELOPMENT and both the
runner and the launcher refuse it at argument parsing** — the budget was chosen by inspecting
fold 0's own curve, so admitting it would be oracle selection.

How the folds are scheduled changes nothing about the result; they are independent runs and
this is wall-clock bookkeeping only. One at a time means a dead session costs one fold, the
log is readable while it is being written, and nothing contends for the GPU.

> **THE REPORTED RESULT IS `step_2000` AND NOTHING ELSE.** The read-off rule is fixed by the
> hashed pre-registration `runs/coral_r4_budget.json` (sha256 `65c20810485968d8…`): train to
> 2000, read the **final** checkpoint. No selector, no argmax, no smoothing — **the budget is
> the selector**. The seven intermediate 250-step checkpoints are **lab record and
> diagnostics only**. The runner verifies that hash at startup and refuses to run if it
> fails.

**Session:** GPU **T4** (×2 is fine — only one device is used per fold), Internet **ON**
(Zoobot pulls from HuggingFace on first use). **Cost:** ~75 min per fold at 2000 steps
(measured: 298 min for 8000 on one T4). Run one fold per session, or all four sequentially in
one session (~5 h, still inside the 12 h cap).

**Dataset:** `lens-paper-coral-r4`, **re-uploaded with the two new reference files and the
new scripts** (see “Bundle” at the bottom — do this first, and bump the version or Kaggle
silently serves the old copy).

---

## Cell 1 — resolve the mount, stage scripts

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
PREREG = find('coral_r4_budget.json')
DEVRUN = find('coral_r4_nb3_dev8000_fold0/run.json')
REFIDS = find('coral_r4_logits/ids.npz')

SCRIPTS = os.path.join(WORK, 'scripts')
shutil.copytree(os.path.dirname(find('scripts/run_coral_r4_confirm.py')), SCRIPTS,
                dirs_exist_ok=True)
os.chdir(WORK)
OUT = f'{WORK}/runs/coral_r4_confirm'

for n, p in [('DATA', DATA), ('SPLITS', SPLITS), ('CACHE', CACHE), ('PREREG', PREREG),
             ('DEVRUN', DEVRUN), ('REFIDS', REFIDS), ('SCRIPTS', SCRIPTS)]:
    assert os.path.exists(p), f'MISSING {n}: {p}'
    print(f'{n:8s} {p}')
for k in (1, 2, 3, 4):                        # every fold's own LP->FT head must be present
    assert os.path.exists(find(f'bakeoff_coral/fold{k}/checkpoints/step_8000.pt'))
print('per-fold phase-1 heads ok |', open(find('VERSION.txt')).read().strip())
print(round(shutil.disk_usage(WORK).free / 1e9, 1), 'GB free')
```

## Cell 2 — pre-handover suite (~1 min, CPU)

```python
!python -u scripts/test_coral_r4_confirm.py 2>&1 | tail -12
```

Must end **`127 passed, 0 failed`**. It checks that fold 0 cannot be selected, that the
pre-registration hash gate fails loudly on every way the file can be altered, that the config
allowlist is enforced *by value*, that fold 4 is treated as 97 positives, and that no
metric-based checkpoint selection exists in the path.

## Cell 3 — dry run on this machine (~2 min)

```python
!python -u scripts/run_coral_r4_confirm.py --fold 1 --dry-run 2>&1 | tail -25
```

Must end **`DRY RUN PASSED`**.

## Cell 4 — verification fold (~5 min)

```python
import subprocess, sys
cmd = [sys.executable, '-u', 'scripts/launch_coral_r4_confirm_kaggle.py',
       '--out-root', OUT, '--data-root', DATA, '--splits', SPLITS, '--cache-dir', CACHE,
       '--prereg', PREREG, '--dev-run-json', DEVRUN, '--reference-ids', REFIDS,
       '--folds', '1', '--verify-only']
print(' '.join(cmd)); print()
subprocess.run(cmd, check=True)
```

Runs fold 1 for 50 steps alone, then **reads the result back** — run.json present, config
mismatches empty, step-0 gate verdict, positive count, reported artifacts on disk. Prints
`[verify] PASSED`. **Do not go on if it does not.** Only needed once per session.

## Cell 5 — one fold (~75 min)

**Set `FOLD` and run this cell. Change it and run it again for the next fold.**

```python
FOLD = 1                      # 1, 2, 3 or 4 -- one per run. Fold 0 is refused.

import subprocess, sys
cmd = [sys.executable, '-u', 'scripts/launch_coral_r4_confirm_kaggle.py',
       '--out-root', OUT, '--data-root', DATA, '--splits', SPLITS, '--cache-dir', CACHE,
       '--prereg', PREREG, '--dev-run-json', DEVRUN, '--reference-ids', REFIDS,
       '--folds', str(FOLD), '--skip-verify']
subprocess.run(cmd, check=True)
```

The child gets `CUDA_VISIBLE_DEVICES` pinned so it sees exactly one GPU and **cannot** shard
the fold across devices — halving the per-device batch would change CORAL's 640×640
covariance estimate from 256 draws to 128 (§7.1a). Each fold writes its own directory and its
own log file, never a pipe.

`launch_report.json` **merges** across invocations, so after each fold it prints
`complete so far` and `STILL TO RUN`, with the exact command for the remainder. Running fold
1 today and fold 2 tomorrow still builds one report.

> Want them back to back instead? `'--folds', '1', '2', '3', '4'` runs all four
> **sequentially** (~5 h). `--concurrency 2` is available on a T4 × 2 if you ever want two at
> a time, but it is opt-in and not the default.

**In each fold's header, check:** its test-positive count (98 for folds 1–3, **97 for fold
4**), its one-positive resolution, its own `FOLD_GATE` value, and the verified
pre-registration hash. **In each fold's diff:** `0 mismatch(es)`, with exactly five keys
marked `*` — `fold`, `max_steps`, `init_head_from_tail`, `init_head_l2`, `epochs_planned`.
Anything else aborts before training.

## Cell 6 — package for download, with checksums

Run this after **each** fold, before the session ends.

```python
!python -u scripts/split_output_parts.py --src {OUT}/fold{FOLD} \
    --arcname coral_r4_confirm/fold{FOLD} --dest {WORK}/parts_fold{FOLD}
!ls -la {WORK}/parts_fold{FOLD} && cat {WORK}/parts_fold{FOLD}/PARTS.sha256
```

**One fold at a time has a useful side effect here:** a fold is 8 checkpoints (~475 MB)
plus its logits and CSVs, so it comes out as a **single part** — nothing to
reassemble. The manifest still ships, so `sha256sum -c` still catches a truncated download.

The tar is streamed straight into parts (no intermediate full-size archive) and
`PARTS.sha256` carries a digest per part **and** for the whole tar. The 900 MiB default is
the size measured to download cleanly; the fold-0 development run's single 1.95 GB file
would not.

If you did run all four in one session, package the parent instead —
`--src {OUT} --arcname coral_r4_confirm --dest {WORK}/parts` — that is ~1.9 GB
and splits into three parts.

Then locally:

```bash
cat fold1.tar.part_a* > fold1.tar && sha256sum -c PARTS.sha256
```

```bash
tar xf fold1.tar -C runs/
```

(With a single part, `cat` is a copy and the check still runs.) The tar's member paths are `coral_r4_confirm/fold1/`, so each fold drops into `runs/` unrenamed and the four accumulate into one `runs/coral_r4_confirm/` tree however many sessions they took.
A truncated download fails at `sha256sum -c` instead of yielding a corrupt archive.

> Want a small archive first? Add `--exclude-suffix .pt` for a few-MB copy holding
> `run.json`, `config_diff.json`, the CSVs and **all the logits** — everything the read-off
> needs. The checkpoints are only for model-space diagnostics.

---

## Bundle — what to add before re-uploading

The dataset slug stays **`lens-paper-coral-r4`**. Add these to `kaggle/coral_r4/dataset/`
(already staged locally) and **bump the version stamp** so Cell 1 can print which copy
mounted:

| add | where | why |
|---|---|---|
| `runs/coral_r4_budget.json` | `reference/coral_r4_budget.json` | the hashed pre-registration; the runner verifies it and reads the budget out of it |
| the fold-0 arm's `run.json` | `reference/coral_r4_nb3_dev8000_fold0/run.json` | the config-diff reference — every confirmatory fold is diffed field by field against it |
| `scripts/run_coral_r4_confirm.py` | `scripts/` | the runner |
| `scripts/launch_coral_r4_confirm_kaggle.py` | `scripts/` | the two-at-a-time launcher |
| `scripts/split_output_parts.py` | `scripts/` | packaging with per-part sha256 |
| `scripts/test_coral_r4_confirm.py` | `scripts/` | the pre-handover suite |
| this file | `docs/KAGGLE_CORAL_R4_CONFIRM_CELLS.md` | the cells |

```bash
kaggle datasets version -p kaggle/coral_r4 -r zip -m "round-4 coral confirmatory folds 1-4"
```

---

## Operational rules encoded here

| rule | what it prevents |
|---|---|
| fold 0 refused **at argument parsing**, with the reason | development data entering the confirmatory set |
| pre-registration hash verified at startup; budget read **out of** the verified file | the trained budget drifting from the pre-registered one |
| config diffed against the fold-0 arm; allowlist pinned **by value** | fold 2 quietly training from fold 3's phase-1 head |
| per-fold positive counts resolved from splits_v6 and printed | the 98-everywhere assumption, which is wrong for fold 4 |
| one output directory and one log file per fold | two invocations interleaving rows into one series |
| `launch_report.json` merges across invocations | a per-fold session erasing the folds run before it |
| `CUDA_VISIBLE_DEVICES` pinned per child, one fold at a time by default | a fold being split across devices and changing the covariance estimate |
| `python -u`, stdout to a file, never a pipe | a log that sits at 0 bytes for an hour while healthy |
| checkpoint every 250 + full-state resume (AdamW moments, scaler, buffers, RNG) | §7.11's three silent restarts |
| per-part sha256 shipped with the parts | a truncated download passing as a good one |
| no argmax / smoothing / selector anywhere in the path, asserted by the suite | the oracle selection the pre-registration exists to prevent |
