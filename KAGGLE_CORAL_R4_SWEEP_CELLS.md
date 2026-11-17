# Kaggle cells — round-4 CORAL gamma sweep (development, fold 0, nb3 only)

**Session settings:** Accelerator **GPU T4 × 2**, Internet **ON** (Zoobot pulls from
HuggingFace on first use).

**Dataset:** uploaded manually through the Kaggle UI. Nothing in this document uploads
anything or needs the Kaggle CLI. The code only has to *read* it.

Your dataset is attached at:

```
/kaggle/input/datasets/navneet78/lens-paper-coral-r4
```

Cell 1 uses that path, and if it is ever different it **searches for the dataset instead of
guessing** — so the exact mount layout stops mattering.

What it needs to find inside, whatever the enclosing path:

```
data/splits/splits.json
data/feature_cache/{source_train,val,test_negatives_shared,target_fold0,test_pos_fold0}.npz
scripts/*.py
runs/bakeoff_coral/fold0/checkpoints/step_8000.pt      <- the LP->FT phase-1 head
```

`/kaggle/input` is read-only, so **the 4.4 GB of data is read straight from the mount and
never copied**. Only `scripts/` (1.7 MB) is copied into `/kaggle/working`, which is the
writable CWD and where all sweep output goes.

---

## Cell 1 — resolve the dataset, stage scripts

`data.zip` auto-extracts to `data/data/`, so the pieces do NOT share one common root: the
data sits under `<ds>/data/data/...` while `scripts/` and `runs/` sit under `<ds>/...`.
Each path is therefore located independently.

```python
import os, glob, shutil

INP, WORK = '/kaggle/input', '/kaggle/working'

def find(pat):
    hits = glob.glob(os.path.join(INP, '**', pat), recursive=True)
    assert hits, f'NOT FOUND: {pat}'
    return sorted(hits, key=len)[0]          # shallowest match

SPLITS = find('splits/splits.json')
DATA   = os.path.dirname(os.path.dirname(SPLITS))          # the data_root LensDataset wants
CACHE  = os.path.dirname(find('feature_cache/source_train.npz'))
HEAD0  = find('bakeoff_coral/fold0/checkpoints/step_8000.pt')

SCRIPTS = os.path.join(WORK, 'scripts')
shutil.copytree(os.path.dirname(find('scripts/coral_round4.py')), SCRIPTS,
                dirs_exist_ok=True)
os.chdir(WORK)

for n, p in [('DATA', DATA), ('SPLITS', SPLITS), ('CACHE', CACHE),
             ('HEAD0', HEAD0), ('SCRIPTS', SCRIPTS)]:
    assert os.path.exists(p), f'MISSING {n}: {p}'
    print(f'{n:8s} {p}')
for sub in ('sim_arc_on_real', 'q1_unlabeled', 'ml_fp_cutouts', 'q1_lenses_processed'):
    assert os.path.isdir(os.path.join(DATA, sub)), f'MISSING {sub} under {DATA}'
print('data subdirs ok |', len(glob.glob(SCRIPTS + '/*.py')), 'scripts |',
      round(shutil.disk_usage(WORK).free / 1e9, 1), 'GB free')
```

## Cell 2 — the test suite (~5 min)

```python
!python -u scripts/test_coral_round4.py 2>&1 | tail -14
```

Must end `193 passed, 0 failed`. Catches a bad mount or a stale script copy before any long
run.

## Cell 3 — verification gate: gamma=1, 100 steps, alone on cuda:0 (~4 min)

```python
import subprocess, sys
cmd = [sys.executable, '-u', 'scripts/launch_coral_sweep_kaggle.py',
       '--out-root', f'{WORK}/runs/coral_r4_sweep',
       '--data-root', DATA, '--splits', SPLITS, '--cache-dir', CACHE,
       '--init-head-from', HEAD0,
       '--verify-only']
print(' '.join(cmd)); print()
subprocess.run(cmd, check=True)
```

Runs gamma=1 for 100 steps alone, then **reads the trace back** — file present, header as
expected, rows growing, last row finite — and prints `[verify] PASSED` with the last row.
**Do not go on if it does not.** `subprocess.run` is used instead of `!` so the paths from
Cell 1 are passed as real Python variables, not shell-interpolated strings.

## Cell 4 — the sweep (~2–4 h)

```python
import subprocess, sys
cmd = [sys.executable, '-u', 'scripts/launch_coral_sweep_kaggle.py',
       '--out-root', f'{WORK}/runs/coral_r4_sweep',
       '--data-root', DATA, '--splits', SPLITS, '--cache-dir', CACHE,
       '--init-head-from', HEAD0,
       '--skip-verify']
subprocess.run(cmd, check=True)
```

- gamma=1 → `cuda:0`, gamma=4 → `cuda:1`, **concurrently**; gamma=16 on whichever frees first
- each arm its own process, `python -u`, stdout+stderr to its **own file**
  (`runs/coral_r4_sweep/logs/g{1,4,16}.log`) — never a pipe
- each arm its **own output directory**, and the runner refuses to start if its trace
  already exists
- checkpoints every **250** steps, so a session timeout at step 1900 is not a total loss

`--skip-verify` is correct only because Cell 3 passed. Watch it live from another cell:
`!tail -n 20 runs/coral_r4_sweep/logs/g1.log`

## Cell 5 — package the traces for download

```python
import tarfile, os
base = f'{WORK}/runs/coral_r4_sweep'
out  = f'{WORK}/coral_r4_sweep_traces.tar.gz'
with tarfile.open(out, 'w:gz') as t:
    for d in ('g1', 'g4', 'g16', 'logs'):
        p = os.path.join(base, d)
        if os.path.isdir(p):
            t.add(p, arcname=f'coral_r4_sweep/{d}',
                  filter=lambda ti: None if ti.name.endswith('.pt') else ti)
    lr = os.path.join(base, 'launch_report.json')
    if os.path.isfile(lr):
        t.add(lr, arcname='coral_r4_sweep/launch_report.json')
print(out, round(os.path.getsize(out) / 1e6, 1), 'MB')
```

Checkpoints excluded — the traces are what the criterion reads and they are a few hundred
KB. Download from the notebook's **Output** tab.

---

## Then, LOCALLY — the criterion evaluation

Not on Kaggle: CPU work on ~2000-row CSVs, it must not burn GPU session time.

```bash
python scripts/analyze_coral_gamma_sweep.py --sweep-dir runs/coral_r4_sweep
```

Prints the pre-registered criterion, the three traces, clause (i) and (ii) per gamma, then
the verdict. The criterion is fixed in `coral_round4.SWEEP_CRITERION` and is **not to be
amended after seeing the traces**:

- **(i)** raw CORAL not monotonically rising over the second half — OLS slope, moving-block
  bootstrap (block 50, B = 5000); **PASS iff the 95% CI's lower bound ≤ 0**, i.e. fail only
  when demonstrably rising.
- **(ii)** source feature-norm growth over the full window measurably damped vs the
  gamma = 1.0 arm; **PASS iff the CI of the difference lies entirely below zero**;
  **INAPPLICABLE** if the reference arm did not grow.

Smallest gamma passing both. If none does, **nothing is selected** — extend the sweep upward
(64, then 256), never relax a clause.

---

## What this session is for

Three independent 2000-step arms of Deep CORAL at `n_blocks=3`, fold 0, gamma ∈ {1, 4, 16},
target EMA off. Per-arm traces of the loss dynamics and the feature-scale monitor. **No
test-set quantity is computed and none is reachable** — the whole run sits inside
`test_label_lockout()`, with no curve hook, no logit dump and no memorization probe.

It answers one question: *at what gamma does the CORAL term stop losing?* The 500-step
calibration showed magnitude-matching is ill-posed unfrozen — raw CORAL scales as (feature
scale)⁴, the encoder moves the scale, so "equalise at the end of training" has no fixed
point (nb3's gamma halved across its own tail, CV 0.868). At gamma = 1.0 the CORAL term is
losing: `sup_loss` falls, raw CORAL rises, the norm rises 25% in 400 steps. An 8000-step run
in that state would produce a null nobody could interpret.

Fold 0 is development (§7.11/§7.14). No number here enters a headline.

## Operational rules encoded here

| rule | what it prevents |
|---|---|
| Cell 1 **searches** for the dataset at any depth and prints the real tree on failure | the single-level glob that could not see `datasets/<owner>/<slug>` |
| data read from the read-only mount, never copied | a pointless 4.4 GB copy into `/kaggle/working` |
| `python -u`, stdout to a file, never a pipe | the calibration's log sat at 0 bytes for an hour while healthy |
| one output directory per gamma, refuse-if-exists | two processes racing on one output file |
| liveness via `Popen.poll()` | `ps -p` in Git Bash cannot see native Windows PIDs and called a live run dead |
| verification arm before committing both GPUs | a path problem costing a session instead of three minutes |
| checkpoints every 250 steps | a timeout at step 1900 losing everything |
