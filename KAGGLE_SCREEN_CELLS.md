# Kaggle cells — LEVER-1 fold-0 screen (dual T4)

**Session settings (both required):**
- Accelerator: **GPU T4 x2**
- Internet: **ON** (the Zoobot encoder downloads from HuggingFace on first use)

Flag names below are the real ones from `run_screen_unfreeze.py` — paste as-is.

Replace `SLUG` with your dataset slug and bump `VERSION` when you re-upload.

---

## Cell 1 — setup, mount check, control-dump verification, 2-GPU assert

```python
import os, sys, json, glob, shutil, subprocess, textwrap
from pathlib import Path

# ---- EDIT THESE TWO ----
SLUG    = "lens-paper-screen"      # your Kaggle dataset slug
VERSION = 1                        # bump on every re-upload; printed below to prove it took
# ------------------------

IN   = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SRC  = None
for cand in IN.glob("*"):
    if (cand / "scripts").is_dir() or (cand / "MANIFEST.json").is_file():
        SRC = cand; break
assert SRC is not None, f"dataset not mounted; saw: {[p.name for p in IN.glob('*')]}"
print("dataset mount :", SRC)

# version-bump check: prove the mounted copy is the one you just uploaded
man_p = SRC / "MANIFEST.json"
assert man_p.is_file(), f"MANIFEST.json missing from {SRC} -- re-upload the bundle"
man = json.loads(man_p.read_text())
print("manifest scripts:", man["n_scripts"])
stamp = SRC / "VERSION.txt"
print("dataset VERSION :", stamp.read_text().strip() if stamp.is_file() else "(none)")

!pip -q install "timm==1.0.26"
import torch, timm
print("torch", torch.__version__, "| timm", timm.__version__)

# ---- 2 GPUs, and they must be T4 (sm_75). NOT P100: sm_60 is incompatible with
# Kaggle's PyTorch build and has no fp16 tensor cores (PROJECT.md 7.7).
n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs (set Accelerator = GPU T4 x2), got {n}"
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}  {p.total_memory/2**30:.1f} GB  sm_{p.major}{p.minor}")
    assert p.major >= 7, f"GPU{i} is sm_{p.major}{p.minor} (P100?) -- T4 only"
    assert p.major < 8 or True, ""   # Ada/Ampere fine too; only sm_60 is disqualifying

# ---- scripts to working dir ----
SCRIPTS = WORK / "scripts"
SCRIPTS.mkdir(parents=True, exist_ok=True)
srcdir = SRC / "scripts" if (SRC / "scripts").is_dir() else SRC
copied = 0
for name in man["scripts"]:
    s = srcdir / name
    assert s.is_file(), f"MANIFEST lists {name} but it is not in the dataset"
    shutil.copy2(s, SCRIPTS / name); copied += 1
print(f"copied {copied}/{len(man['scripts'])} script files -> {SCRIPTS}")

# ---- data: unzip to LOCAL disk (much faster than reading /kaggle/input) ----
DATA = WORK / "data"
if not DATA.is_dir():
    z = next(iter(glob.glob(str(SRC / "**" / "data.zip"), recursive=True)), None)
    if z:
        print("unzipping", z)
        !unzip -q -o "{z}" -d "{WORK}"
    else:
        cand = SRC / "data"
        assert cand.is_dir(), "no data.zip and no data/ in the dataset"
        DATA = cand
print("data root     :", DATA)
for need in ["splits/splits.json", "feature_cache/source_train.npz",
             "feature_cache/test_negatives_shared.npz",
             "feature_cache/test_pos_fold0.npz", "feature_cache/target_fold0.npz"]:
    assert (DATA / need).exists(), f"missing {need} under {DATA}"
print("data checks   : OK")

# ---- control dumps: the gate FAILS CLOSED without these, and without their ids ----
import numpy as np
CTRL = SRC / "control_scores"
assert CTRL.is_dir(), f"control_scores/ missing from the dataset ({SRC})"
for f in ("nb0_image_fold0.npz", "nb0_cached_fold0.npz"):
    p = CTRL / f
    assert p.is_file(), f"control dump {f} missing -- nb1/nb3 will refuse to start"
    with np.load(p, allow_pickle=False) as zz:
        miss = [k for k in ("pos_scores","neg_scores","pos_ids","neg_ids") if k not in zz.files]
    assert not miss, (f"{f} is missing {miss}. Row alignment cannot be verified, so the "
                      f"control gate fails closed. Re-export with ids.")
    print(f"  control dump ok: {f}")
REF_CFG = SRC / "reference" / "run_config.json"
assert REF_CFG.is_file(), "reference/run_config.json missing -- the §1 precondition check needs it"
print("control+reference: OK")

os.environ["LENS_DATA_ROOT"] = str(DATA)
print("\nSETUP COMPLETE")
```

---

## Cell 2 — measure REAL T4 step times before committing (~3 min)

Do not skip this. The local numbers are from an Ada RTX 4050; this measures the actual T4
so the wall-clock projection is real. If a rung will not fit the session, you find out here
rather than losing it at the 6-hour cut-off.

```python
%cd /kaggle/working/scripts
# TRUE batch 256, all rows in ONE process, first row discarded (cuDNN autotune inflates it).
!python bench_unfreeze_steptime.py --fp16 --mode synthetic --iters 30 \
    --n-blocks "0:0,0:0,1:0,3:1,3:0" \
    --out /kaggle/working/t4_steptime_baselineA.json
```

```python
# turn measured ms/step into a per-rung wall clock, INCLUDING the live-encode eval
import json
r = json.load(open("/kaggle/working/t4_steptime_baselineA.json"))["rows"]
ms = {}
for row in r[1:]:                                   # row 0 is the discarded warmup
    if not row.get("oom"):
        ms[(row["n_blocks"], row["grad_checkpointing"])] = row["ms_per_step"]
EVAL_S_PER_POINT = 60.0        # live re-encode of 39,279 rows; nb0 uses the cache (~0)
N_POINTS, STEPS  = 32, 8000
print(f"{'rung':<10} {'ms/step':>9} {'train h':>9} {'eval h':>8} {'TOTAL h':>9}")
for label, key, live in (("nb0_image", (0, False), False),
                         ("nb1",       (1, False), True),
                         ("nb3",       (3, True),  True)):
    if key not in ms:
        print(f"{label:<10} (not measured)"); continue
    tr = ms[key] * STEPS / 1000 / 3600
    ev = (N_POINTS * EVAL_S_PER_POINT / 3600) if live else 0.0
    print(f"{label:<10} {ms[key]:9.1f} {tr:9.2f} {ev:8.2f} {tr+ev:9.2f}")
print("\nGPU0 runs nb0_image THEN nb1 (chained) -- add those two totals.")
print("GPU1 runs nb3 alone.")
print("Kaggle session limit is 12h (9h GPU). If GPU0's chain exceeds it, run nb0_image")
print("and nb1 in separate sessions -- a rung still running at session end is LOST.")
```

---

## Cell 3 — launch both rungs concurrently on separate GPUs

GPU 0: `nb0_image` replicate → then `nb1` (chained).
GPU 1: `nb3` with checkpointing **ON** (mandatory — it OOMs without).

```python
import os, subprocess
from pathlib import Path
WORK = Path("/kaggle/working"); SCRIPTS = WORK/"scripts"
DATA  = Path(os.environ["LENS_DATA_ROOT"])
SRC   = Path([p for p in Path("/kaggle/input").glob("*") if (p/"MANIFEST.json").is_file()][0])
CTRL  = SRC/"control_scores"; REF = SRC/"reference"/"run_config.json"
LOGS  = WORK/"logs"; LOGS.mkdir(exist_ok=True)

COMMON = (f'--data-root "{DATA}" --splits "{DATA}/splits/splits.json" '
          f'--cache-dir "{DATA}/feature_cache" --fp16 --num-workers 2 '
          f'--control-scores "{CTRL}" --reference-config "{REF}"')

# GPU 0: replicate the frozen control on THIS hardware, then nb1.
#   --dir-suffix _kaggle keeps it distinct from the local run so the report can compare
#   venues (a >1-lens disagreement is a portability finding, not a nuisance).
#   '&&' chains: nb1 only starts if the replicate succeeded.
gpu0 = (f'cd {SCRIPTS} && '
        f'CUDA_VISIBLE_DEVICES=0 python run_screen_unfreeze.py --run nb0_image '
        f'--dir-suffix _kaggle {COMMON} > {LOGS}/nb0_image_kaggle.log 2>&1 && '
        f'CUDA_VISIBLE_DEVICES=0 python run_screen_unfreeze.py --run nb1 '
        f'{COMMON} > {LOGS}/nb1.log 2>&1')

# GPU 1: nb3. --grad-checkpointing is REQUIRED (numerically exact; without it the
# activations for a true-batch-256 backward through stages[1..3] do not fit).
gpu1 = (f'cd {SCRIPTS} && '
        f'CUDA_VISIBLE_DEVICES=1 python run_screen_unfreeze.py --run nb3 '
        f'--grad-checkpointing {COMMON} > {LOGS}/nb3.log 2>&1')

p0 = subprocess.Popen(["bash","-lc",gpu0])
p1 = subprocess.Popen(["bash","-lc",gpu1])
print("launched: GPU0 pid", p0.pid, "(nb0_image_kaggle -> nb1)")
print("launched: GPU1 pid", p1.pid, "(nb3, checkpointing ON)")
print("\nBoth are backgrounded; this cell returns immediately. Use Cell 4 to watch.")
```

---

## Cell 4 — monitoring (re-runnable)

```python
import subprocess, glob, os
from pathlib import Path
LOGS = Path("/kaggle/working/logs"); WORK = Path("/kaggle/working")
print(subprocess.run(["nvidia-smi",
    "--query-gpu=index,utilization.gpu,memory.used","--format=csv,noheader"],
    capture_output=True, text=True).stdout)
for lg in sorted(LOGS.glob("*.log")):
    print("="*76); print(lg.name); print("="*76)
    txt = lg.read_text(errors="replace").splitlines()
    steps = [l for l in txt if "[step" in l]
    bad   = [l for l in txt if any(k in l for k in
             ("Traceback","ABORT","REFUSING","Error","OOM","MISMATCH","FAILED CLOSED"))]
    for l in steps[-3:]: print("  ", l.strip())
    for l in bad[-6:]:   print("  !!", l.strip())
    if not steps and not bad:
        print("   (no step lines yet — still loading data / building the encoder)")
# progress by checkpoint count (filesystem truth, not log buffering)
for d in sorted(glob.glob(str(WORK/"scripts"/".."/"runs"/"baselineA_fold0_*"))):
    n = len(glob.glob(os.path.join(d,"checkpoints","step_*.pt")))
    print(f"{os.path.basename(d):<44} {n}/32 curve checkpoints")
```

---

## Cell 5 — collect results into `/kaggle/working` (only this dir persists)

```python
import os, glob, shutil, json
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK/"screen_out"; OUT.mkdir(exist_ok=True)
RUNS = Path(glob.glob(str(WORK/"**"/"runs"), recursive=True)[0])
print("runs dir:", RUNS)

LIMIT_GB = 18.0          # Kaggle working-dir budget; keep headroom
def dirsize(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())/2**30

# ALWAYS keep: curves, score dumps (with ids), configs, results. These are the deliverables.
kept = []
for d in sorted(RUNS.glob("baselineA_fold0_*")):
    dst = OUT/d.name; (dst/"scores").mkdir(parents=True, exist_ok=True)
    for f in ("curve.csv","run_config.json","results.json"):
        if (d/f).is_file(): shutil.copy2(d/f, dst/f)
    for s in (d/"scores").glob("*.npz"): shutil.copy2(s, dst/"scores"/s.name)
    kept.append(d.name)
for lg in (WORK/"logs").glob("*.log"): shutil.copy2(lg, OUT/lg.name)
for j in WORK.glob("t4_steptime_*.json"): shutil.copy2(j, OUT/j.name)
print("kept (curves+scores+configs):", kept, f"-> {dirsize(OUT):.2f} GB")

# R1 checkpoints ONLY if they fit. Curves+scores are what the report needs; checkpoints
# are large and reproducible from the config, so they are the first thing dropped.
budget_left = LIMIT_GB - dirsize(OUT)
for d in sorted(RUNS.glob("baselineA_fold0_*")):
    r1 = d/"checkpoints"/"step_8000.pt"
    if r1.is_file() and r1.stat().st_size/2**30 < budget_left:
        (OUT/d.name/"checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy2(r1, OUT/d.name/"checkpoints"/"step_8000.pt")
        budget_left -= r1.stat().st_size/2**30
        print(f"  + R1 checkpoint for {d.name}")
    elif r1.is_file():
        print(f"  - SKIPPED R1 checkpoint for {d.name} (would exceed the working-dir limit)")

# the report itself
os.chdir(WORK/"scripts")
!python report_screen_unfreeze.py --out /kaggle/working/screen_out/lever1_screen_report.json
print(f"\nFINAL /kaggle/working/screen_out = {dirsize(OUT):.2f} GB")
print("Download screen_out/ before the session ends -- nothing else persists.")
```

---

## Local — the decomposition run

Splits augmentation from precision. Task 2 measured weak-aug+fp16 at **−0.0714 TPR** vs
none+fp32 but could not attribute it, because both varied together. This is `nb0` with
**weak augmentation and fp32**, so against the two completed controls it isolates each:

- vs `nb0_image` (weak, **fp16**) → the **precision** effect
- vs `nb0_image_noaug_fp32` (none, **fp32**) → the **augmentation** effect

```bash
python scripts/run_screen_unfreeze.py --run nb0_image --aug weak --dir-suffix _weak_fp32 --diagnostic --num-workers 2
```

There is no `--fp16` flag: omitting it *is* fp32. `--diagnostic` is required — this run
deliberately varies `precision`, so the §1 check would otherwise (correctly) abort it.
`--diagnostic` still prints the full mismatch table for the record, and it is **refused for
`nb1`/`nb3`**, so it cannot be used to sneak a confound into the matched screen. It has no
effect on the control gate.

Expected wall clock ≈ **56 min** (measured: the completed `noaug_fp32` control took 56.3 min
on the local RTX 4050 at `num_workers=2`).
