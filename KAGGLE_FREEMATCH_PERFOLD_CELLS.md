# Kaggle cells — FreeMatch × nb3, ONE FOLD PER SESSION

**Pre-registration: PROJECT.md §7.11, frozen 2026-08-04, AMENDMENT 1 dated 2026-08-06
(budget 2000 → 1500, R1 = step 1500).** Hashes for both the original and the amended
section are in `PREREG_freematch_nb3_5fold.sha256`. Nothing here may be revised after
seeing folds 1–4.

**Session:** Accelerator **GPU T4 × 2**, Internet **ON**. One fold ≈ **1.6–1.8 h** at 1500
steps — huge margin against the 12 h cap even if the estimate runs 2× low again (the fold-0
8000-step run was predicted at 4.5–6 h and took 7.2 h).

**Change ONE variable — `FOLD` in Cell 1.** Everything downstream derives from it: the
phase-1 head checkpoint, the step-0 gate value, the head-norm check, the run directory and
the archive name. A wrong `FOLD` fails at the gate in two minutes rather than producing a
plausible wrong number.

**Run order** (`ORDER` below is a reminder, not enforced):

| # | session | what | ~time |
|---|---|---|---|
| 1–4 | `FOLD=1,2,3,4` | confirmatory, 1500 steps | ~1.6–1.8 h each |
| 5 | `FOLD=0` | development, last — it contributes nothing to the headline | ~1.6–1.8 h |

**The fold-3 frozen spot check is already DONE and still stands** — it validated
continuation-equivalence, which is budget-independent. Do not re-run it.

**Nothing run at the old 2000-step budget carries forward as confirmatory** — fold 1's
2000/3375 run is superseded. All five folds rerun at 1500.

---

## Cell 1 — setup (set `FOLD` here, nothing else)

```python
FOLD       = 3        # <<<<<< THE ONLY VARIABLE TO CHANGE
SPOT_CHECK = True     # True = frozen continuation spot check; False = the nb3 run

# ---- pre-registered per-fold gate table (§7.11). A wrong FOLD fails here, not silently. ----
FOLD_GATE = {0: (0.3877551020408163, 29.414692), 1: (0.336734693877551,  29.280975),
             2: (0.35714285714285715, 29.292334), 3: (0.3163265306122449, 29.465052),
             4: (0.3402061855670103, 29.237820)}
assert FOLD in FOLD_GATE, f"FOLD must be 0-4, got {FOLD}"
GATE_TPR, GATE_L2 = FOLD_GATE[FOLD]
RUN  = "nb0_control" if SPOT_CHECK else "nb3_unfrozen"
TAG  = f"fold{FOLD}" + ("_spotcheck" if SPOT_CHECK else "")
print(f"FOLD {FOLD} | {RUN} | gate TPR {GATE_TPR:.6f} | ||head||2 {GATE_L2}")

import os, sys, json, glob, shutil, subprocess, time, hashlib
from pathlib import Path
import numpy as np

WORK = Path("/kaggle/working"); INPUT = Path("/kaggle/input")
cands = list(INPUT.rglob("MANIFEST.json")); assert cands, "dataset not attached?"
SRC = cands[0].parent
man = json.loads((SRC/"MANIFEST.json").read_text())
print("mount:", SRC, "| bundle v", man.get("version"))

!pip -q install "timm==1.0.26"
import torch, timm
print("torch", torch.__version__, "| timm", timm.__version__)
assert torch.cuda.get_device_properties(0).major >= 7, "T4 only (§7.7)"

# ---- scripts: copy EVERY .py found; the manifest is a completeness CHECK, not the list ----
SCRIPTS = WORK/"scripts"; SCRIPTS.mkdir(parents=True, exist_ok=True)
srcdir = list(INPUT.rglob("run_unfreeze_freematch.py"))[0].parent
n = sum(1 for p in srcdir.glob("*.py") if shutil.copy2(p, SCRIPTS/p.name))
miss = [f for f in man.get("import_closure", []) if not (SCRIPTS/f).is_file()]
assert not miss, f"import closure incomplete: {miss} -- re-upload the bundle"
print(f"copied {n} scripts | closure complete")

# ---- data root: located by CONTENT, depth-agnostic ----
NEED = ["splits/splits.json", "feature_cache/source_train.npz", "feature_cache/val.npz",
        "feature_cache/test_negatives_shared.npz", f"feature_cache/test_pos_fold{FOLD}.npz",
        f"feature_cache/target_fold{FOLD}.npz", "q1_lenses_processed/lens_cutouts.npz"]
def find_data_root():
    seen = set()
    for base in (WORK, INPUT):
        for sj in base.rglob("splits.json"):
            if sj.parent.name != "splits": continue
            root = sj.parent.parent
            if root in seen: continue
            seen.add(root)
            if all((root/x).exists() for x in NEED): return root
    return None
DATA = find_data_root()
if DATA is None:
    z = next(iter(glob.glob(str(INPUT/"**"/"data.zip"), recursive=True)), None)
    assert z, "no complete data root and no data.zip"
    print("unzipping", z, flush=True)
    assert subprocess.run(["unzip","-q","-o",z,"-d",str(WORK)]).returncode == 0
    DATA = find_data_root()
assert DATA is not None, "no complete data root after unzip"
print("data root:", DATA)

# ---- THIS FOLD's phase-1 head, by sha256 where available else by path ----
HEAD = next(iter(glob.glob(str(INPUT/"**"/"bakeoff_freematch_5fold"/f"fold{FOLD}"/"**"/"step_8000.pt"),
                           recursive=True)), None)
assert HEAD, (f"fold {FOLD}'s round-1 step_8000 checkpoint is not in the dataset. Each fold "
              f"initialises from ITS OWN head -- upload runs/bakeoff_freematch_5fold/fold*/ .")
REF = next(iter(glob.glob(str(INPUT/"**"/"bakeoff_baseline_A"/"**"/"step_8000.pt"),
                          recursive=True)), None)
assert REF, "Baseline A step_8000 (agreement reference) is not in the dataset"

sys.path.insert(0, str(SCRIPTS))
import evaluate as ev
head, _std, _m = ev.load_checkpoint(HEAD, "cpu")
l2 = float(torch.sqrt(sum((p.detach().float()**2).sum() for p in head.parameters())))
print(f"phase-1 head : {HEAD}\n  ||head||2 = {l2:.6f}  (fold {FOLD} expects {GATE_L2})")
assert abs(l2 - GATE_L2) < 1e-4, (
    f"this is NOT fold {FOLD}'s phase-1 head -- ||head||2 {l2:.6f} vs {GATE_L2}. "
    f"Check FOLD and the uploaded checkpoints.")
print("agreement ref:", REF)

os.environ.update(LENS_DATA_ROOT=str(DATA), FM_HEAD=HEAD, FM_REF=REF,
                  FM_FOLD=str(FOLD), FM_RUN=RUN, FM_TAG=TAG)
NW = max(1, min(4, os.cpu_count() or 4))
os.environ["FM_NW"] = str(NW)
print(f"cpu_count {os.cpu_count()} -> num-workers {NW}")
print(f"free space {shutil.disk_usage(WORK).free/2**30:.1f} GB")
print("\nSETUP COMPLETE")
```

---

## Cell 2 — pre-flight (~3 min)

```python
# NO --max-steps: the driver's built-in nb3_unfrozen budget IS the pre-registered 1500
# (§7.11). Routing it through the flag would stamp a _SMOKE2000 dir suffix and
# "not the pre-registered budget" provenance onto a confirmatory run. A budget precondition
# in the driver ABORTS at step 0 if the effective budget is anything else
# (verified: --max-steps 2000, the OLD budget, now aborts).
MAX_STEPS = 1500   # AMENDMENT 1 (2026-08-06): was 2000
COMMON = (f'--fold {FOLD} --data-root "{DATA}" --splits "{DATA}/splits/splits.json" '
          f'--cache-dir "{DATA}/feature_cache" --fp16 --num-workers {NW} '
          f'--init-head-from "{HEAD}" --agreement-ref "{REF}" '
          f'--mu-step-offset 8000 --sat-passes 40 --adam-warmup-steps 200 '
          f'--curve-every 125 --ckpt-every 500')
GC = "" if SPOT_CHECK else "--grad-checkpointing "

def run(cmd, tail=3000):
    r = subprocess.run(f'cd {SCRIPTS} && {cmd}', shell=True, capture_output=True, text=True)
    print((r.stdout or "")[-tail:]); 
    if r.returncode: print("STDERR:", (r.stderr or "")[-2000:])
    return r

r = run(f'python run_unfreeze_freematch.py --run {RUN} {GC}{COMMON} --dry-run', 2500)
assert r.returncode == 0, "dry-run failed"

# ADVERSARIAL: the WRONG fold's head must abort, so the gate table is doing its job
wrong = 0 if FOLD != 0 else 1
wh = glob.glob(str(INPUT/"**"/"bakeoff_freematch_5fold"/f"fold{wrong}"/"**"/"step_8000.pt"),
               recursive=True)
if wh:
    r2 = subprocess.run(f'cd {SCRIPTS} && python run_unfreeze_freematch.py --run {RUN} '
                        f'{GC}{COMMON.replace(HEAD, wh[0])} --dry-run',
                        shell=True, capture_output=True, text=True)
    assert "ABORT" in (r2.stdout + r2.stderr), "the wrong-head check did NOT abort"
    print(f"adversarial ok: fold {wrong}'s head is rejected for FOLD={FOLD}")

r = run('python test_unfreeze_freematch.py', 1500)
fails = [l for l in (r.stdout or "").splitlines() if "[FAIL]" in l]
assert not fails, "pre-flight suite FAILED:\n" + "\n".join(fails)
print("\nPRE-FLIGHT COMPLETE")
```

---

## Cell 3 — run this fold (blocks; ~0.7 h spot check / ~1.8 h nb3)

```python
RUNS = WORK/"runs"; LOGS = WORK/"logs"; LOGS.mkdir(exist_ok=True)
RD = RUNS/(("freematch_nb0_control" if SPOT_CHECK else "freematch_nb3_unfrozen") + f"_fold{FOLD}")
assert not RD.exists(), f"move/delete {RD} first"
LOG = LOGS/f"{TAG}.log"

p = subprocess.Popen(["bash","-lc",
    f'cd {SCRIPTS} && CUDA_VISIBLE_DEVICES=0 python run_unfreeze_freematch.py '
    f'--run {RUN} {GC}{COMMON} > {LOG} 2>&1'])
t0 = time.time(); seen = set()
BAD = ("Traceback","ABORT","REFUSING","out of memory","OOM","retention failed","VERDICT: FAIL")
WANT_CK    = MAX_STEPS // 500 + 1     # +1 for step 0
WANT_CURVE = MAX_STEPS // 125 + 1
while p.poll() is None:
    nck = len(glob.glob(str(RD/"checkpoints"/"step_*.pt")))
    c = RD/"curve.csv"; rows = sum(1 for _ in open(c))-1 if c.is_file() else 0
    print(f"[{(time.time()-t0)/60:6.1f} min] ckpt {nck}/{WANT_CK}  "
          f"curve {rows}/{WANT_CURVE}", flush=True)
    if LOG.is_file():
        for l in LOG.read_text(errors="replace").splitlines():
            if l in seen: continue
            if any(k in l for k in BAD): seen.add(l); print("  !!", l.strip()[:170], flush=True)
            elif "WARN" in l and "GradScaler" in l: seen.add(l); print("  ~~", l.strip()[:170], flush=True)
    time.sleep(120)
print(f"\nrc={p.poll()} in {(time.time()-t0)/60:.1f} min")
print(LOG.read_text()[-3500:])
assert p.poll() == 0, f"fold {FOLD} FAILED -- see {LOG}"
```

---

## Cell 4 — report, package, and (spot check only) the stop decision

```python
import csv
r = run(f'python package_screen_results.py --runs-dir "{RUNS}" --out "{WORK}/out_{TAG}" '
        f'--logs "{LOGS}" --pattern "{RD.name}"', 4000)

cfg = json.load(open(RD/"run_config.json")); res = json.load(open(RD/"results.json"))
st  = res.get("freematch_stability") or {}
print(f"\n=== FOLD {FOLD} | {RUN} ===")
print("  step-0 gate :", cfg["step0_gate"]["passed"], cfg["step0_gate"]["step0_tpr"],
      "| expected", cfg["step0_gate"]["expected"])
print("  R1 TPR@1e-3 :", round(res["tpr"], 6), "| round-1:", GATE_TPR)
print("  ckpts", cfg["n_checkpoints_saved"], "/", cfg["n_checkpoints_expected"],
      "| peak", round(cfg.get("peak_gpu_gb") or 0, 2), "GB | min", round(res["train_minutes"],1))
inst = cfg.get("instrument") or {}
print("  fp16 skip rate:", round(inst.get("skip_rate_fp16", 0), 5),
      "| worst window", inst.get("max_window_skip_rate"))
print("  mu_eff@0:", st.get("mu_eff_at_step0"),
      "| tau rebuilt:", round((st.get("sat_reconstruction") or {}).get("tau_direct", 0), 6))

if SPOT_CHECK:
    c = {int(float(x["step"])): float(x[f"fold{FOLD}_tpr_1e-3"])
         for x in csv.DictReader(open(RD/"curve.csv"))}
    probe = [250,500,1000,1500,2000]; v = [c[s] for s in probe if s in c]
    t = np.array([c[s] for s in sorted(c) if s > 0])
    print(f"\n  SPOT CHECK -- fold {FOLD} frozen continuation")
    print("   ", " ".join(f"{s}:{c[s]:.4f}" for s in probe if s in c))
    print(f"    spread {max(v)-min(v):.4f} | drift from step0 {max(abs(x-c[0]) for x in v):.4f}"
          f" | mean {t.mean():.4f} sd {t.std():.4f} | 3-positive band 0.0306")
    v_shape = min(v) < min(v[0], v[-1]) - 0.0102
    ok = (max(v)-min(v)) <= 0.0306 and not v_shape
    verdict = ("PASS -- round 1 can serve as the comparison arm" if ok else
               "FAIL -- STOP and report before running folds 1-4")
    print(f"    V-shaped: {v_shape}   ->  {verdict}")

print("\nDOWNLOAD:")
for f in sorted((WORK/f"out_{TAG}").glob("*")):
    print(f"  {f.name:<56}{f.stat().st_size/1e6:>9.1f} MB")
```

Download **every file** in `/kaggle/working/out_{TAG}/` — `screen_core.tar.gz`,
`MANIFEST.json`, and all `ckpt_*` parts. Independent self-contained tars; verify the part
count against `MANIFEST.json` before closing the session.

---

## Locally, after each fold lands

```bash
python scripts/aggregate_folds.py --also runs/freematch_fold0_nb3_unfrozen
```

Works on a **partial set** — it reports on whatever folds are present, names the missing
ones, and marks everything provisional until folds 1–4 are all in. That's what lets you see
progress without waiting for the full set.
