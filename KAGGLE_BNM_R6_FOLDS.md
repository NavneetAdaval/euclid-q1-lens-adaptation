# Round 6 — BNM × nb3 CONFIRMATORY FOLDS 1–4

**Pre-registration hashed:** `runs/bnm_r6_prereg.json`,
sha256 `d205f6784213b75d8bdbd37a989836b37029e2ace17c7166bb3647bfd35d4587` (§7.25).
Nothing in it may be amended — §7.21 applies. Annotations live in §7.26 and §7.27.

    gamma        0.00787
    budget       1000   (outcome `no_matched_arm_budget` -- NOT cross-arm comparable)
    run to       4000   so all four readouts exist from one run
    cadence      every 25 steps to 500 (statistics only), then every 250
    read-off     R1 -- the final checkpoint at each readout step

**The budget selects the headline checkpoint; it does not shorten the run.**

| step | role |
|---:|---|
| **1000** | headline — BNM read where its own objective says it is done. **Not cross-arm comparable.** |
| **4000** | **mandatory** — the λ=0 known-null's budget. Carries every cross-arm contrast. |
| 1500 | matched to FreeMatch |
| 2000 | matched to CORAL |

---

## ⚠ ONE FOLD PER COMMITTED SESSION

Each fold is 4000 steps, ≈ **2.5 hours** (the λ=0 folds ran 138–152 min at the same step
count). Four folds is ~10 hours — too long for one session, and `run_bnm_r6.py` still has
**no resume** (its sha256 is in the hashed pre-registration; §7.26 explains why it was not
added). So:

> **Set `FOLD` in Cell 3, then Save Version → Save & Run All (Commit). One fold per session.
> Repeat four times.** An interactive session idles out after ~20 minutes and would lose the
> run.

Download each fold's results before starting the next.

---

## Cell 1 — the accelerator must be a T4

§7.7: results are not portable across GPU architectures at this precision, and every arm
this is compared against ran on a T4. **The GPU name is recorded per fold** in
`run_config.json`.

```python
import torch
print("cuda:", torch.cuda.is_available())
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("gpu :", name)
assert name and "T4" in name, (
    f"THIS MUST RUN ON A T4, got {name!r}. Settings -> Accelerator -> GPU T4 x2.")
print("OK -- T4 confirmed")
```

---

## Cell 2 — resolve the mount, stage, run the suites

Needs the bundle at **v20 or later**.

```python
import os, sys, glob, json, shutil, subprocess

# SELF-SUFFICIENT ON PURPOSE. An earlier version of this cell read INPUT_ROOT from the
# environment, which is set by Cell 2 of KAGGLE_LAMBDA0_R6.md -- so pasting this cell into
# a fresh notebook, or restarting the kernel, raised KeyError: 'INPUT_ROOT'. A cell that
# depends on another DOCUMENT having been run in the same session has a prerequisite it
# cannot check, so this one resolves the mount itself. Running Cell 2 first is harmless;
# not running it is now also harmless.

WORK = "/kaggle/working"

# --- resolve the mount BY CONTENT: find MANIFEST.json and read bundle == "dann_r5" ---
# The dataset is named `lens-paper-coral-r4` (historical -- the bundle inside is dann_r5)
# and Kaggle mounts it several levels deep, so neither a name nor a path shape is evidence.
found = []
for d in range(1, 6):
    for mp in glob.glob("/kaggle/input/" + "*/" * d + "MANIFEST.json"):
        try:
            found.append((os.path.dirname(mp), json.load(open(mp))))
        except Exception:
            pass
cands = [(r, m) for r, m in found if m.get("bundle") == "dann_r5"]
if not cands:
    raise SystemExit(
        "no dann_r5 bundle found under /kaggle/input.\n"
        "  Add Data -> the dataset holding it (currently named 'lens-paper-coral-r4').\n"
        f"  manifests seen: {[r for r, _ in found] or 'none'}")
cands.sort(key=lambda rm: int(rm[1].get("version") or 0))
INPUT, man = cands[-1]
os.environ["INPUT_ROOT"] = INPUT

ver = os.path.join(INPUT, "VERSION.txt")
print("mount   :", INPUT)
print("VERSION :", open(ver).read().strip() if os.path.isfile(ver) else "ABSENT")
print(f"manifest: {man.get('bundle')} v{man.get('version')}  built {man.get('built_utc')}")

# --- what STAGE 2 needs, held HERE so no mounted manifest can shrink the list ---
NEED_SCRIPTS = ["bnm_round6.py", "run_bnm_r6.py", "test_bnm_round6.py", "test_da_bnm.py"]
NEED_ARTIFACTS = ["runs/bnm_r6_stage0a.json"]   # stage0a_anchors() reads this on import

missing = [f"scripts/{n}" for n in NEED_SCRIPTS
           if not os.path.isfile(os.path.join(INPUT, "scripts", n))]
missing += [a for a in NEED_ARTIFACTS if not os.path.isfile(os.path.join(INPUT, a))]
if missing:
    raise SystemExit(
        "THE MOUNTED DATASET PREDATES ROUND 6.\n"
        + "".join(f"    absent from the mount: {m}\n" for m in missing)
        + f"  mounted version: v{man.get('version')}\n"
          "  The notebook is attached to an older dataset version -- the one the lambda=0\n"
          "  run used (v14). Cell 2 would NOT catch this: it validates against the MOUNTED\n"
          "  manifest, which lists only what that older bundle contained.\n"
          "  FIX: upload kaggle/dann_r5/ at v15 or later, then Add Data -> REMOVE the\n"
          "  dataset -> add it again. Kaggle keeps a notebook pinned to the version it was\n"
          "  attached with; a new upload does not move it.")
print(f"  {len(NEED_SCRIPTS)} scripts + {len(NEED_ARTIFACTS)} artifact(s) present on the mount")

# --- stage the WHOLE script set. run_bnm_r6 imports ~15 sibling modules, so copying
#     only the four round-6 files would fail on the first import if Cell 2 never ran. ---
os.makedirs(f"{WORK}/scripts", exist_ok=True)
n_copied = 0
for src in glob.glob(os.path.join(INPUT, "scripts", "*.py")):
    dst = os.path.join(WORK, "scripts", os.path.basename(src))
    if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy2(src, dst)
        n_copied += 1
print(f"  staged {n_copied} script(s) into {WORK}/scripts "
      f"({len(glob.glob(f'{WORK}/scripts/*.py'))} present)")

# --- stage the ARTIFACTS too. An earlier version of this cell checked that
# runs/bnm_r6_stage0a.json was PRESENT ON THE MOUNT and then copied only *.py, so
# stage0a_anchors() -- which resolves <repo>/runs/bnm_r6_stage0a.json, i.e. under
# /kaggle/working -- raised FileNotFoundError inside T6. A presence check on the mount is
# not a presence check where the code reads. ---
for rel in NEED_ARTIFACTS:
    src, dst = os.path.join(INPUT, rel), os.path.join(WORK, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy2(src, dst)
    print(f"  staged {rel} -> {dst}")

# --- and VERIFY where the code will actually READ, not where it was copied from.
# This is the check the missing-artifact bug slipped past. ---
unstaged = [f"scripts/{n}" for n in NEED_SCRIPTS
            if not os.path.isfile(os.path.join(WORK, "scripts", n))]
unstaged += [a for a in NEED_ARTIFACTS if not os.path.isfile(os.path.join(WORK, a))]
if unstaged:
    raise SystemExit("present on the mount but NOT staged into the working tree:"
                     + "".join(os.linesep + "    " + u for u in unstaged))

# the anchors must LOAD, here, before any test claims they do
sys.path.insert(0, f"{WORK}/scripts")
import bnm_round6 as _B
_a = _B.stage0a_anchors()
print(f"  stage-0a anchors load: init={_a['init']:.4f}  "
      f"init_pred_pos_frac={_a['init_pred_pos_frac']:.4f}")

for t in ("test_da_bnm.py", "test_bnm_round6.py"):
    r = subprocess.run([sys.executable, "-u", f"scripts/{t}"],
                       cwd=WORK, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
    print(f"{t:<24} rc={r.returncode}  {tail}")
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-2000:])
        raise SystemExit(f"{t} FAILED on the venue -- do not proceed")
print("\nboth suites green on the venue")
```

---

## Cell 3 — the fold (~2.5 h)

**Set `FOLD` to 1, 2, 3 or 4.**

```python
import subprocess, sys, os, time

FOLD = 1          # <-- 1, 2, 3, 4 -- ONE PER SESSION

INPUT = os.environ["INPUT_ROOT"]
OUT = f"/kaggle/working/runs/bnm_r6_fold{FOLD}"
LOG = f"/kaggle/working/bnm_r6_fold{FOLD}.log"

cmd = [sys.executable, "-u", "scripts/run_bnm_r6.py",
       "--fold", str(FOLD), "--gamma", "0.00787", "--max-steps", "4000",
       "--input-root", INPUT, "--out-dir", OUT]
print(" ".join(cmd), flush=True)

t0 = time.time()
with open(LOG, "w", buffering=1) as lf:
    p = subprocess.Popen(cmd, cwd="/kaggle/working", stdout=lf,
                         stderr=subprocess.STDOUT, text=True)
    shown = 0
    while p.poll() is None:
        time.sleep(60)
        lines = open(LOG, errors="replace").readlines()
        for ln in lines[shown:]:
            print(ln.rstrip(), flush=True)
        shown = len(lines)
    for ln in open(LOG, errors="replace").readlines()[shown:]:
        print(ln.rstrip(), flush=True)

print(f"\nrc={p.returncode}  wall={(time.time() - t0) / 60:.1f} min")
if p.returncode != 0:
    raise SystemExit(f"FOLD {FOLD} FAILED -- read the log above")
```

---

## Cell 4 — verify before packaging

The **step-0 gate is non-skippable**: at step 0 the model *is* Baseline A, so it must
reproduce that fold's pinned TPR exactly, `|diff| = 0.0`.

```python
import csv, json, os

FOLD = 1          # <-- same fold as Cell 3
OUT = f"/kaggle/working/runs/bnm_r6_fold{FOLD}"
cfg = json.load(open(os.path.join(OUT, "run_config.json")))

PINNED = {1: 0.14285714285714285, 2: 0.19387755102040816,
          3: 0.15306122448979592, 4: 0.1958762886597938}

full = list(csv.DictReader(open(os.path.join(OUT, "checkpoints.csv"))))
sp = os.path.join(OUT, "stats_only.csv")
stats = list(csv.DictReader(open(sp))) if os.path.isfile(sp) else []
trace = list(csv.DictReader(open(os.path.join(OUT, "trace.csv"))))

print("=" * 78)
print(f"FOLD {FOLD} -- gamma 0.00787, 4000 steps")
print("=" * 78)
print(f"  steps completed        {cfg['steps_completed']} / 4000")
print(f"  wall                   {cfg['wall_minutes']:.1f} min")
print(f"  GPU (recorded, 7.7)    {cfg.get('gpu_name')}  fp16={cfg.get('fp16')}")
print(f"  FULL checkpoints       {len(full)}   (expect 16)")
print(f"  stats-only rows        {len(stats)}  (expect 18)")

g = cfg["step0_gate"]
print(f"\n  STEP-0 GATE            tpr={g['tpr']:.12f}")
print(f"    pinned for fold {FOLD}    {PINNED[FOLD]:.12f}")
print(f"    |diff|                 {abs(g['tpr'] - PINNED[FOLD]):.3e}")

inv = cfg["gamma_invariant"]
print(f"\n  GAMMA INVARIANT        held={inv['held']}  n_obs={inv['n_observations']}")
print(f"    range                [{inv['gamma_effective_min']:.10g}, "
      f"{inv['gamma_effective_max']:.10g}]")

adv = [r for r in trace if r.get("collapse_advisory_ceiling_ok") == "0"]
gate = [r for r in trace if r.get("collapse_gate_ok") == "0"]
print(f"\n  advisory ceiling       {len(adv)}/{len(trace)} rows"
      + (f"  first step {adv[0]['step']}" if adv else "  -- never"))
print(f"  gating floor           {len(gate)}/{len(trace)} rows"
      + (f"  first step {gate[0]['step']}" if gate else "  -- never"))

missing = [r["step"] for r in full
           if not os.path.isfile(os.path.join(OUT, r["logit_file"]))]
print(f"\n  per-checkpoint logits  {len(full) - len(missing)}/{len(full)} present"
      + (f"  MISSING {missing}" if missing else "  -- all present"))

print("\n  THE FOUR READOUTS (logit files must exist for each):")
for s in (1000, 1500, 2000, 4000):
    row = next((r for r in full if int(r["step"]) == s), None)
    ok = row and os.path.isfile(os.path.join(OUT, row["logit_file"]))
    print(f"    step {s:>5}  {'present' if ok else 'ABSENT'}")

print("\n  704/78 at the readouts (both morphology instruments kept separate, 7.19):")
print(f"    {'step':>6} {'nucGap':>9} {'entGap':>9} {'probe':>8} {'headPool':>9}")
for r in full:
    if int(r["step"]) in (1000, 1500, 2000, 4000):
        print(f"    {int(r['step']):>6} {float(r['nuclear_gap']):>+9.4f} "
              f"{float(r['entropy_gap']):>+9.4f} "
              f"{float(r['frozen_probe_auroc']):>8.4f} "
              f"{float(r['morph_head_auroc_pool']):>9.4f}")

assert cfg["steps_completed"] == 4000, "did not reach 4000"
assert len(full) == 16, f"expected 16 full checkpoints, got {len(full)}"
assert not missing, "some per-checkpoint logits are absent"
assert inv["held"] is True, "THE GAMMA INVARIANT DID NOT HOLD -- do not use this fold"
assert abs(g["tpr"] - PINNED[FOLD]) == 0.0, (
    f"STEP-0 GATE: |diff| is not exactly 0 for fold {FOLD}")
print("\nALL CHECKS PASSED -- ready to package")
```

---

## Cell 5 — package

```python
import subprocess, sys
FOLD = 1          # <-- same fold

r = subprocess.run(
    [sys.executable, "-u", "scripts/build_dann_r5_bundle.py",
     "--pack-results", f"/kaggle/working/runs/bnm_r6_fold{FOLD}",
     "--results-out", "/kaggle/working/results",
     "--logs", f"/kaggle/working/bnm_r6_fold{FOLD}.log"],
    cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-6000:]); print(r.stderr[-2000:])
```

**Download `screen_core.tar.gz` and `MANIFEST.json`** — that is all the analysis needs. The
checkpoint tars are only needed if a later stage wants weights.

Send me each fold's Cell 4 block as it finishes; I do not need to wait for all four.

---

## What I will report, per §5 of the brief

Primary **TPR@FPR=1e-3**, interpolated, `drop_intermediate=False`, always a rate.
Quantisation **1/98** folds 1–3, **1/97** fold 4. Per-fold ROC; **pooling is invalid**.
Paired **t-interval on 4 differences, df=3** — never a bootstrap at n=4.

Contrasts, each carrying the §7.18 disclosure that **a budget match removes the budget
confound only** (head init and γ-tuning budget still travel):

- **BNM(4000) − λ=0(4000)** — the headline, and the only budget-matched contrast against the
  known-null
- BNM(1500) − FreeMatch(1500), BNM(2000) − CORAL(2000), BNM(4000) − DANN(4000)

No BNM number is differenced against any round-1 number. Results stated as a **bound**, not a
null. Multiplicity disclosed: intervals counted, no correction applied.

The 704/78 prediction scored under **all three readings** declared in §7.27 — fold-mean of
the read-off value (primary), any-checkpoint maximum, and within-fold mean over checkpoints.

**The interval is computed before the sentence is written.** Fold 0 read 0.306 at the budget
and 0.398 at 4000; both go in the table, and no sentence is anchored to whichever preview
reads better.
