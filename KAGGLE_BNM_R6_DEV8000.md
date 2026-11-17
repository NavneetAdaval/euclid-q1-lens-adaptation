# Round 6 — BNM × nb3 DEVELOPMENT RUN (fold 0, cap 8000)

**The pre-registration is hashed.** `runs/bnm_r6_prereg.json`,
sha256 `d205f6784213b75d8bdbd37a989836b37029e2ace17c7166bb3647bfd35d4587`, PROJECT.md §7.25.
Nothing in it may be amended from here — §7.21's annotate-don't-rewrite rule applies.

    fold           0  (development -- enters no mean, concludes nothing about ranking)
    gamma          0.00787
    cap            8000 steps
    cadence        every 25 steps to 500 (STATISTICS ONLY), then every 250 (full evaluation)
    expected wall  ~5 hours

---

## ⚠ READ THIS BEFORE STARTING: use *Save & Run All*, not an interactive session

**`run_bnm_r6.py` has no resume support.** DANN's runner has it; this one does not. I am
deliberately **not** adding it, because `run_bnm_r6.py`'s sha256 is recorded in the hashed
pre-registration's `code_artifacts`, and changing the file now would break the record that
exists precisely to say what ran.

That gap is tolerable on the evidence: the four λ=0 folds ran 4000 steps in 138–152 min with
**0 resumes**, and DANN's arm 1 ran the full 8000 in 267.8 min, also with 0 resumes. Resume
has never actually been needed. The real risk is a *browser disconnect*, and the fix for that
is not code:

> **Save Version → Save & Run All (Commit).** The notebook then runs headless on Kaggle's
> side and survives you closing the tab. An interactive session idles out after ~20 minutes
> of inactivity and would lose a 5-hour run.

If you would rather have resume, say so **before** starting — it means amending the code and
re-hashing the pre-registration before any run, which is clean only while no run has started.

---

## Cell 1 — the accelerator must be a T4

```python
import torch
print("cuda:", torch.cuda.is_available())
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("gpu :", name)
assert name and "T4" in name, (
    f"THIS MUST RUN ON A T4, got {name!r}. §7.7: results are not portable across GPU "
    "architectures at this precision, and every arm this is compared against ran on a T4.")
print("OK -- T4 confirmed")
```

---

## Cell 2 — resolve the mount, stage, run the suites

Needs the bundle at **v19 or later**.

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

## Cell 3 — the development run (~5 hours)

```python
import subprocess, sys, os, time

INPUT = os.environ["INPUT_ROOT"]
OUT = "/kaggle/working/runs/bnm_r6_dev8000_fold0"
LOG = "/kaggle/working/bnm_r6_dev8000.log"

cmd = [sys.executable, "-u", "scripts/run_bnm_r6.py",
       "--fold", "0", "--gamma", "0.00787", "--max-steps", "8000",
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
    raise SystemExit("DEVELOPMENT RUN FAILED -- read the log above")
```

---

## Cell 4 — verify before packaging

Checks the things the brief asks to be reported, and fails loudly rather than packaging a
half-run.

```python
import csv, json, os

OUT = "/kaggle/working/runs/bnm_r6_dev8000_fold0"
cfg = json.load(open(os.path.join(OUT, "run_config.json")))

full = list(csv.DictReader(open(os.path.join(OUT, "checkpoints.csv"))))
sp = os.path.join(OUT, "stats_only.csv")
stats = list(csv.DictReader(open(sp))) if os.path.isfile(sp) else []

print("=" * 78)
print("DEVELOPMENT RUN -- fold 0, gamma 0.00787, cap 8000")
print("=" * 78)
print(f"  steps completed        {cfg['steps_completed']} / 8000")
print(f"  wall                   {cfg['wall_minutes']:.1f} min")
print(f"  gpu / precision        {cfg.get('gpu_name')}  fp16={cfg.get('fp16')}")
print(f"  cadence                {cfg.get('cadence')}")
print(f"  FULL checkpoints       {len(full)}   (expect 32)")
print(f"  stats-only rows        {len(stats)}   (expect 18)")

inv = cfg["gamma_invariant"]
print(f"\n  GAMMA INVARIANT        held={inv['held']}  n_obs={inv['n_observations']}")
print(f"    gamma_effective range [{inv['gamma_effective_min']:.10g}, "
      f"{inv['gamma_effective_max']:.10g}]  (declared 0.00787)")

# the advisory ceiling: did it EVER trip, at any point in the run?
tr = list(csv.DictReader(open(os.path.join(OUT, "trace.csv"))))
adv = [r for r in tr if r.get("collapse_advisory_ceiling_ok") == "0"]
gate = [r for r in tr if r.get("collapse_gate_ok") == "0"]
print(f"\n  advisory ceiling tripped  {len(adv)} of {len(tr)} trace rows"
      + (f"  FIRST AT step {adv[0]['step']}" if adv else "  -- never"))
print(f"  gating floor fired        {len(gate)} of {len(tr)} trace rows"
      + (f"  FIRST AT step {gate[0]['step']}" if gate else "  -- never"))

# per-checkpoint logits must actually exist on disk
missing = [r["step"] for r in full
           if not os.path.isfile(os.path.join(OUT, r["logit_file"]))]
print(f"\n  per-checkpoint logits     {len(full) - len(missing)}/{len(full)} present"
      + (f"  MISSING: {missing}" if missing else ""))

print("\n  704/78 TRAJECTORY (full grid):")
print(f"    {'step':>6} {'nucGap':>9} {'entGap':>9} {'posGap':>9} {'probe':>8}")
for r in full:
    print(f"    {int(r['step']):>6} {float(r['nuclear_gap']):>+9.4f} "
          f"{float(r['entropy_gap']):>+9.4f} {float(r['pred_pos_frac_gap']):>+9.4f} "
          f"{float(r['frozen_probe_auroc']):>8.4f}")

print("\n  SUB-GRID (statistics only, no evaluation):")
print(f"    {'step':>6} {'||A||_*':>9} {'entropy':>9} {'pos%':>8} {'nucGap':>9}")
for r in stats:
    print(f"    {int(r['step']):>6} {float(r['bnm_nuclear_norm']):>9.4f} "
          f"{float(r['bnm_target_entropy']):>9.4f} "
          f"{float(r['bnm_pred_pos_frac']):>8.4f} {float(r['nuclear_gap']):>+9.4f}")

assert cfg["steps_completed"] == 8000, "run did not reach the cap"
assert len(full) == 32, f"expected 32 full checkpoints, got {len(full)}"
assert not missing, "some per-checkpoint logits are absent"
assert inv["held"] is True, "THE GAMMA INVARIANT DID NOT HOLD -- do not use this run"
print("\nALL CHECKS PASSED -- ready to package")
```

---

## Cell 5 — package for download

Core first (small: logits, CSVs, configs). Checkpoint weights ride in separate tars and are
only needed if a later stage wants them.

```python
import subprocess, sys

r = subprocess.run(
    [sys.executable, "-u", "scripts/build_dann_r5_bundle.py",
     "--pack-results", "/kaggle/working/runs/bnm_r6_dev8000_fold0",
     "--results-out", "/kaggle/working/results",
     "--logs", "/kaggle/working/bnm_r6_dev8000.log"],
    cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-6000:])
print(r.stderr[-2000:])
```

Download **`screen_core.tar.gz` and `MANIFEST.json` first** — that is all the analysis needs.

---

## Then, locally

```bash
python scripts/analyze_coral_dev8000_budget.py --run-dir runs/bnm_r6_dev8000_fold0 --arm bnm_nb3_fold0 --cap 8000 --out runs/bnm_r6_budget.json
```

That is the **Task-A decay detector, applied unmodified** — same definition, same
shared-resample paired bootstrap, B = 5000, seed 0, percentile interval, α = 0.05, on the
**250-grid points only**. It is fully parameterised (`--run-dir`, `--arm`, `--cap`, `--out`),
so nothing about it is edited for BNM; that is what keeps Δ comparable to CORAL's +0.0278 and
FreeMatch's +0.0951.

I will report: Δ with CI; the budget the rule returns and which of the four named outcomes
fired; the 704/78 trajectory across the full run scored against the pre-registered prediction
and its falsifier; the γ invariant's observation count and range; whether the advisory ceiling
was ever tripped; and confirmation that per-checkpoint logits came home.

**Nothing about ranking is concluded from fold 0.**
