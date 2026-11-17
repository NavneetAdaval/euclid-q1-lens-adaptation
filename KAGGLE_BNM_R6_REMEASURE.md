# Re-measure the step-0 gradient norms on T4/fp16 — PATH 2

**The original session is gone, so the swept grid's true provenance is unrecoverable.** This
is a **second, later measurement**. It is recorded as one, under its own filename, and it is
**not** the provenance of the γ grid that was swept.

**What this does NOT do — stated first, because it is the whole risk of path 2:**

- it does **not** re-derive the γ grid. `[0.00787, 0.039351, 0.157406]` was swept and stays.
- it does **not** re-select γ. **0.00787 stands.**
- it does **not** let the pre-registration claim verified provenance. The record says the
  original is lost.

`write_bnm_r6_prereg.py` asserts both of the first two on every run, so this cannot drift.

**Declared before you run this** (`MATERIAL_RATIO_DELTA = 0.05`): a re-measured BNM/CE ratio
differing from the back-computed **12.7062** by more than **5%** is *material* and becomes a
**within-architecture variance** data point for §7.7. Under 5% it corroborates the grid
without being its provenance. Same architecture, same precision, same seed, same step 0 — the
only expected disagreement is fp16 reduction-order nondeterminism, far below 5%. The number
is fixed now so it cannot be chosen after seeing the result.

**Needs the bundle at v18 or later.** ~3 minutes of GPU. No training run.

---

## Cell 1 — the accelerator must be a T4

A P100 or L4 makes this a *cross*-architecture comparison, which is a different measurement
than the one being made. Stop if this fails.

```python
import torch
print("cuda:", torch.cuda.is_available())
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("gpu :", name)
assert name and "T4" in name, (
    f"THIS MUST RUN ON A T4, got {name!r}. Settings -> Accelerator -> GPU T4 x2. "
    "The original measurement was on a T4; comparing against a different architecture "
    "measures something else entirely.")
print("OK -- T4 confirmed")
```

---

## Cell 2 — resolve the mount, stage, run the suites

Identical to Stage 2's Cell A, and self-sufficient: it resolves the mount by content, stages
every script and the stage-0a artifact, and refuses a bundle that predates round 6.

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

## Cell 3 — the re-measurement

`--measure-step0` does one forward/backward and **exits before any optimiser step**. It is
metric-blind by construction (`test_bnm_round6.T11`).

```python
import subprocess, sys, os, json

INPUT = os.environ["INPUT_ROOT"]
OUT = "/kaggle/working/bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json"

r = subprocess.run([sys.executable, "-u", "scripts/run_bnm_r6.py",
                    "--fold", "0", "--measure-step0",
                    "--input-root", INPUT, "--step0-out", OUT],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-4000:])
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit("RE-MEASUREMENT FAILED")

d = json.load(open(OUT))
assert str(d["device"]).startswith("cuda"), "not on GPU"
assert d["fp16"] is True, "not fp16"
assert d["exited_before_any_optimiser_step"] and d["reads_a_metric"] is False

BACK = 12.7062          # back-computed from the SWEPT grid
MATERIAL = 0.05         # declared before this ran

ratio = d["ratio_bnm_over_ce_at_gamma_1"]
delta = ratio / BACK - 1.0

print("=" * 76)
print("RE-MEASUREMENT (path 2) -- a SECOND, LATER measurement")
print("=" * 76)
print(f"  gpu                         {d['gpu_name']}   fp16={d['fp16']}")
print(f"  ||grad CE||                 {d['ce_grad_norm']:.6f}")
print(f"  ||grad BNM|| at gamma=1     {d['bnm_grad_norm_at_gamma_1']:.6f}")
print(f"  ratio BNM/CE                {ratio:.6f}")
print(f"  back-computed from sweep    {BACK:.4f}")
print(f"  delta                       {100 * delta:+.2f}%   "
      f"({'MATERIAL' if abs(delta) > MATERIAL else 'not material'}, "
      f"threshold {int(100 * MATERIAL)}%)")
print()
print("  THE GRID THIS FILE IMPLIES (recorded, NEVER used):")
for f, g in zip(d["fractions"], d["gamma_grid"]):
    print(f"    {f * 100:>5.0f}%  ->  gamma = {g:.6g}")
print(f"  THE GRID THAT WAS SWEPT AND STAYS: [0.00787, 0.039351, 0.157406]")
print(f"  gamma REMAINS 0.00787.")
print()
print("SEND THIS WHOLE BLOCK BACK, and download the JSON below.")
print(f"\n  file: {OUT}  ({os.path.getsize(OUT)} bytes)")
```

---

## Cell 4 — bring it home

Download the file from the notebook's **Output** panel, or copy the JSON this prints, and
save it locally as:

    C:\Users\krris\ML\lens_paper\runs\bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json

```python
print(open("/kaggle/working/bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json").read())
```

Then, locally:

```bash
python scripts/write_bnm_r6_prereg.py --final
```

It prints the sha256 for PROJECT.md. The provenance block will record that the original was
**LOST**, that this is a second measurement, the delta, and whether it was material.
