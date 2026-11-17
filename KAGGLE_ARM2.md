# Arm 2 on Kaggle — the development run

**What this is.** The same model as arm 1, on the same fold, for the same 8000 steps —
with the adversary **switched on**. λ ramps from 0 to ~1 over its first 2000 steps and
stays there. Arm 1 answered "what does unfreezing three encoder blocks do on its own?"
Arm 2 answers "what does the adversary add on top of that?"

**Everything is compared against arm 1 at matched steps.** Arm 1's `steps.csv` ships
inside the dataset and Cell 4 points `--control-curve` at it automatically. You do not
have to find it, download it, or upload it.

**What it produces.** 8000 steps on fold 0, per-step monitors, 32 checkpoints, a
`steps.csv`. Arm 1 took **267.8 min on a T4**; arm 2 does strictly more work per step (the
discriminator's backward pass is live rather than zeroed), so budget **~270–300 min**.

**Nothing here can abort your run.** Arm 2 is `log_only`. The abort detectors compute and
record, and they do not stop anything. That is deliberate: conditions 3 and 4 were
**re-specified from arm 1 after it ran**, so letting them act on arm 2 would mean deciding
a threshold and then judging a run by it. They bind on folds 1–4, which have not started.

**What you do with the output.** Cell 6 prints the deliverable block: arm 2 against arm 1
on every monitored column, plus **whether either re-specified condition would have fired**
had it been armed. **Send that whole block back.**

**Nothing here needs a decision from you.** Every flag is fixed.

---

## Before you start

1. Sync `C:\Users\krris\ML\lens_paper\kaggle\dann_r5\` to the Kaggle dataset (the same one
   arm 1 used). This is bundle **v2** — it adds arm 1's control curve, which arm 2 cannot
   run without.
2. New notebook → **Accelerator: GPU T4 ×2** (only one is used) → **Add Data** → that
   dataset.
3. Paste the cells below in order.

> **Re-uploading creates a new dataset VERSION, and Kaggle can serve you the old one.**
> Cell 2 prints `VERSION.txt` and **aborts if it is not v2 or later**, because a v1 mount
> has no control curve and arm 2 would fail four cells later with a less obvious message.

---

## Cell 1 — environment check (~10 s)

Identical to arm 1's. Confirms a GPU is present and is a T4, and that there is disk room.
Fails immediately and legibly rather than four hours in.

```python
import subprocess, sys, shutil, os
import torch

print("torch", torch.__version__, "| CUDA", torch.version.cuda,
      "| available", torch.cuda.is_available())
assert torch.cuda.is_available(), "NO GPU. Settings -> Accelerator -> GPU T4 x2."

n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory/2**30:.2f} GiB  sm_{p.major}{p.minor}")

name = torch.cuda.get_device_name(0)
if "T4" not in name:
    print("\n" + "!" * 78)
    print(f"!!! NOT A T4 -- this is a {name}.")
    print("!!! ARM 1 RAN ON A T4. Arm 2 is compared against it step for step, and")
    print("!!! PROJECT.md 7.7 records that results are NOT portable across GPU")
    print("!!! architectures at this precision. A cross-architecture difference would")
    print("!!! be indistinguishable from an adversarial one. STOP and get a T4.")
    print("!" * 78 + "\n")
else:
    print("\n  T4 confirmed -- the same architecture arm 1 ran on.")

free = shutil.disk_usage("/kaggle/working").free / 2**30
print(f"  /kaggle/working free: {free:.1f} GiB")
assert free > 8, f"only {free:.1f} GiB free; the run writes ~2 GB of checkpoints"
print("  CPU cores:", os.cpu_count())
```

---

## Cell 2 — mount, stage, verify (~1–2 min)

Copies **only `scripts/`** into `/kaggle/working` and verifies it against the bundle's
MANIFEST. The 4.6 GB of data stays in `/kaggle/input` and is **read in place**.

The mount is found by **searching recursively for a `MANIFEST.json` whose `bundle` field
says `dann_r5`**, not by matching the dataset's name. Kaggle nests the mount differently
depending on how the dataset was added (`/kaggle/input/<slug>` in some sessions,
`/kaggle/input/datasets/<owner>/<slug>` in others), and the dataset is *not* renamed for
each round — so the name proves nothing and the manifest proves everything.

```python
import os, sys, glob, json, shutil, hashlib

# find the bundle by its MANIFEST, not by the dataset's name
found = []
for man_path in glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True):
    try:
        m = json.load(open(man_path))
    except Exception:
        continue
    if m.get("bundle") == "dann_r5":
        found.append((man_path, m))
assert found, ("no dann_r5 bundle found under /kaggle/input. Add Data -> the dataset "
               "built from kaggle/dann_r5/. Present instead: "
               + str(sorted(os.listdir('/kaggle/input'))))
found.sort(key=lambda t: -int(t[1].get("version", 0)))
MAN_PATH, man = found[0]
INPUT = os.path.dirname(MAN_PATH)
print("mount:", INPUT)
if len(found) > 1:
    print(f"  ({len(found)} dann_r5 mounts visible; using the highest version)")

ver = os.path.join(INPUT, "VERSION.txt")
print("VERSION:", open(ver).read().strip() if os.path.isfile(ver) else "*** MISSING ***")
assert int(man.get("version", 0)) >= 2, (
    f"this is bundle v{man.get('version')}. Arm 2 needs v2 or later -- v1 does not "
    f"contain arm 1's control curve. Detach the dataset and re-add it; if it is still "
    f"v1, the upload did not go through.")

print(f"bundle {man['bundle']} v{man['version']}  built {man['built_utc']}")
print(f"  {man['n_scripts']} scripts, import closure {len(man['import_closure'])} modules")

WORK = "/kaggle/working"
os.makedirs(f"{WORK}/scripts", exist_ok=True)
for f in man["scripts"]:
    shutil.copy2(os.path.join(INPUT, "scripts", f), f"{WORK}/scripts/{f}")
print(f"  copied {len(man['scripts'])} scripts to {WORK}/scripts (data stays in /kaggle/input)")

missing = [n for n in man["cell_requires"] if not os.path.isfile(f"{WORK}/scripts/{n}")]
assert not missing, f"MISSING SCRIPTS: {missing}"
clo = [m for m in man["import_closure"] if not os.path.isfile(f"{WORK}/scripts/{m}")]
assert not clo, f"IMPORT CLOSURE INCOMPLETE: {clo}"
print("  cell-required scripts present; import closure complete")

def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(n), b""):
            h.update(b)
    return h.hexdigest()

bad = []
for rel, info in man["checkpoints"].items():
    p = os.path.join(INPUT, rel)
    if not os.path.isfile(p):
        bad.append(f"{rel}: MISSING"); continue
    if info.get("sha256") and sha(p) != info["sha256"]:
        bad.append(f"{rel}: CHECKSUM MISMATCH")
assert not bad, "ARTIFACT VERIFICATION FAILED:\n  " + "\n  ".join(bad)
print(f"  {len(man['checkpoints'])} shipped artifacts verified by sha256")

pre = json.load(open(os.path.join(INPUT, "runs/dann_r5_prereg.json")))
got = pre.pop("sha256_of_body_without_this_field")
again = hashlib.sha256(json.dumps(pre, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert got == again, f"PRE-REGISTRATION DIGEST FAILED: {got} != {again}"
print(f"  pre-registration self-certifies: {got}")

CONTROL = os.path.join(INPUT, "runs/dann_r5_arm1_control/steps.csv")
assert os.path.isfile(CONTROL), f"arm 1's control curve is not in the mount: {CONTROL}"
a1 = json.load(open(os.path.join(INPUT, "runs/dann_r5_arm1_control/run_config.json")))
print(f"  control curve: arm 1, {a1.get('steps_completed')}/{a1.get('max_steps')} steps, "
      f"{a1.get('n_checkpoints_saved')} checkpoints, "
      f"{a1.get('wall_minutes', 0):.1f} min, gate "
      f"{(a1.get('step0_gate') or {}).get('verdict')}")

sys.path.insert(0, f"{WORK}/scripts")
os.environ["INPUT_ROOT"] = INPUT
os.environ["CONTROL_CURVE"] = CONTROL
print("\nOK -- mount verified.")
```

---

## Cell 3 — preflight, the minute-2 check (~3–5 min)

Builds the whole arm and **stops before training**. This is §7.7's "a second rung validates
at minute 2" discipline: a bad bundle must cost two minutes, not four hours.

Arm 2 does one thing arm 1 could not: it checks its own config **against arm 1's**, key by
key, over `PINNED_KEYS`. Anything that must not differ between the two arms — batch size,
the fold gate, the head init, the monitor cadences — is compared, and a mismatch aborts.
Two arms that differ in something other than λ are not a controlled comparison.

```python
import os, json, subprocess, sys
INPUT = os.environ["INPUT_ROOT"]; CONTROL = os.environ["CONTROL_CURVE"]

# arm 1's config, re-emitted here from the SAME construction code, as the reference
r0 = subprocess.run([sys.executable, "-u", "scripts/run_dann_r5.py",
                     "--arm", "dann_nb3_lambda0", "--fold", "0", "--morph-every", "25",
                     "--input-root", INPUT,
                     "--emit-config", "/kaggle/working/arm1_ref.json"],
                    cwd="/kaggle/working", capture_output=True, text=True)
if r0.returncode != 0:
    print(r0.stdout[-3000:]); print(r0.stderr[-3000:])
    raise SystemExit("could not re-emit arm 1's reference config")

r = subprocess.run([sys.executable, "-u", "scripts/run_dann_r5.py",
                    "--arm", "dann_nb3", "--fold", "0", "--morph-every", "25",
                    "--control-curve", CONTROL,
                    "--reference-config", "/kaggle/working/arm1_ref.json",
                    "--input-root", INPUT,
                    "--emit-config", "/kaggle/working/arm2_ref.json"],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-7000:])
if r.returncode != 0:
    print(r.stderr[-4000:]); raise SystemExit("PREFLIGHT FAILED -- do not start the run")

cfg = json.load(open("/kaggle/working/arm2_ref.json"))
print("\n" + "=" * 78)
print("WHAT THIS RUN WILL DO")
print("=" * 78)
for k in ("arm", "fold", "max_steps", "abort_mode", "abort_mode_source",
          "dann_lambda_zero", "dann_lambda_horizon", "morph_every",
          "abort_cond3_column", "abort_cond3_margin",
          "abort_cond4_src_margin", "abort_cond4_tgt_margin",
          "batch_size", "grad_checkpointing"):
    print(f"  {k:<28}: {cfg.get(k)}")
print(f"  {'control curve':<28}: {cfg.get('control_curve')}")
print(f"  {'step-0 gate (fold 0)':<28}: TPR {cfg['expect_step0_tpr']:.6f} "
      f"(tol {cfg['step0_tpr_tolerance']:.6f})  AUROC {cfg['expect_step0_auroc']:.6f} "
      f"(tol {cfg['step0_auroc_tolerance']:.0e})")
print(f"  {'phase-1 head ||.||_2':<28}: {cfg['init_head_l2']} "
      f"(pinned {cfg['expect_head_l2']})")

assert cfg["abort_mode"] == "log_only", (
    "arm 2 must be log_only. Conditions 3 and 4 were re-specified FROM arm 1 after it "
    "ran; arming them on arm 2 would mean judging a run by thresholds chosen after "
    "seeing its control.")
assert cfg["dann_lambda_zero"] is False, "arm 2 is the ADVERSARIAL arm -- lambda is live"
assert cfg["dann_lambda_horizon"] == 2000, "lambda horizon is a declared constant, not the run length"
assert cfg["max_steps"] == 8000 and cfg["morph_every"] == 25
assert cfg.get("control_curve"), "no control curve -- condition 3 would be INACTIVE"
assert cfg["abort_cond3_margin"] is None, "the margins stay UNSET until they are hashed"

# lambda's actual shape, so it is a curve you have seen and not a formula you trust
import importlib; sys.path.insert(0, "/kaggle/working/scripts")
from da_losses import dann_lambda_at_step
print("\n  lambda(step):", "  ".join(
    f"{s}:{dann_lambda_at_step(s, cfg['dann_lambda_horizon']):.3f}"
    for s in (0, 100, 250, 500, 1000, 2000, 4000, 8000)))

m0 = json.load(open(os.path.join(INPUT, "runs/dann_r5_morph_step0.json")))
rd = m0["readings_step0"]["dann_nb3"]
print(f"\n  step-0 morphology, PROBE : {rd['morph_proxy_auroc']:.6f} (adapter -- THIS is "
      f"what condition 3 reads) / {rd['morph_proxy_auroc_encoder']:.6f} (raw encoder)")
print(f"  step-0 morphology, HEAD  : {rd['morph_head_auroc_pool']:.6f} / "
      f"{rd['morph_head_auroc']:.6f}   (logged, NO detector attached)")
print("\nPREFLIGHT OK -- safe to start Cell 4.")
```

---

## Cell 4 — arm 2, the training run (~270–300 min)

The real run. Unbuffered and **tee'd to a log inside `/kaggle/working`**, so a browser
disconnect loses nothing. Prints a heartbeat that keeps the notebook session alive.

> **Check the clock before starting.** Arm 1 measured **267.8 min**; arm 2 runs the
> discriminator's backward pass for real, so expect **270–300 min**. Kaggle's session limit
> is **12 h**, so it fits with room — but if you are already several hours into this
> session, start fresh instead.

```python
import subprocess, sys, os, time, threading
INPUT = os.environ["INPUT_ROOT"]; CONTROL = os.environ["CONTROL_CURVE"]
OUT = "/kaggle/working/runs/dann_r5_nb3_fold0"
LOG = "/kaggle/working/arm2.log"

print(f"expected wall time : 270-300 min on a T4 (arm 1 took 267.8)")
print(f"Kaggle session cap : 12 h")
print(f"log file           : {LOG}   (survives a browser disconnect)")
print(f"output dir         : {OUT}")
print(f"control curve      : {CONTROL}\n")

cmd = [sys.executable, "-u", "scripts/run_dann_r5.py",
       "--arm", "dann_nb3", "--fold", "0", "--morph-every", "25",
       "--abort-mode", "log_only",
       "--control-curve", CONTROL,
       "--reference-config", "/kaggle/working/arm1_ref.json",
       "--input-root", INPUT, "--out-dir", OUT]

t0 = time.time()
stop = threading.Event()
def heartbeat():
    while not stop.wait(300):
        print(f"    ... still running, {(time.time()-t0)/60:.0f} min elapsed", flush=True)
threading.Thread(target=heartbeat, daemon=True).start()

with open(LOG, "w") as lf:
    p = subprocess.Popen(cmd, cwd="/kaggle/working", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        lf.write(line); lf.flush()
        print(line, end="", flush=True)
    rc = p.wait()
stop.set()

print(f"\nexit code {rc} after {(time.time()-t0)/60:.1f} min")
if rc != 0:
    print("RUN FAILED -- read the last lines above, then see the failure notes at the "
          "bottom of this document.")
```

*If the session dies part-way through, go to Cell 5 — no other action.*

---

## Cell 5 — resume (safe to run whether or not a resume is needed)

Same command. The runner **detects** an existing run and resumes from `resume/latest.pt`;
if there is nothing to resume it starts clean and says so. Nothing to edit.

Resume is exact where it matters: **λ carries no hidden state** (it is a pure function of
the step and a declared constant horizon — which is exactly why the horizon is 2000 and not
`total_steps`; a run resumed at a different total would otherwise resume onto a different
λ curve), **dropout carries none**, and **D's weights ride in the checkpoint's `schedule`
payload** — a checkpoint without them is *refused* rather than silently restarting D from a
fresh init. AdamW's moments, the GradScaler and the RNG are all restored. The one honest
gap: augmentation offsets drawn *after* a resume are not guaranteed to match an
uninterrupted run's (a ±3px translate); every resume is recorded in `run.json` so a resumed
curve is never mistaken for an unresumed one.

```python
import subprocess, sys, os, time, threading
INPUT = os.environ["INPUT_ROOT"]; CONTROL = os.environ["CONTROL_CURVE"]
OUT = "/kaggle/working/runs/dann_r5_nb3_fold0"
LOG = "/kaggle/working/arm2_resume.log"

rp = f"{OUT}/resume/latest.pt"
if os.path.isfile(rp):
    import torch
    sch = (torch.load(rp, map_location="cpu", weights_only=False).get("schedule") or {})
    print(f"RESUMING from step {sch.get('global_step')} (saved {sch.get('saved_utc')})")
    print(f"  discriminator state present: {bool(sch.get('discriminator_state'))}")
    print(f"  lambda horizon in checkpoint: {sch.get('dann_lambda_horizon')}")
else:
    print("No resume checkpoint found -- this will start a FRESH run.")

t0 = time.time(); stop = threading.Event()
def hb():
    while not stop.wait(300):
        print(f"    ... {(time.time()-t0)/60:.0f} min elapsed", flush=True)
threading.Thread(target=hb, daemon=True).start()

with open(LOG, "w") as lf:
    p = subprocess.Popen([sys.executable, "-u", "scripts/run_dann_r5.py",
                          "--arm", "dann_nb3", "--fold", "0",
                          "--morph-every", "25", "--abort-mode", "log_only",
                          "--control-curve", CONTROL,
                          "--reference-config", "/kaggle/working/arm1_ref.json",
                          "--input-root", INPUT, "--out-dir", OUT],
                         cwd="/kaggle/working", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        lf.write(line); lf.flush(); print(line, end="", flush=True)
    rc = p.wait()
stop.set()
print(f"\nexit code {rc} after {(time.time()-t0)/60:.1f} min")
```

---

## Cell 6 — the numbers to send back (~20 s)

**This is the deliverable.** Copy the whole printed block into your reply.

It reports three things:

1. **Run integrity** — did all 8000 steps run, all 32 checkpoints land, did the gate pass.
2. **Arm 2 against arm 1, on the readings only.** `steps.csv` has 8000 rows but morphology
   is measured every 25 steps and **forward-filled in between**; averaging all 8000 rows
   counts each reading 25 times and reports a jitter 25× too small. This cell recovers the
   measurement grid the same way `--morph-stats` does and works only on real readings.
3. **Would either re-specified condition have fired?** Arm 2 is `log_only`, so nothing
   acted. This replays the two differential detectors offline across every margin in each
   admissible band and reports the answer as a function of the margin — because the margins
   are not declared yet, and a single number here would be one chosen after seeing the run.

```python
import subprocess, sys, json, os, csv
import numpy as np
sys.path.insert(0, "/kaggle/working/scripts")
import dann_round5 as D5

OUT = "/kaggle/working/runs/dann_r5_nb3_fold0"
CONTROL = os.environ["CONTROL_CURVE"]

# ---- 1. run integrity -------------------------------------------------------
rec = json.load(open(f"{OUT}/run_config.json"))
print("=" * 78); print("RUN INTEGRITY  (arm 2, dann_nb3, fold 0)"); print("=" * 78)
print(f"  steps completed   : {rec.get('steps_completed')} / {rec.get('max_steps')}")
print(f"  checkpoints       : {rec.get('n_checkpoints_saved')} of "
      f"{rec.get('n_checkpoints_expected')}   (must be 32 of 32)")
print(f"  aborted           : {rec.get('aborted')}   reason {rec.get('abort_reason')}")
print("  wall minutes      : "
      + (f"{rec['wall_minutes']:.1f}  (arm 1: 267.8)" if rec.get("wall_minutes") else "n/a"))
print(f"  fp16 steps skipped: {rec.get('fp16_skipped_cumulative')}")
print(f"  resumes           : {len(rec.get('resume_events') or [])}")
print(f"  step-0 gate       : {(rec.get('step0_gate') or {}).get('verdict')}")
w = rec.get("abort_watch") or {}
print(f"  abort watch       : mode {w.get('abort_mode')}, fired {w.get('fired')}")
print(f"  control curve     : {rec.get('control_curve')}")

# ---- the measurement grid, recovered from value change-points ---------------
def readings(path, col, morph_every=25):
    rows = list(csv.DictReader(open(path, newline="")))
    st = [int(r["step"]) for r in rows]
    v = [float(r[col]) if r.get(col) not in (None, "", "nan") else float("nan")
         for r in rows]
    ch = [st[i] for i in range(1, len(v)) if v[i] != v[i - 1] and v[i] == v[i]]
    if len(ch) < 3:
        return st, v
    stride = int(np.bincount(np.diff(ch)).argmax())
    off = ch[0] % stride if stride else 0
    g = [(s, x) for s, x in zip(st, v) if stride and (s - off) % stride == 0]
    return [s for s, _ in g], [x for _, x in g]

print("\n" + "=" * 78)
print("ARM 2 vs ARM 1 -- ON THE READINGS ONLY (not the forward-filled rows)")
print("=" * 78)
COLS = ("morph_proxy_auroc",           # <- condition 3's DETECTOR column
        "src_participation_ratio",     # <- condition 4 src
        "tgt_participation_ratio",     # <- condition 4 tgt
        "morph_head_auroc_pool",       # <- logged, NO detector
        "morph_head_auroc",            # <- logged, NO detector
        "morph_proxy_auroc_encoder")
print(f"  {'column':<28} {'arm2 start':>10} {'arm2 end':>10} {'arm2 med':>10} "
      f"{'arm1 med':>10} {'delta':>9} {'n':>5}")
paired = {}
for c in COLS:
    try:
        s2, v2 = readings(f"{OUT}/steps.csv", c)
        s1, v1 = readings(CONTROL, c)
    except KeyError:
        print(f"  {c:<28}  (not present)"); continue
    if not v2:
        print(f"  {c:<28}  (no readings)"); continue
    m1 = dict(zip(s1, v1))
    pr = [(s, a - m1[s]) for s, a in zip(s2, v2) if s in m1]
    paired[c] = pr
    d = np.median([x for _, x in pr]) if pr else float("nan")
    print(f"  {c:<28} {v2[0]:10.4f} {v2[-1]:10.4f} {np.median(v2):10.4f} "
          f"{np.median(v1):10.4f} {d:+9.4f} {len(v2):5d}")
    if len(pr) != len(v2):
        print(f"      *** {len(v2)-len(pr)} of arm 2's readings have NO arm-1 row at the "
              f"same step; those comparisons are REFUSED, not interpolated")

# ---- 3. would either re-specified condition have fired? ---------------------
print("\n" + "=" * 78)
print("WOULD THE RE-SPECIFIED DETECTORS HAVE FIRED?  (arm 2 was log_only -- none acted)")
print("=" * 78)
print("  Reported as a function of the margin across each ADMISSIBLE BAND, because the")
print("  margins are not declared yet. A single number here would be one picked after")
print("  seeing this run.\n")

def worst_streak(pairs, margin, sustain=500):
    """Longest run, IN ABSOLUTE STEPS, of (arm2 - arm1) < -margin."""
    run = best = 0; prev = None
    for s, d in pairs:
        if d < -margin:
            run += 0 if prev is None else (s - prev)
            best = max(best, run)
        else:
            run = 0
        prev = s
    return best

for c, lo, hi, label in (
        ("morph_proxy_auroc", 0.023438, D5.COND3_M_CEILING, "cond 3a  (M)"),
        ("src_participation_ratio", 0.517530, D5.COND4_SRC_CEILING, "cond 4 src"),
        ("tgt_participation_ratio", 1.326254, D5.COND4_TGT_CEILING, "cond 4 tgt")):
    pr = paired.get(c) or []
    print(f"  {label}   band [{lo:.4f}, {hi:.4f}]   sustain {D5.COND4_SUSTAIN} steps")
    if not pr:
        print("    no paired readings\n"); continue
    worst = min(d for _, d in pr)
    print(f"    deepest excursion below the control: {worst:+.4f} "
          f"at step {min(pr, key=lambda t: t[1])[0]}")
    fires = []
    for m in np.linspace(lo, hi, 9):
        st = worst_streak(pr, m)
        fires.append((m, st, st >= D5.COND4_SUSTAIN))
    for m, st, f in fires:
        print(f"      margin {m:8.4f}  longest streak {st:5d} steps  "
              f"{'WOULD FIRE' if f else 'no'}")
    n_fire = sum(1 for _, _, f in fires if f)
    print(f"    -> fires at {n_fire} of 9 margins across the band "
          f"({'THE WHOLE BAND' if n_fire == 9 else 'NOWHERE IN THE BAND' if n_fire == 0 else 'PART OF THE BAND'})\n")

# 3b is a floor, not a margin -- it has one answer
pv = readings(f"{OUT}/steps.csv", "morph_proxy_auroc")[1]
below = [x for x in pv if x < D5.COND3_SIGN_FLIP]
print(f"  cond 3b  floor {D5.COND3_SIGN_FLIP}: arm 2 minimum {min(pv):.4f}, "
      f"{len(below)} of {len(pv)} readings below the floor -> "
      f"{'WOULD FIRE' if below else 'no'}")

# ---- arm 2's own J, on the column the detector reads ------------------------
print("\n" + "=" * 78)
r = subprocess.run([sys.executable, "-u", "scripts/run_dann_r5.py",
                    "--morph-stats", f"{OUT}/steps.csv"],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr[-3000:])

print("\nSEND THE WHOLE OUTPUT OF THIS CELL BACK.")
```

---

## Cell 7 — package for download (~5–10 min)

Tars the run into independent, self-verifiable parts with a MANIFEST carrying per-part size
and sha256. **Refuses to package an incomplete run** — if the retention guard says fewer
than 32 checkpoints, fix the run rather than archiving it.

```python
import subprocess, sys, os
OUT = "/kaggle/working/runs/dann_r5_nb3_fold0"
DST = "/kaggle/working/dann_r5_arm2_out"

r = subprocess.run([sys.executable, "-u", "scripts/build_dann_r5_bundle.py",
                    "--pack-results", OUT, "--results-out", DST,
                    "--logs", "/kaggle/working/arm2.log"],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-6000:])
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit("PACKAGING REFUSED -- see the message above")

tot = 0
for f in sorted(os.listdir(DST)):
    p = os.path.join(DST, f); sz = os.path.getsize(p); tot += sz
    print(f"  {f:<44} {sz/1e6:8.1f} MB")
print(f"\n  {DST}   TOTAL {tot/1e9:.2f} GB")
print("  Download every part from the Output panel. Each part is self-contained --")
print("  a part that fails to download can be re-fetched on its own.")
```

---

## If something goes wrong

| what you see | what to do |
|---|---|
| **Cell 2: no dann_r5 bundle found** | The dataset is not attached, or the upload did not finish. The cell prints what *is* under `/kaggle/input` — send that. Note it searches by MANIFEST, not by dataset name, so a dataset whose name still says an older round is fine. |
| **Cell 2: "this is bundle v1"** | Stale mount. v1 has no arm-1 control curve. Detach the dataset, re-add it, re-run Cell 2. If it is still v1, the upload did not go through. |
| **Cell 3: PINNED KEY MISMATCH** | Arm 2's config differs from arm 1's in something other than λ, so the two are **not** a controlled comparison. The message names the key. Send it and stop — do not "fix" it by removing the reference config. |
| **Cell 3 or 4: step-0 gate ABORT** | Send the printed TPR/AUROC and stop. §7.7 measured venue drift as **arm-specific**, and arm 1 passed this gate with |diff| exactly 0.0 — so a failure here is a real difference, not venue noise. **Fix by re-scoring the table on the venue, never by widening the gate.** |
| **Cell 4: "the control curve is missing N of the M steps"** | The two arms' morphology cadences disagree. Condition 3 compares step for step and **refuses to interpolate**. Both cells here pass `--morph-every 25`, the same value arm 1 used, so this should not happen — send the message rather than reaching for `--allow-control-gaps`, which does not fix it, it only silences it. |
| **Session dies mid-run** | **Cell 5. Nothing else.** Do not delete the output directory — the resume needs it. |
| **CUDA out of memory** | Gradient checkpointing is **mandatory** at n_blocks=3 and is already on; §7.1a measured 4.76 GB with it and an OOM without. Arm 1 peaked at 3.30 GB with the discriminator resident, so arm 2 has headroom. If it OOMs anyway, something is wrong with the bundle, not the batch size. **Do not reduce the batch — 256 is pinned by §7.4 and changing it invalidates the whole cross-method comparison.** Report it. |
| **Wall time far outside 270–300 min** | Say so before assuming it is fine. Much faster usually means fewer steps ran than you think; much slower usually means the GPU is not a T4. Cell 6 prints the step count — check it against 8000. |
| **Cell 6: "comparisons are REFUSED, not interpolated"** | Some of arm 2's readings have no arm-1 row at the same step. Those pairs are dropped rather than invented. A handful is survivable; a large number means the cadences drifted, so send the count. |
| **Cell 7: packaging refused** | The retention guard found fewer than 32 checkpoints. The run is incomplete — resume it (Cell 5) rather than archiving what is there. |

---

## What happens after this

1. You send back Cell 6's block.
2. **k is declared, then the three margins** — cond 3a's M within **[0.0234, 0.2481]**,
   cond 4 src within **[0.518, 2.592]**, cond 4 tgt within **[1.326, 5.608]** — and hashed.
   The order matters: a smoothing window chosen after a candidate margin is on the table has
   been selected to make that margin work.
3. **The budget is declared and hashed** into `runs/dann_r5_budget.json`. §7.13 and §7.14
   both establish that these curves do not locate a budget, so expect **"declared", not
   "derived"**.
4. **Arm 3** — folds 1–4, **armed**, against the declared margins.

> **One thing Cell 6 cannot tell you, and will not pretend to.** Arm 2 is a single fold. If
> it separates from arm 1 on the probe column, that is one fold's difference, not a result;
> §7.4's five-fold rule is what decides whether DANN did anything, and only arm 3 can
> answer it.
