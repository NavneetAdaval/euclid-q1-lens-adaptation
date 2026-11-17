# Arm 1 on Kaggle — the λ=0 control

**What this is.** Round 5 tests DANN (domain-adversarial training) with the top 3 encoder
blocks unfrozen. Arm 1 is the **control**: the exact same model and the exact same
discriminator, with the adversarial coefficient λ pinned to zero. Nothing about DANN is
being tested here. It exists to be the **reference trajectory** — the thing arm 2 gets
compared against at matched steps, so that "unfreezing moved this" can be subtracted from
"the adversary moved this."

**What it produces.** An 8000-step run on fold 0, writing per-step monitors, 32
checkpoints, and a `steps.csv`. Roughly **3–5 hours on a T4**.

**What you do with the output.** Cell 6 prints a summary block — `J`, `J_early`, the IQR,
`m_min_admissible` and a `M_DECLARABLE` verdict. **Send that block back.** It is the input
to declaring **M**, the margin for abort condition 3a, which gets hashed before arm 2
starts and does not move afterwards. Cell 7 packages the run for download.

**Nothing here needs a decision from you.** Every flag is fixed. If a cell aborts, the
message says what to do; the failure notes at the bottom cover the four things that
realistically go wrong.

---

## Before you start

1. Sync `C:\Users\krris\ML\lens_paper\kaggle\dann_r5\` to the Kaggle dataset
   **`lens-paper-dann-r5`**.
2. New notebook → **Accelerator: GPU T4 ×2** (only one is used; T4 ×2 is what the quota
   gives) → **Add Data** → your `lens-paper-dann-r5` dataset.
3. Paste the cells below in order.

> **Re-uploading creates a new dataset VERSION, and Kaggle can serve you the old one.**
> Cell 2 prints `VERSION.txt` from the mount. If that line does not match the version you
> just uploaded, you are on a stale mount — detach and re-add the dataset. This is why the
> version is printed rather than assumed.

---

## Cell 1 — environment check (~10 s)

Confirms a GPU is present and is a T4, and that there is disk room. Fails immediately and
legibly if not, rather than three hours in.

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
    print("!!! PROJECT.md 7.7's wall-clock estimates are T4-specific, and it also records")
    print("!!! that results are NOT portable across GPU architectures at this precision.")
    print("!!! The run will still work. The TIMINGS below will not, and any comparison")
    print("!!! against a T4-run arm needs that disclosed.")
    print("!" * 78 + "\n")
else:
    print("\n  T4 confirmed -- the venue PROJECT.md 7.7's timings are measured on.")

free = shutil.disk_usage("/kaggle/working").free / 2**30
print(f"  /kaggle/working free: {free:.1f} GiB")
assert free > 8, f"only {free:.1f} GiB free; the run writes ~2 GB of checkpoints"
print("  CPU cores:", os.cpu_count())
```

---

## Cell 2 — mount, stage, verify (~1–2 min)

Copies **only `scripts/`** into `/kaggle/working` and verifies it against the bundle's
MANIFEST. The 4.6 GB of data stays in `/kaggle/input` and is **read in place** — copying it
would burn most of the working directory and several minutes for no benefit. Prints the
mounted dataset version and verifies the pre-registration's digest. **Aborts on any
mismatch**, so a bad or stale mount cannot reach training.

```python
import os, sys, glob, json, shutil, hashlib

SLUG = "lens-paper-dann-r5"
CANDIDATES = sorted(glob.glob(f"/kaggle/input/*{SLUG}*")) or sorted(glob.glob("/kaggle/input/*"))
assert CANDIDATES, "no dataset mounted -- Add Data -> lens-paper-dann-r5"
INPUT = CANDIDATES[0]
print("mount:", INPUT)

ver = os.path.join(INPUT, "VERSION.txt")
print("VERSION:", open(ver).read().strip() if os.path.isfile(ver) else "*** MISSING ***")
print("  ^ if this is not the version you just uploaded, you are on a STALE MOUNT.")
print("    Detach the dataset and re-add it. Kaggle serves the previous version silently.")

man = json.load(open(os.path.join(INPUT, "MANIFEST.json")))
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

sys.path.insert(0, f"{WORK}/scripts")
os.environ["INPUT_ROOT"] = INPUT
print("\nOK -- mount verified.")
```

---

## Cell 3 — preflight, the minute-2 check (~3–5 min)

Builds the whole arm — encoder, discriminator, data, morphology pool — and **stops before
training**. This is §7.7's "a second rung validates at minute 2" discipline: a bad bundle
must cost two minutes, not three hours. Prints the fold-gate values it will enforce and
the step-0 morphology readings. **Aborts on mismatch.**

```python
import os, json, subprocess, sys
INPUT = os.environ["INPUT_ROOT"]

r = subprocess.run([sys.executable, "-u", "scripts/run_dann_r5.py",
                    "--arm", "dann_nb3_lambda0", "--fold", "0", "--morph-every", "25",
                    "--input-root", INPUT,
                    "--emit-config", "/kaggle/working/arm1_ref.json"],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-6000:])
if r.returncode != 0:
    print(r.stderr[-4000:]); raise SystemExit("PREFLIGHT FAILED -- do not start the run")

cfg = json.load(open("/kaggle/working/arm1_ref.json"))
print("\n" + "=" * 78)
print("WHAT THIS RUN WILL DO")
print("=" * 78)
for k in ("arm", "fold", "max_steps", "abort_mode", "abort_mode_source",
          "dann_lambda_zero", "dann_lambda_horizon", "morph_every",
          "abort_cond3_column", "abort_cond3_margin", "batch_size", "grad_checkpointing"):
    print(f"  {k:<28}: {cfg.get(k)}")
print(f"  {'step-0 gate (fold 0)':<28}: TPR {cfg['expect_step0_tpr']:.6f} "
      f"(tol {cfg['step0_tpr_tolerance']:.6f})  AUROC {cfg['expect_step0_auroc']:.6f} "
      f"(tol {cfg['step0_auroc_tolerance']:.0e})")
print(f"  {'phase-1 head ||.||_2':<28}: {cfg['init_head_l2']} "
      f"(pinned {cfg['expect_head_l2']})")
print(f"  {'D holdout':<28}: {cfg['dann_d_holdout_n']} of the target pool")

assert cfg["abort_mode"] == "log_only", "arm 1 must be log_only -- it is the reference"
assert cfg["abort_cond3_margin"] is None, "M must be UNSET before arm 1"
assert cfg["dann_lambda_zero"] is True, "arm 1 must have lambda pinned to zero"
assert cfg["max_steps"] == 8000 and cfg["morph_every"] == 25

m0 = json.load(open(os.path.join(INPUT, "runs/dann_r5_morph_step0.json")))
rd = m0["readings_step0"]["dann_nb3_lambda0"]
print(f"\n  step-0 morphology, HEAD  : {rd['morph_head_auroc_pool']:.6f} (whole pool, "
      f"what the detectors read) / {rd['morph_head_auroc']:.6f} (eval half)")
print(f"  step-0 morphology, PROBE : {rd['morph_proxy_auroc']:.6f} (adapter) / "
      f"{rd['morph_proxy_auroc_encoder']:.6f} (raw encoder)")
print("\nPREFLIGHT OK -- safe to start Cell 4.")
```

---

## Cell 4 — arm 1, the training run (~170–300 min)

The real run. Unbuffered and **tee'd to a log inside `/kaggle/working`**, so a browser
disconnect loses nothing. Prints a heartbeat that keeps the notebook session alive.

> **Check the clock before starting.** 8000 steps at n_blocks=3 measured **~170–300 min**
> on a T4 (§7.7: CORAL's equivalent arm was 298 min). Kaggle's session limit is **12 h**,
> so it fits with room — but if you are already several hours into this session, start
> fresh instead.

```python
import subprocess, sys, os, time, threading
INPUT = os.environ["INPUT_ROOT"]
OUT = "/kaggle/working/runs/dann_r5_nb3_lambda0_fold0"
LOG = "/kaggle/working/arm1.log"

print(f"expected wall time : 170-300 min on a T4 (8000 steps, n_blocks=3)")
print(f"Kaggle session cap : 12 h")
print(f"log file           : {LOG}   (survives a browser disconnect)")
print(f"output dir         : {OUT}\n")

cmd = [sys.executable, "-u", "scripts/run_dann_r5.py",
       "--arm", "dann_nb3_lambda0", "--fold", "0", "--morph-every", "25",
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
the step and a declared constant horizon), **dropout carries none**, and **D's weights ride
in the checkpoint's `schedule` payload** — a checkpoint without them is *refused* rather
than silently restarting D from a fresh init. AdamW's moments, the GradScaler and the RNG
are all restored. The one honest gap: augmentation offsets drawn *after* a resume are not
guaranteed to match an uninterrupted run's (a ±3px translate); every resume is recorded in
`run.json` so a resumed curve is never mistaken for an unresumed one.

```python
import subprocess, sys, os, time, threading
INPUT = os.environ["INPUT_ROOT"]
OUT = "/kaggle/working/runs/dann_r5_nb3_lambda0_fold0"
LOG = "/kaggle/working/arm1_resume.log"

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
                          "--arm", "dann_nb3_lambda0", "--fold", "0",
                          "--morph-every", "25", "--input-root", INPUT, "--out-dir", OUT],
                         cwd="/kaggle/working", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        lf.write(line); lf.flush(); print(line, end="", flush=True)
    rc = p.wait()
stop.set()
print(f"\nexit code {rc} after {(time.time()-t0)/60:.1f} min")
```

---

## Cell 6 — the numbers to send back (~10 s)

**This is the deliverable.** Copy the whole printed block into your reply.

```python
import subprocess, sys, json, os, glob, csv
OUT = "/kaggle/working/runs/dann_r5_nb3_lambda0_fold0"

r = subprocess.run([sys.executable, "-u", "scripts/run_dann_r5.py",
                    "--morph-stats", f"{OUT}/steps.csv"],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr[-3000:])

rec = json.load(open(f"{OUT}/run_config.json"))
print("=" * 78)
print("RUN INTEGRITY")
print("=" * 78)
print(f"  steps completed   : {rec.get('steps_completed')} / {rec.get('max_steps')}")
print(f"  checkpoints       : {rec.get('n_checkpoints_saved')} of "
      f"{rec.get('n_checkpoints_expected')}   (must be 32 of 32)")
print(f"  aborted           : {rec.get('aborted')}   reason {rec.get('abort_reason')}")
print(f"  wall minutes      : {rec.get('wall_minutes'):.1f}" if rec.get('wall_minutes')
      else "  wall minutes      : n/a")
print(f"  fp16 steps skipped: {rec.get('fp16_skipped_cumulative')}")
print(f"  resumes           : {len(rec.get('resume_events') or [])}")
print(f"  step-0 gate       : {(rec.get('step0_gate') or {}).get('verdict')}")
w = rec.get("abort_watch") or {}
print(f"  abort conditions  : fired {w.get('fired')}   mode {w.get('abort_mode')}")

rows = list(csv.DictReader(open(f"{OUT}/steps.csv", newline="")))
def col(c):
    v = [float(x[c]) for x in rows if x.get(c) not in (None, "", "nan")]
    return v
for c in ("morph_head_auroc_pool", "morph_head_auroc", "morph_proxy_auroc"):
    v = col(c)
    if v:
        print(f"  {c:<24}: start {v[0]:.6f}  end {v[-1]:.6f}  min {min(v):.6f}  "
              f"max {max(v):.6f}  ({len(v)} readings)")
print("\nSEND THE WHOLE OUTPUT OF THIS CELL BACK. It is the input to declaring M.")
```

---

## Cell 7 — package for download (~5–10 min)

Tars the run into independent, self-verifiable parts with a MANIFEST carrying per-part
size and sha256. **Refuses to package an incomplete run** — if the retention guard says
fewer than 32 checkpoints, fix the run rather than archiving it.

```python
import subprocess, sys, os
OUT = "/kaggle/working/runs/dann_r5_nb3_lambda0_fold0"
DST = "/kaggle/working/dann_r5_arm1_out"

r = subprocess.run([sys.executable, "-u", "scripts/build_dann_r5_bundle.py",
                    "--pack-results", OUT, "--results-out", DST],
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
| **Cell 3 or 4: step-0 gate ABORT** | The fold-gate table was scored locally; this run is on a T4. §7.7 measured venue drift as **arm-specific** — CORAL nb3 was identical across venues. **Fix by re-scoring the table on the venue, never by widening the gate.** Send the gate's printed TPR/AUROC and stop. |
| **Session dies mid-run** | **Cell 5. Nothing else.** Do not delete the output directory — the resume needs it. |
| **CUDA out of memory** | Gradient checkpointing is **mandatory** at n_blocks=3 and is already on; §7.1a measured 4.76 GB with it and an OOM without. If it OOMs anyway, something is wrong with the bundle, not the batch size. **Do not reduce the batch — 256 is pinned by §7.4 and changing it invalidates the whole cross-method comparison.** Report it. |
| **Wall time far outside 170–300 min** | Say so before assuming it is fine. Much faster usually means fewer steps ran than you think; much slower usually means the GPU is not a T4 or the data is being read from the wrong place. Cell 6 prints the step count — check it against 8000. |
| **Cell 2: VERSION does not match** | Stale mount. Detach the dataset and re-add it. |
| **Cell 7: packaging refused** | The retention guard found fewer than 32 checkpoints. The run is incomplete — resume it (Cell 5) rather than archiving what is there. |

---

## What happens after this

1. You send back Cell 6's block.
2. **k is declared first, then M** — in that order, because a smoothing window chosen after
   a candidate M is on the table has been selected to make that M work. `M_DECLARABLE` may
   come back **false**, and that is a legitimate answer: 3a is then reported as
   uncalibrable at this pool size and condition 3b — the sign-flip test matched to the
   historical failure — carries the condition alone.
3. M (or the uncalibrable finding) is hashed.
4. **Arm 2**, the development run, with `--control-curve` pointed at this run's `steps.csv`.
