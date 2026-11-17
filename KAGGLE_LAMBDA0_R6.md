# λ=0 baseline, folds 1–4, on Kaggle — round 6

**What this is.** The missing cell in the paper's main table: **unfrozen encoder, no
domain-adaptation term.** Every DA number in the nb3 table is currently compared against
*frozen* Baseline A at 0.176 while the DA arms are unfrozen, so two things differ across
that comparison — the encoder was unfrozen, *and* a DA term was added — and nothing in the
current data attributes the 0.176 → ~0.42 movement to either one. This run separates them.

It is round 5's **arm 1** extended from fold 0 to folds 1–4. λ is pinned to 0, so the
discriminator sits in the graph and contributes no gradient; what remains is supervised
training on source with the top three encoder blocks unfrozen.

**No new loss and no new code path.** The same runner, the same arm, four different folds.

| | encoder | DA term | TPR@1e-3 |
|---|---|---|---|
| Baseline A | frozen | none | 0.176 |
| **this run** | **unfrozen (nb3)** | **none** | **fold 0 only so far: 0.1531** |
| CORAL / FreeMatch / DANN × nb3 | unfrozen | yes | 0.4348 / 0.4169 / 0.4167 |

**What it produces.** Four 4000-step runs, 16 checkpoints each, per-step monitors, and —
this time — the **per-checkpoint logits**, which round 5's arm 1 never brought home.
~153 min/fold on a T4, ~10.2 GPU-h total. **Two sessions, two folds each.**

**Nothing here needs a decision from you.** Budget, gates and conditions are all fixed in
`runs/lambda0_r6_prereg.json`, sha256
**`c1694a479eafb040cf7ad108e554047a95811cde8ce646444a448e8b1d717a25`**, written before any
fold ran. You edit exactly one line: `FOLD`.

---

## What is already settled, so you do not have to think about it

**The budget is 4000, under a rule that reads no metric:** *the control is read where its
treatment is read* — 4000 is DANN × nb3's declared budget. Arms 1 and 2 share an
initialisation, a fold, a seed, a GPU and a step grid and differ in λ alone; reading the
control at a different step than its treatment would introduce a second difference and
destroy the only fully controlled contrast in the project. It is **not** a claim that 4000
is optimal for this arm — fold 0 peaks at step 250 — and "optimal" was never available
(§7.13).

**Sensitivity readouts at 1500 and 2000 are free** — checkpoints 6 and 8 of 16 — and give
the budget-matched comparators against FreeMatch (1500) and CORAL (2000). The
11-component schedule audit was **re-run on the current bytes** and every component is flat
or absolute-step pinned, which is what makes those readouts valid rather than merely
convenient.

**Three advisory conditions will fire, and their firing means nothing.** This was checked
before the pre-registration was written, not discovered afterwards:

| condition | why it fires | what it means |
|---|---|---|
| `3b_sign_flip` on the **head** column | `morph_head_auroc_pool` is 0.5657 at true step 0 but **0.4356 at step 1**, and the detector is `< 0.5` | it fires at step 1 on a run with **no adversary at all** |
| `4_rank_collapse`, src **and** tgt | participation ratio runs **2.03–9.69** across all 8000 steps of arm 1, entirely below the declared threshold of **20** of 640 | fires on **100%** of a healthy control's steps; it has never had discriminative power here |

Both are properties of the thresholds, not of your run. **Every condition on this arm is
advisory** — `run_dann_r5.default_abort_mode` returns `log_only` for every fold of a
λ=0 arm, so nothing can stop a run. The one detector in this family with real headroom is
`3b` on the **probe** column (`morph_proxy_auroc < 0.7`): 0.9202 at step 0, longest
sub-threshold streak **0** over 8000 steps. That is the one to watch.

---

## Before you start

1. Sync `C:\Users\krris\ML\lens_paper\kaggle\dann_r5\` to the Kaggle dataset you have
   been using since round 4 — it is still named **`lens-paper-coral-r4`**, a historical
   name; the bundle *inside* it is `dann_r5`. Round 5's arm 1 ran from
   `/kaggle/input/datasets/<owner>/lens-paper-coral-r4/`, and
   `runs/dann_r5_arm1_control/run_config.json` records that path. The bundle is at
   **v14** and carries the three round-6 artifacts (`lambda0_r6_prereg.json`,
   `lambda0_r6_audit.json`, `lambda0_r6_step0_conditions.json`).
   *(`kaggle/dann_r5/dataset-metadata.json` says `lens-paper-dann-r5`. That file is read
   only by the Kaggle CLI and has never matched the dataset actually in use — ignore it
   when uploading through the web UI.)*
2. New notebook → **Accelerator: GPU T4 ×2** → **Add Data** → that dataset.
3. Paste the cells below in order.

> **Re-uploading creates a new dataset VERSION, and your notebook stays pinned to the
> one it was attached with.** Pushing a new version does *not* move an already-attached
> notebook onto it — this is the single most common failure here, and it is not a
> problem with your upload. Cell 2 checks for the three round-6 **artifacts** (not for a
> version label) and prints `VERSION.txt` either way. If it refuses, run **Cell 2a**,
> which names the cause. The usual fix is: Add Data panel → remove the dataset
> (`lens-paper-coral-r4`) → add it again.

**Session plan.** Session A: folds **1** then **2**. Session B: folds **3** then **4**.
Two folds is ~5.1 h against Kaggle's 12 h cap, which leaves room for one resume.

---

## Cell 1 — environment check (~10 s)

```python
import shutil, os
import torch

print("torch", torch.__version__, "| CUDA", torch.version.cuda,
      "| available", torch.cuda.is_available())
assert torch.cuda.is_available(), "NO GPU. Settings -> Accelerator -> GPU T4 x2."

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory/2**30:.2f} GiB  sm_{p.major}{p.minor}")

name = torch.cuda.get_device_name(0)
if "T4" not in name:
    print("\n" + "!" * 78)
    print(f"!!! NOT A T4 -- this is a {name}.")
    print("!!! PROJECT.md 7.7 records that results are NOT portable across GPU")
    print("!!! architectures at this precision. DANN's folds 1-4 ran on T4. Running")
    print("!!! this control on anything else puts a GPU difference inside the one")
    print("!!! fully controlled contrast in the project. STOP and switch to a T4.")
    print("!" * 78 + "\n")
else:
    print("\n  T4 confirmed -- the same architecture DANN's folds 1-4 ran on.")

free = shutil.disk_usage("/kaggle/working").free / 2**30
print(f"  /kaggle/working free: {free:.1f} GiB")
assert free > 8, f"only {free:.1f} GiB free; two folds write ~2 GB of checkpoints"
print("  CPU cores:", os.cpu_count())
```

---

## Cell 2 — mount, stage, verify (~1–2 min)

Copies **only `scripts/`** into `/kaggle/working`; the 4.6 GB of data stays in
`/kaggle/input` and is read in place. Verifies both pre-registrations by digest and
**aborts on any mismatch**, so a bad or stale mount cannot reach training.

```python
import os, sys, glob, json, shutil, hashlib

# RESOLVE THE MOUNT BY CONTENT, NOT BY NAME.
#
# An earlier draft globbed for a SLUG ("lens-paper-dann-r5") and fell back to the first
# entry under /kaggle/input. Both were wrong. The dataset that actually carries this
# bundle is named `lens-paper-coral-r4` -- a HISTORICAL name, created for round 4 and
# re-uploaded ever since -- and Kaggle mounts it several directories deep, at
# /kaggle/input/datasets/<owner>/<slug>/. Round 5's arm 1 ran from exactly that path
# (see init_head_from in runs/dann_r5_arm1_control/run_config.json). The old fallback
# then silently selected /kaggle/input/datasets, a PARENT directory, and the cell died
# on a missing MANIFEST.json a few lines later.
#
# So: find every MANIFEST.json under /kaggle/input and pick the one whose manifest says
# it is the dann_r5 bundle. The bundle identifies itself; a directory name is not
# evidence of anything.
found = []
for _d in range(1, 6):
    for _mp in glob.glob("/kaggle/input/" + "*/" * _d + "MANIFEST.json"):
        try:
            found.append((os.path.dirname(_mp), json.load(open(_mp))))
        except Exception as e:
            print(f"  (unreadable manifest at {_mp}: {e})")

print(f"MANIFEST.json files under /kaggle/input: {len(found)}")
for _root, _m in found:
    print(f"    {_root}")
    print(f"        bundle={_m.get('bundle')!r} v{_m.get('version')} "
          f"built {_m.get('built_utc')}")

cands = [(r, m) for r, m in found if m.get("bundle") == "dann_r5"]
assert cands, ("no dann_r5 bundle found under /kaggle/input.\n"
               "  Add Data -> the dataset holding the bundle. It is currently named\n"
               "  'lens-paper-coral-r4' -- a historical name from round 4; the bundle\n"
               "  inside it is dann_r5. If it IS attached, the upload did not include\n"
               "  MANIFEST.json. Run Cell 2a.")
cands.sort(key=lambda rm: int(rm[1].get("version") or 0))
INPUT, man = cands[-1]
if len(cands) > 1:
    print(f"  {len(cands)} dann_r5 bundles mounted; taking the highest version")
print("\nmount:", INPUT)

ver = os.path.join(INPUT, "VERSION.txt")
vtxt = open(ver).read().strip() if os.path.isfile(ver) else "*** MISSING ***"
print("VERSION:", vtxt)

print(f"bundle {man['bundle']} v{man['version']}  built {man['built_utc']}")

# WHAT ROUND 6 ACTUALLY NEEDS is these three files -- not a version LABEL. An earlier
# draft of this cell asserted `"v14" in VERSION.txt`, which tests a proxy: it fails on a
# perfectly good mount after any later rebuild, and it would pass on a v14 that somehow
# lacked the files. Assert on the artifacts, and report the version as information.
R6 = ["runs/lambda0_r6_prereg.json", "runs/lambda0_r6_audit.json",
      "runs/lambda0_r6_step0_conditions.json"]
absent = [r for r in R6 if not os.path.isfile(os.path.join(INPUT, r))]
if absent:
    raise SystemExit(
        "STALE OR INCOMPLETE MOUNT -- these round-6 artifacts are not on the mount:\n"
        + "".join(f"    {r}\n" for r in absent)
        + f"  the mount says: {vtxt}\n"
        + f"  MANIFEST says : v{man['version']} built {man['built_utc']}\n"
        "  Your LAPTOP is almost certainly fine. A Kaggle notebook is pinned to the\n"
        "  dataset VERSION it was attached with, and uploading a new version does NOT\n"
        "  move an attached notebook onto it. Fix: Add Data panel -> remove the\n"
        "  dataset (currently named 'lens-paper-coral-r4' -- a historical name; the\n"
        "  bundle inside it is dann_r5) -> add it again, which attaches the newest\n"
        "  version. Run Cell 2a for the exact cause.")
print(f"  all {len(R6)} round-6 artifacts present on the mount")

WORK = "/kaggle/working"
os.makedirs(f"{WORK}/scripts", exist_ok=True)
for f in man["scripts"]:
    shutil.copy2(os.path.join(INPUT, "scripts", f), f"{WORK}/scripts/{f}")
print(f"  copied {len(man['scripts'])} scripts (data stays in /kaggle/input)")

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

for rel, expect in (("runs/dann_r5_prereg.json", None),
                    ("runs/lambda0_r6_prereg.json",
                     "c1694a479eafb040cf7ad108e554047a95811cde8ce646444a448e8b1d717a25")):
    pre = json.load(open(os.path.join(INPUT, rel)))
    got = pre.pop("sha256_of_body_without_this_field")
    again = hashlib.sha256(json.dumps(pre, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    assert got == again, f"{rel} DIGEST FAILED: {got} != {again}"
    if expect:
        assert got == expect, (f"{rel} is NOT the pre-registration round 6 declared.\n"
                               f"  expected {expect}\n  got      {got}")
    print(f"  {rel} self-certifies: {got[:32]}...")

sys.path.insert(0, f"{WORK}/scripts")
os.environ["INPUT_ROOT"] = INPUT
print("\nOK -- mount verified.")
```

---

## Cell 2a — DIAGNOSTIC, only if Cell 2 refused (~5 s)

Run this when Cell 2 says the mount is stale or incomplete. It prints what is actually
there and names the cause. **It asserts nothing**, so it always completes.

```python
import os, glob, json, hashlib

EXPECT_PREREG = "c1694a479eafb040cf7ad108e554047a95811cde8ce646444a448e8b1d717a25"

print("=" * 78)
print("MOUNT DIAGNOSTIC")
print("=" * 78)

# Find dataset ROOTS by looking for MANIFEST.json at ANY depth. Kaggle mounts this
# dataset at /kaggle/input/datasets/<owner>/<slug>/, not at /kaggle/input/<slug>/.
mounts = []
for _d in range(1, 6):
    mounts += [os.path.dirname(q)
               for q in glob.glob("/kaggle/input/" + "*/" * _d + "MANIFEST.json")]
mounts = sorted(set(mounts))
print(f"\ndirectories under /kaggle/input holding a MANIFEST.json ({len(mounts)}):")
for m in mounts:
    print(f"    {m}")
if not mounts:
    print("    NONE FOUND. Either nothing is attached, or what is attached has no")
    print("    MANIFEST.json. Everything under /kaggle/input, 3 levels deep:")
    for _d in range(1, 4):
        for q in sorted(glob.glob("/kaggle/input/" + "*/" * _d))[:40]:
            print(f"      {q}")

for INPUT in mounts:
    print(f"\n--- {INPUT} ---")
    v = os.path.join(INPUT, "VERSION.txt")
    print(f"  VERSION.txt : {open(v).read().strip()!r}" if os.path.isfile(v)
          else "  VERSION.txt : ABSENT")
    mp = os.path.join(INPUT, "MANIFEST.json")
    if os.path.isfile(mp):
        m = json.load(open(mp))
        print(f"  MANIFEST    : {m.get('bundle')} v{m.get('version')} "
              f"built {m.get('built_utc')}  ({m.get('n_scripts')} scripts)")
        listed = [k for k in m.get("checkpoints", {}) if "lambda0_r6" in k]
        print(f"  round-6 artifacts LISTED in manifest : {len(listed)}")
    else:
        print("  MANIFEST.json: ABSENT -- this is probably not the bundle")

    print("  round-6 artifacts ON DISK:")
    for r in ("runs/lambda0_r6_prereg.json", "runs/lambda0_r6_audit.json",
              "runs/lambda0_r6_step0_conditions.json"):
        p = os.path.join(INPUT, r)
        print(f"    {'present' if os.path.isfile(p) else 'ABSENT ':<8} {r}")

    pre = os.path.join(INPUT, "runs/lambda0_r6_prereg.json")
    if os.path.isfile(pre):
        b = json.load(open(pre))
        got = b.pop("sha256_of_body_without_this_field")
        calc = hashlib.sha256(json.dumps(b, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
        print(f"  prereg sha256 : {got[:32]}...")
        print(f"    self-certifies      : {got == calc}")
        print(f"    matches round 6's   : {got == EXPECT_PREREG}")

    for r6 in ("audit_lambda0_r6.py", "lambda0_r6_step0_conditions.py",
               "write_lambda0_r6_prereg.py"):
        p = os.path.join(INPUT, "scripts", r6)
        print(f"    {'present' if os.path.isfile(p) else 'ABSENT ':<8} scripts/{r6}")

print("\n" + "=" * 78)
print("HOW TO READ THIS")
print("=" * 78)
print("""
  VERSION.txt says an OLDER version, and no round-6 artifacts
      -> The notebook is PINNED to an older dataset version. This is the usual
         cause and it is NOT a problem with your upload. Kaggle attaches a
         notebook to a SPECIFIC dataset version; pushing a new version does not
         move an already-attached notebook onto it.
         FIX: Add Data panel -> remove the dataset (named 'lens-paper-coral-r4')
              -> add it again. (Or use its version selector and pick the newest.)

  VERSION.txt says v14 but files are ABSENT
      -> A PARTIAL upload. Kaggle's "New Version" REPLACES the dataset contents
         with exactly what you upload; anything not included is gone from that
         version. FIX: re-upload the WHOLE kaggle/dann_r5/ folder.

  VERSION.txt ABSENT / MANIFEST.json ABSENT at the path a cell chose
      -> The cell picked the WRONG DIRECTORY; the dataset is probably fine.
         Kaggle mounts this one at /kaggle/input/datasets/<owner>/<slug>/,
         several levels deep, and it is named 'lens-paper-coral-r4' -- a
         historical name from round 4; the bundle inside it is dann_r5. Cell 2
         now resolves the mount by FINDING MANIFEST.json and reading
         bundle=='dann_r5', so neither a name nor a depth can mislead it. If the
         list above is EMPTY, the upload genuinely did not include MANIFEST.json.

  Everything present, prereg matches
      -> The mount is fine. Re-run Cell 2; it will pass.
""")
```

---

## Cell 3 — set the fold, then preflight (~3–5 min)

**The only line you edit is `FOLD`.** Builds the whole arm and stops before training —
§7.7's "a second rung validates at minute 2" discipline. A bad bundle must cost two
minutes, not two and a half hours.

```python
FOLD = 1          # <<<<<< EDIT THIS. Session A: 1 then 2. Session B: 3 then 4.
```

```python
import os, json, subprocess, sys
INPUT = os.environ["INPUT_ROOT"]

assert FOLD in (1, 2, 3, 4), "fold 0 is DEVELOPMENT and was run in round 5; it enters no mean"
OUT = f"/kaggle/working/runs/lambda0_r6_fold{FOLD}"
REF = f"/kaggle/working/lambda0_r6_ref_fold{FOLD}.json"

r = subprocess.run([sys.executable, "-u", "scripts/run_dann_r5.py",
                    "--arm", "dann_nb3_lambda0", "--fold", str(FOLD),
                    "--max-steps", "4000", "--morph-every", "25",
                    "--input-root", INPUT, "--out-dir", OUT,
                    "--emit-config", REF],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-6000:])
if r.returncode != 0:
    print(r.stderr[-4000:]); raise SystemExit("PREFLIGHT FAILED -- do not start the run")

cfg = json.load(open(REF))
pre = json.load(open(os.path.join(INPUT, "runs/lambda0_r6_prereg.json")))

print("\n" + "=" * 78)
print(f"WHAT FOLD {FOLD} WILL DO")
print("=" * 78)
for k in ("arm", "fold", "max_steps", "abort_mode", "abort_mode_source",
          "dann_lambda_zero", "batch_size", "grad_checkpointing", "ckpt_every",
          "log_every", "n_test_positives"):
    print(f"  {k:<26}: {cfg.get(k)}")
print(f"  {'step-0 gate':<26}: TPR {cfg['expect_step0_tpr']:.6f} "
      f"(tol {cfg['step0_tpr_tolerance']:.6f})  AUROC {cfg['expect_step0_auroc']:.6f} "
      f"(tol {cfg['step0_auroc_tolerance']:.0e})")
print(f"  {'phase-1 head ||.||_2':<26}: {cfg['init_head_l2']}")

# --- the pre-registration's assertions, enforced rather than trusted ---
assert cfg["arm"] == "dann_nb3_lambda0"
assert cfg["dann_lambda_zero"] is True,  "lambda must be pinned to zero"
assert cfg["max_steps"] == 4000,         "budget is 4000 -- see the prereg"
assert cfg["ckpt_every"] == 250 and cfg["log_every"] == 125
assert cfg["abort_mode"] == "log_only",  "conditions are ADVISORY on every fold"
assert cfg["step0_gate_skippable"] is False, "the gate is non-skippable"
g = pre["step0_gate"]["expected"][str(FOLD)]
assert abs(cfg["expect_step0_tpr"] - g["tpr"]) < 1e-12, "gate TPR != pre-registered"
assert abs(cfg["expect_step0_auroc"] - g["auroc"]) < 1e-12, "gate AUROC != pre-registered"
print("\n  every pre-registered setting matches the emitted config")

print(f"\n  on_declared_steps = {cfg['on_declared_steps']}  <-- EXPECTED to be False.")
print("    ARMS['dann_nb3_lambda0']['steps'] is round 5's DEVELOPMENT maximum of 8000.")
print("    4000 is ROUND 6's confirmatory budget and the prereg is its authority.")
print("    run_dann_r5.py is deliberately NOT edited: editing it would break the")
print("    byte-match against DANN's confirmatory folds, which is the match that matters.")

print("\n  ADVISORY conditions expected to fire (properties of the thresholds, not of")
print("  this run -- see the prereg's STEP_0_EVALUATION block):")
for k in pre["conditions"]["STEP_0_EVALUATION"][
        "THREE_CONDITIONS_FIRE_IMMEDIATELY_ON_A_HEALTHY_CONTROL"]["which"]:
    print(f"    - {k}")
print("  none of them can stop anything: abort_mode is log_only.")
print(f"\nPREFLIGHT OK for fold {FOLD} -- safe to start Cell 4.")
```

---

## Cell 4 — the training run (~150–170 min per fold)

Unbuffered and tee'd to a log inside `/kaggle/working`, so a browser disconnect loses
nothing.

> **Check the clock.** 4000 steps at n_blocks=3 measured **~153 min** on a T4 (§7.18's
> cost estimate; arm 1's 8000 steps took 267.8 min). Two folds is ~5.1 h against a 12 h
> cap. If you are already several hours into this session, start a fresh one.

```python
import subprocess, sys, os, time, threading
INPUT = os.environ["INPUT_ROOT"]
OUT = f"/kaggle/working/runs/lambda0_r6_fold{FOLD}"
LOG = f"/kaggle/working/lambda0_r6_fold{FOLD}.log"

print(f"fold               : {FOLD}")
print(f"expected wall time : ~153 min on a T4 (4000 steps, n_blocks=3)")
print(f"log file           : {LOG}   (survives a browser disconnect)")
print(f"output dir         : {OUT}\n")

cmd = [sys.executable, "-u", "scripts/run_dann_r5.py",
       "--arm", "dann_nb3_lambda0", "--fold", str(FOLD),
       "--max-steps", "4000", "--morph-every", "25",
       "--input-root", INPUT, "--out-dir", OUT]

# THE CHILD WRITES STRAIGHT TO THE LOG FILE. Do not put a PIPE between them.
#
# An earlier version ran the child with stdout=PIPE and had this cell read every line
# and print it. That couples training to the browser: when the notebook frontend stops
# accepting output, the parent blocks in print(), stops draining the pipe, the 64 KB
# OS pipe buffer fills, and THE CHILD BLOCKS ON WRITE. Training stops. It looks exactly
# like a hang with no error, and the heartbeat stops too, because the heartbeat thread
# lives in this cell and its print() blocks on the same thing. That is what happened on
# fold 2: 90 minutes of silence after a checkpoint that had completed normally.
#
# With stdout redirected to a file the child never touches the notebook's output stream,
# so a disconnected browser cannot slow it down or stop it. This cell only TAILS the
# file. If this cell blocks, dies, or is interrupted, the run keeps going.
t0 = time.time()
logf = open(LOG, "w", buffering=1)
proc = subprocess.Popen(cmd, cwd="/kaggle/working", stdout=logf,
                        stderr=subprocess.STDOUT, text=True)
print(f"child pid {proc.pid}, writing to {LOG}")
print("tailing every 60 s; the run does NOT depend on this cell staying alive.\n")

shown = 0
try:
    while proc.poll() is None:
        time.sleep(60)
        try:
            with open(LOG, "r", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for line in lines[shown:]:
            print(line, end="", flush=True)
        shown = len(lines)
        if not lines[shown - 1:]:
            print(f"    ... {(time.time() - t0) / 60:.0f} min elapsed, "
                  f"{len(lines)} log lines", flush=True)
    rc = proc.wait()
except KeyboardInterrupt:
    # Interrupting this cell must not orphan the child -- two runs writing the same
    # out-dir would corrupt both.
    print("\ninterrupted -- terminating the child so it cannot orphan")
    proc.terminate()
    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
finally:
    logf.close()

with open(LOG, "r", errors="replace") as fh:
    tail = fh.readlines()
for line in tail[shown:]:
    print(line, end="")

print(f"\nexit code {rc} after {(time.time()-t0)/60:.1f} min")
if rc != 0:
    print("RUN FAILED -- read the last lines above, then see the failure notes at the "
          "bottom of this document.")
```

*If the session dies part-way through, go to Cell 5 — no other action.*

---

## Cell 5 — resume (safe to run whether or not a resume is needed)

Same command. The runner **detects** an existing run and resumes from `resume/latest.pt`;
with nothing to resume it starts clean and says so.

Resume is exact where it matters. On this arm it is *more* exact than on arm 2: λ is pinned
to 0.0 and is a pure function of the step, so there is no schedule position to restore at
all. AdamW's moments, the GradScaler, the RNG and D's weights are all in the checkpoint —
a checkpoint missing D is **refused** rather than silently restarting it from a fresh init.
The one honest gap: augmentation offsets drawn after a resume are not guaranteed to match
an uninterrupted run's (a ±3px translate). Every resume is recorded in `run.json`, so a
resumed curve is never mistaken for an unresumed one.

```python
import subprocess, sys, os, time, threading
INPUT = os.environ["INPUT_ROOT"]
OUT = f"/kaggle/working/runs/lambda0_r6_fold{FOLD}"
LOG = f"/kaggle/working/lambda0_r6_fold{FOLD}_resume.log"

rp = f"{OUT}/resume/latest.pt"
if os.path.isfile(rp):
    import torch
    sch = (torch.load(rp, map_location="cpu", weights_only=False).get("schedule") or {})
    print(f"RESUMING fold {FOLD} from step {sch.get('global_step')} "
          f"(saved {sch.get('saved_utc')})")
    print(f"  discriminator state present: {bool(sch.get('discriminator_state'))}")
else:
    print(f"No resume checkpoint for fold {FOLD} -- this will start a FRESH run.")

t0 = time.time()

# Same file-not-pipe construction as Cell 4 -- see the note there.
logf = open(LOG, "w", buffering=1)
proc = subprocess.Popen([sys.executable, "-u", "scripts/run_dann_r5.py",
                         "--arm", "dann_nb3_lambda0", "--fold", str(FOLD),
                         "--max-steps", "4000", "--morph-every", "25",
                         "--input-root", INPUT, "--out-dir", OUT],
                        cwd="/kaggle/working", stdout=logf,
                        stderr=subprocess.STDOUT, text=True)
print(f"child pid {proc.pid}, writing to {LOG}\n")
shown = 0
try:
    while proc.poll() is None:
        time.sleep(60)
        with open(LOG, "r", errors="replace") as fh:
            lines = fh.readlines()
        for line in lines[shown:]:
            print(line, end="", flush=True)
        shown = len(lines)
    rc = proc.wait()
except KeyboardInterrupt:
    print("\ninterrupted -- terminating the child")
    proc.terminate()
    rc = proc.wait(timeout=60)
finally:
    logf.close()
with open(LOG, "r", errors="replace") as fh:
    for line in fh.readlines()[shown:]:
        print(line, end="")
print(f"\nexit code {rc} after {(time.time()-t0)/60:.1f} min")
```

---

## Cell 6 — integrity, and THE LOGITS CHECK (~10 s)

**Run this before the session ends, every fold.** Round 5's per-checkpoint logits never
came home from arm 1 and the curves had to be rebuilt by a separate re-score (§7.18). This
cell is the check that prevents a repeat: it asserts the per-checkpoint logits exist and
that there are **16** of them.

```python
import json, os, glob, csv
OUT = f"/kaggle/working/runs/lambda0_r6_fold{FOLD}"

rec = json.load(open(f"{OUT}/run_config.json"))
print("=" * 78)
print(f"RUN INTEGRITY -- fold {FOLD}")
print("=" * 78)
print(f"  steps completed   : {rec.get('steps_completed')} / {rec.get('max_steps')}")
print(f"  checkpoints       : {rec.get('n_checkpoints_saved')} of "
      f"{rec.get('n_checkpoints_expected')}   (must be 16 of 16)")
print(f"  aborted           : {rec.get('aborted')}   reason {rec.get('abort_reason')}")
wm = rec.get("wall_minutes")
print(f"  wall minutes      : {wm:.1f}" if wm else "  wall minutes      : n/a")
print(f"  fp16 steps skipped: {rec.get('fp16_skipped_cumulative')}")
print(f"  resumes           : {len(rec.get('resume_events') or [])}")
print(f"  GPU               : {rec.get('gpu_name')}")
gate = rec.get("step0_gate") or {}
print(f"  step-0 gate       : {gate.get('verdict')}   "
      f"TPR diff {gate.get('tpr_abs_diff')}   AUROC diff {gate.get('auroc_abs_diff')}")
print("    ^ PASS/FAIL is at the CALIBRATED one-positive tolerance (1/98, 1/97 on fold 4),")
print("      NOT at exactly 0.0 -- 7.7 records a correct nb3 run failing an exact gate by")
print("      one positive on live-encode drift. A diff of 0.0 is expected but not required.")

# ---- THE LOGITS CHECK. This is the cell's reason for existing. ----
ldir = f"{OUT}/logits/dann_nb3_lambda0_fold{FOLD}"
steps = sorted(int(os.path.basename(p)[5:-4]) for p in glob.glob(f"{ldir}/step_*.npz"))
print(f"\n  per-checkpoint logits: {len(steps)} files in {ldir}")
print(f"    steps: {steps}")
expect = list(range(250, 4001, 250))
missing = [s for s in expect if s not in steps]
assert os.path.isfile(f"{OUT}/logits/ids.npz"), "logits/ids.npz MISSING"
assert not missing, (f"MISSING PER-CHECKPOINT LOGITS for steps {missing}. "
                     f"This is round 5's collection defect. DO NOT end the session -- "
                     f"the curves cannot be rebuilt without these.")
print("    all 16 present, including the 1500 and 2000 sensitivity readouts")

rows = list(csv.DictReader(open(f"{OUT}/steps.csv", newline="")))
def col(c):
    return [float(x[c]) for x in rows if x.get(c) not in (None, "", "nan")]
print("\n  monitors (start / end / min / max):")
for c in ("morph_head_auroc_pool", "morph_proxy_auroc",
          "src_participation_ratio", "tgt_participation_ratio",
          "d_pred_source_frac", "src_perdim_std", "tgt_perdim_std"):
    v = col(c)
    if v:
        print(f"    {c:<26}: {v[0]:.6f}  {v[-1]:.6f}  {min(v):.6f}  {max(v):.6f}")

fired = {c: any(float(r[c]) > 0 for r in rows)
         for c in rows[0] if c.endswith("_fired")}
print(f"\n  advisory conditions fired: {[k for k, v in fired.items() if v]}")
print("    3b on the head column and condition 4 on both domains are EXPECTED "
      "(prereg STEP_0_EVALUATION).")
print("    abort3b on the PROBE column, or conditions 1/2, firing would be NEW -- report it.")
print(f"\nSEND THIS BLOCK BACK for fold {FOLD}.")
```

---

## Cell 7 — package for download (~5 min per fold)

Run once per fold, after Cell 6 passes. Refuses to package an incomplete run.

```python
import subprocess, sys, os
OUT = f"/kaggle/working/runs/lambda0_r6_fold{FOLD}"
DST = f"/kaggle/working/lambda0_r6_fold{FOLD}_out"

r = subprocess.run([sys.executable, "-u", "scripts/build_dann_r5_bundle.py",
                    "--pack-results", OUT, "--results-out", DST],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-6000:])
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit("PACKAGING REFUSED -- see the message above")

tot = 0
for f in sorted(os.listdir(DST)):
    p = os.path.join(DST, f); sz = os.path.getsize(p); tot += sz
    print(f"  {f:<48} {sz/1e6:8.1f} MB")
print(f"\n  {DST}   TOTAL {tot/1e9:.2f} GB")
print("  Download every part from the Output panel. Each part is self-contained.")
```

**Then go back to Cell 3, change `FOLD`, and repeat Cells 3 → 7 for the second fold of
this session.**

---

## Cell 7a — direct download links (~10 s)

Use this when the Output panel will not download. It builds **clickable links served by
the notebook itself**, which bypasses the Output panel entirely.

**Read this before downloading a gigabyte you do not need.** Per fold, Cell 7 writes:

| file | size | holds |
|---|---:|---|
| `screen_core.tar.gz` | **~193 MB** | the analysis files, **plus `resume/latest.pt` at 190 MB** |
| `ckpt_..._partNNofNN.tar` ×3 | **~1060 MB** (16 checkpoints, 66.2 MB each) | model weights |

> **Correction — an earlier version of this section said the core archive was ~2.7 MB.
> It is not.** `package_screen_results.build_core_archive` excludes only the
> `checkpoints/` directory, so **`resume/latest.pt` rides along inside the core**,
> measured at **190 MB** on round 5's arm 1. The small figure came from a
> `core_fold1.tar.gz` that had been trimmed by hand afterwards, and is not what Cell 7
> produces.

The files the analysis genuinely needs total **~2.5 MB**, but no archive Cell 7 writes
contains exactly them — so **run Cell 7c below** and download that instead. Round 5 needed
the checkpoint tars *because its logits never came home*; Cell 6 refuses to let that
happen this round, so the weights are optional, and `resume/latest.pt` matters only for
resuming an interrupted run.

```python
import os, glob, hashlib
from IPython.display import display, FileLink, HTML

WORK = "/kaggle/working"
TARGET = ""          # "" = auto-detect this session's packed output dirs; or a dir/file
DATA_URI_MAX_MB = 20 # files under this also get a self-contained button (see note below)

def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} TB"

def sha16(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:16]

roots = ([TARGET] if TARGET else
         sorted(glob.glob(f"{WORK}/lambda0_r6_fold*_out")) or
         sorted(d for d in glob.glob(f"{WORK}/*_out") if os.path.isdir(d)))
if not roots:
    print(f"nothing found. Contents of {WORK}:")
    for q in sorted(glob.glob(f"{WORK}/*")):
        print("   ", q)
    raise SystemExit("run Cell 7 first, or set TARGET to a path")

files = []
for r in roots:
    if os.path.isfile(r):
        files.append(r)
    else:
        for d, _, fs in os.walk(r):
            files += [os.path.join(d, f) for f in sorted(fs)]

# Smallest first, so the cheap files are the ones you click. NOTE: "core" in a name does
# NOT mean small -- screen_core.tar.gz carries resume/latest.pt and runs ~193 MB. Classify
# by SIZE, which is the thing that actually costs bandwidth.
SMALL_MB = 50
def is_core(p):
    return os.path.getsize(p) <= SMALL_MB * 1024 * 1024
files.sort(key=lambda p: (not is_core(p), os.path.getsize(p)))

small_bytes = sum(os.path.getsize(f) for f in files if is_core(f))
big_bytes = sum(os.path.getsize(f) for f in files if not is_core(f))
print(f"{len(files)} files in {len(roots)} dir(s)")
print(f"  small (<={SMALL_MB} MB each) : {human(small_bytes)}")
print(f"  large                      : {human(big_bytes)}")
print(f"  TOTAL                      : {human(small_bytes + big_bytes)}")
print("  NB: screen_core.tar.gz is ~193 MB -- it carries resume/latest.pt.")
print("      For the ~2.5 MB the analysis needs, run Cell 7c.\n")

for f in files:
    rel = os.path.relpath(f, WORK)
    if rel.startswith(".."):
        print(f"  SKIP (outside {WORK}, cannot be served): {f}")
        continue
    sz = os.path.getsize(f)
    tag = "small" if is_core(f) else "LARGE"
    print(f"{tag} {human(sz):>10}  sha256 {sha16(f)}  {rel}")
    display(FileLink(rel))

    # A second, independent route for small files: embed the bytes in the page as a
    # data: URI. It does not touch the notebook file server at all, so it still works
    # when FileLink does not. Only for small files -- base64 inflates by 4/3 and a big
    # one will hang the browser tab.
    if sz <= DATA_URI_MAX_MB * 1024 * 1024:
        import base64
        b64 = base64.b64encode(open(f, "rb").read()).decode()
        display(HTML(
            f'<a download="{os.path.basename(f)}" '
            f'href="data:application/octet-stream;base64,{b64}" '
            f'style="display:inline-block;margin:2px 0 10px 0;padding:6px 12px;'
            f'background:#20beff;color:#fff;border-radius:4px;'
            f'text-decoration:none;font-family:sans-serif">'
            f'&#11015; {os.path.basename(f)} ({human(sz)}) — self-contained</a>'))
    print()

print("If a link 404s or stalls: File -> Save Version -> Save & Run All (Commit), then")
print("download from that version's Output tab. A committed version serves files more")
print("reliably than a live session, and it survives the session ending.")
```

**If the ~1 GB checkpoint parts still will not come down** and you decide you want them:
each `ckpt_..._partNNofNN.tar` is **self-contained** — its MANIFEST records the part's own
sha256 and the exact checkpoint steps inside it — so fetch them one at a time and re-try
individual failures without restarting the others. The `sha256` printed above is the first
16 hex characters of each part's digest; check it against `MANIFEST.json` after
downloading. **Do not delete a part that fails — re-fetch it.**

---

## If the download is SLOW — use the Kaggle API

Kaggle's in-browser download is throttled and not resumable. **The CLI is neither**, and
it is already installed and authenticated on this laptop (Kaggle API 1.7.4.5,
`~/.kaggle/kaggle.json`, user `navneet78`).

**First commit the notebook** — `File → Save Version → Save & Run All (Commit)`. The API
serves a *committed version's* output, not a live session's. Then, from the laptop:

```bash
kaggle kernels list --mine
```

Find the notebook's slug, then pull its output:

```bash
kaggle kernels output navneet78/<notebook-slug> -p runs/lambda0_r6_incoming
```

Resumable, scriptable, and typically several times faster than the browser. Verify each
part afterwards against the `sha256` values in the downloaded `MANIFEST.json`; Cell 7a
prints the first 16 hex characters of the same digests.

---

## Cell 7c — the ANALYSIS SET only, ~2.5 MB per fold (run this one)

Everything this round's declared analysis reads, and nothing else: no model weights, no
resume checkpoint. Run it after Cell 6 passes, then fetch it with Cell 7a's link — a few
MB downloads in seconds even on a throttled connection.

It is deliberately a *separate, additional* archive rather than a change to Cell 7. Cell
7's coverage check — every checkpoint the run named must appear in some tar — exists
because round 5 packaged cleanly, verified every checksum, and still arrived with all 32
logit files missing (§7.18). Trimming what that packer emits, to save bandwidth, would
weaken the check that catches exactly that failure.

```python
import os, tarfile, hashlib, glob

OUT = f"/kaggle/working/runs/lambda0_r6_fold{FOLD}"
DST = f"/kaggle/working/analysis_fold{FOLD}.tar.gz"

want = []
for f in ("run_config.json", "run.json", "steps.csv", "trace.csv", "checkpoints.csv"):
    q = os.path.join(OUT, f)
    if os.path.isfile(q):
        want.append(q)
    else:
        print(f"  WARNING: {f} not found")
want += sorted(glob.glob(os.path.join(OUT, "logits", "**", "*.npz"), recursive=True))

# The same expectation Cell 6 enforces, re-checked here so a thin archive cannot be built
# from an incomplete run and then mistaken later for a complete one.
steps = sorted(int(os.path.basename(q)[5:-4])
               for q in want if os.path.basename(q).startswith("step_"))
missing = [x for x in range(250, 4001, 250) if x not in steps]
assert not missing, f"per-checkpoint logits missing for steps {missing} -- see Cell 6"

with tarfile.open(DST, "w:gz") as tf:
    for q in want:
        arc = f"lambda0_r6_fold{FOLD}/" + os.path.relpath(q, OUT).replace("\\", "/")
        tf.add(q, arcname=arc)

h = hashlib.sha256()
with open(DST, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 20), b""):
        h.update(b)

print(f"{DST}")
print(f"  {os.path.getsize(DST)/1e6:.2f} MB   {len(want)} files   sha256 {h.hexdigest()}")
print(f"  logits: {len(steps)} checkpoints, {steps[0]}..{steps[-1]}")
print("\nThis is the whole analysis input for this fold: TPR@1e-3 at 4000 and at the")
print("1500/2000 readouts, all four paired contrasts, and the checkpoint-to-checkpoint sd.")
```

---

## Cell 7b — OPTIONAL: only the read-off checkpoints (~199 MB instead of 1060 MB)

Only if you want model *weights* and the full 1060 MB is too slow. It tars the **three
steps the pre-registration reads** — the declared budget 4000 and the sensitivity
readouts 1500 and 2000 — and leaves the other 13 behind.

```python
import os, tarfile, hashlib

OUT = f"/kaggle/working/runs/lambda0_r6_fold{FOLD}"
KEEP = [1500, 2000, 4000]
DST = f"/kaggle/working/lambda0_r6_fold{FOLD}_readoff_ckpts.tar"

ck = os.path.join(OUT, "checkpoints")
have = sorted(int(f[5:-3]) for f in os.listdir(ck) if f.startswith("step_"))
missing = [x for x in KEEP if x not in have]
assert not missing, f"checkpoints {missing} not on disk; have {have}"

with tarfile.open(DST, "w") as tf:                 # uncompressed: float weights
    for x in KEEP:
        src = os.path.join(ck, f"step_{x}.pt")
        tf.add(src, arcname=f"lambda0_r6_fold{FOLD}/checkpoints/step_{x}.pt")
        print(f"  + step_{x}.pt  {os.path.getsize(src)/1e6:.1f} MB")

h = hashlib.sha256()
with open(DST, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 20), b""):
        h.update(b)
print(f"\n{DST}")
print(f"  {os.path.getsize(DST)/1e6:.1f} MB   sha256 {h.hexdigest()}")
print(f"  contains steps {KEEP} of the {len(have)} on disk")
print("\nPARTIAL BY DESIGN -- record alongside it that 13 of 16 checkpoints were not")
print("retained, so nobody later reads the gap as a collection defect.")
```

**Write down that this archive is partial.** Round 5's collection defect was a
missing-file problem that looked like a complete run; an intentionally partial archive
that is not labelled as one is the same trap with a different cause.

---

## If something goes wrong

| what you see | what to do |
|---|---|
| **Cell 2: `FileNotFoundError` on `MANIFEST.json`, or `VERSION: *** MISSING ***`** | The mount resolved to the wrong directory. Fixed: Cell 2 now finds `MANIFEST.json` at any depth and selects the bundle that identifies itself as `dann_r5`, so neither the dataset's historical name (`lens-paper-coral-r4`) nor Kaggle's `/kaggle/input/datasets/<owner>/<slug>/` layout can mislead it. If it still finds nothing, run **Cell 2a** — it lists every directory under `/kaggle/input` holding a manifest. |
| **Cell 2: mount stale or incomplete** | Run **Cell 2a**, which names the cause. Almost always the notebook is pinned to an older dataset version — Kaggle does not move an attached notebook onto a new version. Fix: Add Data panel → remove the dataset (`lens-paper-coral-r4`) → add it again. If `VERSION.txt` already says v14 but the artifacts are absent, it was a partial upload: Kaggle's "New Version" **replaces** the dataset contents, so re-upload the whole `kaggle/dann_r5/` folder. |
| **Cell 3: step-0 gate mismatch** | The gate is the only artifact proving fold identity — the phase-1 head is fold-independent, so a mis-pointed fold trains happily and produces a plausible wrong number. Send the printed TPR/AUROC and **stop**. Fix by re-scoring the fold table on the venue, **never** by widening the gate. |
| **Download is extremely slow, or will not start** | Kaggle's browser download is throttled and not resumable. Commit the notebook (`Save Version → Save & Run All`), then pull it with the CLI, which is already installed and authenticated here: `kaggle kernels output navneet78/<slug> -p runs/lambda0_r6_incoming`. Then run **Cell 7c**, which packs the **~2.5 MB** the analysis actually needs. Do NOT assume `screen_core.tar.gz` is small — it is ~193 MB, because it carries `resume/latest.pt`. **Cell 7a** serves files from the notebook itself if you want a browser route; **Cell 7b** packs only the three read-off checkpoints (~199 MB). |
| **Cell 6: missing per-checkpoint logits** | **Do not end the session.** This is round 5's collection defect. The curves cannot be rebuilt without them. Report before shutting down. |
| **Output stops for many minutes and the run is overdue** | The pace is not the problem — check it first from the log: windows without a checkpoint should sit near **1.68 s/step** and windows containing one near **2.69 s/step**, projecting ~146 min. If the numbers were normal and then output simply stopped, **the notebook's output stream is the suspect, not training.** The heartbeat lives in the cell, so if IT has also stopped, the cell is blocked, not the trainer. **1.** Hard-reload the browser (Ctrl+Shift+R) — Kaggle often reattaches and shows the run further along or finished. **2.** Check the session's GPU/CPU meters in the right sidebar; ~0% on both means genuinely stalled. **3.** If stalled: interrupt the cell and run **Cell 5**. `save_resume` completes *before* the `[ckpt N]` line prints, so a visible `[ckpt N]` proves `resume/latest.pt` is on disk at step N — you lose at most the steps since, ≤ 250 (~7 min). Cell 4 is now written so this cannot recur: the child writes to the log file, not through a pipe this cell has to drain. |
| **Session dies mid-run** | **Cell 5. Nothing else.** Do not delete the output directory — the resume needs it. |
| **CUDA out of memory** | Gradient checkpointing is mandatory at n_blocks=3 and is already on. **Do not reduce the batch — 256 is pinned by §7.4 and changing it invalidates the whole cross-method comparison.** Report it. |
| **Not a T4** | Stop. DANN's folds 1–4 ran on T4 and §7.7 records that results are not portable across architectures at this precision. A GPU difference inside the DANN − λ=0 contrast destroys the one clean comparison. |
| **Wall time far from ~153 min** | Much faster usually means fewer steps ran than you think — Cell 6 prints the step count, check it against 4000. Much slower usually means the GPU is not a T4 or the data is being read from the wrong place. |
| **`on_declared_steps` is False** | **Expected.** See Cell 3's explanation — it means "not on round 5's development maximum of 8000", which is correct. |
| **Advisory conditions fired** | Expected for `3b` on the head column and `4` on both domains. New firings — `3b` on the probe column, or conditions 1 or 2 — are worth reporting. Nothing can abort: `abort_mode` is `log_only`. |

---

## What happens after all four folds are home

1. Per-fold TPR@FPR=1e-3 (interpolated, `drop_intermediate=False`, **always a rate**) at
   4000, and again at the pre-declared readouts 1500 and 2000.
2. The four paired contrasts, on folds 1–4, with a **paired t-interval on 4 differences
   (df = 3)** — not a bootstrap, because at n = 4 a bootstrap draws from 4⁴ = 256 distinct
   resamples and its tails are a lattice artifact.
3. **The disclosure that travels in the same sentence as the numbers**, not in a footnote:
   this arm initialises from Baseline A's head, as DANN × nb3 does, so **only the DANN row
   is a matched contrast**. CORAL × nb3 initialises from round-1 CORAL heads and
   FreeMatch × nb3 from its own — for those two this is a legitimate *method-neutral*
   baseline but **not** a matched control. A second asymmetry the audit found: DANN's
   confirmatory folds ran `abort_mode='armed'` while this arm runs `log_only`; DANN's folds
   did not abort, so no training step can have differed, but it is disclosed.
4. This arm's checkpoint-to-checkpoint sd on folds 1–4, added to §7.18's stability table
   alongside fold 0's sd/mean 0.276.
5. **The decision, already declared in the pre-registration so it cannot be chosen
   afterwards:** folds low (near 0.1531) → the DA term owns the 0.15 → 0.42 movement, and
   BNM is **GO**; folds near the DA arms (~0.40+) → the DA terms are decorative, the finding
   is about encoder capacity, BNM is **NO-GO and the bake-off stops** (that is a different
   paper, and still a publishable one); in between or ambiguous → **NO-GO by default**.
