# 1:10 diluted-target experiment on Kaggle — launch cells

**What this is.** DANN, BNM, CORAL and FreeMatch × nb3, folds 1–4, rerun on a target diluted to
1:10 and sampled uniformly (no use of the latent labels). FreeMatch's frozen stage is rerun first,
because its new heads initialise FreeMatch × nb3. CORAL's frozen stage already ran locally on the
CPU, and its heads ship in this bundle. Everything is fixed in advance by
`runs/dilute10_prereg_part1.json` (`bd537a02…`) and `runs/dilute10_prereg_part2.json` (`b8e09197…`),
both of which ship in the bundle and are verified before anything runs (PROJECTv7.md §7.36–§7.39).
From bundle v3, `runs/dilute10_prereg_part3.json` (`1b583ea2…`, §7.48) ships too and is verified by
Cell 2. It binds CORAL × nb3 and FreeMatch × nb3, which refuse to start without it; Cell 3 asserts
its digest for those two arms.

**One fold at a time, on one T4** (author's decision, 2026-10-14). There is no concurrency; every
run uses `CUDA_VISIBLE_DEVICES=0`. Each session stays under **9 h** of wall time, which leaves room
under Kaggle's 12 h cap.

> *Note for the author: check the Kaggle quota page before S1. It is not confirmed here whether a
> T4 ×2 session draws quota per session hour or per GPU hour. One fold at a time uses one of the two
> GPUs.*

**Nothing here needs a decision from you.** For each run you edit one line: `RUN`.

---

## Session plan

The times are the recorded per-fold wall times from part 1 (T4), plus about 20 min per fold for
preflight, the step-0 gate and packaging.

| session | runs, in order | est. wall | download afterwards |
|---|---|---:|---|
| **S1** | DANN fold 1, fold 2, fold 3 | 8.3 h | `out_dann_f{1,2,3}/` |
| **S2** | DANN fold 4, BNM fold 1, BNM fold 2 | 7.9 h | `out_dann_f4/`, `out_bnm_f{1,2}/` |
| **S3** | BNM fold 3, BNM fold 4, CORAL nb3 fold 1, CORAL nb3 fold 2 | 7.9 h | `out_bnm_f{3,4}/`, `out_coral_nb3_f{1,2}/` |
| **S4** | FreeMatch frozen fold 1, fold 2, fold 3 | 7.3 h | `out_freematch_frozen_f{1,2,3}/` — **includes the heads** |
| **S5** | FreeMatch frozen fold 4, CORAL nb3 fold 3, CORAL nb3 fold 4 | 5.6 h | `out_freematch_frozen_f4/`, `out_coral_nb3_f{3,4}/` |
| — | **break** | | The author downloads S4 and S5. A short ingestion brief verifies the four FreeMatch frozen heads and their `gate_values.json`, then builds bundle **v3** with them. |
| **S6** | FreeMatch nb3 fold 1, fold 2, fold 3 (bundle v3) | 7.3 h | `out_freematch_nb3_f{1,2,3}/` |
| **S7** | FreeMatch nb3 fold 4 (bundle v3) | 2.5 h | `out_freematch_nb3_f4/` |

Per-fold minutes:
- DANN 152.6 / 137.9 / 147.1 / 139.6
- BNM 139.4 / 135.1 / 122.3 / 131.4
- CORAL nb3 70.8 / 68.3 / 71.8 / 65.5
- FreeMatch frozen 124.2 / 128.2 / 123.7 / 138.3 (logged epoch time)
- FreeMatch nb3 122.2 / 131.3 / 127.2 / 129.1 (including the larger SAT-rebuild upper bound)

**Resume.**
- **DANN and CORAL nb3 resume** from `resume/latest.pt` (Cell 5), and the runner records each
  resume in `run.json`.
- **BNM, FreeMatch frozen and FreeMatch nb3 have no resume path** in their runners, and adding one
  would mean editing a frozen module. If a session dies mid-fold, that fold restarts from step 0.
  Part 1's stop condition 6 permits a restart **only for a logged infrastructure failure**, and
  Cell 5 requires that reason in writing.
- **A completed fold is never rerun.** Cell 3 refuses a run whose `dilute10_post_run_check.json`
  exists.

---

## Before you start

1. Upload `C:\Users\krris\ML\lens_paper\kaggle\dilute10\` as a **new** Kaggle dataset (for example
   `lens-paper-dilute10`). Upload the **whole folder**: Kaggle extracts `data.zip` itself, and the
   cells verify the extracted contents.
2. New notebook → **Accelerator: GPU T4 ×2** → **Add Data** → that dataset.
3. Run Cells 1–2 once per session. Then, for each run of the session, run Cells 3 → 4 → 6 → 7.

> **A notebook stays pinned to the dataset VERSION it was attached with.** For S6 and S7 (bundle
> v3), remove the dataset in the Add Data panel and add it again. Cell 2 prints the bundle version,
> and Cell 3 refuses FreeMatch nb3 runs before v3.

> **v2 = the provenance fix:** every run records part 2's sha256 and the bundle version and
> manifest, and packaging archives `stage.json`. DANN folds 1–4 ran on v1; every run from BNM on
> needs v2 or later (Cell 3 refuses a config without `part2_sha256`).

---

## Cell 1 — environment check (~10 s)

```python
import shutil, os, torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| available", torch.cuda.is_available())
assert torch.cuda.is_available(), "NO GPU. Settings -> Accelerator -> GPU T4 x2."
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory/2**30:.2f} GiB  sm_{p.major}{p.minor}")
assert "T4" in torch.cuda.get_device_name(0), (
    "NOT A T4. The original runs were on T4; PROJECT 7.7 records that results are not "
    "portable across GPU architectures at this precision. Switch to a T4.")
free = shutil.disk_usage("/kaggle/working").free / 2**30
print(f"  /kaggle/working free: {free:.1f} GiB")
assert free > 10, f"only {free:.1f} GiB free"
print("  CPU cores:", os.cpu_count())
```

---

## Cell 2 — mount by content, verify, stage (~2–4 min)

This cell finds the `MANIFEST.json` whose `bundle == "dilute10"` and **verifies every listed file by
sha256**. It checks the extracted data file by file, and `splits.json` by hash. It then stages
`scripts/`, `data/splits/` and `runs/` into `/kaggle/working`, links the image folders from the
mount, and re-verifies parts 1, 2 and the pool in place. **Any mismatch aborts.** The logic lives in
`scripts/stage_dilute10.py`, the same script the local self-test ran.

```python
import os, sys, glob, json, subprocess
WORK = "/kaggle/working"
cand = [p for d in range(1, 7) for p in glob.glob("/kaggle/input/" + "*/" * d + "MANIFEST.json")]
roots = []
for p in cand:
    try:
        if json.load(open(p)).get("bundle") == "dilute10":
            roots.append((json.load(open(p))["version"], os.path.dirname(p)))
    except Exception:
        pass
assert roots, "no dilute10 bundle under /kaggle/input -- Add Data -> the dilute10 dataset"
roots.sort()
BUNDLE = roots[-1][1]
os.makedirs(f"{WORK}/scripts", exist_ok=True)
import shutil; shutil.copy2(f"{BUNDLE}/scripts/stage_dilute10.py", f"{WORK}/scripts/stage_dilute10.py")
r = subprocess.run([sys.executable, "-u", "scripts/stage_dilute10.py",
                    "--input-root", "/kaggle/input", "--work", WORK],
                   cwd=WORK, capture_output=True, text=True)
print(r.stdout[-4000:])
if r.returncode != 0:
    print(r.stderr[-4000:]); raise SystemExit("STAGING FAILED -- nothing may run")
ST = json.load(open(f"{WORK}/stage.json"))
os.environ["INPUT_ROOT"] = "/kaggle/input"
print("bundle version", ST["version"], "| data", ST["data_dir"])
```

---

## Cell 3 — choose the run, then preflight (~3–6 min)

**The only line you edit is `RUN`.** The preflight builds the whole run and trains nothing. It
checks the config against parts 1 and 2 and refuses a fold that has already completed.

```python
RUN = ("dann", 1)     # <<<<<< EDIT: (arm, fold); arm in dann, bnm, coral_nb3, freematch_frozen, freematch_nb3
```

```python
import os, sys, json, subprocess
WORK, (ARM, FOLD) = "/kaggle/working", RUN
ST = json.load(open(f"{WORK}/stage.json"))
assert FOLD in (1, 2, 3, 4)
SCRIPT = {"dann": "run_dilute10_dann.py", "bnm": "run_dilute10_bnm.py",
          "coral_nb3": "run_dilute10_coral_nb3.py",
          "freematch_frozen": "run_dilute10_freematch_frozen.py",
          "freematch_nb3": "run_dilute10_freematch_nb3.py"}[ARM]
OUT = f"{WORK}/runs/dilute10/{ARM}/fold{FOLD}"
if os.path.isfile(f"{OUT}/dilute10_post_run_check.json"):
    raise SystemExit(f"{ARM} fold {FOLD} has COMPLETED -- part 1 stop condition 6: never rerun "
                     f"a completed fold")
if ARM == "freematch_nb3":
    assert int(ST["version"]) >= 3, "FreeMatch nb3 needs bundle v3 (the FreeMatch frozen heads)"
    assert os.path.isfile(f"{WORK}/runs/dilute10/freematch_frozen/fold{FOLD}/gate_values.json")
if ARM == "coral_nb3":
    assert os.path.isfile(f"{WORK}/runs/dilute10/coral_frozen/fold{FOLD}/gate_values.json")
BASE = [sys.executable, "-u", f"scripts/{SCRIPT}", "--fold", str(FOLD),
        "--target-pool", "dilute10", "--target-sampler", "uniform"]
if ARM in ("dann", "bnm", "coral_nb3"):
    BASE += ["--input-root", os.environ["INPUT_ROOT"]]
REF = f"{WORK}/preflight_{ARM}_f{FOLD}.json"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
r = subprocess.run(BASE + ["--emit-config", REF], cwd=WORK, env=env, capture_output=True, text=True)
print(r.stdout[-3000:])
if r.returncode != 0:
    print(r.stderr[-4000:]); raise SystemExit("PREFLIGHT FAILED -- do not start the run")
cfg = json.load(open(REF))
d = cfg["dilute10"]
assert d["part1_sha256"].startswith("bd537a02") and d["pool_sha256"].startswith("18c73d6c")
assert d["target_pool"] == "dilute10" and d["target_sampler"] == "uniform"
assert d["part2_sha256"].startswith("b8e09197"), "config lacks part 2's sha256 -- needs bundle v2+"
if ARM in ("coral_nb3", "freematch_nb3"):   # part 3 binds these two (PROJECTv7 7.48)
    assert (d.get("part3_sha256") or "").startswith("1b583ea2"), "config lacks part 3's sha256 -- needs bundle v3+"
print("bundle version", d["bundle_version"], "| manifest", (d["bundle_manifest_sha256"] or "")[:16])
print("realised training pool:", d["realised_training_pool"])
print("rebound names:"); [print("   ", x) for x in d["rebound_names"]]
print(f"\nPREFLIGHT OK -- {ARM} fold {FOLD}. Start Cell 4.")
```

---

## Cell 4 — the run (one fold, GPU 0; see the session table for the time)

The child writes **straight to a log file**, never to a PIPE (§7.21: a blocked pipe stalls
training). This cell only tails that file every 60 s with a heartbeat. Interrupting the cell
**terminates the child**, so it cannot orphan.

```python
import subprocess, sys, os, time
LOG = f"{WORK}/{ARM}_f{FOLD}.log"
print(f"{ARM} fold {FOLD} -> {OUT}\nlog {LOG}\n")
t0 = time.time()
logf = open(LOG, "w", buffering=1)
proc = subprocess.Popen(BASE, cwd=WORK, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
shown = 0
try:
    while proc.poll() is None:
        time.sleep(60)
        try:
            lines = open(LOG, errors="replace").readlines()
        except OSError:
            continue
        for line in lines[shown:]:
            print(line, end="", flush=True)
        if len(lines) == shown:
            print(f"    ... heartbeat {(time.time()-t0)/60:.0f} min, {len(lines)} log lines", flush=True)
        shown = len(lines)
    rc = proc.wait()
except KeyboardInterrupt:
    print("\ninterrupted -- terminating the child so it cannot orphan")
    proc.terminate()
    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill(); rc = proc.wait()
finally:
    logf.close()
for line in open(LOG, errors="replace").readlines()[shown:]:
    print(line, end="")
print(f"\nexit code {rc} after {(time.time()-t0)/60:.1f} min")
if rc != 0:
    print("RUN STOPPED. A '[dilute10 STOP: ...]' line names the part-2 stop condition. "
          "Send the last 60 log lines back; do not rerun on your own.")
```

---

## Cell 5 — resume after a dead session

```python
import os, json, time
RESUMABLE = ARM in ("dann", "coral_nb3")
rp = f"{OUT}/resume/latest.pt"
if RESUMABLE and os.path.isfile(rp):
    print(f"RESUMING {ARM} fold {FOLD} from {rp} -- the runner records it in run.json")
    BASE_RESUME = BASE + ["--force"]          # the wrapper refuses an existing dir without it
else:
    REASON = ""     # <<<<<< REQUIRED for a restart: the logged infrastructure failure
    assert REASON, (f"{ARM} cannot resume. A restart from step 0 is allowed ONLY for a logged "
                    f"infrastructure failure (part 1, stop condition 6). Write it in REASON.")
    os.rename(OUT, f"{OUT}_DEAD_{time.strftime('%Y%m%dT%H%M%S')}")
    with open(f"{WORK}/restarts.log", "a") as fh:
        fh.write(json.dumps({"arm": ARM, "fold": FOLD, "utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), "reason": REASON}) + "\n")
    BASE_RESUME = BASE
print("now re-run Cell 4 with BASE = BASE_RESUME")
BASE = BASE_RESUME
```

---

## Cell 6 — the run's own checks, and THE LOGITS CHECK (~10 s)

**Run this before the session ends, for every run.** The wrapper has already run part 2's
post-run checks (budget, logit coverage). This cell re-reads them and the lens-fraction summary.
It prints verdicts only, never metric values.

```python
import json, os, csv
chk = json.load(open(f"{OUT}/dilute10_post_run_check.json"))
print(json.dumps(chk, indent=1))
assert not chk["missing_logits"] and not chk["named_in_checkpoints_csv_but_missing"], "LOGITS MISSING"
lf = f"{OUT}/target_lens_fraction_summary.json"
if os.path.isfile(lf):
    s = json.load(open(lf))
    print(f"lens fraction: run-mean {s['mean']:.5f}, expected {s['expected']:.5f}, band {s['band']}, batches {s['n_batches']}")
for f in ("run.json", "run_config.json", "results.json"):
    p = f"{OUT}/{f}"
    if os.path.isfile(p):
        r = json.load(open(p))
        g = r.get("step0_gate") or {}
        print(f"{f}: steps {r.get('steps_completed', r.get('steps'))}, aborted {r.get('aborted')}, "
              f"gate verdict {g.get('verdict', g.get('passed'))}, GPU {r.get('gpu_name')}")
gv = f"{OUT}/gate_values.json"
if os.path.isfile(gv):
    print("gate_values.json written (frozen stage) -- values not printed; the analysis brief reads them")
print(f"\nSEND THIS BLOCK BACK for {ARM} fold {FOLD}.")
```

---

## Cell 7 — package for download (~5 min)

This cell refuses an incomplete run, then packs by glob into independent tars with per-part
sha256 and a coverage check. Every logit named in `checkpoints.csv`, and every logit on the grid,
must be archived. It also writes a small `…_analysis.tar.gz` holding exactly what the analysis
reads.

```python
import subprocess, sys, os
DST = f"{WORK}/out_{ARM}_f{FOLD}"
r = subprocess.run([sys.executable, "-u", "scripts/pack_dilute10_results.py", "--arm", ARM,
                    "--fold", str(FOLD), "--run-dir", OUT, "--out", DST, "--log", LOG],
                   cwd=WORK, capture_output=True, text=True)
print(r.stdout[-5000:])
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit("PACKAGING REFUSED")
for f in sorted(os.listdir(DST)):
    print(f"  {f:<56} {os.path.getsize(os.path.join(DST, f))/1e6:9.1f} MB")
print("Download every file in", DST, "-- each part is self-contained; the *_analysis.tar.gz is the one the analysis needs first.")
```

**Then go back to Cell 3, set the next `RUN` from the session table, and repeat Cells 3 → 4 → 6 → 7.**
Only `/kaggle/working` persists. Download each `out_*` folder before the session ends.
