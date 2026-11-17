# Folds 1–4 on Kaggle — the confirmatory runs

**What this is.** The run that decides the result. Four folds of DANN × n_blocks=3 at the
**budget of 4000 steps**, which is what §7.4's five-fold rule adjudicates. Fold 0 was
development and does not appear in any headline number.

**What the fold-0 development run found, so you know what these are testing.** DANN scored
mean TPR@1e-3 **0.4534** against the λ=0 control's **0.1454** — above it at 32 of 32
matched steps. That is one fold. These four decide whether it holds.

**Budget: 4000 steps** — `runs/dann_r5_budget.json`, sha256 `abf0dae0…`, hashed before any
of these folds existed. **Declared under a named rule, corroborated by the curve:**

> Budget = 2 × the λ horizon, so that half of each confirmatory run trains at fixed
> adversarial strength. Derived from DANN's own configuration; reads no metric.

That is neither a pure cost declaration (CORAL's was) nor derived from the metric. On fold
0 the two λ regimes measure very differently — early (250–2000, λ still ramping) mean 0.378
sd **0.131**, settled (2250–8000, λ fixed) mean 0.479 sd **0.058** — so the budget decides
which side of the transition all four folds report from, and 4000 puts every fold well
inside the settled regime rather than at its edge. Cell 3 verifies the hash and refuses to
run if it moved.

**Cost: ~153 min per fold, ~10.2 GPU-h for all four.** Measured from arm 2's 306.8 min at
8000 steps.

**You run ONE FOLD AT A TIME.** Cell 5 has a single `FOLD = 1` line at the top; set it,
run, and repeat for 2, 3, 4. Two folds fit comfortably in one 12 h Kaggle session (~5.1 h),
so this is **two sessions** — but nothing forces that, and each fold is an independent run,
so a session that dies costs you only the fold in flight. Cell 6 tells you at any point
which folds are done and resumes the next unfinished one.

> **The three nb3 rows sit at different budgets — 1500 / 2000 / 4000.** FreeMatch × nb3
> was confirmed at **1500** (verified from its five `run_config.json` files), CORAL × nb3
> at **2000**, DANN at 4000. **Pre-declared before these folds run:** the headline DANN
> figure is the 5-fold mean at 4000; the 5-fold mean at **step 2000** is a
> budget-sensitivity check and does not replace it, whichever is higher. Step 2000 is
> checkpoint 8 of 16 on every fold, so it costs nothing extra.

**Abort conditions.** Conditions 1 and 2 (the discriminator overpowering; degenerate
collapse) are **armed** — they catch the adversarial game genuinely breaking, and on fold 0
they never came close (5- and 9-step streaks against a 500-step sustain). Conditions 3 and
4 are **advisory on every arm**: they compute, they log, they record when they *would* have
fired, and they never stop a run. So no fold can abort at step 451 and throw away the TPR
it exists to produce.

> **Condition 3's interpretation changed on 2026-09-01** — it is a *population-separability*
> monitor, not a morphology monitor and not a domain proxy (§7.17). **That is harmless
> here**: advisory means it records a column and does not act. Nothing about these runs
> changes.

**One thing needs setting: `FOLD` in Cell 5.** Every other flag is fixed.

---

## Before you start

**One dataset.** Sync `C:\Users\krris\ML\lens_paper\kaggle\dann_r5\` — **v12**. Same
dataset as the re-score, rebuilt to carry the budget.

New notebook → **Accelerator: GPU T4 ×2** → **Add Data** → that dataset.

> **Use a T4.** Fold 0 ran on one, and §7.7 records that results are not portable across
> GPU architectures at this precision. These four are compared against each other and
> against fold 0's arm.

---

## Cell 1 — environment check (~10 s)

```python
import shutil, os
import torch

print("torch", torch.__version__, "| CUDA", torch.version.cuda)
assert torch.cuda.is_available(), "NO GPU. Settings -> Accelerator -> GPU T4 x2."
p = torch.cuda.get_device_properties(0)
print(f"  cuda:0  {p.name}  {p.total_memory/2**30:.2f} GiB")
if "T4" not in torch.cuda.get_device_name(0):
    print("\n" + "!" * 78)
    print(f"!!! NOT A T4 -- this is a {torch.cuda.get_device_name(0)}.")
    print("!!! Fold 0 ran on a T4 and these four are compared against it. 7.7 records")
    print("!!! that results are NOT portable across GPU architectures at this precision.")
    print("!" * 78 + "\n")
else:
    print("\n  T4 confirmed.")
free = shutil.disk_usage("/kaggle/working").free / 2**30
print(f"  /kaggle/working free: {free:.1f} GiB")
assert free > 10, f"only {free:.1f} GiB free; four folds write ~4 GB of checkpoints"
```

---

## Cell 2 — mount, stage, verify (~2 min)

```python
import os, sys, glob, json, shutil, hashlib

hits = []
for mp in glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True):
    try:
        m = json.load(open(mp))
    except Exception:
        continue
    if m.get("bundle") == "dann_r5":
        hits.append((mp, m))
hits.sort(key=lambda t: -int(t[1].get("version", 0)))
assert hits, ("no dann_r5 bundle under /kaggle/input. Present: "
              + str(sorted(os.listdir('/kaggle/input'))))
MAN_P, man = hits[0]
INPUT = os.path.dirname(MAN_P)
print(f"mount: {INPUT}   v{man['version']}  built {man['built_utc']}")
assert int(man.get("version", 0)) >= 12, (
    f"this is bundle v{man.get('version')}. The confirmatory folds need v12 or later -- "
    f"that is the version carrying the 4000-step budget. Detach and re-add the dataset.")

WORK = "/kaggle/working"
os.makedirs(f"{WORK}/scripts", exist_ok=True)
for f in man["scripts"]:
    shutil.copy2(os.path.join(INPUT, "scripts", f), f"{WORK}/scripts/{f}")
print(f"  copied {len(man['scripts'])} scripts (data stays in /kaggle/input)")

def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(n), b""):
            h.update(b)
    return h.hexdigest()

bad = []
for rel, info in man["checkpoints"].items():
    q = os.path.join(INPUT, rel)
    if not os.path.isfile(q):
        bad.append(f"{rel}: MISSING")
    elif info.get("sha256") and sha(q) != info["sha256"]:
        bad.append(f"{rel}: CHECKSUM MISMATCH")
assert not bad, "BUNDLE VERIFICATION FAILED:\n  " + "\n  ".join(bad)
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

## Cell 3 — the budget, verified before anything runs (~5 s)

The budget was decided and hashed **before** these folds existed. This checks the file
still self-certifies, that the code agrees with it, and that the code's pinned hash matches.
A budget that moved between its decision and the runs it governs is not a decided budget.

```python
import os, json, hashlib, sys
INPUT = os.environ["INPUT_ROOT"]
b = json.load(open(os.path.join(INPUT, "runs/dann_r5_budget.json")))
got = b.pop("sha256_of_body_without_this_field")
calc = hashlib.sha256(json.dumps(b, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert got == calc, f"BUDGET DIGEST FAILED: {got} != {calc}"

sys.path.insert(0, "/kaggle/working/scripts")
import run_dann_r5 as R5
assert R5.CONFIRM_BUDGET_DECLARED, "the code says no budget is declared. Do not run."
assert R5.CONFIRM_STEPS == b["budget_steps"], (
    f"code says CONFIRM_STEPS={R5.CONFIRM_STEPS}, the hashed budget says "
    f"{b['budget_steps']}. Do not run.")
assert R5.CONFIRM_BUDGET_SHA256 == got, (
    f"the code's pinned budget hash ({R5.CONFIRM_BUDGET_SHA256[:16]}...) does not match "
    f"the file's ({got[:16]}...). The budget moved. Do not run.")
assert b["budget_steps"] == 4000

print(f"  budget            : {b['budget_steps']} steps")
print(f"  basis             : {b['derived_or_declared']}")
print(f"  rule              : {b['THE_RULE']}")
print(f"  sha256            : {got}")
print(f"  decided           : {b['declared_utc']}   from {b['decided_from']}")
r = b["why_the_rule"]["the_two_regimes_on_fold_0"]
print(f"  fold-0 regimes    : early {r['early_lambda_still_ramping']['mean']} "
      f"sd {r['early_lambda_still_ramping']['sd']}  |  settled "
      f"{r['settled_lambda_fixed']['mean']} sd {r['settled_lambda_fixed']['sd']}  "
      f"({r['sd_ratio_early_over_settled']}x)")
co = b["corroboration_NOT_derivation"]
print(f"  corroboration     : reading {co['reading_at_the_budget']:.4f} is "
      f"{co['distance_in_settled_sds']} settled-sd from the settled mean -- unremarkable, "
      f"which is the point")
c = b["cost_accounting"]["at_the_budget"]
print(f"  cost              : {c['minutes_per_fold']} min/fold, "
      f"{c['gpu_hours_folds_1_4']} GPU-h total, {c['checkpoints_per_fold']} ckpts/fold")
lam = b["lambda_at_the_budget"]
print(f"  lambda coverage   : {lam['fraction_of_run_at_lambda_ge_0.99']:.3f} of the run at "
      f"lambda >= 0.99 (development run: "
      f"{lam['development_run_for_comparison']['fraction_at_lambda_ge_0.99']:.3f})")
pd = b["PRE_DECLARED_headline_and_sensitivity"]
print(f"\n  PRE-DECLARED (before any fold runs):")
print(f"    {pd['THE_DECLARATION']}")
print(f"    step 2000 on the confirmatory grid: {pd['step_2000_on_the_confirmatory_grid']} "
      f"-- checkpoint 8 of 16, costs nothing")
cb = b["cross_method_budgets_VERIFIED"]
print(f"  cross-method      : the three nb3 rows sit at {cb['the_three_rows_sit_at']} "
      f"(DANN {cb['dann_nb3']['budget']}, CORAL {cb['coral_nb3']['budget']}, "
      f"FreeMatch {cb['freematch_nb3']['budget']})")
assert R5.SENSITIVITY_STEP == 2000 and R5.SENSITIVITY_STEP % 250 == 0
print("\nBUDGET VERIFIED -- safe to start Cell 4.")
```

---

## Cell 4 — preflight, one fold, the minute-2 check (~3–5 min)

Builds fold 1 and **stops before training**. §7.7's discipline: a bad bundle must cost two
minutes, not five hours.

```python
import os, json, subprocess, sys
INPUT = os.environ["INPUT_ROOT"]

r = subprocess.run([sys.executable, "-u", "scripts/run_dann_r5.py",
                    "--arm", "dann_nb3", "--fold", "1", "--morph-every", "25",
                    "--abort-mode", "armed", "--max-steps", "4000",
                    "--input-root", INPUT,
                    "--emit-config", "/kaggle/working/fold1_ref.json"],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-6000:])
if r.returncode != 0:
    print(r.stderr[-4000:]); raise SystemExit("PREFLIGHT FAILED -- do not start Cell 5")

cfg = json.load(open("/kaggle/working/fold1_ref.json"))
print("\n" + "=" * 78); print("WHAT EACH FOLD WILL DO"); print("=" * 78)
for k in ("arm", "fold", "max_steps", "confirm_steps", "is_preregistered_budget",
          "abort_mode", "abort_mode_source", "dann_lambda_zero", "dann_lambda_horizon",
          "morph_every", "batch_size", "grad_checkpointing"):
    print(f"  {k:<26}: {cfg.get(k)}")
print(f"  {'advisory detectors':<26}: {cfg.get('abort_advisory_only_detectors')}")
print(f"  {'step-0 gate (fold 1)':<26}: TPR {cfg['expect_step0_tpr']:.6f} "
      f"(tol {cfg['step0_tpr_tolerance']:.6f})  AUROC {cfg['expect_step0_auroc']:.6f}")

assert cfg["max_steps"] == 4000, "the confirmatory folds run at the DECIDED budget"
assert cfg["is_preregistered_budget"] is True, "folds 1-4 run at a pre-registered budget"
assert cfg["abort_mode"] == "armed", "conditions 1 and 2 are armed on confirmatory folds"
assert cfg["dann_lambda_zero"] is False and cfg["dann_lambda_horizon"] == 2000
print("\nPREFLIGHT OK -- safe to start Cell 5.")
```

> **The per-fold step-0 gate is load-bearing and non-skippable.** The phase-1 head is
> **fold-independent** — the same file initialises all five folds — so the gate's per-fold
> TPR/AUROC is the *only* thing proving this run is on the fold it says it is. A
> mis-pointed fold would otherwise produce a perfectly plausible wrong number.

---

## Cell 5 — ONE fold (~153 min). Run it four times.

**Set `FOLD` and run. That is the only thing you change in this whole document.**

Change it to 2, run again. Then 3. Then 4. One fold per run, so nothing is in flight that
you did not just start, and a session that dies costs you only that fold.

```python
FOLD = 1          # <<<<<<  1, then 2, then 3, then 4
```

```python
import subprocess, sys, os, time, threading, json
INPUT = os.environ["INPUT_ROOT"]
assert FOLD in (1, 2, 3, 4), "folds 1-4 only -- fold 0 is development and already done"

OUT = f"/kaggle/working/runs/dann_r5_nb3_fold{FOLD}"
LOG = f"/kaggle/working/fold{FOLD}.log"

# already finished? then say so and stop, rather than silently redoing 153 minutes
rc_p = os.path.join(OUT, "run_config.json")
if os.path.isfile(rc_p):
    _r = json.load(open(rc_p))
    if _r.get("steps_completed") == 4000:
        raise SystemExit(f"fold {FOLD} is ALREADY COMPLETE (4000/4000, gate "
                         f"{(_r.get('step0_gate') or {}).get('verdict')}). "
                         f"Change FOLD and re-run this cell.")
    print(f"fold {FOLD}: an unfinished run exists at step "
          f"{_r.get('steps_completed')} -- this will RESUME it.\n")

print(f"FOLD {FOLD}   ->  {OUT}")
print(f"  budget   : 4000 steps   (~153 min on a T4, 16 checkpoints)")
print(f"  log      : {LOG}   (survives a browser disconnect)")
print(f"  progress : one fold per run of this cell; do 1, 2, 3, 4 in turn\n")

t0 = time.time(); stop = threading.Event()
def hb():
    while not stop.wait(300):
        el = (time.time() - t0) / 60
        print(f"    ... fold {FOLD}, {el:.0f} min elapsed, ~{max(0, 153-el):.0f} left",
              flush=True)
threading.Thread(target=hb, daemon=True).start()

with open(LOG, "w") as lf:
    p = subprocess.Popen([sys.executable, "-u", "scripts/run_dann_r5.py",
                          "--arm", "dann_nb3", "--fold", str(FOLD),
                          "--morph-every", "25", "--abort-mode", "armed",
                          "--max-steps", "4000",
                          "--input-root", INPUT, "--out-dir", OUT],
                         cwd="/kaggle/working", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        lf.write(line); lf.flush(); print(line, end="", flush=True)
    rc = p.wait()
stop.set()

print(f"\nfold {FOLD}: exit {rc} after {(time.time()-t0)/60:.1f} min")
if rc == 0:
    done = [k for k in (1, 2, 3, 4)
            if os.path.isfile(f"/kaggle/working/runs/dann_r5_nb3_fold{k}/run_config.json")
            and json.load(open(f"/kaggle/working/runs/dann_r5_nb3_fold{k}/run_config.json")
                          ).get("steps_completed") == 4000]
    left = [k for k in (1, 2, 3, 4) if k not in done]
    print(f"  COMPLETE: {done}      STILL TO RUN: {left}")
    if left:
        print(f"  -> set FOLD = {left[0]} and run this cell again.")
    else:
        print("  -> all four done. Go to Cell 7.")
else:
    print("  FAILED -- read the last lines above, then see the failure notes at the "
          "bottom. Re-running this cell with the same FOLD RESUMES it.")
```

> **On "one epoch at a time".** One epoch here is **187 steps** (48,000 source images ÷
> batch 256) ≈ 7 min, so the 4000-step budget is ~21.4 epochs. Stopping per epoch would
> mean 21 restarts per fold, and 187 does not divide the 250-step checkpoint grid, so
> checkpoints and resume points would no longer line up with the budget. This cell
> therefore runs **one FOLD at a time** — which is the natural "one at a time" unit here.
> If you did mean literal epochs, say so and I will add a `--max-steps` override cell; the
> budget of 4000 is what is hashed, so it would have to be reached across the pieces.

---

## Cell 6 — status, and resume anything unfinished (~1 min, or ~153 if it resumes)

Run this any time to see where you are. It resumes **one** unfinished fold — the lowest —
so it stays one-at-a-time like Cell 5.

```python
import subprocess, sys, os, time, json
INPUT = os.environ["INPUT_ROOT"]

status = {}
for k in (1, 2, 3, 4):
    rc_p = f"/kaggle/working/runs/dann_r5_nb3_fold{k}/run_config.json"
    if not os.path.isfile(rc_p):
        status[k] = "not started"; continue
    r = json.load(open(rc_p))
    n = r.get("steps_completed")
    status[k] = ("COMPLETE" if n == 4000
                 else f"incomplete at step {n} of 4000")
for k, v in status.items():
    print(f"  fold {k}: {v}")

todo = [k for k, v in status.items() if v != "COMPLETE"]
if not todo:
    print("\nAll four folds complete. Go to Cell 7.")
else:
    k = todo[0]
    OUT = f"/kaggle/working/runs/dann_r5_nb3_fold{k}"
    print(f"\nResuming fold {k} (the lowest unfinished). "
          f"{len(todo)-1} more after this one.")
    if os.path.isfile(f"{OUT}/resume/latest.pt"):
        import torch
        sch = (torch.load(f"{OUT}/resume/latest.pt", map_location="cpu",
                          weights_only=False).get("schedule") or {})
        print(f"  resuming from step {sch.get('global_step')}")
    else:
        print("  no resume checkpoint -- this starts fold "
              f"{k} FRESH")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", "scripts/run_dann_r5.py",
                        "--arm", "dann_nb3", "--fold", str(k), "--morph-every", "25",
                        "--abort-mode", "armed", "--max-steps", "4000",
                        "--input-root", INPUT, "--out-dir", OUT],
                       cwd="/kaggle/working")
    print(f"\nfold {k}: exit {r.returncode} after {(time.time()-t0)/60:.1f} min")
    print("  Run this cell again for the next unfinished fold.")
```

---

## Cell 7 — where you are, and the numbers to send back (~1 min)

**Safe to run at any point**, with any number of folds done. It skips folds that have not started, flags any that are incomplete, and tells you what to run next. Copy the whole printed block into your reply.

```python
import json, os, glob

print("=" * 84)
print("CONFIRMATORY FOLDS -- DANN x nb3 at the budget B = 4000")
print("=" * 84)

rows, missing, partial = [], [], []
for k in (1, 2, 3, 4):
    OUT = f"/kaggle/working/runs/dann_r5_nb3_fold{k}"
    rc_p = f"{OUT}/run_config.json"
    if not os.path.isfile(rc_p):
        missing.append(k)
        print(f"\nFOLD {k}\n  NOT STARTED")
        continue
    rc = json.load(open(rc_p))
    g = rc.get("step0_gate") or {}
    w = rc.get("abort_watch") or {}
    n = rc.get("steps_completed")
    if n != 4000:
        partial.append(k)
    print(f"\nFOLD {k}{'' if n == 4000 else '   *** INCOMPLETE ***'}")
    print(f"  steps        : {n} / {rc.get('max_steps')}"
          f"   checkpoints {rc.get('n_checkpoints_saved')} of "
          f"{rc.get('n_checkpoints_expected')}")
    print(f"  budget       : confirm_steps {rc.get('confirm_steps')}, "
          f"is_preregistered_budget {rc.get('is_preregistered_budget')}")
    wm = rc.get("wall_minutes")
    print(f"  wall         : {wm:.1f} min" if wm else "  wall         : n/a",
          f"  resumes {len(rc.get('resume_events') or [])}"
          f"   fp16 skipped {rc.get('fp16_skipped_cumulative')}")
    if g:
        print(f"  step-0 gate  : {g.get('verdict')}  TPR {g.get('tpr'):.6f} "
              f"(expect {g.get('expected_tpr'):.6f}, "
              f"|diff| {g.get('tpr_abs_diff'):.6f})")
    print(f"  aborted      : {rc.get('aborted')}   reason {rc.get('abort_reason')}")
    print(f"  fired        : ARMING {w.get('fired_arming')}   "
          f"ADVISORY {w.get('fired_advisory')}")
    rows.append({"fold": k, "gate": g.get("verdict"), "aborted": rc.get("aborted"),
                 "steps": n, "arming": w.get("fired_arming"),
                 "advisory": w.get("fired_advisory")})

print("\n" + "=" * 84)
print("PROGRESS")
print("=" * 84)
done = [r["fold"] for r in rows if r["steps"] == 4000]
print(f"  complete        : {done}  ({len(done)} of 4)")
if partial:
    print(f"  INCOMPLETE      : {partial}   -- resume with Cell 6")
if missing:
    print(f"  not started     : {missing}   -- set FOLD and run Cell 5")

if rows:
    print("\n" + "=" * 84)
    print("INTEGRITY (folds that have run)")
    print("=" * 84)
    print(f"  gates PASS      : {sum(1 for r in rows if r['gate'] == 'PASS')} "
          f"of {len(rows)}")
    print(f"  aborted         : {[r['fold'] for r in rows if r['aborted']] or 'none'}")
    print(f"  ARMING fired    : "
          f"{[(r['fold'], r['arming']) for r in rows if r['arming']] or 'none'}")
    print(f"  advisory fired  : "
          f"{[(r['fold'], r['advisory']) for r in rows if r['advisory']] or 'none'}")
    print("\n  NOTE: advisory detectors firing is EXPECTED and stops nothing -- fold 0's")
    print("  probe fell early and stayed down. Only ARMING detectors can abort.")

if len(done) == 4:
    print("\n  ALL FOUR COMPLETE -- go to Cell 8 and package.")
else:
    print(f"\n  {4 - len(done)} fold(s) still to go. This cell is safe to re-run at any "
          f"point.")

print("\nNO TPR IS COMPUTED HERE. metric_hygiene: checkpoints are SCORED (logits")
print("persisted) but never MEASURED. The fold-mean is computed locally from the")
print("logits, after all four folds are home.")
print("\nSEND THE WHOLE OUTPUT OF THIS CELL BACK.")
```

---

## Cell 8 — package for download (~10 min)

Packs every fold that is **complete**, skipping ones not started or still part-way. **Refuses** any run whose `checkpoints.csv` names an artifact the archive does not contain — the per-checkpoint logits are what the fold-mean is computed from, and they went missing once already.

```python
import subprocess, sys, os, json

# only fold dirs that exist AND finished. Packing a partial run gives an archive that
# looks complete and is not -- and packing a missing one just crashes.
OUTS, skipped = [], []
for k in (1, 2, 3, 4):
    d = f"/kaggle/working/runs/dann_r5_nb3_fold{k}"
    rc_p = f"{d}/run_config.json"
    if not os.path.isfile(rc_p):
        skipped.append((k, "not started")); continue
    n = json.load(open(rc_p)).get("steps_completed")
    if n != 4000:
        skipped.append((k, f"incomplete at {n}/4000")); continue
    OUTS.append(d)

print(f"  packing {len(OUTS)} complete fold(s): "
      f"{[os.path.basename(d) for d in OUTS] or 'NONE'}")
for k, why in skipped:
    print(f"  skipping fold {k}: {why}")
assert OUTS, "no complete fold to package -- run Cell 5 first"
if skipped:
    print("\n  NOTE: this is a PARTIAL package. Re-run this cell after the remaining")
    print("  folds finish; it overwrites the output directory with the full set.")

DST = "/kaggle/working/dann_r5_folds_out"
r = subprocess.run([sys.executable, "-u", "scripts/build_dann_r5_bundle.py",
                    "--pack-results", *OUTS, "--results-out", DST,
                    "--logs", "/kaggle/working"],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-8000:])
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit("PACKAGING REFUSED -- see the message above")

tot = 0
for f in sorted(os.listdir(DST)):
    p = os.path.join(DST, f); sz = os.path.getsize(p); tot += sz
    print(f"  {f:<50} {sz/1e6:8.1f} MB")
print(f"\n  {DST}   TOTAL {tot/1e9:.2f} GB")
print(f"  {len(OUTS)} of 4 folds packaged. Download every part from the Output panel.")
```

---

## If something goes wrong

| what you see | what to do |
|---|---|
| **Cell 2: "this is bundle v11"** (or older) | Stale mount. v12 carries the 4000-step budget and the pre-declaration. Detach and re-add. |
| **Cell 3: BUDGET DIGEST FAILED**, or the pinned hash does not match | The budget moved after it was hashed. **Stop and report it** — a budget that changed between its decision and the runs it governs is not a decided budget. |
| **Cell 3: code/budget disagree on CONFIRM_STEPS** | Same thing from the other side. Stop. |
| **Cell 4 or 5: step-0 gate ABORT** | The gate is the only proof a run is on the fold it claims. Send the printed per-fold TPR/AUROC and stop. **Fix by re-scoring the table on the venue, never by widening the gate.** |
| **A fold aborts with an ARMING detector fired** | Conditions 1 or 2 — the adversarial game genuinely broke. On fold 0 they never came within 490 steps of firing, so this is real. Send `abort_reason` and the fold number; do not re-run it to get a different answer. |
| **Advisory detectors fired** | **Expected, and it stops nothing.** Fold 0's probe fell early and stayed down. It is recorded, not acted on. |
| **Session dies mid-fold** | **Cell 6.** It skips completed folds and resumes the interrupted one. Do not delete output directories. |
| **CUDA out of memory** | Gradient checkpointing is mandatory at nb3 and is already on; fold 0 peaked at 3.3 GB. **Do not reduce the batch — 256 is pinned by §7.4 and changing it invalidates the cross-method comparison.** Report it. |
| **A fold much faster than ~153 min** | Check `steps_completed` against 4000 in Cell 7 — usually it means fewer steps ran than you think. |
| **Cell 8: packaging refused** | It found a named artifact missing from the archive. That is the collection defect the check exists for. Report it rather than downloading a partial set. |

---

## What happens after this

1. You send back Cell 7's block and the download.
2. The **fold-mean** TPR@1e-3 over folds 1–4 is computed locally from the persisted
   logits — never pooled, since these are per-fold models (§2).
3. That number, against CORAL × nb3's 0.4348 (budget **2000**) and FreeMatch × nb3's
   0.4169 (budget **1500**), is what §7.4's five-fold rule adjudicates. **The three rows
   sit at 1500 / 2000 / 4000** and that must be disclosed alongside the comparison.
4. The **step-2000 5-fold mean** is computed from the same persisted logits — checkpoint 8
   of 16, already written — and reported as the pre-declared budget-sensitivity check.
4. Fold 0's +0.33 played no part in it, and neither did any number in this document.
