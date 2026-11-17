# BNM round 6, Stage 2 — venue re-measurement, then the γ sweep

**Two jobs, in order.** First a ~2-minute measurement that fixes the γ grid; then a
three-arm fold-0 sweep that selects γ. Nothing here reads TPR, AUROC, or any test-set
quantity — the only test-set number computed anywhere is the step-0 identity gate.

**Why the re-measurement is not optional.** The grid was measured locally on **CPU in
fp32**: ‖∇CE‖ 6.033024, ‖∇BNM‖@γ=1 60.05960, ratio 9.955, giving γ ∈ {0.010045, 0.050225,
0.200901}. The confirmatory runs are **fp16 on a T4**, and §7.7 records that results are
not portable across GPU architectures at this precision — gradient norms are exactly the
quantity that moves. **The CPU numbers are a sanity reference and must not be hashed.**

---

## Before you start

Same dataset as always — it is still named **`lens-paper-coral-r4`** (historical; the
bundle inside is `dann_r5`). Sync `kaggle/dann_r5/` so it carries the round-6 scripts
(`bnm_round6.py`, `run_bnm_r6.py`, `test_bnm_round6.py`), then attach it to a **T4 ×2**
notebook.

**Cell A below is self-sufficient**: it resolves the mount by content, stages every
script, checks the bundle is new enough, and sets `INPUT_ROOT` for the later cells. You do
not need any cell from `KAGGLE_LAMBDA0_R6.md` first — running its Cell 1 (the GPU/T4
check) is still worth doing, but nothing here depends on it.

> **The bundle must be at v15 or later.** v14 was built for the lambda=0 run and
> predates every round-6 script. **Cell 2 will not catch that** -- it validates
> against the manifest on the mount, so an older bundle passes its own check.
> Cell A does catch it, against a list held in this document.

---

## Cell A — the tests must pass ON THE VENUE first (~1 min)

The wiring is CPU-verified locally; this confirms it survives the venue's torch build
before any GPU time is spent on a measurement that depends on it.

**Cell 2 cannot catch a stale mount here, and that is worth understanding rather than
working around.** Cell 2 asserts the scripts named in `MANIFEST["cell_requires"]` -- but it
reads that list *from the mounted manifest*. An older bundle lists an older set, so it
passes its own check and copies its own scripts happily. The round-6 files are then simply
absent, and the first symptom is a bare `No such file or directory` from whatever runs
next. So this cell checks the mount against a list held **in this document**, which no
mounted manifest can shrink.

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

## Cell B — the venue measurement (~2 min GPU)

`--measure-step0` does one forward/backward and **exits before any optimiser step**. It is
metric-blind by construction: nothing on that path computes a metric or scores the test
pool (`test_bnm_round6.T11` asserts it, with the scoring path as the known-null).

```python
import subprocess, sys, os, json
INPUT = os.environ["INPUT_ROOT"]
OUT = "/kaggle/working/runs/bnm_r6_step0_gradnorms_t4fp16.json"

r = subprocess.run([sys.executable, "-u", "scripts/run_bnm_r6.py",
                    "--fold", "0", "--measure-step0",
                    "--input-root", INPUT, "--step0-out", OUT],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-5000:])
if r.returncode != 0:
    print(r.stderr[-4000:]); raise SystemExit("MEASUREMENT FAILED")

d = json.load(open(OUT))
CPU = {"ce": 6.033024, "bnm": 60.05960, "ratio": 9.955140,
       "grid": [0.010045, 0.050225, 0.200901]}

print("=" * 78)
print("VENUE (T4/fp16) vs LOCAL (CPU/fp32)")
print("=" * 78)
print(f"  {'quantity':<26}{'venue':>16}{'cpu':>16}{'venue/cpu':>12}")
for k, key in (("||grad CE||", "ce_grad_norm"),
               ("||grad BNM|| at gamma=1", "bnm_grad_norm_at_gamma_1"),
               ("ratio BNM/CE", "ratio_bnm_over_ce_at_gamma_1")):
    v = d[key]; c = CPU["ce" if "CE||" in k else "bnm" if "BNM||" in k else "ratio"]
    print(f"  {k:<26}{v:>16.6f}{c:>16.6f}{v/c:>12.4f}")

print(f"\n  device {d['device']}  fp16={d['fp16']}  gpu={d['gpu_name']}")
assert str(d["device"]).startswith("cuda"), "NOT ON GPU -- this is the whole point"
assert d["fp16"] is True, "NOT fp16 -- the confirmatory runs are fp16"
assert d["exited_before_any_optimiser_step"] and d["reads_a_metric"] is False

print(f"\n  THE GRID TO USE (from the venue file, NOT the CPU one):")
for f, g in zip(d["fractions"], d["gamma_grid"]):
    print(f"    {f*100:>5.0f}% of the CE gradient norm  ->  gamma = {g:.6g}")

move = abs(d["ratio_bnm_over_ce_at_gamma_1"] / CPU["ratio"] - 1.0)
print(f"\n  the ratio moved {move*100:.1f}% from CPU/fp32 to T4/fp16.")
print("  A LARGE move is not a problem -- it is a 7.7 data point worth a line in")
print("  PROJECT.md. What matters is that the grid comes from THIS file.")
print("\nSEND THIS WHOLE BLOCK BACK.")
```

---

## Cell C — the γ sweep, three arms on fold 0 (~3 × 40 min)

Each arm runs a short fold-0 segment and logs **loss magnitudes and prediction/feature
statistics only**. The `GammaInvariant` runs throughout: if γ ever varies with the step the
arm stops immediately and says so.

> **Set `SWEEP_STEPS` from the cost you are willing to spend, not from a curve.** 1000
> steps is ~40 min/arm on a T4, ~2 h for three. The sweep reads no metric, so its length
> cannot be tuned toward a result — but declare it here rather than adjusting it after
> seeing arm 1.

```python
SWEEP_STEPS = 1000        # declared before any arm runs
```

```python
import subprocess, sys, os, json, time, csv
INPUT = os.environ["INPUT_ROOT"]
d = json.load(open("/kaggle/working/runs/bnm_r6_step0_gradnorms_t4fp16.json"))
GRID = d["gamma_grid"]
print(f"grid from the VENUE file: {GRID}\n")

results = []
for g in GRID:
    out = f"/kaggle/working/runs/bnm_r6_sweep_g{g:.6g}"
    log = f"/kaggle/working/bnm_sweep_g{g:.6g}.log"
    print(f"=== gamma {g:.6g} -> {out}", flush=True)
    t0 = time.time()
    with open(log, "w", buffering=1) as lf:
        p = subprocess.Popen([sys.executable, "-u", "scripts/run_bnm_r6.py",
                              "--fold", "0", "--gamma", str(g),
                              "--max-steps", str(SWEEP_STEPS),
                              "--input-root", INPUT, "--out-dir", out],
                             cwd="/kaggle/working", stdout=lf,
                             stderr=subprocess.STDOUT, text=True)
        shown = 0
        while p.poll() is None:
            time.sleep(60)
            lines = open(log, errors="replace").readlines()
            for ln in lines[shown:]:
                print(ln, end="", flush=True)
            shown = len(lines)
        rc = p.wait()
    for ln in open(log, errors="replace").readlines()[shown:]:
        print(ln, end="")
    print(f"  gamma {g:.6g}: rc={rc}, {(time.time()-t0)/60:.1f} min", flush=True)
    if rc != 0:
        raise SystemExit(f"arm gamma={g} FAILED -- read the log above")
    results.append({"gamma": g, "out": out})

print("\nall three arms finished")
```

---

## Cell D — selection, and the 704/78 diagnostic (~10 s)

Applies the criterion declared **before** the sweep: ‖A‖\* moves materially from its 9.77
init; predicted-positive fraction stays above the **gating floor 0.0292**; the frozen probe
does not collapse. Two or more pass → the smaller. None → one ×4 extension. That failing
too → stop and report.

```python
import sys, os, json, csv
sys.path.insert(0, "/kaggle/working/scripts")
import bnm_round6 as B

anchors = B.stage0a_anchors()
band = B.collapse_band(anchors["init_pred_pos_frac"])

arms = []
print(f"{'gamma':>10}{'||A||* end':>12}{'pred-pos':>10}"
      f"{'nuc gap':>10}{'ent gap':>10}{'probe':>9}")
for r in results:
    rows = list(csv.DictReader(open(os.path.join(r["out"], "checkpoints.csv"))))
    last = rows[-1]
    cfg = json.load(open(os.path.join(r["out"], "run_config.json")))
    arms.append({"gamma": r["gamma"],
                 "nuclear_norm_final": float(last["bnm_nuclear_norm"]),
                 "pred_pos_frac_final": float(last["bnm_pred_pos_frac"]),
                 # NO DEFAULT. An absent column would make condition 3 pass silently,
                 # which is worse than not having the condition at all.
                 "frozen_probe_auroc_final": float(last["frozen_probe_auroc"]),
                 "frozen_probe_auroc_step0": float(rows[0]["frozen_probe_auroc"])})
    print(f"{r['gamma']:>10.6g}{float(last['bnm_nuclear_norm']):>12.4f}"
          f"{float(last['bnm_pred_pos_frac']):>10.4f}"
          f"{float(last['nuclear_gap']):>10.4f}{float(last['entropy_gap']):>10.4f}"
          f"{arms[-1]['frozen_probe_auroc_final']:>9.4f}")
print("")
print("  head instrument (~0.57 scale) vs frozen probe (~0.92) -- 7.19: never")
print("  quote one as the other. Condition 3 reads the FROZEN PROBE column.")

sel = B.select_gamma(arms, band=band, init_nuclear=anchors["init"])
print("\n" + "=" * 78)
print("SELECTION")
print("=" * 78)
for a in sel["arms"]:
    print(f"  gamma {a['gamma']:<10.6g} c1_moved={a['c1_nuclear_moved']!s:<6}"
          f" c2_floor={a['c2_floor_ok']!s:<6} c3_probe={a['c3_probe_ok']!s:<6}"
          f" -> {'PASS' if a['passes'] else 'fail'}"
          f"{'   [advisory ceiling breached]' if not a['c2_advisory_ceiling_ok'] else ''}")
print(f"\n  passing: {sel['n_passing']}   selected: {sel.get('selected')}")
if not sel.get("decided"):
    print(f"  ACTION: {sel['action']}")
    if sel["action"] == "EXTEND":
        print(f"  next grid: {sel['next_grid']}  (one extension permitted, then STOP)")
    else:
        print(f"  reason: {sel['reason_code']}")
key = "ADVISORY_CEILING_BREACHED_BY_SELECTED_ARM"
if key in sel:
    print(f"\n  *** {key} ***")
    print(f"  {json.dumps(sel[key], indent=2)}")

print("\n  THE 704/78 MEMORISATION DIAGNOSTIC (size-matched, 78 vs 78):")
print("    a POSITIVE nuclear gap with a NEGATIVE entropy gap is the MEMORISING")
print("    signature; both slices moving together is GENERALISING.")
for r in results:
    rows = list(csv.DictReader(open(os.path.join(r["out"], "checkpoints.csv"))))
    print(f"\n    gamma {r['gamma']:.6g}:")
    print(f"      {'step':>6}{'trained-on':>12}{'held-out':>11}{'nuc gap':>10}"
          f"{'ent gap':>10}{'pos gap':>10}")
    for row in rows:
        print(f"      {int(float(row['step'])):>6}"
              f"{float(row['trainedon_nuclear']):>12.4f}"
              f"{float(row['heldout_nuclear']):>11.4f}"
              f"{float(row['nuclear_gap']):>10.4f}"
              f"{float(row['entropy_gap']):>10.4f}"
              f"{float(row['pred_pos_frac_gap']):>10.4f}")

json.dump({"selection": sel, "sweep_steps": SWEEP_STEPS,
           "grid_source": "runs/bnm_r6_step0_gradnorms_t4fp16.json"},
          open("/kaggle/working/runs/bnm_r6_gamma_selection.json", "w"), indent=2)
print("\nwrote runs/bnm_r6_gamma_selection.json -- SEND THIS WHOLE CELL BACK.")
```

---

## Cell E — γ invariant check, and package (~5 min)

```python
import json, os, glob
print("GAMMA INVARIANT, every arm:")
for r in results:
    cfg = json.load(open(os.path.join(r["out"], "run_config.json")))
    inv = cfg["gamma_invariant"]
    print(f"  gamma {r['gamma']:<10.6g} held={inv['held']!s:<6}"
          f" n_obs={inv['n_observations']:<6}"
          f" eff=[{inv['gamma_effective_min']:.8g}, {inv['gamma_effective_max']:.8g}]")
    assert inv["held"], f"THE INVARIANT FAILED on gamma={r['gamma']}: {inv['first_divergence']}"
print("\n  held on every arm -- gamma never varied with the step.")

import subprocess, sys
DST = "/kaggle/working/bnm_r6_sweep_out"
dirs = [r["out"] for r in results]
p = subprocess.run([sys.executable, "-u", "scripts/build_dann_r5_bundle.py",
                    "--pack-results", *dirs, "--results-out", DST],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(p.stdout[-3000:])
```

Then use `KAGGLE_LAMBDA0_R6.md`'s **Cell 7a** for download links, or the Kaggle CLI —
`kaggle kernels output navneet78/<slug> -p runs/bnm_r6_incoming`. The sweep's analysis
files are small; the checkpoints are optional.

---

## What comes back

1. **Cell B's block** — the venue gradient norms, how far they moved from CPU, and the
   grid they imply.
2. **Cell D's block** — which γ passed which criterion, the selection, any advisory-ceiling
   escalation, and the 704/78 gap at every γ and step.
3. **Cell E's invariant line** — `held=True` on all three arms.

Stage 3 is written and hashed only after that: the pre-registration needs the selected γ,
the recorded 704/78 predicted direction (already in §7.20), and the declared extension
rule (already in §7.20) all in place.
