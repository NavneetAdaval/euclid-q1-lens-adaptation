# Kaggle cells — FreeMatch × n_blocks=3, fold-0 diagnostic (BRIEF_freematch_nb3_fold0.md)

**Session settings (both required):**
- Accelerator: **GPU T4 × 2**
- Internet: **ON** (Zoobot downloads from HuggingFace on first use)

**Dataset:** build and verify the upload bundle first —
`python scripts/build_freematch_bundle.py`, then `kaggle datasets create -p kaggle/freematch
-r zip`. See `kaggle/freematch/README.md`. Bump `--version` on every re-upload: Kaggle
serves the previous version silently if the bump did not take, and Cell 1 prints
`VERSION.txt` so you can see which copy you actually mounted.

**What this session produces:** two runs, **sequentially on one GPU, control first** —
`freematch_fold0_nb0_control` (frozen, 2000 steps) then
`freematch_fold0_nb3_unfrozen` (n_blocks=3, grad-checkpointing, 8000 steps) — both
initialised from the converged round-1 FreeMatch fold-0 step-8000 head, both at 64 curve
points, 16 retained weight checkpoints, per-curve-point derived logits, FreeMatch-internal
diagnostics, the memorization probe, and the collapse gate's firing decision.

**This is an exploratory fold-0 run.** Its CI half-width is ~0.10–0.12 (PROJECT.md §7.8) and
§7.8 already showed fold 0 cannot resolve the unfreeze effect *even with oracle selection*.
No number from this session is a result for the paper; the output is a decision about the
diagnostics and step budget for a later 5-fold confirmatory run.

---

## Before you start — read this correction

The brief's §2 gate value of **0.357 is wrong for this configuration.** That number is
PROJECT.md §6's Drive listing for `freematch_fold0_standard`, the earlier fold-0 *variant
screen* — a different run at a different protocol. The matched round-1 run whose fold mean
is the reported 0.348 ± 0.024 is `runs/bakeoff_freematch_5fold/`, and its fold-0 step_8000
checkpoint — **the head both runs here initialise from** — re-scores to

| quantity | value |
|---|---|
| TPR@FPR=1e-3 | **0.387755** |
| AUROC | 0.973624 |
| ‖head‖₂ | 29.414692 |
| head params | 1,921 |

verified locally against `data/feature_cache` and matching that run's own `results.json`.
The driver defaults to 0.387755 and the local smoke reproduced it **exactly, at tolerance
0**, on both the cached path (nb0) and the live-encode path (nb3). Nothing to change — this
note exists so the mismatch with the brief is not rediscovered mid-session.

---

## Two deliberate deviations from a literal reading of the brief

**1. The control keeps the 8000-step SCHEDULE while stopping at 2000.**
FreeMatch's mu cosine warmup and its hard consistency-warmup gate are fractions of the
*planned* run. A control with a 2000-step schedule would have a compressed mu curve, and its
step 250 would not be comparable to the unfrozen run's step 250 — which is the entire
comparison the control exists to make. `--schedule-steps 8000` is pinned for both runs, so
the control is a genuine truncation: consistency is withheld for the same first 800 steps in
both. `schedule_total_steps` is a **pinned** key precisely so `max_steps` can be exempt.

**2. Weight checkpoints every 500, curve points every 125.**
The brief asks for both (§5.1: 16 checkpoints; §5.2/§5.3: 64 curve points). A curve point
still needs a file on disk to score, so non-retained points write one reused scratch file
that is deleted at the end. Retaining all 64 at 59.4 MB would cost ~3.8 GB for artifacts the
derived-logit dump already replaces (§7.7's scaling flag).

---

## Ordering decision: SEQUENTIAL, control first

**This reverses the concurrent T4×2 pattern used in `KAGGLE_TASK5_CELLS.md`.** Both runs go
on one GPU, one after the other, control first.

**Why not concurrent.** `num_workers` is fixed when the DataLoader is constructed. Under
concurrency the unfrozen run would be built with ~2 of the session's ~4 vCPUs and keep 2 for
its entire 4–6 hours — the workers do not expand when the control exits an hour in.
FreeMatch's unlabeled branch computes weak *and* strong augmentation, so the CPU-side work is
2 views × batch 256 per step on top of the source view. If the input pipeline binds at all,
concurrency costs more than the hour it saves. Sequential is ~1 h + ~5 h ≈ 6 h, comfortably
inside the 12 h cap, and each run gets all the workers.

**Why the control first.** It is the cheap gate on interpretability. If its curve is not flat
across 250/500/1000/1500/2000, the extra-steps confound is live and the unfrozen result
cannot be attributed to unfreezing — worth knowing before committing five hours to a number
that will not be attributable. Cell 3 prints the flatness verdict between the two runs and
lets you stop there.

Cell 2 settles whether concurrency was ever viable, by sampling GPU utilization during the
smoke. A run sitting at high GPU utilization is compute-bound and would have tolerated
sharing; low or spiky utilization means the dataloader binds and sequential was the right
call. That answer is reusable for every image-space run in this project.

---

## Wall-clock projection — MEASURE IT, do not trust this table

| stage | basis | time |
|---|---|---|
| setup (pip, unzip, encoder download) | Task-5 session | ~10 min |
| Cell 2 pre-flight + 200-step smoke | | ~15 min |
| **run 1** — `nb0_control`, 2000 steps + 17 cached curve points | ~¼ of a frozen 8000-step run (§7.7: 63.8 min), all workers | ~20 min |
| **run 2** — `nb3_unfrozen`, 8000 steps + **64** live-encode curve points | extrapolated from §7.7's Baseline A nb3 (167.8 min @ 32 points), + FreeMatch's extra forwards, + double the curve density | **4–6 h, UNVERIFIED** |
| packaging | ~1.0 GB of nb3 checkpoints | ~5 min |
| **TOTAL** | | **≈ 5–7 h** |

**The nb3 figure is an extrapolation and it is the one that could overrun the 12 h cap.**
§7.7 is explicit that local benchmarks do not transfer and that timings are valid only within
one process. Cell 2 times real steps and real curve points on *this* session's T4 and
projects the total. If the projection exceeds ~8 h, drop to `--curve-every 250` (32 points,
the round-1 density) — that changes no trained weight, only how often the model is scored.

Kaggle: 12 h session cap, ~30 GPU-hours/week. Verify your current quota before launching.

**GPU T4 × 2 is still the right accelerator setting** even though only one GPU is used: the
two-GPU sessions are what carry the larger CPU/RAM allocation. Cell 2 reports
`os.cpu_count()` so the `num_workers` choice is grounded in what the session actually has.

---

## Cell 1 — setup, GPU assert, phase-1 head check

```python
import os, sys, json, glob, shutil, subprocess, time
from pathlib import Path
import numpy as np

WORK = Path("/kaggle/working")
cands = list(Path("/kaggle/input").rglob("MANIFEST.json"))
assert cands, "no MANIFEST.json under /kaggle/input -- is the dataset attached?"
SRC = cands[0].parent
print("dataset mount :", SRC)

man = json.loads((SRC/"MANIFEST.json").read_text())
print("bundle version:", (SRC/"VERSION.txt").read_text().strip()
      if (SRC/"VERSION.txt").is_file() else "(none)")
for need in ("run_unfreeze_freematch.py", "unfreeze_instrument.py",
             "report_unfreeze_freematch.py", "pseudolabel_precision.py",
             "run_freematch.py", "live_encode.py", "curve_hook.py",
             "package_screen_results.py"):
    assert need in man["scripts"], (
        f"{need} is not in the uploaded scripts -- re-upload the WHOLE scripts/ folder "
        f"(the import closure is 20+ files and stale scripts silently run old behaviour)")

!pip -q install "timm==1.0.26"
import torch, timm
print("torch", torch.__version__, "| timm", timm.__version__)

n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs (Accelerator = GPU T4 x2), got {n}"
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}  {p.total_memory/2**30:.1f} GB  sm_{p.major}{p.minor}")
    assert p.major >= 7, f"GPU{i} is sm_{p.major}{p.minor} (P100?) -- T4 only (§7.7)"

SCRIPTS = WORK/"scripts"; SCRIPTS.mkdir(parents=True, exist_ok=True)

# LOCATE the scripts by SEARCHING for a known member, never by assuming a layout. Kaggle
# may deliver `scripts/` as a directory, nested one or more levels down, or as an archive
# it did not auto-extract -- and the old `else SRC` fallback silently pointed at the mount
# root, which fails on the first filename in the manifest.
PROBE = "run_unfreeze_freematch.py"
INPUT = Path("/kaggle/input")
hits = list(INPUT.rglob(PROBE))
srcdir = hits[0].parent if hits else None

if srcdir is None:                      # not extracted -- find the archive holding it
    import zipfile, tarfile
    arcs = [p for p in list(INPUT.rglob("*.zip")) + list(INPUT.rglob("*.tar"))
            if p.name != "data.zip"]
    for arc in arcs:
        try:
            names = (zipfile.ZipFile(arc).namelist() if arc.suffix == ".zip"
                     else tarfile.open(arc).getnames())
        except Exception:
            continue
        if any(n.endswith(PROBE) for n in names):
            dest = WORK/"bundle_scripts"          # /kaggle/input is READ-ONLY
            print(f"extracting {arc.name} -> {dest}", flush=True)
            shutil.unpack_archive(str(arc), str(dest))
            h = list(dest.rglob(PROBE))
            srcdir = h[0].parent if h else None
            break
assert srcdir is not None, (
    f"{PROBE} is nowhere under /kaggle/input, extracted or archived. The dataset does not "
    f"contain the scripts folder -- re-upload with: kaggle datasets version -p "
    f"kaggle/freematch -r zip")
print("scripts source:", srcdir)

copied, missing = 0, []
for name in man["scripts"]:
    s = srcdir/name
    if not s.is_file():                 # per-file fallback if the folder is split up
        f = list(INPUT.rglob(name)) or list(WORK.rglob(name))
        s = f[0] if f else None
    if s is None:
        missing.append(name); continue
    shutil.copy2(s, SCRIPTS/name); copied += 1
print(f"copied {copied}/{len(man['scripts'])} scripts -> {SCRIPTS}")

# Only the IMPORT CLOSURE has to be present. The bundle ships the whole scripts folder as
# belt-and-braces (§7.7: the closure has grown twice mid-project), so a stray missing
# one-off analysis script is not a reason to abort a session.
required = set(man.get("import_closure") or man["scripts"])
blocking = sorted(set(missing) & required)
assert not blocking, f"missing REQUIRED scripts: {blocking}"
if missing:
    print(f"NOTE: {len(missing)} non-closure script(s) absent, not blocking: {missing[:6]}")

# ---- data root: LOCATED by content, not assumed from a path shape ----
# Kaggle may present the data as <mount>/data, nested one deeper (<mount>/data/data), or
# leave data.zip unextracted. Rather than guess the depth, find the directory that actually
# holds splits/splits.json AND the feature cache -- the same fix as the scripts lookup, for
# the same reason: a layout assumption fails loudly at best and picks a half-populated
# directory at worst.
NEED = ["splits/splits.json", "feature_cache/source_train.npz", "feature_cache/val.npz",
        "feature_cache/test_negatives_shared.npz", "feature_cache/test_pos_fold0.npz",
        "feature_cache/target_fold0.npz", "q1_lenses_processed/lens_cutouts.npz"]

def find_data_root():
    seen = set()
    for base in (WORK, INPUT):              # prefer an already-unzipped copy in /working
        for sj in base.rglob("splits.json"):
            if sj.parent.name != "splits":
                continue
            root = sj.parent.parent
            if root in seen:
                continue
            seen.add(root)
            if all((root/n).exists() for n in NEED):
                return root
    return None

DATA = find_data_root()
if DATA is None:
    z = next(iter(glob.glob(str(INPUT/"**"/"data.zip"), recursive=True)), None)
    assert z, ("no directory under /kaggle/working or /kaggle/input holds "
               "splits/splits.json + the feature cache, and there is no data.zip to "
               "unpack. Check the dataset actually contains data.zip or an extracted data/.")
    # subprocess, not `!unzip`: an IPython shell escape is only valid at the START of a
    # line, so the `print(...); !unzip ...` form fails to parse in the notebook too.
    print("unzipping", z, flush=True)
    assert subprocess.run(["unzip","-q","-o",z,"-d",str(WORK)]).returncode == 0, \
        f"unzip failed for {z}"
    DATA = find_data_root()

assert DATA is not None, "data.zip unpacked but no complete data root appeared"
missing = [n for n in NEED if not (DATA/n).exists()]
assert not missing, f"missing {missing} under {DATA}"
print("data root     :", DATA)
os.environ["LENS_DATA_ROOT"] = str(DATA)

# ---- BRIEF §2 PRECONDITION: the phase-1 head must EXIST and LOAD before anything else.
# §7.7 records a packaging step that copied only step_8000.pt for some runs and an A1 study
# that lost its checkpoints outright. runs/ is NOT packed into data.zip, so these two files
# are uploaded explicitly.
#
# IDENTIFIED BY SHA256, NOT BY PATH. Both files are named step_8000.pt, so if the uploader
# flattened runs/ the way it flattened scripts/, a path-shaped lookup either misses or --
# worse -- silently returns the wrong one, and the agreement column is only comparable
# across runs when every run uses the SAME reference file. MANIFEST.json records both
# hashes, so content decides.
import hashlib
def _sha(p, chunk=1<<20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

WANT = {v["sha256"]: k for k, v in man.get("checkpoints", {}).items()
        if isinstance(v, dict) and v.get("sha256")}
found = {}
for p in INPUT.rglob("*.pt"):
    if p.stat().st_size > 4_000_000:        # the two we want are ~17 KB
        continue
    h = _sha(p)
    if h in WANT and WANT[h] not in found:
        found[WANT[h]] = str(p)

HEAD_KEY = "runs/bakeoff_freematch_5fold/fold0/checkpoints/step_8000.pt"
REF_KEY  = "runs/bakeoff_baseline_A/checkpoints/step_8000.pt"
HEAD, REF = found.get(HEAD_KEY), found.get(REF_KEY)
if not (HEAD and REF):                      # older bundle without hashes -> path lookup
    HEAD = HEAD or next(iter(glob.glob(str(INPUT/"**"/"bakeoff_freematch_5fold"/"**"/"step_8000.pt"),
                                       recursive=True)), None)
    REF  = REF  or next(iter(glob.glob(str(INPUT/"**"/"bakeoff_baseline_A"/"**"/"step_8000.pt"),
                                       recursive=True)), None)
assert HEAD, ("the round-1 FreeMatch fold-0 step_8000 checkpoint is not in the dataset. "
              "Both runs initialise from it; upload it (it is ~17 KB).")
assert REF, "Baseline A step_8000 (the agreement reference) is not in the dataset"

sys.path.insert(0, str(SCRIPTS))
import evaluate as ev
head, std, _ = ev.load_checkpoint(HEAD, "cpu")
l2 = float(torch.sqrt(sum((p.detach().float()**2).sum() for p in head.parameters())))
print(f"phase-1 head  : {HEAD}\n  ||head||_2 = {l2:.6f}  (expected 29.414692)")
assert abs(l2 - 29.414692) < 1e-4, "this is NOT the round-1 FreeMatch fold-0 head"
print("agreement ref :", REF)
assert HEAD != REF, "the phase-1 head and the agreement reference resolved to one file"

free_gb = shutil.disk_usage(WORK).free/2**30
print(f"free space    : {free_gb:.1f} GB  (need ~4 GB: ~1.0 GB checkpoints + archives)")
assert free_gb > 8, "not enough free space"
os.environ["FM_HEAD"] = HEAD; os.environ["FM_REF"] = REF
print("\nSETUP COMPLETE")
```

---

## Cell 2 — dry-run both configs, diff them, and MEASURE the step time (~10 min)

Everything that can fail cheaply, before four-plus GPU-hours are committed.

```python
import os, json, subprocess, time
from pathlib import Path
WORK = Path("/kaggle/working"); SCRIPTS = WORK/"scripts"
DATA = Path(os.environ["LENS_DATA_ROOT"])
HEAD, REF = os.environ["FM_HEAD"], os.environ["FM_REF"]
REFCFG = WORK/"ref_nb0"/"run_config.json"; REFCFG.parent.mkdir(exist_ok=True)

# num_workers: sequential execution means one run has the whole session's CPU, so take all
# of it. §7.7's "2" was a CONCURRENCY compromise (two processes sharing ~4 vCPUs) and no
# longer applies. Capped at 4 because the augmentation is the only CPU work and oversubscribing
# a 4-vCPU box adds context switching, not throughput.
NW = max(1, min(4, (os.cpu_count() or 4)))
print(f"os.cpu_count() = {os.cpu_count()}  ->  --num-workers {NW}")

COMMON = (f'--data-root "{DATA}" --splits "{DATA}/splits/splits.json" '
          f'--cache-dir "{DATA}/feature_cache" --fp16 --num-workers {NW} '
          f'--init-head-from "{HEAD}" --agreement-ref "{REF}"')
os.environ["FM_NW"] = str(NW)

def run(cmd, tail=2500):
    r = subprocess.run(f'cd {SCRIPTS} && {cmd}', shell=True, capture_output=True, text=True)
    print((r.stdout or "")[-tail:] or (r.stderr or "")[-tail:])
    return r

# (a) DECLARE the control's config without training, so both runs can start together.
print("="*78); print("DECLARING THE §1 REFERENCE"); print("="*78)
r = run(f'python run_unfreeze_freematch.py --run nb0_control {COMMON} '
        f'--emit-config "{REFCFG}"', 800)
assert REFCFG.is_file(), "reference config was not emitted"
cfg = json.load(open(REFCFG))
for k, want in [("lr",1e-3), ("weight_decay",1e-4), ("local_threshold_mode","standard"),
                ("n_blocks",0), ("precision","fp16"), ("eval_precision","fp32"),
                ("max_steps",2000), ("schedule_total_steps",8000),
                ("curve_every",125), ("ckpt_every",500)]:
    assert cfg[k] == want, f"declared reference has {k}={cfg[k]!r}, expected {want!r}"
print(f"  head lr {cfg['lr']}  wd {cfg['weight_decay']}  mode {cfg['local_threshold_mode']}")
print(f"  control: {cfg['max_steps']} steps on the {cfg['schedule_total_steps']}-step schedule")

# (b) dry-run the unfrozen config against it -- the §1 table must show ONLY n_blocks,
#     grad_checkpointing and max_steps differing.
print("="*78); print("§1 PRECONDITION CHECK (dry-run)"); print("="*78)
r = run(f'python run_unfreeze_freematch.py --run nb3_unfrozen --grad-checkpointing '
        f'{COMMON} --reference-config "{REFCFG}" --dry-run', 4000)
assert "ALL PINNED SETTINGS MATCH" in r.stdout, "§1 FAILED -- do not launch"

# (c) ADVERSARIAL: the same command WITHOUT --fp16 must abort.
r = run(f'python run_unfreeze_freematch.py --run nb3_unfrozen --grad-checkpointing '
        f'--data-root "{DATA}" --splits "{DATA}/splits/splits.json" '
        f'--cache-dir "{DATA}/feature_cache" --num-workers 2 --init-head-from "{HEAD}" '
        f'--agreement-ref "{REF}" --reference-config "{REFCFG}" --dry-run', 600)
assert "ABORT" in (r.stdout + r.stderr), "the §1 check did NOT abort on a dropped --fp16"
print("  adversarial check ok: a dropped --fp16 aborts")

# (d) the pre-flight unit suite (BRIEF §7 items 1-3), on mock data
r = run('python test_unfreeze_freematch.py', 1200)
assert "ALL CHECKS PASSED" in r.stdout, "pre-flight suite FAILED -- do not launch"

# (e) MEASURE. A 200-step smoke of the REAL unfrozen config at the REAL batch 256: proves
#     no OOM, reports peak memory against the predicted 4.76 GB, and gives a step time to
#     project from. §7.7: time everything in ONE process and discard the first
#     configuration -- so read the SECOND curve interval, not the first.
#     NOTE: no --reference-config here. The smoke deliberately varies curve_every and
#     ckpt_every (both PINNED keys) so that 200 steps still produces curve points and a
#     retained R1, and §1 would -- correctly -- abort on that. §1 was already verified in
#     step (b) against the real settings; this run produces no number, only a timing.
print("="*78); print("200-STEP SMOKE + TIMING/UTILIZATION PROBE (GPU 0)"); print("="*78)

# Sample GPU utilization ALONGSIDE the smoke. This is the measurement that decides whether
# any future image-space run in this project can share a GPU: high sustained utilization =>
# compute-bound, concurrency would have been fine; low or spiky => the dataloader binds and
# sequential is correct. The 10th percentile is the informative one -- a mean of 60% could
# be a steady 60% or an alternation between 100% and stalls, and only the low tail
# distinguishes them.
UTIL = WORK/"gpu_util.csv"
sampler = subprocess.Popen(
    ['bash','-lc', f'nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory '
                   f'--format=csv,noheader,nounits -l 1 > "{UTIL}"'])
t0 = time.time()
try:
    r = run(f'CUDA_VISIBLE_DEVICES=0 python run_unfreeze_freematch.py --run nb3_unfrozen '
            f'--grad-checkpointing {COMMON} '
            f'--max-steps 200 --curve-every 100 --ckpt-every 100', 5000)
finally:
    sampler.terminate(); sampler.wait()
smoke_min = (time.time()-t0)/60
assert r.returncode == 0, "the 200-step smoke FAILED -- read the output above"
assert "VERDICT: PASS" in r.stdout, "the step-0 HARD GATE did not pass in the smoke"

import numpy as _np
rows = [l.split(",") for l in UTIL.read_text().splitlines() if l.strip()]
g0 = _np.array([float(c[1]) for c in rows if c[0].strip() == "0"])
m0 = _np.array([float(c[2]) for c in rows if c[0].strip() == "0"])
if g0.size:
    print(f"\n  GPU0 utilization over {g0.size} 1 s samples: "
          f"mean {g0.mean():.1f}%   p10 {_np.percentile(g0,10):.1f}%   "
          f"median {_np.median(g0):.1f}%   max {g0.max():.0f}%")
    print(f"  GPU0 memory-bus utilization:  mean {m0.mean():.1f}%   "
          f"p10 {_np.percentile(m0,10):.1f}%")
    print(f"  os.cpu_count() = {os.cpu_count()}   --num-workers = {NW}")
    if _np.percentile(g0,10) < 50:
        print("  READ: the 10th percentile is low -- the GPU is idling on the input "
              "pipeline for a meaningful fraction of the time. This run is (at least "
              "partly) DATALOADER-BOUND, and sharing the GPU with a second run would have "
              "made it worse. Sequential was the right call.")
    else:
        print("  READ: utilization stays high even at the 10th percentile -- this run is "
              "COMPUTE-BOUND and concurrency would have been viable. Worth recording for "
              "future image-space runs; it does not change today's plan.")
else:
    print("  (nvidia-smi produced no samples -- utilization unmeasured, not assumed)")

import numpy as np, glob as _g, json as _j
sd = WORK/"runs"/"freematch_fold0_nb3_unfrozen_SMOKE200"
dumps = sorted(_g.glob(str(sd/"logits"/"step_*.npz")))
assert dumps, "no logit dumps were written"
with np.load(dumps[-1]) as z:
    assert all(np.isfinite(z[k]).all() for k in z.files if k != "step"), "non-finite logits"
    print(f"  logit dump ok: {[(k, z[k].shape) for k in z.files if k!='step']}")
print(f"  checkpoints written: {len(_g.glob(str(sd/'checkpoints'/'step_*.pt')))}")
# measured INSIDE the training process and written to its config -- reading
# torch.cuda.max_memory_allocated() here would report this notebook's context, i.e. 0.
peak = _j.load(open(sd/"run_config.json")).get("peak_gpu_gb")
print(f"  peak GPU memory: {peak:.2f} GB   (§7.1a predicts 4.76 GB at true batch 256; "
      f"the T4 has 14.56 GiB)")
assert peak is None or peak < 12.0, "peak memory is close to the T4 ceiling -- stop"

# project the full run from the smoke, stated as an estimate, not a promise
import csv
diag = list(csv.DictReader(open(sd/"diagnostics.csv")))
print(f"\n  smoke wall clock: {smoke_min:.1f} min for 200 steps + 3 curve points")
print(f"  fp16 skipped-step rate in the LAST window: "
      f"{diag[-1].get('step_skipped_mean')}  (a startup transient of GradScaler's scale "
      f"search is normal; a persistently high rate is not)")
print("\n  Project 8000 steps + 64 curve points from the SECOND curve interval above.")
print("  If that exceeds ~8 h, relaunch Cell 3 with --curve-every 250 (32 points).")
print("\nPRE-FLIGHT COMPLETE -- safe to launch")
```

> If peak memory is anywhere near 14.5 GB, stop. Gradient checkpointing is mandatory and
> gradient accumulation is **not** a substitute — FreeMatch's threshold is an EMA over
> per-batch confidence statistics, so micro-batching changes the method (§7.1a).

---

## Cell 3a — the CONTROL, alone (~20 min), then the flatness decision

Sequential, control first: this is the cheap gate on whether the expensive run's result will
be attributable at all.

```python
import os, subprocess, time, glob, json
from pathlib import Path
WORK = Path("/kaggle/working"); SCRIPTS = WORK/"scripts"; RUNS = WORK/"runs"
DATA = Path(os.environ["LENS_DATA_ROOT"])
HEAD, REF = os.environ["FM_HEAD"], os.environ["FM_REF"]
NW = os.environ.get("FM_NW", "4")
REFCFG = WORK/"ref_nb0"/"run_config.json"
LOGS = WORK/"logs"; LOGS.mkdir(exist_ok=True)

COMMON = (f'--data-root "{DATA}" --splits "{DATA}/splits/splits.json" '
          f'--cache-dir "{DATA}/feature_cache" --fp16 --num-workers {NW} '
          f'--init-head-from "{HEAD}" --agreement-ref "{REF}" '
          f'--reference-config "{REFCFG}"')

BAD = ("Traceback","ABORT","REFUSING","out of memory","OOM","MISMATCH","retention failed",
       "VERDICT: FAIL")

def launch(cmd, log, run_dir, want_ckpt, label, poll=120):
    """Run one arm to completion, blocking. Kaggle may time the session out otherwise."""
    p = subprocess.Popen(["bash","-lc", f'cd {SCRIPTS} && {cmd} > {LOGS}/{log} 2>&1'])
    t0 = time.time(); seen = set()
    print(f"pid {p.pid}  {label}", flush=True)
    while True:
        d = p.poll()
        n = len(glob.glob(str(RUNS/run_dir/"checkpoints"/"step_*.pt")))
        c = RUNS/run_dir/"curve.csv"
        rows = sum(1 for _ in open(c)) - 1 if c.is_file() else 0
        print(f"  [{(time.time()-t0)/60:6.1f} min] ckpt {n:>2}/{want_ckpt}  curve {rows:>3}  "
              f"{'running' if d is None else f'DONE rc={d}'}", flush=True)
        lp = LOGS/log
        if lp.is_file():
            for l in lp.read_text(errors="replace").splitlines():
                if l in seen: continue
                if any(k in l for k in BAD):
                    seen.add(l); print(f"   !! {l.strip()[:170]}", flush=True)
                elif "WARN" in l and "GradScaler" in l:
                    seen.add(l); print(f"   ~~ {l.strip()[:170]}", flush=True)
        if d is not None:
            break
        time.sleep(poll)
    print(f"  {label} finished in {(time.time()-t0)/60:.1f} min, rc={p.poll()}\n", flush=True)
    return p.poll()

rc = launch(f'CUDA_VISIBLE_DEVICES=0 python run_unfreeze_freematch.py --run nb0_control '
            f'{COMMON}', "nb0_control.log", "freematch_fold0_nb0_control", 5,
            "nb0_control  (2000 steps on the 8000-step schedule, ~20 min)")
assert rc == 0, "the CONTROL failed -- see logs/nb0_control.log"
```

```python
# THE FLATNESS DECISION -- read this before committing 4-6 hours.
import csv, sys
CTL = RUNS/"freematch_fold0_nb0_control"
rows = {int(float(r["step"])): r for r in csv.DictReader(open(CTL/"curve.csv"))}
probe = [250, 500, 1000, 1500, 2000]
vals = {s: float(rows[s]["fold0_tpr_1e-3"]) for s in probe if s in rows}
step0 = float(rows[0]["fold0_tpr_1e-3"])
ONE_POS = 1/98

print(f"  step     0: {step0:.4f}   (the loaded phase-1 model)")
for s in probe:
    print(f"  step {s:>5}: {vals.get(s, float('nan')):.4f}")
spread = max(vals.values()) - min(vals.values())
drift  = max(abs(v - step0) for v in vals.values())
flat = spread <= ONE_POS and drift <= ONE_POS
print(f"\n  spread {spread:.4f}   max drift from step 0 {drift:.4f}   "
      f"one-positive resolution {ONE_POS:.4f}")
print("  VERDICT:", "CONTROLLED -- the extra steps move nothing on their own; the unfrozen "
      "result will be attributable to unfreezing." if flat else
      "NOT CONTROLLED -- the extra steps move the head by themselves, so the unfrozen "
      "result cannot be attributed to unfreezing alone.")
print("\n  NOTE (§7.10): flat here rules out drift over steps 0-2000 under the pinned "
      "8000-step schedule.\n  It does NOT rule out late-onset drift beginning after 2000. "
      "The argument for stopping at\n  2000 is that a converged 1,921-parameter head "
      "drifts early if at all (§7.4: FreeMatch\n  flattens ~2800) -- an argument, not a "
      "proof. If the unfrozen result turns on a\n  difference comparable to this spread, "
      "extend the control to 8000 (~1 frozen GPU-hour).")
if not flat:
    print("\n  >>> STOP AND DECIDE. Running the unfrozen arm now still produces a number, "
          "but not\n      an attributable one. Continuing is a legitimate choice IF you "
          "record that the\n      confound is live -- it is not a choice to make by "
          "default.")
```

---

## Cell 3b — the UNFROZEN run, alone (4–6 h)

```python
rc = launch(f'CUDA_VISIBLE_DEVICES=0 python run_unfreeze_freematch.py --run nb3_unfrozen '
            f'--grad-checkpointing {COMMON}', "nb3_unfrozen.log",
            "freematch_fold0_nb3_unfrozen", 17,
            "nb3_unfrozen  (8000 steps, use YOUR Cell-2 projection)", poll=300)
assert rc == 0, "the UNFROZEN run failed -- see logs/nb3_unfrozen.log"

# fp16 health, post-settle: the GradScaler's initial scale search costs a handful of skipped
# steps by design. A persistently high rate after it settles is a different thing, and it is
# indistinguishable from "unfreezing does not help" in the metric.
import csv
diag = list(csv.DictReader(open(RUNS/"freematch_fold0_nb3_unfrozen"/"diagnostics.csv")))
post = [float(d["step_skipped_mean"]) for d in diag[2:] if d.get("step_skipped_mean")]
if post:
    rate = sum(post)/len(post)
    print(f"\nfp16 skip rate after the scaler settles: {rate:.4f}  "
          f"(worst window {max(post):.3f})")
    print(f"scale trajectory (instantaneous, last-in-window): "
          f"{[d.get('amp_scale_last') for d in diag[:6]]}")
    if rate > 0.05:
        print("  >>> ABOVE 5% -- lost updates are material. Report this before treating "
              "the TPR as a measurement of unfreezing.")
```

**If the step-0 gate fails, the run aborts in the first two minutes.** That is the design.
Likely causes, in order: wrong `--init-head-from`; the feature standardizer not restored;
a wrong `local_threshold_mode`; an fp16 eval encode against an fp32 reference; venue drift
(§7.7: local Ada vs Kaggle T4 differ by 0.0204, twice the one-positive resolution).

---

## Cell 4 — the BRIEF §9 report, the gate, and the post-hoc mechanism figure

```python
import subprocess
from pathlib import Path
WORK = Path("/kaggle/working"); SCRIPTS = WORK/"scripts"; RUNS = WORK/"runs"
UNF = RUNS/"freematch_fold0_nb3_unfrozen"; CTL = RUNS/"freematch_fold0_nb0_control"

def run(cmd, tail=9000):
    r = subprocess.run(f'cd {SCRIPTS} && {cmd}', shell=True, capture_output=True, text=True)
    print((r.stdout or "")[-tail:] or (r.stderr or "")[-tail:]); return r

# 1-6, 8: the required outputs, including the EXPLICIT confound verdict
run(f'python report_unfreeze_freematch.py --unfrozen "{UNF}" --control "{CTL}"')

# 5: the collapse gate already RAN as part of each run and wrote gate_fold0.json, which the
# report above reads for the firing decision -- a required output does not depend on someone
# remembering a follow-up command. Re-run it here only if you want the full curve printout.
#   python collapse_gate.py --run "<dir>" --from-curve --tpr-col fold0_tpr_1e-3

# 5.6: pseudo-label precision -- POST-HOC ONLY. It must never inform a checkpoint,
# hyperparameter or method choice; the moment it does it is oracle selection.
for d in (UNF, CTL):
    run(f'python pseudolabel_precision.py --run "{d}" '
        f'--splits "{Path(__import__("os").environ["LENS_DATA_ROOT"])}/splits/splits.json"',
        4000)
```

---

## Cell 5 — package, verify, download

```python
import subprocess, json
from pathlib import Path
WORK = Path("/kaggle/working"); SCRIPTS = WORK/"scripts"
OUT = WORK/"fm_nb3_out"; REFCFG = WORK/"ref_nb0"/"run_config.json"

r = subprocess.run(
    f'cd {SCRIPTS} && python package_screen_results.py --runs-dir "{WORK}/runs" '
    f'--out "{OUT}" --logs "{WORK}/logs" --pattern "freematch_fold0_*" '
    f'--declared-reference "{REFCFG}" --nb0-run freematch_fold0_nb0_control',
    shell=True, capture_output=True, text=True)
print((r.stdout or "")[-7000:] or (r.stderr or "")[-4000:])
assert r.returncode == 0, "PACKAGING FAILED -- read the output before downloading"

man = json.load(open(OUT/"MANIFEST.json"))
print(f"\n{'part':<52}{'MB':>9}  sha256[:16]")
for p in man["parts"]:
    print(f"{p['name']:<52}{p['mb']:>9.1f}  {p['sha256'][:16]}")
print(f"\nreassembly verified : {man['reassembly_verified']}")
print(f"reference check     : {(man['reference_check'] or {}).get('ok')}")
for rn, v in man["retention"].items():
    print(f"retention {rn:<44} {v['n_found']}/{v['n_expected']}")
assert man["reassembly_verified"], "DO NOT DOWNLOAD -- reassembly verification failed"
assert all(v["ok"] for v in man["retention"].values()), "a run is missing checkpoints"
print("\nDownload every file in /kaggle/working/fm_nb3_out, MANIFEST.json included.")
```

Expected shape (nb3 checkpoint = 59.4 MB):

| part | contents | size |
|---|---|---|
| `screen_core.tar.gz` | curves, configs, results, score dumps, **diagnostics.csv, memorization.csv, all 64 logit dumps**, logs, and the control's 5 tiny checkpoints | ~15 MB |
| `ckpt_freematch_fold0_nb3_unfrozen_part01of03.tar` | steps 0–2500 | ≤475 MB |
| `…part02of03.tar` | steps 3000–5500 | ≤475 MB |
| `…part03of03.tar` | steps 6000–8000 | ≤475 MB |

Independent self-contained tars, never `tar | split`: a split part cannot be verified or
extracted until every part is present and concatenated, whereas these can be checked and
re-downloaded singly (§7.7).

---

## What to read first, when it lands

1. **The step-0 gate** in both `run_config.json`s — if either is not `passed: true`, nothing
   else in the run means anything.
2. **The confound verdict** (report section 6). Flat control ⇒ the unfrozen number is
   attributable to unfreezing over 0–2000. Not flat ⇒ it is not, and that is a finding about
   the design. Either way, §7.10's limitation stands: this rules out drift over 0–2000 under
   the pinned schedule, not over 2000–8000.
3. **The fp16 skipped-step rate** (report section 8), read from `amp_scale_last` — the
   instantaneous scale, not `amp_scale_mean`. A high post-settle rate means updates were
   discarded, and "lost updates" and "unfreezing does not help" look identical in the metric.
4. **The collapse gate's firing decision** (section 5, from `gate_fold0.json`). If it FIRED,
   that is the first positive case in four rounds and is itself a result — inspect it before
   reading the run's TPR as a measurement of unfreezing. Also check the control's
   inline-vs-post-hoc lock-step check: it is the only evidence the agreement column the
   unfrozen run relies on is correct.
5. **The memorization probe** (section 4) — read `target_minus_test_matched`, not the raw
   means.
6. Only then the TPR. And per PROJECT.md §9: read the numbers before writing a verdict
   sentence — "fails the significance bar" and "is worse" are different claims.
