# Re-score on Kaggle — build the TPR curves for both arms

**What this is.** Arms 1 and 2 both ran clean, and both came home missing every
per-checkpoint logit file. Those logits are the only way a TPR@1e-3 curve is ever
produced — the runs deliberately **score** checkpoints without **measuring** them, so no
curve appears as a side effect of training. This job loads all 64 checkpoints, re-scores
them, and writes one `curve.csv` per arm.

**Why it is the critical path.** Arm 2 showed DANN destroys the morphology probe. It did
not show what happens to **lens-finding**, and two opposite sentences are still live:

- DANN destroys morphology **and the task** → adversarial alignment is harmful here.
- DANN destroys morphology **and the task is fine** → morphology loss is the price of
  alignment and costs nothing that matters.

Nothing currently in hand tells them apart. This does.

**Both arms, not just arm 2.** Arm 1 is the λ=0 control and its curve is the
unfreezing-only reference. Without it, a TPR difference cannot be blamed on the adversary
rather than on unfreezing three encoder blocks — which is the mistake arm 1 was built to
prevent, and which the round has already made once.

**Wall time: ~75 min per arm, ~150 min for both**, plus ~10 min of setup and extraction.
That is measured, not guessed: both runs logged their own scoring cost at 111 s (arm 1)
and 123 s (arm 2) per checkpoint, stable to ±2 s across all 64.

**No training happens.** Nothing is optimised, no weights change.

---

## Before you start

**One dataset.** Sync `C:\Users\krris\ML\lens_paper\kaggle\dann_r5\` — now **v8**, **8.3 GB**. It carries the data, the
scripts, the splits, **both arms' 64 checkpoints**, and the small run files the step-0
gate needs. Nothing else has to be uploaded, extracted by hand, or attached alongside.

New notebook → **Accelerator: GPU T4 ×2** → **Add Data** → that dataset.

> **Kaggle auto-extracts archives on upload, and that is fine.** The checkpoints ship as
> ten `.tar` files; Kaggle unpacks them and keeps the 64 `step_N.pt` files instead. This
> is the same thing that has always turned `data.zip` into `<mount>/data/...`. Cell 2
> therefore looks for the checkpoints, not for the tars, and handles every layout Kaggle
> might have used. **Do not re-upload because the tars are gone** — they are supposed to
> be.

> **Use a T4.** Both arms were scored on one. §7.7 records that results are not portable
> across GPU architectures at this precision, and the step-0 gate compares against each
> arm's own stored logits element by element.

---

## Cell 1 — environment check (~10 s)

```python
import shutil, os
import torch

print("torch", torch.__version__, "| CUDA", torch.version.cuda,
      "| available", torch.cuda.is_available())
assert torch.cuda.is_available(), "NO GPU. Settings -> Accelerator -> GPU T4 x2."

name = torch.cuda.get_device_name(0)
p = torch.cuda.get_device_properties(0)
print(f"  cuda:0  {p.name}  {p.total_memory/2**30:.2f} GiB")
if "T4" not in name:
    print("\n" + "!" * 78)
    print(f"!!! NOT A T4 -- this is a {name}.")
    print("!!! Both arms were SCORED on a T4 and the step-0 gate below compares against")
    print("!!! their own stored logits element by element. On another architecture that")
    print("!!! check may fail for the GPU rather than for anything real. STOP.")
    print("!" * 78 + "\n")
else:
    print("\n  T4 confirmed -- the architecture both arms were scored on.")

free = shutil.disk_usage("/kaggle/working").free / 2**30
print(f"  /kaggle/working free: {free:.1f} GiB")
assert free > 12, f"only {free:.1f} GiB free; extracting both arms needs ~5 GB"
```

---

## Cell 2 — mount, stage, verify (~2 min)

**One dataset.** Checkpoints, data, scripts and both arms' run files are all in it, listed
in one MANIFEST. Nothing else needs attaching.

> **Kaggle auto-extracts archives when you upload a dataset.** That is not a problem and
> nothing needs re-uploading because of it — it is how `data.zip` has always become
> `<mount>/data/...` in every round. It does the same to the ten checkpoint `.tar` files:
> the tars do not survive, **their contents do**. So this cell looks for the 64
> `step_N.pt` files, not for the tars, and the manifest carries a sha256 for every one of
> them individually.

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
assert hits, ("no dann_r5 bundle under /kaggle/input. Add Data -> the dataset built "
              "from kaggle/dann_r5/. Present instead: "
              + str(sorted(os.listdir('/kaggle/input'))))
MAN_P, man = hits[0]
INPUT = os.path.dirname(MAN_P)
print(f"mount: {INPUT}   v{man['version']}  built {man['built_utc']}")
assert int(man.get("version", 0)) >= 8, (
    f"this is bundle v{man.get('version')}. The re-score needs v8 or later -- that is "
    f"the version whose manifest hashes each checkpoint individually. Re-add the dataset.")

parts = man["rescore_checkpoints"]
cfiles = [r for r in parts if r["kind"] == "checkpoint_file"]
metas = [r for r in parts if r["kind"] == "run_meta"]
assert len(cfiles) == 64 and len(metas) == 8, (
    f"manifest lists {len(cfiles)} checkpoints and {len(metas)} run files; expected "
    f"64 and 8. This bundle was not built for the re-score.")

WORK = "/kaggle/working"
os.makedirs(f"{WORK}/scripts", exist_ok=True)
for f in man["scripts"]:
    shutil.copy2(os.path.join(INPUT, "scripts", f), f"{WORK}/scripts/{f}")
print(f"  copied {len(man['scripts'])} scripts (data stays in /kaggle/input)")
assert os.path.isfile(f"{WORK}/scripts/rescore_dann_r5.py")

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

# ---- LOCATE THE 64 CHECKPOINTS, however Kaggle chose to store them ----------
# Three layouts are all fine and all handled: the tars survived; Kaggle extracted each
# tar in place; Kaggle extracted each tar into a folder named after it. Rather than
# guess, index every step_*.pt on the mount and match by SIZE + the run subdirectory in
# its path. The two arms cannot be confused: "dann_r5_nb3_lambda0_fold0" does not
# contain "dann_r5_nb3_fold0" as a path component.
RUN_SUB = {"dann_nb3_lambda0": "dann_r5_nb3_lambda0_fold0",
           "dann_nb3": "dann_r5_nb3_fold0"}
found = {}
for q in glob.glob(f"{INPUT}/**/step_*.pt", recursive=True):
    comps = q.replace("\\", "/").split("/")
    for arm, sub in RUN_SUB.items():
        if sub in comps:
            found[(arm, os.path.basename(q))] = q
            break

tars_present = [r for r in parts
                if r["kind"] == "checkpoint_tar"
                and os.path.isfile(os.path.join(INPUT, r["path"]))]
missing = [r for r in cfiles if (r["arm"], r["basename"]) not in found]

print(f"  loose checkpoints found on the mount : {len(found)} of 64")
print(f"  checkpoint tars still present        : {len(tars_present)} of 10 "
      f"(0 is normal -- Kaggle extracts them)")

if missing and not tars_present:
    raise SystemExit(
        f"CANNOT FIND {len(missing)} OF 64 CHECKPOINTS ON THIS MOUNT.\n"
        f"  missing e.g.: {[(r['arm'], r['basename']) for r in missing[:5]]}\n"
        f"  step_*.pt files seen anywhere on the mount: {len(found)}\n"
        f"  Run the DIAGNOSTIC cell below and send its output -- do not re-upload yet.")

for r in cfiles:
    q = found.get((r["arm"], r["basename"]))
    if q and os.path.getsize(q) != r["bytes"]:
        raise SystemExit(f"{r['arm']} {r['basename']}: {os.path.getsize(q)} bytes, "
                         f"manifest says {r['bytes']} -- this file is truncated")

bad_m = []
for r in metas:
    q = os.path.join(INPUT, r["path"])
    if not os.path.isfile(q):
        bad_m.append(f"{r['path']}: MISSING")
    elif sha(q) != r["sha256"]:
        bad_m.append(f"{r['path']}: CHECKSUM MISMATCH")
assert not bad_m, ("the small run files did not arrive intact:\n  "
                   + "\n  ".join(bad_m))
print(f"  all {len(metas)} run files verified by sha256 (gate reference included)")

pre = json.load(open(os.path.join(INPUT, "runs/dann_r5_prereg.json")))
got = pre.pop("sha256_of_body_without_this_field")
again = hashlib.sha256(json.dumps(pre, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert got == again, f"PRE-REGISTRATION DIGEST FAILED: {got} != {again}"
print(f"  pre-registration self-certifies: {got}")

sys.path.insert(0, f"{WORK}/scripts")
os.environ["INPUT_ROOT"] = INPUT
json.dump({f"{k[0]}|{k[1]}": v for k, v in found.items()},
          open("/kaggle/working/found_checkpoints.json", "w"))
print("\nOK -- mount verified.")
```

### DIAGNOSTIC — only if Cell 2 says it cannot find the checkpoints

Prints what is actually on the mount. Send the output; do not re-upload first.

```python
import os, glob, json, collections
INPUT = os.path.dirname([p for p in glob.glob("/kaggle/input/**/MANIFEST.json",
                                              recursive=True)
                         if json.load(open(p)).get("bundle") == "dann_r5"][0])
print("mount:", INPUT, "\n")
print("top level:", sorted(os.listdir(INPUT)), "\n")
for d in ("checkpoints_rescore",):
    q = os.path.join(INPUT, d)
    print(f"{d}/ exists: {os.path.isdir(q)}")
    if os.path.isdir(q):
        for e in sorted(os.listdir(q))[:20]:
            f = os.path.join(q, e)
            print(f"   {'DIR ' if os.path.isdir(f) else 'FILE'}  {e}"
                  + ("" if os.path.isdir(f) else f"  {os.path.getsize(f)/1e6:.1f} MB"))
print("\nevery *.pt on the mount, by directory:")
c = collections.Counter(os.path.dirname(p)[len(INPUT):]
                        for p in glob.glob(f"{INPUT}/**/*.pt", recursive=True))
for k, v in sorted(c.items()):
    print(f"  {v:4d}  {k}")
print("\nany *.tar:", len(glob.glob(f"{INPUT}/**/*.tar", recursive=True)))
```

---

## Cell 3 — place the checkpoints and verify each one (~6 min)

Puts both runs where the re-score expects them and **verifies every checkpoint by its own
sha256** — 64 files, individually, however they arrived. Refuses to continue on a partial
set: a curve rebuilt from 30 of 32 checkpoints is not a partial curve, it is a wrong one.

Nothing here is manual.

```python
import os, json, tarfile, hashlib, glob, shutil
INPUT = os.environ["INPUT_ROOT"]
man = json.load(open(os.path.join(INPUT, "MANIFEST.json")))
parts = man["rescore_checkpoints"]
found = {tuple(k.split("|", 1)): v
         for k, v in json.load(open("/kaggle/working/found_checkpoints.json")).items()}

ARM_DIR = {"dann_nb3_lambda0": "/kaggle/working/runs/dann_arm1/extracted",
           "dann_nb3":         "/kaggle/working/runs/dann_arm2_fold0/extracted"}
RUN_SUB = {"dann_nb3_lambda0": "dann_r5_nb3_lambda0_fold0",
           "dann_nb3": "dann_r5_nb3_fold0"}
for d in ARM_DIR.values():
    os.makedirs(d, exist_ok=True)

def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(n), b""):
            h.update(b)
    return h.hexdigest()

# if any tar did survive, extract it -- both layouts end in the same place
for r in parts:
    if r["kind"] != "checkpoint_tar":
        continue
    src = os.path.join(INPUT, r["path"])
    if os.path.isfile(src):
        with tarfile.open(src) as tf:
            tf.extractall(ARM_DIR[r["arm"]])
        print(f"  extracted  {os.path.basename(r['path'])}")

# the small run files
for r in parts:
    if r["kind"] != "run_meta":
        continue
    dst = os.path.join(ARM_DIR[r["arm"]], r["extract_to"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(INPUT, r["path"]), dst)
print(f"  placed     {len([r for r in parts if r['kind'] == 'run_meta'])} run files")

# every checkpoint, verified individually
bad, placed = [], 0
for r in parts:
    if r["kind"] != "checkpoint_file":
        continue
    dst = os.path.join(ARM_DIR[r["arm"]], RUN_SUB[r["arm"]], "checkpoints", r["basename"])
    if not os.path.isfile(dst):
        src = found.get((r["arm"], r["basename"]))
        if not src:
            bad.append(f"{r['arm']}/{r['basename']}: NOT FOUND"); continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        placed += 1
    got = sha(dst)
    if got != r["sha256"]:
        bad.append(f"{r['arm']}/{r['basename']}: sha256 {got[:16]} != {r['sha256'][:16]}")
assert not bad, ("CHECKPOINTS DID NOT ARRIVE INTACT:\n  " + "\n  ".join(bad[:10])
                 + f"\n  ({len(bad)} of 64 bad)")
print(f"  verified   64 of 64 checkpoints by sha256 ({placed} copied into place)")

for arm, sub in RUN_SUB.items():
    d = os.path.join(ARM_DIR[arm], sub)
    n = len(glob.glob(os.path.join(d, "checkpoints", "step_*.pt")))
    exp = json.load(open(os.path.join(d, "run_config.json")))["n_checkpoints_expected"]
    assert n == exp, (f"{arm}: {n} checkpoints, run recorded {exp}. Do not re-score a "
                      f"partial set -- the curve would look complete and be wrong.")
    for need in ("run_config.json", "checkpoints.csv",
                 "logits/ids.npz", "logits/true_init.npz"):
        assert os.path.isfile(os.path.join(d, need)), f"{arm}: {need} did not land"
    print(f"  {arm:<18} {n} of {exp} checkpoints, run files present   {d}")

print("\nBoth arms complete and verified. Cell 4 next.")
```

---

## Cell 4 — THE STEP-0 GATE, both arms (~10 min). **Stop here if it fails.**

Builds each arm through the runner's own `build_arm`, scores the fresh model before any
checkpoint is loaded, and checks it **two ways**:

- **element by element** against that run's own `true_init.npz` — 26,497 stored logits,
  98 positives and 26,399 negatives. This is the strong check. Two different logit vectors
  can share a TPR; at 98 positives the metric moves in steps of 0.0102 and collides
  easily. Matching 26,497 numbers is not a coincidence that happens.
- against the **fold-0 gate** the run itself made and the pre-registration names.

If this fails, the re-score path differs from the run path and every one of the other 63
numbers is suspect. §7.1a: scoring a real nb3 checkpoint from the stale cache once gave
TPR 0.194 against a logged 0.020 — a different model, not a rounding difference.

```python
import subprocess, sys, os
INPUT = os.environ["INPUT_ROOT"]

r = subprocess.run([sys.executable, "-u", "scripts/rescore_dann_r5.py",
                    "--gate-only", "--fold", "0", "--input-root", INPUT],
                   cwd="/kaggle/working", capture_output=True, text=True)
print(r.stdout[-8000:])
if r.returncode != 0:
    print(r.stderr[-4000:])
    raise SystemExit("STEP-0 GATE FAILED -- do not run Cell 5. Send the output above.")
print("\nBOTH GATES PASS -- the re-score path is the run path. Cell 5 is safe to start.")
```

---

## Cell 5 — the re-score (~150 min for both arms)

Loads all 64 checkpoints and writes their logits, then computes the curves.

**The lockout boundary, so it is on the record:** every logit file is produced with
`test_label_lockout()` active, so no scoring step can consult a metric — enforced, not
asserted. TPR and AUROC are computed **afterwards**, outside the lockout, from the files
on disk. Logits inside, numbers outside.

```python
import subprocess, sys, os, time, threading
INPUT = os.environ["INPUT_ROOT"]
LOG = "/kaggle/working/rescore.log"

print("expected: ~75 min per arm (arm 1 logged 111 s/checkpoint, arm 2 123 s),")
print("          ~150 min for both, plus setup. Kaggle session cap is 12 h.\n")

t0 = time.time(); stop = threading.Event()
def hb():
    while not stop.wait(300):
        print(f"    ... {(time.time()-t0)/60:.0f} min elapsed", flush=True)
threading.Thread(target=hb, daemon=True).start()

with open(LOG, "w") as lf:
    p = subprocess.Popen([sys.executable, "-u", "scripts/rescore_dann_r5.py",
                          "--fold", "0", "--input-root", INPUT],
                         cwd="/kaggle/working", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        lf.write(line); lf.flush(); print(line, end="", flush=True)
    rc = p.wait()
stop.set()
print(f"\nexit code {rc} after {(time.time()-t0)/60:.1f} min")
if rc != 0:
    print("RE-SCORE FAILED -- see the failure notes at the bottom.")
```

*There is no resume cell. If the session dies, re-run this one: it starts over, which
costs 150 minutes and nothing else. Nothing is trained, so nothing is lost.*

---

## Cell 6 — the numbers to send back (~10 s)

**This is the deliverable.** Copy the whole printed block into your reply.

```python
import csv, os, json
import numpy as np

A1 = "/kaggle/working/runs/dann_arm1/extracted/dann_r5_nb3_lambda0_fold0/curve.csv"
A2 = "/kaggle/working/runs/dann_arm2_fold0/extracted/dann_r5_nb3_fold0/curve.csv"

def load(p):
    rows = list(csv.DictReader(open(p, newline="")))
    return ([int(r["step"]) for r in rows],
            [float(r["tpr_at_1e-3"]) for r in rows],
            [float(r["auroc"]) for r in rows])

s1, t1, u1 = load(A1)
s2, t2, u2 = load(A2)
m1t, m1u = dict(zip(s1, t1)), dict(zip(s1, u1))

print("=" * 84)
print("TPR@1e-3 AND AUROC -- arm 2 (lambda live) vs arm 1 (lambda = 0), MATCHED STEPS")
print("=" * 84)
print(f"{'step':>6} {'TPR arm2':>10} {'TPR arm1':>10} {'dTPR':>9} "
      f"{'AUR arm2':>10} {'AUR arm1':>10} {'dAUR':>9}")
dt, du = [], []
for st, a, b in zip(s2, t2, u2):
    if st not in m1t:
        print(f"{st:6d}  no arm-1 row at this step -- comparison REFUSED"); continue
    dt.append(a - m1t[st]); du.append(b - m1u[st])
    print(f"{st:6d} {a:10.6f} {m1t[st]:10.6f} {a-m1t[st]:+9.6f} "
          f"{b:10.6f} {m1u[st]:10.6f} {b-m1u[st]:+9.6f}")

print("-" * 84)
print(f"  arm 1: TPR min {min(t1):.6f}  max {max(t1):.6f} (step {s1[int(np.argmax(t1))]})"
      f"  last {t1[-1]:.6f}")
print(f"  arm 2: TPR min {min(t2):.6f}  max {max(t2):.6f} (step {s2[int(np.argmax(t2))]})"
      f"  last {t2[-1]:.6f}")
print(f"  paired dTPR : median {np.median(dt):+.6f}  mean {np.mean(dt):+.6f}  "
      f"min {min(dt):+.6f}  max {max(dt):+.6f}")
print(f"  paired dAUROC: median {np.median(du):+.6f}  min {min(du):+.6f}  "
      f"max {max(du):+.6f}")
n_dn = sum(1 for d in dt if d < 0)
print(f"  arm 2 BELOW arm 1 at {n_dn} of {len(dt)} matched steps "
      f"({100*n_dn/len(dt):.0f}%)")

print("\n" + "=" * 84)
print("AGAINST THE PRE-REGISTERED PREDICTION")
print("=" * 84)
print("  prediction_arc_suppression (hashed 2026-08-30, BEFORE any curve existed):")
print("    'TPR@1e-3 FALLS on arm 2 relative to arm 1 at matched steps.'")
print(f"  -> median paired dTPR = {np.median(dt):+.6f}, arm 2 lower at "
      f"{n_dn}/{len(dt)} steps.")
print("  READ IT AS A DIRECTION, NOT A TEST. This is ONE FOLD; PROJECT.md 7.4's")
print("  five-fold rule is what decides whether DANN did anything, and 98 test")
print("  positives means TPR moves in steps of 0.0102.")

rs = json.load(open("/kaggle/working/runs/dann_r5_rescore.json"))
print("\n" + "=" * 84)
print("RE-SCORE INTEGRITY")
print("=" * 84)
for a in rs["arms"]:
    g = a["step0_gate"]
    print(f"  {a['arm']:<18} {a['n_checkpoints']} checkpoints, "
          f"{a['phase_a_minutes']:.1f} min")
    print(f"    step-0 gate {g['verdict']}: {g['n_logits_compared']} logits compared "
          f"element-wise, max|diff| "
          f"{max(g['max_abs_diff_pos'], g['max_abs_diff_neg']):.3e}")
    print(f"    fold gate   TPR {g['tpr']:.6f} (|diff| {g['tpr_abs_diff']:.6f})  "
          f"AUROC {g['auroc']:.6f} (|diff| {g['auroc_abs_diff']:.2e})")
print(f"  total {rs['total_minutes']:.1f} min, encoder_eval = {rs['encoder_eval']}")
print("\nSEND THE WHOLE OUTPUT OF THIS CELL BACK.")
```

---

## Cell 7 — package for download (~3 min)

The curves and logits are small; the checkpoints are already home and are not re-packaged.

```python
import tarfile, os, hashlib
DST = "/kaggle/working/dann_r5_rescore_out"
os.makedirs(DST, exist_ok=True)
name = os.path.join(DST, "rescore_curves.tar.gz")

roots = {
    "dann_r5_nb3_lambda0_fold0":
        "/kaggle/working/runs/dann_arm1/extracted/dann_r5_nb3_lambda0_fold0",
    "dann_r5_nb3_fold0":
        "/kaggle/working/runs/dann_arm2_fold0/extracted/dann_r5_nb3_fold0",
}
n = 0
with tarfile.open(name, "w:gz") as tf:
    for arc, root in roots.items():
        for rel in ("curve.csv",):
            p = os.path.join(root, rel)
            assert os.path.isfile(p), f"MISSING {p}"
            tf.add(p, arcname=f"{arc}/{rel}"); n += 1
        lg = os.path.join(root, "logits")
        for dirpath, _d, files in os.walk(lg):          # RECURSIVE -- the fix
            for f in sorted(files):
                fp = os.path.join(dirpath, f)
                tf.add(fp, arcname=f"{arc}/logits/{os.path.relpath(fp, lg)}")
                n += 1
    for extra in ("/kaggle/working/runs/dann_r5_rescore.json",
                  "/kaggle/working/rescore.log"):
        if os.path.isfile(extra):
            tf.add(extra, arcname=f"meta/{os.path.basename(extra)}"); n += 1

h = hashlib.sha256()
with open(name, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 20), b""):
        h.update(b)
print(f"  {name}  {os.path.getsize(name)/1e6:.1f} MB  {n} files")
print(f"  sha256 {h.hexdigest()}")

# the collection defect that caused all this, checked on the way out
import csv as _csv
for arc, root in roots.items():
    want = {r["logit_file"] for r in _csv.DictReader(
        open(os.path.join(root, "checkpoints.csv"), newline="")) if r.get("logit_file")}
    with tarfile.open(name) as tf:
        have = {m[len(arc) + 1:] for m in tf.getnames() if m.startswith(arc + "/")}
    missing = sorted(want - have)
    print(f"  {arc}: {len(want)} logit files named, {len(missing)} missing from the "
          f"archive{' -- ' + str(missing[:3]) if missing else ''}")
    assert not missing, "THE COLLECTION DEFECT AGAIN -- do not download this, report it"
print("\n  Download it from the Output panel.")
```

---

## If something goes wrong

| what you see | what to do |
|---|---|
| **Cell 2: no dann_r5 bundle found** | The dataset is not attached. The cell prints what *is* under `/kaggle/input` — send that. |
| **Cell 2: "this is bundle v7"** (or older) | Stale mount. v8 is the version with the fixed re-score script. Detach and re-add. |
| **Cell 2: "checkpoint tars still present: 0 of 10"** | **Normal, not an error.** Kaggle extracted them on upload. What matters is the line above it: loose checkpoints found, 64 of 64. |
| **Cell 2: CANNOT FIND N OF 64 CHECKPOINTS** | Run the DIAGNOSTIC cell and send its output **before re-uploading**. It prints every `.pt` on the mount by directory, which says immediately whether the files are absent or just somewhere unexpected. |
| **Cell 3: CHECKPOINTS DID NOT ARRIVE INTACT** | A real corruption: each file is checked against its own sha256, and the message names which failed. |
| **Cell 3: sha256 mismatch on a part** | That part uploaded corrupt. Re-upload **that part only** — the tars are independent by design. |
| **Cell 3: fewer than 32 checkpoints** | A part did not extract. Do not continue: a curve from a partial set looks complete and is wrong. |
| **Cell 3: a run file did not land** | It ships in the same dataset, so this means the upload is incomplete rather than that something must be fetched. Re-run Cell 2 — it will say which parts are bad. Do **not** skip the gate to get moving. |
| **Cell 4: STEP-0 GATE FAILED** | **Stop.** The re-score path differs from the run path. Send the printed element-wise max\|diff\| and the fold-gate numbers. **Do not widen the tolerance** — §7.7's rule is to fix by re-scoring on the venue, never by widening the gate. |
| **Cell 4: element-wise passes, fold gate fails** | Would mean the stored logits themselves disagree with the gate table — a different problem from a bad re-score path. Send both numbers. |
| **CUDA out of memory** | Scoring is `no_grad` and batched; both arms peaked at 3.3 GB while *training*. An OOM here means something else is on the GPU. Restart the session; do not change the batch size. |
| **Wall time far outside ~150 min** | Much faster usually means fewer checkpoints ran than you think — Cell 6 prints the count per arm, check it against 32. |
| **Cell 7: "THE COLLECTION DEFECT AGAIN"** | Report it. That assertion exists because this exact failure already happened once silently. |

---

## What happens after this

1. You send back Cell 6's block.
2. The budget gets declared from arm 2's curve per §7.14 — and **expect "declared", not
   "derived"**: §7.13 and §7.14 both establish that these curves do not locate a budget,
   and §7.11 set the precedent for declining the argmax on the winner's-curse argument.
   No budget will be proposed off a first look at the shape.
3. The three advisory margins and `k` get declared and hashed.
4. **Arm 3** — folds 1–4. Conditions 3 and 4 are now **advisory on every arm**: they
   compute, they log, they record when they *would* have fired, and they never stop a run.
   Conditions 1 and 2 stay armed. So no confirmatory fold will abort at step 451 and throw
   away the TPR it exists to produce.
