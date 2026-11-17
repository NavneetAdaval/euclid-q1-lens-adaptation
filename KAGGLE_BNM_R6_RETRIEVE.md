# Retrieve the venue step-0 gradient norms — PATH 1

**Goal:** recover `bnm_r6_step0_gradnorms_t4fp16.json`, the T4/fp16 measurement the swept γ
grid came from. This is the *only* path that preserves true provenance: path 2 re-measures,
which means the hashed file is a different run from the one that produced the grid.

**No GPU needed. Do not start a training run.** If the file is found, download it and the
pre-registration can be hashed.

Try the cells in order. Stop at the first one that finds it.

---

## Cell 1 — is it still in the working directory?

If the notebook session that ran Cell B is *still alive* (not restarted, not a new session),
the file is simply there.

```python
import glob, json, os

hits = sorted(glob.glob("/kaggle/working/**/*gradnorm*", recursive=True))
print("in /kaggle/working:", hits or "NOTHING")

for h in hits:
    d = json.load(open(h))
    print("\n===", h)
    for k in ("device", "fp16", "gpu_name", "ce_grad_norm",
              "bnm_grad_norm_at_gamma_1", "ratio_bnm_over_ce_at_gamma_1",
              "fractions", "gamma_grid"):
        if k in d:
            print(f"  {k:32} {json.dumps(d[k])}")
```

**What confirms it is the right file**, and it must satisfy all four:

    device                            "cuda"
    fp16                              true
    gamma_grid                        [0.00787, 0.039351, 0.157406]
    ratio_bnm_over_ce_at_gamma_1      ~12.706

If `gamma_grid` is `[0.010045, 0.050225, 0.200901]` this is the **CPU** file — the wrong one.
It must not be hashed as the grid's provenance.

---

## Cell 2 — is it in a committed version's output?

Kaggle keeps `/kaggle/working` per **committed version**. A session that was restarted loses
the live directory, but a *committed* run's output survives and can be re-attached.

```python
import glob, json, os

# outputs of previous notebook versions arrive under /kaggle/input when attached
hits = sorted(glob.glob("/kaggle/input/**/*gradnorm*", recursive=True))
print("in /kaggle/input:", hits or "NOTHING")
for h in hits:
    try:
        d = json.load(open(h))
        print(f"  {os.path.basename(h)}: device={d.get('device')} "
              f"fp16={d.get('fp16')} grid={d.get('gamma_grid')}")
    except Exception as e:
        print("  (unreadable)", h, e)
```

If this finds nothing, open the notebook's **Version history** in the Kaggle UI, find the
version whose log contains `VENUE (T4/fp16) vs LOCAL (CPU/fp32)`, and use **Add Data → Your
Work → Notebook Output** to attach that version's output, then re-run this cell.

---

## Cell 3 — found it: send it home

```python
import glob, shutil, os

SRC = ""      # <-- paste the path Cell 1 or 2 printed
assert SRC and os.path.isfile(SRC), "set SRC to the file Cell 1 or 2 found"

dst = "/kaggle/working/bnm_r6_step0_gradnorms_t4fp16.json"
if os.path.abspath(SRC) != os.path.abspath(dst):
    shutil.copy2(SRC, dst)
print("ready to download:", dst, os.path.getsize(dst), "bytes")
print(open(dst).read())
```

Download it from the notebook's **Output** panel (or just copy the printed JSON), and save it
locally as:

    C:\Users\krris\ML\lens_paper\runs\bnm_r6_step0_gradnorms_t4fp16.json

Then:

```bash
python scripts/write_bnm_r6_prereg.py --final
```

That refuses today and will stop refusing once the file is in place. It prints the sha256 to
put in PROJECT.md.

---

## If all three cells come up empty

The session is gone and path 1 has failed. Say so and I will set up path 2: re-run
`--measure-step0` on T4/fp16 and record it as a **second, later measurement** — both numbers
written, the sweep's use of the first disclosed, and any material difference logged as a
**within-architecture variance** data point for §7.7. **The grid is not re-derived from the
new file**; γ = 0.00787 was selected and stays selected.
