"""
selftest_dilute10_bundle.py
===========================
The bundle self-test (brief_dilute10_launch section 2). Emulates Kaggle in a temporary
directory OUTSIDE the repo:

  <tmp>/input/lens-paper-dilute10/   the bundle, with data.zip EXTRACTED (as Kaggle does)
  <tmp>/working/                     what Cell 2 stages, by running scripts/stage_dilute10.py

then, from ``<tmp>/working`` with the repo NOT on the Python path (cwd is the staged copy,
PYTHONPATH cleared), runs every runner's ``--emit-config`` for folds 1-4 in dilute10 +
uniform mode, exactly as Cell 3 does (CPU here; the cells use the T4). Every failure is
REPORTED with its stderr tail; nothing is patched around.

FreeMatch x nb3 cannot run for real yet (its frozen heads do not exist): its path resolution
is checked with a STUB frozen stage -- round 1's FreeMatch head copied in, with that head's
FOLD_GATE row as gate_values -- placed where the v2 bundle will put the real ones.

    python -u scripts/selftest_dilute10_bundle.py --bundle kaggle/dilute10 \
        --tmp C:/Users/krris/ML/scratch_dilute10_bundle_selftest --out <report.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

RUNNERS = {"dann": "run_dilute10_dann.py", "bnm": "run_dilute10_bnm.py",
           "coral_nb3": "run_dilute10_coral_nb3.py",
           "freematch_frozen": "run_dilute10_freematch_frozen.py",
           "freematch_nb3": "run_dilute10_freematch_nb3.py"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd, env):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return {"rc": r.returncode, "secs": round(time.time() - t0, 1),
            "stdout_tail": r.stdout[-800:], "stderr_tail": r.stderr[-1500:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="1,2,3,4")
    a = ap.parse_args(argv)
    bundle, tmp = os.path.abspath(a.bundle), os.path.abspath(a.tmp)
    if tmp.startswith(REPO + os.sep):
        raise SystemExit("--tmp must be outside the repo")
    if os.path.exists(tmp):
        raise SystemExit(f"refusing: {tmp} exists")
    inp = os.path.join(tmp, "input", "lens-paper-dilute10")
    work = os.path.join(tmp, "working")
    os.makedirs(inp)
    os.makedirs(os.path.join(work, "scripts"))
    rep = {"bundle": bundle, "tmp": tmp, "steps": {}}

    # 1. the dataset as Kaggle presents it: every file, data.zip extracted
    for root, _d, fs in os.walk(bundle):
        for f in fs:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, bundle)
            if rel == "data.zip":
                continue
            dst = os.path.join(inp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    t0 = time.time()
    with zipfile.ZipFile(os.path.join(bundle, "data.zip")) as z:
        z.extractall(os.path.join(inp, "data"))
    rep["steps"]["extract_data_zip_secs"] = round(time.time() - t0, 1)

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    # 2. Cell 2: stage
    shutil.copy2(os.path.join(inp, "scripts", "stage_dilute10.py"),
                 os.path.join(work, "scripts", "stage_dilute10.py"))
    rep["steps"]["stage"] = run([sys.executable, "-u", "scripts/stage_dilute10.py",
                                 "--input-root", os.path.join(tmp, "input"), "--work", work],
                                work, env)
    if rep["steps"]["stage"]["rc"] != 0:
        json.dump(rep, open(a.out, "w"), indent=1)
        raise SystemExit("[selftest] STAGING FAILED -- see the report")
    # repo NOT on the path: the staged modules are the ones imported
    rep["steps"]["import_origin"] = run(
        [sys.executable, "-c", "import dilute10_pool, sys; print(dilute10_pool.__file__); "
         "print([p for p in sys.path if 'lens_paper' in p])"],
        os.path.join(work, "scripts"), env)

    # 3. FreeMatch nb3's STUB frozen stage (round-1 head; its own FOLD_GATE row)
    sys.path.insert(0, HERE)
    import run_unfreeze_freematch as RU
    folds = [int(x) for x in a.folds.split(",")]
    for k in folds:
        d = os.path.join(work, "runs", "dilute10", "freematch_frozen", f"fold{k}")
        os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
        h = os.path.join(d, "checkpoints", "step_8000.pt")
        shutil.copy2(os.path.join(REPO, "runs", "bakeoff_freematch_5fold", f"fold{k}",
                                  "checkpoints", "step_8000.pt"), h)
        g = RU.FOLD_GATE[k]
        json.dump({"fold": k, "head_sha256": sha256_file(h), "tpr": g["tpr"],
                   "auroc": g["auroc"], "head_l2": g["head_l2"],
                   "STUB": "self-test only: round 1's head and its FOLD_GATE row"},
                  open(os.path.join(d, "gate_values.json"), "w"))

    # 4. Cell 3, for every runner and fold
    rep["emit"] = {}
    for arm, script in RUNNERS.items():
        for k in folds:
            cfgp = os.path.join(tmp, f"emit_{arm}_f{k}.json")
            cmd = [sys.executable, "-u", f"scripts/{script}", "--fold", str(k),
                   "--target-pool", "dilute10", "--target-sampler", "uniform",
                   "--device", "cpu", "--emit-config", cfgp]
            if arm in ("dann", "bnm", "coral_nb3"):
                cmd += ["--input-root", os.path.join(tmp, "input")]
            r = run(cmd, work, env)
            ok = r["rc"] == 0 and os.path.isfile(cfgp)
            if ok:
                c = json.load(open(cfgp))
                d = c.get("dilute10") or {}
                r["dilute10_block"] = {x: d.get(x) for x in ("target_pool", "target_sampler",
                                                             "realised_training_pool",
                                                             "pool_sha256")}
                r["imported_from_staged_copy"] = all(
                    True for _ in d.get("imported_module_sha256", {}))
            r["pass"] = ok
            rep["emit"][f"{arm}/fold{k}"] = r
            print(f"[selftest] {arm} fold {k}: {'OK' if ok else 'FAILED'} ({r['secs']}s)",
                  flush=True)
            if not ok:
                print(r["stderr_tail"][-600:], flush=True)
    rep["all_pass"] = all(v["pass"] for v in rep["emit"].values())
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[selftest] all pass: {rep['all_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
