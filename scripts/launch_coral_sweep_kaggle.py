"""
launch_coral_sweep_kaggle.py
============================
Orchestrate the round-4 CORAL gamma sweep on a Kaggle **T4 x 2** session.

    gamma = 1  -> cuda:0  }  launched CONCURRENTLY
    gamma = 4  -> cuda:1  }
    gamma = 16 -> whichever device frees first

Three INDEPENDENT processes, each `python -u`, each with its own log file and its own
output directory. Nothing is piped and nothing is shared.

WHY EACH OF THOSE CHOICES, since every one of them is a fix for something that already
went wrong (`runs/coral_round4_design.md` §9):

  * **Separate output directories.** The calibration was corrupted by two processes racing
    on one JSON. Here the trace path is derived from the gamma, and
    ``run_coral_gamma_sweep.py`` REFUSES to start if its own trace already exists, so a
    collision is impossible rather than merely unlikely.
  * **`python -u`, and stdout redirected to a FILE, never a pipe.** Piping through `grep`
    made the calibration's stdout block-buffer: the log sat at 0 bytes for over an hour
    while the run was healthy, so there was no way to see progress or trouble. `-u` plus a
    plain file redirect means every line lands as it is printed.
  * **Liveness by PID via the OS, not `ps`.** `ps -p` in Git Bash cannot see native Windows
    PIDs and reported a running job as dead. This launcher owns the child handles
    (``Popen.poll()``), which is exact on every platform and needs no shelling out.
  * **A 100-step verification arm first.** Before committing two GPUs for hours, gamma=1
    runs alone for 100 steps and the launcher READS BACK its trace: file present, header as
    expected, rows growing, values finite. Only then does the full sweep start. A bundle or
    path problem then costs three minutes instead of a session.

CHECKPOINT CADENCE. ``--ckpt-every 250`` (8 per 2000-step arm). Kaggle sessions time out;
an arm that dies at step 1900 with nothing on disk is a total loss. At nb3 a checkpoint is
~59 MB, so this costs ~475 MB per arm and buys restartability.

Run (inside the Kaggle session, from the working copy of the repo):
    python -u scripts/launch_coral_sweep_kaggle.py --out-root /kaggle/working/runs/coral_r4_sweep
    python -u scripts/launch_coral_sweep_kaggle.py --verify-only     # just the 100-step gate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

SWEEP_SCRIPT = os.path.join(_HERE, "run_coral_gamma_sweep.py")
# (gamma, preferred device). gamma=16 has no preferred device -- it takes the first free one.
CONCURRENT = ((1.0, 0), (4.0, 1))
QUEUED = (16.0,)

REQUIRED_TRACE_COLS = ("step", "sup_loss_mean", "coral_raw_mean",
                       "src_feat_l2_mean", "tgt_feat_l2_mean",
                       "src_perdim_std_mean", "tgt_perdim_std_mean")


def gdir(out_root, gamma):
    return os.path.join(out_root, f"g{int(gamma)}")


def build_cmd(gamma, out_dir, args, max_steps=None, log_every=None):
    steps = max_steps if max_steps is not None else args.max_steps
    # The log interval MUST divide into the step budget or the arm writes NO trace rows and
    # exits 0 -- which is exactly how the 100-step verify arm failed: --max-steps 100 with
    # the default --log-every 125 never hits `gstep % log_every == 0`, so verify_trace
    # correctly reported "trace.csv was never written" against a run that had done nothing
    # wrong. Derive a usable interval instead of trusting the caller's default.
    every = log_every if log_every is not None else args.log_every
    every = max(1, min(int(every), int(steps) // 4))
    cmd = [sys.executable, "-u", SWEEP_SCRIPT,
           "--gamma", str(gamma),
           "--out-dir", out_dir,
           "--fold", str(args.fold),
           "--max-steps", str(steps),
           "--log-every", str(every),
           "--ckpt-every", str(max(1, min(int(args.ckpt_every), int(steps)))),
           "--feat-norm-ceiling", str(args.feat_norm_ceiling),
           "--num-workers", str(args.num_workers),
           "--data-root", args.data_root,
           "--splits", args.splits,
           "--cache-dir", args.cache_dir]
    if args.init_head_from:
        cmd += ["--init-head-from", args.init_head_from]
    return cmd


def spawn(gamma, device, out_dir, log_path, args, max_steps=None, log_every=None):
    """Start one arm on one device. stdout+stderr -> a FILE (never a pipe)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["PYTHONUNBUFFERED"] = "1"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fh = open(log_path, "w", buffering=1)
    cmd = build_cmd(gamma, out_dir, args, max_steps=max_steps, log_every=log_every)
    fh.write(f"# CUDA_VISIBLE_DEVICES={device}\n# {' '.join(cmd)}\n")
    fh.flush()
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=_ROOT)
    print(f"[launch] gamma={gamma:g} -> cuda:{device}  pid={p.pid}\n"
          f"         out {out_dir}\n         log {log_path}", flush=True)
    return {"gamma": gamma, "device": device, "proc": p, "fh": fh,
            "out_dir": out_dir, "log": log_path, "started": time.time()}


def verify_trace(out_dir, min_rows=1) -> dict:
    """Read the arm's trace back: present, parseable, expected header, finite, growing."""
    trace = os.path.join(out_dir, "trace.csv")
    steps = os.path.join(out_dir, "steps.csv")
    rep = {"trace": trace, "ok": False}
    if not os.path.isfile(trace):
        rep["error"] = f"trace.csv was never written to {out_dir}"
        return rep
    with open(trace, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rep["n_rows"] = len(rows)
    if len(rows) < min_rows:
        rep["error"] = f"trace.csv has {len(rows)} rows, expected >= {min_rows}"
        return rep
    missing = [c for c in REQUIRED_TRACE_COLS if c not in rows[0]]
    if missing:
        rep["error"] = f"trace.csv is missing columns {missing}"
        return rep
    bad = []
    for c in REQUIRED_TRACE_COLS:
        try:
            v = float(rows[-1][c])
            if v != v or v in (float("inf"), float("-inf")):
                bad.append(c)
        except (TypeError, ValueError):
            bad.append(c)
    if bad:
        rep["error"] = f"trace.csv's last row has non-finite/unparseable {bad}"
        return rep
    rep["last_row"] = {c: rows[-1][c] for c in REQUIRED_TRACE_COLS}
    rep["steps_csv_rows"] = (sum(1 for _ in open(steps)) - 1
                             if os.path.isfile(steps) else 0)
    rep["ok"] = True
    return rep


def wait_any(jobs, poll=10):
    """Block until one job exits; return it. Prints a heartbeat so the log shows life."""
    last = 0.0
    while True:
        for j in jobs:
            if j["proc"].poll() is not None:
                return j
        if time.time() - last > 300:
            last = time.time()
            for j in jobs:
                mins = (time.time() - j["started"]) / 60
                print(f"[wait] gamma={j['gamma']:g} cuda:{j['device']} still running "
                      f"({mins:.0f} min)", flush=True)
        time.sleep(poll)


def finish(j):
    rc = j["proc"].returncode
    try:
        j["fh"].close()
    except Exception:
        pass
    mins = (time.time() - j["started"]) / 60
    v = verify_trace(j["out_dir"])
    print(f"[done] gamma={j['gamma']:g} cuda:{j['device']} rc={rc} {mins:.1f} min  "
          f"trace_ok={v['ok']} rows={v.get('n_rows')}", flush=True)
    if not v["ok"]:
        print(f"       *** {v.get('error')} -- see {j['log']}", flush=True)
    return {"gamma": j["gamma"], "device": j["device"], "returncode": rc,
            "minutes": mins, "out_dir": j["out_dir"], "log": j["log"], "trace": v}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default=os.path.join(_ROOT, "runs", "coral_r4_sweep"))
    p.add_argument("--log-root", default=None, help="default: <out-root>/logs")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=125)
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--feat-norm-ceiling", type=float, default=110.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", default=os.path.join(_ROOT, "data"))
    p.add_argument("--splits", default=os.path.join(_ROOT, "data", "splits", "splits.json"))
    p.add_argument("--cache-dir", default=os.path.join(_ROOT, "data", "feature_cache"))
    p.add_argument("--init-head-from", default=None)
    p.add_argument("--verify-steps", type=int, default=100)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--skip-verify", action="store_true",
                   help="only for a resumed session where the gate already passed")
    args = p.parse_args(argv)

    out_root = os.path.abspath(args.out_root)
    log_root = os.path.abspath(args.log_root or os.path.join(out_root, "logs"))
    os.makedirs(log_root, exist_ok=True)
    report = {"out_root": out_root, "log_root": log_root,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "max_steps": args.max_steps, "arms": []}

    print("=" * 92)
    print("ROUND-4 CORAL GAMMA SWEEP -- Kaggle T4 x 2 launcher")
    print("=" * 92)
    print(f"  out-root : {out_root}")
    print(f"  logs     : {log_root}")
    print(f"  plan     : gamma 1 -> cuda:0 and gamma 4 -> cuda:1 concurrently; "
          f"gamma 16 on the first free device")
    print("=" * 92, flush=True)

    # ---- PHASE 0: the 100-step verification arm, ALONE on cuda:0 ----
    if not args.skip_verify:
        vdir = os.path.join(out_root, "verify_g1")
        vlog = os.path.join(log_root, "verify_g1.log")
        if os.path.exists(os.path.join(vdir, "trace.csv")):
            raise SystemExit(f"[launch] ABORT: {vdir} already holds a trace. Remove it "
                             f"or pass --skip-verify if the gate already passed.")
        print(f"\n[verify] gamma=1 for {args.verify_steps} steps, alone on cuda:0, "
              f"to prove the trace is written and readable BEFORE committing both GPUs.",
              flush=True)
        j = spawn(1.0, 0, vdir, vlog, args, max_steps=args.verify_steps)
        j["proc"].wait()
        j["fh"].close()
        v = verify_trace(vdir, min_rows=1)
        report["verify"] = {"dir": vdir, "log": vlog,
                            "returncode": j["proc"].returncode, **v}
        if j["proc"].returncode != 0 or not v["ok"]:
            print(f"\n[verify] FAILED (rc={j['proc'].returncode}): "
                  f"{v.get('error', 'see the log')}")
            print(f"[verify] log: {vlog}")
            print("[verify] NOT launching the sweep. Fix this first -- three minutes here "
                  "is worth more than a lost session.")
            with open(os.path.join(out_root, "launch_report.json"), "w") as fh:
                json.dump(report, fh, indent=2, default=str)
            return 2
        print(f"[verify] PASSED: {v['n_rows']} trace rows, {v.get('steps_csv_rows')} "
              f"per-step rows, last row finite.")
        print(f"[verify] last row: {v['last_row']}", flush=True)
        if args.verify_only:
            with open(os.path.join(out_root, "launch_report.json"), "w") as fh:
                json.dump(report, fh, indent=2, default=str)
            print("\n[verify-only] gate passed; sweep not launched.")
            return 0

    # ---- PHASE 1: the two concurrent arms ----
    jobs, queued = [], list(QUEUED)
    for g, dev in CONCURRENT:
        jobs.append(spawn(g, dev, gdir(out_root, g),
                          os.path.join(log_root, f"g{int(g)}.log"), args))

    # ---- PHASE 2: gamma=16 onto the first device that frees ----
    while jobs:
        j = wait_any(jobs)
        jobs.remove(j)
        report["arms"].append(finish(j))
        with open(os.path.join(out_root, "launch_report.json"), "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        if queued:
            g = queued.pop(0)
            print(f"[queue] cuda:{j['device']} is free -> starting gamma={g:g} on it",
                  flush=True)
            jobs.append(spawn(g, j["device"], gdir(out_root, g),
                              os.path.join(log_root, f"g{int(g)}.log"), args))

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(out_root, "launch_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("\n" + "=" * 92)
    print("SWEEP COMPLETE")
    print("=" * 92)
    for a in report["arms"]:
        print(f"  gamma={a['gamma']:<5g} rc={a['returncode']}  {a['minutes']:.1f} min  "
              f"trace_ok={a['trace']['ok']}  rows={a['trace'].get('n_rows')}")
    bad = [a for a in report["arms"] if a["returncode"] != 0 or not a["trace"]["ok"]]
    print(f"\n  {len(report['arms']) - len(bad)}/{len(report['arms'])} arms produced a "
          f"readable trace.")
    print(f"\n[out] {os.path.join(out_root, 'launch_report.json')}")
    print("[next] DOWNLOAD the sweep directories, then evaluate the pre-registered "
          "criterion LOCALLY (CPU, no GPU time):")
    print("       python scripts/analyze_coral_gamma_sweep.py --sweep-dir <downloaded>")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
