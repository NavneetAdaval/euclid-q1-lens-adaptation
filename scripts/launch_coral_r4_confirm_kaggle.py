"""
launch_coral_r4_confirm_kaggle.py
=================================
Runs round-4 CORAL confirmatory folds. **ONE FOLD AT A TIME BY DEFAULT.**

    --folds 1            one fold, this session          (~75 min)
    --folds 1 2 3 4      all four, SEQUENTIALLY          (~5 h, the default set)
    --concurrency 2      two at a time, one per device   (opt in, T4 x2 only)

The folds are independent runs, so how they are scheduled changes nothing about the result
-- it is wall-clock bookkeeping only. Running one at a time is the safer default: a session
that dies takes one fold with it, the logs are readable while they are being written, and
there is no question about two processes contending for the same GPU.

**A single fold is never split across devices**, at any concurrency: training is sequential,
and halving the per-device batch would change CORAL's 640x640 covariance estimate from 256
draws to 128 (PROJECT.md 7.1a -- the same reason gradient accumulation is refused for this
method). Every child gets CUDA_VISIBLE_DEVICES pinned to exactly one GPU, so it cannot shard
even by accident.

Every operational rule the round has paid for is encoded here rather than remembered
(`runs/coral_round4_design.md` §9):

    - `python -u`, stdout+stderr to that fold's OWN FILE, never a pipe. The calibration's
      log sat at 0 bytes for an hour because `grep` block-buffered it.
    - one output directory per fold, and the runner refuses to start over an existing run,
      so two invocations cannot interleave rows into one series.
    - liveness via `Popen.poll()`. Git Bash's `ps -p` cannot see native Windows PIDs and
      once reported a live run dead.
    - a short verification fold, read back, BEFORE anything long starts.
    - `launch_report.json` MERGES across invocations, so running fold 1 today and fold 2
      tomorrow still builds one report, and it says which folds are still missing.

Cost, measured: 298 min for 8000 steps on one T4, so ~75 min per fold at the pre-registered
2000. One fold per session is comfortable; all four sequentially is ~5 h, still inside the
12 h cap.

Run:
    python -u scripts/launch_coral_r4_confirm_kaggle.py --folds 1 --verify-only
    python -u scripts/launch_coral_r4_confirm_kaggle.py --folds 1 --skip-verify
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
RUNNER = os.path.join(_HERE, "run_coral_r4_confirm.py")

DEFAULT_FOLDS = (1, 2, 3, 4)      # fold 0 is DEVELOPMENT; the runner refuses it outright
DEFAULT_CONCURRENCY = 1           # one fold at a time
VERIFY_FOLD = 1
VERIFY_STEPS = 50


def _confirmatory_fold(v):
    """argparse type: the launcher cannot schedule fold 0 either."""
    try:
        k = int(v)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--folds takes integers, got {v!r}")
    if k == 0:
        raise argparse.ArgumentTypeError(
            "FOLD 0 IS DEVELOPMENT (PROJECT.md 7.11/7.14) and cannot be scheduled as a "
            "confirmatory fold: the budget was chosen by inspecting its own curve. "
            "Confirmatory folds are 1, 2, 3, 4.")
    if k not in DEFAULT_FOLDS:
        raise argparse.ArgumentTypeError(f"--folds must be from {list(DEFAULT_FOLDS)}, "
                                         f"got {k}")
    return k


def waves_for(folds, concurrency):
    """Chunk the requested folds into waves of `concurrency`. Default: one per wave."""
    f = list(folds)
    n = max(1, int(concurrency))
    return [tuple(f[i:i + n]) for i in range(0, len(f), n)]


def build_cmd(fold, out_dir, args, max_steps=None):
    # The log interval MUST divide into the budget or the arm writes NO trace rows and exits
    # 0 -- exactly how the sweep's 100-step verify arm "failed". Derive a usable interval
    # instead of trusting the default.
    cmd = [sys.executable, "-u", RUNNER,
           "--fold", str(fold),
           "--out-dir", out_dir,
           "--num-workers", str(args.num_workers),
           "--ckpt-every", str(args.ckpt_every),
           "--log-every", str(args.log_every)]
    if max_steps is not None:
        every = max(1, min(int(args.log_every), int(max_steps) // 4))
        cmd += ["--max-steps", str(max_steps), "--log-every", str(every),
                "--ckpt-every", str(max(1, min(int(args.ckpt_every), int(max_steps))))]
    for flag, val in (("--data-root", args.data_root), ("--splits", args.splits),
                      ("--cache-dir", args.cache_dir), ("--prereg", args.prereg),
                      ("--dev-run-json", args.dev_run_json),
                      ("--reference-ids", args.reference_ids),
                      ("--input-root", args.input_root)):
        if val:
            cmd += [flag, val]
    return cmd


def spawn(fold, device, out_dir, log_path, args, max_steps=None):
    """Start one fold on one device. stdout+stderr -> a FILE (never a pipe)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)     # the child sees ONE GPU, so it cannot shard
    env["PYTHONUNBUFFERED"] = "1"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fh = open(log_path, "w", buffering=1)
    cmd = build_cmd(fold, out_dir, args, max_steps=max_steps)
    fh.write(f"# CUDA_VISIBLE_DEVICES={device}\n# {' '.join(cmd)}\n")
    fh.flush()
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=_ROOT)
    print(f"[launch] fold {fold} -> cuda:{device}  pid={p.pid}\n"
          f"          out {out_dir}\n          log {log_path}", flush=True)
    return {"fold": fold, "device": device, "proc": p, "fh": fh, "log": log_path,
            "out_dir": out_dir, "started": time.time()}


def wait_all(jobs, poll=30):
    """Block until every job exits, reporting progress. Liveness via Popen.poll()."""
    done = []
    while jobs:
        time.sleep(poll)
        still = []
        for j in jobs:
            rc = j["proc"].poll()
            if rc is None:
                still.append(j)
            else:
                j["fh"].close()
                mins = (time.time() - j["started"]) / 60.0
                print(f"[done] fold {j['fold']} cuda:{j['device']} rc={rc} "
                      f"{mins:.1f} min", flush=True)
                done.append({"fold": j["fold"], "device": j["device"], "returncode": rc,
                             "minutes": mins, "out_dir": j["out_dir"], "log": j["log"]})
        if still and len(still) != len(jobs):
            pass
        jobs = still
        if jobs:
            names = ", ".join(f"fold {j['fold']} ({(time.time()-j['started'])/60:.0f} min)"
                              for j in jobs)
            print(f"[wait] still running: {names}", flush=True)
    return done


def read_back(out_dir, fold, budget):
    """Did this fold actually produce the REPORTED artifact? Read it back, do not assume."""
    d = os.path.join(out_dir, f"fold{fold}")
    rj = os.path.join(d, "run.json")
    out = {"fold": fold, "dir": d, "run_json": os.path.isfile(rj)}
    if not out["run_json"]:
        out["ok"] = False
        return out
    with open(rj) as fh:
        r = json.load(fh)
    ck = os.path.join(d, "checkpoints", f"step_{budget}.pt")
    lg = glob_one(os.path.join(d, "logits"), f"step_{budget}.npz")
    out.update({"steps_completed": r.get("steps_completed"), "aborted": r.get("aborted"),
                "abort_reason": r.get("abort_reason"),
                "reported_checkpoint": ck, "reported_checkpoint_present": os.path.isfile(ck),
                "reported_logits": lg, "reported_logits_present": bool(lg),
                "step0_gate": (r.get("step0_gate") or {}).get("verdict"),
                "config_mismatches": (r.get("config_diff") or {}).get("mismatches"),
                "prereg_sha256": (r.get("pre_registration") or {}).get("sha256"),
                "n_test_positives": (r.get("fold_facts") or {}).get("n_test_positives"),
                "wall_minutes": r.get("wall_minutes")})
    out["ok"] = bool(out["reported_checkpoint_present"] and out["reported_logits_present"]
                     and not r.get("aborted")
                     and int(r.get("steps_completed") or 0) >= int(budget))
    return out


def glob_one(root, name):
    for dirpath, _d, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    return None


def load_report(out_root):
    """Read an existing launch_report.json so separate invocations accumulate into one.

    Running one fold at a time means the report is written once per fold; without this each
    write would erase the previous folds and the file would claim they were never run.
    """
    p = os.path.join(out_root, "launch_report.json")
    if os.path.isfile(p):
        try:
            with open(p) as fh:
                r = json.load(fh)
            r.setdefault("folds", [])
            return r
        except (OSError, ValueError):
            pass
    return {"folds": [], "verify": None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="/kaggle/working/runs/coral_r4_confirm")
    ap.add_argument("--input-root", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--dev-run-json", default=None)
    ap.add_argument("--reference-ids", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=125)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--budget", type=int, default=2000,
                    help="for the read-back only; the RUNNER takes the budget from the "
                         "hashed pre-registration, never from this flag")
    ap.add_argument("--folds", nargs="+", type=_confirmatory_fold,
                    default=list(DEFAULT_FOLDS),
                    help="which confirmatory folds to run in THIS invocation "
                         "(default: all four, sequentially). Fold 0 is refused.")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    choices=(1, 2),
                    help="folds at a time. 1 (default) = one fold at a time on one "
                         "device. 2 = one fold per device on a T4 x2. A fold is "
                         "NEVER split across devices at either setting.")
    ap.add_argument("--device-index", type=int, default=0,
                    help="which GPU to pin when --concurrency 1")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--verify-only", action="store_true", default=False)
    ap.add_argument("--skip-verify", action="store_true", default=False)
    a = ap.parse_args(argv)

    folds = sorted(dict.fromkeys(a.folds))
    waves = waves_for(folds, a.concurrency)
    out_root = os.path.abspath(a.out_root)
    log_root = os.path.join(out_root, "logs")
    os.makedirs(log_root, exist_ok=True)
    report = load_report(out_root)      # MERGE: folds run in separate sessions accumulate
    report.update({"out_root": out_root, "log_root": log_root,
                   "last_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "waves_this_invocation": [list(w) for w in waves],
                   "budget_for_readback": a.budget})

    print("=" * 92)
    print(f"ROUND-4 CORAL CONFIRMATORY -- folds {folds}, "
          f"{'ONE AT A TIME' if a.concurrency == 1 else 'two at a time, one per device'}")
    print("=" * 92)
    print(f"  out root    : {out_root}")
    print(f"  waves       : {' then '.join(str(list(w)) for w in waves)}")
    print(f"  concurrency : {a.concurrency}  (CUDA_VISIBLE_DEVICES pinned per fold; a fold "
          f"is NEVER split across devices)")
    print(f"  est. cost   : ~75 min per fold at 2000 steps -> about "
          f"{75 * len(waves) / 60:.1f} h for this invocation")
    print(f"  NOTE        : fold 0 is DEVELOPMENT and is refused by the runner AND here.")
    print("=" * 92, flush=True)

    # ---- verification fold: short, alone on cuda:0, read back before committing ----
    if not a.skip_verify:
        vdir = os.path.join(out_root, "_verify")
        vlog = os.path.join(log_root, f"verify_fold{VERIFY_FOLD}.log")
        print(f"\n[verify] fold {VERIFY_FOLD}, {VERIFY_STEPS} steps, alone on cuda:0 ...",
              flush=True)
        j = spawn(VERIFY_FOLD, 0, vdir, vlog, a, max_steps=VERIFY_STEPS)
        rc = j["proc"].wait()
        j["fh"].close()
        chk = read_back(vdir, VERIFY_FOLD, VERIFY_STEPS)
        chk["returncode"] = rc
        report["verify"] = chk
        print(f"[verify] rc={rc}  steps={chk.get('steps_completed')}  "
              f"gate={chk.get('step0_gate')}  mismatches={chk.get('config_mismatches')}  "
              f"positives={chk.get('n_test_positives')}")
        if rc != 0 or not chk.get("ok"):
            print(f"\n[verify] FAILED -- see {vlog}. NOT committing both devices.",
                  flush=True)
            with open(os.path.join(out_root, "launch_report.json"), "w") as fh:
                json.dump(report, fh, indent=1)
            return 1
        print("[verify] PASSED. The mount, the pre-registration gate, the config diff, the "
              "step-0 gate and the logit dump all work on this machine.\n", flush=True)
        if a.verify_only:
            with open(os.path.join(out_root, "launch_report.json"), "w") as fh:
                json.dump(report, fh, indent=1)
            print("[verify] --verify-only: stopping here. Re-run with --skip-verify.")
            return 0

    # ---- the folds; a wave is ONE fold unless --concurrency 2 ----
    done = []
    for wave in waves:
        jobs = []
        for slot, fold in enumerate(wave):
            device = a.device_index if a.concurrency == 1 else slot
            jobs.append(spawn(fold, device, out_root,
                              os.path.join(log_root, f"fold{fold}.log"), a))
        done += wait_all(jobs, poll=a.poll)

    for f in done:
        f.update(read_back(out_root, f["fold"], a.budget))
    by_fold = {int(f["fold"]): f for f in report.get("folds", [])}
    by_fold.update({int(f["fold"]): f for f in done})        # this invocation wins
    report["folds"] = [by_fold[k] for k in sorted(by_fold)]
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["folds_complete"] = sorted(k for k, f in by_fold.items() if f.get("ok"))
    report["folds_outstanding"] = [k for k in DEFAULT_FOLDS
                                   if k not in report["folds_complete"]]
    report["all_ok"] = (report["folds_outstanding"] == [])
    with open(os.path.join(out_root, "launch_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    print("\n" + "=" * 92)
    print("SUMMARY -- the REPORTED checkpoint is step_%d for every fold" % a.budget)
    print("=" * 92)
    print(f"  {'fold':>4} {'rc':>3} {'steps':>6} {'pos':>4} {'gate':>5} {'min':>6}  "
          f"reported ckpt + logits")
    for f in sorted(report["folds"], key=lambda x: x["fold"]):
        print(f"  {f['fold']:>4} {f['returncode']:>3} {str(f.get('steps_completed')):>6} "
              f"{str(f.get('n_test_positives')):>4} {str(f.get('step0_gate')):>5} "
              f"{f['minutes']:>6.1f}  "
              f"{'OK' if f.get('ok') else 'MISSING/INCOMPLETE'}")
    print(f"\n  complete so far : {report['folds_complete']}")
    if report["folds_outstanding"]:
        print(f"  STILL TO RUN    : {report['folds_outstanding']}   (re-run with "
              f"--folds {' '.join(str(k) for k in report['folds_outstanding'])} "
              f"--skip-verify)")
    print(f"  all four folds complete and reported artifacts present: {report['all_ok']}")
    print(f"  [out] {os.path.join(out_root, 'launch_report.json')}")
    print("\n  The intermediate 250-step checkpoints are LAB RECORD ONLY. The result is "
          "step_%d." % a.budget)
    print("=" * 92, flush=True)
    # EXIT ON THIS INVOCATION, not on the whole four-fold set. With one fold at a
    # time, "all four done" is false after a perfectly good single fold, and exiting
    # non-zero there makes subprocess.run(check=True) raise on a run that succeeded.
    this_ok = bool(done) and all(f.get("ok") for f in done) and len(done) == len(folds)
    return 0 if this_ok else 1


if __name__ == "__main__":
    sys.exit(main())
