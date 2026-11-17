"""
package_screen_results.py
=========================
Task 5: get EVERYTHING home from a Kaggle screen session, in pieces a browser can download.

THE FAILURE THIS EXISTS TO PREVENT
-----------------------------------
Task 4's A1 study could not compute IM/SND/ATC on ``nb1`` or ``nb3`` -- the two runs the
verdict rested on -- because only ``step_8000.pt`` came home. The trainer had written all 32
checkpoints; the Kaggle COLLECTION cell copied one. So this script's first job is not
archiving, it is REFUSING to archive an incomplete run.

TWO INDEPENDENT RETENTION CHECKS, deliberately:
  1. ``run_screen_unfreeze.assert_checkpoint_retention`` -- at end of training, the files
     exist on the training box.
  2. HERE -- the files made it into the archives, verified by content hash.
Only (2) would have caught what actually happened.

WHY INDEPENDENT PARTS, NOT ``split``
-------------------------------------
``tar | split`` produces parts that are useless alone: nothing can be verified or extracted
until every part is present and concatenated in order. A ~2 GB browser download that fails
at 90% then has no incremental recovery. Instead each part here is a SELF-CONTAINED tar
holding a whole number of checkpoints, so a part that fails to download can be re-fetched
and verified on its own, and the parts can be extracted in any order.

Checkpoints are stored UNCOMPRESSED (``tar``, not ``tar.gz``): they are float32 weight
tensors, which deflate barely shrinks, and compressing 1.8 GB would cost session minutes for
a few percent. The small archive IS gzipped -- it is text and tiny.

VERIFICATION IS STREAMING, NOT AN EXTRACTION. Each part is re-opened and every member is
hashed straight from the archive and compared to the hash of the original file on disk. That
proves reassembly without writing a second 1.8 GB copy into ``/kaggle/working``.

USAGE
    # on Kaggle, at the end of the session
    python scripts/package_screen_results.py --runs-dir /kaggle/working/scripts/../runs \\
        --out /kaggle/working/screen_out --logs /kaggle/working/logs \\
        --declared-reference /kaggle/working/ref_zoobot/run_config.json

    # locally, after downloading -- checks every part against the manifest
    python scripts/package_screen_results.py --verify /path/to/downloaded/screen_out
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ~500 MB per the brief, in DECIMAL MB because that is what a browser reports. Held a little
# under so a part never lands just over the line.
DEFAULT_MAX_PART_BYTES = 480 * 1000 * 1000
CHUNK = 1 << 20


def sha256_file(path, chunk=CHUNK):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_stream(fobj, chunk=CHUNK):
    h = hashlib.sha256()
    for b in iter(lambda: fobj.read(chunk), b""):
        h.update(b)
    return h.hexdigest()


def _gb(n):
    return n / 2 ** 30


# ---------------------------------------------------------------------------
# retention check #2 + the post-hoc §1 reference verification
# ---------------------------------------------------------------------------

def _retention_module(cfg):
    """Which runner's retention convention this run follows.

    The Lever-1 screen retained ONE checkpoint per curve point, so its expected-step set is
    keyed to ``curve_every``. The FreeMatch x nb3 run separates the two (16 weights at 500,
    64 curve points at 125 -- BRIEF §5.1, made affordable by the derived-logit dump), so its
    set is keyed to ``ckpt_every`` and includes step 0. Dispatching on the config the run
    actually wrote is what keeps ONE packaging path correct for both, instead of a second
    packager that could drift.
    """
    if cfg.get("ckpt_every") is not None:
        import run_unfreeze_freematch as m
        return m, int(cfg["ckpt_every"])
    import run_screen_unfreeze as m
    return m, int(cfg.get("curve_every", 0))


def check_retention(run_dir, verbose=True):
    """Read the run's own config and confirm every expected checkpoint is on disk."""
    cfg_p = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(cfg_p):
        return {"ok": False, "error": f"no run_config.json in {run_dir}"}
    cfg = json.load(open(cfg_p))
    rs, every = _retention_module(cfg)
    budget = int(cfg.get("budget_steps", 0) or cfg.get("max_steps", 0))
    if not budget or not every:
        return {"ok": False, "error": "run_config lacks budget_steps/curve_every|ckpt_every"}
    want = rs.expected_checkpoint_steps(budget, every)
    ck = os.path.join(run_dir, "checkpoints")
    have = sorted(int(f[5:-3]) for f in os.listdir(ck)
                  if f.startswith("step_") and f.endswith(".pt")) if os.path.isdir(ck) else []
    missing = [s for s in want if s not in set(have)]
    out = {"ok": not missing, "n_expected": len(want), "n_found": len(have),
           "missing": missing[:12],
           "bytes": sum(os.path.getsize(os.path.join(ck, f))
                        for f in os.listdir(ck)) if os.path.isdir(ck) else 0}
    if verbose:
        tag = "ok  " if out["ok"] else "FAIL"
        print(f"  [{tag}] {os.path.basename(run_dir):<40} "
              f"{out['n_found']}/{out['n_expected']} checkpoints  "
              f"({_gb(out['bytes']):.2f} GB)")
        if missing:
            print(f"         MISSING {missing}")
    return out


def verify_declared_reference(declared_path, nb0_run_dir, verbose=True):
    """Post-hoc §1 closure: the reference DECLARED up front (so the pair could run
    concurrently) must equal the config nb0 ACTUALLY emitted, on every pinned key.

    Without this, ``--emit-config`` would be a way to hand nb3 a reference that no real run
    ever produced. With it, concurrency costs nothing in rigour.
    """
    emitted_p = os.path.join(nb0_run_dir, "run_config.json")
    if not os.path.isfile(declared_path):
        return {"ok": False, "error": f"declared reference not found: {declared_path}"}
    if not os.path.isfile(emitted_p):
        return {"ok": False, "error": f"nb0 emitted no run_config.json at {emitted_p}"}
    # the PINNED key set is the one belonging to the runner that produced the config, for
    # the same reason as _retention_module: the FreeMatch pair pins the FreeMatch knobs
    # (a silent local_threshold_mode change is worth ~0.18 TPR, §7.4) which the Lever-1
    # Baseline A key set does not contain at all.
    rs, _ = _retention_module(json.load(open(emitted_p)))
    dec = rs._normalise_cfg(json.load(open(declared_path)))
    emi = rs._normalise_cfg(json.load(open(emitted_p)))
    diffs = {k: {"declared": dec.get(k, "<absent>"), "emitted": emi.get(k, "<absent>")}
             for k in rs.PINNED_KEYS if dec.get(k, "<absent>") != emi.get(k, "<absent>")}
    out = {"ok": not diffs, "diffs": diffs, "declared": declared_path,
           "emitted": emitted_p, "n_pinned": len(rs.PINNED_KEYS)}
    if verbose:
        if diffs:
            print(f"  [FAIL] declared reference != nb0's emitted config on {len(diffs)} "
                  f"pinned key(s):")
            for k, v in diffs.items():
                print(f"         {k}: declared={v['declared']!r} emitted={v['emitted']!r}")
        else:
            print(f"  [ok  ] declared reference matches nb0's emitted config on all "
                  f"{len(rs.PINNED_KEYS)} pinned keys -- concurrency did not weaken §1")
    return out


# ---------------------------------------------------------------------------
# archive building
# ---------------------------------------------------------------------------

def plan_parts(files, max_bytes):
    """Greedily pack (arcname, path, size) into parts, each <= max_bytes.

    A single file larger than max_bytes still gets its own part rather than being split --
    the part-independence property matters more than the exact cap.
    """
    parts, cur, cur_b = [], [], 0
    for arc, p, sz in files:
        if cur and cur_b + sz > max_bytes:
            parts.append(cur)
            cur, cur_b = [], 0
        cur.append((arc, p, sz))
        cur_b += sz
    if cur:
        parts.append(cur)
    return parts


def build_checkpoint_parts(run_dir, run_name, out_dir, max_bytes, verbose=True):
    """One or more self-contained tars, each holding whole checkpoints."""
    ck = os.path.join(run_dir, "checkpoints")
    # EVERY .pt, not just step_*: train_freematch_fold also writes an end-of-epoch
    # checkpoint, and the reassembly verification compares against the full source set --
    # so packing only step_* left a real file in no part at all (caught by that check).
    def _key(f):
        m = re.fullmatch(r"step_(\d+)\.pt", f)
        return (0, int(m.group(1))) if m else (1, f)

    names = sorted((f for f in os.listdir(ck) if f.endswith(".pt")), key=_key)
    files = [(f"{run_name}/checkpoints/{f}", os.path.join(ck, f),
              os.path.getsize(os.path.join(ck, f))) for f in names]
    plan = plan_parts(files, max_bytes)
    n = len(plan)
    written = []
    for i, members in enumerate(plan, 1):
        name = f"ckpt_{run_name}_part{i:02d}of{n:02d}.tar"
        path = os.path.join(out_dir, name)
        t0 = time.time()
        with tarfile.open(path, "w") as tf:          # UNCOMPRESSED: float weights
            for arc, p, _ in members:
                tf.add(p, arcname=arc)
        size = os.path.getsize(path)
        rec = {"name": name, "bytes": size, "mb": round(size / 1e6, 1),
               "sha256": sha256_file(path), "n_files": len(members),
               "members": [m[0] for m in members],
               "steps": [int(mm.group(1)) for mm in
                         (re.fullmatch(r"step_(\d+)\.pt", os.path.basename(m[0]))
                          for m in members) if mm]}
        written.append(rec)
        if verbose:
            print(f"  wrote {name}  {rec['mb']:>7.1f} MB  {len(members)} ckpts  "
                  f"steps {rec['steps'][0] if rec['steps'] else '-'}-"
                  f"{rec['steps'][-1] if rec['steps'] else '-'}  ({time.time()-t0:.0f}s)")
    return written


def build_core_archive(run_dirs, out_dir, logs_dir, extra_files, small_ckpt_limit,
                       verbose=True):
    """One small gzipped tar with everything that is NOT a large checkpoint set."""
    name = "screen_core.tar.gz"
    path = os.path.join(out_dir, name)
    added = []
    with tarfile.open(path, "w:gz") as tf:
        for rd in run_dirs:
            rn = os.path.basename(rd)
            # diagnostics.csv / memorization.csv are absent from Lever-1 runs and present
            # in the FreeMatch x nb3 pair (BRIEF §5.3/§5.4); listing both costs nothing.
            # every small per-run artifact. Enumerating by GLOB rather than by a hand-kept
            # list: gate_fold0.json -- a BRIEF §9 required output -- was silently left out
            # of an archive because it post-dated the list, and a required output that can
            # go missing without anything noticing is the failure mode §7.7 is about.
            for p in sorted(glob.glob(os.path.join(rd, "*.csv"))
                            + glob.glob(os.path.join(rd, "*.json"))
                            + glob.glob(os.path.join(rd, "*.log"))):
                if os.path.isfile(p):
                    f = os.path.basename(p)
                    tf.add(p, arcname=f"{rn}/{f}")
                    added.append(f"{rn}/{f}")
            # ...and every SUBDIRECTORY except checkpoints, which the size-aware path below
            # handles. Globbed for the same reason as the files above: a ("scores",
            # "logits") tuple silently drops whatever a future run adds next.
            #
            # WALKED RECURSIVELY, not listed one level deep. The single-level version
            # dropped every per-checkpoint logit file in round 5: the runner writes them
            # to logits/<arm>_fold<k>/step_N.npz -- TWO levels down -- so os.listdir()
            # saw one directory entry, `isfile` said no, and 32 files per arm left the
            # venue silently in both arm 1 and arm 2. The comment above was right that a
            # hand-kept tuple is the hazard; a hand-kept DEPTH is the same hazard wearing
            # a different hat. The manifest check in pack_results() is the real backstop.
            for sd in sorted(d for d in glob.glob(os.path.join(rd, "*"))
                             if os.path.isdir(d)
                             and os.path.basename(d) != "checkpoints"):
                for root, _dirs, files in os.walk(sd):
                    for s in sorted(files):
                        fp = os.path.join(root, s)
                        if os.path.isfile(fp):
                            rel = os.path.relpath(fp, rd).replace("\\", "/")
                            tf.add(fp, arcname=f"{rn}/{rel}")
                            added.append(f"{rn}/{rel}")
            # small checkpoint sets (nb0: 17 KB each) ride along in the core archive
            ck = os.path.join(rd, "checkpoints")
            if os.path.isdir(ck):
                tot = sum(os.path.getsize(os.path.join(ck, f)) for f in os.listdir(ck))
                if tot <= small_ckpt_limit:
                    for f in sorted(os.listdir(ck), key=lambda x: int(x[5:-3])
                                    if x.startswith("step_") else 0):
                        tf.add(os.path.join(ck, f), arcname=f"{rn}/checkpoints/{f}")
                        added.append(f"{rn}/checkpoints/{f}")
        if logs_dir and os.path.isdir(logs_dir):
            for lg in sorted(os.listdir(logs_dir)):
                tf.add(os.path.join(logs_dir, lg), arcname=f"logs/{lg}")
                added.append(f"logs/{lg}")
        for p in extra_files:
            if os.path.isfile(p):
                tf.add(p, arcname=f"meta/{os.path.basename(p)}")
                added.append(f"meta/{os.path.basename(p)}")
    size = os.path.getsize(path)
    rec = {"name": name, "bytes": size, "mb": round(size / 1e6, 1),
           "sha256": sha256_file(path), "n_files": len(added), "members": added}
    if verbose:
        print(f"  wrote {name}  {rec['mb']:>7.1f} MB  {len(added)} files")
    return rec


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_parts_against_source(parts, out_dir, source_hashes, verbose=True):
    """Stream every member out of every part and hash-compare against the ORIGINAL file.

    This is the "do the parts reassemble" check, done without writing a second copy: if
    every expected member appears exactly once across the parts and its bytes hash to the
    same value as the file on disk, the set reassembles.
    """
    seen, problems = {}, []
    for rec in parts:
        p = os.path.join(out_dir, rec["name"])
        actual = sha256_file(p)
        if actual != rec["sha256"]:
            problems.append(f"{rec['name']}: archive sha256 mismatch")
            continue
        with tarfile.open(p, "r") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                f = tf.extractfile(m)
                h = sha256_stream(f)
                if m.name in seen:
                    problems.append(f"{m.name} appears in more than one part")
                seen[m.name] = h
                want = source_hashes.get(m.name)
                if want is None:
                    problems.append(f"{m.name} is in an archive but not in the source set")
                elif want != h:
                    problems.append(f"{m.name}: content hash differs from the original")
    missing = [k for k in source_hashes if k not in seen]
    for k in missing:
        problems.append(f"{k} is in the source set but in NO part")
    ok = not problems
    if verbose:
        print(f"  members checked: {len(seen)}  expected: {len(source_hashes)}")
        if ok:
            print("  [ok  ] every member reassembles byte-identically from the parts")
        else:
            print(f"  [FAIL] {len(problems)} problem(s):")
            for x in problems[:12]:
                print(f"         {x}")
    return {"ok": ok, "n_members": len(seen), "problems": problems[:40]}


def verify_downloaded(out_dir):
    """Local mode: re-check every part in a downloaded directory against MANIFEST.json."""
    man_p = os.path.join(out_dir, "MANIFEST.json")
    if not os.path.isfile(man_p):
        print(f"no MANIFEST.json in {out_dir}")
        return 1
    man = json.load(open(man_p))
    print("=" * 78)
    print(f"VERIFYING DOWNLOAD: {out_dir}")
    print(f"  packaged {man.get('created')}  total {man.get('total_mb')} MB "
          f"in {len(man['parts'])} part(s)")
    print("=" * 78)
    bad, missing = [], []
    for rec in man["parts"]:
        p = os.path.join(out_dir, rec["name"])
        if not os.path.isfile(p):
            print(f"  MISSING  {rec['name']:<34} ({rec['mb']} MB) -- re-download this part")
            missing.append(rec["name"])
            continue
        sz = os.path.getsize(p)
        if sz != rec["bytes"]:
            print(f"  TRUNCATED{rec['name']:<34} {sz/1e6:.1f} MB != {rec['mb']} MB "
                  f"-- re-download this part")
            bad.append(rec["name"])
            continue
        h = sha256_file(p)
        if h != rec["sha256"]:
            print(f"  CORRUPT  {rec['name']:<34} sha256 mismatch -- re-download this part")
            bad.append(rec["name"])
        else:
            print(f"  ok       {rec['name']:<34} {rec['mb']:>7.1f} MB  "
                  f"{rec['n_files']:>3} files")
    print("-" * 78)
    if missing or bad:
        print(f"FAILED: {len(missing)} missing, {len(bad)} corrupt. Re-download ONLY these:")
        for n in missing + bad:
            print(f"  {n}")
        return 1
    print("ALL PARTS PRESENT AND INTACT.")
    print("\nExtract with:")
    print(f"  cd {out_dir} && for f in *.tar *.tar.gz; do tar -xf \"$f\"; done")
    return 0


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", metavar="DIR", default=None,
                    help="local mode: verify a downloaded screen_out against its MANIFEST")
    ap.add_argument("--runs-dir", default=os.path.join(_ROOT, "runs"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "screen_out"))
    ap.add_argument("--logs", default=None)
    ap.add_argument("--pattern", default="baselineA_fold0_*")
    ap.add_argument("--declared-reference", default=None,
                    help="the --emit-config reference nb3 validated against; checked "
                         "post-hoc against nb0's emitted config")
    ap.add_argument("--nb0-run", default=None,
                    help="nb0 run dir name for the reference check (default: auto-detect)")
    ap.add_argument("--max-part-bytes", type=int, default=DEFAULT_MAX_PART_BYTES)
    ap.add_argument("--small-ckpt-limit", type=int, default=64 * 1000 * 1000,
                    help="checkpoint sets below this total ride in the core archive")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="package even if a run is missing checkpoints (NOT the default: "
                         "an incomplete run is what broke the Task-4 validator study)")
    args = ap.parse_args(argv)

    if args.verify:
        return verify_downloaded(args.verify)

    import glob
    run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs_dir, args.pattern))
                      if os.path.isdir(d))
    if not run_dirs:
        print(f"no run dirs matching {args.pattern} under {args.runs_dir}")
        return 1

    print("=" * 78)
    print("PACKAGING SCREEN RESULTS")
    print(f"  runs : {args.runs_dir}")
    print(f"  out  : {args.out}")
    print("=" * 78)

    print("\n[1] RETENTION CHECK #2 -- did every curve point's checkpoint survive?")
    retention = {}
    for rd in run_dirs:
        retention[os.path.basename(rd)] = check_retention(rd)
    incomplete = [k for k, v in retention.items() if not v["ok"]]
    if incomplete and not args.allow_incomplete:
        print(f"\nABORT: {incomplete} are missing checkpoints. Packaging them would repeat "
              f"the Task-4 failure (only step_8000 came home, so IM/SND/ATC could not be "
              f"computed). Pass --allow-incomplete only if you have decided to accept it.")
        return 1

    ref_check = None
    if args.declared_reference:
        print("\n[2] POST-HOC §1 CHECK -- declared reference vs nb0's emitted config")
        nb0 = args.nb0_run or next(
            (os.path.basename(d) for d in run_dirs if "nb0" in os.path.basename(d)), None)
        if nb0 is None:
            print("  [FAIL] no nb0 run dir found for the reference check")
            return 1
        ref_check = verify_declared_reference(args.declared_reference,
                                              os.path.join(args.runs_dir, nb0))
        if not ref_check["ok"]:
            print("\nABORT: the reference nb3 validated against is not the config nb0 "
                  "actually produced, so the pair is NOT matched. Do not use these results.")
            return 1

    os.makedirs(args.out, exist_ok=True)
    print("\n[3] BUILDING ARCHIVES")
    parts = [build_core_archive(run_dirs, args.out, args.logs, args.extra,
                                args.small_ckpt_limit)]
    source_hashes = {}
    for rd in run_dirs:
        rn = os.path.basename(rd)
        ck = os.path.join(rd, "checkpoints")
        if not os.path.isdir(ck):
            continue
        tot = sum(os.path.getsize(os.path.join(ck, f)) for f in os.listdir(ck))
        if tot <= args.small_ckpt_limit:
            continue                                  # already in the core archive
        parts.extend(build_checkpoint_parts(rd, rn, args.out, args.max_part_bytes))
        for f in os.listdir(ck):
            source_hashes[f"{rn}/checkpoints/{f}"] = sha256_file(os.path.join(ck, f))

    print("\n[4] REASSEMBLY VERIFICATION (streamed; no second copy written)")
    ckpt_parts = [p for p in parts if p["name"].startswith("ckpt_")]
    reasm = verify_parts_against_source(ckpt_parts, args.out, source_hashes) \
        if ckpt_parts else {"ok": True, "n_members": 0, "problems": []}

    total = sum(p["bytes"] for p in parts)
    man = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_bytes": total, "total_mb": round(total / 1e6, 1),
        "n_parts": len(parts),
        "parts": parts,
        "retention": retention,
        "reference_check": ref_check,
        "reassembly_verified": reasm["ok"],
        "reassembly": reasm,
        "notes": ("Each ckpt_* part is a SELF-CONTAINED tar: extract in any order, and "
                  "re-download a single failed part without re-fetching the rest. Verify "
                  "a download with: python scripts/package_screen_results.py --verify DIR"),
    }
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"{'part':<36}{'MB':>10}{'files':>8}  sha256[:16]")
    print("-" * 78)
    for p in parts:
        print(f"{p['name']:<36}{p['mb']:>10.1f}{p['n_files']:>8}  {p['sha256'][:16]}")
    print("-" * 78)
    print(f"{'TOTAL':<36}{total/1e6:>10.1f}{sum(p['n_files'] for p in parts):>8}")
    print("=" * 78)
    # --allow-incomplete means the missing checkpoints were ACCEPTED by the operator, so
    # they must not also fail the exit code -- otherwise the flag permits nothing. The fact
    # is still recorded loudly here and in MANIFEST.retention.
    ok = reasm["ok"] and (ref_check is None or ref_check["ok"]) \
        and (not incomplete or args.allow_incomplete)
    if incomplete and args.allow_incomplete:
        print(f"WARNING: packaged with INCOMPLETE checkpoint sets ({incomplete}) because "
              f"--allow-incomplete was passed. The validator study cannot be run on those "
              f"runs. Recorded in MANIFEST.retention.")
    print("PACKAGING " + ("COMPLETE -- verified. Download every part listed above."
                          if ok else "FINISHED WITH PROBLEMS -- see above."))
    print(f"Free space check: {_gb(total):.2f} GB of archives in {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
