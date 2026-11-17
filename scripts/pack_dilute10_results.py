"""
pack_dilute10_results.py
========================
Package ONE finished dilute10 run for download (KAGGLE_DILUTE10.md, Cell 7).

  1. REFUSES unless the run's own part-2 checks pass again here: the budget is complete and
     every per-checkpoint logit on the grid, and every logit named in checkpoints.csv,
     exists (``dilute10_common.post_run_check``).
  2. Full package: ``build_dann_r5_bundle.pack_results`` -- the established packager
     (independent self-contained tars, globbed, per-part sha256, refuses an incomplete run,
     coverage check that every artifact named in checkpoints.csv is archived). The run is
     packed through a link named ``dilute10_<arm>_fold<k>`` so archive paths carry the arm.
  3. A small ANALYSIS core, ``dilute10_<arm>_fold<k>_analysis.tar.gz``: every top-level
     csv/json/log, ``logits/**``, ``scores/**``, ``gate_values.json`` -- no checkpoints, no
     resume state -- with its own coverage check over the logit grid. Added to MANIFEST.json.
  4. ``stage.json`` (Cell 2's record: bundle version, manifest, part 1/2 and pool digests)
     goes into BOTH archives -- the core as ``meta/stage.json`` (an extra file), the analysis
     core as ``dilute10_<arm>_fold<k>/stage.json`` -- and the coverage check refuses unless
     each archived copy hashes to the source (brief_dilute10_part2_provenance, change 2).

    python -u scripts/pack_dilute10_results.py --arm dann --fold 1 \
        --run-dir /kaggle/working/runs/dilute10/dann/fold1 --out /kaggle/working/out_dann_f1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dilute10_common as C                                  # noqa: E402

# arm -> (budget, ckpt_every, first_step, logit_subdir_fmt, steps_json, steps_key, r1)
ARMS = {
    "dann": (4000, 250, 250, "logits/dann_nb3_fold{k}", "run.json", "steps_completed", None),
    "bnm": (4000, 250, 250, "logits/bnm_nb3_fold{k}", "run.json", "steps_completed", None),
    "coral_nb3": (2000, 250, 250, "logits/g16_2000_fold{k}", "run.json", "steps_completed",
                  "checkpoints/step_2000.pt"),
    "freematch_nb3": (1500, 125, 0, "logits", "results.json", "steps",
                      "checkpoints/step_1500.pt"),
    "freematch_frozen": (8000, 250, 250, "logits", None, None, "checkpoints/step_8000.pt"),
    "coral_frozen": (8000, 250, 250, "logits", None, None, "checkpoints/step_8000.pt"),
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def link_dir(src, dst):
    try:
        os.symlink(src, dst, target_is_directory=True)
    except (OSError, NotImplementedError):
        import _winapi                                          # Windows: a junction
        _winapi.CreateJunction(src, dst)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--fold", type=int, required=True, choices=(1, 2, 3, 4))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", action="append", default=[], help="log file(s) to include")
    ap.add_argument("--max-part-bytes", type=int, default=480 * 1000 * 1000)
    ap.add_argument("--stage-json", default="/kaggle/working/stage.json",
                    help="Cell 2's stage.json; archived in the core and the analysis core")
    a = ap.parse_args(argv)
    rd, out, k = os.path.abspath(a.run_dir), os.path.abspath(a.out), a.fold
    if os.path.exists(out):
        raise SystemExit(f"[pack] refusing: {out} exists")
    budget, every, first, lsub, sj, skey, r1 = ARMS[a.arm]
    stage = os.path.abspath(a.stage_json)
    if not os.path.isfile(stage):
        raise SystemExit(f"[pack] refusing: {stage} (Cell 2's stage.json) is missing")
    stage_sha = sha256_file(stage)

    # 1. the part-2 run-time checks, again, on what is actually on disk
    C.post_run_check(arm=a.arm, out_dir=rd, budget=budget, ckpt_every=every,
                     first_step=first, logit_subdir=lsub.format(k=k), steps_json=sj,
                     steps_key=skey or "steps_completed", r1_ckpt=r1)
    print(f"[pack] part-2 checks pass: budget {budget}, logits {first}..{budget}/{every}")

    # 2. the established packager, through a uniquely named link
    import build_dann_r5_bundle as B
    tmp = tempfile.mkdtemp(prefix="d10pack_")
    name = f"dilute10_{a.arm}_fold{k}"
    link = os.path.join(tmp, name)
    link_dir(rd, link)
    try:
        B.pack_results([link], out, None, a.max_part_bytes,
                       extra_files=tuple(a.log) + (stage,))
    finally:
        try:
            os.unlink(link) if os.path.islink(link) else os.rmdir(link)
        except OSError:
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    # 3. the analysis core
    core = os.path.join(out, f"{name}_analysis.tar.gz")
    members = []
    with tarfile.open(core, "w:gz") as tf:
        for p in sorted(glob.glob(os.path.join(rd, "*.csv")) + glob.glob(os.path.join(rd, "*.json"))
                        + glob.glob(os.path.join(rd, "*.log")) + list(a.log)):
            if os.path.isfile(p):
                arc = f"{name}/{os.path.basename(p)}"
                tf.add(p, arcname=arc)
                members.append(arc)
        stage_arc = f"{name}/stage.json"
        tf.add(stage, arcname=stage_arc)
        members.append(stage_arc)
        for sub in ("logits", "scores"):
            for root, _d, files in os.walk(os.path.join(rd, sub)):
                for f in sorted(files):
                    p = os.path.join(root, f)
                    arc = f"{name}/" + os.path.relpath(p, rd).replace("\\", "/")
                    tf.add(p, arcname=arc)
                    members.append(arc)
    grid = list(range(first, budget + 1, every))
    want = [f"{name}/{lsub.format(k=k)}/step_{s}.npz" for s in grid]
    miss = [w for w in want if w not in members]
    if miss:
        raise SystemExit(f"[pack] ABORT: analysis core lacks {len(miss)} logit files, e.g. "
                         f"{miss[:4]}")
    mp = os.path.join(out, "MANIFEST.json")
    man = json.load(open(mp))
    # 4. stage.json coverage: every archive that should hold it does, byte for byte
    stage_cov = {}
    want_stage = {os.path.basename(core): stage_arc}
    for pt in man["parts"]:
        if pt["name"].endswith(".tar.gz") and pt["name"] != os.path.basename(core):
            want_stage[pt["name"]] = "meta/stage.json"
    for tname, arc in want_stage.items():
        with tarfile.open(os.path.join(out, tname)) as tf:
            try:
                got = hashlib.sha256(tf.extractfile(arc).read()).hexdigest()
            except KeyError:
                got = None
        stage_cov[tname] = {"member": arc, "matches_source": got == stage_sha}
    if len(want_stage) < 2 or not all(v["matches_source"] for v in stage_cov.values()):
        raise SystemExit(f"[pack] ABORT: stage.json coverage failed: {stage_cov}")
    man["bundle"] = "dilute10_results"
    man["arm"], man["fold"] = a.arm, k
    man["analysis_core"] = {"name": os.path.basename(core), "bytes": os.path.getsize(core),
                            "sha256": sha256_file(core), "n_members": len(members),
                            "logit_grid": [first, budget, every],
                            "coverage": f"{len(want)}/{len(want)} grid logits present"}
    man["stage_json"] = {"sha256": stage_sha, "archives": stage_cov}
    with open(mp, "w") as fh:
        json.dump(man, fh, indent=2)
    print(f"[pack] analysis core {os.path.basename(core)}: {len(members)} files, "
          f"{os.path.getsize(core) / 1e6:.1f} MB; all {len(want)} grid logits present")
    print(f"[pack] stage.json {stage_sha[:16]}... archived in {sorted(stage_cov)}")
    print(f"[pack] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
