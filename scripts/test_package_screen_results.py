"""
test_package_screen_results.py
==============================
Tests for the Task-5 split-archive packaging.

The properties that matter, in the order they would hurt:
  1. an INCOMPLETE run is refused (this is the Task-4 failure, and the whole point)
  2. every checkpoint reassembles byte-identically out of the parts
  3. each part is independently downloadable, verifiable and under the size cap
  4. a corrupt or missing part is identified BY NAME so only it is re-fetched
  5. a declared §1 reference that does not match nb0's emitted config aborts

Synthetic run dirs with small random "checkpoints" and a small part cap, so the split /
manifest / verify logic is exercised in seconds instead of on 1.8 GB.

    python scripts/test_package_screen_results.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import contextlib

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import package_screen_results as pk                          # noqa: E402

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def make_run(root, name, n_ckpt=32, ckpt_bytes=100_000, budget=8000, every=250,
             keep_only_last=False, cfg_extra=None):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(d, "scores"), exist_ok=True)
    steps = [s for s in range(every, budget + 1, every)][:n_ckpt]
    if keep_only_last:
        steps = steps[-1:]
    rng = os.urandom
    for s in steps:
        with open(os.path.join(d, "checkpoints", f"step_{s}.pt"), "wb") as fh:
            fh.write(rng(ckpt_bytes))
    with open(os.path.join(d, "curve.csv"), "w", newline="") as fh:
        fh.write("step,fold0_tpr_1e-3\n")
        for s in range(every, budget + 1, every):
            fh.write(f"{s},0.1\n")
    cfg = {"budget_steps": budget, "curve_every": every, "lr": 1e-4, "weight_decay": 0.05,
           "backbone_lr": None, "lr_decay": 0.75, "n_blocks": 0, "precision": "fp16",
           "augmentation": "weak (+-3px translate, PROJECT.md 7.3)", "seed": 42,
           "batch_size": 256, "pos_fraction": 0.5, "head_type": "linear", "fold": 0,
           "true_batch": True, "gradient_accumulation": False, "grad_checkpointing": False}
    cfg.update(cfg_extra or {})
    json.dump(cfg, open(os.path.join(d, "run_config.json"), "w"), indent=2)
    json.dump({"tpr": 0.2}, open(os.path.join(d, "results.json"), "w"))
    with open(os.path.join(d, "scores", f"{name}_fold0.npz"), "wb") as fh:
        fh.write(b"\x00" * 2048)
    return d


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = pk.main(argv)
    return rc, buf.getvalue()


def test_complete_run_packages_and_verifies():
    print("\n[1] a complete run packages, splits, and reassembles")
    tmp = tempfile.mkdtemp()
    try:
        runs = os.path.join(tmp, "runs")
        os.makedirs(runs)
        make_run(runs, "baselineA_fold0_nb3", n_ckpt=32, ckpt_bytes=100_000)
        make_run(runs, "baselineA_fold0_nb0_image", n_ckpt=32, ckpt_bytes=1_000)
        out = os.path.join(tmp, "screen_out")
        cap = 900_000                                    # forces several parts
        rc, log = _run(["--runs-dir", runs, "--out", out, "--max-part-bytes", str(cap),
                        "--small-ckpt-limit", "100000"])
        check(rc == 0, f"exit 0, got {rc}")

        man = json.load(open(os.path.join(out, "MANIFEST.json")))
        check(man["reassembly_verified"] is True, "manifest records reassembly verified")

        ck_parts = [p for p in man["parts"] if p["name"].startswith("ckpt_")]
        core = [p for p in man["parts"] if p["name"] == "screen_core.tar.gz"]
        check(len(core) == 1, "exactly one core archive")
        check(len(ck_parts) >= 3, f"nb3 split into several parts, got {len(ck_parts)}")
        check(all(p["bytes"] <= cap * 1.05 for p in ck_parts),
              f"every part is within the cap; max={max(p['bytes'] for p in ck_parts)}")

        # every nb3 checkpoint appears exactly once across the parts
        steps = [s for p in ck_parts for s in p["steps"]]
        check(sorted(steps) == list(range(250, 8001, 250)),
              f"all 32 nb3 steps present exactly once, got {len(steps)}")

        # nb0's tiny checkpoints rode in the core archive, not their own part
        check(not any("nb0" in p["name"] for p in ck_parts),
              "nb0's 32 tiny checkpoints are in the core archive, not separate parts")
        import tarfile
        with tarfile.open(os.path.join(out, "screen_core.tar.gz")) as tf:
            names = tf.getnames()
        n_nb0 = sum(1 for n in names if "nb0_image/checkpoints/step_" in n)
        check(n_nb0 == 32, f"core archive carries all 32 nb0 checkpoints, got {n_nb0}")
        check(any(n.endswith("curve.csv") for n in names), "core carries curve.csv")
        check(any("/scores/" in n for n in names), "core carries the score dumps")

        # and the local --verify pass agrees
        rc2, log2 = _run(["--verify", out])
        check(rc2 == 0, "local --verify passes on the freshly built set")
        check("ALL PARTS PRESENT AND INTACT" in log2, "verify reports the intact verdict")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_incomplete_run_is_refused():
    print("\n[2] an INCOMPLETE run is REFUSED (the Task-4 failure)")
    tmp = tempfile.mkdtemp()
    try:
        runs = os.path.join(tmp, "runs")
        os.makedirs(runs)
        make_run(runs, "baselineA_fold0_nb3", keep_only_last=True, ckpt_bytes=1000)
        out = os.path.join(tmp, "screen_out")
        rc, log = _run(["--runs-dir", runs, "--out", out])
        check(rc == 1, f"exits non-zero, got {rc}")
        check("ABORT" in log and "Task-4" in log,
              "the abort explains it would repeat the Task-4 failure")
        check(not os.path.exists(os.path.join(out, "MANIFEST.json")),
              "nothing was packaged")

        # ...and the escape hatch exists but must be asked for explicitly
        rc2, _ = _run(["--runs-dir", runs, "--out", out, "--allow-incomplete"])
        check(rc2 == 0, "--allow-incomplete permits it deliberately")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corrupt_and_missing_parts_are_named():
    print("\n[3] a corrupt or missing part is identified BY NAME")
    tmp = tempfile.mkdtemp()
    try:
        runs = os.path.join(tmp, "runs")
        os.makedirs(runs)
        make_run(runs, "baselineA_fold0_nb3", n_ckpt=32, ckpt_bytes=100_000)
        out = os.path.join(tmp, "screen_out")
        _run(["--runs-dir", runs, "--out", out, "--max-part-bytes", "900000",
              "--small-ckpt-limit", "1000"])
        man = json.load(open(os.path.join(out, "MANIFEST.json")))
        victim = [p for p in man["parts"] if p["name"].startswith("ckpt_")][1]["name"]

        # corrupt one byte in the middle -- same size, different content
        p = os.path.join(out, victim)
        with open(p, "r+b") as fh:
            fh.seek(os.path.getsize(p) // 2)
            b = fh.read(1)
            fh.seek(os.path.getsize(p) // 2)
            fh.write(bytes([b[0] ^ 0xFF]))
        rc, log = _run(["--verify", out])
        check(rc == 1, "verify fails on a corrupted part")
        check("CORRUPT" in log and victim in log, f"names the corrupt part ({victim})")
        check("Re-download ONLY these" in log, "tells you to re-fetch just that part")

        # now delete it entirely
        os.remove(p)
        rc, log = _run(["--verify", out])
        check("MISSING" in log and victim in log, "names the missing part")
        # the other parts still verify -- that is the point of independent parts
        others = [x["name"] for x in man["parts"] if x["name"] != victim]
        check(all(f"ok       {n}" in log for n in others),
              "every other part still verifies independently")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_truncated_part_detected():
    print("\n[4] a truncated (partially downloaded) part is detected")
    tmp = tempfile.mkdtemp()
    try:
        runs = os.path.join(tmp, "runs")
        os.makedirs(runs)
        make_run(runs, "baselineA_fold0_nb3", n_ckpt=32, ckpt_bytes=100_000)
        out = os.path.join(tmp, "screen_out")
        _run(["--runs-dir", runs, "--out", out, "--max-part-bytes", "900000",
              "--small-ckpt-limit", "1000"])
        man = json.load(open(os.path.join(out, "MANIFEST.json")))
        victim = [p for p in man["parts"] if p["name"].startswith("ckpt_")][0]["name"]
        p = os.path.join(out, victim)
        with open(p, "r+b") as fh:
            fh.truncate(os.path.getsize(p) // 2)
        rc, log = _run(["--verify", out])
        check(rc == 1 and "TRUNCATED" in log and victim in log,
              "a half-downloaded part is caught by size before hashing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_declared_reference_mismatch_aborts():
    print("\n[5] a declared §1 reference that nb0 did not produce ABORTS")
    tmp = tempfile.mkdtemp()
    try:
        runs = os.path.join(tmp, "runs")
        os.makedirs(runs)
        nb0 = make_run(runs, "baselineA_fold0_nb0_image", n_ckpt=32, ckpt_bytes=1000)
        out = os.path.join(tmp, "screen_out")

        # matching declaration -> packages fine
        good = os.path.join(tmp, "ref_good.json")
        shutil.copy2(os.path.join(nb0, "run_config.json"), good)
        rc, log = _run(["--runs-dir", runs, "--out", out, "--declared-reference", good])
        check(rc == 0 and "matches nb0's emitted config" in log,
              "a faithful declared reference passes the post-hoc check")

        # tampered declaration -> abort, naming the key
        bad = os.path.join(tmp, "ref_bad.json")
        cfg = json.load(open(good))
        cfg["weight_decay"] = 1e-4                    # the pre-protocol-match value
        json.dump(cfg, open(bad, "w"))
        out2 = os.path.join(tmp, "screen_out2")
        rc2, log2 = _run(["--runs-dir", runs, "--out", out2,
                          "--declared-reference", bad])
        check(rc2 == 1, f"exits non-zero on a mismatched reference, got {rc2}")
        check("weight_decay" in log2, "names the offending pinned key")
        check("NOT matched" in log2, "says plainly that the pair is not matched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_part_planning():
    print("\n[6] part planning respects the cap and keeps files whole")
    files = [(f"a/{i}", f"/tmp/{i}", 100) for i in range(10)]
    parts = pk.plan_parts(files, 250)
    check(all(sum(f[2] for f in p) <= 250 for p in parts),
          f"no part exceeds the cap: {[sum(f[2] for f in p) for p in parts]}")
    check(sum(len(p) for p in parts) == 10, "every file is placed exactly once")
    # a single oversized file gets its own part rather than being split
    parts2 = pk.plan_parts([("big", "/tmp/big", 999)], 250)
    check(len(parts2) == 1 and len(parts2[0]) == 1,
          "an oversized single file becomes its own part (never split)")


def main():
    print("=" * 74)
    print("test_package_screen_results.py")
    print("=" * 74)
    test_complete_run_packages_and_verifies()
    test_incomplete_run_is_refused()
    test_corrupt_and_missing_parts_are_named()
    test_truncated_part_detected()
    test_declared_reference_mismatch_aborts()
    test_part_planning()
    print("\n" + "=" * 74)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED")
        for m in FAILED:
            print("  -", m)
        return 1
    print("ALL PACKAGING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
