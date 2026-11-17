r"""
make_code_review_fixes.py -- the numbers the code-review fixes add (PROJECTv7 7.66).
====================================================================================

    python paper/scripts/make_code_review_fixes.py

Writes ``paper/tables/M_codereview.tex`` (macros) and nothing else. Every value is read from
the code that ran (its constants, parsed from the source) or from the stored run records;
nothing is trained or typed. ``crosscheck.build_code_review`` recomputes each one by a
separate path.

  * the augmentation of Section 3.2, from ``scripts/lens_dataset.py``'s ``LensTransform``
    defaults (one version of that file ran in every arm) and the 64-pixel native size;
  * CORAL's covariance shrinkage, from every CORAL run record (balanced and 1:10);
  * FreeMatch's threshold momentum, fairness weight, consistency warm-up and step offset, from
    every unfrozen FreeMatch run record;
  * BNM's nuclear-norm divisor, min(B, C), from the batch size and the head's two classes;
  * the DANN discriminator's learning rate and weight decay, from every DANN-path run record;
  * the half-precision skipped updates of the balanced runs, from their checkpoint logs;
  * the checkpoint spacing behind Table 12's window, from the stored logits;
  * CORAL's main-design gate band, from ``run_coral_r4_confirm.step0_gate``;
  * FreeMatch fold 1's missing gate keys, and the 1:10 helper scripts whose recorded digests
    no shipped file matches.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import sys

import _common as C

FOLDS = (1, 2, 3, 4)
NATIVE_PX = 64                      # the cutouts' native size (Section 3.1), asserted below
BATCH = 256
CORAL_BAL = "kaggle/coral_r4/runs/main/extracted/coral_r4_confirm/fold{}"
L0_DIR = "runs/baseline_unfrozen/fold{0}/extracted/lambda0_r6_fold{0}"
DANN_DIR = "runs/dann_main/fold{0}/extracted/fold{0}"
FM_BAL = "runs/freematch_nb3_unfrozen_fold{}"
DL = "runs/dilute10/{}/fold{}"
WORD = {2: "two", 3: "three", 4: "four"}


def J(rel):
    return json.load(open(C.R(rel), encoding="utf-8"))


def src(rel):
    return open(C.R(rel), encoding="utf-8").read()


def init_defaults(path, cls):
    """{arg: default} of ``cls.__init__`` in a source file, by AST (nothing is imported)."""
    tree = ast.parse(src(path))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    a = init.args
    pos = a.args[len(a.args) - len(a.defaults):]
    out = {p.arg: ast.literal_eval(d) for p, d in zip(pos, a.defaults)}
    out.update({p.arg: ast.literal_eval(d) for p, d in zip(a.kwonlyargs, a.kw_defaults)
                if d is not None})
    return out


def num(x):
    """A value as the paper prints it: no trailing zeros."""
    return f"{x:g}"


def last_skipped(rel):
    rows = list(csv.DictReader(open(C.R(rel, "checkpoints.csv"), encoding="utf-8")))
    return int(float(rows[-1]["skipped_cumulative"])), int(rows[-1]["step"])


def main() -> int:
    out = []
    add = lambda n, v: out.append((n, v))

    # ---- augmentation (Section 3.2) ----
    lt = init_defaults("scripts/lens_dataset.py", "LensTransform")
    s = src("scripts/lens_dataset.py")
    assert "v2.RandomHorizontalFlip(p=0.5)" in s and "v2.RandomVerticalFlip(p=0.5)" in s
    assert "fill=0.0" in s and "v2.RandomErasing(p=0.5" in s
    assert re.search(r'SIM_LENS_KEY = "lenses"\s+# sim shards: \(N,64,64\)', s)
    tx = lt["translate"][0] * NATIVE_PX                  # torchvision: int(round(U(-tx, tx)))
    add("augRotDeg", num(lt["degrees"]))
    add("augShiftPx", str(int(round(tx))))
    add("augZoomLo", num(lt["strong_scale"][0]))
    add("augZoomHi", num(lt["strong_scale"][1]))
    add("augEraseLoPct", num(100 * lt["erase_scale"][0]))
    add("augEraseHiPct", num(100 * lt["erase_scale"][1]))

    # ---- CORAL: shrinkage in every run record ----
    shr = {J(CORAL_BAL.format(k) + "/run.json")["coral_shrinkage"] for k in FOLDS}
    shr |= {J(DL.format("coral_nb3", k) + "/run.json")["coral_shrinkage"] for k in FOLDS}
    shr |= {J(DL.format("coral_frozen", k) + "/run_config.json")["coral_shrinkage"] for k in FOLDS}
    shr |= {J("runs/bakeoff_coral/run_config.json")["coral_shrinkage"]}
    assert len(shr) == 1, shr
    add("coralShrinkPct", num(100 * shr.pop()))

    # ---- FreeMatch: every unfrozen run record, balanced and 1:10 ----
    fm = [J(FM_BAL.format(k) + "/run_config.json") for k in FOLDS]
    fm += [J(DL.format("freematch_nb3", k) + "/run_config.json") for k in FOLDS]
    one = lambda key: {c[key] for c in fm}
    assert one("fairness_target") == {None}
    assert one("mu_step_offset") == one("schedule_total_steps") == {8000}
    for key in ("sat_ema_momentum", "saf_weight", "consistency_warmup_frac"):
        assert len(one(key)) == 1, key
    add("fmSatMomentum", num(one("sat_ema_momentum").pop()))
    add("fmSafWeight", num(one("saf_weight").pop()))
    add("fmWarmupPct", num(100 * one("consistency_warmup_frac").pop()))

    # ---- BNM: nuclear norm / min(B, C) ----
    d = src("scripts/da_losses.py")
    assert "denom = float(min(prob2.shape[0], prob2.shape[1]))" in d
    bnm = {J(f"runs/bnm_r6_folds/bnm_r6_fold{k}/run_config.json")["batch_size"] for k in FOLDS}
    assert bnm == {BATCH}, bnm
    add("bnmNucDenom", str(min(BATCH, 2)))

    # ---- DANN discriminator: the head's AdamW settings, no lr_decay ----
    assert "d_lr=LR, weight_decay=WEIGHT_DECAY)" in src("scripts/run_dann_r5.py")
    recs = [J(DANN_DIR.format(k) + "/run_config.json") for k in FOLDS]
    recs += [J(L0_DIR.format(k) + "/run_config.json") for k in FOLDS]
    recs += [J(DL.format("dann", k) + "/run_config.json") for k in FOLDS]
    lr = {r["lr"] for r in recs}
    wd = {r["weight_decay"] for r in recs}
    assert len(lr) == len(wd) == 1 and {r["dann_lambda_horizon"] for r in recs} == {2000}
    e = int(round(math.log10(lr.pop())))
    add("dannDiscLr", rf"10^{{{e}}}")
    add("dannDiscWd", num(wd.pop()))

    # ---- skipped half-precision updates, balanced runs ----
    sk = {}
    for arm, pat, steps in (("Lam", L0_DIR, 4000), ("Dann", DANN_DIR, 4000),
                            ("Coral", CORAL_BAL, 2000)):
        sk[arm] = []
        for k in FOLDS:
            n, last = last_skipped(pat.format(k))
            assert last == steps
            sk[arm].append(n)
        add(f"skip{arm}List", ", ".join(map(str, sk[arm])))
    fmsk = {J(FM_BAL.format(k) + "/results.json")["freematch_stability"]["instrument"]
            ["steps_skipped_fp16"] for k in FOLDS}
    assert len(fmsk) == 1, fmsk
    add("skipFmEach", str(fmsk.pop()))
    add("lamSkipFoldTwo", str(sk["Lam"][1]))
    add("dannSkipFoldTwo", str(sk["Dann"][1]))

    # ---- checkpoint spacing behind the window average ----
    gaps = {}
    for arm, spec in C.ARMS.items():
        if arm.startswith("Baseline"):
            continue
        for k in FOLDS:
            st = sorted(spec["load"](k))
            gaps.setdefault(arm, set()).update(b - a for a, b in zip(st, st[1:]))
    rows = list(csv.DictReader(open(C.R("runs", "bakeoff_baseline_A", "curve.csv"), encoding="utf-8")))
    bs = [int(r["step"]) for r in rows]
    gaps["Baseline A"] = {b - a for a, b in zip(bs, bs[1:])}
    assert all(len(g) == 1 for g in gaps.values()), gaps
    fmg = gaps.pop("FreeMatch x nb3").pop()
    other = {g.pop() for g in gaps.values()}
    assert len(other) == 1, other
    other = other.pop()
    add("ckptFm", str(fmg))
    add("ckptOther", str(other))
    add("winEightSpanFm", str(8 * fmg))
    add("winEightSpanOther", str(8 * other))

    # ---- CORAL's main-design gate band ----
    m = re.findall(r"def step0_gate\(scorer, record, fold, n_pos, tol_abort=(0\.\d+), "
                   r"auroc_tol_abort=(0\.\d+)\):", src("scripts/run_coral_r4_confirm.py"))
    assert len(m) == 1
    assert re.search(r"^STEP0_AUROC_TOL = 1e-3 ", src("scripts/run_coral_round4.py"), re.M)
    add("coralGateAurocAbort", m[0][1])
    add("coralGateAurocWarn", r"10^{-3}")

    # ---- FreeMatch fold 1: the gate keys today's runner always writes ----
    ks = [set(J(FM_BAL.format(k) + "/run_config.json")) for k in FOLDS]
    assert ks[1] == ks[2] == ks[3] and ks[0] < ks[1]          # fold 1 lacks keys, has none extra
    miss1 = sorted(ks[1] - ks[0])
    r = src("scripts/run_unfreeze_freematch.py")
    assert all(f'"{x}":' in r for x in miss1)
    assert J(FM_BAL.format(1) + "/run_config.json")["step0_gate"]["tolerance"] == 0.0
    add("fmFoldOneMissingWord", WORD[len(miss1)])

    # ---- the 1:10 helper scripts whose recorded digests no shipped file matches ----
    lost = {}
    for arm in ("dann", "bnm", "coral_frozen", "freematch_frozen", "coral_nb3", "freematch_nb3"):
        for k in FOLDS:
            f = DL.format(arm, k) + "/run_config.json"
            if not os.path.isfile(C.R(f)):
                f = DL.format(arm, k) + "/run.json"
            dig = J(f)["dilute10"]["imported_module_sha256"]
            for name, pre in dig.items():
                h = hashlib.sha256(open(C.R("scripts", name), "rb").read()).hexdigest()
                if not h.startswith(pre[:16]):
                    lost.setdefault(arm, set()).add(name)
    assert set(lost) == {"dann", "bnm", "coral_frozen", "freematch_frozen"}, lost
    names = {frozenset(n - {f"run_dilute10_{a}.py"}) for a, n in lost.items()}
    assert names == {frozenset({"dilute10_common.py", "dilute10_batch_log.py"})}, names
    assert all(len(n) == 3 for n in lost.values())
    add("dlLostArmsWord", WORD[len(lost)])
    add("dlLostFilesWord", WORD[3])

    C.write_table(os.path.join(C.TABLES, "M_codereview.tex"),
                  "\n".join(rf"\newcommand{{\{n}}}{{{v}}}" for n, v in out),
                  header_comment=("The code-review fixes' numbers (PROJECTv7 7.66), read from the "
                                  "code that ran and the run records by\n"
                                  "paper/scripts/make_code_review_fixes.py; "
                                  "crosscheck.build_code_review recomputes each."), size="")
    for n, v in out:
        print(f"  \\{n} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
