r"""
make_rewrite_v2_macros.py -- macros for the values rewrite v2 adds to the text.
===============================================================================

    python paper/scripts/make_rewrite_v2_macros.py

Every value is read from a read-only source and asserted where it is read:

  * ``PROJECTv7.md`` -- the deflector pool (19,167), the arc-capable subset (~4,182) and the
    reuse limit ("up to 10"); ``crosscheck.py`` re-reads them independently;
  * ``audit/review_r1/out/F5_host_properties.json`` (through make_review_macros.review, which
    re-checks every recorded input hash) -- the distinct deflectors of the training positives;
  * the arms' stored run configs -- batch size and optimiser settings. The grouping the text
    states is asserted: CORAL, DANN, BNM and lambda=0 share one setting; unfrozen FreeMatch
    (folds 1-4) keeps its frozen-comparison run's head learning rate and weight decay;
    Baseline A is read from its own config;
  * ``audit/curation_fold1/out/fold0_only.json`` -- FreeMatch's fold-0 development curve on the
    cleaned pool (the peak-location values of Table 11 row 1 and Section 7). Its recorded input
    hashes are re-checked before any value is read; ``crosscheck.py`` recomputes the values
    from the stored logits;
  * the budget paragraph's list of arms fixed before their development curve existed, checked
    against the record lines that state it.

Writes ``paper/tables/M_rewrite2.tex`` and ``paper/data/rewrite2_macros.json`` and nothing
else. Trains nothing, loads no model.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re

import _common as C
import make_review_macros as RV

MACROS: list = []  # (name, printed, value, source)


def m(name, printed, value, source):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    MACROS.append((name, printed, value, source))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def thousands(n: int) -> str:
    """The paper's number style (rewrite v5b, item 12): four-digit numbers take no separator;
    five or more digits are grouped with a thin space."""
    return str(n) if abs(n) < 10000 else f"{n:,}".replace(",", r"\,")


def sci(x: float) -> str:
    e = round(__import__("math").log10(x))
    assert abs(x - 10.0 ** e) < 1e-12, x
    return f"10^{{{e}}}"


WORDS = {10: "ten"}


# The lab log is read through scripts/record_extract.py (the log, or without it the extract),
# which holds the patterns below as rw.* values and checks.
import record_extract as RX  # noqa: E402


def deflectors():
    # the one log line naming the pool and its arc-capable subset (rw.defl_pool / rw.defl_arc)
    (pool, i), (arc, i2) = RX.value("rw.defl_pool"), RX.value("rw.defl_arc")
    assert i == i2
    pool, arc = int(pool.replace(",", "")), int(arc.replace(",", ""))
    assert RX.check("rw.arc_subset_theta_e_075"), "the arc subset is no longer the theta_E >= 0.75 one"
    m("rwDeflPool", thousands(pool), pool, f"PROJECTv7.md:{i}")
    m("rwDeflArc", thousands(arc), arc, f"PROJECTv7.md:{i}")
    n, j = RX.value("rw.defl_reuse")
    n = int(n)
    m("rwDeflReuse", WORDS[n], n, f"PROJECTv7.md:{j}")
    f5 = RV.review(RV.R1, "F5_host_properties.json")
    s = f5["populations"]["a"]
    n_pos, n_def = (int(x) for x in re.search(
        r"(\d+) source-train simulated positives \(one host per positive; (\d+) distinct", s).groups())
    assert n_pos == 24000, n_pos
    m("rwDeflDistinct", thousands(n_def), n_def,
      "audit/review_r1/out/F5_host_properties.json -> result.populations.a")


def optimiser():
    da = {"CORAL": sorted(glob.glob(C.R("kaggle", "coral_r4", "runs", "main", "extracted",
                                         "coral_r4_confirm", "fold*", "run.json"))),
          "DANN": sorted(glob.glob(C.R("runs", "dann_main", "fold*", "extracted", "fold*",
                                       "run_config.json"))),
          "BNM": sorted(glob.glob(C.R("runs", "bnm_r6_folds", "bnm_r6_fold*", "run_config.json"))),
          "lambda0": sorted(glob.glob(C.R("runs", "baseline_unfrozen", "fold*", "extracted",
                                          "lambda0_r6_fold*", "run_config.json")))}
    fm = [C.R("runs", f"freematch_nb3_unfrozen_fold{f}", "run_config.json") for f in (1, 2, 3, 4)]
    fm_frozen = C.R("runs", "bakeoff_freematch_5fold", "run_config.json")
    base = C.R("runs", "bakeoff_baseline_A", "run_config.json")
    seen = set()
    for arm, files in da.items():
        assert len(files) == 4, (arm, files)
        for p in files:
            c = json.load(open(p, encoding="utf-8"))
            assert c["n_blocks"] == 3 and c["batch_size"] == 256, p
            seen.add((c["lr"], c["lr_decay"], c["weight_decay"]))
    assert len(seen) == 1, f"the four unfrozen non-FreeMatch arms do not share one setting: {seen}"
    lr, decay, wd = seen.pop()
    srcs = "run configs of CORAL (kaggle/coral_r4/.../coral_r4_confirm/fold1-4/run.json), DANN, BNM, lambda=0 folds 1-4"
    m("rwOptLr", sci(lr), lr, srcs)
    m("rwOptDecay", f"{decay:g}", decay, srcs)
    m("rwOptWd", f"{wd:g}", wd, srcs)
    fz = json.load(open(fm_frozen, encoding="utf-8"))
    fm_set = set()
    for p in fm:
        c = json.load(open(p, encoding="utf-8"))
        assert c["n_blocks"] == 3 and c["batch_size"] == 256 and c["max_steps"] == 1500, p
        fm_set.add((c["lr"], c["backbone_lr"], c["weight_decay"]))
    assert len(fm_set) == 1, fm_set
    h_lr, e_lr, f_wd = fm_set.pop()
    assert (h_lr, f_wd) == (fz["lr"], fz["weight_decay"]) and fz["batch_size"] == 256, \
        "unfrozen FreeMatch no longer keeps its frozen-comparison head LR and weight decay"
    assert (h_lr, e_lr, f_wd) != (lr, lr, wd), "FreeMatch is not grouped apart from the others"
    s2 = "runs/freematch_nb3_unfrozen_fold1-4/run_config.json (= runs/bakeoff_freematch_5fold/run_config.json lr, weight_decay)"
    m("rwFmHeadLr", sci(h_lr), h_lr, s2)
    m("rwFmEncLr", sci(e_lr), e_lr, s2)
    m("rwFmWd", sci(f_wd), f_wd, s2)
    b = json.load(open(base, encoding="utf-8"))
    assert b["batch_size"] == 256
    m("rwBaseLr", sci(b["lr"]), b["lr"], "runs/bakeoff_baseline_A/run_config.json -> lr")
    m("rwBaseWd", sci(b["weight_decay"]), b["weight_decay"],
      "runs/bakeoff_baseline_A/run_config.json -> weight_decay")
    m("rwBatch", "256", 256, "every run config above -> batch_size")


FOLD0 = C.R("audit", "curation_fold1", "out", "fold0_only.json")


def fold0_only():
    d = json.load(open(FOLD0, encoding="utf-8"))
    inp = d["inputs"]
    assert sha(C.R(*inp["curve_csv"]["path"].split("/"))) == inp["curve_csv"]["sha256"], \
        "fold-0 curve.csv changed since fold0_only.json was computed"
    assert sha(C.R(*inp["run_dir"].split("/"), "logits", "ids.npz")) == inp["logits"]["ids_npz_sha256"]
    n_files = len(glob.glob(C.R(*inp["run_dir"].split("/"), "logits", "step_*.npz")))
    assert n_files == inp["logits"]["n_files"] == 65
    v = d["D1"]
    steps = v["a_within_two_positives_steps"]
    w = v["b_within_one_sd"]["ddof0"]
    assert w["n"] == v["b_within_one_sd"]["ddof1"]["n"], "the one-sd count depends on ddof"
    src = "audit/curation_fold1/out/fold0_only.json -> D1"
    m("rwFzSteps", ", ".join(str(s) for s in steps), None, src + ".a_within_two_positives_steps")
    m("rwFzStepMin", str(min(steps)), min(steps), src + ".a_within_two_positives_steps")
    m("rwFzStepMax", str(max(steps)), max(steps), src + ".a_within_two_positives_steps")
    m("rwFzWithinSd", str(w["n"]), w["n"], src + ".b_within_one_sd")
    m("rwFzPoints", str(w["of"]), w["of"], src + ".b_within_one_sd.of")
    assert w["of"] == v["n_points"]


def budget_list():
    # Baseline A: R1 fixed before round-1 results (section 7.4); BNM: the rule and its ceiling
    # were declared before the development curve existed (section 7.20). Both lines are tested
    # by record_extract.check("rw.budget_list_lines").
    assert RX.check("rw.budget_list_lines"), \
        "Baseline A's frozen-before-results line or BNM's before-the-curve line moved"
    m("rwBudgetPreCurve", "BNM's rule and Baseline A's budget", None,
      "PROJECTv7.md section 7.4 (R1 pre-registered and FROZEN before results); section 7.20 "
      "(ceiling declared before the curve exists)")


def main() -> int:
    deflectors()
    optimiser()
    fold0_only()
    budget_list()
    lines = [rf"\newcommand{{\{n}}}{{{p}}}" for n, p, _v, _s in MACROS]
    C.write_table(os.path.join(C.TABLES, "M_rewrite2.tex"), "\n".join(lines),
                  header_comment="Macros added by rewrite v2 (make_rewrite_v2_macros.py); "
                                 "sources in paper/data/rewrite2_macros.json.")
    with open(os.path.join(C.DATA, "rewrite2_macros.json"), "w", encoding="utf-8") as fh:
        json.dump([dict(name=n, printed=p, value=v, source=s) for n, p, v, s in MACROS], fh,
                  indent=1)
    print(f"  wrote tables/M_rewrite2.tex ({len(MACROS)} macros)")
    return 0


def build():
    MACROS.clear()
    deflectors()
    optimiser()
    fold0_only()
    budget_list()
    return list(MACROS)


if __name__ == "__main__":
    raise SystemExit(main())
