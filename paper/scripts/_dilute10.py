r"""
_dilute10.py -- the artifacts of the 1:10 diluted-target experiment, read in one place.
======================================================================================

Used by the paper-v6 generators (make_dilute10_macros.py, make_dilute10_tables.py,
fig_f1_setup_v6.py, fig_f3_forest_v6.py, fig_f7_dilute10.py). ``crosscheck.py`` does NOT use
this module: it re-reads every artifact itself, so a slip here cannot also hide there.

Every statistic comes from two analysis files, written and enrolled before the paper was:

  * ``runs/dilute10_analysis/results.json``      (PROJECTv7 7.53; plan 68367e25...)
  * ``runs/dann_lambda0_holdout_gap/results.json`` (PROJECTv7 7.54; plan 9dee7d6a...)

The rest are descriptive run facts read from the run files themselves: the pool definition
(``data/splits/dilute10_pool.json``), the head norms (``runs/dilute10_head_norms/``), CORAL
x nb3's ``steps.csv`` and advisory log, and each run's venue fields. Opens no checkpoint.
"""
from __future__ import annotations

import csv
import json
import os

import _common as C

FOLDS = (1, 2, 3, 4)
RESULTS = C.R("runs", "dilute10_analysis", "results.json")
GAP = C.R("runs", "dann_lambda0_holdout_gap", "results.json")
HEADS = C.R("runs", "dilute10_head_norms", "head_norms.json")
POOL = C.R("data", "splits", "dilute10_pool.json")
PART1 = C.R("runs", "dilute10_prereg_part1.json")
PART2 = C.R("runs", "dilute10_prereg_part2.json")
PART3 = C.R("runs", "dilute10_prereg_part3.json")
PLAN = C.R("runs", "dilute10_analysis", "plan.json")
GAP_PLAN = C.R("runs", "dann_lambda0_holdout_gap", "plan.json")
RUNS = C.R("runs", "dilute10")

# The contrast set of the 1:10 experiment and its mark. It lives here, not beside
# CONTRAST_ROWS in _common.py: runs/dilute10_analysis/plan.json hashes _common.py as an input of
# the analysis, so editing it would stop scripts/analyze_dilute10.py from re-running against its
# own plan. Source: runs/dilute10_prereg_part1.json ("contrasts"): G = diluted - lambda=0
# (primary; matched-step sensitivity forms for CORAL and FreeMatch) and D = balanced - diluted
# (secondary, no verdict). crosscheck.py asserts this list equals results.json's G and D keys.
# (kind, arm, diluted readout, comparator readout, role)
DILUTE10 = "dilute10"
DILUTE10_CONTRASTS = [
    ("G", "dann", 4000, 4000, "primary"),
    ("G", "bnm", 4000, 4000, "primary"),
    ("G", "coral", 2000, 4000, "primary"),
    ("G", "coral", 2000, 2000, "sensitivity"),
    ("G", "freematch", 1500, 4000, "primary"),
    ("G", "freematch", 1500, 1500, "sensitivity"),
    ("D", "dann", 4000, 4000, "secondary"),
    ("D", "bnm", 4000, 4000, "secondary"),
    ("D", "coral", 2000, 2000, "secondary"),
    ("D", "freematch", 1500, 1500, "secondary"),
]
# Every dilution contrast carries the DILUTE10 mark (a section sign in Table 2 and Figure 3):
# pre-registered, in the dilution experiment's own three-part declaration.
DILUTE10_PREREG = {i: DILUTE10 for i in range(len(DILUTE10_CONTRASTS))}

# paper key -> results.json arm name; the order of the text (DANN, BNM, CORAL, FreeMatch)
ARMS = [("dann", "DANN x nb3", "DANN", "Dann"), ("bnm", "BNM x nb3", "BNM", "Bnm"),
        ("coral", "CORAL x nb3", "CORAL", "Coral"), ("freematch", "FreeMatch x nb3", "FreeMatch", "Fm")]
NAME = {k: n for k, n, _l, _m in ARMS}
LABEL = {k: l for k, _n, l, _m in ARMS}
MAC = {k: m for k, _n, _l, m in ARMS}
# the arms with a matched-step (sensitivity) G form
MATCHED_G = ("coral", "freematch")
# where each balanced arm and its lambda=0 comparator are read (part 1 4.1)
READ = {"dann": (4000, 4000), "bnm": (4000, 4000), "coral": (2000, 4000),
        "freematch": (1500, 4000)}
READ_MATCHED = {"coral": (2000, 2000), "freematch": (1500, 1500)}


def _j(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


_CACHE: dict = {}


def load(p):
    if p not in _CACHE:
        _CACHE[p] = _j(p)
    return _CACHE[p]


def results():
    return load(RESULTS)


def gap():
    return load(GAP)


def heads():
    return load(HEADS)


def pool():
    return load(POOL)


def G(arm, form="primary"):
    return results()["G"][f"{NAME[arm]}|{form}"]


def D(arm):
    return results()["D"][NAME[arm]]


def bal_minus_l0(arm, form="primary"):
    return results()["balanced_minus_lambda0"][form][NAME[arm]]


def per_arm(arm, metric="tpr"):
    return results()["per_arm"][NAME[arm]]["metrics"][metric]


def read_at(arm):
    return int(results()["per_arm"][NAME[arm]]["read_at"])


def verdict(arm):
    return results()["verdicts"][NAME[arm]]


def steps_csv(fold):
    with open(os.path.join(RUNS, "coral_nb3", f"fold{fold}", "steps.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def advisory(fold):
    return _j(os.path.join(RUNS, "coral_nb3", f"fold{fold}", "dilute10_feature_norm_advisory.json"))


def advisory_rows(fold):
    with open(os.path.join(RUNS, "coral_nb3", f"fold{fold}", "dilute10_feature_norm_advisory.csv"),
              encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_record(arm, fold):
    """The run's own record: run.json (CORAL x nb3) or run_config.json (the others)."""
    for name in ("run.json", "run_config.json"):
        p = os.path.join(RUNS, arm, f"fold{fold}", name)
        if os.path.exists(p):
            return _j(p)
    raise FileNotFoundError(f"{arm} fold {fold}: no run record")


# ---- number style (make_rewrite_v5_macros.paper_number): four digits plain, five grouped ----
def paper_number(n: int) -> str:
    return str(n) if abs(n) < 10000 else f"{n:,}".replace(",", r"\,")


def s2(x):
    """Signed, two decimals; a value that rounds to zero prints unsigned."""
    return "0.00" if round(x, 2) == 0.0 else f"{x:+.2f}"


def s4(x):
    return "0.0000" if round(x, 4) == 0.0 else f"{x:+.4f}"


def u2(x):
    return f"{x:.2f}"


def u4(x):
    return f"{x:.4f}"
