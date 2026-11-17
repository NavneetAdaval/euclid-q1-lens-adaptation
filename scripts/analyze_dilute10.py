"""
analyze_dilute10.py
===================
The analysis of the 1:10 diluted-target experiment (brief_dilute10_analysis; PROJECTv7.md
7.53), under the hashed plan ``runs/dilute10_analysis/plan.json``. Parts 1-3 govern.

    python -u scripts/analyze_dilute10.py --plan runs/dilute10_analysis/plan.json \
        --out runs/dilute10_analysis --stage check     # hashes, reproduction, ids; no new value
    python -u scripts/analyze_dilute10.py --plan runs/dilute10_analysis/plan.json \
        --out runs/dilute10_analysis --stage all       # writes results.json and REPORT.md

STORED LOGITS ONLY. No model is loaded, nothing is trained or re-scored. Every value goes
through the paper's own code, which is imported, not re-implemented:

  * ``metrics.metric_suite``                    -- every metric;
  * ``paper/scripts/_common.pool_filter``       -- the cleaned (D1), original and reduced
                                                    (D2) pools, through each run's ids.npz;
  * ``analyze_lambda0_r6.paired_t``             -- the paired t-interval, df = 3, with the
                                                    sign count of ``_common.contrast``;
  * ``compare_methods.compare``                 -- the frozen comparison's class-conditional
                                                    paired bootstrap (B = 1000, seed 0).

The reproduction check runs first; if the paper's Table 5 per-fold TPRs or Table 4
contrasts do not reproduce at four decimals, it STOPS before any diluted value exists.
Writes only into --out, and refuses to overwrite without --force.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "paper", "scripts"))

import numpy as np                                            # noqa: E402

import _common as C                                           # noqa: E402  (pool filter)
import metrics as me                                          # noqa: E402
from analyze_lambda0_r6 import paired_t                       # noqa: E402
import compare_methods as CM                                  # noqa: E402
from integrity import verify as _verify_integrity             # noqa: E402

FOLDS = (1, 2, 3, 4)
TARGET_FPR = 1e-3
ARMS = ("DANN x nb3", "BNM x nb3", "CORAL x nb3", "FreeMatch x nb3")
SHORT = {"DANN x nb3": "DANN", "BNM x nb3": "BNM", "CORAL x nb3": "CORAL",
         "FreeMatch x nb3": "FreeMatch"}
DISPLAY = {a: a.replace(" x ", " × ") for a in ARMS}
READ = {"DANN x nb3": 4000, "BNM x nb3": 4000, "CORAL x nb3": 2000, "FreeMatch x nb3": 1500}
L0_PRIMARY = {a: 4000 for a in ARMS}
L0_SENS = {"CORAL x nb3": 2000, "FreeMatch x nb3": 1500}
METRICS = ("tpr", "auroc", "comp_at_purity", "purity_corr")
POOL_NAMES = {"D1": "cleaned", "orig": "original", "D2": "reduced"}
# Edits after the plan recorded this driver at 74f3876f... (plan.json "code"); each one is
# copied into results.json.
DRIVER_EDITS = [
    "score_predictions: D unpacked as (DANN, CORAL, BNM, FM) from ARMS, whose order is "
    "(DANN, BNM, CORAL, FM), so components 2a/2b(i)/3a/4b(i) read BNM's D as CORAL's and vice "
    "versa in the first full run; fixed to (DANN, BNM, CORAL, FM). No other line changed; "
    "every contrast, branch and descriptive value is unaffected."]


def R(*p):
    return os.path.join(REPO, *p)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def self_cert(d):
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ------------------------------------------------------------------ plan and inputs
class Plan:
    def __init__(self, path):
        self.path = path
        self.d = jload(path)
        dg = self_cert(self.d)
        if dg != self.d["sha256_of_body_without_this_field"]:
            raise SystemExit("STOP: plan does not hash to its own digest")
        rec = open(R("PROJECTv7.md"), encoding="utf-8").read()
        if dg not in rec:
            raise SystemExit("STOP: the plan's digest is not in PROJECTv7.md -- the plan must "
                             "be recorded before any value is computed")
        self.digest = dg
        self.inp = self.d["inputs"]

    def verify_inputs(self):
        n = 0
        bad = []

        def walk(x):
            nonlocal n
            if isinstance(x, dict):
                if set(x) >= {"path", "sha256"} and isinstance(x["path"], str):
                    n += 1
                    if sha(R(x["path"])) != x["sha256"]:
                        bad.append(x["path"])
                    return
                for v in x.values():
                    walk(v)
        walk(self.inp)
        walk(self.d["pool"])
        for name, e in self.d["code"].items():
            if name == "scripts/analyze_dilute10.py":
                continue                    # the driver: recorded below, edits declared
            walk(e)
        if bad:
            raise SystemExit(f"STOP: {len(bad)} planned input(s) changed: {bad}")
        return n

    def diluted(self, arm, k):
        return self.inp["diluted"][arm]["folds"][str(k)]["logits"]["path"]

    def balanced(self, arm, k, step):
        return self.inp["balanced"][arm]["folds"][str(k)][f"step_{step}"]["path"]

    def frozen(self, stage, k):
        return self.inp["diluted_frozen"][stage]["folds"][str(k)]


# ------------------------------------------------------------------ measurement
_CACHE = {}


def suite(relpath, fold):
    """metric_suite on a stored logits file, through the current pool's filter."""
    key = (C.POOL, relpath, fold)
    if key not in _CACHE:
        pos, neg = C._pooled_npz(R(relpath), fold)
        y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
        out = dict(me.metric_suite(y, np.concatenate([pos, neg]), target_fpr=TARGET_FPR))
        out["n_pos"], out["n_neg"] = int(pos.size), int(neg.size)
        _CACHE[key] = out
    return _CACHE[key]


def fold_mean(vals):
    """_common.fold_mean's rule: n/a for the whole arm if any fold is undefined."""
    v = [vals[k] for k in FOLDS]
    if any(not fin(x) for x in v):
        return float("nan"), float("nan")
    return st.mean(v), st.stdev(v)


def contrast(a, b):
    """_common.contrast's form: paired_t on a - b per fold, plus the sign count."""
    diffs = [a[k] - b[k] for k in FOLDS]
    out = paired_t(diffs)
    sign = 1.0 if out["mean"] >= 0 else -1.0
    out["n_same_sign"] = sum(1 for d in diffs if d * sign > 0)
    out["a"], out["b"] = dict(a), dict(b)
    return out


def perfold(paths, metric="tpr"):
    return {k: float(suite(paths[k], k)[metric]) for k in FOLDS}


def paths_dil(P, arm):
    return {k: P.diluted(arm, k) for k in FOLDS}


def paths_bal(P, arm, step):
    return {k: P.balanced(arm, k, step) for k in FOLDS}


# ------------------------------------------------------------------ section 2: the check
def _tex_rows(path, first_only=True):
    txt = open(R(path), encoding="utf-8").read()
    blocks = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", txt, re.S)
    rows = []
    for b in (blocks[:1] if first_only else blocks):
        for ln in b.splitlines():
            if "&" in ln and ln.rstrip().endswith(r"\\") and not ln.startswith("Arm") \
                    and not ln.startswith("Contrast"):
                rows.append([c.strip() for c in ln.rstrip()[:-2].split("&")])
    return rows


NUM = re.compile(r"[+-]?\d+\.\d{4}")


def reproduction(P):
    out = {"per_fold": [], "contrasts": [], "mismatches": []}
    with C.pool("D1"):
        # Table 5, TPR block
        want = {("lambda=0", 1500), ("lambda=0", 2000), ("lambda=0", 4000),
                ("BNM x nb3", 4000), ("FreeMatch x nb3", 1500), ("DANN x nb3", 4000),
                ("CORAL x nb3", 2000)}
        seen = set()
        for cells in _tex_rows(P.inp["paper_reference_tables"]["per_fold_tpr (Table 5)"]
                               ["path"]):
            lab, step = cells[0], int(cells[1])
            arm = ("lambda=0" if "lambda=0" in lab else
                   next((a for a in ARMS if lab.startswith(SHORT[a])), None))
            if (arm, step) not in want:
                continue
            seen.add((arm, step))
            vals = perfold(paths_bal(P, arm, step))
            m, s = fold_mean(vals)
            printed = [NUM.search(c).group(0) for c in cells[3:9]]
            mine = [f"{vals[k]:.4f}" for k in FOLDS] + [f"{m:.4f}", f"{s:.4f}"]
            ok = printed == mine
            out["per_fold"].append({"arm": arm, "step": step, "printed": printed,
                                    "computed": mine, "match": ok})
            if not ok:
                out["mismatches"].append(("per_fold", arm, step, printed, mine))
        if seen != want:
            out["mismatches"].append(("per_fold rows missing", sorted(want - seen)))
        # Table 4, balanced - lambda=0 rows
        rows = {("DANN x nb3", 4000, 4000), ("BNM x nb3", 4000, 4000),
                ("CORAL x nb3", 2000, 4000), ("CORAL x nb3", 2000, 2000),
                ("FreeMatch x nb3", 1500, 4000), ("FreeMatch x nb3", 1500, 1500)}
        seen = set()
        for cells in _tex_rows(P.inp["paper_reference_tables"]
                               ["contrasts_four_decimals (Table 4)"]["path"]):
            lab = cells[0]
            if r"$-$ $\lambda=0$" not in lab:
                continue
            arm = next((a for a in ARMS if lab.startswith(SHORT[a])), None)
            ra, rb = (int(x) for x in cells[1].split("/"))
            if (arm, ra, rb) not in rows:
                continue
            seen.add((arm, ra, rb))
            r = contrast(perfold(paths_bal(P, arm, ra)), perfold(paths_bal(P, "lambda=0", rb)))
            printed = NUM.findall(cells[2]) + NUM.findall(cells[3]) + [cells[4]]
            mine = [f"{r['mean']:+.4f}", f"{r['ci_lo']:+.4f}", f"{r['ci_hi']:+.4f}",
                    f"{r['n_same_sign']}/4"]
            ok = printed == mine
            out["contrasts"].append({"arm": arm, "steps": [ra, rb], "printed": printed,
                                     "computed": mine, "match": ok, "mean": r["mean"],
                                     "ci": [r["ci_lo"], r["ci_hi"]]})
            if not ok:
                out["mismatches"].append(("contrast", arm, ra, rb, printed, mine))
        if seen != rows:
            out["mismatches"].append(("contrast rows missing", sorted(rows - seen)))
    out["passed"] = not out["mismatches"]
    return out


def ids_check(P):
    """Diluted test ids: splits.json negatives (in order, 265 removed by D1) + the fold's
    positives as the balanced and lambda=0 comparators store them."""
    S = C.pool_sets()
    out = {}
    for k in FOLDS:
        ref_pos = list(C._stored_ids(R(P.balanced("lambda=0", k, 4000)), k)[0])
        files = {a: P.diluted(a, k) for a in ARMS}
        files.update({f"{st_} (frozen)": P.frozen(st_, k)["logits"]["path"]
                      for st_ in ("coral_frozen", "freematch_frozen")})
        for name, p in files.items():
            pos_ids, neg_ids = C._stored_ids(R(p), k)
            neg_ids = [str(x) for x in neg_ids]
            pos_ids = [str(x) for x in pos_ids]
            bal_arm = name if name in ARMS else None
            bal_pos = ([str(x) for x in C._stored_ids(R(P.balanced(bal_arm, k, READ[bal_arm])),
                                                      k)[0]] if bal_arm else None)
            keep = C.pool_keep_neg(neg_ids)
            rec = {"neg_order_equals_splits": neg_ids == S["split_neg_locs"],
                   "n_neg": len(neg_ids), "d1_removed": int((~keep).sum()),
                   "d1_kept": int(keep.sum()),
                   "pos_equal_lambda0": pos_ids == [str(x) for x in ref_pos],
                   "pos_equal_balanced_same_arm": (pos_ids == bal_pos) if bal_pos else None,
                   "n_pos": len(pos_ids)}
            rec["pass"] = (rec["neg_order_equals_splits"] and rec["d1_removed"] == 265
                           and rec["pos_equal_lambda0"]
                           and rec["pos_equal_balanced_same_arm"] in (True, None))
            out[f"{name}/fold{k}"] = rec
    return {"per_file": out, "passed": all(v["pass"] for v in out.values())}


# ------------------------------------------------------------------ section 3: compute
def per_arm_table(P):
    tab = {}
    for arm in ARMS:
        tab[arm] = {"read_at": READ[arm], "metrics": {}}
        for m in METRICS:
            v = perfold(paths_dil(P, arm), m)
            mean, sd = fold_mean(v)
            tab[arm]["metrics"][m] = {"per_fold": v, "mean": mean, "sd": sd}
        tab[arm]["n_pos"] = {k: suite(P.diluted(arm, k), k)["n_pos"] for k in FOLDS}
        tab[arm]["n_neg"] = {k: suite(P.diluted(arm, k), k)["n_neg"] for k in FOLDS}
    return tab


def comparator_table(P):
    tab = {}
    for arm, steps in (("lambda=0", (4000, 2000, 1500)),) + tuple(
            (a, (READ[a],)) for a in ARMS):
        for s in steps:
            row = {}
            for m in METRICS:
                v = perfold(paths_bal(P, arm, s), m)
                mean, sd = fold_mean(v)
                row[m] = {"per_fold": v, "mean": mean, "sd": sd}
            tab[f"{arm} (balanced) @{s}" if arm != "lambda=0" else f"lambda=0 @{s}"] = row
    return tab


def contrasts(P):
    G, D = {}, {}
    for arm in ARMS:
        dil = perfold(paths_dil(P, arm))
        G[(arm, "primary")] = contrast(dil, perfold(paths_bal(P, "lambda=0", L0_PRIMARY[arm])))
        G[(arm, "primary")].update(diluted_at=READ[arm], lambda0_at=L0_PRIMARY[arm])
        if arm in L0_SENS:
            G[(arm, "sensitivity")] = contrast(dil, perfold(paths_bal(P, "lambda=0",
                                                                      L0_SENS[arm])))
            G[(arm, "sensitivity")].update(diluted_at=READ[arm], lambda0_at=L0_SENS[arm])
        D[arm] = contrast(perfold(paths_bal(P, arm, READ[arm])), dil)
        D[arm].update(at=READ[arm])
    return G, D


def balanced_minus_l0(P, arm, l0_step):
    return contrast(perfold(paths_bal(P, arm, READ[arm])),
                    perfold(paths_bal(P, "lambda=0", l0_step)))


def branch_of(g, d):
    if not g["excludes_zero"]:
        return "COLLAPSES"
    if g["ci_lo"] > 0:
        return "SHRINKS" if d["ci_lo"] > 0 else "SURVIVES"
    return None                       # wholly below zero: no part-1 branch fits


def s4(x):
    return f"{x:+.4f}"


def ci4(r):
    return f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"


def verdicts(P, G, D, bml):
    table = {b["branch"]: b["licensed_sentence"]
             for b in P.d["verdict"]["part1_branches_VERBATIM"]["table_VERBATIM"]}
    qual = P.d["verdict"]["part1_branches_VERBATIM"]["mandatory_qualifier_VERBATIM"]
    nv = P.d["no_verdict_rule_part3"]["applied_updates_from_run_files"]
    out = {}
    for arm in ARMS:
        key = {"CORAL x nb3": "coral_nb3", "FreeMatch x nb3": "freematch_nb3"}.get(arm)
        fires = bool(key and any(v["fires_below_half"] for v in nv[key].values()))
        g, d = G[(arm, "primary")], D[arm]
        rec = {"no_verdict_rule_fires": fires, "G_primary": [g["mean"], g["ci_lo"], g["ci_hi"]],
               "D": [d["mean"], d["ci_lo"], d["ci_hi"]],
               "balanced_minus_lambda0_primary": bml[arm]["mean"]}
        if fires:
            rec.update(branch=None, reason="part 3 no-verdict rule")
            out[arm] = rec
            continue
        b = branch_of(g, d)
        rec["branch"] = b
        if b is None:
            rec["reason"] = "G's interval lies wholly below zero: no part-1 branch fits"
            out[arm] = rec
            continue
        sent = table[b].strip().strip('"').strip("\u201c\u201d")
        sent = sent.replace("[arm]", DISPLAY[arm])
        sent = sent.replace("G [CI]", f"G = {s4(g['mean'])} {ci4(g)}")
        sent = sent.replace("D [CI]", f"D = {s4(d['mean'])} {ci4(d)}")
        rec["licensed_sentence"] = sent
        half = 0.5 * bml[arm]["mean"]
        rec["qualifier_test"] = {"D_ci_hi": d["ci_hi"], "half_balanced_minus_lambda0": half,
                                 "applies": b == "SURVIVES",
                                 "fires": bool(b == "SURVIVES" and d["ci_hi"] > half)}
        if rec["qualifier_test"]["fires"]:
            q = qual.split("add:", 1)[1].strip().strip('"').strip("\u201c\u201d")
            q = q.replace("[D upper]", f"{d['ci_hi']:.4f}")
            q = q.replace("[balanced \u2212 \u03bb=0]", f"{bml[arm]['mean']:.4f}")
            rec["licensed_sentence"] = sent.rstrip() + " " + q
        # the sensitivity form's branch, reported beside, never the verdict
        if (arm, "sensitivity") in G:
            rec["sensitivity_form_branch"] = branch_of(G[(arm, "sensitivity")], d)
        out[arm] = rec
    return out


def bookkeeping():
    out = {}
    for key, budget in (("dann", 4000), ("coral_nb3", 2000)):
        for k in FOLDS:
            rows = list(csv.DictReader(open(R("runs", "dilute10", key, f"fold{k}", "steps.csv"),
                                            encoding="utf-8")))
            rows = [r for r in rows if 1 <= int(r["step"]) <= budget]
            skips = sum(int(float(r["step_skipped"] or 0)) for r in rows)
            out[f"{key}/fold{k}"] = {"skips": skips, "applied": budget - skips,
                                     "applied_fraction": (budget - skips) / budget}
    for k in FOLDS:
        rows = list(csv.DictReader(open(R("runs", "dilute10", "freematch_nb3", f"fold{k}",
                                          "diagnostics.csv"), encoding="utf-8")))
        sk = int(round(sum(float(r["step_skipped_mean"]) * int(r["n_steps_in_window"])
                           for r in rows if r["n_steps_in_window"] not in ("", "0"))))
        out[f"freematch_nb3/fold{k}"] = {"skips": sk, "applied": 1500 - sk,
                                         "applied_fraction": (1500 - sk) / 1500}
        out[f"bnm/fold{k}"] = {"skips": None, "note": "not recorded (PROJECTv7 7.45)"}
        for stg in ("coral_frozen", "freematch_frozen"):
            out[f"{stg}/fold{k}"] = {"skips": None,
                                     "note": ("fp32 on CPU, no GradScaler" if stg == "coral_frozen"
                                              else "not recorded (PROJECTv7 7.49)")}
    lens = {}
    for key in ("dann", "bnm", "coral_nb3", "freematch_nb3", "coral_frozen", "freematch_frozen"):
        for k in FOLDS:
            s = jload(R("runs", "dilute10", key, f"fold{k}", "target_lens_fraction_summary.json"))
            lens[f"{key}/fold{k}"] = {"mean": s["mean"], "sd": s["sd"],
                                      "expected": s["expected"], "n_batches": s["n_batches"]}
    adv = {f"coral_nb3/fold{k}": jload(R("runs", "dilute10", "coral_nb3", f"fold{k}",
                                         "dilute10_feature_norm_advisory.json"))
           for k in FOLDS}
    adv = {k: {"n_firings": v["n_firings"], "first_step": v["first_step"],
               "last_step": v["last_step"], "kinds": v["kinds"]} for k, v in adv.items()}
    return {"skips": out, "lens_fraction": lens, "coral_feature_norm_advisory": adv}


def bnm_trajectory():
    out = {}
    for tag, tmpl in (("diluted", ("runs", "dilute10", "bnm", "fold{k}", "trace.csv")),
                      ("balanced_context", ("runs", "bnm_r6_folds", "bnm_r6_fold{k}",
                                            "trace.csv"))):
        out[tag] = {}
        for k in FOLDS:
            rows = list(csv.DictReader(open(R(*[t.format(k=k) for t in tmpl]),
                                            encoding="utf-8")))
            out[tag][k] = [(int(r["step"]), float(r["bnm_pred_pos_frac"])) for r in rows]
    return out


def head_growth(P):
    rec_ln = {"diluted": {1: 97.0, 2: 103.7, 3: 98.2, 4: 102.2},
              "round1": {1: 22.4, 2: 33.1, 3: 24.0, 4: 23.4}}
    rec_head = {"diluted": {1: 134.2, 2: 139.0, 3: 133.4, 4: 137.1},
                "round1": {1: 67.5, 2: 72.6, 3: 68.2, 4: 66.0}}
    gv = {k: jload(R(P.frozen("coral_frozen", k)["gate_values"]["path"]))["head_l2"]
          for k in FOLDS}
    fm = {k: jload(R(P.frozen("freematch_frozen", k)["gate_values"]["path"]))["head_l2"]
          for k in FOLDS}
    return {"source": "PROJECTv7 7.46 (LayerNorm gain norm trunk.0.weight; ||head||_2), "
                      "unit gain sqrt(640) = 25.3; head_l2 re-read from gate_values.json",
            "coral_ln_gain_norm": rec_ln, "coral_head_l2_7_46": rec_head,
            "coral_head_l2_gate_values": gv, "freematch_frozen_head_l2_gate_values": fm}


def frozen_stages(P):
    """Diluted frozen heads (step 8000) per fold, decomposition, and the paper's starts."""
    out = {}
    for stage in ("coral_frozen", "freematch_frozen"):
        per = {}
        for m in METRICS:
            v = {k: float(suite(P.frozen(stage, k)["logits"]["path"], k)[m]) for k in FOLDS}
            mean, sd = fold_mean(v)
            per[m] = {"per_fold": v, "mean": mean, "sd": sd}
        gate = {k: jload(R(P.frozen(stage, k)["gate_values"]["path"]))["cleaned_pool"]
                for k in FOLDS}
        per["gate_values_cleaned_tpr"] = {k: gate[k]["tpr"] for k in FOLDS}
        per["gate_values_agree"] = all(abs(gate[k]["tpr"] - per["tpr"]["per_fold"][k]) < 1e-12
                                       for k in FOLDS)
        out[stage] = per
    r1 = {}
    for tag, run in (("CORAL", "bakeoff_coral"), ("FreeMatch", "bakeoff_freematch_5fold")):
        s = C.round1_summary(run, folds=FOLDS)
        r1[tag] = {k: float(s["per_fold"][k]["tpr"]) for k in FOLDS}
    out["round1_tpr"] = r1
    out["round1_start_mean"] = {t: float(np.mean([v[k] for k in FOLDS])) for t, v in r1.items()}
    return out


def decomposition(P, G, fz):
    lam = fold_mean(perfold(paths_bal(P, "lambda=0", 4000)))[0]
    u_d = fold_mean(perfold(paths_dil(P, "FreeMatch x nb3")))[0]
    u_b = fold_mean(perfold(paths_bal(P, "FreeMatch x nb3", 1500)))[0]
    start_d = fz["freematch_frozen"]["tpr"]["mean"]
    start_b = fz["round1_start_mean"]["FreeMatch"]
    out = {"basis": "recomputed from the persisted step_8000 logits through pool_filter + "
                    "metric_suite (cleaned pool); equal to gate_values.json cleaned_pool: "
                    + str(fz["freematch_frozen"]["gate_values_agree"]),
           "lambda0_4000_mean": lam,
           "diluted": {"start": start_d, "frozen_part": start_d - lam,
                       "unfrozen_part": u_d - start_d, "G_primary": u_d - lam},
           "balanced": {"start": start_b, "frozen_part": start_b - lam,
                        "unfrozen_part": u_b - start_b, "total": u_b - lam},
           "coral_start": {"diluted": fz["coral_frozen"]["tpr"]["mean"],
                           "balanced_round1": fz["round1_start_mean"]["CORAL"]}}
    assert abs(out["diluted"]["G_primary"] - G[("FreeMatch x nb3", "primary")]["mean"]) < 1e-12
    return out


def frozen_bootstrap(P):
    """compare_methods.compare (the frozen comparison's bootstrap), diluted - round 1."""
    raw = CM._load_fold
    tag_of = {}

    def pooled(run_dir, k):
        if run_dir in tag_of:                         # a diluted frozen stage
            f = P.frozen(tag_of[run_dir], k)
            z = np.load(R(f["logits"]["path"]))
            pos_ids, neg_ids = C._stored_ids(R(f["logits"]["path"]), k)
            d = {"pos_scores": np.asarray(z[f"test_pos_fold{k}"], float),
                 "neg_scores": np.asarray(z["test_negatives_shared"], float),
                 "neg_ids": np.asarray(neg_ids), "pos_ids": np.asarray(pos_ids)}
            del d["pos_ids"]                          # round-1 dumps carry none to match
        else:
            d = raw(run_dir, k)
        keep = C.pool_keep_neg(d["neg_ids"])
        d["pos_scores"], d["neg_scores"] = C.pool_filter(
            d["pos_scores"], d["neg_scores"], d.get("pos_ids"), d["neg_ids"],
            where=f"frozen bootstrap {run_dir} fold {k}")
        d["neg_ids"] = np.asarray(d["neg_ids"])[keep]
        return d

    out = {}
    CM._load_fold = pooled
    try:
        with C.pool("D1"):
            for tag, stage, r1 in (("FreeMatch", "freematch_frozen", "bakeoff_freematch_5fold"),
                                   ("CORAL", "coral_frozen", "bakeoff_coral")):
                dil_dir = f"<dilute10:{stage}>"
                tag_of[dil_dir] = stage
                r = CM.compare(R("runs", r1), dil_dir, folds=list(FOLDS), B=1000, seed=0,
                               name_a=f"round-1 frozen {tag}", name_b=f"diluted frozen {tag}",
                               verbose=False)
                out[tag] = jsonable(r)
    finally:
        CM._load_fold = raw
    return out


def sensitivities(P, G_c, D_c):
    out = {}
    for pool in ("orig", "D2"):
        with C.pool(pool):
            tab = per_arm_table(P)
            comp = comparator_table(P)
            G, D = contrasts(P)
        agree = {}
        for key, r in G.items():
            agree[f"G {SHORT[key[0]]} {key[1]}"] = (r["excludes_zero"] ==
                                                     G_c[key]["excludes_zero"])
        for arm, r in D.items():
            agree[f"D {SHORT[arm]}"] = r["excludes_zero"] == D_c[arm]["excludes_zero"]
        out[pool] = {"per_arm": tab, "comparators": comp,
                     "G": {f"{a}|{f}": v for (a, f), v in G.items()}, "D": D,
                     "zero_inclusion_matches_cleaned": agree}
    return out


# ------------------------------------------------------------------ predictions
def score_predictions(P, G, D, bml, verd, fz, boot, traj):
    rel = {a: D[a]["mean"] / bml[a]["mean"] for a in ARMS}
    out = []

    def add(pid, text, verdict, basis):
        out.append({"id": pid, "component": text, "score": verdict, "rests_on": basis})

    def hf(b):
        return "held" if b else "failed"

    Dd, Db, Dc, Df = (D[a] for a in ARMS)              # ARMS order: DANN, BNM, CORAL, FM
    Gd = G[("DANN x nb3", "primary")]
    # 1 DANN
    add("1a", "DANN degrades (D point > 0)", hf(Dd["mean"] > 0), f"D = {s4(Dd['mean'])}")
    add("1b(i)", "G point < +0.31 (as stated)", hf(Gd["mean"] < 0.31), f"G = {s4(Gd['mean'])}")
    add("1b(ii)", "G point < reproduced balanced - lambda=0",
        hf(Gd["mean"] < bml["DANN x nb3"]["mean"]),
        f"G = {s4(Gd['mean'])} vs {s4(bml['DANN x nb3']['mean'])}")
    add("1c(i)", "D is positive: point", hf(Dd["mean"] > 0), f"D = {s4(Dd['mean'])}")
    add("1c(ii)", "D is positive: interval excludes zero above", hf(Dd["ci_lo"] > 0),
        f"D CI {ci4(Dd)}")
    add("1d", "G's interval (probably) includes zero", hf(not Gd["excludes_zero"]),
        f"G CI {ci4(Gd)}")
    add("1e", "branch COLLAPSES or SHRINKS", hf(verd["DANN x nb3"]["branch"] in
                                                ("COLLAPSES", "SHRINKS")),
        f"branch {verd['DANN x nb3']['branch']}")
    add("1e'", "'with a wide D'", "not testable", "'wide' is not operationalised in part 1")
    add("1f", "mechanism (marginal alignment pulls lenses toward galaxies; prior collapses)",
        "not testable", "no stored quantity measures it")
    # 2 CORAL
    Gc = G[("CORAL x nb3", "primary")]
    add("2a", "CORAL degrades (D point > 0)", hf(Dc["mean"] > 0), f"D = {s4(Dc['mean'])}")
    add("2b(i)", "less than DANN, absolute", hf(Dc["mean"] < Dd["mean"]),
        f"D CORAL {s4(Dc['mean'])} vs DANN {s4(Dd['mean'])}")
    add("2b(ii)", "less than DANN, relative", hf(rel["CORAL x nb3"] < rel["DANN x nb3"]),
        f"relative loss CORAL {rel['CORAL x nb3']:.4f} vs DANN {rel['DANN x nb3']:.4f}")
    add("2c", "G's interval excludes zero", hf(Gc["excludes_zero"]), f"G CI {ci4(Gc)}")
    add("2d", "branch SHRINKS or SURVIVES", hf(verd["CORAL x nb3"]["branch"] in
                                               ("SHRINKS", "SURVIVES")),
        f"branch {verd['CORAL x nb3']['branch']}")
    add("2e", "mechanism (moment matching a weaker pull)", "not testable", "no stored quantity")
    # 3 BNM
    add("3a", "BNM degrades (D point > 0)", hf(Db["mean"] > 0), f"D = {s4(Db['mean'])}")
    top = max(ARMS, key=lambda a: rel[a])
    add("3b", "the most of the four in relative terms", hf(top == "BNM x nb3"),
        "relative loss " + ", ".join(f"{SHORT[a]} {rel[a]:.4f}" for a in ARMS))
    add("3c", "G's interval may include zero", "not testable",
        "permissive wording; G CI " + ci4(G[("BNM x nb3", "primary")]))
    fin_ = {k: traj["diluted"][k][-1] for k in FOLDS}
    assert all(fin_[k][0] == 4000 for k in FOLDS), fin_
    n_i = sum(1 for k in FOLDS if fin_[k][1] > 0.09)
    n_ii = sum(1 for k in FOLDS if any(v > 0.09 for _s, v in traj["diluted"][k]))

    def mixed(n):
        return "held" if n == 4 else "failed" if n == 0 else f"mixed {n}/4"
    add("3d(i)", "predicted-positive fraction > 0.09 at step 4000", mixed(n_i),
        "step-4000 values " + ", ".join(f"f{k} {fin_[k][1]:.4f}" for k in FOLDS))
    add("3d(ii)", "predicted-positive fraction > 0.09 at any logged step", mixed(n_ii),
        "max per fold " + ", ".join(f"f{k} {max(v for _s, v in traj['diluted'][k]):.4f}"
                                    for k in FOLDS))
    add("3e", "mechanism (high-scoring negatives)", "not testable", "needs a new measurement")
    # 4 FreeMatch
    Gf = G[("FreeMatch x nb3", "primary")]
    add("4a", "FreeMatch degrades (D point > 0)", hf(Df["mean"] > 0), f"D = {s4(Df['mean'])}")
    add("4b(i)", "less than BNM, absolute", hf(Df["mean"] < Db["mean"]),
        f"D FreeMatch {s4(Df['mean'])} vs BNM {s4(Db['mean'])}")
    add("4b(ii)", "less than BNM, relative", hf(rel["FreeMatch x nb3"] < rel["BNM x nb3"]),
        f"relative loss FreeMatch {rel['FreeMatch x nb3']:.4f} vs BNM {rel['BNM x nb3']:.4f}")
    add("4c", "G's interval still excludes zero", hf(Gf["excludes_zero"]), f"G CI {ci4(Gf)}")
    sd_, sb_ = fz["freematch_frozen"]["tpr"]["mean"], fz["round1_start_mean"]["FreeMatch"]
    add("4d(i)", "head start shrinks: point", hf(sd_ < sb_),
        f"diluted frozen {sd_:.4f} vs round-1 frozen {sb_:.4f}")
    b = boot["FreeMatch"]["aggregate"]["tpr"]
    add("4d(ii)", "head start shrinks: bootstrap interval wholly below zero",
        hf(b["ci"][1] < 0), f"diluted - round 1 = {b['diff']:+.4f} "
                            f"[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]")
    add("4e", "mechanism (uniform fairness target; adaptive thresholds)", "not testable",
        "no stored quantity")
    return {"relative_loss": rel, "components": out}


# ------------------------------------------------------------------ report
def fmt(x, nd=4, signed=False):
    if not fin(x):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def report_md(res):
    L = []
    a = L.append
    a("# 1:10 diluted-target experiment — analysis report\n")
    a(f"Plan `runs/dilute10_analysis/plan.json`, self-certifying sha256 `{res['plan_sha256']}`. "
      "Parts 1–3 govern. Stored logits only; every value through `metrics.metric_suite`, "
      "`_common.pool_filter`, `analyze_lambda0_r6.paired_t` and `compare_methods.compare`. "
      "Cleaned pool (D1, rule `b83fa019…`) unless stated. Folds 1–4.\n")
    a("## Reproduction check (§2)\n")
    rp = res["reproduction"]
    a(f"**{'PASSED' if rp['passed'] else 'FAILED'}** — Table 5 per-fold TPR rows and Table 4 "
      "balanced − λ=0 rows, exact at four decimals.\n")
    a("| row | printed (f1, f2, f3, f4, mean, sd) | match |\n|---|---|---|")
    for r in rp["per_fold"]:
        a(f"| {r['arm']} @{r['step']} | {', '.join(r['printed'])} | {r['match']} |")
    a("\n| contrast | steps | printed (Δ, lo, hi, signs) | match |\n|---|---|---|---|")
    for r in rp["contrasts"]:
        a(f"| {r['arm']} − λ=0 | {r['steps'][0]}/{r['steps'][1]} | {', '.join(r['printed'])} "
          f"| {r['match']} |")
    a(f"\nIDs check: **{'PASSED' if res['ids_check']['passed'] else 'FAILED'}** "
      f"({len(res['ids_check']['per_file'])} files: negatives = splits.json order, 265 removed "
      "by D1, positives = the fold's positives in the balanced and λ=0 comparators).\n")

    a("## Per arm × fold (§3.1), diluted, cleaned pool\n")
    names = {"tpr": "TPR@10⁻³", "auroc": "AUROC", "comp_at_purity": "completeness @24% purity",
             "purity_corr": "purity @50% completeness (1:2200)"}
    for m in METRICS:
        a(f"**{names[m]}**\n")
        a("| arm | step | f1 | f2 | f3 | f4 | mean | sd |\n|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            t = res["per_arm"][arm]["metrics"][m]
            a(f"| {DISPLAY[arm]} | {READ[arm]} | " + " | ".join(fmt(t['per_fold'][str(k)])
                                                             for k in FOLDS)
              + f" | {fmt(t['mean'])} | {fmt(t['sd'])} |")
        a("")
    a("## Contrasts (§3.2–3.3), paired t, df = 3\n")
    a("| contrast | steps | mean | 95% CI | excl. 0 | signs |\n|---|---|---|---|---|---|")
    for key, r in res["G"].items():
        arm, form = key.split("|")
        a(f"| G {SHORT[arm]} ({form}) | {r['diluted_at']}/{r['lambda0_at']} | "
          f"{fmt(r['mean'], signed=True)} | [{fmt(r['ci_lo'], signed=True)}, "
          f"{fmt(r['ci_hi'], signed=True)}] | {r['excludes_zero']} | {r['n_same_sign']}/4 |")
    for arm, r in res["D"].items():
        a(f"| D {SHORT[arm]} (balanced − diluted) | {r['at']}/{r['at']} | "
          f"{fmt(r['mean'], signed=True)} | [{fmt(r['ci_lo'], signed=True)}, "
          f"{fmt(r['ci_hi'], signed=True)}] | {r['excludes_zero']} | {r['n_same_sign']}/4 |")
    a("\n**Multiplicity:** 10 t-intervals plus 2 bootstrap intervals, no correction.\n")

    a("## Branches (§3.4)\n")
    a("Part 3's no-verdict rule: applied fractions CORAL nb3 "
      + ", ".join(f"{v['applied_fraction']:.4f}" for v in
                  res["no_verdict_inputs"]["coral_nb3"].values())
      + "; FreeMatch nb3 " + ", ".join(f"{v['applied_fraction']:.4f}" for v in
                                       res["no_verdict_inputs"]["freematch_nb3"].values())
      + " — fires on none.\n")
    a("| arm | G (primary) | D | branch | qualifier | sensitivity-form branch |\n"
      "|---|---|---|---|---|---|")
    for arm, v in res["verdicts"].items():
        g, d = v["G_primary"], v["D"]
        q = v.get("qualifier_test", {})
        a(f"| {DISPLAY[arm]} | {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}] | {d[0]:+.4f} "
          f"[{d[1]:+.4f}, {d[2]:+.4f}] | **{v['branch']}** | "
          f"{('fires' if q.get('fires') else 'n/a' if not q.get('applies') else 'does not fire')}"
          f" | {v.get('sensitivity_form_branch', '—')} |")
    a("")
    for arm, v in res["verdicts"].items():
        if v.get("licensed_sentence"):
            a(f"- **{DISPLAY[arm]}:** {v['licensed_sentence']}")
    a("")
    a("## Descriptive (§3.5)\n")
    rf = res["descriptive"]["retained_fraction"]
    a("Retained fraction G / (balanced − λ=0), point estimates: "
      + "; ".join(f"{k} {fmt(v)}" for k, v in rf.items()) + ".\n")
    dc = res["descriptive"]["freematch_decomposition"]
    a("**FreeMatch decomposition** (" + dc["basis"] + "):\n")
    a("| | start (frozen head) | frozen part | unfrozen part | total |\n|---|---|---|---|---|")
    a(f"| diluted | {fmt(dc['diluted']['start'])} | {fmt(dc['diluted']['frozen_part'], signed=True)}"
      f" | {fmt(dc['diluted']['unfrozen_part'], signed=True)} | "
      f"{fmt(dc['diluted']['G_primary'], signed=True)} |")
    a(f"| balanced (paper §6.3) | {fmt(dc['balanced']['start'])} | "
      f"{fmt(dc['balanced']['frozen_part'], signed=True)} | "
      f"{fmt(dc['balanced']['unfrozen_part'], signed=True)} | "
      f"{fmt(dc['balanced']['total'], signed=True)} |")
    a(f"\nCORAL frozen start: diluted {fmt(dc['coral_start']['diluted'])}, round 1 "
      f"{fmt(dc['coral_start']['balanced_round1'])}. λ=0 @4000 = {fmt(dc['lambda0_4000_mean'])}.\n")
    fz = res["descriptive"]["frozen_stages"]
    a("**Diluted frozen heads, step 8000, TPR per fold** (round 1 beside):\n")
    a("| stage | f1 | f2 | f3 | f4 | mean | round-1 f1–f4 | round-1 mean |\n"
      "|---|---|---|---|---|---|---|---|")
    for stage, tag in (("freematch_frozen", "FreeMatch"), ("coral_frozen", "CORAL")):
        t = fz[stage]["tpr"]
        a(f"| {tag} | " + " | ".join(fmt(t["per_fold"][str(k)]) for k in FOLDS)
          + f" | {fmt(t['mean'])} | " + ", ".join(fmt(fz['round1_tpr'][tag][str(k)])
                                                  for k in FOLDS)
          + f" | {fmt(fz['round1_start_mean'][tag])} |")
    a("\n**Frozen stages against round 1** (compare_methods, class-conditional paired "
      "bootstrap, shared negatives resampled once per replicate, B = 1000, seed 0):\n")
    a("| stage | round 1 | diluted | diluted − round 1 | 95% CI |\n|---|---|---|---|---|")
    for tag, r in res["descriptive"]["frozen_vs_round1"].items():
        t = r["aggregate"]["tpr"]
        a(f"| {tag} | {fmt(t['A'])} | {fmt(t['B'])} | {fmt(t['diff'], signed=True)} | "
          f"[{fmt(t['ci'][0], signed=True)}, {fmt(t['ci'][1], signed=True)}] |")
    hg = res["descriptive"]["coral_head_growth"]
    a("\n**CORAL frozen-stage head growth** (" + hg["source"] + "): LayerNorm gain norm "
      "diluted " + "/".join(f"{hg['coral_ln_gain_norm']['diluted'][str(k)]}" for k in FOLDS)
      + " vs round 1 " + "/".join(f"{hg['coral_ln_gain_norm']['round1'][str(k)]}" for k in FOLDS)
      + "; ‖head‖₂ (gate_values) " + "/".join(f"{hg['coral_head_l2_gate_values'][str(k)]:.2f}"
                                             for k in FOLDS)
      + " vs round 1 " + "/".join(f"{hg['coral_head_l2_7_46']['round1'][str(k)]}" for k in FOLDS)
      + ". FreeMatch frozen ‖head‖₂ " + "/".join(
          f"{hg['freematch_frozen_head_l2_gate_values'][str(k)]:.2f}" for k in FOLDS) + ".\n")
    bk = res["descriptive"]["bookkeeping"]
    a("**Run bookkeeping**\n")
    a("| run | f1 | f2 | f3 | f4 |\n|---|---|---|---|---|")
    for key in ("dann", "coral_nb3", "freematch_nb3"):
        a(f"| {key} skips / applied fraction | " + " | ".join(
            f"{bk['skips'][f'{key}/fold{k}']['skips']} / "
            f"{bk['skips'][f'{key}/fold{k}']['applied_fraction']:.4f}" for k in FOLDS) + " |")
    a("| bnm, frozen stages skips | not recorded (BNM, FreeMatch frozen); CORAL frozen is fp32 "
      "CPU | | | |")
    a("| coral_nb3 feature-norm advisory firings | " + " | ".join(
        str(bk["coral_feature_norm_advisory"][f"coral_nb3/fold{k}"]["n_firings"])
        for k in FOLDS) + " |")
    for key in ("dann", "bnm", "coral_nb3", "freematch_nb3", "coral_frozen", "freematch_frozen"):
        a(f"| {key} lens fraction run-mean (expected) | " + " | ".join(
            f"{bk['lens_fraction'][f'{key}/fold{k}']['mean']:.6f} "
            f"({bk['lens_fraction'][f'{key}/fold{k}']['expected']:.6f})" for k in FOLDS) + " |")
    tr = res["descriptive"]["bnm_pred_pos_trajectory"]
    a("\n**BNM predicted-positive fraction** (trace.csv, target batches; balanced run as "
      "context), steps 125 / 500 / 1000 / 2000 / 3000 / 4000:\n")
    a("| fold | diluted | balanced (context) |\n|---|---|---|")
    pick = (125, 500, 1000, 2000, 3000, 4000)
    for k in FOLDS:
        dd = dict(tr["diluted"][str(k)])
        bb = dict(tr["balanced_context"][str(k)])
        a(f"| {k} | " + " / ".join(f"{dd[s]:.3f}" for s in pick if s in dd) + " | "
          + " / ".join(f"{bb[s]:.3f}" for s in pick if s in bb) + " |")
    a("\n## Prediction scoring (§3.6)\n")
    a("Relative loss D / (balanced − λ=0): " + ", ".join(
        f"{SHORT[a_]} {v:.4f}" for a_, v in res["predictions"]["relative_loss"].items()) + ".\n")
    a("| id | component | score | rests on |\n|---|---|---|---|")
    for c in res["predictions"]["components"]:
        a(f"| {c['id']} | {c['component']} | **{c['score']}** | {c['rests_on']} |")
    a("\nNo prediction was amended; the four predictions are part 1's, verbatim.\n")
    a("## Sensitivities (§3.8, appendix only)\n")
    a("| contrast | cleaned excl. 0 | original | matches | reduced | matches |\n"
      "|---|---|---|---|---|---|")
    so, sr = res["sensitivities"]["orig"], res["sensitivities"]["D2"]
    for key, r in res["G"].items():
        arm, form = key.split("|")
        lab = f"G {SHORT[arm]} ({form})"
        a(f"| {lab} {r['diluted_at']}/{r['lambda0_at']} | {r['excludes_zero']} | "
          f"{fmt(so['G'][key]['mean'], signed=True)} [{fmt(so['G'][key]['ci_lo'], signed=True)}, "
          f"{fmt(so['G'][key]['ci_hi'], signed=True)}] | "
          f"{so['zero_inclusion_matches_cleaned'][f'G {SHORT[arm]} {form}']} | "
          f"{fmt(sr['G'][key]['mean'], signed=True)} [{fmt(sr['G'][key]['ci_lo'], signed=True)}, "
          f"{fmt(sr['G'][key]['ci_hi'], signed=True)}] | "
          f"{sr['zero_inclusion_matches_cleaned'][f'G {SHORT[arm]} {form}']} |")
    for arm, r in res["D"].items():
        a(f"| D {SHORT[arm]} | {r['excludes_zero']} | "
          f"{fmt(so['D'][arm]['mean'], signed=True)} [{fmt(so['D'][arm]['ci_lo'], signed=True)}, "
          f"{fmt(so['D'][arm]['ci_hi'], signed=True)}] | "
          f"{so['zero_inclusion_matches_cleaned'][f'D {SHORT[arm]}']} | "
          f"{fmt(sr['D'][arm]['mean'], signed=True)} [{fmt(sr['D'][arm]['ci_lo'], signed=True)}, "
          f"{fmt(sr['D'][arm]['ci_hi'], signed=True)}] | "
          f"{sr['zero_inclusion_matches_cleaned'][f'D {SHORT[arm]}']} |")
    a("\nPer-fold values on both pools are in `results.json` → `sensitivities`.\n")
    a("## Inputs\n")
    a(f"Every planned input re-hashed at run time ({res['n_inputs_verified']} files) and equal "
      "to the plan. Driver `scripts/analyze_dilute10.py` sha256 "
      f"`{res['driver_sha256']}` (plan recorded `{res['driver_sha256_in_plan']}`). Edits since "
      "the plan: " + " ".join(res["driver_edits_after_plan"]) + "\n")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("check", "all"), default="check")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)
    res_p, rep_p = os.path.join(out_dir, "results.json"), os.path.join(out_dir, "REPORT.md")
    if a.stage == "all" and (os.path.exists(res_p) or os.path.exists(rep_p)) and not a.force:
        raise SystemExit("REFUSING to overwrite results.json / REPORT.md without --force")

    _verify_integrity(where="start")
    P = Plan(a.plan)
    n_in = P.verify_inputs()
    print(f"[plan] {P.digest}; {n_in} planned inputs re-hashed and equal")

    rp = reproduction(P)
    for r in rp["per_fold"] + rp["contrasts"]:
        print(f"  repro {r['arm']} {r.get('step', r.get('steps'))}: printed {r['printed']} "
              f"computed {r['computed']} -> {'OK' if r['match'] else 'MISMATCH'}")
    if not rp["passed"]:
        raise SystemExit(f"STOP: the reproduction check failed: {rp['mismatches']}")
    ic = ids_check(P)
    if not ic["passed"]:
        raise SystemExit(f"STOP: ids check failed: "
                         f"{[k for k, v in ic['per_file'].items() if not v['pass']]}")
    print(f"[check] reproduction PASSED ({len(rp['per_fold'])} per-fold rows, "
          f"{len(rp['contrasts'])} contrasts); ids PASSED ({len(ic['per_file'])} files)")
    if a.stage == "check":
        _verify_integrity(where="end")
        return 0

    with C.pool("D1"):
        tab = per_arm_table(P)
        comp = comparator_table(P)
        G, D = contrasts(P)
        bml = {arm: balanced_minus_l0(P, arm, L0_PRIMARY[arm]) for arm in ARMS}
        bml_s = {arm: balanced_minus_l0(P, arm, L0_SENS[arm]) for arm in L0_SENS}
        verd = verdicts(P, G, D, bml)
        fz = frozen_stages(P)
        dec = decomposition(P, G, fz)
    if any(v["branch"] is None and not v["no_verdict_rule_fires"] for v in verd.values()):
        print("STOP: a G interval lies wholly below zero; no branch fits (plan stop rule)")
    boot = frozen_bootstrap(P)
    traj = bnm_trajectory()
    retained = {f"{SHORT[a_]} (primary)": G[(a_, "primary")]["mean"] / bml[a_]["mean"]
                for a_ in ARMS}
    retained.update({f"{SHORT[a_]} (matched)": G[(a_, "sensitivity")]["mean"] /
                     bml_s[a_]["mean"] for a_ in L0_SENS})
    stop_preds = any(v["branch"] is None and not v["no_verdict_rule_fires"]
                     for v in verd.values())
    preds = (None if stop_preds else
             score_predictions(P, G, D, bml, verd, fz, boot,
                               {t: {k: v for k, v in d.items()} for t, d in traj.items()}))
    sens = sensitivities(P, G, D)

    res = {
        "plan_sha256": P.digest,
        "driver_sha256": sha(os.path.abspath(__file__)),
        "driver_sha256_in_plan": P.d["code"]["scripts/analyze_dilute10.py"]["sha256"],
        "driver_edits_after_plan": DRIVER_EDITS,
        "n_inputs_verified": n_in,
        "inputs_sha256": P.inp,
        "pool": "D1 (cleaned) unless stated",
        "reproduction": rp, "ids_check": ic,
        "per_arm": tab, "comparators": comp,
        "G": {f"{arm}|{form}": v for (arm, form), v in G.items()}, "D": D,
        "balanced_minus_lambda0": {"primary": bml, "matched": bml_s},
        "no_verdict_inputs": P.d["no_verdict_rule_part3"]["applied_updates_from_run_files"],
        "verdicts": verd,
        "descriptive": {"retained_fraction": retained, "freematch_decomposition": dec,
                        "frozen_stages": fz, "frozen_vs_round1": boot,
                        "coral_head_growth": head_growth(P), "bookkeeping": bookkeeping(),
                        "bnm_pred_pos_trajectory": traj},
        "predictions": preds,
        "multiplicity": "10 paired t-intervals (6 G, 4 D) plus 2 bootstrap intervals "
                        "(frozen stages vs round 1), no correction",
        "sensitivities": sens,
        "pool_filter_log_n": len(C.FILTER_LOG),
    }
    res = jsonable(res)
    os.makedirs(out_dir, exist_ok=True)
    with open(res_p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    with open(rep_p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report_md(res))
    _verify_integrity(where="end")
    print(f"wrote {os.path.relpath(res_p, REPO)} {sha(res_p)}")
    print(f"wrote {os.path.relpath(rep_p, REPO)} {sha(rep_p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
