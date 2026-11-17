r"""
crosscheck_dilute10.py -- the paper-v6 block of crosscheck.py.
==============================================================

Called by ``crosscheck.py`` (``build_dilute10``). Every number the v6 generators print for the
1:10 diluted-target experiment and the DANN - lambda=0 memorisation diagnostic is recomputed
here from its artifact, independently of ``_dilute10.py``, ``make_dilute10_macros.py``,
``make_dilute10_tables.py`` and the figure scripts:

  * the diluted per-fold metrics from the STORED LOGITS each run wrote (the paths and sha256
    that runs/dilute10_analysis/results.json records), through ``_common.pool_filter`` (D1,
    original and reduced pools) and ``metrics.metric_suite``; the balanced arms and the
    lambda=0 control through ``_common.per_fold``; the frozen stages from their step-8000
    logits and the round-1 frozen heads through ``_common.round1_summary``;
  * every G and D paired t-interval from those per-fold values (scipy t, df = 3), compared
    with results.json and with every macro and table cell that prints it;
  * the branches from G and D under part 1's table;
  * the two bootstrap intervals, the predictions and the memorisation diagnostic against
    their analysis files (they are that analysis's own output: a seeded bootstrap and a
    scoring that this script does not re-run);
  * the head norms against runs/dilute10_head_norms/head_norms.json; CORAL x nb3's feature
    norms, skips and advisory firings from its steps.csv and advisory logs; the pool counts
    from data/splits/dilute10_pool.json; the hash-table rows re-derived through
    scripts/integrity.py.

PROJECTv7.md quotes the same values (7.46, 7.51, 7.53, 7.54); a handful are also checked
against its printed text, in the crosscheck's usual sense.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics as st
import sys

import numpy as np

import _common as C

sys.path.insert(0, C.REPO_SCRIPTS)
import integrity  # noqa: E402

FOLDS = (1, 2, 3, 4)
ARMS = {"dann": "DANN x nb3", "bnm": "BNM x nb3", "coral": "CORAL x nb3",
        "freematch": "FreeMatch x nb3"}
LAB = {"dann": "DANN", "bnm": "BNM", "coral": "CORAL", "freematch": "FreeMatch"}
MAC = {"dann": "Dann", "bnm": "Bnm", "coral": "Coral", "freematch": "Fm"}
READ = {"dann": 4000, "bnm": 4000, "coral": 2000, "freematch": 1500}
METRICS = ("tpr", "auroc", "comp_at_purity", "purity_corr")
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
WORD = {2: "two", 4: "four", 6: "six", 8: "eight", 10: "ten", 13: "thirteen", 14: "Fourteen"}


def _j(*p):
    with open(C.R(*p), encoding="utf-8") as f:
        return json.load(f)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _ptt(diffs):
    from scipy import stats as sps
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    hw = float(sps.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    s = 1.0 if m >= 0 else -1.0
    return dict(mean=m, lo=m - hw, hi=m + hw, signs=sum(1 for d in diffs if d * s > 0),
                excl=(m - hw > 0) or (m + hw < 0))


def _body(name):
    p = os.path.join(C.TABLES, name)
    return [l.rstrip("\n") for l in open(p, encoding="utf-8") if not l.startswith("%")]


def _rows(name):
    """Data rows of a generated table: the lines between \\midrule/\\toprule and \\bottomrule
    that end in \\\\ and are not a \\multicolumn panel title or a header."""
    out = []
    inside = False
    for l in _body(name):
        if l.startswith(r"\begin{tabular}"):
            inside = True
            continue
        if l.startswith(r"\end{tabular}"):
            inside = False
        if inside and l.rstrip().endswith("\\\\") and not l.startswith(r"\multicolumn"):
            out.append([c.strip() for c in re.split(r"(?<!\\)&", l.rstrip().rstrip("\\").strip())])
    return out


def _nums(cell):
    return [float(t) for t in NUM.findall(cell.replace("$", "").replace("{,}", ""))]


BUDGET = {"dann": 4000, "coral_nb3": 2000, "freematch_nb3": 1500}


def _skips(run, k):
    """(skipped updates, applied fraction) of one half-precision run, from its own logs:
    steps.csv for DANN and CORAL x nb3, the windowed diagnostics.csv for FreeMatch x nb3."""
    base = C.R("runs", "dilute10", run, f"fold{k}")
    n = BUDGET[run]
    if run == "freematch_nb3":
        rows = list(csv.DictReader(open(os.path.join(base, "diagnostics.csv"), encoding="utf-8")))
        s = int(round(sum(float(r["step_skipped_mean"]) * int(r["n_steps_in_window"])
                          for r in rows if r["n_steps_in_window"] not in ("", "0"))))
        assert sum(int(r["n_steps_in_window"] or 0) for r in rows) == n
    else:
        rows = [r for r in csv.DictReader(open(os.path.join(base, "steps.csv"), encoding="utf-8"))
                if 1 <= int(r["step"]) <= n]
        assert len(rows) == n
        s = sum(int(float(r["step_skipped"] or 0)) for r in rows)
    return s, (n - s) / n


def _printed_macros():
    txt = open(os.path.join(C.TABLES, "M_dilute10.tex"), encoding="utf-8").read()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", txt))


# ======================================================================================
def recompute():
    """Everything recomputed from stored logits and run files. Returns a dict of values."""
    R = _j("runs", "dilute10_analysis", "results.json")
    ins = R["inputs_sha256"]
    V = {"R": R}
    # ---- diluted per-fold metrics from the stored logits, on three pools ----
    dil = {p: {} for p in ("D1", "orig", "D2")}
    for arm, name in ARMS.items():
        for pool in dil:
            dil[pool][arm] = {m: {} for m in METRICS}
        for k in FOLDS:
            f = ins["diluted"][name]["folds"][str(k)]
            lp, ip = C.R(*f["logits"]["path"].split("/")), C.R(*f["ids"]["path"].split("/"))
            assert _sha(lp) == f["logits"]["sha256"] and _sha(ip) == f["ids"]["sha256"], lp
            pos, neg = C._RAW_NPZ(lp, k)                  # the raw reader, before any filter
            zi = np.load(ip)
            pid, nid = ((zi["pos_ids"], zi["neg_ids"]) if "neg_ids" in zi.files
                        else (zi[f"test_pos_fold{k}"], zi["test_negatives_shared"]))
            for pool in dil:
                p, n = C.pool_filter(pos, neg, list(pid), list(nid),
                                     where=f"crosscheck dilute10 {arm} f{k}", pool_name=pool)
                y = np.concatenate([np.ones(p.size, int), np.zeros(n.size, int)])
                s = C.me.metric_suite(y, np.concatenate([p, n]), target_fpr=C.TARGET_FPR)
                for m in METRICS:
                    dil[pool][arm][m][k] = float(s[m])
    V["dil"] = dil
    # ---- balanced arms and the control, on the same three pools ----
    bal, lam = {}, {}
    for pool in dil:
        with C.pool(pool):
            bal[pool] = {a: {m: C.per_fold(a, m, READ[a]) for m in METRICS} for a in ARMS}
            lam[pool] = {ro: {m: C.per_fold("lambda0", m, ro) for m in METRICS}
                         for ro in (4000, 2000, 1500)}
    V["bal"], V["lam"] = bal, lam
    # ---- G and D on each pool ----
    G, D = {}, {}
    for pool in dil:
        G[pool], D[pool] = {}, {}
        for arm in ARMS:
            t = dil[pool][arm]["tpr"]
            G[pool][(arm, "primary")] = _ptt([t[k] - lam[pool][4000]["tpr"][k] for k in FOLDS])
            if arm in ("coral", "freematch"):
                G[pool][(arm, "sensitivity")] = _ptt(
                    [t[k] - lam[pool][READ[arm]]["tpr"][k] for k in FOLDS])
            D[pool][arm] = _ptt([bal[pool][arm]["tpr"][k] - t[k] for k in FOLDS])
    V["G"], V["D"] = G, D
    V["B"] = {arm: _ptt([bal["D1"][arm]["tpr"][k] - lam["D1"][4000]["tpr"][k] for k in FOLDS])
              for arm in ARMS}
    V["Bm"] = {arm: _ptt([bal["D1"][arm]["tpr"][k] - lam["D1"][READ[arm]]["tpr"][k]
                          for k in FOLDS]) for arm in ("coral", "freematch")}
    # ---- the frozen stages: diluted from their step-8000 logits, round 1 from score dumps ----
    fz = {}
    for st_, key in (("coral_frozen", "CORAL"), ("freematch_frozen", "FreeMatch")):
        fz[key] = {m: {} for m in ("tpr", "auroc")}
        for k in FOLDS:
            f = ins["diluted_frozen"][st_]["folds"][str(k)]
            lp, ip = C.R(*f["logits"]["path"].split("/")), C.R(*f["ids"]["path"].split("/"))
            assert _sha(lp) == f["logits"]["sha256"], lp
            z, zi = np.load(lp), np.load(ip)
            p, n = C.pool_filter(z[f"test_pos_fold{k}"], z["test_negatives_shared"],
                                 list(zi[f"test_pos_fold{k}"]), list(zi["test_negatives_shared"]),
                                 where=f"crosscheck {st_} f{k}", pool_name="D1")
            y = np.concatenate([np.ones(p.size, int), np.zeros(n.size, int)])
            s = C.me.metric_suite(y, np.concatenate([p, n]), target_fpr=C.TARGET_FPR)
            fz[key]["tpr"][k], fz[key]["auroc"][k] = float(s["tpr"]), float(s["auroc"])
    with C.pool("D1"):
        r1 = {"CORAL": C.round1_summary("bakeoff_coral", folds=FOLDS),
              "FreeMatch": C.round1_summary("bakeoff_freematch_5fold", folds=FOLDS)}
    V["fz"] = fz
    V["r1"] = {k: {f: float(v["per_fold"][f]["tpr"]) for f in FOLDS} for k, v in r1.items()}
    V["r1_auroc"] = {k: {f: float(v["per_fold"][f]["auroc"]) for f in FOLDS} for k, v in r1.items()}
    # ---- CORAL x nb3 run files ----
    cn = {"feat_min": [], "feat_max": [], "start": [], "start_both": [], "skips": [], "n": [],
          "fires": [],
          "ceiling": set(), "first_skips_contiguous": True}
    for k in FOLDS:
        base = C.R("runs", "dilute10", "coral_nb3", f"fold{k}")
        rows = list(csv.DictReader(open(os.path.join(base, "steps.csv"), encoding="utf-8")))
        cn["n"].append(len(rows))
        cn["feat_min"].append(min(min(float(r["src_feat_l2"]), float(r["tgt_feat_l2"])) for r in rows))
        cn["feat_max"].append(max(max(float(r["src_feat_l2"]), float(r["tgt_feat_l2"])) for r in rows))
        cn["start"].append(float(rows[0]["src_feat_l2"]))
        cn["start_both"].append((float(rows[0]["src_feat_l2"]), float(rows[0]["tgt_feat_l2"])))
        sk = [int(float(r["step_skipped"])) for r in rows]
        cn["skips"].append(sum(sk))
        cn["first_skips_contiguous"] &= (sk[:sum(sk)] == [1] * sum(sk))
        adv = list(csv.DictReader(open(os.path.join(base, "dilute10_feature_norm_advisory.csv"),
                                       encoding="utf-8")))
        cn["fires"].append(len(adv))
        cn["ceiling"] |= {float(r["ceiling"]) for r in adv}
    V["cn"] = cn
    return V


# ======================================================================================
def build(check) -> None:
    V = recompute()
    R = V["R"]
    SRC_L = "stored diluted logits -> _common.pool_filter -> metrics.metric_suite"
    SRC_R = "runs/dilute10_analysis/results.json"

    # ---- (1) recomputation against the analysis file ----
    for arm, name in ARMS.items():
        for m in METRICS:
            for k in FOLDS:
                q = f"{R['per_arm'][name]['metrics'][m]['per_fold'][str(k)]:.4f}"
                check("V6-RE", f"{name} f{k} {m} (results.json)", q, V["dil"]["D1"][arm][m][k], SRC_L)
        for form in (("primary", "sensitivity") if arm in ("coral", "freematch") else ("primary",)):
            r, g = R["G"][f"{name}|{form}"], V["G"]["D1"][(arm, form)]
            for key, gk in (("mean", "mean"), ("ci_lo", "lo"), ("ci_hi", "hi")):
                check("V6-RE", f"G {name} {form} {key}", f"{r[key]:.6f}", g[gk], SRC_L)
            check("V6-RE", f"G {name} {form} signs", str(r["n_same_sign"]), g["signs"], SRC_L)
        r, d = R["D"][name], V["D"]["D1"][arm]
        for key, dk in (("mean", "mean"), ("ci_lo", "lo"), ("ci_hi", "hi")):
            check("V6-RE", f"D {name} {key}", f"{r[key]:.6f}", d[dk], SRC_L)
        b = R["balanced_minus_lambda0"]["primary"][name]
        check("V6-RE", f"balanced - lambda=0 {name}", f"{b['mean']:.6f}", V["B"][arm]["mean"],
              "_common.per_fold (D1)")
        # the branch, from G and D under part 1's table
        g, d = V["G"]["D1"][(arm, "primary")], V["D"]["D1"][arm]
        br = "COLLAPSES" if g["lo"] <= 0 <= g["hi"] else ("SHRINKS" if d["lo"] > 0 else "SURVIVES")
        check("V6-RE", f"branch {name} = results.json verdict", "1",
              float(br == R["verdicts"][name]["branch"]), SRC_R)
        for pool, pk in (("orig", "orig"), ("D2", "D2")):
            S = R["sensitivities"][pk]
            for form in (("primary", "sensitivity") if arm in ("coral", "freematch") else ("primary",)):
                check("V6-RE", f"{pool} G {name} {form}", f"{S['G'][f'{name}|{form}']['mean']:.6f}",
                      V["G"][pool][(arm, form)]["mean"], SRC_L + f" ({pool})")
            check("V6-RE", f"{pool} D {name}", f"{S['D'][name]['mean']:.6f}",
                  V["D"][pool][arm]["mean"], SRC_L + f" ({pool})")
    # every key of the analysis's G and D, and _dilute10.DILUTE10_CONTRASTS, name the same set
    import _dilute10 as X  # only for the declared list, which must match the analysis file
    keys = {f"{ARMS[a]}|{role}" for kind, a, _r, _c, role in X.DILUTE10_CONTRASTS if kind == "G"}
    keysd = {ARMS[a] for kind, a, _r, _c, _role in X.DILUTE10_CONTRASTS if kind == "D"}
    check("V6-RE", "DILUTE10_CONTRASTS = results.json G and D keys", "1",
          float(keys == set(R["G"]) and keysd == set(R["D"])), SRC_R)
    for kind, a, rd, rc, role in X.DILUTE10_CONTRASTS:
        rr = R["G"][f"{ARMS[a]}|{role}"] if kind == "G" else None
        if rr:
            check("V6-RE", f"DILUTE10_CONTRASTS {a} {role} steps", f"{rd}{rc}",
                  float(f"{rr['diluted_at']}{rr['lambda0_at']}"), SRC_R)
    # the frozen stages
    fzr = R["descriptive"]["frozen_stages"]
    for key, st_ in (("CORAL", "coral_frozen"), ("FreeMatch", "freematch_frozen")):
        for k in FOLDS:
            check("V6-RE", f"{key} frozen f{k} TPR", f"{fzr[st_]['tpr']['per_fold'][str(k)]:.6f}",
                  V["fz"][key]["tpr"][k], "step-8000 logits -> pool_filter (D1)")
            check("V6-RE", f"{key} round-1 frozen f{k} TPR",
                  f"{fzr['round1_tpr'][key][str(k)]:.6f}", V["r1"][key][k],
                  "_common.round1_summary (D1)")
    # PROJECTv7 7.53 prints the G/D table; 7.46 the head norms (spot checks, as printed)
    for name, g, d in (("DANN x nb3", "+0.3350", "-0.0206"), ("BNM x nb3", "+0.1941", "+0.0257"),
                       ("CORAL x nb3", "+0.0102", "+0.3273"), ("FreeMatch x nb3", "+0.2533", "+0.0793")):
        arm = [a for a, n in ARMS.items() if n == name][0]
        check("7.53", f"{name} G (primary)", g, V["G"]["D1"][(arm, "primary")]["mean"], SRC_L)
        check("7.53", f"{name} D", d, V["D"]["D1"][arm]["mean"], SRC_L)

    # ---- (2) the macros ----
    E = expected_macros(V)
    P = _printed_macros()
    missing = sorted(set(P) - set(E))
    check("V6-MAC", "every M_dilute10 macro has an independent expectation", "1",
          float(not missing), "crosscheck_dilute10.expected_macros")
    assert not missing, missing
    for name, shown in sorted(P.items()):
        e = E[name]
        if isinstance(e, str):
            check("V6-MAC", f"{name} = {shown}", "1", float(shown == e), "crosscheck_dilute10")
        else:
            check("V6-MAC", f"{name} = {shown}", shown.replace(r"\,", ""), e, "crosscheck_dilute10")

    # ---- (3) the tables ----
    tables(check, V)


def expected_macros(V) -> dict:
    """Each macro's value, recomputed. Numbers as floats (judged at the printed precision),
    words and labels as the exact strings."""
    R = V["R"]
    E = {}
    for arm in ARMS:
        m = MAC[arm]
        t = [V["dil"]["D1"][arm]["tpr"][k] for k in FOLDS]
        a = [V["dil"]["D1"][arm]["auroc"][k] for k in FOLDS]
        E[f"dlTpr{m}"] = E[f"dlTpr{m}Four"] = st.mean(t)
        E[f"dlSd{m}"] = E[f"dlSd{m}Four"] = st.stdev(t)
        E[f"dlAuroc{m}"] = st.mean(a)
        E[f"dlRead{m}"] = READ[arm]
        for tag, r in (("G", V["G"]["D1"][(arm, "primary")]), ("D", V["D"]["D1"][arm])):
            for suf, key in (("", "mean"), ("Lo", "lo"), ("Hi", "hi")):
                E[f"dl{tag}{m}{suf}"] = E[f"dl{tag}{m}{suf}Four"] = r[key]
            E[f"dl{tag}{m}Signs"] = r["signs"]
        E[f"dlD{m}HiAbs"] = E[f"dlD{m}HiAbsFour"] = abs(V["D"]["D1"][arm]["hi"])
        b = V["B"][arm]["mean"]
        E[f"dlBal{m}"] = E[f"dlBal{m}Four"] = E[f"dlBal{m}Abs"] = E[f"dlBal{m}AbsFour"] = b
        E[f"dlRet{m}"] = V["G"]["D1"][(arm, "primary")]["mean"] / b
        g, d = V["G"]["D1"][(arm, "primary")], V["D"]["D1"][arm]
        E[f"dlBranch{m}"] = ("COLLAPSES" if g["lo"] <= 0 <= g["hi"]
                             else ("SHRINKS" if d["lo"] > 0 else "SURVIVES"))
        if arm in ("coral", "freematch"):
            gm = V["G"]["D1"][(arm, "sensitivity")]
            for suf, key in (("", "mean"), ("Lo", "lo"), ("Hi", "hi")):
                E[f"dlGm{m}{suf}"] = E[f"dlGm{m}{suf}Four"] = gm[key]
            E[f"dlGm{m}Signs"] = gm["signs"]
            E[f"dlBalm{m}"] = V["Bm"][arm]["mean"]
            E[f"dlRetm{m}"] = gm["mean"] / V["Bm"][arm]["mean"]
    gb, gf = V["G"]["D1"][("bnm", "primary")], V["G"]["D1"][("freematch", "primary")]
    E.update(dlGBnmLoThree=gb["lo"], dlGBnmLoAbsThree=abs(gb["lo"]), dlGFmLoThree=gf["lo"],
             dlGFmLoAbsThree=abs(gf["lo"]))
    bd = [V["bal"]["D1"]["dann"]["tpr"][k] for k in FOLDS]
    E.update(dlTprDannBal=st.mean(bd), dlSdDannBal=st.stdev(bd),
             dlLamTpr=st.mean(V["lam"]["D1"][4000]["tpr"].values()),
             dlLamAuroc=st.mean(V["lam"]["D1"][4000]["auroc"].values()))
    # FreeMatch decomposition, recomputed: start = frozen head, frozen part = start - lambda=0
    l4 = st.mean(V["lam"]["D1"][4000]["tpr"].values())
    fs, fsb = st.mean(V["fz"]["FreeMatch"]["tpr"].values()), st.mean(V["r1"]["FreeMatch"].values())
    fu = st.mean(V["dil"]["D1"]["freematch"]["tpr"].values())
    fub = st.mean(V["bal"]["D1"]["freematch"]["tpr"].values())
    for nm, v in (("dlFmStart", fs), ("dlFmStartBal", fsb), ("dlFmFrozenPart", fs - l4),
                  ("dlFmUnfrozenPart", fu - fs), ("dlFmFrozenPartBal", fsb - l4),
                  ("dlFmUnfrozenPartBal", fub - fsb)):
        E[nm] = E[nm + "Four"] = v
    E["dlCoralStart"] = st.mean(V["fz"]["CORAL"]["tpr"].values())
    E["dlCoralStartBal"] = st.mean(V["r1"]["CORAL"].values())
    fvr = R["descriptive"]["frozen_vs_round1"]
    for m, key in (("Fm", "FreeMatch"), ("Coral", "CORAL")):
        agg = fvr[key]["aggregate"]
        agg = json.loads(agg) if isinstance(agg, str) else agg
        t = agg["tpr"]
        # the bootstrap's point estimate is the plain difference of fold-means: recomputed
        diff = st.mean(V["fz"][key]["tpr"].values()) - st.mean(V["r1"][key].values())
        assert abs(diff - t["diff"]) < 1e-12, (key, diff, t["diff"])
        for suf, v in (("", diff), ("Lo", t["ci"][0]), ("Hi", t["ci"][1])):
            E[f"dl{m}FrozenDiff{suf}"] = E[f"dl{m}FrozenDiff{suf}Four"] = v
    E["dlCoralFrozenAuroc"] = st.mean(V["fz"]["CORAL"]["auroc"].values())
    H = _j("runs", "dilute10_head_norms", "head_norms.json")["heads"]
    gd = [H["coral"]["diluted"][str(k)]["ln_gain_norm"] for k in FOLDS]
    gr = [H["coral"]["round1"][str(k)]["ln_gain_norm"] for k in FOLDS]
    E.update(dlCoralGainMin=min(gd), dlCoralGainMax=max(gd), dlCoralGainRoneMin=min(gr),
             dlCoralGainRoneMax=max(gr),
             dlCoralGainRatioWord={4: "four"}[round(st.mean(gd) / st.mean(gr))],
             dlUnitGain=math.sqrt(640))
    cn = V["cn"]
    assert cn["first_skips_contiguous"] and len(set(cn["n"])) == 1 and cn["fires"] == cn["n"]
    E.update(dlCoralSteps=cn["n"][0], dlCoralFeatMin=min(cn["feat_min"]),
             dlCoralFeatMax=max(cn["feat_max"]),
             dlCoralFeatStart=10 * round(st.mean(cn["start"]) / 10),
             dlCoralFeatStartMin=min(min(p) for p in cn["start_both"]),
             dlCoralFeatStartMax=max(max(p) for p in cn["start_both"]),
             dlCoralCeiling=min(cn["ceiling"]), dlCoralSkipMin=min(cn["skips"]),
             dlCoralSkipMax=max(cn["skips"]),
             dlCoralAppliedPct=100 * min((n - s) / n for n, s in zip(cn["n"], cn["skips"])))
    fm = [_skips("freematch_nb3", k) for k in FOLDS]
    E.update(dlFmSkipMin=min(s for s, _ in fm), dlFmSkipMax=max(s for s, _ in fm),
             dlFmAppliedPct=100 * min(a for _, a in fm))
    # BNM's predicted-positive fraction, from each run's trace.csv
    pp, first, nlog, steps_logged = [], [], set(), []
    for k in FOLDS:
        rows = list(csv.DictReader(open(C.R("runs", "dilute10", "bnm", f"fold{k}", "trace.csv"),
                                        encoding="utf-8")))
        v = [float(r["bnm_pred_pos_frac"]) for r in rows]
        pp += v
        first.append((int(rows[0]["step"]), v[0]))
        steps_logged.append([int(r["step"]) for r in rows])
        nlog.add(len(v))
    # r3: one logging interval, every logged step a multiple of it, the last at the budget
    gaps = {b - a for s_ in steps_logged for a, b in zip(s_, s_[1:])}
    assert len(gaps) == 1 and all(s_[0] == min(gaps) and s_[-1] == 4000 for s_ in steps_logged)
    E["dlBnmPpEvery"] = gaps.pop()
    bal_ctx = [v for f in FOLDS for _s, v in R["descriptive"]["bnm_pred_pos_trajectory"]["balanced_context"][str(f)]]
    E.update(dlBnmPpMin=min(pp), dlBnmPpMax=max(pp), dlBnmPpMean=st.mean(pp),
             dlBnmPpFirst=st.mean(v for _s, v in first), dlBnmPpFirstStep=first[0][0],
             dlBnmPpN=nlog.pop(), dlBnmPpBalMin=min(bal_ctx), dlBnmPpBalMax=max(bal_ctx))
    P1 = _j("runs", "dilute10_prereg_part1.json")
    rr = P1["pool"]["realised_ratios"]
    E["dlBnmLensFrac"] = st.mean(rr[str(k)]["dann_bnm_training_pool_lens_fraction"] for k in FOLDS)
    E["dlBnmPpThreshold"] = 0.09
    assert "drifts above the true ~0.09" in P1["predictions_VERBATIM"]["items"][2]
    E["dlNPredWord"] = WORD[len(P1["predictions_VERBATIM"]["items"])]
    nG, nD, nB = len(R["G"]), len(R["D"]), len(R["descriptive"]["frozen_vs_round1"])
    E.update(dlNt=nG + nD, dlNtWord=WORD[nG + nD], dlNtG=nG, dlNtD=nD, dlNboot=nB,
             dlNbootWord=WORD[nB])
    go = V["G"]["orig"][("freematch", "primary")]
    E.update(dlGFmOrig=go["mean"], dlGFmOrigLo=go["lo"], dlGFmOrigHi=go["hi"])

    def excl(r):
        return r["excl"]
    match_o = match_r = 0
    for pool, cnt in (("orig", "o"), ("D2", "r")):
        n = 0
        for key, r in V["G"][pool].items():
            n += excl(r) == excl(V["G"]["D1"][key])
        for arm in ARMS:
            n += excl(V["D"][pool][arm]) == excl(V["D"]["D1"][arm])
        if cnt == "o":
            match_o = n
        else:
            match_r = n
    E.update(dlOrigMatchN=match_o, dlOrigMatchWord=WORD[match_o], dlReducedMatchN=match_r)
    # the pool file
    pool = _j("data", "splits", "dilute10_pool.json")
    pf = {f["fold"]: f for f in pool["folds"]}
    E.update(dlPoolLens=len(pf[1]["target_lens_rows"]), dlPoolLensFour=len(pf[4]["target_lens_rows"]),
             dlPoolGal=len(pf[1]["target_neg_object_ids"]),
             dlPoolGalFour=len(pf[4]["target_neg_object_ids"]),
             dlRatio=pool["ratio_galaxies_per_lens"],
             dlFresh=sum(pf[k]["n_fresh_negatives"] for k in FOLDS),
             dlHosts="/".join(str(P1["pool"]["existing_negatives_screen_hits_NOT_REMOVED"][str(k)]
                                  ["b_simulation_hosts"]) for k in FOLDS))
    l1, g1 = (int(x) for x in rr["1"]["dann_bnm_training_pool"].split(":"))
    l4_, g4 = (int(x) for x in rr["4"]["dann_bnm_training_pool"].split(":"))
    hold = int(rr["1"]["dann_bnm_holdout_kept_byte_identical"].split("=")[1])
    # the holdout is half lenses: the training pool is the whole pool less half the holdout each
    assert l1 == E["dlPoolLens"] - hold // 2 and g1 == E["dlPoolGal"] - hold // 2
    E.update(dlTrainLens=l1, dlTrainGal=g1, dlTrainLensFour=l4_, dlTrainGalFour=g4, dlHoldout=hold,
             dlTrainFracThree=st.mean(rr[str(k)]["dann_bnm_training_pool_lens_fraction"] for k in FOLDS),
             dlPoolFracThree=1 / (1 + pool["ratio_galaxies_per_lens"]))
    rows = 256
    m = re.search(r"\\newcommand\{\\batchTargetRows\}\{(\d+)\}",
                  open(os.path.join(C.TABLES, "M_batch.tex"), encoding="utf-8").read())
    rows = int(m.group(1))
    E.update(dlBatchLens=round(rows / (1 + pool["ratio_galaxies_per_lens"])), dlBatchLensBal=rows // 2,
             dlBatchLensTrain=round(rows * l1 / (l1 + g1)),
             dlDenserWord=float(f"{2200 / pool['ratio_galaxies_per_lens']:.2g}"),
             dlExposureRatio=(1 + pool["ratio_galaxies_per_lens"]) / 2)
    # v6e: one test lens in the completeness ratio to 0.50 (Section 6.6), one fold and the mean
    folds = _j("data", "splits", "splits.json")["real_lenses_kfold"]["folds"]
    pp = [100 / (len(f["test_lens_rows"]) * 0.50) for f in folds if f["fold"] in FOLDS]
    E["dlLensFoldPpWord"] = {2: "two"}[round(max(pp))]
    E["dlLensMeanPpWord"] = {0.5: "half a"}[round(st.mean(pp) / len(FOLDS) * 2) / 2]
    P2 = _j("runs", "dilute10_prereg_part2.json")
    E["dlRngSteps"] = int(re.search(r"for (\d+) steps", P2["rng_coupling"]["finding"]).group(1))
    P3 = _j("runs", "dilute10_prereg_part3.json")
    d = re.search(r"(\d{4})(\d{2})(\d{2})", P3["coral_nb3_fold1_rerun"]["record"])
    E["dlAbortDate"] = f"{d.group(1)}-{d.group(2)}-{d.group(3)}"
    for nm, p in (("dlShaPartOne", ("runs", "dilute10_prereg_part1.json")),
                  ("dlShaPartTwo", ("runs", "dilute10_prereg_part2.json")),
                  ("dlShaPartThree", ("runs", "dilute10_prereg_part3.json")),
                  ("dlShaPlan", ("runs", "dilute10_analysis", "plan.json")),
                  ("dlShaGapPlan", ("runs", "dann_lambda0_holdout_gap", "plan.json"))):
        E[nm] = integrity.self_certifying_sha(C.R(*p))[:8]
    E["dlReproNWord"] = WORD[len(R["reproduction"]["contrasts"])]
    # the memorisation diagnostic: paired t recomputed from its per-fold readings
    Gp = _j("runs", "dann_lambda0_holdout_gap", "results.json")
    rp = Gp["readings_per_fold"]
    for nm, key in (("mzPrim", "at_4000"), ("mzSec", "within_fold_mean")):
        t = _ptt([rp["dann"][str(k)][key] - rp["lambda0"][str(k)][key] for k in FOLDS])
        for suf, v in (("", t["mean"]), ("Lo", t["lo"]), ("Hi", t["hi"])):
            E[nm + suf] = E[nm + suf + "Four"] = v
    kn = Gp["known_null"]["per_fold"]["1"]
    E["mzViewShift"] = abs(kn["s729_json"] - kn["nuclear_gap"])
    E["mzNckpt"] = len(Gp["per_checkpoint"]["dann"]["1"])
    E["mzHoldout"] = Gp["per_checkpoint"]["dann"]["1"]["4000"]["n_per_slice"]
    return E


# ======================================================================================
def tables(check, V) -> None:
    R = V["R"]
    order = ("dann", "bnm", "coral", "freematch")
    # ---- main-text table ----
    rows = _rows("T_dilute10_main.tex")[1:]                # skip the header row
    assert [r[0] for r in rows] == [LAB[a] for a in order], rows
    for r, arm in zip(rows, order):
        t = V["dil"]["D1"][arm]["tpr"]
        g, d = V["G"]["D1"][(arm, "primary")], V["D"]["D1"][arm]
        exp = [READ[arm], st.mean(t.values()), st.stdev(t.values()), g["mean"], g["lo"], g["hi"],
               g["signs"], 4, d["mean"], d["lo"], d["hi"]]
        got = [n for c in r[1:7] for n in NUM.findall(c.replace("$", ""))]
        for q, e in zip(got, exp):
            check("V6-T", f"T_dilute10_main {arm}", q, e, "recomputed (crosscheck_dilute10)")
        check("V6-T", f"T_dilute10_main {arm} cell count", str(len(exp)), len(got), "table")
        br = "collapses" if g["lo"] <= 0 <= g["hi"] else ("shrinks" if d["lo"] > 0 else "survives")
        check("V6-T", f"T_dilute10_main {arm} branch", "1", float(r[7] == br), "table")
    # ---- Table 2's new block ----
    body = _body("T3v6_contrasts.tex")
    i = [n for n, l in enumerate(body) if "1:10 target" in l and l.startswith(r"\multicolumn")]
    assert len(i) == 1
    blk = body[i[0] + 1:i[0] + 5]
    for l, arm in zip(blk, order):
        g = V["G"]["D1"][(arm, "primary")]
        cells = [c.strip() for c in re.split(r"(?<!\\)&", l.rstrip("\\ "))]
        assert cells[0].startswith(LAB[arm] + r"$^{\S}$"), cells[0]
        got = [n for c in cells[1:] for n in NUM.findall(c.replace("$", ""))]
        exp = [READ[arm], 4000, g["mean"], g["lo"], g["hi"], g["signs"], 4]
        for q, e in zip(got, exp):
            check("V6-T", f"T3v6 1:10 row {arm}", q, e, "recomputed (crosscheck_dilute10)")
        check("V6-T", f"T3v6 1:10 row {arm} cell count", str(len(exp)), len(got), "table")
    # the rest of T3v6 is T3v2 unchanged
    v2 = [l for l in _body("T3v2_contrasts.tex") if not l.startswith("%")]
    v6 = body[:i[0] - 1] + body[i[0] + 5:]
    tab = lambda L: [l for l in L if l.startswith((r"\begin{tabular}", r"\toprule", r"\midrule",  # noqa: E731
                                                     r"\bottomrule", r"\multicolumn")) or "&" in l]
    check("V6-T", "T3v6 = T3v2 outside the 1:10 block", "1", float(tab(v2) == tab(v6)), "table")
    # ---- Appendix K: per-fold ----
    rows = _rows("AK_perfold.tex")[1:]
    exp_rows = [(m, a) for m in METRICS for a in order]
    assert len(rows) == len(exp_rows)
    for r, (m, a) in zip(rows, exp_rows):
        v = [V["dil"]["D1"][a][m][k] for k in FOLDS]
        exp = [READ[a]] + v + [st.mean(v), st.stdev(v)]
        got = [n for c in r[2:] for n in NUM.findall(c.replace("$", ""))]
        for q, e in zip(got, exp):
            check("V6-T", f"AK_perfold {m} {a}", q, e, "recomputed")
        check("V6-T", f"AK_perfold {m} {a} n", str(len(exp)), len(got), "table")
    # ---- Appendix K: the ten intervals ----
    rows = _rows("AK_contrasts4.tex")[1:]
    exp_rows = []
    for a in order:
        exp_rows.append(("G", a, "primary"))
        if a in ("coral", "freematch"):
            exp_rows.append(("G", a, "sensitivity"))
    exp_rows += [("D", a, "secondary") for a in order]
    assert len(rows) == len(exp_rows), (len(rows), len(exp_rows))
    for r, (kind, a, role) in zip(rows, exp_rows):
        x = V["G"]["D1"][(a, role)] if kind == "G" else V["D"]["D1"][a]
        steps = ((READ[a], 4000 if role == "primary" else READ[a]) if kind == "G" else (READ[a], READ[a]))
        exp = list(steps) + [x["mean"], x["lo"], x["hi"], x["signs"], 4]
        got = [n for c in r[1:5] for n in NUM.findall(c.replace("$", "").replace("^{*}", ""))]
        for q, e in zip(got, exp):
            check("V6-T", f"AK_contrasts4 {kind} {a} {role}", q, e, "recomputed")
        check("V6-T", f"AK_contrasts4 {kind} {a} {role} star", "1",
              float(("^{*}" in r[2]) == x["excl"]), "table")
        check("V6-T", f"AK_contrasts4 {kind} {a} {role} role", "1", float(r[5] == role), "table")
    # ---- Appendix K: frozen stages and the decomposition ----
    rows = _rows("AK_frozen.tex")
    fvr = R["descriptive"]["frozen_vs_round1"]
    l4 = st.mean(V["lam"]["D1"][4000]["tpr"].values())
    data = [r for r in rows if r[1] in ("1:1 (main design)", "1:10")]
    assert len(data) == 6, data
    for r, (key, tgt) in zip(data[:4], [("FreeMatch", "r1"), ("FreeMatch", "dl"),
                                        ("CORAL", "r1"), ("CORAL", "dl")]):
        v = list(V["r1"][key].values()) if tgt == "r1" else [V["fz"][key]["tpr"][k] for k in FOLDS]
        # r3: the AUROC column, fold-mean, after the TPR mean
        auc = V["r1_auroc"][key].values() if tgt == "r1" else V["fz"][key]["auroc"].values()
        exp = v + [st.mean(v), st.mean(auc)]
        if tgt == "dl":
            agg = fvr[key]["aggregate"]
            agg = json.loads(agg) if isinstance(agg, str) else agg
            exp += [st.mean(v) - st.mean(V["r1"][key].values())] + list(agg["tpr"]["ci"])
        got = [n for c in r[2:] for n in NUM.findall(c.replace("$", ""))]
        for q, e in zip(got, exp):
            check("V6-T", f"AK_frozen {key} {tgt}", q, e, "recomputed / results.json bootstrap")
        check("V6-T", f"AK_frozen {key} {tgt} n", str(len(exp)), len(got), "table")
    for r, (st_v, un) in zip(data[4:], [(st.mean(V["r1"]["FreeMatch"].values()),
                                         st.mean(V["bal"]["D1"]["freematch"]["tpr"].values())),
                                        (st.mean(V["fz"]["FreeMatch"]["tpr"].values()),
                                         st.mean(V["dil"]["D1"]["freematch"]["tpr"].values()))]):
        exp = [st_v, st_v - l4, un - st_v, un - l4]
        got = [n for c in r[2:] for n in NUM.findall(c.replace("$", ""))]
        for q, e in zip(got, exp):
            check("V6-T", f"AK_frozen decomposition {r[1]}", q, e, "recomputed")
    # ---- Appendix K: head norms ----
    H = _j("runs", "dilute10_head_norms", "head_norms.json")["heads"]
    rows = _rows("AK_head_norms.tex")[1:]
    exp_rows = [(a, s, q) for a in ("coral", "freematch") for s in ("round1", "diluted")
                for q in ("ln_gain_norm", "ln_bias_norm", "head_l2")]
    assert len(rows) == len(exp_rows)
    for r, (a, s, q) in zip(rows, exp_rows):
        got = [n for c in r[3:] for n in NUM.findall(c.replace("$", ""))]
        for k, g in zip(FOLDS, got):
            check("V6-T", f"AK_head_norms {a} {s} {q} f{k}", g, H[a][s][str(k)][q],
                  "runs/dilute10_head_norms/head_norms.json")
    for k, (q, v) in enumerate([("ln_gain_norm", (97.0, 103.7, 98.2, 102.2)),
                                ("head_l2", (134.2, 139.0, 133.4, 137.1))]):
        for f, x in zip(FOLDS, v):
            check("7.46", f"CORAL diluted {q} f{f}", f"{x:.1f}", H["coral"]["diluted"][str(f)][q],
                  "runs/dilute10_head_norms/head_norms.json")
    # ---- Appendix K: predictions (text from results.json, row for row) ----
    rows = _rows("AK_predictions.tex")[1:]
    comps = R["predictions"]["components"]
    SRC = "runs/dilute10_analysis/results.json"
    check("V6-T", "AK_predictions rows = scored components", str(len(comps)), len(rows), SRC)
    for r, c in zip(rows, comps):
        check("V6-T", f"AK_predictions {c['id']} id+score", "1",
              float(r[0] == c["id"] and r[2] == c["score"]), "runs/dilute10_analysis/results.json")
        nums_tab = [float(x) for x in NUM.findall(r[3].replace("$-$", "-").replace("$", ""))]
        # v6e: the table says "main design" where results.json says "round 1" (no number there)
        src = c["rests_on"].replace("round-1", "main-design").replace("round 1", "main design")
        nums_src = [float(x) for x in NUM.findall(src)]
        check("V6-T", f"AK_predictions {c['id']} rests-on numbers", "1",
              float(nums_tab == nums_src), "runs/dilute10_analysis/results.json")
    # ---- Appendix K: sensitivities ----
    rows = _rows("AK_sensitivity.tex")[1:]
    assert len(rows) == len(exp_rows := [("G", a, ro) for a in order for ro in
                                         (("primary", "sensitivity") if a in ("coral", "freematch")
                                          else ("primary",))] + [("D", a, "") for a in order]), rows
    for r, (kind, a, role) in zip(rows, exp_rows):
        xs = [(V["G"][p][(a, role)] if kind == "G" else V["D"][p][a]) for p in ("D1", "orig", "D2")]
        exp_c = "excl." if xs[0]["excl"] else "incl."
        check("V6-T", f"AK_sensitivity {kind} {a} {role} cleaned", "1", float(r[2] == exp_c), "table")
        for col, x in ((3, xs[1]), (5, xs[2])):
            got = NUM.findall(r[col].replace("$", ""))
            for q, e in zip(got, (x["mean"], x["lo"], x["hi"])):
                check("V6-T", f"AK_sensitivity {kind} {a} {role} col{col}", q, e, "recomputed")
        for col, x in ((4, xs[1]), (6, xs[2])):
            check("V6-T", f"AK_sensitivity {kind} {a} {role} match{col}", "1",
                  float((r[col] == "yes") == (x["excl"] == xs[0]["excl"])), "recomputed")
    # ---- Appendix K: bookkeeping ----
    rows = _rows("AK_bookkeeping.tex")[1:]
    bk = R["descriptive"]["bookkeeping"]
    cn = V["cn"]
    sk_rows = [r for r in rows if r[1] == "skips / applied"]
    for r in sk_rows:
        run = {"DANN": "dann", "CORAL": "coral_nb3", "FreeMatch": "freematch_nb3"}[r[0]]
        for k, c in zip(FOLDS, r[2:]):
            s, a = NUM.findall(c.replace("$", ""))
            es, ea = _skips(run, k)
            check("V6-T", f"AK_bookkeeping {run} f{k} skips", s, es, "run steps.csv")
            check("V6-T", f"AK_bookkeeping {run} f{k} applied", a, ea, "run steps.csv")
    # v6e: each run's "mean $\pm$ sd" row is followed by its "expected" row
    lf_idx = [i for i, r in enumerate(rows) if r[1] == r"lens fraction, mean $\pm$ sd"]
    check("V6-T", "AK_bookkeeping lens-fraction rows", "6", len(lf_idx), "table")
    for i in lf_idx:
        r, rexp = rows[i], rows[i + 1]
        assert rexp[0] == "" and rexp[1] == "expected", rexp
        run = {"DANN": "dann", "BNM": "bnm", "CORAL": "coral_nb3", "FreeMatch": "freematch_nb3",
               "CORAL frozen": "coral_frozen", "FreeMatch frozen": "freematch_frozen"}[r[0]]
        for k, c, ce in zip(FOLDS, r[2:], rexp[2:]):
            m_, sd_ = NUM.findall(c.replace("$", "").replace(r"\pm", " "))
            (e_,) = NUM.findall(ce.replace("$", ""))
            sm = _j("runs", "dilute10", run, f"fold{k}", "target_lens_fraction_summary.json")
            flat = json.dumps(sm)
            mean = [float(x) for x in re.findall(r'"(?:run_mean|mean)": ([0-9.]+)', flat)]
            check("V6-T", f"AK_bookkeeping {run} f{k} lens fraction", m_, mean[0] if mean else None,
                  f"runs/dilute10/{run}/fold{k}/target_lens_fraction_summary.json")
            # v6e: the sd beside the mean, recomputed from the run's per-batch log
            lf = [float(x["lens_fraction"]) for x in csv.DictReader(open(
                C.R("runs", "dilute10", run, f"fold{k}", "target_lens_fraction.csv"), encoding="utf-8"))]
            check("V6-T", f"AK_bookkeeping {run} f{k} lens fraction sd", sd_, st.stdev(lf),
                  f"runs/dilute10/{run}/fold{k}/target_lens_fraction.csv")
            check("V6-T", f"AK_bookkeeping {run} f{k} expected", e_,
                  bk["lens_fraction"][f"{run}/fold{k}"]["expected"], SRC)
    # r3: the guard's input at step 1, its smallest value and its threshold, recomputed
    for j, side in enumerate(("source", "target")):
        r = [r for r in rows if r[1] == f"norm at step 1, {side}"][0]
        assert r[0] == ("CORAL" if side == "source" else "")
        for k, c, both in zip(FOLDS, r[2:], cn["start_both"]):
            check("V6-T", f"AK_bookkeeping CORAL f{k} step-1 norm {side}", c.replace("$", ""),
                  both[j], "steps.csv")
    r = [r for r in rows if r[1] == "smallest norm"][0]
    for k, c, e in zip(FOLDS, r[2:], cn["feat_min"]):
        check("V6-T", f"AK_bookkeeping CORAL f{k} smallest norm", c.replace("$", ""), e, "steps.csv")
    r = [r for r in rows if r[1] == "guard threshold"][0]
    assert len(cn["ceiling"]) == 1
    for k, c in zip(FOLDS, r[2:]):
        check("V6-T", f"AK_bookkeeping CORAL f{k} guard threshold", c.replace("$", ""),
              min(cn["ceiling"]), "dilute10_feature_norm_advisory.csv")
    adv = [r for r in rows if r[1] == "advisory firings"][0]
    for k, c in zip(FOLDS, adv[2:]):
        check("V6-T", f"AK_bookkeeping advisory f{k}", c, cn["fires"][k - 1],
              "dilute10_feature_norm_advisory.csv")
    # ---- Appendix K: memorisation ----
    Gp = _j("runs", "dann_lambda0_holdout_gap", "results.json")
    rp = Gp["readings_per_fold"]
    mx = Gp["any_checkpoint_max_descriptive"]["values"]
    rows = _rows("AK_memorisation.tex")[1:]
    exp = []
    for key in ("at_4000", "within_fold_mean"):
        for arm in ("dann", "lambda0"):
            v = [rp[arm][str(k)][key] for k in FOLDS]
            exp.append(v + [st.mean(v)])
        dd = [rp["dann"][str(k)][key] - rp["lambda0"][str(k)][key] for k in FOLDS]
        t = _ptt(dd)
        exp.append(dd + [t["mean"], t["lo"], t["hi"]])
    for arm in ("dann", "lambda0"):
        exp.append([mx[arm][str(k)]["value"] for k in FOLDS])
        exp.append([mx[arm][str(k)]["step"] for k in FOLDS])
    assert len(rows) == len(exp), (len(rows), len(exp))
    for r, e in zip(rows, exp):
        got = [n for c in r[2:] for n in NUM.findall(c.replace("$", ""))]
        check("V6-T", f"AK_memorisation row '{r[1]}' n", str(len(e)), len(got), "table")
        for q, x in zip(got, e):
            check("V6-T", f"AK_memorisation {r[1]}", q, x,
                  "runs/dann_lambda0_holdout_gap/results.json (paired t recomputed)")
    for nm, ck, key in (("primary", "primary_at_4000", "at_4000"),
                        ("secondary", "secondary_within_fold_mean", "within_fold_mean")):
        t = _ptt([rp["dann"][str(k)][key] - rp["lambda0"][str(k)][key] for k in FOLDS])
        r = Gp["contrast_dann_minus_lambda0"][ck]
        check("7.54", f"DANN - lambda=0 {nm} (results.json)", f"{r['mean']:.6f}", t["mean"],
              "runs/dann_lambda0_holdout_gap/results.json")
    check("7.54", "DANN - lambda=0 primary, as 7.54 prints it", "+0.0987",
          _ptt([rp["dann"][str(k)]["at_4000"] - rp["lambda0"][str(k)]["at_4000"] for k in FOLDS])["mean"],
          "runs/dann_lambda0_holdout_gap/results.json")
    # ---- the relabelled v6 copies: identical to their sources but for the label swap ----
    swaps = {"A7_prereg_pool_v6.tex": ("A7_prereg_pool_v2.tex", [("tab:contrasts6", "tab:contrasts2"),
                                                                 ("tab:prereg_pool6", "tab:prereg_pool2")]),
             "C_k8_pairs_v6.tex": ("C_k8_pairs.tex", [("tab:contrasts6", "tab:contrasts2"),
                                                      ("tab:k8pairs6", "tab:k8pairs")])}
    for dst, (src, pairs) in swaps.items():
        a = "\n".join(l for l in _body(dst) if l.strip())
        for new, old in pairs:
            a = a.replace(new, old)
        b = "\n".join(l for l in _body(src) if l.strip())
        check("V6-T", f"{dst} = {src} but for its label swap", "1", float(a == b), "table")
    h6 = [l for l in _body("A2_hashes_v6.tex")
          if not (l.startswith("1:10 ") or l.startswith(r"DANN $-$ $\lambda=0$ memorisation")
                  or "1:10 diluted-target experiment" in l or "Analysis declarations" in l
                  or l.startswith(r"\caption") or l.startswith(r"\label"))]
    h2 = [l for l in _body("A2_hashes_v2.tex") if not (l.startswith(r"\caption") or l.startswith(r"\label"))]
    # v6b: the notes name the first 1:10 run (CORAL's frozen stage, fold 1, run locally)
    import _dilute10 as X  # run_record: the run's own record, as the table generator reads it
    first_note = " The first 1:10 run was CORAL's frozen stage on fold 1, run locally."
    check("V6-T", "A2_hashes_v6 notes name the first 1:10 run", "1",
          float(sum(l.count(first_note) for l in h6) == 1), "table")
    first = min((X.run_record(a, k).get("started_utc")
                 or (X.run_record(a, k).get("dilute10") or {}).get("recorded_utc"), a, k)
                for a in ("coral_frozen", "freematch_frozen", "dann", "bnm", "coral_nb3",
                          "freematch_nb3") for k in FOLDS)
    check("V6-T", "first 1:10 run is CORAL frozen, fold 1", "1",
          float(first[1:] == ("coral_frozen", 1)), "runs/dilute10/*/fold*/run record")
    h6 = [l.replace(first_note, "") for l in h6]
    h6 = [l.replace("tab:hashes6", "tab:hashes2").replace("other thirteen", "other seven") for l in h6]
    check("V6-T", "A2_hashes_v6 = A2_hashes_v2 outside the 1:10 rows and the counts", "1",
          float([l for l in h6 if l.strip() and l != r"\midrule"] ==
                [l for l in h2 if l.strip() and l != r"\midrule"]), "table")

    # ---- Appendix B: the new hash rows ----
    inv = {r["path"]: r for r in json.load(open(integrity.INVENTORY, encoding="utf-8"))["artifacts"]}
    body = _body("A2_hashes_v6.tex")
    want = {"1:10 pool definition": "data/splits/dilute10_pool.json",
            "1:10 pre-registration, part 1": "runs/dilute10_prereg_part1.json",
            "1:10 pre-registration, part 2": "runs/dilute10_prereg_part2.json",
            "1:10 pre-registration, part 3": "runs/dilute10_prereg_part3.json",
            "1:10 analysis plan": "runs/dilute10_analysis/plan.json",
            r"DANN $-$ $\lambda=0$ memorisation plan (retrospective)": "runs/dann_lambda0_holdout_gap/plan.json"}
    for label, rel in want.items():
        line = [l for l in body if l.startswith(label + " &")]
        assert len(line) == 1, label
        cells = [c.strip() for c in re.split(r"(?<!\\)&", line[0].rstrip("\\ "))]
        p = C.R(*rel.split("/"))
        flav = inv[rel]["flavour"]
        got = integrity.self_certifying_sha(p) if flav == "self_certifying" else integrity.whole_file_sha(p)
        check("V6-T", f"A2_hashes_v6 {rel} digest", "1",
              float(cells[6] == f"\\texttt{{{got[:16]}}}" and got == inv[rel]["sha256"]),
              "scripts/integrity.py")
        j = json.load(open(p, encoding="utf-8"))
        rec = j.get("recorded_utc")
        check("V6-T", f"A2_hashes_v6 {rel} recorded", "1",
              float(cells[1] == (rec.replace("T", " ")[:16] if rec else "---")), rel)
    # the first governed runs, from the run records
    starts = []
    for arm in ("coral_frozen", "freematch_frozen", "dann", "bnm", "coral_nb3", "freematch_nb3"):
        for k in FOLDS:
            for nm in ("run.json", "run_config.json"):
                p = C.R("runs", "dilute10", arm, f"fold{k}", nm)
                if os.path.exists(p):
                    rj = json.load(open(p, encoding="utf-8"))
                    starts.append((rj.get("started_utc") or rj["dilute10"]["recorded_utc"], arm))
                    break
    first_all = min(s for s, _ in starts)
    first_p3 = min(s for s, a in starts if a in ("coral_nb3", "freematch_nb3"))
    for label, first in (("1:10 pre-registration, part 1", first_all),
                         ("1:10 pre-registration, part 2", first_all),
                         ("1:10 pool definition", first_all),
                         ("1:10 pre-registration, part 3", first_p3)):
        line = [l for l in body if l.startswith(label + " &")][0]
        cells = [c.strip() for c in re.split(r"(?<!\\)&", line.rstrip("\\ "))]
        check("V6-T", f"A2_hashes_v6 {label} first run", "1",
              float(cells[3] == first.replace("T", " ")[:16] and cells[4] == "before"), "run records")
