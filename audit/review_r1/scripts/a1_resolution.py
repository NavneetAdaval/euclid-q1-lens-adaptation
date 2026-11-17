"""
a1_resolution.py -- review round 1, Part A1: the §7 resolution limit, redone.
POST HOC AND EXPLORATORY.

    python -B audit/review_r1/scripts/a1_resolution.py

Per-checkpoint TPR@1e-3 series (D1 pool) for every arm whose per-checkpoint logits are
stored: the fold-0 development runs on §7's 32-point 250..8000 grid, and folds 1-4 on each
arm's own stored grid. Window combination and positions reuse
``scripts/analyze_window_average`` (``evaluate`` / ``mean_logit``) unchanged.

Writes out/A1_availability.json, out/A1_series.json, out/A1_resolution.json and
fig/fig5_detrended_draft.png.
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_common as RC                                   # noqa: E402

C = RC.C
SCRIPT = os.path.abspath(__file__)
POOL = "D1"
KS = [1, 2, 4, 8]

F0_NAME = {"coral": "CORAL x nb3", "freematch": "FreeMatch x nb3",
           "dann": "DANN arm 2 (lambda live)", "lambda0": "DANN arm 1 (lambda=0)",
           "bnm": "BNM x nb3"}
PAPER_RAW = {"coral": ("wzCoralSdOne", 0.0529), "freematch": ("wzFreeMatchSdOne", 0.0588),
             "dann": ("wzDannSdOne", 0.1123), "lambda0": ("wzLambdaSdOne", 0.0493)}
SERIES_ARMS = ("lambda0", "bnm", "freematch", "dann", "coral")
PAIRS = [("dann", "coral"), ("dann", "freematch"), ("coral", "freematch"),
         ("bnm", "coral"), ("bnm", "dann"), ("bnm", "freematch")]


# ------------------------------------------------------------------ estimators
def sd_raw(x):
    return st.stdev(x) if len(x) > 1 else None


def sd_diff(x):
    """(a) sd of successive differences / sqrt(2). Needs >= 3 points."""
    d = np.diff(np.asarray(x, float))
    return float(np.std(d, ddof=1) / math.sqrt(2)) if d.size >= 2 else None


def lin_resid(steps, x):
    s, x = np.asarray(steps, float), np.asarray(x, float)
    b = np.polyfit(s, x, 1)
    return x - np.polyval(b, s), b


def sd_resid(steps, x):
    """(b) sd of residuals after a least-squares line in step, ddof = 2. Needs >= 3 points."""
    if len(x) < 3:
        return None
    e, _ = lin_resid(steps, x)
    return float(math.sqrt((e ** 2).sum() / (len(x) - 2)))


def lag1(steps, x):
    """Lag-1 autocorrelation of the linear-fit residuals: sum e_t e_{t+1} / sum e_t^2."""
    if len(x) < 3:
        return None
    e, _ = lin_resid(steps, x)
    den = float((e ** 2).sum())
    return float((e[:-1] * e[1:]).sum() / den) if den > 0 else None


def stats(steps, x):
    return {"n": len(x), "raw_sd": sd_raw(x), "detrended_a_diff": sd_diff(x),
            "detrended_b_linfit_resid": sd_resid(steps, x), "lag1_autocorr_resid": lag1(steps, x)}


# ------------------------------------------------------------------ loading
def load_series(paths, fold, arm, I, grid=None):
    steps = sorted(s for s in paths if grid is None or s in grid)
    loaded = {}
    for s in steps:
        v = RC.load_vec(paths[s], fold, arm, POOL, I)
        loaded[s] = (v["pos"], v["neg"])
    return loaded


def tpr_window(loaded, w):
    return float(C._AWA.evaluate(loaded, w, "mean_logit")["tpr"])


def main():
    I = RC.Inputs()
    import analyze_window_fold0 as AWF
    I.add(RC.R("scripts", "analyze_window_fold0.py"), RC.R("scripts", "analyze_window_average.py"))
    GRID = list(AWF.GRID)

    # ---------------- availability + series ----------------
    avail, series, loaded_all = {}, {}, {}
    f0 = RC.fold0_dev_paths()
    for arm in SERIES_ARMS:
        name = F0_NAME[arm]
        ld = load_series(f0[name], 0, arm, I, grid=set(GRID))
        loaded_all[(arm, 0)] = ld
        st_ = sorted(ld)
        series[f"{arm}|fold0"] = {"run": name, "steps": st_,
                                  "tpr": [tpr_window(ld, [s]) for s in st_]}
        avail[f"{arm}|fold0"] = {"run": name, "stored_steps_total": len(f0[name]),
                                 "used_grid": "250..8000 step 250 (32 points, §7's grid)",
                                 "n_used": len(st_)}
    for arm in SERIES_ARMS:
        for f in RC.CONF_FOLDS:
            paths = RC.step_paths(arm, f)
            ld = load_series(paths, f, arm, I)
            loaded_all[(arm, f)] = ld
            st_ = sorted(ld)
            series[f"{arm}|fold{f}"] = {"steps": st_, "tpr": [tpr_window(ld, [s]) for s in st_]}
            cad = sorted(set(np.diff(st_).tolist()))
            avail[f"{arm}|fold{f}"] = {"n_steps": len(st_), "first": st_[0], "last": st_[-1],
                                       "cadence": cad}
    for f in (0,) + RC.CONF_FOLDS:
        avail[f"baseline|fold{f}"] = {
            "n_steps": 1, "steps": [8000],
            "note": "only step 8000's scores are stored; earlier steps exist only as head "
                    "checkpoints, which this brief may not load -> no series"}

    # ---------------- reproduction of the paper's fold-0 raw sd ----------------
    import re
    mp = open(I.add(RC.R("paper", "tables", "M_prose.tex")), encoding="utf-8").read()
    wf = __import__("json").load(open(I.add(C.art("window_fold0")), encoding="utf-8"))
    repro = {}
    for arm, (macro, val) in PAPER_RAW.items():
        got = re.search(r"\\newcommand\{\\" + macro + r"\}\{\$?([0-9.]+)\$?\}", mp).group(1)
        mine = sd_raw(series[f"{arm}|fold0"]["tpr"])
        stored = wf["arms"][F0_NAME[arm]]["1"]["sd"]
        repro[arm] = {"paper_macro": macro, "paper_value": got, "brief_value": val,
                      "recomputed": mine, "stored_window_fold0_D1": stored,
                      "equal_to_stored_exactly": mine == stored,
                      "matches_paper_4dp": f"{mine:.4f}" == got == f"{val:.4f}"}

    # ---------------- per-series stats ----------------
    per = {k: stats(v["steps"], v["tpr"]) for k, v in series.items()}
    sigma = {}
    for arm in SERIES_ARMS:
        rec = {"fold0": per[f"{arm}|fold0"]}
        for est in ("raw_sd", "detrended_a_diff", "detrended_b_linfit_resid",
                    "lag1_autocorr_resid"):
            vals = [per[f"{arm}|fold{f}"][est] for f in RC.CONF_FOLDS]
            rec[f"folds1_4_mean_{est}"] = (float(np.mean(vals))
                                          if all(v is not None for v in vals) else None)
            rec[f"folds1_4_per_fold_{est}"] = vals
        sigma[arm] = rec

    # ---------------- gaps and like-for-like ratios ----------------
    def fm(arm, step):
        vals = []
        for f in RC.CONF_FOLDS:
            ld = loaded_all[(arm, f)]
            if step not in ld:
                return None
            vals.append(tpr_window(ld, [step]))
        return float(np.mean(vals))
    declared_readouts = {k: list(C.READOUTS[k]) for k in SERIES_ARMS}
    gaps = {}
    for a, b in PAIRS:
        decl = (RC.DA_HEADLINE[a], RC.DA_HEADLINE[b])
        m = min(decl)
        for kind, (sa, sb) in (("declared_budget", decl), ("budget_matched", (m, m))):
            ga, gb = fm(a, sa), fm(b, sb)
            if ga is None or gb is None:
                gaps[f"{a}-{b}|{kind}"] = {"steps": [sa, sb], "available": False}
                continue
            gap = ga - gb
            rec = {"steps": [sa, sb], "available": True, "fold_mean_a": ga, "fold_mean_b": gb,
                   "gap": gap,
                   "steps_are_declared_readouts": [sa in declared_readouts[a],
                                                   sb in declared_readouts[b]]}
            for est in ("detrended_a_diff", "detrended_b_linfit_resid"):
                sA = sigma[a][f"folds1_4_mean_{est}"]
                sB = sigma[b][f"folds1_4_mean_{est}"]
                noise = math.sqrt(sA ** 2 + sB ** 2) / 2
                rec[f"noise_on_fold_mean_diff_{est}"] = noise
                rec[f"ratio_{est}"] = noise / abs(gap)
                rec[f"sigma_source_{est}"] = "folds 1-4 (mean of per-fold detrended sd)"
            rec["old_ratio_raw_fold0_sd_over_gap"] = {
                a: sigma[a]["fold0"]["raw_sd"] / abs(gap),
                b: sigma[b]["fold0"]["raw_sd"] / abs(gap)}
            rec["old_ratio_inputs_raw_fold0_sd"] = {a: sigma[a]["fold0"]["raw_sd"],
                                                    b: sigma[b]["fold0"]["raw_sd"]}
            gaps[f"{a}-{b}|{kind}"] = rec

    # ---------------- window averaging on detrended series ----------------
    win = {}
    for arm in SERIES_ARMS:
        win[arm] = {}
        for scope, folds in (("fold0", (0,)), ("folds1_4", RC.CONF_FOLDS)):
            win[arm][scope] = {}
            for k in KS:
                rec_f = {}
                for f in folds:
                    ld = loaded_all[(arm, f)]
                    steps = sorted(ld)
                    if len(steps) < k:
                        rec_f[str(f)] = {"n_positions": 0}
                        continue
                    pos_list = [steps[i - k + 1:i + 1] for i in range(k - 1, len(steps))]
                    end = [w[-1] for w in pos_list]
                    vk = [tpr_window(ld, w) for w in pos_list]
                    v1 = [tpr_window(ld, [w[-1]]) for w in pos_list]
                    rec_f[str(f)] = {"n_positions": len(pos_list),
                                     "n_successive_diffs": max(len(pos_list) - 1, 0),
                                     "sd_k_a": sd_diff(vk), "sd_1_a_same_positions": sd_diff(v1),
                                     "sd_k_b": sd_resid(end, vk),
                                     "sd_1_b_same_positions": sd_resid(end, v1),
                                     "sd_k_raw": sd_raw(vk), "sd_1_raw_same_positions": sd_raw(v1)}
                out = {"per_fold": rec_f, "sqrt_k": math.sqrt(k),
                       "white_noise_expectation_for_estimator_a": float(k),
                       "n_positions_per_fold": [rec_f[str(f)]["n_positions"] for f in folds]}
                for e in ("a", "b", "raw"):
                    kk = [rec_f[str(f)].get(f"sd_k_{e}") for f in folds]
                    k1 = [rec_f[str(f)].get(f"sd_1_{e}_same_positions") for f in folds]
                    if all(x is not None for x in kk + k1) and kk:
                        mk, m1 = float(np.mean(kk)), float(np.mean(k1))
                        out[f"sd_k_{e}_foldmean"] = mk
                        out[f"sd_1_{e}_foldmean"] = m1
                        out[f"reduction_{e}"] = m1 / mk if mk > 0 else None
                        out[f"reduction_{e}_over_sqrt_k"] = (m1 / mk / math.sqrt(k)
                                                             if mk > 0 else None)
                    else:
                        out[f"reduction_{e}"] = None
                if out.get("reduction_a") is not None and k > 1 and \
                        out["reduction_a"] > math.sqrt(k):
                    out["exceeds_sqrt_k_estimator_a"] = True
                    out["lag1_autocorr_of_k1_series"] = (
                        sigma[arm]["fold0"]["lag1_autocorr_resid"] if scope == "fold0"
                        else sigma[arm]["folds1_4_mean_lag1_autocorr_resid"])
                    out["lag1_autocorr_of_k1_series_per_fold"] = (
                        None if scope == "fold0"
                        else sigma[arm]["folds1_4_per_fold_lag1_autocorr_resid"])
                elif k > 1:
                    out["exceeds_sqrt_k_estimator_a"] = (False if out.get("reduction_a")
                                                         is not None else None)
                win[arm][scope][str(k)] = out

    # window-averaged sigma for the like-for-like ratio vs k (figure panel a)
    ratio_k = {}
    for key, g in gaps.items():
        if not g.get("available"):
            continue
        a, b = key.split("|")[0].split("-")
        row = {}
        for k in KS:
            sA = win[a]["folds1_4"][str(k)].get("sd_k_a_foldmean")
            sB = win[b]["folds1_4"][str(k)].get("sd_k_a_foldmean")
            row[str(k)] = (math.sqrt(sA ** 2 + sB ** 2) / 2 / abs(g["gap"])
                           if sA is not None and sB is not None else None)
        ratio_k[key] = row

    RC.write_json("A1_availability.json", {"grids": avail}, script=SCRIPT, inputs=I, pool=POOL,
                  folds=[0, 1, 2, 3, 4])
    RC.write_json("A1_series.json", {"series": series}, script=SCRIPT, inputs=I, pool=POOL,
                  folds=[0, 1, 2, 3, 4],
                  notes=["tpr = analyze_window_average.evaluate(single checkpoint, mean_logit)"])
    RC.write_json("A1_resolution.json", {
        "reproduction_fold0_raw_sd": repro,
        "per_series": per,
        "sigma_by_arm": sigma,
        "gaps_and_ratios": gaps,
        "ratio_vs_k_estimator_a_window_averaged": ratio_k,
        "window_averaging": win,
        "definitions": {
            "detrended_a_diff": "sd(successive differences, ddof=1)/sqrt(2)",
            "detrended_b_linfit_resid": "sqrt(sum(resid^2)/(n-2)) after a least-squares line "
                                        "in step",
            "lag1_autocorr_resid": "sum(e_t e_t+1)/sum(e_t^2) of the linear-fit residuals",
            "sigma_folds1_4": "mean over folds 1-4 of the per-fold detrended sd",
            "noise_on_fold_mean_diff": "sqrt(sigma_A^2 + sigma_B^2) / 2 (2 = sqrt of 4 folds)",
            "ratio": "noise_on_fold_mean_diff / |gap|",
            "old_ratio": "raw fold-0 32-point checkpoint sd of each arm / |gap|",
            "gap": "fold-mean TPR difference, folds 1-4, D1, single checkpoints",
            "window": "k consecutive stored checkpoints, mean_logit combination, every valid "
                      "position (analyze_window_average's scatter-reduction positions); "
                      "sd_1 taken at the same positions (each window's last checkpoint)",
            "white_noise_expectation_for_estimator_a": "for an overlapping k-window mean of "
                "independent noise, successive window means differ by (x_new - x_old)/k, so "
                "estimator (a) shrinks by k, not sqrt(k); this column states that analytic "
                "expectation and is not a measurement"},
    }, script=SCRIPT, inputs=I, pool=POOL, folds=[0, 1, 2, 3, 4],
        notes=["Baseline A has no per-checkpoint series (only step 8000 stored)."])
    make_figure(gaps, ratio_k, win)


def make_figure(gaps, ratio_k, win):
    plt = C.use_paper_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 4.2))
    col = {"dann-coral": "#D55E00", "dann-freematch": "#E69F00", "coral-freematch": "#009E73",
           "bnm-coral": "#56B4E9", "bnm-dann": "#0072B2", "bnm-freematch": "#CC79A7"}
    lab = {"dann": "DANN", "coral": "CORAL", "freematch": "FreeMatch", "bnm": "BNM",
           "lambda0": r"$\lambda{=}0$"}
    for key, row in ratio_k.items():
        pair, kind = key.split("|")
        xs = [k for k in KS if row[str(k)] is not None]
        ys = [row[str(k)] for k in xs]
        a, b = pair.split("-")
        g = gaps[key]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.1 if kind == "declared_budget" else 0.8,
                ls="-" if kind == "declared_budget" else (0, (3, 2)), color=col[pair],
                label=(f"{lab[a]}$-${lab[b]} ({g['steps'][0]}/{g['steps'][1]})"))
    ax.axhline(1.0, color="k", lw=0.9, ls=(0, (4, 2)))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("checkpoints averaged, $k$")
    ax.set_ylabel(r"$\sqrt{\sigma_A^2+\sigma_B^2}/2\ /\ |$gap$|$")
    ax.set_title("(a) like-for-like: detrended noise on a 4-fold mean / gap\n"
                 "solid = declared budgets, dashed = budget-matched", fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=5.4, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.17))
    ax.grid(color="#EEEEEE", lw=0.5)
    kk = KS
    bx.plot(kk, [math.sqrt(k) for k in kk], color="k", lw=0.9, ls=(0, (4, 2)),
            label=r"$\sqrt{k}$")
    bx.plot(kk, kk, color="#999999", lw=0.9, ls=(0, (1, 1.5)),
            label=r"$k$ (white noise, estimator a)")
    for arm, ck in (("lambda0", "lambda0"), ("bnm", "bnm"), ("freematch", "freematch"),
                    ("dann", "dann"), ("coral", "coral")):
        r = win[arm]["folds1_4"]
        xs = [k for k in KS if r[str(k)].get("reduction_a") is not None]
        bx.plot(xs, [r[str(k)]["reduction_a"] for k in xs], marker=C.ARM_MARKER[ck], ms=3.5,
                lw=1.1, color=C.ARM_COLOUR[ck], label=lab[arm] + " (folds 1-4)")
    bx.set_xscale("log", base=2)
    bx.set_xticks(KS)
    bx.set_xticklabels([str(k) for k in KS])
    bx.set_xlabel("checkpoints averaged, $k$")
    bx.set_ylabel(r"detrended reduction $\sigma^{(a)}_1/\sigma^{(a)}_k$")
    bx.set_title("(b) achieved reduction, estimator (a)", fontsize=7, loc="left")
    bx.legend(frameon=False, fontsize=5.6, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.17))
    bx.grid(color="#EEEEEE", lw=0.5)
    fig.text(0.01, 0.005, "DRAFT -- post hoc, exploratory. D1 pool. Not the paper's Figure 5.",
             fontsize=6, color="#AA0000")
    fig.subplots_adjust(wspace=0.32, bottom=0.33, top=0.88, left=0.08, right=0.98)
    os.makedirs(RC.FIG, exist_ok=True)
    p = os.path.join(RC.FIG, "fig5_detrended_draft.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"  wrote {RC.rel(p)}")


if __name__ == "__main__":
    main()
