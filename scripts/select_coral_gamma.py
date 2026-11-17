"""
select_coral_gamma.py
=====================
Solve Sun & Saenko's magnitude-matching rule on the round-4 CORAL gamma sweep traces,
apply the PRE-REGISTERED decision rule, and write the result as the pre-registration for
the confirmatory runs. Local, CPU, read-only. Launches nothing.

WHY THE RULE IS WELL-POSED HERE (the earlier objection, and why it no longer applies)
=====================================================================================
The 500-step calibration objection was a SCALE-FEEDBACK argument: raw CORAL scales as
(feature scale)^4, gamma moves the scale, so "equalise the losses at the end of training"
chases a moving target and has no fixed point. That objection was CORRECT FOR THE DATA IT
WAS MADE ON -- with the target EMA ON, nb3's equalising gamma HALVED across its own tail
window (1.138 -> 0.629, CV 0.868) while the feature norm rose 25% in 400 steps.

The sweep contradicts it. Across a 16x range of gamma the source feature norm moves by
only -0.89, -1.08, -1.63 (about -1.4% to -2.5% of a ~64 norm). The scale is effectively
INVARIANT to gamma, so the feedback loop the objection depended on is not operating, and
R(g) = g * coral_raw(g) / sup_loss(g) is a well-behaved function of gamma that can be
solved for R = 1.

The difference between the two runs is round 4's Change 1: the target EMA is now OFF. The
stale-EMA reference is the most likely cause of the scale growth that motivated the
objection -- consistent with the measured 6.0e+03 vs 8.6e-07 translation-invariance
breakage. Recorded here as the reason the rule is APPLIED rather than re-litigated.

The shrink pressure the eps^4 argument predicts IS present and IS monotone in gamma
(-0.89, -1.08, -1.63 at 1/4/16). It is small over the swept range but it is real and it
grows with gamma, which is why the decision rule's branch C refuses to extrapolate past 64.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(ROOT, ".."))
SWEEP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "kaggle", "coral_r4", "sweep_run", "runs", "coral_r4_sweep")
OUT = os.path.join(REPO, "runs", "coral_r4_gamma_selection.json")
GAMMAS, TAIL, WINDOW = (1.0, 4.0, 16.0), 200, 2000


def col(rows, k):
    return [float(r[k]) for r in rows if r.get(k) not in (None, "")]


def stats(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    h = n // 2
    f = sum(v[:h]) / h
    s = sum(v[h:]) / (n - h)
    return {"mean": m, "sd": sd, "cv": (sd / m if m else float("nan")),
            "first_half": f, "second_half": s,
            "half_ratio": (s / f if f else float("nan")), "n": n}


arms = {}
for g in GAMMAS:
    rows = list(csv.DictReader(open(os.path.join(SWEEP, f"g{int(g)}", "steps.csv"))))
    assert len(rows) == WINDOW, f"g{int(g)}: {len(rows)} rows, expected {WINDOW}"
    tail = rows[-TAIL:]
    sup, cor = col(tail, "sup_loss"), col(tail, "coral_raw")
    # Per-step R, so its dispersion carries BOTH losses' noise jointly. They are correlated
    # within a step, so taking the two CVs separately and combining them would misstate it.
    R = [g * c / s for c, s in zip(cor, sup) if s > 0]
    head = col(rows[:TAIL], "coral_raw")
    a = {
        "gamma": g,
        "sup_loss": stats(sup),
        "coral_raw": stats(cor),
        "R": stats(R),
        "coral_raw_START_mean_first200": sum(head) / len(head),
        "coral_raw_step1": float(rows[0]["coral_raw"]),
        "src_feat_l2_first200": sum(col(rows[:TAIL], "src_feat_l2")) / TAIL,
        "src_feat_l2_last200": sum(col(tail, "src_feat_l2")) / TAIL,
    }
    a["src_feat_l2_growth"] = a["src_feat_l2_last200"] - a["src_feat_l2_first200"]
    a["src_feat_l2_growth_pct"] = 100.0 * a["src_feat_l2_growth"] / a["src_feat_l2_first200"]
    arms[g] = a

# ---- functional form -----------------------------------------------------------------
# R(g) = g * coral_raw(g) / sup_loss(g). R is strictly positive and varies over orders of
# magnitude across a GEOMETRIC gamma grid (1, 4, 16 = 4^0, 4^1, 4^2). A power law
# R = A * g^B is therefore the natural form: exactly linear in (log g, log R), positive by
# construction, and it uses all three points rather than bracketing two. OLS on the logs;
# R^2 is reported so a poor power law cannot pass silently.
x = [math.log(g) for g in GAMMAS]
y = [math.log(arms[g]["R"]["mean"]) for g in GAMMAS]
n = len(x)
mx = sum(x) / n
my = sum(y) / n
B = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sum((xi - mx) ** 2 for xi in x)
A = math.exp(my - B * mx)
ss_res = sum((yi - (math.log(A) + B * xi)) ** 2 for xi, yi in zip(x, y))
ss_tot = sum((yi - my) ** 2 for yi in y)
r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
g_star = (1.0 / A) ** (1.0 / B)                      # solve A * g^B = 1

# ---- uncertainty: propagate the dispersion of R through the same fit ------------------
# Each arm's tail-mean R carries a standard error CV/sqrt(n); the fit is redone with every
# log R shifted up and down by that, giving a band on the solution.
band = {}
for sgn, name in ((+1, "hi"), (-1, "lo")):
    yy = [math.log(arms[g]["R"]["mean"]
                   * (1 + sgn * arms[g]["R"]["cv"] / math.sqrt(arms[g]["R"]["n"])))
          for g in GAMMAS]
    myy = sum(yy) / n
    Bb = (sum((xi - mx) * (yi - myy) for xi, yi in zip(x, yy))
          / sum((xi - mx) ** 2 for xi in x))
    Ab = math.exp(myy - Bb * mx)
    band[name] = (1.0 / Ab) ** (1.0 / Bb)
lo, hi = sorted(band.values())

# ---- PRE-REGISTERED DECISION RULE, applied as written ---------------------------------
if 1.0 <= g_star <= 16.0:
    branch, sel = "A", g_star
    note = "solution lies inside the swept range [1, 16]; select it directly"
elif 16.0 < g_star <= 64.0:
    branch, sel = "B", g_star
    note = ("solution lies in (16, 64] -- selected, but a 500-step confirmation at this "
            "gamma is REQUIRED before any confirmatory run, to verify the source feature "
            "norm stays within the measured -1.5% band. Shrink pressure is monotone in "
            "gamma (-0.89, -1.08, -1.63 at 1/4/16), so it must be checked, not assumed.")
else:
    branch, sel = "C", 16.0
    note = ("solution exceeds 64 -- the magnitude-matching rule is declared INAPPLICABLE "
            "in this regime and gamma=16 is selected: the largest swept value with "
            "MEASURED non-collapsing, strictly-aligning behaviour. Not extrapolated, "
            "because the eps^4 shrink pressure is real and monotone in gamma.")

out = {
    "purpose": ("PRE-REGISTRATION for the round-4 CORAL confirmatory runs: the gamma "
                "read-off, fixed before any confirmatory fold exists (PROJECT.md 7.14)."),
    "rule": ("Sun & Saenko arXiv:1607.01719 sec.4 -- lambda set so the classification and "
             "CORAL losses are of comparable magnitude at the END of training, i.e. "
             "R(g) = g * coral_raw(g) / sup_loss(g) = 1."),
    "well_posedness": __doc__.split("=====\n")[-1].strip(),
    "source": os.path.relpath(SWEEP, REPO).replace("\\", "/"),
    "window": {"total_steps": WINDOW, "tail_steps": TAIL},
    "arms": {str(int(g)): arms[g] for g in GAMMAS},
    "fit": {
        "form": "R(g) = A * g^B, OLS on (log g, log R)",
        "why": ("the gamma grid is geometric (4^0, 4^1, 4^2) and R spans orders of "
                "magnitude; a power law is linear in the logs, positive by construction, "
                "and uses all three points rather than bracketing two"),
        "A": A, "B": B, "r_squared": r2,
        "solution_gamma": g_star,
        "uncertainty_band": [lo, hi],
        "band_basis": ("standard error of each arm's tail-mean R (CV/sqrt(200)) "
                       "propagated through the same OLS fit"),
    },
    "decision_rule": {
        "A": "solution in [1,16] -> select it",
        "B": "solution in (16,64] -> select it + require a 500-step feature-norm check",
        "C": "solution > 64 -> rule INAPPLICABLE, select gamma=16",
    },
    "branch_taken": branch,
    "selected_gamma": sel,
    "note": note,
    "constraints_for_confirmatory_runs": {
        "same_gamma_both_arms": ("coral_weight is a PINNED key; nb0 and nb3 MUST use this "
                                 "value or nb3-minus-nb0 stops isolating unfreezing"),
        "coral_target_ema": False,
        "feat_norm_ceiling": 110.0,
    },
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
body = json.dumps(out, indent=2, sort_keys=True)
out["sha256_of_body_without_this_field"] = hashlib.sha256(body.encode()).hexdigest()
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=2, sort_keys=True)

# ---- report --------------------------------------------------------------------------
print("=" * 92)
print("EQUALISING GAMMA -- round-4 CORAL sweep, trailing 200 of 2000 steps")
print("=" * 92)
print(f"{'g':>3} {'sup_loss':>10} {'coral_raw':>11} {'R=g*cor/sup':>12} "
      f"{'R cv':>7} {'R 1st/2nd half':>16} {'sup cv':>7} {'cor cv':>7}")
for g in GAMMAS:
    a = arms[g]
    print(f"{int(g):>3} {a['sup_loss']['mean']:>10.6f} {a['coral_raw']['mean']:>11.7f} "
          f"{a['R']['mean']:>12.5f} {a['R']['cv']:>7.3f} "
          f"{a['R']['first_half']:>7.4f}/{a['R']['second_half']:<8.4f} "
          f"{a['sup_loss']['cv']:>7.3f} {a['coral_raw']['cv']:>7.3f}")
print(f"\n{'g':>3} {'coral START (1st200)':>21} {'coral step1':>12} {'END':>11} "
      f"{'|f| start':>10} {'|f| end':>9} {'growth':>8} {'%':>7}")
for g in GAMMAS:
    a = arms[g]
    print(f"{int(g):>3} {a['coral_raw_START_mean_first200']:>21.7f} "
          f"{a['coral_raw_step1']:>12.7f} {a['coral_raw']['mean']:>11.7f} "
          f"{a['src_feat_l2_first200']:>10.3f} {a['src_feat_l2_last200']:>9.3f} "
          f"{a['src_feat_l2_growth']:>+8.3f} {a['src_feat_l2_growth_pct']:>+7.2f}")
print(f"\nFIT  R(g) = {A:.6g} * g^{B:.6g}   R^2 = {r2:.6f}")
print(f"SOLUTION  R(g)=1  ->  gamma* = {g_star:.4g}   band [{lo:.4g}, {hi:.4g}]")
print(f"INSIDE [1,16]? {'YES' if 1.0 <= g_star <= 16.0 else 'NO'}")
print(f"\nBRANCH {branch}  ->  SELECTED GAMMA = {sel:g}")
print(note)
print(f"\n[out] {OUT}")
print(f"sha256 {out['sha256_of_body_without_this_field']}")
