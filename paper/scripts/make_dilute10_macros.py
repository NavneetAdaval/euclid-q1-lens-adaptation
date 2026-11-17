r"""
make_dilute10_macros.py -- the numbers of the 1:10 diluted-target experiment (paper v6).
========================================================================================

    python paper/scripts/make_dilute10_macros.py

Writes ``paper/tables/M_dilute10.tex`` and nothing else. Every value comes from an artifact
(``_dilute10`` names them); none is typed. Naming follows the paper's macros: two decimals by
default (main text), suffix ``Four`` for the appendix's four decimals, ``Three`` where the text
needs a third decimal to show a bound that rounds to zero at two. Prefix ``dl`` is the dilution
experiment, ``mz`` the DANN - lambda=0 memorisation diagnostic (PROJECTv7 7.54).

Asserted before writing: every branch in results.json follows from its own G and D under part
1's table; the mandatory qualifier fires exactly where part 1's condition says; the near-boundary
pair (BNM's G clears zero, FreeMatch's misses it) holds; and the FreeMatch decomposition sums.
"""
from __future__ import annotations

import math
import os
import re
import statistics as st

import _common as C
import _dilute10 as X

WORD = {2: "two", 3: "three", 4: "four", 6: "six", 8: "eight", 10: "ten", 12: "twelve"}


def branch_of(g, d):
    """Part 1's table, applied to (mean, lo, hi) tuples."""
    if g[1] <= 0.0 <= g[2]:
        return "COLLAPSES"
    if d[1] > 0.0:
        return "SHRINKS"
    return "SURVIVES"


def build():
    R = X.results()
    out = []

    def add(name, val):
        assert re.fullmatch(r"[A-Za-z]+", name), name
        out.append((name, val))

    # ---------------- per arm: diluted TPR, G, D, branch ----------------
    for arm, name, label, m in X.ARMS:
        tpr = X.per_arm(arm, "tpr")
        au = X.per_arm(arm, "auroc")
        add(f"dlTpr{m}", X.u2(tpr["mean"]))
        add(f"dlTpr{m}Four", X.u4(tpr["mean"]))
        add(f"dlSd{m}", X.u2(tpr["sd"]))
        add(f"dlSd{m}Four", X.u4(tpr["sd"]))
        add(f"dlAuroc{m}", X.u2(au["mean"]))
        add(f"dlRead{m}", str(X.read_at(arm)))
        assert X.read_at(arm) == X.READ[arm][0]
        g, d, b = X.G(arm), X.D(arm), X.bal_minus_l0(arm)
        assert (g["diluted_at"], g["lambda0_at"]) == X.READ[arm], (arm, g["diluted_at"])
        for tag, r in (("G", g), ("D", d)):
            add(f"dl{tag}{m}", X.s2(r["mean"]))
            add(f"dl{tag}{m}Lo", X.s2(r["ci_lo"]))
            add(f"dl{tag}{m}Hi", X.s2(r["ci_hi"]))
            add(f"dl{tag}{m}Four", X.s4(r["mean"]))
            add(f"dl{tag}{m}LoFour", X.s4(r["ci_lo"]))
            add(f"dl{tag}{m}HiFour", X.s4(r["ci_hi"]))
            add(f"dl{tag}{m}Signs", str(r["n_same_sign"]))
        add(f"dlD{m}HiAbs", X.u2(d["ci_hi"]))
        add(f"dlD{m}HiAbsFour", X.u4(d["ci_hi"]))      # 7.65: the branch sentences' four places
        add(f"dlBal{m}", X.s2(b["mean"]))
        add(f"dlBal{m}Four", X.s4(b["mean"]))
        add(f"dlBal{m}Abs", X.u2(b["mean"]))
        add(f"dlBal{m}AbsFour", X.u4(b["mean"]))
        ret = R["descriptive"]["retained_fraction"][f"{label} (primary)"]
        assert abs(ret - g["mean"] / b["mean"]) < 1e-12
        add(f"dlRet{m}", X.u2(ret))
        v = X.verdict(arm)
        gt = (g["mean"], g["ci_lo"], g["ci_hi"])
        dt = (d["mean"], d["ci_lo"], d["ci_hi"])
        assert branch_of(gt, dt) == v["branch"], (arm, v["branch"])
        assert list(v["G_primary"]) == list(gt) and list(v["D"]) == list(dt)
        assert not v["no_verdict_rule_fires"]
        qual = v["branch"] == "SURVIVES" and d["ci_hi"] > 0.5 * b["mean"]
        assert qual == ("though the interval admits" in v["licensed_sentence"]), arm
        add(f"dlBranch{m}", v["branch"])
        # matched (sensitivity) G forms
        if arm in X.MATCHED_G:
            gm = X.G(arm, "sensitivity")
            bm = X.bal_minus_l0(arm, "matched")
            assert (gm["diluted_at"], gm["lambda0_at"]) == X.READ_MATCHED[arm]
            add(f"dlGm{m}", X.s2(gm["mean"]))
            add(f"dlGm{m}Lo", X.s2(gm["ci_lo"]))
            add(f"dlGm{m}Hi", X.s2(gm["ci_hi"]))
            add(f"dlGm{m}Four", X.s4(gm["mean"]))
            add(f"dlGm{m}LoFour", X.s4(gm["ci_lo"]))
            add(f"dlGm{m}HiFour", X.s4(gm["ci_hi"]))
            add(f"dlGm{m}Signs", str(gm["n_same_sign"]))
            add(f"dlBalm{m}", X.s2(bm["mean"]))
            assert branch_of((gm["mean"], gm["ci_lo"], gm["ci_hi"]), dt) == v["branch"]
            # v6e: the retained fraction with the control read at the arm's own step
            retm = R["descriptive"]["retained_fraction"][f"{label} (matched)"]
            assert abs(retm - gm["mean"] / bm["mean"]) < 1e-12
            add(f"dlRetm{m}", f"{retm:.2f}")

    # which arms survive, in words, and the qualifier arms
    surv = [X.LABEL[a] for a, *_ in X.ARMS if X.verdict(a)["branch"] == "SURVIVES"]
    coll = [X.LABEL[a] for a, *_ in X.ARMS if X.verdict(a)["branch"] == "COLLAPSES"]
    assert surv == ["DANN", "BNM"] and coll == ["CORAL", "FreeMatch"], (surv, coll)

    # ---------------- the near-boundary pair (BNM clears zero, FreeMatch misses it) ----------
    gb, gf = X.G("bnm"), X.G("freematch")
    assert 0 < gb["ci_lo"] < 0.005 and -0.005 < gf["ci_lo"] < 0, (gb["ci_lo"], gf["ci_lo"])
    assert gf["mean"] > gb["mean"]
    add("dlGBnmLoThree", f"{gb['ci_lo']:+.3f}")
    add("dlGBnmLoAbsThree", f"{abs(gb['ci_lo']):.3f}")
    add("dlGFmLoThree", f"{gf['ci_lo']:+.3f}")
    add("dlGFmLoAbsThree", f"{abs(gf['ci_lo']):.3f}")

    # ---------------- DANN: diluted against balanced, spread ----------------
    comp = R["comparators"]
    bal_dann = comp["DANN x nb3 (balanced) @4000"]["tpr"]
    add("dlTprDannBal", X.u2(bal_dann["mean"]))
    add("dlSdDannBal", X.u2(bal_dann["sd"]))
    lam = comp["lambda=0 @4000"]
    add("dlLamTpr", X.u2(lam["tpr"]["mean"]))
    add("dlLamAuroc", X.u2(lam["auroc"]["mean"]))

    # ---------------- FreeMatch decomposition; frozen stages vs round 1 ----------------
    dec = R["descriptive"]["freematch_decomposition"]
    dd, db = dec["diluted"], dec["balanced"]
    assert abs(dd["frozen_part"] + dd["unfrozen_part"] - dd["G_primary"]) < 1e-12
    assert abs(db["frozen_part"] + db["unfrozen_part"] - db["total"]) < 1e-12
    assert abs(dd["G_primary"] - gf["mean"]) < 1e-12
    for nm, v in (("dlFmStart", dd["start"]), ("dlFmStartBal", db["start"])):
        add(nm, X.u2(v))
        add(nm + "Four", X.u4(v))
    for nm, v in (("dlFmFrozenPart", dd["frozen_part"]), ("dlFmUnfrozenPart", dd["unfrozen_part"]),
                  ("dlFmFrozenPartBal", db["frozen_part"]),
                  ("dlFmUnfrozenPartBal", db["unfrozen_part"])):
        add(nm, X.s2(v))
        add(nm + "Four", X.s4(v))
    cs = dec["coral_start"]
    add("dlCoralStart", X.u2(cs["diluted"]))
    add("dlCoralStartBal", X.u2(cs["balanced_round1"]))
    fvr = R["descriptive"]["frozen_vs_round1"]
    for arm, key in (("Fm", "FreeMatch"), ("Coral", "CORAL")):
        agg = __import__("json").loads(fvr[key]["aggregate"]) if isinstance(
            fvr[key]["aggregate"], str) else fvr[key]["aggregate"]
        t = agg["tpr"]
        add(f"dl{arm}FrozenDiff", X.s2(t["diff"]))
        add(f"dl{arm}FrozenDiffLo", X.s2(t["ci"][0]))
        add(f"dl{arm}FrozenDiffHi", X.s2(t["ci"][1]))
        add(f"dl{arm}FrozenDiffFour", X.s4(t["diff"]))
        add(f"dl{arm}FrozenDiffLoFour", X.s4(t["ci"][0]))
        add(f"dl{arm}FrozenDiffHiFour", X.s4(t["ci"][1]))
    fz = R["descriptive"]["frozen_stages"]
    add("dlCoralFrozenAuroc", X.u2(fz["coral_frozen"]["auroc"]["mean"]))
    assert fz["coral_frozen"]["gate_values_agree"] and fz["freematch_frozen"]["gate_values_agree"]

    # ---------------- CORAL: the frozen-stage head (head_norms.json) ----------------
    H = X.heads()
    assert H["checks"]["passed"]
    g_d = [H["heads"]["coral"]["diluted"][str(k)]["ln_gain_norm"] for k in X.FOLDS]
    g_r = [H["heads"]["coral"]["round1"][str(k)]["ln_gain_norm"] for k in X.FOLDS]
    add("dlCoralGainMin", f"{min(g_d):.0f}")
    add("dlCoralGainMax", f"{max(g_d):.0f}")
    add("dlCoralGainRoneMin", f"{min(g_r):.0f}")
    add("dlCoralGainRoneMax", f"{max(g_r):.0f}")
    ratio = st.mean(g_d) / st.mean(g_r)
    assert 3.5 <= ratio < 4.5, ratio                                # "about four times"
    add("dlCoralGainRatioWord", WORD[round(ratio)])
    add("dlUnitGain", f"{H['unit_gain_norm']:.1f}")

    # ---------------- CORAL x nb3: steps.csv, advisory log, skips ----------------
    feat_min, feat_max, start, n_steps, fires, skips, applied, ceil = [], [], [], [], [], [], [], set()
    start_both = []                                  # r3: step-1 norms, source and target
    for k in X.FOLDS:
        rows = X.steps_csv(k)
        steps = [int(r["step"]) for r in rows]
        assert steps == list(range(1, len(rows) + 1))
        n_steps.append(len(rows))
        both = [min(float(r["src_feat_l2"]), float(r["tgt_feat_l2"])) for r in rows]
        feat_min.append(min(both))
        feat_max.append(max(max(float(r["src_feat_l2"]), float(r["tgt_feat_l2"])) for r in rows))
        start.append(float(rows[0]["src_feat_l2"]))
        start_both += [float(rows[0]["src_feat_l2"]), float(rows[0]["tgt_feat_l2"])]
        sk = [int(float(r["step_skipped"])) for r in rows]
        # the skips are the run's first steps, contiguous from step 1
        first = sum(1 for _ in iter(lambda it=iter(sk): next(it) == 1, False))
        assert sum(sk) == first, (k, sum(sk), first)
        skips.append(sum(sk))
        applied.append((len(rows) - sum(sk)) / len(rows))
        adv = X.advisory(k)
        fires.append(adv["n_firings"])
        ceil |= {float(r["ceiling"]) for r in X.advisory_rows(k)}
        rr = X.run_record("coral_nb3", k)
        assert rr["fp16_skipped_cumulative"] == sum(sk) and rr["steps_completed"] == len(rows)
    assert len(set(n_steps)) == 1 and fires == n_steps and len(ceil) == 1
    ceiling = ceil.pop()
    assert min(feat_min) > ceiling
    nb = X.results()["no_verdict_inputs"]["coral_nb3"]
    assert [nb[str(k)]["skips"] for k in X.FOLDS] == skips
    add("dlCoralSteps", str(n_steps[0]))
    add("dlCoralFeatMin", f"{min(feat_min):.1f}")
    add("dlCoralFeatMax", f"{max(feat_max):.1f}")
    add("dlCoralFeatStart", f"{10 * round(st.mean(start) / 10):.0f}")
    add("dlCoralFeatStartMin", f"{min(start_both):.1f}")
    add("dlCoralFeatStartMax", f"{max(start_both):.1f}")
    add("dlCoralCeiling", f"{ceiling:.0f}")
    add("dlCoralSkipMin", str(min(skips)))
    add("dlCoralSkipMax", str(max(skips)))
    add("dlCoralAppliedPct", f"{100 * min(applied):.1f}")
    fmnb = X.results()["no_verdict_inputs"]["freematch_nb3"]
    add("dlFmSkipMin", str(min(fmnb[str(k)]["skips"] for k in X.FOLDS)))
    add("dlFmSkipMax", str(max(fmnb[str(k)]["skips"] for k in X.FOLDS)))
    add("dlFmAppliedPct", f"{100 * min(fmnb[str(k)]['applied_fraction'] for k in X.FOLDS):.1f}")

    # ---------------- BNM's predicted-positive fraction on diluted target batches ----------------
    tr = R["descriptive"]["bnm_pred_pos_trajectory"]
    dil = [v for f in X.FOLDS for _s, v in tr["diluted"][str(f)]]
    bal = [v for f in X.FOLDS for _s, v in tr["balanced_context"][str(f)]]
    n_logged = {len(tr["diluted"][str(f)]) for f in X.FOLDS}
    assert len(n_logged) == 1
    add("dlBnmPpMin", X.u2(min(dil)))
    add("dlBnmPpMax", X.u2(max(dil)))
    add("dlBnmPpMean", X.u2(st.mean(dil)))
    add("dlBnmPpFirst", X.u2(st.mean(tr["diluted"][str(f)][0][1] for f in X.FOLDS)))
    add("dlBnmPpFirstStep", str(tr["diluted"]["1"][0][0]))
    n_log = n_logged.pop()
    add("dlBnmPpN", str(n_log))
    # r3: the trace's logging interval (every logged step is a multiple of it, up to the budget)
    gaps = {b[0] - a[0] for f in X.FOLDS for a, b in zip(tr["diluted"][str(f)], tr["diluted"][str(f)][1:])}
    assert len(gaps) == 1
    every = gaps.pop()
    assert all(tr["diluted"][str(f)][0][0] == every and tr["diluted"][str(f)][-1][0] == every * n_log
               for f in X.FOLDS)
    add("dlBnmPpEvery", str(every))
    add("dlBnmPpBalMin", X.u2(min(bal)))
    add("dlBnmPpBalMax", X.u2(max(bal)))
    lf = R["descriptive"]["bookkeeping"]["lens_fraction"]
    exp = {lf[f"bnm/fold{k}"]["expected"] for k in X.FOLDS}
    add("dlBnmLensFrac", X.u2(st.mean(exp)))
    comp3d = [c for c in R["predictions"]["components"] if c["id"] == "3d(ii)"]
    assert len(comp3d) == 1 and comp3d[0]["score"] == "held"
    thr = re.search(r"> (0\.\d+)", comp3d[0]["component"]).group(1)
    add("dlBnmPpThreshold", thr)

    # ---------------- predictions, contrasts counted ----------------
    p1 = X.load(X.PART1)
    n_pred = len(p1["predictions_VERBATIM"]["items"])
    add("dlNPredWord", WORD[n_pred])
    nG, nD = len(R["G"]), len(R["D"])
    nB = len(R["descriptive"]["frozen_vs_round1"])
    assert R["multiplicity"].startswith(f"{nG + nD} paired t-intervals ({nG} G, {nD} D) plus {nB} ")
    add("dlNt", str(nG + nD))
    add("dlNtWord", WORD[nG + nD])
    add("dlNtG", str(nG))
    add("dlNtD", str(nD))
    add("dlNboot", str(nB))
    add("dlNbootWord", WORD[nB])

    # ---------------- sensitivities: FreeMatch G on the original pool ----------------
    so = R["sensitivities"]["orig"]["G"]["FreeMatch x nb3|primary"]
    assert so["excludes_zero"] and not R["sensitivities"]["orig"]["zero_inclusion_matches_cleaned"][
        "G FreeMatch primary"]
    add("dlGFmOrig", X.s2(so["mean"]))
    add("dlGFmOrigLo", X.s2(so["ci_lo"]))
    add("dlGFmOrigHi", X.s2(so["ci_hi"]))
    zm = R["sensitivities"]["orig"]["zero_inclusion_matches_cleaned"]
    add("dlOrigMatchN", str(sum(zm.values())))
    add("dlOrigMatchWord", WORD[sum(zm.values())])
    zr = R["sensitivities"]["D2"]["zero_inclusion_matches_cleaned"]
    assert all(zr.values())
    add("dlReducedMatchN", str(sum(zr.values())))

    # ---------------- the pool (data/splits/dilute10_pool.json) ----------------
    P = X.pool()
    pf = {f["fold"]: f for f in P["folds"]}
    nl = [pf[k]["n_lenses"] for k in X.FOLDS]
    ng = [pf[k]["n_negatives"] for k in X.FOLDS]
    assert nl[:3] == [nl[0]] * 3 and ng[:3] == [ng[0]] * 3
    assert all(g == P["ratio_galaxies_per_lens"] * l for l, g in zip(nl, ng))
    add("dlPoolLens", str(nl[0]))
    add("dlPoolLensFour", str(nl[3]))
    add("dlPoolGal", X.paper_number(ng[0]))
    add("dlPoolGalFour", X.paper_number(ng[3]))
    add("dlRatio", str(P["ratio_galaxies_per_lens"]))
    fresh = sum(pf[k]["n_fresh_negatives"] for k in X.FOLDS)
    assert fresh == p1["pool"]["capacity"]["fresh_needed_total"]
    add("dlFresh", X.paper_number(fresh))
    hits = p1["pool"]["existing_negatives_screen_hits_NOT_REMOVED"]
    hosts = [hits[str(k)]["b_simulation_hosts"] for k in X.FOLDS]
    assert all(hits[str(k)]["a_discovery_engine_grade_AB_lenses"] == 0
               and hits[str(k)]["c_grade_X_hard_negatives"] == 0 for k in X.FOLDS)
    add("dlHosts", "/".join(str(h) for h in hosts))
    rr = p1["pool"]["realised_ratios"]
    tp = [rr[str(k)]["dann_bnm_training_pool"] for k in X.FOLDS]
    assert tp[:3] == [tp[0]] * 3
    l1, g1 = (int(x) for x in tp[0].split(":"))
    l4, g4 = (int(x) for x in tp[3].split(":"))
    add("dlTrainLens", str(l1))
    add("dlTrainGal", X.paper_number(g1))
    add("dlTrainLensFour", str(l4))
    add("dlTrainGalFour", X.paper_number(g4))
    hold = {rr[str(k)]["dann_bnm_holdout_kept_byte_identical"] for k in X.FOLDS}
    assert len(hold) == 1
    hn = int(hold.pop().split("=")[1])
    add("dlHoldout", str(hn))
    tfrac = [rr[str(k)]["dann_bnm_training_pool_lens_fraction"] for k in X.FOLDS]
    wfrac = [rr[str(k)]["whole_pool_lens_fraction"] for k in X.FOLDS]
    add("dlTrainFracThree", f"{st.mean(tfrac):.3f}")
    add("dlPoolFracThree", f"{st.mean(wfrac):.3f}")
    rows = int(_printed("M_batch.tex", "batchTargetRows"))
    add("dlBatchLens", str(round(rows * wfrac[0])))
    # v6e: DANN and BNM batches come from the training pool, without the half-lens holdout
    add("dlBatchLensTrain", str(round(rows * tfrac[0])))
    add("dlBatchLensBal", str(rows // 2))
    sky = 2200                                                      # the text's Euclid Q1 rarity
    dens = sky / P["ratio_galaxies_per_lens"]
    # v6e: two significant figures (2200/10 = 220; one figure printed 200)
    add("dlDenserWord", str(int(float(f"{dens:.2g}"))))
    # v6e (Section 6.6): what one test lens moves in the completeness ratio to the benchmark's
    # 0.50, on one fold and on the four-fold mean, from the folds' test-lens counts
    S = X.load(C.R("data", "splits", "splits.json"))
    npos = {f["fold"]: len(f["test_lens_rows"]) for f in S["real_lenses_kfold"]["folds"]}
    pp = [100 / (npos[k] * C.LINES_COMPLETENESS) for k in X.FOLDS]
    assert all(1.5 < x < 2.5 for x in pp), pp                      # "about two points"
    assert 0.4 < st.mean(pp) / len(X.FOLDS) < 0.6, pp               # "about half a point"
    add("dlLensFoldPpWord", "two")
    add("dlLensMeanPpWord", "half a")

    # each target lens is seen (1/2) / (lens fraction) times less often per step than at 1:1
    add("dlExposureRatio", f"{0.5 / wfrac[0]:.1f}")
    # the lambda=0 reuse check: source batches bit-identical over N steps (part 2, rng_coupling)
    rng = X.load(X.PART2)["rng_coupling"]
    assert rng["lambda0_reuse"] == "UNQUALIFIED"
    assert all(v["identical_2000_steps"] for v in rng["per_arm_fold"].values())
    n_rng = int(re.search(r"bit-identical for (\d+) steps", rng["finding"]).group(1))
    add("dlRngSteps", str(n_rng))
    # CORAL x nb3 fold 1's aborted first attempt: its record folder and part 3's permission
    p3 = X.load(X.PART3)
    ab = p3["coral_nb3_fold1_rerun"]
    assert ab["permitted"]
    dm = re.search(r"ABORTED_(\d{4})(\d{2})(\d{2})/?$", ab["record"])
    assert os.path.isdir(C.R(*ab["record"].strip("/").split("/")))
    add("dlAbortDate", f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}")
    nvr = re.search(r"fewer than (\d+)% of its budget", p3["no_verdict_rule"]).group(1)
    assert nvr == "50"
    # short digests, as Appendix B prints them (self-certifying, from the files themselves)
    for nm, p in (("dlShaPartOne", X.PART1), ("dlShaPartTwo", X.PART2), ("dlShaPartThree", X.PART3),
                  ("dlShaPlan", X.PLAN), ("dlShaGapPlan", X.GAP_PLAN)):
        add(nm, X.load(p)["sha256_of_body_without_this_field"][:8])

    # ---------------- venues (the run records) ----------------
    gpus = {a: {X.run_record(a, k).get("gpu_name") for k in X.FOLDS}
            for a in ("dann", "bnm", "coral_nb3", "freematch_frozen", "freematch_nb3")}
    assert gpus["dann"] == gpus["bnm"] == gpus["coral_nb3"] == {"Tesla T4"}, gpus
    assert gpus["freematch_frozen"] == gpus["freematch_nb3"] == {None}, gpus
    assert p1["head_provenance"]["CORAL x nb3"]["frozen_stage_device"].startswith("cpu")
    rep = X.results()["reproduction"]
    assert rep["passed"] and not rep["mismatches"]
    assert all(r["match"] for r in rep["contrasts"] + rep["per_fold"])
    add("dlReproNWord", WORD[len(rep["contrasts"])])

    # ---------------- the memorisation diagnostic (7.54) ----------------
    Gp = X.gap()
    c = Gp["contrast_dann_minus_lambda0"]
    for nm, r in (("mzPrim", c["primary_at_4000"]), ("mzSec", c["secondary_within_fold_mean"])):
        assert not r["excludes_zero"]
        add(nm, X.s2(r["mean"]))
        add(nm + "Lo", X.s2(r["ci_lo"]))
        add(nm + "Hi", X.s2(r["ci_hi"]))
        add(nm + "Four", X.s4(r["mean"]))
        add(nm + "LoFour", X.s4(r["ci_lo"]))
        add(nm + "HiFour", X.s4(r["ci_hi"]))
    kn = Gp["known_null"]["per_fold"]
    shift = {k: abs(v["s729_json"] - v["nuclear_gap"]) for k, v in kn.items()}
    moved = [k for k, v in shift.items() if v > 1e-6]
    assert moved == ["1"], shift                                    # only the unseeded fold moved
    add("mzViewShift", X.u2(shift["1"]))
    n_ck = {len(v) for arm in Gp["per_checkpoint"].values() for v in arm.values()}
    assert len(n_ck) == 1
    add("mzNckpt", str(n_ck.pop()))
    add("mzHoldout", str(Gp["per_checkpoint"]["dann"]["1"]["4000"]["n_per_slice"]))
    return out


def _printed(fname, macro):
    txt = open(os.path.join(C.TABLES, fname), encoding="utf-8").read()
    m = re.search(r"\\newcommand\{\\" + macro + r"\}\{([^}]*)\}", txt)
    assert m, (fname, macro)
    return m.group(1)


def main() -> int:
    vals = build()
    names = [n for n, _ in vals]
    assert len(names) == len(set(names)), "duplicate macro"
    # no clash with any macro an existing table file defines
    for p in os.listdir(C.TABLES):
        if p.startswith("M_") and p != "M_dilute10.tex":
            txt = open(os.path.join(C.TABLES, p), encoding="utf-8").read()
            clash = [n for n in names if "\\newcommand{\\" + n + "}" in txt]
            assert not clash, (p, clash)
    C.write_table(os.path.join(C.TABLES, "M_dilute10.tex"),
                  "\n".join(rf"\newcommand{{\{n}}}{{{v}}}" for n, v in vals),
                  header_comment="Macros of the 1:10 diluted-target experiment (paper v6). Sources: "
                                 "runs/dilute10_analysis/results.json, "
                                 "runs/dann_lambda0_holdout_gap/results.json, "
                                 "runs/dilute10_head_norms/head_norms.json, "
                                 "data/splits/dilute10_pool.json, runs/dilute10/*/fold*/ run files.",
                  size="")
    print(f"  {len(vals)} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
