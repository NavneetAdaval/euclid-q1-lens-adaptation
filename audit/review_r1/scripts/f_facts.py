"""
f_facts.py -- review round 1, Part F (facts to look up). POST HOC AND EXPLORATORY.

    python -B audit/review_r1/scripts/f_facts.py

Writes audit/review_r1/out/F1_prereg_contrasts.json ... F7_lenses_in_test_pool.json.
Reads the project; writes nothing outside audit/review_r1/out.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_common as RC                                   # noqa: E402

C = RC.C
SCRIPT = os.path.abspath(__file__)


def find_lines(path, needle, regex=False):
    """1-based line numbers of every line containing ``needle``."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, ln in enumerate(fh, 1):
            if (re.search(needle, ln) if regex else needle in ln):
                out.append(i)
    return out


def line_text(path, n):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, ln in enumerate(fh, 1):
            if i == n:
                return ln.rstrip("\n")
    return None


def cite(path, needle, regex=False, first_only=True):
    ls = find_lines(path, needle, regex)
    assert ls, (path, needle)
    ls = ls[:1] if first_only else ls
    return [{"file": RC.rel(path), "line": n, "text": line_text(path, n).strip()} for n in ls]


# ======================================================================= F1
def f1():
    I = RC.Inputs()
    p = I.add(RC.R("runs", "lambda0_r6_prereg.json"))
    pre = json.load(open(p, encoding="utf-8"))
    rep = pre["reporting"]
    contrasts = rep["paired_contrasts"]
    res = {
        "file": RC.rel(p),
        "paired_contrasts_verbatim": {
            "key": "reporting.paired_contrasts",
            "entries": contrasts,
            "line_citations": [cite(p, json.dumps(c["contrast"]))[0] for c in contrasts],
        },
        "step_fields_in_the_file": {
            "budget.CONFIRM_STEPS": pre["budget"]["CONFIRM_STEPS"],
            "budget.rule": pre["budget"]["rule"],
            "reporting.computed_at": rep["computed_at"],
            "sensitivity_readouts.steps": pre["sensitivity_readouts"]["steps"],
            "sensitivity_readouts.clause": pre["sensitivity_readouts"]["clause"],
            "sensitivity_readouts.what_they_buy": pre["sensitivity_readouts"]["what_they_buy"],
            "budget.alternative_considered_and_declined":
                pre["budget"]["alternative_considered_and_declined"],
            "line_citations": (cite(p, '"CONFIRM_STEPS"') + cite(p, '"computed_at"') +
                               cite(p, '"what_they_buy"') + cite(p, '"clause"') +
                               cite(p, '"steps": [')),
        },
        "contrast_entries_name_a_step": any(re.search(r"\d{3,4}", c["contrast"])
                                            for c in contrasts),
    }
    # The other arm in each contrast is read at its own declared step (from _common.ARM_META,
    # which crosscheck.py verifies against each arm's hashed budget file).
    lam = pre["budget"]["CONFIRM_STEPS"]
    sens = pre["sensitivity_readouts"]["steps"]
    meta = {a: C.ARM_META[a]["readout"] for a in C.ARM_META}
    reading = {
        "lambda=0 - Baseline A (frozen)": dict(
            arm_a="lambda0", step_a=lam, arm_b="baseline", step_b=meta["baseline"],
            sensitivity_forms=[]),
        "DANN x nb3 - lambda=0": dict(
            arm_a="dann", step_a=meta["dann"], arm_b="lambda0", step_b=lam,
            sensitivity_forms=[]),
        "CORAL x nb3 - lambda=0": dict(
            arm_a="coral", step_a=meta["coral"], arm_b="lambda0", step_b=lam,
            sensitivity_forms=[dict(arm_a="coral", step_a=meta["coral"], arm_b="lambda0",
                                    step_b=2000)]),
        "FreeMatch x nb3 - lambda=0": dict(
            arm_a="freematch", step_a=meta["freematch"], arm_b="lambda0", step_b=lam,
            sensitivity_forms=[dict(arm_a="freematch", step_a=meta["freematch"],
                                    arm_b="lambda0", step_b=1500)]),
    }
    assert set(reading) == {c["contrast"] for c in contrasts}
    assert 2000 in sens and 1500 in sens
    # CONTRAST_ROWS of paper/scripts/_common.py
    rows = []
    cp = RC.R("paper", "scripts", "_common.py")
    I.add(cp)
    for i, r in enumerate(C.CONTRAST_ROWS):
        rows.append(dict(index=i, panel=r[0], label=r[1], arm_a=r[2], step_a=r[3],
                         arm_b=r[4], step_b=r[5], budget=r[6]))
    for name, rd in reading.items():
        def match(a, sa, b, sb):
            return [x["index"] for x in rows if (x["arm_a"], x["step_a"], x["arm_b"],
                                                 x["step_b"]) == (a, sa, b, sb)]
        rd["headline_form_rows"] = match(rd["arm_a"], rd["step_a"], rd["arm_b"], rd["step_b"])
        for s in rd["sensitivity_forms"]:
            s["rows"] = match(s["arm_a"], s["step_a"], s["arm_b"], s["step_b"])
    res["reading_of_each_contrast"] = reading
    res["contrast_rows"] = {"file": RC.rel(cp),
                            "line_of_CONTRAST_ROWS": find_lines(cp, "CONTRAST_ROWS = [")[0],
                            "rows": rows}
    used = set()
    for rd in reading.values():
        used |= set(rd["headline_form_rows"])
        for s in rd["sensitivity_forms"]:
            used |= set(s["rows"])
    res["contrast_rows_not_corresponding_to_any_preregistered_contrast"] = [
        x["index"] for x in rows if x["index"] not in used]
    RC.write_json("F1_prereg_contrasts.json", res, script=SCRIPT, inputs=I, pool="n/a",
                  folds="n/a",
                  notes=["Steps: the contrast entries name no step. lambda=0's step is the "
                         "file's budget (4000) with 1500/2000 as declared sensitivity readouts; "
                         "the other arm's step is its own declared read-off "
                         "(_common.ARM_META, verified by crosscheck.py)."])


# ======================================================================= F2
def f2():
    I = RC.Inputs()
    dl = I.add(RC.R("scripts", "da_losses.py"))
    ts = I.add(RC.R("scripts", "train_skeleton.py"))
    ru = I.add(RC.R("scripts", "run_unfreeze_freematch.py"))
    rb = I.add(RC.R("scripts", "run_bnm_r6.py"))
    cfgs = sorted(glob.glob(RC.R("runs", "freematch_nb3_unfrozen_fold*", "run_config.json")))
    I.add(*cfgs)
    fm_cfg = {}
    for c in cfgs:
        j = json.load(open(c, encoding="utf-8"))
        fm_cfg[RC.rel(c)] = {"fairness_target": j.get("fairness_target"),
                             "saf_weight": j.get("saf_weight"),
                             "line": find_lines(c, '"fairness_target"')[0]}
    bnm_cfgs = sorted(glob.glob(RC.R("runs", "bnm_r6_folds", "bnm_r6_fold*", "run_config.json")))
    I.add(*bnm_cfgs)
    bnm_keys = {}
    for c in bnm_cfgs:
        j = json.load(open(c, encoding="utf-8"))
        bnm_keys[RC.rel(c)] = sorted(k for k in j if re.search(
            r"prior|fair|marginal|class_frac|target_frac", k, re.I))
    # every BNM-related source line mentioning a prior/marginal/fairness word
    lo = find_lines(dl, "class BNMLoss")[0]
    hi = find_lines(dl, 'return f"gamma={self.gamma}, target_only=True')[0]
    bnm_src = open(dl, encoding="utf-8").read().splitlines()[lo - 1:hi]
    bnm_hits = [lo + i for i, ln in enumerate(bnm_src)
                if re.search(r"prior|marginal|fairness|class_frac", ln, re.I)]
    res = {
        "freematch": {
            "value_used": "uniform [0.5, 0.5] over [non-lens, lens] (fairness_target=None -> "
                          "torch.full((num_classes,), 1/num_classes), num_classes=2)",
            "config_key": "fairness_target",
            "set_where": {
                "runner_constant": cite(ru, "FAIRNESS_TARGET = None"),
                "runner_passes_it": cite(ru, "fairness_target=FAIRNESS_TARGET"),
                "train_config_default": cite(ts, "freematch_fairness_target: tuple | None"),
                "train_skeleton_passes_it": cite(ts, "fairness_target=cfg.freematch_fairness_target"),
            },
            "code_lines": (cite(dl, "fairness_target : sequence[float] | None, default None") +
                           cite(dl, "SAF target marginal. None -> uniform") +
                           cite(dl, "init = 1.0 / self.num_classes") +
                           cite(dl, "if fairness_target is None:") +
                           cite(dl, "ft = torch.full((self.num_classes,), init)") +
                           cite(dl, 'self.register_buffer("fairness_target", ft)') +
                           cite(dl, "saf = self_adaptive_fairness_penalty(mean_pred, self.fairness_target")
                           + cite(dl, "uniform [0.5, 0.5] -- this is the anti-collapse")),
            "stored_run_configs": fm_cfg,
            "saf_weight_in_run_configs": sorted({v["saf_weight"] for v in fm_cfg.values()}),
        },
        "bnm": {
            "has_class_prior_parameter": False,
            "constructor_signature": cite(dl, "def __init__(self, gamma: float = 1.0):"),
            "objective_lines": (cite(dl, "L_bnm = - nuclear_norm(A) / min(B, C)") +
                                cite(dl, "prob2 = binary_prob2(tgt_logits)") +
                                cite(dl, "nuc = bnm_nuclear_norm(prob2)") +
                                cite(dl, "        return -nuc")),
            "prior_or_marginal_words_inside_BNMLoss": bnm_hits,
            "BNMLoss_span": [lo, hi],
            "run_config_keys_matching_prior_fair_marginal": bnm_keys,
            "related_but_not_in_the_objective": {
                "what": "the BNM runner's TARGET batch sampler is class-balanced using the "
                        "target lens/non-lens split (pos_fraction 0.5); this sets batch "
                        "composition, not a term in the loss",
                "lines": (cite(rb, "POS_FRACTION = 0.5") +
                          cite(rb, "tgt_sampler = ClassBalancedBatchSampler(holdout[\"train_pos\"]") +
                          find_and_cite_after(rb, "tgt_sampler = ClassBalancedBatchSampler",
                                              "pos_fraction=POS_FRACTION")),
            },
        },
    }
    RC.write_json("F2_class_priors.json", res, script=SCRIPT, inputs=I, pool="n/a",
                  folds="n/a")


def find_and_cite_after(path, anchor, needle):
    a = find_lines(path, anchor)[0]
    for n in find_lines(path, needle):
        if n > a:
            return [{"file": RC.rel(path), "line": n, "text": line_text(path, n).strip()}]
    return []


# ======================================================================= F3
def f3():
    I = RC.Inputs()
    pj = I.add(RC.R("PROJECTv7.md"))
    sec = I.add(RC.R("paper", "sections", "03_data.tex"))
    cur = I.add(RC.R("paper", "RELEASE_CURATION_REPORT.md"))
    cands = {}
    for name, desc_needles in {
        "validate_singleband.py": ["def feats(stack):", "return np.stack([stack.mean(1), stack.std(1),",
                                   "proba = cross_val_predict(LogisticRegression(max_iter=1000)",
                                   "BRIGHTNESS-SHORTCUT AUROC (sim vs real)",
                                   'ds = LensDataset("target_unlabeled"'],
        "probe_deflector_gap.py": ["SIM_NONLENS_DIR =", "REAL_NEG_DIR =", "def three_stats(imgs):",
                                   'clf = LogisticRegression(max_iter=2000, class_weight="balanced")',
                                   "stat_auc = cv_auroc(", "enc_auc = cv_auroc(feats, y)"],
        "probe_arc_on_real.py": ["SIM_NPZ =", "REAL_NEG_DIR =",
                                 "auc_lens_vs_real = cv_auroc(", "auc_lens_vs_bare = cv_auroc(",
                                 "auc_bare_vs_real = cv_auroc("],
        "probe_realism.py": ["SIM_NPZ =", "REAL_LENS_NPZ =", "A. sim-lens  vs real-lens",
                             "B. real-lens vs real-neg", "C. sim-lens  vs real-neg",
                             "D. sim bare-defl vs real-lens"],
    }.items():
        p = I.add(RC.R("scripts", name))
        cands[name] = {"lines": [c for n in desc_needles for c in cite(p, n)],
                       "writes_files": [c for n in ("savefig", "np.save(", "json.dump", "to_csv(")
                                        for c in (cite(p, n) if find_lines(p, n) else [])]}
    # the stored outputs the candidates could have produced
    outputs = {
        "sim_vs_real_v2.png (validate_singleband.py's only file output)":
            [RC.rel(x) for x in glob.glob(RC.R("**", "sim_vs_real*"), recursive=True)
             if os.sep + "venv" + os.sep not in x],
        "data/sim_arc_on_real/arc_on_real_debug.npz (probe_arc_on_real/probe_realism input)":
            os.path.exists(RC.R("data", "sim_arc_on_real", "arc_on_real_debug.npz")),
        "data/sim_nonlens/shards (probe_deflector_gap input, path as coded)":
            os.path.isdir(RC.R("data", "sim_nonlens", "shards")),
        "data_old/sim_nonlens/shards (same name under data_old/)":
            os.path.isdir(RC.R("data_old", "sim_nonlens", "shards")),
    }
    arc_dbg = outputs["data/sim_arc_on_real/arc_on_real_debug.npz (probe_arc_on_real/"
                      "probe_realism input)"]
    res = {
        "paper_text": cite(sec, "AUROC 0.995 on per-image brightness alone") +
                      cite(sec, "brought the residual morphology gap between the classes"),
        "project_record": cite(pj, "a classifier hit AUROC 0.995") +
                          cite(pj, "brought the residual gap to a legitimate ~0.84 morphology"),
        "earlier_search": cite(cur, "the producers of §3's 0.995 and 0.84 **could not be identified**"),
        "auroc_0.995_per_image_brightness": {
            "producer": "not identified",
            "evidence": "no log, output file or note links the value 0.995 to a script; both "
                        "candidates print their AUROC to stdout only (validate_singleband.py's "
                        "one file output is a montage PNG, not found on disk) and no captured "
                        "stdout exists in logs/ or runs/",
        },
        "residual_morphology_gap_0.84": {
            "producer": "not identified",
            "evidence": "as above: probe_arc_on_real.py and probe_realism.py print their "
                        "AUROCs to stdout only and no captured output exists. Their input "
                        "data/sim_arc_on_real/arc_on_real_debug.npz is " +
                        ("on disk" if arc_dbg else "not on disk") + "; re-running either "
                        "would load the frozen Zoobot encoder, which this brief does not allow",
        },
        "candidates_what_their_code_measures": cands,
        "candidate_inputs_and_outputs_on_disk": outputs,
        "search_scope": "text files (*.md *.txt *.log *.json *.py *.csv *.ipynb) under the "
                        "repo excluding venv/ and data dirs, searched for '0.995', "
                        "'brightness shortcut', '3-stat', 'arc_on_real_debug', 'realism'; "
                        "hits for 0.995 are training curves / AUROC columns of unrelated runs "
                        "and the paper/PROJECT text itself",
    }
    RC.write_json("F3_dev_probes.json", res, script=SCRIPT, inputs=I, pool="n/a", folds="n/a")


# ======================================================================= F4
def f4():
    I = RC.Inputs()
    sp = json.load(open(I.add(C.ART["splits"]), encoding="utf-8"))
    folds = sp["real_lenses_kfold"]["folds"]
    F = [f["fold"] for f in folds]
    tl = {f["fold"]: set(f["target_lens_rows"]) for f in folds}
    tn = {f["fold"]: set(int(x) for x in f["target_neg_object_ids"]) for f in folds}
    te = {f["fold"]: set(f["test_lens_rows"]) for f in folds}

    def mat(sets, frac=True):
        out = {}
        for a in F:
            out[str(a)] = {}
            for b in F:
                n = len(sets[a] & sets[b])
                out[str(a)][str(b)] = ({"count": n, "fraction_of_row_fold_pool":
                                        n / len(sets[a])} if frac else n)
        return out
    src = sp["source_train"]
    n_src = {"sim_lens_rows": len(src["sim_lens_rows"]),
             "neg_mlfp": len(src["neg_mlfp"]["object_ids"]),
             "neg_bare_deflector": len(src["neg_bare_deflector"]["deflector_object_ids"]),
             "neg_random": len(src["neg_random"]["object_ids"])}
    n_test_neg = len(sp["test_negatives_shared"]["object_ids"])
    res = {
        "pool_sizes": {"target_lens_rows": {str(f): len(tl[f]) for f in F},
                       "target_neg_object_ids": {str(f): len(tn[f]) for f in F},
                       "test_lens_rows": {str(f): len(te[f]) for f in F},
                       "union_of_target_lens_rows": len(set().union(*tl.values()))},
        "shared_target_lenses": mat(tl),
        "shared_target_negatives": mat(tn),
        "shared_test_lenses_check": mat(te, frac=False),
        "shared_source_rows": {
            "structure": "splits.json has ONE source_train block (no per-fold copy); every "
                         "fold trains on the same source rows",
            "per_pair_count": {str(a): {str(b): sum(n_src.values()) for b in F} for a in F},
            "by_type": n_src},
        "shared_test_negatives": {
            "structure": "splits.json has ONE test_negatives_shared block; every fold is "
                         "scored on the same negatives",
            "per_pair_count_pre_registered_pool": {str(a): {str(b): n_test_neg for b in F}
                                                   for a in F},
            "per_pair_count_D1": {str(a): {str(b): n_test_neg - 265 for b in F} for a in F}},
        "lens_target_vs_test_disjoint_within_fold": {str(f): len(tl[f] & te[f]) == 0 for f in F},
    }
    RC.write_json("F4_fold_overlap.json", res, script=SCRIPT, inputs=I, pool="n/a (splits)",
                  folds=F, notes=["fraction_of_row_fold_pool = shared / size of the row fold's "
                                  "own target pool"])


# ======================================================================= F5
def _auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not pos.size or not neg.size:
        return None
    y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    a = float(roc_auc_score(y, np.r_[pos, neg]))
    return {"auroc": a, "auroc_direction_free": max(a, 1 - a), "n_pos": int(pos.size),
            "n_neg": int(neg.size)}


def _dist(x):
    x = np.asarray(x, float)
    n_all = int(x.size)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "n_total": n_all}
    q = np.percentile(x, [1, 5, 16, 25, 50, 75, 84, 95, 99])
    return {"n": int(x.size), "n_total": n_all, "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else None,
            "min": float(x.min()), "max": float(x.max()),
            "p01": q[0], "p05": q[1], "p16": q[2], "p25": q[3], "p50": q[4], "p75": q[5],
            "p84": q[6], "p95": q[7], "p99": q[8]}


def f5():
    import exact_ids as X
    import pandas as pd
    I = RC.Inputs()
    sp = json.load(open(I.add(C.ART["splits"]), encoding="utf-8"))
    src = sp["source_train"]
    # deflector catalogue (stellar mass, photo-z, mvis), keyed by the stored rounded ID
    dp = I.add(C.ORIG["deflectors"])
    with np.load(dp) as d:
        D = {k: np.asarray(d[k]) for k in ("object_id", "ra", "dec", "stellarmass",
                                           "phz_median", "mvis")}
    r2i = {}
    for i, r in enumerate(D["object_id"].astype("int64")):
        r2i.setdefault(int(r), i)
    d_exact = np.array([X.rebuild(float(a), float(b)) for a, b in zip(D["ra"], D["dec"])],
                       np.int64)
    # training positives' hosts: sim_lens_rows index the concatenation of sim shard lenses
    host_rows = []
    offs, oid = 0, []
    for fn in sp["sim_shards"]:
        p = I.add(RC.R("data", "sim_arc_on_real", fn))
        with np.load(p) as s:
            oid.append(np.asarray(s["object_id"], np.int64))
    oid = np.concatenate(oid)
    rows = np.asarray(src["sim_lens_rows"], int)
    pos_defl_idx = np.array([r2i[int(o)] for o in oid[rows]])
    bare_idx = np.array([r2i[int(o)] for o in src["neg_bare_deflector"]["deflector_object_ids"]])
    # MER query output: VIS PSF flux (uJy), keyed by exact object_id
    mer = RC.mer_table(I)["flux_vis_psf"]

    def mer_mag(ids):
        f = mer.reindex(np.asarray(ids, np.int64)).to_numpy(float)
        return RC.ab_mag(f)
    rand_ids = np.asarray([int(o) for o in src["neg_random"]["object_ids"]], np.int64)
    mlfp_ids = np.asarray([int(o) for o in src["neg_mlfp"]["object_ids"]], np.int64)
    # grade-X catalogue copy (v0.0.1): flux_vis_1fwhm_aper, a different flux definition
    cat1 = C.catalogue("v0.0.1")
    I.add(C.ORIG["catalogue_v0.0.1"])
    cat1 = cat1.drop_duplicates("object_id").set_index("object_id")
    aper = cat1["flux_vis_1fwhm_aper"].reindex(mlfp_ids).to_numpy(float)

    pops = {
        "a_training_positive_hosts": {
            "stellarmass": D["stellarmass"][pos_defl_idx],
            "phz_median": D["phz_median"][pos_defl_idx],
            "mvis_deflector_table": D["mvis"][pos_defl_idx],
            "vis_psf_mag_MER": mer_mag(d_exact[pos_defl_idx]),
        },
        "b1_bare_deflectors": {
            "stellarmass": D["stellarmass"][bare_idx],
            "phz_median": D["phz_median"][bare_idx],
            "mvis_deflector_table": D["mvis"][bare_idx],
            "vis_psf_mag_MER": mer_mag(d_exact[bare_idx]),
        },
        "b2_random_galaxies": {"vis_psf_mag_MER": mer_mag(rand_ids)},
        "b3_grade_X_ml_fp": {"vis_psf_mag_MER": mer_mag(mlfp_ids),
                             "vis_1fwhm_aper_mag_catalogue_v0.0.1": RC.ab_mag(aper)},
    }
    availability = {pk: {vk: f"{int(np.isfinite(v).sum())}/{v.size}" for vk, v in pv.items()}
                    for pk, pv in pops.items()}
    dists = {pk: {vk: _dist(v) for vk, v in pv.items()} for pk, pv in pops.items()}
    # single-variable AUROCs: positives vs every source-train negative that HAS the variable
    au = {}
    negs = ["b1_bare_deflectors", "b2_random_galaxies", "b3_grade_X_ml_fp"]
    for var in ("stellarmass", "phz_median", "mvis_deflector_table", "vis_psf_mag_MER"):
        P = pops["a_training_positive_hosts"].get(var)
        if P is None:
            continue
        rec = {"vs_each_negative_type": {}}
        pooled = []
        for n in negs:
            if var in pops[n]:
                rec["vs_each_negative_type"][n] = _auroc(P, pops[n][var])
                pooled.append(pops[n][var])
            else:
                rec["vs_each_negative_type"][n] = "variable not available for this population"
        rec["vs_all_source_negatives_having_it"] = _auroc(P, np.concatenate(pooled))
        au[var] = rec
    res = {
        "populations": {"a": f"{rows.size} source-train simulated positives (one host per "
                             f"positive; {np.unique(pos_defl_idx).size} distinct deflectors)",
                        "b1": f"{bare_idx.size} bare-deflector negatives",
                        "b2": f"{rand_ids.size} random-galaxy negatives",
                        "b3": f"{mlfp_ids.size} grade-X (ML false-positive) negatives"},
        "variable_sources": {
            "stellarmass, phz_median, mvis_deflector_table": RC.rel(dp) + " (stored deflector "
                "query output; the query itself is not stored, so mvis's band/aperture "
                "definition is not documented on disk)",
            "vis_psf_mag_MER": C.INPUT_LABEL["mer_flux"] + " flux_vis_psf, AB mag = 23.9 - "
                "2.5 log10(uJy), the §3 definition; joined on exact object_id (deflectors: "
                "ID rebuilt from RA/Dec)",
            "vis_1fwhm_aper_mag_catalogue_v0.0.1": C.INPUT_LABEL["catalogue_v0.0.1"] +
                " flux_vis_1fwhm_aper with the same 23.9 - 2.5 log10 conversion (a 1-FWHM "
                "aperture flux, NOT the PSF flux; listed separately)"},
        "availability": availability,
        "distributions": dists,
        "single_variable_auroc_positive_vs_negative": au,
        "auroc_convention": "score = the variable itself, positives labelled 1; "
                            "auroc_direction_free = max(AUROC, 1-AUROC)",
        "catalogue_values_describe_the_HOST": "positives' values are their deflector's; the "
                                              "painted arc is not in any catalogue value",
    }
    RC.write_json("F5_host_properties.json", res, script=SCRIPT, inputs=I,
                  pool="source_train (training data, not a test pool)", folds="n/a (shared)")


# ======================================================================= F6
def f6():
    import exact_ids as X
    I = RC.Inputs()
    for k in ("lens_cutouts", "deflectors"):
        I.add(C.ORIG[k])
    I.add(C.ART["splits"])
    I.add(*sorted(glob.glob(os.path.join(C.ORIG["sim_shards"], "shard_*.npz"))))
    Iden = X.identities()
    S = X.contaminant_sets(Iden)
    sp = Iden["splits"]
    lm = RC.lens_meta(I)
    row_of = {e: i for i, e in enumerate(Iden["lens_row_exact"])}
    fold_of = {}
    for f in sp["real_lenses_kfold"]["folds"]:
        for r in f["test_lens_rows"]:
            fold_of[int(r)] = f["fold"]
    r2e = Iden["defl_r2e"]
    src = sp["source_train"]
    lists = {
        "source_train.neg_random.object_ids": {int(o) for o in src["neg_random"]["object_ids"]},
        "source_train.neg_mlfp.object_ids": {int(o) for o in src["neg_mlfp"]["object_ids"]},
        "source_train.neg_bare_deflector.deflector_object_ids (rounded; exact via RA/Dec)":
            {r2e[int(o)] for o in src["neg_bare_deflector"]["deflector_object_ids"]},
        "val.neg_random.object_ids": {int(o) for o in sp["val"]["neg_random"]["object_ids"]},
        "val.neg_bare_deflector.deflector_object_ids (rounded; exact via RA/Dec)":
            {r2e[int(o)] for o in sp["val"]["neg_bare_deflector"]["deflector_object_ids"]},
        "simulation host (sim_arc_on_real shard object_id -> deflector table, exact via RA/Dec)":
            set(Iden["defl_used_exact"]),
    }
    role_name = {
        "source_train.neg_random.object_ids": "training negative (random galaxy)",
        "source_train.neg_mlfp.object_ids": "training negative (grade-X / ML false positive)",
        "source_train.neg_bare_deflector.deflector_object_ids (rounded; exact via RA/Dec)":
            "training negative (bare deflector)",
        "val.neg_random.object_ids": "validation negative (random galaxy)",
        "val.neg_bare_deflector.deflector_object_ids (rounded; exact via RA/Dec)":
            "validation negative (bare deflector)",
        "simulation host (sim_arc_on_real shard object_id -> deflector table, exact via RA/Dec)":
            "simulation host",
    }
    for f in sp["real_lenses_kfold"]["folds"]:
        k = f"real_lenses_kfold.folds[{f['fold']}].target_neg_object_ids"
        lists[k] = {int(o) for o in f["target_neg_object_ids"]}
        role_name[k] = f"target negative (fold {f['fold']} adaptation pool)"
    rows = []
    for e in sorted(S["d2_pos_remove"], key=lambda e: row_of[e]):
        r = row_of[e]
        hits = [k for k, s in lists.items() if e in s]
        rows.append(dict(lens_row=r, id_str=str(lm["id_str"][r]), exact_object_id=int(e),
                         fold=fold_of[r], grade=str(lm["grade"][r]),
                         roles=[role_name[k] for k in hits], matched_in=hits,
                         match_method="exact Euclid object ID (lens ID from id_str, "
                                      "rebuilt-from-RA/Dec check passed)"))
    assert len(rows) == 20
    tally = {}
    for x in rows:
        for ro in x["roles"]:
            tally[ro] = tally.get(ro, 0) + 1
    n_target = sum(any(k.startswith("real_lenses_kfold") for k in x["matched_in"]) for x in rows)
    res = {"n": len(rows), "lenses": rows, "role_tally": tally,
           "n_d2_lenses_that_are_a_target_negative_in_any_fold": n_target,
           "grade_tally": {g: sum(x["grade"] == g for x in rows) for g in "ABC"},
           "fold_tally": {str(f): sum(x["fold"] == f for x in rows) for f in range(5)},
           "definition": "scripts/exact_ids.contaminant_sets d2_pos_remove = lenses also a "
                         "training negative, validation negative, or simulation host "
                         "(target negatives are listed for information; they are not part "
                         "of the D2 rule)"}
    I.add(RC.R("scripts", "exact_ids.py"))
    RC.write_json("F6_d2_lenses.json", res, script=SCRIPT, inputs=I, pool="D2 removal set",
                  folds=[0, 1, 2, 3, 4])


# ======================================================================= F7
def f7():
    import exact_ids as X
    I = RC.Inputs()
    Iden = X.identities()
    S = X.contaminant_sets(Iden)
    l28 = set(S["lens_in_test"])
    assert len(l28) == 28
    loc2id = Iden["test_neg_loc2id"]
    lm = RC.lens_meta(I)
    row_of = {e: i for i, e in enumerate(Iden["lens_row_exact"])}
    out = {}
    for arm, step in RC.HEADLINE:
        rec = {}
        for f in RC.CONF_FOLDS:
            v = RC.load_vec(RC.step_paths(arm, f)[step], f, arm, "orig", I)
            thr = RC.threshold_of(v)
            ids = np.array([loc2id[l] for l in v["neg_locs"]], np.int64)
            is28 = np.isin(ids, list(l28))
            assert is28.sum() == 28
            # rank among the pre-registered pool's negatives, 1 = highest score
            order = np.argsort(-v["neg"], kind="mergesort")
            rank = np.empty(v["neg"].size, int)
            rank[order] = np.arange(1, v["neg"].size + 1)
            sc = v["neg"][is28]
            pos_rows = set(v["pos_rows"])
            in_fold_pos = [row_of[int(i)] in pos_rows for i in ids[is28]]
            rec[str(f)] = dict(
                threshold=thr, n_neg=int(v["neg"].size),
                n_negatives_at_or_above_threshold=int((v["neg"] >= thr).sum()),
                n28_at_or_above_threshold=int((sc >= thr).sum()),
                median_rank_of_28=float(np.median(rank[is28])),
                ranks_of_28=sorted(int(x) for x in rank[is28]),
                n28_that_are_also_this_folds_test_positives=int(sum(in_fold_pos)),
                tpr_orig=RC.tpr_of(v))
        rec["sum_over_folds_n28_above"] = sum(rec[str(f)]["n28_at_or_above_threshold"]
                                              for f in RC.CONF_FOLDS)
        out[RC.tag(arm, step)] = rec
    grades = {}
    for e in l28:
        g = str(lm["grade"][row_of[e]])
        grades[g] = grades.get(g, 0) + 1
    res = {"per_arm": out, "grades_of_the_28": grades,
           "definition": "the 28 = catalogued graded lenses sharing an exact Euclid ID with a "
                         "shared test negative (exact_ids.contaminant_sets lens_in_test); "
                         "threshold = metrics.operating_threshold at FPR=1e-3 on that fold's "
                         "pre-registered-pool score vector; 'above' means score >= threshold"}
    RC.write_json("F7_lenses_in_test_pool.json", res, script=SCRIPT, inputs=I, pool="orig",
                  folds=RC.CONF_FOLDS)


if __name__ == "__main__":
    which = sys.argv[1:] or ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
    for w in which:
        print(f"== {w}")
        globals()[w]()
