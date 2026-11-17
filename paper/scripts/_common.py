"""
_common.py -- shared machinery for every table and figure script in ``paper/scripts``.
=====================================================================================

ONE RULE GOVERNS THIS WHOLE DIRECTORY, and it is the reason this module exists:

    **Every number that reaches a table or a figure is computed from a run artifact.
    Nothing is retyped from PROJECT.md.** PROJECT.md is the CROSS-CHECK
    (``crosscheck.py``), never the source.

So the arm score loaders are imported from ``scripts/analyze_window_average.ARMS`` --
the project's existing single source of truth for where each arm's per-checkpoint
logits live -- and the metric is ``scripts/metrics.metric_suite``, the same function
every arm's own analysis called. Nothing is re-implemented here; a second
implementation that agreed to four decimals would still be a second code path whose
disagreements nobody would ever see.

WHAT THIS MODULE DECLARES (and what it does NOT)
------------------------------------------------
It declares the *bookkeeping* each arm carries -- its declared budget, the step it is
read at, its encoder regime, and where that budget came from. Those are protocol
facts recorded in each arm's own hashed budget/pre-registration file, and
``crosscheck.py`` verifies the budget and read-off of every arm against those files
on disk. It declares no measured quantity.

THE INTERVAL ESTIMATOR
----------------------
``paired_t`` is imported from ``scripts/analyze_lambda0_r6``: a paired t-interval on
n = 4 differences, df = 3, as fixed in the round-5/6 pre-registration reporting block.
Never a bootstrap at n = 4 (4**4 = 256 distinct resamples; its tails are a lattice
artifact). Where a published number came from the earlier fold-aware paired bootstrap
it is READ FROM THAT ARTIFACT rather than re-estimated, and the estimator is named in
the table.

FOLD 0 IS DEVELOPMENT and enters no confirmatory mean. The confirmatory basis is folds
1-4 throughout. Baseline A is a single source-only model, so a folds-0-4 mean is also
legitimate for it and both are computed -- but only the folds-1-4 mean (0.1714) may be
differenced against anything on the confirmatory basis.

THE TEST POOL
-------------
Every stored score vector passes through ``pool_filter`` before any metric sees it. The
default pool is D1 (``audit/decontam/rule.json``): the shared test negatives without the
objects that share an exact Euclid ID with a catalogued lens or a simulation host galaxy.
``PAPER_POOL=orig`` (or ``with pool("orig"):``) gives the pre-registered pool, unfiltered,
on which every table regenerates byte-identical to its pre-v2 state. Baseline A is read
from its stored step-8000 scores, so no head is loaded.

No script in this directory trains anything, touches a GPU, or writes outside
``paper/``.
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys

import numpy as np

# --------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER)
REPO_SCRIPTS = os.path.join(REPO, "scripts")
# PAPER_TABLES_DIR / PAPER_FIGURES_DIR redirect the outputs (used to regenerate the
# pre-registered-pool tables into a scratch directory and diff them against a backup).
TABLES = os.environ.get("PAPER_TABLES_DIR", os.path.join(PAPER, "tables"))
FIGURES = os.environ.get("PAPER_FIGURES_DIR", os.path.join(PAPER, "figures"))

if REPO_SCRIPTS not in sys.path:
    sys.path.insert(0, REPO_SCRIPTS)

import metrics as me                                    # noqa: E402
from analyze_lambda0_r6 import paired_t                 # noqa: E402  (df = n-1)
from analyze_window_average import ARMS                 # noqa: E402  (score loaders)

TARGET_FPR = 1e-3
CONFIRMATORY_FOLDS = (1, 2, 3, 4)
LINES_COMPLETENESS = me.LINES_SOURCE_ONLY_COMPLETENESS   # 0.50
LINES_PURITY = me.LINES_SOURCE_ONLY_PURITY               # 0.24


def R(*p):
    return os.path.join(REPO, *p)


# --------------------------------------------------------------------------------
# artifact locations that are not reached through ARMS
# --------------------------------------------------------------------------------
ART = {
    "lines_ratios": R("runs", "round7", "lines_ratios.json"),
    "window_average": R("runs", "round7", "window_average.json"),
    "window_fold0": R("runs", "round7", "window_fold0.json"),
    "window_prereg": R("runs", "round7", "window_prereg.json"),
    "foldmean_bootstrap": R("runs", "foldmean_bootstrap.json"),
    "lambda0_results": R("runs", "lambda0_r6_results.json"),
    "bnm_results": R("runs", "bnm_r6_folds_results.json"),
    "dann_paired": R("runs", "dann_main", "paired_analysis.json"),
    "coral_budget": R("runs", "coral_r4_budget.json"),
    "dann_budget": R("runs", "dann_r5_budget.json"),
    "bnm_budget": R("runs", "bnm_r6_budget.json"),
    "bnm_prereg": R("runs", "bnm_r6_prereg.json"),
    "lambda0_prereg": R("runs", "lambda0_r6_prereg.json"),
    "dann_prereg": R("runs", "dann_r5_prereg.json"),
    "r1_baseline": R("runs", "bakeoff_baseline_A", "metric_suite.json"),
    "r1_coral": R("runs", "bakeoff_coral", "metric_suite.json"),
    "r1_adamatch": R("runs", "bakeoff_adamatch_5fold", "metric_suite.json"),
    "r1_freematch": R("runs", "bakeoff_freematch_5fold", "metric_suite.json"),
    "r1_dann_probe": R("runs", "curve_dann_fold0", "results.json"),
    "dann_arm1_curve": R("runs", "dann_r5_curves", "arm1_lambda0_curve.csv"),
    "dann_arm2_curve": R("runs", "dann_r5_curves", "arm2_dann_curve.csv"),
    "freematch_cfg": R("runs", "freematch_nb3_unfrozen_fold1", "run_config.json"),
    "splits": R("data", "splits", "splits.json"),
    "lever1_aug_precision": R("runs", "lever1_aug_precision.json"),
    "pool_rule": R("audit", "decontam", "rule.json"),
    "pool_numbers": R("audit", "decontam", "numbers.csv"),
}


# --------------------------------------------------------------------------------
# THE TEST POOL -- one filter, applied to every stored score vector
# --------------------------------------------------------------------------------
# ``D1`` (the default, and the pool the paper reports) removes from every fold's shared
# test negatives each object that shares an exact Euclid ID with a catalogued lens or a
# simulation host galaxy, per ``audit/decontam/rule.json``. ``orig`` is the pre-registered
# pool, untouched. ``D2`` additionally drops the test positives that also appear as a
# training or validation negative or as a simulation deflector (a sensitivity reading
# only). Select with the environment variable PAPER_POOL or ``with pool("orig"):``.
#
# The removal sets come from ``scripts/exact_ids.contaminant_sets`` -- exact identity
# only; no score, threshold or metric is read to decide what goes. Every vector is mapped
# to identities through its run's own ``ids.npz`` (or, for stored score dumps, the ids
# stored beside the scores), and the negative order is asserted equal to splits.json.
POOLS = ("orig", "D1", "D2")
POOL = os.environ.get("PAPER_POOL", "D1")
assert POOL in POOLS, f"PAPER_POOL must be one of {POOLS}, got {POOL!r}"
FILTER_LOG: list = []           # one record per filtered vector: where, pool, counts

_POOLSETS: dict = {}


class pool:
    """``with pool("orig"): ...`` -- compute under another pool, then restore."""

    def __init__(self, name):
        assert name in POOLS, name
        self.name = name

    def __enter__(self):
        global POOL
        self.prev, POOL = POOL, self.name
        return self

    def __exit__(self, *exc):
        global POOL
        POOL = self.prev
        return False


def pool_rule() -> dict:
    """``rule.json``, after checking that it hashes to its own recorded digest."""
    import hashlib
    import json
    rule = json.load(open(ART["pool_rule"], encoding="utf-8"))
    body = {k: v for k, v in rule.items() if k != "sha256_of_body_without_this_field"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"))
                            .encode()).hexdigest()
    assert digest == rule["sha256_of_body_without_this_field"], \
        "audit/decontam/rule.json does not hash to its own recorded digest -- STOP"
    return rule


# --------------------------------------------------------------------------------
# INPUTS THAT CANNOT BE REDISTRIBUTED -- one loader per input
# --------------------------------------------------------------------------------
# A few inputs carry Euclid pixels, catalogue copies, or the project record. Each loader
# below reads the ORIGINAL file exactly as the generators always have when it is present,
# and otherwise its pixel-free extract in ``public_inputs/`` (written by
# ``scripts/export_public_inputs.py``). ``PUBLIC_INPUTS=1`` forces the extracts even when
# the originals exist -- that is how the two paths are proved to give identical outputs.
# The extracts hold only exact Euclid IDs; where an original stores a float-rounded ID the
# loader returns int64(float64(exact)), which the export asserts equals the stored value.
PUBLIC_DIR = os.path.join(REPO, "public_inputs")
ORIG = {
    "lens_cutouts": R("data", "q1_lenses_processed", "lens_cutouts.npz"),
    "deflectors": R("data_old", "sim_deflectors", "deflector_cutouts_clean.npz"),
    "sim_shards": R("data", "sim_arc_on_real"),
    "catalogue_v0.0.1": R("data_old", "q1_lenses", "q1_discovery_engine_lens_catalog_old.csv"),
    "catalogue_v0.0.3": R("data_old", "q1_lenses", "q1_discovery_engine_lens_catalog.csv"),
    "mer_flux": R("data_old", "q1_unlabeled", "metadata", "partitioned_catalog.csv"),
    "mer_flux_tiles": R("data_old", "q1_unlabeled", "metadata",
                        "partitioned_catalog_with_tiles.csv"),
}
PUBLIC = {
    "lens_cutouts": "lens_ids.npz",
    "deflectors": "deflectors.npz",
    "sim_shards": "sim_meta.npz",
    "catalogue_v0.0.1": "discovery_engine_v0.0.1_hard_negatives.csv",
    "catalogue_v0.0.3": "discovery_engine_v0.0.3_hard_negative_rows.csv",
    "mer_flux": "mer_flux_test_negatives.csv",
    "mer_flux_tiles": "mer_flux_test_negatives_tiles.csv",
}
# How an input is named in printed or stored provenance: the same in both paths, so the
# outputs do not depend on which path supplied the values.
INPUT_LABEL = {k: os.path.relpath(v, REPO).replace("\\", "/") for k, v in ORIG.items()}


def use_public(key: str) -> bool:
    """True when ``key`` is read from public_inputs/: forced, or the original is absent."""
    return os.environ.get("PUBLIC_INPUTS") == "1" or not os.path.exists(ORIG[key])


def _stored(exact):
    return np.asarray(exact, dtype=np.int64).astype(np.float64).astype(np.int64)


def lens_ids() -> dict:
    """``id_str``, ``ra``, ``dec`` of the graded lenses, in row order."""
    if use_public("lens_cutouts"):
        with np.load(os.path.join(PUBLIC_DIR, PUBLIC["lens_cutouts"])) as z:
            return {k: np.asarray(z[k]) for k in ("id_str", "ra", "dec")}
    with np.load(ORIG["lens_cutouts"], allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in ("id_str", "ra", "dec")}


def deflector_table() -> dict:
    """The deflector sample's catalogue columns; ``object_id`` is the STORED (rounded) ID."""
    keys = ("object_id", "ra", "dec", "stellarmass", "phz_median", "mvis")
    if use_public("deflectors"):
        with np.load(os.path.join(PUBLIC_DIR, PUBLIC["deflectors"])) as z:
            out = {k: np.asarray(z[k]) for k in keys if k != "object_id"}
            out["object_id"] = _stored(z["object_id_exact"])
        return out
    with np.load(ORIG["deflectors"]) as d:
        return {k: np.asarray(d[k]) for k in keys}


def sim_shards(keys) -> list:
    """``[(shard_name, {key: array})]`` in sorted shard order; ``object_id`` is the stored ID."""
    import glob as _glob
    if use_public("sim_shards"):
        with np.load(os.path.join(PUBLIC_DIR, PUBLIC["sim_shards"])) as z:
            shard = np.asarray(z["shard"])
            cols = {k: (_stored(z["object_id_exact"]) if k == "object_id" else np.asarray(z[k]))
                    for k in keys}
        names = sorted(set(shard.tolist()))
        return [(n, {k: v[shard == n] for k, v in cols.items()}) for n in names]
    out = []
    for p in sorted(_glob.glob(os.path.join(ORIG["sim_shards"], "shard_*.npz"))):
        with np.load(p) as s:
            out.append((os.path.basename(p), {k: np.asarray(s[k]) for k in keys}))
    return out


def catalogue(version: str, **read_csv_kw):
    """The Discovery Engine catalogue (``"v0.0.1"`` or ``"v0.0.3"``), via pandas.read_csv.

    The public extract holds only the rows the build uses -- those of the 4,455 hard
    negatives -- with the columns id_str, grade, right_ascension, declination.
    """
    import pandas as pd
    key = f"catalogue_{version}"
    src = os.path.join(PUBLIC_DIR, PUBLIC[key]) if use_public(key) else ORIG[key]
    return pd.read_csv(src, **read_csv_kw)


def mer_flux(tiles: bool = False, **read_csv_kw):
    """The stored MER query output (``object_id``, ``flux_vis_psf``), via pandas.read_csv.

    ``tiles=True`` is the tile-assigned copy. The public extract holds the 26,399 shared
    test negatives only.
    """
    import pandas as pd
    key = "mer_flux_tiles" if tiles else "mer_flux"
    src = os.path.join(PUBLIC_DIR, PUBLIC[key]) if use_public(key) else ORIG[key]
    return pd.read_csv(src, **read_csv_kw)


def _identities_public(X) -> dict:
    """``exact_ids.identities()`` rebuilt from the loaders above, same checks, same result."""
    import json
    z = lens_ids()
    lens_row_exact = [X.from_idstr(str(s)) for s in z["id_str"]]
    assert all(X.rebuild(float(r), float(d)) == e
               for e, r, d in zip(lens_row_exact, z["ra"], z["dec"]))
    lens_idstr = [str(s) for s in z["id_str"]]
    d = deflector_table()
    r2e = {}
    for r, ra, de in zip(d["object_id"].astype("int64"), d["ra"], d["dec"]):
        e = X.rebuild(float(ra), float(de))
        assert r2e.setdefault(int(r), e) == e, "rounded deflector ID collides"
    used = set()
    for _n, s in sim_shards(["object_id"]):
        used |= {r2e[int(o)] for o in s["object_id"].astype("int64")}
    sp = json.load(open(ART["splits"], encoding="utf-8"))
    # a splits file may carry the deflector IDs exact (the public release) or rounded (the
    # project's); both resolve to the same exact ID
    for e in list(r2e.values()):
        assert r2e.setdefault(int(e), e) == e, "exact deflector ID collides with a rounded one"
    tn = sp["test_negatives_shared"]
    loc2id = {f"random:{a}:{b}": int(o) for (a, b), o in zip(tn["locations"], tn["object_ids"])}
    return dict(splits=sp, lens_row_exact=lens_row_exact, lens_idstr=lens_idstr,
                defl_used_exact=used, defl_r2e=r2e,
                test_neg_ids=[int(o) for o in tn["object_ids"]], test_neg_loc2id=loc2id)


def pool_sets() -> dict:
    """The exact-ID removal sets and identity maps, with the rule's counts asserted."""
    if _POOLSETS:
        return _POOLSETS
    import exact_ids as X                                # scripts/exact_ids.py
    rule = pool_rule()
    exp = rule["D1_primary_test_side"]["expected_counts_from_part_A"]
    public = any(use_public(k) for k in ("lens_cutouts", "deflectors", "sim_shards"))
    I = _identities_public(X) if public else X.identities()
    S = X.contaminant_sets(I)
    assert len(S["lens_in_test"]) == exp["a_lens_positives"] == 28, len(S["lens_in_test"])
    assert len(S["defl_in_test"]) == exp["b_deflector_hosts"] == 237, len(S["defl_in_test"])
    assert len(S["d1_neg_remove"]) == 265, len(S["d1_neg_remove"])
    assert len(S["d2_pos_remove"]) == 20, len(S["d2_pos_remove"])
    tn = I["splits"]["test_negatives_shared"]
    _POOLSETS.update(
        rule_sha256=rule["sha256_of_body_without_this_field"],
        rule_recorded_utc=rule["recorded_utc"],
        neg_remove=set(S["d1_neg_remove"]), pos_remove=set(S["d2_pos_remove"]),
        n_lens_in_test=len(S["lens_in_test"]), n_defl_in_test=len(S["defl_in_test"]),
        loc2id=I["test_neg_loc2id"], row2id=I["lens_row_exact"],
        defl_used=set(I["defl_used_exact"]), splits=I["splits"],
        split_neg_locs=[f"random:{a}:{b}" for a, b in tn["locations"]])
    return _POOLSETS


def pool_keep_neg(neg_ids):
    """Boolean keep-mask over a test-negative id vector under D1/D2 (all True for orig)."""
    S = pool_sets()
    return np.array([S["loc2id"][str(l)] not in S["neg_remove"] for l in neg_ids])


# Artifacts whose numbers depend on the pool. Under orig they are the project's own
# files (ART); under D1/D2 they are the recomputations build_pool_recomputes.py writes
# into paper/data/, from the same stored scores by the same code.
POOLED_ART = {"lines_ratios", "foldmean_bootstrap", "window_average", "lever1_aug_precision",
              "window_fold0"}
# PAPER_DATA_DIR redirects the intermediate JSON directory, read and written (used to run the
# build into a scratch copy and diff it against paper/data).
DATA = os.environ.get("PAPER_DATA_DIR", os.path.join(PAPER, "data"))


def art(key: str) -> str:
    """Path of artifact ``key`` for the current pool."""
    if POOL == "orig" or key not in POOLED_ART:
        return ART[key]
    p = os.path.join(DATA, f"{key}_{POOL}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing -- run paper/scripts/build_pool_recomputes.py")
    return p


def pool_filter(pos, neg, pos_ids, neg_ids, where="", pool_name=None):
    """THE filter. Returns (pos, neg) restricted to the current pool.

    ``neg_ids`` must equal splits.json's test-negative locations, in order. ``pos_ids``
    may be None only where no positive is ever removed (orig, D1). Under D1 it asserts
    exactly 265 negatives and 0 positives removed from the vector; under D2, 265
    negatives and only positives in the declared set.
    """
    p = POOL if pool_name is None else pool_name
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if p == "orig":
        FILTER_LOG.append(dict(where=where, pool=p, n_pos=int(pos.size), n_neg=int(neg.size),
                               pos_removed=0, neg_removed=0))
        return pos, neg
    S = pool_sets()
    neg_ids = [str(x) for x in neg_ids]
    assert neg_ids == S["split_neg_locs"], f"{where}: negative order differs from splits.json"
    assert len(neg_ids) == neg.size, f"{where}: {len(neg_ids)} ids for {neg.size} negatives"
    keep_neg = pool_keep_neg(neg_ids)
    keep_pos = np.ones(pos.size, bool)
    if p == "D2":
        assert pos_ids is not None, f"{where}: D2 needs positive ids and none are stored"
        pos_x = [S["row2id"][int(str(l).rsplit(":", 1)[1])] for l in pos_ids]
        assert len(pos_x) == pos.size, where
        keep_pos = np.array([x not in S["pos_remove"] for x in pos_x])
    n_neg_rm, n_pos_rm = int((~keep_neg).sum()), int((~keep_pos).sum())
    assert n_neg_rm == 265, f"{where}: {n_neg_rm} negatives removed, rule says 265"
    if p == "D1":
        assert n_pos_rm == 0, f"{where}: D1 removed {n_pos_rm} positives, rule says 0"
    FILTER_LOG.append(dict(where=where, pool=p, n_pos=int(pos.size), n_neg=int(neg.size),
                           pos_removed=n_pos_rm, neg_removed=n_neg_rm))
    return pos[keep_pos], neg[keep_neg]


def _ids_npz_for(path):
    for d in (os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
        p = os.path.join(d, "ids.npz")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no ids.npz beside {path}")


def _stored_ids(path, fold):
    """(pos_ids, neg_ids) for a per-checkpoint logits file, from its run's ids.npz."""
    z = np.load(_ids_npz_for(path))
    if "neg_ids" in z.files:                     # CORAL, DANN, lambda=0, BNM layout
        return list(z["pos_ids"]), list(z["neg_ids"])
    return list(z[f"test_pos_fold{fold}"]), list(z["test_negatives_shared"])  # FreeMatch


import analyze_window_average as _AWA                   # noqa: E402

_RAW_NPZ = _AWA._npz_pos_neg


def _pooled_npz(path, fold):
    pos, neg = _RAW_NPZ(path, fold)
    pos_ids, neg_ids = _stored_ids(path, fold)
    return pool_filter(pos, neg, pos_ids, neg_ids, where=f"{path} fold {fold}")


def baseline_stored(fold):
    """Baseline A's STORED step-8000 scores (no head is loaded), through the pool filter.

    ``runs/bakeoff_baseline_A/scores/step_8000_fold{f}.npz`` carries the scores and their
    ids; on the pre-registered pool it reproduces every Baseline A value in the tables.
    """
    p = R("runs", "bakeoff_baseline_A", "scores", f"step_8000_fold{fold}.npz")
    z = np.load(p, allow_pickle=False)
    pos, neg = pool_filter(z["pos_scores"], z["neg_scores"], list(z["pos_ids"]),
                           list(z["neg_ids"]), where=f"{p}")
    return {8000: (pos, neg)}


def round1_scores(run: str, fold: int):
    """(pos, neg) of a round-1 run's stored ``scores/*_fold{k}.npz``, through the filter.

    Round-1 score dumps store negative ids but no positive ids, so they support orig and
    D1 (which removes no positive) and refuse D2.
    """
    import glob as _glob
    hits = sorted(_glob.glob(R("runs", run, "scores", f"*_fold{fold}.npz")))
    assert len(hits) == 1, (run, fold, hits)
    z = np.load(hits[0], allow_pickle=False)
    pos_ids = list(z["pos_ids"]) if "pos_ids" in z.files else None
    return pool_filter(z["pos_scores"], z["neg_scores"], pos_ids, list(z["neg_ids"]),
                       where=hits[0])


def round1_summary(run: str, folds=(0, 1, 2, 3, 4)) -> dict:
    """Round-1 5-fold summary from the stored score dumps, on the current pool.

    The same quantities ``scripts/report_bakeoff.compute`` stores in the run's
    ``metric_suite.json`` -- per-fold ``metric_suite``, the across-fold mean and
    POPULATION sd (``np.nanstd``), and the pooled single-ROC suite where every fold shares
    the identical negative vector (a single model, i.e. Baseline A only).
    """
    per, negs, poss = {}, [], []
    for f in folds:
        pos, neg = round1_scores(run, f)
        y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
        per[f] = dict(me.metric_suite(y, np.concatenate([pos, neg]), target_fpr=TARGET_FPR))
        poss.append(pos)
        negs.append(neg)
    out = {"per_fold": per, "across_fold": {}, "pooled": None}
    for key in ("tpr", "comp_at_purity"):
        v = np.array([per[f][key] for f in folds], float)
        out["across_fold"][key] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                                   "n": int(np.isfinite(v).sum())}
    if all(n.shape == negs[0].shape and np.array_equal(n, negs[0]) for n in negs[1:]):
        pp = np.concatenate(poss)
        y = np.concatenate([np.ones(pp.size, int), np.zeros(negs[0].size, int)])
        out["pooled"] = dict(me.metric_suite(y, np.concatenate([pp, negs[0]]),
                                             target_fpr=TARGET_FPR))
    return out


# Route every arm loader through the filter. ``_scan_dir`` looks ``_npz_pos_neg`` up at
# call time, so rebinding the module global reaches every per-checkpoint loader; the
# Baseline A entry is replaced by the stored-score loader (no model is loaded).
_AWA._npz_pos_neg = _pooled_npz
ARMS["Baseline A (frozen)"]["load"] = baseline_stored


# --------------------------------------------------------------------------------
# ARM BOOKKEEPING -- protocol facts, verified against the hashed files by crosscheck.py
# --------------------------------------------------------------------------------
# key -> the name used by scripts/analyze_window_average.ARMS
ARM_KEY = {
    "baseline":  "Baseline A (frozen)",
    "lambda0":   "lambda=0",
    "bnm":       "BNM x nb3",
    "freematch": "FreeMatch x nb3",
    "dann":      "DANN x nb3",
    "coral":     "CORAL x nb3",
}

# Display order for the main table: source-only references first, then the DA arms.
MAIN_ORDER = ["baseline", "lambda0", "bnm", "freematch", "dann", "coral"]

ARM_META = {
    "baseline": dict(
        label="Baseline A (frozen)", short="Baseline A",
        budget=8000, readout=8000, regime="frozen (nb0)",
        budget_basis="round-1 matched protocol (8000 steps, R1 = last checkpoint)",
        budget_file=None,
        note="single source-only model; row uses its folds 1-4 mean",
    ),
    "lambda0": dict(
        label=r"$\lambda=0$ unfrozen control", short=r"$\lambda=0$",
        budget=4000, readout=4000, regime="unfrozen (nb3)",
        budget_basis="declared: the control is read where its treatment is read",
        budget_file="lambda0_prereg",
    ),
    "bnm": dict(
        label=r"BNM $\times$ nb3", short="BNM",
        budget=1000, readout=4000, regime="unfrozen (nb3)",
        budget_basis=r"rule-derived (2 $\times$ nuclear-norm crossing); "
                     r"\texttt{no\_matched\_arm\_budget}",
        budget_file="bnm_budget",
        note="read at the mandatory 4000 readout; 1000 is its own-objective reading only",
    ),
    "freematch": dict(
        label=r"FreeMatch $\times$ nb3", short="FreeMatch",
        budget=1500, readout=1500, regime="unfrozen (nb3)",
        budget_basis="pre-registered (amendment 1)",
        budget_file=None,
    ),
    "dann": dict(
        label=r"DANN $\times$ nb3", short="DANN",
        budget=4000, readout=4000, regime="unfrozen (nb3)",
        budget_basis=r"declared under a named rule (2 $\times$ the $\lambda$ horizon)",
        budget_file="dann_budget",
    ),
    "coral": dict(
        label=r"CORAL $\times$ nb3", short="CORAL",
        budget=2000, readout=2000, regime="unfrozen (nb3)",
        budget_basis="declared on cost",
        budget_file="coral_budget",
    ),
}

# Every readout each arm's logits support and this paper reports, from
# analyze_window_average.ARMS (the arms' own declared mandatory readouts).
READOUTS = {k: list(ARMS[v]["readouts"]) for k, v in ARM_KEY.items()}


# --------------------------------------------------------------------------------
# THE CONTRAST SET -- declared once, used by T2 and by F3 so they cannot disagree
# --------------------------------------------------------------------------------
# (panel, label, arm_a, readout_a, arm_b, readout_b, budget-match state, footnote)
#
# ``budget`` is "matched" only where both arms are read at the SAME step. A match
# removes the BUDGET confound and nothing else -- head initialisation and each arm's own
# tuning budget still travel with every row, so DANN - lambda=0 is the only fully
# controlled pair here. Budget-unmatched forms are kept ONLY where PROJECT.md keeps
# them, and are labelled unmatched in their own row.
MATCHED, UNMATCHED, NOT_APPLICABLE = "matched", r"\textbf{unmatched}", "---"

CONTRAST_ROWS = [
    ("A", r"$\lambda=0$ $-$ Baseline A", "lambda0", 4000, "baseline", 8000,
     NOT_APPLICABLE, r"$^{a}$"),
    ("B", r"DANN $-$ $\lambda=0$", "dann", 4000, "lambda0", 4000, MATCHED, r"$^{b}$"),
    ("B", r"CORAL $-$ $\lambda=0$", "coral", 2000, "lambda0", 2000, MATCHED, ""),
    ("B", r"FreeMatch $-$ $\lambda=0$", "freematch", 1500, "lambda0", 1500, MATCHED, ""),
    ("B", r"BNM $-$ $\lambda=0$", "bnm", 4000, "lambda0", 4000, MATCHED, ""),
    ("B", r"CORAL $-$ $\lambda=0$ (at 4000)", "coral", 2000, "lambda0", 4000,
     UNMATCHED, r"$^{c}$"),
    ("B", r"FreeMatch $-$ $\lambda=0$ (at 4000)", "freematch", 1500, "lambda0", 4000,
     UNMATCHED, r"$^{c}$"),
    ("C", r"BNM $-$ CORAL", "bnm", 2000, "coral", 2000, MATCHED, ""),
    ("C", r"BNM $-$ DANN", "bnm", 4000, "dann", 4000, MATCHED, ""),
    ("C", r"BNM $-$ FreeMatch", "bnm", 1500, "freematch", 1500, MATCHED, ""),
    ("C", r"DANN $-$ CORAL", "dann", 4000, "coral", 2000, UNMATCHED, r"$^{d}$"),
    ("C", r"DANN $-$ FreeMatch", "dann", 4000, "freematch", 1500, UNMATCHED, r"$^{d}$"),
    ("C", r"DANN $-$ CORAL (both at 2000)", "dann", 2000, "coral", 2000, MATCHED, ""),
]

# Pre-registration status of each CONTRAST_ROWS entry, by row index (rewrite v1). Kept as a
# parallel map rather than a ninth tuple field, because T2, F3 and A3 unpack the rows as
# 8-tuples and verify_orig_tables.py checks their outputs byte for byte. Source:
# audit/review_r1/out/F1_prereg_contrasts.json (crosscheck.py asserts the two agree):
# rows 0, 1, 5, 6 are the four contrasts of runs/lambda0_r6_prereg.json in their headline
# form (lambda=0 read at its budget, 4000); rows 2 and 3 are the same contrasts at the
# pre-declared sensitivity readouts (1500, 2000); every other row is neither.
PREREG, SENSITIVITY, NEITHER = "prereg", "sensitivity", "neither"
CONTRAST_PREREG = {0: PREREG, 1: PREREG, 5: PREREG, 6: PREREG,
                   2: SENSITIVITY, 3: SENSITIVITY,
                   4: NEITHER, 7: NEITHER, 8: NEITHER, 9: NEITHER, 10: NEITHER,
                   11: NEITHER, 12: NEITHER}
assert sorted(CONTRAST_PREREG) == list(range(len(CONTRAST_ROWS)))

PANEL_TITLE = {
    "A": "Frozen against unfrozen treatment",
    "B": "A target-domain training signal vs. none",
    "C": "Objective against objective",
}


# --------------------------------------------------------------------------------
# the measurement -- one metric suite per (arm, readout, fold), straight from logits
# --------------------------------------------------------------------------------
_SUITE_CACHE: dict = {}


def suite(arm: str, fold: int, readout: int) -> dict:
    """``metrics.metric_suite`` on the arm's own stored logits for that fold and step.

    Returns the dict of scalars (tpr, purity_corr, purity_raw, auprc, comp_at_purity,
    auroc) plus n_pos / n_neg. ``comp_at_purity`` is NaN where 24% purity is
    unreachable -- that NaN is carried through and printed as n/a, never interpolated.
    """
    key = (POOL, arm, fold, readout)
    if key in _SUITE_CACHE:
        return _SUITE_CACHE[key]
    spec = ARMS[ARM_KEY[arm]]
    by_step = spec["load"](fold)
    if readout not in by_step:
        raise KeyError(f"{arm} fold {fold}: no stored logits at step {readout} "
                       f"(available: {sorted(by_step)})")
    pos, neg = by_step[readout]
    y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
    out = dict(me.metric_suite(y, np.concatenate([pos, neg]), target_fpr=TARGET_FPR))
    out["n_pos"] = int(pos.size)
    out["n_neg"] = int(neg.size)
    _SUITE_CACHE[key] = out
    return out


def per_fold(arm: str, metric: str = "tpr", readout: int | None = None,
             folds=CONFIRMATORY_FOLDS) -> dict:
    """{fold: value} for one metric, at the arm's read-off step unless told otherwise."""
    ro = ARM_META[arm]["readout"] if readout is None else readout
    return {f: float(suite(arm, f, ro)[metric]) for f in folds}


def fold_mean(arm: str, metric: str = "tpr", readout: int | None = None,
              folds=CONFIRMATORY_FOLDS):
    """(mean, sd, {fold: value}). Returns (nan, nan, values) if ANY fold is undefined.

    The project's rule for a metric undefined on any fold is n/a for the whole arm --
    no fold-mean, no interval, never an average over the reachable subset.
    """
    v = per_fold(arm, metric, readout, folds)
    vals = list(v.values())
    if any(not math.isfinite(x) for x in vals):
        return float("nan"), float("nan"), v
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else float("nan")), v


def contrast(arm_a: str, ro_a: int, arm_b: str, ro_b: int, metric: str = "tpr",
             folds=CONFIRMATORY_FOLDS) -> dict:
    """Paired t-interval, df = n-1, on (arm_a at ro_a) - (arm_b at ro_b), per fold."""
    a = per_fold(arm_a, metric, ro_a, folds)
    b = per_fold(arm_b, metric, ro_b, folds)
    diffs = [a[f] - b[f] for f in folds]
    out = paired_t(diffs)
    # Folds agreeing with the SIGN OF THE POINT ESTIMATE -- which is what the tables and
    # the forest plot say the column counts, so the code says the same thing.
    sign = 1.0 if out["mean"] >= 0 else -1.0
    out["n_same_sign"] = sum(1 for d in diffs if d * sign > 0)
    out["a"], out["b"] = a, b
    return out


# --------------------------------------------------------------------------------
# LaTeX helpers
# --------------------------------------------------------------------------------
NA = r"\emph{n/a}"


def f4(x, nd=4, signed=False):
    """Format a rate in math mode, so a minus sign renders as one. n/a stays n/a.

    Undefined values print as n/a and never acquire an interval -- no interpolation,
    no average over the reachable subset.
    """
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return NA
    return f"${x:+.{nd}f}$" if signed else f"${x:.{nd}f}$"


def rounds_to_zero(x, nd=4) -> bool:
    """True for a non-zero value that prints as zero at ``nd`` decimals (signed, it would
    print as ``-0.0000`` or ``+0.0000``). Such a cell prints unsigned and is footnoted with
    :func:`unrounded`, so a sign count beside it still has a visible referent."""
    return (x is not None and isinstance(x, float) and math.isfinite(x) and x != 0.0
            and round(x, nd) == 0.0)


def unrounded(x) -> str:
    """``x`` signed, to its first significant digit (-2.63e-05 -> ``$-0.00003$``)."""
    nd = max(1, -math.floor(math.log10(abs(x))))
    return f"${x:+.{nd}f}$"


def ci(lo, hi, nd=4):
    if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi):
        return NA
    return f"$[{lo:+.{nd}f}, {hi:+.{nd}f}]$"


def pct(x, nd=1, signed=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return NA
    return (f"${x:+.{nd}f}\\%$" if signed else f"${x:.{nd}f}\\%$")


def write_table(path, body: str, header_comment: str = "", size: str = "footnotesize",
                colsep: str = "4pt", placement: str = "tbp") -> None:
    """Write a generated .tex table, stamped so no reader mistakes it for hand-typed.

    Three normalisations happen here so that every table gets them identically and no
    generator can forget one:

      * **starred floats are unstarred.** ``table*`` is legal in a one-column document
        (LaTeX falls back to the ordinary float) but it buys nothing there, and the
        document class for this stage is single-column ``article``. Swapping in a
        two-column journal style later is then one edit in this function rather than
        six edits across the generators.
      * **a size command and a column separation are inserted before the tabular**, so
        the wide tables fit the text block without anyone hand-tuning a float.
      * **every table float gets one placement, ``[tbp]`` by default.** A ``[t]``- or
        ``[h]``-only float that is taller than the top-float area cannot be placed in
        the text and is deferred to the end of the document; ``[tbp]`` lets it take a
        float page instead, and the float barrier that ``placeins`` puts at every
        section heading (main.tex) keeps it inside the section that inputs it. The
        appendix generators pass ``placement="htbp"``: an appendix that is little
        more than its table should have the table under its heading.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = body.replace(r"\begin{table*}", r"\begin{table}") \
               .replace(r"\end{table*}", r"\end{table}")
    import re
    body = re.sub(r"\\begin\{table\}\[[a-zA-Z!]*\]",
                  lambda m: r"\begin{table}[" + placement + "]", body)
    if size:
        body = body.replace(
            r"\begin{tabular}",
            "\\" + size + "\n\\setlength{\\tabcolsep}{" + colsep + "}\n"
            + r"\begin{tabular}")
    script = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "make_all.py"
    stamp = ("% GENERATED FILE -- do not edit by hand.\n"
             "% Rebuild:  python make_all.py (generator: " + script + ")\n")
    if header_comment:
        stamp += "".join(f"% {ln}\n" for ln in header_comment.strip().splitlines())
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(stamp + body.rstrip() + "\n")
    print(f"  wrote {os.path.relpath(path, REPO)}")


def save_fig(fig, path) -> None:
    """Vector PDF, tight bounds, no title inside the figure (the caption does that).

    Setting ``PAPER_FIG_PNG_DIR`` additionally drops a raster preview there -- a
    convenience for eyeballing a figure without a PDF viewer. It never writes a preview
    into ``paper/`` and the PDF is unaffected.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"  wrote {os.path.relpath(path, REPO)}")
    preview = os.environ.get("PAPER_FIG_PNG_DIR")
    if preview:
        os.makedirs(preview, exist_ok=True)
        png = os.path.join(preview,
                           os.path.splitext(os.path.basename(path))[0] + ".png")
        fig.savefig(png, format="png", dpi=190, bbox_inches="tight", pad_inches=0.02)
        print(f"  preview {png}")


# --------------------------------------------------------------------------------
# figure style -- colour-blind-safe (Okabe-Ito), readable at single-column width
# --------------------------------------------------------------------------------
# Okabe & Ito (2008), the standard colour-blind-safe qualitative palette.
OKABE_ITO = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky":        "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "grey":       "#999999",
}

ARM_COLOUR = {
    "baseline":  OKABE_ITO["grey"],
    "lambda0":   OKABE_ITO["black"],
    "bnm":       OKABE_ITO["orange"],
    "freematch": OKABE_ITO["blue"],
    "dann":      OKABE_ITO["vermillion"],
    "coral":     OKABE_ITO["green"],
}

ARM_MARKER = {"baseline": "s", "lambda0": "D", "bnm": "^",
              "freematch": "o", "dann": "v", "coral": "P"}


def use_paper_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,          # embed TrueType, not Type 3 -- arXiv-safe
        "ps.fonttype": 42,
        "figure.constrained_layout.use": False,
    })
    return plt
