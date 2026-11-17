"""
build_dilute10_pool.py
======================
Builds ``data/splits/dilute10_pool.json`` -- the 1:10 DILUTED target pool for folds 1-4
(brief_dilute10_prereg.md section 2, PROJECTv7.md 7.36).

WHAT IT BUILDS, per fold k in {1, 2, 3, 4}:

  * lenses    : the fold's ``target_lens_rows``, copied unchanged from ``splits.json``;
  * galaxies  : the fold's EXISTING ``target_neg_*`` (unchanged, original order, FIRST),
                then FRESH draws from partition P2, appended, up to exactly
                10 x (number of lenses).

Existing negatives come first and keep their order, so every dataset position of the
original 782/784-image pool is unchanged (``LensDataset`` lists a target fold's lenses,
then its negatives). The DANN/BNM holdout, which is recorded by dataset position, still
points at the same 78 objects.

THE DRAW. Deterministic: ``numpy.random.default_rng([20261013, k])`` -- the seed
namespaced by fold -- choosing without replacement from the SORTED eligible P2 ids, folds
processed in order 1, 2, 3, 4, each fold's draws removed from the eligible set before the
next fold draws (so the fresh draws are disjoint across folds by construction, and the
audit checks it anyway).

ELIGIBILITY. A P2 object is eligible only if it is in NONE of:

  1. any role of ``splits.json`` -- source random / grade-X / bare-deflector negatives,
     the simulation hosts of the source and val sim lenses, val negatives, the shared test
     negatives, all five folds' target negatives, and every real lens row (all 489, taken
     from ``lens_cutouts.npz`` without reading which fold tests which lens);
  2. (fresh draws of another fold -- by construction, then audited);
  3. the exact-ID screens, IDs rebuilt from RA/Dec as ``audit/decontam/rule.json`` defines
     (``exact_ids.rebuild``):
       (a) the 497 catalogued Discovery Engine grade A/B lenses,
       (b) every simulation host galaxy,
       (c) the grade-X hard negatives of the v0.0.1 catalogue.

Negative ids are taken as stored (exact), as rule.json states.

NO TEST LABEL is read: the script never opens a score, a logit or a metric, and never reads
a fold's test-lens assignment. Asserted at start-up with ``code_scan.scan`` against this
file and ``exact_ids.py``, with a known-null that must fire.

    python -u scripts/build_dilute10_pool.py --out data/splits/dilute10_pool.json

Refuses to overwrite without ``--force``. Aborts, writing nothing, on any audit failure.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import code_scan  # noqa: E402
import exact_ids  # noqa: E402
import integrity  # noqa: E402

SEED = 20261013
FOLDS = (1, 2, 3, 4)
RATIO = 10                       # galaxies per lens
P_TARGET_NEG = 2                 # partition role map, as build_splits.py
HOLDOUT_FRAC = 0.10              # dann_round5.DEFAULT_HOLDOUT_FRAC, re-read below

SPLITS = os.path.join(REPO, "data", "splits", "splits.json")
RULE = os.path.join(REPO, "audit", "decontam", "rule.json")
NEG_DIR = os.path.join(REPO, "data", "q1_unlabeled", "shards")
CAT_CUR = os.path.join(REPO, "data_old", "q1_lenses", "q1_discovery_engine_lens_catalog.csv")
CAT_OLD = os.path.join(REPO, "data_old", "q1_lenses",
                       "q1_discovery_engine_lens_catalog_old.csv")
DEFL = os.path.join(REPO, "data_old", "sim_deflectors", "deflector_cutouts_clean.npz")
LENS_NPZ = os.path.join(REPO, "data", "q1_lenses_processed", "lens_cutouts.npz")

N_DE_AB_EXPECTED = 497
N_GRADE_X_EXPECTED = 4940

# tokens whose presence in code would mean a test label, a score or a metric is read
FORBIDDEN = ["test_lens_rows", "test_pos", "metric_suite", "roc_auc_score", "roc_curve",
             "tpr_at_fpr", "auroc", "logits", "pos_scores", "neg_scores", "score_features",
             "load_checkpoint"]
KNOWN_NULL = ("from metrics import metric_suite\n"
              "rows = fold['test_lens_rows']\n"
              "m = metric_suite(y, s)\n")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in sorted(ids)).encode()).hexdigest()


def no_test_label_scan() -> dict:
    out = {}
    for path in (os.path.abspath(__file__), os.path.join(HERE, "exact_ids.py")):
        src = open(path, encoding="utf-8").read()
        r = code_scan.scan(src, FORBIDDEN, known_null_src=KNOWN_NULL)
        if not r["clean"]:
            raise SystemExit(f"[dilute10] ABORT: {os.path.basename(path)} reads a test "
                             f"label / score / metric token: {r['hits']}")
        out[os.path.relpath(path, REPO).replace("\\", "/")] = {
            "clean": True, "known_null_fired_on": r["known_null_hits"]}
    return out


def load_shards():
    """object_id -> (partition, [shard_name, idx]); refuses duplicate ids."""
    part, loc = {}, {}
    for f in sorted(glob.glob(os.path.join(NEG_DIR, "*.npz"))):
        name = os.path.basename(f)
        with np.load(f) as d:
            oids = d["object_ids"].astype("int64")
            parts = d["partitions"].astype(int)
        for i, (o, p) in enumerate(zip(oids, parts)):
            o = int(o)
            if o in part:
                raise SystemExit(f"[dilute10] ABORT: object {o} appears twice in the shards")
            part[o] = int(p)
            loc[o] = [name, int(i)]
    return part, loc


def reference_sets(I) -> dict:
    """The exact-ID screens (criterion 3), rebuilt from RA/Dec."""
    cur = list(csv.DictReader(open(CAT_CUR, encoding="utf-8")))
    de_ab = {exact_ids.rebuild(float(r["right_ascension"]), float(r["declination"]))
             for r in cur if r["subset"] == "discovery_engine" and r["grade"] in ("A", "B")}
    old = list(csv.DictReader(open(CAT_OLD, encoding="utf-8")))
    gx = {exact_ids.rebuild(float(r["right_ascension"]), float(r["declination"]))
          for r in old if r["grade"] == "X"}
    hosts = set(I["defl_used_exact"])
    fail = []
    if len(de_ab) != N_DE_AB_EXPECTED:
        fail.append(f"Discovery Engine grade A/B: {len(de_ab)} unique exact ids, "
                    f"expected {N_DE_AB_EXPECTED}")
    if len(gx) != N_GRADE_X_EXPECTED:
        fail.append(f"grade X: {len(gx)} unique exact ids, expected {N_GRADE_X_EXPECTED}")
    if not set(I["lens_row_exact"]) <= de_ab:
        fail.append("a lens_cutouts row is not among the 497 Discovery Engine A/B lenses")
    if fail:
        raise SystemExit("[dilute10] ABORT (reference sets):\n  " + "\n  ".join(fail))
    return OrderedDict([("a_discovery_engine_grade_AB_lenses", de_ab),
                        ("b_simulation_hosts", hosts),
                        ("c_grade_X_hard_negatives", gx)])


def splits_roles(I) -> dict:
    """Every object in every role of splits.json, as exact ids (criterion 1)."""
    sp = I["splits"]
    r2e = I["defl_r2e"]
    st, va = sp["source_train"], sp["val"]
    roles = OrderedDict()
    roles["source_train.neg_random"] = {int(o) for o in st["neg_random"]["object_ids"]}
    roles["source_train.neg_mlfp"] = {int(o) for o in st["neg_mlfp"]["object_ids"]}
    roles["source_train.neg_bare_deflector"] = {
        r2e[int(o)] for o in st["neg_bare_deflector"]["deflector_object_ids"]}
    roles["source_train+val.sim_lens_hosts"] = set(I["defl_used_exact"])
    roles["val.neg_random"] = {int(o) for o in va["neg_random"]["object_ids"]}
    roles["val.neg_bare_deflector"] = {
        r2e[int(o)] for o in va["neg_bare_deflector"]["deflector_object_ids"]}
    roles["test_negatives_shared"] = {int(o) for o in sp["test_negatives_shared"]["object_ids"]}
    for f in sp["real_lenses_kfold"]["folds"]:
        roles[f"target_fold{f['fold']}.neg"] = {int(o) for o in f["target_neg_object_ids"]}
    roles["real_lens_rows (all 489; covers every target and test lens)"] = set(
        I["lens_row_exact"])
    return roles


def holdout_k(n: int) -> int:
    """dann_round5.build_target_holdout's per-class count for a pool of n."""
    k = int(round(n * float(HOLDOUT_FRAC)))
    return min(max(k, 0), max(n - 1, 0))


def build() -> tuple[dict, dict]:
    import dann_round5                                       # read, not transcribed
    assert dann_round5.DEFAULT_HOLDOUT_FRAC == HOLDOUT_FRAC, dann_round5.DEFAULT_HOLDOUT_FRAC

    I = exact_ids.identities()
    sp = I["splits"]
    part, loc = load_shards()
    refs = reference_sets(I)
    roles = splits_roles(I)

    p2 = sorted(o for o, p in part.items() if p == P_TARGET_NEG)
    p2_set = set(p2)

    # ---- screen-out counts, on the whole P2 partition, before any draw ----
    in_role = set().union(*roles.values())
    in_ref = set().union(*refs.values())
    screened = OrderedDict()
    screened["1_splits_roles_any"] = len(p2_set & in_role)
    for name, s in roles.items():
        screened[f"1_splits_role:{name}"] = len(p2_set & s)
    for name, s in refs.items():
        screened[f"3{name}"] = len(p2_set & s)
    screened["3_any_exact_screen"] = len(p2_set & in_ref)
    screened["3_exact_screen_not_already_in_a_role"] = len((p2_set & in_ref) - in_role)
    screened["total_excluded"] = len(p2_set & (in_role | in_ref))

    eligible = [o for o in p2 if o not in in_role and o not in in_ref]
    folds_by_id = {f["fold"]: f for f in sp["real_lenses_kfold"]["folds"]}
    need = {k: RATIO * len(folds_by_id[k]["target_lens_rows"])
            - len(folds_by_id[k]["target_neg_object_ids"]) for k in FOLDS}
    capacity = {"p2_partition_size": len(p2), "eligible_after_screens": len(eligible),
                "fresh_needed_total": int(sum(need.values())),
                "fresh_needed_per_fold": {str(k): int(v) for k, v in need.items()},
                "sufficient": len(eligible) >= sum(need.values())}
    if not capacity["sufficient"]:
        raise SystemExit(f"[dilute10] ABORT: P2 lacks capacity: {capacity}")

    # ---- the draw ----
    remaining = list(eligible)                               # sorted ascending
    out_folds, fresh_by_fold = [], {}
    for k in FOLDS:
        f = folds_by_id[k]
        rng = np.random.default_rng([SEED, k])
        pool = np.asarray(remaining, dtype=np.int64)
        pick = rng.choice(pool, size=need[k], replace=False)
        fresh = sorted(int(o) for o in pick)
        fresh_by_fold[k] = fresh
        drop = set(fresh)
        remaining = [o for o in remaining if o not in drop]
        existing_ids = [int(o) for o in f["target_neg_object_ids"]]
        existing_loc = [list(x) for x in f["target_neg_locations"]]
        out_folds.append(OrderedDict([
            ("fold", k),
            ("target_lens_rows", list(f["target_lens_rows"])),
            ("target_neg_object_ids", existing_ids + fresh),
            ("target_neg_locations", existing_loc + [loc[o] for o in fresh]),
            ("n_lenses", len(f["target_lens_rows"])),
            ("n_existing_negatives", len(existing_ids)),
            ("n_fresh_negatives", len(fresh)),
            ("n_negatives", len(existing_ids) + len(fresh)),
            ("existing_negatives_are_positions", [len(f["target_lens_rows"]),
                                                  len(f["target_lens_rows"]) + len(existing_ids)]),
        ]))

    # ======================= AUDIT (abort on any failure) =======================
    fail = []
    all_fresh = [o for k in FOLDS for o in fresh_by_fold[k]]
    if len(all_fresh) != len(set(all_fresh)):
        fail.append("fresh draws are not disjoint across folds 1-4")
    for fd in out_folds:
        k = fd["fold"]
        src = folds_by_id[k]
        fresh = set(fresh_by_fold[k])
        if fd["target_lens_rows"] != list(src["target_lens_rows"]):
            fail.append(f"fold {k}: lens rows differ from splits.json")
        if fd["n_negatives"] != RATIO * fd["n_lenses"]:
            fail.append(f"fold {k}: {fd['n_negatives']} negatives != {RATIO} x {fd['n_lenses']}")
        if fd["target_neg_object_ids"][:fd["n_existing_negatives"]] != \
                [int(o) for o in src["target_neg_object_ids"]]:
            fail.append(f"fold {k}: existing negatives changed or reordered")
        if len(set(fd["target_neg_object_ids"])) != fd["n_negatives"]:
            fail.append(f"fold {k}: duplicate negative inside the fold")
        if any(part.get(o) != P_TARGET_NEG for o in fresh):
            fail.append(f"fold {k}: a fresh draw is not from P2")
        if fresh & in_role:
            fail.append(f"fold {k}: {len(fresh & in_role)} fresh draws are in a splits.json role")
        for name, s in refs.items():
            if fresh & s:
                fail.append(f"fold {k}: {len(fresh & s)} fresh draws hit screen {name}")
        for o, l in zip(fd["target_neg_object_ids"], fd["target_neg_locations"]):
            if loc.get(o) != list(l):
                fail.append(f"fold {k}: location of {o} disagrees with the shards")
                break
    # determinism: an independent second draw reproduces every fold
    again, rem = {}, list(eligible)
    for k in FOLDS:
        pick = np.random.default_rng([SEED, k]).choice(np.asarray(rem, dtype=np.int64),
                                                       size=need[k], replace=False)
        again[k] = sorted(int(o) for o in pick)
        rem = [o for o in rem if o not in set(again[k])]
    if again != fresh_by_fold:
        fail.append("the draw is not reproducible")
    if fail:
        raise SystemExit("[dilute10] AUDIT FAILED:\n  " + "\n  ".join(fail))

    # ---- realised ratios ----
    ratios = OrderedDict()
    for fd in out_folds:
        nl, nn = fd["n_lenses"], fd["n_negatives"]
        hl, hn = holdout_k(nl), holdout_k(fd["n_existing_negatives"])
        ratios[str(fd["fold"])] = OrderedDict([
            ("whole_pool", f"{nl}:{nn}"),
            ("whole_pool_lens_fraction", nl / (nl + nn)),
            ("dann_bnm_holdout_kept_byte_identical", f"{hl}+{hn}={hl + hn}"),
            ("dann_bnm_training_pool", f"{nl - hl}:{nn - hn}"),
            ("dann_bnm_training_pool_lens_fraction", (nl - hl) / (nl - hl + nn - hn)),
            ("dann_bnm_training_pool_negatives_per_lens", (nn - hn) / (nl - hl)),
        ])

    # ---- informational: what the EXISTING negatives (kept unchanged) contain ----
    existing_contam = OrderedDict()
    for fd in out_folds:
        ex = set(fd["target_neg_object_ids"][:fd["n_existing_negatives"]])
        existing_contam[str(fd["fold"])] = {name: len(ex & s) for name, s in refs.items()}

    audit = OrderedDict([
        ("passed", True),
        ("checks", [
            "fresh draws disjoint across folds 1-4",
            "fresh draws disjoint from every object in every role of splits.json "
            "(all five target folds, the shared test negatives, source, val, sim hosts, "
            "all 489 lens rows)",
            "fresh draws clear of the 497 Discovery Engine grade A/B lenses, every "
            "simulation host and the 4,940 grade-X hard negatives, by exact Euclid ID",
            "every fresh draw is partition P2",
            "lens rows identical to splits.json; existing negatives unchanged and first",
            "exactly 10 x lenses negatives per fold, no duplicate inside a fold",
            "every location agrees with the shard scan",
            "an independent second draw reproduces every fold",
        ]),
        ("screened_out_of_P2", screened),
        ("screened_out_note", (
            "Counts over the whole P2 partition before any draw. Per-criterion counts "
            "overlap; '3_exact_screen_not_already_in_a_role' is what the exact-ID screens "
            "remove beyond criterion 1.")),
        ("capacity", capacity),
        ("realised_ratios", ratios),
        ("existing_negatives_screen_hits_NOT_REMOVED", existing_contam),
        ("existing_negatives_note", (
            "The brief keeps each fold's existing target negatives unchanged. They were drawn "
            "in round 1 under the rounded-ID purge and are NOT re-screened; these counts "
            "record what they contain, they do not change the pool.")),
    ])

    body = OrderedDict([
        ("schema", "dilute10_pool_v1"),
        ("what", ("1:10 diluted target pool for folds 1-4: each fold's target lenses "
                  "unchanged, its existing target negatives unchanged and first, plus fresh "
                  "P2 draws up to exactly 10 x lenses")),
        ("seed", SEED),
        ("seed_namespacing", "numpy.random.default_rng([seed, fold]); folds drawn in order "
                             "1, 2, 3, 4 from the sorted eligible P2 ids, each fold's draws "
                             "removed before the next"),
        ("ratio_galaxies_per_lens", RATIO),
        ("folds", out_folds),
        ("fold_format", ("same keys as splits.json real_lenses_kfold.folds[*] "
                         "(target_lens_rows, target_neg_object_ids, target_neg_locations); "
                         "LensDataset lists lenses then negatives, so positions "
                         "0..n_lenses+n_existing_negatives-1 are the original pool")),
        ("builder", "scripts/build_dilute10_pool.py"),
        ("builder_sha256", sha256_file(os.path.abspath(__file__))),
        ("inputs_sha256", OrderedDict([
            ("data/splits/splits.json", sha256_file(SPLITS)),
            ("audit/decontam/rule.json", sha256_file(RULE)),
            ("data_old/q1_lenses/q1_discovery_engine_lens_catalog.csv", sha256_file(CAT_CUR)),
            ("data_old/q1_lenses/q1_discovery_engine_lens_catalog_old.csv",
             sha256_file(CAT_OLD)),
            ("data_old/sim_deflectors/deflector_cutouts_clean.npz", sha256_file(DEFL)),
            ("data/q1_lenses_processed/lens_cutouts.npz", sha256_file(LENS_NPZ)),
            ("scripts/exact_ids.py", sha256_file(os.path.join(HERE, "exact_ids.py"))),
        ])),
        ("rule_json_body_sha256", json.load(open(RULE, encoding="utf-8"))
         ["sha256_of_body_without_this_field"]),
        ("identity", ("exact Euclid object ID = sign(dec)*(floor(RA*1e7)*1e9 + "
                      "floor(|Dec|*1e7)) (exact_ids.rebuild), as rule.json; negatives as "
                      "stored (exact)")),
        ("reference_sets", OrderedDict(
            (name, {"n": len(s), "sha256_of_sorted_ids": sha256_ids(s)})
            for name, s in refs.items())),
        ("reference_set_definitions", OrderedDict([
            ("a_discovery_engine_grade_AB_lenses",
             "q1_discovery_engine_lens_catalog.csv rows with subset == 'discovery_engine' "
             "and grade in {A, B}; RA/Dec -> exact id (Lines et al.'s 497)"),
            ("b_simulation_hosts",
             "every deflector galaxy that hosts a simulated lens in data/sim_arc_on_real "
             "shards, exact id from deflector_cutouts_clean.npz RA/Dec "
             "(exact_ids.identities()['defl_used_exact'], as rule.json D1(b))"),
            ("c_grade_X_hard_negatives",
             "q1_discovery_engine_lens_catalog_old.csv (Zenodo v0.0.1) rows with grade X; "
             "RA/Dec -> exact id"),
        ])),
        ("no_test_label_scan", no_test_label_scan()),
        ("audit", audit),
    ])
    return body, audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    if os.path.exists(out) and not a.force:
        raise SystemExit(f"[dilute10] refusing to overwrite {out} (pass --force)")

    integrity.verify(where="(build_dilute10_pool start)")
    print("[dilute10] no-test-label scan:", no_test_label_scan())
    body, audit = build()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1)
        fh.write("\n")
    whole = sha256_file(out)
    integrity.verify(where="(build_dilute10_pool end)")

    print("=" * 78)
    print("DILUTED 1:10 TARGET POOL, folds 1-4")
    print("=" * 78)
    print(f"  file            : {out}")
    print(f"  whole-file sha  : {whole}")
    print(f"  builder sha     : {body['builder_sha256']}")
    print(f"  seed            : {SEED}")
    for fd in body["folds"]:
        r = audit["realised_ratios"][str(fd["fold"])]
        print(f"  fold {fd['fold']}: lenses {fd['n_lenses']}  negatives {fd['n_negatives']} "
              f"(existing {fd['n_existing_negatives']} + fresh {fd['n_fresh_negatives']})  "
              f"pool {r['whole_pool']}  DANN/BNM train {r['dann_bnm_training_pool']}")
    print("  screened out of P2:")
    for k, v in audit["screened_out_of_P2"].items():
        print(f"    {v:6d}  {k}")
    print(f"  capacity: {audit['capacity']}")
    print(f"  existing negatives' screen hits (kept): "
          f"{audit['existing_negatives_screen_hits_NOT_REMOVED']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
