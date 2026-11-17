"""Exact Euclid identities for the decontamination audit. READ-ONLY on the project.

Every path below is opened for reading only; nothing is ever written under PROJECT.

The Euclid Q1 object ID encodes position:
    id = sign(dec) * (floor(RA * 1e7) * 1e9 + floor(|Dec| * 1e7))
so an exact ID can be rebuilt from RA/Dec even where the stored ``object_id`` was
float-rounded. Verified in the previous pass on 489/489 lenses and 50,000/50,000
catalogue rows; re-verified here by ``identities()``.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

PROJECT = r"C:\Users\krris\ML\lens_paper"


def P(*p):
    return os.path.join(PROJECT, *p)


def rebuild(ra: float, dec: float) -> int:
    a = int(np.floor(ra * 1e7 + 1e-6))
    b = int(np.floor(abs(dec) * 1e7 + 1e-6))
    v = a * 10**9 + b
    return -v if dec < 0 else v


def from_idstr(s: str) -> int:
    t = s.split("_", 1)[1]
    return -int(t[3:]) if t.startswith("NEG") else int(t)


def rounded_compare(rounded: set, exact: set) -> set:
    """The ORIGINAL (buggy) comparison: stored rounded IDs against exact IDs."""
    return rounded & exact


def exact_compare(a: set, b: set) -> set:
    return a & b


def identities():
    """Everything the audit needs, as exact IDs. Returns a dict."""
    z = np.load(P("data", "q1_lenses_processed", "lens_cutouts.npz"), allow_pickle=True)
    lens_row_exact = [from_idstr(str(s)) for s in z["id_str"]]
    assert all(rebuild(float(r), float(d)) == e for e, r, d in zip(lens_row_exact, z["ra"], z["dec"]))
    lens_idstr = [str(s) for s in z["id_str"]]

    d = np.load(P("data_old", "sim_deflectors", "deflector_cutouts_clean.npz"), allow_pickle=True)
    r2e = {}
    for r, ra, de in zip(d["object_id"].astype("int64"), d["ra"], d["dec"]):
        e = rebuild(float(ra), float(de))
        assert r2e.setdefault(int(r), e) == e, "rounded deflector ID collides"
    used = set()
    for f in sorted(glob.glob(P("data", "sim_arc_on_real", "shard_*.npz"))):
        with np.load(f) as s:
            used |= {r2e[int(o)] for o in s["object_id"].astype("int64")}

    sp = json.load(open(P("data", "splits", "splits.json"), encoding="utf-8"))
    tn = sp["test_negatives_shared"]
    loc2id = {f"random:{a}:{b}": int(o) for (a, b), o in zip(tn["locations"], tn["object_ids"])}
    return dict(splits=sp, lens_row_exact=lens_row_exact, lens_idstr=lens_idstr,
                defl_used_exact=used, defl_r2e=r2e,
                test_neg_ids=[int(o) for o in tn["object_ids"]], test_neg_loc2id=loc2id)


def contaminant_sets(I):
    """The declared contaminant sets (rule.json), as exact IDs."""
    sp = I["splits"]
    lens_all = set(I["lens_row_exact"])
    test_neg = set(I["test_neg_ids"])
    lens_in_test = lens_all & test_neg                                        # D1 (a)
    defl_in_test = I["defl_used_exact"] & test_neg                            # D1 (b)
    src = set(int(o) for o in sp["source_train"]["neg_random"]["object_ids"])
    src |= set(int(o) for o in sp["source_train"]["neg_mlfp"]["object_ids"])
    src |= {I["defl_r2e"][int(o)] for o in sp["source_train"]["neg_bare_deflector"]["deflector_object_ids"]}
    val = set(int(o) for o in sp["val"]["neg_random"]["object_ids"])
    val |= {I["defl_r2e"][int(o)] for o in sp["val"]["neg_bare_deflector"]["deflector_object_ids"]}
    d2_lenses = (lens_all & src) | (lens_all & val) | (lens_all & I["defl_used_exact"])
    return dict(lens_in_test=lens_in_test, defl_in_test=defl_in_test,
                d1_neg_remove=lens_in_test | defl_in_test,
                lens_in_src=lens_all & src, lens_in_val=lens_all & val,
                lens_as_defl=lens_all & I["defl_used_exact"], d2_pos_remove=d2_lenses)
