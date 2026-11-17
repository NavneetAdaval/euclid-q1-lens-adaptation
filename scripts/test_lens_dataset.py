"""
test_lens_dataset.py
====================
Self-auditing harness for `LensDataset`, in the spirit of build_splits.py: it fabricates
a tiny dataset whose every pixel encodes the identity of the array element it came from,
builds a splits_v6 file that references known elements, then asserts the dataset fetched
exactly the intended element for every reference. Exits non-zero on any failure.

The identity trick: image (shard s, row r, source-kind k) is filled with the constant
    enc(k, s, r) = k*100000 + s*1000 + r
which is exactly representable in float32 and trivially decodable from get_raw().

Hardest thing verified: `sim_lens_rows` resolve against the CONCATENATION of sim shards
in `sim_shards` ORDER, which is made to differ from sorted-filename order, so an
order-agnostic bug would fetch the wrong shard and be caught.
"""

import os
import sys
import json
import tempfile
import traceback

import numpy as np
import torch

from lens_dataset import LensDataset, LensTransform, zoobot_stretch, UNLABELED, SOURCE, TARGET

# kind codes for the identity encoding
K_SIMLENS, K_SIMNON, K_MLFP, K_UNLAB, K_LENSCUT = 1, 2, 3, 4, 5
KIND_NAME = {K_SIMLENS: "sim_lens", K_SIMNON: "bare_deflector", K_MLFP: "mlfp",
             K_UNLAB: "random", K_LENSCUT: "real_lens"}

# shard-id assignment for encoding (by filename, independent of splits order)
SIM_SID = {"shard_a.npz": 1, "shard_b.npz": 2, "shard_c.npz": 3}
UNL_SID = {"u0.npz": 1, "u1.npz": 2}
MLFP_SID = {"mlfp0.npz": 1}
LENSCUT_SID = 0


def enc(k, s, r):
    return float(k * 100000 + s * 1000 + r)


def dec(x):
    v = int(round(float(x)))
    return (v // 100000, (v % 100000) // 1000, v % 1000)


def const_img(val, n=64):
    return np.full((n, n), val, dtype=np.float32)


# ---------------------------------------------------------------------------
# result tracking
# ---------------------------------------------------------------------------
_fails = []
def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _fails.append(msg)


# ---------------------------------------------------------------------------
# build the fake dataset on disk
# ---------------------------------------------------------------------------
def build_mock(root):
    sim_dir = os.path.join(root, "sim_arc_on_real")
    unl_dir = os.path.join(root, "q1_unlabeled", "shards")
    mlfp_dir = os.path.join(root, "ml_fp_cutouts", "shards")
    lc_dir = os.path.join(root, "q1_lenses_processed")
    for d in (sim_dir, unl_dir, mlfp_dir, lc_dir):
        os.makedirs(d, exist_ok=True)

    # sim shards: UNEVEN lens counts; identity encodes (kind, shard-id, local row)
    sim_counts = {"shard_a.npz": (5, 3), "shard_b.npz": (3, 2), "shard_c.npz": (4, 2)}  # (lenses, nonlenses)
    for fn, (nl, nn) in sim_counts.items():
        sid = SIM_SID[fn]
        lenses = np.stack([const_img(enc(K_SIMLENS, sid, r)) for r in range(nl)])
        nonl = np.stack([const_img(enc(K_SIMNON, sid, r)) for r in range(nn)])
        np.savez(os.path.join(sim_dir, fn),
                 lenses=lenses.astype(np.float32),
                 nonlenses=nonl.astype(np.float32),
                 object_id=np.arange(nl))

    # unlabeled shards: 6 images each
    for fn in ("u0.npz", "u1.npz"):
        sid = UNL_SID[fn]
        imgs = np.stack([const_img(enc(K_UNLAB, sid, r)) for r in range(6)])
        np.savez(os.path.join(unl_dir, fn),
                 images=imgs.astype(np.float32),
                 object_ids=np.arange(6) + sid * 1000,
                 partitions=np.zeros(6, dtype=int))

    # ml-fp shard: 4 images
    imgs = np.stack([const_img(enc(K_MLFP, MLFP_SID["mlfp0.npz"], r)) for r in range(4)])
    np.savez(os.path.join(mlfp_dir, "mlfp0.npz"),
             images=imgs.astype(np.float32), object_ids=np.arange(4))

    # lens cutouts: 10 real lenses
    imgs = np.stack([const_img(enc(K_LENSCUT, LENSCUT_SID, r)) for r in range(10)])
    np.savez(os.path.join(lc_dir, "lens_cutouts.npz"),
             images=imgs.astype(np.float32),
             grade=np.array(["A"] * 10), object_id=np.arange(10))


def build_splits():
    # sim_shards order deliberately NOT sorted-filename order.
    # order [b(3), a(5), c(4)] -> concat offsets  b:[0,3)  a:[3,8)  c:[8,12)
    return {
        "schema": "splits_v6_label_free_da",
        "seed": 42, "n_folds": 2, "fold_mode": "stratified_group",
        "sim_shards": ["shard_b.npz", "shard_a.npz", "shard_c.npz"],
        "source_train": {
            # global rows 0->b.0, 6->a.3 (the order-sensitive one), 11->c.3
            "sim_lens_rows": [0, 6, 11],
            "neg_mlfp": {"object_ids": [0, 2],
                         "locations": [["mlfp0.npz", 0], ["mlfp0.npz", 2]]},
            "neg_bare_deflector": {"deflector_object_ids": [0, 0],
                                   "locations": [["shard_a.npz", 1], ["shard_c.npz", 0]]},
            "neg_random": {"object_ids": [0, 0],
                           "locations": [["u0.npz", 0], ["u1.npz", 1]]},
        },
        "val": {
            "sim_lens_rows": [3, 8],  # 3->a.0, 8->c.0
            "neg_bare_deflector": {"deflector_object_ids": [0],
                                   "locations": [["shard_b.npz", 0]]},
            "neg_random": {"object_ids": [0], "locations": [["u1.npz", 0]]},
        },
        "test_negatives_shared": {"object_ids": [0, 0, 0],
                                  "locations": [["u0.npz", 1], ["u0.npz", 2], ["u1.npz", 2]]},
        "real_lenses_kfold": {"folds": [
            {"fold": 0, "test_lens_rows": [0, 1], "target_lens_rows": [2, 3, 4],
             "target_neg_object_ids": [111, 222],
             "target_neg_locations": [["u0.npz", 3], ["u1.npz", 3]]},
            {"fold": 1, "test_lens_rows": [2, 3], "target_lens_rows": [0, 1, 4],
             "target_neg_object_ids": [333, 444],
             "target_neg_locations": [["u0.npz", 4], ["u1.npz", 4]]},
        ]},
    }


# ---------------------------------------------------------------------------
# expected identity sets, derived straight from the splits (source of truth)
# ---------------------------------------------------------------------------
def expected_for_role(splits, role, fold=None):
    """Return list of (kind_code, shard_id, row) each served item should decode to."""
    exp = []
    sim_shards = splits["sim_shards"]
    # concat map for sim lens global rows -> (shard_filename, local)
    concat = []
    for fn in sim_shards:
        nl = {"shard_a.npz": 5, "shard_b.npz": 3, "shard_c.npz": 4}[fn]
        for r in range(nl):
            concat.append((fn, r))

    if role in ("source_train", "val"):
        node = splits[role]
        for g in node["sim_lens_rows"]:
            fn, r = concat[g]
            exp.append((K_SIMLENS, SIM_SID[fn], r))
        if role == "source_train":
            for fn, idx in node["neg_mlfp"]["locations"]:
                exp.append((K_MLFP, MLFP_SID[fn], idx))
        for fn, idx in node["neg_bare_deflector"]["locations"]:
            exp.append((K_SIMNON, SIM_SID[fn], idx))
        for fn, idx in node["neg_random"]["locations"]:
            exp.append((K_UNLAB, UNL_SID[fn], idx))
    elif role == "target":
        f = splits["real_lenses_kfold"]["folds"][fold]
        for r in f["target_lens_rows"]:
            exp.append((K_LENSCUT, LENSCUT_SID, r))
        for fn, idx in f["target_neg_locations"]:
            exp.append((K_UNLAB, UNL_SID[fn], idx))
    elif role == "test":
        f = splits["real_lenses_kfold"]["folds"][fold]
        for r in f["test_lens_rows"]:
            exp.append((K_LENSCUT, LENSCUT_SID, r))
        for fn, idx in splits["test_negatives_shared"]["locations"]:
            exp.append((K_UNLAB, UNL_SID[fn], idx))
    return exp


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def run():
    torch.manual_seed(0)
    root = tempfile.mkdtemp(prefix="lensds_")
    build_mock(root)
    splits = build_splits()
    with open(os.path.join(root, "splits.json"), "w") as fh:
        json.dump(splits, fh)
    splits_path = os.path.join(root, "splits.json")

    # ---- 1. identity / index-resolution for every role -------------------
    print("\n[1] index resolution (decode every fetched pixel, compare to splits)")
    role_specs = [("source_train", None), ("val", None),
                  ("target", 0), ("target", 1), ("test", 0), ("test", 1)]
    for role, fold in role_specs:
        ds = LensDataset(splits_path, root, role=role, fold=fold, aug="none")
        got = sorted(dec(ds.get_raw(i)[0][0, 0]) for i in range(len(ds)))
        exp = sorted(expected_for_role(splits, role, fold))
        tag = f"{role}" + (f"[fold {fold}]" if fold is not None else "")
        check(got == exp, f"{tag}: {len(ds)} items map to intended array elements")

    # ---- 2. concatenation ORDER (the load-bearing one) -------------------
    print("\n[2] sim concatenation order (sim_shards order != sorted filename order)")
    ds = LensDataset(splits_path, root, role="source_train", aug="none")
    # global row 6 -> under correct order [b(3),a(5),c(4)] -> shard_a local 3
    pos6 = next(i for i, d in enumerate(ds._descriptors) if d["ref"] == ("sim_lens", 6))
    d6 = dec(ds.get_raw(pos6)[0][0, 0])
    check(d6 == (K_SIMLENS, SIM_SID["shard_a.npz"], 3),
          f"global sim_lens row 6 -> shard_a local 3 (got {d6}; sorted-order bug would give "
          f"{(K_SIMLENS, SIM_SID['shard_b.npz'], 1)})")

    # ---- 3. labels / domains / kinds per role ----------------------------
    print("\n[3] label / domain / kind semantics")
    ds = LensDataset(splits_path, root, role="source_train", aug="none")
    ok = all((ds.labels[i] == (1 if ds.kinds[i] == "sim_lens" else 0)) and ds.domains[i] == SOURCE
             for i in range(len(ds)))
    check(ok, "source_train: sim_lens=1, negatives=0, domain=SOURCE")

    ds = LensDataset(splits_path, root, role="val", aug="none")
    check("mlfp" not in ds.kinds, "val: contains no ML-FP negatives")
    check(all(ds.domains[i] == SOURCE for i in range(len(ds))), "val: domain=SOURCE")

    ds = LensDataset(splits_path, root, role="target", fold=0, aug="none")
    check(all(ds.labels[i] == UNLABELED for i in range(len(ds))),
          "target: every label is UNLABELED (-1) -> no real-label leak")
    check(all(ds.domains[i] == TARGET for i in range(len(ds))), "target: domain=TARGET")
    lat_ok = all(ds.latent_labels[i] == (1 if ds.kinds[i] == "real_lens" else 0)
                 for i in range(len(ds)))
    check(lat_ok, "target: latent_labels correct (lens=1/neg=0) for diagnostics only")
    # __getitem__ must not expose the latent label
    check(all(ds[i]["label"] == UNLABELED for i in range(len(ds))),
          "target: __getitem__ never returns a real label")

    ds = LensDataset(splits_path, root, role="test", fold=0, aug="none")
    ok = all((ds.labels[i] == (1 if ds.kinds[i] == "real_lens" else 0)) and ds.domains[i] == TARGET
             for i in range(len(ds)))
    check(ok, "test: real_lens=1, negatives=0, domain=TARGET")

    # ---- 4. shared vs fresh negatives ------------------------------------
    print("\n[4] shared test negatives vs fresh target negatives")
    t0 = LensDataset(splits_path, root, role="test", fold=0, aug="none")
    t1 = LensDataset(splits_path, root, role="test", fold=1, aug="none")
    neg0 = sorted(dec(t0.get_raw(i)[0][0, 0]) for i in range(len(t0)) if t0.kinds[i] == "random")
    neg1 = sorted(dec(t1.get_raw(i)[0][0, 0]) for i in range(len(t1)) if t1.kinds[i] == "random")
    check(neg0 == neg1, "test negatives identical across folds (shared P4 pool)")
    g0 = LensDataset(splits_path, root, role="target", fold=0, aug="none")
    g1 = LensDataset(splits_path, root, role="target", fold=1, aug="none")
    tn0 = set(dec(g0.get_raw(i)[0][0, 0]) for i in range(len(g0)) if g0.kinds[i] == "random")
    tn1 = set(dec(g1.get_raw(i)[0][0, 0]) for i in range(len(g1)) if g1.kinds[i] == "random")
    check(tn0.isdisjoint(tn1), "target negatives disjoint across folds (fresh per fold)")

    # ---- 5. transform / stretch ------------------------------------------
    print("\n[5] transform + stretch")
    z = zoobot_stretch(torch.full((1, 8, 8), 3.14))
    check(torch.allclose(z, torch.zeros_like(z)) and torch.isfinite(z).all(),
          "constant image -> stretch returns zeros (no NaN/inf)")
    ramp = torch.linspace(0, 1, 64 * 64).reshape(64, 64).numpy()
    out = LensTransform("none")(ramp)
    check(out.shape == (1, 224, 224) and out.dtype == torch.float32,
          f"aug=none -> (1,224,224) float32 (got {tuple(out.shape)}, {out.dtype})")
    check(torch.isfinite(out).all() and out.min() >= -1e-3 and out.max() <= 1.0 + 1e-3,
          f"stretched output finite & ~in [0,0.88] (min={out.min():.3f} max={out.max():.3f})")
    ws = LensTransform("weak")(ramp), LensTransform("strong")(ramp)
    check(all(o.shape == (1, 224, 224) for o in ws), "weak & strong views -> (1,224,224)")

    # ---- 6. n_views=2 (AdaMatch) -----------------------------------------
    print("\n[6] n_views=2 emits weak+strong")
    ds = LensDataset(splits_path, root, role="source_train", aug="weak", n_views=2)
    it = ds[0]
    check("image" in it and "image_strong" in it, "item has 'image' and 'image_strong'")
    check(it["image"].shape == (1, 224, 224) and it["image_strong"].shape == (1, 224, 224),
          "both views are (1,224,224)")
    check(it["label"] == ds[0]["label"] and it["domain"] == ds[0]["domain"],
          "views share label/domain")

    # ---- 7. index maps + counts ------------------------------------------
    print("\n[7] convenience index maps & counts")
    ds = LensDataset(splits_path, root, role="source_train", aug="none")
    n_pos = len(splits["source_train"]["sim_lens_rows"])
    n_neg = (len(splits["source_train"]["neg_mlfp"]["locations"])
             + len(splits["source_train"]["neg_bare_deflector"]["locations"])
             + len(splits["source_train"]["neg_random"]["locations"]))
    check(len(ds.indices_by_label.get(1, [])) == n_pos
          and len(ds.indices_by_label.get(0, [])) == n_neg,
          f"indices_by_label groups correctly (pos={n_pos}, neg={n_neg})")
    check(len(ds) == n_pos + n_neg, "length == total references in splits")

    # ---- 8. schema guard + bad-index guard -------------------------------
    print("\n[8] guards")
    bad = dict(splits); bad = json.loads(json.dumps(splits)); bad["schema"] = "old_v5"
    try:
        LensDataset(bad, root, role="val", aug="none"); raised = False
    except ValueError:
        raised = True
    check(raised, "wrong schema raises under strict_schema")
    bad2 = json.loads(json.dumps(splits))
    bad2["source_train"]["sim_lens_rows"] = [999]      # beyond total sim lenses (12)
    try:
        LensDataset(bad2, root, role="source_train", aug="none"); raised = False
    except IndexError:
        raised = True
    check(raised, "out-of-range sim_lens_row raises IndexError")

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 64)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
