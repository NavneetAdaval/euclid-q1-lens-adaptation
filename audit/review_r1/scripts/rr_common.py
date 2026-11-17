"""
rr_common.py -- shared plumbing for the review-round-1 analysis scripts.
=======================================================================

POST HOC AND EXPLORATORY. Nothing here is pre-registered.

Every metric goes through the repo's own code, imported, never re-implemented:
  * ``paper/scripts/_common.py`` -- the arm loaders, the D1/orig pool filter, ``suite``;
  * ``scripts/metrics.py`` -- ``metric_suite`` / ``tpr_at_fpr`` / ``operating_threshold``.

The scripts read the project and write ONLY under ``audit/review_r1/{out,fig}``. Run them
with ``python -B`` so no bytecode cache is written next to the imported repo modules.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RR = os.path.dirname(HERE)                          # audit/review_r1
REPO = os.path.dirname(os.path.dirname(RR))         # the project root
OUT = os.path.join(RR, "out")
FIG = os.path.join(RR, "fig")

sys.dont_write_bytecode = True
for p in (os.path.join(REPO, "paper", "scripts"), os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import _common as C                                  # noqa: E402  paper/scripts/_common.py
import metrics as me                                 # noqa: E402  scripts/metrics.py

STATUS = "post_hoc_exploratory"
TARGET_FPR = C.TARGET_FPR
CONF_FOLDS = tuple(C.CONFIRMATORY_FOLDS)

# Every arm at its headline readout, plus DANN's pre-declared 2000 sensitivity readout.
HEADLINE = [("baseline", 8000), ("lambda0", 4000), ("bnm", 4000), ("freematch", 1500),
            ("dann", 4000), ("dann", 2000), ("coral", 2000)]
DA_ARMS = ("bnm", "freematch", "dann", "coral")
DA_HEADLINE = {"bnm": 4000, "freematch": 1500, "dann": 4000, "coral": 2000}


def tag(arm, step):
    return f"{arm}@{step}"


def R(*p):
    return os.path.join(REPO, *p)


def rel(p):
    return os.path.relpath(os.path.abspath(p), REPO).replace("\\", "/")


# ------------------------------------------------------------------ provenance
_SHA: dict = {}


def sha256(path):
    path = os.path.abspath(path)
    if path not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


class Inputs:
    """Collects every input file a script reads, with its sha256."""

    def __init__(self):
        self.paths = set()

    def add(self, *paths):
        for p in paths:
            if p and os.path.isfile(p):
                self.paths.add(os.path.abspath(p))
        return paths[0] if len(paths) == 1 else paths

    def record(self):
        return {rel(p): sha256(p) for p in sorted(self.paths)}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_json(name, payload, *, script, inputs: Inputs, pool, folds, notes=None):
    os.makedirs(OUT, exist_ok=True)
    body = {
        "status": STATUS,
        "producing_script": rel(script),
        "test_pool": pool,
        "folds": list(folds) if not isinstance(folds, str) else folds,
        "metric": "scripts/metrics.py metric_suite / tpr_at_fpr (TPR at FPR=1e-3, full ROC "
                  "with drop_intermediate=False, np.interp); thresholds from "
                  "metrics.operating_threshold on the same curve",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "notes": notes or [],
        "inputs_sha256": inputs.record(),
        "result": _clean(payload),
    }
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(body, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"  wrote {rel(path)}")
    return path


# ------------------------------------------------------------------ score vectors with ids
def step_paths(arm, fold):
    """{step: path} of an arm's per-checkpoint logits, via the repo's own ARMS loaders.

    ``_scan_dir`` looks ``_npz_pos_neg`` up at call time, so it is pointed for the duration
    of the call at a function returning the path; the loader's directory logic is reused
    unchanged (no path is retyped here).
    """
    if arm == "baseline":
        return {8000: R("runs", "bakeoff_baseline_A", "scores", f"step_8000_fold{fold}.npz")}
    spec = C.ARMS[C.ARM_KEY[arm]]
    saved = C._AWA._npz_pos_neg
    C._AWA._npz_pos_neg = lambda p, f: p
    try:
        return dict(spec["load"](fold))
    finally:
        C._AWA._npz_pos_neg = saved


def fold0_dev_paths():
    """{name: {step: path}} for the fold-0 development runs the paper's §7 uses."""
    import analyze_window_fold0 as AWF
    saved = C._AWA._npz_pos_neg
    C._AWA._npz_pos_neg = lambda p, f: p
    try:
        return {n: dict(C._AWA._scan_dir(d, 0)) for n, d in AWF.ARMS_F0.items()}
    finally:
        C._AWA._npz_pos_neg = saved


def ids_npz_path(path, arm):
    if arm == "baseline":
        return path
    return C._ids_npz_for(path)


def load_vec(path, fold, arm, pool, inputs: Inputs | None = None):
    """Scores + identities of one stored vector, restricted to ``pool``.

    Returns dict(pos, neg, pos_rows, neg_locs). The restriction is computed by
    ``_common.pool_keep_neg`` and the result is asserted identical to
    ``_common.pool_filter`` on the same vector.
    """
    if arm == "baseline":
        z = np.load(path, allow_pickle=False)
        pos, neg = np.asarray(z["pos_scores"], float), np.asarray(z["neg_scores"], float)
        pos_ids, neg_ids = [str(x) for x in z["pos_ids"]], [str(x) for x in z["neg_ids"]]
    else:
        pos, neg = C._RAW_NPZ(path, fold)
        pi, ni = C._stored_ids(path, fold)
        pos_ids, neg_ids = [str(x) for x in pi], [str(x) for x in ni]
    if inputs is not None:
        inputs.add(path, ids_npz_path(path, arm))
    assert len(pos_ids) == pos.size and len(neg_ids) == neg.size, path
    fp, fn = C.pool_filter(pos, neg, pos_ids, neg_ids, where=f"rr:{path}", pool_name=pool)
    if pool == "orig":
        keep = np.ones(neg.size, bool)
    else:
        keep = C.pool_keep_neg(neg_ids)
    assert np.array_equal(fn, neg[keep]) and np.array_equal(fp, pos), path
    pos_rows = [int(s.rsplit(":", 1)[1]) for s in pos_ids]
    return dict(pos=pos, neg=neg[keep], pos_rows=pos_rows,
                neg_locs=[l for l, k in zip(neg_ids, keep) if k])


def suite_of(v):
    y = np.concatenate([np.ones(v["pos"].size, int), np.zeros(v["neg"].size, int)])
    return dict(me.metric_suite(y, np.concatenate([v["pos"], v["neg"]]),
                                target_fpr=TARGET_FPR))


def threshold_of(v):
    y = np.concatenate([np.ones(v["pos"].size, int), np.zeros(v["neg"].size, int)])
    return float(me.operating_threshold(y, np.concatenate([v["pos"], v["neg"]]),
                                        target_fpr=TARGET_FPR))


def tpr_of(v):
    y = np.concatenate([np.ones(v["pos"].size, int), np.zeros(v["neg"].size, int)])
    return float(me.tpr_at_fpr(y, np.concatenate([v["pos"], v["neg"]]),
                               target_fpr=TARGET_FPR))


# ------------------------------------------------------------------ Table 2 reproduction
def table2_values():
    """{arm@step: (fold-mean, {fold: tpr})} parsed from paper/tables/T5_perfold.tex.

    T5 is the per-fold table behind Table 2 (tab:main, generated as T1_main.tex); it is
    read only to CHECK the recomputation, never as a source.
    """
    import re
    txt = open(R("paper", "tables", "T5_perfold.tex"), encoding="utf-8").read()
    first = txt.split(r"\label{tab:perfold-tpr}")[1].split(r"\end{tabular}")[0]
    name = {"Baseline A": "baseline", r"$\lambda=0$": "lambda0", "BNM": "bnm",
            "FreeMatch": "freematch", "DANN": "dann", "CORAL": "coral"}
    out = {}
    for ln in first.splitlines():
        m = re.match(r"^(.*?) & (\d+) & (.*?) \\\\", ln.strip())
        if not m:
            continue
        lab = m.group(1).split(r"\,")[0].strip()
        if lab not in name:
            continue
        cells = [c.strip().strip("$") for c in m.group(3).split("&")]
        vals = {f: float(cells[f]) for f in (1, 2, 3, 4)}
        out[tag(name[lab], int(m.group(2)))] = (float(cells[5]), vals)
    return out


def t1_main_values():
    """{arm: fold-mean string} parsed from paper/tables/T1_main.tex (Table 2 in the PDF)."""
    import re
    txt = open(R("paper", "tables", "T1_main.tex"), encoding="utf-8").read()
    body = txt.split(r"\midrule")[1].split(r"\bottomrule")[0]
    out = {}
    for ln in body.strip().splitlines():
        cells = [c.strip() for c in ln.rstrip("\\ ").split("&")]
        out[cells[0]] = (float(cells[1].strip("$")), int(cells[4]))
    return out


def reproduce_headlines(inputs: Inputs):
    """Recompute each arm's headline (and DANN@2000) fold-mean on D1 from the loaders here
    and check it equals (a) ``_common.suite`` exactly and (b) the printed table to 4 dp."""
    t5 = table2_values()
    t1 = t1_main_values()
    inputs.add(R("paper", "tables", "T5_perfold.tex"), R("paper", "tables", "T1_main.tex"))
    rows = {}
    with C.pool("D1"):
        for arm, step in HEADLINE:
            per = {}
            for f in CONF_FOLDS:
                v = load_vec(step_paths(arm, f)[step], f, arm, "D1", inputs)
                mine = suite_of(v)["tpr"]
                theirs = float(C.suite(arm, f, step)["tpr"])
                assert mine == theirs, (arm, step, f, mine, theirs)
                per[f] = mine
            mean = float(np.mean([per[f] for f in CONF_FOLDS]))
            printed_mean, printed = t5[tag(arm, step)]
            ok = (f"{mean:.4f}" == f"{printed_mean:.4f}" and
                  all(f"{per[f]:.4f}" == f"{printed[f]:.4f}" for f in CONF_FOLDS))
            rows[tag(arm, step)] = dict(per_fold=per, fold_mean=mean,
                                        printed_T5_fold_mean=printed_mean,
                                        printed_T5_per_fold=printed,
                                        exact_equal_to__common_suite=True,
                                        matches_printed_to_4dp=ok)
    t1_ok = {}
    lab = {"baseline": "Baseline A (frozen)", "lambda0": r"$\lambda=0$ unfrozen control",
           "bnm": r"BNM $\times$ nb3", "freematch": r"FreeMatch $\times$ nb3",
           "dann": r"DANN $\times$ nb3", "coral": r"CORAL $\times$ nb3"}
    for arm, step in HEADLINE:
        if (arm, step) == ("dann", 2000):
            continue
        pv, ro = t1[lab[arm]]
        t1_ok[tag(arm, step)] = dict(printed_T1_main=pv, read_at=ro,
                                     matches=(ro == step and
                                              f"{rows[tag(arm, step)]['fold_mean']:.4f}" ==
                                              f"{pv:.4f}"))
    all_ok = all(r["matches_printed_to_4dp"] for r in rows.values()) and \
        all(r["matches"] for r in t1_ok.values())
    return dict(rows=rows, table2_T1_main=t1_ok, all_match=all_ok)


# ------------------------------------------------------------------ lens / negative metadata
def lens_meta(inputs: Inputs):
    p = R("data", "q1_lenses_processed", "lens_cutouts.npz")
    inputs.add(p)
    with np.load(p, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in ("id_str", "grade", "expert_score", "ra", "dec",
                                              "object_id")}


def mer_table(inputs: Inputs):
    """object_id -> (ra, dec, flux_vis_psf) from the stored MER query output."""
    import pandas as pd
    p = C.ORIG["mer_flux"]
    inputs.add(p)
    t = pd.read_csv(p, usecols=["object_id", "right_ascension", "declination",
                                "flux_vis_psf"])
    assert not t["object_id"].duplicated().any()
    return t.set_index("object_id")


def ab_mag(flux_ujy):
    """AB magnitude of MER flux_vis_psf in microjansky, exactly as §3 / make_m_prose."""
    f = np.asarray(flux_ujy, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 23.9 - 2.5 * np.log10(f)


def neg_image(loc, inputs: Inputs | None = None, _cache={}):
    """One test-negative cutout from its stored location 'random:<shard>:<row>'."""
    _, shard, row = loc.split(":")
    p = R("data", "q1_unlabeled", "shards", shard)
    if p not in _cache:
        if inputs is not None:
            inputs.add(p)
        with np.load(p) as z:
            _cache[p] = (np.asarray(z["images"]), np.asarray(z["object_ids"]))
    return _cache[p][0][int(row)], int(_cache[p][1][int(row)])


def lens_image(row, inputs: Inputs | None = None, _cache={}):
    p = R("data", "q1_lenses_processed", "lens_cutouts.npz")
    if "imgs" not in _cache:
        if inputs is not None:
            inputs.add(p)
        with np.load(p, allow_pickle=True) as z:
            _cache["imgs"] = np.asarray(z["images"])
    return _cache["imgs"][int(row)]


# ------------------------------------------------------------------ two-stretch galleries
def stretches(img):
    """(encoder stretch, gentle stretch). The first is lens_dataset.zoobot_stretch, the
    per-image arcsinh the model sees (as Figure 2); the second a linear clip at the 1st
    and 99.5th percentiles."""
    import torch
    from lens_dataset import zoobot_stretch
    a = zoobot_stretch(torch.from_numpy(np.asarray(img, np.float32))[None, ...])[0].numpy()
    lo, hi = np.percentile(img, [1.0, 99.5])
    b = np.clip((np.asarray(img, float) - lo) / max(hi - lo, 1e-12), 0, 1)
    return a, b


def gallery(items, path, title):
    """items: list of (img, label). A 5x5 page of objects, each drawn twice side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 5
    fig, axes = plt.subplots(n, 2 * n, figsize=(15, 8.6))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    for i, (img, label) in enumerate(items[:25]):
        r, c = divmod(i, n)
        a, b = stretches(img)
        for j, (im, s) in enumerate(((a, "enc"), (b, "lin 1-99.5"))):
            ax = axes[r, 2 * c + j]
            ax.axis("on")
            ax.imshow(im, cmap="gray", origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_title(label, fontsize=6.0, loc="left", pad=1.5)
            ax.text(0.02, 0.02, s, transform=ax.transAxes, fontsize=4.8, color="yellow",
                    va="bottom")
    fig.suptitle(title, fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.04,
                        hspace=0.32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {rel(path)}")
