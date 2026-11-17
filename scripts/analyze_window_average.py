"""
analyze_window_average.py
=========================
ROUND 7, TASK 3 -- CHECKPOINT-WINDOW AVERAGING. **DECLARED SECONDARY ANALYSIS.**

Replaces no hashed headline. Every pre-registered single-point number in 7.12, 7.15,
7.18, 7.22 and 7.28 remains the arm's reported result.

    python -u scripts/analyze_window_average.py --out runs/round7/window_average.json

Zero GPU. Every arm's per-checkpoint logits are already on disk; Baseline A's per-step
score vectors are RECOMPUTED on CPU from the frozen 640-d feature cache and the retained
per-step head weights, because round 1 kept only ``step_8000``'s dumps.

DELIBERATELY REUSED, NOT REIMPLEMENTED:
  * ``metrics.metric_suite`` -- the same metric path every arm's own analysis used;
  * ``analyze_lambda0_r6.paired_t`` -- the df = n-1 paired t-interval. Never a bootstrap.

The window size, the k grid, the two score combinations and the window-placement rule
are all fixed by ``runs/round7/window_prereg.json``, written and hashed BEFORE any
windowed number existed. This script READS that file and asserts its hash; it contains
no k of its own.

Writes NOTHING outside --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics as st
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
REPO = os.path.dirname(_HERE)

import metrics as me                                  # noqa: E402
from analyze_lambda0_r6 import paired_t               # noqa: E402
from integrity import verify as _verify_integrity     # noqa: E402

PREREG = os.path.join(REPO, "runs", "round7", "window_prereg.json")
TARGET_FPR = 1e-3
FOLDS = (1, 2, 3, 4)
BASELINE_FOLDS = (0, 1, 2, 3, 4)


def L(*p):
    return os.path.join(REPO, *p)


# ---------------------------------------------------------------------------
# per-arm score loaders -- each returns {step: (pos_scores, neg_scores)}
# ---------------------------------------------------------------------------

def _npz_pos_neg(path, fold):
    z = np.load(path)
    if "pos" in z.files:
        return np.asarray(z["pos"], float), np.asarray(z["neg"], float)
    return (np.asarray(z[f"test_pos_fold{fold}"], float),
            np.asarray(z["test_negatives_shared"], float))


def _scan_dir(d, fold):
    """Every ``step_N.npz`` in ``d``, step 0 excluded (it is the initialisation)."""
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if not (fn.startswith("step_") and fn.endswith(".npz")):
            continue
        step = int(fn[5:-4])
        if step == 0:
            continue
        out[step] = _npz_pos_neg(os.path.join(d, fn), fold)
    return out


def load_coral(fold):
    return _scan_dir(L("kaggle", "coral_r4", "runs", "main", "extracted",
                       "coral_r4_confirm", f"fold{fold}", "logits",
                       f"g16_2000_fold{fold}"), fold)


def load_freematch(fold):
    return _scan_dir(L("runs", f"freematch_nb3_unfrozen_fold{fold}", "logits"), fold)


def load_dann(fold):
    return _scan_dir(L("runs", "dann_main", f"fold{fold}", "extracted", f"fold{fold}",
                       "logits", f"dann_nb3_fold{fold}"), fold)


def load_lambda0(fold):
    return _scan_dir(L("runs", "baseline_unfrozen", f"fold{fold}", "extracted",
                       f"lambda0_r6_fold{fold}", "logits",
                       f"dann_nb3_lambda0_fold{fold}"), fold)


def load_bnm(fold):
    return _scan_dir(L("runs", "bnm_r6_folds", f"bnm_r6_fold{fold}", "logits",
                       f"bnm_nb3_fold{fold}"), fold)


# --- Baseline A: frozen, so the score vectors are a CPU head forward on cached feats ---

_BASELINE_CACHE = {}


def _baseline_scores(fold):
    """{step: (pos, neg)} for the frozen Baseline A, recomputed from the feature cache.

    Frozen encoder => the 640-d features never change, so a checkpoint's scores are
    ``head(standardizer(features))``. No GPU, no encoder, no training.
    """
    if _BASELINE_CACHE:
        return _BASELINE_CACHE[fold]

    import torch
    from heads import build_head, FeatureStandardizer

    fc = L("data", "feature_cache")
    neg = torch.from_numpy(
        np.load(os.path.join(fc, "test_negatives_shared.npz"))["features"]).float()
    pos = {f: torch.from_numpy(
        np.load(os.path.join(fc, f"test_pos_fold{f}.npz"))["features"]).float()
        for f in BASELINE_FOLDS}

    ckdir = L("runs", "bakeoff_baseline_A", "checkpoints")
    steps = sorted(int(fn[5:-3]) for fn in os.listdir(ckdir)
                   if fn.startswith("step_") and fn.endswith(".pt")
                   and int(fn[5:-3]) != 0)
    for f in BASELINE_FOLDS:
        _BASELINE_CACHE[f] = {}
    for s in steps:
        ck = torch.load(os.path.join(ckdir, f"step_{s}.pt"), map_location="cpu",
                        weights_only=False)
        head = build_head(ck["head_type"], in_dim=neg.shape[1])
        head.load_state_dict(ck["head"])
        head.eval()
        std = FeatureStandardizer(dim=neg.shape[1])
        std.load_state_dict(ck["standardizer"])
        std.eval()
        with torch.no_grad():
            neg_s = head(std(neg)).squeeze(-1).numpy().astype(float)
            for f in BASELINE_FOLDS:
                pos_s = head(std(pos[f])).squeeze(-1).numpy().astype(float)
                _BASELINE_CACHE[f][s] = (pos_s, neg_s)
    return _BASELINE_CACHE[fold]


ARMS = {
    "CORAL x nb3":         {"load": load_coral,     "folds": FOLDS,          "readouts": [2000]},
    "FreeMatch x nb3":     {"load": load_freematch, "folds": FOLDS,          "readouts": [1500]},
    "DANN x nb3":          {"load": load_dann,      "folds": FOLDS,          "readouts": [4000, 2000]},
    "lambda=0":            {"load": load_lambda0,   "folds": FOLDS,          "readouts": [4000, 2000, 1500]},
    "BNM x nb3":           {"load": load_bnm,       "folds": FOLDS,          "readouts": [1000, 1500, 2000, 4000]},
    "Baseline A (frozen)": {"load": _baseline_scores, "folds": BASELINE_FOLDS, "readouts": [8000]},
}


# ---------------------------------------------------------------------------
# the two DECLARED score combinations
# ---------------------------------------------------------------------------

def _avg_ranks(v):
    """Average ranks (ties shared), normalised to [0, 1]. No scipy dependency."""
    n = v.size
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks / (n - 1)


def combine(scores_by_step, steps, how):
    """Combine the window's per-checkpoint score vectors into one score vector."""
    vecs = []
    for s in steps:
        pos, neg = scores_by_step[s]
        v = np.concatenate([pos, neg])
        if how == "mean_rank":
            v = _avg_ranks(v)
        vecs.append(v)
    n_pos = scores_by_step[steps[0]][0].size
    return np.mean(np.vstack(vecs), axis=0), n_pos


def window(available_steps, readout, k):
    """The k consecutive AVAILABLE checkpoints ending AT ``readout``, inclusive."""
    ge = [s for s in available_steps if s <= readout]
    if not ge or readout not in available_steps:
        return []
    return ge[-k:]


def evaluate(scores_by_step, steps, how):
    s, n_pos = combine(scores_by_step, steps, how)
    y = np.concatenate([np.ones(n_pos, int), np.zeros(s.size - n_pos, int)])
    return me.metric_suite(y, s, target_fpr=TARGET_FPR)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    _verify_integrity(where="before the round-7 window-averaging analysis")

    pre = json.load(open(PREREG, encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k != "sha256_of_body_without_this_field"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != pre["sha256_of_body_without_this_field"]:
        raise SystemExit("the window declaration does not hash to itself -- STOP")

    K_GRID = list(pre["k"]["grid"])
    K_PRIMARY = int(pre["k"]["primary"])
    COMBOS = [pre["score_combination"]["primary"]["name"],
              pre["score_combination"]["secondary"]["name"]]

    print("=" * 100)
    print("ROUND 7, TASK 3 -- CHECKPOINT-WINDOW AVERAGING")
    print("*** DECLARED SECONDARY ANALYSIS -- replaces no hashed headline ***")
    print("=" * 100)
    print(f"  declaration {os.path.relpath(PREREG, REPO)}")
    print(f"  sha256      {digest}")
    print(f"  k grid {K_GRID}, primary k = {K_PRIMARY}; combinations {COMBOS}")
    print("  the window ends AT each arm's pre-registered read-off step, so the rule")
    print("  reads no metric. k was fixed before any number below existed.")

    out = {"round": 7, "task": "checkpoint_window_averaging",
           "status": "DECLARED SECONDARY ANALYSIS -- replaces no hashed headline",
           "prereg_sha256": digest, "k_grid": K_GRID, "k_primary": K_PRIMARY,
           "combinations": COMBOS, "arms": {}}

    loaded = {}
    for name, spec in ARMS.items():
        loaded[name] = {f: spec["load"](f) for f in spec["folds"]}
        grids = {f: sorted(loaded[name][f]) for f in spec["folds"]}
        n = {f: len(grids[f]) for f in spec["folds"]}
        print(f"\n  [{name}] checkpoints per fold: " +
              ", ".join(f"f{f}={n[f]}" for f in spec["folds"]))

    # ---------------- 1. windowed readouts ----------------
    print("\n" + "=" * 100)
    print("1. WINDOWED TPR@1e-3 -- fold-mean at each mandatory readout")
    print("=" * 100)

    for name, spec in ARMS.items():
        folds = spec["folds"]
        out["arms"][name] = {"folds": list(folds), "readouts": {}}
        for ro in spec["readouts"]:
            print(f"\n  {name}  --  read-off step {ro}")
            rec = {}
            for how in COMBOS:
                print(f"    {how}")
                print(f"      {'k':>3} {'ach':>4}  " +
                      "".join(f"{'f' + str(f):>9}" for f in folds) +
                      f"{'mean':>10}{'sd':>9}{'AUROC':>9}")
                for k in K_GRID:
                    vals, aur, comps, ach = [], [], [], []
                    for f in folds:
                        steps = sorted(loaded[name][f])
                        w = window(steps, ro, k)
                        if not w:
                            vals.append(float("nan")); aur.append(float("nan"))
                            comps.append(float("nan")); ach.append(0)
                            continue
                        su = evaluate(loaded[name][f], w, how)
                        vals.append(float(su["tpr"])); aur.append(float(su["auroc"]))
                        comps.append(float(su["comp_at_purity"])); ach.append(len(w))
                    ok = [v for v in vals if np.isfinite(v)]
                    m = st.mean(ok) if len(ok) == len(folds) else float("nan")
                    sd = st.stdev(ok) if len(ok) == len(folds) else float("nan")
                    achs = sorted(set(ach))
                    print(f"      {k:>3} {(str(achs[0]) if len(achs) == 1 else str(achs)):>4}  " +
                          "".join(f"{v:>9.4f}" for v in vals) +
                          f"{m:>10.4f}{sd:>9.4f}{np.nanmean(aur):>9.4f}")
                    rec.setdefault(how, {})[str(k)] = {
                        "achieved_k": ach, "tpr_per_fold": vals, "tpr_mean": m,
                        "tpr_sd": sd, "auroc_per_fold": aur,
                        "comp_at_purity_per_fold": comps,
                        "comp_at_purity_mean": (st.mean(comps)
                                                if all(np.isfinite(c) for c in comps)
                                                else None)}
            out["arms"][name]["readouts"][str(ro)] = rec

    # ---------------- 2. scatter reduction vs sqrt(k) ----------------
    print("\n" + "=" * 100)
    print("2. SCATTER REDUCTION -- sliding window over every valid position, vs sqrt(k)")
    print("=" * 100)
    print("  sd is taken ACROSS WINDOW POSITIONS within one fold, on that arm's own grid.")
    print("  Both sds use the SAME positions, so they are comparable. Expectation:")
    print("  sd_k = sd_1 / sqrt(k); the ratio sd_1/sd_k is compared against sqrt(k).")

    scat = {}
    for name, spec in ARMS.items():
        folds = spec["folds"]
        print(f"\n  {name}")
        print(f"      {'k':>3}  {'sd(k) fold-mean':>16}{'sd(1) same pos':>16}"
              f"{'ratio':>9}{'sqrt(k)':>9}{'achieved/expected':>19}")
        scat[name] = {}
        for k in K_GRID:
            sd_k, sd_1 = [], []
            for f in folds:
                steps = sorted(loaded[name][f])
                if len(steps) < k:
                    continue
                pos_list = [steps[i - k + 1:i + 1] for i in range(k - 1, len(steps))]
                vk = [evaluate(loaded[name][f], w, "mean_logit")["tpr"] for w in pos_list]
                v1 = [evaluate(loaded[name][f], [w[-1]], "mean_logit")["tpr"]
                      for w in pos_list]
                vk = [x for x in vk if np.isfinite(x)]
                v1 = [x for x in v1 if np.isfinite(x)]
                if len(vk) > 1:
                    sd_k.append(st.stdev(vk))
                if len(v1) > 1:
                    sd_1.append(st.stdev(v1))
            if not sd_k:
                continue
            mk, m1 = st.mean(sd_k), st.mean(sd_1)
            ratio = m1 / mk if mk > 0 else float("nan")
            exp = math.sqrt(k)
            print(f"      {k:>3}  {mk:>16.4f}{m1:>16.4f}{ratio:>9.2f}{exp:>9.2f}"
                  f"{ratio / exp:>19.2f}")
            scat[name][str(k)] = {"sd_k_foldmean": mk, "sd_1_same_positions": m1,
                                  "ratio": ratio, "sqrt_k": exp,
                                  "achieved_over_expected": ratio / exp,
                                  "sd_k_per_fold": sd_k, "sd_1_per_fold": sd_1}
    out["scatter_reduction"] = scat

    # ---------------- 3. the 7.16 orderings ----------------
    print("\n" + "=" * 100)
    print("3. THE 7.16 ORDERINGS -- do they survive window averaging?")
    print("=" * 100)

    ver = {}

    def tpr_at(name, ro, k, how="mean_logit"):
        r = out["arms"][name]["readouts"][str(ro)][how][str(k)]
        return r["tpr_per_fold"], r["tpr_mean"]

    # Row 5 -- the read-off inversion. DANN at 2000 vs 4000 against CORAL and FreeMatch.
    print("\n  ROW 5 -- 'nb3 method ranking under read-off choice' (7.18)")
    print("    the claim that died: DANN reads 0.4551 at step 2000 and 0.4167 at 4000,")
    print("    inverting first-of-three and last-of-three against CORAL 0.4348 and")
    print("    FreeMatch 0.4169. Does the inversion survive window averaging?")
    row5 = {}
    for k in (1, K_PRIMARY):
        d2 = tpr_at("DANN x nb3", 2000, k)[1]
        d4 = tpr_at("DANN x nb3", 4000, k)[1]
        c = tpr_at("CORAL x nb3", 2000, k)[1]
        fm = tpr_at("FreeMatch x nb3", 1500, k)[1]
        rank2 = sorted([("DANN@2000", d2), ("CORAL", c), ("FreeMatch", fm)],
                       key=lambda t: -t[1])
        rank4 = sorted([("DANN@4000", d4), ("CORAL", c), ("FreeMatch", fm)],
                       key=lambda t: -t[1])
        inverted = rank2[0][0].startswith("DANN") != rank4[0][0].startswith("DANN")
        print(f"\n    k = {k}:  DANN@2000 {d2:.4f}   DANN@4000 {d4:.4f}   "
              f"CORAL {c:.4f}   FreeMatch {fm:.4f}")
        print(f"      order at DANN's 2000 read-off: " +
              " > ".join(f"{n} {v:.4f}" for n, v in rank2))
        print(f"      order at DANN's 4000 read-off: " +
              " > ".join(f"{n} {v:.4f}" for n, v in rank4))
        print(f"      DANN's position flips between the two read-offs: "
              f"{'YES' if inverted else 'NO'}")
        row5[str(k)] = {"dann_2000": d2, "dann_4000": d4, "coral": c, "freematch": fm,
                        "order_at_2000": [n for n, _ in rank2],
                        "order_at_4000": [n for n, _ in rank4],
                        "dann_position_flips": bool(inverted),
                        "spread_of_three_at_2000": max(v for _, v in rank2) -
                                                   min(v for _, v in rank2),
                        "spread_of_three_at_4000": max(v for _, v in rank4) -
                                                   min(v for _, v in rank4)}
    ver["row5_readoff_inversion"] = row5

    # Row 6 -- DANN's spread exceeds the other arms'.
    print("\n  ROW 6 -- \"DANN's spread exceeds the other arms'\" (7.18)")
    print("    the claim that died: holds at declared budgets, REVERSES budget-matched")
    print("    (0.63x at 1500). Window averaging changes the per-fold values it is")
    print("    computed from, so the across-fold sd is recomputed here at k=1 and k=8.")
    row6 = {}
    for k in (1, K_PRIMARY):
        sds = {"DANN@4000": tpr_at("DANN x nb3", 4000, k),
               "CORAL@2000": tpr_at("CORAL x nb3", 2000, k),
               "FreeMatch@1500": tpr_at("FreeMatch x nb3", 1500, k),
               "lambda=0@4000": tpr_at("lambda=0", 4000, k),
               "BNM@4000": tpr_at("BNM x nb3", 4000, k)}
        line = {n: st.stdev(v[0]) for n, v in sds.items()}
        rank = sorted(line.items(), key=lambda t: -t[1])
        print(f"\n    k = {k}: across-fold sd  " +
              "   ".join(f"{n} {s:.4f}" for n, s in rank))
        print(f"      DANN is the widest: "
              f"{'YES' if rank[0][0].startswith('DANN') else 'NO -- ' + rank[0][0] + ' is'}")
        row6[str(k)] = {"across_fold_sd": line,
                        "widest": rank[0][0],
                        "dann_is_widest": bool(rank[0][0].startswith("DANN"))}
    ver["row6_dann_spread"] = row6

    # Row 8 -- the resolution argument itself.
    print("\n  EIGHTH ENTRY -- 'within-run checkpoint sd is 2.8-5.1x the between-method")
    print("    differences'. The scatter table above is the direct test: if window")
    print("    averaging pulls the within-run sd below the between-method differences,")
    print("    the eighth entry is a read-off artifact; if it does not, it is the data.")
    diffs_to_beat = {"DANN - CORAL": 0.018, "DANN - FreeMatch": 0.0002}
    row8 = {"between_method_differences_7_18": diffs_to_beat, "by_k": {}}
    for k in K_GRID:
        sds = {n: scat[n][str(k)]["sd_k_foldmean"] for n in scat if str(k) in scat[n]}
        nb3 = {n: v for n, v in sds.items() if "nb3" in n or n == "lambda=0"}
        if not nb3:
            continue
        lo, hi = min(nb3.values()), max(nb3.values())
        print(f"    k = {k}: nb3-arm within-run sd {lo:.4f}-{hi:.4f}  -> "
              f"{lo / 0.018:.1f}x-{hi / 0.018:.1f}x the DANN-CORAL difference (0.018)")
        row8["by_k"][str(k)] = {"within_run_sd_lo": lo, "within_run_sd_hi": hi,
                                "multiple_of_dann_minus_coral_lo": lo / 0.018,
                                "multiple_of_dann_minus_coral_hi": hi / 0.018,
                                "per_arm": nb3}
    ver["row8_resolution_argument"] = row8

    # Row 9 -- lambda=0 is the least stable arm, grid-matched.
    print("\n  ROW 9 -- 'unfreezing without a DA term is the least stable arm' (7.22)")
    print("    the claim that died: holds on 250-8000 (0.276 vs 0.202), DISSOLVES")
    print("    grid-matched on 250-4000 (0.258 vs 0.256). Recomputed here as the")
    print("    within-fold range across window positions on the matched 250-4000 grid.")
    row9 = {}
    for k in (1, K_PRIMARY):
        rng = {}
        for n in ("lambda=0", "DANN x nb3", "BNM x nb3"):
            vals = []
            for f in FOLDS:
                steps = [s for s in sorted(loaded[n][f]) if s <= 4000]
                if len(steps) < k:
                    continue
                pos_list = [steps[i - k + 1:i + 1] for i in range(k - 1, len(steps))]
                v = [evaluate(loaded[n][f], w, "mean_logit")["tpr"] for w in pos_list]
                v = [x for x in v if np.isfinite(x)]
                if v:
                    vals.append(max(v) - min(v))
            rng[n] = st.mean(vals) if vals else float("nan")
        rank = sorted(rng.items(), key=lambda t: -t[1])
        print(f"\n    k = {k}: mean within-fold range (250-4000 matched)  " +
              "   ".join(f"{n} {v:.4f}" for n, v in rank))
        print(f"      lambda=0 is the least stable: "
              f"{'YES' if rank[0][0] == 'lambda=0' else 'NO -- ' + rank[0][0] + ' is'}")
        row9[str(k)] = {"mean_within_fold_range": rng, "widest": rank[0][0],
                        "lambda0_is_widest": rank[0][0] == "lambda=0"}
    ver["row9_lambda0_stability"] = row9

    out["ordering_verdicts"] = ver

    # ---------------- 4. cross-arm contrasts at k = primary ----------------
    print("\n" + "=" * 100)
    print(f"4. CROSS-ARM PAIRED CONTRASTS at k = {K_PRIMARY} -- paired t on 4 differences, df = 3")
    print("=" * 100)
    print("  Budget-matched exactly as 7.28: each contrast reads both arms at the same")
    print("  step. The 7.18 disclosure travels unchanged -- a budget match removes the")
    print("  BUDGET confound only; head initialisation and tuning budget still travel.")
    pairs = [
        ("BNM(4000) - lambda=0(4000)", ("BNM x nb3", 4000), ("lambda=0", 4000)),
        ("DANN(4000) - lambda=0(4000)", ("DANN x nb3", 4000), ("lambda=0", 4000)),
        ("BNM(2000) - CORAL(2000)", ("BNM x nb3", 2000), ("CORAL x nb3", 2000)),
        ("BNM(1500) - FreeMatch(1500)", ("BNM x nb3", 1500), ("FreeMatch x nb3", 1500)),
        ("BNM(4000) - DANN(4000)", ("BNM x nb3", 4000), ("DANN x nb3", 4000)),
        ("DANN(2000) - CORAL(2000)", ("DANN x nb3", 2000), ("CORAL x nb3", 2000)),
    ]
    con = {}
    print(f"\n  {'contrast':<32}{'k=1 mean':>10}{'k=%d mean' % K_PRIMARY:>11}"
          f"{'k=%d 95%% CI' % K_PRIMARY:>24}  excl0  same-sign")
    for label, (na, sa), (nb, sb) in pairs:
        row = {}
        for k in (1, K_PRIMARY):
            va = tpr_at(na, sa, k)[0]
            vb = tpr_at(nb, sb, k)[0]
            row[str(k)] = paired_t([x - y for x, y in zip(va, vb)])
        t = row[str(K_PRIMARY)]
        print(f"  {label:<32}{row['1']['mean']:>+10.4f}{t['mean']:>+11.4f}"
              f"{'[' + f'{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}' + ']':>24}"
              f"{'  yes ' if t['excludes_zero'] else '  no  '}"
              f"{'   yes' if t['all_same_sign'] else '   no'}")
        con[label] = row
    print(f"\n  MULTIPLICITY: {len(pairs)} intervals, NO correction applied. Disclosed.")
    out["contrasts_at_k_primary"] = con

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n[out] {os.path.relpath(os.path.abspath(a.out), REPO)}")

    _verify_integrity(where="after the round-7 window-averaging analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
