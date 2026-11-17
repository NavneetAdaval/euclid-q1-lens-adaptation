"""
report_screen_unfreeze.py
=========================
LEVER 1 fold-0 SCREEN report. Reads only artifacts already on disk (scores/*.npz +
curve.csv + results.json) and produces the screen's deliverables:

  3. CONTROL      baselineA_fold0_nb0_image vs baselineA_fold0_nb0_cached, explicit
                  pass/fail with the paired-bootstrap CI.
  4. HEADLINE     fold-0 TPR@FPR=1e-3 for nb0_image / nb1 / nb3 with paired-bootstrap CIs
                  computed AGAINST nb0_image (the image-space control, NOT the cached run --
                  comparing an unfrozen image-space run to the cached run would confound the
                  unfreeze with the code path + augmentation).
  5. FLATNESS     per rung: the 6000-8000 drift vs the bootstrap CI half-width (is R1 still
                  defensible?), where the curve actually flattens, and the collapse gate.
  6. DECISION     applies the PRE-REGISTERED rule (recorded in each run_config.json before
                  any result existed) and states the outcome.

It computes nothing that the runs did not already measure, and it never re-selects a
checkpoint: R1 (step 8000) is structural.

Run:
    python scripts/report_screen_unfreeze.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import (tpr_at_fpr, paired_bootstrap, default_tpr_metric,
                     DEFAULT_TARGET_FPR)                      # noqa: E402

_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
BUDGET = 8000
WINDOW = (6000, 8000)

RUNS = [
    ("nb0_cached", "bakeoff_baseline_A", f"scores/step_{BUDGET}_fold0.npz", 0),
    ("nb0_image", "baselineA_fold0_nb0_image",
     "scores/baselineA_fold0_nb0_image_fold0.npz", 0),
    # the isolating code-path control: aug=none + fp32, i.e. matched to the cached run in
    # BOTH respects the weak-aug/fp16 run differs in.
    ("nb0_image_noaug", "baselineA_fold0_nb0_image_noaug_fp32",
     "scores/baselineA_fold0_nb0_image_fold0.npz", 0),
    # SAME-HARDWARE baseline: nb1/nb3 run on Kaggle T4 (Turing) while the local nb0_image
    # was measured on an Ada 4050. Precision alone was measured at -0.0714 TPR, so a
    # cross-architecture reference is not good enough for the PRIMARY comparison. When this
    # replicate is present it becomes the reference; the local run is still reported.
    ("nb0_image_kaggle", "baselineA_fold0_nb0_image_kaggle",
     "scores/baselineA_fold0_nb0_image_fold0.npz", 0),
    ("nb1", "baselineA_fold0_nb1", "scores/baselineA_fold0_nb1_fold0.npz", 1),
    ("nb3", "baselineA_fold0_nb3", "scores/baselineA_fold0_nb3_fold0.npz", 3),
]

ONE_LENS = 1.0 / 98.0      # fold-0 TPR quantum; venue disagreement is judged against this


def load_scores(path):
    with np.load(path, allow_pickle=False) as z:
        return (z["pos_scores"], z["neg_scores"],
                z["pos_ids"] if "pos_ids" in z.files else None,
                z["neg_ids"] if "neg_ids" in z.files else None)


def read_curve(run_dir):
    p = os.path.join(run_dir, "curve.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _col(rows, name):
    out = []
    for r in rows:
        v = r.get(name, "")
        if v not in ("", None):
            out.append((int(r["step"]), float(v)))
    return out


def flatten_step(pts, tol):
    """First step after which the metric stays within ``tol`` of its final value.

    TOLERANCE CHOICE MATTERS. TPR@1e-3 on ~98 fold positives is DISCRETE: one lens is
    1/98 = 0.0102 TPR. A relative tolerance like 5% of the final value is ~0.006 here --
    SMALLER THAN ONE LENS -- so it can never be satisfied by a curve that moves at all, and
    would report "never flattened" for every run as a pure artifact of the quantisation.
    The caller passes the bootstrap CI half-width (floored at one lens), which is the same
    yardstick the flatness verdict uses, so "flattened" means "no longer moving by an amount
    the metric can resolve".
    """
    if len(pts) < 3:
        return None
    final = pts[-1][1]
    for i, (s, v) in enumerate(pts):
        if all(abs(v2 - final) <= tol for _, v2 in pts[i:]):
            return s
    return None          # None == did NOT flatten within the budget (a real finding)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_ROOT, "runs",
                                                  "lever1_screen_report.json"))
    args = ap.parse_args(argv)

    print("=" * 84)
    print("LEVER 1 -- FOLD-0 SCREEN REPORT   (metric = TPR@FPR=1e-3, R1 = step 8000)")
    print("=" * 84)

    have, missing = {}, []
    for key, d, rel, nb in RUNS:
        p = os.path.join(_ROOT, "runs", d, *rel.split("/"))
        if os.path.exists(p):
            ps, ns, pid, nid = load_scores(p)
            have[key] = {"pos": ps, "neg": ns, "pos_ids": pid, "neg_ids": nid,
                         "dir": os.path.join(_ROOT, "runs", d), "n_blocks": nb}
        else:
            missing.append((key, p))
    for key, p in missing:
        print(f"  [missing] {key}: {os.path.relpath(p, _ROOT)}")
    if "nb0_image" not in have:
        print("\nSTOP: the control run has not produced scores yet. Nothing to report.")
        return 3

    def yz(k):
        d = have[k]
        y = np.concatenate([np.ones(d["pos"].size, np.int64),
                            np.zeros(d["neg"].size, np.int64)])
        s = np.concatenate([d["pos"], d["neg"]])
        return y, s

    report = {"target_fpr": args.target_fpr, "bootstrap": args.bootstrap,
              "selection_rule": "R1_last_checkpoint_step8000", "runs": {}}

    # ---------------- 3. CONTROL ----------------
    print("\n" + "=" * 84)
    print("[3] CONTROL -- nb0_image (image-space, frozen) vs nb0_cached (round 1)")
    print("=" * 84)
    control = {}
    if "nb0_cached" in have:
        y, s_cac = yz("nb0_cached")
        t_cac = tpr_at_fpr(y, s_cac, args.target_fpr)
        print(f"  reference  nb0_cached (round 1, aug=none, fp32) : TPR = {t_cac:.6f}")
        print("\n  Two image-space controls. The weak-aug/fp16 run differs from the cached")
        print("  run in TWO ways at once, so it cannot by itself decide whether the code")
        print("  path is right; the noaug/fp32 run is matched in both and isolates it.\n")
        for key, desc in (("nb0_image_noaug", "aug=none, fp32  (ISOLATING code-path control)"),
                          ("nb0_image", "aug=weak, fp16  (the protocol run)")):
            if key not in have:
                print(f"  [missing] {key}")
                continue
            _, s = yz(key)
            a, b = have[key], have["nb0_cached"]
            aligned = None
            if a["pos_ids"] is not None and b["pos_ids"] is not None:
                aligned = bool(np.array_equal(a["pos_ids"], b["pos_ids"])
                               and np.array_equal(a["neg_ids"], b["neg_ids"]))
            t = tpr_at_fpr(y, s, args.target_fpr)
            bs = paired_bootstrap(y, s, s_cac,
                                  metric_fn=default_tpr_metric(args.target_fpr),
                                  B=args.bootstrap, seed=args.seed)
            ok = bool(bs["lo"] <= 0.0 <= bs["hi"])
            print(f"  {key:<16} {desc}")
            print(f"      TPR = {t:.6f}   diff vs cached = {t - t_cac:+.4f}   "
                  f"CI [{bs['lo']:+.4f}, {bs['hi']:+.4f}]   row-aligned={aligned}")
            print(f"      -> {'PASS' if ok else 'FAIL'} (CI "
                  f"{'contains' if ok else 'EXCLUDES'} 0)")
            control[key] = {"tpr": float(t), "diff_vs_cached": float(t - t_cac),
                            "ci": [bs["lo"], bs["hi"]], "passed": ok,
                            "row_aligned": aligned}
        control["tpr_nb0_cached"] = float(t_cac)

        # the aug+fp16 effect, measured directly (both endpoints are image-space)
        if "nb0_image" in have and "nb0_image_noaug" in have:
            _, s_w = yz("nb0_image")
            _, s_n = yz("nb0_image_noaug")
            bs = paired_bootstrap(y, s_w, s_n,
                                  metric_fn=default_tpr_metric(args.target_fpr),
                                  B=args.bootstrap, seed=args.seed)
            print(f"\n  AUGMENTATION+fp16 EFFECT on the FROZEN encoder "
                  f"(weak/fp16 - none/fp32):")
            print(f"      {bs['point']:+.4f}   CI [{bs['lo']:+.4f}, {bs['hi']:+.4f}]   "
                  f"significant={bool(bs.get('significant'))}")
            print("      [UNVERIFIED ATTRIBUTION] these two runs vary augmentation AND")
            print("      precision together, so the split between them is NOT established.")
            print("      Separating it needs one more run (aug=none + fp16).")
            control["aug_fp16_effect"] = {"diff": float(bs["point"]),
                                          "ci": [bs["lo"], bs["hi"]],
                                          "significant": bool(bs.get("significant")),
                                          "attribution": "UNVERIFIED (aug and fp16 vary together)"}
        # the control gate: the ISOLATING run is what decides the code path
        iso = control.get("nb0_image_noaug", {}).get("passed")
        control["passed"] = bool(iso) if iso is not None else \
            bool(control.get("nb0_image", {}).get("passed"))
    else:
        print("  [missing] round-1 cached scores -- control cannot be evaluated.")
        control["passed"] = None
    report["control"] = control

    # ---------------- 3b. VENUE PORTABILITY ----------------
    # nb1/nb3 run on Kaggle T4 (Turing); the local nb0_image was measured on Ada. If the
    # same configuration disagrees across venues by more than ONE LENS, that is a finding
    # about portability, not a nuisance -- and it would undermine using either as the
    # baseline for the other's rungs.
    if "nb0_image_kaggle" in have and "nb0_image" in have:
        print("\n" + "=" * 84)
        print("[3b] VENUE PORTABILITY -- nb0_image: local (Ada 4050) vs Kaggle (T4)")
        print("=" * 84)
        y_v, s_loc = yz("nb0_image")
        _, s_kag = yz("nb0_image_kaggle")
        t_loc = tpr_at_fpr(y_v, s_loc, args.target_fpr)
        t_kag = tpr_at_fpr(y_v, s_kag, args.target_fpr)
        d = t_kag - t_loc
        bs = paired_bootstrap(y_v, s_kag, s_loc,
                              metric_fn=default_tpr_metric(args.target_fpr),
                              B=args.bootstrap, seed=args.seed)
        print(f"  local  nb0_image (Ada 4050) : {t_loc:.4f}")
        print(f"  Kaggle nb0_image (T4)       : {t_kag:.4f}")
        print(f"  difference                  : {d:+.4f}  "
              f"({abs(d) / ONE_LENS:.1f} lenses)  CI [{bs['lo']:+.4f}, {bs['hi']:+.4f}]")
        if abs(d) > ONE_LENS:
            print(f"  *** VENUE DISAGREEMENT > 1 lens ({ONE_LENS:.4f}). This is a FINDING "
                  f"about cross-venue")
            print(f"      portability, not a nuisance: the same configuration and seed give "
                  f"different")
            print(f"      results on Turing vs Ada. Report it, and use the SAME-VENUE "
                  f"baseline for each rung. ***")
        else:
            print(f"  Agreement within one lens ({ONE_LENS:.4f}) -- venues are "
                  f"interchangeable for this metric.")
        report["venue_portability"] = {
            "tpr_local": float(t_loc), "tpr_kaggle": float(t_kag), "diff": float(d),
            "lenses": abs(d) / ONE_LENS, "ci": [bs["lo"], bs["hi"]],
            "exceeds_one_lens": bool(abs(d) > ONE_LENS)}

    # ---------------- 4. HEADLINE ----------------
    # Prefer the SAME-HARDWARE replicate as the reference when it exists.
    ref_key = "nb0_image_kaggle" if "nb0_image_kaggle" in have else "nb0_image"
    print("\n" + "=" * 84)
    print(f"[4] FOLD-0 TPR@FPR=1e-3 -- each rung, paired-bootstrap CI vs {ref_key}")
    print("=" * 84)
    print(f"{'run':>12} | {'n_blocks':>8} | {'TPR@1e-3':>9} | {'diff vs nb0_image':>18} | "
          f"{'95% CI':>22} | signif")
    print("-" * 84)
    y, s_ctrl = yz(ref_key)
    ci_half = {}
    for key in (ref_key, "nb1", "nb3"):
        if key not in have:
            print(f"{key:>12} | {have.get(key, {}).get('n_blocks', '-'):>8} | "
                  f"{'(not run)':>9} |")
            continue
        _, s = yz(key)
        t = tpr_at_fpr(y, s, args.target_fpr)
        if key == ref_key:
            bs = paired_bootstrap(y, s, metric_fn=default_tpr_metric(args.target_fpr),
                                  B=args.bootstrap, seed=args.seed)
            ci_half[key] = (bs["hi"] - bs["lo"]) / 2.0
            print(f"{key:>12} | {have[key]['n_blocks']:>8} | {t:9.4f} | "
                  f"{'(reference)':>18} | [{bs['lo']:+.4f}, {bs['hi']:+.4f}]".ljust(78)
                  + " | --")
            report["runs"][key] = {"tpr": float(t), "own_ci": [bs["lo"], bs["hi"]],
                                   "n_blocks": have[key]["n_blocks"]}
            continue
        bs = paired_bootstrap(y, s, s_ctrl, metric_fn=default_tpr_metric(args.target_fpr),
                              B=args.bootstrap, seed=args.seed)
        ci_half[key] = (bs["hi"] - bs["lo"]) / 2.0
        sig = "YES" if bs.get("significant") else "no"
        print(f"{key:>12} | {have[key]['n_blocks']:>8} | {t:9.4f} | "
              f"{bs['point']:+18.4f} | [{bs['lo']:+.4f}, {bs['hi']:+.4f}]".ljust(78)
              + f" | {sig}")
        report["runs"][key] = {"tpr": float(t), "diff_vs_nb0_image": float(bs["point"]),
                               "ci": [bs["lo"], bs["hi"]],
                               "significant": bool(bs.get("significant")),
                               "ci_half_width": ci_half[key],
                               "n_blocks": have[key]["n_blocks"]}
    print("-" * 84)
    print("  'signif' = the paired-difference CI excludes 0 (class-conditional bootstrap).")
    print(f"  Reference = {ref_key} (weak aug, fp16) -- MATCHED to nb1/nb3 in augmentation")
    print("  and precision, so this comparison isolates the unfreeze and nothing else.")
    print("  It is the reference named in the PRE-REGISTERED rule ('nb0_image'); when the")
    print("  same-hardware Kaggle replicate exists it is used, since nb1/nb3 run on T4.")

    # ---- SUPPLEMENTARY bar: the BEST frozen configuration ----
    # Measurement showed nb0_image (weak/fp16) is a DEGRADED frozen configuration
    # (significantly below the aug=none/fp32 frozen run). Beating a degraded reference is
    # an easier test than beating the best frozen Baseline A, so reporting only the
    # pre-registered comparison would flatter the unfrozen rungs. The pre-registered rule
    # is applied EXACTLY as written in [6] -- this is reported ALONGSIDE it, never instead
    # of it, and it is NOT a retroactive rule change.
    if "nb0_cached" in have:
        print("\n  SUPPLEMENTARY -- same rungs vs the BEST FROZEN configuration "
              "(nb0_cached = aug none/fp32):")
        _, s_best = yz("nb0_cached")
        t_best = tpr_at_fpr(y, s_best, args.target_fpr)
        print(f"    {'run':>12} | {'TPR@1e-3':>9} | {'diff vs best frozen':>20} | "
              f"{'95% CI':>22} | signif")
        print("    " + "-" * 76)
        for key in ("nb1", "nb3"):
            if key not in have:
                continue
            _, s = yz(key)
            bs = paired_bootstrap(y, s, s_best,
                                  metric_fn=default_tpr_metric(args.target_fpr),
                                  B=args.bootstrap, seed=args.seed)
            print(f"    {key:>12} | {tpr_at_fpr(y, s, args.target_fpr):9.4f} | "
                  f"{bs['point']:+20.4f} | [{bs['lo']:+.4f}, {bs['hi']:+.4f}]".ljust(72)
                  + f" | {'YES' if bs.get('significant') else 'no'}")
            report["runs"].setdefault(key, {})["vs_best_frozen"] = {
                "diff": float(bs["point"]), "ci": [bs["lo"], bs["hi"]],
                "significant": bool(bs.get("significant"))}
        print(f"    (best frozen TPR = {t_best:.4f}; this bar mixes the unfreeze WITH the "
              f"augmentation/precision")
        print("     difference, so it answers 'is the unfrozen config better than the best")
        print("     frozen config?', NOT 'does unfreezing help, all else equal?'.)")

    # ---------------- 5. FLATNESS / CURVE / GATE ----------------
    print("\n" + "=" * 84)
    print("[5] R1 FLATNESS RE-CHECK + where each curve flattens + collapse gate")
    print("=" * 84)
    print("  R1 (last checkpoint) was verified FLAT-TIGHT under a FROZEN encoder. A")
    print("  trainable encoder can degrade on target late while source-val keeps improving,")
    print("  so the check is re-run per rung rather than inherited.")
    # iterate the SAME keys section [4] scored, so every rung has a CI half-width to test
    # its drift against. Using a hardcoded 'nb0_image' here silently gave half=None for the
    # reference run, which then fell through to a spurious "NOT FLAT" verdict.
    for key in (ref_key, "nb1", "nb3"):
        if key not in have:
            continue
        rows = read_curve(have[key]["dir"])
        pts = _col(rows, "fold0_tpr_1e-3")
        agr = _col(rows, "agreement_fold0")
        print(f"\n  --- {key} (n_blocks={have[key]['n_blocks']}, {len(pts)} curve points) ---")
        if not pts:
            print("      no curve.csv -- skipped")
            continue
        win = [(s, v) for s, v in pts if WINDOW[0] <= s <= WINDOW[1]]
        half = ci_half.get(key)
        if win:
            drift = max(v for _, v in win) - min(v for _, v in win)
            verdict = ("FLAT (drift <= CI half-width) -> R1 defensible"
                       if half is not None and drift <= half else
                       "NOT FLAT (drift > CI half-width) -> R1 NOT defensible; R3 "
                       "(median checkpoint) becomes live")
            print(f"      window {WINDOW[0]}-{WINDOW[1]}: n={len(win)}  "
                  f"drift={drift:.4f}  CI half-width={half:.4f}" if half is not None
                  else f"      window drift={drift:.4f}")
            print(f"      VERDICT: {verdict}")
        elif half is None:
            drift, verdict = None, ("NOT TESTED -- no bootstrap CI for this run (it was "
                                    "not scored in section [4])")
            print(f"      VERDICT: {verdict}")
        else:
            drift, verdict = None, "no in-window checkpoints"
            print(f"      VERDICT: {verdict}")
        n_pos = have[key]["pos"].size
        one_lens = 1.0 / max(n_pos, 1)
        tol = max(half if half is not None else 0.0, one_lens)
        fs = flatten_step(pts, tol)
        final = pts[-1][1]
        print(f"      flattens at ~step {fs if fs is not None else 'NEVER'} "
              f"(stays within {tol:.4f} = max(CI half-width, 1 lens={one_lens:.4f}) of the "
              f"final {final:.4f}); max over curve = {max(v for _, v in pts):.4f}")
        if fs is None:
            print(f"      [FLAG] the curve had NOT flattened by {BUDGET}. The 8000-step "
                  f"budget was derived from FROZEN-encoder convergence (baseline ~6000, "
                  f"FreeMatch ~2800, DANN ~4600) and does not necessarily transfer. "
                  f"Reported, NOT silently extended.")
        gate = None
        if agr:
            a = np.array([v for _, v in agr])
            print(f"      agreement: first={a[0]:.4f} last={a[-1]:.4f} min={a.min():.4f} "
                  f"max={a.max():.4f} range={a.max()-a.min():.4f}")
            gate = {"first": float(a[0]), "last": float(a[-1]), "min": float(a.min()),
                    "max": float(a.max())}
        report["runs"].setdefault(key, {}).update(
            {"window_drift": (None if drift is None else float(drift)),
             "flatness_verdict": verdict, "flatten_step": fs,
             "agreement": gate, "curve_points": len(pts)})

    # ---------------- 6. DECISION ----------------
    print("\n" + "=" * 84)
    print("[6] PRE-REGISTERED DECISION RULE")
    print("=" * 84)
    rule = None
    for key in ("nb1", "nb3", "nb0_image"):
        p = os.path.join(have.get(key, {}).get("dir", ""), "run_config.json")
        if os.path.exists(p):
            rule = json.load(open(p)).get("decision_rule")
            if rule:
                break
    if rule:
        print(f"  RULE      : {rule['rule']}")
        print(f"  AMENDMENT : {rule['amendment_rationale']}")
        print(f"  RECORDED BEFORE ANY SCREEN RESULT EXISTED: "
              f"{rule.get('recorded_before_any_screen_result_existed')}")
    report["decision_rule"] = rule

    if control and control.get("passed") is False:
        print("\n  OUTCOME: BLOCKED -- the control failed; no rung may be promoted.")
    else:
        winners = [(k, report["runs"][k]) for k in ("nb1", "nb3")
                   if k in report["runs"] and "diff_vs_nb0_image" in report["runs"][k]]
        if not winners:
            print("\n  OUTCOME: PENDING -- nb1/nb3 have not both run yet.")
        else:
            beats = [(k, v) for k, v in winners
                     if v["diff_vs_nb0_image"] > v["ci_half_width"]]
            if not beats:
                print("\n  OUTCOME: DO NOT COMMIT -- no rung beats nb0_image by more than "
                      "the fold-0 CI half-width.")
            else:
                best = max(beats, key=lambda kv: kv[1]["diff_vs_nb0_image"])
                tied = [k for k, v in beats
                        if abs(v["diff_vs_nb0_image"] - best[1]["diff_vs_nb0_image"])
                        <= best[1]["ci_half_width"]]
                pick = "nb3" if "nb3" in tied else best[0]
                print(f"\n  OUTCOME: COMMIT with {pick}"
                      + ("  (tie -> prefer nb3, per the amended rule: protocol match with "
                         "Lines et al.)" if len(tied) > 1 else ""))
                report["outcome"] = pick

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
