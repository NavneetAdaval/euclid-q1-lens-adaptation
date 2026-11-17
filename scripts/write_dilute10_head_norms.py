"""
write_dilute10_head_norms.py
============================
Writes ``runs/dilute10_head_norms/head_norms.json``: the norms of the stored final frozen-stage
heads that initialised the unfrozen CORAL x nb3 and FreeMatch x nb3 arms, on the diluted (1:10)
target and in round 1 (balanced). PROJECTv7 7.46 gave the CORAL values in prose only; this file
is their artifact (paper v6 brief, section 1).

    python -u scripts/write_dilute10_head_norms.py

Reads state dicts only (``ckpt["head"]``): no encoder, no data, no scorer, no free parameter.
Per head: the L2 norm of the LayerNorm gain (``trunk.0.weight``) and of its bias
(``trunk.0.bias``), ||head||_2 over every head tensor (the definition of
``run_unfreeze_freematch.head_l2_norm``, which wrote gate_values.json's ``head_l2``), and the
file's sha256.

Checks, each a hard stop:
  * CORAL's gain norms, bias-free ||head||_2 values reproduce 7.46 to one decimal;
  * every diluted ||head||_2 equals its gate_values.json ``head_l2`` (6 decimals, as recorded);
  * every round-1 ||head||_2 equals the ``init_head_l2`` its balanced unfrozen run recorded.
Refuses if the output file already exists.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
OUT_DIR = os.path.join(REPO, "runs", "dilute10_head_norms")
OUT = os.path.join(OUT_DIR, "head_norms.json")
FOLDS = (1, 2, 3, 4)


def R(*p):
    return os.path.join(REPO, *p)


HEADS = {
    "coral": {
        "diluted": "runs/dilute10/coral_frozen/fold{k}/checkpoints/step_8000.pt",
        "round1": "runs/bakeoff_coral/fold{k}/checkpoints/step_8000.pt",
    },
    "freematch": {
        "diluted": "runs/dilute10/freematch_frozen/fold{k}/checkpoints/step_8000.pt",
        "round1": "runs/bakeoff_freematch_5fold/fold{k}/checkpoints/step_8000.pt",
    },
}
# Where each head's ||head||_2 was recorded when it was used.
RECORDED_L2 = {
    ("coral", "diluted"): ("runs/dilute10/coral_frozen/fold{k}/gate_values.json", "head_l2"),
    ("freematch", "diluted"): ("runs/dilute10/freematch_frozen/fold{k}/gate_values.json", "head_l2"),
    ("coral", "round1"): ("kaggle/coral_r4/runs/main/extracted/coral_r4_confirm/fold{k}/run.json",
                          "init_head_l2"),
    ("freematch", "round1"): ("runs/freematch_nb3_unfrozen_fold{k}/run_config.json", "init_head_l2"),
}
# PROJECTv7 7.46, CORAL only, as printed there (one decimal).
S746 = {
    "ln_gain_norm": {"diluted": (97.0, 103.7, 98.2, 102.2), "round1": (22.4, 33.1, 24.0, 23.4)},
    "head_l2": {"diluted": (134.2, 139.0, 133.4, 137.1), "round1": (67.5, 72.6, 68.2, 66.0)},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def l2(t):
    return float(math.sqrt(float((t.detach().double() ** 2).sum())))


def main():
    import torch

    if os.path.exists(OUT):
        sys.exit(f"refusing: {OUT} exists")
    out = {"what": "L2 norms of the stored final frozen-stage heads (step 8000) that initialise "
                   "the unfrozen CORAL x nb3 and FreeMatch x nb3 arms; diluted (1:10) vs round 1",
           "reads": "ckpt['head'] state dicts only; no encoder, no data, no scorer",
           "head_l2_definition": "sqrt of the sum of squares over every head tensor "
                                 "(trunk.0.weight, trunk.0.bias, classifier.weight, classifier.bias), "
                                 "as run_unfreeze_freematch.head_l2_norm",
           "unit_gain_norm": math.sqrt(640),
           "torch_version": torch.__version__,
           "script_sha256": sha256(os.path.abspath(__file__)),
           "heads": {}, "checks": {}}
    failures = []
    for arm, stages in HEADS.items():
        out["heads"][arm] = {}
        for stage, pat in stages.items():
            out["heads"][arm][stage] = {}
            rec_path, rec_key = RECORDED_L2[(arm, stage)]
            for k in FOLDS:
                rel = pat.format(k=k)
                ck = torch.load(R(rel), map_location="cpu", weights_only=False)
                head = ck["head"]
                keys = sorted(head.keys())
                if keys != ["classifier.bias", "classifier.weight", "trunk.0.bias", "trunk.0.weight"]:
                    sys.exit(f"STOP: unexpected head keys in {rel}: {keys}")
                row = {
                    "path": rel,
                    "sha256": sha256(R(rel)),
                    "ln_gain_norm": l2(head["trunk.0.weight"]),
                    "ln_bias_norm": l2(head["trunk.0.bias"]),
                    "classifier_weight_norm": l2(head["classifier.weight"]),
                    # float32, summed in state-dict (= head.parameters()) order, exactly as
                    # head_l2_norm computed the recorded value; float64 beside it.
                    "head_l2": float(torch.sqrt(sum((head[n].detach().float() ** 2).sum()
                                                    for n in head.keys()))),
                    "head_l2_float64": float(math.sqrt(sum(
                        float((head[n].detach().double() ** 2).sum()) for n in keys))),
                }
                rp = rec_path.format(k=k)
                with open(R(rp), encoding="utf-8") as f:
                    rec = float(json.load(f)[rec_key])
                row["recorded_head_l2"] = {"path": rp, "key": rec_key, "value": rec,
                                           "match_6dp": round(row["head_l2"], 6) == round(rec, 6)}
                if not row["recorded_head_l2"]["match_6dp"]:
                    failures.append(f"{arm}/{stage}/fold{k}: head_l2 {row['head_l2']:.6f} != "
                                    f"{rp} {rec_key} {rec:.6f}")
                out["heads"][arm][stage][str(k)] = row
    # 7.46, CORAL, one decimal
    s746 = {}
    for q, by_stage in S746.items():
        for stage, vals in by_stage.items():
            for k, v in zip(FOLDS, vals):
                got = out["heads"]["coral"][stage][str(k)][q]
                ok = f"{got:.1f}" == f"{v:.1f}"
                s746[f"{q}/{stage}/fold{k}"] = {"s746": v, "computed_1dp": f"{got:.1f}", "match": ok}
                if not ok:
                    failures.append(f"7.46 {q} {stage} fold{k}: {got:.1f} != {v:.1f}")
    out["checks"]["s746_coral_1dp"] = s746
    out["checks"]["recorded_head_l2_6dp"] = "all match" if not any(
        "head_l2" in f and "7.46" not in f for f in failures) else "MISMATCH"
    out["checks"]["passed"] = not failures
    out["checks"]["failures"] = failures
    if failures:
        print("\n".join(failures))
        sys.exit("STOP: mismatch; nothing written")
    out["written_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for arm in HEADS:
        for stage in ("diluted", "round1"):
            rows = out["heads"][arm][stage]
            print(arm, stage, " | ".join(
                f"g {rows[str(k)]['ln_gain_norm']:.1f} b {rows[str(k)]['ln_bias_norm']:.1f} "
                f"h {rows[str(k)]['head_l2']:.2f}" for k in FOLDS))
    print("checks passed; wrote", OUT, sha256(OUT))


if __name__ == "__main__":
    main()
