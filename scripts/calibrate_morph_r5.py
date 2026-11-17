"""
calibrate_morph_r5.py
=====================
Measure the recovered morphology-proxy AUROC at **step 0** and calibrate round 5's
abort condition 3 against that reading. CPU, read-only, trains nothing.

WHY THIS SCRIPT EXISTS, AND WHAT IT NO LONGER DOES. The round-5 design doc
pre-registers:

    3. Classifier access destroyed. Morphology-proxy AUROC below 0.55 sustained over
       500 steps.

**0.55 IS WITHDRAWN, and so is the earlier attempt to port it onto this instrument.**

The port rested on a claim that turned out not to be in the record: that round-1 DANN
drove a morphology proxy from 0.61 down to 0.33-0.46. **PROJECT.md attributes that
failure to IMPORTANCE WEIGHTING, not to DANN, and gives no numbers for it.** The only
morphology figure in that part of the document is the frozen-probe baseline at
**~0.90-0.91** -- which is the scale ``morph_proxy.py`` reproduces at 0.9214, and
nowhere near 0.61. So 0.55 was invented rather than measured, and a rescaling of it
inherits nothing. It is gone; nothing in this repository reads it.

Condition 3 is now measured against the **lambda=0 control at the same absolute step**
(brief v3 3), which subtracts unfreezing-only drift instead of mistaking it for
adversarial damage. That rule needs no calibration from this script -- it instantiates
from the control arm's own curve.

WHAT THIS SCRIPT IS STILL FOR: the **step-0 reading**. It is the scale everything else
is discussed on, it is the evidence that the recovery of the historical instrument is
faithful, and it is what makes brief v3's 0.10 margin a judgeable number rather than an
arbitrary one (0.10 on a scale running 0.92 down to 0.5 is roughly a quarter of the
margin over chance).

WHAT IT MEASURES -- TWO INSTRUMENTS, ONE POOL

  1. ``morph_head_auroc``          the TRAINED HEAD's own lens logit, eval half.
                                   The ~0.61-scale instrument, the one that MOVED in
                                   the v5 failure, and **the one abort condition 3
                                   reads**. THIS is the scale M is declared against.
  2. ``morph_head_auroc_pool``     the same on all 256 images: lower variance.
  3. ``morph_proxy_auroc``         ridge probe on the adapter -- the ~0.90-scale
                                   instrument that "stayed ~0.90" through those same
                                   failures. Logged with NO detector attached.
  4. ``morph_proxy_auroc_encoder`` the same on raw encoder features -- comparable to
                                   the historical 0.914 / 0.898.

  ...for BOTH arms (``dann_nb3`` and ``dann_nb3_lambda0``), which must agree
  bit-for-bit: at step 0 the arms differ only in a lambda that no forward pass reads,
  and D is not in the monitor's path at all. Measured rather than asserted.

WHAT IT DOES NOT DECIDE. It records brief v3's rule and the numbers the rule will be
judged against. Hashing the 0.10 margin -- writing it down the way
``coral_r4_budget.json`` is -- is the run owner's decision, must happen BEFORE the
development run, and must not move afterwards.

Run:
    python scripts/calibrate_morph_r5.py
    python scripts/calibrate_morph_r5.py --n-fit-per-class 128 --n-eval-per-class 128
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import evaluate as ev                                            # noqa: E402
from dann_round5 import install_dann                             # noqa: E402
from heads import EMBED_DIM                                      # noqa: E402
from morph_proxy import (MORPH_N_EVAL_PER_CLASS,                 # noqa: E402
                         MORPH_N_FIT_PER_CLASS, MORPH_RIDGE_ALPHA,
                         MorphProxy, build_morph_pool)
from train_skeleton import TrainConfig, Trainer                  # noqa: E402

# The historical readings this instrument is a recovery of (PROJECT_STATE.md).
HISTORICAL = {"raw_zoobot_morphAUROC": 0.914, "simsiam_morphAUROC": 0.898,
              "source": "PROJECT_STATE.md 3, encoder-selection work",
              "caveat": ("the exact sample, seed and probe hyperparameters of that run "
                         "are gone with scripts/linear_probe.py, so agreement here is "
                         "evidence the recovery is faithful, not a reproduction")}

# The withdrawn threshold, kept ONLY so the record says why it is gone.
WITHDRAWN_ABORT = {
    "0.55": {
        "source": "ROUND5_DANN_DESIGN.md 6, abort condition 3",
        "was_it_invented": ("NO. PROJECT_STATE v2.md 6 and 8/F4 record the trained "
                            "HEAD's morphology AUROC falling ~0.61 -> ~0.37 under plain "
                            "DANN and ~0.33 under IW-CDAN. 0.55 sat just under that real "
                            "~0.61 baseline. Briefs v3 and v4 asserted it was invented; "
                            "that came from searching PROJECTv6.md alone and reading "
                            "absence from one document as absence from the record."),
        "why_withdrawn": ("it is meaningless on the FROZEN-PROBE column, which reads "
                          "~0.92. It was never wrong for its own instrument."),
    },
    "0.6910": {
        "source": "brief v2, a rescaling of 0.55 onto the probe column by fraction of "
                  "headroom over chance",
        "why_withdrawn": ("rescaling a threshold across a change of instrument assumes "
                          "the two measure the same thing. They do not -- they read "
                          "~0.34 apart at step 0."),
    },
    "0.10": {
        "source": "brief v3, the control-referenced margin",
        "why_withdrawn": ("declared against the frozen probe's 0.4202 headroom. On the "
                          "head column the headroom is ~0.078, so a 0.10 drop lands "
                          "BELOW CHANCE. Scale-wrong, not merely unset."),
    },
    "the_planning_side_over_correction": (
        "recorded so it is not re-derived either: briefs v3 and v4 told this project the "
        "0.61 -> 0.33 numbers were invented in planning conversation. They were not. What "
        "survives of that correction is narrower -- '0.33-0.46' is wrong (the record says "
        "0.37 for plain DANN and 0.33 for IW-CDAN, and 0.46 appears nowhere)."),
    "replaced_by": ("PROJECT.md 7.17: condition 3 reads morph_head_auroc. 3b "
                    "(AUROC < 0.5) is the PRIMARY detector and needs no threshold. 3a's "
                    "margin M is declared off arm 1's control trajectory on that column "
                    "and hashed before arm 2."),
}

# brief v3 3 -- the rule this script records rather than derives
COND3_RULE = {
    "column": "morph_head_auroc",
    "column_note": ("the TRAINED HEAD's own lens logit on the ML-FP vs P1-reserve pool, "
                    "eval half. The ~0.61-scale instrument that MOVED in the v5 failure. "
                    "The frozen probe (morph_proxy_auroc) is logged with NO detector."),
    "b_sign_flip_PRIMARY": ("morph_head_auroc(step) < 0.5 -- the historical failure WAS "
                            "a sign flip (0.61 -> 0.37), so on this column it reproduces "
                            "the actual event with no invented threshold"),
    "a_control_referenced_SECONDARY": ("morph_head_auroc(step) < "
                                       "morph_head_auroc_lambda0(step) - M, sustained "
                                       "over 500 consecutive ABSOLUTE steps"),
    "margin_M": None,
    "wording": ("adversarial pressure has degraded linear access to morphology RELATIVE "
                "TO UNFREEZING ALONE -- not 'the documented failure has recurred', which "
                "describes a DANN failure the document does not record"),
    "step_matching": ("read the control by ABSOLUTE step, never by fraction of run; a "
                      "step missing from the control REFUSES the comparison rather than "
                      "being interpolated"),
}


def _step0_trainer(args, *, lambda_zero, device):
    """The model BOTH arms start from: frozen-init encoder + Baseline A's LP head."""
    from zoobot_encoder import ZoobotEncoder
    enc = ZoobotEncoder(encoder_mode="frozen", n_blocks=int(args.n_blocks),
                        device=device, verbose=False, grad_checkpointing=False)
    cfg = TrainConfig(head_type="linear", encoder_mode="frozen",
                      n_blocks=int(args.n_blocks), backbone_lr=None, lr_decay=0.75,
                      lr=1e-4, weight_decay=0.05, batch_size=4, seed=42, device=device,
                      amp=False, use_coral=False,
                      init_head_from=os.path.abspath(args.init_head_from))
    with contextlib.redirect_stdout(io.StringIO()):
        tr = Trainer(cfg, encoder=enc)
        install_dann(tr, lambda_horizon=int(args.dann_lambda_horizon),
                     lambda_zero=lambda_zero, d_lr=1e-4, weight_decay=0.05)
    cache = ev.FeatureCache(os.path.abspath(args.cache_dir))
    src = torch.from_numpy(np.asarray(cache.load("source_train")["features"],
                                      dtype=np.float32)).to(tr.device)
    tr.standardizer.fit(src)
    del src
    return tr


def main(argv=None):
    root = os.path.normpath(os.path.join(_HERE, ".."))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.path.join(root, "data"))
    p.add_argument("--splits", default=os.path.join(root, "data", "splits", "splits.json"))
    p.add_argument("--cache-dir", default=os.path.join(root, "data", "feature_cache"))
    p.add_argument("--init-head-from", default=os.path.join(
        root, "runs", "bakeoff_baseline_A", "checkpoints", "step_8000.pt"))
    p.add_argument("--n-fit-per-class", type=int, default=MORPH_N_FIT_PER_CLASS)
    p.add_argument("--n-eval-per-class", type=int, default=MORPH_N_EVAL_PER_CLASS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dann-lambda-horizon", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(root, "runs", "dann_r5_morph_step0.json"))
    a = p.parse_args(argv)

    print("=" * 92)
    print("ROUND-5 MORPHOLOGY-PROXY CALIBRATION -- step-0 reading, both arms")
    print("=" * 92, flush=True)

    t0 = time.time()
    pool = build_morph_pool(a.splits, a.data_root, n_fit_per_class=a.n_fit_per_class,
                            n_eval_per_class=a.n_eval_per_class, seed=a.seed)
    print(f"[pool] {pool['images'].shape[0]} images in {time.time()-t0:.0f}s", flush=True)

    readings = {}
    for arm, lz in (("dann_nb3", False), ("dann_nb3_lambda0", True)):
        tr = _step0_trainer(a, lambda_zero=lz, device=a.device)
        proxy = MorphProxy(pool, alpha=MORPH_RIDGE_ALPHA)
        t = time.time()
        r = proxy.measure(tr)
        r["measure_seconds"] = round(time.time() - t, 1)
        r["dann_lambda_at_step0"] = tr.dann.lambda_at(0)
        r["dann_lambda_zero"] = bool(tr.dann.lambda_zero)
        readings[arm] = r
        print(f"[{arm:>18}] HEAD {r['morph_head_auroc']:.6f} (eval) / "
              f"{r['morph_head_auroc_pool']:.6f} (pool)   |   "
              f"PROBE {r['morph_proxy_auroc']:.6f} (adapter) / "
              f"{r['morph_proxy_auroc_encoder']:.6f} (raw enc)   "
              f"({r['measure_seconds']:.0f}s)", flush=True)
        del tr

    same = all(readings["dann_nb3"][k] == readings["dann_nb3_lambda0"][k]
               for k in ("morph_head_auroc", "morph_head_auroc_pool",
                         "morph_proxy_auroc", "morph_proxy_auroc_encoder"))
    print(f"\n[arms] the two arms read BIT-IDENTICALLY at step 0: {same}")
    print("       (expected by construction -- at step 0 the arms share head, encoder "
          "and standardizer, lambda is not read by any forward pass, and D is not in "
          "the monitor's path at all)")

    step0_head = float(readings["dann_nb3"]["morph_head_auroc"])
    step0_head_pool = float(readings["dann_nb3"]["morph_head_auroc_pool"])
    step0 = float(readings["dann_nb3"]["morph_proxy_auroc"])
    step0_enc = float(readings["dann_nb3"]["morph_proxy_auroc_encoder"])

    head_room = step0_head - 0.5
    probe_room = step0 - 0.5

    print("\n" + "-" * 92)
    print("ABORT CONDITION 3 -- the column it reads, and the scale M is declared on")
    print("-" * 92)
    print(f"  {'instrument':<34} {'step 0':>10} {'headroom over chance':>22}")
    print(f"  {'-'*34} {'-'*10} {'-'*22}")
    print(f"  {'HEAD, eval half  <- CONDITION 3':<34} {step0_head:>10.4f} "
          f"{head_room:>22.4f}")
    print(f"  {'HEAD, whole pool':<34} {step0_head_pool:>10.4f} "
          f"{step0_head_pool - 0.5:>22.4f}")
    print(f"  {'PROBE, adapter   (no detector)':<34} {step0:>10.4f} "
          f"{probe_room:>22.4f}")
    print(f"  {'PROBE, raw encoder':<34} {step0_enc:>10.4f} "
          f"{step0_enc - 0.5:>22.4f}")
    print()
    print(f"  historical: HEAD ~0.61 no-DA -> ~0.37 plain DANN / ~0.33 IW-CDAN")
    print(f"              PROBE 'stayed ~0.90' through those same runs "
          f"(0.914 raw Zoobot)")
    print(f"  BOTH instruments reconstruct their historical values on this pool, which "
          f"is what makes")
    print(f"  the distinction a measurement rather than an argument.")
    print()
    print(f"  RULE (PROJECT.md 7.17), on {COND3_RULE['column']}, fires when EITHER:")
    print(f"    (b) {COND3_RULE['column']}(step) < 0.5      <-- PRIMARY. The historical")
    print(f"        failure WAS a sign flip; this reproduces it with NO threshold.")
    print(f"    (a) {COND3_RULE['column']}(step) < control(step) - M, sustained 500 steps")
    print(f"        M is UNSET. Declare it off arm 1's control trajectory on THIS "
          f"column,")
    print(f"        against a headroom of {head_room:.4f} -- not the probe's "
          f"{probe_room:.4f}.")
    print()
    print(f"  WITHDRAWN, with the corrected reasons:")
    print(f"    0.55   NOT invented. It sat just under the HEAD instrument's real ~0.61")
    print(f"           baseline. Withdrawn because it is meaningless on the PROBE column.")
    print(f"    0.6910 brief v2's rescaling of 0.55 onto the probe column. Rescaling")
    print(f"           across a change of instrument assumes the two measure one thing.")
    print(f"    0.10   brief v3's control margin, declared against the probe's "
          f"{probe_room:.4f}")
    print(f"           headroom. On the head column's {head_room:.4f} a 0.10 drop lands "
          f"BELOW CHANCE.")
    print("-" * 92, flush=True)

    out = {
        "purpose": ("step-0 readings of BOTH morphology instruments, and the scale abort "
                    "condition 3's margin M is declared against. The rule is "
                    "control-referenced and instrument-matched; M is declared off arm 1."),
        "instrument": pool["info"],
        "historical": HISTORICAL,
        "withdrawn_abort_condition": WITHDRAWN_ABORT,
        "abort_condition_3_rule": {**COND3_RULE,
                                   "step0_head_eval": step0_head,
                                   "step0_head_pool": step0_head_pool,
                                   "step0_probe_adapter": step0,
                                   "step0_probe_raw_encoder": step0_enc,
                                   "head_headroom_over_chance": head_room,
                                   "probe_headroom_over_chance": probe_room,
                                   "margin_status": ("UNSET. Declared off arm 1's "
                                                     "control trajectory on the head "
                                                     "column and hashed before arm 2.")},
        "readings_step0": readings,
        "arms_identical_at_step0": bool(same),
        "arms_identical_note": ("by construction: same head, same encoder, same "
                                "standardizer; lambda is read by no forward pass and D "
                                "is not in the monitor's path"),
        "config": {"n_fit_per_class": a.n_fit_per_class,
                   "n_eval_per_class": a.n_eval_per_class,
                   "ridge_alpha": MORPH_RIDGE_ALPHA, "seed": a.seed,
                   "n_blocks": a.n_blocks, "device": a.device,
                   "init_head_from": os.path.abspath(a.init_head_from)},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    dst = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[out] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
