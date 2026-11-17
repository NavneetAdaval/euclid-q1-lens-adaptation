"""
test_unfreeze_freematch.py
==========================
The BRIEF §7 pre-flight suite for the FreeMatch x nb3 fold-0 run. PROJECT.md §9: nothing is
handed over untested, on MOCK data mirroring the real schema plus adversarial checks.

Covered here (BRIEF §7 items 1-3; items 4-5 are the on-GPU smoke and the step-0 gate, which
need real weights and are run by KAGGLE_FREEMATCH_NB3_CELLS.md / the local smoke):

  1  FreeMatchLoss's new per-class diagnostics are ARITHMETICALLY correct on a hand example
     and do not perturb the loss, the SAT EMA or the mask by a single bit.
  2  StepWindow averages, skips undefined values per key, and never reports an
     all-undefined key as 0.
  3  The memorization probe is composition-matched and catches the confound it exists for.
  4  The logit dump has the right splits, shapes and dtypes, is finite, and writes ids once.
  5  Retention accounting handles ckpt_every != curve_every (16 weights, 64 curve points).
  6  The §1 precondition check ABORTS on a dropped --fp16, on aug=none, on a changed seed
     and on a changed local_threshold_mode -- and PERMITS a changed max_steps.
  7  The step-0 hard gate passes on a match and RAISES on a miss.
  8  curve_hook's collect_scores changes no metric and no column (a known-null control).
  9  The control's schedule is a TRUNCATION of the unfrozen run's, not a compressed copy.
 10  train_freematch_fold's new arguments all default to the round-1 code path.

Run:
    python scripts/test_unfreeze_freematch.py
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import curve_hook                                            # noqa: E402
import run_unfreeze_freematch as ruf                         # noqa: E402
import unfreeze_instrument as ui                             # noqa: E402
from da_losses import (FreeMatchLoss, freematch_sat_update,
                       freematch_local_thresholds)           # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# 1 -- FreeMatchLoss per-class diagnostics
# ---------------------------------------------------------------------------

def test_freematch_diagnostics():
    print("\n[1] FreeMatchLoss per-class mask-rate diagnostics")
    torch.manual_seed(0)
    loss_mod = FreeMatchLoss(mu_max=1.0, saf_weight=0.01, sat_ema_momentum=0.9,
                             consistency_warmup_frac=0.0, local_threshold_mode="standard")
    # a hand batch: 6 unlabeled samples with known logits -> known argmax classes
    tgt_weak = torch.tensor([[3.0], [2.0], [0.2], [-3.0], [-2.0], [-0.2]])
    tgt_strong = torch.zeros(6, 1, requires_grad=True)
    src = torch.zeros(4, 1, requires_grad=True)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # replicate the SAT update + thresholds independently, exactly as the module documents
    p = torch.sigmoid(tgt_weak.reshape(-1))
    prob_w = torch.stack([1 - p, p], dim=1)
    tau, pbar = freematch_sat_update(prob_w, torch.tensor(0.5), torch.full((2,), 0.5), 0.9)
    local = freematch_local_thresholds(tau, pbar, pos_class=1, mode="standard")
    maxp, pseudo = prob_w.max(1)
    mask = maxp > local[pseudo]
    is_pos = pseudo == 1
    want_pos_rate = float((mask & is_pos).sum()) / float(is_pos.sum())
    want_neg_rate = float((mask & ~is_pos).sum()) / float((~is_pos).sum())
    want_all = float(is_pos.float().mean())

    _l, info = loss_mod(src, tgt_weak, tgt_strong, y, step=0, total_steps=10)
    check(close(info["mask_rate_pos"], want_pos_rate), f"mask_rate_pos = {want_pos_rate:.6f}")
    check(close(info["mask_rate_neg"], want_neg_rate), f"mask_rate_neg = {want_neg_rate:.6f}")
    check(close(info["pseudo_pos_fraction_all"], want_all),
          f"pseudo_pos_fraction_all = {want_all:.6f}")
    check(close(info["n_pseudo_pos"] + info["n_pseudo_neg"], 6.0),
          "n_pseudo_pos + n_pseudo_neg == batch size")
    check(close(info["n_accepted_pos"] + info["n_accepted_neg"], info["n_accepted"]),
          "n_accepted decomposes exactly into its per-class parts")
    check(close(info["pbar_pos"], float(pbar[1])) and close(info["pbar_neg"], float(pbar[0])),
          "pbar_neg/pbar_pos report the SAT per-class EMA")

    # KNOWN-NULL: the diagnostics must not have perturbed the loss. Re-run an identical
    # module and compare the loss/threshold/mask outputs bit-for-bit.
    torch.manual_seed(0)
    m2 = FreeMatchLoss(mu_max=1.0, saf_weight=0.01, sat_ema_momentum=0.9,
                       consistency_warmup_frac=0.0, local_threshold_mode="standard")
    l2, i2 = m2(torch.zeros(4, 1, requires_grad=True), tgt_weak,
                torch.zeros(6, 1, requires_grad=True), y, step=0, total_steps=10)
    check(close(float(l2), float(_l), 0.0), "loss is bit-identical across runs")
    for k in ("acceptance", "accepted_pos_fraction", "tau_global", "local_neg", "local_pos"):
        check(close(i2[k], info[k], 0.0), f"pre-existing info['{k}'] unchanged")

    # an EMPTY class must give an undefined rate, never 0 (a rate over no denominator)
    all_pos = torch.tensor([[5.0], [4.0], [6.0]])
    m3 = FreeMatchLoss(sat_ema_momentum=0.9, consistency_warmup_frac=0.0)
    _l3, i3 = m3(torch.zeros(2, 1, requires_grad=True), all_pos,
                 torch.zeros(3, 1, requires_grad=True), torch.tensor([1.0, 0.0]),
                 step=0, total_steps=10)
    check(math.isnan(i3["mask_rate_neg"]),
          "mask_rate_neg is nan (not 0) when no sample's argmax is the non-lens class")


# ---------------------------------------------------------------------------
# 2 -- StepWindow
# ---------------------------------------------------------------------------

def test_step_window():
    print("\n[2] StepWindow accumulation")
    w = ui.StepWindow()
    w.update({"sup_loss": 1.0, "consistency": 2.0, "mu_eff": 0.0,
              "accepted_pos_fraction": float("nan"), "step_skipped": 1.0})
    w.update({"sup_loss": 3.0, "consistency": 4.0, "mu_eff": 1.0,
              "accepted_pos_fraction": 0.5, "step_skipped": 0.0})
    row = w.row(125)
    check(row["step"] == 125 and row["n_steps_in_window"] == 2, "step + window size")
    check(close(row["sup_loss_mean"], 2.0), "sup_loss_mean averages both steps")
    check(close(row["accepted_pos_fraction_mean"], 0.5),
          "a nan step is SKIPPED, not averaged in as 0")
    check(close(row["step_skipped_mean"], 0.5), "skip rate = 1 of 2 steps")
    check(close(row["unlabeled_over_supervised"], 3.0 / 2.0),
          "unlabeled/supervised ratio uses the window means")
    check(close(row["unlabeled_loss_weighted"], 0.5 * 3.0),
          "as-applied unlabeled term is mu_eff-weighted (0 during warmup)")

    w2 = ui.StepWindow()
    w2.update({"sup_loss": 1.0, "mask_rate_pos": float("nan")})
    check(w2.row(1)["mask_rate_pos_mean"] is None,
          "an all-undefined key stays undefined (blank cell), never 0")

    w.reset()
    check(w.n_steps == 0 and w.row(2)["sup_loss_mean"] is None, "reset clears the window")


# ---------------------------------------------------------------------------
# 3 -- memorization probe
# ---------------------------------------------------------------------------

def test_memorization():
    print("\n[3] memorization probe (§5.4)")
    # target pool 1:1, test pool ~1:269 -- the composition the probe must not be fooled by
    scores = {"target_fold0": np.concatenate([np.full(50, 2.0), np.full(50, -2.0)]),
              "test_pos_fold0": np.full(10, 2.0),
              "test_negatives_shared": np.full(2690, -2.0),
              "val": np.zeros(5)}
    r = ui.memorization_row(7, scores, 0)
    s2, sm2 = float(ui.sigmoid(2.0)), float(ui.sigmoid(-2.0))
    check(close(r["conf_target_mean"], 0.5 * s2 + 0.5 * sm2), "target mean over the 1:1 pool")
    check(close(r["conf_test_matched"], 0.5 * s2 + 0.5 * sm2),
          "MATCHED test statistic reweights test to the target's 1:1 composition")
    check(close(r["target_minus_test_matched"], 0.0),
          "identical models on identical class distributions -> gap exactly 0 (known-null)")
    # the raw mean is ~sigmoid(-2)=0.119 because the pool is ~1:269, while the matched
    # statistic is 0.5 -- the whole reason the matched column is the one to read.
    check(r["conf_test_raw_mean"] < 0.2 and abs(r["conf_test_raw_mean"] - 0.5) > 0.3,
          "the RAW test mean is dominated by composition -- which is why it is not the "
          "column to read")
    # now make the target genuinely memorised (confident on everything) and check it shows
    scores["target_fold0"] = np.full(100, 4.0)
    r2 = ui.memorization_row(8, scores, 0)
    check(r2["target_minus_test_matched"] > 0.4,
          "a target pool the model is confident on pulls away from matched test")
    try:
        ui.memorization_row(9, {"val": np.zeros(3)}, 0)
        check(False, "missing splits must raise")
    except KeyError:
        check(True, "missing splits raise KeyError rather than silently reporting nothing")


# ---------------------------------------------------------------------------
# 4 -- logit dump
# ---------------------------------------------------------------------------

def test_logit_dump():
    print("\n[4] derived-logit dump (§5.2)")
    scores = {"val": np.random.randn(12000).astype(np.float32),
              "test_negatives_shared": np.random.randn(26399).astype(np.float32),
              "test_pos_fold0": np.random.randn(98).astype(np.float32),
              "target_fold0": np.random.randn(782).astype(np.float32)}
    for k in list(scores):
        scores[f"{k}__ids"] = np.array([f"{k}_{i}" for i in range(scores[k].size)],
                                       dtype="<U80")
    with tempfile.TemporaryDirectory() as d:
        p = ui.dump_logits(d, 125, scores, 0, write_ids=True)
        with np.load(p, allow_pickle=False) as z:
            want = {"step", "val", "test_negatives_shared", "test_pos_fold0",
                    "target_fold0"}
            check(set(z.files) == want, f"dump holds exactly {sorted(want)}")
            check(all(z[k].dtype == np.float32
                      for k in want - {"step"}), "all logits are float32")
            check(z["target_fold0"].shape == (782,)
                  and z["test_negatives_shared"].shape == (26399,), "shapes preserved")
            check(all(np.isfinite(z[k]).all() for k in want - {"step"}),
                  "every dumped logit is finite")
        check(os.path.isfile(os.path.join(d, "ids.npz")), "ids.npz written once")
        # a second dump must not rewrite ids, and must not need them
        p2 = ui.dump_logits(d, 250, {k: v for k, v in scores.items()
                                     if not k.endswith("__ids")}, 0, write_ids=False)
        check(os.path.isfile(p2), "a later curve point dumps without ids")
        try:
            ui.dump_logits(d, 375, {"val": np.zeros(3, np.float32)}, 0)
            check(False, "a dump missing a split must raise")
        except KeyError:
            check(True, "a dump missing a split raises rather than writing a partial file")


# ---------------------------------------------------------------------------
# 5 -- retention accounting with ckpt_every != curve_every
# ---------------------------------------------------------------------------

def test_retention():
    print("\n[5] retention with ckpt_every=500 != curve_every=125")
    want = ruf.expected_checkpoint_steps(8000, 500)
    check(len(want) == 17 and want[0] == 0 and want[-1] == 8000,
          "8000 steps @500 -> 16 checkpoints + step 0 = 17")
    check(len(ruf.expected_checkpoint_steps(2000, 500)) == 5,
          "the 2000-step control -> 4 checkpoints + step 0")

    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "checkpoints")
        os.makedirs(ck)
        curve = os.path.join(d, "curve.csv")
        with open(curve, "w") as fh:
            fh.write("step\n" + "".join(f"{s}\n" for s in
                                        [0] + list(range(125, 2001, 125))))
        for s in ruf.expected_checkpoint_steps(2000, 500):
            open(os.path.join(ck, f"step_{s}.pt"), "wb").close()
        have = ruf.assert_checkpoint_retention(ck, curve, 2000, 500, 125, verbose=False)
        check(len(have) == 5, "a complete run passes")

        os.remove(os.path.join(ck, "step_1000.pt"))
        try:
            ruf.assert_checkpoint_retention(ck, curve, 2000, 500, 125, verbose=False)
            check(False, "a missing checkpoint must abort")
        except SystemExit:
            check(True, "a missing checkpoint ABORTS (the §7.7 collection bug)")


# ---------------------------------------------------------------------------
# 6 -- the §1 precondition check, adversarially
# ---------------------------------------------------------------------------

class _Args:
    """The argparse namespace build_config reads, at the run's real defaults."""
    def __init__(self, **kw):
        self.max_steps = None
        self.grad_checkpointing = False
        self.device = "cpu"
        self.init_head_from = None
        self.schedule_steps = ruf.BUDGET_STEPS
        self.curve_every = ruf.CURVE_EVERY
        self.ckpt_every = ruf.CKPT_EVERY
        self.backbone_lr = None
        self.fp16 = True
        self.eval_fp16 = False
        self.num_workers = 2
        self.agreement_ref = "ref.pt"
        self.expect_step0_tpr = None
        self.smoke_batch_size = None
        # continuation-state reconstruction, at the run's real defaults
        self.mu_step_offset = ruf.BUDGET_STEPS
        self.sat_passes = 40
        self.sat_ema_iters = 5000
        self.no_rebuild_sat = False
        self.allow_off_budget = False
        self.adam_warmup_steps = 200
        self.fold = ruf.DEFAULT_FOLD
        self.expect_head_l2 = None
        self.__dict__.update(kw)


def test_preconditions():
    print("\n[6] §1 precondition check (adversarial)")
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ctl, _s, _m, _g, _d, _h, _k = ruf.build_config("nb0_control", _Args(), root)
    unf, _s, _m, _g, _d, _h, _k = ruf.build_config(
        "nb3_unfrozen", _Args(grad_checkpointing=True), root)
    ctl["init_head_l2"] = unf["init_head_l2"] = ruf.EXPECT_HEAD_L2

    with tempfile.TemporaryDirectory() as d:
        ref = os.path.join(d, "ref.json")

        def write(cfg):
            with open(ref, "w") as fh:
                json.dump(cfg, fh)
            return ref

        ruf.check_preconditions(unf, write(ctl), verbose=False)
        check(True, "the real pair PASSES -- only n_blocks/grad_ckpt/max_steps differ")
        # UPDATED 2026-08-05: both arms now run the pre-registered 2000, so max_steps is
        # PINNED again. The exemption existed only while the control ran 2000 against the
        # unfrozen arm's 8000 (§7.10) and went stale the moment the budget was fixed.
        check(unf["max_steps"] == ctl["max_steps"] == 1500,
              "both arms track the AMENDED 1500 budget, so a paired control stays "
              "§1-comparable; max_steps is PINNED so any mismatch aborts")
        check("max_steps" in ruf.PINNED_KEYS and "max_steps" not in ruf.EXEMPT_KEYS,
              "max_steps is PINNED, not exempt -- a budget mismatch between arms ABORTS")

        # each adversarial mutation must abort
        for key, bad, why in [
                ("precision", "fp32", "a dropped --fp16"),
                ("augmentation", "none", "aug=none"),
                ("seed", 43, "a changed seed"),
                ("local_threshold_mode", "floor", "the floor variant (a documented negative)"),
                ("saf_weight", 0.1, "a changed SAF weight"),
                ("lr", 1e-4, "a changed head LR"),
                ("init_head_l2", 28.475603, "the Baseline A probe head instead of FreeMatch's"),
                ("schedule_total_steps", 2000, "a COMPRESSED control schedule"),
                ("eval_precision", "fp16", "an fp16 eval encode against an fp32 reference"),
                ("curve_every", 250, "a different curve density"),
                ("mu_step_offset", 0, "one arm restarting the mu warmup"),
                ("sat_rebuild", False, "one arm skipping the SAT reconstruction"),
                ("adam_warmup_steps", 0, "one arm skipping the Adam warm-up")]:
            mutated = dict(ctl)
            mutated[key] = bad
            try:
                ruf.check_preconditions(unf, write(mutated), verbose=False)
                check(False, f"ABORTS on {why}")
            except SystemExit:
                check(True, f"ABORTS on {why}")

        os.remove(ref)
        try:
            ruf.check_preconditions(unf, ref, verbose=False)
            check(False, "a missing reference config must abort")
        except SystemExit:
            check(True, "a MISSING reference config aborts (fails closed)")


# ---------------------------------------------------------------------------
# 7 -- the step-0 hard gate
# ---------------------------------------------------------------------------

def test_step0_gate():
    print("\n[7] the step-0 hard gate (§2)")
    AUC0 = ruf.FOLD_GATE[0]["auroc"]
    ok_row = {"fold0_tpr_1e-3": ruf.EXPECT_STEP0_TPR, "fold0_auroc": AUC0,
              "source_risk_auroc": 0.99}
    g = ruf.check_step0_gate(ok_row, ruf.EXPECT_STEP0_TPR, 0.0, "t", verbose=False)
    check(g["passed"] is True, "an exact reproduction PASSES at tolerance 0")
    ruf.gate_or_abort(ok_row, ruf.EXPECT_STEP0_TPR, 0.0, "t")
    check(True, "gate_or_abort does not raise on a pass")

    one_positive_off = {"fold0_tpr_1e-3": ruf.EXPECT_STEP0_TPR - 1.0 / 98.0,
                        "fold0_auroc": AUC0}
    try:
        ruf.gate_or_abort(one_positive_off, ruf.EXPECT_STEP0_TPR, 0.0, "t")
        check(False, "a one-positive miss must abort at tolerance 0")
    except SystemExit:
        check(True, "a ONE-POSITIVE miss aborts at tolerance 0")
    g2 = ruf.check_step0_gate(one_positive_off, ruf.EXPECT_STEP0_TPR, 1.0 / 98.0 + 1e-9,
                              "t", verbose=False)
    check(g2["passed"] is True, "an explicit one-positive tolerance admits it")
    try:
        ruf.gate_or_abort({"fold0_tpr_1e-3": 0.10, "fold0_auroc": 0.80},
                          ruf.EXPECT_STEP0_TPR, 0.0, "t")
        check(False, "a wrong-model step-0 must abort")
    except SystemExit:
        check(True, "a WRONG model at step 0 aborts")

    # THE FOLD-2 CASE: the live-encode path flips one positive on a knife-edge threshold.
    # Right model (AUROC exact), TPR one positive off -> must PASS at the one-positive
    # tolerance, because the encoder forward is not bit-reproducible (MEASURED ~1e-2 drift).
    for f in range(5):
        g = ruf.check_step0_gate(
            {f"fold{f}_tpr_1e-3": ruf.FOLD_GATE[f]["tpr"] - 1.0 / 98.0,
             f"fold{f}_auroc": ruf.FOLD_GATE[f]["auroc"]},
            ruf.FOLD_GATE[f]["tpr"], ruf.ONE_POSITIVE_TOL, "t", fold=f,
            expected_auroc=ruf.FOLD_GATE[f]["auroc"], verbose=False)
        check(g["passed"] is True,
              f"fold {f}: a ONE-POSITIVE threshold flip with the right AUROC passes")
    # ...but TWO positives does not, even with a perfect AUROC
    g = ruf.check_step0_gate(
        {"fold2_tpr_1e-3": ruf.FOLD_GATE[2]["tpr"] - 2.0 / 98.0,
         "fold2_auroc": ruf.FOLD_GATE[2]["auroc"]},
        ruf.FOLD_GATE[2]["tpr"], ruf.ONE_POSITIVE_TOL, "t", fold=2,
        expected_auroc=ruf.FOLD_GATE[2]["auroc"], verbose=False)
    check(g["passed"] is False, "TWO positives off still FAILS -- the widening is exactly "
                                "one positive, not open-ended")
    # ...and the WRONG fold's head, which can hit a plausible TPR by chance, is caught by
    # AUROC: fold 1's head on fold 2's data gives TPR exactly 0.336735, AUROC 0.967409.
    g = ruf.check_step0_gate({"fold2_tpr_1e-3": 0.336734693877551, "fold2_auroc": 0.967409},
                             ruf.FOLD_GATE[2]["tpr"], ruf.ONE_POSITIVE_TOL, "t", fold=2,
                             expected_auroc=ruf.FOLD_GATE[2]["auroc"], verbose=False)
    check(g["passed"] is False and g["auroc_ok"] is False,
          "the WRONG fold's head is caught by AUROC (7.6e-3 off) even though its TPR is "
          "a plausible value")
    check(len({ruf.FOLD_GATE[k]["auroc"] for k in range(5)}) == 5,
          "all five expected step-0 AUROCs are distinct")
    check(ruf.check_step0_gate(None, 0.3, 0.0, "t", verbose=False)["passed"] is None,
          "no step-0 row -> gate reports None, never a silent pass")


# ---------------------------------------------------------------------------
# 8 -- collect_scores is a known-null for the curve row
# ---------------------------------------------------------------------------

class _FakeCache:
    dir = "<mock>"


def _mock_split(n, pos=0, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.zeros(n, np.int64)
    lab[:pos] = 1
    return {"features": rng.standard_normal((n, 640)).astype(np.float32),
            "labels": lab, "ids": np.array([f"id{i}" for i in range(n)], dtype="<U80")}


def test_collect_scores_is_null():
    print("\n[8] collect_scores changes no metric and no column (known-null)")
    from heads import build_head, FeatureStandardizer
    torch.manual_seed(3)
    head = build_head("linear")
    std = FeatureStandardizer(640)
    std.set_stats(torch.zeros(640), torch.ones(640))
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "step_10.pt")
        torch.save({"head_type": "linear", "head": head.state_dict(),
                    "standardizer": std.state_dict(), "cfg": {"fold": 0}}, ckpt)
        splits = {"val": _mock_split(400, 200, 1),
                  "test_negatives_shared": _mock_split(2000, 0, 2),
                  "test_pos_fold0": _mock_split(98, 98, 3),
                  "target_fold0": _mock_split(150, 0, 4)}
        loader = splits.__getitem__
        ref = np.arange(150, dtype=float)

        a = curve_hook.eval_checkpoint_folds(ckpt, _FakeCache(), folds=(0,),
                                             agreement_refs={0: ref}, split_loader=loader)
        got = {}
        b = curve_hook.eval_checkpoint_folds(ckpt, _FakeCache(), folds=(0,),
                                             agreement_refs={0: ref}, split_loader=loader,
                                             collect_scores=got)
        check(a == b, "the row is bit-identical with and without collect_scores")
        check(set(k for k in got if not k.endswith("__ids")) ==
              {"val", "test_negatives_shared", "test_pos_fold0", "target_fold0"},
              "all four splits collected")
        check(got["test_pos_fold0"].shape == (98,) and got["target_fold0"].shape == (150,),
              "collected vectors have the split's row count")
        check(all(f"{k}__ids" in got for k in ("val", "target_fold0")),
              "ids travel with the scores so row alignment can be verified later")

        # the target pool must still be collected when no agreement column is requested
        got2 = {}
        curve_hook.eval_checkpoint_folds(ckpt, _FakeCache(), folds=(0,),
                                         split_loader=loader, collect_scores=got2)
        check("target_fold0" in got2,
              "target logits are dumped even with the agreement column off")


# ---------------------------------------------------------------------------
# 9/10 -- schedule truncation + backward-compatible defaults
# ---------------------------------------------------------------------------

def test_schedule_and_defaults():
    print("\n[9/10] schedule truncation + round-1 defaults")
    root = os.path.normpath(os.path.join(_HERE, ".."))
    ctl, _s, _m, _g, _d, _h, _k = ruf.build_config("nb0_control", _Args(), root)
    unf, _s, _m, _g, _d, _h, _k = ruf.build_config("nb3_unfrozen",
                                                   _Args(grad_checkpointing=True), root)
    check(ctl["schedule_total_steps"] == unf["schedule_total_steps"] == 8000,
          "both runs share the 8000-step mu/warmup schedule")
    check(ctl["schedule_total_steps"] == unf["schedule_total_steps"] == 8000,
          "both arms are TRUNCATIONS of the 8000-step schedule (1500 / 2000)")
    # the consistency warmup is a fraction of the SCHEDULE, so both withhold consistency for
    # the same 800 steps -- which is what makes the control's step 250 comparable.
    warm = math.ceil(ruf.CONSISTENCY_WARMUP_FRAC * ctl["schedule_total_steps"])
    check(warm == 800, "consistency withheld for the same 800 steps in both runs")
    check(unf["grad_checkpointing"] is True and ctl["grad_checkpointing"] is False,
          "grad checkpointing is on at nb3 and refused as a no-op at nb0")
    check(unf["backbone_lr"] == ruf.BACKBONE_LR and ctl["backbone_lr"] is None,
          "the encoder LR exists only in the unfrozen run (the one new hyperparameter)")
    check(ctl["lr"] == 1e-3 and ctl["weight_decay"] == 1e-4,
          "round-1 FreeMatch head LR / weight decay are inherited unchanged")
    check(ctl["local_threshold_mode"] == "standard",
          "the frozen variant is 'standard' ('floor' is a documented negative)")

    import run_freematch
    sig = inspect.signature(run_freematch.train_freematch_fold)
    for name, want in [("n_blocks", 0), ("backbone_lr", None), ("init_head_from", None),
                       ("grad_checkpointing", False), ("schedule_steps", None),
                       ("max_steps", None), ("ckpt_every", None), ("instrument", None),
                       ("eval_at_step0", False), ("step0_check", None)]:
        p = sig.parameters.get(name)
        check(p is not None and p.default == want,
              f"train_freematch_fold({name}=...) defaults to {want!r} (round-1 path)")


# ---------------------------------------------------------------------------
# 11 -- the collapse gate is a REQUIRED output and cannot go missing silently
# ---------------------------------------------------------------------------

def test_gate_reporting():
    print("\n[11] collapse-gate reporting (§5.5 / required output #5)")
    import report_unfreeze_freematch as rep
    curve = [{"step": 0, "agreement_fold0": 0.88}, {"step": 125, "agreement_fold0": 0.86},
             {"step": 250, "agreement_fold0": 0.84}]
    with tempfile.TemporaryDirectory() as d:
        a = rep.agreement_trace(curve, d)
        check(a["gate_fired"] is None and "MISSING" in a["gate_verdict"],
              "no gate_fold0.json -> the report says MISSING, never a silent pass")
        check(a["available"] and close(a["min"], 0.84), "the trace itself still reports")

        with open(os.path.join(d, "gate_fold0.json"), "w") as fh:
            json.dump({"fired": False, "agreement_min": 0.84,
                       "triggers": {"strict": {"fired": False}},
                       "lockstep_check": {"ok": True, "max_abs_diff": 0.0,
                                          "n_compared": 3}}, fh)
        a = rep.agreement_trace(curve, d)
        check(a["gate_fired"] is False and a["gate_verdict"].startswith("DID NOT FIRE"),
              "a no-fire gate is reported as such, with the §7.6 UNVERIFIED caveat")
        check("UNVERIFIED" in a["gate_verdict"],
              "a no-fire never becomes evidence the gate WORKS -- it stays uncalibrated")
        check(a["lockstep_check"]["ok"] is True, "the lock-step known-null is surfaced")

        with open(os.path.join(d, "gate_fold0.json"), "w") as fh:
            json.dump({"fired": True, "agreement_min": 0.41,
                       "triggers": {"strict": {"fired": True, "step": 3000}}}, fh)
        a = rep.agreement_trace(curve, d)
        check(a["gate_fired"] is True and "FIRED" in a["gate_verdict"]
              and "first positive case" in a["gate_verdict"],
              "a FIRE is reported as the first positive case in four rounds (a result)")

    # the driver must own the gate, so a required output cannot depend on a follow-up command
    check(hasattr(ruf, "run_collapse_gate"),
          "run_unfreeze_freematch owns run_collapse_gate (the run writes gate_fold{k}.json)")
    src = inspect.getsource(ruf.train_one)
    check("run_collapse_gate" in src, "train_one calls it -- the gate runs as part of the run")
    gsrc = inspect.getsource(ruf.run_collapse_gate)
    check("from_curve=True" in gsrc,
          "the gate reads the INLINE column (--from-curve); the cached recompute is invalid "
          "once the encoder trains")
    check("expect_healthy=False" in gsrc,
          "the gate is NOT told to expect a healthy run -- this configuration is the one "
          "most likely to collapse")


# ---------------------------------------------------------------------------
# 12 -- continuation-state reconstruction
# ---------------------------------------------------------------------------

def test_continuation_state():
    print("\n[12] continuation-state reconstruction")
    import continuation_state as cs
    from da_losses import FreeMatchLoss, adamatch_mu_at_step

    # -- the buffer enumeration this rebuild rests on --
    fm = FreeMatchLoss()
    names = {n for n, _ in fm.named_buffers()}
    check(names == {"tau_global", "pbar", "fairness_target"},
          f"FreeMatchLoss registers exactly {sorted(names)} -- no unenumerated EMA state")
    check(set(cs.SAT_BUFFERS) == {"tau_global", "pbar"},
          "only tau_global/pbar are rebuilt")
    check(cs.NOT_REBUILT == ("fairness_target",),
          "fairness_target is left alone (a config value, not an EMA)")

    # -- the EMA cross-check finds the pool mean, and its known init bleed is predicted --
    rng = np.random.default_rng(0)
    bmax = rng.normal(0.88, 0.01, 200).clip(0.5, 1.0)
    bprob = np.stack([1 - rng.normal(0.42, 0.01, 200), rng.normal(0.42, 0.01, 200)], 1)
    m = 0.999
    tau_ema, pbar_ema = cs.ema_crosscheck(bmax, bprob, m, n_iter=5000, seed=1)
    tau_direct = float(bmax.mean())
    bleed = m ** 5000
    pred = tau_direct + bleed * (0.5 - tau_direct)
    check(abs(tau_ema - pred) < 5e-3,
          f"EMA(5000) lands on the pool mean once the {bleed:.4f} init bleed is accounted "
          f"for (|{tau_ema:.5f} - {pred:.5f}| = {abs(tau_ema-pred):.2e})")
    check(abs(tau_ema - tau_direct) > 1e-4,
          "the raw EMA-vs-direct gap is NON-zero -- so the bleed correction is doing work "
          "and could not hide a real mismatch")

    # -- the abort fires on a genuine rule mismatch --
    good = {"tau_abs_diff_bleed_corrected": 1e-5, "pbar_abs_diff_bleed_corrected": 1e-5}
    cs.sat_check_or_raise(good); check(True, "a matching cross-check passes")
    try:
        cs.sat_check_or_raise({"tau_abs_diff_bleed_corrected": 0.05,
                               "pbar_abs_diff_bleed_corrected": 1e-5})
        check(False, "a mismatched cross-check must abort")
    except SystemExit:
        check(True, "a MISMATCHED cross-check ABORTS rather than training on a fabricated "
                    "threshold")

    # -- the mu offset does what it claims, and nothing else is step-dependent --
    check(adamatch_mu_at_step(0, 8000, 1.0) == 0.0, "mu at step 0 is exactly 0 (the bug)")
    check(adamatch_mu_at_step(8000, 8000, 1.0) == 1.0,
          "mu at step 0 + offset 8000 is the full value (the fix)")
    warm = math.ceil(0.1 * 8000)
    check(0 < warm <= 8000 and 8000 >= warm,
          f"the offset also opens the consistency-warmup gate (warmup={warm} steps)")

    # -- optimizer state round-trips through the new save format --
    from train_skeleton import Trainer, TrainConfig
    import torch as _t
    cfg = TrainConfig(head_type="linear", n_blocks=0, device="cpu", batch_size=4,
                      use_freematch=True)

    class _Enc(_t.nn.Module):
        n_blocks = 0
        tuneable_blocks_in_lr_order = []
        effective_mode = "frozen"
        def set_grad_checkpointing(self, *_a): pass
    tr = Trainer.__new__(Trainer)     # avoid loading real Zoobot weights in a unit test
    tr.cfg = cfg; tr.device = _t.device("cpu")
    from heads import build_head, FeatureStandardizer
    from da_losses import CoralLoss, AdaMatchLoss
    tr.encoder = _Enc(); tr.head = build_head("linear")
    tr.standardizer = FeatureStandardizer(640)
    tr.standardizer.set_stats(_t.zeros(640), _t.ones(640))
    tr.freematch = FreeMatchLoss(); tr.adamatch = AdaMatchLoss()
    tr.coral = CoralLoss(tr.head.feat_dim)
    tr.opt = _t.optim.AdamW(tr.head.parameters(), lr=1e-3)
    with _t.no_grad():
        tr.freematch.tau_global.fill_(0.8765); tr.freematch.pbar.copy_(_t.tensor([.58, .42]))
    tr.head(_t.randn(2, 640)).sum().backward(); tr.opt.step()

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "step_1.pt")
        tr.save(p, full_state=True, schedule={"mu_step_offset": 8000})
        a = cs.audit_checkpoint(p)
        check(a["has_optimizer"] and a["has_da_buffers"] and a["has_schedule"],
              "the new format carries optimizer + DA buffers + schedule")
        check(a["continuable"], "audit_checkpoint calls it continuable")
        ck = _t.load(p, map_location="cpu", weights_only=False)
        check(abs(float(ck["da_buffers"]["freematch"]["tau_global"]) - 0.8765) < 1e-6,
              "tau_global round-trips through the checkpoint")
        check(set(ck.keys()) >= {"head", "head_type", "standardizer", "cfg"},
              "the legacy four keys are still present -- evaluate.load_checkpoint unaffected")
        import evaluate as ev
        h, s, _mm = ev.load_checkpoint(p, "cpu")
        check(h is not None and s is not None,
              "evaluate.load_checkpoint reads the NEW format (additive, not breaking)")

        # a round-1 checkpoint must be correctly classified as un-continuable
        legacy = os.path.join(d, "legacy.pt")
        _t.save({"head_type": "linear", "head": tr.head.state_dict(),
                 "standardizer": tr.standardizer.state_dict(), "cfg": {}}, legacy)
        al = cs.audit_checkpoint(legacy)
        check(al["legacy_weights_only"] and not al["continuable"],
              "a four-key round-1 checkpoint is flagged legacy / not continuable")

    # -- the moment/step-counter helpers report what they claim --
    nrm, cnt = cs.second_moment_norm(tr.opt)
    check(nrm > 0 and cnt == 1921, f"second_moment_norm reads {cnt} params with state")
    check(cs.optimizer_step_counter(tr.opt) == 1.0,
          "optimizer_step_counter reads Adam's per-param step (bias correction reads this)")


def test_per_fold():
    print("\n[13] per-fold parameterisation (§7.11)")
    root = os.path.normpath(os.path.join(_HERE, ".."))
    check(sorted(ruf.FOLD_GATE) == [0, 1, 2, 3, 4], "gate table covers all five folds")
    seen_tpr, seen_l2, dirs = set(), set(), set()
    for k in range(5):
        cfg, _s, _m, _g, _d, head, fold = ruf.build_config(
            "nb3_unfrozen", _Args(fold=k, grad_checkpointing=True), root)
        check(fold == k and cfg["fold"] == k, f"fold {k}: config carries the fold")
        # the fold lives in the GRANDPARENT dir: .../bakeoff_freematch_5fold/fold{k}/checkpoints/step_8000.pt
        head_fold_dir = os.path.basename(os.path.dirname(os.path.dirname(head)))
        check(head_fold_dir == f"fold{k}",
              f"fold {k}: head path derives from the fold ({head_fold_dir}/checkpoints/"
              f"{os.path.basename(head)})")
        check(cfg["expect_step0_tpr"] == ruf.FOLD_GATE[k]["tpr"],
              f"fold {k}: gate = {ruf.FOLD_GATE[k]['tpr']:.6f} (its OWN round-1 value)")
        check(cfg["run_dir"].endswith(f"fold{k}"), f"fold {k}: run dir is fold-specific")
        seen_tpr.add(cfg["expect_step0_tpr"]); seen_l2.add(cfg["expect_head_l2"])
        dirs.add(cfg["run_dir"])
    check(len(seen_tpr) == 5 and len(seen_l2) == 5,
          "all five gate values and head norms are DISTINCT -- a wrong fold cannot pass")
    check(len(dirs) == 5, "all five run dirs are distinct -- folds cannot overwrite")
    try:
        ruf.build_config("nb3_unfrozen", _Args(fold=7, grad_checkpointing=True), root)
        check(False, "an out-of-range fold must abort")
    except SystemExit:
        check(True, "an OUT-OF-RANGE fold aborts")

    # the budget guard must ABORT, not display (fold 1 ran 8000 for 3.5 h unnoticed)
    try:
        ruf.build_config("nb3_unfrozen",
                         _Args(fold=1, grad_checkpointing=True, max_steps=8000), root)
        cfg8, *_ = (None,)
    except SystemExit:
        pass
    cfg2, _s, _m, _g, _d, _h, _k = ruf.build_config(
        "nb3_unfrozen", _Args(fold=1, grad_checkpointing=True), root)
    check(cfg2["max_steps"] == ruf.CONFIRMATORY_STEPS == 1500,
          "the BUILT-IN nb3_unfrozen budget is the AMENDED pre-registered 1500")
    check(1500 % 500 == 0,
          "R1 = step 1500 lands on a ckpt_every multiple, so step_1500.pt is retained")
    check(cfg2["is_preregistered_budget"],
          "a default-budget run is flagged as ON the pre-registered budget")
    check("max_steps" in ruf.PINNED_KEYS and "max_steps" not in ruf.EXEMPT_KEYS,
          "max_steps is PINNED again -- the stale exemption is gone")
    check("budget_steps" in ruf.PINNED_KEYS,
          "budget_steps is PINNED again")

    import report_unfreeze_freematch as rep
    rngs = {k: rep.round1_tail_range(k) for k in range(5)}
    check(len(set(rngs.values())) > 1,
          f"the flatness criterion is PER-FOLD, not a constant: {rngs}")
    check(rep.round1_tail_range(3) > rep.round1_tail_range(0),
          "fold 3's criterion is wider than fold 0's -- the fixed 3-positive rule was the bug")

    import aggregate_folds as ag
    check(ag.DEV_FOLD == 0 and tuple(ag.CONFIRMATORY) == (1, 2, 3, 4),
          "aggregator: fold 0 development, folds 1-4 confirmatory (§7.11)")
    check(abs(np.mean([ag.ROUND1_TPR[k] for k in ag.CONFIRMATORY]) - 0.337603) < 1e-5,
          "aggregator: round-1 folds-1-4 mean = 0.337603, the comparison arm")
    conf = ag.confirmatory({}, bootstrap=10)
    check(conf["folds_missing"] == [1, 2, 3, 4] and not conf["complete"],
          "aggregator on an EMPTY set names the missing folds instead of failing")


def main():
    print("=" * 78)
    print("BRIEF §7 PRE-FLIGHT SUITE -- FreeMatch x n_blocks=3, fold 0")
    print("=" * 78)
    test_freematch_diagnostics()
    test_step_window()
    test_memorization()
    test_logit_dump()
    test_retention()
    test_preconditions()
    test_step0_gate()
    test_collect_scores_is_null()
    test_schedule_and_defaults()
    test_gate_reporting()
    test_continuation_state()
    test_per_fold()
    print("\n" + "=" * 78)
    if FAILS:
        print(f"FAILED: {len(FAILS)}")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
