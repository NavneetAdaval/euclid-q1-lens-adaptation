"""
bnm_round6.py
=============
ROUND 6 -- **BNM as the minimal-target-signal control**: the installer, the collapse
monitor, and the two rules the brief leaves to be implemented (the gamma grid and the
budget). The loss itself lives in ``da_losses`` (``BNMLoss`` / ``bnm_loss`` /
``bnm_nuclear_norm`` / ``bnm_analytic_reference``) and Stage 0a is already recorded in
``runs/bnm_r6_stage0a.json`` -- neither is re-derived here.

WHY A HOOK AND NOT A DIRECT INSTALL. ``train_skeleton.Trainer.train_step``'s coral branch
calls ``self.coral(src_adapter, tgt_adapter)`` -- two ADAPTER-FEATURE tensors. BNM is
target-only and consumes LOGITS. :class:`BNMHook` is the adapter between the two: it
ignores the source tensor entirely and pushes the target adapter features through the
head's own classifier to get the logits BNM needs. That is not a reimplementation of the
head -- ``ClassifierHead.forward_with_features`` returns ``(classifier(f), f)``, so
``classifier(tgt_adapter)`` is by construction the same tensor the head would have
produced for the target batch.

THE HEAD REFERENCE IS DELIBERATELY NOT A SUBMODULE. ``nn.Module.__setattr__`` registers
any Module assigned to an attribute, so a plain ``self.head = head`` would make every head
parameter show up a second time in ``trainer.coral.parameters()`` and in the ``da_buffers``
payload ``Trainer.save`` writes. It is stored through ``object.__setattr__`` instead, so
this module owns NO parameters and NO buffers -- which is also what keeps BNM exempt from
7.10's weights-only-continuation problem.

WHAT THIS ARM IS. Not a fourth entry in the nb3 ranking table -- 7.16 bounds any true
difference between the existing arms at +/-0.17 and the within-run checkpoint sd on one
fold is 2.8-5.1x the DANN-CORAL difference. It is the FLOOR TEST on 7.22: every arm that
produced the +0.26..+0.28 is elaborate and needs source and target together in the loss;
BNM is the minimum, target predictions only. If it also buys ~+0.28 the machinery in the
other three arms is decoration; if it does not, the floor is located.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from da_losses import BNM_EPS, bnm_loss, bnm_nuclear_norm, binary_prob2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
STAGE0A = os.path.join(_REPO, "runs", "bnm_r6_stage0a.json")


# ---------------------------------------------------------------------------
# Stage-0a anchors -- READ FROM THE ARTIFACT, not retyped
# ---------------------------------------------------------------------------

def stage0a_anchors(path: str = STAGE0A) -> dict:
    """The four numbers Stages 2-4 depend on, loaded from Stage 0a's own JSON.

    Retyping them here would create a second source of truth for values that a rule
    reads, which is the failure 7.15 records for the CORAL gamma table. If the file is
    missing this raises rather than falling back to a literal.
    """
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    a, b = d["part_a"], d["part_b"]
    if b.get("SKIPPED"):
        raise SystemExit(f"{path}: part B was skipped; the empirical anchors do not exist")
    dec = b["decomposition"]
    return {
        "balanced_confident": a["states"]["balanced_confident"]["nuclear_over_min_bc"],
        "collapsed_confident": a["states"]["collapsed_confident"]["nuclear_over_min_bc"],
        "maximally_uncertain": a["states"]["maximally_uncertain"]["nuclear_over_min_bc"],
        "init": dec["init"],
        "sharpening_ceiling": dec["confident_at_current_split"],
        "balanced_max": dec["balanced_confident_max"],
        "confidence_share": dec["confidence_share"],
        "diversity_share": dec["diversity_share"],
        "init_pred_pos_frac": b["at_init"]["pred_pos_frac_mean"],
        "source": os.path.relpath(path, _REPO).replace("\\", "/"),
    }


# ---------------------------------------------------------------------------
# the collapse monitor -- DRIFT FROM STEP 0, never an absolute band
# ---------------------------------------------------------------------------
#
# The previously proposed 0.25-0.75 ABSOLUTE band fires at step 0: Baseline A's head
# predicts positive on 0.2292 of the target before training does anything (7.20). That
# would have been the third detector in this project to fire on its own starting state,
# after round 5's abort3b and condition 4. The band below is expressed as drift and is
# ASYMMETRIC, because the two directions are not symmetric in what they mean:
#
#   DOWNWARD from 0.2292 heads directly to the all-negative degenerate solution, and
#   there is only 0.2292 of room before it arrives -- so the floor is a drift bound.
#   UPWARD is the PREDICTED HEALTHY DIRECTION (7.20 predicts drift toward ~0.5, the
#   target's true ~1:1 balance), so a ceiling tight enough to be a real detector could
#   fire on legitimate behaviour. That is resolved by STATUS, not by loosening -- see
#   the next block.
#
# THE CEILING IS ADVISORY-ONLY, AND 0.75 RATHER THAN 0.90. A ceiling that can only
# trigger on catastrophe is a formality, not a detector: BNM draws ~70% of its pull from
# confidence (Stage 0a) and runs on the 782 images 7.12 showed get memorised, so the
# realistic failure is a slide toward all-positive that would be well advanced long before
# 0.90. But it cannot simply be tightened either -- at 0.75 it sits close enough to 7.20's
# predicted ~0.5 endpoint that legitimate behaviour could reach it.
#
# So it is not resolved with a threshold. It is resolved the way round 5 resolved abort3b:
# a condition that might fire on legitimate behaviour goes ADVISORY rather than being
# loosened until it cannot fire. The ceiling LOGS and is read at analysis time against the
# healthy-direction prediction; it never gates anything, and in particular it is NOT part
# of Stage 2's gamma selection criterion. The FLOOR does gate selection -- it is a genuine
# degeneracy bound that cannot fire on the prediction, because the prediction moves away
# from it.
#
# Nothing here can abort a run in any case: advisory, never aborting, on every arm and
# every fold.
COLLAPSE_MAX_DOWNWARD_DRIFT = 0.20      # relative to the step-0 value; GATES selection
COLLAPSE_ABSOLUTE_CEILING = 0.75        # the all-positive direction; ADVISORY ONLY


def collapse_band(step0_pred_pos_frac: float) -> dict:
    """The declared band, its two halves' different STATUS, and its step-0 evaluation.

    ``fires_at_step0`` MUST be False for any band this project declares -- the standing
    rule earned three times over (round 5's abort3b, condition 4, and the withdrawn
    0.25-0.75 absolute band). It is returned rather than asserted so a caller can print
    it into a pre-registration.
    """
    lo = float(step0_pred_pos_frac) - COLLAPSE_MAX_DOWNWARD_DRIFT
    hi = COLLAPSE_ABSOLUTE_CEILING
    v = float(step0_pred_pos_frac)
    return {
        "form": "drift-from-step-0 floor (gating) + absolute ceiling (advisory only)",
        "step0_value": v,
        "floor": lo, "ceiling": hi,
        "max_downward_drift": COLLAPSE_MAX_DOWNWARD_DRIFT,
        "floor_status": "GATES Stage 2 gamma selection",
        "ceiling_status": ("ADVISORY ONLY -- logged every checkpoint, read at analysis "
                           "time against 7.20's healthy-direction prediction, never a "
                           "gate and never a selection criterion"),
        "fires_at_step0": not (lo <= v <= hi),
        "floor_fires_at_step0": v < lo,
        "ceiling_fires_at_step0": v > hi,
        "rationale": (
            "Downward from the init heads to the all-negative solution with only the "
            "init's own value of room, so the floor is a DRIFT bound and it gates. "
            "Upward is the PREDICTED healthy direction (7.20 predicts drift toward ~0.5), "
            "so a ceiling tight enough to be a real detector could fire on legitimate "
            "behaviour -- and one loose enough that it cannot (0.90) detects only "
            "catastrophe. 0.75 is deliberately in the useful range and therefore "
            "ADVISORY, which is round 5's abort3b arrangement generalised: a condition "
            "that might fire on legitimate behaviour goes advisory rather than being "
            "loosened until it cannot fire."),
        "advisory_overall": True,
    }


def collapse_check(pred_pos_frac, band: dict) -> dict:
    """Evaluate a reading. The floor and the ceiling are reported SEPARATELY.

    ``gate_ok`` is what Stage 2's selection criterion reads -- the floor alone.
    ``advisory_ceiling_ok`` is logged and interpreted, never acted on automatically.
    """
    v = float(pred_pos_frac)
    floor_ok = v >= band["floor"]
    ceil_ok = v <= band["ceiling"]
    return {"value": v,
            "gate_ok": bool(floor_ok),
            "advisory_ceiling_ok": bool(ceil_ok),
            "inside": bool(floor_ok and ceil_ok),
            "drift_from_step0": v - band["step0_value"]}


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------

class BNMHook(nn.Module):
    """Adapts BNM's target-only objective to the Trainer's two-argument coral hook.

    ``forward(src_adapter, tgt_adapter)`` ignores ``src_adapter`` -- deliberately, and
    that is the whole point of the arm: no source tensor enters this loss, so there is no
    domain comparison in it at all. Returns the RAW negated normalised nuclear norm; the
    Trainer multiplies by ``cfg.coral_weight``, which is gamma.
    """

    def __init__(self, head, gamma: float):
        super().__init__()
        # NOT a submodule -- see the module docstring.
        object.__setattr__(self, "_head", head)
        self.gamma = float(gamma)
        self.last_stats = {}

    @property
    def head(self):
        return object.__getattribute__(self, "_head")

    def target_logits(self, tgt_adapter: torch.Tensor) -> torch.Tensor:
        return self.head.classifier(tgt_adapter)

    def forward(self, src_adapter: torch.Tensor, tgt_adapter: torch.Tensor):
        if tgt_adapter.dim() != 2:
            raise ValueError(f"BNM expects 2-D target adapter features, got "
                             f"{tuple(tgt_adapter.shape)}")
        if self.training and torch.is_grad_enabled():
            assert tgt_adapter.requires_grad, (
                "BNMHook expected grad-carrying TARGET adapter features; got a detached "
                "tensor -- is it wired to the frozen encoder features by mistake?")

        logits = self.target_logits(tgt_adapter)
        value = bnm_loss(logits)

        with torch.no_grad():
            prob2 = binary_prob2(logits)
            p_pos = prob2[:, 1]
            # float32: BNM_EPS (1e-12) is below fp16's smallest subnormal, so on
            # an fp16 tensor the clamp rounds back to 0 and 0*log(0) is nan.
            _p32 = prob2.float()
            ent = -(_p32.clamp_min(BNM_EPS).log() * _p32).sum(1).mean()
            self.last_stats = {
                "bnm_nuclear_norm": float(bnm_nuclear_norm(prob2)),
                "bnm_target_entropy": float(ent),
                "bnm_pred_pos_frac": float((p_pos > 0.5).float().mean()),
                "bnm_mean_p_pos": float(p_pos.mean()),
                "bnm_raw_loss": float(value.detach()),
            }
        return value

    def extra_repr(self) -> str:
        return (f"gamma={self.gamma}, target_only=True, source_in_loss=False, "
                f"params=0, buffers=0, ramp=None")


def install_bnm(trainer, *, gamma: float):
    """Install BNM into the Trainer's ``coral`` hook slot and set gamma.

    Refuses to overwrite a real CORAL or DANN module for the same reason
    ``dann_round5.install_dann`` does: the three methods share this one slot, and
    silently discarding an installed module produces a run that LOGS one method's
    settings while TRAINING another.
    """
    from coral_round4 import Fp32CoralLoss
    if isinstance(trainer.coral, Fp32CoralLoss):
        raise RuntimeError(
            "install_bnm() was handed a Trainer with Fp32CoralLoss installed. The methods "
            "share the da_loss hook slot; installing BNM over CORAL would train BNM while "
            "every CORAL setting still reads as configured.")
    if getattr(trainer, "dann", None) is not None:
        raise RuntimeError("install_bnm() was handed a Trainer with DANN installed.")
    if isinstance(trainer.coral, BNMHook):
        raise RuntimeError("install_bnm() called twice on the same Trainer.")
    if trainer.cfg.use_adamatch or trainer.cfg.use_freematch:
        raise RuntimeError("use_adamatch / use_freematch are set; BNM takes the CORAL "
                           "branch and the three are mutually exclusive.")

    hook = BNMHook(trainer.head, gamma)
    trainer.coral = hook
    trainer.cfg.use_coral = True
    trainer.cfg.coral_weight = float(gamma)
    return hook


# ---------------------------------------------------------------------------
# STAGE 2's grid rule -- gradient-norm calibration, reads no metric
# ---------------------------------------------------------------------------

GRID_FRACTIONS = (0.10, 0.50, 2.00)     # {10%, 50%, 200%} of the CE gradient norm


def _grad_norm(loss, params) -> float:
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    tot = 0.0
    for x in g:
        if x is not None:
            tot += float(x.detach().pow(2).sum())
    return tot ** 0.5


def bnm_gradient_calibration(trainer, src_batch, tgt_batch, *,
                             fractions=GRID_FRACTIONS) -> dict:
    """Measure BNM's gradient contribution against source CE at step 0, and derive gamma.

    THE DECLARED RULE (brief 7): pick three gamma spanning roughly {10%, 50%, 200%} of the
    source cross-entropy gradient norm, measured at step 0. **Reads no metric** -- no TPR,
    no AUROC, no test-set quantity anywhere. CORAL's {1, 4, 16} is NOT imported: the loss
    scales are not comparable, and 7.15 already recorded that CORAL's own magnitude-parity
    rule turned out undefined in this regime.

    Both norms are taken over the SAME parameter set (everything trainable) so the ratio
    is a like-for-like comparison rather than an artifact of which tensors each term
    happens to touch.
    """
    from train_skeleton import encode_features

    trainer.head.train()
    params = [p for p in trainer.head.parameters() if p.requires_grad]
    params += [p for p in trainer.encoder.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("no trainable parameters -- calibration is meaningless")

    src_feat = trainer.standardizer(
        encode_features(trainer.encoder, src_batch["image"].to(trainer.device),
                        trainer.cfg.encoder_mode))
    tgt_feat = trainer.standardizer(
        encode_features(trainer.encoder, tgt_batch["image"].to(trainer.device),
                        trainer.cfg.encoder_mode))
    src_logits, _ = trainer.head.forward_with_features(src_feat)
    _, tgt_adapter = trainer.head.forward_with_features(tgt_feat)

    y = src_batch["label"].float().to(trainer.device).view(-1, 1)
    ce = trainer.bce(src_logits, y)
    bnm = bnm_loss(trainer.head.classifier(tgt_adapter))

    ce_norm = _grad_norm(ce, params)
    bnm_norm = _grad_norm(bnm, params)
    if bnm_norm <= 0:
        raise SystemExit("BNM gradient norm is zero at step 0 -- the term is inert; STOP")

    grid = [round(f * ce_norm / bnm_norm, 6) for f in fractions]
    return {
        "rule": ("three gamma spanning {10%, 50%, 200%} of the source-CE gradient norm at "
                 "step 0, measured over the same trainable parameter set"),
        "reads_a_metric": False,
        "ce_grad_norm": ce_norm,
        "bnm_grad_norm_at_gamma_1": bnm_norm,
        "ratio_bnm_over_ce_at_gamma_1": bnm_norm / ce_norm,
        "fractions": list(fractions),
        "gamma_grid": grid,
        "n_params_in_norm": len(params),
        "note": ("gamma_i = fraction_i * ||grad CE|| / ||grad BNM(gamma=1)||, so gamma_i "
                 "makes the BNM term contribute that fraction of CE's gradient norm at "
                 "step 0. CORAL's {1,4,16} is deliberately NOT imported."),
    }


# ---------------------------------------------------------------------------
# STAGE 3's budget rule -- the nuclear-norm crossing, reads no metric
# ---------------------------------------------------------------------------

BUDGET_MULTIPLIER = 2       # DANN's precedent (2 x the lambda horizon), INHERITED

# THE BUDGET CEILING, DECLARED BEFORE THE DEVELOPMENT CURVE EXISTS. 8000 is not arbitrary:
# it is the DEVELOPMENT cap, which 7.14 already fixes as this project's maximum on
# power-matching grounds. "A confirmatory budget never exceeds the development budget" is
# statable in one line and defensible anywhere, and it makes brief 4's declined alternative
# coherent -- 8000 was considered as the budget and declined, so adopting it as the ceiling
# is the same number doing a weaker job.
#
# ABOVE IT THE RULE DOES NOT FIRE. It is NOT silently clamped to 8000: a clamped budget is
# indistinguishable in the JSON from a derived one, and six months later nobody can tell
# which happened. Refusal is the same branch as never-crossing -- stop and report.
#
# In practice, with the 2x multiplier, any crossing past step 4000 lands in the refusal
# branch. If that fires it is INFORMATIVE, not a dead end: a nuclear norm that has not
# exhausted sharpening-only progress by step 4000 on a 782-image target says something
# about how much grip the objective has here, and that is reportable.
BUDGET_CEILING    = 8000

# THE MATCHED-ARM BUDGET SET, declared BEFORE the development run (Stage-3 brief 3,
# revised by the prereg-edits brief 1).
#
# A FLOOR WAS THE WRONG INSTRUMENT and was replaced before hashing. 7.22 established that
# cross-arm contrasts are made by reading at a MATCHED STEP, not by clearing a threshold --
# lambda=0 was read at 2000 for CORAL and at 1500 for FreeMatch. 7.16 showed the read-off
# point matters enormously, so a budget of 1600 is not "comparable to FreeMatch's 1500"
# merely because it exceeds it. Membership of the declared set is the property that
# actually licenses a contrast; being above some number is not.
#
# WHY THIS COULD BE CHANGED NOW without it being a rescue: it is OUTCOME-NEUTRAL for this
# round. The Stage-2 trajectory implies a budget of 1000, which is neither >= 1500 nor a
# member of {1500, 2000, 4000} -- the verdict is identical under either definition. A
# definition change that cannot alter the outcome it is being applied to is a repair of the
# instrument, not a selection of the result. (PROJECT.md 7.23 records this reasoning.)
#
# Stage 2 measured the saturation -- with C = 2 the confidence term is ~99% spent inside
# 1000 steps -- so a budget far below every other arm is foreseeable. THE RULE ITSELF DOES
# NOT CHANGE; only which number may be quoted against other arms.
MATCHED_ARM_BUDGETS = (1500, 2000, 4000)   # FreeMatch, CORAL, lambda=0/DANN


def nuclear_norm_crossing(steps, values, ceiling: float,
                          multiplier: int = BUDGET_MULTIPLIER,
                          budget_ceiling: int = BUDGET_CEILING,
                          matched_arm_budgets=MATCHED_ARM_BUDGETS) -> dict:
    """First step at which the normalised nuclear norm reaches the sharpening ceiling.

    THE DECLARED RULE (brief 4): budget = ``multiplier`` x the fold-0 step at which
    normalised ||A||_* first reaches the sharpening-only ceiling (10.84 from Stage 0a).

    WHY THAT CEILING. It is a genuine PHASE BOUNDARY in the objective, fixed analytically
    before any run: below it the loss is obtainable by sharpening alone, because
    temperature cannot reassign an image between classes; above it, further gain REQUIRES
    the model to actually re-balance. "Run until confidence-only progress is exhausted,
    then the same again for re-balancing" is a statement about the loss, not about
    performance. The 2x is DANN's precedent and is stated as inherited, not derived.

    READS NO METRIC: the trajectory is the nuclear norm, never TPR or AUROC.

    TWO REFUSAL BRANCHES, both declared before the development curve exists:
      * the sharpening ceiling is never reached, or
      * the rule returns more than ``budget_ceiling`` (8000, the development cap 7.14
        already fixes as this project's maximum -- a confirmatory budget never exceeds
        the development budget).
    In either case ``fires`` is False and ``budget`` is None. **The result is NOT clamped
    to the ceiling**: a clamped budget looks identical to a derived one in this JSON.
    Stop and report.
    """
    pairs = sorted((int(s), float(v)) for s, v in zip(steps, values))
    hit = next((s for s, v in pairs if v >= ceiling), None)
    raw = int(multiplier * hit) if hit is not None else None
    over = raw is not None and raw > int(budget_ceiling)

    if hit is None:
        fires, budget, why = False, None, "the ceiling was never reached"
        outcome = "never_crossed"
    elif over:
        fires, budget, why = False, None, (
            f"the rule returned {raw}, above the declared ceiling {budget_ceiling}")
        outcome = "exceeds_ceiling"
    elif raw not in tuple(int(b) for b in matched_arm_budgets):
        # THIRD OUTCOME. It FIRES -- the budget is derived and used -- but it changes
        # which number a contrast table may quote. Distinct from the two refusals on
        # purpose: refusing here would discard a legitimate reading of the rule.
        fires, budget, why = True, raw, None
        outcome = "no_matched_arm_budget"
    else:
        fires, budget, why = True, raw, None
        outcome = "normal"

    return {
        "rule": (f"budget = {multiplier} x the first step at which normalised ||A||_* "
                 f"reaches {ceiling:.4f}, the sharpening-only ceiling; NEVER exceeding "
                 f"the declared ceiling of {budget_ceiling}"),
        "reads_a_metric": False,
        "ceiling": float(ceiling),
        "multiplier": int(multiplier),
        "budget_ceiling": int(budget_ceiling),
        "crossed": hit is not None,
        "crossing_step": hit,
        "budget_before_ceiling": raw,
        "exceeds_ceiling": bool(over),
        "fires": fires,
        "budget": budget,
        "outcome": outcome,
        "matched_arm_budgets": [int(b) for b in matched_arm_budgets],
        "no_matched_arm_budget": outcome == "no_matched_arm_budget",
        "cross_arm_comparable": (outcome != "no_matched_arm_budget") if fires else None,
        "budget_confound_removed_by_a_match": (
            "A MATCH REMOVES THE BUDGET CONFOUND ONLY. Head initialisation and the "
            "gamma-tuning budget still travel with every contrast (7.18) and are not "
            "addressed by reading at a matched step."),
        "which_number_is_quoted": (
            "the rule-derived budget is not one of the declared arm budgets "
            "{1500, 2000, 4000}, so it is quoted ONLY as BNM's own-objective "
            "reading -- 'BNM read where its own objective says it is done'. All "
            "cross-arm contrasts use MATCHED READOUTS, with 4000 mandatory as the "
            "lambda=0 known-null's budget"
            if outcome == "no_matched_arm_budget" else
            "the rule-derived budget is quoted normally"),
        "refusal_reason": why,
        "n_points": len(pairs),
        "max_observed": max((v for _, v in pairs), default=None),
        "if_it_does_not_fire": (
            "STOP AND REPORT. Do NOT substitute a fallback chosen after seeing the curve, "
            "and do NOT clamp to the ceiling -- a clamped budget is indistinguishable "
            "from a derived one in this JSON. With multiplier 2 and ceiling "
            f"{budget_ceiling}, any crossing past step {budget_ceiling // multiplier} "
            "refuses; that refusal is itself reportable, because a nuclear norm that has "
            "not exhausted sharpening-only progress by then says something about the "
            "objective's grip on a 782-image target."),
    }


# ---------------------------------------------------------------------------
# THE RUNTIME INVARIANT -- gamma never varies with the step
# ---------------------------------------------------------------------------
#
# This guards the arm's DEFINING PROPERTY. "BNM has no horizon" is why it is the
# minimal-signal control rather than a fourth ranking entry, so it is enforced rather
# than asserted in prose.
#
# WHY NOT A TOKEN SCAN. A static scan for set_step / lambda_at / total_steps / horizon /
# step / warmup / ramp / schedule is defeated by arithmetic:
#
#     w = min(1.0, i / 2000.0)          # a ramp, and none of those tokens appear
#     loss = ce + w * gamma * bnm       # written in good faith while tidying the loop
#
# So the check is made at the LOSS-ASSEMBLY SITE instead. train_skeleton computes
# ``da_loss = cfg.coral_weight * self.coral(...)`` and returns ``da_loss``; the hook
# separately records the UNWEIGHTED value. Their ratio is the gamma actually multiplied
# in, and ANY extra factor introduced anywhere between the hook's return and the total
# loss changes it. That is what a token scan cannot see.
#
# TWO LAYERS, WITH DIFFERENT EXACTNESS, AND THE DIFFERENCE IS NOT COSMETIC:
#
#   RECORDED GAMMA (cfg.coral_weight, hook.gamma) -- no arithmetic is involved, so these
#   are compared for BIT-IDENTITY. They catch a ramp implemented by mutating either.
#
#   EFFECTIVE GAMMA (da_loss / raw) -- arithmetic IS involved: da_loss is a float32
#   product re-widened to float64, so the ratio is not bit-stable across steps even when
#   nothing is wrong. Measured over 8000 draws the worst relative deviation from gamma is
#   7.1e-08, consistent with float32's 2^-24 = 6.0e-08 half-ulp. EFFECTIVE_RTOL = 1e-6 is
#   ~14x that, while the smallest ramp step the brief's example can take (w = 1/2000)
#   shifts the ratio by a relative 0.9995 -- roughly a million times the tolerance. The
#   tolerance is therefore forced by float32, not chosen to be permissive.
GAMMA_EFFECTIVE_RTOL = 1e-6


class GammaHorizonDetected(RuntimeError):
    """gamma moved. The arm is no longer the minimal-signal control."""


class GammaInvariant:
    """Records the gamma actually multiplied into the loss, and refuses drift.

    ``observe`` is called every step and raises :class:`GammaHorizonDetected` on the
    FIRST divergence, naming the step and both values. ``to_artifact`` returns the full
    sequence so the invariant is verifiable after the fact rather than only enforced
    during -- a guard whose output is not persisted cannot be checked by a reader.
    """

    def __init__(self, declared_gamma: float, rtol: float = GAMMA_EFFECTIVE_RTOL):
        self.declared = float(declared_gamma)
        self.rtol = float(rtol)
        self.step0 = None
        self.steps: list = []
        self.cfg_gammas: list = []
        self.hook_gammas: list = []
        self.effective: list = []
        self.held = True
        self.first_divergence = None

    def observe(self, step: int, *, cfg_coral_weight, hook_gamma,
                da_loss, raw_loss) -> dict:
        cfg_g, hook_g = float(cfg_coral_weight), float(hook_gamma)

        # The ratio is undefined when the unweighted term is exactly zero; record that and
        # skip only THAT check rather than silently passing the step.
        eff = float("nan") if raw_loss == 0.0 else float(da_loss) / float(raw_loss)

        if self.step0 is None:
            self.step0 = {"step": int(step), "cfg_gamma": cfg_g, "hook_gamma": hook_g,
                          "gamma_effective": eff}

        self.steps.append(int(step))
        self.cfg_gammas.append(cfg_g)
        self.hook_gammas.append(hook_g)
        self.effective.append(eff)

        ref = self.step0
        # --- layer 1: bit-identity on the recorded values (no arithmetic) ---
        if cfg_g != ref["cfg_gamma"]:
            self._fail(step, "cfg.coral_weight", ref["cfg_gamma"], cfg_g, exact=True)
        if hook_g != ref["hook_gamma"]:
            self._fail(step, "hook.gamma", ref["hook_gamma"], hook_g, exact=True)

        # --- layer 2: the effective gamma, at the tolerance float32 forces ---
        r0 = ref["gamma_effective"]
        if eff == eff and r0 == r0:                     # both finite
            if r0 == 0.0:
                if eff != 0.0:
                    self._fail(step, "gamma_effective (declared gamma is 0)", r0, eff,
                               exact=True)
            elif abs(eff - r0) / abs(r0) > self.rtol:
                self._fail(step, "gamma_effective (da_loss / raw_bnm)", r0, eff,
                           exact=False)

        return {"step": int(step), "cfg_gamma": cfg_g, "hook_gamma": hook_g,
                "gamma_effective": eff, "held": self.held}

    def _fail(self, step, what, expected, got, *, exact):
        self.held = False
        self.first_divergence = {"step": int(step), "quantity": what,
                                 "step0_value": expected, "value": got,
                                 "comparison": ("bit-identity" if exact
                                                else f"relative <= {self.rtol}")}
        how = "bit-identity" if exact else f"relative tolerance {self.rtol}"
        raise GammaHorizonDetected(
            f"gamma MOVED at step {step}: {what} was {expected!r} at step "
            f"{self.step0['step']} and is {got!r} now ({how} violated).\n"
            f"  BNM's whole claim is that it has NO HORIZON -- that is what makes it the "
            f"minimal-signal control rather than a fourth ranking entry. A gamma that "
            f"varies with the step makes this a different arm, so the run stops here "
            f"rather than producing a number under a name that no longer fits it.")

    def to_artifact(self) -> dict:
        finite = [x for x in self.effective if x == x]
        return {
            "invariant": ("gamma is constant: the value actually multiplied into the loss "
                          "is bit-stable in its recorded form and stable to "
                          f"{self.rtol} relative in its effective form"),
            "declared_gamma": self.declared,
            "held": self.held,
            "first_divergence": self.first_divergence,
            "n_observations": len(self.steps),
            "step0": self.step0,
            "effective_rtol": self.rtol,
            "gamma_effective_min": min(finite) if finite else None,
            "gamma_effective_max": max(finite) if finite else None,
            "n_undefined_ratio_steps": sum(1 for x in self.effective if x != x),
            "cfg_gamma_distinct": sorted(set(self.cfg_gammas)),
            "hook_gamma_distinct": sorted(set(self.hook_gammas)),
            "steps": self.steps,
            "gamma_effective": self.effective,
            "why_the_two_layers_differ": (
                "the recorded values involve no arithmetic and are compared for "
                "bit-identity; the effective ratio is a float32 product re-widened to "
                "float64, whose worst observed deviation over 8000 draws is 7.1e-08 "
                "against float32's 6.0e-08 half-ulp, so it gets a relative tolerance "
                f"of {self.rtol} -- ~14x the rounding and ~1e6x below the smallest ramp "
                "step the guard exists to catch"),
        }


def gamma_grid_from_file(path: str) -> dict:
    """Stage 2's grid, READ FROM the --measure-step0 artifact rather than recomputed.

    The same discipline :func:`stage0a_anchors` follows: a rule reads a file, so there is
    one source of truth and a test can corrupt the file to prove no literal survives
    anywhere. Recomputing here would create a second path to the same number that could
    silently disagree with what the run actually measured.
    """
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    for k in ("ce_grad_norm", "bnm_grad_norm_at_gamma_1", "fractions"):
        if k not in d:
            raise SystemExit(f"{path}: missing {k}; run --measure-step0 first")
    ce, bnm = float(d["ce_grad_norm"]), float(d["bnm_grad_norm_at_gamma_1"])
    if bnm <= 0:
        raise SystemExit(f"{path}: BNM gradient norm is {bnm}; the term is inert. STOP.")
    fr = [float(x) for x in d["fractions"]]
    return {"source": path, "ce_grad_norm": ce, "bnm_grad_norm_at_gamma_1": bnm,
            "fractions": fr, "gamma_grid": [round(f * ce / bnm, 6) for f in fr],
            "reads_a_metric": False}


# ---------------------------------------------------------------------------
# THE 704 / 78 MEMORISATION DIAGNOSTIC -- SIZE-MATCHED, and that is not optional
# ---------------------------------------------------------------------------
#
# The 78 held-out target images are a slice the BNM loss NEVER SEES. BNM is the arm most
# exposed to 7.12's memorisation finding: ~70% of its pull is the confidence term (7.20),
# applied to images the encoder has already memorised. Comparing the trained-on pool
# against the held-out slice separates GENERALISING confidence (both sharpen together)
# from MEMORISING confidence (trained-on sharpens, held-out does not) -- which is exactly
# the mechanism question BNM's negative branch turns on.
#
# THE TRAP, measured before writing the diagnostic. Normalised ||A||_* is
# ||A||_*/min(B,C), and at a confident state that is sqrt(B/C) -- it GROWS WITH THE SAMPLE
# SIZE. Comparing 704 images against 78 at IDENTICAL behaviour gives
# sqrt(704/2)/sqrt(78/2) = 3.00x, so a naive "704 vs 78 gap" would be almost entirely
# sqrt(B) and would look like an enormous memorisation signal in a perfectly healthy run.
#
# So the trained-on probe is a FIXED, DETERMINISTIC, LATENT-STRATIFIED sample of exactly
# n_holdout images -- the same size, drawn the same way, materialised once. Fixed for the
# same reason materialise_holdout_images is fixed: if the probe were re-drawn each time, a
# change could be the model moving or the sample moving, and the monitor exists to say
# which.
PROBE_SEED_OFFSET = 7717        # distinct from the holdout's own seed


def build_trainedon_probe(target_ds, holdout: dict, *, seed: int) -> dict:
    """A size-matched, latent-stratified probe drawn from the TRAINED-ON target pool.

    Mirrors ``dann_round5.build_target_holdout``'s construction so the two slices differ
    in one thing only: whether the loss saw them.
    """
    import numpy as _np

    held = set(int(i) for i in holdout["holdout"])
    n_want = len(held)
    by_latent = target_ds.indices_by_latent
    rng = _np.random.default_rng(int(seed) + PROBE_SEED_OFFSET)

    picked = []
    for lat in sorted(by_latent):
        pool = sorted(int(i) for i in by_latent[lat] if int(i) not in held)
        share = int(round(n_want * len(by_latent[lat]) / len(target_ds)))
        share = min(share, len(pool))
        if share:
            picked += list(rng.choice(pool, size=share, replace=False))
    # top up deterministically if rounding left the probe short
    if len(picked) < n_want:
        rest = sorted(set(range(len(target_ds))) - held - set(picked))
        picked += list(rng.choice(rest, size=n_want - len(picked), replace=False))
    picked = sorted(int(i) for i in picked)[:n_want]

    return {"probe": picked, "n_probe": len(picked), "n_holdout": n_want,
            "seed": int(seed) + PROBE_SEED_OFFSET,
            "size_matched": len(picked) == n_want,
            "disjoint_from_holdout": not (set(picked) & held),
            "why_size_matched": (
                "normalised ||A||_* is sqrt(B/C) at a confident state, so it grows with "
                "the sample size: 704 vs 78 reads 3.00x higher at IDENTICAL behaviour. An "
                "unmatched comparison would report sqrt(B) as memorisation.")}


@torch.no_grad()
def probe_stats(trainer, images) -> dict:
    """normalised ||A||_*, mean entropy and predicted-positive fraction on a fixed slice."""
    from train_skeleton import encode_features

    was_train = trainer.head.training
    trainer.head.eval()
    feats = trainer.standardizer(
        encode_features(trainer.encoder, images.to(trainer.device),
                        trainer.cfg.encoder_mode))
    logits, _ = trainer.head.forward_with_features(feats)
    prob2 = binary_prob2(logits)
    p_pos = prob2[:, 1]
    # float32 for the same reason as the hook: fp16 + a 1e-12 clamp is nan.
    _p32 = prob2.float()
    ent = -(_p32.clamp_min(BNM_EPS).log() * _p32).sum(1).mean()
    if was_train:
        trainer.head.train()
    return {"n": int(images.shape[0]),
            "nuclear_norm": float(bnm_nuclear_norm(prob2)),
            "entropy": float(ent),
            "pred_pos_frac": float((p_pos > 0.5).float().mean()),
            "mean_p_pos": float(p_pos.mean())}


def holdout_gap(trained_on: dict, held_out: dict) -> dict:
    """trained-on MINUS held-out, on size-matched slices.

    A positive nuclear gap with a negative entropy gap is the MEMORISING signature: the
    loss sharpened the images it saw and not the ones it did not. Both moving together is
    the GENERALISING signature.
    """
    if trained_on["n"] != held_out["n"]:
        raise ValueError(
            f"the slices are not size-matched ({trained_on['n']} vs {held_out['n']}); "
            f"normalised ||A||_* scales as sqrt(B) and the gap would be meaningless")
    return {"n_per_slice": trained_on["n"],
            "nuclear_gap": trained_on["nuclear_norm"] - held_out["nuclear_norm"],
            "entropy_gap": trained_on["entropy"] - held_out["entropy"],
            "pred_pos_frac_gap": trained_on["pred_pos_frac"] - held_out["pred_pos_frac"],
            "trained_on": trained_on, "held_out": held_out}


# ---------------------------------------------------------------------------
# STAGE 2's SELECTION, and the EXTENSION RULE for "none of the three passes"
# ---------------------------------------------------------------------------
#
# The criterion says take the smaller if two gamma pass. It did not say what happens if
# NONE does, and that outcome is plausible: 10% of the CE gradient norm may not move
# ||A||_* materially off its 9.77 init. Extending the grid upward reads only the nuclear
# norm, so it stays metric-blind -- but only if the rule is DECLARED BEFORE THE SWEEP.
# Declared here, before any sweep arm has run:
GRID_EXTENSION_MULTIPLIER = 4      # matches the grid's own spacing (0.10 -> 0.50 -> 2.00)
MAX_GRID_EXTENSIONS = 1
NUCLEAR_MATERIAL_MOVE = 0.10       # absolute, on the normalised nuclear norm


def extend_grid(grid, *, multiplier: int = GRID_EXTENSION_MULTIPLIER) -> list:
    """One extension: multiply the TOP gamma by the grid's own spacing factor."""
    return [round(max(grid) * multiplier, 6)]


def select_gamma(arms, *, band: dict, init_nuclear: float,
                 material: float = NUCLEAR_MATERIAL_MOVE,
                 n_extensions_used: int = 0) -> dict:
    """Apply Stage 2's three-condition criterion. READS NO METRIC.

    ``arms`` is ``[{"gamma", "nuclear_norm_final", "pred_pos_frac_final",
    "frozen_probe_auroc_final", "frozen_probe_auroc_step0"}, ...]``.

    1. normalised ||A||_* moves materially from its init (the round-1 CORAL guard);
    2. predicted-positive fraction stays above the GATING FLOOR -- the 0.75 ceiling is
       ADVISORY and is recorded but never selects;
    3. the frozen-feature probe does not collapse.

    Two or more pass -> the SMALLER (7.15's winner's-curse argument). One -> that one.
    None -> the caller applies :func:`extend_grid` once; if that also fails, STOP AND
    REPORT with reason code ``no_gamma_passed_after_extension`` -- the same refusal shape
    as the budget rule's two branches, and for the same reason.
    """
    rows = []
    for a in arms:
        moved = abs(float(a["nuclear_norm_final"]) - float(init_nuclear)) >= material
        floor_ok = float(a["pred_pos_frac_final"]) >= float(band["floor"])
        ceil_ok = float(a["pred_pos_frac_final"]) <= float(band["ceiling"])
        probe0 = float(a["frozen_probe_auroc_step0"])
        probe_ok = float(a["frozen_probe_auroc_final"]) >= 0.5 + 0.5 * (probe0 - 0.5)
        rows.append({**a, "c1_nuclear_moved": moved, "c2_floor_ok": floor_ok,
                     "c2_advisory_ceiling_ok": ceil_ok, "c3_probe_ok": probe_ok,
                     "passes": bool(moved and floor_ok and probe_ok)})

    passing = sorted((r for r in rows if r["passes"]), key=lambda r: float(r["gamma"]))
    out = {"arms": rows, "n_passing": len(passing),
           "criterion": ("(1) ||A||_* moves >= "
                         f"{material} from init {init_nuclear:.4f}; (2) pred-pos frac >= "
                         f"the GATING floor {band['floor']:.4f} -- the {band['ceiling']} "
                         f"ceiling is ADVISORY and never selects; (3) the frozen probe "
                         f"retains >= half its headroom over chance"),
           "reads_a_metric": False,
           "n_extensions_used": int(n_extensions_used),
           "extension_rule": (f"if none passes, multiply the TOP gamma by "
                              f"{GRID_EXTENSION_MULTIPLIER} (the grid's own spacing) and "
                              f"re-run; at most {MAX_GRID_EXTENSIONS} extension"),
           }
    if passing:
        chosen = passing[0]
        out.update(selected=float(chosen["gamma"]), decided=True,
                   basis=("the smaller of the passing arms (7.15's winner's curse)"
                          if len(passing) > 1 else "the only passing arm"))
        # THE ADVISORY CEILING NEVER GATES -- that is the declared design (a condition
        # that might fire on legitimate behaviour goes advisory rather than being
        # loosened until it cannot fire). The consequence is that an arm well up the
        # all-positive direction CAN be selected. So when the selected arm breaches it,
        # the breach is raised to the TOP of the result and repeated in a field a reader
        # cannot skim past, because the mechanism that catches it is a human at the
        # Stage-2 report, not this function.
        if not chosen["c2_advisory_ceiling_ok"]:
            out["ADVISORY_CEILING_BREACHED_BY_SELECTED_ARM"] = {
                "gamma": float(chosen["gamma"]),
                "pred_pos_frac": float(chosen["pred_pos_frac_final"]),
                "ceiling": float(band["ceiling"]),
                "status": ("SELECTED ANYWAY -- by design, the ceiling is advisory and "
                           "never gates. This is the intended escalation path: it is "
                           "surfaced here so a human decides at the Stage-2 report, "
                           "before the gamma is hashed into a pre-registration."),
            }
    elif int(n_extensions_used) < MAX_GRID_EXTENSIONS:
        out.update(selected=None, decided=False, action="EXTEND",
                   next_grid=extend_grid([float(a["gamma"]) for a in arms]))
    else:
        out.update(selected=None, decided=False, action="STOP_AND_REPORT",
                   reason_code="no_gamma_passed_after_extension",
                   reason=("no gamma satisfied the three conditions, and the one "
                           "permitted extension also failed. STOP AND REPORT -- do not "
                           "widen the grid again, and do not relax a condition after "
                           "seeing the sweep. Same refusal shape as the budget rule."))
    return out
