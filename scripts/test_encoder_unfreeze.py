"""
test_encoder_unfreeze.py
========================
Tests for LEVER 1 -- `n_blocks` partial unfreezing (LP->FT) of the Zoobot greyscale
ConvNeXT-Nano encoder, plus the layer-decayed optimizer groups in train_skeleton.py.

These run on the REAL checkpoint (they must: the whole point is that the block
convention and the param counts match Zoobot's own `configure_optimizers`, and a mock
model could not prove that). Weights are downloaded once and then cached by timm.

THE SEVEN CHECKS (spec §2)
--------------------------
  1. EXACT param-count table for n_blocks 0..5 -- hardcoded MEASURED anchors. This is the
     regression anchor: if timm renames/reshapes a stage, or the reversal order flips,
     these numbers move and the test fails loudly.
  2. Correct block IDENTITY at n_blocks=3: exactly stages[1], stages[2], stages[3]
     trainable; stem and stages[0] frozen. (n_blocks counts timm STAGES, not ConvNeXt
     residual blocks -- the easy thing to get wrong.)
  3. `head.norm` frozen at EVERY n_blocks including 5 -- it sits outside Zoobot's
     tuneable_blocks, so replicating the benchmark means never tuning it.
  4. FROZEN-PATH INVARIANCE at n_blocks=0: 0 trainable params AND a forward pass that is
     BYTE-IDENTICAL to the pre-Lever-1 encoder. The reference is an INDEPENDENT
     reimplementation of the original 20-line forward (timm.create_model with the same
     slug/in_chans -> forward_features -> forward_head(pre_logits=True) under no_grad),
     inlined below rather than diffed against a file copy -- an independent recompute is
     a stronger check than comparing a module against a snapshot of itself.
     This protects the existing frozen results in runs/.
  5. Gradient flow at n_blocks=1: stages[3] gets non-None, non-zero grads; stages[2],
     stages[0] and stem have `grad is None` (autograd stops at the first trainable module
     -- which is the compute saving requires_grad=False buys over Zoobot's
     omit-from-optimizer approach).
  6. LR assignment at n_blocks=3, backbone_lr=L, lr_decay=0.75: the three encoder groups
     are L, 0.75L, 0.5625L mapped to stages[3], stages[2], stages[1] respectively, and
     the head group is head_lr with NO decay factor applied.
  7. ADVERSARIAL: n_blocks=6 must RAISE (not silently clamp to 5).

Run:
    python scripts\\test_encoder_unfreeze.py
Exits 0 on success, 1 on any failed check, 2 on an unexpected exception.
"""

import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import timm

from zoobot_encoder import (ZoobotEncoder, MODEL_SLUG, IN_CHANS, INPUT_SIZE, EMBED_DIM,
                            TRAINABLE_PARAMS_BY_N_BLOCKS)
from train_skeleton import Trainer, TrainConfig

# ---- the MEASURED regression anchor (verified locally with timm 1.0.26; the spec's
# table was verified with 1.0.28 and agrees exactly). HARDCODED on purpose. ----
EXPECTED_TRAINABLE = {
    0: 0,
    1: 7_448_320,       # stages[3]
    2: 14_355_840,      # + stages[2]
    3: 14_835_520,      # + stages[1]   <- Lines et al.'s setting == 99.2% == effectively full FT
    4: 14_947_200,      # + stages[0]
    5: 14_948_720,      # + stem        (total 14,950,000 minus head.norm's 1,280)
}
TOTAL_ENCODER_PARAMS = 14_950_000
HEAD_NORM_PARAMS = 1_280

_fails = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _fails.append(msg)


def _build(n_blocks, **kw):
    """Construct the encoder quietly (verbose prints are asserted-unchanged elsewhere)."""
    return ZoobotEncoder(n_blocks=n_blocks, device="cpu", verbose=False,
                         self_check=False, **kw)


# ---------------------------------------------------------------------------
# 1. exact param-count table
# ---------------------------------------------------------------------------

def test_param_count_table():
    print("\n[1] exact trainable-param table for n_blocks 0..5")
    for n in range(6):
        enc = _build(n)
        got = enc.num_trainable_params()
        want = EXPECTED_TRAINABLE[n]
        check(got == want,
              f"n_blocks={n}: trainable encoder params {got:,} == {want:,}")
        total = sum(p.numel() for p in enc.model.parameters())
        if n == 0:
            check(total == TOTAL_ENCODER_PARAMS,
                  f"total encoder params {total:,} == {TOTAL_ENCODER_PARAMS:,}")
        del enc
    # the module's own published table must agree with the test's hardcoded one
    check(dict(TRAINABLE_PARAMS_BY_N_BLOCKS) == EXPECTED_TRAINABLE,
          "zoobot_encoder.TRAINABLE_PARAMS_BY_N_BLOCKS matches the hardcoded anchor")


# ---------------------------------------------------------------------------
# 2. correct block identity at n_blocks=3
# ---------------------------------------------------------------------------

def _all_frozen(module):
    return all(not p.requires_grad for p in module.parameters())


def _all_trainable(module):
    ps = list(module.parameters())
    return bool(ps) and all(p.requires_grad for p in ps)


def test_block_identity():
    print("\n[2] block identity at n_blocks=3 (timm STAGES, top-first)")
    enc = _build(3)
    m = enc.model
    check(_all_trainable(m.stages[3]), "stages[3] trainable")
    check(_all_trainable(m.stages[2]), "stages[2] trainable")
    check(_all_trainable(m.stages[1]), "stages[1] trainable")
    check(_all_frozen(m.stages[0]), "stages[0] FROZEN")
    check(_all_frozen(m.stem), "stem FROZEN")
    check(enc.trainable_block_names == ["stages.3", "stages.2", "stages.1"],
          f"trainable_block_names == ['stages.3','stages.2','stages.1'] "
          f"(got {enc.trainable_block_names})")
    # the exposed ordering is the one the optimizer consumes -- pin it
    names = [n for n, _ in enc.tuneable_blocks_in_lr_order]
    check(names == ["stages.3", "stages.2", "stages.1", "stages.0", "stem"],
          f"tuneable_blocks_in_lr_order is top-first {names}")


# ---------------------------------------------------------------------------
# 3. head.norm frozen at every n_blocks
# ---------------------------------------------------------------------------

def test_head_norm_always_frozen():
    print("\n[3] head.norm frozen at EVERY n_blocks (0..5), including 5")
    for n in range(6):
        enc = _build(n)
        hn = enc.model.head.norm
        npar = sum(p.numel() for p in hn.parameters())
        check(_all_frozen(hn) and npar == HEAD_NORM_PARAMS,
              f"n_blocks={n}: head.norm ({npar:,} params) requires_grad=False")
        del enc


# ---------------------------------------------------------------------------
# 4. frozen-path invariance (byte-identical to the pre-Lever-1 encoder)
# ---------------------------------------------------------------------------

def _reference_forward(x):
    """INDEPENDENT reimplementation of the ORIGINAL frozen forward, pre-Lever-1.

    Original zoobot_encoder.py did exactly this and nothing else:
        model = timm.create_model(SLUG, pretrained=True, num_classes=0, in_chans=1)
        model.eval(); every param requires_grad_(False)
        with torch.no_grad():
            feats  = model.forward_features(x)
            pooled = model.forward_head(feats, pre_logits=True)
    """
    torch.manual_seed(0)
    model = timm.create_model(MODEL_SLUG, pretrained=True, num_classes=0, in_chans=IN_CHANS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        return model.forward_head(model.forward_features(x), pre_logits=True)


def test_frozen_invariance():
    print("\n[4] n_blocks=0 frozen-path invariance (byte-identical output)")
    torch.manual_seed(1234)
    x = torch.rand(4, IN_CHANS, INPUT_SIZE, INPUT_SIZE) * 0.881   # in-contract [0, asinh(1)]

    enc = _build(0)
    check(enc.num_trainable_params() == 0, "n_blocks=0: 0 trainable params")
    check(getattr(enc, "_frozen", False) is True,
          "n_blocks=0: _frozen=True (the torch.no_grad() fast path is preserved)")
    out = enc(x)
    check(out.requires_grad is False and out.grad_fn is None,
          "n_blocks=0: output still detached (no autograd graph)")
    check(tuple(out.shape) == (4, EMBED_DIM), f"output shape {tuple(out.shape)}")

    ref = _reference_forward(x)
    check(torch.equal(out, ref),
          f"n_blocks=0 output is BYTE-IDENTICAL to the pre-Lever-1 encoder "
          f"(max|diff|={float((out - ref).abs().max()):.3e})")


# ---------------------------------------------------------------------------
# 5. gradient flow at n_blocks=1
# ---------------------------------------------------------------------------

def test_gradient_flow():
    print("\n[5] gradient flow at n_blocks=1 (autograd stops below stages[3])")
    enc = _build(1)
    check(getattr(enc, "_frozen", True) is False,
          "n_blocks=1: _frozen=False (no_grad fast path correctly disabled)")

    torch.manual_seed(7)
    x = torch.rand(2, IN_CHANS, INPUT_SIZE, INPUT_SIZE) * 0.881
    out = enc(x)
    check(out.requires_grad is True and out.grad_fn is not None,
          "n_blocks=1: output carries grad (features are trainable)")
    out.pow(2).mean().backward()

    m = enc.model
    s3 = [p for p in m.stages[3].parameters()]
    check(all(p.grad is not None for p in s3), "stages[3]: every param has a non-None grad")
    check(any(float(p.grad.abs().sum()) > 0 for p in s3),
          "stages[3]: gradients are non-zero")
    for label, mod in (("stages[2]", m.stages[2]), ("stages[0]", m.stages[0]),
                       ("stem", m.stem)):
        check(all(p.grad is None for p in mod.parameters()),
              f"{label}: grad is None (no backward compute spent below the unfreeze point)")
    check(all(p.grad is None for p in m.head.norm.parameters()),
          "head.norm: grad is None")


# ---------------------------------------------------------------------------
# 6. LR assignment (layer decay), consuming tuneable_blocks_in_lr_order
# ---------------------------------------------------------------------------

def test_lr_assignment():
    print("\n[6] layer-decayed LR groups at n_blocks=3, backbone_lr=L, lr_decay=0.75")
    L, HEAD_LR = 1e-4, 1e-3
    enc = _build(3)
    cfg = TrainConfig(head_type="linear", n_blocks=3, lr=HEAD_LR, backbone_lr=L,
                      lr_decay=0.75, device="cpu", seed=42)
    tr = Trainer(cfg, encoder=enc)
    groups = tr.opt.param_groups

    check(len(groups) == 4, f"4 param groups (head + 3 encoder blocks), got {len(groups)}")
    names = [g.get("name") for g in groups]
    check(names == ["head", "stages.3", "stages.2", "stages.1"],
          f"group order is head-first then top-down encoder blocks: {names}")

    check(abs(groups[0]["lr"] - HEAD_LR) < 1e-12,
          f"group 0 (head): lr == head_lr {HEAD_LR:g} (NO decay factor applied)")
    for i, want in enumerate([L, 0.75 * L, 0.75 * 0.75 * L]):
        g = groups[i + 1]
        check(abs(g["lr"] - want) < 1e-15,
              f"group {i+1} ({g['name']}): lr {g['lr']:.6e} == L*0.75^{i} = {want:.6e}")

    # the encoder groups must hold the ACTUAL block params, not just the right count
    m = enc.model
    for gi, mod in ((1, m.stages[3]), (2, m.stages[2]), (3, m.stages[1])):
        gids = {id(p) for p in groups[gi]["params"]}
        check(gids == {id(p) for p in mod.parameters()},
              f"group {gi} ({groups[gi]['name']}) holds exactly that block's params")

    # ...and n_blocks=0 must still build the ORIGINAL single-group optimizer
    enc0 = _build(0)
    tr0 = Trainer(TrainConfig(head_type="linear", n_blocks=0, lr=HEAD_LR, device="cpu"),
                  encoder=enc0)
    check(len(tr0.opt.param_groups) == 1,
          f"n_blocks=0 builds ONE param group (unchanged), got {len(tr0.opt.param_groups)}")
    check(abs(tr0.opt.param_groups[0]["lr"] - HEAD_LR) < 1e-12,
          "n_blocks=0 single group keeps lr=cfg.lr")
    n_enc_in_opt = sum(1 for g in tr0.opt.param_groups for p in g["params"]
                       if p.requires_grad and p.numel() > 0
                       and any(p is q for q in enc0.model.parameters()))
    check(n_enc_in_opt == 0, "n_blocks=0: NO encoder param reaches the optimizer")


# ---------------------------------------------------------------------------
# 6b. C1/C2 PROTOCOL MATCH -- Zoobot v2.0.1 published defaults, adopted not tuned
# ---------------------------------------------------------------------------

def test_zoobot_default_lr_groups():
    """At n_blocks=3 with ZOOBOT DEFAULTS (i.e. passing NOTHING), the group LRs must be
    1e-4 / 1e-4 / 7.5e-5 / 5.625e-5 and weight_decay 0.05 on EVERY group.

    The head/top-block equality is the load-bearing part: Zoobot creates the head group with
    `lr` and gives encoder block i=0 `lr * (lr_decay ** 0)`, so they are the same number.
    Before the protocol match our sentinel resolved backbone_lr to lr/10, which would make
    the top block 1e-5 -- an order of magnitude off.
    """
    print("\n[6b] Zoobot v2.0.1 default LR groups at n_blocks=3 (protocol match C1/C2)")
    enc = _build(3)
    # Pass NOTHING but the structural args -- this test is about the DEFAULTS.
    cfg = TrainConfig(head_type="linear", n_blocks=3, device="cpu", seed=42)
    check(abs(cfg.lr - 1e-4) < 1e-15, f"TrainConfig.lr default == 1e-4, got {cfg.lr:g}")
    check(abs(cfg.weight_decay - 0.05) < 1e-15,
          f"TrainConfig.weight_decay default == 0.05, got {cfg.weight_decay:g}")
    check(abs(cfg.lr_decay - 0.75) < 1e-15,
          f"TrainConfig.lr_decay default == 0.75, got {cfg.lr_decay:g}")
    check(cfg.backbone_lr is None,
          "TrainConfig.backbone_lr default is the None sentinel (resolves to head lr)")

    tr = Trainer(cfg, encoder=enc)
    groups = tr.opt.param_groups
    check(len(groups) == 4, f"4 param groups, got {len(groups)}")

    want = [("head", 1e-4), ("stages.3", 1e-4), ("stages.2", 7.5e-5),
            ("stages.1", 5.625e-5)]
    for g, (name, lr) in zip(groups, want):
        check(g.get("name") == name, f"group name {g.get('name')!r} == {name!r}")
        check(abs(g["lr"] - lr) < 1e-15,
              f"group {name}: lr {g['lr']:.6e} == {lr:.6e} (Zoobot default)")

    # head and TOP BLOCK must be EQUAL -- the point of the protocol match.
    check(groups[0]["lr"] == groups[1]["lr"],
          f"head lr == stages.3 lr ({groups[0]['lr']:.6e}) -- Zoobot's lr*lr_decay**0")

    # weight_decay 0.05 on EVERY group.
    for g in groups:
        check(abs(g["weight_decay"] - 0.05) < 1e-15,
              f"group {g.get('name')}: weight_decay {g['weight_decay']:g} == 0.05")

    # cosine_schedule stays OFF: a plain AdamW with no LR scheduler attached.
    check(isinstance(tr.opt, torch.optim.AdamW), f"optimizer is AdamW, got {type(tr.opt)}")
    check(getattr(tr, "lr_scheduler", None) is None and getattr(tr, "sched", None) is None,
          "no LR scheduler is attached (cosine_schedule=False)")

    # ...and the flags remain OVERRIDABLE (the brief requires this explicitly).
    ov = Trainer(TrainConfig(head_type="linear", n_blocks=3, lr=3e-3, backbone_lr=7e-4,
                             weight_decay=0.123, lr_decay=0.5, device="cpu", seed=42),
                 encoder=_build(3)).opt.param_groups
    check(abs(ov[0]["lr"] - 3e-3) < 1e-15, "explicit --lr overrides the Zoobot default")
    check(abs(ov[1]["lr"] - 7e-4) < 1e-15,
          "explicit --backbone-lr overrides the head-lr sentinel")
    check(abs(ov[2]["lr"] - 3.5e-4) < 1e-15, "explicit --lr-decay is applied (7e-4*0.5)")
    check(all(abs(g["weight_decay"] - 0.123) < 1e-15 for g in ov),
          "explicit --weight-decay overrides 0.05 on every group")


def test_protocol_provenance():
    """run_config.json must record WHERE these values came from (C2 deliverable 2)."""
    print("\n[6c] protocol-match provenance recorded in run_config.json")
    from train_skeleton import zoobot_protocol_provenance
    p = zoobot_protocol_provenance()
    check(p["protocol_match"] == "zoobot_v2.0.1_defaults",
          f"protocol_match == 'zoobot_v2.0.1_defaults', got {p['protocol_match']!r}")
    check(p["protocol_match_tuned"] is False, "records that the values were NOT tuned")
    check("without tuning" in p["protocol_match_rationale"],
          "the rationale string is recorded verbatim")
    v = p["protocol_match_values"]
    check(v["learning_rate_head"] == v["learning_rate_top_block"] == 1e-4,
          f"head and top-block LR both 1e-4, got {v}")
    check(v["weight_decay"] == 0.05 and v["lr_decay"] == 0.75, f"wd/lr_decay: {v}")
    check(v["cosine_schedule"] is False, "cosine_schedule recorded as off")

    # the screen runner must actually EMBED it, not just be able to build it
    import run_screen_unfreeze as rs
    check(abs(rs.LR - 1e-4) < 1e-15, f"run_screen_unfreeze.LR == 1e-4, got {rs.LR:g}")
    check(abs(rs.WEIGHT_DECAY - 0.05) < 1e-15,
          f"run_screen_unfreeze.WEIGHT_DECAY == 0.05, got {rs.WEIGHT_DECAY:g}")
    check(abs(rs.LR_DECAY - 0.75) < 1e-15, "run_screen_unfreeze.LR_DECAY == 0.75")


# ---------------------------------------------------------------------------
# 7. adversarial: n_blocks=6 must raise
# ---------------------------------------------------------------------------

def test_out_of_range_raises():
    print("\n[7] adversarial: n_blocks=6 raises (does NOT silently clamp)")
    try:
        enc = _build(6)
    except AssertionError as e:
        msg = str(e)
        check("6" in msg and "5" in msg,
              f"AssertionError names both numbers: {msg[:110]}...")
        return
    except Exception as e:  # wrong exception type is still a failure
        check(False, f"expected AssertionError, got {type(e).__name__}: {e}")
        return
    check(False, f"n_blocks=6 did NOT raise (silently clamped to "
                 f"{enc.n_blocks}, {enc.num_trainable_params():,} trainable)")


# ---------------------------------------------------------------------------
# 8. gradient checkpointing is NUMERICALLY EXACT (and a no-op at n_blocks=0)
# ---------------------------------------------------------------------------

def _loss_and_grads(n_blocks, grad_checkpointing, seed=99):
    """Run one identical fwd+bwd and return (loss, {param_name: grad}) on CPU."""
    torch.manual_seed(seed)
    enc = ZoobotEncoder(n_blocks=n_blocks, device="cpu", verbose=False, self_check=False,
                        grad_checkpointing=grad_checkpointing)
    torch.manual_seed(seed)                       # identical head init + identical batch
    head = torch.nn.Linear(EMBED_DIM, 1)
    x = torch.rand(4, IN_CHANS, INPUT_SIZE, INPUT_SIZE) * 0.881
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    loss = torch.nn.functional.binary_cross_entropy_with_logits(head(enc(x)), y)
    loss.backward()
    grads = {f"stages.3.{n}": p.grad.detach().clone()
             for n, p in enc.model.stages[3].named_parameters() if p.grad is not None}
    return float(loss.detach()), grads, enc


def test_grad_checkpointing_exact():
    print("\n[8] gradient checkpointing: numerically EXACT + no-op at n_blocks=0")

    # -- the flag must be REFUSED (not silently set) on the frozen encoder --
    enc0 = ZoobotEncoder(n_blocks=0, device="cpu", verbose=False, self_check=False,
                         grad_checkpointing=True)
    check(enc0.grad_checkpointing is False,
          "n_blocks=0: --grad-checkpointing is a no-op (flag stays False)")
    torch.manual_seed(1234)
    xf = torch.rand(4, IN_CHANS, INPUT_SIZE, INPUT_SIZE) * 0.881
    check(torch.equal(enc0(xf), _reference_forward(xf)),
          "n_blocks=0 + checkpointing flag: forward STILL byte-identical to pre-Lever-1")

    # -- exactness at n_blocks=3 (the rung that needs it) --
    l_off, g_off, e_off = _loss_and_grads(3, False)
    l_on, g_on, e_on = _loss_and_grads(3, True)
    check(e_on.grad_checkpointing is True and e_off.grad_checkpointing is False,
          "n_blocks=3: the flag is actually applied on/off as requested")
    check(l_off == l_on,
          f"loss bit-for-bit identical: off={l_off!r} on={l_on!r}")
    check(set(g_off) == set(g_on) and len(g_off) > 0,
          f"same {len(g_off)} stages[3] grad tensors compared")
    worst, worst_k = 0.0, None
    for k in g_off:
        d = float((g_off[k] - g_on[k]).abs().max())
        if d > worst:
            worst, worst_k = d, k
    check(worst == 0.0,
          f"stages[3] gradients bit-for-bit identical (max|diff|={worst:.3e}"
          + (f" at {worst_k}" if worst_k else "") + ")")


# ---------------------------------------------------------------------------
# 9. LP->FT head loading (--init-head-from) -- the highest-risk untested path
# ---------------------------------------------------------------------------

def test_lp_ft_head_load(tmpdir=None):
    print("\n[9] LP->FT --init-head-from: exact load, loud failure, survives optimizer")
    import copy
    import tempfile
    from heads import build_head
    from train_skeleton import load_head_weights

    tmpdir = tmpdir or tempfile.mkdtemp(prefix="lpft_")
    ckpt_path = os.path.join(tmpdir, "probe_head.pt")

    # -- a head with known NON-TRIVIAL weights (not zeros, not a fresh init) --
    torch.manual_seed(11)
    trained = build_head("linear")
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(torch.randn_like(p) * 0.37 + 0.11)
    torch.save({"head_type": "linear", "head": trained.state_dict(),
                "standardizer": {}, "cfg": {"epoch": 100}}, ckpt_path)

    # -- (a) exact load: every parameter matches --
    fresh = build_head("linear")
    load_head_weights(fresh, ckpt_path)
    allmatch = all(torch.equal(fresh.state_dict()[k], trained.state_dict()[k])
                   for k in trained.state_dict())
    check(allmatch, "loaded head matches the saved head EXACTLY, parameter by parameter")

    # -- (b) a perturbed key NAME must raise, not warn / partial-load --
    bad = {k: v.clone() for k, v in trained.state_dict().items()}
    k0 = sorted(bad)[0]
    bad[k0 + "_typo"] = bad.pop(k0)
    p_badkey = os.path.join(tmpdir, "badkey.pt")
    torch.save({"head_type": "linear", "head": bad}, p_badkey)
    try:
        load_head_weights(build_head("linear"), p_badkey)
        check(False, "perturbed key name did NOT raise (silent partial load!)")
    except RuntimeError as e:
        check("key mismatch" in str(e).lower() or "missing" in str(e).lower(),
              "perturbed key NAME raises RuntimeError naming the mismatch")

    # -- (c) a perturbed tensor SHAPE must raise --
    bad2 = {k: v.clone() for k, v in trained.state_dict().items()}
    wk = [k for k, v in bad2.items() if v.dim() == 2][0]
    bad2[wk] = torch.randn(bad2[wk].shape[0], bad2[wk].shape[1] - 1)
    p_badshape = os.path.join(tmpdir, "badshape.pt")
    torch.save({"head_type": "linear", "head": bad2}, p_badshape)
    try:
        load_head_weights(build_head("linear"), p_badshape)
        check(False, "perturbed shape did NOT raise")
    except RuntimeError as e:
        check("shape mismatch" in str(e).lower(),
              "perturbed tensor SHAPE raises RuntimeError naming the shape")

    # -- (d) an all-zeros head must be refused --
    zeros = {k: torch.zeros_like(v) for k, v in trained.state_dict().items()}
    p_zero = os.path.join(tmpdir, "zeros.pt")
    torch.save({"head_type": "linear", "head": zeros}, p_zero)
    try:
        load_head_weights(build_head("linear"), p_zero)
        check(False, "all-zeros head did NOT raise (would silently be a random-head FT)")
    except RuntimeError as e:
        check("all-zeros" in str(e).lower() or "uninitialised" in str(e).lower(),
              "all-zeros head is REFUSED (LP->FT cannot degrade to random-head FT)")

    # -- (e) the loaded head must DIFFER from a fresh init under the same seed --
    torch.manual_seed(11)
    fresh_same_seed = build_head("linear")
    differs = any(not torch.equal(fresh.state_dict()[k], fresh_same_seed.state_dict()[k])
                  for k in fresh.state_dict())
    check(differs,
          "loaded head differs from a fresh init under the SAME seed (a real load happened)")

    # -- (f) the loaded values must SURVIVE Trainer/optimizer construction --
    # This is the failure that would silently turn nb1/nb3 into random-head fine-tuning.
    enc0 = _build(0)
    cfg = TrainConfig(head_type="linear", n_blocks=0, lr=1e-3, device="cpu",
                      init_head_from=ckpt_path)
    tr = Trainer(cfg, encoder=enc0)
    survived = all(torch.equal(tr.head.state_dict()[k].cpu(), trained.state_dict()[k])
                   for k in trained.state_dict())
    check(survived,
          "loaded head weights are STILL the loaded values after Trainer + optimizer build")
    l2 = float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in tr.head.parameters())))
    l2_ref = float(torch.sqrt(sum((p.detach().float() ** 2).sum()
                                  for p in trained.parameters())))
    check(abs(l2 - l2_ref) < 1e-6,
          f"logged ||head||_2 matches the source checkpoint ({l2:.6f} vs {l2_ref:.6f})")


def run():
    print("=" * 74)
    print("LEVER 1 -- n_blocks partial-unfreeze tests (REAL Zoobot checkpoint)")
    print("=" * 74)
    test_param_count_table()
    test_block_identity()
    test_head_norm_always_frozen()
    test_frozen_invariance()
    test_gradient_flow()
    test_lr_assignment()
    test_zoobot_default_lr_groups()
    test_protocol_provenance()
    test_out_of_range_raises()
    test_grad_checkpointing_exact()
    test_lp_ft_head_load()

    print("\n" + "=" * 74)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s)")
        for m in _fails:
            print("   - " + m)
        return 1
    print("ALL LEVER-1 ENCODER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
